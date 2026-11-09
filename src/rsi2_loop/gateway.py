"""The fixed side's game gateway for one evaluation (docs/rsi.md: the game engine and scorer are out of the proposer's
reach). A root process outside every sandbox serves the pool through the SDK's competition-mode REST API
(``taaf.competition_arcade``, the server the arena's public rehearsal starts in-process); the candidate's arena
connects to it over loopback (``RSI2_GATEWAY_URL``, read by the fixed ``arena/competition.py``) and never sees a game
file. Every ``--every`` seconds the open scorecards are written to ``--snapshots`` (root-only): per exposed game the
score, levels completed and level scores, from the gateway's own scorecard manager.

    python -m rsi2_loop.gateway --games-dir D --games g1,g2,... --port P --snapshots F [--every 20]
        [--log-actions F] [--prefixes F]

``--log-actions``: every step of every instance (action, data, levels after it; the board hash at the instance's
start and at each level-up), for the frontier selection (rsi2_loop.frontier). ``--prefixes``: {exposed id: {source,
level, steps, hash}}: the steps are replayed into that instance when it is created, before the agent's first frame;
an instance whose replayed board hash or level count differs is listed as void in every snapshot (its attempt does not
count).
"""
import argparse
import json
import os
import signal
import threading
import time


def snapshot(server, games):
    from arc_agi.scorecard import EnvironmentScorecard

    arcade = server._arcade
    for _ in range(5):  # request threads may add a card while we iterate
        try:
            cards = list(arcade.scorecard_manager.scorecards.items())
            break
        except RuntimeError:
            time.sleep(0.05)
    else:
        return None
    exposed = [env.game_id for env in arcade.available_environments]
    out = {}
    for card_id, card in cards:
        try:
            env_card = EnvironmentScorecard.from_scorecard(card, arcade.available_environments)
        except Exception:  # noqa: BLE001 - a card being written; the next snapshot has it
            continue
        rows = {}
        for env in env_card.environments:
            run = env.runs[0] if getattr(env, "runs", None) else None
            rows[env.id] = {"score": env.score, "levels": env.levels_completed,
                            "level_count": getattr(env, "level_count", None),
                            "level_scores": list(getattr(run, "level_scores", None) or []),
                            "level_actions": list(getattr(run, "level_actions", None) or []),
                            "actions": getattr(env, "actions", None), "state": getattr(run, "state", None)}
        out[card_id] = {"score": env_card.score, "games": rows}
    return {"t": time.time(), "exposed": exposed,
            "source": {g: games[i % len(games)] for i, g in enumerate(exposed)}, "cards": out,
            "void": sorted(VOID)}


VOID = set()


def instrument(arcade, log_path=None, prefixes=None):
    """Wrap ``arcade.make`` (the REST app calls it per new instance): replay a prefix, log steps."""
    from arcengine import GameAction

    from .frontier import board_hash

    lock = threading.Lock()
    log = open(log_path, "a") if log_path else None
    if log_path:
        os.chmod(log_path, 0o600)
    original = arcade.make

    def write(row):
        if log is not None:
            with lock:
                log.write(json.dumps(row) + "\n")
                log.flush()

    def make(game_id, *args, **kwargs):
        env = original(game_id, *args, **kwargs)
        if env is None:
            return env
        spec = (prefixes or {}).get(game_id)
        if spec is not None:
            resp = env.observation_space
            try:
                for step in spec["steps"]:
                    resp = env.step(action=GameAction.from_id(int(step["a"])), data=step.get("d") or {})
                h, lv = board_hash(resp.frame), int(resp.levels_completed)
            except Exception as exc:  # noqa: BLE001
                h, lv = f"error {type(exc).__name__}", -1
            ok = h == spec["hash"] and lv == int(spec["level"]) - 1
            if not ok:
                VOID.add(game_id)
            write({"g": game_id, "k": "replay", "ok": ok, "h": h, "lv": lv, "want": spec["hash"]})
        start = env.observation_space
        write({"g": game_id, "k": "start", "h": board_hash(start.frame) if start is not None else None,
               "lv": int(start.levels_completed) if start is not None else None})
        step = env.step
        state = {"lv": int(start.levels_completed) if start is not None else 0}

        def logged(action, data=None, reasoning=None):
            resp = step(action=action, data=data, reasoning=reasoning)
            if resp is not None:
                lv = int(resp.levels_completed)
                up = lv > state["lv"]
                state["lv"] = lv
                write({"g": game_id, "k": "step", "a": int(action.value), "d": data or {}, "lv": lv, "up": up,
                       **({"h": board_hash(resp.frame)} if up else {})})
            return resp

        env.step = logged
        return env

    arcade.make = make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--games", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--every", type=float, default=20.0)
    ap.add_argument("--log-actions")
    ap.add_argument("--prefixes")
    a = ap.parse_args()
    os.environ["ONLY_RESET_LEVELS"] = "true"  # as open_arcade sets it for the public rehearsal
    from taaf.competition_arcade import CompetitionArcadeServer

    games = a.games.split(",")
    server = CompetitionArcadeServer(game_ids=games, total_runs=len(games), environments_dir=a.games_dir,
                                     save_all_recordings=False, port=a.port).start()
    if a.log_actions or a.prefixes:
        instrument(server._arcade, a.log_actions,
                   json.loads(open(a.prefixes).read()) if a.prefixes else None)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    print(json.dumps({"event": "rsi2_gateway_ready", "url": server.base_url, "games": len(games)}), flush=True)
    with open(a.snapshots, "a") as out:
        os.chmod(a.snapshots, 0o600)
        while True:
            snap = snapshot(server, games)
            if snap is not None:
                out.write(json.dumps(snap) + "\n")
                out.flush()
            if stop.wait(a.every):
                break
        snap = snapshot(server, games)
        if snap is not None:
            out.write(json.dumps({**snap, "final": True}) + "\n")
    server.stop()


if __name__ == "__main__":
    main()
