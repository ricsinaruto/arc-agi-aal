"""Local competition gateway for the dfranzen-exact Modal run.

Stands in for Kaggle's hidden-rerun gateway (``http://gateway:8001``): the arc_agi REST app in competition mode on
localhost, serving 110 cyclic clones ``k000``..``k109`` of the 20-game public roster (the 25 public games minus ft09,
lp85, ar25, vc33, tu93), in the same order as our dfr-combined 110-clone previews (``taaf.competition_arcade``).
Self-contained so it runs in its own venv with the competition's arc_agi wheels; the notebook's interpreter is
untouched.

Monitoring only (never read by the notebook): every ``--interval`` seconds the open scorecards are read in-process and
appended to ``<out>/scorecards.jsonl`` (score = mean over the 110 instances of the per-game official score, levels,
actions, per-instance runs); the closed scorecard is written to ``<out>/scorecard-final.json``.
"""

import argparse
import json
import logging
import socketserver
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

import arc_agi
from arc_agi import server as arc_agi_server
from arc_agi.scorecard import EnvironmentScorecard

API_KEY = "test-key-123"


class _ThreadingWSGIServer(socketserver.ThreadingMixIn, WSGIServer):
    daemon_threads = True


class _Quiet(WSGIRequestHandler):
    def log_message(self, format, *args):
        return


def summary(card: EnvironmentScorecard, total: int) -> dict:
    envs = []
    for env in card.environments:
        best = max(env.runs, key=lambda r: r.score) if env.runs else None
        envs.append({
            "id": env.id, "score": float(env.score), "levels": int(env.levels_completed),
            "actions": int(env.actions), "resets": int(env.resets), "completed": bool(env.completed),
            "level_actions": best.level_actions if best else None,
            "level_baseline_actions": best.level_baseline_actions if best else None,
            "level_scores": best.level_scores if best else None,
            "number_of_levels": best.number_of_levels if best else None,
            "state": str(best.state) if best and best.state is not None else None,
        })
    return {
        "card_id": card.card_id,
        "score_mean_110": sum(e["score"] for e in envs) / total,
        "levels": sum(e["levels"] for e in envs),
        "actions": sum(e["actions"] for e in envs),
        "games_played": sum(1 for e in envs if e["actions"] > 0),
        "games_won": sum(1 for e in envs if e["completed"]),
        "environments": envs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environments-dir", required=True)
    parser.add_argument("--games", required=True, help="JSON list of the source game ids, roster order")
    parser.add_argument("--total", type=int, default=110)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    games = json.loads(args.games)

    logger = logging.getLogger("dfr-exact.gateway")
    logger.setLevel(logging.WARNING)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                            environments_dir=args.environments_dir, logger=logger)
    by_id = {env.game_id: env for env in arcade.available_environments}
    clones = []
    for i in range(args.total):
        source = by_id[games[i % len(games)]]
        tags = list(source.private_tags or []) + [f"taaf_source_game:{source.game_id}"]
        clones.append(source.model_copy(deep=True, update={"game_id": f"k{i:03d}", "private_tags": tags}))
    arcade.available_environments = clones
    (out / "roster.json").write_text(json.dumps(
        [{"game_id": c.game_id, "source_game_id": games[i % len(games)]} for i, c in enumerate(clones)], indent=1))

    lock = threading.Lock()

    def on_close(card):
        with lock:
            data = card.model_dump(mode="json", exclude={"api_key"})
            data["_summary"] = summary(card, args.total)
            data["_closed_epoch"] = time.time()
            (out / "scorecard-final.json").write_text(json.dumps(data, indent=1))
        print(json.dumps({"event": "scorecard_closed", "score_mean_110": data["_summary"]["score_mean_110"],
                          "levels": data["_summary"]["levels"]}), flush=True)

    app, _api = arc_agi_server.create_app(arcade, competition_mode=True, save_all_recordings=False,
                                          include_frame_data=True, on_scorecard_close=on_close)

    def anon_key():
        return json.dumps({"api_key": API_KEY}), 200, {"Content-Type": "application/json"}

    app.add_url_rule("/api/games/anonkey", methods=["GET"], view_func=anon_key, endpoint="anon_key")
    server = make_server("127.0.0.1", args.port, app, server_class=_ThreadingWSGIServer, handler_class=_Quiet)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(json.dumps({"event": "gateway_ready", "port": args.port, "instances": len(clones),
                      "arc_agi": getattr(arc_agi, "__version__", None)}), flush=True)

    with (out / "scorecards.jsonl").open("a") as log:
        while True:
            time.sleep(args.interval)
            cards = []
            for _ in range(3):  # request threads may add or close a card while we iterate
                try:
                    cards = list(arcade.scorecard_manager.scorecards.items())
                    break
                except RuntimeError:
                    time.sleep(0.05)
            for card_id, scorecard in cards:
                try:
                    card = EnvironmentScorecard.from_scorecard(scorecard, arcade.available_environments)
                    row = {"epoch": time.time(), "utc": datetime.now(timezone.utc).isoformat(),
                           "open_at": scorecard.open_at.isoformat() if getattr(scorecard, "open_at", None) else None,
                           **summary(card, args.total)}
                except Exception as exc:  # monitoring must never take the gateway down
                    row = {"epoch": time.time(), "card_id": card_id, "error": f"{type(exc).__name__}: {exc}"}
                log.write(json.dumps(row) + "\n")
                log.flush()


if __name__ == "__main__":
    main()
