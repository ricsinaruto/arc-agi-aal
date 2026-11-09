"""Solver and admission tests for generated games (paper-notes 2.7, "Admission").

A game is admitted only if:
- every level is solvable, found by breadth-first search over the rule state, and the optimal
  length (the level's baseline action count) is at least ``min_actions``;
- the solution replayed from a full reset through the real ``arc_agi`` offline arcade clears each
  level exactly at the end of its path, and wins at the end;
- the reset state is not already won, and every frame is a 64x64 grid of colours 0-15;
- two replays give identical frames (determinism);
- a random policy with ``random_budget`` x baseline actions clears each level in at most
  ``random_max_rate`` of ``random_rollouts`` tries (the level is not solvable by flailing).
"""

import collections
import hashlib
import os
import random
import tempfile

import numpy as np

KEY_ACTIONS = (1, 2, 3, 4, 5, 7)


def level_actions(game):
    """Every key action, plus a click on every displayed cell. A game may narrow the clicks with
    ``solver_clicks()``: one cell per distinct click effect on the current level (clicks elsewhere
    change nothing), which only shortens the search (the random policy still clicks anywhere)."""
    keys = [a for a in game.ACTIONS if a in KEY_ACTIONS]
    clicks = []
    if 6 in game.ACTIONS:
        cells = game.solver_clicks() if hasattr(game, "solver_clicks") else None
        if cells is None:
            rows, cols = game.render_grid().shape  # the displayed board (a skin may turn it)
            cells = [(x, y) for y in range(rows) for x in range(cols)]
        clicks = [(6, tuple(cell)) for cell in cells]
    return [(a, None) for a in keys] + clicks


def solve_level(cls, index, *, max_states=300_000):
    """Shortest action list [(action, cell)] that clears level ``index``, or None."""
    game = cls()
    game.set_level(index)
    start = game.snapshot()
    seen = {game.state_key()}
    queue = collections.deque([(start, [])])
    while queue:
        snap, path = queue.popleft()
        for action in level_actions(game):
            game.restore(snap)
            result = game.apply(*action)
            if result == "win":
                return path + [action]
            if result == "lose":
                continue
            key = game.state_key()
            if key in seen:
                continue
            seen.add(key)
            if len(seen) > max_states:
                raise RuntimeError(f"level {index + 1}: search exceeded {max_states} states")
            queue.append((game.snapshot(), path + [action]))
    return None


def display_xy(width, height, cell):
    """Centre of a grid cell in 64x64 display coordinates (inverse of Camera.display_to_grid)."""
    scale = min(64 // width, 64 // height)
    pad_x, pad_y = (64 - width * scale) // 2, (64 - height * scale) // 2
    return pad_x + cell[0] * scale + scale // 2, pad_y + cell[1] * scale + scale // 2


def engine_actions(cls, index, path):
    """[(action name, data)] for the arcade from a solver path on level ``index``."""
    rows = cls.LEVELS[index]["map"]
    width, height = len(rows[0]), len(rows)
    out = []
    for action, cell in path:
        if action == 6:
            x, y = display_xy(width, height, cell)
            out.append(("ACTION6", {"x": x, "y": y}))
        else:
            out.append((f"ACTION{action}", {}))
    return out


def random_clear_rate(cls, index, budget, rollouts, seed=0):
    """Share of ``rollouts`` random plays of ``budget`` actions that clear level ``index``. Each step
    picks an available action uniformly and, for a click, a uniformly random displayed cell (for a
    keys-only or click-only game this is a uniform choice over ``level_actions`` without the
    ``solver_clicks`` narrowing)."""
    rng = random.Random(seed)
    game = cls()
    game.set_level(index)
    start = game.snapshot()
    keys = [a for a in game.ACTIONS if a in KEY_ACTIONS]
    rows, cols = game.render_grid().shape
    cells = [(x, y) for y in range(rows) for x in range(cols)]
    kinds = keys + ([6] if 6 in game.ACTIONS else [])

    def pick():
        action = kinds[0] if len(kinds) == 1 else rng.choice(kinds)
        return (6, rng.choice(cells)) if action == 6 else (action, None)

    clears = 0
    for _ in range(rollouts):
        game.restore(start)
        for _ in range(budget):
            result = game.apply(*pick())
            if result == "win":
                clears += 1
                break
            if result == "lose":
                break
    return clears / rollouts


def replay(environments_dir, game_id, plan):
    """Play ``plan`` (per level lists of (name, data)) from reset; returns (frame hashes, level marks, state)."""
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    os.environ["ONLY_RESET_LEVELS"] = "true"
    with tempfile.TemporaryDirectory() as recordings:
        arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(environments_dir),
                        recordings_dir=recordings)
        env = arcade.make(game_id)
        if env is None:
            raise RuntimeError(f"arcade could not load {game_id}")
        obs = env.observation_space or env.reset()
        frames = [np.array(obs.frame[-1])]
        marks = []
        start_levels, start_state = int(obs.levels_completed), getattr(obs.state, "name", str(obs.state))
        for level, actions in enumerate(plan):
            for position, (name, data) in enumerate(actions):
                obs = env.step(GameAction[name], data)
                frames.extend(np.array(f) for f in obs.frame)
                marks.append((level, position, int(obs.levels_completed)))
        state = getattr(obs.state, "name", str(obs.state))
    digest = [hashlib.sha256(f.astype(np.int16).tobytes()).hexdigest()[:16] for f in frames]
    return {"frames": frames, "hashes": digest, "marks": marks, "state": state,
            "start": (start_levels, start_state)}


def check_level(cls, spec, *, min_actions=4, random_budget=3, random_rollouts=200, random_max_rate=0.05,
                max_states=300_000):
    """Per-level admission tests on one candidate level spec; returns (ok, baseline, random rate, reason)."""
    single = type(cls.__name__, (cls,), {"LEVELS": [spec]})
    try:
        path = solve_level(single, 0, max_states=max_states)
    except RuntimeError:
        return False, None, None, "search too large"
    if path is None:
        return False, None, None, "unsolvable"
    if len(path) < min_actions:
        return False, len(path), None, "too short"
    rate = random_clear_rate(single, 0, random_budget * len(path), random_rollouts)
    if rate > random_max_rate:
        return False, len(path), rate, "random clears"
    return True, len(path), rate, "ok"


def admit(cls, environments_dir, game_id, *, min_actions=4, random_budget=3, random_rollouts=200,
          random_max_rate=0.05, max_states=300_000):
    """Run every admission test; returns a report with ``admitted`` and per-level details."""
    report = {"game_id": game_id, "levels": [], "issues": []}
    plan, baselines = [], []
    for index in range(len(cls.LEVELS)):
        try:
            path = solve_level(cls, index, max_states=max_states)
        except RuntimeError as exc:
            report["issues"].append(str(exc))
            path = None
        if path is None:
            report["issues"].append(f"level {index + 1}: unsolvable")
            report["levels"].append({"level": index + 1, "solved": False})
            continue
        rate = random_clear_rate(cls, index, random_budget * len(path), random_rollouts)
        row = {"level": index + 1, "solved": True, "baseline": len(path), "random_clear_rate": rate,
               "path": [[a, list(c) if c else None] for a, c in path]}
        report["levels"].append(row)
        if len(path) < min_actions:
            report["issues"].append(f"level {index + 1}: solved in {len(path)} < {min_actions} actions")
        if rate > random_max_rate:
            report["issues"].append(f"level {index + 1}: random policy clears {rate:.0%}")
        plan.append(engine_actions(cls, index, path))
        baselines.append(len(path))
    report["baseline_actions"] = baselines
    if report["issues"]:
        report["admitted"] = False
        return report
    first = replay(environments_dir, game_id, plan)
    second = replay(environments_dir, game_id, plan)
    if first["start"][0] != 0 or first["start"][1] == "WIN":
        report["issues"].append(f"reset state already scores: {first['start']}")
    if first["hashes"] != second["hashes"]:
        report["issues"].append("replays differ (non-deterministic)")
    for frame in first["frames"]:
        if frame.shape != (64, 64) or frame.min() < 0 or frame.max() > 15:
            report["issues"].append(f"bad frame shape/colours {frame.shape} {frame.min()}..{frame.max()}")
            break
    for level, actions in enumerate(plan):
        levels_after = [m[2] for m in first["marks"] if m[0] == level]
        if levels_after[-1] != level + 1 or any(v != level for v in levels_after[:-1]):
            report["issues"].append(f"level {level + 1}: arcade replay did not clear exactly at the path end "
                                    f"({levels_after})")
    if first["state"] != "WIN":
        report["issues"].append(f"final state {first['state']}, expected WIN")
    report["replay_frames"] = len(first["frames"])
    report["admitted"] = not report["issues"]
    return report
