"""Level-bench start points: recorded action prefixes that reach the start of a level (27 September).

A start point is (game, k, prefix): replaying ``prefix`` on a fresh environment leaves the game with
``levels == k`` completed, at the first board of level k + 1 (k = 0: a fresh game, empty prefix). The
set is built offline from recorded Flash Duck-transfer runs (``scripts/arena_analysis/
level_bench_start_points.py``) and each prefix is checked by replaying it on the local environment
files: the level count and the final board must match the recording. The bench replays the prefix
again at run time and refuses an attempt whose board differs (``replay``).

Actions are strings: ``ACTION1``..``ACTION7``, ``RESET`` and ``ACTION6@x,y`` (x = column, y = row).
"""

from arcengine import GameAction

from arc_infra.perception import board_hash

SCORE_CAP = 115.0  # arc_agi.scorecard.EnvironmentScoreCalculator.add_level


def encode_action(action):
    """A logged ``Action.payload()`` dict (``{"action", "x", "y"}``) as a prefix string, or None when it
    cannot be replayed (the explorer's aggregated ``EXPLORER`` step, unknown names)."""
    if not isinstance(action, dict):
        return None
    name = action.get("action")
    if name == "RESET" or name in {f"ACTION{i}" for i in range(1, 8) if i != 6}:
        return name
    if name == "ACTION6" and isinstance(action.get("x"), int) and isinstance(action.get("y"), int):
        return f"ACTION6@{action['x']},{action['y']}"
    return None


def decode_action(text):
    """``(name, x, y)``; x and y are None except for ACTION6."""
    name, _, where = str(text).partition("@")
    if name == "ACTION6":
        x, y = (int(v) for v in where.split(","))
        return name, x, y
    if where or not (name == "RESET" or name in {f"ACTION{i}" for i in range(1, 8)}):
        raise ValueError(f"Bad prefix action: {text!r}")
    return name, None, None


def last_board(raw, previous=None):
    """The last 64x64 frame of a raw observation (the previous board when the step returned none)."""
    import numpy as np

    frames = list(getattr(raw, "frame", None) or [])
    if not frames:
        return previous
    return np.array(frames[-1], dtype=np.uint8)


def replay(env, prefix, trace=None):
    """Replay ``prefix`` on ``env`` (fresh from ``Arcade.make``): ``(raw, board)`` after the last action. ``trace``
    (a list) receives ``(action text or None, raw)`` for the first observation and after every action."""
    raw = env.observation_space
    board = last_board(raw)
    if trace is not None:
        trace.append((None, raw))
    for text in prefix:
        name, x, y = decode_action(text)
        if name == "RESET":
            raw = env.reset()
        else:
            raw = env.step(GameAction[name], {"x": x, "y": y} if name == "ACTION6" else {})
        if raw is None:
            raise RuntimeError(f"Environment returned no observation while replaying {text}")
        board = last_board(raw, board)
        if trace is not None:
            trace.append((text, raw))
    return raw, board


def check_replay(env, point, trace=None):
    """Replay a start point and compare with its recording: ``(ok, detail, raw)``."""
    raw, board = replay(env, point["prefix"], trace)
    levels = int(raw.levels_completed)
    state = getattr(raw.state, "name", str(raw.state))
    digest = board_hash(board) if board is not None else None
    problems = []
    if levels != int(point["level_index"]):
        problems.append(f"levels {levels} != {point['level_index']}")
    if digest != point["board_hash"]:
        problems.append("board differs")
    if state != "NOT_FINISHED":
        problems.append(f"state {state}")
    return not problems, "; ".join(problems) or "ok", raw


def level_score(baseline_actions, actions, cleared):
    """The competition's per-level score: min(115, 100 * (baseline / actions)^2) for a clear, else 0."""
    if not cleared or not actions or not baseline_actions or baseline_actions < 0:
        return 0.0
    return min(SCORE_CAP, 100.0 * (float(baseline_actions) / float(actions)) ** 2)


def point_id(game, level_index):
    """``<game prefix>-L<level played>`` (1-based: ``ar25-L1`` is a fresh game)."""
    return f"{str(game)[:4]}-L{int(level_index) + 1}"
