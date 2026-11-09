"""One game's vet, run in a sandboxed subprocess (game code is untrusted): ``python -I vet.py <env_dir> <game_id>
<trace.json>`` prints one JSON line. Standalone (stdlib, arc_agi, arcengine, numpy): the sandbox gives it nothing else.

The reference trace (``[[action text or "RESET", ...], ...]`` per level, ``ACTION6@x,y`` for clicks) is replayed twice
on fresh offline environments; each level must clear on its trace's last action, and the two replays must give the
same boards. Random play: 3 seeds x 300 uniform random actions; ``random_levels`` is the levels each seed cleared
(``random_level1_wins``: seeds that cleared level 1, ``random_level2_wins``: seeds that cleared two or more)."""

import hashlib
import json
import logging
import os
import random
import sys
import tempfile
from pathlib import Path


def decode_action(text):
    name, _, where = str(text).partition("@")
    if name == "ACTION6":
        x, y = (int(v) for v in where.split(","))
        return name, x, y
    if where or not (name == "RESET" or name in {f"ACTION{i}" for i in range(1, 8)}):
        raise ValueError(f"bad action: {text!r}")
    return name, None, None


def last_board(raw, previous=None):
    import numpy as np

    frames = list(getattr(raw, "frame", None) or [])
    return previous if not frames else np.array(frames[-1], dtype=np.uint8)


def board_hash(board):
    import numpy as np

    return hashlib.blake2s(np.asarray(board, dtype=np.uint8).tobytes(), digest_size=12).hexdigest()


def main():
    env_dir, game_id, trace_path = sys.argv[1:4]
    os.environ["ONLY_RESET_LEVELS"] = "true"
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    logger = logging.getLogger("arc.environment")
    logger.setLevel(logging.WARNING)
    levels = json.loads(Path(trace_path).read_text())
    tmp = Path(tempfile.mkdtemp())

    def fresh():
        arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(Path(env_dir).resolve()),
                     recordings_dir=str(tmp / "rec"), logger=logger)
        env = arc.make(game_id)
        if env is None:
            raise RuntimeError("arcade.make returned None")
        return env

    def step(env, text):
        name, x, y = decode_action(text)
        if name == "RESET":
            return env.reset()
        return env.step(GameAction[name], {"x": x, "y": y} if name == "ACTION6" else {})

    out = {"game_id": game_id, "ok": False}
    runs = []
    for _ in range(2):
        env = fresh()
        raw = env.observation_space
        first = board_hash(last_board(raw))
        board, hashes, per_level, problems = last_board(raw), [], [], []
        for index, actions in enumerate(levels):
            before = int(raw.levels_completed)
            counted = 0
            for text in actions:
                raw = step(env, text)
                board = last_board(raw, board)
                hashes.append(board_hash(board))
                if text != "RESET":
                    counted += 1
            after = int(raw.levels_completed)
            state = getattr(raw.state, "name", str(raw.state))
            per_level.append({"level": index + 1, "actions": counted, "levels_completed": after, "state": state})
            if after != before + 1:
                problems.append(f"level {index + 1}: levels {before}->{after} state {state}")
                break
        runs.append({"first": first, "hashes": hashes, "levels": per_level, "problems": problems,
                     "win_levels": int(getattr(raw, "win_levels", 0) or 0), "final": getattr(raw.state, "name", "")})
    out.update(board_hash=runs[0]["first"], levels=runs[0]["levels"], problems=runs[0]["problems"],
               deterministic=runs[0]["hashes"] == runs[1]["hashes"] and runs[0]["first"] == runs[1]["first"],
               win_levels=runs[0]["win_levels"], final_state=runs[0]["final"])
    reached = []
    for seed in range(3):
        env, rng = fresh(), random.Random(seed)
        raw = env.observation_space
        best = 0
        for _ in range(300):
            names = [a for a in (raw.available_actions or [1, 2, 3, 4, 5, 6, 7])]
            choice = rng.choice(names)
            choice = int(getattr(choice, "value", choice))
            if choice == 0:
                continue
            raw = step(env, f"ACTION6@{rng.randrange(64)},{rng.randrange(64)}" if choice == 6 else f"ACTION{choice}")
            best = max(best, int(raw.levels_completed))
            state = getattr(raw.state, "name", "")
            if state == "WIN":
                break
            if state == "GAME_OVER":
                raw = env.reset()
        reached.append(best)
    out["random_levels"] = reached
    out["random_level1_wins"] = sum(1 for r in reached if r >= 1)
    out["random_level2_wins"] = sum(1 for r in reached if r >= 2)
    out["ok"] = not out["problems"] and out["deterministic"] and len(out["levels"]) == len(levels)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
