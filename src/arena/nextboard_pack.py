"""Next-board aux data (envs arm): one level attempt = one packed conversation, the next board an
assistant reply at every step (board predictions are scored and trained only at assistant
positions; user/tool positions are degenerate on every stack).

Turn 1 (user): the task, the level's first board as ``current_frame.ascii`` text and the first action. Each
assistant turn: the patch that action made (``aux_tasks.patch_text``: ``@ row=R col=C`` rectangles or NO CHANGE).
Each later user turn: the next action, and every ``board_every`` steps the true current board again (a long attempt
cannot be integrated from patches alone). No system prompt, no images: ~200-500 input tokens per target against
~8K for a one-transition example with the play prompt.

Attempts come from a run's own tries (``log.jsonl`` observations: settled board = the last frame); an attempt ends
at a level change, GAME_OVER, RESET or the end of the log; the clearing transition is never in it (its next board is
the next level's)."""

import hashlib
import json
import random
from pathlib import Path

NAMES = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE",
         "ACTION6": "MOUSE", "ACTION7": "UNDO", "RESET": "RESET"}
INTRO = ("Next-board practice. The board is a 64x64 grid of colour letters (`current_frame.ascii`: row 0 at the top, "
         "column 0 at the left). Each of my turns names one action taken on it. Reply with the patch that action makes "
         "to the board: one or more rectangles covering every cell that changes, each a line `@ row=R col=C` (its "
         "top-left cell) followed by its rows of letters; or the single line NO CHANGE.")
HELD_OUT = ("tr87", "r11l", "m0r0", "dc22", "s5i5", "su15", "sp80", "g50t", "lf52", "bp35")  # 15/10 split, 20 Oct


def grid(board):
    """A board as 64 hex rows (or one 4096-char string) -> uint8 [64, 64]."""
    import numpy as np

    text = "".join(board) if not isinstance(board, str) else board
    raw = np.frombuffer(text.encode(), dtype=np.uint8)
    return np.where(raw <= 57, raw - 48, raw - 87).astype(np.uint8).reshape(64, 64)


def label(action):
    name = str(action.get("action"))
    if name == "ACTION6":
        return f"MOUSE(row={action.get('y')}, col={action.get('x')})"
    return NAMES.get(name, name)


def level_attempts(log_path, min_steps=2):
    """Level attempts of one try's ``log.jsonl``: ``{"level", "boards": [hex], "actions", "end"}``; boards[0] is the
    level's first board seen, boards[i + 1] the settled board after actions[i]."""
    out, cur, prev = [], None, None

    def close(end):
        nonlocal cur
        if cur and len(cur["actions"]) >= min_steps:
            cur["end"] = end
            out.append(cur)
        cur = None

    with open(log_path) as file:
        for line in file:
            if not line.startswith('{"kind":"observation"'):
                continue
            row = json.loads(line)
            frames = row.get("frames") or []
            if not frames or len(frames[-1]) != 64 or any(len(r) != 64 for r in frames[-1]):
                close("bad_frame")
                prev = None
                continue
            board = "".join(frames[-1])
            action = row.get("action")
            name = (action or {}).get("action") if isinstance(action, dict) else None
            state = row.get("state")
            same = (prev is not None and name is not None and name != "RESET" and not row.get("full_reset")
                    and row.get("levels") == prev["levels"])
            if same and state == "NOT_FINISHED":
                if cur is None:
                    cur = {"level": int(prev["levels"] or 0), "boards": [prev["board"]], "actions": []}
                cur["boards"].append(board)
                cur["actions"].append({k: action[k] for k in ("action", "x", "y") if k in action})
            else:
                close("level_change" if prev and row.get("levels") != prev["levels"]
                      else "reset" if name == "RESET" or row.get("full_reset") else str(state or "reset").lower())
            prev = None if state == "WIN" else {"board": board, "levels": row.get("levels")}
    close("eof")
    return out


COLOR_CHARS = "WwgGcBMPRbSYOrNp"  # inference.utils.grid_utils.ARC_COLOR_CHARS


def patch_rects(before, after, gap=1):
    """``posttrain.aux_tasks.patch_rects`` without scipy (loop bundles carry neither ``posttrain`` nor scipy): changed
    cells grown by ``gap`` cross-shaped steps, 8-connected components labelled in raster order, each component's
    bounding box of changed cells, overlapping boxes merged; the same boxes in the same order (tested against it)."""
    import numpy as np

    before, after = np.asarray(before), np.asarray(after)
    mask = before != after
    if not mask.any():
        return []
    grown = mask.copy()
    for _ in range(gap):
        g = grown.copy()
        g[1:] |= grown[:-1]
        g[:-1] |= grown[1:]
        g[:, 1:] |= grown[:, :-1]
        g[:, :-1] |= grown[:, 1:]
        grown = g
    h, w = grown.shape
    label = np.zeros((h, w), dtype=np.int32)
    boxes, n = [], 0
    for r in range(h):
        for c in range(w):
            if not grown[r, c] or label[r, c]:
                continue
            n += 1
            label[r, c] = n
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and grown[yy, xx] and not label[yy, xx]:
                            label[yy, xx] = n
                            stack.append((yy, xx))
            hit = [(y, x) for y, x in cells if mask[y, x]]
            if hit:
                ys, xs = [y for y, _ in hit], [x for _, x in hit]
                boxes.append([min(ys), min(xs), max(ys) + 1, max(xs) + 1])
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return [(int(r0), int(c0), after[r0:r1, c0:c1]) for r0, c0, r1, c1 in sorted(boxes)]


def patch_text(before, after):
    rects = patch_rects(before, after)
    if not rects:
        return "NO CHANGE"
    parts = []
    for row, col, block in rects:
        parts.append(f"@ row={row} col={col}")
        parts.extend("".join(COLOR_CHARS[int(v)] for v in line) for line in block)
    return "\n".join(parts)


def ascii_board(board):
    return "\n".join("".join(COLOR_CHARS[max(0, min(15, int(v)))] for v in row) for row in board)


def render(att, board_every=4, max_steps=48, max_patch_chars=3000):
    """Packed conversations of one attempt (split every ``max_steps`` steps; each part opens with its full board):
    ``[{"turns": [{"user", "target", "train", "changed"}], "steps", "level", "first"}]``. A patch over
    ``max_patch_chars`` stays in the context untrained (``train`` False)."""
    boards = [grid(b) for b in att["boards"]]
    out = []
    for start in range(0, len(att["actions"]), max_steps):
        turns = []
        for i in range(start, min(start + max_steps, len(att["actions"]))):
            k = i - start
            act = f"Action: {label(att['actions'][i])}"
            if k == 0:
                user = f"{INTRO}\n\nBoard:\n```text\n{ascii_board(boards[i])}\n```\n{act}"
            elif board_every and k % board_every == 0:
                user = f"Board now:\n```text\n{ascii_board(boards[i])}\n```\n{act}"
            else:
                user = act
            patch = patch_text(boards[i], boards[i + 1])
            turns.append({"user": user, "target": patch, "train": len(patch) <= max_patch_chars,
                          "changed": int((boards[i] != boards[i + 1]).sum())})
        out.append({"turns": turns, "steps": len(turns), "level": att["level"], "first": start})
    return out


def transition_keys(att):
    return {hashlib.md5((b + json.dumps(a, sort_keys=True)).encode()).hexdigest()[:16]
            for b, a in zip(att["boards"], att["actions"])}


def build(tries, *, k=16, seed=0, board_every=4, max_steps=48, min_steps=4, held_out=HELD_OUT):
    """``tries``: ``[(game, attempt dir)]``. Picks up to ``k`` level attempts, greedily by new (board, action)
    transitions with a change, one game at a time round-robin (breadth); never a held-out game. Returns the packed
    rows (``game``, ``source``, ``turns``...) and a summary."""
    rng = random.Random(seed)
    by_game = {}
    for game, folder in tries:
        if str(game)[:4] in held_out:
            continue
        log = Path(folder) / "log.jsonl"
        if not log.exists():
            continue
        for a in level_attempts(log, min_steps=min_steps):
            a["game"], a["source"] = str(game), str(folder)
            by_game.setdefault(str(game), []).append(a)
    for atts in by_game.values():
        rng.shuffle(atts)
    seen, picked, games = set(), [], sorted(by_game)
    rng.shuffle(games)
    while len(picked) < k and any(by_game.values()):
        for game in games:
            if len(picked) >= k or not by_game[game]:
                continue
            best = max(by_game[game], key=lambda a: len(transition_keys(a) - seen))
            by_game[game].remove(best)
            if not transition_keys(best) - seen:
                continue
            seen |= transition_keys(best)
            picked.append(best)
    rows = []
    for a in picked:
        for part in render(a, board_every=board_every, max_steps=max_steps):
            rows.append({"game": a["game"], "source": a["source"], **part})
    steps = sum(r["steps"] for r in rows)
    unchanged = sum(1 for r in rows for t in r["turns"] if not t["changed"])
    summary = {"attempts": len(picked), "parts": len(rows), "steps": steps, "games": len({a["game"] for a in picked}),
               "copy_em": round(unchanged / steps, 4) if steps else None,
               "untrained": sum(1 for r in rows for t in r["turns"] if not t["train"])}
    return rows, summary


def write(rows, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
    tmp.rename(dest)
    return dest
