"""Token cost of one board observation under different encodings (28 September, CPU only).

The Flash arena harness sends each board as a 256 x 256 PNG (64 x 64 cells at MULTIMODAL_UPSCALE 4); the
vision tower merges 32 x 32 px per token, so the image is 8 x 8 = 64 tokens plus two markers. This script
tokenizes the recorded frames of a run's games with the served tokenizer under the text encodings a text-only
harness would use, for comparison:

- ``full``: 64 rows of 64 letter-coded cells (``arena.text_observations`` style);
- ``rle``: every row run-length encoded (``W12 b3 ...``);
- ``diff``: only the cells that changed since the previous frame, as ``r,c:X`` triples (a full board at the
  first frame of each level).

    uv run --with tokenizers python scripts/arena_analysis/board_encoding_tokens.py --trial <trial dir> [--games 20]
"""

import argparse
import json
import statistics
from pathlib import Path

from prefill_attribution import TOKENIZER, Counter_

LETTERS = "WwgGcBMPRbSYOrNp"  # ARC colour ids 0-15 as the harness letters them


def letter_rows(frame):
    return ["".join(LETTERS[int(ch, 16)] for ch in row) for row in frame]


def rle(row):
    out, i = [], 0
    while i < len(row):
        j = i
        while j < len(row) and row[j] == row[i]:
            j += 1
        out.append(f"{row[i]}{j - i}" if j - i > 1 else row[i])
        i = j
    return " ".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--tokenizer", default=TOKENIZER)
    args = parser.parse_args()
    count = Counter_(args.tokenizer)
    full, runl, diff, changed = [], [], [], []
    for game_dir in sorted(Path(args.trial, "games").iterdir())[: args.games]:
        prev, level = None, None
        for line in (game_dir / "log.jsonl").open():
            rec = json.loads(line)
            if rec.get("kind") != "observation" or not rec.get("frames"):
                continue
            rows = letter_rows(rec["frames"][-1])
            full.append(count("\n".join(rows)))
            runl.append(count("\n".join(rle(r) for r in rows)))
            lv = rec.get("levels")
            if prev is None or lv != level:
                diff.append(full[-1])
            else:
                cells = [f"{r},{c}:{rows[r][c]}" for r in range(len(rows)) for c in range(len(rows[r]))
                         if rows[r][c] != prev[r][c]]
                changed.append(len(cells))
                diff.append(count(" ".join(cells)) if cells else 1)
            prev, level = rows, lv
    def s(x):
        return {"n": len(x), "mean": round(statistics.mean(x), 1), "median": statistics.median(x),
                "p90": sorted(x)[int(0.9 * (len(x) - 1))]}
    print(json.dumps({"image_png_256": 66, "full_text": s(full), "rle_rows": s(runl), "diff_or_full_at_level_start": s(diff),
                      "changed_cells_per_step": s(changed)}, indent=1))


if __name__ == "__main__":
    main()
