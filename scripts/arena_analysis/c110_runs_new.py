"""Level attempts of the c110 runs recorded after c110-path/c110-attempts.jsonl (28 September): kernels-v2,
deep-explore, mtp3-wide-kernels and the two depthwise-mtp3 runs, in the ``c110_runs.py`` row format (``game`` = source
game, ``clone`` = clone id). Input to the depthwise simulator's refitted world (``depthwise_sim.py``, world ``w2``).

    PYTHONPATH=src:scripts/arena_analysis uv run python scripts/arena_analysis/c110_runs_new.py \
        --out docs/validation-20260927/depthwise-sim/c110-attempts-new.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c110_runs as c  # noqa: E402

NEW = {
    "arena-duck-60m-c110-kernels-v2-modal-20260928": ("kernels-v2", "fixed-2x", "mtp0k-6"),
    "arena-duck-60m-c110-deep-explore-v1-modal-20260928": ("deep-explore", "deep-explore", "mtp0k-6"),
    "arena-duck-60m-c110-mtp3-wide-kernels-v1-modal-20260928": ("mtp3-wide-kernels", "fixed-2x", "mtp3p4-6"),
    "arena-duck-60m-c110-depthwise-mtp3-v1-modal-20260928": ("depthwise-mtp3", "depthwise", "mtp3p4-6"),
    "arena-duck-60m-c110-depthwise-mtp3-fixed-order-v1-modal-20260928": ("depthwise-mtp3-fixed-order",
                                                                         "depthwise", "mtp3p4-6"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./runs/modal-validation")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for run, meta in NEW.items():
        c.RUNS[run] = meta
        r, s = c.summarise(args.root, run)
        rows += r
        print(f"{meta[0]:28s} {len(r):4d} attempts, score {s['score']:.2f}, levels {s['levels']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
