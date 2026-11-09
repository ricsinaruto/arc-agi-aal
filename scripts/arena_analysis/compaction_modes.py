"""Level-clear hazard after a compaction, by compaction mode (29 September, CPU only).

docs/validation-20260927/hero-v2-analysis/harness.md: the ``compaction_levels`` exposure split (each level attempt's
tokens cut at its first in-attempt compaction) on three families of recorded runs, "prefilled" (hero-v2 2 h),
"thinking" (compaction_thinking "gameplay") and "off" (thinking-off summaries), with the Mantel-Haenszel rate ratio
stratified by (game, level, token band) and the piecewise-exponential Poisson fit.

    uv run python scripts/arena_analysis/compaction_modes.py [--boot 300] [--out modes.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compaction_levels as cl  # noqa: E402

FAMILIES = {
    "prefilled": ["arena-hero-v2-2h-v1-modal-20260929"],
    "thinking": ["arena-duck-60m-c110-hero-21x42k-v1-modal-20260928", "arena-duck-60m-c110-hero-6x64k-v1-modal-20260928",
                 "arena-duck-60m-c110-depthwise-mtp3-v1-modal-20260928",
                 "arena-duck-60m-c110-depthwise-mtp3-fixed-order-v1-modal-20260928",
                 "arena-duck-60m-c110-mtp3-wide-kernels-v1-modal-20260928"],
    "off": ["arena-duck-60m-c110-kernels-v2-modal-20260928", "arena-duck-60m-c110-deep-explore-v1-modal-20260928",
            "arena-duck-60m-c110-mtp0-wide-v1-modal-20260927", "arena-duck-60m-c110-gittins-wide-v1-modal-20260927",
            "arena-duck-60m-c110-base-v2-modal-20260927", "arena-duck-60m-c110-gittins-c110-v1-modal-20260927",
            "arena-duck-60m-c110-base-v1-r2-modal-20260927"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out = {}
    for family, runs in FAMILIES.items():
        rows, _ = cl.load({run: (run, family) for run in runs})
        rows = [r for r in rows if cl.usable(r)]
        for name, keep in (("all", lambda r: True), ("deep", lambda r: r["level"] >= 2)):
            sub = [r for r in rows if keep(r)]
            out[f"{family}/{name}"] = {
                "attempts": len(sub),
                "mh_after_vs_rest": cl.mantel_haenszel(sub, lambda st, r: int(st == "after"), args.boot, rng),
                "poisson": cl.fit_rr(sub, ("earlier", "after"), args.boot, rng),
            }
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
