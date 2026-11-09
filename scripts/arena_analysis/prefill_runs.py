"""Run ``prefill_attribution`` over the c110 runs that have telemetry and write one summary JSON (28 September).

    uv run --with tokenizers python scripts/arena_analysis/prefill_runs.py --root ./runs/modal-validation \
        --out docs/validation-20260927/prefill-attribution.json [--runs kernels-v2 base-v1-r2]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefill_attribution import TOKENIZER, TokenCounter, attribute, link_prev_reasoning, summarize  # noqa: E402

RUNS = {
    "kernels-v2": "arena-duck-60m-c110-kernels-v2-modal-20260928",
    "base-v1-r2": "arena-duck-60m-c110-base-v1-r2-modal-20260927",
    "mtp0-wide": "arena-duck-60m-c110-mtp0-wide-v1-modal-20260927",
    "gittins-wide": "arena-duck-60m-c110-gittins-wide-v1-modal-20260927",
    "base-v2": "arena-duck-60m-c110-base-v2-modal-20260927",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    parser.add_argument("--tokenizer", default=TOKENIZER)
    args = parser.parse_args()
    count = TokenCounter(args.tokenizer)
    out = {}
    for label in args.runs:
        run = RUNS[label]
        trial = Path(args.root, run, run, "arena", "trials", "duck-transfer")
        res = attribute(trial, count)
        link_prev_reasoning(res["requests"])
        out[label] = {"run": run, **summarize(res)}
        print(label, out[label]["measured_uncached"], [(t["key"], t["tokens"]) for t in out[label]["table"]])
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
