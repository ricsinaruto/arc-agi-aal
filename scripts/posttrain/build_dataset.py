"""Build the pilot SFT mixture and the held-out evaluation split (CPU only).

Usage:
    uv run --with tokenizers python scripts/posttrain/build_dataset.py [--out data/posttrain/dataset]
        [--dev-max-share 0.7] [--teacher-epochs 2] [--aux-max-share 0.1] [--with-solver]
        [--solver-max-share 0.02] [--cap-per-level 8] [--near-duplicate 0.9] [--no-write]

Inputs (rebuild them first when their sources change):
    uv run python scripts/posttrain/extract_segments.py                    # dev segments (full context)
    uv run python scripts/posttrain/extract_segments.py --source teacher   # generated-game teacher runs
    uv run python scripts/posttrain/build_solver_sft.py                    # solver records (+ eval split)
    uv run python scripts/posttrain/build_aux_tasks.py [--generated | --eval]

Adding a teacher run: one line in ``posttrain.segments.TEACHER_RUNS``, then the teacher extraction and
this build. Writes ``train_packed.jsonl``, ``train_turns.jsonl``, ``eval_packed.jsonl`` and
``eval_turns.jsonl`` (git-ignored) and ``stats.json`` (tracked). See ``posttrain.dataset`` for the
filters, the budgets and the formats.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from posttrain.dataset import ChatTemplate, Mixture, TokenCounter, build  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = Mixture()
    parser.add_argument("--out", default=str(ROOT / "data/posttrain/dataset"))
    parser.add_argument("--dev-max-share", type=float, default=defaults.dev_max_share)
    parser.add_argument("--teacher-epochs", type=int, default=defaults.teacher_epochs)
    parser.add_argument("--aux-max-share", type=float, default=defaults.aux_max_share)
    parser.add_argument("--with-solver", action="store_true", help="add solver records (loss on the tool call)")
    parser.add_argument("--solver-max-share", type=float, default=defaults.solver_max_share)
    parser.add_argument("--cap-per-level", type=int, default=defaults.cap_per_level)
    parser.add_argument("--near-duplicate", type=float, default=defaults.near_duplicate)
    parser.add_argument("--template", default=None, help="chat template (default: the Ornith-1.5 fixture)")
    parser.add_argument("--tokenizer", default=None, help="tokenizer.json (default: the local Qwen3.6 cache)")
    parser.add_argument("--no-write", action="store_true", help="compute the stats only")
    args = parser.parse_args(argv)
    mixture = Mixture(dev_max_share=args.dev_max_share, teacher_epochs=args.teacher_epochs,
                      aux_max_share=args.aux_max_share, solver=args.with_solver,
                      solver_max_share=args.solver_max_share, cap_per_level=args.cap_per_level,
                      near_duplicate=args.near_duplicate)
    counter = TokenCounter(args.tokenizer)
    if not counter.exact:
        print("tokenizers or the Qwen3.6 tokenizer.json not found: token counts are estimates "
              "(run with `uv run --with tokenizers`)", file=sys.stderr)
    template = ChatTemplate(args.template) if args.template else ChatTemplate()
    started = time.monotonic()
    stats = build(args.out, mixture=mixture, template=template, counter=counter, write=not args.no_write)
    stats["seconds"] = round(time.monotonic() - started, 1)
    if not args.no_write:
        (Path(args.out) / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    mix = stats["mixture"]
    for source in ("dev", "teacher", "aux", "solver"):
        row = mix[source]
        print(f"{source:8s} {row['target_tokens']:>10,} target tokens ({row['share']:.1%}), "
              f"{row['packed']['items']} packed / {row['per_turn']['items']} per-turn items")
    print(f"total {mix['total_target_tokens']:,} target tokens; packed {mix['packed']['tokens']:,} tokens in "
          f"{mix['packed']['items']} items; per-turn {mix['per_turn']['tokens']:,} tokens; {stats['seconds']} s")


if __name__ == "__main__":
    main()
