"""Build M0, the LoRA pipeline-test training set (level-1 packs, tokenized, <= 32K) and its held-out
evaluation set, from the built pilot dataset (``scripts/posttrain/build_dataset.py``). CPU only.

Usage:
    uv run --with tokenizers python scripts/posttrain/build_m0.py [--out data/posttrain/m0]
        [--dataset data/posttrain/dataset] [--tokenizer PATH] [--max-tokens 32768] [--dev-cap-per-game 8]

Writes ``train.jsonl`` and ``eval.jsonl`` (git-ignored), ``stats.json`` and ``manifest.json`` (sha256 of
each file; the Modal trainer checks them). See ``posttrain.m0`` for the selection and the format.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from posttrain.m0 import DATASET, M0Config, Tokenizer, build  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "data/posttrain/m0"))
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--tokenizer", default=None, help="Ornith tokenizer.json (default: the local HF cache)")
    parser.add_argument("--max-tokens", type=int, default=M0Config.max_tokens)
    parser.add_argument("--dev-cap-per-game", type=int, default=M0Config.dev_cap_per_game)
    args = parser.parse_args(argv)
    config = M0Config(max_tokens=args.max_tokens, dev_cap_per_game=args.dev_cap_per_game)
    stats = build(args.out, dataset=args.dataset, tokenizer=Tokenizer(args.tokenizer), config=config)
    for split in ("train", "eval"):
        row = stats[split]
        print(json.dumps({split: {k: row[k] for k in ("sequences", "tokens", "loss_tokens", "targets",
                                                       "targets_dropped_by_window", "truncated_sequences",
                                                       "images", "straddling_tokens", "by_source")}}))


if __name__ == "__main__":
    main()
