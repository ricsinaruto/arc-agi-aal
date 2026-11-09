#!/usr/bin/env python3
"""Inputs of the long-context drift test (29 September; docs/validation-20260927/hero-behavior.md, Test A).

Teacher-forced sequences at arena context lengths (the phase-2/kv8/hero drift checks used ~512-token passages): each
c110 game's ``prompts/prompt.log`` holds its latest model request; ``scripts/expert_usage_inputs.py`` parses it back
into chat messages with the recorded texts, renders it with the served chat template (tools, xhigh effort) and
tokenizes it with the served tokenizer (the expert-quant end-to-end inputs' path). The longest request of each dev
family (never a holdout game, ``posttrain.split``) is kept, capped at ``--cap`` tokens (a prefix), longest families
first, until ``--sequences``.

Output (``--out``): ``quality/prompts.json`` in the kernel-quality format (``teacher_forced`` items with token ids and
assistant spans; no greedy items) and ``selection.json`` (what was chosen, sha256 of the prompts file).

Usage: <tokvenv>/bin/python scripts/drift_long_inputs.py --out ./runs/drift-long/inputs-v1
Needs ``tokenizers`` and ``jinja2``; nothing touches a GPU.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from expert_quant_inputs import render_sequence  # noqa: E402
from expert_usage_inputs import TOKENIZER_DIR, assistant_spans, family, renderer  # noqa: E402
from specialization_vocab import RUNS  # noqa: E402

from posttrain.split import DEV, HOLDOUT  # noqa: E402

C110_GLOBS = ("modal-validation/arena-duck-60m-c110-*/*/arena/trials/duck-transfer",
              "kaggle-arena-flash-competition-hero-v1/arena/trials/duck-transfer")


def candidates():
    for pattern in C110_GLOBS:
        for trial in sorted(glob.glob(os.path.join(RUNS, pattern))):
            run = os.path.relpath(trial, RUNS).split("/")[1 if pattern.startswith("modal") else 0]
            for gdir in sorted(glob.glob(os.path.join(trial, "games", "*"))):
                log = os.path.join(gdir, "prompts", "prompt.log")
                if os.path.isfile(log):
                    yield run, gdir, log, family(trial, gdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cap", type=int, default=44_000)
    parser.add_argument("--min-tokens", type=int, default=36_000)
    parser.add_argument("--sequences", type=int, default=7)
    # prefix-unit-int4 (29 September): several games per family (distinct games, longest first), for >= 20 contexts.
    parser.add_argument("--per-family", type=int, default=1)
    args = parser.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(TOKENIZER_DIR, "tokenizer.json"))
    template = renderer()
    best = {}
    for run, gdir, log, fam in candidates():
        if fam not in DEV or fam in HOLDOUT:
            continue
        rendered = render_sequence(template, tok, gdir, log, args.cap)
        if rendered is None:
            continue
        ids, full, match = rendered
        if full < args.min_tokens or not assistant_spans(ids):
            continue
        best.setdefault(fam, []).append({"run": run, "game": os.path.basename(gdir), "family": fam,
                                         "tokens": len(ids), "full_tokens": full, "ids": ids, "exact_match": match})
    kept = []
    for rows in best.values():
        # one entry per game (a clone's run/game pair), longest first
        rows = sorted(rows, key=lambda r: -r["full_tokens"])
        seen = set()
        for r in rows:
            if (r["run"], r["game"]) not in seen and len(seen) < args.per_family:
                seen.add((r["run"], r["game"]))
                kept.append(r)
    # longest families first, then round-robin so the first N stay spread over families
    kept.sort(key=lambda r: -r["full_tokens"])
    rank = {}
    for r in kept:
        rank[id(r)] = sum(1 for q in kept if q["family"] == r["family"] and q["full_tokens"] > r["full_tokens"])
    chosen = sorted(kept, key=lambda r: (rank[id(r)], -r["full_tokens"]))[:args.sequences]
    teacher = [{"index": i, "text": None, "ids": r["ids"], "source": f"{r['run']}/{r['game']}", "family": r["family"],
                "assistant_spans": assistant_spans(r["ids"])} for i, r in enumerate(chosen)]
    os.makedirs(os.path.join(args.out, "quality"), exist_ok=True)
    path = os.path.join(args.out, "quality", "prompts.json")
    with open(path, "w") as f:
        json.dump({"teacher_forced": teacher, "greedy": []}, f)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    selection = {"cap": args.cap, "min_tokens": args.min_tokens, "families_eligible": sorted(best),
                 "holdout_excluded": sorted(HOLDOUT), "prompts_sha256": digest,
                 "rows": [{k: r[k] for k in ("run", "game", "family", "tokens", "full_tokens", "exact_match")}
                          for r in chosen],
                 "positions": sum(len(t["ids"]) - 1 for t in teacher),
                 "assistant_positions": sum(b - a for t in teacher for a, b in t["assistant_spans"])}
    with open(os.path.join(args.out, "selection.json"), "w") as f:
        json.dump(selection, f, indent=1)
    print(json.dumps({k: v for k, v in selection.items() if k != "rows"}))
    for r in selection["rows"]:
        print(r)


if __name__ == "__main__":
    main()
