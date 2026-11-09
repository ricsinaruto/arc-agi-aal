#!/usr/bin/env python3
"""Recorded arena contexts for the speculative-decoding pilot (30 September; docs/validation-20260930/spec-decode.md).

Each context is a real Duck request of the three-block run (``arena-hero-v2-blocks-v1``, 37 x 93K, int4 KV): a game's
logged latest request (``prompts/prompt.log``, parsed back into chat messages with the recorded texts by
``expert_usage_inputs``), cut at an assistant-turn boundary so that the model is asked for exactly the reply it gave
there, rendered with the served chat template (tools, xhigh effort, generation prompt) and tokenized with the served
tokenizer. Targets are spread uniformly over ``--low``..``--high`` tokens (the pilot must hold 40 of them at MTP-2 on
the int4 pool), one game per family in turn so that consecutive contexts come from different source games.

Output (``--out``): ``{"contexts": [{"id", "game", "family", "tokens", "target", "cut_message", "prompt_ids"}], ...}``.

Usage: <venv with tokenizers + jinja2>/bin/python scripts/spec_decode_inputs.py --out <file>
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
from expert_usage_inputs import (PARAMETER_DESCRIPTION, TOKENIZER_DIR, exact_sources, exactify, family,  # noqa: E402
                                 parse_prompt_log, renderer)
from specialization_vocab import RUNS  # noqa: E402

TRIAL = ("modal-validation/arena-hero-v2-blocks-v1-modal-20260929/arena-hero-v2-blocks-v1-modal-20260929/arena/"
         "trials/duck-transfer")


def render(template, tok, description, messages):
    tools = [{"type": "function", "function": {
        "name": "python", "description": description,
        "parameters": {"type": "object", "properties": {"code": {"type": "string",
                                                                  "description": PARAMETER_DESCRIPTION}},
                       "required": ["code"]}}}]
    text = template.render(messages=messages, tools=tools, add_generation_prompt=True,
                           enable_thinking=True, reasoning_effort="xhigh")
    return tok.encode(text, add_special_tokens=False).ids


def best_cut(template, tok, description, messages, target):
    """The longest prefix ending before an assistant turn whose rendered prompt is at most ``target`` tokens."""
    cuts = [j for j, m in enumerate(messages) if m["role"] == "assistant" and j >= 2]
    lo, hi, best = 0, len(cuts) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        ids = render(template, tok, description, messages[:cuts[mid]])
        if len(ids) <= target:
            best, lo = (cuts[mid], ids), mid + 1
        else:
            hi = mid - 1
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--low", type=int, default=9000)
    parser.add_argument("--high", type=int, default=30000)
    args = parser.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(TOKENIZER_DIR, "tokenizer.json"))
    template = renderer()
    trial = os.path.join(RUNS, TRIAL)
    by_family = {}
    for gdir in sorted(glob.glob(os.path.join(trial, "games", "*"))):
        log = os.path.join(gdir, "prompts", "prompt.log")
        if os.path.isfile(log):
            by_family.setdefault(family(trial, gdir), []).append(gdir)
    order = []
    for rank in range(max(len(v) for v in by_family.values())):
        order.extend(v[rank] for _, v in sorted(by_family.items()) if rank < len(v))
    targets = [int(args.low + (i + 0.5) / args.count * (args.high - args.low)) for i in range(args.count)]
    # Interleave short and long targets so the game order and the length are independent.
    targets = targets[0::2] + targets[1::2][::-1]
    contexts, skipped, used, cache = [], [], set(), {}
    for target in targets:
        for gdir in order:  # the first unused game (family round-robin order) with a cut in [0.8 T, T]
            if gdir in used:
                continue
            if gdir not in cache:
                parsed = parse_prompt_log(os.path.join(gdir, "prompts", "prompt.log"))
                if parsed is not None:
                    stats = {"system": [0, 0], "assistant": [0, 0], "tool": [0, 0]}
                    parsed = (parsed[0], exactify(parsed[1], exact_sources(gdir), stats))
                cache[gdir] = parsed
            if cache[gdir] is None:
                continue
            description, messages = cache[gdir]
            cut = best_cut(template, tok, description, messages, target)
            if cut is None or len(cut[1]) < 0.8 * target:
                continue
            j, ids = cut
            used.add(gdir)
            contexts.append({"id": len(contexts), "game": os.path.basename(gdir), "family": family(trial, gdir),
                             "tokens": len(ids), "target": target, "cut_message": j,
                             "recorded_reply_role": messages[j]["role"], "prompt_ids": ids})
            break
        else:
            skipped.append(target)
    blob = json.dumps(contexts, separators=(",", ":"))
    out = {"source": TRIAL, "count": len(contexts), "low": args.low, "high": args.high,
           "mean_tokens": round(sum(c["tokens"] for c in contexts) / len(contexts)),
           "families": sorted({c["family"] for c in contexts}), "skipped": skipped,
           "contexts_sha256": hashlib.sha256(blob.encode()).hexdigest(), "contexts": contexts}
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(json.dumps({k: v for k, v in out.items() if k != "contexts"}))
    print([(c["family"], c["tokens"]) for c in contexts])


if __name__ == "__main__":
    main()
