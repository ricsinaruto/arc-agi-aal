#!/usr/bin/env python3
"""Real recorded contexts for the kernel-profile pilot (30 September; docs/validation-20260930/kernel-profile.md).

Each game's ``prompts/prompt.log`` in a recorded blocks run holds its latest Duck request (system prompt, user turns,
every earlier assistant turn with its reasoning and tool call, tool results). It is parsed back into chat messages
(``scripts/expert_usage_inputs.py``), and the longest message prefix that ends on a user / tool turn and renders to at
most a target length (served chat template, python tool, xhigh effort, local tokenizer) becomes one context. The
server receives the messages through ``/v1/chat/completions`` with the tool list and ``tool_choice`` auto, as the
Duck harness sends them; the board images are not in the log (64 tokens each), so the served counts are a little
lower than the recorded prompts.

Targets: ``--long`` contexts evenly spread over [``--long-min``, ``--long-max``] (the 24 / 40-stream points) and
``--short`` contexts over [``--short-min``, ``--short-max``] (added for the 56-stream point, so that 56 fit the gate),
one game per context, drawn from distinct games.

Usage: tokvenv/bin/python scripts/kernel_profile_inputs.py --run-dir <trial dir> --out <file>
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_usage_inputs import (  # noqa: E402
    PARAMETER_DESCRIPTION,
    TOKENIZER_DIR,
    exact_sources,
    exactify,
    parse_prompt_log,
    renderer,
)


def tools_for(description):
    return [{"type": "function", "function": {
        "name": "python", "description": description,
        "parameters": {"type": "object", "properties": {"code": {"type": "string",
                                                                  "description": PARAMETER_DESCRIPTION}},
                       "required": ["code"]}}}]


def api_messages(messages):
    """The parsed messages in the OpenAI chat shape the harness sends (reasoning field, JSON-string arguments)."""
    out = []
    for m in messages:
        m = dict(m)
        if m["role"] == "assistant":
            reasoning = m.pop("reasoning_content", "")
            if reasoning:
                m["reasoning"] = reasoning
            calls = m.get("tool_calls") or []
            if calls:
                m["tool_calls"] = [{"id": c.get("id") or f"call_{i}", "type": "function",
                                    "function": {"name": c["function"]["name"],
                                                 "arguments": json.dumps(c["function"]["arguments"])}}
                                   for i, c in enumerate(calls)]
            else:
                m.pop("tool_calls", None)
        out.append(m)
    return out


def boundaries(messages):
    """Prefix lengths k (messages[:k]) that end on a user or tool turn, the harness's request boundaries."""
    return [k for k in range(2, len(messages) + 1) if messages[k - 1]["role"] in ("user", "tool")]


def spread(count, low, high):
    return [round(low + (high - low) * (i + 0.5) / count) for i in range(count)]


def main():
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, action="append",
                        help="recorded trial directories with games/*/prompts/prompt.log (repeatable)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--long", type=int, default=40)
    parser.add_argument("--long-min", type=int, default=28000)
    parser.add_argument("--long-max", type=int, default=56000)
    parser.add_argument("--short", type=int, default=16)
    parser.add_argument("--short-min", type=int, default=12000)
    parser.add_argument("--short-max", type=int, default=24000)
    args = parser.parse_args()
    tok = Tokenizer.from_file(os.path.join(TOKENIZER_DIR, "tokenizer.json"))
    template = renderer()
    match = {"system": [0, 0], "assistant": [0, 0], "tool": [0, 0]}

    def length(messages, tools):
        text = template.render(messages=messages, tools=tools, add_generation_prompt=True,
                               enable_thinking=True, reasoning_effort="xhigh")
        return len(tok.encode(text, add_special_tokens=False).ids)

    games = []
    logs = [log for run in args.run_dir for log in sorted(glob.glob(os.path.join(run, "games/*/prompts/prompt.log")))]
    for log in logs:
        gdir = os.path.dirname(os.path.dirname(log))
        parsed = parse_prompt_log(log)
        if parsed is None:
            continue
        description, messages = parsed
        messages = exactify(messages, exact_sources(gdir), match)
        tools = tools_for(description)
        cuts = boundaries(messages)
        if not cuts:
            continue
        games.append({"game": os.path.basename(gdir), "run": gdir.split("/arena/trials/")[0].rsplit("/", 1)[-1], "messages": messages, "tools": tools, "cuts": cuts,
                      "full": length(messages[:cuts[-1]], tools)})
    games.sort(key=lambda g: -g["full"])
    targets = spread(args.long, args.long_min, args.long_max)[::-1] + spread(args.short, args.short_min,
                                                                             args.short_max)[::-1]
    chosen, used = [], set()
    for target in targets:
        # The game with the smallest full length that still reaches the target (keeps long games for long targets).
        fits = [g for g in games if (g["run"], g["game"]) not in used and g["full"] >= target]
        if not fits:
            raise SystemExit(f"no unused game reaches {target} tokens")
        g = min(fits, key=lambda g: g["full"])
        lo, hi, best = 0, len(g["cuts"]) - 1, None
        while lo <= hi:  # the longest cut at or below the target
            mid = (lo + hi) // 2
            n = length(g["messages"][:g["cuts"][mid]], g["tools"])
            if n <= target:
                best, lo = (g["cuts"][mid], n), mid + 1
            else:
                hi = mid - 1
        if best is None:
            continue
        used.add((g["run"], g["game"]))
        k, n = best
        chosen.append({"run": g["run"], "game": g["game"], "target": target, "tokens_local": n, "messages_kept": k,
                       "messages_total": len(g["messages"]),
                       "messages": api_messages(g["messages"][:k]), "tools": g["tools"]})
    source = {"run_dir": args.run_dir, "tokenizer": TOKENIZER_DIR, "exact_text_matches": match,
              "chat_template_sha256": hashlib.sha256(
                  open(os.path.join(TOKENIZER_DIR, "chat_template.jinja"), "rb").read()).hexdigest()}
    data = {"format": "arc-kernel-profile-contexts-v1", "source": source,
            "long": args.long, "short": args.short, "contexts": chosen}
    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    lengths = [c["tokens_local"] for c in chosen]
    print(json.dumps({"contexts": len(chosen), "long_mean": round(sum(lengths[:args.long]) / args.long),
                      "all_mean": round(sum(lengths) / len(lengths)), "min": min(lengths), "max": max(lengths),
                      "games": len(games), "matches": match}))


if __name__ == "__main__":
    main()
