"""The decode step at the context depth a play has. The probe's ladder (``arc2.l4_serving.decode_steps``) decodes
after prompts of about 30 tokens; a play decodes 28-32 rows at 20-60K tokens of context, where its step was about
68 ms against the ladder's 51-56 (rehearsals A and v14, 11 October), and the 5 ms the 8-bit dense path saves on
the ladder did not show. ``python -m arc2.l4_depth --study file --output file.json`` against the ready server:
the ladder's rows first, then for each depth the same rows with prompts of that many tokens.

A long prompt is first sent for one token (the prefill, alone in its window), then again for the measured reply:
the prefix cache serves it up to its last block, so every stream decodes within seconds of the others and the
rate is taken in the later part of the level's tokens. Stream ``i``'s text is its own (no blocks shared between
streams, as in a play) and a deeper prompt extends a shallower one: the deepest goes first and the others are its
cached prefix.
"""

import argparse
import json
import random
import threading
import time
from pathlib import Path

from . import tasks
from .l4_serving import cards, decode_steps, post

LINES = 400  # the sample the tokens-a-line figure comes from


def messages(i, lines):
    rng = random.Random(1000 + i)
    body = "\n".join(str(rng.randrange(10**4, 10**6)) for _ in range(lines))
    return [{"role": "user", "content": (
        "A list of numbers follows; it is only background.\n" + body + f"\nNow write the integers from "
        f"{1000 * (i + 1) + 1} upward, one per line, without stopping and without any other text.")}]


def depth_probe(base_url, study, *, depths=(16000, 32000), rows=(32, 16), tokens=640, budget_seconds=900,
                used=cards):
    spec, begun = study["arc2"], time.monotonic()
    root, chat = base_url.rstrip("/").rsplit("/v1", 1)[0], base_url.rstrip("/") + "/chat/completions"
    model = json.loads(post(base_url.rstrip("/") + "/models", timeout=60)["body"])["data"][0]["id"]
    fields = {"temperature": (spec.get("settings") or {}).get("temperature") or float(study["temperature"]),
              "effort": (spec.get("settings") or {}).get("effort") or study["reasoning_effort"]}
    out = {"rows": sorted(rows), "tokens": tokens, "depths": [], "skipped": []}
    post(chat, {"model": model, "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                "max_tokens": 16}, timeout=240)  # the kernels compiled at first use

    def count(lines):
        body = tasks.chat_request(model, messages(0, lines), 1, **fields)
        reply = post(root + "/tokenize", {key: body[key] for key in ("model", "messages", "chat_template_kwargs")},
                     timeout=60)
        return int(json.loads(reply["body"])["count"])

    empty = count(0)
    a_line = (count(LINES) - empty) / LINES
    out.update(tokens_a_line=round(a_line, 3), prompt_overhead=empty)
    out["shallow"] = decode_steps(base_url, model, fields, sorted(rows), tokens=384, used=used)
    for k, depth in enumerate(sorted(depths, reverse=True)):  # deepest first: the rest is its cached prefix
        lines, top = int((depth - empty) / a_line), max(rows)
        row = {"depth": depth, "lines": lines}
        if time.monotonic() - begun > budget_seconds - (150 + (0 if k else top * depth / 2500)):  # no time for it
            out["skipped"].append(depth)
            continue
        replies = [None] * top

        def warm(i, lines=lines, replies=replies):
            replies[i] = post(chat, tasks.chat_request(model, messages(i, lines), 1, **fields), timeout=900)

        threads = [threading.Thread(target=warm, args=(i,), daemon=True) for i in range(top)]
        t0 = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        good = [json.loads(r["body"]).get("usage") or {} for r in replies if r and r["status"] == 200]
        row["prefill"] = {"answered": len(good), "seconds": round(time.monotonic() - t0, 1),
                          "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in good),
                          "cached_tokens": sum((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                                               for u in good),
                          "first_error": next((f"{r['status']}: {r['body'][:300]}" for r in replies
                                               if not r or r["status"] != 200), None)}
        if len(good) == top:
            # largest first: its prompts are the ones just prefilled; a smaller level reads a subset of them
            traced = spec.get("l4x4", {}).get("profile") and k == 0  # the deepest level's largest row count
            row["decode"] = [step for n in sorted(rows, reverse=True)
                             for step in decode_steps(base_url, model, fields, [n], tokens=tokens, level_seconds=600,
                                                      used=used, ask=lambda i, lines=lines: messages(i, lines),
                                                      window=(0.5, 0.95),
                                                      profile=(n,) if traced and n == top else ())]
        out["depths"].append(row)
        if len(good) < top:
            break
    out["seconds"] = round(time.monotonic() - begun, 1)
    out["depths"].sort(key=lambda row: row["depth"])
    out["summary"] = [{"depth": 0, **{str(r["at_once"]): r.get("step_ms") for r in out["shallow"]}}] + [
        {"depth": row["depth"], "prompt_tokens_a_stream": round(row["prefill"]["prompt_tokens"] / max(1, max(rows))),
         **{str(r["at_once"]): r.get("step_ms") for r in row.get("decode") or [] if not r.get("profiled")},
         "cached_share": {str(r["at_once"]): round(r["cached_tokens"] / max(1, r["prompt_tokens"]), 3)
                          for r in row.get("decode") or [] if not r.get("profiled")}} for row in out["depths"]]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--depths", default="16000,32000")
    parser.add_argument("--rows", default="32,16")
    parser.add_argument("--tokens", type=int, default=640)
    parser.add_argument("--budget-seconds", type=float, default=900)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    args = parser.parse_args(argv)
    found = depth_probe(args.base_url, json.loads(Path(args.study).read_text()),
                        depths=[int(x) for x in args.depths.split(",")], rows=[int(x) for x in args.rows.split(",")],
                        tokens=args.tokens, budget_seconds=args.budget_seconds)
    Path(args.output).write_text(json.dumps(found, indent=1))
    print("ARC2_L4_DEPTH " + json.dumps({"summary": found["summary"], "skipped": found["skipped"]}), flush=True)


if __name__ == "__main__":
    main()
