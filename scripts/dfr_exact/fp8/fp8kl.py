"""FP8 dense root-cause screen (throughput arm, 21 Oct; gpu-script beside an eager --fp8-toggle server).

Per prompt, every arm scores the same token ids cold (prefix cache reset before each call) with prompt_logprobs=K over
the last TAIL positions; arms differ only in the toggle file (fp8_dense._arc_toggle_bf16: absent = FP8 everywhere,
"all" = BF16, {"bf16": keys} / {"fp8": keys}). Arm order rotates by prompt. Per arm vs the reference arm: mean KL on the
reference's top-K (tokens missing from the arm's top-K take the arm's K-th log-prob, an upper bound on their mass),
actual-token NLL difference, top-1 agreement. A second BF16 arm gives the floor (server nondeterminism).
"""
import argparse
import asyncio
import json
import math
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvoff_exact import prompts  # noqa: E402  (copied next to this script by the builder)


def parse_entries(reply, ids, tail):
    entries = reply["choices"][0].get("prompt_logprobs") or []
    out = []
    for pos in range(len(ids) - tail, len(ids)):
        e = entries[pos] or {}
        d = {int(k): v["logprob"] for k, v in e.items()}
        out.append((ids[pos], d))
    return out


TOPK = 20


def compare(ref, arm):
    kl, dnll, top1, n = 0.0, 0.0, 0, 0
    for (tok, r), (_, q) in zip(ref, arm):
        if not r or not q:
            continue
        vals = sorted(q.values(), reverse=True)
        floor = vals[min(len(vals), TOPK) - 1]  # the arm's K-th log-prob (the dict also holds the actual token)
        kl += sum(math.exp(lr) * (lr - q.get(t, floor)) for t, lr in r.items())
        dnll += (-q.get(tok, floor)) - (-r.get(tok, min(r.values())))
        top1 += max(r, key=r.get) == max(q, key=q.get)
        n += 1
    return {"kl": kl / max(n, 1), "dnll": dnll / max(n, 1), "top1": top1 / max(n, 1), "n": n}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--arms", required=True, help="JSON file: {name: null (no file) | 'all' | {bf16|fp8: [keys]}}")
    ap.add_argument("--toggle", required=True)
    ap.add_argument("--reference", default="bf16")
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--tail", type=int, default=256)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--lo", type=int, default=12000, help="prompt cut range (tokens), low end")
    ap.add_argument("--hi", type=int, default=30000, help="prompt cut range (tokens), high end")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global TOPK
    TOPK = a.topk
    url = os.environ["ARC_BENCH_BASE_URL"]
    model = os.environ["ARC_BENCH_MODEL"]
    root = url.removesuffix("/v1")
    from arc_infra.model_templates import template_kwargs
    from arena.isolation import isolate_trial

    kwargs = template_kwargs(model, {"enable_thinking": True,
                                    "reasoning_effort": os.environ.get("ARC_BENCH_EFFORT", "xhigh")})
    arms = json.load(open(a.arms))
    names = list(arms)
    ps = prompts(a.contexts, a.prompts, lo=a.lo, hi=a.hi)
    report = {"arms": arms, "reference": a.reference, "tail": a.tail, "topk": a.topk,
              "prompts": [{k: p[k] for k in ("game", "cut", "tokens")} for p in ps], "rows": []}
    started = time.time()
    async with httpx.AsyncClient(timeout=1800) as client:
        for i, p in enumerate(ps):
            r = await client.post(root + "/tokenize", json={"model": model, "messages": p["messages"], "tools": p["tools"],
                                                           "add_generation_prompt": True, "chat_template_kwargs": kwargs})
            r.raise_for_status()
            ids = r.json()["tokens"]
            got = {}
            order = names[i % len(names):] + names[:i % len(names)]
            for name in order:
                spec = arms[name]
                if spec is None:
                    if os.path.exists(a.toggle):
                        os.remove(a.toggle)
                else:
                    tmp = a.toggle + ".tmp"
                    with open(tmp, "w") as fh:
                        fh.write(spec if isinstance(spec, str) else json.dumps(spec))
                    os.replace(tmp, a.toggle)
                await asyncio.sleep(0.05)
                await isolate_trial(url)
                body = {"model": model, "prompt": ids, "temperature": 0.0, "seed": 0, "max_tokens": 1,
                        "prompt_logprobs": a.topk}
                resp = await client.post(url + "/completions", json=body)
                if resp.status_code != 200:
                    raise RuntimeError(f"completions {resp.status_code}: {resp.text[:300]}")
                got[name] = parse_entries(resp.json(), ids, a.tail)
            row = {"i": i, "tokens": len(ids), "vs_ref": {n: compare(got[a.reference], got[n]) for n in names
                                                          if n != a.reference}}
            report["rows"].append(row)
            print(json.dumps({"i": i + 1, "seconds": round(time.time() - started), "tokens": len(ids),
                              **{n: round(v["kl"], 4) for n, v in row["vs_ref"].items()}}), flush=True)
            with open(a.out, "w") as fh:
                json.dump(report, fh)
    if os.path.exists(a.toggle):
        os.remove(a.toggle)
    summary = {}
    for n in names:
        if n == a.reference:
            continue
        xs = [r["vs_ref"][n] for r in report["rows"]]
        kl = [x["kl"] for x in xs]
        m = sum(kl) / len(kl)
        sd = (sum((v - m) ** 2 for v in kl) / max(len(kl) - 1, 1)) ** 0.5
        summary[n] = {"kl": m, "kl_se": sd / len(kl) ** 0.5, "dnll": sum(x["dnll"] for x in xs) / len(xs),
                      "top1": sum(x["top1"] for x in xs) / len(xs)}
    report["summary"] = summary
    with open(a.out, "w") as fh:
        json.dump(report, fh)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
