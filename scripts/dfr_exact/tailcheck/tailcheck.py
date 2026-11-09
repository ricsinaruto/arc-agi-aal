"""State-level check of request-end tail entries under MTP-k (gpu-script; throughput item 4, 22 Oct).

Per sample: request N = a real game prompt, sampled at the play temperature until it stops; request N+1 = N's prompt +
N's reply tokens + the turn suffix (<|im_end|> \\n <|im_start|>user ... <|im_start|>assistant \\n), greedy, 32 tokens, top-20
logprobs. N+1 runs three times: right after N (it may resume from N's tail entry: cached_tokens says how far), then twice
cold after a prefix-cache reset. D = cached_tokens - (len(N prompt) + len(N reply)) names the entry kind: 0 in-flight
verify, 1 dropped bonus (MTP-1 path), >= 2 multi-drop (--tail-multidrop), negative = no tail hit. An entry whose
state is wrong shows as first-token KL(hit, cold) far above KL(cold, cold2) and short greedy agreement."""
import argparse
import asyncio
import json
import math
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvoff_exact import prompts  # noqa: E402

IM_START, IM_END = 248045, 248046


def kl(p, q):
    """KL(p||q) over p's top-K (dict token -> logprob); a token missing from q gets q's smallest logprob - 1."""
    if not p or not q:
        return None
    qmin = min(q.values()) - 1.0
    zp = math.log(sum(math.exp(v) for v in p.values()))
    zq = math.log(sum(math.exp(q.get(t, qmin)) for t in p))
    return sum(math.exp(v - zp) * ((v - zp) - (q.get(t, qmin) - zq)) for t, v in p.items())


def agree(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--batches", type=int, default=7)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lo", type=int, default=6000)
    ap.add_argument("--hi", type=int, default=40000)
    ap.add_argument("--max-reply", type=int, default=4000)
    ap.add_argument("--probes", default="2,4,8,12,16,24", help="positions j: next-token top-20 after N+1 + the hit run's first j tokens")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exclude-seed", type=int, default=-1, help="drop (game, cut) pairs drawn by an earlier check's seed (its batches x batch) and duplicates")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    url = os.environ["ARC_BENCH_BASE_URL"]
    model = os.environ["ARC_BENCH_MODEL"]
    root = url.removesuffix("/v1")
    from arc_infra.model_templates import template_kwargs
    from arena.isolation import isolate_trial

    kwargs = template_kwargs(model, {"enable_thinking": True, "reasoning_effort": os.environ.get("ARC_BENCH_EFFORT", "xhigh")})
    games = json.load(open(a.contexts))["games"]
    n = a.batches * a.batch
    if a.exclude_seed >= 0:
        old = {(p["game"], p["cut"]) for p in prompts(a.contexts, n, seed=a.exclude_seed, lo=a.lo, hi=a.hi)}
        seen, ps = set(old), []
        for p in prompts(a.contexts, 4 * n, seed=a.seed, lo=a.lo, hi=a.hi):
            if (p["game"], p["cut"]) not in seen:
                seen.add((p["game"], p["cut"]))
                ps.append(p)
        ps = ps[:n]
    else:
        ps = prompts(a.contexts, n, seed=a.seed, lo=a.lo, hi=a.hi)
    rows = []
    async with httpx.AsyncClient(timeout=3600) as client:
        async def tok(body):
            r = await client.post(root + "/tokenize", json={"model": model, **body})
            r.raise_for_status()
            return r.json()["tokens"]

        async def first(i, p):
            ids = await tok({"messages": p["messages"], "tools": p["tools"], "add_generation_prompt": True,
                             "chat_template_kwargs": kwargs})
            r = await client.post(url + "/completions", json={"model": model, "prompt": ids, "temperature": 0.7,
                                                            "max_tokens": a.max_reply, "seed": i, "return_token_ids": True})
            r.raise_for_status()
            c = r.json()["choices"][0]
            out = list(c.get("token_ids") or [])
            g = next((x for x in games if x["game"] == p["game"]), None)
            nxt = ""
            if g is not None and p["cut"] + 1 < len(g["messages"]):
                m = g["messages"][p["cut"] + 1]
                nxt = m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))[:4000]
            suffix_text = "\n<|im_start|>user\n<tool_response>\n" + (nxt or "ok")[:4000] + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            suffix = await tok({"prompt": suffix_text, "add_special_tokens": False})
            body = ids + out + ([] if out and out[-1] == IM_END else [IM_END]) + suffix
            return {"i": i, "game": p["game"], "cut": p["cut"], "prompt": len(ids), "reply": len(out),
                    "finish": c.get("finish_reason"), "ends_im_end": bool(out) and out[-1] == IM_END,
                    "suffix_ok": IM_START in suffix, "next_ids": body}

        async def probe(ids):
            r = await client.post(url + "/completions", json={"model": model, "prompt": ids, "temperature": 0.0,
                                                            "max_tokens": 1, "logprobs": 20})
            r.raise_for_status()
            return ((r.json()["choices"][0].get("logprobs") or {}).get("top_logprobs") or [None])[0] or {}

        async def probes(row, h):
            return await asyncio.gather(*[probe(row["next_ids"] + h["ids"][:j]) for j in js if j <= len(h["ids"])])

        async def nxt(row):
            r = await client.post(url + "/completions", json={"model": model, "prompt": row["next_ids"], "temperature": 0.0,
                                                            "max_tokens": 32, "logprobs": 20, "return_token_ids": True})
            r.raise_for_status()
            j = r.json()
            c = j["choices"][0]
            lp = c.get("logprobs") or {}
            top = (lp.get("top_logprobs") or [None])[0] or {}
            cached = ((j.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
            return {"ids": list(c.get("token_ids") or []), "top": top, "cached": cached}

        js = [int(x) for x in a.probes.split(",") if x]
        for b in range(a.batches):
            batch = ps[b * a.batch:(b + 1) * a.batch]
            await isolate_trial(url)
            firsts = await asyncio.gather(*[first(b * a.batch + k, p) for k, p in enumerate(batch)])
            firsts = [f for f in firsts if f["finish"] == "stop"]
            hits = await asyncio.gather(*[nxt(f) for f in firsts])
            phit = await asyncio.gather(*[probes(f, h) for f, h in zip(firsts, hits)])  # resume from N's entry chain
            await isolate_trial(url)
            colds = await asyncio.gather(*[nxt(f) for f in firsts])
            await isolate_trial(url)
            pc1 = await asyncio.gather(*[probes(f, h) for f, h in zip(firsts, hits)])
            await isolate_trial(url)
            colds2 = await asyncio.gather(*[nxt(f) for f in firsts])
            await isolate_trial(url)
            pc2 = await asyncio.gather(*[probes(f, h) for f, h in zip(firsts, hits)])
            for f, h, c1, c2, ph, q1, q2 in zip(firsts, hits, colds, colds2, phit, pc1, pc2):
                d = (h["cached"] - f["prompt"] - f["reply"]) if h["cached"] is not None else None
                row = {k: v for k, v in f.items() if k != "next_ids"}
                row.update(cached=h["cached"], cold_cached=c1["cached"], D=d,
                           kl_hit=kl(c1["top"], h["top"]), kl_floor=kl(c1["top"], c2["top"]),
                           top1_hit=h["ids"][:1] == c1["ids"][:1], top1_floor=c2["ids"][:1] == c1["ids"][:1],
                           agree_hit=agree(h["ids"], c1["ids"]), agree_floor=agree(c2["ids"], c1["ids"]),
                           pkl_hit=[kl(x, y) for x, y in zip(q1, ph)], pkl_floor=[kl(x, y) for x, y in zip(q1, q2)])
                rows.append(row)
                print(json.dumps(row), flush=True)
            with open(a.out, "w") as fh:
                json.dump({"rows": rows}, fh, indent=1)

    def summ(sel):
        sel = [r for r in sel if r["kl_hit"] is not None and r["kl_floor"] is not None]
        n = len(sel)
        if not n:
            return {"n": 0}
        ph = [v for r in sel for v in r.get("pkl_hit") or [] if v is not None]
        pf = [v for r in sel for v in r.get("pkl_floor") or [] if v is not None]
        probe = {"probe_n": len(ph), "probe_kl_hit": round(sum(ph) / max(len(ph), 1), 4), "probe_kl_floor": round(sum(pf) / max(len(pf), 1), 4)}
        return {**probe, "n": n, "kl_hit": round(sum(r["kl_hit"] for r in sel) / n, 4), "kl_floor": round(sum(r["kl_floor"] for r in sel) / n, 4),
                "top1_hit": round(sum(r["top1_hit"] for r in sel) / n, 3), "top1_floor": round(sum(r["top1_floor"] for r in sel) / n, 3),
                "agree_hit": round(sum(r["agree_hit"] for r in sel) / n, 2), "agree_floor": round(sum(r["agree_floor"] for r in sel) / n, 2)}
    summary = {"multidrop_D>=2": summ([r for r in rows if r["D"] is not None and r["D"] >= 2]),
               "bonus_D=1": summ([r for r in rows if r["D"] == 1]),
               "inflight_D=0": summ([r for r in rows if r["D"] == 0]),
               "no_tail_D<0": summ([r for r in rows if r["D"] is not None and r["D"] < 0]),
               "D_counts": {str(k): sum(r["D"] == k for r in rows) for k in sorted({r["D"] for r in rows if r["D"] is not None})[:40]}}
    print(json.dumps(summary), flush=True)
    with open(a.out, "w") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
