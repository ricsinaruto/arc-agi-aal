"""Exactness probe for serving changes (throughput arm, 18 Oct): greedy replies to recorded late-game contexts,
sequentially (batch 1, deterministic) and concurrently (twice, for the run-to-run spread of batched kernels), from
the bench's idle server. Compare the JSON of two bundles (base vs change) offline.

    python kvoff_exact.py --contexts bench/steady-contexts.json --out exact.json
"""
import argparse
import asyncio
import json
import os
import random
import time

import httpx

def prompts(path, n, seed=0, lo=20000, hi=90000):
    data = json.load(open(path))
    rng = random.Random(seed)
    out = []
    games = data["games"]
    for i in range(n):
        g = games[i % len(games)]
        target = rng.uniform(lo, hi)
        k = max([c for c, t in zip(g["cuts"], g["cut_tokens"]) if t <= target] or [g["cuts"][0]])
        msgs = [dict(m) for m in g["messages"][:k]]
        while msgs and msgs[-1]["role"] == "assistant":
            msgs.pop()
        out.append({"game": g["game"], "cut": k, "tokens": dict(zip(g["cuts"], g["cut_tokens"]))[k],
                    "messages": msgs, "tools": g["tools"]})
    return out


async def ask(client, url, model, p, max_tokens, kwargs):
    body = {"model": model, "messages": p["messages"], "tools": p["tools"], "temperature": 0.0, "max_tokens": max_tokens,
            "tool_choice": "auto", "seed": 0, "chat_template_kwargs": kwargs}
    t = time.time()
    r = await client.post(url + "/chat/completions", json=body, timeout=1800)
    if r.status_code != 200:
        return {"error": r.status_code, "text": r.text[:300], "seconds": time.time() - t}
    m = r.json()["choices"][0]["message"]
    text = json.dumps({"reasoning": m.get("reasoning_content") or m.get("reasoning"), "content": m.get("content"),
                       "tool_calls": [c["function"] for c in (m.get("tool_calls") or [])]}, sort_keys=True)
    return {"text": text, "usage": r.json().get("usage"), "seconds": round(time.time() - t, 2)}


async def logprob_mode(a, url, model, kwargs):
    """Sampling-free check: actual-token prompt logprobs of the last TAIL positions, cold and with the prefix cached,
    plus greedy completions; the whole pass twice (A, B) for the server's own repeatability."""
    from arena.isolation import isolate_trial

    root = url.removesuffix("/v1")
    ps = prompts(a.contexts, a.sequential, lo=12000, hi=30000)
    report = {"mode": "logprobs", "model": model, "tail": a.tail,
              "prompts": [{k: p[k] for k in ("game", "cut", "tokens")} for p in ps], "passes": {}}
    async with httpx.AsyncClient(timeout=1800) as client:
        ids_all = []
        for p in ps:
            r = await client.post(root + "/tokenize", json={"model": model, "messages": p["messages"], "tools": p["tools"],
                                                           "add_generation_prompt": True, "chat_template_kwargs": kwargs})
            r.raise_for_status()
            ids_all.append(r.json()["tokens"])

        async def comp(ids, **extra):
            body = {"model": model, "prompt": ids, "temperature": 0.0, "seed": 0, "max_tokens": 1, **extra}
            r = await client.post(url + "/completions", json=body)
            if r.status_code != 200:
                raise RuntimeError(f"completions {r.status_code}: {r.text[:300]}")
            return r.json()

        def tail_logprobs(reply, ids):
            entries = reply["choices"][0].get("prompt_logprobs") or []
            out = []
            for pos in range(len(ids) - a.tail, len(ids)):
                e = entries[pos] or {}
                hit = e.get(str(ids[pos])) or e.get(ids[pos]) or {}
                out.append(hit.get("logprob"))
            return out

        if a.toggle:
            # Paired arms on one server (FP8 screen, 19 Oct): per prompt and pass, cold scoring with the toggle file
            # present (arm "on": the BF16 control) and absent (the variant), order alternating by prompt.
            import os as _os

            twin = {**report, "passes": {}}
            for name in ("A", "B"):
                rows_off, rows_on = [], []
                for i, ids in enumerate(ids_all):
                    for on in ((True, False) if (i + (name == "B")) % 2 == 0 else (False, True)):
                        if on:
                            open(a.toggle, "w").close()
                        elif _os.path.exists(a.toggle):
                            _os.remove(a.toggle)
                        await isolate_trial(url)
                        cold = await comp(ids, prompt_logprobs=1)
                        (rows_on if on else rows_off).append({"cold": tail_logprobs(cold, ids)})
                    print(json.dumps({"phase": name, "i": i + 1, "tokens": len(ids)}), flush=True)
                report["passes"][name], twin["passes"][name] = rows_off, rows_on
                with open(a.out, "w") as f:
                    json.dump(report, f)
                with open(a.out.replace(".json", "-control.json"), "w") as f:
                    json.dump(twin, f)
            if _os.path.exists(a.toggle):
                _os.remove(a.toggle)
            return
        for name in ("A", "B"):
            rows = []
            for ids in ids_all:
                await isolate_trial(url)
                cold = await comp(ids, prompt_logprobs=1)
                row = {"cold": tail_logprobs(cold, ids)}
                if not a.cold_only:
                    await isolate_trial(url)
                    await comp(ids[:-a.tail])
                    warm = await comp(ids, prompt_logprobs=1)
                    gen = await comp(ids, max_tokens=a.max_tokens, logprobs=1, return_tokens_as_token_ids=True)
                    lp = gen["choices"][0].get("logprobs") or {}
                    row.update(warm=tail_logprobs(warm, ids),
                               warm_cached=((warm.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
                               gen_tokens=lp.get("tokens"), gen_logprobs=lp.get("token_logprobs"))
                rows.append(row)
                print(json.dumps({"phase": name, "i": len(rows), "tokens": len(ids)}), flush=True)
            report["passes"][name] = rows
            with open(a.out, "w") as f:
                json.dump(report, f)
        # History check: prompt 0 cold three times back to back (identical recent history after the first).
        repeats = []
        for k in range(3):
            await isolate_trial(url)
            repeats.append(tail_logprobs(await comp(ids_all[0], prompt_logprobs=1), ids_all[0]))
            print(json.dumps({"phase": "R", "i": k + 1}), flush=True)
        report["repeat_first_cold"] = repeats
        with open(a.out, "w") as f:
            json.dump(report, f)


def score_entries(entries, ids):
    """B1 screens (20 Oct): per position p >= 1 of a prompt_logprobs=1 reply, the actual token's log-prob and rank and
    the top-1 token's id and log-prob."""
    lp, rank, top_id, top_lp = [], [], [], []
    for pos in range(1, len(ids)):
        e = (entries[pos] if pos < len(entries) else None) or {}
        hit = e.get(str(ids[pos])) or e.get(ids[pos]) or {}
        lp.append(hit.get("logprob"))
        rank.append(hit.get("rank"))
        best = next(((int(k), v.get("logprob")) for k, v in e.items() if v.get("rank") == 1), (None, None))
        top_id.append(best[0])
        top_lp.append(best[1])
    return {"lp": lp, "rank": rank, "top_id": top_id, "top_lp": top_lp}


async def ids_mode(a, url, model):
    """B1 screens (20 Oct): every position of held-out token sequences (the all25 corpus screen, 25 games x 16K
    tokens), cold, two passes. Restartable: finished sequences in --out are kept and skipped."""
    from arena.isolation import isolate_trial

    seqs = json.load(open(a.contexts))
    report = (json.load(open(a.out)) if os.path.exists(a.out) else
              {"mode": "ids", "model": model, "contexts": os.path.basename(a.contexts),
               "sequences": [{k: s.get(k) for k in ("family", "game", "run", "tokens", "assistant_spans")} for s in seqs],
               "passes": {}})
    async with httpx.AsyncClient(timeout=3600) as client:
        for name in ("A", "B"):
            rows = report["passes"].setdefault(name, [])
            for i in range(len(rows), len(seqs)):
                ids = seqs[i]["ids"]
                await isolate_trial(url)
                t = time.time()
                r = await client.post(url + "/completions", json={"model": model, "prompt": ids, "temperature": 0.0,
                                                                  "seed": 0, "max_tokens": 1, "prompt_logprobs": 1})
                if r.status_code != 200:
                    raise RuntimeError(f"completions {r.status_code}: {r.text[:300]}")
                row = score_entries(r.json()["choices"][0].get("prompt_logprobs") or [], ids)
                row["seconds"] = round(time.time() - t, 2)
                row["cached"] = ((r.json().get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
                rows.append(row)
                tmp = a.out + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(report, f)
                os.replace(tmp, a.out)
                lps = [x for x in row["lp"] if x is not None]
                print(json.dumps({"phase": name, "i": i + 1, "tokens": len(ids), "seconds": row["seconds"],
                                  "mean_lp": round(sum(lps) / max(len(lps), 1), 4), "missing": len(row["lp"]) - len(lps)}),
                      flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sequential", type=int, default=12)
    ap.add_argument("--concurrent", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--mode", choices=("text", "logprobs", "ids"), default="text")
    ap.add_argument("--tail", type=int, default=512)
    ap.add_argument("--toggle", default="", help="logprob mode: paired arms via this flag file (present = control); "
                                                   "the control's report goes to <out>-control.json")
    ap.add_argument("--cold-only", action="store_true", help="logprob mode: cold scoring only (quality screens)")
    a = ap.parse_args()
    url = os.environ["ARC_BENCH_BASE_URL"]
    model = os.environ["ARC_BENCH_MODEL"]
    from arc_infra.model_templates import template_kwargs

    kwargs = template_kwargs(model, {"enable_thinking": True,
                                    "reasoning_effort": os.environ.get("ARC_BENCH_EFFORT", "xhigh")})
    if a.mode == "logprobs":
        return await logprob_mode(a, url, model, kwargs)
    if a.mode == "ids":
        return await ids_mode(a, url, model)
    ps = prompts(a.contexts, max(a.sequential, a.concurrent))
    report = {"model": model, "max_tokens": a.max_tokens,
              "prompts": [{k: p[k] for k in ("game", "cut", "tokens")} for p in ps], "phases": {}}

    def save():
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)

    from arena.isolation import isolate_trial

    async with httpx.AsyncClient() as client:
        for name in ("sequential", "sequential_repeat"):  # each from an empty prefix cache: run-to-run determinism
            await isolate_trial(url)
            seq = []
            for p in ps[:a.sequential]:
                seq.append(await ask(client, url, model, p, a.max_tokens, kwargs))
                print(json.dumps({"phase": name, "i": len(seq), "seconds": seq[-1].get("seconds")}), flush=True)
            report["phases"][name] = seq
            save()
        for name in ("concurrent_1", "concurrent_2"):
            t = time.time()
            res = await asyncio.gather(*(ask(client, url, model, p, a.max_tokens, kwargs) for p in ps[:a.concurrent]))
            report["phases"][name] = res
            print(json.dumps({"phase": name, "seconds": round(time.time() - t, 1),
                              "errors": sum("error" in r for r in res)}), flush=True)
            save()
    c1, c2 = report["phases"]["concurrent_1"], report["phases"]["concurrent_2"]
    s1, s2 = report["phases"]["sequential"], report["phases"]["sequential_repeat"]
    report["self_agreement"] = {"concurrent_1_vs_2": sum(x.get("text") == y.get("text") for x, y in zip(c1, c2)),
                                "sequential_vs_repeat": sum(x.get("text") == y.get("text") for x, y in zip(s1, s2)),
                                "of": [len(c1), len(s1)]}
    save()
    print(json.dumps(report["self_agreement"]), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
