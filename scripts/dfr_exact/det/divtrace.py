"""Divergence trace driver (gpu-script; throughput, 21 Oct). Per prompt: three cold prefills of the same ids (tags A, B,
C back to back), each with exact per-module output checksums (divtrace_runtime). A module whose input checksum matches
across passes but whose output differs is a nondeterminism source; the first divergence in execution order is where
the passes split."""
import argparse
import asyncio
import collections
import json
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvoff_exact import prompts  # noqa: E402


IM_START, IM_END, ASSISTANT = 248045, 248046, 74455  # <|im_start|>, <|im_end|>, "assistant" (Flash tokenizer.json)


def assistant_positions(ids):
    """Positions of every assistant-turn token (after "<|im_start|>assistant\\n", through its <|im_end|>): the tokens
    the model produced, where prompt log-probs are trained. User/tool positions carry ~40% special-token mass and
    KL 0.5-1.0 between passes on every stack, HF included (P0, 21 Oct)."""
    out, j = [], 0
    while j < len(ids) - 1:
        if ids[j] == IM_START and ids[j + 1] == ASSISTANT:
            k = j + 3
            while k < len(ids):
                out.append(k)
                if ids[k] == IM_END:
                    break
                k += 1
            j = k
        j += 1
    return out


def assistant_prompts(path, n, lo, hi):
    """kvoff_exact.prompts, each with the assistant reply that followed its cut appended (the reply is scored)."""
    games = json.load(open(path))["games"]
    out = []
    for i, p in enumerate(prompts(path, n, lo=lo, hi=hi)):
        g = games[i % len(games)]
        nxt = g["messages"][len(p["messages"])] if len(p["messages"]) < len(g["messages"]) else None
        if nxt and nxt["role"] == "assistant":
            p = dict(p, messages=p["messages"] + [dict(nxt)])
        out.append(p)
    return out


def load(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def rel(x, y):
    if not x or not y or not x[0] or not y[0]:
        return None
    a, b = x[0][4], y[0][4]
    num = sum((p - q) ** 2 for p, q in zip(a, b)) ** 0.5
    den = sum(p * p for p in a) ** 0.5
    return num / den if den else None


def compare(a, b):
    ib = {(r["m"], r["k"]): r for r in b}
    first, sources, n_diff = None, collections.Counter(), 0
    src_rel = collections.defaultdict(list)
    for r in a:
        s = ib.get((r["m"], r["k"]))
        if s is None:
            continue
        if r["out"] != s["out"]:
            n_diff += 1
            if r["in"] == s["in"]:
                src_rel[r["m"]].append(rel(r["out"], s["out"]))
            if first is None:
                first = {"m": r["m"], "k": r["k"], "in_equal": r["in"] == s["in"], "out_rel": rel(r["out"], s["out"])}
            if r["in"] == s["in"]:
                sources[r["m"].split("layers.")[-1].split(".", 1)[-1] if "." in r["m"].split("layers.")[-1] else "layer"] += 1
    return {"records": len(a), "matched": sum((r["m"], r["k"]) in ib for r in a), "differ": n_diff, "first": first,
            "sources_by_block": dict(sources),
            "source_rel": {m: [round(v, 6) for v in vs if v is not None][:6] for m, vs in src_rel.items()},
            "drift_rel_by_layer": [round(v, 5) if v is not None else None for v in
                                   [rel(r["out"], ib[(r["m"], r["k"])]["out"]) for r in a
                                    if r["m"].count(".") == 2 and r["k"] == 0 and (r["m"], r["k"]) in ib]]}


BINS = [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, float("inf")]


def _hist(t):
    import torch
    t = t.float()
    return [int(((t >= lo) & (t < hi)).sum()) for lo, hi in zip(BINS[:-1], BINS[1:])]


def sel_compare(dx, dy):
    """Per layer: MoE tokens whose top-k set differs and QSA rows whose selected set differs between two passes."""
    import re
    import torch
    if not (os.path.isdir(dx) and os.path.isdir(dy)):
        return None
    moe, qsa, first = {}, {}, None
    files = sorted(set(os.listdir(dx)) & set(os.listdir(dy)))
    for f in files:
        m = re.match(r"model\.layers\.(\d+)\.(mlp\.gate|self_attn\.indexer)-(\d+)\.pt", f)
        if not m:
            continue
        layer, kind, call = int(m.group(1)), m.group(2), int(m.group(3))
        a, b = torch.load(os.path.join(dx, f)), torch.load(os.path.join(dy, f))
        if kind == "mlp.gate":
            if a["ids"].shape != b["ids"].shape:
                continue
            flip = (a["ids"] != b["ids"]).any(dim=1)
            e = moe.setdefault(layer, {"tokens": 0, "flips": 0, "margin_all": [0] * 8, "margin_flip": [0] * 8,
                                       "experts": a["experts"]})
            e["tokens"] += int(flip.numel()); e["flips"] += int(flip.sum())
            e["margin_all"] = [p + q for p, q in zip(e["margin_all"], _hist(a["margin"]))]
            e["margin_flip"] = [p + q for p, q in zip(e["margin_flip"], _hist(a["margin"][flip]))]
        else:
            if a["hash"].shape != b["hash"].shape:
                continue
            flip = (a["hash"] != b["hash"]) | (a["count"] != b["count"])
            e = qsa.setdefault(layer, {"rows": 0, "flips": 0})
            e["rows"] += int(flip.numel()); e["flips"] += int(flip.sum())
            if "rows" in a and "rows" in b and int(flip.sum()):
                ra, rb = a["rows"][flip], b["rows"][flip]
                diff = [len(set(x.tolist()) ^ set(y.tolist())) // 2 for x, y in zip(ra[:256], rb[:256])]
                e["swapped_mean"] = sum(diff) / len(diff)
                e["swapped_max"] = max(diff)
        if int(flip.sum()) and (first is None or (call, layer) < (first["call"], first["layer"])):
            first = {"call": call, "layer": layer, "kind": kind, "n": int(flip.sum()), "of": int(flip.numel())}
    return {"first_flip": first, "bins": BINS[1:-1],
            "moe": {k: moe[k] for k in sorted(moe)}, "qsa": {k: qsa[k] for k in sorted(qsa)}}


def _kl(p, q):
    import math
    floor = min(q.values())
    zp = math.log(sum(math.exp(v) for v in p.values()))
    return sum(math.exp(v - zp) * ((v - zp) - q.get(t, floor)) for t, v in p.items())


async def gencheck(client, url, model, ids, top_first, n, isolate_trial):
    """Row j of a long prefill's prompt logprobs vs the last row of a cold request for ids[:j] (sampler logprobs)."""
    base = len(ids) - len(top_first)
    special = [j for j, d in enumerate(top_first) if d and int(max(d, key=d.get)) >= 248000]
    other = [j for j, d in enumerate(top_first) if d and int(max(d, key=d.get)) < 248000]
    pick = special[:: max(1, len(special) // max(1, n // 2))][: n // 2] + other[:: max(1, len(other) // max(1, n - n // 2))][: n - n // 2]
    out = []
    for j in pick:
        await isolate_trial(url)
        r = await client.post(url + "/completions", json={"model": model, "prompt": ids[: base + j], "temperature": 0.0,
                                                          "max_tokens": 8, "logprobs": 20,
                                                          "return_tokens_as_token_ids": True})
        r.raise_for_status()
        ch = r.json()["choices"][0]
        first = (ch.get("logprobs") or {}).get("top_logprobs") or [{}]
        gen = {k.split(":")[-1]: v for k, v in (first[0] or {}).items()}
        ref = top_first[j]
        out.append({"j": j, "special": j in special, "prompt_top": max(ref, key=ref.get),
                    "prompt_top_p": round(__import__("math").exp(max(ref.values())), 3),
                    "gen_top": max(gen, key=gen.get) if gen else None,
                    "gen_top_p": round(__import__("math").exp(max(gen.values())), 3) if gen else None,
                    "kl_prompt_gen": round(_kl(ref, gen), 4) if gen else None,
                    "gen_text_ids": ((ch.get("logprobs") or {}).get("tokens") or [])[:8]})
    agree = [o["prompt_top"] == o["gen_top"] for o in out]
    return {"n": len(out), "top1_agree": sum(agree) / max(len(agree), 1),
            "kl_mean_special": sum(o["kl_prompt_gen"] or 0 for o in out if o["special"]) / max(1, sum(o["special"] for o in out)),
            "kl_mean_other": sum(o["kl_prompt_gen"] or 0 for o in out if not o["special"]) / max(1, sum(not o["special"] for o in out)),
            "rows": out}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", default="A,B,C", help="tags; a tag ending in Z/R fills the cache storages first")
    ap.add_argument("--pairs", default="A:B,B:C")
    ap.add_argument("--gencheck", type=int, default=0, help="N positions per prompt: last-row logprobs of a cold "
                    "request for ids[:j] against pass A's prompt logprob at j (half where a special token is top-1)")
    ap.add_argument("--asst", action="store_true", help="append each cut's assistant reply and keep top-K at every "
                    "assistant-turn position (all turns) instead of the last 512 positions")
    ap.add_argument("--lo", type=int, default=6000)
    ap.add_argument("--hi", type=int, default=12000)
    ap.add_argument("--topk", type=int, default=1, help="prompt_logprobs K; K > 1 also saves lp-p<i><tag>.json")
    a = ap.parse_args()
    if a.asst and a.gencheck:
        raise SystemExit("--gencheck reads the last positions; not with --asst")
    url = os.environ["ARC_BENCH_BASE_URL"]
    model = os.environ["ARC_BENCH_MODEL"]
    root = url.removesuffix("/v1")
    from arc_infra.model_templates import template_kwargs
    from arena.isolation import isolate_trial

    kwargs = template_kwargs(model, {"enable_thinking": True,
                                    "reasoning_effort": os.environ.get("ARC_BENCH_EFFORT", "xhigh")})
    ps = (assistant_prompts(a.contexts, a.prompts, a.lo, a.hi) if a.asst
          else prompts(a.contexts, a.prompts, lo=a.lo, hi=a.hi))
    os.makedirs(a.dir, exist_ok=True)  # also without the runtime hooks (plain KL reads)
    report = {"rows": []}
    async with httpx.AsyncClient(timeout=1800) as client:
        for i, p in enumerate(ps):
            r = await client.post(root + "/tokenize", json={"model": model, "messages": p["messages"], "tools": p["tools"],
                                                           "add_generation_prompt": not a.asst, "chat_template_kwargs": kwargs})
            r.raise_for_status()
            ids = r.json()["tokens"]
            sel = assistant_positions(ids) if a.asst else list(range(max(0, len(ids) - 512), len(ids)))
            with open(os.path.join(os.path.dirname(os.path.abspath(a.out)), f"ids-p{i}.json"), "w") as fh:
                json.dump({"game": p["game"], "cut": p["cut"], "ids": ids}, fh)  # full prompt ids (external controls)
            lps = {}
            tags = a.passes.split(",")
            for tag in tags:
                await isolate_trial(url)
                with open(a.dir + "/tag.tmp", "w") as fh:
                    fh.write(f"p{i}{tag}")
                os.replace(a.dir + "/tag.tmp", a.dir + "/tag")
                await asyncio.sleep(0.05)
                resp = await client.post(url + "/completions", json={"model": model, "prompt": ids, "temperature": 0.0,
                                                                     "max_tokens": 1, "prompt_logprobs": a.topk})
                resp.raise_for_status()
                os.remove(a.dir + "/tag")
                ent = resp.json()["choices"][0].get("prompt_logprobs") or []
                ent = [ent[j] if j < len(ent) else None for j in sel]
                lps[tag] = [((e or {}).get(str(ids[j])) or {}).get("logprob") for e, j in zip(ent, sel)]
                if tag == tags[0]:
                    top_first = [{k: v.get("logprob") for k, v in (e or {}).items()} for e in ent]
                if a.topk > 1:  # top-K prompt logprobs of the last 512 positions, for KL against other runs
                    with open(os.path.join(os.path.dirname(os.path.abspath(a.out)), f"lp-p{i}{tag}.json"), "w") as fh:
                        json.dump({"ids": [ids[j] for j in sel], "pos": sel,
                                   "lp": [{k: v.get("logprob") for k, v in (e or {}).items()} for e in ent]}, fh)
            rec = {t: load(f"{a.dir}/p{i}{t}.jsonl") for t in tags}
            import glob
            import shutil
            for f in glob.glob(f"{a.dir}/p{i}*.pad.jsonl"):  # CG padding checks (ARC_DIVTRACE_PAD), every prompt
                shutil.copy(f, os.path.dirname(os.path.abspath(a.out)))
            for f in glob.glob(f"{a.dir}/p{i}*.pt") + glob.glob(f"{a.dir}/p{i}*.fill.json") if i == 0 else []:
                shutil.copy(f, os.path.dirname(os.path.abspath(a.out)))
            row = {"i": i, "tokens": len(ids), "scored": len(sel), "game": p["game"], "cut": p["cut"]}
            for pair in a.pairs.split(","):
                x, y = pair.split(":")
                d = [abs(p - q) for p, q in zip(lps[x], lps[y]) if p is not None and q is not None]
                row[x + y] = compare(rec[x], rec[y])
                row[x + y]["logprob_median_abs"] = sorted(d)[len(d) // 2] if d else None
                row[x + y]["logprob_gt1"] = sum(v > 1 for v in d) / max(len(d), 1)
                try:
                    row[x + y]["sel"] = sel_compare(f"{a.dir}/sel/p{i}{x}", f"{a.dir}/sel/p{i}{y}")
                except Exception as error:  # diagnostic: never lose the row
                    row[x + y]["sel"] = {"error": repr(error)}
            first = rec[tags[0]]
            row["rerun_A"] = {r["m"]: r["rerun"] for r in first if r.get("rerun")}
            row["ops_A"] = {r["m"]: r["ops"] for r in first if r.get("ops")}
            by = collections.defaultdict(list)
            for m, v in row["rerun_A"].items():
                by[m.split(".", 3)[-1]].append(max(z["rel"] for z in v))
            row["rerun_max_by_block"] = {k: [round(max(v), 8), sum(x > 0 for x in v), len(v)] for k, v in by.items()}
            if a.gencheck:
                try:
                    row["gencheck"] = await gencheck(client, url, model, ids, top_first, a.gencheck, isolate_trial)
                except Exception as error:  # diagnostic: never lose the row
                    row["gencheck"] = {"error": repr(error)[:300]}
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
            with open(a.out, "w") as fh:
                json.dump(report, fh, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
