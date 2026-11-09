"""GDN-state capture driver (throughput arm, 23 Oct; gpu-script beside a --gdn-dump server).

Sends N c100 contexts of at least --min tokens (different games) one at a time, prefix cache reset before each, as
completions with max_tokens 1, so each prefill starts from a zero GDN state and arc_gdn_dump saves its chunks.
Writes the prompts and the dump's file list/sizes to --out.
"""
import argparse
import asyncio
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pick(path, n, lo):
    data = json.load(open(path))
    out, seen = [], set()
    for g in data["games"]:
        fam = g["game"].split("-")[0]
        if fam in seen:
            continue
        k = max([c for c, t in zip(g["cuts"], g["cut_tokens"]) if t >= lo] or [0])
        if not k:
            continue
        msgs = [dict(m) for m in g["messages"][:k]]
        while msgs and msgs[-1]["role"] == "assistant":
            msgs.pop()
        out.append({"game": g["game"], "cut": k, "messages": msgs, "tools": g["tools"]})
        seen.add(fam)
        if len(out) == n:
            break
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--min", type=int, default=50000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    url = os.environ["ARC_BENCH_BASE_URL"]
    model = os.environ["ARC_BENCH_MODEL"]
    root = url.removesuffix("/v1")
    from arc_infra.model_templates import template_kwargs
    from arena.isolation import isolate_trial

    kwargs = template_kwargs(model, {"enable_thinking": True, "reasoning_effort": os.environ.get("ARC_BENCH_EFFORT", "xhigh")})
    ps = pick(a.contexts, a.n, a.min)
    report = {"prompts": [], "dump": {}}
    open(os.environ.get("ARC_GDN_DUMP_ARMED", "/tmp/arc_gdn_dump_armed"), "w").close()
    async with httpx.AsyncClient(timeout=3600) as client:
        for p in ps:
            r = await client.post(root + "/tokenize", json={"model": model, "messages": p["messages"], "tools": p["tools"],
                                                           "add_generation_prompt": True, "chat_template_kwargs": kwargs})
            r.raise_for_status()
            ids = r.json()["tokens"]
            await isolate_trial(url)
            t = time.time()
            resp = await client.post(url + "/completions", json={"model": model, "prompt": ids, "temperature": 0.0,
                                                                  "max_tokens": 1, "seed": 0})
            if resp.status_code != 200:
                raise RuntimeError(f"completions {resp.status_code}: {resp.text[:300]}")
            report["prompts"].append({"game": p["game"], "cut": p["cut"], "tokens": len(ids),
                                      "seconds": round(time.time() - t, 1)})
            print(json.dumps(report["prompts"][-1]), flush=True)
    d = os.environ.get("ARC_GDN_DUMP_DIR") or "/tmp/arena/trials/duck-transfer/gdn-dump"
    for layer in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        files = sorted(os.listdir(os.path.join(d, layer)))
        report["dump"][layer] = {"files": len(files),
                                 "bytes": sum(os.path.getsize(os.path.join(d, layer, f)) for f in files)}
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)
    if not report["dump"]:
        raise SystemExit("no GDN dump files")


if __name__ == "__main__":
    asyncio.run(main())
