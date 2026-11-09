"""Humming MoE tile sweep driver (gpu-script; throughput item 3, 22 Oct). Arms the worker hook (htune_runtime, ARC_HTUNE),
sends one real game prompt so the hook records every MoE layer's routing, then waits for the worker's sweep and copies
its rows next to --out. Summary: served vs best config per GEMM at the tuning M, bit-equal layers, transfer to other M."""
import argparse
import asyncio
import json
import os
import shutil
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvoff_exact import prompts  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--dir", default="/tmp/arc_htune")
    ap.add_argument("--wait", type=float, default=3300)
    ap.add_argument("--lo", type=int, default=3000)
    ap.add_argument("--hi", type=int, default=8000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    url = os.environ["ARC_BENCH_BASE_URL"]
    model = os.environ["ARC_BENCH_MODEL"]
    root = url.removesuffix("/v1")
    from arc_infra.model_templates import template_kwargs

    kwargs = template_kwargs(model, {"enable_thinking": True, "reasoning_effort": os.environ.get("ARC_BENCH_EFFORT", "xhigh")})
    p = prompts(a.contexts, 1, lo=a.lo, hi=a.hi)[0]
    os.makedirs(a.dir, exist_ok=True)
    report = {"game": p["game"], "cut": p["cut"]}
    async with httpx.AsyncClient(timeout=1800) as client:
        r = await client.post(root + "/tokenize", json={"model": model, "messages": p["messages"], "tools": p["tools"],
                                                       "add_generation_prompt": True, "chat_template_kwargs": kwargs})
        r.raise_for_status()
        ids = r.json()["tokens"]
        report["tokens"] = len(ids)
        open(a.dir + "/go", "w").close()
        resp = await client.post(url + "/completions", json={"model": model, "prompt": ids, "temperature": 0.0, "max_tokens": 1})
        resp.raise_for_status()
    open(a.dir + "/run", "w").close()
    t0 = time.time()
    while not os.path.exists(a.dir + "/done") and time.time() - t0 < a.wait:
        await asyncio.sleep(10)
    report["done"] = os.path.exists(a.dir + "/done")
    report["waited_s"] = round(time.time() - t0)
    rows = []
    if os.path.exists(a.dir + "/htune.jsonl"):
        shutil.copy(a.dir + "/htune.jsonl", os.path.dirname(os.path.abspath(a.out)))
        rows = [json.loads(line) for line in open(a.dir + "/htune.jsonl") if line.strip()]
    report["base"] = [r for r in rows if r.get("event") == "base"]
    report["best"] = [r for r in rows if r.get("event") == "best"]
    report["transfer"] = [r for r in rows if r.get("event") == "transfer"]
    report["errors"] = [r for r in rows if r.get("event") in ("error", "fatal", "wrong")][:20]
    report["n_cand"] = sum(r.get("event") == "cand" for r in rows)
    print(json.dumps({k: report[k] for k in ("tokens", "done", "waited_s", "n_cand")}), flush=True)
    for r in report["base"] + report["best"] + report["transfer"]:
        print(json.dumps({k: v for k, v in r.items() if k != "cfg"}), flush=True)
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)
    if not report["done"] or any(r.get("event") == "fatal" for r in rows):
        raise SystemExit("htune: sweep incomplete")


if __name__ == "__main__":
    asyncio.run(main())
