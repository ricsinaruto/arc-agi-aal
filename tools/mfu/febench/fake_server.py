"""CPU preflight stand-in for vLLM serve: /v1/models, /v1/load_lora_adapter, /v1/chat/completions (usage, token ids,
prompt logprobs), /metrics. Accepts the serve flags the bench adds (--api-server-count, --lora-modules)."""
import asyncio
import json
import random
import sys
import time

from aiohttp import web

argv = sys.argv
served = argv[argv.index("--served-model-name") + 1]
models = {served}
if "--lora-modules" in argv:
    models.add(json.loads(argv[argv.index("--lora-modules") + 1])["name"])
C = {"gen": 0, "prompt": 0, "n": 0, "ttft": 0.0, "q": 0.0, "pf": 0.0, "dec": 0.0, "e2e": 0.0, "running": 0,
     "pq": 0, "ph": 0}
SEEN = set()  # prefix cache stand-in: prompts seen since the last reset (sleep clears it unless ARC_SLEEP_KEEP_KV=1)


async def sleep_h(req):
    import os
    if os.environ.get("ARC_SLEEP_KEEP_KV") != "1":
        SEEN.clear()
    await asyncio.sleep(0.2)
    return web.Response(text="")


async def wake_h(_):
    await asyncio.sleep(0.2)
    return web.Response(text="")


async def unload_h(req):
    body = await req.json()
    models.discard(body["lora_name"])
    return web.Response(text="ok")


async def models_h(_):
    return web.json_response({"data": [{"id": m} for m in sorted(models)]})


async def load_h(req):
    body = await req.json()
    models.add(body["lora_name"])
    return web.Response(text="ok")


async def chat_h(req):
    body = await req.json()
    if body["model"] not in models:
        return web.json_response({"error": "no model"}, status=404)
    t0 = time.time()
    pt = max(1, len(json.dumps(body["messages"])) // 4)
    key = json.dumps(body["messages"])[:-200]  # the shared prefix (the probe appends a short suffix on resends)
    C["pq"] += pt
    C["ph"] += int(pt * 0.95) if key in SEEN else 0
    SEEN.add(key)
    ct = body.get("max_tokens") or 16
    C["running"] += 1
    await asyncio.sleep(0.05 + 0.0004 * ct)
    C["running"] -= 1
    C["gen"] += ct
    C["prompt"] += pt
    C["n"] += 1
    dt = time.time() - t0
    C["ttft"] += dt / 3
    C["q"] += dt / 10
    C["pf"] += dt / 10
    C["dec"] += dt * 2 / 3
    C["e2e"] += dt
    rnd = random.Random(pt)
    d = {"id": "chatcmpl-x", "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "length",
                                          "token_ids": [rnd.randrange(1000) for _ in range(ct)]}],
         "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
                   "prompt_tokens_details": {"cached_tokens": int(pt * 0.9)}}}
    if body.get("return_token_ids"):
        d["prompt_token_ids"] = [rnd.randrange(1000) for _ in range(min(pt, 2000))]
    if body.get("logprobs"):
        ids = d["choices"][0]["token_ids"]
        d["choices"][0]["logprobs"] = {"content": [
            {"token": f"token_id:{t}", "logprob": -0.5, "bytes": [],
             "top_logprobs": [{"token": f"token_id:{t + k}", "logprob": -0.5 - k, "bytes": []} for k in range(20)]}
            for t in ids]}
    if body.get("prompt_logprobs") is not None:
        d["prompt_logprobs"] = [None] + [{str(i): {"logprob": -rnd.random()}} for i in range(min(pt, 3000))]
    return web.json_response(d)


async def reset_h(_):
    return web.Response(text="ok")


async def metrics_h(_):
    lines = [f'vllm:generation_tokens_total{{model_name="m"}} {C["gen"]}',
             f'vllm:prompt_tokens_total{{model_name="m"}} {C["prompt"]}',
             f'vllm:num_requests_running{{model_name="m"}} {C["running"]}',
             f'vllm:num_requests_waiting{{model_name="m"}} 0',
             f'vllm:prefix_cache_queries_total{{model_name="m"}} {C["pq"]}',
             f'vllm:prefix_cache_hits_total{{model_name="m"}} {C["ph"]}']
    for name, key in (("time_to_first_token_seconds", "ttft"), ("request_queue_time_seconds", "q"),
                      ("request_prefill_time_seconds", "pf"), ("request_decode_time_seconds", "dec"),
                      ("e2e_request_latency_seconds", "e2e")):
        lines += [f'vllm:{name}_sum{{model_name="m"}} {C[key]}', f'vllm:{name}_count{{model_name="m"}} {C["n"]}',
                  f'vllm:{name}_bucket{{le="1"}} 0']
    lines += [f'vllm:request_generation_tokens_sum{{model_name="m"}} {C["gen"]}']
    return web.Response(text="\n".join(lines) + "\n")


async def start_prof_h(_):
    return web.Response(text="")


async def stop_prof_h(_):  # a tiny kineto-like trace in the --profiler-config dir
    import gzip, os
    if "--profiler-config" in argv:
        d = json.loads(argv[argv.index("--profiler-config") + 1])["torch_profiler_dir"]
        os.makedirs(d, exist_ok=True)
        ev = []
        for st in range(4):  # 4 annotated steps: a graph launch and two eager kernels each
            t0 = st * 1000
            ev.append({"ph": "X", "cat": "user_annotation", "name": f"execute_context_{st % 2}({st * 7})_generation_60(60)",
                       "ts": t0, "dur": 300})
            for j, (n, dur) in enumerate([("fused_moe_lora_kernel", 50), ("_lora_shrink_kernel", 20),
                                          ("marlin_moe_wna16", 90)]):
                corr = st * 10 + j
                ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch" if j == 0 else "cudaLaunchKernel",
                           "ts": t0 + 10 + j, "dur": 5, "args": {"correlation": corr}})
                ev.append({"ph": "X", "cat": "kernel", "name": n, "ts": t0 + 400 + j * 100, "dur": dur,
                           "args": {"correlation": corr}})
        with gzip.open(os.path.join(d, "fake_worker.pt.trace.json.gz"), "wt") as h:
            json.dump({"traceEvents": ev}, h)
    return web.Response(text="")


app = web.Application(client_max_size=64 << 20)
app.add_routes([web.get("/v1/models", models_h), web.post("/v1/load_lora_adapter", load_h),
                web.post("/v1/chat/completions", chat_h), web.get("/metrics", metrics_h),
                web.post("/reset_prefix_cache", reset_h), web.post("/start_profile", start_prof_h),
                web.post("/stop_profile", stop_prof_h), web.post("/sleep", sleep_h), web.post("/wake_up", wake_h),
                web.post("/v1/unload_lora_adapter", unload_h)])
web.run_app(app, host="127.0.0.1", port=1234, print=None)
