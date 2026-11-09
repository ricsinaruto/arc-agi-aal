"""mmrtx Stage A (1 RTX PRO 6000, Modal): the pruned NVFP4 checkpoint on stock vLLM (minimax-m31-flash build), FP8 KV +
FP8 index keys. For each server config (no spec; DSpark K=7; DSpark K=3) and each concurrency (1, 2, 5 streams at
~30K-token held-out prompts; 5 streams at ~60K): per-stream decode rate (streamed, first token -> last token), server
spec-decode counters (acceptance per draft position), GPU util / VRAM. Step time = tokens per verify step / per-stream
rate. Metrics rows for tools/ops/watch.py in /m/stagea/<run>/metrics.jsonl (progress = finished measurements).

usage: modal run --detach stage_a.py --run RUN --model minimax-m31f-p24-v1 [--configs nospec,k7,k3]"""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

import sys as _sys
_sys.path[:0] = [str(__import__('pathlib').Path(__file__).resolve().parent), '/root']
from image import IMAGE  # noqa: E402

app = modal.App("mmrtx-stagea")
models = modal.Volume.from_name("mmrtx-models")
PORT = 8000
CONFIGS = {
    "nospec": {},
    "k7": {"speculative_config": {"method": "dspark", "num_speculative_tokens": 7, "draft_sample_method": "probabilistic"}},
    "k3": {"speculative_config": {"method": "dspark", "num_speculative_tokens": 3, "draft_sample_method": "probabilistic"}},
    # gates: random weights from a config-only copy, one short request; a failure stops the run in minutes
    "gate4": {"dummy_layers": 4},
    "gatek7": {"dummy_layers": 60, "speculative_config": {"method": "dspark", "num_speculative_tokens": 7,
                                                          "draft_sample_method": "probabilistic"}},
}
BOOT_CEILING_S = 1800  # crash ceiling only
POINTS = [(1, "short"), (2, "short"), (5, "short"), (5, "long")]


def server_cmd(model, cfg, max_len):
    cmd = ["vllm", "serve", model, "--port", str(PORT), "--trust-remote-code", "--block-size", "128",
           "--kv-cache-dtype", "fp8", "--attention-config", json.dumps({"indexer_kv_dtype": "fp8"}),
           "--max-model-len", str(max_len), "--gpu-memory-utilization", "0.92", "--max-num-seqs", "16",
           "--disable-hybrid-kv-cache-manager", "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0}),
           "--enable-prefix-caching", "--served-model-name", "m"]
    if cfg.get("speculative_config"):
        cmd += ["--speculative-config", json.dumps(cfg["speculative_config"])]
    if cfg.get("dummy_layers"):
        cmd += ["--load-format", "dummy"]
    return cmd


def dummy_copy(model, layers):
    import shutil
    d = Path(f"/tmp/dummy{layers}")
    shutil.copytree(model, d, ignore=shutil.ignore_patterns("*.safetensors", "*.index.json", ".cache"), dirs_exist_ok=True)
    cfg = json.loads((d / "config.json").read_text())
    cfg["text_config"]["num_hidden_layers"] = layers
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


def scrape(url=f"http://127.0.0.1:{PORT}/metrics"):
    import urllib.request
    text = urllib.request.urlopen(url, timeout=10).read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or "spec_decode" not in line and "generation_tokens_total" not in line:
            continue
        name, _, val = line.rpartition(" ")
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return out


def stream_one(prompt_ids, max_tokens, res, i):
    import http.client
    body = json.dumps({"model": "m", "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95,
                       "ignore_eos": True, "stream": True})
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=3600)
    t0 = time.time()
    c.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    first = last = None
    n = 0
    for raw in r:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        txt = (d.get("choices") or [{}])[0].get("text") or ""
        if txt:
            now = time.time()
            first = first or now
            last = now
            n += 1  # chunks: one per engine step output for this request
    usage_tokens = max_tokens
    res[i] = {"ttft": (first or t0) - t0, "decode_s": (last - first) if first and last else None, "chunks": n,
              "tokens": usage_tokens, "rate": (usage_tokens - 1) / (last - first) if first and last and last > first else None}


def measure(prompts, conc, max_tokens):
    before = scrape()
    res = [None] * conc
    th = [threading.Thread(target=stream_one, args=(prompts[i], max_tokens, res, i)) for i in range(conc)]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    after = scrape()
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    rates = [r["rate"] for r in res if r and r["rate"]]
    out = {"streams": conc, "wall": round(time.time() - t0, 1), "per_stream_tok_s": [round(x, 2) for x in rates],
           "agg_decode_tok_s": round(sum(rates), 1), "ttft": [round(r["ttft"], 1) for r in res if r],
           "chunks_per_req": [r["chunks"] for r in res if r]}
    drafts = sum(v for k, v in delta.items() if "num_drafts" in k)
    acc = sum(v for k, v in delta.items() if "num_accepted_tokens_total" in k or k.startswith("vllm:spec_decode_num_accepted_tokens{") or k == "vllm:spec_decode_num_accepted_tokens_total")
    per_pos = sorted(((k, v) for k, v in delta.items() if "per_pos" in k), key=lambda kv: kv[0])
    if drafts:
        out["tau"] = round(1 + acc / drafts, 3)
        out["per_pos"] = {k.split("position=")[-1].strip('"}') if "position=" in k else k: round(v / drafts, 3) for k, v in per_pos}
    if rates:
        tau = out.get("tau", 1.0)
        out["step_ms"] = round(1000 * tau / (sum(rates) / len(rates)), 1)
    out["delta"] = {k: v for k, v in delta.items() if v}
    return out


@app.function(image=IMAGE, gpu="RTX-PRO-6000", timeout=3 * 3600, retries=0, volumes={"/m": models})
def bench(run: str, model: str, configs: str = "gate4,gatek7,nospec,k7,k3", sets: str = "sets-v2", max_tokens: int = 512):
    import gzip
    import urllib.request
    out = Path("/m/stagea") / run
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for name in ("held3", "held2") for line in gzip.open(f"/m/{sets}/{name}.jsonl.gz", "rt")]
    short = [r["tokens"][:s] for r in rows for s, _ in r["spans"][-1:] if 20000 <= s <= 32000][:5]
    fits = [json.loads(line) for line in gzip.open(f"/m/{sets}/fit3.jsonl.gz", "rt")]
    long = [r["tokens"][:60000] for r in fits if len(r["tokens"]) >= 60000][:5]
    assert len(short) >= 5 and len(long) >= 5, (len(short), len(long))
    state = {"progress": 0, "stop": False}
    results = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}

    def sampler():
        while not state["stop"]:
            m = {"stagea.progress": state["progress"]}
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=20).stdout.strip().split(",")
                m["infra.gpu_util.0"], m["infra.vram_gib.0"] = float(q[0]), round(float(q[1]) / 1024, 2)
            except Exception:  # noqa: BLE001
                pass
            with open(out / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"t": time.time(), "m": m}) + "\n")
            try:
                models.commit()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(30)

    threading.Thread(target=sampler, daemon=True).start()
    for name in configs.split(","):
        if name in results:
            continue
        log = open(out / f"server-{name}.log", "w")
        cfg = CONFIGS[name]
        path = dummy_copy(f"/m/{model}", cfg["dummy_layers"]) if cfg.get("dummy_layers") else f"/m/{model}"
        env = {**os.environ, "VLLM_DEEP_GEMM_WARMUP": "skip"} if cfg.get("dummy_layers") else None
        proc = subprocess.Popen(server_cmd(path, cfg, 70000), stdout=log, stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                models.commit()
                raise RuntimeError(f"server {name} exited {proc.returncode}; see server-{name}.log")
            if time.time() - t0 > BOOT_CEILING_S:
                proc.kill()
                models.commit()
                raise RuntimeError(f"server {name} not healthy after {BOOT_CEILING_S} s; see server-{name}.log")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
                break
            except OSError:
                time.sleep(10)
        boot = round(time.time() - t0)
        state["progress"] += 1
        res = {"boot_s": boot, "points": {}}
        for conc, kind in ([(1, "short")] if cfg.get("dummy_layers") else POINTS):
            prompts = short if kind == "short" else long
            if cfg.get("dummy_layers"):
                res["points"]["gate"] = measure(prompts, 1, 16)
                print(f"GATE OK {name} " + json.dumps(res["points"]["gate"])[:300], flush=True)
                break
            measure(prompts, conc, 16)  # warm the prefix cache so the timed run measures decode
            res["points"][f"{conc}x{kind}"] = measure(prompts, conc, max_tokens)
            state["progress"] += 1
            print(name, conc, kind, json.dumps(res["points"][f"{conc}x{kind}"])[:500], flush=True)
        res["server_log_tail"] = Path(out / f"server-{name}.log").read_text()[-4000:]
        proc.terminate()
        try:
            proc.wait(120)
        except subprocess.TimeoutExpired:
            proc.kill()
        results[name] = res
        (out / "results.json").write_text(json.dumps(results, indent=1))
        models.commit()
    state["stop"] = True
    print("RUN COMPLETE " + json.dumps({k: {p: v.get("step_ms") for p, v in r["points"].items()} for k, r in results.items()}))
    models.commit()
    return results


@app.local_entrypoint()
def main(run: str, model: str, configs: str = "gate4,gatek7,nospec,k7,k3"):
    call = bench.spawn(run, model, configs)
    print(json.dumps({"call": call.object_id, "app_id": app.app_id}))
