"""mmrtx task read (Nebius image kind, 1 RTX PRO 6000): vLLM serves a candidate checkpoint; the arc2 harness
(mm-tool-early overrides, text grids) plays a hard38 slice through arc2.api_proxy pointed at the local server
(OpenRouter-only fields dropped, chat_template_kwargs / reasoning_content kept, tool-call history repaired as on the
API path). Stops at a completion-token mark or a wall ceiling; scores right outputs at the mark like the minimax
arm (candidates with clock <= mark). Outputs in $ARC_OUT: result.json, metrics.jsonl, vllm.log, the harness files.

usage (inside the image): python3 taskread.py --model cand-drop-k24 --limit 10 --mark 1188000 --wall 6600 [--spec 3]"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("MMRTX_ROOT", "/w/mmrtx-models"))
OUT = Path(os.environ.get("ARC_OUT", "/out"))
PORT, PROXY_PORT = 8000, 8001
OVERRIDES = {  # minimax arm mm-tool-early (origin/minimax scripts/arc2_minimax/mm-tool-early.json), images off,
    # window cut to the RTX's 131,072 (API: 1,048,576)
    "study": {"temperature": 1.0, "windows": [131072], "completion_cap": 98304},
    "arc2": {"prefill": None, "gate_match": None}, "stop_rule": False, "repair_tool_args": True,
    "replace_arm": {"name": "tool-mm-early", "tool": True, "effort": "low", "cap": 98304, "augment": 4,
                    "confirm_attempts": 1, "images": False, "image_side": 224, "max_attempts": 0, "review": True,
                    "review_style": "second", "review_tools": False, "review_finish": True, "early_tool": True}}


def log(msg):
    print(f"[taskread {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def prom(text, name):
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name) and not line.startswith("#"):
            try:
                total += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return total


def serve_cmd(model, slots, max_len, spec):
    cmd = ["vllm", "serve", str(ROOT / model), "--served-model-name", "m", "--host", "127.0.0.1", "--port", str(PORT),
           "--trust-remote-code", "--block-size", "128", "--kv-cache-dtype", "fp8",
           "--attention-config", json.dumps({"indexer_kv_dtype": "fp8"}), "--max-model-len", str(max_len),
           "--gpu-memory-utilization", os.environ.get("MMRTX_UTIL", "0.92"), "--max-num-seqs", str(slots + 2), "--disable-hybrid-kv-cache-manager",
           "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0}), "--enable-prefix-caching",
           "--enable-auto-tool-choice", "--tool-call-parser", "minimax_m3", "--reasoning-parser", "minimax_m3"]
    if os.environ.get("MMRTX_MOE_BACKEND"):  # e.g. humming (with MMRTX_MOE_A16=1 + VLLM_HUMMING_ONLINE_QUANT_CONFIG)
        cmd += ["--moe-backend", os.environ["MMRTX_MOE_BACKEND"]]
    if spec:
        cmd += ["--speculative-config", json.dumps({"method": "dspark", "num_speculative_tokens": int(spec),
                                                    "draft_sample_method": "probabilistic"})]
    return cmd


def start_server(model, slots, max_len, spec, ceiling=1800):
    """(process, spec used): a server that fails with DSpark is restarted once without it."""
    if os.environ.get("TASKREAD_FAKE_SERVER"):  # local test: a stand-in server already listens on PORT
        (OUT / "vllm.log").write_text("GPU KV cache size: 400,000 tokens\n")
        return types.SimpleNamespace(terminate=lambda: None), 0
    for use in ([spec, 0] if spec else [0]):
        logf = open(OUT / "vllm.log", "a")
        logf.write(f"\n=== serve spec={use} ===\n")
        logf.flush()
        proc = subprocess.Popen(serve_cmd(model, slots, max_len, use), stdout=logf, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < ceiling:
            if proc.poll() is not None:
                log(f"server exited {proc.returncode} (spec={use})")
                break
            try:
                get(f"http://127.0.0.1:{PORT}/health")
                log(f"server healthy in {time.time() - t0:.0f} s (spec={use})")
                return proc, use
            except OSError:
                time.sleep(10)
        else:
            proc.kill()
            log(f"server not healthy after {ceiling} s (spec={use})")
    raise SystemExit("no healthy server")


def score(out, order, mark, bundle):
    """Right outputs at the mark (candidates with clock <= mark), as the minimax arm counts them."""
    sys.path[:0] = [str(bundle / "src")]
    from arc2.loop import guesses
    rows = {}
    path = out / "tasks.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("{"):
                r = json.loads(line)
                rows[r["task"].split("@")[0].split("#")[0]] = r
    per, outputs = {}, 0
    for t in order:
        task = json.loads((bundle / "data/arc2/evaluation" / f"{t}.json").read_text())
        truth = [x["output"] for x in task["test"]]
        outputs += len(truth)
        r = rows.get(t)
        cands = [c for c in (r or {}).get("candidates", []) if (c.get("clock") if c.get("clock") is not None else 1e18) <= mark]
        picks = guesses(cands, [x["input"] for x in task["test"]]) if cands else [[] for _ in truth]
        per[t] = sum(any(g == tr for g, _ in picks[k]) for k, tr in enumerate(truth))
    return {"right": sum(per.values()), "outputs": outputs, "per_task": per}


def hum_schema(spec):
    """'uint3:32' -> the no-zero-point Humming weight schema hum_build.py writes."""
    dtype, group = spec.split(":")
    return {"quant_method": "humming", "b_dtype": dtype, "group_size": int(group), "has_zero_point": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bundle", default="arc2-bundle-v1")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--mark", type=int, default=1188000)
    ap.add_argument("--wall", type=int, default=6600)
    ap.add_argument("--slots", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=131072)
    ap.add_argument("--spec", type=int, default=0)
    ap.add_argument("--cap", type=int, default=0, help="reply cap (default 98,304: the minimax arm's)")
    ap.add_argument("--util", default="", help="vLLM gpu-memory-utilization (default 0.92)")
    ap.add_argument("--hum-layers", default="", help="comma list of Humming layers in a mixed model")
    ap.add_argument("--humming", default="", help="DTYPE:GROUP (e.g. uint3:32) of the model's hum_build experts")
    a = ap.parse_args()
    if a.util:
        os.environ["MMRTX_UTIL"] = a.util
    if a.humming:  # mmrtx_patch/sitecustomize.py: routed experts stored in this Humming schema (hum_build.py), served as is
        os.environ.update(PYTHONPATH=str(Path(__file__).resolve().parent / "mmrtx_patch") + os.pathsep + os.environ.get("PYTHONPATH", ""),
                          MMRTX_MOE_HUMMING=json.dumps(hum_schema(a.humming)))
        if a.hum_layers:  # mixed model (hum_build.py variant): only these layers are Humming
            os.environ["MMRTX_MOE_HUMMING_LAYERS"] = a.hum_layers
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.time()
    try:
        import scipy  # noqa: F401 - the harness needs it; the vLLM image lacks it
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scipy==1.15.3"], check=False)
    bundle = Path(os.environ.get("TASKREAD_WORK", "/tmp")) / "bundle"
    shutil.copytree(ROOT / a.bundle, bundle, dirs_exist_ok=True)
    os.environ["TASKREAD_BUNDLE"] = str(bundle)
    state = {"t0": time.time(), "gen": 0.0, "stop": False}
    halt = threading.Event()

    def sampler():
        last_t, last_gen = time.time(), 0.0
        while not halt.wait(30):
            m = {}
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=20).stdout.strip().split(",")
                m["infra.gpu_util.0"], m["infra.vram_gib.0"] = float(q[0]), round(float(q[1]) / 1024, 2)
            except Exception:  # noqa: BLE001
                pass
            try:
                text = get(f"http://127.0.0.1:{PORT}/metrics")
                gen = prom(text, "vllm:generation_tokens_total")
                now = time.time()
                m["tr.tok_s"] = round((gen - last_gen) / max(now - last_t, 1), 1)
                last_t, last_gen = now, gen
                m["tr.running"] = prom(text, "vllm:num_requests_running")
                m["tr.waiting"] = prom(text, "vllm:num_requests_waiting")
                m["tr.accepted"] = prom(text, "vllm:spec_decode_num_accepted_tokens_total")
                m["tr.drafts"] = prom(text, "vllm:spec_decode_num_drafts_total")
            except Exception:  # noqa: BLE001
                pass
            try:
                state["gen"] = prom(get(f"http://127.0.0.1:{PROXY_PORT}/metrics"), "vllm:generation_tokens_total")
            except Exception:  # noqa: BLE001
                pass
            m["tr.progress"] = round(state["gen"] / 1000)
            with open(OUT / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"t": time.time(), "m": m}) + "\n")
    threading.Thread(target=sampler, daemon=True).start()

    proc, spec_used = start_server(a.model, a.slots, a.max_len, a.spec)
    sys.path[:0] = [str(bundle / "src"), str(Path(__file__).resolve().parent)]
    from arc2 import api_proxy
    from arena.stock_serving import layout_line
    from arena.early_stop import stop_paths
    import taskread_harness as H
    api_proxy.UPSTREAM = f"http://127.0.0.1:{PORT}/v1"
    api_proxy.MAX_LEN, api_proxy.TIMEOUT = a.max_len, 3600
    repairs = (OUT / "repairs.jsonl").open("a")
    lock = threading.Lock()

    def note(how, raw):
        with lock:
            repairs.write(json.dumps({"t": time.time(), "how": how, "raw": str(raw)[:4000]}) + "\n")
            repairs.flush()

    def local_body(body, target, keep_reasoning=True):
        out = {k: v for k, v in body.items() if k != "model"}
        out["model"] = "m"
        out["messages"] = H.repair_history(out.get("messages") or [], note)
        return out
    api_proxy.upstream_body = local_body
    threading.Thread(target=api_proxy.serve_forever, args=(PROXY_PORT, "m", "local"),
                     kwargs={"log_path": str(OUT / "proxy.jsonl")}, daemon=True).start()
    for _ in range(100):
        try:
            get(f"http://127.0.0.1:{PROXY_PORT}/health", 1)
            break
        except OSError:
            time.sleep(0.1)
    overrides = json.loads(json.dumps(OVERRIDES))
    if a.cap:  # per-reply think budget: a reply cut here gets the harness's TRUNCATED nudge and the attempt goes on
        overrides["replace_arm"]["cap"] = overrides["study"]["completion_cap"] = int(a.cap)
    overrides["trial"] = {"marks": {"slice": a.mark}, "curve_step": 95000, "tiers": [1000000000], "copies": 1}
    stop_file = stop_paths(OUT, {})[0]

    def stopper():  # the mark (2% over, so the last replies before it finish) or the wall ceiling
        while not halt.wait(15):
            over = state["gen"] >= a.mark * 1.02
            late = time.time() - began > a.wall
            if (over or late) and not stop_file.exists():
                stop_file.write_text(json.dumps({"reason": "mark" if over else "wall", "gen": state["gen"]}))
                log(f"stop: {'mark' if over else 'wall'} at {state['gen']:.0f} tokens")
    threading.Thread(target=stopper, daemon=True).start()
    report = H.screen_driver(OUT, f"http://127.0.0.1:{PROXY_PORT}/v1", layout_line(OUT / "vllm.log", a.max_len),
                             a.slots, a.wall, a.limit, images=False, overrides=overrides)
    halt.set()
    study = json.loads((bundle / "configs/comparison.json").read_text())
    order = study["trials"][0]["tasks"][:a.limit]
    res = score(OUT, order, a.mark, bundle)
    replies = sum(1 for line in open(OUT / "proxy.jsonl")) if (OUT / "proxy.jsonl").exists() else 0
    nrep = sum(1 for _ in open(OUT / "repairs.jsonl"))
    res.update(model=a.model, spec=spec_used, generated=state["gen"], seconds=round(time.time() - began),
               replies=replies, repaired_tool_args=nrep, stop=json.loads(stop_file.read_text()) if stop_file.exists() else None,
               report_status=report.get("status"))
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    log("RESULT " + json.dumps({k: v for k, v in res.items() if k != "per_task"}))
    proc.terminate()
    print("RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
