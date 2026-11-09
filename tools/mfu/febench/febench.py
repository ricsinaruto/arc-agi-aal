"""MFU-2 front-end serving A/B (RL ask 22 Oct): v6x's server-1 (bundle v6x2hm settings) with the r11 adapter under a
closed-loop replay of R12's recorded requests, one container, arms in sequence:
  A  today's server (arena.serving, adapter loaded at runtime as the loop does);
  B  the same command + --api-server-count 2, adapter static (--lora-modules; runtime LoRA updating is refused there);
  A2 today's command relaunched (exactness floor and drift).
Per arm: adapter visible from every front end, prompt logprobs (+greedy ids) on 20 recorded prompts, then (A, B) a
warm-up and a measured window with engine running/waiting (scheduler sampler), vLLM metrics, per-process CPU and
py-spy profiles of the API server(s) and EngineCore. Results: arc-prolong-runs:/validation/mfu2-fe/<run>/.

    ARC_RL1_BUNDLE=<v6x2hm> modal run --detach febench.py --run <run> [--arms A,B,A2] [--cpu]
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
WT = Path(os.environ.get("FE_WT") or (Path(__file__).resolve().parent / "wt"))  # serving_fast overlays (mfu2-boot branch files)
BOOT_WT = os.environ.get("FE_BOOT_WT")  # boot mode: modal_rl1 and src/ from this worktree (mfu2-boot)
RLA = "./.claude/worktrees/rl-async"
PINNED = "~/arc-shared/pinned-vllm"
if modal.is_local():
    spec = importlib.util.spec_from_file_location(
        "modal_rl1", f"{BOOT_WT}/scripts/posttrain/modal_rl1.py" if BOOT_WT
        else "./.claude/worktrees/rl/scripts/posttrain/modal_rl1.py")
    M = importlib.util.module_from_spec(spec)
    sys.modules["modal_rl1"] = M
    spec.loader.exec_module(M)
    image = (M.image.add_local_dir(str(HERE / "site"), "/opt/fe-site", copy=True)
             .add_local_file(str(HERE / "driver.py"), "/opt/fe/driver.py", copy=True)
             .add_local_file(str(HERE / "fake_server.py"), "/opt/fe/fake_server.py", copy=True)
             .add_local_file(str(WT / "serving_fast_patch.py"), "/opt/fe/serving_fast_patch.py", copy=True)
             .add_local_file(str(WT / "image_cache_runtime.py"), "/opt/fe/image_cache_runtime.py", copy=True)
             .add_local_file(str(WT / "lora_fused_runtime.py"), "/opt/fe/lora_fused_runtime.py", copy=True)
             .add_local_file(str(WT / "moe_lora_runtime.py"), "/opt/fe/moe_lora_runtime.py", copy=True)
             .add_local_file(str(WT / "fast_load_runtime.py"), "/opt/fe/fast_load_runtime.py", copy=True)
             .add_local_file(f"{RLA}/src/arc_infra/lora_segment_patch.py", "/opt/fe/lora_segment_patch.py", copy=True)
             .add_local_file(f"{RLA}/src/arc_infra/lora_dense_decode_patch.py", "/opt/fe/lora_dense_decode_patch.py",
                             copy=True)
             .add_local_file(f"{RLA}/src/arc_infra/lora_dense_runtime.py", "/opt/fe/lora_dense_runtime.py", copy=True)
             .add_local_file(str(HERE.parents[2] / "src/arc_infra/sleep_keep_kv_patch.py"), "/opt/fe/sleep_keep_kv_patch.py",
                             copy=True)  # this checkout's copy (trunk/rl)
             .add_local_file(f"{PINNED}/vllm/device_allocator/sleep_mode_backend.py",
                             "/opt/fe/pinned/vllm/device_allocator/sleep_mode_backend.py", copy=True)
             .add_local_file(f"{PINNED}/vllm/v1/engine/core.py", "/opt/fe/pinned/vllm/v1/engine/core.py", copy=True)
             .add_local_file(f"{PINNED}/vllm/device_allocator/cumem.py",
                             "/opt/fe/pinned/vllm/device_allocator/cumem.py", copy=True)
             .add_local_file(f"{PINNED}/vllm/v1/worker/gpu_worker.py", "/opt/fe/pinned/vllm/v1/worker/gpu_worker.py",
                             copy=True)
             .add_local_file(f"{RLA}/tests/test_lora_segment_patch.py", "/opt/fe/test_lora_segment_patch.py",
                             copy=True)
             .add_local_file(str(HERE / "shim/pytest.py"), "/opt/fe/pytest_shim.py", copy=True)
             .add_local_file(f"{PINNED}/vllm/lora/punica_wrapper/punica_gpu.py",
                             "/opt/fe/pinned/vllm/lora/punica_wrapper/punica_gpu.py", copy=True)
             .add_local_file(f"{PINNED}/vllm/lora/ops/triton_ops/lora_shrink_op.py",
                             "/opt/fe/pinned/vllm/lora/ops/triton_ops/lora_shrink_op.py", copy=True)
             .add_local_file(f"{PINNED}/vllm/lora/ops/triton_ops/lora_expand_op.py",
                             "/opt/fe/pinned/vllm/lora/ops/triton_ops/lora_expand_op.py", copy=True)
             .add_local_file(str(HERE.parents[2] / "src/arc_infra/boot_cache.py"), "/opt/fe/boot_cache_new.py",
                             copy=True)  # this checkout's copy (trunk/rl): J arms' carry
             .add_local_file(str(HERE / "moel/moe_lora_fast_patch.py"), "/opt/fe/moe_lora_fast_patch.py", copy=True)
             .add_local_file(str(HERE / "moel/moe_lora_fast_runtime.py"), "/opt/fe/moe_lora_fast_runtime.py",
                             copy=True))
    if BOOT_WT:
        image = image.add_local_dir(f"{BOOT_WT}/src", "/opt/boot-src", copy=True,
                                    ignore=["**/__pycache__/**", "**/*.pyc"])
else:
    sys.path.insert(0, "/root")
    import modal_rl1 as M  # noqa: E402

    image = M.image

app = modal.App("arc-mfu2-febench")
DATA = "/runs/validation/mfu2-fe/data-r12"
LOAD_DATA = "/runs/validation/mfu1-fe/data-v6j"  # MFU-1 mm-cache A/B: 212 long multi-image v6j/v6j3/v6k tries
ROOT = "http://127.0.0.1:1234"
SERVED = "Qwen/Qwen3.8-Flash-Next-NVFP4"
PYSPY = "/workspace/axolotl-venv/bin/py-spy"
WORK = Path("/tmp/fe")


def proc_table():
    """pid -> (category, name, cpu ticks)."""
    rows = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            cmd = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            stat = open(f"/proc/{d}/stat").read()
            fields = stat[stat.rindex(")") + 2:].split()
            ticks = int(fields[11]) + int(fields[12])
        except (OSError, ValueError, IndexError):
            continue
        if "APIServer" in cmd:
            cat = "api"
        elif "EngineCore" in cmd:
            cat = "engine"
        elif "VLLM::Worker" in cmd or "Worker_" in cmd:
            cat = "worker"
        elif " serve " in cmd and "--served-model-name" in cmd:
            cat = "serve_main"
        elif "/opt/fe/driver.py" in cmd:
            cat = "driver"
        else:
            cat = "other"
        rows[int(d)] = (cat, cmd[:60], ticks)
    return rows


class Sampler(threading.Thread):
    """Every 5 s: per-category CPU% (100 = one core) and the vLLM /metrics lines (no buckets)."""

    def __init__(self, folder):
        super().__init__(daemon=True)
        self.folder, self.stop = folder, threading.Event()

    def run(self):
        import httpx
        prev, prev_t = proc_table(), time.time()
        cpu_f = open(self.folder / "cpu.jsonl", "a")
        met_f = open(self.folder / "metrics.jsonl", "a")
        while not self.stop.wait(5):
            now, t = proc_table(), time.time()
            dt = t - prev_t
            cats, per = {}, {}
            for pid, (cat, name, ticks) in now.items():
                if pid in prev:
                    d = (ticks - prev[pid][2]) / 100 / dt * 100
                    cats[cat] = cats.get(cat, 0) + d
                    if cat in ("api", "engine", "serve_main"):
                        per[f"{cat}:{pid}"] = round(d, 1)
            cpu_f.write(json.dumps({"t": round(t, 2), "cats": {k: round(v, 1) for k, v in cats.items()},
                                    "per": per}) + "\n")
            cpu_f.flush()
            prev, prev_t = now, t
            try:
                text = httpx.get(ROOT + "/metrics", timeout=10).text
                vals = {}
                for line in text.splitlines():
                    if line.startswith("vllm:") and "_bucket" not in line:
                        key, _, v = line.rpartition(" ")
                        name = key.split("{")[0]
                        try:
                            vals[name] = vals.get(name, 0.0) + float(v) if not name.endswith(("_running", "_waiting",
                                                                                               "_perc")) \
                                else max(vals.get(name, 0.0), float(v))
                        except ValueError:
                            pass
                met_f.write(json.dumps({"t": round(t, 2), "m": vals}) + "\n")
                met_f.flush()
            except Exception as exc:  # noqa: BLE001
                met_f.write(json.dumps({"t": round(t, 2), "err": str(exc)[:200]}) + "\n")


def wait_ready(proc=None, timeout=1500):
    import httpx
    began = time.time()
    while time.time() - began < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode}")
        try:
            if httpx.get(ROOT + "/v1/models", timeout=5).status_code == 200:
                return round(time.time() - began, 1)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    raise RuntimeError("server not ready")


def serve_record():
    files = sorted(WORK.glob("serve-*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(files[0].read_text()) if files else None


def stop_server(out):
    """SIGTERM then SIGKILL every vLLM process (serve main, API servers, EngineCore, workers); wait for the card."""
    import signal
    began = time.time()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = [p for p, (cat, _, _) in proc_table().items()
                if cat in ("api", "engine", "worker", "serve_main", "fake") and p != os.getpid()]
        pids += [p for p, (_, name, _) in proc_table().items() if "fake_server.py" in name]
        for p in pids:
            try:
                os.kill(p, sig)
            except OSError:
                pass
        for _ in range(60):
            if not any(cat in ("api", "engine", "worker", "serve_main") or "fake_server" in name
                       for cat, name, _ in proc_table().values()):
                break
            time.sleep(1)
        else:
            continue
        break
    mem = None
    for _ in range(60):
        try:
            mem = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=30).stdout.split()[0])
        except Exception:  # noqa: BLE001
            break
        if mem < 2000:
            break
        time.sleep(2)
    out["stop_s"], out["card_mib_after_stop"] = round(time.time() - began, 1), mem


def apply_fast(record, stub, spec=None):
    """serving_fast overlay on the runtime the recorded server imports (its vllm package's parent)."""
    spec = spec or {"logprob_fast": True, "image_cache": True}
    if stub:
        return {"stub": True, "spec": spec}
    sys.path.insert(0, "/opt/fe")
    import serving_fast_patch
    argv = record["orig_argv"] or [record["executable"]]
    site = subprocess.run([argv[0], "-c", "import os, importlib.util as u; print(os.path.dirname(os.path.dirname(u.find_spec('vllm').origin)))"],
                          env=record["env"], capture_output=True, text=True, timeout=600).stdout.strip().splitlines()[-1]
    return {"site": site, "identity": serving_fast_patch.apply(site, str(WORK / "fast"), spec)}


def relaunch(record, extra, env_mod, log):
    argv = record["orig_argv"] or [record["executable"], *record["argv"]]
    env = {**record["env"], **env_mod}
    for k in [k for k, v in env_mod.items() if v is None]:
        env.pop(k, None)
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)
    handle = open(log, "w")
    proc = subprocess.Popen([*argv, *extra], env=env, cwd=record.get("cwd") or "/", stdout=handle,
                            stderr=subprocess.STDOUT, start_new_session=True)
    handle.close()
    return proc


def exact_prompts(n=20, max_tokens=12000):
    """The first live request of each L1 attempt under max_tokens prompt tokens (sorted files), as chat bodies."""
    import glob
    picks = []
    for f in sorted(glob.glob(f"{DATA}/r12-*L1/*.jsonl")):
        payload, tools = [], None
        for line in open(f):
            r = json.loads(line)
            payload = payload[:r.get("keep", len(payload))] + r["messages"]
            if "tools" in r:
                tools = r["tools"]
            if not r.get("scripted") and r["usage"]["prompt_tokens"] <= max_tokens:
                picks.append({"messages": payload, "tools": tools, "chat_template_kwargs": r.get("template") or {}})
                break
        if len(picks) >= n:
            break
    return picks


def exactness(picks, model):
    import httpx
    res = []
    with httpx.Client(trust_env=False, timeout=600) as c:
        for i, p in enumerate(picks):
            body = {"model": model, **{k: v for k, v in p.items() if v is not None}, "max_tokens": 1 if i >= 4 else 32,
                    "temperature": 0.0, "prompt_logprobs": 0, "return_token_ids": True}
            r = c.post(ROOT + "/v1/chat/completions", json=body)
            if r.status_code != 200:
                res.append({"error": r.text[:300]})
                continue
            d = r.json()
            lp = []
            for pos in d.get("prompt_logprobs") or []:
                if pos is None:
                    lp.append(None)
                else:
                    best = next(iter(pos.values()))
                    lp.append(best["logprob"] if isinstance(best, dict) else best)
            res.append({"n": len(lp), "lp": lp, "ids": d["choices"][0].get("token_ids"),
                        "pt": (d.get("usage") or {}).get("prompt_tokens")})
    return res


def reply_prompts(n=16, lo=6000, hi=30000):
    """Live R12 requests with lo..hi prompt tokens (every third such row, sorted files), as chat bodies."""
    import glob
    picks = []
    for f in sorted(glob.glob(f"{DATA}/r12-*/*.jsonl")):
        payload, tools, k = [], None, 0
        for line in open(f):
            r = json.loads(line)
            payload = payload[:r.get("keep", len(payload))] + r["messages"]
            if "tools" in r:
                tools = r["tools"]
            if not r.get("scripted") and lo <= r["usage"]["prompt_tokens"] <= hi:
                k += 1
                if k % 3 == 1:
                    s = r.get("sampling") or {}
                    picks.append({"messages": payload, "tools": tools, "chat_template_kwargs": r.get("template") or {},
                                  **{x: s[x] for x in ("temperature", "top_p", "top_k") if s.get(x) is not None},
                                  "max_tokens": min(192, max(1, r["usage"]["completion_tokens"]))})
            if len(picks) >= n:
                return picks
    return picks


def exact_replies(picks, model):
    """Each request alone after a prefix-cache reset, seeded, with the loop's logprob/token-id fields; the raw reply
    with request/tool-call ids and the creation time blanked."""
    import hashlib
    import re

    import httpx
    res = []
    with httpx.Client(trust_env=False, timeout=900) as c:
        for i, p in enumerate(picks):
            c.post(ROOT + "/reset_prefix_cache")
            body = {"model": model, **{k: v for k, v in p.items() if v is not None}, "seed": 1000 + i,
                    "logprobs": True, "top_logprobs": 20, "return_tokens_as_token_ids": True, "return_token_ids": True}
            r = c.post(ROOT + "/v1/chat/completions", json=body)
            text = re.sub(r'"created":\s*\d+', '"created":0', re.sub(r'"id":\s*"[^"]*"', '"id":""', r.text))
            res.append({"status": r.status_code, "sha256": hashlib.sha256(text.encode()).hexdigest(), "bytes": len(text),
                        "head": text[:200] if r.status_code != 200 else None})
    return res


ROUTES = Path("/tmp/arc-served-route")
KEEP = {}
MODEL_DIR = "/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
_TOK = []


def tokenizer():
    if not _TOK:
        from transformers import AutoTokenizer
        _TOK.append(AutoTokenizer.from_pretrained(MODEL_DIR))
    return _TOK[0]


STUB = []


def routes_of(api_id, wait=30):
    """The served-route record of one finished request (R3 writer, asynchronous): {lo, hi, tokens, ids} or None."""
    sys.path.insert(0, "/opt/arc-validation/src")
    from arena.served_route import read
    began = time.time()
    wait = 0.2 if STUB else wait
    while time.time() - began < wait:
        done = list(ROUTES.glob(f"{api_id}-*.json"))
        if done:
            rts = list(ROUTES.glob(f"{api_id}-*.rt"))
            return read(rts[0]) if rts else None
        time.sleep(0.5)
    return None


def chat(c, body):
    r = c.post(ROOT + "/v1/chat/completions", json=body)
    return r.status_code, (r.json() if r.status_code == 200 else {"error": r.text[:300]})


def gen_and_record(c, p, i, max_tokens=128):
    """Seeded reply after a prefix-cache reset: prompt ids, reply ids, sampled logprobs, top-20 check, routes."""
    c.post(ROOT + "/reset_prefix_cache")
    body = {"model": "policy", **{k: v for k, v in p.items() if v is not None and k != "max_tokens"}, "seed": 1000 + i,
            "max_tokens": max_tokens, "logprobs": True, "top_logprobs": 20, "return_tokens_as_token_ids": True,
            "return_token_ids": True}
    st, d = chat(c, body)
    if st != 200:
        return {"error": d}
    ch = d["choices"][0]
    content = (ch.get("logprobs") or {}).get("content") or []
    import math
    import re
    well = all(len(e.get("top_logprobs") or []) == 20 and re.fullmatch(r"token_id:\d+", e["token"]) and
               all(re.fullmatch(r"token_id:\d+", t["token"]) and math.isfinite(t["logprob"]) for t in e["top_logprobs"])
               for e in content)
    rec = routes_of(d["id"])
    return {"prompt": d.get("prompt_token_ids"), "reply": ch.get("token_ids"), "lp": [e["logprob"] for e in content],
            "top20_ok": bool(well and len(content) == len(ch.get("token_ids") or [])), "routes": rec}


def teacher_force(c, p, prompt, reply, i, model="policy"):
    """Prompt + reply re-rendered as a final assistant message (reasoning split as the template expects), prompt
    logprobs and routes of the whole sequence after a cache reset; usable positions = the matching prefix."""
    tok = tokenizer()
    raw = tok.decode(reply, skip_special_tokens=False)
    reasoning, sep, content = raw.partition("</think>")
    if not sep:
        reasoning, content = raw, ""
    reasoning = reasoning.split("<think>")[-1]
    msgs = list(p["messages"]) + [{"role": "assistant", "content": content.lstrip("\n"),
                                   "reasoning": reasoning}]
    c.post(ROOT + "/reset_prefix_cache")
    body = {"model": model, "messages": msgs, "tools": p.get("tools"), "max_tokens": 1, "temperature": 0.0,
            "chat_template_kwargs": p.get("chat_template_kwargs") or {},
            "continue_final_message": True, "add_generation_prompt": False, "prompt_logprobs": 0,
            "return_token_ids": True, "seed": 2000 + i}
    if not body["tools"]:
        body.pop("tools")
    st, d = chat(c, body)
    if st != 200:
        return {"error": d}
    ids = d.get("prompt_token_ids") or []
    n = len(prompt)
    if ids[:n] != prompt:
        return {"prefix_mismatch": True, "n": len(ids)}
    m = 0
    while m < len(reply) and n + m < len(ids) and ids[n + m] == reply[m]:
        m += 1
    lp = []
    for j in range(m):
        pos = (d.get("prompt_logprobs") or [])[n + j]
        lp.append(pos.get(str(reply[j]), {}).get("logprob") if pos else None)
    return {"match": m, "of": len(reply), "lp": lp, "routes": routes_of(d["id"])}


def teacher_force_nocache_reset(c, p, prompt, reply, i):
    """teacher_force without the cache reset: right after another teacher-forced request on the same text."""
    orig = c.post
    c.post = lambda url, *a, **k: None if url.endswith("/reset_prefix_cache") else orig(url, *a, **k)
    try:
        return teacher_force(c, p, prompt, reply, i)
    finally:
        c.post = orig


def route_agree(dec, tf, n, m):
    """Top-k set agreement (exact set, and mean overlap) on the fed reply positions n .. n+m-2."""
    import numpy as np
    if dec is None or tf is None or m < 2:
        return None
    lo, hi = n, n + m - 1
    if dec["lo"] > lo or tf["lo"] > lo or dec["hi"] < hi or tf["hi"] < hi:
        return None
    a, b = dec["ids"][lo:hi], tf["ids"][lo:hi]
    ok = dec["covered"][lo:hi] & tf["covered"][lo:hi]
    a, b = np.sort(a[ok], axis=-1), np.sort(b[ok], axis=-1)
    same = (a == b).all(axis=-1)
    over = np.array([[len(set(x) & set(y)) / len(x) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]) if len(a) else []
    return {"positions": int(ok.sum()), "set_equal": float(same.mean()) if same.size else None,
            "overlap": float(np.mean(over)) if len(over) else None}


def lp_diff(a, b):
    d = [abs(x - y) for x, y in zip(a or [], b or []) if x is not None and y is not None]
    return {"mean": round(float(sum(d) / len(d)), 5) if d else None, "max": round(max(d), 4) if d else None,
            "n": len(d)}


def completion_prompt_lp(c, model, ids):
    r = c.post(ROOT + "/v1/completions", json={"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0,
                                                "prompt_logprobs": 0})
    if r.status_code != 200:
        return None
    out = []
    for pos, t in zip(r.json()["choices"][0].get("prompt_logprobs") or [], ids):
        out.append(None if pos is None else (pos.get(str(t)) or {}).get("logprob"))
    return out


def floor_checks(c, picks, keep_a1):
    """Prompt-logprob floor (RL hypotheses): text-only token prompts repeated (LoRA and base), the image chat
    teacher forcing repeated with the prefix cache reset, the same with a cache hit, and LoRA vs base."""
    tok = tokenizer()
    out = {}
    text = "\n".join(m.get("content") if isinstance(m.get("content"), str) else " ".join(
        x.get("text", "") for x in (m.get("content") or []) if isinstance(x, dict)) for m in picks[0]["messages"])
    ids = tok.encode(text, add_special_tokens=False)[:4000]
    for model in ("policy", SERVED):
        runs = []
        for _ in range(2):
            c.post(ROOT + "/reset_prefix_cache")
            runs.append(completion_prompt_lp(c, model, ids))
        out[f"text_repeat_{'lora' if model == 'policy' else 'base'}"] = lp_diff(*runs)
        out.setdefault("_text", {})[model] = runs[0]
    out["text_lora_vs_base"] = lp_diff(out["_text"]["policy"], out["_text"][SERVED])
    c.post(ROOT + "/reset_prefix_cache")
    hit = completion_prompt_lp(c, "policy", ids)  # cache warm from no reset: the second of two without a reset
    hit2 = completion_prompt_lp(c, "policy", ids)
    out["text_cached_vs_uncached"] = lp_diff(hit, hit2)
    out.pop("_text")
    for i in range(min(4, len(picks))):
        g = keep_a1.get(i)
        if not g:
            continue
        p = picks[i]
        t1 = teacher_force(c, p, g["prompt"], g["reply"], i)
        tb = teacher_force(c, p, g["prompt"], g["reply"], i, model=SERVED)
        t2 = teacher_force(c, p, g["prompt"], g["reply"], i)
        # without a reset: the request after t2's own reset re-reads the cached prompt
        body_cached = teacher_force_nocache_reset(c, p, g["prompt"], g["reply"], i)
        row = {"image_tf_repeat": lp_diff(t1.get("lp"), t2.get("lp")),
               "image_tf_lora_vs_base": lp_diff(t1.get("lp"), tb.get("lp")),
               "image_tf_cached_vs_uncached": lp_diff(t1.get("lp"), body_cached.get("lp")),
               "match": t1.get("match")}
        out[f"pick{i}"] = row
    return out


def mtp_checks(arm, folder, picks, keep):
    """Per pick: seeded reply (+top-20 check, routes), and for A arms the teacher-forced own reply; A2 also forces
    M's replies. keep: replies recorded by earlier arms."""
    import httpx
    import numpy as np
    out = {"picks": []}
    keep_tf = {}
    with httpx.Client(trust_env=False, timeout=900) as c:
        for i, p in enumerate(picks):
            g = gen_and_record(c, p, i)
            row = {"top20_ok": g.get("top20_ok"), "error": g.get("error"), "reply_len": len(g.get("reply") or [])}
            if "error" not in g:
                keep.setdefault(arm, {})[i] = g
                targets = [] if arm.startswith("M") else [(arm, g)]
                if arm == "A2" and i in keep.get("M", {}):
                    targets.append(("M", keep["M"][i]))
                for src, rec in targets:
                    tf = teacher_force(c, p, rec["prompt"], rec["reply"], i)
                    if "error" in tf or tf.get("prefix_mismatch"):
                        row[f"tf_{src}"] = {k: v for k, v in tf.items() if k != "routes"}
                        continue
                    m = tf["match"]
                    if arm == "A1" and src == "A1":
                        keep_tf[i] = tf["lp"]
                    d = [abs(x - y) for x, y in zip(rec["lp"][:m], tf["lp"]) if y is not None]
                    row[f"tf_{src}"] = {"match": m, "of": tf["of"], "lp_absdiff_mean": float(np.mean(d)) if d else None,
                                        "lp_n": len(d), "routes": route_agree(rec["routes"], tf["routes"],
                                                                              len(rec["prompt"]), m)}
                if arm == "A1" and i < 4:  # floor of the teacher-forced path: the same request again, and the base model
                    again = teacher_force(c, p, g["prompt"], g["reply"], i)
                    base = teacher_force(c, p, g["prompt"], g["reply"], i, model=SERVED)
                    first = row.get("tf_A1") or {}
                    for name, other in (("tf_repeat", again), ("tf_base", base)):
                        if "lp" in other and first.get("match"):
                            ref = keep_tf.get(i)
                            d = [abs(x - y) for x, y in zip(ref or [], other["lp"]) if x is not None and y is not None]
                            row[name] = {"lp_absdiff_mean": float(np.mean(d)) if d else None, "n": len(d)}
            out["picks"].append(row)
        if arm == "A1":
            try:
                out["floor"] = floor_checks(c, picks, keep.get("A1", {}))
            except Exception as exc:  # noqa: BLE001
                out["floor"] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    (folder / "gen.json").write_text(json.dumps({i: {"prompt_len": len(g["prompt"] or []), "reply": g["reply"],
                                                    "lp": g["lp"]} for i, g in keep.get(arm, {}).items()}))
    (folder / "mtp_checks.json").write_text(json.dumps(out, indent=1))
    return out


def summarize_mtp(results):
    import numpy as np

    def agg(arm, src):
        rows = [r.get(f"tf_{src}") for r in (results.get(arm, {}).get("mtp") or {}).get("picks", [])]
        rows = [r for r in rows if r and r.get("routes")]
        if not rows:
            return None
        pos = sum(r["routes"]["positions"] for r in rows)
        w = lambda k: sum(r["routes"][k] * r["routes"]["positions"] for r in rows if r["routes"][k] is not None) / max(pos, 1)
        lps = [(r["lp_absdiff_mean"], r["lp_n"]) for r in rows if r["lp_absdiff_mean"] is not None]
        return {"requests": len(rows), "positions": pos, "set_equal": round(w("set_equal"), 5),
                "overlap": round(w("overlap"), 5),
                "lp_absdiff_mean": round(sum(a * n for a, n in lps) / max(1, sum(n for _, n in lps)), 5),
                "match_share": round(sum(r["match"] for r in rows) / max(1, sum(r["of"] for r in rows)), 4)}
    out = {"A1": agg("A1", "A1"), "A2": agg("A2", "A2"), "M": agg("A2", "M")}
    if out["A1"] and out["A2"] and out["M"]:
        noise = abs(out["A1"]["set_equal"] - out["A2"]["set_equal"])
        rate_a = (out["A1"]["set_equal"] + out["A2"]["set_equal"]) / 2
        out["route_bar"] = {"rate_A": round(rate_a, 5), "noise": round(noise, 5), "rate_M": out["M"]["set_equal"],
                            "pass": out["M"]["set_equal"] >= rate_a - noise}
    dec = {}
    for x, y in (("A1", "A2"), ("M", "A1"), ("M", "A2")):
        gx, gy = KEEP.get(x, {}), KEEP.get(y, {})
        diffs, matched, total = [], 0, 0
        for i in set(gx) & set(gy):
            rx, ry = gx[i]["reply"] or [], gy[i]["reply"] or []
            m = 0
            while m < min(len(rx), len(ry)) and rx[m] == ry[m]:
                m += 1
            matched += m
            total += min(len(rx), len(ry))
            diffs += [abs(a - b) for a, b in zip(gx[i]["lp"][:m], gy[i]["lp"][:m])]
        dec[f"{x}_vs_{y}"] = {"matched_positions": matched, "of": total,
                              "mean_absdiff": round(sum(diffs) / len(diffs), 5) if diffs else None,
                              "max_absdiff": round(max(diffs), 4) if diffs else None}
    out["decode_sampled_lp"] = dec
    out["floor"] = (results.get("A1", {}).get("mtp") or {}).get("floor")
    out["top20_ok"] = {a: all(r.get("top20_ok") for r in (results.get(a, {}).get("mtp") or {}).get("picks", []))
                       for a in results}
    return out


def compare(a, b):
    diff = mx = pos = 0
    ids_same = True
    for x, y in zip(a, b):
        if "lp" not in x or "lp" not in y or x["n"] != y["n"]:
            return {"comparable": False}
        for u, v in zip(x["lp"], y["lp"]):
            if u is None or v is None:
                continue
            pos += 1
            if u != v:
                diff += 1
                mx = max(mx, abs(u - v))
        ids_same &= x["ids"] == y["ids"]
    return {"positions": pos, "differing": diff, "max_abs": mx, "greedy_ids_same": ids_same}


def models_seen(k=24):
    """Distinct-connection /v1/models reads (each new connection may land on another front end)."""
    import httpx
    seen = []
    for _ in range(k):
        with httpx.Client(trust_env=False, timeout=30) as c:
            seen.append(sorted(m["id"] for m in c.get(ROOT + "/v1/models").json()["data"]))
    return {"reads": k, "all_have_policy": all("policy" in s for s in seen), "variants": len({tuple(s) for s in seen})}


def pyspy(pid, seconds, path):
    return subprocess.Popen([PYSPY, "record", "--pid", str(pid), "--duration", str(seconds), "--rate", "50",
                             "--format", "raw", "--threads", "--nonblocking", "--output", str(path)],
                            stdout=subprocess.DEVNULL, stderr=open(str(path) + ".err", "w"))


CATS = {"r3_routing": ("served_route", "routed_experts", "arc_served_route"),
        "multimodal": ("multimodal", "/mm_", "image", "PIL", "pixel", "processor", "hash"),
        "logprobs": ("logprob",),
        "render_tokenize": ("render", "tokeniz", "chat_template", "jinja", "encode"),
        "serialize": ("json", "msgspec", "serial", "pydantic", "model_dump", "validate", "orjson", "msgpack"),
        "detok_parse": ("detok", "reasoning", "tool_parser", "parse"),
        "zmq_io": ("zmq", "send", "recv", "socket")}


IDLE = {"select", "poll", "wait", "_wait", "acquire", "sleep", "get", "recv", "recv_multipart", "_recv", "_poll",
        "wait_for", "_wait_for_tstate_lock", "join", "accept", "readinto", "dequeue", "busy_loop_wait"}


def summarize_prof(path):
    """In-process stack samples of one process: busy share per thread (leaf frame not a known blocking call), top
    leaf and inclusive frames over busy samples, and the share of busy samples in each suspect category."""
    d = json.loads(Path(path).read_text())
    selfc, incl, threads, busy, total = {}, {}, {}, 0, 0
    cat = {k: 0 for k in CATS}
    for key, n in d["counts"].items():
        tid, _, stack = key.partition(";")
        frames = stack.split(";")
        leaf = frames[-1].split(" (")[0] if frames and frames[-1] else ""
        total += n
        tname = d["threads"].get(tid, tid)[:40]
        t = threads.setdefault(tname, [0, 0])
        t[1] += n
        if leaf in IDLE:
            continue
        t[0] += n
        busy += n
        selfc[frames[-1][:140]] = selfc.get(frames[-1][:140], 0) + n
        for f in set(fr[:140] for fr in frames):
            incl[f] = incl.get(f, 0) + n
        low = stack.lower()
        for k, keys in CATS.items():
            if any(x.lower() in low for x in keys):
                cat[k] += n
    ticks = max(d["ticks"], 1)
    top = lambda dd, k: [[f, round(c / max(busy, 1), 3)] for f, c in sorted(dd.items(), key=lambda x: -x[1])[:k]]
    return {"cmd": d["cmd"][:80], "ticks": d["ticks"],
            "threads_busy_share": {k: round(v[0] / ticks, 3) for k, v in sorted(threads.items(), key=lambda x: -x[1][0])[:10]},
            "busy_threads_sum": round(busy / ticks, 3), "self": top(selfc, 25), "inclusive": top(incl, 45),
            "cats": {k: round(v / max(busy, 1), 3) for k, v in cat.items()}}


def run_load(arm, folder, python, warm, measure, status, stub, streams=None, model="policy", no_min=False, extra=0):
    """extra: seconds the load keeps running after the stats window (the torch profile of S/P arms runs there)."""
    import shutil
    folder.mkdir(parents=True, exist_ok=True)
    stop = folder / "stop"
    nproc, streams = (6, streams or 110) if not stub else (2, 8)
    stop.unlink(missing_ok=True)  # a stale stop file from an earlier run into the same folder ends the drivers at once
    sampler = Sampler(folder)
    sampler.start()
    drivers = []
    t_start = time.time()
    for i in range(nproc):
        k = streams // nproc + (i < streams % nproc)
        # load corpus: v6j (FE_DATA_GLOB v6j-*, each replay from FE_START_FRAC of its try) unless the glob names r12
        data_glob = "r12-*" if stub else os.environ.get("FE_DATA_GLOB", "v6j-*")
        r12 = data_glob.startswith("r12")
        drivers.append(subprocess.Popen(
            [python, "/opt/fe/driver.py", str(i), str(nproc), str(k), model, ROOT,
             str(WORK / "stubdata") if stub else (DATA if r12 else LOAD_DATA),
             str(folder / f"drv-{i}.jsonl"), str(stop)],
            stdout=open(folder / f"drv-{i}.out", "w"), stderr=subprocess.STDOUT,
            env={**os.environ, "FE_BENCH": "0", "FE_NO_MIN": "1" if no_min else "0", "FE_DATA_GLOB": data_glob,
                 "FE_START_FRAC": os.environ.get("FE_START_FRAC", "0" if r12 else "0.5")}))
    ws, we = t_start + warm, t_start + warm + measure
    profiled = False
    while time.time() < we + extra:
        time.sleep(min(30, max(1, we + extra - time.time())))
        if not profiled and time.time() > ws + min(180, measure / 3):
            profiled = True
            prof_until = time.time() + (60 if not stub else 5)
            (WORK / "prof.on").write_text(json.dumps({"until": prof_until, "tag": arm}))
        status.update(phase=f"{arm}-load", load_elapsed=round(time.time() - t_start), **live_stats(folder, ws))
        status.dump()
        if any(d.poll() is not None for d in drivers):
            status.update(driver_exit=[d.poll() for d in drivers])
    stop.write_text("1")
    for d in drivers:
        try:
            d.wait(timeout=90)
        except subprocess.TimeoutExpired:
            d.kill()
    sampler.stop.set()
    summary = window_stats(folder, ws, we)
    table = proc_table()
    summary["profile"] = {}
    for f in sorted(WORK.glob(f"prof-*-{arm}.json")):
        try:
            pid = int(f.name.split("-")[1])
            shutil.copy(f, folder / f.name)
            row = summarize_prof(f)
            summary["profile"][f"{table.get(pid, ('gone',))[0]}:{pid}"] = row
        except Exception as exc:  # noqa: BLE001
            summary["profile"][f.name] = {"error": str(exc)[:300]}
    return summary


# --- tput mode (RL play-throughput levers, 23 Oct): arms "<kind>-<client streams>", kind L (r5's server), B (no
# LoRA: the upper bound for LoRA kernel work), M1/M3 (MTP-k with runtime LoRA, serving_fast.mtp_lora_draft).
LORA_FLAGS = {"--enable-lora": 0, "--max-lora-rank": 1, "--max-loras": 1, "--max-cpu-loras": 1,
              "--enable-mixed-moe-lora-format": 0}
DECODE = {}


def tput_arm(arm):
    """"L-70+110" -> ("L", 70, [110]): loads at 70 then 110 client streams on the same server."""
    kind, _, streams = arm.partition("-")
    counts = [int(x) for x in streams.split("+") if x]
    return kind, (counts[0] if counts else None), counts[1:]


def argv_without_lora(argv):
    out, skip = [], 0
    for a in argv:
        if skip:
            skip -= 1
            continue
        if a in LORA_FLAGS:
            skip = LORA_FLAGS[a]
            continue
        out.append(a)
    return out


def mtp_extra(argv, k):
    """Speculative config plus graph sizes covering (k+1) positions per stream up to 80 streams, and 8 GiB less KV
    for the draft (KV use stays under 0.6 at these loads)."""
    sizes = [1, 2, 4, 8] + list(range(16, 153, 8)) + list(range(168, 80 * (k + 1) + 1, 16))
    extra = ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": k}, separators=(",", ":")),
             "--compilation-config", json.dumps({"cudagraph_capture_sizes": sizes}, separators=(",", ":")),
             "--max-cudagraph-capture-size", str(sizes[-1])]
    if "--kv-cache-memory-bytes" in argv:
        kv = int(argv[argv.index("--kv-cache-memory-bytes") + 1])
        extra += ["--kv-cache-memory-bytes", str(kv - 8 * 2**30)]
    return extra


def gen_lp(c, p, i, model, greedy):
    c.post(ROOT + "/reset_prefix_cache")
    body = {"model": model, **{k: v for k, v in p.items() if v is not None and k != "max_tokens"}, "seed": 1000 + i,
            "max_tokens": 128, "logprobs": True, "return_tokens_as_token_ids": True, "return_token_ids": True}
    if greedy:
        body.update(temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0)
    st, d = chat(c, body)
    if st != 200:
        return {"error": str(d)[:300]}
    ch = d["choices"][0]
    content = (ch.get("logprobs") or {}).get("content") or []
    return {"reply": ch.get("token_ids") or [], "lp": [e["logprob"] for e in content]}


def decode_checks(arm, picks, model, repeat=False):
    import httpx
    out = DECODE.setdefault(arm, {})
    errors = []
    with httpx.Client(trust_env=False, timeout=900) as c:
        for i, p in enumerate(picks):
            for mode in ("sampled", "greedy") + (("sampled_repeat",) if repeat else ()):
                g = gen_lp(c, p, i, model, mode == "greedy")
                if "error" in g:
                    errors.append(g["error"])
                else:
                    out.setdefault(mode, {})[i] = g
    return {"picks": len(picks), "errors": errors[:3], "n_errors": len(errors)}


def decode_compare(x, y):
    diffs, matched, total, reqs = [], 0, 0, 0
    for i in set(x) & set(y):
        rx, ry = x[i]["reply"], y[i]["reply"]
        m = 0
        while m < min(len(rx), len(ry)) and rx[m] == ry[m]:
            m += 1
        reqs += 1
        matched += m
        total += min(len(rx), len(ry))
        diffs += [abs(a - b) for a, b in zip(x[i]["lp"][:m], y[i]["lp"][:m])]
    return {"requests": reqs, "matched_positions": matched, "of": total,
            "mean_absdiff": round(sum(diffs) / len(diffs), 5) if diffs else None,
            "max_absdiff": round(max(diffs), 4) if diffs else None}


def summarize_tput(results, first):
    out = {"arms": {}, "decode_vs_" + first: {}}
    keys = ("client_inflight_mean", "engine_running_mean", "engine_waiting_mean", "kv_mean", "client_gen_tok_s",
            "metrics_gen_tok_s", "per_stream_decode_tok_s", "spec_acceptance", "spec_tokens_per_step", "requests",
            "errors", "client_e2e_mean_s", "ttft_s", "mean_step_ms", "engine_steps_s", "gen_per_step", "tok_per_step", "kv_p90",
            "kv_max", "preemptions", "client_computed_prompt_tok_s", "engine_running_p10", "engine_running_p90")
    for a, r in results.items():
        out["arms"][a] = {k: (r.get("load") or {}).get(k) for k in keys} | {"error": r.get("error"),
                                                                            "boot_wall_s": r.get("boot_wall_s"),
                                                                            "graph_facts": r.get("graph_facts")}
        if r.get("kernels"):
            out.setdefault("kernels", {})[a] = r["kernels"] | {"calls": r.get("profile_calls")}
        if r.get("steps"):
            out.setdefault("steps", {})[a] = r["steps"]
        for k2, v in r.items():
            if k2.startswith("load_") and k2[5:].isdigit():
                out["arms"][f"{a}@{k2[5:]}"] = {k: (v or {}).get(k) for k in keys}
    ref = DECODE.get(first, {})
    if "sampled_repeat" in ref:
        out["within_boot_" + first] = decode_compare(ref.get("sampled", {}), ref["sampled_repeat"])
    for a in results:
        if a == first or a.startswith(("B", "R", "GR")) or "P-" in a or "S-" in a:
            continue
        out["decode_vs_" + first][a] = {m: decode_compare(ref.get(m, {}), DECODE.get(a, {}).get(m, {}))
                                        for m in ("sampled", "greedy")}
    return out


SEGMENT = {}


def adapter_names(arm, mode):
    """[LQF]<n>: n adapter slots (policy, policy1, ...) loaded from the same files, replayed round-robin."""
    kind = tput_arm(arm)[0] if mode == "tput" else ""
    m = re.fullmatch(r"[LQFHK](\d)[SPX]?", kind)
    return ["policy"] + [f"policy{i}" for i in range(1, int(m.group(1)) if m else 1)]


PATCHED_FILES = ("vllm/lora/punica_wrapper/punica_gpu.py", "vllm/lora/ops/triton_ops/lora_shrink_op.py",
                 "vllm/lora/ops/triton_ops/lora_expand_op.py", "vllm/device_allocator/sleep_mode_backend.py",
                 "vllm/v1/engine/core.py", "vllm/device_allocator/cumem.py", "vllm/v1/worker/gpu_worker.py")


def runtime_site(record, stub=False):
    """The site-packages the recorded server imports; in a stub preflight a copy of the pinned files (so the patch,
    restore and tag checks run for real)."""
    if stub:
        site = WORK / "stubsite"
        if not site.exists():
            for rel in PATCHED_FILES:
                (site / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f"/opt/fe/pinned/{rel}", site / rel)
        return str(site)
    argv = record["orig_argv"] or [record["executable"]]
    return subprocess.run([argv[0], "-c", "import os, importlib.util as u; print(os.path.dirname(os.path.dirname("
                           "u.find_spec('vllm').origin)))"], env=record["env"], capture_output=True, text=True,
                          timeout=600).stdout.strip().splitlines()[-1]


def restore_pristine(record, stub):
    """Every patched runtime file back to its state before this run's first patch."""
    site = Path(runtime_site(record, stub))
    for rel in PATCHED_FILES:
        f = site / rel
        ORIGINALS.setdefault(str(f), f.read_bytes())  # first seen before any arm patched it
    for f, data in ORIGINALS.items():
        Path(f).write_bytes(data)
    return {rel: hashlib.sha256((site / rel).read_bytes()).hexdigest()[:12] for rel in PATCHED_FILES}


def _patch_module(name, runtime=None):
    sys.path.insert(0, "/opt/fe")
    mod = importlib.import_module(name)
    if runtime:
        mod.runtime_source = lambda: Path(f"/opt/fe/{runtime}").read_bytes()
    return mod


def segment_patch(record, stub):
    """lora_segment_patch (item 1) on the restored runtime."""
    site = runtime_site(record, stub)
    return {"site": site, "identity": _patch_module("lora_segment_patch").apply(site, str(WORK / "segment"))}


def dense_patch(record, stub):
    """lora_dense_decode_patch (item 2) on the restored runtime."""
    site = runtime_site(record, stub)
    P = _patch_module("lora_dense_decode_patch", "lora_dense_runtime.py")
    return {"site": site, "identity": P.apply(site, str(WORK / "dense"))}


def moe_patch(record, stub):
    """moe_lora_fast_patch (MFU-1, mfu1-moel 839cd738c) on the restored runtime."""
    site = runtime_site(record, stub)
    P = _patch_module("moe_lora_fast_patch", "moe_lora_fast_runtime.py")
    return {"site": site, "identity": P.apply(site, str(WORK / "moe"))}


def sleep_patch(record, stub):
    """sleep_keep_kv_patch (level-1 sleep keeps KV on host + the prefix cache) on the restored runtime."""
    site = runtime_site(record, stub)
    return {"site": site, "identity": _patch_module("sleep_keep_kv_patch").apply(site, str(WORK / "sleepkv"))}


def zero_adapter(adapter, out):
    """A copy of ``adapter`` with every lora_B tensor zeroed (serves the base model under a LoRA name)."""
    shutil.copytree(adapter, out, dirs_exist_ok=True)
    try:
        from safetensors.torch import load_file, save_file
        tensors = load_file(str(out / "adapter_model.safetensors"))
        save_file({k: (v.zero_() if "lora_B" in k else v) for k, v in tensors.items()},
                  str(out / "adapter_model.safetensors"))
        return sum("lora_B" in k for k in tensors)
    except ImportError:  # stub container: the copy only
        return None


def keep_lora_probe(adapter, stub):
    """mg.timeshare.keep_lora read (MFU-2 25 Oct): two adapters (policy, policy-zero = lora_B zeroed) loaded once;
    greedy single requests (prefix cache reset before each) on both, twice awake (A, A2 = the exactness floor), then
    level-1 keep-KV sleep -> wake with NO adapter reload -> B, twice. Bars: B == A per adapter, and policy != zero."""
    import httpx
    zero = WORK / "adapter-zero"
    out = {"lora_b_zeroed": zero_adapter(adapter, zero)}
    r = httpx.post(ROOT + "/v1/load_lora_adapter", json={"lora_name": "policy-zero", "lora_path": str(zero),
                                                       "is_3d_lora_weight": True}, timeout=600)
    out["load_zero"] = [r.status_code, r.text[:200]]
    picks = reply_prompts()[:2 if stub else 6]

    def pass_():
        return {name: greedy_probe(picks, model=name, max_tokens=32) for name in ("policy", "policy-zero")}

    def same(x, y):
        return sum(a.get("ids") == b.get("ids") and a.get("lp") == b.get("lp") and bool(a.get("ids")) for a, b in zip(x, y))

    a, a2 = pass_(), pass_()
    cycles = []
    for _ in range(2):
        t0 = time.time()
        s1 = httpx.post(ROOT + "/sleep?level=1", timeout=900).status_code
        w1 = httpx.post(ROOT + "/wake_up", timeout=900).status_code
        b = pass_()
        cycles.append({"sleep_wake": [s1, w1, round(time.time() - t0, 1)],
                       **{f"{n}_exact_vs_a": same(a[n], b[n]) for n in a}})
    out.update(n=len(picks), aa={n: same(a[n], a2[n]) for n in a},
               policy_vs_zero_same=sum(x.get("ids") == y.get("ids") for x, y in zip(a["policy"], a["policy-zero"])),
               cycles=cycles)
    return out


def host_mem_gib():
    out = {}
    for key, path in (("cg_usage", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
                      ("cg_max_usage", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"),
                      ("cg_current", "/sys/fs/cgroup/memory.current")):
        try:
            out[key] = round(int(open(path).read().strip()) / 2**30, 2)
        except Exception:  # noqa: BLE001
            pass
    info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
    out["used"] = round((int(info["MemTotal"].split()[0]) - int(info.get("MemAvailable", info["MemFree"]).split()[0]))
                        / 2**20, 2)
    return out


def sleep_probe(arm, keep, adapter, stub, cycles, reload=True):
    """Keep-KV sleep read: a held set of long distinct prompts primed, resent (A), then per cycle
    sleep level 1 -> wake -> LoRA reload (as rl_loop) -> one 1-token request (wake-to-first-decode) -> resend (B).
    Prefix hit rate over each resend from /metrics deltas; B vs A outputs (tokens and logprobs) for exactness; decode
    tok/s on the held set before sleep and after the last wake; host memory sampled every 0.5 s."""
    import asyncio
    import random
    import httpx
    n = 3 if stub else int(os.environ.get("FE_SLEEP_N", "24"))
    lens = [200] * n if stub else [int(x) for x in random.Random(7).choices(range(20000, 60001, 1000), k=n)]
    vocab = [f"{w}" for w in random.Random(11).sample(range(10000, 99999), 4000)]
    # 5-digit words are ~5.4 tokens each (sleepkv1: 2.35M tokens at L / 2.2)
    prompts = [" ".join(random.Random(1000 + i).choices(vocab, k=int(L / 5.4))) for i, L in enumerate(lens)]
    peak = {"used": 0.0, "cg_usage": 0.0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            m = host_mem_gib()
            for k in peak:
                peak[k] = max(peak[k], m.get(k) or 0.0)
            time.sleep(0.5)
    threading.Thread(target=sample, daemon=True).start()

    def metric(*names):
        text = httpx.get(ROOT + "/metrics", timeout=60).text
        tot = {n_: 0.0 for n_ in names}
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            for n_ in names:
                if name in (n_, n_ + "_total"):
                    tot[n_] += float(line.rsplit(" ", 1)[1])
        return tot

    async def send(suffix, max_tokens=16, only=None):
        async with httpx.AsyncClient(timeout=1800) as c:
            async def one(i):
                body = {"model": "policy", "messages": [{"role": "user", "content": prompts[i] + suffix}],
                        "max_tokens": max_tokens, "temperature": 0, "logprobs": True, "top_logprobs": 1}
                t0 = time.time()
                r = (await c.post(ROOT + "/v1/chat/completions", json=body)).json()
                ch = r["choices"][0]
                lp = [(x.get("token"), x.get("logprob")) for x in ((ch.get("logprobs") or {}).get("content") or [])]
                return {"lp": lp, "usage": r.get("usage"), "s": round(time.time() - t0, 3)}
            idx = range(n) if only is None else only
            t0 = time.time()
            res = await asyncio.gather(*(one(i) for i in idx))
            return res, round(time.time() - t0, 2)

    def hit(fn):
        m0 = metric("vllm:prefix_cache_hits", "vllm:prefix_cache_queries")
        out = fn()
        m1 = metric("vllm:prefix_cache_hits", "vllm:prefix_cache_queries")
        q = m1["vllm:prefix_cache_queries"] - m0["vllm:prefix_cache_queries"]
        return out, round((m1["vllm:prefix_cache_hits"] - m0["vllm:prefix_cache_hits"]) / q, 4) if q else None

    def decode_tps():
        res, wall = asyncio.run(send(" Continue.", max_tokens=32 if stub else int(os.environ.get("FE_SLEEP_BURST", "2048"))))
        return round(sum((r["usage"] or {}).get("completion_tokens", 0) for r in res) / wall, 1)

    def compare(x, y):
        return {"exact": sum(a["lp"] == b["lp"] for a, b in zip(x, y)),
                "tokens_equal": sum([u for u, _ in a["lp"]] == [u for u, _ in b["lp"]] for a, b in zip(x, y)),
                "max_abs_lp": max((abs(u[1] - v[1]) for a, b in zip(x, y) for u, v in zip(a["lp"], b["lp"])
                                   if u[1] is not None and v[1] is not None), default=None)}

    row = {"n": n, "keep": keep, "prompt_tokens": None, "cycles": []}
    try:
        (prime, prime_s), _ = hit(lambda: asyncio.run(send("")))
        row["prompt_tokens"] = sum((r["usage"] or {}).get("prompt_tokens", 0) for r in prime)
        row["prime_s"] = prime_s
        (resend_a, a_s), row["hit_a"] = hit(lambda: asyncio.run(send(" ok")))
        row["resend_a_s"] = a_s
        (resend_a2, _), row["hit_a2"] = hit(lambda: asyncio.run(send(" ok")))
        row["aa"] = compare(resend_a, resend_a2)  # no-sleep repeat: the exactness floor
        row["decode_tps_before"] = decode_tps()
        row["decode_tps_before2"] = decode_tps()  # the burst's own repeat floor
        row["mem_awake"] = host_mem_gib()
        for c in range(cycles):
            cyc = {}
            t0 = time.time()
            r = httpx.post(ROOT + "/sleep?level=1", timeout=900)
            cyc["sleep"] = [r.status_code, round(time.time() - t0, 2)]
            cyc["mem_asleep"] = host_mem_gib()
            t0 = time.time()
            r = httpx.post(ROOT + "/wake_up", timeout=900)
            cyc["wake"] = [r.status_code, round(time.time() - t0, 2)]
            t1 = time.time()
            if reload:
                httpx.post(ROOT + "/v1/unload_lora_adapter", json={"lora_name": "policy"}, timeout=600)
                r = httpx.post(ROOT + "/v1/load_lora_adapter", json={"lora_name": "policy", "lora_path": str(adapter),
                                                                   "is_3d_lora_weight": True}, timeout=600)
                cyc["reload"] = [r.status_code, round(time.time() - t1, 2)]
            else:  # the adapter's GPU tensors come back with the weights (cumem "weights" tag)
                cyc["reload"] = [None, 0.0]
            (first, _), cyc["hit_first"] = hit(lambda: asyncio.run(send(" ok", max_tokens=1, only=[0])))
            cyc["first_s"] = first[0]["s"]
            cyc["wake_to_first_decode_s"] = round(cyc["wake"][1] + cyc["reload"][1] + first[0]["s"], 2)
            (resend_b, b_s), cyc["hit_b"] = hit(lambda: asyncio.run(send(" ok")))
            cyc["resend_b_s"] = b_s
            cyc.update(compare(resend_a, resend_b))
            row["cycles"].append(cyc)
        row["decode_tps_after"] = decode_tps()
    except Exception as exc:  # noqa: BLE001
        row["error"] = repr(exc)[:400]
    stop.set()
    row["mem_peak"] = dict(peak)
    return row


def patch_check(record, kind, log, stub):
    """Abort the arm unless exactly its own patch is in effect: every other arm's tag absent
    from the runtime files. Q also needs ARC_LORA_SEGMENT_USED on the boot warm-up step; H/K the FAST/BM16 flags as
    the server's python reads them (both skipped in a stub preflight, which has no vLLM)."""
    D = _patch_module("lora_dense_decode_patch")
    S = _patch_module("lora_segment_patch")
    MP = _patch_module("moe_lora_fast_patch")
    site = Path(runtime_site(record, stub))
    text = log.read_text(errors="replace") if log.is_file() else ""
    punica = (site / D.PUNICA).read_text()
    out = {"segment_file": S.TAG in (site / S.SHRINK).read_text() or S.TAG in (site / S.EXPAND).read_text(),
           "dense_file": D.TAG in punica, "moe_file": MP.TAG in punica}
    own = {"Q": "segment_file", "F": "dense_file", "H": "moe_file", "K": "moe_file", "L": None}[kind]
    want = {k: k == own for k in ("segment_file", "dense_file", "moe_file")}
    if not stub:
        out["segment_used"] = "ARC_LORA_SEGMENT_USED" in text
        want["segment_used"] = kind == "Q"
        if kind == "Q":  # tput15: no Triton LoRA specialization may be first launched inside graph capture
            import ast
            seen, bad = {}, []
            for line in text.splitlines():
                if "ARC_LORA_SEGMENT_FIRST " in line:
                    f = ast.literal_eval(line.split("ARC_LORA_SEGMENT_FIRST ", 1)[1].strip())
                    k = repr(f["key"])
                    seen.setdefault(k, []).append(f["path"])
                    if f["path"] == "capture" and "triton" not in seen[k]:
                        bad.append(f)
            out["first_launch"] = {"keys": len(seen), "capture_before_triton": bad[:5], "n_bad": len(bad)}
        if kind == "F":
            out["dense_helper"] = (site / D.HELPER).is_file()
            want["dense_helper"] = True
        if kind in "HK":
            argv = record["orig_argv"] or [record["executable"]]
            env = {**record["env"], "ARC_MOE_LORA_BM16": "1", "ARC_MOE_LORA_FAST": "1" if kind == "K" else "0"}
            r = subprocess.run([argv[0], "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
                                "import arc_moe_lora_fast as m; print(int(m.FAST), int(m.BM16))", str(site / "vllm")],
                               env=env, capture_output=True, text=True, timeout=600)
            out["fast_bm16"] = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
            want["fast_bm16"] = "1 1" if kind == "K" else "0 1"
    bad = {k: out[k] for k, v in want.items() if out[k] != v}
    if bad:
        raise RuntimeError(f"patch check failed for {kind}: {bad} (all: {out})")
    return out


def selftest_cpu(python, status):
    """Extract the pinned runtime as the server does, build the server's environment (serving_setup
    runtime_environment; no GPU, so the CUDA driver lookup is stubbed), then run segment_gpu_test with that record:
    the same command, env and PYTHONPATH handling as on the B300, CUDA cases on CPU."""
    sys.path.insert(0, "/opt/arc-validation/src")
    from arena import runtime_cache as RC
    out = {}
    try:
        recipe, runtime_dir = RC.pinned_recipe()
        t = time.time()
        recipe.verify_and_extract_runtime(Path(runtime_dir), full_layer_hashes=False, scan_extracted_caches=False)
        out["extract_s"] = round(time.time() - t, 1)
        # the PLE patch needs the bundle's patch dir (not mounted in this recipe copy); the self-test does not import
        # the PLE layer, so accept the file as extracted
        ple = recipe.RUNTIME_ROOT / "usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
        recipe.VLLM_PLE_PATCHED_SHA256 = recipe.sha256_file(ple)
        recipe._find_real_cuda_driver = lambda: Path("/usr/lib/x86_64-linux-gnu/libc.so.6")
        recipe.cuda_arch = lambda: "10.3"  # nvidia-smi query; B300
        env, _ = recipe.runtime_environment(deep_preload_validation=False)
        out["pythonpath"] = env.get("PYTHONPATH")
        record = {"executable": str(python), "orig_argv": None, "env": env}
        out["test"] = segment_gpu_test(record, False, require_cuda=False)
    except Exception as exc:  # noqa: BLE001
        import traceback
        out["error"] = traceback.format_exc()[-2500:]
    status.update(phase="done", selftest=out)
    status.dump()
    return out


def segment_gpu_test(record, stub, require_cuda=True):
    """test_lora_segment_patch.py on this GPU with the server's python (CUDA_LAUNCH_BLOCKING=1, pytest stand-in),
    before any server runs the segment path (tput11: a cuBLAS fault in the first Q boot)."""
    if stub:
        return {"stub": True, "ok": True}
    argv = record["orig_argv"] or [record["executable"]]
    d = WORK / "segtest"
    (d / "arc_infra").mkdir(parents=True, exist_ok=True)
    (d / "arc_infra" / "__init__.py").write_text("")
    shutil.copy("/opt/fe/lora_segment_patch.py", d / "arc_infra" / "lora_segment_patch.py")
    shutil.copy("/opt/fe/test_lora_segment_patch.py", d / "test_lora_segment_patch.py")
    shutil.copy("/opt/fe/pytest_shim.py", d / "pytest.py")  # the server venv has no pytest (tput12) and no pip index
    # prepend: the record's PYTHONPATH is where the server's torch/vLLM live (tput13: replacing it lost torch)
    path = os.pathsep.join(x for x in (str(d), record["env"].get("PYTHONPATH", "")) if x)
    # the server's own (pristine at this point) vLLM files stand in for the pinned copy: real prepare_tensors, the
    # boot warm-up replay with debug bounds asserts, anchors
    env = {**record["env"], "PYTHONPATH": path, "CUDA_LAUNCH_BLOCKING": "1", "ARC_PINNED_VLLM": runtime_site(record)}
    r = subprocess.run([argv[0], "-c", "import sys, pytest, torch; print('TORCH', torch.__version__, torch.__file__); "
                        + ("assert torch.cuda.is_available(); " if require_cuda else "")
                        + "sys.exit(0 if pytest.run_file(sys.argv[1]) else 1)", str(d / "test_lora_segment_patch.py")],
                       env=env, cwd=str(d), capture_output=True, text=True, timeout=900)
    tail = (r.stdout + r.stderr)[-1500:]
    return {"ok": r.returncode == 0 and " 0 failed" in r.stdout, "tail": tail}


def prof_extra(arm):
    iters = 40 if tput_arm(arm)[0].endswith("S") else 300
    d = str(WORK / f"tprof-{arm}")
    return ["--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_dir": d,
                                             "torch_profiler_with_stack": False, "ignore_frontend": True,
                                             "torch_profiler_use_gzip": True, "max_iterations": iters},
                                            separators=(",", ":"))]


def profile_window(arm, seconds=25):
    """Torch profiler over ~300 engine steps under load (worker CPU + GPU kernels, CUDA-graph kernels included)."""
    import httpx
    out = {}
    try:
        t = time.time()
        r = httpx.post(ROOT + "/start_profile", timeout=120)
        out["start"] = [r.status_code, round(time.time() - t, 1)]
        time.sleep(seconds)
        t = time.time()
        r = httpx.post(ROOT + "/stop_profile", timeout=900)
        out["stop"] = [r.status_code, round(time.time() - t, 1)]
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


KCLASS = [("lora", ("lora", "sgmv", "bgmv", "punica", "_arc_moe")), ("moe", ("moe", "marlin", "expert", "topk", "routing")),
          ("attention", ("attn", "attention", "flash", "fmha", "mla", "indexer", "sparse")),
          ("linear_attn", ("gdn", "gated_delta", "chunk_", "fused_recurrent", "causal_conv", "conv1d", "mamba")),
          ("gemm", ("gemm", "cutlass", "cublas", "sm90", "sm100", "nvjet", "matmul", "ampere")),
          ("sampling", ("sample", "topk_topp", "softmax", "logsumexp", "argmax")),
          ("norm_act", ("norm", "silu", "act_and_mul", "rotary", "rope", "gelu")),
          ("copy", ("copy", "memcpy", "memset", "cat", "index", "scatter", "gather", "fill"))]


def kernel_summary(arm, folder):
    """Kernel time by class and the top kernels from the worker trace(s) of this arm."""
    import glob
    import gzip
    files = [f for f in glob.glob(str(WORK / f"tprof-{arm}" / "**" / "*.json*"), recursive=True)]
    by, cls, n_ev, span = {}, {}, 0, [None, None]
    for f in files:
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rt") as h:
            trace = json.load(h)
        for e in trace.get("traceEvents", []):
            if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
                continue
            n_ev += 1
            name, dur = e.get("name", "?"), float(e.get("dur", 0))
            by[name] = by.get(name, 0.0) + dur
            low = name.lower()
            c = next((k for k, keys in KCLASS if any(x in low for x in keys)), "other")
            cls[c] = cls.get(c, 0.0) + dur
            ts = float(e.get("ts", 0))
            span = [ts if span[0] is None else min(span[0], ts), ts + dur if span[1] is None else max(span[1], ts + dur)]
    total = sum(by.values())
    out = {"files": len(files), "kernels": n_ev, "gpu_busy_ms": round(total / 1e3, 1),
           "span_ms": round((span[1] - span[0]) / 1e3, 1) if span[0] is not None else None,
           "by_class_ms": {k: round(v / 1e3, 1) for k, v in sorted(cls.items(), key=lambda x: -x[1])},
           "top": [[k, round(v / 1e3, 2)] for k, v in sorted(by.items(), key=lambda x: -x[1])[:60]]}
    (folder / "kernels.json").write_text(json.dumps(out, indent=1))
    return {k: v for k, v in out.items() if k != "top"} | {"top": out["top"][:15]}


ORIGINALS = {}


def arm_overlay(record, stub, spec, arm):
    """serving_fast switches for one arm on the runtime the recorded server imports, from the original files (each
    arm starts from the unpatched punica file, whatever the arm before it applied)."""
    if stub:
        return {"stub": True, "spec": spec}
    sys.path.insert(0, "/opt/fe")
    import serving_fast_patch as P
    argv = record["orig_argv"] or [record["executable"]]
    site = subprocess.run([argv[0], "-c", "import os, importlib.util as u; print(os.path.dirname(os.path.dirname("
                           "u.find_spec('vllm').origin)))"], env=record["env"], capture_output=True, text=True,
                          timeout=600).stdout.strip().splitlines()[-1]
    for pin, _, _ in P.plan(P.validate(spec)):
        f = Path(site) / pin["relative"]
        ORIGINALS.setdefault(str(f), f.read_bytes())
    for f, data in ORIGINALS.items():
        Path(f).write_bytes(data)
    return {"site": site, "identity": P.apply(site, str(WORK / f"fast-{arm}"), spec)}


def _kclass(name):
    low = name.lower()
    if "fused_moe_lora" in low or "moe_lora" in low or low.startswith("_arc_moe"):  # + MFU-1's stage-2 kernels
        return "moe_lora"
    if "lora" in low or "sgmv" in low or "bgmv" in low:
        return "dense_lora"
    return next((k for k, keys in KCLASS if any(x in low for x in keys)), "other")


def step_split(arm, folder):
    """Per engine step (worker annotation execute_context_A(B)_generation_C(D)): GPU busy / idle from the step's first
    kernel to the next step's first kernel, kernel time by class, graph vs eager launches, host syncs."""
    import glob
    import gzip
    import re as _re
    import shutil
    files = [f for f in glob.glob(str(WORK / f"tprof-{arm}" / "**" / "*.json*"), recursive=True)]
    out = {"files": len(files)}
    if not files:
        return out
    f = max(files, key=os.path.getsize)  # the worker trace
    try:
        shutil.copy(f, folder / ("trace" + ("".join(Path(f).suffixes[-2:]))))
    except OSError:
        pass
    ev = json.load((gzip.open if f.endswith(".gz") else open)(f, "rt")).get("traceEvents", [])
    steps, launch, gpu = [], {}, []
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat, name = e.get("cat", ""), e.get("name", "")
        if cat == "user_annotation" and name.startswith("execute_context_"):
            m = _re.match(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", name)
            steps.append({"t0": e["ts"], "t1": e["ts"] + e.get("dur", 0), "ctx": int(m.group(1)) if m else -1,
                          "ctx_tok": int(m.group(2)) if m else -1, "gen": int(m.group(4)) if m else -1})
        elif cat in ("cuda_runtime", "cuda_driver"):
            launch[(e.get("args") or {}).get("correlation")] = (e["ts"], name, e.get("dur", 0))
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu.append((e["ts"], e.get("dur", 0), name, (e.get("args") or {}).get("correlation"), cat))
    steps.sort(key=lambda s: s["t0"])
    starts = [s["t0"] for s in steps]
    import bisect
    for s in steps:
        s.update(k=[], graphs=0, launches=0, syncs=0, sync_us=0.0)
    for corr, (ts, name, dur) in launch.items():
        i = bisect.bisect_right(starts, ts) - 1
        if i < 0 or ts > steps[i]["t1"]:
            continue
        s = steps[i]
        if name == "cudaGraphLaunch":
            s["graphs"] += 1
        elif "LaunchKernel" in name or name.startswith("cuLaunch"):
            s["launches"] += 1
        elif "Synchronize" in name or name in ("cudaMemcpy", "cudaStreamWaitEvent") or "MemcpyAsync" in name:
            if "Synchronize" in name or name == "cudaMemcpy":
                s["syncs"] += 1
                s["sync_us"] += dur
    for ts, dur, name, corr, cat in gpu:
        lt = launch.get(corr)
        if lt is None:
            continue
        i = bisect.bisect_right(starts, lt[0]) - 1
        if i >= 0 and lt[0] <= steps[i]["t1"]:
            steps[i]["k"].append((ts, dur, name, cat))
    rows = []
    with_k = [s for s in steps if s["k"]]
    for j, s in enumerate(with_k[:-1]):
        k = sorted(s["k"])
        g0, nxt = k[0][0], min(x[0] for x in with_k[j + 1]["k"])
        busy, cur_s, cur_e, cls = 0.0, None, None, {}
        for ts, dur, name, cat in k:
            c = "copy" if cat != "kernel" else _kclass(name)
            cls[c] = cls.get(c, 0.0) + dur
            if cur_e is None or ts > cur_e:
                if cur_e is not None:
                    busy += cur_e - cur_s
                cur_s, cur_e = ts, ts + dur
            else:
                cur_e = max(cur_e, ts + dur)
        busy += cur_e - cur_s
        wall = nxt - g0
        rows.append({"kind": "decode" if s["ctx"] == 0 else "mixed", "gen": s["gen"], "ctx_tok": s["ctx_tok"],
                     "wall_ms": wall / 1e3, "busy_ms": busy / 1e3, "idle_ms": max(0.0, wall - busy) / 1e3,
                     "graphs": s["graphs"], "launches": s["launches"], "syncs": s["syncs"],
                     "sync_ms": s["sync_us"] / 1e3, "cls_ms": {c: v / 1e3 for c, v in cls.items()}})
    for kind in ("decode", "mixed"):
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        mean = lambda key: round(sum(r[key] for r in rs) / len(rs), 3)
        cls = {}
        for r in rs:
            for c, v in r["cls_ms"].items():
                cls[c] = cls.get(c, 0.0) + v / len(rs)
        out[kind] = {"steps": len(rs), "gen_tokens": mean("gen"), "ctx_tokens": mean("ctx_tok"),
                     "wall_ms": mean("wall_ms"), "busy_ms": mean("busy_ms"), "idle_ms": mean("idle_ms"),
                     "graph_launches": mean("graphs"), "kernel_launches": mean("launches"), "syncs": mean("syncs"),
                     "sync_ms": mean("sync_ms"),
                     "cls_ms": {c: round(v, 3) for c, v in sorted(cls.items(), key=lambda x: -x[1])}}
    out["annotated_steps"] = len(steps)
    (folder / "steps.json").write_text(json.dumps({"summary": out, "rows": rows}, indent=1))
    return out


CAPTURE_SIZES = [1, 2, 4] + list(range(8, 153, 8)) + [168, 184, 200, 224, 256, 288, 320, 384, 448, 512, 576, 640, 768,
                                                     896, 1024]


def capture_extra():
    """Piecewise graphs for mixed batches up to 1,024 tokens (FULL decode graphs stay capped at max_num_seqs=152;
    GDN metadata is UNIFORM_BATCH, so FULL cannot serve mixed batches)."""
    return ["--compilation-config", json.dumps({"cudagraph_capture_sizes": CAPTURE_SIZES}, separators=(",", ":")),
            "--max-cudagraph-capture-size", str(CAPTURE_SIZES[-1])]


def graph_facts(log):
    import re
    if not log.is_file():
        return {}
    t = log.read_text(errors="replace")
    out = {}
    for k, pat in (("compile_s", r"torch\.compile took ([0-9.]+) s"), ("kv_tokens", r"GPU KV cache size: ([0-9,]+) tokens"),
                   ("init_engine_s", r"init engine \(profile, create kv cache, warmup model\) took ([0-9.]+) s")):
        m = re.findall(pat, t)
        if m:
            out[k] = float(m[-1].replace(",", ""))
    m = re.findall(r"Graph capturing finished in ([0-9]+) secs, took ([0-9.]+) GiB", t)
    if m:
        out["capture_s"], out["graph_gib"] = float(m[-1][0]), float(m[-1][1])
    out["capture_counts"] = dict(re.findall(r"Capturing CUDA graphs \((\w+)\): 100%\|[^|]*\| (\d+)/", t))
    out["aot_loaded"] = t.count("Directly load AOT compilation")
    out["aot_saved"] = t.count("saved AOT compiled function")
    m = re.findall(r"Dynamo bytecode transform time: ([0-9.]+) s", t)
    out["dynamo_s"] = float(m[-1]) if m else None
    return out


AOT_ROOTS = (Path("/tmp/qwen38-flash-next-vllm-cache"), Path("/tmp/qwen38-flash-next-vllm-compile-cache"))


def _boot_cache():
    spec = importlib.util.spec_from_file_location("fe_boot_cache", "/opt/fe/boot_cache_new.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aot_wipe(kernels):
    """The fingerprinted vLLM roots (AOT graphs + their Inductor caches); with ``kernels`` also the Triton/Inductor/CUDA
    kernel caches and bytecode (a boot-cache key miss today)."""
    cache, comp = AOT_ROOTS
    paths = [cache / "vllm-env"] + ([comp / d for d in ("triton", "torchinductor", "cuda")] + [cache / "pyc"]
                                    if kernels else [])
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)
    return [str(p) for p in paths]


def aot_dirs():
    return sorted(str(p.relative_to(AOT_ROOTS[0])) for p in
                  AOT_ROOTS[0].glob("vllm-env/*/torch_compile_cache/torch_aot_compile/*"))


def aot_arm(kind):
    """MFU-2 AOT carry read (25 Oct). J1/J3: archive the roots as a boot ends (boot_cache.save), wipe everything, carry
    from that archive as a key miss would (boot_cache.carry). J2: wipe the AOT graphs only (today's boot on the other
    B300 SKU: kernel caches warm, AOT hash missing). J0: wipe all (today's key miss)."""
    bc = _boot_cache()
    for root in AOT_ROOTS:
        root.mkdir(parents=True, exist_ok=True)
    rows = []
    out = {"before": aot_dirs()}
    if kind in ("J1", "J3"):
        snap = WORK / "aotsnap"
        shutil.rmtree(snap, ignore_errors=True)
        bc.save(snap, AOT_ROOTS, emit=lambda r, **k: rows.append(json.loads(r)), cache_key="snap")
        out["wiped"] = aot_wipe(True)
        bc.carry(snap, AOT_ROOTS, emit=lambda r, **k: rows.append(json.loads(r)))
    else:
        out["wiped"] = aot_wipe(kind == "J0")
    out.update(rows=rows, after=aot_dirs())
    return out


def relaunch_argv(record, argv, extra, env_mod, log):
    rec = dict(record)
    rec["orig_argv"] = argv
    return relaunch(rec, extra, env_mod, log)


def read_jsonl(path):
    out = []
    if Path(path).is_file():
        for line in open(path):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def gauge(folder, lo, hi, name="vllm:num_requests_running"):
    return [m["m"][name] for m in read_jsonl(folder / "metrics.jsonl") if "m" in m and lo <= m["t"] <= hi
            and name in m["m"]]


def live_stats(folder, ws):
    recs = [r for f in folder.glob("drv-*.jsonl") for r in read_jsonl(f)]
    last = [r for r in recs if r["t1"] > time.time() - 60 and r["status"] == 200]
    run = gauge(folder, time.time() - 60, time.time() + 1)
    return {"live": {"done_60s": len(last), "errors_total": sum(r["status"] != 200 for r in recs),
                     "done_total": len(recs), "gen_60s": sum(r.get("ct") or 0 for r in last),
                     "engine_running_60s": round(sum(run) / len(run), 1) if run else None,
                     "warm": time.time() < ws}}


def window_stats(folder, ws, we):
    span = we - ws
    recs = [r for f in folder.glob("drv-*.jsonl") for r in read_jsonl(f)]
    done = [r for r in recs if ws <= r["t1"] <= we and r["status"] == 200]
    errs = [r for r in recs if ws <= r["t1"] <= we and r["status"] != 200]
    # client in flight: integral over the window of open requests
    infl = sum(max(0, min(r["t1"], we) - max(r["t0"], ws)) for r in recs) / span
    met = [m for m in read_jsonl(folder / "metrics.jsonl") if "m" in m and ws - 6 <= m["t"] <= we + 6]
    out = {"window_s": round(span), "requests": len(done), "errors": len(errs),
           "error_samples": [e.get("err") for e in errs[:3]],
           "client_inflight_mean": round(infl, 1),
           "client_gen_tok_s": round(sum(r["ct"] or 0 for r in done) / span, 1),
           "client_prompt_tok_s": round(sum(r["pt"] or 0 for r in done) / span, 1),
           "client_computed_prompt_tok_s": round(sum((r["pt"] or 0) - (r["cached"] or 0) for r in done) / span, 1),
           "client_hit": round(sum(r["cached"] or 0 for r in done) / max(1, sum(r["pt"] or 0 for r in done)), 4),
           "client_e2e_mean_s": round(sum(r["t1"] - r["t0"] for r in done) / max(1, len(done)), 2),
           "len_ok": sum((r["ct"] == r["ct_want"]) for r in done)}
    run, wait, kv = gauge(folder, ws, we), gauge(folder, ws, we, "vllm:num_requests_waiting"), \
        gauge(folder, ws, we, "vllm:kv_cache_usage_perc")
    if run:
        out.update(engine_running_mean=round(sum(run) / len(run), 1), engine_running_p90=pct(run, 0.9),
                   engine_running_p10=pct(run, 0.1), engine_waiting_mean=round(sum(wait) / max(1, len(wait)), 1),
                   kv_mean=round(sum(kv) / max(1, len(kv)), 3), kv_p90=pct(kv, 0.9), kv_max=max(kv) if kv else None,
                   gauge_samples=len(run))
    if len(met) >= 2:
        a, b = met[0]["m"], met[-1]["m"]
        dt = met[-1]["t"] - met[0]["t"]
        d = lambda k: b.get(k, 0) - a.get(k, 0)
        mean = lambda h: round(d(h + "_sum") / d(h + "_count"), 3) if d(h + "_count") else None
        ttft, queue, prefill = (mean("vllm:time_to_first_token_seconds"), mean("vllm:request_queue_time_seconds"),
                                mean("vllm:request_prefill_time_seconds"))
        if d("vllm:spec_decode_num_draft_tokens_total"):
            out["spec_acceptance"] = round(d("vllm:spec_decode_num_accepted_tokens_total") /
                                           d("vllm:spec_decode_num_draft_tokens_total"), 4)
            if d("vllm:spec_decode_num_drafts_total"):  # tokens per target step: 1 + accepted per draft
                out["spec_tokens_per_step"] = round(1 + d("vllm:spec_decode_num_accepted_tokens_total") /
                                                    d("vllm:spec_decode_num_drafts_total"), 3)
        steps = d("vllm:iteration_tokens_total_count")
        if steps:
            out.update(engine_steps_s=round(steps / dt, 2), mean_step_ms=round(1000 * dt / steps, 1),
                       gen_per_step=round(d("vllm:generation_tokens_total") / steps, 1),
                       tok_per_step=round(d("vllm:iteration_tokens_total_sum") / steps, 1))
        out["preemptions"] = d("vllm:num_preemptions_total")
        out.update(metrics_gen_tok_s=round(d("vllm:generation_tokens_total") / dt, 1),
                   metrics_prompt_tok_s=round(d("vllm:prompt_tokens_total") / dt, 1),
                   ttft_s=ttft, queue_s=queue, prefill_s=prefill,
                   pre_engine_s=round(ttft - queue - prefill, 3) if None not in (ttft, queue, prefill) else None,
                   decode_s=mean("vllm:request_decode_time_seconds"),
                   e2e_server_s=mean("vllm:e2e_request_latency_seconds"),
                   per_stream_decode_tok_s=(round(d("vllm:request_generation_tokens_sum") /
                                                  d("vllm:request_decode_time_seconds_sum"), 2)
                                            if d("vllm:request_decode_time_seconds_sum") else None),
                   metric_names=sorted(b)[:80])
    cpu = [c for c in read_jsonl(folder / "cpu.jsonl") if ws <= c["t"] <= we]
    if cpu:
        keys = {k for c in cpu for k in c["cats"]}
        out["cpu_pct"] = {k: round(sum(c["cats"].get(k, 0) for c in cpu) / len(cpu), 1) for k in sorted(keys)}
        pk = {k for c in cpu for k in c["per"]}
        out["cpu_pct_per_process"] = {k: round(sum(c["per"].get(k, 0) for c in cpu) / len(cpu), 1)
                                      for k in sorted(pk)}
    return out


BOOT_STUDY = "configs/boot-comparison.json"
SHADOW = "/kaggle/input-shadow"
KAGGLE_INPUT, VOLUME_INPUT = "/kaggle/input", "/assets/flash-input"
SITE_FILES = {}  # relative -> {"on": bytes, "off": bytes} (boot mode toggles)


def boot_setup(source, status, stub):
    """Boot mode: mfu2-boot src over the bundle, a study with pin_overlap + mm_warmup_overlap, /kaggle/input as a
    switchable symlink (shadow tree with local shard copies, or the volume), local staging with a reader check."""
    import hashlib
    import random
    import shutil
    shutil.copytree("/opt/boot-src", source / "src", dirs_exist_ok=True)
    study = json.loads((source / "configs/comparison.json").read_text())
    study["server"]["serving_kernels"] = {**study["server"]["serving_kernels"], "pin_overlap": True}
    study["server"]["serving_fast"] = {"mm_warmup_overlap": True}
    (source / BOOT_STUDY).write_text(json.dumps(study, indent=1))

    def mem():
        info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        du = shutil.disk_usage("/tmp")
        return {"mem_free_gib": round(int(info["MemFree"].split()[0]) / 2**20, 1),
                "mem_avail_gib": round(int(info.get("MemAvailable", info["MemFree"]).split()[0]) / 2**20, 1),
                "tmp_free_gib": round(du.free / 2**30, 1), "tmp_total_gib": round(du.total / 2**30, 1),
                "tmp_fs": next((l.split()[2] for l in open("/proc/mounts") if l.split()[1] in ("/tmp", "/")), None)}
    shadow = M.shadow_input(target=SHADOW)
    Path("/kaggle/input").symlink_to(SHADOW, target_is_directory=True)
    st = status["staging"] = {"before": mem()}
    done = threading.Event()

    def stage():
        st["row"] = M.stage_local_weights({}, 16, shadow, limit=2 if stub else None)
        st["after"] = mem()
        done.set()

    def readers():  # open shards through the shadow path while links switch: complete bytes either way
        names = sorted(p.name for p in Path(M.MODEL_ROOT).glob("layer-*.safetensors"))[: (2 if stub else 400)]
        checks = st["reader_checks"] = []
        rnd = random.Random(1)
        while not done.is_set() and len(checks) < (2 if stub else 24):
            name = rnd.choice(names)
            link = Path(shadow) / name
            via = "local" if "/tmp/" in os.readlink(link) else "volume"
            h1 = hashlib.sha256(link.read_bytes()).hexdigest()
            h2 = hashlib.sha256((Path(M.MODEL_ROOT) / name).read_bytes()).hexdigest()
            checks.append({"name": name, "opened_via": via, "equal": h1 == h2,
                           "via_after": "local" if "/tmp/" in os.readlink(link) else "volume"})
    threading.Thread(target=stage, daemon=True).start()
    threading.Thread(target=readers, daemon=True).start()
    return done


def load_setup(source):
    """Load mode: mfu2-boot src over the bundle; the study with (c) mm_warmup_overlap, (d) pin_overlap and the batched
    expert loading patch (on/off per boot through ARC_FAST_EXPERT_LOAD); a parameter digest after each load."""
    import shutil
    shutil.copytree("/opt/boot-src", source / "src", dirs_exist_ok=True)
    study = json.loads((source / "configs/comparison.json").read_text())
    study["server"]["serving_kernels"] = {**study["server"]["serving_kernels"], "pin_overlap": True}
    study["server"]["serving_fast"] = {"mm_warmup_overlap": True, "expert_load_fast": True}
    (source / BOOT_STUDY).write_text(json.dumps(study, indent=1))
    os.environ["FE_DIGEST"] = "1"


def boot_toggle(record, on, stub, site=None):
    """Switches on/off between boots: /kaggle/input -> shadow or volume; the renderer/AsyncLLM overlay; pin_overlap
    in the serving-kernels settings."""
    tmp = Path(KAGGLE_INPUT + ".tmp")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(SHADOW if on else VOLUME_INPUT, target_is_directory=True)
    os.replace(tmp, KAGGLE_INPUT)
    if stub:
        return {"input": os.readlink(KAGGLE_INPUT), "stub": True}
    import hashlib
    sys.path.insert(0, os.environ.get("FE_BOOT_SRC", "/opt/arc-validation/src"))
    from arc_infra import serving_fast_patch as fp
    if site is None:
        argv = record["orig_argv"] or [record["executable"]]
        site = subprocess.run([argv[0], "-c", "import os, importlib.util as u; print(os.path.dirname(os.path.dirname(u.find_spec('vllm').origin)))"],
                              env=record["env"], capture_output=True, text=True, timeout=600).stdout.strip().splitlines()[-1]
    site = Path(site)
    if not SITE_FILES:
        for pin, edits in ((fp.RENDERER, fp.WARMUP_EDITS), (fp.ASYNC_LLM, fp.ASYNC_EDITS)):
            patched = (site / pin["relative"]).read_bytes()
            orig = patched.decode()
            for old, new in edits:
                assert orig.count(new) == 1
                orig = orig.replace(new, old)
            orig = orig.encode()
            assert hashlib.sha256(orig).hexdigest() == pin["sha256"], pin["relative"]
            SITE_FILES[pin["relative"]] = {"on": patched, "off": orig}
        settings = json.loads((site / "vllm/arc_serving_kernels.json").read_text())
        assert settings.get("pin_overlap") is True
        off = {k: v for k, v in settings.items() if k != "pin_overlap"}
        SITE_FILES["vllm/arc_serving_kernels.json"] = {
            "on": (site / "vllm/arc_serving_kernels.json").read_bytes(),
            "off": (json.dumps(off, sort_keys=True) + "\n").encode()}
    for rel, v in SITE_FILES.items():
        (site / rel).write_bytes(v["on" if on else "off"])
    return {"input": os.readlink(KAGGLE_INPUT), "site": str(site),
            "files": {rel: hashlib.sha256((site / rel).read_bytes()).hexdigest()[:12] for rel in SITE_FILES}}


def greedy_probe(picks, model="policy", max_tokens=48):
    """Fixed prompts (R12 requests with images), greedy, one at a time after a prefix-cache reset: token ids and the
    sampled tokens' decode logprobs."""
    import httpx
    out = []
    with httpx.Client(timeout=900) as c:
        for i, p in enumerate(picks):
            c.post(ROOT + "/reset_prefix_cache")
            body = {"model": model, "messages": p["messages"], "tools": p.get("tools"),
                    "chat_template_kwargs": p.get("chat_template_kwargs") or {}, "temperature": 0.0,
                    "max_tokens": max_tokens, "logprobs": True, "top_logprobs": 0, "return_token_ids": True}
            r = c.post(ROOT + "/v1/chat/completions", json={k: v for k, v in body.items() if v is not None})
            row = {"i": i, "status": r.status_code}
            if r.status_code == 200:
                ch = r.json()["choices"][0]
                lp = (ch.get("logprobs") or {}).get("content") or []
                row.update(ids=ch.get("token_ids"), lp=[x.get("logprob") for x in lp])
            else:
                row["error"] = r.text[:300]
            out.append(row)
    return out


def greedy_compare(a, b):
    same, firsts, maxd, n = 0, [], 0.0, 0
    for x, y in zip(a, b):
        ia, ib = x.get("ids") or [], y.get("ids") or []
        k = next((j for j, (u, v) in enumerate(zip(ia, ib)) if u != v), min(len(ia), len(ib)))
        same += ia == ib and bool(ia)
        firsts.append(k)
        for u, v in zip((x.get("lp") or [])[:k], (y.get("lp") or [])[:k]):
            if u is not None and v is not None:
                maxd, n = max(maxd, abs(u - v)), n + 1
    first = (a[0].get("ids") == b[0].get("ids"), firsts[0] if firsts else None)
    return {"identical": same, "of": len(a), "first_divergence": firsts, "max_abs_dlp_matched": round(maxd, 5),
            "matched_positions": n, "first_request_identical": first[0]}


PHASES = {"pin": r'ARC_SK_HOST_TABLE (\{.*\})', "weights": r"Loading weights took ([\d.]+) seconds",
          "graphs": r"Graph capturing finished in ([\d.]+) secs", "init_engine": r"init engine .* took ([\d.]+) s",
          "mm_warmup": r"Multi-modal warmup completed in ([\d.]+)s"}


def boot_phases(log):
    import re
    import datetime
    text = Path(log).read_text(errors="replace") if Path(log).is_file() else ""
    out = {}
    for k, pat in PHASES.items():
        m = re.search(pat, text)
        if m:
            out[k] = json.loads(m.group(1)) if k == "pin" else float(m.group(1))
    stamps = {}
    for key, pat in (("engine_done", r"(\d\d-\d\d \d\d:\d\d:\d\d) \[core.py:\d+\] init engine"),
                     ("serving", r"(\d\d-\d\d \d\d:\d\d:\d\d) \[api_server.py:\d+\] Starting vLLM server"),
                     ("first", r"(\d\d-\d\d \d\d:\d\d:\d\d) \[")):
        m = re.search(pat, text)
        if m:
            stamps[key] = datetime.datetime.strptime(m.group(1), "%m-%d %H:%M:%S")
    if "engine_done" in stamps and "serving" in stamps:
        out["engine_to_serving_s"] = (stamps["serving"] - stamps["engine_done"]).total_seconds()
    if "first" in stamps and "serving" in stamps:
        out["log_first_to_serving_s"] = (stamps["serving"] - stamps["first"]).total_seconds()
    return out


def write_final(dest, status):
    """Clean end (phase done): final.json plus the runops END_MARKER line, once, so watch.py sees a clean end."""
    if status.get("phase") != "done" or status.get("final_written"):  # not the file: a rerun of a name finds the old one
        return False
    status["final_written"] = True
    (dest / "final.json").write_text(json.dumps({"state": "ok", "run": status.get("run"), "t": status.get("t"),
                                                 "total_s": status.get("total_s")}, indent=1))
    print(f"RUNOPS END ok run={status.get('run')}", flush=True)
    return True


class Status(dict):
    def __init__(self, dest):
        super().__init__()
        self.dest = dest
        self.lock = threading.Lock()

    def dump(self):
        with self.lock:
            self["t"] = round(time.time())
            (self.dest / "status.json").write_text(json.dumps(self, indent=1))
            write_final(self.dest, self)
            M.runs.commit()


    def gpu_sampler(self):
        """GPU util every 10 s; low_since = start of the current run of samples under 30% (watcher: 15 min idle)."""
        def loop():
            window, low_since, last_dump = [], None, time.time()
            while True:
                try:
                    u = float(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                             capture_output=True, text=True, timeout=30).stdout.split()[0])
                except Exception:  # noqa: BLE001 - CPU preflight has no GPU
                    u = None
                if u is not None:
                    window = (window + [u])[-6:]
                    low_since = (low_since or time.time()) if u < 30 else None
                self["gpu"] = {"util": u, "util_60s": round(sum(window) / len(window), 1) if window else None,
                               "low_since": low_since}
                if time.time() - last_dump >= 60:
                    self.dump()
                    last_dump = time.time()
                time.sleep(10)
        threading.Thread(target=loop, daemon=True).start()


@app.function(image=image, gpu="B300", timeout=int(os.environ.get("FE_TIMEOUT", "3600")), retries=0, max_containers=1, single_use_containers=True,
              volumes={"/assets": M.assets, "/runs": M.runs})
def bench(run: str, arms: str = "A,C", warm: int = 150, measure: int = 600, mode: str = "fe"):
    return body(run, arms, warm, measure, stub=False, mode=mode)


@app.function(image=image, timeout=3600, retries=0, max_containers=1, single_use_containers=True, cpu=(8, 16),
              volumes={"/assets": M.assets, "/runs": M.runs})
def bench_cpu(run: str, arms: str = "A,C", warm: int = 20, measure: int = 100, mode: str = "fe"):
    """CPU preflight of the same path: fake_server.py stands in for vLLM (OpenAI chat + /metrics + LoRA load), the
    relaunch replays its recorded command; the driver replays 6 staged attempts."""
    return body(run, arms, warm, measure, stub=True, mode=mode)


HOOKCHECK = """
import json, sys
mods = ["starlette.applications", "vllm.utils.async_utils", "vllm.renderers.base", "vllm.renderers.hf",
        "vllm.entrypoints.openai.chat_completion.serving", "vllm.v1.engine.async_llm",
        "vllm.v1.engine.output_processor", "vllm.v1.serial_utils", "vllm.v1.engine.core"]
errs = {}
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        errs[m] = repr(e)[:300]
fe_ts = sys.modules.get("fe_ts")
print("HOOKCHECK " + json.dumps({"patched": getattr(fe_ts, "PATCHED", None), "import_errors": errs}))
"""
TS_MODULES = 9


def hookcheck(record):
    """ts mode: import every hooked module in the recorded server's interpreter and environment; all must patch."""
    argv = record["orig_argv"] or [record["executable"]]
    env = {**record["env"], "FE_BENCH": "1", "FE_TS": "1", "FE_TS_DIR": str(WORK / "hookcheck")}
    p = subprocess.run([argv[0], "-c", HOOKCHECK], env=env, capture_output=True, text=True, timeout=900,
                       cwd=record.get("cwd") or "/")
    line = [x for x in p.stdout.splitlines() if x.startswith("HOOKCHECK ")]
    out = json.loads(line[-1][10:]) if line else {"rc": p.returncode, "stderr": p.stderr[-800:]}
    out["ok"] = sum(v is True for v in (out.get("patched") or {}).values()) == TS_MODULES
    return out


def body(run, arms, warm, measure, stub, mode="fe"):
    import shutil
    t0 = time.time()
    ts = mode == "ts"  # tput arms with fe_ts timestamps on (FE_TS=1), no decode checks
    if ts:
        mode = "tput"
        os.environ["FE_TS"] = "1"
    if stub:
        STUB.append(1)
    source = Path("/opt/arc-validation")
    dest = Path("/runs/validation/mfu2-fe") / (f"preflight-{run}" if stub else run)
    dest.mkdir(parents=True, exist_ok=True)
    status = Status(dest)
    status.update(run=run, arms=arms, stub=stub, host_cpu=M.host_cpu(), phase="setup", t0=t0)
    status.dump()
    status.gpu_sampler()
    WORK.mkdir(parents=True, exist_ok=True)
    Path("/kaggle").mkdir(exist_ok=True)
    boot_done = None
    if mode == "boot":
        boot_done = boot_setup(Path("/opt/arc-validation"), status, stub)
    elif mode == "load":
        load_setup(Path("/opt/arc-validation"))
        Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    elif not Path("/kaggle/input").exists():
        Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    os.environ.update(PYTHONPATH=f"/opt/fe-site:{source}/src", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", PIP_NO_INDEX="1", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1",
                      ARC_FORCE_CHROOT="1", FE_BENCH="1")
    os.environ["LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", "")
    boot = M.boot_settings(source)
    if boot.get("expert_mask_skip_read"):  # tput10: the run's own server layout (modal_rl1 sets these for the loop)
        os.environ["ARC_EXPERT_MASK_SKIP_READ"] = "1"
    if boot.get("expert_mask_compact"):
        os.environ["ARC_EXPERT_MASK_COMPACT"] = "1"
    if not stub:
        if boot.get("prefetch_threads") and mode not in ("boot", "load"):
            threading.Thread(target=M.prefetch_weights, args=({}, boot["prefetch_threads"]), daemon=True).start()
        key = M.compile_key(source)
        M.compile_restore(key, status, keep=True)
        if boot.get("runtime_cache"):
            os.environ["ARC_RUNTIME_CACHE"] = "/runs/runtime-cache"
    sys.path.insert(0, str(source))
    from bootstrap import install_agent_environment
    env = dict(os.environ)
    python = install_agent_environment(source, Path("/tmp/arena/agent-venv"), env)
    status["venv_s"] = round(time.time() - t0, 1)
    if mode == "selftest":  # the Q GPU self-test's exact launch in the real runtime, on CPU
        return selftest_cpu(python, status)
    adapter = WORK / "adapter"
    shutil.copytree(f"{DATA}/adapter-r11", adapter, dirs_exist_ok=True)
    if stub:
        import glob
        sd = WORK / "stubdata" / "r12-stub-L1"
        sd.mkdir(parents=True, exist_ok=True)
        for f in sorted(glob.glob(f"{DATA}/r12-sc25-L1/*.jsonl"))[:4] + sorted(glob.glob(f"{DATA}/r12-sc25-L3/*.jsonl"))[:2]:
            shutil.copy(f, sd / Path(f).name)
    study = json.loads((source / "configs/comparison.json").read_text())
    extra = {str(k): str(v) for k, v in (study["trials"][0].get("server_env") or {}).items()}
    if any(tput_arm(a)[0] in ("J0", "J1", "J2", "J3") for a in arms.split(",")):  # serving_setup.aot_root_env's root
        extra["VLLM_CACHE_ROOT"] = str(_boot_cache().vllm_cache_root(AOT_ROOTS[0], {**env, **extra}))
        status["vllm_cache_root"] = extra["VLLM_CACHE_ROOT"]
    picks = exact_prompts()
    rpicks = reply_prompts()
    keep = KEEP
    results = {}
    record = None
    for arm in arms.split(","):
        if ts and not stub and status.get("hookcheck") and not status["hookcheck"]["ok"]:
            break  # no timestamps: nothing to measure
        folder = dest / arm
        folder.mkdir(parents=True, exist_ok=True)
        row = results[arm] = {"arm": arm}
        status.update(phase=f"{arm}-boot")
        status.dump()
        began = time.time()
        proc = None
        try:
            if record is None:  # the first arm: today's launch path
                if stub:
                    subprocess.Popen([python, "/opt/fe/fake_server.py", "serve", "--served-model-name", SERVED],
                                     env={**env, "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"},
                                     stdout=open(folder / "server.log", "w"), stderr=subprocess.STDOUT,
                                     start_new_session=True)
                    row["ready_s"] = wait_ready()
                else:
                    study_path = source / (BOOT_STUDY if mode in ("boot", "load") else "configs/comparison.json")
                    subprocess.run([python, "-m", "arena.serving", "--study", str(study_path),
                                    "--working", "/tmp/arena/serving", "--input-root", "/kaggle/input"],
                                   env={**env, **extra}, check=True, timeout=2400)
                    row["ready_s"] = wait_ready()
                record = serve_record()
                if record is None:
                    raise RuntimeError("no serve record (sitecustomize did not run in the server)")
                (dest / "serve-record.json").write_text(json.dumps(
                    {k: v for k, v in record.items() if k != "env"} | {"env_keys": sorted(record["env"])}, indent=1))
                if ts:
                    status["hookcheck"] = hookcheck(record)
                    status.dump()
                    if not status["hookcheck"]["ok"] and not stub:  # the stub server's venv has no vLLM
                        raise RuntimeError(f"fe_ts hooks incomplete: {json.dumps(status['hookcheck'])[:600]}")
            elif mode == "tput":
                kind = tput_arm(arm)[0]
                argv = record["orig_argv"] or [record["executable"], *record["argv"]]
                pextra = prof_extra(arm) if kind.endswith(("P", "S")) else []
                if kind in ("B", "BP", "BS"):
                    proc = relaunch_argv(record, argv_without_lora(argv), pextra,
                                         {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": None}, WORK / f"{arm}-server.log")
                elif re.fullmatch(r"Y\d+", kind):  # MFU-1 (25 Oct): --mm-processor-cache-gb N; before M ("MC4" crashed)
                    row["mm_cache_gb"] = int(kind[1:])
                    proc = relaunch(record, ["--mm-processor-cache-gb", kind[1:]], {}, WORK / f"{arm}-server.log")
                elif kind.startswith("M"):
                    row["overlay"] = apply_fast(record, stub, {"mtp_lora_draft": True})
                    proc = relaunch(record, mtp_extra(argv, int(kind[1:])), {}, WORK / f"{arm}-server.log")
                elif kind in ("C", "CF"):  # lora_meta_cpu (C), plus lora_fused_dense (CF)
                    spec = {"lora_meta_cpu": True, **({"lora_fused_dense": True} if kind == "CF" else {})}
                    row["overlay"] = arm_overlay(record, stub, spec, arm)
                    proc = relaunch(record, [], {}, WORK / f"{arm}-server.log")
                elif kind in ("G", "GR", "GO"):  # capture range to 1,024 (GR: warm relaunch, boot only; GO: occupancy)
                    proc = relaunch(record, capture_extra(), {}, WORK / f"{arm}-server.log")
                elif kind == "GM":  # capture range + serving_fast.moe_lora_fast
                    row["overlay"] = arm_overlay(record, stub, {"moe_lora_fast": True}, arm)
                    proc = relaunch(record, capture_extra(), {}, WORK / f"{arm}-server.log")
                elif re.fullmatch(r"TG[ab]\d+", kind):  # API-process GC A/B (fe_gc): a = gen0 50,000 with automatic
                    # full collections kept, b = gen0 50,000, no automatic full collections, gc.collect() every 10 s
                    tokens = int(kind[3:]) * 1024
                    row["max_num_batched_tokens"] = tokens
                    row["fe_gc"] = {"a": "50000,10,10,0", "b": "50000,20,1000000000,10"}[kind[2]]
                    proc = relaunch(record, ["--max-num-batched-tokens", str(tokens)], {"FE_GC": row["fe_gc"]},
                                    WORK / f"{arm}-server.log")
                elif re.fullmatch(r"T\d+R?", kind):  # max_num_batched_tokens = N K (T8R: a repeat of T8), the record's
                    # argv otherwise (its capture range included); the later flag wins in vLLM's parser
                    tokens = int(re.match(r"T(\d+)", kind).group(1)) * 1024
                    row["max_num_batched_tokens"] = tokens
                    proc = relaunch(record, ["--max-num-batched-tokens", str(tokens)], {}, WORK / f"{arm}-server.log")
                elif kind in ("J0", "J1", "J2", "J3"):  # AOT carry read (aot_arm): today's record, caches per kind
                    row["aot"] = aot_arm(kind)
                    proc = relaunch(record, [], {}, WORK / f"{arm}-server.log")
                elif kind in ("W0", "W1", "W2", "W3"):  # level-1 sleep read: W0 stock, W1/W2 sleep_keep_kv_patch (KV to
                    # host); W2 skips the LoRA reload after wake; W3 keep_lora read (two adapters, no reload)
                    row["pristine"] = restore_pristine(record, stub)
                    if kind in ("W1", "W2", "W3"):
                        row["overlay"] = sleep_patch(record, stub)
                        if not stub and not (row["overlay"] or {}).get("identity"):
                            raise RuntimeError(f"sleep_keep_kv patch did not apply: {row['overlay']}")
                    kvb = None if stub else os.environ.get("FE_SLEEP_KV_BYTES", "128687957606")  # hero server 2 KV pool
                    proc = relaunch(record, ["--kv-cache-memory-bytes", kvb] if kvb else [],
                                    {"ARC_SLEEP_KEEP_KV": "1" if kind in ("W1", "W2", "W3") else None,
                                     "VLLM_SERVER_DEV_MODE": "1", "ARC_LORA_SEGMENT": None,
                                     "ARC_SLEEP_VERIFY": "1" if "verify" in str(status.get("run")) else None},
                                    WORK / f"{arm}-server.log")
                elif re.fullmatch(r"[LQFHK]\d[SPX]?", kind):  # n adapters round-robin; Q: lora_segment_patch on,
                    # F: serving_fast.lora_fused_dense (decode-size dense LoRA in one kernel), segment off
                    row["pristine"] = restore_pristine(record, stub)  # all patched runtime files, every arm
                    if "gpu_test" not in SEGMENT and any(tput_arm(x)[0][:1] == "Q" for x in arms.split(",")):
                        # first arm after R-0: a harness failure costs one arm, not a Q boot's worth of B300 time
                        SEGMENT["gpu_test"] = segment_gpu_test(record, stub)
                        row["gpu_test"] = SEGMENT["gpu_test"]
                        if not SEGMENT["gpu_test"].get("ok"):
                            raise RuntimeError(f"segment GPU unit test failed: {SEGMENT['gpu_test']}")
                    if kind[0] == "Q":  # every Q arm re-applies: the restore above undid the op-file edits
                        SEGMENT["segment"] = segment_patch(record, stub)
                    if kind[0] == "F":
                        row["overlay"] = dense_patch(record, stub)
                    moe_env = {}
                    if kind[0] in "HK":  # MFU-1 moe_lora_fast_patch: H = vLLM one-shot at BLOCK_M 16, K = + stage 2
                        row["overlay"] = moe_patch(record, stub)
                        moe_env = {"ARC_MOE_LORA_BM16": "1", "ARC_MOE_LORA_FAST": "1" if kind[0] == "K" else "0"}
                    row["segment"] = SEGMENT.get("segment")
                    # X: fault hunt (tput14) - segment debug (bounds asserts, sync around every eager LoRA op call,
                    # call/prepare log) and CUDA_LAUNCH_BLOCKING=1
                    dbg = {"ARC_LORA_SEGMENT_DEBUG": "1", "CUDA_LAUNCH_BLOCKING": "1"} if kind.endswith("X") else {}
                    proc = relaunch(record, pextra, {"ARC_LORA_SEGMENT": "1" if kind[0] == "Q" else "0", **moe_env,
                                                     **dbg},
                                    WORK / f"{arm}-server.log")
                    row["boot_ready_s"] = wait_ready(proc)
                    row["patch_check"] = patch_check(record, kind[0], WORK / f"{arm}-server.log", stub)
                elif kind == "D":  # vLLM's dual-stream LoRA: LoRA kernels on an aux stream beside the base GEMMs
                    proc = relaunch(record, [], {"VLLM_LORA_ENABLE_DUAL_STREAM": "1"}, WORK / f"{arm}-server.log")
                else:
                    proc = relaunch(record, pextra, {}, WORK / f"{arm}-server.log")
                row["ready_s"] = wait_ready(proc)
            else:
                if mode == "boot":  # N* = all three switches on, F* = off
                    row["toggle"] = boot_toggle(record, arm.startswith("N"), stub)
                if arm.startswith("C"):  # serving_fast overlay on the extracted runtime, then today's command
                    row["overlay"] = apply_fast(record, stub)
                if arm.startswith("M"):
                    spec = json.dumps({"method": "mtp", "num_speculative_tokens": 1}, separators=(",", ":"))
                    proc = relaunch(record, ["--speculative-config", spec], {}, WORK / f"{arm}-server.log")
                elif arm.startswith("B"):
                    lora = json.dumps({"name": "policy", "path": str(adapter), "base_model_name": SERVED,
                                       "is_3d_lora_weight": True})
                    proc = relaunch(record, ["--api-server-count", "2", "--lora-modules", lora],
                                    {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": None}, WORK / f"{arm}-server.log")
                else:  # load mode: N* = batched expert loading on, F* = off (same patched runtime)
                    env_mod = {"ARC_FAST_EXPERT_LOAD": "1" if arm.startswith("N") else "0"} if mode == "load" else {}
                    proc = relaunch(record, [], env_mod, WORK / f"{arm}-server.log")
                row["ready_s"] = wait_ready(proc)
            row["boot_wall_s"] = round(time.time() - began, 1)
            if mode == "tput" and tput_arm(arm)[0] in ("R", "GR"):  # record only: the launch path's command, nothing measured
                continue
            if not arm.startswith("B"):
                import httpx
                r = httpx.post(ROOT + "/v1/load_lora_adapter", json={"lora_name": "policy", "lora_path": str(adapter),
                                                                       "is_3d_lora_weight": True}, timeout=600)
                row["load_adapter"] = [r.status_code, r.text[:200]]
                for name in adapter_names(arm, mode)[1:]:
                    r = httpx.post(ROOT + "/v1/load_lora_adapter", json={"lora_name": name, "lora_path": str(adapter),
                                                                           "is_3d_lora_weight": True}, timeout=600)
                    row["load_adapter_" + name] = [r.status_code, r.text[:200]]
            row["models"] = models_seen()
            status.update(phase=f"{arm}-exact")
            status.dump()
            if mode == "load":
                g = greedy_probe(rpicks[:4])
                (folder / "greedy.json").write_text(json.dumps(g))
                row["greedy_ok"] = sum(x["status"] == 200 for x in g)
                continue
            if mode == "boot":
                if record is not None and "toggle" not in row:  # the first arm: record what it ran with
                    row["toggle"] = boot_toggle(record, arm.startswith("N"), stub)
                g = greedy_probe(rpicks)
                (folder / "greedy.json").write_text(json.dumps(g))
                row["greedy_ok"] = sum(x["status"] == 200 for x in g)
                continue
            if mode == "tput" and tput_arm(arm)[0] == "W3":
                status.update(phase=f"{arm}-keep-lora")
                status.dump()
                row["keep_lora"] = keep_lora_probe(adapter, stub)
                (dest / f"{arm}-keep-lora.json").write_text(json.dumps(row["keep_lora"], indent=1))
                continue
            if mode == "tput" and tput_arm(arm)[0] in ("W0", "W1", "W2"):
                kind = tput_arm(arm)[0]
                status.update(phase=f"{arm}-sleep")
                status.dump()
                log_text = (WORK / f"{arm}-server.log").read_text(errors="replace") if (WORK / f"{arm}-server.log").is_file() else ""
                site = Path(runtime_site(record, stub))
                row["patch_tag"] = all("ARC sleep_keep_kv" in (site / rel).read_text() for rel in
                                       ("vllm/device_allocator/sleep_mode_backend.py", "vllm/v1/engine/core.py",
                                        "vllm/device_allocator/cumem.py", "vllm/v1/worker/gpu_worker.py"))
                if row["patch_tag"] != (kind in ("W1", "W2")):
                    raise RuntimeError(f"sleep patch state wrong for {kind}: {row['patch_tag']}")
                row["sleep"] = sleep_probe(arm, kind != "W0", adapter, stub, cycles=1 if kind == "W0" else 2,
                                           reload=kind != "W2")
                log_text = (WORK / f"{arm}-server.log").read_text(errors="replace") if (WORK / f"{arm}-server.log").is_file() else ""
                row["keep_kv_logged"] = "ARC_SLEEP_KEEP_KV suspend" in log_text
                row["cumem_lines"] = [l[-220:] for l in log_text.splitlines() if "CuMemAllocator: sleep freed" in l][-3:]
                row["exact_pin_lines"] = [l[-220:] for l in log_text.splitlines() if "ARC_SLEEP_EXACT_PIN" in l][-4:]
                row["prepin_lines"] = [l[-220:] for l in log_text.splitlines() if "ARC_SLEEP_PREPIN" in l][-2:]
                row["verify_lines"] = [l[-220:] for l in log_text.splitlines() if "ARC_SLEEP_VERIFY checked" in l][-4:]
                (dest / f"{arm}-sleep.json").write_text(json.dumps(row["sleep"], indent=1))
                continue
            if mode == "tput":
                kind, streams, more = tput_arm(arm)
                model = SERVED if kind in ("B", "BP", "BS") else ",".join(adapter_names(arm, mode))
                status.update(phase=f"{arm}-decode")
                status.dump()
                if re.fullmatch(r"[LQFHK]\d[SPX]?", kind) and not ts:  # lp vs the first such arm (segment exactness)
                    first = next(a for a in arms.split(",") if re.fullmatch(r"[LQFHK]\d[SPX]?", tput_arm(a)[0]))
                    row["decode"] = decode_checks(arm, rpicks[:12] if not stub else rpicks[:2], "policy",
                                                  repeat=arm == first)
                if kind.endswith(("P", "S")):  # profiled arms: kernel split (P) or per-step split (S) under load
                    holder = {}
                    # the torch profile after the stats window: its trace dump stalls the engine (tput11: 2 min)
                    timer = threading.Timer(warm + measure + 5 if not stub else 5,
                                            lambda: holder.update(profile_window(arm)))
                    timer.start()
                    row["load"] = run_load(arm, folder, python, warm, measure, status, stub, streams=streams,
                                           model=model, no_min=True, extra=180 if not stub else 10)
                    timer.join()
                    row["profile_calls"] = holder
                    row["kernels"] = kernel_summary(arm, folder)
                    if kind.endswith("S"):
                        row["steps"] = step_split(arm, folder)
                    continue
                if kind == "GO" and not stub:  # occupancy levels on late-round (L3) replays only
                    os.environ["FE_DATA_GLOB"] = "r12-*L3"
                if kind not in ("B", "BP", "GO") and not re.fullmatch(r"Y\d+", kind) and not ts:  # B serves another model: no logprob comparison
                    first = next(a for a in arms.split(",") if tput_arm(a)[0] not in ("R", "GR"))
                    row["decode"] = decode_checks(arm, rpicks[:12] if not stub else rpicks[:2], model,
                                                  repeat=arm == first)
                if kind in ("J0", "J1", "J2", "J3"):  # boot read: ready time + decode only
                    continue
                row["load"] = run_load(arm, folder, python, warm, measure, status, stub, streams=streams, model=model,
                                       no_min=True)
                for n in more:
                    row[f"load_{n}"] = run_load(f"{arm}@{n}", dest / f"{arm}@{n}", python, warm, measure, status, stub,
                                                streams=n, model=model, no_min=True)
                continue
            if mode == "mtp":
                row["mtp"] = mtp_checks(arm, folder, rpicks, keep)
                if arm != "A2":
                    row["load"] = run_load(arm, folder, python, warm, measure, status, stub)
                continue
            ex = exactness(picks, "policy")
            (folder / "exact.json").write_text(json.dumps(ex))
            row["exact_ok"] = sum("lp" in x for x in ex)
            rep_ = exact_replies(rpicks, "policy")
            (folder / "replies.json").write_text(json.dumps(rep_))
            row["replies_ok"] = sum(x["status"] == 200 for x in rep_)
            if not arm.endswith("2"):
                row["load"] = run_load(arm, folder, python, warm, measure, status, stub)
        except Exception as exc:  # noqa: BLE001 - the record keeps what was measured
            row["error"] = f"{type(exc).__name__}: {exc}"[:1500]
        finally:
            log = Path("/tmp/arena/serving/vllm-openai-server.log")
            if arm == arms.split(",")[0] and log.is_file():
                shutil.copy(log, folder / "server.log")
            elif (WORK / f"{arm}-server.log").is_file():
                shutil.copy(WORK / f"{arm}-server.log", folder / "server.log")
            if mode == "tput":
                row["graph_facts"] = graph_facts(folder / "server.log")
            if mode in ("boot", "load"):
                row["phases"] = boot_phases(folder / "server.log")
            if mode == "load":
                digests = sorted(WORK.glob("digest-*.json"), key=lambda p: p.stat().st_mtime)
                if digests and digests[-1].name not in {r.get("digest_file") for r in results.values()}:
                    shutil.copy(digests[-1], folder / "digest.json")
                    row["digest_file"] = digests[-1].name
                    dj = json.loads(digests[-1].read_text())
                    row["load_weights_s"], row["digest_s"] = dj["load_weights_s"], dj["digest_s"]
                    row["digest_params"] = len(dj["params"])
                fl = re.findall(r"ARC_FAST_LOAD (\{.*\})", (folder / "server.log").read_text(errors="replace")) \
                    if (folder / "server.log").is_file() else []
                row["fast_load"] = json.loads(fl[-1]) if fl else None
            stop_server(row)
            for f in WORK.glob("ts-*.jsonl"):  # ts mode: this arm's timestamp rows (all its loads)
                shutil.copy(f, folder / f.name)
                f.unlink()
            if mode == "tput":  # the summary so far, in case the crash ceiling ends the run
                status["tput"] = summarize_tput(results, next((a for a in arms.split(",") if tput_arm(a)[0] not in ("R", "GR")), arms.split(",")[0]))
                (dest / "decode.json").write_text(json.dumps(DECODE))
            status.update(phase=f"{arm}-done", results=results)
            status.dump()
            (dest / "results.json").write_text(json.dumps(results, indent=1))
    if mode == "load":
        dig = {a: json.loads((dest / a / "digest.json").read_text())["params"] for a in results
               if (dest / a / "digest.json").is_file()}
        names = list(dig)
        status["digest_equal"] = {}
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                diff = sorted(k for k in set(dig[x]) | set(dig[y]) if dig[x].get(k) != dig[y].get(k))
                status["digest_equal"][f"{x}_vs_{y}"] = {"params": len(dig[x]), "differ": len(diff),
                                                         "first": diff[:5]}
        greedy = {a: json.loads((dest / a / "greedy.json").read_text()) for a in results
                  if (dest / a / "greedy.json").is_file()}
        gn = list(greedy)
        status["exact"] = {f"{x}_vs_{y}": greedy_compare(greedy[x], greedy[y])
                           for i, x in enumerate(gn) for y in gn[i + 1:]}
        status["phases"] = {a: r.get("phases") for a, r in results.items()}
        status.update(phase="done", total_s=round(time.time() - t0))
        status.dump()
        return {k: v for k, v in status.items() if k != "results"}
    if mode == "boot":
        boot_done.wait(timeout=600)
        greedy = {a: json.loads((dest / a / "greedy.json").read_text()) for a in results
                  if (dest / a / "greedy.json").is_file()}
        names = list(greedy)
        status["exact"] = {f"{x}_vs_{y}": greedy_compare(greedy[x], greedy[y])
                           for i, x in enumerate(names) for y in names[i + 1:]}
        status["phases"] = {a: r.get("phases") for a, r in results.items()}
        status.update(phase="done", total_s=round(time.time() - t0))
        status.dump()
        return {k: v for k, v in status.items() if k != "results"}
    if mode == "tput":
        status["tput"] = summarize_tput(results, next((a for a in arms.split(",") if tput_arm(a)[0] not in ("R", "GR")), arms.split(",")[0]))
        (dest / "decode.json").write_text(json.dumps(DECODE))
        status.update(phase="done", total_s=round(time.time() - t0))
        status.dump()
        return {k: v for k, v in status.items() if k != "results"}
    if mode == "mtp":
        status["mtp"] = summarize_mtp(results)
        status.update(phase="done", total_s=round(time.time() - t0))
        status.dump()
        return {k: v for k, v in status.items() if k != "results"}
    names = list(results)
    base = names[0]
    bex = json.loads((dest / base / "exact.json").read_text()) if (dest / base / "exact.json").is_file() else []
    status["exact_vs_" + base] = {}
    for arm in names[1:]:
        p = dest / arm / "exact.json"
        if p.is_file():
            status["exact_vs_" + base][arm] = compare(bex, json.loads(p.read_text()))
    status["replies_vs_" + base] = {}
    brep = json.loads((dest / base / "replies.json").read_text()) if (dest / base / "replies.json").is_file() else []
    for arm in names[1:]:
        p = dest / arm / "replies.json"
        if p.is_file():
            other = json.loads(p.read_text())
            status["replies_vs_" + base][arm] = {
                "identical": sum(x["sha256"] == y["sha256"] and x["status"] == 200 for x, y in zip(brep, other)),
                "of": len(brep)}
    status.update(phase="done", total_s=round(time.time() - t0))
    status.dump()
    return {k: v for k, v in status.items() if k != "results"}


@app.local_entrypoint()
def main(run: str, arms: str = "A,C", cpu: bool = False, warm: int = 0, measure: int = 0, mode: str = "fe"):
    fn = bench_cpu if cpu else bench
    kw = {k: v for k, v in (("warm", warm), ("measure", measure)) if v}
    kw["mode"] = mode
    # spawn, not remote: a gRPC deadline in this local client must not cancel the detached call (mtp2/mtp3 lost)
    call = fn.spawn(run, arms, **kw)
    print(json.dumps({"call_id": call.object_id, "run": run}), flush=True)
