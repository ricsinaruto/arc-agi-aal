"""MFU stage D check (RL arm, 21 Oct): one MoE layer's routed experts on one B200, the trainer's Axolotl dequant path
against ``native_experts`` (and Axolotl's W4A4 fp4_cute path): speed, peak memory and numerics against an FP32
reference (dequantised base + s B A), on the layer's real router over captured real hidden states.

    uv run --extra modal modal run --detach scripts/posttrain/modal_mfu_d.py --run-id mfu-d-v1 [--cfg '{"tokens": 32768}']

Output: arc-prolong-runs:/validation/mfu-d/<run id>/ (log.txt, result.json)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BODY = __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "body"
VENV_PY = "/workspace/axolotl-venv/bin/python"
MODEL = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
CAPTURE = "/runs/validation/b2-tp-quant-capture-all25v71b-48l-v1-nebius-20261010/expert-capture/eval"

if not BODY:
    import modal

    AXOLOTL_IMAGE = "axolotlai/axolotl@sha256:af1ff77a35170b794d5fd3d01d97d69fe11ba9a098bca43eabe0849ff3d2b010"
    RL_PIP = ("quack-kernels==0.6.1 nvidia-cutlass-dsl==4.6.0 apache-tvm-ffi>=0.1.10,<0.2", "--no-deps triton==3.6.0")
    image = (modal.Image.from_registry(AXOLOTL_IMAGE, add_python="3.12").entrypoint([])  # modal_rl1.image's layers
             .apt_install("libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "build-essential",
                          "ninja-build")
             .run_commands("uv pip install --python /workspace/axolotl-venv/bin/python 'py-spy==0.4.0'")
             .env({"PYTHONUNBUFFERED": "1"}))
    for command in RL_PIP:
        image = image.run_commands("uv pip install --python /workspace/axolotl-venv/bin/python "
                                   + " ".join(f"'{item}'" for item in command.split()))
    here = Path(__file__).resolve().parent if modal.is_local() else Path("/root")
    image = (image.add_local_file(str(here / "native_experts.py"), "/root/native_experts.py", copy=True)
             .add_local_file(str(here / "modal_mfu_d.py"), "/root/modal_mfu_d.py", copy=True))
    app = modal.App("arc-mfu-d", include_source=False)
    assets = modal.Volume.from_name("arc-prolong-assets")
    runs = modal.Volume.from_name("arc-prolong-runs")

    @app.function(image=image, gpu="B200", timeout=2400, retries=0, max_containers=1, scaledown_window=2,
                  volumes={"/assets": assets, "/runs": runs})
    def bench(run_id: str, cfg: str):
        out = Path("/runs/validation/mfu-d") / run_id
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONPATH="/root", AXOLOTL_SONICMOE_NVFP4_BACKEND="dequant",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", MFU_D_OUT=str(out))
        began, last = time.time(), time.time()
        with open(out / "log.txt", "w") as log:
            proc = subprocess.Popen([VENV_PY, "-u", "/root/modal_mfu_d.py", "body", cfg], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, env=env)
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if time.time() - last > 60:
                    runs.commit()
                    last = time.time()
            code = proc.wait()
        print(f"[bench] exit {code} after {time.time() - began:.0f} s", flush=True)
        runs.commit()
        return code

    @app.function(image=image, timeout=600, retries=0, max_containers=1, scaledown_window=2)
    def sources():
        """CPU: the image's Axolotl call path into grouped_moe_reference_forward (the native switch patches it)."""
        code = ("import inspect; from axolotl.integrations.kernels.libs.sonicmoe import experts, nvfp4_lora, "
                "experts_lora_fastpath as f, lora; print(inspect.getsource(experts._sonicmoe_nvfp4_forward)); "
                "print(inspect.signature(nvfp4_lora.grouped_moe_reference_forward)); "
                "print(inspect.getsource(f.patch_paramwrapper_sonicmoe_fastpath)); "
                "print(inspect.getsource(lora.sonicmoe_runtime_lora_factors))")
        return subprocess.run([VENV_PY, "-c", code], capture_output=True, text=True).stdout[-12000:]

    @app.local_entrypoint()
    def main(run_id: str = "", cfg: str = "{}", src: bool = False):
        if src:
            print(sources.remote())
            return
        print(json.dumps({"run_id": run_id, "cfg": json.loads(cfg)}))
        print("exit", bench.remote(run_id, cfg))


# ---------------------------------------------------------------------------------------------------- the body
def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def env_info():
    import torch
    import triton

    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
            "gpu": torch.cuda.get_device_name(), "cap": list(torch.cuda.get_device_capability()),
            "grouped_mm": hasattr(torch, "_grouped_mm"), "scaled_grouped_mm": hasattr(torch, "_scaled_grouped_mm")}
    for mod in ("deep_gemm", "quack", "cutlass", "flashinfer", "fbgemm_gpu", "torchao", "axolotl"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "yes")
        except Exception as exc:  # noqa: BLE001
            info[mod] = f"no ({type(exc).__name__})"
    return info


def load_layer(layer, dev):
    import torch
    from safetensors import safe_open

    from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import fuse_nvfp4_experts

    conf = json.loads((Path(MODEL) / "config.json").read_text())
    text = conf.get("text_config", conf)
    wmap = json.loads((Path(MODEL) / "model.safetensors.index.json").read_text())["weight_map"]
    base = f"model.language_model.layers.{layer}.mlp."
    other = sorted(k for k in wmap if k.startswith(base) and ".experts." not in k)
    E = int(text.get("num_experts") or text.get("n_routed_experts"))
    handles = {}

    def get(key):
        shard = wmap[key]
        if shard not in handles:
            handles[shard] = safe_open(str(Path(MODEL) / shard), framework="pt", device="cpu")
        return handles[shard].get_tensor(key)

    def proj(p):
        pre = [f"{base}experts.{e}.{p}." for e in range(E)]
        return {"qd": [get(k + "weight").view(torch.uint8).to(dev) for k in pre],
                "sc": [get(k + "weight_scale").view(torch.float8_e4m3fn).to(dev) for k in pre],
                "pts": [get(k + "weight_scale_2").float().reshape(()).to(dev) for k in pre]}

    began = time.time()
    w1 = fuse_nvfp4_experts([proj("gate_proj"), proj("up_proj")])
    w2 = fuse_nvfp4_experts([proj("down_proj")])
    router = get(f"{base}gate.weight").to(dev)
    log(f"layer {layer}: w1 {tuple(w1.shape)} w2 {tuple(w2.shape)} router {tuple(router.shape)} {router.dtype} "
        f"in {time.time() - began:.0f} s; mlp keys {other[:6]}")
    keys = ("hidden_size", "moe_intermediate_size", "num_experts", "num_experts_per_tok", "norm_topk_prob",
            "hidden_act", "shared_expert_intermediate_size")
    return w1, w2, router, {k: text.get(k) for k in keys}


def capture(layer, H):
    import numpy as np
    import torch

    meta = json.loads((Path(CAPTURE) / "meta.json").read_text())
    n = int(meta["rows"][str(layer)])
    x = np.fromfile(Path(CAPTURE) / f"L{layer:02d}.x", dtype=np.int16, count=n * H).reshape(n, H)
    return torch.from_numpy(x.copy()).view(torch.bfloat16)


def route(x, router, k, norm):
    import torch

    logits = torch.nn.functional.linear(x, router)
    probs = torch.softmax(logits, dtype=torch.float, dim=-1)
    val, idx = torch.topk(probs, k, dim=-1)
    if norm:
        val = val / val.sum(-1, keepdim=True)
    return idx, val.to(logits.dtype)


def lora_factors(E, R, H, I, dev, seed=0):
    """PEFT layouts: A [R E, in] (kaiming-uniform like PEFT), B [out, R E] ~ N(0, 0.01) (a trained-like state)."""
    import torch

    g = torch.Generator(device=dev).manual_seed(seed)
    A1 = (torch.rand(R * E, H, device=dev, generator=g) * 2 - 1) / H ** 0.5
    B1 = torch.randn(2 * I, R * E, device=dev, generator=g) * 0.01
    A2 = (torch.rand(R * E, I, device=dev, generator=g) * 2 - 1) / I ** 0.5
    B2 = torch.randn(H, R * E, device=dev, generator=g) * 0.01
    return A1, B1, A2, B2


def reference(x, idx, wts, W13, W2, F, s1, s2, proj):
    """FP32 loop over experts: y = sum_k w_k (silu(g) u) (W2 + s2 B2 A2)^T, [g|u] = x (W13 + s1 B1 A1)^T."""
    import torch
    import torch.nn.functional as Fn

    T, H = x.shape
    E = W13.shape[0]
    I = W2.shape[2]
    R = F["A1"].shape[0] // E
    xl = x.float().detach().requires_grad_(True)
    wl = wts.float().detach().requires_grad_(True)
    L = {k: v.float().detach().clone().requires_grad_(True) for k, v in F.items()}
    A13, B13 = L["A1"].reshape(E, R, H), L["B1"].reshape(2 * I, R, E)
    A2r, B2r = L["A2"].reshape(E, R, I), L["B2"].reshape(H, R, E)
    y = torch.zeros(T, H, device=x.device)
    for e in torch.unique(idx).tolist():
        tok, pos = (idx == e).nonzero(as_tuple=True)
        xe = xl[tok]
        h = xe @ W13[e].T + s1 * ((xe @ A13[e].T) @ B13[:, :, e].T)
        a = Fn.silu(h[:, :I]) * h[:, I:]
        o = a @ W2[e].T + s2 * ((a @ A2r[e].T) @ B2r[:, :, e].T)
        y = y.index_add(0, tok, o * wl[tok, pos, None])
    (y * proj).sum().backward()
    return y.detach(), {"dx": xl.grad, "dw": wl.grad, **{k: v.grad for k, v in L.items()}}


def run_path(fn, x, idx, wts, F, proj, grad=True):
    import torch

    xl = x.detach().clone().requires_grad_(grad)
    wl = wts.detach().clone().requires_grad_(grad)
    L = {k: v.detach().clone().requires_grad_(grad) for k, v in F.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = fn(xl, idx, wl, (L["A1"].to(torch.bfloat16), L["B1"].to(torch.bfloat16)),
               (L["A2"].to(torch.bfloat16), L["B2"].to(torch.bfloat16)))
    if not grad:
        return y, None
    (y.float() * proj).sum().backward()
    return y.detach(), {"dx": xl.grad, "dw": wl.grad, **{k: v.grad for k, v in L.items()}}


def compare(y, grads, y_ref, g_ref):
    def rel(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm())

    def cos(a, b):
        a, b = a.float().flatten(), b.float().flatten()
        return float((a @ b) / (a.norm() * b.norm()))

    out = {"y_rel": rel(y, y_ref)}
    if grads is not None:
        for k, v in g_ref.items():
            if grads.get(k) is not None:
                out[k] = {"cos": round(cos(grads[k], v), 6), "rel": round(rel(grads[k], v), 5)}
    return out


def timed(fn, reps):
    import torch

    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    return round(times[len(times) // 2], 3)


def speed(fn, x, idx, wts, F, proj, reps=5):
    """Median ms: forward without grad (the prefix pass), forward with grad, backward; peak extra GiB."""
    import torch

    def fwd_nograd():
        with torch.no_grad():
            run_path(fn, x, idx, wts, F, proj, grad=False)

    state = {}

    def fwd():
        xl = x.detach().clone().requires_grad_(True)
        wl = wts.detach().clone().requires_grad_(True)
        L = {k: v.detach().clone().requires_grad_(True) for k, v in F.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = fn(xl, idx, wl, (L["A1"].to(torch.bfloat16), L["B1"].to(torch.bfloat16)),
                   (L["A2"].to(torch.bfloat16), L["B2"].to(torch.bfloat16)))
        state["loss"] = (y.float() * proj).sum()

    def bwd():
        state.pop("loss").backward()

    for _ in range(2):
        fwd_nograd()
        fwd()
        bwd()
    torch.cuda.synchronize()
    out = {"fwd_nograd_ms": timed(fwd_nograd, reps)}
    f, b = [], []
    for _ in range(reps):
        f.append(timed(fwd, 1))
        b.append(timed(bwd, 1))
    f.sort(), b.sort()
    out.update(fwd_ms=f[len(f) // 2], bwd_ms=b[len(b) // 2])
    out["fwd_bwd_ms"] = round(out["fwd_ms"] + out["bwd_ms"], 3)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fwd()
    bwd()
    torch.cuda.synchronize()
    out["peak_extra_gib"] = round((torch.cuda.max_memory_allocated() - base) / 2 ** 30, 2)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        fwd_nograd()
    out["nograd_peak_extra_gib"] = round((torch.cuda.max_memory_allocated() - base) / 2 ** 30, 2)
    return out


def profile(fn, x, idx, wts, F, proj, top=18):
    import torch
    from torch.profiler import ProfilerActivity, profile as prof

    with prof(activities=[ProfilerActivity.CUDA]) as p:
        run_path(fn, x, idx, wts, F, proj)
        torch.cuda.synchronize()
    ev = [e for e in p.key_averages() if e.device_time_total > 0]
    ev.sort(key=lambda e: e.device_time_total, reverse=True)
    total = sum(e.device_time_total for e in ev)
    return {"device_ms": round(total / 1e3, 2), "launches": sum(e.count for e in ev),
            "top": [[e.key[:90], e.count, round(e.device_time_total / 1e3, 2)] for e in ev[:top]]}


def body(cfg):
    import traceback

    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    out_dir = Path(os.environ.get("MFU_D_OUT", "/tmp/mfu-d"))
    result = {"cfg": cfg, "env": env_info()}
    log(json.dumps(result["env"]))
    layer, T, R, s = int(cfg.get("layer", 24)), int(cfg.get("tokens", 32768)), int(cfg.get("rank", 16)), 2.0
    w1, w2, router, conf = load_layer(layer, dev)
    result["config"] = conf
    H, E = w1.shape[2], w1.shape[0]
    I = w2.shape[2]
    K = int(conf.get("num_experts_per_tok") or 10)
    xc = capture(layer, H).to(dev)
    log(f"capture rows {xc.shape[0]}; E {E} H {H} I {I} K {K} R {R}")
    F = dict(zip(("A1", "B1", "A2", "B2"), lora_factors(E, R, H, I, dev)))

    from axolotl.integrations.kernels.libs.sonicmoe import nvfp4_lora
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4 import dequantize_expert_weight

    import native_experts

    original = nvfp4_lora.grouped_moe_reference_forward

    def path(backend):
        def fn(x, idx, wts, l1, l2):
            return original(x, idx, wts, w1, None, w2, None, l1, l2, E, act="silu", backend=backend, concat=True,
                            scaling1=s, scaling2=s)
        return fn

    def native(x, idx, wts, l1, l2):
        native_experts.CACHE["key"] = None  # each layer's forward decodes in the trainer (its backward hits the slot)
        return native_experts.moe_forward(x, idx, wts, w1, None, w2, None, l1, l2, E, act="silu", backend="dequant",
                                          concat=True, scaling1=s, scaling2=s)

    native_experts.ORIGINAL["fn"] = original
    paths = {"dequant": path("dequant"), "native": native, "fp4_cute": path("fp4_cute")}
    paths = {k: v for k, v in paths.items() if k in cfg.get("paths", list(paths))}

    # numerics: the capture's real rows, the layer's own router
    if cfg.get("numerics", True):
        n = min(int(cfg.get("numerics_tokens", 8192)), xc.shape[0])
        x = xc[:n]
        idx, wts = route(x, router, K, conf.get("norm_topk_prob", True))
        proj = torch.randn(n, H, device=dev, generator=torch.Generator(device=dev).manual_seed(1))
        W13 = dequantize_expert_weight(w1).float()
        W2 = dequantize_expert_weight(w2).float()
        y_ref, g_ref = reference(x, idx, wts, W13, W2, F, s, s, proj)
        Z = {k: (torch.zeros_like(v) if k.startswith("B") else v) for k, v in F.items()}
        y_ref0, _ = reference(x, idx, wts, W13, W2, Z, s, s, proj)
        del W13, W2
        torch.cuda.empty_cache()
        result["lora_effect_rel"] = float((y_ref - y_ref0).norm() / y_ref0.norm())
        num, outs = {}, {}
        for name, fn in paths.items():
            try:
                y, grads = run_path(fn, x, idx, wts, F, proj)
                torch.cuda.synchronize()
                num[name] = compare(y, grads, y_ref, g_ref)
                outs[name] = y.float()
                y0, _ = run_path(fn, x, idx, wts, Z, proj, grad=False)
                num[name]["delta_cos"] = float(((y.float() - y0.float()) * (y_ref - y_ref0)).sum()
                                               / ((y.float() - y0.float()).norm() * (y_ref - y_ref0).norm()))
                log(name, json.dumps(num[name]))
            except Exception:  # noqa: BLE001
                num[name] = {"error": traceback.format_exc()[-1500:]}
                log(name, "FAILED", num[name]["error"])
        if "fp4_cute" in outs and "dequant" in outs:
            num["fp4_cute_vs_dequant_y_rel"] = float((outs["fp4_cute"] - outs["dequant"]).norm() / outs["dequant"].norm())
        result["numerics"] = num
        result["numerics_tokens"] = n
        (out_dir / "result.json").write_text(json.dumps(result, indent=1))

    # speed: a T-token window (capture rows repeated), routed by the real router
    for T in ([T] + [int(t) for t in cfg.get("more_tokens", [16384])] if cfg.get("speed", True) else []):
        x = xc[torch.arange(T, device=dev) % xc.shape[0]]
        idx, wts = route(x, router, K, conf.get("norm_topk_prob", True))
        counts = torch.bincount(idx.flatten(), minlength=E)
        result["routing" if T == int(cfg.get("tokens", 32768)) else f"routing_{T}"] = {"tokens": T, "rows": T * K, "per_expert_mean": float(counts.float().mean()),
                             "max": int(counts.max()), "min": int(counts.min()), "p90": float(counts.float().quantile(0.9)),
                             "unused": int((counts == 0).sum())}
        log("routing", T, json.dumps(counts.float().quantile(torch.tensor([0.5, 0.9, 0.99], device=dev)).tolist()))
        proj = torch.randn(T, H, device=dev, generator=torch.Generator(device=dev).manual_seed(2))
        sp = {}
        for name, fn in paths.items():
            try:
                sp[name] = speed(fn, x, idx, wts, F, proj, reps=int(cfg.get("reps", 5)))
                log(name, json.dumps(sp[name]))
                if cfg.get("profile", True) and T == int(cfg.get("tokens", 32768)):
                    sp[name]["profile"] = profile(fn, x, idx, wts, F, proj)
                    log(name, "profile", json.dumps(sp[name]["profile"]))
            except Exception:  # noqa: BLE001
                sp[name] = {"error": traceback.format_exc()[-1500:]}
                log(name, "FAILED", sp[name]["error"])
            torch.cuda.empty_cache()
        if "dequant" in sp and "native" in sp and "fwd_bwd_ms" in sp["native"] and "fwd_bwd_ms" in sp["dequant"]:
            sp["speedup"] = {k: round(sp["dequant"][k] / sp["native"][k], 2)
                             for k in ("fwd_nograd_ms", "fwd_ms", "bwd_ms", "fwd_bwd_ms")}
            log("speedup", json.dumps(sp["speedup"]))
        result["speed" if T == int(cfg.get("tokens", 32768)) else f"speed_{T}"] = sp
        result["native_stats"] = dict(native_experts.STATS)
        result["best_configs"] = {k: str(getattr(getattr(native_experts, k), "best_config", None))
                                  for k in ("_wgrad",)}
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    log("RESULT", json.dumps({k: v for k, v in result.items() if k != "speed"}))
    for key in [k for k in result if k.startswith("speed")]:
        log(key, json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "profile"} if isinstance(v, dict) else v)
                             for k, v in result[key].items()}))


if BODY:
    body(json.loads(sys.argv[2]))
