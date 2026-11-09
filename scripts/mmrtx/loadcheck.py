"""Real vLLM weight-loader check on Modal CPU (rule: no GPU relaunch after a loader/boot failure before this passes).

Builds the actual MiniMax model class from a candidate's config cut to N layers (layer 3 is the first MoE layer) under
the mmrtx patch env, then runs the model's own load_weights on that candidate's real tensors for those layers.
Checks every routed-expert param was written (no NaN sentinel left) and reports param shapes/bytes. Kernel repack
(process_weights_after_loading) is CUDA-only and is left to the GPU boot.

usage: modal run loadcheck.py --model cand-h-int3g32-38 --env '{"MMRTX_MOE_HUMMING": "..."}' [--layers 4]"""
import json
from pathlib import Path

import modal

import sys as _sys
_sys.path[:0] = [str(Path(__file__).resolve().parent), '/root']
from image import IMAGE  # noqa: E402

app = modal.App("mmrtx-loadcheck")
models = modal.Volume.from_name("mmrtx-models")


@app.function(image=IMAGE, timeout=1800, cpu=(8, 16), volumes={"/m": models})
def check(model: str, env: str, layers: int, patch: str):
    import os
    import subprocess
    import sys
    os.makedirs("/tmp/mm/mmrtx_patch", exist_ok=True)
    Path("/tmp/mm/mmrtx_patch/sitecustomize.py").write_text(patch)
    Path("/tmp/mm/child.py").write_text(CHILD)
    e = {**os.environ, **json.loads(env), "PYTHONPATH": "/tmp/mm/mmrtx_patch", "VLLM_LOGGING_LEVEL": "WARNING"}
    r = subprocess.run([sys.executable, "/tmp/mm/child.py", f"/m/{model}", str(layers)], env=e, capture_output=True,
                       text=True, timeout=1500)
    return {"rc": r.returncode, "out": r.stdout[-6000:], "err": r.stderr[-6000:]}


CHILD = r'''
import json, os, shutil, sys
from pathlib import Path
import torch
src, layers = sys.argv[1], int(sys.argv[2])
d = Path("/tmp/cut"); shutil.copytree(src, d, ignore=shutil.ignore_patterns("*.safetensors", ".cache"), dirs_exist_ok=True)
cfg = json.loads((d / "config.json").read_text()); cfg["text_config"]["num_hidden_layers"] = layers
(d / "config.json").write_text(json.dumps(cfg))
from vllm.config import VllmConfig, ModelConfig, DeviceConfig, LoadConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
mc = ModelConfig(model=str(d), trust_remote_code=True, max_model_len=4096, limit_mm_per_prompt={"image": 0, "video": 0})
vc = VllmConfig(model_config=mc, device_config=DeviceConfig("cpu"), load_config=LoadConfig(load_format="dummy"))
from vllm.model_executor.model_loader.utils import initialize_model
import vllm.model_executor.layers.quantization.modelopt as mo  # patched by sitecustomize on this import
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
_gq = mo.ModelOptMxFp8Config.get_quant_method
def _cpu_gq(self, layer, prefix):  # dense MXFP8 kernels need a GPU; only routed experts are under test
    return UnquantizedLinearMethod() if isinstance(layer, LinearBase) else _gq(self, layer, prefix)
mo.ModelOptMxFp8Config.get_quant_method = _cpu_gq
from vllm.platforms import current_platform  # Humming's backend choice reads the capability: report the RTX's sm_120
from vllm.platforms.interface import DeviceCapability
type(current_platform).get_device_capability = classmethod(lambda cls, device_id=0: DeviceCapability(12, 0))
import types, humming.device as _hdv, humming.utils.device as _hud  # Humming's device query needs libcuda: RTX values
_rtx = dict(index=0, name="NVIDIA RTX PRO 6000 Blackwell Server Edition", is_ppu=False, sm_count=188,
            max_threads_per_block=1024, max_threads_per_sm=1536, max_registers_per_sm=65536, sm_major=12, sm_minor=0,
            sm_version=120, l2_cache_size=128 << 20, l2_cache_size_mb=128.0, l1_cache_size=128 << 10, l1_cache_size_kb=128.0,
            default_smem_size=48 << 10, max_smem_size=99 << 10, max_smem_size_kb=99.0, memory_clock_khz=14000000,
            memory_bus_width=512, sm_clock_khz=2400000, memory_bandwidth_gbps=1792.0, tensorcore_tops={})
_fake = types.SimpleNamespace(_DeviceInfo=lambda i: types.SimpleNamespace(**_rtx))
_hud._get_device_info_extension = _hdv._get_device_info_extension = lambda: _fake
_hdv.current_device._value = None
import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 as _nv
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend, backend_to_kernel_cls
_marlin = backend_to_kernel_cls(NvFp4MoeBackend.MARLIN)[0]  # sm_120 serves NVFP4 experts with Marlin; no CPU kernel
_nv.select_nvfp4_moe_backend = lambda config, weight_key, activation_key: (NvFp4MoeBackend.MARLIN, _marlin)
ctx = set_current_vllm_config(vc); ctx.__enter__()
init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="tcp://127.0.0.1:29511", backend="gloo")
initialize_model_parallel(1, 1)
with torch.device("cpu"):
    model = initialize_model(vllm_config=vc)
print("MODEL", type(model).__name__, flush=True)
experts = {n: p for n, p in model.named_parameters() if ".experts." in n}
IV = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
def iv(p):  # integer bit view of any param
    return p.data.view(IV[p.element_size()])
def fill(which):  # two passes with bit sentinels (iinfo max, then min): unwritten = kept its sentinel both times
    for p in experts.values():
        v = iv(p); v.fill_(torch.iinfo(v.dtype).max if which == 0 else torch.iinfo(v.dtype).min)
print("METHOD", sorted({type(m.quant_method).__name__ for m in model.modules() if hasattr(m, "quant_method") and ".experts" in str(type(m))} ) , flush=True)
from safetensors import safe_open
synth = os.environ.get("SYNTH_SCHEMA")
fed = {}
if synth:  # Humming-native tensors with real shapes, distinct values per expert/shard
    from humming.schema.base import BaseWeightSchema
    sch = BaseWeightSchema.from_config(json.loads(synth))
    w13 = next(p for n, p in experts.items() if n.endswith(("w13_weight", "w13_weight_packed")))
    w2 = next(p for n, p in experts.items() if n.endswith(("w2_weight", "w2_weight_packed")))
    E = w13.shape[0]
    moe_layers = sorted({int(n.split(".layers.")[1].split(".")[0]) for n in experts})
    only = {int(x) for x in os.environ.get("MMRTX_MOE_HUMMING_LAYERS", "").split(",") if x.strip()}
    H = int(cfg["text_config"]["hidden_size"]); I = int(cfg["text_config"].get("moe_intermediate_size") or cfg["text_config"]["intermediate_size"])
    print("SHAPES E", E, "H", H, "I", I, "w13", list(w13.shape), "w2", list(w2.shape), flush=True)
    def gen():  # reseeded: both passes feed identical tensors
        g = torch.Generator().manual_seed(0)
        for L in moe_layers:
            for e in range(E):
                for w, (N, K) in (("w1", (I, H)), ("w3", (I, H)), ("w2", (H, I))):
                    if only and L not in only:  # NVFP4 layer (compressed-tensors names and shapes)
                        at = {"weight_packed": {"shape": (1, N, K // 2), "dtype": torch.uint8},
                              "weight_scale": {"shape": (1, N, K // 16), "dtype": torch.float8_e4m3fn},
                              "weight_global_scale": {"shape": (1,), "dtype": torch.float32},
                              "input_global_scale": {"shape": (1,), "dtype": torch.float32}}
                    else:
                        at = sch.get_tensors_attrs(shape_n=N, shape_k=K, param_dtype=torch.bfloat16, num_experts=1)
                    for k, a in at.items():
                        shp = tuple(a["shape"])[1:]
                        dt = a["dtype"]
                        if dt == torch.int32:
                            t = torch.randint(-2**30, 2**30, shp, generator=g, dtype=dt)
                        elif dt == torch.uint8:
                            t = torch.randint(0, 256, shp, generator=g, dtype=torch.int32).to(dt)
                        elif k in ("weight_global_scale", "input_global_scale"):  # one value per expert (w1 == w3)
                            t = torch.tensor(float(L * 1000 + e + 1), dtype=dt)
                        else:
                            t = (torch.rand(shp, generator=g) + 1).to(dt)
                        name = f"language_model.model.layers.{L}.block_sparse_moe.experts.{e}.{w}.{k}"
                        fed[(L, e, w, k)] = t
                        yield name, t
    want = ["synth"]
else:
    idx = json.loads((Path(src) / "model.safetensors.index.json").read_text())["weight_map"]
    want = [n for n in idx if ".block_sparse_moe.experts." in n and int(n.split(".layers.")[1].split(".")[0]) < layers]
    files = {}
    for n in want: files.setdefault(idx[n], []).append(n)
    def gen():
        for f, names in files.items():
            with safe_open(f"{src}/{f}", "pt") as h:
                for n in names: yield n, h.get_tensor(n)
fill(0)
loaded = model.load_weights(gen())
kept0 = {n: (iv(p) == torch.iinfo(iv(p).dtype).max) for n, p in experts.items()}
fill(1)
loaded = model.load_weights(gen())
mism = []
if synth:  # placement: w1 -> w13 rows [0:I], w3 -> [I:2I], w2 whole
    params = dict(experts)
    def same(a, b):
        if a.dtype == torch.float8_e4m3fn:
            a, b = a.view(torch.uint8), b.view(torch.uint8)
        return a.shape == b.shape and torch.equal(a.cpu(), b)
    for (L, e, w, k), t in fed.items():
        cands = [n for n in params if f".layers.{L}." in n and n.endswith(("w2_" if w == "w2" else "w13_") + k)]
        if not cands:
            if k == "input_global_scale":  # not modelled by every method (skipped by load_weights)
                continue
            mism.append([f"L{L} {w}.{k}", e, w, "no param", list(t.shape)]); continue
        d = params[cands[0]].data[e]
        if t.dim() == 0:  # per-tensor scale: [E] holds it, or [E, 2] (w1, w3) for w13
            ok = float(d.reshape(-1)[0 if w != "w3" or d.numel() == 1 else 1]) == float(t)
        else:
            part = d if w == "w2" else (d[: d.shape[0] // 2] if w == "w1" else d[d.shape[0] // 2:])
            ok = same(part, t)
        if not ok:
            mism.append([cands[0], e, w, list(d.shape), list(t.shape)])
    print("PER_LAYER_METHOD", json.dumps({n: type(m.quant_method).__name__ for n, m in model.named_modules()
                                          if type(m).__name__ == "RoutedExperts"}), flush=True)
    print("MISPLACED", len(mism), json.dumps(mism[:6]), flush=True)
bad = {}
for n, p in experts.items():
    left = int((kept0[n] & (iv(p) == torch.iinfo(iv(p).dtype).min)).sum())
    if left: bad[n] = [left, p.numel()]
print("PARAMS", json.dumps({n: [list(p.shape), str(p.dtype)] for n, p in experts.items()}), flush=True)
print("NTENSORS", len(want), "LOADED", len(loaded), flush=True)
print("UNWRITTEN", json.dumps(bad), flush=True)
print("PASS" if not bad and want and not mism else "FAIL", flush=True)
'''


@app.local_entrypoint()
def main(model: str, env: str = "{}", layers: int = 4, patch_file: str = ""):
    """env may hold SYNTH_SCHEMA (Humming schema JSON) for synthetic native tensors."""
    patch = Path(patch_file or Path(__file__).resolve().parent / "mmrtx_patch/sitecustomize.py").read_text()
    r = check.remote(model, env, layers, patch)
    print("rc", r["rc"]); print(r["out"]); print("---- stderr tail"); print(r["err"][-4000:])
