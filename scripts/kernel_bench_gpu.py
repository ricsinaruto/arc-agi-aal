#!/usr/bin/env python3
"""Dense-GEMM decode microbenchmark for the Qwen3.8-Flash-Next shapes (kernel work, phase 1).

Runs standalone on one CUDA GPU (the Modal launcher is scripts/modal_kernel_bench.py). For every
dense BF16 projection shape of the served model (TP=1) and every token count M, it times:

- BF16: torch F.linear (cuBLASLt default), F.linear with the non-Lt cuBLAS backend, x @ W^T stored
  K-major, a Triton split-K GEMM (config sweep), vLLM's CuTe-DSL skinny GEMM (M <= 16);
- FP8 W8A8: torch._scaled_mm per-tensor and row-wise, vLLM cutlass_scaled_mm per-tensor and
  per-token x per-channel, each GEMM-only (pre-quantized x) and with vLLM's dynamic act quant;
- W8A16 weight-only: vLLM FP8 Marlin, Triton FP8 and INT8 split-K kernels (per-channel scales);
- NVFP4 W4A4: vLLM cutlass_scaled_fp4_mm (GEMM-only and with activation quantization).

Timing: each method's weights are replicated into enough distinct copies to stream several times
the L2 size per pass (no L2 reuse, as in the model, where ~13 GB of other weights separate two
reads of one matrix); one pass over all copies is captured in a CUDA graph and replayed. The
reported time is per call (graph node spacing included, as in vLLM's graphs).

Quantization error: relative Frobenius error of each method's output against an FP32 reference
of the BF16 weights, using the real checkpoint weights when --model-dir is given (random
Gaussian activations; weight-only error is representative, W8A8/W4A4 activation error is a lower
bound because real activations have outlier channels).
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# name, N, K, checkpoint tensors concatenated along dim 0 (None: random), pad rows
SHAPES = [
    ("gdn_in_qkvz", 16384, 2560, ["model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                                  "model.language_model.layers.0.linear_attn.in_proj_z.weight"], 0),
    ("attn_out", 2560, 6144, ["model.language_model.layers.0.linear_attn.out_proj.weight"], 0),
    ("lm_head", 248320, 2560, ["lm_head.weight"], 0),
    ("qsa_qkv", 13312, 2560, ["model.language_model.layers.3.self_attn.q_proj.weight",
                              "model.language_model.layers.3.self_attn.k_proj.weight",
                              "model.language_model.layers.3.self_attn.v_proj.weight"], 0),
    ("hc_down_inject", 336, 10240, [
        "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
        "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight"], 12),
    ("hc_up", 10240, 320, ["model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight"], 0),
    ("shared_gate_up", 1280, 2560, ["model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
                                    "model.language_model.layers.0.mlp.shared_expert.up_proj.weight"], 0),
    ("router", 512, 2560, ["model.language_model.layers.0.mlp.gate.weight"], 0),
    ("shared_down", 2560, 640, ["model.language_model.layers.0.mlp.shared_expert.down_proj.weight"], 0),
    ("gdn_in_ba", 96, 2560, ["model.language_model.layers.0.linear_attn.in_proj_b.weight",
                             "model.language_model.layers.0.linear_attn.in_proj_a.weight"], 0),
    ("qsa_index_qk", 640, 2560, ["model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight"], 0),
    ("h2560", 2560, 2560, ["model.language_model.layers.1.ple.value_proj.weight"], 0),
    ("ple_key", 10240, 2560, ["model.language_model.layers.1.ple.key_proj.weight"], 0),
    ("gdn_in_qkvzba_merged", 16480, 2560, None, 0),
    ("hc_mixer_down", 320, 10240, ["mtp.hyper_connection_mixer.input_mix_weight_down.weight"], 0),
]
MS = [1, 2, 4, 6, 8, 16, 24, 32, 64]
TRITON_MS = {1, 4, 8, 24, 64}
SKINNY_MS = {1, 4, 8, 16}
# (BLOCK_N, BLOCK_K, num_warps, num_stages)
TRITON_CONFIGS = [(16, 256, 4, 3), (32, 256, 4, 3), (32, 128, 4, 4), (64, 128, 4, 3)]
FP8 = torch.float8_e4m3fn


@triton.jit
def _gemm_splitk_kernel(x_ptr, w_ptr, s_ptr, out_ptr, M, N, K, K_PER_SPLIT,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        PARTIAL: tl.constexpr, SCALED: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k_start = pid_k * K_PER_SPLIT
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    x_ptrs = x_ptr + offs_m[:, None] * K + (k_start + offs_k)[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * K + (k_start + offs_k)[None, :]
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[:, None] < N
    for k in range(0, K_PER_SPLIT, BLOCK_K):
        k_mask = (k_start + k + offs_k)[None, :] < K
        a = tl.load(x_ptrs, mask=m_mask & k_mask, other=0.0)
        # v1 run (28 Sept) used other=0, which Triton cannot cast to fp8e4nv: the FP8 W8A16 variant
        # never compiled. A float fill value is castable to bf16, fp8e4nv and int8.
        b = tl.load(w_ptrs, mask=n_mask & k_mask, other=0.0, eviction_policy="evict_first")
        acc += tl.dot(a, tl.trans(b.to(tl.bfloat16)))
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    if SCALED:
        s = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc * s[None, :]
    o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if PARTIAL:
        o_ptrs = out_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
        tl.store(o_ptrs, acc, mask=o_mask)
    else:
        o_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=o_mask)


@triton.jit
def _splitk_reduce_kernel(p_ptr, out_ptr, MN, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(p_ptr + s * MN + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)


def next_pow2(v):
    return 1 << (max(1, v) - 1).bit_length()


def triton_gemm(x, w, scale, cfg, num_sms):
    m, k = x.shape
    n = w.shape[0]
    bn, bk, warps, stages = cfg
    bm = 16 if m <= 16 else 64
    n_tiles = triton.cdiv(n, bn)
    split = 1
    while split < 16 and n_tiles * split < 2 * num_sms and k // (split * 2) >= 2 * bk:
        split *= 2
    k_per = triton.cdiv(triton.cdiv(k, split), bk) * bk
    split = triton.cdiv(k, k_per)
    s_arg = scale if scale is not None else x
    if split == 1:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _gemm_splitk_kernel[(n_tiles, 1)](x, w, s_arg, out, m, n, k, k_per, BLOCK_M=bm, BLOCK_N=bn,
                                          BLOCK_K=bk, PARTIAL=False, SCALED=scale is not None,
                                          num_warps=warps, num_stages=stages)
        return out
    part = torch.empty((split, m, n), dtype=torch.float32, device=x.device)
    _gemm_splitk_kernel[(n_tiles, split)](x, w, s_arg, part, m, n, k, k_per, BLOCK_M=bm, BLOCK_N=bn,
                                          BLOCK_K=bk, PARTIAL=True, SCALED=scale is not None,
                                          num_warps=warps, num_stages=stages)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _splitk_reduce_kernel[(triton.cdiv(m * n, 1024),)](part, out, m * n, SPLIT=split, BLOCK=1024)
    return out


def log(obj):
    print(json.dumps(obj, default=str), flush=True)


class Bench:
    def __init__(self, args):
        self.args = args
        self.dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        self.props = props
        self.num_sms = props.multi_processor_count
        self.l2 = getattr(props, "L2_cache_size", 128 << 20)
        self.rot_bytes = max(2 * self.l2, 256 << 20)
        self.start = time.time()
        self.deadline = self.start + args.deadline_seconds
        self.results = {"env": {}, "bandwidth": {}, "results": [], "act_quant": [], "errors": [],
                        "notes": []}
        self.skinny_compile_seconds = 0.0
        self.vllm_ops = None
        self.last_flush = 0.0

    # ------------------------------------------------------------------ bookkeeping
    def left(self):
        return self.deadline - time.time()

    def flush(self, force=False):
        if not force and time.time() - self.last_flush < 30:
            return
        self.last_flush = time.time()
        self.results["elapsed_seconds"] = time.time() - self.start
        tmp = self.args.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.results, fh, default=str)
        os.replace(tmp, self.args.out)

    def env(self):
        env = {
            "torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
            "gpu": self.props.name, "capability": list(torch.cuda.get_device_capability()),
            "sms": self.num_sms, "l2_bytes": self.l2, "total_memory": self.props.total_memory,
            "rotation_bytes": self.rot_bytes, "cublas_lt_version": None,
        }
        try:
            env["cublas_lt_version"] = torch._C._cuda_getCompiledVersion()
        except Exception:
            pass
        for mod in ("vllm", "flashinfer", "cutlass", "quack"):
            try:
                env[mod] = __import__(mod).__version__
            except Exception as exc:
                env[mod] = f"unavailable: {type(exc).__name__}: {exc}"[:200]
        try:
            import ctypes
            import glob
            libs = glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cu*", "lib",
                                          "libcublasLt.so*"))
            libs += glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cublas", "lib",
                                           "libcublasLt.so*"))
            if libs:
                lib = ctypes.CDLL(sorted(libs)[-1])
                lib.cublasLtGetVersion.restype = ctypes.c_size_t
                env["cublas_lt_version"] = int(lib.cublasLtGetVersion())
                env["cublas_lt_library"] = sorted(libs)[-1]
        except Exception as exc:
            env["cublas_lt_probe_error"] = str(exc)[:200]
        try:
            env["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit,"
                 "memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=30
            ).stdout.strip()
        except Exception as exc:
            env["nvidia_smi"] = str(exc)
        self.results["env"] = env
        log({"env": env})

    # ------------------------------------------------------------------ timing
    def graph_time(self, calls, target_ms=15.0):
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for call in calls:
                call()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for call in calls:
                call()
        graph.replay()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        graph.replay()
        e1.record()
        e1.synchronize()
        one = max(e0.elapsed_time(e1), 1e-3)
        reps = max(1, min(100, int(target_ms / one)))
        samples = []
        for _ in range(3):
            e0.record()
            for _ in range(reps):
                graph.replay()
            e1.record()
            e1.synchronize()
            samples.append(e0.elapsed_time(e1) / reps / len(calls) * 1000.0)
        del graph
        return statistics.median(samples), min(samples)

    def copies_for(self, nbytes):
        return int(min(1024, max(2, math.ceil(self.rot_bytes / max(nbytes, 1)))))

    @staticmethod
    def replicate(packed, copies):
        """Stack each tensor of a packed weight dict into `copies` distinct buffers."""
        out = [dict() for _ in range(copies)]
        for key, t in packed.items():
            if key.startswith("_") or not isinstance(t, torch.Tensor):
                for d in out:
                    d[key] = t
                continue
            base = t.contiguous()
            raw = base.view(torch.uint8) if base.element_size() == 1 else base
            buf = torch.empty((copies,) + tuple(raw.shape), dtype=raw.dtype, device=raw.device)
            buf.copy_(raw.unsqueeze(0).expand_as(buf))
            if raw is not base:
                buf = buf.view(base.dtype)
            for i in range(copies):
                out[i][key] = buf[i]
        return out

    # ------------------------------------------------------------------ weights
    def load_weight(self, names, n, k, pad):
        if names and self.args.model_dir:
            try:
                from safetensors import safe_open
                index = self.index
                parts = []
                for name in names:
                    path = os.path.join(self.args.model_dir, index[name])
                    with safe_open(path, framework="pt", device="cpu") as fh:
                        parts.append(fh.get_tensor(name).to(torch.bfloat16))
                w = torch.cat(parts, 0)
                if pad:
                    w = torch.cat([w, torch.zeros(pad, w.shape[1], dtype=w.dtype)], 0)
                if tuple(w.shape) == (n, k):
                    return w.to(self.dev), "checkpoint"
                self.results["notes"].append(f"{names}: checkpoint shape {tuple(w.shape)} != {(n, k)}")
            except Exception as exc:
                self.results["notes"].append(f"{names}: load failed {type(exc).__name__}: {exc}"[:300])
        g = torch.Generator(device="cpu").manual_seed(n * 7 + k)
        return (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16).to(self.dev), "random"

    # ------------------------------------------------------------------ methods
    def methods(self):
        """Return {name: (pack(W)->dict, run(p, xin)->y, xprep(x)->xin, allowed(M,N,K))}."""
        ops = self.vllm_ops
        dev = self.dev
        num_sms = self.num_sms
        meths = {}

        def always(m, n, k):
            return True

        ident = lambda x: x  # noqa: E731

        meths["bf16_linear"] = (lambda w: {"w": w}, lambda p, x: F.linear(x, p["w"]), ident, always)
        meths["bf16_linear_altblas"] = (lambda w: {"w": w}, lambda p, x: F.linear(x, p["w"]), ident, always)
        meths["bf16_mm_kmajor"] = (lambda w: {"wt": w.t().contiguous()}, lambda p, x: torch.mm(x, p["wt"]),
                                   ident, always)

        def per_tensor_fp8(w):
            s = (w.float().abs().amax() / 448.0).clamp(min=1e-12)
            return (w.float() / s).clamp(-448, 448).to(FP8), s.reshape(())

        def per_channel_fp8(w):
            s = (w.float().abs().amax(dim=1) / 448.0).clamp(min=1e-12)
            return (w.float() / s[:, None]).clamp(-448, 448).to(FP8), s

        def per_channel_int8(w):
            s = (w.float().abs().amax(dim=1) / 127.0).clamp(min=1e-12)
            return torch.round(w.float() / s[:, None]).clamp(-127, 127).to(torch.int8), s

        def xq_tensor(x):
            s = (x.float().abs().amax() / 448.0).clamp(min=1e-12)
            return ((x.float() / s).clamp(-448, 448).to(FP8), s.reshape(()))

        def xq_token(x):
            s = (x.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
            return ((x.float() / s).clamp(-448, 448).to(FP8), s)

        def fp8_ok(m, n, k):
            return n % 16 == 0 and k % 16 == 0

        def pack_t(w):
            q, s = per_tensor_fp8(w)
            return {"wq": q, "s": s}

        def pack_c(w):
            q, s = per_channel_fp8(w)
            return {"wq": q, "s": s.reshape(1, -1).contiguous()}

        meths["fp8_scaled_mm_tensor"] = (
            pack_t, lambda p, xi: torch._scaled_mm(xi[0], p["wq"].t(), scale_a=xi[1], scale_b=p["s"],
                                                   out_dtype=torch.bfloat16), xq_tensor, fp8_ok)
        meths["fp8_scaled_mm_rowwise"] = (
            pack_c, lambda p, xi: torch._scaled_mm(xi[0], p["wq"].t(), scale_a=xi[1], scale_b=p["s"],
                                                   out_dtype=torch.bfloat16), xq_token, fp8_ok)

        for bm_name, cfg in [("triton_bf16", None), ("triton_w8a16_fp8", per_channel_fp8),
                             ("triton_w8a16_int8", per_channel_int8)]:
            def pack_tr(w, cfg=cfg):
                if cfg is None:
                    return {"w": w}
                q, s = cfg(w)
                return {"w": q, "s": s.float().contiguous()}
            meths[bm_name] = (pack_tr, None, ident, lambda m, n, k: m in TRITON_MS)

        if ops is not None:
            cap = torch.cuda.get_device_capability()
            capi = cap[0] * 10 + cap[1]
            try:
                fp8_cutlass = bool(ops.cutlass_scaled_mm_supports_fp8(capi))
            except Exception:
                fp8_cutlass = False
            try:
                fp4_cutlass = bool(ops.cutlass_scaled_mm_supports_fp4(capi))
            except Exception:
                fp4_cutlass = False
            self.results["env"]["vllm_cutlass_fp8"] = fp8_cutlass
            self.results["env"]["vllm_cutlass_fp4"] = fp4_cutlass
            if fp8_cutlass:
                def pack_ct(w):
                    q, s = per_tensor_fp8(w)
                    return {"wq": q, "s": s.reshape(1)}

                def pack_cc(w):
                    q, s = per_channel_fp8(w)
                    return {"wq": q, "s": s.reshape(-1, 1).contiguous()}

                meths["vllm_cutlass_fp8_tensor"] = (
                    pack_ct, lambda p, xi: ops.cutlass_scaled_mm(xi[0], p["wq"].t(), xi[1].reshape(1), p["s"],
                                                                 torch.bfloat16),
                    xq_tensor, fp8_ok)
                meths["vllm_cutlass_fp8_channel"] = (
                    pack_cc, lambda p, xi: ops.cutlass_scaled_mm(xi[0], p["wq"].t(), xi[1], p["s"], torch.bfloat16),
                    xq_token, fp8_ok)

                def run_dyn_t(p, x):
                    xq, xs = ops.scaled_fp8_quant(x)
                    return ops.cutlass_scaled_mm(xq, p["wq"].t(), xs, p["s"], torch.bfloat16)

                def run_dyn_c(p, x):
                    xq, xs = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                    return ops.cutlass_scaled_mm(xq, p["wq"].t(), xs, p["s"], torch.bfloat16)

                meths["vllm_cutlass_fp8_tensor+dynq"] = (pack_ct, run_dyn_t, ident, fp8_ok)
                meths["vllm_cutlass_fp8_channel+dynq"] = (pack_cc, run_dyn_c, ident, fp8_ok)
            try:
                from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
                    apply_fp8_marlin_linear,
                    prepare_fp8_layer_for_marlin,
                )

                def pack_marlin(w):
                    q, s = per_channel_fp8(w)
                    layer = torch.nn.Module()
                    layer.output_size_per_partition = w.shape[0]
                    layer.input_size_per_partition = w.shape[1]
                    layer.orig_dtype = torch.bfloat16
                    layer.weight = torch.nn.Parameter(q, requires_grad=False)
                    layer.weight_scale = torch.nn.Parameter(s.float().contiguous(), requires_grad=False)
                    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
                    return {"w": layer.weight.data, "s": layer.weight_scale.data, "_ws": layer.workspace,
                            "_n": w.shape[0], "_k": w.shape[1]}

                def run_marlin(p, x):
                    return apply_fp8_marlin_linear(x, p["w"], p["s"], p["_ws"], size_n=p["_n"], size_k=p["_k"],
                                                   bias=None)

                meths["marlin_w8a16_fp8"] = (pack_marlin, run_marlin, ident, always)
            except Exception as exc:
                self.results["notes"].append(f"marlin unavailable: {exc}"[:300])
            if fp4_cutlass:
                def pack_fp4(w):
                    gw = (448.0 * 6.0 / w.float().abs().amax().clamp(min=1e-12)).reshape(1).float()
                    wq, wsf = ops.scaled_fp4_quant(w.contiguous(), gw)
                    return {"wq": wq, "wsf": wsf, "gw": gw}

                def xprep_fp4(x):
                    gx = (448.0 * 6.0 / x.float().abs().amax().clamp(min=1e-12)).reshape(1).float()
                    xq, xsf = ops.scaled_fp4_quant(x.contiguous(), gx)
                    return (xq, xsf, gx, x)

                def run_fp4(p, xi):
                    alpha = 1.0 / (xi[2] * p["gw"])
                    return ops.cutlass_scaled_fp4_mm(xi[0], p["wq"], xi[1], p["wsf"], alpha, torch.bfloat16)

                def run_fp4_q(p, xi):
                    xq, xsf = ops.scaled_fp4_quant(xi[3], xi[2])
                    alpha = 1.0 / (xi[2] * p["gw"])
                    return ops.cutlass_scaled_fp4_mm(xq, p["wq"], xsf, p["wsf"], alpha, torch.bfloat16)

                def fp4_ok(m, n, k):
                    return k % 32 == 0 and n % 16 == 0

                meths["vllm_cutlass_nvfp4"] = (pack_fp4, run_fp4, xprep_fp4, fp4_ok)
                meths["vllm_cutlass_nvfp4+actq"] = (pack_fp4, run_fp4_q, xprep_fp4, fp4_ok)
            try:
                from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
                    shape_dynamic_skinny_gemm,
                )

                if shape_dynamic_skinny_gemm.is_available():
                    meths["cute_skinny_bf16"] = (
                        lambda w: {"w": w}, lambda p, x: shape_dynamic_skinny_gemm(x, p["w"]), ident,
                        lambda m, n, k: m in SKINNY_MS)
                else:
                    self.results["notes"].append("cute skinny gemm: cuteDSL unavailable")
            except Exception as exc:
                self.results["notes"].append(f"cute skinny unavailable: {exc}"[:300])
        del dev, num_sms
        return meths

    # ------------------------------------------------------------------ measurements
    def bandwidth(self):
        n = 1 << 30  # 1 Gi bf16 elements = 2 GiB
        a = torch.empty(n, dtype=torch.bfloat16, device=self.dev).normal_()
        b = torch.empty_like(a)
        out = {}
        for name, fn, nbytes in [("copy_read_write", lambda: b.copy_(a), 4 * n),
                                 ("sum_read", lambda: a.sum(dtype=torch.float32), 2 * n)]:
            us, us_min = self.graph_time([fn], target_ms=100)
            out[name] = {"us": us, "tbps": nbytes / (us * 1e-6) / 1e12, "tbps_best": nbytes / (us_min * 1e-6) / 1e12}
        # graph node spacing: a trivial kernel repeated
        tiny = torch.zeros(16, device=self.dev)
        us, _ = self.graph_time([lambda: tiny.add_(1.0)] * 512, target_ms=20)
        out["graph_node_trivial_kernel_us"] = us
        a = b = None  # release the 4 GiB before the GEMM passes
        torch.cuda.empty_cache()
        self.results["bandwidth"] = out
        log({"bandwidth": out})

    def act_quant(self):
        ops = self.vllm_ops
        if ops is None:
            return
        for k in (2560, 6144, 10240, 320, 640):
            for m in (1, 4, 8, 16, 32, 64):
                x = torch.randn(m, k, dtype=torch.bfloat16, device=self.dev)
                row = {"K": k, "M": m}
                for name, fn in [("fp8_dynamic_tensor", lambda: ops.scaled_fp8_quant(x)),
                                 ("fp8_dynamic_token", lambda: ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)),
                                 ("nvfp4", lambda: ops.scaled_fp4_quant(x, torch.ones(1, device=self.dev)))]:
                    try:
                        row[name] = self.graph_time([fn] * 64, target_ms=10)[0]
                    except Exception as exc:
                        row[name] = f"error: {type(exc).__name__}: {exc}"[:200]
                self.results["act_quant"].append(row)
                log({"act_quant": row})

    def run_shape(self, name, n, k, names, pad, meths):
        w, source = self.load_weight(names, n, k, pad)
        log({"shape": name, "N": n, "K": k, "weights": source})
        wf = w.float()
        packs = {}
        for mname, (pack, run, xprep, allowed) in meths.items():
            if self.left() < 30:
                break
            try:
                t0 = time.time()
                p0 = pack(w)
                nbytes = sum(t.numel() * t.element_size() for key, t in p0.items()
                             if isinstance(t, torch.Tensor) and not key.startswith("_") and key != "gw")
                copies = self.copies_for(nbytes)
                packs[mname] = (p0, self.replicate(p0, copies), nbytes, copies, time.time() - t0)
            except Exception as exc:
                self.record_error(name, None, mname, exc)
        for m in MS:
            x = torch.randn(m, k, dtype=torch.bfloat16, device=self.dev)
            ref = x.float() @ wf.t()
            ref_norm = ref.norm().item()
            for mname, (pack, run, xprep, allowed) in meths.items():
                if mname not in packs or not allowed(m, n, k):
                    continue
                if self.left() < 20:
                    self.results["notes"].append(f"deadline reached at {name} M={m} {mname}")
                    return
                if mname == "cute_skinny_bf16" and self.skinny_compile_seconds > self.args.skinny_compile_budget:
                    continue
                p0, reps, nbytes, copies, _ = packs[mname]
                try:
                    if mname.startswith("triton_"):
                        scale = "s" in p0
                        best = None
                        for cfg in TRITON_CONFIGS:
                            bn, bk, _, stages = cfg
                            bm = 16 if m <= 16 else 64
                            smem = stages * (bm * bk * 2 + bn * bk * p0["w"].element_size())
                            if smem > 96 * 1024:
                                continue
                            calls = [(lambda p=p: triton_gemm(x, p["w"], p.get("s") if scale else None, cfg,
                                                              self.num_sms)) for p in reps]
                            try:
                                us, us_min = self.graph_time(calls)
                            except Exception as exc:
                                self.record_error(name, m, f"{mname}{cfg}", exc)
                                continue
                            if best is None or us < best[0]:
                                best = (us, us_min, cfg)
                        if best is None:
                            continue
                        us, us_min, cfg = best
                        y = triton_gemm(x, p0["w"], p0.get("s") if scale else None, cfg, self.num_sms)
                        extra = {"config": cfg}
                    else:
                        xi = xprep(x)
                        prev_blas = None
                        if mname == "bf16_linear_altblas":
                            prev_blas = torch.backends.cuda.preferred_blas_library()
                            torch.backends.cuda.preferred_blas_library(self.alt_blas)
                        try:
                            t0 = time.time()
                            y = run(p0, xi)
                            torch.cuda.synchronize()
                            if mname == "cute_skinny_bf16":
                                self.skinny_compile_seconds += time.time() - t0
                            calls = [(lambda p=p: run(p, xi)) for p in reps]
                            us, us_min = self.graph_time(calls)
                        finally:
                            if prev_blas is not None:
                                torch.backends.cuda.preferred_blas_library(prev_blas)
                        extra = {}
                    err = ((y.float() - ref).norm().item() / ref_norm) if ref_norm > 0 else None
                    row = {"shape": name, "N": n, "K": k, "M": m, "method": mname, "us": us, "us_min": us_min,
                           "weight_bytes": nbytes, "tbps": nbytes / (us * 1e-6) / 1e12, "copies": copies,
                           "rel_err": err, "weights": source, **extra}
                    self.results["results"].append(row)
                    log(row)
                except Exception as exc:
                    self.record_error(name, m, mname, exc)
                self.flush()
        del packs, wf
        torch.cuda.empty_cache()

    def record_error(self, shape, m, method, exc):
        row = {"shape": shape, "M": m, "method": method, "error": f"{type(exc).__name__}: {exc}"[:500],
               "trace": traceback.format_exc()[-1500:]}
        self.results["errors"].append(row)
        log({"error": row})

    def main(self):
        self.env()
        default_blas = str(torch.backends.cuda.preferred_blas_library())
        self.alt_blas = "cublas" if "lt" in default_blas.lower() else "cublaslt"
        self.results["env"]["default_blas"] = default_blas
        self.results["env"]["alt_blas"] = self.alt_blas
        try:
            import vllm._custom_ops as ops
            try:
                import vllm._C_stable_libtorch  # noqa: F401  (registers torch.ops._C on CUDA)
            except Exception as exc:
                self.results["notes"].append(f"_C_stable_libtorch import: {exc}"[:300])
            self.vllm_ops = ops
        except Exception as exc:
            self.results["notes"].append(f"vllm ops unavailable: {exc}"[:500])
            log({"note": "vllm ops unavailable", "error": str(exc)[:500]})
        self.index = {}
        if self.args.model_dir:
            try:
                with open(os.path.join(self.args.model_dir, "model.safetensors.index.json")) as fh:
                    self.index = json.load(fh)["weight_map"]
            except Exception as exc:
                self.results["notes"].append(f"index unavailable: {exc}"[:300])
        if self.args.bandwidth:
            try:
                self.bandwidth()
            except Exception as exc:
                self.record_error("bandwidth", None, "bandwidth", exc)
            self.flush(force=True)
        meths = self.methods()
        if self.args.methods:
            wanted = self.args.methods.split(",")
            meths = {k: v for k, v in meths.items() if k in wanted}
        self.results["methods"] = list(meths)
        log({"methods": list(meths)})
        order = SHAPES if not self.args.shapes else [s for s in SHAPES if s[0] in self.args.shapes.split(",")]
        for name, n, k, names, pad in order:
            if self.left() < 60:
                self.results["notes"].append(f"deadline: skipped {name}")
                continue
            if not meths:
                break
            try:
                self.run_shape(name, n, k, names, pad, meths)
            except Exception as exc:
                self.record_error(name, None, "run_shape", exc)
            self.flush(force=True)
        if self.args.act_quant and self.left() > 60:
            try:
                self.act_quant()
            except Exception as exc:
                self.record_error("act_quant", None, "act_quant", exc)
        self.results["finished"] = True
        self.flush(force=True)
        log({"done": True, "elapsed": time.time() - self.start, "rows": len(self.results["results"]),
             "errors": len(self.results["errors"])})


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--deadline-seconds", type=float, default=1300)
    ap.add_argument("--skinny-compile-budget", type=float, default=240)
    ap.add_argument("--shapes", default="")
    ap.add_argument("--methods", default="")
    ap.add_argument("--bandwidth", action="store_true")
    ap.add_argument("--act-quant", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    Bench(args).main()


if __name__ == "__main__":
    sys.exit(cli())
