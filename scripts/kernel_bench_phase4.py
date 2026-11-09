#!/usr/bin/env python3
"""Phase-4 kernel microbenchmark (kernel work, phase 4): fused MoE, HC fusion, top-k-first sampler.

Runs standalone on one RTX PRO 6000 (launcher: scripts/modal_kernel_bench_phase4.py). Parts:

- ``moe``: FlashInfer's fused SM120 NvFp4 MoE (b12x, ``B12xMoEWrapper``) against the served FlashInfer
  CUTLASS path (activation FP4 quantization + ``cutlass_fused_moe``) on one layer's real NvFp4 experts,
  M = 1-32 tokens, uniform and Zipf routing, CUDA-graph replay over 16 routings (the working set is far
  above L2). Error of both against an FP32 reference built from the dequantized checkpoint weights.
  Also the b12x compile time per token count and its workspace size.
- ``draft``: the MTP draft layer's BF16 experts quantized to NvFp4 exactly as the serving overlay does
  (per-expert global scale, E4M3 block scales), on b12x; error against the BF16 layer.
- ``hc``: the fused silu + HC up + gate-mix kernel (arc_infra/serving_kernels_runtime.py) against the
  pinned three-kernel chain (pinned hc.py kernels + cuBLAS), real HC up weights, M = 1-64.
- ``sampler``: top-k-first (the runtime helper) against the pinned sort path (B < 8) and the pinned Triton
  path (B >= 8) at top-k 20 / top-p 0.95 on a 248,320-token vocabulary, B = 1-32: time, kept-set agreement
  and identical Gumbel samples.
"""

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import traceback
from collections import defaultdict

import torch
import torch.nn.functional as F

E, H, INTER, TOPK, VOCAB = 512, 2560, 640, 10, 248320
MS = [1, 6, 4, 24, 2, 8, 16, 32, 48]  # priority order (the deadline may cut the tail); 48: dynamic kernel
FP8 = torch.float8_e4m3fn
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def log(obj):
    print(json.dumps(obj, default=str), flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rel(a, b):
    b = b.float()
    return ((a.float() - b).norm() / b.norm().clamp(min=1e-12)).item()


class Bench:
    def __init__(self, args):
        self.args = args
        self.dev = torch.device("cuda")
        self.start = time.time()
        self.deadline = self.start + args.deadline_seconds
        self.results = {"env": {}, "moe": [], "moe_error": {}, "moe_compile": [], "draft": {}, "hc": [],
                        "hc_error": [], "sampler": [], "errors": [], "notes": []}
        self.helper = load_module("arc_serving_kernels", args.helper)
        # The served b12x configuration (static kernels up to 32 tokens, dynamic above), set before FlashInfer
        # reads it on the first b12x call.
        os.environ[self.helper.B12X_CUTOVER_ENV] = str(self.helper.B12X_STATIC_CUTOVER_PAIRS)
        self.index = None
        if args.model_dir:
            with open(os.path.join(args.model_dir, "model.safetensors.index.json")) as fh:
                self.index = json.load(fh)["weight_map"]
        self.last_flush = 0.0

    def left(self):
        return self.deadline - time.time()

    def flush(self, force=False):
        if not force and time.time() - self.last_flush < 20:
            return
        self.last_flush = time.time()
        self.results["elapsed_seconds"] = time.time() - self.start
        with open(self.args.out + ".tmp", "w") as fh:
            json.dump(self.results, fh, default=str)
        os.replace(self.args.out + ".tmp", self.args.out)

    def env(self):
        props = torch.cuda.get_device_properties(0)
        env = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": props.name,
               "capability": list(torch.cuda.get_device_capability()), "sms": props.multi_processor_count,
               "l2_bytes": getattr(props, "L2_cache_size", None)}
        for mod in ("vllm", "flashinfer", "cutlass", "triton"):
            try:
                env[mod] = __import__(mod).__version__
            except Exception as exc:
                env[mod] = f"unavailable: {type(exc).__name__}: {exc}"[:200]
        try:
            import flashinfer_jit_cache  # noqa: F401
            env["flashinfer_jit_cache"] = getattr(flashinfer_jit_cache, "__version__", "present")
        except Exception as exc:
            env["flashinfer_jit_cache"] = f"unavailable: {type(exc).__name__}"
        self.results["env"] = env
        log({"env": env})

    # ------------------------------------------------------------------ timing
    def graph_time(self, calls, target_ms=20.0):
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
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        graph.replay()
        e1.record()
        e1.synchronize()
        reps = max(1, min(100, int(target_ms / max(e0.elapsed_time(e1), 1e-3))))
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

    @staticmethod
    def eager_time(fn, reps=30):
        fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) / reps * 1e6)
        return statistics.median(samples), min(samples)

    # ------------------------------------------------------------------ weights
    def tensors(self, names):
        from safetensors import safe_open

        by_file = defaultdict(list)
        for name in names:
            by_file[self.index[name]].append(name)
        out = {}
        for file, group in by_file.items():
            with safe_open(os.path.join(self.args.model_dir, file), framework="pt", device="cpu") as fh:
                for name in group:
                    out[name] = fh.get_tensor(name)
        return out

    def load_nvfp4_layer(self, layer):
        base = f"model.language_model.layers.{layer}.mlp.experts."
        suffixes = ("weight", "weight_scale", "weight_scale_2", "input_scale")
        names = [f"{base}{e}.{p}.{s}" for e in range(E) for p in ("gate_proj", "up_proj", "down_proj") for s in suffixes]
        t0 = time.time()
        t = self.tensors(names)

        def g(e, p, s):
            value = t[f"{base}{e}.{p}.{s}"]
            return value.view(torch.uint8) if value.dtype == FP8 else value  # CPU cat/stack on raw bytes

        w13 = torch.stack([torch.cat([g(e, "gate_proj", "weight"), g(e, "up_proj", "weight")]) for e in range(E)])
        w13_sf = torch.stack([torch.cat([g(e, "gate_proj", "weight_scale"), g(e, "up_proj", "weight_scale")])
                              for e in range(E)]).view(FP8)
        w13_s2 = torch.tensor([[g(e, "gate_proj", "weight_scale_2").item(), g(e, "up_proj", "weight_scale_2").item()]
                               for e in range(E)])
        a13 = torch.tensor([[g(e, "gate_proj", "input_scale").item(), g(e, "up_proj", "input_scale").item()]
                            for e in range(E)])
        w2 = torch.stack([g(e, "down_proj", "weight") for e in range(E)])
        w2_sf = torch.stack([g(e, "down_proj", "weight_scale") for e in range(E)]).view(FP8)
        w2_s2 = torch.tensor([g(e, "down_proj", "weight_scale_2").item() for e in range(E)])
        a2 = torch.tensor([g(e, "down_proj", "input_scale").item() for e in range(E)])
        d = self.dev
        layer_w = {"w13": w13.to(d), "w13_sf": w13_sf.to(d), "w13_s2": w13_s2.float().to(d), "a13": a13.float().to(d),
                   "w2": w2.to(d), "w2_sf": w2_sf.to(d), "w2_s2": w2_s2.float().to(d), "a2": a2.float().to(d)}
        info = {"layer": layer, "load_seconds": time.time() - t0, "w13": list(w13.shape), "w13_dtype": str(w13.dtype),
                "w13_sf_dtype": str(w13_sf.dtype), "w2": list(w2.shape),
                "gate_up_scale2_equal_fraction": (w13_s2[:, 0] == w13_s2[:, 1]).float().mean().item(),
                "input_scale_range": [a13.min().item(), a13.max().item(), a2.min().item(), a2.max().item()]}
        self.results["notes"].append({"nvfp4_layer": info})
        log({"nvfp4_layer": info})
        return layer_w

    @staticmethod
    def dequant(packed, sf, s2):
        """[N, K/2] uint8 E2M1 pairs (low nibble first), [N, K/16] E4M3 scales, FP32 scalar -> [N, K] FP32."""
        lut = E2M1.to(packed.device)
        lo = lut[(packed & 0xF).long()]
        hi = lut[(packed >> 4).long()]
        vals = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
        scale = sf.float().repeat_interleave(16, dim=1)
        return vals * scale * s2

    def moe_reference(self, lw, x, ids, weights):
        out = torch.zeros((x.shape[0], H), dtype=torch.float32, device=x.device)
        cache = {}
        for t in range(x.shape[0]):
            for j in range(ids.shape[1]):
                e = int(ids[t, j])
                if e not in cache:
                    g = self.dequant(lw["w13"][e, :INTER], lw["w13_sf"][e, :INTER], lw["w13_s2"][e, 0])
                    u = self.dequant(lw["w13"][e, INTER:], lw["w13_sf"][e, INTER:], lw["w13_s2"][e, 1])
                    dn = self.dequant(lw["w2"][e], lw["w2_sf"][e], lw["w2_s2"][e])
                    cache[e] = (g, u, dn)
                g, u, dn = cache[e]
                xt = x[t].float()
                out[t] += weights[t, j] * (dn @ (F.silu(g @ xt) * (u @ xt)))
        return out

    # ------------------------------------------------------------------ backends
    def b12x_prepare(self, w13, w13_sf, s13, w2, w2_sf, s2):
        """vLLM's FlashInfer-B12X path: [w1; w3] -> [w3; w1], swizzled block scales with the per-expert global
        scale baked in (as FlashInferB12xExperts.process_weights_after_loading), MMA-layout views."""
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

        w13 = torch.cat([w13[:, INTER:], w13[:, :INTER]], dim=1).contiguous()
        w13_sf = torch.cat([w13_sf[:, INTER:], w13_sf[:, :INTER]], dim=1).contiguous()
        sw13 = swizzle_blockscale(w13_sf)
        sw2 = swizzle_blockscale(w2_sf.contiguous())
        sw13 = (sw13.float() * s13.view(-1, 1, 1)).to(sw13.dtype)
        sw2 = (sw2.float() * s2.view(-1, 1, 1)).to(sw2.dtype)
        n1, m1, k1 = sw13.shape
        n2, m2, k2 = sw2.shape
        sf1 = convert_sf_to_mma_layout(sw13.reshape(n1 * m1, k1), m=m1, k=k1 * 16, num_groups=n1)
        sf2 = convert_sf_to_mma_layout(sw2.reshape(n2 * m2, k2), m=m2, k=k2 * 16, num_groups=n2)
        ones = torch.ones(E, dtype=torch.float32, device=self.dev)
        return {"w1": w13, "sf1": sf1, "w2": w2.contiguous(), "sf2": sf2, "alpha1": ones, "alpha2": ones.clone(),
                "fc2": ones.clone(), "_keep": (sw13, sw2)}

    def b12x_wrapper(self):
        from flashinfer.fused_moe import B12xMoEWrapper

        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        wrapper = B12xMoEWrapper(num_experts=E, top_k=TOPK, hidden_size=H, intermediate_size=INTER,
                                 use_cuda_graph=True, max_num_tokens=4096, activation="silu")
        torch.cuda.synchronize()
        self.results["notes"].append({"b12x_workspace_bytes_max_tokens_4096": torch.cuda.memory_allocated() - before})
        return wrapper

    @staticmethod
    def b12x_run(wrapper, p, x, ids, wts):
        return wrapper.run(x=x, w1_weight=p["w1"], w1_weight_sf=p["sf1"], w1_alpha=p["alpha1"],
                           fc2_input_scale=p["fc2"], w2_weight=p["w2"], w2_weight_sf=p["sf2"], w2_alpha=p["alpha2"],
                           token_selected_experts=ids, token_final_scales=wts)

    def cutlass_prepare(self, lw):
        """vLLM's FlashInfer-CUTLASS path: [w3; w1], swizzled scales, one activation global scale per input,
        g = weight_scale_2 x input_scale (FlashInferExperts.process_weights_after_loading)."""
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

        w13 = torch.cat([lw["w13"][:, INTER:], lw["w13"][:, :INTER]], dim=1).contiguous()
        w13_sf = torch.cat([lw["w13_sf"][:, INTER:], lw["w13_sf"][:, :INTER]], dim=1).contiguous()
        a1 = lw["a13"].max().repeat(E)
        a2 = lw["a2"].max().repeat(E)
        s13 = lw["w13_s2"][:, 0].contiguous()
        return {"w1": w13, "sf1": swizzle_blockscale(w13_sf), "w2": lw["w2"].contiguous(),
                "sf2": swizzle_blockscale(lw["w2_sf"].contiguous()), "a1_gscale": (1.0 / a1).contiguous(),
                "a2_gscale": (1.0 / a2).contiguous(), "g1": (s13 * a1).contiguous(), "g2": (lw["w2_s2"] * a2).contiguous()}

    @staticmethod
    def cutlass_run(p, x, ids, wts, out):
        from flashinfer.fused_moe import cutlass_fused_moe
        from vllm import _custom_ops as ops

        xq, xsf = ops.scaled_fp4_quant(x, p["a1_gscale"][:1])
        cutlass_fused_moe(input=xq, token_selected_experts=ids, token_final_scales=wts,
                          fc1_expert_weights=p["w1"].view(torch.long), fc2_expert_weights=p["w2"].view(torch.long),
                          output_dtype=torch.bfloat16,
                          quant_scales=[p["a1_gscale"], p["sf1"].view(torch.int32), p["g1"], p["a2_gscale"],
                                        p["sf2"].view(torch.int32), p["g2"]],
                          input_sf=xsf, output=out)
        return out

    @staticmethod
    def routes(m, count, kind, gen, dev):
        if kind == "uniform":
            ids = torch.stack([torch.stack([torch.randperm(E, generator=gen)[:TOPK] for _ in range(m)])
                               for _ in range(count)])
        else:  # Zipf(1) over a fixed random expert order: correlated routing, fewer distinct experts
            order = torch.randperm(E, generator=gen)
            prob = 1.0 / torch.arange(1, E + 1, dtype=torch.float64)
            ids = torch.stack([torch.stack([order[torch.multinomial(prob, TOPK, replacement=False, generator=gen)]
                                            for _ in range(m)]) for _ in range(count)])
        wts = torch.softmax(torch.randn(count, m, TOPK, generator=gen), dim=-1)
        return ids.to(torch.int32).to(dev), wts.float().to(dev)

    # ------------------------------------------------------------------ parts
    def part_moe(self):
        lw = self.load_nvfp4_layer(self.args.layer)
        gen = torch.Generator().manual_seed(0)
        x_all = (torch.randn(max(MS), H, generator=gen)).to(torch.bfloat16).to(self.dev)
        b12x = self.b12x_prepare(lw["w13"], lw["w13_sf"], lw["w13_s2"][:, 0].contiguous(), lw["w2"], lw["w2_sf"],
                                 lw["w2_s2"])
        wrapper = self.b12x_wrapper()
        cutlass, cutlass_ok = None, False
        try:
            cutlass = self.cutlass_prepare(lw)
            t0 = time.time()
            out = torch.empty(4, H, dtype=torch.bfloat16, device=self.dev)
            ids, wts = self.routes(4, 1, "uniform", gen, self.dev)
            self.cutlass_run(cutlass, x_all[:4], ids[0], wts[0], out)
            torch.cuda.synchronize()
            cutlass_ok = True
            self.results["notes"].append({"cutlass_first_call_seconds": time.time() - t0})
        except Exception as exc:
            self.results["errors"].append(f"cutlass unavailable: {type(exc).__name__}: {str(exc)[:400]}")
            log({"cutlass_error": traceback.format_exc()[-1500:]})
        # Error against the dequantized FP32 reference (4 tokens, uniform routes) and at M = 1.
        for m in (1, 4):
            ids, wts = self.routes(m, 1, "uniform", gen, self.dev)
            x = x_all[:m]
            ref = self.moe_reference(lw, x, ids[0], wts[0])
            row = {"m": m}
            yb = self.b12x_run(wrapper, b12x, x, ids[0], wts[0]).clone()
            row["b12x_vs_fp32_ref"] = rel(yb, ref)
            if cutlass_ok:
                yc = self.cutlass_run(cutlass, x, ids[0], wts[0], torch.empty(m, H, dtype=torch.bfloat16, device=self.dev))
                row["cutlass_vs_fp32_ref"] = rel(yc, ref)
                row["b12x_vs_cutlass"] = rel(yb, yc)
                row["max_abs_b12x_minus_cutlass"] = (yb.float() - yc.float()).abs().max().item()
            row["ref_rms"] = ref.pow(2).mean().sqrt().item()
            self.results["moe_error"][f"M{m}"] = row
            log({"moe_error": row})
        self.flush(True)
        for m in MS:
            if self.left() < 60:
                self.results["notes"].append(f"moe stopped before M={m}: time")
                break
            x = x_all[:m].contiguous()
            ids0, wts0 = self.routes(m, 1, "uniform", gen, self.dev)
            t0 = time.time()
            self.b12x_run(wrapper, b12x, x, ids0[0], wts0[0])
            torch.cuda.synchronize()
            self.results["moe_compile"].append({"m": m, "b12x_first_call_seconds": time.time() - t0})
            for kind in ("uniform", "zipf"):
                ids, wts = self.routes(m, 16, kind, gen, self.dev)
                row = {"m": m, "routing": kind,
                       "distinct_experts_per_call": statistics.mean(len(set(ids[r].flatten().tolist())) for r in range(16))}
                try:
                    calls = [lambda r=r: self.b12x_run(wrapper, b12x, x, ids[r], wts[r]) for r in range(16)]
                    row["b12x_us"], row["b12x_us_min"] = self.graph_time(calls)
                except Exception as exc:
                    row["b12x_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                if cutlass_ok:
                    outs = [torch.empty(m, H, dtype=torch.bfloat16, device=self.dev) for _ in range(16)]
                    try:
                        calls = [lambda r=r: self.cutlass_run(cutlass, x, ids[r], wts[r], outs[r]) for r in range(16)]
                        row["cutlass_us"], row["cutlass_us_min"] = self.graph_time(calls)
                        row["b12x_over_cutlass"] = row.get("b12x_us", float("nan")) / row["cutlass_us"]
                    except Exception as exc:
                        row["cutlass_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                self.results["moe"].append(row)
                log({"moe": row})
                self.flush()

    def part_draft(self):
        from vllm import _custom_ops as ops

        t = self.tensors(["mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj"])
        gu, dn = t["mtp.layers.0.mlp.experts.gate_up_proj"], t["mtp.layers.0.mlp.experts.down_proj"]
        info = {"gate_up_shape": list(gu.shape), "down_shape": list(dn.shape), "dtype": str(gu.dtype)}
        # vLLM layout: w13 [E, 2I, H] ([gate; up]), w2 [E, H, I]; a transposed fused checkpoint is [E, H, 2I].
        if tuple(gu.shape) == (E, H, 2 * INTER):
            gu = gu.transpose(1, 2)
            info["transposed"] = True
        if tuple(dn.shape) == (E, INTER, H):
            dn = dn.transpose(1, 2)
        w13 = gu.contiguous().to(torch.bfloat16).to(self.dev)
        w2 = dn.contiguous().to(torch.bfloat16).to(self.dev)
        del gu, dn, t

        def quantize(w):  # vLLM's online NvFp4 MoE quantizer (online/nvfp4.py)
            amax = w.flatten(1).abs().amax(dim=-1).float().clamp_min(1e-8)
            gs = (6.0 * 448.0) / amax
            q, sf = zip(*[ops.scaled_fp4_quant(w[e], gs[e:e + 1], is_sf_swizzled_layout=False) for e in range(E)])
            return torch.stack(q), torch.stack(sf), (1.0 / gs).float()

        q13, sf13, s13 = quantize(w13)
        q2, sf2, s2 = quantize(w2)
        info["nvfp4_bytes"] = sum(x.numel() * x.element_size() for x in (q13, sf13, q2, sf2))
        info["bf16_bytes"] = (w13.numel() + w2.numel()) * 2
        p = self.b12x_prepare(q13, sf13.view(FP8), s13, q2, sf2.view(FP8), s2)
        wrapper = self.b12x_wrapper()
        gen = torch.Generator().manual_seed(1)
        for m in (1, 4):
            x = torch.randn(m, H, generator=gen).to(torch.bfloat16).to(self.dev)
            ids, wts = self.routes(m, 1, "uniform", gen, self.dev)
            ref = torch.zeros(m, H, device=self.dev)
            for tok in range(m):
                for j in range(TOPK):
                    e = int(ids[0, tok, j])
                    h = w13[e].float() @ x[tok].float()
                    ref[tok] += wts[0, tok, j] * (w2[e].float() @ (F.silu(h[:INTER]) * h[INTER:]))
            y = self.b12x_run(wrapper, p, x, ids[0], wts[0])
            info[f"M{m}_nvfp4_b12x_vs_bf16"] = rel(y, ref)
            info[f"M{m}_cosine"] = F.cosine_similarity(y.float().flatten(), ref.flatten(), dim=0).item()
        self.results["draft"] = info
        log({"draft": info})
        del w13, w2, p
        torch.cuda.empty_cache()

    def part_hc(self):
        hc = None
        if self.args.pinned and os.path.isfile(os.path.join(self.args.pinned, "hc.py")):
            hc = load_module("pinned_hc", os.path.join(self.args.pinned, "hc.py"))
        else:
            self.results["errors"].append("pinned hc.py missing: HC baseline skipped")
            return
        name = "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight"
        w = self.tensors([name])[name].to(torch.bfloat16).to(self.dev)
        dim, rank = w.shape
        copies = max(2, math.ceil((256 << 20) / (w.numel() * 2)))
        ws = [w.clone() for _ in range(copies)]
        gen = torch.Generator().manual_seed(2)
        helper = self.helper

        def chain(xn, lora, weight):
            silu = hc._hc_silu(lora, 4)
            gate = F.linear(silu, weight)
            return hc._hc_gate_mix(xn, gate, 4)

        configs = [(16, 16, 4, 1), (16, 32, 4, 1), (16, 16, 8, 1), (16, 32, 8, 2)]
        for m in [1, 2, 4, 6, 8, 16, 24, 32, 48, 64]:
            if self.left() < 45:
                self.results["notes"].append(f"hc stopped before M={m}: time")
                break
            xn = torch.randn(m, dim, generator=gen).to(torch.bfloat16).to(self.dev)
            down = (torch.randn(m, 336, generator=gen) * 4).to(torch.bfloat16).to(self.dev)
            lora = down[:, :rank]
            ref = chain(xn, lora, w)
            row = {"m": m}
            row["pinned_chain_us"], _ = self.graph_time([lambda i=i: chain(xn, lora, ws[i]) for i in range(copies)])
            for cfg in configs:
                bm, bi, warps, stages = cfg
                fused = helper.hc_up_mix_triton(xn, lora, w, 4, block_m=bm, block_i=bi, num_warps=warps,
                                                num_stages=stages)
                diff = fused.float() - ref.float()
                key = f"fused_bm{bm}_bi{bi}_w{warps}_s{stages}"
                row[key + "_us"], _ = self.graph_time(
                    [lambda i=i: helper.hc_up_mix_triton(xn, lora, ws[i], 4, block_m=bm, block_i=bi, num_warps=warps,
                                                         num_stages=stages) for i in range(copies)])
                self.results["hc_error"].append({"m": m, "config": key, "rel_error": rel(fused, ref),
                                                 "max_abs": diff.abs().max().item(),
                                                 "bitwise_equal": (fused == ref).float().mean().item()})
            self.results["hc"].append(row)
            log({"hc": row})
            self.flush()

    def part_sampler(self):
        triton_mod = None
        if self.args.pinned and os.path.isfile(os.path.join(self.args.pinned, "topk_topp_triton.py")):
            try:
                triton_mod = load_module("pinned_topk_topp_triton", os.path.join(self.args.pinned, "topk_topp_triton.py"))
            except Exception as exc:
                self.results["errors"].append(f"pinned triton sampler import: {type(exc).__name__}: {exc}"[:300])
        if triton_mod is None:
            from vllm.v1.sample.ops import topk_topp_triton as triton_mod
            self.results["notes"].append("sampler triton baseline: stock vLLM 0.30 file")
        helper = self.helper

        def pinned_sort(logits, k, p):  # the pinned apply_top_k_top_p_pytorch (identical code)
            logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
            top_k_mask = logits_sort.size(1) - k.to(torch.long)
            top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
            top_k_mask = logits_sort < top_k_mask
            logits_sort.masked_fill_(top_k_mask, -float("inf"))
            probs_sort = logits_sort.softmax(dim=-1)
            probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
            top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
            top_p_mask[:, -1] = False
            logits_sort.masked_fill_(top_p_mask, -float("inf"))
            return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)

        gen = torch.Generator().manual_seed(3)
        gpu_gen = torch.Generator(device=self.dev).manual_seed(4)
        for b in [1, 2, 4, 6, 8, 16, 24, 32]:
            if self.left() < 30:
                self.results["notes"].append(f"sampler stopped before B={b}: time")
                break
            base = torch.randn(b, VOCAB, generator=gen) * 2.5
            hot = torch.randint(0, VOCAB, (b, 40), generator=gen)
            base.scatter_(1, hot, torch.randn(b, 40, generator=gen).abs() * 4 + 8)
            base = (base.to(torch.bfloat16).float() / 0.7).to(self.dev)  # BF16 logits (ties), temperature 0.7
            k = torch.full((b,), 20, dtype=torch.int32, device=self.dev)
            p = torch.full((b,), 0.95, dtype=torch.float32, device=self.dev)
            copies = [base.clone() for _ in range(24)]
            it = iter(range(10**9))

            def fresh():
                return copies[next(it) % len(copies)]

            row = {"b": b}
            row["pinned_sort_us"], _ = self.eager_time(lambda: pinned_sort(fresh(), k, p))
            try:
                row["pinned_triton_us"], _ = self.eager_time(lambda: triton_mod.apply_top_k_top_p_triton(fresh(), k, p))
            except Exception as exc:
                row["pinned_triton_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            row["topk_first_us"], _ = self.eager_time(lambda: helper.topk_first_impl(fresh(), k, p, 20))
            row["served_baseline_us"] = row.get("pinned_triton_us") if b >= 8 else row["pinned_sort_us"]
            ref = pinned_sort(base.clone(), k, p)
            new = helper.topk_first_impl(base.clone(), k, p, 20)
            same_mask = (torch.isfinite(ref) == torch.isfinite(new))
            row["kept_set_identical_rows"] = int(same_mask.all(dim=1).sum().item())
            row["kept_tokens_ref"] = int(torch.isfinite(ref).sum().item())
            row["kept_tokens_new"] = int(torch.isfinite(new).sum().item())
            both = torch.isfinite(ref) & torch.isfinite(new)
            row["values_equal_where_kept"] = bool((ref[both] == new[both]).all().item())
            identical = 0
            for _ in range(16):  # the same Gumbel noise on both masks: the sampled tokens must agree
                noise = -torch.log(-torch.log(torch.rand(b, VOCAB, generator=gpu_gen, device=self.dev).clamp_min(1e-20)))
                identical += int(((ref + noise).argmax(-1) == (new + noise).argmax(-1)).sum().item())
            row["gumbel_identical_samples"] = identical
            row["gumbel_samples"] = 16 * b
            self.results["sampler"].append(row)
            log({"sampler": row})
            self.flush()

    def run(self):
        self.env()
        parts = {"moe": self.part_moe, "draft": self.part_draft, "hc": self.part_hc, "sampler": self.part_sampler}
        for name in self.args.parts.split(","):
            if self.left() < 30:
                self.results["notes"].append(f"part {name} skipped: time")
                continue
            t0 = time.time()
            try:
                parts[name]()
            except Exception as exc:
                self.results["errors"].append(f"{name}: {type(exc).__name__}: {str(exc)[:500]}")
                log({"part_error": name, "traceback": traceback.format_exc()[-3000:]})
            self.results.setdefault("part_seconds", {})[name] = time.time() - t0
            self.flush(True)
            torch.cuda.empty_cache()
        self.flush(True)


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--helper", required=True, help="arc_infra/serving_kernels_runtime.py")
    ap.add_argument("--pinned", default="", help="directory with the pinned hc.py and topk_topp_triton.py")
    ap.add_argument("--parts", default="moe,draft,hc,sampler")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--deadline-seconds", type=float, default=600)
    Bench(ap.parse_args()).run()


if __name__ == "__main__":
    cli()
