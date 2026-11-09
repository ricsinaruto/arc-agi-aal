"""Stage-1 microbenchmark of the routed-expert quantization on real ARC activations (28 September;
docs/validation-20260927/expert-quant.md).

Runs in the pinned serving runtime (its torch, vLLM and FlashInfer: the served kernels themselves) after the
capture server stopped, as ``python -m arena.expert_quant_bench`` (``arena.expert_quant`` starts it). Per layer:

- **identity**: the router weight of the served checkpoint against the BF16 source (byte equality), and ModelOpt
  RTN of the BF16 experts against the served NVFP4 tensors (codes, block scales, ``weight_scale_2``);
- **references** on held-out rows (``eval`` capture): the MoE output (routed experts only, served routing) with
  the true BF16 experts and with the dequantized served experts, both from BF16 activations in float32;
- **served**: FlashInfer CUTLASS NvFp4 W4A4 exactly as vLLM calls it (activation FP4 quantization with the layer's
  global input scale = the max of the checkpoint's per-expert ``input_scale``);
- **ARC input scale**: the same kernel with ``input_scale = amax / (6 * 448)`` of the ``calib`` capture
  (MoE input for FC1; the FC1 activation, computed with the dequantized weights, for FC2);
- **W4A16**: vLLM's Marlin NvFp4 MoE on the same NVFP4 weights (BF16 activations);
- **RTN + scale search**: calibration-free re-rounding of the BF16 experts (per-block scale candidates), weights
  emulated with BF16 activations;
- **timing**: CUTLASS and Marlin per call at M = 1-48 tokens with real routings (CUDA-graph replay over 16 calls).

Errors are ||y - ref|| / ||ref|| over all held-out rows, and cosine. Weight rounding = dequantized vs BF16;
activation quantization = served vs dequantized; total = served vs BF16.
"""

import argparse
import json
import os
import statistics
import sys
import time
import traceback

E, H, INTER, TOPK = 512, 2560, 640, 10
MS = (1, 6, 24, 4, 8, 16, 48)
FP8_MAX, FP4_MAX = 448.0, 6.0


def log(obj):
    print(json.dumps({"event": "expert_quant_bench", **obj}, default=str), flush=True)


class Bench:
    def __init__(self, args):
        import torch

        from arena import expert_quant_math as qm

        self.torch, self.qm, self.args = torch, qm, args
        self.dev = torch.device(args.device)
        self.start = time.time()
        self.deadline = self.start + args.deadline_seconds
        self.results = {"env": {}, "layers": {}, "errors": [], "notes": []}
        with open(os.path.join(args.model_dir, "model.safetensors.index.json")) as fh:
            self.index = json.load(fh)["weight_map"]
        with open(os.path.join(args.bf16_dir, "model.safetensors.index.json")) as fh:
            self.bf16_index = json.load(fh)["weight_map"]

    def left(self):
        return self.deadline - time.time()

    def flush(self):
        self.results["elapsed_seconds"] = time.time() - self.start
        tmp = self.args.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.results, fh, default=str)
        os.replace(tmp, self.args.out)

    def env(self):
        torch = self.torch
        env = {"torch": torch.__version__, "cuda": torch.version.cuda}
        if self.dev.type == "cuda":
            # The stopped server's processes release the GPU asynchronously: wait (up to 60 s) for 80 GB free.
            waited = time.time()
            while torch.cuda.mem_get_info()[0] < 80e9 and time.time() - waited < 60:
                time.sleep(2)
            props = torch.cuda.get_device_properties(0)
            env.update(gpu=props.name, capability=list(torch.cuda.get_device_capability()),
                       free_bytes=torch.cuda.mem_get_info()[0], waited_for_gpu=round(time.time() - waited, 1))
        for mod in ("vllm", "flashinfer"):
            try:
                env[mod] = __import__(mod).__version__
            except Exception as exc:  # noqa: BLE001
                env[mod] = f"unavailable: {type(exc).__name__}"
        self.results["env"] = env
        log({"env": env})

    # ------------------------------------------------------------------ loading
    def tensors(self, root, index, names):
        from collections import defaultdict

        from safetensors import safe_open

        by_file = defaultdict(list)
        for name in names:
            by_file[index[name]].append(name)
        out = {}
        for file, group in by_file.items():
            with safe_open(os.path.join(root, file), framework="pt", device="cpu") as fh:
                for name in group:
                    out[name] = fh.get_tensor(name)
        return out

    def load_served(self, layer):
        torch = self.torch
        fp8 = torch.float8_e4m3fn
        base = f"model.language_model.layers.{layer}.mlp.experts."
        suffixes = ("weight", "weight_scale", "weight_scale_2", "input_scale")
        names = [f"{base}{e}.{p}.{s}" for e in range(E) for p in ("gate_proj", "up_proj", "down_proj") for s in suffixes]
        t = self.tensors(self.args.model_dir, self.index, names)

        def g(e, p, s):
            value = t[f"{base}{e}.{p}.{s}"]
            return value.view(torch.uint8) if value.dtype == fp8 else value

        d = self.dev
        out = {
            "w13": torch.stack([torch.cat([g(e, "gate_proj", "weight"), g(e, "up_proj", "weight")]) for e in range(E)]),
            "w13_sf": torch.stack([torch.cat([g(e, "gate_proj", "weight_scale"), g(e, "up_proj", "weight_scale")])
                                   for e in range(E)]).view(fp8),
            "s2_gate": torch.tensor([g(e, "gate_proj", "weight_scale_2").item() for e in range(E)]),
            "s2_up": torch.tensor([g(e, "up_proj", "weight_scale_2").item() for e in range(E)]),
            "a13": torch.tensor([[g(e, "gate_proj", "input_scale").item(), g(e, "up_proj", "input_scale").item()]
                                 for e in range(E)]),
            "w2": torch.stack([g(e, "down_proj", "weight") for e in range(E)]),
            "w2_sf": torch.stack([g(e, "down_proj", "weight_scale") for e in range(E)]).view(fp8),
            "s2_down": torch.tensor([g(e, "down_proj", "weight_scale_2").item() for e in range(E)]),
            "a2": torch.tensor([g(e, "down_proj", "input_scale").item() for e in range(E)]),
        }
        return {k: v.to(d) for k, v in out.items()}

    def load_bf16(self, layer):
        prefix = f"model.language_model.layers.{layer}.mlp."
        names = [prefix + "experts.gate_up_proj", prefix + "experts.down_proj", prefix + "gate.weight"]
        t = self.tensors(self.args.bf16_dir, self.bf16_index, names)
        gate_up, down = t[names[0]], t[names[1]]
        if tuple(gate_up.shape) != (E, 2 * INTER, H) or tuple(down.shape) != (E, H, INTER):
            raise RuntimeError(f"unexpected BF16 expert shapes {tuple(gate_up.shape)} {tuple(down.shape)}")
        return gate_up.to(self.dev), down.to(self.dev), t[names[2]]

    def load_capture(self, split, layer):
        """(x bf16 [n, H], ids int32 [n, k], weights float32 [n, k]) of one captured split."""
        import numpy as np

        torch = self.torch
        root = os.path.join(self.args.capture, split)
        meta = json.load(open(os.path.join(root, "meta.json")))
        rows = int(meta["rows"][str(layer)])
        x = np.fromfile(os.path.join(root, f"L{layer:02d}.x"), dtype=np.int16).reshape(rows, H)
        ids = np.fromfile(os.path.join(root, f"L{layer:02d}.ids"), dtype=np.int16).reshape(rows, TOPK)
        w = np.fromfile(os.path.join(root, f"L{layer:02d}.w"), dtype=np.float32).reshape(rows, TOPK)
        return (torch.from_numpy(x).view(torch.bfloat16).to(self.dev), torch.from_numpy(ids.astype(np.int32)).to(self.dev),
                torch.from_numpy(w).to(self.dev))

    # ------------------------------------------------------------------ math
    def moe_reference(self, x, ids, wts, expert, capture_h=False):
        """Routed-expert MoE output in float32 from BF16 activations; ``expert(e)`` -> (gate_up [2I, H], down [H, I])
        float32. Optionally also the max |FC1 activation| (for the FC2 input scale)."""
        torch, qm = self.torch, self.qm
        token, weight, bounds = qm.routed_rows(ids, wts, range(E), E)
        out = torch.zeros((x.shape[0], H), dtype=torch.float32, device=self.dev)
        amax_h = torch.zeros((), device=self.dev)
        xf = x.float()
        for e in range(E):
            a, b = bounds[e], bounds[e + 1]
            if a == b:
                continue
            gu, dn = expert(e)
            rows = xf[token[a:b]]
            y = rows @ gu.t()
            h = torch.nn.functional.silu(y[:, :INTER]) * y[:, INTER:]
            if capture_h:
                amax_h = torch.maximum(amax_h, h.abs().max())
            out.index_add_(0, token[a:b], (h @ dn.t()) * weight[a:b].unsqueeze(1))
        return out, amax_h

    def dequant_expert(self, served):
        qm = self.qm
        s13 = served["s2_gate"]

        def expert(e):
            gu = qm.dequantize(served["w13"][e:e + 1], served["w13_sf"][e:e + 1], s13[e:e + 1])[0]
            dn = qm.dequantize(served["w2"][e:e + 1], served["w2_sf"][e:e + 1], served["s2_down"][e:e + 1])[0]
            return gu, dn
        return expert

    # ------------------------------------------------------------------ kernels
    def cutlass_prepare(self, served, a13=None, a2=None):
        torch = self.torch
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

        w13 = torch.cat([served["w13"][:, INTER:], served["w13"][:, :INTER]], dim=1).contiguous()
        w13_sf = torch.cat([served["w13_sf"][:, INTER:], served["w13_sf"][:, :INTER]], dim=1).contiguous()
        a1 = (served["a13"].max() if a13 is None else a13).reshape(1).float().repeat(E)
        a2 = (served["a2"].max() if a2 is None else a2).reshape(1).float().repeat(E)
        s13 = served["s2_gate"].float().contiguous()
        return {"w1": w13, "sf1": swizzle_blockscale(w13_sf), "w2": served["w2"].contiguous(),
                "sf2": swizzle_blockscale(served["w2_sf"].contiguous()), "a1_gscale": (1.0 / a1).contiguous(),
                "a2_gscale": (1.0 / a2).contiguous(), "g1": (s13 * a1).contiguous(),
                "g2": (served["s2_down"].float() * a2).contiguous()}

    def cutlass_run(self, p, x, ids, wts, out):
        torch = self.torch
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

    def marlin_prepare(self, served):
        import types

        torch = self.torch
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            prepare_nvfp4_moe_layer_for_marlin,
        )

        layer = types.SimpleNamespace(num_experts=E, hidden_size=H, intermediate_size_per_partition=INTER,
                                      params_dtype=torch.bfloat16)
        w13, s13, g13, w2, s2, g2 = prepare_nvfp4_moe_layer_for_marlin(
            layer=layer, w13=served["w13"].contiguous(), w13_scale=served["w13_sf"].contiguous(),
            w13_scale_2=served["s2_gate"].float().contiguous(), w2=served["w2"].contiguous(),
            w2_scale=served["w2_sf"].contiguous(), w2_scale_2=served["s2_down"].float().contiguous(),
            is_act_and_mul=True)
        return {"w1": w13, "s1": s13, "g1": g13, "w2": w2, "s2": s2, "g2": g2, "workspace": layer.workspace}

    def marlin_run(self, p, x, ids, wts, out):
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
        from vllm.scalar_type import scalar_types

        return fused_marlin_moe(hidden_states=x, w1=p["w1"], w2=p["w2"], bias1=None, bias2=None, w1_scale=p["s1"],
                                w2_scale=p["s2"], topk_weights=wts, topk_ids=ids,
                                quant_type_id=scalar_types.float4_e2m1f.id, global_num_experts=E,
                                global_scale1=p["g1"], global_scale2=p["g2"], workspace=p["workspace"], output=out)

    def chunked(self, run, p, x, ids, wts, chunk=1024):
        torch = self.torch
        outs = []
        for s in range(0, x.shape[0], chunk):
            out = torch.empty((min(chunk, x.shape[0] - s), H), dtype=torch.bfloat16, device=self.dev)
            outs.append(run(p, x[s:s + chunk].contiguous(), ids[s:s + chunk].contiguous(),
                            wts[s:s + chunk].contiguous(), out).clone())
        return torch.cat(outs)

    def graph_time(self, calls, target_ms=20.0):
        torch = self.torch
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
        return statistics.median(samples)

    # ------------------------------------------------------------------ one layer
    def layer(self, layer):
        torch, qm = self.torch, self.qm
        res = {"layer": layer}
        self.results["layers"][str(layer)] = res
        t0 = time.time()
        served = self.load_served(layer)
        gate_up, down, router = self.load_bf16(layer)
        res["load_seconds"] = round(time.time() - t0, 1)
        # Identity: router bytes and ModelOpt RTN reproduction.
        served_router = self.tensors(self.args.model_dir, self.index,
                                     [f"model.language_model.layers.{layer}.mlp.gate.weight"])
        served_router = next(iter(served_router.values()))
        res["router_identical"] = bool(torch.equal(served_router.view(torch.int16), router.view(torch.int16)))
        s13 = qm.scale2_of(gate_up[:, :INTER], gate_up[:, INTER:])
        s2d = qm.scale2_of(down)
        p13, sf13 = qm.rtn(gate_up, s13)
        p2, sf2 = qm.rtn(down, s2d)
        res["rtn_reproduction"] = {
            "scale2_gate_equal": float((s13.cpu() == served["s2_gate"].cpu()).float().mean()),
            "scale2_up_equal": float((s13.cpu() == served["s2_up"].cpu()).float().mean()),
            "scale2_down_equal": float((s2d.cpu() == served["s2_down"].cpu()).float().mean()),
            "w13_bytes_equal": float((p13 == served["w13"]).float().mean()),
            "w13_scales_equal": float((sf13.view(torch.uint8) == served["w13_sf"].view(torch.uint8)).float().mean()),
            "w2_bytes_equal": float((p2 == served["w2"]).float().mean()),
            "w2_scales_equal": float((sf2.view(torch.uint8) == served["w2_sf"].view(torch.uint8)).float().mean()),
        }
        res["input_scales"] = {"a13_min": float(served["a13"].min()), "a13_max": float(served["a13"].max()),
                               "a2_min": float(served["a2"].min()), "a2_max": float(served["a2"].max())}
        deq = self.dequant_expert(served)
        res["weight_rel_error"] = {
            "gate_up": qm.relative_error(qm.dequantize(served["w13"], served["w13_sf"], served["s2_gate"]), gate_up),
            "down": qm.relative_error(qm.dequantize(served["w2"], served["w2_sf"], served["s2_down"]), down)}
        del p13, sf13, p2, sf2
        log({"layer": layer, "identity": res["rtn_reproduction"], "router_identical": res["router_identical"],
             "weight_rel_error": res["weight_rel_error"], "load_seconds": res["load_seconds"]})
        self.flush()
        # Captured rows.
        xc, idc, wc = self.load_capture("calib", layer)
        xe, ide, we = self.load_capture("eval", layer)
        n = min(self.args.eval_rows, xe.shape[0])
        pick = torch.linspace(0, xe.shape[0] - 1, n, device=self.dev).round().long()
        xe, ide, we = xe[pick].contiguous(), ide[pick].contiguous(), we[pick].contiguous()
        res["rows"] = {"calib": int(xc.shape[0]), "eval_used": int(n)}

        def bf16_expert(e):
            return gate_up[e].float(), down[e].float()

        ref_bf16, _ = self.moe_reference(xe, ide, we, bf16_expert)
        ref_deq, _ = self.moe_reference(xe, ide, we, deq)
        errors = {}

        def score(name, y):
            errors[name] = {"vs_bf16": qm.relative_error(y, ref_bf16), "vs_dequant": qm.relative_error(y, ref_deq),
                            "cos_bf16": qm.cosine(y, ref_bf16)}

        score("dequant_bf16act", ref_deq)
        res["errors"] = errors
        # ARC activation scales from the calibration rows (FC2 input: the FC1 activation with the served weights).
        m = min(self.args.eval_rows * 2, xc.shape[0])
        calib_pick = torch.linspace(0, xc.shape[0] - 1, m, device=self.dev).round().long()
        _, amax_h = self.moe_reference(xc[calib_pick], idc[calib_pick], wc[calib_pick], deq, capture_h=True)
        amax_x = xc.float().abs().max()
        arc_a13, arc_a2 = amax_x / (FP4_MAX * FP8_MAX), amax_h / (FP4_MAX * FP8_MAX)
        eval_amax_x = xe.float().abs().max()
        res["activation_scales"] = {
            "checkpoint_a13": float(served["a13"].max()), "arc_a13": float(arc_a13),
            "checkpoint_a2": float(served["a2"].max()), "arc_a2": float(arc_a2),
            "calib_amax_x": float(amax_x), "eval_amax_x": float(eval_amax_x),
            "eval_blocks_saturating_checkpoint_a13": self.saturation(xe, served["a13"].max()),
            "eval_blocks_saturating_arc_a13": self.saturation(xe, arc_a13)}
        self.flush()
        # Kernels.
        try:
            cut = self.cutlass_prepare(served)
            y = self.chunked(self.cutlass_run, cut, xe, ide, we)
            score("served_cutlass_w4a4", y)
            small = {}
            for mm in (1, 6):
                ys = self.chunked(self.cutlass_run, cut, xe[:64], ide[:64], we[:64], chunk=mm)
                small[f"M{mm}_vs_M1024"] = qm.relative_error(ys, y[:64])
            res["cutlass_m_consistency"] = small
            arc = self.cutlass_prepare(served, arc_a13, arc_a2)
            score("cutlass_w4a4_arc_input_scale", self.chunked(self.cutlass_run, arc, xe, ide, we))
        except Exception as exc:  # noqa: BLE001
            cut = None
            self.results["errors"].append(f"L{layer} cutlass: {type(exc).__name__}: {str(exc)[:400]}")
            log({"layer": layer, "cutlass_error": traceback.format_exc()[-1500:]})
        self.flush()
        try:
            mar = self.marlin_prepare(served)
            score("marlin_w4a16", self.chunked(self.marlin_run, mar, xe, ide, we))
        except Exception as exc:  # noqa: BLE001
            mar = None
            self.results["errors"].append(f"L{layer} marlin: {type(exc).__name__}: {str(exc)[:400]}")
            log({"layer": layer, "marlin_error": traceback.format_exc()[-1500:]})
        self.flush()
        log({"layer": layer, "errors": errors, "activation_scales": res["activation_scales"]})
        # Timing with real routings.
        if self.left() > 30:
            res["timing"] = self.timing(cut, mar, xe, ide, we)
            self.flush()
        # Calibration-free RTN with a per-block scale search (weights only, BF16 activations).
        if self.left() > 40 and self.args.rtn_search:
            try:
                t1 = time.time()
                pu, su = qm.rtn_search(gate_up, s13)
                pd, sd = qm.rtn_search(down, s2d)

                def rs_expert(e):
                    return (qm.dequantize(pu[e:e + 1], su[e:e + 1], s13[e:e + 1])[0],
                            qm.dequantize(pd[e:e + 1], sd[e:e + 1], s2d[e:e + 1])[0])

                y, _ = self.moe_reference(xe, ide, we, rs_expert)
                score("rtn_scale_search_bf16act", y)
                res["rtn_search_seconds"] = round(time.time() - t1, 1)
                del pu, su, pd, sd
            except Exception as exc:  # noqa: BLE001
                self.results["errors"].append(f"L{layer} rtn-search: {type(exc).__name__}: {str(exc)[:300]}")
            self.flush()
        log({"layer": layer, "errors": errors, "seconds": round(time.time() - t0, 1)})

    def saturation(self, x, a13):
        """Share of 16-element activation blocks whose FP8 block scale would exceed 448 (clipped)."""
        blocks = x.float().reshape(x.shape[0], -1, 16).abs().amax(-1)
        return float(((blocks / FP4_MAX) / float(a13) > FP8_MAX).float().mean())

    def timing(self, cut, mar, x, ids, wts):
        torch = self.torch
        gen = torch.Generator(device="cpu").manual_seed(0)
        rows = []
        for m in MS:
            if self.left() < 20:
                break
            picks = [torch.randint(0, x.shape[0], (m,), generator=gen).to(self.dev) for _ in range(16)]
            xs = [x[p].contiguous() for p in picks]
            ids_s = [ids[p].contiguous() for p in picks]
            ws = [wts[p].contiguous() for p in picks]
            row = {"m": m, "distinct_experts": statistics.mean(len(set(i.flatten().tolist())) for i in ids_s)}
            for name, run, prep in (("cutlass_us", self.cutlass_run, cut), ("marlin_us", self.marlin_run, mar)):
                if prep is None:
                    continue
                outs = [torch.empty(m, H, dtype=torch.bfloat16, device=self.dev) for _ in range(16)]
                try:
                    calls = [lambda r=r, run=run, prep=prep: run(prep, xs[r], ids_s[r], ws[r], outs[r])
                             for r in range(16)]
                    row[name] = self.graph_time(calls)
                except Exception as exc:  # noqa: BLE001
                    row[name + "_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            if "cutlass_us" in row and "marlin_us" in row:
                row["marlin_over_cutlass"] = row["marlin_us"] / row["cutlass_us"]
            rows.append(row)
            log({"timing": row})
        return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--bf16-dir", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--deadline-seconds", type=float, required=True)
    parser.add_argument("--eval-rows", type=int, default=4096)
    parser.add_argument("--rtn-search", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    bench = Bench(args)
    bench.env()
    for layer in [int(x) for x in args.layers.split(",")]:
        if bench.left() < 60:
            bench.results["notes"].append(f"stopped before layer {layer}: time")
            break
        try:
            bench.layer(layer)
        except Exception as exc:  # noqa: BLE001
            bench.results["errors"].append(f"L{layer}: {type(exc).__name__}: {str(exc)[:400]}")
            log({"layer": layer, "error": traceback.format_exc()[-2000:]})
        finally:
            if bench.dev.type == "cuda":
                bench.torch.cuda.empty_cache()
            bench.flush()
    bench.flush()
    log({"done": True, "elapsed": round(time.time() - bench.start, 1), "errors": bench.results["errors"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
