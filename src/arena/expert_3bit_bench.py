"""3-bit routed-expert MoE micro-benchmark on sm_120 (30 September; docs/validation-20260930/expert-3bit.md).

Runs in the pinned serving runtime (its torch, vLLM Marlin and Humming 0.1.12) without a server. Per layer (served
NVFP4 experts of the checkpoint, real held-out routings of cn04/sk48/wa30):

- **marlin_nvfp4**: vLLM's ``fused_marlin_moe`` on the served NVFP4 tensors (the hero's MoE path, W4A16);
- **humming_nvfp4**: Humming's indexed MoE GEMM on the same NVFP4 tensors (separates kernel from bits);
- **humming_int3_g{16,32,64}**: the NVFP4 experts re-rounded to the ``int3`` grid (``arena.expert_3bit_math``;
  signed E4M3 group scale per 16/32/64 inputs and the per-expert float32 scale): 3.5 / 3.25 / 3.125 bits per weight;
- **humming_uint3zp_g128**: Humming's own RTN to uint3 with a 3-bit zero point and a BF16 scale per 128 (3.15 bits),
  the smallest common 3-bit layout, speed reference only.

Each Humming MoE is vLLM's indexed pipeline (``HummingIndexedExperts.apply``): block alignment, gate-up GEMM, SiLU-mul,
down GEMM, weighted top-k sum. Correctness: every kernel's output against the float32 MoE of its own dequantized
weights on 256 held-out rows. Timing: CUDA-graph replay of 16 real routings per M (M = decode tokens).

    python -m arena.expert_3bit_bench --model-dir <ckpt> --capture <dir with L20.x ...> --layers 20,30 --out r.json
"""

import argparse
import json
import os
import statistics
import time
import traceback

E, H, INTER, TOPK = 512, 2560, 640, 10
MS = (40, 24, 56, 8, 1, 32, 48)  # the bars first
SEARCH = (1.0, 0.95, 0.9, 0.85, 0.8)


def log(obj):
    print(json.dumps({"event": "expert_3bit_bench", **obj}, default=str), flush=True)


class Run:
    def __init__(self, args):
        import torch

        from arena import expert_3bit_math as tm
        from arena import expert_quant_math as qm

        self.torch, self.qm, self.tm, self.args = torch, qm, tm, args
        self.dev = torch.device("cuda")
        self.deadline = time.time() + args.deadline_seconds
        self.results = {"env": {}, "layers": {}, "errors": []}
        with open(os.path.join(args.model_dir, "model.safetensors.index.json")) as fh:
            self.index = json.load(fh)["weight_map"]
        self.locks = torch.zeros(1024, dtype=torch.int32, device=self.dev)

    def left(self):
        return self.deadline - time.time()

    def flush(self):
        tmp = self.args.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.results, fh, default=str, indent=1)
        os.replace(tmp, self.args.out)

    # ------------------------------------------------------------------ inputs
    def load_served(self, layer):
        from arena.expert_quant_bench import Bench

        shim = Bench.__new__(Bench)
        shim.torch, shim.dev, shim.index, shim.args = self.torch, self.dev, self.index, self.args
        return Bench.load_served(shim, layer)

    def load_rows(self, layer):
        import numpy as np

        torch = self.torch
        root = self.args.capture
        meta = json.load(open(os.path.join(root, "meta.json")))
        rows = int(meta["rows"][str(layer)])
        x = np.fromfile(os.path.join(root, f"L{layer:02d}.x"), dtype=np.int16).reshape(rows, H)
        ids = np.fromfile(os.path.join(root, f"L{layer:02d}.ids"), dtype=np.int16).reshape(rows, TOPK)
        w = np.fromfile(os.path.join(root, f"L{layer:02d}.w"), dtype=np.float32).reshape(rows, TOPK)
        return (torch.from_numpy(x).view(torch.bfloat16).to(self.dev),
                torch.from_numpy(ids.astype(np.int32)).to(self.dev), torch.from_numpy(w).to(self.dev))

    # ------------------------------------------------------------------ references
    def reference(self, x, ids, wts, gate_up, down):
        """float32 routed-expert MoE of BF16 activations with float32 weights [E, 2I, H] / [E, H, I]."""
        torch, qm = self.torch, self.qm
        token, weight, bounds = qm.routed_rows(ids, wts, range(E), E)
        out = torch.zeros((x.shape[0], H), dtype=torch.float32, device=self.dev)
        xf = x.float()
        for e in range(E):
            a, b = bounds[e], bounds[e + 1]
            if a == b:
                continue
            y = xf[token[a:b]] @ gate_up[e].t()
            h = torch.nn.functional.silu(y[:, :INTER]) * y[:, INTER:]
            out.index_add_(0, token[a:b], (h @ down[e].t()) * weight[a:b].unsqueeze(1))
        return out

    # ------------------------------------------------------------------ marlin (served)
    def marlin(self, served):
        from arena.expert_quant_bench import Bench

        shim = Bench.__new__(Bench)
        shim.torch = self.torch
        p = Bench.marlin_prepare(shim, served)
        return p, lambda x, ids, wts, out: Bench.marlin_run(shim, p, x, ids, wts, out)

    # ------------------------------------------------------------------ humming
    def humming_sublayer(self, schema, tensors, shape_n, shape_k):
        from humming.config import GemmType
        from humming.schema import HummingInputSchema
        from humming.transform import prepare_layer_config, transform_humming_tensors
        from humming.tune import get_heuristics_config

        torch = self.torch
        config = prepare_layer_config(shape_n=shape_n, shape_k=shape_k, weight_schema=schema,
                                      input_schema=HummingInputSchema(), num_experts=E, pad_n_to_multiple=256,
                                      pad_k_to_multiple=128, has_bias=False, torch_dtype=torch.bfloat16)
        tensors = transform_humming_tensors(config, tensors)
        tuning = get_heuristics_config(layer_config=config, use_f16_accum=False, use_batch_invariant=False,
                                       gemm_type=GemmType.INDEXED)
        return {"config": config, "tensors": tensors, "tuning": tuning, "tuning_str": json.dumps(tuning)}

    def humming(self, w13_schema, w13, w2_schema, w2):
        p = {"w13": self.humming_sublayer(w13_schema, w13, 2 * INTER, H),
             "w2": self.humming_sublayer(w2_schema, w2, H, INTER)}
        p["compute"] = json.dumps({"use_batch_invariant": False, "use_f16_accum": False, "gemm_type": "indexed"})
        return p, lambda x, ids, wts, out: self.humming_run(p, x, ids, wts, out)

    def humming_run(self, p, x, ids, wts, out):
        torch = self.torch
        from humming.forward import humming_forward
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation, apply_moe_activation
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum

        m = x.shape[0]
        valid = m * TOPK
        block = 64
        for lo, hi, cfg in p["w13"]["tuning"]:
            if lo < valid <= hi:
                block = cfg["block_shape"][0]
                break
        sorted_ids, expert_ids, ntp = moe_align_block_size(topk_ids=ids, block_size=block, num_experts=E,
                                                           expert_map=None, ignore_invalid_experts=True)
        common = {"sorted_ids": sorted_ids, "expert_ids": expert_ids, "num_tokens_padded": ntp,
                  "compute_config": p["compute"], "valid_shape_m": valid, "locks": self.locks}
        gu = torch.empty((valid, 2 * INTER), dtype=torch.bfloat16, device=self.dev)
        t13 = p["w13"]["tensors"]
        humming_forward(p["w13"]["config"], inputs=x, weight=t13["weight"], weight_scale=t13.get("weight_scale"),
                        zero_point=t13.get("zero_point"), weight_scale_2=t13.get("weight_scale_2"), outputs=gu,
                        top_k=TOPK, tuning_config=p["w13"]["tuning_str"], **common)
        act = torch.empty((valid, INTER), dtype=torch.bfloat16, device=self.dev)
        apply_moe_activation(MoEActivation.SILU, act, gu)
        dn = torch.empty((valid, H), dtype=torch.bfloat16, device=self.dev)
        t2 = p["w2"]["tensors"]
        humming_forward(p["w2"]["config"], inputs=act, weight=t2["weight"], weight_scale=t2.get("weight_scale"),
                        zero_point=t2.get("zero_point"), weight_scale_2=t2.get("weight_scale_2"), outputs=dn,
                        top_k=1, tuning_config=p["w2"]["tuning_str"], **common)
        moe_fused_mul_sum(inputs=dn.view(m, TOPK, H), topk_weights=wts, topk_ids=ids, expert_map=None, outputs=out)
        return out

    def nvfp4_schema(self):
        from humming import dtypes
        from humming.config.enum import WeightScale2Type, WeightScaleType
        from humming.schema import HummingWeightSchema

        return HummingWeightSchema(b_dtype=dtypes.float4e2m1, bs_dtype=dtypes.float8e4m3, weight_scale_group_size=16,
                                   weight_scale_type=WeightScaleType.GROUP,
                                   weight_scale_2_type=WeightScale2Type.TENSOR)

    def int3_schema(self, group):
        from humming import dtypes
        from humming.config.enum import WeightScale2Type, WeightScaleType
        from humming.schema import HummingWeightSchema

        return HummingWeightSchema(b_dtype=dtypes.uint3, bs_dtype=dtypes.float8e4m3, weight_scale_group_size=group,
                                   weight_scale_type=WeightScaleType.GROUP,
                                   weight_scale_2_type=WeightScale2Type.TENSOR)

    def int3_tensors(self, weight, group):
        """Humming uint3 tensors of float weights [E, N, K] on the int3 grid (RTN with the scale search) and their
        exact dequantization (float32)."""
        torch, tm = self.torch, self.tm
        from humming import ops

        s2 = tm.scale2_of(weight, "int3")
        packed, scales, deqs = [], [], []
        for s in range(0, E, 64):
            deq, sc = tm.rtn(weight[s:s + 64], s2[s:s + 64], "int3", group, SEARCH)
            div = tm.divisor(sc, s2[s:s + 64], group)
            codes = (deq / div).round().to(torch.int32) + 4
            if int(codes.min()) < 0 or int(codes.max()) > 7:
                raise ValueError("int3 codes out of range")
            packed.append(ops.pack_weight(codes.contiguous(), 3))
            scales.append(sc.to(torch.float8_e4m3fn))
            deqs.append(deq)
            del codes, div
        return ({"weight": torch.cat(packed), "weight_scale": torch.cat(scales), "weight_scale_2": s2.float()},
                torch.cat(deqs))

    # ------------------------------------------------------------------ timing
    def graph_time(self, calls, target_ms=20.0):
        from arena.expert_quant_bench import Bench

        shim = Bench.__new__(Bench)
        shim.torch = self.torch
        return Bench.graph_time(shim, calls, target_ms)

    def timing(self, runs, x, ids, wts):
        torch = self.torch
        gen = torch.Generator(device="cpu").manual_seed(0)
        rows = []
        for m in MS:
            if self.left() < 60:
                self.results["errors"].append(f"timing stopped at M={m}: deadline")
                break
            picks = [torch.randint(0, x.shape[0], (m,), generator=gen).to(self.dev) for _ in range(16)]
            xs = [x[p].contiguous() for p in picks]
            ids_s = [ids[p].contiguous() for p in picks]
            ws = [wts[p].contiguous() for p in picks]
            row = {"m": m, "distinct_experts": statistics.mean(len(set(i.flatten().tolist())) for i in ids_s)}
            for name, run in runs.items():
                outs = [torch.empty(m, H, dtype=torch.bfloat16, device=self.dev) for _ in range(16)]
                try:
                    calls = [lambda r=r, run=run: run(xs[r], ids_s[r], ws[r], outs[r]) for r in range(16)]
                    row[name] = self.graph_time(calls)
                except Exception as exc:  # noqa: BLE001
                    row[name + "_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            base = row.get("marlin_nvfp4")
            if base:
                for name in runs:
                    if name in row and name != "marlin_nvfp4":
                        row[name + "_over_marlin"] = row[name] / base
            rows.append(row)
            log({"timing": row})
        return rows

    # ------------------------------------------------------------------ one layer
    def layer(self, layer):
        torch, qm = self.torch, self.qm
        res = {"layer": layer}
        self.results["layers"][str(layer)] = res
        t0 = time.time()
        served = self.load_served(layer)
        x, ids, wts = self.load_rows(layer)
        w13 = qm.dequantize(served["w13"], served["w13_sf"], served["s2_gate"])
        w2 = qm.dequantize(served["w2"], served["w2_sf"], served["s2_down"])
        res["load_seconds"] = round(time.time() - t0, 1)
        check = torch.linspace(0, x.shape[0] - 1, 256, device=self.dev).round().long()
        xc, idc, wc = x[check].contiguous(), ids[check].contiguous(), wts[check].contiguous()
        ref_nvfp4 = self.reference(xc, idc, wc, w13, w2)
        runs, refs, bytes_per_expert = {}, {}, {}

        def add(name, build, ref):
            try:
                t = time.time()
                prep, run = build()
                out = torch.empty((xc.shape[0], H), dtype=torch.bfloat16, device=self.dev)
                ys = []
                for s in range(0, xc.shape[0], 32):  # decode-sized calls
                    o = out[s:s + 32]
                    run(xc[s:s + 32].contiguous(), idc[s:s + 32].contiguous(), wc[s:s + 32].contiguous(), o)
                    ys.append(o.float())
                torch.cuda.synchronize()
                y = torch.cat(ys)
                res.setdefault("correctness", {})[name] = {
                    "vs_own_dequant": qm.relative_error(y, ref), "vs_nvfp4_dequant": qm.relative_error(y, ref_nvfp4),
                    "build_seconds": round(time.time() - t, 1)}
                runs[name] = run
                if isinstance(prep, dict) and "w13" in prep and isinstance(prep["w13"], dict):
                    bytes_per_expert[name] = sum(v.numel() * v.element_size() for sub in ("w13", "w2")
                                                 for v in prep[sub]["tensors"].values()) / E
                log({"layer": layer, "built": name, **res["correctness"][name]})
            except Exception as exc:  # noqa: BLE001
                self.results["errors"].append(f"L{layer} {name}: {type(exc).__name__}: {str(exc)[:500]}")
                log({"layer": layer, "failed": name, "traceback": traceback.format_exc()[-2500:]})
            self.flush()

        add("marlin_nvfp4", lambda: self.marlin(served), ref_nvfp4)
        nv13 = {"weight": served["w13"].view(torch.int32), "weight_scale": served["w13_sf"],
                "weight_scale_2": served["s2_gate"].float()}
        nv2 = {"weight": served["w2"].view(torch.int32), "weight_scale": served["w2_sf"],
               "weight_scale_2": served["s2_down"].float()}
        add("humming_nvfp4", lambda: self.humming(self.nvfp4_schema(), nv13, self.nvfp4_schema(), nv2), ref_nvfp4)
        for group in self.args.groups:
            try:
                t13, d13 = self.int3_tensors(w13, group)
                t2, d2 = self.int3_tensors(w2, group)
                ref = self.reference(xc, idc, wc, d13, d2)
                res.setdefault("int3_rtn_vs_nvfp4", {})[str(group)] = qm.relative_error(ref, ref_nvfp4)
                del d13, d2
            except Exception as exc:  # noqa: BLE001
                self.results["errors"].append(f"L{layer} int3 g{group} build: {type(exc).__name__}: {str(exc)[:400]}")
                log({"layer": layer, "failed": f"int3_g{group}", "traceback": traceback.format_exc()[-2500:]})
                continue
            add(f"humming_int3_g{group}",
                lambda t13=t13, t2=t2, g=group: self.humming(self.int3_schema(g), t13, self.int3_schema(g), t2), ref)
        if self.args.zp:
            add("humming_uint3zp_g128", lambda: self.zp_layer(w13, w2), None)
        res["bytes_per_expert"] = bytes_per_expert
        del w13, w2
        torch.cuda.empty_cache()
        res["timing"] = self.timing(runs, x, ids, wts)
        res["seconds"] = round(time.time() - t0, 1)
        self.flush()

    def zp_layer(self, w13, w2):
        """Humming's own schema RTN (uint3, int zero point, BF16 scale per 128); speed reference."""
        from humming import dtypes
        from humming.config.enum import WeightScaleType
        from humming.schema import HummingWeightSchema

        torch = self.torch
        schema = HummingWeightSchema(b_dtype=dtypes.uint3, bs_dtype=dtypes.bfloat16, weight_scale_group_size=128,
                                     weight_scale_type=WeightScaleType.GROUP, has_zero_point=True)
        t13 = schema.quant_tensor(w13.to(torch.bfloat16), schema, torch.bfloat16)
        t2 = schema.quant_tensor(w2.to(torch.bfloat16), schema, torch.bfloat16)
        return self.humming(schema, t13, schema, t2)

    def env(self):
        torch = self.torch
        env = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
               "capability": list(torch.cuda.get_device_capability())}
        for mod in ("vllm", "humming"):
            try:
                m = __import__(mod)
                env[mod] = getattr(m, "__version__", "?")
            except Exception as exc:  # noqa: BLE001
                env[mod] = f"unavailable: {type(exc).__name__}: {exc}"
        self.results["env"] = env
        log({"env": env})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--layers", default="20,30")
    ap.add_argument("--groups", default="16,32")
    ap.add_argument("--zp", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-seconds", type=float, default=1500)
    args = ap.parse_args(argv)
    args.groups = [int(g) for g in args.groups.split(",") if g]
    run = Run(args)
    run.env()
    run.flush()
    for layer in [int(v) for v in args.layers.split(",")]:
        if run.left() < 120:
            run.results["errors"].append(f"skipped layer {layer}: deadline")
            break
        try:
            run.layer(layer)
        except Exception as exc:  # noqa: BLE001
            run.results["errors"].append(f"L{layer}: {type(exc).__name__}: {str(exc)[:500]}")
            log({"layer": layer, "traceback": traceback.format_exc()[-3000:]})
        run.flush()
    return run.results


if __name__ == "__main__":
    main()
