"""INT2-cold MoE kernel micro-benchmark (throughput arm, 7 October; docs/research-throughput/LOG.md section 21).

Question: what does serving the cold half of the kept experts at 2 bits buy in the Humming MoE, given that one Humming
tensor has one weight dtype? Per layer, on the kept experts of an int3 overlay whose cold experts lie on the int2 grid
(``arena.expert_3bit_overlay`` with ``int2``) and the pruned server's own routings (a fit run's eval capture, ids in
the kept experts' ranks):

- **int3_all**: one uint3 tensor of all kept experts, one indexed launch per projection (the locked layout);
- **split_33**: hot and cold experts in two uint3 tensors, two launches per projection (the cost of the split);
- **split_32**: hot uint3 + cold uint2 (the candidate; same weights as int3_all, so the outputs must agree);
- **int2_all**: every expert clamped to int2, one launch (speed bound only, wrong weights).

``--widths`` (8 October, neuron pruning) replaces the split variants by **int3_w<K>**: every kept expert cut to its
first K hidden neurons (gate/up rows, down columns), one launch. A kernel-shape timing only: the weights are the
overlay's, not a pruned fit, and the overlay needs no int2 experts.

Cold experts are found in the overlay itself (every value in -2..2 with no group holding both ends). Each variant is
vLLM's indexed pipeline as in ``arena.expert_3bit_bench``; a split variant aligns the routing once per part with an
expert map (the other part's slots are invalid and skipped) and both parts write the same output rows. Timing: CUDA
graph replay of 16 routings per M; a routing is M / 2 random anchors and their next rows (two positions per stream).

    python -m arena.expert_split_bench --model-dir <ckpt> --overlay <dir> --capture <eval dir> --out r.json
"""

import argparse
import json
import os
import statistics
import time
import traceback

E, H, INTER, TOPK = 512, 2560, 640, 10


def log(obj):
    print(json.dumps({"event": "expert_split_bench", **obj}, default=str), flush=True)


def grid_codes(values, scales16, bits):
    """(uint codes int32 [..., K], signed per-16 scales) of values on the ``bits``-bit grid (-2^(b-1)..2^(b-1)-1 per
    group after an optional sign flip); raises when a value does not fit."""
    import torch

    half = 1 << (bits - 1)
    if not torch.equal(values, values.round()) or float(values.abs().max()) > half:
        raise ValueError(f"values are not on the int{bits} grid")
    k = values.shape[-1]
    g = values.reshape(*values.shape[:-1], k // 16, 16)
    pos, neg = (g == half).any(-1), (g == -half).any(-1)
    if bool((pos & neg).any()):
        raise ValueError(f"an int{bits} group holds both +{half} and -{half}")
    sign = torch.where(pos, -1.0, 1.0)
    codes = (g * sign.unsqueeze(-1) + half).to(torch.int32).reshape(values.shape)
    return codes, (scales16.float() * sign).to(scales16.dtype)


def fits(values, bits):
    """[n] bool: which experts' values [n, N, K] lie on the ``bits``-bit grid."""
    half = 1 << (bits - 1)
    g = values.reshape(values.shape[0], values.shape[1], -1, 16)
    both = ((g == half).any(-1) & (g == -half).any(-1)).flatten(1).any(1)
    return (values.abs().flatten(1).amax(1) <= half) & ~both


class Run:
    def __init__(self, args):
        import torch

        from arena import expert_quant_math as qm

        self.torch, self.qm, self.args = torch, qm, args
        self.dev = torch.device("cuda")
        self.deadline = time.time() + args.deadline_seconds
        self.results = {"env": {}, "args": vars(args), "layers": {}, "errors": []}
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
    def load(self, layer):
        """The overlay's kept experts of one layer as NVFP4 containers (uint8 codes, FP8 scales, scale_2)."""
        from safetensors import safe_open

        torch = self.torch
        fp8 = torch.float8_e4m3fn
        base = f"model.language_model.layers.{layer}.mlp.experts."
        shards = sorted({self.index[f"{base}{e}.gate_proj.weight"] for e in range(E)})
        handles = {s: safe_open(os.path.join(self.args.overlay, s), framework="pt", device="cpu") for s in shards}
        keys = {s: set(h.keys()) for s, h in handles.items()}

        def get(e, proj, kind):
            name = f"{base}{e}.{proj}.{kind}"
            value = handles[self.index[name]].get_tensor(name)
            return value.view(torch.uint8) if value.dtype == fp8 else value

        kept = [e for e in range(E) if f"{base}{e}.gate_proj.weight" in keys[self.index[f"{base}{e}.gate_proj.weight"]]]
        out = {
            "kept": kept,
            "w13": torch.stack([torch.cat([get(e, "gate_proj", "weight"), get(e, "up_proj", "weight")]) for e in kept]),
            "w13_sf": torch.stack([torch.cat([get(e, "gate_proj", "weight_scale"), get(e, "up_proj", "weight_scale")])
                                   for e in kept]).view(fp8),
            "s2_13": torch.tensor([get(e, "gate_proj", "weight_scale_2").item() for e in kept]),
            "w2": torch.stack([get(e, "down_proj", "weight") for e in kept]),
            "w2_sf": torch.stack([get(e, "down_proj", "weight_scale") for e in kept]).view(fp8),
            "s2_2": torch.tensor([get(e, "down_proj", "weight_scale_2").item() for e in kept]),
        }
        return {k: (v.to(self.dev) if hasattr(v, "to") else v) for k, v in out.items()}

    def load_rows(self, layer, limit):
        import numpy as np

        torch = self.torch
        root = self.args.capture
        rows = int(json.load(open(os.path.join(root, "meta.json")))["rows"][str(layer)])
        n = min(rows, limit)
        x = np.fromfile(os.path.join(root, f"L{layer:02d}.x"), dtype=np.int16, count=n * H).reshape(n, H)
        ids = np.fromfile(os.path.join(root, f"L{layer:02d}.ids"), dtype=np.int16, count=n * TOPK).reshape(n, TOPK)
        w = np.fromfile(os.path.join(root, f"L{layer:02d}.w"), dtype=np.float32, count=n * TOPK).reshape(n, TOPK)
        return (torch.from_numpy(x).view(torch.bfloat16).to(self.dev),
                torch.from_numpy(ids.astype(np.int32)).to(self.dev), torch.from_numpy(w).to(self.dev))

    def values(self, packed):
        from arc_infra.expert_int3_runtime import unpack_e2m1

        return unpack_e2m1(packed)

    # ------------------------------------------------------------------ humming
    def schema(self, bits):
        from humming import dtypes
        from humming.config.enum import WeightScale2Type, WeightScaleType
        from humming.schema import HummingWeightSchema

        return HummingWeightSchema(b_dtype=getattr(dtypes, f"uint{bits}"), bs_dtype=dtypes.float8e4m3,
                                   weight_scale_group_size=16, weight_scale_type=WeightScaleType.GROUP,
                                   weight_scale_2_type=WeightScale2Type.TENSOR)

    def pack(self, packed, sf, s2, idx, bits, clamp=False):
        """Humming uint tensors of experts ``idx`` (their NVFP4 values re-encoded, exact unless ``clamp``)."""
        from humming import ops

        torch = self.torch
        weights, scales = [], []
        for s in range(0, idx.numel(), 32):
            sel = idx[s:s + 32]
            values = self.values(packed[sel])
            if clamp:
                half = 1 << (bits - 1)
                values = values.clamp(-half, half - 1)
            codes, scale = grid_codes(values, sf[sel], bits)
            weights.append(ops.pack_weight(codes.contiguous(), bits))
            scales.append(scale)
            del values, codes
        return {"weight": torch.cat(weights), "weight_scale": torch.cat(scales).contiguous(),
                "weight_scale_2": s2[idx].float().contiguous()}

    def sublayer(self, bits, tensors, shape_n, shape_k, experts):
        from humming.config import GemmType
        from humming.schema import HummingInputSchema
        from humming.transform import prepare_layer_config, transform_humming_tensors
        from humming.tune import get_heuristics_config

        config = prepare_layer_config(shape_n=shape_n, shape_k=shape_k, weight_schema=self.schema(bits),
                                      input_schema=HummingInputSchema(), num_experts=experts, pad_n_to_multiple=256,
                                      pad_k_to_multiple=128, has_bias=False, torch_dtype=self.torch.bfloat16)
        tensors = transform_humming_tensors(config, tensors)
        tuning = get_heuristics_config(layer_config=config, use_f16_accum=False, use_batch_invariant=False,
                                       gemm_type=GemmType.INDEXED)
        return {"config": config, "tensors": tensors, "tuning": tuning, "tuning_str": json.dumps(tuning)}

    def narrow(self, served, width):
        """``served`` with every expert cut to its first ``width`` hidden neurons (a multiple of 16)."""
        torch = self.torch
        rows = torch.cat([torch.arange(width), INTER + torch.arange(width)]).to(self.dev)
        fp8 = torch.float8_e4m3fn
        return {**served, "w13": served["w13"][:, rows].contiguous(),
                "w13_sf": served["w13_sf"].view(torch.uint8)[:, rows].contiguous().view(fp8),
                "w2": served["w2"][:, :, :width // 2].contiguous(),
                "w2_sf": served["w2_sf"][:, :, :width // 16].contiguous()}

    def part(self, served, idx, bits, kept, clamp=False):
        """One Humming tensor pair over experts ``idx`` (ranks in the kept list) and its expert map (None = all)."""
        torch = self.torch
        inter = served["w13"].shape[1] // 2
        p = {"w13": self.sublayer(bits, self.pack(served["w13"], served["w13_sf"], served["s2_13"], idx, bits, clamp),
                                  2 * inter, H, idx.numel()),
             "w2": self.sublayer(bits, self.pack(served["w2"], served["w2_sf"], served["s2_2"], idx, bits, clamp),
                                 H, inter, idx.numel()),
             "inter": inter}
        p["map"] = None
        if idx.numel() != kept:
            p["map"] = torch.full((kept,), -1, dtype=torch.int32, device=self.dev)
            p["map"][idx] = torch.arange(idx.numel(), dtype=torch.int32, device=self.dev)
        p["bytes"] = sum(v.numel() * v.element_size() for sub in ("w13", "w2") for v in p[sub]["tensors"].values()
                         if hasattr(v, "numel"))
        return p

    def forward(self, parts, kept, x, ids, wts, out):
        torch = self.torch
        from humming.forward import humming_forward
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation, apply_moe_activation
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum

        m = x.shape[0]
        valid = m * TOPK
        inter = parts[0]["inter"]
        compute = json.dumps({"use_batch_invariant": False, "use_f16_accum": False, "gemm_type": "indexed"})
        gu = torch.empty((valid, 2 * inter), dtype=torch.bfloat16, device=self.dev)
        aligned = []
        for p in parts:
            block = 64
            for lo, hi, cfg in p["w13"]["tuning"]:
                if lo < valid <= hi:
                    block = cfg["block_shape"][0]
                    break
            sorted_ids, expert_ids, ntp = moe_align_block_size(topk_ids=ids, block_size=block, num_experts=kept,
                                                               expert_map=p["map"], ignore_invalid_experts=True)
            common = {"sorted_ids": sorted_ids, "expert_ids": expert_ids, "num_tokens_padded": ntp,
                      "compute_config": compute, "valid_shape_m": valid, "locks": self.locks}
            t = p["w13"]["tensors"]
            humming_forward(p["w13"]["config"], inputs=x, weight=t["weight"], weight_scale=t.get("weight_scale"),
                            zero_point=t.get("zero_point"), weight_scale_2=t.get("weight_scale_2"), outputs=gu,
                            top_k=TOPK, tuning_config=p["w13"]["tuning_str"], **common)
            aligned.append(common)
        act = torch.empty((valid, inter), dtype=torch.bfloat16, device=self.dev)
        apply_moe_activation(MoEActivation.SILU, act, gu)
        dn = torch.empty((valid, H), dtype=torch.bfloat16, device=self.dev)
        for p, common in zip(parts, aligned):
            t = p["w2"]["tensors"]
            humming_forward(p["w2"]["config"], inputs=act, weight=t["weight"], weight_scale=t.get("weight_scale"),
                            zero_point=t.get("zero_point"), weight_scale_2=t.get("weight_scale_2"), outputs=dn,
                            top_k=1, tuning_config=p["w2"]["tuning_str"], **common)
        moe_fused_mul_sum(inputs=dn.view(m, TOPK, H), topk_weights=wts, topk_ids=ids, expert_map=None, outputs=out)
        return out

    def reference(self, served, x, ids, wts):
        """float32 MoE of BF16 activations with the overlay's dequantized weights."""
        torch, qm = self.torch, self.qm
        kept = len(served["kept"])
        token, weight, bounds = qm.routed_rows(ids, wts, range(kept), kept)
        out = torch.zeros((x.shape[0], H), dtype=torch.float32, device=self.dev)
        xf = x.float()
        for e in range(kept):
            a, b = int(bounds[e]), int(bounds[e + 1])
            if a == b:
                continue
            w13 = qm.dequantize(served["w13"][e:e + 1], served["w13_sf"][e:e + 1], served["s2_13"][e:e + 1])[0]
            w2 = qm.dequantize(served["w2"][e:e + 1], served["w2_sf"][e:e + 1], served["s2_2"][e:e + 1])[0]
            y = xf[token[a:b]] @ w13.t()
            h = torch.nn.functional.silu(y[:, :INTER]) * y[:, INTER:]
            out.index_add_(0, token[a:b], (h @ w2.t()) * weight[a:b].unsqueeze(1))
        return out

    def graph_time(self, calls):
        from arena.expert_quant_bench import Bench

        shim = Bench.__new__(Bench)
        shim.torch = self.torch
        return Bench.graph_time(shim, calls, 20.0)

    # ------------------------------------------------------------------ one layer
    def layer(self, layer):
        torch, qm = self.torch, self.qm
        res = {"layer": layer}
        self.results["layers"][str(layer)] = res
        t0 = time.time()
        served = self.load(layer)
        kept = len(served["kept"])
        x, ids, wts = self.load_rows(layer, self.args.rows)
        if int(ids.max()) >= kept or int(ids.min()) < 0:
            raise ValueError(f"capture ids {int(ids.min())}..{int(ids.max())} are not ranks of {kept} kept experts")
        cold = torch.zeros(kept, dtype=torch.bool, device=self.dev)
        for s in range(0, kept, 32):
            cold[s:s + 32] = fits(self.values(served["w13"][s:s + 32]), 2) & fits(self.values(served["w2"][s:s + 32]), 2)
        everyone = torch.arange(kept, device=self.dev)
        hot_idx, cold_idx = everyone[~cold], everyone[cold]
        slots = cold[ids.long()].float().mean().item()
        mass = float((wts * cold[ids.long()]).sum() / wts.sum())
        res.update(kept=kept, cold=int(cold.sum()), rows=int(x.shape[0]), cold_slot_share=slots, cold_routing_mass=mass,
                   load_seconds=round(time.time() - t0, 1))
        log({"layer": layer, **{k: res[k] for k in ("kept", "cold", "rows", "cold_slot_share", "cold_routing_mass")}})
        widths = self.args.widths
        if not widths and not 0 < hot_idx.numel() < kept:
            raise ValueError(f"layer {layer}: {int(cold.sum())} of {kept} experts are int2")
        check = torch.linspace(0, x.shape[0] - 1, 256, device=self.dev).round().long()
        xc, idc, wc = x[check].contiguous(), ids[check].contiguous(), wts[check].contiguous()
        ref = self.reference(served, xc, idc, wc)
        runs, outputs = {}, {}

        def add(name, build, exact=True):
            try:
                t = time.time()
                parts = build()
                out = torch.empty((xc.shape[0], H), dtype=torch.bfloat16, device=self.dev)
                for s in range(0, xc.shape[0], 32):  # decode-sized calls
                    self.forward(parts, kept, xc[s:s + 32].contiguous(), idc[s:s + 32].contiguous(),
                                 wc[s:s + 32].contiguous(), out[s:s + 32])
                torch.cuda.synchronize()
                outputs[name] = out.float()
                res.setdefault("variants", {})[name] = {
                    "vs_dequant": qm.relative_error(outputs[name], ref) if exact else None,
                    "vs_int3_all": (qm.relative_error(outputs[name], outputs["int3_all"])
                                    if exact and "int3_all" in outputs else None),
                    "bytes": sum(p["bytes"] for p in parts), "launches_per_projection": len(parts),
                    "build_seconds": round(time.time() - t, 1)}
                runs[name] = parts
                log({"layer": layer, "built": name, **res["variants"][name]})
            except Exception as exc:  # noqa: BLE001
                self.results["errors"].append(f"L{layer} {name}: {type(exc).__name__}: {str(exc)[:500]}")
                log({"layer": layer, "failed": name, "traceback": traceback.format_exc()[-2500:]})
            self.flush()

        add("int3_all", lambda: [self.part(served, everyone, 3, kept)])
        for width in widths:
            add(f"int3_w{width}", lambda width=width: [self.part(self.narrow(served, width), everyone, 3, kept)],
                exact=False)
        if not widths:
            add("split_33", lambda: [self.part(served, hot_idx, 3, kept), self.part(served, cold_idx, 3, kept)])
            add("split_32", lambda: [self.part(served, hot_idx, 3, kept), self.part(served, cold_idx, 2, kept)])
            add("int2_all", lambda: [self.part(served, everyone, 2, kept, clamp=True)], exact=False)
        gen = torch.Generator(device="cpu").manual_seed(0)
        rows = []
        for m in self.args.ms:
            if self.left() < 60:
                self.results["errors"].append(f"L{layer} timing stopped at M={m}: deadline")
                break
            anchors = [torch.randint(0, x.shape[0] - 1, (m // 2,), generator=gen).to(self.dev) for _ in range(16)]
            picks = [torch.stack([a, a + 1], 1).flatten() for a in anchors]
            xs = [x[p].contiguous() for p in picks]
            ids_s = [ids[p].contiguous() for p in picks]
            ws = [wts[p].contiguous() for p in picks]
            row = {"m": m, "distinct_experts": statistics.mean(len(set(i.flatten().tolist())) for i in ids_s),
                   "distinct_cold": statistics.mean(int(cold[i.flatten().long().unique()].sum()) for i in ids_s)}
            for name, parts in runs.items():
                outs = [torch.empty(m, H, dtype=torch.bfloat16, device=self.dev) for _ in range(16)]
                try:
                    calls = [lambda r=r, parts=parts: self.forward(parts, kept, xs[r], ids_s[r], ws[r], outs[r])
                             for r in range(16)]
                    row[name] = self.graph_time(calls)
                except Exception as exc:  # noqa: BLE001
                    row[name + "_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            for name in runs:
                if name in row and "int3_all" in row:
                    row[name + "_over_int3_all"] = row[name] / row["int3_all"]
            rows.append(row)
            log({"layer": layer, "timing": row})
        res["timing"] = rows
        res["seconds"] = round(time.time() - t0, 1)
        del served, runs, outputs
        torch.cuda.empty_cache()
        self.flush()

    def env(self):
        torch = self.torch
        env = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
        for mod in ("vllm", "humming"):
            try:
                env[mod] = getattr(__import__(mod), "__version__", "?")
            except Exception as exc:  # noqa: BLE001
                env[mod] = f"unavailable: {type(exc).__name__}: {exc}"
        try:
            from humming import dtypes

            env["humming_uint2"] = str(dtypes.uint2)
        except Exception as exc:  # noqa: BLE001
            env["humming_uint2"] = f"unavailable: {type(exc).__name__}: {exc}"
        self.results["env"] = env
        log({"env": env})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--layers", default="4,20,33,44")
    ap.add_argument("--ms", default="88,64,104,120")
    ap.add_argument("--rows", type=int, default=16384)
    ap.add_argument("--widths", default="", help="hidden widths (multiples of 16 below 640) timed instead of the split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-seconds", type=float, default=1200)
    args = ap.parse_args(argv)
    args.layers = [int(v) for v in args.layers.split(",")]
    args.ms = [int(v) for v in args.ms.split(",")]
    args.widths = [int(v) for v in args.widths.split(",") if v]
    if any(w % 16 or not 0 < w < INTER for w in args.widths):
        raise ValueError("widths are multiples of 16 below 640")
    run = Run(args)
    run.env()
    for layer in args.layers:
        if run.left() < 150:
            run.results["errors"].append(f"layer {layer} skipped: deadline")
            break
        try:
            run.layer(layer)
        except Exception as exc:  # noqa: BLE001
            run.results["errors"].append(f"L{layer}: {type(exc).__name__}: {str(exc)[:500]}")
            log({"layer": layer, "failed": "layer", "traceback": traceback.format_exc()[-2500:]})
        run.flush()
    run.flush()
    log({"done": True, "errors": run.results["errors"]})


if __name__ == "__main__":
    main()
