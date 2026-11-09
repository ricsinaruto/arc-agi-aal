"""Stage-2 ARC-tuned re-quantization of the routed experts (28 September; docs/validation-20260927/expert-quant.md).

Runs in the pinned serving runtime after the capture server stopped (``arena.expert_quant``, phase ``calibrate``),
layer by layer (the next layer's BF16 experts load in a background thread):

1. **Hessians** from the ARC calibration capture (MoE input rows of the calibration games, E4M3 with a per-row scale,
   and the served top-10 routing): per expert ``sum_t w_te^2 x_t x_t^T`` (the routing weight makes the objective the
   expert's share of the MoE output error);
2. **GPTQ** of each expert's fused gate/up projection [1280, 2560] onto the served NVFP4 grid (``weight_scale_2``
   kept from the checkpoint, which is ModelOpt's amax rule; block scales from the error-updated weights), then the
   down projection [2560, 640] with Hessians of the FC1 activation computed with the *new* gate/up weights
   (sequential within the expert);
3. experts with fewer than ``--min-rows`` calibration rows keep the checkpoint's tensors (counted);
4. **held-out errors** on the evaluation capture (other games): the MoE output (routed experts, served routing)
   against the true BF16 experts, for the checkpoint and the GPTQ weights, with BF16 activations (W4A16) and through
   the served FlashInfer CUTLASS W4A4 kernel;
5. the new tensors are written as replacement **shards with the checkpoint's names, dtypes and shapes**
   (``layer-000LL-experts-XXXX-YYYY.safetensors``); ``input_scale`` and ``weight_scale_2`` are copied unchanged, so
   vLLM loads them with no change at all. ``overlay-manifest.json`` lists every shard's bytes and sha256.
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import traceback

from arena import expert_quant_bench as B
from arena.expert_quant_bench import FP4_MAX, FP8_MAX, Bench, log

FORMAT = "arc-expert-overlay-v1"


def sha256_file(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class Gptq(Bench):
    def __init__(self, args):
        super().__init__(args)
        self.results.update(settings={k: getattr(args, k) for k in ("percdamp", "min_rows", "blocksize",
                                                                     "scale_search", "eval_rows", "calib_rows",
                                                                     "prior_rows", "act_order", "prune_mask",
                                                                     "omit_pruned")},
                            shards={}, counts={})
        self.prefetched = {}
        self.kept = None
        self.prune_sha256 = None
        if args.prune_mask:
            # Phase 3: a capture on a load-time-pruned server routes in the kept experts' ranks (arc_expert_prune).
            import hashlib

            raw = open(args.prune_mask, "rb").read()
            mask = json.loads(raw)
            self.prune_sha256 = hashlib.sha256(raw).hexdigest()
            self.kept = {int(layer): sorted(set(range(int(mask["experts"]))) - {int(e) for e in ids})
                         for layer, ids in mask["pruned"].items()}

    # ------------------------------------------------------------------ inputs
    def load_calib(self, layer):
        """E4M3 rows with per-row scales -> bf16 [n, H], ids int32 [n, k], weights float32 [n, k]."""
        H, TOPK = B.H, B.TOPK  # noqa: N806
        import numpy as np

        torch = self.torch
        root = self.args.calib
        meta = json.load(open(os.path.join(root, "meta.json")))
        rows = int(meta["rows"][str(layer)])
        take = rows if self.args.calib_rows <= 0 else min(rows, self.args.calib_rows)
        q = np.fromfile(os.path.join(root, f"L{layer:02d}.x"), dtype=np.uint8, count=take * H).reshape(take, H)
        s = np.fromfile(os.path.join(root, f"L{layer:02d}.s"), dtype=np.float32, count=take)
        ids = np.fromfile(os.path.join(root, f"L{layer:02d}.ids"), dtype=np.int16, count=take * TOPK).reshape(take, TOPK)
        w = np.fromfile(os.path.join(root, f"L{layer:02d}.w"), dtype=np.float32, count=take * TOPK).reshape(take, TOPK)
        x = (torch.from_numpy(q).to(self.dev).view(torch.float8_e4m3fn).float()
             * torch.from_numpy(s).to(self.dev).unsqueeze(1)).to(torch.bfloat16)
        return x, torch.from_numpy(ids.astype(np.int32)).to(self.dev), torch.from_numpy(w).to(self.dev)

    def prefetch_bf16(self, layer):
        def work():
            try:
                from safetensors import safe_open

                prefix = f"model.language_model.layers.{layer}.mlp.experts."
                out = {}
                for proj in ("gate_up_proj", "down_proj"):
                    name = prefix + proj
                    with safe_open(os.path.join(self.args.bf16_dir, self.bf16_index[name]), framework="pt",
                                   device="cpu") as fh:
                        out[proj] = fh.get_tensor(name)
                self.prefetched[layer] = out
            except Exception as exc:  # noqa: BLE001
                self.prefetched[layer] = exc

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        return thread

    def bf16_layer(self, layer, thread):
        E, H, INTER = B.E, B.H, B.INTER  # noqa: N806
        if thread is not None:
            thread.join()
        got = self.prefetched.pop(layer, None)
        if got is None or isinstance(got, Exception):
            gate_up, down, _ = self.load_bf16(layer)
            return gate_up, down
        gate_up, down = got["gate_up_proj"], got["down_proj"]
        if tuple(gate_up.shape) != (E, 2 * INTER, H) or tuple(down.shape) != (E, H, INTER):
            raise RuntimeError(f"unexpected BF16 expert shapes {tuple(gate_up.shape)} {tuple(down.shape)}")
        return gate_up.to(self.dev), down.to(self.dev)

    # ------------------------------------------------------------------ GPTQ
    def gptq_batched(self, weight, scale_2, hessian, counts, batch):
        """GPTQ of [E, N, K] weights in expert batches; returns packed codes and block scales (all experts)."""
        torch, qm = self.torch, self.qm
        packs, scales = [], []
        for s in range(0, weight.shape[0], batch):
            h = hessian[s:s + batch].clone()
            qm.damp_hessian(h, counts[s:s + batch], percdamp=self.args.percdamp)
            if self.args.act_order:
                p, sc = qm.gptq_static(weight[s:s + batch], h, scale_2[s:s + batch], act_order=True,
                                       blocksize=self.args.blocksize)
                del h
            else:
                upper = qm.inverse_cholesky_upper(h)
                del h
                p, sc = qm.gptq(weight[s:s + batch], upper, scale_2[s:s + batch], blocksize=self.args.blocksize,
                                scale_search=tuple(self.args.scale_search))
                del upper
            packs.append(p)
            scales.append(sc)
        return torch.cat(packs), torch.cat(scales)

    def fc1_hessians(self, x, ids, wts):
        E = B.E  # noqa: N806
        return self.qm.hessians(x, ids, wts, range(E), E)

    def fc2_hessians(self, x, ids, wts, gate_up_deq):
        """Hessians of the FC1 activation h = silu(g x) * (u x) computed with ``gate_up_deq(e)`` [2I, H]."""
        E, INTER = B.E, B.INTER  # noqa: N806
        torch, qm = self.torch, self.qm
        token, weight, bounds = qm.routed_rows(ids, wts, range(E), E)
        out = torch.zeros((E, INTER, INTER), dtype=torch.float32, device=self.dev)
        counts = torch.zeros(E, dtype=torch.float32, device=self.dev)
        amax = torch.zeros((), device=self.dev)
        for e in range(E):
            a, b = bounds[e], bounds[e + 1]
            if a == b:
                continue
            y = x[token[a:b]].float() @ gate_up_deq(e).t()
            h = torch.nn.functional.silu(y[:, :INTER]) * y[:, INTER:]
            amax = torch.maximum(amax, h.abs().max())
            rows = h * weight[a:b].unsqueeze(1)
            out[e] = rows.t() @ rows
            counts[e] = float(b - a)
        return out, counts, amax

    # ------------------------------------------------------------------ one layer
    def layer(self, layer, thread):
        qm = self.qm
        E, INTER = B.E, B.INTER  # noqa: N806
        res = {"layer": layer}
        self.results["layers"][str(layer)] = res
        t0 = time.time()
        served = self.load_served(layer)
        gate_up, down = self.bf16_layer(layer, thread)
        x, ids, wts = self.load_calib(layer)
        if self.kept is not None:
            ids = qm.remap_pruned_ids(ids, self.kept[layer])
        res["load_seconds"] = round(time.time() - t0, 1)
        s13, s2d = served["s2_gate"].float(), served["s2_down"].float()
        amax_rule = qm.scale2_of(gate_up[:, :INTER], gate_up[:, INTER:])
        ratio = (s13 / amax_rule).cpu()
        res["scale2_modelopt_equal"] = {"gate_up": float((amax_rule.cpu() == s13.cpu()).float().mean()),
                                        "down": float((qm.scale2_of(down).cpu() == s2d.cpu()).float().mean())}
        res["gate_up_scale2_over_amax_rule"] = {"min": float(ratio.min()), "median": float(ratio.median()),
                                                "max": float(ratio.max())}
        # FC1: Hessians, GPTQ, fallback.
        t1 = time.time()
        h1, c1 = self.fc1_hessians(x, ids, wts)
        keep = c1 >= self.args.min_rows
        if self.args.prior_rows > 0:
            # Phase 3: every expert shrinks towards the layer's pooled input moment, so none needs a fallback.
            qm.pool_hessians(h1, c1, self.args.prior_rows)
            keep = self.torch.ones_like(keep)
        if self.kept is not None:
            pruned = self.torch.ones_like(keep)
            pruned[self.kept[layer]] = False
            keep &= ~pruned  # pruned experts are dropped at load: keep the checkpoint's bytes
        res["calib_rows"] = int(x.shape[0])
        res["experts_gptq"] = int(keep.sum())
        res["experts_below_min_rows"] = int((c1 < self.args.min_rows).sum())
        res["rows_per_expert"] = {"min": float(c1.min()), "median": float(c1.median()), "max": float(c1.max())}
        res["row_counts"] = [int(v) for v in c1.tolist()]
        p13, f13 = self.gptq_batched(gate_up, s13, h1, c1, self.args.batch)
        del h1
        fb = ~keep
        p13[fb], f13[fb] = served["w13"][fb], served["w13_sf"][fb]
        res["fc1_seconds"] = round(time.time() - t1, 1)

        def gu_new(e):
            return qm.dequantize(p13[e:e + 1], f13[e:e + 1], s13[e:e + 1])[0]

        # FC2 with the new FC1 weights.
        t2 = time.time()
        h2, c2, amax_h = self.fc2_hessians(x, ids, wts, gu_new)
        if self.args.prior_rows > 0:
            qm.pool_hessians(h2, c2, self.args.prior_rows)
        p2, f2 = self.gptq_batched(down, s2d, h2, c2, E)
        del h2
        p2[fb], f2[fb] = served["w2"][fb], served["w2_sf"][fb]
        res["fc2_seconds"] = round(time.time() - t2, 1)
        res["activation_amax"] = {"calib_x": float(x.float().abs().max()), "calib_fc1_out": float(amax_h),
                                  "checkpoint_a13_amax": float(served["a13"].max() * FP4_MAX * FP8_MAX),
                                  "checkpoint_a2_amax": float(served["a2"].max() * FP4_MAX * FP8_MAX)}
        del x, ids, wts
        new = {**served, "w13": p13, "w13_sf": f13, "w2": p2, "w2_sf": f2}
        res["changed_fraction"] = {"w13": float((p13 != served["w13"]).float().mean()),
                                   "w2": float((p2 != served["w2"]).float().mean())}
        # Held-out errors.
        try:
            res["errors"] = self.evaluate(layer, served, new, gate_up, down)
        except Exception as exc:  # noqa: BLE001
            self.results["errors"].append(f"L{layer} eval: {type(exc).__name__}: {str(exc)[:300]}")
            log({"layer": layer, "eval_error": traceback.format_exc()[-1500:]})
        # Replacement shards.
        t3 = time.time()
        res["shards"] = self.write_shards(layer, new)
        res["write_seconds"] = round(time.time() - t3, 1)
        res["seconds"] = round(time.time() - t0, 1)
        log({"layer": layer, "errors": res.get("errors"), "experts_gptq": res["experts_gptq"],
             "rows_per_expert": res["rows_per_expert"], "seconds": res["seconds"]})

    def evaluate(self, layer, served, new, gate_up, down):
        torch, qm = self.torch, self.qm
        xe, ide, we = self.load_capture("eval", layer)
        if self.kept is not None:
            ide = qm.remap_pruned_ids(ide, self.kept[layer])
        n = min(self.args.eval_rows, xe.shape[0])
        pick = torch.linspace(0, xe.shape[0] - 1, n, device=self.dev).round().long()
        xe, ide, we = xe[pick].contiguous(), ide[pick].contiguous(), we[pick].contiguous()

        def bf16_expert(e):
            return gate_up[e].float(), down[e].float()

        ref, _ = self.moe_reference(xe, ide, we, bf16_expert)
        errors = {"rows": int(n)}
        for name, weights in (("checkpoint", served), ("gptq", new)):
            deq = self.dequant_expert(weights)
            y, _ = self.moe_reference(xe, ide, we, deq)
            errors[f"{name}_w4a16"] = {"vs_bf16": qm.relative_error(y, ref), "cos_bf16": qm.cosine(y, ref)}
            for kernel, run, prepare in (("w4a4_cutlass", self.cutlass_run, self.cutlass_prepare),
                                         ("w4a16_marlin", self.marlin_run, self.marlin_prepare)):
                try:
                    y = self.chunked(run, prepare(weights), xe, ide, we)
                    errors[f"{name}_{kernel}"] = {"vs_bf16": qm.relative_error(y, ref), "cos_bf16": qm.cosine(y, ref)}
                except Exception as exc:  # noqa: BLE001
                    errors[f"{name}_{kernel}"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        return errors

    def write_shards(self, layer, new):
        """One replacement shard per checkpoint shard of this layer: same tensor names, dtypes and shapes."""
        INTER = B.INTER  # noqa: N806
        from safetensors import safe_open
        from safetensors.torch import save_file

        torch = self.torch
        base = f"model.language_model.layers.{layer}.mlp.experts."
        shards = sorted({f for k, f in self.index.items() if k.startswith(base)})
        kept_set = set(self.kept[layer]) if self.kept is not None else None
        written = {}
        for shard in shards:
            names = [k for k, f in self.index.items() if f == shard]
            tensors = {}
            with safe_open(os.path.join(self.args.model_dir, shard), framework="pt", device="cpu") as fh:
                metadata = fh.metadata()
                for name in names:
                    original = fh.get_tensor(name)
                    if not name.startswith(base):  # anything else in the shard is copied unchanged
                        tensors[name] = original
                        continue
                    expert, proj, kind = name[len(base):].split(".")
                    e = int(expert)
                    if self.args.omit_pruned and self.kept is not None and e not in kept_set:
                        continue
                    if kind == "weight":
                        value = {"gate_proj": new["w13"][e, :INTER], "up_proj": new["w13"][e, INTER:],
                                 "down_proj": new["w2"][e]}[proj]
                    elif kind == "weight_scale":
                        value = {"gate_proj": new["w13_sf"][e, :INTER], "up_proj": new["w13_sf"][e, INTER:],
                                 "down_proj": new["w2_sf"][e]}[proj]
                    else:  # weight_scale_2, input_scale: unchanged
                        value = original
                    value = value.to("cpu").contiguous()
                    if value.dtype != original.dtype and value.view(torch.uint8).dtype == original.dtype:
                        value = value.view(original.dtype)
                    if value.dtype != original.dtype or tuple(value.shape) != tuple(original.shape):
                        raise RuntimeError(f"{name}: {value.dtype} {tuple(value.shape)} != {original.dtype} "
                                           f"{tuple(original.shape)}")
                    tensors[name] = value
            path = os.path.join(self.args.overlay_out, shard)
            save_file(tensors, path + ".tmp", metadata=metadata)
            os.replace(path + ".tmp", path)
            written[shard] = {"bytes": os.path.getsize(path), "sha256": sha256_file(path), "tensors": len(tensors),
                              "base_bytes": os.path.getsize(os.path.join(self.args.model_dir, shard))}
            self.results["shards"][shard] = written[shard]
        return written

    def manifest(self):
        layers = sorted(int(k) for k, v in self.results["layers"].items() if v.get("shards"))
        manifest = {"format": FORMAT, "base_model_dir": self.args.model_dir, "layers": layers,
                    "shards": self.results["shards"], "settings": self.results["settings"],
                    "complete": len(layers) == 48, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if self.kept is not None:
            manifest["expert_prune_sha256"] = self.prune_sha256
            manifest["omits_pruned"] = bool(self.args.omit_pruned)
        path = os.path.join(self.args.overlay_out, "overlay-manifest.json")
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=1, sort_keys=True)
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", required=True)
    parser.add_argument("--capture", required=True, help="the folder holding the eval capture")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--bf16-dir", required=True)
    parser.add_argument("--overlay-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--deadline-seconds", type=float, required=True)
    parser.add_argument("--layers", default=",".join(str(i) for i in range(48)))
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--min-rows", type=int, default=64)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--scale-search", type=float, nargs="+", default=[1.0])
    parser.add_argument("--eval-rows", type=int, default=4096)
    parser.add_argument("--calib-rows", type=int, default=0)
    parser.add_argument("--prior-rows", type=float, default=0.0,
                        help="pooled-Hessian prior strength in rows (0: v1, experts below --min-rows keep the checkpoint)")
    parser.add_argument("--act-order", action="store_true",
                        help="static block scales, columns in decreasing Hessian diagonal (gptq_static)")
    parser.add_argument("--prune-mask", default=None, help="an arc-expert-prune-v1 mask the capture server served")
    parser.add_argument("--omit-pruned", action="store_true",
                        help="with --prune-mask: leave the pruned experts out of the replacement shards")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    args.rtn_search = False
    os.makedirs(args.overlay_out, exist_ok=True)
    job = Gptq(args)
    job.env()
    layers = [int(x) for x in args.layers.split(",")]
    thread = job.prefetch_bf16(layers[0])
    for i, layer in enumerate(layers):
        if job.left() < 45:
            job.results["notes"].append(f"stopped before layer {layer}: time")
            break
        following = job.prefetch_bf16(layers[i + 1]) if i + 1 < len(layers) else None
        try:
            job.layer(layer, thread)
        except Exception as exc:  # noqa: BLE001
            job.results["errors"].append(f"L{layer}: {type(exc).__name__}: {str(exc)[:400]}")
            log({"layer": layer, "error": traceback.format_exc()[-2000:]})
        finally:
            if job.dev.type == "cuda":
                job.torch.cuda.empty_cache()
            job.flush()
        thread = following
    job.results["manifest"] = job.manifest()
    job.flush()
    log({"done": True, "elapsed": round(time.time() - job.start, 1), "layers": job.results["manifest"]["layers"],
         "errors": job.results["errors"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
