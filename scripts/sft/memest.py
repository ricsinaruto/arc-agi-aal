"""Per-rank card memory of a student at N GPUs (shard.py) against a measured one-GPU reference, from the safetensors
headers only (CPU, no weights read):

    python scripts/sft/memest.py <student> --gpus 2 [--keep-layers 1] [--tokens 125248 --window 49152] [--ref ...]

The prediction is the reference's measured peak plus the difference of the modelled terms (target at N minus the
reference at its N), so everything the two runs share (activations of the same window, the n-gram rows, the
allocator's slack) cancels. Modelled, per rank:
- state: every trainable tensor's BF16 weight and gradient (the warm-up holds every gradient), the routed trunk experts
  at this rank's share (the largest rank's ``emax`` of E); on-card optimizer states (2 B per parameter at 8 bits)
  unless paged, then ``depth`` copies of the largest parameter's states; the QAT grid's scales on the experts.
- layer transients (the expert weights a pass holds): one GPU with fake quant holds the grid copy of max(1, kept)
  layers; N GPUs hold the gathered layers (max(1, kept)) plus the grid copy of one shard projection while it gathers.
- backward transients: the full gradient of the largest projection (both); N GPUs add the reduce-scatter's send
  buffer (<= shard.RS["bytes"]), its receive chunk and the shard-size result.
- activations of the warm-up's example (the data's longest: total tokens, gradient window as train.warm_window):
  each kept layer (train.keep: no recompute) and the one layer under recompute hold ``KEEP_GIB_KTOK`` per 1K window
  tokens; every layer's checkpointed input holds ``LAYER_GIB_KTOK`` per 1K total tokens. Measured on B300 warm-ups:
  p75 2 GPUs kept 3 -> 2 at window 49,152 = -20.5 GiB, p50d7 4 GPUs kept 2 -> 1 at 65,536 = -28.2 GiB (0.42, 0.43),
  p50d7 kept 1 longest 116,147 -> 130,461 tokens = +1.3 GiB over 41 layers.
n-gram tables (host, ple_table) and the vision tower's frozen weights are the same in both students and cancel.
"""
import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1, "I32": 4, "I64": 8, "F8_E4M3": 1}
REF = {"student": "/assets/sft-heal/student-p75", "gpus": 1, "keep_layers": 0, "peak_gib": 243.5,  # p75, 3 flags
       "tokens": 130260, "window": 49152}  # its warm-up example
# measured warm-up peaks (B300, int3served QAT, paged states, BF16 dense): a run on one of these students predicts from
# its own measurement (same GPU count first), so what the terms do not model for that student cancels (p50d7 at 4 GPUs
# sits 11.6 GiB above the p75-referenced prediction)
MEASURED = [
    {"student": "/assets/sft-heal/student-p75", "gpus": 2, "keep_layers": 2, "peak_gib": 224.41, "tokens": 125248,
     "window": 49152},  # sft-p75-a2r2t10-c1047-v1, 2 Nov
    {"student": "/assets/sft-heal/student-p50d7-c647", "gpus": 4, "keep_layers": 1, "peak_gib": 222.97, "tokens": 130461,
     "window": 65536},  # sft-p50d7-joint-c942-v1, 2 Nov (a2r2t v3 at 125,248 tokens: 222.49)
    {"student": "/assets/sft-heal/student-p50d8-c647", "gpus": 4, "keep_layers": 1, "peak_gib": 220.99, "tokens": 130807,
     "window": 65536},  # sft-p50d8-lr3e6, 1 Nov
]
KEEP_GIB_KTOK = 0.42  # GiB per 1K window tokens per layer without recompute
LAYER_GIB_KTOK = 0.0023  # GiB per 1K total tokens per layer (checkpointed layer inputs)
GIB = 2 ** 30


def tensors(folder):
    """{name: (numel, bytes per element)} from the headers of a safetensors folder."""
    folder = Path(folder)
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    got = {}
    for f in sorted(set(index.values())):
        path = Path(f) if Path(f).is_absolute() else folder / f
        with open(path, "rb") as h:
            meta = json.loads(h.read(struct.unpack("<Q", h.read(8))[0]))
        meta.pop("__metadata__", None)
        for k, v in meta.items():
            if index.get(k) == f:
                got[k] = (math.prod(v["shape"]), BYTES.get(v["dtype"], 2))
    return got


FUSED = re.compile(r"^(model\.language_model\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)$")  # as shard.py's
PER = re.compile(r"^(model\.language_model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def layers(ts):
    """{trunk layer prefix: {"experts": E, "proj": {projection: numel}}} of the routed trunk experts."""
    fused, per = FUSED, PER
    out = {}
    for k, (n, _) in ts.items():
        m = fused.match(k)
        if m:
            d = out.setdefault(m.group(1), {"experts": 0, "proj": {}})
            d["proj"][m.group(2)] = d["proj"].get(m.group(2), 0) + n
            continue
        m = per.match(k)
        if m:
            d = out.setdefault(m.group(1), {"experts": 0, "proj": {}, "ids": set()})
            d.setdefault("ids", set()).add(int(m.group(2)))
            name = "down_proj" if m.group(3) == "down_proj" else "gate_up_proj"
            d["proj"][name] = d["proj"].get(name, 0) + n
    for d in out.values():
        d["experts"] = len(d.pop("ids", ())) or d["experts"]
    return out


def experts_of(folder):
    config = json.loads((Path(folder) / "config.json").read_text())
    return int(config.get("text_config", config)["num_experts"])


def model(folder, gpus, keep_layers=0, fake_quant=True, grid="int3served", paged=True, depth=2, rs_bytes=1 << 30,
          tokens=REF["tokens"], window=REF["window"]):
    """The modelled terms (bytes) of one rank of ``folder`` at ``gpus`` GPUs."""
    ts = tensors(folder)
    lay = layers(ts)
    experts = experts_of(folder)
    emax = math.ceil(experts / gpus)
    share = emax / experts if gpus > 1 else 1.0
    expert_keys = {k for k in ts if FUSED.match(k) or PER.match(k)}
    host = {k for k in ts if ".ngram_embedding." in k}
    other = [n for k, (n, _) in ts.items() if k not in expert_keys and k not in host]
    e_total = sum(ts[k][0] for k in expert_keys)
    e_rank = e_total * share
    per_param = 4 + (0 if paged else 2)  # BF16 weight + gradient (+ 8-bit states on the card)
    biggest = max(max(other, default=0), max((max(d["proj"].values()) for d in lay.values()), default=0) * share)
    scale_b = (2 / 16 if grid == "int3served" else 2 / 128) if fake_quant else 0
    layer_full = max((sum(d["proj"].values()) for d in lay.values()), default=0) * 2  # BF16 bytes
    proj_full = max((max(d["proj"].values()) for d in lay.values()), default=0) * 2
    held = 1  # expert weights: one layer gathered (grid copy) at a time; kept layers keep activations only
    t = {"experts_state": e_rank * per_param, "other_state": sum(other) * per_param,
         "paged_window": depth * biggest * 2 if paged else 0, "qat_scales": e_rank * scale_b,
         "backward_grad": proj_full}
    if gpus > 1:
        rows = proj_full / 2 / experts  # elements per expert row of the largest projection
        t["gathered_layers"] = held * layer_full
        t["shard_grid_copy"] = proj_full * share if fake_quant else 0
        t["scatter_buffers"] = min(rs_bytes, gpus * emax * rows * 4) * (1 + 1 / gpus) + proj_full * share
    else:
        t["grid_copy_layers"] = held * layer_full if fake_quant else 0
    win = min(window, tokens)
    t["window_activations"] = (keep_layers + 1) * KEEP_GIB_KTOK * win / 1000 * GIB
    t["layer_inputs"] = len(lay) * LAYER_GIB_KTOK * tokens / 1000 * GIB
    return {"experts": experts, "layers": len(lay), "expert_params": e_total, "expert_params_rank": int(e_rank),
            "other_params": sum(other), "layer_gib": round(layer_full / GIB, 3), "terms_gib": {k: round(v / GIB, 2) for k, v in t.items()},
            "total_gib": round(sum(t.values()) / GIB, 2)}


def reference(folder, gpus):
    """The measured warm-up to predict from: this student at this GPU count, else this student, else REF (p75, 1 GPU)."""
    own = [m for m in [REF, *MEASURED] if m["student"].rstrip("/") == str(folder).rstrip("/")]
    return min(own, key=lambda m: m["gpus"] != gpus) if own else REF


def predict(folder, gpus, keep_layers=0, fake_quant=True, grid="int3served", paged=True, depth=2, ref=None,
            tokens=REF["tokens"], window=REF["window"]):
    ref = dict(reference(folder, gpus), **(ref or {}))
    tgt = model(folder, gpus, keep_layers, fake_quant, grid, paged, depth, tokens=tokens, window=window)
    base = model(ref["student"], ref["gpus"], ref["keep_layers"], fake_quant, grid, paged, depth,
                 tokens=ref["tokens"], window=ref["window"])
    delta = tgt["total_gib"] - base["total_gib"]
    return {"target": tgt, "reference": dict(ref, model=base), "delta_gib": round(delta, 2),
            "predicted_peak_gib": round(ref["peak_gib"] + delta, 1)}


def fit(folder, gpus, keep_layers, budget, **kw):
    """As the trainer's warm-up: kept layers step down one at a time until the predicted peak fits the budget."""
    for k in range(keep_layers, -1, -1):
        got = predict(folder, gpus, k, **kw)
        if got["predicted_peak_gib"] <= budget or k == 0:
            return dict(got, keep_layers=k, configured_keep_layers=keep_layers, budget_gib=budget)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("student")
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--keep-layers", type=int, default=0)
    p.add_argument("--tokens", type=int, default=REF["tokens"], help="the data's longest example (total tokens)")
    p.add_argument("--window", type=int, default=REF["window"], help="its gradient window")
    p.add_argument("--grid", default="int3served")
    p.add_argument("--no-fake-quant", action="store_true")
    p.add_argument("--no-paged", action="store_true")
    p.add_argument("--ref", help="reference student (default: its own measured warm-up, else p75 at 1 GPU)")
    p.add_argument("--ref-gpus", type=int)
    p.add_argument("--ref-keep-layers", type=int)
    p.add_argument("--ref-peak", type=float)
    p.add_argument("--ref-tokens", type=int)
    p.add_argument("--ref-window", type=int)
    a = p.parse_args(argv)
    ref = {k: v for k, v in {"student": a.ref, "gpus": a.ref_gpus, "keep_layers": a.ref_keep_layers, "peak_gib": a.ref_peak,
                             "tokens": a.ref_tokens, "window": a.ref_window}.items() if v is not None}
    got = predict(a.student, a.gpus, a.keep_layers, not a.no_fake_quant, a.grid, not a.no_paged, ref=ref or None,
                  tokens=a.tokens, window=a.window)
    print(json.dumps(got, indent=1))


if __name__ == "__main__":
    sys.exit(main())
