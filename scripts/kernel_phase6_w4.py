"""Weight-only INT4 against the served INT8 on real BF16 weights of this model (kernel work, phase 6; CPU).

The local copy of real dense weights is the MTP layer of the draft-head pilot overlay (a fine-tuned copy of the
checkpoint's MTP layer: QSA q/k/v/o, shared expert, router, HC and fc projections, all BF16). The GPU job repeats
the check on the target's own INT8-family tensors (``arc_infra.phase6_microbench`` part ``w4``).

Run: PYTHONPATH=src <python with torch> scripts/kernel_phase6_w4.py --out docs/validation-20260927/kernels-phase6-w4.json
"""

import argparse
import json
import struct
from pathlib import Path

import torch

from arc_infra.phase6_microbench import w4_errors

OVERLAY = Path("./runs/modal-validation/arena-draft-head-pilot-v2-modal-20260928/arena/trials/"
               "draft-head/trainer/overlay/overlay.safetensors")
# Fused as vLLM serves them: QSA qkv = q (+ gate) | k | v -> the qsa_qkv INT8 family's shape (13312 x 2560).
FUSED = {"qsa_qkv (13312x2560)": ("mtp.layers.0.self_attn.q_proj.weight", "mtp.layers.0.self_attn.k_proj.weight",
                                   "mtp.layers.0.self_attn.v_proj.weight"),
         "attn_out (2560x6144)": ("mtp.layers.0.self_attn.o_proj.weight",),
         "shared gate_up (1280x2560)": ("mtp.layers.0.mlp.shared_expert.gate_proj.weight",
                                        "mtp.layers.0.mlp.shared_expert.up_proj.weight"),
         "shared down (2560x640)": ("mtp.layers.0.mlp.shared_expert.down_proj.weight",),
         "router (512x2560)": ("mtp.layers.0.mlp.gate.weight",),
         "fc_hidden (2560x2560)": ("mtp.fc_hidden.weight",),
         "HC down (320x10240)": ("mtp.hyper_connection_mixer.input_mix_weight_down.weight",)}


def read_safetensors(path, names):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
        base = 8 + size
        out = {}
        for name in names:
            meta = header[name]
            if meta["dtype"] != "BF16":
                raise TypeError(f"{name}: {meta['dtype']}")
            start, end = meta["data_offsets"]
            handle.seek(base + start)
            raw = bytearray(handle.read(end - start))
            out[name] = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(meta["shape"]).clone()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--overlay", type=Path, default=OVERLAY)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    results = {"source": str(args.overlay), "method": "relative Frobenius error of x @ W^T against FP32 on the BF16 "
               "weight, x Gaussian BF16 [16, K], first 8,192 output channels", "families": {}}
    for family, names in FUSED.items():
        tensors = read_safetensors(args.overlay, names)
        weight = torch.cat([tensors[n] for n in names], dim=0)
        results["families"][family] = w4_errors(weight)
        row = results["families"][family]
        print(family, {k: v["output_rel"] for k, v in row.items() if isinstance(v, dict) and "output_rel" in v})
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
