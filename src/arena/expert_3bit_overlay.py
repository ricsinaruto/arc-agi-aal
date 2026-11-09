"""Build an int3 routed-expert overlay in the NVFP4 container (30 September; docs/validation-20260930/expert-3bit.md).

Every replacement shard keeps the checkpoint's names, dtypes and shapes (``arena.expert_overlay`` serves it unchanged
on Marlin; the Humming backend with ``server.expert_int3`` repacks it to 3 bits at load, bit-exact). The source is a
served NVFP4 expert set (the checkpoint or the GPTQ-ARC v1 overlay): each expert is dequantized exactly as vLLM serves
it (the gate's ``weight_scale_2`` for the fused gate/up) and re-rounded to the int3 grid with the per-group scale
search (``expert_3bit_math.rtn``); gate/up/down get the new per-expert ``weight_scale_2`` (gate and up share it).

    python -m arena.expert_3bit_overlay --base-dir <ckpt> --source-dir <v1 overlay> --out <dir> --group 16
"""

import argparse
import hashlib
import json
import os
import time

E_SHARD = 128
INTER = 640
SEARCH = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)
FORMAT = "arc-expert-overlay-v1"


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def convert_shard(torch, source, out, group, device):
    from safetensors import safe_open
    from safetensors.torch import save_file

    from arena import expert_3bit_math as tm
    from arena import expert_quant_math as qm

    tensors, metadata = {}, None
    with safe_open(source, framework="pt", device="cpu") as fh:
        metadata = fh.metadata()
        for name in fh.keys():
            tensors[name] = fh.get_tensor(name)
    experts = {}
    for name in tensors:
        if ".mlp.experts." not in name:
            continue
        prefix, rest = name.split(".mlp.experts.")
        e, proj, kind = rest.split(".")
        experts.setdefault((prefix, int(e)), {})[(proj, kind)] = name
    keys = sorted(experts)
    fp8 = torch.float8_e4m3fn

    def get(k, proj, kind):
        return tensors[experts[k][(proj, kind)]]

    def u8(t):
        return t.view(torch.uint8) if t.dtype == fp8 else t

    w13 = torch.stack([torch.cat([u8(get(k, "gate_proj", "weight")), u8(get(k, "up_proj", "weight"))]) for k in keys])
    f13 = torch.stack([torch.cat([u8(get(k, "gate_proj", "weight_scale")), u8(get(k, "up_proj", "weight_scale"))])
                       for k in keys]).view(fp8)
    s13 = torch.tensor([float(get(k, "gate_proj", "weight_scale_2")) for k in keys])
    w2 = torch.stack([u8(get(k, "down_proj", "weight")) for k in keys])
    f2 = torch.stack([u8(get(k, "down_proj", "weight_scale")) for k in keys]).view(fp8)
    s2d = torch.tensor([float(get(k, "down_proj", "weight_scale_2")) for k in keys])
    stats = {}
    new = {}
    for label, packed, scales, s2 in (("w13", w13, f13, s13), ("w2", w2, f2, s2d)):
        deq = qm.dequantize(packed.to(device), scales.to(device), s2.to(device))
        n2 = tm.scale2_of(deq, "int3")
        q, sc = tm.rtn(deq, n2, "int3", group, SEARCH)
        p, b = tm.to_nvfp4(q, sc, n2, group)
        stats[label] = qm.relative_error(q, deq)
        new[label] = (p.cpu(), b.cpu(), n2.float().cpu())
        del deq, q, sc
    for i, k in enumerate(keys):
        p13, b13, n13 = new["w13"]
        p2, b2, n2 = new["w2"]
        repl = {("gate_proj", "weight"): p13[i, :INTER], ("up_proj", "weight"): p13[i, INTER:],
                ("down_proj", "weight"): p2[i], ("gate_proj", "weight_scale"): b13[i, :INTER],
                ("up_proj", "weight_scale"): b13[i, INTER:], ("down_proj", "weight_scale"): b2[i]}
        for (proj, kind), value in repl.items():
            name = experts[k][(proj, kind)]
            original = tensors[name]
            value = value.contiguous().clone()
            if value.dtype != original.dtype:
                value = value.view(original.dtype)
            if value.dtype != original.dtype or value.shape != original.shape:
                raise RuntimeError(f"{name}: {value.dtype} {tuple(value.shape)} != {original.dtype} {tuple(original.shape)}")
            tensors[name] = value
        for proj, scale in (("gate_proj", n13[i]), ("up_proj", n13[i]), ("down_proj", n2[i])):
            name = experts[k][(proj, "weight_scale_2")]
            tensors[name] = scale.reshape(tensors[name].shape).to(tensors[name].dtype).clone()
    save_file(tensors, out + ".tmp", metadata=metadata)
    os.replace(out + ".tmp", out)
    return {"experts": len(keys), "requant_rel_error": stats}


def verify_shard(torch, source, out, max_rel_error=0.35):
    """Check a finished int3 shard against its source shard: the same tensor names, dtypes and shapes; every
    tensor that is not a routed expert's weight / weight_scale / weight_scale_2 byte-identical; every routed-expert
    weight on the int3 grid (E2M1 magnitudes 0..4 only, no 16-group holding both +4 and -4, which the Humming uint3
    repack refuses); finite scales; and the re-rounding error against the source experts below ``max_rel_error``.
    Returns the per-projection relative error; raises on any failure."""
    from safetensors import safe_open

    from arena import expert_quant_math as qm

    fp8 = torch.float8_e4m3fn
    with safe_open(source, framework="pt", device="cpu") as src, safe_open(out, framework="pt", device="cpu") as dst:
        names = set(src.keys())
        if set(dst.keys()) != names:
            raise RuntimeError(f"{out}: tensor names differ from the source")
        if dst.metadata() != src.metadata():
            raise RuntimeError(f"{out}: safetensors metadata differs from the source")
        err = {"w13": [0.0, 0.0], "w2": [0.0, 0.0]}
        for name in sorted(names):
            a, b = src.get_tensor(name), dst.get_tensor(name)
            if a.dtype != b.dtype or a.shape != b.shape:
                raise RuntimeError(f"{name}: {b.dtype} {tuple(b.shape)} != source {a.dtype} {tuple(a.shape)}")
            kind = name.rsplit(".", 1)[1]
            if ".mlp.experts." not in name or kind not in ("weight", "weight_scale", "weight_scale_2"):
                if not torch.equal(a.view(torch.uint8) if a.dtype == fp8 else a,
                                   b.view(torch.uint8) if b.dtype == fp8 else b):
                    raise RuntimeError(f"{name}: not byte-identical to the source")
                continue
            if kind == "weight_scale_2":
                if not bool(torch.isfinite(b.float()).all()) or not bool((b.float() > 0).all()):
                    raise RuntimeError(f"{name}: scale_2 not finite and positive")
                continue
            if kind == "weight_scale":
                if not bool(torch.isfinite(b.float()).all()):
                    raise RuntimeError(f"{name}: non-finite block scales")
                continue
            packed = b.view(torch.uint8)
            codes = torch.stack([packed & 0xF, packed >> 4], -1).reshape(*packed.shape[:-1], -1)
            mag = codes & 7
            if bool(((mag == 1) | (mag == 3) | (mag == 7)).any()):  # E2M1 0.5, 1.5, 6: off the int3 grid
                raise RuntimeError(f"{name}: values off the int3 grid")
            g = codes.reshape(*codes.shape[:-1], -1, 16)
            if bool(((g == 6).any(-1) & (g == 14).any(-1)).any()):
                raise RuntimeError(f"{name}: a 16-group holds both +4 and -4")
            prefix = name.rsplit(".", 1)[0]
            deq_b = qm.dequantize(packed.unsqueeze(0), dst.get_tensor(prefix + ".weight_scale").unsqueeze(0),
                                  dst.get_tensor(prefix + ".weight_scale_2").reshape(1).float())
            deq_a = qm.dequantize(a.view(torch.uint8).unsqueeze(0),
                                  src.get_tensor(prefix + ".weight_scale").unsqueeze(0),
                                  src.get_tensor(prefix + ".weight_scale_2").reshape(1).float())
            slot = err["w2" if ".down_proj." in name else "w13"]
            slot[0] += float((deq_b - deq_a).pow(2).sum())
            slot[1] += float(deq_a.pow(2).sum())
    rel = {k: (v[0] / v[1]) ** 0.5 if v[1] else None for k, v in err.items()}
    if any(r is None or not r < max_rel_error for r in rel.values()):
        raise RuntimeError(f"{out}: re-rounding error {rel} not below {max_rel_error}")
    return rel


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--layers", default=",".join(str(i) for i in range(48)))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    torch.set_num_threads(os.cpu_count())
    index = json.load(open(os.path.join(args.base_dir, "model.safetensors.index.json")))["weight_map"]
    layers = [int(v) for v in args.layers.split(",")]
    shards = sorted({f for k, f in index.items() if ".mlp.experts." in k and not k.startswith("mtp.")
                     and int(k.split(".layers.")[1].split(".")[0]) in layers})
    os.makedirs(args.out, exist_ok=True)
    results = {"shards": {}, "stats": {}}
    t0 = time.time()
    for n, shard in enumerate(shards):
        path = os.path.join(args.out, shard)
        source = os.path.join(args.source_dir, shard)
        if not os.path.exists(path):
            results["stats"][shard] = convert_shard(torch, source, path, args.group, args.device)
        results["shards"][shard] = {"bytes": os.path.getsize(path), "sha256": sha256_file(path),
                                    "base_bytes": os.path.getsize(os.path.join(args.base_dir, shard))}
        if results["shards"][shard]["bytes"] != results["shards"][shard]["base_bytes"]:
            raise RuntimeError(f"{shard}: size differs from the base shard")
        print(json.dumps({"shard": shard, "n": n + 1, "of": len(shards), "seconds": round(time.time() - t0, 1),
                          **results["stats"].get(shard, {})}), flush=True)
    manifest = {"format": FORMAT, "base_model_dir": args.base_dir, "layers": layers, "shards": results["shards"],
                "settings": {"grid": "int3", "group": args.group, "method": "rtn-search", "search": list(SEARCH),
                             "source_dir": args.source_dir},
                "complete": len(layers) == 48, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(args.out, "overlay-manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    with open(os.path.join(args.out, "build-stats.json"), "w") as fh:
        json.dump(results["stats"], fh, indent=1)
    return manifest


if __name__ == "__main__":
    main()
