"""Trained student -> served overlay (format arc-heal-overlay-v1, from scripts/posttrain/heal_export.py): the student's
tensors under the served AutoRound checkpoint's names (uncut layer and expert numbering). Routed experts go as AutoRound
int4 sym g128 (``int4_rtn``); dense/shared tensors, the MTP block (per expert, all 512) and embed_tokens / lm_head go
in BF16. Pruned router rows come from ``router_src``; PLE tables and the vision tower stay the served copy's (frozen).
The play bundle (research_checkpoint.py heal_overlay) maps every overlay key onto the served index.
Modal entry: scripts/sft/modal_export.py."""
import json
import re
from pathlib import Path

LAYER = re.compile(r"^(model\.language_model\.layers\.)(\d+)(\..*)$")
PER_EXPERT = re.compile(r"^(model\.language_model\.layers\.\d+\.mlp\.experts\.)(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
FUSED = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
ROUTER = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")
MTP_FUSED = re.compile(r"^(mtp\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)$")
TRAINED_EMBED = ("model.language_model.embed_tokens.weight", "lm_head.weight")


def tensors(root, names):
    """``names`` from a checkpoint folder (index lookup, one open per shard)."""
    from safetensors import safe_open

    index = json.loads((Path(root) / "model.safetensors.index.json").read_text())["weight_map"]
    out, by = {}, {}
    for n in names:
        by.setdefault(index[n], []).append(n)
    for shard, keys in by.items():
        with safe_open(str(Path(root) / shard), framework="pt") as f:
            for k in keys:
                out[k] = f.get_tensor(k)
    return out


def int4_codes(w, group=128, zero=8, search=0, levels=16):
    """AutoRound sym int4 of ``w`` [..., out, in] (any leading dims; each [out, in] alone): signed codes (-zero ..
    15 - zero) [..., in, out] and fp16 scales [..., in/group, out]. AutoRound's rule: the group's largest-magnitude
    value maps to code 0 (-8), so scale = -w_max / 8, signed. ``search`` > 0: also try that many shrunken scales (1.0
    down to 0.8 of it) per group and keep the lowest squared error. The export's grid and the trainer's fake quant.
    ``levels`` 8 with ``zero`` 4: the int3 grid of the trainer's eval-only screens (no export)."""
    import torch

    wt = w.float().transpose(-1, -2).contiguous()  # [..., in, out]
    *lead, inp, out = wt.shape
    g = wt.reshape(*lead, inp // group, group, out)
    idx = g.abs().argmax(-2, keepdim=True)
    peak = g.gather(-2, idx).squeeze(-2)
    base = torch.where(peak == 0, torch.full_like(peak, 1e-10), -peak / zero)

    def quant(scale):
        q = (g / scale.unsqueeze(-2)).round().clamp(-zero, levels - 1 - zero)
        return ((q * scale.unsqueeze(-2) - g) ** 2).sum(-2)

    scale, err = base, quant(base)
    for f in torch.linspace(1.0, 0.8, search + 1)[1:] if search else []:
        e2 = quant(base * f)
        better = e2 < err
        scale, err = torch.where(better, base * f, scale), torch.where(better, e2, err)
    scale16 = scale.to(torch.float16)
    q = (g / scale16.float().unsqueeze(-2)).round().clamp(-zero, levels - 1 - zero)  # codes against the stored fp16 scale
    return q.reshape(*lead, inp, out), scale16


def int4_dequant(codes, scale16, group=128):
    """fp32 weights [..., out, in] of signed codes [..., in, out] and fp16 scales [..., in/group, out]."""
    *lead, inp, out = codes.shape
    w = codes.float().reshape(*lead, inp // group, group, out) * scale16.float().unsqueeze(-2)
    return w.reshape(*lead, inp, out).transpose(-1, -2)


def int4_rtn(w, group=128, zero=8, search=0):
    """``int4_codes`` of one [out, in] weight in AutoRound's GPTQ layout (qweight [in/8, out], qzeros [in/group,
    out/8] holding ``zero`` - 1 as GPTQ v1 does, fp16 scales [in/group, out])."""
    import torch

    q, scale16 = int4_codes(w, group, zero, search)
    inp, out = q.shape
    codes = (q + zero).to(torch.int32)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    qweight = (codes.reshape(inp // 8, 8, out) << shifts.view(1, -1, 1)).sum(1, dtype=torch.int32)
    z = torch.full((inp // group, out), zero - 1, dtype=torch.int32)
    qzeros = (z.reshape(inp // group, out // 8, 8) << shifts.view(1, 1, -1)).sum(2, dtype=torch.int32)
    return qweight, qzeros, scale16


def export(student_dir, out_dir, *, served_index, prune_file, router_src, search=20):
    """Overlay shards + overlay.json for the student at ``student_dir`` into ``out_dir``. ``served_index``: the served
    AutoRound model.safetensors.index.json (every written name must be in it); ``prune_file``: the arc-expert-prune-v1
    spec the student was cut with (None when its config has no prune_file); ``router_src``: a checkpoint folder holding
    the full source routers. embed_tokens / lm_head go in only when the served index has them (else kept as frozen)."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    root = Path(student_dir)
    cut = json.loads((root / "config.json").read_text())["student_cut"]
    keep = cut["source_layers"]
    if cut.get("prune_file") and not prune_file:
        raise ValueError(f"student was cut with {cut['prune_file']}: pass prune_file")
    spec = json.loads(Path(prune_file).read_text()) if cut.get("prune_file") else None
    experts = ({int(layer): [e for e in range(spec["experts"]) if e not in set(pruned)]
                for layer, pruned in spec["pruned"].items()} if spec else None)
    served = json.loads(Path(served_index).read_text())["weight_map"]
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    weight_map, buffer, size, part = {}, {}, 0, 0
    counts = {"bf16": 0, "experts": 0, "router_rows": 0, "skipped_frozen": 0}
    base_router = {}

    def put(name, tensor):
        nonlocal size
        if name not in served:
            raise KeyError(f"not in the served checkpoint: {name}")
        buffer[name] = tensor.contiguous()
        size += tensor.numel() * tensor.element_size()

    def flush():
        nonlocal buffer, size, part
        if buffer:
            part += 1
            file = f"heal-{part:05d}.safetensors"
            save_file(buffer, str(dst / file), metadata={"format": "pt"})
            weight_map.update({k: file for k in buffer})
            buffer, size = {}, 0

    def expert(key, w):
        qw, qz, sc = int4_rtn(w, search=search)
        put(f"{key}.qweight", qw); put(f"{key}.qzeros", qz); put(f"{key}.scales", sc)
        counts["experts"] += 1

    for shard in sorted(set(index.values())):
        with safe_open(str(root / shard), framework="pt") as f:
            for name in f.keys():
                if index.get(name) != shard:
                    continue  # superseded: a trainer checkpoint maps its trained tensors to its own files
                if ".ple." in name or name.startswith("model.visual.") or (name in TRAINED_EMBED and name not in served):
                    counts["skipped_frozen"] += 1
                    continue  # frozen: the served copy stays (PLE is FP8 there)
                t = f.get_tensor(name)
                m = LAYER.match(name)
                mtp = MTP_FUSED.match(name)
                if mtp:  # the MTP block is served in BF16 with per-expert tensors (all 512 experts)
                    for e in range(t.shape[0]):
                        key = f"{mtp.group(1)}.{e}"
                        if mtp.group(2) == "down_proj":
                            put(f"{key}.down_proj.weight", t[e].to(torch.bfloat16))
                        else:
                            put(f"{key}.gate_proj.weight", t[e][: t.shape[1] // 2].to(torch.bfloat16))
                            put(f"{key}.up_proj.weight", t[e][t.shape[1] // 2:].to(torch.bfloat16))
                    counts["bf16"] += 1
                    continue
                if not m:  # embed_tokens, lm_head, final norm, other mtp.* (per-expert MTP included)
                    put(name, t.to(torch.bfloat16)); counts["bf16"] += 1
                    continue
                old = keep[int(m.group(2))]
                base = f"{m.group(1)}{old}"
                per, fu = PER_EXPERT.match(name), FUSED.match(name)
                if fu:  # [E, 2I, H] (gate rows then up rows) or [E, H, I]; per expert [out, in]
                    for e in range(t.shape[0]):
                        orig = experts[old][e] if experts else e
                        key = f"{base}.mlp.experts.{orig}"
                        if fu.group(2) == "down_proj":
                            expert(f"{key}.down_proj", t[e])
                        else:
                            expert(f"{key}.gate_proj", t[e][: t.shape[1] // 2])
                            expert(f"{key}.up_proj", t[e][t.shape[1] // 2:])
                elif per:  # save_pretrained's per-expert form
                    e, proj = int(per.group(2)), per.group(3)
                    expert(f"{base}.mlp.experts.{experts[old][e] if experts else e}.{proj}", t)
                elif ROUTER.match(name) and experts:
                    base_router[f"{base}.mlp.gate.weight"] = (old, t)
                else:
                    put(f"{base}{m.group(3)}", t.to(torch.bfloat16)); counts["bf16"] += 1
                if size >= 4 << 30:
                    flush()
    if base_router:  # pruned rows: the source router's (served copies are BF16 and equal to the source)
        full = tensors(router_src, list(base_router))
        for key, (old, t) in base_router.items():
            row = full[key].clone()
            row[torch.tensor(experts[old])] = t.to(row.dtype)
            put(key, row.to(torch.bfloat16)); counts["router_rows"] += len(experts[old])
    flush()
    (dst / "overlay.json").write_text(json.dumps({"format": "arc-heal-overlay-v1", "student": str(student_dir),
                                                  "fmt": "autoround", "search": search, "weight_map": weight_map,
                                                  "counts": counts, "drop_layers": cut["drop"]}, indent=1))
    return counts


def export_nvfp4_dense(student_dir, base_dir, out_dir, *, prune_file):
    """The student's trained tensors other than routed experts as whole replacement shards of the served NVFP4
    checkpoint ``base_dir`` (format arc-expert-overlay-v1, served by ``arena.expert_overlay``): every base shard
    holding a trained tensor is rewritten with the trained values (names, shapes, dtypes unchanged; pruned router rows
    keep the base's), plus overlay-manifest.json. Routed experts are left to the INT3 expert overlay. A trained tensor
    outside the base's non-expert shards is an error; untrained tensors (index entries outside the checkpoint's own
    trained-* files) keep the base's values."""
    import hashlib

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    root, base = Path(student_dir), Path(base_dir)
    keep = json.loads((root / "config.json").read_text())["student_cut"]["source_layers"]  # a dropped layer stays base
    spec = json.loads(Path(prune_file).read_text()) if prune_file else None
    experts = ({int(layer): [e for e in range(spec["experts"]) if e not in set(pruned)]
                for layer, pruned in spec["pruned"].items()} if spec else None)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    served = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    trained = {}  # served name -> (student name, kept router rows or None)
    for name, shard in index.items():
        if Path(shard).is_absolute() or not Path(shard).name.startswith("trained-"):
            continue
        if FUSED.match(name) or PER_EXPERT.match(name):
            continue
        m = LAYER.match(name)
        at = keep[int(m.group(2))] if m else None
        rows = experts[at] if m and experts and ROUTER.match(name) else None
        base_name = f"{m.group(1)}{at}{m.group(3)}" if m else name
        if base_name not in served or "-experts-" in served[base_name] or "plefp8" in served[base_name]:
            raise KeyError(f"trained tensor not in a base non-expert shard: {name}")
        trained[base_name] = (name, rows)
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    shards, counts = {}, {"replaced": 0, "router_rows": 0, "kept_base": 0}
    for shard in sorted({served[k] for k in trained}):
        out = {}
        with safe_open(str(base / shard), framework="pt") as f:
            meta = f.metadata()
            for k in f.keys():
                out[k] = f.get_tensor(k)
        new = tensors(root, [trained[k][0] for k in out if k in trained])
        for k in out:
            if k not in trained:
                counts["kept_base"] += 1
                continue
            t, rows = new[trained[k][0]], trained[k][1]
            if rows is not None:
                full = out[k].clone()
                full[torch.tensor(rows)] = t.to(full.dtype)
                t, counts["router_rows"] = full, counts["router_rows"] + len(rows)
            if t.shape != out[k].shape:
                raise ValueError(f"{k}: trained {tuple(t.shape)} vs base {tuple(out[k].shape)}")
            out[k] = t.to(out[k].dtype).contiguous()
            counts["replaced"] += 1
        save_file(out, str(dst / shard), metadata=meta)
        with open(dst / shard, "rb") as fh:
            shards[shard] = {"bytes": (dst / shard).stat().st_size, "sha256": hashlib.file_digest(fh, "sha256").hexdigest()}
    if counts["replaced"] != len(trained):
        raise RuntimeError(f"replaced {counts['replaced']} of {len(trained)} trained tensors")
    (dst / "overlay-manifest.json").write_text(json.dumps(
        {"format": "arc-expert-overlay-v1", "kind": "nvfp4-dense", "student": str(student_dir), "base": str(base_dir),
         "shards": shards, "complete": False, "counts": counts}, indent=1))
    return counts
