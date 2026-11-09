"""Healed student -> served overlay (PTH, 18 Oct): the student's BF16 tensors written back under the uncut checkpoint's
names (48-layer numbering, 512 experts), routed experts quantised to ModelOpt NVFP4 like the RadixArk build (W4A4,
group 16: E2M1 codes two per byte, low nibble first; FP8 E4M3 block scales; FP32 global scale amax / (6 x 448)).
The served stack symlinks the NVFP4 checkpoint and points the overlay's keys at these shards; dropped layers and
pruned experts keep their original tensors (masked at load).

    modal run scripts/posttrain/heal_export.py::pull_rollouts --run RUN --name NAME   # play rollouts -> /runs/heal-data/NAME
    modal run scripts/posttrain/heal_export.py::check            # quantiser vs the RadixArk tensors (bit match)
    modal run --detach scripts/posttrain/heal_export.py --student student-p50g24c3 --out heal-overlay/p50g24c3-step0
"""
import json
import re
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REV = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
NV_ROOT = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
LAYER = re.compile(r"^(model\.language_model\.layers\.)(\d+)(\..*)$")
PER_EXPERT = re.compile(r"^(model\.language_model\.layers\.\d+\.mlp\.experts\.)(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
FUSED = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
ROUTER = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")
MTP_FUSED = re.compile(r"^(mtp\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)$")


def nvfp4(w, amax=None):
    """ModelOpt NVFP4 of one [out, in] weight: (packed uint8 [out, in/2], FP8 E4M3 scales [out, in/16], FP32 scalar).
    ``amax``: the global amax to use (gate and up of one expert share it: vLLM fuses them into w13 with one scale)."""
    import torch

    w = w.float()
    out, inp = w.shape
    scale_2 = (w.abs().amax() if amax is None else torch.as_tensor(float(amax))) / (6.0 * 448.0)
    blocks = w.reshape(out, inp // 16, 16)
    scale = (blocks.abs().amax(-1) / 6.0 / scale_2).clamp(max=448.0).to(torch.float8_e4m3fn)
    denom = (scale.float() * scale_2).unsqueeze(-1)
    q = torch.where(denom > 0, blocks / denom, torch.zeros_like(blocks)).reshape(out, inp)
    grid = torch.tensor(E2M1)
    mag = q.abs().clamp(max=6.0)
    code = (mag.unsqueeze(-1) - grid).abs().argmin(-1).to(torch.uint8)  # nearest grid point (ties: lower)
    code = code | (q < 0).to(torch.uint8) << 3  # ModelOpt keeps the sign of values that round to zero
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, scale.contiguous(), scale_2.reshape(())


def dequant(packed, scale, scale_2):
    import torch

    lo, hi = packed & 0xF, packed >> 4
    codes = torch.stack([lo, hi], -1).reshape(packed.shape[0], -1).long()
    vals = torch.tensor(E2M1)[codes & 7] * torch.where(codes & 8 > 0, -1.0, 1.0)
    return (vals.reshape(vals.shape[0], -1, 16) * scale.float().unsqueeze(-1) * scale_2.float()).reshape(vals.shape[0], -1)


TESTS = HERE.parents[1] / "tests/test_heal_kd_loss.py" if modal.is_local() else Path("/opt/heal/tests/test_heal_kd_loss.py")
app = modal.App("sft-heal-export", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.0", "safetensors", "numpy", extra_index_url="https://download.pytorch.org/whl/cpu")
         .add_local_file(str(HERE / "heal_prune-t50-excl5-v1.json"), "/opt/heal/prune-t50.json", copy=True)
         .add_local_file(str(HERE / "heal_prune-t75-excl5-v1.json"), "/opt/heal/prune-t75.json", copy=True)
         .add_local_file(str(HERE / "heal_prune-t87-excl5-v1.json"), "/opt/heal/prune-t87.json", copy=True)
         .pip_install("pytest")
         .add_local_file(str(HERE / "heal_kd.py"), "/opt/heal/scripts/posttrain/heal_kd.py", copy=True)
         .add_local_file(str(TESTS), "/opt/heal/tests/test_heal_kd_loss.py", copy=True)
         .add_local_file(__file__, "/root/heal_export.py", copy=True))


pull_image = (modal.Image.debian_slim(python_version="3.12").apt_install("rclone")
              .add_local_file(__file__, "/root/heal_export.py", copy=True))
BUCKET, ENDPOINT, REGION = "arc-runs-bucket", "https://storage.eu-south1.nebius.cloud", "eu-south1"


@app.function(image=pull_image, timeout=1800, retries=0, volumes={"/runs": runs},
              secrets=[modal.Secret.from_name("nebius-s3")])
def pull_rollouts(run: str, name: str, prefix: str = ""):
    """A Nebius play run's games/*/rollout.jsonl -> /runs/heal-data/NAME (rclone here, not via the workstation);
    ``prefix``: game folders renamed PREFIX<game> into an existing NAME (several runs in one segments folder)."""
    import os
    import shutil
    import subprocess
    import time

    os.environ.update(RCLONE_CONFIG_ARC_TYPE="s3", RCLONE_CONFIG_ARC_PROVIDER="Other", RCLONE_CONFIG_ARC_ENV_AUTH="true",
                      RCLONE_CONFIG_ARC_ENDPOINT=ENDPOINT, RCLONE_CONFIG_ARC_REGION=REGION,
                      RCLONE_CONFIG_ARC_NO_CHECK_BUCKET="true")
    final = Path("/runs/heal-data") / name
    if final.exists() and not prefix:
        raise SystemExit(f"{final} exists")
    dst = Path("/tmp/pull") if prefix else final
    started = time.time()
    src = f"arc:{BUCKET}/runs/validation/{run}/arena/working/trials/duck-transfer/games/"
    subprocess.run(["rclone", "copy", "--include", "*/rollout.jsonl", src, str(dst), "--transfers", "16", "--retries", "5"],
                   check=True)
    files = sorted(dst.glob("*/rollout.jsonl"))
    if prefix:
        final.mkdir(parents=True, exist_ok=True)
        clash = [f.parent.name for f in files if (final / (prefix + f.parent.name)).exists()]
        if clash:
            raise SystemExit(f"{final}: {len(clash)} folders exist, e.g. {prefix}{clash[0]}")
        for f in files:
            shutil.copytree(f.parent, final / (prefix + f.parent.name))
        files = sorted(final.glob(f"{prefix}*/rollout.jsonl"))
    rows = comp = win = chains = 0
    for f in files:
        last = None
        for line in open(f):
            r = json.loads(line); rows += 1
            u = r.get("usage") or {}
            comp += u.get("completion_tokens") or 0
            if last is not None and r["keep"] == 0:
                win += last; chains += 1
            last = (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
        if last:
            win += last; chains += 1
    runs.commit()
    out = {"name": name, "files": len(files), "rows": rows, "completion_tokens": comp, "chains": chains,
           "window_tokens_upper": win, "seconds": round(time.time() - started)}
    print(json.dumps(out))
    return out



@app.function(image=image, timeout=600)
def unit_tests():
    """The heal loss tests (no local torch): ``modal run scripts/posttrain/heal_export.py::unit_tests``."""
    import subprocess

    r = subprocess.run(["python", "-m", "pytest", "-q", "/opt/heal/tests"], capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-1000:])
    return r.returncode


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


@app.function(image=image, timeout=3600, volumes={"/assets": assets})
def check(layer: int = 4, experts: str = "0,1,2,3,200,511"):
    """The quantiser against the RadixArk NVFP4 tensors of the same BF16 source: share of identical codes and scales."""
    import torch

    src = f"/assets/expert-bf16/{REV}"
    index = json.loads((Path(src) / "model.safetensors.index.json").read_text())["weight_map"]
    fused = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj" in index
    rows = []
    for e in (int(x) for x in experts.split(",")):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            nv = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}"
            ref = tensors(NV_ROOT, [f"{nv}.weight", f"{nv}.weight_scale", f"{nv}.weight_scale_2"])
            if fused:
                part = "down_proj" if proj == "down_proj" else "gate_up_proj"
                full = tensors(src, [f"model.language_model.layers.{layer}.mlp.experts.{part}"])[
                    f"model.language_model.layers.{layer}.mlp.experts.{part}"][e]
                if proj != "down_proj":
                    half = full.shape[0] // 2 if full.shape[0] != ref[f"{nv}.weight"].shape[0] else None
                    full = full[:half] if proj == "gate_proj" else full[half:] if half else full
                w = full
            else:
                w = tensors(src, [f"{nv}.weight"])[f"{nv}.weight"]
            if w.shape[0] != ref[f"{nv}.weight"].shape[0]:
                w = w.t()
            p, s, s2 = nvfp4(w, amax=float(ref[f"{nv}.weight_scale_2"]) * 6 * 448 if proj != "down_proj" else None)
            rp, rs, rs2 = ref[f"{nv}.weight"], ref[f"{nv}.weight_scale"], ref[f"{nv}.weight_scale_2"]
            err = (dequant(p, s, s2) - w.float()).pow(2).mean().sqrt() / w.float().pow(2).mean().sqrt()
            rerr = (dequant(rp, rs, rs2) - w.float()).pow(2).mean().sqrt() / w.float().pow(2).mean().sqrt()
            rows.append({"expert": e, "proj": proj, "shape": list(w.shape), "fused_source": fused,
                         "scale_2": [float(s2), float(rs2)],
                         "scale_match": float((s.view(torch.uint8) == rs.view(torch.uint8)).float().mean()),
                         "byte_match": float((p == rp).float().mean()),
                         "rel_rmse_ours": round(float(err), 5), "rel_rmse_radixark": round(float(rerr), 5)})
            print(json.dumps(rows[-1]), flush=True)
    return rows



def gptq_unpack(qweight, qzeros, scales):
    """AutoRound ``auto_round:auto_gptq`` int4 tensors -> (codes [in, out], zeros [groups, out], scales [groups, out])."""
    import torch

    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    codes = ((qweight.unsqueeze(1) >> shifts.view(1, -1, 1)) & 0xF).reshape(-1, qweight.shape[1])
    zeros = ((qzeros.unsqueeze(2) >> shifts.view(1, 1, -1)) & 0xF).reshape(qzeros.shape[0], -1)
    return codes, zeros, scales


def int4_rtn(w, group=128, zero=8, search=0):
    """Symmetric int4 of one [out, in] weight in AutoRound's GPTQ layout (qweight [in/8, out], qzeros [in/group,
    out/8] holding ``zero`` - 1 as GPTQ v1 does, fp16 scales [in/group, out]). AutoRound's rule (check_ar): the
    group's largest-magnitude value maps to code 0 (-8), so scale = -w_max / 8, signed. ``search`` > 0: also try
    that many shrunken scales (1.0 down to 0.8 of it) per group and keep the lowest squared error."""
    import torch

    wt = w.float().t().contiguous()  # [in, out]
    inp, out = wt.shape
    g = wt.reshape(inp // group, group, out)
    idx = g.abs().argmax(1, keepdim=True)
    peak = g.gather(1, idx).squeeze(1)
    base = torch.where(peak == 0, torch.full_like(peak, 1e-10), -peak / zero)

    def quant(scale):
        q = (g / scale.unsqueeze(1)).round().clamp(-zero, 15 - zero)
        return q, ((q * scale.unsqueeze(1) - g) ** 2).sum(1)

    scale = base
    q, err = quant(scale)
    for f in torch.linspace(1.0, 0.8, search + 1)[1:] if search else []:
        q2, e2 = quant(base * f)
        better = e2 < err
        scale, err = torch.where(better, base * f, scale), torch.where(better, e2, err)
        q = torch.where(better.unsqueeze(1), q2, q)
    scale16 = scale.to(torch.float16)
    q = (g / scale16.float().unsqueeze(1)).round().clamp(-zero, 15 - zero)  # codes against the stored fp16 scale
    codes = (q + zero).reshape(inp, out).to(torch.int32)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    qweight = (codes.reshape(inp // 8, 8, out) << shifts.view(1, -1, 1)).sum(1, dtype=torch.int32)
    z = torch.full((inp // group, out), zero - 1, dtype=torch.int32)
    qzeros = (z.reshape(inp // group, out // 8, 8) << shifts.view(1, 1, -1)).sum(2, dtype=torch.int32)
    return qweight, qzeros, scale16


@app.function(image=image, timeout=3600, volumes={"/assets": assets})
def check_ar(layer: int = 4, experts: str = "0,1,200,511"):
    """AutoRound's shipped int4 tensors against the BF16 source: zero-point storage, scale rule, and RTN's error vs
    AutoRound's tuned rounding (relative RMSE of each against the BF16 weight)."""
    import torch
    from safetensors import safe_open

    src = f"/assets/expert-bf16/{REV}"
    ar = safe_open("/assets/sft-heal/ar-check/model-00003-of-00017.safetensors", framework="pt")
    rows = []
    for e in (int(x) for x in experts.split(",")):
        full_gu = tensors(src, [f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"])[
            f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"][e]
        full_d = tensors(src, [f"model.language_model.layers.{layer}.mlp.experts.down_proj"])[
            f"model.language_model.layers.{layer}.mlp.experts.down_proj"][e]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            k = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}"
            qw, qz, sc = ar.get_tensor(f"{k}.qweight"), ar.get_tensor(f"{k}.qzeros"), ar.get_tensor(f"{k}.scales")
            codes, zeros, scales = gptq_unpack(qw, qz, sc)
            out = codes.shape[1]
            if proj == "down_proj":
                w = full_d
            else:
                half = full_gu.shape[0] // 2 if full_gu.shape[0] == 2 * out else None
                w = (full_gu[:half] if proj == "gate_proj" else full_gu[half:]) if half else full_gu
            if w.shape[0] != out:
                w = w.t()
            wt = w.float().t()
            groups = wt.shape[0] // scales.shape[0]
            zero = zeros.float().mode().values.mode().values.item()
            ar_w = ((codes.float() - (zero + 1)) * scales.float().repeat_interleave(groups, 0))
            amax = wt.reshape(-1, groups, wt.shape[1]).abs().amax(1)
            ratio = (amax / scales.float()).median().item()
            norm = wt.pow(2).mean().sqrt()
            errs = {}
            for search in (0, 20):
                qw2, qz2, sc2 = int4_rtn(w, group=groups, search=search)
                c2, z2, s2 = gptq_unpack(qw2, qz2, sc2)
                rtn_w = (c2.float() - (z2.float()[0, 0] + 1)) * s2.float().repeat_interleave(groups, 0)
                errs[search] = (round(((rtn_w - wt).pow(2).mean().sqrt() / norm).item(), 5), round((c2 == codes).float().mean().item(), 4))
            rows.append({"expert": e, "proj": proj, "group": groups, "stored_zero": zero, "zero_values": sorted(set(zeros.flatten().tolist()))[:4],
                         "amax_over_scale_median": round(ratio, 4), "codes_range": [int(codes.min()), int(codes.max())],
                         "rel_rmse_autoround": round(((ar_w - wt).pow(2).mean().sqrt() / norm).item(), 5),
                         "rtn": errs[0], "rtn_search20": errs[20]})
            print(json.dumps(rows[-1]), flush=True)
    return rows

AR_INDEX = "/assets/sft-heal/ar-check/model.safetensors.index.json"  # Intel AutoRound W4A16, what AV2C serves


@app.function(image=image, timeout=7200, volumes={"/assets": assets, "/runs": runs})
def export(student: str, out: str, fmt: str = "autoround", search: int = 20, prune: str = "/opt/heal/prune-t50.json"):
    """Overlay shards for ``/assets/sft-heal/<student>`` under the served checkpoint's names (uncut layer and expert
    numbering). ``fmt`` "autoround": routed experts as AutoRound int4 sym g128 (``int4_rtn``, the shipped scale rule,
    ``search`` shrink steps); "nvfp4": as the RadixArk ModelOpt build. Every other trained tensor goes in BF16; the
    router's pruned rows, PLE tables, embeddings and the vision tower stay the served copy's."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    root = Path(student) if student.startswith("/") else Path("/assets/sft-heal") / student  # /runs/...: a healed round
    config = json.loads((root / "config.json").read_text())
    cut = config["student_cut"]
    keep = cut["source_layers"]
    spec = json.loads(Path(prune).read_text()) if cut.get("prune_file") else None
    experts = ({int(layer): [e for e in range(spec["experts"]) if e not in set(pruned)]
                for layer, pruned in spec["pruned"].items()} if spec else None)
    served = json.loads(Path(AR_INDEX if fmt == "autoround" else Path(NV_ROOT) / "model.safetensors.index.json").read_text())["weight_map"]
    dst = Path("/assets") / out
    dst.mkdir(parents=True, exist_ok=True)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    weight_map, buffer, size, part = {}, {}, 0, 0
    counts = {"bf16": 0, "experts": 0, "router_rows": 0, "skipped_frozen": 0}
    pending, base_router = {}, {}

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

    def expert(key, w, amax=None):
        if fmt == "autoround":
            qw, qz, sc = int4_rtn(w, search=search)
            put(f"{key}.qweight", qw); put(f"{key}.qzeros", qz); put(f"{key}.scales", sc)
        else:
            p, s, s2 = nvfp4(w, amax=amax)
            put(f"{key}.weight", p); put(f"{key}.weight_scale", s); put(f"{key}.weight_scale_2", s2)
        counts["experts"] += 1

    for shard in sorted(set(index.values())):
        with safe_open(str(root / shard), framework="pt") as f:
            for name in f.keys():
                if ".ple." in name or name.startswith(("model.visual.", "model.language_model.embed_tokens", "lm_head")):
                    counts["skipped_frozen"] += 1
                    continue  # frozen in healing: the served copy stays (PLE is FP8 there)
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
                if not m:
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
                            gate, up = t[e][: t.shape[1] // 2], t[e][t.shape[1] // 2:]
                            amax = max(float(gate.float().abs().amax()), float(up.float().abs().amax()))
                            expert(f"{key}.gate_proj", gate, amax); expert(f"{key}.up_proj", up, amax)
                elif per:  # save_pretrained's per-expert form: gate waits for its up (NVFP4 shares the amax)
                    e, proj = int(per.group(2)), per.group(3)
                    key = f"{base}.mlp.experts.{experts[old][e] if experts else e}"
                    pending[(e, proj)] = t
                    if proj == "down_proj":
                        expert(f"{key}.down_proj", pending.pop((e, proj)))
                    elif (e, "gate_proj") in pending and (e, "up_proj") in pending:
                        gate, up = pending.pop((e, "gate_proj")), pending.pop((e, "up_proj"))
                        amax = max(float(gate.float().abs().amax()), float(up.float().abs().amax()))
                        expert(f"{key}.gate_proj", gate, amax); expert(f"{key}.up_proj", up, amax)
                elif ROUTER.match(name) and experts:
                    base_router[f"{base}.mlp.gate.weight"] = (old, t)
                else:
                    put(f"{base}{m.group(3)}", t.to(torch.bfloat16)); counts["bf16"] += 1
                if size >= 4 << 30:
                    flush()
    if pending:
        raise RuntimeError(f"unpaired expert tensors: {sorted(pending)[:5]}")
    if base_router:  # pruned rows: the source router's (served copies are BF16 and equal to the source)
        src = f"/assets/expert-bf16/{REV}"
        full = tensors(src, list(base_router))
        for key, (old, t) in base_router.items():
            row = full[key].clone()
            row[torch.tensor(experts[old])] = t.to(row.dtype)
            put(key, row.to(torch.bfloat16)); counts["router_rows"] += len(experts[old])
    flush()
    (dst / "overlay.json").write_text(json.dumps({"format": "arc-heal-overlay-v1", "student": student, "fmt": fmt,
                                                  "search": search, "weight_map": weight_map, "counts": counts,
                                                  "drop_layers": cut["drop"]}, indent=1))
    assets.commit()
    return counts


@app.local_entrypoint()
def main(student: str = "student-p50g24c3", out: str = "sft-heal/heal-overlay/p50g24c3-step0", fmt: str = "autoround",
         prune: str = "t50"):
    print(json.dumps(export.remote(student, out, fmt, prune=f"/opt/heal/prune-{prune}.json")))
