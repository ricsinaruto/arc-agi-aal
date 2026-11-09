"""Calibrated FP8 dense fit on Modal CPU (throughput arm, 20 Oct; owner item 1). One container per key
(D<layer>in / D<layer>out, layers 0-47 + the MTP layer 48), fp8_calib.fit on the excl5 dense capture.

    modal run --detach scripts/dfr_exact/fp8/modal_fp8_calib.py --capture-run RUN --out-id fp8calib-x5-v1 \
        [--keys D03in,D03out] [--summary-path FILE]

Inputs: /runs/validation/<RUN>/expert-capture/{calib,eval}/D*.x(.s) + meta.json (pulled from the bucket), the BF16
checkpoint at BF16_DIR. Outputs: dataset folder /assets/flash-input/datasets/richardcsaky/<out-id>/L<ll>-<side>.safetensors (per
checkpoint tensor: <name> float8_e4m3fn codes, <name>_scale float32 per row; metadata "order" = row order of the
merged projection) + L<ll>-<side>.json stats. Mirror the prefix to the bucket with scripts/nebius_mirror.py.
"""
import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
BF16_DIR = "/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540"
app = modal.App("arc-fp8-calib")
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("numpy==2.2.6", "safetensors==0.6.2", "packaging==24.2")
         .add_local_file(str(HERE / "fp8_calib.py"), "/root/fp8_calib.py", copy=True)
         .add_local_file(__file__, "/root/modal_fp8_calib.py"))


def layer_tensors(index, layer, side):
    """(kind, [checkpoint tensor names]) of a key; layer 48 is the MTP layer."""
    import fp8_calib as fc

    root = "mtp.layers.0." if layer == 48 else f"model.language_model.layers.{layer}."
    for kind, names in (("gdn", fc.TENSORS[side][0]), ("qsa", fc.TENSORS[side][1])):
        full = [root + n + ".weight" for n in names]
        if all(f in index for f in full):
            return kind, full
    raise KeyError(f"layer {layer} {side}: no GDN or QSA tensors")


def load_rows(folder, key, kind_meta):
    import numpy as np
    import torch

    k = int(kind_meta["shapes"][key]["hidden"])
    raw = np.fromfile(folder / f"{key}.x", dtype=np.uint8)
    if (folder / f"{key}.s").exists():  # fp8 rows + per-row scale
        x = torch.from_numpy(raw).view(torch.float8_e4m3fn).reshape(-1, k).float()
        s = torch.from_numpy(np.fromfile(folder / f"{key}.s", dtype=np.float32))
        if s.numel() != x.shape[0]:
            raise RuntimeError(f"{key}: {x.shape[0]} rows but {s.numel()} scales")
        return x * s.unsqueeze(1)
    return torch.from_numpy(raw).view(torch.bfloat16).reshape(-1, k).float()


@app.function(image=image, volumes={"/assets": assets, "/runs": runs.read_only()},
              timeout=3600)
def fit_key(capture_run: str, out_id: str, key: str):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    import fp8_calib as fc

    torch.set_num_threads(8)
    t0 = time.time()
    layer, side = int(key[1:3]), key[3:]
    cap = Path(f"/runs/validation/{capture_run}/expert-capture")
    metas = {split: json.loads((cap / split / "meta.json").read_text())["dense"] for split in ("calib", "eval")}
    index = json.loads(Path(BF16_DIR, "model.safetensors.index.json").read_text())["weight_map"]
    kind, names = layer_tensors(index, layer, side)
    parts = []
    for name in names:
        with safe_open(str(Path(BF16_DIR) / index[name]), framework="pt") as f:
            parts.append(f.get_tensor(name))
    w = torch.cat(parts).float()
    if tuple(w.shape) != fc.SHAPES[(side, kind)]:
        raise RuntimeError(f"{key}: weight {tuple(w.shape)} != {fc.SHAPES[(side, kind)]}")
    xc = load_rows(cap / "calib", key, metas["calib"])
    xe = load_rows(cap / "eval", key, metas["eval"])
    if xc.shape[1] != w.shape[1] or xe.shape[1] != w.shape[1]:
        raise RuntimeError(f"{key}: rows {xc.shape[1]}/{xe.shape[1]} wide, weight K {w.shape[1]}")
    codes, scale, stats = fc.fit(w, xc, xe)
    out = Path(f"/assets/flash-input/datasets/richardcsaky/{out_id}")
    out.mkdir(parents=True, exist_ok=True)
    tensors, start = {}, 0
    for name, p in zip(names, parts):
        rows = p.shape[0]
        tensors[name] = codes[start:start + rows].contiguous()
        tensors[name + "_scale"] = scale[start:start + rows].contiguous()
        start += rows
    save_file(tensors, str(out / f"L{layer:02d}-{side}.safetensors"), metadata={"order": json.dumps(names)})
    stats.update(key=key, kind=kind, tensors=names, shape=list(w.shape), seconds=round(time.time() - t0, 1))
    (out / f"L{layer:02d}-{side}.json").write_text(json.dumps(stats, indent=1))
    assets.commit()
    return stats


@app.local_entrypoint()
def main(capture_run: str, out_id: str, keys: str = "", summary_path: str = ""):
    wanted = keys.split(",") if keys else [f"D{l:02d}{s}" for l in range(49) for s in ("in", "out")]
    rows = list(fit_key.map([capture_run] * len(wanted), [out_id] * len(wanted), wanted, return_exceptions=True))
    ok = [r for r in rows if isinstance(r, dict)]
    bad = [(k, repr(r)[:300]) for k, r in zip(wanted, rows) if not isinstance(r, dict)]
    summary = {"capture_run": capture_run, "out_id": out_id, "keys": len(wanted), "ok": len(ok), "failed": bad}
    for name in ("rtn", "clip", "gptq"):
        for metric in ("weight", "served"):
            vals = [r["eval"][name][metric] for r in ok]
            if vals:
                summary[f"{name}_{metric}_mean"] = sum(vals) / len(vals)
    if ok:
        summary["act_only_mean"] = sum(r["eval_act_only"] for r in ok) / len(ok)
    print(json.dumps(summary, indent=1))
    if summary_path:
        Path(summary_path).write_text(json.dumps({"summary": summary, "keys": ok}, indent=1))
