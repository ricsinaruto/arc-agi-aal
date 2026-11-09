"""A trainer checkpoint's served adapter, rebuilt off the trainer (RL held-out read of a checkpoint, 25 Oct).

rl1_trainer's ``save_adapter`` writes ``get_peft_model_state_dict`` in BF16; a checkpoint keeps the same params as FP32
``master.safetensors`` under their ``named_parameters`` names (``.default`` adapter infix), and ``op_load_checkpoint``
copies them back with ``.to(dtype)``. So the served tensors are ``bf16(master)`` under the PEFT names. The reference
adapter (a real ``save_adapter`` output of the same trainer) fixes the config and must match the names, shapes and
dtypes exactly.

    modal run --detach scripts/posttrain/export_adapter.py --master validation/RUN/state/sNNNN \
        --out flash-input/models/arc-rl/NAME --ref validation/rl1-xfer-p2b/adapter-r02

``master`` and ``ref`` are runs-volume paths; ``out`` is an assets-volume path (Nebius mirrors flash-input/ prefixes;
on both platforms it reads as /kaggle/input/... ). Writes adapter_model.safetensors, adapter_config.json,
adapter-tensors.json and export.json (sha256 of the tensors file, source checkpoint, checks).
"""
import hashlib
import json
import struct
from pathlib import Path

import modal

ADAPTER = "default"


def served_name(name, adapter=ADAPTER):
    """``get_peft_model_state_dict``'s key for a named parameter: the adapter infix removed."""
    return name.replace(f".{adapter}", "")


def served_state(master, adapter=ADAPTER):
    """{served name: BF16 tensor} of the LoRA params in ``master`` (FP32), refusing anything that is not LoRA."""
    import torch

    out = {}
    for name, tensor in master.items():
        if "lora_" not in name:
            raise ValueError(f"not a LoRA param: {name}")
        key = served_name(name, adapter)
        if key in out:
            raise ValueError(f"two params map to {key}")
        out[key] = tensor.detach().to(torch.bfloat16).contiguous()
    return out


def header(path):
    """{name: (dtype, shape)} from a safetensors file's header, without reading the tensors."""
    with open(path, "rb") as file:
        size = struct.unpack("<Q", file.read(8))[0]
        meta = json.loads(file.read(size))
    return {k: (v["dtype"], tuple(v["shape"])) for k, v in meta.items() if k != "__metadata__"}


def compare(state, ref):
    """Problems of ``state`` against a reference header: missing/extra names, shape or dtype differences."""
    dtypes = {"torch.bfloat16": "BF16"}
    got = {k: (dtypes.get(str(v.dtype), str(v.dtype)), tuple(v.shape)) for k, v in state.items()}
    problems = []
    missing, extra = sorted(set(ref) - set(got)), sorted(set(got) - set(ref))
    if missing:
        problems.append(f"{len(missing)} reference tensors missing, e.g. {missing[:3]}")
    if extra:
        problems.append(f"{len(extra)} tensors not in the reference, e.g. {extra[:3]}")
    bad = [k for k in set(got) & set(ref) if got[k] != ref[k]]
    if bad:
        problems.append(f"{len(bad)} shape/dtype differences, e.g. {[(k, got[k], ref[k]) for k in sorted(bad)[:3]]}")
    return problems


def export(master_dir, out_dir, ref_dir):
    from safetensors.torch import load_file, save_file

    master_dir, out_dir, ref_dir = Path(master_dir), Path(out_dir), Path(ref_dir)
    state = served_state(load_file(str(master_dir / "master.safetensors")))
    problems = compare(state, header(ref_dir / "adapter_model.safetensors"))
    if problems:
        raise ValueError("; ".join(problems))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out_dir / "adapter_model.safetensors"))
    config = json.loads((ref_dir / "adapter_config.json").read_text())
    config["inference_mode"] = True
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=1))
    (out_dir / "adapter-tensors.json").write_text(json.dumps({"tensors": len(state), "expected": len(state)}))
    digest = hashlib.sha256()
    with open(out_dir / "adapter_model.safetensors", "rb") as file:
        for chunk in iter(lambda: file.read(8 << 20), b""):
            digest.update(chunk)
    row = {"source": str(master_dir), "reference": str(ref_dir), "tensors": len(state),
           "gib": round(sum(t.numel() * 2 for t in state.values()) / 2 ** 30, 3),
           "sha256": digest.hexdigest(), "checks": "names, shapes and dtypes equal the reference adapter"}
    (out_dir / "export.json").write_text(json.dumps(row, indent=1))
    return row


app = modal.App("arc-export-adapter", include_source=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.8.0", "safetensors==0.6.2", "numpy", "packaging").add_local_file(
    __file__, "/root/export_adapter.py", copy=True)
runs = modal.Volume.from_name("arc-prolong-runs")
assets = modal.Volume.from_name("arc-prolong-assets")


@app.function(image=image, timeout=1800, retries=0, cpu=(0.125, 8), volumes={"/runs": runs, "/assets": assets})
def run(master: str, out: str, ref: str):
    import sys

    sys.path.insert(0, "/root")
    import export_adapter

    target = Path("/assets") / out
    if (target / "export.json").exists():
        raise SystemExit(f"{target} already exported")
    row = export_adapter.export(Path("/runs") / master, target, Path("/runs") / ref)
    assets.commit()
    print(json.dumps(row), flush=True)
    return row


@app.local_entrypoint()
def main(master: str, out: str, ref: str):
    if not out.startswith("flash-input/models/"):
        raise SystemExit("out must be an assets path under flash-input/models/ (the Nebius mirror's prefix)")
    print(json.dumps(run.remote(master, out, ref)))
