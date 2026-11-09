"""Prepare a merged LoRA overlay (the merge run's ``overlay/`` folder, downloaded by
``poll_modal_lora.py``) as a private Kaggle dataset upload folder, after re-checking it.

Usage:
    uv run python scripts/posttrain/prepare_overlay_dataset.py runs/modal-lora/<merge run>/overlay \\
        --dataset richardcsaky/arc-ornith-lora-m0-overlay [--out dist/arc-ornith-lora-m0-overlay]

Checks: the manifest format and base (the served checkpoint inventory), the overlay file's bytes and
sha256, the tensor list and per-tensor sha256, and the zero-adapter check recorded by the merge (every
unchanged tensor re-quantized bit for bit). Writes ``overlay.safetensors``, ``overlay-manifest.json`` and
``dataset-metadata.json`` into ``--out`` (hard links when possible) and prints the upload command and the
arena build flag. Uploading is a separate step through the authenticated wrapper.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from arena.candidate_profiles import ORNITH_35B  # noqa: E402
from arena.weights_overlay import (  # noqa: E402
    MANIFEST_FILE,
    OVERLAY_FILE,
    check_base,
    load_manifest,
    read_header,
    read_tensor,
    sha256_bytes,
    sha256_file,
)


def check_overlay(folder, inventory):
    folder = Path(folder)
    manifest = load_manifest(folder)
    check_base(manifest, inventory)
    path = folder / manifest["overlay"]["file"]
    if (path.stat().st_size, sha256_file(path)) != (manifest["overlay"]["bytes"], manifest["overlay"]["sha256"]):
        raise ValueError(f"{path}: differs from the manifest")
    header = read_header(path)
    if set(header[1]) != set(manifest["tensors"]):
        raise ValueError("overlay file and manifest list different tensors")
    for name, spec in manifest["tensors"].items():
        if sha256_bytes(read_tensor(path, name, header)) != spec["sha256"]:
            raise ValueError(f"{name}: tensor sha256 differs from the manifest")
    zero = ((manifest.get("requant") or {}).get("zero_adapter_check")) or {}
    if not zero or zero.get("exact") != zero.get("tensors"):
        raise ValueError(f"the merge's zero-adapter check did not reproduce every served tensor: {zero}")
    return manifest


def prepare(folder, dataset, out, *, inventory=None):
    inventory = inventory or json.loads(ORNITH_35B.manifest.read_text())
    manifest = check_overlay(folder, inventory)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in (OVERLAY_FILE, MANIFEST_FILE):
        target = out / name
        if target.exists():
            target.unlink()
        try:
            os.link(Path(folder) / name, target)
        except OSError:
            shutil.copyfile(Path(folder) / name, target)
    adapter = (manifest.get("adapter") or {}).get("sha256") or "unknown"
    (out / "dataset-metadata.json").write_text(json.dumps({
        "id": dataset, "title": dataset.split("/")[1][:50], "licenses": [{"name": "other"}],
        "description": (f"Private weights overlay for {manifest['base']['repo']}@{manifest['base']['revision']}: "
                        f"{manifest['overlay']['tensors']} re-quantized tensors of a merged LoRA (adapter sha256 "
                        f"{adapter}); overlay sha256 {manifest['overlay']['sha256']}. Applied by arena.weights_overlay."),
    }, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("overlay")
    parser.add_argument("--dataset", required=True, help="owner/slug of the private Kaggle dataset")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    out = Path(args.out or ROOT / "dist" / args.dataset.split("/")[1])
    manifest = prepare(args.overlay, args.dataset, out)
    print(json.dumps({"out": str(out), "overlay_sha256": manifest["overlay"]["sha256"],
                      "tensors": manifest["overlay"]["tensors"], "identical_to_base": manifest["identical_to_base"],
                      "upload": f"uv run python scripts/kaggle_cli.py datasets create -p {out}",
                      "build_flag": f"--weights-overlay {out}"}, indent=1))


if __name__ == "__main__":
    main()
