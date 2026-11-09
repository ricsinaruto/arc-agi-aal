#!/usr/bin/env python3
"""Deeper layer-aware expert cuts: the l45x5 tool-repair 15-min gate bundle with another
excl5 layer-aware mask (same overlay, same harness); router mid-wide shapes set to the mask's kept counts.

Usage: build_ldeep_gate.py --base dist/sabl-v85p-l45x5-trep-c100-15m-v1 --mask data/expert_prune/prune-l60-excl5-v1.json --out dist/...
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for k in ("--base", "--mask", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    base, mask, out = Path(a.base), Path(a.mask), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    m = json.loads(mask.read_text())
    if m.get("format") != "arc-expert-prune-v1" or "excl5" not in mask.name:
        raise SystemExit("an excl5 arc-expert-prune-v1 mask")
    kept = sorted({512 - int(n) for n in m["pruned_counts"].values()})
    shutil.copytree(base, out, symlinks=True)
    study = json.loads((out / "configs/comparison.json").read_text())
    (out / study["server"]["expert_prune"]["mask"]).unlink()  # the base's served mask; recorded history masks stay
    rel = f"configs/expert-prune/{mask.name}"
    shutil.copyfile(mask, out / rel)
    prune = {"mask": rel, "sha256": sha(mask)}
    for block in (study["server"], study["server_patch"]):
        if "expert_prune" not in block:
            raise SystemExit("base has no expert_prune block")
        block["expert_prune"] = dict(prune)
        if "dense_kernels" in block and "mid_wide_shapes" in block["dense_kernels"]:
            block["dense_kernels"]["mid_wide_shapes"] = [[k, 2560] for k in kept]
    study["name"] = out.name
    (out / "configs/comparison.json").write_text(json.dumps(study, indent=2) + "\n")
    ident = json.loads((out / "MANIFEST.json").read_text())
    ident["files"].pop(json.loads((base / "configs/comparison.json").read_text())["server"]["expert_prune"]["mask"])
    ident["files"][rel] = sha(out / rel)
    ident["files"]["configs/comparison.json"] = sha(out / "configs/comparison.json")
    ident["study"] = out.name
    (out / "MANIFEST.json").write_text(json.dumps(ident, indent=2) + "\n")
    print(json.dumps({"out": str(out), "mask": prune, "kept_shapes": kept}))


if __name__ == "__main__":
    main()
