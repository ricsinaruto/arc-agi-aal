#!/usr/bin/env python3
"""The l45 submission's run bundle (throughput arm): the l45x5 c100 bundle with the all25 layer-aware
l45 mask and its INT3 overlay (v8.5p's all25 tensors for the kept experts), at the 300K run shape the kernel builder's
--cand expects. Source of truth for build_v82.py --v85p --safety --cand; the tool repair is added by --tool-repair there.

Usage: build_l45a25_cand.py --base dist/sabl-v85p-l45x5-c100-150k-v1 --out dist/sabl-cand-l45a25-c100-300k-v1 \
           --mask data/expert_prune/prune-l45-all25v71b-v1.json --dataset richardcsaky/arc-expert-int3-requant-l45-all25v71b-v1 \
           --manifest-sha256 <overlay-manifest.json sha256>
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "MANIFEST.json" and p.suffix != ".ipynb"
            and p.name not in ("kernel-metadata.json", "build-manifest.json") and "dataset-upload" not in p.parts}


def main():
    ap = argparse.ArgumentParser()
    for k in ("--base", "--out", "--mask", "--dataset", "--manifest-sha256"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    base, out, mask = Path(a.base), Path(a.out), Path(a.mask)
    if out.exists():
        raise SystemExit(f"{out} exists")
    study = json.loads((base / "configs/comparison.json").read_text())
    m = json.loads(mask.read_text())
    if "all25" not in mask.name or m.get("format") != "arc-expert-prune-v1":
        raise SystemExit("an all25 arc-expert-prune-v1 mask")
    kept = sorted({512 - int(n) for n in m["pruned_counts"].values()})
    if [list(s) for s in study["server"]["dense_kernels"]["mid_wide_shapes"]] != [[k, 2560] for k in kept]:
        raise SystemExit(f"router mid-wide shapes do not cover the kept counts {kept}")
    shutil.copytree(base, out, symlinks=True, ignore=shutil.ignore_patterns("dataset-upload", "*.ipynb", "source.zip"))
    for f in (out / "configs/expert-prune").iterdir():
        f.unlink()
    rel = f"configs/expert-prune/{mask.name}"
    shutil.copyfile(mask, out / rel)
    prune = {"mask": rel, "sha256": sha(mask)}
    overlay = {"dataset": a.dataset, "manifest_sha256": a.manifest_sha256, "check": "size"}
    for block in (study["server"], study["server_patch"]):
        block["expert_prune"] = dict(prune)
        block["expert_overlay"] = dict(overlay)
    study["model_datasets"] = [a.dataset]
    study["study_patch"]["model_datasets"] = [a.dataset]
    study["token_stop"] = study["serving_ablation"]["stop_tokens"] = 300000
    study["wall_seconds"] = study["competition"]["preview_seconds"] = study["b200"]["stop"]["wall_seconds"] = 23400
    study["notebook_seconds"] = 25200
    study["name"] = out.name
    (out / "configs/comparison.json").write_text(json.dumps(study, indent=2))
    ident = json.loads((out / "MANIFEST.json").read_text())
    ident["files"] = manifest(out)
    ident["study"] = out.name
    (out / "MANIFEST.json").write_text(json.dumps(ident, indent=2))
    print(json.dumps({"out": str(out), "mask": prune, "overlay": overlay, "kept_shapes": kept}))


if __name__ == "__main__":
    main()
