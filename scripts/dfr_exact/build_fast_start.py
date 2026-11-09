#!/usr/bin/env python3
"""A built x5-family bundle (omits-pruned expert overlay) that starts faster on Nebius (throughput arm, 20 Oct):
the runner skips the base expert shards the overlay replaces (``nebius_asset_skip``; the overlay farm links the
overlay shard whether or not the base file is there), and optionally a bandwidth-sized boot disk
(``nebius_boot_disk_gib``). Served files and config are unchanged; ``--wall`` makes a short start test.

Usage: build_fast_start.py --base dist/sabl-v85p-x5-c100-150k-v1 --out dist/fs-v85p-x5-start-v1 --wall 600 \
           --disk-gib 1116 --manifests DIR
DIR holds the bucket's assets-manifests JSON for the base model and the overlay dataset (rclone copy).
"""
import argparse
import fnmatch
import hashlib
import json
import shutil
from pathlib import Path

BASE = "flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4"
PATTERN = "pytorch/radixark-modelopt-fp4/1/layer-*-experts-*.safetensors"
# The bundle's own expert_overlay.py (builders patch it) gets the same farm change as src/arena/expert_overlay.py.
FARM_OLD = """    for item in sorted(base_dir.iterdir()):
        source = overlay_dir / item.name if item.name in shards else item.resolve()
        (dest / item.name).symlink_to(source)
"""
FARM_NEW = """    # A base shard the overlay replaces may be absent (20 Oct: the Nebius runner skips fetching them); the farm is
    # the same either way: base files, overlay shards over them.
    names = {item.name for item in base_dir.iterdir()} | set(shards)
    for name in sorted(names):
        source = overlay_dir / name if name in shards else (base_dir / name).resolve()
        (dest / name).symlink_to(source)
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "MANIFEST.json" and p.suffix != ".ipynb"
            and p.name not in ("kernel-metadata.json", "build-manifest.json") and "dataset-upload" not in p.parts}


def slug(prefix):
    return prefix.strip("/").replace("/", "--") + ".json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--wall", type=int, help="play seconds (a start test); default: the base's")
    ap.add_argument("--disk-gib", type=int, help="boot disk floor (93 GiB steps): bandwidth")
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    study = json.loads((base / "configs/comparison.json").read_text())
    overlay = (study.get("server") or {}).get("expert_overlay") or {}
    datasets = study.get("model_datasets") or []
    if not overlay or overlay.get("dataset") not in datasets:
        raise SystemExit("base serves no expert overlay")
    base_files = json.loads((Path(a.manifests) / slug(BASE)).read_text())["files"]
    over_files = json.loads((Path(a.manifests) / slug(f"flash-input/datasets/{overlay['dataset']}")).read_text())["files"]
    skipped = [rel for rel in base_files if fnmatch.fnmatchcase(rel, PATTERN)]
    names = {rel.rsplit("/", 1)[-1] for rel in over_files}
    missing = [rel for rel in skipped if rel.rsplit("/", 1)[-1] not in names]
    if not skipped or missing:
        raise SystemExit(f"overlay does not replace every skipped shard: {len(skipped)} skipped, missing {missing[:3]}")
    shutil.copytree(base, out, symlinks=True, ignore=shutil.ignore_patterns("dataset-upload", "*.ipynb", "source.zip"))
    target = out / "src/arena/expert_overlay.py"
    text = target.read_text()
    if text.count(FARM_OLD) != 1:
        raise SystemExit(f"{target}: the farm loop is not the expected one")
    target.write_text(text.replace(FARM_OLD, FARM_NEW))
    study["nebius_asset_skip"] = {BASE: [PATTERN]}
    if a.disk_gib:
        study["nebius_boot_disk_gib"] = a.disk_gib
    if a.wall:
        for keys in (("wall_seconds",), ("competition", "preview_seconds"), ("b200", "stop", "wall_seconds")):
            node = study
            for key in keys[:-1]:
                node = node[key]
            node[keys[-1]] = a.wall
        study["notebook_seconds"] = a.wall + 2700
    study["fast_start"] = {"base": base.name, "base_manifest_sha256": sha(base / "MANIFEST.json"),
                           "skipped_files": len(skipped), "skipped_bytes": sum(base_files[r]["size"] for r in skipped),
                           "disk_gib": a.disk_gib, "wall": a.wall}
    study["name"] = out.name
    (out / "configs/comparison.json").write_text(json.dumps(study, indent=2))
    ident = json.loads((out / "MANIFEST.json").read_text())
    ident["files"] = manifest(out)
    ident["study"] = out.name
    (out / "MANIFEST.json").write_text(json.dumps(ident, indent=2))
    print(json.dumps({"out": str(out), **study["fast_start"]}))


if __name__ == "__main__":
    main()
