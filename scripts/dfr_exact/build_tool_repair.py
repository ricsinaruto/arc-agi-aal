#!/usr/bin/env python3
"""A built v8.5p-source bundle plus the malformed tool-name repair (harness/tool_repair.py; throughput arm, 20 Oct):
src patched, drain_half.drop_unknown_tools on, everything else unchanged.

Usage: build_tool_repair.py --base dist/sabl-v85p-l45x5-c100-15m-v1 --out dist/sabl-v85p-l45x5-trep-c100-15m-v1
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "harness"))
import tool_repair  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "MANIFEST.json" and p.suffix != ".ipynb"
            and p.name not in ("kernel-metadata.json", "build-manifest.json") and "dataset-upload" not in p.parts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    study = json.loads((base / "configs/comparison.json").read_text())
    if not (study.get("drain_half") or {}).get("drop_malformed"):
        raise SystemExit("base has no drain_half with drop_malformed")
    shutil.copytree(base, out, symlinks=True, ignore=shutil.ignore_patterns("dataset-upload", "*.ipynb", "source.zip"))
    tool_repair.apply(out / "src")
    study["drain_half"]["drop_unknown_tools"] = True
    study["tool_repair"] = {"base": base.name, "base_manifest_sha256": sha(base / "MANIFEST.json"),
                            "drain_half_key": "drop_unknown_tools"}
    study["name"] = out.name
    (out / "configs/comparison.json").write_text(json.dumps(study, indent=2))
    ident = json.loads((out / "MANIFEST.json").read_text())
    ident["files"] = manifest(out)
    (out / "MANIFEST.json").write_text(json.dumps(ident, indent=2))
    print(json.dumps({"out": str(out), "drain_half": study["drain_half"]}))


if __name__ == "__main__":
    main()
