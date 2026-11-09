"""Golden (redesign cleanup, 26 Oct): what each live config's bundle ships: its files map (dest -> repo path, path-loaded
scripts included, e.g. bundle_bootstrap.py as bootstrap.py), packages and seed destinations; every mapped repo path
must still exist. A removal here is intended only when the config itself changes."""
import json

from common import ROOT, check

LIVE = ["launch/rl/runs/rl1-loop-seq-real-v8.json", "launch/rl/runs/rl1-loop-seq-glong-v5.json",
        "launch/rl/runs/b2-rl-hread-rd-v1.json"]


def test_live_bundle_file_maps():
    got, missing = {}, []
    for rel in LIVE:
        cfg = json.loads((ROOT / rel).read_text())
        files = cfg.get("files") or {}
        missing += [f"{rel}: {src}" for src in files.values() if not (ROOT / src).exists()]
        got[rel] = {"files": sorted(files.items()), "packages": sorted(cfg.get("packages") or []),
                    "seeds": sorted(s.get("dest") for s in cfg.get("seeds") or [])}
    assert not missing, missing
    check("bundle_files", got)
