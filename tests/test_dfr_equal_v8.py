"""The equal scheduler patched onto the v8 harness (dfrv8-eq) orders and parks exactly as on v8.2 (A v2)."""
import importlib.util
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"v8": ROOT.parent / "v8/dist/arena-flash-competition-v8-sub",
           "v82": ROOT.parent / "v82sub/dist/v82sub-all25-w128-c110-24h-v1"}

PROBE = r'''
import json, random, sys, types
sys.path.insert(0, sys.argv[1])
from arena import priority, scheduler
cfg = priority.settings({"equal": True, "park_tokens": 300000, "tail_fade": False})
S = scheduler.Scheduler
class Clock:
    def __init__(self, t): self.t = t
    def completion_tokens(self): return self.t
out = []
rng = random.Random(0)
for trial in range(300):
    games = []
    for i in range(rng.randint(2, 12)):
        g = types.SimpleNamespace(index=i, readmissions=rng.choice([0, 1, 3]), level_capped=False,
                                  clock=Clock(rng.choice([0, 5000, 120000, 299999, 300000, 450000, rng.randint(0, 600000)])))
        games.append(g)
    me = types.SimpleNamespace(priority_cfg=cfg)
    me._equal_parked = types.MethodType(S._equal_parked, me)
    cands = []
    for seq, g in enumerate(games):
        value, info = S.priority_value(me, g)
        cands.append((g, value, info, rng.randint(0, 50) if trial % 2 else seq))
    ranked = S._priority_rank(me, cands)
    out.append([[c[0].index, c[1], c[2], c[3]] for c in ranked])
print(json.dumps(out))
'''


def load_builder():
    spec = importlib.util.spec_from_file_location("b", ROOT / "scripts/dfr_exact/build_v82_his_serving.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not all(p.exists() for p in SOURCES.values()), reason="v8 / v8.2 bundles not built here")
def test_equal_patch_v8_matches_v82(tmp_path):
    builder = load_builder()
    results, kinds = {}, {}
    for name, bundle in SOURCES.items():
        out = tmp_path / name
        shutil.copytree(bundle / "src", out / "src", ignore=shutil.ignore_patterns("__pycache__"))
        kinds[name] = builder.equal_patch(out, 300000)
        probe = subprocess.run([sys.executable, "-c", PROBE, str(out / "src")], capture_output=True, text=True, check=True)
        results[name] = json.loads(probe.stdout)
    assert kinds == {"v8": True, "v82": False}
    assert results["v8"] == results["v82"]
    parked_last = [r for r in results["v8"] if any(info.get("parked") for _, _, info, _ in r)]
    assert parked_last and all(
        [bool(info.get("parked")) for _, _, info, _ in r] == sorted(bool(info.get("parked")) for _, _, info, _ in r)
        for r in parked_last)
