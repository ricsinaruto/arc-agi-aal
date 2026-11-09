"""scripts/sft/vol.py per-run volume plan (no torch)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/sft"), str(ROOT / "src")]


def test_per_run_volume_plan():
    """Per-run volumes: "volume": "run" -> arc-run-<run> at /runs, sft-data and other runs' folders read
    at /in/<vol>/, the compile cache at /cache; a config without it keeps arc-prolong-runs and no inputs."""
    import vol
    spec = json.loads((ROOT / "launch/sft/sft-p75-qad-p75-v1.json").read_text())
    assert vol.volume_plan(spec) == ("arc-prolong-runs", [], [])
    per = {**spec, "volume": "run"}
    _, _, probs = vol.volume_plan(per)
    assert any(p.startswith("/runs/sft-data/s2-kl-v2: not this run's folder") for p in probs)
    assert not any("/runs/validation/sft-p75-qad-p75-v1/train" in p for p in probs)  # its own output stays at /runs
    fixed = json.loads(json.dumps(per).replace('"/runs/sft-data/', '"/in/arc-prolong-runs/sft-data/'))
    fixed["train"]["init_ckpt"] = "/in/arc-run-sft-p75-lrprobe-base/validation/sft-p75-lrprobe-base/ckpt-116"
    assert vol.volume_plan(fixed) == ("arc-run-sft-p75-qad-p75-v1", ["arc-prolong-runs", "arc-run-sft-p75-lrprobe-base"], [])
    assert "a cache on a per-run volume" in vol.volume_plan({**fixed, "cache_seed": "/runs/compile-cache/x"})[2][0]
    assert vol.volume_plan({**spec, "volume": "shared"})[2]
    leg = {"run": "job-a", "volume": "run", "volume_run": "job", "score": {"data": "/runs/validation/job/root", "out": "/runs/validation/job-a"}}
    assert vol.volume_plan(leg) == ("arc-run-job", [], [])  # legs of one job on the first run's volume
    try:
        from arc_infra.runops import records  # main's naming, when this checkout has it
    except ImportError:
        return
    if hasattr(records, "run_volume_name"):
        for run in ("sft-p75-qad-p75-v1", "x" * 80, "a/b c"):
            assert vol.run_volume_name(run) == records.run_volume_name(run)
