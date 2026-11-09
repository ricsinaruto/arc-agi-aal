"""play_ready.py RUN [RUN...]: exit 0 when every named Nebius play run has written a non-empty rollout.jsonl to its
bucket stream (serving and playing), else 1 with what is missing. The B300 trainer of a streaming round is submitted
only after this."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "./.claude/worktrees/nebius-opd/scripts")
import nebius_validation as nv  # noqa: E402

cloud = nv.Cloud(nv.load_config(), echo=lambda *a, **k: None)
missing = {}
for run in [r for r in sys.argv[1:] if r]:
    rec = nv.load_record(Path("."), cloud, run)
    current = (rec.get("attempts") or [{}])[-1].get("run_id", run)
    got = cloud.rclone("lsjson", "--recursive", "--files-only", cloud.key(nv.RUNS_PREFIX, "validation", current, "stream/trials"),
                       check=False, timeout=120)
    try:
        files = json.loads(got.stdout) if got is not None and got.returncode == 0 else []
    except ValueError:
        files = []
    rolls = [f for f in files if f["Path"].endswith("rollout.jsonl") and f.get("Size", 0) > 0]
    print(json.dumps({"run": run, "rollouts": len(rolls), "bytes": sum(f["Size"] for f in rolls)}))
    if not rolls:
        missing[run] = "no rollout yet"
sys.exit(1 if missing else 0)
