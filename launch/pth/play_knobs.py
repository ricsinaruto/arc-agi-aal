"""play_knobs.py PLAY_RUN '{"active_limit": 56, "kv_margin": 0.85}': set a running OPD play's live knobs (bucket
validation/<play>/stream/knobs.json; the runner copies it to the VM, the trial applies it within ~40 s and logs a
live_knobs event in its gate log). Only plays built with live_knobs (g1-v18 on) read it."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "./.claude/worktrees/nebius-opd/scripts")
import nebius_validation as nv  # noqa: E402

play, knobs = sys.argv[1], json.loads(sys.argv[2])
bad = set(knobs) - {"active_limit", "kv_margin"}
if bad or not isinstance(knobs.get("active_limit", 1), int) or not 0 < float(knobs.get("kv_margin", 0.5)) <= 1:
    sys.exit(f"REFUSED: knobs are active_limit (int) and kv_margin (0..1]; got {knobs}")
cloud = nv.Cloud(nv.load_config())
current = (nv.load_record(Path("."), cloud, play).get("attempts") or [{}])[-1].get("run_id", play)
cloud.put_text(json.dumps(knobs) + "\n", nv.RUNS_PREFIX, "validation", current, "stream/knobs.json")
print(json.dumps({"path": f"validation/{current}/stream/knobs.json", "knobs": knobs}))
