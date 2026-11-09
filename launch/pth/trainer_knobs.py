"""trainer_knobs.py ROUND_RUN '{"plays": ["b2-...-v23", "b2-...-v22"], "lr": 5e-6}' (each key optional; the file holds
every knob, so pass the full set): swap a running streaming OPD round's student plays and/or set its peak lr (bucket validation/<round>/stream/knobs.json; the round's puller reads it every poll, pulls the new list and the
trainer's delta export goes back to full versions until every play in "confirm" (default: all) has applied one). Only
rounds on a bundle with modal_rl1.stream_knobs read it; the round's report logs each change under stream_knobs."""
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, "./.claude/worktrees/nebius-opd/scripts")
import nebius_validation as nv  # noqa: E402

SRC = Path(__file__).resolve().parents[2] / "scripts/posttrain/modal_rl1.py"  # the round's own check (no modal import)
ns = {}
exec(compile(ast.Module([n for n in ast.parse(SRC.read_text()).body if isinstance(n, ast.FunctionDef)
                         and n.name == "stream_knobs"], []), str(SRC), "exec"), ns)
stream_knobs = ns["stream_knobs"]

run, knobs = sys.argv[1], json.loads(sys.argv[2])
try:
    got = stream_knobs(knobs)
except ValueError as exc:
    sys.exit(f"REFUSED: {exc}")
if not run.startswith("sft-heal-round-"):
    sys.exit(f"REFUSED: {run} is not a heal round")
cloud = nv.Cloud(nv.load_config())
body = {k: v for k, v in got.items() if v is not None}
cloud.put_text(json.dumps(body) + "\n", nv.RUNS_PREFIX, "validation", run, "stream/knobs.json")
print(json.dumps({"path": f"validation/{run}/stream/knobs.json", **body}))
