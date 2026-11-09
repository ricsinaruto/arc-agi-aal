"""usage: fork_snapshot.py RUN_DIR OUT.json -- arc2.fork snapshot of a finished conversation run (tasks.jsonl and
events.jsonl from its arena/trials/c240 folder)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from arc2 import fork  # noqa: E402

run, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(x) for x in (run / "tasks.jsonl").read_text().splitlines() if x.strip()]
events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines() if x.strip()]
snap = fork.snapshot(rows, events)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(snap, separators=(",", ":")))
sub = sum(e["submitted"] for e in snap["entries"].values())
print(f"{out}: {len(snap['entries'])} entries, {sub} submitted, first-pass tokens {snap['first_pass_tokens']:,.0f}")
