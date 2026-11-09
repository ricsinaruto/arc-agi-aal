"""tools/ops/reconcile.py: a relaunch under the same run id reads as its current attempt (RL 26 Oct: attempt 2 of
rl1-loop-seq-t3-v1 was ended "unclassified" because attempt 1's failed row answered for the run id)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/ops"))
spec = importlib.util.spec_from_file_location("ops_reconcile", ROOT / "tools/ops/reconcile.py")
reconcile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reconcile)


def test_ledger_states_take_the_current_attempt():
    recs = [{"run": "r", "ledger": [{"row": {"run_id": "r", "state": "failed"}},
                                    {"row": {"run_id": "r/attempt-2", "state": "reserved"}}]},
            {"run": "s", "ledger": [{"row": {"run_id": "s", "state": "completed"}}]}, {"run": "q"}]
    states = reconcile.ledger_states(recs)
    assert states["r"] == "reserved" and states["r/attempt-2"] == "reserved" and states["s"] == "completed"
    assert "q" not in states
