"""Live engine-vs-pi_old check (22 Oct): per window |actor - engine reference| on valid sampled tokens, pooled per lag."""
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))

from arena.rl_loop import engine_gap_summary  # noqa: E402


def test_engine_gap_counts_valid_engine_tokens_only():
    import rl1_trainer

    actor = torch.tensor([-1.0, -2.0, -0.5, -3.0])
    ref = torch.tensor([-1.1, float("nan"), -1.5, -3.0])
    valid = torch.tensor([True, True, True, False])
    n, s, b = rl1_trainer.engine_gap(actor, ref, valid)
    assert n == 2 and math.isclose(s, 1.1, rel_tol=1e-9) and b == 1


def test_summary_pools_by_lag_with_a_window_se():
    rows = [["t-a-w0", 100, 5.0, 1], ["t-a-w1", 100, 7.0, 3], ["t-b-w0", 50, 10.0, 10], ["t-c", 0, 0.0, 0]]
    out = engine_gap_summary(rows, {"t-a": 0, "t-b": 1})
    assert out["lag0"]["tokens"] == 200 and out["lag0"]["mean_abs"] == 0.06 and out["lag0"]["beyond_half"] == 0.02
    assert out["lag1"] == {"windows": 1, "tokens": 50, "mean_abs": 0.2, "se": None, "beyond_half": 0.2}
    assert out["all"]["windows"] == 3 and out["all"]["se"] > 0
    assert engine_gap_summary(rows[:2], {})["lag0"]["tokens"] == 200  # no lag map: every window is lag 0
