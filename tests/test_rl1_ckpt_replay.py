"""adaptive_ckpt replayed on D2's 24 logged grad windows (rl1-loop-seq-d2-v2, 27 Oct; torch-free): the gate before a
B300 update leg. Each window's true allocated peak = its logged base + the D2 fit at the replayed n + the window's
residual at its logged n (the fit's scatter kept); the reserve grows by the window's logged gap (else ``gap``) when
its peak passes the reserve held, and is trimmed at each accumulate call's start, as D2's cards show."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))

import rl1_memory as m  # noqa: E402

WINDOWS = json.loads((ROOT / "tests/fixtures/d2_ckpt_windows.json").read_text())["ranks"]
FIT = (2.7242, 0.441, 3.2716)  # D2 least squares over the 22 warm windows (a, b, c): rms 0.64 GiB


def fit(tail, n):
    k = tail / 1024
    return k * FIT[0] + n * (k * FIT[1] + FIT[2])


class DecoderLayer:
    gradient_checkpointing = True


class Model:
    def __init__(self):
        self.layers = [type("FakeDecoderLayer", (DecoderLayer,), {})() for _ in range(48)]

    def modules(self):
        return iter(self.layers)


class ReplayMem:
    """rl1_memory.CudaMem on the replay: ``reset`` gives the window's logged base and the reserve held; ``peak`` the
    reserve's and the allocation's peaks."""

    def __init__(self):
        self.ctx, self.held, self.window, self.n = 0.0, 0.0, None, 0

    def reset(self):
        return self.window["base"]

    def peak(self):
        w = self.window
        allocated = w["base"] + fit(w["tail"], self.n) + (w["peak"] - w["base"] - fit(w["tail"], w["n"]))
        if allocated > self.held:
            self.held = allocated + (w.get("gap") if w.get("gap") is not None else self.gap)
        return self.held, allocated


@pytest.fixture(autouse=True)
def ckpt_state():
    old = {k: (dict(v) if isinstance(v, dict) else v) for k, v in m.CKPT.items()}
    yield
    m.CKPT.clear()
    m.CKPT.update(old)


def replay(rank, gap, keep=False):
    """D2's config (budget 250, gap prior 10, calibrate): per window (n, predicted card, card)."""
    m.CKPT.update(windows=0, raised=0, ooms=0, calibrated=None, given={}, gap=4.4, gap_source="prior",
                  a=2.72, b=0.441, c=3.27, first_margin_gib=5.0, step_peak=0.0, force=None)
    m.ckpt_settings({"adaptive_ckpt": True, "ckpt_calibrate": True, "ckpt_budget_gib": 250, "ckpt_gap_gib": 10})
    mem, model, out, acc = ReplayMem(), Model(), [], None
    mem.gap = gap
    m.CKPT["mem"] = mem
    for w in WINDOWS[rank]:
        if w["acc"] != acc:  # the call's start: D2's reserve trimmed back to the allocation (kept: RL1_KEEP_RESERVE)
            acc, mem.held = w["acc"], (mem.held if keep else w["base"])
        mem.window = w
        plan = m.ckpt_begin(model, w["tail"])
        mem.held = max(mem.held, w["base"])
        plan["held"] = mem.held
        mem.n = sum(not layer.gradient_checkpointing for layer in model.layers)
        row = m.ckpt_end(plan)
        out.append((row["n"], row["pred_gib"], row["card_gib"]))
    return out


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("gap", [2.5, 4.5])  # an unlogged window's gap: D2's median-ish and its largest
@pytest.mark.parametrize("rank", ["0", "1"])
def test_d2_windows_land_in_the_band(rank, gap, keep):
    rows = replay(rank, gap, keep)
    first, warm = rows[0], rows[1:]
    assert first[2] <= 255.0, first  # D2 logged 258 / 267
    in_band = [r for r in warm if 241.0 <= r[2] <= 252.0]
    assert len(in_band) >= 0.8 * len(warm), rows
    assert all(r[2] <= 255.0 for r in rows), rows
    assert all(r[1] <= 250.0 for r in rows), rows  # every plan within the budget (card terms)
    assert m.CKPT["gap"] < 10.0  # sampled from windows whose peak grew the reserve, not a stale high-water mark


def test_stale_reserve_is_not_a_gap():
    """D2 rank 1 window 2: reserve 267.4 GiB held from window 1, its own peak 190.6: no gap sample (logged 76.8)."""
    m.CKPT.update(on=True, windows=1, gap=0.49, gap_source="measured", mem=ReplayMem(), force=11, calibrate=False)
    mem = m.CKPT["mem"]
    mem.window, mem.held, mem.n = {"base": 78.763, "tail": 10336, "n": 11, "peak": 190.641}, 267.418, 11
    plan = m.ckpt_begin(Model(), 10336)
    plan["held"] = 267.418
    row = m.ckpt_end(plan)
    assert "gap_gib" not in row and m.CKPT["gap"] == 0.49


U1 = json.loads((ROOT / "tests/fixtures/u1_ckpt_windows.json").read_text())["ranks"]["0"]
U1_FIT = (2.4529, 0.4871, 2.915)  # U1 least squares over its 95 warm windows: rms 1.48, max 4.44 GiB
U1_GAP = 1.47  # U1's measured reserve gap


def test_u2_settings_put_u1_windows_in_the_band():
    """U2's knobs (budget 253, U1's fit as the prior, no calibration) on U1's 96 windows (truth: U1's fit at the replayed
    n plus each window's residual; U1 itself, budget 250 with the D2 prior: 64% of windows over 241.7, replayed 66%):
    allocated + gap over 241.7 in >= 80% of warm windows, every window <= 255."""
    m.CKPT.update(windows=0, raised=0, ooms=0, calibrated=None, given={}, gap=4.4, gap_source="prior",
                  first_margin_gib=5.0, step_peak=0.0, force=None)
    m.ckpt_settings({"adaptive_ckpt": True, "ckpt_calibrate": False, "ckpt_budget_gib": 253, "ckpt_gap_gib": 10,
                     "ckpt_a": U1_FIT[0], "ckpt_b": U1_FIT[1], "ckpt_c": U1_FIT[2]})
    mem, model, card = ReplayMem(), Model(), []
    m.CKPT["mem"] = mem
    for w in U1:
        mem.window = w
        plan = m.ckpt_begin(model, w["tail"])
        mem.held = max(mem.held, w["base"])
        plan["held"] = mem.held
        mem.n = sum(not layer.gradient_checkpointing for layer in model.layers)
        k = w["tail"] / 1024
        allocated = w["base"] + k * U1_FIT[0] + mem.n * (k * U1_FIT[1] + U1_FIT[2]) + (
            w["peak"] - w["base"] - (k * U1_FIT[0] + w["n"] * (k * U1_FIT[1] + U1_FIT[2])))
        mem.held = max(mem.held, allocated + U1_GAP)
        mem.peak = lambda a=allocated: (mem.held, a)
        m.ckpt_end(plan)
        card.append(allocated + U1_GAP)
    warm = card[1:]
    assert sum(c > 241.7 for c in warm) >= 0.8 * len(warm), card
    assert max(card) <= 255.0, card
