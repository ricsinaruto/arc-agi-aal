"""adaptive_ckpt (rl1_memory.CKPT, trainer cmd adaptive_ckpt, default off): per grad window n decoder layers run
without activation checkpointing. LoRA gradients and the served (R3) routes are the same at every n on the tiny
Flash-shaped model (test_rl1_cp's); 3 gloo ranks pick different n in one step (one rank empty); the max-tracking
raise after an underestimated window keeps the next window within the budget."""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
pytest.importorskip("peft")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))
sys.path.insert(0, str(ROOT / "tests"))

import test_rl1_cp as base  # noqa: E402  (its tiny Flash-shaped model, data and served table)
from arc_infra.runops.multirank import run_ranks  # noqa: E402


@pytest.fixture(autouse=True)
def ckpt_state():
    import rl1_memory

    old = {k: (dict(v) if isinstance(v, dict) else v) for k, v in rl1_memory.CKPT.items()}
    rl1_memory.CKPT.update(c=0.0, first_margin_gib=0.0)  # the two-term model these cases were written for (D2 fit: c)
    yield
    rl1_memory.CKPT.clear()
    rl1_memory.CKPT.update(old)


class StubMem:
    """CPU stand-in for rl1_memory.CudaMem: base ``base`` GiB; the window's peak from ``true`` (a, b per 1K tokens)
    and the layers left uncheckpointed when the forward ran (``seen``)."""

    def __init__(self, base=0.0, true=(0.0, 0.0), gap=0.0):
        self.base, self.true, self.ctx, self.tail, self.seen, self.gap = base, tuple(true) + (0.0,) * (3 - len(true)), 0.0, 0, 0, gap

    def reset(self):
        self.held = self.base  # the reserve held at the window's start: none above the allocated base
        return self.base

    def peak(self):
        a, b, c = self.true
        grown = self.base + self.tail / 1024 * (a + self.seen * b) + self.seen * c
        return grown + self.gap, grown  # (card, allocated)


def uncheckpointed(model):
    import rl1_memory

    return [i for i, layer in enumerate(rl1_memory.ckpt_layers(model)) if not layer.gradient_checkpointing]


def test_plan_levels_spread_and_knobs():
    import rl1_memory as m

    m.CKPT.update(on=True, budget_gib=245.0, a=2.5, b=0.5, margin=1.0, force=None, given={})
    assert m.ckpt_plan(4096, 100.0, 48) == (48, 100 + 4 * (2.5 + 24))  # 206 <= 245
    n, pred = m.ckpt_plan(16384, 100.0, 48)  # 16 x (2.5 + 0.5 n) <= 145: n <= 13.1
    assert n == 13 and pred <= 245
    assert m.ckpt_plan(65536, 100.0, 48)[0] == 0  # long window: full checkpointing
    assert m.ckpt_plan(4096, 100.0, 6)[0] == 6  # n above the layer count is skipped
    for layers, n in ((48, 6), (48, 16), (48, 48), (6, 3), (8, 1)):
        spread = m.ckpt_spread(layers, n)
        assert len(set(spread)) == n and all(0 <= i < layers for i in spread)
    assert m.ckpt_spread(48, 6) == [4, 12, 20, 28, 36, 44]
    m.ckpt_settings({"adaptive_ckpt": True, "ckpt_b": 0.4, "ckpt_c": 2.0, "ckpt_first_margin_gib": 7})
    assert m.CKPT["on"] and m.CKPT["b"] == 0.4 and m.CKPT["budget_gib"] == 245.0
    assert m.CKPT["c"] == 2.0 and m.CKPT["first_margin_gib"] == 7.0
    assert m.ckpt_peak(4096, 100.0, 48) == pytest.approx(100 + 4 * 2.5 + 48 * (4 * 0.4 + 2.0))
    m.CKPT["c"] = 0.0
    m.CKPT["b"] = 0.9  # raised by max-tracking: the same knob value on the next op keeps it
    m.ckpt_settings({"adaptive_ckpt": True, "ckpt_b": 0.4, "ckpt_budget_gib": 240})
    assert m.CKPT["b"] == 0.9 and m.CKPT["budget_gib"] == 240.0
    m.ckpt_settings({})
    assert not m.CKPT["on"] and m.ckpt_begin(None, 100) is None and m.ckpt_end(None) is None


def test_underestimate_on_first_window_cannot_oom_the_second():
    """Window 1 (4K tail) runs with b under its true value: all 48 layers off, measured 49 GiB over its prediction.
    The raise lands before window 2 (6K) is planned: its true peak stays within the 245 budget, where the
    unadapted plan (48 off) would pass the card (268 GiB)."""
    import rl1_memory as m

    layers = torch.nn.ModuleList([FakeDecoderLayer() for _ in range(48)])
    true = (2.556, 0.68)
    mem = StubMem(base=80.0, true=true)
    m.CKPT.update(on=True, budget_gib=245.0, a=2.556, b=0.40, margin=1.05, force=None, mem=mem, raised=0, gap=0.0)

    def window(tail):
        mem.tail = tail
        plan = m.ckpt_begin(layers, tail)
        mem.seen = len(uncheckpointed(layers))  # the forward / backward run here
        return m.ckpt_end(plan)

    stale = m.ckpt_plan(6144, 80.0, 48)[0]
    first = window(4096)
    assert first["n"] == 48 and first["peak_gib"] > first["pred_gib"] + 40 and first["raised"] > 0
    assert 80.0 + 6 * (true[0] + stale * true[1]) > 268  # the plan without the raise: past the card
    second = window(6144)
    assert 0 < second["n"] < 48 and second["peak_gib"] <= second["pred_gib"] <= 245.0, second
    assert m.CKPT["raised"] == 1
    assert not uncheckpointed(layers)  # every layer checkpointed again after each window
    for n in m.CKPT_LEVELS:  # the raise covers every n: none predicted below its true peak
        m.CKPT["force"] = n
        assert m.ckpt_plan(6144, 80.0, 48)[1] >= 80.0 + 6 * (true[0] + n * true[1])
    m.CKPT["force"] = None


def test_oom_window_restores_layers_and_raises_b():
    import rl1_memory as m

    layers = torch.nn.ModuleList([FakeDecoderLayer() for _ in range(48)])
    m.CKPT.update(on=True, budget_gib=245.0, a=2.556, b=0.4, margin=1.05, force=None, mem=StubMem(base=80.0), ooms=0)
    plan = m.ckpt_begin(layers, 4096)
    assert plan["n"] == 48 and len(uncheckpointed(layers)) == 48
    row = m.ckpt_end(plan, RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert row["oom"] and m.CKPT["b"] == 0.8 and m.CKPT["ooms"] == 1 and not uncheckpointed(layers)
    assert m.ckpt_plan(4096, 80.0, 48)[0] < 48


class FakeCuda:
    """torch.cuda's memory counters (bytes): ``alloc`` moves ``allocated`` and its peak; reset_peak_memory_stats."""

    def __init__(self):
        self.allocated = self.peak = self.reserved = self.reserved_peak = 0

    def alloc(self, gib):
        self.allocated += int(gib * 2**30)
        self.peak = max(self.peak, self.allocated)
        self.reserved = max(self.reserved, self.allocated)
        self.reserved_peak = max(self.reserved_peak, self.reserved)

    def synchronize(self):
        pass

    def reset_peak_memory_stats(self):
        self.peak, self.reserved_peak = self.allocated, self.reserved

    def mem_get_info(self):
        return 275 * 2**30 - self.reserved - 2**30, 275 * 2**30  # 1 GiB context

    def memory_allocated(self):
        return self.allocated

    def memory_reserved(self):
        return self.reserved

    def max_memory_allocated(self):
        return self.peak

    def max_memory_reserved(self):
        return self.reserved_peak


def test_step_peak_survives_per_window_resets():
    """mg.window_guard reads the step's peaks: with adaptive_ckpt resetting torch's peak per window, the step's
    torch_peak_gib (max of torch's and CKPT step_peak) is still the largest window's (the prefix pass's included),
    and each window's row reads its own peak over the 1 GiB context."""
    import types

    import rl1_memory as m

    cuda = FakeCuda()
    mem = m.CudaMem()
    mem.torch = types.SimpleNamespace(cuda=cuda)
    layers = torch.nn.ModuleList([FakeDecoderLayer() for _ in range(48)])
    m.CKPT.update(on=True, budget_gib=245.0, a=2.556, b=0.68, margin=1.05, force=None, mem=mem, step_peak=0.0)
    cuda.alloc(70)  # the model
    rows = []
    for prefix, grown in ((30, 100), (5, 40)):  # window 1: a big prefix pass and growth; window 2 small
        cuda.alloc(prefix)  # the prefix cache, kept through the window
        plan = m.ckpt_begin(layers, 8192)
        cuda.alloc(grown)
        cuda.alloc(-grown)
        rows.append(m.ckpt_end(plan))
        cuda.alloc(-prefix)
    step = max(cuda.max_memory_allocated() / 2**30, m.CKPT["step_peak"])
    assert step == pytest.approx(200.0) and cuda.max_memory_allocated() / 2**30 == pytest.approx(115.0)
    assert [r["peak_gib"] for r in rows] == [pytest.approx(201.0), pytest.approx(116.0)]
    assert [r["base_gib"] for r in rows] == [pytest.approx(101.0), pytest.approx(76.0)]


class FakeDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = True


def tail_pass(model, ids, weights, p0, table, routes=None):
    """window_pass's grad path on CPU: a read-only no-grad prefix cache, the tail with served routes, the backward;
    ``routes``: per router call (forward and recompute) its layer and indices."""
    import rl1_trainer as rt
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    modeling.Qwen4ExpTextDecoderLayer._can_checkpoint_with_cache = True
    rt.ROUTE.update(router="served", window=table, unserved_calls=0, agree=None, used=None)
    rt.ROUTE["call"] = rt.served_call(0, p0)
    model.eval()  # as the trainer's prefix_pass (train mode with checkpointing drops the cache)
    with torch.no_grad():
        cache = rt.fq.read_only(model.model(input_ids=ids[:, :p0], use_cache=True).past_key_values)
    model.train()
    rt.ROUTE["call"] = rt.served_call(p0, ids.shape[1])
    hidden = model.model(input_ids=ids[:, p0:], past_key_values=cache, use_cache=False).last_hidden_state
    base.loss_of(model, hidden, weights[:, p0:]).backward()
    rt.ROUTE.update(call=None, window=None, router="off")
    assert rt.ROUTE["unserved_calls"] == 0


def window_grads(n, p0=60, rows=None):
    """LoRA grads of one window with ``n`` layers uncheckpointed (through ckpt_begin / ckpt_end, n forced), the
    router calls' (layer, indices) and the layers that ran uncheckpointed."""
    import rl1_memory as m
    import rl1_trainer as rt

    with base.trainer_path():
        model, peft = base.build(ckpt=True)
        ids, weights = base.data()
        if rows is not None:
            ids, weights = ids[:, rows[0]:rows[1]], weights[:, rows[0]:rows[1]]
        rt.route_patch()
        layers = list(range(len(model.model.layers)))
        rt.route_columns(model, layers)
        table = base.served_table(len(layers))[: ids.shape[1]]
        calls = []
        hooks = [mod.register_forward_hook(lambda mod, a, out: calls.append((mod._arc_col, out[2].sort(-1).values)))
                 for mod in model.modules() if hasattr(mod, "_arc_col")]
        m.CKPT.update(on=True, force=n, mem=StubMem())
        plan = m.ckpt_begin(model, ids.shape[1] - p0)
        off = uncheckpointed(model)
        try:
            tail_pass(model, ids, weights, p0, table)
        finally:
            m.ckpt_end(plan)
        for h in hooks:
            h.remove()
        assert not uncheckpointed(model)
        return base.lora_grads(peft), calls, off, table, p0


@pytest.mark.parametrize("n", [1, 3, 6])
def test_grads_and_served_routes_equal_at_every_n(n):
    """Against full checkpointing (n 0): LoRA grads bitwise; the tail's routes are the table's in the forward and in
    every checkpointed layer's recompute; uncheckpointed layers do not recompute."""
    ref, ref_calls, ref_off, table, p0 = window_grads(0)
    got, calls, off, _, _ = window_grads(n)
    assert ref_off == [] and len(off) == n
    for name, g in ref.items():
        assert torch.equal(got[name], g), name
    assert any(float(g.abs().max()) > 0 for g in ref.values())
    layers = len(table[0])
    tail = [c for c in calls if c[1].shape[0] == table.shape[0] - p0]
    assert len(tail) == 2 * layers - n  # forward + recompute of the checkpointed layers only
    assert len([c for c in ref_calls if c[1].shape[0] == table.shape[0] - p0]) == 2 * layers
    want = table[p0:].long().sort(-1).values
    for col, idx in tail:
        assert torch.equal(idx, want[:, col]), col


def dp_rank(rank, world, arg):
    """One DP rank: its own window (rows), its own forced n; an empty rank runs no window. Grads all-reduced."""
    import torch.distributed as dist

    import rl1_memory as m

    rows, n = arg
    m.CKPT.update(on=True, force=n, mem=StubMem())
    if rows is None:
        model, peft = base.build(ckpt=True)
        grads, off = base.lora_grads(peft), []
    else:
        grads, _, off, _, _ = window_grads(n, p0=40, rows=rows)
    for g in grads.values():
        dist.all_reduce(g)
    return {k: g.numpy() for k, g in grads.items()}, off


@pytest.mark.multirank
def test_three_ranks_different_n_and_an_empty_rank():
    """3 gloo ranks, asymmetric windows (130 and 190 tokens, rank 2 empty), n 6 / 1 / - in the same step: the
    reduced grads equal the all-full-checkpointing step's bitwise."""
    import numpy as np

    rows = [(0, 130), (0, 190), None]
    full = run_ranks(dp_rank, world=3, inputs=[(r, 0) for r in rows], timeout=300)
    mixed = run_ranks(dp_rank, world=3, inputs=[(r, n) for r, n in zip(rows, (6, 1, 3))], timeout=300)
    assert [len(o[1]) for o in mixed] == [6, 1, 0] and all(not o[1] for o in full)
    for name, g in full[0][0].items():
        assert float(np.abs(g).max()) > 0, name
        for rank in range(3):
            assert np.array_equal(mixed[rank][0][name], g), (rank, name)


def test_saved_bytes_fixture():
    """The measured per-layer saved bytes the trainer's defaults rest on (D2 compares predicted vs measured)."""
    import rl1_memory as m

    fx = json.loads((ROOT / "tests" / "fixtures" / "ckpt_saved_bytes.json").read_text())
    no = fx["no_ckpt"]
    assert fx["full_ckpt"]["per_layer_kb"] == pytest.approx(4 * 2560 * 4 / 1024, rel=0.01)  # fp32 4 x hidden stream
    mean = (36 * no["gdn_layer_kb"]["total"] + 12 * no["attention_layer_kb"]["total"] + no["ple_kb"]) / 48
    b = (mean - fx["full_ckpt"]["per_layer_kb"]) * 1024 / 2**20  # GiB per 1K tokens beyond the checkpoint input
    assert b == pytest.approx(fx["trainer_defaults_gib_per_1k_tokens"]["b_per_uncheckpointed_layer"], abs=0.01)
    import importlib

    defaults = importlib.reload(m).CKPT  # module defaults: the D2 GPU fit (b 0.441 + c 3.27 per layer) under the CPU bound
    assert defaults["b"] == 0.441 < b and defaults["c"] == 3.27


# D1 (26 Oct, 2 x B300, full checkpointing): per accumulate call the card peak (nvidia-smi) and its windows
# (tokens, tail); every prefix 5.2K. D1's fit: 72.2 GiB + 2.556 GiB per 1K window tokens.
D1 = [(110.80, [(13161, 7938), (15150, 9927), (14559, 9336)]),
      (125.71, [(12744, 7521), (13525, 8302), (20615, 15392)]),
      (124.90, [(13687, 8464), (20950, 15727), (20511, 15288)]),
      (128.21, [(14832, 9609), (16848, 11625), (21795, 16572)]),
      (149.86, [(17349, 12123), (30703, 25477), (28547, 23321)]),
      (160.56, [(17428, 12202), (20486, 15260), (34584, 29358)]),
      (153.71, [(15833, 10607), (31709, 26483), (28930, 23704)]),
      (157.22, [(21348, 16122), (18833, 13607), (33171, 27945)])]


def test_predictor_against_d1_peaks():
    """The n 0 prediction (base: D1's intercept + the prefix at its slope; a 2.556, margin 1.05) of each call's
    largest window lies within -1% / +3% of the call's measured card peak (111-161 GiB); measured -0.1% to +1.4%
    (the card peak holds the allocator's reserve, which the allocated-terms predictor leaves to the margin). At 245
    GiB with every n 0-48: at b 0.68 (default) all 24 windows plan within 228-245 GiB, 14 at >= 241; at a calibrated
    b 0.4, 19 of 24 at >= 241 and every call's largest window (the call's peak) at >= 241, the low ones the longest
    tails (29.4K: n 6, 234 GiB; one layer there costs 12 GiB)."""
    import rl1_memory as m

    # D1's base is its card fit's intercept (card terms already): no gap added here
    m.CKPT.update(a=2.556, b=0.68, margin=1.05, budget_gib=245.0, force=None, levels=m.CKPT_LEVELS)
    for peak, windows in D1:
        tokens, tail = max(windows)
        base = 72.2 + 2.556 * (tokens - tail) / 1024
        m.CKPT["force"] = 0
        pred = m.ckpt_plan(tail, base, 48)[1]
        m.CKPT["force"] = None
        assert 0.99 * peak <= pred <= 1.03 * peak, (peak, pred)
    for b, low, at_bar, calls in ((0.68, 228.0, 14, 7), (0.4, 234.0, 19, 8)):
        m.CKPT["b"] = b
        hits, call_hits = 0, 0
        for _, windows in D1:
            planned = []
            for tokens, tail in windows:
                n, pred = m.ckpt_plan(tail, 72.2 + 2.556 * (tokens - tail) / 1024, 48)
                assert low <= pred <= 245.0 and n >= 4, (b, tail, n, pred)
                planned.append(pred)
            hits += sum(p >= 241.0 for p in planned)
            call_hits += max(planned) >= 241.0
        assert (hits, call_hits) == (at_bar, calls), b


def test_calibrate_lowers_an_over_cautious_c_once():
    """ckpt_calibrate: c 5.0 per layer (true 0). Window 1 (cold) is not used; window 2 (n 12) solves c to ~0; window
    3 plans n 25 (was 12), its true peak under the prediction and the budget; after that only raises (b)."""
    import rl1_memory as m

    layers = torch.nn.ModuleList([FakeDecoderLayer() for _ in range(48)])
    mem = StubMem(base=80.0, true=(2.556, 0.3, 0.0))
    m.CKPT.update(on=True, calibrate=True, calibrated=None, windows=0, budget_gib=245.0, a=2.556, b=0.3, c=5.0,
                  margin=1.0, force=None, mem=mem, raised=0, gap=0.0)

    def window(tail=16384):
        mem.tail = tail
        plan = m.ckpt_begin(layers, tail)
        mem.seen = len(uncheckpointed(layers))
        return m.ckpt_end(plan)

    first, second, third = window(), window(), window()
    assert first["n"] == second["n"] == 12 and "calibrated" not in first
    assert second["calibrated"]["c_old"] == 5.0 and second["calibrated"]["c_new"] == pytest.approx(0.0, abs=2e-3)
    assert third["n"] == 25 and third["peak_gib"] <= third["pred_gib"] <= 245.0 and "calibrated" not in third
    mem.true = (2.556, 0.3, 1.0)  # after the one shot: max-tracking raises only
    fourth = window()
    assert fourth["raised"] > 0 and m.CKPT["b"] > 0.3
    assert m.CKPT["calibrated"]["c_new"] == pytest.approx(0.0, abs=2e-3)


def calib_rank(rank, world, arg):
    """DP rank with ckpt_calibrate: ``windows`` windows of its rows (a virtual 16K tail for the stub card; 6 layers,
    levels 0-6, c 40 over a true 0); an empty rank none. Returns the reduced grads, its rows and c."""
    import torch.distributed as dist

    import rl1_memory as m
    import rl1_trainer as rt

    rows, windows, on = arg
    mem = StubMem(base=80.0, true=(2.556, 0.5))
    m.CKPT.update(on=on, calibrate=True, calibrated=None, windows=0, budget_gib=245.0, a=2.556, b=0.5, c=40.0,
                  margin=1.0, force=None, mem=mem, levels=tuple(range(7)), gap=0.0, first_margin_gib=0.0)
    out = []
    with base.trainer_path():
        model, peft = base.build(ckpt=True)
        ids, weights = base.data()
        rt.route_patch()
        rt.route_columns(model, list(range(len(model.model.layers))))
        if rows is not None:
            ids, weights = ids[:, rows[0]:rows[1]], weights[:, rows[0]:rows[1]]
            table = base.served_table(len(model.model.layers))[: ids.shape[1]]
            for _ in range(windows):
                mem.tail = 16384
                plan = m.ckpt_begin(model, 16384)
                mem.seen = len(uncheckpointed(model))
                try:
                    tail_pass(model, ids, weights, 40, table)
                finally:
                    out.append(m.ckpt_end(plan))
        grads = base.lora_grads(peft)
    for g in grads.values():
        dist.all_reduce(g)
    return {k: g.numpy() for k, g in grads.items()}, out, m.CKPT["c"]


@pytest.mark.multirank
def test_three_ranks_one_calibrates_one_empty():
    """Rank 0: 3 windows (cold, calibrating, then n up); rank 1: one cold window (keeps c); rank 2: none. Each rank
    plans with its own c; the reduced grads equal the full-checkpointing step's bitwise."""
    import numpy as np

    args = [((0, 130), 3), ((0, 190), 1), (None, 0)]
    full = run_ranks(calib_rank, world=3, inputs=[(r, w, False) for r, w in args], timeout=300)
    got = run_ranks(calib_rank, world=3, inputs=[(r, w, True) for r, w in args], timeout=300)
    rows0, rows1 = got[0][1], got[1][1]
    assert [r["n"] for r in rows0] == [2, 2, 6] and rows0[1]["calibrated"]["c_new"] == pytest.approx(0.0, abs=2e-3)
    assert [r["n"] for r in rows1] == [2] and got[1][2] == 40.0 and got[2][1] == [] and got[2][2] == 40.0
    assert all(r["peak_gib"] <= r["pred_gib"] <= 245.0 for r in rows0 + rows1)
    for name, g in full[0][0].items():
        assert float(np.abs(g).max()) > 0, name
        for rank in range(3):
            assert np.array_equal(got[rank][0][name], g), (rank, name)


def gap_window(layers, mem, tail):
    import rl1_memory as m

    mem.tail = tail
    plan = m.ckpt_begin(layers, tail)
    mem.seen = len(uncheckpointed(layers))
    return m.ckpt_end(plan)


def test_reserve_gap_lowers_the_next_window():
    """RL 26 Oct: window 1 (default gap 4.4) observes a 12 GiB reserve gap; window 2 (same tail) plans fewer
    layers, its card peak (allocated + 12) within its prediction and 245. A later smaller gap replaces it (D2 27 Oct)."""
    import rl1_memory as m

    layers = torch.nn.ModuleList([FakeDecoderLayer() for _ in range(48)])
    mem = StubMem(base=80.0, true=(2.556, 0.4), gap=12.0)
    m.CKPT.update(on=True, budget_gib=245.0, a=2.556, b=0.4, margin=1.05, force=None, mem=mem, raised=0, gap=4.4)
    first = gap_window(layers, mem, 12288)
    second = gap_window(layers, mem, 12288)
    assert first["gap_used"] == 4.4 and first["gap_gib"] == 12.0 and m.CKPT["gap"] == 12.0
    assert second["n"] < first["n"] and second["gap_used"] == 12.0
    assert second["card_gib"] <= second["pred_gib"] <= 245.0 < first["pred_gib"] - 4.4 + 12.0
    small = StubMem(base=80.0, true=(2.556, 0.4), gap=3.0)
    m.CKPT["mem"] = small
    assert gap_window(layers, small, 4096)["gap_gib"] == 3.0 and m.CKPT["gap"] == 3.0


def test_card_peak_within_budget_with_a_20_gib_reserve_gap():
    """RL 26 Oct: the budget is card terms. The allocator holds 20 GiB above the allocated peak whenever its reserve
    grows. Window 1 plans with the default gap (4.4) and measures 20: its card can pass 245 by up to 15.6 GiB (here
    251.6, under the 255 guard). From window 2 on every card peak stays within its prediction and 245; without the
    gap term (the old allocated-terms planner) the 30K window's card would pass 255."""
    import rl1_memory as m

    layers = torch.nn.ModuleList([FakeDecoderLayer() for _ in range(48)])
    mem = StubMem(base=80.0, true=(2.556, 0.4), gap=20.0)
    m.CKPT.update(on=True, budget_gib=245.0, a=2.556, b=0.4, margin=1.05, force=None, mem=mem, raised=0,
                  gap=4.4)

    def window(tail):
        mem.tail = tail
        plan = m.ckpt_begin(layers, tail)
        mem.seen = len(uncheckpointed(layers))
        return m.ckpt_end(plan)

    first = window(8192)
    assert first["gap_used"] == 4.4 and first["gap_gib"] == 20.0 and m.CKPT["gap"] == 20.0
    assert first["card_gib"] <= 245.0 + 20.0 - 4.4 and first["card_gib"] < 255.0
    for tail in (4096, 8192, 12288, 16384, 24576, 30000):
        row = window(tail)
        assert row["gap_used"] == 20.0 and row["card_gib"] <= row["pred_gib"] <= 245.0, row
    n_old = m.ckpt_plan(30000, 80.0, 48)[0]  # the allocated-terms plan (no gap): n 7 where the gap leaves fewer
    assert n_old > row["n"] and 80.0 + 30000 / 1024 * (2.556 + n_old * 0.4) + 20.0 > 255.0
    small = StubMem(base=80.0, true=(2.556, 0.4), gap=12.0)  # a later, smaller gap replaces it
    m.CKPT.update(mem=small, gap=20.0)
    small.tail = 4096
    plan = m.ckpt_begin(layers, 4096)
    small.seen = len(uncheckpointed(layers))
    assert m.ckpt_end(plan)["gap_gib"] == 12.0 and m.CKPT["gap"] == 12.0


def test_gap_prior_replaced_by_the_first_measurement():
    """RL 26 Oct (D2: budget 250, prior 10): the prior plans window 1 only; its measured 4.4 replaces it (down), so
    the windows after plan the card within [245, 250] instead of pinned ~5.6 GiB lower; a larger gap later is
    tracked up and back down. Window 1's card stays within 250 + (true gap - prior)."""
    import rl1_memory as m

    layers = torch.nn.ModuleList([FakeDecoderLayer() for _ in range(48)])
    mem = StubMem(base=80.0, true=(2.556, 0.4), gap=4.4)
    m.CKPT.update(on=True, budget_gib=250.0, a=2.556, b=0.4, margin=1.05, force=None, mem=mem, raised=0,
                  gap=4.4, gap_source="prior", given={})
    m.ckpt_settings({"adaptive_ckpt": True, "ckpt_budget_gib": 250, "ckpt_gap_gib": 10})
    assert m.CKPT["gap"] == 10.0 and m.CKPT["gap_source"] == "prior"
    first = gap_window(layers, mem, 8192)
    assert first["gap_source"] == "prior" and first["gap_used"] == 10.0 and first["gap_gib"] == 4.4
    assert first["card_gib"] <= 250.0 + (4.4 - 10.0)
    assert m.CKPT["gap"] == 4.4 and m.CKPT["gap_source"] == "measured"
    m.ckpt_settings({"adaptive_ckpt": True, "ckpt_budget_gib": 250, "ckpt_gap_gib": 10})  # same knob: kept
    for tail in (8192, 10240, 12288):
        row = gap_window(layers, mem, tail)
        assert row["gap_source"] in ("measured", "tracked") and row["gap_used"] == 4.4  # each sample replaces
        assert 245.0 <= row["pred_gib"] <= 250.0 and row["card_gib"] <= row["pred_gib"], row
    mem.gap = 6.0
    gap_window(layers, mem, 8192)
    row = gap_window(layers, mem, 8192)
    assert row["gap_source"] == "tracked" and row["gap_used"] == 6.0 and row["card_gib"] <= row["pred_gib"] <= 250.0
    mem.gap = 2.0
    gap_window(layers, mem, 8192)
    assert m.CKPT["gap"] == 2.0 and m.CKPT["gap_source"] == "tracked"  # each sample replaces (D2 27 Oct)
