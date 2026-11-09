"""MFU-1 (23 Oct) feed fixes: k3_hold (the post-step k3 takes the step's kept host windows), the expand memo and its
parallel parse, host_mrope (M-RoPE positions from the encoder) and the pre marks. Each must hand the passes the same
tensors and give the same results as the path without it."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from test_rl1_window_feed import bits, make_trainer, tries  # noqa: E402,F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import flash_qlora_job as fq  # noqa: E402
import rl1_encode  # noqa: E402

COMMON = {"share_prefix": True, "prefix_chunk": 16384, "signed": True, "norm": "cap", "norm_cap": 1000}


class FakeCore:
    """Stands in for the model's core: ``get_rope_index`` as the library calls it (self.config, the vision rule)."""

    config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2), image_token_id=7)

    @staticmethod
    def get_vision_position_ids(self, start, grid, *a, **k):  # noqa: ARG004 - never reached by the stub windows
        raise AssertionError

    def get_rope_index(self, input_ids, mm_token_type_ids=None, image_grid_thw=None, **kwargs):
        merge = self.config.vision_config.spatial_merge_size
        base = torch.arange(input_ids.shape[1]).view(1, 1, -1) * merge + input_ids.view(1, 1, -1) % 5
        return torch.cat([base, base + 1, base + 2]), None

    def get_image_features(self, pixel_values, image_grid_thw, return_dict=True):
        return SimpleNamespace(pooler_output=(pixel_values * 2.0,))


def stub_inputs(n=40, images=3):
    ids = torch.arange(n).view(1, -1) % 11
    return {"input_ids": ids, "mm_token_type_ids": torch.zeros_like(ids), "image_grid_thw": torch.ones(images, 3),
            "pixel_values": torch.randn(images, 4, generator=torch.Generator().manual_seed(1))}


def run_k3(monkeypatch, tries, fast, hold):  # noqa: F811
    """Two groups accumulated (keep_old on a subsample), one step, the post-step k3: what the k3's passes saw."""
    log = []
    t = make_trainer(monkeypatch, log)
    t.args.tis_cap = 2.0
    sub = [tries[0], tries[3]]
    keep = [s["name"] for s in sub]
    extra = {**fast, **({"k3_hold": 1.0} if hold else {})}
    try:
        for i, g in enumerate([tries[:2], tries[2:]]):
            t.op_accumulate({"attempts": g, "lr": 1.0, "max_windows": 100, "first": i == 0, "keep_old": keep,
                             **COMMON, **extra})
        t.op_step({"lr": 1.0, **COMMON})
        n = len(log)
        k3 = t.op_k3_score({"attempts": sub, **COMMON, **extra, "gspo_clip": [0.015, 0.019]})
    finally:
        if t.encoders is not None:
            t.encoders[1].shutdown()
    return k3, log[n:], t


@pytest.mark.parametrize("fast", [{}, {"encode_procs": 2, "prefetch": 2}])
def test_k3_hold_scores_the_kept_windows_identically(monkeypatch, tries, fast):  # noqa: F811
    ref, ref_log, _ = run_k3(monkeypatch, tries, fast, hold=False)
    k3, k3_log, t = run_k3(monkeypatch, tries, fast, hold=True)
    assert ref["ok"] and k3["ok"] and ref["approx_kl"] > 0
    assert k3_log == ref_log  # the same tensors into every k3 pass
    for key in ("approx_kl", "approx_kl_tok", "clip_frac", "replies", "windows", "kept"):
        assert k3[key] == ref[key], key
    assert k3["encode"]["hold_hits"] == k3["windows"] > 0 and k3["encode"]["encoded"] == 0
    assert k3["encode"]["held"]["kept"] == k3["windows"] and t.k3_cache is None  # dropped after the measure
    assert ref["encode"]["hold_hits"] == 0


def test_k3_hold_resets_on_a_new_batch(monkeypatch, tries):  # noqa: F811
    t = make_trainer(monkeypatch, [])
    keep = [tries[0]["name"]]
    t.op_accumulate({"attempts": tries[:2], "lr": 1.0, "max_windows": 100, "first": True, "keep_old": keep,
                     "k3_hold": 1.0, **COMMON})
    first = t.k3_cache
    assert first is not None and first.stats["kept"] > 0
    t.op_accumulate({"attempts": tries[2:], "lr": 1.0, "max_windows": 100, "first": False, "keep_old": keep,
                     "k3_hold": 1.0, **COMMON})
    assert t.k3_cache is first  # later groups of the batch add to the same store
    t.op_accumulate({"attempts": tries[:2], "lr": 1.0, "max_windows": 100, "first": True, "keep_old": keep,
                     "k3_hold": 1.0, **COMMON})
    assert t.k3_cache is not first and t.k3_cache.stats["kept"] == first.stats["kept"]


@pytest.mark.parametrize("procs", [0, 2])
def test_expand_memo_and_parallel_parse_equal_the_plain_expand(monkeypatch, tries, procs):  # noqa: F811
    t = make_trainer(monkeypatch, [])
    plain = [s for spec in tries for s in rl1_encode.expand_one(spec, t.args.window)]
    try:
        first = t.expand(tries, procs=procs)
        calls = []
        monkeypatch.setattr(rl1_encode, "expand_one", lambda *a: calls.append(a) or [])
        again = t.expand(tries, procs=procs)
    finally:
        if t.encoders is not None:
            t.encoders[1].shutdown()
    assert bits(first) == bits(plain) == bits(again) and len(plain) > len(tries) and calls == []
    again[0]["w"] = 99.0  # callers may change their copies
    assert t.expand(tries)[0]["w"] == plain[0]["w"]


def test_positions_and_images_given_positions_and_features_match(monkeypatch):
    model = SimpleNamespace(model=FakeCore())
    inputs = stub_inputs()
    inputs["input_ids"][0, [5, 6, 20]] = FakeCore.config.image_token_id
    plain = fq.positions_and_images(model, inputs)
    marks = {}
    given = fq.positions_and_images(model, inputs, positions=rl1_encode.host_mrope(inputs, {
        "cls": f"{__name__}.FakeCore", "merge": 2}), features=fq.image_features(model, inputs), marks=marks)
    for start, stop in ((0, 6), (6, 21), (21, 40), (0, 40)):
        a, b = plain(start, stop), given(start, stop)
        assert bits(a["position_ids"]) == bits(b["position_ids"])
        assert ("mm_encoder_outputs" in a) == ("mm_encoder_outputs" in b)
        if "mm_encoder_outputs" in a:
            assert bits(a["mm_encoder_outputs"]["image"].pooler_output) == bits(b["mm_encoder_outputs"]["image"].pooler_output)
    assert set(marks) == {"vision", "rope"}


@pytest.mark.parametrize("fast", [{"encode_procs": 2, "prefetch": 2}, {"k3_hold": 1.0}])
def test_host_mrope_reaches_the_pass(monkeypatch, tries, fast):  # noqa: F811
    log, seen = [], []
    t = make_trainer(monkeypatch, log)
    t.model = SimpleNamespace(model=FakeCore())
    inner = t.window_pass
    t.window_pass = lambda made, **kw: seen.append((made["name"], made.get("mrope"), made["inputs"])) or inner(made, **kw)
    try:
        t.op_update({"attempts": tries, "lr": 0.0, "max_windows": 100, "host_mrope": True,
                     "keep_old": [s["name"] for s in tries], **COMMON, **fast})
    finally:
        if t.encoders is not None:
            t.encoders[1].shutdown()
    assert seen and all(m is not None for _, m, _ in seen)
    for _, positions, inputs in seen:
        expect, _ = FakeCore().get_rope_index(inputs["input_ids"])
        assert bits(positions) == bits(expect)
    # off: no positions in the window
    seen.clear()
    t.op_update({"attempts": tries[:1], "lr": 0.0, "max_windows": 100, **COMMON, "encode_procs": 0})
    assert seen and all(m is None for _, m, _ in seen)


def run_ahead(monkeypatch, tries, ahead):  # noqa: F811
    """Two accumulate calls, a step, the k3: with ``ahead`` each call names the next call's attempts."""
    log = []
    t = make_trainer(monkeypatch, log)
    t.args.tis_cap = 2.0
    fast = {"encode_procs": 2, "prefetch": 2}
    sub = [tries[0], tries[3]]
    keep = [s["name"] for s in sub]
    groups = [tries[:2], tries[2:]]
    nxt = [groups[1], sub]
    rows = []
    try:
        for i, g in enumerate(groups):
            rows.append(t.op_accumulate({"attempts": g, "lr": 1.0, "max_windows": 100, "first": i == 0, "keep_old": keep,
                                         "digest": True, **COMMON, **fast, **({"ahead": nxt[i]} if ahead else {})}))
        step = t.op_step({"lr": 1.0, "digest": True, **COMMON, **fast})
        k3 = t.op_k3_score({"attempts": sub, **COMMON, **fast, "gspo_clip": [0.015, 0.019]})
    finally:
        if t.encoders is not None:
            t.encoders[1].shutdown()
    return rows, step, k3, log


def test_ahead_encodes_the_next_call_identically(monkeypatch, tries):  # noqa: F811
    rows, step, k3, log = run_ahead(monkeypatch, tries, ahead=False)
    a_rows, a_step, a_k3, a_log = run_ahead(monkeypatch, tries, ahead=True)
    assert a_log == log and a_step["grad_digest"] == step["grad_digest"]
    for key in ("approx_kl", "approx_kl_tok", "clip_frac", "windows"):
        assert a_k3[key] == k3[key], key
    assert a_rows[0]["encode"]["ahead_sent"] > 0 and a_rows[1]["encode"]["ahead_hits"] > 0
    assert a_rows[1]["encode"]["ahead_sent"] > 0 and a_k3["encode"]["ahead_hits"] > 0  # kept through op_step
    assert rows[1]["encode"]["ahead_hits"] == 0 and k3["encode"]["ahead_hits"] == 0
