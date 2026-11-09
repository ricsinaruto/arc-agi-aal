"""MFU-1 C prefix carry (rl1_trainer CARRY): a window continuing the previous window's carried end equals the
recomputed prefix in fp32 (CPU, tiny qwen4_exp); a token mismatch, a later-first order or a keep_old window falls back
to the carry-free path bit for bit."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl1_memory  # noqa: E402
import rl1_trainer  # noqa: E402
from test_rl1_memory import V, tiny_model  # noqa: E402


@pytest.fixture
def trainer(monkeypatch):
    model = tiny_model()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(rl1_trainer.fq, "positions_and_images", lambda model, inputs, **kw: (lambda start, stop: {}))
    t = rl1_trainer.Trainer.__new__(rl1_trainer.Trainer)
    t.model, t.args, t.policy = model, SimpleNamespace(tis_cap=2.0), {}
    rl1_trainer.batch_head_hook(model)
    t.head = rl1_memory.head_capture(model)
    t.prefix_chunk = t.path_chunk = 64
    t.snapshots, t.snap_keep = True, 3
    return t


IDS = torch.randint(2, V, (1, 320), generator=torch.Generator().manual_seed(7))


def window(end, tail, ids=IDS, name="w"):
    ids = ids[:, :end]
    g = torch.Generator().manual_seed(end)
    cut = end - tail
    positions = torch.sort(torch.randperm(tail - 1, generator=g)[:24] + cut + 1).values
    n = len(positions)
    support = torch.full((n, 4), -1)
    arrays = {"positions": positions, "support": support, "weights": torch.linspace(-1, 2, n),
              "reference": -torch.rand(n, generator=g) * 3, "reply": torch.arange(n) // 7}
    return {"name": name, "inputs": {"input_ids": ids}, "tail": tail, "arrays": arrays, "temperature": 0.7,
            "argmax": False}


def run(t, made, shared):
    t.model.zero_grad(set_to_none=True)
    out = t.window_pass(made, scale=0.1, shared=shared)
    grads = {n: p.grad.clone() for n, p in t.model.named_parameters() if p.grad is not None}
    return out, grads


def same(a, b):
    return torch.equal(a[0]["actor"], b[0]["actor"]) and a[1].keys() == b[1].keys() and all(
        torch.equal(a[1][k], b[1][k]) for k in a[1])


def close(a, b, tol=1e-5):
    return torch.allclose(a[0]["actor"], b[0]["actor"], atol=tol, rtol=0) and all(
        torch.allclose(a[1][k], b[1][k], atol=tol, rtol=1e-4) for k in a[1])


def test_carried_equals_recomputed_and_fallbacks(trainer):
    A, B = window(200, 100, name="t#0"), window(300, 100, name="t#1")
    base = {}
    a0, b0 = run(trainer, A, base), run(trainer, B, base)
    assert not b0[0]["prefix_carried"] and b0[0]["prefix_start"] == 64  # the grid snapshot A left (cut 100)

    shared = {"carry_on": True}
    made0 = dict(rl1_trainer.CARRY)
    a1 = run(trainer, A, shared)
    assert same(a1, a0) and shared["carry"]["at"] == 200 and rl1_trainer.CARRY["made"] == made0["made"] + 1
    b1 = run(trainer, B, shared)
    assert b1[0]["prefix_carried"] and b1[0]["prefix_start"] == 200
    assert close(b1, b0)  # fp32: the tail as one forward vs prefix chunks of 64

    # a keep_old (k3) window takes no carried state: bit for bit the carry-free path
    shared = {"carry_on": True}
    run(trainer, A, shared)
    kept = run(trainer, {**B, "no_carry": True}, shared)
    assert not kept[0]["prefix_carried"] and same(kept, b0)

    # a token mismatch before the carried end: no carry
    other = IDS.clone()
    other[0, 150] = (other[0, 150] + 1) % V
    B2 = window(300, 100, ids=other, name="t#1")
    ref = run(trainer, B2, {})
    shared = {"carry_on": True}
    run(trainer, A, shared)
    mis = run(trainer, B2, shared)
    assert not mis[0]["prefix_carried"] and same(mis, ref)

    # the later window first: its carried end lies past the earlier window's cut, so nothing is reused
    shared = {"carry_on": True}
    run(trainer, B, shared)
    late = run(trainer, A, shared)
    assert not late[0]["prefix_carried"] and same(late, a0)


def test_carry_off_records_nothing(trainer):
    shared = {}
    run(trainer, window(200, 100), shared)
    assert "carry" not in shared


def test_op_update_orders_windows_per_try_and_marks_keep_old(monkeypatch, tmp_path):
    from test_rl1_window_feed import make_trainer, write_rollout

    base = {"window": 600, "floor": 64, "margin": 32, "credited_only": True, "whole": True}
    tries = [{**base, "rollout": write_rollout(tmp_path / f"t{i}", 12, i), "w": w, "share": share, "name": f"t{i}"}
             for i, (w, share) in enumerate([(1.0, "s0"), (-0.5, "s0"), (0.7, "s1")])]
    common = {"share_prefix": True, "prefix_chunk": 16384, "signed": True, "norm": "cap", "norm_cap": 1000,
              "max_windows": 100, "lr": 0.0}
    logs = {}
    for carry in (False, True):
        log = logs[carry] = []
        t = make_trainer(monkeypatch, log)
        t.args.tis_cap = 2.0
        out = t.op_update({"attempts": tries, **common, "prefix_carry": carry, "keep_old": ["t1"]})
        assert out["ok"]
    names = {c: [r["name"] for r in logs[c] if r["grad"]] for c in logs}
    assert sorted(names[True]) == sorted(names[False]) and len(names[True]) > 3  # the same windows
    import rl1_encode

    end_of = {w["name"]: w["end"] for tr in tries for w in rl1_encode.expand_one(tr, 600)}
    seq = {}
    for n in names[True]:
        seq.setdefault(n.split("#")[0], []).append(end_of[n])
    assert all(v == sorted(v) for v in seq.values())  # each try's windows in conversation order (window ends)
    order = [n.split("#")[0] for n in names[True]]
    assert order == sorted(order, key=lambda n: (n != "t0" and n != "t1", n))  # by start, then try
    marked = {r["name"].split("#")[0]: r["made"].get("no_carry") for r in logs[True] if r["grad"]}
    assert marked["t1"] is True and not marked.get("t0") and not marked.get("t2")


def test_margin_mode_moves_the_cut_to_the_carried_end(trainer):
    A = window(200, 100, name="t#0")
    C = window(300, 120, name="t#1")  # cut 180: overlaps A's end by 20 tokens
    C["arrays"] = {k: v[C["arrays"]["positions"] > 201] if torch.is_tensor(v) and v.shape[:1] == C["arrays"][
        "positions"].shape else v for k, v in C["arrays"].items()}
    ref = run(trainer, C, {})
    strict = {"carry_on": True}
    run(trainer, A, strict)
    s = run(trainer, C, strict)
    assert not s[0]["prefix_carried"] and s[0]["carry_moved"] == 0 and same(s, ref)  # strict: cut 180 < 200
    margin = {"carry_on": "margin"}
    run(trainer, A, margin)
    m = run(trainer, C, margin)
    assert m[0]["prefix_carried"] and m[0]["carry_moved"] == 20 and m[0]["cut"] == 200
    assert torch.allclose(m[0]["actor"], ref[0]["actor"], atol=1e-5, rtol=0)  # forward: the same numbers
    flat = lambda g: torch.cat([g[k].flatten() for k in sorted(g)])  # noqa: E731
    cos = torch.nn.functional.cosine_similarity(flat(m[1]), flat(ref[1]), dim=0)
    assert 0.99 < float(cos) < 1.0  # 20 context tokens' K/V no longer get a gradient


def test_carry_check_runs_every_arm_and_restores_the_chunk(monkeypatch, tmp_path):
    from test_rl1_window_feed import make_trainer, write_rollout

    base = {"window": 600, "floor": 64, "margin": 32, "credited_only": True, "whole": True}
    tries = [{**base, "rollout": write_rollout(tmp_path / f"t{i}", 8, i), "w": 1.0, "share": "s0", "name": f"t{i}"}
             for i in range(2)]
    t = make_trainer(monkeypatch, [])
    t.args.tis_cap = 2.0
    t.ref_by_try = lambda actor: {}
    chunks = []
    inner = t.window_pass
    t.window_pass = lambda made, **kw: (chunks.append(t.prefix_chunk), inner(made, **kw))[1]
    r = t.op_carry_check({"attempts": tries, "share_prefix": True, "signed": True, "norm": "cap", "norm_cap": 1000,
                          "max_windows": 100, "prefix_chunk": 16384})
    assert r["ok"] and list(r["arms"]) == ["base", "half_chunk", "repeat", "carry", "carry_margin"]
    assert all(r["arms"][a]["compare"]["grad_cos"] == pytest.approx(1.0) for a in r["arms"] if a != "base")
    assert 8192 in chunks and t.prefix_chunk == t.path_chunk == 16384 and t.ref is None


def test_carry_check_dict_arms_refs_ties_and_rows(monkeypatch, tmp_path):
    """MFU-1 floor read: dict arms keep / compare / both against labelled references, set QSA ties and per-arm
    fixes for that arm only, and ref_rows adds the worst-row read."""
    from test_rl1_window_feed import make_trainer, write_rollout

    base = {"window": 600, "floor": 64, "margin": 32, "credited_only": True, "whole": True}
    tries = [{**base, "rollout": write_rollout(tmp_path / f"t{i}", 8, i), "w": 1.0, "share": "s0", "name": f"t{i}"}
             for i in range(2)]
    t = make_trainer(monkeypatch, [])
    t.args.tis_cap = 2.0
    t.ref_by_try = lambda actor: {}
    seen = []
    inner = t.op_update
    t.op_update = lambda cmd: (seen.append((cmd["ref"], rl1_trainer.TIES["mode"], cmd.get("sel_cache"), t.prefix_chunk)),
                               inner(cmd))[1]
    monkeypatch.setitem(rl1_trainer.TIES, "mode", "index")
    monkeypatch.setitem(rl1_trainer.SEL, "on", True)
    monkeypatch.setattr(rl1_trainer, "op_paths", lambda model, cmd: rl1_trainer.SEL.update(on=bool(cmd.get("sel_cache"))))
    arms = [{"name": "A", "ref": "keep"}, {"name": "A8", "chunk": "half"},
            {"name": "T", "ref": "both", "ties": "topk"}, {"name": "T8", "ref_of": "T", "chunk": "half", "ties": "topk"},
            {"name": "S", "ref": "both", "fixes": {"sel_cache": False}}, {"name": "S8", "ref_of": "S", "chunk": "half"}]
    r = t.op_carry_check({"attempts": tries, "share_prefix": True, "signed": True, "norm": "cap", "norm_cap": 1000,
                          "max_windows": 100, "prefix_chunk": 16384, "arms": arms, "ref_rows": True, "sel_cache": True})
    assert r["ok"] and list(r["arms"]) == ["A", "A8", "T", "T8", "S", "S8"]
    assert [s[:3] for s in seen] == [("keep", "index", True), ("compare", "index", True), ("both", "topk", True),
                                     ("compare", "topk", True), ("both", "index", False), ("compare", "index", True)]
    assert [s[3] for s in seen] == [16384, 8192, 16384, 8192, 16384, 8192]
    assert r["arms"]["A"]["compare"] is None and all(r["arms"][a]["compare"] for a in ("A8", "T", "T8", "S", "S8"))
    rows = r["arms"]["A8"]["compare"]["rows"]
    assert rows["windows"] and rows["worst"] and {"d", "window", "pos", "after_cut", "after_image"} <= set(rows["worst"][0])
    assert rl1_trainer.TIES["mode"] == "index" and t.prefix_chunk == 16384 and t.ref is None


def test_rows_read_places_rows_against_the_grid_and_images():
    d = torch.tensor([0.0, 0.5, 2.0])
    geom = {"w": {"positions": torch.tensor([300, 310, 320]), "cut": 290, "prefix_start": 0,
                  "spans": [(90, 110), (150, 170), (295, 299)]}}
    r = rl1_trainer.rows_read({"w": (d, torch.tensor([True, True, True]))}, geom, step=100)
    w = r["windows"][0]
    assert w["prefix_images"] == 2 and w["split"] == 1 and w["split_odd"] == 1  # 90-110 crosses 100 (odd)
    assert r["worst"][0] == {"d": 2.0, "window": "w", "pos": 320, "after_cut": 30, "after_image": 21}
    r2 = rl1_trainer.rows_read({"w": (d, torch.tensor([True, True, True]))},
                               {"w": {**geom["w"], "spans": [(190, 210)]}}, step=100)
    assert r2["windows"][0]["split"] == 1 and r2["windows"][0]["split_odd"] == 0  # 200: on the 2*step grid too


def test_carry_check_applies_each_arms_paths_and_restores(monkeypatch, tmp_path):
    """m1fl1: the dispatcher applies op_paths once per op; an arm with fixes applies its own and restores the op's."""
    from test_rl1_window_feed import make_trainer, write_rollout

    base = {"window": 600, "floor": 64, "margin": 32, "credited_only": True, "whole": True}
    tries = [{**base, "rollout": write_rollout(tmp_path / "t0", 8, 0), "w": 1.0, "share": "s0", "name": "t0"}]
    t = make_trainer(monkeypatch, [])
    t.args.tis_cap, t.model = 2.0, None
    t.ref_by_try = lambda actor: {}
    applied = []
    monkeypatch.setitem(rl1_trainer.SEL, "on", True)
    monkeypatch.setattr(rl1_trainer, "op_paths", lambda model, cmd: (applied.append(cmd.get("sel_cache")),
                                                                     rl1_trainer.SEL.update(on=bool(cmd.get("sel_cache")))))
    arms = [{"name": "A", "ref": "keep"}, {"name": "S", "ref": "both", "fixes": {"sel_cache": False}}, {"name": "A2"}]
    r = t.op_carry_check({"attempts": tries, "share_prefix": True, "signed": True, "norm": "cap", "norm_cap": 1000,
                          "max_windows": 100, "arms": arms, "sel_cache": True})
    assert r["ok"] and applied == [False, True]
    assert r["arms"]["S"]["paths"]["sel_cache"] is False and r["arms"]["A2"]["paths"]["sel_cache"] is True
    noop = [{"name": "A", "ref": "keep"}, {"name": "S", "fixes": {"sel_cache": True}}]  # changes nothing: a failed arm
    r = t.op_carry_check({"attempts": tries, "share_prefix": True, "signed": True, "norm": "cap", "norm_cap": 1000,
                          "max_windows": 100, "arms": noop, "sel_cache": True})
    assert not r["ok"] and "left the live paths" in r["arms"]["S"]["error"]


def test_causal_check_on_a_causal_model_is_clean(monkeypatch):
    """causal_check on the tiny fp32 model: rows before ``length`` match between the short and the long call."""
    from test_rl1_memory import tiny_model

    model = tiny_model()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(rl1_trainer.fq, "positions_and_images", lambda model, inputs, **kw: (lambda start, stop: {}))
    monkeypatch.setitem(rl1_trainer.SEL, "on", True)
    monkeypatch.setattr(rl1_trainer, "op_paths", lambda model, cmd: rl1_trainer.SEL.update(on=bool(cmd.get("sel_cache", True))))
    t = rl1_trainer.Trainer.__new__(rl1_trainer.Trainer)
    t.model = model
    t.expand = lambda specs, procs=0: specs
    t.attempt_window = lambda spec, seed, release: {"inputs": {"input_ids": IDS}}
    r = t.op_causal_check({"attempts": [{"rollout": "x"}], "length": 200, "extra": 100, "rows": 50,
                           "configs": [{"name": "on"}, {"name": "off", "fixes": {"sel_cache": False}, "ties": "index"},
                                       {"name": "rep", "replay": "moe"}]})
    assert r["ok"] and r["length"] == 200 and r["longer"] == 300 and set(r["configs"]) == {"on", "off", "rep"}
    for conf in r["configs"].values():
        assert conf["modules"] and conf["first_over_bar"] is None
        assert all(m["max"] < 1e-4 for m in conf["modules"].values()), conf["modules"]
        assert conf["logprob"]["rows"] == 50 and conf["logprob"]["max"] < 1e-4
        assert conf["content"] and conf["first_over_bar_content"] is None and conf["logprob_content"]["max"] < 1e-4
    assert r["reversed"] == 100 and r["configs"]["off"]["paths"]["sel_cache"] is False
    assert model.training and rl1_trainer.ROUTE["call"] is None and rl1_trainer.SEL["on"] is True
    assert rl1_trainer.ROUTE["router"] == "off" and rl1_trainer.ROUTE["qsa"] == "off"


def test_prefix_route_replay_forces_the_short_calls_picks(monkeypatch):
    """ROUTE "prefix": a router call takes the recorded picks on the rows the record covers, its own on the rest."""
    from test_rl1_memory import tiny_model

    model = tiny_model().eval()
    rl1_trainer.route_patch()
    router = next(m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextTopKRouter")
    hidden = torch.randn(10, router.hidden_dim)
    monkeypatch.setitem(rl1_trainer.ROUTE, "router", "off")
    _, _, free = router(hidden)
    forced = torch.stack([torch.arange(router.top_k)] * 4).to(torch.int16)  # experts 0..k-1 on the first 4 rows
    monkeypatch.setitem(rl1_trainer.ROUTE, "router", "prefix")
    monkeypatch.setitem(rl1_trainer.ROUTE, "rec", [forced])
    monkeypatch.setitem(rl1_trainer.ROUTE, "i", 0)
    _, weights, got = router(hidden)
    assert torch.equal(got[:4], forced.long()) and torch.equal(got[4:], free[4:]) and rl1_trainer.ROUTE["i"] == 1
    assert torch.allclose(weights.float().sum(-1), torch.ones(10), atol=1e-3) or not router.norm_topk_prob


def test_carry_check_gap_dump_saves_engine_trainer_pairs(monkeypatch, tmp_path):
    from test_rl1_window_feed import make_trainer, write_rollout

    base = {"window": 600, "floor": 64, "margin": 32, "credited_only": True, "whole": True}
    tries = [{**base, "rollout": write_rollout(tmp_path / "t0", 8, 0), "w": 1.0, "share": "s0", "name": "t0"}]
    t = make_trainer(monkeypatch, [])
    t.args.tis_cap = 2.0
    t.ref_by_try = lambda actor: {}
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    t.mismatch = rl1_trainer.Trainer.mismatch.__get__(t)
    r = t.op_carry_check({"attempts": tries, "share_prefix": True, "signed": True, "norm": "cap", "norm_cap": 1000,
                          "max_windows": 100, "arms": [{"name": "A", "ref": "keep"}], "gap_dump": str(tmp_path / "d" / "gap_dump.pt")})
    assert r["ok"] and r["gap_dump"]["windows"] >= 1 and not rl1_trainer.GAP_DUMP
    rows = torch.load(tmp_path / "d" / "gap_dump.pt", weights_only=False)
    w = rows[0]
    assert w["engine"].shape == w["trainer"].shape == w["positions"].shape == w["tokens"].shape


def test_carry_reject_names_the_first_failing_check():
    ids = IDS[:, :120]
    carry = {"at": 80, "ids": ids[:, :80].clone(), "pkey": "k80", "route": None}
    on = {"carry_on": True}
    pk = lambda at: f"k{at}"
    reject = rl1_trainer.carry_reject
    assert reject(carry, on, False, 100, None, ids, None, pk) is None
    assert reject(carry, {}, False, 100, None, ids, None, pk) == "off"
    assert reject(None, {}, False, 100, None, ids, None, pk) is None
    assert reject(carry, on, True, 100, None, ids, None, pk) == "pure_window"
    assert reject(None, on, False, 100, None, ids, None, pk) == "none_held"
    assert reject(carry, on, False, 70, None, ids, None, pk) == "past_cut"
    assert reject(carry, on, False, 100, {"at": 96}, ids, None, pk) == "snapshot_deeper"
    other = ids.clone()
    other[0, 5] = (other[0, 5] + 1) % V
    assert reject(carry, on, False, 100, None, other, None, pk) == "ids"
    assert reject(carry, on, False, 100, None, ids, None, lambda at: "x") == "images"
    assert reject(carry, on, False, 100, None, ids, torch.zeros(120, 2, 2), pk) == "route_presence"
