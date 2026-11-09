"""scripts/sft/pick.py score curve and keep rule; scripts/sft/data.py reader vs heal_stream.SessionTail."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sel = load("sft_select", "scripts/sft/pick.py")


def log(*events):
    """events: ("m", completion tokens) or ("o", levels)."""
    return [{"kind": "model", "usage": {"completion_tokens": v}} if k == "m" else {"kind": "observation", "levels": v}
            for k, v in events]


def test_curve_and_score():
    points, tokens = sel.curve(log(("o", 0), ("m", 100), ("o", 0), ("m", 50), ("o", 1), ("m", 200), ("o", 1),
                                   ("o", 0), ("m", 10), ("o", 2)))
    assert tokens == 360 and points == [(0, 0), (150, 1), (360, 2)]
    ls, n = [100.0, 50.0, 0.0], 3  # weights 1, 2, 3
    assert sel.score_at(points, ls, n, 149) == 0
    assert sel.score_at(points, ls, n, 150) == pytest.approx(100 / 6)
    assert sel.score_at(points, ls, n, 10**9) == pytest.approx(200 / 6)
    assert sel.score_at(points, [115.0, 115.0, 0.0], n, 10**9) == pytest.approx(300 / 6)  # capped at 100 a level
    assert sel.score_at(points, [115.0, 80.0, 0.0], n, 10**9) == pytest.approx(275 / 6)  # cap on the sum only


def test_area_is_exact():
    points, ls, n = [(0, 0), (150, 1), (360, 2)], [100.0, 50.0, 0.0], 3  # scores 0, 100/6, 200/6
    assert sel.area(points, ls, n, 100) == 0
    assert sel.area(points, ls, n, 150) == 0
    assert sel.area(points, ls, n, 360) == pytest.approx(210 * 100 / 6)
    assert sel.area(points, ls, n, 400) == pytest.approx(210 * 100 / 6 + 40 * 200 / 6)
    assert sel.area([(0, 0), (0, 1)], [60.0, 0.0], 2, 10) == pytest.approx(200)  # a clear at token 0 counts from 0


def test_keep_rule_curve_and_mark():
    def inst(points, ls, tokens, reply=1):
        return {"game": "cd82", "points": points, "level_scores": ls, "level_count": 2, "tokens": tokens,
                "card_score": sel.score_at(points, ls, 2, tokens), "reply_tokens": reply}

    def run(name, *a, **k):
        return {"run": name, "trials": {"t": {"harness": h, "instances": {"k0": inst(*a, **k)}}}}

    h = {"config": {}, "games": []}
    ref = run("ref", [(0, 0), (500, 1)], [90.0, 0.0], 1000)  # score 30 from token 500
    # clears within its own 400 tokens: m = 400, ref 0 over [0, 400] -> kept by both rules
    fast = run("a", [(0, 0), (300, 1)], [30.0, 0.0], 400)
    # clears at 800 of 2000: m = 1000; mean over [0, m] 4 vs ref 15, score at m 20 vs 30 -> dropped by both
    slow = run("b", [(0, 0), (800, 1)], [60.0, 0.0], 2000)
    # equal at the mark (30 vs 30) but 400 tokens later: mean 3 vs 15 -> kept by the mark rule only
    late = run("d", [(0, 0), (900, 1)], [90.0, 0.0], 1000)
    # mean 12.5 vs 15: kept at 0.8, dropped at 0.9; kept by the mark rule (30 vs 30)
    near = run("e", [(0, 0), (583, 1)], [90.0, 0.0], 1000)
    # passes but has no recording: a row, never kept
    unrec = run("f", [(0, 0), (300, 1)], [90.0, 0.0], 1000, reply=0)
    other = {"run": "c", "trials": {"t": {"harness": {"config": {"temperature": 1.0}, "games": []}, "instances": {}}}}
    rows, excluded = sel.select([fast, slow, late, near, unrec, other], ref)
    got = [(r["run"], r["mark"], r["kept"], r["kept_mark"], r["recorded"]) for r in rows]
    assert got == [("a", 400, True, True, True), ("b", 1000, False, False, True), ("d", 1000, False, True, True),
                   ("e", 1000, False, True, True), ("f", 1000, False, True, False)]
    assert rows[2]["area_run"] == pytest.approx(3) and rows[2]["area_ref"] == pytest.approx(15)
    assert excluded[0]["run"] == "c" and "temperature" in excluded[0]["reason"]
    assert [r["kept"] for r in sel.rethreshold(rows, 0.8)] == [True, False, False, True, False]
    t = sel.table(rows)
    assert {k: v["pairs"] for k, v in t.items()} == {"curve>=1.0": 1, "curve>=0.9": 1, "curve>=0.8": 2,
                                                     "curve>=0.7": 2, "mark rule": 3}
    assert t["curve>=0.9"]["unrecorded"]["pairs"] == 1 and t["curve>=0.9"]["unrecorded"]["reply_tokens"] == 1000


def test_cap_preview():
    rows = [{"run": f"r{i}", "game": g, "reply_tokens": v, "kept": True}
            for i, (g, v) in enumerate([("a", 100), ("b", 200), ("c", 900), ("c", 100)])]
    c = sel.cap_preview(rows)
    assert c["median"] == 200 and c["before"] == {"a": 100, "b": 200, "c": 1000} and c["after"]["c"] == 400


def test_apply_rebuilt_and_not_stock():
    m = {"rows": [{"run": "a", "bucket_run": "a", "trial": "t", "game": "cd82", "instance_reply": {"k0": 0, "k1": 0},
                   "reply_tokens": 0, "recorded": False, "area_run": 2.0, "area_ref": 1.0, "kept_mark": True}],
         "heldout": [{"bucket_run": "a", "trial": "t", "game": "sp80", "instance_reply": {"k2": 5}, "reply_tokens": 5}]}
    res = [{"job": {"run": "a", "trial": "t", "k": "k0", "game_id": "cd82"}, "checks": {"ok": True}, "reply_tokens": 7},
           {"job": {"run": "a", "trial": "t", "k": "k1", "game_id": "cd82"}, "checks": {"ok": False}},
           {"job": {"run": "a", "trial": "t", "k": "k3", "game_id": "sp80"}, "checks": {"ok": True}, "reply_tokens": 4}]
    m = sel.apply_rebuilt(m, res)
    row = sel.rethreshold(m["rows"], 0.9)[0]
    assert row["instance_reply"] == {"k0": 7, "k1": 0} and row["kept"] and row["source"] == "rebuilt"
    assert [(h["game"], h["instance_reply"]) for h in m["heldout"]] == [("sp80", {"k3": 4})]
    assert m["rebuild"]["failed"] == ["a:k1"]
    assert sel.not_stock({"run": "b2-sft4x-c1s-v1a"}) and sel.not_stock({"run": "x", "assets": [
        "flash-input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround"]}) and not sel.not_stock({"run": "x"})



def test_kd_pool():
    h = {"config": {"temperature": 0.7, "x": 1}, "games": []}
    ref = {"run": "r", "trials": {"t": {"harness": h, "instances": {}}}}
    inst = {"k0": {"game": "cd82", "reply_tokens": 5}, "k1": {"game": "sp80", "reply_tokens": 9}}
    runs = [{"run": "a", "trials": {"t": {"harness": h, "instances": inst, "temperature": 0.7}}},
            {"run": "b", "trials": {"t": {"harness": {**h, "config": {"temperature": 1.0, "x": 1}}, "instances": inst,
                                          "temperature": 1.0}}},
            {"run": "c", "trials": {"t": {"harness": {**h, "config": {"temperature": 0.7, "x": 2}}, "instances": inst}}},
            {"run": "d", "trials": {"t": {"error": "unfinished", "partial": {"k4": {"game": "cd82", "reply_tokens": 3}},
                                          "temperature": 1.0}}}]
    pool = sel.kd_pool(runs, ref)
    assert [(r["bucket_run"], r["game"], r["t1"], r["finished"]) for r in pool["rows"]] == [
        ("a", "cd82", False, True), ("b", "cd82", True, True), ("d", "cd82", True, False)]
    assert [s["run"] for s in pool["skipped"]] == ["c"]
    t = sel.kd_table(pool)
    assert t["per_game"] == {"cd82": {"base": 5, "t1": 8}} and t["unfinished_reply_tokens"] == 3
    assert sel.find_key({"a": [{"b": 1}, {"temperature": 0.7}]}, "temperature") == 0.7


def test_build_levels_per_request():
    """build.levels_of: each request's level = levels completed before it + 1, on pick's curve rule."""
    bld = load("sft_build", "scripts/sft/build.py")
    rows = log(("m", 10), ("m", 5), ("o", 1), ("m", 7), ("o", 1), ("m", 3), ("o", 3))
    points = bld.level_points(rows)
    assert points == [(0, 0), (15, 1), (25, 3)] == [tuple(p) for p in sel.curve(rows)[0]]
    caps = [{"usage": {"completion_tokens": n}} for n in (10, 5, 7, 3, 4)]
    assert bld.levels_of(points, caps) == [1, 1, 2, 2, 4]


def test_build_loop_share_drops_degenerate_replies():
    """build.loop_share: share of 64-token windows repeating an earlier one; LOOP_MAX keeps normal replies (<= 1.4%
    on s2-kl-v2) and drops a looping one (52% / 87% there)."""
    bld = load("sft_build", "scripts/sft/build.py")
    normal = list(range(64 * 50))
    assert bld.loop_share(normal) == 0.0 and bld.loop_share([]) == 0.0
    loop = list(range(64 * 10)) + list(range(64, 128)) * 30  # 30 repeats of one window after 10 distinct
    assert abs(bld.loop_share(loop) - 30 / 40) < 1e-9 and bld.loop_share(loop) > bld.LOOP_MAX
    assert bld.loop_share(normal + list(range(64))) == 1 / 51 < bld.LOOP_MAX
    assert "reply_loop" in bld.REASONS


def test_build_level_segments_jobs_and_curriculum_order():
    """build.jobs_of carries a row's per-instance levels; curriculum_order lists the train examples of each level
    segment in the manifest's curriculum order (dropped files and unlisted segments left out)."""
    bld = load("sft_build", "scripts/sft/build.py")
    man = {"rows": [{"bucket_run": "r", "trial": "t", "game": "g", "kept": True, "instance_reply": {"k1": 5, "k2": 3},
                     "levels": {"k1": [2, 4], "k2": [1]}}], "heldout": []}
    train, held = bld.jobs_of(man)
    assert [(j["k"], j["levels"]) for j in train] == [("k1", [2, 4]), ("k2", [1])] and held == []

    def ex(k, lv, j):
        return {"file": f"train/r__t__{k}__L{lv:02d}__{j:03d}.pt", "run": "r", "k": k, "seg_level": lv}

    results = [{"job": {"split": "train"}, "examples": [ex("k1", 2, 0), ex("k1", 4, 1), ex("k1", 4, 2)]},
               {"job": {"split": "train"}, "examples": [ex("k2", 1, 0), ex("k2", 3, 1)]}]
    cur = [{"run": "r", "k": "k1", "level": 4}, {"run": "r", "k": "k2", "level": 1}, {"run": "r", "k": "k1", "level": 2}]
    order = bld.curriculum_order(results, cur, drop={"train/r__t__k1__L04__002.pt"})
    assert order == ["train/r__t__k1__L04__001.pt", "train/r__t__k2__L01__000.pt", "train/r__t__k1__L02__000.pt"]


def test_build_select_built_top_share_among_whole_segments():
    """select_built: only segments built whole (>= 99% of their level tokens) compete; per (game, level) the top
    ceil(20%) by rhae/tok stay; the rest of the examples are dropped and the kept curriculum keeps its order."""
    bld = load("sft_build", "scripts/sft/build.py")

    def ex(k, lv, n):
        return {"file": f"train/r__t__{k}__L{lv:02d}__000.pt", "run": "r", "k": k, "seg_level": lv, "reply_tokens": n}

    cur = [{"run": "r", "k": f"k{i}", "level": 1, "game": "g", "tokens": 100, "rhae_per_tok": i} for i in range(6)]
    cur.append({"run": "r", "k": "k9", "level": 2, "game": "g", "tokens": 100, "rhae_per_tok": 1})
    # k5 (best) built short, k4 next best whole; level 2 has no whole segment
    results = [{"job": {"split": "train"}, "examples": [ex(f"k{i}", 1, 100 if i != 5 else 50) for i in range(6)]},
               {"job": {"split": "train"}, "examples": [ex("k9", 2, 0)]}]
    drop, chosen, rec = bld.select_built(results, cur)
    assert [c["k"] for c in chosen] == ["k4"]  # ceil(0.2 x 5 whole) = 1
    assert [e["k"] for r in results for e in r["examples"]] == ["k4"] and len(drop) == 6
    assert rec["whole"] == 5 and rec["cells_without_whole"] == [["g", 2]]


def test_candidates_keep_student_runs_only_on_policy():
    """pick.py --on-policy: student / W4A16-served runs are candidates; by default they are not."""
    stock = {"run": "b2-v85p-x", "assets": ["flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4"]}
    w4 = {"run": "b2-onpol-p75c268-g1", "assets": [f"flash-input/models/{sel.W4A16}"]}
    student = {"run": sel.STUDENT[0], "assets": []}
    assert sel.candidates([stock, w4, student]) == [stock]
    assert sel.candidates([stock, w4, student], on_policy=True) == [stock, w4, student]


def test_mask_capped_keeps_ids_and_clears_only_extended_spans():
    torch = pytest.importorskip("torch")
    data = load("sft_data_mask", "scripts/sft/data.py")
    ids = torch.arange(20)
    labels = torch.full((20,), -100)
    labels[5:8] = ids[5:8]
    labels[12:16] = ids[12:16]
    other = ids[:8].clone()
    other[0] = 99  # a request of another chain at the same positions: not this example's
    kept = labels.clone()
    assert data.mask_capped(ids, labels, [(5, 8, other), (30, 34, torch.arange(34))]) == 0
    assert torch.equal(labels, kept)
    assert data.mask_capped(ids, labels, [(12, 16, ids[:16].clone())]) == 4
    assert torch.equal(labels[5:8], ids[5:8]) and bool((labels[8:] == -100).all())
    assert data.mask_capped(ids, labels, [(12, 16, ids[:16].clone())]) == 0


def test_grid_only_makes_one_real_call_per_image_size():
    """build.GridOnly: the first pass's grids, in image order, without re-processing a size it has seen."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    class Image:
        mode = "RGB"

        def __init__(self, size):
            self.size = size

    calls = []

    def real(images, **kwargs):
        calls.append(images[0].size)
        return {"image_grid_thw": torch.tensor([[1, *images[0].size]])}

    bld = load("sft_build", "scripts/sft/build.py")
    real.merge_size = 2
    shim = bld.GridOnly(real)
    out = shim([Image((4, 6)), Image((8, 2)), Image((4, 6))], return_tensors="pt")
    assert calls == [(4, 6), (8, 2)]
    assert out["image_grid_thw"].tolist() == [[1, 4, 6], [1, 8, 2], [1, 4, 6]]
    assert out["pixel_values"].numel() == 0
    assert shim.merge_size == 2  # the rest of the real processor shows through
