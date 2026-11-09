import random

from arena.curriculum import Curriculum, usefulness


def pool(shares=None, **cfg):
    c = Curriculum({"groups": 10, "group_size": 6, "probe_share": 0.0, **cfg},
                   shares or {"pub": 0.5, "gen": 0.5})
    for src in ("pub", "gen"):
        for f in range(10):
            for lv in (1, 2):
                p = (f + 1) / 11
                c.add_unit(f"{src}{f}", lv, src, f"{src}-fam{f}", clears=round(8 * p), fails=8 - round(8 * p),
                           tokens=[10000, 12000])
                c.add_start({"game": f"{src}{f}", "level": lv, "source": src, "start": {"fresh": True, "seed": lv}})
    return c


def test_usefulness_peaks_mid_band():
    assert usefulness(0.5, 6) > usefulness(0.1, 6) > usefulness(0.0, 6) == 0.0
    assert usefulness(1.0, 6, token=0.3) == 0.3  # all-clear groups keep the token signal only


def test_plan_respects_shares_family_cap_and_size():
    c = pool()
    picks = c.plan(1)
    assert len(picks) == 10
    assert sum(p["source"] == "pub" for p in picks) == 5
    fams = [p["family"] for p in picks]
    assert len(fams) == len(set(fams))  # family cap 10% of 10 groups = 1
    assert all(p["tries"] == 6 for p in picks)


def test_frontier_preferred_over_extremes():
    c = pool(token_share=0.0)
    hits = []
    for upd in range(1, 30):
        hits += [p["p"] for p in c.plan(upd, seed=upd)]
    mid = sum(0.25 <= p <= 0.8 for p in hits) / len(hits)
    assert mid > 0.6


def test_token_share_cap():
    c = Curriculum({"groups": 8, "group_size": 6, "probe_share": 0.0, "token_share": 0.25, "family_cap": 1.0},
                   {"a": 1.0})
    for i in range(20):
        c.add_unit(f"g{i}", 1, "a", f"f{i}", clears=8, fails=0)
        c.add_start({"game": f"g{i}", "level": 1, "source": "a", "start": {"fresh": True}})
    picks = c.plan(1)
    assert sum(p["p"] > 0.85 for p in picks) <= 2


def test_observe_decay_retire_park_and_probe():
    c = Curriculum({"groups": 4, "group_size": 6, "probe_share": 0.5, "park_tries": 6, "reprobe_every": 2},
                   {"a": 1.0})
    for i in range(6):
        c.add_unit(f"g{i}", 1, "a", f"f{i}", clears=4, fails=4, tokens=[20000])
        c.add_start({"game": f"g{i}", "level": 1, "source": "a", "start": {"attempt": f"x{i}", "request": 3}})
    c.update = 1
    c.observe("g0", 1, [(False, 0)] * 6)
    c.observe("g0", 1, [(False, 0)] * 6)
    assert c.units["g0@L1"].state == "parked"
    for _ in range(6):  # prior fades with the half-life; tokens flat
        c.decay()
        c.observe("g1", 1, [(True, 8000)] * 6)
    assert c.units["g1@L1"].state == "retired"
    picks = c.plan(4)
    assert "g0@L1" in [p["unit"] for p in picks if p["why"] == "probe"]  # due a re-probe after 2 updates
    assert "g1@L1" not in [p["unit"] for p in picks]


def test_starts_rotate_without_repeat_and_harvest_adds_units():
    c = Curriculum({"groups": 1, "group_size": 6, "probe_share": 0.0}, {"a": 1.0})
    c.add_unit("g", 2, "a", "f", clears=3, fails=3)
    for r in (5, 9):
        c.add_start({"game": "g", "level": 2, "source": "a", "start": {"attempt": "rec", "request": r}})
    got = [c.plan(u)[0]["start"]["request"] for u in (1, 2)]
    assert sorted(got) == [5, 9]
    u = c.add_start({"game": "h", "level": 3, "source": "a", "family": "fh",
                     "start": {"attempt": "run/r01", "request": 40}})
    assert u.key == "h@L3" and u.seen == 0


def test_state_roundtrip():
    c = pool()
    c.plan(1)
    c.observe("pub3", 1, [(True, 9000), (False, 0)])
    d = Curriculum.load_state(c.state_dict())
    assert d.units["pub3@L1"].a == c.units["pub3@L1"].a
    assert [p["unit"] for p in d.plan(2, seed=7)] == [p["unit"] for p in c.plan(2, seed=7)]


def test_thin_source_filled_from_others():
    c = Curriculum({"groups": 6, "group_size": 6, "probe_share": 0.0, "family_cap": 1.0}, {"a": 0.5, "b": 0.5})
    c.add_unit("b0", 1, "b", "fb", clears=3, fails=3)
    c.add_start({"game": "b0", "level": 1, "source": "b", "start": {"fresh": True}})
    for i in range(10):
        c.add_unit(f"a{i}", 1, "a", f"fa{i}", clears=3, fails=3)
        c.add_start({"game": f"a{i}", "level": 1, "source": "a", "start": {"fresh": True}})
    assert len(c.plan(1, seed=random.Random(0).random())) == 6


def test_reward_spread_not_solve_rate_drives_the_score():
    c = Curriculum({"groups": 2, "group_size": 6, "probe_share": 0.0, "family_cap": 1.0, "progress": 0.0}, {"a": 1.0})
    flat = c.add_unit("flat", 1, "a", "f1", tokens=[10000])
    wide = c.add_unit("wide", 1, "a", "f2", tokens=[10000])
    for _ in range(3):  # both always clear; only "wide" spreads its tokens (reward = log(T_b / t))
        c.observe("flat", 1, [(True, 10000)] * 6)
        c.observe("wide", 1, [(True, t) for t in (3000, 5000, 10000, 20000, 30000, 40000)])
    assert c.predict(wide)[1] > 1.5 * c.predict(flat)[1]
    assert c.predict(flat)[0] > c.predict(wide)[0] - 1.0


def test_budget_outs_raise_the_cap_tier_before_parking():
    c = Curriculum({"groups": 1, "group_size": 6, "park_tries": 12, "max_tier": 1}, {"a": 1.0})
    u = c.add_unit("g", 2, "a", "f", cap=6000)
    for _ in range(2):
        c.observe("g", 2, [(False, 6000, None, None, True)] * 6)
    assert (u.tier, u.cap, u.state) == (1, 9000, "active")
    for _ in range(2):
        c.observe("g", 2, [(False, 9000, None, None, True)] * 6)
    assert u.state == "parked"  # top tier reached
    v = c.add_unit("h", 2, "a", "f2", cap=6000)
    for _ in range(2):
        c.observe("h", 2, [(False, 2000, None, None, False)] * 6)  # wrong moves, not the cap
    assert (v.tier, v.state) == (0, "parked")


def test_mastery_needs_clears_well_inside_the_cap():
    c = Curriculum({"groups": 1, "group_size": 6}, {"a": 1.0})
    near = c.add_unit("near", 1, "a", "f", tokens=[30000], cap=33000)
    easy = c.add_unit("easy", 1, "a", "f2", tokens=[8000], cap=33000)
    for _ in range(3):
        c.observe("near", 1, [(True, 30000, None, 1.0)] * 6)
        c.observe("easy", 1, [(True, 8000, None, 1.0)] * 6)
    assert near.state == "active" and easy.state == "retired"


def test_learning_progress_in_rhae_per_token():
    c = Curriculum({"groups": 1, "group_size": 6}, {"a": 1.0})
    u = c.add_unit("g", 1, "a", "f", tokens=[20000])
    for t in (20000, 20000, 10000, 8000):
        c.observe("g", 1, [(True, t, None, 1.0), (False, t)] * 3)
    assert u.progress() > 0.4


def test_small_pool_spreads_groups_over_units_by_score():
    c = Curriculum({"groups": 12, "group_size": 12, "probe_share": 0.0}, {"a": 1.0})
    for name, (cl, fl) in {"L1": (6, 6), "L2": (12, 0), "L3": (1, 11)}.items():
        c.add_unit("sc25", int(name[1]), "a", "sc25", clears=cl, fails=fl, tokens=[10000])
        c.add_start({"game": "sc25", "level": int(name[1]), "source": "a", "start": {"fresh": True}})
    picks = c.plan(1, seed=3)
    assert len(picks) == 12
    per = {}
    for p in picks:
        per[p["unit"]] = per.get(p["unit"], 0) + 1
    assert max(per.values()) <= 6 and per.get("sc25@L1", 0) >= per.get("sc25@L3", 0)


def test_probe_prior_survives_decay_for_unseen_units():
    # multi-game: most units go unplayed for many updates; their probe prior must not decay to Beta(1, 1)
    c = Curriculum({"groups": 1, "group_size": 12, "half_life": 2.0}, {"a": 1.0})
    easy = c.add_unit("e", 1, "a", "fe", clears=8, fails=0)
    hard = c.add_unit("h", 1, "a", "fh", clears=0, fails=8)
    seen = c.add_unit("s", 1, "a", "fs", clears=2, fails=2)
    c.observe("s", 1, [(True, 9000)] * 12)
    run_a = seen.a - 1.0 - seen.pa
    for _ in range(10):
        c.decay()
    assert easy.mean == 5.0 / 6.0 and hard.mean == 1.0 / 6.0  # prior weight 4: Beta(5, 1), Beta(1, 5)
    assert seen.a - 1.0 - seen.pa == run_a * 0.5 ** 5  # run evidence halves every 2 updates
    d = Curriculum.load_state(c.state_dict())
    assert d.units["e@L1"].pa == 4.0
    old = c.state_dict()
    for u in old["units"].values():  # a state written before the prior slots loads (prior then decays as before)
        u.pop("pa"), u.pop("pb")
    assert Curriculum.load_state(old).units["e@L1"].pa == 0.0


def test_plan_honours_exclude_and_start_limits():
    c = Curriculum({"groups": 6, "group_size": 12, "probe_share": 0.0}, {"a": 1.0})
    for g in ("x", "y", "z"):
        c.add_unit(g, 2, "a", f"f{g}", clears=3, fails=3)
        c.add_start({"game": g, "level": 2, "source": "a", "start": {"attempt": g, "request": 1}})
    picks = c.plan(1, exclude={"z@L2"}, limit={"x@L2": 1, "y@L2": 9})
    per = {}
    for p in picks:
        per[p["unit"]] = per.get(p["unit"], 0) + 1
    assert "z@L2" not in per and per.get("x@L2", 0) <= 1
    assert sum(per.values()) <= 6 and not c.blocked
    top = c.plan(1, groups=2, exclude={"x@L2", "y@L2"}, salt="+1")
    assert [p["unit"] for p in top] == ["z@L2", "z@L2"]


def test_loop_harvest_never_creates_units():
    c = Curriculum({"groups": 1, "group_size": 6}, {"a": 1.0})
    assert c.add_start({"game": "g", "level": 3, "source": "harvest", "start": {"attempt": "p", "request": 2}},
                       create=False) is None and not c.units


def test_depth_bins_split_the_update_equally():
    c = Curriculum({"groups": 12, "group_size": 12, "probe_share": 0.0, "depth_bins": [1, 2, 3, 4]},
                   {"a": 0.6, "b": 0.4})
    for i in range(30):
        for lv in range(1, 7):
            src = "a" if i % 2 else "b"
            c.add_unit(f"g{i}", lv, src, f"f{i}", clears=3, fails=3)
            c.add_start({"game": f"g{i}", "level": lv, "source": src, "start": {"fresh": True}})
    per, fams = {}, []
    for upd in range(1, 21):
        picks = c.plan(upd, seed=upd)
        assert len(picks) == 12
        fams.append(len({p["family"] for p in picks}))
        for p in picks:
            d = min(p["level"], 4)
            per[d] = per.get(d, 0) + 1
    assert per == {1: 60, 2: 60, 3: 60, 4: 60} and min(fams) == 12  # one group per family per update
    src_a = sum(1 for e in c.log for k, _ in e["picks"] if int(k[1:k.index("@")]) % 2)
    assert 0.5 < src_a / 240 < 0.7


def test_depth_bins_give_an_empty_depth_to_the_others():
    c = Curriculum({"groups": 6, "group_size": 12, "probe_share": 0.0, "depth_bins": [1, 2, 3], "family_cap": 1.0},
                   {"a": 1.0})
    for i in range(4):
        c.add_unit(f"g{i}", 1, "a", f"f{i}", clears=3, fails=3)
        c.add_start({"game": f"g{i}", "level": 1, "source": "a", "start": {"fresh": True}})
        c.add_unit(f"g{i}", 2, "a", f"f{i}", clears=3, fails=3)  # L2 without starts yet: not playable
    picks = c.plan(1, seed=1)
    assert len(picks) == 6 and all(p["level"] == 1 for p in picks)
