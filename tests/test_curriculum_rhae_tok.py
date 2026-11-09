"""Hero rebuild: curriculum spread and learning progress on the per-try rhae/tok z, the cap-out tier
trigger and the cap ceiling."""

import json
import math
import random
from types import SimpleNamespace

import pytest

from arena import curriculum as C
from arena import play_explorer


def cur(**cfg):
    c = C.Curriculum({"score": "rhae_tok", "group_size": 4, "park_tries": 4, **cfg}, {"public": 1.0})
    c.add_unit("ab12", 2, "public", "public:ab12", clears=2, fails=2, tokens=[10000, 10000], cap=20000)
    return c


def test_spread_z_values():
    assert C.spread_z(0.0, 12, 1.1, 0.5) == 0.0  # all-fail groups carry nothing
    assert C.spread_z(1.0, 12, 1.1, 0.5) == pytest.approx(0.5 * math.sqrt(11 / 12))  # all clear: the clears' spread
    var = 0.5 * (0.25 + 1.21) - 0.25 * 1.21
    assert C.spread_z(0.5, 12, 1.1, 0.5) == pytest.approx(math.sqrt(var * 11 / 12) * (1 - 0.5 ** 12))


def test_z_is_rhae_times_reference_over_tokens_and_zero_for_fails_and_cuts():
    c = cur()
    u = c.units[c.key("ab12", 2)]
    assert c.z_value(u, False, None) == 0.0
    assert c.z_value(u, True, 5000, 0.8) == pytest.approx(0.8 * 10000 / 5000)
    assert c.z_value(u, True, 100, 1.0) == 6.0  # z_hi
    c.observe("ab12", 2, [(True, 10000, None, 1.0, False), (True, 5000, None, 1.0, False),
                          (False, None, None, None, True), (False, None, None, None, False)])
    assert u.zn == 2 and u.zs == pytest.approx(3.0) and u.zss == pytest.approx(5.0)
    upd, mean, se = u.zrate[-1]
    assert mean == pytest.approx(0.75)  # the cut and the fail count as 0 in the per-try mean
    assert se > 0


def test_progress_z_is_a_clipped_t_statistic():
    u = cur().units[C.Curriculum.key("ab12", 2)]
    u.zrate = [(i, 1.0, 0.1) for i in range(6)]
    assert u.progress_z() == 0.0
    u.zrate = [(0, 1.0, 0.1), (1, 1.0, 0.1), (2, 1.0, 0.1), (3, 2.0, 0.1), (4, 2.0, 0.1), (5, 2.0, 0.1)]
    assert u.progress_z() == 1.0  # |t| far over 3, clipped
    u.zrate = [(0, 1.0, 0.5), (1, 1.1, 0.5)]
    assert 0.0 < u.progress_z() < 0.2  # a thin, noisy change reads as little progress


def test_rhae_tok_score_is_seeded_and_explores_thin_units():
    c = cur()
    c.add_unit("cd34", 1, "public", "public:cd34", clears=2, fails=2, tokens=[10000], cap=20000)
    thick, thin = c.units[c.key("ab12", 2)], c.units[c.key("cd34", 1)]
    for _ in range(10):
        c.observe("ab12", 2, [(True, 10000, None, 1.0, False), (False, None, None, None, True)] * 2)
    a = [c.score(thick, random.Random(s)) for s in range(200)]
    b = [c.score(thin, random.Random(s)) for s in range(200)]
    assert a == [c.score(thick, random.Random(s)) for s in range(200)]
    assert max(b) - min(b) > max(a) - min(a)  # fewer clears, wider draws
    assert C.Curriculum({"group_size": 4}).cfg["score"] == "logeq"  # the old rule stays the default


def test_cap_out_trigger_tiers_up_a_level_with_clears_and_the_ceiling_holds():
    c = cur(tier_trigger="cap_out", park_p=0.0, cap_ceiling=25000)
    u = c.units[c.key("ab12", 2)]
    c.observe("ab12", 2, [(True, 9000, None, 1.0, False)] + [(False, None, None, None, True)] * 3)
    assert u.tier == 1 and u.cap == 25000  # 20,000 x 1.5 = 30,000, held at the ceiling
    c.observe("ab12", 2, [(True, 9000, None, 1.0, False)] + [(False, None, None, None, True)] * 3)
    assert u.tier == 1 and u.cap == 25000  # at the ceiling: no further tier
    d = cur(park_p=0.0)  # default trigger: a unit with a clear never tiers up at park_p 0
    v = d.units[d.key("ab12", 2)]
    d.observe("ab12", 2, [(True, 9000, None, 1.0, False)] + [(False, None, None, None, True)] * 3)
    assert v.tier == 0 and v.cap == 20000


def test_add_unit_and_running_caps_respect_the_ceiling():
    c = cur(cap_ceiling=15000)
    c.add_unit("ef56", 3, "public", "public:ef56", cap=64000)
    assert c.units[c.key("ef56", 3)].cap == 15000
    u = SimpleNamespace(cap=10000, tier=1)
    loop = SimpleNamespace(cfg={"mg_sets": {"ef56-L3": {"role": "train", "game": "ef56", "level": 3, "cap": 10000}}},
                           mg_curr=SimpleNamespace(cfg={"tier_factor": 1.5, "cap_ceiling": 15000},
                                                   key=lambda g, l: f"{g}-L{l}", units={"ef56-L3": u}),
                           mg_level_obs={"ef56-L3": {"clears": [16000, 16000, 16000], "censored": []}})
    assert play_explorer.update_caps(loop, None, {"max": 15000}) == {"ef56-L3": 15000}  # running caps: mg.running_caps max
    assert loop.cfg["mg_sets"]["ef56-L3"]["cap"] == 15000 and u.cap == 15000  # 15K x 1.5 tier, held at 15K


def test_state_round_trip_and_an_older_state_without_z():
    c = cur()
    c.observe("ab12", 2, [(True, 10000, None, 1.0, False), (False, None, None, None, True)] * 2)
    back = C.Curriculum.load_state(json.loads(json.dumps(c.state_dict())))
    u, v = c.units[c.key("ab12", 2)], back.units[c.key("ab12", 2)]
    assert (v.zn, v.zs, v.zss) == (u.zn, u.zs, u.zss) and [list(x) for x in v.zrate] == [list(x) for x in u.zrate]
    old = c.state_dict()
    for d in old["units"].values():
        for k in ("zn", "zs", "zss", "zrate"):
            d.pop(k)
    w = C.Curriculum.load_state(old).units[c.key("ab12", 2)]
    assert w.zn == 0.0 and list(w.zrate) == []


def test_token_cost_prefers_cheap_units_from_their_own_tries_only():
    """envs audit 24 Oct: long units took 34% of hero tokens at half the reward spread per token. no
    reference; the factor uses each level's own tries this run."""
    c = cur()
    c.add_unit("cd34", 2, "public", "public:cd34", clears=2, fails=2, tokens=[64000], cap=80000)
    cheap, dear = c.units[c.key("ab12", 2)], c.units[c.key("cd34", 2)]
    c.cfg["token_cost"] = 1.0
    assert c.token_factor(cheap) == 1.0 == c.token_factor(dear)  # no tries yet: the prior's tokens never count
    c.observe("ab12", 2, [(True, 10000, None, 1.0, False), (False, None, None, None, True)])  # budget-out at cap 20000
    c.observe("cd34", 2, [(True, 40000, None, 1.0, False), (False, None, None, None, False)])  # other fail: own median
    assert c.own_tokens(cheap) == 15000 and c.own_tokens(dear) == 40000
    mid = 40000  # median of [15000, 40000] (upper middle)
    assert c.token_factor(cheap) == pytest.approx(mid / 15000) and c.token_factor(dear) == pytest.approx(1.0)
    s1 = [c.score(cheap, random.Random(s)) - c.score(dear, random.Random(s)) for s in range(50)]
    c.cfg["token_cost"] = 0.0
    assert c.token_factor(cheap) == 1.0
    s0 = [c.score(cheap, random.Random(s)) - c.score(dear, random.Random(s)) for s in range(50)]
    assert sum(s1) > sum(s0)
    c2 = C.Curriculum.load_state(json.loads(json.dumps(c.state_dict())))
    assert c2.own_tokens(c2.units[c.key("ab12", 2)]) == 15000
    old = c.state_dict()
    for d in old["units"].values():
        d.pop("tok_obs")  # a state saved before tok_obs
    assert C.Curriculum.load_state(old).units[c.key("ab12", 2)].tok_obs == []


def test_deep_share_keeps_deep_groups_in_the_window():
    c = C.Curriculum({"score": "rhae_tok", "group_size": 4, "window": 4, "probe_share": 0.0, "family_cap": 1.0,
                      "deep_share": 0.5, "deep_level": 3}, {"public": 1.0})
    c.add_unit("ab12", 1, "public", "public:ab12", clears=4, fails=4, tokens=[5000], cap=20000)
    c.add_unit("ab12", 3, "public", "public:ab12", clears=1, fails=7, tokens=[20000], cap=40000)
    for u in c.units.values():
        u.starts = [{"fresh": True}]
    levels = [c.plan_one(1, seed=i)["level"] for i in range(8)]
    assert sum(lv >= 3 for lv in levels[-4:]) >= 2
    c2 = C.Curriculum({**c.cfg, "deep_share": 0.0}, {"public": 1.0})
    assert c2.cfg["deep_share"] == 0.0


def _sim(token_cost=0.0, deep_share=0.0, groups=120, seed=0, score="rhae_tok"):
    """plan_one + observe over a mixed pool: per source, short and long units at L1-L4 (long = 4x the tokens per try,
    same clear rate and RHAE spread); L3/L4 only in two games per source."""
    shares = {"public": 0.5, "generated": 0.3, "sonpham": 0.2}
    c = C.Curriculum({"score": score, "group_size": 6, "window": 12, "probe_share": 0.1, "family_cap": 0.25,
                      "token_cost": token_cost, "deep_share": deep_share, "deep_level": 3}, shares)
    for src in shares:
        for i, long in enumerate((False, True) * 3):
            for lv in ((1, 2, 3, 4) if i < 2 else (1, 2)):  # deep starts are scarce (hero: 61 of 863 L2+ units)
                g = f"{src[0]}{i}"
                c.add_unit(g, lv, src, f"{src}:{g}", clears=2, fails=2, tokens=[10000], cap=80000 if long else 20000)
                c.units[c.key(g, lv)].starts = [{"fresh": True}]
    rng = random.Random(seed)
    tok = {"long": 0, "short": 0}
    made = {s: 0 for s in shares}
    deep = 0
    for k in range(groups):
        p = c.plan_one(k // 8, seed=seed)
        u = c.units[p["unit"]]
        long = u.cap == 80000
        rows = []
        for _ in range(6):
            t = int(rng.lognormvariate(math.log(40000 if long else 10000), 0.4))
            ok = rng.random() < 0.5
            rows.append((ok, t if ok else None, None, rng.uniform(0.3, 1.0), not ok))
            tok["long" if long else "short"] += t if ok else u.cap
        c.observe(u.game, u.level, rows)
        made[u.source] += 1
        deep += u.level >= 3
    return tok["long"] / (tok["long"] + tok["short"]), deep / groups, {s: made[s] / groups for s in made}


@pytest.mark.parametrize("score", ["rhae_tok", "rhae_own"])
def test_scheduler_sim_token_cost_and_deep_share_hold_the_mix(score):
    base_long, base_deep, base_mix = _sim(score=score)
    long_tc, _, mix_tc = _sim(token_cost=0.5, score=score)
    _, deep_ds, mix_ds = _sim(deep_share=0.5, score=score)
    assert long_tc < base_long - 0.05  # fewer tokens on long units at equal spread
    assert deep_ds >= 0.45 and deep_ds > base_deep  # the window keeps its deep share
    for mix in (base_mix, mix_tc, mix_ds):  # source quotas hold either way
        assert abs(mix["public"] - 0.5) < 0.05 and abs(mix["generated"] - 0.3) < 0.05
