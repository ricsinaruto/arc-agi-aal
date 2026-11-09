"""rhae_own: the curriculum on each level's own RHAE/token spread, no reference and no clear rate in
the score, a wide game/global prior so unplayed games get played."""
import math
import random
import statistics

import pytest

from arena import curriculum as C


def cur(**cfg):
    return C.Curriculum({"score": "rhae_own", "group_size": 4, "park_tries": 4, **cfg}, {"public": 1.0})


def test_z_uses_the_levels_own_median_and_the_group_sd_is_kept():
    c = cur()
    u = c.add_unit("ab12", 1, "public", "public:ab12", tokens=[99999], cap=200000)  # a reference never enters
    u.tokens = []
    c.observe("ab12", 1, [(True, 10000, None, 1.0, False), (True, 20000, None, 1.0, False),
                          (False, None, None, None, True), (False, None, None, None, False)])
    assert u.median_tokens() == 15000
    assert c.z_value(u, True, 15000, 1.0) == pytest.approx(1.0)
    zs = [1.5, 0.75, 0.0, 0.0]
    assert u.sdz[-1][1] == pytest.approx(statistics.pstdev(zs))


def test_unplayed_units_take_the_game_then_global_prior_and_draw_wide():
    c = cur()
    for g in ("ab12", "cd34"):
        for lvl in (1, 2):
            c.add_unit(g, lvl, "public", f"public:{g}", cap=200000).starts.append({"fresh": True})
    for _ in range(3):
        c.observe("ab12", 1, [(True, 10000, None, 1.0, False), (False, None, None, None, True)] * 2)
    mu, sd, games = c.own_prior()
    assert games == {"ab12": pytest.approx(c.units["ab12@L1"].sdz[-1][1])} and sd >= c.cfg["own_prior_floor"]
    seen, fresh = c.units["ab12@L1"], c.units["cd34@L1"]
    a = [c.score(seen, random.Random(s)) for s in range(300)]
    b = [c.score(fresh, random.Random(s)) for s in range(300)]
    assert statistics.pstdev(b) > statistics.pstdev(a)  # no evidence: the widest draws
    assert a == [c.score(seen, random.Random(s)) for s in range(300)]  # seeded


def breadth(score, seed, games=60, played=34, steps=40, per_step=8):
    """Steps of ``per_step`` groups (plan_one) until every game has had a group; ``played`` games start observed."""
    rng = random.Random(seed)
    c = C.Curriculum({"score": score, "group_size": 12, "park_tries": 12, "family_cap": 0.10, "window": 12},
                     {"public": 0.3, "envs": 0.7})
    truth = {}
    for i in range(games):
        g, src = f"g{i:02d}", "public" if i < 15 else "envs"
        for lvl in (1, 2, 3):
            truth[(g, lvl)] = (rng.uniform(0.05, 0.95), rng.lognormvariate(math.log(30000), 0.6))
            c.add_unit(g, lvl, src, f"{src}:{g}", cap=120000).starts.append({"fresh": True, "seed": i * 10 + lvl})

    def play(g, lvl):
        p, t = truth[(g, lvl)]
        rows = []
        for _ in range(12):
            ok = rng.random() < p
            rows.append((ok, int(t * rng.lognormvariate(0, 0.45)) if ok else None, None, 1.0 if ok else None, not ok))
        c.observe(g, lvl, rows)

    seen = {f"g{i:02d}" for i in range(played)}
    for g in sorted(seen):
        play(g, 1)
        play(g, 1)
    for step in range(1, steps + 1):
        for _ in range(per_step):
            pick = c.plan_one(step, seed=seed)
            play(pick["game"], pick["level"])
            seen.add(pick["game"])
        c.decay()
        if len(seen) == games:
            return step
    return steps + 1


def test_scheduler_sim_every_game_gets_a_group_sooner_than_under_the_reference_score():
    """60 games (34 played, as v6g at step 6), 8 groups per step: rhae_own reaches all 60 in a median of ~14 steps
    (rhae_tok ~30, two seeds never within 40)."""
    own = [breadth("rhae_own", s) for s in range(10)]
    ref = [breadth("rhae_tok", s) for s in range(10)]
    assert statistics.median(own) <= 16, own
    assert statistics.median(own) < statistics.median(ref) - 8, (own, ref)


def test_state_round_trip_keeps_sdz_and_an_older_state_loads():
    c = cur()
    c.add_unit("ab12", 1, "public", "public:ab12", cap=20000)
    c.observe("ab12", 1, [(True, 10000, None, 1.0, False), (False, None, None, None, True)])
    back = C.Curriculum.load_state(c.state_dict())
    assert back.units["ab12@L1"].sdz == c.units["ab12@L1"].sdz
    st = c.state_dict()
    del st["units"]["ab12@L1"]["sdz"]
    assert C.Curriculum.load_state(st).units["ab12@L1"].sdz == []
