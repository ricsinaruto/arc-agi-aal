"""RL redesign stage B (arena.rl_backlog): roster, historical rows, ranking, budgets, backlog."""
import json
import math
import random
import statistics
from pathlib import Path

import pytest

from arena import rl_backlog as rb

FIX = Path(__file__).parent / "fixtures" / "rl_backlog"
ROWS = json.loads((FIX / "v85p_rows.json").read_text())["rows"]  # real v8.5p rows: sb26 s5i5 ka59 su15
POINTS = Path(__file__).parents[1] / "launch" / "rl" / "data" / "rl1_points.json"


def try_(cleared, tokens, rhae=1.0):
    return {"cleared": cleared, "tokens": tokens, "actions": 10, "rhae": rhae if cleared else 0.0}


# ---- roster ----

def test_roster_is_the_owner_split_disjoint_and_all_25():
    assert len(rb.TRAIN) == 15 and len(rb.HELD_OUT) == 10
    assert not set(rb.TRAIN) & set(rb.HELD_OUT)
    public = {"ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "ls20", "m0r0", "r11l",
              "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30"}
    assert set(rb.TRAIN) | set(rb.HELD_OUT) == public and len(public) == 25
    assert public <= set(rb.load_baselines(POINTS))  # every game has human baselines
    assert set(rb.LEFT_OUT) <= set(rb.HELD_OUT)


def test_game_split_takes_the_top_15_of_the_roster():
    scores = {g: float(i) for i, g in enumerate(sorted(set(rb.TRAIN) | set(rb.HELD_OUT) - set(rb.LEFT_OUT)))}
    s = rb.game_split(scores)
    assert len(s["train"]) == 15 and s["held_out"][-5:] == list(rb.LEFT_OUT)
    assert min(scores[g] for g in s["train"]) > max(scores[g] for g in s["held_out"][:5])


def test_settings_flag():
    assert rb.settings({}) is None and rb.settings({"redesign": False}) is None
    assert rb.settings({"redesign": True})["group_size"] == 12
    assert rb.settings({"redesign": {"backlog": {"retire_age": 3}}})["retire_age"] == 3
    with pytest.raises(ValueError):
        rb.settings({"redesign": {"backlog": {"nope": 1}}})


# ---- historical rows (trimmed real run: tn36 won, lf52 and s5i5 at the wall) ----

def test_extract_run_on_real_logs():
    got = rb.extract_run(FIX / "run", rb.load_baselines(POINTS), run="v85p")
    by = {}
    for r in got["rows"]:
        by.setdefault(r["instance"], []).append(r)
    tn = by["k000"]  # won all 7 levels
    assert [r["actions"] for r in tn] == [11, 34, 9, 18, 20, 49, 69]
    assert all(r["cleared"] and not r["censored"] for r in tn)
    assert sum(r["tokens"] for r in tn) == 224695  # every completion token lands on a level
    assert math.isclose(tn[0]["rhae"], 1.15)  # (32 / 11)^2 capped
    assert math.isclose(tn[1]["rhae"], round((72 / 34) ** 2, 6) if (72 / 34) ** 2 < 1.15 else 1.15)
    s5 = by["k076"]  # 3 clears, then the wall on L4
    assert [r["actions"] for r in s5[:3]] == [23, 33, 41]
    assert s5[-1]["level"] == 4 and s5[-1]["censored"] and not s5[-1]["cleared"]
    study = json.loads((FIX / "run" / "study.json").read_text())["trials"][0]["report"]["games"]
    g76 = next(g for g in study if g["game_id"] == "k076")
    assert s5[-1]["tokens"] == g76["completion_tokens"] - g76["time_to_level"][-1]["budget_seconds"] > 0
    lf = by["k001"]  # cleared L5 at the very end: the last clear's observation came from the action count
    assert [r["level"] for r in lf if r["cleared"]] == [1, 2, 3, 4, 5]
    assert lf[4]["actions"] == 401 - sum(r["actions"] for r in lf[:4])
    assert lf[-1]["censored"] and lf[-1]["tokens"] == 0


# ---- metric, ranking, budgets on real rows ----

def test_rhae_tok_and_spread():
    assert rb.rhae_tok(try_(False, 5000)) == 0.0
    assert rb.rhae_tok(try_(True, 4000, 0.5)) == 0.5 / 4000
    vals = [0.5 / 4000, 0.0, 1.0 / 2000]
    assert math.isclose(rb.spread([try_(True, 4000, 0.5), try_(False, 9), try_(True, 2000)]),
                        (sum((v - sum(vals) / 3) ** 2 for v in vals) / 3) ** 0.5)


def test_quantile_matches_numpy_linear():
    assert rb.quantile([1, 2, 3, 4, 5], 0.8) == pytest.approx(4.2)
    assert rb.quantile([10], 0.8) == 10 and rb.quantile([10, 20], 0.8) == pytest.approx(18)


def test_historical_budget_and_flags():
    cur = rb.Curriculum(ROWS)
    toks = sorted(r["tokens"] for r in ROWS if r["game"] == "sb26" and r["level"] == 1 and r["cleared"])
    assert len(toks) == 5
    assert cur.budget("sb26", 1) == math.ceil(1.25 * rb.quantile(toks, 0.8))
    flagged = {(r["game"], r["level"]) for r in cur.table(["s5i5", "su15", "ka59", "sb26"]) if r["flag"]}
    assert {("s5i5", 4), ("s5i5", 5), ("s5i5", 6), ("s5i5", 7), ("ka59", 7), ("su15", 5)} <= flagged
    assert not any(g == "sb26" for g, _ in flagged)
    # 1 clear: 1.25 x that clear
    one = [r["tokens"] for r in ROWS if r["game"] == "s5i5" and r["level"] == 5 and r["cleared"]]
    assert cur.budget("s5i5", 5) == math.ceil(1.25 * one[0])
    # 0 clears: the game's median level budget x the level index
    own = [cur.hist_budget("ka59", k) for k in range(1, 7)]
    assert cur.budget("ka59", 7) == min(300_000, math.ceil(statistics.median(own) * 7))


def test_censored_try_counts_only_past_the_budget():
    cur = rb.Curriculum(ROWS)
    b = cur.hist_budget("s5i5", 4)
    rows = [r for r in ROWS if r["game"] == "s5i5" and r["level"] == 4]
    kept = cur.hist_tries("s5i5", 4)
    assert len(kept) == sum(1 for r in rows if not r["censored"] or r["tokens"] >= b)


def test_ranking_is_spread_times_level_then_freshest_group():
    cur = rb.Curriculum(ROWS, {"rank": "abs"})  # the 26 Oct rule
    hist = cur.hist_tries("sb26", 6)
    assert cur.score("sb26", 6) == pytest.approx(rb.spread(hist) * 6)
    cur.record_group("sb26", 6, [try_(True, 1000)] * 12)  # no spread
    assert cur.score("sb26", 6) == 0.0
    g2 = [try_(True, 1000)] * 6 + [try_(False, 9000)] * 6
    cur.record_group("sb26", 6, g2)
    assert cur.score("sb26", 6) == pytest.approx(rb.spread(g2) * 6)  # freshest only
    assert cur.score("zz99", 1) is None and cur.rank_key("zz99", 1) == math.inf


def test_fresh_budgets_switch_and_smooth_over_two_groups():
    cur = rb.Curriculum(ROWS)
    h = cur.budget("sb26", 1)
    cur.record_group("sb26", 1, [try_(True, 1000)] * 12)  # >= 3 historical clears: one fresh group switches
    assert cur.budget("sb26", 1) == math.ceil(1.25 * 1000)
    cur.record_group("sb26", 1, [try_(True, 3000)] * 12)
    assert cur.budget("sb26", 1) == math.ceil(1.25 * rb.quantile([1000] * 12 + [3000] * 12, 0.8))
    cur.record_group("sb26", 1, [try_(True, 500)] * 12)  # the oldest group drops out
    assert cur.budget("sb26", 1) == math.ceil(1.25 * rb.quantile([3000] * 12 + [500] * 12, 0.8))
    assert h != cur.budget("sb26", 1)
    # sparse level (1 historical clear): historical rule until 2 fresh groups
    hs = cur.budget("s5i5", 5)
    cur.record_group("s5i5", 5, [try_(True, 100)] * 12)
    assert cur.budget("s5i5", 5) == hs
    cur.record_group("s5i5", 5, [try_(True, 200)] * 12)
    assert cur.budget("s5i5", 5) == math.ceil(1.25 * rb.quantile([100] * 12 + [200] * 12, 0.8))
    # fresh groups without clears fall back to the historical rule
    cur.record_group("su15", 1, [try_(False, 9)] * 12)
    assert cur.budget("su15", 1) == math.ceil(cur.hist_budget("su15", 1))
    assert rb.Curriculum(ROWS, {"budget_ceiling": 50_000}).budget("ka59", 7) == 50_000
    # 300K tokens per attempt caps every budget by default, the no-clear fallback included
    free = rb.Curriculum(ROWS, {"budget_ceiling": None})
    over = [(g, k) for (g, k) in list(free.hist) if (free.budget(g, k) or 0) > 300_000]
    assert over and all(rb.Curriculum(ROWS).budget(g, k) == 300_000 for g, k in over)


# ---- backlog ----

GAMES = ["sb26", "s5i5", "ka59", "su15"]


def backlog(streams=50, **cfg):
    return rb.Backlog(GAMES, rb.Curriculum(ROWS, cfg), streams, cfg)


def finish(b, grp, clear_until=0, version=0):
    """End every try of ``grp``: the first ``clear_until`` reach the next level."""
    for i in range(grp["tries"]):
        if i < clear_until:
            b.on_level_reached(grp["id"], i, state={"s": i}, context={"c": i}, version=version, tokens=1000 + i,
                               actions=10, rhae=0.5)
        else:
            b.on_try_done(grp["id"], i, cleared=False, tokens=5000)
    return grp["id"] not in b.groups


def test_phase_opens_with_distinct_best_l1_groups():
    b = backlog(50)
    gs = b.start_phase()
    assert len(gs) == 4 and b.parked == 2
    assert all(g["level"] == 1 and g["tries"] == 12 for g in gs)
    assert len({g["game"] for g in gs}) == 4
    cur = b.cur
    assert [g["game"] for g in gs] == sorted(GAMES, key=lambda x: -cur.score(x, 1))
    assert gs[0]["budget"] == cur.budget(gs[0]["game"], 1)


def test_reached_level_saves_a_start_and_a_group_opens_at_12_parked():
    b = backlog(50)
    gs = b.start_phase()
    sb = next(g for g in gs if g["game"] == "sb26")
    for i in range(9):
        sid = b.on_level_reached(sb["id"], i, state={"board": i}, context={"msgs": i}, version=0, tokens=900,
                                 actions=12, rhae=0.9)
        assert b.starts[sid]["level"] == 2 and b.starts[sid]["state"] == {"board": i}
        assert b.next_group() is None  # 2 + i + 1 <= 11 parked
    assert b.parked == 11 and b.next_group() is None  # 11: no room for a group
    b.on_try_done(sb["id"], 9, cleared=False, tokens=6000)
    g = b.next_group()  # exactly 12 free: a group opens
    assert g is not None and g["game"] == "sb26" and g["level"] == 2 and b.parked == 0
    assert g["start"]["id"] not in b.starts  # used once
    assert b.on_try_done(sb["id"], 9, cleared=False, tokens=1) is None  # repeat ignored
    assert b.parked == 0


def test_admitted_streams_stay_within_a_group_of_the_cap():
    """Scheduler sim: 400 streams, tries of 300-1,500 s, tries end one by one (a group's tries finish
    at different times); admitted streams within [cap - 12, cap] for >= 95% of each phase's time; the phase ends at 16
    full groups; carried groups' tries hold their streams into the next phase and count there when they fill."""
    import heapq

    b = rb.Backlog(GAMES, rb.Curriculum(ROWS), 400, {"phase_groups": 16})
    rng = random.Random(7)
    events, t = [], 0.0

    def launch(grp, now):
        for i in range(grp["tries"]):
            heapq.heappush(events, (now + rng.uniform(300, 1500), grp["id"], i))

    for phase in range(3):
        for g in b.start_phase():
            launch(g, t)
        began, last, inside, admitted = t, t, 0.0, None
        while not b.phase_done:
            now, gid, i = heapq.heappop(events)
            admitted = b.streams - b.parked
            inside += (now - last) * (b.streams - 12 <= admitted <= b.streams)
            last = now
            if rng.random() < 0.3:
                b.on_level_reached(gid, i, state=None, context=None, version=b.version, tokens=900, actions=10, rhae=0.5)
            else:
                b.on_try_done(gid, i, cleared=False, tokens=5000)
            while (g := b.next_group()) is not None:
                launch(g, now)
            assert b.streams - b.parked <= b.streams
        t = last
        assert inside >= 0.95 * (t - began), (phase, inside, t - began)
        out = b.end_phase()
        assert len(out["full"]) == 16 and out["aborted"] == []
        carried = set(out["carried"])
        assert carried == set(b.groups) and all(gid in {e[1] for e in events} for gid in carried)
        b.on_update(phase + 1)
    busy = sum(g["group"]["tries"] - len(g["results"]) for g in b.groups.values())
    assert busy == len(events)  # every carried try still holds its stream


def test_saved_start_ranks_by_its_level_and_l1_stays():
    b = backlog(26, rank="abs")
    gs = b.start_phase()  # 2 groups: sb26, then the next L1
    sb = gs[0]
    for i in range(12):
        b.on_level_reached(sb["id"], i, state=None, context=None, version=0, tokens=900, actions=12, rhae=0.9)
    assert "sb26-L1" in b.starts
    g = b.next_group()  # sb26-L2 (0.498) beats every free L1 (s5i5 0.233, ka59 0.227, su15 0.412 in flight?)
    free_l1 = [x for x in GAMES if x not in {gs[0]["game"], gs[1]["game"]}]
    best = max([("sb26", 2)] + [(x, 1) for x in free_l1] + [("sb26", 1)], key=lambda gl: b.cur.score(*gl))
    assert (g["game"], g["level"]) == best


def test_phase_ends_after_16_full_groups_and_in_flight_starts_return():
    b = backlog(50, carry_inflight=False)
    live = b.start_phase()
    rng = random.Random(0)
    while not b.phase_done:
        assert finish(b, live.pop(0), clear_until=rng.randint(0, 4))
        while (g := b.next_group()) is not None:
            live.append(g)
    assert len(b.full) == 16
    saved_in_flight = [g["start"]["id"] for g in live if not g["start"]["l1"]]
    out = b.end_phase()
    assert len(out["full"]) == 16 and set(out["aborted"]) == {g["id"] for g in live}
    assert all(s in b.starts for s in saved_in_flight)


def test_in_flight_groups_carry_over_the_train_step():
    b = backlog(50)
    live = b.start_phase()
    rng = random.Random(1)
    while not b.phase_done:
        assert finish(b, live.pop(0), clear_until=rng.randint(0, 4))
        while (g := b.next_group()) is not None:
            live.append(g)
    assert live and b.next_group() is None
    half = live[0]
    for i in range(5):  # in flight: some tries end before the switch
        b.on_try_done(half["id"], i, cleared=False, tokens=10)
    out = b.end_phase()
    assert len(out["full"]) == 16 and out["aborted"] == [] and set(out["carried"]) == {g["id"] for g in live}
    b.on_update(1)
    busy = sum(g["tries"] for g in live) - 5
    new = b.start_phase()
    assert b.parked == 50 - busy - 12 * len(new) and 0 <= b.parked < 12
    for i in range(5, 12):
        b.on_try_done(half["id"], i, cleared=False, tokens=10)
    assert half["id"] not in b.groups and b.full[-1]["group"]["id"] == half["id"]  # counts in the new phase


def test_retire_after_two_updates_l1_never():
    b = backlog(50)
    gs = b.start_phase()
    finish(b, gs[0], clear_until=3, version=0)
    saved = [s for s in b.starts.values() if not s["l1"]]
    assert len(saved) == 3
    b.end_phase()
    assert b.on_update(1) == []
    gone = b.on_update(2)
    assert sorted(gone) == sorted(s["id"] for s in saved)
    assert all(f"{g}-L1" in b.starts for g in GAMES)
    # live knob
    b2 = backlog(50)
    finish(b2, b2.start_phase()[0], clear_until=1, version=0)
    assert b2.on_update(2, retire_age=3) == [] and len(b2.on_update(3)) == 1


def test_resume_keeps_the_curriculum_depth():
    """v17 a8 (30 Oct): a resume dropped 29 in-flight groups (their starts returned) and the pending update's version
    bump retired 54 saved starts at once; the next phases opened L1 groups only. Now: an update without a play phase
    ages no start, and the starts a resume returned reopen first, aged from the resume."""
    b = backlog(400)
    gs = b.start_phase()
    finish(b, gs[0], clear_until=3, version=0)  # 3 saved deeper starts, v0
    saved = sorted(s["id"] for s in b.starts.values() if not s["l1"])
    launched = [b.next_group() for _ in range(6)]
    deep = next(g for g in launched if g and g["level"] > 1)  # a saved start in flight, then the container dies
    saved = [sid for sid in saved if sid in b.starts] + [deep["start"]["id"]]
    saved.sort()
    b.on_update(1)  # a played update: ages as before
    returned = b.abandon_inflight()  # the resume
    assert deep["id"] in returned and b.starts[deep["start"]["id"]]["resumed"]
    assert b.on_update(2, played=False) == []  # the resume's pending update: no play, no aging
    assert sorted(s["id"] for s in b.starts.values() if not s["l1"]) == saved
    nxt = b.next_group()  # the returned start reopens first, ahead of every ranked level
    assert nxt["start"]["id"] == deep["start"]["id"] and "resumed" not in nxt["start"]
    # aging resumes over played updates: the other v0 starts (aged from 1 after the pending update) retire at 3
    others = [sid for sid in saved if sid != deep["start"]["id"] and sid in b.starts]
    assert b.on_update(2) == [] and sorted(b.on_update(3)) == sorted(others)
    st = json.loads(json.dumps(b.state_dict()))  # the age clock survives a state round trip
    c = backlog(50)
    c.load_state(st)
    assert c.state_dict() == b.state_dict()


def test_state_round_trip_through_json():
    b = backlog(50)
    gs = b.start_phase()
    finish(b, gs[0], clear_until=2)
    b.on_try_done(gs[1]["id"], 0, cleared=True, tokens=777, rhae=0.3)
    st = json.loads(json.dumps(b.state_dict()))
    c = backlog(50)
    c.load_state(st)
    assert c.state_dict() == b.state_dict()
    assert c.on_try_done(gs[1]["id"], 0, cleared=False, tokens=1) is None  # already recorded
    assert c.cur.score(gs[0]["game"], 1) == b.cur.score(gs[0]["game"], 1)


def test_l1_reused_only_when_nothing_else_is_free():
    b = rb.Backlog(["sb26"], rb.Curriculum(ROWS), 40)
    gs = b.start_phase()  # one game, 3 groups fit: sb26-L1 three times
    assert [g["level"] for g in gs] == [1, 1, 1]


def test_level_inflight_cap_spreads_saved_starts_over_levels():
    for cap, want in ((None, 2), (1, 1)):
        b = backlog(38, level_inflight=cap)
        gs = b.start_phase()  # 3 groups
        sb = next(g for g in gs if g["game"] == "sb26")
        for i in range(12):
            b.on_level_reached(sb["id"], i, state=None, context=None, version=0, tokens=900, actions=12, rhae=0.9)
        finish(b, next(g for g in gs if g["game"] != "sb26"))  # 26 parked: room for two groups
        b.cur.hist = {k: v for k, v in b.cur.hist.items() if k != ("sb26", 2)}  # sb26-L2 unknown: ranks first
        got = [b.next_group(), b.next_group()]
        assert sum(1 for g in got if g and (g["game"], g["level"]) == ("sb26", 2)) == want


def test_sibling_starts_tie_break_by_a_seeded_draw_that_survives_resume():
    """among saved starts of one (game, level) the pick is random (seeded), not the max id; the RNG
    state rides in the backlog state so a resume continues the same draws."""
    def picks(seed, cut=None):
        b = backlog(50, seed=seed)
        gs = b.start_phase()
        finish(b, gs[0], clear_until=6)  # 6 sibling L2 starts of one game
        sibs = sorted(s["id"] for s in b.starts.values() if not s["l1"])
        order = sorted(sibs, key=lambda i: b.starts[i]["tie"], reverse=True)
        if cut is not None:  # resume from a JSON round trip: the next draws continue the same stream
            c = backlog(50, seed=seed)
            c.load_state(json.loads(json.dumps(b.state_dict())))
            return order, b.rng.random() == c.rng.random()
        return order, None

    a, _ = picks(0)
    assert a == picks(0)[0]  # reproducible
    assert a != sorted(a, reverse=True) or picks(1)[0] != a  # not the max-id order for every seed
    assert len({tuple(picks(s)[0]) for s in range(6)}) > 1  # the seed changes the order
    assert picks(0, cut=True)[1]
    old = backlog(50)
    st = old.state_dict()
    st.pop("rng")
    for s in st["starts"]:
        s.pop("tie")
    new = backlog(50)
    new.load_state(json.loads(json.dumps(st)))  # a pre-draw state (T4 leg 1) loads: ties drawn on resume
    assert all("tie" in s for s in new.starts.values())


def test_seed_starts_stay_after_use_and_budget_override():
    """G-long (27 Oct): recorded starts stay after use (several groups from one deep start); every group's budget is
    the override."""
    seeds = [{"game": "sb26", "level": 3, "attempt": "/runs/x/a000", "request": 41},
             {"game": "zz99", "level": 4, "attempt": "/runs/y", "request": 9}]  # not a roster game: skipped
    b = backlog(60, seed_starts=seeds, budget_override=300000)
    sid = "sb26-L3-seed0"
    assert sid in b.starts and "zz99-L4-seed1" not in b.starts
    assert b.starts[sid]["state"] == {"attempt": "/runs/x/a000", "request": 41} and b.starts[sid]["keep"]
    gs = b.start_phase()
    assert all(g["budget"] == 300000 for g in gs)
    used = [g for g in gs if g["start"]["id"] == sid]
    assert len(used) >= 2 and sid in b.starts  # used by several groups, still there


def test_rel_ranking_is_pstdev_over_median_times_level():
    """spread / median of RHAE/tok (pstdev alone ranked levels cleared in few tokens first)."""
    cheap = [try_(True, t) for t in (1000, 1200, 1400)] * 4
    dear = [try_(True, t) for t in (10000, 30000)] * 6
    cur = rb.Curriculum(ROWS)
    cur.record_group("sb26", 2, cheap)
    cur.record_group("su15", 2, dear)
    v = [1 / t for t in (1000, 1200, 1400)] * 4
    assert cur.score("sb26", 2) == pytest.approx(statistics.pstdev(v) / statistics.median(v) * 2)
    assert cur.score("su15", 2) > cur.score("sb26", 2)  # abs ranks them the other way
    old = rb.Curriculum(ROWS, {"rank": "abs"})
    old.record_group("sb26", 2, cheap)
    old.record_group("su15", 2, dear)
    assert old.score("sb26", 2) > old.score("su15", 2)


def test_rel_spread_floors_a_zero_median_and_scores_all_fail_zero():
    mostly_fail = [try_(True, 1000)] * 3 + [try_(False, 9000)] * 9  # median 0
    v = [1 / 1000] * 3 + [0.0] * 9
    assert rb.rel_spread(mostly_fail) == pytest.approx(statistics.pstdev(v) / (0.5 * statistics.fmean(v)))
    assert rb.rel_spread([try_(False, 9000)] * 12) == 0.0
    assert rb.rel_spread([try_(True, 1000)]) == 0.0


def test_level_inflight_holds_saved_starts_back_even_when_every_l1_is_in_flight():
    """28 Oct: the all-in-flight fallback returned every start with the fewest groups, held-back saved starts too."""
    b = backlog(200, level_inflight=1)
    gs = b.start_phase()  # every L1 in flight (the fallback opens L1s again)
    sb = next(g for g in gs if g["game"] == "sb26")
    for i in range(12):
        b.on_level_reached(sb["id"], i, state=None, context=None, version=0, tokens=900, actions=12, rhae=0.9)
    g = b.next_group()
    assert g["level"] == 2  # the first L2 opens
    b.parked += 24  # two more groups' streams free
    nxt = [b.next_group(), b.next_group()]
    assert [x["level"] for x in nxt] == [1, 1]  # the other L2 starts stay held back


def test_rel_ranking_puts_levels_with_under_5_tries_behind_ranked_ones():
    cur = rb.Curriculum(ROWS)
    cur.record_group("sb26", 3, [try_(True, 1000), try_(True, 3000), try_(False, 9000)])  # 3 tries: unranked
    cur.record_group("sb26", 2, [try_(True, t) for t in (1000, 1100, 1200, 1300, 1400)])  # 5: ranked
    assert cur.rank_key("sb26", 3) == -math.inf and cur.score("sb26", 3) > 0
    assert cur.rank_key("sb26", 2) == pytest.approx(cur.score("sb26", 2)) and cur.rank_key("sb26", 2) > -math.inf
    assert cur.rank_key("zz99", 1) == math.inf  # no tries at all: unknown_first as before


def test_level_starts_keeps_the_newest_two_saved_starts_per_level():
    """the backlog stores at most 2 starts per level; L1 starts are not saved starts."""
    b = backlog(200, level_starts=2)
    gs = b.start_phase()
    sb = next(g for g in gs if g["game"] == "sb26")
    sids = [b.on_level_reached(sb["id"], i, state=None, context=None, version=0, tokens=900, actions=12, rhae=0.9)
            for i in range(5)]
    kept = [s for s in b.starts.values() if s["game"] == "sb26" and s["level"] == 2]
    assert [s["id"] for s in kept] == sids[-2:] and b.stats["capped"] == 3
    assert "sb26-L1" in b.starts


def test_budget_floor_lifts_fresh_budgets_to_the_v85p_p75_and_historical_ones_to_p90():
    """off by default; on, a fresh budget below the level's v8.5p p75 tokens-to-clear is raised to
    it (U8 cd82-L3: 34.4K vs p50 38.5K, 0/8), the historical rule floors at p90; never above the ceiling."""
    hist = [{"game": "aa11", "level": 2, "cleared": True, "tokens": t, "actions": 5} for t in (100, 200, 300, 400, 500)]
    cur = rb.Curriculum(hist)
    fresh = [{"cleared": True, "tokens": 40, "rhae": 1.0}] * 8
    cur.record_group("aa11", 2, fresh)
    assert cur.budget("aa11", 2) == 50  # 1.25 x p80 of fresh clears
    cur.cfg["budget_floor"] = True
    assert cur.budget("aa11", 2) == 400  # v8.5p p75
    assert cur.budget("aa11", 3) is not None
    cur2 = rb.Curriculum(hist, {"budget_floor": True})
    assert cur2.budget("aa11", 2) == 525  # historical 1.25 x p80 (525) is above p90 (460)
    cur3 = rb.Curriculum(hist, {"budget_floor": True, "budget_mult": 0.5})
    assert cur3.budget("aa11", 2) == 460  # historical rule below p90: floored at p90
    cur4 = rb.Curriculum(hist, {"budget_floor": True, "budget_ceiling": 300})
    cur4.record_group("aa11", 2, fresh)
    assert cur4.budget("aa11", 2) == 300  # the ceiling still caps


def test_budget_guard_owner_examples():
    """a new budget never cuts more than 20% of the window's tries, cut tries valued at their budget.
    (1) <= 20% cut at b = 60K and the next 10% at 50K: a new budget of 50K is allowed; (2) > 20% cut at 60K: it holds."""
    cut = {"cleared": False, "tokens": 60500, "budget": 60000}  # a cut try overshoots its budget to the request's end
    clear = lambda t: {"cleared": True, "tokens": t, "budget": 60000, "rhae": 1.0}  # noqa: E731
    one = [clear(t) for t in (20000, 24000, 28000, 30000, 34000, 38000, 40000)] + [clear(50000), cut, cut]
    assert rb.Curriculum.guard([one]) == 50000
    two = [clear(t) for t in (20000, 24000, 28000, 30000, 34000, 38000, 40000)] + [cut, cut, cut]
    assert rb.Curriculum.guard([two]) == 60000
    hist = [{"game": "aa11", "level": 2, "cleared": True, "tokens": t, "actions": 5} for t in (100, 200, 300, 400, 500)]
    # the rule alone is 1.25 x p80 of the clears: 49K (one) and 46.5K (two); the guard lifts them to 50K / 60K
    for tries, off, on in ((one, 49000, 50000), (two, 46500, 60000)):
        cur = rb.Curriculum(hist)
        cur.record_group("aa11", 2, tries)
        assert cur.budget("aa11", 2) == off
        cur.cfg["budget_guard"] = True
        assert cur.budget("aa11", 2) == on
    # the old rule wins when it is above the guard; an early fail (not at its budget) counts at its tokens; tries the
    # hook never saw (0 tokens) and rows from before budgets were kept are handled
    cur = rb.Curriculum(hist, {"budget_guard": True})
    cur.record_group("aa11", 2, [clear(50000)] * 8)
    assert cur.budget("aa11", 2) == 62500
    early = [clear(10000), {"cleared": False, "tokens": 30000, "budget": 60000}, {"cleared": False, "tokens": 0}]
    assert rb.Curriculum.guard([early]) == 30000
    assert rb.Curriculum.guard([[{"cleared": False, "tokens": 700}]]) == 700 and rb.Curriculum.guard([[]]) is None


def test_backlog_records_each_trys_group_budget():
    b = rb.Backlog.__new__(rb.Backlog)
    recorded = []
    b.groups = {"g1": {"group": {"game": "aa11", "level": 2, "tries": 1, "budget": 1234}, "results": {}}}
    b.parked, b.full = 0, []
    b.cur = type("C", (), {"record_group": lambda self, g, k, t: recorded.append(t)})()
    b.on_try_done("g1", 0, cleared=False, tokens=1300)
    assert recorded == [[{"cleared": False, "tokens": 1300, "actions": 0, "rhae": 0.0, "budget": 1234}]]


def test_drop_failed_records_nothing_and_frees_the_streams():
    """v17 30 Oct: a bench that raised before any try (deleted start folder) is no curriculum fail."""
    b = rb.Backlog.__new__(rb.Backlog)
    recorded = []
    b.groups = {"g1": {"group": {"game": "aa11", "level": 2, "tries": 8, "budget": 1234}, "results": {}}}
    b.parked, b.full, b.stats = 0, [], {}
    b.cur = type("C", (), {"record_group": lambda self, g, k, t: recorded.append(t)})()
    assert b.drop_failed("g1")["game"] == "aa11"
    assert recorded == [] and b.groups == {} and b.parked == 8 and b.stats["failed_starts"] == 1
    assert b.drop_failed("g1") is None
    b.on_try_done("g1", 0, cleared=False, tokens=0)  # the later fill loop finds no group: nothing recorded
    assert recorded == []


def test_budget_censored_counts_cuts_at_their_budget_and_rises_at_most_1_25x():
    """off by default; on, 3 of 8 cut at 60K put the p80 at 60K: 1.25 x 60K = 75K (the growth cap); few cuts: the
    p80 over all tries, cuts at the budget."""
    hist = [{"game": "aa11", "level": 2, "cleared": True, "tokens": t, "actions": 5} for t in (100, 200, 300, 400, 500)]
    cut = {"cleared": False, "tokens": 60500, "budget": 60000}
    clear = lambda t: {"cleared": True, "tokens": t, "budget": 60000, "rhae": 1.0}  # noqa: E731
    cur = rb.Curriculum(hist, {"budget_censored": True})
    cur.record_group("aa11", 2, [clear(t) for t in (20000, 24000, 28000, 30000, 34000)] + [cut] * 3)
    assert cur.budget("aa11", 2) == 75000
    cur = rb.Curriculum(hist, {"budget_censored": True})
    cur.record_group("aa11", 2, [clear(t) for t in (10000, 12000, 14000, 16000, 18000, 20000, 22000)] + [cut])
    assert cur.budget("aa11", 2) == 1.25 * (20000 + 0.6 * 2000)  # p80 of all 8 (the cut at 60K is above it)


# ---- update mix ----

def full_group(b, game, tokens):
    """A full group of ``game`` straight into ``full`` (finish order = call order)."""
    b.counter += 1
    grp = {"id": f"x{b.counter}", "start": b.starts[f"{game}-L1"], "game": game, "level": 1, "tries": 2,
           "budget": 1, "version": 0}
    b.groups[grp["id"]] = {"group": grp, "results": {}}
    for i in range(2):
        b.on_try_done(grp["id"], i, cleared=False, tokens=tokens // 2)
    return grp["id"]


def test_update_mix_off_keeps_the_first_phase_groups():
    b = backlog(50, phase_groups=3)
    ids = [full_group(b, "sb26", 100) for _ in range(5)]
    assert b.phase_done and [fg["group"]["id"] for fg in b.end_phase()["full"]] == ids[:3]
    assert [fg["group"]["id"] for fg in b.full] == ids[3:]


def test_game_cap_skips_a_capped_games_later_groups_until_the_next_update():
    b = backlog(50, phase_groups=3, game_cap=1)
    s1, s2, k1 = full_group(b, "sb26", 100), full_group(b, "sb26", 100), full_group(b, "ka59", 100)
    assert not b.phase_done  # 2 picked: sb26 once, ka59 once
    s3 = full_group(b, "sb26", 100)
    assert not b.phase_done
    u1 = full_group(b, "su15", 100)
    assert b.phase_done
    out = b.end_phase()
    assert [fg["group"]["id"] for fg in out["full"]] == [s1, k1, u1]
    assert [fg["group"]["id"] for fg in b.full] == [s2, s3]  # wait for the next update, oldest first
    assert [i for i in b.update_pick()[0]] == [0]  # next update: one sb26 again


def test_game_cap_holds_launches_of_a_game_with_twice_its_cap_untrained():
    b = backlog(50, game_cap=1)
    full_group(b, "sb26", 100)
    assert "sb26" not in b.held_games()
    full_group(b, "sb26", 100)
    assert b.held_games() == {"sb26"}
    assert all(s["game"] != "sb26" for s in b._available())
    b.cfg["game_cap"] = None
    assert any(s["game"] == "sb26" for s in b._available())


def test_min_update_tokens_waits_for_tokens_up_to_max_update_groups():
    b = backlog(50, phase_groups=2, min_update_tokens=1000)
    full_group(b, "sb26", 100)
    full_group(b, "ka59", 100)
    assert not b.phase_done  # 2 groups but 200 tokens
    full_group(b, "su15", 900)
    assert b.phase_done and len(b.end_phase()["full"]) == 3
    b.cfg["max_update_groups"] = 3
    for g in ("sb26", "ka59"):
        full_group(b, g, 10)
    assert not b.phase_done
    full_group(b, "s5i5", 10)
    assert b.phase_done and len(b.end_phase()["full"]) == 3  # the ceiling ends it short of the tokens


def test_update_mix_survives_a_state_round_trip():
    b = backlog(50, phase_groups=2, game_cap=1)
    full_group(b, "sb26", 100)
    full_group(b, "sb26", 100)
    c = backlog(50, phase_groups=2)
    c.load_state(json.loads(json.dumps(b.state_dict())))
    assert c.cfg["game_cap"] == 1 and not c.phase_done and c.held_games() == {"sb26"}
