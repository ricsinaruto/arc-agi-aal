"""Chains (explorers dropped) and the whole-game held-out read."""
import asyncio
import json
import types

import pytest

from arena import v7_envs
from arena.async_rl import Book
from arena.curriculum import Curriculum
from arena.v7_envs import EnvsV7Mixin, game_score, ratio_ci


def cur_with(units, **cfg):
    c = Curriculum({"group_size": 12, "score": "rhae_tok", "chain_groups": 2, "family_cap": 0.5, "window": 12, **cfg},
                   {"public": 1.0})
    for game, level, fam in units:
        c.add_unit(game, level, "public", fam, cap=64000)
        c.add_start({"game": game, "level": level, "source": "public", "start": {"attempt": f"/a/{game}{level}", "request": 1}})
    return c


def test_plan_chain_bonus_family_cap_and_sure_quota():
    c = cur_with([("aa01", 2, "a"), ("aa01", 3, "a"), ("bb02", 2, "b")])
    p = c.plan_chain(1, {"aa01@L2": {"deepest": False}, "aa01@L3": {"deepest": True}}, seed=0)
    assert p["why"] == "chain" and p["unit"] == "aa01@L3"  # same prior: the deepest open level's bonus decides
    assert c.made["public"] == 1 and c.recent[-1][0] == "aa01@L3"
    u = c.units["bb02@L2"]
    u.a, u.b = 99.0, 1.0  # near-certain, no trend
    assert c.sure(u)
    assert c.plan_chain(1, {"bb02@L2": {"deepest": True}}, sure_ok=False) is None
    assert c.plan_chain(1, {"bb02@L2": {"deepest": True}}, sure_ok=True)["sure"] is True
    c.cfg["family_cap"] = 1 / 12  # one group per family per window: family a is full
    assert c.plan_chain(1, {"aa01@L2": {"deepest": True}}) is None
    assert c.plan_chain(1, {}) is None


def test_game_score_matches_the_competition_calculator():
    from arc_agi.scorecard import EnvironmentScoreCalculator

    for cleared, total, base, acts in (({1: 100.0, 2: 25.0}, 4, [10, 10, 10, 10], [10, 20, 0, 0]),
                                       ({}, 3, [5, 5, 5], [0, 0, 0]),
                                       ({1: 115.0, 2: 115.0, 3: 115.0}, 3, [10, 10, 10], [9, 9, 9])):
        calc = EnvironmentScoreCalculator()
        for i in range(total):
            done = (i + 1) in cleared
            calc.add_level(i + 1, done, acts[i], base[i])
        ref = calc.to_score().score
        mine = game_score({k: min(115.0, 100.0 * (base[k - 1] / acts[k - 1]) ** 2) for k in cleared}, total)
        assert mine == pytest.approx(ref)


def test_ratio_ci():
    r = ratio_ci(2.0, 0.2, 1.0, 0.1)
    assert r[0] == 2.0 and r[1] < 2.0 < r[2]
    assert ratio_ci(1.0, 0.1, 0.0, 0.1) is None


class Feed:
    def __init__(self):
        self.rows = []

    def emit(self, kind, **kw):
        self.rows.append({"kind": kind, **kw})


class Loop(EnvsV7Mixin):
    def __init__(self, cur, sets, rows):
        self.mg_curr = cur
        self.cfg = {"mg": {"cap": 64000, "harvest": {"max_uses": 2, "stale": 4}, "curriculum": {}}, "mg_sets": sets}
        self.mg_unit_set = {k: k.replace("@", "-") for k in cur.units}
        self.v7 = {"book": Book(3), "made": {"by_set": {}}, "cfg": {}}
        self.v7_cfg_now = {"s_max": 2}
        self.mg_harvest, self.mg_harvest_uses = rows, {}
        self.v7e = {"active": ["aa01", "bb02"], "groups_by_source": {}, "tokens_train": 0, "chain": {}, "gmeta": {},
                    "held_starts": {}, "src_count": {}, "chain_log": []}
        self.feed = Feed()


def sets_for(units):
    return {f"{g}-L{lv}": {"name": f"{g}-L{lv}", "game": g, "level": lv, "role": "train", "cap": 64000,
                           **({"fresh": True} if lv == 1 else {})} for g, lv, _ in units}


def row(i, game, level, tokens=0, played_by=0):
    return {"id": f"r{i}", "target": f"{game}-L{level}", "game": game, "level": level, "attempt": f"/h/{i}",
            "request": 3, "played_by": played_by, "source": "harvest", "chain_tokens": tokens}


def test_chain_pick_one_per_game_quota_budget_and_step_reset():
    units = [("aa01", 1, "a"), ("aa01", 2, "a"), ("aa01", 3, "a"), ("bb02", 1, "b"), ("bb02", 2, "b")]
    cur = cur_with(units)
    loop = Loop(cur, sets_for(units), [row(1, "aa01", 2), row(2, "aa01", 3, tokens=200000), row(3, "bb02", 2),
                                       row(4, "aa01", 3, tokens=250000)])
    sets = loop.cfg["mg_sets"]
    a = loop.v7e_pick(sets, {})
    b = loop.v7e_pick(sets, {})
    games = {a[0][:4], b[0][:4]}
    assert games == {"aa01", "bb02"}  # one chain group per game per step, 2 per step (chain_groups)
    assert loop.v7e["src_count"]["chain"] == 2
    assert a[0] != "aa01-L3" or a[1][0] == "/h/2"  # 250,000 + 64,000 > 300,000: r4 never a chain start
    c = loop.v7e_pick(sets, {})
    assert loop.v7e["src_count"].get("chain") == 2 and c[0] is not None  # quota spent: the curriculum's pick
    loop.v7["book"].current = 1
    loop.mg_harvest_uses.clear()
    loop.v7e_pick(sets, {})
    assert loop.v7e["src_count"]["chain"] == 3  # a new step: chains again


def test_stuck_rule_closes_and_a_clear_reopens():
    units = [("aa01", 1, "a"), ("aa01", 2, "a")]
    loop = Loop(cur_with(units), sets_for(units), [])
    g = types.SimpleNamespace(gid="g1", unit="aa01-L2", version=0)
    spec = {"name": "aa01-L2", "game": "aa01", "level": 2, "role": "train"}
    fails = [{"outcome": "budget", "completion_tokens": 10}] * 12
    for gid in ("g1", "g2"):
        loop.v7e["gmeta"][gid] = {"chain": True, "unit": "aa01@L2", "tokens": 0}
        asyncio.run(loop.v7e_group_done(types.SimpleNamespace(gid=gid, unit="aa01-L2", version=0), spec, "/f", fails))
    assert "aa01@L2" in loop.v7e["chain"]["closed"]
    loop.v7e["gmeta"]["g3"] = {"chain": False, "unit": "aa01@L2", "tokens": 0}
    asyncio.run(loop.v7e_group_done(types.SimpleNamespace(gid="g3", unit="aa01-L2", version=0), spec, "/f",
                                    [{"outcome": "cleared", "completion_tokens": 10}] + fails[:11]))
    assert "aa01@L2" not in loop.v7e["chain"]["closed"]


def test_whole_game_read_segments_score_and_held_starts(monkeypatch):
    sets = {f"tr87-L{lv}": {"name": f"tr87-L{lv}", "game": "tr87", "level": lv, "role": "test", "cap": 64000,
                            **({"read": True} if lv == 1 else {"read": "screen"})} for lv in (1, 2, 3)}
    loop = Loop(cur_with([]), sets, [])
    loop.cfg["mg"]["wread"] = {"every": 8, "budget": 150000, "seg_tokens": 65536}
    loop.v7e_root = lambda: __import__("pathlib").Path("/tmp/wread-test")
    plays = iter([
        {"label": "a", "outcome": "cleared", "completion_tokens": 40000, "total_levels": 3,
         "level_rows": [{"score": 80.0}]},
        {"label": "b", "outcome": "budget", "completion_tokens": 65536, "total_levels": 3},
        {"label": "c", "outcome": "cleared", "completion_tokens": 20000, "total_levels": 3,
         "level_rows": [{"score": 50.0}]},
        {"label": "d", "outcome": "budget", "completion_tokens": 24464, "total_levels": 3},
    ])
    seen = []

    async def fake_play(name, spec, starts, version, wall=None, floor=True, count=True):
        seen.append((spec["level"], bool(spec.get("fresh")), list(starts), spec["cap"], floor, count))
        return "/f", [next(plays)]

    loop.v7e_explore_play = fake_play
    loop.v7_catch_up = lambda: asyncio.sleep(0)
    monkeypatch.setattr(v7_envs, "cut_clear", lambda att, dest, r, lvl, checks, game: ((dest, 7, 900), None))
    monkeypatch.setattr(v7_envs, "cut_mid", lambda att, dest, after: ((dest, after + 5), None))
    out = asyncio.run(loop.v7e_wread(8, 2))
    g = out["games"]["tr87"]
    assert [s[0] for s in seen] == [1, 2, 2, 3] and seen[0][1] and not seen[1][1]
    assert all(not s[4] and not s[5] for s in seen)  # normal priority, no explored count
    assert g["cleared"] == {"1": 80.0, "2": 50.0} and g["tokens"] == 150000 and g["why"] == "budget"
    assert g["score"] == pytest.approx((80 * 1 + 50 * 2) / 6)
    assert out["score_per_mtok"] == pytest.approx(g["score"] / 0.15)
    assert set(loop.v7e["held_starts"]) == {"tr87-L2", "tr87-L3"}
    assert loop.feed.rows[-1]["kind"] == "wread"
    extra = loop.v7e_wread_extra({"tr87-L1": None})
    assert list(extra) == ["tr87-L2", "tr87-L3"]
    loop.cfg["mg"]["wread"]["extra_sets"] = 1
    assert list(loop.v7e_wread_extra({"tr87-L1": None, "tr87-L2": [("x", 1)]})) == ["tr87-L3"]


def test_wread_preflight_shrink(monkeypatch):
    monkeypatch.setenv("ARC_PREFLIGHT_STUB", '{"shrink": true}')
    w = v7_envs.wread_settings({"wread": {"every": 8}})
    assert w["every"] == 2 and w["budget"] == 32768 and w["seg_tokens"] == 16384
    monkeypatch.delenv("ARC_PREFLIGHT_STUB")
    assert v7_envs.wread_settings({"wread": {"every": 8}})["every"] == 8


def test_resume_without_explorers_restores_chain_state_and_held_starts(tmp_path):
    """hero-v6c D2 leg: a resumed attempt's state has explore None (explorers off); held-out starts cut under the dead
    attempt's wread/ are found in the resume root."""
    units = [("aa01", 1, "a"), ("aa01", 2, "a")]
    loop = Loop(cur_with(units), sets_for(units), [row(1, "aa01", 2, tokens=1234)])
    loop.v7e["chain"] = {"step": 3, "made": 1, "games": ["aa01"], "sure": 0, "closed": {"aa01@L2": 2}, "dry": {}}
    loop.v7e["held_starts"] = {"tr87-L2": [["/runs/validation/x/wread/w0000-tr87-L2", 7]]}
    loop.v7e["wread_base"] = {"score": 1.5, "se": 0.2, "per": 5.0}
    loop.cfg["mg"]["running_caps"] = None
    loop.mg_harvest_uses = {"r1": 1}
    loop.output = str(tmp_path)
    loop.mg_explorers = None
    state = loop.v7e_state()
    assert state["explore"] is None
    root = tmp_path / "lost"
    (root / "wread" / "w0000-tr87-L2").mkdir(parents=True)
    (root / "wread" / "w0000-tr87-L2" / "rollout.jsonl").write_text("{}\n")
    (root / "harvest" / "h1").mkdir(parents=True)
    (root / "harvest" / "h1" / "rollout.jsonl").write_text("{}\n")
    state["harvest"] = [{**row(1, "aa01", 2, tokens=1234), "attempt": "/runs/validation/x/harvest/h1"}]
    fresh = Loop(cur_with(units), sets_for(units), [])
    fresh.v7_resume_root = str(root)
    fresh.output = str(tmp_path)
    asyncio.run(fresh.v7e_setup(json.loads(json.dumps(state))))
    assert fresh.v7e["chain"]["closed"] == {"aa01@L2": 2} and fresh.v7e["wread_base"]["score"] == 1.5
    assert fresh.v7e["held_starts"]["tr87-L2"][0] == [str(root / "wread" / "w0000-tr87-L2"), 7]
    assert fresh.mg_harvest[0]["chain_tokens"] == 1234 and fresh.mg_explorers is None
