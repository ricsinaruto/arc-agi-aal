"""Envs Run 1 data setup on the v7 loop (arena.v7_envs; hero run): explorer chains (mid-level
continuation = the real play's prompt), start pool freshness by version, the per-group curriculum, rotation, the
held-out read's size and share, and the loop state across resume."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import FakeEnv, observation
from test_arena_harvest import GAME, base_point, board, play
from test_level_bench import QuietMonitor

from arena import level_bench, restore, v7_envs
from arena.async_rl import Book
from arena.curriculum import Curriculum
from arena.level_points import point_id


class SlowLadder:
    """Level k cleared after 3k actions (5 levels), a new board each step."""

    def open_scorecard(self, tags):
        return "card"

    def close_scorecard(self, card):
        return None

    def make(self, game_id, **kwargs):
        return FakeEnv([observation(board(k), levels=min(k // 3, 5)) for k in range(60)])


@pytest.fixture
def ladder(monkeypatch):
    import test_arena_harvest

    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    monkeypatch.setattr(test_arena_harvest, "LadderArcade", SlowLadder)


def live_view(payload):
    """The request's conversation and tools, the mock server's call ids (numbered by its request count, so they
    differ between two runs) renamed in order of appearance."""
    import re

    text = json.dumps({"messages": payload["messages"], "tools": payload.get("tools")}, sort_keys=True)
    ids = {}
    return json.loads(re.sub(r'"c\d+"', lambda m: ids.setdefault(m.group(0), f'"id{len(ids)}"'), text))


async def test_mid_level_continuation_sends_the_unsegmented_plays_prompt(tmp_path, ladder):
    """a segment cut after its tokens restores the real play (same conversation, same context
    state); the continued segment's first live request is the request the unsegmented play sent at that point, and
    every later one matches too."""
    full_sent = []
    fresh = base_point(id=f"{point_id(GAME, 0)}-L1-sc", through=False, segment=True, budget=400, harvest=True)
    row, _ = await play(tmp_path, "full", fresh, full_sent)
    assert row["outcome"] == "cleared" and row["actions"] == 3
    seg_sent = []
    row1, att1 = await play(tmp_path, "seg1", {**fresh, "budget": 4}, seg_sent)
    assert row1["outcome"] == "budget", row1
    got, why = v7_envs.cut_mid(att1, tmp_path / "harvest" / "m1", -1)
    assert why is None
    dest, r = got
    assert r >= 1 and restore.load(dest, r)["level"] == 1
    seg2_sent = [{"pad": True}] * 50  # the mock numbers its call ids by request count: keep seg2's apart from seg1's
    cont = base_point(id=f"{point_id(GAME, 0)}-L1-m", level=1, harvest=True,
                      restore={"attempt": str(dest), "request": r, "check_only": False, "budget": 400})
    row2, att2 = await play(tmp_path, "seg2", cont, seg2_sent)
    assert row2["restore"]["ok"], row2["restore"]
    assert row2["outcome"] == "cleared" and row2["actions"] == 3  # actions counted from the level's first board
    gameplay = [p for p in full_sent if p.get("max_tokens") != 1]
    live2 = [p for p in seg2_sent[50:] if p.get("max_tokens") != 1]
    assert live_view(live2[0]) == live_view(gameplay[r])  # the continued prompt = the unsegmented play's
    assert [live_view(p) for p in live2] == [live_view(p) for p in gameplay[r:]]
    # the chained clear is cut as the next level's start (harvest request), restorable at level 2
    got, why = v7_envs.cut_clear(att2, tmp_path / "harvest" / "c2", row2, 2, (SlowLadder(), "card"), GAME)
    assert why is None and restore.load(got[0], got[1])["level"] == 2


async def test_cut_mid_refuses_a_segment_without_new_requests(tmp_path, ladder):
    fresh = base_point(id=f"{point_id(GAME, 0)}-L1-sc", through=False, segment=True, budget=4, harvest=True)
    row1, att1 = await play(tmp_path, "seg1", fresh, [])
    got, why = v7_envs.cut_mid(att1, tmp_path / "harvest" / "m1", -1)
    assert why is None
    assert v7_envs.cut_mid(att1, tmp_path / "harvest" / "m2", got[1]) == (None, "no_progress")


# ---- curriculum per group ----------------------------------------------------------------------------------------

SHARES = {"public": 0.50, "generated": 0.21, "puzzlescript": 0.14, "sonpham": 0.11, "witness": 0.04}


def mix_curriculum(families=6, **cfg):
    cur = Curriculum({"group_size": 12, "family_cap": 0.17, **cfg}, SHARES)
    for src in SHARES:
        for f in range(families):
            for level in (1, 2):
                u = cur.add_unit(f"{src[0]}{src[-1]}{f:02d}", level, src, f"{src}-{f}", clears=3, fails=5, tokens=[10000])
                u.starts = [{"fresh": True}]
    return cur


def test_plan_one_holds_the_source_mix_over_groups():
    cur = mix_curriculum()
    got = [cur.plan_one(k // 4, seed=1)["source"] for k in range(400)]
    for src, share in SHARES.items():
        assert abs(got.count(src) / 400 - share) <= 0.01, (src, got.count(src))
    first = got[:12]  # plan(groups=1) gave every group to the largest share; by deficit the mix shows at once
    assert len(set(first)) >= 4


def test_plan_one_family_cap_over_the_window_and_never_stalls():
    cur = mix_curriculum(families=6)
    picks = [cur.plan_one(1, seed=2) for _ in range(120)]
    for i in range(12, 120):
        fams = [p["family"] for p in picks[i - 12:i]]
        assert max(fams.count(f) for f in set(fams)) <= max(1, int(0.17 * 12))
    one = mix_curriculum(families=1)  # every source has one family: over the cap at once, picked anyway
    got = [one.plan_one(1, seed=3) for _ in range(30)]
    assert all(p is not None for p in got) and {p["why"] for p in got} >= {"over_cap"}


def test_plan_one_excludes_units_without_a_start_and_returns_none_when_nothing_plays():
    cur = mix_curriculum(families=2)
    allowed = {k for k in cur.units if k.endswith("@L1")}
    blocked = set(cur.units) - allowed
    for _ in range(50):
        assert cur.plan_one(1, exclude=blocked)["unit"] in allowed
    assert cur.plan_one(1, exclude=set(cur.units)) is None


def test_plan_one_state_roundtrip_keeps_the_mix_counts():
    cur = mix_curriculum()
    for _ in range(25):
        cur.plan_one(1, seed=4)
    back = Curriculum.load_state(json.loads(json.dumps(cur.state_dict())))
    assert back.made == cur.made and back.recent == cur.recent
    old = cur.state_dict()
    old.pop("made"), old.pop("recent")  # a state written before plan_one
    assert Curriculum.load_state(old).made == {}


# ---- start pool freshness ----------------------------------------------------------------------------------------

def pool_loop(rows, lineage, current=None):
    book = Book(6)
    for v in lineage[1:]:
        book.publish(v)
    loop = SimpleNamespace(cfg={"mg": {"harvest": {"levels": [2], "switch_round": 1, "max_uses": 2, "stale": 4}},
                                "mg_sets": {}},
                           v7={"book": book}, v7_cfg_now={"s_max": 8}, v7e={"expired": 0},
                           mg_harvest=list(rows), mg_harvest_uses={})
    for name in ("v7e_fresh", "v7e_prune"):
        setattr(loop, name, getattr(v7_envs.EnvsV7Mixin, name).__get__(loop))
    return loop


def start_row(i, played_by, target="g-L2"):
    return {"id": f"x{i}", "target": target, "played_by": played_by, "attempt": f"/a/{i}", "request": 3}


def test_starts_fresh_while_at_most_stale_versions_behind_and_under_max_uses():
    loop = pool_loop([start_row(i, v) for i, v in enumerate(range(0, 11))], list(range(0, 11)))
    assert sorted(r["played_by"] for r in loop.v7e_fresh()["g-L2"]) == [6, 7, 8, 9, 10]
    loop.mg_harvest_uses["x10"] = 2
    assert 10 not in [r["played_by"] for r in loop.v7e_fresh()["g-L2"]]


def test_a_rollback_makes_the_dropped_versions_starts_stale_and_the_older_fresh_again():
    loop = pool_loop([start_row(i, v) for i, v in enumerate(range(0, 11))], list(range(0, 11)))
    loop.v7["book"].rollback(5)
    assert sorted(r["played_by"] for r in loop.v7e_fresh()["g-L2"]) == [1, 2, 3, 4, 5]


def test_prune_drops_spent_and_out_of_reach_rows():
    loop = pool_loop([start_row(i, v) for i, v in enumerate(range(0, 21))], list(range(0, 21)))
    loop.mg_harvest_uses["x20"] = 2
    gone = loop.v7e_prune()
    assert gone == 9 and min(r["played_by"] for r in loop.mg_harvest) == 8 and "x20" not in loop.mg_harvest_uses


# ---- the v7 loop with the multi-game pool (stub play, fake trainer) -----------------------------------------------

from test_rl_async_loop import events, v7_loop  # noqa: E402


def mg_sets():
    out = []
    for code, src, reserve in (("pa11", "public", False), ("ge22", "generated", False), ("re33", "generated", True)):
        for level in (1, 2, 3):
            out.append({"name": f"{code}-L{level}", "game": code, "level": level, "role": "train", "source": src,
                        "family": code, "cap": 20000, "prior": {"clears": 2, "tries": 4, "median_to_clear": 9000},
                        **({"fresh": True} if level == 1 else {}), **({"reserve": True} if reserve else {})})
    return out + [{"name": "te", "game": "zz99", "level": 1, "role": "test", "fresh": True, "read": True, "tries": 2}]


def mg_v7(tmp_path, monkeypatch, log, *, steps=3, explore=True, v7=None, mg=None, seg=None, play_delay=0.03):
    cfg = {"sets": mg_sets(), "group_size": 2,
           "curriculum": {"groups": 1, "group_size": 2, "seed": 1, "allow_no_ref": True, "shares": {"public": 0.5, "generated": 0.5}},
           "harvest": {"levels": [2, 3], "switch_round": 1, "max_uses": 2, "stale": 4, "fill": False},
           **({"explore": {"games": ["pa11", "ge22"], "streams": 1, "poll": 0.01, "budget": 300000,
                           "seg_tokens": 65536, "board_check": False}} if explore else {}),
           **(mg or {})}
    loop = v7_loop(tmp_path, monkeypatch, log, steps=steps, play_delay=play_delay, mg=cfg)
    loop.cfg["mg"]["v7"].update(v7 or {})
    fast = loop.mg_play
    loop.group_plays = []

    async def play(name, spec, starts, tries, **kw):  # rows the curriculum reads (a clear with its level row)
        loop.group_plays.append({"name": name, "set": spec.get("name"), "starts": list(starts), "tries": tries,
                                 "cap": spec.get("cap"), "concurrency": kw.get("concurrency")})
        folder, rows, play_ = await fast(name, spec, starts, tries, **kw)
        for i, r in enumerate(rows):
            r.update(point=f"{name}-p", **{"pass": i + 1}, outcome="cleared" if r["cleared"] else "budget",
                     level_rows=[{"tokens": 1000, "score": 80.0}] if r["cleared"] else [])
        return folder, rows, play_

    loop.mg_play = play
    loop.mg_reward = lambda r, name: 1.0 if r.get("cleared") else -1.0
    loop.mg_level_ref = lambda name, game, level: {"tokens": 9000, "actions": 10}
    loop.explore_segs = []
    script = seg or {}

    planned = {}

    def points(spec, starts, name, harvest=False):  # mg_points needs the bundle's points file: the plan kept here
        planned[name] = (dict(spec), list(starts), harvest)
        return loop.work / f"{name}-points.json", []

    loop.mg_points = points

    async def bench(name, games, passes, budgets, *, model, own=None, **kw):  # one explorer segment's bench
        spec, starts, harvest = planned[name]
        version = int(own["policy_version"])
        assert own.get("server1_only") and harvest and passes == 1
        before = len(log)
        loop.explore_segs.append({"name": name, "level": int(spec["level"]), "version": version, "model": model,
                                  "starts": list(starts), "cap": spec["cap"]})
        k = int(spec["level"])
        await asyncio.sleep(script.get("delay", 0.05) if not callable(script.get("delay")) else script["delay"](k))
        assert ("server.unload", model) not in log[before:]  # a version an explorer holds is never unloaded
        folder = loop.output / "rl" / name
        (folder / "attempts" / "a0").mkdir(parents=True, exist_ok=True)
        mid = script.get("mid_at") == k and (not starts or "-m" not in str(starts[0][0]))
        out = "budget" if mid else "cleared"
        row = {"label": "a0", "game": games[0], "outcome": out, "completion_tokens": 4000, "total_levels": 3,
               "harvest": {"request": 7, "levels": k, "state": "NOT_FINISHED"} if out == "cleared" else None}
        if starts:
            row["restore"] = {"ok": True}
        return folder, [row], {}

    loop.bench = bench

    def cut_clear(attempt, dest, row, level, checks=None, game=None, ceiling=None):
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "rollout.jsonl").write_text("{}\n")
        return (Path(dest), 7, 100), None

    def cut_mid(attempt, dest, after):
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "rollout.jsonl").write_text("{}\n")
        return (Path(dest), int(after) + 2), None

    monkeypatch.setattr(v7_envs, "cut_clear", cut_clear)
    monkeypatch.setattr(v7_envs, "cut_mid", cut_mid)
    return loop


def test_v7_multi_game_explorers_feed_the_pool_groups_draw_fresh_starts(tmp_path, monkeypatch):
    log = []
    loop = mg_v7(tmp_path, monkeypatch, log, steps=4, seg={"mid_at": 2},
                 v7={"read_per_start": 2, "read_fresh_tries": 4, "read_base": True, "read_every": 2})
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    setup = events(loop, "v7e_setup")[0]
    assert setup["active"] == 2 and setup["reserve"] == 1 and not setup["resume"]
    # explorer chains: L1 fresh, L2 cut mid-level once and continued from its own cut, L3; each segment on the
    # version served when it started; the reserve game never explored
    segs = loop.explore_segs
    assert segs and segs[0]["level"] == 1 and not segs[0]["starts"]
    l2 = [s for s in segs if s["level"] == 2]
    assert len(l2) >= 2 and "-m" in l2[1]["starts"][0][0]
    assert {s["name"].split("-")[2] for s in segs} <= {"pa11", "ge22"}
    assert all(s["model"] == ("base" if s["version"] == 0 else f"policy-r{s['version']:02d}") for s in segs)
    plays = events(loop, "explore_play")
    assert plays and plays[0]["outcome"] == "won" and plays[0]["levels"] == 3 and plays[0]["segments"] == 4
    assert events(loop, "explore_first_deep")
    # groups: L2/L3 groups restore explorer starts; every group created had a start (no starved row)
    restored = [p for p in loop.group_plays if p["set"] and not p["set"].endswith("L1") and not p["name"].startswith("x")]
    assert restored and all(p["starts"] and "/harvest/x" in p["starts"][0][0] for p in restored)
    assert not events(loop, "starved")
    assert {d["set"][:4] for d in events(loop, "group_start")} <= {"pa11", "ge22"}
    # reads: version 0 base read as its own row, then every 2 steps; 4 tries on the fresh held-out set
    reads = events(loop, "read")
    assert reads[0]["step"] == 0 and reads[0]["version"] == 0 and reads[0]["base"]
    assert [r["step"] for r in reads[1:]] == [2, 4]
    assert all(p["tries"] == 4 for p in loop.group_plays if p["name"].startswith("x"))
    rows = events(loop, "envs")
    assert [r["step"] for r in rows] == [1, 2, 3, 4] and rows[-1]["explore"]["live"] <= 1
    state = json.loads((tmp_path / "out" / "v7_state.json").read_text())
    assert state["envs"]["harvest"] and state["envs"]["active"] == ["ge22", "pa11"]
    assert state["curriculum"]["made"]


def test_v7_explorer_pins_count_against_max_loras_at_the_floor(tmp_path, monkeypatch):
    """RL condition: explorer segments hold their version like groups; with max_loras 2 live versions never pass 2
    and a held version is never unloaded (asserted inside the fake segment)."""
    log = []
    loop = mg_v7(tmp_path, monkeypatch, log, steps=10, v7={"max_loras": 2},
                 seg={"delay": lambda k: 0.4 if k == 1 else 0.02})
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    assert all(len([x for x in s["live_versions"] if x != 0]) <= 2 for s in events(loop, "step"))  # v0: the base, no slot
    assert not [a for a in events(loop, "alert") if "max_loras" in a.get("why", "")]
    assert loop.explore_segs[0]["version"] == 0 and any(s["version"] > 0 for s in loop.explore_segs)


def test_v7_resume_restores_pool_chains_and_mix_and_an_old_state_still_resumes(tmp_path, monkeypatch):
    log = []
    first = mg_v7(tmp_path, monkeypatch, log, steps=2, seg={"delay": lambda k: 0.3})
    asyncio.run(first.mg_loop())
    state = json.loads((tmp_path / "out" / "v7_state.json").read_text())
    made1 = sum(state["curriculum"]["made"].values())
    chains = state["envs"]["explore"]["chains"]
    assert chains and state["envs"]["harvest"] is not None
    second = mg_v7(tmp_path / "b", monkeypatch, [], steps=4, seg={"delay": lambda k: 0.05})
    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "out"))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "out"))
    assert asyncio.run(second.mg_loop()) == "max_steps"
    setup = events(second, "v7e_setup")[0]
    assert setup["resume"] and setup["chains"] == len(chains)
    resumed = [s for s in second.explore_segs if s["name"].startswith(chains[0]["name"])]
    assert resumed and (resumed[0]["starts"] == ([tuple(chains[0]["start"])] if chains[0]["start"] else []))
    assert sum(second.mg_curr.made.values()) > made1
    # an older state (no envs, no plan_one counts): resumes with a fresh pool beside the restored loop state
    old = json.loads((tmp_path / "out" / "v7_state.json").read_text())
    old.pop("envs")
    old["curriculum"].pop("made"), old["curriculum"].pop("recent")
    (tmp_path / "out" / "v7_state.json").write_text(json.dumps(old))
    third = mg_v7(tmp_path / "c", monkeypatch, [], steps=old["step"] + 1)
    monkeypatch.setenv("ARC_RL1_RESUME", str(tmp_path / "out"))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "out"))
    assert asyncio.run(third.mg_loop()) == "max_steps"
    assert not events(third, "v7e_setup")[0]["resume"]


def test_v7_no_start_anywhere_emits_a_starved_row_a_minute_never_a_sleep(tmp_path, monkeypatch):
    log = []
    loop = mg_v7(tmp_path, monkeypatch, log, steps=2, explore=False)
    for s in loop.cfg["mg"]["sets"]:  # restored levels only, no pool: nothing to play
        s.pop("fresh", None)

    async def run():
        try:
            await asyncio.wait_for(loop.mg_loop(), 1.0)
        except TimeoutError:
            pass

    asyncio.run(run())
    starved = events(loop, "starved")
    assert len(starved) == 1 and not events(loop, "step")


def test_rotation_swaps_a_game_that_stopped_teaching_for_a_reserve_game_of_its_source(tmp_path, monkeypatch):
    log = []
    loop = mg_v7(tmp_path, monkeypatch, log, steps=1, mg={"rotation": {"min_explored": 1}})
    asyncio.run(loop.mg_loop())
    cur = loop.mg_curr
    gen = [u for u in cur.units.values() if u.game == "ge22"]
    for u in gen:
        u.seen, u.state = 1, "parked"
    loop.v7e["explored"]["ge22"] = 0
    assert loop.v7e_rotate(2) == []  # never explored: deeper levels had no chance yet
    loop.v7e["explored"]["ge22"] = 1
    gen[0].state = "active"
    assert loop.v7e_rotate(2) == []  # an observed unit still teaches
    gen[0].state = "retired"
    (row,) = loop.v7e_rotate(2)
    assert row["out"] == "ge22" and row["in"] == "re33" and loop.v7e["active"] == ["pa11", "re33"]
    assert "re33" in loop.mg_explorers.rot and loop.v7e["reserve"] == []
    for u in [u for u in cur.units.values() if u.game == "pa11"]:  # public games never rotate
        u.seen, u.state = 1, "retired"
    loop.v7e["explored"]["pa11"] = 5
    assert loop.v7e_rotate(3) == []
    picks = [loop.v7e_pick(loop.cfg["mg_sets"], {})[0] for _ in range(20)]
    assert picks and not [n for n in picks if n and n.startswith("ge22")]
    assert any(n and n.startswith("re33") for n in picks)
    assert loop.v7e_state()["rotated"][0]["in"] == "re33"


def test_group_cut_merges_into_the_live_pool_while_it_changes(tmp_path, monkeypatch):
    """A training group's harvest cuts run off the loop on a copy; rows the explorers add (and the step's prune)
    meanwhile are kept, and the cut rows join the pool and the curriculum."""
    log = []
    loop = mg_v7(tmp_path, monkeypatch, log, steps=1)
    asyncio.run(loop.mg_loop())
    from arena import harvest as harvest_mod

    def collect(shim, number, plays, played_by, checks=None):
        loop.mg_harvest.append(start_row("late", played_by, target="pa11-L3"))  # an explorer's add meanwhile
        shim.mg_harvest.append({**start_row(f"r{number}", played_by, target="pa11-L3"), "game": "pa11", "level": 3,
                                "source": "harvest", "start": {"attempt": "/a/cut", "request": 3}})
        return {"offered": 1, "new": 1, "dropped": {}}

    monkeypatch.setattr(harvest_mod, "collect", collect)
    g = SimpleNamespace(gid="g00042", unit="pa11-L2", version=1)
    spec = loop.cfg["mg_sets"]["pa11-L2"]
    out = asyncio.run(loop.v7e_group_done(g, spec, tmp_path, [{"harvest": {"request": 1}, "completion_tokens": 10}]))
    assert out["harvest"]["new"] == 1
    ids = [r["id"] for r in loop.mg_harvest]
    assert "xlate" in ids and "xr42" in ids
    assert {"attempt": "/a/cut", "request": 3} in loop.mg_curr.units[loop.mg_curr.key("pa11", 3)].starts


def test_read_share_caps_concurrent_read_tries_and_read_every_is_a_knob(tmp_path, monkeypatch):
    log = []
    knobs = tmp_path / "knobs.json"
    loop = mg_v7(tmp_path, monkeypatch, log, steps=4, explore=False,
                 v7={"read_per_start": 2, "read_fresh_tries": 4, "read_share": 0.25, "read_every": 1, "read_base": True},
                 mg={"active_tries": 8, "sets": mg_sets() + [{"name": "te2", "game": "zz98", "level": 1, "role": "test",
                                                             "fresh": True, "read": True, "tries": 2}]})
    monkeypatch.setenv("ARENA_KNOBS", str(knobs))
    live, peak = [0], [0]
    inner = loop.mg_play

    async def play(name, spec, starts, tries, **kw):
        if not name.startswith("x"):
            return await inner(name, spec, starts, tries, **kw)
        n = min(tries * max(1, len(starts)), kw.get("concurrency") or 10 ** 6)  # the bench's tries at once
        live[0] += n
        peak[0] = max(peak[0], live[0])
        try:
            await asyncio.sleep(0.05)
            return await inner(name, spec, starts, tries, **kw)
        finally:
            live[0] -= n

    loop.mg_play = play
    step = loop.v7_step

    async def stepped(n, groups):
        row = await step(n, groups)
        if n == 1:
            knobs.write_text(json.dumps({"read_every": 2}))
        return row

    loop.v7_step = stepped
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    assert peak[0] <= 2  # 25% of 8 active tries, taken out of the pool
    reads = events(loop, "read")
    assert [r["step"] for r in reads] == [0, 2, 4] and reads[0]["base"]  # the knob applies at step 1's boundary
    assert [k for k in events(loop, "knobs") if k.get("changed", {}).get("read_every") == [1, 2]]
    assert all(p["concurrency"] == 2 for p in loop.group_plays if p["name"].startswith("x"))


def test_read_gate_cap_shrinks_and_grows_in_place_while_reads_are_open(tmp_path, monkeypatch):
    """RL review: an active_tries change while a read is open moves the shared gate in place; every slot comes back."""
    log = []
    loop = mg_v7(tmp_path, monkeypatch, log, steps=1, explore=False, v7={"read_per_start": 2, "read_fresh_tries": 4,
                                                                      "read_share": 0.5}, mg={"active_tries": 8})
    asyncio.run(loop.mg_loop())
    inner = loop.mg_play

    async def slow(name, spec, starts, tries, **kw):
        await asyncio.sleep(0.1)
        return await inner(name, spec, starts, tries, **kw)

    loop.mg_play = slow
    fixed = {"te": None}
    sets = loop.cfg["mg_sets"]

    async def reads():
        mg = loop.cfg["mg"]
        a = asyncio.ensure_future(loop.v7_read(11, 1, sets, fixed))
        await asyncio.sleep(0.02)
        mg["active_tries"] = 4  # cap 4 -> 2 while a holds 4
        b = asyncio.ensure_future(loop.v7_read(12, 1, sets, fixed))
        await asyncio.sleep(0.02)
        mg["active_tries"] = 16  # cap -> 8
        c = asyncio.ensure_future(loop.v7_read(13, 1, sets, fixed))
        await asyncio.gather(a, b, c)

    asyncio.run(reads())
    gate = loop.v7["read_gate"]
    assert gate["cap"] == 8 and gate["free"] == 8
    assert [r["step"] for r in events(loop, "read")][-3:] == [11, 12, 13]


def test_v7_refuses_a_trainer_fast_knob_without_a_gate(tmp_path, monkeypatch):
    log = []
    knobs = tmp_path / "knobs.json"
    knobs.write_text(json.dumps({"trainer_fast": {"new_path": True}}))
    loop = mg_v7(tmp_path, monkeypatch, log, steps=3, explore=False, mg={"trainer_fast": {"old": True}})
    monkeypatch.setenv("ARENA_KNOBS", str(knobs))  # after make_loop, which clears it
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    assert loop.cfg["mg"]["trainer_fast"] == {"old": True} and loop.trainer.always == {"old": True}
    refused = [k for k in events(loop, "knobs") if k.get("why") == "v7: no trainer_fast gate"]
    assert len(refused) == 1, events(loop, "knobs")  # refused once, not again while the file keeps the value


def test_rotation_by_its_own_thresholds_with_the_curriculums_park_and_retire_off(tmp_path, monkeypatch):
    log = []
    loop = mg_v7(tmp_path, monkeypatch, log, steps=1, mg={"rotation": {"min_explored": 1, "min_tries": 12}})
    asyncio.run(loop.mg_loop())
    gen = [u for u in loop.mg_curr.units.values() if u.game == "ge22"]
    for u in gen:
        u.seen, u.state, u.tries, u.won = 2, "active", 12, 12  # mastered by the rotation's own line
    gen[1].won = 6  # still in the band: the game teaches
    loop.v7e["explored"]["ge22"] = 1
    assert loop.v7e_rotate(2) == []
    gen[1].won = 1  # too hard now
    assert loop.v7e_rotate(2)[0]["in"] == "re33"


def test_envs_hero_cfg_sets_active_reserve_reads_and_passes_check_mg():
    from arena.envs_hero import HELD, envs_hero_cfg
    from arena.rl_loop import check_mg

    pub = [f"pb{i:02d}" for i in range(15)]
    env = [f"ev{i:02d}" for i in range(45)]
    res = ["rs01", "rs02", "nohash"]
    sets, pts = [], []
    for g in pub + env + res:
        src = "public" if g in pub else "generated"
        for level in (1, 2):
            sets.append({"name": f"{g}-L{level}", "game": g, "level": level, "role": "train", "cap": 30000, "source": src,
                         "family": g, "ref": {"tokens": 9000, "actions": 9, "baseline": 7 if level == 1 or g != "ev00" else None},
                         "prior": {"tries": 6, "clears": 3 if g == "rs02" else 6, "median_to_clear": 9000},
                         **({"fresh": True} if level == 1 else {})})
        pts.append({"id": f"{g}-game", "game": f"{g}-abc", "total_levels": 2, "baselines": [7, None],
                    "board_hash": None if g == "nohash" else "h"})
    bundle = [p for p in pts if p["id"][:4] in pub + env] + [{"id": f"{c}-game", "game": f"{c}-x", "total_levels": 3}
                                                            for c in HELD]
    seed = {"envs_seed": {"env_games": env}, "rl_loop": {"mg": {"record": [{"game": g} for g in pub + list(HELD)]}}}
    mg = {"v7": {"max_loras": 6}}
    info = envs_hero_cfg(mg, pool={"sets": sets, "points": pts, "shares": SHARES}, seed_study=seed, points=bundle,
                         seed_run="seed-v2")
    check_mg(mg)
    assert info["reserve"] == ["rs02", "rs01"]  # band levels first; no board hash: never reserve
    assert info["no_action_ref"] == ["ev00-L2"] and len(info["active"]) == 60
    assert {p["id"] for p in info["points_add"]} == {"rs01-game", "rs02-game"}
    reserve = [s for s in mg["sets"] if s.get("reserve")]
    assert {s["game"] for s in reserve} == {"rs01", "rs02"} and all(s["role"] == "train" for s in reserve)
    reads = [s for s in mg["sets"] if s["role"] == "test"]
    assert len(reads) == 30 and {s["read"] for s in reads} == {True, "screen"}
    assert mg["v7"]["max_loras"] == 6 and mg["v7"]["batch_groups"] == 8 and mg["v7"]["read_every"] == 8
    assert mg["harvest"]["stale"] == 4 and mg["harvest"]["max_uses"] == 2 and mg["curriculum"]["half_life"] == 3.0
    assert mg["explore"]["games"] == info["active"] and mg["async"]


def test_queued_explorer_tries_never_stop_group_creation(tmp_path, monkeypatch):
    """xv7a pfA: explorer benches queued at floor priority counted as server 1's queue and stopped group creation."""
    log = []
    loop = mg_v7(tmp_path, monkeypatch, log, steps=4, mg={"active_tries": 8}, seg={"delay": lambda k: 0.3})
    bench = loop.bench

    async def queued_bench(name, games, passes, budgets, **kw):  # the explorer's floor bench stays queued throughout
        level_bench.SHARED_SLOTS.want(name.rsplit("-s", 1)[0], -2 * 10 ** 9, 5, join=False)
        return await bench(name, games, passes, budgets, **kw)

    loop.bench = queued_bench

    async def run():
        return await asyncio.wait_for(loop.mg_loop(), 20)

    assert asyncio.run(run()) == "max_steps"


def test_bundle_read_every_is_not_a_knob_change(tmp_path, monkeypatch):
    """xv7a pfA: the step's knob sync put the bundle's read_every (8) over the shrunk one (2); only a value the
    knobs file changes moves it."""
    log = []
    loop = mg_v7(tmp_path, monkeypatch, log, steps=4, explore=False, v7={"read_every": 8})
    cfg = loop.v7_cfg
    loop.v7_cfg = lambda: {**cfg(), "read_every": 2}  # as a preflight shrink
    assert asyncio.run(loop.mg_loop()) == "max_steps"
    assert [r["step"] for r in events(loop, "read")] == [2, 4]


def test_explorer_chains_admitted_one_per_stagger(tmp_path, monkeypatch):
    """explorers are admitted staggered by the shared pool's admit_stagger_s (the bench tries'
    rule, one implementation); with a stagger past the run only one chain starts, without it both start at once."""
    import re

    from arena import v7_envs

    monkeypatch.setattr(v7_envs.Chains, "limit", lambda self: int(self.cfg["streams"]))  # no measured KV gate here

    def chains(stagger):
        log = []
        loop = mg_v7(tmp_path / f"s{int(stagger)}", monkeypatch, log, steps=2,
                     mg={"active_tries": 40, "admit_stagger_s": stagger,
                         "explore": {"games": ["pa11", "ge22"], "streams": 2, "poll": 0.01, "budget": 300000,
                                     "seg_tokens": 65536, "board_check": False}})
        assert asyncio.run(loop.mg_loop()) == "max_steps"
        return {re.search(r"explore-(\d{4})", s["name"]).group(1) for s in loop.explore_segs}

    assert len(chains(0.0)) >= 2
    assert chains(1000.0) == {"0001"}


def test_hero_explorers_four_at_131k_staggered():
    from arena import play_explorer

    assert "stagger_s" not in play_explorer.KEYS  # one stagger: mg.admit_stagger_s (v7_envs.admit_staggered)
    src = (Path(__file__).resolve().parents[1] / "src/arena/envs_hero.py").read_text()
    assert '"streams": 4' in src and '"context": 131072' in src and '"stagger_s"' not in src


def test_explorer_order_unplayed_first_then_candidates_then_rotation():
    """2 explorer streams on never-played active games first, then the envs candidate (reserve) games,
    each once; then the usual rotation over active games."""
    from collections import deque
    from types import SimpleNamespace

    from arena import v7_envs

    ch = v7_envs.Chains.__new__(v7_envs.Chains)
    unit = lambda g, seen: SimpleNamespace(game=f"{g}-00000000", seen=seen, picked=0)
    ch.loop = SimpleNamespace(mg_curr=SimpleNamespace(units={"a": unit("aa11", 1), "b": unit("bb22", 0)}),
                              v7e_active=lambda: ["aa11", "bb22", "cc33"])
    ch.cfg = {"order": "unplayed_first", "candidates": ["rr99", "zz00"], "per_game": 1}
    ch.source = {c: "envs" for c in ("aa11", "bb22", "cc33", "rr99")}  # zz00 has no training set: skipped
    ch.explored, ch.live, ch.rot = set(), {}, deque(["aa11", "bb22", "cc33"])
    picks = [ch.next_game() for _ in range(5)]
    assert picks[:3] == ["bb22", "cc33", "rr99"]  # unplayed actives (sorted), then the candidate
    assert picks[3:] == ["aa11", "bb22"]  # then the rotation
    assert ch.state.__func__  # explored persists in state
    ch.pending, ch.count, ch.first_deep, ch.t0, ch.done_chains = [], 0, None, 0.0, 0
    assert ch.state()["explored"] == ["bb22", "cc33", "rr99"]
