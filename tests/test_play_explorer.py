"""Explorers (envs 22 Oct, owner redesign): whole-game plays whose level starts feed the pool; running caps;
start staleness."""
import json
from pathlib import Path
from types import SimpleNamespace

from test_arena_harvest import GAME, LadderArcade, base_point, play
from test_level_bench import QuietMonitor

from arena import harvest, restore
from arena import play_explorer as explorer


def sets_for(game=GAME, n=5, source="generated"):
    out = {}
    for k in range(1, n + 1):
        out[f"g-L{k}"] = {"name": f"g-L{k}", "role": "train", "game": game, "level": k, "source": source,
                          "family": "fa", "cap": 1000 * k, **({"fresh": True} if k == 1 else {})}
    return out


def fake_loop(tmp_path, sets, **x):
    feed = []
    loop = SimpleNamespace(cfg={"mg": {"explore": {"games": ["aa11"], **x}}, "mg_sets": sets}, output=tmp_path / "loop",
                           feed=SimpleNamespace(emit=lambda kind, **k: feed.append((kind, k))), games_dir=None,
                           mg_point=lambda code: base_point(through=True), serving_adapter=None, mg_curr=None,
                           server=SimpleNamespace(model="base"), mg_play_version=lambda m: 0)
    loop.__dict__["mg_harvest"] = []
    return loop, feed


async def test_a_whole_game_play_yields_a_restorable_start_per_level(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vol"))
    row, att = await play(tmp_path, "explore-0001-aa11", base_point(through=True, budget=400), [])
    assert row["levels"] >= 3, row
    loop, feed = fake_loop(tmp_path, sets_for())
    ex = explorer.Explorers(loop)
    ex.checks = (LadderArcade(), "card")
    rec = {"folder": tmp_path / "explore-0001-aa11", "code": "aa11", "gid": GAME, "seen": set(), "n": 1, "version": 0}
    got = ex.collect_one_sync("explore-0001-aa11", rec)
    assert sorted(r["level"] for r in got) == list(range(2, min(5, row["levels"] + 1) + 1))
    for r in got:
        info = restore.load(r["attempt"], r["request"])
        assert info["level"] == r["level"] and r["target"] == f"g-L{r['level']}" and r["source"] == "explore"
        assert harvest.board_check(LadderArcade(), "card", GAME, Path(r["attempt"]), r["request"])[0]
        rows = [json.loads(x) for x in (Path(r["attempt"]) / "rollout.jsonl").read_text().splitlines()]
        assert [x["n"] for x in rows] == list(range(r["request"] + 1))  # cut at the start's request
    assert ex.collect_one_sync("explore-0001-aa11", rec) == []  # each level once
    assert ex.collect_one_sync("explore-0001-aa11", rec, final=True) == []
    ex.add(got)
    assert len(loop.mg_harvest) == len(got) and (tmp_path / "vol/harvest.jsonl").is_file()


def test_running_caps_from_clears_then_censored_tokens(tmp_path):
    assert explorer.running_cap({}) is None
    assert explorer.running_cap({"censored": [90000, 250000]}) is None  # failures never set a cap
    assert explorer.running_cap({"clears": [10000, 20000, 30000, 40000, 50000]}) == 50000  # 1.25 x p80 (40K)
    sets = sets_for()
    loop, _ = fake_loop(tmp_path, sets)
    for t in (8000, 9000, 10000):
        explorer.record_tokens(loop, "aa11", 2, t, True)
    explorer.record_tokens(loop, "aa11", 3, 70000, False)
    cur = SimpleNamespace(cfg={"tier_factor": 1.5}, key=lambda g, lv: f"{g}@L{lv}",
                          units={f"{GAME}@L2": SimpleNamespace(cap=2000, tier=1)})
    loop.mg_curr = cur
    changed = explorer.update_caps(loop, {})
    assert changed == {"g-L2": 11000}  # 1.25 x 9000 rounded; L3 (no clears) keeps its cap
    assert cur.units[f"{GAME}@L2"].cap == 16500  # the unit keeps its tier on the new cap


def test_repeated_failures_leave_the_cap_and_the_stuck_stop_unchanged(tmp_path):
    loop, _ = fake_loop(tmp_path, sets_for(), stuck=2.0)
    ex = explorer.Explorers(loop)
    before = ex.point("aa11")["level_caps"]
    for _ in range(5):  # explorers stuck on L3: each spends its whole stuck stop
        explorer.record_tokens(loop, "aa11", 3, before[2], False)
        assert explorer.update_caps(loop, {}) == {}
        assert ex.point("aa11")["level_caps"] == before
    assert loop.cfg["mg_sets"]["g-L3"]["cap"] == 3000


def test_stuck_stop_for_env_games_full_budget_for_public(tmp_path):
    sets = sets_for()
    for s in sets.values():
        s["prior"] = {"clears": 1}
    sets["g-L4"]["prior"] = {"clears": 0}  # no data: the no-data cap, not the set's pool cap
    loop, _ = fake_loop(tmp_path, sets, stuck=2.0)
    p = explorer.Explorers(loop).point("aa11")
    assert p["through"] and p["level_caps"] == [2000, 4000, 6000, 54000, 10000]
    loop, _ = fake_loop(tmp_path, sets_for(source="public"))
    assert "level_caps" not in explorer.Explorers(loop).point("aa11")


def test_rotation_keeps_per_game_concurrency(tmp_path):
    loop, _ = fake_loop(tmp_path, sets_for(), games=["aa11", "bb22"], per_game=1)
    ex = explorer.Explorers(loop)
    ex.live = {"x": {"code": "aa11"}}
    assert ex.next_game() == "bb22"
    ex.live["y"] = {"code": "bb22"}
    assert ex.next_game() is None


def test_stale_starts_leave_the_pick(tmp_path):
    h = {**harvest.KEYS, "stale": 4}
    assert harvest.fresh_enough(h, {"played_by": 0}, 5) and not harvest.fresh_enough(h, {"played_by": 0}, 6)
    assert harvest.fresh_enough({**harvest.KEYS}, {"played_by": 0}, 60)
    loop = SimpleNamespace(cfg={"mg": {"harvest": {"switch_round": 1, "stale": 4, "max_uses": 2}}},
                           feed=SimpleNamespace(emit=lambda *a, **k: None))
    loop.mg_harvest = [{"id": "s0", "target": "L2", "round": 0, "played_by": 0, "attempt": "/a", "request": 1},
                       {"id": "x5", "target": "L2", "round": 5, "played_by": 5, "attempt": "/b", "request": 2}]
    starts, _ = harvest.pick(loop, 7, "L2", {"level": 2}, None, k=2, tries=12)
    assert starts == [("/b", 2)]  # the base-seed start is 6 updates old at round 7


def test_progress_read_by_version():
    ev = [{"kind": "explore_play", "version": 0, "levels": 2, "full": True},
          {"kind": "explore_play", "version": 0, "levels": 4, "full": True},
          {"kind": "explore_play", "version": 3, "levels": 5, "full": True},
          {"kind": "explore_play", "version": 3, "levels": 9, "full": False}]
    assert explorer.progress(ev) == {"0": {"plays": 2, "levels_mean": 3.0}, "3": {"plays": 1, "levels_mean": 5.0}}


def test_explorer_benches_wait_for_every_other_bench_and_never_release_a_hold():
    from arena.level_bench import SharedSlots

    pool = SharedSlots(4)
    pool.hold(1)
    pool.want("x", 300000 - 2 * 10 ** 9, 1, join=False)
    assert not pool.take("x")  # the round's bench has not registered: the hold stands
    pool.want("r", 64000, 1)
    assert not pool.take("x") and pool.take("r") and pool.take("x")


def test_live_plays_bounded_by_the_kv_share(tmp_path, monkeypatch):
    from arena import level_bench

    loop, feed = fake_loop(tmp_path, sets_for(), streams=24)
    ex = explorer.Explorers(loop)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    assert ex.limit() == 1  # the gate is not measured yet
    pool = level_bench.SharedSlots(128)
    pool.capacity = 10_000_000
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", pool)
    assert ex.limit() == 11 and feed[-1][0] == "explore_limit"  # 0.15 x 10M / 131,072
    ex.live = {"explore-0001-a": {}, "explore-0002-b": {}}
    ex.reserve()  # the live plays' KV counts in the tries' kv_fill gate
    assert pool.reserved == 2 * level_bench.KV_PROJECT["ctx_frac"] * 131072


def test_explore_kv_share_knob_steps_the_live_limit_down(tmp_path, monkeypatch):
    import json
    from arena import level_bench
    from arena.rl_single import MultiMixin

    loop, feed = fake_loop(tmp_path, sets_for(), streams=24, kv_share=0.3)
    loop.trainer, loop.KNOBS = None, MultiMixin.KNOBS
    loop.mg_explorers = ex = explorer.Explorers(loop)
    pool = level_bench.SharedSlots(128)
    pool.capacity = 10_000_000
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", pool)
    assert ex.limit() == 22  # 0.30 x 10M / 131,072
    knobs = tmp_path / "knobs.json"
    monkeypatch.setenv("ARENA_KNOBS", str(knobs))
    knobs.write_text(json.dumps({"explore_kv_share": 0.2}))
    MultiMixin.mg_knobs(loop, 5)
    assert ex.limit() == 15 and loop.cfg["mg"]["explore"]["kv_share"] == 0.2
    assert [k for k, kw in feed if k == "knobs"] and feed[-2][1]["changed"] == {"explore_kv_share": [0.3, 0.2]}
    knobs.write_text(json.dumps({"explore_kv_share": 0}))  # out of range: refused, limit unchanged
    MultiMixin.mg_knobs(loop, 6)
    assert ex.limit() == 15 and feed[-1][1]["refused"] == ["explore_kv_share"]


def test_stale_zero_is_refused():
    import pytest

    with pytest.raises(ValueError, match="stale"):
        harvest.settings({"harvest": {"stale": 0}})


async def test_seed_starts_from_the_screen_recordings(tmp_path, monkeypatch):
    import arc_infra.metrics
    from arena.rl_single import MultiMixin

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    row, att = await play(tmp_path, "rec-aa11", base_point(through=True, budget=400), [])
    assert (tmp_path / "rec-aa11/level-bench.json").is_file()
    feed = []
    fake = SimpleNamespace(feed=SimpleNamespace(emit=lambda kind, **k: feed.append(kind)),
                           mg_attempt_start=MultiMixin.mg_attempt_start)
    sets = {**sets_for(), "g-L2": {**sets_for()["g-L2"], "max_prompt": 10 ** 9}}
    screen = {"recorded": {"aa11": str(tmp_path / "rec-aa11"), "zz99": str(tmp_path / "nope")}}
    got = MultiMixin.mg_screen_starts(fake, screen, sets)
    assert sorted(r["level"] for r in got) == list(range(2, min(5, row["levels"] + 1) + 1))
    assert all(r["played_by"] == 0 and r["source"] == "seed" for r in got) and feed == ["mg_seed_starts_missing"]
    for r in got:
        assert restore.load(r["attempt"], r["request"])["level"] == r["level"]
    # ARC_RL1_SEED_SCAN: one scan per recording in spawned processes, the same rows (and the same missing row)
    fake.mg_seed_scans = lambda screen, names, workers: MultiMixin.mg_seed_scans(fake, screen, names, workers)
    feed.clear()
    assert MultiMixin.mg_screen_starts(fake, screen, sets, workers=2) == got
    assert feed == ["mg_seed_scan", "mg_seed_starts_missing"]
    for attempt in {r["attempt"] for r in got}:
        first, prompts, starts = restore.attempt_scan(attempt)
        assert starts == restore.level_starts(attempt) and first == restore.unsafe_from(attempt)[0]


async def test_plays_keep_their_adapter_under_its_own_name_and_unused_names_unload(tmp_path):
    calls = []

    class Server:
        model, names = "base", set()

        async def load(self, name, path):
            calls.append(("load", name, path))
            self.names.add(name)
            return 0.1

        async def unload(self, name):
            calls.append(("unload", name))
            self.names.discard(name)

    loop, feed = fake_loop(tmp_path, sets_for())
    loop.server = Server()
    ex = explorer.Explorers(loop)
    assert await ex.model() == ("base", 0) and not calls  # the base round: no adapter
    loop.serving_adapter, loop.mg_version = "/a/r01", 1
    assert await ex.model() == ("explore-r01", 1) and await ex.model() == ("explore-r01", 1)
    assert calls == [("load", "explore-r01", "/a/r01")]  # loaded once
    ex.live = {"p": {"model": "explore-r01", "code": "aa11"}}
    loop.serving_adapter, loop.mg_version = "/a/r02", 2
    await ex.release()
    assert ("unload", "explore-r01") not in calls  # a live play still uses it
    assert await ex.model() == ("explore-r02", 2)
    ex.live = {}
    await ex.release()
    assert calls[-1] == ("unload", "explore-r01") and set(ex.own) == {"explore-r02"}


def test_new_tries_go_to_the_less_loaded_server_so_long_tries_spread_over_both_cards():
    from arena.level_bench import SharedSlots, balanced

    one, two = SharedSlots(8), SharedSlots(8)
    for p in (one, two):
        p.want("b", 300000, 8)
    got = []
    for _ in range(6):  # a longest-first bench: its first (longest) tries alternate between the servers
        p = balanced(one, two)[0]
        assert p.take("b")
        got.append(1 if p is one else 2)
    assert got == [1, 2, 1, 2, 1, 2]
    two.active = False  # a parked time-share server takes nothing
    assert balanced(one, two)[0] is one and balanced(None, one) == [one]
