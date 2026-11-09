"""mg.lookahead_below (continuous play V1): versioned adapters (policy-rNN), the look-ahead round launched on the play
server once the round's tries have all started, train-time policy lag, and kill/resume with look-ahead tries in
flight. The mg loop runs on the stand-ins of test_rl_startup (trainer, server, play)."""
import asyncio
import json
import shutil
import time
from pathlib import Path

import pytest
from test_rl_startup import Server, make_loop

from arena import level_bench, rl_loop, rl_single
from arena.fake_trainer import adapter


class VServer(Server):
    async def unload(self, name):
        self.log.append(("server.unload", name))


class Pool:  # level_bench.SharedSlots as the watcher reads it: every round bench launched, few tries running
    def __init__(self):
        self.pending, self.used, self.peak, self.kv_waits, self.peak_load = {}, 0, 0, 0, 0


def la_loop(root, monkeypatch, log, *, mg=None, source=None, play_delay=0.3):
    loop = make_loop(root, monkeypatch, log, play_delay=play_delay, source=source,
                     mg={"lookahead_below": 64, "max_rounds": 11, "updates": 11, "cap": 1000,
                         "sets": [{"name": "tr", "game": "g1", "role": "train", "tries": 2},
                                  {"name": "te", "game": "g2", "role": "test", "tries": 2}], **(mg or {})})
    loop.server = VServer(log)
    monkeypatch.setattr(rl_single, "LOOKAHEAD_MIN_S", 0)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", Pool())
    out = root / "out"
    loop.picks = []

    def pick(number, name, s, pool):
        loop.picks.append((number, name))
        att = root / "starts" / f"att{number}"  # a restorable attempt folder (resume maps recorded starts to it)
        att.mkdir(parents=True, exist_ok=True)
        (att / "rollout.jsonl").write_text("{}\n")
        return [(str(att), number)], 2

    async def bench(name, games, passes, budgets, *, model, own=None, **kw):
        """LevelBench as the look-ahead uses it: its own keep/reuse roots over the globals, its version in every row;
        try a1 ends at once, a2 at the end. The real mg_play runs above it (version, name users)."""
        own = own or {}
        version = own.get("policy_version")
        roots = own.get("try_keep") or {}
        keep = roots.get("keep") if "keep" in roots else level_bench.TRY_KEEP
        reuse = roots.get("reuse") if "reuse" in roots else level_bench.TRY_REUSE
        kept = level_bench.kept_tries(name, reuse)
        log.append(("play_start", name, model, version, bool(own.get("server1_only")),
                    sorted((k, r.get("policy_version")) for k, (r, _) in kept.items()), loop.points.get(name)))
        rows = []

        def finish(label):
            if label in kept:
                rows.append(kept[label][0])
                return
            row = {"label": label, "outcome": "cleared", "policy_version": version, "game": games[0],
                   "point": f"{games[0]}-L1-x"}
            rows.append(row)
            if keep is not None:
                dest = Path(keep) / name
                (dest / "attempts" / label).mkdir(parents=True, exist_ok=True)
                (dest / "attempts" / label / "rollout.jsonl").write_text("{}\n")
                with (dest / "tries.jsonl").open("a") as file:
                    file.write(json.dumps(row) + "\n")

        finish("a1")
        await asyncio.sleep(loop.delays.get(name, play_delay))
        finish("a2")
        log.append(("play_end", name))
        folder = out / "arena" / "trials" / "rl-loop" / "rl" / name
        folder.mkdir(parents=True, exist_ok=True)
        return folder, rows, {"seconds": play_delay, "attempts": 2, "usable": 2, "tokens": 0}

    def points(spec, starts, name, harvest=False):
        loop.points[name] = [list(x) for x in starts]
        return root / f"{name}.json", [{}]

    async def update(number, data, probes):
        log.append(("update", number, sorted(int(item[1]["policy_version"]) for item in data)))
        serve = loop.adapters / f"r{number:02d}"
        adapter(serve)
        return {"ok": True}, str(serve)

    async def after_update(number, row, update_row, serve, load=True):
        ckpt = out / "state" / f"r{number:02d}"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "state.json").write_text("{}")
        (ckpt / "mg_state.json").write_text("{}")
        if not (ckpt / "adapter").exists():  # the loop's own state save may have put it there
            shutil.copytree(serve, ckpt / "adapter")
        await loop.mg_serve(serve, number)
        loop.serving_adapter, loop.keep_checkpoint = serve, f"r{number:02d}"
        return None, True

    loop.points, loop.delays = {}, {}
    loop.mg_pick = pick
    loop.mg_play = rl_single.MultiMixin.mg_play.__get__(loop)
    loop.bench, loop.mg_points, loop.game_rows = bench, points, lambda rows, game: rows
    loop.mg_select = lambda folder, rows, name=None: ([(folder, r, 1.0, "k") for r in rows], {})
    loop.mg_update = update
    loop.mg_after_update = after_update
    return loop


def kinds(loop, kind):
    return [d for k, d in loop.feed.events if k == kind]


def starts_of(log, name):
    return [e for e in log if e[:2] == ("play_start", name)]


# --- settings ------------------------------------------------------------------------------------------------------

def test_check_mg_takes_the_keys_and_refuses_bad_values_and_paths():
    rl_loop.check_mg({"lookahead_below": 64, "lookahead_lag": 2})
    for bad in ({"lookahead_below": 0}, {"lookahead_below": 64, "lookahead_lag": 3}, {"lookahead_below": 1.5},
                {"lookahead_below": 64, "single_card": True}, {"lookahead_below": 64, "curriculum": {"x": 1}},
                {"lookahead_below": 64, "carry": True}):
        with pytest.raises(ValueError):
            rl_loop.check_mg(bad)


def test_key_off_keeps_the_single_policy_name(tmp_path, monkeypatch):
    log = []
    loop = make_loop(tmp_path, monkeypatch, log)
    assert not loop.mg_versioned() and loop.mg_name() == loop.mg_name(7) == "policy"
    asyncio.run(loop.mg_loop())
    assert {e[2] for e in log if e[0] == "play_start"} == {"policy"}
    assert not {"lookahead", "policy_lag", "adapter_unloaded"} & set(loop.feed.kinds())
    assert "lookahead" not in json.loads((tmp_path / "out" / "mg_state.json").read_text())


# --- versioned adapters ----------------------------------------------------------------------------------------------

def test_names_stay_loaded_while_a_bench_uses_them(tmp_path, monkeypatch):
    log = []
    loop = la_loop(tmp_path, monkeypatch, log)

    async def run():
        await loop.mg_serve("/a/r05", 5)
        loop.mg_users = {"policy-r05": 1}  # a look-ahead bench still plays r05
        await loop.mg_serve("/a/r06", 6)
        assert ("server.unload", "policy-r05") not in log
        loop.mg_users["policy-r05"] = 0
        await loop.mg_serve("/a/r07", 7)

    asyncio.run(run())
    assert [e for e in log if e[0] == "server.unload"] == [("server.unload", "policy-r05"), ("server.unload", "policy-r06")]
    assert loop.mg_name() == "policy-r07" and loop.mg_loaded == {"policy-r07": "/a/r07"}
    assert loop.mg_play_version("policy-r05") == 5 and loop.mg_play_version("base") == 0


def test_mg_play_writes_the_version_and_releases_the_name(tmp_path, monkeypatch):
    log = []
    loop = la_loop(tmp_path, monkeypatch, log)
    from arena.rl_single import MultiMixin
    seen = {}

    async def bench(name, games, passes, budgets, **kw):
        seen.update(kw)
        assert loop.mg_users["policy-r05"] == 1
        return tmp_path, [], {"seconds": 0}

    loop.bench, loop.game_rows = bench, lambda rows, game: rows
    loop.mg_points = lambda spec, starts, name, harvest=False: (tmp_path / "p.json", [{}])

    async def run():
        await loop.mg_serve("/a/r05", 5)
        await loop.mg_serve("/a/r06", 6)  # r05 unused: unloaded at once
        loop.mg_users = {"policy-r05": 0}
        loop.mg_loaded["policy-r05"] = "/a/r05"  # loaded again for the bench below
        await MultiMixin.mg_play(loop, "r07-tr", {"game": "g", "cap": 10}, [], 1, model="policy-r05", seed=None,
                                 own={"server1_only": True})

    asyncio.run(run())
    assert seen["own"] == {"policy_version": 5, "server1_only": True} and "tail" not in seen
    assert loop.mg_users["policy-r05"] == 0 and "policy-r05" not in loop.mg_loaded  # released after the bench


def test_lag_counts_served_updates_and_drops_beyond_the_cap(tmp_path, monkeypatch):
    loop = la_loop(tmp_path, monkeypatch, [])

    async def serve():
        for v in (4, 5, 7, 8):  # r06 never served (a resume's first round has no update)
            await loop.mg_serve(f"/a/r{v:02d}", v)

    asyncio.run(serve())
    assert loop.mg_lineage == [4, 5, 7, 8]
    data = [(None, {"label": "a", "policy_version": 7}, 1.0, "k", "s1"),
            (None, {"label": "b", "policy_version": 5}, 1.0, "k", "s1"),
            (None, {"label": "c", "policy_version": 4}, 1.0, "k", "s2"),
            (None, {"label": "d"}, 1.0, "k", "s2"),  # no version: last round's adapter (lag 1)
            (None, {"label": "e", "policy_version": 6}, 1.0, "k", "s2")]  # off the lineage
    kept = loop.mg_lag_filter(9, data, 8)
    assert [x[1]["label"] for x in kept] == ["a", "b", "d"]
    assert loop.mg_lag_row == {"update": 9, "played_by": 8, "trained_by_lag": {"1": 2, "2": 1, "3": 1, "None": 1},
                               "dropped_by_set": {"s2": 2}, "cap": 2, "missing_version": 1,
                               "per_try": [["a", 1], ["b", 2], ["d", 1]]}
    asyncio.run(loop.mg_serve("/a/r05", 5))  # a rollback to r05: its later branch leaves the lineage
    assert loop.mg_lineage == [4, 5] and loop.mg_lag({"policy_version": 7}) is None


def test_ready_waits_for_every_launch_and_the_running_floor(tmp_path, monkeypatch):
    loop = la_loop(tmp_path, monkeypatch, [])
    pool = Pool()
    assert not loop.mg_lookahead_ready(0, None, 64)
    pool.pending = {"r06-tr": (0, 3)}
    assert not loop.mg_lookahead_ready(0, pool, 64)  # a bench still has tries to launch
    pool.pending, pool.used = {"r06-tr": (0, 0)}, 64
    assert not loop.mg_lookahead_ready(0, pool, 64)
    pool.used = 63
    assert loop.mg_lookahead_ready(0, pool, 64)
    monkeypatch.setattr(rl_single, "LOOKAHEAD_MIN_S", 60)
    assert not loop.mg_lookahead_ready(time.time(), pool, 64)  # the round's first minute


def test_kept_tries_take_the_root_version_unless_the_row_has_one(tmp_path):
    for root, rows in (("v5", [{"label": "a1", "outcome": "cleared"}]),
                       ("v6", [{"label": "a2", "outcome": "cleared", "policy_version": 4}])):
        for r in rows:
            (tmp_path / root / "b" / "attempts" / r["label"]).mkdir(parents=True)
            (tmp_path / root / "b" / "attempts" / r["label"] / "rollout.jsonl").write_text("{}\n")
        (tmp_path / root / "b" / "tries.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    kept = level_bench.kept_tries("b", [(str(tmp_path / "v5"), 5), [str(tmp_path / "v6"), 6]])
    assert {k: r["policy_version"] for k, (r, _) in kept.items()} == {"a1": 5, "a2": 4}
    assert "policy_version" not in level_bench.kept_tries("b", [tmp_path / "v5"])["a1"][0]


# --- the loop ----------------------------------------------------------------------------------------------------------

def test_lookahead_round_plays_the_served_adapter_and_trains_at_lag_two(tmp_path, monkeypatch):
    log = []
    loop = la_loop(tmp_path, monkeypatch, log)
    assert asyncio.run(loop.mg_loop()) == "max_rounds"
    # resume at r05 (rounds 6-11, U7-U11): round 7's training set starts during round 6 on r05 (server 1 only);
    # round 8's would lag 3 (round 6 had no update) and is skipped; from round 9 on every round has one
    tr = {n: starts_of(log, f"r{n:02d}-tr")[0][2:5] for n in range(6, 12)}
    assert tr == {6: ("policy-r05", 5, False), 7: ("policy-r05", 5, True), 8: ("policy-r05", 5, True),
                  9: ("policy-r07", 7, True), 10: ("policy-r08", 8, True), 11: ("policy-r09", 9, True)}
    assert log.index(starts_of(log, "r07-tr")[0]) < log.index(("play_end", "r06-tr"))  # overlapped with round 6
    assert [d["round"] for d in kinds(loop, "lookahead")] == [7, 8, 9, 10, 11]
    assert [d["round"] for d in kinds(loop, "lookahead_join")] == [7, 8, 9, 10, 11]
    assert kinds(loop, "lookahead_skip") == []
    # reads stay on the round's current adapter
    assert {n: starts_of(log, f"r{n:02d}-te")[0][2] for n in (8, 9, 10, 11)} == \
        {8: "policy-r07", 9: "policy-r08", 10: "policy-r09", 11: "policy-r10"}
    # U(n) trains round n-1; lag = updates served since the try's adapter (round 6 had none: r06 never existed)
    assert {e[1]: e[2] for e in log if e[0] == "update"} == {7: [5, 5], 8: [5, 5], 9: [5, 5], 10: [7, 7], 11: [8, 8]}
    assert {d["update"]: d["trained_by_lag"] for d in kinds(loop, "policy_lag")} == \
        {7: {"0": 2}, 8: {"1": 2}, 9: {"2": 2}, 10: {"2": 2}, 11: {"2": 2}}
    # a name leaves the server only after every bench on it has ended
    for i, e in enumerate(log):
        if e[0] == "server.unload":
            ends = [log.index(("play_end", s[1])) for s in log if s[0] == "play_start" and s[2] == e[1]]
            assert ends and max(ends) < i, e
    assert [e[1] for e in log if e[0] == "server.unload"] == [f"policy-r{n:02d}" for n in (5, 7, 8, 9, 10)]
    assert loop.picks == [(n, "tr") for n in range(6, 12)]  # one draw per round, before it when looked ahead
    rows = {r["round"]: r for r in loop.history}
    assert sorted(n for n, r in rows.items() if r.get("lag")) == [7, 8, 9, 10, 11]
    assert sorted(n for n, r in rows.items() if r.get("lookahead_joined")) == [7, 8, 9, 10, 11]
    launch = {n: r["lookahead"] for n, r in rows.items() if r.get("lookahead")}
    assert sorted(launch) == [6, 7, 8, 9, 10] and all(v["for_round"] == n + 1 for n, v in launch.items())
    assert all(v["tries"] == 2 and v["finished_by_round_end"] >= 1 and v["running_at_launch"] == 0
               and v["launched_at_s"] >= 0 for v in launch.values())
    assert all(r["lag"]["missing_version"] == 0 for r in rows.values() if r.get("lag"))
    assert rows[9]["lag"]["per_try"] == [["a1", 2], ["a2", 2]]
    assert json.loads((tmp_path / "out" / "mg_state.json").read_text())["lineage"] == [5, 7, 8, 9, 10, 11]


def test_lookahead_skips_when_its_tries_would_lag_past_the_cap(tmp_path, monkeypatch):
    log = []
    loop = la_loop(tmp_path, monkeypatch, log, mg={"lookahead_lag": 1})
    asyncio.run(loop.mg_loop())
    assert [d["round"] for d in kinds(loop, "lookahead")] == [7]  # round 6 has no update: lag 1
    assert [(d["round"], d["lag"]) for d in kinds(loop, "lookahead_skip")] == [(8, 2), (9, 2), (10, 2), (11, 2)]
    assert all(d["dropped_by_set"] == {} for d in kinds(loop, "policy_lag"))


def test_read_final_has_no_lookahead_into_the_reads_only_round(tmp_path, monkeypatch):
    log = []
    loop = la_loop(tmp_path, monkeypatch, log, mg={"read_final": True})
    asyncio.run(loop.mg_loop())
    assert [d["round"] for d in kinds(loop, "lookahead")] == [7, 8, 9, 10] and starts_of(log, "r11-tr") == []


def kill_when(loop, log, entry, after=0.05):
    async def main():
        task = asyncio.ensure_future(loop.mg_loop())
        while not any(e[:len(entry)] == entry for e in log):
            await asyncio.sleep(0.01)
        await asyncio.sleep(after)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await loop.mg_ahead_cancel("killed")  # the process dies with them

    asyncio.run(main())


def test_kill_with_lookahead_in_flight_then_resume_reuses_its_finished_tries(tmp_path, monkeypatch):
    log = []
    leg1 = tmp_path / "leg1"
    loop = la_loop(leg1, monkeypatch, log)
    loop.delays["r09-tr"] = 3.0  # round 9's look-ahead (r07) still plays when round 9 starts
    kill_when(loop, log, ("play_start", "r09-te"))  # killed in round 9, its look-ahead try a2 unfinished
    state = json.loads((leg1 / "out" / "mg_state.json").read_text())
    assert state["round"] == 8 and state["played_by"] == 8
    ahead = state["lookahead"]
    start = str(leg1 / "starts" / "att9")
    assert ahead["round"] == 9 and ahead["version"] == 7 and ahead["picks"] == {"tr": [[[start, 9]], 2]}
    assert ahead["key"] == f"p07-{ahead['uid']}" and ahead["uid"] != state["policy_uid"] and "tasks" not in ahead
    finished = leg1 / "out" / "partial" / ahead["key"] / "r09-tr" / "tries.jsonl"
    assert [json.loads(x)["label"] for x in finished.read_text().splitlines()] == ["a1"]
    feed = leg1 / "out" / "arena" / "trials" / "rl-loop" / "rl" / "feed.jsonl"
    feed.write_text(json.dumps({"kind": "round", "round": 8, "accepted_updates": 1,
                                "update": {"probes": {"p1": {"kl_base": 0.02}}}}) + "\n")

    moved = leg1 / "out" / "harvest" / "att9"  # a packed leg: the snapshot is read from the resumed run's copy
    moved.parent.mkdir(parents=True)
    shutil.move(start, moved)
    log2 = []
    loop2 = la_loop(tmp_path / "leg2", monkeypatch, log2, source=leg1 / "out")
    monkeypatch.setenv("ARC_RL1_RESUME", str(leg1 / "out"))
    assert asyncio.run(loop2.mg_loop()) == "max_rounds"
    assert [d["round"] for d in kinds(loop2, "lookahead_resume")] == [9]
    first9 = starts_of(log2, "r09-tr")[0]
    assert first9[2:4] == ("policy-r08", 8)  # replayed on the served adapter
    assert first9[5] == [("a1", 7)] and first9[6] == [[str(moved), 9]]  # same start; a1 reused at its own version, a2 again
    assert (9, "tr") not in loop2.picks and loop2.picks[0] == (10, "tr")  # the recorded picks, not a new draw
    assert {e[1]: e[2] for e in log2 if e[0] == "update"}[10] == [7, 8]
    assert [d["trained_by_lag"] for d in kinds(loop2, "policy_lag") if d["update"] == 10] == [{"0": 1, "1": 1}]


def test_lag_cost_splits_tis_and_clipping_by_lag():
    from arena.rl_single import lag_cost
    def g(r, tis, cap, clip, alt):
        return {"gspo": {"replies": r, "tis_mean": tis, "tis_capped": cap, "clipped": clip, "clipped_alt": {"0.004": alt}}}
    windows = [{"name": "t-a#0", "trained": 300, "signed": g(4, 1.0, 0, 1, 2)},
               {"name": "t-a#1", "trained": 100, "signed": g(2, 1.3, 1, 0, 1)},
               {"name": "t-b", "trained": 600, "signed": g(6, 1.1, 3, 3, 3)}]
    out = lag_cost(windows, {"t-a": 1, "t-b": 2})
    assert out["1"] == {"windows": 2, "tokens": 400, "token_share": 0.4, "replies": 6, "tis_mean": 1.1,
                        "tis_capped_share": 0.1667, "clip_share": 0.1667, "clip_share_0004": 0.5}
    assert out["2"]["token_share"] == 0.6 and out["2"]["tis_capped_share"] == 0.5 and out["2"]["clip_share"] == 0.5


def test_knob_rollback_cancels_the_lookahead_and_none_of_its_tries_train(tmp_path, monkeypatch):
    log = []
    loop = la_loop(tmp_path, monkeypatch, log)
    knobs = tmp_path / "knobs.json"
    knobs.write_text(json.dumps({"rollback": {"to": "r08", "after_round": 9}}))  # applies at round 10's top
    monkeypatch.setenv("ARENA_KNOBS", str(knobs))
    loop.uids = {}
    after = loop.mg_after_update

    async def after_update(number, row, update_row, serve, load=True):
        out = await after(number, row, update_row, serve, load)
        loop.uids[number] = loop.policy_uid
        return out

    async def rollback_to(number, to, played_by, probes, why):  # the real one's serving and uid steps
        back = int(to[1:])
        serve = str(loop.adapters / to)
        await loop.mg_serve(serve, back)
        loop.serving_adapter, loop.keep_checkpoint = serve, to
        loop.policy_uid = loop.uids[back]  # the state's recorded uid (same weights)
        loop.feed.emit("rollback", round=number, to=to)
        return back

    loop.mg_after_update, loop.mg_rollback_to = after_update, rollback_to
    assert asyncio.run(loop.mg_loop()) == "max_rounds"
    cancel = kinds(loop, "lookahead_cancel")
    assert [(d["round"], d["version"], d["why"]) for d in cancel] == [(10, 8, "knob rollback")]
    r10 = starts_of(log, "r10-tr")
    assert [e[2:5] for e in r10] == [("policy-r08", 8, True), ("policy-r08", 8, False)]  # cancelled, then replayed
    assert r10[1][5] == []  # this run's own keeps are never reused in-run: played fresh on the restored state
    assert ("play_end", "r10-tr") in log and log.count(("play_end", "r10-tr")) == 1  # the cancelled one never ended
    updates = {e[1]: e[2] for e in log if e[0] == "update"}
    assert 10 not in updates and updates[11] == [8, 8]  # round 9's tries dropped with the rollback; U11: the replay
    assert loop.mg_lineage[:3] == [5, 7, 8] and 9 not in loop.mg_lineage  # r09 left the lineage
    assert loop.mg_lag({"policy_version": 9}) is None  # a kept r09 try would never train
    assert loop.mg_lag({"policy_version": 8}) is not None  # r08's (the cancelled look-ahead's) are back on it


def test_start_tag_is_the_volume_path_for_an_unpacked_leg():
    from arena.rl_single import MultiMixin
    spec = {"level": 2}
    assert MultiMixin.mg_tag(spec, "/tmp/validation/preflight/x/harvest/a", 7) == \
        MultiMixin.mg_tag(spec, "/runs/validation/preflight/x/harvest/a", 7)
    assert MultiMixin.mg_tag(spec, "/runs/validation/y", 7) != MultiMixin.mg_tag(spec, "/runs/validation/z", 7)


def test_plan_digest_reads_an_unpacked_leg_path_as_its_volume_path():
    def pt(root):
        return {"id": "x", "restore": {"attempt": f"{root}/validation/preflight/a/harvest/h", "request": 3}}
    assert level_bench.plan_digest("a1", pt("/tmp"), 100, 10) == level_bench.plan_digest("a1", pt("/runs"), 100, 10)
    assert level_bench.plan_digest("a1", pt("/runs"), 100, 10) != level_bench.plan_digest("a2", pt("/runs"), 100, 10)
