import json
import os
import random

import pytest

from rsi_loop import candidates, policy, proxy, registry
from rsi_loop.score import level_actions_from_record, rhae


def test_rhae_official():
    assert rhae([10, 10, 10], [10, 10, 10]) == 100.0
    assert abs(rhae([10, 10, 10], [10]) - 100 / 6) < 1e-9 and rhae([10, 10], []) == 0.0
    assert rhae([10], [5]) == 100.0  # per-level cap 115, game score clipped to the cleared share


def test_record_level_actions():
    rows = [{"event": "start", "levels_completed": 0}]
    rows += [{"n": i, "levels_completed": 1 if i >= 5 else 0} for i in range(1, 8)]
    rows += [{"n": 8, "levels_completed": 2, "state": "WIN"}]
    assert level_actions_from_record(rows) == [5, 3]


def test_policy_decide():
    rng = random.Random(0)
    games = [f"g{i}" for i in range(20)]
    cands = {"c0000": {"status": "done", "pool": {g: 10.0 + i for i, g in enumerate(games[:15])}, "live": True},
             "c0001": {"status": "done", "parent": "c0000", "pool": {g: 20.0 + i for i, g in enumerate(games[:15])},
                       "live": True},
             "c0002": {"status": "new", "parent": "c0001", "pool": {}, "live": True}}
    out = policy.decide({"candidates": cands, "pool_games": games, "practice": {}, "practice_games": ["p1", "p2"],
                         "seed": 1})
    assert out["parent"] in ("c0000", "c0001")
    first = [e for e in out["evaluate"] if e["candidate"] == "c0002"]
    assert first and len(first[0]["games"]) == 15 and out["practice"][0]["games"]
    stand = policy.standing(cands)
    assert stand["c0001"]["effect"] > stand["c0000"]["effect"]
    assert len(policy.minibatch(games, cands["c0000"]["pool"], 15, rng)) == 5


def test_candidate_check(tmp_path):
    tree = tmp_path / "c"
    (tree / "player/workspace").mkdir(parents=True)
    (tree / "serving").mkdir()
    (tree / "player/game-prompt.txt").write_text("play")
    (tree / "player/run.json").write_text(json.dumps({"argv": ["node", "x"], "concurrency": 14}))
    (tree / "serving/server.json").write_text("{}")
    assert candidates.check(tree) == []
    (tree / "player/run.json").write_text(json.dumps({"argv": [], "concurrency": 0}))
    assert len(candidates.check(tree)) == 2
    before = candidates.serving_hash(tree)
    (tree / "serving/env.json").write_text("{}")
    assert candidates.serving_hash(tree) != before


def test_proxy_usage_parse():
    sse = (b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
           b'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\ndata: [DONE]\n\n')
    usage, finish = proxy._usage(sse)
    assert usage["completion_tokens"] == 3 and finish == "stop"
    usage, finish = proxy._usage(b'{"choices":[{"finish_reason":"length"}],"usage":{"completion_tokens":5}}')
    assert usage["completion_tokens"] == 5 and finish == "length"


def test_roles_no_overlap():
    vet = {"ok": True}
    games = [{"game_id": "tw01-a", "role": "pool", "vet_ok": True, "vet": {**vet, "random_level1_wins": 0}},
             {"game_id": "tw02-a", "role": "pool", "vet_ok": True, "vet": {**vet, "random_level1_wins": 2}},
             {"game_id": "wp01-a", "role": "practice", "vet_ok": True, "vet": vet},
             {"game_id": "wp02-a", "role": "practice", "vet_ok": True, "vet": vet},
             {"game_id": "sg01-a", "role": "pool", "vet_ok": True, "vet": {**vet, "random_level1_wins": 1}}]
    pool, practice, dropped = registry.roles({"games": games}, "level1", "split")
    assert [e["game_id"] for e in pool] == ["tw01-a"]
    assert sorted(e["game_id"] for e in practice) == ["tw02-a", "wp02-a"]
    assert sorted(dropped) == ["sg01-a", "tw02-a"]


def test_policy_ranges():
    with pytest.raises(ValueError):
        policy.check_policy({"minibatch": 0})


def test_sandbox_script():
    from rsi_loop import sandbox

    argv = sandbox.command(["node", "x"], "rsip00", "/rsi/slots/00", {"A": "1"},
                           extra_binds=[("/a", "/home/session", False), ("/b", "/candidate", True)],
                           hide=["/rsi", "/tmp/arena"], net="proxy", proxy=("/rsi/psock", 18080))
    script = argv[-1]
    assert argv[:4] == ["unshare", "--mount", "--ipc", "--net"]
    assert "rsi-tmp /tmp" in script and "rsi-shm /dev/shm" in script and "rsi-vartmp /var/tmp" in script
    assert "rsi-hidden /rsi" in script and "rsi-hidden /tmp/arena" not in script
    assert "remount,bind,ro \"$m\"" in script and "/proxy/proxy.sock 18080" in script
    host = sandbox.command(["x"], "rsiserve", None, {}, net="host", tmp="/rsi/serve/tmp",
                           tmp_binds=[("/tmp/arena/serving", "arena/serving", False)])
    assert "--net" not in host and "mount --rbind /rsi/serve/tmp /tmp" in host[-1]
    assert host[-1].index("/rsi/serve/tmp/arena/serving") < host[-1].index("--rbind")
    nested = sandbox.command(["x"], "rsip00", "/rsi/slots/00", {}, hide=["/assets", "/assets/games"], net="none")[-1]
    assert "rsi-hidden /assets" in nested and "rsi-hidden /assets/games" not in nested  # v13: no mount point left


def test_policy_counts_added_games_played():
    cands = {"c0000": {"status": "done", "pool": {"g0": 1.0}, "pool_levels": {"g0": 1, "zz": 2}, "live": True}}
    out = policy.decide({"candidates": cands, "pool_games": ["g0", "zz", "g1"], "practice": {}, "practice_games": [],
                         "seed": 0})
    assert [e["games"] for e in out["evaluate"]] == [["g1"]]


def test_concurrency_cap(tmp_path):
    tree = tmp_path / "c"
    (tree / "player/workspace").mkdir(parents=True)
    (tree / "serving").mkdir()
    (tree / "player/game-prompt.txt").write_text("play")
    (tree / "serving/server.json").write_text("{}")
    for value, ok in ((97, True), (98, False), (1, False)):
        (tree / "player/run.json").write_text(json.dumps({"argv": ["node"], "concurrency": value}))
        assert (candidates.check(tree) == []) == ok
    for extra, ok in (({"game_seconds": 3600}, True), ({"game_seconds": 30}, False), ({"game_seconds": 30000}, False),
                      ({"game_token_cap": 400000}, True), ({"game_token_cap": 5000}, False),
                      ({"game_token_cap": 1.5e5}, False)):
        (tree / "player/run.json").write_text(json.dumps({"argv": ["node"], **extra}))
        assert (candidates.check(tree) == []) == ok, extra


def test_job_wall_by_candidate():
    from types import SimpleNamespace

    from rsi_loop.loop import Loop

    fake = SimpleNamespace(state={"concurrency": 14, "candidates": {"a": {"concurrency": 28}, "b": {},
                                                                    "c": {"game_seconds": 4000, "game_token_cap": 50000}}},
                           cfg={"slice_seconds": 10.0, "smoke_seconds": 240})
    fake.concurrency = lambda: Loop.concurrency(fake)
    fake.job_concurrency = lambda job: Loop.job_concurrency(fake, job)
    assert Loop.job_wall(fake, {"kind": "pool", "candidate": "a"}) == 280.0
    assert Loop.job_wall(fake, {"kind": "pool", "candidate": "b"}) == 140.0
    assert Loop.job_wall(fake, {"kind": "smoke", "candidate": "a"}) == 240.0
    assert Loop.job_wall(fake, {"kind": "pool", "candidate": "c"}) == 4000.0  # the candidate's own length
    assert Loop.job_wall(fake, {"kind": "practice", "candidate": "c", "seconds": 600}) == 600.0
    fake.cfg["heldout_token_cap"] = 100000
    assert Loop.job_token_cap(fake, {"kind": "pool", "candidate": "c"}) == 50000
    assert Loop.job_token_cap(fake, {"kind": "pool", "candidate": "b"}) is None
    assert Loop.job_token_cap(fake, {"kind": "heldout_a", "candidate": "c"}) == 100000
    assert Loop.job_token_cap(fake, {"kind": "smoke", "candidate": "c"}) is None


def test_proxy_allowlist():
    assert ("POST", "/v1/chat/completions") in proxy.ALLOWED and ("POST", "/collective_rpc") not in proxy.ALLOWED


def test_escaping_links(tmp_path):
    import os

    root = tmp_path / "c"
    (root / "a").mkdir(parents=True)
    os.symlink(".", root / "s")              # points at the tree itself
    os.symlink("s/..", root / "up")           # lexically inside, physically the tree's parent
    os.symlink("../a", root / "a/ok")
    os.symlink("/etc/passwd", root / "abs")
    bad = candidates.escaping_links(root)
    assert "abs" in bad and "up" in bad and "a/ok" not in bad and "s" not in bad
    with pytest.raises((OSError, ValueError)):
        candidates.copy_regular(root / "abs", tmp_path / "x")
    os.mkfifo(root / "fifo")
    with pytest.raises(ValueError):
        candidates.copy_regular(root / "fifo", tmp_path / "y")


def test_report_paired():
    from rsi_loop import report

    rows = []
    for run, bump in (("h1", 0), ("h2", 1)):
        for game in ("g1", "g2", "g3"):
            for seed in range(3):
                rows.append({"run": run, "kind": "heldout_a", "game": game, "seed": seed, "levels": 1 + bump,
                             "score": 10.0 + 10 * bump + seed})
    runs = [{"id": "h1", "candidate": "c0000"}, {"id": "h2", "candidate": "c0005"}]
    out = report.point_report(50, runs, rows, {"heldout_a": {"g1": 3, "g2": 3, "g3": 3}})["sets"]["heldout_a"]
    assert out["paired_rhae"]["diff"] == 10.0 and out["paired_levels"]["diff"] == 1.0
    assert out["paired_by_level"]["diff"] == 1.0 and out["gen0"]["rhae"] == 11.0


def test_paired_minibatch_and_pool_rule():
    games = [f"g{i}" for i in range(30)]
    first = policy.minibatch(games, [], 15, random.Random(1), paired=True)
    assert first == policy.minibatch(games, [], 15, random.Random(2), paired=True)
    assert policy.minibatch(games, first, 15, random.Random(3), paired=True) == [
        g for g in policy.minibatch(games, [], 30, random.Random(0), paired=True) if g not in first]
    vet = {"random_level1_wins": 2, "random_level2_wins": 0}
    assert registry.trivial(vet, "level1") and not registry.trivial(vet, "level2")
    with pytest.raises(ValueError):
        registry.trivial({"random_level1_wins": 0}, "level2")


def test_witness_modes():
    vet = lambda wins2: {"random_level1_wins": 3, "random_level2_wins": wins2}  # noqa: E731
    reg = {"games": [{"game_id": "sg01-a", "role": "pool", "vet_ok": True, "vet": vet(0)},
                     {"game_id": "sg02-a", "role": "pool", "vet_ok": True, "vet": vet(1)},
                     {"game_id": "tw01-a", "role": "pool", "vet_ok": True, "vet": vet(0)},
                     {"game_id": "tw02-a", "role": "pool", "vet_ok": True, "vet": vet(2)},
                     {"game_id": "wp01-a", "role": "practice", "vet_ok": True, "vet": vet(0)}]}
    ids = lambda rows: sorted(e["game_id"] for e in rows)  # noqa: E731
    pool, practice, dropped = registry.roles(reg, "level2", "practice")
    assert ids(pool) == ["sg01-a"] and ids(practice) == ["tw01-a", "tw02-a", "wp01-a"]
    pool, practice, _ = registry.roles(reg, "level2", "split")
    assert ids(pool) == ["sg01-a", "tw01-a"] and ids(practice) == ["tw02-a"]
    assert registry.roles(reg, "level2", "drop")[1] == []



def test_untrusted_tree_ops(tmp_path):
    from rsi_loop import candidates as cand

    secret = tmp_path / "secret"
    (secret / "sub").mkdir(parents=True)
    (secret / "sub/held.txt").write_text("held-out")
    ws = tmp_path / "ws"
    (ws / "work/real").mkdir(parents=True)
    (ws / "work/real/a.txt").write_text("a")
    os.symlink(secret, ws / "work/top")
    os.symlink(secret / "sub", ws / "work/real/inner")
    with pytest.raises(OSError):
        cand.copy_untrusted(ws, "work/top", tmp_path / "c1")
    assert not (tmp_path / "c1/sub").exists()
    cand.copy_untrusted(ws, "work/real", tmp_path / "c2")
    assert (tmp_path / "c2/a.txt").read_text() == "a" and (tmp_path / "c2/inner").is_symlink()
    os.symlink(secret / "sub/held.txt", ws / "events.jsonl")
    cand.write_untrusted(ws, "events.jsonl", b"row\n", append=True)
    assert (secret / "sub/held.txt").read_text() == "held-out" and (ws / "events.jsonl").read_text() == "row\n"
    with pytest.raises(ValueError):
        cand.open_dir(ws, "work/../..")
    assert cand.read_untrusted(ws, "work/real/a.txt") == b"a"
    with pytest.raises(OSError):
        cand.read_untrusted(ws, "work/real/inner/held.txt")


def test_check_decision_and_robust_gen0():
    from rsi_loop.loop import Loop

    view = {"candidates": {"c0000": {"status": "done", "parent": None, "pool": {"g1": 1.0}, "live": True,
                                     "queued": 0},
                           "c0001": {"status": "new", "parent": ["bad"], "pool": {}, "live": True, "queued": 0}},
            "pool_games": ["g1", "g2"], "practice_games": ["p1"], "practice": {}, "seed": 1}
    ok = Loop.check_decision(None, {"parent": "c0000", "evaluate": [{"candidate": "c0001", "games": ["g1", "g1"]}],
                                    "best": "c0000"}, view)
    assert ok["evaluate"][0]["games"] == ["g1"] and ok["cancel"] == [] and ok["best"] == "c0000"
    for bad in ([], {"parent": ["x"]}, {"evaluate": [{"candidate": "c9", "games": []}]},
                {"evaluate": [{"candidate": "c0001", "games": ["nope"]}]}, {"cancel": "c0001"}, {"best": 3}):
        with pytest.raises(ValueError):
            Loop.check_decision(None, bad, view)
    Loop.check_decision(None, policy.decide(view), view)  # gen 0 survives a non-string parent


class StubProposer:
    def __init__(self, decision=None, error=None):
        self.decision, self.error, self.notes, self.paused = decision, error, [], False
        self.process = None

    async def policy(self, view, meta_dir=None):
        return self.decision, self.error

    def notify(self, row):
        self.notes.append(row)

    async def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


class StubStacks:
    def __init__(self, live="new"):
        self.live, self.calls = live, []

    async def switch(self, new_dir, new_tag, old_dir, old_tag, gpu_job=None, fallback=None):
        self.calls.append((new_tag, old_tag, fallback is not None))
        return {"from": old_tag, "to": new_tag, "ok": self.live == "new", "live": self.live, "seconds": 0.0}


def make_loop(tmp_path, extra=None):
    import time

    from rsi_loop import loop as L

    src = tmp_path / "bundle"
    (src / "configs/rsi").mkdir(parents=True)
    sets = {"pool": ["pa1-v1", "pa2-v1", "pa3-v1"], "practice": ["pb1-v1"], "heldout_a": ["pc1-v1"],
            "heldout_b": ["pd1-v1"]}
    for role, games in sets.items():
        (src / f"configs/rsi/{role}_points.json").write_text(json.dumps(
            {"points": [{"game": g, "total_levels": 2, "baselines": [4, 24]} for g in games]}))
    study = {"served_model": "m", "server": {}, "rsi_loop": {"root": str(tmp_path / "rsi"), "best_min_games": 2, "heldout_mode": "local",
                                                        **(extra or {})}}
    (src / "configs/comparison.json").write_text(json.dumps(study))
    loop = L.Loop(study, {}, source=str(src), games_dir=None, output=str(tmp_path / "out"),
                  server_dir=str(tmp_path / "srv"), base_url="http://127.0.0.1:9/v1", deadline_epoch=time.time() + 9e4,
                  log=lambda *a, **k: None)
    loop.proposer, loop.stacks = StubProposer(), StubStacks()
    cand = lambda h, status, pool: {"parent": None, "status": status, "serving_hash": h, "concurrency": 4,  # noqa
                                    "pool": pool, "pool_levels": {}, "problems": []}
    loop.state["candidates"] = {"c0000": cand("h0", "done", {"pa1-v1": 10.0, "pa2-v1": 10.0}),
                                "c0001": cand("h1", "done", {"pa1-v1": 90.0, "pa2-v1": 90.0}),
                                "c0002": cand("h0", "new", {})}
    loop.state["stack"].update(live="c0000", live_hash="h0")
    loop.state["meta"]["live"] = "m0000"
    return loop


def test_loop_policy_fallback_and_cancel(tmp_path):
    import asyncio

    loop = make_loop(tmp_path)
    loop.proposer.decision = {"parent": ["not", "an", "id"]}
    asyncio.run(loop.run_policy())
    assert loop.state["policy_fallback"]["count"] == 1 and loop.proposer.notes[-1]["event"] == "policy_fallback"
    assert any(j["candidate"] == "c0002" and j["kind"] == "pool" for j in loop.state["jobs"])  # gen 0 decided
    assert loop.state["candidates"]["c0002"]["status"] == "evaluating"
    loop.proposer.decision, loop.proposer.error = {"cancel": ["c0002"], "best": "c0001"}, None
    asyncio.run(loop.run_policy())
    assert not any(j["candidate"] == "c0002" and j["kind"] == "pool" for j in loop.state["jobs"])
    assert loop.state["candidates"]["c0002"]["status"] == "done" and loop.best() == "c0001"


def test_loop_infra_failure_ends_evaluation_and_best_across_stacks(tmp_path):
    import asyncio

    loop = make_loop(tmp_path)
    assert loop.best() == "c0001"  # on another stack than the live one
    loop.state["candidates"]["c0002"]["status"] = "evaluating"
    job = {"id": "j1", "kind": "pool", "candidate": "c0002", "game": "pa3-v1", "seed": 0, "retries": 2}
    asyncio.run(loop.done(job, {"outcome": "error", "infrastructure": True}))
    assert loop.state["candidates"]["c0002"]["status"] == "done" and not loop.state["jobs"][1:]


def test_loop_probation_revert_and_fallback_switch(tmp_path):
    import asyncio

    loop = make_loop(tmp_path)
    loop.request_switch("c0001", None, purpose="proposer")
    asyncio.run(loop.maybe_switch())
    assert loop.state["stack"]["live"] == "c0001" and loop.state["stack"]["probation"]["back_to"] == "c0000"
    for _ in range(6):
        loop.probation_note(game={"tokens": 0})
    assert loop.state["pending_switch"]["purpose"] == "revert" and loop.state["pending_switch"]["candidate"] == "c0000"
    assert loop.proposer.notes[-1]["event"] == "stack_reverted"
    loop.state["pending_switch"] = None
    loop.stacks = StubStacks(live="fallback")
    loop.request_switch("c0001", None, purpose="proposer")
    asyncio.run(loop.maybe_switch())
    assert loop.state["stack"]["live"] == "c0000" and loop.stacks.calls[-1][2]


def test_loop_heldout_skip_exclusive_and_drain_meter(tmp_path):
    loop = make_loop(tmp_path)
    run = loop.open_heldout("c0001", ("heldout_a",), 1)
    loop.heldout_switch_failed("c0001")
    assert any(j.get("run") == run["id"] for j in loop.state["jobs"])
    loop.heldout_switch_failed("c0001")
    assert not any(j.get("run") == run["id"] for j in loop.state["jobs"]) and run["skipped"]
    loop.open_heldout("c0000", ("heldout_a",), 1)
    assert loop.trusted_live() and loop.heldout_exclusive()  # gen 0's held-out games play alone too
    loop.state["heldout_hold"] = "held"
    loop.running = {"j9": ({"kind": "pool", "candidate": "c0000"}, None)}
    before = dict(loop.state["meters"])
    loop.last_tick -= 10
    loop.tick_meters()
    assert loop.state["meters"]["heldout_s"] == before["heldout_s"]  # draining loop games stay on the loop meter


def test_loop_reverted_stack_ineligible_until_probation_passes(tmp_path):
    import asyncio

    loop = make_loop(tmp_path)
    loop.state["candidates"]["c0003"] = {**loop.state["candidates"]["c0001"], "pool": {"pa1-v1": 50.0, "pa2-v1": 50.0},
                                         "serving_hash": "h3"}
    run = loop.open_heldout("c0001", ("heldout_a",), 1)
    loop.state["policy_best"] = "c0001"
    loop.request_switch("c0001", None, purpose="proposer")
    asyncio.run(loop.maybe_switch())
    for _ in range(6):
        loop.probation_note(game={"tokens": 0})
    assert loop.stack_reverted("c0001") and not loop.stack_reverted("c0000")
    assert loop.best() == "c0003"  # the policy's pick is ineligible; next eligible by standing
    assert not any(j["candidate"] == "c0001" and j["kind"].startswith("heldout") for j in loop.state["jobs"])
    assert run["skipped"] and loop.view()["candidates"]["c0001"]["stack_reverted"]
    loop.state["candidates"]["c0003"]["serving_hash"] = "h1"  # same stack: ineligible too; gen 0 is left
    assert loop.best() == "c0000"
    loop.state["candidates"]["c0003"]["serving_hash"] = "h3"
    asyncio.run(loop.maybe_switch())  # the revert
    loop.request_switch("c0001", None, purpose="proposer")  # adopted again: passes this time
    asyncio.run(loop.maybe_switch())
    for _ in range(20):
        loop.probation_note(game={"tokens": 5})
    assert not loop.stack_reverted("c0001") and loop.best() == "c0001"


def test_fit_slots_and_gen0_serving_study():
    from rsi_loop.serving import fit_slots

    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src/rsi_loop/gen0/serving"
    study = {"generation_slots": 52, **json.loads((root / "study.json").read_text()),
             "server": json.loads((root / "server.json").read_text())}
    assert study["generation_slots"] <= study["server"]["max_num_seqs"]  # gen 0 starts without the clamp
    low = fit_slots({"generation_slots": 52, "server": {"max_num_seqs": 8}})
    assert low["generation_slots"] == 8
    assert fit_slots({"generation_slots": 4, "server": {"max_num_seqs": 8}})["generation_slots"] == 4


def test_slim_session_log():
    from rsi_loop.attempt import slim_line

    big = {"type": "message_update", "message": {"content": "x" * 5000},
           "assistantMessageEvent": {"type": "thinking_delta", "delta": "ab", "partial": {"content": "x" * 5000}}}
    out = json.loads(slim_line((json.dumps(big) + "\n").encode()))
    assert out == {"type": "message_update", "assistantMessageEvent": {"type": "thinking_delta", "delta": "ab"}}
    end = (json.dumps({"type": "message_end", "message": {"content": "y"}}) + "\n").encode()
    assert slim_line(end) == end and slim_line(b"not json\n") == b"not json\n"


def test_time_feedback_and_loop_facts(tmp_path):
    import time

    from rsi_loop import feedback, proposer

    loop = make_loop(tmp_path)
    games = [f"g{i}" for i in range(10)]
    loop.state["candidates"]["c0000"]["pool"] = {g: float(10 * i) for i, g in enumerate(games)}
    loop.state["candidates"]["c0001"]["pool"] = {g: float(10 * i + (5 if i % 2 else -3)) for i, g in enumerate(games)}
    loop.state["candidates"]["c0001"]["pool_runs"] = {"g1": [10.0, 20.0]}
    now = time.time()
    loop.state["rows"] = [{"kind": "smoke", "candidate": "c0001", "seconds": 240.0, "wall": 240.0, "tokens": 9,
                           "started": now - 300, "ended": now - 60},
                          {"kind": "pool", "candidate": "c0001", "seconds": 600.0, "wall": 600.0, "tokens": 900,
                           "started": now - 900, "ended": now - 300},
                          {"id": "j9", "kind": "practice", "candidate": "c0001", "outcome": "error",
                           "error": "FileNotFoundError", "infrastructure": True, "seconds": None}]  # 13 Oct: broke status
    out = feedback.feedback(loop.state, loop.cfg, loop.concurrency(), loop.slots(), 7200.0, now)
    json.dumps(out)
    assert out["stages"]["smoke"]["share_at_wall"] == 1.0 and out["stages"]["pool"]["games"] == 1
    noise = out["noise"]
    assert noise["paired_differences"] == 10 and noise["paired_sd"] and noise["repeat_sd"] == 7.1
    assert noise["vs_c0000"]["c0001"]["paired_mean"] == 1.0 and 0 < noise["null_clears"]["+3 on 8"] < 0.5
    (tmp_path / "m").mkdir()
    (tmp_path / "m/AGENTS.md").write_text("mine\n")
    assert proposer.meta_file(tmp_path / "m", "AGENTS.md").decode().startswith("mine\n\n## Loop facts")
    (tmp_path / "m/AGENTS.md").write_text("2. Look. A\n   candidate with one change tells you what that change is worth;"
                                          " several changes at once are allowed. Keep.\n")
    text = proposer.meta_file(tmp_path / "m", "AGENTS.md").decode()
    assert "one change tells" not in text and "fewer changes are easier to attribute" in text.split("## Loop")[0]
    job = {"kind": "practice", "candidate": "c0000", "seconds": 30}
    assert loop.job_wall(job) == 60.0 and loop.job_wall({**job, "seconds": 1e6}) == loop.job_wall(
        {"kind": "pool", "candidate": "c0000"})


def test_gate_metrics_and_meta_history(tmp_path):
    import asyncio

    from rsi_loop import feedback
    from rsi_loop.proxy import Gate

    async def run():
        gate, order = Gate(100), []

        async def req(name, cost, hold):
            await gate.acquire(cost)
            order.append(name)
            await asyncio.sleep(hold)
            gate.release(cost)

        await asyncio.gather(req("a", 80, 0.05), req("b", 50, 0.01), req("c", 10, 0.01), req("huge", 500, 0.01))
        return order, gate.view()

    order, view = asyncio.run(run())
    assert order[0] == "a" and order[1] == "b" and set(order) == {"a", "b", "c", "huge"}  # FIFO, alone always fits
    assert view["reserved_tokens"] == 0 and view["in_flight"] == 0
    text = ('vllm:num_requests_running{model_name="m"} 8.0\nvllm:kv_cache_usage_perc{model_name="m"} 0.41233\n'
            'vllm:cache_config_info{block_size="8",kv_cache_size_tokens="1720668",num_gpu_blocks="617"} 1.0\n'
            'vllm:generation_tokens_total{model_name="m"} 6.1e+06\n')
    m = feedback.server_metrics(text)
    assert m == {"running": 8.0, "kv_use": 0.412, "kv_capacity_tokens": 1720668, "kv_blocks": 617,
                 "generation_tokens_total": 6.1e6}
    root = tmp_path / "meta"
    for mid, text in (("m0", "a\nb\n"), ("m1", "a\nB\n")):
        (root / mid).mkdir(parents=True)
        (root / mid / "policy.py").write_text(text)
    state = {"meta": {"items": {"m0": {"created": 1, "status": "previous", "parent": None},
                                "m1": {"created": 5, "status": "live", "parent": "m0", "note": "why"}}},
             "proposer": {"sessions": [{"meta": "m1", "tag": "session", "started": 6, "ended": 3606,
                                        "submitted": 2, "reason": "requested"}]},
             "candidates": {"c0000": {"status": "done", "note": "", "pool": {"g": 1.0}},
                            "c0001": {"status": "held", "note": "x", "author": "proposer", "created": 7,
                                      "pool": {"g": 3.0}}}}
    hist = feedback.meta_history(root, state)
    assert hist[1]["note"] == "why" and "+B" in hist[1]["changed"]["policy.py"]
    assert hist[1]["sessions"]["hours"] == 1.0 and "c0001" in hist[1]["candidates"] and not hist[0]["candidates"]


def test_proposer_status_time_and_end_session(tmp_path, monkeypatch):
    import asyncio
    import pwd
    from types import SimpleNamespace

    from rsi_loop import proposer as P

    loop = make_loop(tmp_path)
    monkeypatch.setattr(pwd, "getpwnam", lambda name: SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid()))
    host = P.ProposerHost(loop)
    (loop.root / "meta/m0000").mkdir(parents=True)
    (loop.root / "meta/m0000/settings.json").write_text('{"session_seconds": 3600}')
    loop.state["meta"]["items"] = {"m0000": {"created": 1, "status": "live", "parent": None}}
    for item in loop.state["candidates"].values():
        item.update(note="n", created=1.0)
    out = asyncio.run(host.op_status())
    assert out["time"]["stages"]["proposer_session"]["setting_minutes"] == 60.0 and "error" in out["time"]["server_now"]
    assert asyncio.run(host.op_end_session())["ending"] and host.end_requested
    assert asyncio.run(host.op_meta_history())[0]["meta"] == "m0000"


def test_second_resume_keeps_restored_trees(tmp_path, monkeypatch):
    loop = make_loop(tmp_path)
    monkeypatch.setattr(loop.runtime, "seal", lambda folder: None)
    gen0 = loop.root / "archive/c0000/tree"
    (gen0 / "player/prime-agent/node_modules").mkdir(parents=True)
    resume = tmp_path / "resume"
    (resume / "archive/c0001/tree/player/workspace").mkdir(parents=True)
    (resume / "archive/c0001/tree/player/workspace/AGENTS.md").write_text("x")
    (resume / "archive/c0002/practice").mkdir(parents=True)  # a tree the earlier leg failed to checkpoint
    loop.state["jobs"] = [{"id": "j1", "kind": "pool", "candidate": "c0002", "game": "pa1-v1", "seed": 0}]
    loop.runtime.restore(loop.ckpt, resume, loop.state)
    assert (loop.root / "archive/c0001/tree/player/workspace/AGENTS.md").exists()
    assert loop.state["candidates"]["c0002"]["status"] == "unrestorable" and not loop.state["jobs"]
    for cid in ("c0001", "c0002"):
        loop.mirror_candidate(cid)
    assert (loop.ckpt / "archive/c0001/tree/player/workspace/AGENTS.md").exists()  # the next leg can restore it
    assert not (loop.ckpt / "archive/c0002/tree").exists()


def test_fixed_pool_games(tmp_path):
    loop = make_loop(tmp_path)
    loop.fixed_pool = ["pa3-v1"]
    loop.cfg["best_min_games"] = 2
    loop.points.setdefault("practice", {})["vg99-v1"] = {"game": "vg99-v1", "total_levels": 2}
    loop.start_fixed()
    assert "vg99-v1" in loop.state["practice_games"]  # bundle practice games join a resumed state
    assert loop.state["fixed_rescore"] == ["c0000", "c0001"]  # the best by the earlier rule is re-scored too
    loop.state["candidates"]["c0002"]["status"] = "new"
    loop.ensure_fixed()
    fixed = sorted(j["candidate"] for j in loop.state["jobs"] if j.get("fixed"))
    assert fixed == ["c0000", "c0001", "c0002"] and loop.state["candidates"]["c0001"]["status"] == "evaluating"
    assert loop.state["candidates"]["c0002"]["status"] == "new" and loop.view()["candidates"]["c0002"]["queued"] == 0
    assert loop.best() is None  # nobody has the fixed game yet
    loop.apply_decision({"parent": None, "best": None, "evaluate": [], "practice": [], "cancel": ["c0001"]},
                        loop.view())
    assert any(j["candidate"] == "c0001" and j.get("fixed") for j in loop.state["jobs"])  # not cancellable
    loop.state["candidates"]["c0001"]["pool"]["pa3-v1"] = 95.0
    assert loop.best() == "c0001"


def test_heldout_token_format_and_budget_stop(tmp_path):
    from rsi_loop import feedback

    loop = make_loop(tmp_path)
    old = loop.open_heldout("c0000", ("heldout_a",), 1)
    old.pop("format")
    old["open"] = False
    loop.state["jobs"] = []
    for run in loop.state["heldout"]["runs"]:  # the migration setup() applies
        if "format" not in run:
            run.update(format="time-slice", note="old")
    loop.state["meters"]["loop_s"] = 10.5 * 3600
    loop.heldout_due()  # gen 0's first point in the new format, although an old-format gen 0 run exists
    new = [r for r in loop.state["heldout"]["runs"] if r.get("format") == "tokens100k"]
    assert len(new) == 1 and new[0]["candidate"] == "c0000"
    job = next(j for j in loop.state["jobs"] if j["kind"] == "heldout_a")
    assert loop.job_wall(job) == 3 * 3600
    assert "ACTION5" not in feedback.LOOP_FACTS and "sokoban" not in feedback.LOOP_FACTS
    assert "not wall time" in feedback.LOOP_FACTS
    loop.state["budget_reached"] = 1.0
    loop.state["jobs"].append({"id": "jp", "kind": "pool", "candidate": "c0000", "game": "pa3-v1", "seed": 0,
                               "queued": 0.0})
    loop.state["heldout_hold"] = "held"
    assert loop.next_job()["kind"] == "heldout_a"  # at the limit only the due held-out point plays
    loop.state["jobs"] = [j for j in loop.state["jobs"] if j["kind"] == "pool"]
    assert loop.next_job() is None


def test_proxy_token_cap(tmp_path):
    import asyncio

    p = proxy.Proxy("http://127.0.0.1:9", "m", tmp_path / "px", 0)
    p.register("k", "s1", "heldout", token_cap=100)
    p.totals["s1"]["completion_tokens"] = 100
    body = json.dumps({"model": "m", "messages": [], "max_tokens": 500}).encode()

    class Req:
        headers = {"Authorization": "Bearer k"}
        match_info = {"tail": "v1/chat/completions"}
        method, content_type, query_string, path = "POST", "application/json", "", "/v1/chat/completions"

        async def read(self):
            return body
    resp = asyncio.run(p.handle(Req()))
    assert resp.status == 429 and p.totals["s1"]["capped"] == 1


def test_session_end_detected_after_compaction(tmp_path):
    from rsi_loop.attempt import Attempt

    log = tmp_path / "session.log"
    log.write_text('{"type":"turn_start"}\n{"type":"message_start"}\n{"type":"agent_end","messages":[]}\n'
                   '{"type":"compaction_start","reason":"threshold"}\n{"type":"compaction_end","aborted":false}\n')
    assert Attempt._ended(log)
    log.write_text(log.read_text() + '{"type":"turn_start"}\n')
    assert not Attempt._ended(log)
    log.write_text('{"type":"turn_start"}\n{"type":"agent_end","messages":[]}\n{"type":"compaction_start"}\n')
    assert not Attempt._ended(log)  # a compaction after agent_end: the autonomous continuation follows


def test_void_cut_rows(tmp_path):
    loop = make_loop(tmp_path)
    loop.fixed_pool = ["pa1-v1"]
    item = loop.state["candidates"]["c0001"]
    item.update(pool={"pa1-v1": 50.0, "pa2-v1": 90.0}, pool_runs={"pa1-v1": [10.0, 90.0], "pa2-v1": [90.0]},
                pool_levels={"pa1-v1": 1, "pa2-v1": 2})
    loop.state["jobs"] = []
    loop.state["rows"] = [
        {"id": "j1", "kind": "pool", "candidate": "c0001", "game": "pa1-v1", "seed": 0, "outcome": "stopped",
         "score": 10.0, "started": 200.0},
        {"id": "j2", "kind": "pool", "candidate": "c0001", "game": "pa2-v1", "seed": 0, "outcome": "stopped",
         "score": 90.0, "started": 200.0},
        {"id": "j3", "kind": "pool", "candidate": "c0001", "game": "pa2-v1", "seed": 0, "outcome": "stopped",
         "score": 5.0, "started": 50.0}]
    loop.void_cut_rows(100.0)
    assert item["pool"] == {"pa1-v1": 90.0} and "pa2-v1" not in item["pool_levels"]
    assert [r.get("void") is not None for r in loop.state["rows"]] == [True, True, False]
    jobs = {(j["game"], j.get("fixed", False)) for j in loop.state["jobs"]}
    assert jobs == {("pa1-v1", True), ("pa2-v1", False)} and item["status"] == "evaluating"
    loop.void_cut_rows(100.0)
    assert len(loop.state["jobs"]) == 2  # once


def test_reanchor_gen0(tmp_path):
    loop = make_loop(tmp_path)
    loop.runtime.seal = lambda folder: None
    tree = loop.candidate_dir("c0000")
    (tree / "player").mkdir(parents=True)
    (tree / "player/run.json").write_text(json.dumps({"argv": ["node", "x", "--autonomous", "--autonomous-max-turns",
                                                               "100000", "--thinking", "xhigh"], "env": {}}))
    loop.fixed_pool = ["pa1-v1"]
    loop.state["candidates"]["c0001"]["pool"]["pa3-v1"] = 50.0
    loop.state["rows"] = [{"id": "j1", "kind": "pool", "candidate": "c0000", "game": "pa1-v1", "score": 10.0}]
    loop.open_heldout("c0000", ("heldout_a",), 1)["open"] = False
    loop.state["jobs"] = []
    loop.reanchor_gen0()
    c = loop.state["candidates"]
    assert c["c0000-v1"]["status"] == "retired" and c["c0000-v1"]["pool"] and not c["c0000"]["pool"]
    assert loop.state["rows"][0]["candidate"] == "c0000-v1"
    assert all(r["candidate"] == "c0000-v1" for r in loop.state["heldout"]["runs"])
    old = json.loads((loop.candidate_dir("c0000-v1") / "player/run.json").read_text())["argv"]
    assert old == ["node", "x", "--autonomous", "--thinking", "xhigh"]
    anchor = sorted(j["game"] for j in loop.state["jobs"] if j.get("anchor"))
    assert anchor == ["pa2-v1", "pa3-v1"]  # every pool game a candidate played, fixed ones left to ensure_fixed
    loop.ensure_fixed()
    assert any(j.get("fixed") and j["candidate"] == "c0000" for j in loop.state["jobs"])
    assert "c0000-v1" not in loop.view()["candidates"] and loop.view()["candidates"]["c0000"]["queued"] == 0
    loop.state["candidates"]["c0000-v1"]["pool"].update({"pa1-v1": 99.0, "pa2-v1": 99.0})
    assert loop.best(require_fixed=False) != "c0000-v1"


def test_heldout_pause_waits_for_proposer_session(tmp_path):
    import asyncio

    loop = make_loop(tmp_path)
    loop.proposer.process = object()  # a session is running
    loop.open_heldout("c0000", ("heldout_a",), 1)
    asyncio.run(loop.heldout_hold())
    assert loop.state["heldout_hold"] == "waiting" and not loop.proposer.paused
    assert loop.proposer.notes[-1]["event"] == "heldout_pause_coming"
    assert loop.next_job() is None  # held-out games wait for the session
    loop.proposer.process = None  # the session ended
    asyncio.run(loop.heldout_hold())
    assert loop.state["heldout_hold"] == "held" and loop.proposer.paused
    assert loop.next_job()["kind"] == "heldout_a"


def test_remote_heldout_request_and_heldout_machine(tmp_path):
    import asyncio

    loop = make_loop(tmp_path, {"heldout_mode": "remote"})
    loop.state["meters"]["loop_s"] = 10.5 * 3600
    loop.heldout_due()
    run = loop.state["heldout"]["runs"][-1]
    assert run["remote"] and not run["open"] and run["candidate"] == "c0000"
    assert not any(j["kind"].startswith("heldout") for j in loop.state["jobs"])  # nothing plays on the loop GPU
    assert not loop.heldout_exclusive()
    request = json.loads((loop.ckpt / "heldout/requests" / f"{run['id']}.json").read_text())
    assert request["candidate"] == "c0000" and request["sets"] == ["heldout_a"] and request["serving_hash"] == "h0"
    loop.state["candidates"]["c0000"]["status"] = "evaluating"  # re-anchored gen 0 not re-scored yet: no best's A
    loop.heldout_due()
    assert len(loop.state["heldout"]["runs"]) == 1
    loop.state["candidates"]["c0000"]["status"] = "done"
    loop.heldout_due()
    assert [r["candidate"] for r in loop.state["heldout"]["runs"]] == ["c0000", "c0001"]
    loop.heldout_due()
    assert len(loop.state["heldout"]["runs"]) == 2  # not requested twice
    # the held-out machine: the loop's state fork, its requests only
    loop.state["jobs"].append({"id": "jp", "kind": "pool", "candidate": "c0001", "game": "pa3-v1", "seed": 0,
                               "queued": 0.0})
    hm = make_loop(tmp_path / "hm", {"role": "heldout", "heldout_requests": [request]})
    hm.state.update({k: json.loads(json.dumps(loop.state[k])) for k in ("jobs", "heldout", "seq")})
    hm.take_requests()
    assert [j["kind"] for j in hm.state["jobs"]] == ["heldout_a"]
    hrun = next(r for r in hm.state["heldout"]["runs"] if r["id"] == run["id"])
    assert hrun["open"] and hrun["played_here"]
    hm.heldout_due()
    assert len(hm.state["heldout"]["runs"]) == 2  # the held-out machine never opens points of its own
    asyncio.run(hm.heldout_hold())
    assert hm.state["heldout_hold"] == "held" and hm.next_job()["kind"] == "heldout_a"
    hm.state["jobs"] = []
    asyncio.run(hm.heldout_hold())
    assert hm.state["heldout_hold"] == "held" and not hm.state.get("pending_switch")  # no scrub: the machine ends
    hm.close_heldout()
    assert not hrun["open"]


def test_remote_scoring_dispatch_meter_and_scorer(tmp_path, monkeypatch):
    import asyncio
    import time

    from rsi_loop import remote

    ex = tmp_path / "ex"
    loop = make_loop(tmp_path, {"scorers": ["s1"], "exchange_root": str(ex), "remote_poll_seconds": 0.05})
    loop.state["concurrency"] = 4
    loop.state["jobs"] = [{"id": f"j{i}", "kind": "pool", "candidate": "c0000", "game": g, "seed": 0, "queued": i}
                          for i, g in enumerate(["pa1-v1", "pa2-v1", "pa3-v1"])]
    box = remote.inbox(loop.cfg, "s1")
    remote.write_json(box / "status.json", {"at": time.time(), "machine_s": 3600.0, "idle_s": 600.0,
                                            "live_hash": "h0", "deadline": time.time() + 9e4})
    before = loop.state["meters"]["loop_s"]
    loop.tick_remote()
    assert loop.state["meters"]["loop_s"] - before == 3600.0  # the scorer's GPU-s are loop GPU-s
    loop.tick_remote()
    assert loop.state["meters"]["loop_s"] - before == 3600.0  # counted once
    assert loop.scoring_slots() == loop.slots() + 4

    async def scenario():
        loop.dispatch_remote()
        assert len(loop.state["jobs"]) == 0 and len(loop.running) == 3  # all three fit the scorer's 4 slots
        await asyncio.sleep(0.2)
        jobs = sorted(p.name for p in (remote.out_dir(loop.cfg) / "jobs/s1").iterdir())
        assert jobs == ["j0.json", "j1.json", "j2.json"]
        remote.write_json(box / "results/j0.json", {"score": 42.0, "levels": 1, "level_actions": [5],
                                                    "total_levels": 2, "outcome": "time", "seconds": 100.0,
                                                    "tokens": 10, "actions": 5})
        await asyncio.sleep(0.3)
        assert "j0" not in loop.running and loop.state["candidates"]["c0000"]["pool"]["pa1-v1"] == 42.0
        loop.requeue_running("test")  # a stack switch: the scorer is told to cancel
        await asyncio.sleep(0.1)
        ctrl = remote.read_json(remote.out_dir(loop.cfg) / "control.json")
        assert {"j1", "j2"} <= set(ctrl["cancel"]) and all("remote" not in j for j in loop.state["jobs"])

    loop.done = _wrap_done(loop)
    asyncio.run(scenario())

    # the scorer side: plays a job file from its inbox on the live stack, writes the row
    sc = make_loop(tmp_path / "sc", {"role": "scorer", "peer": "L", "scorer_id": "s1", "exchange_root": str(ex)})
    sc.state["candidates"]["c0000"]["concurrency"] = 4
    (sc.candidate_dir("c0000")).mkdir(parents=True)
    scorer = remote.Scorer(sc)
    lbox = remote.inbox(sc.cfg, "L")
    remote.write_json(lbox / "control.json", {"at": time.time(), "live": "c0000", "live_hash": "h0",
                                              "dispatching": True, "end": False, "cancel": []})
    remote.write_json(lbox / "jobs/s1/j7.json", {"job": {"id": "j7", "kind": "pool", "candidate": "c0000",
                                                         "game": "pa1-v1", "seed": 0}, "wall": 60.0,
                                                 "token_cap": None, "concurrency": 4})

    class FakeAttempt:
        def __init__(self, runtime, job, point, tree, **kw):
            self.point, self.kw = point, kw

        async def run(self):
            return {"score": 7.0, "levels": 1, "outcome": "time", "wall": self.kw["wall"]}

    monkeypatch.setattr(remote, "Attempt", FakeAttempt)

    async def play():
        scorer.start_jobs()
        assert "j7" in sc.running
        await asyncio.sleep(0.05)
        scorer.start_jobs()  # taken once

    asyncio.run(play())
    row = remote.read_json(remote.out_dir(sc.cfg) / "results/j7.json")
    assert row["score"] == 7.0 and row["wall"] == 60.0 and not sc.running
    scorer.status()
    assert remote.read_json(remote.out_dir(sc.cfg) / "status.json")["live_hash"] == "h0"

    # a failed read of control.json (torn mid-sync) keeps the last good control instead of ending the scorer
    good = scorer.control()
    (lbox / "control.json").write_text('{"at": 12')
    assert scorer.control() == good
    (lbox / "control.json").unlink()
    assert scorer.control() == good


def _wrap_done(loop):
    original = loop.done

    async def done(job, row):
        loop.write_scores = lambda cid: None
        loop.run_policy = _noop
        loop.probation_note = lambda **k: None
        await original(job, row)
    return done


async def _noop():
    return None


def test_by_game_length_feedback():
    from rsi_loop import feedback

    cfg = {"slice_seconds": 294.5}
    cands = {"c0000": {"status": "done", "concurrency": 14, "pool": {"g1": 10.0, "g2": 20.0, "g3": 5.0},
                       "pool_runs": {"g1": [8.0, 12.0]}},
             "c0001": {"status": "done", "concurrency": 14, "game_seconds": 8000, "pool": {"g1": 15.0, "g2": 18.0,
                                                                                            "g3": 9.0}}}
    state = {"candidates": cands, "rows": [{"kind": "pool", "candidate": "c0001", "seconds": 7000.0, "started": 1.0,
                                            "ended": 7001.0, "game": "g1"}]}
    out = feedback.by_length(state, cfg, slots=26, pool_games=3)
    assert set(out["lengths"]) == {"69 min", "133 min"}
    long = out["lengths"]["133 min"]
    assert long["gpu_hours_per_game"] == round(8000 / 14 / 3600, 3) and long["games_to_detect_gain"]["+3"] > 0
    assert out["candidates"]["c0001"]["pool_gpu_hours_spent"] == round(7000 / 14 / 3600, 2)


def test_loop_facts_hide_heldout_format_and_kaggle_ram():
    from types import SimpleNamespace

    from rsi_loop import feedback
    from rsi_loop.attempt import Runtime

    assert "capped at 100K" not in feedback.LOOP_FACTS and "100K generated" not in feedback.LOOP_FACTS
    assert "Correction (operator)" in feedback.LOOP_FACTS
    assert "RTX PRO 6000" in feedback.LOOP_FACTS and "4 x L4" not in feedback.LOOP_FACTS
    rt = SimpleNamespace()
    import rsi_loop.attempt as A
    real = A.sandbox.mem_total
    A.sandbox.mem_total = lambda: 218 * 10 ** 9
    try:
        gib = Runtime.memory(rt, "player", 14) / 2 ** 30
    finally:
        A.sandbox.mem_total = real
    assert 6.3 < gib < 6.4
