"""In-run harvest (envs, 21 Oct): a cleared try records the next level's first request; the round's harvest snapshots
it as a start one level deeper; the next round restores it (and can harvest again)."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
from conftest import FakeEnv, observation, tool_reply
from test_level_bench import QuietMonitor

from arc_infra.perception import board_hash
from arena import harvest, level_bench, restore
from arena.level_points import point_id

ROOT = Path(__file__).resolve().parents[1]
GAME = "aa11-00000000"


def board(k):
    b = np.zeros((64, 64), dtype=np.uint8)
    b[k, k] = 1 + k % 9
    return b


class LadderArcade:
    """Every action clears one level (5 levels), with a new board each step."""

    def open_scorecard(self, tags):
        return "card"

    def close_scorecard(self, card):
        return None

    def make(self, game_id, **kwargs):
        return FakeEnv([observation(board(k), levels=min(k, 5)) for k in range(60)])


def handler_for(sent):
    def handler(request):
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": 1000, "max_model_len": 65536})
        payload = json.loads(request.content)
        sent.append(payload)
        if payload.get("max_tokens") == 1:
            return httpx.Response(200, json={"id": "h", "choices": [{"finish_reason": "length", "message": {
                "role": "assistant", "content": "x"}}], "usage": {"completion_tokens": 1, "prompt_tokens": 4100}})
        return tool_reply("action([{'action':'UP'}])", call_id=f"c{len(sent)}",
                          usage={"completion_tokens": 3, "prompt_tokens": 4000})
    return handler


def run_bench(tmp_path, out, points, sent):
    path = tmp_path / f"{out}-points.json"
    path.write_text(json.dumps({"schema": 1, "points": points}))
    study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
             "server": {"mtp_tokens": 0, "max_model_len": 65536}, "wall_seconds": 30,
             "level_bench": {"start_points": str(path), "capacity_tokens": 200000, "window": 16384,
                             "token_budget": 40, "concurrency": 1, "passes": 1, "min_wall_seconds": 5,
                             "launch_cutoff_seconds": 0, "progress_seconds": 1, "rollout": {"top_logprobs": 0}}}
    bench = level_bench.LevelBench(study, {"name": "level-bench", "kind": "level-bench"}, source=ROOT,
                                   games_dir=None, output=tmp_path / out, server_dir=tmp_path,
                                   base_url="http://127.0.0.1:9/v1", deadline_epoch=time.time() + 60,
                                   log=lambda *a, **k: None, arcade=LadderArcade(),
                                   transport=httpx.MockTransport(handler_for(sent)))
    return bench


def base_point(**extra):
    return {"id": point_id(GAME, 0), "game": GAME, "split": "dev", "level_index": 0, "level": 1, "total_levels": 5,
            "baseline_actions": 2, "prefix": [], "prefix_actions": 0, "board_hash": board_hash(board(0)), **extra}


async def play(tmp_path, out, point, sent):
    bench = run_bench(tmp_path, out, [point], sent)
    report = await bench.run()
    (row,) = report["attempts"]
    return row, tmp_path / out / "attempts" / row["label"]


async def test_cleared_try_records_the_next_level_start_and_it_restores_and_harvests_again(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    sent = []
    fresh = base_point(id=f"{point_id(GAME, 0)}-L1-sc", through=False, segment=True, budget=40, harvest=True)
    row, att = await play(tmp_path, "r1", fresh, sent)
    assert row["outcome"] == "cleared" and row["harvest"]["request"] == 1 and row["harvest"]["levels"] == 1
    assert row["harvest"]["state"] == "NOT_FINISHED"
    assert [p.get("max_tokens") for p in sent] == [4096, 1]
    assert row["gameplay_requests"] == 1 and row["completion_tokens"] == 3  # the harvest request is not the try's
    rollout = [json.loads(x) for x in (att / "rollout.jsonl").read_text().splitlines()]
    assert len(rollout) == 1 and not any(r.get("harvest") for r in rollout)  # the trainer's file is unchanged
    (hrow,) = [json.loads(x) for x in (att / "harvest.jsonl").read_text().splitlines()]
    assert hrow["harvest"] is True and hrow["n"] == 1
    kinds = [json.loads(x)["kind"] for x in (att / "log.jsonl").read_text().splitlines()]
    assert kinds.index("level_bench_clear") < kinds.index("level_bench_level") < kinds.index("level_bench_harvest")

    snap = harvest.snapshot(att, tmp_path / "harvest" / "s1")
    info = restore.load(snap, 1)
    assert info["level"] == 2 and info["stop"]["levels"] == 1 and info["actions"] == ["ACTION1"]
    assert restore.level_starts(snap)[-1] == (1, 1, 1)
    ok, detail = harvest.board_check(LadderArcade(), "card", GAME, snap, 1)
    assert ok, detail
    assert (snap / "log.jsonl").read_text().splitlines()[-1].startswith(harvest.MARK)

    # next round: the harvested start, restored and played live; its clear is harvested again (L2 -> L3)
    sent.clear()
    restored = base_point(id=f"{point_id(GAME, 0)}-L2-x", level=2,
                          restore={"attempt": str(snap), "request": 1, "check_only": False, "budget": 40},
                          harvest=True)
    row2, att2 = await play(tmp_path, "r2", restored, sent)
    assert row2["restore"]["ok"], row2["restore"]
    assert row2["outcome"] == "cleared" and row2["harvest"]["levels"] == 2 and row2["actions"] == 1
    snap2 = harvest.snapshot(att2, tmp_path / "harvest" / "s2")
    r = row2["harvest"]["request"]
    assert restore.load(snap2, r)["level"] == 3 and restore.load(snap2, r)["actions"] == ["ACTION1", "ACTION1"]
    assert harvest.board_check(LadderArcade(), "card", GAME, snap2, r)[0]
    assert [rr["n"] for rr in map(json.loads, (snap2 / "rollout.jsonl").read_text().splitlines())] == list(range(r + 1))
    # a cut inside the restored level (branch) counts the level's actions from its first board, not from step 0
    assert restore.load(att2, 1)["level_start_step"] == 1

    # a wrong board fails the CPU check
    class Other(LadderArcade):
        def make(self, game_id, **kwargs):
            return FakeEnv([observation(board(k + 1), levels=min(k, 5)) for k in range(60)])

    assert not harvest.board_check(Other(), "card", GAME, snap, 1)[0]


async def test_untagged_tries_and_aside_requests_harvest_nothing(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    sent = []
    row, att = await play(tmp_path, "r1", base_point(id="x-L1-sc", through=False, segment=True, budget=40), sent)
    assert row["outcome"] == "cleared" and "harvest" not in row and not (att / "harvest.jsonl").exists()
    assert [p.get("max_tokens") for p in sent] == [4096]

    agent = level_bench.LevelBenchAgent.__new__(level_bench.LevelBenchAgent)
    logged = []
    agent.workspace = SimpleNamespace(append=lambda kind, **k: logged.append(kind))
    agent.bench_harvest_state, agent.steps, agent.bench_outcome = "due", 3, "cleared"
    agent.game = SimpleNamespace(clock=SimpleNamespace(completion_tokens=lambda: 0))
    try:
        await agent.aside_request()
    except Exception:  # noqa: BLE001 - the gate stops the finished attempt
        pass
    assert agent.bench_harvest_state == "cancelled" and logged[0] == "level_bench_harvest_cancelled"


# --- the loop side: collect / pick / state ---------------------------------------------------------------------

def fake_loop(tmp_path, **h):
    feed = []
    sets = {"L1": {"name": "L1", "role": "train", "game": GAME, "level": 1, "fresh": True},
            "L2": {"name": "L2", "role": "train", "game": GAME, "level": 2, "train_starts": 4},
            "L3": {"name": "L3", "role": "train", "game": GAME, "level": 3},
            "T": {"name": "T", "role": "test", "game": "bb22", "level": 1}}
    from arena.rl_single import MultiMixin

    loop = SimpleNamespace(cfg={"mg": {"harvest": {"switch_round": 3, **h}}, "mg_sets": sets}, output=tmp_path / "loop",
                           feed=SimpleNamespace(emit=lambda kind, **k: feed.append((kind, k))),
                           mg_attempt_start=MultiMixin.mg_attempt_start, games_dir=None)
    return loop, sets, feed


def recorded_try(folder, label, levels=1, steps=1, state="NOT_FINISHED"):
    """A finished harvested try on disk (one live reply, its clear at ``steps``, the harvest request after)."""
    att = folder / "attempts" / label
    att.mkdir(parents=True)
    obs = [{"kind": "observation", "step": s, "levels": min(s, levels), "frames": [["0" * 64] * 64],
            "action": {"action": "ACTION1"} if s else None} for s in range(steps + 1)]
    log = obs + [{"kind": "level_bench_clear", "step": steps}, {"kind": "level_bench_level", "step": steps,
                 "level": levels, "steps": steps}, {"kind": "level_bench_harvest", "step": steps, "request": 1},
                 {"kind": "model", "purpose": "gameplay", "step": steps}]
    (att / "log.jsonl").write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in log))
    (att / "rollout.jsonl").write_text(json.dumps({"n": 0, "step": 0, "gameplay": True, "usage": {"prompt_tokens": 9}}) + "\n")
    (att / "harvest.jsonl").write_text(json.dumps({"n": 1, "step": steps, "gameplay": True, "harvest": True,
                                                   "usage": {"prompt_tokens": 50}}) + "\n")
    return {"label": label, "game": GAME, "outcome": "cleared", "tokens_to_clear": 100, "level_rows": [{"tokens": 100}],
            "harvest": {"request": 1, "prompt_tokens": 50, "step": steps, "levels": levels, "state": state}}


def test_collect_snapshots_checks_and_feeds_the_next_set(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vol"))
    loop, sets, feed = fake_loop(tmp_path, max_prompt=60)
    f1 = tmp_path / "r01-L1"
    rows = [recorded_try(f1, "a"), recorded_try(f1, "b", state="GAME_OVER"), recorded_try(f1, "c", levels=2),
            {**recorded_try(f1, "d"), "restore": {"ok": False}}]
    big = recorded_try(f1, "e")
    big["harvest"]["prompt_tokens"] = 99
    rows.append(big)
    rows.append({"label": "f", "outcome": "budget"})  # not cleared: no harvest
    got = harvest.collect(loop, 1, {"L1": (f1, rows, {}), "T": (f1, rows, {})}, 0)
    assert got["new"] == 1 and got["by_set"] == {"L2": 1}
    assert got["dropped"] == {"game_state": 1, "level_mismatch": 1, "parent_restore_not_exact": 1,
                              "prompt_over_ceiling": 1}
    (rec,) = loop.mg_harvest
    assert rec["attempt"] == str(tmp_path / "vol/harvest/r01-L1-a") and rec["level"] == 2 and rec["target"] == "L2"
    assert (tmp_path / "vol/harvest.jsonl").is_file() and (tmp_path / "loop/rl/harvest.jsonl").is_file()
    assert harvest.collect(loop, 1, {"L1": (f1, rows, {})}, 0)["new"] == 0  # a second pass adds nothing
    # an unsafe event before the start drops it
    f2 = tmp_path / "r02-L1"
    r = recorded_try(f2, "g")
    with open(f2 / "attempts/g/log.jsonl", "r+") as file:
        lines = file.readlines()
        file.seek(0)
        file.writelines(lines[:1] + ['{"kind":"playbook_handoff","step":0}\n'] + lines[1:])
    assert harvest.collect(loop, 2, {"L1": (f2, [r], {})}, 1)["dropped"] == {"unsafe_event": 1}


def test_pick_switch_use_counts_fill_and_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vol"))
    loop, sets, feed = fake_loop(tmp_path, train_starts=3, train_tries=8)
    for rnd, played_by in ((1, 0), (2, 1)):
        folder = tmp_path / f"r{rnd:02d}-L1"
        rows = [recorded_try(folder, f"t{i}") for i in range(2)]
        harvest.collect(loop, rnd, {"L1": (folder, rows, {})}, played_by)
    bank = ([("bank0", 5), ("bank1", 6), ("bank2", 7)], 12)
    assert harvest.pick(loop, 2, "L2", sets["L2"], bank) is None  # before the switch round: the bank
    assert harvest.pick(loop, 3, "L1", sets["L1"], bank) is None  # a fresh set never draws
    st, tries = harvest.pick(loop, 3, "L2", sets["L2"], bank)
    assert tries == 8 and [Path(a).name for a, _ in st] == ["r02-L1-t0", "r02-L1-t1", "r01-L1-t0"]  # newest policy
    st, _ = harvest.pick(loop, 4, "L2", sets["L2"], bank)
    assert [Path(a).name for a, _ in st] == ["r01-L1-t1", "bank0", "bank1"]  # used once each, then the bank fills
    saved = json.loads(json.dumps(harvest.state(loop)))
    loop2, _, _ = fake_loop(tmp_path, train_starts=3, train_tries=8)
    assert harvest.restore_state(loop2, saved) == {"pool": 4, "missing": 0}
    moved = tmp_path / "unpacked"  # a packed preflight read back from its unpacked copy
    (tmp_path / "vol").rename(moved)
    loop4, _, _ = fake_loop(tmp_path)
    assert harvest.restore_state(loop4, saved, moved) == {"pool": 4, "missing": 0}
    assert all(r["attempt"].startswith(str(moved)) for r in loop4.mg_harvest)
    assert harvest.restore_state(loop4, saved) == {"pool": 0, "missing": 4}
    moved.rename(tmp_path / "vol")
    assert harvest.pick(loop2, 5, "L2", sets["L2"], None) is None  # spent pool, no bank: nothing to draw
    assert harvest.pick(loop2, 5, "L3", sets["L3"], bank)[0] == bank[0]  # L3 has no harvest yet: the bank
    loop3, _, _ = fake_loop(tmp_path, train_starts=3, train_tries=8, max_uses=2)
    harvest.restore_state(loop3, saved)
    assert len([a for a, _ in harvest.pick(loop3, 5, "L2", sets["L2"], None)[0]]) == 3  # second use allowed
    loop5, _, _ = fake_loop(tmp_path, train_starts=3, train_tries=8, max_uses=2, since=2)  # a restart: only rows cut from R2 on
    harvest.restore_state(loop5, saved)
    st, _ = harvest.pick(loop5, 5, "L2", sets["L2"], bank)
    assert [Path(a).name for a, _ in st] == ["r02-L1-t0", "r02-L1-t1", "bank0"]


def test_wants_only_training_sets_with_a_restored_set_one_level_down(tmp_path):
    loop, sets, _ = fake_loop(tmp_path)
    mg = loop.cfg["mg"]
    assert harvest.wants(mg, sets, sets["L1"]) and harvest.wants(mg, sets, sets["L2"])
    assert not harvest.wants(mg, sets, sets["L3"])  # never L3 -> L4
    assert not harvest.wants(mg, sets, sets["T"])
    assert not harvest.wants({}, sets, sets["L1"])


def test_clear_rate_by_source(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vol"))
    loop, sets, _ = fake_loop(tmp_path)
    folder = tmp_path / "r01-L1"
    harvest.collect(loop, 1, {"L1": (folder, [recorded_try(folder, "a")], {})}, 0)
    h = loop.mg_harvest[0]["attempt"]
    won = {"outcome": "cleared", "level_rows": [{"tokens": 5}]}
    rows = [{**won, "restore": {"source": h}}, {"outcome": "budget", "restore": {"source": h}},
            {**won, "restore": {"source": "/bank/x"}}]
    assert harvest.sources(loop, {"L2": (None, rows, {})}) == {"L2": {"harvest": "1/2", "bank": "1/1"}}
