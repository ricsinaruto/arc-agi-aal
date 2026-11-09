"""Resume mid-try (RL 24 Oct, proposal b): a running try's rollout and log are snapshotted to the keep root; a relaunch's
replay of the bench restores the try at its last request, keeps its own replies trainable, and finishes it as the
uninterrupted try would have."""

import asyncio
import json
import shutil
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from conftest import FakeEnv, observation, tool_reply
from test_level_bench import QuietMonitor

from arc_infra.perception import board_hash
from arena import harvest, level_bench
from arena.level_points import point_id

ROOT = Path(__file__).resolve().parents[1]
GAME = "aa11-00000000"
PER_LEVEL = 4  # actions per level


def board(k):
    b = np.zeros((64, 64), dtype=np.uint8)
    b[k % 64, (3 * k) % 64] = 1 + k % 9
    return b


class SlowArcade:
    """Each level takes PER_LEVEL actions, a new board every step."""

    def open_scorecard(self, tags):
        return "card"

    def close_scorecard(self, card):
        return None

    def make(self, game_id, **kwargs):
        return FakeEnv([observation(board(k), levels=min(k // PER_LEVEL, 5)) for k in range(80)])


def handler_for(sent, block_at=None):
    async def handler(request):
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"count": 1000, "max_model_len": 65536})
        payload = json.loads(request.content)
        if block_at is not None and len(sent) >= block_at:
            await asyncio.Event().wait()  # the container dies mid-try
        sent.append(payload)
        if payload.get("max_tokens") == 1:
            return httpx.Response(200, json={"id": "h", "choices": [{"finish_reason": "length", "message": {
                "role": "assistant", "content": "x"}}], "usage": {"completion_tokens": 1, "prompt_tokens": 4100}})
        return tool_reply("action([{'action':'UP'}])", call_id=f"c{len(sent)}",
                          usage={"completion_tokens": 3, "prompt_tokens": 4000 + len(sent)})
    return handler


def make_bench(tmp_path, out, points, handler, keep=None, reuse=(), snap_s=0.0, version=None):
    (tmp_path / out).mkdir(exist_ok=True)
    path = tmp_path / out / "points.json"
    path.write_text(json.dumps({"schema": 1, "points": points}))
    study = {"completion_cap": 4096, "reasoning_effort": "xhigh", "temperature": 0.7, "generation_slots": 4,
             "server": {"mtp_tokens": 0, "max_model_len": 65536}, "wall_seconds": 60,
             "level_bench": {"start_points": str(path), "capacity_tokens": 200000, "window": 16384,
                             "token_budget": 60, "concurrency": 1, "passes": 1, "min_wall_seconds": 5,
                             "launch_cutoff_seconds": 0, "progress_seconds": 1, "rollout": {"top_logprobs": 0}}}
    if keep is not None:
        study["try_keep"] = {"keep": str(keep), "reuse": [str(r) for r in reuse], "snap_s": snap_s}
    if version is not None:
        study["policy_version"] = version
    return level_bench.LevelBench(study, {"name": "level-bench", "kind": "level-bench"}, source=ROOT,
                                  games_dir=None, output=tmp_path / out / "bench", server_dir=tmp_path,
                                  base_url="http://127.0.0.1:9/v1", deadline_epoch=time.time() + 120,
                                  log=lambda *a, **k: None, arcade=SlowArcade(),
                                  transport=httpx.MockTransport(handler))


def base_point(**extra):
    return {"id": point_id(GAME, 0), "game": GAME, "split": "dev", "level_index": 0, "level": 1, "total_levels": 5,
            "baseline_actions": 2, "prefix": [], "prefix_actions": 0, "board_hash": board_hash(board(0)), **extra}


def rows_of(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


async def killed_leg(tmp_path, point, block_at, keep, prefix=0):
    """Leg 1: plays until request ``block_at`` hangs, waits for a periodic snapshot, then the snapshot as a hard kill
    would leave it (copied before the bench's own stop adds its final lines)."""
    sent = []
    bench = make_bench(tmp_path, "leg1", [point], handler_for(sent, block_at), keep=keep, snap_s=0.05, version=3)
    task = asyncio.create_task(bench.run())
    live = keep / "bench" / "live"
    for _ in range(400):
        await asyncio.sleep(0.05)
        metas = list(live.glob("*/meta.json"))
        if metas and json.loads(metas[0].read_text())["rows"] >= prefix + block_at:
            break
    killed = tmp_path / "killed"
    shutil.copytree(keep, killed)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return sent, killed


async def test_fresh_try_resumes_at_its_last_request_and_finishes_like_the_uninterrupted_try(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    point = base_point(id=f"{point_id(GAME, 0)}-L1-sc", through=False, segment=True, budget=60)
    ref_sent = []
    ref = await make_bench(tmp_path, "ref", [point], handler_for(ref_sent)).run()
    (ref_row,) = ref["attempts"]
    assert ref_row["outcome"] == "cleared" and ref_row["actions"] == PER_LEVEL

    sent1, killed = await killed_leg(tmp_path, point, 2, tmp_path / "keep1")
    (meta_path,) = (killed / "bench" / "live").glob("*/meta.json")
    meta = json.loads(meta_path.read_text())
    assert meta["rows"] == 2 and meta["policy_version"] == 3 and meta["label"] == ref_row["label"]
    assert level_bench.resume_request(meta_path.parent) == (1, 0)  # the last request: its reply is played again

    sent2 = []
    report = await make_bench(tmp_path, "leg2", [point], handler_for(sent2), keep=tmp_path / "keep2",
                              reuse=[killed], snap_s=0.05, version=5).run()
    (row,) = report["attempts"]
    assert row["resumed"]["ok"] and row["resumed"]["request"] == 1, row.get("resumed")
    assert "restore" not in row  # a fresh start: no lineage to report
    assert row["policy_version"] == 3  # the oldest reply's version
    for key in ("outcome", "actions", "completion_tokens", "tokens_to_clear", "gameplay_requests"):
        assert row[key] == ref_row[key], (key, row[key], ref_row[key])
    assert len(sent2) == len(ref_sent) - 1  # request 0 answered from the snapshot
    rollout = rows_of(tmp_path / "leg2" / "bench" / "attempts" / row["label"] / "rollout.jsonl")
    assert [r["n"] for r in rollout] == list(range(len(rollout)))
    assert rollout[0].get("resumed") and not rollout[0].get("scripted")  # its own reply: trainable
    assert not any(r.get("scripted") or r.get("resumed") for r in rollout[1:])
    assert row["rollout"]["resumed"] == 1
    # finished: the snapshot is gone, the kept copy is there
    assert not list((tmp_path / "keep2" / "bench" / "live").glob("*"))
    assert [r["label"] for r in rows_of(tmp_path / "keep2" / "bench" / "tries.jsonl")] == [row["label"]]


@pytest.mark.parametrize("through", [None, True])  # True: the base game point's flag, as mg_points spreads it (v6i)
async def test_restored_start_resumes_with_the_start_prefix_scripted_and_its_own_replies_live(tmp_path, monkeypatch,
                                                                                             through):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    fresh = base_point(id=f"{point_id(GAME, 0)}-L1-sc", through=False, segment=True, budget=60, harvest=True)
    first = await make_bench(tmp_path, "r1", [fresh], handler_for([])).run()
    (r1,) = first["attempts"]
    snap = harvest.snapshot(tmp_path / "r1" / "bench" / "attempts" / r1["label"], tmp_path / "harvest" / "s1")
    start = r1["harvest"]["request"]
    point = base_point(id=f"{point_id(GAME, 0)}-L2-x", level=2, harvest=True,
                       restore={"attempt": str(snap), "request": start, "check_only": False, "budget": 60},
                       **({"through": True} if through else {}))
    ref_sent = []
    ref = await make_bench(tmp_path, "ref", [point], handler_for(ref_sent)).run()
    (ref_row,) = ref["attempts"]
    assert ref_row["restore"]["ok"] and ref_row["outcome"] == "cleared"

    sent1, killed = await killed_leg(tmp_path, point, 3, tmp_path / "keep1", prefix=start)
    found = level_bench.resume_request(next((killed / "bench" / "live").glob("*")))
    assert found is not None and found[0] == start + 2 and found[1] == 3 * start  # prefix tokens: the start's

    sent2 = []
    report = await make_bench(tmp_path, "leg2", [point], handler_for(sent2), keep=tmp_path / "keep2",
                              reuse=[killed]).run()
    (row,) = report["attempts"]
    assert row["resumed"]["ok"] and row["restore"]["source"] == str(snap) and row["restore"]["request"] == start
    for key in ("outcome", "actions", "completion_tokens", "tokens_to_clear", "gameplay_requests"):
        assert row[key] == ref_row[key], (key, row[key], ref_row[key])
    assert row["harvest"]["levels"] == ref_row["harvest"]["levels"]
    rollout = rows_of(tmp_path / "leg2" / "bench" / "attempts" / row["label"] / "rollout.jsonl")
    assert all(r.get("scripted") for r in rollout[:start])
    assert all(r.get("resumed") and not r.get("scripted") for r in rollout[start:start + 2])
    assert len(sent2) == len(ref_sent) - 2


async def test_no_snapshot_digest_mismatch_or_nothing_to_keep_plays_from_the_start(tmp_path, monkeypatch):
    import arc_infra.metrics

    monkeypatch.setattr(arc_infra.metrics, "GpuMonitor", QuietMonitor)
    point = base_point(id=f"{point_id(GAME, 0)}-L1-sc", through=False, segment=True, budget=60)
    _, killed = await killed_leg(tmp_path, point, 2, tmp_path / "keep1")
    other = {**point, "budget": 50}  # another plan entry: the snapshot is not this try's
    sent = []
    (row,) = (await make_bench(tmp_path, "leg2", [other], handler_for(sent), keep=tmp_path / "k2",
                               reuse=[killed]).run())["attempts"]
    assert "resumed" not in row and row["outcome"] == "cleared"
    # a snapshot with only the first request: no own reply to keep
    one = tmp_path / "one"
    shutil.copytree(next((killed / "bench" / "live").glob("*")), one)
    lines = (one / "rollout.jsonl").read_text().splitlines()
    (one / "rollout.jsonl").write_text(lines[0] + "\n")
    assert level_bench.resume_request(one) is None
    # an unsafe event before the last request: restore before it, or not at all
    with open(one / "log.jsonl", "a") as file:
        file.write('{"kind":"playbook_handoff","step":0}\n')
    (one / "rollout.jsonl").write_text("\n".join(lines) + "\n")
    assert level_bench.resume_request(one) is None


def test_snap_copy_appends_whole_lines_only(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    (src / "rollout.jsonl").write_text('{"n":0}\n{"n":1')
    (src / "log.jsonl").write_text('{"kind":"a"}\n')
    offsets = {}
    assert level_bench.snap_copy(src, dest, offsets) == {"rollout.jsonl": 1, "log.jsonl": 1}
    assert (dest / "rollout.jsonl").read_text() == '{"n":0}\n'
    with open(src / "rollout.jsonl", "a") as file:
        file.write('}\n{"n":2}\n')
    assert level_bench.snap_copy(src, dest, offsets) == {"rollout.jsonl": 3, "log.jsonl": 1}
    assert (dest / "rollout.jsonl").read_text() == (src / "rollout.jsonl").read_text()


def test_v7_keep_carries_the_snapshot_interval(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from arena.rl_async import AsyncMixin

    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path))
    loop = SimpleNamespace(cfg={"mg": {}})
    assert AsyncMixin.v7_keep(loop)["try_keep"]["snap_s"] == 300
    loop.cfg["mg"]["try_snap_s"] = 0
    assert AsyncMixin.v7_keep(loop)["try_keep"]["snap_s"] == 0
