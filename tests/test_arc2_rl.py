"""arc2 RL glue on CPU: the pool, the reward, the recorded rollouts and the driver against a scripted server and
trainer (the GPU parts are the RL arm's, unchanged)."""

import asyncio
import json
import os
import sys
import types
from pathlib import Path

import httpx
import pytest

from arc2 import rl_loop, rl_pool

TRAIN = [{"input": [[1, 2], [3, 4]], "output": [[1, 3], [2, 4]]}, {"input": [[5, 6]], "output": [[5], [6]]}]
RIGHT = "```python\ndef transform(grid):\n    return [list(r) for r in zip(*grid)]\n```"
WRONG = "```python\ndef transform(grid):\n    return grid\n```"
RL_SRC = Path(__file__).resolve().parents[1] / "../rl/scripts/posttrain"


def row(name, holdout_output=((7,), (8,), (9,))):
    return {"id": f"arc2-train:{name}", "source": "arc2-train", "train": TRAIN,
            "holdout": [{"input": [[7, 8, 9]], "output": [list(r) for r in holdout_output]}]}


def write_task(path, task):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(task))


def test_pool_build_excludes_and_dedups(tmp_path):
    big = [[1, 1, 1], [2, 2, 2]]
    write_task(tmp_path / "ARC-AGI-2/data/evaluation/e1.json", {"train": [{"input": big, "output": big}], "test": []})
    write_task(tmp_path / "ARC-AGI-2/data/training/a.json", {"train": TRAIN, "test": [TRAIN[0]]})
    write_task(tmp_path / "ARC-AGI-2/data/training/h.json", {"train": TRAIN[:1], "test": [TRAIN[1]]})
    write_task(tmp_path / "ARC-AGI/data/training/a.json", {"train": TRAIN, "test": [TRAIN[1]]})  # same id: dropped
    write_task(tmp_path / "ARC-AGI/data/training/b.json", {"train": [{"input": big, "output": [[0]]}], "test": []})
    write_task(tmp_path / "ARC-AGI/data/evaluation/c.json", {"train": TRAIN, "test": [TRAIN[0]]})  # same grids as a
    rows, stats = rl_pool.build(tmp_path, {"h"})
    assert [r["id"] for r in rows] == ["arc2-train:a"]
    assert stats["dropped"] == {"shared_grid": 1, "duplicate": 2, "excluded_id": 1}
    view = rl_pool.model_view(rows[0])
    assert "output" not in view["test"][0] and view["train"] == TRAIN


def test_reward_and_advantages():
    good = row("t")
    assert rl_pool.reward(good, [p["output"] for p in TRAIN] + [[[7], [8], [9]]]) == 1.0
    assert rl_pool.reward(good, [p["output"] for p in TRAIN] + [[[7, 8, 9]]]) == 0.0
    assert rl_loop.advantages({"a": [1, 0, 0, 1], "b": [0, 0], "c": [1, 1]}) == {"a": [0.5, -0.5, -0.5, 0.5]}


class Model:
    """A scripted OpenAI-style server: ``/tokenize`` and chat completions with log-probabilities; replies RIGHT and
    WRONG in turn."""

    def __init__(self):
        self.calls, self.models = 0, []

    def __call__(self, request):
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 50 * len(body["messages"])})
        assert request.url.path == "/v1/chat/completions" and body["return_token_ids"]
        self.models.append(body["model"])
        content = RIGHT if self.calls % 2 == 0 else WRONG
        self.calls += 1
        entries = [{"token": f"token_id:{t}", "logprob": -0.1,
                    "top_logprobs": [{"token": f"token_id:{t}", "logprob": -0.1}, {"token": "token_id:9", "logprob": -2.5}]}
                   for t in (11, 12, 13)]
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": content, "reasoning_content": "think"},
                         "finish_reason": "stop", "logprobs": {"content": entries}, "token_ids": [11, 12, 13]}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 3}, "prompt_token_ids": [1, 2, 3]})


def cfg(**extra):
    return {**rl_loop.DEFAULTS, "window": 65536, "attempt_replies": 1, "images": False, **extra}


async def sandboxed():
    from arc2 import sandbox

    strict = sandbox.self_test()["ok"]

    async def execute(code, inputs, **kw):
        return await asyncio.to_thread(sandbox.run, code, inputs, strict=strict, **kw)

    async def explore(cells, code, task, **kw):
        return await asyncio.to_thread(sandbox.run_cells, cells, code, task, strict=strict, **kw)
    return execute, explore


def test_play_records_rollouts_and_rewards(tmp_path):
    model = Model()

    async def go():
        client = httpx.AsyncClient(base_url="http://x", transport=httpx.MockTransport(model))
        transport = rl_loop.RLTransport(client, "m", "m", "medium", 0.7, 4)
        execute, explore = await sandboxed()
        # the held-out output decides: the same verified program earns 1 on the first task and 0 on the second
        rows = [row("good"), row("trap", ((7, 8, 9),))]
        out = await rl_loop.play(rows, 2, transport=transport, execute=execute, explore=explore, cfg=cfg(),
                                 folder=tmp_path, label="screen")
        await client.aclose()
        return out, transport

    out, transport = asyncio.run(go())
    assert sorted(r for r, _, _ in out["arc2-train:good"]) == [0.0, 1.0]
    assert [r for r, _, _ in out["arc2-train:trap"]] == [0.0, 0.0]
    assert transport.stats == {"replies": 4, "tokens": 12, "without_logprobs": 0, "count_mismatches": 0}
    path = Path(out["arc2-train:good"][0][1])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["n"] == 0 and rows[0]["keep"] == 0 and rows[0]["gameplay"]
    assert rows[0]["tokens"] == [11, 12, 13] and rows[0]["top"][0] == [[11, 9], [-0.1, -2.5]]
    assert rows[0]["tools"][0]["function"]["name"] == "python" and rows[0]["template"]["enable_thinking"]
    assert rows[0]["sampling"]["temperature"] == 0.7 and rows[0]["messages"][0]["role"] == "system"
    episode = json.loads((path.parent / "episode.json").read_text())
    assert episode["attempts"] == 1 and episode["done"] in ("verified", "attempts")


def test_local_count_after_the_first_reply(tmp_path):
    """local_count: /tokenize for an attempt's first turn only; later turns are the last usage + an estimate."""
    model = Model()
    calls = []

    def counting(request):
        calls.append(request.url.path)
        return model(request)

    async def go():
        client = httpx.AsyncClient(base_url="http://x", transport=httpx.MockTransport(counting))
        transport = rl_loop.RLTransport(client, "m", "m", "medium", 0.7, 4)
        transport.local_count = True
        transport.files["t"] = tmp_path / "r.jsonl"
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "x" * 300}]
        first = await transport.count(messages, {})
        reply = await transport.complete("t", messages, first, 100, {})
        messages.append(reply["message"])
        messages.append({"role": "tool", "tool_call_id": "1", "content": "y" * 90})
        second = await transport.count(messages, {})
        other = await transport.count([{"role": "user", "content": "z"}], {})
        await client.aclose()
        return first, second, other, transport

    first, second, other, transport = asyncio.run(go())
    assert calls.count("/tokenize") == 2 and transport.stats["estimated"] == 1
    assert first == 100 and second == 103 + rl_loop.estimate_tokens({"role": "tool", "content": "y" * 90}) and other == 50


def test_play_leaves_out_request_failures(tmp_path):
    """An episode whose requests keep failing ends on ``errors``: recorded, not a zero in its group."""
    def failing(request):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 50})
        return httpx.Response(500, text="boom")

    async def go():
        client = httpx.AsyncClient(base_url="http://x", transport=httpx.MockTransport(failing))
        transport = rl_loop.RLTransport(client, "m", "m", "medium", 0.7, 4)
        transport.episodes = asyncio.Semaphore(1)
        execute, explore = await sandboxed()
        out = await rl_loop.play([row("good")], 2, transport=transport, execute=execute, explore=explore, cfg=cfg(),
                                 folder=tmp_path, label="screen")
        await client.aclose()
        return out, transport

    out, transport = asyncio.run(go())
    assert out == {} and transport.stats["invalid"] == 2
    episode = json.loads((tmp_path / "screen/arc2-train_good/0/episode.json").read_text())
    assert episode["done"] == "errors" and episode["valid"] is False
    assert rl_loop.load_prior(tmp_path / "screen") == {}


@pytest.mark.skipif(not (RL_SRC / "../../src/posttrain/rl_rollouts.py").exists(), reason="rl worktree absent")
def test_rollouts_read_by_the_trainer_reader(tmp_path):
    """The trainer's reader (rl_rollouts.captures) restores every request from the recorded rows."""
    sys.path.insert(0, str((RL_SRC / "../../src").resolve()))
    try:
        from posttrain import rl_rollouts
    finally:
        sys.path.pop(0)
    model = Model()

    async def go():
        client = httpx.AsyncClient(base_url="http://x", transport=httpx.MockTransport(model))
        transport = rl_loop.RLTransport(client, "m", "m", "medium", 0.7, 4)
        execute, explore = await sandboxed()
        model.calls = 1  # WRONG first: refinement turns, so later requests keep the earlier messages
        out = await rl_loop.play([row("good")], 1, transport=transport, execute=execute, explore=explore,
                                 cfg=cfg(attempt_replies=3), folder=tmp_path, label="x")
        await client.aclose()
        return out

    out = asyncio.run(go())
    path = out["arc2-train:good"][0][1]
    caps = rl_rollouts.captures(path)
    assert len(caps) == 2 and caps[1]["payload"]["tools"][0]["function"]["name"] == "python"
    assert len(caps[1]["payload"]["messages"]) > len(caps[0]["payload"]["messages"])
    assert caps[1]["payload"]["messages"][:2] == caps[0]["payload"]["messages"][:2]
    assert all(c["prompt_tokens"] and c["usage"]["completion_tokens"] for c in caps)
    compact = rl_rollouts.compact(json.loads(Path(path).read_text().splitlines()[0]))
    assert compact["tokens"] == [11, 12, 13]


def test_driver_order_and_report(tmp_path, monkeypatch):
    calls = []

    class Server:
        def __init__(self, base_url, model):
            self.root, self.model = "http://x", model

        async def sleep(self):
            calls.append("sleep")
            return 0.1

        async def wake(self):
            calls.append("wake")
            return 0.2

        async def load(self, name, path):
            calls.append(f"load {name}")
            return 0.3

    class Trainer:
        def __init__(self, folder, command, env, log_path):
            calls.append(f"trainer dense={env.get('RL1_DENSE_ONLY')} window={command[command.index('--window') + 1]}")

        async def ready(self, timeout):
            return {"state": "ready", "loaded": {"trainable": 1}}

        async def must(self, op, timeout=3600, **payload):
            calls.append(op)
            if op == "update":
                assert payload["signed"] and all(a["whole"] and a["w"] for a in payload["attempts"])
                assert payload["share_prefix"] and payload["batch_max"] == 8 and all(a["share"] for a in payload["attempts"])
                assert abs(sum(a["w"] for a in payload["attempts"])) < 1e-6  # centred within each full group
                return {"loss": 0.1, "windows": [{"trained": 3, "mismatch": {"tokens": 3, "mean_abs": 0.01,
                                                                            "mean": 0.0, "beyond_half": 0.0}}]}
            if op == "profile":
                assert not payload["spec"]["whole"] and payload["warm"]
                return {"wall_seconds": 1.0}
            if op == "save":
                return {"path": payload["path"], "tensors": 10, "gib": 0.1}
            return {"gib": 50.0, "seconds": 1.0}

        call = must

        def close(self):
            calls.append("close")

    def trainer_command(c, source, work):
        return ["python", "trainer.py", "--window", str(c["window"])], {}

    monkeypatch.setitem(sys.modules, "arena.rl_loop", types.SimpleNamespace(
        Server=Server, Trainer=Trainer, trainer_command=trainer_command))
    model = Model()
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(**{**kw, "transport": httpx.MockTransport(model)}))
    source = tmp_path / "bundle"
    (source / "data/rl").mkdir(parents=True)
    pool = [row(f"t{i}") for i in range(4)]
    (source / "data/rl/pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool))
    (source / "data/rl/eval.json").write_text(json.dumps([pool[3]["id"]]))
    study = {"windows": [65536], "arc2_rl": {"eval_ids": "data/rl/eval.json", "eval_tasks": 1, "eval_samples": 2,
                                             "screen_tasks": 3, "screen_samples": 2, "tasks": 3, "group": 2,
                                             "rounds": 1, "per_update": 4, "attempt_replies": 1, "images": False,
                                             "profile": True}}
    import time

    report = asyncio.run(rl_loop.run_arc2_rl(study, {}, source=source, output=tmp_path / "out", server_dir=None,
                                             base_url="http://x/v1", deadline_epoch=time.time() + 7200))
    assert report["status"] == "complete", report.get("traceback")
    assert calls[:5] == ["sleep", "trainer dense=1 window=32768", "park", "wake", "sleep"] or calls[:4] == [
        "sleep", "trainer dense=1 window=32768", "park", "wake"]
    assert calls[-6:] == ["profile", "save", "park", "wake", "load r00", "close"]
    assert report["rounds"][0]["profile"] == {"wall_seconds": 1.0}
    assert report["screen"]["band"] == 3 and report["rounds"][0]["groups_with_spread"] == 3
    assert report["rounds"][0]["updates"][0]["mismatch"]["mean_abs"] == 0.01
    assert model.models[-2:] == ["r00", "r00"] and "eval_base" in report and "eval_final" in report
    status = json.loads((tmp_path / "out/rl/status.json").read_text())
    assert status["status"] == "complete"


def test_screen_from_skips_the_screen(tmp_path, monkeypatch):
    """screen_from: the band comes from an earlier screen's rates; play starts with the round."""
    calls = []

    class Server:
        def __init__(self, base_url, model):
            self.root = "http://x"

        async def sleep(self):
            return 0.0

        async def wake(self):
            return 0.0

        async def load(self, name, path):
            return 0.0

    class Trainer:
        def __init__(self, *a):
            pass

        async def ready(self, timeout):
            return {}

        async def must(self, op, timeout=3600, **payload):
            calls.append(op)
            return {"path": payload.get("path"), "windows": []}

        call = must

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "arena.rl_loop", types.SimpleNamespace(
        Server=Server, Trainer=Trainer, trainer_command=lambda c, s, w: (["x", "--window", "1"], {})))
    model = Model()
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(**{**kw, "transport": httpx.MockTransport(model)}))
    source = tmp_path / "bundle"
    (source / "data/rl").mkdir(parents=True)
    pool = [row(f"t{i}") for i in range(4)]
    (source / "data/rl/pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool))
    (source / "data/rl/rates.json").write_text(json.dumps({pool[0]["id"]: 0.5, pool[1]["id"]: 1.0, pool[2]["id"]: 0.25}))
    study = {"windows": [65536], "arc2_rl": {"screen_from": "data/rl/rates.json", "tasks": 8, "group": 2, "rounds": 1,
                                             "attempt_replies": 1, "images": False}}
    import time

    report = asyncio.run(rl_loop.run_arc2_rl(study, {}, source=source, output=tmp_path / "out", server_dir=None,
                                             base_url="http://x/v1", deadline_epoch=time.time() + 7200))
    assert report["status"] == "complete", report.get("traceback")
    assert report["screen"]["band"] == 2 and model.calls == 4  # 2 band tasks x 2, no screen replies


def test_trainer_speed_steps(tmp_path, monkeypatch):
    ops = []

    class Trainer:
        def __init__(self, folder, command, env, log):
            assert env["RL1_DENSE_ONLY"] == "1"

        async def ready(self, timeout):
            return {}

        async def call(self, op, timeout=900, **payload):
            ops.append((op, payload.get("mode"), payload.get("share_prefix")))
            return {"trained_tokens": 1000, "train_seconds": 2.0, "windows": []} if op == "update" else {"wall_seconds": 1}

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "arena.rl_loop", types.SimpleNamespace(
        Trainer=Trainer, trainer_command=lambda c, s, w: (["x", "--window", str(c["window"])], {})))
    rollouts = [str(tmp_path / f"t{i // 4}/{i % 4}/rollout.jsonl") for i in range(8)]
    report = asyncio.run(rl_loop.trainer_speed(dict(rl_loop.DEFAULTS), tmp_path, rollouts, tmp_path))
    assert report["status"] == "complete"
    assert [o[0] for o in ops] == ["update", "update", "knobs", "update", "knobs", "update"]
    assert report["steps"][1]["tokens_per_second"] == 500.0 and report["steps"][4]["step"] == "knobs_inline"


def test_long_run_checkpoints_and_resumes(tmp_path, monkeypatch):
    """Long-run mode: prior screen episodes, band_hi, round 0 on the screen groups, eval every 2 rounds, probe KL,
    a checkpoint per round under $ARC_VALIDATION_OUTPUT/ckpt, one served name; then a second run resumes from it."""
    calls = []

    class Server:
        def __init__(self, base_url, model):
            self.root = "http://x"

        async def sleep(self):
            calls.append("sleep")
            return 0.0

        async def wake(self):
            calls.append("wake")
            return 0.0

        async def load(self, name, path):
            calls.append(f"load {name}")
            assert Path(path).exists()
            return 0.0

    class Trainer:
        def __init__(self, folder, command, env, log):
            calls.append("trainer")
            assert env["RL1_PREFETCH"] == "0"

        async def ready(self, timeout):
            return {}

        async def must(self, op, timeout=3600, **payload):
            calls.append(op + (f" {payload['label']}" if payload.get("label") else ""))
            if op == "score":
                assert all(Path(p["rollout"]).exists() for p in payload["windows"])
                return {"windows": {p["name"]: {"kl_base": 0.001} for p in payload["windows"]}}
            if op in ("save", "checkpoint"):
                Path(payload["path"]).mkdir(parents=True)
            if op == "update":
                assert "share_prefix" not in payload
                return {"trained_tokens": 100, "train_seconds": 2.0, "windows": []}
            return {"path": payload.get("path")}

        call = must

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "arena.rl_loop", types.SimpleNamespace(
        Server=Server, Trainer=Trainer, trainer_command=lambda c, s, w: (["x", "--window", "1"], {})))
    model = Model()
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(**{**kw, "transport": httpx.MockTransport(model)}))
    source = tmp_path / "bundle"
    (source / "data/rl").mkdir(parents=True)
    pool = [row(f"t{i}") for i in range(6)]
    (source / "data/rl/pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool))
    (source / "data/rl/eval.json").write_text(json.dumps([pool[5]["id"]]))
    for k, done in enumerate(("verified", "verified", None, "errors")):  # an earlier screen: t0 solved twice; a
        folder = tmp_path / "prior" / "arc2-train_t0" / str(k)          # deadline cut and a request failure: left out
        folder.mkdir(parents=True)
        (folder / "episode.json").write_text(json.dumps({"task": pool[0]["id"], "reward": float(k < 2), "spent": 10 + k,
                                                         "done": done}))
        (folder / "rollout.jsonl").write_text("{}\n")
    block = {"eval_ids": "data/rl/eval.json", "eval_tasks": 1, "eval_samples": 2, "screen_samples": 4,
             "screen_prior": str(tmp_path / "prior"), "band_hi": 0.75, "round0_from_screen": True, "tasks": 2,
             "group": 4, "rounds": 3, "per_update": 4, "attempt_replies": 1, "images": False, "eval_every": 2,
             "probes": 2, "checkpoint": True, "adapter_name": "policy", "update_extra": {}, "eval_seconds": 1,
             "trainer_env": {"RL1_PREFETCH": 0}, "episode_cap": 3, "round_play_seconds": 600, "local_count": True}
    study = {"windows": [65536], "arc2_rl": block}
    import time

    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "run1"))
    report = asyncio.run(rl_loop.run_arc2_rl(study, {}, source=source, output=tmp_path / "out1", server_dir=None,
                                             base_url="http://x/v1", deadline_epoch=time.time() + 7200))
    assert report["status"] == "complete", report.get("traceback")
    assert report["screen_prior_episodes"] == 2 and report["screen"]["band"] >= 2
    assert len(report["rounds"]) == 3 and report["rounds"][0]["model"] == "base"
    assert report["rounds"][0]["play_seconds"] == 0  # round 0 trained on the screen groups
    assert report["rounds"][1]["left_out"] == 0
    assert [e["label"] for e in report["evals"]] == ["eval-r02"] and report["eval_base"] and report["eval_final"]
    assert calls.count("score base") == 1 and set(report["kl_floor"]) == {p["name"] for p in report["probes"]}
    assert calls[-6:] == ["score", "save", "checkpoint", "park", "wake", "load policy"]
    ck = tmp_path / "run1/ckpt"
    state = json.loads((ck / "state.json").read_text())
    assert state["round"] == 2 and len(state["rounds"]) == 3 and Path(state["adapter"]).exists()
    assert not (ck / "r00/state").exists() and (ck / "r01/state").exists() and (ck / "r00/adapter").exists()
    calls.clear()
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "run2"))
    monkeypatch.setenv("ARC2_RL_RESUME", str(ck))
    study["arc2_rl"] = {**block, "rounds": 4}
    again = asyncio.run(rl_loop.run_arc2_rl(study, {}, source=source, output=tmp_path / "out2", server_dir=None,
                                            base_url="http://x/v1", deadline_epoch=time.time() + 7200))
    assert again["status"] == "complete", again.get("traceback")
    # the resumed state is scored at once (the trainer forgets unaccepted scores on a load)
    assert calls[:8] == ["sleep", "trainer", "score base", "score", "load_checkpoint", "score", "park", "wake"]
    assert calls[8] == "load policy" and "screen" not in again and [r["round"] for r in again["rounds"]] == [3]
    assert again["rounds"][0]["model"] == "policy" and again["eval_final"]["updates_before"] == 4
    # c240 split (19 Oct): resume onto a new pool and eval set; earlier base rewards seed the new tasks' rates, a 0/2
    # task is tried, the base eval of the new set runs beside the first round
    calls.clear()
    pool2 = pool + [row("t6"), row("t7")]
    (source / "data/rl/pool2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool2))
    (source / "data/rl/evalB.jsonl").write_text(json.dumps(row("t8")) + "\n")
    (source / "data/rl/prior.json").write_text(json.dumps({pool2[6]["id"]: [1, 0], pool2[7]["id"]: [0, 0],
                                                            pool[0]["id"]: [0, 0]}))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "run3"))
    study["arc2_rl"] = {**block, "rounds": 4, "pool": "data/rl/pool2.jsonl", "eval_ids": None,
                        "eval_pool": "data/rl/evalB.jsonl", "prior_hist": "data/rl/prior.json", "eval_base_fresh": True,
                        "try_unsolved": 2, "tasks": 8}
    third = asyncio.run(rl_loop.run_arc2_rl(study, {}, source=source, output=tmp_path / "out3", server_dir=None,
                                            base_url="http://x/v1", deadline_epoch=time.time() + 7200))
    assert third["status"] == "complete", third.get("traceback")
    assert third["prior_hist"]["tasks"] == 2  # t0 keeps the resumed run's rewards
    assert third["eval_base"]["label"] == "eval-base" and third["eval_base"]["episodes"] == 2 and third["evals"] == []
    played = {p.parent.name for p in (tmp_path / "out3/rl/r03").glob("*/*")}
    assert {"arc2-train_t6", "arc2-train_t7"} <= played and third["eval_final"]["episodes"] == 2
    study["arc2_rl"]["eval_pool"] = "data/rl/pool2.jsonl"
    study["arc2_rl"]["eval_tasks"] = 1
    with pytest.raises(ValueError):
        asyncio.run(rl_loop.run_arc2_rl(study, {}, source=source, output=tmp_path / "out4", server_dir=None,
                                        base_url="http://x/v1", deadline_epoch=time.time() + 7200))


def test_screen_only_trial(tmp_path, monkeypatch):
    """arc2-screen: base episodes of every pool task, first pass before the second, no logprobs asked."""
    model = Model()
    bodies = []
    real = httpx.AsyncClient

    def spy(request):
        if request.url.path == "/v1/chat/completions":
            bodies.append(json.loads(request.content))
        return model(request)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(**{**kw, "transport": httpx.MockTransport(spy)}))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "run"))
    source = tmp_path / "bundle"
    (source / "data/rl").mkdir(parents=True)
    pool = [row(f"t{i}") for i in range(3)]
    (source / "data/rl/pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool))
    study = {"windows": [65536], "arc2_rl": {"pool": "data/rl/pool.jsonl", "screen_samples": 2, "attempt_replies": 1,
                                             "images": False, "episode_cap": 2}}
    import time

    report = asyncio.run(rl_loop.run_screen(study, {"name": "screen"}, source=source, output=tmp_path / "out",
                                            base_url="http://x/v1", deadline_epoch=time.time() + 3600))
    assert report["status"] == "complete" and report["episodes"] == 6 and report["tasks_done"] == 3
    assert all("top_logprobs" not in b for b in bodies) and sum(report["counts_by_solved"].values()) == 3
    live = json.loads((tmp_path / "run/arena/live/arc2-screen.json").read_text())
    assert live["episodes"] == 6 and live["by_source"]["arc2-train"]["tasks"] == 3


def test_screen_top_up_in_pool_order(tmp_path, monkeypatch):
    """screen_have: a task plays only its missing episodes, numbered after them; passes by global index, pool order."""
    model = Model()
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(**{**kw, "transport": httpx.MockTransport(model)}))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "run"))
    source = tmp_path / "bundle"
    (source / "data/rl").mkdir(parents=True)
    pool = [row(f"t{i}") for i in range(3)]
    (source / "data/rl/pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool))
    (source / "data/rl/have.json").write_text(json.dumps({"arc2-train:t1": 1, "arc2-train:t2": 2}))
    study = {"windows": [65536], "arc2_rl": {"pool": "data/rl/pool.jsonl", "screen_samples": 2, "attempt_replies": 1,
                                             "images": False, "episode_cap": 1, "screen_shuffle": False,
                                             "screen_have": "data/rl/have.json"}}
    import time

    report = asyncio.run(rl_loop.run_screen(study, {"name": "screen"}, source=source, output=tmp_path / "out",
                                            base_url="http://x/v1", deadline_epoch=time.time() + 3600))
    assert report["status"] == "complete" and report["episodes"] == 3 and report["episodes_planned"] == 3
    files = sorted((tmp_path / "out/rl/screen").glob("*/*/episode.json"), key=lambda p: p.stat().st_mtime_ns)
    keys = [str(p.parent.relative_to(tmp_path / "out/rl/screen")) for p in files]  # one at a time: play order
    assert keys == ["arc2-train_t0/0", "arc2-train_t0/1", "arc2-train_t1/1"]


def test_loo_and_push_balance():
    assert rl_loop.advantages({"a": [1, 0, 0, 1]}, loo=True) == {"a": [0.666667, -0.666667, -0.666667, 0.666667]}
    specs = [{"w": 0.5, "tokens": 100}, {"w": -0.5, "tokens": 400}, {"w": -0.25, "tokens": 200}]
    assert rl_loop.balance(specs) == 0.2
    up = sum(s["w"] * s["tokens"] for s in specs if s["w"] > 0)
    assert abs(sum(-s["w"] * s["tokens"] for s in specs if s["w"] < 0) - up) < 1e-6
    assert rl_loop.balance([{"w": 1.0, "tokens": 5}]) == 1.0


@pytest.mark.parametrize("kl_floor", [0.0, 0.02, {"untrained-0": 0.02}])
def test_pilot_fixed_tasks_kl_controller(tmp_path, monkeypatch, kl_floor):
    """Pilot mode: the same tasks every round (no screen), LOO + push balance, the per-step KL controller (raise lr
    after a quiet update, roll back a jumpy one and serve the last adapter), then a play-only measuring round. A
    configured perturbation floor is subtracted (0.021 reads as quiet over a 0.02 floor)."""
    lift = kl_floor if isinstance(kl_floor, float) else 0.02
    calls, kl = [], iter([0.001 + lift, 0.001 + lift, 0.2 + lift])

    class Server:
        def __init__(self, base_url, model):
            self.root = "http://x"

        async def sleep(self):
            return 0.0

        async def wake(self):
            calls.append("wake")
            return 0.0

        async def load(self, name, path):
            calls.append(f"load {name}")
            return 0.0

    class Trainer:
        def __init__(self, folder, command, env, log):
            pass

        async def ready(self, timeout):
            return {}

        async def must(self, op, timeout=3600, **payload):
            calls.append(op + (f" lr={payload['lr']:.3g}" if op == "update" else "")
                         + (" hold" if payload.get("hold_previous") else ""))
            if op == "score":
                value = next(kl) if payload.get("hold_previous") else 0.0
                return {"windows": {p["name"]: {"kl_base": 0.0, "kl_previous": 1.0 if p["name"][:4] in ("won-", "lost", "held")
                                                else value, "mismatch": {"trainer_nll": 0.3}} for p in payload["windows"]}}
            if op in ("save", "checkpoint"):
                Path(payload["path"]).mkdir(parents=True, exist_ok=True)
            if op == "update":
                assert all(a["w"] != 0 and "tokens" not in a for a in payload["attempts"])
                return {"trained_tokens": 10, "train_seconds": 1.0, "windows": []}
            return {"path": payload.get("path")}

        call = must

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "arena.rl_loop", types.SimpleNamespace(
        Server=Server, Trainer=Trainer, trainer_command=lambda c, s, w: (["x"], {})))
    model = Model()
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(**{**kw, "transport": httpx.MockTransport(model)}))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "run"))
    source = tmp_path / "bundle"
    (source / "data/rl").mkdir(parents=True)
    pool = [row(f"t{i}") for i in range(3)]
    (source / "data/rl/pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool))
    (source / "data/rl/fixed.json").write_text(json.dumps([pool[0]["id"], pool[1]["id"]]))
    probe = tmp_path / "probe.jsonl"
    probe.write_text("{}\n")
    block = {"fixed_tasks": "data/rl/fixed.json", "group": 4, "rounds": 3, "update_rounds": 2, "per_update": 8,
             "attempt_replies": 1, "images": False, "update_extra": {}, "advantage": "loo", "push_balance": True,
             "kl_target": 0.02, "lr": 2e-5, "probe_rollouts": [str(probe)], "adapter_name": "policy",
             "kl_floor": kl_floor, "trained_probes": 2, "held_successes": True,
             "reserve_seconds": 100, "eval_seconds": 1950}  # no eval pool: the eval estimate must not stop round 2
    import time

    report = asyncio.run(rl_loop.run_arc2_rl({"windows": [65536], "arc2_rl": block}, {}, source=source,
                                             output=tmp_path / "out", server_dir=None, base_url="http://x/v1",
                                             deadline_epoch=time.time() + 2000))
    assert report["status"] == "complete", report.get("traceback")
    rounds = report["rounds"]
    assert [r["measure"] for r in rounds] == [False, False, True] and report["screen"]["tasks"] == 0
    assert rounds[0]["control"]["action"] == "full" and abs(rounds[0]["control"]["lr_next"] - 3e-5) < 1e-12
    assert rounds[1]["control"]["action"] == "rollback" and abs(rounds[1]["control"]["lr_next"] - 2e-5) < 1e-12
    assert "saved" not in rounds[1] and rounds[2]["model"] == "policy" and rounds[2]["trained_episodes"] == 0
    assert calls.count("load_checkpoint") == 1 and calls.count("accept_reference") == 1
    assert calls.count("load policy") == 1 and rounds[0]["push_scale"] is not None
    names = [p["name"] for p in report["probes"]]
    assert names[0] == "untrained-0" and {n.split("-")[0] for n in names[1:]} == {"won", "lost", "held"}, names
    assert set(rounds[0]["probes"]) == set(names)
    assert rounds[0]["replies"]["replies"] > 0 and 0 <= rounds[0]["replies"]["at_cap_share"] <= 1
    held = [n for n in names if n.startswith("held-")]
    assert set(report.get("held_ref") or {}) == set(held)
    if held:  # held successes are scored each round and never trained
        assert rounds[0]["held"]["windows"] == len(held)


class TrainerRefs:
    """rl1_trainer's probe references (score_windows, op_accept_reference, forget_unaccepted), as the real trainer
    enforces them: a probe once scored is tracked, a load forgets every unaccepted score, and accept_reference fails
    while any tracked probe lacks a fresh one (P5b crashed on it after its resume)."""

    def __init__(self):
        self.refs = {}

    def apply(self, op, payload):
        if op == "score":
            for w in payload["windows"]:
                refs = self.refs.setdefault(w["name"], set())
                refs.add("last")
                if payload.get("label") == "base" or not payload.get("hold_previous"):
                    refs.add("previous")
        elif op == "load_checkpoint":
            for refs in self.refs.values():
                refs.discard("last")
        elif op == "accept_reference":
            missing = sorted(n for n, r in self.refs.items() if "last" not in r)
            if missing:
                raise ValueError(f"no scored state to accept for {missing}")


def test_p2_looser_drift_rules(tmp_path, monkeypatch):
    """P2 rules: no per-step stop at 1.5 x target, rollback only past 0.1 nat of excess, and a play
    guard: a round whose play breaks against round 0 (reply length over 2 x base) undoes the last update, trains
    nothing and serves the restored state."""
    kl_floor = 0.0
    calls, kl = [], iter([0.05, 0.05, 0.15, 0.02, 0.02])  # round 0 step + end read; round 2 step; round 3 both
    refs = TrainerRefs()

    class Server:
        def __init__(self, base_url, model):
            self.root = "http://x"

        async def sleep(self):
            return 0.0

        async def wake(self):
            calls.append("wake")
            return 0.0

        async def load(self, name, path):
            calls.append(f"load {name}")
            return 0.0

    class Trainer:
        def __init__(self, folder, command, env, log):
            pass

        async def ready(self, timeout):
            return {}

        async def must(self, op, timeout=3600, **payload):
            calls.append(op + (f" lr={payload['lr']:.3g}" if op == "update" else "")
                         + (" hold" if payload.get("hold_previous") else ""))
            refs.apply(op, payload)
            if op == "score":
                value = next(kl) if payload.get("hold_previous") else 0.0
                return {"windows": {p["name"]: {"kl_base": 0.0, "kl_previous": 1.0 if p["name"][:4] in ("won-", "lost", "held")
                                                else value, "mismatch": {"trainer_nll": 0.3}} for p in payload["windows"]}}
            if op in ("save", "checkpoint"):
                Path(payload["path"]).mkdir(parents=True, exist_ok=True)
            if op == "update":
                assert all(a["w"] != 0 and "tokens" not in a for a in payload["attempts"])
                return {"trained_tokens": 10, "train_seconds": 1.0, "windows": []}
            return {"path": payload.get("path")}

        call = must

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "arena.rl_loop", types.SimpleNamespace(
        Server=Server, Trainer=Trainer, trainer_command=lambda c, s, w: (["x"], {})))
    stats = iter([{"replies": 10, "mean_tokens": 1000, "at_cap_share": 0.1, "failures": 2, "no_code_share": 0.05},
                  {"replies": 10, "mean_tokens": 2500, "at_cap_share": 0.1, "failures": 2, "no_code_share": 0.05},
                  {"replies": 10, "mean_tokens": 1100, "at_cap_share": 0.1, "failures": 2, "no_code_share": 0.05},
                  {"replies": 10, "mean_tokens": 1050, "at_cap_share": 0.1, "failures": 2, "no_code_share": 0.05}])
    monkeypatch.setattr(rl_loop, "reply_stats", lambda paths: next(stats))
    model = Model()
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(**{**kw, "transport": httpx.MockTransport(model)}))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "run"))
    source = tmp_path / "bundle"
    (source / "data/rl").mkdir(parents=True)
    pool = [row(f"t{i}") for i in range(3)]
    (source / "data/rl/pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool))
    (source / "data/rl/fixed.json").write_text(json.dumps([pool[0]["id"], pool[1]["id"]]))
    probe = tmp_path / "probe.jsonl"
    probe.write_text("{}\n")
    block = {"fixed_tasks": "data/rl/fixed.json", "group": 4, "rounds": 4, "per_update": 8,
             "attempt_replies": 1, "images": False, "update_extra": {}, "advantage": "loo", "push_balance": True,
             "kl_target": 0.03, "kl_stop_at": None, "kl_rollback_excess": 0.1, "play_guard": True,
             "kl_stop": None, "lr": 2e-5, "probe_rollouts": [str(probe)], "adapter_name": "policy",
             "kl_floor": kl_floor, "trained_probes": 2, "held_successes": True,
             "reserve_seconds": 100, "eval_seconds": 1950}  # no eval pool: the eval estimate must not stop round 2
    import time

    report = asyncio.run(rl_loop.run_arc2_rl({"windows": [65536], "arc2_rl": block}, {}, source=source,
                                             output=tmp_path / "out", server_dir=None, base_url="http://x/v1",
                                             deadline_epoch=time.time() + 2000))
    assert report["status"] == "complete", report.get("traceback")
    rounds = report["rounds"]
    # every load is followed by a score of all probes, so the round after either rollback accepts its update
    assert [r.get("control", {}).get("action") for r in rounds] == ["full", "rollback_play", "rollback", "full"], rounds
    loads = [i for i, c in enumerate(calls) if c == "load_checkpoint"]
    assert loads and all(calls[i + 1] == "score" for i in loads)
    assert rounds[0]["control"]["excess"] == 0.05  # above 1.5 x 0.03, kept: no per-step stop
    assert rounds[1]["trained_episodes"] == 0 and rounds[1]["play_guard"] == {"mean_tokens": [2500, 1000]}
    assert "saved" in rounds[1] and "saved" not in rounds[2]
    assert calls.count("load_checkpoint") == 2 and report["base_replies"]["mean_tokens"] == 1000
    assert "stop" not in report



def test_p5_positives_only_warmup_and_round0_from_prior(tmp_path, monkeypatch):
    """P5: round 0 trains on an earlier run's base episodes of the fixed tasks (no round-0 play),
    updates are positives only (signed False), the run's first two steps warm up to lr, and the step clock is kept
    in state.json."""
    calls, updates = [], []

    class Server:
        def __init__(self, base_url, model):
            self.root = "http://x"

        async def sleep(self):
            return 0.0

        async def wake(self):
            return 0.0

        async def load(self, name, path):
            return 0.0

    class Trainer:
        def __init__(self, folder, command, env, log):
            pass

        async def ready(self, timeout):
            return {}

        async def must(self, op, timeout=3600, **payload):
            calls.append(op)
            if op == "score":
                return {"windows": {p["name"]: {"kl_base": 0.0, "kl_previous": 0.001, "mismatch": {"trainer_nll": 0.3}}
                                    for p in payload["windows"]}}
            if op in ("save", "checkpoint"):
                Path(payload["path"]).mkdir(parents=True, exist_ok=True)
            if op == "update":
                updates.append((payload["lr"], payload["signed"], len(payload["attempts"])))
                return {"trained_tokens": 10, "train_seconds": 1.0, "windows": []}
            return {"path": payload.get("path")}

        call = must

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "arena.rl_loop", types.SimpleNamespace(
        Server=Server, Trainer=Trainer, trainer_command=lambda c, s, w: (["x"], {})))
    model = Model()
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(**{**kw, "transport": httpx.MockTransport(model)}))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "run"))
    source = tmp_path / "bundle"
    (source / "data/rl").mkdir(parents=True)
    pool = [row(f"t{i}") for i in range(3)]
    (source / "data/rl/pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool))
    (source / "data/rl/fixed.json").write_text(json.dumps([pool[0]["id"], pool[1]["id"]]))
    prior = tmp_path / "p4/r00"
    for t, task in enumerate(pool[:2]):
        for k in range(4):  # 2 of 4 solved per task; one cut episode left out
            folder = prior / task["id"].replace(":", "_") / str(k)
            folder.mkdir(parents=True)
            (folder / "rollout.jsonl").write_text("{}\n")
            done = None if (t, k) == (1, 3) else ("verified" if k < 2 else "replies")
            (folder / "episode.json").write_text(json.dumps({"task": task["id"], "reward": 1.0 if k < 2 else 0.0,
                                                             "spent": 100 + k, "done": done, "reply_tokens": [50],
                                                             "at_cap": 0, "no_code": 0}))
    probe = tmp_path / "probe.jsonl"
    probe.write_text("{}\n")
    block = {"fixed_tasks": "data/rl/fixed.json", "group": 4, "rounds": 3, "update_rounds": 2, "per_update": 2,
             "attempt_replies": 1, "images": False, "update_extra": {}, "advantage": "loo", "push_balance": False,
             "signed": False, "warmup_steps": 2, "screen_prior": str(prior), "round0_from_screen": True,
             "kl_target": 0.03, "kl_stop_at": None, "kl_rollback_excess": 0.1, "play_guard": True, "kl_stop": None,
             "lr": 3e-5, "lr_max": 3e-5, "probe_rollouts": [str(probe)], "adapter_name": "policy", "checkpoint": True,
             "reserve_seconds": 100, "eval_seconds": 0}
    import time

    report = asyncio.run(rl_loop.run_arc2_rl({"windows": [65536], "arc2_rl": block}, {}, source=source,
                                             output=tmp_path / "out", server_dir=None, base_url="http://x/v1",
                                             deadline_epoch=time.time() + 2000))
    assert report["status"] == "complete", report.get("traceback")
    assert report["screen_prior_episodes"] == 7
    rounds = report["rounds"]
    assert not (tmp_path / "out/rl/r00").exists()  # round 0 played nothing
    assert (tmp_path / "out/rl/r01").is_dir() and (tmp_path / "out/rl/r02").is_dir()
    assert rounds[0]["trained_episodes"] == 7 and rounds[0]["positive_episodes"] == 4
    assert rounds[0]["replies"]["mean_tokens"] == 50
    assert report["base_replies"]["mean_tokens"] == 50
    assert all(signed is False for _, signed, _ in updates)
    lrs = [u["lr"] for r in rounds for u in r["updates"]]
    assert lrs[:3] == [1.5e-5, 3e-5, 3e-5] and len(lrs) == len(updates)
    assert [u["step"] for r in rounds for u in r["updates"]] == list(range(1, len(lrs) + 1))
    assert rounds[2]["measure"] and not rounds[2]["updates"]
    state = json.loads((tmp_path / "run/ckpt/state.json").read_text())
    assert state["steps"] == len(lrs)


def test_p5b_resume_sets_lr_and_keeps_guard_references(tmp_path, monkeypatch):
    """P5b: a resume from P5's round-0 checkpoint runs its next round at resume_lr, keeps the play
    guard's base replies and the held successes' references, and can roll a broken play back to the pre-update state."""
    calls, updates, refs = [], [], TrainerRefs()

    class Server:
        def __init__(self, base_url, model):
            self.root = "http://x"

        async def sleep(self):
            return 0.0

        async def wake(self):
            return 0.0

        async def load(self, name, path):
            return 0.0

    class Trainer:
        def __init__(self, folder, command, env, log):
            pass

        async def ready(self, timeout):
            return {}

        async def must(self, op, timeout=3600, **payload):
            calls.append((op, payload.get("path")))
            refs.apply(op, payload)
            if op == "score":
                return {"windows": {p["name"]: {"kl_base": 0.0, "kl_previous": 0.001, "mismatch": {"trainer_nll": 0.25}}
                                    for p in payload["windows"]}}
            if op in ("save", "checkpoint"):
                Path(payload["path"]).mkdir(parents=True, exist_ok=True)
            if op == "update":
                updates.append(payload["lr"])
                return {"trained_tokens": 10, "train_seconds": 1.0, "windows": []}
            return {"path": payload.get("path")}

        call = must

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "arena.rl_loop", types.SimpleNamespace(
        Server=Server, Trainer=Trainer, trainer_command=lambda c, s, w: (["x"], {})))
    model = Model()
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(**{**kw, "transport": httpx.MockTransport(model)}))
    source = tmp_path / "bundle"
    (source / "data/rl").mkdir(parents=True)
    pool = [row(f"t{i}") for i in range(3)]
    (source / "data/rl/pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool))
    (source / "data/rl/fixed.json").write_text(json.dumps([pool[0]["id"], pool[1]["id"]]))
    probe = tmp_path / "probe.jsonl"
    probe.write_text("{}\n")
    block = {"fixed_tasks": "data/rl/fixed.json", "group": 4, "rounds": 1, "per_update": 2, "attempt_replies": 1,
             "images": False, "update_extra": {}, "advantage": "loo", "signed": False, "warmup_steps": 2,
             "kl_target": 0.03, "kl_stop_at": None, "kl_rollback_excess": 0.1, "play_guard": True, "kl_stop": None,
             "lr": 3e-5, "lr_max": 3e-5, "probe_rollouts": [str(probe)], "adapter_name": "policy", "checkpoint": True,
             "held_successes": True, "reserve_seconds": 100, "eval_seconds": 0}
    import time

    first = tmp_path / "p5"
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(first))
    report = asyncio.run(rl_loop.run_arc2_rl({"windows": [65536], "arc2_rl": block}, {}, source=source,
                                             output=first / "arena/trials/rl-loop", server_dir=None,
                                             base_url="http://x/v1", deadline_epoch=time.time() + 2000))
    assert report["status"] == "complete", report.get("traceback")
    state = json.loads((first / "ckpt/state.json").read_text())
    assert state["base_replies"] == report["base_replies"] and state["held_ref"] == report.get("held_ref", {})
    state.pop("base_replies"), state.pop("held_ref")  # as P5's state, written before these keys existed
    (first / "ckpt/state.json").write_text(json.dumps(state))
    (first / "arena/trials/rl-loop/rl/status.json").write_text(json.dumps(
        {"base_replies": report["base_replies"], "held_ref": {"held-x": 0.5}}))
    updates.clear(), calls.clear()
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "p5b"))
    monkeypatch.setenv("ARC2_RL_RESUME", str(first / "ckpt"))
    resumed = asyncio.run(rl_loop.run_arc2_rl(
        {"windows": [65536], "arc2_rl": {**block, "rounds": 3, "update_rounds": 2, "resume_lr": 1e-4, "lr_max": 1.5e-4}},
        {}, source=source, output=tmp_path / "p5b-out", server_dir=None, base_url="http://x/v1",
        deadline_epoch=time.time() + 2000))
    assert resumed["status"] == "complete", resumed.get("traceback")
    assert [r["round"] for r in resumed["rounds"]] == [1, 2]
    assert updates and updates[0] == 1e-4 and max(updates) <= 1.5e-4  # no warm-up again: the step clock resumed
    assert resumed["base_replies"] == report["base_replies"] and resumed["held_ref"]["held-x"] == 0.5
    assert "play_guard" in resumed["rounds"][0]
    pre = [p for op, p in calls if op == "checkpoint" and p and p.endswith("pre-update")]
    load = [i for i, (op, _) in enumerate(calls) if op == "load_checkpoint"]
    assert pre and calls.index(("checkpoint", pre[0])) < load[0]  # the zero state saved before the old state loads

def test_p5c_dose_rule_sets_each_rounds_lr(tmp_path, monkeypatch):
    """P5c: after a resume at 1e-4, each later round's lr comes from its measured excess (KL grows
    with lr squared: lr x sqrt(target / excess), within lr_max), with the trainer's reference rule enforced."""
    updates, refs = [], TrainerRefs()

    class Server:
        def __init__(self, base_url, model):
            self.root = "http://x"

        async def sleep(self):
            return 0.0

        async def wake(self):
            return 0.0

        async def load(self, name, path):
            return 0.0

    class Trainer:
        def __init__(self, folder, command, env, log):
            pass

        async def ready(self, timeout):
            return {}

        async def must(self, op, timeout=3600, **payload):
            refs.apply(op, payload)
            if op == "score":
                value = 0.0037 if payload.get("hold_previous") else 0.0
                return {"windows": {p["name"]: {"kl_base": 0.0, "kl_previous": value,
                                                "mismatch": {"trainer_nll": 0.25}} for p in payload["windows"]}}
            if op in ("save", "checkpoint"):
                Path(payload["path"]).mkdir(parents=True, exist_ok=True)
            if op == "update":
                updates.append(payload["lr"])
                return {"trained_tokens": 10, "train_seconds": 1.0, "windows": []}
            return {"path": payload.get("path")}

        call = must

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "arena.rl_loop", types.SimpleNamespace(
        Server=Server, Trainer=Trainer, trainer_command=lambda c, s, w: (["x"], {})))
    model = Model()
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(**{**kw, "transport": httpx.MockTransport(model)}))
    source = tmp_path / "bundle"
    (source / "data/rl").mkdir(parents=True)
    pool = [row(f"t{i}") for i in range(3)]
    (source / "data/rl/pool.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pool))
    (source / "data/rl/fixed.json").write_text(json.dumps([pool[0]["id"], pool[1]["id"]]))
    probe = tmp_path / "probe.jsonl"
    probe.write_text("{}\n")
    block = {"fixed_tasks": "data/rl/fixed.json", "group": 4, "rounds": 1, "per_update": 2, "attempt_replies": 1,
             "images": False, "update_extra": {}, "advantage": "loo", "signed": False, "warmup_steps": 2,
             "kl_target": 0.03, "kl_stop_at": None, "kl_rollback_excess": 0.1, "play_guard": True, "kl_stop": None,
             "lr": 3e-5, "lr_max": 3e-5, "probe_rollouts": [str(probe)], "adapter_name": "policy", "checkpoint": True,
             "held_successes": True, "reserve_seconds": 100, "eval_seconds": 0}
    import time

    first = tmp_path / "p5"
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(first))
    report = asyncio.run(rl_loop.run_arc2_rl({"windows": [65536], "arc2_rl": block}, {}, source=source,
                                             output=first / "arena/trials/rl-loop", server_dir=None,
                                             base_url="http://x/v1", deadline_epoch=time.time() + 2000))
    assert report["status"] == "complete", report.get("traceback")
    updates.clear()
    refs.refs.clear()  # a new trainer process
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "p5c"))
    monkeypatch.setenv("ARC2_RL_RESUME", str(first / "ckpt"))
    resumed = asyncio.run(rl_loop.run_arc2_rl(
        {"windows": [65536], "arc2_rl": {**block, "rounds": 4, "update_rounds": 3, "resume_lr": 1e-4, "lr_max": 3e-4,
                                         "lr_rule": "dose"}},
        {}, source=source, output=tmp_path / "p5c-out", server_dir=None, base_url="http://x/v1",
        deadline_epoch=time.time() + 2000))
    assert resumed["status"] == "complete", resumed.get("traceback")
    rounds = resumed["rounds"]
    assert [r["round"] for r in rounds] == [1, 2, 3] and rounds[2]["measure"]
    dosed = 1e-4 * (0.03 / 0.0037) ** 0.5  # 2.85e-4: under lr_max
    assert {u["lr"] for u in rounds[0]["updates"]} == {1e-4}
    assert all(abs(u["lr"] - dosed) < 1e-12 for u in rounds[1]["updates"])
    assert rounds[0]["control"]["rule"] == "dose" and abs(rounds[0]["control"]["lr_next"] - dosed) < 1e-12
    assert rounds[1]["control"]["lr_next"] == 3e-4  # 2.85e-4 x 2.85 capped at lr_max


def test_dose_curve(tmp_path, monkeypatch):
    """P0: held-back successes, LOO + balance, each lr from the same fresh adapter, a read after every step."""
    calls = []

    class Trainer:
        def __init__(self, folder, command, env, log):
            assert env["RL1_DENSE_ONLY"] == "1"

        async def ready(self, timeout):
            return {}

        async def must(self, op, timeout=3600, **payload):
            calls.append(op)
            if op == "score":
                return {"windows": {p["name"]: {"kl_base": 0.01, "mismatch": {"trainer_nll": 0.3}} for p in payload["windows"]}}
            if op == "update":
                assert len(payload["attempts"]) == 3 and all(a["w"] != 0 for a in payload["attempts"])
                return {"trained_tokens": 5}
            return {}

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "arena.rl_loop", types.SimpleNamespace(
        Trainer=Trainer, trainer_command=lambda c, s, w: (["x"], {})))
    for t, rewards in {"a": [1, 0, 1, 0, 0, 1], "b": [1, 1, 1, 1, 1, 1], "c": [0, 0, 0, 0, 0, 1, 0]}.items():
        for k, r in enumerate(rewards):
            d = tmp_path / "prior" / f"arc2-train_{t}" / str(k)
            d.mkdir(parents=True)
            (d / "episode.json").write_text(json.dumps({"task": f"arc2-train:{t}", "reward": r, "spent": 10 + k,
                                                        "done": "verified" if r else "replies"}))
            (d / "rollout.jsonl").write_text("{}\n")
    out = tmp_path / "out"
    out.mkdir()
    report = asyncio.run(rl_loop.dose_curve(cfg(update_extra={}), tmp_path, tmp_path / "prior", out,
                                            lrs=(1e-5, 1e-4), steps=2, batch=3))
    assert report["status"] == "complete", report.get("traceback")
    assert report["tasks"] == ["arc2-train:a"] and report["held"] == 1 and report["train_episodes"] == 5
    assert report["probes"] == ["untrained-arc2-train:b"] and set(report["curves"]) == {"1e-05", "0.0001"}
    assert calls.count("load_checkpoint") == 2 and calls.count("update") == 4 and len(report["curves"]["1e-05"]) == 2
    assert report["curves"]["0.0001"][1]["held_nll_drop"] == 0.0


def test_in_band_probability():
    assert rl_loop.in_band_probability([1, 0, 1, 0], (0.5, 0.5)) > 0.9
    assert rl_loop.in_band_probability([1] * 8, (0.5, 0.5)) < 0.3
    assert rl_loop.in_band_probability([0] * 8, (0.5, 0.5)) < 0.3
    assert abs(rl_loop.in_band_probability([], (1, 1)) - 0.8) < 1e-3


def test_dynamic_sampling_fills_spread_groups(tmp_path, monkeypatch):
    """Signal yield: saturated and never-solved tasks leave the band after the screen; a round whose first batch has
    too few groups with spread plays more band tasks, and at most `tasks` spread groups are trained."""
    played = []

    async def play(rows, samples, *, label, **kw):
        out = {}
        for r in rows:
            n = samples.get(r["id"], 0) if isinstance(samples, dict) else samples
            kind = r["id"].split(":")[1].rstrip("0123456789")
            played.append((label, r["id"]))
            rewards = ([1.0] * n if kind == "easy" else [0.0] * n if kind == "hard" or
                       (kind == "trap" and label != "screen") else [float(k % 2 == 0) for k in range(n)])
            out[r["id"]] = [(x, f"/x/{r['id']}/{k}", 100) for k, x in enumerate(rewards)]
        return out

    class Server:
        def __init__(self, base_url, model):
            self.root = "http://x"

        async def sleep(self):
            return 0.0

        async def wake(self):
            return 0.0

        async def load(self, name, path):
            return 0.0

    class Trainer:
        def __init__(self, folder, command, env, log):
            pass

        async def ready(self, timeout):
            return {}

        async def must(self, op, timeout=3600, **payload):
            if op in ("save", "checkpoint"):
                Path(payload["path"]).mkdir(parents=True, exist_ok=True)
            if op == "score":
                return {"windows": {}}
            if op == "update":
                return {"trained_tokens": 10, "train_seconds": 1.0, "windows": []}
            return {"path": payload.get("path")}

        call = must

        def close(self):
            pass

    monkeypatch.setattr(rl_loop, "play", play)
    monkeypatch.setitem(sys.modules, "arena.rl_loop", types.SimpleNamespace(
        Server=Server, Trainer=Trainer, trainer_command=lambda c, s, w: (["x"], {})))
    model = Model()
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(**{**kw, "transport": httpx.MockTransport(model)}))
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "run"))
    source = tmp_path / "bundle"
    (source / "data/rl").mkdir(parents=True)
    names = [f"{k}{i}" for k in ("mid", "trap") for i in range(4)] + ["easy0", "easy1", "hard0", "hard1"]
    (source / "data/rl/pool.jsonl").write_text("".join(json.dumps(row(n)) + "\n" for n in names))
    block = {"group": 4, "rounds": 1, "tasks": 2, "screen_samples": 4, "per_update": 8, "attempt_replies": 1,
             "images": False, "update_extra": {}, "advantage": "loo", "spread_prior": {"arc2-train": [0.5, 0.5]},
             "spread_min": 0.6, "min_spread": 2, "extra_play": 3}
    import time

    report = asyncio.run(rl_loop.run_arc2_rl({"windows": [65536], "arc2_rl": block}, {}, source=source,
                                             output=tmp_path / "out", server_dir=None, base_url="http://x/v1",
                                             deadline_epoch=time.time() + 7200))
    assert report["status"] == "complete", report.get("traceback")
    r0 = report["rounds"][0]
    rounds_played = {t for label, t in played if label != "screen"}
    assert not any(":easy" in t or ":hard" in t for t in rounds_played)
    assert r0["groups_with_spread"] >= 2 and r0["groups_trained"] == 2 and r0["trained_episodes"] == 8
    first = [t for label, t in played if label == "r00"]
    assert sum(":mid" in t for t in first) < 2 and any(label == "r00x1" for label, _ in played)  # seeded draw


def test_episode_token_cap_is_a_valid_failure(tmp_path):
    model = Model()

    def wrong(request):  # every reply WRONG (odd call counts)
        model.calls = 1
        return model(request)

    async def go():
        client = httpx.AsyncClient(base_url="http://x", transport=httpx.MockTransport(wrong))
        transport = rl_loop.RLTransport(client, "m", "m", "medium", 0.7, 4)
        execute, explore = await sandboxed()
        out = await rl_loop.play([row("good")], 1, transport=transport, execute=execute, explore=explore,
                                 cfg=cfg(attempt_replies=3, episode_tokens=4), folder=tmp_path, label="r00")
        await client.aclose()
        return out

    out = asyncio.run(go())
    reward, path, spent = out["arc2-train:good"][0]
    episode = json.loads((Path(path).parent / "episode.json").read_text())
    assert reward == 0.0 and episode["done"] == "cap" and episode["valid"] and spent == 6


def test_pick_tries(tmp_path):
    for task, sizes in {"t2": [5, 0, 5, 5], "t1": [5, 5, 5]}.items():
        for k, size in enumerate(sizes):
            d = tmp_path / task / str(k)
            d.mkdir(parents=True)
            (d / "rollout.jsonl").write_text("x" * size)
    (tmp_path / "t1" / "10").mkdir()  # no rollout: skipped
    picked = rl_loop.pick_tries(tmp_path, 2)
    assert [str(Path(p).relative_to(tmp_path)) for p in picked] == [
        "t1/0/rollout.jsonl", "t1/1/rollout.jsonl", "t2/0/rollout.jsonl", "t2/2/rollout.jsonl"]


def test_chunk_check_windows_pool_and_resume(tmp_path, monkeypatch):
    """rl1-chunk port: one trainer call per window, chunk.json after each, pooled gaps token-weighted, a resumed run
    keeps finished windows and redoes a window with a failed setting."""
    calls = []
    settings = [{"chunk": 16384, "select": "qsa"}, {"chunk": 16384, "select": "qsa", "tag": "qsa@16384#2"},
                {"chunk": 2048, "select": "qsa"}, {"chunk": 16384, "select": "dense"}]

    def cell(gap=None, default=None, count=10, abs_sum=1.0, log_s=(0.01, -0.03)):
        out = {"seconds": 1.0, "sampler": {"tokens": count, "mean_abs": 0.05, "mean": 0.01, "beyond_half": 0.0}}
        for label, against in (("gap", gap), ("gap_default", default)):
            if against:
                out[label] = {"against": against, "replies": {"log_s": list(log_s)},
                              **{f: {"sum": 0.5, "abs_sum": abs_sum, "abs_max": abs_sum / 2, "beyond_half": 1, "count": count}
                                 for f in ("actor", "raw")}}
        return out

    class Trainer:
        def __init__(self, folder, command, env, log):
            assert env["RL1_DENSE_ONLY"] == "1" and env["RL1_SWITCH_INTERVAL"] == "0.0001"

        async def ready(self, timeout):
            return {}

        async def must(self, op, timeout=3600, **payload):
            assert op == "chunk_check" and payload["settings"] == settings and len(payload["attempts"]) == 1
            a = payload["attempts"][0]
            assert a["floor"] == 1024 and a["window"] == 32768 and a["w"] == 1.0
            calls.append(a["name"])
            cells = {"qsa@16384": cell(), "qsa@16384#2": cell("qsa@16384", "qsa@16384", abs_sum=0.0),
                     "qsa@2048": cell("qsa@16384", "qsa@16384", abs_sum=2.0 if len(calls) == 1 else 4.0),
                     "dense@16384": cell(None, "qsa@16384")}
            if a["name"] == "t2-0" and len(calls) <= 3:
                cells["dense@16384"] = {"error": "RuntimeError('boom')", "seconds": 0.1}
            return {"windows": [{"name": a["name"], "tokens": 100, "tail": 60, "cut": 40, "targets": 10, "settings": cells}]}

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "arena.rl_loop", types.SimpleNamespace(
        Trainer=Trainer, trainer_command=lambda c, s, w: (["x"], {})))
    rollouts = [str(tmp_path / f"src/{t}/{k}/rollout.jsonl") for t, k in (("t1", 0), ("t1", 1), ("t2", 0))]
    cfg = {**rl_loop.DEFAULTS, "trainer_env": {"RL1_SWITCH_INTERVAL": "0.0001"}, "trainer_timeout": 60}
    out1 = tmp_path / "a"
    out1.mkdir()
    report = asyncio.run(rl_loop.chunk_check(cfg, tmp_path, rollouts, out1, settings=settings))
    assert report["status"] == "complete", report.get("traceback")
    assert calls == ["t1-0", "t1-1", "t2-0"]
    pooled = json.loads((out1 / "chunk.json").read_text())["pooled"]
    assert pooled["qsa@2048"]["gap"]["actor"]["mean_abs"] == round((2.0 + 4.0 + 4.0) / 30, 6)
    assert pooled["qsa@2048"]["gap"]["against"] == "qsa@16384" and pooled["qsa@16384#2"]["gap"]["raw"]["mean_abs"] == 0.0
    assert pooled["qsa@2048"]["gap"]["reply_log_s"]["max"] == 0.03 and pooled["dense@16384"]["windows"] == 2
    assert "gap" not in pooled["dense@16384"] and pooled["dense@16384"]["gap_default"]["against"] == "qsa@16384"
    assert pooled["qsa@16384"]["sampler"]["mean_abs"] == 0.05 and "gap" not in pooled["qsa@16384"]
    out2 = tmp_path / "b"
    out2.mkdir()
    report = asyncio.run(rl_loop.chunk_check(cfg, tmp_path, rollouts, out2, settings=settings, resume=out1 / "chunk.json"))
    assert calls[3:] == ["t2-0"] and report["status"] == "complete"  # t2-0 had a failed setting: redone
    assert [w["name"] for w in report["windows"]] == ["t1-0", "t1-1", "t2-0"]
    assert report["pooled"]["dense@16384"]["windows"] == 3
    out3 = tmp_path / "c"
    out3.mkdir()
    report = asyncio.run(rl_loop.chunk_check(cfg, tmp_path, rollouts, out3, settings=settings, deadline=1.0))
    assert report["status"] == "partial" and report["windows"] == [] and len(calls) == 4


def test_snr_calls_doses_and_fit():
    pos = [{"task": "t:a", "k": k, "w": 0.5, "rollout": f"/r/a/{k}"} for k in (3, 1, 7, 5, 2)] + [
        {"task": "t:b", "k": 4, "w": 1.0, "rollout": "/r/b/4"}]
    calls = rl_loop.snr_calls(pos)
    assert [(c["name"], c["episodes"]) for c in calls] == [("t_a-h0", 3), ("t_a-h1", 2), ("t_b-h1", 1)]
    assert [a["name"] for a in calls[0]["attempts"]] == ["r00:t:a:1", "r00:t:a:3", "r00:t:a:7"]
    assert calls[1]["attempts"][0] == {"rollout": "/r/a/2", "w": 0.5, "name": "r00:t:a:2", "whole": True, "share": "t:a"}
    assert rl_loop.next_dose(2.0, 0.005, 0.02) == 4.0  # sqrt(4) = 2: KL ~ D^2
    assert rl_loop.next_dose(2.0, 0.0001, 0.02) == 8.0  # capped at 4x
    assert rl_loop.next_dose(2.0, 0.019, 0.02) == 4.0 and rl_loop.next_dose(2.0, 0.05, 0.02) == round(2.0 * 0.4 ** 0.5, 4)
    assert rl_loop.next_dose(2.0, 0.0, 0.02) == 8.0
    # dNLL = p sqrt(KL) + q KL recovered from two points
    pts = [{"kl": k, "d_nll_held": -0.1 * k ** 0.5 + 0.7 * k} for k in (0.01, 0.04)]
    fit = rl_loop.snr_fit(pts, "held", 0.02)
    assert abs(fit["p"] + 0.1) < 1e-4 and abs(fit["q"] - 0.7) < 1e-3
    assert rl_loop.snr_fit(pts[:1], "held", 0.02) is None
    base = {"u1": {"nll": 0.3}, "h1": {"nll": 0.4}, "l1": {"nll": 0.5}}
    row = rl_loop.snr_point(base, {"u1": {"kl_base": 0.02, "nll": 0.31}, "h1": {"kl_base": 0.01, "nll": 0.39},
                                   "l1": {"kl_base": 0.03, "nll": 0.52}}, {"untrained": {"u1"}, "held": {"h1"}, "lost": {"l1"}})
    assert row["kl"] == 0.02 and row["d_nll_held"] == -0.01 and row["per_kl_lost"] == 1.0 and row["kl_held"] == 0.01


def _torch_python():
    import shutil
    import subprocess

    for cand in (os.environ.get("ARC2_SNR_PYTHON"), "~/miniforge3/bin/python3", shutil.which("python3")):
        if cand and Path(cand).exists() and subprocess.run([cand, "-c", "import torch, safetensors"],
                                                           capture_output=True).returncode == 0:
            return cand
    return None


class _FileTrainer:  # arena.rl_loop.Trainer's file protocol, fast polling
    def __init__(self, folder, command, env, log):
        import subprocess

        self.folder = Path(folder)
        (self.folder / "cmd").mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.process = subprocess.Popen(command, env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT)

    async def ready(self, timeout):
        while not (self.folder / "alive.json").exists():
            await asyncio.sleep(0.05)
        return {}

    async def must(self, op, timeout=3600, **payload):
        index, self.n = self.n, self.n + 1
        tmp = self.folder / "cmd" / f"{index:04d}.json.tmp"
        tmp.write_text(json.dumps({"op": op, **payload}))
        tmp.rename(self.folder / "cmd" / f"{index:04d}.json")
        done = self.folder / "done" / f"{index:04d}.json"
        while not done.exists():
            assert self.process.poll() is None
            await asyncio.sleep(0.05)
        out = json.loads(done.read_text())
        if not out.get("ok", True):
            raise RuntimeError(f"trainer {op}: {out.get('error')}")
        return out

    def close(self):
        self.process.terminate()


def test_snr_check_tensor_standin_resume_and_gate(tmp_path, monkeypatch):
    """The snr driver end to end on the preflight stand-in in tensor mode (rl1_trainer's checkpoint format, real AdamW
    steps): every call checked (restored start, one step), the analysis, (b) points with the KL-aimed second dose; a
    resumed run keeps byte-identical calls and points; a reload that does not restore fails the gate before (b)."""
    import subprocess

    python = _torch_python()
    if not python:
        pytest.skip("no python with torch + safetensors")

    monkeypatch.setitem(sys.modules, "arena.rl_loop", types.SimpleNamespace(
        Trainer=_FileTrainer, trainer_command=lambda c, s, w: (["x"], dict(os.environ))))
    monkeypatch.setenv("ARC2_PREFLIGHT", "1")
    monkeypatch.setenv("ARC2_PREFLIGHT_TENSORS", python)
    monkeypatch.setenv("ARC2_SNR_PYTHON", python)

    def rollout(*parts):
        path = tmp_path.joinpath("r", *parts, "rollout.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        return str(path)

    plan = {"positives": [{"task": t, "k": k, "w": w, "rollout": rollout(t.replace(":", "_"), str(k))}
                          for t, ks, w in (("c240:aa", (1, 2, 3, 4), 0.25), ("c240:bb", (0, 5, 9), 0.5), ("c240:cc", (2,), 1.0))
                          for k in ks],
            "probes": {"untrained": [{"name": f"untrained-{i}", "rollout": rollout("u", str(i))} for i in range(2)],
                       "held": [{"name": f"held-{t}", "rollout": rollout("h", t)} for t in ("aa", "bb")],
                       "lost": [{"name": "lost-aa", "rollout": rollout("l", "aa")}]},
            "lr": 1e-4, "clip": 1.0, "doses": {"first": 2.0, "target_kl": 0.02}, "scopes": ["all", "random"], "seed": 0}
    source = Path(rl_loop.__file__).resolve().parents[2]
    cfg = {**rl_loop.DEFAULTS, "trainer_timeout": 60, "trainer_python": python}
    out1 = tmp_path / "a"
    out1.mkdir()
    rep = asyncio.run(rl_loop.snr_check(cfg, source, plan, out1, work=tmp_path / "w1"))
    assert rep["status"] == "complete", (rep.get("error"), rep.get("traceback"), rep.get("gate"))
    names = ["c240_aa-h0", "c240_aa-h1", "c240_bb-h0", "c240_bb-h1", "c240_cc-h0", "c240_bb-h0~repeat"]
    assert list(rep["calls"]) == names and all(c["ok"] for c in rep["calls"].values())
    assert all(c["extract"]["step_rel_err"] < 1e-5 and c["loaded_updates"] == 0 for c in rep["calls"].values())
    assert rep["gate"]["ok"] and rep["gate"]["repeat_cos"] > 0.9999
    a = rep["summary"]["a"]
    assert a["episodes"] == [3, 4] and 0.0 < a["halves_cos"] < 1.0 and a["within_task_cos"] is not None
    assert json.loads((out1 / "analysis.json").read_text())["unpaired"] == ["c240:cc"]
    assert rep["zero_path"]["kl"] < 1e-6  # B = 0 vs the 1e-12 nudge: no distance in the stand-in
    pts = rep["summary"]["b"]
    # KL 0.003 D^2: 0.012 at D 2 is within 30% of the 0.02 aim, so the second point is at twice the distance
    assert [p["D"] for p in pts["all"]["points"]] == [2.0, 4.0] and set(pts) == {"all", "random"}
    assert abs(pts["all"]["points"][1]["kl"] - 0.048) < 5e-4
    assert pts["all"]["held"]["p"] < 0 < pts["all"]["untrained"]["q"]
    assert (out1 / "analysis.json").is_file() and len(list((out1 / "grads").glob("*.safetensors"))) == 6
    # resume: everything kept
    out2 = tmp_path / "b"
    out2.mkdir()
    rep2 = asyncio.run(rl_loop.snr_check(cfg, source, plan, out2, resume=out1, work=tmp_path / "w2"))
    ops = [s["op"] for s in rep2["steps"]]
    assert rep2["status"] == "complete" and rep2["resume_reuse"] and "call" not in ops and "point" not in ops
    assert ops.count("call_kept") == 6 and ops.count("point_kept") == 4
    assert rep2["summary"]["a"]["halves_cos"] == a["halves_cos"]
    # partial resume: the first two calls and the first point kept, the rest redone, same numbers
    old = json.loads((out1 / "snr.json").read_text())
    old["calls"] = {k: old["calls"][k] for k in names[:2]}
    old["points"] = {"all@2": old["points"]["all@2"]}
    (out1 / "snr.json").write_text(json.dumps(old))
    out3 = tmp_path / "c"
    out3.mkdir()
    rep3 = asyncio.run(rl_loop.snr_check(cfg, source, plan, out3, resume=out1, work=tmp_path / "w3"))
    ops = [s["op"] for s in rep3["steps"]]
    assert rep3["status"] == "complete" and ops.count("call_kept") == 2 and ops.count("call") == 4
    assert ops.count("point_kept") == 1 and ops.count("point") == 3
    assert rep3["summary"]["a"]["halves_cos"] == a["halves_cos"]
    # a reload that keeps the previous state: caught at the second call, no (b)
    monkeypatch.setenv("ARC2_PREFLIGHT_BREAK", "restore")
    out4 = tmp_path / "d"
    out4.mkdir()
    rep4 = asyncio.run(rl_loop.snr_check(cfg, source, plan, out4, work=tmp_path / "w4"))
    assert rep4["status"] == "gate_failed" and "c240_aa-h1" in rep4["gate"]["why"] and not rep4["points"]


def test_grad_bank_tensor_standin_gram_resume_and_gate(tmp_path, monkeypatch):
    """bank mode (arc2 RL plan step 1): one gradient per episode, each call checked, the repeat at cos >= 0.999, Gram
    blocks that equal the gradients' dot products; a resumed run keeps every call; a reload that does not restore
    fails the gate."""
    python = _torch_python()
    if not python:
        pytest.skip("no python with torch + safetensors")
    monkeypatch.setitem(sys.modules, "arena.rl_loop", types.SimpleNamespace(
        Trainer=_FileTrainer, trainer_command=lambda c, s, w: (["x"], dict(os.environ))))
    monkeypatch.setenv("ARC2_PREFLIGHT", "1")
    monkeypatch.setenv("ARC2_PREFLIGHT_TENSORS", python)
    monkeypatch.setenv("ARC2_SNR_PYTHON", python)

    def rollout(*parts):
        path = tmp_path.joinpath("r", *parts, "rollout.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        return str(path)

    eps = [{"name": f"r00:{t}:{k}", "task": t, "k": k, "reward": float(k % 2), "tokens": 1000 + k,
            "rollout": rollout(t.replace(":", "_"), str(k))} for t in ("c240:aa", "c240:bb") for k in range(3)]
    plan = {"mode": "bank", "episodes": eps, "lr": 1e-4, "clip": 1.0}
    source = Path(rl_loop.__file__).resolve().parents[2]
    cfg = {**rl_loop.DEFAULTS, "trainer_timeout": 60, "trainer_python": python}
    out1 = tmp_path / "a"
    out1.mkdir()
    rep = asyncio.run(rl_loop.snr_check(cfg, source, plan, out1, work=tmp_path / "w1"))
    assert rep["status"] == "complete", (rep.get("error"), rep.get("traceback"), rep.get("gate"))
    assert list(rep["calls"]) == [e["name"] for e in eps] + ["r00:c240:aa:0~repeat"]
    assert all(c["ok"] and c["extract"]["step_rel_err"] < 1e-5 and c["loaded_updates"] == 0 for c in rep["calls"].values())
    assert rep["gate"]["ok"] and rep["gate"]["repeat_cos"] > 0.9999
    gram = json.loads((out1 / "gram.json").read_text())
    assert gram["n"] == 6 and gram["episodes"] == [e["name"] for e in eps] and gram["blocks"]
    total = [[sum(b[i][j] for b in gram["blocks"].values()) for j in range(6)] for i in range(6)]
    import subprocess

    check = subprocess.run([python, "-c", (
        "import json,sys; from safetensors.torch import load_file; import torch\n"
        "fs=json.loads(sys.argv[1]); g=[load_file(f)['g'].double() for f in fs]\n"
        "print(json.dumps([[float(a.dot(b)) for b in g] for a in g]))"),
        json.dumps([str(out1 / "grads" / f"{e['name'].replace(':', '_')}.safetensors") for e in eps])],
        capture_output=True, text=True, check=True)
    direct = json.loads(check.stdout)
    assert all(abs(total[i][j] - direct[i][j]) <= 1e-9 * max(1.0, abs(direct[i][j])) for i in range(6) for j in range(6))
    out2 = tmp_path / "b"
    out2.mkdir()
    rep2 = asyncio.run(rl_loop.snr_check(cfg, source, plan, out2, resume=out1, work=tmp_path / "w1"))
    assert rep2["status"] == "complete" and all(c.get("resumed") for c in rep2["calls"].values())
    assert json.loads((out2 / "gram.json").read_text())["blocks"] == gram["blocks"]
    monkeypatch.setenv("ARC2_PREFLIGHT_BREAK", "restore")
    out3 = tmp_path / "c"
    out3.mkdir()
    rep3 = asyncio.run(rl_loop.snr_check(cfg, source, plan, out3, work=tmp_path / "w3"))
    assert rep3["status"] == "gate_failed" and "restore" in rep3["gate"]["why"]


def test_bank_dose_tensor_standin_updates_points_resume_and_gate(tmp_path, monkeypatch):
    """dose mode (arc2 RL plan step 3): from one start, one signed trainer update per planned weight set, each checked
    (restored start, one AdamW step, step identity); directions combined from those gradients (scope, random control);
    every planned point scored against the base; a resumed run keeps updates and points; a reload that does not restore
    fails the gate before any point."""
    python = _torch_python()
    if not python:
        pytest.skip("no python with torch + safetensors")
    monkeypatch.setitem(sys.modules, "arena.rl_loop", types.SimpleNamespace(
        Trainer=_FileTrainer, trainer_command=lambda c, s, w: (["x"], dict(os.environ))))
    monkeypatch.setenv("ARC2_PREFLIGHT", "1")
    monkeypatch.setenv("ARC2_PREFLIGHT_TENSORS", python)
    monkeypatch.setenv("ARC2_SNR_PYTHON", python)

    def rollout(*parts):
        path = tmp_path.joinpath("r", *parts, "rollout.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        return str(path)

    eps = [{"name": f"r00:{t}:{k}", "task": t, "k": k, "reward": float(k % 2), "tokens": 1000 + k,
            "rollout": rollout(t.replace(":", "_"), str(k))} for t in ("c240:aa", "c240:bb") for k in range(4)]
    source = Path(rl_loop.__file__).resolve().parents[2]
    cfg = {**rl_loop.DEFAULTS, "trainer_timeout": 60, "trainer_python": python}
    batch = [e for e in eps if e["k"] < 3]
    plan = {"mode": "dose", "episodes": batch, "random_of": "grpo",
            "updates": {"grpo": {"weights": {e["name"]: (0.6 if e["reward"] else -0.4) for e in batch}},
                        "pos": {"weights": {e["name"]: 1.0 for e in batch if e["reward"]}}},
            "directions": {"grpo": {"update": "grpo"}, "grpo_gdn": {"update": "grpo", "scope": "gdn"}, "pos": {"update": "pos"}},
            "points": {"grpo": [1.0, 2.0], "grpo_gdn": [2.0], "random": [2.0], "pos": [2.0], "sgd:grpo": [0.5]},
            "probes": {"held": [{"name": "held-aa", "rollout": eps[3]["rollout"]}],
                       "lost": [{"name": "lost-bb", "rollout": eps[7]["rollout"]}],
                       "untrained": [{"name": f"untrained-{i}", "rollout": rollout("u", str(i))} for i in range(2)]}}
    out1 = tmp_path / "d1"
    rep1 = asyncio.run(rl_loop.snr_check(cfg, source, plan, out1, work=tmp_path / "w1"))
    assert rep1["status"] == "complete", (rep1.get("error"), rep1.get("traceback"), rep1.get("gate"))
    assert set(rep1["updates"]) == {"grpo", "pos"} and all(u["ok"] and u["extract"]["step_rel_err"] < 1e-5
                                                           for u in rep1["updates"].values())
    assert rep1["updates"]["grpo"]["episodes"] == len(batch)
    assert list(rep1["points"]) == ["grpo@1", "grpo@2", "grpo_gdn@2", "random@2", "pos@2", "sgd:grpo@0.5"]
    d = rep1["directions"]
    assert d["grpo_gdn"]["elements"] < d["grpo"]["elements"] and d["random"]["of"] == "grpo" and abs(d["cos"]["grpo|pos"]) < 1
    p1, p2 = rep1["points"]["grpo@1"], rep1["points"]["grpo@2"]
    assert 0 < p1["kl"] < p2["kl"] and p2["lr_equiv"] == pytest.approx(2 * p1["lr_equiv"])
    assert all(r.get("d_nll_held") is not None and r.get("d_nll_untrained") is not None for r in rep1["points"].values())
    rep2 = asyncio.run(rl_loop.snr_check(cfg, source, plan, tmp_path / "d2", resume=out1, work=tmp_path / "w2"))
    assert rep2["status"] == "complete" and all(u.get("resumed") for u in rep2["updates"].values())
    assert all(r.get("resumed") for r in rep2["points"].values())
    monkeypatch.setenv("ARC2_PREFLIGHT_BREAK", "restore")
    rep3 = asyncio.run(rl_loop.snr_check(cfg, source, plan, tmp_path / "d3", work=tmp_path / "w3"))
    assert rep3["status"] == "gate_failed" and "restore" in rep3["gate"]["why"] and not rep3["points"]


def test_grad_bank_answer_credit_gate_first_and_skip(tmp_path, monkeypatch):
    """bank mode with answer credit (arc2 RL plan step 3'): the real-trainer gate runs first (the smallest try, its
    repeat at cos >= 0.999, the same try with full credit, which must differ); every banked call carries the credited
    count; a try with no answer tokens is skipped, not banked, and does not fail the run."""
    python = _torch_python()
    if not python:
        pytest.skip("no python with torch + safetensors")
    monkeypatch.setitem(sys.modules, "arena.rl_loop", types.SimpleNamespace(
        Trainer=_FileTrainer, trainer_command=lambda c, s, w: (["x"], dict(os.environ))))
    monkeypatch.setenv("ARC2_PREFLIGHT", "1")
    monkeypatch.setenv("ARC2_PREFLIGHT_TENSORS", python)
    monkeypatch.setenv("ARC2_SNR_PYTHON", python)

    def rollout(*parts):
        path = tmp_path.joinpath("r", *parts, "rollout.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        return str(path)

    eps = [{"name": f"r00:{t}:{k}", "task": t, "k": k, "reward": float(k % 2), "tokens": 1000 + 10 * k + (t == "c240:bb"),
            "rollout": rollout(t.replace(":", "_"), str(k))} for t in ("c240:aa", "c240:bb") for k in range(3)]
    source = Path(rl_loop.__file__).resolve().parents[2]
    cfg = {**rl_loop.DEFAULTS, "trainer_timeout": 60, "trainer_python": python}
    plan = {"mode": "bank", "episodes": eps, "lr": 1e-4, "clip": 1.0, "credit": "answer", "gate_first": True}
    rep = asyncio.run(rl_loop.snr_check(cfg, source, plan, tmp_path / "a", work=tmp_path / "w1"))
    assert rep["status"] == "complete", (rep.get("error"), rep.get("traceback"), rep.get("gate"))
    first = "r00:c240:aa:0"
    assert list(rep["calls"])[:3] == [first, first + "~repeat", first + "~full"]
    g = rep["gate_first"]
    assert g["repeat_cos"] > 0.9999 and g["cos_answer_full"] < 0.99 and g["credited_share"] == pytest.approx(0.2)
    banked = [n for n in rep["calls"] if "~" not in n]
    assert all(rep["calls"][n]["credited"] == 10 and rep["calls"][n]["ok"] for n in banked)
    assert rep["calls"][first + "~full"]["credited"] is None and rep["gram"]["n"] == len(eps)
    monkeypatch.setenv("ARC2_PREFLIGHT_NO_ANSWER", "c240:bb:2")
    rep2 = asyncio.run(rl_loop.snr_check(cfg, source, plan, tmp_path / "b", work=tmp_path / "w2"))
    assert rep2["status"] == "complete", (rep2.get("error"), rep2.get("gate"))
    assert rep2["calls"]["r00:c240:bb:2"]["skip"] == "no answer tokens" and rep2["gram"]["n"] == len(eps) - 1
