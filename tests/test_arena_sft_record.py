"""arena.sft_record: per-request SFT records are complete, rebuild exactly (messages and prompt ids), rotate parts,
end with an outcome, and a relaunch (attempt 2) writes beside attempt 1 without touching it."""

import asyncio
import json

import httpx
import numpy as np
from conftest import FakeEnv, duck_agent_for, observation

from arena import sft_record
from arena.clock import ActiveClock, GlobalClock
from arena.duck import ArenaTransferDuckAgent
from arena.gate import GateSlot, KVGate
from arena.scheduler import GameRecord
from arena.sft_record import SftRecorder, rebuild

WINDOW, CAP = 131072, 16384


def reply(n, prompt_ids):
    code = "action([{'action':'UP'}])"
    return httpx.Response(200, request=httpx.Request("POST", "http://local/v1/chat/completions"), json={
        "id": f"r{n}", "prompt_token_ids": prompt_ids,
        "choices": [{"finish_reason": "tool_calls", "token_ids": [7, 8, 9, n],
                     "message": {"role": "assistant", "content": f"step {n}", "reasoning_content": f"think {n}",
                                 "tool_calls": [{"id": f"c{n}", "type": "function", "function": {
                                     "name": "python", "arguments": json.dumps({"code": code})}}]}}],
        "usage": {"completion_tokens": 40, "prompt_tokens": len(prompt_ids)}})


def play(tmp_path, attempt, actions=4):
    sent = []

    class Transport:
        async def post(self, url, **kwargs):
            payload = kwargs["json"]
            sent.append(json.loads(json.dumps(payload)))  # the list grows after the reply
            # A growing prompt: each request extends the previous one (append-only context).
            ids = list(range(1000 + 50 * len(sent)))
            return reply(len(sent), ids)

    agent, _ = duck_agent_for(tmp_path, Transport(), agent_class=ArenaTransferDuckAgent)
    boards = [np.zeros((64, 64), dtype=np.uint8) for _ in range(actions + 1)]
    for k, b in enumerate(boards):
        b[k, k] = 3
    obs = [observation(b, levels=0) for b in boards[:-1]] + [observation(boards[-1], levels=1, state="WIN")]
    env = FakeEnv(obs)
    agent.arc.make = lambda *a, **kw: env
    cfg = agent.config
    cfg.max_model_len, cfg.max_output_tokens, cfg.max_actions = WINDOW, CAP, 100
    cfg.reasoning_effort, cfg.temperature = "xhigh", 0.7
    wall = GlobalClock(900)
    game = GameRecord("k007", 7, ActiveClock(wall))
    game.ready = asyncio.Event()
    gate = KVGate(int(291_900 * 0.9), slots=8)
    agent.attach(game, window=WINDOW, cap=CAP)
    agent.client.slots = GateSlot(gate, game, cap=CAP, window=WINDOW)
    study = {"name": "sft-test", "games": [f"g{i}" for i in range(5)], "sft_record": {"attempt": attempt}}
    out = tmp_path.parent.parent  # the trial output (tmp_path = <out>/games/k007)
    (out / "public-roster.json").write_text(json.dumps(
        [{"game_id": f"k{i:03d}", "source_game_id": f"g{i % 5}"} for i in range(10)]))
    agent.sft_recorder = SftRecorder(study, out, "k007", 7)
    wall.release()
    game.clock.release()
    return agent, sent


async def test_records_rebuild_rotate_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(sft_record, "PART_BYTES", 6000)  # several parts
    game_dir = tmp_path / "games" / "k007"
    game_dir.mkdir(parents=True)
    agent, sent = play(game_dir, attempt=1)
    result = await agent.run()
    assert result["error"] is None and sent
    root = tmp_path / "games" / "k007" / "sft"
    parts = sorted(root.glob("turns-a1-*.jsonl"))
    assert len(parts) >= 2
    recs = [json.loads(line) for p in parts for line in p.open()]
    reqs = [r for r in recs if r["kind"] == "req"]
    assert len(reqs) == len(sent) and recs[-1]["kind"] == "end" and recs[-1]["won"]
    assert all(json.loads(p.open().readline())["kind"] == "meta" for p in parts)
    meta = recs[0]
    assert meta["source_game_id"] == "g2" and meta["clone"] == 1 and meta["tools"] and meta["attempt"] == 1
    # every served request rebuilds exactly: messages == the served payload's, prompt ids == the server's
    rebuilt = list(rebuild(parts))
    assert len(rebuilt) == len(sent)
    for (m, rec, messages, ids), payload in zip(rebuilt, sent):
        assert messages == json.loads(json.dumps(payload["messages"]))
        assert list(ids) == list(range(1000 + 50 * rec["i"]))
        assert rec["params"]["return_token_ids"] is True and "messages" not in rec["params"]
        assert rec["response"]["token_ids"][-1] == rec["i"]
        assert rec["response"]["choice"]["message"]["reasoning_content"] == f"think {rec['i']}"
        assert rec["response"]["usage"]["prompt_tokens"] == len(ids)
        assert "board" in rec or "board_ref" in rec
    assert sum(len(r["actions_before"]) for r in reqs) + len(recs[-1]["actions_after"]) == result["actions"]
    assert any(r["messages"]["keep"] > 0 for r in reqs[1:])  # deltas, not full copies
    out = json.loads((root / "outcome.json").read_text())
    assert out["final"] and out["won"] and out["requests"] == len(sent) and out["recorder_errors"] == 0
    # relaunch: attempt 2 writes new parts; attempt 1's are untouched
    before = {p.name: p.read_bytes() for p in parts}
    agent2, sent2 = play(game_dir, attempt=2)
    await agent2.run()
    assert {p.name: p.read_bytes() for p in parts} == before
    parts2 = sorted(root.glob("turns-a2-*.jsonl"))
    assert parts2 and len(list(rebuild(parts2))) == len(sent2)


def test_common_prefix():
    from array import array

    assert sft_record.common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert sft_record.common_prefix(array("I", range(100)), array("I", range(50))) == 50
    assert sft_record.common_prefix([], [1]) == 0
