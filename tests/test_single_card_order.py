"""Single card (server and trainer take turns): no trainer GPU op while the server is awake. ovf3r2 (23 Oct) OOMed on
the knob gate's rescore at the round-6 top of a resumed read-only round: the resume woke the server, then a changed
trainer_fast knob rescored the probes on the same card."""
import asyncio
import json

import pytest

from test_rl_startup import Trainer, make_loop

GPU_OPS = {"score", "update", "load_checkpoint", "save", "checkpoint", "park_check", "accept_reference"}


class CardServer:
    model = "base"

    def __init__(self, log):
        self.log, self.asleep = log, False

    async def sleep(self):
        self.asleep = True
        self.log.append(("server.sleep",))
        return 0.0

    async def wake(self):
        self.asleep = False
        self.log.append(("server.wake",))
        return 0.0

    async def load(self, name, path):
        assert not self.asleep, "adapter load on a sleeping server"
        self.log.append(("server.load", str(path)))
        return 0.1


def card_loop(tmp_path, monkeypatch, log, mg, knobs=None):
    loop = make_loop(tmp_path, monkeypatch, log, mg={"single_card": True, **mg})
    server = loop.server = CardServer(log)
    if knobs is not None:
        path = tmp_path / "knobs.json"
        path.write_text(json.dumps(knobs))
        monkeypatch.setenv("ARENA_KNOBS", str(path))
    real_start = loop.start_trainer

    async def start_trainer(gate=None):
        out = await real_start(gate)
        inner = loop.trainer

        class Checked(Trainer):
            def __init__(self):
                self.__dict__.update(inner.__dict__)

            async def call(self, op, timeout=3600, **payload):
                if op in GPU_OPS:
                    assert server.asleep, f"trainer {op} while the server holds the card"
                if op == "score":
                    await asyncio.sleep(0)
                    self.ops.append((op, payload))
                    self.log.append(("trainer", op))
                    return {"ok": True, "windows": {"p1": {"kl_base": 0.02, "kl_previous": 0.01}}}
                return await super().call(op, timeout=timeout, **payload)

        loop.trainer = Checked()
        return out

    loop.start_trainer = start_trainer
    return loop


FAST = {"prefetch": 2, "union_attention": True}


@pytest.mark.parametrize("knobs", [None, {"trainer_fast": FAST}, {"trainer_fast": FAST, "active_tries": 100}])
def test_resumed_read_only_last_round_keeps_the_card_order(tmp_path, monkeypatch, knobs):
    log = []
    loop = card_loop(tmp_path, monkeypatch, log, {"max_rounds": 6, "trainer_fast": {"prefetch": 2}}, knobs)
    assert asyncio.run(loop.mg_loop()) == "max_rounds"
    kinds = loop.feed.kinds()
    assert "error" not in kinds and "round" in kinds
    if knobs:
        assert [d["ok"] for k, d in loop.feed.events if k == "knob_gate"] == [True]
        assert loop.cfg["mg"]["trainer_fast"] == FAST
    plays = [i for i, e in enumerate(log) if e[0] == "play_start"]
    wakes = [i for i, e in enumerate(log) if e[0] == "server.wake"]
    sleeps = [i for i, e in enumerate(log) if e[0] == "server.sleep"]
    for p in plays:  # every play on an awake server holding the served adapter
        assert max(w for w in wakes if w < p) > max([s for s in sleeps if s < p] or [-1])


def test_knob_gate_mid_run_hands_the_card_back(tmp_path, monkeypatch):
    log = []
    loop = card_loop(tmp_path, monkeypatch, log, {"max_rounds": 8, "trainer_fast": {"prefetch": 2}}, {"trainer_fast": FAST})
    assert asyncio.run(loop.mg_loop()) == "max_rounds"
    assert [d["ok"] for k, d in loop.feed.events if k == "knob_gate"] == [True]
    assert loop.server.asleep is False or log[-1][0] != "play_start"
