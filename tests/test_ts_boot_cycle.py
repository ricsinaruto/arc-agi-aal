"""mg.timeshare.boot_cycle: a lazy-booted second server takes its first (slow) level-1 sleep right after boot."""
import asyncio
import types

import pytest

pytest.importorskip("httpx")
from arena import level_bench, rl_loop, rl_single  # noqa: E402


class FakeServer2:
    def __init__(self, server_dir, model, **kw):
        self.calls, self.memory, self.root, self.folder = [], {}, "http://127.0.0.1:1235", server_dir

    async def boot(self):
        self.calls.append("boot")
        return {"boot_seconds": 1.0}

    async def sleep(self):
        self.calls.append("sleep")
        return 29.8

    async def wake(self):
        self.calls.append("wake")
        return 1.3


def loop_for(tmp_path, timeshare):
    events = []
    loop = types.SimpleNamespace(cfg={"trainer_device": "1", "mg": {"timeshare": timeshare, "active_tries": 8}},
                                 server_dir=tmp_path, server=types.SimpleNamespace(model="base"), study={},
                                 feed=types.SimpleNamespace(emit=lambda kind, **kw: events.append((kind, kw))))
    return loop, events


@pytest.mark.parametrize("cycle,sleep,want", [(True, False, ["boot", "sleep", "wake"]), (False, False, ["boot"]),
                                               (True, True, ["boot", "sleep"])])
def test_boot_cycle_only_on_a_lazy_boot_when_asked(tmp_path, monkeypatch, cycle, sleep, want):
    monkeypatch.setattr(rl_loop, "SecondServer", FakeServer2)
    monkeypatch.setattr(level_bench, "SHARED_SLOTS", None)
    loop, events = loop_for(tmp_path, {"boot_cycle": cycle} if cycle else {})
    boot = asyncio.run(rl_single.MultiMixin.ts_boot(loop, sleep=sleep))
    assert loop.server2.calls == want
    assert ("boot_cycle" in boot) == (cycle and not sleep)
    if cycle and not sleep:
        assert boot["boot_cycle"] == {"sleep_s": 29.8, "wake_s": 1.3} and events[-1][1]["boot_cycle"] == boot["boot_cycle"]
