import asyncio
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/l4/l4_dp2_src.py"
spec = importlib.util.spec_from_file_location("l4_dp2_src", SOURCE)
dp2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dp2)

PINNED = "x\n    def use_sequence_parallel_moe(self) -> bool:\n        return (\n            self.a\n        )\n"


def test_patched_turns_sequence_parallel_off_once():
    text = dp2.patched(PINNED)
    assert "return False and (" in text and dp2.patched(text) == text
    with pytest.raises(RuntimeError):
        dp2.patched("nothing here")


def test_apply_edits_and_records(tmp_path):
    path = tmp_path / dp2.RELATIVE
    path.parent.mkdir(parents=True)
    path.write_text(PINNED)
    record = dp2.apply(tmp_path, tmp_path)
    assert record["changed"] and dp2.MARK in path.read_text() and (tmp_path / "dp2-patch.json").is_file()
    assert not dp2.apply(tmp_path, tmp_path)["changed"]


def test_replica_kv_reads_engines_in_order():
    text = ('vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.5\n'
            'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25\nvllm:other{engine="0"} 9\n')
    assert dp2.replica_kv(text) == [0.25, 0.5]


class Gate:
    budget, log, peak = None, None, 0

    def __init__(self, slots):
        self.capacity, self.slots, self.reserved = 100, slots, 3

    def snapshot(self):
        return dict.fromkeys(("reserved", "inflight", "waiting", "peak_reserved", "peak_inflight", "peak_waiting",
                              "admitted"), 1) | {"capacity": self.capacity, "slots": self.slots, "units": "blocks"}


class Part:
    def __init__(self, gate):
        self.gate, self.flying, self.server_errors, self.calls, self.drops, self.deadline = gate, 2, [1], [], [], 0

    async def count(self, messages, settings):
        return 7

    async def complete(self, task_id, *args):
        self.calls.append(task_id)
        return self.route

    def drop(self, task_id):
        self.drops.append(task_id)


def test_tasks_stay_on_an_engine_and_new_ones_balance():
    transport, gate = dp2.split(Part, Gate, 45)
    assert [part.gate.slots for part in transport.parts] == [23, 22]
    done = asyncio.run(transport.complete("a#0", [], 1, 1, {}))
    assert done == {"X-data-parallel-rank": "0"}
    assert asyncio.run(transport.complete("b#0", [], 1, 1, {}))["X-data-parallel-rank"] == "1"
    assert asyncio.run(transport.complete("a#1", [], 1, 1, {}))["X-data-parallel-rank"] == "0"  # a reopen of a
    transport.drop("a#0")
    transport.drop("a#1")
    assert transport.parts[0].drops == ["a#0", "a#1"] and transport.open == [0, 1]
    assert asyncio.run(transport.complete("c#0", [], 1, 1, {}))["X-data-parallel-rank"] == "0"
    transport.drop("never sent")
    assert transport.open == [1, 1]
    assert asyncio.run(transport.count([], {})) == 7
    assert transport.flying == 4 and transport.server_errors == [1, 1]
    transport.deadline = 9
    assert [part.deadline for part in transport.parts] == [9, 9]
    row = gate.snapshot()
    assert row["capacity"] == 200 and row["slots"] == 45 and row["units"] == "blocks" and len(row["replicas"]) == 2
    assert gate.reserved == 6 and gate.capacity == 200
