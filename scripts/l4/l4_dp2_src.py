"""DP 2 + TP 2 + wide EP on four L4s (l4kv): two engines of two cards each, every card one EP rank.

Two parts of ``dp2`` in the options:
- ``apply``: the runtime's ``use_sequence_parallel_moe`` stays False, so Flash's HC check passes and the MoE takes
  the stock AllGather + ReduceScatter over the 2-rank DP group, then the TP all-reduce (``moe_runner``'s late path).
  A card holds 128 of 512 experts and sees both engines' rows; only two-rank collectives run.
- ``split``: one ``Transport`` and one ``KVGate`` per engine, a task stays on its engine (a prefix lives in one pool),
  ``X-data-parallel-rank`` routes it. New tasks go to the engine with the fewer open attempts.

Standard library only at import. ``apply`` edits the extracted runtime (disposable), once, by its exact anchor.
"""

import hashlib
import json
import re
from pathlib import Path

RELATIVE = "vllm/config/parallel.py"
ANCHOR = "    def use_sequence_parallel_moe(self) -> bool:\n        return (\n"
MARK = "ARC2 L4 dp2"
REPLACEMENT = ("    def use_sequence_parallel_moe(self) -> bool:\n"
               "        # ARC2 L4 dp2: Flash refuses sequence-parallel MoE; AG + RS over the DP group, then the TP all-reduce\n"
               "        return False and (\n")
KV_LINE = re.compile(r'^vllm:kv_cache_usage_perc\{[^}]*engine="(\d+)"[^}]*\}\s+([-+\d.eE]+)$', re.M)


def patched(text):
    """The file's text with sequence-parallel MoE off (unchanged if it is already off)."""
    if MARK in text:
        return text
    if text.count(ANCHOR) != 1:
        raise RuntimeError("dp2: use_sequence_parallel_moe is not where it was; re-read scripts/l4/l4_dp2_src.py")
    return text.replace(ANCHOR, REPLACEMENT)


def apply(site_packages, working):
    """Edit the runtime's ParallelConfig and leave a record in ``working``."""
    path = Path(site_packages) / RELATIVE
    before = path.read_text()
    after = patched(before)
    if after != before:
        path.write_text(after)
    record = {"relative": RELATIVE, "changed": after != before,
              "original_sha256": hashlib.sha256(before.encode()).hexdigest(),
              "patched_sha256": hashlib.sha256(after.encode()).hexdigest()}
    (Path(working) / "dp2-patch.json").write_text(json.dumps(record, indent=1))
    return record


def replica_kv(text):
    """KV use per engine from /metrics, in engine order."""
    rows = {int(engine): float(value) for engine, value in KV_LINE.findall(text)}
    return [rows[key] for key in sorted(rows)]


class ReplicaGate:
    """The gates' sum, for the reports and the live line."""

    def __init__(self, parts):
        self.parts = parts
        self.budget, self.log = parts[0].budget, parts[0].log

    def snapshot(self):
        rows = [part.snapshot() for part in self.parts]
        out = dict(rows[0])
        for key in ("capacity", "slots", "reserved", "inflight", "waiting", "peak_reserved", "peak_inflight",
                    "peak_waiting", "admitted"):
            out[key] = sum(row[key] for row in rows)
        out["replicas"] = [{key: row[key] for key in ("reserved", "inflight", "waiting", "peak_reserved", "admitted")}
                           for row in rows]
        return out

    @property
    def capacity(self):
        return sum(part.capacity for part in self.parts)

    @property
    def reserved(self):
        return sum(part.reserved for part in self.parts)

    def __getattr__(self, name):  # the first gate's fixed fields; counters go through snapshot
        return getattr(self.parts[0], name)


class ReplicaTransport:
    """``Transport``'s surface over one transport per engine; a task's attempts stay on one engine."""

    OWN = ("parts", "owner", "open", "attempts")

    def __init__(self, parts):
        self.parts, self.owner, self.open, self.attempts = parts, {}, [0] * len(parts), {}

    def __setattr__(self, name, value):
        if name in self.OWN:
            object.__setattr__(self, name, value)
        else:  # stopped, deadline: every engine's
            for part in self.parts:
                setattr(part, name, value)

    def __getattr__(self, name):
        return getattr(self.parts[0], name)

    @property
    def flying(self):
        return sum(part.flying for part in self.parts)

    @property
    def server_errors(self):
        return [row for part in self.parts for row in part.server_errors]

    def pick(self, task_id):
        """The engine of an attempt id (``task#k`` or ``task``): the task's, else the one with fewer open attempts."""
        task = str(task_id).split("#", 1)[0]
        index = self.owner.get(task)
        if index is None:
            index = self.owner[task] = min(range(len(self.parts)), key=lambda k: (self.open[k], k))
        if task_id not in self.attempts:
            self.attempts[task_id] = index
            self.open[index] += 1
        return index

    async def count(self, messages, settings):
        return await self.parts[0].count(messages, settings)

    async def complete(self, task_id, messages, count, max_tokens, settings):
        return await self.parts[self.pick(task_id)].complete(task_id, messages, count, max_tokens, settings)

    def drop(self, task_id):
        index = self.attempts.pop(task_id, None)
        if index is not None:
            self.open[index] -= 1
            self.parts[index].drop(task_id)


def split(make_transport, make_gate, slots, replicas=2):
    """(transport, gate) over ``replicas`` engines; ``slots`` is the whole run's, shared out evenly."""
    gates = [make_gate((slots + replicas - 1 - k) // replicas) for k in range(replicas)]
    parts = [make_transport(gate) for gate in gates]
    for k, part in enumerate(parts):
        part.route = {"X-data-parallel-rank": str(k)}
    return ReplicaTransport(parts), ReplicaGate(gates)
