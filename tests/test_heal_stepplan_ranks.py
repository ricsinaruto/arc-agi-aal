"""MFU-1 HEAL_STREAM_STEPPLAN on 3 gloo ranks (arc_infra.runops.multirank): rank 0 reads the stream folder and plans the
step; every rank must get the same picks per take and issue the same FSDP-gathering calls in dp_sequence, with
asymmetric piece mixes (long/short prefixes, uneven loss sizes) and a rank with nothing to train in a slot."""
import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))
from arc_infra.runops.multirank import run_ranks  # noqa: E402

W, CHUNK = 64, 16
SHAPES = [(300, 200), (90, 10), (260, 180), (120, 30), (200, 150), (80, 2), (310, 240), (100, 40), (150, 90)]


def _rank(rank, world, inp):
    import transformers

    import heal_kd

    for k in [k for k in os.environ if k.startswith("HEAL_")]:  # only this test's knobs (an inherited HEAL_FRESH_HOLDOUT hung it)
        os.environ.pop(k)
    os.environ.update(HEAL_INERT_PREFIX="1", HEAL_SKIP_INERT_LOSS="1", HEAL_CAPTURE_CACHE="1",
                      HEAL_STREAM_STEPPLAN=inp["plan"])
    q = heal_kd.StreamQueue(inp["folder"], rank, coalesce=True)
    q.window, q.chunk = W, CHUNK
    room = sum(t - r0 for t, r0 in SHAPES) / world + 1
    picks = [q.take(world, 5, max_rows=room) for _ in range(3)]
    seq, w = [], torch.ones(1, requires_grad=True)

    class Cache:
        def __init__(self, *a, **k):
            pass

    transformers.DynamicCache = Cache
    heal_kd.commit_captured = lambda runner: Cache()

    class Runner:
        training, mtp = True, None
        config = type("C", (), {"text_config": None})()

        def eval(self):
            self.training = False

        def train(self, was=True):
            self.training = was

        def __call__(self, mode, ids, a, b, cache, **kw):
            seq.append(mode)
            if heal_kd.mtp_inputs(ids, a, b) is not None:
                seq.append("mtp")
            if mode == "extend":
                return cache
            loss = (w * float(b - a)).sum()
            loss.register_hook(lambda g: seq.append("bwd"))
            return loss, 0.0, 0.0, 0.0, 0.0, 0.0

    def data(total):
        top = (torch.zeros(total, 2), torch.zeros(total, 2, dtype=torch.int32), torch.zeros(total))
        return {"ids": torch.zeros(1, total, dtype=torch.long), "teacher": top, "reply": torch.zeros(total),
                "mask": torch.zeros(total)}

    dummy = {**data(16), "piece": None}
    for k, units in enumerate(picks):
        mine = None
        if not (k == 2 and rank == 2):  # slot 3: rank 2 has nothing to train (padding only)
            first, last = q.meta[units[rank][0]], q.meta[units[rank][-1]]
            total, row0 = last["tokens"], first["tokens"] - first["rows"]
            d = data(total)
            mine = {"data": d, "mask": d["mask"], "piece": None, "prefix": max(row0 - 1, 0) // 64 * 64,
                    "general": False}
        heal_kd.dp_sequence(Runner(), mine, dummy, window=W, chunk=CHUNK, scale=1.0, mtp_weight=0.0)
    return {"picks": picks, "seq": seq, "plan_left": len(q.plan)}


@pytest.mark.multirank
@pytest.mark.parametrize("plan", ["0", "1"])
def test_step_plan_same_picks_and_calls_on_every_rank(tmp_path, plan):
    for i, (tokens, row0) in enumerate(SHAPES):
        (tmp_path / f"s-c{i:03d}-p0000.pt").write_bytes(b"")
        (tmp_path / f"s-c{i:03d}-p0000.ready").write_text(json.dumps(
            {"version": 5, "tokens": tokens, "rows": tokens - row0, "reply_rows": tokens - row0}))
    out = run_ranks(_rank, inputs=[{"folder": str(tmp_path), "plan": plan}] * 3, timeout=300)  # CI load: 75 s hang on 6f1211eff, 30 s solo
    assert out[0]["picks"] == out[1]["picks"] == out[2]["picks"]
    assert out[0]["seq"] == out[1]["seq"] == out[2]["seq"]
    used = sorted(p for slot in out[0]["picks"] for u in slot for p in u)
    assert len(used) == len(set(used)) == len(SHAPES)  # every piece once
    if plan == "1":
        assert out[0]["plan_left"] == 0
