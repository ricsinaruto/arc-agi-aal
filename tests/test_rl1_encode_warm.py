"""RL1_ENCODE_WARM: every encoder process returns one warm-up result through the pool before any window."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))


def fake_loader(model, window):  # module level: the spawned processes import it
    return None, 0


@pytest.mark.parametrize("procs", [1, 3])
def test_each_process_warms_once(procs):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import rl1_trainer as rt

    t = rt.Trainer.__new__(rt.Trainer)
    t.args = SimpleNamespace(model=None, window=4096)
    t.encoders, t.encoder_loader = None, fake_loader
    pool = t.encoder_pool(procs, warm=True)
    try:
        state = t.warm_encoders(pool, procs)
        state["thread"].join(120)
        assert not state.get("error"), state
        assert state["processes"] == procs and state["met"] == procs  # one warm-up per process, none twice
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def test_warm_without_barrier_and_pickled():
    torch = pytest.importorskip("torch")
    import rl1_encode

    kept = dict(rl1_encode.STATE)
    try:
        rl1_encode.STATE.update(shm=False, barrier=None)
        packed = rl1_encode.warm(nbytes=64)
        made = rl1_encode.unpack(packed)
        assert made["met"] is None and torch.equal(made["inputs"]["input_ids"], torch.zeros(8, dtype=torch.int64))
    finally:
        rl1_encode.STATE.clear()
        rl1_encode.STATE.update(kept)
