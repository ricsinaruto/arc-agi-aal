"""c7q (25 Oct): rank 0 waited for fresh stream chunks (HEAL_STREAM_STALL 1800 s) while ranks 1-2 sat in take()'s
gather; the 600 s NCCL watchdog killed the run. The process groups' timeout now exceeds the stream stall."""
import os
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arc_infra.runops.multirank import RankFailure, run_ranks  # noqa: E402


def _wait_then_gather(rank, world, env):
    sys.path.insert(0, env.pop("_PATH"))
    os.environ.update(env)
    import torch.distributed as dist

    import heal_kd

    group = dist.new_group(backend="gloo", timeout=heal_kd.pg_timeout())
    if rank == 0:
        time.sleep(3.5)  # rank 0 polls the stream for fresh chunks (longer than the floor)
    every = [None] * world
    dist.all_gather_object(every, rank, group=group)
    return every


@pytest.mark.multirank
def test_stream_wait_longer_than_the_floor_keeps_the_ranks_alive():
    path = str(ROOT / "scripts" / "posttrain")
    env = {"_PATH": path, "HEAL_STREAM": "/x", "HEAL_STREAM_STALL": "3", "HEAL_PG_TIMEOUT_FLOOR": "1",
           "HEAL_PG_TIMEOUT_GRACE": "2"}  # timeout 5 s > rank 0's 3.5 s wait > the 1 s floor
    out = run_ranks(_wait_then_gather, world=3, inputs=[dict(env) for _ in range(3)], timeout=30)
    assert out == [[0, 1, 2]] * 3


@pytest.mark.multirank
def test_without_the_stall_the_floor_alone_kills_the_wait():
    path = str(ROOT / "scripts" / "posttrain")
    env = {"_PATH": path, "HEAL_PG_TIMEOUT_FLOOR": "1"}  # no stream: timeout = floor 1 s < the 3.5 s wait
    with pytest.raises(RankFailure):
        run_ranks(_wait_then_gather, world=3, inputs=[dict(env) for _ in range(3)], timeout=30)
