"""Multi-rank CPU tests (distributed-test standard, 24 Oct): run a function on N processes joined by torch.distributed
over gloo, each rank with its OWN input, so a trainer change is tested with the asymmetric per-rank data that real
runs see (uneven batches, a rank with nothing to do, different sequence lengths). A rank that skips or adds a
collective hangs its peers: here that is a test failure within ``timeout`` seconds naming the hung ranks, not a stuck
GPU job (c7f, 24 Oct: NCCL hang at 100% util after step 15).

    from arc_infra.runops.multirank import run_ranks

    def step(rank, world, batch):                 # module level (spawned processes import it)
        ...                                       # torch.distributed calls; returns something picklable
    def test_uneven_batches():
        out = run_ranks(step, inputs=[[1, 2, 3], [4], []])   # 3 ranks, rank 2 empty
        assert out[0] == out[1] == out[2]

Mark such tests ``@pytest.mark.multirank`` (CI runs them in the fast suite; each takes seconds).
"""
import multiprocessing as mp
import os
import socket
import traceback
from datetime import timedelta


class RankFailure(AssertionError):
    def __init__(self, failures):
        self.failures = failures  # {rank: "error: ..." | "hung (no result in N s)" | "exit code N"}
        super().__init__("multi-rank run failed: " + "; ".join(f"rank {r}: {m}" for r, m in sorted(failures.items())))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _child(fn, rank, world, inp, port, timeout, q):
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timedelta(seconds=timeout))
        dist.barrier()  # every rank's mesh is up before any returns (an early destroy broke a peer's connect: flake)
        res = fn(rank, world, inp)
        q.put((rank, "ok", res))
    except BaseException:  # noqa: BLE001 - every failure goes back to the parent
        q.put((rank, "error", traceback.format_exc()[-2000:]))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def run_ranks(fn, world=3, inputs=None, timeout=30.0):
    """[result of rank 0, 1, ...]; raises RankFailure if any rank errs, exits abnormally or hangs past ``timeout``.
    ``inputs``: one entry per rank (default None for all)."""
    inputs = list(inputs) if inputs is not None else [None] * world
    if len(inputs) != world:
        raise ValueError(f"{len(inputs)} inputs for world {world}")
    ctx = mp.get_context("spawn")
    q, port = ctx.Queue(), _free_port()
    procs = [ctx.Process(target=_child, args=(fn, r, world, inputs[r], port, timeout, q), daemon=True)
             for r in range(world)]
    for p in procs:
        p.start()
    results, failures = {}, {}
    import queue as _queue
    import time

    deadline = time.time() + timeout + 15  # gloo's own timeout fires first; this catches a rank stuck outside it
    while len(results) + len(failures) < world and time.time() < deadline:
        try:
            rank, status, val = q.get(timeout=0.5)
        except _queue.Empty:
            for r, p in enumerate(procs):
                if r not in results and r not in failures and not p.is_alive() and p.exitcode not in (0, None):
                    failures[r] = f"exit code {p.exitcode}"
            continue
        if status == "ok":
            results[rank] = val
        else:
            failures[rank] = "error: " + val.strip().splitlines()[-1][:300] + "\n" + val
    for r, p in enumerate(procs):
        if r not in results and r not in failures:
            failures[r] = f"hung (no result in {timeout + 15:.0f} s)"
        p.join(timeout=2)
        if p.is_alive():
            p.kill()
    if failures:
        raise RankFailure(failures)
    return [results[r] for r in range(world)]
