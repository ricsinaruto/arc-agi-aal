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


STARTUP_S = 180.0  # spawned ranks import torch first; under a loaded CI host that alone took longer than a 5 s timeout


def _child(fn, rank, world, inp, port, timeout, q, backend="gloo", ready=None, done=None):
    """One rank. ``ready``: a barrier of the ranks after their imports, so the process group's rendezvous (bounded
    by ``timeout``) starts together (CI 26 Oct: a rank still importing under load timed the others out at 5 s).
    ``done``: set by the parent once it has every rank's row; a rank keeps its process group until then, so one
    that returned early never tears down under peers still inside a collective (its row was lost: CI 26 Oct)."""
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    inited = False
    try:
        if ready is not None:
            ready.wait(STARTUP_S)
            q.put((rank, "ready", None))
        if backend == "nccl":  # GPU tests: rank r on cuda:r
            import torch

            torch.cuda.set_device(rank)
        dist.init_process_group(backend, rank=rank, world_size=world, timeout=timedelta(seconds=timeout))
        inited = True
        dist.barrier()  # every rank's mesh is up before any returns (an early destroy broke a peer's connect: flake)
        res = fn(rank, world, inp)
        q.put((rank, "ok", res))
    except BaseException:  # noqa: BLE001 - every failure goes back to the parent
        q.put((rank, "error", traceback.format_exc()[-2000:]))
    finally:
        try:  # the row reaches the parent before anything else can kill this process
            q.close()
            q.join_thread()
        except Exception:  # noqa: BLE001
            pass
        if done is not None:
            done.wait(timeout + 30)
        if inited:
            try:
                dist.destroy_process_group()
            except Exception:  # noqa: BLE001
                pass


def run_ranks(fn, world=3, inputs=None, timeout=30.0, backend="gloo"):
    """[result of rank 0, 1, ...]; raises RankFailure if any rank errs, exits abnormally or hangs past ``timeout``.
    ``inputs``: one entry per rank (default None for all). ``backend`` "nccl": GPU tests, rank r on cuda:r."""
    inputs = list(inputs) if inputs is not None else [None] * world
    if len(inputs) != world:
        raise ValueError(f"{len(inputs)} inputs for world {world}")
    ctx = mp.get_context("spawn")
    q, port = ctx.Queue(), _free_port()
    ready, done = ctx.Barrier(world), ctx.Event()
    procs = [ctx.Process(target=_child, args=(fn, r, world, inputs[r], port, timeout, q, backend, ready, done),
                         daemon=True) for r in range(world)]
    for p in procs:
        p.start()
    results, failures = {}, {}
    import queue as _queue
    import time

    # the hang clock starts once every rank is past its imports (startup is bounded separately, STARTUP_S)
    started, seen = time.time(), set()
    deadline = started + STARTUP_S + timeout + 15
    while len(results) + len(failures) < world and time.time() < deadline:
        try:
            rank, status, val = q.get(timeout=0.5)
        except _queue.Empty:
            for r, p in enumerate(procs):
                if r not in results and r not in failures and not p.is_alive() and p.exitcode not in (0, None):
                    failures[r] = f"exit code {p.exitcode}"
            continue
        if status == "ready":
            seen.add(rank)
            if len(seen) == world:
                deadline = time.time() + timeout + 15  # gloo's own timeout fires first; this catches a rank stuck outside it
        elif status == "ok":
            results[rank] = val
            failures.pop(rank, None)  # its row arrived: it passed, whatever its exit code read meanwhile
        else:
            failures[rank] = "error: " + val.strip().splitlines()[-1][:300] + "\n" + val
    done.set()  # every rank may tear down now
    for r, p in enumerate(procs):
        if r not in results and r not in failures:
            failures[r] = f"hung (no result in {timeout + 15:.0f} s)"
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
    if failures:
        raise RankFailure(failures)
    return [results[r] for r in range(world)]
