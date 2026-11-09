"""The park (real-v5 27 Oct .. gate leg 5): the base snapshot shared by the ranks (rl1_park), reuse with the checksum
check, register and the early snapshot; the loop's host watch."""
import asyncio
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/posttrain"))
import rl1_park  # noqa: E402
from arena import rl_seq  # noqa: E402


def base(seed, empty=False):
    import torch

    g = torch.Generator().manual_seed(seed)
    if empty:
        return []
    return [torch.randn(3, 5, generator=g), torch.randint(0, 255, (7,), dtype=torch.uint8, generator=g),
            torch.randn(2, 2, generator=g).to(torch.bfloat16), torch.zeros(0)]


def _rank(rank, world, inp):
    import torch

    seed, empty, path = inp
    tensors = base(seed, empty)
    views, info = rl1_park.snapshot(tensors, rank, path=path, timeout=20.0)
    return {"equal": all(torch.equal(v, t) for v, t in zip(views, tensors)), "gib": info["gib"],
            "tensors": info["tensors"], "shared": info["shared"], "own_n": info.get("own_n", 0), "own": info["own"]}


@pytest.mark.multirank
def test_snapshot_shared_by_ranks_else_per_rank(tmp_path):
    pytest.importorskip("torch")
    from arc_infra.runops.multirank import run_ranks

    path = str(tmp_path / "rl1-base-x")
    out = run_ranks(_rank, world=3, inputs=[(1, False, path), (1, False, path), (2, False, path)])
    assert all(o["equal"] for o in out) and out[1]["own_n"] == 0 and out[1]["shared"] == 4
    assert out[2]["own_n"] == 3 and out[2]["own"] == ["0", "1", "2"]  # other values: its own copies (the empty one shares)
    path2 = str(tmp_path / "rl1-base-y")
    out = run_ranks(_rank, world=3, inputs=[(1, False, path2), (1, True, path2), (1, False, path2)])  # an empty rank
    assert out[1]["tensors"] == 0 and out[1]["equal"]  # a rank with no base tensors
    assert out[2]["own_n"] == 0 and out[2]["equal"]


def test_views_write_through_one_mapping(tmp_path):
    torch = pytest.importorskip("torch")
    tensors = base(3)
    views, info = rl1_park.snapshot(tensors, 0, path=tmp_path / "s")
    assert info["tensors"] == 4 and all(torch.equal(v, t) for v, t in zip(views, tensors))
    again, _ = rl1_park.snapshot(tensors, 1, path=tmp_path / "s")  # a second rank maps the same file
    assert all(torch.equal(a, t) for a, t in zip(again, tensors))
    assert (tmp_path / "s.json").is_file() and not (tmp_path / "s.part").exists()


class Feed:
    def __init__(self):
        self.rows = []

    def emit(self, kind, **kw):
        self.rows.append({"kind": kind, **kw})


class Trainer:
    def __init__(self):
        self.ops = []

    async def must(self, op, **kw):
        self.ops.append(op)
        return {"ok": True, "seconds": 3.0}


def loop_seq(pending=True):
    loop = types.SimpleNamespace(feed=Feed(), trainer=Trainer())
    seq = types.SimpleNamespace(loop=loop, step=4, ckpt_pending={"checkpoint": "x"} if pending else None)
    seq.ckpt_flush = lambda: True
    return loop, seq


def test_host_free_reads_cgroup_and_meminfo(tmp_path):
    (tmp_path / "meminfo").write_text(f"MemTotal: {1024 * 2**20} kB\nMemAvailable: {300 * 2**20} kB\n")
    assert rl_seq.host_free_gib(tmp_path / "meminfo", tmp_path / "none") == 300.0
    (tmp_path / "memory.max").write_text(str(1024 * 2**30))
    (tmp_path / "memory.stat").write_text(f"anon {900 * 2**30}\nfile 5\n")
    assert rl_seq.host_free_gib(tmp_path / "meminfo", tmp_path) == 124.0


def test_settings_accept_the_new_keys():
    from arena import rl_loop

    assert not {"host_guard", "park_host", "park_prepin"} & rl_loop.MG_KEYS
    assert rl_seq.settings({"redesign": {"streams": 8, "max_steps": 1}})["host_close_gib"] == 190.0


def _real_park_rank(rank, world, folder):
    """The real trainer on a tiny real model (test_rl1_dist.trainer), each rank after a different forward (rank 2
    none): base_tensors() by name, then park_base_shm -> unpark -> park on the shared snapshot."""
    import os

    os.environ["RL1_BASE_ANY_DEVICE"] = "1"
    os.environ["RL1_PARK_SHM_DIR"] = folder
    os.environ["LOCAL_RANK"] = str(rank)
    sys.path.insert(0, str(ROOT / "tests"))
    import torch

    import rl1_trainer as rt
    import test_rl1_dist as D

    t = D.trainer(folder)
    rt.CARD = "cpu"
    if rank < 2:  # asymmetric inputs: lazily sized state differs by the longest sequence a rank has seen
        with torch.no_grad():
            t.model(input_ids=torch.randint(2, D.V, (1, 8 + 24 * rank)))
    tensors = list(t.base_tensors())
    names = list(t.base_names)
    before = {n: v.detach().clone() for n, v in zip(names, tensors)}
    moved, kept = t.park_base_shm()
    info = dict(t.base_shm_info)
    parked_ok = all(torch.equal(v, before[n]) for n, v in zip(names, tensors))
    t.unpark()
    after = list(t.base_tensors())
    back_ok = all(torch.equal(v, before[n]) for n, v in zip(t.base_names, after))
    t.base_shm_info = None
    t.park_base_shm()  # the second park: the same snapshot, nothing written
    return {"names": names, "shapes": {n: list(before[n].shape) for n in names}, "info": info, "moved": moved,
            "kept": kept, "parked_ok": parked_ok, "back_ok": back_ok, "second_wrote": t.base_shm_info is not None}


@pytest.mark.multirank
def test_real_trainer_park_on_asymmetric_ranks(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    pytest.importorskip("peft")
    from arc_infra.runops.multirank import run_ranks

    out = run_ranks(_real_park_rank, world=3, inputs=[str(tmp_path)] * 3, timeout=240)
    for r in out:
        assert r["parked_ok"] and r["back_ok"] and not r["second_wrote"] and r["kept"] == 0 and r["moved"] > 0
    diff = sorted({n for r in out[1:] for n in r["info"]["own"]})
    print("PARK_SPLIT", [{k: r["info"][k] for k in ("shared", "own_n", "own_gib")} for r in out], "differing:", diff,
          {n: [r["shapes"].get(n) for r in out] for n in diff})
    assert out[0]["info"]["shared"] == out[0]["info"]["tensors"]


def test_host_watch_closes_the_phase_with_the_full_groups():
    """real-v7 27 Oct: free host below redesign.host_close_gib during play closes the phase on the groups already full."""
    loop, seq = loop_seq()
    seq.backlog, seq.phase_done = types.SimpleNamespace(full=[]), asyncio.Event()
    frees = iter([300.0] + [150.0] * 50)

    def free():
        if next_full:
            seq.backlog.full.append(next_full.pop())
        return next(frees)

    next_full = []
    got = asyncio.run(asyncio.wait_for(_close_after_one_full(loop, seq, 190.0, free, next_full), 5))
    assert got == 150.0 and seq.phase_done.is_set()
    row = loop.feed.rows[-1]
    assert row["kind"] == "seq_host_close" and row["groups"] == 1 and row["free_gib"] == 150.0
    loop, seq = loop_seq()  # off at 0: returns at once, never closes
    seq.backlog, seq.phase_done = types.SimpleNamespace(full=[1]), asyncio.Event()
    assert asyncio.run(rl_seq.host_watch(loop, seq, 0, free_gib=lambda: 1.0)) is None and not seq.phase_done.is_set()


async def _close_after_one_full(loop, seq, close_gib, free, next_full):
    """Below the line with no full group it waits; the first full group closes the phase."""
    task = asyncio.ensure_future(rl_seq.host_watch(loop, seq, close_gib, every_s=0.0, free_gib=free))
    for _ in range(3):
        await asyncio.sleep(0)
    assert not seq.phase_done.is_set()
    next_full.append({"group": "g1"})
    return await task


def test_checksum_is_exact_across_chunks():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(5)
    t = torch.randint(0, 255, (4099,), dtype=torch.uint8, generator=g)  # a 3-byte tail past the int32 words
    a = rl1_park.checksum(t, chunk=7)
    assert a == rl1_park.checksum(t.clone(), chunk=1 << 24)  # chunking does not change it
    for i in (0, 1500, 4098):  # any single byte, words and tail
        u = t.clone()
        u[i] ^= 1
        assert rl1_park.checksum(u, chunk=7) != a
    u = t.clone()
    u[[8, 12]] = u[[12, 8]]  # two words swapped: the plain sum holds, the weighted one does not
    assert rl1_park.checksum(u) != rl1_park.checksum(t) or torch.equal(u, t)
    assert rl1_park.checksum(torch.zeros(0)) == (0, 0)
    b = torch.randn(3, 5, generator=g).to(torch.bfloat16)
    assert rl1_park.checksum(b) == rl1_park.checksum(b.clone())


def test_reuse_keeps_unchanged_views_and_writes_only_the_rest(tmp_path):
    """gate leg 4: a smaller base layout reuses the snapshot; changed shape or bytes are written to a delta file."""
    torch = pytest.importorskip("torch")
    tensors = base(3)[:3] + [torch.arange(6, dtype=torch.float32)]
    names = ["a", "b", "c", "d"]
    views, info = rl1_park.snapshot(tensors, 0, names=names, path=tmp_path / "s")
    snap = {"names": names, "views": views, "info": info}
    sums = {n: rl1_park.checksum(t) for n, t in zip(names, tensors)}
    later = [tensors[0], tensors[2].clone(), torch.arange(9, dtype=torch.float32)]  # "b" dropped, "d" resized
    later[1][0, 0] += 1  # "c" changed bytes: never reused
    got, rinfo = rl1_park.reuse(snap, later, ["a", "c", "d"], sums, 0, tmp_path / "s", 1)
    assert got[0] is views[0] and rinfo["reused"] == 1 and rinfo["reused_names"] == {"a"}
    assert rinfo["new_n"] == 2 and sorted(rinfo["new"]) == ["c", "d"] and rinfo["dropped"] == 3  # b, c, d: not reused
    assert all(torch.equal(v, t) for v, t in zip(got, later))
    assert (tmp_path / "s-d1-r0").is_file()


def test_register_off_cuda_falls_back_with_a_report(tmp_path):
    """register: no CUDA (or a refused registration) -> ok False with the error and seconds."""
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CPU-only check")
    _, info = rl1_park.snapshot(base(3), 0, path=tmp_path / "s")
    out = rl1_park.register(info["maps"])
    assert out["ok"] is False and out["error"] and out["seconds"] >= 0 and out["bufs"] == []


def _trainer_cpu(folder):
    """The real trainer on the tiny model, in a spawned process only: test_rl1_dist.trainer patches module globals
    (fq.positions_and_images, torch.cuda.synchronize, the attention forward) that would leak into later tests in the
    same worker (CI 41a5b554a: test_rl1_host_index KeyError 'position_ids')."""
    import os

    os.environ.update(RL1_BASE_ANY_DEVICE="1", RL1_PARK_SHM_DIR=folder, LOCAL_RANK="0")
    sys.path.insert(0, str(ROOT / "tests"))
    import rl1_trainer as rt
    import test_rl1_dist as D

    t = D.trainer(folder)
    rt.CARD = "cpu"
    return t


def _reuse_rank(rank, world, folder):
    import torch

    t = _trainer_cpu(folder)
    full = type(t).base_tensors
    tensors = list(full(t))
    loaded = {n: v.detach().clone() for n, v in zip(list(t.base_names), tensors)}
    out = {"n": len(loaded)}
    t.park_base_shm()
    out["first"] = {k: v for k, v in t.base_shm_info.items() if k in ("tensors", "sums_s")}
    t.unpark()

    def fewer(self):  # the first op's effect: one tensor no longer owns its storage
        got = list(full(self))
        self.base_names = self.base_names[1:]
        return got[1:]

    type(t).base_tensors = fewer
    t.base_shm_info = None
    t.park_base_shm()
    out["reuse"] = {k: v for k, v in t.base_shm_info.items() if k in ("reused", "new_n", "dropped")}
    res = t.unpark()
    out["verify"], out["has_base_s"] = res.get("base_verify"), "base_s" in res
    out["back_equal"] = all(torch.equal(v, loaded[n]) for n, v in zip(list(t.base_names), fewer(t)))
    type(t).base_tensors = full  # layout back to the full one: reuse again, every tensor checked at the next unpark
    t.park_base_shm()
    victim = t.base_parked[-1]
    victim.data.view(torch.uint8).reshape(-1)[0] ^= 1  # a wrong view under a reused tensor
    try:
        t.unpark()
        out["corrupt"] = None
    except RuntimeError as exc:
        out["corrupt"] = str(exc)[:200]
    return out


def test_real_trainer_reuses_the_snapshot_and_checks_it_at_unpark(tmp_path):
    """gate leg 4: after a layout change the park writes no reused tensor; the unpark checks every reused tensor's
    full checksum against the loaded weights, and aborts on a wrong view."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    pytest.importorskip("peft")
    from arc_infra.runops.multirank import run_ranks

    out = run_ranks(_reuse_rank, world=1, inputs=[str(tmp_path)], timeout=240)[0]
    n = out["n"]
    assert out["first"]["tensors"] == n and "sums_s" in out["first"]
    assert out["reuse"] == {"reused": n - 1, "new_n": 0, "dropped": 1}
    assert out["verify"]["tensors"] == n - 1 and out["has_base_s"] and out["back_equal"]
    assert out["corrupt"] and "checksum mismatch" in out["corrupt"]


def _early_rank(rank, world, folder):
    import torch

    t = _trainer_cpu(folder)
    tensors = list(t.base_tensors())
    loaded = {n: v.detach().clone() for n, v in zip(list(t.base_names), tensors)}
    started = t.start_base_snap()
    t.park_base_shm()
    info = dict(t.base_shm_info)
    t.base_shm_info = None
    t.unpark()
    back = all(torch.equal(v, loaded[n]) for n, v in zip(list(t.base_names), list(t.base_tensors())))
    t.park_base_shm()
    return {"n": len(loaded), "started": started, "early": info.get("early"), "tensors": info.get("tensors"),
            "back": back, "second": t.base_shm_info}


def test_real_trainer_early_snapshot_leaves_the_park_nothing_to_write(tmp_path):
    """The early snapshot: the snapshot is written on a thread from the load; the first park joins it and copies no
    base; unpark restores the loaded values."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    pytest.importorskip("peft")
    from arc_infra.runops.multirank import run_ranks

    out = run_ranks(_early_rank, world=1, inputs=[str(tmp_path)], timeout=240)[0]
    assert out["started"]["started"] and out["early"]["error"] is None and out["early"]["tensors"] == out["n"]
    assert out["tensors"] == out["n"] and out["back"] and out["second"] is None  # the second park writes nothing
