"""HEAL_STEP_HELDOUT: the per-step held-out KD read. truncate_cpu keeps whole images; on 3 gloo ranks
with asymmetric files (one shorter than a window) the read issues the same calls on every rank, leaves gradients and
every accumulator (the step's training sums included) unchanged, and every rank logs the same per-rank values."""
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))
import sys as _sys  # trunk 24 Oct: PTH's trainer copies (scripts/posttrain/pth) shadow RL's for this module
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "pth"))
for _m in ("rl1_trainer", "flash_qlora_job", "rl1_sparse"):
    _sys.modules.pop(_m, None)
from arc_infra.runops.multirank import run_ranks  # noqa: E402

W, CHUNK = 64, 16


def _cpu(n, images=(), k=4):
    ids = torch.arange(n, dtype=torch.long)[None] + 100
    mm = torch.zeros(1, n, dtype=torch.long)
    for a, b in images:
        ids[0, a:b], mm[0, a:b] = 7, 1
    inputs = {"input_ids": ids, "mm_token_type_ids": mm}
    if images:
        inputs["pixel_values"] = torch.arange(4 * len(images) * 3, dtype=torch.float).view(-1, 3)
        inputs["image_grid_thw"] = torch.tensor([[1, 2, 2]] * len(images))
    return {"inputs": inputs, "teacher": (torch.zeros(n, k), torch.zeros(n, k, dtype=torch.int32), torch.zeros(n)),
            "mask": torch.ones(n), "reply": torch.ones(n), "labels": torch.ones(n, dtype=torch.long),
            "t_tok": torch.zeros(n), "b": None, "w": 1.0, "name": "x"}


def test_truncate_cpu_keeps_whole_images():
    import heal_kd

    cpu = _cpu(40, images=((10, 14), (20, 24)))
    got = heal_kd.truncate_cpu(cpu, 22)  # inside the second image: back to its start
    assert got["inputs"]["input_ids"].shape[1] == 20 and got["inputs"]["image_grid_thw"].shape[0] == 1
    assert got["inputs"]["pixel_values"].shape[0] == 4 and got["teacher"][0].shape[0] == 20
    assert got["mask"][-1] == 0 and got["reply"][-1] == 0 and got["labels"][-1] == -100 and got["mask"][:-1].all()
    assert cpu["mask"].all()  # the source is untouched
    whole = heal_kd.truncate_cpu(_cpu(30), 99)
    assert whole["inputs"]["input_ids"].shape[1] == 30 and "pixel_values" not in whole["inputs"]


def _rank(rank, world, inp):
    import transformers

    import heal_kd

    os.environ.pop("HEAL_CAPTURE_CACHE", None)
    seq, w = [], torch.nn.Parameter(torch.ones(3))
    w.grad = torch.full((3,), 0.25)

    class Cache:
        def __init__(self, *a, **k):
            pass

    transformers.DynamicCache = Cache

    class Runner:
        training, mtp = True, None
        config = type("C", (), {"text_config": None})()

        def eval(self):
            self.training = False

        def train(self, was=True):
            self.training = was

        def __call__(self, mode, ids, a, b, cache, held=None, **kw):
            seq.append(mode)
            if mode == "extend":
                return cache
            n = float(held.reply[a:b].sum()) if held is not None and held.reply is not None else 0.0
            loss = (w * float(b - a)).sum()
            return loss, 0.0, 0.0, 0.01 * (rank + 1) * n, 0.0, 0.0

    def data(total):
        d = heal_kd.teacher_to(heal_kd.truncate_cpu(_cpu(total), total), "cpu")
        return d

    sh = heal_kd.StepHeldout.__new__(heal_kd.StepHeldout)
    sh.runner, sh.window, sh.chunk = Runner(), W, CHUNK
    sh.data = data(inp["tokens"][rank])
    sh.slot = {"data": sh.data, "mask": sh.data["mask"], "piece": None}
    sh.dummy = {**data(16), "piece": None}
    sh.file, sh.tokens, sh.reply_rows = Path(f"f{rank}.pt"), inp["tokens"][rank], int(sh.data["host"]["reply_n"])
    heal_kd.TRAIN_ACC.update(on=True, value=1.5 + rank, n=10.0)
    heal_kd.REV_ACC.update(on=True, kl=0.3)
    heal_kd.READ_ACC.update(mix=2.0)
    before = {k: dict(v) for k, v in (("t", heal_kd.TRAIN_ACC), ("r", heal_kd.REV_ACC), ("d", heal_kd.READ_ACC),
                                      ("s", heal_kd.SAMP_ACC), ("c", heal_kd.CLASS_ACC))}
    got = sh.read()
    after = {k: dict(v) for k, v in (("t", heal_kd.TRAIN_ACC), ("r", heal_kd.REV_ACC), ("d", heal_kd.READ_ACC),
                                     ("s", heal_kd.SAMP_ACC), ("c", heal_kd.CLASS_ACC))}
    return {"got": got, "setup": sh.setup_row(), "calls": len(seq), "same_acc": before == after,
            "grad": w.grad.tolist()}


@pytest.mark.multirank
def test_step_heldout_is_neutral_and_lockstep():
    tokens = [150, 90, 20]  # rank 2: shorter than a window
    out = run_ranks(_rank, inputs=[{"tokens": tokens}] * 3, timeout=60)
    assert out[0]["got"] == out[1]["got"] == out[2]["got"]
    assert len({o["calls"] for o in out}) == 1
    assert all(o["same_acc"] and o["grad"] == [0.25] * 3 for o in out)
    per = out[0]["got"]["heldout_kd_rank"]
    assert per == [round(0.01 * (r + 1), 6) for r in range(3)]  # each rank's own file, own value
    assert out[0]["got"]["heldout_kd"] == round(sum(per) / 3, 6)
    assert out[0]["setup"]["reply_rows"] == [t - 1 for t in tokens]


def test_live_lr_ramps_each_change(tmp_path):
    """HEAL_LIVE_KNOBS: the first value applies at once, each later change ramps over 4 steps, a bad
    or missing value means the schedule as launched."""
    import heal_kd

    f = tmp_path / "trainer.json"
    f.write_text('{"lr": 7e-6}')
    live = heal_kd.LiveLr(f, 7e-6, ramp=4)
    assert live.scale(10, 0) == 1.0
    f.write_text('{"lr": 5e-6}')
    got = [round(live.scale(s, 0), 6) for s in range(11, 16)]
    assert got == [round(1 - (2 / 7) * k / 4, 6) for k in (1, 2, 3, 4, 4)]
    f.write_text('{"lr": "x"}')
    assert round(live.scale(16, 0), 6) == round(5 / 7 + (1 - 5 / 7) * 1 / 4, 6)  # back toward the launched lr
    f.unlink()
    assert live.scale(30, 0) == 1.0 and live.seen is None


def _lr_rank(rank, world, inp):
    import heal_kd

    live = heal_kd.LiveLr(inp["file"] if rank == 0 else inp["file"] + ".missing", 7e-6)
    return [live.scale(s, rank) for s in (1, 2)]


@pytest.mark.multirank
def test_live_lr_is_rank_zeros_value_on_every_rank(tmp_path):
    f = tmp_path / "trainer.json"
    f.write_text('{"lr": 1e-5}')
    out = run_ranks(_lr_rank, inputs=[{"file": str(f)}] * 3, timeout=60)
    assert out[0] == out[1] == out[2] == [1e-5 / 7e-6] * 2


def _fresh_rank(rank, world, inp):
    import json as js
    import types

    import transformers

    import heal_kd

    os.environ["HEAL_FRESH_HOLDOUT"] = "0.5"
    os.environ.pop("HEAL_CAPTURE_CACHE", None)
    sizes = {}

    class Cache:
        def __init__(self, *a, **k):
            pass

    transformers.DynamicCache = Cache
    sys.modules["flash_qlora_job"] = types.SimpleNamespace(positions_and_images=lambda m, i: None)
    heal_kd.load_teacher_cpu = lambda path, k=None: _cpu(sizes.setdefault(path, 40 + 10 * len(sizes)))
    heal_kd.dp_dummy = lambda model, device, k: {**heal_kd.teacher_to(_cpu(16), "cpu"), "piece": None}
    w = torch.nn.Parameter(torch.ones(2))

    class Runner:
        training, mtp = True, None
        config = type("C", (), {"text_config": None})()

        def eval(self):
            self.training = False

        def train(self, was=True):
            self.training = was

        def __call__(self, mode, ids, a, b, cache, held=None, **kw):
            if mode == "extend":
                return cache
            n = float(held.reply[a:b].sum()) if held is not None and held.reply is not None else 0.0
            return (w * 1.0).sum(), 0.0, 0.0, 0.5 * n, 0.0, 0.0

    q = heal_kd.StreamQueue(inp["folder"], rank)
    trained = sorted(r[2] for r in q.ready()) if rank == 0 else None
    fr = heal_kd.FreshHoldout(Runner(), None, q, "cpu", window=W, chunk=CHUNK, rank=rank, world=world)
    first, second, third = fr.read(9), fr.read(10), fr.read(11)
    return {"trained": trained, "held": sorted(q.held) if rank == 0 else None, "first": first, "second": second,
            "third": third}


@pytest.mark.multirank
def test_fresh_holdout_keeps_whole_chains_out_and_reads_each_piece_once(tmp_path):
    import json as js

    import heal_kd

    names = [f"s-g{g}--c{c:03d}-p{p:04d}" for g in range(6) for c in range(1) for p in range(2)]
    for i, name in enumerate(names):
        (tmp_path / f"{name}.pt").write_bytes(b"")
        (tmp_path / f"{name}.ready").write_text(js.dumps({"version": 8 if i % 2 else 7, "tokens": 40, "rows": 40}))
    (tmp_path / "b-base--c000-p0000.pt").write_bytes(b"")
    (tmp_path / "b-base--c000-p0000.ready").write_text(js.dumps({"version": 0, "base": True}))
    out = run_ranks(_fresh_rank, inputs=[{"folder": str(tmp_path)}] * 3, timeout=60)
    r0 = out[0]
    os.environ["HEAL_FRESH_HOLDOUT"] = "0.5"
    q = heal_kd.StreamQueue(str(tmp_path), 0)
    held_chains = {q.chain_of(n)[0] for n in names if q.held_out(n + ".pt")}
    assert 0 < len(held_chains) < 6
    for path in r0["trained"]:  # no piece of a held-out chain is trained; base never held out
        assert q.chain_of(path)[0] not in held_chains
    assert any("b-base" in p for p in r0["trained"])
    reads = [r0[k] for k in ("first", "second", "third")]
    assert all(o[k] == r0[k] for o in out for k in ("first", "second", "third"))
    n_held = 2 * len(held_chains)
    got = [x for r in reads if r for x in r["fresh_kd_rank"] if x is not None]
    assert len(got) == min(n_held, 9) and all(abs(x - 0.5) < 1e-6 for x in got)
    assert reads[0]["fresh_age"][0] == 1  # freshest first (version 8 at step 9)


def _reshard_rank(rank, world, inp):
    """FSDP2 on gloo: a root group with an embedding (as Runner: decoder layers and the root fully_shard'ed); a
    forward without backward leaves the root unsharded; local_params must still save rank shards that reload."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    import heal_kd

    torch.manual_seed(0)
    mesh = init_device_mesh("cpu", (world,))

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(30, 4)
            self.layer = torch.nn.Linear(4, 4)

        def forward(self, ids):
            return self.layer(self.embed(ids)).sum()

    net = Net()
    fully_shard(net.layer, mesh=mesh)
    fully_shard(net, mesh=mesh)
    shard_rows = {n: (p.to_local().shape[0] if hasattr(p, "to_local") else None) for n, p in net.named_parameters()}
    with torch.no_grad():  # the held-out read: no backward
        net(torch.arange(6)[None])
    full_after = {n: not hasattr(p, "to_local") for n, p in net.named_parameters()}
    got = heal_kd.local_params(net)
    rows = {n: int(t.shape[0]) for n, t in got.items()}
    folder = Path(inp["folder"])
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    torch.save(got, folder / f"params-rank{rank}.pt")
    dist.barrier()
    with torch.no_grad():
        net(torch.arange(6)[None])
    n = heal_kd.load_local_params(net, folder, rank)
    return {"shard_rows": shard_rows, "rows": rows, "unsharded_by_read": full_after, "loaded": n}


@pytest.mark.multirank
def test_checkpoint_after_a_no_grad_read_saves_shards(tmp_path):
    """c7k preflight (24 Oct): a step held-out read (no backward) left the FSDP root unsharded, the checkpoint saved
    embed_tokens whole (248320 x 64) and the resume refused it against its 82774-row shard."""
    out = run_ranks(_reshard_rank, inputs=[{"folder": str(tmp_path / "ck")}] * 3, timeout=300)
    for r in out:
        assert r["rows"]["embed.weight"] == r["shard_rows"]["embed.weight"] == 10, r
        assert r["loaded"] == 3, r
    assert any(r["unsharded_by_read"]["embed.weight"] for r in out), out  # the read does leave the root whole
