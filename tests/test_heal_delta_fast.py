"""MFU-1 HEAL_DELTA_FAST: batched int4 and row-only gathers write the same delta files as the per-row path."""
import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import sys as _sys  # trunk 24 Oct: PTH's trainer copies (scripts/posttrain/pth) shadow RL's for this module
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "pth"))
for _m in ("rl1_trainer", "flash_qlora_job", "rl1_sparse"):
    _sys.modules.pop(_m, None)
import heal_delta  # noqa: E402

NAME = "model.language_model.layers.{}.mlp.experts.{}"


def _cut(tmp_path):
    prune = {"format": "arc-expert-prune-v1", "experts": 12, "pruned": {str(l): [0, 2, 5, 7] for l in range(4)}}
    (tmp_path / "heal_prune-t50-excl5-v1.json").write_text(json.dumps(prune))
    return heal_delta.Cut({"student_cut": {"source_layers": [0, 1, 3],
                                           "prune_file": str(tmp_path / "heal_prune-t50-excl5-v1.json")}})


def _same(a, b):
    assert [k for k, _ in a] == [k for k, _ in b]
    for (k, x), (_, y) in zip(a, b):
        assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y), k


def test_int4_rows_bitwise():
    torch.manual_seed(1)
    ws = torch.randn(5, 64, 256) * 0.02
    ws[2, :, :128] = 0.0  # an all-zero group
    got = heal_delta.int4_rtn_rows(ws, search=20, chunk=2)
    for e in range(5):
        ref = heal_delta.int4_rtn(ws[e], search=20)
        for x, y in zip(got, ref):
            assert torch.equal(x[e], y)


@pytest.mark.parametrize("kind", ["gate_up_proj", "down_proj"])
@pytest.mark.parametrize("rows", [None, [1, 3, 6]])
def test_entries_fast_equal(tmp_path, monkeypatch, kind, rows):
    cut = _cut(tmp_path)
    torch.manual_seed(2)
    fused = torch.randn(8, 256, 128) * 0.02 if kind == "gate_up_proj" else torch.randn(8, 128, 256) * 0.02
    name = NAME.format(2, kind)
    monkeypatch.delenv("HEAL_DELTA_FAST", raising=False)
    slow = heal_delta.entries(name, fused, cut, rows=rows, search=5)
    monkeypatch.setenv("HEAL_DELTA_FAST", "1")
    _same(heal_delta.entries(name, fused, cut, rows=rows, search=5), slow)
    if rows:
        _same(heal_delta.entries(name, fused[rows], cut, rows=rows, search=5, picked=True), slow)
    assert heal_delta.entries(name, fused[:0], cut, rows=[], search=5, picked=True) == []


def _rank(rank, world, root, port, fast, asym=False):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    if fast:
        os.environ["HEAL_DELTA_FAST"] = "1"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    torch.manual_seed(3)
    full = {"model." + NAME.format(0, "gate_up_proj"): torch.randn(8, 256, 128) * 0.02,
            "model." + NAME.format(1, "down_proj"): torch.randn(8, 128, 256) * 0.02,
            "model.model.language_model.layers.0.self_attn.o_proj.weight": torch.randn(16, 16),
            "model.mtp.layers.0.mlp.experts.down_proj": torch.randn(8, 4, 4),
            "model.model.language_model.embed_tokens.weight": torch.randn(9, 4)}
    params = [(k, torch.nn.Parameter(distribute_tensor(v, mesh, [Shard(0)]))) for k, v in full.items()]
    runner = type("R", (), {"named_parameters": lambda self: params})()
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    items = []
    for k, p in params[:2]:
        _, off = compute_local_shape_and_global_offset(p.shape, p.device_mesh, p.placements)
        sel = [1, 2, 6]  # spans the 3/3/2 shards
        mine = [e - off[0] for e in sel if off[0] <= e < off[0] + p.to_local().shape[0]]
        items.append({"p": p, "rows": torch.tensor(mine, dtype=torch.long), "start": off[0]})
    rows = type("Rows", (), {"items": items})()
    cut = heal_delta.Cut({"student_cut": {"source_layers": [0, 1]}})
    ex = heal_delta.DeltaExporter(Path(root), cut, rank, files=2, search=5, generation=7,
                                  **({"confirm": lambda g: True} if asym else {}))
    info = [ex.export(runner, rows, 0, force=True, first=True)]
    if asym:  # c7f: only rank 0's writer polls the play's confirm
        ex.wait()
        ex.confirmed = rank == 0
    info.append(ex.export(runner, rows, 1, force=True))
    ex.wait()
    dist.barrier()
    dist.destroy_process_group()
    return info


def _spawn(root, port, fast, asym=False):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_rank, args=(r, 3, root, port, fast, asym)) for r in range(3)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
        assert p.exitcode == 0


def test_exporter_fast_files_equal(tmp_path):
    from safetensors.torch import load_file

    _spawn(str(tmp_path / "slow"), 29611, False)
    _spawn(str(tmp_path / "fast"), 29612, True)
    for v in ("v0", "v1"):
        a = {k: t for f in sorted((tmp_path / "slow" / v).glob("*.safetensors")) for k, t in load_file(str(f)).items()}
        b = {k: t for f in sorted((tmp_path / "fast" / v).glob("*.safetensors")) for k, t in load_file(str(f)).items()}
        assert sorted(a) == sorted(b) and a
        for k in a:
            assert torch.equal(a[k], b[k]), (v, k)
        ma = json.loads((tmp_path / "slow" / v / "manifest.json").read_text())
        mb = json.loads((tmp_path / "fast" / v / "manifest.json").read_text())
        assert ma["full"] == mb["full"] == (v == "v0") and ma["tensors"] == mb["tensors"]
    assert len([k for k in a if ".experts." in k]) == 2 * 3 * 3 + 3 * 3  # v1: rows 1, 2, 6 only


def test_exporter_fast_agrees_on_full_when_only_rank0_confirmed(tmp_path):
    """c7f (24 Oct): rank 0 saw the play's confirm, ranks 1-2 did not; the fast path's gathers then differed and the
    round hung in an all_gather. Every rank now takes rank 0's full/delta view."""
    _spawn(str(tmp_path / "asym"), 29613, True, asym=True)
    m = json.loads((tmp_path / "asym" / "v1" / "manifest.json").read_text())
    assert m["full"] is False
