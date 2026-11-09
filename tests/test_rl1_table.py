"""RL1_SHARED_TABLE (rl1_table): the n-gram table built once from the FP8 shards into a read-only shared mapping."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/posttrain"))
sys.path.insert(0, str(ROOT / "src"))

KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


def checkpoint(folder, shards=12, rows=37, cols=8, scale="rows"):
    """A sharded FP8 table (shard_0..shard_<n-1> in two files, keys stored out of numeric order) plus its scale."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    gen = torch.Generator().manual_seed(0)
    total = shards * rows
    files, weight_map = {"a.safetensors": {}, "b.safetensors": {}}, {}
    for index in range(shards):
        piece = (torch.randn(rows, cols, generator=gen) * 3).to(torch.float8_e4m3fn)
        name = "a.safetensors" if index % 2 else "b.safetensors"
        files[name][f"{KEY}.shard_{index}.weight"] = piece
    if scale == "rows":
        files["a.safetensors"][f"{KEY}.weight_scale"] = (torch.rand(total, 1, generator=gen) + 0.5)
    elif scale == "scalar":
        files["a.safetensors"][f"{KEY}.weight_scale"] = torch.tensor(0.37)
    files["b.safetensors"]["model.other.weight"] = torch.ones(2, 2)
    for name, tensors in files.items():
        save_file(tensors, str(Path(folder) / name))
        weight_map.update({k: name for k in tensors})
    weight_map = dict(sorted(weight_map.items()))  # lexicographic: shard_10 before shard_2
    (Path(folder) / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return files, total, cols


def reference(files, total, cols):
    """The load today: each shard FP8 -> BF16 (the loader's cast), concatenated in shard order, then
    apply_table_scale's in-place ``(piece.float() * rows).to(bfloat16)``."""
    import torch

    tensors = {k: v for f in files.values() for k, v in f.items()}
    count = sum(1 for k in tensors if ".shard_" in k)
    shards = [tensors[f"{KEY}.shard_{i}.weight"].to(torch.bfloat16) for i in range(count)]
    table = torch.cat(shards)
    scale = tensors.get(f"{KEY}.weight_scale")
    if scale is not None:
        scale = scale.float()
        step = 50
        for start in range(0, table.shape[0], step):
            piece = table[start:start + step]
            rows = scale[start:start + step] if scale.dim() and scale.shape[0] == table.shape[0] else scale
            piece.copy_((piece.float() * rows).to(piece.dtype))
    assert table.shape == (total, cols)
    return table


@pytest.mark.parametrize("scale", ["rows", "scalar", None])
def test_build_equals_the_load_plus_table_scale(tmp_path, scale):
    torch = pytest.importorskip("torch")
    import rl1_table

    files, total, cols = checkpoint(tmp_path, scale=scale)
    spec = rl1_table.plan(tmp_path)
    assert [s["key"].split(".")[-2] for s in spec["shards"]][:3] == ["shard_0", "shard_1", "shard_2"]
    assert (spec["rows"], spec["cols"], spec["bytes"]) == (total, cols, total * cols * 2)
    assert spec["name"] == f"{KEY}.weight"
    out = torch.empty(total, cols, dtype=torch.bfloat16)
    rl1_table.fill(out, spec, tmp_path, workers=3, step=10)  # pieces cut across shards and threads
    assert torch.equal(out.view(torch.int16), reference(files, total, cols).view(torch.int16))


def test_plan_refuses_a_gap_and_start_falls_back(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import rl1_table

    checkpoint(tmp_path)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    del index["weight_map"][f"{KEY}.shard_3.weight"]
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        rl1_table.plan(tmp_path)
    assert rl1_table.start(tmp_path, {"RL1_SHARED_TABLE": "1"}) is False  # today's load
    assert rl1_table.start(tmp_path, {}) is False


def env(tmp_path, ranks=1, local=0, **extra):
    return {"RL1_SHARED_TABLE": "1", "RL1_SHARED_TABLE_DIR": str(tmp_path / "shm"), "LOCAL_WORLD_SIZE": str(ranks),
            "LOCAL_RANK": str(local), **extra}


def test_one_rank_builds_maps_read_only_and_unlinks(tmp_path):
    torch = pytest.importorskip("torch")
    import rl1_table

    (tmp_path / "ckpt").mkdir()
    (tmp_path / "shm").mkdir()
    files, total, cols = checkpoint(tmp_path / "ckpt")
    spec = rl1_table.plan(tmp_path / "ckpt")
    table, row = rl1_table.obtain(spec, tmp_path / "ckpt", env(tmp_path), workers=2)
    assert row["source"] == "built" and row["read_only"] and not row.get("error")
    assert torch.equal(table.view(torch.int16), reference(files, total, cols).view(torch.int16))
    assert list((tmp_path / "shm").iterdir()) == []  # unlinked; the mapping lives on


def test_a_write_to_the_mapping_kills_the_process(tmp_path):
    pytest.importorskip("torch")
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "shm").mkdir()
    checkpoint(tmp_path / "ckpt")
    code = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(ROOT / 'scripts/posttrain')!r})
        import rl1_table
        spec = rl1_table.plan({str(tmp_path / 'ckpt')!r})
        env = {{"RL1_SHARED_TABLE_DIR": {str(tmp_path / 'shm')!r}, "LOCAL_WORLD_SIZE": "1", "LOCAL_RANK": "0"}}
        table, row = rl1_table.obtain(spec, {str(tmp_path / 'ckpt')!r}, env, workers=1)
        print("read", float(table[0, 0]), row["read_only"], flush=True)
        table[0, 0] = 1.0
        print("wrote", flush=True)
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "read" in out.stdout and "True" in out.stdout
    assert "wrote" not in out.stdout and out.returncode < 0  # SIGSEGV / SIGBUS, never a silent private copy


def test_stale_launch_files_are_removed(tmp_path):
    import rl1_table

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (tmp_path / f"rl1-table-{dead.pid}").write_bytes(b"x")
    (tmp_path / f"rl1-table-{dead.pid}.done0").write_bytes(b"")
    (tmp_path / f"rl1-table-{os.getpid()}").write_bytes(b"x")  # a live launch keeps its file
    (tmp_path / "other").write_bytes(b"x")
    removed = rl1_table.remove_stale(tmp_path, "rl1-table-0")
    assert sorted(removed) == sorted([f"rl1-table-{dead.pid}", f"rl1-table-{dead.pid}.done0"])
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([f"rl1-table-{os.getpid()}", "other"])


def test_a_reader_without_a_builder_builds_its_own(tmp_path):
    torch = pytest.importorskip("torch")
    import rl1_table

    (tmp_path / "ckpt").mkdir()
    (tmp_path / "shm").mkdir()
    files, total, cols = checkpoint(tmp_path / "ckpt")
    spec = rl1_table.plan(tmp_path / "ckpt")
    table, row = rl1_table.obtain(spec, tmp_path / "ckpt", env(tmp_path, ranks=2, local=1,
                                                                RL1_SHARED_TABLE_WAIT_S="0.3"), workers=1, poll=0.05)
    assert row["error"] == "wait timed out" and row["source"] == "own" and row["read_only"]
    assert torch.equal(table.view(torch.int16), reference(files, total, cols).view(torch.int16))


def shared_rank(rank, world, inp):
    """A local rank of one launch (the ranks share the test process as their parent, as torchrun's agent)."""
    import torch

    import rl1_table

    folder, fail = inp
    os.environ.update(env(Path(folder), ranks=world, local=rank, RL1_SHARED_TABLE_WAIT_S="60"))
    if fail:  # the builder's shared file fails (e.g. a full tmpfs): every rank builds its own
        real = rl1_table.build_file

        def build_file(path, *args, **kwargs):
            if ".own" not in str(path):
                raise OSError(28, "no space (test)")
            return real(path, *args, **kwargs)

        rl1_table.build_file = build_file
    spec = rl1_table.plan(Path(folder) / "ckpt")
    table, row = rl1_table.obtain(spec, Path(folder) / "ckpt", workers=2, poll=0.05)
    torch.distributed.barrier()  # every rank done before the listing
    left = sorted(p.name for p in (Path(folder) / "shm").iterdir())
    return {"source": row["source"], "read_only": row["read_only"], "digest": rl1_table.sample_digest(table, 8, 4),
            "ptr_shared": row["source"] in ("built", "mapped"), "left": left, "error": row.get("error")}


@pytest.mark.multirank
@pytest.mark.parametrize("fail", [False, True])
def test_ranks_share_one_table(tmp_path, fail):
    pytest.importorskip("torch")
    from arc_infra.runops.multirank import run_ranks

    (tmp_path / "ckpt").mkdir()
    (tmp_path / "shm").mkdir()
    checkpoint(tmp_path / "ckpt")
    out = run_ranks(shared_rank, world=3, inputs=[(str(tmp_path), fail)] * 3, timeout=120)
    assert len({o["digest"] for o in out}) == 1 and all(o["read_only"] for o in out)
    if fail:
        assert [o["source"] for o in out] == ["own"] * 3
        assert out[1]["error"] == "builder failed" and "no space" in out[0]["error"]
    else:
        assert [o["source"] for o in out] == ["built", "mapped", "mapped"] and not any(o["error"] for o in out)
    assert out[0]["left"] == out[1]["left"] == out[2]["left"] == []  # the last rank unlinked name and markers


def test_meta_loading_skips_only_the_table_shards(monkeypatch):
    torch = pytest.importorskip("torch")
    import types

    import rl1_table

    calls = []
    fake = types.ModuleType("transformers.core_model_loading")
    fake.get_device = lambda device_map, name, valid_torch_device=False: "cpu"
    fake.spawn_materialize = lambda pool, tensor, device=None, dtype=None, **kw: calls.append(tensor) or "real"
    package = types.ModuleType("transformers")
    package.core_model_loading = fake
    monkeypatch.setitem(sys.modules, "transformers", package)
    monkeypatch.setitem(sys.modules, "transformers.core_model_loading", fake)
    rl1_table.PENDING.clear()
    rl1_table.meta_loading()
    rl1_table.meta_loading()  # once

    class Slice:
        def get_shape(self):
            return [5, 3]

        def get_dtype(self):
            return "F8_E4M3"

    shard = f"{KEY}.shard_0.weight"
    assert fake.get_device({}, shard, valid_torch_device=True) == "meta"
    value = fake.spawn_materialize(None, Slice(), "meta", torch.bfloat16)()
    assert value.device.type == "meta" and tuple(value.shape) == (5, 3) and value.dtype == torch.bfloat16
    assert fake.get_device({}, f"{KEY}.weight") == "meta"  # the concatenated table stays on meta
    assert fake.get_device({}, "model.layers.0.mlp.weight", valid_torch_device=True) == "cpu"
    assert fake.spawn_materialize(None, Slice(), "cpu", None) == "real" and len(calls) == 1  # not a table shard
    assert rl1_table.PENDING["shards"] == 1


def test_attach_and_guard(tmp_path):
    torch = pytest.importorskip("torch")
    import rl1_table

    (tmp_path / "ckpt").mkdir()
    (tmp_path / "shm").mkdir()
    files, total, cols = checkpoint(tmp_path / "ckpt")
    assert rl1_table.start(tmp_path / "ckpt", env(tmp_path), workers=2)

    class Ple(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ngram_embedding = torch.nn.Embedding(total, cols, device="meta", dtype=torch.bfloat16)
            self.lora = torch.nn.Linear(2, 2)

    model = torch.nn.Sequential(Ple())
    row = rl1_table.attach(None, model)
    assert row["source"] == "built" and row["param"] == "0.ngram_embedding.weight" and not row["loaded_by_transformers"]
    weight = model[0].ngram_embedding.weight
    assert isinstance(weight, torch.nn.Parameter) and not weight.requires_grad
    assert torch.equal(weight.view(torch.int16), reference(files, total, cols).view(torch.int16))
    assert model[0].ngram_embedding(torch.tensor([0, total - 1])).shape == (2, cols)  # reads work
    good = torch.optim.AdamW(model[0].lora.parameters())
    assert rl1_table.guard(model, good)["ok"]
    with pytest.raises(RuntimeError, match="optimizer"):
        rl1_table.guard(model, torch.optim.AdamW([weight, *model[0].lora.parameters()]))
    weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="requires_grad"):
        rl1_table.guard(model, good)
    weight.requires_grad_(False)
    weight.data = weight.data.clone()  # a private copy: no longer the shared mapping
    with pytest.raises(RuntimeError, match="mapping"):
        rl1_table.guard(model, good)
    rl1_table.STATE.clear()


def test_attach_refuses_another_tensor_left_on_meta(tmp_path):
    torch = pytest.importorskip("torch")
    import rl1_table

    (tmp_path / "ckpt").mkdir()
    (tmp_path / "shm").mkdir()
    _, total, cols = checkpoint(tmp_path / "ckpt")
    assert rl1_table.start(tmp_path / "ckpt", env(tmp_path), workers=1)
    model = torch.nn.ModuleDict({"ngram_embedding": torch.nn.Embedding(total, cols, device="meta", dtype=torch.bfloat16),
                                 "x": torch.nn.Linear(2, 2, device="meta")})
    with pytest.raises(RuntimeError, match="still on meta"):
        rl1_table.attach(None, model)
    rl1_table.STATE.clear()


def test_preload_room_with_the_shared_table():
    fq = pytest.importorskip("flash_qlora_job")
    memory = {"mem_available_gib": 1015.0}
    reserve = {"LOCAL_WORLD_SIZE": "2", "RL1_PRELOAD_RESERVE_GIB": "184.8"}
    assert not fq.preload_room(reserve, memory)["ok"]  # H4/H5: 1,034 GiB asked
    shared = fq.preload_room({**reserve, "RL1_SHARED_TABLE": "1"}, memory)
    expect = (fq.PRELOAD_NEED_SHARED_GIB * 2 + fq.PRELOAD_TABLE_GIB + 184.8) * 1.05
    assert shared["ok"] and shared["shared_table"] and shared["need_gib"] == round(expect, 1)
