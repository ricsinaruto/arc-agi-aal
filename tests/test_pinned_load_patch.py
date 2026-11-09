"""pinned_load_patch (MFU-2 B300 boot, 24 Oct): the patched pinned safetensors iterator yields the same names, order,
dtypes, shapes and bytes with ARC_PINNED_LOAD=1 as the mmap path (chunk splits, skipped weights, over-limit and
direct tensors); idempotent; an unknown file is skipped, never raised on."""

import ast
import json
import os
import re
import shutil
import time
import types
from pathlib import Path

import pytest

from arc_infra import pinned_load_patch as patch

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
PINNED = Path(os.environ.get("ARC_PINNED_VLLM", "~/arc-shared/pinned-vllm"))
pytestmark = pytest.mark.skipif(not PINNED.exists(), reason="no local copy of the pinned vLLM")
NAMES = ["safetensors_weights_iterator", "_natural_sort_key", "_arc_pinned_plan", "_arc_pinned_read",
         "_arc_pinned_iterator"]


def iterator(text, skip):
    tree = ast.parse(text)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in NAMES:
            node.decorator_list, node.returns = [], None
            for a in node.args.args + node.args.kwonlyargs:
                a.annotation = None
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_ARC_DTYPES" for t in node.targets):
            nodes.append(node)
    ns = {"torch": torch, "os": os, "json": json, "re": re, "time": time, "safe_open": safetensors_torch.safe_open,
          "load": safetensors_torch.load, "tqdm": lambda it, **kw: it, "enable_tqdm": lambda x: False,
          "_BAR_FORMAT": "", "is_offload_process": lambda: False, "_get_fs_type": lambda files: "ext4",
          "_get_checkpoints_size_bytes": lambda files: 0, "_get_available_ram_bytes": lambda: 1 << 40,
          "logger": types.SimpleNamespace(info_once=lambda *a: None, warning_once=lambda *a: None),
          "DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS": 1, "DEFAULT_SAFETENSORS_PREFETCH_BLOCK_SIZE": 1,
          "should_skip_weight": skip}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "weight_utils", "exec"), ns)
    return ns["safetensors_weights_iterator"]


def checkpoint(tmp_path):
    g = torch.Generator().manual_seed(0)
    files = []
    for i in range(3):
        tensors = {f"layers.{i}.w{j}": torch.randn(64 + j, 300, generator=g).to(torch.bfloat16) for j in range(12)}
        tensors[f"layers.{i}.scale"] = torch.randn(64, 16, generator=g).to(torch.float8_e4m3fn)
        tensors[f"layers.{i}.packed"] = torch.randint(0, 255, (128, 77), dtype=torch.uint8, generator=g)
        tensors[f"layers.{i}.flag"] = torch.tensor([True, False, True])
        tensors[f"layers.{i}.s"] = torch.tensor(0.5)
        tensors[f"layers.{i}.big"] = torch.randn(600, 1024, generator=g)  # 2.4 MB: over a 2 MB limit
        tensors[f"layers.{i}.experts.3.odd"] = torch.randint(0, 9, (7,), dtype=torch.int64, generator=g)
        path = tmp_path / f"model-0000{i + 1}-of-00003.safetensors"
        safetensors_torch.save_file(tensors, str(path))
        files.append(str(path))
    return files[::-1]


def stream(fn, files):
    return [(n, t.dtype, tuple(t.shape), t.contiguous().view(torch.uint8).reshape(-1).clone()
             if t.dim() else t.reshape(1).view(torch.uint8).clone()) for n, t in fn(files, False)]


@pytest.mark.parametrize("chunk_mb", ["1", "64"])
def test_same_stream_on_and_off(tmp_path, monkeypatch, capsys, chunk_mb):
    original = (PINNED / patch.RELATIVE).read_text()
    text = patch.patched_source(original.encode()).decode()
    skip = lambda name, ids: ".experts.3." in name  # noqa: E731 - a skipped weight is never read
    files = checkpoint(tmp_path)
    monkeypatch.delenv(patch.ENV, raising=False)
    ref = stream(iterator(original, skip), files)
    off = stream(iterator(text, skip), files)
    monkeypatch.setenv(patch.ENV, "1")
    monkeypatch.setenv("ARC_PINNED_LOAD_CHUNK_MB", chunk_mb)
    monkeypatch.setenv("ARC_PINNED_LOAD_MAX_MB", "2")
    on = stream(iterator(text, skip), files)
    assert len(ref) == 3 * 17 and not any(".experts.3." in r[0] for r in ref)
    for a, b, c in zip(ref, off, on):
        assert a[:3] == b[:3] == c[:3] and torch.equal(a[3], b[3]) and torch.equal(a[3], c[3])
    assert len(on) == len(ref)
    stats = json.loads(capsys.readouterr().out.split("ARC_PINNED_LOAD ")[1].splitlines()[0])
    assert stats["files"] == 3 and stats["direct_tensors"] == 3 and stats["chunk_tensors"] == 3 * 16


def test_apply_idempotent_and_skip(tmp_path, capsys):
    site = tmp_path / "site"
    (site / patch.RELATIVE).parent.mkdir(parents=True)
    shutil.copy(PINNED / patch.RELATIVE, site / patch.RELATIVE)
    identity = patch.apply(site, tmp_path / "out")
    assert json.loads((tmp_path / "out/pinned-load.json").read_text()) == identity
    assert patch.apply(site, tmp_path / "out") == identity
    (site / patch.RELATIVE).write_text("x = 1\n")
    assert patch.apply(site, tmp_path / "out2") is None
    assert "ARC_PINNED_LOAD_PATCH_SKIPPED" in capsys.readouterr().out
