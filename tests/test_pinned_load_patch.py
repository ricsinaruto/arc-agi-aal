"""arc_infra.pinned_load_patch: ARC_DRAFT_LOAD_FILTER (the MTP draft reads only the names its own remap keeps; exact)
and ARC_PINNED_LOAD_FIXED (one chunk size class), stage H 26 Oct."""
import shutil
import sys
import types
from pathlib import Path

import pytest

from arc_infra import pinned_load_patch as P

WEIGHT_UTILS = '''import json
import os
import time

import torch
from safetensors import safe_open


def tqdm(x, **kwargs):
    return x


def should_skip_weight(name, local_expert_ids):
    return False


def safetensors_weights_iterator(sorted_files, local_expert_ids=None, safetensors_load_strategy=None):
    leftover_state_dict: dict[str, torch.Tensor] = {}
    for st_file in tqdm(
        sorted_files):
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)
'''
DEFAULT_LOADER = '''import os


class DefaultModelLoader:
    def __init__(self, files):
        self.files = files

    def _init_ep_weight_filter(self, model_config):
        self.local_expert_ids = None

    def get_all_weights(self, model_config, model):
        from vllm.model_executor.model_loader import weight_utils
        return weight_utils.safetensors_weights_iterator(self.files, None, None)

    def load_weights(self, model, model_config):
        self._init_ep_weight_filter(model_config)

        loaded_weights = model.load_weights(self.get_all_weights(model_config, model))
        return loaded_weights
'''
PINNED = Path("~/arc-shared/pinned-vllm")


def site(tmp_path):
    root = tmp_path / "site"
    (root / "vllm/model_executor/model_loader").mkdir(parents=True)
    (root / P.RELATIVE).write_text(WEIGHT_UTILS)
    (root / P.LOADER_RELATIVE).write_text(DEFAULT_LOADER)
    return root


def test_loader_edit_only_with_the_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_DRAFT_LOAD_FILTER", raising=False)
    s = site(tmp_path)
    ident = P.apply(s, tmp_path / "out")
    assert "draft_load_filter" not in ident and "ARC draft_load_filter" not in (s / P.LOADER_RELATIVE).read_text()
    s2 = site(tmp_path / "b")
    monkeypatch.setenv("ARC_DRAFT_LOAD_FILTER", "1")
    ident = P.apply(s2, tmp_path / "out2")
    assert "draft_load_filter" in ident and "ARC draft_load_filter" in (s2 / P.LOADER_RELATIVE).read_text()


def test_loader_anchor_must_be_unique():
    with pytest.raises(RuntimeError, match="anchor"):
        P.patched_loader((DEFAULT_LOADER + DEFAULT_LOADER).encode())


@pytest.mark.skipif(not (PINNED / P.RELATIVE).is_file(), reason="pinned vLLM tree not on this machine")
def test_patches_compile_on_the_pinned_runtime_files(tmp_path, monkeypatch):
    root = tmp_path / "site"
    for rel in (P.RELATIVE, P.LOADER_RELATIVE):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PINNED / rel, root / rel)
    monkeypatch.setenv("ARC_DRAFT_LOAD_FILTER", "1")
    ident = P.apply(root, tmp_path / "out")
    assert ident["original_sha256"].startswith("25bb3393") and ident["draft_load_filter"]["original_sha256"]


def load(tmp_path, monkeypatch, env):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    s = site(tmp_path)
    P.apply(s, tmp_path / "out")
    tensors = {"model.language_model.layers.0.w": torch.arange(6, dtype=torch.float32).reshape(2, 3),
               "model.language_model.mtp.layers.0.w": torch.ones(4, dtype=torch.bfloat16),
               "model.language_model.embed_tokens.weight": torch.full((3, 2), 2.0),
               "lm_head.weight": torch.full((2, 2), 3.0)}
    files = []
    for i, (name, t) in enumerate(tensors.items()):
        files.append(str(tmp_path / f"m-{i}.safetensors"))
        st.save_file({name: t}, files[-1])
    pkg = "vllm.model_executor.model_loader"
    wu = types.ModuleType(pkg + ".weight_utils")
    exec(compile((s / P.RELATIVE).read_text(), "weight_utils.py", "exec"), wu.__dict__)
    loader_pkg = types.ModuleType(pkg)
    loader_pkg.weight_utils = wu
    for name, mod in (("vllm", types.ModuleType("vllm")), ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
                      (pkg, loader_pkg), (pkg + ".weight_utils", wu)):
        monkeypatch.setitem(sys.modules, name, mod)
    dl = types.ModuleType(pkg + ".default_loader")
    exec(compile((s / P.LOADER_RELATIVE).read_text(), "default_loader.py", "exec"), dl.__dict__)

    def remap(name):  # the draft's own remap (shape of qwen3_8_flash_next/nvidia/mtp.py)
        name = name.removeprefix("model.language_model.")
        return name if name.startswith(("mtp.", "embed_tokens.", "lm_head.")) else None

    mtp_mod = types.ModuleType("fake_mtp")
    mtp_mod._remap_mtp_weight_name = remap
    monkeypatch.setitem(sys.modules, "fake_mtp", mtp_mod)

    class Model:
        def load_weights(self, weights):
            self.seen = {n: t.clone() for n, t in weights}
            return set(self.seen)

    class FakeMTP(Model):
        pass

    FakeMTP.__module__ = "fake_mtp"
    return tensors, files, dl.DefaultModelLoader(files), Model, FakeMTP


def test_draft_reads_only_its_names_and_the_target_reads_all(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    tensors, files, loader, Model, FakeMTP = load(tmp_path, monkeypatch, {"ARC_PINNED_LOAD": "1",
                                                                          "ARC_DRAFT_LOAD_FILTER": "1"})
    draft = FakeMTP()
    loader.load_weights(draft, None)
    assert set(draft.seen) == {"model.language_model.mtp.layers.0.w", "model.language_model.embed_tokens.weight",
                               "lm_head.weight"}
    for name, t in draft.seen.items():
        assert torch.equal(t, tensors[name]) and t.dtype == tensors[name].dtype
    target = Model()
    loader.load_weights(target, None)  # the filter is set per load: the target after the draft reads everything
    assert set(target.seen) == set(tensors)
    for name, t in target.seen.items():
        assert torch.equal(t, tensors[name])


def test_without_the_flag_the_draft_reads_everything(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.delenv("ARC_DRAFT_LOAD_FILTER", raising=False)
    tensors, files, loader, Model, FakeMTP = load(tmp_path, monkeypatch, {"ARC_PINNED_LOAD": "1"})
    draft = FakeMTP()
    loader.load_weights(draft, None)
    assert set(draft.seen) == set(tensors)


def test_fixed_chunks_give_the_same_bytes(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    tensors, files, loader, Model, FakeMTP = load(tmp_path, monkeypatch, {"ARC_PINNED_LOAD": "1",
                                                                          "ARC_PINNED_LOAD_FIXED": "1"})
    target = Model()
    loader.load_weights(target, None)
    assert set(target.seen) == set(tensors)
    for name, t in target.seen.items():
        assert torch.equal(t, tensors[name])
