"""l4 arm: the custom all-reduce switch (arc2.l4_comm) and its l4_serving options."""
import json
import types

import pytest

from arc2 import l4_comm, l4_serving


def test_install_patches_each_class_once():
    class Base:
        @classmethod
        def is_fully_connected(cls, ids):
            raise NotImplementedError

    class Nvml(Base):
        @classmethod
        def is_fully_connected(cls, ids):
            return False

    module = types.SimpleNamespace(CudaPlatformBase=Base, NvmlCudaPlatform=Nvml)
    assert l4_comm.install(module) == ["CudaPlatformBase", "NvmlCudaPlatform"]
    assert Nvml.is_fully_connected([0, 1, 2, 3]) is True and Base.is_fully_connected([0, 1]) is True
    assert l4_comm.install(module) == []


def test_register_needs_env(monkeypatch, capsys):
    monkeypatch.delenv(l4_comm.ENV, raising=False)
    assert l4_comm.register() is None and capsys.readouterr().out == ""


def _study():
    return {"name": "s", "server": {"max_num_batched_tokens": 4096}, "trials": [{"tasks": ["t"] * 64, "split": "e", "arms": []}],
            "cache_experiment": {"prefix_cache": {}}, "arc2": {}}


def test_options_reach_l4x4_and_plugin():
    st = l4_serving.l4_study(_study(), streams=32, custom_ar=True, nccl_p2p_level="PHB")
    x = st["arc2"]["l4x4"]
    assert x["custom_ar"] is True and x["nccl_p2p_level"] == "PHB"
    files = l4_serving.plugin_files({"custom_ar": True})
    assert "arc2.l4_comm:register" in files["entry_points.txt"]
    with pytest.raises(ValueError):
        l4_serving.l4_study(_study(), streams=32, nccl_p2p_level="LOC")
