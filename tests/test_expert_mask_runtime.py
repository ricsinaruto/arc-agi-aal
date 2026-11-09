import json
import sys
import types

import pytest


def test_mtp_draft_router_passes_through_unmasked(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    worker = types.ModuleType("vllm.v1.worker.worker_base")
    worker.WorkerWrapperBase = type("WorkerWrapperBase", (), {})
    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.worker_base", worker)
    from arc_infra import expert_mask_runtime as R

    settings = tmp_path / "arc_expert_mask.json"
    settings.write_text(json.dumps({"experts": 8, "pruned": {"0": [1, 2], "1": []}, "label": "t"}))
    monkeypatch.setattr(R, "SETTINGS", str(settings))
    monkeypatch.setattr(R, "_STATE", None)
    x = torch.zeros(4, 8)
    assert R.on_router_logits(types.SimpleNamespace(layer_id=0), x)[:, 1].eq(R.MASK_VALUE).all()
    draft = types.SimpleNamespace(layer_id=48, layer_name="mtp.layers.48.mlp.experts")  # beyond the mask's rows
    assert R.on_router_logits(draft, x) is x and R.on_router_logits(draft, x) is x
    with pytest.raises(RuntimeError, match="no row"):
        R.on_router_logits(types.SimpleNamespace(layer_id=5), x)


def test_load_stacks_name_the_slow_consumer(monkeypatch, capsys):
    pytest.importorskip("torch")
    import time

    from arc_infra import expert_mask_runtime as R

    def weights():  # the read side: fast
        for i in range(30):
            yield f"w{i}", i

    def slow_consumer(item):  # the loader's per-tensor work: where rest_s goes
        end = time.perf_counter() + 0.01
        while time.perf_counter() < end:
            pass

    monkeypatch.setenv(R.STACKS_ENV, "1")
    monkeypatch.setitem(R.PROFILE, "loads", [])
    for item in R._profiled(weights, "fake")():
        slow_consumer(item)
    row = json.loads(capsys.readouterr().out.split("ARC_LOAD_PROFILE ", 1)[1])
    assert [n for n, _ in row["stacks"]["self"]][0].endswith(":slow_consumer") and row["stacks"]["samples"] > 10
    monkeypatch.delenv(R.STACKS_ENV)
    for item in R._profiled(weights, "fake")():
        pass
    assert "stacks" not in json.loads(capsys.readouterr().out.split("ARC_LOAD_PROFILE ", 1)[1])
