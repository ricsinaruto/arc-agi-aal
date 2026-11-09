"""arc_infra.lora_pin_patch: ARC_LORA_NO_PIN keeps an adapter's host copy pageable (v6: 115 s of per-tensor pinning)."""
import os

import pytest

from arc_infra import lora_pin_patch as P

LORA_MODEL = 'PIN_MEMORY = True\n\ndef f(device):\n    if True:\n        pin_memory = str(device) == "cpu" and PIN_MEMORY\n    return pin_memory\n'
MODEL_MANAGER = ('PIN_MEMORY = True\n# compact overlay edit above\n\ndef g(lora_device):\n    if True:\n'
                 '        pin_memory = str(lora_device) == "cpu" and PIN_MEMORY\n    return pin_memory\n')


def site(tmp_path, lora_model=LORA_MODEL, model_manager=MODEL_MANAGER):
    (tmp_path / "vllm/lora").mkdir(parents=True)
    (tmp_path / "vllm/lora/lora_model.py").write_text(lora_model)
    (tmp_path / "vllm/lora/model_manager.py").write_text(model_manager)
    return tmp_path


def run(path, fn, arg):
    ns = {}
    exec(compile(path.read_text(), str(path), "exec"), ns)
    return ns[fn](arg)


def test_both_files_gated_on_the_env(tmp_path, monkeypatch):
    s = site(tmp_path)
    ident = P.apply(s, tmp_path / "out")
    assert set(ident) == {"vllm/lora/lora_model.py", "vllm/lora/model_manager.py"}
    for rel, fn in (("vllm/lora/lora_model.py", "f"), ("vllm/lora/model_manager.py", "g")):
        monkeypatch.delenv(P.ENV, raising=False)
        assert run(s / rel, fn, "cpu") is True  # unchanged without the switch
        monkeypatch.setenv(P.ENV, "1")
        assert run(s / rel, fn, "cpu") is False
    assert P.apply(s, tmp_path / "out")["vllm/lora/lora_model.py"]["already"]  # idempotent on a restart


def test_unrecognized_file_raises(tmp_path):
    s = site(tmp_path, model_manager="pin_memory = PIN_MEMORY\n")
    with pytest.raises(RuntimeError, match="anchor"):
        P.apply(s, tmp_path / "out")
