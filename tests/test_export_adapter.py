import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("modal")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import export_adapter  # noqa: E402


def write_ref(folder, state):
    from safetensors.torch import save_file

    folder.mkdir(parents=True)
    save_file(state, str(folder / "adapter_model.safetensors"))
    (folder / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "inference_mode": True}))


def test_export_equals_trainer_save(tmp_path):
    """bf16(master) under the PEFT names = what save_adapter writes for params loaded from that checkpoint."""
    from safetensors.torch import load_file, save_file

    gen = torch.Generator().manual_seed(0)
    names = ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
             "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
             "base_model.model.model.layers.0.mlp.experts.lora_A.default.weight"]
    shapes = [(2, 8), (8, 2), (6, 2, 8)]
    master = {n: torch.randn(*s, generator=gen) * 0.01 for n, s in zip(names, shapes)}
    (tmp_path / "ckpt").mkdir()
    save_file(master, str(tmp_path / "ckpt" / "master.safetensors"))
    # the trainer: params (bf16) <- master.to(bf16) (op_load_checkpoint), save_adapter -> bf16 under peft names
    trainer = {n.replace(".default", ""): master[n].to(torch.bfloat16) for n in names}
    write_ref(tmp_path / "ref", trainer)
    row = export_adapter.export(tmp_path / "ckpt", tmp_path / "out", tmp_path / "ref")
    got = load_file(str(tmp_path / "out" / "adapter_model.safetensors"))
    assert set(got) == set(trainer) and all(torch.equal(got[k], trainer[k]) for k in got)
    assert row["tensors"] == 3 and json.loads((tmp_path / "out" / "adapter_config.json").read_text())["r"] == 2


def test_export_refuses_layout_mismatch(tmp_path):
    from safetensors.torch import save_file

    master = {"m.q_proj.lora_A.default.weight": torch.zeros(2, 8)}
    (tmp_path / "ckpt").mkdir()
    save_file(master, str(tmp_path / "ckpt" / "master.safetensors"))
    write_ref(tmp_path / "ref", {"m.q_proj.lora_A.weight": torch.zeros(4, 8, dtype=torch.bfloat16),
                                 "m.k_proj.lora_A.weight": torch.zeros(2, 8, dtype=torch.bfloat16)})
    with pytest.raises(ValueError, match="missing.*shape"):
        export_adapter.export(tmp_path / "ckpt", tmp_path / "out", tmp_path / "ref")
    with pytest.raises(ValueError, match="not a LoRA param"):
        export_adapter.served_state({"m.q_proj.weight": torch.zeros(1)})
