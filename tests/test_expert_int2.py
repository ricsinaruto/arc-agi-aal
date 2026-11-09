"""int2 experts inside the int3 overlay (7 October screen): values on -2..1, still a valid int3 shard."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from arena import expert_3bit_math as tm  # noqa: E402
from arena import expert_3bit_overlay as ov  # noqa: E402
from arena import expert_quant_math as qm  # noqa: E402


def test_int2_experts_are_exact_int3_tensors(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import save_file

    gen = torch.Generator().manual_seed(0)
    tensors = {"other.weight": torch.randn(4, 4, generator=gen)}
    for e in range(2):
        for proj, shape in (("gate_proj", (ov.INTER, 64)), ("up_proj", (ov.INTER, 64)), ("down_proj", (32, ov.INTER))):
            w = torch.randn(1, *shape, generator=gen)
            s2 = tm.scale2_of(w, "int3")
            q, sc = tm.rtn(w, s2, "int3", 16)
            packed, block = tm.to_nvfp4(q, sc, s2, 16)
            prefix = f"model.language_model.layers.7.mlp.experts.{e}.{proj}."
            tensors[prefix + "weight"] = packed[0]
            tensors[prefix + "weight_scale"] = block[0]
            tensors[prefix + "weight_scale_2"] = s2.reshape(())
        # gate and up share one scale_2 in the served fused tensor
        tensors[f"model.language_model.layers.7.mlp.experts.{e}.up_proj.weight_scale_2"] = tensors[
            f"model.language_model.layers.7.mlp.experts.{e}.gate_proj.weight_scale_2"].clone()
    source, out = str(tmp_path / "source.safetensors"), str(tmp_path / "out.safetensors")
    save_file(tensors, source, metadata={"format": "pt"})
    stats = ov.convert_shard(torch, source, out, 16, "cpu", int2={"7": [1]})
    assert stats["experts"] == 2 and stats["int2_experts"] == 1
    assert stats["requant_rel_error"]["w2_int2"] > stats["requant_rel_error"]["w2_int3_of_int2_experts"]
    rel = ov.verify_shard(torch, source, out, 0.7)
    assert 0 < rel["w2"] < 0.7
    with safe_open(out, framework="pt") as fh:
        for e, top in ((0, 4), (1, 2)):
            packed = fh.get_tensor(f"model.language_model.layers.7.mlp.experts.{e}.down_proj.weight").view(torch.uint8)
            values = qm.fp4_values(torch.stack([packed & 0xF, packed >> 4], -1).reshape(packed.shape[0], -1))
            assert float(values.abs().max()) == top
        assert torch.equal(fh.get_tensor("other.weight"), tensors["other.weight"])
