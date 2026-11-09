"""flash_qlora_job.cast_expert_inputs: live in every RL run to 26 Oct 16:05 (48 hooks on Flash); d4727e4a7 dropped it
and the trainer's forward moved 0.023 nats from T1a's refs (T1c-v4). Restored: the loaded row reports the hooks, one
per BF16 routed-experts module of a Flash-shaped model (48 layers), and the experts see BF16 inputs."""
import ast
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))


def model(layers):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    lin, full = "linear_attention", "full_attention"
    config = Qwen4ExpTextConfig(
        vocab_size=96, hidden_size=32, num_hidden_layers=layers, layer_types=([lin, lin, lin, full] * layers)[:layers],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, partial_rotary_factor=0.25,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        moe_intermediate_size=16, shared_expert_intermediate_size=16, num_experts=8, num_experts_per_tok=2,
        hc_count=4, hc_lowrank=8, ple_embed_dim=32, ple_layer_ids=[2], ngram_vocab_size_base=101,
        split_ngram_parts=4, indexer_n_heads=2, indexer_head_dim=8, indexer_kv_heads=1, indexer_budget=8,
        indexer_compress_ratio=4, eos_token_id=1, bos_token_id=1, pad_token_id=None, output_gate_type="sigmoid")
    torch.manual_seed(0)
    return modeling.Qwen4ExpForCausalLM(config)


def test_flash_shaped_model_gets_48_expert_casts():
    import flash_qlora_job as fq

    with torch.device("meta"):
        m = model(48).to(torch.bfloat16)
    assert fq.cast_expert_inputs(m) == 48


def test_experts_see_bf16_inputs_and_packed_experts_are_left_alone():
    import flash_qlora_job as fq

    m = model(4).to(torch.bfloat16)
    assert fq.cast_expert_inputs(m) == 4
    experts = next(mod for name, mod in m.named_modules() if name.endswith("mlp.experts"))
    hooks = [h for k, h in experts._forward_pre_hooks.items() if k in experts._forward_pre_hooks_with_kwargs]
    assert len(hooks) == 1
    hidden, index, weights = torch.randn(5, 32), torch.zeros(5, 2, dtype=torch.long), torch.rand(5, 2)
    args, kwargs = hooks[0](experts, (hidden, index, weights), {})
    assert args[0].dtype == torch.bfloat16 and args[1] is index and args[2] is weights  # routing passes unchanged
    args, kwargs = hooks[0](experts, (), {"hidden_states": hidden, "top_k_weights": weights})
    assert kwargs["hidden_states"].dtype == torch.bfloat16 and kwargs["top_k_weights"] is weights

    packed = model(4).to(torch.bfloat16)
    for name, module in packed.named_modules():
        if name.endswith("mlp.experts"):
            module.register_buffer("packed_weight", torch.zeros(4, dtype=torch.uint8))
            module.packed = torch.nn.Parameter(torch.zeros(4, dtype=torch.uint8), requires_grad=False)
    assert fq.cast_expert_inputs(packed) == 0


def test_load_reports_expert_casts():
    """The loaded row carries ``expert_casts`` from cast_expert_inputs (the T1c-v4 diagnosis read it there)."""
    tree = ast.parse((ROOT / "scripts/posttrain/flash_qlora_job.py").read_text())
    load = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load")
    calls = [c for c in ast.walk(load) if isinstance(c, ast.Call)]
    assert any(getattr(c.func, "id", "") == "cast_expert_inputs" for c in calls)
    event = [c for c in calls if getattr(c.func, "attr", "") == "event" and c.args
             and getattr(c.args[0], "value", None) == "loaded"]
    assert event and any(k.arg == "expert_casts" for k in event[0].keywords)
