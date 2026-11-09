"""MFU-1 item 2: rl1_cast (one BF16 copy per fp32 Linear input, PEFT round trip skipped) is bitwise equal to autocast."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import rl1_cast  # noqa: E402


@pytest.fixture(autouse=True)
def off():
    yield
    rl1_cast.install(False)


class Block(torch.nn.Module):
    """An fp32 activation read by three frozen BF16 Linears and a trainable fp32 one, plus a non-Linear use."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.q, self.k, self.v = (torch.nn.Linear(16, 16, bias=False).to(torch.bfloat16) for _ in range(3))
        self.a = torch.nn.Linear(16, 4, bias=False)
        for m in (self.q, self.k, self.v, self.a):
            m.weight.data = (torch.randn(m.weight.shape, generator=g) * 0.3).to(m.weight.dtype)
        for m in (self.q, self.k, self.v):
            m.weight.requires_grad_(False)

    def forward(self, x):
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)  # fp32, read by four Linears
        return (self.q(h) * self.k(h)).float().sum(-1) + self.v(h).float().mean(-1) + self.a(h).float().sum(-1) + h.sum(-1)


def run(on, checkpoint=False):
    rl1_cast.install(on)
    block = Block()
    x = torch.randn(2, 9, 16, generator=torch.Generator().manual_seed(1), requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False) if checkpoint else block(x)
        loss = (y * torch.linspace(-1, 1, 9)).sum()
    loss.backward()
    return y.detach(), x.grad.clone(), block.a.weight.grad.clone()


@pytest.mark.parametrize("checkpoint", [False, True])
def test_reuse_is_bitwise_equal(checkpoint):
    base = run(False, checkpoint)
    rl1_cast.CAST.update(made=0, reused=0)
    got = run(True, checkpoint)
    assert all(torch.equal(a, b) for a, b in zip(base, got))
    assert rl1_cast.CAST["reused"] >= 3 * (2 if checkpoint else 1)


def test_cache_respects_in_place_changes_and_no_grad():
    rl1_cast.install(True)
    x = torch.randn(4, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
        a = rl1_cast.bf16(x)
        x.mul_(2)
        b = rl1_cast.bf16(x)
    assert torch.equal(b, (x).to(torch.bfloat16)) and not torch.equal(a, b)


def test_tiny_qwen4_with_lora_is_bitwise_equal():
    modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    peft = pytest.importorskip("peft")
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    config = Qwen4ExpTextConfig(
        vocab_size=320, hidden_size=32, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, partial_rotary_factor=0.25,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        moe_intermediate_size=16, shared_expert_intermediate_size=16, num_experts=8, num_experts_per_tok=2,
        hc_count=4, hc_lowrank=8, ple_embed_dim=32, ple_layer_ids=[2], ngram_vocab_size_base=101,
        split_ngram_parts=4, indexer_n_heads=2, indexer_head_dim=8, indexer_kv_heads=1, indexer_budget=8,
        indexer_compress_ratio=4, eos_token_id=1, bos_token_id=1, pad_token_id=None)
    targets = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a",
               "linear_attn.out_proj", "shared_expert.up_proj", "shared_expert.down_proj", "shared_expert.gate_proj",
               "q_proj", "k_proj", "v_proj", "o_proj"]

    def go(stream, on):
        torch.manual_seed(0)
        model = modeling.Qwen4ExpForCausalLM(config).to(torch.bfloat16)
        model = peft.inject_adapter_in_model(peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                                             target_modules=targets), model)
        g = torch.Generator().manual_seed(1)
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.data = torch.randn(p.shape, generator=g) * 0.1
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
                if stream == "fp32":  # an fp32 residual stream, as the trainer's hyper path keeps
                    p.data = p.data.float()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        rl1_cast.install(on, model)
        ids = torch.randint(2, 320, (1, 40), generator=torch.Generator().manual_seed(2))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(input_ids=ids, use_cache=False).logits
            loss = logits.float().log_softmax(-1)[0, torch.arange(39), ids[0, 1:]].mean()
        loss.backward()
        return logits.detach(), loss.detach(), {n: p.grad for n, p in model.named_parameters() if p.grad is not None}

    for stream in ("bf16", "fp32"):
        a, b = go(stream, False), go(stream, True)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert a[2].keys() == b[2].keys() and len(a[2]) > 20
        assert all(torch.equal(a[2][k], b[2][k]) for k in a[2]), stream
    assert rl1_cast.CAST["reused"] > 0 and rl1_cast.CAST.get("peft_skipped", 0) > 0
