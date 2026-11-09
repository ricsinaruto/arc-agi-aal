"""Golden (d): the trainer's window pass (loss terms and parameter gradients) on a tiny CPU qwen4_exp and a fixed
window: rl1_trainer.Trainer.window_pass as the hero calls it per window (GSPO weights, references, TIS cap)."""
import sys
from types import SimpleNamespace

import pytest

import json
import os

from common import DATA, ROOT

torch = pytest.importorskip("torch")
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))
V, H = 320, 32


def tiny_model(rl1_trainer):
    modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    config = Qwen4ExpTextConfig(
        vocab_size=V, hidden_size=H, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, partial_rotary_factor=0.25,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        moe_intermediate_size=16, shared_expert_intermediate_size=16, num_experts=8, num_experts_per_tok=2,
        hc_count=4, hc_lowrank=8, ple_embed_dim=32, ple_layer_ids=[2], ngram_vocab_size_base=101,
        split_ngram_parts=4, indexer_n_heads=2, indexer_head_dim=8, indexer_kv_heads=1, indexer_budget=8,
        indexer_compress_ratio=4, eos_token_id=1, bos_token_id=1, pad_token_id=None)
    torch.manual_seed(0)
    model = modeling.Qwen4ExpForCausalLM(config).float()
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                param.normal_(0, 0.08)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rl1_trainer.set_checkpointing(model, "full")
    model.train()
    return model


def window(end, tail):
    ids = torch.randint(2, V, (1, 320), generator=torch.Generator().manual_seed(7))[:, :end]
    g = torch.Generator().manual_seed(end)
    positions = torch.sort(torch.randperm(tail - 1, generator=g)[:24] + end - tail + 1).values
    n = len(positions)
    arrays = {"positions": positions, "support": torch.full((n, 4), -1), "weights": torch.linspace(-1, 2, n),
              "reference": -torch.rand(n, generator=g) * 3, "reply": torch.arange(n) // 7}
    return {"name": f"w{end}", "inputs": {"input_ids": ids}, "tail": tail, "arrays": arrays, "temperature": 0.7,
            "argmax": False}


def digest(t):
    """Summary floats of a tensor (sum, |x| sum, max |x|, norm): compared with a tolerance, since host-dependent CPU
    reduction order moves values by ~1e-8 (S4: a 6-decimal rounding flipped on 1 of 4 base reruns)."""
    t = t.detach().double().flatten()
    return [float(t.sum()), float(t.abs().sum()), float(t.abs().max()) if t.numel() else 0.0, float(t.norm())]


def close(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), (path, sorted(set(a) ^ set(b)))
        for k in a:
            close(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(float(a) - float(b)) <= 1e-7 + 1e-5 * abs(float(b)), (path, a, b)
    else:
        assert a == b, (path, a, b)


def test_window_pass_loss_and_grads(monkeypatch):
    import rl1_memory
    import rl1_trainer

    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(rl1_trainer.fq, "positions_and_images", lambda model, inputs, **kw: (lambda start, stop: {}))
    t = rl1_trainer.Trainer.__new__(rl1_trainer.Trainer)
    t.model, t.args, t.policy = tiny_model(rl1_trainer), SimpleNamespace(tis_cap=2.0), {}
    t.head = rl1_memory.head_capture(t.model)
    t.prefix_chunk = t.path_chunk = 64
    t.snapshots, t.snap_keep = True, 3
    out = {}
    for end, tail in ((200, 80), (320, 120)):
        t.model.zero_grad(set_to_none=True)
        res = t.window_pass(window(end, tail), scale=0.1, shared=None)
        grads = {n: digest(p.grad) for n, p in t.model.named_parameters() if p.grad is not None}
        res = {k: v for k, v in res.items() if not k.endswith("_seconds") and k != "host"}  # timings
        out[f"{end}/{tail}"] = {"out": {k: (digest(v) if torch.is_tensor(v) else v) for k, v in sorted(res.items())},
                                "grads": grads}
    out = json.loads(json.dumps(out, sort_keys=True))
    path = DATA / "trainer.json"
    if os.environ.get("GOLDEN_RECORD") == "1" or not path.exists():
        print("GOLDEN-RECORD trainer " + json.dumps(out, sort_keys=True))
        if not path.exists():
            pytest.fail("golden trainer: no recorded data")
        path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    close(out, json.loads(path.read_text()))
