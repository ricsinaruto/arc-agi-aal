"""MFU stage M: chunked LM head and checkpoint-boundary offload against the trainer's current window pass (CPU, fp32)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import rl1_memory  # noqa: E402
import rl1_trainer  # noqa: E402
import rl_job  # noqa: E402

V, H = 320, 32


@pytest.fixture(autouse=True)
def flags():
    yield
    rl1_memory.HEAD.update(chunked=False, capture=False, chunk=1024)
    rl1_memory.TAIL.update(mode="off", cpu=False, min_bytes=1 << 20, width=None)


def head_inputs(n=37, seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(n, H, generator=g)
    tokens = torch.randint(0, V, (n,), generator=g)
    support = torch.full((n, 6), -1)
    for i in range(n):
        if i % 3:  # a sampling set (some miss the token: invalid rows), else the whole vocabulary
            ids = torch.randperm(V, generator=g)[: 3 + i % 4]
            if i % 5:
                ids[0] = tokens[i]
            support[i, : len(ids)] = ids
    vocab = [torch.randint(0, V, (n, 20), generator=g)]
    return hidden, tokens, support, vocab


def test_chunked_head_matches_the_logits_path():
    hidden, tokens, support, vocab = head_inputs()
    head = torch.nn.Linear(H, V, bias=False)
    torch.nn.init.normal_(head.weight, 0, 0.3)
    weights = torch.linspace(-1, 1, len(tokens))

    a = hidden.clone().requires_grad_(True)
    logits = head(a)
    actor, valid = rl_job.actor_logprobs(logits, tokens, support, 0.7)
    want = {"raw": rl_job.raw_logprobs(logits.detach(), tokens), "hit": logits.detach().argmax(-1) == tokens,
            "entropy": rl1_trainer.token_entropy(logits.detach(), valid, 0.7),
            "dist": rl1_trainer.vocab_top(logits.detach(), at_ids=vocab)}
    (-(weights[valid] * actor[valid]).sum()).backward()
    grad_a, grad_w = a.grad.clone(), head.weight.grad.clone()
    head.weight.grad = None

    b = hidden.clone().requires_grad_(True)
    seen = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: seen.append(tuple(t.shape)) or t, lambda t: t):
        got = rl1_memory.chunked_head(head.forward, b, tokens, support, 0.7, hit=True, vocab=vocab, chunk=8)
        loss = -(weights[got["valid"]] * got["actor"][got["valid"]]).sum()
    loss.backward()
    assert not any(V in s for s in seen)  # nothing of vocabulary width is kept for the backward
    assert torch.equal(got["valid"], valid) and bool((~valid).any())
    assert torch.allclose(got["actor"], actor, atol=1e-5, rtol=0)
    assert torch.allclose(got["raw"], want["raw"], atol=1e-5, rtol=0)
    assert torch.equal(got["hit"], want["hit"])
    assert got["entropy"][1] == want["entropy"][1] and abs(got["entropy"][0] - want["entropy"][0]) < 1e-4
    assert torch.equal(got["dist"]["ids"], want["dist"]["ids"])
    assert torch.allclose(got["dist"]["logp"], want["dist"]["logp"], atol=1e-5, rtol=0)
    assert torch.allclose(got["dist"]["at"][0], want["dist"]["at"][0], atol=1e-5, rtol=0)
    assert torch.allclose(b.grad, grad_a, atol=1e-5, rtol=0)
    assert torch.allclose(head.weight.grad, grad_w, atol=1e-5, rtol=0)


def test_chunked_head_without_grad_and_empty():
    hidden, tokens, support, _ = head_inputs(n=5)
    head = torch.nn.Linear(H, V, bias=False)
    with torch.no_grad():
        got = rl1_memory.chunked_head(head.forward, hidden, tokens, support, 1.0, chunk=2)
        actor, valid = rl_job.actor_logprobs(head(hidden), tokens, support, 1.0)
    assert torch.allclose(got["actor"], actor, atol=1e-6) and got["hit"] is None and got["dist"] is None
    empty = rl1_memory.chunked_head(head.forward, hidden[:0], tokens[:0], support[:0], 1.0)
    assert empty["actor"].numel() == 0 and empty["raw"].numel() == 0 and empty["entropy"] == (0.0, 0)


# ---- the trainer's window pass on a tiny qwen4_exp


def tiny_model():
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


@pytest.fixture
def trainer(monkeypatch):
    model = tiny_model()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(rl1_trainer.fq, "positions_and_images", lambda model, inputs, **kw: (lambda start, stop: {}))
    t = rl1_trainer.Trainer.__new__(rl1_trainer.Trainer)
    t.model, t.args, t.policy = model, SimpleNamespace(tis_cap=2.0), {}
    rl1_trainer.batch_head_hook(model)  # as Trainer.__init__
    t.head = rl1_memory.head_capture(model)
    t.prefix_cache = lambda inputs, ids, cut, piece, shared: (None, False, None, None)
    return t


def window(kind, seed=0, length=48):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, V, (1, length), generator=g)
    positions = torch.sort(torch.randperm(length - 1, generator=g)[:30] + 1).values
    n = len(positions)
    tokens = ids[0, positions]
    support = torch.full((n, 4), -1)
    for i in range(0, n, 2):  # half the rows sample from a set of 4 (one without the token)
        support[i] = torch.randint(0, V, (4,), generator=g)
        if i != 4:
            support[i, 1] = tokens[i]
    arrays = {"positions": positions, "support": support, "weights": torch.linspace(-1, 2, n),
              "reference": -torch.rand(n, generator=g) * 3, "reply": torch.arange(n) // 7}
    if kind == "opd":
        arrays.update(opd_teacher=-torch.rand(n, generator=g) * 4, opd_valid=torch.rand(n, generator=g) > 0.2,
                      opd_clip=1.0)
    return {"name": "w", "inputs": {"input_ids": ids}, "tail": length, "arrays": arrays, "temperature": 0.7,
            "argmax": True}


def run(t, made, scale=0.1, vocab=None):
    t.model.zero_grad(set_to_none=True)
    out = t.window_pass(made, scale=scale, vocab=vocab)
    grads = {n: p.grad.clone() for n, p in t.model.named_parameters() if p.grad is not None}
    return out, grads


def same(a, b, exact):
    if exact:
        return torch.equal(a, b)
    return a.shape == b.shape and torch.allclose(a.float(), b.float(), atol=1e-5, rtol=0)


def compare(o1, g1, o2, g2, exact, grads_exact=None):
    for key in ("actor", "raw", "valid", "hit"):
        assert same(o1[key], o2[key], exact or key in ("valid", "hit")), key
    assert o1["entropy"][1] == o2["entropy"][1] and abs(o1["entropy"][0] - o2["entropy"][0]) <= (0 if exact else 1e-4)
    if o1["loss"] is not None:
        assert abs(o1["loss"] - o2["loss"]) <= (0 if exact else 1e-5)
    assert set(g1) == set(g2) and len(g1) > 20
    for name in g1:
        assert same(g1[name], g2[name], exact if grads_exact is None else grads_exact), name


@pytest.mark.parametrize("kind", ["tis", "gspo", "opd"])
def test_window_pass_chunked_head_and_offload_match_the_current_path(trainer, kind):
    if kind == "gspo":
        trainer.policy = {"loss": "gspo"}
    made = window(kind)
    base_out, base_grads = run(trainer, made)
    assert rl1_memory.HEAD["calls"] >= 0 and base_out["loss"] is not None
    rl1_memory.HEAD.update(chunked=True, chunk=7)
    chunk_out, chunk_grads = run(trainer, made)
    compare(base_out, base_grads, chunk_out, chunk_grads, exact=False)
    rl1_memory.HEAD["chunked"] = False
    rl1_memory.TAIL.update(mode="offload", cpu=True, min_bytes=0)
    before = rl1_memory.TAIL["offloaded"]
    off_out, off_grads = run(trainer, made)
    assert rl1_memory.TAIL["offloaded"] - before >= 4  # the four layers' checkpoint inputs (+ the final mixer's)
    # offload is ulp-inexact in the gradients: the restored saved tensors' strides change the backward's reduction
    # order (window seed 0: 103/105 grads, |diff| <= 9e-8; seed 1: bitwise); log-probs and loss stay bit-equal.
    # A bundle that turns tail_ckpt offload on needs stride-preserving unpack and this back to exact first.
    compare(base_out, base_grads, off_out, off_grads, exact=True, grads_exact=False)
    rl1_memory.HEAD.update(chunked=True)
    both_out, both_grads = run(trainer, made)
    compare(base_out, base_grads, both_out, both_grads, exact=False)


def test_flags_off_window_pass_is_the_logits_path_bit_for_bit(trainer):
    made = window("tis", seed=3)
    vocab = [torch.randint(0, V, (30, 20))]
    out, grads = run(trainer, made, vocab=vocab)
    # the pre-stage-M window pass, inline
    arrays, ids = made["arrays"], made["inputs"]["input_ids"]
    keep = arrays["positions"] - 1
    tokens = ids[0, arrays["positions"]]
    trainer.model.zero_grad(set_to_none=True)
    logits = trainer.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=keep).logits[0]
    actor, valid = rl_job.actor_logprobs(logits, tokens, arrays["support"], made["temperature"])
    loss, _, _ = trainer.weighted_loss(arrays, actor, valid, 0.1)
    loss.backward()
    want = {"actor": actor.detach(), "raw": rl_job.raw_logprobs(logits.detach(), tokens), "valid": valid,
            "hit": logits.detach().argmax(-1) == tokens, "entropy": rl1_trainer.token_entropy(logits.detach(), valid, 0.7),
            "loss": float(loss.detach())}
    compare(want, {n: p.grad for n, p in trainer.model.named_parameters() if p.grad is not None}, out, grads, exact=True)
    dist = rl1_trainer.vocab_top(logits.detach(), at_ids=vocab)
    assert torch.equal(out["dist"]["logp"], dist["logp"]) and torch.equal(out["dist"]["at"][0], dist["at"][0])
    # chunked no-grad reads (old log-probs, scores) equal it too
    rl1_memory.HEAD["chunked"] = True
    with torch.no_grad():
        scored = trainer.window_pass(made, vocab=vocab)
    assert trainer.model.training and scored["loss"] is None
    assert torch.allclose(scored["actor"], want["actor"], atol=1e-5) and torch.equal(scored["dist"]["ids"], dist["ids"])


def test_op_paths_reads_the_flags(monkeypatch):
    monkeypatch.setattr(rl1_trainer, "COMPILE", {"on": False, "ready": None})
    monkeypatch.setitem(rl1_trainer.KERNEL, "experts", "dequant")
    rl1_trainer.op_paths(None, {"chunked_head": True, "tail_ckpt": "offload"})
    assert rl1_memory.HEAD["chunked"] and rl1_memory.TAIL["mode"] == "offload"
    rl1_trainer.op_paths(None, {})
    assert not rl1_memory.HEAD["chunked"] and rl1_memory.TAIL["mode"] == "off"
    with pytest.raises(ValueError):
        rl1_trainer.op_paths(None, {"tail_ckpt": "bogus"})


def test_batch_pass_chunked_head_matches(trainer, monkeypatch):
    rl1_trainer.selected_attention()
    monkeypatch.setitem(rl1_trainer.SPARSE, "mode", "selected")
    trainer.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    monkeypatch.setattr(rl1_trainer.fq, "positions_and_images", lambda model, inputs, **kw: (
        lambda start, stop: {"position_ids": torch.arange(start, stop).view(1, 1, -1).expand(3, 1, -1)}))
    first, second = window("tis", seed=7), window("tis", seed=8, length=40)

    def batch():
        trainer.model.zero_grad(set_to_none=True)
        outs = trainer.batch_pass([first, second], [0.1, 0.2])
        return outs, {n: p.grad.clone() for n, p in trainer.model.named_parameters() if p.grad is not None}

    base, base_grads = batch()
    rl1_memory.HEAD.update(chunked=True, chunk=5)
    got, got_grads = batch()
    for a, b in zip(base, got):
        for key in ("actor", "raw"):
            assert torch.allclose(a[key], b[key], atol=1e-5, rtol=0), key
        assert torch.equal(a["valid"], b["valid"]) and abs(a["loss"] - b["loss"]) < 1e-5
        assert a["entropy"][1] == b["entropy"][1] and abs(a["entropy"][0] - b["entropy"][0]) < 1e-4
    for name in base_grads:
        assert torch.allclose(base_grads[name], got_grads[name], atol=1e-5, rtol=0), name
