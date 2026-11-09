"""rl1_cp (RL C1, context parallelism): a tiny Flash-shaped model (GDN, QSA attention, PLE, MoE, hyper-connections;
the real transformers qwen4_exp classes on CPU, the trainer's qsa_select + SelectedAttention) split over 1-3 gloo ranks
with uneven and empty shards, its context filled by the sharded prefill: hidden states and summed LoRA gradients
against the trainer's single-rank path (prefix cache, then the segment) within fp32 tolerance."""
import contextlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
pytest.importorskip("peft")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))

from arc_infra.runops.multirank import run_ranks  # noqa: E402

V, H, N = 96, 32, 190  # vocab, hidden, window tokens (prefix + segment)
TARGETS = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a",
           "linear_attn.out_proj", "shared_expert.up_proj", "shared_expert.down_proj", "shared_expert.gate_proj",
           "q_proj", "k_proj", "v_proj", "o_proj"]  # flash_qlora_job.TARGET_MODULES


def build(ckpt=False):
    """(base model, peft model): 6 layers (GDN x4, QSA attention at 2 and 5, PLE on layer 1), fp32, LoRA B non-zero."""
    import rl1_trainer as rt
    from peft import LoraConfig, get_peft_model
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    torch.set_num_threads(1)
    rt.selected_attention()  # the trainer's single-rank path: qsa_select + SelectedAttention (mode set by trainer_path)
    lin, full = "linear_attention", "full_attention"
    config = Qwen4ExpTextConfig(
        vocab_size=V, hidden_size=H, num_hidden_layers=6, layer_types=[lin, lin, full, lin, lin, full],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, partial_rotary_factor=0.25,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        moe_intermediate_size=16, shared_expert_intermediate_size=16, num_experts=8, num_experts_per_tok=2,
        hc_count=4, hc_lowrank=8, ple_embed_dim=32, ple_layer_ids=[2], ngram_vocab_size_base=101,
        split_ngram_parts=4, indexer_n_heads=2, indexer_head_dim=8, indexer_kv_heads=1, indexer_budget=8,
        indexer_compress_ratio=4, eos_token_id=1, bos_token_id=1, pad_token_id=None, output_gate_type="sigmoid")
    torch.manual_seed(0)
    model = modeling.Qwen4ExpForCausalLM(config).float()
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                param.normal_(0, 0.08)
    peft = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TARGETS))
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for name, param in sorted(peft.named_parameters()):
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=g) * 0.05)
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, peft


@contextlib.contextmanager
def trainer_path():
    """SPARSE mode "selected" and index tie order (shard-width-independent picks: needed for exact picks under CP),
    restored after (other tests in this process run the trainer's defaults)."""
    import rl1_trainer as rt

    old = rt.SPARSE["mode"], rt.TIES["mode"]
    rt.SPARSE["mode"], rt.TIES["mode"] = "selected", "index"
    try:
        yield
    finally:
        rt.SPARSE["mode"], rt.TIES["mode"] = old


def data():
    ids = torch.randint(2, V, (1, N), generator=torch.Generator().manual_seed(7))
    ids[0, [17, 60, 61, 133]] = 1  # eos: the n-gram ids' segment resets
    weights = torch.randn(1, N, V, generator=torch.Generator().manual_seed(3))
    return ids, weights


def loss_of(model, hidden, weights):
    return (model.lm_head(hidden).log_softmax(-1) * weights).sum()


def lora_grads(peft):
    return {n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for n, p in sorted(peft.named_parameters())
            if "lora_" in n}


def reference(p0, ckpt=False):
    """The trainer's single-rank path: the prefix through a cache (no grad), then the segment with grad."""
    with trainer_path():
        return _reference(p0, ckpt)


def _reference(p0, ckpt):
    model, peft = build(ckpt)
    ids, weights = data()
    cache = None
    if p0:
        with torch.no_grad():
            cache = model.model(input_ids=ids[:, :p0], use_cache=True).past_key_values
        import rl1_trainer as rt

        cache = rt.fq.read_only(cache)  # as the trainer's window pass (pure updates: no in-place state writes)
    hidden = model.model(input_ids=ids[:, p0:], past_key_values=cache, use_cache=cache is not None).last_hidden_state
    loss_of(model, hidden, weights[:, p0:]).backward()
    return hidden.detach()[0].numpy(), {n: g.numpy() for n, g in lora_grads(peft).items()}


def cp_rank(rank, world, arg):
    """One rank: sharded prefill of the context (``pre`` lengths), then its segment shard with grad; LoRA grads summed
    over ranks."""
    import rl1_cp
    import rl1_trainer as rt
    import torch.distributed as dist

    pre, seg, ckpt, offload = (*arg, False)[:4]
    rt.SPARSE["mode"], rt.TIES["mode"] = "selected", "index"
    model, peft = build(ckpt)
    ids, weights = data()
    p0 = sum(pre)
    prefix = None
    if p0:
        off = sum(pre[:rank])
        prefix = rl1_cp.prefill(model, rl1_cp.Ctx(pre, ids=ids[:, :p0]), ids[:, off:off + pre[rank]])
    cp = rl1_cp.Ctx(seg, ids=ids[:, p0:], prefix=prefix)
    start = p0 + cp.offset
    with rl1_cp.active(cp), offloading(model, offload):  # the checkpoint recompute runs the CP layers too
        hidden = rl1_cp.forward(model, cp, ids[:, start:start + cp.T])
        loss_of(model, hidden, weights[:, start:start + cp.T]).backward()
    stats_off = offload_stats(offload)
    grads = lora_grads(peft)
    for g in grads.values():
        dist.all_reduce(g)
    stats = {**rl1_cp.STATS, **stats_off}
    return hidden.detach()[0].numpy(), {n: g.numpy() for n, g in grads.items()}, stats


@contextlib.contextmanager
def offloading(model, on):
    """The trainer's tail_ckpt "offload" (rl1_memory.tail_scope) on CPU tensors (TAIL cpu, no size floor)."""
    import rl1_memory

    if not on:
        yield
        return
    old = dict(rl1_memory.TAIL)
    rl1_memory.TAIL.update(mode="offload", cpu=True, min_bytes=0, offloaded=0, bytes=0, kept=0)
    try:
        with rl1_memory.tail_scope(model):
            yield
    finally:
        counts = {k: rl1_memory.TAIL[k] for k in ("offloaded", "bytes")}
        rl1_memory.TAIL.clear()
        rl1_memory.TAIL.update(old, **{f"last_{k}": v for k, v in counts.items()})


def offload_stats(on):
    import rl1_memory

    return {"offloaded": rl1_memory.TAIL.get("last_offloaded", 0)} if on else {}


def check(pre, seg, ckpt=False, offload=False):
    import numpy as np

    ref_hidden, ref_grads = reference(sum(pre))
    out = run_ranks(cp_rank, world=len(seg), inputs=[(pre, seg, ckpt, offload)] * len(seg), timeout=300)
    hidden = np.concatenate([o[0] for o in out])
    assert hidden.shape == ref_hidden.shape
    scale = float(np.abs(ref_hidden).max())
    err = float(np.abs(hidden - ref_hidden).max())
    assert err <= 2e-5 * scale, ("hidden", err, scale)
    worst = 0.0
    for rank_out in out:
        for name, g in ref_grads.items():
            got = rank_out[1][name]
            gs = float(np.abs(g).max())
            assert gs > 0, name
            assert float(np.abs(got - g).max()) <= 1e-4 * gs, (name, float(np.abs(got - g).max()), gs)
            worst = max(worst, float(np.abs(got - g).max()) / gs)
    print(f"cp {pre} {seg}: hidden max rel err {err / scale:.2e}, LoRA grad max rel err {worst:.2e}")
    return out


@pytest.mark.multirank
def test_cp_two_ranks_prefix():
    out = check([25, 15], [97, 53])
    assert out[0][2]["state_bytes"] > 0 and out[0][2]["gather_bytes"] > 0 and out[0][2]["reduce_bytes"] > 0


@pytest.mark.multirank
def test_cp_three_ranks_empty_shard_checkpointed():
    check([0, 30, 10], [70, 0, 80], ckpt=True)


@pytest.mark.multirank
def test_cp_three_ranks_tiny_shards_no_prefix():
    check([], [4, 1, 185])  # the PLE halo (9 rows) and the GDN conv halo (3) reach across two earlier ranks


@pytest.mark.multirank
def test_cp_one_rank():
    check([40], [150])


def test_tail_offload_single_card():
    """tail_ckpt "offload" (restored 26 Oct for 130K on CP2): the checkpoint inputs leave and come back; the LoRA
    grads equal those without it within the path's own run-to-run spread (two runs without it)."""
    import numpy as np
    import rl1_memory

    def grads(offload):
        with trainer_path():
            model, peft = build(True)
            ids, weights = data()
            with offloading(model, offload):
                hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
                loss_of(model, hidden, weights).backward()
            return {n: g.numpy() for n, g in lora_grads(peft).items()}

    off, off2, on = grads(False), grads(False), grads(True)
    assert rl1_memory.TAIL["last_offloaded"] > 0
    for name, g in off.items():
        scale = float(np.abs(g).max())
        spread = float(np.abs(off2[name] - g).max())
        # the host copies' alignment changes CPU reduction order (measured 1.1e-6 rel; runs without it: bitwise)
        assert float(np.abs(on[name] - g).max()) <= max(2 * spread, 1e-5 * scale), (name, spread, scale)


@pytest.mark.multirank
def test_cp_two_ranks_tail_offload_checkpointed():
    out = check([25, 15], [97, 53], ckpt=True, offload=True)
    assert all(o[2]["offloaded"] > 0 for o in out)


# ---- Flash-shaped (T1c-v2 26 Oct: CP diverged from the single-card pass on GPU, grad cos 0.934): slow GDN decay
# (the state crosses shards: the model's random init forgets within a few tokens), Flash's 3 GDN : 1 QSA pattern with
# PLE on layer 1, M-RoPE positions with an image's jump, the whole model through call / prefill_call on the 64-token
# shard grid (rl1_cp.shard_lengths), as the trainer's cp_window_pass.
NR, P0R = 560, 230  # window tokens, carried context


def build_real(ckpt=False):
    import rl1_trainer as rt
    from peft import LoraConfig, get_peft_model
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    torch.set_num_threads(1)
    rt.selected_attention()
    lin, full = "linear_attention", "full_attention"
    config = Qwen4ExpTextConfig(
        vocab_size=V, hidden_size=H, num_hidden_layers=8, layer_types=[lin, lin, lin, full] * 2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, partial_rotary_factor=0.25,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        moe_intermediate_size=16, shared_expert_intermediate_size=16, num_experts=8, num_experts_per_tok=2,
        hc_count=4, hc_lowrank=8, ple_embed_dim=32, ple_layer_ids=[1], ngram_vocab_size_base=101,
        split_ngram_parts=4, indexer_n_heads=2, indexer_head_dim=8, indexer_kv_heads=1, indexer_budget=32,
        indexer_compress_ratio=4, eos_token_id=1, bos_token_id=1, pad_token_id=None, output_gate_type="sigmoid")
    torch.manual_seed(0)
    model = modeling.Qwen4ExpForCausalLM(config).float()
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                param.normal_(0, 0.08)
        for module in model.modules():  # Flash-like memory: per-token decay ~0.2-3% (half-lives of 20-300 tokens)
            if isinstance(module, modeling.Qwen4ExpTextGatedDeltaNet):
                module.A_log.copy_(torch.empty(module.A_log.shape).uniform_(0.02, 0.4, generator=g).log())
                module.dt_bias.fill_(-2.0)
    peft = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TARGETS))
    with torch.no_grad():
        for name, param in sorted(peft.named_parameters()):
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=g) * 0.05)
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, peft


def real_positions(n=NR, at=90, grid=(6, 8)):
    """M-RoPE [3, 1, n]: text, an image of grid h x w patches at token ``at`` (t fixed, h / w its grid), text again
    from the grid's max + 1 (positions run behind token indices from there)."""
    h, w = grid
    rows = torch.arange(n).repeat(3, 1)
    i = torch.arange(h * w)
    rows[:, at:at + h * w] = torch.stack([torch.full_like(i, at), at + i // w, at + i % w])
    rows[:, at + h * w:] = torch.arange(n - at - h * w) + at + max(h, w)
    return rows[:, None, :]


def data_real():
    ids = torch.randint(2, V, (1, NR), generator=torch.Generator().manual_seed(17))
    ids[0, [40, 300, 301, 470]] = 1
    return ids, torch.randn(1, NR, V, generator=torch.Generator().manual_seed(13))


def real_reference(p0):
    """The single-card window: the context through a read-only cache (no grad), then the segment; logits, LoRA grads."""
    import rl1_trainer as rt

    with trainer_path():
        model, peft = build_real()
        ids, weights = data_real()
        pos = real_positions()
        cache = None
        if p0:
            with torch.no_grad():
                cache = model.model(input_ids=ids[:, :p0], position_ids=pos[..., :p0], use_cache=True).past_key_values
            cache = rt.fq.read_only(cache)
        hidden = model.model(input_ids=ids[:, p0:], position_ids=pos[..., p0:], past_key_values=cache,
                             use_cache=cache is not None).last_hidden_state
        logits = model.lm_head(hidden)
        (logits.log_softmax(-1) * weights[:, p0:]).sum().backward()
        return logits.detach()[0].numpy(), {n: g.numpy() for n, g in lora_grads(peft).items()}


def real_cp_rank(rank, world, arg):
    """cp_window_pass's calls on CPU: the context by prefill_call over shard_lengths(P0), the segment by call."""
    import rl1_cp
    import rl1_trainer as rt
    import torch.distributed as dist

    p0, ckpt = arg
    rt.SPARSE["mode"], rt.TIES["mode"] = "selected", "index"
    model, peft = build_real(ckpt)
    ids, weights = data_real()
    pos = real_positions()
    pre = None
    if p0:
        pc = rl1_cp.Ctx(rl1_cp.shard_lengths(p0, world), ids=ids[:, :p0], position_ids=pos[..., :p0])
        a, b = pc.offset, pc.offset + pc.T
        model.eval()
        pre = rl1_cp.prefill_call(peft, pc, input_ids=ids[:, a:b], position_ids=pos[..., a:b])
        model.train()
    cp = rl1_cp.Ctx(rl1_cp.shard_lengths(NR - p0, world), ids=ids[:, p0:], prefix=pre, position_ids=pos)
    g0, g1 = p0 + cp.offset, p0 + cp.offset + cp.T
    with rl1_cp.active(cp):
        logits = rl1_cp.call(peft, cp, input_ids=ids[:, g0:g1], position_ids=pos[..., g0:g1]).logits
        (logits.log_softmax(-1) * weights[:, g0:g1]).sum().backward()
    grads = lora_grads(peft)
    for g in grads.values():
        dist.all_reduce(g)
    return logits.detach()[0].numpy(), {n: g.numpy() for n, g in grads.items()}, (g0, g1)


def check_real(p0, world, ckpt=False):
    import numpy as np

    ref_logits, ref_grads = real_reference(p0)
    out = run_ranks(real_cp_rank, world=world, inputs=[(p0, ckpt)] * world, timeout=600)
    logits = np.concatenate([o[0] for o in out])
    assert logits.shape == ref_logits.shape
    per_row = np.abs(logits - ref_logits).max(-1)
    scale = float(np.abs(ref_logits).max())
    bad = [(int(i) + p0, float(per_row[i])) for i in np.argsort(-per_row)[:5]]
    shards = [o[2] for o in out]
    assert float(per_row.max()) <= 2e-5 * scale, ("logits", shards, bad, scale)
    for name, g in ref_grads.items():
        gs = float(np.abs(g).max())
        got = out[0][1][name]
        assert float(np.abs(got - g).max()) <= 1e-4 * gs, (name, float(np.abs(got - g).max()), gs)
    print(f"cp real p0 {p0} world {world} shards {shards}: logits max rel {float(per_row.max()) / scale:.2e}")


def test_real_reference_carries_state_across_the_cut():
    """The realism is load-bearing: with the slow decay a context cut that drops the carried state changes the
    segment's logits far beyond the CP tolerance (the model's own init would forget within a few tokens)."""
    import numpy as np

    with_ctx = real_reference(P0R)[0]
    model, _ = build_real()
    ids, _ = data_real()
    with torch.no_grad(), trainer_path():
        alone = model.lm_head(model.model(input_ids=ids[:, P0R:], position_ids=real_positions()[..., P0R:]
                                          ).last_hidden_state)[0].numpy()
    tail = slice(NR - P0R - 64, NR - P0R)  # the segment's last 64 rows, 300+ tokens past the cut
    assert float(np.abs(with_ctx[tail] - alone[tail]).max()) > 1e-2 * float(np.abs(with_ctx).max())


@pytest.mark.multirank
def test_cp_real_two_ranks():
    check_real(P0R, 2)


@pytest.mark.multirank
def test_cp_real_three_ranks_checkpointed():
    check_real(P0R, 3, ckpt=True)


def test_gdn_core_simulated_torch_slow_decay():
    """The fold with Flash-like decay (state carried across every cut) and a non-zero h0."""
    kernel = __import__("rl1_cp").TorchGDN()
    inp, h0, do = _core_inputs(300, torch.float32, "cpu", slow=True)
    ref_o, ref_g = _full(kernel, inp, h0, do)
    out, grads = _simulate(kernel, inp, h0, [64, 128, 192, 256], do)
    fresh = _full(kernel, tuple(t[:, 64:] for t in inp[:5]) + inp[5:], h0 * 0, do[:, 64:])[0]
    assert float((fresh - ref_o[:, 64:])[:, :8].abs().max()) > 1e-2 * float(ref_o.abs().max())  # the carry matters
    assert float((out - ref_o).abs().max()) <= 1e-5 * float(ref_o.abs().max())
    for got, want in zip(grads, ref_g):
        assert float((got - want).abs().max()) <= 1e-4 * float(want.abs().max())


def _simulate(kernel, inp, h0, cuts, do):
    """CP of one GDN core in one process: per shard local states, the fold, outputs; dh0, fold_back, grads."""
    import rl1_cp

    bounds = list(zip([0] + cuts, cuts + [inp[0].shape[1]]))
    shards = [tuple(t[:, a:b].contiguous() for t in inp[:5]) + inp[5:] for a, b in bounds]
    local = [kernel.local(s) for s in shards]
    ms, ss = [x[1] for x in local], [x[0] for x in local]
    s_in = [rl1_cp.fold(ms, ss, h0, r) for r in range(len(shards))]
    out = torch.cat([kernel.out(s, s_in[r], local[r][2]) for r, s in enumerate(shards)], dim=1)
    keep = [kernel.saved(x[2]) if hasattr(kernel, "saved") else None for x in local]
    dos = [do[:, a:b].contiguous() for a, b in bounds]
    ds = [kernel.dh0(s, s_in[r], dos[r], keep[r]).float() for r, s in enumerate(shards)]
    needs = [True] * 5 + [False, False]
    grads = [kernel.bwd(s, s_in[r], dos[r], rl1_cp.fold_back(ms, ds, r), keep[r], needs)
             for r, s in enumerate(shards)]
    return out, [torch.cat([g[i] for g in grads], dim=1) for i in range(5)]


def _core_inputs(T, dtype, device, hk=2, hv=6, d=16, slow=False):
    """``slow``: Flash-like decay (A 0.02-0.4, dt_bias -2: ~0.3-4% per token), so states cross shard cuts; the default
    (A 0.5-1.5, dt_bias ~N(0, 1)) forgets within a few tokens."""
    import rl1_cp

    g = torch.Generator(device="cpu").manual_seed(11)

    def r(*shape):
        return torch.randn(*shape, generator=g).to(device)

    q = rl1_cp._l2norm(r(1, T, hk, d)).to(dtype)
    k = rl1_cp._l2norm(r(1, T, hk, d)).to(dtype)
    v = r(1, T, hv, d).to(dtype)
    a = r(1, T, hv).to(dtype)
    beta = torch.sigmoid(r(1, T, hv)).to(dtype)
    if slow:
        A_log = torch.rand(hv, generator=g).mul(0.38).add(0.02).log().to(device)
        dt_bias = torch.full((hv,), -2.0).to(device)
        _ = r(hv)  # the generator's draws as the default's
    else:
        A_log = torch.rand(hv, generator=g).add(0.5).log().to(device)
        dt_bias = r(hv)
    h0 = r(1, hv, d, d).float() * 0.1
    return (q, k, v, a, beta, A_log, dt_bias), h0, r(1, T, hv, d).to(dtype)


def _full(kernel, inp, h0, do):
    s_loc, m, pre = kernel.local(inp)
    out = kernel.out(inp, h0, pre)
    keep = kernel.saved(pre) if hasattr(kernel, "saved") else None
    return out, kernel.bwd(inp, h0, do, None, keep, [True] * 5 + [False, False])[:5]


def test_gdn_core_simulated_torch():
    """The fold / fold_back algebra on the torch kernel: shards (incl. empty, off the 64-token chunk grid) vs one call."""
    import rl1_cp

    kernel = rl1_cp.TorchGDN()
    inp, h0, do = _core_inputs(300, torch.float32, "cpu")
    ref_o, ref_g = _full(kernel, inp, h0, do)
    out, grads = _simulate(kernel, inp, h0, [37, 37, 128, 250], do)
    assert float((out - ref_o).abs().max()) <= 1e-5 * float(ref_o.abs().max())
    for got, want in zip(grads, ref_g):
        assert float((got - want).abs().max()) <= 1e-4 * float(want.abs().max())
    print("gdn core simulated: o", float((out - ref_o).abs().max() / ref_o.abs().max()),
          "grads", [float((g - w).abs().max() / w.abs().max()) for g, w in zip(grads, ref_g)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU: fla kernels")
def test_fla_simulated():
    """FlaGDN (BF16, fla's kernels) sharded on the 64-token grid vs one fla call (GPU only; the CPU suite skips it).
    Not bitwise: M and S_local carry the kernel's BF16 state casts (see rl1_cp)."""
    pytest.importorskip("fla")
    import rl1_cp

    kernel = rl1_cp.FlaGDN()
    inp, h0, do = _core_inputs(4096, torch.bfloat16, "cuda", hk=16, hv=48, d=128)
    ref_o, ref_g = _full(kernel, inp, h0, do)
    out, grads = _simulate(kernel, inp, h0, [1024, 1024, 2048, 3072], do)
    err = float((out.float() - ref_o.float()).abs().max()) / float(ref_o.float().abs().max())
    print("fla o rel err", err)
    assert err < 2e-2
    for i, (got, want) in enumerate(zip(grads, ref_g)):
        e = float((got.float() - want.float()).abs().max()) / float(want.float().abs().max())
        print("fla grad", i, "rel err", e)
        assert e < 5e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU: fla kernels")
def test_fla_simulated_slow_decay():
    """As test_fla_simulated with Flash-like decay (the state crosses each cut; the default inputs forget within a
    chunk, so the fold went unchecked: rel err 0.0 on T1a) and T1c's shard sizes (two 7,360-token shards after a
    15,296-token one). Prints the rows' error just after each cut."""
    pytest.importorskip("fla")
    import rl1_cp

    kernel = rl1_cp.FlaGDN()
    cuts = [15296, 15296 + 7360]
    inp, h0, do = _core_inputs(15296 + 2 * 7360, torch.bfloat16, "cuda", hk=16, hv=48, d=128, slow=True)
    ref_o, ref_g = _full(kernel, inp, h0, do)
    out, grads = _simulate(kernel, inp, h0, cuts, do)
    scale = float(ref_o.float().abs().max())
    for c in cuts:
        print("fla slow: rows after cut", c, "rel err", float((out - ref_o)[:, c:c + 256].float().abs().max()) / scale)
    err = float((out.float() - ref_o.float()).abs().max()) / scale
    print("fla slow o rel err", err)
    assert err < 2e-2
    for i, (got, want) in enumerate(zip(grads, ref_g)):
        e = float((got.float() - want.float()).abs().max()) / float(want.float().abs().max())
        print("fla slow grad", i, "rel err", e)
        assert e < 5e-2


def served_table(layers, k=2, experts=8):
    """A served route table [N, layers, k]: per token and layer k distinct expert ids (deterministic)."""
    g = torch.Generator().manual_seed(23)
    return torch.stack([torch.stack([torch.randperm(experts, generator=g)[:k] for _ in range(layers)])
                        for _ in range(N)]).to(torch.int16)


def applied_routes(model, call):
    """Per MoE layer the expert sets the call applied (sorted): each router's returned indices (its last call)."""
    got, hooks = {}, []
    for i, layer in enumerate(model.model.layers):
        for m in layer.mlp.modules():
            if type(m).__name__ == "Qwen4ExpTextTopKRouter":
                hooks.append(m.register_forward_hook(
                    lambda _m, _a, out, key=f"L{i:02d}.route": got.__setitem__(key, out[2].detach().sort(-1).values.clone())))
    try:
        call()
    finally:
        for h in hooks:
            h.remove()
    return got


def served_rank(rank, world, arg):
    """cp_window_pass's routing on CPU: the served table sliced by global token (prefill shard [a, b), tail shard
    [g0, g1)); every MoE layer must apply exactly the table's experts on its rows."""
    import rl1_cp
    import rl1_trainer as rt

    pre, seg = arg
    rt.SPARSE["mode"], rt.TIES["mode"] = "selected", "index"
    model, _ = build()
    ids, _ = data()
    rt.route_patch()
    layers = sorted({i for i, _ in enumerate(model.model.layers)})
    rt.route_columns(model, layers)
    table = served_table(len(layers))
    rt.ROUTE.update(router="served", window=table)
    p0 = sum(pre)
    pc = rl1_cp.Ctx(pre, ids=ids[:, :p0])
    a, b = pc.offset, pc.offset + pc.T
    rt.ROUTE["call"] = rt.served_call(a, b)
    with torch.no_grad():
        got_pre = applied_routes(model, lambda: rl1_cp.prefill(model, pc, ids[:, a:b]))
        pre_prefix = rl1_cp.prefill(model, rl1_cp.Ctx(pre, ids=ids[:, :p0]), ids[:, a:b])
    cp = rl1_cp.Ctx(seg, ids=ids[:, p0:], prefix=pre_prefix)
    g0, g1 = p0 + cp.offset, p0 + cp.offset + cp.T
    rt.ROUTE["call"] = rt.served_call(g0, g1)
    with torch.no_grad(), rl1_cp.active(cp):
        got_tail = applied_routes(model, lambda: rl1_cp.forward(model, cp, ids[:, g0:g1]))
    rt.ROUTE.update(router="off", window=None, call=None)
    bad = []
    for name, got, lo, hi in (("pre", got_pre, a, b), ("tail", got_tail, g0, g1)):
        for key, applied in got.items():
            col = int(key[1:3])
            want = table[lo:hi, layers.index(col)].long().sort(-1).values
            if applied.shape != want.shape or not bool((applied == want).all()):
                bad.append((name, key, lo, hi))
    return {"bad": bad, "layers": len(got_tail), "rows": [a, b, g0, g1]}


@pytest.mark.multirank
def test_cp_served_routes_by_global_token():
    """RL 26 Oct: a shard-offset bug (rank 1 reading its served rows by local index). With a
    served table at P0 > 0 on 2 ranks, every MoE layer of every rank applies the table's experts of its global
    tokens, in the prefill and in the tail."""
    out = run_ranks(served_rank, world=2, inputs=[([25, 15], [97, 53])] * 2, timeout=300)
    assert all(o["layers"] > 0 for o in out)
    assert all(not o["bad"] for o in out), [o["bad"][:4] for o in out]


def test_tail_offload_only_past_min_tokens():
    """real-v4: tail_ckpt_min keeps windows under it on the card (no hooks); windows at or past it offload."""
    import rl1_memory

    old = dict(rl1_memory.TAIL)
    rl1_memory.TAIL.update(mode="offload", cpu=True, min_bytes=0, min_tokens=1000, width=4, offloaded=0, bytes=0, kept=0)
    try:
        x = torch.randn(3, 4, requires_grad=True)
        for tail, moved in ((999, False), (1000, True), (None, True)):
            rl1_memory.TAIL["offloaded"] = 0
            with rl1_memory.tail_scope(None, tail):
                y = (x * x).sum()
            y.backward()
            assert (rl1_memory.TAIL["offloaded"] > 0) == moved, (tail, rl1_memory.TAIL["offloaded"])
    finally:
        rl1_memory.TAIL.clear()
        rl1_memory.TAIL.update(old)
