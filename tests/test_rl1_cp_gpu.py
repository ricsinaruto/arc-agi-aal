"""rl1_cp on GPU with the trainer's fast paths (T1c-v2 26 Oct: CP2 diverged in the forward, grad cos 0.934, while the
CPU tests with the torch kernels are exact). A Flash-shaped model at kernel-compatible sizes (GDN K = V = 128, QSA
head 256, indexer 128, block 4), BF16 under autocast as the trainer, 2 ranks over NCCL. Each rank runs the single-card
window (context into a cache, then the segment) and the CP window (prefill_call, call), and compares per layer
(GDN out, attention out, decoder layer out) and the logits on its shard's rows. The floor: the same single-card
window with the context in two cache chunks (a correct but differently-rounded pass). Variants switch one fast path
off at a time; the first layer whose CP error passes 4 x the floor names the op. GPU only (2 cards, fla, triton)."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
pytest.importorskip("peft")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))

from arc_infra.runops.multirank import run_ranks  # noqa: E402

GPU2 = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="2 GPUs")
VG, NG, P0G = 1024, 7200, 4090  # vocab, window, context (off the 64 grid, as T1c's 30,578)
FAST = {"gdn_fast": True, "gdn_keep": True, "union_attention": True, "prefix_nograd": True, "sel_cache": True,
        "hyper_fast": True, "hyper_xb": True, "cast_once": True, "fast_topk": True, "fast_topk_wide": "filter"}
VARIANTS = {"all": FAST, "no_union": {**FAST, "union_attention": False, "prefix_nograd": False},
            "no_fast_topk": {**FAST, "fast_topk": False}, "no_gdn_fast": {**FAST, "gdn_fast": False, "gdn_keep": False},
            "no_hyper_cast": {**FAST, "hyper_fast": False, "hyper_xb": False, "cast_once": False}}
TARGETS = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a",
           "linear_attn.out_proj", "shared_expert.up_proj", "shared_expert.down_proj", "shared_expert.gate_proj",
           "q_proj", "k_proj", "v_proj", "o_proj"]


def build_gpu(device, dtype=torch.bfloat16):
    import rl1_trainer as rt
    from peft import LoraConfig, get_peft_model
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    rt.selected_attention()
    lin, full = "linear_attention", "full_attention"
    config = Qwen4ExpTextConfig(
        vocab_size=VG, hidden_size=512, num_hidden_layers=8, layer_types=[lin, lin, lin, full] * 2,
        num_attention_heads=8, num_key_value_heads=2, head_dim=256, partial_rotary_factor=0.25,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=128, linear_value_head_dim=128,
        moe_intermediate_size=128, shared_expert_intermediate_size=128, num_experts=8, num_experts_per_tok=2,
        hc_count=4, hc_lowrank=16, ple_embed_dim=64, ple_layer_ids=[1], ngram_vocab_size_base=1009,
        split_ngram_parts=4, indexer_n_heads=4, indexer_head_dim=128, indexer_kv_heads=1, indexer_budget=256,
        indexer_compress_ratio=4, eos_token_id=1, bos_token_id=1, pad_token_id=None, output_gate_type="sigmoid")
    torch.manual_seed(0)
    model = modeling.Qwen4ExpForCausalLM(config)
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                param.normal_(0, 0.03)
        for module in model.modules():  # Flash-like memory (states cross every shard cut)
            if isinstance(module, modeling.Qwen4ExpTextGatedDeltaNet):
                module.A_log.copy_(torch.empty(module.A_log.shape).uniform_(0.02, 0.4, generator=g).log())
                module.dt_bias.fill_(-2.0)
    model = model.to(device=device, dtype=dtype)
    peft = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=TARGETS))
    with torch.no_grad():
        for name, param in sorted(peft.named_parameters()):
            if "lora_B" in name:
                param.copy_((torch.randn(param.shape, generator=g) * 0.02).to(param))
    model.train()  # forward-only compares: no checkpointing (the recompute runs the same CP layers)
    return model, peft


def positions(n, at, grid):
    h, w = grid
    rows = torch.arange(n).repeat(3, 1)
    i = torch.arange(h * w)
    rows[:, at:at + h * w] = torch.stack([torch.full_like(i, at), at + i // w, at + i % w])
    rows[:, at + h * w:] = torch.arange(n - at - h * w) + at + max(h, w)
    return rows[:, None, :]


def hooks(model, store):
    """Per layer: GDN / attention / decoder-layer outputs (fp32 on the host), keyed (layer, op)."""
    handles = []
    for i, layer in enumerate(model.model.layers):
        def grab(key):
            def hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                store[key] = t.detach().float().cpu()
            return hook
        mixer = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
        handles.append(mixer.register_forward_hook(grab((i, "gdn" if layer.layer_type == "linear_attention"
                                                          else "attn"))))
        handles.append(layer.register_forward_hook(grab((i, "layer"))))
    return handles


SIZES = {"gpu": (NG, P0G, 900, (16, 24)), "cpu": (1300, 690, 300, (6, 8))}  # window, context, image at, grid


def single(model, ids, pos, chunks, p0, amp):
    """The single-card window: the context through a cache in ``chunks`` (no grad), then the segment (read-only cache)."""
    import rl1_trainer as rt

    model.eval()
    cache, at = None, 0
    with torch.no_grad(), amp():
        for end in chunks:
            cache = model.model(input_ids=ids[:, at:end], position_ids=pos[..., at:end], past_key_values=cache,
                                use_cache=True).past_key_values
            at = end
    model.train()
    store = {}
    handles = hooks(model, store)
    try:
        with torch.no_grad(), amp():
            hidden = model.model(input_ids=ids[:, p0:], position_ids=pos[..., p0:],
                                 past_key_values=rt.fq.read_only(cache), use_cache=True).last_hidden_state
            store["logits"] = model.lm_head(hidden).float().log_softmax(-1).cpu()
    finally:
        for h in handles:
            h.remove()
    return store


def gpu_rank(rank, world, arg):
    """One rank: the single-card window twice (the floor: context in two chunks), then the CP window; per key the
    errors on this rank's rows. ``arg`` (variant, kind): kind "cpu" is the harness's own check (torch kernels, fp32,
    gloo; CP must match to fp32 rounding there)."""
    import contextlib

    import rl1_cp
    import rl1_trainer as rt

    variant, kind = arg
    n, p0, at, grid = SIZES[kind]
    gpu = kind == "gpu"
    device = torch.device("cuda", rank) if gpu else torch.device("cpu")
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if gpu else contextlib.nullcontext
    rt.SPARSE["mode"], rt.TIES["mode"] = ("triton" if gpu else "selected"), "index"
    if not gpu:
        torch.set_num_threads(2)
    model, peft = build_gpu(device, torch.bfloat16 if gpu else torch.float32)
    if gpu:
        rt.op_paths(peft, VARIANTS[variant])
    ids = torch.randint(2, VG, (1, n), generator=torch.Generator().manual_seed(17)).to(device)
    pos = positions(n, at, grid).to(device)
    ref = single(model, ids, pos, [p0], p0, amp)
    floor = single(model, ids, pos, [p0 // 2, p0], p0, amp)  # the context in two chunks
    pc = rl1_cp.Ctx(rl1_cp.shard_lengths(p0, world), ids=ids[:, :p0], position_ids=pos[..., :p0])
    a, b = pc.offset, pc.offset + pc.T
    model.eval()
    with amp():
        pre = rl1_cp.prefill_call(peft, pc, input_ids=ids[:, a:b], position_ids=pos[..., a:b])
    model.train()
    cp = rl1_cp.Ctx(rl1_cp.shard_lengths(n - p0, world), ids=ids[:, p0:], prefix=pre, position_ids=pos)
    g0, g1 = p0 + cp.offset, p0 + cp.offset + cp.T
    mine = {}
    handles = hooks(model, mine)
    try:
        with torch.no_grad(), rl1_cp.active(cp), amp():
            mine["logits"] = rl1_cp.call(peft, cp, input_ids=ids[:, g0:g1],
                                         position_ids=pos[..., g0:g1]).logits.float().log_softmax(-1).cpu()
    finally:
        for h in handles:
            h.remove()
    rows = slice(g0 - p0, g1 - p0)
    out = {}
    for key in ref:
        r, f, c = ref[key][0, rows], floor[key][0, rows], mine[key][0]
        scale = float(r.abs().max()) or 1.0
        err = (c - r).abs().amax(-1)
        out[key] = {"cp": float(err.max()) / scale, "floor": float((f - r).abs().max()) / scale,
                    "cp_mean": float((c - r).abs().mean()), "floor_mean": float((f - r).abs().mean()),
                    "worst_row": g0 + int(err.argmax()), "first_rows": float(err[:64].max()) / scale}
    gdn = sys.modules.get("rl1_gdn")
    return {"shard": (g0, g1), "cmp": out, "gdn": dict(gdn.GDN.get("why", {})) if gdn else None}


def report(variant, out):
    """Print per rank and key; the first (layer, op) beyond 4 x floor + 1e-3 (None: none)."""
    first = None
    for rank, res in enumerate(out):
        print(f"[{variant}] rank {rank} shard {res['shard']} gdn fallbacks {res['gdn']}")
        for key in sorted(res["cmp"], key=order):
            c = res["cmp"][key]
            bad = c["cp"] > 4 * c["floor"] + 1e-3
            print(f"  {key}: cp {c['cp']:.3e} (mean {c['cp_mean']:.3e}, first 64 rows {c['first_rows']:.3e}, worst "
                  f"row {c['worst_row']}) floor {c['floor']:.3e} (mean {c['floor_mean']:.3e})"
                  f"{'  <-- BEYOND' if bad else ''}")
            if bad and key != "logits" and (first is None or key < first[1]):
                first = (rank, key)
    return first


def order(key):
    return (-1, "") if key == "logits" else key


@pytest.mark.multirank
@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU check (on a GPU box the fast paths' Triton kernels engage)")
def test_cp_harness_on_cpu():
    """The harness on CPU (torch kernels, fp32): CP matches the single-card window to fp32 rounding at every layer."""
    out = run_ranks(gpu_rank, world=2, inputs=[("all", "cpu")] * 2, timeout=600)
    report("cpu", out)
    for res in out:
        for key, c in res["cmp"].items():
            assert c["cp"] <= 1e-4, (key, c)


@GPU2
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_cp_gpu_matches_single_card_within_the_floor(variant):
    pytest.importorskip("fla")
    out = run_ranks(gpu_rank, world=2, inputs=[(variant, "gpu")] * 2, timeout=900, backend="nccl")
    first = report(variant, out)
    lp = [res["cmp"]["logits"] for res in out]
    assert first is None and all(c["cp_mean"] <= 4 * c["floor_mean"] + 1e-4 for c in lp), (variant, first, lp)
