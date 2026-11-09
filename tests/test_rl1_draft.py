"""rl1_draft (RL stage D): the MTP draft head's LoRA trained beside the policy's. The tiny Flash-shaped model of
test_rl1_dist (real qwen4_exp classes, peft LoRA) with a tiny draft head: checkpoint loading (per-expert mtp.* fused),
NVFP4 values, prefix chunking invariance, the policy's gradients untouched, the step's draft fields, the export and
its merge into vLLM's packed draft weights, a policy-only checkpoint resumed with the draft, and the 3-rank
DP + CP path (an empty prefix shard, a rank with no data-parallel window) against one process."""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
pytest.importorskip("peft")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))
sys.path.insert(0, str(ROOT / "tests"))

import test_rl1_dist as base  # noqa: E402  (its tiny trainer harness)
from arc_infra import draft_lora_runtime  # noqa: E402
from arc_infra.runops.multirank import run_ranks  # noqa: E402

# name -> (window tokens, tail): "big" runs sequence-parallel (prefix shards 20 / 0 / 0), "a" (cut 140: prefix
# chunks 64 / 64 / 12) and "b" (no prefix) data-parallel; rank 2 has no data-parallel window
WINDOWS = {"big": (190, 170), "a": (200, 60), "b": (100, 100)}
LR = base.LR


def config():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    lin, full = "linear_attention", "full_attention"
    return Qwen4ExpTextConfig(
        vocab_size=base.V, hidden_size=base.H, num_hidden_layers=4, layer_types=[lin, lin, full, lin],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, partial_rotary_factor=0.25,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        moe_intermediate_size=16, shared_expert_intermediate_size=16, num_experts=8, num_experts_per_tok=2,
        hc_count=4, hc_lowrank=8, ple_embed_dim=32, ple_layer_ids=[2], ngram_vocab_size_base=101,
        split_ngram_parts=4, indexer_n_heads=2, indexer_head_dim=8, indexer_kv_heads=1, indexer_budget=8,
        indexer_compress_ratio=4, eos_token_id=1, bos_token_id=1, pad_token_id=None, output_gate_type="sigmoid")


def head(seed=11):
    """A tiny draft head, random weights, LoRA (r 4, alpha 8) with B non-zero."""
    import rl1_draft

    torch.manual_seed(seed)
    h = rl1_draft.DraftHead(config()).float()
    with torch.no_grad():
        for p in h.parameters():
            p.normal_(0, 0.08) if p.dim() > 1 else p.normal_(0, 0.05)
    for p in h.parameters():
        p.requires_grad_(False)
    rl1_draft.add_lora(h, 4, 8)
    h.lora_rank, h.lora_alpha = 4, 8
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for n, p in sorted(h.named_parameters()):
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return h


@pytest.fixture
def sparse():
    import rl1_trainer as rt

    rt.selected_attention()
    old = rt.SPARSE["mode"], rt.TIES["mode"]
    rt.SPARSE["mode"], rt.TIES["mode"] = "selected", "index"
    yield
    rt.SPARSE["mode"], rt.TIES["mode"] = old


def test_nvfp4_values():
    import rl1_draft

    w = torch.randn(6, 64, generator=torch.Generator().manual_seed(0)) * 0.02
    q = rl1_draft.nvfp4_values(w)
    for block in q.reshape(-1, 16):  # E2M1: at most 8 magnitudes per block
        assert len(torch.unique(block.abs())) <= 8
    err = ((q - w).abs().reshape(6, 4, 16).amax(-1) / w.abs().reshape(6, 4, 16).amax(-1)).max()
    assert 0 < float(err) <= 0.26  # E2M1: half a step at the block's top (6 -> step 2) after the FP8 scale
    assert torch.equal(rl1_draft.nvfp4_values(torch.zeros(2, 16)), torch.zeros(2, 16))


def test_checkpoint_mtp_state_fuses_experts(tmp_path):
    import rl1_draft
    from safetensors.torch import save_file

    src = head()
    plain = {k: v for k, v in src.state_dict().items() if "lora_" not in k and "rotary" not in k}
    ckpt = {}
    for k, v in plain.items():  # the served checkpoint's names: LoRA's base. unwrapped, routed experts per expert
        k = k.replace(".base.", ".")
        if k.endswith("experts.gate_up_proj"):
            for e in range(v.shape[0]):
                gate, up = v[e].chunk(2, 0)
                ckpt[f"mtp.layers.0.mlp.experts.{e}.gate_proj.weight"] = gate.contiguous()
                ckpt[f"mtp.layers.0.mlp.experts.{e}.up_proj.weight"] = up.contiguous()
        elif k.endswith("experts.down_proj"):
            for e in range(v.shape[0]):
                ckpt[f"mtp.layers.0.mlp.experts.{e}.down_proj.weight"] = v[e].contiguous()
        else:
            ckpt["mtp." + k] = v.contiguous()
    ckpt["model.language_model.norm.weight"] = torch.zeros(4)
    save_file(ckpt, str(tmp_path / "m.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "m.safetensors" for k in ckpt}}))
    exact = rl1_draft.load(rl1_draft.DraftHead(config()), rl1_draft.mtp_state(tmp_path), fp4=False)
    want = {k.replace(".base.", "."): v for k, v in plain.items()}
    for k, v in exact.state_dict().items():
        if "rotary" not in k:
            assert torch.equal(v, want[k]), k
    four = rl1_draft.load(rl1_draft.DraftHead(config()), rl1_draft.mtp_state(tmp_path), fp4=True)
    gu = four.layers[0].mlp.experts.gate_up_proj
    assert torch.equal(gu[3], rl1_draft.nvfp4_values(want["layers.0.mlp.experts.gate_up_proj"][3]))
    assert not any(p.requires_grad for p in four.parameters())


def _run(h, ids, hc, cut, chunks, kept, trunk):
    """Draft prefix in ``chunks`` (trunk positions), then the window [cut, T): (sums, LoRA grads)."""
    import rl1_draft

    d = rl1_draft.Draft(h, weight=1.0)
    embed = torch.nn.Embedding(base.V, base.H)
    torch.manual_seed(3)
    torch.nn.init.normal_(embed.weight, 0, 0.5)
    embed.weight.requires_grad_(False)
    lm = torch.randn(base.V, base.H, generator=torch.Generator().manual_seed(4)) * 0.3
    pos = lambda a, b: torch.arange(a, b).view(1, 1, -1).expand(3, 1, -1)  # noqa: E731
    for p in h.parameters():
        p.grad = None
    state, start = None, 0
    for stop in chunks:
        state = d.prefix_chunk(state, ids, start, stop, hc[:, start:stop], pos, embed)
        start = stop
    sums = d.window(state, ids, cut, hc[:, cut:], pos, embed, lm, trunk, None, kept, 0.7)
    return sums, {n: p.grad.clone() for n, p in rl1_draft.lora_params(h)}


def test_prefix_chunking_is_exact_and_values_match_one_pass(sparse):
    T, cut = 90, 40
    g = torch.Generator().manual_seed(9)
    ids = torch.randint(2, base.V, (1, T), generator=g)
    hc = torch.randn(1, T, 4 * base.H, generator=g)
    kept = torch.sort(torch.randperm(T - cut - 1, generator=g)[:20] + cut + 1).values
    trunk = torch.randn(len(kept), base.V, generator=g) * 2
    h = head()
    one, _ = _run(h, ids, hc, cut, [cut], kept, trunk)
    three, g3 = _run(h, ids, hc, cut, [16, 32, cut], kept, trunk)
    _, g1 = _run(h, ids, hc, cut, [cut], kept, trunk)
    assert one["rows"] == three["rows"] == 20 and abs(one["kl"] - three["kl"]) <= 1e-5 * one["kl"]
    for n in g1:
        assert torch.allclose(g1[n], g3[n], rtol=1e-4, atol=1e-7), n
    assert any(float(v.abs().sum()) > 0 for k, v in g1.items() if "lora_A" in k)
    whole, _ = _run(h, ids, hc, 0, [], kept, trunk)  # no prefix: the same rows' values in one pass
    assert whole["rows"] == 20 and abs(whole["kl"] - one["kl"]) <= 1e-5 * one["kl"]
    assert whole["top1"] == one["top1"] and whole["agree"] == one["agree"]
    assert abs(whole["paccept"] - one["paccept"]) <= 1e-5


def test_export_and_merge_into_vllm_draft_weights(tmp_path):
    import rl1_draft
    from safetensors.torch import load_file

    h = head()
    meta = rl1_draft.export(h, tmp_path)
    tensors = load_file(str(tmp_path / "draft_lora.safetensors"))
    assert json.loads((tmp_path / "draft_lora.json").read_text()) == meta and meta["format"] == rl1_draft.FORMAT
    assert set(meta["modules"]) == set(rl1_draft.TARGETS) and len(tensors) == 2 * len(rl1_draft.TARGETS)
    want = {"fc_embedding": ("model.fc_embedding", None), "fc_hidden": ("model.fc_hidden", None),
            "layers.0.self_attn.q_proj": ("model.layers.0.self_attn.qkv_proj", "q"),
            "layers.0.self_attn.k_proj": ("model.layers.0.self_attn.qkv_proj", "k"),
            "layers.0.self_attn.v_proj": ("model.layers.0.self_attn.qkv_proj", "v"),
            "layers.0.self_attn.o_proj": ("model.layers.0.self_attn.o_proj", None),
            "layers.0.mlp.shared_expert.gate_proj": ("model.layers.0.mlp.shared_expert.gate_up_proj", 0),
            "layers.0.mlp.shared_expert.up_proj": ("model.layers.0.mlp.shared_expert.gate_up_proj", 1),
            "layers.0.mlp.shared_expert.down_proj": ("model.layers.0.mlp.shared_expert.down_proj", None)}
    assert {k: (v["vllm"], v["shard"]) for k, v in meta["modules"].items()} == want
    for k in tensors:  # PEFT names whose module names vLLM's adapter parser keeps (mtp. prefix, *_proj suffixes)
        assert k.startswith("base_model.model.mtp.") and k.endswith((".lora_A.weight", ".lora_B.weight"))
    # vLLM's draft weights (TP 1): q | k | v and gate | up packed in checkpoint order
    sd = {k.replace(".base.", "."): v.to(torch.bfloat16) for k, v in h.state_dict().items() if "lora_" not in k}
    attn, mlp = "layers.0.self_attn.", "layers.0.mlp.shared_expert."
    served = {"model.fc_embedding.weight": sd["fc_embedding.weight"].clone(),
              "model.fc_hidden.weight": sd["fc_hidden.weight"].clone(),
              "model.layers.0.self_attn.qkv_proj.weight": torch.cat([sd[attn + f"{x}_proj.weight"] for x in "qkv"]),
              "model.layers.0.self_attn.o_proj.weight": sd[attn + "o_proj.weight"].clone(),
              "model.layers.0.mlp.shared_expert.gate_up_proj.weight": torch.cat([sd[mlp + "gate_proj.weight"],
                                                                                 sd[mlp + "up_proj.weight"]]),
              "model.layers.0.mlp.shared_expert.down_proj.weight": sd[mlp + "down_proj.weight"].clone()}
    before = {k: v.clone() for k, v in served.items()}
    touched = draft_lora_runtime.merge(served, tensors, meta)
    assert len(touched) == 6
    for name in rl1_draft.TARGETS:  # each shard's rows = base + 2 B A (alpha / r = 2), from the exported factors
        param, shard = want[name]
        mod = h.get_submodule(name)
        delta = 2.0 * (tensors[f"base_model.model.mtp.{name}.lora_B.weight"].float()
                       @ tensors[f"base_model.model.mtp.{name}.lora_A.weight"].float())
        rows = mod.base.weight.shape[0]
        start = {None: 0, "q": 0, "k": sd[attn + "q_proj.weight"].shape[0],
                 "v": sd[attn + "q_proj.weight"].shape[0] + sd[attn + "k_proj.weight"].shape[0],
                 0: 0, 1: sd[mlp + "gate_proj.weight"].shape[0]}[shard]
        got = served[param + ".weight"][start:start + rows].float()
        assert torch.allclose(got, (before[param + ".weight"][start:start + rows].float() + delta).to(torch.bfloat16).float())
    draft_lora_runtime.merge(served, tensors, meta, sign=-1.0)
    for k in served:
        assert (served[k].float() - before[k].float()).abs().max() <= 2 ** -6 * before[k].float().abs().max() + 1e-3


# ---- the trainer --------------------------------------------------------------------------------------------------

def window(name):
    saved = dict(base.WINDOWS)
    base.WINDOWS.clear()
    base.WINDOWS.update(WINDOWS)
    try:
        return base.window(name)
    finally:
        base.WINDOWS.clear()
        base.WINDOWS.update(saved)


def specs():
    return [{"name": n, "w": 1.0, "est_tokens": e, "est_tail": t} for n, (e, t) in WINDOWS.items()]


def trainer(folder, draft=True):
    import rl1_draft
    import rl1_trainer as rt

    t = base.trainer(folder)

    def feed(specs_, cmd, **kw):
        todo = list(specs_)
        return (lambda: window(todo.pop(0)["name"])), (lambda: None), {"waits": 0.0}
    t.window_feed = feed
    if draft:
        t.attach_draft(rl1_draft.Draft(head(), weight=1.0), 1.0)
    t.initial = {n: p.detach().clone() for n, p in t.params if "lora_A" in n}
    t.optimizer = t.make_optimizer(LR)
    t.draft_grads = []
    finish = t.draft_finish if draft else None
    if draft:
        def capture(cmd):
            out = finish(cmd)
            t.draft_grads = [p.grad.detach().clone() for _, p in t.draft_params()]
            return out
        t.draft_finish = capture
    del rt
    return t


def one_process(chunked=False, draft=True):
    import tempfile

    import rl1_trainer as rt

    kept = (rt.SPARSE["mode"], rt.TIES["mode"], rt.fq.positions_and_images, torch.cuda.synchronize)
    try:
        with base.isolated(), tempfile.TemporaryDirectory() as folder:
            t = trainer(folder, draft)
            extra = {"chunked_head": True} if chunked else {}
            acc = t.op_accumulate({"attempts": specs(), "lr": LR, "first": True, **extra})
            assert acc["ok"], acc
            step = t.op_step({"lr": LR, "ref": "keep", **extra})
            assert step["ok"], step
            saved = t.save_adapter(Path(folder) / "adapter")
            files = sorted(p.name for p in (Path(folder) / "adapter").iterdir())
            return {"grads": [g.numpy() for g in t.ref["grads"]], "draft": [g.numpy() for g in t.draft_grads],
                    "step": step, "saved": saved, "files": files,
                    "params": {n: p.detach().clone() for n, p in t.params}}
    finally:
        rt.SPARSE["mode"], rt.TIES["mode"], rt.fq.positions_and_images, torch.cuda.synchronize = kept


def test_trainer_draft_step_leaves_the_policy_alone():
    import numpy as np

    off, on = one_process(draft=False), one_process()
    assert len(on["grads"]) == len(off["grads"])
    for a, b in zip(on["grads"], off["grads"]):  # the draft adds no gradient to the policy
        assert np.array_equal(a, b)
    s = on["step"]
    for k in ("loss", "grad_norm", "update_norm", "trained_tokens"):
        assert s[k] == off["step"][k], k
    trained = sum(min(40, t - 1) for _, t in WINDOWS.values())  # "b" (no prefix) drops a token at position 1
    assert s["draft_rows"] in (trained - 1, trained) and s["draft_windows"] == 3 and not s["draft_skipped"]
    assert s["draft_kl"] > 0 and 0 <= s["draft_top1"] <= 1 and 0 < s["draft_paccept"] < 1
    assert s["draft_update_norm"] > 0 and s["draft_applied"] and s["adapter_norm"] == off["step"]["adapter_norm"]
    assert on["saved"]["tensors"] == off["saved"]["tensors"] and on["saved"]["draft_modules"] == 9
    assert "draft_lora.safetensors" in on["files"] and "draft_lora.safetensors" not in off["files"]
    chunked = one_process(chunked=True)["step"]  # the chunked LM head (hidden rows) gives the same targets
    for k in ("draft_kl", "draft_paccept"):
        assert abs(chunked[k] - s[k]) <= 1e-5 * abs(s[k]), k
    assert chunked["draft_top1"] == s["draft_top1"] and chunked["draft_rows"] == s["draft_rows"]


def steps(n, lr_scale=None):
    """``n`` accumulate + step rounds on one trainer (draft on at ``lr_scale``, None = off), deterministic: per round
    the policy's accumulated gradients before the step, its divided gradients (the kept reference) and its parameters
    after the step."""
    import tempfile

    import rl1_draft
    import rl1_trainer as rt

    kept = (rt.SPARSE["mode"], rt.TIES["mode"], rt.fq.positions_and_images, torch.cuda.synchronize)
    det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with base.isolated(), tempfile.TemporaryDirectory() as folder:
            t = trainer(folder, draft=False)
            if lr_scale is not None:
                t.attach_draft(rl1_draft.Draft(head(), weight=1.0), lr_scale)
                t.optimizer = t.make_optimizer(LR)
            draft0 = [p.detach().clone() for _, p in t.draft_params()] if lr_scale is not None else []
            rounds = []
            for _ in range(n):
                acc = t.op_accumulate({"attempts": specs(), "lr": LR, "first": True})
                assert acc["ok"], acc
                raw = [p.grad.detach().clone() for _, p in t.policy_params() if p.grad is not None]
                step = t.op_step({"lr": LR, "ref": "keep"})
                assert step["ok"], step
                rounds.append({"raw": raw, "grads": [g.clone() for g in t.ref["grads"]], "step": step,
                               "params": {n_: p.detach().clone() for n_, p in t.policy_params()},
                               "draft": [p.detach().clone() for _, p in t.draft_params()] if lr_scale else [],
                               "draft0": draft0, "acc": acc})
            return rounds
    finally:
        torch.use_deterministic_algorithms(det)
        rt.SPARSE["mode"], rt.TIES["mode"], rt.fq.positions_and_images, torch.cuda.synchronize = kept


def test_two_steps_policy_bitwise_with_the_draft_on_or_off():
    """T1b (26 Oct, B300): step 2's policy loss differed ON vs OFF (1e-3) and between draft lr scales 1 and 3. On CPU,
    deterministic: the policy's gradients before each step and its parameters after steps 1 and 2 are bitwise those
    of the draft-off run at both scales, so the draft adds nothing to the policy; the GPU gap is the backward's floor
    (the same-process repeat: grad cos 0.9999) through Adam's first steps."""
    off = steps(2)
    for scale in (1.0, 3.0):
        on = steps(2, scale)
        for k, (a, b) in enumerate(zip(on, off)):
            assert len(a["raw"]) == len(b["raw"]) and all(torch.equal(x, y) for x, y in zip(a["raw"], b["raw"])), (scale, k)
            assert all(torch.equal(x, y) for x, y in zip(a["grads"], b["grads"])), (scale, k)
            assert a["params"].keys() == b["params"].keys()
            assert all(torch.equal(a["params"][n], b["params"][n]) for n in a["params"]), (scale, k)
            for key in ("loss", "grad_norm", "update_norm"):
                assert a["step"][key] == b["step"][key], (scale, k, key)
            assert a["step"]["draft_applied"] and a["step"]["draft_update_norm"] > 0
        assert on[1]["step"]["draft_kl"] != on[0]["step"]["draft_kl"]  # the draft did move
        call = on[0]["acc"]["draft_call"]  # the accumulate's own draft read (T1b-v2's held-out window)
        assert call["rows"] == on[0]["step"]["draft_rows"] and call["kl"] == round(on[0]["step"]["draft_kl"], 6)


@base.isolate
def test_frozen_policy_steps_train_only_the_draft(tmp_path):
    """mg.redesign.freeze_policy (lora-l2 GPU leg: MTP acceptance and gen tok/s at one policy): "policy_frozen" steps
    the policy's group at lr 0, so its parameters stay bitwise; the draft's group steps at lr x its scale, as unfrozen.
    A trainer without a draft refuses the flag (it would train nothing)."""
    import rl1_draft
    import rl1_trainer as rt

    kept = (rt.SPARSE["mode"], rt.TIES["mode"], rt.fq.positions_and_images, torch.cuda.synchronize)
    det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        runs = {}
        for frozen in (False, True):
            t = trainer(tmp_path / f"f{int(frozen)}", draft=False)
            t.attach_draft(rl1_draft.Draft(head(), weight=1.0), 3.0)
            t.optimizer = t.make_optimizer(LR)
            policy0 = {n: p.detach().clone() for n, p in t.policy_params()}
            drafts = []
            for _ in range(2):
                assert t.op_accumulate({"attempts": specs(), "lr": LR, "first": True, "policy_frozen": frozen})["ok"]
                step = t.op_step({"lr": LR, "policy_frozen": frozen})
                assert step["ok"], step
                drafts.append([p.detach().clone() for _, p in t.draft_params()])
            runs[frozen] = {"step": step, "policy0": policy0, "policy": dict(t.policy_params()), "draft": drafts[0]}
        on = runs[True]
        assert all(torch.equal(on["policy"][n], on["policy0"][n]) for n in on["policy0"])
        assert on["step"]["update_norm"] == 0 and on["step"]["draft_update_norm"] > 0
        # step 1 (same policy, same windows): the draft's update is the unfrozen run's
        assert all(torch.equal(a, b) for a, b in zip(on["draft"], runs[False]["draft"]))
        assert any(not torch.equal(runs[False]["policy"][n], on["policy0"][n]) for n in on["policy0"])
        bare = trainer(tmp_path / "bare", draft=False)
        with pytest.raises(ValueError, match="policy_frozen"):
            bare.op_update({"attempts": specs(), "lr": LR, "policy_frozen": True})
    finally:
        torch.use_deterministic_algorithms(det)
        rt.SPARSE["mode"], rt.TIES["mode"], rt.fq.positions_and_images, torch.cuda.synchronize = kept


@base.isolate
def test_policy_only_checkpoint_resumes_with_the_draft(tmp_path):
    import rl1_trainer as rt

    kept = (rt.SPARSE["mode"], rt.TIES["mode"], rt.fq.positions_and_images, torch.cuda.synchronize)
    try:
        t0 = trainer(tmp_path / "a", draft=False)
        t0.op_update({"attempts": specs(), "lr": LR})
        t0.op_checkpoint({"path": str(tmp_path / "ckpt")})
        t1 = trainer(tmp_path / "b")
        draft_before = {n: p.detach().clone() for n, p in t1.draft_params()}
        out = t1.op_load_checkpoint({"path": str(tmp_path / "ckpt")})
        assert out["updates"] == 1
        for (n, p), (m, q) in zip(t0.params, t1.policy_params()):
            assert n == m and torch.equal(p, q)
        for n, p in t1.draft_params():
            assert torch.equal(p, draft_before[n])
        step = t1.op_update({"attempts": specs(), "lr": LR})  # Adam state for the policy, fresh for the draft
        assert step["ok"] and step["draft_rows"] > 0
        t1.op_checkpoint({"path": str(tmp_path / "ckpt2")})  # with the draft: loads back exactly
        t2 = trainer(tmp_path / "c")
        t2.op_load_checkpoint({"path": str(tmp_path / "ckpt2")})
        for (n, p), (m, q) in zip(t1.params, t2.params):
            assert n == m and torch.equal(p, q)
    finally:
        rt.SPARSE["mode"], rt.TIES["mode"], rt.fq.positions_and_images, torch.cuda.synchronize = kept


COMMANDS = [{"op": "accumulate", "lr": LR, "first": True, "fit_tokens": base.FIT},
            {"op": "step", "lr": LR, "ref": "keep", "dist_digest": True}, {"op": "exit"}]


def serve_rank(rank, world, folder):
    import os

    os.environ["RL1_DIST"] = "1"
    os.environ["RL1_DIST_POLL_S"] = "0.05"
    t = trainer(folder)
    t.serve()
    return {"grads": [g.numpy() for g in t.ref["grads"]], "draft": [g.numpy() for g in t.draft_grads]}


@pytest.mark.multirank
def test_dp_cp_draft_matches_one_process(tmp_path):
    import numpy as np

    ref = one_process()
    (tmp_path / "cmd").mkdir(parents=True)
    for i, cmd in enumerate(COMMANDS):
        cmd = {**cmd, "attempts": specs()} if cmd["op"] == "accumulate" else cmd
        (tmp_path / "cmd" / f"{i:04d}.json").write_text(json.dumps(cmd))
    out = run_ranks(serve_rank, world=3, inputs=[str(tmp_path)] * 3, timeout=300)
    acc, step, _ = [json.loads((tmp_path / "done" / f"{i:04d}.json").read_text()) for i in range(3)]
    assert acc["ok"] and step["ok"], (acc.get("error"), step.get("error"))
    assert [r["dist_plan"] for r in acc["ranks"]] == [{"cp": 1, "dp": 1, "load": [200, 100, 0]},
                                                      {"cp": 1, "dp": 1, "load": [200, 100, 0]},
                                                      {"cp": 1, "dp": 0, "load": [200, 100, 0]}]
    assert step["dist_digest"]["equal"]
    s, r = step, ref["step"]
    assert s["draft_rows"] == r["draft_rows"] and s["draft_windows"] == r["draft_windows"] == 3
    assert s["draft_top1"] == r["draft_top1"] and s["draft_agree"] == r["draft_agree"]
    for k in ("draft_kl", "draft_paccept", "draft_grad_norm"):
        assert abs(s[k] - r[k]) <= 1e-4 * abs(r[k]), (k, s[k], r[k])
    worst = {"policy": 0.0, "draft": 0.0}
    for rank_out in out:
        for key, mine, want in (("policy", rank_out["grads"], ref["grads"]), ("draft", rank_out["draft"], ref["draft"])):
            assert len(mine) == len(want)
            for got, w in zip(mine, want):
                worst[key] = max(worst[key], float(np.abs(got - w).max()) / (float(np.abs(w).max()) or 1.0))
    assert worst["policy"] <= 1e-4 and worst["draft"] <= 1e-4, worst
    print(f"dist vs one process: max rel grad err {worst}")


def test_trainer_command_draft_env():
    from arena import rl_loop

    _, env = rl_loop.trainer_command(rl_loop.DEFAULTS | {"mg": {"draft_weight": 0.5, "draft_lr_scale": 4}}, "/s", "/w")
    assert env["RL1_DRAFT_WEIGHT"] == "0.5" and env["RL1_DRAFT_LR_SCALE"] == "4.0"
    _, off = rl_loop.trainer_command(rl_loop.DEFAULTS | {"mg": {"draft_weight": 0}}, "/s", "/w")
    assert "RL1_DRAFT_WEIGHT" not in off
    rl_loop.check_mg({"draft_weight": 1.0, "draft_lr_scale": 2.0})
