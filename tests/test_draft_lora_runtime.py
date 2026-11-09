"""draft_lora_runtime (ARC_DRAFT_LORA): the served draft's weights through vLLM's adapter lifecycle (read at load,
merge at activation, restore at deactivation), on a stand-in draft module with the pinned MTP's parameter names (TP 1,
packed q | k | v and gate | up). Exact on a reload under a new id, tensors never reallocated."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc_infra import draft_lora_runtime as R  # noqa: E402

H, Q, KV, I, R_ = 64, 96, 16, 24, 4  # hidden, q rows (with gate), k / v rows, shared expert width, rank
MODULES = {  # rl1_draft.export's json (vllm_module mapping)
    "fc_embedding": ("model.fc_embedding", None, H, H),
    "fc_hidden": ("model.fc_hidden", None, H, H),
    "layers.0.self_attn.q_proj": ("model.layers.0.self_attn.qkv_proj", "q", Q, H),
    "layers.0.self_attn.k_proj": ("model.layers.0.self_attn.qkv_proj", "k", KV, H),
    "layers.0.self_attn.v_proj": ("model.layers.0.self_attn.qkv_proj", "v", KV, H),
    "layers.0.self_attn.o_proj": ("model.layers.0.self_attn.o_proj", None, H, Q // 2),
    "layers.0.mlp.shared_expert.gate_proj": ("model.layers.0.mlp.shared_expert.gate_up_proj", 0, I, H),
    "layers.0.mlp.shared_expert.up_proj": ("model.layers.0.mlp.shared_expert.gate_up_proj", 1, I, H),
    "layers.0.mlp.shared_expert.down_proj": ("model.layers.0.mlp.shared_expert.down_proj", None, H, I),
}
SHAPES = {"model.fc_embedding": (H, H), "model.fc_hidden": (H, H), "model.layers.0.self_attn.qkv_proj": (Q + 2 * KV, H),
          "model.layers.0.self_attn.o_proj": (H, Q // 2), "model.layers.0.mlp.shared_expert.gate_up_proj": (2 * I, H),
          "model.layers.0.mlp.shared_expert.down_proj": (H, I)}


class Draft(torch.nn.Module):
    """The draft's dense params under their vLLM names, plus an untouched one (an expert stand-in)."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        for name, shape in SHAPES.items():
            mod = self
            for part in name.split("."):
                if not hasattr(mod, part):
                    mod.add_module(part, torch.nn.Module())
                mod = getattr(mod, part)
            mod.weight = torch.nn.Parameter((torch.randn(shape, generator=g) * 0.05).to(torch.bfloat16),
                                            requires_grad=False)
        self.experts = torch.nn.Parameter(torch.zeros(3, 2, dtype=torch.uint8), requires_grad=False)

    def state(self):
        return {k: v.detach().clone() for k, v in self.named_parameters()}


def write_adapter(folder, seed, scale=1.0, r=R_, alpha=2 * R_):
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    tensors, modules = {}, {}
    for name, (param, shard, out, inp) in MODULES.items():
        tensors[R.key(name, "A")] = (torch.randn(r, inp, generator=g) * 0.1).to(torch.bfloat16)
        tensors[R.key(name, "B")] = (torch.randn(out, r, generator=g) * 0.1 * scale).to(torch.bfloat16)
        modules[name] = {"vllm": param, "shard": shard}
    save_file(tensors, str(folder / "draft_lora.safetensors"))
    meta = {"format": R.FORMAT, "r": r, "lora_alpha": float(alpha), "modules": modules}
    (folder / "draft_lora.json").write_text(json.dumps(meta))  # shards 0 / 1 come back as ints, q / k / v as strs
    return folder


def reference(base, folder):
    """base + alpha / r x B A per shard, full fp32 matmuls, one BF16 rounding (independent of R.merge)."""
    from safetensors.torch import load_file

    meta = json.loads((folder / "draft_lora.json").read_text())
    t = load_file(str(folder / "draft_lora.safetensors"))
    out = {k: v.float().clone() for k, v in base.items()}
    offsets = {"q": 0, "k": Q, "v": Q + KV, 0: 0, 1: I, None: 0}
    for name, spec in meta["modules"].items():
        d = meta["lora_alpha"] / meta["r"] * (t[R.key(name, "B")].float() @ t[R.key(name, "A")].float())
        start = offsets[spec["shard"]]
        out[spec["vllm"] + ".weight"][start:start + d.shape[0]] += d
    return {k: v.to(base[k].dtype) for k, v in out.items()}


def lora(lora_id, folder):
    return SimpleNamespace(id=lora_id, _arc_draft=R.read(folder) if folder is not None else None)


@pytest.fixture
def draft():
    model = Draft()
    R.register_draft(model)
    yield model
    R._STATE.update(model=None, base={}, active=None, warned=False)


def ptrs(model):
    return {k: v.data_ptr() for k, v in model.named_parameters()}


def test_merge_matches_reference_and_restores_exactly(draft, tmp_path, capsys):
    base, where = draft.state(), ptrs(draft)
    a = write_adapter(tmp_path / "a", seed=1)
    R.activate(lora(1, a))
    want = reference(base, a)
    for k, v in draft.named_parameters():
        assert torch.allclose(v.float(), want[k].float(), atol=2 ** -8, rtol=2 ** -7), k
    assert torch.equal(draft.experts, base["experts"])
    assert any(not torch.equal(v, base[k]) for k, v in draft.named_parameters())
    merged = capsys.readouterr().out
    assert "ARC_DRAFT_LORA_MERGED" in merged and '"modules": 9' in merged
    R.deactivate(1)
    assert all(torch.equal(v, base[k]) for k, v in draft.named_parameters())  # host copy back, bit for bit
    assert ptrs(draft) == where


def test_reload_under_a_new_id_is_exact(draft, tmp_path):
    """v1 -> v2 by unload/load (deactivate then activate), by LRU eviction (activate over an active one) and by
    remove_all: every path serves the same bits as v2 merged once onto the base."""
    base, where = draft.state(), ptrs(draft)
    a, b = write_adapter(tmp_path / "a", seed=1), write_adapter(tmp_path / "b", seed=2, scale=3.0)
    R.activate(lora(7, b))
    once = draft.state()
    R.deactivate_all()
    assert all(torch.equal(v, base[k]) for k, v in draft.named_parameters())

    R.activate(lora(1, a))
    R.deactivate(1)
    R.activate(lora(2, b))
    assert all(torch.equal(v, once[k]) for k, v in draft.named_parameters())
    R.activate(lora(3, a))  # LRU: activation over an active adapter (its deactivate may come first or not at all)
    R.activate(lora(4, b))
    assert all(torch.equal(v, once[k]) for k, v in draft.named_parameters())
    R.deactivate(3)  # a stale id: no change
    assert all(torch.equal(v, once[k]) for k, v in draft.named_parameters())
    assert ptrs(draft) == where


def test_dummy_and_policy_only_adapters_serve_the_base(draft, tmp_path):
    base = draft.state()
    R.activate(lora(1, write_adapter(tmp_path / "a", seed=1)))
    R.activate(lora(2, None))  # add_dummy_lora / an adapter folder without draft files
    assert all(torch.equal(v, base[k]) for k, v in draft.named_parameters())
    assert R.read(tmp_path) is None


def test_chunked_rows_match_one_slice(draft, tmp_path, monkeypatch):
    a = write_adapter(tmp_path / "a", seed=5)
    R.activate(lora(1, a))
    whole = draft.state()
    R.deactivate(1)
    monkeypatch.setattr(R, "CHUNK", 7)  # slices crossing shard and chunk edges
    R.activate(lora(2, a))
    for k, v in draft.named_parameters():
        assert torch.allclose(v.float(), whole[k].float(), atol=2 ** -9), k


def test_wrapper_prefix_and_bad_shapes(tmp_path):
    outer = torch.nn.Module()
    outer.model = Draft()  # a wrapper: names gain a prefix, matched by suffix
    R.register_draft(outer)
    try:
        base = {k: v.clone() for k, v in outer.state_dict().items()}
        R.activate(lora(1, write_adapter(tmp_path / "a", seed=1)))
        assert not torch.equal(outer.model.model.fc_hidden.weight, base["model.model.fc_hidden.weight"])
        R.deactivate(1)
        bad = write_adapter(tmp_path / "bad", seed=1, r=R_)
        meta = json.loads((bad / "draft_lora.json").read_text())
        meta["modules"]["layers.0.self_attn.o_proj"]["vllm"] = "model.fc_hidden"  # B / A shapes do not fit
        (bad / "draft_lora.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError):
            R.activate(lora(2, bad))
        assert all(torch.equal(v, base[k]) for k, v in outer.state_dict().items())  # checked before any write
    finally:
        R._STATE.update(model=None, base={}, active=None, warned=False)


def test_read_refuses_partial_files(tmp_path):
    a = write_adapter(tmp_path / "a", seed=1)
    meta = json.loads((a / "draft_lora.json").read_text())
    meta["modules"]["extra"] = {"vllm": "model.fc_hidden", "shard": None}
    (a / "draft_lora.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="tensors"):
        R.read(a)
    meta["format"] = "other"
    (a / "draft_lora.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="format"):
        R.read(a)


def test_no_draft_registered_warns_once(tmp_path, capsys):
    R._STATE.update(model=None, base={}, active=None, warned=False)
    a = write_adapter(tmp_path / "a", seed=1)
    R.activate(lora(1, a))
    R.activate(lora(2, a))
    assert capsys.readouterr().out.count("ARC_DRAFT_LORA_NO_DRAFT") == 1
    R._STATE["warned"] = False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay needs a GPU")
def test_cuda_graph_reads_the_merged_weights(tmp_path):
    model = Draft().cuda()
    R.register_draft(model)
    try:
        w = model.model.fc_hidden.weight
        x = torch.randn(4, H, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = x @ w.t()
        R.activate(lora(1, write_adapter(tmp_path / "a", seed=1)))
        graph.replay()
        assert torch.equal(y, x @ w.t())
    finally:
        R._STATE.update(model=None, base={}, active=None, warned=False)
