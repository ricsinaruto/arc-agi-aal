"""Hero run: the trainer serves the t37 model as an additive router mask. Masked routing (softmax,
top-k, renormalize over all experts with MASK_VALUE on the pruned ones) equals the renumbered cut's routing (the kept
experts only: expert_prune_runtime's semantics), and install_expert_mask attaches one mask per layer."""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))

import rl1_trainer  # noqa: E402


def route(logits, k):
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    value, index = torch.topk(probs, k, dim=-1)
    return value / value.sum(-1, keepdim=True), index


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_masked_routing_equals_the_cut_models(dtype):
    g = torch.Generator().manual_seed(0)
    E, k, T = 64, 10, 200
    # distinct per row in either dtype (bf16 ties would make top-k order depend on column positions)
    logits = torch.stack([(torch.randperm(E, generator=g) - E / 2) / 8 for _ in range(T)]).to(dtype)
    pruned = sorted(torch.randperm(E, generator=g)[:24].tolist())
    kept = [e for e in range(E) if e not in set(pruned)]
    mask = torch.zeros(E)
    mask[pruned] = rl1_trainer.MASK_VALUE
    w_mask, i_mask = route(rl1_trainer.mask_logits(logits, mask), k)
    w_cut, i_cut = route(logits[:, kept], k)
    i_cut = torch.tensor(kept)[i_cut]  # renumbered ids back to the checkpoint's
    assert not set(i_mask.flatten().tolist()) & set(pruned)
    assert torch.equal(i_mask.sort(-1).values, i_cut.sort(-1).values)
    order_m, order_c = i_mask.argsort(-1), i_cut.argsort(-1)
    torch.testing.assert_close(w_mask.gather(-1, order_m), w_cut.gather(-1, order_c), rtol=0, atol=2e-6)
    assert rl1_trainer.mask_logits(logits, None) is logits


class Qwen4ExpTextTopKRouter(torch.nn.Module):
    def __init__(self, experts, top_k=10):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(experts, 8))
        self.top_k = top_k


def model_of(layers, experts):
    root = torch.nn.Module()
    root.layers = torch.nn.ModuleList()
    for _ in range(layers):
        block = torch.nn.Module()
        block.mlp = torch.nn.Module()
        block.mlp.gate = Qwen4ExpTextTopKRouter(experts)
        root.layers.append(block)
    return root


def mask_file(tmp_path, pruned, experts=32):
    path = tmp_path / "mask.json"
    path.write_text(json.dumps({"format": "arc-expert-prune-v1", "experts": experts, "fraction": 0.375,
                                "pruned": {str(k): v for k, v in pruned.items()}}))
    return path


def test_install_attaches_each_layers_mask(tmp_path):
    model = model_of(3, 32)
    pruned = {0: [1, 2, 3], 1: [], 2: list(range(20, 32))}
    got = rl1_trainer.install_expert_mask(model, mask_file(tmp_path, pruned), router_class="Qwen4ExpTextTopKRouter",
                                         patch=False)
    assert got["layers"] == 3 and got["pruned"] == 15
    for layer, ids in pruned.items():
        m = model.layers[layer].mlp.gate._arc_mask
        assert m.shape == (32,) and sorted(torch.nonzero(m).flatten().tolist()) == ids
        assert set(m[ids].tolist()) <= {rl1_trainer.MASK_VALUE}


@pytest.mark.parametrize("pruned, why", [({0: [1], 1: [2]}, "no mask row"), ({0: [1], 1: [2], 2: [3], 3: [4]}, "routers for"),
                                         ({0: [1, 1], 1: [], 2: []}, "bad pruned"), ({0: list(range(25)), 1: [], 2: []}, "bad pruned")])
def test_install_refuses_a_mismatched_mask(tmp_path, pruned, why):
    with pytest.raises(ValueError, match=why):
        rl1_trainer.install_expert_mask(model_of(3, 32), mask_file(tmp_path, pruned), patch=False)
    with pytest.raises(ValueError, match="router"):
        rl1_trainer.install_expert_mask(model_of(3, 16), mask_file(tmp_path, {0: [], 1: [], 2: []}), patch=False)
