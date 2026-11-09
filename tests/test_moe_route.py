"""Fused MoE routing (kernel profile project 2; docs/validation-20260930/kernel-profile.md), under the Triton
interpreter (``TRITON_INTERPRET=1``, Linux container as tests/test_dense_fused.py; skipped where Triton is missing).

Against the torch reference of the pinned semantics (``topk_softmax`` with renormalization, ``moe_align_block_size``):
identical top-k ids, weights to FP32 rounding, the same padded total and per-block experts, and per expert the same
set of (token x k) pairs followed by padding. FP32 logits: this interpreter build mis-reads BF16 tensors (see
tests/test_dense_wide.py); the kernel upcasts either dtype on load."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TRITON_INTERPRET", "1")
torch = pytest.importorskip("torch")
pytest.importorskip("triton")

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("arc_moe_route_under_test", ROOT / "src/arc_infra/moe_route_runtime.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


route = load()


def check(m, experts=448, top_k=10, block=16, seed=0, skew=0.0):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn((m, experts), generator=generator) * 2.0
    if skew:
        # Correlated routing (a few experts favoured by every token), as real decode batches are.
        logits[:, :24] += skew
    w, ids, sorted_ids, expert_ids, ntpp = route.route_align(logits, top_k, block)
    rw, rids, rsorted, rexperts, rtotal = route.reference(logits, top_k, block)
    assert torch.equal(ids, rids)
    assert torch.allclose(w, rw, rtol=2e-6, atol=1e-7)
    assert torch.allclose(w.sum(-1), torch.ones(m), atol=1e-6)
    total = int(ntpp.item())
    assert total == rtotal and total % block == 0
    assert torch.equal(expert_ids[:total // block], rexperts)
    assert bool((expert_ids[total // block:] == -1).all())
    assert bool((sorted_ids[total:] == m * top_k).all())
    # Per block: the same pairs as the reference (order within an expert may differ), padding last.
    for b in range(total // block):
        got = sorted_ids[b * block:(b + 1) * block]
        want = rsorted[b * block:(b + 1) * block]
        assert sorted(got.tolist()) == sorted(want.tolist())
    # Every pair appears exactly once.
    pairs = sorted_ids[:total]
    assert sorted(pairs[pairs < m * top_k].tolist()) == list(range(m * top_k))
    return total


@pytest.mark.parametrize("m", [1, 7, 16, 33, 40, 56, 64])
def test_matches_the_pinned_semantics(m):
    check(m, seed=m)


def test_correlated_routing_and_multi_block_experts():
    # Most tokens share the favoured experts: counts above one block per expert (up to 64 / 16 = 4 blocks).
    total = check(56, skew=6.0, seed=3)
    assert total > 56 * 10  # padding present


def test_unpruned_router_and_other_block_sizes():
    check(40, experts=512, seed=5)
    check(24, experts=448, block=32, seed=6)
    check(48, experts=448, top_k=8, block=16, seed=7)
    # W4A16 Marlin with BF16 activations picks block_size_m 8 at every decode size (M x 10 / 448 / 8 < 0.9).
    check(40, experts=448, block=8, seed=8)
    check(64, experts=448, block=8, seed=9, skew=6.0)


def test_rejects_batches_above_the_decode_range():
    with pytest.raises(ValueError):
        route.route_align(torch.zeros((65, 448)), 10, 16)
