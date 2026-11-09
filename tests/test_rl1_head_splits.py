"""U2 (27 Oct): the chunked head with host-built row splits (no mask test, no per-chunk sync) gives the values and
gradients of the card-mask path bitwise: actor, valid, raw, entropy and d(loss)/d(hidden), on rows mixing whole-vocab
rows (support -1) and sampling-set rows, a chunk of only one kind, and a token outside its set (invalid)."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/posttrain"))


def window(seed=0, n=37, h=16, v=50, k=5):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(n, h, generator=g)
    weight = torch.randn(v, h, generator=g) * 0.3
    tokens = torch.randint(0, v, (n,), generator=g)
    support = torch.full((n, k), -1, dtype=torch.long)
    for i in range(n):
        if i % 3 or 8 <= i < 16:  # rows 8..15: a chunk of only set rows
            ids = torch.randperm(v, generator=g)[:k]
            if i % 7 != 0:  # the sampled token inside its set (else: invalid)
                ids[int(torch.randint(0, k, (1,), generator=g))] = tokens[i]
            ids[k - 1 - (i % 2)] = -1 if i % 5 == 0 else ids[k - 1 - (i % 2)]  # some padded sets
            support[i] = ids
    weights = torch.randn(n, generator=g)
    return hidden, weight, tokens, support, weights


def run(split, seed):
    import rl1_memory

    hidden, weight, tokens, support, weights = window(seed)
    hidden = hidden.clone().requires_grad_(True)
    project = lambda x: x @ weight.T  # noqa: E731
    out = rl1_memory.chunked_head(project, hidden, tokens, support, 0.7, chunk=8,
                                  support_host=support if split else None)
    loss = -(weights * out["actor"] * out["valid"]).sum()
    loss.backward()
    return out, hidden.grad, loss.detach()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_host_splits_match_the_card_masks_bitwise(seed):
    a, ga, la = run(False, seed)
    b, gb, lb = run(True, seed)
    assert torch.equal(a["actor"], b["actor"]) and torch.equal(a["valid"], b["valid"])
    assert torch.equal(a["raw"], b["raw"]) and a["entropy"] == b["entropy"]
    assert torch.equal(ga, gb) and torch.equal(la, lb)
    assert not bool(a["valid"].all())  # the case with an invalid row is in the window


def test_actor_logprobs_split_matches_masks():
    import rl_job

    hidden, weight, tokens, support, _ = window(3)
    rows = hidden @ weight.T
    whole = support[:, 0] < 0
    split = (torch.nonzero(whole).flatten(), torch.nonzero(~whole).flatten())
    a = rl_job.actor_logprobs(rows, tokens, support, 0.9)
    b = rl_job.actor_logprobs(rows, tokens, support, 0.9, split=split)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_entropy_read_once_matches_the_per_chunk_reads():
    """Entropy (sum, count) read once per window vs the old per-chunk host reads: count exact, sum within 1e-12
    relative (float64 sums in a different order)."""
    import rl1_trainer

    hidden, weight, tokens, support, _ = window(4)
    b, _, _ = run(True, 4)
    rows = (hidden @ weight.T).detach()
    total, count = 0.0, 0
    for i in range(0, rows.shape[0], 8):
        s, c = rl1_trainer.token_entropy(rows[i:i + 8], b["valid"][i:i + 8], 0.7)
        total, count = total + s, count + c
    assert b["entropy"][1] == count and abs(b["entropy"][0] - total) <= 1e-12 * abs(total)
