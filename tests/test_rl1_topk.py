"""MFU-1: the exact radix top-k (rl1_topk) against the host reference of torch's CUDA selection (radix key order,
ties to the lower index) and, on a GPU, against torch.topk itself. CPU: the Triton interpreter, small shapes."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TRITON_INTERPRET", "1")
torch = pytest.importorskip("torch")
pytest.importorskip("triton")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import rl1_topk  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def tie_heavy(rows, n, seed):
    """As the indexer's scores (relu sums, many exact zeros) plus repeated positives, -0.0, +inf and NaN."""
    g = torch.Generator().manual_seed(seed)
    s = torch.relu(torch.randn(rows, n, generator=g))
    s[torch.rand(rows, n, generator=g) < 0.5] = 0.0
    pick = torch.rand(rows, n, generator=g) < 0.2
    s[pick] = torch.round(s[pick] * 4) / 4
    s[0, : min(7, n)] = -0.0
    s[1, n // 2] = float("inf")
    s[2, n // 3] = float("nan")
    s[3, :] = 0.0  # a whole row of ties
    return s.contiguous()


@pytest.mark.parametrize("n,k,block", [(37, 5, 16), (300, 64, 64), (1000, 128, 256), (513, 512, 128), (64, 64, 64)])
def test_topk_rows_equals_the_reference(n, k, block):
    rows = 12
    s = tie_heavy(rows, n, n + k)
    limit = torch.randint(0, n + 1, (rows,), generator=torch.Generator().manual_seed(k))
    limit[0], limit[1], limit[4], limit[5] = n, 1, 0, k - 1  # full, one valid, none valid, one short of k
    want = rl1_topk.reference_topk(s, k, limit)
    for method in ("bisect", "radix"):
        got = rl1_topk.topk_rows(s.to(DEV), k, limit.to(DEV), block=block, method=method).cpu()
        assert torch.equal(got, want), method


@pytest.mark.parametrize("n,k,step,cand,chunk", [(300, 16, 4, 128, 64), (1000, 64, 4, 512, 256), (1000, 64, 2, 128, 128),
                                                 (777, 32, 8, 512, 128), (513, 512, 4, 512, 128)])
def test_topk_rows_filter_equals_the_reference(n, k, step, cand, chunk, monkeypatch):
    """filter: sample bound, compaction, select; cand small enough that tie-heavy rows overflow into the radix path."""
    monkeypatch.setitem(rl1_topk.FILTER, "step", step)
    monkeypatch.setitem(rl1_topk.FILTER, "cand", cand)
    monkeypatch.setitem(rl1_topk.FILTER, "chunk", chunk)
    rows = 12
    for seed, make in ((1, tie_heavy), (2, lambda r, m, sd: torch.rand(r, m, generator=torch.Generator().manual_seed(sd)))):
        s = make(rows, n, n + k + seed).contiguous()
        limit = torch.randint(0, n + 1, (rows,), generator=torch.Generator().manual_seed(k + seed))
        limit[0], limit[1], limit[4], limit[5], limit[6] = n, 1, 0, k - 1, k + 1  # full, one, none, k-1, k+1 valid
        want = rl1_topk.reference_topk(s, k, limit)
        got = rl1_topk.topk_rows(s.to(DEV), k, limit.to(DEV), method="filter").cpu()
        assert torch.equal(got, want), (seed, (got != want).any(dim=1).nonzero().flatten().tolist())


def test_reference_rule():
    """The reference itself: -0.0 below +0.0, NaN on top, ties to the lower index, masked columns as -inf."""
    s = torch.tensor([[0.0, -0.0, 0.0, 1.0, float("nan"), 0.0]])
    assert rl1_topk.reference_topk(s, 3).tolist() == [[0, 3, 4]]
    assert rl1_topk.reference_topk(s, 5).tolist() == [[0, 2, 3, 4, 5]]
    assert rl1_topk.reference_topk(s, 2, torch.tensor([4])).tolist() == [[0, 3]]
    assert rl1_topk.reference_topk(s, 3, torch.tensor([0])).tolist() == [[0, 1, 2]]


@pytest.mark.skipif(DEV != "cuda", reason="torch.topk's CUDA selection")
@pytest.mark.parametrize("n", [600, 4096, 16384, 30000])
def test_topk_rows_equals_cuda_torch_topk(n):
    s = tie_heavy(1024, n, n).cuda()
    s[:, : n // 2] = torch.where(s[:, : n // 2] > 0, s[:, : n // 2], torch.zeros_like(s[:, : n // 2]))
    limit = torch.clamp(torch.arange(1024, device="cuda") * 4 + (n - 4096), 0, n)
    k = min(512, n)
    masked = s.masked_fill(torch.arange(n, device="cuda")[None, :] >= limit[:, None], float("-inf"))
    want = torch.sort(masked.topk(k, dim=-1).indices, dim=1).values
    for method in ("bisect", "radix", "filter", "torch"):
        assert torch.equal(rl1_topk.topk_rows(s, k, limit, method=method), want), method


def test_trainer_verify_mode_counts_rows():
    """fast_topk "verify": the torch picks drive the pass; the radix picks are compared per row (no tie straddles
    the k-th place here, so CPU torch.topk is the reference too; CPU torch breaks -inf ties its own way, so every
    row keeps >= k valid columns)."""
    import rl1_trainer

    s = torch.rand(16, 300, generator=torch.Generator().manual_seed(5)).contiguous()
    limit = torch.tensor([300, 200, 40, 33, 32, 32, 290, 64] * 2)
    valid = torch.arange(300)[None, :] < limit[:, None]
    before = dict(rl1_trainer.TOPK)
    picked = rl1_trainer.topk_verify(s, valid, 32, limit, rl1_topk)
    assert torch.equal(picked, s.masked_fill(~valid, float("-inf")).topk(32, dim=-1).indices)
    assert rl1_trainer.TOPK["verified"] == before["verified"] + 1 and rl1_trainer.TOPK["rows"] == before["rows"] + 16
    assert rl1_trainer.TOPK["bad_rows"] == before["bad_rows"]
