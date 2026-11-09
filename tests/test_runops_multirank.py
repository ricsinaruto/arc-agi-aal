"""The gloo multi-rank helper: asymmetric per-rank inputs, a skipped collective (hang) and a rank error are failures
naming the rank, within the timeout."""
import pytest

torch = pytest.importorskip("torch")

from arc_infra.runops.multirank import RankFailure, run_ranks  # noqa: E402

pytestmark = pytest.mark.multirank


def mean_over_ranks(rank, world, batch):
    """Correct: every rank joins both collectives, also with an empty batch."""
    import torch.distributed as dist

    s = torch.tensor([float(sum(batch)), float(len(batch))])
    dist.all_reduce(s)
    return (s[0] / s[1]).item()


def skips_when_empty(rank, world, batch):
    """The c7f class of bug: a rank with nothing to do returns before the collective its peers wait in."""
    import torch.distributed as dist

    if not batch:
        return None
    s = torch.tensor([float(sum(batch))])
    dist.all_reduce(s)
    return s.item()


def raises_on_rank1(rank, world, batch):
    if rank == 1:
        raise ValueError("bad shard")
    return rank


def test_asymmetric_inputs_agree():
    assert run_ranks(mean_over_ranks, inputs=[[1, 2, 3], [4], []], timeout=20) == [2.5, 2.5, 2.5]


def test_skipped_collective_is_a_named_failure_not_a_hang():
    with pytest.raises(RankFailure) as e:
        run_ranks(skips_when_empty, inputs=[[1], [2], []], timeout=5)
    f = e.value.failures
    assert set(f) == {0, 1} and 2 not in f, f  # rank 2 returned; its peers timed out in all_reduce


def test_rank_error_is_reported_with_its_rank():
    with pytest.raises(RankFailure, match="rank 1: error: ValueError: bad shard"):
        run_ranks(raises_on_rank1, timeout=10)
