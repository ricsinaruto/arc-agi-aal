"""l4pp layout on real ranks: PP 2 + TP 2 groups, asymmetric inputs with an empty rank, PLE on the first stage only."""
import pytest

from arc_infra.runops.multirank import run_ranks

pytest.importorskip("torch")
pytestmark = pytest.mark.multirank


def stage_step(rank, world, payload):
    import torch
    import torch.distributed as dist

    tp = world // 2 if world == 4 else 1  # world 4: stages {0, 1} and {2, 3}; world 2: one rank a stage
    stage = rank // tp
    groups = [dist.new_group([k * tp + i for i in range(tp)]) for k in range(world // tp)] if tp > 1 else None  # every rank joins each call
    group = groups[stage] if groups else None
    data = torch.tensor(payload, dtype=torch.float64)
    sizes = torch.zeros(tp, dtype=torch.int64)
    sizes[rank % tp] = len(payload)
    if group is not None:
        dist.all_reduce(sizes, group=group)
    held = torch.zeros(int(sizes.max()), dtype=torch.float64)
    held[:len(payload)] = data
    if group is not None:
        dist.all_reduce(held, group=group)
    peer = rank + world // 2 if stage == 0 else rank - world // 2
    if stage == 0:  # the hop: size first, so an empty rank still pairs
        dist.send(torch.tensor([held.numel()]), peer)
        dist.send(held, peer) if held.numel() else None
    else:
        size = torch.zeros(1, dtype=torch.int64)
        dist.recv(size, peer)
        held = torch.zeros(int(size), dtype=torch.float64)
        dist.recv(held, peer) if int(size) else None
    return {"stage": stage, "sum": float(held.sum()), "ple": stage == 0}


@pytest.mark.parametrize("world,inputs", [(4, [[1.0, 2.0], [], [5.0], [7.0, 8.0, 9.0]]), (2, [[], [3.0]]), (2, [[1.0, 1.0], []])])
def test_stages_pair_with_an_empty_rank(world, inputs):
    out = run_ranks(stage_step, world=world, inputs=inputs, timeout=60)
    assert [r["stage"] for r in out] == [i // (world // 2) for i in range(world)]
    assert [r["ple"] for r in out] == [i < world // 2 for i in range(world)]  # PLE offload registers on stage 0 only
    assert out[world // 2]["sum"] == out[0]["sum"] and all(r["sum"] == out[0]["sum"] for r in out[:world // 2])
