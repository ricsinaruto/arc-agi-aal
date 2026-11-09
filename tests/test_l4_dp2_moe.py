"""l4kv: the DP 2 + TP 2 + wide EP expert sum on 4 gloo ranks, against one process holding every expert.

Runs the served vLLM (runtime on PYTHONPATH, dp2 applied): real groups, ParallelConfig, expert map,
MoERunner, kernel, naive DP/EP prepare/finalize, AgRsAll2AllManager and CudaCommunicator. Stand-ins: the experts' maths,
the top-k and sum (C ops) and NCCL (gloo, pynccl's semantics). Skipped where vllm is absent (CI); run by scripts/modal_arc2_l4.py::dp2_sum_main.
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from arc_infra.runops.multirank import run_ranks  # noqa: E402

pytestmark = pytest.mark.multirank

EXPERTS, TOPK, HIDDEN, WORLD = 128, 8, 16, 4  # 128 experts, 32 a card: the served pruned model
ROWS = {"unequal": (5, 3), "equal": (4, 4), "idle_second": (7, 1), "idle_first": (1, 7), "empty_second": (6, 0),
        "empty_first": (0, 6), "one_row": (1, 1), "prefill": (256, 1)}


class GlooNccl:
    """pynccl's calls on a gloo group."""
    disabled = False

    def __init__(self, group, ranks, rank_in_group):
        self.group, self.ranks, self.rank = group, ranks, rank_in_group
        self.world_size = len(ranks)

    def group_start(self):
        pass

    group_end = group_start

    def all_reduce(self, tensor):
        out = tensor.clone()
        torch.distributed.all_reduce(out, group=self.group)
        return out

    def all_gatherv(self, output, tensor, sizes):
        start = 0
        for root, size in enumerate(sizes):
            part = output[start:start + size]
            if size:
                if root == self.rank:
                    part.copy_(tensor)
                torch.distributed.broadcast(part, src=self.ranks[root], group=self.group)
            start += size

    def all_gather(self, output, tensor):
        self.all_gatherv(output, tensor, [len(tensor)] * self.world_size)

    def reduce_scatterv(self, output, tensor, sizes):
        total = self.all_reduce(tensor)
        start = sum(sizes[:self.rank])
        output.copy_(total[start:start + sizes[self.rank]])

    def reduce_scatter(self, output, tensor):
        self.reduce_scatterv(output, tensor, [len(tensor) // self.world_size] * self.world_size)


def fake_communicator(group, all2all=None):
    """The real wrappers (all_gatherv, reduce_scatterv, all_reduce, dispatch, combine) over GlooNccl."""
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

    comm = CudaCommunicator.__new__(CudaCommunicator)
    comm.__dict__.update(cpu_group=group.cpu_group, device_group=group.device_group, world_size=group.world_size,
                         rank_in_group=group.rank_in_group, ranks=group.ranks, device=torch.device("cpu"),
                         pynccl_comm=GlooNccl(group.device_group, group.ranks, group.rank_in_group), qr_comm=None,
                         fi_ar_comm=None, aiter_ar_comm=None, ca_comm=None, symm_mem_comm=None,
                         all2all_manager=all2all)
    return comm


def weights():
    made = torch.Generator().manual_seed(11)
    return (torch.randn(EXPERTS, HIDDEN, HIDDEN, generator=made), torch.randn(HIDDEN, HIDDEN, generator=made),
            torch.randn(HIDDEN, EXPERTS, generator=made))


class Router:
    def select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
        weights_, ids = router_logits.softmax(-1).topk(TOPK, dim=-1)
        return weights_ / weights_.sum(-1, keepdim=True), ids


def experts_out(table, x, ids):
    return torch.einsum("mkij,mj->mki", table[ids], x.tanh())


def expected(x, router):
    """One process, every expert."""
    table, shared, gate = weights()
    w, ids = router.select_experts(x, x @ gate)
    return (w.unsqueeze(-1) * experts_out(table, x, ids.long())).sum(1) + x @ shared.T


def setup(rank, control):
    import vllm.platforms
    from vllm.platforms.cpu import CpuPlatform

    vllm.platforms._current_platform = CpuPlatform()
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state as ps
    from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
    from vllm.model_executor.layers.fused_moe import config as moe, modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize
    from vllm.model_executor.layers.fused_moe.expert_map_manager import determine_expert_map
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner, _moe_forward_shared
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (TopKWeightAndReduceDelegate,
                                                                             TopKWeightAndReduceNoOP)

    ps.init_distributed_environment(world_size=WORLD, rank=rank, distributed_init_method="unused", local_rank=rank,
                                    backend="gloo")
    dp_rank, tp_rank = rank // 2, rank % 2
    parallel = ParallelConfig(tensor_parallel_size=2, data_parallel_size=2, data_parallel_rank=dp_rank,
                              enable_expert_parallel=True, all2all_backend="allgather_reducescatter")
    config = VllmConfig(parallel_config=parallel)
    context = set_current_vllm_config(config)
    context.__enter__()
    ps.initialize_model_parallel(tensor_model_parallel_size=2, backend="gloo")
    dp, tp, ep = ps.get_dp_group(), ps.get_tp_group(), ps.get_ep_group()
    assert (dp.ranks, tp.ranks, ep.ranks) == ([rank % 2, rank % 2 + 2], [dp_rank * 2, dp_rank * 2 + 1], [0, 1, 2, 3])
    dp.device_communicator, tp.device_communicator = fake_communicator(dp), fake_communicator(tp)
    ep.device_communicator = fake_communicator(ep)
    ep.device_communicator.all2all_manager = AgRsAll2AllManager(ep.cpu_group)

    parallel_moe = moe.FusedMoEParallelConfig.make(2, 1, 2, 1, parallel)
    assert (parallel_moe.ep_size, parallel_moe.ep_rank, parallel_moe.dp_size, parallel_moe.sp_size) == (WORLD, rank, 2, 1)
    assert parallel_moe.use_all2all_kernels and not parallel.use_sequence_parallel_moe
    ep_rank = rank & ~1 if control == "same_experts" else rank
    local, expert_map, _ = determine_expert_map(WORLD, ep_rank, EXPERTS)
    assert local == EXPERTS // WORLD
    table, shared, _ = weights()
    moe_config = moe.FusedMoEConfig(
        num_experts=EXPERTS, experts_per_token=TOPK, hidden_dim=HIDDEN, intermediate_size=HIDDEN,
        num_local_experts=local, num_logical_experts=EXPERTS, activation=moe.MoEActivation.SILU, device="cpu",
        routing_method=moe.RoutingMethodType.Default, moe_parallel_config=parallel_moe, in_dtype=torch.float32,
        skip_final_all_reduce=control == "no_tp_reduce")
    return SimpleNamespace(**locals())


def make_runner(s, style):
    mk, moe = s.mk, s.moe

    class Experts(mk.FusedMoEExpertsModular):
        """``indexed``: as HummingIndexedExperts, the weighted sum of local experts, nothing to reduce after.
        ``contiguous``: as TritonExperts (the MTP layer), (m, topk, k) and the stock weight-and-reduce."""
        activation_format = staticmethod(lambda: mk.FusedMoEActivationFormat.Standard)
        _supports_current_device = _supports_no_act_and_mul = staticmethod(lambda: True)
        _supports_quant_scheme = staticmethod(lambda *a: True)
        _supports_activation = _supports_parallel_config = staticmethod(lambda *a: True)
        supports_expert_map = lambda self: True

        def workspace_shapes(self, M, N, K, topk, *rest):
            return (M * topk, 2 * N), (M * topk, N), ((M, K) if style == "indexed" else (M * topk, K))

        def finalize_weight_and_reduce_impl(self):
            return s.TopKWeightAndReduceNoOP() if style == "indexed" else s.TopKWeightAndReduceDelegate()

        def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
                  expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
                  apply_router_weight_on_input):
            ids = topk_ids.long()
            mine = (expert_map[ids] >= 0).unsqueeze(-1)
            out = experts_out(s.table, hidden_states, ids) * mine
            if style == "indexed":
                output.copy_((topk_weights.unsqueeze(-1) * out).sum(1))
            else:
                output.copy_(out.view(output.shape))

    quant = moe.FusedMoEQuantConfig.make()
    experts = Experts(s.moe_config, quant)
    finalize = s.maybe_make_prepare_finalize(s.moe_config, quant, allow_new_interface=True, use_monolithic=False)
    assert type(finalize).__name__ == "MoEPrepareAndFinalizeNaiveDPEPModular" and not finalize.output_is_reduced()
    assert finalize.num_dispatchers() == WORLD
    kernel = mk.FusedMoEKernel(finalize, experts)
    w1, w2 = torch.zeros(s.local, 2 * HIDDEN, HIDDEN), torch.zeros(s.local, HIDDEN, HIDDEN)

    class Routed(torch.nn.Module):
        quant_method = SimpleNamespace(is_monolithic=False, supports_internal_mk=True, moe_kernel=kernel,
                                       topk_indices_dtype=None, mk_can_overlap_shared_experts=False,
                                       skip_forward_padding=False, has_unpadded_output=False)

        def _ensure_moe_quant_config_init(self):
            pass

        def forward_modular(self, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
            return kernel.apply(x, w1, w2, topk_weights, topk_ids, moe.MoEActivation.SILU, EXPERTS, s.expert_map, False,
                                shared_experts=shared_experts, shared_experts_input=shared_experts_input)

    class Shared(torch.nn.Module):
        """Row-parallel down, reduce_results False: each TP rank holds its half of the output."""

        def forward(self, x):
            half = torch.arange(HIDDEN) % 2 == s.tp_rank
            return x[:, half] @ s.shared.T[half]

    runner = s.MoERunner("layer.0", s.moe_config, s.router, Routed(), shared_experts=Shared())
    runner._forward_entry = s._moe_forward_shared
    return runner


def one_pass(rank, world, job):
    from vllm.forward_context import DPMetadata, create_forward_context, override_forward_context
    from vllm.v1.worker.workspace import init_workspace_manager

    from vllm import _custom_ops

    _custom_ops.moe_sum = lambda fused, output: output.copy_(fused.sum(1))  # a C op
    s = setup(rank, job.get("control"))
    s.router = Router()
    init_workspace_manager(torch.device("cpu"), 1)
    runner = make_runner(s, job["style"])
    errors = {}
    for name, (first, second) in ROWS.items():
        x = torch.randn(first + second, HIDDEN, generator=torch.Generator().manual_seed(first * 1000 + second))
        rows = slice(0, first) if s.dp_rank == 0 else slice(first, first + second)
        local = x[rows]
        meta = DPMetadata.make(s.parallel, len(local), torch.tensor([first, second], dtype=torch.int32))
        with override_forward_context(create_forward_context({}, s.config, dp_metadata=meta)):
            got = runner(local, local @ weights()[2])
        assert got.shape == local.shape
        want = expected(x, s.router)[rows]
        errors[name] = float((got - want).abs().max()) if len(local) else 0.0
    return errors


def worst(style, control=None):
    out = run_ranks(one_pass, world=WORLD, inputs=[{"style": style, "control": control}] * WORLD, timeout=240)
    return {name: max(rank[name] for rank in out) for name in ROWS}


@pytest.mark.parametrize("style", ["indexed", "contiguous"])
def test_sum_matches_one_process(style):
    errors = worst(style)
    print(style, errors)
    assert max(errors.values()) < 2e-4, errors


@pytest.mark.parametrize("control", ["no_tp_reduce", "same_experts"])
def test_controls_catch_a_wrong_sum(control):
    errors = worst("indexed", control)
    print(control, errors)
    assert min(errors.values()) > 1e-2, errors
