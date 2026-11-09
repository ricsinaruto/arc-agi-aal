"""V9C's INT8 format on L4: one scale per output channel, amax / 127, Triton W8A16 (arc_infra.
dense_kernels_runtime), as ``tools`` for ``arc2.l4_dense.convert``. A row-parallel projection holds a quarter of each
row, so its scale takes the row's amax over the four ranks: the codes equal TP 1's."""
import torch

from arc_infra import dense_kernels_runtime as kernels

ROW_PARALLEL_K = 1536  # .linear_attn.out_proj, .self_attn.o_proj at TP 4
TILES = {(4096, 2560): ((32, 256, 4, 3), (64, 128, 4, 3)), (2560, 1536): ((32, 256, 4, 3), (64, 128, 4, 3)),
         (3584, 2560): ((16, 256, 4, 3), (64, 128, 4, 3)), (62080, 2560): ((32, 256, 4, 3), (32, 256, 4, 3))}
WARM = {}


def amax_rows(weight):
    amax = weight.float().abs().amax(dim=1)
    if weight.shape[1] == ROW_PARALLEL_K:
        from vllm.distributed.parallel_state import get_tp_group

        torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX, group=get_tp_group().device_group)
    return amax


def quantize(torch_, weight):
    scale = (amax_rows(weight) / 127.0).clamp(min=1e-12)
    codes = torch.round(weight.float() / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    kernels._register_op()
    kernels.TRITON_TILES.update(TILES)
    return {"format": "v9c"}, {"weight": codes, "weight_scale": scale}


def forward(config, inputs, weight, weight_scale, zero_point=None, bias=None, weight_scale_2=None, locks=None,
            compute_config=None):
    WARM.setdefault((weight.shape[0], weight.shape[1]), (weight, weight_scale))
    return torch.ops.vllm.arc_w8a16_linear(inputs, weight, weight_scale)


def warm():
    """Every Triton specialization the served row counts can hit, before graph capture."""
    with torch.no_grad():
        for (n, k), (weight, scale) in WARM.items():
            x = torch.zeros((kernels.TRITON_MAX_M, k), dtype=torch.bfloat16, device=weight.device)
            for m in range(1, kernels.TRITON_MAX_M + 1):
                kernels.w8a16_gemm(x[:m], weight, scale)
        torch.cuda.synchronize()
    WARM.clear()
