"""MFU-1 expert compaction: routes on global ids mapped to compact ids, the kept experts' LoRA factors
sliced from the 512-layout parameters. CPU: the grouped-LoRA MoE math (per-expert loop, fp32) full vs compact,
outputs and LoRA grads bitwise. GPU: rl1_experts.moe_forward on synthetic NVFP4 experts, full vs compacted."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))


def _helpers():
    """rl1_experts' pure-torch helpers without importing triton (CPU boxes)."""
    import ast
    import types

    here = Path(__file__).resolve()
    paths = [here.parents[1] / "scripts" / "posttrain" / "rl1_experts.py", here.parent / "rl1_experts.py"]
    src = next(q for q in paths if q.is_file()).read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_cut_lora", "compact_route")]
    mod = types.ModuleType("compact_helpers")
    mod.torch = torch
    exec(compile(ast.Module(body=keep, type_ignores=[]), "rl1_experts", "exec"), mod.__dict__)
    return mod


def moe_ref(x, idx, w, W1, W2, A1, B1, A2, B2, E, s=0.5):
    """Grouped MoE with per-expert LoRA (A [r*E, d], B [d_out, r*E] as Axolotl lays them out), sorted by expert."""
    T, K = idx.shape
    r = A1.shape[0] // E
    flat = idx.reshape(-1)
    tok = torch.arange(T).repeat_interleave(K)
    order = torch.sort(flat, stable=True).indices
    out = torch.zeros(T, W2.shape[1])
    A1e, A2e = A1.reshape(E, r, -1), A2.reshape(E, r, -1)
    B1e, B2e = B1.reshape(B1.shape[0], r, E).permute(2, 0, 1), B2.reshape(B2.shape[0], r, E).permute(2, 0, 1)
    for e in range(E):
        rows = order[flat[order] == e]
        if not len(rows):
            continue
        xe = x[tok[rows]]
        h = xe @ W1[e].T + s * (xe @ A1e[e].T) @ B1e[e].T
        g, u = h.chunk(2, dim=-1)
        a = torch.nn.functional.silu(g) * u
        y = a @ W2[e].T + s * (a @ A2e[e].T) @ B2e[e].T
        out.index_add_(0, tok[rows], y * w.reshape(-1)[rows, None])
    return out


def test_compact_moe_equals_full_cpu():
    h = _helpers()
    g = torch.Generator().manual_seed(0)
    E, K, T, H, I, r = 16, 3, 40, 32, 24, 4
    pruned = [1, 4, 5, 9, 15]
    kept = torch.tensor([e for e in range(E) if e not in pruned])
    remap = torch.full((E,), -1, dtype=torch.long)
    remap[kept] = torch.arange(len(kept))
    logits = torch.randn(T, E, generator=g)
    logits[:, pruned] = -1e4
    w, idx = torch.softmax(logits, -1).topk(K, -1)
    x = torch.randn(T, H, generator=g)
    W1, W2 = torch.randn(E, 2 * I, H, generator=g), torch.randn(E, H, I, generator=g)
    params = [torch.randn(r * E, H, generator=g), torch.randn(2 * I, r * E, generator=g),
              torch.randn(r * E, I, generator=g), torch.randn(H, r * E, generator=g)]

    def run(compact):
        ps = [p.clone().requires_grad_(True) for p in params]
        if compact:
            A1, B1 = h._cut_lora((ps[0], ps[1]), kept, E)
            A2, B2 = h._cut_lora((ps[2], ps[3]), kept, E)
            out = moe_ref(x, h.compact_route(idx, remap), w, W1[kept], W2[kept], A1, B1, A2, B2, len(kept))
        else:
            out = moe_ref(x, idx, w, W1, W2, *ps, E)
        out.square().sum().backward()
        return out.detach(), [p.grad for p in ps]

    (o1, g1), (o2, g2) = run(False), run(True)
    assert torch.equal(o1, o2)
    for a, b in zip(g1, g2):
        assert torch.equal(a, b)
    for grad in g2:  # the masked experts' factors get exact zeros
        assert grad.abs().sum() > 0


def test_compact_route_monotonic():
    h = _helpers()
    remap = torch.tensor([0, -1, 1, 2, -1, 3])
    idx = torch.tensor([[5, 0], [3, 2]])
    assert h.compact_route(idx, remap).tolist() == [[3, 0], [2, 1]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 experts on the GPU")
def test_moe_forward_compacted_bitwise_gpu():
    import rl1_dequant
    import rl1_experts
    from axolotl.integrations.kernels.libs.sonicmoe import nvfp4_lora
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    torch.manual_seed(0)
    rl1_experts.install()
    rl1_dequant.install()
    rl1_experts.EXPERTS["on"] = True
    rl1_dequant.DEQUANT["on"] = True
    E, K, T, H, I, r = 64, 8, 2048, 1024, 512, 16
    pruned = sorted(torch.randperm(E)[:24].tolist())
    kept = torch.tensor([e for e in range(E) if e not in pruned], device="cuda")
    def packed(N, Kd):  # random FP4 codes, positive e4m3 block scales, per-expert fp32 scales (the loaders' layout)
        qd = torch.randint(0, 256, (E, N, Kd // 2), dtype=torch.uint8, device="cuda")
        sc = (torch.rand(E, N, Kd // 16, device="cuda") * 0.05 + 0.01).to(torch.float8_e4m3fn)
        return qd, sc, torch.rand(E, 1, 1, device="cuda") + 0.5

    p1, p2 = packed(2 * I, H), packed(H, I)

    def q(parts):
        qd, sc, pts = parts
        return NVFP4Tensor(qd.clone(), sc.clone(), 16, torch.bfloat16, per_tensor_scale=pts.clone())

    logits = torch.randn(T, E, device="cuda")
    logits[:, pruned] = -1e4
    wts, idx = torch.softmax(logits, -1).topk(K, -1)
    x0 = torch.randn(T, H, device="cuda")
    loras = [torch.randn(r * E, H, device="cuda") * 0.01, torch.randn(2 * I, r * E, device="cuda") * 0.01,
             torch.randn(r * E, I, device="cuda") * 0.01, torch.randn(H, r * E, device="cuda") * 0.01]
    probe = torch.randn(T, H, device="cuda")
    results = {}
    for compact in (False, True):
        w1, w2 = q(p1), q(p2)
        rl1_experts.COMPACT.update(on=False, by={}, layers=0, freed_bytes=0, calls=0)
        if compact:
            remap = torch.full((E,), -1, dtype=torch.long, device="cuda")
            remap[kept] = torch.arange(len(kept), device="cuda")
            rl1_experts._cut_nvfp4(w1, kept)
            rl1_experts._cut_nvfp4(w2, kept)
            info = {"E": E, "kept": kept, "remap": remap, "layer": 0}
            rl1_experts.COMPACT["by"] = {id(w1): info, id(w2): info}
            rl1_experts.COMPACT["on"] = True
        x = x0.clone().requires_grad_(True)
        ps = [p.clone().requires_grad_(True) for p in loras]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = nvfp4_lora.grouped_moe_reference_forward(x, idx, wts, w1, None, w2, None, (ps[0], ps[1]), (ps[2], ps[3]),
                                                           E, act="silu", backend="dequant", concat=True, scaling1=0.5,
                                                           scaling2=0.5)
        (out.float() * probe).sum().backward()
        results[compact] = [out.detach(), x.grad, *[p.grad for p in ps]]
        assert rl1_experts.EXPERTS["fallback"] == 0, rl1_experts.EXPERTS["why"]
    assert rl1_experts.COMPACT["calls"] == 1
    for name, a, b in zip(["out", "dx", "dA1", "dB1", "dA2", "dB2"], results[False], results[True]):
        assert torch.equal(a, b), (name, float((a != b).float().mean()))


if __name__ == "__main__":  # the GPU check as a plain script (modal_mfu1.py segment)
    test_compact_moe_equals_full_cpu()
    test_moe_forward_compacted_bitwise_gpu()
    print("COMPACT EXACT", flush=True)
