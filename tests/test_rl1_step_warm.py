"""U3 (27 Oct): the step path's kernels run once at load on stand-ins; the real parameters, their optimizer state and
the next real step are unchanged."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/posttrain"))


def step(params, opt):
    torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True)
    opt.step()
    opt.zero_grad(set_to_none=True)


def make(seed):
    g = torch.Generator().manual_seed(seed)
    params = [torch.randn(4, 3, generator=g, requires_grad=True), torch.randn(7, generator=g, requires_grad=True)]
    for p in params:
        p.grad = torch.randn(p.shape, generator=g)
    return params


@pytest.mark.parametrize("groups", [1, 2])
def test_warm_step_leaves_the_real_step_bitwise_equal(groups):
    import rl1_trainer as rt

    a, b = make(0), make(0)
    opt_a = torch.optim.AdamW(a, lr=1e-3, fused=False)
    opt_b = torch.optim.AdamW(b, lr=1e-3, fused=False)
    out = rt.warm_step(b, groups)
    assert out["parts"] and set(out["parts"]) == {"cpu/float32"}
    assert all(torch.equal(x, y) for x, y in zip(a, b))  # parameters untouched
    assert all(torch.equal(x.grad, y.grad) for x, y in zip(a, b))  # gradients untouched
    assert not opt_b.state  # no optimizer state made
    step(a, opt_a)
    step(b, opt_b)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_warm_step_covers_every_dtype():
    import rl1_trainer as rt

    params = [torch.zeros(2, requires_grad=True), torch.zeros(2, dtype=torch.bfloat16, requires_grad=True)]
    assert set(rt.warm_step(params)["parts"]) == {"cpu/float32", "cpu/bfloat16"}
