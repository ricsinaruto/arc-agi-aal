"""rl1-route (3c): router top-k record and replay; off mode keeps the router's own arithmetic."""
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import rl1_trainer  # noqa: E402


class Router(torch.nn.Module):  # the model's Qwen4ExpTextTopKRouter forward, verbatim
    def __init__(self, experts=16, top_k=4, hidden=8):
        super().__init__()
        self.top_k, self.num_experts, self.norm_topk_prob, self.hidden_dim = top_k, experts, True, hidden
        self.weight = torch.nn.Parameter(torch.randn(experts, hidden))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = torch.nn.functional.linear(hidden_states, self.weight)
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices


@pytest.fixture
def patched(monkeypatch):
    original = Router.forward
    modeling = types.ModuleType("modeling_qwen4_exp")
    modeling.Qwen4ExpTextTopKRouter = Router
    for name, mod in (("transformers", types.ModuleType("transformers")), ("transformers.models", types.ModuleType("m")),
                      ("transformers.models.qwen4_exp", types.ModuleType("q")),
                      ("transformers.models.qwen4_exp.modeling_qwen4_exp", modeling)):
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules["transformers.models.qwen4_exp"].modeling_qwen4_exp = modeling
    monkeypatch.setattr(rl1_trainer, "ROUTE", {"router": "off", "qsa": "off", "rec": [], "qsa_rec": [], "i": 0, "qi": 0,
                                               "calls": 0, "qsa_calls": 0})
    rl1_trainer.route_patch()
    yield original
    Router.forward = original
    Router._rl1_route = False


def test_off_mode_matches_the_original_and_replay_forces_recorded_experts(patched):
    torch.manual_seed(0)
    router, x = Router(), torch.randn(32, 8)
    want = patched(router, x)
    got = router(x)
    assert all(torch.equal(a, b) for a, b in zip(want, got))
    R = rl1_trainer.ROUTE
    R["router"] = "capture"
    _, w0, idx0 = router(x)
    assert len(R["rec"]) == 1
    R["router"] = "off"
    with torch.no_grad():
        router.weight.add_(0.5 * torch.randn_like(router.weight))  # enough to flip some top-k choices
    _, _, idx_free = router(x)
    assert not torch.equal(idx_free, idx0)
    R["router"], R["i"] = "replay", 0
    logits, w1, idx1 = router(x)
    assert torch.equal(idx1, idx0.long()) and R["i"] == 1
    probs = torch.softmax(logits.float(), dim=-1).gather(-1, idx0.long())
    assert torch.allclose(w1, probs / probs.sum(-1, keepdim=True))
    with pytest.raises(RuntimeError):
        router(x)  # a call past the record


def served_trainer(applied, seed=0):
    """A stand-in Trainer: raw log-prob per token = base + sum over parameter groups of coef x the group's lora_B
    sum; the sampler's recorded log-probs use only the ``applied`` groups (plus noise)."""
    g = torch.Generator().manual_seed(seed)
    names = {"down": "model.layers.0.mlp.experts.lora_B.default.weight",
             "gate_up": "model.layers.0.mlp.experts.base_layer.lora_B.default.weight",
             "dense": "model.layers.0.self_attn.q_proj.lora_B.default.weight"}
    params = {k: torch.nn.Parameter(torch.randn(4, 3, generator=g)) for k in names}
    t = rl1_trainer.Trainer.__new__(rl1_trainer.Trainer)
    t.params = [(names[k], params[k]) for k in names] + [("model.layers.0.mlp.experts.lora_A.default.weight", torch.nn.Parameter(torch.ones(2)))]
    tokens = 4000
    base = -torch.rand(tokens, generator=g, dtype=torch.float64) * 3
    coef = {k: torch.randn(tokens, generator=g, dtype=torch.float64) * 0.05 for k in names}

    def raw(use):
        return base + sum(coef[k] * float(params[k].detach().sum()) for k in use)

    engine = raw(applied) + torch.randn(tokens, generator=g, dtype=torch.float64) * 0.08
    t.expand = lambda attempts, **kw: list(attempts)
    t.attempt_window = lambda spec, start, train: {"engine_raw": engine.tolist()}
    t.window_pass = lambda made, shared=None: {"raw": raw(names)}
    return t, params


@pytest.mark.parametrize("applied,expect", [(("down", "gate_up", "dense"), {"all": 1.0, "gate_up": 1.0, "down": 1.0}),
                                            (("dense",), {"all": 0.0, "gate_up": 0.0, "down": 0.0}),
                                            (("down", "dense"), {"all": None, "gate_up": 0.0, "down": 1.0})])
def test_served_check_reads_which_expert_parts_the_server_applies(applied, expect):
    t, params = served_trainer(applied)
    before = {k: p.detach().clone() for k, p in params.items()}
    out = t.op_served_check({"attempts": [{"rollout": "x"}]})
    for tag, want in expect.items():
        if want is not None:
            assert abs(out["parts"][tag]["slope"] - want) < 0.1, (tag, out["parts"][tag])
    assert all(torch.equal(params[k], before[k]) for k in params)  # restored
    assert out["parts"]["all"]["params"] == 24 and out["parts"]["down"]["params"] == 12
    dense = t.op_served_check({"attempts": [{"rollout": "x"}], "zero": ["dense"]})["parts"]["dense"]
    assert dense["params"] == 12 and abs(dense["slope"] - (1.0 if "dense" in applied else 0.0)) < 0.1
