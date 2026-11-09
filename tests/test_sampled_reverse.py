"""Sampled-token reverse KL (heal_kd.sampled_reverse_terms / sampler_logprob) against the exact full-vocab reverse KL
on a tiny model: the importance-weighted estimate, its policy-gradient surrogate and the bucket estimator's bound.
Needs torch (runs on Modal CPU: modal_heal_tiny.py --pytest)."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
sys.path.insert(0, "/opt/heal")
import sys as _sys  # trunk 24 Oct: PTH's trainer copies (scripts/posttrain/pth) shadow RL's for this module
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "pth"))
for _m in ("rl1_trainer", "flash_qlora_job", "rl1_sparse"):
    _sys.modules.pop(_m, None)
import heal_kd  # noqa: E402

V, ROWS, N = 40, 4, 200_000


def exact_reverse(s_logits, t_logits):
    s, t = s_logits.log_softmax(-1), t_logits.log_softmax(-1)
    return (s.exp() * (s - t)).sum(-1)


def draw(logp, n, gen):
    return torch.multinomial(logp.exp(), n, replacement=True, generator=gen)  # [rows, n]


def estimate(s_logits, t_logits, b_logp, tokens, clip):
    s, t = s_logits.log_softmax(-1), t_logits.log_softmax(-1)
    rows = torch.arange(s.shape[0])[:, None].expand_as(tokens)
    est, _, w = heal_kd.sampled_reverse_terms(s[rows, tokens], t[rows, tokens], b_logp[rows, tokens], clip=clip)
    return est.mean(-1), w.mean(-1)


@pytest.fixture
def logits():
    gen = torch.Generator().manual_seed(0)
    s = torch.randn(ROWS, V, generator=gen) * 2.0
    t = s + torch.randn(ROWS, V, generator=gen) * 0.8
    return s, t, gen


def test_on_policy_estimate_matches_exact(logits):
    s, t, gen = logits
    tokens = draw(s.log_softmax(-1), N, gen)
    est, w = estimate(s, t, s.log_softmax(-1), tokens, clip=float("inf"))
    exact = exact_reverse(s, t)
    assert torch.allclose(w, torch.ones_like(w))
    assert torch.allclose(est, exact, rtol=0.03, atol=3e-3), (est, exact)


def test_tempered_behavior_is_corrected_by_the_weights(logits):
    s, t, gen = logits
    b = heal_kd.sampler_logprob(s.repeat_interleave(V, 0), torch.arange(V).repeat(ROWS), temperature=0.7).view(ROWS, V)
    tokens = draw(b, N, gen)
    est, _ = estimate(s, t, b, tokens, clip=float("inf"))
    raw = (s.log_softmax(-1) - t.log_softmax(-1)).gather(-1, tokens).mean(-1)
    exact = exact_reverse(s, t)
    assert torch.allclose(est, exact, rtol=0.05, atol=5e-3), (est, exact)
    assert (raw - exact).abs().max() > (est - exact).abs().max()  # the plain log-ratio is biased under T = 0.7


def test_surrogate_gradient_matches_exact_gradient(logits):
    s, t, gen = logits
    s = s.clone().requires_grad_(True)
    exact = exact_reverse(s, t).sum()
    (g_exact,) = torch.autograd.grad(exact, s)
    tokens = draw(s.detach().log_softmax(-1), N, gen)
    logp = s.log_softmax(-1)
    rows = torch.arange(ROWS)[:, None].expand_as(tokens)
    _, surrogate, _ = heal_kd.sampled_reverse_terms(logp[rows, tokens], t.log_softmax(-1)[rows, tokens],
                                                    logp.detach()[rows, tokens], clip=float("inf"))
    (g_est,) = torch.autograd.grad(surrogate.mean(-1).sum(), s)
    cos = torch.nn.functional.cosine_similarity(g_est.flatten(), g_exact.flatten(), dim=0)
    assert cos > 0.98, float(cos)


def test_sampler_logprob_top_k_top_p():
    z = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])
    all_tokens = torch.arange(5)
    lp = heal_kd.sampler_logprob(z.repeat(5, 1), all_tokens, temperature=1.0, top_k=3)
    assert torch.isinf(lp[3:]).all() and torch.allclose(lp[:3].exp().sum(), torch.tensor(1.0))
    p = z.softmax(-1)[0]  # top_p 0.8: the smallest prefix reaching 0.8 mass
    keep = int((p.cumsum(0) - p <= 0.8).sum())
    lp = heal_kd.sampler_logprob(z.repeat(5, 1), all_tokens, top_p=0.8)
    assert torch.isfinite(lp[:keep]).all() and torch.isinf(lp[keep:]).all()


def test_bucket_reverse_lower_bounds_exact(logits):
    s, t, _ = logits
    sl, tl = s.log_softmax(-1), t.log_softmax(-1)
    idx = tl.topk(8, dim=-1).indices
    t_top = tl.gather(-1, idx)
    t_tail = torch.log1p(-t_top.exp().sum(-1))
    _, reverse = heal_kd.bucket_kl(sl.gather(-1, idx), t_top, t_tail)
    assert (reverse <= exact_reverse(s, t) + 1e-5).all()


def test_default_sampler_is_untruncated_t1(monkeypatch):
    """OPD rollouts sample at T 1 with no top-k / top-p: the default behaviour transform is the identity."""
    monkeypatch.delenv("HEAL_SAMPLER", raising=False)
    _, sampler, _ = heal_kd.sampled_settings()
    assert sampler == (1.0, 0, 1.0)
    z = torch.randn(6, V, generator=torch.Generator().manual_seed(3))
    tok = torch.arange(6)
    assert torch.allclose(heal_kd.sampler_logprob(z, tok, *sampler), z.log_softmax(-1)[torch.arange(6), tok])


def test_t1_samples_charge_the_tail_the_bucket_misses(logits):
    """Untruncated on-policy samples reach tokens outside the teacher's top-k; their IS estimate matches the exact
    full-vocabulary reverse KL, while the top-8 bucket estimator only bounds it from below (the P1-x blind spot)."""
    s, t, gen = logits
    sl, tl = s.log_softmax(-1), t.log_softmax(-1)
    tokens = draw(sl, N, gen)
    top = tl.topk(8, dim=-1).indices
    outside = ~(tokens[..., None] == top[:, None, :]).any(-1)
    assert outside.float().mean() > 0.01  # the samples do visit the teacher's tail
    est, _ = estimate(s, t, sl, tokens, clip=2.0)
    exact = exact_reverse(s, t)
    assert torch.allclose(est, exact, rtol=0.03, atol=3e-3), (est, exact)
    t_top = tl.gather(-1, top)
    _, bucket = heal_kd.bucket_kl(sl.gather(-1, top), t_top, torch.log1p(-t_top.exp().sum(-1)))
    assert (bucket <= exact + 1e-5).all()


def test_truncated_sampler_never_draws_the_tail(logits):
    """The old transform (T 0.7, top-k 20 of 40) gives the tail zero behaviour mass: no sample can charge it."""
    s, _, _ = logits
    b = heal_kd.sampler_logprob(s.repeat_interleave(V, 0), torch.arange(V).repeat(ROWS), 0.7, 20, 0.95).view(ROWS, V)
    assert (torch.isinf(b).sum(-1) >= V - 20).all()


def test_student_file_without_tok_logp_is_refused(tmp_path, monkeypatch):
    """A student-sampled file must carry the teacher's realised-token log p under HEAL_SAMPLED (a top-k lookup would
    drop exactly the tail tokens)."""
    L, K = 12, 4
    d = {"ids": torch.arange(L).int(), "top_ids": torch.zeros(L, K, dtype=torch.int32),
         "top_logp": torch.full((L, K), -2.0).half(), "tail": torch.full((L,), -0.1).half(),
         "reply": torch.ones(L, dtype=torch.bool), "image": torch.zeros(L, dtype=torch.bool)}
    f = tmp_path / "s-k000--e001.pt"
    torch.save(d, f)
    monkeypatch.setenv("HEAL_SAMPLED", "1")
    monkeypatch.delenv("HEAL_ALLOW_TOPK_TTOK", raising=False)
    with pytest.raises(ValueError, match="tok_logp"):
        heal_kd.load_teacher_cpu(f)
    d["tok_logp"] = torch.full((L,), -1.5).half()
    torch.save(d, f)
    assert torch.isfinite(heal_kd.load_teacher_cpu(f)["t_tok"][:-1]).all()
