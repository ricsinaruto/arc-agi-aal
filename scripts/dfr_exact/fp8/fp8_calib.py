"""Calibrated FP8 dense weights (throughput arm, 20 Oct; owner item 1: BF16 -> FP8 calibrated on game data, then the
same method to FP4).

Served path (fp8/fp8_dense.py): FP8 E4M3 weights with one scale per output row, per-token dynamic FP8 activations
(scaled_fp8_quant), cutlass_scaled_mm. Uncalibrated, the scale is amax/448 and the codes plain rounding (RTN).
Calibrated, per key (a layer's projection input, ``D<layer>in`` / ``D<layer>out`` of the dense capture):
  1. H = X^T X from the excl5 calibration rows (dense capture, fp8 rows x per-row scale: the served activation
     quantizer, so H is the second moment the served GEMM sees);
  2. per-row scale = amax/448 x alpha, alpha from ``ALPHAS`` minimising that row's output error d^T H d under RTN;
  3. GPTQ with those fixed scales (act order, block 128, 1% damping): codes chosen column by column with the
     H^-1 error feedback.
Evaluated on held-out eval rows (BF16, other sequences): weight-only and served (activations FP8 per token too)
output error relative to the BF16 output, RTN vs calibrated. Rows are independent in every step, so a fused
projection is fitted as the concatenation of its checkpoint tensors (vLLM's merged layout, checked at load).
"""
import torch

FP8_MAX = 448.0
ALPHAS = tuple(round(1.0 - 0.01 * i, 2) for i in range(21))  # 1.00 .. 0.80

# key side -> (GDN tensors, QSA tensors), checkpoint names under model.language_model.layers.<l>. (or mtp.layers.0.)
TENSORS = {"in": (("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
                  ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
           "out": (("linear_attn.out_proj",), ("self_attn.o_proj",))}
SHAPES = {("in", "gdn"): (16384, 2560), ("in", "qsa"): (13312, 2560), ("out", "gdn"): (2560, 6144),
          ("out", "qsa"): (2560, 6144)}


def fp8_round(x):
    """Nearest E4M3 value (saturating at +-448), as float32."""
    return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()


def token_quant(x):
    """vLLM scaled_fp8_quant per token (dynamic): scale = amax/448 per row; returns the dequantized rows."""
    s = (x.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    return fp8_round(x / s) * s


def hessian(x, chunk=8192):
    """X^T X / rows: float32 chunk products, float64 accumulation."""
    k = x.shape[1]
    h = torch.zeros((k, k), dtype=torch.float64)
    for i in range(0, x.shape[0], chunk):
        b = x[i:i + chunk].float()
        h += (b.t() @ b).double()
    return (h / max(1, x.shape[0])).float()


def damp(h, percdamp=0.01):
    h = h.clone()
    d = torch.diagonal(h)
    d[d <= 0] = 1.0
    d += percdamp * d.mean()
    return h


def rtn(w, scale):
    return fp8_round(w / scale) * scale


def row_error(d, h):
    """Per-row output error d^T H d of weight error rows d [N, K]."""
    return ((d @ h) * d).sum(1)


def clip_search(w, h, alphas=ALPHAS):
    """Per-row alpha minimising d^T H d of RTN at scale = amax/448 x alpha. Returns (scale [N, 1], alpha [N])."""
    base = (w.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    best_err, best_a = None, None
    for a in alphas:
        e = row_error(w - rtn(w, base * a), h)
        if best_err is None:
            best_err, best_a = e, torch.full_like(e, a)
        else:
            take = e < best_err
            best_err = torch.where(take, e, best_err)
            best_a = torch.where(take, torch.full_like(e, a), best_a)
    return base * best_a.unsqueeze(1), best_a


def inverse_cholesky_upper(h):
    for attempt in range(6):
        chol, info = torch.linalg.cholesky_ex(h)
        if int(info) == 0:
            break
        if attempt == 5:
            raise RuntimeError("Cholesky failed")
        h = h.clone()
        torch.diagonal(h).add_(0.1 * torch.diagonal(h).mean() * (4 ** attempt))
    upper, info = torch.linalg.cholesky_ex(torch.cholesky_inverse(chol), upper=True)
    if int(info) != 0:
        raise RuntimeError("Cholesky of the inverse failed")
    return upper


def gptq_fp8(w, h, scale, *, act_order=True, blocksize=128):
    """GPTQ onto E4M3 with fixed per-row scales [N, 1]; ``h`` damped. Returns the dequantized weights [N, K]."""
    n, k = w.shape
    perm = torch.argsort(torch.diagonal(h), descending=True) if act_order else torch.arange(k)
    wp = w[:, perm].clone()
    upper = inverse_cholesky_upper(h[perm][:, perm])
    q = torch.zeros_like(wp)
    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        w1 = wp[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        u1 = upper[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = w1[:, i:i + 1]
            qc = fp8_round(col / scale) * scale
            q[:, i1 + i:i1 + i + 1] = qc
            err = (col - qc) / u1[i, i]
            w1[:, i:] -= err * u1[i:i + 1, i:]
            err1[:, i:i + 1] = err
        wp[:, i2:] -= err1 @ upper[i1:i2, i2:]
    out = torch.empty_like(q)
    out[:, perm] = q
    return out


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


def evaluate(x, w, qw, chunk=4096):
    """Relative output errors on rows x [M, K]: weight-only and served (per-token FP8 activations)."""
    num_w = num_s = den = 0.0
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk]
        ref = xb @ w.t()
        num_w += float((xb @ qw.t() - ref).pow(2).sum())
        num_s += float((token_quant(xb) @ qw.t() - ref).pow(2).sum())
        den += float(ref.pow(2).sum())
    return {"weight": (num_w / den) ** 0.5, "served": (num_s / den) ** 0.5}


def fit(w, x_calib, x_eval, *, act_order=True, percdamp=0.01):
    """Fit one key. Returns (codes [N, K] float8_e4m3fn, scale [N] float32, stats)."""
    w = w.float()
    h = hessian(x_calib)
    hd = damp(h, percdamp)
    base = (w.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    q_rtn = rtn(w, base)
    scale, alpha = clip_search(w, hd)
    q_clip = rtn(w, scale)
    q_cal = gptq_fp8(w, hd, scale, act_order=act_order)
    stats = {"rows_calib": int(x_calib.shape[0]), "rows_eval": int(x_eval.shape[0]),
             "alpha_mean": float(alpha.mean()), "alpha_lt1": float((alpha < 1).float().mean()),
             "calib_err": {name: float(row_error(w - q, hd).sum() / row_error(w, hd).sum()) ** 0.5
                           for name, q in (("rtn", q_rtn), ("clip", q_clip), ("gptq", q_cal))},
             "eval": {name: evaluate(x_eval, w, q) for name, q in (("rtn", q_rtn), ("clip", q_clip), ("gptq", q_cal))},
             "eval_act_only": evaluate(x_eval, w, w)["served"]}
    codes = (q_cal / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if not torch.equal(codes.float() * scale, q_cal):
        raise RuntimeError("calibrated weights are not on the scaled E4M3 grid")
    return codes, scale.squeeze(1), stats
