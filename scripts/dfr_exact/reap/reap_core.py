"""Item 6 (22 Oct): expert masks by REAP score vs routed mass, and a router refit for the kept experts, on one layer.

Routing (Qwen3-Next MoE): logits = x @ W_gate^T, softmax, top-10, weights renormalised over the 10 (norm_topk_prob).
A pruned server masks the pruned experts' logits, so it routes top-10 among the kept experts with weights renormalised
over those 10. Scores on the calibration rows: mass_j = sum of w_tj; reap_j = mean over routed rows of w_tj * ||E_j(x_t)||
(REAP, Lasby et al. 2025); ean_j = sum of w_tj * ||E_j(x_t)||. Error = ||y_full - y|| / ||y_full|| of the routed-expert
output (shared expert unchanged), on held-out rows. Refit: logits' = x @ (W[kept] + dW)^T + b over each row's top-C kept
candidates (by original logits), Adam on the relative squared error + l2 on dW; variants bias-only and full.
"""
import torch

TOPK = 10


def route(logits, k=TOPK):
    """(ids [n, k], weights [n, k]) from logits [n, E] (float32): softmax, top-k, renormalised."""
    top, ids = torch.topk(logits.float(), k, dim=-1)
    w = torch.softmax(top, dim=-1)  # softmax over all then renormalise over the top-k == softmax over the top-k
    return ids, w


def expert_out(x, gate_up, down, e, inter):
    gu = gate_up[e].float()
    y = x.float() @ gu.t()
    h = torch.nn.functional.silu(y[:, :inter]) * y[:, inter:]
    return h @ down[e].float().t()


def moe(x, ids, w, gate_up, down, inter, norms=None):
    """Routed-expert output float32 [n, H]; with ``norms`` (dict) also per-expert sums of w*||E(x)||, counts, w sums."""
    n, H = x.shape[0], down.shape[1]
    out = torch.zeros((n, H), dtype=torch.float32, device=x.device)
    flat_e = ids.reshape(-1)
    flat_t = torch.arange(n, device=x.device).repeat_interleave(ids.shape[1])
    flat_w = w.reshape(-1).float()
    order = torch.argsort(flat_e, stable=True)
    flat_e, flat_t, flat_w = flat_e[order], flat_t[order], flat_w[order]
    E = gate_up.shape[0]
    bounds = torch.searchsorted(flat_e, torch.arange(E + 1, device=x.device)).tolist()
    for e in range(E):
        a, b = bounds[e], bounds[e + 1]
        if a == b:
            continue
        t = flat_t[a:b]
        y = expert_out(x[t], gate_up, down, e, inter)
        out.index_add_(0, t, y * flat_w[a:b, None])
        if norms is not None:
            nrm = y.norm(dim=1)
            norms["wn"][e] += (flat_w[a:b] * nrm).sum()
            norms["cnt"][e] += b - a
            norms["w"][e] += flat_w[a:b].sum()
    return out


def rel(y, ref):
    return float((y - ref).norm() / ref.norm())


def kept_logits(logits, kept):
    """Logits with every pruned expert at -inf (the pruned server's routing)."""
    m = torch.full_like(logits, float("-inf"))
    m[:, kept] = logits[:, kept]
    return m


def scores(x, W, gate_up, down, inter, chunk=8192):
    E = W.shape[0]
    acc = {k: torch.zeros(E, dtype=torch.float64, device=x.device) for k in ("wn", "cnt", "w")}
    for i in range(0, x.shape[0], chunk):
        xs = x[i:i + chunk]
        ids, w = route(xs.float() @ W.float().t())
        moe(xs, ids, w, gate_up, down, inter, norms=acc)
    reap = acc["wn"] / acc["cnt"].clamp(min=1)
    return {"mass": acc["w"], "reap": reap, "ean": acc["wn"], "count": acc["cnt"]}


def keep_top(score, n_keep):
    return torch.sort(torch.topk(score, n_keep).indices).values


def mask_error(x, W, gate_up, down, inter, masks, chunk=8192):
    """Relative error of each mask's routed output vs the full model on rows x (pooled over chunks)."""
    num = {k: 0.0 for k in masks}
    den = 0.0
    for i in range(0, x.shape[0], chunk):
        xs = x[i:i + chunk]
        logits = xs.float() @ W.float().t()
        ref = moe(xs, *route(logits), gate_up, down, inter)
        den += float(ref.pow(2).sum())
        for k, spec in masks.items():
            kept, extra = spec if isinstance(spec, tuple) else (spec, None)
            lg = kept_logits(logits if extra is None else extra(xs, logits), kept)
            y = moe(xs, *route(lg), gate_up, down, inter)
            num[k] += float((y - ref).pow(2).sum())
    return {k: (v / den) ** 0.5 for k, v in num.items()}


def refit(x, W, gate_up, down, inter, kept, *, cand=24, steps=300, lr=1e-3, l2=1e-2, full=True, chunk=8192, log=None):
    """Router refit for the kept experts: (dW [K, H], b [K]) minimising the routed-output error vs the full model.
    Candidates per row: the top-``cand`` kept experts by original logits; their outputs are precomputed (bf16)."""
    dev = x.device
    K, H = len(kept), W.shape[1]
    kept_t = torch.as_tensor(kept, device=dev)
    Wk = W[kept_t].float()
    xs_all, ref_all, cid_all, cy_all = [], [], [], []
    for i in range(0, x.shape[0], chunk):
        xs = x[i:i + chunk]
        logits = xs.float() @ W.float().t()
        ref_all.append(moe(xs, *route(logits), gate_up, down, inter))
        cid = torch.topk(logits[:, kept_t], cand, dim=-1).indices  # kept-rank ids [n, C]
        cy = torch.empty((xs.shape[0], cand, H), dtype=torch.bfloat16, device=dev)
        flat = cid.reshape(-1)
        rows = torch.arange(xs.shape[0], device=dev).repeat_interleave(cand)
        slot = torch.arange(cand, device=dev).repeat(xs.shape[0])
        order = torch.argsort(flat, stable=True)
        f, r, s = flat[order], rows[order], slot[order]
        bounds = torch.searchsorted(f, torch.arange(K + 1, device=dev)).tolist()
        for j in range(K):
            a, b = bounds[j], bounds[j + 1]
            if a < b:
                cy[r[a:b], s[a:b]] = expert_out(xs[r[a:b]], gate_up, down, kept[j], inter).to(torch.bfloat16)
        xs_all.append(xs); cid_all.append(cid); cy_all.append(cy)
    dW = torch.zeros((K, H), device=dev, requires_grad=full)
    b = torch.zeros(K, device=dev, requires_grad=True)
    opt = torch.optim.Adam([b] + ([dW] if full else []), lr=lr)
    den = sum(float(r.pow(2).sum()) for r in ref_all)
    hist = []
    for step in range(steps):
        opt.zero_grad()
        tot = 0.0
        for xs, ref, cid, cy in zip(xs_all, ref_all, cid_all, cy_all):
            Wc = (Wk + dW)[cid] if full else Wk[cid]  # [n, C, H]
            lg = torch.einsum("nh,nch->nc", xs.float(), Wc) + b[cid]
            # soft top-k surrogate (hard top-k has no gradient through selection): membership sigmoid around the
            # midpoint of the 10th/11th logits, temperature annealed 1.0 -> 0.02; evaluation uses the hard top-k.
            srt = torch.topk(lg, TOPK + 1, dim=-1).values.detach()
            thr = (srt[:, TOPK - 1] + srt[:, TOPK]) / 2
            temp = 1.0 * (0.02 / 1.0) ** (step / max(steps - 1, 1))
            member = torch.sigmoid((lg - thr[:, None]) / temp)
            z = member * torch.exp(lg - srt[:, :1])
            w = z / z.sum(-1, keepdim=True)
            y = torch.einsum("nc,nch->nh", w, cy.float())
            loss = (y - ref).pow(2).sum() / den
            loss.backward()
            tot += float(loss.detach())
        if full:
            (l2 * dW.pow(2).sum() / K).backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            hist.append((step, tot ** 0.5))
            if log:
                log(f"refit step {step} rel {tot ** 0.5:.5f}")
    dW_f = dW.detach() if full else torch.zeros((K, H), device=dev)
    b_f = b.detach()

    def extra(xs, logits):
        lg = logits.clone()
        lg[:, kept_t] = xs.float() @ (Wk + dW_f).t() + b_f
        return lg
    return dW_f, b_f, extra, hist
