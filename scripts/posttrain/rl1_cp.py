"""RL redesign stage C1 (26 Oct): sequence (context) parallelism for one Flash window across a torch.distributed group.
The window's trained segment [P0, P0 + L) is split into contiguous shards, rank r holding lengths[r] tokens (any
lengths, empty shards allowed); the carried-in context [0, P0) is a replicated read-only ``Prefix`` (no gradient),
filled by ``prefill`` -- the same sharded pass under no_grad, whose all-gathered K/V, indexer keys, conv tails and
folded GDN states every rank keeps.

Per layer type (Flash: 36 GDN, 12 QSA attention, PLE at layer index 1):
- GDN (``GDNCore``): each rank runs its shard from a zero state and, with the same kernel, the state transition of
  the shard (the state recurrence from the identity with zero values: M, [K, K] per value head). Ranks all-gather
  (M, S_local); each folds S_in(r) = M_{r-1} ... (h0) + ... and recomputes its output from S_in(r): exact (the
  recurrence is linear in the state). Backward: dh0 of each shard with dh_T = 0 (linear again), all-gathered, folded
  in reverse through M^T into each shard's dh_T, then the kernel's own backward with (do, dh_T). The causal conv
  takes its 3 preceding pre-conv rows from the earlier ranks (``Halo``; the halo's gradient goes back to its owners).
- QSA attention: indexer keys all-gathered (no grad: the selection is discrete), the trainer's own ``qsa_select`` per
  shard with global positions / causal counts, K/V all-gathered (``GatherSeq``), the trainer's selected-attention
  kernel over [prefix | all shards]; the backward reduce-scatters dK/dV (fp32 for the torch kernel) to the owners.
  Prefix keys carry no gradient (they are not leaves of the graph; the Union kernel's P0 skips them).
- PLE: the n-gram ids' 2 preceding ids from the replicated segment ids; the dilated short conv's 9 preceding rows by
  ``Halo``. MoE, hyper-connections, norms, LM head: token-local, unchanged.

Kernels: ``TorchGDN`` (transformers' torch_chunk_gated_delta_rule: the CPU reference, used by the tests) and
``FlaGDN`` (fla's chunk kernels as rl1_gdn calls them: use_gate_in_kernel, q/k at the key heads, chunk 64, the
pinned BK=BV=64 o-kernel). Usage:

    pre = prefill(model, Ctx(ctx_lengths, ids=context_ids), my_context_ids)   # optional: the carried context
    cp = Ctx(lengths, ids=segment_ids, prefix=pre, position_ids=pos)          # all ranks: same lengths / ids / pos
    with active(cp):                                                          # forward + backward (recompute)
        hidden = forward(model, cp, my_segment_ids)                           # [B, lengths[rank], hidden]
        loss(hidden).backward()                                               # then all-reduce the LoRA grads
"""

import contextlib
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F

CP = {"ctx": None}  # the active context: the patched forwards read it (None: the model's / trainer's code)
STATS = {"gather_bytes": 0, "reduce_bytes": 0, "halo_bytes": 0, "state_bytes": 0}  # bytes sent to peers


def _count(key, nbytes):
    STATS[key] += int(nbytes)


# ------------------------------------------------------------------------------------------------ context
class Prefix:
    """The carried-in context, replicated on every rank, read-only: ``length`` tokens, the last token ids and per
    layer index: GDN {"conv": [B, 3, C], "state": [B, Hv, K, V] fp32}, attention {"k", "v": [1, G, P0, D],
    "idx": [1, P0, d]}, PLE {"ple": [B, 9, 4H]}."""

    def __init__(self, length=0, ids=None, layers=None):
        self.length, self.ids, self.layers = int(length), ids, dict(layers or {})

    def nbytes(self):
        return sum(t.numel() * t.element_size() for d in self.layers.values() for t in d.values() if torch.is_tensor(t))


class Ctx:
    """One CP call. ``lengths``: every rank's shard length (same list on all ranks); ``ids`` [B, L] the whole segment's
    token ids (PLE context; may be None without PLE); ``position_ids`` the window's positions over prefix + segment
    ([B, P0 + L], or [3 | 4, B, P0 + L] mrope; None: arange); ``capture``: record the next Prefix (``prefill``)."""

    def __init__(self, lengths, ids=None, prefix=None, position_ids=None, group=None, capture=False):
        self.group = group
        self.rank = dist.get_rank(group) if dist.is_initialized() else 0
        self.world = dist.get_world_size(group) if dist.is_initialized() else 1
        self.lengths = [int(n) for n in lengths]
        if len(self.lengths) != self.world:
            raise ValueError(f"{len(self.lengths)} shard lengths for world {self.world}")
        self.total = sum(self.lengths)
        self.offset = sum(self.lengths[:self.rank])
        self.T = self.lengths[self.rank]
        self.prefix = prefix or Prefix()
        self.P0 = self.prefix.length
        self.ids = ids
        self.position_ids = position_ids
        self.capture = capture
        self.captured = {}
        self.rope = None  # (cos, sin) over prefix + segment, set by ``forward``
        self.nccl = dist.is_initialized() and dist.get_backend(group) == "nccl"

    def layer(self, idx):
        return self.prefix.layers.get(idx, {})

    # -------------------------------------------------------------------------------------------- collectives
    def gather_fixed(self, t):
        """all_gather of a same-shape tensor: list of world tensors."""
        t = t.contiguous()
        if self.world == 1:
            return [t]
        out = [torch.empty_like(t) for _ in range(self.world)]
        dist.all_gather(out, t, group=self.group)
        return out

    def gather_rows(self, x, dim):
        """All ranks' shards of ``x`` along ``dim`` (each rank's own length), concatenated in rank order."""
        if self.world == 1:
            return x
        top = max(self.lengths)
        xm = x.movedim(dim, 0)
        pad = xm.new_zeros((top, *xm.shape[1:]))
        pad[:xm.shape[0]] = xm
        parts = self.gather_fixed(pad)
        _count("gather_bytes", pad.numel() * pad.element_size() * (self.world - 1))
        return torch.cat([p[:n] for p, n in zip(parts, self.lengths)]).movedim(0, dim)

    def reduce_scatter_rows(self, g, dim):
        """Sum over ranks of ``g`` (the gathered layout along ``dim``), each rank keeping its own shard's rows."""
        if self.world == 1:
            return g
        top = max(self.lengths)
        gm = g.movedim(dim, 0)
        stack = gm.new_zeros((self.world, top, *gm.shape[1:]))
        start = 0
        for j, n in enumerate(self.lengths):
            stack[j, :n] = gm[start:start + n]
            start += n
        _count("reduce_bytes", stack[0].numel() * stack.element_size() * (self.world - 1))
        if self.nccl:
            own = stack.new_empty(stack.shape[1:])
            dist.reduce_scatter_tensor(own, stack, group=self.group)
        else:  # gloo has no reduce_scatter
            dist.all_reduce(stack, group=self.group)
            own = stack[self.rank]
        return own[:self.T].movedim(0, dim)

    def assemble(self, tails, upto, n, init):
        """The ``n`` rows before rank ``upto``'s first token from the ranks' padded tails [B, n, C] (``init``: rows
        before the segment, None = zeros); also the plan [(rank, rows)] (the halo's last rows first)."""
        need, parts, plan = n, [], []
        for j in range(upto - 1, -1, -1):
            take = min(need, self.lengths[j], n)
            if take:
                parts.insert(0, tails[j][:, n - take:])
                plan.append((j, take, need))  # rows [need - take, need) of the halo
            need -= take
            if not need:
                break
        if need:
            ref = tails[0]
            head = init[:, init.shape[1] - need:].to(ref.dtype) if init is not None else ref.new_zeros(
                (ref.shape[0], need, ref.shape[2]))
            parts.insert(0, head)
        return torch.cat(parts, dim=1), plan


@contextlib.contextmanager
def active(ctx):
    install()
    prev = CP["ctx"]
    CP["ctx"] = ctx
    try:
        yield ctx
    finally:
        CP["ctx"] = prev


# ------------------------------------------------------------------------------------------------ autograd ops
class GatherSeq(torch.autograd.Function):
    """All ranks' shards along ``dim`` (rank order), as ``dtype``; backward: the gradient reduce-scattered to the owners
    (in fp32 when ``dtype`` is fp32: the partial dK / dV sums are not rounded before the single cast)."""

    @staticmethod
    def forward(ctx, x, dim, cp, dtype):
        ctx.dim, ctx.cp, ctx.dtype = dim, cp, x.dtype
        out = cp.gather_rows(x, dim).to(dtype)
        return out.clone() if out is x else out

    @staticmethod
    def backward(ctx, g):
        return ctx.cp.reduce_scatter_rows(g.float().contiguous(), ctx.dim).to(ctx.dtype), None, None, None


class Halo(torch.autograd.Function):
    """x [B, T, C] -> the ``n`` rows before this rank's first token (earlier ranks' rows, then ``init``: the prefix's
    rows or zeros). Backward: each halo row's gradient back to the rank owning that row. ``key``: under capture, the
    window's last ``n`` rows go to ``cp.captured[key]["conv"]`` (the next prefix's conv tail)."""

    @staticmethod
    def forward(ctx, x, n, cp, init, key):
        B, T, C = x.shape
        take = min(T, n)
        buf = x.new_zeros((B, n, C))
        if take:
            buf[:, n - take:] = x[:, T - take:]
        tails = cp.gather_fixed(buf)
        _count("halo_bytes", buf.numel() * buf.element_size() * (cp.world - 1))
        halo, plan = cp.assemble(tails, cp.rank, n, init)
        if cp.capture and key is not None:
            cp.captured.setdefault(key[0], {})[key[1]] = cp.assemble(tails, cp.world, n, init)[0].detach().clone()
        ctx.plan, ctx.cp, ctx.n, ctx.T, ctx.shape = plan, cp, n, T, (B, C)
        return halo

    @staticmethod
    def backward(ctx, gh):
        cp, n, T = ctx.cp, ctx.n, ctx.T
        B, C = ctx.shape
        G = gh.new_zeros((cp.world, B, n, C), dtype=torch.float32)
        for j, take, end in ctx.plan:
            G[j, :, n - take:] += gh[:, end - take:end].float()
        if cp.world > 1:
            dist.all_reduce(G, group=cp.group)
            _count("halo_bytes", G[0].numel() * G.element_size() * (cp.world - 1))
        gx = gh.new_zeros((B, T, C))
        take = min(T, n)
        if take:
            gx[:, T - take:] = G[cp.rank, :, n - take:].to(gh.dtype)
        return gx, None, None, None, None


# ------------------------------------------------------------------------------------------------ GDN kernels
class TorchGDN:
    """transformers' torch_chunk_gated_delta_rule (the model's CPU path): q / k [B, T, Hk, K] (L2-normalised), v [B, T,
    Hv, V], a (gate input) and beta [B, T, Hv]; the gate -exp(A_log) softplus(a + dt_bias) as the model's."""

    def _run(self, q, k, v, a, beta, A_log, dt_bias, h0):
        from transformers.models.qwen4_exp import modeling_qwen4_exp as m

        rep = v.shape[2] // q.shape[2]
        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
        if rep > 1:
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        return m.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=h0, output_final_state=True,
                                              use_qk_l2norm_in_kernel=False)

    def local(self, inp):
        q, k, v, a, beta, A_log, dt_bias = inp
        B, T, Hv, V = v.shape
        K = k.shape[-1]
        eye = torch.eye(K, dtype=torch.float32, device=v.device).expand(B, Hv, K, K).contiguous()
        if T == 0:
            return v.new_zeros((B, Hv, K, V), dtype=torch.float32), eye, ()
        with torch.no_grad():
            s_loc = self._run(q, k, v, a, beta, A_log, dt_bias, None)[1]
            m = self._run(q, k, v.new_zeros((B, T, Hv, K)), a, beta, A_log, dt_bias, eye)[1]
        return s_loc.float(), m.float(), ()

    def out(self, inp, s_in, pre):
        if inp[2].shape[1] == 0:
            return inp[2].new_zeros(inp[2].shape)
        with torch.no_grad():
            return self._run(*inp, s_in)[0]

    def _grad(self, inp, s_in, do, dht, needs):
        leaves = [t.detach().requires_grad_(need) if torch.is_floating_point(t) else t for t, need in zip(inp, needs)]
        h = s_in.detach().requires_grad_(True)
        with torch.enable_grad():
            o, final = self._run(*leaves, h)
            want = [h] + [t for t, need in zip(leaves, needs) if need]
            outs, grads = [o], [do]
            if dht is not None:
                outs.append(final)
                grads.append(dht)
            got = torch.autograd.grad(outs, want, grads, allow_unused=True)
        it = iter(got[1:])
        return got[0], [next(it) if need else None for need in needs]

    def dh0(self, inp, s_in, do, pre):
        if inp[2].shape[1] == 0:
            return torch.zeros_like(s_in)
        return self._grad(inp, s_in, do, None, [False] * 7)[0]

    def bwd(self, inp, s_in, do, dht, pre, needs):
        if inp[2].shape[1] == 0:
            return [torch.zeros_like(t) if need else None for t, need in zip(inp, needs)]
        return self._grad(inp, s_in, do, dht, needs)[1]


class FlaGDN:
    """fla's chunk kernels with rl1_gdn's settings (use_gate_in_kernel, chunk 64, GVA q / k at the key heads, the
    pinned o-kernel config). Calls only the fla entry points rl1_gdn already runs (fwd_keep / bwd_keep / the full
    backward). Not exercised on CPU: the GPU check is tests/test_rl1_cp.py::test_fla_simulated (skipped without fla)."""

    def _C(self):
        import fla.ops.gated_delta_rule.chunk as C
        import rl1_gdn

        rl1_gdn.pin_fla()
        return C

    def _gate(self, C, inp):
        q, k, v, a, beta, A_log, dt_bias = inp
        g = C.gdn_gate_chunk_cumsum(g=a, A_log=A_log, chunk_size=64, scale=C.RCP_LN2, dt_bias=dt_bias,
                                    cu_seqlens=None, chunk_indices=None)
        w, u, A = C.chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, cu_seqlens=None, chunk_indices=None,
                                                     chunk_size=64)
        return g, w, u, A

    def _h(self, C, k, w, u, g, h0):
        return C.chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g, initial_state=h0, output_final_state=True,
                                              cu_seqlens=None, chunk_indices=None, state_v_first=False, chunk_size=64)

    def local(self, inp):
        q, k, v, a, beta, A_log, dt_bias = inp
        B, T, Hv, V = v.shape
        K = k.shape[-1]
        eye = torch.eye(K, dtype=torch.float32, device=v.device).expand(B, Hv, K, K).contiguous()
        if T == 0:
            return v.new_zeros((B, Hv, K, V), dtype=torch.float32), eye, None
        C = self._C()
        g, w, u, A = self._gate(C, inp)
        s_loc = self._h(C, k, w, u, g, None)[2]
        m = self._h(C, k, w, u.new_zeros((B, T, Hv, K)), g, eye)[2]
        return s_loc, m, (g, w, u, A)

    def out(self, inp, s_in, pre):
        q, k, v = inp[:3]
        if v.shape[1] == 0:
            return v.new_zeros(v.shape)
        C = self._C()
        g, w, u, A = pre
        h, v_new, _ = self._h(C, k, w, u, g, s_in)
        return C.chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=k.shape[-1] ** -0.5, cu_seqlens=None,
                             chunk_indices=None, state_v_first=False, chunk_size=64)

    def saved(self, pre):  # what the backward keeps of ``local``'s (g, w, u, A): g and A (rl1_gdn's default)
        return () if pre is None else (pre[0], pre[3])

    def dh0(self, inp, s_in, do, pre):
        q, k, v, a, beta = inp[:5]
        if v.shape[1] == 0:
            return torch.zeros_like(s_in)
        C = self._C()
        g, A = pre
        w, _, _ = C.chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, cu_seqlens=None, chunk_indices=None,
                                                     chunk_size=64)
        scale = k.shape[-1] ** -0.5
        dv = C.chunk_bwd_dv_local(q=q, k=k, g=g, do=do, scale=scale, cu_seqlens=None, chunk_indices=None,
                                  chunk_size=64)
        _, dh0, _ = C.chunk_gated_delta_rule_bwd_dhu(q=q, k=k, w=w, g=g, h0=s_in, dht=None, do=do, dv=dv,
                                                     scale=scale, cu_seqlens=None, chunk_indices=None,
                                                     state_v_first=False, chunk_size=64)
        return dh0

    def bwd(self, inp, s_in, do, dht, pre, needs):
        q, k, v, a, beta, A_log, dt_bias = inp
        if v.shape[1] == 0:
            return [torch.zeros_like(t) if need else None for t, need in zip(inp, needs)]
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd

        self._C()
        g, A = pre
        # (as GatedDeltaFast.backward: with use_gate_in_kernel the 5th grad is d(g_input), i.e. d a)
        dq, dk, dv, db, da, _, dA_log, ddt = chunk_gated_delta_rule_bwd(
            q=q, k=k, v=v, g=g, beta=beta, A=A, scale=k.shape[-1] ** -0.5, initial_state=s_in, do=do, dht=dht,
            use_gate_in_kernel=True, g_input=a, A_log=A_log, dt_bias=dt_bias)[:8]
        return [t if need else None for t, need in zip((dq, dk, dv, da, db, dA_log, ddt), needs)]


def fold(ms, ss, h0, upto):
    """S_in of shard ``upto`` (and with upto = world the window's final state): S <- M_j S + S_j from h0."""
    s = h0
    for j in range(upto):
        s = ss[j] if s is None else torch.matmul(ms[j], s) + ss[j]
    return s


def fold_back(ms, ds, rank):
    """dh_T of shard ``rank``: dh0 of the later shards folded back through M^T (None for the last shard)."""
    dht = None
    for j in range(len(ms) - 1, rank, -1):
        dht = ds[j] if dht is None else ds[j] + torch.matmul(ms[j].transpose(-1, -2), dht)
    return dht


class GDNCore(torch.autograd.Function):
    """The gated delta rule over this rank's shard, exact for the whole segment (see the module doc). Inputs as the
    kernel's (``TorchGDN`` / ``FlaGDN``); ``h0`` the prefix's state (no grad). Returns o [B, T, Hv, V]."""

    @staticmethod
    def forward(ctx, cp, kernel, h0, key, q, k, v, a, beta, A_log, dt_bias):
        inp = (q, k, v, a, beta, A_log, dt_bias)
        s_loc, m, pre = kernel.local(inp)
        both = cp.gather_fixed(torch.stack([m, s_loc]) if m.shape == s_loc.shape else torch.cat(
            [m.flatten(), s_loc.flatten()]))
        _count("state_bytes", (m.numel() + s_loc.numel()) * 4 * (cp.world - 1))
        if m.shape == s_loc.shape:
            ms, ss = [b[0] for b in both], [b[1] for b in both]
        else:
            ms = [b[:m.numel()].view(m.shape) for b in both]
            ss = [b[m.numel():].view(s_loc.shape) for b in both]
        h0 = h0.float() if h0 is not None else torch.zeros_like(s_loc)
        s_in = fold(ms, ss, h0, cp.rank)
        if cp.capture and key is not None:
            cp.captured.setdefault(key, {})["state"] = fold(ms, ss, h0, cp.world).detach().clone()
        o = kernel.out(inp, s_in, pre)
        keep = kernel.saved(pre) if hasattr(kernel, "saved") else ()
        ctx.save_for_backward(*inp, s_in, *keep)
        ctx.ms, ctx.cp, ctx.kernel, ctx.nkeep = torch.stack(ms), cp, kernel, len(keep)
        return o

    @staticmethod
    def backward(ctx, do):
        saved = ctx.saved_tensors
        inp, s_in, keep = saved[:7], saved[7], saved[8:]
        cp, kernel = ctx.cp, ctx.kernel
        do = do.contiguous()
        pre = keep if keep else None
        d_loc = kernel.dh0(inp, s_in, do, pre).float()
        ds = cp.gather_fixed(d_loc)
        _count("state_bytes", d_loc.numel() * 4 * (cp.world - 1))
        dht = fold_back(list(ctx.ms), ds, cp.rank)
        needs = list(ctx.needs_input_grad[4:])
        grads = kernel.bwd(inp, s_in, do, dht, pre, needs)
        return (None, None, None, None, *grads)


def gdn_kernel(x):
    """FlaGDN on CUDA when fla imports, else the torch reference."""
    if x.is_cuda:
        try:
            import fla.ops.gated_delta_rule.chunk  # noqa: F401

            return FlaGDN()
        except ImportError:
            pass
    return TorchGDN()


def _l2norm(x):  # the model's l2norm (fp32), rounded to the input dtype once (as rl1_gdn's conv kernel)
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + 1e-6)).to(x.dtype)


# ------------------------------------------------------------------------------------------------ patched layers
def gdn_forward(mod, hidden_states, cp):
    """Qwen4ExpTextGatedDeltaNet.forward over this rank's shard (no cache, no padding mask), exact for the window."""
    from transformers.activations import ACT2FN

    B, T, _ = hidden_states.shape
    li = mod.layer_idx
    pre = cp.layer(li)
    y = mod.in_proj_qkv(hidden_states)
    hk, hv = mod.num_k_heads, mod.num_v_heads  # explicit head counts: reshape(-1) is ambiguous on an empty shard
    z = mod.in_proj_z(hidden_states).reshape(B, T, hv, mod.head_v_dim)
    b = mod.in_proj_b(hidden_states)
    a = mod.in_proj_a(hidden_states)
    halo = Halo.apply(y, mod.conv_kernel_size - 1, cp, pre.get("conv"), (li, "conv"))
    w = mod.conv1d.weight  # [C, 1, 4]
    x = torch.cat([halo, y], dim=1).transpose(1, 2).to(w.dtype)
    # an empty shard: no conv (input shorter than the kernel), but the halo stays in the graph (its backward is a
    # collective every rank joins)
    conv = F.conv1d(x, w, mod.conv1d.bias, padding=0, groups=mod.conv_dim) if T else x[..., :0]
    mixed = ACT2FN[mod.activation](conv).to(y.dtype).transpose(1, 2)
    q, k, v = torch.split(mixed, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
    q = _l2norm(q.reshape(B, T, hk, mod.head_k_dim))
    k = _l2norm(k.reshape(B, T, hk, mod.head_k_dim))
    v = v.reshape(B, T, hv, mod.head_v_dim)
    beta = b.sigmoid()
    kernel = gdn_kernel(hidden_states)
    if isinstance(kernel, FlaGDN):
        q, k, v, a, beta = (t.contiguous() for t in (q, k, v, a, beta))
    o = GDNCore.apply(cp, kernel, pre.get("state"), li, q, k, v, a, beta, mod.A_log, mod.dt_bias)
    core = mod.norm(o.reshape(-1, mod.head_v_dim), z.reshape(-1, mod.head_v_dim))
    return mod.out_proj(core.reshape(B, T, mod.value_dim))


class _IndexerKeys:
    """qsa_select's cache argument: ``update_indexer`` returns the window's indexer keys (prefix + every shard,
    all-gathered without grad) for this shard's own keys."""

    def __init__(self, cp, prefix_keys, key):
        self.cp, self.prefix, self.key = cp, prefix_keys, key

    def update_indexer(self, raw_keys, layer_idx):
        with torch.no_grad():
            keys = self.cp.gather_rows(raw_keys, 1)
            if self.prefix is not None:
                keys = torch.cat([self.prefix.to(keys.dtype), keys], dim=1)
        if self.cp.capture and self.key is not None:
            self.cp.captured.setdefault(self.key, {})["idx"] = keys.detach().clone()
        return keys


_KERNEL = {"torch": None}


def attention_kernel():
    """The trainer's attention kernel for its SPARSE mode: (class, gathered K/V dtype or None = the input's)."""
    import rl1_trainer as rt

    if rt.SPARSE["mode"] in ("triton", "triton2"):
        return rt.triton_kernel(), None
    if _KERNEL["torch"] is None:
        _KERNEL["torch"] = rt.selected_attention_function()
    return _KERNEL["torch"], torch.float32  # the torch kernel's products run in fp32: same values, fp32 dK / dV


def select(indexer, hidden_states, cp, prefix_keys, li):
    """The trainer's qsa_select for this shard's queries over the window's keys (global positions)."""
    import rl1_trainer as rt

    keys = _IndexerKeys(cp, prefix_keys, li)
    start = cp.P0 + cp.offset
    end = start + cp.T
    if cp.T == 0:  # no queries; the indexer-key all-gather still runs (every rank joins it)
        with torch.no_grad():
            keys.update_indexer(hidden_states.new_zeros((1, 0, indexer.index_head_dim)), li)
        return torch.full((0, indexer.compress_ratio - 1), -1, dtype=torch.long, device=hidden_states.device)
    counts = torch.arange(start + 1, end + 1, device=hidden_states.device)
    cos, sin = cp.rope
    return rt.qsa_select(indexer, hidden_states, (cos[:, :end], sin[:, :end]), counts, keys)


def attention_forward(mod, hidden_states, position_embeddings, cp):
    """Qwen4ExpTextAttention.forward (the trainer's selected path) over this rank's shard against the window's keys."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    if hidden_states.shape[0] != 1:
        raise ValueError("CP attention: batch 1")
    li = mod.layer_idx
    pre = cp.layer(li)
    picked = select(mod.indexer, hidden_states, cp, pre.get("idx"), li)
    input_shape = hidden_states.shape[:-1]
    heads, groups = mod.config.num_attention_heads, mod.config.num_key_value_heads  # explicit (empty shards)
    query_states, gate = torch.chunk(mod.q_proj(hidden_states).view(*input_shape, heads, mod.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, heads * mod.head_dim)
    query_states = mod.q_norm(query_states.reshape(*input_shape, heads, mod.head_dim)).transpose(1, 2)
    key_states = mod.k_norm(mod.k_proj(hidden_states).view(*input_shape, groups, mod.head_dim)).transpose(1, 2)
    value_states = mod.v_proj(hidden_states).view(*input_shape, groups, mod.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = modeling.apply_rotary_pos_emb(query_states, key_states, cos, sin)
    kernel, dtype = attention_kernel()
    dtype = dtype or key_states.dtype
    keys = GatherSeq.apply(key_states, 2, cp, dtype)
    values = GatherSeq.apply(value_states, 2, cp, dtype)
    if pre.get("k") is not None:
        keys = torch.cat([pre["k"].to(dtype), keys], dim=2)
        values = torch.cat([pre["v"].to(dtype), values], dim=2)
    if cp.capture:
        cp.captured.setdefault(li, {}).update(k=keys.detach().to(key_states.dtype).clone(),
                                              v=values.detach().to(value_states.dtype).clone())
    union = hasattr(kernel, "NOGRAD_PREFIX")  # rl1_sparse2: under CP the no-grad prefix is P0 keys, not keys - L
    if union:
        kernel.P0 = cp.P0
    try:
        import rl1_trainer as rt

        out = kernel.apply(query_states, keys, values, picked, mod.scaling,
                           rt.SPARSE["chunk"] if kernel is _KERNEL["torch"] else None)
    finally:
        if union:
            kernel.P0 = None
    out = out.to(query_states.dtype).transpose(1, 2).reshape(*input_shape, heads * mod.head_dim).contiguous()
    out = out * torch.sigmoid(gate)
    return mod.o_proj(out)


class _ContextIds:
    """The n-gram embedding's cache argument: the ids before this shard as its ``conv_states[2]``."""

    def __init__(self, li, ids):
        self.layers = {li: types.SimpleNamespace(conv_states={2: ids})}

    def has_previous_state(self, layer_idx, state_idx=0):
        return True

    def update_conv_state(self, *args, **kwargs):
        return None


def context_ids(cp, n, eos, like):
    """The ``n`` token ids before this rank's first token: prefix ids, then the earlier shards' (eos-padded)."""
    parts = []
    if cp.prefix.ids is not None:
        parts.append(cp.prefix.ids.to(like.device))
    if cp.offset:
        parts.append(cp.ids[:, :cp.offset].to(like.device))
    hist = torch.cat(parts, dim=1) if parts else like.new_zeros((like.shape[0], 0))
    hist = hist[:, hist.shape[1] - min(n, hist.shape[1]):]
    if hist.shape[1] < n:
        hist = torch.cat([hist.new_full((hist.shape[0], n - hist.shape[1]), eos), hist], dim=1)
    return hist.long()


def ngram_forward(original, mod, input_ids, cp):
    if input_ids.shape[1] == 0:  # the model's [:, -0:] slice would return the context rows
        dim = mod.ngram_embedding.embedding_dim * mod.ngram_heads
        return mod.ngram_embedding.weight.new_zeros((input_ids.shape[0], 0, dim))
    return original(mod, input_ids, _ContextIds(mod.layer_idx, context_ids(cp, mod.context_len, mod.eos_token_id,
                                                                          input_ids)))


def short_conv(mod, hidden_states, cp):
    """PLELayer._short_conv: the dilated depthwise conv over the 9 preceding rows (``Halo``) and the shard."""
    n = mod.short_conv_state_len
    halo = Halo.apply(hidden_states, n, cp, cp.layer(mod.layer_idx).get("ple"), (mod.layer_idx, "ple"))
    x = torch.cat([halo, hidden_states], dim=1).transpose(1, 2)
    return F.silu(mod.conv1d(x) if hidden_states.shape[1] else x[..., :0]).transpose(1, 2)  # (empty: as gdn_forward)


def install():
    """Wrap the GDN, attention, n-gram embedding and PLE short-conv forwards (outermost; re-wrapped if another patch
    replaced them since); each runs the CP path while ``CP["ctx"]`` is set, the previous forward otherwise."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    def wrap(cls, name, make):
        current = getattr(cls, name)
        if getattr(current, "_rl1_cp", False):
            return
        new = make(current)
        new._rl1_cp = True
        setattr(cls, name, new)

    def gdn(original):
        def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
            cp = CP["ctx"]
            if cp is None:
                return original(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask,
                                **kwargs)
            return gdn_forward(self, hidden_states, cp)
        return forward

    def attn(original):
        def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
            cp = CP["ctx"]
            if cp is None:
                return original(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
            return attention_forward(self, hidden_states, position_embeddings, cp), None
        return forward

    def ngram(original):
        def forward(self, input_ids, past_key_values=None):
            cp = CP["ctx"]
            if cp is None:
                return original(self, input_ids, past_key_values)
            return ngram_forward(original, self, input_ids, cp)
        return forward

    def conv(original):
        def forward(self, hidden_states, past_key_values=None):
            cp = CP["ctx"]
            if cp is None:
                return original(self, hidden_states, past_key_values)
            return short_conv(self, hidden_states, cp)
        return forward

    def norm(original):
        def _norm(self, x):  # the grouped norm's reshape(-1) fails on an empty shard's 0 rows
            if CP["ctx"] is not None and x.numel() == 0:
                return x * 1.0
            return original(self, x)
        return _norm

    wrap(m.Qwen4ExpTextRMSNorm, "_norm", norm)
    wrap(m.Qwen4ExpTextGatedDeltaNet, "forward", gdn)
    wrap(m.Qwen4ExpTextAttention, "forward", attn)
    wrap(m.Qwen4ExpTextNGramEmbedding, "forward", ngram)
    wrap(m.Qwen4ExpTextPLELayer, "_short_conv", conv)


# ------------------------------------------------------------------------------------------------ model level
class NoMasks(dict):
    def __missing__(self, key):
        return None


def _rope_ids(position_ids, length, device):
    """The rotary input [3, B, N] from window positions in any form the model's forward takes."""
    if position_ids is None:
        return torch.arange(length, device=device).view(1, 1, -1).expand(3, 1, -1)
    pos = position_ids.to(device)
    if pos.dim() == 2:
        pos = pos[None].expand(4, *pos.shape)
    if pos.shape[0] == 4:
        return pos[1:]
    if pos.shape[0] == 1:
        return pos.expand(3, -1, -1)
    return pos


def _shard_positions(cp, device):
    start = cp.P0 + cp.offset
    if cp.position_ids is None:
        return (torch.arange(cp.T, device=device) + start).view(1, 1, -1).expand(4, 1, -1)
    return cp.position_ids.to(device)[..., start:start + cp.T]


def forward(model, cp, input_ids, **kwargs):
    """This rank's hidden states [B, T, H] (the text model's last_hidden_state) for its shard ``input_ids``.
    With gradient checkpointing the backward re-runs the layers: run forward and backward inside ``active(cp)``
    (the recompute must see the same context; outside it the layers run the model's / trainer's code)."""
    text = model.model if hasattr(model, "model") and hasattr(model.model, "rotary_emb") else model
    if input_ids.shape[1] != cp.T:
        raise ValueError(f"rank {cp.rank}: {input_ids.shape[1]} ids for a shard of {cp.T}")
    device = input_ids.device
    with (contextlib.nullcontext() if CP["ctx"] is cp else active(cp)):
        cp.rope = text.rotary_emb(text.embed_tokens.weight[:1], _rope_ids(cp.position_ids, cp.P0 + cp.total, device))
        out = text(input_ids=input_ids, position_ids=_shard_positions(cp, device), attention_mask=NoMasks(),
                   use_cache=False, **kwargs)
    return out.last_hidden_state


def prefill(model, cp, input_ids):
    """The carried-in context as a replicated Prefix: this sharded no-grad pass (each rank its shard ``input_ids``,
    ``cp`` over the context's tokens; its own prefix, if any, continues). Returns the Prefix for the next Ctx."""
    cp.capture = True
    with torch.no_grad():
        forward(model, cp, input_ids)
    return _prefix_out(cp)


def _prefix_out(cp):
    parts = [t for t in (cp.prefix.ids, cp.ids) if t is not None]
    ids = torch.cat(parts, dim=1)[:, -8:] if parts else None
    return Prefix(cp.P0 + cp.total, ids, {li: dict(d) for li, d in cp.captured.items()})


_ROTARY = {}


def rotary(model):
    """The text rotary embedding of any Flash wrapper (text model, causal LM, conditional generation, peft)."""
    key = id(model)
    if key not in _ROTARY:
        _ROTARY[key] = next(m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextRotaryEmbedding")
    return _ROTARY[key]


def call(model, cp, **kwargs):
    """The whole model (any wrapper; the trainer's ``self.model``) on this rank's shard. ``kwargs`` as the model takes
    them for the shard: ``input_ids`` [B, T], its ``position_ids`` (default: the shard's slice of ``cp.position_ids``),
    ``mm_encoder_outputs``, ``logits_to_keep`` ...; masks are the CP layers' own (no mask is built). Run forward and
    backward inside ``active(cp)`` (checkpoint recompute)."""
    ids = kwargs["input_ids"]
    if ids.shape[1] != cp.T:
        raise ValueError(f"rank {cp.rank}: {ids.shape[1]} ids for a shard of {cp.T}")
    kwargs.setdefault("position_ids", _shard_positions(cp, ids.device))
    with (contextlib.nullcontext() if CP["ctx"] is cp else active(cp)):
        cp.rope = rotary(model)(model.get_input_embeddings().weight[:1],
                                _rope_ids(cp.position_ids, cp.P0 + cp.total, ids.device))
        return model(attention_mask=NoMasks(), use_cache=False, **kwargs)


def prefill_call(model, cp, **kwargs):
    """``prefill`` through ``call`` (images, the trainer's model): the Prefix for the next Ctx."""
    cp.capture = True
    kwargs.setdefault("logits_to_keep", 1)
    with torch.no_grad():
        call(model, cp, **kwargs)
    return _prefix_out(cp)


def shard_lengths(length, world, grain=64):
    """Contiguous shard lengths of ``length`` tokens over ``world`` ranks, every cut on the ``grain`` grid (fla's
    64-token chunks: a shard's chunks are the single pass's); the last ranks may get fewer or none."""
    per = -(-int(length) // world)
    per = -(-per // grain) * grain
    out, left = [], int(length)
    for _ in range(world):
        out.append(min(per, left))
        left -= out[-1]
    return out


class GatherCat(torch.autograd.Function):
    """Every rank's rows [n_r, ...] (any n_r, dim 0) concatenated in rank order, so every rank computes the same
    function of all of them (a loss over a whole CP window: GSPO's per-reply ratios, the token sums). Backward: this
    rank's rows' part of the gradient only (each rank's copy of that function is the same, so nothing is summed)."""

    @staticmethod
    def forward(ctx, x, cp):
        sizes = cp.gather_fixed(torch.tensor([x.shape[0]], dtype=torch.long, device=x.device))
        sizes = [int(s) for s in sizes]
        top = max(sizes)
        pad = x.new_zeros((top, *x.shape[1:]))
        pad[:x.shape[0]] = x
        parts = cp.gather_fixed(pad)
        ctx.start, ctx.n = sum(sizes[:cp.rank]), x.shape[0]
        return torch.cat([p[:n] for p, n in zip(parts, sizes)])

    @staticmethod
    def backward(ctx, g):
        return g[ctx.start:ctx.start + ctx.n], None


def gather_cat(x, cp):
    """``GatherCat`` (grad) or the plain gather of a tensor without grad."""
    if x.requires_grad:
        return GatherCat.apply(x, cp)
    with torch.no_grad():
        if x.dtype == torch.bool:  # (gloo has no bool collectives)
            return GatherCat.apply(x.to(torch.uint8), cp).bool()
        return GatherCat.apply(x, cp)
