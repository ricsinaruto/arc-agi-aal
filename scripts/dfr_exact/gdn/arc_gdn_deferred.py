"""GDN deferred commit (throughput arm, 19 Oct; M2a). Installed as ``vllm/arc_gdn_deferred.py``.

ARC_GDN_DEFER=real : pure spec-decode steps run the deferred kernel (one state write per step instead of one per
                     token; bit-exact with the pinned fused_gdn_decode_post_conv_mtp). Mixed steps first flush
                     deferred requests back to the pinned layout, then run the pinned path. Prefill rows clear their
                     block's flags.
ARC_GDN_DEFER=check: the pinned kernels run for real; the deferred kernel runs on a row-indexed shadow next to them
                     and every output, commit, near-boundary per-token state and flush is compared bit for bit
                     (counts in ARC_GDN_DEFER_STATS, written on mixed steps).
ARC_GDN_DEFER_LIB  : path to libgdn2.so (gdn_m2.cu built for sm_120).
"""

import ctypes
import json
import os
import time

import torch

MODE = os.environ.get("ARC_GDN_DEFER", "")
LIB_PATH = os.environ.get("ARC_GDN_DEFER_LIB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "arc_libgdn2.so")
STATS_PATH = os.environ.get("ARC_GDN_DEFER_STATS", "/tmp/arc_gdn_defer_stats.json")
CHECK_ROWS = int(os.environ.get("ARC_GDN_DEFER_CHECK_ROWS", "16"))
WIDTH = int(os.environ.get("ARC_GDN_DEFER_W", "4"))  # 1 + MTP tokens
STAT_KEYS = ("pure_calls", "rows", "valid_rows", "near_rows", "out_neq", "commit_neq", "near_tok_neq",
             "flush_calls", "flush_rows", "flush_neq", "prefill_clears", "skipped_big", "real_steps", "pinned_noseq",
             "pinned_width")

_LIB = None
_STATE = {}
_STATS = None
_LAST_DUMP = [0.0]


class Ptrs(ctypes.Structure):
    _fields_ = [("commit", ctypes.c_void_p), ("commit_index", ctypes.c_void_p), ("commit_stride", ctypes.c_int64),
                ("commit_index_stride", ctypes.c_int),
                ("buf", ctypes.c_void_p), ("buf_index", ctypes.c_void_p), ("buf_stride", ctypes.c_int64),
                ("buf_index_stride", ctypes.c_int),
                ("tok", ctypes.c_void_p), ("tok_index", ctypes.c_void_p), ("tok_stride", ctypes.c_int64),
                ("tok_width", ctypes.c_int),
                ("read_state", ctypes.c_void_p), ("read_stride", ctypes.c_int64),
                ("flags", ctypes.c_void_p), ("flag_index", ctypes.c_void_p), ("flag_index_stride", ctypes.c_int),
                ("ident", ctypes.c_void_p), ("ident_stride", ctypes.c_int),
                ("seq_lens", ctypes.c_void_p), ("block_size", ctypes.c_int)]


def lib():
    global _LIB
    if _LIB is None:
        so = ctypes.CDLL(LIB_PATH)
        f = so.gdn2_launch
        P, I, F, L = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_longlong
        f.argtypes = [I, I, P, P, P, P, P, I, P, I, P, P, P, P, P, I, P, I, I, I, F, F, L, L, L, L, L,
                      ctypes.POINTER(Ptrs), P]
        f.restype = I
        if so.gdn2_ptrs_size() != ctypes.sizeof(Ptrs):
            raise RuntimeError(f"DeferredPtrs size {so.gdn2_ptrs_size()} != {ctypes.sizeof(Ptrs)}")
        _LIB = f
    return _LIB


def stats(device):
    global _STATS
    if _STATS is None:
        _STATS = torch.zeros(len(STAT_KEYS), dtype=torch.int64, device=device)
    return _STATS


def bump(key, value):
    _STATS[STAT_KEYS.index(key)] += value


def dump(force=False):
    if _STATS is None or torch.cuda.is_current_stream_capturing():
        return
    now = time.time()
    if not force and now - _LAST_DUMP[0] < 20:
        return
    _LAST_DUMP[0] = now
    values = dict(zip(STAT_KEYS, _STATS.tolist()))
    values.update(mode=MODE, time=now)
    tmp = STATS_PATH + ".tmp"
    try:
        os.makedirs(os.path.dirname(STATS_PATH) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(values, f)
        os.replace(tmp, STATS_PATH)
    except OSError as e:
        print(f"ARC_GDN_DEFER stats dump failed: {e}", flush=True)
    print("ARC_GDN_DEFER_STATS " + json.dumps(values), flush=True)


def dtype_code(t):
    return {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}[t.dtype]


def layer_state(layer, state, W=WIDTH):
    """Per-layer flag table (real) or row-indexed shadow (check), allocated outside graph capture."""
    key = id(layer)
    st = _STATE.get(key)
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("arc_gdn_deferred: first call inside CUDA graph capture")
        HV = state.shape[1]
        dev = state.device
        st = {"flags": torch.zeros(state.shape[0] * HV * 2, dtype=torch.int32, device=dev)}
        if MODE == "check":
            R = CHECK_ROWS
            st["sh0"] = torch.zeros((R + 1,) + tuple(state.shape[1:]), dtype=state.dtype, device=dev)
            st["sh1"] = torch.zeros_like(st["sh0"])
            st["shtok"] = torch.zeros((R * W + 1,) + tuple(state.shape[1:]), dtype=state.dtype, device=dev)
            st["shflags"] = torch.zeros((R + 1) * HV * 2, dtype=torch.int32, device=dev)
            st["rows1"] = torch.arange(1, R + 1, dtype=torch.int32, device=dev)
            st["tokidx"] = (1 + torch.arange(R * W, dtype=torch.int32, device=dev)).contiguous()
            st["shout"] = None
        stats(dev)
        _STATE[key] = st
    return st


def launch(mode, layer, mixed_qkv, a, b, si, cu, na, state, gate, out, n, ptrs):
    f = lib()
    W = si.shape[1]
    rc = f(mode, int(layer.norm.activation == "sigmoid"), mixed_qkv.data_ptr() if mixed_qkv is not None else 0,
           a.data_ptr() if a is not None else 0, b.data_ptr() if b is not None else 0, layer.A_log.data_ptr(),
           layer.dt_bias.data_ptr(), dtype_code(layer.dt_bias), si.data_ptr(), W, cu.data_ptr(), na.data_ptr(),
           state.data_ptr(), gate.data_ptr() if gate is not None else 0, layer.norm.weight.data_ptr(),
           int(layer.norm.weight.dtype == torch.bfloat16), out.data_ptr() if out is not None else 0, n,
           state.shape[1] // 3, state.shape[1], float(layer.head_k_dim ** -0.5), float(layer.layer_norm_epsilon),
           mixed_qkv.stride(0) if mixed_qkv is not None else 0, a.stride(0) if a is not None else 0,
           b.stride(0) if b is not None else 0, gate.stride(0) if gate is not None else 0, state.stride(0),
           ctypes.byref(ptrs), torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"gdn2_launch mode {mode} failed: {rc}")


def real_ptrs(st, state, si, seq, block):
    W = si.shape[1]
    s = state.stride(0)
    return Ptrs(state.data_ptr(), si.data_ptr(), s, W, state.data_ptr(), si.data_ptr() + 4, s, W,
                state.data_ptr(), si.data_ptr(), s, W, state.data_ptr(), s,
                st["flags"].data_ptr(), si.data_ptr(), W, si.data_ptr(), W, seq.data_ptr(), block)


def shadow_ptrs(st, state, si, seq, block):
    W = si.shape[1]
    s0 = st["sh0"].stride(0)
    return Ptrs(st["sh0"].data_ptr(), st["rows1"].data_ptr(), s0, 1, st["sh1"].data_ptr(), st["rows1"].data_ptr(),
                s0, 1, st["shtok"].data_ptr(), st["tokidx"].data_ptr(), st["shtok"].stride(0), W,
                state.data_ptr(), state.stride(0), st["shflags"].data_ptr(), st["rows1"].data_ptr(), 1,
                si.data_ptr(), W, seq.data_ptr(), block)


def near_mask(seq, cu, n, W, block):
    T = (cu[1:n + 1] - cu[:n]).to(torch.int64)
    pos0 = seq[:n].to(torch.int64) - T
    lo = torch.clamp(pos0 - 2 * W - 2, min=0)
    hi = pos0 + 2 * W + 2
    return (hi // block) != (lo // block), T


def neq_rows(x, y):
    return (x.view(torch.int16) != y.view(torch.int16)).flatten(1).sum(1)


def pure_step(layer, mixed_qkv, b, a, output_gate, core_attn_out, md, pinned):
    si = md.spec_state_indices_tensor
    n = md.num_spec_decodes
    cu = md.spec_query_start_loc
    na = md.num_accepted_tokens
    seq = getattr(md, "arc_seq_lens", None)
    state = layer.kv_cache[1]
    stats(state.device)
    if seq is None or si.shape[1] < 2:
        bump("pinned_noseq", 1)  # a GPU add: counts graph replays too
        return pinned()
    block = md.arc_block
    W = si.shape[1]
    if W > WIDTH:
        bump("pinned_width", 1)
        return pinned()
    st = layer_state(layer, state)
    si_n = si[:n]
    if MODE == "real":
        launch(0, layer, mixed_qkv, a, b, si_n, cu, na, state, output_gate, core_attn_out, n,
               real_ptrs(st, state, si_n, seq, block))
        bump("real_steps", 1)
        return None
    # check mode
    bump("pure_calls", 1)
    if n > CHECK_ROWS:
        bump("skipped_big", 1)
        return pinned()
    HV = state.shape[1]
    num_tokens = core_attn_out.shape[0]
    if st["shout"] is None or st["shout"].shape[0] < num_tokens:
        if torch.cuda.is_current_stream_capturing():
            st["shout"] = torch.empty_like(core_attn_out)
        else:
            st["shout"] = torch.empty((max(num_tokens, CHECK_ROWS * W),) + tuple(core_attn_out.shape[1:]),
                                      dtype=core_attn_out.dtype, device=core_attn_out.device)
    shout = st["shout"][:num_tokens]
    near, T = near_mask(seq, cu, n, W, block)
    rowv = T > 0
    near = near & rowv
    acc = na[:n].to(torch.int64)
    pos0 = seq[:n].to(torch.int64) - T
    shf = st["shflags"].view(-1, HV, 2)[1:n + 1, 0]
    valid = (shf[:, 0] == si_n[:, 0]) & (shf[:, 1].to(torch.int64) == pos0 - acc + 1) & rowv
    launch(0, layer, mixed_qkv, a, b, si_n, cu, na, state, output_gate, shout, n,
           shadow_ptrs(st, state, si_n, seq, block))
    src_idx = si_n.gather(1, (acc - 1).clamp(min=0)[:, None]).flatten().long()
    commit_neq = (neq_rows(st["sh0"][1:n + 1], state.index_select(0, src_idx)) * (~near & rowv)).sum()
    pinned()
    tokv = torch.arange(num_tokens, device=T.device) < cu[n].to(torch.int64)
    out_neq = (neq_rows(shout, core_attn_out[:num_tokens]) * tokv).sum()
    tok_real = state.index_select(0, si_n.flatten().long()).view((n, W) + tuple(state.shape[1:]))
    tok_sh = st["shtok"][1:n * W + 1].view((n, W) + tuple(state.shape[1:]))
    tmask = (torch.arange(W, device=T.device)[None, :] < T[:, None]) & near[:, None]
    near_neq = ((tok_real.view(torch.int16) != tok_sh.view(torch.int16)).flatten(2).sum(2) * tmask).sum()
    for k, v in (("rows", n), ("valid_rows", valid.sum()), ("near_rows", near.sum()), ("out_neq", out_neq),
                 ("commit_neq", commit_neq), ("near_tok_neq", near_neq)):
        bump(k, v)
    return None


def mixed_prepare(layer, md):
    """Before the pinned mixed path: flush deferred spec rows to the pinned layout; clear flags of prefill rows."""
    state = layer.kv_cache[1]
    HV = state.shape[1]
    si = md.spec_state_indices_tensor
    n = md.num_spec_decodes
    seq = getattr(md, "arc_seq_lens", None)
    if si is not None and n > 0 and seq is not None and si.shape[1] >= 2:
        W = si.shape[1]
        st = layer_state(layer, state)
        si_n = si[:n].contiguous()
        cu = md.spec_query_start_loc
        na = md.num_accepted_tokens
        block = md.arc_block
        if MODE == "real":
            launch(1, layer, None, None, None, si_n, cu, na, state, None, None, n,
                   real_ptrs(st, state, si_n, seq, block))
        elif n <= CHECK_ROWS:
            bump("flush_calls", 1)
            acc = na[:n].to(torch.int64)
            T = (cu[1:n + 1] - cu[:n]).to(torch.int64)
            pos0 = seq[:n].to(torch.int64) - T
            shf = st["shflags"].view(-1, HV, 2)[1:n + 1, 0]
            valid = (shf[:, 0] == si_n[:, 0]) & (shf[:, 1].to(torch.int64) == pos0 - acc + 1)
            launch(1, layer, None, None, None, si_n, cu, na, state, None, None, n,
                   shadow_ptrs(st, state, si_n, seq, block))
            sh_rows = (1 + torch.arange(n, device=acc.device) * W + (acc - 1).clamp(min=0)).long()
            src_idx = si_n.gather(1, (acc - 1).clamp(min=0)[:, None]).flatten().long()
            neq = (neq_rows(st["shtok"].index_select(0, sh_rows), state.index_select(0, src_idx)) * valid).sum()
            bump("flush_rows", valid.sum())
            bump("flush_neq", neq)
    ns = md.non_spec_state_indices_tensor
    if md.num_prefills > 0 and ns is not None:
        st = layer_state(layer, state)
        ids = ns.flatten().long()
        st["flags"].view(-1, HV, 2)[ids] = 0
        if MODE == "check":
            shf = st["shflags"].view(-1, HV, 2)
            hit = torch.isin(shf[:, 0, 0], ns.flatten())
            shf[hit] = 0
        bump("prefill_clears", ids.numel())
    dump()


def install_layer(cls):
    if not MODE or getattr(cls, "_arc_gdn_defer", False):
        return
    pinned_pure = cls._forward_core_decode_spec_post_conv_fused_norm
    pinned_core = cls._forward_core

    def _forward_core_decode_spec_post_conv_fused_norm(self, mixed_qkv, b, a, output_gate, core_attn_out,
                                                         attn_metadata):
        def pinned():
            return pinned_pure(self, mixed_qkv=mixed_qkv, b=b, a=a, output_gate=output_gate,
                               core_attn_out=core_attn_out, attn_metadata=attn_metadata)
        return pure_step(self, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata, pinned)

    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
        from vllm.forward_context import get_forward_context
        raw = get_forward_context().attn_metadata
        if isinstance(raw, dict) and self.prefix in raw:
            mixed_prepare(self, raw[self.prefix])
        return pinned_core(self, mixed_qkv=mixed_qkv, b=b, a=a, core_attn_out=core_attn_out)

    cls._forward_core_decode_spec_post_conv_fused_norm = _forward_core_decode_spec_post_conv_fused_norm
    cls._forward_core = _forward_core
    cls._arc_gdn_defer = True
    print(f"ARC_GDN_DEFER installed mode={MODE} lib={LIB_PATH} pid={os.getpid()}", flush=True)


def install_builder(cls):
    if not MODE or getattr(cls, "_arc_gdn_defer", False):
        return
    pinned_build = cls.build

    def build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens=None,
              num_decode_draft_tokens_cpu=None, fast_build=False):
        md = pinned_build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens=num_accepted_tokens,
                          num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu, fast_build=fast_build)
        n = md.num_spec_decodes
        if n > 0:
            m = common_attn_metadata
            if md.num_prefills == 0 and md.num_decodes == 0:
                md.arc_seq_lens = m.seq_lens[:n]
            else:
                mask = num_decode_draft_tokens_cpu >= 0
                idx = torch.nonzero(mask).flatten().to(m.seq_lens.device, non_blocking=True)
                md.arc_seq_lens = m.seq_lens.index_select(0, idx).contiguous()
            md.arc_block = int(self.kv_cache_spec.block_size)
        return md

    cls.build = build
    cls._arc_gdn_defer = True
