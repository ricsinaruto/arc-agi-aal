"""Query-union selected-key attention in Triton (rl1 trainer, ``RL1_ATTENTION=triton2``): a drop-in for
``rl1_sparse.TritonSelectedAttention`` (same arguments, same outputs, same selection semantics as ``reference``).

The current kernel runs one program per (query, KV group): 12 heads padded to 16 MMA rows, every query gathers its own
2,051 keys, and the backward adds dk / dv for every (query, key) pair with fp32 atomics. Here a program owns a tile of
``BM`` (query, head) rows of one KV group (BM = 64: 5 whole queries = 60 rows, or with ``QT`` = 0 64 consecutive
rows that may split a query between two tiles) and walks the *union*
of its queries' selections; a per-key bitmask says which of the tile's queries picked the key (exact: unselected
(row, key) scores are -inf). Neighbouring queries share most of their picks (measured on served picks: union of
8 consecutive rows = 2.79x one row's), so each gathered key serves several queries, the MMAs are full-size, and the
backward issues one atomic per (tile, union key) instead of one per (query, key); dq is written once per row.

Union build (per call, saved for the backward; no gradient): picks are grouped in aligned 4-key blocks (the indexer
selects whole compress-ratio-4 blocks; arbitrary picks still work, one entry per key), a per-tile bitmap over the
key blocks marks the union (atomic or), a popcount scan gives each tile's sorted block list, and a second pass ORs
each query's 4-bit key mask into its slot of the block's member word (slot = query - first query of the tile).

Forward: online softmax in base 2, K / V gathered by Blackwell TMA (``tma``) in a warp-specialized loop (``ws``).
Backward: one fused kernel (dq in tensor memory, written once per row; dk / dv per union key with fp32 atomics, so
dk / dv are not bitwise deterministic run to run; out and dq are)."""
import torch
import triton
import triton.language as tl

KB = 4  # key block of the union lists (the indexer's compress ratio; any picks are valid)


@triton.jit
def _popc(x):  # popcount of uint32 lanes
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return (x * 0x01010101) >> 24


@triton.jit
def _tile_of(t, REP: tl.constexpr, BM: tl.constexpr, QT: tl.constexpr):
    """First and last tile holding query t's rows, and t's slot in a tile ``tile``: see ``_rows``."""
    if QT > 0:
        ja = t // QT
        jb = ja
    else:
        ja = (t * REP) // BM
        jb = (t * REP + REP - 1) // BM
    return ja, jb


@triton.jit
def _first_query(tile, REP: tl.constexpr, BM: tl.constexpr, QT: tl.constexpr):
    if QT > 0:
        first = tile * QT
    else:
        first = (tile * BM) // REP
    return first


@triton.jit
def _scan(P, S, spl, KEYS, BMAP, W, OFF, PREF, MASK,
          REP: tl.constexpr, BM: tl.constexpr, QT: tl.constexpr, BS: tl.constexpr, MEMBER: tl.constexpr):
    """Query t, picks [c0, c0 + BS): the first pick of each (aligned 4-pick group, key block) pair leads with the
    group's sub-key mask for that block. MEMBER False: mark the block in the bitmaps of the (1 or 2) tiles holding
    t's rows; True: OR the sub-key mask into t's slot of the block's member word."""
    t = tl.program_id(0)
    i = tl.program_id(1) * BS + tl.arange(0, BS)
    key = tl.load(P + t * spl + i, mask=i < S, other=-1)
    ok = (key >= 0) & (key < KEYS)
    blk = tl.where(ok, key // 4, -1)
    lead = ok
    sub = tl.zeros([BS], tl.int32)
    base = i - i % 4
    for o in tl.static_range(4):
        j = base + o
        kj = tl.load(P + t * spl + j, mask=j < S, other=-1)
        same = ok & (kj >= 0) & (kj < KEYS) & (kj // 4 == blk)
        lead = lead & ~(same & (j < i))
        sub = sub | tl.where(same, 1 << (kj % 4).to(tl.int32), 0)
    ja, jb = _tile_of(t, REP, BM, QT)
    word = tl.where(lead, blk // 32, 0)
    bit = tl.full([BS], 1, tl.uint32) << (blk % 32).to(tl.uint32)
    for jj in tl.static_range(2):
        tile = ja + jj
        go = lead & (tile <= jb)
        if MEMBER:
            w = tl.load(BMAP + tile * W + word, mask=go, other=0).to(tl.uint32, bitcast=True)
            pos = tl.load(OFF + tile) + tl.load(PREF + tile * W + word, mask=go, other=0) + _popc(w & (bit - 1)).to(tl.int32)
            slot = t - _first_query(tile, REP, BM, QT)
            mt = MASK.dtype.element_ty
            tl.atomic_or(MASK + pos, sub.to(mt) << (slot * 4).to(mt), mask=go, sem="relaxed")
        else:
            tl.atomic_or(BMAP + tile * W + word, bit.to(tl.int32, bitcast=True), mask=go, sem="relaxed")


@triton.jit
def _count(BMAP, W, PREF, TOT, CW: tl.constexpr):
    """Per tile: exclusive popcount prefix over its bitmap words, and the tile's union size."""
    tile = tl.program_id(0)
    run = tile * 0
    for w0 in range(0, W, CW):
        w = w0 + tl.arange(0, CW)
        x = tl.load(BMAP + tile * W + w, mask=w < W, other=0).to(tl.uint32, bitcast=True)
        pc = _popc(x).to(tl.int32)
        tl.store(PREF + tile * W + w, run + tl.cumsum(pc, 0) - pc, mask=w < W)
        run += tl.sum(pc, 0)
    tl.store(TOT + tile, run)


@triton.jit
def _expand(BMAP, W, OFF, PREF, UL, CW: tl.constexpr):
    """Per (tile, word chunk): the set bits, ascending, as block ids into the tile's union list."""
    tile = tl.program_id(0)
    w = tl.program_id(1) * CW + tl.arange(0, CW)
    x = tl.load(BMAP + tile * W + w, mask=w < W, other=0).to(tl.uint32, bitcast=True)
    pos = tl.load(OFF + tile) + tl.load(PREF + tile * W + w, mask=w < W, other=0)
    for b in tl.static_range(32):
        on = ((x >> b) & 1).to(tl.int32)
        tl.store(UL + pos, w * 32 + b, mask=on != 0)
        pos += on


def tiling(length, rep, bm, aligned):
    """``(tiles, qt, slots)``: tile count, whole queries per tile (0: rows run on across queries) and the most
    queries one tile touches (4 member bits each, at most 16)."""
    if aligned:
        qt = bm // rep
        return triton.cdiv(length, qt), qt, qt
    return triton.cdiv(length * rep, bm), 0, (bm - 1) // rep + 2


def union_tiles(picked, keys, rep, bm, aligned=True):
    """``(ulist, member, offsets)``: per tile its union of key blocks (ascending int32 block ids, CSR by ``offsets``
    [tiles + 1]) and per block the member word (bit 4 * slot + key % 4; int32 when the slots fit, else int64)."""
    length, width = picked.shape
    tiles, qt, nslot = tiling(length, rep, bm, aligned)
    assert nslot * 4 <= 64 and (qt > 0 or not aligned), "member word: 1 to 16 queries per tile"
    words = triton.cdiv(triton.cdiv(keys, KB), 32)
    dev = picked.device
    bmap = torch.zeros((tiles, words), dtype=torch.int32, device=dev)
    pref = torch.empty((tiles, words), dtype=torch.int32, device=dev)
    tot = torch.empty((tiles,), dtype=torch.int32, device=dev)
    offsets = torch.zeros((tiles + 1,), dtype=torch.int32, device=dev)
    bs = 256
    grid = (length, triton.cdiv(width, bs))
    _scan[grid](picked, width, picked.stride(0), keys, bmap, words, offsets, pref, bmap,
                REP=rep, BM=bm, QT=qt, BS=bs, MEMBER=False)
    _count[(tiles,)](bmap, words, pref, tot, CW=1024)
    offsets[1:] = torch.cumsum(tot, 0, dtype=torch.int32)
    total = int(offsets[-1])  # one sync: the union list's size
    ulist = torch.empty((max(total, 1),), dtype=torch.int32, device=dev)
    member = torch.zeros((max(total, 1),), dtype=torch.int32 if nslot * 4 <= 32 else torch.int64, device=dev)
    _expand[(tiles, triton.cdiv(words, 256))](bmap, words, offsets, pref, ulist, CW=256)
    _scan[grid](picked, width, picked.stride(0), keys, bmap, words, offsets, pref, member,
                REP=rep, BM=bm, QT=qt, BS=bs, MEMBER=True)
    return ulist, member, offsets


LOG2E = 1.4426950408889634


@triton.jit
def _rows(j, L, REP: tl.constexpr, BM: tl.constexpr, QT: tl.constexpr):
    """Tile j's rows: query t, head h, valid, 4 * slot. QT > 0: queries j * QT .. + QT - 1, rows past QT * REP idle;
    QT = 0: rows j * BM .. + BM - 1 of the (query, head) order."""
    r = tl.arange(0, BM)
    if QT > 0:
        t = j * QT + r // REP
        h = r % REP
        rv = (r < QT * REP) & (t < L)
        slot = (r // REP) * 4
    else:
        t = (j * BM + r) // REP
        h = (j * BM + r) % REP
        rv = t < L
        slot = (t - (j * BM) // REP) * 4
    return t, h, rv, slot


@triton.jit
def _step(UL, MK, K, V, KD, VD, g, skg, skn, u0, end, KEYS, dims, BN: tl.constexpr, TMA: tl.constexpr):
    """One step of the tile's union: BN keys (BN / 4 blocks), their member words and K / V rows."""
    cols = tl.arange(0, BN)
    sub = cols % 4
    u = u0 + cols
    um = u < end
    blk = tl.load(UL + u // 4, mask=um, other=0)
    bits = tl.load(MK + u // 4, mask=um, other=0)
    key = blk * 4 + sub
    kv = um & (key < KEYS)
    if TMA:  # Blackwell TMA gather (rows of the [groups * keys, D] view); masked keys read row 0 (scores masked)
        rows = g * KEYS + tl.where(kv, key, 0)
        k = KD.gather(rows, 0)
        v = VD.gather(rows, 0)
    else:
        k = tl.load(K + g * skg + key[:, None] * skn + dims[None, :], mask=kv[:, None], other=0.0)
        v = tl.load(V + g * skg + key[:, None] * skn + dims[None, :], mask=kv[:, None], other=0.0)
    return sub, bits, key, kv, k, v


@triton.jit
def _fwd(Q, K, V, KD, VD, UL, MK, OFF, O, LSE, qk2, L, KEYS,
         sqh, sql, skg, skn, soh, sol,
         REP: tl.constexpr, BM: tl.constexpr, QT: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
         TMA: tl.constexpr, WS: tl.constexpr):
    """Online softmax in base 2 (``qk2`` = scale * log2 e); LSE kept in base 2."""
    j = tl.program_id(0)
    g = tl.program_id(1)
    t, h, rv, slot = _rows(j, L, REP, BM, QT)
    heads = g * REP + h
    dims = tl.arange(0, D)
    q = tl.load(Q + heads[:, None] * sqh + t[:, None] * sql + dims[None, :], mask=rv[:, None], other=0.0)
    start = tl.load(OFF + j) * 4
    end = tl.load(OFF + j + 1) * 4
    m = tl.full([BM], float("-inf"), tl.float32)
    den = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for u0 in tl.range(start, end, BN, warp_specialize=WS):
        sub, bits, key, kv, k, v = _step(UL, MK, K, V, KD, VD, g, skg, skn, u0, end, KEYS, dims, BN, TMA)
        s = tl.dot(q, tl.trans(k)) * qk2
        mem = ((bits[None, :] >> (slot[:, None] + sub[None, :]).to(bits.dtype)) & 1) != 0
        s = tl.where(mem, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(s - m_use[:, None])
        alpha = tl.exp2(m - m_use)
        den = den * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    den_safe = tl.where(den > 0, den, 1.0)
    out = acc / den_safe[:, None]
    tl.store(O + heads[:, None] * soh + t[:, None] * sol + dims[None, :], out.to(O.dtype.element_ty), mask=rv[:, None])
    lse = tl.where(den > 0, tl.where(m == float("-inf"), 0.0, m) + tl.log2(den_safe), float("inf"))
    tl.store(LSE + heads * L + t, lse, mask=rv)


@triton.jit
def _bwd(Q, K, V, KD, VD, UL, MK, OFF, DO, LSE, DELTA, DQ, DK, DV, scale, qk2, L, KEYS,
         sqh, sql, skg, skn, sdh, sdl, sdkg, sdkn,
         REP: tl.constexpr, BM: tl.constexpr, QT: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
         PAT: tl.constexpr, TMA: tl.constexpr, WS: tl.constexpr, P0):
    """dq (one store per row) and dk, dv (one fp32 atomic per (tile, union key, dim)); ``PAT`` has bit 4 * slot set
    for every slot (a key some query of the tile picked). Keys below ``P0`` (MFU-1 stage P: the no-grad prefix of a
    cached-prefix window) get no dk / dv: a step whose keys all lie there skips the dk / dv products and atomics."""
    j = tl.program_id(0)
    g = tl.program_id(1)
    t, h, rv, slot = _rows(j, L, REP, BM, QT)
    heads = g * REP + h
    dims = tl.arange(0, D)
    q = tl.load(Q + heads[:, None] * sqh + t[:, None] * sql + dims[None, :], mask=rv[:, None], other=0.0)
    do = tl.load(DO + heads[:, None] * sdh + t[:, None] * sdl + dims[None, :], mask=rv[:, None], other=0.0)
    lse = tl.load(LSE + heads * L + t, mask=rv, other=float("inf"))
    delta = tl.load(DELTA + heads * L + t, mask=rv, other=0.0)
    start = tl.load(OFF + j) * 4
    end = tl.load(OFF + j + 1) * 4
    dq = tl.zeros([BM, D], tl.float32)
    for u0 in tl.range(start, end, BN, warp_specialize=WS):
        sub, bits, key, kv, k, v = _step(UL, MK, K, V, KD, VD, g, skg, skn, u0, end, KEYS, dims, BN, TMA)
        s = tl.dot(q, tl.trans(k)) * qk2
        mem = ((bits[None, :] >> (slot[:, None] + sub[None, :]).to(bits.dtype)) & 1) != 0
        p = tl.where(mem, tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * scale
        dq += tl.dot(ds.to(k.dtype), k)
        if tl.max(tl.where(kv, key, -1)) >= P0:
            hit = kv & (((bits >> sub.to(bits.dtype)) & PAT) != 0) & (key >= P0)
            offs = g * sdkg + key[:, None] * sdkn + dims[None, :]
            tl.atomic_add(DV + offs, tl.dot(tl.trans(p.to(do.dtype)), do), mask=hit[:, None], sem="relaxed")
            tl.atomic_add(DK + offs, tl.dot(tl.trans(ds.to(q.dtype)), q), mask=hit[:, None], sem="relaxed")
    tl.store(DQ + heads[:, None] * sqh + t[:, None] * sql + dims[None, :], dq.to(DQ.dtype.element_ty), mask=rv[:, None])


def descriptors(k, v, cfg):
    """Host TMA descriptors over the [groups * keys, D] views (``cfg["tma"]``), else None."""
    if not cfg.get("tma"):
        return None, None
    from triton.tools.tensor_descriptor import TensorDescriptor

    dim = k.shape[-1]
    return (TensorDescriptor.from_tensor(k[0].reshape(-1, dim), [1, dim]),
            TensorDescriptor.from_tensor(v[0].reshape(-1, dim), [1, dim]))


class UnionSelectedAttention(torch.autograd.Function):
    """Same call as ``TritonSelectedAttention.apply(q, k, v, picked, scale, chunk)``: q [1, heads, L, D], k / v
    [1, groups, keys, D], picked [L, S] (int64, -1 padding). Settings (class attributes, read at each forward):
    BM rows per tile, ALIGNED whole queries per tile, launch configs FWD / BWD (B200 sweep, mfu-e)."""
    BM = 64
    ALIGNED = True
    FWD = {"BN": 64, "num_warps": 8, "num_stages": 2, "tma": True, "ws": True}
    BWD = {"BN": 32, "num_warps": 8, "num_stages": 2}  # B300 sweep mfu1-ubwd-b3b: -2.6% at win32k, dq bitwise (BN kept)
    NOGRAD_PREFIX = False  # MFU-1 stage P: keys before the queries' own (keys - L) get no dk / dv (a cached prefix)
    TILE_CACHE = None  # MFU-1 stage S: a dict whose "tiles" keeps this call's union tiles for the checkpoint recompute

    @staticmethod
    def forward(ctx, q, k, v, picked, scale, chunk=None):
        cls = UnionSelectedAttention
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        heads, length, dim = q.shape[1], q.shape[2], q.shape[3]
        groups, keys = k.shape[1], k.shape[2]
        rep = heads // groups
        bm, aligned = cls.BM, cls.ALIGNED and cls.BM >= rep
        while bm > 16 and tiling(length, rep, bm, aligned)[2] * 4 > 64:
            bm //= 2
        tiles, qt, nslot = tiling(length, rep, bm, aligned)
        kept = cls.TILE_CACHE
        tiles_key = (length, keys, rep, bm, aligned)
        if kept is not None and kept.get("tiles") is not None and kept["tiles"][0] == tiles_key:
            ulist, member, offsets = kept["tiles"][1]
        else:
            ulist, member, offsets = union_tiles(picked.contiguous(), keys, rep, bm, aligned)
            if kept is not None:
                kept["tiles"] = (tiles_key, (ulist, member, offsets))
        out = torch.empty_like(q)
        lse = torch.empty((heads, length), dtype=torch.float32, device=q.device)
        f = cls.FWD
        kd, vd = descriptors(k, v, f)
        _fwd[(tiles, groups)](q[0], k[0], v[0], kd, vd, ulist, member, offsets, out[0], lse, float(scale) * LOG2E,
                              length, keys, q.stride(1), q.stride(2), k.stride(1), k.stride(2),
                              out.stride(1), out.stride(2), REP=rep, BM=bm, QT=qt, BN=f["BN"], D=dim,
                              TMA=f.get("tma", False), WS=f.get("ws", False),
                              num_warps=f["num_warps"], num_stages=f["num_stages"])
        ctx.save_for_backward(q, k, v, ulist, member, offsets, out, lse)
        ctx.scale, ctx.tiling, ctx.bwd = float(scale), (tiles, bm, qt, nslot), dict(cls.BWD)
        ctx.p0 = max(0, keys - length) if cls.NOGRAD_PREFIX else 0  # the trainer's prefix cache is no-grad
        return out

    @staticmethod
    def backward(ctx, grad):
        q, k, v, ulist, member, offsets, out, lse = ctx.saved_tensors
        grad = grad.contiguous()
        heads, length, dim = q.shape[1], q.shape[2], q.shape[3]
        groups, keys = k.shape[1], k.shape[2]
        tiles, bm, qt, nslot = ctx.tiling
        delta = (grad[0].float() * out[0].float()).sum(-1).contiguous()  # [heads, L]
        dq = torch.empty_like(q)
        dk = torch.zeros((1, groups, keys, dim), dtype=torch.float32, device=q.device)
        dv = torch.zeros_like(dk)
        b = ctx.bwd
        kd, vd = descriptors(k, v, b)
        _bwd[(tiles, groups)](q[0], k[0], v[0], kd, vd, ulist, member, offsets, grad[0], lse, delta, dq[0], dk[0],
                              dv[0], ctx.scale, ctx.scale * LOG2E, length, keys,
                              q.stride(1), q.stride(2), k.stride(1), k.stride(2), grad.stride(1), grad.stride(2),
                              dk.stride(1), dk.stride(2), REP=heads // groups, BM=bm, QT=qt, BN=b["BN"], D=dim,
                              PAT=sum(1 << (4 * i) for i in range(nslot)), TMA=b.get("tma", False),
                              WS=b.get("ws", False), num_warps=b["num_warps"], num_stages=b["num_stages"],
                              P0=ctx.p0)
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None
