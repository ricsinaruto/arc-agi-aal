"""KV-offload microbenchmark (throughput arm, 18 Oct): one QSA layer's int4 sparse attention at decode with the
main KV in pinned host memory (read over PCIe through UVA) and a per-request GPU cache of 4-token groups, against
the served kernel with everything on the GPU. Picks and the cache state come from the recorded QSA trace (LRU,
capacity a share of the context). Reports per-layer times (attention, cache fill), the bitwise check against the
baseline output, and the implied decode-step overhead for 13 layers.

    python kvoff_bench.py --trace DIR --out kvoff.json [--requests 32,48,64,96] [--share 0.25]
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from arc_infra import kv4_runtime as kv4  # noqa: E402
from arc_infra import kv8_runtime as kv8  # noqa: E402

HEAD_DIM, KV_HEADS, Q_HEADS, PAGE = 256, 2, 24, 3136
HEAD_BYTES = 264  # [K 128 | K scale 4 | V 128 | V scale 4]
TOKEN_BYTES = KV_HEADS * HEAD_BYTES  # 528
GROUP = 4
GROUP_BYTES = GROUP * TOKEN_BYTES  # 2112
TOPK = 2051
INTERP = os.environ.get("TRITON_INTERPRET") == "1"


@triton.jit
def _kvoff_attn_kernel(
    q_ptr, base_ptr, indices_ptr, block_table_ptr, token_to_req_ptr, remap_ptr,
    partial_output_ptr, partial_lse_ptr, output_ptr,
    host_delta, slots_per_req,
    stride_q_row, stride_q_head, stride_indices_row, stride_table_req, stride_remap_req,
    stride_output_row, stride_output_head,
    num_rows, num_requests,
    TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr, GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr, NUM_QUERY_HEADS: tl.constexpr, NUM_SPLITS: tl.constexpr, NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, TOKEN_BYTES: tl.constexpr, HEAD_BYTES: tl.constexpr,
    GROUP_TOKENS: tl.constexpr, INTERPRET: tl.constexpr = False,
) -> None:
    # _qsa_kv4_splitk_kernel (KEY_BITS 4) with the page address replaced: a pick whose 4-token group has a cache
    # slot (remap >= 0) reads the GPU cache [slot, 4 tokens, heads, 264 B]; any other pick reads its host page
    # through UVA (host_delta = host pool address - cache address). Same loads, same order, same arithmetic.
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    half_offsets = tl.arange(0, HEAD_DIM // 2)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    head_mask = head_offsets[:, None] < GROUP_SIZE
    q_rows = q_ptr + row * stride_q_row + (first_head + head_offsets[:, None]) * stride_q_head
    q_even = tl.load(q_rows + 2 * half_offsets[None, :], mask=head_mask, other=0.0)
    q_odd = tl.load(q_rows + 2 * half_offsets[None, :] + 1, mask=head_mask, other=0.0)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_even = tl.zeros((BLOCK_M, HEAD_DIM // 2), dtype=tl.float32)
    acc_odd = tl.zeros((BLOCK_M, HEAD_DIM // 2), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634
    slot_base = safe_request.to(tl.int64) * slots_per_req

    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(indices_ptr + row * stride_indices_row + columns, mask=columns < TOPK, other=-1)
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (request >= 0) & (request < num_requests) & (logical_token >= 0) & (logical_page < PAGE_TABLE_WIDTH)
        physical_page = tl.load(
            block_table_ptr + safe_request * stride_table_req + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid, other=-1,
        )
        valid &= physical_page >= 0
        slot = tl.load(remap_ptr + safe_request * stride_remap_req + safe_token // GROUP_TOKENS, mask=valid, other=-1)
        host_off = host_delta + (tl.maximum(physical_page, 0).to(tl.int64) * PAGE_SIZE + page_offset) * TOKEN_BYTES
        cache_off = ((slot_base + tl.maximum(slot, 0)) * GROUP_TOKENS + safe_token % GROUP_TOKENS) * TOKEN_BYTES
        token_off = tl.where(slot >= 0, cache_off, host_off) + kv_head * HEAD_BYTES
        key_scale_raw = tl.load((base_ptr + token_off + 128).to(tl.pointer_type(tl.int32)), mask=valid, other=0)
        packed = tl.load(base_ptr + token_off[None, :] + half_offsets[:, None], mask=valid[None, :], other=0).to(tl.int32)
        key_zero = key_scale_raw & 0xF
        key_scale = (key_scale_raw & -16).to(tl.float32, bitcast=True)
        k_even = ((packed & 0xF) - key_zero[None, :]).to(tl.float32)
        k_odd = (((packed >> 4) & 0xF) - key_zero[None, :]).to(tl.float32)
        if INTERPRET:
            scores = tl.dot(q_even.to(tl.float32), k_even) + tl.dot(q_odd.to(tl.float32), k_odd)
        else:
            scores = tl.dot(q_even, k_even.to(tl.bfloat16)) + tl.dot(q_odd, k_odd.to(tl.bfloat16))
        scores *= key_scale[None, :] * softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0)
        value_packed = tl.load(base_ptr + token_off[:, None] + 132 + half_offsets[None, :], mask=valid[:, None],
                               other=0).to(tl.int32)
        value_scale_raw = tl.load((base_ptr + token_off + 260).to(tl.pointer_type(tl.int32)), mask=valid, other=0)
        value_zero = value_scale_raw & 0xF
        value_scale = (value_scale_raw & -16).to(tl.float32, bitcast=True)
        v_even = ((value_packed & 0xF) - value_zero[:, None]).to(tl.float32)
        v_odd = (((value_packed >> 4) & 0xF) - value_zero[:, None]).to(tl.float32)
        weights = (probabilities * value_scale[None, :]).to(tl.bfloat16)
        if INTERPRET:
            acc_even = tl.dot(weights.to(tl.float32), v_even, acc=acc_even * alpha[:, None])
            acc_odd = tl.dot(weights.to(tl.float32), v_odd, acc=acc_odd * alpha[:, None])
        else:
            acc_even = tl.dot(weights, v_even.to(tl.bfloat16), acc=acc_even * alpha[:, None])
            acc_odd = tl.dot(weights, v_odd.to(tl.bfloat16), acc=acc_odd * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    denominator = tl.maximum(normalizer[:, None], 1.0e-20)
    out_even = tl.where(has_values[:, None], acc_even / denominator, 0.0)
    out_odd = tl.where(has_values[:, None], acc_odd / denominator, 0.0)
    even_columns = 2 * half_offsets[None, :]
    if NUM_SPLITS == 1:
        out_rows = output_ptr + row * stride_output_row + (first_head + head_offsets[:, None]) * stride_output_head
        tl.store(out_rows + even_columns, out_even, mask=head_mask)
        tl.store(out_rows + even_columns + 1, out_odd, mask=head_mask)
    else:
        partial_lse = tl.where(has_values, max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)), -float("inf"))
        partial_rows = partial_output_ptr + ((split_id * num_rows + row) * NUM_QUERY_HEADS + first_head
                                             + head_offsets[:, None]) * HEAD_DIM
        tl.store(partial_rows + even_columns, out_even, mask=head_mask)
        tl.store(partial_rows + even_columns + 1, out_odd, mask=head_mask)
        tl.store(partial_lse_ptr + (split_id * num_rows + row) * NUM_QUERY_HEADS + first_head + head_offsets,
                 partial_lse, mask=head_offsets < GROUP_SIZE)


@triton.jit
def _kvoff_fill_kernel(base_ptr, host_delta, miss_req_ptr, miss_group_ptr, miss_slot_ptr, block_table_ptr,
                       stride_table_req, slots_per_req, PAGE_SIZE: tl.constexpr, WORDS: tl.constexpr,
                       BLOCK: tl.constexpr, GROUP_TOKENS: tl.constexpr, TOKEN_BYTES: tl.constexpr):
    # One program per missed group: copy its 4 tokens (contiguous in the host page) into the request's slot.
    i = tl.program_id(0)
    req = tl.load(miss_req_ptr + i).to(tl.int64)
    group = tl.load(miss_group_ptr + i).to(tl.int64)
    slot = tl.load(miss_slot_ptr + i).to(tl.int64)
    token = group * GROUP_TOKENS
    page = tl.load(block_table_ptr + req * stride_table_req + token // PAGE_SIZE).to(tl.int64)
    src = (base_ptr + host_delta + (page * PAGE_SIZE + token % PAGE_SIZE) * TOKEN_BYTES).to(tl.pointer_type(tl.int32))
    dst = (base_ptr + (req * slots_per_req + slot) * GROUP_TOKENS * TOKEN_BYTES).to(tl.pointer_type(tl.int32))
    w = tl.arange(0, BLOCK)
    tl.store(dst + w, tl.load(src + w, mask=w < WORDS), mask=w < WORDS)


def load_trace(directory, layer):
    files = sorted(glob.glob(f"{directory}/**/qsa-trace-*.npz", recursive=True))
    meta = np.concatenate([np.load(f)["meta"] for f in files])
    groups = np.concatenate([np.load(f)["groups"] for f in files])
    series = defaultdict(list)
    for m, g in zip(meta, groups):
        if int(m[0]) != layer:
            continue
        key = int(m[2])
        s = series[key]
        if s and not (0 <= m[3] - s[-1][0] <= 8):
            series[(key, len(series))] = s[:]
            s.clear()
        s.append((int(m[3]), g[g >= 0].astype(np.int64)))
    return [s for s in series.values() if len(s) >= 300]


def lru_states(s, share, steps):
    """Cache contents (group -> slot) before each step in ``steps``, LRU at share x context groups."""
    cache, free, out, want = OrderedDict(), [], {}, set(steps)
    next_slot = 0
    for i, (seq, groups) in enumerate(s):
        if i in want:
            out[i] = dict(cache)
        cap = max(1, int(share * seq / GROUP))
        for x in groups.tolist():
            if x in cache:
                cache.move_to_end(x)
            else:
                if len(cache) >= cap:
                    _, slot = cache.popitem(last=False)
                    free.append(slot)
                cache[x] = free.pop() if free else next_slot
                if cache[x] == next_slot:
                    next_slot += 1
        if i >= max(want):
            break
    return out, next_slot


def timed(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return float(np.median(times))


def case(series, requests, share, device, rng):
    # Request i replays series i % n at a step offset; row 2i = step t, row 2i+1 = step t+1 (the verify row).
    n = len(series)
    picks, seqs, states, slots_needed = [], [], [], 0
    by_series = defaultdict(list)
    for i in range(requests):
        s = series[i % n]
        t = 100 + (i // n * 97 + 31 * i) % (len(s) - 102)
        by_series[i % n].append((i, t))
    meta = {}
    for k, items in by_series.items():
        st, used = lru_states(series[k], share, [t for _, t in items])
        slots_needed = max(slots_needed, used)
        for i, t in items:
            meta[i] = (series[k][t], series[k][t + 1], st[t])
    rows_tokens, row_req = [], []
    for i in range(requests):
        a, b, _ = meta[i]
        seq = b[0]
        seqs.append(seq)
        for seq_r, g in (a, b):
            tok = (g[:, None] * GROUP + np.arange(GROUP)[None, :]).reshape(-1)
            tok = tok[tok < seq_r][:TOPK]
            row = np.full(TOPK, -1, dtype=np.int32)
            row[: len(tok)] = tok
            rows_tokens.append(row)
            row_req.append(i)
    pages_per = [-(-s // PAGE) for s in seqs]
    width = max(pages_per)
    table = np.full((requests, width), -1, dtype=np.int32)
    p = 0
    for i, k in enumerate(pages_per):
        table[i, :k] = np.arange(p, p + k)
        p += k
    num_pages = p
    max_groups = max(seqs) // GROUP + 2
    remap = np.full((requests, max_groups), -1, dtype=np.int32)
    hits = total = 0
    for i in range(requests):
        for g, slot in meta[i][2].items():
            remap[i, g] = slot
    for r, row in enumerate(rows_tokens):
        g = np.unique(row[row >= 0] // GROUP)
        total += len(g)
        hits += int((remap[row_req[r], g] >= 0).sum())
    # misses of this step (both rows), to fill after attention: group -> new slot past the used ones
    # Fill-before-attend: every missed group (both rows) gets a fresh slot past the used ones, so after the fill
    # kernel the attention reads only the GPU cache (remap_full).
    miss_req, miss_group, miss_slot = [], [], []
    remap_full = remap.copy()
    most = 0
    for i in range(requests):
        gs = set()
        for r in (2 * i, 2 * i + 1):
            row = rows_tokens[r]
            gs |= set((row[row >= 0] // GROUP).tolist())
        nxt = slots_needed
        for g in sorted(gs):
            if remap[i, g] < 0:
                miss_req.append(i)
                miss_group.append(g)
                miss_slot.append(nxt)
                remap_full[i, g] = nxt
                nxt += 1
        most = max(most, nxt)
    return dict(rows=np.stack(rows_tokens), row_req=np.asarray(row_req, dtype=np.int32), table=table,
                num_pages=num_pages, remap=remap, remap_full=remap_full, slots=max(most, 1), hit=hits / max(total, 1),
                miss=(np.asarray(miss_req, np.int32), np.asarray(miss_group, np.int32), np.asarray(miss_slot, np.int32)),
                seqs=seqs)


def run(c, device, rng, iters):
    rows = c["rows"].shape[0]
    requests = c["table"].shape[0]
    # Device pool (baseline) and an identical pinned host pool; physical layout (pages, tokens, heads, 264).
    pool = torch.randint(0, 256, (c["num_pages"], PAGE, KV_HEADS, HEAD_BYTES), dtype=torch.uint8, device=device)
    scales = torch.empty((c["num_pages"], PAGE, KV_HEADS, 2), dtype=torch.float32, device=device).uniform_(0.01, 0.02)
    words = (scales.view(torch.int32) & -16) | torch.randint(0, 16, scales.shape, dtype=torch.int32, device=device)
    pv = pool.view(c["num_pages"], PAGE, KV_HEADS, HEAD_BYTES)
    pv[..., 128:132] = words[..., 0:1].contiguous().view(torch.uint8).view(c["num_pages"], PAGE, KV_HEADS, 4)
    pv[..., 260:264] = words[..., 1:2].contiguous().view(torch.uint8).view(c["num_pages"], PAGE, KV_HEADS, 4)
    host = torch.empty(pool.shape, dtype=torch.uint8, pin_memory=True)
    host.copy_(pool.cpu())
    logical = pool.permute(0, 2, 1, 3)  # (blocks, heads, tokens, content) as the served cache
    k_codes, v_codes, k_scale, v_scale = kv4.cache_views(logical, "int4")
    q = (torch.randn((rows, Q_HEADS, HEAD_DIM), device=device) * 0.5).to(torch.bfloat16)
    indices = torch.from_numpy(c["rows"]).to(device)
    table = torch.from_numpy(c["table"]).to(device)
    t2r = torch.from_numpy(c["row_req"]).to(device)
    remap = torch.from_numpy(c["remap"]).to(device)
    # GPU cache: (requests, slots, 4 tokens, 528 B); fill cached groups from the device pool
    cache = torch.zeros((requests, c["slots"], GROUP, TOKEN_BYTES), dtype=torch.uint8, device=device)
    flat_pool = pool.view(-1, TOKEN_BYTES)
    rq, gq = torch.nonzero(remap >= 0, as_tuple=True)
    sl = remap[rq, gq].long()
    tok = gq.long()[:, None] * GROUP + torch.arange(GROUP, device=device)[None, :]
    page = table.long()[rq[:, None], torch.clamp(tok // PAGE, max=table.shape[1] - 1)]
    src = page * PAGE + tok % PAGE
    ok = (tok < torch.tensor(c["seqs"], device=device)[rq][:, None]) & (page >= 0)
    cache[rq[:, None].expand_as(tok)[ok], sl[:, None].expand_as(tok)[ok], (tok % GROUP)[ok]] = flat_pool[src[ok]]
    host_ptr = host.data_ptr()
    delta = host_ptr - cache.data_ptr()

    profile = kv8.launch_profile(rows, KV_HEADS, Q_HEADS // KV_HEADS, TOPK)
    out_ref = torch.empty((rows, Q_HEADS, HEAD_DIM), dtype=torch.float32, device=device)
    out_off = torch.empty_like(out_ref)
    splits = profile["splits"]
    part_o = torch.empty((splits, rows, Q_HEADS, HEAD_DIM), dtype=torch.float32, device=device)
    part_l = torch.empty((splits, rows, Q_HEADS), dtype=torch.float32, device=device)

    def baseline():
        kv4.sparse_attention(q, k_codes, v_codes, k_scale, v_scale, indices, table, t2r, out_ref, 4, profile)

    def offload(remap_t):
        po, pl = (out_off, out_off) if splits == 1 else (part_o, part_l)
        _kvoff_attn_kernel[(rows, KV_HEADS, splits)](
            q, cache, indices, table, t2r, remap_t, po, pl, out_off, delta, c["slots"],
            q.stride(0), q.stride(1), indices.stride(0), table.stride(0), remap_t.stride(0),
            out_off.stride(0), out_off.stride(1), rows, requests,
            TOPK=TOPK, PAGE_SIZE=PAGE, PAGE_TABLE_WIDTH=table.shape[1], GROUP_SIZE=Q_HEADS // KV_HEADS,
            HEAD_DIM=HEAD_DIM, NUM_QUERY_HEADS=Q_HEADS, NUM_SPLITS=splits, NUM_TILES=profile["tiles"],
            BLOCK_M=profile["block_m"], BLOCK_N=profile["block_n"], TOKEN_BYTES=TOKEN_BYTES, HEAD_BYTES=HEAD_BYTES,
            GROUP_TOKENS=GROUP, INTERPRET=INTERP, num_warps=profile["warps"], num_stages=2)
        if splits > 1:
            kv8._qsa_kv8_merge_kernel[(rows, Q_HEADS)](
                part_o, part_l, out_off, out_off.stride(0), out_off.stride(1), rows,
                HEAD_DIM=HEAD_DIM, NUM_QUERY_HEADS=Q_HEADS, NUM_SPLITS=splits,
                BLOCK_SPLITS=triton.next_power_of_2(splits), num_warps=2, num_stages=1)

    mr, mg, ms = (torch.from_numpy(x).to(device) for x in c["miss"])

    def fill():
        if len(mr):
            _kvoff_fill_kernel[(len(mr),)](cache, delta, mr, mg, ms, table, table.stride(0), c["slots"],
                                           PAGE_SIZE=PAGE, WORDS=GROUP_BYTES // 4, BLOCK=1024, GROUP_TOKENS=GROUP,
                                           TOKEN_BYTES=TOKEN_BYTES)

    baseline()
    offload(remap)
    torch.cuda.synchronize()
    exact = bool(torch.equal(out_ref, out_off))
    all_host = torch.full_like(remap, -1)
    offload(all_host)
    torch.cuda.synchronize()
    exact_host = bool(torch.equal(out_ref, out_off))
    remap_full = torch.from_numpy(c["remap_full"]).to(device)
    fill()
    offload(remap_full)
    torch.cuda.synchronize()
    exact_full = bool(torch.equal(out_ref, out_off))
    res = {
        "requests": requests, "rows": rows, "hit": round(c["hit"], 4), "misses": int(len(mr)),
        "context_mean": int(np.mean(c["seqs"])), "cache_slots_per_request": c["slots"],
        "baseline_ms": timed(baseline, iters), "offload_ms": timed(lambda: offload(remap), iters),
        "all_host_ms": timed(lambda: offload(all_host), iters), "fill_ms": timed(fill, iters),
        "cached_attn_ms": timed(lambda: offload(remap_full), iters),
        "exact": exact, "exact_all_host": exact_host, "exact_fill_first": exact_full,
    }
    res["overhead_ms_13_layers"] = round(13 * (res["offload_ms"] + res["fill_ms"] - res["baseline_ms"]), 3)
    res["fill_first_overhead_ms_13_layers"] = round(13 * (res["cached_attn_ms"] + res["fill_ms"] - res["baseline_ms"]), 3)
    del pool, host, cache
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--requests", default="32,48,64,96")
    ap.add_argument("--share", default="0.25")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    device = torch.device("cuda")
    rng = np.random.default_rng(0)
    series = load_trace(a.trace, a.layer)
    report = {"gpu": torch.cuda.get_device_name(), "series": len(series), "layer": a.layer, "results": []}
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)
    for share in (float(x) for x in a.share.split(",")):
        for requests in (int(x) for x in a.requests.split(",")):
            t0 = time.time()
            c = case(series, requests, share, device, rng)
            res = run(c, device, rng, a.iters)
            res["share"] = share
            res["setup_s"] = round(time.time() - t0, 1)
            report["results"].append(res)
            print(json.dumps(res), flush=True)
            with open(a.out, "w") as f:
                json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
