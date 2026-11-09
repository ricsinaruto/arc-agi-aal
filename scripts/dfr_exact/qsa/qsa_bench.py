"""QSA prefill attention bench (throughput arm, 19 Oct; query-union tiling, step 1-2). Replays captured QSA calls
(qsa_dump.py: real picks, row->request map and block tables from the steady bench) on a synthetic int4 cache with the
same page layout, and reports:

- pick overlap: for R consecutive rows of one request, |union of picked 4-token groups| / picks per row (prefill rows
  = requests with > 8 rows in the call; decode rows = MTP verify rows, <= 8);
- the served kernel's time per call (cold L2 and warm) and a sweep of its tile profile (BLOCK_N, warps, stages, split
  launch of prefill and decode rows), each checked against the served output (same arithmetic, so only the
  summation order may differ).

    python qsa_bench.py --dumps DIR --out qsa-bench.json
"""
import argparse
import glob
import json
import os
import sys
import time

import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from arc_infra import kv4_runtime as kv4  # noqa: E402

HEAD_DIM, CONTENT = 256, 264
GROUP = 4


def compact_pages(block_table, t2r, page, kv_stride, device):
    """Renumber the referenced physical pages 0..n-1 (relative order kept) and build an int4 cache of n pages with
    the dumped layout (heads-outer or tokens-outer), filled with random codes and valid scale/zero-point words."""
    reqs = torch.unique(t2r[t2r >= 0]).long()
    used = block_table[reqs]
    ids = torch.unique(used[used >= 0])
    remap = torch.full((int(block_table.max()) + 2,), -1, dtype=torch.int32)
    remap[ids.long()] = torch.arange(len(ids), dtype=torch.int32)
    table = torch.where(block_table >= 0, remap[block_table.clamp_min(0).long()], block_table)
    n = len(ids)
    heads_outer = kv_stride[1] > kv_stride[2]
    shape = (n, 2, page, CONTENT) if heads_outer else (n, page, 2, CONTENT)
    raw = torch.randint(0, 256, shape, dtype=torch.uint8, device=device)
    cache = raw if heads_outer else raw.transpose(1, 2)  # logical (blocks, heads, tokens, content)
    k, v, ks, vs = kv4.cache_views(cache, "int4")
    for s in (ks, vs):
        sc = torch.empty(s.shape, device=device).uniform_(0.01, 0.05)
        zp = torch.randint(0, 16, s.shape, dtype=torch.int32, device=device)
        s.copy_((sc.view(torch.int32) & -16) | zp)
    return cache, table.to(device), n


def overlap(indices, t2r):
    """Union of picked groups over R consecutive rows of one request, in units of one row's picks."""
    idx = indices.long()
    rows = idx.shape[0]
    grp = torch.where(idx >= 0, idx // GROUP, torch.full_like(idx, -1))
    out = {}
    req = t2r.long()
    seg_rows = {}
    for r in req.unique().tolist():
        seg_rows[r] = (req == r).nonzero().flatten()
    for kind, sel in (("prefill", lambda n: n > 8), ("decode", lambda n: n <= 8)):
        for R in (2, 4, 8, 16):
            ratios, saves = [], []
            for r, rr in seg_rows.items():
                if r < 0 or not sel(len(rr)):
                    continue
                Rk = min(R, len(rr)) if kind == "decode" else R
                for s in range(0, len(rr) - Rk + 1, Rk):
                    g = grp[rr[s:s + Rk]]
                    per = [int((torch.unique(x[x >= 0])).numel()) for x in g]
                    u = torch.unique(g[g >= 0]).numel()
                    if max(per):
                        ratios.append(u / max(per))
                        saves.append(sum(per) / u)
            if ratios:
                t = torch.tensor(ratios)
                out[f"{kind}_R{R}"] = {"groups": len(ratios), "union_over_picks_mean": round(t.mean().item(), 3),
                                       "p10": round(t.quantile(0.1).item(), 3), "p90": round(t.quantile(0.9).item(), 3),
                                       "load_saving_mean": round(sum(saves) / len(saves), 3)}
            if kind == "decode":
                break  # decode: one union over the request's verify rows
    valid = (idx >= 0).sum(1).float()
    out["rows"] = rows
    out["picks_per_row_mean"] = round(valid.mean().item(), 1)
    out["rows_full_topk"] = round((valid == idx.shape[1]).float().mean().item(), 3)
    out["prefill_rows"] = int(sum(len(v) for k, v in seg_rows.items() if k >= 0 and len(v) > 8))
    return out


def launch(q, views, indices, table, t2r, out, profile, stages):
    k, v, ks, vs = views
    p = profile
    if p["splits"] == 1:
        po, pl = out, out
    else:
        po = torch.empty((p["splits"], *q.shape), dtype=torch.float32, device=q.device)
        pl = torch.empty((p["splits"], q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    gs = q.shape[1] // k.shape[2]
    kv4._qsa_kv4_splitk_kernel[(q.shape[0], k.shape[2], p["splits"])](
        q, k, v, ks, vs, indices, table, t2r, po, pl, out,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
        ks.stride(0), ks.stride(1), ks.stride(2), vs.stride(0), vs.stride(1), vs.stride(2),
        indices.stride(0), table.stride(0), out.stride(0), out.stride(1), q.shape[0], k.shape[0], table.shape[0],
        TOPK=indices.shape[1], PAGE_SIZE=k.shape[1], PAGE_TABLE_WIDTH=table.shape[1], GROUP_SIZE=gs,
        HEAD_DIM=HEAD_DIM, NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=p["splits"], NUM_TILES=p["tiles"],
        BLOCK_M=p["block_m"], BLOCK_N=p["block_n"], KEY_BITS=4, INTERPRET=False, num_warps=p["warps"],
        num_stages=stages)
    if p["splits"] > 1:
        kv4._kv8._qsa_kv8_merge_kernel[(q.shape[0], q.shape[1])](
            po, pl, out, out.stride(0), out.stride(1), q.shape[0], HEAD_DIM=HEAD_DIM, NUM_QUERY_HEADS=q.shape[1],
            NUM_SPLITS=p["splits"], BLOCK_SPLITS=triton.next_power_of_2(p["splits"]), num_warps=2, num_stages=1)


def timed(fn, flush, reps=10):
    fn()
    torch.cuda.synchronize()
    res = {}
    for cold in (True, False):
        ts = []
        for _ in range(reps):
            if cold:
                flush.zero_()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        ts.sort()
        res["cold_ms" if cold else "warm_ms"] = round(ts[len(ts) // 2], 4)
    return res


def export_sources(out_dir, words=("humming",)):
    """Tar the runtime's Python/CUDA sources whose path contains one of ``words`` (for reading the MoE kernel config
    offline; nothing is executed)."""
    import tarfile
    seen = set()
    with tarfile.open(os.path.join(out_dir, "runtime-src.tgz"), "w:gz") as tar:
        for base in sys.path:
            if not os.path.isdir(base):
                continue
            for top in os.listdir(base):
                if top not in ("vllm", "humming") and not any(w in top.lower() for w in words):
                    continue
                for root, _, names in os.walk(os.path.join(base, top)):
                    for n in names:
                        f = os.path.join(root, n)
                        if (any(w in f.lower() for w in words) and n.endswith((".py", ".cu", ".cuh", ".h", ".json"))
                                and f not in seen and os.path.getsize(f) < 4 << 20):
                            seen.add(f)
                            tar.add(f, arcname=os.path.relpath(f, base))
    return len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-dumps", type=int, default=12)
    a = ap.parse_args()
    dev = torch.device("cuda")
    flush = torch.empty(256 * 2**20, dtype=torch.uint8, device=dev)
    files = sorted(glob.glob(os.path.join(a.dumps, "*.pt")))[:a.max_dumps]
    report = {"device": torch.cuda.get_device_name(), "calls": []}
    try:
        report["exported_sources"] = export_sources(os.path.dirname(os.path.abspath(a.out)))
    except Exception as exc:
        report["exported_sources"] = f"error: {exc}"
    sweep = [(bn, w, st) for bn in (16, 32, 64, 128) for w in (2, 4, 8) for st in (1, 2, 3)]
    for f in files:
        d = torch.load(f)
        t0 = time.time()
        idx, t2r, bt = d["indices"].to(torch.int32), d["token_to_req"].to(torch.int32), d["block_table"].to(torch.int32)
        page = d["kv_shape"][2]
        rec = {"file": os.path.basename(f), "layer": d["layer"], "t": round(d["t"], 1), "q_shape": d["q_shape"],
               "topk": idx.shape[1], "page": page, "overlap": overlap(idx, t2r)}
        cache, table, npages = compact_pages(bt, t2r, page, d["kv_stride"], dev)
        rec["pages"] = npages
        views = kv4.cache_views(cache, "int4")
        rows, heads = d["q_shape"][0], d["q_shape"][1]
        q = (torch.randn(rows, heads, HEAD_DIM, device=dev) * 0.5).to(torch.bfloat16)
        idx_d, t2r_d = idx.to(dev), t2r.to(dev)
        served = kv4._kv8.launch_profile(rows, 2, heads // 2, idx.shape[1])
        rec["served_profile"] = served
        ref = torch.empty(q.shape, dtype=torch.float32, device=dev)
        rec["served"] = timed(lambda: launch(q, views, idx_d, table, t2r_d, ref, served, 2), flush)
        best = None
        rec["sweep"] = []
        for bn, w, st in sweep:
            if bn > 64 and w < 4:
                continue
            p = {"block_m": served["block_m"], "block_n": bn, "splits": 1, "warps": w,
                 "tiles": triton.cdiv(idx.shape[1], bn)}
            out = torch.empty_like(ref)
            try:
                tm = timed(lambda: launch(q, views, idx_d, table, t2r_d, out, p, st), flush, reps=5)
            except Exception as exc:  # resource limits on some tiles
                rec["sweep"].append({"bn": bn, "w": w, "st": st, "error": str(exc)[:120]})
                continue
            err = ((out - ref).norm() / ref.norm()).item()
            row = {"bn": bn, "w": w, "st": st, **tm, "rel_err": float(f"{err:.2e}")}
            rec["sweep"].append(row)
            if err < 1e-4 and (best is None or tm["cold_ms"] < best["cold_ms"]):
                best = row
        rec["best"] = best
        rec["seconds"] = round(time.time() - t0, 1)
        report["calls"].append(rec)
        print(json.dumps({k: rec[k] for k in ("file", "layer", "q_shape", "overlap", "served", "best")}), flush=True)
        del cache, views
        torch.cuda.empty_cache()
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
