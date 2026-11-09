

# ----------------------------------------------------------------------------- Humming MoE tile sweep (throughput item 3, 22 Oct)
# ARC_HTUNE=<dir>: once <dir>/go exists, the next prefill forward with >= 128 tokens records every Humming indexed-MoE
# layer's input rows (a fixed random subset of 128 tokens) and top-k ids; after the last layer a background thread
# (own CUDA stream; the server is idle) times the w13 and w2 GEMMs of all layers back to back (weights stay cold:
# ~GBs per pass vs 128 MB L2) at decode-sized M for the served heuristic config and coordinate-descent variants of it,
# and compares every variant's outputs with the served config's bit for bit. Results: <dir>/htune.jsonl (one line per
# config, written before (pending) and after its run, so a crash names the culprit), <dir>/done.
import json as _ht_json
import os as _ht_os
import threading as _ht_threading
import time as _ht_time

_HT = {"recs": [], "ids": set(), "started": False}


def _ht_log(root, row):
    with open(root + "/htune.jsonl", "a") as fh:
        fh.write(_ht_json.dumps(row) + "\n")


def _ht_key(cfg):
    return _ht_json.dumps(cfg, sort_keys=True)


def _ht_legal(exp, which, cfg):
    """Humming's own legality rules (testing.tuning) plus its shared-memory and thread limits."""
    from humming.config import ComputeConfig
    from humming.testing.tuning import _is_legal_geometry, create_tuning_config
    from humming.utils.device import fits_device_smem
    lc = exp.humming_configs[which]
    bs, ws = tuple(cfg["block_shape"]), tuple(cfg["warp_shape"])
    if not _is_legal_geometry(lc, bs, ws):
        return False
    threads = (bs[0] // ws[0]) * (bs[1] // ws[1]) * (bs[2] // ws[2]) * 32 + (128 if cfg.get("use_warp_spec") else 0)
    if threads > 1024 or threads * cfg.get("num_ctas_per_sm", 1) > 1536:
        return False
    try:
        cc = ComputeConfig(**_ht_json.loads(exp.compute_config_str))
        return fits_device_smem(lc, cc, create_tuning_config(dict(cfg, block_shape=bs, warp_shape=ws)))
    except Exception:
        return False


def _ht_neighbors(cfg, which):
    """One-knob variants of ``cfg`` (block_m fixed: the indexed GEMM's rows are aligned to it)."""
    out = []
    bm, bn, bk = cfg["block_shape"]
    wm, wn, wk = cfg["warp_shape"]

    def v(**kw):
        c = dict(cfg)
        c.update(kw)
        out.append(c)
    for s in (2, 3, 4, 5, 6):
        v(num_stages=s)
    for c in (1, 2, 3, 4):
        v(num_ctas_per_sm=c)
    v(use_stream_k=not cfg.get("use_stream_k", True))
    for n in (47, 94, 141, 188, 282, 376):
        v(num_sms=n)
    for nbn in (64, 128, 256, 512):
        for nwn in (16, 32, 64):
            if nbn % nwn == 0:
                v(block_shape=[bm, nbn, bk], warp_shape=[wm, nwn, wk])
    for nwk in (32, 64, 128):
        for kw in (1, 2, 4):
            v(block_shape=[bm, bn, nwk * kw], warp_shape=[wm, wn, nwk])
    for nwm in {bm, bm // 2, 16, 32}:
        if nwm and bm % nwm == 0 and nwm % 16 == 0:
            v(warp_shape=[nwm, wn, wk])
    v(num_write_splits=2 if cfg.get("num_write_splits", 1) == 1 else 1)
    for name in ("use_mbarrier", "use_cp_async", "use_tma", "use_tma_b", "use_tma_bs"):
        v(**{name: not bool(cfg.get(name, False))})
    return out


def _ht_sweep(root):
    import torch
    seconds = float(_ht_os.environ.get("ARC_HTUNE_SECONDS", "2400"))
    ms = [int(x) for x in (_ht_os.environ.get("ARC_HTUNE_MS") or "64,80,96,112,128").split(",")]
    tune_m = int(_ht_os.environ.get("ARC_HTUNE_TUNE_M", "96"))
    t_end = _ht_time.time() + seconds
    recs = _HT["recs"]
    if not recs:
        raise RuntimeError("no Humming indexed-MoE call with >= 128 rows was recorded")
    exp0 = recs[0]["exp"]
    stream = torch.cuda.Stream()
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import moe_align_block_size
    _ht_log(root, {"event": "start", "layers": len(recs), "ms": ms, "tune_m": tune_m,
                   "w13": str(exp0.humming_configs["w13"].to_str())[:600], "w2": str(exp0.humming_configs["w2"].to_str())[:600],
                   "compute": exp0.compute_config_str, "table13": exp0.w13_tuning_config, "table2": exp0.w2_tuning_config})

    def problem(m):
        """Per layer: inputs, routing and the served config's outputs at M tokens."""
        probs = []
        for r in recs:
            e = r["exp"]
            ids = r["ids"][:m].contiguous()
            vsm = -(-ids.numel() * e.num_experts // e.global_num_experts)  # estimate_local_valid_shape_m without DP
            base13 = next(c for lo, hi, c in e.w13_tuning_config if lo < vsm <= hi)
            base2 = next(c for lo, hi, c in e.w2_tuning_config if lo < vsm <= hi)
            bm = base13["block_shape"][0]
            sorted_ids, expert_ids, ntp = moe_align_block_size(topk_ids=ids, block_size=bm, num_experts=e.global_num_experts,
                                                               expert_map=r["emap"], ignore_invalid_experts=True)
            l13, l2 = e.humming_configs["w13"], e.humming_configs["w2"]
            n13 = l13.shape_n - l13.pad_shape_n
            inter = l2.shape_k - l2.pad_shape_k
            hid = r["hs"].shape[-1]
            p = {"e": e, "r": r, "ids": ids, "vsm": vsm, "base13": base13, "base2": base2, "bm": bm,
                 "kw": {"sorted_ids": sorted_ids, "expert_ids": expert_ids, "num_tokens_padded": ntp,
                        "compute_config": e.compute_config_str, "valid_shape_m": vsm},
                 "x13": e.quantize_input("w13", inputs=r["hs"][:m].contiguous(), quanted_input=None),
                 "o13": torch.empty(m * ids.size(1), n13, dtype=r["hs"].dtype, device=r["hs"].device),
                 "act": torch.empty(m * ids.size(1), inter, dtype=r["hs"].dtype, device=r["hs"].device),
                 "o2": torch.empty(m * ids.size(1), hid, dtype=r["hs"].dtype, device=r["hs"].device)}
            probs.append(p)
        return probs

    def run(p, which, cfg, out=None):
        e, r = p["e"], p["r"]
        kw = dict(p["kw"], tuning_config=_ht_json.dumps([[0, 1 << 30, cfg]]))
        if which == "w13":
            x, s = p["x13"]
            e.humming_forward("w13", inputs=x, weight=r["w1"], input_scale=s, outputs=p["o13"] if out is None else out,
                              top_k=p["ids"].size(1), **kw)
        else:
            x, s = p["x2"]
            e.humming_forward("w2", inputs=x, weight=r["w2"], input_scale=s, outputs=p["o2"] if out is None else out,
                              top_k=1, **kw)

    def timed(probs, which, cfgs, reps=7):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        for p, c in zip(probs, cfgs):  # compile + warm
            run(p, which, c)
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            ev[0].record()
            for p, c in zip(probs, cfgs):
                run(p, which, c)
            ev[1].record()
            torch.cuda.synchronize()
            ts.append(ev[0].elapsed_time(ev[1]))
        ts.sort()
        return ts[len(ts) // 2]

    def exact(probs, which, cfgs, refs):
        eq, rel = 0, 0.0
        for p, c, ref in zip(probs, cfgs, refs):
            o = torch.empty_like(ref)
            run(p, which, c, out=o)
            torch.cuda.synchronize()
            if torch.equal(o, ref):
                eq += 1
            else:
                d = (o.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)
                rel = max(rel, float(d))
        return eq, rel

    def weight_bytes(probs, which):
        tot = 0
        for p in probs:
            w = p["r"]["w1" if which == "w13" else "w2"]
            q = p["e"].quant_config
            sc = q.w1_scale if which == "w13" else q.w2_scale
            per = (w.numel() * w.element_size() + (sc.numel() * sc.element_size() if sc is not None else 0)) / w.shape[0]
            tot += per * int(torch.unique(p["ids"]).numel())
        return tot

    with torch.cuda.stream(stream), torch.no_grad():
        x = torch.empty(2**28, dtype=torch.uint8, device="cuda")
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        y = torch.empty_like(x)
        y.copy_(x)
        ev[0].record()
        for _ in range(10):
            y.copy_(x)
        ev[1].record()
        torch.cuda.synchronize()
        bw = 2 * 10 * x.numel() / (ev[0].elapsed_time(ev[1]) / 1e3)
        del x, y
        _ht_log(root, {"event": "bandwidth", "copy_gbps": round(bw / 1e9, 1)})
        results = {}
        for m in ms:
            probs = problem(m)
            for p in probs:  # served config outputs (references) and w2 inputs
                run(p, "w13", p["base13"])
                p["e"].apply_activation(activation=p["r"]["act"], input=p["o13"], output=p["act"])
                p["x2"] = p["e"].quantize_input("w2", inputs=p["act"], quanted_input=None)
            torch.cuda.synchronize()
            for which in ("w13", "w2"):
                base = [p["base13" if which == "w13" else "base2"] for p in probs]
                refs = []
                for p, c in zip(probs, base):
                    o = torch.empty_like(p["o13"] if which == "w13" else p["o2"])
                    run(p, which, c, out=o)
                    refs.append(o)
                torch.cuda.synchronize()
                t0 = timed(probs, which, base)
                gb = weight_bytes(probs, which)
                row = {"event": "base", "m": m, "which": which, "ms": round(t0, 4), "cfg": base[0],
                       "same_cfg_all_layers": len({_ht_key(c) for c in base}) == 1, "bytes_gb": round(gb / 1e9, 4),
                       "floor_frac": round(gb / bw / (t0 / 1e3), 3), "vsm": probs[0]["vsm"], "bm": probs[0]["bm"]}
                _ht_log(root, row)
                results[(m, which)] = {"base": t0, "refs": refs, "probs": probs, "best": (t0, base[0])}
                if m != tune_m or len({_ht_key(c) for c in base}) != 1:
                    continue
                # coordinate descent from the served config, two passes, until the time budget ends
                best_t, best = t0, dict(base[0])
                tried = {_ht_key(best)}
                for rnd in range(2):
                    improved = False
                    for cand in _ht_neighbors(best, which):
                        if _ht_time.time() > t_end:
                            break
                        k = _ht_key(cand)
                        if k in tried or not _ht_legal(probs[0]["e"], which, cand):
                            continue
                        tried.add(k)
                        _ht_log(root, {"event": "pending", "m": m, "which": which, "cfg": cand})
                        try:
                            eq, rel = exact(probs[:4], which, [cand] * 4, refs[:4])
                            if rel > 1e-2:
                                _ht_log(root, {"event": "wrong", "m": m, "which": which, "cfg": cand, "rel": rel})
                                continue
                            t = timed(probs, which, [cand] * len(probs))
                            eq48, rel48 = exact(probs, which, [cand] * len(probs), refs)
                        except Exception as err:
                            _ht_log(root, {"event": "error", "m": m, "which": which, "cfg": cand, "err": repr(err)[:300]})
                            continue
                        _ht_log(root, {"event": "cand", "m": m, "which": which, "ms": round(t, 4), "speedup": round(t0 / t, 4),
                                       "bit_equal_layers": eq48, "max_rel": rel48, "cfg": cand})
                        if t < best_t * 0.995:
                            best_t, best, improved = t, cand, True
                    if not improved:
                        break
                results[(m, which)]["best"] = (best_t, best)
                _ht_log(root, {"event": "best", "m": m, "which": which, "ms": round(best_t, 4), "base_ms": round(t0, 4),
                               "speedup": round(t0 / best_t, 4), "cfg": best})
        # the tuned configs at every other M (does one config serve the decode band?)
        for which in ("w13", "w2"):
            if (tune_m, which) not in results:
                continue
            best = results[(tune_m, which)]["best"][1]
            for m in ms:
                res = results.get((m, which))
                if res is None or m == tune_m:
                    continue
                probs = res["probs"]
                if best["block_shape"][0] != probs[0]["bm"]:
                    _ht_log(root, {"event": "transfer", "m": m, "which": which, "skip": "block_m differs"})
                    continue
                try:
                    t = timed(probs, which, [best] * len(probs))
                    eq, rel = exact(probs, which, [best] * len(probs), res["refs"])
                except Exception as err:
                    _ht_log(root, {"event": "transfer", "m": m, "which": which, "err": repr(err)[:300]})
                    continue
                _ht_log(root, {"event": "transfer", "m": m, "which": which, "ms": round(t, 4), "base_ms": round(res["base"], 4),
                               "speedup": round(res["base"] / t, 4), "bit_equal_layers": eq, "max_rel": rel})


def _ht_thread(root):
    try:
        _ht_sweep(root)
    except Exception as err:
        import traceback
        _ht_log(root, {"event": "fatal", "err": repr(err)[:500], "tb": traceback.format_exc()[-3000:]})
    finally:
        open(root + "/done", "w").close()


def install_htune(module):
    root = _ht_os.environ.get("ARC_HTUNE")
    if not root or _HT.get("installed"):
        return
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import HummingIndexedExperts
    _ht_os.makedirs(root, exist_ok=True)
    orig = HummingIndexedExperts.apply

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts, expert_map,
              *rest, **kw):
        out = orig(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts, expert_map,
                   *rest, **kw)
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        if (not _HT["started"] and hidden_states.size(0) >= 128
                and _ht_os.path.exists(root + "/go") and id(self) not in _HT["ids"]):
            import torch
            n = hidden_states.size(0)
            g = torch.Generator(device="cpu").manual_seed(0)
            sel = torch.randperm(n, generator=g)[:128].to(hidden_states.device)
            _HT["ids"].add(id(self))
            _HT["recs"].append({"exp": self, "hs": hidden_states.index_select(0, sel).clone(),
                                "ids": topk_ids.index_select(0, sel).clone(), "w1": w1, "w2": w2, "act": activation,
                                "emap": expert_map})
        return out

    def waiter():  # the driver writes <dir>/run after its request returned: every layer of that prefill is recorded
        while not _ht_os.path.exists(root + "/run"):
            _ht_time.sleep(2)
        _HT["started"] = True
        _ht_thread(root)

    HummingIndexedExperts.apply = apply
    _HT["installed"] = True
    _ht_threading.Thread(target=waiter, daemon=True).start()
