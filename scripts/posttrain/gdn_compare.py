"""MFU stage G compare (bench_gdn.py ``bench``): the model's GDN path vs rl1_gdn's on one real layer of Flash with the
trainer's LoRA, a 64K prefix state and 16K/32K windows (checkpointed forward, backward through the recompute):
speed, peak memory, numerics against each other, against an FP32 run of the model's path, and the model path's own
run-to-run spread. Extra checks: batched tails (B=2 on a widened prefix cache) and a window without a prefix."""
import json
import time

import torch
import torch.nn.functional as F

import bench_gdn as bg
import rl1_gdn as G

OUT = bg.OUT


def rel(a, b):
    a, b = a.float(), b.float()
    return round(float((a - b).norm() / b.norm().clamp_min(1e-30)), 6)


def cos(a, b):
    return round(float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)), 7)


def run(gdn, xs, xw, gy, idx, fast):
    G.GDN["fast"] = fast
    cache = bg.prefix_pass(gdn, xs, idx) if xs is not None else None
    states = None if cache is None else (cache.layers[idx].conv_states[0].clone(),
                                         cache.layers[idx].recurrent_states[0].clone())
    bg.zero(gdn)
    out, dx = bg.window(gdn, xw, cache, gy)
    grads = bg.lora_grads(gdn)
    bg.zero(gdn)
    return {"states": states, "out": out, "dx": dx, "grads": grads, "cache": cache}


def compare(a, b):
    row = {"out_rel": rel(a["out"], b["out"]), "dx_rel": rel(a["dx"], b["dx"]), "dx_cos": cos(a["dx"], b["dx"])}
    gc = {n: cos(a["grads"][n], b["grads"][n]) for n in b["grads"]}
    gr = {n: rel(a["grads"][n], b["grads"][n]) for n in b["grads"]}
    row["lora_cos_min"] = min(gc.values())
    row["lora_rel_max"] = max(gr.values())
    row["lora_cos"] = {n.replace("lora_", "").replace(".default.weight", ""): v for n, v in gc.items()}
    if a["states"] is not None and b["states"] is not None:
        row["conv_state_rel"] = rel(a["states"][0], b["states"][0])
        row["rec_state_rel"] = rel(a["states"][1], b["states"][1])
    return row


def timing(gdn, xs, xw, gy, idx, fast):
    G.GDN["fast"] = fast
    cache = bg.prefix_pass(gdn, xs, idx)
    t_prefix = bg.timed(lambda: bg.prefix_pass(gdn, xs, idx), reps=3)

    def fwd():
        bg.window(gdn, xw, cache, gy, backward=False)

    def both():
        bg.zero(gdn)
        bg.window(gdn, xw, cache, gy)

    t_fwd, t_all = bg.timed(fwd), bg.timed(both)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    both()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
    return {"prefix64k_ms": t_prefix, "fwd_ms": t_fwd, "bwd_ms": round(t_all - t_fwd, 3), "fwd_bwd_ms": t_all,
            "total_ms": round(t_prefix + t_all, 3), "window_peak_gib": round(peak, 2)}


def main(lengths):
    import rl1_gdn_test

    began = time.time()
    try:
        rl1_gdn_test.main()
    except SystemExit:
        bg.log("kernel tests FAILED on GPU (continuing to the layer compare)")
    G.install()
    gdn, text, _ = bg.build()
    H, idx = text.hidden_size, bg.LAYER
    xs = bg.hidden(bg.PREFIX, H)
    xw0 = bg.hidden(max(lengths), H, offset=bg.PREFIX)
    bg.calibrate(gdn, xw0[:, :8192])
    truth_mod = bg.build()[0]
    truth_mod.load_state_dict(gdn.state_dict())
    truth_mod.float()
    report = {"layer": idx, "prefix": bg.PREFIX, "lengths": {}}
    for L in lengths:
        xw = xw0[:, :L]
        gy = torch.randn(1, L, H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
        model = run(gdn, xs, xw, gy, idx, False)
        model2 = run(gdn, xs, xw, gy, idx, False)
        G.GDN.update(calls=0, fallback=0, why={})
        fast = run(gdn, xs, xw, gy, idx, True)
        fast2 = run(gdn, xs, xw, gy, idx, True)
        used = {"fast_calls": G.GDN["calls"], "fallback": G.GDN["fallback"], "why": dict(G.GDN["why"])}
        G.GDN["fast"] = False
        # fp32 truth: the model's path at fp32 weights/activations, no autocast
        tcache = bg.Cache(idx)
        with torch.no_grad():
            for s in range(0, xs.shape[1], bg.CHUNK):
                truth_mod(xs[:, s:s + bg.CHUNK].float(), cache_params=tcache)
        ro = tcache.read_only()
        x = xw.float().detach().requires_grad_(True)
        bg.zero(truth_mod)
        out = truth_mod(x, cache_params=ro)
        (out * gy).sum().backward()
        truth = {"out": out.detach(), "dx": x.grad, "grads": bg.lora_grads(truth_mod),
                 "states": (tcache.layers[idx].conv_states[0], tcache.layers[idx].recurrent_states[0])}
        bg.zero(truth_mod)
        del out, x, ro, tcache
        row = {"fast_vs_model": compare(fast, model), "model_rerun": compare(model2, model),
               "fast_rerun": compare(fast2, fast), "model_vs_fp32": compare(model, truth),
               "fast_vs_fp32": compare(fast, truth), "path_use": used}
        del model, model2, fast, fast2, truth
        torch.cuda.empty_cache()
        row["speed_model"] = timing(gdn, xs, xw, gy, idx, False)
        row["speed_fast"] = timing(gdn, xs, xw, gy, idx, True)
        sm, sf = row["speed_model"], row["speed_fast"]
        row["speedup"] = {k: round(sm[k] / sf[k], 2) for k in ("prefix64k_ms", "fwd_ms", "bwd_ms", "fwd_bwd_ms",
                                                                "total_ms")}
        report["lengths"][str(L)] = row
        bg.log(json.dumps({"L": L, **row}))
        torch.cuda.empty_cache()
    # batched tails (B=2 rows on a widened 16K prefix cache) and a window with no prefix: fast vs model
    extra = {}
    xs2 = xs[:, :16384]
    for name, B, pre in (("batched_b2", 2, True), ("no_prefix", 1, False)):
        L = 4096
        xw = torch.cat([bg.hidden(L, H, offset=bg.PREFIX + 7919 * b) for b in range(B)])
        gy = torch.randn(B, L, H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2))
        res = {}
        for fast in (False, True):
            G.GDN["fast"] = fast
            cache = None
            if pre:
                cache = bg.prefix_pass(gdn, xs2, idx)
                layer = cache.layers[idx]
                layer.conv_states[0] = layer.conv_states[0].expand(B, -1, -1).contiguous()
                layer.recurrent_states[0] = layer.recurrent_states[0].expand(B, -1, -1, -1).contiguous()
            bg.zero(gdn)
            out, dx = bg.window(gdn, xw, cache, gy)
            res[fast] = {"out": out, "dx": dx, "grads": bg.lora_grads(gdn), "states": None}
        extra[name] = compare(res[True], res[False])
    report["extra"] = extra
    report["path_use_total"] = {"fast_calls": G.GDN["calls"], "fallback": G.GDN["fallback"], "why": G.GDN["why"]}
    report["seconds"] = round(time.time() - began)
    bg.log(json.dumps({"extra": extra}))
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    bg.log("report written")


def keep(lengths):
    """MFU-1 GDN["keep"] (24 Oct): rl1_gdn's fast path with fla's w / h / v_new kept from the checkpoint recompute vs
    recomputed in the backward, on the real layer: bitwise out / dx / LoRA grads, window time, window peak."""
    import rl1_gdn_test

    ok = rl1_gdn_test.keep_check("cuda")
    G.install()
    gdn, text, _ = bg.build()
    H, idx = text.hidden_size, bg.LAYER
    xs = bg.hidden(bg.PREFIX, H)
    xw0 = bg.hidden(max(lengths), H, offset=bg.PREFIX)
    bg.calibrate(gdn, xw0[:, :8192])
    report = {"layer": idx, "prefix": bg.PREFIX, "kernel_check_ok": ok, "lengths": {}}
    for L in lengths:
        xw = xw0[:, :L]
        gy = torch.randn(1, L, H, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
        res, times = {}, {}
        for on in (False, True, False, True):
            G.GDN["keep"] = on
            G.GDN["kept"] = 0
            got = run(gdn, xs, xw, gy, idx, True)
            res.setdefault(on, []).append(got)
            times[on] = timing(gdn, xs, xw, gy, idx, True)
            times[on]["kept_calls"] = G.GDN["kept"]
            if on and not G.GDN["kept"]:
                raise RuntimeError("GDN keep on but bwd_keep never ran")
        G.GDN["keep"] = False

        def same(a, b):
            return {"out": bool(torch.equal(a["out"], b["out"])), "dx": bool(torch.equal(a["dx"], b["dx"])),
                    "lora": all(torch.equal(a["grads"][n], b["grads"][n]) for n in b["grads"])}

        row = {"keep_vs_base": same(res[True][0], res[False][0]), "base_rerun": same(res[False][1], res[False][0]),
               "keep_rerun": same(res[True][1], res[True][0]), "base": times[False], "keep": times[True],
               "bwd_saved_ms": round(times[False]["bwd_ms"] - times[True]["bwd_ms"], 3)}
        report["lengths"][str(L)] = row
        bg.log(json.dumps({"L": L, **row}))
        del res
        torch.cuda.empty_cache()
    (OUT / "keep.json").write_text(json.dumps(report, indent=1))
    return report
