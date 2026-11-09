"""One-B300 smoke of the trunk/sft fast paths (modal_sft.py --smoke): ``kernels`` checks single real layers of the
student, fast path vs the model's (GDN, hyper-connections, full attention; forward and every grad) and the export
grid's scales on GPU vs CPU; ``report`` reads the two trainer boots' events and writes <out>/smoke.json with each
pre-registered bar (launch config ``smoke``)."""
import json
import sys
import time
from pathlib import Path

import torch

import attention
import modelio
import qat

BF16 = torch.bfloat16


def layer(student, text, idx, device="cuda"):
    """Decoder layer ``idx`` of the student with its tensors."""
    from safetensors import safe_open
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    student = Path(student)
    index = json.loads((student / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{idx}."
    state = {}
    for name, f in index.items():
        if name.startswith(prefix):
            with safe_open(str(student / f), framework="pt") as h:
                state[name[len(prefix):]] = h.get_tensor(name)
    mod = m.Qwen4ExpTextDecoderLayer(text, idx).to(device=device, dtype=BF16)
    missing, unexpected = mod.load_state_dict(state, strict=False)
    if unexpected or [k for k in missing if "rotary" not in k]:
        raise RuntimeError(f"layer {idx}: missing {missing[:4]} unexpected {unexpected[:4]}")
    return mod


def compare(run, mod, x):
    """run() twice per path: (fwd rel err, fwd cos, min grad cos, worst grad, fast repeat bitwise)."""
    def once(fast):
        mod.zero_grad(set_to_none=True)
        x.grad = None
        out = run(fast)
        g = torch.Generator(device=x.device).manual_seed(0)
        (out.float() * torch.randn(out.shape, generator=g, device=x.device)).sum().backward()
        grads = {"x": x.grad.clone(), **{n: p.grad.clone() for n, p in mod.named_parameters() if p.grad is not None}}
        return out.detach().clone(), grads

    ref, gref = once(False)
    got, ggot = once(True)
    again, _ = once(True)
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), 0))  # noqa: E731
    gc = {k: cos(ggot[k], gref[k]) for k in gref if k in ggot}
    worst = min(gc, key=gc.get)
    return {"fwd_rel": float((got.float() - ref.float()).abs().max() / ref.float().abs().max()), "fwd_cos": cos(got, ref),
            "fwd_bitwise": bool(torch.equal(got, ref)), "grad_cos_min": gc[worst], "grad_worst": worst,
            "grads_missing": sorted(set(gref) - set(ggot)), "repeat_bitwise": bool(torch.equal(got, again))}


def kernels(spec, T=8192):
    from transformers import AutoConfig, DynamicCache

    import gdn  # Triton (GPU); the module imports on CPU without them
    import hyper

    student = spec["train"]["student"]
    text = AutoConfig.from_pretrained(student).text_config
    types = list(text.layer_types)
    lin = next(i for i, t in enumerate(types) if t == "linear_attention" and i + 1 not in (text.ple_layer_ids or []))
    full = next(i for i, t in enumerate(types) if t != "linear_attention")
    out, g = {"T": T, "linear_layer": lin, "full_layer": full}, torch.Generator(device="cuda").manual_seed(1)
    gdn.install()
    hyper.install()
    attention.install(mode="masked")
    dec = layer(student, text, lin)

    # GDN, no cache and over a no-grad prefix (the trainer's window pass)
    x = (torch.randn(1, T, text.hidden_size, generator=g, device="cuda") * 0.5).to(BF16).requires_grad_()

    def gdn_run(fast, cache=None):
        gdn.GDN["fast"] = fast
        return dec.linear_attn(x, cache_params=cache)

    out["gdn"] = compare(gdn_run, dec.linear_attn, x)

    def gdn_prefix(fast):
        gdn.GDN["fast"] = fast
        cache = DynamicCache(config=text)
        with torch.no_grad():
            dec.linear_attn(x[:, : T // 2].detach(), cache_params=cache)
        return dec.linear_attn(x[:, T // 2:], cache_params=modelio.read_only(cache))

    out["gdn_prefix"] = compare(gdn_prefix, dec.linear_attn, x)
    for n in ("in_proj_qkv", "in_proj_z", "out_proj"):  # dense_int8_train: the served INT8 weights, both paths
        qat.wrap_int8(getattr(dec.linear_attn, n))
    qat.DENSE["on"] = True
    try:
        out["gdn_int8"] = compare(gdn_run, dec.linear_attn, x)
    finally:
        qat.DENSE["on"] = False
    gdn.GDN["fast"] = False

    # hyper-connections: the whole decoder layer, BF16 stream
    s = (torch.randn(1, T, text.hc_count * text.hidden_size, generator=g, device="cuda")).to(BF16).requires_grad_()

    def hyper_run(fast):
        hyper.HYPER["on"] = fast
        return dec(s, position_embeddings=None)

    out["hyper"] = compare(hyper_run, dec, s)
    out["hyper"]["fallback"] = dict(hyper.HYPER["why"])
    hyper.HYPER["on"] = False
    del dec

    att = layer(student, text, full)
    out["attention"] = attention_check(att, text, T, g)

    # the export grid's scales computed on GPU vs CPU (search 20): near-tie flips
    flips = total = 0
    for name in ("gate_up_proj", "down_proj"):
        w = getattr(att.mlp.experts, name).detach()[:16]
        a, b = qat.scales_of(w).cpu(), qat.scales_of(w.cpu())
        flips, total = flips + int((a != b).sum()), total + a.numel()
    out["grid"] = {"groups": total, "scale_flips": flips, "share": flips / total}
    return out


def attention_check(att, text, T, g, device="cuda", fast="triton", impl="sdpa"):
    """Full attention: the model's forward (the indexer's boolean mask, SDPA as the trainer loads it) vs the indexer +
    selected-key kernel. A layer built from the bare config has no attention backend and falls back to eager, which
    adds a boolean mask as numbers (no masking): the 26 Oct smoke's cos 0.973."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    att.self_attn.config._attn_implementation = impl
    h = (torch.randn(1, T, text.hidden_size, generator=g, device=device) * 0.5).to(BF16).requires_grad_()
    rope = m.Qwen4ExpTextRotaryEmbedding(config=text).to(device)
    pos = torch.arange(T, device=device).view(1, 1, -1).expand(3, 1, -1)
    pe = rope(h, pos)
    mask = torch.ones(T, T, dtype=torch.bool, device=device).tril()[None, None]

    def run(on):
        attention.SPARSE["mode"] = fast if on else "masked"
        attention.reset()
        return att.self_attn(h, pe, attention_mask=mask)[0]

    calls = attention.SPARSE["calls"]
    got = compare(run, att.self_attn, h)
    got["calls"] = attention.SPARSE["calls"] - calls
    attention.SPARSE["mode"] = "masked"
    return got


def fp8_experts(model):
    """The routed experts to W8A16 in place (rl1_trainer.ExpertsFP8, sft-heal: per-row RTN to e4m3, fp32 scales;
    each MoE block dequantizes into one shared BF16 scratch in a pre-hook, Triton, bf16(fp32(code) x scale))."""
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(codes, scales, out, cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        sc = tl.load(scales + row)
        for c in range(0, cols, BLOCK):
            offs = c + tl.arange(0, BLOCK)
            mask = offs < cols
            base = row.to(tl.int64) * cols
            x = tl.load(codes + base + offs, mask=mask).to(tl.float32)
            tl.store(out + base + offs, (x * sc).to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)

    def rtn(w, chunk=8):
        codes = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
        scales = torch.empty(w.shape[:-1] + (1,), dtype=torch.float32, device=w.device)
        for e in range(0, w.shape[0], chunk):
            x = w[e:e + chunk].float()
            sc = (x.abs().amax(-1, keepdim=True) / 448.0).clamp(min=torch.finfo(torch.float32).tiny)
            codes[e:e + chunk], scales[e:e + chunk] = (x / sc).to(torch.float8_e4m3fn), sc
        return codes, scales

    store, scratch = {}, None
    for lay in model.model.language_model.layers:
        ex = lay.mlp.experts
        packed = [rtn(ex.gate_up_proj.data), rtn(ex.down_proj.data)]
        if scratch is None:
            scratch = (ex.gate_up_proj.data, ex.down_proj.data)
        ex.gate_up_proj.data, ex.down_proj.data = scratch
        store[id(ex)] = packed

        def hook(mod, args):
            for (c, sc), out in zip(store[id(mod)], scratch):
                kernel[(c.numel() // c.shape[-1],)](c, sc, out, c.shape[-1], BLOCK=1024)

        ex.register_forward_pre_hook(hook)
    torch.cuda.empty_cache()


def teacher_speed(spec, n=3, k=64, save=None):
    """Teacher scoring speed on one card over recorded examples, each over its whole recorded context (teacher.topk:
    8K-token pieces over a cache), with the trainer's fast paths (GDN, hyper-connections, QSA indexer + Triton
    selected attention): BF16 Flash with the n-gram table in host RAM (if it loads), then the same model with W8A16
    routed experts; top-1 agreement FP8 vs BF16; the scoring pass over the data pool projected from each tok/s.
    ``save(out)`` after each mode."""
    from transformers import AutoModelForMultimodalLM

    import feed
    import gdn
    import hyper
    import teacher

    t = spec["smoke"]["teacher"]
    root = Path(spec["train"]["data"])
    index = json.loads((root / "index.json").read_text())
    pool = {s: sum(int(e["tokens"]) for e in index[s]["files"]) for s in ("train", "heldout")}
    files = index["heldout"]["files"][:n]
    exs = [feed.load_example(root / e["file"]) for e in files]
    sizes = [int(e["input_ids"].shape[1]) for e in exs]
    tokens = sum(sizes)
    out = {"examples": n, "tokens": tokens, "sizes": sizes, "reply_rows": 0, "pool_tokens": pool}
    save = save or (lambda _: None)
    attention.install(mode="triton")
    gdn.install()
    gdn.GDN.update(fast=True)
    hyper.install()
    hyper.HYPER["on"] = True
    modelio.table_in_host_ram()
    began = time.time()
    try:
        with modelio.parallel_reads(t) as how:
            model = AutoModelForMultimodalLM.from_pretrained(t, dtype=BF16, device_map={"": "cuda:0"},
                                                             experts_implementation="grouped_mm")
    except torch.OutOfMemoryError as error:
        return {**out, "bf16": f"does not fit one card: {str(error)[:200]}"}
    out["load"] = {"seconds": round(time.time() - began, 1), "gib": round(torch.cuda.memory_allocated() / 2**30, 1), **how}
    model.eval()
    tops = {}
    for mode in ("bf16", "fp8"):
        if mode == "fp8":
            fp8_experts(model)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        calls, falls = attention.SPARSE["calls"], attention.SPARSE["fallback"]
        began, got, each = time.time(), [], []
        for ex in exs:
            t0 = time.time()
            got.append(teacher.topk(model, ex, k))
            torch.cuda.synchronize()
            each.append(round(time.time() - t0, 1))
        sec = time.time() - began
        tops[mode] = torch.cat([g["ids"][:, 0] for g in got])
        rate = tokens / sec
        out[mode] = {"seconds": round(sec, 1), "per_example_s": each, "tok_s": round(rate, 1),
                     "pool_hours": {s: round(v / rate / 3600, 2) for s, v in pool.items()},
                     "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                     "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 1),
                     "selected_calls": attention.SPARSE["calls"] - calls,
                     "attention_fallbacks": attention.SPARSE["fallback"] - falls,
                     "gdn_fallbacks": dict(gdn.GDN.get("why", {})), "hyper_fallbacks": dict(hyper.HYPER["why"])}
        save(out)
    out["reply_rows"] = int(tops["bf16"].numel())
    out["top1_fp8_vs_bf16"] = float((tops["fp8"] == tops["bf16"]).float().mean())
    return out


BARS = """
gdn, gdn_prefix, gdn_int8 (dense_int8_train), attention: fwd_cos >= 0.9999 and fwd_rel <= 0.02; grad_cos_min >= 0.999; no grads missing; fast repeat bitwise
hyper: fwd_bitwise; grad_cos_min >= 0.999; no fallback (BF16 stream)
grid: scale_flips share <= 1e-5 (else the export runs the grid on the trainer's device)
boot1: step-1 peak <= 254.3 GiB (95% of 267.7); report vs the 244 GiB prediction; step-1 peak - warm peak <= 4 GiB (vision ahead)
snapshot: page-locked; buffer seconds <= 90; host avail after >= 32 GiB; the async write ends (ckpt_written)
boot2 (resume from boot1's last checkpoint, fake_quant on): warm_optimizer <= 25% of boot1's; new Triton cache files <= 5% of
  boot1's; step-3 peak <= 254.3 GiB and within 6 GiB of boot1's step-1 peak (QAD prediction 250)
teacher (measured, no bar; fast paths on, whole recorded contexts): BF16 Flash on one card (fits or not), its tok/s and
  peak; W8A16-expert tok/s and peak; top-1 agreement FP8 vs BF16; pool scoring hours (reference: PTH's FP8 teacher ~10K
  tok/s, c7n)
"""


def report(out, kernels_json, triton_counts):
    if not (Path(out) / "events.jsonl").is_file():  # a teacher-only job
        tj = Path(out) / "teacher.json"
        result = {"teacher": json.loads(tj.read_text()) if tj.is_file() else "not run"}
        (Path(out) / "smoke.json").write_text(json.dumps(result, indent=1))
        print("SMOKE " + json.dumps(result), flush=True)
        return result
    rows = [json.loads(x) for x in (Path(out) / "events.jsonl").read_text().splitlines()]
    boots = [r for r in rows if r.get("event") == "boot" and "seconds" in r]
    steps = [r for r in rows if r.get("event") == "step"]
    buf = [r for r in rows if r.get("event") == "ckpt_buffer"]
    written = [r for r in rows if r.get("event") == "ckpt_written"]
    k = json.loads(Path(kernels_json).read_text()) if Path(kernels_json).is_file() else None
    ok = {}
    for name in [n for n in ("gdn", "gdn_prefix", "gdn_int8", "attention") if n in k] if k else ():
        r = k[name]
        ok[name] = (r["fwd_cos"] >= 0.9999 and r["fwd_rel"] <= 0.02 and r["grad_cos_min"] >= 0.999
                    and not r["grads_missing"] and r["repeat_bitwise"])
    if k:  # a boots-only job has no kernel checks
        ok["hyper"] = k["hyper"]["fwd_bitwise"] and k["hyper"]["grad_cos_min"] >= 0.999 and not k["hyper"]["fallback"]
        ok["grid"] = k["grid"]["share"] <= 1e-5
    if not boots or not any(r["step"] == 1 for r in steps):  # boot 1 never stepped: nothing to judge
        result = {"ok": False, "bars": ok, "kernels": k, "boots": boots, "error": "boot 1 reached no step"}
        (Path(out) / "smoke.json").write_text(json.dumps(result, indent=1))
        print("SMOKE " + json.dumps(result), flush=True)
        return result
    b1, b2 = boots[0], boots[-1]
    s1 = next(r for r in steps if r["step"] == 1)
    s3 = next((r for r in steps if r["step"] == 3), None)
    warm_peak = b1["warm"].get("peak_gib") or 0
    ok["peak"] = s1["peak_gib"] <= 254.3
    ok["vision_ahead"] = s1["peak_gib"] - warm_peak <= 4
    ok["snapshot"] = bool(buf) and buf[0]["pinned"] and buf[0]["seconds"] <= 90 and bool(written)
    w1, w2 = b1["seconds"].get("warm_optimizer", 0), b2["seconds"].get("warm_optimizer", 0)
    ok["compile_cache"] = len(boots) == 2 and w2 <= 0.25 * max(w1, 1e-9) and triton_counts[1] <= 0.05 * max(triton_counts[0], 1)
    ok["qad_peak"] = s3 is not None and s3["peak_gib"] <= 254.3 and s3["peak_gib"] - s1["peak_gib"] <= 6
    result = {"ok": all(ok.values()), "bars": ok, "kernels": k, "boot1": b1, "boot2": b2, "step1": s1, "step3": s3,
              "snapshot": buf[:1] + written[:2], "triton_new_files": triton_counts, "bars_text": BARS.strip(),
              "teacher": json.loads(tj.read_text()) if (tj := Path(out) / "teacher.json").is_file() else "not run"}
    (Path(out) / "smoke.json").write_text(json.dumps(result, indent=1))
    print("SMOKE " + json.dumps({"ok": result["ok"], "bars": ok}), flush=True)
    return result


if __name__ == "__main__":
    mode, arg = sys.argv[1], json.loads(sys.argv[2])
    if mode == "kernels":
        began = time.time()
        got = kernels(arg)
        got["seconds"] = round(time.time() - began, 1)
        Path(arg["kernels_json"]).parent.mkdir(parents=True, exist_ok=True)
        Path(arg["kernels_json"]).write_text(json.dumps(got, indent=1))
        print("KERNELS " + json.dumps(got), flush=True)
    elif mode == "teacher":
        def save(got):
            Path(arg["teacher_json"]).parent.mkdir(parents=True, exist_ok=True)
            Path(arg["teacher_json"]).write_text(json.dumps(got, indent=1))
            print("TEACHER-PART " + json.dumps(got), flush=True)

        got = teacher_speed(arg, save=save)
        save(got)
        print("TEACHER " + json.dumps(got), flush=True)
    else:
        report(arg["out"], arg["kernels_json"], arg["triton"])
