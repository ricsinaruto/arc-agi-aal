"""MFU-1 probes (22 Oct). ``names``: dequant-related names and signatures in Axolotl's sonicmoe NVFP4 modules (no
source). ``decode L``: one MoE layer's NVFP4 experts decoded by Axolotl's dequant path and by native_experts' decode,
compared bit for bit (share equal, signed mean relative difference)."""
import inspect
import json
import os
import sys


def source(*names_):
    """inspect.getsource of named Axolotl sonicmoe functions (module.attr), as modal_mfu_d's sources() prints them."""
    import importlib

    for full in names_:
        mod, attr = full.rsplit(".", 1)
        pkg = mod if mod.startswith("transformers.") else f"axolotl.integrations.kernels.libs.sonicmoe.{mod}"
        obj = getattr(importlib.import_module(pkg), attr)
        print(f"# ---- {full}\n{inspect.getsource(obj)}", flush=True)


def names():
    import importlib

    for mod in ("nvfp4", "nvfp4_lora", "experts"):
        m = importlib.import_module(f"axolotl.integrations.kernels.libs.sonicmoe.{mod}")
        for n in dir(m):
            if any(k in n.lower() for k in ("deq", "decode", "dense", "fp4", "unpack", "reference", "backend")):
                obj = getattr(m, n)
                try:
                    sig = str(inspect.signature(obj)) if callable(obj) else type(obj).__name__
                except (TypeError, ValueError):
                    sig = type(obj).__name__
                print(json.dumps({"module": mod, "name": n, "sig": sig[:300]}), flush=True)


MODEL = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"


def load_experts(layer):
    """One layer's routed experts as the trainer's loader fuses them (modal_mfu_d.load_layer): w1 (gate|up), w2."""
    from pathlib import Path

    import torch
    from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (
        fuse_nvfp4_experts,
    )
    from safetensors import safe_open

    conf = json.loads((Path(MODEL) / "config.json").read_text())
    text = conf.get("text_config", conf)
    wmap = json.loads((Path(MODEL) / "model.safetensors.index.json").read_text())["weight_map"]
    base = f"model.language_model.layers.{layer}.mlp."
    E = int(text["num_experts"])
    handles = {}

    def get(key):
        if wmap[key] not in handles:
            handles[wmap[key]] = safe_open(str(Path(MODEL) / wmap[key]), framework="pt", device="cpu")
        return handles[wmap[key]].get_tensor(key)

    def proj(p):
        pre = [f"{base}experts.{e}.{p}." for e in range(E)]
        return {"qd": [get(k + "weight").view(torch.uint8).cuda() for k in pre],
                "sc": [get(k + "weight_scale").view(torch.float8_e4m3fn).cuda() for k in pre],
                "pts": [get(k + "weight_scale_2").float().reshape(()).cuda() for k in pre]}

    return fuse_nvfp4_experts([proj("gate_proj"), proj("up_proj")]), fuse_nvfp4_experts([proj("down_proj")])


def timed(fn, reps=5):
    import time

    import torch

    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return round((time.time() - t) / reps * 1000, 3)


def decode(layer="12"):
    import native_experts as N
    import torch
    from axolotl.integrations.kernels.libs.sonicmoe import nvfp4_lora

    w1, w2 = load_experts(int(layer))
    for name, w in (("w1", w1), ("w2", w2)):
        ref = nvfp4_lora.dequantize_expert_weight(w)
        W = N.Weights(w)
        out = torch.empty(W.E, W.N, W.K, device="cuda", dtype=torch.bfloat16)
        N.decode_into(W, out)
        ref_v = ref.reshape(out.shape) if ref.numel() == out.numel() else None
        row = {"proj": name, "ref_shape": list(ref.shape), "ref_dtype": str(ref.dtype), "native_shape": list(out.shape),
               "pts_unique": int(W.pts.unique().numel())}
        if ref_v is not None:
            a, b = out.float(), ref_v.float()
            nz = b != 0
            row.update(bitwise_share=float((out.view(torch.int16) == ref_v.to(torch.bfloat16).view(torch.int16)).float().mean()),
                       rel_signed_mean=float(((a - b)[nz] / b[nz].abs()).mean()), rel_abs_max=float(((a - b)[nz] / b[nz].abs()).abs().max()),
                       ms_axolotl=timed(lambda: nvfp4_lora.dequantize_expert_weight(w)),
                       ms_native=timed(lambda: N.decode_into(W, out)),
                       gbytes_out=round(out.numel() * 2 / 1e9, 2))
            diff = (out.view(torch.int16) != ref_v.to(torch.bfloat16).view(torch.int16)).nonzero()[:5].tolist()
            row["first_diffs"] = [[float(out[tuple(i)]), float(ref_v[tuple(i)])] for i in diff]
        print(json.dumps(row), flush=True)


def formula(layer="12"):
    """Which arithmetic reproduces Axolotl's dequant bit for bit (w2: native's (q s) pts differs in 0.17%)."""
    import native_experts as N
    import torch
    from axolotl.integrations.kernels.libs.sonicmoe import nvfp4_lora

    _, w2 = load_experts(int(layer))
    ref = nvfp4_lora.dequantize_expert_weight(w2).view(torch.int16)
    W = N.Weights(w2)
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
    q = W.q.to(torch.int32)
    qv = table[torch.stack([q & 15, q >> 4], dim=-1).reshape(W.E, W.N, W.K)]  # low nibble first
    sc = W.s.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    pts = W.pts.view(-1, 1, 1)
    bf = torch.bfloat16
    cands = {"(q*s)*pts": (qv * sc) * pts, "q*(s*pts)": qv * (sc * pts),
             "bf(q*s)*pts": (qv * sc).to(bf).float() * pts, "q*bf(s*pts)": qv * (sc * pts).to(bf).float(),
             "bf(bf(q*s)*bf(pts))": ((qv * sc).to(bf) * pts.to(bf)).float(),
             "(q*s)*bf(pts)": (qv * sc) * pts.to(bf).float()}
    for name, v in cands.items():
        same = (v.to(bf).view(torch.int16) == ref).float().mean()
        print(json.dumps({"formula": name, "bitwise_share": float(same)}), flush=True)
    calls = {"n": 0}
    original = nvfp4_lora.dequantize_expert_weight

    def counting(w):
        calls["n"] += 1
        return original(w)

    nvfp4_lora.dequantize_expert_weight = counting
    x = torch.randn(64, W.K * 4, device="cuda", dtype=bf)  # hidden 2560
    idx = torch.randint(0, W.E, (64, 10), device="cuda")
    wts = torch.full((64, 10), 0.1, device="cuda", dtype=bf)
    try:
        w1, _ = load_experts(int(layer))
        nvfp4_lora.grouped_moe_reference_forward(x, idx, wts, w1, None, w2, None, None, None, W.E, act="silu",
                                                 backend="dequant", concat=True, scaling1=1.0, scaling2=1.0)
    except Exception as exc:  # the call path is what we want to see
        print(json.dumps({"forward_error": repr(exc)[:300]}))
    print(json.dumps({"dequantize_expert_weight_calls": calls["n"]}), flush=True)


def qcheck(*layers):
    """Stage Q: rl1_dequant against Axolotl's dequant, bit for bit, and their speeds."""
    import rl1_dequant as Q
    import torch
    from axolotl.integrations.kernels.libs.sonicmoe import nvfp4_lora

    Q.install()
    Q.DEQUANT["on"] = True
    for layer in layers or ("0", "12", "47"):
        for name, w in zip(("w1", "w2"), load_experts(int(layer))):
            ref = Q.ORIGINAL["fn"](w)
            out = nvfp4_lora.dequantize_expert_weight(w)
            row = {"layer": int(layer), "proj": name, "shape_same": tuple(out.shape) == tuple(ref.shape),
                   "bitwise": bool(torch.equal(out.view(torch.int16), ref.view(torch.int16))),
                   "ms_axolotl": timed(lambda: Q.ORIGINAL["fn"](w)), "ms_fast": timed(lambda: Q.dequantize(w)),
                   "calls": Q.DEQUANT["calls"], "fallback": Q.DEQUANT["fallback"]}
            row["tb_s_fast"] = round(out.numel() * 2.25 / row["ms_fast"] / 1e9, 2)
            print(json.dumps(row), flush=True)
            del ref, out
            torch.cuda.empty_cache()



def trainer_patch():
    """rl1_trainer.grouped_expert_lora (copied: the trainer module is heavy to import): under autocast the grouped
    LoRA projections get BF16 casts of x and the fp32 LoRA factors."""
    import torch
    from axolotl.integrations.kernels.libs.sonicmoe import nvfp4_lora

    for cls in (nvfp4_lora.GroupedUpProjLoRA, nvfp4_lora.GroupedDownProjLoRA):
        if getattr(cls, "_probe_patched", False):
            continue
        original = cls.apply

        def apply(*args, _original=original):
            if torch.is_autocast_enabled(args[0].device.type):
                args = (args[0].to(torch.bfloat16), args[1], args[2], args[3].to(torch.bfloat16),
                        args[4].to(torch.bfloat16), *args[5:])
            return _original(*args)

        cls.apply = staticmethod(apply)
        cls._probe_patched = True


def xcheck(tokens="2048", layer="12", bench="1", mode="bf16"):
    """Stage X: rl1_experts vs Axolotl's dequant MoE forward/backward on one real layer: bitwise shares, rel errors,
    speed (fwd + bwd). mode "trainer": the trainer's dtypes (fp32 hidden, router weights and LoRA master, BF16
    autocast, ``grouped_expert_lora``)."""
    import contextlib

    import rl1_experts as X
    import torch
    from axolotl.integrations.kernels.libs.sonicmoe import nvfp4_lora

    T, E, K, H, I, R = int(tokens), 512, 10, 2560, 640, 16
    w1, w2 = load_experts(int(layer))
    X.install()
    g = torch.Generator(device="cuda").manual_seed(0)
    hidden = (torch.randn(T, H, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    logits = torch.randn(T, E, device="cuda", generator=g)
    vals, idx = torch.softmax(logits, -1).topk(K, -1)
    weights = (vals / vals.sum(-1, keepdim=True)).to(torch.bfloat16)
    lora = [(torch.rand(R * E, H, device="cuda", generator=g) * 2 - 1) / H ** 0.5,
            torch.randn(2 * I, R * E, device="cuda", generator=g) * 0.01,
            (torch.rand(R * E, I, device="cuda", generator=g) * 2 - 1) / I ** 0.5,
            torch.randn(H, R * E, device="cuda", generator=g) * 0.01]
    dout = torch.randn(T, H, device="cuda", generator=g).to(torch.bfloat16)
    if mode == "trainer":
        trainer_patch()
        hidden, weights = hidden.float() + torch.randn(T, H, device="cuda", generator=g) * 1e-3, vals / vals.sum(-1, keepdim=True)
        ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        lora = [t.to(torch.bfloat16) for t in lora]
        ctx = contextlib.nullcontext

    def run(on):
        X.EXPERTS["on"] = on
        leaves = [hidden.clone().requires_grad_(True), weights.clone().requires_grad_(True),
                  *[t.clone().requires_grad_(True) for t in lora]]
        with ctx():
            out = nvfp4_lora.grouped_moe_reference_forward(leaves[0], idx, leaves[1], w1, None, w2, None,
                                                           (leaves[2], leaves[3]), (leaves[4], leaves[5]), E,
                                                           act="silu", backend="dequant", concat=True, scaling1=2.0,
                                                           scaling2=2.0)
        out.backward(dout)
        return out.detach(), [t.grad for t in leaves]

    ref, ref_g = run(False)
    got, got_g = run(True)
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))
    row = {"mode": mode, "dtypes": dict(X.EXPERTS["dtypes"]), "why": dict(X.EXPERTS["why"]),
           "grad_dtypes": [str(t.dtype)[6:] for t in got_g], "tokens": T, "out_bitwise": float((got.view(torch.int16) == ref.view(torch.int16)).float().mean()),
           "out_rel": rel(got, ref), "calls": X.EXPERTS["calls"], "fallback": X.EXPERTS["fallback"]}
    for name, a, b in zip(("dx", "dw", "dA1", "dB1", "dA2", "dB2"), got_g, ref_g):
        row[name] = {"bitwise": round(float((a == b).float().mean()), 5), "rel": round(rel(a, b), 6)}
    print(json.dumps(row), flush=True)
    if bench == "1":
        def step(on):
            def fn():
                X.EXPERTS["on"] = on
                h = hidden.clone().requires_grad_(True)
                lw = [t.clone().requires_grad_(True) for t in lora]
                with ctx():
                    out = nvfp4_lora.grouped_moe_reference_forward(h, idx, weights, w1, None, w2, None,
                                                                   (lw[0], lw[1]), (lw[2], lw[3]), E, act="silu",
                                                                   backend="dequant", concat=True, scaling1=2.0,
                                                                   scaling2=2.0)
                out.backward(dout)
            return fn
        print(json.dumps({"bench_ms_fwd_bwd": {"axolotl": timed(step(False), 3), "fused": timed(step(True), 3)},
                          "tokens": T}), flush=True)


def order(T="2048", H="2560", K="10"):
    """Which fp32 summation order torch's ``x.view(T, K, H).sum(1, dtype=float32)`` uses (combine / gather sums):
    share of elements equal to a sequential sum and to vt0-accumulator orders, for fp32 and bf16 inputs; and
    for ``(a * b).sum(-1)`` over H (dw) against a sequential and a pairwise sum."""
    import torch

    T, H, K = int(T), int(H), int(K)
    g = torch.Generator(device="cuda").manual_seed(0)
    for dt in (torch.float32, torch.bfloat16):
        x = (torch.randn(T, K, H, device="cuda", generator=g) * torch.rand(T, K, 1, device="cuda", generator=g)).to(dt)
        ref = x.sum(1, dtype=torch.float32)
        xf = x.float()
        seq = xf[:, 0].clone()
        for k in range(1, K):
            seq = seq + xf[:, k]
        row = {"dtype": str(dt)[6:], "seq": float((seq == ref).float().mean())}
        for vt in (2, 4, 8):
            acc = [None] * vt
            for k in range(K):
                acc[k % vt] = xf[:, k].clone() if acc[k % vt] is None else acc[k % vt] + xf[:, k]
            tot = acc[0]
            for a in acc[1:]:
                if a is not None:
                    tot = tot + a
            row[f"vt{vt}"] = float((tot == ref).float().mean())
        print(json.dumps(row), flush=True)


def ccheck():
    """plain_causal's host max memo (rl1_trainer, by ast): equals the counts' max for causal masks of every shape the
    trainer makes (prefix chunks, windows after a cache), floor by any ratio equals the max of the floors, the memo
    tensor is the one qsa_select receives, and a reset clears it."""
    import ast

    import torch

    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl1_trainer.py")).read()
    tree = ast.parse(src)
    names = {"plain_causal", "_plain_causal", "causal_reset"}
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("CAUSAL", "SEL") for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "rl1_trainer_part", "exec"), ns)
    causal, ok, rows = ns["CAUSAL"], True, 0
    causal["on"] = True
    for length, keys in ((5, 5), (7, 20), (1, 1), (300, 4096), (1000, 1000), (513, 70000)):
        ns["causal_reset"]()
        mask = torch.ones(1, 1, length, keys, dtype=torch.bool).tril(keys - length)
        counts = ns["plain_causal"](mask)
        ok &= causal["counts"] is counts and causal["max"] == int(counts.max()) == keys
        ok &= all(causal["max"] // r == int(torch.div(counts, r, rounding_mode="floor").max()) for r in (4, 64, 128, 7))
        ok &= ns["plain_causal"](mask) is counts and causal["max"] == keys  # a hit keeps the memo
        float_mask = torch.where(mask, 0.0, float("-inf"))
        ns["causal_reset"]()
        ok &= causal["max"] is None and ns["plain_causal"](float_mask) is causal["counts"] and causal["max"] == keys
        rows += 1
    ns["causal_reset"]()
    odd = torch.ones(1, 1, 4, 6, dtype=torch.bool).triu(1)  # not a plain causal mask: no counts, no max
    ok &= ns["plain_causal"](odd) is None and causal["max"] is None
    print(json.dumps({"check": "causal max memo", "ok": bool(ok), "shapes": rows}), flush=True)


def ncheck():
    """PinnedLookup (rl1_trainer, extracted by ast: the trainer module is heavy to import) against the host
    embedding + .to(card): bitwise over calls of varying sizes, with the two buffer slots cycling."""
    import ast

    import torch

    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl1_trainer.py")).read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "pinned_lookup")
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "NGRAM" for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "rl1_trainer_part", "exec"), ns)
    ns["NGRAM"]["on"] = True
    emb = torch.nn.Embedding(200000, 128).to(torch.bfloat16).requires_grad_(False)
    parent = torch.nn.Module()  # as the trainer installs it: a child module of the n-gram module
    parent.ngram_embedding = emb
    names = [n for n, _ in parent.named_parameters()]
    parent.ngram_embedding = ns["pinned_lookup"](emb, torch.device("cuda"))
    look = parent.ngram_embedding
    same_names = [n for n, _ in parent.named_parameters()] == names and parent.ngram_embedding.weight is emb.weight
    g = torch.Generator().manual_seed(0)
    ok, outs = True, []
    for rows in (5000, 70000, 3000, 70000, 120000, 1):
        ids = torch.randint(0, 200000, (1, rows, 12), generator=g)
        got = look(ids)
        outs.append((got, emb(ids).to("cuda")))
    torch.cuda.synchronize()
    for got, ref in outs:  # compared after later calls reused the slots: earlier results must be intact
        ok &= bool(torch.equal(got, ref)) and got.shape == ref.shape
    print(json.dumps({"check": "ngram pinned", "bitwise": ok, "same_param_names": same_names,
                      "state_dict_keys": list(parent.state_dict()), "calls": ns["NGRAM"]["calls"]}), flush=True)


def pcheck(reps="5"):
    """Stage P: union attention backward with the cached prefix's keys excluded from dk / dv, against the full
    backward (dq and the tail's dk / dv equal up to the fp32 atomics' order: compared with a repeat of the full one)."""
    import bench_sparse2 as B
    import rl1_sparse2 as S
    import torch

    U = S.UnionSelectedAttention
    scale = 1 / 16.0
    for name, length, prefix in (("win8k@64k", 8192, 57344), ("win24k@124k", 24576, 100000), ("win32k@130k", 32768, 98304),
                                 ("win4k@4k", 4096, 0)):
        q, k, v, go = B.inputs(length, prefix, "cuda")
        picked = B.picks_real(length, prefix, "cuda")
        qs, ks, vs = (x.detach().clone().requires_grad_() for x in (q, k, v))

        def grads(on, qs=qs, ks=ks, vs=vs, picked=picked, go=go):
            U.NOGRAD_PREFIX = on
            o = U.apply(qs, ks, vs, picked, scale, None)
            return torch.autograd.grad(o, (qs, ks, vs), go)

        a, a2, b = grads(False), grads(False), grads(True)
        rel = lambda x, y: float((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-30))
        tail = slice(prefix, prefix + length)
        row = {"shape": name, "dq_equal": bool(torch.equal(a[0], b[0])),
               "dk_tail_rel": rel(b[1][:, :, tail], a[1][:, :, tail]), "dk_tail_rel_repeat": rel(a2[1][:, :, tail], a[1][:, :, tail]),
               "dv_tail_rel": rel(b[2][:, :, tail], a[2][:, :, tail]), "dv_tail_rel_repeat": rel(a2[2][:, :, tail], a[2][:, :, tail]),
               "prefix_grad_zero": bool((b[1][:, :, :prefix] == 0).all() and (b[2][:, :, :prefix] == 0).all())}
        for on in (False, True):
            U.NOGRAD_PREFIX = on
            o = U.apply(qs, ks, vs, picked, scale, None)
            row[f"bwd_ms_{'p' if on else 'full'}"] = B.timed(lambda o=o, xs=(qs, ks, vs), go=go: torch.autograd.grad(o, xs, go, retain_graph=True), int(reps))
        U.NOGRAD_PREFIX = False
        print(json.dumps(row), flush=True)
        del q, k, v, go, picked, qs, ks, vs, a, a2, b
        torch.cuda.empty_cache()


if __name__ == "__main__":
    {"names": names, "decode": decode, "formula": formula, "qcheck": qcheck, "source": source, "xcheck": xcheck, "pcheck": pcheck, "order": order, "ncheck": ncheck, "ccheck": ccheck}[sys.argv[1]](*sys.argv[2:])
