

# ----------------------------------------------------------------------------- divergence trace (throughput, 21 Oct)
# ARC_DIVTRACE=<dir>: forward hooks on every decoder layer and its main blocks record an exact checksum of each output
# while <dir>/tag holds a tag (one JSON line per hooked call to <dir>/<tag>.jsonl). Two cold passes of the same prompt
# then show the first module whose output differs. Eager only; diagnostic (syncs on every hooked call).
import json as _dt_json
import os as _dt_os
import re as _dt_re

_DT = {"stamp": None, "tag": None, "calls": {}}
_DT_DUMP = {k for k in (_dt_os.environ.get("ARC_DIVTRACE_DUMP") or "").split(",") if k}
# ARC_DIVTRACE_RERUN=<suffix>: on each tagged pass, the first call of every module ending in <suffix> (pure blocks
# only, e.g. mlp.experts) is re-run 3 times in place on the same inputs and compared with its own output.
_DT_RERUN = tuple(k for k in (_dt_os.environ.get("ARC_DIVTRACE_RERUN") or "").split(",") if k)
_DT_RERUN_LAYER = "LAYER" in _DT_RERUN  # whole decoder layers (model.layers.N) too
_DT_RERUN = tuple(k for k in _DT_RERUN if k != "LAYER") or ("\0",)


def _dt_rerun_match(name):
    return name.endswith(_DT_RERUN) or (_DT_RERUN_LAYER and _dt_re.search(r"layers\.\d+$", name) is not None)
# A tag ending in Z / R fills every cache storage (attributes named kv_cache, nested) with 0x00 / 0x3C bytes before the
# pass's first hooked call: a module whose output changes with the fill from equal inputs reads memory it never wrote.


def _dt_fill(root_mod, byte):
    seen, n, info = set(), 0, {}

    def walk(v, path):
        nonlocal n
        if isinstance(v, torch.Tensor):
            if v.numel() and v.is_cuda:
                st = v.untyped_storage()
                if st.data_ptr() not in seen:
                    seen.add(st.data_ptr())
                    torch.empty(0, dtype=torch.uint8, device=v.device).set_(st).fill_(byte)
                    n += st.nbytes()
                    info[path] = st.nbytes()
        elif isinstance(v, (list, tuple)):
            for j, x in enumerate(v):
                walk(x, f"{path}[{j}]")
    for name, mod in root_mod.named_modules():
        if "kv_cache" in vars(mod):
            walk(vars(mod)["kv_cache"], name + ".kv_cache")
    torch.cuda.synchronize()
    return {"byte": byte, "storages": len(seen), "gib": round(n / 2**30, 2), "first": sorted(info)[:6]}


def _dt_cmp(a, b):
    a, b = a.float(), b.float()
    d = (a - b)
    return {"rel": float(d.norm() / a.norm().clamp_min(1e-30)), "max_abs": float(d.abs().max()),
            "n_diff": int((d != 0).sum()), "numel": a.numel()}
_DT_RX = _dt_re.compile(r"(?:^|\.)layers\.(\d+)(?:\.(self_attn|linear_attn|mlp|mlp\.gate|mlp\.experts|mlp\.shared_expert"
                        r"|input_layernorm|post_attention_layernorm|self_attn\.indexer))?$")


def _dt_tag(root):
    try:
        st = _dt_os.stat(root + "/tag")
    except FileNotFoundError:
        return None
    stamp = (st.st_mtime_ns, st.st_size, st.st_ino)
    if _DT["stamp"] != stamp:
        with open(root + "/tag") as fh:
            _DT.update(stamp=stamp, tag=fh.read().strip() or None, calls={})
    return _DT["tag"]


def _dt_sum(t):
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return None
    x = t.detach().contiguous()
    bits = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[x.element_size()]
    v = x.view(bits).reshape(-1).to(torch.int64)
    w = torch.arange(1, v.numel() + 1, device=v.device, dtype=torch.int64) % 1000003
    flat = x.reshape(-1)
    idx = torch.arange(256, device=flat.device, dtype=torch.int64) * (flat.numel() - 1) // 255  # int math: no OOB
    return [int(v.sum()), int((v * w).sum()), float(x.float().abs().sum()), list(x.shape),
            [round(float(f), 6) for f in flat[idx].float().tolist()]]


_DT_SEL = _dt_os.environ.get("ARC_DIVTRACE_SEL") == "1"
_DT_TOPK = {}
_DT_MULT = {}


def _dt_select(root, tag, name, k, out, topk):
    """Selection record (ARC_DIVTRACE_SEL=1): MoE top-k ids + k-th margin from the router logits; QSA selected-set hash
    per row (full sorted rows for layer 3's first call)."""
    d = f"{root}/sel/{tag}"
    _dt_os.makedirs(d, exist_ok=True)
    if name.endswith("mlp.gate"):
        logits = out.float()
        v, i = logits.topk(topk + 1, dim=-1)
        rec = {"ids": i[:, :topk].sort(dim=-1).values.to(torch.int16).cpu(),
               "margin": (v[:, topk - 1] - v[:, topk]).cpu(), "experts": logits.shape[-1]}
    else:
        idx = out
        srt = torch.where(idx < 0, 2147483647, idx).sort(dim=-1).values
        key = (srt.shape[1], str(srt.device))
        if key not in _DT_MULT:
            g = torch.Generator(device="cpu").manual_seed(12345)
            _DT_MULT[key] = torch.randint(1, 2**31 - 1, (srt.shape[1],), generator=g, dtype=torch.int64).to(srt.device)
        valid = srt != 2147483647
        h = (torch.where(valid, srt.long() + 1, 0) * _DT_MULT[key]).sum(dim=-1)
        rec = {"hash": h.cpu(), "count": valid.sum(dim=-1).to(torch.int32).cpu()}
        if name.endswith("layers.3.self_attn.indexer") and k == 0:
            rec["rows"] = srt.to(torch.int32).cpu()
    torch.save(rec, f"{d}/{name}-{k}.pt")


_DT_OPS = _dt_os.environ.get("ARC_DIVTRACE_OPS") == "1"


def _dt_sig(t):
    x = t.detach()
    if x.numel() == 0 or not x.is_cuda:
        return [list(x.shape)]
    x = x.contiguous()
    bits = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[x.element_size()]
    v = x.view(bits).reshape(-1).to(torch.int64)
    w = torch.arange(1, v.numel() + 1, device=v.device, dtype=torch.int64) % 1000003
    return [list(x.shape), int(v.sum()), int((v * w).sum())]


def _dt_ops_trace(mod, sa, sk):
    """Op-level trace of one call of a FusedMoE layer's inner path (_forward_impl, below the opaque custom op): per
    dispatched op, input / mutated-input / output signatures. Triton launches are invisible; their effect shows as the
    next op's inputs."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class Rec(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.rows = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            flat = [t for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)]
            ins = [_dt_sig(t) for t in flat[:8]]
            out = func(*args, **kwargs)
            after = [_dt_sig(t) for t in flat[:8]]
            outs = [_dt_sig(t) for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)][:4]
            self.rows.append([str(func), ins, after, outs])
            return out

    cl = lambda v: v.detach().clone() if isinstance(v, torch.Tensor) else v  # noqa: E731
    args, kw = [cl(v) for v in sa], {k: cl(v) for k, v in sk.items()}
    rec = Rec()
    had = "_forward_entry" in mod.__dict__
    old = mod.__dict__.get("_forward_entry")
    if hasattr(mod, "_forward_impl"):  # bypass the opaque moe_forward custom op so its insides are dispatched
        mod._forward_entry = lambda h, lg, sh, ids, *rest: mod._forward_impl(h, lg, sh, ids)
    try:
        with rec:
            mod(*args, **kw)
    finally:
        if had:
            mod._forward_entry = old
        elif "_forward_entry" in mod.__dict__:
            del mod._forward_entry
    torch.cuda.synchronize()
    return rec.rows


def _dt_ops_compare(traces):
    base = traces[0]
    report = {"ops": len(base), "lengths": [len(t) for t in traces], "first": None,
              "names": [r[0] for r in base][:80]}
    for other in traces[1:]:
        for i, (a, b) in enumerate(zip(base, other)):
            if a != b:
                where = "inputs" if a[1] != b[1] else ("outputs/mutated" if (a[2] != b[2] or a[3] != b[3]) else "?")
                cand = {"index": i, "op": a[0], "differs_in": where,
                        "prev_op": base[i - 1][0] if i else None}
                if report["first"] is None or i < report["first"]["index"]:
                    report["first"] = cand
                break
    return report


def install_divtrace(module):
    root = _dt_os.environ.get("ARC_DIVTRACE")
    if not root or getattr(module, "_arc_divtrace", False):
        return
    if "mtp" in type(module).__name__.lower():  # the MTP draft's own tree is also named model.layers.N: never hook it
        log("DIVTRACE", {"skipped": type(module).__name__})
        return
    _dt_os.makedirs(root, exist_ok=True)
    hooked = 0
    mods = dict(module.named_modules())
    for name in mods:
        if name.endswith("mlp.gate") and hasattr(mods.get(name[:-4] + "experts"), "top_k"):
            _DT_TOPK[name] = int(mods[name[:-4] + "experts"].top_k)
    for name, child in module.named_modules():
        if "mtp" in name or not _DT_RX.search(name):
            continue

        def pre(mod, args, kwargs, name=name):
            tag = _dt_tag(root)
            if tag is None or _DT.get("inside"):
                return
            if tag[-1:] in ("Z", "R") and _DT.get("filled") != tag:
                _DT["filled"] = tag
                rep = _dt_fill(module, 0 if tag.endswith("Z") else 0x3C)
                with open(f"{root}/{tag}.fill.json", "w") as fh:
                    fh.write(_dt_json.dumps(rep))
            ins = [v for v in (*args, *kwargs.values()) if isinstance(v, torch.Tensor)]
            _DT.setdefault("pre", {}).setdefault(name, []).append([_dt_sum(v) for v in ins[:4]])
            if _dt_rerun_match(name) and _DT["calls"].get(name, 0) == 0:
                cl = lambda v: v.detach().clone() if isinstance(v, torch.Tensor) else v  # noqa: E731
                _DT.setdefault("saved", {})[name] = ([cl(v) for v in args], {k: cl(v) for k, v in kwargs.items()})

        def hook(mod, args, kwargs, out, name=name):
            tag = _dt_tag(root)
            if tag is None:
                return
            k = _DT["calls"].get(name, 0)
            _DT["calls"][name] = k + 1
            outs = out if isinstance(out, (tuple, list)) else (out,)
            if _DT.get("inside"):
                return
            if _DT_SEL and (name.endswith("mlp.gate") or name.endswith("self_attn.indexer")) \
                    and isinstance(outs[0], torch.Tensor):
                _dt_select(root, tag, name, k, outs[0], _DT_TOPK.get(name, 8))
            stack = _DT.get("pre", {}).get(name) or [None]
            rec = {"m": name, "k": k, "out": [_dt_sum(o) for o in outs[:2]], "in": stack.pop()}
            if _dt_rerun_match(name) and k == 0 and isinstance(outs[0], torch.Tensor):
                ref = outs[0].detach().clone()
                _DT["inside"] = True
                try:
                    reruns = []
                    sa, sk = _DT["saved"].pop(name)
                    cl = lambda v: v.detach().clone() if isinstance(v, torch.Tensor) else v  # noqa: E731
                    for _ in range(3):
                        again = mod(*[cl(v) for v in sa], **{kk: cl(v) for kk, v in sk.items()})
                        again = again[0] if isinstance(again, (tuple, list)) else again
                        reruns.append(_dt_cmp(ref, again))
                    rec["rerun"] = reruns
                    if _DT_OPS and name.endswith("mlp.experts") and name.split(".")[-3] in ("0", "1", "4"):
                        try:
                            rec["ops"] = _dt_ops_compare([_dt_ops_trace(mod, sa, sk) for _ in range(4)])
                        except Exception as error:  # diagnostic only
                            rec["ops"] = {"error": repr(error)[:300]}
                finally:
                    _DT["inside"] = False
            if f"{name}:{k}" in _DT_DUMP:
                ins = [v.detach().cpu() for v in (*args, *kwargs.values()) if isinstance(v, torch.Tensor)]
                torch.save({"in": ins, "out": [o.detach().cpu() for o in outs if isinstance(o, torch.Tensor)]},
                           f"{root}/{tag}-{name}-{k}.pt")
            with open(f"{root}/{tag}.jsonl", "a") as fh:
                fh.write(_dt_json.dumps(rec) + "\n")

        child.register_forward_pre_hook(pre, with_kwargs=True)
        child.register_forward_hook(hook, with_kwargs=True)
        hooked += 1
    if _DT_PAD:
        _dt_install_pad(module, root)
    module._arc_divtrace = True
    log("DIVTRACE", {"hooked": hooked, "dir": root, "pad": _DT_PAD})

# ARC_DIVTRACE_PAD=<sizes> (22 Oct, item 2): CUDA-graph padding exactness. For leaf *Linear* layers called with M rows where M is not a capture size, re-run the module
# on the input padded to the next capture size (pad rows = copies of real rows, so the router load is realistic) and
# compare the first M output rows bit for bit. Up to 3 such calls per module per tag; one JSON line per check.
_DT_PAD = sorted(int(v) for v in (_dt_os.environ.get("ARC_DIVTRACE_PAD") or "").split(",") if v)


def _dt_install_pad(module, root):
    import bisect as _bs
    seen = {}

    def first_rows(t):
        return t[0] if isinstance(t, (tuple, list)) else t

    for name, child in module.named_modules():
        if "mtp" in name:
            continue
        cls = type(child).__name__
        # leaf linear layers only (stateless per-row): MoE runners keep shared-expert output slots, so re-running
        # mlp.experts breaks the real forward (pad v1, 22 Oct: AssertionError in shared_experts.forward)
        if "Linear" not in cls or any(True for _ in child.children()):
            continue

        def hook(mod, args, kwargs, out, name=name, cls=cls):
            tag = _dt_tag(root)
            if tag is None or _DT.get("inside_pad"):
                return
            xs = [v for v in args if isinstance(v, torch.Tensor)]
            if not xs:
                return
            m = xs[0].shape[0]
            if m <= 104 or m in _DT_PAD or m > _DT_PAD[-1] or seen.get((tag, name), 0) >= 3:
                return
            seen[(tag, name)] = seen.get((tag, name), 0) + 1
            bucket = _DT_PAD[_bs.bisect_left(_DT_PAD, m)]
            idx = torch.arange(bucket, device=xs[0].device) % m

            def pad(v):
                return v[idx].contiguous() if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == m else v
            rec = {"m": name, "cls": cls, "M": m, "bucket": bucket, "tag": tag}
            _DT["inside_pad"] = True
            try:
                ref = first_rows(out).detach().clone()
                again = first_rows(mod(*[pad(v) for v in args], **{k: pad(v) for k, v in kwargs.items()}))
                again2 = first_rows(mod(*args, **kwargs))
                d = (again[:m].float() - ref.float())
                rec.update(n_diff=int((d != 0).sum()), numel=ref.numel(), max_abs=float(d.abs().max()),
                           rel=float(d.norm() / ref.float().norm().clamp_min(1e-30)),
                           same_m_rerun_n_diff=int((again2.float() - ref.float() != 0).sum()))
            except Exception as error:  # diagnostic only
                rec["error"] = repr(error)[:300]
            finally:
                _DT["inside_pad"] = False
            with open(f"{root}/{tag}.pad.jsonl", "a") as fh:
                fh.write(_dt_json.dumps(rec) + "\n")

        child.register_forward_hook(hook, with_kwargs=True)
