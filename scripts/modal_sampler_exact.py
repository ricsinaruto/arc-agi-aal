"""Sampler exactness check of the served sampling path: do the MTP draft head's verification
and the top-k / top-p filters reproduce the target's next-token distribution? Modal CPU only (Triton interpreter).

The served path (v8.2 / v8.4 save-run log: vLLM 0.1.dev20073+g8e685d198, ``SpeculativeConfig(method='mtp',
num_spec_tokens=1)``, "Using FlashInfer for top-p & top-k sampling"):

- every step verifies one greedy draft token d (no draft probabilities) against the target logits after temperature and
  ``apply_top_k_top_p`` (``rejection_sampler.apply_sampling_constraints``); with 8 or more draft rows in the batch that is
  ``apply_top_k_top_p_triton`` (a pivot-based approximate top-k / top-p kernel, GPU tiles 8,192 / 4,096), else the PyTorch
  sort; acceptance ``p(d) >= u``; on rejection a Gumbel-max sample from p with d masked (``sample_recovered_tokens``);
- the bonus token (all drafts accepted) comes from FlashInfer's sampler (GPU only: not checked here).

Checks, on the pinned sources read from the assets volume (``research-20260930/tp-vllm-src``, version file checked):

1. ``apply_top_k_top_p_triton`` at the GPU tile sizes against ``apply_top_k_top_p_pytorch`` (exact sort): kept set and
   the total-variation distance of the filtered distributions, per row, on 248,320-wide rows of five shapes;
2. the full MTP-1 verify step (``rejection_sample`` and its kernels, Triton interpreter): the output-token frequency
   over many independent copies of one row against the exact filtered distribution (chi-square, TV), for drafts at
   rank 1, 2, 5 and outside the kept set;
3. our ``topk_first_impl`` (src/arc_infra/serving_kernels_runtime.py) against the same exact reference.

    uv run --extra modal modal run --detach scripts/modal_sampler_exact.py --run-id tp-sampler-exact-v1-20261012
"""

import json
import os
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
CPU, MEMORY_GIB, TIMEOUT = 8.0, 16, 5400
SRC = "/assets/research-20260930/tp-vllm-src/vllm"
VERSION = "0.1.dev20073+g8e685d198"

app = modal.App("arc-sampler-exact", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.9.0", "triton==3.5.0", "numpy<2", "scipy<1.15",
                      extra_index_url="https://download.pytorch.org/whl/cpu")
         .env({"TRITON_INTERPRET": "1"})
         .add_local_file(str(REPO / "src/arc_infra/serving_kernels_runtime.py"), "/root/serving_kernels_runtime.py",
                         copy=True)
         .add_local_file(__file__, "/root/modal_sampler_exact.py", copy=True))


def _stub_vllm():
    """Just enough of ``vllm`` for the two pinned modules (no CUDA, no package import)."""
    import contextlib
    import sys
    import types

    import triton
    import triton.language as tl

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    mod("vllm")
    mod("vllm.triton_utils", tl=tl, triton=triton, HAS_TRITON=True)
    mod("vllm.utils")
    mod("vllm.utils.gpu_sync_debug", gpu_sync_allowed=contextlib.nullcontext)
    mod("vllm.utils.math_utils", next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())
    mod("vllm.utils.platform_utils", num_compute_units=lambda index=None: 188)  # RTX PRO 6000 Blackwell SMs


ROW_LOOP = "    for row_id in tl.range(pid, BATCH_SIZE, num_programs):\n"


def _load_topk_triton(rows=None):
    """The pinned module; with ``rows`` the strided row loop becomes one program over rows 0..rows-1 (the Triton
    interpreter cannot index ``tl.range`` by program-id tensors). Each row's arithmetic is unchanged."""
    import sys
    import types

    code = Path(f"{SRC}/v1/sample/ops/topk_topp_triton.py").read_text()
    if rows is not None:
        if code.count(ROW_LOOP) != 1:
            raise RuntimeError("pinned Triton top-k/top-p kernel: row loop not found once")
        code = code.replace(ROW_LOOP, (f"    for row_index in tl.range(0, {int(rows)}, 1):\n"
                                       "        row_id = tl.full((), row_index, tl.int32)\n"))
    m = types.ModuleType("pinned_topk_topp_triton")
    exec(compile(code, "topk_topp_triton.py", "exec"), m.__dict__)
    sys.modules[m.__name__] = m
    return m


def _load_topk_pytorch():
    """``apply_top_k_top_p_pytorch`` and ``apply_top_k_only`` from the pinned file (pure torch functions)."""
    import ast

    import torch

    code = Path(f"{SRC}/v1/sample/ops/topk_topp_sampler.py").read_text()
    tree = ast.parse(code)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("apply_top_k_top_p_pytorch", "apply_top_k_only")]
    ns = {"torch": torch}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "topk_topp_sampler.py", "exec"), ns)
    return ns


def _load_rejection(apply_top_k_top_p):
    """The pinned rejection-sampling functions and kernels, with the served ``apply_top_k_top_p`` dispatch."""
    import ast

    import torch
    import triton
    import triton.language as tl

    code = Path(f"{SRC}/v1/sample/rejection_sampler.py").read_text()
    tree = ast.parse(code)
    names = {"rejection_sample", "apply_sampling_constraints", "expand_batch_to_tokens", "generate_uniform_probs",
             "sample_recovered_tokens", "rejection_greedy_sample_kernel", "rejection_random_sample_kernel",
             "expand_kernel", "sample_recovered_tokens_kernel"}
    keep = [n for n in tree.body if (isinstance(n, ast.ImportFrom) and n.module == "__future__") or (
        isinstance(n, ast.FunctionDef) and n.name in names) or (
        isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") in ("PLACEHOLDER_TOKEN_ID", "GREEDY_TEMPERATURE",
                                                                           "MAX_SPEC_LEN")) or (
        isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "MAX_SPEC_LEN" for t in n.targets))]
    ns = {"torch": torch, "triton": triton, "tl": tl, "apply_top_k_top_p": apply_top_k_top_p,
          "__annotations__": {}}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "rejection_sampler.py", "exec"), ns)
    found = names - set(ns)
    if found:
        raise RuntimeError(f"pinned rejection sampler lacks {sorted(found)}")
    return ns


def _rows(torch, vocab, seed):
    """Five row shapes (logits, before temperature) over the full vocabulary."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(vocab, generator=g) * 2.0  # a broad tail
    rows, kinds = [], []
    zipf = -1.6 * torch.log(torch.arange(1, 65, dtype=torch.float32))
    for kind in ("peaked", "zipf", "flat30", "ties_at_k", "two_way"):
        x = base.clone() + torch.randn(vocab, generator=g) * 0.3
        perm = torch.randperm(vocab, generator=g)
        if kind == "peaked":
            x[perm[0]] = x.max() + 6.0
        elif kind == "zipf":
            x[perm[:64]] = x.max() + 4.0 + zipf
        elif kind == "flat30":
            x[perm[:30]] = x.max() + 3.0 + torch.randn(30, generator=g) * 0.05
        elif kind == "ties_at_k":
            top = x.max() + 3.0
            x[perm[:18]] = top + torch.linspace(1.0, 0.1, 18)
            x[perm[18:24]] = top  # six tokens tied at ranks 19-24 (top-k 20 keeps all of them)
        elif kind == "two_way":
            x[perm[0]] = x.max() + 5.0
            x[perm[1]] = x[perm[0]] - 0.05
        rows.append(x)
        kinds.append(kind)
    return torch.stack(rows), kinds


@app.function(image=image, cpu=CPU, memory=MEMORY_GIB * 1024, timeout=TIMEOUT, volumes={"/assets": assets})
def check(rows_per_kind: int = 4, trials: int = 20000, small_vocab: int = 4096):
    import importlib.util
    import sys

    import numpy as np
    import torch
    from scipy.stats import chisquare

    t0 = time.time()
    version = Path(f"{SRC}/_version.py").read_text()
    if VERSION not in version:
        raise RuntimeError(f"pinned vLLM source is not {VERSION}")
    _stub_vllm()
    tri = _load_topk_triton()
    pyt = _load_topk_pytorch()
    spec = importlib.util.spec_from_file_location("serving_kernels_runtime_src", "/root/serving_kernels_runtime.py")
    out = {"vllm_version": VERSION, "temperature": 0.7, "top_k": 20, "top_p": 0.95}

    def exact(logits, k, p):
        return pyt["apply_top_k_top_p_pytorch"](logits.clone(), k, p)

    kernels = {}

    def triton_gpu_tiles(logits, k, p):
        # apply_top_k_top_p_triton picks CPU tiles on a CPU tensor; call the kernel with the GPU tiles (8,192 / 4,096).
        logits = logits.clone().contiguous()
        b, v = logits.shape
        if b not in kernels:
            kernels[b] = _load_topk_triton(rows=b)
        mod = kernels[b]
        buffer = logits.new_empty((1, v))
        cdf = logits.new_tensor(tri._NORMAL_CDF_TO_SIGMA_TABLE)
        pct = logits.new_tensor(tri._PERCENTILE_TO_STD_TABLE)
        mod._topk_topp_kernel[(1,)](
            logits, logits.stride(0), buffer, pct, cdf, k.to(torch.int32), p.to(torch.float32),
            BATCH_SIZE=b, MASK_VALUE=float("-inf"), VOCAB_SIZE=v, BLOCK_SIZE=8192, BLOCK_SIZE_TRUNC=4096,
            TOPK_ENABLED=True, TOPP_ENABLED=True)
        return logits

    def tv(a, b):
        pa, pb = a.softmax(-1), b.softmax(-1)
        return 0.5 * (pa - pb).abs().sum(-1)

    # 1 and 3: filters on full-width rows (8+ rows: the served dispatch takes the Triton kernel)
    vocab = 248_320
    all_rows, kinds = [], []
    for r in range(rows_per_kind):
        x, kk = _rows(torch, vocab, 1000 + r)
        all_rows.append(x)
        kinds += kk
    logits = torch.cat(all_rows) / 0.7
    b = logits.shape[0]
    k = torch.full((b,), 20, dtype=torch.long)
    p = torch.full((b,), 0.95, dtype=torch.float32)
    ref = exact(logits, k, p)
    t1 = time.time()
    got = triton_gpu_tiles(logits, k, p)
    t_tri = time.time() - t1
    mask_eq = (torch.isinf(ref) == torch.isinf(got)).all(-1)
    out["triton_vs_exact"] = {
        "rows": b, "seconds": round(t_tri, 1),
        "kept_set_equal": int(mask_eq.sum()), "tv_max": float(tv(ref, got).max()),
        "by_kind": {kind: {"equal": int(sum(bool(mask_eq[i]) for i in range(b) if kinds[i] == kind)),
                           "rows": kinds.count(kind),
                           "tv_max": float(max(float(tv(ref[i:i + 1], got[i:i + 1])) for i in range(b)
                                                if kinds[i] == kind)),
                           "kept_exact": [int((~torch.isinf(ref[i])).sum()) for i in range(b) if kinds[i] == kind],
                           "kept_triton": [int((~torch.isinf(got[i])).sum()) for i in range(b) if kinds[i] == kind]}
                    for kind in sorted(set(kinds))}}
    skr = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(skr)
        ours = skr.topk_first_impl(logits.clone(), k, p, 20)
        eq = (torch.isinf(ref) == torch.isinf(ours)).all(-1)
        out["topk_first_vs_exact"] = {"rows": b, "kept_set_equal": int(eq.sum()), "tv_max": float(tv(ref, ours).max())}
    except Exception as error:  # noqa: BLE001 - recorded, the other checks still run
        out["topk_first_vs_exact"] = {"error": repr(error)[:400]}

    # 2: the MTP-1 verify step, Monte Carlo over independent copies of one row (a narrower vocabulary: the kernels'
    # arithmetic does not depend on the width, and the interpreter's cost does)
    rej = _load_rejection(lambda lg, kk, pp: (triton_gpu_tiles(lg, kk, pp) if lg.shape[0] >= 8
                                               else exact(lg, kk, pp)))
    mc = []
    for kind_index, kind in enumerate(("zipf", "flat30", "peaked", "ties_at_k")):
        x, kk = _rows(torch, small_vocab, 7 + kind_index)
        row = x[kk.index(kind)]
        filt = exact((row / 0.7).unsqueeze(0), torch.tensor([20]), torch.tensor([0.95]))[0]
        target = filt.softmax(-1)
        order = torch.argsort(target, descending=True)
        kept = int((target > 0).sum())
        drafts = {"rank1": int(order[0]), "rank2": int(order[min(1, kept - 1)]),
                  "rank5": int(order[min(4, kept - 1)]), "outside": int(order[kept])}
        for label, draft in drafts.items():
            n = trials

            def meta_for(rows):
                meta = type("M", (), {})()
                meta.all_greedy, meta.all_random = False, True
                meta.temperature = torch.full((rows,), 0.7)
                meta.top_k = torch.full((rows,), 20, dtype=torch.long)
                meta.top_p = torch.full((rows,), 0.95)
                meta.generators = {}
                return meta

            # The pinned constraint step on 8 rows (the served dispatch: Triton at 8+ rows); rows are independent,
            # so the n copies reuse it (all 8 must agree).
            eight = rej["apply_sampling_constraints"](row.unsqueeze(0).repeat(8, 1).float(),
                                                      torch.arange(1, 9, dtype=torch.int32), meta_for(8))
            if not all(torch.equal(eight[0], eight[i]) for i in range(1, 8)):
                raise RuntimeError("the constraint step differs between identical rows")
            meta = meta_for(n)
            cu = torch.arange(1, n + 1, dtype=torch.int32)
            constrained = eight[:1].repeat(n, 1).contiguous()
            # The bonus token is FlashInfer's (not checked); a sentinel marks "all accepted".
            bonus = torch.full((n, 1), -7, dtype=torch.int32)
            t2 = time.time()
            res = rej["rejection_sample"](torch.full((n,), draft, dtype=torch.int32), [1] * n, 1, cu, None,
                                          constrained, bonus, meta)
            first = res[:, 0].long().numpy()
            counts = np.bincount(first, minlength=small_vocab).astype(float)
            expect = target.double().numpy() * n
            support = expect > 0
            outside = int(counts[~support].sum())
            inside = counts[support]
            stat = (chisquare(inside, expect[support] * inside.sum() / expect[support].sum())
                    if support.sum() > 1 and inside.sum() > 0 else None)
            mc.append({"kind": kind, "draft": label, "kept": kept, "constrained_equals_exact": bool(torch.equal(
                           torch.isinf(eight[0]), torch.isinf(filt))), "trials": n, "seconds": round(time.time() - t2, 1),
                       "accept_rate": float((first == draft).mean()), "p_draft": float(target[draft]),
                       "tokens_outside_kept": outside, "tv": float(0.5 * np.abs(counts / n - expect / n).sum()),
                       "chi2_p": None if stat is None else float(stat.pvalue)})
    out["mtp1_verify_mc"] = mc
    out["seconds"] = round(time.time() - t0, 1)
    return out


@app.local_entrypoint()
def main(run_id: str, rows_per_kind: int = 4, trials: int = 20000):
    record = REPO / "docs/validation-20261012" / f"{run_id}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    result = check.remote(rows_per_kind=rows_per_kind, trials=trials)
    result["run_id"] = run_id
    record.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1)[:6000])
