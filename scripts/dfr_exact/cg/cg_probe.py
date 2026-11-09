"""Exact CUDA graphs over mixed steps: the offline bit-identity probe (throughput arm, 20 Oct; owner item 10).

One served CG bundle (capture sizes to 4,096) answers two questions on its own weights and kernels:
1. A/B on real steps: ARC_CG_EAGER_FLAG=<path> makes every step over ARC_CG_EAGER_ABOVE tokens (104, v8.5p's graph
   bound) run eager while the file exists, exactly as v8.5p serves it; otherwise the step replays the piecewise graph
   padded to the next capture size. The trial (arena/cg_exact.py) sends each recorded context prefix three times at
   concurrency 1 with the prefix cache reset (eager, graph, eager again) and compares the first generated token's
   top-20 logprobs bit for bit: eager vs eager is the determinism control, eager vs graph the exactness test.
2. Op classes: the worker RPC ``arc_cg_probe`` runs every distinct dense projection (its served quant method) and a
   few MoE blocks (router + routed + shared experts) at every M in [m_min, m_max] on fixed random rows and records
   where the first 32 output rows change bits (a kernel-choice boundary), plus repeat (determinism) and
   other-rows (padding/routing dependence) checks. Capture sizes on the class edges make padding exact.
"""

RUNTIME_CODE = r'''

# ----------------------------------------------------------------------------- exact-CG probe (throughput arm, 20 Oct)
import os as _arc_cg_os

CG_EAGER_FLAG = _arc_cg_os.environ.get("ARC_CG_EAGER_FLAG") or ""
CG_EAGER_ABOVE = int(_arc_cg_os.environ.get("ARC_CG_EAGER_ABOVE", "104"))
_CG = {"installed": False, "eager_steps": 0, "graph_steps": 0}


def _install_cg_probe():
    if _CG["installed"]:
        return
    import vllm.v1.worker.gpu.model_runner as _mr
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    orig = _mr.dispatch_cg_and_sync_dp

    def dispatch(cudagraph_manager, num_reqs, num_tokens, *args, need_eager=False, **kwargs):
        if num_tokens > CG_EAGER_ABOVE:
            if CG_EAGER_FLAG and _arc_cg_os.path.exists(CG_EAGER_FLAG):
                need_eager = True
                _CG["eager_steps"] += 1
            else:
                _CG["graph_steps"] += 1
        return orig(cudagraph_manager, num_reqs, num_tokens, *args, need_eager=need_eager, **kwargs)

    _mr.dispatch_cg_and_sync_dp = dispatch
    WorkerWrapperBase.arc_cg_probe = arc_cg_probe
    if _arc_cg_os.environ.get("ARC_HUMMING_BI") == "1":
        # Humming's own batch-invariant configs, without the global VLLM_BATCH_INVARIANT (refused by GDN_ATTN).
        from vllm.model_executor.layers.fused_moe.experts import fused_humming_moe as _fh

        cls = next(c for c in vars(_fh).values() if isinstance(c, type) and hasattr(c, "init_humming_moe"))
        orig_init = cls.init_humming_moe

        def init_humming_moe(self):
            had = _arc_cg_os.environ.get("VLLM_BATCH_INVARIANT")
            _arc_cg_os.environ["VLLM_BATCH_INVARIANT"] = "1"
            try:
                return orig_init(self)
            finally:
                if had is None:
                    _arc_cg_os.environ.pop("VLLM_BATCH_INVARIANT", None)
                else:
                    _arc_cg_os.environ["VLLM_BATCH_INVARIANT"] = had
                _CG.setdefault("humming_configs", set()).add(getattr(self, "compute_config_str", ""))

        cls.init_humming_moe = init_humming_moe
    if _arc_cg_os.environ.get("ARC_QSA_SORT") == "1":
        # The indexer's radix top-k returns the selected blocks in a run-dependent order; sort them (invalid -1 last)
        # so the sparse attention sums in a fixed order.
        from vllm.models.qwen3_8_flash_next.nvidia.ops import qsa as _qsa

        orig_expand = _qsa.expand_qsa_block_indices_cuda

        def expand_sorted(blocks, *args, **kwargs):
            big = torch.iinfo(blocks.dtype).max
            key = torch.where(blocks < 0, torch.full_like(blocks, big), blocks)
            key = key.sort(dim=1).values
            blocks.copy_(torch.where(key == big, torch.full_like(key, -1), key))
            _CG["qsa_sorted_calls"] = _CG.get("qsa_sorted_calls", 0) + 1
            return orig_expand(blocks, *args, **kwargs)

        _qsa.expand_qsa_block_indices_cuda = expand_sorted
    _CG["installed"] = True


def _cg_classes(fn, x, lo, hi, sig_rows=32, every=128):
    """Bit boundaries of fn's first sig_rows output rows over M in [lo, hi] (x: [hi, K] fixed rows)."""
    prev, edges, nondet, other = None, [], [], []
    gen = torch.Generator(device=x.device).manual_seed(1)
    for m in range(lo, hi + 1):
        y = fn(x[:m])[:sig_rows].clone()
        if prev is not None and not torch.equal(y, prev):
            edges.append(m)
        prev = y
        if m % every == 0 or m in (lo, hi):
            if not torch.equal(fn(x[:m])[:sig_rows], y):
                nondet.append(m)
            alt = x[:m].clone()
            alt[sig_rows:] = torch.randn(alt[sig_rows:].shape, generator=gen, device=x.device, dtype=x.dtype)
            if m > sig_rows and not torch.equal(fn(alt)[:sig_rows], y):
                other.append(m)
    return {"edges": edges, "nondeterministic": nondet, "other_rows_change": other}


def arc_cg_probe(self, spec_json):
    """Worker RPC: op classes of the served dense projections and MoE blocks (see the module docstring)."""
    import json as _json
    import time as _time

    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.linear import LinearBase

    spec = _json.loads(spec_json)
    lo, hi = int(spec.get("m_min", 33)), int(spec.get("m_max", 4096))
    runner = self.worker.model_runner
    model = runner.model if hasattr(runner, "model") else runner.get_model()
    out = {"ok": True, "dense": [], "moe": [], "errors": [], "timing_ms": {},
           "steps": {k: (sorted(v) if isinstance(v, set) else v) for k, v in _CG.items()}}
    t0 = _time.time()
    gen = torch.Generator(device="cuda").manual_seed(0)
    with torch.inference_mode():
        seen = set()
        for name, mod in model.named_modules():
            w = getattr(mod, "weight", None)
            qm = getattr(mod, "quant_method", None)
            if not isinstance(mod, LinearBase) or w is None or w.dim() != 2 or qm is None:
                continue
            k = int(getattr(mod, "input_size_per_partition", w.shape[1]))
            key = (int(w.shape[0]), k, type(qm).__name__)
            if key in seen:
                continue
            seen.add(key)
            try:
                x = torch.randn((hi, k), generator=gen, device=w.device, dtype=torch.bfloat16)
                res = _cg_classes(lambda xm: qm.apply(mod, xm), x, lo, hi)
                out["dense"].append({"name": name, "shape": list(w.shape), "k": k, "method": key[2], **res})
            except Exception as error:  # noqa: BLE001
                out["errors"].append({"name": name, "error": repr(error)[:300]})
        layers = next((m for n, m in model.named_modules() if n.endswith("model.layers")
                       and isinstance(m, torch.nn.ModuleList)), [])
        cfg = runner.vllm_config
        for li in spec.get("moe_layers", [0, 23, 47]):
            try:
                block = layers[li].mlp
                hidden = int(cfg.model_config.hf_text_config.hidden_size)
                x = torch.randn((hi, hidden), generator=gen, device="cuda", dtype=torch.bfloat16)

                def run(xm, _block=block):
                    with set_forward_context(None, cfg, num_tokens=xm.shape[0]):
                        return _block(xm)

                out["moe"].append({"layer": li, **_cg_classes(run, x, lo, hi)})
                for m in (64, 256, 1024, 4096):
                    if m <= hi:
                        torch.cuda.synchronize()
                        t1 = _time.time()
                        for _ in range(20):
                            run(x[:m])
                        torch.cuda.synchronize()
                        out["timing_ms"][f"moe{li}@{m}"] = round((_time.time() - t1) / 20 * 1000, 3)
            except Exception as error:  # noqa: BLE001
                out["errors"].append({"moe_layer": li, "error": repr(error)[:300]})
    out["seconds"] = round(_time.time() - t0, 1)
    return out


_arc_cg_enable = enable


def enable(module, dtype):  # noqa: F811 - wraps the overlay's entry point to install the probe in the worker
    _install_cg_probe()
    return _arc_cg_enable(module, dtype)
'''

TRIAL_CODE = r'''"""Exact-CG A/B on one server (throughput arm, 20 Oct; scripts/dfr_exact/cg/cg_probe.py): each recorded context
prefix three times at concurrency 1 with the prefix cache reset, eager (flag file present), graph (flag absent),
eager again; the first generated token's top-20 logprobs compared bit for bit. Then the worker's op-class probe."""

import asyncio
import json
import time
from pathlib import Path

import httpx

from arc_infra.serving import MODEL

INPUTS = "configs/cg-exact/prompts.json"


async def _reset(http, root):
    for _ in range(30):
        r = await http.post(root + "/reset_prefix_cache", params={"reset_external": "true"})
        if r.status_code == 200 and r.json().get("success") is True:
            return True
        await asyncio.sleep(0.5)
    raise RuntimeError("prefix cache reset failed")


async def _one(http, base, ids):
    r = await http.post(base + "/completions", json={"model": MODEL, "prompt": ids, "max_tokens": 1,
                                                     "temperature": 0.0, "logprobs": 20})
    if r.status_code != 200:
        raise RuntimeError(f"completions {r.status_code}: {r.text[:300]}")
    choice = r.json()["choices"][0]
    lp = (choice.get("logprobs") or {})
    top = (lp.get("top_logprobs") or [{}])[0] or {}
    return {"text": choice.get("text"), "top": {k: float(v) for k, v in top.items()}}


async def run_cg_exact(study, trial, *, source, output, base_url, deadline_epoch, log=print):
    cfg = study["cg_exact"]
    flag = Path(cfg["flag"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prompts = json.loads((Path(source) / INPUTS).read_text())
    report = {"phase": "cg-exact", "status": "error", "errors": [], "rows": [], "probe": None}
    path = output / "cg-exact.json"

    def save():
        path.write_text(json.dumps(report, indent=1))

    base = base_url.rstrip("/")
    root = base.removesuffix("/v1")
    async with httpx.AsyncClient(timeout=1800, trust_env=False) as http:
        try:
            for i, item in enumerate(prompts):
                if time.time() > deadline_epoch - cfg.get("probe_reserve_seconds", 900):
                    report["errors"].append(f"deadline after {i} prompts")
                    break
                row = {"id": item["id"], "tokens": len(item["ids"])}
                for mode in ("eager", "graph", "eager2"):
                    if mode == "graph":
                        flag.unlink(missing_ok=True)
                    else:
                        flag.write_text("eager\n")
                    await _reset(http, root)
                    row[mode] = await _one(http, base, item["ids"])
                flag.unlink(missing_ok=True)
                e, g, e2 = row["eager"], row["graph"], row["eager2"]
                row["det"] = e == e2
                row["exact"] = e == g
                common = set(e["top"]) & set(g["top"])
                row["max_abs"] = max((abs(e["top"][t] - g["top"][t]) for t in common), default=None)
                report["rows"].append(row)
                if i % 10 == 0:
                    log(json.dumps({"event": "expert_quant_cg_exact", "done": i + 1, "of": len(prompts),
                                    "exact": sum(r["exact"] for r in report["rows"]),
                                    "det": sum(r["det"] for r in report["rows"])}), flush=True)
                    save()
            rows = report["rows"]
            report["summary"] = {"prompts": len(rows), "det": sum(r["det"] for r in rows),
                                 "exact": sum(r["exact"] for r in rows),
                                 "exact_given_det": sum(r["exact"] for r in rows if r["det"]),
                                 "det_rows": sum(1 for r in rows if r["det"])}
            save()
            r = await http.post(root + "/collective_rpc", json={
                "method": "arc_cg_probe", "args": [json.dumps(cfg.get("probe") or {})], "timeout": 3000})
            report["probe"] = r.json()["results"][0] if r.status_code == 200 else {"error": r.text[:500]}
            report["status"] = "complete" if not report["errors"] else "complete_with_errors"
        except Exception as error:  # noqa: BLE001
            report["errors"].append(repr(error)[:500])
        finally:
            flag.unlink(missing_ok=True)
            save()
    log(json.dumps({"event": "expert_quant_cg_exact", "summary": report.get("summary"),
                    "probe_errors": (report.get("probe") or {}).get("errors")}), flush=True)
    return report
'''

EQ_OLD = '''    settings = study["expert_quant"]
    phase = trial.get("phase")
'''
EQ_NEW = '''    if trial.get("phase") == "cg-exact":  # exact-CG probe (throughput arm, 20 Oct; arena.cg_exact)
        from .cg_exact import run_cg_exact

        return await run_cg_exact(study, trial, source=source, output=output, base_url=base_url,
                                  deadline_epoch=deadline_epoch, log=log)
    settings = study["expert_quant"]
    phase = trial.get("phase")
'''


def prompts_from(sequences, lengths_per_seq=8, seed=0, lo=105, hi=4096):
    """Context prefixes covering one-step M values: per sequence one control length in [33, 104] (decode-size graph in
    both modes), lengths_per_seq - 2 one-chunk lengths spread over [lo, hi], one two-chunk length (4,096 + r)."""
    import random

    rng = random.Random(seed)
    out = []
    n = len(sequences)
    span = hi - lo + 1
    for si, seq in enumerate(sequences):
        ids = seq["ids"]
        lens = [rng.randint(33, 104)]
        for j in range(lengths_per_seq - 2):
            slot = (si * (lengths_per_seq - 2) + j) % (n * (lengths_per_seq - 2))
            a = lo + span * slot // (n * (lengths_per_seq - 2))
            b = lo + span * (slot + 1) // (n * (lengths_per_seq - 2)) - 1
            lens.append(rng.randint(a, max(a, b)))
        lens.append(4096 + rng.randint(lo, hi - 1))
        for L in lens:
            out.append({"id": f"s{si:02d}-L{L}", "seq": si, "ids": ids[:L]})
    return out
