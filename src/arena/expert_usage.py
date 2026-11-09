"""Expert-usage measurement (28 September; docs/validation-20260927/expert-usage.md).

An ``expert-usage`` trial runs on a server started with ``server.expert_probe`` (``arc_infra.expert_probe_patch``):
every MoE layer's router logits pass through a probe that records routing statistics and applies a pruning
mask, both switched through ``/collective_rpc`` (``arc_expert_probe``). Inputs are fixed token sequences in
the bundle's ``configs/expert-usage`` (``scripts/expert_usage_inputs.py``): recorded arena requests, split
by game family into a usage set and a held-out set.

Passes (teacher-forced prefill: ``/v1/completions`` on token ids, ``max_tokens`` 1):

1. **usage**: record on, no mask; ``prompt_logprobs`` 0 (so every position is computed, never read from the
   prefix cache, and the recorded token's log-probability comes back). Dump: per-(layer, expert) routing
   counts, gate-weight mass and softmax mass.
2. **reference**: held-out set, no mask, ``prompt_logprobs`` 20, record on (held-out routing statistics).
3. **repeat**: the same at another concurrency: the full model's own run-to-run noise (batch composition).
4. **masks**: per layer, the least-used experts by usage-set gate-weight mass are masked (router top-k and
   weights renormalize over the rest) at each fraction; held-out passes as the reference.

Each pass after the reference is compared with it position by position: KL(reference || pass) on the
partition {tokens in both top-20 lists} + {rest} (exact for the partition, a lower bound on the full KL), top-1
agreement, the recorded token's NLL, and the share of positions whose top-1 leaves the reference top-20; over
all positions and over the assistant-turn positions. Passes run in priority order and are skipped when the
remaining time would not cover one more (estimated from the reference pass).
"""

import asyncio
import json
import re
import time
from pathlib import Path

import httpx
import numpy as np

from arc_infra.serving import MODEL

from .isolation import isolate_trial
from .kernel_gate import bounded, token_id

INPUTS = "configs/expert-usage"
EXPERTS = 512
TOP_K = 10
SETTINGS = {
    "fractions": [0.25, 0.125, 0.5, 0.375],   # run order (priority)
    "rank_by": "weight",                      # usage-set gate-weight mass (REAP-style proxy without norms)
    "usage_concurrency": 3, "usage_prompt_logprobs": 0,
    "heldout_prompt_logprobs": 20,
    "reference_concurrency": 3, "repeat_concurrency": 2, "mask_concurrency": 3,
    "coverage": [0.9, 0.99, 0.999],
}


def load_inputs(source):
    root = Path(source) / INPUTS
    return json.loads((root / "usage.json").read_text()), json.loads((root / "heldout.json").read_text())


# ----------------------------------------------------------------------------- statistics
def coverage(mass, fractions=SETTINGS["coverage"]):
    """Experts needed per layer to carry each fraction of a layer's mass (rows: layers)."""
    mass = np.asarray(mass, dtype=np.float64)
    ordered = -np.sort(-mass, axis=1)
    cum = np.cumsum(ordered, axis=1) / np.maximum(ordered.sum(axis=1, keepdims=True), 1e-300)
    return {str(f): [int(np.searchsorted(row, f - 1e-12) + 1) for row in cum] for f in fractions}


def masks(weight, count, fractions, layers):
    """Pruned expert ids per layer: the least gate-weight mass first (routing count breaks ties)."""
    weight, count = np.asarray(weight, dtype=np.float64), np.asarray(count, dtype=np.float64)
    order = np.stack([np.lexsort((count[i], weight[i])) for i in range(weight.shape[0])])
    result = {}
    for fraction in fractions:
        m = int(round(fraction * EXPERTS))
        result[str(fraction)] = {str(layer): sorted(int(e) for e in order[i, :m]) for i, layer in enumerate(layers)}
    return result


def mass_on(pruned, stats):
    """Share of the held-out routing (counts, gate weight) that a mask's pruned experts carried unmasked."""
    layers = stats["dump_layers"]
    out = {}
    for key in ("count", "weight"):
        values = np.asarray(stats[key], dtype=np.float64)
        per_layer = []
        for i, layer in enumerate(layers):
            ids = pruned.get(str(layer), [])
            total = values[i].sum()
            per_layer.append(float(values[i, ids].sum() / total) if total else 0.0)
        out[key] = {"mean": float(np.mean(per_layer)), "max": float(np.max(per_layer)), "per_layer": per_layer}
    return out


# ----------------------------------------------------------------------------- teacher forcing
def parse_positions(entries, ids, top):
    """Arrays from a prompt_logprobs list: recorded-token logprob [n-1] and the top-``top`` ids/logprobs."""
    n = len(ids) - 1
    if len(entries) != len(ids):
        raise RuntimeError(f"prompt_logprobs length {len(entries)} != {len(ids)}")
    actual = np.empty(n, dtype=np.float32)
    top_ids = np.full((n, max(top, 1)), -1, dtype=np.int32)
    top_lps = np.full((n, max(top, 1)), -np.inf, dtype=np.float32)
    for i, entry in enumerate(entries[1:]):
        target = ids[i + 1]
        ranked = sorted(((token_id(t), v["logprob"]) for t, v in (entry or {}).items()),
                        key=lambda row: row[1], reverse=True)
        found = None
        for t, lp in ranked:
            if t == target:
                found = lp
                break
        if found is None:
            raise RuntimeError(f"recorded token missing at position {i + 1}")
        actual[i] = found
        if top:
            # The entry holds the top-k plus the recorded token when it ranks lower: the k highest are the top-k.
            for j, (t, lp) in enumerate(ranked[:top]):
                top_ids[i, j], top_lps[i, j] = t, lp
    return actual, top_ids, top_lps


def assistant_mask(n, spans):
    """Positions (predicting token i+1) whose target lies in an assistant turn."""
    mask = np.zeros(n, dtype=bool)
    for start, end in spans:
        mask[max(start - 1, 0):max(end - 1, 0)] = True
    return mask


def compare(ref, cand, chunk=8192):
    """Per-position KL(ref || cand) on the common-top-k partition, top-1 agreement, top-1 outside ref top-k."""
    (ref_ids, ref_lps), (cand_ids, cand_lps) = ref, cand
    n = ref_ids.shape[0]
    kl = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        a_ids, a_lp = ref_ids[s:s + chunk], ref_lps[s:s + chunk].astype(np.float64)
        b_ids, b_lp = cand_ids[s:s + chunk], cand_lps[s:s + chunk].astype(np.float64)
        same = (a_ids[:, :, None] == b_ids[:, None, :]) & (a_ids[:, :, None] >= 0)
        in_b = same.any(axis=2)
        b_at_a = np.where(same, b_lp[:, None, :], 0.0).sum(axis=2)
        pa = np.where(in_b, np.exp(a_lp), 0.0)
        pb = np.where(in_b, np.exp(b_at_a), 0.0)
        body = np.where(in_b, pa * (a_lp - np.where(in_b, b_at_a, 0.0)), 0.0).sum(axis=1)
        rest_a = np.clip(1.0 - pa.sum(axis=1), 1e-12, None)
        rest_b = np.clip(1.0 - pb.sum(axis=1), 1e-12, None)
        kl[s:s + chunk] = np.maximum(body + rest_a * np.log(rest_a / rest_b), 0.0)
    agree = ref_ids[:, 0] == cand_ids[:, 0]
    outside = ~(cand_ids[:, :1] == ref_ids).any(axis=1)
    return kl, agree, outside


def summarize(kl, agree, outside, nll_ref, nll_cand):
    if not len(kl):
        return {"positions": 0}
    quantiles = np.quantile(kl, [0.5, 0.9, 0.99])
    return {"positions": int(len(kl)), "mean_kl": float(kl.mean()), "p50_kl": float(quantiles[0]),
            "p90_kl": float(quantiles[1]), "p99_kl": float(quantiles[2]), "max_kl": float(kl.max()),
            "top1_agreement": float(agree.mean()), "top1_outside_ref_top20": float(outside.mean()),
            "nll_ref": float(nll_ref.mean()), "nll": float(nll_cand.mean()),
            "delta_nll": float((nll_cand - nll_ref).mean())}


class Probe:
    def __init__(self, base_url, model, transport=None):
        self.base_url = base_url.rstrip("/")
        self.root = self.base_url.removesuffix("/v1")
        self.model = model
        self.http = httpx.AsyncClient(timeout=1800, trust_env=False, transport=transport,
                                      limits=httpx.Limits(max_connections=16))

    async def close(self):
        await self.http.aclose()

    async def rpc(self, **command):
        response = await self.http.post(self.root + "/collective_rpc", json={
            "method": "arc_expert_probe", "args": [json.dumps(command)], "timeout": 120})
        if response.status_code != 200:
            raise RuntimeError(f"collective_rpc {response.status_code}: {response.text[:500]}")
        reply = response.json()["results"][0]
        if not isinstance(reply, dict) or not reply.get("ok"):
            raise RuntimeError(f"expert probe {command.get('op')}: {str(reply)[:500]}")
        return reply

    async def prefill(self, ids, prompt_logprobs):
        response = await self.http.post(self.base_url + "/completions", json={
            "model": self.model, "prompt": ids, "max_tokens": 1, "temperature": 0.0,
            "prompt_logprobs": prompt_logprobs, "return_tokens_as_token_ids": True})
        if response.status_code != 200:
            raise RuntimeError(f"completions {response.status_code}: {response.text[:300]}")
        return response.json()["choices"][0].get("prompt_logprobs") or []


async def run_pass(probe, rows, *, concurrency, prompt_logprobs, keep):
    """One teacher-forced pass; ``keep(row, actual, top_ids, top_lps)`` consumes each sequence's arrays."""
    started = time.monotonic()

    async def work(row):
        entries = await probe.prefill(row["ids"], prompt_logprobs)
        arrays = await asyncio.to_thread(parse_positions, entries, row["ids"], prompt_logprobs)
        del entries
        keep(row, *arrays)
        return {"index": row["index"], "tokens": len(row["ids"])}

    done = await bounded(rows, concurrency, work)
    failed = [r for r in done if "error" in r]
    seconds = time.monotonic() - started
    tokens = sum(r["tokens"] for r in done if "error" not in r)
    return {"seconds": round(seconds, 1), "tokens": tokens, "tokens_per_second": round(tokens / max(seconds, 1e-9)),
            "failed": failed, "concurrency": concurrency}


async def server_check(study, trial, *, base_url, output, log=print, transport=None):
    """A server-start diagnostic arm: the server reached this trial (it started), then one prefill with
    prompt_logprobs, one short greedy decode and, when the probe is loaded, its status."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    probe = Probe(base_url, study.get("served_model", MODEL), transport=transport)
    result = {"trial": trial["name"], "server_env": trial.get("server_env") or {}, "errors": []}
    try:
        started = time.monotonic()
        entries = await probe.prefill(list(range(1000, 1600)), 1)
        result["prefill"] = {"positions": len(entries), "seconds": round(time.monotonic() - started, 2)}
        started = time.monotonic()
        response = await probe.http.post(probe.base_url + "/completions", json={
            "model": probe.model, "prompt": "The capital of France is", "max_tokens": 32, "temperature": 0.0})
        response.raise_for_status()
        result["decode"] = {"text": response.json()["choices"][0]["text"][:200],
                            "seconds": round(time.monotonic() - started, 2)}
        try:
            status = await probe.rpc(op="status")
            result["probe"] = {"layers": len(status["layers"]), "calls": status["calls"]}
        except Exception as exc:  # noqa: BLE001 - the no-probe arm has no RPC method
            result["probe"] = {"loaded": False, "detail": str(exc)[:200]}
    except Exception as exc:  # noqa: BLE001 - recorded
        result["errors"].append(f"{type(exc).__name__}: {str(exc)[:500]}")
    finally:
        await probe.close()
    result["status"] = "complete" if not result["errors"] else "error"
    (output / "server-check.json").write_text(json.dumps(result, indent=2))
    log(json.dumps({"event": "server_check", **{k: result.get(k) for k in ("trial", "status", "prefill", "decode",
                                                                             "probe", "errors")}}), flush=True)
    return result


async def compare_pass(probe, heldout, reference, concurrency, top):
    """One held-out pass compared position by position with ``reference`` ({index: (actual, ids, lps, mask)})."""
    collected = {"all": [], "assistant": []}
    per_sequence = []

    def keep(row, actual, ids, lps):
        ref_actual, ref_ids, ref_lps, amask = reference[row["index"]]
        kl, agree, outside = compare((ref_ids, ref_lps), (ids, lps))
        parts = (kl, agree, outside, -ref_actual, -actual)
        collected["all"].append(parts)
        collected["assistant"].append(tuple(p[amask] for p in parts))
        per_sequence.append({"index": row["index"], "positions": int(len(kl)),
                             "assistant_positions": int(amask.sum()),
                             "kl_sum": float(kl.sum()), "assistant_kl_sum": float(kl[amask].sum()),
                             "nll_ref_sum": float(-ref_actual.sum()), "nll_sum": float(-actual.sum()),
                             "assistant_nll_ref_sum": float(-ref_actual[amask].sum()),
                             "assistant_nll_sum": float(-actual[amask].sum()),
                             "agree": int(agree.sum()), "assistant_agree": int(agree[amask].sum())})

    timing = await run_pass(probe, [r for r in heldout if r["index"] in reference], concurrency=concurrency,
                            prompt_logprobs=top, keep=keep)
    summary = {}
    for scope, parts in collected.items():
        if parts:
            summary[scope] = summarize(*(np.concatenate([p[i] for p in parts]) for i in range(5)))
    return timing, summary, sorted(per_sequence, key=lambda r: r["index"])


SERVER_EVIDENCE = (("model_loading_gib", re.compile(r"Model loading took ([0-9.]+) GiB")),
                   ("kv_cache_tokens", re.compile(r"GPU KV cache size: ([0-9,]+) tokens")),
                   ("marlin", re.compile(r"Using 'MARLIN' NvFp4 MoE backend")),
                   ("expert_prune", re.compile(r"ARC_EXPERT_PRUNE (\{.*\})")))


def server_evidence(server_dir):
    """What the served log says about the model's size, the KV pool, the MoE kernel and the pruning overlay."""
    path = Path(server_dir) / "vllm-openai-server.log"
    found = {"model_loading_gib": None, "kv_cache_tokens": None, "marlin": False, "expert_prune": []}
    if not path.is_file():
        return found
    for line in path.read_text(errors="replace").splitlines():
        for key, pattern in SERVER_EVIDENCE:
            match = pattern.search(line)
            if not match:
                continue
            if key == "model_loading_gib":
                found[key] = float(match.group(1))
            elif key == "kv_cache_tokens":
                found[key] = int(match.group(1).replace(",", ""))
            elif key == "marlin":
                found[key] = True
            else:
                found[key].append(json.loads(match.group(1)))
    return found


async def run_prune_check(study, trial, *, source, output, server_dir, base_url, deadline_epoch, log=print,
                          transport=None):
    """GPU check of the load-time pruned hero (``prune-quality`` trials; nothing masked at run time).

    ``role: reference`` (the unpruned server): the held-out pass (top-20) saved for the candidates, then the repeat at
    another concurrency (the noise floor). ``role: candidate`` (a pruned server): the same pass against the reference
    trial's file. Every trial records the server's own evidence (model size, KV pool, MoE kernel, pruning log)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _, heldout = load_inputs(source)
    probe = Probe(base_url, study.get("served_model", MODEL), transport=transport)
    top = SETTINGS["heldout_prompt_logprobs"]
    result = {"trial": trial["name"], "role": trial.get("role"), "server_env": trial.get("server_env") or {},
              "errors": [], "passes": {}}
    path = output / "prune-quality.json"

    def event(name, **extra):
        log(json.dumps({"event": f"prune_quality_{name}", "trial": trial["name"], **extra}), flush=True)

    try:
        if trial.get("role") not in ("reference", "candidate"):
            raise ValueError("a prune-quality trial is a reference or a candidate")
        await isolate_trial(base_url)
        result["server"] = server_evidence(server_dir)
        event("server", **result["server"])
        if trial["role"] == "reference":
            reference = {}

            def keep(row, actual, ids, lps):
                reference[row["index"]] = (actual, ids, lps, assistant_mask(len(actual), row["assistant_spans"]))

            timing = await run_pass(probe, heldout, concurrency=SETTINGS["reference_concurrency"],
                                    prompt_logprobs=top, keep=keep)
            result["passes"]["reference"] = {"timing": timing}
            np.savez(output / "reference.npz", **{f"{i}_{k}": v for i, arrays in reference.items()
                                                   for k, v in zip(("actual", "ids", "lps", "mask"), arrays)})
            event("reference", seconds=timing["seconds"], failed=len(timing["failed"]))
            if time.time() + 1.25 * timing["seconds"] + 45 < deadline_epoch:
                timing, summary, per_sequence = await compare_pass(probe, heldout, reference,
                                                                   SETTINGS["repeat_concurrency"], top)
                result["passes"]["repeat"] = {"timing": timing, "summary": summary, "per_sequence": per_sequence}
                event("pass", pass_name="repeat", assistant=summary.get("assistant"))
        else:
            stored = np.load(output.parent / trial["reference"] / "reference.npz")
            reference = {}
            for key in stored.files:
                index, kind = key.split("_", 1)
                reference.setdefault(int(index), {})[kind] = stored[key]
            reference = {i: (v["actual"], v["ids"], v["lps"], v["mask"]) for i, v in reference.items()}
            timing, summary, per_sequence = await compare_pass(probe, heldout, reference,
                                                               SETTINGS["mask_concurrency"], top)
            result["passes"]["candidate"] = {"timing": timing, "summary": summary, "per_sequence": per_sequence}
            event("pass", pass_name=trial["name"], assistant=summary.get("assistant"))
        try:  # a greedy sanity answer (recorded, never fatal)
            response = await probe.http.post(probe.base_url + "/completions", json={
                "model": probe.model, "prompt": "What is 17 * 23? Reply with the number only.", "max_tokens": 16,
                "temperature": 0.0})
            result["sanity"] = response.json()["choices"][0]["text"][:100]
        except Exception as exc:  # noqa: BLE001
            result["sanity"] = f"failed: {type(exc).__name__}"
    except Exception as exc:  # noqa: BLE001 - the partial result is kept
        result["errors"].append(f"{type(exc).__name__}: {str(exc)[:500]}")
        event("error", error=result["errors"][-1])
    finally:
        await probe.close()
    result["status"] = "complete" if result["passes"] and not result["errors"] else "error"
    path.write_text(json.dumps(result))
    return result


async def run_expert_usage(study, trial, *, source, output, base_url, deadline_epoch, log=print, transport=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    usage, heldout = load_inputs(source)
    fractions = (study.get("expert_usage") or {}).get("fractions")
    if fractions:  # phase 5 (4 October): the study's mask levels instead of the defaults
        SETTINGS["fractions"] = [float(f) for f in fractions]
    probe = Probe(base_url, study.get("served_model", MODEL), transport=transport)
    result = {"trial": trial["name"], "settings": dict(SETTINGS), "server": study["server"],
              "inputs": study.get("expert_usage", {}).get("inputs"), "passes": {}, "errors": [], "skipped": []}
    path = output / "expert-usage.json"

    def save():
        path.write_text(json.dumps(result))

    def event(name, **extra):
        log(json.dumps({"event": f"expert_usage_{name}", "trial": trial["name"], **extra}), flush=True)

    try:
        await isolate_trial(base_url)
        status = await probe.rpc(op="status")
        result["probe_status"] = {k: v for k, v in status.items() if k != "tokens_per_layer"}
        layers = status["layers"]
        event("probe", layers=len(layers), calls=status["calls"])
        if len(layers) < 2:
            raise RuntimeError(f"expert probe saw {len(layers)} MoE layers")

        # 1. usage set: routing statistics.
        await probe.rpc(op="mask", pruned={}, label=None)
        await probe.rpc(op="reset")
        await probe.rpc(op="record", on=True)
        usage_nll = {"sum": 0.0, "count": 0, "assistant_sum": 0.0, "assistant_count": 0}

        def keep_usage(row, actual, _ids, _lps):
            mask = assistant_mask(len(actual), row["assistant_spans"])
            usage_nll["sum"] += float(-actual.sum())
            usage_nll["count"] += int(len(actual))
            usage_nll["assistant_sum"] += float(-actual[mask].sum())
            usage_nll["assistant_count"] += int(mask.sum())

        options = study.get("expert_usage") or {}
        if options.get("per_family"):
            # Balanced refit: one pass per family with a dump each, so every family's routing is
            # kept separately; the usage statistics are their sum (equal tokens per family by construction).
            families = sorted({r["family"] for r in usage})
            result["usage_per_family"] = {}
            total, timing = None, {"seconds": 0.0, "tokens": 0, "failed": [], "concurrency":
                                   SETTINGS["usage_concurrency"]}
            for fam in families:
                await probe.rpc(op="reset")
                await probe.rpc(op="record", on=True)
                part = await run_pass(probe, [r for r in usage if r["family"] == fam],
                                      concurrency=SETTINGS["usage_concurrency"],
                                      prompt_logprobs=SETTINGS["usage_prompt_logprobs"], keep=keep_usage)
                await probe.rpc(op="record", on=False)
                fam_dump = await probe.rpc(op="dump")
                result["usage_per_family"][fam] = {k: fam_dump[k] for k in ("count", "weight", "tokens")}
                timing["seconds"] += part["seconds"]
                timing["tokens"] += part["tokens"]
                timing["failed"] += part["failed"]
                if total is None:
                    total = {k: np.asarray(fam_dump[k], dtype=np.float64) for k in ("count", "weight", "prob",
                                                                                      "tokens")}
                    total["dump_layers"] = fam_dump["dump_layers"]
                else:
                    for k in ("count", "weight", "prob", "tokens"):
                        total[k] += np.asarray(fam_dump[k], dtype=np.float64)
                event("usage_family", family=fam, seconds=part["seconds"], tokens=part["tokens"])
                save()
            timing["tokens_per_second"] = round(timing["tokens"] / max(timing["seconds"], 1e-9))
            dump = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in total.items()}
        else:
            timing = await run_pass(probe, usage, concurrency=SETTINGS["usage_concurrency"],
                                    prompt_logprobs=SETTINGS["usage_prompt_logprobs"], keep=keep_usage)
            await probe.rpc(op="record", on=False)
            dump = await probe.rpc(op="dump")
        result["usage"] = {"timing": timing, "nll": usage_nll,
                           **{k: dump[k] for k in ("dump_layers", "count", "weight", "prob", "tokens")}}
        event("usage", **{k: v for k, v in timing.items() if k != "failed"}, failed=len(timing["failed"]))
        if timing["failed"]:
            result["errors"].append(f"usage: {len(timing['failed'])} failed: {timing['failed'][:3]}")
        save()
        layers = dump["dump_layers"]
        pruned = masks(dump[SETTINGS["rank_by"]], dump["count"], SETTINGS["fractions"], layers)
        result["masks"] = pruned
        result["coverage"] = {key: coverage(dump[key]) for key in ("count", "weight")}

        # 2. held-out reference (record on: held-out routing statistics).
        top = SETTINGS["heldout_prompt_logprobs"]
        reference = {}

        def keep_reference(row, actual, ids, lps):
            reference[row["index"]] = (actual, ids, lps, assistant_mask(len(actual), row["assistant_spans"]))

        await probe.rpc(op="reset")
        await probe.rpc(op="record", on=True)
        timing = await run_pass(probe, heldout, concurrency=SETTINGS["reference_concurrency"], prompt_logprobs=top,
                                keep=keep_reference)
        await probe.rpc(op="record", on=False)
        held_dump = await probe.rpc(op="dump")
        result["heldout"] = {k: held_dump[k] for k in ("dump_layers", "count", "weight", "prob", "tokens")}
        result["heldout"]["coverage"] = {key: coverage(held_dump[key]) for key in ("count", "weight")}
        result["heldout"]["pruned_mass"] = {f: mass_on(p, held_dump) for f, p in pruned.items()}
        result["passes"]["reference"] = {"timing": timing, "nll": {
            "all": float(np.mean(np.concatenate([-v[0] for v in reference.values()]))) if reference else None,
            "assistant": float(np.mean(np.concatenate([-v[0][v[3]] for v in reference.values()])))
            if reference else None}}
        event("reference", **{k: v for k, v in timing.items() if k != "failed"}, failed=len(timing["failed"]))
        save()
        pass_seconds = timing["seconds"]

        # 3-4. repeat (noise) and masks, compared with the reference.
        plan = [("repeat", None, SETTINGS["repeat_concurrency"])] + [
            (f"mask-{f}", f, SETTINGS["mask_concurrency"]) for f in SETTINGS["fractions"]]
        # Existing masks screened on the same held-out set (6 October: dev t37 vs t37-all vs the balanced refit).
        extra = {name: json.loads((Path(source) / rel).read_text())["pruned"]
                 for name, rel in (options.get("extra_masks") or {}).items()}
        plan += [(f"extra-{name}", name, SETTINGS["mask_concurrency"]) for name in extra]
        for name, fraction, concurrency in plan:
            if time.time() + 1.25 * pass_seconds + 45 > deadline_epoch:
                result["skipped"].append(name)
                event("skipped", pass_name=name, remaining=round(deadline_epoch - time.time()))
                continue
            chosen = (extra[fraction] if name.startswith("extra-") else pruned[str(fraction)]
                      if fraction is not None else {})
            await probe.rpc(op="mask", pruned=chosen, label=name if fraction is not None else None)
            timing, summary, per_sequence = await compare_pass(probe, heldout, reference, concurrency, top)
            result["passes"][name] = {"fraction": fraction, "timing": timing, "summary": summary,
                                      "per_sequence": per_sequence}
            event("pass", pass_name=name, seconds=timing["seconds"], failed=len(timing["failed"]),
                  all=summary.get("all"), assistant=summary.get("assistant"))
            save()
        await probe.rpc(op="mask", pruned={}, label=None)
    except Exception as exc:  # noqa: BLE001 - the partial result is kept
        result["errors"].append(f"{type(exc).__name__}: {str(exc)[:500]}")
        event("error", error=result["errors"][-1])
    finally:
        await probe.close()
    done = [k for k in result["passes"] if k != "reference"]
    result["status"] = ("complete" if "usage" in result and "reference" in result["passes"] and done
                        and not result["errors"] else "error")
    save()
    return result
