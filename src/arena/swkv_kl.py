"""Sliding-window cache measurement (8 October; launcher test ``tp-swkv-kl``, trial kind ``prune-quality`` with role
``swkv``; server with ``ARENA_SWKV_PROBE=1``, ``ARENA_WARM_PROMPT_LOGPROBS=1``, ``ARENA_ENFORCE_EAGER=1``).

Does play change when a drain keeps the kept half's cached state instead of re-reading it? Per recorded post-drain
context (``configs/swkv/contexts.json``, built from DEV games: system S, dropped half D, kept half K, the drain note
and the recorded continuation N) the next-token distributions on N's assistant tokens are compared between arms:

- ``A``   re-read: S + K + note + N from scratch (what the server computes today; ``A2`` repeats it: noise);
- ``A0``  the recorded layout, the note before K;
- ``C``   never drained: S + D + K + note + N (``C2`` repeats it);
- ``Br``  sliding window, whole 3,136-token blocks dropped: C's tokens with the QSA selection of everything from the
  note on masked on the dropped blocks (``arena.swkv_probe_runtime``); ``Ba`` drops exactly D (token-aligned);
- ``S<n>`` A with every RoPE position shifted by n (positions past the trained 262,144).

Each arm is a primer (the tokens before the note, or before N for A0; no logprobs) and a warm request with
``prompt_logprobs`` that resumes from it, under its own ``cache_salt`` so no arm reads another arm's state. Arms run one
at a time (the probe switches are server-wide). Passes run in priority order until the deadline; arrays and the
summary are saved after every context.
"""

import json
import time
from pathlib import Path

import httpx
import numpy as np

from arc_infra.serving import MODEL

from .expert_usage import compare
from .isolation import isolate_trial
from .kernel_gate import token_id

INPUTS = "configs/swkv/contexts.json"
ARMS = {
    "A": {"seq": "A"}, "A2": {"seq": "A"}, "A0": {"seq": "A0"},
    "C": {"seq": "C", "mask": "ragged", "count_only": True}, "C2": {"seq": "C"},
    "Br": {"seq": "C", "mask": "ragged"}, "Ba": {"seq": "C", "mask": "aligned"},
    "S200": {"seq": "A", "shift": 200_000}, "S500": {"seq": "A", "shift": 500_000},
    "S900": {"seq": "A", "shift": 900_000},
}
PAIRS = [("C", "A"), ("C", "Br"), ("C", "Ba"), ("C", "C2"), ("A", "A2"), ("A", "Br"), ("A", "Ba"), ("A", "A0"),
         ("A", "S200"), ("A", "S500"), ("A", "S900"), ("Br", "Ba")]
DEFAULT_PASSES = [{"arms": ["A", "C", "Br", "A2"]}, {"arms": ["S500"]}, {"arms": ["Ba"]}, {"arms": ["S200", "S900"]},
                  {"arms": ["A0"]}, {"arms": ["C2"]}]


def tail_arrays(entries, ids, first, offset, top):
    """Recorded-token logprob and the top-``top`` ids / logprobs for targets ``first`` .. len(ids) - 1 (entry index =
    position - offset)."""
    n = len(ids) - first
    actual = np.empty(n, dtype=np.float32)
    top_ids = np.full((n, top), -1, dtype=np.int32)
    top_lps = np.full((n, top), -np.inf, dtype=np.float32)
    for i in range(n):
        position = first + i
        entry = entries[position - offset]
        if not entry:
            raise RuntimeError(f"no logprobs at position {position}")
        ranked = sorted(((token_id(t), v["logprob"]) for t, v in entry.items()), key=lambda row: row[1], reverse=True)
        found = next((lp for t, lp in ranked if t == ids[position]), None)
        if found is None:
            raise RuntimeError(f"recorded token missing at position {position}")
        actual[i] = found
        for j, (t, lp) in enumerate(ranked[:top]):
            top_ids[i, j], top_lps[i, j] = t, lp
    return actual, top_ids, top_lps


def span_mask(n, spans):
    """Targets (index 0 = tail position 1) inside the assistant spans, given relative to the tail start."""
    mask = np.zeros(n, dtype=bool)
    for start, end in spans:
        mask[max(start - 1, 0):max(end - 1, 0)] = True
    return mask


def pair_rows(context, arrays):
    """Per (reference, candidate) pair of one context: assistant positions, KL sum, top-1 agreement, NLL sums."""
    rows = {}
    for ref, cand in PAIRS:
        if ref not in arrays or cand not in arrays:
            continue
        (ra, ri, rl), (ca, ci, cl) = arrays[ref], arrays[cand]
        mask = span_mask(len(ra), context["spans"])
        kl, agree, _ = compare((ri, rl), (ci, cl))
        rows[f"{ref}|{cand}"] = {"positions": int(mask.sum()), "kl_sum": float(kl[mask].sum()),
                                 "agree": int(agree[mask].sum()), "nll_ref_sum": float(-ra[mask].sum()),
                                 "nll_sum": float(-ca[mask].sum())}
    return rows


def pooled(per_context, bootstrap=4000, seed=20261008):
    """Pooled means per pair, and the two decision differences with a 95% bootstrap over contexts."""
    out = {"pairs": {}, "decisions": {}}
    names = sorted(per_context)
    for key in sorted({k for rows in per_context.values() for k in rows}):
        have = [per_context[n][key] for n in names if key in per_context[n]]
        positions = sum(r["positions"] for r in have)
        if positions:
            out["pairs"][key] = {"contexts": len(have), "positions": positions,
                                 "mean_kl": sum(r["kl_sum"] for r in have) / positions,
                                 "top1_agreement": sum(r["agree"] for r in have) / positions,
                                 "nll_ref": sum(r["nll_ref_sum"] for r in have) / positions,
                                 "nll": sum(r["nll_sum"] for r in have) / positions}
    rng = np.random.default_rng(seed)

    def difference(first, second, field, name):
        both = [n for n in names if first in per_context[n] and second in per_context[n]]
        if not both:
            return
        a = np.array([[per_context[n][first][field], per_context[n][first]["positions"]] for n in both])
        b = np.array([[per_context[n][second][field], per_context[n][second]["positions"]] for n in both])

        def stat(index):
            return a[index, 0].sum() / a[index, 1].sum() - b[index, 0].sum() / b[index, 1].sum()

        draws = [stat(rng.integers(0, len(both), len(both))) for _ in range(bootstrap)] if len(both) > 1 else []
        out["decisions"][name] = {"contexts": len(both), "value": float(stat(np.arange(len(both)))),
                                  "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
                                  if draws else None}

    difference("C|Br", "C|A", "kl_sum", "KL(C||Br) - KL(C||A)")
    difference("C|Ba", "C|A", "kl_sum", "KL(C||Ba) - KL(C||A)")
    difference("C|Br", "C|C2", "kl_sum", "KL(C||Br) - KL(C||C2)")
    difference("A|Br", "A|A2", "kl_sum", "KL(A||Br) - KL(A||A2)")
    difference("A|Br", "A|A2", "nll_sum", "NLL(Br) - NLL(A2)")
    difference("C|Br", "C|A", "nll_sum", "NLL(Br) - NLL(A)")
    difference("C|Ba", "C|A", "nll_sum", "NLL(Ba) - NLL(A)")
    for shift in ("S200", "S500", "S900"):
        difference(f"A|{shift}", "A|A2", "kl_sum", f"KL(A||{shift}) - KL(A||A2)")
        difference(f"A|{shift}", "A|A2", "nll_sum", f"NLL({shift}) - NLL(A2)")
    difference("A|A0", "A|A2", "kl_sum", "KL(A||A0) - KL(A||A2)")
    return out


class Client:
    def __init__(self, base_url, model, transport=None):
        self.base_url = base_url.rstrip("/")
        self.root = self.base_url.removesuffix("/v1")
        self.model = model
        self.http = httpx.AsyncClient(timeout=900, trust_env=False, transport=transport)

    async def close(self):
        await self.http.aclose()

    async def rpc(self, **command):
        response = await self.http.post(self.root + "/collective_rpc", json={
            "method": "arc_swkv_probe", "args": [json.dumps(command)], "timeout": 120})
        if response.status_code != 200:
            raise RuntimeError(f"collective_rpc {response.status_code}: {response.text[:300]}")
        reply = response.json()["results"][0]
        if not isinstance(reply, dict) or not reply.get("ok"):
            raise RuntimeError(f"swkv probe {command.get('op')}: {str(reply)[:300]}")
        return reply

    async def complete(self, **body):
        response = await self.http.post(self.base_url + "/completions", json={
            "model": self.model, "max_tokens": 1, "temperature": 0.0, **body})
        if response.status_code != 200:
            raise RuntimeError(f"completions {response.status_code}: {response.text[:300]}")
        return response.json()


async def run_arm(client, context, name, top, salt):
    """One arm of one context: ``(arrays, record)``."""
    arm = ARMS[name]
    ids = context["seqs"][arm["seq"]]
    primer, start = context["primer"][arm["seq"]], context["start"][arm["seq"]]
    command = {"op": "set", "shift": arm.get("shift", 0)}
    if arm.get("mask"):
        command.update(context["masks"][arm["mask"]], count_only=bool(arm.get("count_only")))
    record = {"arm": name, "tokens": len(ids), "primer": primer, "start": start}
    armed = await client.rpc(**command)
    try:
        began = time.monotonic()
        await client.complete(prompt=ids[:primer], cache_salt=salt)
        record["primer_seconds"] = round(time.monotonic() - began, 2)
        began = time.monotonic()
        reply = await client.complete(prompt=ids, cache_salt=salt, prompt_logprobs=top,
                                      return_tokens_as_token_ids=True, vllm_xargs={"arc_read_prefix_cache": 1})
        record["scored_seconds"] = round(time.monotonic() - began, 2)
    finally:
        status = await client.rpc(op="clear")
    record["counters"] = status.get("before")
    # Evidence that the switches were live in the worker that computed the arm (QSA forwards under this shift).
    record["probe"] = {"shift": armed.get("shift"), "cut": armed.get("cut"),
                       "forward_calls": (status.get("forward_calls") or 0) - (armed.get("forward_calls") or 0),
                       "select_calls": (status.get("select_calls") or 0) - (armed.get("select_calls") or 0)}
    entries = reply["choices"][0].get("prompt_logprobs") or []
    cached = ((reply.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    record["cached"] = cached
    if len(entries) == len(ids):
        offset = 0
    elif cached and len(entries) == len(ids) - cached:
        offset = cached
    else:
        raise RuntimeError(f"{name}: prompt_logprobs length {len(entries)} for {len(ids)} tokens, {cached} cached")
    if offset > start:
        raise RuntimeError(f"{name}: the hit ({cached}) is past the scored span ({start})")
    arrays = tail_arrays(entries, ids, start + 1, offset, top)
    return arrays, record


async def run_swkv_kl(study, trial, *, source, output, server_dir, base_url, deadline_epoch, log=print,
                      transport=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = study.get("swkv") or {}
    top = int(settings.get("top", 20))
    passes = settings.get("passes") or DEFAULT_PASSES
    contexts = json.loads((Path(source) / INPUTS).read_text())
    client = Client(base_url, study.get("served_model", MODEL), transport=transport)
    result = {"trial": trial["name"], "role": "swkv", "server_env": trial.get("server_env") or {}, "errors": [],
              "arms": [], "skipped": [], "per_context": {}, "contexts": [
                  {k: c[k] for k in ("index", "run", "game", "family", "meta")} for c in contexts]}
    arrays = {c["index"]: {} for c in contexts}
    rates = {"primer": 8000.0, "scored": 2500.0}  # tokens per second, updated from every arm
    path = output / "swkv-kl.json"

    def event(name, **extra):
        log(json.dumps({"event": f"prune_quality_swkv_{name}", "trial": trial["name"], **extra}), flush=True)

    def save(context=None):
        if context is not None:
            np.savez_compressed(output / f"arrays-{context['index']}.npz", **{
                f"{arm}_{kind}": value for arm, parts in arrays[context["index"]].items()
                for kind, value in zip(("actual", "ids", "lps"), parts)})
            result["per_context"][str(context["index"])] = pair_rows(context, arrays[context["index"]])
        result["summary"] = pooled(result["per_context"])
        path.write_text(json.dumps(result))

    try:
        await isolate_trial(base_url)
        status = await client.rpc(op="clear")
        event("probe", select_calls=status["select_calls"], forward_calls=status["forward_calls"])
        for number, spec in enumerate(passes):
            chosen = [c for c in contexts if spec.get("contexts") is None or c["index"] in spec["contexts"]]
            for context in chosen:
                for name in spec["arms"]:
                    arm = ARMS[name]
                    ids = context["seqs"][arm["seq"]]
                    primer = context["primer"][arm["seq"]]
                    estimate = primer / rates["primer"] + (len(ids) - primer) / rates["scored"] + 4
                    if time.time() + estimate > deadline_epoch - 15:
                        result["skipped"].append({"context": context["index"], "arm": name, "pass": number})
                        continue
                    try:
                        parts, record = await run_arm(client, context, name, top,
                                                      f"swkv-{trial['name']}-{context['index']}-{name}")
                    except Exception as exc:  # noqa: BLE001 - one arm is recorded, the passes go on
                        result["errors"].append(f"context {context['index']} {name}: {type(exc).__name__}: "
                                                f"{str(exc)[:300]}")
                        event("arm_error", context=context["index"], arm=name, error=result["errors"][-1])
                        if len(result["errors"]) >= 4:
                            raise RuntimeError("four arms failed") from exc
                        continue
                    arrays[context["index"]][name] = parts
                    rates["primer"] = 0.5 * rates["primer"] + 0.5 * primer / max(record["primer_seconds"], 0.2)
                    rates["scored"] = 0.5 * rates["scored"] + 0.5 * (len(ids) - record["cached"]) / max(
                        record["scored_seconds"], 0.2)
                    record.update(context=context["index"], game=context["game"], **{"pass": number})
                    result["arms"].append(record)
                    event("arm", **record)
                save(context)
            event("pass", number=number, arms=spec["arms"], summary=result["summary"])
    except Exception as exc:  # noqa: BLE001 - the partial result is kept
        result["errors"].append(f"{type(exc).__name__}: {str(exc)[:500]}")
        event("error", error=result["errors"][-1])
    finally:
        await client.close()
    result["status"] = "complete" if result["arms"] and not result["errors"] else (
        "complete_with_errors" if result["arms"] else "error")
    save()
    return result
