"""Arena adapter for the shared watcher: arena/roster runs (throughput c100, PTH H10 reads, arc2 c240) write no
train-metrics rows; their live status maps onto the same infra.* keys, so the plan-vs-actual checks apply unchanged.

* ARC-AGI-3 arena: ``arena/live/progress.jsonl`` (one row per minute: epoch, live.engine, gate, live.host, errors).
* ARC-AGI-2 roster: ``arena/live/arc2-<trial>.json`` (one snapshot, rewritten every minute, no epoch: the caller
  stamps it with the file's modification time).
"""


def _get(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def progress_rows(rows):
    """Metric rows from arena progress rows (sorted by epoch). gen tok/s is the generated-token delta between rows
    (engine counters restart at 0 after an engine restart: the row's own recent rate is used there); the counter
    itself is infra.gen_tokens, for the watcher's frozen rule (a hung engine behind a live trial loop)."""
    out, prev = [], None
    for r in sorted((r for r in rows if r.get("epoch")), key=lambda r: r["epoch"]):
        eng = _get(r, "live", "engine") or {}
        gen = _num(eng.get("generated_tokens"))
        rate = _num(eng.get("tokens_per_second_recent"))
        if prev and gen is not None and prev[1] is not None and gen >= prev[1] and r["epoch"] > prev[0]:
            rate = (gen - prev[1]) / (r["epoch"] - prev[0])
        if rate is not None and rate < 0:  # the recent-rate field across an engine restart
            rate = None
        prev = (r["epoch"], gen)
        waiting = sum(x for x in (_num(eng.get("waiting")), _num(_get(r, "gate", "waiting"))) if x is not None)
        m = {"infra.gen_tok_s": rate, "infra.gen_tokens": gen, "infra.running": _num(eng.get("running")), "infra.waiting": waiting,
             "infra.kv_frac": _num(eng.get("kv_cache_fraction")), "infra.host_avail_gib": _num(_get(r, "live", "host",
                                                                                                  "available_gib")),
             "play.slots": float(len(r["active"])) if isinstance(r.get("active"), list) else None,
             "play.errors": _num(_get(r, "live", "errors")), "play.preemptions": _num(eng.get("preemptions")),
             "play.score": _num(_get(r, "live", "score", "score"))}
        out.append({"t": float(r["epoch"]), "m": {k: v for k, v in m.items() if v is not None}})
    return out


def arc2_row(snap, t, prev=None):
    """One metric row from an ARC-AGI-2 live snapshot taken at ``t``; ``prev`` = (t, completion_tokens) of the last
    one, for the token rate between snapshots (else the run's mean rate)."""
    tok = _num(snap.get("completion_tokens"))
    rate = _num(snap.get("tokens_per_second"))
    if prev and tok is not None and prev[1] is not None and tok >= prev[1] and t > prev[0]:
        rate = (tok - prev[1]) / (t - prev[0])
    gpu = _get(snap, "gpu", "recent") or _get(snap, "gpu", "play") or {}
    gate = snap.get("gate") or {}
    m = {"infra.gen_tok_s": rate, "infra.gen_tokens": tok, "infra.running": _num(gate.get("inflight")), "infra.waiting": _num(gate.get("waiting")),
         "play.slots": _num(gate.get("slots")), "infra.gpu_util.0": _num(gpu.get("util_mean")),
         "infra.vram_gib.0": _num(gpu.get("memory_peak_gib") or gpu.get("memory_mean_gib")),
         "play.errors": sum(x for x in (_num(snap.get("failures")), _num(snap.get("request_errors"))) if x is not None),
         "play.score": _num(_get(snap, "final", "score_outputs"))}
    return {"t": float(t), "m": {k: v for k, v in m.items() if v is not None}}
