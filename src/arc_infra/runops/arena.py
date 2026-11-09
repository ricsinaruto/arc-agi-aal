"""Arena adapter for the shared watcher: arena/roster runs (throughput c100, PTH H10 reads, arc2 c240) write no
train-metrics rows; their live status maps onto the same infra.* keys, so the plan-vs-actual checks apply unchanged.

* ARC-AGI-3 arena: ``arena/live/progress.jsonl`` (one row per minute: epoch, live.engine, gate, live.host, errors).
* ARC-AGI-2 roster: ``arena/live/arc2-<trial>.json`` (one snapshot, rewritten every minute, no epoch: the caller
  stamps it with the file's modification time).
* RL runs (runops sampler): ``validation/<run>/live/sampler.jsonl`` (one row a minute: ``t`` seconds since the sampler
  started, per-card ``util``/``mem_gib``, the play server's ``vllm`` gauges and counters) and ``util/cpulog.txt``
  (epoch-stamped host lines with ``mem_gib``).
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


def arc2_tokens(snap):
    """Generated tokens of an ARC-AGI-2 live snapshot: the read-out entries' plus the extra runs' (copy3 thirds, twin
    second tries), which run on the same server once the entries have ended."""
    tok = _num(snap.get("completion_tokens"))
    return None if tok is None else tok + (_num(snap.get("extra_tokens")) or 0)


def arc2_row(snap, t, prev=None):
    """One metric row from an ARC-AGI-2 live snapshot taken at ``t``; ``prev`` = (t, ``arc2_tokens``) of the last
    one, for the token rate between snapshots (else the run's mean rate)."""
    tok = arc2_tokens(snap)
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
    started, replies = _num(_get(snap, "where_tokens_go", "by_attempt", "1", "attempts")), _num(snap.get("replies"))
    if started and replies is not None:  # fork runs: a stale bundle answers once per attempt (fk-naive2 p2, 29 Oct)
        m["play.replies_per_attempt"] = replies / started
    return {"t": float(t), "m": {k: v for k, v in m.items() if v is not None}}


def level_bench_rows(rows, mtime):
    """Metric rows from a level bench's progress.jsonl (``level_bench_progress`` lines; RL held-out reads, 25 Oct).
    The rows carry elapsed seconds only: the last one is stamped with the file's mtime, the others back from it."""
    rows = [r for r in rows if r.get("event") == "level_bench_progress" and _num(r.get("elapsed_seconds")) is not None]
    if not rows or mtime is None:
        return []
    last = rows[-1]["elapsed_seconds"]
    out, prev = [], None
    for r in rows:
        t = float(mtime) - (last - r["elapsed_seconds"])
        tok = _num(r.get("completion_tokens"))
        rate = (tok - prev[1]) / (t - prev[0]) if prev and tok is not None and prev[1] is not None and t > prev[0] \
            and tok >= prev[1] else None
        prev = (t, tok)
        gate = r.get("gate") or {}
        m = {"infra.gen_tok_s": rate, "infra.gen_tokens": tok, "infra.running": _num(gate.get("inflight")),
             "infra.waiting": _num(gate.get("waiting")), "play.slots": _num(r.get("running")),
             "play.errors": _num(r.get("failed")), "play.done": _num(r.get("done")), "play.clears": _num(r.get("clears"))}
        out.append({"t": t, "m": {k: v for k, v in m.items() if v is not None}})
    return out


def server_card(port):
    """The card of an RL play server: :1234 on card 0, rl_loop.SecondServer's 1235 + i on card i."""
    port = int(port)
    return 0 if port == 1234 else port - 1235


def sampler_rows(rows, launched_epoch=None, mtime=None):
    """Metric rows from the runops sampler's rows. ``t`` is relative: the last row is stamped with the file's ``mtime``
    when known (the container starts minutes after the launch), else rows are ``launched_epoch`` + t; a last row with
    its own ``epoch`` (the sampler writes it since 27 Oct) fixes the base exactly. gen tok/s is the
    generation-token counter's delta between rows (None across a server restart, where it falls). On N cards the
    sampler's ``vllm_servers`` (per port) become ``<key>.s<port>`` (``server_card`` maps a port to its card); the
    unsuffixed keys stay the totals."""
    rows = sorted((r for r in rows if _num(r.get("t")) is not None), key=lambda r: r["t"])
    if not rows:
        return []
    if _num(rows[-1].get("epoch")) is not None:
        t0 = float(rows[-1]["epoch"]) - float(rows[-1]["t"])
    elif mtime is not None:
        t0 = float(mtime) - float(rows[-1]["t"])
    elif launched_epoch is not None:
        t0 = float(launched_epoch)
    else:
        return []
    out, prev, prev_s, prev_p, prev_sp = [], None, {}, None, {}
    for r in rows:
        t = t0 + float(r["t"])
        m = {}
        for port, v in sorted((r.get("vllm_servers") or {}).items()):  # N cards (H2 26 Oct): one server per card
            gen = _num(v.get("generation_tokens_total"))
            last = prev_s.get(port)
            rate = (gen - last[1]) / (t - last[0]) if gen is not None and last and gen >= last[1] and t > last[0] else None
            if gen is not None:
                prev_s[port] = (t, gen)
            kv = _num(v.get("kv_cache_usage_perc"))
            ph, pq = _num(v.get("prefix_cache_hits_total")), _num(v.get("prefix_cache_queries_total"))
            if ph is not None and pq is not None:  # per-server prefix hit rate between rows
                lp = prev_sp.get(port)
                if lp and pq > lp[1] and ph >= lp[0]:
                    m[f"infra.prefix_hit.s{port}"] = (ph - lp[0]) / (pq - lp[1])
                prev_sp[port] = (ph, pq)
            m.update({f"infra.running.s{port}": _num(v.get("num_requests_running")),
                      f"infra.waiting.s{port}": _num(v.get("num_requests_waiting")),
                      f"infra.kv_frac.s{port}": kv / 100 if kv is not None and kv > 1 else kv,
                      f"infra.gen_tokens.s{port}": gen, f"infra.gen_tok_s.s{port}": rate,
                      f"play.preemptions.s{port}": _num(v.get("num_preemptions_total"))})
        for i, u in enumerate(r.get("util") or []):
            m[f"infra.gpu_util.{i}"] = _num(u)
        for i, g in enumerate(r.get("mem_gib") or []):
            m[f"infra.vram_gib.{i}"] = _num(g)
        v = r.get("vllm") or {}
        gen = _num(v.get("generation_tokens_total"))
        rate = None
        if gen is not None and prev and gen >= prev[1] and t > prev[0]:
            rate = (gen - prev[1]) / (t - prev[0])
        if gen is not None:
            prev = (t, gen)
        kv = _num(v.get("kv_cache_usage_perc"))
        hits, queries = _num(v.get("prefix_cache_hits_total")), _num(v.get("prefix_cache_queries_total"))
        if hits is not None and queries is not None:  # prefix hit rate between rows (per-phase target)
            if prev_p and queries > prev_p[1] and hits >= prev_p[0]:
                m["infra.prefix_hit"] = (hits - prev_p[0]) / (queries - prev_p[1])
            prev_p = (hits, queries)
        m.update({"infra.running": _num(v.get("num_requests_running")), "infra.waiting": _num(v.get("num_requests_waiting")),
                  "infra.kv_frac": kv / 100 if kv is not None and kv > 1 else kv, "infra.gen_tokens": gen,
                  "infra.gen_tok_s": rate, "play.preemptions": _num(v.get("num_preemptions_total"))})
        out.append({"t": t, "m": {k: x for k, x in m.items() if x is not None}})
    return out


def cpulog_rows(text, every=30.0):
    """infra.host_ram_gib from the runops cpulog (``<epoch> cpu ... mem_gib X ...`` lines, ~1 s apart), one row per
    ``every`` seconds; the plan's host.total_gib turns it into available RAM."""
    out, last = [], None
    for line in (text or "").splitlines():
        parts = line.split()
        try:
            t = float(parts[0])
            gib = float(parts[parts.index("mem_gib") + 1])
        except (ValueError, IndexError):
            continue
        if last is None or t - last >= every:
            out.append({"t": t, "m": {"infra.host_ram_gib": gib}})
            last = t
    return out


SEQ_PLAY = ("seq_start", "seq_phase")  # written once play (re)starts (seq_phase: after to_play)
SEQ_TRAIN = ("seq_train",)  # written when a train phase starts (before to_train)


def seq_feed_rows(feed):
    """Phase rows from an RL loop feed (rl-loop/rl/feed.jsonl): ``seq.phase`` 1 at each play start, 0 at each train
    start (seq_train, or the first seq_accumulate after a play start when the feed has no seq_train), plus train
    progress: ``seq.train_rows`` (cumulative seq_accumulate rows) and ``train.step``."""
    out, phase, n, pending, calls, ahead = [], None, 0, False, 0, False
    for r in sorted((r for r in feed if _num(r.get("epoch")) is not None), key=lambda r: r["epoch"]):
        k, m = r.get("kind"), {}
        if k == "seq_resume" and (_num(r.get("pending_groups")) or 0) > 0:
            # a resume that trains its saved groups first (real-v6 27 Oct, pending_only): a train phase until step 1;
            # its seq_start comes before the replay and plays nothing
            phase, pending, m["seq.phase"] = 0, True, 0.0
        elif k in SEQ_PLAY and not (pending and k == "seq_start"):
            phase, m["seq.phase"] = 1, 1.0
        elif k in SEQ_TRAIN or (k == "seq_accumulate" and phase != 0):
            phase, m["seq.phase"] = 0, 0.0
        if k == "seq_accumulate":
            n += 1
            m["seq.train_rows"] = float(n)
        if k == "step" and _num(r.get("step")) is not None:
            m["train.step"] = float(r["step"])
            pending = False
        # per-phase targets: switch times, setup, the update's whole-call util, API-server CPU
        if k == "seq_phase":
            m.update({"seq.to_train_s": _num((r.get("stop") or {}).get("switch_s")),
                      "seq.to_play_s": _num((r.get("start") or {}).get("switch_s"))})
        if k == "seq_start":
            m["seq.setup_s"] = _num(r.get("setup_seconds"))
        if (k == "seq_early_ahead" and (_num(r.get("specs")) or 0) > 0) or (
                k == "seq_ahead_pending" and (_num(r.get("groups")) or 0) > 0):
            ahead = True  # the next update's windows were handed to the trainer to encode early
        if k == "seq_accumulate" and r.get("one_call") and (ahead or r.get("ahead_expected")):
            # gate legs 28 Oct: 0 of 192 ahead windows taken in every rl_seq run (key mismatch), unseen for weeks; an
            # ahead update without a reading counts as 0 (the trainer's encode stats never reached the feed)
            share = _num(r.get("ahead_hit_share")) or 0.0
            if "ahead_misses" in r and (_num(r.get("ahead_misses")) or 0) > 0:  # gate-v3 rows (trunk a14120868): that
                share = (_num(r.get("ahead_hits")) or 0.0) / _num(r["ahead_misses"])  # field held the encoded total
            m["train.ahead_hit_share"] = share
            ahead = False
        if k == "seq_accumulate" and r.get("one_call") and "encode_errors" in r:
            # v17 a3 (30 Oct): R3 coverage and encoder errors per call (rows from trunk a5c9944c7 on); a row without
            # served_share had no served routing at all, so it counts 0
            m["train.served_share"] = _num(r.get("served_share")) or 0.0
            m["train.encode_errors"] = _num(r.get("encode_errors"))
        if k == "seq_accumulate" and r.get("one_call"):
            # the feed's first update follows the trainer's load (cold: real-v5 57.5 / 54.5 s, real-v6 59.3 / 28.7 s):
            # logged as *_cold, outside the per-update targets; warm updates carry the judged keys
            cold = "_cold" if calls == 0 else ""
            calls += 1
            m.update({f"train.call_util{cold}": _num(r.get("call_util")), f"train.setup_s{cold}": _num(r.get("setup_seconds")),
                      f"train.card_peak_gib{cold}": _num(r.get("card_peak_gib"))})  # VRAM in the update too
        if k == "seq_admit" and isinstance(r.get("cpu"), dict):  # the API servers' processes ("python:<pid>")
            api = [_num(v) for name, v in r["cpu"].items() if str(name).startswith("python:") and _num(v) is not None]
            if api:
                m["infra.api_cpu"] = max(api)
        m = {key: v for key, v in m.items() if v is not None}
        if k == "alert" and r.get("why"):  # the run's own alerts (heap legs 27 Oct: "guard blind" when /metrics
            out.append({"t": float(r["epoch"]), "m": m, "alert": str(r["why"])})  # stalled 20 s+; the watcher missed it)
        elif m:
            out.append({"t": float(r["epoch"]), "m": m})
    return out
