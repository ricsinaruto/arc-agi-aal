"""Live feed of RL step-1 runs (rl1-probe-*, rl1-loop-*; docs/rl-plan.md) for mon.py, the dashboard card and the arm
agent's watcher. Each pass reads the run's rl/feed.jsonl (and modal-identity.json once the job ended) from the Modal
volume. `python rl1_feed.py RUN` prints the row; `python rl1_feed.py watch RUN [seen-file] [interval]` polls until a
deviation not in seen-file appears (exit 3) or the job ends (exit 0), printing the row each pass."""
import json, pathlib, subprocess, sys, time
import paths
OUT = paths.SCRATCH / "mon"
VOL = "arc-prolong-runs"
FEED = "arena/trials/rl-loop/rl/feed.jsonl"
STALE_S = 900          # no new feed row for 15 min
UTIL_ALERT = 70.0      # GPU use below this in a round or probe phase
MARGIN_GIB = 10.0      # free memory at a wake beyond the awake server's need
GRAD_SPIKE = 3.0       # grad norm above this multiple of the run's median so far


def get(remote, local):
    from arc_infra.runops import volfetch  # the feed export_rl and watch read too: one fetch through the mirror

    return volfetch.copy(VOL, remote, local, modal=str(paths.ROOT_PATH / ".venv/bin/modal"))


def jsonl(path):
    out = []
    for line in pathlib.Path(path).read_text().splitlines() if pathlib.Path(path).exists() else []:
        try: out.append(json.loads(line))
        except ValueError: pass
    return out


def bench_heartbeat(run):
    for _ in range(6):  # a container exec fails now and then (01:42 on rl1-probe-v8; three in a row on v9c)
        hb = _bench_heartbeat(run)
        if hb is not None: return hb
        time.sleep(20)
    return None


def _bench_heartbeat(run):
    """Seconds since the running bench's last progress line, read inside the job's container (a bench writes no feed
    row until it ends, ~1 h; the feed staleness rule would fire during every bench). None if not readable."""
    try:
        _, rec = _rec(run)  # the current attempt's app (an attempt's ledger row is <run>/attempt-N in spend.json)
        app = next((x["row"].get("app_id") for x in reversed(rec.get("ledger") or []) if x.get("row", {}).get("app_id")), None) or next(
            (r.get("app_id") for r in json.loads((paths.ROOT_PATH / "docs/spend.json").read_text())["runs"] if r.get("run_id") == run), None)
        ls = subprocess.run(["uv", "run", "--project", str(paths.ROOT_PATH), "modal", "container", "list", "--json"],
                            capture_output=True, text=True, timeout=120)
        box = next((c["container_id"] for c in json.loads(ls.stdout) if c.get("app_id") == app), None)
        if not box: return None
        r = subprocess.run(["uv", "run", "--project", str(paths.ROOT_PATH), "modal", "container", "exec", box, "--", "sh", "-c",
                            "now=$(date +%s); for f in /tmp/arena/trials/rl-loop/rl/*/progress.jsonl /tmp/arena/trials/rl-loop/rl/trainer.log; do "
                            "echo $((now - $(stat -c %Y $f))) $(tail -n1 $f); done"], capture_output=True, text=True, timeout=120)
        lines = [l for l in r.stdout.splitlines() if l[:1].isdigit()]
        if not lines: return None
        ages = [(int(l.split(" ", 1)[0]), l.split(" ", 1)[1]) for l in lines]
        age, last = min(ages)
        try: e = json.loads(last)
        except ValueError: e = {}
        if e.get("event") != "level_bench_progress": return {"age": age, "source": "trainer.log"}
        return {"age": age, "tok_s": e.get("tokens_per_second"), "done": e.get("done"), "running": e.get("running"),
                "clears": e.get("clears"), "elapsed": e.get("elapsed_seconds")}
    except Exception:
        return None


def _rec(run):
    try:
        sys.path.insert(0, str(paths.ROOT_PATH / "src"))
        from arc_infra.runops import records as R
        return R, R.get(run) or {}
    except Exception:
        return None, {}


def folder(run):
    """The current attempt's volume folder (records.out_id), as export_rl reads it (28 Oct: real-v11 attempt 2)."""
    R, rec = _rec(run)
    return R.out_id(run, rec=rec) if R and rec else run


def row(run):
    d = OUT / run; d.mkdir(parents=True, exist_ok=True)
    fo = folder(run)
    if not get(f"validation/{fo}/{FEED}", d / "feed.jsonl"):
        ended = get(f"validation/{fo}/modal-identity.json", d / "identity.json")
        return {"run": run, "state": "job ended before any feed row" if ended else "starting: no feed yet"}
    ev = jsonl(d / "feed.jsonl")
    ended = get(f"validation/{fo}/modal-identity.json", d / "identity.json")
    ident = json.loads((d / "identity.json").read_text()) if ended else None
    kinds = [e["kind"] for e in ev]
    rounds = [e for e in ev if e["kind"] == "round"]
    probes = [e for e in ev if e["kind"] == "probe_update"]
    evals = [e for e in ev if e["kind"] == "eval"]
    wakes = [e for e in ev if e["kind"] == "wake"]
    steps = [e for e in ev if e["kind"] == "update_step"]
    fit = next((e for e in ev if e["kind"] == "fit"), None)
    parity = next((e for e in ev if e["kind"] == "parity"), None)
    settings = next((e.get("settings") for e in ev if e["kind"] == "start"), {}) or {}
    upd = [(r.get("update_summary") or {}) for r in rounds if r.get("round", 0) > 0] + probes
    grads = [u.get("grad_norm") for u in upd if u.get("grad_norm") is not None]
    reward = [(r["round"], (r.get("stats") or {}).get("reward_mean")) for r in rounds if (r.get("stats") or {}).get("reward_mean") is not None]
    lp85 = [(e.get("label"), (e.get("lp85") or {}).get("mean"), (e.get("lp85") or {}).get("se")) for e in evals]
    sb26 = [(e.get("label"), (e.get("sb26") or {}).get("mean"), (e.get("sb26") or {}).get("se")) for e in evals]
    plays = [r.get("play") or {} for r in rounds] + [e.get("play") or {} for e in evals]
    tok_s = [p.get("tok_s") for p in plays if p.get("tok_s")]
    last = ev[-1] if ev else {}
    return {"run": run, "rl1": True, "mode": settings.get("mode") or ("probe" if "probe" in run else "loop"),
            "rank": settings.get("rank"), "lr": settings.get("lr"), "phase": last.get("kind"), "last_epoch": last.get("epoch"),
            "hours": (last.get("t") or 0) / 3600, "rounds": len([r for r in rounds if r.get("round", 0) > 0]),
            "of": settings.get("rounds"), "probe_updates": len(probes),
            "probe_total": len(settings.get("probe_rates") or []) * (settings.get("probe_updates") or 0),
            "reward": reward, "lp85": lp85, "sb26": sb26, "grads": grads, "tok_s": tok_s,
            "fit": None if not fit else {"chosen": fit.get("chosen"), "ok": fit.get("ok"), "modes": fit.get("modes")},
            "parity": None if not parity else {k: parity.get(k) for k in ("ok", "shift_corr", "base_mean_abs", "server_repeat_noise")},
            "rounds_rows": rounds, "probes_rows": probes, "wakes": wakes, "steps_rows": steps,
            "trend": next((r.get("trend") for r in reversed(rounds) if r.get("trend")), None), "limits": settings.get("limits") or {},
            "pick": next((e for e in ev if e["kind"] == "lr_pick"), None), "stop": next((e for e in ev if e["kind"] in ("stop", "final")), None),
            "errors": [e for e in ev if e["kind"] == "error"], "ended": ended, "identity": ident, "kinds": kinds[-12:],
            "utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime())}


def deviations(r, now=None):
    """The the alert list (13 Oct): stops and errors, mismatch, KL, grad spikes, reward collapse, GPU use, capped
    replies, a stale feed, memory margin at a wake, the job ending."""
    now = now or time.time(); d = []; lim = r["limits"]
    for e in r["errors"]: d.append(f"error: {str(e.get('error'))[:200]}")
    if r["stop"] and r["stop"]["kind"] == "stop": d.append(f"stop: {r['stop'].get('reason')}")
    if r["fit"] and r["fit"]["ok"] is False: d.append("fit: no native fit")
    if r["parity"] and r["parity"]["ok"] is False: d.append(f"parity failed: {r['parity']}")
    for x in r["rounds_rows"]:
        n = x.get("round")
        if x.get("stop"): d.append(f"round {n} stop: {x['stop']}")
        for a in x.get("alerts") or []: d.append(f"round {n} alert: {a}")
        util = ((x.get("gpu") or {}).get("all") or {}).get("util_mean")
        if util is not None and util < UTIL_ALERT: d.append(f"round {n} GPU use {util:.0f}% (< {UTIL_ALERT:.0f}%)")
        mm = x.get("mismatch") or {}
        if (lim.get("mismatch_rule") != "split" and mm.get("mean_abs") is not None
                and mm["mean_abs"] > lim.get("mismatch_mean_abs", 0.05)):  # split rule: the loop's own alert decides
            d.append(f"round {n} mismatch {mm['mean_abs']}")
        for name, p in (x.get("probes") or {}).items():
            if (p.get("kl_previous") or 0) > lim.get("kl_previous", 0.05): d.append(f"round {n} kl_previous {name} {p['kl_previous']}")
    for p in r["probes_rows"]:
        tag = f"probe rate {p.get('rate')} update {p.get('update')}"
        if p.get("ok") is False: d.append(f"{tag}: update failed")
        if (p.get("kl_previous_max") or 0) > lim.get("kl_previous", 0.05): d.append(f"{tag} kl_previous {p['kl_previous_max']}")
        if (p.get("untrained_kl_base") or 0) > lim.get("kl_base_untrained", 0.02): d.append(f"{tag} lp85 drift {p['untrained_kl_base']}")
        for m in p.get("mismatch") or []:
            if m and m.get("mean_abs") is not None and m["mean_abs"] > lim.get("mismatch_mean_abs", 0.05):
                d.append("probe updates: play = train mismatch above the limit"); break  # once, not per update
    for s in r.get("steps_rows") or []:
        for name, v in (s.get("kl_previous") or {}).items():
            if (v or 0) > lim.get("kl_previous", 0.05): d.append(f"round {s.get('round')} step {s.get('step')} kl_previous {name} {v}")
    g = r["grads"]
    for i in range(3, len(g)):
        med = sorted(g[:i])[i // 2]
        if med and g[i] > GRAD_SPIKE * med: d.append(f"grad norm spike {g[i]:.3f} (median {med:.3f}, update {i + 1})")
    if len(r["reward"]) >= 3:
        base = r["reward"][0][1]; last2 = [v for _, v in r["reward"][-2:]]
        if base is not None and sum(last2) / 2 < base - lim.get("collapse_drop", 1.0): d.append(f"reward collapse {last2} vs round 0 {base}")
    for w in r["wakes"]:
        if w.get("free_gib") is not None and w["free_gib"] - (w.get("needed_gib") or 0) < MARGIN_GIB:
            d.append(f"memory margin {w['free_gib'] - (w.get('needed_gib') or 0):.1f} GiB at a wake")
    if not r["ended"] and r["last_epoch"] and now - r["last_epoch"] > STALE_S:
        hb = bench_heartbeat(r["run"])
        r["bench"] = hb
        if hb is None or hb["age"] > STALE_S:
            d.append(f"stale feed: no row for {(now - r['last_epoch']) / 60:.0f} min (last {r['phase']}; "
                     + ("no bench heartbeat)" if hb is None else f"bench {hb['age'] / 60:.0f} min old)"))
    if r["ended"]: d.append(f"job ended: {json.dumps(r['identity'])[:200]}")
    return d


def text(r):
    b = [f"{r['mode']} rank {r['rank']}", f"phase {r['phase']}", f"{r['hours']:.1f} h"]
    if r["mode"] == "probe": b.append(f"probe update {r['probe_updates']} of {r['probe_total']}")
    else: b.append(f"round {r['rounds']}" + (f" of {r['of']}" if r["of"] is not None else ""))  # rl_seq: no round count
    if r["reward"]: b.append(f"reward {r['reward'][0][1]:.2f} -> {r['reward'][-1][1]:.2f}")
    if r["lp85"]: b.append(f"lp85 {r['lp85'][-1][1]}")
    if r["tok_s"]: b.append(f"{r['tok_s'][-1]:.0f} tok/s play")
    if r["pick"]: b.append(f"LR pick {r['pick'].get('chosen')}")
    tr = r.get("trend") or {}
    if tr.get("slope") is not None: b.append(f"trend {tr['slope']:+.3f}/round [{tr['ci95'][0]:+.3f}, {tr['ci95'][1]:+.3f}]")
    if r.get("steps_rows"):
        s = r["steps_rows"][-1]; kb = s.get("kl_base") or {}
        b.append(f"step r{s.get('round')}.{s.get('step')} KL base " + " ".join(f"{k.split('-')[1]} {v:.4f}" for k, v in kb.items() if v is not None))
    return " · ".join(b)


def bits(r):
    """Card lines (status_html)."""
    total = r['probe_total'] if r['mode'] == 'probe' else r['of']
    m = [f"<div class='v'>{r['probe_updates'] if r['mode'] == 'probe' else r['rounds']} <small>"
         + (f"of {total} " if total is not None else "") + f"{'probe updates' if r['mode'] == 'probe' else 'rounds'}"
         f" · {r['hours']:.1f} h · phase {r['phase']}</small></div>"]
    if r["reward"]:
        m.append("<div class='m'>sb26 reward per round: " + " · ".join(f"r{n} <b>{v:.2f}</b>" for n, v in r["reward"][-6:]) + "</div>")
    if r["lp85"]:
        m.append("<div class='m'>lp85 (untrained) eval: " + " · ".join(f"{lab} <b>{v}</b>±{se}" for lab, v, se in r["lp85"][-4:]) + "</div>")
    last = (r["rounds_rows"][-1].get("update_summary") if r["rounds_rows"] and r["rounds_rows"][-1].get("round") else None) or (r["probes_rows"][-1] if r["probes_rows"] else None)
    if last:
        m.append(f"<div class='m'>last update: loss <b>{last.get('loss')}</b> · grad norm <b>{last.get('grad_norm')}</b> · "
                 f"update norm {last.get('update_norm')} · KL prev max {last.get('kl_previous_max', '–')}</div>")
    if r["wakes"]:
        w = r["wakes"][-1]
        m.append(f"<div class='m'>free GPU memory at last wake <b>{w.get('free_gib')}</b> GiB (need {w.get('needed_gib')})</div>")
    tr = r.get("trend") or {}
    if tr.get("slope") is not None:
        m.append(f"<div class='m'>reward trend (primary): <b>{tr['slope']:+.3f}</b>/round, 95% CI [{tr['ci95'][0]:+.3f}, {tr['ci95'][1]:+.3f}] over {tr['n']} rounds</div>")
    if r["pick"]: m.append(f"<div class='m'>LR pick: <b>{r['pick'].get('chosen')}</b> of {r['pick'].get('rates')}</div>")
    dev = deviations(r)
    m.append("<div class='m' style='color:#e03131'>" + "; ".join(dev[:3]) + "</div>" if dev else "<div class='m'>no alerts</div>")
    m.append(f"<div class='m'>read {paths.local(r['utc']):%H:%M} {paths.tz()}</div>")
    return m


def watch(run, seen_path=None, interval=300):
    seen_path = pathlib.Path(seen_path) if seen_path else OUT / run / "seen.json"
    seen_path.parent.mkdir(parents=True, exist_ok=True)
    began = time.time()
    while True:
        r = row(run)
        if "state" in r:
            print(time.strftime("%H:%M"), r["state"], flush=True)
            if r["state"].startswith("job ended"): return 0
            if time.time() - began > 2700:  # boot plus fit take ~25 min; the feed starts at the trial's start
                print("DEVIATION", json.dumps([f"no feed after {(time.time() - began) / 60:.0f} min"]), flush=True)
                return 3
        else:
            seen = set(json.loads(seen_path.read_text())) if seen_path.exists() else set()
            dev = deviations(r)
            new = [x for x in dev if x not in seen]
            print(time.strftime("%H:%M"), text(r), ("bench " + json.dumps(r["bench"])) if r.get("bench") else "", flush=True)
            if new:
                seen_path.write_text(json.dumps(sorted(seen | set(new))))
                print("DEVIATION", json.dumps(new), flush=True)
                return 0 if r["ended"] and all(x.startswith("job ended") for x in new) else 3
            if r["ended"]: return 0
        time.sleep(interval)


if __name__ == "__main__":
    if sys.argv[1] == "watch":
        sys.exit(watch(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None, int(sys.argv[4]) if len(sys.argv) > 4 else 300))
    r = row(sys.argv[1])
    print(json.dumps({k: v for k, v in r.items() if k not in ("rounds_rows", "probes_rows")}, indent=1, default=str)[:4000])
    if "state" not in r: print("DEVIATIONS", json.dumps(deviations(r), indent=1))
