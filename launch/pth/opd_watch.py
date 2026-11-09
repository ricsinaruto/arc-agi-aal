"""opd_watch.py RUN APP_ID: live watcher of a heal_round (every 5 min) on gpu.jsonl, train/events.jsonl and teacher0.log.
opd_watch.py --backtest DIR [CASE]: replays DIR/gpu.jsonl + DIR/events.jsonl at 5-min reads, prints the first alert.
opd_watch.py --stop-test APP: verified non-interactive stop (the agent's own stop, after inspecting an alert).

Alerts and exits (run it as the agent's background task, so the exit wakes the agent) on:
- a non-finite step;
- KL blow-up: reply KL of 2 consecutive steps > 2x the median of the (up to 5) steps before them;
- divergence: grad norm of 2 consecutive steps > 10x the median of the (up to 5) steps before them, or both > GN_MAX (2.0);
- no progress: reply KL at steps 9-10 not >= 10% below steps 1-2 (healthy OPD r1: -40% by step 11);
- optional KL_BAR=A-B:X: mean reply KL of steps A-B > X (pre-registered mid-run bar);
- optional MID_RULES=1: OPD round stop rules on the trainer's mid_read events (teacher-H forward > 1.10 x D, dh10
  top-32 tail > 2 x D, dh10 sampled reverse not falling on 2 consecutive reads);
- memory over the launch's prediction by > 3% (env PRED_LOADED: loaded gpu_gib; PRED_STEP1: step-1 peak_gib;
  PRED_PEAK: later steps' peak_gib; GiB per GPU);
- silence, two signals: no gpu.jsonl row for 15 min AND no progress record for 15 min (train events in the train
  phase, teacher0.log growth in the teacher phase); healthy max gaps: gpu rows 15 s, train events 249 s;
- data age (env AGE_MED, AGE_MAX; streaming OPD): 2 consecutive steps over either bar;
- over plan (env MAX_SECONDS): the run is older than its planned wall;
- throughput, two signals on 3 reads in a row: step util (5-min mean) < 70% AND median step time of the read
  > 1.5x the reference (env REF_DT, seconds: OPD rounds 94 measured on opd1-v5; unset: the run's own steps 2-4);
  healthy reads: util 75.6-91.4%, step time 77-94 s median, 126 s max.
"""
import json, os, statistics, subprocess, sys, time
from pathlib import Path

MV = "./.venv/bin/modal"
SILENCE = 900


def damage(ev, dfwd=1.05, gen=1.20, consecutive=2):
    mids = [e for e in ev if e.get("event") == "mid_read" and e.get("d_fwd") is not None]
    h0 = [e for e in ev if e.get("event") == "heldout2_before" and e.get("kd_reply") is not None]
    series = []
    if mids and mids[0].get("step") == 0:
        series.append(("mid dh10 forward", mids[0]["d_fwd"], [(m["step"], m["d_fwd"]) for m in mids[1:]]))
    if h0:
        series.append(("heldout2 dh10 forward", h0[0]["kd_reply"],
                       [(e.get("step", "end"), e["kd_reply"]) for e in ev if e.get("event") == "heldout2" and e.get("kd_reply") is not None]))
    for name, base, reads in series:
        run = 0
        for at, x in reads:
            run = run + 1 if base and x > dfwd * base else 0
            if run >= consecutive:
                return f"damage: {name} {x:.4f} > {dfwd} x start {base:.4f} on {consecutive} consecutive reads (last at {at})"
    g = [e for e in ev if e.get("event") == "general_probe" and e.get("kd") is not None]
    g0 = next((e["kd"] for e in g if e.get("when") == "before"), None)
    later = [e["kd"] for e in g if e.get("when") != "before"] + [m["g_kd"] for m in mids if m.get("g_kd") is not None]
    for x in later:
        if g0 and x > gen * g0:
            return f"damage: general-text KL {x:.4f} > {gen} x start {g0:.4f}"
    return None


def follow_lag(ev, follow, now, lag=900):
    """Play weights (c5/c6, 23 Oct: v13 halted following on a repoint and served c4's v1 for both runs, silently):
    any 'halted' row, or a trainer export (delta_export, not skipped, version > 0) older than ``lag`` s with no
    'applied' row of that version or newer written after it (a resumed or restarted trainer re-uses versions the
    play already holds, which it ignores)."""
    halted = [r for r in follow if r.get("kind") == "halted" and r.get("t", 0) <= now]
    if halted:
        return f"play weights: follower halted ({halted[-1].get('reason')}); restart the play"
    applied = [(r["t"], r["version"]) for r in follow if r.get("kind") == "applied" and r.get("t", 0) <= now]
    for e in ev:
        if (e.get("event") == "delta_export" and e.get("tensors") and not e.get("skipped") and (e.get("step") or 0) > 0
                and e["t"] <= now - lag):
            if not any(ta >= e["t"] and va >= e["step"] for ta, va in applied):
                have = max((va for ta, va in applied), default=None)
                return (f"play weights: v{e['step']} exported {now - e['t']:.0f} s ago, not applied (play applied "
                        f"v{have}; bar {lag:.0f} s)")
    return None


def full_rows(ev, follow, now, allowed=2):
    """Deltas after the confirm (c7, 24 Oct: the play never confirmed, so every version was full: 43-49 s of gather per
    step vs 6-8 s): more than ``allowed`` full-row exports written after the play's follower started."""
    starts = [r["t"] for r in follow if r.get("kind") == "start" and r.get("t", 0) <= now]
    if not starts:
        return None
    full = [e for e in ev if e.get("event") == "delta_export" and not e.get("skipped") and e.get("full")
            and e["t"] >= starts[0] and e["t"] <= now]
    if len(full) > allowed:
        return (f"play weights: {len(full)} full-row exports since the play started (v{full[-1].get('step')}; bar "
                f"{allowed}): the play has not confirmed this generation")
    return None


def host_low(ev, need_gib, total=1024.0, floor=0.10):
    """Host RAM (c7b 24 Oct: the container thrashed at its 1,024 GiB limit at the step-1 checkpoint): the newest
    host_mem / step reading below the next checkpoint's host need x1.2 plus 10% of the limit."""
    rows = [(e["t"], e.get("avail_gib") if e.get("event") == "host_mem" else e.get("host_avail_gib")) for e in ev
            if e.get("event") in ("host_mem", "step")]
    rows = [(t, a) for t, a in rows if a is not None]
    if not rows:
        return None
    avail = max(rows)[1]
    bar = 1.2 * need_gib + floor * total
    if avail < bar:
        return f"host RAM: {avail:.0f} GiB available < {bar:.0f} (checkpoint need {need_gib:.0f} x1.2 + {floor:.0%} of {total:.0f})"
    return None


def step_stall(steps, now, k=2.5, floor=480.0):
    """c7f (24 Oct): an NCCL desync hung all ranks at 98-100% util for 11 min, so util rules saw nothing. No new step
    for k x the median step interval (at least ``floor`` s) after the second step is a stall."""
    if len(steps) < 2:
        return None
    gaps = [b["t"] - a["t"] for a, b in zip(steps, steps[1:])]
    bar = max(floor, k * statistics.median(gaps))
    if now - steps[-1]["t"] > bar:
        return f"step stall: no step for {now - steps[-1]['t']:.0f} s after step {steps[-1].get('step')} (> {bar:.0f} s = {k} x median step)"
    return None


def verdict(gpu, ev, now, pred, ref_dt, state, teacher_age=None, follow=None):
    """The stop reason for one read (None: keep going); ``state`` carries the low-throughput count."""
    steps = [e for e in ev if e.get("event") == "step"]
    if os.environ.get("STALL_K", "2.5") != "0" and (why := step_stall(steps, now, float(os.environ.get("STALL_K") or 2.5))):
        return why
    loaded = next((e for e in ev if e.get("event") == "loaded"), None)
    if loaded and pred.get("PRED_LOADED") and loaded["gpu_gib"] > 1.03 * pred["PRED_LOADED"]:
        return f"memory: loaded {loaded['gpu_gib']} GiB > 1.03 x predicted {pred['PRED_LOADED']}"
    for e in steps:
        p = pred.get("PRED_STEP1" if e is steps[0] else "PRED_PEAK")
        if p and (e.get("peak_gib") or 0) > 1.03 * p:
            return f"memory: step {e['step']} peak {e['peak_gib']} GiB > 1.03 x predicted {p}"
    if any(not e.get("finite", True) for e in steps):
        return "non-finite step"
    rk = [e.get("kd_reply") or e["kd"] for e in steps]
    gn = [e["grad_norm"] for e in steps]
    # every consecutive pair, not only the last two: a read sees 2-3 new steps, so a jump that stays up was compared
    # with an already-raised median one read later (backtest on cut1-v2, 19 Oct: grad norm x20 from step 9 missed)
    for j in range(3, len(steps)):
        base = statistics.median(rk[max(0, j - 6):j - 1])  # the (up to 5) steps before the pair
        if rk[j - 1] > 2 * base and rk[j] > 2 * base:
            return f"KL blow-up: reply KL {rk[j - 1]:.3f}, {rk[j]:.3f} (steps {j}, {j + 1}) > 2x the median of the steps before ({base:.3f})"
        g = gn[max(2, j - 6):j - 1] or gn[:j - 1]  # steps 1-2 carry the warm-start transient (warm1: 26.6, 85.3, then 2-5)
        g0 = statistics.median(g)
        if gn[j - 1] > 10 * g0 and gn[j] > 10 * g0:
            return f"divergence: grad norm {gn[j - 1]}, {gn[j]} (steps {j}, {j + 1}) > 10x the median of the steps before ({g0})"
    gmax = float(os.environ.get("GN_MAX", "2.0"))  # two consecutive steps above 2.0
    if gmax > 0:
        for j in range(1, len(gn)):
            if gn[j - 1] > gmax and gn[j] > gmax:
                return f"grad norm {gn[j - 1]}, {gn[j]} (steps {j}, {j + 1} of this read) above {gmax} twice"
    # SKIP_RULES=no_progress: a restart after that alert was read and the run kept (the other rules stay live)
    if len(rk) >= 10 and "no_progress" not in os.environ.get("SKIP_RULES", "") and sum(rk[8:10]) / 2 > 0.9 * sum(rk[:2]) / 2:
        return f"no progress: reply KL steps 9-10 {sum(rk[8:10]) / 2:.3f} not 10% below steps 1-2 {sum(rk[:2]) / 2:.3f}"
    # KL_BAR=A-B:X: pre-registered mid-run bar, mean reply KL of steps A-B <= X (cut 2 vs cut 1's curve on the same data)
    if os.environ.get("KL_BAR"):
        span, x = os.environ["KL_BAR"].split(":"); a, b = map(int, span.split("-"))
        if len(rk) >= b and sum(rk[a - 1:b]) / (b - a + 1) > float(x):
            return f"KL bar: reply KL steps {a}-{b} mean {sum(rk[a - 1:b]) / (b - a + 1):.3f} > {x}"
    # MID_RULES=1 (OPD round 1), on the trainer's "mid_read" events (5 held-out files each, the
    # step-0 read = D): teacher-H forward > MID_HFWD (1.10) x D; dh10 top-32 tail > MID_TAIL (2.0) x D; dh10 sampled
    # reverse not below the previous read on 2 consecutive reads
    if os.environ.get("MID_RULES") == "1":
        mids = [e for e in ev if e.get("event") == "mid_read"]
        if mids and mids[0].get("step") == 0:
            d0, hf, tl = mids[0], float(os.environ.get("MID_HFWD") or 1.10), float(os.environ.get("MID_TAIL") or 2.0)
            for m in mids[1:]:
                if d0.get("h_fwd") and m.get("h_fwd") is not None and m["h_fwd"] > hf * d0["h_fwd"]:
                    return f"mid read step {m['step']}: teacher-H forward {m['h_fwd']} > {hf} x D {d0['h_fwd']}"
                if d0.get("d_tail32") and m.get("d_tail32") is not None and m["d_tail32"] > tl * d0["d_tail32"]:
                    return f"mid read step {m['step']}: dh10 top-32 tail {m['d_tail32']} > {tl} x D {d0['d_tail32']}"
            sr = [m.get("d_sampled") for m in mids if m.get("d_sampled") is not None]
            for j in range(2, len(sr)):
                if sr[j - 1] >= sr[j - 2] and sr[j] >= sr[j - 1]:
                    return f"mid reads: dh10 sampled reverse stopped falling on 2 consecutive reads {sr[j - 2:j + 1]}"
    # damage (alerts only, DAMAGE=0 turns them off): dh10 forward reply KL over DMG_DFWD (1.05) x its
    # start on 2 consecutive reads (mid reads vs the step-0 mid read; end-of-round heldout2 vs heldout2_before), and the
    # general-text KL over DMG_GEN (1.20) x its start (general_probe before -> after / mid-read g_kd)
    if os.environ.get("DAMAGE", "1") == "1":
        why = damage(ev, float(os.environ.get("DMG_DFWD") or 1.05), float(os.environ.get("DMG_GEN") or 1.20))
        if why:
            return why
    # AGE_MED / AGE_MAX (streaming OPD, 22 Oct): data age of a step (its median / max updates behind the weights that
    # sampled it, the trainer's step event); alert on 2 consecutive steps over either
    if os.environ.get("AGE_MED") or os.environ.get("AGE_MAX"):
        am, ax = float(os.environ.get("AGE_MED") or 1e9), float(os.environ.get("AGE_MAX") or 1e9)
        aged = [e for e in steps if e.get("age_median") is not None]
        for x, y in zip(aged, aged[1:]):
            if (x["age_median"] > am and y["age_median"] > am) or ((x.get("age_max") or 0) > ax and (y.get("age_max") or 0) > ax):
                return (f"data age: steps {x['step']}, {y['step']} age median {x['age_median']}, {y['age_median']} "
                        f"(bar {am}), max {x.get('age_max')}, {y.get('age_max')} (bar {ax})")
    # WAIT_SHARE (OPD 3+1): trainer data wait over this share of the step time on 2 consecutive steps
    # from step 2 (step 1 waits on the teacher's first pieces); stream_wait_s is cumulative (train_export differences it)
    if os.environ.get("WAIT_SHARE"):
        ws = float(os.environ["WAIT_SHARE"])
        cum, per = 0.0, []
        for e in steps:
            w = e.get("stream_wait_s")
            if isinstance(w, (int, float)):
                if e.get("step", 0) >= 2 and e.get("seconds"):
                    per.append((e["step"], max(0.0, w - cum) / e["seconds"]))
                cum = w
        for (sx, fx), (sy, fy) in zip(per, per[1:]):
            if fx > ws and fy > ws:
                return f"data wait: steps {sx}, {sy} waited {fx:.1%}, {fy:.1%} of the step (bar {ws:.0%})"
    if follow is not None and os.environ.get("FOLLOW_LAG", "900") != "0":
        why = follow_lag(ev, follow, now, float(os.environ.get("FOLLOW_LAG") or 900))
        if why:
            return why
    skip = [e for e in ev if e.get("event") == "ckpt_skip"]
    if skip:  # heal_kd's checkpoint guard: host too low for the copy, no restart point at this step
        e = skip[-1]
        return f"checkpoint skipped at step {e.get('step')}: host avail {e.get('avail_gib')} < floor {e.get('floor_gib')} GiB"
    if os.environ.get("CKPT_NEED_GIB"):
        why = host_low(ev, float(os.environ["CKPT_NEED_GIB"]))
        if why:
            return why
    if follow is not None and os.environ.get("FULL_AFTER_START", "2") != "0":
        why = full_rows(ev, follow, now, int(os.environ.get("FULL_AFTER_START") or 2))
        if why:
            return why
    # MAX_SECONDS: the planned wall (a B300 run over 1 h needs explicit approval): alert once the run passes it
    first = min((g["epoch"] for g in gpu if g.get("epoch")), default=now)
    if os.environ.get("MAX_SECONDS") and now - first > float(os.environ["MAX_SECONDS"]):
        return f"over plan: {now - first:.0f} s > MAX_SECONDS {os.environ['MAX_SECONDS']}"
    # LOW_UTIL: trainer cards (0-1) or teacher cards (2-3) of a streaming round under LOW_UTIL %
    # (30) over the last LOW_MIN s (900), counted only after the start reads (the step-0 mid read, else step 1)
    if os.environ.get("LOW_UTIL"):
        lim, span = float(os.environ["LOW_UTIL"]), float(os.environ.get("LOW_MIN") or 900)
        marks = [e["t"] for e in ev if e.get("event") == "mid_read" and e.get("step") == 0] or [e["t"] for e in steps[:1]]
        if marks and now - marks[0] >= span:
            rows = [g for g in gpu if g.get("epoch") and max(marks[0], now - span) <= g["epoch"] <= now and len(g.get("util") or []) >= 2]
            if len(rows) >= span / 120:  # rows cover the window (sampler every 10 s; 60 s export buckets)
                for name, cards in (("trainer", (0, 1)), ("teacher", (2, 3))):
                    if any(len(g["util"]) <= max(cards) for g in rows):
                        continue
                    u = sum(sum(g["util"][c] for c in cards) / len(cards) for g in rows) / len(rows)
                    if u < lim:
                        return f"low util: {name} cards {u:.1f}% < {lim:.0f}% over the last {span:.0f} s (after the start reads)"
    gpu_age = now - max((g["epoch"] for g in gpu if g.get("epoch")), default=now)
    phase = gpu[-1].get("phase") if gpu else None
    if phase == "train":
        prog_age = now - max(e["t"] for e in ev) if ev else gpu_age
    else:
        prog_age = teacher_age if teacher_age is not None else gpu_age
    if gpu_age > SILENCE and prog_age > SILENCE:
        return f"silence: no gpu row for {gpu_age:.0f} s and no progress record for {prog_age:.0f} s"
    if len(steps) >= 3:
        lo = max(steps[1]["t"], now - 300)
        rows = [g for g in gpu if g.get("epoch") and lo <= g["epoch"] <= now]
        util = sum(sum(g["util"]) / len(g["util"]) for g in rows) / len(rows) if rows else None
        times = [b["t"] - a["t"] for a, b in zip(steps, steps[1:]) if b["t"] > now - 300]
        ref = ref_dt or (statistics.median([b["t"] - a["t"] for a, b in zip(steps[1:4], steps[2:5])])
                         if len(steps) >= 5 else None)
        open_gap = now - steps[-1]["t"]  # a read without a finished step counts with the time since the last one
        slow_time = statistics.median(times) if times else open_gap
        slow = bool(ref and slow_time > 1.5 * ref)
        state["low"] = state.get("low", 0) + 1 if (util is not None and util < 70 and slow) else 0
        if state["low"] >= 3:
            return f"throughput: util {util:.1f}% < 70 and step time {slow_time:.0f} s > 1.5 x {ref:.0f} on 3 reads"
    return None


def stop_app(app):
    """Stops a Modal app non-interactively and verifies its containers are gone (True) within 5 min."""
    subprocess.run([MV, "app", "stop", "-y", app], capture_output=True, stdin=subprocess.DEVNULL)
    for _ in range(20):
        listed = subprocess.run([MV, "container", "list", "--json"], capture_output=True, text=True,
                                stdin=subprocess.DEVNULL).stdout
        if app not in listed:
            print("verified stopped", flush=True)
            return True
        time.sleep(15)
    print("WARNING: container still listed", flush=True)
    return False


def preds():
    return {k: float(os.environ[k]) for k in ("PRED_LOADED", "PRED_STEP1", "PRED_PEAK") if os.environ.get(k)}


def backtest(folder):
    folder = Path(folder)
    gpu = [json.loads(l) for l in open(folder / "gpu.jsonl") if l.strip()]
    ev = [json.loads(l) for l in open(folder / "events.jsonl") if l.strip()] if (folder / "events.jsonl").exists() else []
    fl = folder / "weights-follow.jsonl"  # the play's follower log (live: PLAY_RUN)
    follow = [json.loads(l) for l in open(fl) if l.strip()] if fl.exists() else None
    last = max(g["epoch"] for g in gpu if g.get("epoch"))
    end = float((folder / "end").read_text()) if (folder / "end").exists() else last
    start = min(g["epoch"] for g in gpu if g.get("epoch"))
    state, now = {}, start + 300
    ref = float(os.environ["REF_DT"]) if os.environ.get("REF_DT") else None
    while now <= end:
        why = verdict([g for g in gpu if g.get("epoch", 0) <= now], [e for e in ev if e["t"] <= now], now, preds(), ref,
                      state, follow=follow)
        if why:
            return f"ALERT at +{now - start:.0f} s: {why}"
        now += 300
    return f"quiet through +{end - start:.0f} s"


def play_follow_rows(play, D):
    """The play's weights-follow.jsonl from the bucket (None if unreadable this pass: the rule skips a pass, never
    fires on a failed read)."""
    sys.path.insert(0, "./.claude/worktrees/nebius-opd/scripts")
    try:
        import nebius_validation as nv
        c = nv.Cloud(nv.load_config())
        out = D / "weights-follow.jsonl"
        r = c.rclone("copyto", c.key(nv.RUNS_PREFIX, "validation", play,
                                     "arena/working/trials/duck-transfer/weights-follow.jsonl"), str(out),
                     check=False, timeout=120)
        return [json.loads(l) for l in open(out) if l.strip()] if r.returncode == 0 else None
    except Exception as exc:  # noqa: BLE001
        print("play follow read failed:", exc, flush=True)
        return None


def app_live(app):
    """The Modal app is still up (ephemeral/deployed with tasks); unknown counts as live."""
    r = subprocess.run([MV, "app", "list", "--json"], capture_output=True, text=True)
    try:
        rows = [a for a in json.loads(r.stdout) if a.get("app_id") == app]
    except ValueError:
        return True
    return bool(rows) and "stopped" not in str(rows[0].get("state", "")) and int(rows[0].get("tasks") or 0) > 0


def live(run, app):
    D = Path(f"~/arc-sft-data/mon/{run}"); D.mkdir(parents=True, exist_ok=True)

    def get(rel, parse=True):
        out = D / rel.replace("/", "_")
        ok = subprocess.run([MV, "volume", "get", "--force", "arc-prolong-runs", f"/validation/{run}/{rel}", str(out)],
                            capture_output=True).returncode == 0
        if not parse:
            return out.stat().st_size if ok else None
        return [json.loads(l) for l in open(out) if l.strip()] if ok else []

    def dash(status="live", alert=None):  # live training dashboard: never let it break the watcher
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            import train_export
            train_export.export(run, status, alert)
        except Exception as exc:  # noqa: BLE001
            print("dashboard export failed:", exc, flush=True)

    def stop(why):  # watchers alert and exit (the agent inspects and decides); no stop here
        print(time.strftime("%H:%M", time.gmtime()), "ALERT", why, flush=True)
        dash(alert=why)
        sys.exit(1)

    ref = float(os.environ["REF_DT"]) if os.environ.get("REF_DT") else None
    state, tsize, tchange = {}, None, time.time()
    while True:
        ident = D / "id.json"
        if subprocess.run([MV, "volume", "get", "--force", "arc-prolong-runs", f"/validation/{run}/modal-identity.json",
                           str(ident)], capture_output=True).returncode == 0 and not app_live(app):
            # c7b (24 Oct): the identity file is written at step 1 too, so it alone does not mean the round ended
            get("train/events.jsonl")
            dash("ended")
            print(time.strftime("%H:%M", time.gmtime()), "ended:", ident.read_text()[:400], flush=True); sys.exit(0)
        gpu = get("gpu.jsonl")
        # SMOKE=1: the B300 smoke's three trainer processes (smoke-a, -b, -c) in order instead of train/
        ev = (get("smoke-a/events.jsonl") + get("smoke-b/events.jsonl") + get("smoke-c/events.jsonl")
              if os.environ.get("SMOKE") else get("train/events.jsonl"))
        size = get(os.environ.get("TEACHER_LOG") or "teacher0.log", parse=False)
        follow = play_follow_rows(os.environ["PLAY_RUN"], D) if os.environ.get("PLAY_RUN") else None
        if follow is not None:
            last = next((r for r in reversed(follow) if r.get("kind") == "applied"), {})
            exp = next((e for e in reversed(ev) if e.get("event") == "delta_export" and e.get("tensors")), {})
            print(time.strftime("%H:%M", time.gmtime()), f"play applied_version {last.get('version')} "
                  f"(apply lag {last.get('lag_seconds')} s) exported {exp.get('step')}", flush=True)
        if size != tsize:
            tsize, tchange = size, time.time()
        steps = [e for e in ev if e.get("event") == "step"]
        s = steps[-1] if steps else {}
        rows = [g for g in gpu if g.get("epoch") and g["epoch"] > time.time() - 300]
        util = round(sum(sum(g["util"]) / len(g["util"]) for g in rows) / len(rows), 1) if rows else None
        mem = max((max(g["mem_gib"]) for g in rows), default=None)
        print(time.strftime("%H:%M", time.gmtime()), f"phase {gpu[-1].get('phase') if gpu else None} step {s.get('step')} "
              f"kd {s.get('kd')} reply {s.get('kd_reply')} mtp {s.get('kd_mtp')} gn {s.get('grad_norm')} util5m {util} "
              f"mem {mem} peak {s.get('peak_gib')}", flush=True)
        dash()
        why = verdict(gpu, ev, time.time(), preds(), ref, state, teacher_age=time.time() - tchange, follow=follow)
        if why:
            stop(why)
        time.sleep(300)


if __name__ == "__main__":
    if sys.argv[1] == "--stop-test":  # teardown check on a throwaway app: opd_watch.py --stop-test APP_ID
        sys.exit(0 if stop_app(sys.argv[2]) else 2)
    if sys.argv[1] == "--backtest":
        print(sys.argv[3] if len(sys.argv) > 3 else sys.argv[2], "->", backtest(sys.argv[2]))
    else:
        live(sys.argv[1], sys.argv[2])
