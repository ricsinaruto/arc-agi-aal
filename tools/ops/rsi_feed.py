"""Live feed of RSI loop runs (rsi-loop-*) for mon.py and the dashboard card, and the check of the live numbers against
what a working loop should show by then (docs/rsi.md; table EXPECT below). Each pass reads three small files of the run's
checkpoint in the bucket: state.json, events.jsonl, heldout/results.jsonl. `python rsi_feed.py RUN` prints the row and its
deviations; `python rsi_feed.py check RUN [seen-file]` prints only deviations not yet in seen-file and exits 3 when any."""
import json, pathlib, subprocess, sys, time
import paths
BUCKET = "nebius-arc:arc-runs-bucket/runs/validation/"
CKPT = "arena/working/trials/level-bench/rsi"
OUT = paths.SCRATCH / "mon"
PLAN_LOOP_H = 100.0       # round 1 as approved: 100 loop GPU-h in ~4 legs
USD_PER_MACHINE_H = 1.827  # RTX PRO 6000 on-demand + disk (launcher rate)
SLICE_S = 32400 / 110     # docs/rsi.md 3: Kaggle runtime / 110 per game
# What a working loop shows by each loop GPU-h mark. A game's slice is SLICE_S of GPU time (the GPU shared by the games
# in play), so at most ~12 games finish per loop GPU-h: a 15-game minibatch costs ~1.2 loop GPU-h, the 54-game base pool
# ~4.4 (docs/rsi.md 3-4; held-out A every 10 loop GPU-h when the best changed, A+B with 3 seeds every 50).
EXPECT = [
    (2, {"proposer_sessions": 1, "candidates": 1, "gen0_pool_games": 10}),
    (6, {"candidates": 4, "evaluated": 2, "practice_rows": 5}),
    (12, {"candidates": 8, "evaluated": 4, "best_beats_gen0": True}),
    (24, {"candidates": 15, "evaluated": 8, "full_pool": 1, "heldout_a_runs": 2}),
    (50, {"candidates": 30, "evaluated": 15, "full_pool": 2, "ab_report": 1}),
]
ALWAYS = "isolation ok; no egress or non-Flash calls; >= 8 games per loop GPU-h; loop GPU-h >= 70% of machine hours; " \
         "a proposer session ends at least every 2 loop GPU-h; < 3 stack reverts per 12 loop GPU-h; < 10 policy fallbacks; best never below gen 0 on its paired pool games; spend on plan (<= per loop GPU-h)"


def cat(path, timeout=120):
    r = subprocess.run(["rclone", "cat", BUCKET + path], capture_output=True, timeout=timeout)
    return r.stdout.decode(errors="replace") if r.returncode == 0 else ""


def jsonl(text):
    out = []
    for line in text.splitlines():
        try: out.append(json.loads(line))
        except ValueError: pass
    return out


def paired(a, b, min_games):
    shared = sorted(set(a) & set(b))
    if len(shared) < min_games: return None
    return {"n": len(shared), "gen0": sum(a[g] for g in shared) / len(shared), "cand": sum(b[g] for g in shared) / len(shared),
            "effect": sum(b[g] - a[g] for g in shared) / len(shared)}


def row(run):
    base = f"{run}/{CKPT}"
    raw = cat(f"{base}/state.json")
    if not raw: return None
    s = json.loads(raw)
    ev = jsonl(cat(f"{base}/events.jsonl", timeout=300))
    held = jsonl(cat(f"{base}/heldout/results.jsonl"))
    m = s.get("meters", {})
    loop_h, held_h, mach_h = m.get("loop_s", 0) / 3600, m.get("heldout_s", 0) / 3600, m.get("machine_s", 0) / 3600
    cands = s.get("candidates", {})
    added = {a.get("game") for a in s.get("added_games") or []}
    basepool = [g for g in s.get("pool_games", []) if g not in added]
    pool_of = lambda c: {g: v for g, v in (cands.get(c, {}).get("pool") or {}).items() if g in basepool}
    g0 = pool_of("c0000")
    best = None
    for c in cands:
        if c == "c0000": continue
        p = paired(g0, pool_of(c), 15)
        if p and (best is None or p["effect"] > best["effect"]): best = {**p, "id": c}
    rows = s.get("rows", [])
    loop_rows = [r for r in rows if r.get("kind") in ("pool", "practice", "smoke")]
    prac = [r for r in rows if r.get("kind") == "practice"]
    names = [e.get("event") for e in ev]
    hist = s.get("stack", {}).get("history", [])
    sessions = s.get("proposer", {}).get("sessions", [])
    probe_egress = (s.get("egress_from_isolation_probes") or [{}])[-1] or {}
    egress = {u: n - probe_egress.get(u, 0) for u, n in (s.get("frontier", {}).get("egress") or {}).items()
              if u != "rsiserve" and n - probe_egress.get(u, 0) > 0}
    iso = next((e for e in reversed(ev) if e.get("event") == "rsi_isolation"), None)
    ha = {}
    for r in held:
        if r.get("kind") == "heldout_a" and r.get("score") is not None:
            ha.setdefault((r.get("candidate"), r.get("run")), []).append(r["score"])
    last_a = {}
    for (c, runid), v in ha.items():
        last_a[c] = (runid, sum(v) / len(v), len(v))  # runs are in time order: the last per candidate wins
    now = time.time()
    recent = [e for e in ev if e.get("event") == "rsi_game" and e.get("time", 0) > now - 3600]
    out = {"run": run, "rsi": True, "loop_h": loop_h, "heldout_h": held_h, "machine_h": mach_h, "switch_h": m.get("switch_s", 0) / 3600,
           "usd": mach_h * USD_PER_MACHINE_H, "plan_h": PLAN_LOOP_H,
           "candidates": sum(c != "c0000" for c in cands), "evaluated": sum(1 for c in cands if c != "c0000" and pool_of(c)),
           "full_pool": sum(1 for c in cands if c != "c0000" and basepool and len(pool_of(c)) >= len(basepool)),
           "rejected": sum(1 for v in cands.values() if v.get("status") == "rejected"),
           "adopted": sum(1 for h in hist if h.get("purpose") == "proposer" and h.get("ok")),
           "reverted": names.count("rsi_stack_revert"), "gen0_pool_games": len(g0), "gen0_pool": sum(g0.values()) / len(g0) if g0 else None,
           "best": best, "policy_best": s.get("policy_best"), "heldout_a": {c: {"run": v[0], "rhae": v[1], "games": v[2]} for c, v in last_a.items()},
           "heldout_a_runs": sum(1 for r in s.get("heldout", {}).get("runs", []) if "heldout_a" in r.get("sets", []) and not r.get("open")),
           "ab_report": sum(1 for p in s.get("heldout", {}).get("points", []) if p.get("report")),
           "practice_rows": len(prac), "practice_win": sum(r.get("outcome") == "win" for r in prac) / len(prac) if prac else None,
           "games": len(loop_rows), "games_per_h": len(loop_rows) / loop_h if loop_h > 0.25 else None, "games_last_h": len(recent),
           "proposer_sessions": len(sessions), "sessions_no_model": sum(1 for x in sessions if not x.get("requests")),
           "last_session_age_h": (now - max(x.get("ended") or 0 for x in sessions)) / 3600 if sessions else None,
           "policy_fallback": (s.get("policy_fallback") or {}).get("count", 0), "policy_failed": names.count("rsi_policy_failed"),
           "self_updates": names.count("rsi_self_update"),
           "isolation_ok": None if iso is None else bool(iso.get("ok")), "egress": egress,
           "non_flash": s.get("frontier", {}).get("non_flash", 0), "unprivileged_ok": (s.get("unprivileged_serving") or {}).get("ok"),
           "problems": [e.get("event") for e in ev if e.get("event") in ("rsi_no_stack", "rsi_server_down", "rsi_stop_launching")],
           "live_stack": s.get("stack", {}).get("live"), "utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime())}
    out["deviations"] = deviations(out)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{run}.rsi.jsonl").open("a") as f:  # history for the card's sparkline
        f.write(json.dumps({"t": now, "loop_h": round(loop_h, 3), "effect": best and round(best["effect"], 3)}) + "\n")
    return out


def deviations(r):
    """Misses against EXPECT at the marks already passed, and the standing rules (ALWAYS)."""
    d, h = [], r["loop_h"]
    if r["isolation_ok"] is False: d.append("isolation self-test failed")
    if r["egress"]: d.append(f"egress from sandboxes: {r['egress']}")
    if r["non_flash"]: d.append(f"{r['non_flash']} non-Flash model calls refused")
    if r["problems"]: d.append("loop problems: " + ", ".join(sorted(set(r["problems"]))))
    if r["unprivileged_ok"] is False: d.append("unprivileged serving probe failed: proposer stacks cannot be served")
    for mark, want in EXPECT:
        if h < mark: break
        for k, v in want.items():
            got = (r["best"] is not None and r["best"]["effect"] > 0) if k == "best_beats_gen0" else r.get(k) or 0
            if (got is not True) if v is True else got < v:
                d.append(f"by {mark} loop GPU-h expected {k} >= {v}, have {got}")
    if r["games_per_h"] is not None and h >= 2 and r["games_per_h"] < 8: d.append(f"{r['games_per_h']:.1f} games per loop GPU-h (< 8)")
    if r["machine_h"] >= 2 and r["loop_h"] < 0.7 * r["machine_h"]:
        d.append(f"loop GPU-h {r['loop_h']:.1f} of {r['machine_h']:.1f} machine h (< 70%)")
    if h >= 2 and (r["last_session_age_h"] is None or r["last_session_age_h"] > 2): d.append("proposer stalled: no session ended in 2 h")
    if r["proposer_sessions"] >= 3 and r["sessions_no_model"] * 2 > r["proposer_sessions"]: d.append("most proposer sessions made no model request")
    if r["reverted"] >= 3 and r["reverted"] > h / 4: d.append(f"{r['reverted']} stack reverts in {h:.1f} loop GPU-h")
    if r["policy_fallback"] >= 10: d.append(f"the live policy keeps failing: {r['policy_fallback']} fallbacks to gen 0's")
    if r["best"] is not None and r["best"]["effect"] < 0 and h >= 12: d.append(f"best candidate below gen 0 on the pool ({r['best']['effect']:+.1f})")
    if h >= 2 and r["usd"] > 3.65 * h: d.append(f"spend ${r['usd']:.0f} for {h:.1f} loop GPU-h (plan per loop GPU-h)")
    return d


def text(r):
    """One line for the run row: loop GPU-h of plan, candidates, best vs gen 0, games/h, deviations."""
    b = [f"{r['loop_h']:.1f} of {r['plan_h']:.0f} loop GPU-h", f"{r['candidates']} candidates ({r['evaluated']} scored, {r['adopted']} stacks adopted, {r['reverted']} reverted)"]
    if r["best"]: b.append(f"best {r['best']['id']} {r['best']['effect']:+.1f} RHAE vs gen 0 on {r['best']['n']} pool games")
    elif r["gen0_pool"] is not None: b.append(f"gen 0 pool RHAE {r['gen0_pool']:.1f} on {r['gen0_pool_games']} games")
    if r["games_per_h"]: b.append(f"{r['games_per_h']:.0f} games/loop GPU-h")
    if r["deviations"]: b.append(f"{len(r['deviations'])} deviation(s): {r['deviations'][0]}")
    return " · ".join(b)


def bits(r):
    """Card lines (status_html): headline numbers, then held-out A, practice, proposer, safety, deviations."""
    v = lambda x, f="{:.1f}": "–" if x is None else f.format(x)
    m = [f"<div class='v'>{r['loop_h']:.1f} <small>of {r['plan_h']:.0f} loop GPU-h · held-out {r['heldout_h']:.1f} h · ${r['usd']:.0f}</small></div>"]
    m.append(f"<div class='m'>candidates <b>{r['candidates']}</b> · scored <b>{r['evaluated']}</b> · full pool <b>{r['full_pool']}</b> · stacks adopted <b>{r['adopted']}</b> · reverted <b>{r['reverted']}</b> · rejected {r['rejected']}</div>")
    if r["best"]:
        m.append(f"<div class='m'>pool RHAE, paired: best <b>{r['best']['id']}</b> {r['best']['cand']:.1f} vs gen 0 {r['best']['gen0']:.1f} (<b>{r['best']['effect']:+.1f}</b>, {r['best']['n']} games)</div>")
    else:
        m.append(f"<div class='m'>pool RHAE: gen 0 <b>{v(r['gen0_pool'])}</b> on {r['gen0_pool_games']} games · no candidate on 15+ shared games yet</div>")
    if r["heldout_a"]:
        m.append("<div class='m'>held-out A RHAE (as scored, report only): " + " · ".join(f"{c} <b>{x['rhae']:.1f}</b> ({x['games']} games)" for c, x in sorted(r["heldout_a"].items())) + "</div>")
    m.append(f"<div class='m'>practice clear rate <b>{v(None if r['practice_win'] is None else 100 * r['practice_win'], '{:.0f}%')}</b> ({r['practice_rows']} games) · "
             f"<b>{v(r['games_per_h'], '{:.0f}')}</b> games/loop GPU-h · {r['games_last_h']} in the last hour</div>")
    m.append(f"<div class='m'>proposer sessions <b>{r['proposer_sessions']}</b> ({r['sessions_no_model']} without model calls) · policy fallbacks {r['policy_fallback']} · "
             f"self-updates {r['self_updates']} · isolation <b>{'ok' if r['isolation_ok'] else 'FAILED' if r['isolation_ok'] is False else '–'}</b> · egress {sum(r['egress'].values()) if r['egress'] else 0} · "
             f"proposer stacks {'ok' if r['unprivileged_ok'] else 'OFF' if r['unprivileged_ok'] is False else '–'}</div>")
    if r["deviations"]:
        m.append("<div class='m' style='color:#e03131'><b>vs expectations:</b> " + "; ".join(r["deviations"][:4]) + "</div>")
    else:
        m.append("<div class='m'>on track against the expectations for this point</div>")
    m.append(f"<div class='m'>read {r['utc'][11:]} UTC</div>")
    return m


if __name__ == "__main__":
    if sys.argv[1] == "check":
        r = row(sys.argv[2])
        if r is None: print("no state yet"); sys.exit(0)
        seen_path = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else None
        seen = set(json.loads(seen_path.read_text())) if seen_path and seen_path.exists() else set()
        new = [x for x in r["deviations"] if x.split(" (")[0].split(":")[0] not in seen]
        print(text(r))
        if seen_path: seen_path.write_text(json.dumps(sorted(seen | {x.split(" (")[0].split(":")[0] for x in r["deviations"]})))
        if new: print("DEVIATION", json.dumps(new)); sys.exit(3)
    else:
        r = row(sys.argv[1]); print(json.dumps(r, indent=1)[:4000] if r else "no state yet")
