"""Run monitor: reconciles docs/modal-queue.json with the machines actually running (reconcile.py; drops finished runs, settling
Nebius ledger rows), reads the live feed of every live run (Modal volume or Nebius bucket), then writes MODAL_STATUS.md / .html and
the arm status pages (banner.py). Usage: mon.py [extra run id]... ; run every 5 minutes by dash_loop.sh."""
import shutil
import json, os, re, subprocess, sys, time, datetime, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import paths
ROOT = str(paths.ROOT_PATH)
OUT = paths.SCRATCH / "mon"
OUT.mkdir(parents=True, exist_ok=True)

NEBIUS = set()  # filled after the queue is reconciled

def nebius_runs():
    q = paths.read_json(pathlib.Path(ROOT, "docs/modal-queue.json"))
    return {e["run"] for e in q.get("live", []) + q.get("queued", []) if e.get("platform") == "nebius"} | {
        p.name[7:-5] for p in pathlib.Path(ROOT, "docs/validation-nebius").glob("nebius-*.json")}

def get(remote, local):
    run = remote.split("/")[1]
    if run in NEBIUS:
        r = subprocess.run(["rclone", "copyto", "nebius-arc:arc-runs-bucket/runs/" + remote, str(local)], capture_output=True, text=True, timeout=300)
        return r.returncode == 0 and pathlib.Path(local).exists()
    vol = "arc-prolong-runs"
    if run.startswith("ds17-"):  # DeepSeek benchmark arm: its own volume, live files under results/<run>/live/
        vol, remote = "arc-ds41-1017", remote.replace(f"validation/{run}/arena/live/", f"results/{run}/live/")
    r = subprocess.run(["uv", "run", "--project", ROOT, "modal", "volume", "get", vol, remote, str(local), "--force"],
                       capture_output=True, text=True, timeout=300)
    return r.returncode == 0

REQUEST_LOGS = ("arena/harness/requests.jsonl", "arena/requests.jsonl")  # his harness: one row per chat completion
SGLANG_METRICS = "arena/working/trials/duck-transfer/vllm-metrics.txt"  # v8.2 arena on SGLang: raw /metrics scrape

def sglang_hit(run, eng):
    """Prefix hits (%) of runs whose server log has no vLLM hit-rate line (SGLang, or his harness on vLLM): the last
    30 min of the per-request cached/prompt tokens, else the SGLang counters' 30-min delta (snapshots kept in OUT),
    else the run's own cumulative ``prefix_hit_total``."""
    if run not in NEBIUS:
        return None
    base = f"nebius-arc:arc-runs-bucket/runs/validation/{run}/"
    for path in REQUEST_LOGS:
        r = subprocess.run(["rclone", "cat", "--tail", "4000000", base + path], capture_output=True, text=True, timeout=300)
        rows = []
        for x in r.stdout.splitlines()[1:]:
            try:
                rows.append(json.loads(x))
            except ValueError:
                pass
        rows = [x for x in rows if x.get("epoch") and x.get("prompt_tokens")]
        if rows:
            recent = [x for x in rows if x["epoch"] >= rows[-1]["epoch"] - 1800]
            prompt = sum(x["prompt_tokens"] for x in recent)
            if prompt:
                return 100 * sum(x.get("cached_tokens") or 0 for x in recent) / prompt
    r = subprocess.run(["rclone", "cat", base + SGLANG_METRICS], capture_output=True, text=True, timeout=300)
    tot = {"prompt": 0.0, "cached": 0.0}
    for name, key in (("sglang:prompt_tokens_total", "prompt"), ("sglang:cached_tokens_total", "cached")):
        for m in re.finditer(rf"^{name}(?:{{[^}}]*}})?\s+([-+\d.eE]+)", r.stdout, re.M):
            tot[key] += float(m.group(1))
    if tot["prompt"]:
        hist = OUT / f"{run}.hits.jsonl"
        with open(hist, "a") as fh:
            fh.write(json.dumps({"t": time.time(), **tot}) + "\n")
        snaps = [json.loads(x) for x in hist.read_text().splitlines() if x.strip()]
        old = [x for x in snaps if x["t"] <= time.time() - 1500]
        a = old[-1] if old else snaps[0]
        dp = tot["prompt"] - a["prompt"]
        return 100 * (tot["cached"] - a["cached"]) / dp if dp > 0 else 100 * tot["cached"] / tot["prompt"]
    if eng.get("prefix_hit_total") is not None:
        return 100 * eng["prefix_hit_total"]
    return None

def sglang_kv(run):
    """KV cache use (0..1) of an SGLang run without vLLM's gauge: sglang:token_usage from the dfr-exact monitor's
    metrics.jsonl (dfrx), else from the arena's raw /metrics scrape (v8.2 on his serving)."""
    if run not in NEBIUS:
        return None
    base = f"nebius-arc:arc-runs-bucket/runs/validation/{run}/"
    r = subprocess.run(["rclone", "cat", "--tail", "200000", base + "arena/metrics.jsonl"], capture_output=True, text=True, timeout=300)
    for x in reversed(r.stdout.splitlines()[1:]):
        try:
            m = json.loads(x).get("m") or {}
        except ValueError:
            continue
        if m.get("sglang:token_usage") is not None:
            return float(m["sglang:token_usage"])
    r = subprocess.run(["rclone", "cat", base + SGLANG_METRICS], capture_output=True, text=True, timeout=300)
    m = re.search(r"^sglang:token_usage(?:{[^}]*})?\s+([-+\d.eE]+)", r.stdout, re.M)
    return float(m.group(1)) if m else None

def row(run):
    base = f"validation/{run}/arena/live/"
    pj, pl, st = OUT / f"{run}.json", OUT / f"{run}.jsonl", OUT / f"{run}.tail"
    if not get(base + "progress.json", pj):
        base = f"throughput/{run}/live/"  # the throughput arm's Modal play runner (BF16 / NVFP4 plays) keeps its arena feed here
        if not get(base + "progress.json", pj):
            base = f"validation/{run}/live/"  # throughput reference reads (mm_arena / sage_arena runners)
            if not get(base + "progress.json", pj):
                return {"run": run, "state": "no live progress yet"}
    get(base + "progress.jsonl", pl); get(base + "server-tail.log", st)
    d = json.loads(pj.read_text())
    t0 = None
    if base == f"validation/{run}/live/" and not d.get("epoch"):
        try:  # reference runners write no epoch: the volume's progress.json mtime (minute resolution) less its elapsed
            r = subprocess.run(["uv", "run", "--project", ROOT, "modal", "volume", "ls", "--json", "arc-prolong-runs", base], capture_output=True, text=True, timeout=180)
            x = [x for x in json.loads(r.stdout) if (x.get("filename") or x.get("Filename") or "").endswith("/progress.json")][0]
            m = x.get("created_modified") or x.get("Created/Modified")  # key spelling differs across modal versions
            t0 = time.mktime(time.strptime(m[:16], "%Y-%m-%d %H:%M")) + 30 - (d.get("elapsed_seconds") or 0)
        except Exception:
            t0 = None
    elif base.startswith("throughput/") and get(base + "container.json", OUT / f"{run}.cont"):
        try:
            t0 = json.loads((OUT / f"{run}.cont").read_text()).get("harness_started")
        except Exception:
            t0 = None
    if t0:
        try:  # its arena rows carry no epoch: stamp them from the runner's harness start
            if t0:
                if not d.get("epoch"): d["epoch"] = t0 + (d.get("elapsed_seconds") or 0)
                keep = []
                for x in pl.read_text().splitlines():
                    try: y = json.loads(x)
                    except Exception: continue
                    if y.get("elapsed_seconds") is not None: keep.append(json.dumps({**y, "epoch": y.get("epoch") or t0 + y["elapsed_seconds"]}))
                pl.write_text("\n".join(keep) + "\n")
        except Exception:
            pass
    live = d.get("live") or {}; eng = live.get("engine") or {}
    sc = live.get("score") or {}
    tok = eng.get("generated_tokens") or 0
    el = d.get("elapsed_seconds") or 0
    recent, mean, containers, restarts, play = None, eng.get("tokens_per_second_mean"), 1, d.get("restarts"), d.get("play_seconds")
    try:
        import compare_runs
        rows = []
        for x in pl.read_text().splitlines():
            try: rows.append(json.loads(x))
            except Exception: pass
        sti = compare_runs.stitch(rows)
        if sti:  # cumulative over engine restarts and containers (a preempted job starts its counters at 0)
            last = sti[-1]; containers, restarts = last[7], last[8]
            if containers > 1 or last[8]:
                tok, el = last[1], last[0]
                if last[2] is not None: sc = {"score": last[2], "at_elapsed_seconds": last[3]}
                mean = tok / last[6] if last[6] else tok / el if el else mean
            pts = [(p[6] if p[6] is not None else p[0], p[1]) for p in sti if p[1]]  # play time where the run reports it
            now = pts[-1][0] if pts else 0
            old = [p for p in pts if p[0] <= now - 1800]
            if old and pts and pts[-1][0] > old[-1][0]:
                a, b = old[-1], pts[-1]
                recent = (b[1] - a[1]) / (b[0] - a[0])
    except Exception:
        pass
    hit = None
    try:
        hs = re.findall(r"Prefix cache hit rate: ([\d.]+)%(?:, External prefix cache hit rate: ([\d.]+)%)?", st.read_text())
        if hs:  # host-RAM (external) hits count: the external rate is over the tokens the GPU cache missed
            g, e = float(hs[-1][0]), float(hs[-1][1] or 0)
            hit = 100 - (100 - g) * (100 - e) / 100
    except Exception:
        pass
    seqs = None  # the server's max_num_seqs, from the launched bundle's config (the run's study.json lands only at the end)
    try:
        rec = next(iter(pathlib.Path(ROOT, "docs").glob(f"validation-*/*{run}.json")))
        cfg = json.loads((pathlib.Path(json.loads(rec.read_text())["bundle"]) / "configs/comparison.json").read_text())
        seqs = (cfg.get("server") or {}).get("max_num_seqs")
    except Exception:
        pass
    run30 = None  # every serving-run card shows the stream cap and the 30-min mean of running streams
    try:
        ps = [json.loads(x) for x in pl.read_text().splitlines() if x.strip()]
        ps = [p for p in ps if p.get("epoch") and ((p.get("live") or {}).get("engine") or {}).get("running") is not None]
        if ps:
            win = [p["live"]["engine"]["running"] for p in ps if p["epoch"] >= ps[-1]["epoch"] - 1800]
            run30 = sum(win) / len(win)
    except Exception:
        pass
    if recent is None: recent = eng.get("tokens_per_second_30m")
    if hit is None and eng.get("prefix_hit_30m") is not None: hit = eng["prefix_hit_30m"] * 100
    if hit is None:
        try:
            hit = sglang_hit(run, eng)
        except Exception:
            pass
    if eng.get("kv_cache_fraction") is None:
        try:
            eng["kv_cache_fraction"] = sglang_kv(run)
        except Exception:
            pass
    if not d.get("epoch"): d["epoch"] = sc.get("at_epoch") or 0
    gate = d.get("gate") or {}; pr = d.get("priority") or {}
    return {"run": run, "age_s": round(time.time() - (d.get("epoch") or 0)), "min": el / 60, "score": sc.get("score"),
            "score_lag_s": el - (sc.get("at_elapsed_seconds") or el), "tokM": tok / 1e6,
            "ppM": (sc.get("score") or 0) / (tok / 1e6) if tok else None, "mean": mean, "containers": containers,
            "restarts": restarts, "play_min": play / 60 if play else None,
            "last30": recent, "hit": hit, "kv": eng.get("kv_cache_fraction"),
            "streams": f"{pr.get('slots') or len(d.get('active') or []) or '?'}/{eng.get('running')}/{gate.get('waiting')}",
            "cap": min([x for x in (pr.get("slots"), seqs) if x] or [None]), "run30": run30,
            "preempt": eng.get("preemptions"), "errors": live.get("errors"), "loop": (d.get("loop_rule") or {}).get("fires"),
            "rss": (live.get("host") or {}).get("vllm_rss_gib"), "phase": d.get("phase"), "finished": d.get("finished"), "budget": d.get("generation_budget"),
            "hostline": hostline(live)}

def arc2_row(run):
    """ARC-AGI-2 roster trial on Nebius: the live row (arena/live/arc2-<trial>.json, synced every minute), else the
    trial's progress.json once synced (at the end), else why there is none."""
    base = f"nebius-arc:arc-runs-bucket/runs/validation/{run}/arena/"
    r = subprocess.run(["rclone", "lsf", "--include", "arc2-*.json", base + "live"], capture_output=True, text=True, timeout=120)
    f = next(iter(r.stdout.split()), None)
    if f:
        d = json.loads(subprocess.run(["rclone", "cat", base + "live/" + f], capture_output=True, text=True, timeout=120).stdout)
        fin, n, el = d.get("final") or {}, d.get("test_outputs") or 0, d.get("elapsed_seconds") or 0
        pct, tok = fin.get("score_outputs"), sum((d.get("arm_tokens") or {}).values())
        return {"run": run, "arc2": True, "pct": pct, "right": round(pct * n / 100) if pct is not None and n else None, "of": n,
                "tokM": tok / 1e6, "tps": tok / el if el else None, "play_s": el}
    for sub in ("trials", "working/trials"):
        r = subprocess.run(["rclone", "lsf", "-R", "--max-depth", "2", "--include", "progress.json", base + sub], capture_output=True, text=True, timeout=120)
        f = next((x for x in r.stdout.split() if x.endswith("progress.json")), None)
        if not f: continue
        d = json.loads(subprocess.run(["rclone", "cat", base + sub + "/" + f], capture_output=True, text=True, timeout=120).stdout)
        fin, n = d.get("final") or {}, d.get("test_outputs") or 0
        pct = fin.get("score_outputs"); tok = d.get("completion_tokens") or sum((d.get("arm_tokens") or {}).values())
        return {"run": run, "arc2": True, "pct": pct, "right": round(pct * n / 100) if pct is not None and n else None, "of": n,
                "tokM": tok / 1e6, "tps": d.get("tokens_per_second"), "play_s": d.get("play_seconds") or d.get("elapsed_seconds")}
    return {"run": run, "state": "ARC-AGI-2 roster trial: no live feed, its progress file syncs at the end"}

def hostline(live):
    """B v3+ (b200_harness): host CPU/RAM and his tool-timeout counters, else ''."""
    h, t = live.get("host") or {}, live.get("tools") or {}
    if "cpu_busy" not in h and not t: return ""
    w = t.get("window") or {}
    return (f", host cpu {h.get('cpu_busy', 0):.0%} of {h.get('vcpus')} (load {h.get('load1')}), ram {h.get('mem_used_gib')}/{h.get('mem_total_gib')} GiB"
            f", tools {t.get('tool_calls')} calls, timeouts cpu {t.get('timeouts_cpu')} wall {t.get('timeouts_wall')}"
            f" (1 h: {w.get('timeouts')}/{w.get('calls')}), gen2 max {t.get('gen2_max_ms')} ms")

def f(v, p="{:.2f}"):
    return "-" if v is None else p.format(v)

def sft_row(run):
    """Live feed of a Modal training run: loss per update and the selection NLL (posttrain/<run>/live.jsonl, events.jsonl)."""
    d = OUT / run; d.mkdir(exist_ok=True)
    if not get(f"posttrain/{run}/live.jsonl", d / "live.jsonl"):
        return {"run": run, "state": "no live progress yet"}
    get(f"posttrain/{run}/events.jsonl", d / "events.jsonl")
    def rows(name):
        out = []
        for x in (d / name).read_text().splitlines() if (d / name).exists() else []:
            try: out.append(json.loads(x))
            except Exception: pass
        return out
    live, ev = rows("live.jsonl"), rows("events.jsonl")
    if not live: return {"run": run, "state": "no live progress yet"}
    last = live[-1]
    sel = [(e["step"], e["mean"]) for e in ev if e.get("event") == "select" and e.get("mean") is not None]
    return {"run": run, "sft": True, "utc": last.get("utc"), "step": last.get("step"), "steps": last.get("steps"), "pass": last.get("pass"), "passes": last.get("passes"),
            "loss": last.get("loss"), "losses": [(e["step"], e["loss"]) for e in ev if e.get("event") == "step" and e.get("loss") is not None],
            "sel": sel, "light": last.get("light_select_mean_delta"), "eta_h": (last.get("eta_seconds") or 0) / 3600,
            "busy": last.get("busy_share"), "stopped": last.get("stopped"), "oom": last.get("out_of_memory")}

def rl_row(run, updates=60, per_update=4):
    """Live feed of an RL pilot (posttrain/<run>/events.jsonl): passes, updates, the shift in log-probability on attempts
    pushed up or down (measured before the attempt is trained on), and the drift probes."""
    d = OUT / run; d.mkdir(exist_ok=True)
    if not get(f"posttrain/{run}/events.jsonl", d / "events.jsonl"):
        return {"run": run, "state": "no live progress yet"}
    ev = []
    for x in (d / "events.jsonl").read_text().splitlines():
        try: ev.append(json.loads(x))
        except Exception: pass
    p = [e for e in ev if e.get("event") == "pass"]
    if not p: return {"run": run, "state": "loading and aligning"}
    al = [e for e in ev if e.get("event") == "alignment"]
    t0 = al[-1]["t"] if al else p[0]["t"]; per = (p[-1]["t"] - t0) / len(p); total = updates * per_update
    probes = [(int(re.sub(r"\D", "", str(e.get("at"))) or 0), e["divergence"]) for e in ev if e.get("event") == "probe" and e.get("divergence") is not None]
    other = [e["event"] for e in ev if e.get("event") not in ("pass", "alignment", "heartbeat", "environment", "loaded", "table_scale", "probe")]
    return {"run": run, "rl": True, "passes": len(p), "total": total, "updates": p[-1].get("after_updates"), "of": updates,
            "up": [(i, e["mean_log_ratio"]) for i, e in enumerate(p) if e.get("weight", 0) > 0],
            "down": [(i, e["mean_log_ratio"]) for i, e in enumerate(p) if e.get("weight", 0) < 0],
            "probes": probes, "per": per, "left_h": max(0, total - len(p)) * per / 3600, "peak": max(e.get("peak_gib", 0) for e in p),
            "hours": ev[-1]["t"] / 3600, "last_event": other[-1] if other else None}

def heal_queue():
    """A pull that conflicted on the queue leaves conflict markers and every pass crashes: resolve it with merge_queue.py and
    commit the merge when the queue is the only conflicted file."""
    q = pathlib.Path(ROOT, "docs/modal-queue.json")
    try:
        json.loads(q.read_text()); return
    except ValueError:
        pass
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)
    if git("rev-parse", "-q", "--verify", "MERGE_HEAD").returncode or git("diff", "--name-only", "--diff-filter=U").stdout.split() != ["docs/modal-queue.json"]:
        return
    if subprocess.run([sys.executable, str(pathlib.Path(ROOT, "tools/ops/merge_queue.py"))], cwd=ROOT).returncode == 0:
        git("add", "docs/modal-queue.json"); git("commit", "-q", "--no-edit"); print("queue merge conflict resolved (merge_queue.py)")


if __name__ == "__main__":
    import reconcile, concurrent.futures as cf
    T0 = time.time(); MARKS = []; mark = lambda n: MARKS.append(f"{n} {time.time() - T0:.0f}s")  # pass timing, printed at the end
    try:  # the training page needs only local feeds: build it first so a slow cloud feed never delays it
        import train_dash; train_dash.build(); TD_ERR = None
    except Exception as e: TD_ERR = f"training dashboard failed: {e}"[:160]; print("train_dash failed:", e)
    mark("train_dash")
    heal_queue()
    st = json.loads(pathlib.Path(ROOT, "docs/html/arm-state.json").read_text())
    print(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    try: reconcile.run(st)
    except Exception as exc: print("reconcile failed:", exc)
    mark("reconcile")
    try: STATE = json.loads((OUT / "state.json").read_text())
    except Exception: STATE = {"runs": {}, "nebius_ok": False}
    NEBIUS |= nebius_runs()
    q = paths.read_json(pathlib.Path(ROOT, "docs/modal-queue.json"))
    extra = [r for r in sys.argv[1:] if r.strip()]
    watch = list(dict.fromkeys(extra + [e["run"] for e in q.get("live", []) if "cpu" not in str(e.get("gpu", "")).lower()]))
    def safe_row(r):
        try: return row(r)
        except Exception as exc: return {"run": r, "state": f"feed failed: {exc}"[:80]}
    with cf.ThreadPoolExecutor(12) as ex: rows = list(ex.map(safe_row, watch))  # cloud reads are I/O bound: one thread per run
    mark("rows")
    print("| Run | Play min | Score | pp/Mtok | Tokens | Mean / 30m tok/s | Prefix hits | KV | Streams tot/act/gate | Cap / 30m running | Preempt |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for r in rows:
        if "state" in r:
            print(f"| {r['run']} | {r['state']} |"); continue
        print(f"| {r['run']} | {r['min']:.0f} | {f(r['score'])}% | {f(r['ppM'], '{:.3f}')} | {r['tokM']:.2f}M | {f(r['mean'], '{:.0f}')} / {f(r['last30'], '{:.0f}')} | {f(r['hit'], '{:.1f}')}% | {f((r['kv'] or 0) * 100, '{:.1f}')}% | {r['streams']} | {r.get('cap') or '?'} / {f(r.get('run30'), '{:.1f}')} | {f(r['preempt'], '{:.0f}')} |")
    for r in rows:
        if "state" not in r:
            print(f"  {r['run']}: snapshot age {r['age_s']}s, score lag {r['score_lag_s']:.0f}s, errors {r['errors']}, loop fires {r['loop']}, phase {r['phase']}, finished {r['finished']}, vllm rss {r['rss']}{r.get('hostline', '')}")
    (OUT / "last.json").write_text(json.dumps(rows, indent=1))
    by = {r["run"]: r for r in rows}
    import bench_feed
    tried = set()
    def feed(run):
        if run.startswith("rsi-loop-") and not (by.get(run) or {}).get("rsi"):
            import rsi_feed
            try: by[run] = rsi_feed.row(run) or {"run": run, "state": "no loop state yet"}
            except Exception as exc: by[run] = {"run": run, "state": f"rsi feed failed: {exc}"[:80]}
            tried.add(run)
            return by[run]
        if run.startswith(("rl1-probe-", "rl1-loop-")) and not (by.get(run) or {}).get("rl1") and run not in tried:
            import rl1_feed
            tried.add(run)
            try: by[run] = rl1_feed.row(run)
            except Exception as exc: by[run] = {"run": run, "state": f"rl1 feed failed: {exc}"[:80]}
            return by[run]
        if run.startswith("arc2-") and run in NEBIUS and run not in tried:
            tried.add(run)
            try: by[run] = arc2_row(run)
            except Exception as exc: by[run] = {"run": run, "state": f"arc2 feed failed: {exc}"[:80]}
            return by[run]
        if run not in by:
            by[run] = sft_row(run) if run.startswith("sft-qlora-train") else rl_row(run) if run.startswith("rl-train") else row(run)
        if "state" in by[run] and run not in tried:  # no arena progress file: a level-bench run keeps its data elsewhere
            tried.add(run)
            try: by[run] = bench_feed.row(run) or by[run]
            except Exception as exc: print("bench feed failed:", run, exc)
        return by[run]
    def latest(run):
        r = feed(run)
        if "state" in r: return "running, " + (r["state"] if r["run"].startswith("rl-train") else "no score feed")
        if r.get("rl"):
            pr = f", drift probe {r['probes'][-1][1]:+.4f} at update {r['probes'][-1][0]} (stop at 0.02)" if r["probes"] else ""
            return f"update {r['updates']} of {r['of']}, pass {r['passes']} of {r['total']}, {r['per']:.0f} s per pass, about {r['left_h']:.1f} h left{pr}"
        if r.get("sft"):
            s0 = r["sel"][0][1] if r["sel"] else None; s1 = r["sel"][-1] if len(r["sel"]) > 1 else None
            return (f"loss {r['loss']:.3f}" if r.get("loss") is not None else "loss -") + (f", selection NLL {s1[1]:.3f} at unit {s1[0]} (start {s0:.3f})" if s1 else (f", selection NLL at start {s0:.3f}" if s0 is not None else "")) + f", unit {r['step']} of {r['steps']}, pass {r['pass']} of {r['passes']}"
        if r.get("bench"): return bench_feed.text(r)
        if r.get("arc2"): return f"{r['right']} of {r['of']} right at {r['tokM']:.2f}M tokens, {r['tps'] or 0:.0f} tok/s" 
        if r.get("rsi"):
            import rsi_feed
            return rsi_feed.text(r)
        if r.get("rl1"):
            import rl1_feed
            return rl1_feed.text(r)
        bits = []
        if r.get("score") is not None: bits.append(f"{r['score']:.2f}%")
        if r.get("tokM"): bits.append(f"{r['tokM']:.1f}M tokens")
        if r.get("last30"): bits.append(f"{r['last30']:.0f} tok/s")
        if "cap" in r: bits.append(f"{f(r.get('run30'), '{:.1f}')} of {r.get('cap') or '?'} streams running (30-min mean)")
        if r.get("kv") is not None: bits.append(f"cache use {100 * r['kv']:.0f}%")
        if r.get("hit") is not None: bits.append(f"prefix hits {r['hit']:.0f}%")
        return ", ".join(bits) + f" after {r['min'] / 60:.1f} h"
    import html
    now = datetime.datetime.now(datetime.timezone.utc)
    bad_rows, clock_rows = [], []  # clock_rows: rows whose UTC clock strings disagree with their epochs (shown as repairs)
    def hours(e):  # never raises: a malformed queue row (missing start/end) shows as queued and is reported, not a crashed pass
        try: return _hours(e)
        except Exception as exc:
            msg = f"{e.get('run', '?')}: queue row unreadable ({type(exc).__name__}: {exc}); shown as queued"
            if msg not in bad_rows: bad_rows.append(msg)
            return -0.05, 2.0
    def _hours(e):
        s = STATE["runs"].get(e["run"]) or {}
        if not e.get("_q") and s.get("state") == "running" and s.get("first"):  # the machines say it runs: their clock wins, receipt or not
            tot = e["end_epoch"] - e["start_epoch"] if e.get("start_epoch") and e.get("end_epoch") else 3600 * float(e.get("hours") or 2.5)
            return (now.timestamp() - s["first"]) / 3600, tot / 3600
        def length():  # planned length: epochs, then hours, then end - start on the row's own clock strings (same zone), else 2.5 h
            if e.get("start_epoch") and e.get("end_epoch"): return (e["end_epoch"] - e["start_epoch"]) / 3600
            if e.get("hours"): return float(e["hours"])
            end = re.search(r"\d\d?:\d\d", str(e.get("end") or e.get("expected_end") or "")); start = re.search(r"\d\d?:\d\d", str(e.get("start") or ""))
            if end and start:
                (h1, m1), (h2, m2) = start.group().split(":"), end.group().split(":")
                return (((int(h2) - int(h1)) * 60 + int(m2) - int(m1)) % 1440 or 1440) / 60
            return 2.5
        if e.get("app_id") in live_apps:  # Modal says the app runs: running since the app was created, whatever the row's clock says
            return (now.timestamp() - live_apps[e["app_id"]]) / 3600, length()
        if e.get("start_epoch") and e.get("start") and not e.get("_q"):  # clock strings are UTC; flag rows written in local time
            h, m = map(int, re.search(r"\d\d?:\d\d", e["start"]).group().split(":")); se = datetime.datetime.fromtimestamp(e["start_epoch"], datetime.timezone.utc)
            off = (h * 60 + m - se.hour * 60 - se.minute) % 1440
            if 30 < off < 1410:
                msg = f"{e['run']}: row start {e['start']} is {min(off, 1440 - off) / 60:.1f} h off its start_epoch ({se:%H:%M} UTC); epoch used (queue clock times are UTC)"
                if msg not in clock_rows: clock_rows.append(msg)
        if not e.get("_q") and e.get("start_epoch") and e.get("platform") != "nebius":  # exact start, with or without an end epoch
            return (now.timestamp() - e["start_epoch"]) / 3600, length()
        if e.get("start") and not e.get("end") and not (e.get("start_epoch") and e.get("end_epoch")):
            h, m = e["start"].split()[0].split(":"); t = (int(h) * 60 + int(m) + round(60 * float(e.get("hours") or 2.5))) % 1440
            e = dict(e, end=f"{t // 60:02d}:{t % 60:02d}")  # a row with a start but no end: its length or 2.5 h
        if e.get("_q") and not e.get("start"):  # queued with no planned clock time: waits for a slot
            if e.get("start_epoch") and e.get("end_epoch"):
                return -max((e["start_epoch"] - now.timestamp()) / 3600, 0.05), (e["end_epoch"] - e["start_epoch"]) / 3600
            return -0.05, float(e.get("hours") or 2.5)
        if e.get("_q"):  # in the queue's "queued" list: not started, whatever its planned start
            h, m = e["start"].split()[0].split(":"); until = (int(h) * 60 + int(m) - now.hour * 60 - now.minute) % 1440
            a, b = [int(x.split()[0][:2]) * 60 + int(x.split()[0][3:5]) for x in (e["start"], e["end"])]
            return -max(until if until < 720 else 0, 1) / 60, ((b - a) % 1440 or 1440) / 60
        if e.get("platform") == "nebius" and not pathlib.Path(ROOT, "docs/validation-nebius", f"nebius-{e['run']}.json").exists():
            if e.get("start_epoch") and e.get("end_epoch"):  # no launch receipt yet: queued, even if the row carries epochs
                return -max((e["start_epoch"] - now.timestamp()) / 3600, 0.05), (e["end_epoch"] - e["start_epoch"]) / 3600
        if e.get("start_epoch") and e.get("end_epoch"):  # stamped by reconcile.py: exact, any length
            return (now.timestamp() - e["start_epoch"]) / 3600, (e["end_epoch"] - e["start_epoch"]) / 3600
        def mins(t): h, m = t.split()[0].split(":"); return int(h) * 60 + int(m)
        st, en = mins(e["start"]), mins(e["end"])
        ago = (now.hour * 60 + now.minute - st) % 1440
        total = en - st + (1440 if "+1d" in e["end"] or en <= st else 0)  # end == start: 24 h
        until = (st - now.hour * 60 - now.minute) % 1440
        if e.get("platform") == "nebius" and not pathlib.Path(ROOT, "docs/validation-nebius", f"nebius-{e['run']}.json").exists():
            return -(until if until < 720 else 6) / 60, total / 60  # no launch receipt: still queued, whatever its planned start
        if ago > total and until < 720: return -until / 60, total / 60  # clock times carry no date: a start ahead of now is a queued row
        return ago / 60, total / 60
    try:
        apps = json.loads(subprocess.run([str(pathlib.Path(ROOT, ".venv/bin/modal")), "app", "list", "--json"], capture_output=True, text=True, timeout=60).stdout or "[]")
    except Exception: apps = None
    live_apps = {a["app_id"]: datetime.datetime.fromisoformat(a["created_at"]).timestamp() for a in apps or []
                 if a.get("state") != "stopped" and int(a.get("tasks") or 0)}
    names = {x["run"] for x in q.get("live", [])}
    live = sorted(q.get("live", []) + [dict(e, _q=True) for e in q.get("queued", []) if e["run"] not in names], key=lambda e: (hours(e)[0] < 0, abs(hours(e)[0])))
    for e in live:
        e["_ago"], e["_total"] = hours(e)
        if e["run"].startswith("hx4-scr"): e["arm"] = "envs"  # envs pre-screen runs reuse the hx4- launcher prefix
        elif e["run"].startswith(("hx", "pt1-", "b2-hx")) and e.get("arm") != "harness": e["arm"] = "harness"  # collaborator runs under harness
        if e["run"].startswith("b2-sft") and e.get("arm") == "throughput": e["arm"] = "sft"  # SFT replays on the shared b2- prefix
    # Self-repair ("whenever there is an issue we should fix automatically"): a machine that runs with no
    # started row gets one built from the machine listing, every pass, and the check lists it as a repair, not a failure
    repairs, vms_now = list(clock_rows), None
    rules = q.get("arm_rules") or {}
    guess = lambda run: next((v for k, v in rules.items() if run.lower().startswith(k.lower())), None) or next(
        (a for p, a in (("arc2-", "arc2"), ("rsi-", "rsi"), ("prime", "prime"), ("arc-prime", "prime"), ("rl1-screen-seed", "envs"), ("rl1-loop-xpf", "envs"), ("rl1-loop-mg", "envs"), ("rl", "rl"), ("sft-", "sft"), ("pf-sft", "sft"), ("b2-hx", "harness"), ("b2-sft", "sft"), ("b2-", "throughput"),
                        ("hx4-scr", "envs"), ("hx", "harness"), ("pt1-", "harness")) if run.lower().startswith(p)), "other")
    for e in live:  # rows written without an arm (22 Oct: v6x-r2, the envs seed) would count under no arm
        if not e.get("arm") or e["arm"] == "?": e["arm"] = guess(e["run"])
        if not e.get("platform") and e.get("app_id"): e["platform"] = "modal"  # Modal rows written without a platform
    started_low = {x["run"].lower() for x in live if x["_ago"] >= 0}
    try: vms_now = reconcile.nebius_vms()
    except Exception: vms_now = None
    for low, v in sorted((vms_now or {}).items()):
        hits = {x["run"].lower() for x in live if x["run"].lower().startswith(low)}  # VM labels cap run ids at 63 characters
        if low not in hits and len(hits) == 1: low = hits.pop()
        if low in started_low or low.startswith(tuple(st.get("ignore_runs", []))): continue
        known = next((x for x in live if x["run"].lower() == low), None)
        hrs = re.search(r"-(\d+)h-", low)
        e = dict(known or {"run": low, "arm": guess(low), "platform": "nebius", "gpu": "RTX PRO 6000"}, _q=False,
                 _ago=(now.timestamp() - v["created"]) / 3600, _state="running", _since=v["created"], _auto=True)
        e["_total"] = (e["end_epoch"] - e["start_epoch"]) / 3600 if e.get("start_epoch") and e.get("end_epoch") else float(hrs.group(1)) if hrs else 2.5
        e.setdefault("what", "Running on Nebius with no started row in the run queue; row added by the dashboard from the machine listing.")
        live = [x for x in live if x["run"].lower() != low] + [e]; started_low.add(low)
        repairs.append(f"{low}: running on Nebius {'but shown as queued' if known else 'with no queue row'}; shown from the machine listing")
    cpu_apps = set(q.get("cpu_apps") or [])  # CPU helpers (preflights, pulls, exports): listed by app id in the queue, never cards
    for a in apps or []:
        if a.get("state") == "stopped" or not int(a.get("tasks") or 0) or a.get("app_id") in cpu_apps: continue
        t = datetime.datetime.fromisoformat(a["created_at"]).timestamp()
        if any(x.get("platform") == "modal" and x["_ago"] >= 0 and abs((x.get("start_epoch") or 0) - t) < 1800 for x in live): continue
        name = str(a.get("description") or a.get("app_id"))
        # no cards for apps outside the queue (CPU helpers: pulls, exports, cut rebuilds); GPU runs always get a
        # queue row at launch, so an unmatched app is only named in the check line
        repairs.append(f"Modal app {name} ({a.get('app_id')}): no queue row (CPU helper?); not shown as a card")
    full = {}
    def safe_feed(run):
        try: return feed(run)
        except Exception as exc: return {"run": run, "state": f"feed failed: {exc}"[:80]}
    todo = []
    for e in live:
        e.setdefault("what", ""); e.setdefault("arm", "?")
        if e["_ago"] < 0: continue
        if e.get("_nofeed"): full[e["run"]] = {"run": e["run"], "state": "Modal app outside the run queue: no live feed"}; continue
        todo.append(e["run"])
    with cf.ThreadPoolExecutor(12) as ex: full.update(zip(todo, ex.map(safe_feed, todo)))
    mark("feeds")
    import status_html, banner
    try: LEDGER = reconcile.ledger_states()
    except Exception: LEDGER = {}
    plain = lambda s: re.sub(r"<[^>]+>", "", s or "")
    failures, cards = list(bad_rows), {}
    try: failures.insert(0, "previous pass crashed: " + (OUT / "failed.txt").read_text().strip().splitlines()[-1][:200])
    except Exception: pass
    try:  # shared watchers whose heartbeat stopped (killed by SIGKILL, reboot): the owning arm restarts them (24 Oct)
        import watch
        stale = watch.stale_beats()
        failures += [f"watcher for {r} ({arm}) dead/hung {age / 60:.0f} min: restart watch.py run <plan>" for r, arm, age, _ in stale]
        (OUT / "watchers.txt").write_text("".join(f"STALE {r} {arm} {age:.0f}s\n" for r, arm, age, _ in stale))
    except Exception as e: failures.append(f"watcher heartbeat check failed: {e!r}"[:200])
    try:  # runs that ended with no stated cause in the last day (app/VM gone, no settle): the arm settles with --cause
        sys.path.insert(0, str(paths.ROOT_PATH / "src")); from arc_infra.runops import records as _R
        recent = [r for r in _R.records() if r.get("end_cause") == "unclassified"
                  and float(((r.get("ledger") or [{}])[-1].get("row") or {}).get("start_epoch") or 0) > time.time() - 86400]
        if recent:
            failures.append(f"{len(recent)} runs ended in the last day without a cause (records.py unclassified; settle each with "
                            f"--cause): " + ", ".join(sorted(r["run"] for r in recent))[:300])
        dropped = _R.dropped_legacy(since=time.time() - 86400)  # phase D: direct view edits are overwritten, not imported
        if dropped:
            failures.append(f"direct edits of docs/spend.json / modal-queue.json DROPPED in the last day (write through "
                            f"records.py): " + ", ".join(dropped)[:300])
    except Exception as e: failures.append(f"end-cause check failed: {e!r}"[:200])
    try:  # runs volume inodes/bytes (volume_usage.py, hourly): ALERT at 80% of a cap (24 Oct: /runs hit 84% of its inodes)
        import volume_usage
        vf = OUT / "volume.json"
        if vf.is_file():
            v = json.loads(vf.read_text())
            failures += volume_usage.alerts(v)
            if time.time() - v["t"] > 3 * 3600: failures.append(f"volume usage reading is {(time.time() - v['t']) / 3600:.0f} h old")
    except Exception as e: failures.append(f"volume check failed: {e!r}"[:200])
    try:  # CI (ci.py): a red head of a watched branch blocks its bundle builds
        ci_red = []
        for f in sorted((paths.ROOT_PATH / "runs/ops/ci").glob("*.json"), key=lambda p: p.stat().st_mtime)[-30:]:
            r = json.loads(f.read_text())
            if r.get("verdict") in ("red", "error") and time.time() - r.get("t", 0) < 86400: ci_red.append(f"{r.get('ref', '?')[:12]} {r.get('summary', '')[:60]}")
        if ci_red: failures.append("CI red (bundle builds blocked): " + "; ".join(ci_red[-3:]))
    except Exception as e: failures.append(f"CI check failed: {e!r}"[:200])
    for e in live:  # state from the machines (reconcile.py), then what the card shows, then the self-check
        s = STATE["runs"].get(e["run"])
        if e.get("_auto"): pass  # state and start already from the machine listing
        elif e["_ago"] >= 0 and s:
            e["_state"], e["_since"] = s["state"], s.get("since")
            if s["state"] == "running" and s.get("first"): e["_ago"] = (now.timestamp() - s["first"]) / 3600
        elif e["_ago"] >= 0 and e.get("platform") == "modal" and any(
                int(a.get("tasks") or 0) and a.get("state") != "stopped"
                and abs(datetime.datetime.fromisoformat(a["created_at"]).timestamp() - (e.get("start_epoch") or 0)) < 1800
                for a in apps or []):
            e["_state"] = "running"  # a Modal job with no receipt (e.g. a CPU fit): matched to its app by start time
        elif e["_ago"] >= 0:
            e["_why"] = ("Nebius listing failed this pass" if e.get("platform") == "nebius" and not STATE.get("nebius_ok")
                         else "Modal listing failed this pass" if e.get("platform") == "modal" and STATE.get("modal_ok") is False else "no launch record or machine found")
        r = full.get(e["run"]) or {}
        try: e["_card"] = c = status_html.summary(e, {} if "state" in r else r, reconcile.no_feed_kind(e["run"]))
        except Exception as exc:
            failures.append(f"{e['run']}: card failed ({type(exc).__name__}: {exc})"[:200])
            e["_card"] = c = {"state": "unknown", "head": "", "pct": None, "lines": [], "score": "", "score_why": "card failed", "tps": "", "tps_why": "card failed"}
        if e["_ago"] < 0: continue
        if "state" in r:  # the feed said why it has nothing: that is the reason, not a blank
            for k in ("score_why", "tps_why"):
                if not c.get(k) or c[k] == "no live feed found": c[k] = r["state"]
        cards[e["run"]] = " · ".join(plain(x) for x in (c["score"], c["tps"]) if x)
        if e.get("platform") == "kaggle":  # Kaggle save runs: no machine listing; a live row only if the run posts one
            c.setdefault("score_why", None); c["score_why"] = c["score_why"] or "Kaggle save run: result at the end"; c["tps_why"] = c["tps_why"] or "Kaggle save run: no live feed"
            if not (c.get("score") or c.get("tps")): cards[e["run"]] = "Kaggle save run"
            continue
        END = ("complete", "completed", "done", "finished", "stopped", "failed", "settled", "observed_lifetime")
        ended = str(e.get("_state") or c.get("state") or "") in END or str(LEDGER.get(e["run"]) or "") in END
        if ended: continue  # finished since the queue was read; reconcile moves the row off live next pass
        app = next((a for a in apps or [] if e.get("app_id") and a.get("app_id") == e["app_id"]), None)
        if app and app.get("state") == "stopped":  # its own Modal app has stopped: ended, the writer just never closed the row
            repairs.append(f"{e['run']}: Modal app {e['app_id']} stopped but the row is still live; shown as ended (close it)")
            continue
        fresh = 0 <= e["_ago"] < 0.25  # launched in the last 15 min: the machine state can lag a pass
        fresh = fresh or (e.get("platform") == "nebius" and not STATE.get("nebius_ok"))  # listing failed this pass: no state to judge
        fresh = fresh or bool(app and int(app.get("tasks") or 0))  # its own Modal app is up: that is the machine state
        bad = [why for ok, why in ((c["state"] in ("running", "waiting", "starting") or fresh, "no state from the machines"),
                                   (c["score"] or (c["score_why"] and c["score_why"] != "no live feed found"), "no score and no reason"),
                                   (c["tps"] or c["tps_why"] and c["tps_why"] != "no live feed found", "no tok/s and no reason"),
                                   (len(str(e.get("what") or "").strip()) >= 20, "no description")) if not ok]
        if bad: failures.append(f"{e['run']}: {', '.join(bad)}")
    # Coverage: every machine running on Nebius or Modal must be a row on the page
    live_low = {x["run"].lower() for x in live if x["_ago"] >= 0}
    try:
        if vms_now is None: raise RuntimeError("listing failed")
        for low in sorted(vms_now or {}):
            if not any(r.startswith(low) for r in live_low) and not low.startswith(tuple(st.get("ignore_runs", []))):
                failures.append(f"Nebius VM {low} is running but has no row on the page")
    except Exception as exc: failures.append(f"coverage check: Nebius listing failed ({exc})"[:160])
    try:
        if apps is None: raise RuntimeError("listing failed")
        mrows = [x for x in live if x.get("platform") == "modal" and x["_ago"] >= 0]
        for a in apps:
            if a.get("state") == "stopped" or not int(a.get("tasks") or 0) or a.get("app_id") in cpu_apps: continue
            t = datetime.datetime.fromisoformat(a["created_at"]).timestamp()
            if any(x.get("app_id") == a.get("app_id") for x in live): continue  # a row names this app (start times can differ)
            if not any(abs((x.get("start_epoch") or x.get("_since") or 0) - t) < 1800 for x in mrows):
                failures.append(f"Modal app {a.get('description')} ({a.get('app_id')}, {int(a.get('tasks') or 0)} tasks, created {datetime.datetime.fromtimestamp(t, datetime.timezone.utc):%H:%M}) has no row on the page")
    except Exception as exc: failures.append(f"coverage check: Modal listing failed ({exc})"[:160])
    # Agent stalls: an agent waiting >20 min on one tool call (a foreground subagent) reads no messages
    try:
        tz = datetime.timezone.utc
        for f in pathlib.Path.home().glob(f".claude/projects/{str(ROOT).replace('/', '-')}/*/subagents/agent-*.jsonl"):
            if f.stat().st_mtime < time.time() - 6 * 3600: continue
            with open(f, "rb") as fh:
                fh.seek(max(0, f.stat().st_size - 400_000)); lines = fh.read().decode("utf-8", "replace").splitlines()[1:]
            pend = {}
            for ln in lines:
                try: d = json.loads(ln)
                except Exception: continue
                cs = (d.get("message") or {}).get("content")
                for c in cs if isinstance(cs, list) else []:
                    if c.get("type") == "tool_use" and d.get("timestamp"): pend[c["id"]] = (d["timestamp"], c["name"], str((c.get("input") or {}).get("description") or "")[:60])
                    elif c.get("type") == "tool_result": pend.pop(c.get("tool_use_id"), None)
            for ts, name, desc in pend.values():
                t = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if time.time() - t.timestamp() > 1200:
                    failures.append(f"agent {f.stem[6:]} blocked since {t.astimezone(tz):%H:%M} UTC on {name} '{desc}' (reads no messages)")
    except Exception as exc: failures.append(f"agent stall check failed ({exc})"[:160])
    # Silent arm agents (22 Oct: a session restart killed every watcher; an agent died on a network error): an arm with
    # live runs whose agent transcript has not moved for 45 min has nothing left to wake it. Arms -> agent ids in <scratch>/agents.json
    try:
        amap = json.loads(pathlib.Path(os.environ.get("ARC_OPS_SCRATCH") or pathlib.Path(ROOT, "runs/ops"), "agents.json").read_text())
        arms_live = {x.get("arm") for x in live}
        for arm, ids in amap.items():  # an arm may have several agents (mfu: MFU-1 and MFU-2): silent only if all are
            if arm not in arms_live: continue
            ids = [ids] if isinstance(ids, str) else list(ids)
            fs = [f for aid in ids for f in pathlib.Path.home().glob(f".claude/projects/{str(ROOT).replace('/', '-')}/*/subagents/agent-{aid}.jsonl")]
            if not fs: continue
            age = time.time() - max(f.stat().st_mtime for f in fs)
            if age > 2700:
                failures.append(f"agent {'/'.join(ids)} ({arm}) silent for {age / 60:.0f} min while its runs are live: no watcher left to wake it")
    except FileNotFoundError: pass
    except Exception as exc: failures.append(f"agent silence check failed ({exc})"[:160])
    # The pages written last pass: leaked Python values or broken markup
    import html.parser
    class _Chk(html.parser.HTMLParser):
        def __init__(self): super().__init__(); self.stack, self.bad, self.text = [], 0, []
        def handle_starttag(self, tag, a):
            if tag in ("div", "table", "tr", "td", "th", "section", "ul", "li", "span", "p", "b", "small", "a"): self.stack.append(tag)
        def handle_endtag(self, tag):
            if tag in ("div", "table", "tr", "td", "th", "section", "ul", "li", "span", "p", "b", "small", "a"):
                if tag in self.stack:
                    while self.stack and self.stack.pop() != tag: self.bad += 1
                else: self.bad += 1
        def handle_data(self, d): self.text.append(d)
    for pg in sorted(pathlib.Path(ROOT, "docs/html").glob("*-status.html")) + [pathlib.Path(ROOT, "docs/html/index.html"), pathlib.Path(ROOT, "MODAL_STATUS.html")]:
        try:
            raw = pg.read_text(); c = _Chk(); c.feed(raw)
            # banner/strip text is quoted from the arm logs (prose such as "loss=NaN" or "None of"): tag-balance it, but leak-check only generated text (23 Oct)
            prose = re.sub(r"<!--(BANNER|STRIP|ARMS):start-->.*?<!--\1:end-->", "", raw, flags=re.S); cp = _Chk(); cp.feed(prose)
            txt = " ".join(x for x in cp.text if "Dashboard check" not in x and "dashboard check" not in x)
            leaks = [w for w in ("None", "nan%", "NaN", "Traceback", "{'", "undefined") if re.search(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", txt)]
            leaks += ["KeyError"] if re.search(r"KeyError\s*[:(\[']", txt) else []  # an exception rendered, not a log line naming one
            if c.bad or leaks: failures.append(f"{pg.name}: " + ", ".join(([f"{c.bad} unbalanced tags"] if c.bad else []) + ([f"leaked text {leaks}"] if leaks else [])))
        except Exception as exc: failures.append(f"{pg.name}: html check failed ({exc})"[:160])
    (OUT / "cards.json").write_text(json.dumps(cards))
    free_gb = shutil.disk_usage(OUT).free / 2**30
    if free_gb < 100: failures.insert(0, f"local disk: only {free_gb:.0f} GB free (agents and loops stop at 0; clear scratch run copies)")
    (OUT / "dashboard-check.txt").write_text(now.strftime("%Y-%m-%d %H:%M UTC\n") + ("\n".join(failures) or "all live cards complete") + "\n"
                                             + "".join(f"repaired: {x}\n" for x in repairs))
    try: ended = [x for x in map(json.loads, (OUT / "ended.jsonl").read_text().splitlines()) if x["t"] > now.timestamp() - 3600]
    except Exception: ended = []
    try:  # training dashboards first, so a later failure in this pass cannot hold them back
        if TD_ERR: failures.append(TD_ERR)
    except Exception: pass
    w, views = banner.run(live, full, now, failures)
    arms = json.loads(pathlib.Path(ROOT, "docs/html/arm-state.json").read_text())["arms"]
    nxt = [f"{arms[k]['name']}: {v['next']}" for k, v in views.items() if v.get("next")] or q.get("next", [])
    out = ["# What is running", "", now.strftime("Updated %Y-%m-%d %H:%M UTC, refreshed every 5 minutes. Most recently started first. Runs on Modal and on Nebius."), "",
           "<table>", '<tr><th width="15%">Run</th><th width="58%">What it tests and why</th><th width="6%">Started, hours ago</th><th width="6%">Planned runtime, hours</th><th width="15%">Latest</th></tr>']
    for e in live:
        where = "Nebius" if e.get("platform") == "nebius" else "Modal"
        ago, total = e["_ago"], e["_total"]
        started = f"{ago:.1f}" if ago >= 0 else f"queued, in {-ago:.1f}"
        c = e["_card"]; txt = "; ".join(plain(x) for x in (c["head"], c["score"] or c["score_why"], c["tps"] or c["tps_why"]) if x)
        out.append(f"<tr><td><b>{e['arm']}</b><br>{e['run']}<br>{where}, {e.get('gpu', '?')}</td><td>{html.escape(e['what'])}</td><td align=\"right\">{started}</td><td align=\"right\">{total:.1f}</td><td>{html.escape(txt)}</td></tr>")
    out.append("</table>")
    if ended: out += ["", "## Ended in the last hour", ""] + [f"- {x['run']}: {x.get('final') or 'final score not read'}" for x in ended]
    if failures: out += ["", "## Dashboard check failed", ""] + [f"- {x}" for x in failures]
    out += ["", "## Next (first item of each arm log's Next)", ""] + [f"- {x}" for x in nxt]
    pathlib.Path(ROOT, "MODAL_STATUS.md").write_text(chr(10).join(out) + chr(10))
    pathlib.Path(ROOT, "MODAL_STATUS.html").write_text(status_html.page(live, nxt, {k: v for k, v in full.items() if "state" not in v}, now, ended, failures))
    print("dashboard check:", failures or "ok")
    print("page words below the banner (budget 220):", w)
    mark("pages"); print("pass timing:", ", ".join(MARKS))
