"""Live feed of level-bench runs (kind hx4-level) for mon.py: per arm the attempts, tokens, mean game score and levels, with the
reference score at matched tokens, plus tok/s, cache use and prefix hits.
Each pass copies the small files whole (progress.jsonl, the bench process log, vLLM metrics) and reads only the new bytes of each
attempt log; the parsed state is <scratch>/mon/<run>.bench.json.
Reference (bench_reference.json): earlier base attempts on the same points, each as its score after every cleared level. Matching: the
reference attempts of the arm's points, weighted by the arm's launched attempts per point, are cut at the one per-attempt budget where
their mean tokens used equals the arm's; the reference score is their mean score at that budget. Arms named X + one lowercase letter
(Pa, Pb, Pc) are pooled; with exactly one such group beside `base` it is shown as the treatment."""
import bisect, collections, concurrent.futures, fcntl, json, pathlib, re, subprocess
import paths
BUCKET = "nebius-arc:arc-runs-bucket/runs/validation/"
OUT = paths.SCRATCH / "mon"
REF = json.loads((pathlib.Path(__file__).parent / "bench_reference.json").read_text())
LABEL = re.compile(r"^a\d+-(.+?)-p\d+(?:-([^-]+))?$")  # arm suffix optional (single-arm runs: "treatment")
ENV = pathlib.Path(__file__).resolve().parents[2] / "environment_files"
def point_ref(point):
    """Total levels and baseline actions of a point: the reference file, else the game's own metadata (whole-game points "<family>-game")."""
    if point in REF["points"]: return REF["points"][point]
    fam = point.split("-")[0]
    for m in sorted(ENV.glob(f"{fam}/*/metadata.json")):
        try: b = json.loads(m.read_text())["baseline_actions"]
        except (ValueError, KeyError): continue
        if point.endswith("-game") and b: return {"total_levels": len(b), "baselines": b}
    return None
WANT = (b'{"kind":"model"', b'{"kind":"level_bench_', b'{"kind":"bon_group"')

def rc(*args, timeout=240):
    return subprocess.run(["rclone", *args], capture_output=True, timeout=timeout)

def pull(base, st):
    """Add what is new in each attempt log: generated tokens by minute, level marks, script end, recorded stop."""
    r = rc("lsjson", "-R", "--files-only", "--include", "*/log.jsonl", base + "/level-bench/attempts")
    if r.returncode: return
    def one(item):
        label = item["Path"].split("/")[0]; a = st["attempts"].get(label)
        if a is None or item["Size"] < a["off"]: a = {"off": 0, "tok": 0, "mins": [], "marks": []}
        if item["Size"] == a["off"]: return label, a
        d = rc("cat", "--offset", str(a["off"]), f"{base}/level-bench/attempts/{item['Path']}").stdout
        d = d[:d.rfind(b"\n") + 1]
        for line in d.splitlines():
            if not line.startswith(WANT): continue
            try: e = json.loads(line)
            except ValueError: continue
            k, t = e["kind"], e.get("time") or 0
            if k == "model":
                a["tok"] += int((e.get("usage") or {}).get("completion_tokens") or 0); m = int(t // 60)
                if a["mins"] and a["mins"][-1][0] == m: a["mins"][-1][1] = a["tok"]
                else: a["mins"].append([m, a["tok"]])
            elif k == "bon_group":  # the samples not played are charged too
                a["tok"] += sum(int(s.get("completion_tokens") or 0) for s in e.get("samples") or [] if s.get("index") != e.get("winner"))
            elif k in ("level_bench_level", "level_bench_budget"):  # the attempt's own token clock; after a replayed script, counted from its end
                if "skip" not in a: a["tok"] = max(a["tok"], int(e["completion_tokens"]))
                if k == "level_bench_level": a["marks"].append([t, a["tok"] - a.get("skip", 0), e["level"], e["steps"]])
            elif k == "level_bench_script_end": a["skip"], a["skip_marks"] = a["tok"], len(a["marks"])
            elif k == "level_bench_recorded": a["recorded"] = e.get("completion_tokens") or a["tok"]
        a["off"] += len(d)
        return label, a
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        for label, a in ex.map(one, json.loads(r.stdout)): st["attempts"][label] = a

def scores(point, marks):
    """Game score after each mark, as arc_agi's EnvironmentScoreCalculator: level credit capped at 115, weighted by
    level number, the total capped at 100 x the weight of the levels cleared."""
    p = point_ref(point)
    if not p: return None
    n = p["total_levels"]; den = n * (n + 1) / 2; out, prev, sc, wt = [], 0, 0.0, 0.0
    for _, _, level, steps in marks:
        used = steps - prev; b = p["baselines"][level - 1] if level <= len(p["baselines"]) else None
        credit = min(115.0, 100.0 * (b / used) ** 2) if b and b > 0 and used > 0 else 0.0
        sc += level * credit; wt += level * 100.0 if credit > 0 else 0.0
        out.append(min(sc, wt) / den); prev = steps
    return out

def ref_curve(weights, cap):
    """(mean tokens used, mean score) of the reference at each per-attempt budget up to cap, on the arm's mix of points."""
    if not weights or any(p not in REF["points"] for p in weights): return None
    n = sum(weights.values()); out = []
    for b in range(0, min(cap, REF["budget"]) + 1, 5000):
        m = s = 0.0
        for p, w in weights.items():
            at = REF["points"][p]["attempts"]
            m += w * sum(min(b, a["end"]) for a in at) / len(at)
            s += w * sum(max([x[1] for x in a["marks"] if x[0] <= b], default=0.0) for a in at) / len(at)
        out.append((m / n, s / n))
    return out

def ref_at(curve, m):
    xs = [c[0] for c in curve]; i = bisect.bisect_left(xs, m)
    if i == 0: return curve[0][1]
    if i >= len(curve): return curve[-1][1]
    (x0, y0), (x1, y1) = curve[i - 1], curve[i]
    return y0 + (y1 - y0) * (m - x0) / max(x1 - x0, 1e-9)

def row(run, base=None):
    """Monitor row of a level-bench run, or None when the run has no bench progress file (yet). base: a landed folder's arena/trials."""
    base = base or BUCKET + run + "/arena/working/trials"; f = OUT / f"{run}.bench"
    files = {k: pathlib.Path(f"{f}.{k}") for k in ("progress", "plan", "process", "metrics", "json", "lock")}
    if rc("copyto", base + "/level-bench/progress.jsonl", str(files["progress"])).returncode or not files["progress"].exists(): return None
    for k, name in (("plan", "/level-bench/level-bench-plan.json"), ("process", "/level-bench-process.log"), ("metrics", "/level-bench/vllm-metrics.txt")):
        if k != "plan" or not files[k].exists(): rc("copyto", base + name, str(files[k]))
    with open(files["lock"], "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        st = json.loads(files["json"].read_text()) if files["json"].exists() else {"attempts": {}, "hits": []}
        pull(base, st)
        prog = []
        for x in files["progress"].read_text().splitlines():
            try: prog.append(json.loads(x))
            except ValueError: pass
        prog = [p for p in prog if p.get("event") == "level_bench_progress"]
        last = prog[-1] if prog else {}; el = last.get("elapsed_seconds") or 0
        mt = files["metrics"].read_text() if files["metrics"].exists() else ""
        def metric(name):
            m = re.search(r"^vllm:" + name + r"(?:\{[^}]*\})? ([-+.\de]+)$", mt, flags=re.M); return float(m.group(1)) if m else None
        hits, asked = metric("prefix_cache_hits_total"), metric("prefix_cache_queries_total")
        if hits is not None and asked and (not st["hits"] or st["hits"][-1][0] < el): st["hits"].append([el, hits, asked])
        tmp = pathlib.Path(f"{f}.tmp"); tmp.write_text(json.dumps(st, separators=(",", ":"))); tmp.replace(files["json"])
    plan = json.loads(files["plan"].read_text()) if files["plan"].exists() else {}
    budget = int(plan.get("token_budget") or REF["budget"])
    ended = {}
    for x in files["process"].read_text(errors="replace").splitlines() if files["process"].exists() else []:
        if x.startswith('{"event": "level_bench_attempt"'):
            try: e = json.loads(x); ended[e["label"]] = e
            except (ValueError, KeyError): pass
    atts = []
    for label, a in sorted(st["attempts"].items()):
        m = LABEL.match(label)
        if not m or not a["mins"]: continue
        point, arm = m.groups(); arm = arm or "treatment"; skip = a.get("skip", 0); marks = a["marks"][a.get("skip_marks", 0):]
        sc = scores(point, marks) if "skip" not in a else None
        total = (point_ref(point) or {}).get("total_levels")
        won = marks[-1] if marks and total and marks[-1][2] >= total else None
        tok = won[1] if won else a["recorded"] - skip if a.get("recorded") else min(max(a["tok"] - skip, 0), budget)
        atts.append({"label": label, "point": point, "group": "base" if arm == "base" else re.sub(r"(?<=[A-Z])[a-z]$", "", arm), "tok": tok,
                     "marks": marks, "sc": sc, "won": won, "skip": skip, "mins": a["mins"], "recorded": bool(a.get("recorded")),
                     "ended": bool(won or a.get("recorded") or label in ended), "error": (ended.get(label) or {}).get("outcome") in ("error", "replay_mismatch")})
    atts = [a for a in atts if not a["error"]]
    groups = collections.OrderedDict()
    for a in sorted(atts, key=lambda a: a["group"] == "base"): groups.setdefault(a["group"], []).append(a)
    planned = {re.sub(r"(?<=[A-Z])[a-z]$", "", p["arm"]) for p in plan.get("plan") or []} or set(groups)
    collector = "base" not in planned and len(planned) == 1
    scored = bool(atts) and all(a["sc"] is not None for a in atts)
    def at(a, t):
        """Tokens used and score (or levels) of one attempt at wall time t."""
        i = bisect.bisect_right([m for m, _ in a["mins"]], t // 60)
        tok = min(max((a["mins"][i - 1][1] if i else 0) - a["skip"], 0), budget)
        if a["won"] and a["won"][0] <= t: tok = a["won"][1]
        n = sum(1 for m in a["marks"] if m[0] <= t)
        return tok, (a["sc"][n - 1] if n else 0.0) if scored else n
    t0 = min((a["mins"][0][0] for a in atts), default=0) * 60; t1 = max((a["mins"][-1][0] for a in atts), default=0) * 60
    arms = []
    for g, v in groups.items():
        name = "base" if g == "base" else "treatment" if len(groups) == 2 and "base" in groups else g
        tok = sum(a["tok"] for a in v); n = len(v); k = tok / n
        score = sum(a["sc"][-1] if a["sc"] else 0.0 for a in v) / n if scored else None
        curve = ref_curve(collections.Counter(a["point"] for a in v), budget) if scored and not collector else None
        ref = ref_at(curve, k) if curve else None
        track = []
        for t in range(t0 + 300, t1, 300):
            live = [at(a, t) for a in v if a["mins"][0][0] * 60 <= t]
            if live: track.append((round(sum(x[0] for x in live) / len(live) / 1e3, 1), round(sum(x[1] for x in live) / len(live), 2)))
        levels = sum(len(a["marks"]) for a in v) / n
        track.append((round(k / 1e3, 1), round(score if scored else levels, 2)))
        arms.append({"name": name, "n": n, "running": sum(not a["ended"] for a in v), "ended": sum(a["ended"] for a in v), "tokM": tok / 1e6, "k": k / 1e3,
                     "score": score, "levels": levels, "ppM": score / (k / 1e6) if scored and k else None, "ref": ref,
                     "ratio": score / ref if ref else None, "ref_k": min(k, curve[-1][0]) / 1e3 if curve else None, "track": track})
    curve = ref_curve(collections.Counter(a["point"] for a in atts), budget) if scored and not collector else None
    old = [p for p in prog if p["elapsed_seconds"] <= el - 1800]
    h0 = [h for h in st["hits"] if h[0] <= el - 1500]; h1 = st["hits"][-1] if st["hits"] else None
    hit = None
    if h1: hit = 100 * (h1[1] - h0[-1][1]) / max(h1[2] - h0[-1][2], 1) if h0 else 100 * h1[1] / h1[2]
    return {"run": run, "bench": True, "collector": collector, "min": el / 60, "tok_total": last.get("completion_tokens") or sum(a["tok"] for a in atts),
            "mean": last.get("tokens_per_second"), "last30": (last["completion_tokens"] - old[-1]["completion_tokens"]) / (el - old[-1]["elapsed_seconds"]) if old else None,
            "kv": metric("kv_cache_usage_perc"), "hit": hit, "preempt": metric("num_preemptions_total"), "launched": last.get("launched") or len(atts), "done": last.get("done"), "clears": last.get("clears"),
            "planned": plan.get("planned_attempts") or last.get("planned"), "running": last["running"] if last.get("running") is not None else sum(not a["ended"] for a in atts), "failed": last.get("failed"),
            "recorded": sum(a["recorded"] for a in atts), "arms": arms, "scored": scored, "ref_name": f"{REF['attempts']} earlier {REF['name']} attempts",
            "ref_curve": [(round(m / 1e3, 1), round(s, 2)) for m, s in curve] if curve else None}

def parts(r):
    """(value of the first arm, the other figures) as the card and the run row show them: reference at matched tokens, then the other arms."""
    fmt = (lambda a: f"{a['score']:.1f}") if r["scored"] else (lambda a: f"{a['levels']:.2f}")
    t = r["arms"][0]; rest = []
    if t["ref"] is not None: rest.append(f"reference {t['ref']:.1f} " + ("at matched tokens" if t["ref_k"] > t["k"] - 0.5 else "at its end"))  # past the reference's end the tokens no longer match
    return fmt(t), rest + [f"{a['name']} {fmt(a)}" for a in r["arms"][1:]]

def text(r, bold=False):
    """One line for the banner's run row and MODAL_STATUS "Latest": score · reference · base · tok/s."""
    b = []; speed = r.get("last30") or r.get("mean")
    if r["collector"]: b.append(f"{r['recorded']} points recorded · {r['tok_total'] / 1e6:.1f}M tokens")
    elif r["arms"]:
        v, rest = parts(r); v += "%" if r["scored"] else ""
        name = "" if r["arms"][0]["name"] == "treatment" else r["arms"][0]["name"] + " "  # several treatments: each by its arm name
        b += [("" if r["scored"] else "levels gained per attempt: ") + name + (f"<b>{v}</b>" if bold else v)] + [x.replace(" at matched tokens", "") for x in rest]
    if speed: b.append(f"{speed:,.0f} tok/s")
    return " · ".join(b)

def chart(r, title, cap=True):
    """Score (or levels) against tokens per attempt: each arm's path so far and the reference dashed."""
    if r["collector"] or not r["arms"]: return ""
    lim = max(a["k"] for a in r["arms"]) * 1.25
    s = [{"name": "concurrent base" if a["name"] == "base" else a["name"], "points": a["track"], "area": False} for a in r["arms"]]
    colors = [c for a, c in zip(r["arms"], ["#2f9e44", "#e8590c", "#7048e8", "#c2255c"] * 2) if a["name"] != "base"] + ["#1c7ed6"] * any(a["name"] == "base" for a in r["arms"])
    if r["ref_curve"]: s.append({"name": "reference", "points": [p for p in r["ref_curve"] if p[0] <= lim], "dash": True, "area": False}); colors.append("#98a2b3")
    spec = {"type": "line", "x": "thousand tokens per attempt", "y": "mean game score" if r["scored"] else "levels per attempt", "smooth": False, "colors": colors, "series": s}
    note = f"<p class='cap'>dashed = {r['ref_name']}, same games, same tokens</p>" if r["ref_curve"] and cap else ""
    return f"<div class='chart'><h3>{title}</h3><div class='chart' data-chart='" + json.dumps(spec) + f"'></div>{note}</div>"
