"""Write the live banner of each arm status page (docs/html), the index cards and the live charts. Banner text comes from the arm's
running log (docs/arm-logs/<log>.md) unless arm-state.json holds hand-written fields stamped newer than the log; runs come from the
reconciled queue, the live feeds and the ledger."""
import datetime, hashlib, html, json, re, subprocess
import paths
ROOT = paths.ROOT_PATH; H = ROOT / "docs/html"; LOGS = ROOT / "docs/arm-logs"
esc = html.escape
HAND = ("chip", "chip_text", "headline", "result", "next", "ask")
MON = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
OWNER = re.compile(r"\bwait(s|ing)? (on|for) the maintainer|\bawaits? the maintainer|\bowner (to )?chooses?\b", re.I)

def spend(prefixes):
    d = paths.read_json(ROOT / "docs/spend.json"); tot = 0.0
    for v in [d.get("runs")] + [x.get("runs") for x in d.values() if isinstance(x, dict)]:
        for r in v if isinstance(v, list) else []:
            if isinstance(r, dict) and str(r.get("run_id") or r.get("run") or "").split("-")[0] in prefixes: tot += r.get("charged_bound_usd") or 0
    return tot

def nums(r):
    if not r or "state" in r: return ""
    if r.get("rl"): return f"update {r['updates']} of {r['of']}"
    if r.get("sft"): return f"loss {r['loss']:.3f} · unit {r['step']} of {r['steps']}" if r.get("loss") is not None else ""
    if r.get("bench"):
        import bench_feed
        return bench_feed.text(r, True)
    b = []
    if r.get("score") is not None: b.append(f"<b>{r['score']:.2f}%</b>")
    if r.get("last30") or r.get("mean"): b.append(f"{(r.get('last30') or r.get('mean')):,.0f} tok/s")
    if r.get("kv") is not None: b.append(f"cache {100 * r['kv']:.0f}%")
    if r.get("hit") is not None: b.append(f"prefix hits {r['hit']:.0f}%")
    return " · ".join(b)

def runs_table(live, full, labels):
    groups = {}
    for e in live:
        lab = next((v for k, v in labels.items() if e["run"].startswith(k)), re.sub(r"-nebius-\d+$", "", e["run"]))
        groups.setdefault((lab, e["_ago"] < 0), []).append(e)
    rows = []
    for (lab, queued), es in sorted(groups.items(), key=lambda kv: (kv[0][1], -max(e["_total"] for e in kv[1]), [int(n) for n in re.findall(r"\d+", kv[0][0])])):
        e = max(es, key=lambda x: x["_ago"]); n = f" ×{len(es)}" if len(es) > 1 else ""
        c = e["_card"]
        if queued:
            rows.append(f"<tr><td>{esc(lab)}{n}</td><td class='p'></td><td class='m'>{esc(c['head'])}</td></tr>"); continue
        bar = f"<i><b style='width:{c['pct']:.0f}%'></b></i>" if c["pct"] is not None else ""
        x = " · ".join(v for v in (c["score"] or esc(c["score_why"] or ""), c["tps"] or esc(c["tps_why"] or "")) if v) if len(es) == 1 else ""
        rows.append(f"<tr><td>{esc(lab)}{n}</td><td class='p'>{bar}</td><td class='m'>{esc(c['head'])}{' · ' + x if x else ''}</td></tr>")
    return "<table class='bn-runs'>" + "".join(rows) + "</table>" if rows else ""

def put(page, a, b, body):
    p = H / page
    if not p.exists(): return
    t = p.read_text(); i, j = t.find(a), t.find(b)
    if i < 0 or j < 0: return
    n = t[:i + len(a)] + "\n" + body + "\n" + t[j:]
    if n != t: p.write_text(n)

def words(page):
    t = (H / page).read_text(); t = t[t.find("<!--BANNER:end-->"):]
    t = re.sub(r"<style.*?</style>|<script.*?</script>|<svg.*?</svg>|<!--.*?-->|<nav.*?</nav>", " ", t, flags=re.S)
    return len(re.sub(r"<[^>]+>", " ", t).split())

def clean(s, n):
    s = re.sub(r"\s+", " ", re.sub(r"[`*]", "", re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s))).strip()
    w = s.split(); return " ".join(w[:n]) + (" …" if len(w) > n else "")

def log_items(lines):
    """Top-level items of one log section: a bullet or numbered line plus its indented continuation (command lines left out)."""
    out, open_ = [], False
    for x in lines:
        m = re.match(r"^ ?(?:[-*]|\d+\.)\s+(.*)", x)
        if m: out.append(m.group(1)); open_ = True
        elif open_ and x.strip() and x[:1] in " \t" and not x.strip().startswith("`") and not re.match(r"\s+(?:[-*]|\d+\.)\s", x): out[-1] += " " + x.strip()
        elif not x.strip() or x[:1] not in " \t": open_ = False
    return out

def read_log(name):
    """Now / Next items, the newest dated Done item and when the log last changed (its last commit, or the file if edited since)."""
    p = LOGS / f"{name}.md"
    if not p.exists(): return None
    sec, cur = {}, None
    for x in p.read_text().splitlines():
        m = re.match(r"^##\s+(\w+)", x)
        if m: cur = m.group(1).lower(); sec[cur] = []
        elif cur: sec[cur].append(x)
    try: c = int(subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%ct", "--", str(p)], capture_output=True, text=True, timeout=30).stdout.strip() or 0)
    except Exception: c = 0
    dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", str(p)], capture_output=True).returncode != 0
    done, best = log_items(sec.get("done", [])), None
    for i, x in enumerate(done):  # undated items rank last; ties go to the one listed first
        m = re.match(r"(\d{1,2}) ([A-Z][a-z]{2})\w*(?: (\d{1,2}):(\d{2}))?(?: UTC)?:?\s*(.*)", x)
        k = (MON.get(m.group(2), 0), int(m.group(1)), int(m.group(3) or -1), int(m.group(4) or 0)) if m else (0, 0, -1, 0)
        if best is None or k > best[0]: best = (k, i, m)
    res = None
    if best:
        m = best[2]
        res = [f"{m.group(1)} {m.group(2)}" + (f" {m.group(3)}:{m.group(4)}" if m.group(3) else ""), m.group(5)] if m else ["", done[best[1]]]
    return {"now": log_items(sec.get("now", [])), "next": log_items(sec.get("next", [])), "result": res, "t": max(c, p.stat().st_mtime if dirty or not c else 0)}

def stamp_edits(st, now):
    """Stamp 'edited' on arm entries whose hand-written fields changed since the last pass (their hashes live in the scratch folder)."""
    f = paths.SCRATCH / "arm-state-hand.json"
    try: old = json.loads(f.read_text())
    except Exception: old = None
    cur = {k: hashlib.sha1(json.dumps([a.get(x) for x in HAND]).encode()).hexdigest() for k, a in st["arms"].items()}
    changed = [k for k in cur if old and k in old and old[k] != cur[k]]
    if changed:
        import fcntl
        with open(H / "arm-state.json", "r+") as g:
            fcntl.flock(g, fcntl.LOCK_EX); d = json.load(g); ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            for k in changed:
                if k in d["arms"]: d["arms"][k]["edited"] = st["arms"][k]["edited"] = ts
            s = json.dumps(d, indent=1, ensure_ascii=False) + "\n"; g.seek(0); g.write(s); g.truncate()
    f.write_text(json.dumps(cur))

def view(a, started, now):
    """Banner fields of one arm: from its log, or the hand-written ones while their 'edited' stamp is newer than the log."""
    log = read_log(a["log"]) if a.get("log") else None
    try: ed = datetime.datetime.fromisoformat(a["edited"].replace("Z", "+00:00")).timestamp() if a.get("edited") else 0
    except Exception: ed = 0
    v = {k: a.get(k) for k in HAND}
    if log and (log["now"] or log["next"]) and ed <= log["t"]:
        items = log["now"]; ask = next((x for x in items + log["next"] if OWNER.search(x)), "")
        v = {"headline": clean(items[0], 32) if items else "Nothing running.", "more": [clean(x, 24) for x in items[1:4]],
             "next": clean(log["next"][0], 36) if log["next"] else "nothing queued", "ask": clean(ask, 28) if ask else "",
             "result": [log["result"][0], clean(log["result"][1], 30)] if log["result"] else a.get("result"),
             "chip": "live" if started else "next", "chip_text": f"{started} running" if started else "waiting" if ask else "nothing running"}
    v["ago"] = (now.timestamp() - log["t"]) / 60 if log and log["t"] else None
    v["result"] = (list(v.get("result") or []) + ["", ""])[:2]
    return v

def run(live, full, now, failures=()):
    st = json.loads((H / "arm-state.json").read_text()); stamp = f"State as of {now.day} {now:%b, %H:%M} UTC · refreshed every 5 minutes"
    try: stamp_edits(st, now)
    except Exception as exc: print("edit stamp failed:", exc)
    arms, views = [], {}
    for key, a in st["arms"].items():
        mine = [e for e in live if e["arm"] in a["queue_arms"] and "cpu" not in str(e.get("gpu", "")).lower()]
        started = sum(e["_ago"] >= 0 and e.get("_state") in ("running", "starting") for e in mine)  # live machines only
        v = views[key] = view(a, started, now); ask = v.get("ask") or ""
        age = "" if v["ago"] is None else f" · log updated {v['ago']:.0f} min ago" if v["ago"] < 120 else f" · log updated {v['ago'] / 60:.1f} h ago"
        more = "".join(f"<li>{esc(x)}</li>" for x in v.get("more") or [])
        body = (f"<section class='banner'><div class='bn-top'><span class='chip {v['chip']}'>{esc(v['chip_text'] or '')}</span><span>{stamp}{age}</span></div>"
                f"<div class='bn-head'>{esc(v['headline'] or '')}</div>"
                + (f"<ul style='margin:-6px 0 12px;padding-left:18px;font-size:13px;color:var(--mut)'>{more}</ul>" if more else "")
                + f"<div class='bn-grid'><div><label>Latest result</label><b>{esc(v['result'][0])}</b>{esc(v['result'][1])}</div>"
                f"<div><label>Next</label>{esc(v['next'] or '')}</div>"
                f"<div class='{'ask' if ask else ''}'><label>Waiting on you</label>{esc(ask) if ask else 'nothing'}</div>"
                f"<div><label>Spend</label><b>${spend(a['spend_prefixes']):,.0f}</b>of ${a['spend_cap']:,}</div></div>"
                + runs_table(mine, full, st["labels"]) + "</section>")
        put(a["page"], "<!--BANNER:start-->", "<!--BANNER:end-->", body)
        arms.append(f"<a href='{a['page']}' style='--c:{a['color']}'><b>{esc(a['name'])}</b><span><span class='chip {v['chip']}'>{esc(v['chip_text'] or '')}</span></span>"
                    f"<span>{esc(v['headline'] or '')} <small>{started} running</small></span></a>")
    try: live_charts(st, live, full)
    except Exception as exc: print("live charts failed:", exc)
    red = f"<p style='color:#e03131;font-size:13px'>Dashboard check: {len(failures)} live card(s) incomplete, see MODAL_STATUS.html</p>" if failures else ""
    put("index.html", "<!--ARMS:start-->", "<!--ARMS:end-->", f"<p class='updated'>{stamp}</p>{red}<div class='bn-arms'>" + "".join(arms) + "</div>")
    strip = "".join(f"<div style='--c:{a['color']}'><b>{esc(views[k]['result'][0])}</b><span>{esc(a['name'])}: {esc(clean(re.split(r'[;(]', views[k]['result'][1])[0], 14))}</span></div>" for k, a in st["arms"].items())
    put("index.html", "<!--STRIP:start-->", "<!--STRIP:end-->", f"<div class='stats'>{strip}</div>")
    return {a["page"]: words(a["page"]) for a in st["arms"].values() if (H / a["page"]).exists()}, views

def short(run, labels):
    return next((v for k, v in labels.items() if run.startswith(k)), re.sub(r"-(nebius|modal)-\d{8}$", "", run))

def family(run):
    return re.sub(r"-v\d+[a-z0-9]*(?=-|$)", "", re.sub(r"-\d{8}$", "", run))

def chart(title, series, x, y, cap):
    return f"<div class='chart'><h3>{title}</h3><div class='chart' data-chart='{esc(json.dumps({'type': 'line', 'x': x, 'y': y, 'smooth': False, 'series': series}))}'></div><p class='cap'>{cap}</p></div>"

def live_charts(st, live, full):
    """Per arm: one chart per live level-bench run (bench_charts), and for live arena runs (live_charts) score by generated tokens and
    tok/s by hour, with the newest finished run of the same family and the arm's chart_refs dashed. Pages keep their last charts when
    nothing is live."""
    import bench_feed, compare_runs
    S = compare_runs.S
    for a in st["arms"].values():
        charts = []
        if a.get("bench_charts"):
            for r in full.values():
                c = bench_feed.chart(r, f"{esc(re.sub(r'-nebius-[0-9]+$', '', r['run']))}: score by tokens per attempt", not charts) if r.get("bench") and r["run"].split("-")[0] in a["spend_prefixes"] else ""
                if c: charts.append(c)
        if a.get("live_charts"):
            mine = [e["run"] for e in live if e["arm"] in a["queue_arms"] and e["_ago"] >= 0 and "cpu" not in str(e.get("gpu", "")).lower()
                    and not {"state", "bench", "sft", "rl"} & set(full.get(e["run"]) or {"state": 1}) and (S / f"{e['run']}.jsonl").exists()]
            if mine:
                names = [short(r, st["labels"]) for r in mine]
                runs = [(n if names.count(n) == 1 else short(r, {}), r, 1.0, False) for n, r in zip(names, mine)]; dashed = []
                fams = {family(r) for r in mine}
                prev = sorted((p for p in S.glob("*.jsonl") if family(p.stem) in fams and p.stem not in mine), key=lambda p: p.stat().st_mtime)
                if prev: runs.append(("previous: " + short(prev[-1].stem, {}), prev[-1].stem, 1.0, False)); dashed.append(runs[-1][0])
                for n, r in (a.get("chart_refs") or {}).items():
                    if (S / f"{r}.jsonl").exists(): runs.append((n, r, 1.0, False)); dashed.append(n)
                ser = compare_runs.series(False, runs)
                live_n = [n for n, *_ in runs[:len(mine)]]
                lim = [max((ser[n][i][-1][0] for n in live_n if ser[n][i]), default=1) * 1.25 for i in (0, 1)]
                extra = {n: p for n, p in compare_runs.ARC2_REF.items()} if any("-full-" in r for r in mine) else {}
                def s(i):
                    out = [{"name": n, "points": [p for p in ser[n][i] if p[0] <= lim[i] or n in live_n], "dash": n in dashed} for n in ser if ser[n][i]]
                    return out + ([{"name": n, "points": p, "dash": True} for n, p in extra.items()] if i == 0 else [])
                cap = "dashed = " + ", ".join(dashed + list(extra)) if dashed or extra else "live runs of this arm"
                charts += [chart("Score by generated tokens", s(0), "million tokens", a.get("chart_y", "score %"), cap + "; refreshed every 5 minutes"),
                           chart("Tokens per second, 15-minute bins", s(1), "hours of play", "tok/s", "gaps are restarts")]
            if mine and not any("&quot;points&quot;: [[" in c for c in charts[-2:]):
                charts[-2:] = ["<p class='cap'>Live: " + esc(", ".join(short(r, st["labels"]) for r in mine)) + ". No scored data yet; the charts appear with the first scorecard read.</p>"]
        if charts:
            put(a["page"], "<!--LIVE:start-->", "<!--LIVE:end-->", f"<div class='g2'>{''.join(charts[:4])}</div>" if len(charts) > 1 else charts[0])
