"""Render MODAL_STATUS.html (the "What is running" dashboard) from the queue rows and live feeds; called by mon.py."""
import datetime, html, re

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2330;--mut:#667085;--line:#e4e7ec;--bar:#e9edf3;--fill:#4f7cff}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--ink:#e6e9ef;--mut:#98a2b3;--line:#2a313d;--bar:#262d39;--fill:#6d93ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:34px 0 10px}.sub{color:var(--mut);font-size:13px;margin-bottom:22px}
.sum{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px}.pill{background:var(--card);color:var(--ink);border:1px solid var(--arm,var(--line));border-radius:999px;padding:4px 12px;font:inherit;font-size:13px;cursor:pointer}.pill.on{background:var(--arm,var(--ink));color:var(--card);border-color:var(--arm,var(--ink))}.pill.on b{color:inherit}.card.hide{display:none}.sep{width:1px;background:var(--line);margin:0 6px}
.card{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--arm);border-radius:10px;padding:16px 18px;margin-bottom:12px;display:grid;grid-template-columns:230px 1fr 210px;gap:20px}
@media (max-width:860px){.card{grid-template-columns:1fr}}
.arm{display:inline-block;background:var(--arm);color:#fff;border-radius:5px;padding:1px 8px;font-size:12px;font-weight:600;letter-spacing:.02em}
.id{font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mut);margin-top:8px;word-break:break-all}.gpu{font-size:13px;margin-top:6px}
.what{font-size:15px}.k{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.v{font-size:20px;font-weight:650;font-variant-numeric:tabular-nums}.v small{font-size:12px;font-weight:400;color:var(--mut)}
.bar{height:6px;background:var(--bar);border-radius:3px;margin:8px 0 12px;overflow:hidden}.bar i{display:block;height:100%;background:var(--fill)}
.m{font-size:13px;color:var(--mut);font-variant-numeric:tabular-nums}.m b{color:var(--ink);font-weight:600}
ul{padding-left:20px;margin:0}li{margin-bottom:8px}
"""
JS = """
const pills=[...document.querySelectorAll('.pill')],cards=[...document.querySelectorAll('.card')];
const sel={arm:new Set(),site:new Set()};
function show(){pills.forEach(p=>p.classList.toggle('on',p.dataset.v?sel[p.dataset.k].has(p.dataset.v):!sel[p.dataset.k].size));
cards.forEach(c=>c.classList.toggle('hide',(sel.arm.size&&!sel.arm.has(c.dataset.arm))||(sel.site.size&&!sel.site.has(c.dataset.site))));
const enc=k=>[...sel[k]].map(encodeURIComponent).join(',');
history.replaceState(null,'',(sel.arm.size||sel.site.size)?'#'+enc('arm')+'|'+enc('site'):location.pathname)}
pills.forEach(p=>p.onclick=()=>{const s=sel[p.dataset.k],v=p.dataset.v;if(!v)s.clear();else if(s.has(v))s.delete(v);else s.add(v);show()});
const h=location.hash.slice(1).split('|'),ok=(k,v)=>pills.some(p=>p.dataset.k===k&&p.dataset.v===v);
['arm','site'].forEach((k,i)=>(h[i]||'').split(',').map(decodeURIComponent).forEach(v=>{if(v&&ok(k,v))sel[k].add(v)}));show();
"""
ARM = {"long run": "#d9480f", "harness": "#2f9e44", "sft": "#7048e8", "throughput": "#1c7ed6", "rl": "#c2255c", "rsi": "#0ca678", "prime": "#7048e8", "arc2": "#e8590c", "envs": "#5c940d"}
OWN_LABEL = {"rsi", "prime", "arc2"}  # arms that keep their own label whatever the run length
is_long = lambda run: "-24h-" in run or "c100-300k" in run  # noqa: E731

def spark(pts, color, w=190, h=44):
    """Small line chart of (x, y) points; y axis fitted to the data."""
    if len(pts) < 2: return ""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx = lambda x: 2 + (w - 4) * (x - x0) / ((x1 - x0) or 1); sy = lambda y: h - 3 - (h - 6) * (y - y0) / ((y1 - y0) or 1)
    d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
    return (f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' role='img'><polyline points='{d}' fill='none' stroke='{color}' stroke-width='1.6' stroke-linejoin='round'/>"
            f"<circle cx='{sx(xs[-1]):.1f}' cy='{sy(ys[-1]):.1f}' r='2.6' fill='{color}'/></svg>")

def smooth(pts, k=5):
    return [(pts[i][0], sum(p[1] for p in pts[max(0, i - k + 1):i + 1]) / len(pts[max(0, i - k + 1):i + 1])) for i in range(len(pts))]

def sft_bits(r):
    """Training run: loss per update (running mean of 5) and the selection NLL on games never trained on."""
    m = []
    if r.get("loss") is not None: m.append(f"<div class='v'>{r['loss']:.3f} <small>training loss</small></div>")
    if len(r.get("losses") or []) > 1:
        ys = [y for _, y in r["losses"]]
        m.append(spark(smooth(r["losses"]), "#7048e8") + f"<div class='m'>loss per update, mean of 5 · range <b>{min(ys):.2f}</b>–<b>{max(ys):.2f}</b></div>")
    sel = r.get("sel") or []
    if sel:
        s0 = sel[0][1]
        if len(sel) > 1:
            m.append(f"<div class='v' style='margin-top:8px'>{sel[-1][1]:.3f} <small>selection NLL, unit {sel[-1][0]}</small></div>" + spark(sel, "#2f9e44")
                     + f"<div class='m'>start <b>{s0:.3f}</b> · change <b>{sel[-1][1] - s0:+.3f}</b> (lower is better)</div>")
        else:
            m.append(f"<div class='m' style='margin-top:8px'>selection NLL at start <b>{s0:.3f}</b>" + (f" · quick check <b>{r['light']:+.3f}</b>" if r.get("light") is not None else "") + "</div>")
    m.append(f"<div class='m'>unit <b>{r['step']}</b> of {r['steps']} · pass {r['pass']} of {r['passes']} · about <b>{r['eta_h']:.1f} h</b> left</div>")
    if r.get("stopped"): m.append(f"<div class='m'><b>stopped: {html.escape(str(r['stopped']))}</b></div>")
    if r.get("utc"): m.append(f"<div class='m'>read {html.escape(str(r['utc'])[11:16])} UTC</div>")
    return m

def rl_bits(r):
    """RL pilot, compact: progress, drift against its stop limit as a bar, and whether training moves attempts the intended way."""
    mean = lambda pts: sum(y for _, y in pts) / len(pts)
    m = [f"<div class='v'>{r['updates']} <small>of {r['of']} updates · about {r['left_h']:.1f} h left</small></div>"]
    pr = r.get("probes") or []
    if pr:
        share = max(0.0, min(1.0, abs(pr[-1][1]) / 0.02)); color = "#2f9e44" if share < 0.6 else "#f08c00" if share < 0.9 else "#e03131"
        m.append(f"<div class='m'>Drift from the starting model: <b>{100 * share:.0f}%</b> of the stop limit"
                 f"<div style='height:6px;background:#e9ecef;border-radius:3px;margin-top:3px'><div style='height:6px;width:{100 * share:.0f}%;background:{color};border-radius:3px'></div></div></div>")
    else:
        m.append("<div class='m'>Drift check: first one at update 10</div>")
    bits = []
    for key, name, want in (("up", "rewarded attempts", 1), ("down", "penalised attempts", -1)):
        pts = r.get(key) or []
        if len(pts) >= 10:
            v = mean(pts[-20:]); ok = v * want > 0
            bits.append(f"{name} {'more' if v > 0 else 'less'} likely {'✓' if ok else '✗'}")
    if bits: m.append("<div class='m'>" + " · ".join(bits) + "</div>")
    m.append(f"<div class='m'>GPU memory {100 * r['peak'] / 268:.0f}% used</div>")
    return m

def bench_bits(r):
    """Level-bench run, in the shape of the arena card: score, the reference at matched tokens and the concurrent base, tokens."""
    if r["collector"]:
        return [f"<div class='v'>{r['recorded']} <small>points recorded</small></div>", f"<div class='m'><b>{r['tok_total'] / 1e6:.1f}M</b> tokens</div>"]
    if not r["arms"]: return []
    import bench_feed
    v, rest = bench_feed.parts(r); a = r["arms"][0]; one = a["name"] == "treatment"
    m = [f"<div class='v'>{v}{'%' if r['scored'] else ''} <small>{'score' if r['scored'] and one else html.escape(a['name']) + (' score' if r['scored'] else ' levels per attempt')}</small></div>"]
    if rest: m.append("<div class='m'>" + " · ".join(re.sub(r"([0-9.]+)", r"<b>\1</b>", html.escape(x), count=1) for x in rest) + "</div>")
    m.append(f"<div class='m'><b>{a['tokM']:.1f}M</b> tokens" + (f" · <b>{a['ppM']:,.0f}</b> points/Mtok" if a.get("ppM") else "") + "</div>")
    return m

def is_live(e):
    """A machine is up for this run (reconcile state running or starting)."""
    return e["_ago"] >= 0 and e.get("_state") in ("running", "starting")


def summary(e, r, kind=None):
    """What a card shows, from the run's state (e["_state"]: running / waiting / None) and its feed row r: head line, progress
    bar share (None: no bar), score (or the reason there is none), tok/s (or the reason), other lines."""
    ago, total, state = e["_ago"], e["_total"], e.get("_state")
    c = {"pct": None, "score": None, "score_why": None, "tps": None, "tps_why": None, "lines": [], "state": state}
    hm = lambda t: datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M")
    if ago < 0:
        c.update(head=f"queued · starts in about {-ago:.1f} h · {total:.1f} h planned", state="queued", score_why="not started", tps_why="not started"); return c
    if state == "starting":
        c.update(head=f"starting: machine requested {hm(e['_since'])} UTC · {total:.1f} h planned", score_why="not started", tps_why="not started"); return c
    if state == "waiting":
        c.update(head=f"waiting for a GPU since {hm(e['_since'])} UTC · {total:.1f} h planned", score_why="not playing: no container", tps_why="not running: no container")
    elif state == "running":
        c["head"] = f"{ago:.1f} h of {total:.1f} h planned" if ago <= total else f"past planned end: {ago:.1f} h running, {total:.1f} h planned"
        if ago <= total: c["pct"] = max(0, min(100, 100 * ago / total)) if total else 0
    else:
        c["head"] = f"state unknown ({e.get('_why') or 'no machine check'}) · {ago:.1f} h since planned start"
    b = lambda x: f"<b>{x}</b>"
    if r.get("sft"):
        c["lines"] = sft_bits(r); c["score"] = f"{r['loss']:.3f} <small>training loss</small>" if r.get("loss") is not None else None
        c["score_why"] = c["tps_why"] = "training job: no score or tok/s feed"; return c
    if r.get("rl"):
        c["lines"] = rl_bits(r); c["score_why"] = c["tps_why"] = "training job: no score or tok/s feed"; return c
    if r.get("rl1"):  # RL step 1 probe / loop (tools/ops/rl1_feed.py)
        import rl1_feed
        c["lines"] = rl1_feed.bits(r)
        if r["reward"]: c["score"] = f"{r['reward'][-1][1]:.2f} <small>sb26 reward, round {r['reward'][-1][0]} (round 0 {r['reward'][0][1]:.2f})</small>"
        else: c["score_why"] = "round 0 not played yet"
        if r["tok_s"]: c["tps"] = f"<b>{r['tok_s'][-1]:.0f}</b> tok/s in play"
        else: c["tps_why"] = "no play phase finished yet"
        return c
    if r.get("rsi"):  # RSI loop: progress against its plan and expectations (tools/ops/rsi_feed.py)
        import rsi_feed
        c["lines"] = rsi_feed.bits(r)
        if r["best"]: c["score"] = f"{r['best']['effect']:+.1f} <small>best vs gen 0, pool RHAE ({r['best']['n']} games)</small>"
        elif r["gen0_pool"] is not None: c["score"] = f"{r['gen0_pool']:.1f} <small>gen 0 pool RHAE ({r['gen0_pool_games']} games)</small>"
        else: c["score_why"] = "no pool game scored yet"
        if r["games_per_h"]: c["tps"] = f"<b>{r['games_per_h']:.1f}</b> games per loop GPU-h"
        else: c["tps_why"] = "loop run: games per loop GPU-h after the first quarter hour"
        return c
    if r.get("arc2"):  # ARC-AGI-2 roster trial (mon.arc2_row): read once its progress file syncs
        if r.get("right") is not None: c["score"] = f"{r['right']} of {r['of']} <small>test outputs right ({r['pct']:.1f}%) at {r['tokM']:.2f}M tokens</small>"
        else: c["score_why"] = "progress file has no score"
        if r.get("tps"): c["tps"] = f"<b>{r['tps']:,.0f}</b> tok/s over the run"
        else: c["tps_why"] = "progress file has no tok/s"
        return c
    if r.get("bench"):
        a = r["arms"][0] if r.get("arms") else None
        if r.get("scored") and a and a.get("score") is not None:
            c["score"] = f"{a['score']:.1f} <small>mean game score{'' if a['name'] == 'treatment' else ', ' + html.escape(a['name'])}</small>"
            if a.get("ref") is not None: c["lines"].append(f"<div class='m'>reference at matched tokens <b>{a['ref']:.1f}</b></div>")
        elif a: c["score"] = f"{a['levels']:.2f} <small>levels per attempt{'' if a['name'] == 'treatment' else ', ' + html.escape(a['name'])}</small>"
        elif r.get("clears") is not None: c["score"] = f"{r['clears']} <small>attempts cleared so far</small>"
        else: c["score_why"] = "no attempt has a level mark yet"
        c["lines"].append(f"<div class='m'><b>{r['tok_total'] / 1e6:.1f}M</b> tokens · "
                          + (f"{b(r['done'])} of {r.get('planned') or '?'} attempts done · {r.get('running') or 0} running" if r.get("done") is not None else f"{r.get('launched') or 0} attempts launched")
                          + (f" · {b(r['clears'])} attempts cleared" if r.get("clears") is not None else "") + "</div>")
    elif r.get("min") is not None or r.get("tokM") is not None:
        if r.get("score") is not None:
            c["score"] = f"{r['score']:.2f}% <small>score</small>"
            if r.get("tokM"): c["lines"].append(f"<div class='m'><b>{r['tokM']:.1f}M</b> tokens · <b>{r['score'] / r['tokM']:.2f}</b> points/Mtok</div>")
        else:
            c["score_why"] = f"no score yet (phase {r.get('phase') or '?'})" if not r.get("tokM") else "no scorecard read yet"
            if r.get("tokM"): c["lines"].append(f"<div class='m'><b>{r['tokM']:.1f}M</b> tokens</div>")
    else:
        why = (f"{kind} job: writes its result at the end, no live score or tok/s" if kind else
               "setting up: no progress record yet" if state == "running" and ago < 0.75 else "no live feed found")
        c["score_why"] = why; c["tps_why"] = c["tps_why"] or why
    if state == "waiting": return c
    if r.get("kv") is not None or r.get("hit") is not None:
        kv = f"cache use <b>{100 * r['kv']:.0f}%</b>" if r.get("kv") is not None else ""; hit = f"prefix hits <b>{r['hit']:.0f}%</b>" if r.get("hit") is not None else ""
        c["lines"].append("<div class='m'>" + " · ".join(x for x in (kv, hit) if x) + "</div>")
    if "cap" in r:  # always show the stream cap and the 30-min mean of running streams on serving runs
        run30 = f"<b>{r['run30']:.1f}</b>" if r.get("run30") is not None else "n/a"
        c["lines"].append(f"<div class='m'>streams: cap <b>{r.get('cap') or '?'}</b> · {run30} running, 30-min mean</div>")
    if state == "starting":
        c.update(head=f"starting: machine requested {hm(e['_since'])} UTC · {total:.1f} h planned", score_why="not started", tps_why="not started"); return c
    if state == "waiting": return c
    speed = r.get("last30") or r.get("mean")
    if speed: c["tps"] = f"<b>{speed:,.0f}</b> tok/s" + (", last 30 min" if r.get("last30") else " since start") + (f" · <b>{r['mean']:,.0f}</b> since start" if r.get("last30") and r.get("mean") else "")
    elif not c["tps_why"]: c["tps_why"] = f"not started (phase {r.get('phase')})" if r.get("phase") else "no tokens generated yet"
    play = r.get("play_min") if r.get("play_min") is not None else r.get("min")
    if play is not None and not r.get("bench"): c["lines"].append(f"<div class='m'>after {play / 60:.1f} h of play</div>")
    return c

def page(live, nxt, rows, now, ended=(), failures=()):
    """live: queue entries with _ago/_total hours and _card; rows: run -> monitor row; ended: runs dropped in the last hour."""
    out = ["<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
           "<meta http-equiv='refresh' content='60'><title>What is running</title><style>" + CSS + "</style></head><body><main>",
           "<h1>What is running</h1>"] + ([f"<div style='color:#e03131;font-size:13px;margin-bottom:6px'>Dashboard check: {len(failures)} card(s) incomplete: " + html.escape("; ".join(failures))[:600] + "</div>"] if failures else []) + [f"<div class='sub'>Updated {now:%Y-%m-%d %H:%M} UTC · regenerated every 5 minutes · most recently started first, queued runs last · <a href='docs/html/training-live.html'>Training runs: live metrics</a></div>"]
    live = [e for e in live if "cpu" not in str(e.get("gpu", "")).lower()]
    live = [dict(e, arm="long run") if is_long(e["run"]) and e["arm"] not in OWN_LABEL else e for e in live]  # label rule: 24 h runs and c100 300K runs; every other run keeps its arm
    live = [dict(e, arm="throughput" if e["run"].startswith("b2-") else "harness") if e["arm"] == "long run" and not is_long(e["run"]) else e for e in live]  # a "long run" queue row on another run is a mislabel
    arms, sites = {}, {}
    for e in live:
        e["_site"] = "nebius" if e.get("platform") == "nebius" else "modal"
        if is_live(e):  # label counts: GPUs actually up, not queued or waiting for a GPU
            g = re.match(r"\s*(\d+)\s*x", str(e.get("gpu") or "")) or re.search(r":\s*(\d+)\s*$", str(e.get("gpu") or ""))
            n = 1 if "L4" in str(e.get("gpu") or "") or not g else int(g.group(1))  # "2x B300" / "B300:2" count 2; a 4xL4 counts one
            arms[e["arm"]] = arms.get(e["arm"], 0) + n
            sites[e["_site"]] = sites.get(e["_site"], 0) + n
    up = sum(map(is_live, live))
    out.append("<div class='sum'>" + f"<button class='pill' data-k='arm' data-v=''><b>{up}</b> running" + (f" · {len(live) - up} queued" if len(live) > up else "") + "</button>"
               + "".join(f"<button class='pill' data-k='arm' data-v='{html.escape(a)}' style='--arm:{ARM.get(a, '#667085')}'>{html.escape(a)} · {n}</button>" for a, n in sorted(arms.items()))
               + "<span class='sep'></span><button class='pill' data-k='site' data-v=''>all machines</button>"
               + "".join(f"<button class='pill' data-k='site' data-v='{k}'>{k.capitalize()} · {n}</button>" for k, n in sorted(sites.items())) + "</div>")
    for e in live:
        c = e.get("_card") or summary(e, rows.get(e["run"]) or {})
        where = "Nebius" if e.get("platform") == "nebius" else "Modal"
        m = []
        if c["score"]: m.append(f"<div class='v'>{c['score']}</div>")
        elif c["score_why"]: m.append(f"<div class='m'>{html.escape(c['score_why'])}</div>")
        m += c["lines"]
        if c["tps"]: m.append(f"<div class='m'>{c['tps']}</div>")
        elif c["tps_why"] and (c["score"] or c["tps_why"] != c["score_why"]): m.append(f"<div class='m'>{html.escape(c['tps_why'])}</div>")
        bar = f"<div class='bar'><i style='width:{c['pct']:.0f}%'></i></div>" if c["pct"] is not None else "<div style='height:8px'></div>"
        out.append(f"<section class='card' data-arm='{html.escape(e['arm'])}' data-site='{e['_site']}' style='--arm:{ARM.get(e['arm'], '#667085')}'><div><span class='arm'>{html.escape(e['arm'])}</span>"
                   f"<div class='id'>{html.escape(e['run'])}</div><div class='gpu'>{where} · {html.escape(e.get('gpu', '?'))}</div></div>"
                   f"<div class='what'>{html.escape(e['what'])}</div>"
                   f"<div><div class='k'>{html.escape(c['head'])}</div>{bar}{''.join(m)}</div></section>")
    if ended:
        out.append("<h2>Ended in the last hour</h2><ul>" + "".join(f"<li><b>{html.escape(x['run'])}</b> ({html.escape(x.get('arm') or '')}) at {datetime.datetime.fromtimestamp(x['t'], datetime.timezone.utc):%H:%M} UTC"
                   + (f": {html.escape(x['final'])}" if x.get("final") else ": final score not read") + "</li>" for x in ended) + "</ul>")
    out.append("<h2>Next</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in nxt) + "</ul></main><script>" + JS + "</script></body></html>")
    return "\n".join(out)
