"""Trim bed read-out. argv: reference_arm run_dir [run_dir ...]   (python -I; run folders are downloaded data)
Per arm: attempts, post-script tokens, levels and cleared weight per Mtok at 50/100/200K, actions per cleared level,
computed prompt tokens per generated token, trims and summary tokens; paired bootstrap of the ratio to the reference."""
import json, sys, random, collections, statistics
from pathlib import Path

ref = sys.argv[1]
MARKS = (50000, 100000, 150000, 200000)
rows, reqs = [], collections.defaultdict(lambda: [0, 0, 0, 0])  # label -> prompt, cached, completion, requests
events = {}
for root in sys.argv[2:]:
    bench = next(p for p in (Path(root) / "arena/trials/level-bench", Path(root) / "working/trials/level-bench") if p.is_dir())
    report = json.loads((bench / "level-bench.json").read_text())
    for r in json.loads((bench / "requests.json").read_text()):
        u = r.get("usage") or {}
        if r.get("status") != 200 or not u:
            continue
        x = reqs[r["instance_id"]]
        x[0] += u.get("prompt_tokens", 0); x[1] += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        x[2] += u.get("completion_tokens", 0); x[3] += 1
    for r in report["attempts"]:
        r["run"] = Path(root).name
        rows.append(r)
        ev = collections.Counter(); summ = []; drains = []
        log = bench / "attempts" / r["label"] / "log.jsonl"
        if log.is_file():
            for line in log.open():
                if not line.startswith('{"kind":"'):
                    continue
                kind = line[9:line.index('"', 9)]
                if kind in ("context_drain", "compaction", "fraction_compaction", "playbook_handoff", "level_bench_script_end"):
                    ev[kind] += 1
                    d = json.loads(line)
                    if kind == "compaction":
                        summ.append(((d.get("usage") or {}).get("completion_tokens", 0), d.get("summary_chars", 0)))
                    if kind == "context_drain":
                        drains.append((d.get("tokens_before"), d.get("tokens_after")))
                    if kind == "playbook_handoff":
                        summ.append((0, d.get("chars", 0)))
        events[r["label"]] = (ev, summ, drains)


def weights(total):
    s = total * (total + 1) / 2
    return {l: 100.0 * l / s for l in range(1, total + 1)}


def measure(r, mark):
    """(tokens used up to mark, levels, weight, actions on those levels) after the script."""
    base = (r.get("script") or {}).get("completion_tokens", 0)
    used = min(mark, max(0, r["completion_tokens"]))
    w = weights(int(r.get("total_levels") or 1))
    levels = weight = actions = 0
    for lv in r.get("level_rows") or []:
        after = lv["at_tokens"] - base
        if 0 < after <= mark:
            levels += 1; weight += w.get(lv["level"], 0.0); actions += lv["actions"]
    if r["outcome"] == "won":
        last = (r.get("level_rows") or [{}])[-1].get("at_tokens", 0) - base
        if last <= mark:
            used = min(used, max(last, 1))
    return used, levels, weight, actions


ok = [r for r in rows if (r.get("script") or {}).get("ok") and r["outcome"] not in ("error", "replay_mismatch")]
print("attempts", len(rows), "usable", len(ok), "outcomes", dict(collections.Counter(r["outcome"] for r in rows)),
      "mismatch", sum(1 for r in rows if not (r.get("script") or {}).get("ok")))
arms = sorted({r["arm"] for r in ok})
by = {a: {r["point"]: r for r in ok if r["arm"] == a} for a in arms}
print(f"{'arm':16s} {'n':>3s} {'Mtok':>6s} | " + " | ".join(f"lv/Mtok wt/Mtok @{m//1000}K" for m in MARKS) + " | act/level  prompt/gen  cached%  trims/att  summary tok (median, max)  kept after trim  computed prompt per trim")
uncached = {}
for a in arms:
    rs = list(by[a].values())
    cells = []
    for m in MARKS:
        t = l = w = 0
        for r in rs:
            u, lv, wt, _ = measure(r, m); t += u; l += lv; w += wt
        cells.append(f"{1e6*l/max(t,1):7.2f} {1e6*w/max(t,1):7.1f}      ")
    t = l = act = 0
    for r in rs:
        u, lv, _, ac = measure(r, MARKS[-1]); t += u; l += lv; act += ac
    p = c = g = 0
    for r in rs:
        x = reqs.get(r["label"], [0, 0, 0, 0]); p += x[0]; c += x[1]; g += x[2]
    trims = [sum(events[r["label"]][0][k] for k in ("context_drain", "fraction_compaction")) for r in rs]
    summ = [s[0] for r in rs for s in events[r["label"]][1]]
    chars = [s[1] for r in rs for s in events[r["label"]][1]]
    after = [d[1] for r in rs for d in events[r["label"]][2] if d[1]]
    print(f"{a:16s} {len(rs):3d} {t/1e6:6.2f} | " + " | ".join(cells) + f" | {act/max(l,1):8.1f}  {(p-c)/max(g,1):9.2f}  {100*c/max(p,1):6.1f}  {statistics.mean(trims) if trims else 0:8.2f}"
          f"  {statistics.median(summ) if summ else 0:.0f}, {max(summ) if summ else 0} ({statistics.median(chars) if chars else 0:.0f} chars)  {statistics.median(after) if after else 0:.0f}  {(p-c)/max(sum(trims),1):.0f}")
    uncached[a] = (p - c) / max(g, 1)
for RMARK in (150000, 200000):
  if ref in by:
    random.seed(0)
    print(f"ratio to {ref} (cleared weight per Mtok at {RMARK//1000}K or the tokens reached, points both arms have; paired bootstrap over points, 2,000 draws)")
    for a in arms:
        if a == ref:
            continue
        pts = sorted(set(by[a]) & set(by[ref]))
        if not pts:
            continue
        def ratio(sample, k=2):
            num = den = 0.0
            for arm, sign in ((a, 0), (ref, 1)):
                t = w = 0.0
                for p in sample:
                    u, lv, wt, _ = measure(by[arm][p], RMARK); t += u; w += (wt if k == 2 else lv)
                if sign: den = w / max(t, 1)
                else: num = w / max(t, 1)
            return num / den if den else float("nan")
        draws = sorted(x for x in (ratio([random.choice(pts) for _ in pts]) for _ in range(2000)) if x == x)
        lev = ratio(pts, 1)
        lo, hi = (draws[int(.025*len(draws))], draws[int(.975*len(draws))-1]) if draws else (float("nan"),) * 2
        # priced: tokens per second scale as 1 / (1 + 0.19 x computed prompt per generated token) (throughput arm, 9 Oct)
        speed = (1 + 0.19 * uncached[ref]) / (1 + 0.19 * uncached[a])
        print(f"  {a:16s} points {len(pts):3d}  weight ratio {ratio(pts):5.2f}  [{lo:.2f}, {hi:.2f}]  levels ratio {lev:5.2f}"
              f"  speed x{speed:.2f}  net {ratio(pts) * speed:5.2f}")
