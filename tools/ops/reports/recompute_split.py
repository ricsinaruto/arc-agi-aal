"""Split a run's recomputed prompt tokens by cause (re-read after a trim, whole prefix lost, new content). argv: requests.json of the run."""
import json, sys
from collections import defaultdict
p = sys.argv[1]
R = json.load(open(p)); t0 = min(r["started_monotonic"] for r in R)
by = defaultdict(list)
for r in R:
    if r.get("status") == 200 and r.get("usage"): by[r["instance_id"]].append(r)
bands = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 12), (12, 16)]
agg = {b: defaultdict(float) for b in bands}
for g, rs in by.items():
    rs.sort(key=lambda r: r["started_monotonic"]); prev = None
    for r in rs:
        u = r["usage"]; pt = u["prompt_tokens"]; ca = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0; comp = pt - ca
        h = (r["started_monotonic"] - t0) / 3600; b = next((x for x in bands if x[0] <= h < x[1]), None)
        if b is None: prev = (pt, u["completion_tokens"], r["started_monotonic"] + r["seconds"]); continue
        A = agg[b]; A["n"] += 1; A["gen"] += u["completion_tokens"]; A["comp"] += comp; A["prompt"] += pt
        if prev is None: A["first"] += comp
        else:
            grow = pt - prev[0] - prev[1]; gap = r["started_monotonic"] - prev[2]
            if pt < prev[0] - 2000: A["trim"] += comp; A["trims"] += 1          # history was cut: drain, loop rule or compaction
            else:
                new = max(0, min(comp, grow + prev[1]))  # appended content incl. the previous reply if it was not cached
                app = max(0, min(comp, grow)); A["append"] += app
                rest = comp - app
                if rest > 0:
                    if ca < 0.5 * prev[0]: A["lost"] += rest; A["losts"] += 1; A["lost_gap"] += gap  # most of the old prefix recomputed: evicted
                    else: A["partial"] += rest; A["partials"] += rest > 200
        prev = (pt, u["completion_tokens"], r["started_monotonic"] + r["seconds"])
print("band h | requests | computed per request | of which: appended content | after a history cut | evicted prefix | partial re-prefill | cuts | evictions (mean idle s) | computed per generated")
for b in bands:
    A = agg[b]
    if not A["n"]: continue
    n = A["n"]; c = A["comp"]
    print(f" {b[0]:>2}-{b[1]:<2} | {n:5.0f} | {c/n:5.0f} | {A['append']/n:5.0f} ({100*A['append']/c:2.0f}%) | {A['trim']/n:5.0f} ({100*A['trim']/c:2.0f}%) | {A['lost']/n:5.0f} ({100*A['lost']/c:2.0f}%) | {A['partial']/n:5.0f} ({100*A['partial']/c:2.0f}%) | {A['trims']:.0f} | {A['losts']:.0f} ({A['lost_gap']/max(1,A['losts']):.0f}) | {c/A['gen']:.2f}")
T = defaultdict(float)
for A in agg.values():
    for k, v in A.items(): T[k] += v
print("whole:", {k: f"{100*T[k]/T['comp']:.0f}%" for k in ("append", "trim", "lost", "partial", "first")}, f"computed {T['comp']/1e6:.0f}M, generated {T['gen']/1e6:.1f}M")
