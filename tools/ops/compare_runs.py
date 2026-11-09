"""Compare long runs at equal play time (tok/s) and at equal generated tokens (score, levels). Edit RUNS for the runs to compare:
(label, run id, speed factor to Nebius hardware, refresh from the bucket). Reads the live progress.jsonl of each run from the Nebius bucket
into the scratch folder; finished or Modal runs must already be there. Usage: compare_runs.py ; also feeds the live charts (banner.py)."""
import json, sys, subprocess, pathlib, bisect
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import paths
S = paths.SCRATCH / "mon"
RUNS = [("v8.2 refit weights 128K", "b2-v82sub-refit25-w128-c110-24h-v1-nebius-20261010", 1.0, True),
        ("v8.2tc 128K", "b2-v82tc-w128-c110-24h-v1-nebius-20261009", 1.0, False), ("v8.2tc 64K", "b2-v82tc-w64-c110-24h-v1-nebius-20261009", 1.0, False),
        ("v8.2tc 196K", "b2-v82tc-w196-c110-24h-v1-nebius-20261009", 1.0, False),
        ("v8.2 128K (old)", "b2-v82-w128-c110-24h-v1-nebius-20261009", 1.0, False), ("v8.2 64K (old)", "b2-v82-w64-c110-24h-v1-nebius-20261009", 1.0, False),
        ("v8.2 196K (old)", "b2-v82-w196-c110-24h-v1b-nebius-20261009", 1.0, False), ("v8.2 128K tick only", "b2-v82t-w128-c110-24h-v1-nebius-20261009", 1.0, False),
        ("v7.1p excl5, Nebius 2 h", "b2-v71p-excl5-c110-2h-rep2-nebius-20261007", 1.0, False), ("v7.1p 24 h, Modal", "b2-v71p-24h-v1-modal-20261007", 1 / 1.088, False)]
# ARC-AGI-2 package runs on four L4s (Modal: mon.py pulls each file while the run is in the queue's live list). The full run is charted
# alone, with the same 240 entries on the RTX PRO 6000 as reference (million tokens, score %); before it starts, the 43-task rehearsals.
ARC2 = [("four L4s, 240 entries, 11.5 h", "arc2-l4-full-v1-modal-20261010", 1.0, False),
        ("rehearsal, Python tool", "arc2-l4-package-v4-modal-20261010", 1.0, False), ("rehearsal, plain loop", "arc2-l4-package-v5-modal-20261010", 1.0, False)]
ARC2_REF = {"RTX PRO 6000, Python tool": [(5.62, 36.8), (7.78, 44.9), (10.05, 50.6)], "RTX PRO 6000, plain loop": [(5.62, 26.0), (7.78, 37.1), (11.66, 41.9), (12.35, 44.6)]}
def stitch(raw):
    """Progress rows made cumulative across server restarts and containers (a preempted Modal job resumes in a new
    container whose elapsed time starts at 0; its token count starts at 0 until the play resumes, and then either
    carries on from the saved state, as the ARC-AGI-2 package does, or starts again): elapsed carries the earlier
    segments' total, tokens are offset only when a counter really started again, and the score never falls below
    the last one read. Each row: (t, tokens, score, score_t, levels, kv, play, containers, restarts)."""
    out, t_off, g_off, last_t, last_g, best, containers, restarts, carry, fresh = [], 0.0, 0, None, 0, None, 1, 0, 0, False
    for r in raw:
        el = r.get("elapsed_seconds")
        if el is None: continue
        l = r.get("live") or {}; e = l.get("engine") or {}; sc = l.get("score") or {}
        g = e.get("generated_tokens") or 0
        if last_t is not None and el < last_t:  # a new container: its first counted tokens tell whether the counter carried on
            t_off += last_t; carry = g_off + last_g; containers += 1; fresh = True; last_g = 0
        if fresh and g:
            fresh = False
            if g + g_off < carry: g_off = carry  # the counter started again
        elif not fresh and g < last_g:  # an engine restart inside the container
            g_off += last_g; restarts += 1
        last_t, last_g = el, g
        s = sc.get("score")
        if s is not None: best = (s, sc.get("at_elapsed_seconds"), sc.get("levels"), t_off) if best is None or s >= best[0] else best
        out.append((t_off + el, carry if fresh else g_off + g, best[0] if best else None,
                    (best[3] + (best[1] if best[1] is not None else 0)) if best else None, best[2] if best else None,
                    e.get("kv_cache_fraction"), r.get("play_seconds"), containers, max(restarts, r.get("restarts") or 0)))
    return out
def load(run, live):
    f = S / f"{run}.jsonl"
    if live: subprocess.run(["rclone", "copyto", f"nebius-arc:arc-runs-bucket/runs/validation/{run}/arena/live/progress.jsonl", str(f)], capture_output=True, timeout=300)
    raw = []
    for x in open(f):
        try: raw.append(json.loads(x))
        except Exception: continue
    return [r[:6] for r in stitch(raw) if r[1]]
def tok_at(rows, t):
    ts = [r[0] for r in rows]; i = bisect.bisect_left(ts, t)
    if i == 0: return rows[0][1] * t / max(rows[0][0], 1)
    if i >= len(rows): return rows[-1][1]
    (t0, g0), (t1, g1) = rows[i - 1][:2], rows[i][:2]
    return g0 + (g1 - g0) * (t - t0) / max(t1 - t0, 1e-9)
def tables():
    data = {n: (load(r, live), k) for n, r, k, live in RUNS}
    edges = [0, 15, 30, 45, 60, 90, 120, 180, 240, 300, 360, 480, 600, 720]
    print("tok/s by play time (Modal run scaled by 1/1.088 to Nebius speed)")
    print("| minutes | " + " | ".join(data) + " |"); print("|---|" + "---:|" * len(data))
    for a, b in zip(edges, edges[1:]):
        cells = []
        for n, (rows, k) in data.items():
            cells.append(f"{k * (tok_at(rows, b * 60) - tok_at(rows, a * 60)) / ((b - a) * 60):,.0f}" if rows and rows[-1][0] >= b * 60 else "")
        if any(cells[:4]): print(f"| {a}-{b} | " + " | ".join(cells) + " |")
    print("\nscore % (levels cleared) at equal generated tokens")
    print("| M tokens | " + " | ".join(data) + " |"); print("|---|" + "---:|" * len(data))
    for m in [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 42, 48, 54, 60, 70, 80, 90, 100, 110, 120]:
        cells = []
        for n, (rows, k) in data.items():
            best = None
            for r in rows:  # last scorecard read whose own time is at or before the token mark
                if r[2] is not None and r[3] is not None and tok_at(rows, r[3]) <= m * 1e6: best = r
            last = max((tok_at(rows, r[3]) for r in rows if r[3] is not None), default=0)
            cells.append(f"{best[2]:.2f} ({best[4]})" if best and last >= m * 1e6 else "")
        if any(cells[:4]): print(f"| {m} | " + " | ".join(cells) + " |")

def series(refresh=True, runs=None):
    """Points for the live charts: score against million generated tokens, and tok/s in 15-minute bins against hours of play."""
    out = {}
    for n, r, k, live in RUNS if runs is None else runs:
        rows = load(r, live and refresh)
        sc = sorted({(round(tok_at(rows, x[3]) / 1e6, 2), round(x[2], 2)) for x in rows if x[2] is not None and x[3] is not None})
        end = rows[-1][0] if rows else 0
        sp = [(round((b + 7.5) / 60, 2), round(k * (tok_at(rows, (b + 15) * 60) - tok_at(rows, b * 60)) / 900)) for b in range(0, int(end // 60) - 14, 15)]
        out[n] = (sc, sp)
    return out
if __name__ == "__main__":
    tables()
