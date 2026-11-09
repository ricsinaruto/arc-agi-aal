"""Level attempts and token clocks from the real c110 one-hour runs (28 September).

The c110 roster is 110 cyclic clones (``k000``-``k109``) of 20 public games; ``public-roster.json`` maps every
clone to its source game. For each run this writes:

- level-attempt rows in the ``level_attempts.py`` format, with ``game`` set to the source game (4 letters),
  ``clone`` to the clone id and ``roster`` to ``"c110"`` (input to the simulator calibration);
- a per-run summary: score, switch, completion tokens before/after the switch, aggregate tok/s in the opening
  and in phase 2, tokens per clone at the switch, first-clear times and tokens, levels per clone, and the
  number of distinct clones and source games that held a focus slot at the same time.

    PYTHONPATH=src uv run python scripts/arena_analysis/c110_runs.py --root ./runs/modal-validation \
        --attempts-out c110-attempts.jsonl --summary-out c110-summary.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from level_attempts import attempts  # noqa: E402

RUNS = {  # run directory -> (label, variant, serving)
    "arena-duck-60m-c110-base-v1-r2-modal-20260927": ("base-v1-r2", "fixed-2x", "mtp3-2"),
    "arena-duck-60m-c110-base-v2-modal-20260927": ("base-v2", "fixed-2x", "mtp3-2"),
    "arena-duck-60m-c110-gittins-c110-v1-modal-20260927": ("gittins-c110", "gittins-c110", "mtp3-2"),
    "arena-duck-60m-c110-mtp0-wide-v1-modal-20260927": ("mtp0-wide", "fixed-2x", "mtp0-6"),
    "arena-duck-60m-c110-gittins-wide-v1-modal-20260927": ("gittins-wide", "gittins-c110", "mtp0-6"),
    "arena-duck-60m-c110-worldmodel-v1-modal-20260927": ("worldmodel", "fixed-2x+worldmodel", "mtp3-2"),
    "arena-duck-60m-c110-level-probe-v1-modal-20260927": ("level-probe", "fixed-2x+probe", "mtp3-2"),
}


def trial_dir(root, run):
    return Path(root, run, run, "arena", "trials", "duck-transfer")


def token_events(trial):
    """(seconds since the run start, clone, completion tokens, busy seconds) per model reply, and
    (seconds, clone, levels) per clear. Busy seconds run from the request's admission (``model_request``) to the
    reply, the clock the fixed-2x budgets use."""
    first = json.loads(Path(trial, "admission.jsonl").open().readline())
    t0 = first["epoch"] - first["wall_seconds"]
    out, clears = [], []
    for game_dir in sorted((trial / "games").iterdir()):
        log = game_dir / "log.jsonl"
        if not log.is_file():
            continue
        level, req = 0, None
        for line in log.open():
            rec = json.loads(line)
            kind = rec.get("kind")
            if kind == "model_request":
                req = rec["time"]
            elif kind == "model":
                busy = rec["time"] - req if req is not None else 0.0
                req = None
                out.append((rec["time"] - t0, game_dir.name, int((rec.get("usage") or {}).get("completion_tokens") or 0),
                            busy))
            elif kind == "observation":
                new = int(rec.get("levels", level))
                if new > level:
                    clears.append((rec["time"] - t0, game_dir.name, new))
                    level = new
    out.sort()
    clears.sort()
    return out, clears


def focus_overlap(report, source):
    """Max number of focus slots held at once by clones of one source game (from the scheduler's events)."""
    sched = report.get("scheduler") or {}
    switch = sched.get("switch_wall_seconds") or 0
    active, best = set(), {}
    for ev in sched.get("events") or []:
        if ev.get("wall_seconds", 0) < switch:
            continue
        if ev.get("event") == "admit" or (ev.get("event") == "grant" and ev.get("reason") == "opening_switch"):
            active.add(ev["game"])
        elif ev.get("event") in ("park", "finish"):
            active.discard(ev["game"])
        counts = {}
        for g in active:
            counts[source[g]] = counts.get(source[g], 0) + 1
        for s, c in counts.items():
            best[s] = max(best.get(s, 0), c)
    return best


def summarise(root, run):
    label, variant, serving = RUNS[run]
    trial = trial_dir(root, run)
    report = json.loads((trial / "arena.json").read_text())
    roster = json.loads((trial / "public-roster.json").read_text())
    source = {r["game_id"]: r["source_game_id"][:4] for r in roster}
    sched = report.get("scheduler") or {}
    switch = float(sched.get("switch_wall_seconds") or 0.0)
    events, clears = token_events(trial)
    t = np.array([e[0] for e in events])
    tok = np.array([e[2] for e in events], dtype=float)
    wall = float(report.get("wall_seconds") or 3600)
    before = float(tok[t <= switch].sum())
    after = float(tok[(t > switch) & (t <= wall)].sum())
    busy = np.array([e[3] for e in events])
    per_clone_switch = {}
    for (ti, g, n, _) in events:
        if ti <= switch:
            per_clone_switch[g] = per_clone_switch.get(g, 0) + n
    at_switch = np.array([per_clone_switch.get(g, 0) for g in source])
    # Relative opening token rate per source game (mean clone tokens at the switch over the roster mean).
    share = {}
    for g, s in source.items():
        share.setdefault(s, []).append(per_clone_switch.get(g, 0) / max(at_switch.mean(), 1.0))
    share = {s: round(float(np.mean(v)), 3) for s, v in share.items()}
    busy_rate = {"opening": round(float(tok[t <= switch].sum() / max(busy[t <= switch].sum(), 1e-9)), 1),
                 "phase2": round(float(tok[t > switch].sum() / max(busy[t > switch].sum(), 1e-9)), 1)}
    # Cumulative tokens every 300 s.
    grid = list(range(300, int(wall) + 1, 300))
    cum = [int(tok[t <= x].sum()) for x in grid]
    levels = {}
    for g in source:
        levels[g] = 0
    for (_, g, lv) in clears:
        levels[g] = max(levels[g], lv)
    first_clears = [(round(ti, 1), g, source[g]) for (ti, g, lv) in clears if lv == 1]
    rows = attempts(trial)
    for r in rows:
        r["clone"] = r["game"]
        r["game"] = source[r["clone"]]
        r["roster"] = "c110"
    # Tokens of the first-level attempt at its clear, for clones that cleared level 1.
    t1 = {r["clone"]: r["tokens"] for r in rows if r["level"] == 1 and r["cleared"]}
    games = report.get("games") or {}
    return rows, {
        "run": run, "label": label, "variant": variant, "serving": serving,
        "score": report.get("mean_score"), "levels": sum(levels.values()), "switch": round(switch, 1),
        "switch_reason": sched.get("switch_reason"), "active_limit": sched.get("active_limit"),
        "tokens_total": int(tok[t <= wall].sum()), "tokens_opening": int(before), "tokens_phase2": int(after),
        "rate_total": round(float(tok[t <= wall].sum()) / wall, 1),
        "rate_opening": round(before / switch, 1) if switch else None,
        "rate_phase2": round(after / max(wall - switch, 1.0), 1),
        "cum_tokens_300s": dict(zip(grid, cum)),
        "tokens_per_clone_at_switch": {"median": float(np.median(at_switch)), "mean": float(at_switch.mean()),
                                       "p90": float(np.percentile(at_switch, 90))},
        "busy_tokens_per_second": busy_rate, "opening_share": share,
        "first_clears": first_clears[:12],
        "t1_tokens": {g: t1[g] for g in sorted(t1)},
        "levels_by_clone": {g: lv for g, lv in levels.items() if lv},
        "source": {g: source[g] for g in levels if levels[g]},
        "max_focus_clones_per_source": focus_overlap(report, source),
        "n_games_reported": len(games),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./runs/modal-validation")
    parser.add_argument("--attempts-out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    all_rows, summary = [], []
    for run in RUNS:
        if not trial_dir(args.root, run).is_dir():
            print("missing", run)
            continue
        rows, s = summarise(args.root, run)
        all_rows += rows
        summary.append(s)
        print(f"{s['label']:14s} score {s['score']:.2f} levels {s['levels']:3d} switch {s['switch']:6.0f}s "
              f"tokens {s['tokens_total'] / 1e3:6.0f}K opening {s['rate_opening']} tok/s phase2 {s['rate_phase2']} tok/s "
              f"median/clone@switch {s['tokens_per_clone_at_switch']['median']:.0f}")
    Path(args.attempts_out).write_text("\n".join(json.dumps(r) for r in all_rows) + "\n")
    Path(args.summary_out).write_text(json.dumps(summary, indent=1))
    print(len(all_rows), "attempt rows")


if __name__ == "__main__":
    main()
