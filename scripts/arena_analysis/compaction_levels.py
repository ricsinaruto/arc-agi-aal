"""Does compaction raise the cost of deep levels? (28 September, CPU only)

Level attempts from the recorded one-hour runs (the 23-game baseline runs of ``optimal_schedule.BASELINE_RUNS``,
128K windows, and the c110 runs: 128K ``base-v1-r2``, ``base-v2``, ``gittins-c110``; 64K ``mtp0-wide``,
``kernels-v2``, ``gittins-wide``), with every compaction placed on the attempt's token clock. An attempt's
tokens are the game's completion tokens (every purpose) while it was on that level, as in ``level_attempts.py``.

Compaction state of a level >= 2 attempt:
- ``none``: no compaction in the game before the attempt ended;
- ``earlier``: one or more compactions before the attempt started, none during it (``recent``: at least one
  since the start of the previous level);
- ``during``: a compaction happened during the attempt.

Attempt-level categories confound state with length (a long attempt is more likely to contain a compaction),
so the main estimate splits every attempt's token exposure at its compactions: the segment before the first
in-attempt compaction carries the attempt's entry state (``none`` or ``earlier``), the rest carries ``after``;
the clear (if any) belongs to the segment it happened in. A piecewise-exponential Poisson model then fits

    clears ~ band(tokens into the attempt) + level + game + state [+ window]   (offset log exposure)

(bands of ``schedule_sim.EDGES``; the within-level hazard rises with tokens spent, so the band terms matter).
Uncertainty: a cluster bootstrap over source games. A Mantel-Haenszel rate ratio stratified by (game, level)
is the non-parametric check. The same machinery compares 64K with 128K windows on levels >= 2.

    python scripts/arena_analysis/compaction_levels.py --out docs/validation-20260927/compaction-levels.json \
        [--rows-out rows.jsonl] [--boot 300]
"""

import argparse
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimal_schedule import BASELINE_RUNS  # noqa: E402
from schedule_sim import EDGES  # noqa: E402

ROOTS = ("./runs/modal-validation", "./runs")
C110 = {  # label -> (run, window)
    "c110-base-v1-r2": "arena-duck-60m-c110-base-v1-r2-modal-20260927",
    "c110-base-v2": "arena-duck-60m-c110-base-v2-modal-20260927",
    "c110-gittins-c110": "arena-duck-60m-c110-gittins-c110-v1-modal-20260927",
    "c110-mtp0-wide": "arena-duck-60m-c110-mtp0-wide-v1-modal-20260927",
    "c110-kernels-v2": "arena-duck-60m-c110-kernels-v2-modal-20260928",
    "c110-gittins-wide": "arena-duck-60m-c110-gittins-wide-v1-modal-20260927",
}
NB = len(EDGES) - 1


def find_trial(run):
    for root in ROOTS:
        for pattern in (f"{root}/{run}/arena/trials/duck-transfer", f"{root}/{run}/{run}/arena/trials/duck-transfer"):
            hits = glob.glob(pattern)
            if hits and Path(hits[0], "arena.json").is_file():
                return Path(hits[0])
    return None


def game_rows(log, meta, game, clone):
    """Level attempts of one game with the compactions on each attempt's token clock."""
    rows = []
    level, tokens, comps_in, comps_total, start_total, prev_start_total = 0, 0, [], 0, 0, 0
    total, t0, started, first_time = None, None, None, None
    for line in log.open():
        rec = json.loads(line)
        kind = rec.get("kind")
        t0 = rec.get("time") if t0 is None else t0
        if kind == "observation":
            total = rec.get("total_levels", total)
            new = int(rec.get("levels", level))
            if new > level:
                rows.append({**meta, "game": game, "clone": clone, "level": level + 1, "total_levels": total,
                             "tokens": tokens, "cleared": True, "comp_before": start_total,
                             "comp_since_prev_level": start_total - prev_start_total, "comp_in": comps_in,
                             "wall_start": started, "wall_end": round(rec["time"] - t0, 1)})
                prev_start_total, start_total = start_total, comps_total
                level, tokens, comps_in, started = new, 0, [], round(rec["time"] - t0, 1)
        elif kind == "model":
            tokens += int((rec.get("usage") or {}).get("completion_tokens") or 0)
            started = round(rec["time"] - t0, 1) if started is None else started
        elif kind == "compaction":
            comps_total += 1
            comps_in.append(tokens)
    if total is None or level < total:
        rows.append({**meta, "game": game, "clone": clone, "level": level + 1, "total_levels": total,
                     "tokens": tokens, "cleared": False, "comp_before": start_total,
                     "comp_since_prev_level": start_total - prev_start_total, "comp_in": comps_in,
                     "wall_start": started, "wall_end": None})
    return rows, comps_total


def load(runs):
    rows, run_stats = [], []
    for label, (run, family) in runs.items():
        trial = find_trial(run)
        if trial is None:
            print("missing", run)
            continue
        report = json.loads((trial / "arena.json").read_text())
        window = (report.get("window_choice") or {}).get("window")
        source = {}
        roster = trial / "public-roster.json"
        if roster.is_file():
            source = {r["game_id"]: r["source_game_id"][:4] for r in json.loads(roster.read_text())}
        meta = {"run": label, "family": family, "window": window}
        comps, games, tokens, active = 0, 0, 0, 0
        for game_dir in sorted((trial / "games").iterdir()):
            log = game_dir / "log.jsonl"
            if not log.is_file():
                continue
            game = source.get(game_dir.name, game_dir.name[:4])
            r, c = game_rows(log, meta, game, game_dir.name)
            rows += r
            comps += c
            games += 1
            spent = sum(x["tokens"] for x in r)
            tokens += spent
            active += spent > 0
        run_stats.append({"run": label, "family": family, "window": window, "games": games, "active_games": active,
                          "wall_hours": (report.get("wall_seconds") or 3600) / 3600, "compactions": comps,
                          "completion_tokens": tokens,
                          "per_game_hour": comps / (games * (report.get("wall_seconds") or 3600) / 3600),
                          "per_1m_tokens": comps / tokens * 1e6 if tokens else None})
    return rows, run_stats


def state_of(r):
    if r["comp_in"]:
        return "during"
    if r["comp_before"] > 0:
        return "earlier"
    return "none"


def segments(r):
    """[(state, band, exposure, event)] for one attempt; the entry state until the first in-attempt compaction."""
    entry = "earlier" if r["comp_before"] > 0 else "none"
    cut = r["comp_in"][0] if r["comp_in"] else None
    out = []
    lo_all, hi_all = EDGES[:-1], EDGES[1:]
    pieces = [(entry, 0, cut if cut is not None else r["tokens"])]
    if cut is not None:
        pieces.append(("after", cut, r["tokens"]))
    event_band = int(np.searchsorted(EDGES, r["tokens"], side="right") - 1)
    for state, a, b in pieces:
        for k in range(NB):
            x = max(0.0, min(b, hi_all[k]) - max(a, lo_all[k]))
            if x > 0:
                ev = int(r["cleared"] and k == event_band and a < r["tokens"] <= b)
                out.append((state, k, x, ev))
    return out


def usable(r):
    return not (r["cleared"] and r["tokens"] < 500)


def poisson_fit(X, y, offset, ridge=1e-6, iters=100):
    beta = np.zeros(X.shape[1])
    beta[0] = math.log(max(y.sum(), 0.5) / np.exp(offset).sum())
    for _ in range(iters):
        eta = X @ beta + offset
        mu = np.exp(eta)
        grad = X.T @ (y - mu) - ridge * beta
        hess = (X * mu[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta


def design(segs, games, levels, terms):
    """Columns: intercept, bands 1.., levels (vs the first), games (vs the first), the named 0/1 terms."""
    cols = ["const"] + [f"band{k}" for k in range(1, NB)] + [f"L{l}" for l in levels[1:]] + [f"g:{g}" for g in games[1:]]
    cols += list(terms)
    index = {c: i for i, c in enumerate(cols)}
    X = np.zeros((len(segs), len(cols)))
    y = np.zeros(len(segs))
    off = np.zeros(len(segs))
    for i, s in enumerate(segs):
        X[i, 0] = 1
        if s["band"] > 0:
            X[i, index[f"band{s['band']}"]] = 1
        if f"L{s['level']}" in index:
            X[i, index[f"L{s['level']}"]] = 1
        if f"g:{s['game']}" in index:
            X[i, index[f"g:{s['game']}"]] = 1
        for t in terms:
            X[i, index[t]] = s[t]
        y[i] = s["event"]
        off[i] = math.log(s["exposure"])
    keep = X.any(axis=0)
    keep[0] = True
    return X[:, keep], y, off, [c for c, k in zip(cols, keep) if k]


def seg_table(rows, state_terms=True, window_term=False, level_cap=5):
    segs = []
    for r in rows:
        for state, band, x, ev in segments(r):
            s = {"game": r["game"], "level": min(r["level"], level_cap), "band": band, "exposure": x, "event": ev}
            if state_terms:
                s["earlier"] = int(state == "earlier")
                s["after"] = int(state == "after")
            if window_term:
                s["w64"] = int(r["window"] == 65536)
            segs.append(s)
    return segs


def fit_rr(rows, terms, boot, rng, **kw):
    segs = seg_table(rows, **kw)
    games = sorted({s["game"] for s in segs})
    levels = sorted({s["level"] for s in segs})
    X, y, off, cols = design(segs, games, levels, terms)
    beta = poisson_fit(X, y, off)
    est = {t: float(math.exp(beta[cols.index(t)])) for t in terms if t in cols}
    by_game = defaultdict(list)
    for s in segs:
        by_game[s["game"]].append(s)
    draws = defaultdict(list)
    for _ in range(boot):
        pick = rng.choice(games, size=len(games), replace=True)
        bs = []
        for n, g in enumerate(pick):
            bs += [{**s, "game": f"{g}#{n}"} for s in by_game[g]]
        bgames = sorted({s["game"] for s in bs})
        Xb, yb, ob, cb = design(bs, bgames, levels, terms)
        try:
            bb = poisson_fit(Xb, yb, ob)
        except np.linalg.LinAlgError:
            continue
        for t in terms:
            if t in cb:
                draws[t].append(float(bb[cb.index(t)]))
    out = {}
    for t, v in est.items():
        d = np.array(draws[t])
        d = d[np.isfinite(d) & (np.abs(d) < 20)]
        lo, hi = (np.exp(np.percentile(d, [2.5, 97.5])) if len(d) > 10 else (None, None))
        out[t] = {"rr": round(v, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)] if lo is not None else None,
                  "boot": len(d)}
    events = Counter()
    expo = Counter()
    for s in segs:
        for t in terms:
            events[(t, s[t])] += s["event"]
            expo[(t, s[t])] += s["exposure"]
    out["_counts"] = {f"{t}={v}": {"clears": events[(t, v)], "tokens": round(expo[(t, v)])} for (t, v) in events}
    return out


def mh_strata(rows, classify):
    """{stratum: [d1, e1, d0, e0]} over (game, level, token band); ``classify(state, r)`` -> 1, 0 or None (skip)."""
    strata = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for r in rows:
        for state, band, x, ev in segments(r):
            c = classify(state, r)
            if c is None:
                continue
            s = strata[(r["game"], r["level"], band)]
            if c:
                s[0] += ev
                s[1] += x
            else:
                s[2] += ev
                s[3] += x
    return strata


def mh_ratio(strata):
    num = den = 0.0
    n = 0
    for d1, e1, d0, e0 in strata.values():
        e = e1 + e0
        if e1 > 0 and e0 > 0:
            n += 1
            num += d1 * e0 / e
            den += d0 * e1 / e
    return (num / den if den > 0 else None), n


def mantel_haenszel(rows, classify, boot, rng):
    """Rate ratio (class 1 vs 0) stratified by (game, level, band), with a game-cluster bootstrap CI."""
    rr, n = mh_ratio(mh_strata(rows, classify))
    by_game = defaultdict(list)
    for r in rows:
        by_game[r["game"]].append(r)
    games = sorted(by_game)
    draws = []
    for _ in range(boot):
        pick = rng.choice(games, size=len(games), replace=True)
        bs = [{**r, "game": f"{g}#{k}"} for k, g in enumerate(pick) for r in by_game[g]]
        v, _ = mh_ratio(mh_strata(bs, classify))
        if v is not None and v > 0:
            draws.append(v)
    ci = [round(float(x), 3) for x in np.percentile(draws, [2.5, 97.5])] if len(draws) > 10 else None
    return {"rr": round(rr, 3) if rr is not None else None, "ci95": ci, "strata": n}


def describe(rows, key):
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    for k, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        clears = sum(r["cleared"] for r in rs)
        tok = sum(r["tokens"] for r in rs)
        ttc = sorted(r["tokens"] for r in rs if r["cleared"])
        out[str(k)] = {"attempts": len(rs), "clears": clears, "tokens": tok,
                       "clears_per_10k": round(clears / tok * 1e4, 4) if tok else None,
                       "tokens_to_clear_median": ttc[len(ttc) // 2] if ttc else None,
                       "tokens_to_clear_mean": round(sum(ttc) / len(ttc)) if ttc else None,
                       "mean_level": round(sum(r["level"] for r in rs) / len(rs), 2)}
    return out


def log_ttc_ols(rows, term):
    """OLS of log tokens-to-clear (cleared attempts) on game + level dummies + a 0/1 term; returns exp(coef)."""
    rs = [r for r in rows if r["cleared"]]
    games = sorted({r["game"] for r in rs})
    levels = sorted({min(r["level"], 5) for r in rs})
    cols = ["const"] + [f"g:{g}" for g in games[1:]] + [f"L{l}" for l in levels[1:]] + [term]
    X = np.zeros((len(rs), len(cols)))
    y = np.array([math.log(r["tokens"]) for r in rs])
    for i, r in enumerate(rs):
        X[i, 0] = 1
        if f"g:{r['game']}" in cols:
            X[i, cols.index(f"g:{r['game']}")] = 1
        if f"L{min(r['level'], 5)}" in cols:
            X[i, cols.index(f"L{min(r['level'], 5)}")] = 1
        X[i, -1] = r[term]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, len(y) - X.shape[1])
    cov = np.linalg.pinv(X.T @ X) * (resid @ resid / dof)
    se = math.sqrt(cov[-1, -1])
    return {"ratio": round(math.exp(beta[-1]), 3), "ci95": [round(math.exp(beta[-1] - 1.96 * se), 3),
                                                           round(math.exp(beta[-1] + 1.96 * se), 3)],
            "n": len(rs), "n_term": int(sum(r[term] for r in rs))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows-out")
    parser.add_argument("--boot", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    runs = {run: (run, "r23-128k") for run in BASELINE_RUNS}
    for label, run in C110.items():
        runs[label] = (run, "c110")
    rows, run_stats = load(runs)
    rows = [r for r in rows if usable(r)]
    if args.rows_out:
        Path(args.rows_out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    deep = [r for r in rows if r["level"] >= 2]
    for r in rows:
        r["state"] = state_of(r)
        r["w64"] = int(r["window"] == 65536)
        r["compacted"] = int(r["state"] != "none")
    res = {"runs": run_stats, "n_attempts": len(rows), "n_deep": len(deep)}
    # Compaction frequency by window.
    freq = defaultdict(lambda: Counter())
    for s in run_stats:
        f = freq[s["window"]]
        f["runs"] += 1
        f["compactions"] += s["compactions"]
        f["game_hours"] += s["games"] * s["wall_hours"]
        f["completion_tokens"] += s["completion_tokens"]
    fam = defaultdict(lambda: Counter())
    for s_ in run_stats:
        f = fam[f"{s_['family']}-{s_['window']}"]
        f["runs"] += 1
        f["compactions"] += s_["compactions"]
        f["game_hours"] += s_["games"] * s_["wall_hours"]
        f["completion_tokens"] += s_["completion_tokens"]
    res["compaction_frequency_by_family"] = {
        k: {"runs": f["runs"], "compactions": f["compactions"],
            "per_game_hour": round(f["compactions"] / f["game_hours"], 4),
            "per_1m_completion_tokens": round(f["compactions"] / f["completion_tokens"] * 1e6, 2)}
        for k, f in fam.items()}
    res["compaction_frequency"] = {str(w): {"runs": f["runs"], "compactions": f["compactions"],
                                            "per_game_hour": round(f["compactions"] / f["game_hours"], 4),
                                            "per_1m_completion_tokens": round(f["compactions"] / f["completion_tokens"] * 1e6, 2)}
                                   for w, f in freq.items()}
    # Descriptive tables.
    res["deep_by_state"] = describe(deep, lambda r: r["state"])
    res["deep_by_state_window"] = describe(deep, lambda r: (r["window"], r["state"]))
    res["deep_by_state_recent"] = describe(
        [r for r in deep if r["state"] == "earlier"], lambda r: "recent" if r["comp_since_prev_level"] else "older")
    res["deep_by_window"] = describe(deep, lambda r: r["window"])
    res["deep_by_window_c110"] = describe([r for r in deep if r["family"] == "c110"], lambda r: r["window"])
    res["level1_by_window_c110"] = describe([r for r in rows if r["family"] == "c110" and r["level"] == 1],
                                            lambda r: r["window"])
    # Exposure-split hazard models (levels >= 2).
    res["poisson_state_all"] = fit_rr(deep, ("earlier", "after"), args.boot, rng)
    res["poisson_state_64k"] = fit_rr([r for r in deep if r["window"] == 65536], ("earlier", "after"), args.boot, rng)
    res["poisson_state_128k"] = fit_rr([r for r in deep if r["window"] != 65536], ("earlier", "after"), args.boot, rng)
    res["mh_state"] = {  # stratified by (game, level, token band into the attempt)
        "earlier_vs_none": mantel_haenszel(deep, lambda st, r: None if st == "after" else int(st == "earlier"),
                                           args.boot, rng),
        "after_vs_not_after": mantel_haenszel(deep, lambda st, r: int(st == "after"), args.boot, rng),
        "compacted_vs_none": mantel_haenszel(deep, lambda st, r: int(st != "none"), args.boot, rng),
    }
    res["poisson_state_window_all"] = fit_rr(deep, ("earlier", "after", "w64"), args.boot, rng, window_term=True)
    c110_deep = [r for r in deep if r["family"] == "c110"]
    res["poisson_window_c110"] = fit_rr(c110_deep, ("w64",), args.boot, rng, state_terms=False, window_term=True)
    res["poisson_window_all"] = fit_rr(deep, ("w64",), args.boot, rng, state_terms=False, window_term=True)
    res["mh_window_c110"] = mantel_haenszel(c110_deep, lambda st, r: int(r["window"] == 65536), args.boot, rng)
    res["mh_window_all"] = mantel_haenszel(deep, lambda st, r: int(r["window"] == 65536), args.boot, rng)
    res["ttc_earlier_vs_none"] = log_ttc_ols([r for r in deep if r["state"] != "during"], "compacted")
    res["ttc_during_vs_rest"] = log_ttc_ols([{**r, "during": int(r["state"] == "during")} for r in deep], "during")
    res["ttc_window_c110"] = log_ttc_ols(c110_deep, "w64")
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
