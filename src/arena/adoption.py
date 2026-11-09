"""Noise floor and adoption rule for arena comparisons (paper-notes.md §2.1, item 1 of §3).

Across the fixed-2x family the number of fast games (ft09, lp85) at level 1 by the opening switch
explains two thirds of the score variance (docs/validation-20260925/noise-floor-20260925.md), so a
candidate is compared with its base *stratified* by that opening outcome, and unstratified only
with a margin of two pooled within-stratum standard deviations scaled by the sample sizes.

``run_metrics(folder)`` reads one trial folder (``arena.json`` plus, when present, the run's
``analysis.json`` for the 900 s checkpoint). ``compare(base, candidate)`` returns the verdict
(``adopt`` / ``reject`` / ``undecided``) with the numbers behind it; ``python -m arena.adoption
--base ... --candidate ...`` prints it.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

FAST_GAMES = ("ft09", "lp85")
SWITCH_TOLERANCE_SECONDS = 30.0
SD_MULTIPLE = 2.0


def find_report(path):
    """``arena.json`` of a trial folder, a run folder, or the file itself."""
    path = Path(path)
    if path.is_file():
        return path
    for candidate in (path / "arena.json", path / "duck-transfer" / "arena.json",
                      path / "arena" / "trials" / "duck-transfer" / "arena.json"):
        if candidate.is_file():
            return candidate
    found = sorted(path.glob("**/duck-transfer/arena.json"))
    if len(found) != 1:
        raise FileNotFoundError(f"No single arena.json under {path}")
    return found[0]


def score_at(analysis_path, seconds):
    if not analysis_path.is_file():
        return None
    data = json.loads(analysis_path.read_text())
    for entry in data if isinstance(data, list) else [data]:
        for checkpoint in (entry.get("score_curve") or {}).get("checkpoints", []):
            if abs(float(checkpoint["wall_seconds"]) - seconds) < 1:
                return float(checkpoint["score"])
    return None


def run_metrics(path, *, fast_games=FAST_GAMES):
    report_path = find_report(path)
    report = json.loads(report_path.read_text())
    games = report["games"]
    switch = (report.get("scheduler") or {}).get("switch_wall_seconds")
    level1 = {}
    for game in games:
        first = next((t["wall_seconds"] for t in game.get("time_to_level", []) if t["level"] == 1), None)
        level1[game["game_id"][:4]] = first
    fast = sum(1 for g in fast_games if level1.get(g) is not None and switch is not None
               and level1[g] <= switch + SWITCH_TOLERANCE_SECONDS)
    analysis = next((p for p in (report_path.parents[2] / "analysis.json", report_path.parents[3] / "analysis.json",
                                 report_path.parent / "analysis.json") if p.is_file()), Path("/nonexistent"))
    return {
        "path": str(report_path),
        "score": float(report["mean_score_equal_weight_25_games"]),
        "clears": int(report["levels"]),
        "games": len(games),
        "games_with_clears": sum(1 for g in games if g["levels"] > 0),
        "fast_games_at_switch": fast,
        "fast_games_present": sum(1 for g in fast_games if g in level1),
        "switch_wall_seconds": switch,
        "level1_times": sorted(round(t) for t in level1.values() if t is not None),
        "score_900": score_at(analysis, 900),
        "score_1800": score_at(analysis, 1800),  # one-hour runs (26 September): the 30-minute point
        "score_3600": score_at(analysis, 3600),
        "wall_seconds": report.get("wall_seconds"),
        "status": report.get("status"),
    }


def _mean(values):
    return statistics.mean(values) if values else None


def pooled_sd(groups):
    """Pooled within-group standard deviation over groups with at least two samples."""
    numerator, dof = 0.0, 0
    for values in groups:
        if len(values) >= 2:
            numerator += (len(values) - 1) * statistics.variance(values)
            dof += len(values) - 1
    return math.sqrt(numerator / dof) if dof > 0 else None


def compare(base, candidate, *, metric="score", sd_multiple=SD_MULTIPLE, floor_sd=None):
    """Verdict on ``candidate`` runs against ``base`` runs (lists of ``run_metrics`` dicts).

    Stratified: per fast-game count present on both sides, the mean difference; the stratified
    estimate is the sample-weighted mean of those differences. Unstratified: the difference of
    means. The margin is ``sd_multiple`` × pooled within-stratum sd × sqrt(1/n_base + 1/n_cand)
    (``floor_sd`` supplies the family's sd when the runs at hand cannot estimate one).
    """
    strata = sorted({r["fast_games_at_switch"] for r in base + candidate})
    by_stratum = {}
    for s in strata:
        b = [r[metric] for r in base if r["fast_games_at_switch"] == s and r[metric] is not None]
        c = [r[metric] for r in candidate if r["fast_games_at_switch"] == s and r[metric] is not None]
        by_stratum[s] = {"base": b, "candidate": c, "diff": (_mean(c) - _mean(b)) if b and c else None}
    sd = pooled_sd([v["base"] + [] for v in by_stratum.values()] + [v["candidate"] for v in by_stratum.values()])
    if sd is None:
        sd = floor_sd
    b_all = [r[metric] for r in base if r[metric] is not None]
    c_all = [r[metric] for r in candidate if r[metric] is not None]
    unstratified = _mean(c_all) - _mean(b_all) if b_all and c_all else None
    weights = {s: min(len(v["base"]), len(v["candidate"])) for s, v in by_stratum.items() if v["diff"] is not None}
    stratified = (sum(by_stratum[s]["diff"] * w for s, w in weights.items()) / sum(weights.values())) if weights else None
    margin = None
    if sd is not None and b_all and c_all:
        margin = sd_multiple * sd * math.sqrt(1 / len(b_all) + 1 / len(c_all))
    estimate = stratified if stratified is not None else unstratified
    if estimate is None or margin is None:
        verdict = "undecided: not enough runs"
    elif estimate > margin:
        verdict = "adopt"
    elif estimate < -margin:
        verdict = "reject"
    else:
        needed = None
        if sd and estimate:
            needed = math.ceil(2 * (sd_multiple * sd / abs(estimate)) ** 2)  # runs per side for the margin to fit
        verdict = f"undecided: |{estimate:.2f}| within the {margin:.2f} margin" + (f"; about {needed} runs per side would resolve it" if needed else "")
    co = {}
    for key in ("clears", "games_with_clears", "score_900"):
        b = [r[key] for r in base if r.get(key) is not None]
        c = [r[key] for r in candidate if r.get(key) is not None]
        co[key] = {"base": _mean(b), "candidate": _mean(c)}
    return {"metric": metric, "verdict": verdict, "estimate": estimate, "stratified": stratified, "unstratified": unstratified,
            "margin": margin, "pooled_sd": sd, "sd_multiple": sd_multiple, "strata": {str(k): v for k, v in by_stratum.items()},
            "n_base": len(b_all), "n_candidate": len(c_all), "co_metrics": co}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", nargs="+", required=True, help="Trial or run folders of the base")
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--metric", default="score", choices=["score", "score_900", "clears", "games_with_clears"])
    parser.add_argument("--fast-games", default=",".join(FAST_GAMES))
    parser.add_argument("--floor-sd", type=float, default=1.40, help="Family sd when the runs cannot estimate one (noise-floor record: 1.40)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    fast = tuple(g for g in args.fast_games.split(",") if g)
    base = [run_metrics(p, fast_games=fast) for p in args.base]
    candidate = [run_metrics(p, fast_games=fast) for p in args.candidate]
    result = compare(base, candidate, metric=args.metric, floor_sd=args.floor_sd)
    if args.json:
        print(json.dumps({"base": base, "candidate": candidate, **result}, indent=2))
        return result
    for label, runs in (("base", base), ("candidate", candidate)):
        for r in runs:
            print(f"{label:9} {r['score']:6.2f}% {r['clears']:3} clears {r['games_with_clears']:2} games  fast@switch {r['fast_games_at_switch']}  "
                  f"900s {r['score_900'] if r['score_900'] is None else round(r['score_900'], 2)}  {Path(r['path']).parents[2].name}")
    print(f"verdict: {result['verdict']}  (estimate {result['estimate']}, margin {result['margin']}, pooled sd {result['pooled_sd']}, "
          f"stratified {result['stratified']}, unstratified {result['unstratified']})")
    return result


if __name__ == "__main__":
    main()
