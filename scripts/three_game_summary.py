"""Tabulate repeated three-game runs: Qwen A/B/C comparison outputs and Duck control outputs.

Each positional argument is a run directory: a Qwen comparison output (holding
``trials/<A|B|C>/report.json``) or a Duck controls output (holding ``study.json`` with
per-trial summaries). Prints one Markdown row per baseline and run, then the per-baseline
spread, so run-to-run variability is visible at a glance. No score is claimed beyond the
listed games and runs.
"""

import argparse
import json
import statistics
from pathlib import Path


def qwen_rows(run):
    rows = []
    for trial in sorted(p for p in (run / "trials").iterdir() if p.is_dir()):
        report = json.loads((trial / "report.json").read_text())
        rows.append(
            {
                "baseline": trial.name,
                "run": run.name,
                "status": report["status"],
                "mean_score_percent": report["mean_score_percent"],
                "levels": report["total_cleared_levels"],
                "actions": report["total_actions"],
                "games": {
                    g["game"]: (g["score"], g["levels_completed"], g["actions"])
                    for g in report["games"]
                },
                "tokens_per_second": (report.get("backend_throughput") or {}).get(
                    "accepted_generation_tokens_per_second"
                ),
            }
        )
    return rows


def duck_rows(run):
    study = json.loads((run / "study.json").read_text())
    rows = []
    for trial in study["trials"]:
        summary = trial.get("summary") or {}
        kind = trial["trial"]["kind"]
        if kind == "corrected":
            games = {
                g["game_id"]: (g["score"], g["levels"], g["actions"])
                for g in summary.get("games", [])
            }
            mean = summary.get("mean_score")
            levels, actions = summary.get("levels"), summary.get("actions")
            tps = (summary.get("engine") or {}).get("accepted_generation_tokens_per_second")
        else:
            games = {
                r["game_id"]: (r["final_score"], r["levels_completed"], r["actions"])
                for r in summary.get("rows", [])
            }
            mean = summary.get("mean_final_score")
            levels = sum(v[1] for v in games.values()) if games else None
            actions = summary.get("total_actions")
            tps = None
        rows.append(
            {
                "baseline": trial["trial"]["name"],
                "run": run.name,
                "status": trial["status"],
                "mean_score_percent": mean,
                "levels": levels,
                "actions": actions,
                "games": games,
                "tokens_per_second": tps,
            }
        )
    return rows


def collect(runs):
    rows = []
    for run in runs:
        run = Path(run)
        if (run / "trials").is_dir():
            rows.extend(qwen_rows(run))
        elif (run / "study.json").is_file():
            rows.extend(duck_rows(run))
        else:
            raise ValueError(f"Not a comparison or controls output: {run}")
    return rows


def fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def render(rows):
    games = sorted({g for r in rows for g in r["games"]})
    lines = [
        "| baseline | run | status | mean % | levels | actions | tok/s | "
        + " | ".join(g.split("-")[0] + " score/levels/actions" for g in games)
        + " |",
        "|" + "---|" * (7 + len(games)),
    ]
    for r in sorted(rows, key=lambda r: (r["baseline"], r["run"])):
        cells = [
            r["baseline"],
            r["run"],
            r["status"],
            fmt(r["mean_score_percent"]),
            str(r["levels"]),
            str(r["actions"]),
            fmt(r["tokens_per_second"], 0),
        ]
        for g in games:
            if g in r["games"]:
                score, levels, actions = r["games"][g]
                cells.append(f"{fmt(score)} / {levels} / {actions}")
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "| baseline | runs | mean of means % | min % | max % | levels (each run) |",
        "|---|---|---|---|---|---|",
    ]
    by = {}
    for r in rows:
        by.setdefault(r["baseline"], []).append(r)
    for baseline, items in sorted(by.items()):
        means = [r["mean_score_percent"] for r in items if r["mean_score_percent"] is not None]
        lines.append(
            f"| {baseline} | {len(items)} | {fmt(statistics.fmean(means)) if means else '-'} | "
            f"{fmt(min(means)) if means else '-'} | {fmt(max(means)) if means else '-'} | "
            f"{', '.join(str(r['levels']) for r in items)} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument(
        "--json", action="store_true", help="Print rows as JSON instead of Markdown"
    )
    args = parser.parse_args()
    rows = collect(args.runs)
    print(json.dumps(rows, indent=2) if args.json else render(rows))


if __name__ == "__main__":
    main()
