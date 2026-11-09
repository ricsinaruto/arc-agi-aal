"""Level attempts from recorded arena runs: completion tokens spent on each level a game played
and whether it cleared (26 September; input to the hazard analysis of the scheduling audit).

One row per (run, game, level attempted). ``tokens`` counts the game's completion tokens (every
purpose) while it was on that level, ``cleared`` is False when the run ended first (censored).
"""

import argparse
import glob
import json
from pathlib import Path


def attempts(trial_dir):
    trial = Path(trial_dir)
    report = json.loads((trial / "arena.json").read_text())
    config = report.get("config") or {}
    meta = {"run": trial.parents[2].name, "model": config.get("model"), "effort": report.get("reasoning_effort"),
            "window": (report.get("window_choice") or {}).get("window"),
            "slots": (report.get("scheduler") or {}).get("active_limit"),
            "wall": report.get("wall_seconds"), "switch": (report.get("scheduler") or {}).get("switch_wall_seconds")}
    rows = []
    for game_dir in sorted((trial / "games").iterdir()):
        log = game_dir / "log.jsonl"
        if not log.is_file():
            continue
        level, total, tokens, calls, actions, t0, started = 0, None, 0, 0, 0, None, None
        levels_seen = []
        for line in log.open():
            rec = json.loads(line)
            kind = rec.get("kind")
            t0 = rec.get("time") if t0 is None else t0
            if kind == "observation":
                total = rec.get("total_levels", total)
                new = int(rec.get("levels", level))
                if rec.get("action") is not None:
                    actions += 1
                if new > level:
                    rows.append({**meta, "game": game_dir.name[:4], "level": level + 1, "total_levels": total,
                                 "tokens": tokens, "calls": calls, "actions": actions, "cleared": True,
                                 "wall_start": started, "wall_end": round(rec["time"] - t0, 1),
                                 "previous": [r["tokens"] for r in levels_seen]})
                    levels_seen.append(rows[-1])
                    level, tokens, calls, actions, started = new, 0, 0, 0, round(rec["time"] - t0, 1)
            elif kind == "model":
                usage = rec.get("usage") or {}
                tokens += int(usage.get("completion_tokens") or 0)
                calls += 1
                started = round(rec["time"] - t0, 1) if started is None else started
        if total is None or level < total:
            rows.append({**meta, "game": game_dir.name[:4], "level": level + 1, "total_levels": total,
                         "tokens": tokens, "calls": calls, "actions": actions, "cleared": False,
                         "wall_start": started, "wall_end": None, "previous": [r["tokens"] for r in levels_seen]})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--glob", nargs="+", default=["runs/modal-validation/arena-duck-*/arena/trials/duck-transfer",
                                                      "runs/kaggle-arena-duck-*/arena/trials/duck-transfer"])
    args = parser.parse_args()
    rows = []
    for pattern in args.glob:
        for trial in sorted(glob.glob(pattern)):
            if "-lost-" in trial or not Path(trial, "arena.json").is_file():
                continue
            try:
                rows += attempts(trial)
            except (OSError, ValueError, KeyError) as exc:
                print("skip", trial, exc)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(len(rows), "attempts from", len({r["run"] for r in rows}), "runs")


if __name__ == "__main__":
    main()
