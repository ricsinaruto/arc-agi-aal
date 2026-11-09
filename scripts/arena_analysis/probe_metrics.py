"""Metrics of short probes (26 September): level clears per 10K completion tokens on the probe games.

Usage: python scripts/arena_analysis/probe_metrics.py RUN_DIR [RUN_DIR ...]
Each RUN_DIR is a run folder (runs/modal-validation/<run>/ or runs/kaggle-<kernel>/) or a trial folder.
Rates are clears / tokens with a Poisson interval (clears +- 1.96 sqrt(clears)); probes compare arms
on the same games, so the rate difference between arms is the effect.
"""

import glob
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from level_attempts import attempts  # noqa: E402


def trial_dir(path):
    path = Path(path)
    if (path / "arena.json").is_file():
        return path
    found = sorted(glob.glob(str(path / "**/trials/duck-transfer"), recursive=True))
    if not found:
        raise FileNotFoundError(f"No duck-transfer trial under {path}")
    return Path(found[0])


def metrics(path):
    trial = trial_dir(path)
    rows = attempts(trial)
    report = json.loads((trial / "arena.json").read_text())
    out = {"run": rows[0]["run"] if rows else str(path), "score": report.get("mean_score_equal_weight_25_games"),
           "status": report.get("status")}
    for label, keep in (("all", lambda r: True), ("first", lambda r: r["level"] == 1),
                        ("deep", lambda r: r["level"] >= 2)):
        sub = [r for r in rows if keep(r)]
        clears = sum(r["cleared"] for r in sub)
        tokens = sum(r["tokens"] for r in sub)
        rate = clears / tokens * 1e4 if tokens else 0.0
        half = 1.96 * math.sqrt(clears) / tokens * 1e4 if tokens else 0.0
        out[label] = {"clears": clears, "tokens": tokens, "per_10k": round(rate, 3), "ci95": round(half, 3)}
    games = {}
    for r in rows:
        g = games.setdefault(r["game"], {"levels": 0, "tokens": 0, "first_clear_tokens": None})
        g["tokens"] += r["tokens"]
        if r["cleared"]:
            g["levels"] = max(g["levels"], r["level"])
            if r["level"] == 1:
                g["first_clear_tokens"] = r["tokens"]
    out["games"] = games
    explorer = [row for g in report.get("games", []) for row in (g.get("explorer") or [])]
    if explorer:
        out["explorer"] = {"runs": len(explorer), "cleared": sum(bool(r.get("cleared")) for r in explorer),
                           "actions": sum(r.get("actions", 0) for r in explorer)}
    return out


def main():
    results = [metrics(p) for p in sys.argv[1:]]
    for m in results:
        a, f, d = m["all"], m["first"], m["deep"]
        print(f"{m['run'][:58]:58s} score {m['score'] or 0:5.2f} | all {a['clears']:2d}/{a['tokens'] / 1e3:5.0f}K "
              f"= {a['per_10k']:.2f}±{a['ci95']:.2f} | L1 {f['clears']:2d}/{f['tokens'] / 1e3:4.0f}K = {f['per_10k']:.2f} "
              f"| deep {d['clears']:2d}/{d['tokens'] / 1e3:4.0f}K = {d['per_10k']:.2f}"
              + (f" | explorer {m['explorer']['cleared']}/{m['explorer']['runs']}" if m.get("explorer") else ""))
        print("   " + "  ".join(f"{g}:L{v['levels']}/{v['tokens'] / 1e3:.0f}K" for g, v in sorted(m["games"].items())))


if __name__ == "__main__":
    main()
