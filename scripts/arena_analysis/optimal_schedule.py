"""Optimal-scheduling ceiling from recorded level costs.

1. Table: mean completion tokens to clear each (game, level), over the recorded one-hour 23-game runs that
   share the baseline's harness and settings (Flash-Next NVFP4, xhigh, 128K windows, Duck-transfer harness;
   runs that differ only in the scheduler). Only cleared attempts enter the mean; `n` shows clears/attempts.
2. Budget: phase-2 aggregate throughput x wall seconds (9 hours for the hidden run).
3. Knapsack: choose, per game copy, how many levels to clear (a prefix: level k costs levels 1..k) to maximise
   the mean game score (level k of an N-level game weighs k / (N(N+1)/2); a cleared level scores
   ``level_score``) within the budget. The 110-game roster repeats the 23 games equally (every game 4 times,
   then the first 18 once more in roster order).

``--cost mean`` uses the mean tokens of the cleared attempts (what the table shows; optimistic, because
failed attempts are left out). ``--cost ratio`` uses all tokens spent on the level over all attempts divided
by the clears (the expected cost per clear under a constant clear rate).

    python scripts/arena_analysis/optimal_schedule.py --attempts <level_attempts.jsonl> [--markdown]
"""

import argparse
import json
import math
from collections import defaultdict

import numpy as np

ROSTER = ["ar25", "bp35", "cd82", "cn04", "dc22", "g50t", "ka59", "lf52", "ls20", "m0r0", "r11l", "re86",
          "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30"]
# One-hour 23-game runs whose harness, model, reasoning and windows equal the baseline's (scheduler-only changes).
BASELINE_RUNS = (
    "arena-duck-60m-r23-base-v2-modal-20260925", "arena-duck-60m-r23-base-v3-modal-20260925",
    "kaggle-arena-duck-60m-r23-base-v4", "kaggle-arena-duck-60m-r23-base-v5",
    "kaggle-arena-duck-60m-r23-budget1x-v1", "arena-duck-60m-r23-budget4x-v1-modal-20260925",
    "arena-duck-60m-r23-reopen-pause-v1-modal-20260925", "arena-duck-60m-r23-slots1-v1-modal-20260925",
    "kaggle-arena-duck-60m-r23-thompson-v1", "kaggle-arena-duck-60m-r23-token-budget-v1",
    "kaggle-arena-duck-60m-r23-index-v2", "arena-duck-60m-r23-index-v3-modal-20260926",
    "kaggle-arena-duck-60m-r23-index-lane-v1", "kaggle-arena-duck-60m-r23-index-lane-v2",
    "kaggle-arena-duck-60m-r23-index-lane-v3", "kaggle-arena-duck-60m-r23-commit-v1",
    "kaggle-arena-duck-60m-r23-commit-queue-v1", "kaggle-arena-duck-60m-r23-gittins-broad-v1",
    "kaggle-arena-duck-60m-r23-gittins-broad-v2", "arena-duck-60m-r23-llm-rank-v1-modal-20260926",
)


def level_table(rows):
    """{game: {level: {"clears", "attempts", "mean", "ratio", "total_levels"}}} over the baseline runs."""
    table = defaultdict(dict)
    acc = defaultdict(lambda: {"clears": [], "spent": 0, "attempts": 0})
    total = {}
    for r in rows:
        if r["run"] not in BASELINE_RUNS or r["game"] not in ROSTER:
            continue
        if r["cleared"] and r["tokens"] <= 0:
            continue  # a level cleared in the same observation as the previous one: no tokens of its own
        a = acc[(r["game"], r["level"])]
        a["attempts"] += 1
        a["spent"] += r["tokens"]
        if r["cleared"]:
            a["clears"].append(r["tokens"])
        total[r["game"]] = max(total.get(r["game"], 0), r["total_levels"] or 0)
    for (game, level), a in acc.items():
        n = len(a["clears"])
        table[game][level] = {"clears": n, "attempts": a["attempts"], "total_levels": total[game],
                              "mean": sum(a["clears"]) / n if n else math.inf,
                              "ratio": a["spent"] / n if n else math.inf}
    return table, total


def prefix_options(table, total, game, cost):
    """[(tokens, score share)] for clearing levels 1..j, j = 0.. while every level so far has a clear."""
    n = total[game]
    weights = n * (n + 1) / 2
    options, tokens, share = [(0.0, 0.0)], 0.0, 0.0
    for level in range(1, n + 1):
        cell = table[game].get(level)
        if cell is None or not math.isfinite(cell[cost]):
            break
        tokens += cell[cost]
        share += level / weights
        options.append((tokens, share))
    return options


def knapsack(items, budget, unit=1000):
    """items: list of option lists [(tokens, value)]; returns (best value, choice index per item)."""
    cap = int(budget // unit)
    best = np.zeros(cap + 1)
    choices = []
    for options in items:
        new = best.copy()
        arg = np.zeros(cap + 1, dtype=np.int16)
        for j, (tokens, value) in enumerate(options[1:], 1):
            c = int(math.ceil(tokens / unit))
            if c > cap:
                break
            cand = np.full(cap + 1, -1.0)
            cand[c:] = best[:cap + 1 - c] + value
            better = cand > new
            new[better], arg[better] = cand[better], j
        choices.append(arg)
        best = new
    b, picks = cap, []
    for options, arg in zip(reversed(items), reversed(choices)):
        j = int(arg[b])
        picks.append(j)
        b -= int(math.ceil(options[j][0] / unit)) if j else 0
    return float(best[cap]), picks[::-1]


def roster(games, size):
    reps = [g for _ in range(size // len(games)) for g in games]
    return reps + games[: size - len(reps)]


def solve(table, total, games, budget, cost, level_score=100.0):
    items = [prefix_options(table, total, g, cost) for g in games]
    value, picks = knapsack(items, budget)
    return 100.0 * value * level_score / 100.0 / len(games), picks, items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", required=True)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.attempts)]
    table, total = level_table(rows)
    runs = sorted({r["run"] for r in rows if r["run"] in BASELINE_RUNS})
    print(f"{len(runs)} baseline-equivalent runs of {len(BASELINE_RUNS)} listed")
    games = [g for g in ROSTER if g in total]
    depth = max(total.values())
    if args.markdown:
        print("\nMean completion tokens (K) to clear each level; clears/attempts in brackets; '-' = never cleared\n")
        print("| level | " + " | ".join(games) + " |")
        print("|---:|" + "---:|" * len(games))
        for level in range(1, depth + 1):
            cells = []
            for g in games:
                if level > total[g]:
                    cells.append("")
                    continue
                c = table[g].get(level)
                if c is None:
                    cells.append("-")
                elif c["clears"] == 0:
                    cells.append(f"- (0/{c['attempts']})")
                else:
                    cells.append(f"{c['mean'] / 1000:.0f} ({c['clears']}/{c['attempts']})")
            print(f"| {level} | " + " | ".join(cells) + " |")
    print()
    for cost in ("mean", "ratio"):
        for label, size, seconds in (("1 h, 23 games", 23, 3600), ("9 h, 110 games", 110, 32400)):
            for rate in (200, 225, 250):
                budget = rate * seconds
                score, picks, _ = solve(table, total, roster(games, size), budget, cost)
                print(f"cost={cost:5s} {label:15s} {rate} tok/s  budget {budget / 1e6:5.2f}M  ceiling {score:6.2f}%  "
                      f"levels {sum(picks)}")
    # The chosen plan at 225 tok/s, 9 h, 110 games, mean cost.
    for cost in ("mean", "ratio"):
        g110 = roster(games, 110)
        score, picks, items = solve(table, total, g110, 225 * 32400, cost)
        per = defaultdict(list)
        for g, j, opts in zip(g110, picks, items):
            per[g].append((j, opts[j][0]))
        print(f"\nPlan (cost={cost}, 9 h, 225 tok/s, 110 games): score {score:.2f}%")
        for g in games:
            levels = [j for j, _ in per[g]]
            print(f"  {g}: levels per copy {levels} of {total[g]}; max clearable {len(prefix_options(table, total, g, cost)) - 1}; "
                  f"tokens per copy {[round(t / 1000) for _, t in per[g]]}K")


if __name__ == "__main__":
    main()
