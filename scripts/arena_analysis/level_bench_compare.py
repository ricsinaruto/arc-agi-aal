"""Pair two level-bench runs on the same start points (27 September; ``arena.level_bench``).

Attempts pair on ``(start point, pass)``: two runs with the same bench settings attempt the same
seeded list in the same order, so the attempts both completed (censored, errored and mismatched ones
excluded) are paired draws on identical boards. Reported:

- McNemar counts (both clear, A only, B only, neither) with the exact two-sided p-value on the
  discordant pairs, and the paired clear-rate difference with a 95% interval;
- clears per 10K completion tokens of each run on the paired attempts, and the rate ratio B/A with an
  exact conditional (binomial) 95% interval;
- the paired token ratio B/A on pairs both cleared (geometric mean with a 95% t interval on the
  logs, median);
- the mean per-level score of each run and the paired difference.

``--budget B`` re-scores both runs at a smaller token budget: an attempt clears at B only if its
tokens to clear are at most B, and its tokens are capped at B.

Usage:
    python scripts/arena_analysis/level_bench_compare.py A B [--budget 30000] [--per-level] [--json out.json]

A and B are ``level-bench.json`` files or directories holding one (searched recursively).
"""

import argparse
import json
import math
import statistics
from pathlib import Path

EXCLUDED = ("censored", "error", "replay_mismatch")


def load(path):
    path = Path(path)
    if path.is_dir():
        found = sorted(path.rglob("level-bench.json"))
        if len(found) != 1:
            raise SystemExit(f"{path}: expected one level-bench.json, found {len(found)}")
        path = found[0]
    report = json.loads(path.read_text())
    return path, report


def at_budget(attempt, budget):
    """``(cleared, tokens)`` of a completed attempt at ``budget`` (None: the run's own budget)."""
    cleared = bool(attempt["cleared"])
    tokens = int(attempt["completion_tokens"])
    if budget is None:
        return cleared, tokens
    if cleared and attempt["tokens_to_clear"] is not None and attempt["tokens_to_clear"] <= budget:
        return True, int(attempt["tokens_to_clear"])
    return False, min(tokens, int(budget))


def completed(report):
    return {(a["point"], a["pass"]): a for a in report["attempts"] if a["outcome"] not in EXCLUDED}


def binom_two_sided(k, n):
    """Exact two-sided binomial test at p = 0.5."""
    if n == 0:
        return 1.0
    from scipy.stats import binomtest

    return float(binomtest(k, n, 0.5).pvalue)


def rate_ratio(clears_a, tokens_a, clears_b, tokens_b, alpha=0.05):
    """Ratio of Poisson rates B/A with the exact conditional (Clopper-Pearson) interval."""
    from scipy.stats import beta

    if not tokens_a or not tokens_b or clears_a + clears_b == 0:
        return None, [None, None]
    n = clears_a + clears_b
    low = 0.0 if clears_b == 0 else beta.ppf(alpha / 2, clears_b, n - clears_b + 1)
    high = 1.0 if clears_b == n else beta.ppf(1 - alpha / 2, clears_b + 1, n - clears_b)
    scale = tokens_a / tokens_b

    def ratio(p):
        return math.inf if p >= 1 else p / (1 - p) * scale

    point = math.inf if clears_a == 0 else clears_b / tokens_b / (clears_a / tokens_a)
    return point, [ratio(low), ratio(high)]


def mean_ci(values, z=1.96):
    if not values:
        return None, [None, None]
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, [None, None]
    half = z * statistics.stdev(values) / math.sqrt(len(values))
    return mean, [mean - half, mean + half]


def compare(report_a, report_b, *, budget=None, split=None):
    a, b = completed(report_a), completed(report_b)
    keys = sorted(k for k in set(a) & set(b) if split is None or a[k]["split"] == split)
    counts = {"both": 0, "a_only": 0, "b_only": 0, "neither": 0}
    tokens = {"a": 0, "b": 0}
    clears = {"a": 0, "b": 0}
    log_ratios, score_diffs, scores = [], [], {"a": [], "b": []}
    per_level = {}
    for key in keys:
        clear_a, tok_a = at_budget(a[key], budget)
        clear_b, tok_b = at_budget(b[key], budget)
        counts["both" if clear_a and clear_b else "a_only" if clear_a else "b_only" if clear_b else "neither"] += 1
        tokens["a"] += tok_a
        tokens["b"] += tok_b
        clears["a"] += clear_a
        clears["b"] += clear_b
        if clear_a and clear_b:
            log_ratios.append(math.log(max(1, tok_b) / max(1, tok_a)))
        score_a = a[key]["level_score"] if clear_a else 0.0
        score_b = b[key]["level_score"] if clear_b else 0.0
        scores["a"].append(score_a)
        scores["b"].append(score_b)
        score_diffs.append(score_b - score_a)
        row = per_level.setdefault(key[0], {"pairs": 0, "clears_a": 0, "clears_b": 0, "tokens_a": 0, "tokens_b": 0})
        row["pairs"] += 1
        row["clears_a"] += clear_a
        row["clears_b"] += clear_b
        row["tokens_a"] += tok_a
        row["tokens_b"] += tok_b
    n = len(keys)
    discordant = counts["a_only"] + counts["b_only"]
    diff = (counts["b_only"] - counts["a_only"]) / n if n else None
    diff_ci = [None, None]
    if n:
        se = math.sqrt(max(0.0, discordant - (counts["b_only"] - counts["a_only"]) ** 2 / n)) / n
        diff_ci = [diff - 1.96 * se, diff + 1.96 * se]
    ratio, ratio_ci = rate_ratio(clears["a"], tokens["a"], clears["b"], tokens["b"])
    log_mean, log_ci = mean_ci(log_ratios)
    score_diff, score_ci = mean_ci(score_diffs)
    return {
        "pairs": n, "budget": budget, "split": split, "mcnemar": counts,
        "mcnemar_p_two_sided": binom_two_sided(counts["b_only"], discordant),
        "clear_rate": {"a": clears["a"] / n if n else None, "b": clears["b"] / n if n else None},
        "clear_rate_diff_b_minus_a": diff, "clear_rate_diff_95": diff_ci,
        "clears_per_10k_tokens": {"a": clears["a"] * 1e4 / tokens["a"] if tokens["a"] else None,
                                  "b": clears["b"] * 1e4 / tokens["b"] if tokens["b"] else None},
        "rate_ratio_b_over_a": ratio, "rate_ratio_95": ratio_ci,
        "both_cleared": len(log_ratios),
        "token_ratio_b_over_a_geomean": None if log_mean is None else math.exp(log_mean),
        "token_ratio_95": [None if v is None else math.exp(v) for v in log_ci],
        "token_ratio_median": math.exp(statistics.median(log_ratios)) if log_ratios else None,
        "mean_level_score": {"a": sum(scores["a"]) / n if n else None, "b": sum(scores["b"]) / n if n else None},
        "level_score_diff_b_minus_a": score_diff, "level_score_diff_95": score_ci,
        "unpaired": {"a": len(set(a) - set(b)), "b": len(set(b) - set(a))},
        "per_level": per_level,
    }


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return "inf" if math.isinf(value) else f"{value:.{digits}f}"
    return str(value)


def describe(report):
    s = report.get("settings") or {}
    return (f"{report.get('trial', {}).get('name')} serving={s.get('serving')} split={s.get('split')} "
            f"budget={s.get('token_budget')} window={report.get('window')} effort={report.get('reasoning_effort')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--budget", type=int, default=None, help="Re-score both runs at this completion-token budget")
    parser.add_argument("--split", choices=["dev", "holdout"], default=None)
    parser.add_argument("--per-level", action="store_true")
    parser.add_argument("--json", default=None, help="Write the comparison here")
    args = parser.parse_args()
    path_a, report_a = load(args.a)
    path_b, report_b = load(args.b)
    for label, report in (("a", report_a), ("b", report_b)):
        own = (report.get("settings") or {}).get("token_budget")
        if args.budget is not None and own is not None and args.budget > own:
            raise SystemExit(f"--budget {args.budget} exceeds run {label}'s own budget {own}")
    result = compare(report_a, report_b, budget=args.budget, split=args.split)
    result.update(a=str(path_a), b=str(path_b))
    print(f"A: {describe(report_a)}\nB: {describe(report_b)}")
    c = result["mcnemar"]
    print(f"pairs {result['pairs']} (unpaired A {result['unpaired']['a']}, B {result['unpaired']['b']})"
          f"{'' if args.budget is None else f' at budget {args.budget}'}")
    print(f"McNemar: both {c['both']}  A only {c['a_only']}  B only {c['b_only']}  neither {c['neither']}  "
          f"exact p {fmt(result['mcnemar_p_two_sided'])}")
    print(f"clear rate A {fmt(result['clear_rate']['a'])}  B {fmt(result['clear_rate']['b'])}  "
          f"diff {fmt(result['clear_rate_diff_b_minus_a'])} [{', '.join(fmt(v) for v in result['clear_rate_diff_95'])}]")
    print(f"clears/10K A {fmt(result['clears_per_10k_tokens']['a'])}  B {fmt(result['clears_per_10k_tokens']['b'])}  "
          f"ratio B/A {fmt(result['rate_ratio_b_over_a'])} [{', '.join(fmt(v) for v in result['rate_ratio_95'])}]")
    print(f"tokens to clear B/A on {result['both_cleared']} pairs both cleared: geomean "
          f"{fmt(result['token_ratio_b_over_a_geomean'])} [{', '.join(fmt(v) for v in result['token_ratio_95'])}], "
          f"median {fmt(result['token_ratio_median'])}")
    print(f"mean level score A {fmt(result['mean_level_score']['a'], 2)}  B {fmt(result['mean_level_score']['b'], 2)}  "
          f"diff {fmt(result['level_score_diff_b_minus_a'], 2)} "
          f"[{', '.join(fmt(v, 2) for v in result['level_score_diff_95'])}]")
    if args.per_level:
        print("\n| start point | pairs | clears A | clears B | tokens A | tokens B |\n|---|---:|---:|---:|---:|---:|")
        for point, row in sorted(result["per_level"].items()):
            print(f"| {point} | {row['pairs']} | {row['clears_a']} | {row['clears_b']} | {row['tokens_a']} | {row['tokens_b']} |")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
