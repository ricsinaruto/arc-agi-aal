"""Verdict of a held-out read (launch/rl/runs/b2-rl-hread-*.json, built by launch/rl/build.py): per arm and set the hero read's metrics (rl_loop.held_metrics: RHAE per
10K tokens, delta-method SE), the arm's L1 mean over the sets, and the pre-registered bar:
adapter - base >= 2 SE_diff, SE_diff = sqrt(sum over sets of se_adapter^2 + se_base^2) / sets.

    python launch/rl/heldout/read_result.py <trial dir with level-bench.json or attempts.jsonl> [--base base]
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def rows_of(folder):
    folder = Path(folder)
    if (folder / "level-bench.json").is_file():
        return json.loads((folder / "level-bench.json").read_text())["attempts"]
    return [json.loads(line) for line in (folder / "attempts.jsonl").read_text().splitlines() if line.strip()]


def verdict(rows, base="base", censored="drop"):
    """Pre-registered 25 Oct, before any result (the RTX read runs near its wall). ``censored="drop"`` (read A):
    tries the bench did not finish (level_bench.HEADLINE_EXCLUDED) are left out on both arms. ``"fail"`` (read B,
    ): a censored try counts as a non-clear at the tokens it used by the cut (dropping favours the
    arm that clears more slowly); other excluded outcomes are still left out. The verdict passes only if both reads
    pass the bar (``final``)."""
    from arena.level_bench import HEADLINE_EXCLUDED
    from arena.rl_loop import held_metrics

    dropped = {}
    for r in rows:
        if r.get("outcome") in HEADLINE_EXCLUDED:
            key = f"{r['arm']}:{r.get('outcome')}"
            dropped[key] = dropped.get(key, 0) + 1
    keep = ("censored",) if censored == "fail" else ()
    rows = [r for r in rows if r.get("outcome") not in HEADLINE_EXCLUDED or r.get("outcome") in keep]

    sets = sorted({str(r["game"])[:4] for r in rows})
    arms = sorted({r["arm"] for r in rows}, key=lambda a: (a != base, a))
    table = {a: {g: held_metrics([r for r in rows if r["arm"] == a and str(r["game"])[:4] == g]) for g in sets}
             for a in arms}
    out = {"censored_mode": censored, "sets": sets, "arms": {}, "excluded_counts": dropped}
    for a in arms:
        vals = [table[a][g]["rhae_per_10k"] or 0.0 for g in sets]
        out["arms"][a] = {"l1_rhae_per_10k": round(sum(vals) / len(sets), 4),
                          "tries": sum(table[a][g]["tries"] for g in sets), "per_set": table[a]}
    for a in arms[1:]:
        diff = out["arms"][a]["l1_rhae_per_10k"] - out["arms"][base]["l1_rhae_per_10k"]
        se = math.sqrt(sum((table[a][g]["rhae_10k_se"] or 0.0) ** 2 + (table[base][g]["rhae_10k_se"] or 0.0) ** 2
                           for g in sets)) / len(sets)
        out["arms"][a].update(diff=round(diff, 4), se_diff=round(se, 4), passes_bar=diff >= 2 * se)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--base", default="base")
    a = ap.parse_args()
    rows = rows_of(a.folder)
    reads = {mode: verdict(rows, a.base, mode) for mode in ("drop", "fail")}
    for mode, res in reads.items():
        print(f"read {mode}: excluded {res['excluded_counts']}")
        for arm, row in res["arms"].items():
            print(" ", arm, {k: v for k, v in row.items() if k != "per_set"})
    final = {arm: all(r["arms"][arm]["passes_bar"] for r in reads.values())
             for arm in reads["drop"]["arms"] if arm != a.base}
    print("FINAL (both reads pass):", final)
    print(json.dumps({"reads": reads, "final": final}))
