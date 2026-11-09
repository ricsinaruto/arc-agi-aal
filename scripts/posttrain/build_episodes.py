"""Build single-level episode games for rejection sampling into data/posttrain/episodes/ (CPU only).

Usage: uv run python scripts/posttrain/build_episodes.py [--root data/posttrain/episodes]
           [--per-mechanic 8] [--batches 4] [--mechanics q001 q003 ...] [--workers 8]

Every training mechanic (catalog q001-q020 minus split.GENERATED_HOLDOUT_MECHANICS; naming a held-out
one is a hard error) gets ``--per-mechanic`` episodes over tiers 1-3 with fresh seeds, some of them
skinned (posttrain.games.episodes.plan). Each episode is its own one-level game q200-q999, admitted
like every generated game. Batch ``b`` goes to ``<root>/batch-<b>/`` (environment_files + notes +
admission.json), a folder that ``scripts/build_arena.py --generated-games`` takes as a whole roster.
The whole set is rebuilt from scratch (existing batch-* folders are replaced); the build is
deterministic, so the game ids repeat. ``<root>/summary.json`` gives the per-mechanic and per-tier
counts.
"""

import argparse
import collections
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from posttrain.games import episodes  # noqa: E402


def run_group(args):
    specs, root, games_root, attempts, rollouts = args
    return episodes.build_group(specs, lambda spec: Path(root) / f"batch-{spec['batch']}", games_root,
                                attempts=attempts, random_rollouts=rollouts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/posttrain/episodes")
    parser.add_argument("--games-root", default="data/posttrain/games",
                        help="the generated-games store: its prefixes are taken, its levels are not repeated")
    parser.add_argument("--mechanics", nargs="*", default=None,
                        help="base prefixes (default: every training mechanic); a held-out one is an error")
    parser.add_argument("--per-mechanic", type=int, default=8)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=1000)
    parser.add_argument("--random-rollouts", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    started = time.monotonic()
    root = Path(args.root)
    mechanics = [episodes.check_mechanic(m) for m in args.mechanics] if args.mechanics \
        else episodes.training_mechanics()
    for old in root.glob("batch-*"):
        shutil.rmtree(old)
    specs = episodes.assign_prefixes(episodes.plan(mechanics, args.per_mechanic, args.batches),
                                     episodes.taken_prefixes(args.games_root))
    groups = collections.defaultdict(list)
    for spec in specs:
        groups[(spec["base"], spec["tier"])].append(spec)
    jobs = [(group, str(root), args.games_root, args.attempts, args.random_rollouts) for group in groups.values()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = [row for rows in pool.map(run_group, jobs) for row in rows]
    order = {spec["prefix"]: i for i, spec in enumerate(specs)}
    results.sort(key=lambda row: order[row[0]["prefix"]])
    batches = collections.defaultdict(lambda: {"episodes": {}, "rejected": {}})
    for spec, report in results:
        kind = "episodes" if report["admitted"] else "rejected"
        batches[spec["batch"]][kind][spec["prefix"]] = report
        status = f"{report['game_id']} baseline={report['baseline_actions']} random={report['random_clear_rate']}" \
            if report["admitted"] else "REJECTED " + "; ".join(report["issues"])
        print(f"  batch-{spec['batch']} {spec['prefix']} {spec['base']} tier {spec['tier']} seed {spec['seed']}"
              f"{' skin' if spec['skinned'] else ''}: {status}")
    summary = {"mechanics": mechanics, "per_mechanic": args.per_mechanic, "batches": {}, "by_mechanic_tier": {},
               "rejected": []}
    table = collections.defaultdict(lambda: collections.Counter())
    for batch, data in sorted(batches.items()):
        folder = root / f"batch-{batch}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "admission.json").write_text(json.dumps(data, indent=1) + "\n")
        admitted = data["episodes"].values()
        summary["batches"][f"batch-{batch}"] = {
            "games": len(data["episodes"]), "rejected": sorted(data["rejected"]),
            "prefixes": [min(data["episodes"]), max(data["episodes"])] if data["episodes"] else None,
            "tiers": dict(sorted(collections.Counter(r["tier"] for r in admitted).items())),
            "skinned": sum(1 for r in admitted if r["skin_of"]),
            "baseline_actions": sorted(r["baseline_actions"][0] for r in admitted)}
        for report in admitted:
            table[report["episode_of"]][f"tier {report['tier']}"] += 1
        summary["rejected"] += [{"prefix": p, **r} for p, r in data["rejected"].items()]
    summary["by_mechanic_tier"] = {m: dict(sorted(c.items())) for m, c in sorted(table.items())}
    (root / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for name, row in summary["batches"].items():
        print(f"{name}: {row['games']} games {row['prefixes']}, tiers {row['tiers']}, {row['skinned']} skinned, "
              f"rejected {row['rejected'] or 'none'}")
    print(f"{round(time.monotonic() - started, 1)} s")


if __name__ == "__main__":
    main()
