"""Build and admit the generated games (posttrain.games.catalog) and their skins into data/posttrain/games/.

Usage: uv run python scripts/posttrain/build_games.py [--root data/posttrain/games] [--only q001 q101 ...]
       [--no-skins]
Admitted games land in <root>/environment_files (loadable by the arc_agi offline arcade, the Duck
harness and the arena); rejected games are removed from it and their reports kept (rejects persist,
paper-notes 2.7). Mechanics (catalog modules qNNN) are built first; then every admitted mechanic gets
one skin q1NN (posttrain.games.skins: same rules and levels, turned board, permuted colours), built
from the admitted base game's levels. Writes <root>/admission.json with the two kinds kept apart:
{"mechanics": {prefix: report}, "skins": {prefix: report}}. ``--only`` filters by exact prefix.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from posttrain.games import admission, skins, store  # noqa: E402


def admit_and_write(root, prefix, source, meta, extra, max_states):
    """Write, admit, and keep (with baselines and solution) or remove one game; returns the report."""
    cls = store.load_class(prefix, source)
    folder = root / "environment_files" / prefix
    if folder.exists():
        shutil.rmtree(folder)
    game_id = store.write_game(root, prefix, source, meta, [0] * len(cls.LEVELS))
    report = admission.admit(cls, root / "environment_files", game_id, max_states=max_states)
    extra = dict(extra)
    if extra.get("expected_baselines") is not None and report["admitted"] \
            and report["baseline_actions"] != extra["expected_baselines"]:
        report["issues"].append(f"baselines {report['baseline_actions']} differ from the base game's "
                                f"{extra['expected_baselines']}")
        report["admitted"] = False
    extra.pop("expected_baselines", None)
    if report["admitted"]:
        store.write_game(root, prefix, source, meta, report["baseline_actions"],
                         {**extra, "solution": [row["path"] for row in report["levels"]]})
    else:
        shutil.rmtree(folder)
    status = "ADMITTED" if report["admitted"] else "REJECTED " + "; ".join(report["issues"])
    print(f"  {game_id}: {status} baselines={report.get('baseline_actions')} "
          f"random={[row.get('random_clear_rate') for row in report['levels']]}")
    return {k: v for k, v in report.items() if k != "levels"} | {
        "levels": [{k: v for k, v in row.items() if k != "path"} for row in report["levels"]]}


def build_mechanic(root, module, attempts):
    prefix = module.name.split("_")[0]
    catalog = store.catalog_module(module)
    max_states = getattr(catalog, "MAX_STATES", 300_000)
    levels, picks = None, []
    if hasattr(catalog, "generate_level"):
        levels = []
        for tier, (low, high) in enumerate(catalog.TIER_ACTIONS):
            rng = random.Random(f"{prefix}-{tier}")
            reasons = {}
            for attempt in range(attempts):
                spec = catalog.generate_level(rng, tier)
                ok, baseline, rate, reason = admission.check_level(catalog.Game, spec, max_states=max_states)
                if ok and not low <= baseline <= high:
                    ok, reason = False, "outside tier band"
                reasons[reason] = reasons.get(reason, 0) + 1
                if ok:
                    levels.append(spec)
                    picks.append({"tier": tier, "attempt": attempt, "baseline": baseline, "random": rate})
                    break
            else:
                print(f"  {prefix}: tier {tier} found no level in {attempts} attempts {reasons}")
        if not levels:
            return None
    prefix, source, meta = store.assemble(module, levels)
    return admit_and_write(root, prefix, source, meta, {"generated_levels": picks}, max_states)


def build_skin(root, module, reports):
    base = module.name.split("_")[0]
    prefix = skins.skin_prefix(base)
    base_report = reports["mechanics"].get(base)
    files = sorted((root / "environment_files" / base).glob(f"*/{base}.py"))
    if not base_report or not base_report.get("admitted") or len(files) != 1:
        print(f"  {prefix}: skipped, base game {base} is not admitted")
        return None
    base_cls = store.load_class(base, files[0].read_text())
    base_notes = json.loads((root / "notes" / f"{base}.json").read_text())
    catalog = store.catalog_module(module)
    skin = skins.make_skin(prefix, getattr(catalog, "MOVES", None), catalog.Game.ACTIONS)
    prefix, source, meta = skins.assemble_skin(module, prefix, base_cls.LEVELS, skin, base_notes)
    extra = {"skin_of": base, "skin_of_game_id": base_notes["game_id"], "skin": skin,
             "expected_baselines": base_report["baseline_actions"]}
    report = admit_and_write(root, prefix, source, meta, extra, getattr(catalog, "MAX_STATES", 300_000))
    return report | {"skin_of": base, "skin_of_game_id": base_notes["game_id"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/posttrain/games")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--attempts", type=int, default=300)
    parser.add_argument("--no-skins", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    out = root / "admission.json"
    reports = json.loads(out.read_text()) if out.exists() else {}
    if "mechanics" not in reports:  # the first builds kept mechanics at the top level
        reports = {"mechanics": reports, "skins": {}}
    print("Mechanics")
    for module in store.catalog_modules():
        prefix = module.name.split("_")[0]
        if args.only and prefix not in args.only:
            continue
        report = build_mechanic(root, module, args.attempts)
        if report is not None:
            reports["mechanics"][prefix] = report
    if not args.no_skins:
        print("Skins")
        for module in store.catalog_modules():
            prefix = skins.skin_prefix(module.name.split("_")[0])
            if args.only and prefix not in args.only:
                continue
            report = build_skin(root, module, reports)
            if report is not None:
                reports["skins"][prefix] = report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=1) + "\n")
    for kind in ("mechanics", "skins"):
        admitted = sorted(p for p, r in reports[kind].items() if r.get("admitted"))
        rejected = sorted(p for p, r in reports[kind].items() if not r.get("admitted"))
        print(f"{kind}: {len(admitted)} admitted, rejected {rejected or 'none'}")


if __name__ == "__main__":
    main()
