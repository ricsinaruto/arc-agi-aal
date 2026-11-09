"""Single-level episode games for rejection sampling.

Flash plays many short episodes on fresh layouts and only the most efficient verified clears are
kept (``posttrain.rejection``). An episode is a separate generated game with one level:

- the level is a fresh layout of a TRAINING mechanic (catalog ``qNNN`` outside
  ``split.GENERATED_HOLDOUT_MECHANICS``) at tier 1-3. Tier N is the mechanic's own level N:
  ``generate_level(rng, N - 1)`` with the module's ``TIER_ACTIONS[N - 1]`` band, or ``LEVELS[N - 1]``
  for a mechanic with fixed levels (q001, q007);
- some episodes are skinned (``posttrain.games.skins``): the same rules and level shown mirrored or
  turned, with permuted colours and screen-true keys. A skin is seeded by the episode, so fixed-level
  mechanics get fresh observations of their hand-made levels;
- the game gets a new prefix ``q200``-``q999`` (``split.is_generated`` accepts any ``qNNN``), and its
  notes record ``episode_of`` (the base mechanic's prefix), ``mechanic``, ``tier``, ``seed``,
  ``skin_of`` and ``batch``, so downstream code groups episodes and rejects held-out mechanics by
  ``episode_of`` (a ``q2NN`` prefix alone does not show the mechanic);
- it passes the same admission as every generated game (``admission.admit``): BFS path of at least
  4 actions, the real-arcade replay clears at the path end and wins, determinism, frames, and a
  random policy at 3x the baseline clears at most 5%. The BFS length is ``baseline_actions``.

Everything is deterministic in ``(mechanic, tier, seed)``: the level generator's rng, the skin and the
duplicate check (against the base game's levels and the same group's lower seeds) depend on nothing
else, so rebuilding gives the same game ids.
"""

import json
import random
import shutil
import tempfile
from pathlib import Path

from posttrain import split
from posttrain.games import admission, skins, store

TIERS = (1, 2, 3)
FIRST_PREFIX, LAST_PREFIX = 200, 999
DEFAULT_ROOT = Path("data/posttrain/episodes")


def catalog_path(base):
    matches = [p for p in store.catalog_modules() if p.name.split("_")[0] == base]
    if len(matches) != 1:
        raise ValueError(f"No catalog mechanic {base}")
    return matches[0]


def check_mechanic(base):
    """Raise unless ``base`` is a catalog mechanic that may give training data (hard error on held-out)."""
    if split.is_generated_holdout(base):
        raise ValueError(f"{base} is a held-out generated mechanic (split.GENERATED_HOLDOUT): "
                         "episodes are training data and are never built from it")
    catalog_path(base)
    return base


def training_mechanics():
    """Catalog mechanic prefixes minus the held-out ones, in catalog order."""
    return [p.name.split("_")[0] for p in store.catalog_modules()
            if not split.is_generated_holdout(p.name.split("_")[0])]


def plan(mechanics, per_mechanic=8, batches=4):
    """Episode specs ``{base, tier, seed, skinned, batch}``, balanced over mechanics and tiers.

    Mechanic ``m`` (its index) gets ``per_mechanic`` episodes; episode ``i`` has tier
    ``TIERS[(m + i) % 3]``, seed ``i // 3`` (so every tier gets fresh seeds) and batch ``i % batches + 1``
    (each batch holds every mechanic ``per_mechanic / batches`` times, and the tier rotation by ``m``
    balances tiers within a batch). Skins: about a third of a generated mechanic's episodes
    (``(i + m + seed) % 3 == 2``, which mixes plain and skinned inside every (mechanic, tier) group and
    spreads skins evenly over the batches); every seed but 0 of a fixed-level mechanic (seed 0 is its
    plain level, the others fresh skins)."""
    specs = []
    for m, base in enumerate(mechanics):
        check_mechanic(base)
        fixed = not hasattr(store.catalog_module(catalog_path(base)), "generate_level")
        for i in range(per_mechanic):
            seed = i // len(TIERS)
            specs.append({"base": base, "tier": TIERS[(m + i) % len(TIERS)], "seed": seed,
                          "skinned": seed >= 1 if fixed else (i + m + seed) % 3 == 2, "batch": i % batches + 1,
                          "order": (i % batches, m, i)})
    specs.sort(key=lambda s: s.pop("order"))
    return specs


def taken_prefixes(*roots):
    """Every qNNN prefix already in use: catalog mechanics, their skins, and the games under ``roots``
    (``environment_files/<prefix>`` folders)."""
    taken = set()
    for path in store.catalog_modules():
        base = path.name.split("_")[0]
        taken |= {base, skins.skin_prefix(base)}
    for root in roots:
        folder = Path(root) / "environment_files"
        if folder.is_dir():
            taken |= {p.name for p in folder.iterdir() if p.is_dir()}
    return taken


def assign_prefixes(specs, taken, start=FIRST_PREFIX):
    """Give each spec the next free prefix from ``q<start>`` on, in plan order (in place)."""
    number = start
    for spec in specs:
        while f"q{number:03d}" in taken:
            number += 1
        if number > LAST_PREFIX:
            raise ValueError(f"Episode prefixes exhausted (q{FIRST_PREFIX}-q{LAST_PREFIX})")
        spec["prefix"] = f"q{number:03d}"
        taken.add(spec["prefix"])
        number += 1
    return specs


def spec_key(spec):
    return json.dumps(spec, sort_keys=True)


def base_game_levels(base, games_root):
    """The level specs of the admitted base game (``data/posttrain/games``), or [] if absent."""
    files = sorted((Path(games_root) / "environment_files" / base).glob(f"*/{base}.py"))
    if len(files) != 1:
        return []
    return list(store.load_class(base, files[0].read_text()).LEVELS)


def assemble_episode(base, prefix, level, skin=None):
    """(prefix, source, meta) for a one-level game of ``base``, plain or through ``skin``."""
    module = catalog_path(base)
    if skin is None:
        return store.assemble(module, [level], prefix=prefix)
    _, _, base_meta = store.assemble(module)
    return skins.assemble_skin(module, prefix, [level], skin, base_meta)


def build_episode(spec, root, *, seen=(), attempts=1000, random_rollouts=200):
    """Generate, admit and write one episode game under ``root`` (``environment_files`` + ``notes``).

    ``spec`` is a plan entry with its ``prefix``. ``seen`` holds level keys that must not repeat (the
    base game's levels, the group's lower seeds). Returns ``(report, level key or None)``; the report
    has ``admitted`` and, when admitted, the game id and baseline."""
    base, tier, seed, prefix = check_mechanic(spec["base"]), int(spec["tier"]), int(spec["seed"]), spec["prefix"]
    if tier not in TIERS:
        raise ValueError(f"Episode tiers are {TIERS}, got {tier}")
    catalog = store.catalog_module(catalog_path(base))
    max_states = getattr(catalog, "MAX_STATES", 300_000)
    fixed = not hasattr(catalog, "generate_level")
    rng = random.Random(f"episode-{base}-t{tier}-s{seed}")
    low, high = (None, None) if fixed else catalog.TIER_ACTIONS[tier - 1]
    reasons = {}
    for attempt in range(attempts if (not fixed or spec["skinned"]) else 1):
        level = catalog.Game.LEVELS[tier - 1] if fixed else catalog.generate_level(rng, tier - 1)
        key = spec_key(level)
        if not fixed:
            if key in seen:
                reasons["duplicate"] = reasons.get("duplicate", 0) + 1
                continue
            ok, baseline, _, reason = admission.check_level(catalog.Game, level, max_states=max_states,
                                                            random_rollouts=random_rollouts)
            if ok and not low <= baseline <= high:
                ok, reason = False, "outside tier band"
            if not ok:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
        skin = None
        if spec["skinned"]:
            skin = skins.make_skin(f"episode-{base}-t{tier}-s{seed}-a{attempt}", getattr(catalog, "MOVES", None),
                                   catalog.Game.ACTIONS)
        prefix, source, meta = assemble_episode(base, prefix, level, skin)
        cls = store.load_class(prefix, source)
        with tempfile.TemporaryDirectory(prefix="episode_") as scratch:
            game_id = store.write_game(scratch, prefix, source, meta, [0])
            report = admission.admit(cls, Path(scratch) / "environment_files", game_id, max_states=max_states,
                                     random_rollouts=random_rollouts)
        if not report["admitted"]:
            reason = "admission: " + "; ".join(report["issues"])
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        row = report["levels"][0]
        episode = {"episode_of": base, "mechanic": meta["name"].removesuffix("-skin"), "tier": tier, "seed": seed,
                   "skin_of": base if skin else None, "batch": spec.get("batch"),
                   "level_source": "fixed" if fixed else "generated"}
        extra = {**episode, "skin": skin, "solution": [row["path"]],
                 "level": {"attempt": attempt, "baseline": row["baseline"], "random": row["random_clear_rate"],
                           "tier_band": None if fixed else [low, high]}}
        folder = Path(root) / "environment_files" / prefix
        if folder.exists():
            shutil.rmtree(folder)
        game_id = store.write_game(root, prefix, source, meta, report["baseline_actions"], extra)
        summary = {"game_id": game_id, "admitted": True, **episode, "baseline_actions": report["baseline_actions"],
                   "random_clear_rate": row["random_clear_rate"], "replay_frames": report["replay_frames"],
                   "attempt": attempt, "issues": [], "attempt_rejections": reasons}
        return summary, key
    return {"prefix": prefix, "admitted": False, "episode_of": base, "tier": tier, "seed": seed,
            "skinned": spec["skinned"], "batch": spec.get("batch"), "attempt_rejections": reasons,
            "issues": [f"no admissible level in {attempts} attempts"]}, None


def build_group(specs, root_of, games_root, attempts=1000, random_rollouts=200):
    """Build one (mechanic, tier) group in seed order; ``root_of(spec)`` is the batch folder."""
    reports = []
    seen = {spec_key(level) for level in base_game_levels(specs[0]["base"], games_root)}
    for spec in sorted(specs, key=lambda s: s["seed"]):
        report, key = build_episode(spec, root_of(spec), seen=seen, attempts=attempts,
                                    random_rollouts=random_rollouts)
        if key is not None:
            seen.add(key)
        reports.append((spec, report))
    return reports


def load_notes(roots=(DEFAULT_ROOT,)):
    """``{prefix: notes}`` for every episode under ``roots`` (each a batch folder or a folder of batches)."""
    notes = {}
    for root in roots:
        root = Path(root)
        for path in sorted(root.glob("notes/*.json")) + sorted(root.glob("*/notes/*.json")):
            data = json.loads(path.read_text())
            if "episode_of" not in data:
                continue
            prefix = split.prefix(data["game_id"])
            if prefix in notes and notes[prefix]["game_id"] != data["game_id"]:
                raise ValueError(f"Two episode games share the prefix {prefix}: "
                                 f"{notes[prefix]['game_id']} and {data['game_id']}")
            notes[prefix] = data | {"notes_path": str(path)}
    return notes


def environments_dir_for(game_id, roots=(DEFAULT_ROOT,)):
    """The ``environment_files`` folder that holds episode ``game_id`` (for engine replay), or None."""
    prefix, version = str(game_id).split("-", 1)
    for root in roots:
        for folder in [Path(root) / "environment_files"] + sorted(Path(root).glob("*/environment_files")):
            if (folder / prefix / version / "metadata.json").is_file():
                return folder
    return None
