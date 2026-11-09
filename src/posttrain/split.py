"""The game split for post-training data (the evolve-v4 dev/holdout split).

Training data may come from DEV games and from generated games only. HOLDOUT games are evaluation
only: no trajectories, auxiliary tasks or solution scripts from them may enter a training set.

GENERATED_HOLDOUT (data audit, section 5) holds five generated mechanics and their skins out of every
training source: q006 mirror (click, hand-made levels), q009 portals (keys), q013 fuel (keys, an
on-screen fuel bar), q015 guards (keys + ACTION5, moving agents) and q018 pairs (click). They are
evaluation-only (held-out-mechanic arena runs, next-action and loss checks); ``check_training_games``
rejects them and ``check_eval_games`` accepts only them.
"""

import re

DEV = ("ar25", "tu93", "vc33", "wa30", "tn36", "lf52", "cn04", "bp35", "r11l", "sp80", "m0r0", "ka59", "sc25", "sk48")
HOLDOUT = ("sb26", "re86", "s5i5", "su15", "dc22", "cd82", "g50t", "ls20", "tr87")
EXCLUDED = ("ft09", "lp85")  # not on the 23-game arena roster
GENERATED_HOLDOUT_MECHANICS = ("q006", "q009", "q013", "q015", "q018")
GENERATED_HOLDOUT = GENERATED_HOLDOUT_MECHANICS + tuple(f"q1{m[2:]}" for m in GENERATED_HOLDOUT_MECHANICS)


def prefix(game_id):
    return str(game_id).split("-")[0]


def is_generated(game_id):
    """Generated games (posttrain.games) have ids ``qNNN-<version>``."""
    return re.match(r"^q\d{3}$", prefix(game_id)) is not None


def is_generated_holdout(game_id):
    return prefix(game_id) in GENERATED_HOLDOUT


def is_dev(game_id):
    return prefix(game_id) in DEV


def is_training_game(game_id):
    return is_dev(game_id) or (is_generated(game_id) and not is_generated_holdout(game_id))


def check_training_games(game_ids):
    """Raise if any game in a training set is neither a dev game nor a training generated game."""
    bad = sorted({g for g in game_ids if not is_training_game(g)})
    if bad:
        raise ValueError(f"Training data from non-dev games (holdout or held-out generated mechanics): {bad}")


def check_eval_games(game_ids):
    """Raise if any game in the generated-holdout evaluation split is not a held-out generated game."""
    bad = sorted({g for g in game_ids if not is_generated_holdout(g)})
    if bad:
        raise ValueError(f"Evaluation split holds games outside GENERATED_HOLDOUT: {bad}")
