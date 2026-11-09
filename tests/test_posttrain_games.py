"""Generated ARC-like games (posttrain.games): assembly, loader compatibility, solver and admission."""

import random

import numpy as np
import pytest

from posttrain import split
from posttrain.games import admission, store


def catalog(prefix):
    return next(p for p in store.catalog_modules() if p.name.startswith(prefix + "_"))


def test_split_keeps_holdout_games_out_of_training_data():
    assert len(split.DEV) == 14 and len(split.HOLDOUT) == 9 and not set(split.DEV) & set(split.HOLDOUT)
    split.check_training_games(["ar25-0c556536", "q001-c8ba2de3"])
    with pytest.raises(ValueError, match="sb26"):
        split.check_training_games(["q001-c8ba2de3", "sb26-7fbdac44"])
    assert split.is_generated("q012-abcdef12") and not split.is_generated("qq01-x")


def test_split_holds_five_generated_mechanics_and_their_skins_out_of_training():
    assert split.GENERATED_HOLDOUT == ("q006", "q009", "q013", "q015", "q018",
                                       "q106", "q109", "q113", "q115", "q118")
    for game_id in ("q006-a3e60ccd", "q109-cbce58d0", "q018-e2171764", "q118-96bf64c5"):
        assert split.is_generated(game_id) and split.is_generated_holdout(game_id)
        assert not split.is_training_game(game_id)
        with pytest.raises(ValueError, match=game_id):
            split.check_training_games(["ar25-0c556536", game_id])
    split.check_training_games(["q001-c8ba2de3", "q101-d8be35ca", "q019-810bf976", "vc33-5430563c"])
    split.check_eval_games(["q013-91bc7865", "q115-51409022"])
    for game_id in ("q001-c8ba2de3", "sb26-7fbdac44", "ar25-0c556536"):
        with pytest.raises(ValueError, match="GENERATED_HOLDOUT"):
            split.check_eval_games([game_id])


def test_assembled_game_hides_the_designer_note_and_loads_through_the_public_arcade(tmp_path):
    prefix, source, meta = store.assemble(catalog("q001"))
    assert prefix == "q001" and "Hidden rules" in meta["note"] and "Hidden rules" not in source
    assert "noqa" not in source and "class Q001(Game)" in source
    cls = store.load_class(prefix, source)
    report_levels = [len(admission.solve_level(cls, i)) for i in range(len(cls.LEVELS))]
    game_id = store.write_game(tmp_path, prefix, source, meta, report_levels)
    report = admission.admit(cls, tmp_path / "environment_files", game_id, random_rollouts=40)
    assert report["admitted"], report["issues"]
    assert report["baseline_actions"] == report_levels and report["replay_frames"] > sum(report_levels)


def test_admission_rejects_unsolvable_and_trivial_levels():
    module = store.catalog_module(catalog("q001"))
    blocked = {"map": ["######", "#@#.X#", "######"]}
    trivial = {"map": ["#####", "#@.X#", "#####"]}
    assert admission.check_level(module.Game, blocked)[3] == "unsolvable"
    assert admission.check_level(module.Game, trivial)[3] == "too short"


def test_generated_levels_are_solvable_by_construction():
    module = store.catalog_module(catalog("q003"))
    rng = random.Random(0)
    solved = 0
    for _ in range(5):
        spec = module.generate_level(rng, 1)
        single = type("One", (module.Game,), {"LEVELS": [spec]})
        path = admission.solve_level(single, 0)
        solved += path is not None
    assert solved == 5


def test_click_coordinates_map_back_to_the_clicked_cell():
    from arcengine import Camera

    for width, height in ((5, 5), (7, 4), (9, 6), (16, 16)):
        camera = Camera(width=width, height=height)
        for cell in ((0, 0), (width - 1, height - 1), (width // 2, height // 2)):
            assert camera.display_to_grid(*admission.display_xy(width, height, cell)) == cell


def _admissible_level(module, tier, attempts=40):
    rng = random.Random(f"test-{tier}")
    for _ in range(attempts):
        spec = module.generate_level(rng, tier)
        ok, baseline, _, _ = admission.check_level(module.Game, spec, max_states=getattr(module, "MAX_STATES", 300_000))
        if ok:
            return spec, baseline
    return None, None


@pytest.mark.parametrize("path", store.catalog_modules(), ids=lambda p: p.stem)
def test_every_catalog_module_builds(path):
    """Each mechanic states its hidden rules, keeps them out of the game file, has at least 4 levels
    (fixed or tiers), and yields an admissible first level."""
    prefix, source, meta = store.assemble(path)
    module = store.catalog_module(path)
    rules = meta["note"].split("Hidden rules", 1)[1].split(":", 1)[1].split()
    assert meta["name"] and meta["tags"] and len(rules) > 30
    assert "Hidden rules" not in source and " ".join(rules[:8]) not in " ".join(source.split())
    if hasattr(module, "generate_level"):
        assert len(module.TIER_ACTIONS) >= 4
        spec, baseline = _admissible_level(module, 0)
        assert spec is not None and baseline >= 4, f"{prefix}: no admissible tier-0 level"
    else:
        assert len(module.Game.LEVELS) >= 4
        store.load_class(prefix, source)


def test_catalog_has_distinct_mechanics_and_uses_every_input_kind():
    modules = [store.catalog_module(p) for p in store.catalog_modules()]
    assert len(modules) >= 20 and len({m.NAME for m in modules}) == len(modules)
    actions = [set(m.Game.ACTIONS) for m in modules]
    assert any(6 in a for a in actions) and any(5 in a for a in actions) and any(a & {1, 2, 3, 4} and 6 in a for a in actions)


@pytest.mark.parametrize("base", ["q003", "q008", "q014"])
def test_skins_preserve_solvability_and_baselines(base):
    """A skin (turned board, permuted colours, remapped keys) solves in exactly the base game's optimal
    counts, and its frames are the base frames recoloured and turned."""
    from posttrain.games import skins

    path = catalog(base)
    module = store.catalog_module(path)
    levels = [_admissible_level(module, tier)[0] for tier in (0, 1)]
    _, base_source, meta = store.assemble(path, levels)
    base_cls = store.load_class(base, base_source)
    prefix = skins.skin_prefix(base)
    skin = skins.make_skin(prefix, getattr(module, "MOVES", None), module.Game.ACTIONS)
    assert skin["flip"] or skin["turns"]
    assert all(i != c for i, c in enumerate(skin["palette"])) and sorted(skin["palette"]) == list(range(16))
    prefix, source, skin_meta = skins.assemble_skin(path, prefix, base_cls.LEVELS, skin, meta)
    assert skin_meta["note"].startswith(meta["note"]) and "Skin of " + base in skin_meta["note"]
    assert "Hidden rules" not in source and "class Q1" in source
    cls = store.load_class(prefix, source)
    for index in range(len(levels)):
        base_path = admission.solve_level(base_cls, index)
        skin_path = admission.solve_level(cls, index)
        assert len(skin_path) == len(base_path)
        game, base_game = cls(), base_cls()
        game.set_level(index)
        base_game.set_level(index)
        expected = np.asarray(skin["palette"])[base_game.render_grid()]
        if skin["flip"]:
            expected = expected[:, ::-1]
        assert (game.render_grid() == np.rot90(expected, -skin["turns"])).all()


def test_skin_keys_follow_the_screen_and_replay_through_the_arcade(tmp_path):
    """In a turned skin, a key moves the avatar that way on screen; a click skin admits through the
    real arcade (clicks land on the displayed cells)."""
    from posttrain.games import skins

    path = catalog("q001")
    module = store.catalog_module(path)
    _, base_source, meta = store.assemble(path)
    base_cls = store.load_class("q001", base_source)
    skin = skins.make_skin("q101", module.MOVES, module.Game.ACTIONS)
    prefix, source, skin_meta = skins.assemble_skin(path, "q101", base_cls.LEVELS, skin, meta)
    cls = store.load_class(prefix, source)
    game = cls()
    game.set_level(0)
    avatar = skin["palette"][9]  # the blue avatar is the only base cell of colour 9
    moved = 0
    for key, (dx, dy) in skins.STANDARD_MOVES.items():
        game.set_level(0)
        before = tuple(int(v) for v in np.argwhere(game.render_grid() == avatar)[0])
        game.apply(key, None)
        after = tuple(int(v) for v in np.argwhere(game.render_grid() == avatar)[0])
        if after != before:
            assert (after[1] - before[1], after[0] - before[0]) == (dx, dy)
            moved += 1
    assert moved >= 1

    click = catalog("q011")
    click_module = store.catalog_module(click)
    levels = [_admissible_level(click_module, tier)[0] for tier in (0, 1)]
    _, base_source, meta = store.assemble(click, levels)
    base_cls = store.load_class("q011", base_source)
    skin = skins.make_skin("q111", None, click_module.Game.ACTIONS)
    prefix, source, skin_meta = skins.assemble_skin(click, "q111", base_cls.LEVELS, skin, meta)
    cls = store.load_class(prefix, source)
    baselines = [len(admission.solve_level(cls, i)) for i in range(len(cls.LEVELS))]
    game_id = store.write_game(tmp_path, prefix, source, skin_meta, baselines)
    report = admission.admit(cls, tmp_path / "environment_files", game_id, random_rollouts=40)
    assert report["admitted"], report["issues"]
