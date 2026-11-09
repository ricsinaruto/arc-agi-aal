"""Single-level episode games (posttrain.games.episodes) and rejection sampling (posttrain.rejection)."""

import json

import pytest

from posttrain import rejection, split
from posttrain.games import admission, episodes, store


def spec(base, tier, seed, skinned=False, prefix="q200", batch=1):
    return {"base": base, "tier": tier, "seed": seed, "skinned": skinned, "batch": batch, "prefix": prefix}


def test_episode_generation_is_deterministic_per_seed(tmp_path):
    first, key = episodes.build_episode(spec("q003", 2, 0), tmp_path / "a", random_rollouts=40)
    again, same_key = episodes.build_episode(spec("q003", 2, 0), tmp_path / "b", random_rollouts=40)
    other, other_key = episodes.build_episode(spec("q003", 2, 1), tmp_path / "c", random_rollouts=40)
    assert first["admitted"] and again["admitted"] and other["admitted"]
    assert first["game_id"] == again["game_id"] and key == same_key
    assert other_key != key and other["game_id"] != first["game_id"]
    # the same seed never repeats a level already seen in its group
    retry, retry_key = episodes.build_episode(spec("q003", 2, 0), tmp_path / "d", seen={key}, random_rollouts=40)
    assert retry["admitted"] and retry_key != key


def test_held_out_mechanics_are_refused():
    for base in split.GENERATED_HOLDOUT_MECHANICS:
        with pytest.raises(ValueError, match="held-out"):
            episodes.check_mechanic(base)
        with pytest.raises(ValueError, match="held-out"):
            episodes.plan([base])
        with pytest.raises(ValueError, match="held-out"):
            episodes.build_episode(spec(base, 1, 0), "/nonexistent")
    mechanics = episodes.training_mechanics()
    assert not set(mechanics) & set(split.GENERATED_HOLDOUT) and "q003" in mechanics and len(mechanics) == 15


@pytest.mark.parametrize("base,tier,skinned", [("q003", 1, False), ("q007", 2, True), ("q012", 3, True)])
def test_single_level_episode_wins_after_the_solver_path(tmp_path, base, tier, skinned):
    report, _ = episodes.build_episode(spec(base, tier, 1, skinned=skinned, prefix="q321"), tmp_path,
                                       random_rollouts=40)
    assert report["admitted"], report
    notes = json.loads((tmp_path / "notes" / "q321.json").read_text())
    assert notes["episode_of"] == base and notes["tier"] == tier and notes["seed"] == 1
    assert notes["skin_of"] == (base if skinned else None) and notes["game_id"] == report["game_id"]
    metadata = json.loads(next((tmp_path / "environment_files" / "q321").glob("*/metadata.json")).read_text())
    assert metadata["baseline_actions"] == [len(notes["solution"][0])] == report["baseline_actions"]
    assert metadata["baseline_actions"][0] >= 4
    cls = store.load_class("q321", next((tmp_path / "environment_files" / "q321").glob("*/q321.py")).read_text())
    assert len(cls.LEVELS) == 1
    path = [(a, tuple(c) if c else None) for a, c in notes["solution"][0]]
    result = admission.replay(tmp_path / "environment_files", report["game_id"],
                              [admission.engine_actions(cls, 0, path)])
    assert result["state"] == "WIN"
    assert [m[2] for m in result["marks"]] == [0] * (len(path) - 1) + [1]
    assert episodes.environments_dir_for(report["game_id"], [tmp_path]) == tmp_path / "environment_files"
    assert set(episodes.load_notes([tmp_path])) == {"q321"}


def test_plan_is_balanced_and_prefixes_avoid_existing_games():
    mechanics = ["q001", "q003", "q004"]
    specs = episodes.plan(mechanics, per_mechanic=8, batches=4)
    assert len(specs) == 24
    for batch in (1, 2, 3, 4):
        rows = [s for s in specs if s["batch"] == batch]
        assert sorted(s["base"] for s in rows) == sorted(mechanics * 2)
    for base in mechanics:
        tiers = sorted(s["tier"] for s in specs if s["base"] == base)
        assert set(tiers) == {1, 2, 3} and max(tiers.count(t) for t in (1, 2, 3)) <= 3
    assert all(s["skinned"] == (s["seed"] >= 1) for s in specs if s["base"] == "q001")  # fixed levels
    taken = episodes.taken_prefixes() | {"q201"}
    assert {"q001", "q020", "q101", "q120"} <= taken
    episodes.assign_prefixes(specs, taken)
    prefixes = [s["prefix"] for s in specs]
    assert prefixes[0] == "q200" and "q201" not in prefixes and len(set(prefixes)) == 24
    assert all(split.is_generated(p + "-00000000") for p in prefixes)


def notes_fixture():
    def row(prefix, base, tier, skin_of=None):
        return {"game_id": f"{prefix}-aaaaaaaa", "episode_of": base, "tier": tier, "seed": 0, "skin_of": skin_of}

    return {"q200": row("q200", "q003", 1), "q201": row("q201", "q003", 1, skin_of="q003"),
            "q202": row("q202", "q003", 1), "q203": row("q203", "q003", 2), "q204": row("q204", "q005", 1),
            "q205": row("q205", "q009", 1)}  # q205: an episode of a held-out mechanic (must never happen)


def segment(sid, prefix, tokens, actions, verified=True, version="aaaaaaaa", level=1):
    return {"id": sid, "game_id": f"{prefix}-{version}", "level": level, "verified": verified,
            "completion_tokens": tokens, "action_count": actions, "baseline_actions": 5}


def test_rejection_keeps_the_best_k_per_group_and_drops_the_expensive_tail():
    segments = [
        segment("a", "q200", 4000, 9), segment("b", "q201", 4000, 7),  # tie on tokens: fewer actions first
        segment("c", "q202", 3000, 12), segment("d", "q200", 30000, 6),  # d > 2x median: dropped
        segment("e", "q202", 5000, 5),
        segment("f", "q203", 9000, 20),  # a group of one
        segment("g", "q204", 1000, 5, verified=False),  # unverified
        segment("h", "q204", 1000, 5, version="bbbbbbbb"),  # another version than the notes
        segment("i", "q205", 1000, 5),  # held-out mechanic
        segment("j", "q999", 1000, 5),  # not an episode
    ]
    selected, stats = rejection.select(segments, notes_fixture(), k=2)
    assert [s["id"] for s in selected] == ["c", "b", "f"]
    assert [s["rejection"]["rank"] for s in selected] == [1, 2, 1]
    group = stats["groups"]["q003/tier-1"]
    assert group["clears"] == 5 and group["median_tokens"] == 4000 and group["over_median_multiple"] == 1
    assert group["beyond_k"] == 2 and group["kept"] == ["c", "b"] and group["episodes"] == 3
    totals = stats["totals"]
    assert totals["episode_clears"] == 6 and totals["selected"] == 3
    assert totals["completion_tokens_all_clears"] == 55000 and totals["completion_tokens_selected"] == 16000
    assert totals["completion_tokens_saved"] == 39000
    assert stats["groups_without_clears"] == ["q005/tier-1"]
    assert stats["skipped"] == {"episode version differs from its notes": 1, "not an episode game": 1, "unverified": 1}
    check = stats["held_out_check"]
    assert not check["passed"] and [r["id"] for r in check["rejected_segments"]] == ["i"]
    assert check["held_out_episodes_in_notes"] == ["q205"]
    selected_k1, _ = rejection.select(segments, notes_fixture(), k=1, max_median_multiple=1.2)
    assert [s["id"] for s in selected_k1] == ["c", "f"]


def test_rejection_passes_the_held_out_check_on_training_episodes(tmp_path):
    notes = {p: n for p, n in notes_fixture().items() if p != "q205"}
    selected, stats = rejection.select([segment("a", "q200", 100, 5)], notes)
    assert stats["held_out_check"]["passed"] and len(selected) == 1
    rejection.write(tmp_path, selected, stats)
    assert json.loads((tmp_path / "stats.json").read_text())["totals"]["selected"] == 1
    assert len((tmp_path / "selected.jsonl").read_text().splitlines()) == 1
