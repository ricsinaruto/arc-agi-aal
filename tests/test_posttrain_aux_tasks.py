"""Auxiliary post-training tasks (predict / reach / transition): rendering, patches, the three
verifiers (reference accepted, wrong answers rejected, sandbox limits) on a real DEV game."""

import json
import random
import time

import numpy as np
import pytest

from inference.utils.grid_utils import format_grid_ascii
from posttrain import aux_tasks as aux
from posttrain.split import is_dev, is_generated

LIBRARY = aux.GameLibrary()
HAS_M0R0 = any(g.startswith("m0r0-") for g in LIBRARY.dirs())
needs_m0r0 = pytest.mark.skipif(not HAS_M0R0, reason="environment_files/m0r0 not present")


def fenced(body, lang="text"):
    return f"reasoning first\n```{lang}\n{body}\n```"


def test_board_text_is_the_harness_rendering_and_round_trips():
    board = np.random.default_rng(0).integers(0, 16, size=(64, 64), dtype=np.uint8)
    text = aux.board_text(board)
    assert text == format_grid_ascii(board.tolist())
    assert np.array_equal(aux.parse_board(text), board)
    with pytest.raises(ValueError):
        aux.parse_board(text[:-1])


def test_action_names_follow_the_harness():
    assert aux.action_label(("ACTION1", None, None)) == "UP"
    assert aux.action_label(("ACTION5", None, None)) == "SPACE"
    assert aux.action_label(("ACTION7", None, None)) == "ACTION7"
    assert aux.action_label(("ACTION6", 7, 4)) == "MOUSE(row=4, col=7)"
    assert aux.action_json(("ACTION6", 7, 4)) == {"action": "MOUSE", "row": 4, "col": 7}
    assert aux.action_dict(("ACTION2", None, None)) == {"action": "DOWN"}
    parsed = aux.parse_action_list('["UP", {"action": "MOUSE", "row": 4, "col": 7}, "ACTION7"]')
    assert parsed == [("ACTION1", None, None), ("ACTION6", 7, 4), ("ACTION7", None, None)]
    assert aux.parse_action_list("UP, MOUSE(row=4, col=7)\nLEFT") == [
        ("ACTION1", None, None), ("ACTION6", 7, 4), ("ACTION3", None, None)]
    assert aux.parse_action_list('[{"action": "MOUSE", "x": 4, "y": 7}]') is None  # legacy x/y rejected
    assert aux.parse_action_list('["RESET"]') is None
    assert aux.parse_action_list("UP, JUMP") is None
    assert aux.parse_action_list('[{"action": "MOUSE", "row": 64, "col": 0}]') is None


def test_patches_round_trip_and_group_nearby_changes():
    rng = np.random.default_rng(1)
    before = rng.integers(0, 16, size=(64, 64), dtype=np.uint8)
    after = before.copy()
    after[10:13, 5:9] = 3
    after[63, 0] = (before[63, 0] + 1) % 16
    after[40, 40] = (before[40, 40] + 1) % 16
    text = aux.patch_text(before, after)
    assert text.count("@ row=") == 3
    assert np.array_equal(aux.apply_patch(before, aux.parse_patch(text)), after)
    assert aux.patch_text(before, before) == "NO CHANGE"
    assert aux.parse_patch("NO CHANGE") == []
    assert aux.parse_patch("@ row=62 col=0\nWWW\nWWW") is not None
    assert aux.apply_patch(before, aux.parse_patch("@ row=63 col=0\nWWW\nWWW")) is None  # off the board
    assert aux.parse_patch("@ row=1 col=1\nWW?W") is None
    assert not aux.substantive(before, np.where(np.arange(64) == 63, 0, before))  # last column only
    assert aux.substantive(before, after)


@pytest.fixture(scope="module")
def m0r0_tasks():
    if not HAS_M0R0:
        pytest.skip("environment_files/m0r0 not present")
    game_id = LIBRARY.resolve("m0r0")
    started = time.monotonic()
    tasks, stats = aux.generate(game_id, LIBRARY.factory(game_id), 3, seed=7, library=LIBRARY,
                                maker_kwargs={"predict": {"patch_fraction": 0.5}})
    return tasks, stats, time.monotonic() - started


def by_family(tasks, family):
    rows = [t for t in tasks if t["family"] == family]
    assert rows, family
    return rows


@needs_m0r0
def test_generation_makes_self_checked_tasks_of_every_family(m0r0_tasks):
    tasks, stats, _seconds = m0r0_tasks
    assert {f: stats[f]["made"] for f in aux.FAMILIES} == {"predict": 3, "reach": 3, "transition": 3}
    assert stats["pool"]["clone_faithful"]
    assert len({t["prompt"] for t in tasks}) == len(tasks)
    for task in tasks:
        assert set(task) >= {"id", "family", "game", "level", "prompt", "answer", "verifier"}
        assert is_dev(task["game"]) and task["id"].startswith(task["game"])
        assert 1 <= task["level"] <= 6
        assert aux.PREAMBLE in task["prompt"]
        assert any(len(line) == 64 and set(line) <= aux.PALETTE for line in task["prompt"].split("\n"))
        json.dumps(task)  # records are plain JSON
        assert aux.self_check(task, library=LIBRARY)


@needs_m0r0
def test_generation_is_deterministic_for_a_seed(m0r0_tasks):
    tasks, _, _ = m0r0_tasks
    game_id = LIBRARY.resolve("m0r0")
    again, _ = aux.generate(game_id, LIBRARY.factory(game_id), 3, seed=7, library=LIBRARY, check=False,
                            maker_kwargs={"predict": {"patch_fraction": 0.5}})
    assert [t["id"] for t in again] == [t["id"] for t in tasks]


@needs_m0r0
def test_predict_verifier_accepts_either_answer_form_and_rejects_wrong_boards(m0r0_tasks):
    for task in by_family(m0r0_tasks[0], "predict"):
        ver = task["verifier"]
        start, end = aux.parse_board(ver["start"]), aux.parse_board(ver["expected"])
        assert aux.verify(task, task["answer"])
        assert aux.verify(task, fenced(ver["expected"]))
        assert aux.verify(task, fenced(aux.patch_text(start, end)))
        if ver["start"] != ver["expected"]:
            assert not aux.verify(task, fenced(ver["start"]))
            assert not aux.verify(task, fenced("NO CHANGE"))
        wrong = end.copy()
        wrong[32, 32] = (wrong[32, 32] + 1) % 16
        assert not aux.verify(task, fenced(aux.board_text(wrong)))
        assert not aux.verify(task, "I think the player moves up.")
        assert not aux.verify(task, None)


@needs_m0r0
def test_reach_verifier_replays_the_level_in_the_engine(m0r0_tasks):
    for task in by_family(m0r0_tasks[0], "reach"):
        ver = task["verifier"]
        answer = json.loads(aux.extract_answer(task["answer"], ("json",)))
        assert len(answer) == ver["max_actions"] <= 6
        assert aux.verify(task, task["answer"], library=LIBRARY)
        labels = ", ".join(aux.action_label(aux.parse_action(a)) for a in answer)
        assert aux.verify(task, labels, library=LIBRARY)  # plain labels, no fence
        assert not aux.verify(task, fenced("[]", "json"), library=LIBRARY)
        assert not aux.verify(task, fenced(json.dumps(answer + answer[-1:]), "json"), library=LIBRARY)  # too long
        if len(answer) > 1:
            assert not aux.verify(task, fenced(json.dumps(answer[:-1]), "json"), library=LIBRARY)
        # the verifier never trusts the stored start: a different target is rejected
        other = dict(task, verifier=dict(ver, target=ver["target"].replace("\n", "", 1) + "W"))
        assert not aux.verify(other, task["answer"], library=LIBRARY)


@needs_m0r0
def test_transition_verifier_runs_programs_on_held_out_pairs(m0r0_tasks):
    for task in by_family(m0r0_tasks[0], "transition"):
        ver = task["verifier"]
        assert task["answer"] is None
        assert 3 <= task["meta"]["n_shown"] <= 6 and len(ver["cases"]) == 4
        assert all(c["board"] not in task["prompt"] for c in ver["cases"])  # held out
        assert aux.verify(task, fenced(aux.oracle_program(task), "python"))
        assert not aux.verify(task, fenced(aux.IDENTITY_PROGRAM, "python"))
        assert not aux.verify(task, fenced("def step(board, action):\n    return board[::-1]\n", "python"))
        assert not aux.verify(task, fenced("def step(board, action) return board", "python"))


@needs_m0r0
def test_self_check_replays_every_stated_board_from_a_fresh_game(m0r0_tasks):
    for task in m0r0_tasks[0]:
        if task["family"] == "reach":
            continue
        assert aux.replay_check(task, LIBRARY)
        tampered = json.loads(json.dumps(task))
        tampered["meta"]["replay"]["level_index"] = (task["level"] % 6)  # another level of m0r0
        assert not aux.self_check(tampered, library=LIBRARY)


def test_unfaithful_deep_copies_switch_the_pool_to_replay():
    """tn36's deep copies diverge from the original after a few clicks on levels 2-5."""
    if not any(g.startswith("tn36-") for g in LIBRARY.dirs()):
        pytest.skip("environment_files/tn36 not present")
    factory = LIBRARY.factory("tn36")
    assert not aux.clone_is_faithful(factory, aux.Sim.start(factory, 0).n_levels, random.Random(0))
    assert aux.clone_is_faithful(LIBRARY.factory("m0r0"), 6, random.Random(0)) if HAS_M0R0 else True
    tasks, stats = aux.generate(LIBRARY.resolve("tn36"), factory, 2, seed=0, library=LIBRARY, families=("reach",))
    assert not stats["pool"]["clone_faithful"] and stats["reach"]["made"] == 2
    assert stats["reach"]["failed_self_check"] == 0


CASES = [{"board": aux.board_text(np.zeros((64, 64), dtype=np.uint8)), "action": {"action": "UP"}}]


def test_sandbox_enforces_timeout_imports_and_builtins():
    started = time.monotonic()
    assert aux.run_step_program("def step(board, action):\n    while True:\n        pass\n", CASES,
                                timeout=2) == {"timeout": True}
    assert time.monotonic() - started < 10
    denied = aux.run_step_program("import os\ndef step(board, action):\n    return board\n", CASES)
    assert "not allowed" in denied["error"]
    lazy = aux.run_step_program("def step(board, action):\n    import subprocess\n    return board\n", CASES)
    assert "not allowed" in lazy["results"][0]["error"]
    dunder = aux.run_step_program("def step(board, action):\n    __import__('socket')\n    return board\n", CASES)
    assert "not allowed" in dunder["results"][0]["error"]
    opened = aux.run_step_program("def step(board, action):\n    open('x', 'w')\n    return board\n", CASES)
    assert "NameError" in opened["results"][0]["error"]
    allowed = aux.run_step_program(
        "import numpy as np\nimport collections, re\n"
        "def step(board, action):\n    g = np.array([list(r) for r in board])\n"
        "    g[0, 0] = 'Y' if action['action'] == 'UP' else 'R'\n    return [''.join(r) for r in g]\n", CASES)
    assert allowed["results"][0].startswith("Y")
    assert aux.run_step_program("x = 1\n", CASES)["error"].startswith("KeyError")


def test_builder_refuses_holdout_games(tmp_path):
    from importlib import util
    from pathlib import Path

    if not any(g.startswith("tr87-") for g in LIBRARY.dirs()):
        pytest.skip("environment_files/tr87 not present")
    path = Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "build_aux_tasks.py"
    spec = util.spec_from_file_location("build_aux_tasks", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="non-dev"):
        module.main(["--games", "tr87", "--per-family", "1", "--out", str(tmp_path)])
    assert not (tmp_path / "tasks.jsonl").exists()


def _aux_builder():
    from importlib import util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "build_aux_tasks.py"
    spec = util.spec_from_file_location("build_aux_tasks", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_keeps_held_out_generated_mechanics_in_their_own_split(tmp_path):
    module = _aux_builder()
    library = aux.GameLibrary(env_dirs=(module.GENERATED_ENV_DIR,))
    if not any(g.startswith("q009-") for g in library.dirs()):
        pytest.skip("no generated games in data/posttrain/games/environment_files")
    with pytest.raises(ValueError, match="q009"):
        module.main(["--generated", "--games", "q009", "--per-family", "1", "--out", str(tmp_path / "train")])
    assert not (tmp_path / "train" / "tasks.jsonl").exists()
    module.main(["--eval", "--games", "q009", "--per-family", "1", "--families", "predict", "reach",
                 "--workers", "1", "--out", str(tmp_path / "eval")])
    tasks = [json.loads(line) for line in open(tmp_path / "eval" / "tasks.jsonl")]
    assert tasks and all(t["split"] == "eval" and t["no_reasoning"] and t["game"].startswith("q009-") for t in tasks)
    stats = json.loads((tmp_path / "eval" / "stats.json").read_text())
    assert stats["config"]["split"] == "eval" and stats["config"]["no_reasoning"]
    with pytest.raises(ValueError, match="GENERATED_HOLDOUT"):
        module.main(["--eval", "--games", "q001", "--per-family", "1", "--out", str(tmp_path / "bad")])


def test_generated_games_load_from_their_environments_dir():
    library = aux.GameLibrary(env_dirs=aux.DEFAULT_ENV_DIRS[1:])
    games = sorted(g for g in library.dirs() if is_generated(g))
    if not games:
        pytest.skip("no generated games in data/posttrain/games/environment_files")
    game_id = games[0]
    tasks, stats = aux.generate(game_id, library.factory(game_id), 1, seed=0, library=library)
    assert {t["family"] for t in tasks} == set(aux.FAMILIES), stats
    for task in tasks:
        assert task["game"] == game_id and aux.self_check(task, library=library)


def test_state_pool_levels_start_like_real_play():
    """``set_level`` boards equal the level start reached by playing (the replay check on 9 DEV games
    used recorded logs; here: level 2 of m0r0 through its own reset path is stable and replayable)."""
    if not HAS_M0R0:
        pytest.skip("environment_files/m0r0 not present")
    factory = LIBRARY.factory("m0r0")
    pool = aux.StatePool("m0r0", factory, random.Random(3), walks=2, walk_max=5)
    states = pool.states(1)
    assert np.array_equal(states[0].board, aux.Sim.start(factory, 1).board)
    for state in states:
        assert np.array_equal(pool.sim(state).board, state.board)
