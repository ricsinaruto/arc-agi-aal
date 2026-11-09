"""Baseline S: fidelity to open-schema, the CLI on the toy env and our bridge, the game loop."""

import ast
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_qwen_core import Environment

from qwen_baselines.bridge import Bridge
from qwen_baselines.clock import Clock
from qwen_baselines.config import Config
from qwen_baselines.evidence import Archive
from schema_qwen import prompts
from schema_qwen.agi3lib import env as env_module
from schema_qwen.agi3lib.events import EventLog
from schema_qwen.game import SchemaGame, schema_summary
from schema_qwen.profile import S_DEFAULTS

ROOT = Path(__file__).resolve().parents[1]
OPEN_SCHEMA = ROOT / "externals/open-schema/harness"
PACKAGE = ROOT / "src/schema_qwen"
pi_only = pytest.mark.skipif(sys.platform != "linux", reason="Requires pinned Node and Linux sandbox")


def segments(source):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found[node.name] = "".join(lines[node.lineno - 1 : node.end_lineno])
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            found[node.targets[0].id] = "".join(lines[node.lineno - 1 : node.end_lineno])
    return found


@pytest.mark.skipif(not OPEN_SCHEMA.is_dir(), reason="open-schema submodule absent")
def test_library_and_cli_match_open_schema():
    for name in ("__init__", "backtest", "bfs", "events", "model", "render", "score"):
        assert (PACKAGE / "agi3lib" / f"{name}.py").read_bytes() == (
            OPEN_SCHEMA / "agi3lib" / f"{name}.py"
        ).read_bytes(), name
    ours, theirs = segments((PACKAGE / "agi3lib/env.py").read_text()), segments((OPEN_SCHEMA / "agi3lib/env.py").read_text())
    for name in ("Frame", "ToyEnv", "TOY_LEVELS", "TOY_WIN_LEVELS", "RESET"):
        assert ours[name] == theirs[name], name
    ours, theirs = segments((PACKAGE / "arc3.py").read_text()), segments((OPEN_SCHEMA / "arc3.py").read_text())
    for name in (
        "_Tee", "Run", "_require_run", "cmd_observe", "cmd_backtest", "cmd_bfs", "_flags",
        "cmd_commit", "cmd_history", "cmd_score", "cmd_finish", "NOTES_TEMPLATE", "MISPREDICT_MSG",
    ):
        assert ours[name] == theirs[name], name
    upstream_prompt = (OPEN_SCHEMA / "make_eval_run.py").read_text()
    for line in ("finishing every level in as few actions", "One misprediction voids the rest of the plan", "think long, act little"):
        assert line in upstream_prompt and line in prompts.SYSTEM_PROMPT


def cli(workdir, *args, env=None):
    return subprocess.run(
        [sys.executable, str(PACKAGE / "arc3.py"), "--workdir", str(workdir), *args],
        capture_output=True, text=True, env={**os.environ, **(env or {})},
    )


def test_cli_plays_the_toy_game_and_records_schema_events(tmp_path):
    run = tmp_path / "run"
    started = cli(run, "init", "--env", "toy", "--agent", "test", "--max-actions", "50")
    assert started.returncode == 0 and "run started" in started.stdout
    observed = cli(run, "observe")
    assert "State: NOT_FINISHED | level 0/3" in observed.stdout and "Your notes" in observed.stdout
    assert observed.stdout.count("\n") > 64  # the hex grid
    assert cli(run, "backtest").returncode != 0  # no model yet
    committed = cli(run, "commit", "--actions", '[{"action": 1}, {"action": 6, "x": 3, "y": 4}]', "--reason", "probe")
    assert committed.returncode == 0 and "Committed 2 action(s)" in committed.stdout
    (run / "world_model.py").write_text(
        "def step(grid, action, x=None, y=None):\n    return grid, {}\n"
    )
    backtest = cli(run, "backtest")
    assert backtest.returncode == 1 and "mismatch" in backtest.stdout.lower()
    history = cli(run, "history", "--last", "1")
    assert history.returncode == 0 and "2 transitions total" in history.stdout
    finished = cli(run, "finish")
    assert finished.returncode == 0
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    kinds = [e["kind"] for e in events]
    assert kinds[:2] == ["run_started", "turn_started"]
    assert kinds.count("turn_committed") == 1 and kinds.count("action_taken") == 2
    assert kinds.count("tool_started") == kinds.count("tool_finished") >= 5
    finished = next(e for e in events if e["kind"] == "run_finished")
    assert finished["has_world_model"] and finished["actions"] == 2
    assert (run / "notes.md").read_text().startswith("# Notes")


def test_rpc_env_talks_to_the_bridge_api(monkeypatch):
    calls = []
    grid = [[0] * 64 for _ in range(64)]

    def observe(grid=False):
        calls.append(("observe", grid))
        return {"status": "NOT_FINISHED", "levels_completed": 1, "win_levels": 5, "legal_actions": ["ACTION1", "ACTION6", "RESET"], "grid": grid_, "frames": [grid_, grid_]}

    def step(action, x=None, y=None, grid=False):
        calls.append(("step", action, x, y, grid))
        return {"status": "GAME_OVER", "levels_completed": 1, "win_levels": 5, "legal_actions": ["RESET"], "grid": grid_, "frames": [grid_]}

    grid_ = grid
    monkeypatch.setitem(sys.modules, "arc_api", SimpleNamespace(observe=observe, step=step))
    env = env_module.make_env({"env": "rpc"})
    frame = env.current()
    assert frame.state == "NOT_FINISHED" and frame.level == 1 and frame.legal == [1, 6, 0]
    assert len(frame.frames) == 2 and frame.win_levels == 5
    frame = env.act(6, 3, 4)
    assert frame.state == "GAME_OVER" and calls[-1] == ("step", "ACTION6", 3, 4, True)
    env.act(0)
    assert calls[-1] == ("step", "RESET", None, None, True)
    assert env.state_dict() == {}


class FakeExecutor:
    """Runs the arc3 CLI in-process on the toy env instead of the sandbox."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.commands = []

    async def execute(self, operation, command, timeout=None):
        assert operation == "bash" and command.startswith("python3 -m arc3 --workdir run ")
        self.commands.append(command)
        args = command.split(" ", 5)[5].replace("--env rpc", "--env toy")
        result = subprocess.run(
            [sys.executable, str(PACKAGE / "arc3.py"), "--workdir", str(self.workspace / "run"), *__import__("shlex").split(args)],
            capture_output=True, text=True,
        )
        return {"exitCode": result.returncode, "output": result.stdout + result.stderr}


class FakeAgent:
    def __init__(self, workspace, plans):
        self.executor = FakeExecutor(workspace)
        self.plans = list(plans)
        self.prompts = []
        self.interrupted = False

    async def prompt(self, text):
        self.prompts.append(text)
        plan = self.plans.pop(0) if self.plans else None
        if plan:
            await self.executor.execute("bash", command="python3 -m arc3 --workdir run commit --actions " + __import__("shlex").quote(json.dumps(plan)))
        return {"text": "ok", "turns": 3, "turn_cap_hit": False, "model_error": None}

    async def interrupt(self):
        self.interrupted = True


async def s_game(tmp_path, plans, clear_at=10000, seconds=600):
    clock = Clock(seconds)
    archive = Archive(tmp_path / "game")
    bridge = Bridge(Environment(clear_at), archive, clock, label="S toy", boundaries=False)
    await bridge.initialize()
    clock.release()
    workspace = tmp_path / "game" / "workspace"
    workspace.mkdir(parents=True)
    agent = FakeAgent(workspace, plans)
    return SchemaGame(bridge, agent, Config(**S_DEFAULTS)), agent


async def test_game_loop_prompts_with_observe_output_and_counts_commits(tmp_path):
    game, agent = await s_game(tmp_path, [[{"action": 1}, {"action": 2}], None, [{"action": 6, "x": 1, "y": 2}]])
    # The toy env inside the CLI does not touch the bridge; stop after three turns.
    original = game.budget_left
    game.budget_left = lambda: original() and game.turns < 3
    await game.run()
    assert len(agent.prompts) == 3
    assert agent.prompts[0].startswith("State: NOT_FINISHED") and "Current grid:" in agent.prompts[0]
    assert not agent.prompts[1].startswith(prompts.NO_COMMIT_NOTE)
    assert agent.prompts[2].startswith(prompts.NO_COMMIT_NOTE)  # turn 2 committed nothing
    assert agent.executor.commands[0].startswith("python3 -m arc3 --workdir run init --env rpc --game toy --agent")
    events = [json.loads(p) for k, p in game.bridge.archive.db.execute("SELECT kind,payload FROM events") if k == "schema_turn"]
    assert [e["commits"] for e in events] == [1, 0, 1] and [e["planned_actions"] for e in events] == [2, 0, 1]
    assert [e["no_commit_note"] for e in events] == [False, False, True]
    log = [json.loads(line) for line in (game.workdir / "events.jsonl").read_text().splitlines()]
    assert log[-1]["kind"] == "run_finished" and log[-1]["harness_finished"]
    summary = schema_summary(game.bridge.archive)
    assert summary["turns"] == 3 and summary["commits"] == 2 and summary["total_actions"] == 3
    assert summary["tool_calls"]["commit"] == 2 and summary["tool_calls"]["observe"] == 3
    assert summary["has_world_model"] is False and summary["state"] == "STOPPED"


def test_summary_reads_mispredictions_and_world_model(tmp_path):
    archive = Archive(tmp_path / "game")
    run = tmp_path / "game/workspace/run"
    run.mkdir(parents=True)
    (run / "world_model.py").write_text("def step(g, a, x=None, y=None): return g, {}\n")
    log = EventLog(run / "events.jsonl")
    log.emit("run_started", game_id="x")
    log.emit("turn_started", turn=1, grid=[[0]])
    log.emit("tool_started", turn=1, call_id="a", name="backtest", args={})
    log.emit("turn_committed", turn=1, plan=[[1, None, None]])
    log.emit("action_taken", turn=1, step_index=0, action=1, level_up=True, state="NOT_FINISHED", level=1, grid=[[0]])
    log.emit("model_mispredicted", turn=1, step_index=0, surprise="x", predicted=[[1]])
    log.emit("run_finished", state="WIN", levels=1, win_levels=1, actions=1, transitions=1, has_world_model=True)
    archive.event("schema_turn", turn=1, seconds=2.0, tool_turns=4, turn_cap_hit=True, model_error=None, commits=1, planned_actions=1, actions_taken=1, mispredicted=1, tool_calls=1, no_commit_note=False)
    summary = schema_summary(archive)
    assert summary["mispredicted"] == 1 and summary["has_world_model"] and summary["state"] == "WIN"
    assert summary["completed_actions"] == [1] and summary["turn_cap_hits"] == 1
    assert summary["tool_calls"] == {"backtest": 1}


@pi_only
async def test_schema_pi_end_to_end_runs_the_cli_in_the_sandbox(tmp_path):
    from test_prolong_qwen import scripted_layer

    from schema_qwen.agent import SchemaPi

    clock = Clock(600)
    bridge = Bridge(Environment(), Archive(tmp_path / "game"), clock, label="S pi-game", boundaries=False)
    await bridge.initialize()
    cfg = Config(**S_DEFAULTS)
    requests = []
    script = [
        ("bash", {"command": "python3 -m arc3 --workdir run backtest; python3 -m arc3 --workdir run history"}),
        # The test Environment fills the whole grid with its step count (mod 16).
        (
            "write",
            {
                "path": "run/world_model.py",
                "content": "def step(grid, action, x=None, y=None):\n"
                "    v = (grid[0][0] + 1) % 16\n"
                "    return [[v] * 64 for _ in range(64)], {}\n",
            },
        ),
        ("bash", {"command": "python3 -m arc3 --workdir run commit --actions '[{\"action\": 1}]' --reason probe"}),
        ("text", "Committed one probe."),
        ("bash", {"command": "python3 -m arc3 --workdir run backtest"}),
        ("text", "Backtest checked."),
    ]
    layer = scripted_layer(bridge, script, requests, cfg)
    agent = SchemaPi(bridge, layer, pi_dir=os.environ.get("ARC_PI_DIR", "/app/pi"))
    await agent.start()
    clock.release()
    game = SchemaGame(bridge, agent, cfg)
    original = game.budget_left
    game.budget_left = lambda: original() and game.turns < 2
    try:
        async with asyncio.timeout(180):
            await game.run()
    finally:
        await agent.close()
    assert bridge.actions == 1
    events = [json.loads(line) for line in (game.workdir / "events.jsonl").read_text().splitlines()]
    kinds = [e["kind"] for e in events]
    assert kinds.count("turn_committed") == 1 and kinds.count("action_taken") == 1
    taken = next(e for e in events if e["kind"] == "action_taken")
    assert taken["action"] == 1 and len(taken["grid"]) == 64
    # Pi appends our prompt to its own system prompt (appendSystemPrompt).
    assert "# Your task: play one ARC-AGI-3 game" in requests[0]["messages"][0]["content"]
    assert "State: NOT_FINISHED" in json.dumps(requests[0]["messages"][1])
    outputs = [bridge.archive.transcript(i)["output"] for i in range(bridge.archive.count("calls"))]
    joined = "".join(outputs)
    assert "no world_model" in joined, joined[:3000]
    assert "Committed 1 action(s)" in joined, joined[-3000:]
    assert "safe to plan with run_bfs" in joined  # the backtest after the model was written
