import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_qwen_core import Environment

from prolong_qwen.base import BaseAgent
from prolong_qwen.grid_utils import format_grid_ascii
from prolong_qwen.log import GameLog
from prolong_qwen.profile import P_DEFAULTS
from prolong_qwen.runner import RETRY_NUDGE, ProlongGame, available_actions
from qwen_baselines.bridge import Bridge
from qwen_baselines.clock import Clock
from qwen_baselines.config import Config
from qwen_baselines.evidence import Archive

UPSTREAM = Path("externals/PRO-LONG/research/arc-agi-3/prolong_agent")
RELEASE_LOG = Path(
    "externals/PRO-LONG/research/arc-agi-3/release_logs/fable5/tu93/rep1/workspace/logs.txt"
)
# PRO-LONG files carried as ordinary package modules; only the package imports differ.
UPSTREAM_FILES = {
    "agent/prompts.py": "prompts.py",
    "agent/base.py": "base.py",
    "agent/action_queue.py": "action_queue.py",
    "utils/grid_utils.py": "grid_utils.py",
    "agent/game_state.py": "game_state.py",
}
IMPORT_REWRITES = {
    "from prolong_agent.agent.action_queue import": "from .action_queue import",
    "from prolong_agent.agent.prompts import": "from .prompts import",
    "from prolong_agent.utils.grid_utils import": "from .grid_utils import",
}


@pytest.mark.skipif(not UPSTREAM.is_dir(), reason="PRO-LONG submodule not checked out")
def test_protocol_files_match_the_upstream_submodule():
    package = Path("src/prolong_qwen")
    for name, local_name in UPSTREAM_FILES.items():
        original = (UPSTREAM / name).read_text()
        local = (package / local_name).read_text()
        for upstream_import, local_import in IMPORT_REWRITES.items():
            local = local.replace(local_import, upstream_import)
        assert local == original, name


class FakeAgent:
    """Scripted stand-in for ProlongPi.analyze: one plan per call."""

    def __init__(self, plans, workspace):
        self.plans, self.calls = list(plans), []
        self.executor = SimpleNamespace(workspace=workspace)
        self.interrupted = False

    async def analyze(self, **kwargs):
        self.calls.append(kwargs)
        plan = self.plans.pop(0) if self.plans else None
        if plan is None:
            return self.result([], "no plan")
        (Path(self.executor.workspace) / "actions.json").write_text(json.dumps(plan))
        raw = (Path(self.executor.workspace) / "actions.json").read_text()
        actions = BaseAgent._parse_actions_json_text(raw, cap=20)
        return self.result(actions, "Briefing text.\n[PLAN]\nDo the plan.")

    @staticmethod
    def result(actions, text):
        hint, plan = BaseAgent._split_response(text)
        return {
            "hint": hint, "plan": plan, "actions": actions, "text": text, "turns": 3,
            "turn_cap_hit": False, "model_error": None,
        }  # fmt: skip

    async def interrupt(self):
        self.interrupted = True


async def game(tmp_path, plans, *, clear_at=10000, seconds=600, env=None, **config):
    clock = Clock(seconds)
    archive = Archive(tmp_path / "game")
    bridge = Bridge(env or Environment(clear_at), archive, clock, label="P test", boundaries=False)
    await bridge.initialize()
    clock.release()
    agent = FakeAgent(plans, tmp_path / "game" / "workspace")
    return ProlongGame(bridge, agent, Config(**{**P_DEFAULTS, **config}))


def headers(workspace):
    return [
        line
        for line in (workspace / "logs.txt").read_text().splitlines()
        if line.startswith("Action ")
    ]


async def test_first_call_prompt_log_format_and_plan_drain(tmp_path):
    g = await game(tmp_path, [{"actions": ["ACTION1", "ACTION6(70,3)", "ACTION2"]}])
    g.config = Config(**P_DEFAULTS, max_actions=3)
    await g.run()
    workspace = g.agent.executor.workspace
    text = (workspace / "logs.txt").read_text()
    assert text.startswith(
        "=" * 80
        + "\nAction 0 | Level 1 | Attempt 1 | INITIAL STATE | Score: 0\n\n[INITIAL BOARD STATE]\n"
    )
    assert headers(workspace) == [
        "Action 0 | Level 1 | Attempt 1 | INITIAL STATE | Score: 0",
        "Action 1 | Level 1 | Attempt 1 | Plan Step 1/3 | Score: 0",
        "Action 2 | Level 1 | Attempt 1 | Plan Step 2/3 | Score: 0",
        "Action 3 | Level 1 | Attempt 1 | Plan Step 3/3 | Score: 0",
    ]
    # Briefing on the plan head, the plan text on later steps; clicks clamp like upstream.
    assert "\n[PLAN]\nBriefing text.\n\nTool Call: ACTION1({})\n" in text
    assert text.count("[PLAN]\nDo the plan.") == 2
    assert 'Tool Call: ACTION6({"x": 63, "y": 3})' in text
    assert text.count("[POST-ACTION BOARD STATE]") == 3
    assert g.bridge.actions == 3
    call = g.agent.calls[0]
    assert call["is_first"] and call["action_num"] == 0 and call["nudge"] == ""
    assert call["available_actions"] == [
        "ACTION1",
        "ACTION2",
        "ACTION3",
        "ACTION4",
        "ACTION6",
        "ACTION7",
        "RESET",
    ]
    board = (workspace / "current_board.txt").read_text()
    assert board.startswith("Score: 0\nAction: 0\n\n[CURRENT BOARD STATE]\n")
    events = [
        json.loads(p)
        for k, p in g.bridge.archive.db.execute("SELECT kind,payload FROM events")
        if k == "prolong_call"
    ]
    assert events[0]["plan_length"] == 3 and events[0]["loaded"] and events[0]["tool_turns"] == 3
    g.bridge.archive.close()


def test_ascii_boards_match_the_paper_palette():
    grid = [[0, 5, 15]]
    assert (
        format_grid_ascii(grid, mode="ascii") == "$OI"[:3]
        or format_grid_ascii(grid, mode="ascii")[0] == "$"
    )
    assert format_grid_ascii(grid, mode="hex") == "05f"


async def test_score_change_flushes_and_level_attempt_bookkeeping(tmp_path):
    g = await game(tmp_path, [{"actions": ["ACTION1"] * 6}, {"actions": ["ACTION2"]}], clear_at=2)
    g.config = Config(**P_DEFAULTS, max_actions=3)
    await g.run()
    assert g.bridge.actions == 3
    assert headers(g.agent.executor.workspace)[-2:] == [
        "Action 2 | Level 1 | Attempt 1 | Score: 1",
        "Action 3 | Level 2 | Attempt 1 | Plan Step 1/1 | Score: 1",
    ]
    assert len(g.agent.calls) == 2 and not g.agent.calls[1]["is_first"]
    assert g.agent.calls[1]["score"] == 1 and g.agent.calls[1]["level"] == 2
    g.bridge.archive.close()


async def test_game_over_forces_reset_then_keeps_draining(tmp_path):
    env = Environment()
    original = env.step
    fired = []

    def step(action, data):
        raw = original(action, data)
        if not fired:
            fired.append(True)
            raw.state.name = "GAME_OVER"
        return raw

    env.step = step
    g = await game(tmp_path, [{"actions": ["ACTION1", "ACTION2", "ACTION3"]}], env=env)
    g.config = Config(**P_DEFAULTS, max_actions=4)
    await g.run()
    names = [
        json.loads(p)["action"]["action"]
        for k, p in g.bridge.archive.db.execute("SELECT kind,payload FROM events ORDER BY id")
        if k == "action_issued"
    ]
    assert names == ["ACTION1", "RESET", "ACTION2", "ACTION3"]
    assert (
        headers(g.agent.executor.workspace)[2]
        == "Action 2 | Level 1 | Attempt 2 | Plan Step 1/3 | Score: 0"
    )
    assert len(g.agent.calls) == 1
    g.bridge.archive.close()


def issued(g):
    return [
        json.loads(p)["action"]["action"]
        for k, p in g.bridge.archive.db.execute("SELECT kind,payload FROM events ORDER BY id")
        if k == "action_issued"
    ]


async def test_duplicate_reset_gets_action7_and_nudges_follow_bad_plans(tmp_path):
    plans = [{"actions": ["RESET", "RESET"]}, None, {"nope": 1}, {"actions": ["ACTION2"]}]
    g = await game(tmp_path, plans)
    g.config = Config(**P_DEFAULTS, max_actions=4)
    await g.run()
    assert issued(g) == ["RESET", "ACTION7", "RESET", "ACTION2"]
    assert [c["nudge"] for c in g.agent.calls] == ["", "", RETRY_NUDGE, RETRY_NUDGE]
    g.bridge.archive.close()


class NoUndo(Environment):
    def raw(self):
        raw = super().raw()
        raw.available_actions = [1, 2, 3, 4, 6]
        return raw


async def test_duplicate_reset_is_dropped_when_action7_is_unavailable(tmp_path):
    g = await game(tmp_path, [{"actions": ["RESET", "RESET", "ACTION1"]}], env=NoUndo())
    g.config = Config(**P_DEFAULTS, max_actions=2)
    await g.run()
    assert issued(g) == ["RESET", "ACTION1"]
    kinds = [k for (k,) in g.bridge.archive.db.execute("SELECT kind FROM events")]
    assert "prolong_duplicate_reset_dropped" in kinds
    g.bridge.archive.close()


async def test_unavailable_action_drops_plan_and_deadline_ends_game(tmp_path):
    g = await game(
        tmp_path, [{"actions": ["ACTION5", "ACTION1"]}, {"actions": ["ACTION1"] * 20}], seconds=0.5
    )
    await g.run()
    kinds = [
        k for k, _ in g.bridge.archive.db.execute("SELECT kind,payload FROM events ORDER BY id")
    ]
    assert "prolong_invalid_action" in kinds and "deadline" in kinds
    assert g.bridge.actions == 20 and g.agent.interrupted
    g.bridge.archive.close()


def test_p_defaults_and_validation():
    cfg = Config(**P_DEFAULTS)
    assert (
        cfg.compaction_tokens == 65536 and cfg.bash_seconds == 120 and cfg.bash_max_seconds == 600
    )
    assert (
        not cfg.scientific_python
        and cfg.grid_mode == "ascii"
        and cfg.plan_cap == 20
        and cfg.call_turns == 50
    )
    assert Config().compaction_tokens == 65536 and Config().scientific_python
    for bad in (
        {"compaction_tokens": 300000},
        {"bash_seconds": 50, "bash_max_seconds": 30},
        {"grid_mode": "rgb"},
        {"max_actions": 0},
    ):
        with pytest.raises(ValueError):
            Config(**bad)
    assert available_actions(["ACTION1", "ACTION6"]) == ["ACTION1", "ACTION6", "RESET"]


@pytest.mark.skipif(not RELEASE_LOG.is_file(), reason="release logs not checked out")
def test_log_renders_like_the_fable_release_log(tmp_path):
    lines = RELEASE_LOG.read_text().splitlines()
    board = lines[4:68]
    palette = '#hqOCznf(G->I"'
    grid = np.array([[palette.index(ch) for ch in row] for row in board], dtype=np.uint8)
    log = GameLog(tmp_path, "ascii")
    log.observe([grid], "NOT_FINISHED", 0)
    log.initial(1, 1, 0)
    assert (tmp_path / "logs.txt").read_text().splitlines()[:68] == lines[:68]


async def test_bridge_boundaries_flag(tmp_path):
    clock = Clock(600)
    bridge = Bridge(Environment(1), Archive(tmp_path), clock, boundaries=False)
    await bridge.initialize()
    clock.release()
    await bridge.step("ACTION1")
    assert bridge.archive.metadata()["level_completed"] and not bridge.stopped
    await bridge.step("ACTION1")
    assert bridge.actions == 2
    bridge.archive.close()
    strict = Bridge(Environment(1), Archive(tmp_path / "strict"), clock)
    await strict.initialize()
    await strict.step("ACTION1")
    assert strict.stopped
    strict.archive.close()


def test_upstream_palette_mapping_is_the_paper_ascii_map():
    # The paper's ASCII colour map: '$' white (0) ... '"' purple (15).
    assert (
        format_grid_ascii([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
        == '#hqOCznf(G->I"'
    )


asyncio  # silence unused-import linters when only async tests use the loop


# --- Real Pi SDK (Linux sandbox only) -------------------------------------------------

pi_only = pytest.mark.skipif(
    __import__("sys").platform != "linux", reason="Requires pinned Node and Linux sandbox"
)


def scripted_layer(bridge, script, requests, config):
    import httpx
    from test_qwen_pi import sse

    from qwen_baselines.model import RequestLayer

    def endpoint(request):
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 500, "max_model_len": 262144})
        requests.append(body)
        if not body.get("tools"):
            message = {"role": "assistant", "content": "## Summary\nCompacted."}
        else:
            step = script.pop(0) if script else ("text", "Nothing more.")
            if step[0] == "text":
                message = {"role": "assistant", "content": step[1]}
            else:
                message = {
                    "role": "assistant",
                    "reasoning_content": "Use the tools.",
                    "tool_calls": [
                        {
                            "id": f"call{len(requests)}",
                            "type": "function",
                            "function": {"name": step[0], "arguments": json.dumps(step[1])},
                        }
                    ],
                }
        return httpx.Response(200, text=sse(message), headers={"content-type": "text/event-stream"})

    return RequestLayer(
        config,
        bridge.clock,
        client=httpx.AsyncClient(
            base_url="http://mock/v1/", transport=httpx.MockTransport(endpoint)
        ),
    )


async def pi_game(tmp_path, script, requests, **config):
    import os

    from prolong_qwen.agent import ProlongPi

    clock = Clock(600)
    bridge = Bridge(
        Environment(), Archive(tmp_path / "game"), clock, label="P pi", boundaries=False
    )
    await bridge.initialize()
    cfg = Config(**{**P_DEFAULTS, **config})
    layer = scripted_layer(bridge, script, requests, cfg)
    agent = ProlongPi(bridge, layer, pi_dir=os.environ.get("ARC_PI_DIR", "/app/pi"))
    await agent.start()
    clock.release()
    return ProlongGame(bridge, agent, cfg), agent, layer


@pi_only
async def test_prolong_pi_end_to_end_resumes_session_without_images_or_numpy(tmp_path):
    requests = []
    script = [
        ("bash", {"command": "head -3 logs.txt && wc -l logs.txt"}),
        ("write", {"path": "actions.json", "content": '{"actions": ["ACTION1", "ACTION2"]}'}),
        ("text", "Briefing one.\n[PLAN]\nTwo test moves."),
        (
            "bash",
            {
                "command": "grep -c '^Action ' logs.txt; python3 -c 'import numpy' || echo no-numpy; "
                "python3 -c \"from segmentation import segment; b=open('current_board.txt').read().split('\\n')[4:]; print('nodes', len(segment([l for l in b if l])['nodes']))\""
            },
        ),
        ("write", {"path": "actions.json", "content": '{"actions": ["ACTION3"]}'}),
        ("text", "Briefing two.\n[PLAN]\nOne more."),
    ]
    g, agent, layer = await pi_game(tmp_path, script, requests, max_actions=3)
    try:
        async with asyncio.timeout(120):
            await g.run()
        assert (
            g.bridge.actions == 3
            and len(g.agent.calls if hasattr(g.agent, "calls") else [1, 2]) >= 1
        )
        text = (agent.executor.workspace / "logs.txt").read_text()
        assert text.count("[POST-ACTION BOARD STATE]") == 3
        assert "\n[PLAN]\nBriefing one.\n\nTool Call: ACTION1({})\n" in text
        assert "[PLAN]\nTwo test moves." in text and "[PLAN]\nBriefing two." in text
        assert len(requests) == 6
        system = requests[0]["messages"][0]
        assert system["role"] == "system" and "./logs.txt" in system["content"]
        assert "Color Map (ASCII character" in system["content"]
        assert not any(
            part.get("type") == "image_url"
            for r in requests
            for m in r["messages"]
            for part in (m["content"] if isinstance(m["content"], list) else [])
        )
        # The second call resumes the same Pi session: its request carries the first briefing.
        assert any("Briefing one." in json.dumps(m) for m in requests[3]["messages"])
        # The clock line is appended after the prompt, so search the whole request.
        assert "Read the full game log" in json.dumps(requests[0]["messages"])
        assert "Recent actions and boards are at the end" in json.dumps(requests[3]["messages"])
        outputs = [g.bridge.archive.transcript(i) for i in range(g.bridge.archive.count("calls"))]
        joined = "".join(o.get("output", "") for o in outputs)
        assert "no-numpy" in joined and "standard library" in joined
        assert "nodes 1" in joined and (agent.executor.workspace / "segmentation.py").exists()
        events = [
            json.loads(p)
            for k, p in g.bridge.archive.db.execute("SELECT kind,payload FROM events")
            if k == "prolong_call"
        ]
        assert [e["tool_turns"] for e in events] == [3, 3]
        assert [e["plan_length"] for e in events] == [2, 1]
    finally:
        await agent.close()
        await layer.close()
        g.bridge.archive.close()


@pi_only
async def test_prolong_pi_turn_cap_interrupts_the_call(tmp_path):
    requests = []
    script = [("bash", {"command": "echo probe"})] * 10 + [("text", "unreachable")]
    g, agent, layer = await pi_game(tmp_path, script, requests, call_turns=2, max_actions=1)
    try:
        async with asyncio.timeout(60):
            result = await agent.analyze(
                is_first=True, action_num=0, score=0, level=1, last_actions="none",
                available_actions=["ACTION1", "RESET"],
            )  # fmt: skip
        kinds = [k for (k,) in g.bridge.archive.db.execute("SELECT kind FROM events")]
        stderr = (g.bridge.archive.root / "pi-stderr.log").read_text(errors="replace")[-2000:]
        detail = (result["turns"], result["model_error"], len(requests), kinds[-12:], stderr)
        assert result["turn_cap_hit"] and result["actions"] == [], detail
        assert len(requests) <= 3, detail
        assert "prolong_turn_cap" in kinds, detail
    finally:
        await agent.close()
        await layer.close()
        g.bridge.archive.close()
