"""Pi coding agent playing Schema's procedure through the arc3 CLI in the sandbox."""

import shutil
from pathlib import Path

from qwen_baselines.executor import Executor
from qwen_baselines.pi_agent import PiAgent

from . import prompts

PACKAGE = Path(__file__).parent
ARC_API = PACKAGE.parents[0] / "qwen_baselines" / "arc_api.py"


class TurnCapReached(TimeoutError):
    pass


class SchemaPi(PiAgent):
    TOOLS = ("read", "write", "edit", "bash")

    def __init__(self, bridge, layer, *, pi_dir, node="node"):
        super().__init__(bridge, layer, pi_dir=pi_dir, node=node)
        self.turn_cap_hit = False
        self.interrupting = False
        stage_cli(self.executor.api)

    def make_executor(self):
        cfg = self.layer.config
        return Executor(
            self.bridge,
            self.bridge.archive.root / "workspace",
            api=False,
            evidence=False,
            scientific=cfg.scientific_python,
            bash_seconds=cfg.bash_seconds,
            bash_max_seconds=cfg.bash_max_seconds,
        )

    def system_prompt(self):
        return prompts.SYSTEM_PROMPT

    async def begin_turn(self):
        # Schema handles GAME_OVER itself: only RESET (action 0) is legal and the agent
        # commits it. No automatic reset between model requests.
        return

    async def admit(self):
        self.turns += 1
        if self.turns > self.layer.config.call_turns:
            self.turn_cap_hit = True
            self.bridge.archive.event("schema_turn_cap", turns=self.turns)
            await self.interrupt()
            raise TurnCapReached("Tool-turn cap reached for this Schema turn")

    async def interrupt(self):
        if self.process and not self.interrupting:
            self.interrupting = True
            await self.send({"type": "interrupt"})

    async def prompt(self, text):
        """One Schema turn: the observe output as the user message; returns the turn record."""
        self.text, self.turns, self.turn_cap_hit, self.interrupting = "", 0, False, False
        self.last_model_error = None
        await self.send({"type": "prompt", "text": text})
        await self.receive_until("done", raise_model_errors=False)
        return {
            "text": self.text,
            "turns": self.turns,
            "turn_cap_hit": self.turn_cap_hit,
            "model_error": self.last_model_error,
        }

    async def run(self):
        raise RuntimeError("SchemaGame drives this agent; call prompt() per turn")


def stage_cli(api_dir):
    """The arc3 CLI, its library and the socket client, importable from sandbox Python."""
    api_dir = Path(api_dir)
    shutil.copyfile(PACKAGE / "arc3.py", api_dir / "arc3.py")
    shutil.copyfile(ARC_API, api_dir / "arc_api.py")
    target = api_dir / "agi3lib"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        PACKAGE / "agi3lib", target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
