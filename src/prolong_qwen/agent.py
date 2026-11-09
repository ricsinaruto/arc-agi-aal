"""Pi coding agent driven by the vendored PRO-LONG prompts and action-file handoff."""

from pathlib import Path

from qwen_baselines import segmentation
from qwen_baselines.executor import Executor
from qwen_baselines.pi_agent import PiAgent

from .base import BaseAgent


class TurnCapReached(TimeoutError):
    pass


class ProlongPi(PiAgent):
    def __init__(self, bridge, layer, *, pi_dir, node="node"):
        super().__init__(bridge, layer, pi_dir=pi_dir, node=node)
        cfg = layer.config
        # Pi's cwd is the workspace, so upstream's Codex-style "./" path rewriting applies.
        self.protocol = BaseAgent(
            grid_mode=cfg.grid_mode, log_window=None, action_cap=cfg.plan_cap, workspace="."
        )
        self.turn_cap_hit = False
        self.interrupting = False
        # Duck's helper as a plain workspace file: the vendored prompt cannot mention it,
        # but the resume prompt already says to check ./ for saved files.
        segmentation.write(self.executor.workspace / "segmentation.py")

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
        legal = self.bridge.archive.metadata()["legal_actions"]
        return self.protocol._build_system_prompt(sorted({*legal, "RESET"}))

    async def admit(self):
        self.turns += 1
        if self.turns > self.layer.config.call_turns:
            self.turn_cap_hit = True
            self.bridge.archive.event("prolong_turn_cap", turns=self.turns)
            await self.interrupt()
            raise TurnCapReached("Tool-turn cap reached for this PRO-LONG call")

    async def interrupt(self):
        if self.process and not self.interrupting:
            self.interrupting = True
            await self.send({"type": "interrupt"})

    async def analyze(
        self, *, is_first, action_num, score, level, last_actions, available_actions, nudge=""
    ):
        workspace = Path(self.executor.workspace)
        BaseAgent._clear_files(workspace, "actions.json")
        prompt = self.protocol._build_prompt(
            "logs.txt",
            is_first,
            action_num=action_num,
            score=score,
            level=level,
            last_actions=last_actions,
            available_actions_list=available_actions,
        )
        if nudge:
            prompt += f"\n\n{self.protocol._workspace_text(nudge)}"
        self.text, self.turns, self.turn_cap_hit, self.interrupting = "", 0, False, False
        self.last_model_error = None
        await self.send({"type": "prompt", "text": prompt})
        await self.receive_until("done", raise_model_errors=False)
        raw = workspace / "actions.json"
        actions = (
            self.protocol._parse_actions_json_text(
                raw.read_text(errors="replace"), cap=self.layer.config.plan_cap
            )
            if raw.exists()
            else []
        )
        hint, plan = self.protocol._split_response(self.text.strip())
        return {
            "hint": hint,
            "plan": plan,
            "actions": actions,
            "text": self.text,
            "turns": self.turns,
            "turn_cap_hit": self.turn_cap_hit,
            "model_error": self.last_model_error,
        }

    async def run(self):
        raise RuntimeError("ProlongGame drives this agent; call analyze() per plan")
