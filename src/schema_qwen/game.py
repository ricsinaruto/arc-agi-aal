"""Baseline S game loop: observe, prompt the Pi agent, count commits; Schema's CLI does the rest."""

import asyncio
import json
import shlex
import time

from .agi3lib import score
from .prompts import NO_COMMIT_NOTE

AGENT_LABEL = "qwen3.8-flash-next-pi"


class SchemaGame:
    def __init__(self, bridge, agent, config):
        self.bridge, self.agent, self.config = bridge, agent, config
        self.workdir = agent.executor.workspace / "run"
        self.game_id = bridge.label.split()[-1] if bridge.label else "game"
        self.turns = 0
        self.seen = 0

    @property
    def meta(self):
        return self.bridge.archive.metadata()

    async def cli(self, *args):
        command = "python3 -m arc3 --workdir run " + shlex.join(args)
        return await self.agent.executor.execute("bash", command=command)

    def events(self):
        path = self.workdir / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def budget_left(self):
        return self.config.max_actions is None or self.bridge.actions < self.config.max_actions

    async def run(self):
        try:
            async with asyncio.timeout(self.bridge.clock.remaining()):
                started = await self.cli(
                    "init", "--env", "rpc", "--game", self.game_id, "--agent", AGENT_LABEL,
                    "--max-actions", str(self.config.max_actions or 3000),
                )
                if started["exitCode"]:
                    raise RuntimeError("Schema init failed: " + started["output"][-2000:])
                self.seen = len(self.events())
                no_commit = False
                while (
                    self.bridge.clock.remaining() > 0
                    and self.meta["status"] != "WIN"
                    and self.budget_left()
                ):
                    observed = await self.cli("observe")
                    text = (NO_COMMIT_NOTE if no_commit else "") + observed["output"]
                    turn_started = time.monotonic()
                    self.turns += 1
                    result = await self.agent.prompt(text)
                    new = self.events()[self.seen :]
                    self.seen += len(new)
                    commits = [e for e in new if e["kind"] == "turn_committed"]
                    self.bridge.archive.event(
                        "schema_turn",
                        turn=self.turns,
                        seconds=time.monotonic() - turn_started,
                        tool_turns=result["turns"],
                        turn_cap_hit=result["turn_cap_hit"],
                        model_error=result["model_error"],
                        commits=len(commits),
                        planned_actions=sum(len(e.get("plan", [])) for e in commits),
                        actions_taken=sum(e["kind"] == "action_taken" for e in new),
                        mispredicted=sum(e["kind"] == "model_mispredicted" for e in new),
                        tool_calls=sum(e["kind"] == "tool_started" for e in new),
                        no_commit_note=no_commit,
                    )
                    no_commit = not commits
        except TimeoutError:
            self.bridge.archive.event("deadline", seconds=self.bridge.clock.elapsed())
            await self.agent.interrupt()
        finally:
            self.finish()

    def finish(self):
        """Host-side run_finished when the CLI did not emit one (deadline, budget, error)."""
        from .agi3lib.events import EventLog, transitions_from_events

        events = self.events()
        if not events or any(e["kind"] == "run_finished" for e in events):
            return
        meta = self.meta
        EventLog(self.workdir / "events.jsonl").emit(
            "run_finished",
            state=meta["status"],
            levels=meta["levels_completed"],
            win_levels=meta["win_levels"],
            actions=len(transitions_from_events(events)),
            transitions=len(transitions_from_events(events)),
            has_world_model=any(self.workdir.glob("world_model*.py")),
            harness_finished=True,
        )


def schema_summary(archive):
    """Per-game Schema statistics from run/events.jsonl and the schema_turn events."""
    path = archive.root / "workspace" / "run" / "events.jsonl"
    if not path.exists():
        return None
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    summary = score.summarize_events(events)
    turns = [
        json.loads(payload)
        for kind, payload in archive.db.execute("SELECT kind,payload FROM events ORDER BY id")
        if kind == "schema_turn"
    ]
    tools = {}
    for event in events:
        if event["kind"] == "tool_started":
            tools[event["name"]] = tools.get(event["name"], 0) + 1
    return {
        "turns": len(turns),
        "commits": sum(e["kind"] == "turn_committed" for e in events),
        "mispredicted": sum(e["kind"] == "model_mispredicted" for e in events),
        "tool_calls": tools,
        "has_world_model": any((archive.root / "workspace" / "run").glob("world_model*.py")),
        "state": summary.state,
        "completed_actions": list(summary.completed_actions),
        "total_actions": summary.total_actions,
        "turn_cap_hits": sum(t["turn_cap_hit"] for t in turns),
        "model_errors": sum(bool(t["model_error"]) for t in turns),
        "tool_turns": [t["tool_turns"] for t in turns],
        "turn_seconds": [round(t["seconds"], 1) for t in turns],
    }
