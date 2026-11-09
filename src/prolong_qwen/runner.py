"""Upstream PRO-LONG game loop (runner.py) on the offline engine with a gameplay clock."""

import asyncio
import time

from .action_queue import ActionQueue, QueueExhausted
from .log import GameLog

ACTION_NAMES = {
    0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3",
    4: "ACTION4", 5: "ACTION5", 6: "ACTION6", 7: "ACTION7",
}  # fmt: skip

RETRY_NUDGE = (
    "Your previous response did not produce a valid /workspace/actions.json. "
    'Please write one with the shape {"actions": [...]}.'
)


def available_actions(legal):
    return sorted({*legal, "RESET"})


class ProlongGame:
    def __init__(self, bridge, agent, config, *, retries=5):
        self.bridge, self.agent, self.config = bridge, agent, config
        self.retries = retries
        self.queue = ActionQueue()
        self.log = GameLog(agent.executor.workspace, config.grid_mode)
        self.level, self.attempt = 1, 1
        self.last_executed = None
        self.recent = []
        self.calls = 0

    @property
    def meta(self):
        return self.bridge.archive.metadata()

    def observe(self):
        meta = self.meta
        frames = [self.bridge.archive.frame(-1, i) for i in range(meta["frame_count"])]
        self.log.observe(frames, meta["status"], meta["levels_completed"])

    def next_action(self):
        """Upstream `_next_action`: forced RESET, queue drain, ACTION7 guard, exhaustion."""
        meta = self.meta
        if meta["status"] in ("NOT_PLAYED", "GAME_OVER") and self.last_executed != "RESET":
            return {"name": "RESET", "data": {}, "forced": True}
        use_queued = bool(self.queue and not self.queue.score_changed)
        if not use_queued:
            self.queue.score_changed = False
        if use_queued and self.queue:
            action = self.queue.pop()
            unsafe = action["name"] == "RESET" and (
                self.last_executed == "RESET" or meta["status"] == "WIN"
            )
            if unsafe:
                if "ACTION7" in meta["legal_actions"]:
                    self.queue.push_front(action)
                    action = {"name": "ACTION7", "data": {}, "injected": True}
                else:
                    # Offline engines rarely advertise ACTION7; drop the duplicate instead.
                    self.bridge.archive.event("prolong_duplicate_reset_dropped")
                    return self.next_action() if self.queue else self.exhausted()
            return action
        return self.exhausted()

    def exhausted(self):
        raise QueueExhausted("Queue empty, no actions from agent")

    async def plan(self):
        action_num = self.bridge.actions
        backoff = 30
        while True:
            for attempt in range(self.retries):
                nudge = RETRY_NUDGE if attempt else ""
                if await self.call_agent(action_num, nudge, attempt):
                    return
            self.bridge.archive.event("prolong_plan_backoff", seconds=backoff, action=action_num)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 300)

    async def call_agent(self, action_num, nudge, attempt):
        meta = self.meta
        self.log.current_board(meta["levels_completed"], action_num)
        started = time.monotonic()
        self.calls += 1
        result = await self.agent.analyze(
            is_first=self.calls == 1,
            action_num=action_num,
            score=meta["levels_completed"],
            level=self.level,
            last_actions=", ".join(self.recent[-5:]) if self.recent else "none",
            available_actions=available_actions(meta["legal_actions"]),
            nudge=nudge,
        )
        loaded = bool(result["actions"]) and self.queue.load(result["actions"])
        self.bridge.archive.event(
            "prolong_call",
            call=self.calls,
            attempt=attempt,
            action=action_num,
            seconds=time.monotonic() - started,
            tool_turns=result["turns"],
            actions_parsed=len(result["actions"]),
            plan_length=self.queue.plan_total if loaded else 0,
            loaded=loaded,
            nudge=bool(nudge),
            turn_cap_hit=result["turn_cap_hit"],
            model_error=result["model_error"],
            hint_chars=len(result["hint"]),
        )
        if loaded:
            self.log.state.set_external_hint(result["hint"])
            self.log.state.set_persistent_hint(result["plan"])
        return loaded

    async def execute(self, action):
        name, data = action["name"], dict(action.get("data", {}))
        if name == "ACTION6":
            data = {k: max(0, min(63, int(data.get(k, 0)))) for k in ("x", "y")}
        before = self.meta["levels_completed"]
        try:
            meta = await self.bridge.step(
                name, data.get("x"), data.get("y"), automatic=bool(action.get("forced"))
            )
        except ValueError as exc:
            # Upstream would have failed the API request; drop the rest of this plan instead.
            self.bridge.archive.event("prolong_invalid_action", action=name, error=str(exc))
            self.queue.clear()
            return
        self.last_executed = name
        self.recent.append(f"ACTION6({data['x']},{data['y']})" if name == "ACTION6" else name)
        self.observe()
        levels, status = meta["levels_completed"], meta["status"]
        self.queue.check_score(levels)
        self.log.action(
            self.bridge.actions, self.level, self.attempt, self.queue.plan_index,
            self.queue.plan_total, levels, name, data,
        )  # fmt: skip
        self.log.post_board()
        if levels > before and status not in ("WIN", "GAME_OVER"):
            self.level += 1
            self.attempt = 1
        elif status == "GAME_OVER":
            self.attempt += 1

    def budget_left(self):
        return self.config.max_actions is None or self.bridge.actions < self.config.max_actions

    async def run(self):
        try:
            async with asyncio.timeout(self.bridge.clock.remaining()):
                self.observe()
                self.log.initial(self.level, self.attempt, self.meta["levels_completed"])
                self.log.run_info(self.bridge.label, "prolong_qwen baseline P")
                self.log.current_board(self.meta["levels_completed"], 0)
                while (
                    self.bridge.clock.remaining() > 0
                    and self.meta["status"] != "WIN"
                    and self.budget_left()
                ):
                    try:
                        action = self.next_action()
                    except QueueExhausted:
                        await self.plan()
                        continue
                    await self.execute(action)
        except TimeoutError:
            self.bridge.archive.event("deadline", seconds=self.bridge.clock.elapsed())
            await self.agent.interrupt()
