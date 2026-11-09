"""Direct and persistent-Python loops, with no forced game actions."""

import asyncio
import json

from . import prompts
from .model import Conversation
from .output_filter import suppress_dump

RECEIPT_KEYS = (
    "observation_id",
    "action",
    "status",
    "levels_completed",
    "win_levels",
    "level_completed",
    "game_over",
    "done",
    "reset",
    "board_changed",
    "transition",
    "frame_count",
    "attempt",
    "actions_taken",
    "elapsed_seconds",
)


def receipt(meta):
    """Compact per-action receipt; the full observation follows as its own message."""
    return {key: meta[key] for key in RECEIPT_KEYS if key in meta}


class Agent:
    def __init__(self, baseline, bridge, layer, worker=None):
        self.baseline, self.bridge, self.layer, self.worker = baseline, bridge, layer, worker
        cfg = layer.config
        self.tools = prompts.tools(baseline, cfg.actions_per_turn)
        self.memory = None
        self.conversation = Conversation(
            prompts.system(
                baseline,
                cfg.gameplay_seconds,
                text_grid=cfg.text_grid,
                image_pixels=cfg.image_pixels,
                actions_per_turn=cfg.actions_per_turn,
                object_view=cfg.object_view,
            ),
            self.observe(),
            bridge.archive,
        )

    def observe(self, index=-1, *, memory=None, objects=True):
        cfg = self.layer.config
        return prompts.observation(
            self.bridge.archive,
            index,
            text_grid=cfg.text_grid,
            pixels=cfg.image_pixels,
            settled_only=self.baseline == "B",
            memory=memory,
            objects=self.bridge.archive.object_view(index, limit=cfg.object_view_limit)
            if objects and cfg.object_view
            else None,
        )

    async def act(self, args, remaining):
        """Execute an ordered action list; stop at a boundary, an error or the response cap."""
        actions = args["actions"] if "actions" in args else [args]
        if not isinstance(actions, list) or not actions:
            raise ValueError("act needs a non-empty list of actions")
        results, executed, stopped = [], 0, None
        for item in actions:
            if stopped:
                results.append({"skipped": True, "reason": stopped})
                continue
            if remaining is not None and remaining <= 0:
                stopped = self.cap_reason()
                results.append({"skipped": True, "reason": stopped})
                continue
            try:
                if not isinstance(item, dict):
                    raise TypeError("Each action must be an object with an action name")
                meta = await self.bridge.step(item["action"], item.get("x"), item.get("y"))
            except (ValueError, RuntimeError, TypeError, KeyError) as exc:
                results.append({"error": str(exc)})
                stopped = "Execution failed; completed actions remain recorded"
                self.bridge.archive.event("tool_error", error=str(exc))
                continue
            results.append(receipt(meta))
            executed += 1
            if remaining is not None:
                remaining -= 1
            if self.bridge.stopped:
                stopped = "Level/reset/terminal boundary reached"
        return {"results": results, "executed": executed, "stopped": stopped}, executed, stopped

    def cap_reason(self):
        limit = self.layer.config.actions_per_turn
        return (
            "Only one action is executed per response; observe the result before acting again"
            if limit == 1
            else f"At most {limit} actions are executed per response; observe the result before acting again"
        )

    def remember(self, message):
        memory = prompts.memory_block(message.get("content"))
        if memory:
            self.memory = memory
            self.bridge.archive.event("memory", chars=len(memory))
            return True
        self.bridge.archive.event("memory_missing", carried=self.memory is not None)
        return False

    async def execute(self, message):
        exchange = [message]
        before = self.bridge.archive.count()
        skip = None
        limit = self.layer.config.actions_per_turn if self.baseline == "A" else None
        executed = 0
        remembered = self.remember(message)
        for call in message.get("tool_calls") or []:
            call_before = self.bridge.archive.count()
            if skip:
                result = {"skipped": True, "reason": skip, "executed": False}
            else:
                try:
                    function = call["function"]
                    args = json.loads(function["arguments"])
                    if self.baseline == "A" and function["name"] == "act":
                        result, count, stopped = await self.act(
                            args, None if limit is None else limit - executed
                        )
                        executed += count
                        if stopped:
                            skip = stopped
                        elif limit is not None and executed >= limit:
                            skip = self.cap_reason()
                    elif self.baseline == "B" and function["name"] == "python":
                        result = await self.worker.execute(args["code"])
                        if self.layer.config.suppress_grid_dumps:
                            result["output"], info = suppress_dump(result["output"])
                            if info:
                                result["output_suppressed"] = info
                                result["retrieval"] = (
                                    f"transcript[{result['call_id']}]['output'] (full output)"
                                )
                                self.bridge.archive.event(
                                    "output_suppressed", call_id=result["call_id"], **info
                                )
                        if self.bridge.stopped:
                            skip = "Level/reset/terminal boundary reached"
                        elif self.bridge.archive.count() > call_before:
                            skip = "Acting Python call completed; await a fresh model decision"
                    else:
                        raise ValueError("Unknown tool")
                except (ValueError, RuntimeError, TypeError, KeyError, TimeoutError) as exc:
                    result = {
                        "error": str(exc),
                        "executed_observations": list(
                            range(call_before, self.bridge.archive.count())
                        ),
                    }
                    skip = "Execution failed; completed actions remain recorded"
                    self.bridge.archive.event("tool_error", error=str(exc))
            exchange.append(
                {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}
            )
        if not message.get("tool_calls"):
            exchange.append(
                {"role": "user", "content": "Continue when ready. The game remains available."}
            )
            self.bridge.archive.event("neutral_continuation")
        # Include each actually returned observation, not merely the final board.
        indices = list(
            range(before, self.bridge.archive.count())
            if self.baseline == "A"
            else ([self.bridge.archive.count() - 1] if self.bridge.archive.count() > before else [])
        )
        for position, index in enumerate(indices):
            # A response without a MEMORY: block gets the carried memory re-attached to the
            # exchange's last observation so trimming cannot lose it.
            carry = self.memory if not remembered and position == len(indices) - 1 else None
            # A appends one observation per action: the object view rides on the last only.
            exchange.append(
                self.observe(index, memory=carry, objects=position == len(indices) - 1)
            )
        latest = exchange[-1] if self.bridge.archive.count() > before else None
        self.conversation.append(exchange)
        if latest:
            self.conversation.latest = latest

    async def run(self):
        try:
            async with asyncio.timeout(self.bridge.clock.remaining()):
                while (
                    self.bridge.clock.remaining() > 0
                    and self.bridge.archive.metadata()["status"] != "WIN"
                ):
                    before = self.bridge.archive.count()
                    await self.bridge.begin_turn()
                    if self.bridge.archive.count() > before:
                        obs = self.observe()
                        self.conversation.append([obs])
                        self.conversation.latest = obs
                    messages = await self.conversation.maintain(self.layer, self.tools)
                    message = await self.layer.complete(messages, self.tools, self.bridge.archive)
                    await self.execute(message)
        except TimeoutError:
            self.bridge.archive.event("deadline", seconds=self.bridge.clock.elapsed())
