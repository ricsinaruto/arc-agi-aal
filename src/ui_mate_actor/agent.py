"""Native screenshot/XML loop using the arena's clocks, cache gate and action archive."""

import asyncio
import base64
import copy
import json
import time

from qwen_baselines.agent import receipt
from qwen_baselines.evidence import png

from . import native
from .interface import execute, response_text, system_prompt


class NativeAgent:
    def __init__(self, bridge, layer, *, game, window):
        self.bridge, self.layer, self.game = bridge, layer, game
        self.window = window
        self.system = {"role": "system", "content": system_prompt(layer.config.gameplay_seconds)}
        # Native templates distinguish the user task from tool-response screenshots.
        # Keep this query outside the evictable exchange history.
        self.task = {"role": "user", "content":
            "Play the ARC game shown in the observations. Infer its rules and complete "
            "as many levels as possible within the gameplay time limit. Use computer_use "
            "according to the system instructions."}
        self.exchanges = []
        self.feedback = None
        self.turns = 0
        self.omitted = 0
        self.metrics = {"tool_calls": 0, "invalid_calls": 0, "skipped_operations": 0,
                        "no_action_replies": 0, "context_reductions": 0, "tool_seconds": 0.0,
                        "context_maintenance_seconds": 0.0}

    def observation(self, automatic):
        archive = self.bridge.archive
        meta = archive.metadata()
        text = {"observation": receipt(meta), "legal_actions": meta["legal_actions"],
                "intermediate_frame_count": meta["intermediate_frame_count"],
                "settled_frame_index": meta["settled_frame_index"],
                "previous_execution": self.feedback, "automatic_receipts": automatic}
        return {"role": "user", "content": [
            {"type": "text", "text": "<tool_response>\n" + json.dumps(text) + "\n"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                base64.b64encode(png(archive.frame(), self.layer.config.image_pixels)).decode()}},
            {"type": "text", "text": "\n</tool_response>"},
        ]}

    def messages(self, observation):
        messages = [copy.deepcopy(self.system), copy.deepcopy(self.task)]
        if self.omitted:
            messages.append({"role": "user", "content":
                f"<tool_response>\n{self.omitted} earlier complete exchanges were omitted from this context. "
                "Only the retained evidence is available to you; the game state has not reset.\n</tool_response>"})
        for exchange in self.exchanges:
            messages.extend(copy.deepcopy(exchange))
        messages.append(copy.deepcopy(observation))
        # Same call as upstream predict(): five images, oldest first, retaining the
        # earliest screenshot still in the history window and the recent four.
        messages, _ = native.collapse_messages(messages, images_to_keep=5, min_removal_threshold=1)
        return messages

    async def payload(self, observation):
        # Native GUI history is bounded; token backstop keeps the common 128K target.
        # Original messages remain in SQLite request/response archives.
        started = time.monotonic()
        removed = 0
        while len(self.exchanges) > 100:
            self.exchanges.pop(0)
            self.omitted += 1
            removed += 1
        while True:
            payload = self.layer.payload(self.messages(observation))
            count = await self.layer.count(self.layer.fresh(payload))
            if count + int(self.layer.config.output_tokens_cap) + 256 <= self.window:
                break
            if not self.exchanges:
                raise RuntimeError("Native UI-Mate observation cannot fit the arena context target")
            self.exchanges.pop(0)
            self.omitted += 1
            removed += 1
        if removed:
            self.metrics["context_reductions"] += 1
            self.bridge.archive.event("ui_mate_history_trim", removed_exchanges=removed,
                                      input_tokens=count, retained_exchanges=len(self.exchanges))
        self.metrics["context_maintenance_seconds"] += time.monotonic() - started
        return payload

    async def run(self):
        try:
            while self.game.clock.remaining() > 0:
                if self.bridge.archive.metadata()["status"] == "WIN":
                    break
                await self.game.clock.wait_active()
                before = self.bridge.archive.count()
                await self.bridge.begin_turn()
                automatic = [receipt(self.bridge.archive.metadata(i))
                             for i in range(before, self.bridge.archive.count())]
                observation = self.observation(automatic)
                payload = await self.payload(observation)
                response = await self.layer.complete_payload(payload, self.bridge.archive)
                message = response.json()["choices"][0]["message"]
                text = response_text(message)
                self.turns += 1
                self.exchanges.append([observation, {"role": "assistant", "content": text}])
                # A game may be parked while its admitted generation finishes.
                # Do not execute the result while it lacks an active allocation.
                await self.game.clock.wait_active()
                started = time.monotonic()
                self.feedback = await execute(message, self.bridge)
                self.metrics["tool_seconds"] += time.monotonic() - started
                self.metrics["tool_calls"] += self.feedback["calls"]
                self.metrics["invalid_calls"] += sum("error" in r for r in self.feedback["results"])
                self.metrics["skipped_operations"] += sum(bool(r.get("skipped")) for r in self.feedback["results"])
                self.metrics["no_action_replies"] += self.feedback["calls"] == 0
                if any("error" in r for r in self.feedback["results"]):
                    self.bridge.archive.event("tool_error", interface="ui-mate-native", details=self.feedback)
                if self.feedback["calls"] == 0:
                    self.bridge.archive.event("neutral_continuation")
                self.bridge.archive.event("ui_mate_execution", turn=self.turns, **self.feedback)
                if self.feedback.get("finished_claim"):
                    self.game.note_finished("model_finished_" + self.feedback["finished_claim"])
                    break
        except (TimeoutError, asyncio.CancelledError):
            self.bridge.archive.event("deadline", seconds=self.game.clock.elapsed())
        except Exception as exc:
            self.game.error = f"{type(exc).__name__}: {exc}"
            self.game.note_finished("error")
            raise

    async def close(self):
        pass  # The trial owns the shared HTTP client and archive.

    def arena_config(self):
        return {**self.layer.config.to_dict(), "interface": "ui-mate-native-arc",
                "images_to_keep": 5, "history_exchanges": 100, "window": self.window,
                "context_policy": "native screenshot collapse; oldest complete exchanges dropped at token cap",
                "prompt_source": "Tencent/UI-Mate@1cb9e1e44ce856e23b593992b02efbd489943fcb",
                "adaptations": "ARC controls/receipts; no desktop actions or code; authoritative scoring"}
