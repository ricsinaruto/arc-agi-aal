"""Exercise the live native protocol on synthetic evidence before public gameplay."""

import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from qwen_baselines.bridge import Bridge
from qwen_baselines.clock import Clock
from qwen_baselines.evidence import Archive
from qwen_baselines.model import RequestLayer

from .agent import NativeAgent
from .interface import execute, response_text


class ProbeEnvironment:
    """A separate synthetic board; never touches public game state or scorecards."""

    def __init__(self):
        self.moves = 0

    @property
    def observation_space(self):
        grid = np.zeros((64, 64), dtype=np.uint8)
        grid[30:34, 30:34] = 2 if self.moves == 0 else 3
        return SimpleNamespace(frame=[grid], state="NOT_FINISHED", levels_completed=0,
                               win_levels=1, available_actions=[1])

    def step(self, action, data):
        if action.name != "ACTION1" or data:
            raise ValueError("Protocol probe accepts only ACTION1")
        self.moves += 1
        return self.observation_space


async def check(config, output, *, client=None):
    """One bounded generation and a follow-up tokenize; no retries or public actions."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = {"status": "running", "kind": "synthetic native action handoff",
              "public_game_actions": 0, "output_cap": 2048,
              "thinking_enabled": False, "timeout_seconds": 60}
    cfg = replace(config, gameplay_seconds=60, output_tokens_cap=2048)
    clock = Clock(60)
    archive = Archive(output / "evidence")
    layer = RequestLayer(cfg, clock, client=client)
    bridge = Bridge(ProbeEnvironment(), archive, clock, label="native protocol preflight")
    agent = NativeAgent(bridge, layer, game=None, window=131072)
    agent.task = {"role": "user", "content":
        "This is a synthetic interface check, not a public game. Issue exactly one "
        "computer_use call with action press and keys [\"1\"]. Do not wait, finish "
        "or plan other actions. The screenshot is synthetic test evidence."}
    try:
        clock.release()
        async with asyncio.timeout(60):
            await bridge.initialize()
            observation = agent.observation([])
            payload = await agent.payload(observation)
            # Disable thinking for this setup check only, not gameplay.
            payload["chat_template_kwargs"] = layer.template_settings(summary=True)
            response = await layer.complete_payload(payload, archive, max_tokens=2048)
            data = response.json()
            record["usage"] = data.get("usage", {})
            if data["choices"][0].get("finish_reason") == "length":
                raise RuntimeError("Native protocol check truncated; no action executed")
            message = data["choices"][0]["message"]
            feedback = await execute(message, bridge)
            record["feedback"] = feedback
            if (feedback["calls"] != 1 or bridge.actions != 1
                    or archive.metadata()["action"] != {"action": "ACTION1"}
                    or any(r.get("error") or r.get("skipped") for r in feedback["results"])):
                raise RuntimeError("Native protocol check did not execute exactly one ACTION1")
            agent.feedback = feedback
            agent.exchanges.append([observation, {"role": "assistant", "content": response_text(message)}])
            # Count the actual next gameplay envelope, including acknowledged action
            # evidence and the restored native history, using the serving tokenizer.
            next_payload = layer.payload(agent.messages(agent.observation([])))
            record["followup_input_tokens"] = await layer.count(layer.fresh(next_payload))
            record["structured_calls"] = bool(message.get("tool_calls"))
            record["synthetic_actions"] = bridge.actions
            record["status"] = "passed"
            return record
    except BaseException as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        archive.close()
        if client is None:
            await layer.close()
