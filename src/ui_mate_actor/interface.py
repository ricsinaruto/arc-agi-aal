"""ARC-only translation of UI-Mate's native computer_use calls; never execute code."""

import asyncio
import json
import math
import re
from xml.sax.saxutils import escape

from qwen_baselines.agent import receipt

from . import native

KEYS = {str(i): f"ACTION{i}" for i in (1, 2, 3, 4, 5, 7)} | {
    "0": "RESET", "r": "RESET", "up": "ACTION1", "down": "ACTION2",
    "left": "ACTION3", "right": "ACTION4", "space": "ACTION5", "backspace": "ACTION7",
}
DESCRIPTION = """Use a mouse and keyboard to play an unfamiliar ARC grid game.
This is a game interface, not a desktop: there are no files, applications or terminal.
The screenshot shows the entire 64x64 board, including any HUD, with no controls outside it.
Use left_click with relative (x,y) coordinates from 0 to 999 inclusive; x is horizontal.
Keyboard mapping: '1' or up = ACTION1, '2' or down = ACTION2, '3' or left = ACTION3,
'4' or right = ACTION4, '5' or space = ACTION5, '7' or backspace = ACTION7.
Use left_click for ACTION6. '0' or r = RESET. Only currently advertised actions are legal.
These are input names, not promises about their effect; infer each game's mechanics.
press accepts an ordered array of discrete keys, not held keys or simultaneous chords.
wait only passes wall time; it does not issue a game action.
finished ends your attempt, but success is measured from the environment, not your claim.
"""


def system_prompt(seconds):
    tool = native.build_tools_def(DESCRIPTION)
    properties = tool["function"]["parameters"]["properties"]
    properties["action"]["enum"] = ["left_click", "press", "wait", "finished"]
    properties["action"]["description"] = (
        "left_click: one ACTION6 at coordinate. press: execute keys in order. "
        "wait: wait time seconds. finished: end your attempt with status success or failure."
    )
    tool["function"]["parameters"]["properties"] = {
        key: properties[key] for key in ("action", "coordinate", "keys", "time", "status")
    }
    task = f"""Infer the rules and goal from screenshots and action outcomes. Complete the levels
using as few game actions as practical within {seconds:g} seconds of gameplay.
Inference, waiting and context maintenance count toward this time; setup does not.
Games may have no avatar, and changing border elements may be HUD. A changed board is
not proof of progress. Only increased levels_completed verifies a clear; WIN verifies
the whole game. Screenshots show the settled board. frame_count includes any preceding
animation frames, in order; it does not specify animation duration. All returned frames
are archived, but this interface provides only settled screenshots.
After a level/reset/terminal boundary or execution error, the rest of the planned
actions are skipped. Observe the new state before another decision. GAME_OVER is reset
by the runner and the reset receipt is reported. No action is automatically replayed.
"""
    return ("You are a helpful GUI agent.\n\n" + native.build_tools_and_format_block(tool)
            + "\n\n" + task + "\n# Response format\n" + native.RESPONSE_FORMAT)


def response_calls(response):
    """Prefer authoritative structured calls; never execute calls quoted in reasoning."""
    if isinstance(response, dict) and response.get("tool_calls"):
        if not isinstance(response["tool_calls"], list):
            return [(None, "tool_calls must be an array")]
        parsed = []
        for call in response["tool_calls"]:
            try:
                function = call.get("function", {})
                if call.get("type", "function") != "function" or function.get("name") != "computer_use":
                    raise ValueError("Only computer_use calls are supported")
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("computer_use arguments must be a JSON object")
                arguments = dict(arguments)
                for key, value in arguments.items():
                    if key not in {"action", "coordinate", "keys", "time", "status"}:
                        raise ValueError(f"Unsupported computer_use parameter: {key}")
                    # Without an API tool schema, the server can return arrays as
                    # strings. Match the pinned upstream XML parser's conversion.
                    if isinstance(value, str) and value.strip().startswith(("[", "{")):
                        try:
                            arguments[key] = json.loads(value)
                        except json.JSONDecodeError:
                            pass
                parsed.append((arguments, None))
            except (AttributeError, TypeError, ValueError) as exc:
                parsed.append((None, str(exc)))
        return parsed
    content = (response.get("content") or "") if isinstance(response, dict) else response
    if not isinstance(content, str):
        return [(None, "Expected textual UI-Mate content")]
    action_text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    action_text = re.split(r"<think>", action_text, maxsplit=1, flags=re.IGNORECASE)[0]
    return [(call, None) for call in native.extract_xml_tool_calls(action_text)]


def response_text(message):
    """Restore native history after the server separates reasoning and tool calls."""
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise ValueError("Expected textual UI-Mate response")
    if message.get("tool_calls"):
        # Some servers return both forms. Replay each authoritative call once.
        content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL)
        for call, error in response_calls(message):
            if error:
                content += "\n[Malformed structured call: " + escape(error) + "]\n"
                continue
            fields = []
            for key, value in call.items():
                value = json.dumps(value) if not isinstance(value, str) else value
                fields.append(f"<parameter={key}>\n{escape(value)}\n</parameter>")
            content += "\n<tool_call>\n<function=computer_use>\n" + "\n".join(fields) + "\n</function>\n</tool_call>"
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning and "<think>" not in content:
        content = f"<think>{reasoning}</think>\n{content}"
    return content


def translate(call):
    """Validate a whole native call before issuing its first action."""
    action = call.get("action")
    if action == "left_click":
        coordinate = call.get("coordinate")
        if not isinstance(coordinate, list) or len(coordinate) != 2:
            raise ValueError("left_click needs exactly two relative coordinates")
        try:
            if any(isinstance(v, bool) for v in coordinate):
                raise ValueError
            coordinate = [float(v) for v in coordinate]
        except (TypeError, ValueError):
            raise ValueError("Coordinates must be finite numbers in 0..999") from None
        if any(not math.isfinite(v) or not 0 <= v <= 999 for v in coordinate):
            raise ValueError("Coordinates must be finite numbers in 0..999")
        # The upstream formula reaches width at 999. Clamp that inclusive edge to
        # the final ARC cell; never wrap or allow an out-of-board action.
        x, y = (min(63, int(v * 64 / 999)) for v in coordinate)
        return [{"action": "ACTION6", "x": x, "y": y}]
    if action == "press":
        keys = native._clean_keys(call.get("keys"))
        # Upstream cleanup can leave quotes after keys=['up']; normalize only
        # quoting, then require every name to match our finite input whitelist.
        keys = [k.strip("'\"").lower() if isinstance(k, str) else k for k in keys]
        if not isinstance(keys, list) or not keys:
            raise ValueError("press needs a nonempty ordered keys array")
        if any(not isinstance(k, str) or k.lower() not in KEYS for k in keys):
            raise ValueError("Unknown key; use the documented discrete game keys")
        return [{"action": KEYS[k.lower()]} for k in keys]
    if action == "wait":
        value = call.get("time")
        if isinstance(value, bool):
            raise ValueError("wait time must be a finite nonnegative number")
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            raise ValueError("wait time must be a finite nonnegative number") from None
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("wait time must be a finite nonnegative number")
        return [{"wait": seconds}]
    if action == "finished" and call.get("status") in ("success", "failure"):
        return [{"finished": call["status"]}]
    raise ValueError("Unsupported computer_use action or invalid finished status")


async def execute(response, bridge):
    """Receipts survive later errors, boundary stops and cancellation in Bridge.step."""
    calls = response_calls(response)
    results, stopped, finished = [], None, None
    if not calls:
        return {"results": [], "calls": 0, "continuation": "No game action was issued. Continue when ready."}
    for index, (call, error) in enumerate(calls):
        if stopped:
            results.append({"call": index, "skipped": True, "reason": stopped})
            continue
        try:
            if error:
                raise ValueError(error)
            operations = translate(call)
        except ValueError as exc:
            results.append({"call": index, "error": str(exc)})
            stopped = "Invalid call; earlier completed actions remain recorded"
            continue
        for offset, operation in enumerate(operations):
            if stopped:
                results.append({"call": index, "operation": offset, "skipped": True, "reason": stopped})
                continue
            wait_active = getattr(bridge.clock, "wait_active", None)
            if wait_active is not None:
                await wait_active()
            if bridge.clock.remaining() <= 0:
                raise TimeoutError("Gameplay deadline reached")
            try:
                if "finished" in operation:
                    finished = operation["finished"]
                    results.append({"call": index, "model_finished_claim": finished})
                    stopped = "Model ended the attempt; this does not change the environment score"
                elif "wait" in operation:
                    async with asyncio.timeout(bridge.clock.remaining()):
                        await asyncio.sleep(operation["wait"])
                    results.append({"call": index, "waited_seconds": operation["wait"]})
                else:
                    result = await bridge.step(**operation)
                    results.append({"call": index, "operation": offset, "receipt": receipt(result)})
                    if bridge.stopped:
                        stopped = "Level/reset/terminal boundary reached"
            except (ValueError, RuntimeError) as exc:
                results.append({"call": index, "operation": offset, "error": str(exc)})
                stopped = "Execution failed; earlier completed actions remain recorded"
    return {"results": results, "calls": len(calls), "stopped": stopped, "finished_claim": finished}
