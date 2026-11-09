"""The Duck analyzer: one observe-plan-act turn per ``analyze`` call over the OpenAI-compatible chat API, with one
``python`` tool (the persistent sandbox, ``duck_transfer.sandbox``).

``ToolAgent`` is driven by its owner (``duck_transfer.agent.DuckAgent``): requests go through ``owner.chat`` on the
the event loop, the runtime state, sandbox and workspace are the maintainer's, and the arena binds its context drain
(``arena.compaction``) and tool-output truncation onto the instance. The prompts are the v5 layout: constant
instructions once in the system prompt, each user turn carries the previous sequence's outcome, the state line and an
image of the board labelled with its step."""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import math
import re
import struct
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from inference.agent.action_names import to_engine_action, to_model_action
from inference.agent.prompts import (
    COMPACT_TOOL_SESSION_ADDENDUM,
    GAME_OVERVIEW_ADDENDUM,
    PYTHON_ADDENDUM,
    STRUCTURED_RUNTIME_STATE_ADDENDUM,
    MULTIMODAL_CONTEXT_ADDENDUM,
    TOOL_CALL_FORMAT_GUIDANCE,
    VISUAL_GAME_ADDENDUM,
)
from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.vision_context import current_grid_image_part

log = logging.getLogger(__name__)

_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)
_TOOL_CALL_PARAMETER_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
    flags=re.DOTALL | re.IGNORECASE,
)
_THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)


def _contains_tool_call_markup(*chunks: str) -> bool:
    for chunk in chunks:
        lowered = chunk.lower()
        if "<tool_call" in lowered or "<function=" in lowered:
            return True
    return False


def _strip_tool_call_markup(text: str) -> str:
    if not text.strip():
        return ""
    stripped = _TOOL_CALL_BLOCK_RE.sub("", text)
    return stripped.strip()


def _recover_tool_calls_from_markup(*chunks: str) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        if not chunk.strip():
            continue
        for match in _TOOL_CALL_BLOCK_RE.finditer(chunk):
            tool_name = str(match.group(1) or "").strip()
            if not tool_name:
                continue
            raw_body = str(match.group(2) or "")
            arguments = {
                str(parameter_name).strip(): value
                for parameter_name, value in _TOOL_CALL_PARAMETER_RE.findall(raw_body)
                if str(parameter_name).strip()
            }
            cache_key = (
                tool_name,
                json.dumps(arguments, ensure_ascii=True, sort_keys=True),
            )
            if cache_key in seen:
                continue
            seen.add(cache_key)
            recovered.append(
                {
                    "id": f"markup-call-{len(recovered) + 1}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, ensure_ascii=True),
                    },
                }
            )
    return recovered


# Sampling for the request builder (duck_transfer.agent.DuckAgent.gameplay_payload).
_LOCAL_ANALYZER_TOP_P = 0.95
_LOCAL_ANALYZER_TOP_K = 20
_TOOL_OUTPUT_TOKENS = 1024
_REQUEST_SAFETY_MARGIN_TOKENS = 512
_CONTEXT_OVERFLOW_RETRY_TRIM_TOKENS = 512
_RESPONSE_META_MAX_CHARS = 4000

_PYTHON_TOOL_DESCRIPTION = (
    "Run a Python snippet in this game's persistent interpreter against preloaded ASCII game state. Available globals: "
    "`current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, "
    "`valid_actions`, `last_action_result`, "
    "and `action(actions)` for executing one or more real environment actions. "
    "`current_frame` and each `history[*].frame` expose only `.ascii`, `.segmentation`, `.step`, and `.level`; "
    "`history[-1].frame` is the current post-action frame, not the previous frame. "
    "For before/after diffs, compare `previous_frame` to `current_frame` or use `last_transition.before_frame` and `.after_frame`. "
    "For MOUSE, pass `row` and `col` integer fields; legacy x/y fields are rejected. "
    "The raw numeric grid is not available. Use `.segmentation` as the primary view; use `.ascii` only to read a small, specific region. "
    "Use `print(...)` for compact output or assign final data to `result`."
)
_PYTHON_CODE_DESCRIPTION = "Python code to run in this game's persistent session; definitions persist across tool calls."

# Receipt keys passed through the action-result compaction: the transition summary and refusals (DuckAgent.step_duck),
# the arena dfr_harness's change/animation annotations and its batch no-op stop.
RECEIPT_KEYS = ("transition", "tried_no_effect_here", "skipped_repeats", "gameplay_changed", "animation",
                "no_op_action", "no_op_action_index", "skipped_actions", "action_trace")

IMAGE_LABEL = "Grid image of the board at step {step} (before this turn's actions):"
GAME_OVER_LINES = "You are still on the same level.\nThe game is over."
FOLLOWUP_PREFIX = "You have not acted yet. Investigate first. "
MARKUP_FOLLOWUP_PREFIX = (
    "You did not call a tool. We detected `<tool_call>` markup inside your reasoning or assistant text, "
    "so no parsed tool call was executed. On this retry, do not add a note or explanation first. "
    "Emit exactly one `python` tool call directly as your next response. "
    "Do not place `<tool_call>` markup inside reasoning, explanation, or notes. "
)
FOLLOWUP = (
    "Then investigate and revise your working world model of what the level contains, what actions appear to do, what the current goal seems to be, and what plan looks best. "
    "If helpful, include short world-model update lines such as `World model:`, `Goal model:`, `Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, or `Cross-level notes:`. "
    "Call the `python` tool with code that inspects `current_frame`, `previous_frame`, `last_transition`, `history`, or `valid_actions` -- use `current_frame.segmentation` as the primary view, and `.ascii` only for a small specific region -- "
    "compare `previous_frame` to `current_frame` for the most recent change, "
    "derives a compact board summary, programs a small search or scorer over candidate actions or short sequences, "
    "then call `action(actions)` inside Python with the best valid action or ordered batch that your code selected. "
    f"{TOOL_CALL_FORMAT_GUIDANCE}"
)


def _normalize_valid_actions(valid_actions: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in valid_actions or []:
        engine_name = to_engine_action(value)
        name = to_model_action(engine_name or value)
        if name and name not in names:
            names.append(name)
    return names


def _format_valid_action_line(valid_actions: list[str] | None) -> str:
    names = _normalize_valid_actions(valid_actions)
    if not names:
        return "unknown"
    return ", ".join(names)


def _terminal_action_reason(result: dict[str, Any]) -> str | None:
    if result.get("run_complete"):
        return "run_complete"
    if result.get("game_over"):
        return "game_over"
    if result.get("level_completed"):
        return "level_completed"
    if result.get("done"):
        return "done"
    return None


def _terminal_action_stop_detail(reason: str | None) -> str:
    if reason == "run_complete":
        return "No further actions were executed because the run is already complete."
    if reason == "game_over":
        return (
            "No further actions were executed because the previous action reached GAME_OVER; "
            "the runner will auto-reset before the next analyzer turn."
        )
    if reason == "level_completed":
        return (
            "No further actions were executed because the previous action completed a level; "
            "re-ground on the new scene before acting again."
        )
    if reason == "done":
        return "No further actions were executed because the environment reported done."
    return "No further actions were executed because the previous action reached a terminal state."


def _display_action_number(action_num: int) -> int:
    return max(1, int(action_num) + 1)


def _request_tool_choice(tools: list[dict[str, Any]] | None) -> str | None:
    return "auto" if tools else None


def _trim_log_text(text: str, *, max_chars: int = _RESPONSE_META_MAX_CHARS) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    omitted = len(stripped) - max_chars
    return f"{stripped[:max_chars].rstrip()}\n... [truncated {omitted} chars]"


def _format_model_response_meta(
    *,
    finish_reason: str,
    reasoning: str,
    content: str,
    tool_calls: list[dict[str, Any]],
    tool_call_markup_in_text: bool,
    recovered_tool_calls_from_markup: bool,
    malformed_argument_errors: list[str],
) -> str:
    lines = [
        f"finish_reason: {finish_reason or '(empty)'}",
        f"tool_call_count: {len(tool_calls)}",
        f"content_chars: {len(content)}",
        f"reasoning_chars: {len(reasoning)}",
        f"tool_call_markup_in_text: {'yes' if tool_call_markup_in_text else 'no'}",
        f"tool_calls_recovered_from_markup: {'yes' if recovered_tool_calls_from_markup else 'no'}",
    ]
    if malformed_argument_errors:
        lines.append("tool_call_argument_issues:")
        lines.extend(f"- {issue}" for issue in malformed_argument_errors)
    if tool_calls:
        lines.append("raw_tool_calls:")
        lines.append(_trim_log_text(json.dumps(tool_calls, indent=2, ensure_ascii=True)))
    return "\n".join(lines)


def _build_system_prompt(*, tool_output_tokens: int) -> str:
    prompt = "You are a coding agent solving a grid-based puzzle game."
    prompt += GAME_OVERVIEW_ADDENDUM
    prompt += STRUCTURED_RUNTIME_STATE_ADDENDUM
    prompt += MULTIMODAL_CONTEXT_ADDENDUM
    prompt += VISUAL_GAME_ADDENDUM
    prompt += PYTHON_ADDENDUM
    prompt += COMPACT_TOOL_SESSION_ADDENDUM.format(tool_output_tokens=tool_output_tokens)
    return prompt


def game_over_note(auto_reset, level):
    actions = ", ".join(str(a) for a in (auto_reset.get("actions") or [])[-10:]) or "none recorded"
    return (
        f"GAME_OVER at step {auto_reset.get('step')} after actions [{actions}]; the level was "
        f"reset automatically and the board below is the fresh start of level {level}. What you "
        "learned about this level's mechanics still applies; only your plan needs revising."
    )


@dataclass(frozen=True)
class AnalyzerTurnResult:
    step_executed: bool
    retryable_failure: bool = False
    reasoning: str = ""
    yielded_control: bool = False


@dataclass(frozen=True)
class _ToolDispatchResult:
    content: str
    step_executed: bool = False


@dataclass
class _ChatCompletionResult:
    message: dict[str, Any]
    finish_reason: str = ""
    usage: dict[str, Any] | None = None


def _ascii_frame_view_payload(frame: Frame | None) -> dict[str, Any] | None:
    if frame is None:
        return None
    return {
        "ascii": frame.ascii,
        "step": frame.step,
        "level": frame.level,
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "grid": [list(row) for row in frame.grid],
    }


def _ascii_history_view_payload(history_entries: list[HistoryEntry]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for entry in history_entries:
        frame_payload = _ascii_frame_view_payload(entry.frame)
        if frame_payload is None:
            continue
        payload.append({"action": entry.action, "frame": frame_payload})
    return payload


def _render_transcript_section(label: str, content: str) -> str:
    rendered_content = content.strip()
    if not rendered_content:
        return ""
    return f"[{label}]\n{rendered_content}\n\n"


def _json_like_payload(value: Any) -> Any | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        return json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _render_scalar_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)


def _render_human_readable_lines(value: Any, *, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, item in value.items():
            key_text = str(key)
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key_text}:")
                lines.extend(_render_human_readable_lines(item, indent=indent + 2))
                continue
            if isinstance(item, str) and "\n" in item:
                multiline = item.splitlines() or [""]
                lines.append(f"{prefix}{key_text}: |")
                lines.extend(f"{prefix}  {line}" for line in multiline)
                continue
            lines.append(f"{prefix}{key_text}: {_render_scalar_value(item)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_render_human_readable_lines(item, indent=indent + 2))
                continue
            if isinstance(item, str) and "\n" in item:
                multiline = item.splitlines() or [""]
                lines.append(f"{prefix}- |")
                lines.extend(f"{prefix}  {line}" for line in multiline)
                continue
            lines.append(f"{prefix}- {_render_scalar_value(item)}")
        return lines
    if isinstance(value, str):
        if "\n" in value:
            multiline = value.splitlines() or [""]
            return [f"{prefix}|", *(f"{prefix}  {line}" for line in multiline)]
        return [f"{prefix}{value}"]
    return [f"{prefix}{_render_scalar_value(value)}"]


def _render_human_readable_value(value: Any) -> str:
    return "\n".join(_render_human_readable_lines(value))


def _render_jsonish_text(value: Any) -> str:
    parsed = _json_like_payload(value)
    if parsed is not None:
        return _render_human_readable_value(parsed)
    return _normalize_message_content(value) if not isinstance(value, str) else value.strip()


def _render_tool_parameter_text(value: Any) -> str:
    if isinstance(value, str):
        return value.rstrip("\n")
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=True)
    return str(value)


def _normalize_tool_call_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return json.loads(json.dumps(arguments))
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        if stripped.startswith("<tool_call>"):
            recovered_tool_calls = _recover_tool_calls_from_markup(stripped)
            if recovered_tool_calls:
                recovered_arguments = recovered_tool_calls[0].get("function", {}).get("arguments", "{}")
                return json.loads(str(recovered_arguments))
            return {}
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("tool call arguments must decode to a JSON object")
    raise ValueError("tool call arguments must be a JSON object or JSON object string")


def _render_tool_call_markup(tool_name: str, arguments: Any) -> str:
    name = str(tool_name or "").strip()
    if not name:
        return ""
    try:
        parsed_arguments = _normalize_tool_call_arguments(arguments)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""

    lines = ["<tool_call>", f"<function={name}>"]
    for parameter_name, parameter_value in parsed_arguments.items():
        lines.append(f"<parameter={parameter_name}>")
        rendered_value = _render_tool_parameter_text(parameter_value)
        if rendered_value:
            lines.extend(rendered_value.splitlines())
        lines.append("</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)


def _render_tool_result_display(content: Any) -> str:
    parsed = _json_like_payload(content) if isinstance(content, str) else (content if isinstance(content, dict) else None)
    if isinstance(parsed, dict):
        stdout = str(parsed.get("stdout", "") or "").rstrip("\n")
        error = str(parsed.get("error", "") or "").rstrip("\n")
        result = parsed.get("result")
        has_result = result not in (None, "", [], {})
        if stdout and not error and not has_result:
            return stdout

        blocks: list[str] = []
        if stdout:
            blocks.append(stdout)
        if has_result:
            rendered_result = _render_human_readable_value(result)
            if stdout:
                blocks.append(f"result:\n{rendered_result}")
            else:
                blocks.append(rendered_result)
        if error:
            if stdout or has_result:
                blocks.append(f"error:\n{error}")
            else:
                blocks.append(error)
        if blocks:
            return "\n\n".join(block for block in blocks if block.strip())

    return _render_jsonish_text(content)


def _render_prompt_log_message(message: dict[str, Any]) -> str:
    role = str(message.get("role", "")).strip().upper() or "UNKNOWN"
    header = f"[{role}]"
    tool_call_id = str(message.get("tool_call_id", "")).strip()
    if role == "TOOL" and tool_call_id:
        header = f"[TOOL RESULT: {tool_call_id}]"
    blocks = [header]

    content = _normalize_message_content(message.get("content", ""))
    if content:
        blocks.append(_render_tool_result_display(content) if role == "TOOL" else content)

    reasoning = _extract_reasoning_text(message)
    if reasoning:
        blocks.append("[REASONING]")
        blocks.append(reasoning)

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        for tool_call in tool_calls:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            name = str(function.get("name", "")).strip() or "unknown"
            blocks.append(f"[ASSISTANT TOOL CALL: {name}]")
            tool_call_id = str(tool_call.get("id", "")).strip()
            if tool_call_id:
                blocks.append(f"id: {tool_call_id}")
            rendered_tool_call = _render_tool_call_markup(name, function.get("arguments", "{}"))
            if rendered_tool_call:
                blocks.append(rendered_tool_call)
            else:
                raw_arguments = function.get("arguments", "{}")
                try:
                    parsed_arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    rendered_arguments = json.dumps(parsed_arguments, indent=2, ensure_ascii=True)
                except (TypeError, ValueError, json.JSONDecodeError):
                    rendered_arguments = str(raw_arguments)
                blocks.append("arguments:")
                blocks.append(rendered_arguments if rendered_arguments.strip() else "{}")

    return "\n".join(blocks)


def _write_prompt_log_snapshot(
    log_path: Path,
    *,
    model_id: str,
    base_url: str,
    display_action_num: int,
    analysis_step: int | None,
    request_index: int,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | None,
    transcript: str,
) -> None:
    rendered_messages = "\n\n".join(_render_prompt_log_message(message) for message in messages)
    rendered_tools: list[str] = []
    for tool in tools or []:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = str(function.get("name", "")).strip() or "unknown"
        description = str(function.get("description", "")).strip()
        if description:
            rendered_tools.append(f"- {name}: {description}")
        else:
            rendered_tools.append(f"- {name}")
    analysis_label = str(analysis_step) if analysis_step is not None else "n/a"
    transcript_text = transcript.strip()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("LATEST MODEL CALL SNAPSHOT\n")
        f.write(f"model: {model_id}\n")
        f.write(f"base_url: {base_url}\n")
        f.write(f"analysis_step: {analysis_label}\n")
        f.write(f"action: {display_action_num}\n")
        f.write(f"request_index_within_turn: {request_index}\n")
        f.write(f"message_count: {len(messages)}\n")
        f.write(f"tool_choice: {tool_choice or '(none)'}\n")
        f.write("\n[AVAILABLE TOOLS]\n")
        f.write("\n".join(rendered_tools) if rendered_tools else "(none)")
        f.write("\n\n[MODEL INPUT]\n")
        f.write(rendered_messages.strip())
        f.write("\n\n[TURN TRANSCRIPT SO FAR]\n")
        f.write(transcript_text)
        f.write("\n")


def _normalize_message_content(content: Any) -> str:
    def _strip_think_tags(text: str) -> str:
        cleaned = _THINK_TAG_RE.sub("", text)
        cleaned = "\n".join(line for line in cleaned.splitlines() if line.strip())
        return cleaned.strip()

    if isinstance(content, str):
        return _strip_think_tags(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return _strip_think_tags("\n".join(part for part in parts if part))
    return ""


def _extract_reasoning_text(message: dict[str, Any]) -> str:
    reasoning = message.get("reasoning")
    if reasoning in (None, ""):
        reasoning = message.get("reasoning_content", "")
    return _normalize_message_content(reasoning)


def _is_context_length_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "maximum context length" in message
        or "reduce the length of the input prompt" in message
        or "parameter=input_tokens" in message
        or '"param":"input_tokens"' in message
    )


def verbatim_history_text(field, message, normalized):
    """The reply's ``field`` ("reasoning" or "content") as the server sent it, ``<think>`` tags removed only
    (reasoning_history "verbatim", 29 September: the next prompt continues the generated tokens exactly, blank lines
    and paragraph breaks kept). ``normalized`` decides only whether the field is kept at all."""
    if field == "reasoning":
        raw = message.get("reasoning")
        if raw in (None, ""):
            raw = message.get("reasoning_content")
    else:
        raw = message.get(field)
    if not isinstance(raw, str) or not normalized:
        return normalized
    return _THINK_TAG_RE.sub("", raw)


# --- token estimate -------------------------------------------------------------------------------------------------

def _text_tokens(text, chars_per_token=3.0):
    return int(len(text) / chars_per_token) + 1


# Qwen3-VL vision tokens (the served Qwen3.6/3.8 ``preprocessor_config.json``, sha256 27225450...: patch 16, merge 2,
# shortest_edge 65,536 px, longest_edge 16,777,216 px): the processor's ``smart_resize`` rounds each side to a multiple
# of 32 px (and rescales below the minimum or above the maximum pixel count), then one token per 32x32 pixels, plus
# the vision start/end markers. The v5b run confirmed 64 for the loop's 256x256 board images (64x64 cells at
# MULTIMODAL_UPSCALE 4); a 640x640 board (the arena's dfr_harness, upscale 10) is 400.
VISION_PIXELS_PER_TOKEN = 32
VISION_MARKER_TOKENS = 2
VISION_MIN_PIXELS = 65536
VISION_MAX_PIXELS = 16777216


def smart_resize(height, width, factor=VISION_PIXELS_PER_TOKEN, min_pixels=VISION_MIN_PIXELS,
                 max_pixels=VISION_MAX_PIXELS):
    """``(height, width)`` after the Qwen2-VL image processor's resize (transformers ``smart_resize``)."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def vision_tokens(width, height):
    """Vision tokens of a ``width`` x ``height`` image as the served processor counts them, plus the markers."""
    h_bar, w_bar = smart_resize(height, width)
    return (h_bar // VISION_PIXELS_PER_TOKEN) * (w_bar // VISION_PIXELS_PER_TOKEN) + VISION_MARKER_TOKENS


def png_dimensions(data_url):
    """``(width, height)`` from a PNG data URL's IHDR chunk, or ``None``."""
    if not isinstance(data_url, str):
        return None
    head, sep, payload = data_url.partition(",")
    if not sep or "image/png" not in head:
        return None
    try:
        raw = base64.b64decode(payload[:64] + "=" * (-len(payload[:64]) % 4), validate=False)
    except (binascii.Error, ValueError):
        return None
    if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", raw[16:24])
    return (width, height) if width and height else None


def image_tokens_for(part, fallback):
    """Vision tokens of one ``image_url`` part from its PNG geometry; ``fallback`` otherwise."""
    url = (part.get("image_url") or {}).get("url") if isinstance(part, dict) else None
    dims = png_dimensions(url)
    if dims is None:
        return int(fallback)
    width, height = dims
    return vision_tokens(width, height)


class ToolAgent:
    """The analyzer of one game; ``owner`` is its ``DuckAgent``, ``loop`` the event loop."""

    # Vision tokens per board image when a PNG cannot be measured, and characters per text token: defaults until the
    # the tokenizer calibration (DuckAgent.calibrate) sets ``_image_tokens`` / ``_chars_per_token``.
    IMAGE_TOKENS = 400
    CHARS_PER_TOKEN = 3.0

    def __init__(self, owner, loop, *, model: str, base_url: str) -> None:
        self._owner, self._loop = owner, loop
        self._model_id, self._base_url = model, base_url
        # The maintainer (DuckAgent.analyzer, then the arena) sets the output cap, reply reserve and context budget.
        self._python_timeout = 30
        self._max_output_tokens: int | None = None
        self._reply_reserve_tokens = 512
        self._tool_output_tokens = _TOOL_OUTPUT_TOKENS
        self._tool_output_chars = _TOOL_OUTPUT_TOKENS * 4
        self._system_prompt = _build_system_prompt(tool_output_tokens=_TOOL_OUTPUT_TOKENS)
        self._request_safety_margin_tokens = _REQUEST_SAFETY_MARGIN_TOKENS
        self._context_budget_tokens = 32768 - self._reply_reserve_tokens - self._request_safety_margin_tokens
        self._chars_per_token = self.CHARS_PER_TOKEN
        self._image_tokens = self.IMAGE_TOKENS
        self._token_cache: dict[int, tuple] = {}
        self._history_messages: list[dict[str, Any]] = []
        self._step_env_callback = None
        self._current_valid_actions: list[str] = []
        self._last_step_summary: dict[str, Any] | None = None
        self._last_action_result: dict[str, Any] | None = None

    # --- requests ---------------------------------------------------------------------------------------------------

    def _chat_completion(self, messages, *, tools):
        owner = self._owner
        future = asyncio.run_coroutine_threadsafe(owner.chat(messages, tools), self._loop)
        owner.chat_future = future
        try:
            return future.result()
        finally:
            owner.chat_future = None

    def _history_text(self, field, message, normalized, markup_stripped):
        # Tool-call markup recovered from the text keeps the normalized text.
        return normalized if markup_stripped else verbatim_history_text(field, message, normalized)

    def _tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "python",
                    "description": _PYTHON_TOOL_DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string", "description": _PYTHON_CODE_DESCRIPTION}},
                        "required": ["code"],
                    },
                },
            }
        ]

    # --- token estimate (memoised per message object and calibration) -----------------------------------------------

    def _message_tokens(self, message):
        content = message.get("content")
        ratio = self._chars_per_token
        tokens = 0
        if isinstance(content, str):
            tokens += _text_tokens(content, ratio)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    tokens += image_tokens_for(part, self._image_tokens)
                elif isinstance(part, dict):
                    tokens += _text_tokens(str(part.get("text", "")), ratio)
                else:
                    tokens += _text_tokens(str(part), ratio)
        rest = {key: value for key, value in message.items() if key != "content"}
        if rest:
            tokens += _text_tokens(json.dumps(rest, ensure_ascii=True, sort_keys=True, default=str), ratio)
        return tokens

    def _estimate_request_input_tokens(self, messages, *, tools=None):
        """Linear, memoised estimate keyed by message identity and the current calibration."""
        cache = self._token_cache
        stamp = (self._chars_per_token, self._image_tokens)
        total = 0
        for message in messages:
            key = id(message)
            hit = cache.get(key)
            if hit is None or hit[0] is not message or hit[2] != stamp:
                hit = cache[key] = (message, self._message_tokens(message), stamp)
            total += hit[1]
        if tools:
            key = id(tools)
            hit = cache.get(key)
            if hit is None or hit[0] is not tools or hit[2] != stamp:
                hit = cache[key] = (
                    tools,
                    _text_tokens(json.dumps(tools, ensure_ascii=True, sort_keys=True, default=str), stamp[0]),
                    stamp,
                )
            total += hit[1]
        return total

    # --- prompts ----------------------------------------------------------------------------------------------------

    def _build_user_message(self, user_prompt: str, current_frame: Frame | None) -> dict[str, Any]:
        image_part = current_grid_image_part(current_frame)
        if image_part is None:
            message = {"role": "user", "content": user_prompt}
        else:
            step = current_frame.step if current_frame is not None else 0
            message = {
                "role": "user",
                "content": [{"type": "text", "text": f"{user_prompt}\n\n{IMAGE_LABEL.format(step=step)}"}, image_part],
            }
        # The arena's dfr_harness: the larger board image, diff and game-over images.
        return self._owner.observation_message(message, current_frame)

    def _summary_lines(self, action_num, current_frame, previous_step_summary):
        if previous_step_summary:
            lines: list[str] = []
            count = previous_step_summary.get("executed_count")
            try:
                normalized_count = int(count) if count is not None else None
            except (TypeError, ValueError):
                normalized_count = None
            action_label = "action" if normalized_count == 1 else "actions"
            lines.append(f"The code executed {normalized_count or 0} {action_label} in the previous sequence.")
            executed_actions = previous_step_summary.get("executed_actions")
            rendered_actions: list[str] = []
            if isinstance(executed_actions, list):
                rendered_actions = [str(name).strip() for name in executed_actions if str(name).strip()]
            if rendered_actions:
                action_prefix = "Executed actions (first 10):" if len(rendered_actions) > 10 else "Executed actions:"
                lines.append(f"{action_prefix} {', '.join(rendered_actions[:10])}.")
            else:
                lines.append("Executed actions: none.")
            if previous_step_summary.get("run_complete"):
                lines.append("You have completed the run!")
            elif previous_step_summary.get("level_transition"):
                lines.append("You have progressed to a new level!")
            else:
                lines.append("You are still on the same level.")
            if previous_step_summary.get("game_over"):
                lines.append("The game is over.")
            return lines
        if (current_frame is not None and current_frame.step > 0) or action_num > 0:
            return ["No previous action sequence was captured."]
        return ["No previous sequence has been executed yet."]

    def _build_user_prompt(
        self,
        action_num: int,
        *,
        valid_actions: list[str] | None,
        current_frame: Frame | None = None,
        history_entries: list[HistoryEntry] | None = None,
        previous_step_summary: dict[str, Any] | None = None,
    ) -> str:
        """The turn's user text: the previous sequence's outcome, the state line, the valid actions (the constant
        instructions are in the system prompt's per-turn discipline), then the arena's rewrites and notices."""
        owner = self._owner
        history_entries = history_entries or []
        current_step = max(current_frame.step if current_frame is not None else 0, max(0, action_num)) + 1
        current_level = current_frame.level if current_frame is not None else 1
        summary_level = None
        if previous_step_summary is not None:
            try:
                summary_level = int(previous_step_summary.get("level"))
            except (TypeError, ValueError):
                summary_level = None
        if summary_level is not None:
            current_level = max(current_level, summary_level)
        observed_max_level = max(
            [current_level, *[entry.frame.level for entry in history_entries if entry.frame is not None]],
            default=current_level,
        )
        lines = self._summary_lines(action_num, current_frame, previous_step_summary)
        state_line = f"Current state: step {current_step}, level {current_level}"
        if observed_max_level > current_level:
            state_line += f" out of observed max level {observed_max_level} so far"
        state_line += "."
        lines.extend([state_line, f"Valid actions right now: {_format_valid_action_line(valid_actions)}.",
                      "end of world model. "])
        if action_num == 0:
            lines.append(
                "Ground yourself in `current_frame` before acting, but start with a compact structural summary rather than restating the full frame."
            )
        else:
            lines.append(
                "Focus on what changed most recently in `history`, update the target environment change if needed, and separate gameplay-object changes from HUD-only changes."
            )
        text = "\n".join(lines)
        if "The game is over." in text and owner.obs.state != "GAME_OVER":
            note = game_over_note(
                owner.auto_reset or {"step": None, "actions": []},
                max(1, min(owner.obs.total_levels, owner.obs.levels + 1)),
            )
            if text.count(GAME_OVER_LINES) != 1:
                raise ValueError("Pinned Duck game-over lines changed")
            text = text.replace(GAME_OVER_LINES, note)
        # The arena's harness_fixes prompt text and dfr_harness additions.
        text = owner.rewrite_user_prompt(text, current_frame)
        notice = owner.stuck.notice(max(1, owner.obs.levels + 1), owner.tried_no_effect_here())
        if notice:
            owner.workspace.append("stuck_notice", step=owner.steps, text=notice)
            text += "\n" + notice
        extra = owner.turn_notice()  # the arena's hx_harness notices
        if extra:
            text += "\n" + extra
        return text

    # --- the python tool --------------------------------------------------------------------------------------------

    def _summarize_step_sequence(self, action_results: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not action_results:
            return None
        executed_results = [item for item in action_results if item.get("executed")]
        if not executed_results:
            return None

        total_executed = 0
        executed_actions: list[str] = []
        for item in executed_results:
            count = item.get("executed_count")
            try:
                parsed = int(count) if count is not None else 1
            except (TypeError, ValueError):
                parsed = 1
            total_executed += max(1, parsed)
            action_names = item.get("executed_actions")
            if isinstance(action_names, list):
                executed_actions.extend(str(name).strip() for name in action_names if str(name).strip())
            else:
                fallback_action = str(item.get("action_display") or "").strip()
                if fallback_action:
                    executed_actions.append(fallback_action)

        last = executed_results[-1]
        try:
            end_action_num = int(last.get("action_num"))
        except (TypeError, ValueError):
            end_action_num = None
        start_action_num = None
        if end_action_num is not None and total_executed > 0:
            start_action_num = max(1, end_action_num - total_executed + 1)

        return {
            "start_action_num": start_action_num,
            "end_action_num": end_action_num,
            "executed_count": total_executed,
            "executed_actions": executed_actions,
            "level": last.get("level"),
            "level_transition": any(bool(item.get("level_completed")) for item in executed_results),
            "run_complete": any(bool(item.get("run_complete")) for item in executed_results),
            "game_over": any(bool(item.get("game_over")) for item in executed_results),
            "board_changed": any(bool(item.get("board_changed")) for item in executed_results),
            "stop_reason": last.get("stop_reason"),
        }

    def _summarize_planned_actions(self, value: Any) -> Any:
        if isinstance(value, dict):
            compacted = {
                key: self._summarize_planned_actions(item)
                for key, item in value.items()
            }
            planned_actions = compacted.pop("planned_actions", None)
            if isinstance(planned_actions, list):
                compacted["planned_action_count"] = len(planned_actions)
                action_result = compacted.get("action_result")
                if isinstance(action_result, dict):
                    executed_count = action_result.get("executed_count")
                    try:
                        compacted["executed_action_count"] = int(executed_count)
                    except (TypeError, ValueError):
                        compacted["executed_action_count"] = 1 if action_result.get("executed") else 0
            return compacted
        if isinstance(value, list):
            return [self._summarize_planned_actions(item) for item in value]
        return value

    def _render_tool_payload(self, payload: dict[str, Any], *, truncate_fields: tuple[str, ...] = ()) -> str:
        result = self._summarize_planned_actions(dict(payload))
        truncated = False
        for field in truncate_fields:
            value = result.get(field)
            if isinstance(value, str):
                result[field], field_truncated = self._trim_tool_text(value)
                truncated = truncated or field_truncated
        if truncated:
            result["truncated"] = True
            result["truncation_note"] = (
                f"Tool output was cut off to stay within the ~{self._tool_output_tokens}-token response budget."
            )
        return json.dumps(result, indent=2)

    def _normalize_python_actions(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, dict):
            items = [value]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(f"Action {index} uses legacy MOUSE x/y fields; use row and col.")
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized

    def _compact_action_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        compact = {
            "executed": bool(payload.get("executed")),
            "action_num": payload.get("action_num"),
            "level": payload.get("level"),
            "score": payload.get("score"),
            "reward": payload.get("reward"),
            "state": payload.get("state"),
            "valid_actions": payload.get("valid_actions", []),
            "board_changed": bool(payload.get("board_changed")),
            "done": bool(payload.get("done")),
            "level_completed": bool(payload.get("level_completed")),
            "game_over": bool(payload.get("game_over")),
            "run_complete": bool(payload.get("run_complete")),
            "action_display": payload.get("action_display") or payload.get("action_name"),
        }
        executed_actions = payload.get("executed_actions")
        if isinstance(executed_actions, list) and executed_actions:
            compact["executed_actions"] = [str(action).strip() for action in executed_actions if str(action).strip()]
        elif compact.get("action_display"):
            compact["executed_actions"] = [str(compact["action_display"]).strip()]
        batch_size = int(payload.get("requested_count") or payload.get("executed_count") or 1)
        if batch_size > 1 or bool(payload.get("stopped_early")):
            compact["requested_count"] = payload.get("requested_count", batch_size)
            compact["executed_count"] = payload.get("executed_count", batch_size)
            compact["stopped_early"] = bool(payload.get("stopped_early"))
        if payload.get("stop_reason"):
            compact["stop_reason"] = payload.get("stop_reason")
        if payload.get("stop_detail"):
            compact["stop_detail"] = payload.get("stop_detail")
        for timing_key in ("run_elapsed_seconds", "time_remaining_seconds"):
            if timing_key in payload:
                compact[timing_key] = payload.get(timing_key)
        if payload.get("error"):
            compact["error"] = payload.get("error")
        for key in RECEIPT_KEYS:
            if key in payload:
                compact[key] = payload[key]
        return compact

    def _run_python_tool(self, state_path: Path, arguments: dict[str, Any]) -> _ToolDispatchResult:
        owner = self._owner
        replaying = getattr(owner, "replaying", None)
        limit = 600 if replaying is not None and replaying() else 30  # a replayed prefix: host load must not cut it
        self._python_timeout = max(0.01, min(limit, owner.remaining()))
        owner.sandbox.cpu_seconds = int(owner.remaining()) + 60
        output = owner.run_python_tool(self._execute_python, state_path, arguments)
        result = json.loads(output.content)
        owner.workspace.append(
            "duck_python",
            step=owner.steps,
            code=arguments.get("code"),
            result=result,
            sandbox_calls=owner.sandbox.calls,
            restarts=owner.sandbox.restarts,
            history_synced=owner.sandbox.synced,
        )
        return output

    def _execute_python(self, state_path: Path, arguments: dict[str, Any]) -> _ToolDispatchResult:
        """One ``python`` call in the sandbox; ``action(...)`` inside it steps the game through the maintainer."""
        owner = self._owner
        code = str(arguments.get("code", "")).rstrip()
        if not code:
            return _ToolDispatchResult(json.dumps({"error": "python requires a non-empty `code` string."}, indent=2))
        try:
            compile(code, "<python_tool>", "exec")
        except SyntaxError as exc:
            return _ToolDispatchResult(json.dumps({"error": f"Python syntax error: {exc}"}, indent=2))

        valid_actions = list(_normalize_valid_actions(self._current_valid_actions))

        def _serialized_runtime_state(
            *,
            next_valid_actions: list[str] | None = None,
            last_action_result: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            refreshed_frame, refreshed_history = owner.runtime_state()
            current_frame_payload = owner.frame_view_payload(refreshed_frame)
            if isinstance(next_valid_actions, list):
                sanitized_actions = [str(item).strip() for item in next_valid_actions if str(item).strip()]
            else:
                sanitized_actions = list(valid_actions)
            persisted_action_result = (
                last_action_result
                if isinstance(last_action_result, dict)
                else self._last_action_result
            )
            return {
                "current_frame": current_frame_payload,
                "history": owner.ascii_history(refreshed_history),
                "valid_actions": sanitized_actions,
                "last_action_result": (
                    dict(persisted_action_result)
                    if isinstance(persisted_action_result, dict)
                    else {}
                ),
            }

        terminal_action_result: dict[str, Any] | None = None

        def _handle_action(actions: list[dict[str, Any]]) -> dict[str, Any]:
            nonlocal terminal_action_result
            if self._step_env_callback is None:
                raise RuntimeError("action(actions) is not available in this session.")
            normalized_actions = self._normalize_python_actions(actions)
            if terminal_action_result is not None:
                reason = _terminal_action_reason(terminal_action_result) or "terminal_state"
                compact_payload = {
                    "executed": False,
                    "action_num": terminal_action_result.get("action_num"),
                    "level": terminal_action_result.get("level"),
                    "score": terminal_action_result.get("score"),
                    "reward": 0.0,
                    "state": terminal_action_result.get("state"),
                    "valid_actions": [],
                    "board_changed": False,
                    "done": bool(terminal_action_result.get("done")),
                    "level_completed": bool(terminal_action_result.get("level_completed")),
                    "game_over": bool(terminal_action_result.get("game_over")),
                    "run_complete": bool(terminal_action_result.get("run_complete")),
                    "requested_count": len(normalized_actions),
                    "executed_count": 0,
                    "stopped_early": True,
                    "stop_reason": f"previous_{reason}",
                    "stop_detail": _terminal_action_stop_detail(reason),
                }
                self._last_action_result = dict(compact_payload)
                return {
                    "action_result": compact_payload,
                    "state": _serialized_runtime_state(
                        next_valid_actions=[],
                        last_action_result=compact_payload,
                    ),
                }
            raw_payload = self._step_env_callback({"actions": normalized_actions})
            if not isinstance(raw_payload, dict):
                raise RuntimeError("action(actions) did not return a JSON-like payload.")
            compact_payload = self._compact_action_result(raw_payload)
            next_valid_actions = raw_payload.get("valid_actions")
            if isinstance(next_valid_actions, list):
                self._current_valid_actions = _normalize_valid_actions(next_valid_actions)
            if compact_payload.get("executed") and _terminal_action_reason(compact_payload):
                terminal_action_result = compact_payload
            self._last_action_result = dict(compact_payload)
            return {
                "action_result": compact_payload,
                "state": _serialized_runtime_state(
                    next_valid_actions=next_valid_actions if isinstance(next_valid_actions, list) else None,
                    last_action_result=compact_payload,
                ),
            }

        sandbox_result = owner.sandbox_runner(
            code=code,
            timeout_seconds=self._python_timeout,
            initial_state=_serialized_runtime_state(),
            action_handler=_handle_action,
        )

        action_results = [
            item
            for item in sandbox_result.get("action_results") or []
            if isinstance(item, dict)
        ]
        payload: dict[str, Any] = {"tool": "python"}
        rendered_stdout = str(sandbox_result.get("stdout", "") or "")
        rendered_error = str(sandbox_result.get("error", "") or "")
        if rendered_error:
            payload["error"] = rendered_error
            if rendered_stdout:
                payload["stdout"] = rendered_stdout
        else:
            payload["returncode"] = 0
            if rendered_stdout:
                payload["stdout"] = rendered_stdout
            elif sandbox_result.get("result") is not None:
                payload["result"] = sandbox_result.get("result")
            elif action_results:
                if len(action_results) == 1:
                    payload["result"] = action_results[-1]
                else:
                    payload["result"] = {
                        "action_calls": len(action_results),
                        "last_action_result": action_results[-1],
                    }

        step_executed = any(bool(item.get("executed")) for item in action_results)
        if step_executed:
            self._last_step_summary = self._summarize_step_sequence(action_results)
        return _ToolDispatchResult(
            self._render_tool_payload(payload, truncate_fields=("stdout", "error", "result")),
            step_executed=step_executed,
        )

    def _dispatch_tool(self, state_path: Path, name: str, arguments: dict[str, Any]) -> _ToolDispatchResult:
        if name == "python":
            return self._run_python_tool(state_path, arguments)
        return _ToolDispatchResult(json.dumps({"error": f"Unknown tool: {name}"}, indent=2))

    # --- one turn ---------------------------------------------------------------------------------------------------

    def analyze(self, state_path, action_num, *, valid_actions, step_env, transcript_path, analysis_step,
                should_stop) -> AnalyzerTurnResult | None:
        """One turn. The transcript goes to ``transcript_path`` through one buffered handle (flushed every 2 s and at
        the end), the latest request's prompt-log snapshot is written once when the turn ends."""
        files: dict[Path, list] = {}
        snapshot: dict[str, Any] = {}

        def append_section(path, label, content):
            section = _render_transcript_section(label, content)
            if not section:
                return
            if path not in files:
                files[path] = [logs.enter_context(open(path, "a", encoding="utf-8", buffering=1 << 16)), 0.0]
            file, flushed = files[path]
            file.write(section)
            now = time.monotonic()
            if now - flushed >= 2.0:
                file.flush()
                files[path][1] = now

        self._token_cache = {}
        with ExitStack() as logs:  # the complete transcript survives errors and interrupted turns, too
            try:
                return self._turn(state_path, action_num, valid_actions, step_env, transcript_path, analysis_step,
                                  should_stop, append_section, snapshot)
            finally:
                if snapshot:
                    _write_prompt_log_snapshot(*snapshot["args"], **snapshot["data"])
                    self._owner.prompt_log_writes += 1

    def _turn(self, state_path, action_num, valid_actions, step_env, analyzer_log, analysis_step, should_stop,
              append_section, snapshot):
        self._step_env_callback = step_env
        self._current_valid_actions = _normalize_valid_actions(valid_actions)

        def record_snapshot(*args, **data):  # written once, when the turn ends (analyze)
            snapshot.clear()
            snapshot.update(args=args, data=data)

        prompt_log = state_path.parent / "prompts" / "prompt.log"
        current_frame, history_entries = self._owner.runtime_state()
        user_prompt = self._build_user_prompt(
            action_num,
            valid_actions=valid_actions,
            current_frame=current_frame,
            history_entries=history_entries,
            previous_step_summary=self._last_step_summary,
        )
        display_action_num = _display_action_number(action_num)

        with open(analyzer_log, "a", encoding="utf-8") as f:
            step_label = f"analysis_step={analysis_step} | " if analysis_step is not None else ""
            transcript_header = (
                f"\n--- {step_label}action={display_action_num} | "
                f"{time.strftime('%H:%M:%S')} | tool-agent ---\n"
            )
            f.write(transcript_header)
        transcript_parts = [transcript_header]

        def append_transcript(label: str, content: str) -> None:
            append_section(analyzer_log, label, content)
            transcript_parts.append(_render_transcript_section(label, content))

        append_transcript("SYSTEM PROMPT", self._system_prompt)
        append_transcript("USER PROMPT", user_prompt)

        previous_history_messages = list(self._history_messages)
        preserve_history = True
        messages: list[dict[str, Any]] = self._trim_messages_for_context(
            [{"role": "system", "content": self._system_prompt}, *self._history_messages,
             self._build_user_message(user_prompt, current_frame)],
            tools=self._tools(),
            preserve_recent=1,
        )
        step_executed = False
        captured_reasoning = ""
        latest_request_messages: list[dict[str, Any]] | None = None
        latest_request_tools: list[dict[str, Any]] | None = None
        latest_request_tool_choice: str | None = None
        latest_request_index = 0
        yielded_control_reason: str | None = None

        def control_yield_reason() -> str | None:
            try:
                if should_stop():
                    return "stop_requested"
            except Exception as exc:
                log.warning("analyzer stop check failed at action %d: %s", display_action_num, exc)
            return None

        def snapshot_latest():
            record_snapshot(
                prompt_log,
                model_id=self._model_id,
                base_url=self._base_url,
                display_action_num=display_action_num,
                analysis_step=analysis_step,
                request_index=latest_request_index,
                messages=latest_request_messages,
                tools=latest_request_tools,
                tool_choice=latest_request_tool_choice,
                transcript="".join(transcript_parts),
            )

        try:
            turn_count = 0
            while True:
                yielded_control_reason = control_yield_reason()
                if yielded_control_reason is not None:
                    break
                turn_count += 1
                tools = self._tools()
                tool_choice = _request_tool_choice(tools)
                messages = self._trim_messages_for_context(messages, tools=tools)
                latest_request_messages = json.loads(json.dumps(messages))
                latest_request_tools = json.loads(json.dumps(tools))
                latest_request_tool_choice = tool_choice
                latest_request_index = turn_count
                snapshot_latest()
                try:
                    result = self._chat_completion(messages, tools=tools)
                except requests.RequestException as exc:
                    if not _is_context_length_error(exc):
                        raise
                    trimmed_messages = self._trim_messages_for_context(
                        messages,
                        tools=tools,
                        extra_safety_tokens=_CONTEXT_OVERFLOW_RETRY_TRIM_TOKENS,
                    )
                    if trimmed_messages == messages:
                        trimmed_messages = self._force_reduce_messages(messages)
                    if trimmed_messages == messages:
                        raise
                    append_transcript(
                        "ANALYZER STATUS",
                        "context_overflow_recovered: dropped older history after server rejected the request as too long.",
                    )
                    messages = trimmed_messages
                    continue
                raw_reasoning = _extract_reasoning_text(result.message)
                raw_content = _normalize_message_content(result.message.get("content", ""))
                tool_calls = json.loads(json.dumps(result.message.get("tool_calls") or []))
                tool_call_markup_in_text = _contains_tool_call_markup(raw_reasoning, raw_content)
                recovered_tool_calls_from_markup = False
                if not tool_calls and tool_call_markup_in_text:
                    tool_calls = _recover_tool_calls_from_markup(raw_reasoning, raw_content)
                    recovered_tool_calls_from_markup = bool(tool_calls)
                reasoning = _strip_tool_call_markup(raw_reasoning) if tool_call_markup_in_text else raw_reasoning
                content = _strip_tool_call_markup(raw_content) if tool_call_markup_in_text else raw_content
                malformed_argument_errors: list[str] = []
                for tool_call in tool_calls:
                    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    tool_name = str(function.get("name", "")).strip() or "unknown"
                    raw_arguments = function.get("arguments", "{}")
                    if isinstance(raw_arguments, str):
                        try:
                            json.loads(raw_arguments)
                        except json.JSONDecodeError as exc:
                            malformed_argument_errors.append(f"{tool_name}: invalid JSON arguments ({exc})")
                response_meta = _format_model_response_meta(
                    finish_reason=result.finish_reason,
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tool_calls,
                    tool_call_markup_in_text=tool_call_markup_in_text,
                    recovered_tool_calls_from_markup=recovered_tool_calls_from_markup,
                    malformed_argument_errors=malformed_argument_errors,
                )
                append_transcript("MODEL RESPONSE META", response_meta)
                assistant_message: dict[str, Any] = {"role": "assistant"}

                if reasoning:
                    captured_reasoning = reasoning
                    append_transcript("THINKING", reasoning)
                    assistant_message["reasoning"] = self._history_text(
                        "reasoning", result.message, reasoning, tool_call_markup_in_text
                    )

                if not tool_calls:
                    if content:
                        append_transcript("ASSISTANT", content)
                        assistant_message["content"] = self._history_text(
                            "content", result.message, content, tool_call_markup_in_text
                        )
                    elif reasoning:
                        assistant_message["content"] = None

                    if content or reasoning:
                        messages.append(assistant_message)
                    yielded_control_reason = control_yield_reason()
                    if yielded_control_reason is not None:
                        break
                    followup_prefix = MARKUP_FOLLOWUP_PREFIX if tool_call_markup_in_text else FOLLOWUP_PREFIX
                    followup_prompt = followup_prefix + FOLLOWUP
                    append_transcript("USER PROMPT", followup_prompt)
                    messages.append({"role": "user", "content": followup_prompt})
                    continue

                if content:
                    append_transcript("ASSISTANT", content)
                    assistant_message["content"] = self._history_text(
                        "content", result.message, content, tool_call_markup_in_text
                    )
                assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)

                for tool_index, tool_call in enumerate(tool_calls):
                    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    tool_name = str(function.get("name", "")).strip()
                    raw_args = function.get("arguments", "{}")
                    try:
                        if isinstance(raw_args, str):
                            arguments = json.loads(raw_args)
                        elif isinstance(raw_args, dict):
                            arguments = json.loads(json.dumps(raw_args))
                        else:
                            arguments = {}
                    except json.JSONDecodeError:
                        arguments = {}
                    rendered_tool_call = _render_tool_call_markup(tool_name, raw_args)
                    append_transcript(
                        f"TOOL CALL: {tool_name}",
                        rendered_tool_call or (json.dumps(arguments, indent=2) if arguments else "{}"),
                    )
                    dispatch = self._dispatch_tool(state_path, tool_name, arguments)
                    if dispatch.step_executed:
                        step_executed = True
                    append_transcript(f"TOOL RESULT: {tool_name}", _render_tool_result_display(dispatch.content))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", ""),
                            "content": dispatch.content,
                        }
                    )
                    if dispatch.step_executed:
                        if tool_index < len(tool_calls) - 1:
                            preserve_history = False
                        break
                    yielded_control_reason = control_yield_reason()
                    if yielded_control_reason is not None:
                        if tool_index < len(tool_calls) - 1:
                            preserve_history = False
                        break
                if yielded_control_reason is not None:
                    break
                if step_executed:
                    break

        except requests.RequestException as exc:
            append_transcript("ANALYZER STATUS", f"request_error: {exc}")
            preserve_history = False
            if latest_request_messages is not None:
                snapshot_latest()
            log.warning("analyzer request failed at action %d: %s", display_action_num, exc)
            return AnalyzerTurnResult(step_executed=False, retryable_failure=True, reasoning=captured_reasoning)
        except Exception as exc:
            append_transcript("ANALYZER STATUS", f"error: {exc}")
            preserve_history = False
            if latest_request_messages is not None:
                snapshot_latest()
            log.warning("analyzer failed at action %d: %s", display_action_num, exc)
            return None
        finally:
            if preserve_history:
                self._history_messages = self._persistent_history_messages(messages, tools=self._tools())
            else:
                self._history_messages = previous_history_messages
            self._step_env_callback = None
            self._current_valid_actions = []

        if step_executed:
            status_message = "Step executed."
        elif yielded_control_reason is not None:
            status_message = f"Yielded control to solver: {yielded_control_reason}."
        else:
            status_message = "No action(...) call was captured."

        status = (
            f"model: {self._model_id}\n"
            f"base_url: {self._base_url}\n"
            f"max_output_tokens: {self._max_output_tokens if self._max_output_tokens is not None else 'server default'}\n"
            f"reply_reserve_tokens: {self._reply_reserve_tokens}\n"
            f"context_budget_tokens: {self._context_budget_tokens}\n"
            f"request_safety_margin_tokens: {self._request_safety_margin_tokens}\n"
            f"tool_output_tokens: {self._tool_output_tokens}\n"
            f"yield_seconds: disabled\n"
            f"available_tools: python\n"
            f"python_timeout_seconds: {self._python_timeout}\n"
            f"history_messages: {len(self._history_messages)}\n"
            f"step_executed: {step_executed}\n"
            f"message: {status_message}"
        )
        append_transcript("ANALYZER STATUS", status)
        if latest_request_messages is not None:
            snapshot_latest()
        return AnalyzerTurnResult(
            step_executed=step_executed,
            reasoning=captured_reasoning,
            yielded_control=yielded_control_reason is not None,
        )
