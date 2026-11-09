"""Unmodified portable UI-Mate prompt/parser/history functions; see SOURCE.json.

Copyright (C) 2026 Tencent. Apache-2.0; see LICENSE.
"""
import json
import re
from typing import Dict, List, Optional, Tuple


COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."


def collapse_messages(
    messages: List[Dict],
    images_to_keep: Optional[int] = 10,
    min_removal_threshold: int = 10,
    collapse_text: str = COLLAPSED_SCREENSHOT_TEXT,
) -> Tuple[List[Dict], bool]:
    """Drop the oldest screenshots from user messages to bound context size.

    The step-0 screenshot (the initial state of the task) is never dropped, and
    removals happen in chunks of ``min_removal_threshold`` so that the shared
    prefix stays stable for vLLM's prefix cache.
    """
    if not messages or images_to_keep is None:
        return messages, False

    total_images = 0
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                total_images += 1

    images_to_remove = total_images - images_to_keep
    images_to_remove -= images_to_remove % min_removal_threshold

    if images_to_remove <= 0:
        return messages, False

    remaining_to_remove = images_to_remove
    collapsed_any = False
    global_img_idx = -1

    for msg in messages:
        if remaining_to_remove <= 0:
            break
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        has_text = any(
            isinstance(block, dict) and block.get("type") == "text" for block in content
        )
        new_content: List[Dict] = []
        removed_here = 0
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                global_img_idx += 1
                if global_img_idx == 0:
                    # step-0 screenshot is pinned for the whole episode
                    new_content.append(block)
                    continue
                if remaining_to_remove > 0:
                    remaining_to_remove -= 1
                    removed_here += 1
                    continue
            new_content.append(block)

        if removed_here > 0:
            collapsed_any = True
            msg["content"] = _replace_with_placeholder(new_content, has_text, collapse_text)

        if remaining_to_remove <= 0:
            break

    return messages, collapsed_any


def _replace_with_placeholder(
    new_content: List[Dict], has_text: bool, collapse_text: str
) -> List[Dict]:
    """Swap a stripped-down user message for its collapse placeholder."""
    remaining_text = "".join(
        block.get("text", "")
        for block in new_content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()

    text_normalized = (
        remaining_text.replace("\n", "").replace(" ", "").replace("\t", "").replace("\r", "")
    )
    is_empty_or_xml_only = (
        not remaining_text
        or text_normalized == "<tool_response></tool_response>"
        or text_normalized == ""
    )

    if not has_text or is_empty_or_xml_only:
        if "<tool_response>" in remaining_text:
            placeholder_text = "<tool_response>\n" + collapse_text + "\n</tool_response>"
        else:
            placeholder_text = collapse_text
        return [{"type": "text", "text": placeholder_text}]
    return [{"type": "text", "text": collapse_text}] + new_content


def build_action_description() -> str:
    return """* `left_click`: Click the left mouse button at the specified (x, y) coordinate.
* `right_click`: Click the right mouse button at the specified (x, y) coordinate.
* `middle_click`: Click the middle mouse button at the specified (x, y) coordinate.
* `double_click`: Double-click the left mouse button at the specified (x, y) coordinate.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) coordinate.
* `drag`: Click and drag the mouse cursor from its current position to the specified (x, y) coordinate.
* `mouse_move`: Move the cursor to the specified (x, y) coordinate without clicking.
* `type`: Type a specified string of text.
* `hotkey`: Press a combination of keys (e.g., ["ctrl", "v"]).
* `press`: Press a single key or a sequence of keys, provided as an array of strings (e.g., ["backspace"], ["enter"], ["a", "b", "c"]).
* `key_down`: Press and HOLD the specified key(s) down in order (no release). Use this for stateful holds like holding Shift while clicking.
* `key_up`: Release the specified key(s) in reverse order.
* `scroll`: Scroll the mouse wheel by a specified number of pixels. Use "direction" to specify vertical (default, positive for up, negative for down) or horizontal (positive for right, negative for left) scrolling.
* `wait`: Pause execution for a specified number of seconds.
* `call_user`: Ask the user for information or confirmation. Use this when you genuinely need user input, or when the task cannot be completed (in that case clearly state why it is infeasible).
* `finished`: Terminate the task and indicate whether it was a 'success' or 'failure'."""


def build_tools_def(description_prompt: str) -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": description_prompt,
            "parameters": {
                "properties": {
                    "action": {
                        "description": build_action_description(),
                        "enum": [
                            "left_click", "right_click", "middle_click",
                            "double_click", "triple_click", "drag", "mouse_move",
                            "type", "hotkey", "press", "key_down", "key_up",
                            "scroll", "wait", "call_user", "finished",
                        ],
                        "type": "string",
                    },
                    "coordinate": {
                        "description": "The (x, y) coordinates (0-999). Required for: clicks, mouse_move, drag.",
                        "type": "array",
                    },
                    "text": {
                        "description": (
                            "The text to type, or the message to the user."
                            " Required for `action=type` and `action=call_user`."
                        ),
                        "type": "string",
                    },
                    "keys": {
                        "description": (
                            "An array of key names (e.g. ['a'], ['ctrl', 'c'])."
                            " Required for: hotkey, press, key_down, key_up."
                        ),
                        "type": "array",
                    },
                    "pixels": {
                        "description": "The number of pixels to scroll. Required only for `action=scroll`.",
                        "type": "number",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["vertical", "horizontal"],
                        "description": (
                            "The scroll direction. 'vertical' (default) for up/down"
                            " scrolling, 'horizontal' for left/right scrolling."
                            " Required only for `action=scroll`."
                        ),
                    },
                    "time": {
                        "description": "Seconds to wait. Required only for `action=wait`.",
                        "type": "number",
                    },
                    "status": {
                        "description": "The outcome of the task. Required only for `action=finished`.",
                        "type": "string",
                        "enum": ["success", "failure"],
                    },
                },
                "required": ["action"],
                "type": "object",
            },
        },
    }


def build_tools_and_format_block(tools_def: Dict) -> str:
    return (
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        "<tools>\n"
        + json.dumps(tools_def)
        + "\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\n"
        "value_1\n"
        "</parameter>\n"
        "<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\n"
        "that can span\n"
        "multiple lines\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified format: an inner"
        " <function=...></function> block must be nested within"
        " <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural"
        " language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal"
        " with your current knowledge and do not tell the user about function calls\n"
        "</IMPORTANT>"
    )


RESPONSE_FORMAT = (
    "Response format for every step:\n"
    "1) Thought: A single <think>...</think> block containing step by step progress"
    " assessment and next action analysis.\n"
    "2) Action: A single <action>...</action> block containing a short imperative describing what to do in the UI.\n"
    "3) Tool Execution: A single or multiple <tool_call>...</tool_call> blocks.\n\n"
    "Rules:\n"
    "- Output exactly in the order: <think>...</think>, <action>...</action>, <tool_call>...</tool_call>.\n"
    "- From a first-person perspective, systematically assess progress and errors,"
    " evaluate potential next steps, and precisely plan text inputs"
    " (cursor position and expected outcomes)\n"
    "- Be brief for Action: one sentence for action description.\n"
    "- Do not output anything else outside those parts.\n"
    "- If finishing, use action=finished in the tool call. If the task is infeasible, finish with status=failure."
)


def extract_action_text(response: str) -> str:
    match = re.search(r"<action>\s*(.*?)\s*</action>", response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def compact_response_for_history(response: str, include_thinking: bool = False) -> str:
    """Trim a past response before replaying it as assistant history.

    Keeping the thinking means replaying the response from ``<think>`` onwards;
    otherwise history starts at ``<action>``.
    """
    tag = r"<think\b[^>]*>" if include_thinking else r"<action\b[^>]*>"
    match = re.search(tag, response, re.IGNORECASE)
    if not match:
        return response
    return response[match.start():].strip()


def parse_xml_tool_call(xml_content: str) -> Optional[Dict]:
    """Parse one XML tool call into a flat params dict."""
    func_match = re.search(r"<function=([^>]+)>", xml_content)
    if not func_match or func_match.group(1) != "computer_use":
        return None

    params: Dict = {}
    for match in re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", xml_content, re.DOTALL):
        name = match.group(1)
        value = match.group(2).strip()
        if value.startswith("[") or value.startswith("{"):
            try:
                params[name] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        params[name] = value
    return params


def extract_xml_tool_calls(response: str) -> List[Dict]:
    results = []
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL):
        params = parse_xml_tool_call(match.group(1))
        if params:
            results.append(params)
    return results


def _clean_keys(raw_keys) -> List:
    keys = raw_keys if isinstance(raw_keys, list) else [raw_keys]
    cleaned_keys = []
    for key in keys:
        if not isinstance(key, str):
            cleaned_keys.append(key)
            continue
        if key.startswith("keys=["):
            key = key[6:]
        if key.endswith("]"):
            key = key[:-1]
        if key.startswith("['") or key.startswith('["'):
            key = key[2:] if len(key) > 2 else key
        if key.endswith("']") or key.endswith('"]'):
            key = key[:-2] if len(key) > 2 else key
        cleaned_keys.append(key.strip())
    return cleaned_keys
