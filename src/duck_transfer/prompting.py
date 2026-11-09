"""The board-image label of a user turn (``inference.agent.tool_agent.IMAGE_LABEL``), matched by the arena's
context drain."""

import re

IMAGE_LABEL_RE = re.compile(r"Grid (?:image|text) of the board at step (\d+) \(before this turn's actions\):")


def _message_text(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _has_image_label(message):
    return IMAGE_LABEL_RE.search(_message_text(message)) is not None
