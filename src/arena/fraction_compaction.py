"""Token-fraction retention with whole message/tool exchanges."""

import hashlib
import json
import math


def message_id(message):
    return hashlib.sha256(json.dumps(message, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def split_fraction(messages, tools, measure, fraction, *, anchor=None):
    """``(source, tail, record)``: the summary covers ``source`` and ``tail`` stays verbatim after it.

    By default the tail is the shortest suffix starting at a user message that meets the token target. With
    ``anchor`` (the index of the current turn's observation, for a turn that has outgrown the context on its own:
    no user message follows it, so no suffix is short enough) the tail is that observation plus the shortest
    suffix of complete tool exchanges after it meeting the target; ``source`` then runs up to that suffix and
    includes the observation, which is both summarized and kept."""
    if not 0 < fraction < 1:
        raise ValueError("compaction_keep_fraction must be between zero and one")
    # Qwen rejects system-only chats. The empty query measures fixed framing,
    # instructions and tools without adding any conversation content.
    fixed = measure([messages[0], {"role": "user", "content": ""}], tools)
    conversation = max(0, measure(messages, tools) - fixed)
    target = math.ceil(conversation * fraction)
    pending, boundaries = set(), []
    for index, message in enumerate(messages[1:], 1):
        if not pending and (message.get("role") == "user" if anchor is None
                            else index > anchor + 1 and message.get("role") in ("user", "assistant")):
            boundaries.append(index)
        for call in message.get("tool_calls") or []:
            pending.add(call["id"])
        if message.get("role") == "tool":
            pending.discard(message.get("tool_call_id"))
    kept = [] if anchor is None else [messages[anchor]]
    # Find the shortest complete suffix meeting the token target. Include the previous
    # summary in the conversation denominator, but never count fixed instructions/tools.
    lo, hi, chosen = 0, len(boundaries) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        index = boundaries[mid]
        tokens = max(0, measure([messages[0], *kept, *messages[index:]], tools) - fixed)
        if tokens >= target:
            chosen = (index, tokens)
            lo = mid + 1
        else:
            hi = mid - 1
    if chosen is None and boundaries and anchor is not None:
        # Even the longest in-turn suffix is under the target: keep it (one exchange at least is summarized).
        index = boundaries[0]
        chosen = (index, max(0, measure([messages[0], *kept, *messages[index:]], tools) - fixed))
    if chosen is None or chosen[0] <= 1:
        raise ValueError("No removable prefix preserves the required complete context tail")
    index, tokens = chosen
    tail = [*kept, *messages[index:]]
    return messages[:index], tail, {
        "keep_fraction": fraction, "conversation_tokens": conversation,
        "target_tail_tokens": target, "retained_tokens": tokens,
        "boundary": "observation" if anchor is None else "turn_exchange",
        "source_message_ids": [message_id(m) for m in messages[1:index]],
        "retained_message_ids": [message_id(m) for m in tail],
    }
