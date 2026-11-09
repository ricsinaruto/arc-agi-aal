"""hero-v2 ``drain_half``: dfranzen's "drop half" context management in place of the summary compaction.

"replace our summary compaction with dfranzen's drop half context management" (opt-in, default off:
every earlier build plays the compaction of ``arena.compaction``). The mechanism follows dfranzen's public ARC-AGI-3
milestone-2 solution notebook (Apache License 2.0; ``inference/agent/tool_agent.py``: ``_trim_messages_for_context``
with ``ARC3_CONTEXT_DRAIN_TOKENS``, ``_drop_oldest_history_block``, ``_keep_recent_history_turns`` with
``ARC3_HISTORY_ASSISTANT_TURNS`` / ``ARC3_HISTORY_TURN_DRAIN`` / ``ARC3_HISTORY_DRAIN_COALESCE``, the forced drain after
a context-length rejection, ``_trim_tool_text`` with ``ARC3_MIDDLE_TRUNCATION`` and the malformed tool-call hygiene).
The functions below are re-implementations of that logic on our message lists; their settings are the notebook's
``setup_env`` values.

- Trigger: our existing compaction trigger (the would-be request's prompt tokens + the completion cap exceed the window,
  i.e. the count exceeds ``_context_budget_tokens``). Then the oldest whole history blocks (a user message and every
  assistant/tool message up to the next user message) are dropped until the request is ``drain_tokens`` below the
  budget (``drain_tokens`` is capped at half the budget: "a mis-set knob should cost history depth, not erase
  history"). No summary request; the system prompt (and tools) and the latest observation with its tool rounds are
  never dropped; the reasoning of every surviving reply stays verbatim.
- Turn cap: more than ``turn_cap`` assistant messages cut the history back to ``turn_cap - turn_drain``; with
  ``coalesce`` the deep cut is taken whenever the token drain already dropped (both evictions in one prefix change).
- A context-length rejection forces the drain (down to the drain target in one step) before the retry.
- Between drains the list is only extended (append-only; the prefix cache keeps every turn).

Additions on our side (not in the notebook): the latest *observation* (the turn's board) is protected, not only the
latest user message (a mid-turn follow-up is a user message too); when the current turn alone overflows the budget its
oldest tool exchange is dropped (dfranzen's turns yield after a token budget; ours run until an action); and an
optional one-message note after a drain says what was dropped and that the persistent python state survives, with the
``harness_fixes`` inventory of live definitions when that sub-fix is on (``note``).
"""

import json

# dfranzen's setup_env: context window (116 + 12)K, max output 12K, ARC3_CONTEXT_DRAIN_TOKENS 58K (capped at
# budget / 2), ARC3_HISTORY_ASSISTANT_TURNS 150, ARC3_HISTORY_TURN_DRAIN 30, ARC3_HISTORY_DRAIN_COALESCE 1,
# LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS 3072, ARC3_MIDDLE_TRUNCATION 1.
DEFAULTS = {
    "drain_tokens": 59392,
    "turn_cap": 150,
    "turn_drain": 30,
    "coalesce": True,
    "tool_output_tokens": 3072,
    "middle_truncation": True,
    "drop_malformed": True,
    "note": True,
}
BOOLEANS = ("coalesce", "middle_truncation", "drop_malformed", "note")


def settings(value):
    """The study block for ``value`` (None/False: off; True: ``DEFAULTS``; a dict of overrides), or None when off."""
    if value is None or value is False:
        return None
    if value is True:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("drain_half is true/false or a dict of settings")
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown drain_half settings {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    result = {**DEFAULTS, **value}
    for key in BOOLEANS:
        if type(result[key]) is not bool:
            raise ValueError(f"drain_half {key} is true/false")
    for key in ("drain_tokens", "turn_cap", "turn_drain", "tool_output_tokens"):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f"drain_half {key} is a non-negative integer")
    if result["tool_output_tokens"] < 64:
        raise ValueError("drain_half tool_output_tokens is at least 64 (the public loop's floor)")
    if result["turn_cap"] and result["turn_drain"] >= result["turn_cap"]:
        raise ValueError("drain_half turn_drain must be below turn_cap (0 turn_cap: no turn cap)")
    return result


def role(message):
    return str(message.get("role", "")).strip()


def drain_target(budget, drain_tokens):
    """``(drain, target)``: the drain capped at half the budget and the token level a drain stops at."""
    drain = min(max(0, int(drain_tokens)), int(budget) // 2)
    return drain, (max(1, int(budget) - drain) if drain else int(budget))


def drop_oldest_block(history, *, preserve_recent=1):
    """dfranzen's ``_drop_oldest_history_block``: pop the oldest block in place; False when nothing may go.

    A block is the oldest message and, when it is a user message, everything up to the next user message (an
    assistant or tool head takes only its trailing tool results). The last ``preserve_recent`` messages are never
    popped, and the last user message never is (with none left the request would have no user query)."""
    if len(history) - preserve_recent <= 0:
        return False
    if not any(role(m) == "user" for m in history[1:]):
        return False
    first = history.pop(0)
    while history and role(history[0]) == "tool" and len(history) > preserve_recent:
        history.pop(0)
    if role(first) in ("assistant", "tool"):
        return True
    while history and role(history[0]) != "user" and len(history) > preserve_recent:
        history.pop(0)
    return True


def turn_cut(history, *, max_turns, drain_turns=0, force=False, protect=None):
    """dfranzen's ``_keep_recent_history_turns`` (+ the user-message fix-up of ``_persistent_history_messages``): the
    index the kept history starts at (0: nothing cut).

    The cap counts assistant messages. With ``drain_turns`` it fires above ``max_turns`` and cuts to ``max_turns -
    drain_turns`` (``force``, the coalesced cut, takes the deep cut at once). The cut lands on a user message: the one
    just before the oldest kept assistant message, else the next one; never after ``protect`` (the latest
    observation)."""
    if max_turns <= 0 or not history:
        return 0
    drain_turns = min(max(0, drain_turns), max(0, max_turns - 1))
    if drain_turns:
        present = sum(role(m) == "assistant" for m in history)
        if present > max_turns or force:
            max_turns -= drain_turns
        else:
            return 0
    seen, cut = 0, 0
    for index in range(len(history) - 1, -1, -1):
        if role(history[index]) == "assistant":
            seen += 1
            if seen >= max_turns:
                cut = index
                break
    else:
        return 0
    if cut and role(history[cut - 1]) == "user":
        cut -= 1
    while cut < len(history) and role(history[cut]) != "user":
        cut += 1
    limit = len(history) - 1 if protect is None else protect
    return min(cut, limit)


def latest_observation(history, is_observation=None):
    """Index of the latest observation: the latest user message ``is_observation`` accepts, else the latest user
    message; None without one."""
    users = [i for i, m in enumerate(history) if role(m) == "user"]
    if not users:
        return None
    if is_observation is not None:
        for index in reversed(users):
            if is_observation(history[index]):
                return index
    return users[-1]


def drop_turn_exchange(history, start):
    """Drop the oldest assistant exchange after ``history[start]`` (the observation) in place, keeping the newest one;
    False when the turn has at most one exchange."""
    exchanges = [i for i in range(start + 1, len(history)) if role(history[i]) == "assistant"]
    if len(exchanges) < 2:
        return False
    del history[exchanges[0]:exchanges[1]]
    return True


def drain_history(history, *, count, budget, estimate, spec, force=False, is_observation=None):
    """The drained history (the messages after the system prompt) and what happened: ``(kept, info)``.

    ``count`` is the measured prompt of the whole request (system, tools and carried messages included) and
    ``estimate(messages)`` a linear estimate of messages, so the request after dropping is ``count - (estimate(history)
    - estimate(kept))``. ``kept`` is ``history`` itself when nothing is dropped. ``force`` (a context-length rejection)
    drains to the target whatever the count says and always drops something: one block at least, else the oldest
    exchange of the current turn."""
    drain, target = drain_target(budget, spec["drain_tokens"])
    work = list(history)
    protect = latest_observation(work, is_observation)
    if protect is None:
        return history, None
    offset = int(count) - estimate(work)

    def used():
        return estimate(work) + offset

    blocks = exchanges = 0
    over = count > budget
    if over or force:
        protected = len(work) - protect  # the latest observation and everything after it
        while used() > target and drop_oldest_block(work, preserve_recent=protected):
            blocks += 1
        if force and not blocks and drop_oldest_block(work, preserve_recent=protected):
            blocks += 1  # the server said no although the count fits: one block at least, as the public retry
        protect = latest_observation(work, is_observation)
        while (used() > budget or (force and not blocks and not exchanges)) and drop_turn_exchange(work, protect):
            exchanges += 1
    token_dropped = bool(blocks or exchanges)
    turn_messages = 0
    if spec["turn_cap"]:
        cut = turn_cut(work, max_turns=spec["turn_cap"], drain_turns=spec["turn_drain"],
                       force=bool(spec["coalesce"] and token_dropped), protect=latest_observation(work, is_observation))
        if cut:
            turn_messages = cut
            del work[:cut]
    if not token_dropped and not turn_messages:
        return history, None
    kind = "forced" if force else "tokens" if token_dropped else "turns"
    info = {"trigger": kind, "prompt_tokens": int(count), "budget": int(budget), "target": target, "drain_tokens": drain,
            "estimated_after": used(), "messages_dropped": len(history) - len(work), "blocks": blocks,
            "turn_exchanges": exchanges, "turn_cap_messages": turn_messages,
            "assistant_messages_kept": sum(role(m) == "assistant" for m in work)}
    return work, info


def middle_truncate(text, limit):
    """dfranzen's ``_trim_tool_text`` with ``ARC3_MIDDLE_TRUNCATION``: keep the head two thirds and the tail third of
    ``limit`` characters around a marker (an error's explanation and a snippet's final print are at the end)."""
    if len(text) <= limit:
        return text, False
    omitted = len(text) - limit
    tail = limit // 3
    head = limit - tail
    return f"{text[:head]}\n... [truncated {omitted} chars from the middle] ...\n{text[-tail:]}", True


def malformed_calls(message):
    """One line per tool call of ``message`` whose arguments string is not valid JSON."""
    errors = []
    for call in message.get("tool_calls") or []:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        name = str(function.get("name", "")).strip() or "unknown"
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                json.loads(arguments)
            except json.JSONDecodeError as exc:
                errors.append(f"{name}: invalid JSON arguments ({exc})")
    return errors


# dfranzen's follow-up prefixes for a reply whose calls were dropped (in place of "You have not acted yet. ...").
TRUNCATED_CALL_FOLLOWUP = (
    "Your previous reply was CUT OFF before the tool call finished, so the call was incomplete and could not be "
    "executed. Keep your reasoning much shorter this time and emit the `python` tool call early enough that it "
    "completes within the output limit. Prefer one small snippet over a long analysis. "
)
MALFORMED_CALL_FOLLOWUP = (
    "Your previous tool call had arguments that were not valid JSON, so it was not executed. Emit exactly one "
    "`python` tool call whose arguments are a well-formed JSON object. "
)

DRAIN_NOTE = (
    "[Context drain] The oldest {messages} messages of this game (about {tokens:,} tokens) were dropped to free "
    "context; the messages after this note are unchanged."
)
PYTHON_NOTE = (
    " Your python interpreter was not touched: the variables, functions and imports you defined are still there "
    "unless a reset is reported."
)


def aggregate(rows):
    """Fleet totals of the games' ``context_drain`` rows."""
    rows = [r for r in rows if r]
    keys = ("drains", "forced", "turn_cap_drains", "tokens_dropped", "messages_dropped", "turn_exchanges",
            "malformed_replies", "notes")
    return {"games": len(rows), **{k: sum(int(r.get(k, 0)) for r in rows) for k in keys}}
