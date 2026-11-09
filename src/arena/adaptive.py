"""Adaptive thinking (variant ``fixed-2x-adaptive-think``; 27 September).

In the one-hour base runs 55-60% of gameplay requests execute no action and hold 50-60% of the
completion tokens; a clear costs ~50K tokens at ~850 per action. Medium effort cut that to ~510 per
action (0.43 no-action share) in the Modal probes. Reasoning is needed to form and repair hypotheses,
not to carry out a plan that is working, so the effort is chosen per request:

- **heavy** (the study's effort, xhigh): the game's first request; after a request that executed no
  action (the model is analysing); after a surprise (the last executed action changed nothing on the
  board, completed a level or ended the game); after a context compaction; and after
  ``max_light_streak`` light requests in a row.
- **light** (``light``: ``none`` = the template's empty think block, or a lower effort): otherwise,
  i.e. right after actions that all changed the board with no level change.

The chat template reads ``enable_thinking`` and ``reasoning_effort`` per request, so the switch is a
request field; the logged ``reasoning_effort`` of every model record is the mode actually sent.
"""

from collections import Counter

DEFAULTS = {"light": "none", "max_light_streak": 3}
LIGHT_MODES = ("none", "low", "medium", "prefill")
# "prefill" (10 October, mixed thinking on v8.2): the request keeps the study's template fields, so the system block
# (effort line, tool list) and the cached prefix are the thinking request's; the reply is a continuation of an
# assistant turn that starts with ``LIGHT_PREFILL`` after the template's empty think block (``continue_final_message``,
# as the prefilled-empty summary). "none" switches ``enable_thinking`` off instead, which drops the effort line from the
# system block and so re-prefills the whole prompt at every switch. The server's reasoning parser may hand the
# continuation back as reasoning and its tool parser is off for the request: the call is read here.
LIGHT_PREFILL = "Last result:"
LIGHT_TURN = (
    "This reply is written without a reasoning phase, so write it out: continue from `Last result:` with two to four "
    "short lines (what the last result showed, what you now believe, what this call does and the result you expect), "
    "then make one `python` call. If the result was not what you expected, or you are unsure of the next step, make a "
    "call that only inspects and takes no action: your next reply will have a reasoning phase."
)


def settings(raw=None):
    merged = {**DEFAULTS, **(raw or {})}
    if set(merged) != set(DEFAULTS):
        raise ValueError(f"Unknown adaptive-thinking settings: {sorted(set(merged) - set(DEFAULTS))}")
    if merged["light"] not in LIGHT_MODES:
        raise ValueError(f"light must be one of {LIGHT_MODES}")
    if type(merged["max_light_streak"]) is not int or merged["max_light_streak"] < 1:
        raise ValueError("max_light_streak must be a positive integer")
    return merged


class AdaptiveThinking:
    """Per-game request mode; the agent reports executed actions and asks before every gameplay request."""

    def __init__(self, cfg, heavy_effort):
        self.cfg = settings(cfg)
        self.heavy_effort = heavy_effort
        self.requests = 0
        self.acted = False  # an action executed since the previous request
        self.surprise = False  # one of those actions surprised (no change, level change, game over)
        self.light_streak = 0
        self.compactions = 0
        self.modes = Counter()
        self.reasons = Counter()
        self.last_light = False  # the mode of the last decided request

    def note_action(self, *, changed, level_completed, game_over):
        self.acted = True
        if not changed or level_completed or game_over:
            self.surprise = True

    def reason(self, compactions):
        if self.requests == 0:
            return "first"
        if compactions != self.compactions:
            return "compaction"
        if not self.acted:
            return "no_action"
        if self.surprise:
            return "surprise"
        if self.light_streak >= self.cfg["max_light_streak"]:
            return "streak"
        return None

    def decide(self, compactions=0):
        """The chat-template fields for the next gameplay request."""
        why = self.reason(compactions)
        mode = self.heavy_effort if why else self.cfg["light"]
        self.requests += 1
        self.light_streak = 0 if why else self.light_streak + 1
        self.last_light = not why
        self.acted = self.surprise = False
        self.compactions = compactions
        self.modes[mode] += 1
        self.reasons[why or "light"] += 1
        return template_fields(mode, self.heavy_effort)

    def summary(self):
        return {**self.cfg, "heavy_effort": self.heavy_effort, "requests": self.requests,
                "modes": dict(self.modes), "reasons": dict(self.reasons)}


def template_fields(mode, heavy_effort):
    if mode == "prefill":
        return {}  # the study's fields: the prefix does not change
    if mode == "none":
        # The template checks the effort only when thinking is on; keep a valid value for the counter.
        return {"enable_thinking": False, "reasoning_effort": heavy_effort}
    return {"enable_thinking": True, "reasoning_effort": mode}


def light_reply(message, number):
    """A "prefill" reply made whole in place: the continuation (whichever of the reply's fields carries it) becomes
    the content after ``LIGHT_PREFILL`` and its ``<tool_call>`` block the reply's tool call. Returns True with a call."""
    import json
    import re

    parts = []
    for key in ("reasoning_content", "reasoning", "content"):
        part = message.get(key)
        if isinstance(part, str) and part.strip() and part not in parts:
            parts.append(part)
    raw = "\n".join(parts)
    calls = list(message.get("tool_calls") or [])
    text = raw
    if not calls:
        found = re.search(r"<tool_call>(.*?)(?:</tool_call>|\Z)", raw, re.S)
        body = found.group(1) if found else None
        start = found.start() if found else None
        if body is None:
            at = min((x for x in (raw.find("<function="), raw.find('{"name"')) if x >= 0), default=-1)
            body, start = (raw[at:], at) if at >= 0 else (None, None)
        if body is not None:
            # The template's own form first: <function=name> with <parameter=key> blocks; JSON otherwise.
            function = re.search(r"<function=([^>\s]+)>(.*?)(?:</function>|\Z)", body, re.S)
            if function:
                call = {"name": function.group(1), "arguments": {
                    key: value.strip("\n") for key, value in
                    re.findall(r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|\Z)", function.group(2), re.S)}}
            else:
                try:
                    call, _ = json.JSONDecoder().raw_decode(body.strip())
                except ValueError:
                    call = None
            if isinstance(call, dict) and call.get("name"):
                arguments = call.get("arguments")
                calls = [{"id": f"light-{number}", "type": "function", "function": {
                    "name": str(call["name"]),
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {})}}]
                text = raw[:start]
    text = text.strip()
    message["content"] = LIGHT_PREFILL + (" " + text if text else "")
    message["tool_calls"] = calls or None
    for key in ("reasoning_content", "reasoning"):
        if key in message:
            message[key] = None
    return bool(calls)

