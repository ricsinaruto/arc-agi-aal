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
LIGHT_MODES = ("none", "low", "medium")


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
        self.acted = self.surprise = False
        self.compactions = compactions
        self.modes[mode] += 1
        self.reasons[why or "light"] += 1
        return template_fields(mode, self.heavy_effort)

    def summary(self):
        return {**self.cfg, "heavy_effort": self.heavy_effort, "requests": self.requests,
                "modes": dict(self.modes), "reasons": dict(self.reasons)}


def template_fields(mode, heavy_effort):
    if mode == "none":
        # The template checks the effort only when thinking is on; keep a valid value for the counter.
        return {"enable_thinking": False, "reasoning_effort": heavy_effort}
    return {"enable_thinking": True, "reasoning_effort": mode}
