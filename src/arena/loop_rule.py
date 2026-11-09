"""v7.1 strict degenerate-loop rule.

Fires once per no-action streak when a game has made ``streak`` gameplay calls without a game action AND, over the last
``streak`` calls, the mean completion is below ``mean_completion`` tokens OR every consecutive reply pair is at least
``similarity`` alike (difflib quick_ratio over the reply's content head and tool-call arguments). Fitted on B200 run 1
and v6d-8h (k013 / k030 / k088 caught, no false positives); no per-game information.

``action``: "log" records the fire only; "fresh_context" (the user's decision) gives the game a fresh context at its next
request: system prompt, the drain note (python state and inventory survive) and the current observation only (needs
``drain_half``). Off (None) on every earlier build.
"""

import difflib
from collections import deque

DEFAULTS = {"action": "fresh_context", "streak": 20, "mean_completion": 300, "similarity": 0.9}
ACTIONS = ("log", "fresh_context")
LOOP_NOTE = (
    "[Loop rule] Your last {streak} calls took no game action, so this game's conversation was cleared; the board "
    "below is the current one (same level). "
)


def settings(value):
    """The study block (None/False: off; True: ``DEFAULTS``; a dict of overrides)."""
    if value is None or value is False:
        return None
    if value is True:
        value = {}
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError(f"loop_rule is true/false or a dict over {sorted(DEFAULTS)}")
    result = {**DEFAULTS, **value}
    if result["action"] not in ACTIONS:
        raise ValueError(f"loop_rule action is one of {ACTIONS}")
    if type(result["streak"]) is not int or result["streak"] < 2:
        raise ValueError("loop_rule streak is an integer >= 2")
    return result


def reply_text(message):
    """The compared text: the content head and the tool-call arguments (as fitted)."""
    calls = message.get("tool_calls") or []
    args = "".join(str((c.get("function") or {}).get("arguments") or "") for c in calls if isinstance(c, dict))
    return str(message.get("content") or "")[:300] + "|" + args[:400]


def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).quick_ratio() if a and b else 0.0


class LoopDetector:
    def __init__(self, spec):
        self.spec = spec
        n = spec["streak"]
        self.step = None
        self.streak = 0
        self.fired = False
        self.text = None
        self.completions = deque(maxlen=n)
        self.sims = deque(maxlen=n - 1)
        self.fires = []

    def note(self, step, completion_tokens, message):
        """One gameplay reply at action count ``step``; returns the fire record when the rule fires now, else None."""
        text = reply_text(message or {})
        if step != self.step:
            self.step, self.streak, self.fired = step, 1, False
            self.completions.clear()
            self.sims.clear()
        else:
            self.streak += 1
            self.sims.append(similarity(text, self.text))
        self.text = text
        self.completions.append(int(completion_tokens or 0))
        spec = self.spec
        if self.fired or self.streak < spec["streak"]:
            return None
        mean = sum(self.completions) / len(self.completions)
        low = min(self.sims) if self.sims else 0.0
        if mean >= spec["mean_completion"] and low < spec["similarity"]:
            return None
        self.fired = True
        fire = {"step": step, "streak": self.streak, "mean_completion": round(mean, 1), "min_similarity": round(low, 3),
                "reason": "short" if mean < spec["mean_completion"] else "similar"}
        self.fires.append(fire)
        return fire


def fresh_context(history, is_observation=None):
    """``history`` (after the system prompt) cut to its latest observation alone, or None without one."""
    from .context_drain import latest_observation

    index = latest_observation(history, is_observation)
    return None if index is None else [history[index]]


def aggregate(rows):
    rows = [r for r in rows if r]
    return {"games": sum(1 for r in rows if r.get("fires")), "fires": sum(int(r.get("fires", 0)) for r in rows),
            "applied": sum(int(r.get("applied", 0)) for r in rows),
            "tokens_freed": sum(int(r.get("tokens_freed", 0)) for r in rows)}
