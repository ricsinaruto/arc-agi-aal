"""Structured stuck procedure (variant ``fixed-2x-procedure``; paper notes 2.6 items 1, 3, 4).

Duck-transfer's stuck notice asks the model to re-derive the goal and probe an untried object.
Here the same notice, at the same cadence, is a checklist instead of advice: the evidence the
level produced, every hypothesis marked CONFIRMED / REFUTED / UNTESTED with the step that decided
it, a bad-execution-versus-bad-idea check, and one targeted probe with a stated expectation
(Self-Harness's retained Qwen3.5-35B-A3B rules, HyperAgents' "structured processes, not attitude
instructions", AIRA2's execution check). A collapse detector reports when most recent actions on
the level were the same action or the same click region.
"""

import re
import time
from collections import Counter, deque

from duck_transfer.feedback import StuckTracker

PROCEDURE_NOTICE = (
    "Stuck check (level {level}: {turns} turns, {actions} actions, {seconds}s; {no_change} consecutive "
    "actions without a board change; tried on this exact board with no effect: {tried}).{collapse}\n"
    "Before any further action, work through this procedure in your reply:\n"
    "1. Evidence: in 3-6 lines, what the actions on this level actually changed (read `transitions` "
    "and `history` for this level; cite step numbers).\n"
    "2. Hypotheses: list each hypothesis about the goal and the mechanics as CONFIRMED, REFUTED or "
    "UNTESTED, each with the step that decided it.\n"
    "3. Bad execution or bad idea: if a plan built on CONFIRMED hypotheses failed, first check the "
    "execution (wrong coordinates, wrong object, wrong order, off by one) before discarding it; drop "
    "REFUTED ideas.\n"
    "4. Pick the smallest UNTESTED hypothesis, state the one action (or short sequence) that confirms "
    "or refutes it and the board change you expect.\n"
    "5. Take only that action, then compare the receipt's `transition` with your expectation and "
    "update the list.\n"
    "Never repeat an action or code cell that already failed on this board. If the level is bricked, "
    "`action(['RESET'])` restarts it."
)

COLLAPSE_LINE = " COLLAPSE DETECTED: {top} of your last {window} actions were {key} (distribution: {dist})."
MOUSE = re.compile(r"MOUSE\(row=(\d+), col=(\d+)\)")


def action_key(label, region=8):
    """Collapse key: the action name, or a click's coarse board region."""
    match = MOUSE.fullmatch(str(label))
    if match:
        row, col = int(match.group(1)) // region, int(match.group(2)) // region
        return f"click region r{row * region}-{row * region + region - 1} c{col * region}-{col * region + region - 1}"
    return str(label)


class ProcedureStuckTracker(StuckTracker):
    """The public stuck tracker (counters and cadence unchanged) with the procedure notice."""

    def __init__(self, cfg, *, collapse_window=30, collapse_share=0.8, now=None):
        if type(collapse_window) is not int or collapse_window < 5 or not 0 < collapse_share <= 1:
            raise ValueError("collapse_window must be an integer >= 5 and collapse_share in (0, 1]")
        self.collapse_window, self.collapse_share = collapse_window, float(collapse_share)
        self.total_notices = 0
        self.collapses = 0
        self.recent = deque(maxlen=collapse_window)
        super().__init__(cfg, now=now)

    def reset_level(self, now=None):
        super().reset_level(now)
        self.recent = deque(maxlen=self.collapse_window)

    def note_label(self, label):
        """Called by the Duck bridge after every executed action with its display label."""
        self.recent.append(action_key(label))

    def collapse(self):
        if len(self.recent) < self.collapse_window:
            return None
        counts = Counter(self.recent)
        key, top = counts.most_common(1)[0]
        if top < self.collapse_share * self.collapse_window:
            return None
        dist = ", ".join(f"{k} x{n}" for k, n in counts.most_common(4))
        return COLLAPSE_LINE.format(top=top, window=self.collapse_window, key=key, dist=dist)

    def notice(self, level, tried, now=None):
        if super().notice(level, tried, now=now) is None:
            return None
        self.total_notices += 1
        collapse = self.collapse()
        self.collapses += collapse is not None
        seconds = int((time.monotonic() if now is None else now) - self.started)
        tried_text = ", ".join(tried[:12]) + (" ..." if len(tried) > 12 else "") if tried else "none"
        return PROCEDURE_NOTICE.format(level=level, turns=self.turns, actions=self.actions, seconds=seconds,
                                       no_change=self.no_change, tried=tried_text, collapse=collapse or "")

    def summary(self):
        return {"notices": self.total_notices, "collapses": self.collapses,
                "collapse_window": self.collapse_window, "collapse_share": self.collapse_share}
