"""Executable-model escalation (variant ``fixed-2x-escalate``; Tycho / Schema idea, 25 September).

Duck-transfer's stuck notice asks the model to re-derive the goal and probe. From the
``after_notices``-th notice on a level, the notice instead demands an executable model of the
level in the persistent interpreter, verified against the recorded transitions before any
further action, and a plan that is abandoned at the first misprediction. The 17 September
harness review found these mechanisms never engaged with local models when offered as the
default loop; here they are the escalation after ordinary probing has failed.
"""

import time

from duck_transfer.feedback import StuckTracker

ESCALATION_NOTICE = (
    "Escalation ({notices} stuck checks on level {level}, {turns} turns, {actions} actions, {seconds}s; "
    "tried on this exact board with no effect: {tried}). Stop probing one action at a time. "
    "In your next python call build an executable model of this level in the persistent "
    "interpreter: `state(frame)` (a compact tuple of the objects that matter), "
    "`transition(state, act)` (the predicted next state for one action, using the mechanics "
    "you believe), and `goal(state)` (a testable win hypothesis). Verify the model before "
    "acting: replay every recorded step of this level from `transitions` / `history` through "
    "`transition` and print the number of steps predicted correctly out of the total; fix the "
    "model until it retrodicts all of them (a model that cannot explain the past cannot plan "
    "the future). Then search over `transition` (BFS or DFS from the current state, bounded "
    "depth) for a sequence that satisfies `goal`, predict the state after each action, and "
    "execute it with one `action([...])` call; on the first receipt that contradicts the "
    "prediction stop, update the model with that transition, and re-plan from the actual "
    "state. If no sequence exists under the model, the goal hypothesis is wrong: derive a new "
    "one from what differs between this level and the last transition, not from more probing."
)


class EscalatingStuckTracker(StuckTracker):
    """The public stuck tracker whose notice escalates from the N-th notice on a level."""

    def __init__(self, cfg, *, after_notices=2, now=None):
        if type(after_notices) is not int or after_notices < 1:
            raise ValueError("after_notices must be a positive integer")
        self.after_notices = after_notices
        self.escalations = 0
        self.total_notices = 0
        super().__init__(cfg, now=now)

    def notice(self, level, tried, now=None):
        text = super().notice(level, tried, now=now)
        if text is None:
            return None
        self.total_notices += 1
        if self.notices < self.after_notices:
            return text
        self.escalations += 1
        seconds = int((time.monotonic() if now is None else now) - self.started)
        tried_text = ", ".join(tried[:12]) + (" ..." if len(tried) > 12 else "") if tried else "none"
        return ESCALATION_NOTICE.format(notices=self.notices, level=level, turns=self.turns,
                                        actions=self.actions, seconds=seconds, tried=tried_text)

    def summary(self):
        return {"after_notices": self.after_notices, "notices": self.total_notices,
                "escalations": self.escalations}
