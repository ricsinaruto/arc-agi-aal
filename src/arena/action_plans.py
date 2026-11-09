"""Action plans: multi-step actions in code (hero-v2 ``action_plans``, 30 September;
docs/validation-20260927/parallel-samples-plans.md).

52% of the recorded gameplay replies take no action and the rest take 1.3-1.6 actions each, while a deep level costs
30-120K generated tokens. With the component on, the persistent python sandbox of the corrected Duck agent defines
``run_plan(actions, stop_on=("no_change", "level", "game_over", "unexpected"), expect=None)``
(``duck_transfer.sandbox.with_plans``): it executes a planned sequence one action at a time through ``action`` (every
step gets its own receipt; the host's refusals, repeat memory and stuck counters see each action as usual), checks the
receipt after each step and stops at the first stop condition, returning one compact receipt per step. One short
paragraph at the end of the system prompt (the cached prefix; only with the component on) names the helper and says
that once the model has a plan it should execute the whole sequence in one call and inspect the receipts, rather than
one action per reply. Everything else (tools, receipts, prompts) is unchanged.

Stats per game (``ArenaDuckMixin.result``): plan calls, actions requested / executed per plan, stop reasons, actions
per gameplay reply and completion tokens per action.
"""

from collections import Counter

PROMPT = (
    "Multi-step plans: `run_plan(actions, stop_on=('no_change', 'level', 'game_over', 'unexpected'), expect=None)` "
    "executes `actions` (the `action(actions)` formats) one action at a time and stops at the first stop condition: "
    "an action that changed nothing on the board, a completed level, GAME_OVER or the end of the game, or a step whose "
    "receipt does not match `expect` (a function `f(i, action, receipt) -> bool`, or a list with one such function, "
    "dict of receipt fields or None per step). An action that is not executed always stops it. It returns "
    "`{'executed', 'requested', 'stop_reason', 'stopped_at', 'receipts'}` with one compact receipt per action tried "
    "(`board_changed`, `changed_cells`, `change_bbox`, `objects`, `level_completed`, `game_over`, `state`, `level`). "
    "Once you have a plan, execute the whole sequence in one call with `run_plan` and inspect the returned receipts, "
    "rather than one action per reply."
)


def settings(raw):
    """``True`` (on) or ``False`` / ``None`` (off)."""
    if raw is None or raw is False:
        return False
    if raw is True:
        return True
    raise ValueError("action_plans is on/off (true/false)")


def summary(plans, *, actions, replies, completion_tokens):
    """Per-game plan stats from the sandbox's run_plan summaries and the game's action / reply / token counts."""
    requested = [int(p.get("requested") or 0) for p in plans]
    executed = [int(p.get("executed") or 0) for p in plans]
    return {
        "plan_calls": len(plans),
        "plan_actions_requested": sum(requested),
        "plan_actions_executed": sum(executed),
        "actions_per_plan": round(sum(executed) / len(plans), 3) if plans else None,
        "requested_per_plan": round(sum(requested) / len(plans), 3) if plans else None,
        "plans_with_expect": sum(1 for p in plans if p.get("expect")),
        "stop_reasons": dict(Counter(str(p.get("stop_reason")) for p in plans).most_common()),
        "actions": int(actions),
        "gameplay_replies": int(replies),
        "actions_per_reply": round(actions / replies, 3) if replies else None,
        "completion_tokens_per_action": round(completion_tokens / actions, 1) if actions else None,
    }


def aggregate(rows):
    """The trial's totals over the games' ``summary`` rows."""
    rows = [r for r in rows if r]
    if not rows:
        return None
    total = {"games": len(rows), "games_with_plans": sum(1 for r in rows if r["plan_calls"])}
    for key in ("plan_calls", "plan_actions_requested", "plan_actions_executed", "plans_with_expect", "actions",
                "gameplay_replies"):
        total[key] = sum(r[key] for r in rows)
    reasons = Counter()
    for row in rows:
        reasons.update(row["stop_reasons"])
    total["stop_reasons"] = dict(reasons.most_common())
    total["actions_per_plan"] = (round(total["plan_actions_executed"] / total["plan_calls"], 3)
                                 if total["plan_calls"] else None)
    total["actions_per_reply"] = (round(total["actions"] / total["gameplay_replies"], 3)
                                  if total["gameplay_replies"] else None)
    return total
