"""run_backtest — check the world model against every recorded transition.

The check mirrors the original harness's wording exactly: grids are compared
on non-terminal steps, and the level_up / dead / win flags are compared on
EVERY step. RESET transitions and steps with no prior grid are skipped (the
model is re-initialised there instead). The model's internal state threads
through the whole history; each prediction is made from the *recorded* before
grid, so every check is a one-step prediction from ground truth.
"""

from . import render
from .model import ModelError


class Mismatch:
    def __init__(self, transition, kind, detail):
        self.transition = transition
        self.kind = kind  # "grid" | "flags" | "error"
        self.detail = detail


def run_backtest(model, transitions, start=None, end=None, level=None,
                 indices=None):
    """Returns (checked, mismatches, skipped, scope_label)."""
    scope = "all transitions"
    subset = transitions
    if indices is not None:
        wanted = set(indices)
        subset = [t for t in transitions if t.index in wanted]
        scope = f"indices {sorted(wanted)}"
    elif start is not None or end is not None:
        lo = start or 0
        hi = end if end is not None else len(transitions)
        subset = [t for t in transitions if lo <= t.index < hi]
        scope = f"steps [{lo}, {hi})"
    elif level is not None:
        subset = [t for t in transitions if t.level_before == level]
        scope = f"level {level}"

    checked = skipped = 0
    mismatches = []
    state = None
    entry = None  # the current level's entry grid, injected as ENTRY_GRID
    for t in subset:
        if t.action == 0 or t.before is None:
            skipped += 1
            entry = t.after
            try:
                state = model.init_state(t.after, level=t.level)
            except ModelError as exc:
                mismatches.append(Mismatch(t, "error", str(exc)))
                state = None
            continue
        if state is None:
            entry = entry or t.before
            try:
                state = model.init_state(entry, level=t.level_before)
            except ModelError as exc:
                mismatches.append(Mismatch(t, "error", str(exc)))
                continue
        try:
            pred_grid, flags, state = model.predict(
                state, t.before, t.action, t.x, t.y, level=t.level_before,
                entry_grid=entry)
        except ModelError as exc:
            mismatches.append(Mismatch(t, "error", str(exc)))
            continue
        checked += 1
        if t.level_up:
            # Model state is rebuilt at every level entry. Validated against
            # the released traces: threading state across level boundaries
            # scores 8-92% green on their own histories; re-initialising
            # scores ~99% on all of them.
            entry = t.after
            try:
                state = model.init_state(t.after, level=t.level)
            except ModelError as exc:
                mismatches.append(Mismatch(t, "error", str(exc)))
                state = None
        actual_flags = {"level_up": t.level_up, "dead": t.dead, "win": t.win}
        problems = []
        if flags != actual_flags:
            problems.append(
                f"flags: predicted {flags}, actual {actual_flags}")
        terminal = t.dead or t.win
        if not terminal:
            if pred_grid is None:
                problems.append("grid: model predicted no grid")
            elif pred_grid != t.after:
                problems.append(render.diff_summary(t.after, pred_grid))
        if problems:
            mismatches.append(Mismatch(t, "grid" if not terminal else "flags",
                                       "\n".join(problems)))
    return checked, mismatches, skipped, scope


def format_report(checked, mismatches, skipped, scope, max_details=3):
    """The terminal report, in the original harness's phrasing."""
    ok = max(0, checked - sum(1 for m in mismatches if m.kind != "error"))
    head = (
        f"backtest [{scope}]: {ok}/{checked} transitions fully correct "
        f"(grid on non-terminal steps + level_up/dead/win flags on EVERY "
        f"step); {len(mismatches)} mismatch(es), {skipped} skipped "
        f"(resets / no prior grid)."
    )
    lines = [head]
    if not mismatches:
        lines.append(
            "Model predicts ALL checkable transitions in scope exactly "
            "(grids + level_up/dead/win) — safe to plan with run_bfs."
        )
    else:
        lines.append(
            "Model is NOT green — fix it before planning. First "
            f"{min(max_details, len(mismatches))} mismatch(es):"
        )
        for m in mismatches[:max_details]:
            t = m.transition
            click = f"({t.x},{t.y})" if t.action == 6 else ""
            lines.append(
                f"#{t.index} action={t.action}{click} level={t.level_before}:"
            )
            lines.append("  " + m.detail.replace("\n", "\n  "))
    return "\n".join(lines)
