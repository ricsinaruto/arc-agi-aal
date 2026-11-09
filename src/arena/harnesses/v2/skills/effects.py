"""effects(n): what this level's last n actions did, from the harness receipts: gameplay change without timer/counter bars, mid-animation effects, scrolls."""
# ruff: noqa: F821  (current_frame and transitions are globals of the Duck interpreter)


def _level_transitions():
    level = current_frame.level
    return [t for t in transitions
            if t.before_frame is not None and t.before_frame.level == level and t.after_frame.level == level]


def _verdict(r):
    if "gameplay_changed" not in r:
        tr = r.get("transition") or {}
        return f"{tr.get('changed_cells')} cells changed (no effect receipt)"
    if r.get("gameplay_changed"):
        text = f"{r.get('gameplay_cells')} gameplay cells in bbox {r.get('gameplay_bbox')}"
    elif r.get("clock_only"):
        text = "NO GAMEPLAY EFFECT (only timer/counter cells changed)"
    else:
        text = "NO EFFECT"
    anim = r.get("animation") or {}
    if anim.get("final_equals_start") and anim.get("peak_cells"):
        text = (f"EFFECT ONLY MID-ANIMATION: up to {anim['peak_cells']} cells in bbox {anim.get('peak_bbox')} over "
                f"{anim.get('frames')} frames, then back to the start (not a no-op)")
    elif anim:
        text += f"; animated over {anim.get('frames')} frames"
    shift = r.get("view_shift")
    if shift:
        text += (f"; whole view shifted by dr={shift['dr']}, dc={shift['dc']} ({shift['share']:.0%} of cells): "
                 "a scroll/pan, not a transformation")
    return text


def effects(n=8):
    """One line per recent action on this level (verdicts from the receipts, not a full-board diff)."""
    lines = []
    for t in _level_transitions()[-n:]:
        r = t.result if isinstance(t.result, dict) else {}
        lines.append(f"step {t.after_frame.step} {t.action}: {_verdict(r)}")
    return "\n".join(lines) if lines else "no actions on this level yet"
