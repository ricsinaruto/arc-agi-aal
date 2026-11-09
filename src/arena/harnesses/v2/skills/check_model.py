"""check_model(predict, observe): replay this level's recorded transitions through your predictor and report the first mismatch."""
# ruff: noqa: F821  (current_frame and transitions are globals of the Duck interpreter)


def check_model(predict, observe, last=30):
    """observe(frame) -> a compact state (e.g. player (row, col) and object positions); predict(state, action) -> the
    state you expect after `action` (the transition's action as the sandbox shows it). Compares
    predict(observe(before), action) with observe(after) for the last `last` transitions of the current level."""
    level = current_frame.level
    ts = [t for t in transitions
          if t.before_frame is not None and t.before_frame.level == level and t.after_frame.level == level][-last:]
    matched = 0
    for t in ts:
        try:
            expected = predict(observe(t.before_frame), t.action)
            seen = observe(t.after_frame)
        except Exception as exc:
            return (f"ERROR at step {t.after_frame.step} ({t.action}): {type(exc).__name__}: {exc}; "
                    f"{matched} earlier transitions matched.")
        if expected != seen:
            return (f"MISMATCH at step {t.after_frame.step} ({t.action}): predicted {expected!r}, observed {seen!r}; "
                    f"{matched} earlier transitions matched. Fix the model before planning with it.")
        matched += 1
    return f"all {matched} recorded transitions on this level reproduced"
