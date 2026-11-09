"""play(actions): run a planned action sequence and stop at the first surprise (no board change, level change, game over)."""
# ruff: noqa: F821  (action is a global of the Duck interpreter)


def play(actions, stop_on_no_change=True):
    """Execute a planned sequence one action at a time. Items are action names ('UP', 'LEFT', 'SPACE', ...), dicts as
    accepted by action(), or ('MOUSE', row, col). Stops at the first action that changed nothing (if stop_on_no_change),
    at a level change, a game over or a refused/invalid action. Returns a one-paragraph report: actions executed,
    where and why it stopped, the last receipt's state and transition summary."""
    done = []
    reason = "sequence complete"
    last = {}
    for index, item in enumerate(actions):
        if isinstance(item, str):
            item = {"action": item}
        elif isinstance(item, (tuple, list)):
            item = {"action": item[0], "row": int(item[1]), "col": int(item[2])}
        last = action([item]) or {}
        if not last.get("executed", False) or last.get("executed_count", 1) == 0:
            reason = f"step {index} not executed: {last.get('stop_reason') or last.get('error') or 'refused'}"
            break
        done.append(last.get("action_display") or item.get("action"))
        if last.get("level_completed"):
            reason = f"level completed at step {index}"
            break
        if last.get("game_over") or last.get("done"):
            reason = f"{'game won' if last.get('done') else 'game over'} at step {index}"
            break
        if stop_on_no_change and not last.get("board_changed", True):
            reason = f"step {index} ({done[-1]}) changed nothing"
            break
    transition = last.get("transition") or {}
    summary = {k: transition.get(k) for k in ("changed_cells", "change_bbox") if k in transition}
    return (f"play: executed {len(done)} of {len(actions)} ({', '.join(map(str, done[-12:]))}); stopped: {reason}; "
            f"state {last.get('state')}, level {last.get('level')}, last transition {summary}")
