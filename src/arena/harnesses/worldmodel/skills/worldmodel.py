"""check_sim(sim, read_state) / plan(sim, state, goal, actions) / run_plan(steps, sim, read_state): verify an executable model of the game against every recorded transition, search it, and execute its plan."""
# ruff: noqa: F821  (transitions, current_frame and action are globals of the Duck interpreter)
import collections
import re


def check_sim(sim, read_state, level=None, last=300, show=3):
    """Replay recorded transitions through your model: for each transition t (both frames on the same
    level; all levels unless `level` is given; the most recent `last`), compare
    sim(read_state(t.before_frame), t.action) with read_state(t.after_frame). Returns a report with the
    match count per level and the first `show` mismatches (action, predicted, observed). A model that
    reproduces every transition is safe to plan with."""
    ts = [t for t in transitions if t.before_frame is not None and t.before_frame.level == t.after_frame.level
          and (level is None or t.before_frame.level == level)][-last:]
    per_level = collections.OrderedDict()
    bad = []
    for t in ts:
        lv = t.before_frame.level
        ok, total = per_level.get(lv, (0, 0))
        try:
            expected = sim(read_state(t.before_frame), t.action)
            seen = read_state(t.after_frame)
            good = expected == seen
        except Exception as exc:
            good, expected, seen = False, f"{type(exc).__name__}: {exc}", None
        per_level[lv] = (ok + good, total + 1)
        if not good and len(bad) < show:
            bad.append(f"  level {lv} step {t.after_frame.step} {t.action}: predicted {expected!r}, observed {seen!r}")
    matched = sum(ok for ok, _ in per_level.values())
    head = f"check_sim: {matched}/{len(ts)} transitions reproduced (" + ", ".join(
        f"level {lv}: {ok}/{n}" for lv, (ok, n) in per_level.items()) + ")"
    return head + ("\n" + "\n".join(bad) if bad else "\nall reproduced: safe to plan with")


def plan(sim, state, goal, actions, max_depth=60, max_nodes=200000):
    """Breadth-first search from `state` over `actions` (action strings as recorded in `transitions`, e.g.
    'UP', 'SPACE', 'MOUSE(row=12, col=30)') using sim; states must be hashable. Returns the shortest action
    list reaching goal(state), or None (the message says why)."""
    if goal(state):
        return []
    parent = {state: None}
    frontier = collections.deque([(state, 0)])
    while frontier:
        s, depth = frontier.popleft()
        if depth >= max_depth:
            continue
        for a in actions:
            try:
                n = sim(s, a)
            except Exception:
                continue
            if n in parent:
                continue
            parent[n] = (s, a)
            if goal(n):
                path = []
                while parent[n] is not None:
                    n, step = parent[n]
                    path.append(step)
                return path[::-1]
            if len(parent) >= max_nodes:
                print(f"plan: node limit {max_nodes} reached at depth {depth}")
                return None
            frontier.append((n, depth + 1))
    print(f"plan: goal unreachable in the model ({len(parent)} states explored, depth limit {max_depth})")
    return None


def _as_action(step):
    m = re.match(r"\s*MOUSE\(\s*row\s*=\s*(\d+)\s*,\s*col\s*=\s*(\d+)\s*\)", str(step))
    if m:
        return {"action": "ACTION6", "row": int(m.group(1)), "col": int(m.group(2))}
    return {"action": str(step)}


def run_plan(steps, sim=None, read_state=None):
    """Execute a planned action list one action at a time. With sim and read_state it stops at the first
    action whose observed state differs from the model's prediction (then fix the model and plan again);
    it always stops at a level change, a game over or a refused action. Returns a one-paragraph report."""
    done = []
    reason = "plan complete"
    for i, step in enumerate(steps):
        before = read_state(current_frame) if (sim and read_state) else None
        receipt = action([_as_action(step)]) or {}
        if not receipt.get("executed", False) or receipt.get("executed_count", 1) == 0:
            reason = f"step {i} ({step}) not executed: {receipt.get('stop_reason') or receipt.get('error') or 'refused'}"
            break
        done.append(step)
        if receipt.get("level_completed"):
            reason = f"level completed at step {i}"
            break
        if receipt.get("game_over") or receipt.get("done"):
            reason = f"{'game won' if receipt.get('done') else 'game over'} at step {i}"
            break
        if before is not None:
            predicted, seen = sim(before, step), read_state(current_frame)
            if predicted != seen:
                reason = f"model mismatch at step {i} ({step}): predicted {predicted!r}, observed {seen!r}"
                break
    return f"run_plan: executed {len(done)}/{len(steps)} actions; {reason}"
