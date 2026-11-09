"""run_bfs — breadth-first search inside the world model.

Once the model is green on the whole history, planning is free: BFS over
modelled states costs no real actions. The action set is the legal simple
actions plus any candidate clicks the caller nominates (the click space is
64x64, so the agent chooses which clicks are worth considering — same design
as the original harness).

Targets: "advance" stops at a state whose transition set level_up (or win);
"win" stops only at win. Dead states are pruned, not expanded.
"""

from collections import deque

from . import render
from .model import ModelError


def _canon(obj):
    """A hashable identity for a model's internal state."""
    if isinstance(obj, dict):
        return tuple(sorted((k, _canon(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_canon(v) for v in obj)
    if isinstance(obj, (set, frozenset)):
        return tuple(sorted(map(_canon, obj), key=repr))
    if hasattr(obj, "tobytes"):  # numpy arrays
        return obj.tobytes()
    try:
        hash(obj)
        return obj
    except TypeError:
        return repr(obj)


class PlanResult:
    def __init__(self, plan, via, expanded, distinct, final_grid, note=""):
        self.plan = plan  # list of {"action": a[, "x": x, "y": y]} or None
        self.via = via
        self.expanded = expanded
        self.distinct = distinct
        self.final_grid = final_grid
        self.note = note


def run_bfs(model, start_grid, level, actions, clicks=(), target="advance",
            max_depth=12, max_nodes=20000):
    """Search for an action sequence reaching level_up/win in the model."""
    moves = [(a, None, None) for a in actions if a not in (0, 6)]
    moves += [(6, int(x), int(y)) for x, y in clicks]
    try:
        state0 = model.init_state(start_grid, level=level)
    except ModelError as exc:
        return PlanResult(None, None, 0, 0, None, note=str(exc))

    key0 = (render.grid_key(start_grid), _canon(state0))
    seen = {key0}
    queue = deque([(start_grid, state0, [])])
    expanded = 0
    errors = 0
    while queue:
        grid, mstate, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for a, x, y in moves:
            if expanded >= max_nodes:
                queue.clear()
                break
            expanded += 1
            try:
                g2, flags, s2 = model.predict(mstate, grid, a, x, y,
                                              level=level)
            except ModelError:
                errors += 1
                continue
            step = {"action": a}
            if a == 6:
                step["x"], step["y"] = x, y
            new_path = path + [step]
            if flags["dead"]:
                continue
            if flags["win"] or (target == "advance" and flags["level_up"]):
                via = "win" if flags["win"] else "level_up"
                note = f" ({errors} predict error(s) ignored)" if errors else ""
                return PlanResult(new_path, via, expanded, len(seen), g2,
                                  note=note)
            if g2 is None:
                continue
            key = (render.grid_key(g2), _canon(s2))
            if key in seen:
                continue
            seen.add(key)
            queue.append((g2, s2, new_path))
    note = f"{errors} predict error(s) ignored" if errors else ""
    return PlanResult(None, None, expanded, len(seen), None, note=note)


def format_report(result, actions, n_clicks):
    if result.plan is None:
        return (
            f"BFS: NO goal found; expanded {result.expanded} nodes, "
            f"{result.distinct} distinct states (actions={sorted(actions)} + "
            f"{n_clicks} click(s)). Widen clicks / raise max_depth, or the "
            f"model's mechanics don't reach the goal yet."
            + (f" [{result.note}]" if result.note else "")
        )
    lines = [
        f"BFS: goal in {len(result.plan)} step(s) via {result.via}; "
        f"expanded {result.expanded} nodes, {result.distinct} distinct "
        f"states (actions={sorted(actions)} + {n_clicks} click(s))."
        + result.note,
        f"Plan (-> commit): {result.plan}",
    ]
    if result.final_grid is not None:
        lines.append("Predicted final grid:")
        lines.append(render.render(result.final_grid))
    return "\n".join(lines)
