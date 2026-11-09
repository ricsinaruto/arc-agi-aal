"""The append-only event log — the run's single source of truth.

``events.jsonl`` uses the exact schema of the released trajectories, so a run
recorded here is scoreable by the release's own ``score_trajectories.py``:

    run_started, turn_started, turn_committed, action_taken,
    model_mispredicted, turn_fallback, run_finished

Transitions (for backtest) are *derived* from the log, never stored twice.
"""

import json
import time


class EventLog:
    def __init__(self, path):
        self.path = str(path)
        self.seq = 0
        try:
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        self.seq = json.loads(line).get("seq", self.seq)
        except FileNotFoundError:
            pass

    def emit(self, kind, **fields):
        self.seq += 1
        ev = {"kind": kind, "seq": self.seq, "ts": time.time()}
        ev.update(fields)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev) + "\n")
        return ev

    def read(self):
        try:
            with open(self.path, encoding="utf-8") as fh:
                return [json.loads(x) for x in fh if x.strip()]
        except FileNotFoundError:
            return []


class Transition:
    """One recorded step: before-grid, action, after-grid, flags."""

    def __init__(self, index, action, x, y, before, after, state, level,
                 level_up, dead, win, level_before):
        self.index = index
        self.action = action
        self.x = x
        self.y = y
        self.before = before  # None when no prior grid is known
        self.after = after
        self.state = state
        self.level = level
        self.level_up = level_up
        self.dead = dead
        self.win = win
        self.level_before = level_before


def transitions_from_events(events):
    """Reconstruct the transition history exactly as the harness saw it."""
    out = []
    prev_grid = None
    prev_level = 0
    for ev in events:
        kind = ev.get("kind")
        if kind == "turn_started" and prev_grid is None:
            prev_grid = ev.get("grid")
            prev_level = ev.get("level", 0)
        elif kind == "action_taken":
            out.append(Transition(
                index=ev["step_index"], action=ev["action"],
                x=ev.get("x"), y=ev.get("y"),
                before=prev_grid, after=ev["grid"],
                state=ev.get("state", "NOT_FINISHED"),
                level=ev.get("level", 0),
                level_up=bool(ev.get("level_up")),
                dead=bool(ev.get("dead")),
                win=bool(ev.get("win")),
                level_before=prev_level,
            ))
            prev_grid = ev["grid"]
            prev_level = ev.get("level", 0)
    return out


def history_summary(transitions):
    """The one-line history header shown by read_history in the traces."""
    by_action = {}
    level_ups = deaths = wins = resets = clicks = 0
    max_level = 0
    for t in transitions:
        by_action[t.action] = by_action.get(t.action, 0) + 1
        level_ups += t.level_up
        deaths += t.dead
        wins += t.win
        resets += t.action == 0
        clicks += t.action == 6
        max_level = max(max_level, t.level)
    acts = ", ".join(f"{a}:{n}" for a, n in sorted(by_action.items()))
    return (
        f"{len(transitions)} transitions total. Summary: level_ups={level_ups} "
        f"deaths={deaths} wins={wins} resets(action0)={resets} "
        f"clicks(action6)={clicks}; by-action={{{acts}}}; max_level={max_level}"
    )
