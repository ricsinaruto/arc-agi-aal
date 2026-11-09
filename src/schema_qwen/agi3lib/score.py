"""RHAE scoring, mirroring the release's score_trajectories.py semantics.

Per completed level i (1-based), with human baseline h_i and agent actions
a_i:  level_score_i = min(115, 100 * (h_i / a_i)^2). The game score is the
level-number-weighted mean, capped by the completion share:
min(weighted_mean, 100 * sum(completed i) / sum(all i)).

Every action counts toward the level it was taken in, resets included.
"""

import csv

PER_LEVEL_CAP = 115.0
GAME_CAP = 100.0


class EventSummary:
    def __init__(self, completed_actions, incomplete_actions, total_actions,
                 state):
        self.completed_actions = completed_actions
        self.incomplete_actions = incomplete_actions
        self.total_actions = total_actions
        self.state = state


def summarize_events(events):
    """Per-level action counts from an event stream (schema-compatible)."""
    completed = []
    current = 0
    total = 0
    final_state = None
    last_state = None
    prev_step = None
    for ev in events:
        kind = ev.get("kind")
        if kind == "action_taken":
            step = ev.get("step_index")
            expected = 0 if prev_step is None else prev_step + 1
            if step != expected:
                raise ValueError(
                    f"non-contiguous step_index: expected {expected}, "
                    f"got {step}")
            prev_step = step
            total += 1
            current += 1
            if ev.get("state") is not None:
                last_state = str(ev["state"]).upper()
            if ev.get("level_up") is True:
                completed.append(current)
                current = 0
        elif kind == "run_finished" and ev.get("state") is not None:
            final_state = str(ev["state"]).upper()
    state = final_state or last_state or "UNKNOWN"
    if state == "NOT_FINISHED":
        state = "STOPPED"
    return EventSummary(tuple(completed),
                        current if current > 0 else None, total, state)


def rhae(completed_actions, baseline_actions):
    """Returns (score, per-level scores) for one game."""
    level_scores = []
    for i, actions in enumerate(completed_actions):
        h = baseline_actions[i]
        level_scores.append(min(PER_LEVEL_CAP, 100.0 * (h / actions) ** 2))
    total_weight = sum(range(1, len(baseline_actions) + 1))
    if not level_scores or total_weight == 0:
        return 0.0, []
    weighted = sum(s * (i + 1) for i, s in enumerate(level_scores))
    raw = weighted / total_weight  # uncompleted levels contribute zero
    cap = GAME_CAP * sum(range(1, len(level_scores) + 1)) / total_weight
    return min(raw, cap), level_scores


def load_baselines(path):
    """baseline_actions.csv -> {game_id: (task, per-level human actions)}."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            game_id = (row.get("game_id") or "").strip()
            task = (row.get("game") or "").strip().lower()
            n = int(row["n_levels"])
            actions = tuple(int(row[f"level{i}"]) for i in range(1, n + 1))
            out[game_id] = (task, actions)
    return out
