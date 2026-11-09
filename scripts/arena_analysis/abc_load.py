"""Loader for the hero-v2 / run 1 / run A / run B comparison (29 September; CPU only, reads recorded runs).

One ``Run`` per trial directory: per clone the source family, the level clears on the trial's wall clock (with the
clone's completion tokens and actions at each clear), the scorecard's per-level actions and baselines, every model
reply (time, purpose, tokens, finish) and the scheduler's admit/park events. ``score_at(t)`` rebuilds the arena score
at any wall time from the clears made by then, with the scorecard's scoring rule:

    level score  ls_k = min(115, 100 (baseline_k / actions_k)^2)
    game score   = min(sum_{k<=L} k ls_k, 100 sum_{k<=L} k) / (N (N + 1) / 2)       (N = the game's level count)
    arena score  = mean over the 110 clones.
"""

import json
from collections import defaultdict
from pathlib import Path

RUNS = Path("./runs/modal-validation")
LABELS = {
    "hero-v2": "arena-hero-v2-2h-v1-modal-20260929",
    "run1": "arena-hero-v2-128k-int4-v1-modal-20260929",
    "runA": "arena-hero-v2-int4-streams-v1-modal-20260929",
    "runB": "arena-hero-v2-128k-streams-v1-modal-20260929",
}


def trial_dir(run_id, root=RUNS):
    return root / run_id / run_id / "arena" / "trials" / "duck-transfer"


def level_score(baseline, actions):
    if not actions:
        return 0.0
    return min(115.0, 100.0 * (baseline / actions) ** 2)


def game_score(level_scores, n_levels):
    """Scorecard rule for one clone from the scores of its cleared levels (in order)."""
    if not level_scores:
        return 0.0
    got = sum((k + 1) * s for k, s in enumerate(level_scores))
    cap = 100.0 * sum(k + 1 for k in range(len(level_scores)))
    return min(got, cap) / (n_levels * (n_levels + 1) / 2)


class Game:
    def __init__(self, gid, family):
        self.id = gid
        self.family = family
        self.n_levels = None
        self.clears = []  # dicts: level (1-based, the level just cleared), wall, tokens, actions
        self.level_actions = []
        self.baselines = []
        self.final_score = 0.0
        self.replies = []  # dicts: wall, purpose, completion, reasoning, prompt, finish, level, actions_in_reply
        self.events = []  # admit/park events from arena.json
        self.row = {}
        self.log_counts = defaultdict(int)
        self.log_events = defaultdict(list)

    def score_at(self, t):
        cleared = [c for c in self.clears if c["wall"] <= t]
        ls = [level_score(self.baselines[c["level"] - 1], self.level_actions[c["level"] - 1]) for c in cleared]
        return game_score(ls, self.n_levels)

    def levels_at(self, t):
        return sum(1 for c in self.clears if c["wall"] <= t)

    def tokens_at(self, t, purposes=None):
        return sum(r["completion"] for r in self.replies if r["wall"] <= t and (purposes is None or r["purpose"] in purposes))


class Run:
    def __init__(self, label, run_id=None, root=RUNS):
        self.label = label
        self.run_id = run_id or LABELS[label]
        self.dir = trial_dir(self.run_id, root)
        self.arena = json.loads((self.dir / "arena.json").read_text())
        self.scorecard = json.loads((self.dir / "scorecard.json").read_text())
        self.wall = float((self.arena.get("wall") or {}).get("ended_seconds") or self.arena.get("wall_seconds"))
        roster = json.loads((self.dir / "public-roster.json").read_text())
        self.games = {r["game_id"]: Game(r["game_id"], r["source_game_id"].split("-")[0]) for r in roster}
        self.epoch0 = self._epoch0()
        self._load_scorecard()
        self._load_rows()
        self._load_logs()
        self.live = self._load_live()

    def _epoch0(self):
        fallback = None
        with (self.dir / "admission.jsonl").open() as fh:
            for line in fh:
                rec = json.loads(line)
                if isinstance(rec.get("epoch"), (int, float)) and isinstance(rec.get("wall_seconds"), (int, float)):
                    # Workers may enqueue before the gameplay barrier, reporting a clamped zero wall.
                    if rec.get("event") == "admit" and rec.get("reason"):
                        return rec["epoch"] - rec["wall_seconds"]
                    if fallback is None and rec["wall_seconds"] > 0:
                        fallback = rec["epoch"] - rec["wall_seconds"]
        if fallback is not None:
            return fallback
        raise ValueError("no epoch in admission.jsonl")

    def _load_scorecard(self):
        for env in self.scorecard["environments"]:
            g = self.games[env["id"]]
            run = env["runs"][0]
            g.n_levels = env["level_count"]
            g.level_actions = run.get("level_actions") or []
            g.baselines = run.get("level_baseline_actions") or []
            g.final_score = env["score"]
            g.final_levels = env["levels_completed"]
            g.final_actions = env["actions"]

    def _load_rows(self):
        for row in self.arena["games"]:
            g = self.games[row["game_id"]]
            g.row = {k: v for k, v in row.items() if k not in ("inference_intervals", "tool_intervals", "charges",
                                                                  "admission_timing")}
            g.events = row.get("admissions") or []
            g.inference_intervals = row.get("inference_intervals") or []

    def _load_logs(self):
        for gid, g in self.games.items():
            path = self.dir / "games" / gid / "log.jsonl"
            level, tokens, actions, step_actions = 0, 0, 0, 0
            last_action_count = 0
            pending_compaction_usage = None
            if not path.exists():
                if g.row.get("admissions") or g.events or g.final_actions or g.final_levels:
                    raise FileNotFoundError(path)
                continue  # Block scheduling can leave roster entries entirely unstarted.
            for line in path.open():
                rec = json.loads(line)
                kind = rec.get("kind")
                g.log_counts[kind] += 1
                wall = rec.get("time", self.epoch0) - self.epoch0
                if kind == "observation":
                    if rec.get("action") is not None:
                        actions += 1
                    new = int(rec.get("levels") or 0)
                    while new > level:
                        level += 1
                        g.clears.append({"level": level, "wall": wall, "tokens": tokens, "actions": actions})
                elif kind == "model":
                    usage = rec.get("usage") or {}
                    if rec.get("purpose") == "compaction":
                        pending_compaction_usage = usage
                    details = usage.get("completion_tokens_details") or {}
                    g.replies.append({"wall": wall, "purpose": rec.get("purpose"), "level": level,
                                      "completion": usage.get("completion_tokens") or 0,
                                      "reasoning": details.get("reasoning_tokens") or 0,
                                      "prompt": usage.get("prompt_tokens") or 0,
                                      "finish": rec.get("finish_reason"), "seconds": rec.get("seconds"),
                                      "actions_before": actions})
                    tokens += usage.get("completion_tokens") or 0
                elif kind == "compaction":
                    usage = rec.get("usage") or {}
                    # Recent traces log both the model response and the compaction event.
                    # Older traces have only the latter. Count the inference once in either format.
                    already_recorded = pending_compaction_usage is not None and pending_compaction_usage == usage
                    pending_compaction_usage = None
                    if not already_recorded:
                        g.replies.append({"wall": wall, "purpose": "compaction", "level": level,
                                      "completion": usage.get("completion_tokens") or 0,
                                      "reasoning": (usage.get("completion_tokens_details") or {}).get(
                                          "reasoning_tokens") or 0,
                                      "prompt": usage.get("prompt_tokens") or 0, "finish": None,
                                      "seconds": rec.get("seconds"), "actions_before": actions})
                        tokens += usage.get("completion_tokens") or 0
                    g.log_events[kind].append({**{k: v for k, v in rec.items() if k != "usage"}, "wall": wall,
                                               "level": level, "prompt_tokens": usage.get("prompt_tokens")})
                elif kind not in ("action_timing", "duck_python", "token_count", "exact_admission_count",
                                  "model_request"):
                    g.log_events[kind].append({**rec, "wall": wall, "level": level})
                elif kind == "duck_python":
                    res = rec.get("result") or {}
                    g.log_events["python"].append({"wall": wall, "level": level, "rc": res.get("returncode"),
                                                   "error": (res.get("error") or "")[-400:],
                                                   "stdout_tail": (res.get("stdout") or "")[-200:]})
            # actions per reply: difference of the action count between consecutive gameplay replies
            gp = [r for r in g.replies if r["purpose"] == "gameplay"]
            for i, r in enumerate(gp):
                nxt = gp[i + 1]["actions_before"] if i + 1 < len(gp) else actions
                r["actions"] = nxt - r["actions_before"]

    def _load_live(self):
        rows, seen = [], set()
        with (self.dir / "progress.jsonl").open() as fh:
            for line in fh:
                rec = json.loads(line)
                sc = ((rec.get("live") or {}).get("score") or {})
                t = sc.get("at_elapsed_seconds")
                if t is None or t in seen:
                    continue
                seen.add(t)
                rows.append({"t": t, "score": sc.get("score"), "levels": sc.get("levels")})
        return rows

    def live_at(self, t):
        best = None
        for row in self.live:
            if row["t"] <= t + 30:
                best = row
        return best

    def score_at(self, t, games=None):
        games = games or self.games.values()
        return sum(g.score_at(t) for g in games) / 110.0

    def families(self):
        fam = defaultdict(list)
        for g in self.games.values():
            fam[g.family].append(g)
        return dict(sorted(fam.items()))


def load_all(labels=("hero-v2", "run1", "runA", "runB")):
    return {label: Run(label) for label in labels}
