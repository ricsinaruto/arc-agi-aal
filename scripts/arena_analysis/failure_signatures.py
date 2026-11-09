"""Failure signatures of Duck-transfer level attempts on the DEV games of the one-hour arena runs
(26 September; input to docs/validation-20260926/failure-signatures.md).

Reads only the 14 DEV games of ``posttrain.split`` (holdout traces are never opened). One row per
(run, game, level attempted) with: requests, completion tokens, tokens of requests that ended
without an action, python errors by type, actions by type, board-change evidence computed from the
logged frames, repeated state-action pairs, replayed prefixes after a GAME_OVER/RESET, no-effect
action runs (timer draining), refusals, stuck notices, compactions and what followed them, and the
carried world model; each stalled attempt (the last, uncleared level of a game) gets signature tags.

Usage:
    python scripts/arena_analysis/failure_signatures.py --out <scratch>/attempts.jsonl [--report]

``--report`` prints the tables quoted in the doc (pooled L1/deep metrics, tag histogram, clusters).

Board-change evidence: a change is "HUD-only" when every changed cell lies in the outer 3-cell band
of the 64x64 board (timer and counter bars); "gameplay" otherwise. Board identity for novelty and
state-action repeats uses the interior (rows/cols 3..60) of the last frame of each observation.
Tag rules are in ``tag`` and in the doc.
"""

import argparse
import collections
import glob
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from posttrain.split import DEV  # noqa: E402

RUN_GLOBS = ("runs/kaggle-arena-duck-60m-r23-*/arena/trials/duck-transfer",
             "runs/modal-validation/arena-duck-60m-r23-*/arena/trials/duck-transfer")
BAND = 3
ERROR_RE = re.compile(r"^(\w*(?:Error|Exception|Interrupt))\b", re.M)
WM_RE = re.compile(r"(World model|Goal model|Action model|Recent findings|Open questions|Plan|Cross-level notes):")
SANDBOX_RE = re.compile(r"not allowed in the sandbox|name '(globals|open|locals|vars|exec|eval|type|setattr)' is not defined")
CLICK = "ACTION6"
LONG_REQUEST = 8000
NOOP_RUN = 5
REPLAY_PREFIX = 4


def run_name(trial):
    parts = Path(trial).parts
    i = parts.index("runs") + 1
    return parts[i + 1] if parts[i] == "modal-validation" else parts[i]


def short_run(name):
    return re.sub(r"-modal-2026\d{4}$", "", re.sub(r"^(kaggle-)?arena-duck-60m-r23-", "", name))


def interior(frame):
    return "".join(row[BAND:64 - BAND] for row in frame[BAND:64 - BAND])


def board_hash(frame):
    return hashlib.md5(interior(frame).encode()).hexdigest()[:12]


def diff(before, after):
    """(gameplay cells changed, HUD-band cells changed)."""
    game = hud = 0
    for r, (a, b) in enumerate(zip(before, after)):
        if a == b:
            continue
        for c, (x, y) in enumerate(zip(a, b)):
            if x != y:
                if BAND <= r < 64 - BAND and BAND <= c < 64 - BAND:
                    game += 1
                else:
                    hud += 1
    return game, hud


def majority_color(frame):
    return collections.Counter(interior(frame)).most_common(1)[0][0]


def action_key(action):
    name = action.get("action")
    return f"{CLICK}@{action.get('y')},{action.get('x')}" if name == CLICK else name


COUNTERS = ("requests", "completion_tokens", "reasoning_tokens", "other_tokens", "noaction_requests",
            "noaction_tokens", "long_requests", "no_tool_replies", "length_finishes", "wm_replies",
            "python_calls", "python_errors", "sandbox_errors", "python_timeouts", "python_truncated",
            "actions", "resets", "model_resets", "clicks", "keys", "gameplay_changes", "hud_only", "no_change",
            "distinct_boards", "hidden_animations", "click_background", "click_change", "click_hud_only",
            "state_action_repeats", "noeffect_repeats", "post_compaction_noeffect_repeats",
            "noop_run_actions", "drain_game_overs", "lives", "replays", "replay_actions",
            "game_overs", "skipped", "stuck_notices", "compactions", "post_compaction_noaction_requests",
            "post_compaction_noaction_tokens", "acting_requests", "max_requests_without_action", "cap_hits",
            "request_timeouts", "coverage_calls", "model_seconds",
            # harness v2 (fixed-2x-harness-v2 runs; zero elsewhere): nudges, new-level notices, clock tracks
            # learned or revoked, skill calls, and requests without an action in the two after a nudge
            "v2_nudges", "v2_nudges_followed", "v2_new_level_notices", "v2_clock_tracks", "v2_clock_revoked",
            "v2_check_model_calls", "v2_effects_calls", "v2_new_elements_calls")


class Attempt:
    def __init__(self, run, game, level, total, prev_frame):
        self.row = {"run": run, "game": game, "level": level, "total_levels": total, "cleared": False,
                    **{k: 0 for k in COUNTERS}, "error_types": collections.Counter(),
                    "action_types": collections.Counter(), "requests_before_first_action": None,
                    "tokens_before_first_action": None, "actions_to_first_change": None,
                    "tokens_to_first_change": None, "first_line": None, "last_line": None}
        self.click_targets = set()
        self.seen_boards = {board_hash(prev_frame)} if prev_frame is not None else set()
        self.seen_sa, self.noeffect_sa = set(), {}
        self.compacted = False
        self.since_action = 0
        self.pending = None           # tokens of the newest request while it has not acted yet
        self.after_compaction = False  # counting requests between a compaction and the next action
        self.noop_run = 0
        self.life, self.lives = [], []
        self.nudge_requests = None  # requests since the latest harness-v2 nudge, until an action

    def end_life(self):
        r = self.row
        if self.life:
            best = max((sum(1 for _ in _prefix(self.life, old)) for old in self.lives), default=0)
            if best >= REPLAY_PREFIX:
                r["replays"] += 1
                r["replay_actions"] += best
            self.lives.append(self.life)
            r["lives"] += 1
        self.life = []

    def close_request(self):
        if self.pending is not None:
            self.row["noaction_requests"] += 1
            self.row["noaction_tokens"] += self.pending
            if self.after_compaction:
                self.row["post_compaction_noaction_requests"] += 1
                self.row["post_compaction_noaction_tokens"] += self.pending
        self.pending = None

    def finish(self):
        self.close_request()
        self.end_life()
        r = self.row
        r["error_types"] = dict(r["error_types"])
        r["action_types"] = dict(r["action_types"])
        r["distinct_click_targets"] = len(self.click_targets)
        r["model_seconds"] = round(r["model_seconds"], 1)
        return r


def _prefix(a, b):
    for x, y in zip(a, b):
        if x != y:
            return
        yield x


def attempts_for_game(log_path, run):
    game = Path(log_path).parent.name[:4]
    rows, level, total, prev_frame, prev_state, cur = [], 0, None, None, None, None
    for n, line in enumerate(open(log_path), 1):
        rec = json.loads(line)
        kind = rec.get("kind")
        if cur is None:
            if kind == "observation":
                total, level = rec.get("total_levels"), int(rec.get("levels") or 0)
                prev_frame = rec["frames"][-1] if rec.get("frames") else None
                cur = Attempt(run, game, level + 1, total, prev_frame)
                cur.row["first_line"] = n
            continue
        r = cur.row
        r["last_line"] = n
        if kind == "model":
            usage = rec.get("usage") or {}
            tokens = int(usage.get("completion_tokens") or 0)
            if rec.get("purpose") not in (None, "gameplay"):
                r["other_tokens"] += tokens
                continue
            cur.close_request()
            cur.pending = tokens
            r["requests"] += 1
            r["completion_tokens"] += tokens
            r["reasoning_tokens"] += int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
            r["model_seconds"] += float(rec.get("seconds") or 0)
            r["long_requests"] += tokens >= LONG_REQUEST
            msg = rec.get("message") or {}
            r["no_tool_replies"] += not msg.get("tool_calls")
            r["length_finishes"] += rec.get("finish_reason") == "length"
            r["wm_replies"] += bool(WM_RE.search(msg.get("content") or ""))
            cur.since_action += 1
            if cur.nudge_requests is not None:
                cur.nudge_requests += 1
                if cur.nudge_requests > 2:
                    cur.nudge_requests = None
            r["max_requests_without_action"] = max(r["max_requests_without_action"], cur.since_action)
            r["cap_hits"] += cur.since_action == 12
            if r["actions"] == 0:
                r["requests_before_first_action"] = r["requests"]
                r["tokens_before_first_action"] = r["completion_tokens"]
        elif kind in ("compaction", "memory_rewrite", "notes_request"):
            r["other_tokens"] += int((rec.get("usage") or {}).get("completion_tokens") or 0)
            if kind == "compaction":
                r["compactions"] += 1
                cur.compacted = cur.after_compaction = True
        elif kind == "duck_python":
            r["python_calls"] += 1
            res = rec.get("result") or {}
            r["coverage_calls"] += "coverage(" in (rec.get("code") or "")
            for skill in ("check_model", "effects", "new_elements"):
                r[f"v2_{skill}_calls"] += f"{skill}(" in (rec.get("code") or "")
            r["python_truncated"] += bool(res.get("truncated"))
            err = res.get("error")
            if err:
                r["python_errors"] += 1
                if "timed out" in err:
                    r["python_timeouts"] += 1
                    r["error_types"]["Timeout"] += 1
                else:
                    found = ERROR_RE.findall(err)
                    r["error_types"][found[-1] if found else "Other"] += 1
                r["sandbox_errors"] += bool(SANDBOX_RE.search(err))
        elif kind == "stuck_notice":
            r["stuck_notices"] += 1
        elif kind == "harness_v2_nudge":
            r["v2_nudges"] += 1
            cur.nudge_requests = 0
        elif kind == "new_level_notice":
            r["v2_new_level_notices"] += 1
        elif kind == "harness_v2_clock":
            r["v2_clock_tracks" if rec.get("event") == "clock" else "v2_clock_revoked"] += 1
        elif kind == "action_skipped":
            r["skipped"] += 1
        elif kind == "auto_reset":
            r["game_overs"] += 1
        elif kind in ("deadline_request_timeout", "worker_request_cancelled"):
            r["request_timeouts"] += 1
        elif kind == "observation":
            action = rec.get("action")
            frame = rec["frames"][-1] if rec.get("frames") else prev_frame
            state = rec.get("state")
            new_level = int(rec.get("levels") or 0)
            total = rec.get("total_levels", total)
            if action is not None:
                key, name = action_key(action), action.get("action")
                if cur.pending is not None:
                    r["acting_requests"] += 1
                    cur.pending = None
                cur.after_compaction = False
                cur.since_action = 0
                if cur.nudge_requests is not None:
                    r["v2_nudges_followed"] += 1
                    cur.nudge_requests = None
                r["actions"] += 1
                r["action_types"][name] += 1
                if r["requests_before_first_action"] is None:
                    r["requests_before_first_action"] = r["requests"]
                    r["tokens_before_first_action"] = r["completion_tokens"]
                if name == "RESET":
                    r["resets"] += 1
                    r["model_resets"] += prev_state != "GAME_OVER"
                    cur.end_life()
                    cur.noop_run = 0
                else:
                    cur.life.append(key)
                    if name == CLICK:
                        r["clicks"] += 1
                        cur.click_targets.add((action.get("y"), action.get("x")))
                    else:
                        r["keys"] += 1
                if name != "RESET" and prev_frame is not None and frame is not None:
                    before = board_hash(prev_frame)
                    game_cells, hud_cells = diff(prev_frame, frame)
                    if name == CLICK:
                        y, x = int(action.get("y", -1)), int(action.get("x", -1))
                        if 0 <= y < 64 and 0 <= x < 64 and prev_frame[y][x] == majority_color(prev_frame):
                            r["click_background"] += 1
                    sa = (before, key)
                    if sa in cur.seen_sa:
                        r["state_action_repeats"] += 1
                        if sa in cur.noeffect_sa:
                            r["noeffect_repeats"] += 1
                            if cur.compacted and cur.noeffect_sa[sa] == "pre":
                                r["post_compaction_noeffect_repeats"] += 1
                    cur.seen_sa.add(sa)
                    if game_cells:
                        r["gameplay_changes"] += 1
                        r["click_change"] += name == CLICK
                        if cur.noop_run >= NOOP_RUN:
                            r["noop_run_actions"] += cur.noop_run
                        cur.noop_run = 0
                        if r["actions_to_first_change"] is None:
                            r["actions_to_first_change"] = r["actions"]
                            r["tokens_to_first_change"] = r["completion_tokens"]
                    else:
                        before_inner = interior(prev_frame)
                        if any(interior(f) != before_inner for f in (rec.get("frames") or [])[:-1]):
                            r["hidden_animations"] += 1  # effect visible only mid-animation; final frame unchanged
                        cur.noop_run += 1
                        if hud_cells:
                            r["hud_only"] += 1
                            r["click_hud_only"] += name == CLICK
                        else:
                            r["no_change"] += 1
                        cur.noeffect_sa.setdefault(sa, "post" if cur.compacted else "pre")
                    if state == "GAME_OVER" and cur.noop_run >= NOOP_RUN:
                        r["drain_game_overs"] += 1
                    h = board_hash(frame)
                    if h not in cur.seen_boards:
                        cur.seen_boards.add(h)
                        r["distinct_boards"] += 1
            prev_frame, prev_state = frame, state
            if new_level > level:
                if cur.noop_run >= NOOP_RUN:
                    r["noop_run_actions"] += cur.noop_run
                r["cleared"] = True
                rows.append(cur.finish())
                level = new_level
                cur = Attempt(run, game, level + 1, total, prev_frame)
                cur.row["first_line"] = n
        elif kind == "finished":
            r["finish_status"] = rec.get("status")
    if cur is not None and (total is None or level < total):
        if cur.noop_run >= NOOP_RUN:
            cur.row["noop_run_actions"] += cur.noop_run
        rows.append(cur.finish())
    return rows


def level_weight(level, total):
    return level / (total * (total + 1) / 2) if total else 0.0


def remaining_weight(level, total):
    return 1 - (level - 1) * level / 2 / (total * (total + 1) / 2) if total else 0.0


def tag(r, clear_cost=None):
    """Dominant failure signatures of one stalled attempt (thresholds documented in the doc).

    ``clear_cost``: median completion tokens of this game's level-1 clears over all runs (None if never cleared).
    """
    acts = r["actions"] - r["resets"]
    req = r["requests"]
    if req < 3:
        return ["x_no_play"]
    tags = []
    change_rate = r["gameplay_changes"] / max(1, acts)
    if req >= 8 and (r["noaction_requests"] >= 0.7 * req or r["max_requests_without_action"] >= 10):
        tags.append("h_analysis_without_action")
    if r["gameplay_changes"] <= 2 or (acts >= 10 and change_rate < 0.3 and r["distinct_boards"] <= 5):
        tags.append("a_no_effect_found")
    if acts >= 10 and (r["state_action_repeats"] >= 0.3 * acts or r["noeffect_repeats"] >= 5 or r["replay_actions"] >= 0.3 * acts):
        tags.append("d_repeat_or_replay")
    if r["noop_run_actions"] >= 10 or r["drain_game_overs"]:
        tags.append("f_noop_runs_timer_drain")
    if r["clicks"] >= 8 and (r["click_hud_only"] + (r["clicks"] - r["click_change"] - r["click_hud_only"])) >= 0.4 * r["clicks"]:
        tags.append("g_clicks_without_effect")
    if acts >= 5 and (r["hud_only"] >= 0.2 * acts or r["hidden_animations"] >= 2):
        tags.append("k_effect_masked_hud_or_animation")
    if r["python_calls"] >= 5 and r["python_errors"] >= 0.15 * r["python_calls"]:
        tags.append("c_python_errors")
    if r["cap_hits"] or r["python_timeouts"] >= 2 or r["length_finishes"] or r["request_timeouts"] >= 2 or r["skipped"] >= 5:
        tags.append("f_harness_limits")
    if r["compactions"] and (r["post_compaction_noeffect_repeats"] >= 3 or r["post_compaction_noaction_requests"] >= 4):
        tags.append("e_post_compaction_rederive")
    if r["gameplay_changes"] >= 10 and change_rate >= 0.5 and r["distinct_boards"] >= 10 and "d_repeat_or_replay" not in tags:
        tags.append("b_mechanics_ok_goal_missing")
    if r["game_overs"] >= 2:
        tags.append("i_repeated_game_over")
    if clear_cost and r["level"] == 1 and r["completion_tokens"] < clear_cost:
        tags.append("j_starved_below_clear_cost")
    return tags or ["untagged"]


def load(globs=RUN_GLOBS):
    rows = []
    for pattern in globs:
        for trial in sorted(glob.glob(str(ROOT / pattern))):
            run = short_run(run_name(trial))
            for log in sorted(Path(trial, "games").glob("*/log.jsonl")):
                if log.parent.name[:4] not in DEV:  # holdout traces are never opened
                    continue
                rows += attempts_for_game(log, run)
    last = collections.defaultdict(int)
    for r in rows:
        last[(r["run"], r["game"])] = max(last[(r["run"], r["game"])], r["level"])
    l1_cost = {}
    for g in DEV:
        costs = [r["completion_tokens"] for r in rows if r["game"] == g and r["level"] == 1 and r["cleared"] and r["requests"]]
        l1_cost[g] = statistics.median(costs) if costs else None
    for r in rows:
        r["stall"] = (not r["cleared"]) and r["level"] == last[(r["run"], r["game"])]
        r["level_weight"] = round(level_weight(r["level"], r["total_levels"]), 4)
        r["remaining_weight"] = round(remaining_weight(r["level"], r["total_levels"]), 4)
        r["tags"] = tag(r, l1_cost[r["game"]]) if r["stall"] else []
    return rows


CLUSTERS = (
    ("C1 effects not perceived", ("a_no_effect_found", "g_clicks_without_effect", "k_effect_masked_hud_or_animation")),
    ("C2 analysis without action", ("h_analysis_without_action",)),
    ("C3 goal not found (mechanics seen)", ("b_mechanics_ok_goal_missing",)),
    ("C4 repeats and replays", ("d_repeat_or_replay", "i_repeated_game_over")),
    ("C5 compaction re-derivation", ("e_post_compaction_rederive",)),
    ("C6 execution and tool errors", ("c_python_errors", "f_harness_limits")),
    ("C7 no-effect runs / timer drain", ("f_noop_runs_timer_drain",)),
    ("S  starved below clear cost (L1)", ("j_starved_below_clear_cost",)),
)


def pooled(sub):
    s = lambda k: sum(r[k] for r in sub)  # noqa: E731
    acts = s("actions") - s("resets")
    return {"attempts": len(sub), "requests": s("requests"), "tokens": s("completion_tokens"),
            "noaction_req_share": round(s("noaction_requests") / max(1, s("requests")), 3),
            "noaction_tok_share": round(s("noaction_tokens") / max(1, s("completion_tokens")), 3),
            "actions": acts, "actions_per_request": round(acts / max(1, s("requests")), 2),
            "gameplay_change_share": round(s("gameplay_changes") / max(1, acts), 3),
            "hud_only_share": round(s("hud_only") / max(1, acts), 3), "hidden_animations": s("hidden_animations"),
            "python_error_rate": round(s("python_errors") / max(1, s("python_calls")), 3),
            "sandbox_errors": s("sandbox_errors"), "timeouts": s("python_timeouts"),
            "state_action_repeat_share": round(s("state_action_repeats") / max(1, acts), 3),
            "replay_actions": s("replay_actions"), "noop_run_actions": s("noop_run_actions"),
            "drain_game_overs": s("drain_game_overs"), "model_resets": s("model_resets"), "game_overs": s("game_overs"),
            "clicks": s("clicks"), "click_effect_share": round(s("click_change") / max(1, s("clicks")), 3),
            "stuck_notices": s("stuck_notices"), "compactions": s("compactions"),
            "post_compaction_noaction_tokens": s("post_compaction_noaction_tokens"),
            "long_requests": s("long_requests"), "wm_replies": s("wm_replies"), "cap_hits": s("cap_hits"),
            "v2_nudges": s("v2_nudges"), "v2_nudges_followed": s("v2_nudges_followed"),
            "v2_new_level_notices": s("v2_new_level_notices"), "v2_clock_tracks": s("v2_clock_tracks"),
            "v2_clock_revoked": s("v2_clock_revoked"), "v2_check_model_calls": s("v2_check_model_calls")}


def report(rows):
    stalls = [r for r in rows if r["stall"]]
    print("== pooled: L1 stalls / L1 clears / deep stalls / deep clears")
    for label, sub in (("L1 stall", [r for r in stalls if r["level"] == 1]),
                       ("L1 clear", [r for r in rows if r["level"] == 1 and r["cleared"]]),
                       ("deep stall", [r for r in stalls if r["level"] >= 2]),
                       ("deep clear", [r for r in rows if r["level"] >= 2 and r["cleared"]])):
        print(label, json.dumps(pooled(sub)))
    print("\n== L1: tokens received by stalls vs tokens to clear, per game (median, K)")
    for g in DEV:
        st = [r["completion_tokens"] for r in stalls if r["game"] == g and r["level"] == 1]
        cl = [r["completion_tokens"] for r in rows if r["game"] == g and r["level"] == 1 and r["cleared"] and r["requests"]]
        rq = [r["requests"] for r in stalls if r["game"] == g and r["level"] == 1]
        print(f"{g}: stalls {len(st):2d} median {statistics.median(st) / 1e3 if st else 0:5.1f}K ({statistics.median(rq) if rq else 0} req)"
              f" | clears {len(cl):2d} median {statistics.median(cl) / 1e3 if cl else 0:5.1f}K")
    print("\n== tag histogram over stalls with play (count, tokens, remaining weight x runs)")
    hist = collections.defaultdict(lambda: [0, 0, 0.0, set()])
    for r in stalls:
        for t in r["tags"]:
            h = hist[t]
            h[0] += 1
            h[1] += r["completion_tokens"]
            h[2] += r["remaining_weight"]
            h[3].add(r["game"])
    for t, (c, tok, w, games) in sorted(hist.items(), key=lambda kv: -kv[1][1]):
        print(f"{t:30s} {c:4d} attempts {tok / 1e6:5.2f}M tokens  blocked-weight {w:6.2f} games {len(games)}")
    print("\n== clusters (a stall counts once per cluster; weight = remaining level weight, game-equivalents per run)")
    runs = len({r["run"] for r in rows})
    for name, members in CLUSTERS:
        sub = [r for r in stalls if any(t in members for t in r["tags"])]
        print(f"{name:34s} stalls {len(sub):3d} (L1 {sum(r['level'] == 1 for r in sub):3d}, deep {sum(r['level'] > 1 for r in sub):2d}) "
              f"runs {len({r['run'] for r in sub}):2d} games {len({r['game'] for r in sub}):2d} "
              f"tokens {sum(r['completion_tokens'] for r in sub) / 1e6:5.2f}M weight/run {sum(r['remaining_weight'] for r in sub) / runs:5.2f}")
    print("\n== L1 stalls per game: tag counts")
    for g in DEV:
        sub = [r for r in stalls if r["game"] == g and r["level"] == 1]
        counts = collections.Counter(t for r in sub for t in r["tags"])
        print(f"{g}: {len(sub):2d} stalls, {sum(r['completion_tokens'] for r in sub) / 1e3:6.1f}K tokens | "
              + ", ".join(f"{t} {n}" for t, n in counts.most_common()))
    print("\n== per game x level stalls (level>=2)")
    for r in sorted((r for r in stalls if r["level"] >= 2), key=lambda r: (r["game"], r["level"], r["run"])):
        print(f"{r['game']} L{r['level']}/{r['total_levels']} {r['run']:16s} req {r['requests']:3d} tok {r['completion_tokens'] / 1e3:5.1f}K "
              f"noact {r['noaction_requests']:3d} acts {r['actions'] - r['resets']:3d} chg {r['gameplay_changes']:3d} "
              f"rep {r['state_action_repeats']:3d} replay {r['replay_actions']:3d} noop {r['noop_run_actions']:3d} "
              f"GO {r['game_overs']} err {r['python_errors']}/{r['python_calls']} comp {r['compactions']} {','.join(r['tags'])}")


def extras(globs=RUN_GLOBS):
    """Run-level counts quoted in the doc that are not per-attempt fields (DEV logs and transcripts only)."""
    c = collections.Counter()
    by_game = collections.defaultdict(collections.Counter)
    modules = collections.Counter()
    logs = [log for pattern in globs for trial in sorted(glob.glob(str(ROOT / pattern)))
            for log in sorted(Path(trial, "games").glob("*/log.jsonl")) if log.parent.name[:4] in DEV]
    for log in logs:
        game = log.parent.name[:4]
        streak, boards, prev = 0, [], None
        for line in open(log):
            rec = json.loads(line)
            kind = rec.get("kind")
            if kind == "model" and rec.get("purpose") in (None, "gameplay"):
                tokens = int((rec.get("usage") or {}).get("completion_tokens") or 0)
                c["tokens"] += tokens
                if streak >= 3:
                    c["tokens_after_3_noaction"] += tokens
                    c["requests_after_3_noaction"] += 1
                streak += 1
                c["action7_mentions"] += "ACTION7" in ((rec.get("message") or {}).get("reasoning") or "")
            elif kind == "stuck_notice":
                c["stuck_notices"] += 1
                m = re.search(r"(\d+) consecutive actions", rec.get("text", ""))
                c["stuck_notices_zero"] += bool(m and m.group(1) == "0")
                c["stuck_notices_tried_none"] += "no effect: none" in rec.get("text", "")
            elif kind == "duck_python":
                err = (rec.get("result") or {}).get("error") or ""
                m = re.search(r"Module '(\w+)' is not allowed|name '(\w+)' is not defined", err)
                if m and SANDBOX_RE.search(err):
                    modules[m.group(1) or m.group(2)] += 1
            elif kind == "observation" and rec.get("frames"):
                frames, action = rec["frames"], rec.get("action")
                if action:
                    streak = 0
                if action and action.get("action") == "ACTION7" and len(boards) >= 2:
                    after = board_hash(frames[-1])
                    by_game[game]["action7"] += 1
                    by_game[game]["action7_undo"] += after == boards[-2]
                if action and action.get("action") != "RESET" and prev is not None:
                    before = interior(prev)
                    if interior(frames[-1]) == before and any(interior(f) != before for f in frames[:-1]):
                        by_game[game]["hidden_animations"] += 1
                prev = frames[-1]
                boards.append(board_hash(frames[-1]))
    for tr in (ROOT / "runs").glob("*/**/duck-transfer/games/*/duck-transcript.txt"):
        if tr.parent.name[:4] not in DEV or "60m-r23" not in str(tr):
            continue
        for block in tr.read_text(errors="replace").split("[USER PROMPT]")[1:]:
            user = block.split("[MODEL RESPONSE META]")[0]
            if "end of world model" in user:
                c["logged_prompts"] += 1
                c["logged_prompts_with_world_model"] += "Working world model carried" in user
    print("\n== run-level counts")
    print(json.dumps(dict(c)))
    print("ACTION7 (uses, undo matches):", {g: (v["action7"], v["action7_undo"]) for g, v in by_game.items() if v["action7"]})
    print("hidden animations by game:", {g: v["hidden_animations"] for g, v in by_game.items() if v["hidden_animations"]})
    print("sandbox-rule errors by name:", dict(modules.most_common()))


SHORT = {"a_no_effect_found": "a", "b_mechanics_ok_goal_missing": "b", "c_python_errors": "c", "d_repeat_or_replay": "d",
         "e_post_compaction_rederive": "e", "f_harness_limits": "f", "f_noop_runs_timer_drain": "t",
         "g_clicks_without_effect": "g", "h_analysis_without_action": "h", "i_repeated_game_over": "i",
         "j_starved_below_clear_cost": "s", "k_effect_masked_hud_or_animation": "k", "x_no_play": "x", "untagged": "-"}


def markdown(rows):
    """Appendix table: one line per DEV game, every run's stalled level and signature codes."""
    stalls = [r for r in rows if r["stall"]]
    won = {(r["run"], r["game"]) for r in rows} - {(r["run"], r["game"]) for r in stalls}
    print("| Game | Stalled level: run (tokens at that level) signature codes |\n| --- | --- |")
    for g in DEV:
        cells = [f"L{r['level']} {r['run']} ({r['completion_tokens'] / 1e3:.0f}K) {''.join(SHORT[t] for t in r['tags'])}"
                 for r in sorted((r for r in stalls if r["game"] == g), key=lambda r: (-r["level"], r["run"]))]
        cells += [f"won {run}" for run, game in sorted(won) if game == g]
        print(f"| {g} | " + "; ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--glob", nargs="+", default=list(RUN_GLOBS))
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--markdown", action="store_true", help="print the per-game appendix table")
    parser.add_argument("--extras", action="store_true", help="print the run-level counts quoted in the doc")
    args = parser.parse_args()
    rows = load(args.glob)
    Path(args.out).write_text("\n".join(json.dumps(r, default=list) for r in rows) + "\n")
    print(len(rows), "attempts,", sum(r["stall"] for r in rows), "stalls,",
          len({r["run"] for r in rows}), "runs,", len({r["game"] for r in rows}), "games")
    if args.report:
        report(rows)
    if args.markdown:
        markdown(rows)
    if args.extras:
        extras(args.glob)


if __name__ == "__main__":
    main()
