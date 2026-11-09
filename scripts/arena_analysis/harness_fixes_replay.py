"""Replay of the hero-v2 ``harness_fixes`` rules on a recorded duck-transfer trial (29 September, CPU only).

Reads the same files as ``harness_audit.py`` and answers, for the recorded run, what the fixes in
``arena.harness_fixes`` would have changed:

- ``compaction``: how many compactions an exact-count trigger at the same budget (W - cap - 512) would have
  needed. Per game, the gameplay requests' served prompt sizes give the growth between requests; a real
  compaction's would-be gameplay size is its summary request's prompt minus the summary prompt. The
  simulation walks the growth, compacts when the would-be size crosses the threshold and continues from the
  post-compaction size of the nearest real compaction. It is run at the effective threshold of the recorded
  estimate trigger (the median would-be size at the real compactions: the calibration, which should give
  about the recorded count) and at the exact budget.
- ``summary_guard``: the recorded compactions whose would-be size + the guard's room exceeded the window (the
  guard would have summarised up to the latest observation), and the smallest output headroom.
- ``summary_headroom``: per game, the exact trigger at W - cap - 512 with the summary rules "none", "guard"
  (summary_guard, as run in A and B) and "headroom" (summary_headroom): compactions, would-be sizes, the tight
  ones (would-be + summary prompt + room > W), the rejected ones (summary room < the 2,048 floor: the mechanical
  handoff), the splits and the tokens they keep verbatim; plus a projected-size trigger for comparison.
- ``stuck``: the stuck notices of the recorded rule (8 turns, then every 4; or 12 no-change actions) against the
  fixes' rule (``arena.harness_fixes.StuckRule``), counted per level, with the level distributions of turns,
  generated tokens and no-change streaks the thresholds were chosen from.

    python scripts/arena_analysis/harness_fixes_replay.py --trial <run>/<run>/arena/trials/duck-transfer [--out f.json]
        [--window 130816]
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from harness_audit import annotate, load_trial, pct  # noqa: E402

WINDOW, CAP, MARGIN = 65536, 16384, 512
SUMMARY_PROMPT_TOKENS = 350  # the Duck summary prompt plus the prefilled "World model:" (harness.md finding 3)


def game_sequence(ann):
    """``(prompts, completions, compactions)`` of one game: gameplay prompt/completion tokens in request order and,
    per real compaction, ``(reply_index, would_be_tokens, post_tokens)``; ``post_tokens`` is None for a compaction
    after the game's last gameplay request (the run stopped before the next request)."""
    replies = ann["replies"]
    prompts = [r["prompt"] for r in replies]
    completions = [r["completion"] for r in replies]
    comps = []
    for c in ann["comps"]:
        usage = (c["model"] or {}).get("usage") or {}
        summary_prompt = int(usage.get("prompt_tokens") or 0)
        k = c["reply_index"]
        if not summary_prompt:
            continue
        comps.append((k, summary_prompt - SUMMARY_PROMPT_TOKENS, prompts[k] if k < len(prompts) else None))
    return prompts, completions, comps


def simulate(prompts, comps, threshold, sizes=None):
    """Compactions of an exact trigger at ``threshold`` over one game's recorded growth; ``sizes`` (a list) collects
    the would-be prompt at each simulated compaction."""
    if not prompts:
        return 0
    real = {k: (would, post) for k, would, post in comps}
    posts = sorted((k, p) for k, _, p in comps if p is not None)
    ctx, count = prompts[0], 0
    for i in range(1, len(prompts) + 1):
        if i == len(prompts) and i not in real:
            break
        if i in real:
            growth = real[i][0] - prompts[i - 1]
        else:
            growth = prompts[i] - prompts[i - 1]
        would = ctx + max(0, growth)
        if would > threshold:
            count += 1
            if sizes is not None:
                sizes.append(would)
            later = [p for k, p in posts if k >= i]
            ctx = later[0] if later else (posts[-1][1] if posts else prompts[min(i, len(prompts) - 1)])
        else:
            ctx = would  # (where the real run compacted, the simulated one kept growing)
    return count


def compaction_replay(games, window=WINDOW, cap=CAP, margin=MARGIN, room=4096):
    budget = window - cap - margin
    per_game, would_all = {}, []
    for gid, g in games.items():
        prompts, _, comps = game_sequence(g["ann"])
        if prompts:
            per_game[gid] = (prompts, comps)
            would_all.extend(w for _, w, _ in comps)
    effective = int(statistics.median(would_all)) if would_all else budget
    real = sum(len(c) for _, c in per_game.values())
    calibration = sum(simulate(p, c, effective) for p, c in per_game.values())
    sizes = []
    exact = sum(simulate(p, c, budget, sizes) for p, c in per_game.values())
    guard_exact = [w for w in sizes if w + room > window]
    guard = [w for w in would_all if w + room > window]
    return {
        "budget": budget, "recorded_compactions": real,
        "would_be_at_real_compactions": pct(would_all),
        "effective_threshold_recorded": effective,
        "simulated_at_effective_threshold": calibration,
        "simulated_exact_trigger": exact,
        "avoided_vs_calibration": calibration - exact,
        "avoided_fraction": round((calibration - exact) / calibration, 3) if calibration else None,
        "would_be_at_simulated_exact_compactions": pct(sizes),
        "summary_guard": {"room": room, "would_trigger": len(guard), "would_be": sorted(guard),
                          "would_trigger_with_exact_trigger": len(guard_exact),
                          "min_exact_room": (window - SUMMARY_PROMPT_TOKENS - max(would_all)) if would_all else None},
        "summary_allowance": summary_allowance(games, window),
    }


def summary_allowance(games, window=WINDOW):
    """The recorded summaries' output allowance (``exact_admission_count`` of the summary request) against the room
    the exact count left: the gap is the estimate-based cap (``window - estimate`` in ``aside_request``)."""
    rows = []
    for gid, g in games.items():
        allowance = None
        for ev in g["events"]:
            if ev["kind"] == "exact_admission_count":
                allowance = (ev.get("prompt_tokens"), ev.get("output_allowance"))
            elif ev["kind"] == "model" and ev.get("purpose") == "compaction" and allowance:
                completion = int((ev.get("usage") or {}).get("completion_tokens") or 0)
                rows.append({"game": gid, "prompt": allowance[0], "allowance": allowance[1],
                             "exact_room": window - allowance[0], "written": completion})
    tight = sorted(rows, key=lambda r: r["allowance"] - r["written"])[:3]
    return {"n": len(rows), "allowance": pct([r["allowance"] for r in rows], (0.0, 0.01, 0.05, 0.5)),
            "exact_room": pct([r["exact_room"] for r in rows], (0.0, 0.01, 0.05, 0.5)),
            "allowance_below_exact_room": sum(r["allowance"] < min(CAP, r["exact_room"]) for r in rows),
            "tightest": tight}


# --- summary_headroom (30 September; docs/validation-20260927/compaction-overshoot-fix.md) ---------------------

PREFILLED_OVERHEAD = 228  # run A: summary request - its exact source, all four guard firings with a measured source
SUMMARY_FLOOR = 2048  # aside_request's floor: less room than this is the rejected compaction (mechanical handoff)


def request_sequence(game):
    """``(requests, compactions)`` of one game from its events.

    ``requests``: per gameplay request ``{"prompt", "completion", "user_after"}``; ``user_after`` is True when the
    next list ends with a user message (the reply acted, so a new observation follows, or it called no tool, so a
    follow-up follows) and False when it ends with the reply's tool result (mid-turn).
    ``compactions``: ``(request_index, would_be, post, guard_source)``: the would-be gameplay size is the guard's
    exact count when the guard logged one (``guard_source``: the exact count of what its summary covered), else the
    summary request's prompt minus its prompt overhead (228 prefilled, 350 otherwise); a rejected compaction without a
    guard event is skipped; ``post`` is the next request's prompt (None after the game's last request)."""
    requests, comps, guard = [], [], None
    for ev in game["events"]:
        kind = ev["kind"]
        if kind == "model" and ev.get("purpose") == "gameplay":
            usage = ev.get("usage") or {}
            calls = (ev.get("message") or {}).get("tool_calls") or []
            requests.append({"prompt": int(usage.get("prompt_tokens") or 0),
                             "completion": int(usage.get("completion_tokens") or 0), "user_after": not calls})
            guard = None
        elif kind == "observation" and ev.get("action") is not None and requests:
            requests[-1]["user_after"] = True
        elif kind == "compaction_guard":
            guard = (int(ev["prompt_tokens"]), int(ev["source_tokens"]))
        elif kind == "compaction":
            usage = ev.get("usage") or {}
            if isinstance(usage, str):
                usage = {}
            overhead = PREFILLED_OVERHEAD if ev.get("thinking") == "prefilled-empty" else SUMMARY_PROMPT_TOKENS
            would = guard[0] if guard is not None else (int(usage.get("prompt_tokens") or 0) - overhead or None)
            if would:
                comps.append([len(requests), would, None, guard[1] if guard is not None else None])
            guard = None
    for comp in comps:
        comp[2] = requests[comp[0]]["prompt"] if comp[0] < len(requests) else None
    return requests, [tuple(c) for c in comps]


def simulate_rule(requests, comps, *, threshold, window, rule, room, overhead, floor=SUMMARY_FLOOR):
    """One game under an exact trigger at ``threshold`` with the summary rule ``rule``: "none" (the whole list is
    summarised), "guard" (summary_guard: without the latest observation at a turn start, up to it mid-turn) or
    "headroom" (summary_headroom: the latest cut before a user or assistant message that leaves ``room``; the kept
    tail grows the post-compaction prompt by what an ordinary compaction would have summarised).

    The growth between requests is the recorded one; the reply that preceded request i is its completion, the rest
    of the growth (``rest``: its tool result and the observation or follow-up) is what the guard excludes at a turn
    start (at a real compaction with the same would-be size, the recorded guard's exact source instead). A
    compaction continues from the nearest real compaction's post-compaction prompt (+ the kept extra); ``post`` is
    None for a compaction after the game's last request. Returns one dict per simulated compaction."""
    if not requests:
        return []
    prompts = [r["prompt"] for r in requests]
    real = {k: would for k, would, _, _ in comps}
    guarded = {k: (would, source) for k, would, _, source in comps if source is not None}
    posts = sorted((k, p) for k, _, p, _ in comps if p is not None)
    ctx = prompts[0]
    history = [ctx]  # simulated prompts of the requests since the last compaction (the cut candidates)
    turn_start = ctx
    out = []
    for i in range(1, len(prompts) + 1):
        if i == len(prompts) and i not in real:
            break
        growth = (real[i] if i in real else prompts[i]) - prompts[i - 1]
        growth = max(0, growth)
        reply = requests[i - 1]["completion"]
        user_after = requests[i - 1]["user_after"]
        rest = max(0, growth - reply)
        would = ctx + growth
        if would <= threshold:
            ctx = would
            history.append(ctx)
            if user_after:
                turn_start = ctx
            continue
        tight = would + overhead + room > window
        extra, split = 0, False
        if rule == "none" or not tight:
            source = would
        elif rule == "guard":
            source = would - rest if user_after else turn_start
            if guarded.get(i, (None,))[0] == would:
                source = guarded[i][1]
        else:
            candidates = ([would - rest] if user_after else []) + history[::-1]
            source = next((c for c in candidates if c + overhead + room <= window), candidates[-1])
            split = source < (would - rest if user_after else turn_start)
            if user_after and source < would - rest:
                extra = (would - source) - rest  # the kept reply(s) and tool results beyond the observation
        summary_room = window - source - overhead
        out.append({"request": i, "would_be": would, "tight": tight, "source": source, "summary_room": summary_room,
                    "rejected": summary_room < floor, "split": split, "extra_kept": extra})
        later = [p for k, p in posts if k >= i]
        ctx = (later[0] if later else (posts[-1][1] if posts else prompts[min(i, len(prompts) - 1)])) + extra
        out[-1]["post"] = ctx if i < len(prompts) else None
        history, turn_start = [ctx], ctx
    return out


def headroom_replay(games, *, window=WINDOW, cap=CAP, margin=MARGIN, room=4096):
    """Exact trigger at W - cap - 512 with the recorded guard and with summary_headroom (and, for comparison, a
    trigger on the projected size: the would-be list + cap + the p90 tool result/observation + summary room)."""
    budget = window - cap - margin
    seqs = {gid: request_sequence(g) for gid, g in games.items()}
    thinking = {ev.get("thinking") for g in games.values() for ev in g["events"] if ev["kind"] == "compaction"}
    overhead = PREFILLED_OVERHEAD if thinking == {"prefilled-empty"} else SUMMARY_PROMPT_TOKENS
    rests = [max(0, r2["prompt"] - r1["prompt"] - r1["completion"]) for reqs, _ in seqs.values()
             for r1, r2 in zip(reqs, reqs[1:]) if r2["prompt"] > r1["prompt"]]
    rest_p90 = int(pct(rests, (0.9,))["p90"]) if rests else 0
    recorded = [(gid, c) for gid, (_, comps) in seqs.items() for c in comps]
    result = {"window": window, "budget": budget, "room": room, "summary_prompt_overhead": overhead,
              "rest_p90": rest_p90,
              "recorded": {"compactions": len(recorded), "guard_events": sum(c[3] is not None for _, c in recorded),
                           "max_would_be": max((c[1] for _, c in recorded), default=None),
                           "tight": sum(c[1] + overhead + room > window for _, c in recorded)}}
    for rule in ("none", "guard", "headroom"):
        rows = []
        for gid, (reqs, comps) in seqs.items():
            rows.extend({"game": gid, **row} for row in simulate_rule(reqs, comps, threshold=budget, window=window,
                                                                      rule=rule, room=room, overhead=overhead))
        tight = [r for r in rows if r["tight"]]
        result[rule] = {
            "compactions": len(rows), "max_would_be": max((r["would_be"] for r in rows), default=None),
            "would_be": pct([r["would_be"] for r in rows], (0.5, 0.9, 0.99, 1.0)),
            "tight": len(tight), "rejected_mechanical": sum(r["rejected"] for r in rows),
            "splits": sum(r["split"] for r in rows),
            "min_summary_room": min((r["summary_room"] for r in rows), default=None),
            "summary_room_when_tight": sorted(r["summary_room"] for r in tight),
            "extra_kept_tokens": sorted(r["extra_kept"] for r in rows if r["extra_kept"]),
            "post_when_tight": sorted(r["post"] for r in tight if r["post"] is not None),
            "tight_games": sorted({r["game"] for r in tight}),
            "rejected_games": sorted({r["game"] for r in rows if r["rejected"]}),
        }
    projected = window - cap - rest_p90 - overhead - room
    prompts_comps = {gid: ([r["prompt"] for r in reqs], [(k, w, p) for k, w, p, _ in comps])
                     for gid, (reqs, comps) in seqs.items()}
    result["projected_trigger"] = {
        "threshold": projected,
        "compactions": sum(simulate(p, c, projected) for p, c in prompts_comps.values()),
        "exact_trigger_compactions": sum(simulate(p, c, budget) for p, c in prompts_comps.values()),
    }
    return result


def level_rows(g):
    """Per (level) of one game: turns (acting replies), generated tokens and the no-change flags of its actions."""
    ann = g["ann"]
    rows = defaultdict(lambda: {"turns": 0, "tokens": 0, "no_effect": [], "cleared": False})
    for r in ann["replies"]:
        row = rows[r["level"]]
        row["tokens"] += r["completion"]
        row["turns"] += r["actions"] > 0
    for a in ann["actions"]:
        rows[a["level"]]["no_effect"].append(a["no_effect"])
        if a["cleared"]:
            rows[a["level"]]["cleared"] = True
    return rows


def longest_streak(flags):
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def notices_old(row, turn_limit=8, interval=4, no_change_limit=12):
    """Recorded rule: turns >= 8 (then every 4 turns) or 12 consecutive no-change actions."""
    turns = row["turns"]
    count = max(0, (turns - turn_limit) // interval + 1) if turns >= turn_limit else 0
    if not count and longest_streak(row["no_effect"]) >= no_change_limit:
        count = 1
    return count


def notices_new(row, rule):
    """The fixes' rule (``arena.harness_fixes.ActiveStuckTracker``) on the level's totals: from ``turns`` turns on,
    every ``interval`` turns, plus one notice each time a no-change streak reaches a multiple of the limit."""
    turns = row["turns"]
    count = (turns - rule["turns"]) // rule["interval"] + 1 if turns >= rule["turns"] else 0
    run = 0
    for flag in row["no_effect"]:
        run = run + 1 if flag else 0
        if run and run % rule["no_change_actions"] == 0:
            count += 1
    return count


def stuck_replay(games, rule):
    turns_clear, tokens_clear, streaks, old, new, levels = [], [], [], 0, 0, 0
    old_cleared, new_cleared = 0, 0
    for g in games.values():
        for level, row in level_rows(g).items():
            levels += 1
            streaks.append(longest_streak(row["no_effect"]))
            o, n = notices_old(row), notices_new(row, rule)
            old += o
            new += n
            if row["cleared"]:
                turns_clear.append(row["turns"])
                tokens_clear.append(row["tokens"])
                old_cleared += o
                new_cleared += n
    return {"rule": rule, "levels": levels, "turns_to_clear": pct(turns_clear), "tokens_to_clear": pct(tokens_clear),
            "longest_no_change_streak_per_level": pct(streaks),
            "levels_with_streak_ge": {k: sum(s >= k for s in streaks) for k in (3, 4, 5, 6, 8, 12)},
            "notices_recorded_rule": old, "notices_fixes_rule": new,
            "on_cleared_levels_recorded_rule": old_cleared, "on_cleared_levels_fixes_rule": new_cleared}


def main():
    from arena.harness_fixes import DEFAULTS

    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True)
    parser.add_argument("--out")
    parser.add_argument("--window", type=int, default=WINDOW, help="the trial's context window (130816 for run B)")
    args = parser.parse_args()
    _, games = load_trial(Path(args.trial))
    for g in games.values():
        g["ann"] = annotate(g)
    rule = {"turns": DEFAULTS["stuck_turns"], "interval": DEFAULTS["stuck_notice_turns"],
            "no_change_actions": DEFAULTS["stuck_no_change_actions"]}
    result = {"trial": args.trial,
              "compaction": compaction_replay(games, window=args.window, room=DEFAULTS["summary_room_tokens"]),
              "summary_headroom": headroom_replay(games, window=args.window, room=DEFAULTS["summary_room_tokens"]),
              "stuck": stuck_replay(games, rule)}
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
