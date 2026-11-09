"""Rejection sampling over teacher clears of single-level episode games.

Flash plays many short episodes (``posttrain.games.episodes``) and ``scripts/posttrain/extract_segments.py``
turns its runs into engine-verified level-clear segments. This module keeps only the most efficient
clears:

1. Only verified segments of known episode games count. The segment's game id must match the
   episode's notes exactly (same prefix and content hash), and the level must be 1 (episodes have one).
2. **Held-out check.** A segment is rejected, and reported, when its episode's ``episode_of`` (or
   ``skin_of``, or the game prefix itself) is a held-out generated mechanic
   (``split.GENERATED_HOLDOUT``). An episode prefix ``q2NN`` does not show its mechanic, so
   ``split.check_training_games`` alone cannot catch this.
3. Clears are grouped by ``(episode_of, tier)``: every layout and skin of one mechanic at one tier.
4. In each group, clears are ranked by completion tokens, then actions, then segment id. A clear whose
   completion tokens exceed ``max_median_multiple`` (default 2) x the group median (over all the
   group's verified clears) is dropped, and the best ``k`` (default 2) of the rest are kept.

Completion tokens are the segment's ``completion_tokens``: every reply of the level including its
reasoning (the compaction summaries are counted apart, in ``compaction_completion_tokens``).
"""

import collections
import json
import statistics
from pathlib import Path

from posttrain import split

DEFAULT_K = 2
DEFAULT_MAX_MEDIAN_MULTIPLE = 2.0


def load_jsonl(paths):
    rows = []
    for path in paths:
        with open(path) as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def group_name(base, tier):
    return f"{base}/tier-{tier}"


def is_held_out(info, game_id):
    return any(split.is_generated_holdout(x) for x in (info.get("episode_of"), info.get("skin_of"), game_id) if x)


def select(segments, notes, *, k=DEFAULT_K, max_median_multiple=DEFAULT_MAX_MEDIAN_MULTIPLE):
    """``(selected segments, stats)``; ``notes`` maps an episode prefix to its notes
    (``posttrain.games.episodes.load_notes``). Selected segments carry a ``rejection`` field."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if max_median_multiple <= 0:
        raise ValueError("max_median_multiple must be positive")
    reasons = collections.Counter()
    held_out, groups = [], collections.defaultdict(list)
    for segment in segments:
        game_id = segment.get("game_id", "")
        info = notes.get(split.prefix(game_id))
        if is_held_out(info or {}, game_id):
            held_out.append({"id": segment.get("id"), "game_id": game_id,
                             "episode_of": (info or {}).get("episode_of")})
        elif not segment.get("verified"):
            reasons["unverified"] += 1
        elif info is None:
            reasons["not an episode game"] += 1
        elif info["game_id"] != game_id:
            reasons["episode version differs from its notes"] += 1
        elif int(segment.get("level", 0)) != 1:
            reasons["level other than 1"] += 1
        else:
            groups[(info["episode_of"], int(info["tier"]))].append(segment)
    selected, rows = [], {}
    totals = collections.Counter()
    for (base, tier), members in sorted(groups.items()):
        tokens = [int(s["completion_tokens"]) for s in members]
        median = statistics.median(tokens)
        ranked = sorted(members, key=lambda s: (int(s["completion_tokens"]), int(s["action_count"]), s["id"]))
        eligible = [s for s in ranked if int(s["completion_tokens"]) <= max_median_multiple * median]
        kept = eligible[:k]
        name = group_name(base, tier)
        for rank, segment in enumerate(kept, 1):
            info = notes[split.prefix(segment["game_id"])]
            selected.append(segment | {"rejection": {
                "group": name, "episode_of": base, "tier": tier, "skin_of": info.get("skin_of"), "rank": rank,
                "group_clears": len(members), "group_median_tokens": median}})
        kept_tokens = [int(s["completion_tokens"]) for s in kept]
        rows[name] = {
            "clears": len(members), "episodes_cleared": len({s["game_id"] for s in members}),
            "median_tokens": median, "over_median_multiple": len(members) - len(eligible),
            "beyond_k": len(eligible) - len(kept), "kept": [s["id"] for s in kept], "kept_tokens": kept_tokens,
            "kept_actions": [int(s["action_count"]) for s in kept],
            "kept_baselines": [s.get("baseline_actions") for s in kept],
            "tokens_all": sum(tokens), "tokens_kept": sum(kept_tokens)}
        totals.update(clears=len(members), kept=len(kept), tokens_all=sum(tokens), tokens_kept=sum(kept_tokens),
                      over_median_multiple=len(members) - len(eligible), beyond_k=len(eligible) - len(kept))
    split.check_training_games(s["game_id"] for s in selected)
    by_group = collections.Counter(group_name(n["episode_of"], n["tier"]) for n in notes.values()
                                   if not is_held_out(n, n["game_id"]))
    for name, count in by_group.items():
        rows.setdefault(name, {"clears": 0, "episodes_cleared": 0, "kept": []})["episodes"] = count
    held_notes = sorted(p for p, n in notes.items() if is_held_out(n, n["game_id"]))
    stats = {
        "rule": {"group_by": ["episode_of", "tier"], "rank_by": ["completion_tokens", "action_count", "id"],
                 "k": k, "max_median_multiple": max_median_multiple,
                 "median_over": "all verified clears of the group"},
        "totals": {"segments": len(segments), "groups": len(groups), "groups_with_episodes": len(by_group),
                   "episode_clears": totals["clears"], "selected": totals["kept"],
                   "dropped_over_median_multiple": totals["over_median_multiple"], "dropped_beyond_k": totals["beyond_k"],
                   "completion_tokens_all_clears": totals["tokens_all"],
                   "completion_tokens_selected": totals["tokens_kept"],
                   "completion_tokens_saved": totals["tokens_all"] - totals["tokens_kept"],
                   "saved_share": round(1 - totals["tokens_kept"] / totals["tokens_all"], 4) if totals["tokens_all"] else None,
                   "mean_tokens_all_clears": round(totals["tokens_all"] / totals["clears"], 1) if totals["clears"] else None,
                   "mean_tokens_selected": round(totals["tokens_kept"] / totals["kept"], 1) if totals["kept"] else None},
        "groups": dict(sorted(rows.items())),
        "groups_without_clears": sorted(name for name, row in rows.items() if not row["clears"]),
        "held_out_check": {"passed": not held_out and not held_notes, "rejected_segments": held_out,
                           "held_out_episodes_in_notes": held_notes, "episodes_checked": len(notes),
                           "held_out_mechanics": list(split.GENERATED_HOLDOUT_MECHANICS)},
        "skipped": dict(sorted(reasons.items())),
    }
    return selected, stats


def write(out_dir, selected, stats):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "selected.jsonl", "w") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
