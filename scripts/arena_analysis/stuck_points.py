"""Stuck-level start points for the level bench (hx4, 7 October).

A (game, level) pair is HARD when, over the recorded attempts that reached it, at most half cleared it within
``--within`` generated tokens (attempts that ended earlier without a clear do not count; at least ``--min-attempts``
must count). For each hard pair the start point is a recorded action prefix that reaches the level's first board,
replayed on the local environment files and kept only when the level count, state and board match the recording
(``arena.level_points.check_replay``). Source preference: the run's tag rank (``v71``, then ``v71c``, then untagged),
a source that carried something into the level, an attempt that itself did not clear the level within ``--within``,
then the shortest prefix.

``warm`` is what the source agent carried into the level, read from its transcript's first user turn of the level:
the ``Cross-level notes`` entry of the carried world model and the ``note()`` line (v7.1 only).

``--lead N`` builds points N levels BEFORE the hard level (its first board; an empty prefix for level 1): the bench
agent plays those levels itself and arrives at the hard level with its own tools and conversation, as in a run.

``--deep N`` builds points INSIDE the stuck level instead: for attempts on a hard pair that never cleared it and
spent at least N + ``--within`` tokens there, the board the source agent was on when it had spent N tokens (so the
source itself did not clear in the following ``--within``). ``warm.knowledge`` is then its whole carried world model
at that turn (goal, world, actions, findings, questions, plan), ``level_start_actions`` the prefix length at the
level's first board; up to ``--per-pair`` sources per pair.

The script only reads logs and transcripts; with ``--split holdout`` nobody needs to look at them.

Usage:
    uv run python scripts/arena_analysis/stuck_points.py --run flash71=<trial dir>:v71 --run b2run1=<trial dir> \
        --split dev --games-dir environment_files --out data/stuck_bench/dev.json
"""

import argparse
import collections
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from arena.level_points import check_replay, encode_action, point_id  # noqa: E402

DEV = "cd82 cn04 ft09 lp85 ls20 re86 sb26 sk48 vc33 wa30".split()
HOLDOUT = "ar25 bp35 dc22 g50t ka59 lf52 m0r0 r11l s5i5 sc25 sp80 su15 tn36 tr87 tu93".split()
SECTION = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \|")
NOTES_PREFIX = "Your notes (persist across turns, levels and context drains; update with note(key, text)): "
CARRIED = "Working world model carried from earlier turns:"
REVISE = "- Revise any item above immediately"
RANK = {"v71": 0, "v71c": 1}


def scan_log(path, deep=None):
    """``(levels, prefix, total)``: per level reached its generated tokens, cleared flag, and the prefix length /
    step / board of its first observation (``start``) and, with ``deep``, of the last playable board before the
    reply that took the level past ``deep`` tokens (``deep``). ``prefix`` is the log's replayable action list."""
    from arc_infra.perception import board_hash, decode_grid

    levels = collections.defaultdict(lambda: {"tokens": 0, "actions": 0, "cleared": False})
    prefix, level, usable, total, last = [], 0, True, None, None
    with open(path) as file:
        for line in file:
            if line.startswith('{"kind":"observation"'):
                rec = json.loads(line)
                action = rec.get("action")
                if action is not None:
                    text = encode_action(action)
                    if text is None:
                        usable = False
                    prefix.append(text)
                    levels[level]["actions"] += 1
                total = rec.get("total_levels", total)
                new = int(rec.get("levels", level))
                if action is None and not prefix and rec.get("frames") and "start" not in levels[0]:
                    levels[0]["start"] = {"len": 0, "step": 0, "board_hash": board_hash(decode_grid(rec["frames"][-1]))}
                here = None
                if usable and rec.get("frames") and rec.get("state") == "NOT_FINISHED":
                    here = {"len": len(prefix), "step": int(rec.get("step", len(prefix))), "frame": rec["frames"][-1]}
                if new > level:
                    for k in range(level, new):
                        levels[k]["cleared"] = True
                    if new == level + 1 and here is not None:
                        levels[new]["start"] = {"len": here["len"], "step": here["step"],
                                                "board_hash": board_hash(decode_grid(here["frame"]))}
                    level = new
                last = here
            elif line.startswith('{"kind":"model"'):
                rec = json.loads(line)
                row = levels[level]
                before = row["tokens"]
                row["tokens"] += int((rec.get("usage") or {}).get("completion_tokens") or 0)
                if deep and before < deep <= row["tokens"] and last is not None and "start" in row:
                    row["deep"] = {"len": last["len"], "step": last["step"],
                                   "board_hash": board_hash(decode_grid(last["frame"]))}
    return dict(levels), prefix, total


KNOWLEDGE = {"World model": "world_model", "Goal model": "goal_model", "Action model": "action_model",
             "Recent findings": "recent_findings", "Open questions": "open_questions", "Plan": "current_plan",
             "Cross-level notes": "cross_level_notes"}


def warm_from_transcript(path, step):
    """``{"cross_level_notes", "knowledge", "notes"}`` from the first user turn at or after ``step`` executed
    actions: the carried world model's entries and the ``note()`` line."""
    out = {"cross_level_notes": "", "knowledge": {}, "notes": {}}
    if not Path(path).is_file():
        return out
    inside, carried, key = False, False, None
    with open(path, errors="replace") as file:
        for line in file:
            match = SECTION.match(line)
            if match:
                if inside:
                    break
                inside = int(match.group(2)) - 1 >= step
                continue
            if not inside:
                continue
            line = line.rstrip("\n")
            if line.startswith("[MODEL RESPONSE META]"):
                break
            if line.startswith(CARRIED):
                carried, key = True, None
            elif carried and line.startswith(REVISE):
                carried, key = False, None
            elif carried and line.startswith("- ") and line[2:].partition(": ")[0] in KNOWLEDGE:
                label, _, text = line[2:].partition(": ")
                key = KNOWLEDGE[label]
                out["knowledge"][key] = text
            elif carried and key is not None:
                out["knowledge"][key] += "\n" + line
            elif line.startswith(NOTES_PREFIX):
                for item in line[len(NOTES_PREFIX):].split(" | "):
                    name, sep, text = item.partition(": ")
                    if sep and name.strip():
                        out["notes"][name.strip()[:40]] = text.strip()[:400]
    out["knowledge"] = {k: v.strip() for k, v in out["knowledge"].items() if v.strip()}
    out["cross_level_notes"] = out["knowledge"].get("cross_level_notes", "")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="name=<trial dir>[:v71|:v71c]")
    parser.add_argument("--split", choices=("dev", "holdout"), required=True)
    parser.add_argument("--games-dir", default=str(ROOT / "environment_files"))
    parser.add_argument("--within", type=int, default=100000)
    parser.add_argument("--min-attempts", type=int, default=2)
    parser.add_argument("--deep", type=int, default=None, help="points inside the level, at this many tokens spent")
    parser.add_argument("--per-pair", type=int, default=2, help="--deep: sources per hard pair")
    parser.add_argument("--lead", type=int, default=None,
                        help="lead points: start N levels before the hard level (the agent plays them itself)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    families = DEV if args.split == "dev" else HOLDOUT
    attempts = collections.defaultdict(list)
    for spec in args.run:
        name, _, rest = spec.partition("=")
        folder, _, tag = rest.partition(":")
        folder = Path(folder)
        roster = {r["game_id"]: r["source_game_id"] for r in json.loads((folder / "public-roster.json").read_text())}
        for label, game in sorted(roster.items()):
            log = folder / "games" / label / "log.jsonl"
            if game[:4] not in families or not log.is_file():
                continue
            levels, prefix, total = scan_log(log, args.deep)
            for k, row in levels.items():
                if row["tokens"] or row["cleared"]:
                    attempts[game, k].append({**row, "prefix_all": prefix, "run": name, "label": label,
                                              "starts": {lv: v["start"] for lv, v in levels.items() if "start" in v},
                                              "v71": tag in RANK,
                                              "rank": RANK.get(tag, len(RANK)),
                                              "transcript": str(folder / "games" / label / "duck-transcript.txt"),
                                              "total_levels": total})
    from arc_infra.arcade import make_arcade

    arc = make_arcade("offline", args.games_dir, Path(tempfile.mkdtemp(prefix="stuck-points-")))
    card = arc.open_scorecard(tags=["stuck-points"])
    baselines = {}
    for meta in Path(args.games_dir).glob("*/*/metadata.json"):
        data = json.loads(meta.read_text())
        baselines[data["game_id"]] = data.get("baseline_actions") or []
    points, table, failures = [], [], []
    for (game, k), rows in sorted(attempts.items()):
        counted = [r for r in rows if r["cleared"] or r["tokens"] >= args.within]
        fast = sum(r["cleared"] and r["tokens"] <= args.within for r in counted)
        hard = len(counted) >= args.min_attempts and fast * 2 <= len(counted)
        stat = {"id": point_id(game, k), "game": game, "level": k + 1, "attempts": len(rows), "counted": len(counted),
                "cleared_within": fast, "cleared": sum(r["cleared"] for r in rows), "hard": hard,
                "tokens": sorted(r["tokens"] for r in rows)}
        table.append(stat)
        if not hard or k == 0:
            continue
        base = baselines.get(game) or []
        if args.deep:
            options = [r for r in rows if "deep" in r and not r["cleared"] and r["tokens"] >= args.deep + args.within]
            where, limit = "deep", args.per_pair
        elif args.lead:
            options = [{**r, "lead": r["starts"][k - args.lead]} for r in rows if k - args.lead in r["starts"]]
            where, limit = "lead", 1
        else:
            options = [r for r in rows if "start" in r]
            where, limit = "start", 1
        for r in options:
            r["warm"] = ({"cross_level_notes": "", "knowledge": {}, "notes": {}} if args.lead
                         else warm_from_transcript(r["transcript"], r[where]["step"]))
        options.sort(key=lambda r: (r["rank"], not (r["warm"]["knowledge"] or r["warm"]["notes"]),
                                    r["cleared"] and r["tokens"] <= args.within, r[where]["len"]))
        kept = 0
        for option in options:
            if kept >= limit:
                break
            spot = option[where]
            point = {"level_index": k - (args.lead or 0), "prefix": option["prefix_all"][: spot["len"]],
                     "board_hash": spot["board_hash"]}
            try:
                ok, detail, _ = check_replay(arc.make(game, scorecard_id=card, save_recording=False), point)
            except Exception as exc:  # noqa: BLE001 - a replay error is a determinism failure
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            if not ok:
                failures.append({"id": stat["id"], "run": option["run"], "label": option["label"], "error": detail})
                continue
            kept += 1
            points.append({
                "id": stat["id"] + (f"-{option['run']}-{option['label']}" if args.deep else "")
                + (f"-lead{args.lead}" if args.lead else ""),
                "game": game, "split": args.split, "level_index": k - (args.lead or 0), "level": k + 1,
                **({"lead_levels": args.lead} if args.lead else {}),
                "total_levels": option["total_levels"], "baseline_actions": base[k] if k < len(base) else None,
                "prefix_actions": spot["len"], "level_start_actions": (option.get("start") or spot)["len"],
                "source_run": option["run"], "source_game": option["label"],
                "source_step": spot["step"], "source_v71": option["v71"], "source_cleared": option["cleared"],
                "source_tokens": option["tokens"], "deep_tokens": args.deep, "board_hash": spot["board_hash"],
                "recorded": {key: stat[key] for key in ("attempts", "counted", "cleared_within", "cleared")},
                "warm": option["warm"], "prefix": point["prefix"],
            })
        if not kept and options:
            failures.append({"id": stat["id"], "error": "no replayable recorded prefix"})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": 1, "rule": f"cleared within {args.within} tokens in at most half of the "
                               f"counted attempts (>= {args.min_attempts})", "split": args.split,
                               "runs": [s.split("=")[0] for s in args.run], "points": points},
                              separators=(",", ":")) + "\n")
    out.with_suffix(".stats.json").write_text(json.dumps({"table": table, "failures": failures}, indent=1) + "\n")
    for p in points:
        print(p["id"], "prefix", p["prefix_actions"], "from", p["source_run"], p["source_game"],
              "v71" if p["source_v71"] else "", "recorded", p["recorded"],
              "tokens", p["source_tokens"], "warm fields", len(p["warm"]["knowledge"]),
              "chars", sum(len(v) for v in p["warm"]["knowledge"].values()), "notes", len(p["warm"]["notes"]))
    print(json.dumps({"points": len(points), "pairs": len(table), "hard": sum(t["hard"] for t in table),
                      "failures": len(failures)}))


if __name__ == "__main__":
    main()
