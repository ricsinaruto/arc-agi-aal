"""v8.2 against v7.1p at equal per-game generated tokens (play side). Run with python3 -I.

argv: <v8.2 dir> <v7.1p dir> [<v7.1p dir> ...] [--games=<environment_files>] [--curve=<arena.json>,<scorecard.json>]
A dir is a duck-transfer folder: games/k*/log.jsonl + public-roster.json (a live copy or a finished run).
--curve adds a finished run that has no game logs here: levels and score at equal tokens only.
Each clone is cut at T = min(tokens generated so far in v8.2, tokens in the reference); only clones started in both count.
Held-out clones appear in pooled numbers only; the per-game table is DEV games.
Copy of a live run: rclone copy <bucket>/<run>/arena/working/trials/duck-transfer/ <new dir> --exclude duck-transcript.txt
  --exclude "vllm-*" --exclude telemetry.jsonl --exclude admission.jsonl
"""
import collections
import glob
import json
import re
import statistics
import sys
import zlib
from pathlib import Path

LABELS = {}
MARKS = (25000, 50000, 100000, 150000, 200000, 300000)
DEV = set("cd82 cn04 ft09 lp85 ls20 re86 sb26 sk48 vc33 wa30".split())
OBS_ACTION = re.compile(r'"action":(null|"[^"]*"|\{[^}]*\})')
OBS_TAIL = re.compile(r'"levels":(\d+),"total_levels":(\d+),"state":"(\w+)"')
HELPERS = re.compile(r"animation_frames\(|animation_timeline\(|animated_steps\(")
LAST_ANIM = re.compile(r"last_animation_frames")
PLAYBOOK_WRITE = re.compile(r"(?<![a-z_])playbook\(\s*[^)\s]")
NOTE_WRITE = re.compile(r"(?<![a-z_])note\(\s*[^)\s]")
ERROR_NAME = re.compile(r"^(?:[\w.]*\.)?(\w+(?:Error|Exception|Interrupt|Refused|Timeout)\w*)(?::|$)", re.M)
VERDICT = re.compile(r"unsolvable|impossible|unreachable|cannot be (?:solved|completed|reached)|no way to (?:win|solve|reach)", re.I)
SAVED = re.compile(r"\[playbook\] saved, ([\d,]+) characters")
MEMORY = re.compile(r"(?<![a-z_])(history|transitions)\b")
COUNTED = ("context_drain", "loop_rule_fresh_context", "fresh_attempt", "dfr_noop_stop", "action_skipped", "stale_state",
           "hx_reflect", "stuck_notice", "length_truncation", "cap_hit_followup", "duck_python_interrupt", "playbook_shown",
           "text_count_mismatch", "deadline_request_timeout", "auto_reset")


def args():
    dirs, games, curves = [], None, []
    for a in sys.argv[1:]:
        if a.startswith("--games="):
            games = a[8:]
        elif a.startswith("--curve="):
            curves.append(a[8:].split(","))
        else:
            dirs.append(a)
    for d in dirs:
        name, _, path = d.partition("=")
        LABELS[path or name] = name if path else (Path(d).parts[-4] if Path(d).name == "duck-transfer" and len(Path(d).parts) > 4 else Path(d).name)
    dirs = [d.partition("=")[2] or d for d in dirs]
    return dirs, games or str(Path(__file__).resolve().parents[0]), curves


def baselines(games_dir):
    out = {}
    for f in glob.glob(games_dir + "/*/*/metadata.json"):
        m = json.loads(Path(f).read_text())
        out[m["game_id"]] = m["baseline_actions"]
    return out


def parse(path, limit=None):
    """One game log -> turns (one per model reply) with everything that happened until the next reply."""
    turns, cur, tok, level, on_level, short = [], None, 0, 0, 0, False
    with open(path, errors="replace") as fh:
        for line in fh:
            if not line.startswith('{"kind":"'):
                continue
            kind = line[9:line.index('"', 9)]
            if kind == "observation":
                action = OBS_ACTION.search(line[:300])
                tail = OBS_TAIL.search(line[-600:])
                if not tail:
                    continue
                acted = bool(action) and action.group(1) != "null"
                levels = int(tail.group(1))
                if cur is not None and acted:
                    i, j = line.find('"frames":'), line.rfind(',"levels":')
                    grids = line[i + 9:j]
                    k = grids.rfind('"],["')
                    cur["boards"].append((zlib.crc32((grids[k + 3:] if k >= 0 else grids).encode()), level,
                                          tail.group(3) == "GAME_OVER"))
                if cur is not None:
                    cur["acts"] += acted
                    cur["frames"] += acted and '"],["' in line  # more than one grid
                    if tail.group(3) == "GAME_OVER":
                        cur["deaths"] += 1
                    if '"full_reset":true' in line[-80:]:
                        cur["resets"] += 1
                on_level += acted
                if levels > level:
                    if cur is not None:
                        cur["clears"].append((levels, on_level, tok))
                    level, on_level = levels, 0
                elif levels < level:  # full game restart
                    level, on_level = levels, 0
            elif kind == "model":
                d = json.loads(line)
                usage = d.get("usage") or {}
                tok += usage.get("completion_tokens") or 0
                if limit is not None and tok > limit:
                    short = True
                    break
                msg = d.get("message") or {}
                cur = {"tok": tok, "ct": usage.get("completion_tokens") or 0, "pt": usage.get("prompt_tokens") or 0,
                       "level": level, "tool": bool(msg.get("tool_calls")), "acts": 0, "frames": 0, "deaths": 0, "resets": 0,
                       "clears": [], "py": 0, "err": 0, "out": 0, "helper": 0, "last_anim": 0, "pb": 0, "pb_chars": None,
                       "note": 0, "memory": 0, "n": collections.Counter(), "seconds": d.get("seconds") or 0.0,
                       "think": len(msg.get("reasoning") or msg.get("reasoning_content") or ""), "say": len(msg.get("content") or ""),
                       "code": 0, "errors": [], "pb_written": 0, "boards": [],
                       "verdict": bool(VERDICT.search(msg.get("reasoning") or msg.get("reasoning_content") or ""))}
                turns.append(cur)
            elif cur is None:
                continue
            elif kind == "duck_python":
                d = json.loads(line)
                code, res = d.get("code") or "", d.get("result") or {}
                text = (res.get("stdout") or "") + (res.get("stderr") or "") + (res.get("error") or "")
                cur["py"] += 1
                failed = bool(res.get("error") or res.get("returncode") or "Traceback (most recent" in text)
                cur["err"] += failed
                cur["code"] += len(code)
                if failed:
                    names = ERROR_NAME.findall(text)
                    cur["errors"].append(names[-1] if names else "other")
                cur["out"] += len(text)
                cur["helper"] += len(HELPERS.findall(code))
                cur["last_anim"] += bool(LAST_ANIM.search(code))
                cur["pb"] += bool(PLAYBOOK_WRITE.search(code))
                cur["note"] += bool(NOTE_WRITE.search(code))
                cur["memory"] += bool(MEMORY.search(code))
                saved = SAVED.findall(text)
                if saved:
                    cur["pb_chars"] = int(saved[-1].replace(",", ""))
                    cur["pb_written"] += sum(int(x.replace(",", "")) for x in saved)
            elif kind == "duck_python_dfr_guards":
                cur["n"]["stale_state"] += line.count('"kind":"stale_state"')
            elif kind == "text_count_mismatch":
                d = json.loads(line)
                cur["n"]["text_count_mismatch"] += 1
                cur["n"]["text_count_mismatch_tokens"] += abs((d.get("served") or 0) - (d.get("admitted") or 0))
            elif kind in COUNTED:
                cur["n"][kind] += 1
                if kind == "context_drain":
                    d = json.loads(line)
                    cur.setdefault("drains", []).append((d.get("tokens_before"), d.get("tokens_after"), d.get("target")))
                elif kind == "playbook_shown":
                    d = json.loads(line)
                    cur.setdefault("shown", []).append(d.get("chars") or 0)
    return turns, short


def credit(base, actions):
    return min(115.0, 100.0 * (base / actions) ** 2) if actions else 0.0


def game_score(levels_credit, total):
    """levels_credit: {level number: credit}."""
    return min(sum(j * c for j, c in levels_credit.items()), 100.0 * sum(levels_credit)) / (total * (total + 1) / 2)


def summarize(turns, cut, base):
    kept = [t for t in turns if t["tok"] <= cut]
    clears, prev = {}, 0
    for t in kept:
        for level, actions, tok in t["clears"]:
            clears[level] = {"actions": actions, "tokens": tok - prev, "at": tok,
                             "credit": credit(base[level - 1], actions) if level - 1 < len(base) else 0.0}
            prev = tok
    return kept, clears, game_score({k: v["credit"] for k, v in clears.items()}, len(base))


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else float("nan")


def paired(diffs):
    if len(diffs) < 2:
        return "n/a"
    return f"{mean(diffs):+.2f} (se {statistics.pstdev(diffs) / len(diffs) ** .5:.2f}, n {len(diffs)})"


def behaviour(name, rows):
    """rows: list of kept-turn lists (one per clone)."""
    turns = [t for r in rows for t in r]
    n = max(len(turns), 1)
    tokens = sum(t["ct"] for t in turns)
    per100k = 100000.0 / max(tokens, 1)
    counts = collections.Counter()
    for t in turns:
        counts.update(t["n"])
    act = [t for t in turns if t["acts"]]
    explore = [t for t in turns if t["py"] and not t["acts"]]
    bare = [t for t in turns if not t["tool"]]
    py = sum(t["py"] for t in turns)
    actions = sum(t["acts"] for t in turns)
    # tokens the harness and the tool output add to the context per turn (next prompt - this prompt - this reply), no trim between
    added = [(b["pt"] - a["pt"] - a["ct"], a) for r in rows for a, b in zip(r, r[1:]) if not a["n"]["context_drain"] and b["pt"] > a["pt"]]
    add_act = [x for x, a in added if a["acts"]]
    add_exp = [x for x, a in added if not a["acts"]]
    ends = [r[-1]["pt"] + r[-1]["ct"] for r in rows if r]
    out = {
        "turns": len(turns), "tokens M": round(tokens / 1e6, 3),
        "tokens per turn mean / median": f"{tokens / n:.0f} / {med([t['ct'] for t in turns]):.0f}",
        "prompt tokens per turn median": f"{med([t['pt'] for t in turns]):.0f}",
        "context added per turn by harness + tool output": f"{mean([x for x, _ in added]):.0f} (action turns {mean(add_act):.0f}, exploration {mean(add_exp):.0f})",
        "context added per generated token": f"{sum(x for x, _ in added) / max(sum(a['ct'] for _, a in added), 1):.2f}",
        "context size at the cut median / max": f"{med(ends):.0f} / {max(ends, default=0)}",
        "seconds per turn median": f"{med([t['seconds'] for t in turns]):.1f}",
        "action turns %": f"{100 * len(act) / n:.1f}", "pure exploration turns %": f"{100 * len(explore) / n:.1f}",
        "no-tool turns %": f"{100 * len(bare) / n:.1f}",
        "tokens in exploration turns %": f"{100 * sum(t['ct'] for t in explore) / max(tokens, 1):.1f}",
        "tokens per action turn / exploration turn": f"{mean([t['ct'] for t in act]):.0f} / {mean([t['ct'] for t in explore]):.0f}",
        "actions": actions, "actions per 100K tokens": f"{actions * per100k:.1f}",
        "actions per action turn": f"{actions / max(len(act), 1):.2f}",
        "tokens per action": f"{tokens / max(actions, 1):.0f}",
        "deaths per 100K tokens": f"{sum(t['deaths'] for t in turns) * per100k:.2f} ({sum(t['deaths'] for t in turns)})",
        "resets (full) per 100K": f"{sum(t['resets'] for t in turns) * per100k:.2f} ({sum(t['resets'] for t in turns)})",
        "python calls": py, "python errors %": f"{100 * sum(t['err'] for t in turns) / max(py, 1):.1f} ({sum(t['err'] for t in turns)})",
        "python error types": ", ".join(f"{k} {v}" for k, v in collections.Counter(e for t in turns for e in t["errors"]).most_common(6)),
        "reply chars per turn: reasoning / text / code": f"{mean([t['think'] for t in turns]):.0f} / {mean([t['say'] for t in turns]):.0f} / {mean([t['code'] for t in turns]):.0f}",
        "turns reasoning 'impossible/unsolvable/unreachable' %": f"{100 * sum(t['verdict'] for t in turns) / n:.1f} ({sum(t['verdict'] for t in turns)})",
        "python output chars per call": f"{sum(t['out'] for t in turns) / max(py, 1):.0f}",
        "turns using history/transitions %": f"{100 * sum(bool(t['memory']) for t in turns) / n:.1f}",
        "no-op stops per 100K": f"{counts['dfr_noop_stop'] * per100k:.2f} ({counts['dfr_noop_stop']})",
        "skipped repeats per 100K": f"{counts['action_skipped'] * per100k:.2f} ({counts['action_skipped']})",
        "stale-state refusals per 100K": f"{counts['stale_state'] * per100k:.2f} ({counts['stale_state']})",
        "STEP BACK (reflect) per 100K": f"{counts['hx_reflect'] * per100k:.2f} ({counts['hx_reflect']})",
        "stuck notices per 100K": f"{counts['stuck_notice'] * per100k:.2f} ({counts['stuck_notice']})",
        "reply-cap hits": counts["length_truncation"], "python timeouts": counts["duck_python_interrupt"],
        "trims (context_drain)": counts["context_drain"], "loop-rule fresh contexts": counts["loop_rule_fresh_context"],
        "text_count_mismatch (tokens off, sum)": f"{counts['text_count_mismatch']} ({counts['text_count_mismatch_tokens']})", "request timeouts": counts["deadline_request_timeout"],
        "animated actions % (frames > 1)": f"{100 * sum(t['frames'] for t in turns) / max(actions, 1):.1f}",
    }
    helper_turns = [t for t in turns if t["helper"]]
    out["animation helper calls"] = f"{sum(t['helper'] for t in turns)} in {len(helper_turns)} turns ({100 * len(helper_turns) / n:.1f}%)"
    out["  their output, est. tokens (chars / 3.4)"] = (f"{sum(t['out'] for t in helper_turns) / 3.4:.0f} total, "
                                                       f"{mean([t['out'] for t in helper_turns]) / 3.4:.0f} per turn "
                                                       f"(other python turns {mean([t['out'] for t in turns if t['py'] and not t['helper']]) / 3.4:.0f})")
    out["last_animation_frames turns %"] = f"{100 * sum(bool(t['last_anim']) for t in turns) / n:.1f}"
    writes = [t for t in turns if t["pb"]]
    sizes = [max((t["pb_chars"] for t in r if t["pb_chars"] is not None), default=None) for r in rows]
    last = [next((t["pb_chars"] for t in reversed(r) if t["pb_chars"] is not None), None) for r in rows]
    shown = [c for t in turns for c in t.get("shown", [])]
    out["playbook writes"] = (f"{len(writes)} in {sum(1 for r in rows if any(t['pb'] for t in r))} of {len(rows)} games; "
                              f"tokens of those turns {sum(t['ct'] for t in writes)} ({100 * sum(t['ct'] for t in writes) / max(tokens, 1):.1f}%)")
    out["playbook chars now median / max"] = f"{med(last):.0f} / {max([s for s in sizes if s is not None], default=0)}"
    out["playbook shown after a cut"] = f"{len(shown)} (chars median {med(shown):.0f})"
    out["playbook text written, est. tokens (chars / 3.4)"] = f"{sum(t['pb_written'] for t in turns) / 3.4:.0f} ({100 * sum(t['pb_written'] for t in turns) / 3.4 / max(tokens, 1):.1f}% of generated)"
    notes = [t for t in turns if t["note"]]
    out["note() write turns"] = f"{len(notes)}; tokens of those turns {sum(t['ct'] for t in notes)} ({100 * sum(t['ct'] for t in notes) / max(tokens, 1):.1f}%)"
    drains = [d for t in turns for d in t.get("drains", [])]
    if drains:
        out["drain before / after / target median"] = f"{med([d[0] for d in drains]):.0f} / {med([d[1] for d in drains]):.0f} / {med([d[2] for d in drains]):.0f}"
    return out


def compare(label_a, a_dir, label_b, b_dir, base_by_game):
    roster = {r["game_id"]: r["source_game_id"] for r in json.loads(Path(a_dir, "public-roster.json").read_text())}
    other = {r["game_id"]: r["source_game_id"] for r in json.loads(Path(b_dir, "public-roster.json").read_text())}
    rows = []
    for clone, source in sorted(roster.items()):
        fa, fb = Path(a_dir, "games", clone, "log.jsonl"), Path(b_dir, "games", clone, "log.jsonl")
        if other.get(clone) != source or not fa.is_file() or not fb.is_file():
            continue
        ta, _ = parse(fa)
        if not ta:
            continue
        tb, more = parse(fb, limit=ta[-1]["tok"])
        if not tb:
            continue
        # both runs have a reply in progress at the cut; a reference game that ended earlier sets the cut
        cut = ta[-1]["tok"] if more else min(ta[-1]["tok"], tb[-1]["tok"])
        base = base_by_game[source]
        ka, ca, sa = summarize(ta, cut, base)
        kb, cb, sb = summarize(tb, cut, base)
        rows.append({"clone": clone, "game": source[:4], "source": source, "cut": cut, "a": (ka, ca, sa), "b": (kb, cb, sb), "total": len(base)})
    print(f"\n===== {label_a} against {label_b}: {len(rows)} clones started in both of {len(roster)}; "
          f"equal tokens per clone, total {sum(r['cut'] for r in rows) / 1e6:.2f}M each (median cut {med([r['cut'] for r in rows]):.0f})")
    for pool, keep in (("all", lambda g: True), ("DEV clones", lambda g: g in DEV), ("held-out clones", lambda g: g not in DEV)):
        sel = [r for r in rows if keep(r["game"])]
        if not sel:
            continue
        la, lb = sum(len(r["a"][1]) for r in sel), sum(len(r["b"][1]) for r in sel)
        ca = [v["credit"] for r in sel for v in r["a"][1].values()]
        cb = [v["credit"] for r in sel for v in r["b"][1].values()]
        tokens = sum(r["cut"] for r in sel) / 1e6
        print(f"  {pool:16s} n {len(sel):3d}  score {mean(r['a'][2] for r in sel):6.2f} vs {mean(r['b'][2] for r in sel):6.2f}  "
              f"paired {paired([r['a'][2] - r['b'][2] for r in sel])}  levels {la} vs {lb}  "
              f"levels per Mtok {la / tokens:.1f} vs {lb / tokens:.1f}  credit per cleared level {mean(ca):.1f} vs {mean(cb):.1f}  "
              f"clones with a level {sum(bool(r['a'][1]) for r in sel)} vs {sum(bool(r['b'][1]) for r in sel)}")
    gain = collections.Counter()
    for r in rows:
        if r["game"] in DEV:
            gain[r["game"]] += r["a"][2] - r["b"][2]
    if gain:
        top = max(gain, key=lambda g: abs(gain[g]))
        rest = [r["a"][2] - r["b"][2] for r in rows if r["game"] != top]
        print(f"  largest single DEV game: {top} carries {gain[top] / len(rows):+.2f} of the paired difference; without its clones {paired(rest)}")
    n110 = len(roster)
    print(f"  score over all {n110} clones (unstarted = 0): {sum(r['a'][2] for r in rows) / n110:.2f} vs {sum(r['b'][2] for r in rows) / n110:.2f}")
    print("  per level number (levels cleared in both runs, same clone): n, tokens on the level median, actions median, credit mean")
    for level in range(1, 8):
        both = [(r["a"][1][level], r["b"][1][level]) for r in rows if level in r["a"][1] and level in r["b"][1]]
        only_a = sum(1 for r in rows if level in r["a"][1] and level not in r["b"][1])
        only_b = sum(1 for r in rows if level in r["b"][1] and level not in r["a"][1])
        if not both and not only_a and not only_b:
            continue
        print(f"    level {level}: both {len(both):3d} (only {label_a} {only_a}, only {label_b} {only_b})  "
              f"tokens {med([x['tokens'] for x, _ in both]):.0f} vs {med([y['tokens'] for _, y in both]):.0f} "
              f"(paired ratio median {med([x['tokens'] / max(y['tokens'], 1) for x, y in both]):.2f})  "
              f"actions {med([x['actions'] for x, _ in both]):.0f} vs {med([y['actions'] for _, y in both]):.0f}  "
              f"credit {mean(x['credit'] for x, _ in both):.1f} vs {mean(y['credit'] for _, y in both):.1f}")
    for level in (1, 2, 3):
        ta = [r["a"][1][level]["at"] for r in rows if level in r["a"][1]]
        tb = [r["b"][1][level]["at"] for r in rows if level in r["b"][1]]
        print(f"    reached level {level} clear: {len(ta)} vs {len(tb)} clones; game tokens at the clear median {med(ta):.0f} vs {med(tb):.0f}")
    stuck_a = [r["cut"] for r in rows if not r["a"][1]]
    stuck_b = [r["cut"] for r in rows if not r["b"][1]]
    print(f"    clones with no level yet: {len(stuck_a)} vs {len(stuck_b)} (their tokens median {med(stuck_a):.0f} vs {med(stuck_b):.0f})")
    ba, bb = behaviour(label_a, [r["a"][0] for r in rows]), behaviour(label_b, [r["b"][0] for r in rows])
    print(f"  behaviour at equal tokens{'':24s}{label_a:>34s}   {label_b}")
    for key in ba:
        print(f"    {key:44s}{str(ba[key]):>34s}   {bb.get(key)}")
    for name, turns_of in (("first level only (level 1 turns)", lambda ts: [t for t in ts if t["level"] == 0]),):
        fa = behaviour(label_a, [turns_of(r["a"][0]) for r in rows])
        fb = behaviour(label_b, [turns_of(r["b"][0]) for r in rows])
        print(f"  {name}")
        for key in ("turns", "tokens M", "tokens per turn mean / median", "pure exploration turns %", "actions per 100K tokens", "python errors %"):
            print(f"    {key:44s}{str(fa[key]):>34s}   {fb[key]}")
    print("  DEV games (clones pooled per game): n, levels, score")
    by = collections.defaultdict(list)
    for r in rows:
        if r["game"] in DEV:
            by[r["game"]].append(r)
    for game, sel in sorted(by.items()):
        print(f"    {game} n {len(sel)}  tokens {sum(r['cut'] for r in sel) / 1e3:.0f}K  levels {sum(len(r['a'][1]) for r in sel)} vs "
              f"{sum(len(r['b'][1]) for r in sel)}  score {mean(r['a'][2] for r in sel):.1f} vs {mean(r['b'][2] for r in sel):.1f}  "
              f"deaths {sum(t['deaths'] for r in sel for t in r['a'][0])} vs {sum(t['deaths'] for r in sel for t in r['b'][0])}")
    return rows


def flow(turns, clears, cut):
    """One clone's kept turns as a flat record: actions (tokens, level, board, death, revisit, no change), trims, deaths."""
    actions, trims, seen, prev, level_now = [], [], set(), None, None
    clear_at = {level: c["at"] for level, c in clears.items()}
    last_clear = max(clear_at.values(), default=0)
    pb_chars, pb_first = None, None
    for t in turns:
        for board, level, death in t["boards"]:
            if level != level_now:
                seen, level_now = set(), level
            actions.append({"tok": t["tok"], "level": level, "death": death, "revisit": board in seen,
                            "same": board == prev, "trims": len(trims)})
            seen.add(board)
            prev = board
        if t["pb_chars"] is not None:
            pb_chars = t["pb_chars"]
        for before, after, target in t.get("drains", []):
            nxt = clear_at.get(t["level"] + 1)
            trims.append({"tok": t["tok"], "level": t["level"], "index": len(actions), "pb": pb_chars,
                          "on_level": t["tok"] - max([a for a in clear_at.values() if a <= t["tok"]], default=0),
                          "followed": cut - t["tok"], "cleared_in": None if nxt is None or nxt < t["tok"] else nxt - t["tok"]})
    return {"actions": actions, "trims": trims, "stuck_tokens": cut - last_clear, "turns": turns}


def share(xs):
    xs = list(xs)
    return f"{100 * sum(xs) / len(xs):.1f}% (n {len(xs)})" if xs else "n/a"


def deep(label_a, label_b, rows, base_by_game):
    """Second part: where the tokens went and what the trims carried (equal tokens per clone, the cut of ``compare``)."""
    fa = [flow(r["a"][0], r["a"][1], r["cut"]) for r in rows]
    fb = [flow(r["b"][0], r["b"][1], r["cut"]) for r in rows]
    print(f"  --- score at equal tokens per clone, by token mark (clones with that many tokens in both): {label_a} vs {label_b}")
    for mark in MARKS:
        sel = [r for r in rows if r["cut"] >= mark]
        if len(sel) < 5:
            continue
        sa, sb, la, lb = [], [], 0, 0
        for r in sel:
            base = base_by_game[r["source"]]
            ca = {k: v["credit"] for k, v in r["a"][1].items() if v["at"] <= mark}
            cb = {k: v["credit"] for k, v in r["b"][1].items() if v["at"] <= mark}
            sa.append(game_score(ca, len(base))); sb.append(game_score(cb, len(base))); la += len(ca); lb += len(cb)
        dev = [x - y for x, y, r in zip(sa, sb, sel) if r["game"] in DEV]
        held = [x - y for x, y, r in zip(sa, sb, sel) if r["game"] not in DEV]
        print(f"    {mark // 1000:4d}K  n {len(sel):3d}  score {mean(sa):6.2f} vs {mean(sb):6.2f}  paired {paired([x - y for x, y in zip(sa, sb)])}"
              f"  levels {la} vs {lb}  DEV {paired(dev)}  held-out {paired(held)}")
    print("  --- by level number: clones that reached it, cleared it (rate), tokens on it (all clears, median), credit (all clears, mean)")
    for pool, keep in (("held-out", lambda g: g not in DEV), ("DEV", lambda g: g in DEV)):
        for level in range(1, 5):
            ra = [r for r in rows if keep(r["game"]) and (level == 1 or level - 1 in r["a"][1])]
            rb = [r for r in rows if keep(r["game"]) and (level == 1 or level - 1 in r["b"][1])]
            ca = [r["a"][1][level] for r in ra if level in r["a"][1]]
            cb = [r["b"][1][level] for r in rb if level in r["b"][1]]
            print(f"    {pool:8s} level {level}: reached {len(ra):3d} vs {len(rb):3d}  cleared {len(ca):3d} vs {len(cb):3d}  tokens "
                  f"{med([c['tokens'] for c in ca]):.0f} vs {med([c['tokens'] for c in cb]):.0f}  credit {mean(c['credit'] for c in ca):.1f} vs {mean(c['credit'] for c in cb):.1f}")
    for level in range(1, 9):
        ra = [r for r in rows if level == 1 or level - 1 in r["a"][1]]
        rb = [r for r in rows if level == 1 or level - 1 in r["b"][1]]
        ca = [r["a"][1][level] for r in ra if level in r["a"][1]]
        cb = [r["b"][1][level] for r in rb if level in r["b"][1]]
        if not ra and not rb:
            continue
        print(f"    level {level}: reached {len(ra):3d} vs {len(rb):3d}  cleared {len(ca):3d} vs {len(cb):3d} "
              f"({100 * len(ca) / max(len(ra), 1):.0f}% vs {100 * len(cb) / max(len(rb), 1):.0f}%)  tokens {med([c['tokens'] for c in ca]):.0f} vs "
              f"{med([c['tokens'] for c in cb]):.0f}  actions {med([c['actions'] for c in ca]):.0f} vs {med([c['actions'] for c in cb]):.0f}"
              f"  credit {mean(c['credit'] for c in ca):.1f} vs {mean(c['credit'] for c in cb):.1f}")
    total = sum(r["cut"] for r in rows)
    for name, flows, side in ((label_a, fa, "a"), (label_b, fb, "b")):
        open_tokens = sum(f["stuck_tokens"] for f in flows)
        none = sum(f["stuck_tokens"] for f, r in zip(flows, rows) if not r[side][1])
        cleared = sum(c["tokens"] for r in rows for c in r[side][1].values())
        long = sum(f["stuck_tokens"] for f in flows if f["stuck_tokens"] >= 100000)
        print(f"    {name}: tokens on cleared levels {100 * cleared / total:.1f}%, on the level still open at the cut {100 * open_tokens / total:.1f}% "
              f"(of which games with no level {100 * none / total:.1f}%, open levels past 100K {100 * long / total:.1f}% in "
              f"{sum(f['stuck_tokens'] >= 100000 for f in flows)} clones); tokens per cleared level {cleared / max(sum(len(r[side][1]) for r in rows), 1):.0f}; "
              f"all tokens per level {total / max(sum(len(r[side][1]) for r in rows), 1):.0f}")
    print(f"  --- trims (context_drain) and what follows them{'':16s}{label_a:>30s}   {label_b}")

    def trim_stats(flows):
        trims = [t for f in flows for t in f["trims"]]
        acts = [a for f in flows for a in f["actions"]]
        pre = [a for f in flows for t in f["trims"] for a in f["actions"][max(0, t["index"] - 30):t["index"]]]
        post = [a for f in flows for t in f["trims"] for a in f["actions"][t["index"]:t["index"] + 30]]
        judged = [t for t in trims if t["followed"] >= 50000 or (t["cleared_in"] is not None and t["cleared_in"] <= 50000)]
        early = [t for t in judged if t["on_level"] < 60000]
        late = [t for t in judged if t["on_level"] >= 60000]
        ok = lambda ts: share(t["cleared_in"] is not None and t["cleared_in"] <= 50000 for t in ts)  # noqa: E731
        before = [a for a in acts if a["trims"] == 0]
        after = [a for a in acts if a["trims"] > 0]
        tok_after = sum(max(0, f["turns"][-1]["tok"] - f["trims"][0]["tok"]) for f in flows if f["trims"] and f["turns"])
        tok_before = sum((f["trims"][0]["tok"] if f["trims"] else (f["turns"][-1]["tok"] if f["turns"] else 0)) for f in flows)
        return {
            "trims / clones with one": f"{len(trims)} / {sum(bool(f['trims']) for f in flows)}",
            "game tokens at the first trim median": f"{med([f['trims'][0]['tok'] for f in flows if f['trims']]):.0f}",
            "tokens on the level at a trim median": f"{med([t['on_level'] for t in trims]):.0f}",
            "levels cleared per 100K tokens before first trim / after": "%.2f / %.2f (%d after)" % (
                1e5 * sum(sum(len(t["clears"]) for t in f["turns"] if not f["trims"] or t["tok"] <= f["trims"][0]["tok"]) for f in flows) / max(tok_before, 1),
                1e5 * sum(sum(len(t["clears"]) for t in f["turns"] if t["tok"] > f["trims"][0]["tok"]) for f in flows if f["trims"]) / max(tok_after, 1),
                sum(sum(len(t["clears"]) for t in f["turns"] if t["tok"] > f["trims"][0]["tok"]) for f in flows if f["trims"])),
            "level in play cleared within 50K after a trim": ok(judged),
            "  trim under 60K tokens into the level": ok(early),
            "  trim 60K or more into the level": ok(late),
            "actions: board seen before on the level, all": share(a["revisit"] for a in acts),
            "  before the first trim / after": f"{share(a['revisit'] for a in before)} / {share(a['revisit'] for a in after)}",
            "  30 actions before a trim / 30 after": f"{share(a['revisit'] for a in pre)} / {share(a['revisit'] for a in post)}",
            "actions: no board change, before first trim / after": f"{share(a['same'] for a in before)} / {share(a['same'] for a in after)}",
            "  30 actions before a trim / 30 after": f"{share(a['same'] for a in pre)} / {share(a['same'] for a in post)}",
            "actions per 100K tokens before first trim / after": f"{1e5 * len(before) / max(tok_before, 1):.1f} / {1e5 * len(after) / max(tok_after, 1):.1f}",
            "playbook at a trim: empty / chars median": f"{sum(t['pb'] is None for t in trims)} of {len(trims)} / {med([t['pb'] for t in trims]):.0f}",
            "playbook writes per 100K before first trim / after": "%.2f / %.2f" % (
                1e5 * sum(t["pb"] for f in flows for t in f["turns"] if not f["trims"] or t["tok"] <= f["trims"][0]["tok"]) / max(tok_before, 1),
                1e5 * sum(t["pb"] for f in flows if f["trims"] for t in f["turns"] if t["tok"] > f["trims"][0]["tok"]) / max(tok_after, 1)),
            "note() turns per 100K before first trim / after": "%.2f / %.2f" % (
                1e5 * sum(bool(t["note"]) for f in flows for t in f["turns"] if not f["trims"] or t["tok"] <= f["trims"][0]["tok"]) / max(tok_before, 1),
                1e5 * sum(bool(t["note"]) for f in flows if f["trims"] for t in f["turns"] if t["tok"] > f["trims"][0]["tok"]) / max(tok_after, 1)),
        }

    ta, tb = trim_stats(fa), trim_stats(fb)
    for key in ta:
        print(f"    {key:52s}{str(ta[key]):>34s}   {tb[key]}")
    print(f"  --- deaths{'':52s}{label_a:>30s}   {label_b}")

    def death_stats(flows):
        per = collections.Counter()
        gaps = []
        for i, f in enumerate(flows):
            last = None
            for n, a in enumerate(f["actions"]):
                if a["death"]:
                    per[(i, a["level"])] += 1
                    if last is not None and last[0] == a["level"]:
                        gaps.append(n - last[1])
                    last = (a["level"], n)
        deaths = sum(per.values())
        tokens = sum(f["turns"][-1]["tok"] for f in flows if f["turns"])
        return {"deaths (per 100K tokens)": f"{deaths} ({1e5 * deaths / max(tokens, 1):.2f})",
                "levels with a death / with 2+ / with 3+": f"{len(per)} / {sum(v >= 2 for v in per.values())} / {sum(v >= 3 for v in per.values())}",
                "deaths that repeat a death on the same level": share([True] * (deaths - len(per)) + [False] * len(per)),
                "actions between deaths on a level median": f"{med(gaps):.0f}",
                "reasoning says impossible / unsolvable, turns": share(t["verdict"] for f in flows for t in f["turns"])}

    da, db = death_stats(fa), death_stats(fb)
    for key in da:
        print(f"    {key:52s}{str(da[key]):>34s}   {db[key]}")
    # animation helpers against the level's outcome (v8.2 side; pooled)
    used = collections.Counter()
    for r in rows:
        levels = collections.defaultdict(int)
        for t in r["a"][0]:
            levels[t["level"]] += t["helper"]
        for level, n in levels.items():
            used[(n > 0, level + 1 in r["a"][1])] += 1
    print(f"  --- {label_a} animation helpers by level attempt: used on {used[(True, True)] + used[(True, False)]} level attempts "
          f"({used[(True, True)]} cleared), not used on {used[(False, True)] + used[(False, False)]} ({used[(False, True)]} cleared)")


def curve(label_a, a_dir, arena_path, card_path, base_by_game):
    """Finished run without logs here: levels and score at each clone's v8.2 token count."""
    roster = {r["game_id"]: r["source_game_id"] for r in json.loads(Path(a_dir, "public-roster.json").read_text())}
    arena = json.loads(Path(arena_path).read_text())
    cards = {e["id"]: e["runs"][0] for e in json.loads(Path(card_path).read_text())["environments"]}
    rows = []
    for g in arena["games"]:
        clone = g["game_id"]
        fa = Path(a_dir, "games", clone, "log.jsonl")
        if clone not in roster or not fa.is_file():
            continue
        ta, _ = parse(fa)
        if not ta:
            continue
        base = base_by_game[roster[clone]]
        if len(base) != g.get("total_levels"):
            print("  roster mismatch at", clone)
            return
        cut = min(ta[-1]["tok"], g["completion_tokens"])
        _, ca, sa = summarize(ta, cut, base)
        scores = cards[clone]["level_scores"]
        got = {t["level"]: scores[t["level"] - 1] for t in g.get("time_to_level") or [] if t["budget_seconds"] <= cut}
        rows.append((roster[clone][:4], cut, len(ca), sa, len(got), game_score(got, len(base))))
    print(f"\n===== {label_a} against {arena_path} (levels and score only): {len(rows)} clones, {sum(r[1] for r in rows) / 1e6:.2f}M tokens each")
    for pool, keep in (("all", lambda g: True), ("DEV clones", lambda g: g in DEV), ("held-out clones", lambda g: g not in DEV)):
        sel = [r for r in rows if keep(r[0])]
        if sel:
            print(f"  {pool:16s} n {len(sel):3d}  score {mean(r[3] for r in sel):6.2f} vs {mean(r[5] for r in sel):6.2f}  "
                  f"paired {paired([r[3] - r[5] for r in sel])}  levels {sum(r[2] for r in sel)} vs {sum(r[4] for r in sel)}")
    for game in sorted(DEV):
        sel = [r for r in rows if r[0] == game]
        if sel:
            print(f"    {game} n {len(sel)}  levels {sum(r[2] for r in sel)} vs {sum(r[4] for r in sel)}  score {mean(r[3] for r in sel):.1f} vs {mean(r[5] for r in sel):.1f}")


def main():
    dirs, games_dir, curves = args()
    if not Path(games_dir, "cd82").is_dir():
        games_dir = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "environment_files")
    base = baselines(games_dir)
    for ref in dirs[1:]:
        deep(LABELS[dirs[0]], LABELS[ref], compare(LABELS[dirs[0]], dirs[0], LABELS[ref], ref, base), base)
    for arena_path, card_path in curves:
        curve(LABELS[dirs[0]], dirs[0], arena_path, card_path, base)


main()
