"""No-harm check read-out. argv: parse_dir run_dir [run_dir ...]  (python -I)"""
import json, sys, re, collections, statistics
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from parse import turns

DEATH = re.compile(r"^(GAME_OVER at step|GAME OVER occurred|The game over occurred immediately|Do NOT re-submit|That fatal action animated)")
IMPOSSIBLE = re.compile(r"unreachable|unsolvable|impossible|cannot be (satisfied|completed|solved)|no solution exists", re.I)
arms = collections.defaultdict(lambda: collections.Counter())
scores = collections.defaultdict(list); paired = collections.defaultdict(dict); gaps = collections.defaultdict(list)
late = collections.defaultdict(lambda: [0, 0])
for root in sys.argv[2:]:
    bench = next(p for p in (Path(root) / "arena/trials/level-bench", Path(root) / "working/trials/level-bench") if p.is_dir())
    report = json.loads((bench / "level-bench.json").read_text())
    for r in report["attempts"]:
        a = arms[r["arm"]]
        a["attempts"] += 1; a["outcome_" + r["outcome"]] += 1
        if r["outcome"] in ("error", "censored"):
            continue
        scores[r["arm"]].append(r["game_score"]); paired[(Path(root).name, r["point"])][r["arm"]] = r["game_score"]
        a["levels"] += r["levels"]; a["tokens"] += r["completion_tokens"]; a["actions"] += r["total_actions"]
        for lv in r.get("level_rows") or []:
            if lv["at_tokens"] > 100000:
                late[r["arm"]][0] += lv["actions"]; late[r["arm"]][1] += 1
        path = bench / "attempts" / r["label"] / "duck-transcript.txt"
        if not path.is_file():
            a["no_transcript"] += 1; continue
        last_death = None
        for t in turns(str(path)):
            for name, text in t["sections"]:
                if name != "USER PROMPT":
                    continue
                a["turns"] += 1; a["user_chars"] += len(text)
                death = [line for line in text.split("\n") if DEATH.match(line)]
                if death:
                    a["death_turns"] += 1; a["death_chars"] += sum(len(x) for x in death)
                    if last_death is not None:
                        gaps[r["arm"]].append(t["action"] - last_death)
                    last_death = t["action"]
                a["stepback_turns"] += "STEP BACK" in text
                a["stepback_on_death"] += bool(death) and "STEP BACK" in text
                a["impossible_turns"] += bool(IMPOSSIBLE.search(text))
                a["wm_gameover_line"] += "- Recent findings: GAME_OVER at step" in text
                a["solvable_seen"] += 0
for arm, a in sorted(arms.items()):
    s = scores[arm]
    print(f"== {arm}: attempts {a['attempts']} " + str({k[8:]: v for k, v in a.items() if k.startswith('outcome_')}))
    if not s:
        continue
    print(f"   game score mean {statistics.mean(s):.2f} (sd {statistics.pstdev(s):.1f}), levels {a['levels']}, tokens {a['tokens']/1e6:.2f}M, actions {a['actions']}, actions per level {a['actions']/max(a['levels'],1):.1f}, "
          f"late (after 100K) actions per cleared level {late[arm][0]/max(late[arm][1],1):.1f} over {late[arm][1]}")
    t = max(a["turns"], 1)
    print(f"   turns {a['turns']}: deaths per 100 turns {100*a['death_turns']/t:.1f}, median actions between deaths {statistics.median(gaps[arm]) if gaps[arm] else None}, "
          f"death text share of user characters {100*a['death_chars']/max(a['user_chars'],1):.1f}% ({a['death_chars']//max(a['death_turns'],1)} chars per death), "
          f"STEP BACK per 100 turns {100*a['stepback_turns']/t:.1f} ({a['stepback_on_death']} on a death turn), "
          f"turns saying impossible/unreachable {a['impossible_turns']} ({100*a['impossible_turns']/t:.1f}%), carried GAME_OVER line in {100*a['wm_gameover_line']/t:.1f}% of turns")
diffs = [v["fix"] - v["base"] for v in paired.values() if "fix" in v and "base" in v]
if diffs:
    print(f"paired fix - base game score: mean {statistics.mean(diffs):+.2f}, sd of pairs {statistics.pstdev(diffs):.1f}, n {len(diffs)}, se {statistics.pstdev(diffs)/len(diffs)**.5:.2f}")
    print(f"   pairs: fix higher {sum(d > 0 for d in diffs)}, lower {sum(d < 0 for d in diffs)}, equal {sum(d == 0 for d in diffs)}; "
          f"fix higher by 30+ points {sum(d >= 30 for d in diffs)}, lower by 30+ {sum(d <= -30 for d in diffs)} (no game names: usable on held-out)")
