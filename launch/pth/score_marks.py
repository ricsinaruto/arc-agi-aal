"""score_marks.py RUN [--marks 100000,...] [--json OUT]: score of a Nebius play read at fixed per-try token marks (owner
28 Oct 11:00: the v8.5p ceiling and p75 held-out reads are scored at the 100K mark per game, the same way).

Per try (games/kNNN/log.jsonl): generated tokens = the running sum of completion tokens of its gameplay replies (the
"model" rows with purpose gameplay; what token_stop counts). Level-ups come from the runtime block of each
model_request (actions_executed, levels_completed), each observation (step, levels) and "finished" rows, each stamped with the tokens generated
before it was seen. At mark X a try counts the level-ups stamped <= X. It is covered at X when it generated >= X
tokens or finished (all levels, game over or a stop) before X. Try score = the official weighted formula (arc_agi
scorecard: level i scores min(115, 100 (baseline_i / actions_i)^2) when cleared, weight i + 1, capped at the cleared
weights' share); game score = mean over its clones; run score = mean over games. Scores print only at >= 95% coverage."""
import argparse
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "nebius-arc:arc-runs-bucket/runs/validation/"
ENV_ROOT = Path("./environment_files")
COVERAGE = 0.95


def try_score(ups, baselines):
    """ups: cumulative action counts at each level-up, in order."""
    per = [u - p for u, p in zip(ups, [0] + ups[:-1])]
    tot = cleared = 0.0
    for i, base in enumerate(baselines):
        s = min(115.0, (base / per[i]) ** 2 * 100) if i < len(per) and per[i] > 0 else 0.0
        cleared += (i + 1) if s > 0 else 0
        tot += s * (i + 1)
    total = len(baselines) * (len(baselines) + 1) / 2
    return min(tot / total, cleared / total * 100)


def parse_log(text):
    """{"ups": [(tokens, actions)], "tokens": total, "finished_at": tokens or None} of one try's log.jsonl text."""
    gen, levels, ups, finished = 0, 0, [], None
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        kind = row.get("kind")
        if kind == "model" and row.get("purpose", "gameplay") == "gameplay":
            usage = row.get("usage") or {}
            gen += int((usage.get("completion_tokens") if isinstance(usage, dict) else 0) or 0)
            continue
        a = c = None
        if kind == "model_request":
            rt = row.get("runtime") or {}
            a, c = rt.get("actions_executed"), rt.get("levels_completed")
        elif kind == "observation":  # after every action: a clear is seen before the next request
            a, c = row.get("step"), row.get("levels")
        elif kind == "finished":
            a, c = row.get("actions"), row.get("levels")
            finished = gen if finished is None else finished
        while a is not None and c is not None and c > levels:
            ups.append((gen, int(a)))
            levels += 1
    return {"ups": ups, "tokens": gen, "finished_at": finished}


def at_mark(t, mark):
    covered = t["tokens"] >= mark or (t["finished_at"] is not None and t["finished_at"] <= mark)
    return covered, [a for g, a in t["ups"] if g <= mark]


def score(tries, roster, marks, env_root=ENV_ROOT):
    """tries: {kNNN: parse_log result}; roster: [{"game_id", "source_game_id"}]."""
    base = {}
    out = {}
    for mark in marks:
        games, cov = {}, []
        for r in roster:
            t = tries.get(r["game_id"])
            if t is None:
                cov.append(False)
                continue
            src = r["source_game_id"]
            if src not in base:
                g, v = src.split("-")
                base[src] = json.loads((env_root / g / v / "metadata.json").read_text())["baseline_actions"]
            covered, ups = at_mark(t, mark)
            cov.append(covered)
            games.setdefault(src.split("-")[0], []).append((try_score(ups, base[src]), len(ups)))
        coverage = sum(cov) / max(len(cov), 1)
        per_game = {g: {"score": round(statistics.mean(s for s, _ in v), 3),
                        "levels": round(statistics.mean(n for _, n in v), 3), "tries": len(v)} for g, v in sorted(games.items())}
        out[mark] = {"coverage": round(coverage, 4), "tries": len(cov),
                     "score": round(statistics.mean(x["score"] for x in per_game.values()), 3) if per_game else None,
                     "levels_per_try": round(statistics.mean(x["levels"] for x in per_game.values()), 3) if per_game else None,
                     "reportable": coverage >= COVERAGE, "games": per_game}
    return out


def _cat(path):
    r = subprocess.run(["rclone", "cat", path], capture_output=True, timeout=300)
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None


def fetch(run):
    trials = BUCKET + f"{run}/arena/working/trials/"
    r = subprocess.run(["rclone", "lsf", "--dirs-only", trials], capture_output=True, text=True, timeout=120)
    trial = next((t for t in r.stdout.split() if t), None)
    if not trial:
        raise SystemExit(f"no trial under {trials}")
    roster = json.loads(_cat(trials + trial + "public-roster.json") or "[]")
    with ThreadPoolExecutor(16) as ex:
        logs = dict(zip([x["game_id"] for x in roster],
                        ex.map(lambda x: _cat(f"{trials}{trial}games/{x['game_id']}/log.jsonl"), roster)))
    return roster, {k: parse_log(v) for k, v in logs.items() if v is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--marks", default="100000")
    ap.add_argument("--json")
    a = ap.parse_args()
    marks = [int(x) for x in a.marks.split(",")]
    roster, tries = fetch(a.run)
    res = score(tries, roster, marks)
    for mark, r in res.items():
        head = f"{a.run} @{mark:,}: coverage {r['coverage']:.1%} of {r['tries']} tries"
        if not r["reportable"]:
            print(head, "(below 95%: no score)")
            continue
        print(head, f"score {r['score']:.2f}  levels/try {r['levels_per_try']:.2f}")
        for g, x in r["games"].items():
            print(f"  {g} {x['score']:7.2f}  levels {x['levels']:.2f}  n={x['tries']}")
    if a.json:
        Path(a.json).write_text(json.dumps({"run": a.run, "marks": res}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
