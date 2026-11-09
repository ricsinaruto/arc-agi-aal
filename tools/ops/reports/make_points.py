"""Collector outputs -> scripted bed points. argv: base_points.json out_dir machines collector_dir [collector_dir ...]
Writes out_dir/points_all.json and out_dir/points_m<k>.json (points dealt over ``machines`` by game, round robin).
Run with python -I: the collector folders are downloaded data."""
import json, sys, collections
from pathlib import Path

KEEP_TRIMS = "--keep-trims" in sys.argv  # path 1: the scripts play through drains
sys.argv = [a for a in sys.argv if a != "--keep-trims"]
base = {p["game"][:4]: p for p in json.loads(Path(sys.argv[1]).read_text())["points"]}
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
machines = int(sys.argv[3])
TRIMS = ("context_drain", "compaction", "fraction_compaction", "loop_rule", "fresh_attempt", "fresh_attempt_context",
         "playbook_handoff", "context_rejection")
points, skipped = [], collections.Counter()
for root in sys.argv[4:]:
    bench = Path(root) / "arena/trials/level-bench"
    if not bench.is_dir():
        bench = Path(root) / "working/trials/level-bench"
    report = json.loads((bench / "level-bench.json").read_text())
    for row in report["attempts"]:
        folder = bench / "attempts" / row["label"]
        if row["outcome"] != "recorded" or not (folder / "script_stop.json").is_file():
            skipped[row["outcome"]] += 1
            continue
        kinds = collections.Counter()
        for line in (folder / "log.jsonl").open():
            if line.startswith('{"kind":"'):
                kinds[line[9:line.index('"', 9)]] += 1
        early = [] if KEEP_TRIMS else [k for k in TRIMS if kinds.get(k)]
        if early:
            skipped["trim before the stop: " + ",".join(early)] += 1
            continue
        replies = [json.loads(line) for line in (folder / "script.jsonl").read_text().splitlines()]
        stop = json.loads((folder / "script_stop.json").read_text())
        if stop["stop"]["requests"] != len(replies):
            skipped["reply count"] += 1
            continue
        game = row["game"][:4]
        points.append({**base[game], "id": f"{game}-t{Path(root).name[-1]}{row['index']:02d}",
                       "script": {"replies": replies, **stop},
                       "source": {"run": Path(root).name, "label": row["label"],
                                  "tokens": stop["stop"]["completion_tokens"], "levels": stop["stop"]["levels"],
                                  "steps": stop["stop"]["steps"], "requests": len(replies)}})
points.sort(key=lambda p: p["id"])
head = {"schema": 1, "rule": "trim bed: a fresh game replayed from recorded replies up to its first 112K-token request",
        "split": "dev"}
(out / "points_all.json").write_text(json.dumps({**head, "points": points}, separators=(",", ":")))
by_game = collections.defaultdict(list)
for p in points:
    by_game[p["game"][:4]].append(p)
deal, k = collections.defaultdict(list), 0
for game in sorted(by_game):
    for p in by_game[game]:
        deal[k % machines].append(p); k += 1
for m in range(machines):
    (out / f"points_m{m}.json").write_text(json.dumps({**head, "points": deal[m]}, separators=(",", ":")))
print("points", len(points), "skipped", dict(skipped))
print("per game", {g: len(v) for g, v in sorted(by_game.items())})
print("per machine", {m: [p["id"] for p in deal[m]] for m in range(machines)})
src = [p["source"] for p in points]
if src:
    import statistics
    print("script tokens median", statistics.median(s["tokens"] for s in src), "requests median", statistics.median(s["requests"] for s in src),
          "steps median", statistics.median(s["steps"] for s in src), "levels at stop", sorted(collections.Counter(s["levels"] for s in src).items()))
