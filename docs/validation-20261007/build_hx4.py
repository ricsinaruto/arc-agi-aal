"""hx4 stuck-level bench bundle (7 October): the v7.1 harness on the locked DEV-fitted serving (INT3 requant + 37.5%
mask, Humming, 44 streams), one ``level-bench`` trial over stuck-level start points with paired arms.

It runs ``docs/validation-20261006/build_v71.py --restore-v6d`` unchanged (same study as the v7.1p bundles apart from
mask and overlay), then replaces the trial with the bench and ships the start points.

argv: output --points=<file[,file]> --arms=<json file or inline json> [--budget=100000] [--passes=2]
[--first-pass=0] [--concurrency=40] [--wall=5400] (bench wall, seconds) [--setup=1500] [--split=dev|holdout]
[--games=cn04,ls20] [--seed=0] [--serving=locked|v71p|gptq-all|v8] [--streams=44] [--lead-budget=80000] [--study=<json: study overrides for every arm>]
[--probe=<jsonl of second-reader prompts> --probe-variants=<json list> --probe-repeats=2] (rsi arm: arena.level_bench reader_probe)
"""
import json
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path.cwd()
MASK = "data/expert_prune/prune-t37-int3-requant-dev10-v1.json"
OVERLAY = ("richardcsaky/arc-expert-int3-requant-t37-dev10-v2:"
           "0afda482026406962cd29fb10aace26937d6366cafa3f1fb179914933897beae")


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


out = Path(sys.argv[1])
arms = arg("arms")
arms = json.loads(Path(arms).read_text() if arms and Path(arms).is_file() else arms) if arms else None
wall, setup = int(arg("wall", "5400")), int(arg("setup", "1500"))
own = sys.argv[:]
# locked: the DEV-fitted requant build; v71p: build_v71's default prune-t37; gptq-all: all experts with the GPTQ
# overlay the recorded v7.1 runs were served with (sft-flash-v71-500k-v2, b2-v71-24h-v1)
SERVING = {"locked": [f"--mask={MASK}", f"--overlay={OVERLAY}"], "v71p": [],
           "gptq-all": ["--all-experts", "--overlay=richardcsaky/arc-gptq-arc-excl5-v3:"
                        "525d629870920cfba55eab87e3f564b4ff8602e54d4b6bab99133dfd5a0ea9d8"]}
serving = arg("serving", "locked")
common = [str(out), "--dependencies", f"--streams={arg('streams', '44')}", "--wall=3600", "--mm-cache-gb=4"]
if serving == "v8":
    # hx5 (8 October): the v8 build as submitted (all-25 weights, its harness flags) with border_moves off
    sys.argv = ["build_v8.py", *common, "--off=border-moves"]
    runpy.run_path(str(ROOT / "docs/validation-20261007/build_v8.py"), run_name="__main__")
else:
    sys.argv = ["build_v71.py", *common, "--restore-v6d", *SERVING[serving]]
    runpy.run_path(str(ROOT / "docs/validation-20261006/build_v71.py"), run_name="__main__")
sys.argv = own
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
from build_arena import manifest  # noqa: E402

from arena.level_bench import settings  # noqa: E402

path = out / "configs/comparison.json"
study = json.loads(path.read_text())
merged = [json.loads(Path(f).read_text()) for f in arg("points", "data/stuck_bench/dev.json").split(",")]
merged = {**merged[0], "points": [p for m in merged for p in m["points"]]}
points = out / "configs/stuck_points.json"
points.write_text(json.dumps(merged, separators=(",", ":")) + "\n")
env = study["trials"][0].get("server_env")
study["trials"] = [{"name": "level-bench", "kind": "level-bench", **({"server_env": env} if env else {})}]
for key in ("competition", "b200", "live_progress"):
    study.pop(key, None)
study.update(json.loads(arg("study", "{}")))
games = arg("games")
study["level_bench"] = settings({
    "split": arg("split", "dev"), "games": games.split(",") if games else None,
    "concurrency": int(arg("concurrency", "40")), "window": study["windows"][0],
    "token_budget": int(arg("budget", "100000")), "attempt_wall_seconds": wall,
    "passes": int(arg("passes", "2")), "first_pass": int(arg("first-pass", "0")), "order_seed": int(arg("seed", "0")),
    "lead_token_budget": int(arg("lead-budget", "80000")),
    "start_points": "configs/stuck_points.json", "serving": serving, "arms": arms})
if study["level_bench"]["concurrency"] > study["active_limit"]:
    raise SystemExit("more workers than KV-gate streams")
if arg("probe"):
    shutil.copyfile(arg("probe"), out / "configs/reader_probe.jsonl")
    study["level_bench"] = settings({**study["level_bench"], "reader_probe": {
        "file": "configs/reader_probe.jsonl", "variants": json.loads(arg("probe-variants")),
        "repeats": int(arg("probe-repeats", "2"))}})
study["name"] = out.name
study["wall_seconds"] = wall
study["notebook_seconds"] = wall + setup
study["hx4"] = {"points": arg("points", "data/stuck_bench/dev.json"),
                "points_sha256": __import__("hashlib").sha256(points.read_bytes()).hexdigest()}
path.write_text(json.dumps(study, indent=2) + "\n")
identity = json.loads((out / "MANIFEST.json").read_text())
identity["files"] = manifest(out)
(out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
runpy.run_path(str(out / "bootstrap.py"))["verify"](out)
selected = [p for p in json.loads(points.read_text())["points"] if p["split"] == study["level_bench"]["split"]
            and (not games or p["game"][:4] in games.split(","))]
from arena.level_bench import plan  # noqa: E402

work = plan(selected, study["level_bench"]["passes"], 0, arms)
print(json.dumps({"output": str(out), "points": len(selected),
                  "attempts": {a: sum(1 for w in work if w[2]["name"] == a) for a in sorted({w[2]["name"] for w in work})},
                  "budget": study["level_bench"]["token_budget"], "concurrency": study["level_bench"]["concurrency"],
                  "wall": wall, "notebook_seconds": study["notebook_seconds"],
                  "overlay": study["server"]["expert_overlay"], "mask": study["server"].get("expert_prune")}))
