"""build_tpb2.py ARM [TAG]: p75 vs p75d4 serving throughput read tpb2. ARM: p75 | d4.

Bundle = the v87g held-out play bundle of that model (the research stack with v87b's KV gate, stream rule and prefix
caching; MTP 3) with one serving-bench trial: resident points at 16K / 50K / 100K prompt tokens over 8 real transcripts
of 8 different games (configs/bench/t*.txt), streams = the play rule on the served pool (capped by what fits at prompt
+ 4,096), plus p75's count on the d4 pool (matched S) and 48 (the context cost curve); one prefill pass, then 3
interleaved rounds of 4,096 forced tokens per stream count. throughput_bench.py from origin/sft-prune (resident
points, steady_window). The 27 Oct bench's 100K drop was its 64 requests on a 53-fit gate (re-prefill)."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

W = Path("./.claude/worktrees")
DIST = W / "sft-prune/dist"
SRC = {"p75": DIST / "qadv2-v87g-p75-ho10x8-v1", "d4": DIST / "qadv2-v87g-d0-ho10x9-v1"}
P75_CAPACITY = 2571  # p75's served pool (2,679 blocks x kv_margin 0.96; b2-qadv2-tpb-p75-v1a bench.json capacity_blocks)
TRANSCRIPTS = [  # >= 420K chars, distinct first frames (pick_transcripts.py over foldA-in-13287)
    "av2-i8s10-c100-60m-v3-nebius-20261015/k003", "av2-i8s10-c100-60m-v3-nebius-20261015/k007",
    "av2-i8s10-c100-60m-v3-nebius-20261015/k011", "av2-i8s10-c100-60m-v3-nebius-20261015/k018",
    "av2-i8x10-c100-60m-v1-nebius-20261015/k010", "av2-mmh10-c100-60m-v1b-nebius-20261015/k012",
    "av2-mmh10-c100-60m-v1b-nebius-20261015/k024", "av2-mmh10-c100-60m-v1c-nebius-20261015/k035"]
CORPUS = Path("~/arc-sft-data/foldA-in-13287")
arm = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else "v1"
src = SRC[arm]
out = DIST / f"qadv2-tpb2-{arm}-{tag}"
run = f"b2-qadv2-tpb2-{arm}-{tag}a"
old = json.loads((src / "configs/comparison.json").read_text())
old_run = next(m for m in __import__("re").findall(r"/runs/validation/(b2-[A-Za-z0-9_.-]+)/", json.dumps(old)))
if out.exists():
    shutil.rmtree(out)
shutil.copytree(src, out, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
bench = subprocess.run(["git", "--git-dir=./.git", "show",
                        "origin/sft-prune:src/arena/throughput_bench.py"], capture_output=True, text=True, check=True).stdout
assert "def steady_window" in bench
(out / "src/arena/throughput_bench.py").write_text(bench)
(out / "configs/bench").mkdir(parents=True, exist_ok=True)
files = []
for i, t in enumerate(TRANSCRIPTS):
    name = f"configs/bench/t{i}.txt"
    shutil.copyfile(CORPUS / t / "duck-transcript.txt", out / name)
    files.append(name)
s = json.dumps(old, indent=2).replace(f"/validation/{old_run}/", f"/validation/{run}/")
assert old_run not in s
d = json.loads(s)
streams = ["rule", 48] if arm == "p75" else ["rule", {"capacity": P75_CAPACITY}, 48]
W_S = 6000
d["trials"] = [{"name": "bench-tpb2", "kind": "serving-bench", "competition": False, "wall_seconds": W_S,
                "server_env": dict(old["trials"][0].get("server_env") or {})}]
d["throughput_bench"] = {"prompt_tokens": [], "concurrency": [], "shared_prefix": None, "output_tokens": 4096,
                         "resident": {"files": files, "prompt_tokens": [16384, 50000, 100000], "output_tokens": 4096,
                                      "decode_passes": 3, "streams": streams}}
d["wall_seconds"] = W_S
d.pop("nebius_image_id", None)  # the baked image was deleted 28 Oct (infra cleanup); v87b boots from the base image
d["chain"] = {"trials": ["bench-tpb2"]}
d["notebook_seconds"] = W_S + 1800
d["roster"] = (f"serving read tpb2 {arm}: 8 transcripts (8 games) resident at 16K/50K/100K, rule streams "
               f"{'+ p75 count ' if arm == 'd4' else ''}+ 48, 3 rounds of 4,096 forced tokens, MTP 3")
(out / "configs/comparison.json").write_text(json.dumps(d, indent=2) + "\n")
ident = json.loads((out / "MANIFEST.json").read_text())
ident["study"] = out.name
ident["files"] = {q.relative_to(out).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(out.rglob("*"))
                  if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(out / "MANIFEST.json").write_text(json.dumps(ident, indent=2) + "\n")
print(out, run, "streams", streams, "mtp", d["server"]["mtp_tokens"], "slots", d.get("generation_slots"),
      "active", d.get("active_limit"), "overlay", d["research_checkpoint"].get("heal_overlay"))
