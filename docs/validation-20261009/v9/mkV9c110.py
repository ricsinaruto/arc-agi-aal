"""9 h Nebius c110 pair: the submitted tree (p50-sub src: wd_nogiveup, v9robust, oom_guard) with the Nebius kf
hooks of p50 v3, p50 v3's study (envelope, park 300K per game, no level budget) on the c110 roster (110 cyclic clones of the
20 public games), same seeds. v9c: c893 dn4g8 overlay + t75 mask + FP4 down + gbps 47.9 as kernel v2.
c647 DATASET MANIFEST_SHA: p50 v3's recipe, only the overlay changes. c268: p50 v3's own weights.
d7: the p50d7-sub kernel tree (p50d7 ckpt-942) the same way as c268.
Usage: mkV9c110.py v9c | c268 | d7 | c647 DATASET SHA"""
import hashlib
import json
import runpy
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parents[2] / "dist"
SUB, V3, KV2 = D / "arena-flash-competition-p50-sub", D / "b2-k256c268-kf-ho10x11-9h-v3", D / "arena-flash-v9c-prev9h"
side = sys.argv[1]
if side == "d7":  # the p50d7-sub kernel tree (ckpt-942 weights) on c110, converted as c268 was
    SUB = D / "arena-flash-competition-p50d7-sub"
RUN = {"v9c": "b2-v9c-p75c893-kf-c110-9h-v1", "c647": "b2-k256c647-kf-c110-9h-v1", "c268": "b2-k256c268-kf-c110-9h-v1",
       "d7": "b2-p50d7tc942-kf-c110-9h-v1"}[side]
OUT = D / RUN


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(SUB, OUT, symlinks=True, ignore=shutil.ignore_patterns("dataset-upload", "__pycache__"))
for f in ("experiment.ipynb", "kernel-metadata.json"):
    (OUT / f).unlink()
for rel in ("bootstrap_arena.py", "src/arena/competition.py", "src/arena/kaggle_faithful.py", "build-manifest.json"):
    shutil.copyfile(V3 / rel, OUT / rel)

sub = json.loads((SUB / "configs/comparison.json").read_text())
s = json.loads((V3 / "configs/comparison.json").read_text().replace(V3.name, RUN))
assert s["priority"]["park_tokens"] == 300000 and s["priority"]["level_cap_tokens"] == 0 and "level_end_tokens" not in s
assert s["competition"]["preview_seconds"] == 31800 and s["competition"]["preview_instances"] == 110
assert s["focus_oversubscription"]["gate_blocks_per_stream"] == 46.3 and s["cache_experiment"]["cpu_cache_gib"] == 0
for k in ("games", "games_excluded", "roster"):
    s[k] = sub[k]
assert len(s["games"]) == 20
s["evidence"]["candidate_p50"] = s.pop("heldout")
if side == "v9c":
    k = json.loads((KV2 / "configs/comparison.json").read_text())
    for key in ("expert_overlay", "expert_prune"):
        s["server"][key] = k["server"][key]
    s["model_datasets"] = k["model_datasets"]
    s["trials"][0]["server_env"].update({x: k["trials"][0]["server_env"][x] for x in
                                         ("ARC_EXPERT_INT3_FP4_DOWN", "ARC_EXPERT_INT3_FP4_LAYERS")})
    assert s["trials"][0]["server_env"] == k["trials"][0]["server_env"]
    s["focus_oversubscription"]["gate_blocks_per_stream"] = 47.9
    s.pop("dn4g8")
    s["fp8_dense"]["note"] = k["fp8_dense"]["note"]
    old = SUB / "configs/expert-prune/prune-t50-r15b-nvfp4-v1.json"
    (OUT / "configs/expert-prune" / old.name).unlink()
    shutil.copyfile(KV2 / s["server"]["expert_prune"]["mask"], OUT / s["server"]["expert_prune"]["mask"])
elif side == "d7":  # the kernel tree's weights fields (its mask and serving files came with the copy)
    for key in ("expert_overlay", "expert_prune"):
        s["server"][key] = sub["server"][key]
    s["server"]["dense_kernels"]["int8_weights"] = sub["server"]["dense_kernels"]["int8_weights"]
    s["fp8_dense"], s["model_datasets"] = sub["fp8_dense"], sub["model_datasets"]
    s["evidence"]["candidate_p50d7"] = sub["evidence"]["candidate_p50d7"]
elif side == "c647":
    dataset, msha = sys.argv[2], sys.argv[3]
    s["server"]["expert_overlay"].update(dataset=dataset, manifest_sha256=msha)
    s["model_datasets"] = [dataset]
assert sha(OUT / s["server"]["expert_prune"]["mask"]) == s["server"]["expert_prune"]["sha256"]
s["evidence"]["c110_pair"] = "9 h c110 pair V9C vs p50 ckpt-647, gap at 300K; Nebius only"
s["name"] = RUN
(OUT / "configs/comparison.json").write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((V3 / "MANIFEST.json").read_text().replace(V3.name, RUN))
m["files"] = {q.relative_to(OUT).as_posix(): sha(q) for q in sorted(OUT.rglob("*"))
              if q.is_file() and "__pycache__" not in q.parts and q.name not in ("MANIFEST.json", "build-manifest.json")}
(OUT / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
runpy.run_path(str(OUT / "bootstrap.py"))["verify"](OUT)
print(OUT, len(s["games"]), s["server"]["expert_overlay"]["dataset"], s["focus_oversubscription"]["gate_blocks_per_stream"])
