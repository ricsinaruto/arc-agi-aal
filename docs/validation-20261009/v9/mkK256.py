"""p50 (k256) 9 h held-out read: the v9 v1 kf tree (b2-v9-p75c268-kf-ho10x11-9h-v1: same harness,
roster, seeds, envelope, KV gate, gbps 47.9, 60-stream target, no CPU tier) serving the k256 student's last checkpoint
(sft-k256-qad-v1b ckpt-268) in the format its QAD trained:
- mask prune-t50-r15b-nvfp4-v1 (256 of 512 experts per layer; fit on the train-15 r15b usage rows);
- experts INT3 g16 RTN+search from ckpt-268's BF16 experts (no refit, no dn4g8: every layer and projection INT3);
- dense INT8 W8A16 (per-row RTN at load) on lm_head, GDN in_proj_qkvz / out_proj, QSA qkv / o_proj, MTP q/k/v/o
  (dense_kernels.int8_weights), no FP8 W8A8 and no FP8 calibration;
- int4 KV, indexer FP8 off, its own MTP head (dense shards), draft experts NvFp4 at load and draft vocab arena16k (as v9).
Streams come from the density rule on the real boot: min(60, floor(gate capacity / 47.9)).
Usage: mkK256.py OVERLAY_DATASET OVERLAY_MANIFEST_SHA MASK_JSON"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

D = Path("./.claude/worktrees/v84sub/dist")
SRC = D / "b2-v9-p75c268-kf-ho10x11-9h-v1"
RUN = "b2-k256c268-kf-ho10x11-9h-v1"
if len(sys.argv) > 1 and sys.argv[1] in ("--v2", "--v3"):
    # v2: v1's bundle at gate_blocks_per_stream 42.4 (v9 v3's density:
    # floor(0.96 x 2,245 / 42.4) = 50 streams), everything else identical
    # v3: gbps 46.3 = 46 streams on the same pool, 48.8 raw blocks/stream (v2's 44.9 is the old
    # 24 h's cliff density)
    import json as _j
    V2, GBPS, WHY = (("b2-k256c268-kf-ho10x11-9h-v2", 42.4, "v1 at gbps 42.4 (KV 0.70 at 43.5 running)")
                     if sys.argv[1] == "--v2" else
                     ("b2-k256c268-kf-ho10x11-9h-v3", 46.3, "v1 at gbps 46.3: 46 streams"))
    src, dst = D / RUN, D / V2
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "dataset-upload"))
    p = dst / "configs/comparison.json"
    text = p.read_text().replace(RUN, V2)
    s = _j.loads(text)
    assert s["focus_oversubscription"]["gate_blocks_per_stream"] == 47.9
    s["focus_oversubscription"]["gate_blocks_per_stream"] = GBPS
    s.setdefault("evidence", {})["k256_" + V2[-2:]] = WHY
    p.write_text(_j.dumps(s, indent=2) + "\n")
    m = _j.loads((dst / "MANIFEST.json").read_text())
    m["study"] = V2
    m["files"] = {q.relative_to(dst).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(dst.rglob("*")) if q.is_file()
                  and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name not in ("MANIFEST.json",)}
    (dst / "MANIFEST.json").write_text(_j.dumps(m, indent=2) + "\n")
    print(dst)
    sys.exit(0)
OUT = D / RUN
overlay, overlay_sha, mask = sys.argv[1], sys.argv[2], Path(sys.argv[3])
OLD_OV, OLD_CAL = "richardcsaky/arc-expert-int3-requant-dn4g8-p75c268-all25-v1", "richardcsaky/arc-fp8-calib-p75c268-all25-v1"
INT8 = ["lm_head", "gdn_in_proj_qkvz", "qsa_qkv", "attn_out"]
DROP_ENV = ("ARC_FP8_CALIB", "ARC_FP8_BF16", "ARC_EXPERT_INT3_FP4_DOWN", "ARC_EXPERT_INT3_FP4_LAYERS")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(SRC, OUT, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "dataset-upload"))
p = OUT / "configs/comparison.json"
text = p.read_text()
assert text.count(f"/runs/validation/{SRC.name}/") == text.count("/runs/validation/b2-")  # only own paths
text = text.replace(f"/runs/validation/{SRC.name}/", f"/runs/validation/{RUN}/")
s = json.loads(text)
srv = s["server"]
mask_rel = f"configs/expert-prune/{mask.name}"
(OUT / "configs/expert-prune/prune-t75-nodrop-excl5-v1.json").unlink()
shutil.copyfile(mask, OUT / mask_rel)
assert srv["expert_prune"]["mask"] == "configs/expert-prune/prune-t75-nodrop-excl5-v1.json"
srv["expert_prune"] = {**srv["expert_prune"], "mask": mask_rel, "sha256": sha(mask)}
assert srv["expert_overlay"]["dataset"] == OLD_OV
srv["expert_overlay"] = {**srv["expert_overlay"], "dataset": f"richardcsaky/{overlay}", "manifest_sha256": overlay_sha}
dk = srv["dense_kernels"]
assert dk["fp8_w8a8"] == ["attn_out", "gdn_in_proj_qkvz", "qsa_qkv"] and dk["int8_weights"] == []
dk["fp8_w8a8"], dk["int8_weights"] = [], INT8
assert s["model_datasets"] == [OLD_OV, OLD_CAL], s["model_datasets"]
s["model_datasets"] = [f"richardcsaky/{overlay}"]
for t in s["trials"]:
    for k in DROP_ENV:
        t["server_env"].pop(k)
s["fp8_dense"] = {"families": [], "toggle": None, "calib": None,
                  "note": "k256: no FP8 W8A8 (QAD trained INT8 W8A16 dense; dense_kernels.int8_weights)"}
s["dn4g8"] = {"note": "k256: off (QAD trained every expert projection at INT3)"}
s["name"] = RUN
s["heldout"] = {**s["heldout"], "mask": mask.name, "overlay": f"richardcsaky/{overlay}",
                "overlay_manifest_sha256": overlay_sha,
                "model": {**s["heldout"]["model"], "student": "/runs/validation/sft-k256-qad-v1b/train/ckpt-268/student",
                          "stack": "k256 (p50) student sft-k256-qad-v1b ckpt-268 on the v9 v1 tree: mask "
                                   "prune-t50-r15b-nvfp4-v1, experts INT3 g16 RTN+search (no refit, no dn4g8), dense "
                                   "INT8 W8A16, no FP8 W8A8 / calibration, kv4, indexer FP8 off, own MTP"},
                "mask_screen": "prune-t50-r15b-nvfp4-v1: train-15 r15b usage rows only"}
s.setdefault("evidence", {})["k256"] = ("9 h 11 x 10 held-out read of the k256 student's last "
                                        "checkpoint, the v9 v1 tree with the served format its QAD trained")
p.write_text(json.dumps(s, indent=2) + "\n")
assert not any(x in p.read_text() for x in ("arc-fp8-calib-p75c268", "requant-dn4g8-p75c268"))
m = json.loads((OUT / "MANIFEST.json").read_text())
m["study"] = RUN
m["files"] = {q.relative_to(OUT).as_posix(): sha(q) for q in sorted(OUT.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(OUT / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
print(OUT)
