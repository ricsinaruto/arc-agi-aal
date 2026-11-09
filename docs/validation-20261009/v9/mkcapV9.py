"""v9 native capture bundle: A's capture bundle cap-v85p-p75c176-v1 (p75 t75
nodrop mask, relaxed expert_overlay.py, MoE + dense capture) with the all-25 inputs of cap-v85p-all25-dense-v1
(inputs-all25-v1, the v8.7b fit's inputs) and overlay = NVFP4 RTN of ckpt-268's experts + ckpt-268 dense/MTP shards.
Usage: mkcapV9.py DATASET MANIFEST_SHA"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

W = Path("./.claude/worktrees")
SRC = W / "sft-prune/dist/cap-v85p-p75c176-v1"
ALL25 = W / "dfr-exact/dist/cap-v85p-all25-dense-v1"
OUT = W / "sft-prune/dist/cap-v85p-p75c268-all25-v1"
RUN_OLD, RUN_NEW = "b2-sft-cap-v85p-p75c176-v1", "b2-tp-cap-v85p-p75c268-all25-v1"
dataset, msha = sys.argv[1], sys.argv[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(SRC, OUT, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
for f in ("calib.json", "eval.json", "selection.json"):
    shutil.copyfile(ALL25 / "configs/expert-quant" / f, OUT / "configs/expert-quant" / f)
p = OUT / "configs/comparison.json"
s = json.loads(p.read_text())
a = json.loads((ALL25 / "configs/comparison.json").read_text())
s["expert_quant"]["inputs"] = a["expert_quant"]["inputs"]
for f, h in s["expert_quant"]["inputs"]["sha256"].items():
    assert sha(OUT / "configs/expert-quant" / f) == h, f
assert s["dense_capture"] == a["dense_capture"], "dense capture block differs from the all-25 dense capture"
s["name"] = OUT.name
s["server"]["expert_overlay"] = {"dataset": dataset, "manifest_sha256": msha, "check": "sha256"}
s["model_datasets"] = [dataset]
text = json.dumps(s, indent=2) + "\n"
text = text.replace(RUN_OLD, RUN_NEW).replace(SRC.name, OUT.name)
assert "p75c176" not in text, [l for l in text.splitlines() if "p75c176" in l][:5]
p.write_text(text)
man = json.loads((OUT / "MANIFEST.json").read_text())
man["files"] = {str(f.relative_to(OUT)): sha(f) for f in sorted(OUT.rglob("*"))
                if f.is_file() and f.name != "MANIFEST.json" and f.suffix != ".ipynb"
                and f.name not in ("kernel-metadata.json", "build-manifest.json") and "dataset-upload" not in f.parts}
(OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
print(OUT, s["server"]["expert_prune"], s["server"]["expert_overlay"], s["expert_quant"]["inputs"]["source"])
