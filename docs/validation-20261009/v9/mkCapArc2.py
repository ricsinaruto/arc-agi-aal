"""Native c240 capture for the plain INT3 refit of SFT's ARC-2 p75 finetune: the c893 capture bundle
(t75 nodrop mask, MoE + dense capture) with the ARC-2 c240 inputs of arc2-quant-capture-c240-k256-v1 (365 attempt
prefixes, 9.5M tokens, all 120 tasks, so the 20 held-out tasks are in the fit, as ordered), overlay = NVFP4 RTN of the
checkpoint's kept experts + its dense/MTP. Usage: mkCapArc2.py DATASET MANIFEST_SHA [RUN]"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

W = Path("./.claude/worktrees")
SRC, INP = W / "sft-prune/dist/cap-v85p-p75c893-tr15-v1", W / "arc2/dist/arc2-quant-capture-c240-k256-v1"
OLD = "b2-tp-cap-v85p-p75c893-tr15-v1"
dataset, msha = sys.argv[1], sys.argv[2]
RUN = sys.argv[3] if len(sys.argv) > 3 else "b2-tp-cap-p75arc2-c240-v1"
OUT = W / "v84sub/dist" / RUN


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(SRC, OUT, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
a = json.loads((INP / "configs/comparison.json").read_text())["expert_quant"]["inputs"]
for f, h in a["sha256"].items():
    assert sha(INP / "configs/expert-quant" / f) == h, f
    shutil.copyfile(INP / "configs/expert-quant" / f, OUT / "configs/expert-quant" / f)
p = OUT / "configs/comparison.json"
s = json.loads(p.read_text())
keep = s["expert_quant"]["inputs"]["folder"]
s["expert_quant"]["inputs"] = {**a, "folder": keep}
s["name"] = OUT.name
s["server"]["expert_overlay"] = {"dataset": dataset, "manifest_sha256": msha, "check": "sha256"}
s["model_datasets"] = [dataset]
s["notebook_seconds"] = 7200  # 3x the c893 capture's tokens; deadline = notebook - 1,500 s reserve
text = json.dumps(s, indent=2).replace(OLD, RUN).replace(SRC.name, OUT.name) + "\n"
p.write_text(text)
m = json.loads((OUT / "MANIFEST.json").read_text())
skip = ("kernel-metadata.json", "build-manifest.json")
m["files"] = {q.relative_to(OUT).as_posix(): sha(q) for q in sorted(OUT.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"
              and q.name not in skip and "dataset-upload" not in q.parts}
(OUT / "MANIFEST.json").write_text(json.dumps(m, indent=1) + "\n")
print(OUT, s["server"]["expert_overlay"], s["expert_quant"]["inputs"]["source"][:60])
