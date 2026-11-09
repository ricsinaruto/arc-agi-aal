"""p50d7 ckpt-942 Kaggle submission kernel (push + 10-min preview; the maintainer submits, never an agent):
arena-flash-competition-p50-sub with exactly the weights changed to the p50d7 ckpt-942 student the 10 x 5 excl5 read
b2-p50d7tc942-ex5x10-v1 served (overlay arc-expert-int3-rtn-p50d7tc942-v1, d7 mask, BF16 dense, no INT8 at load), with
the serving files that read ran for the 41-layer cut (expert_overlay, serving, expert_prune_patch/runtime, PLE gate fix).
Streams: p50-sub's boot formula, min(focus_slots 60, floor(.96 x booted blocks / 46.3)), unchanged (p50d7 c294 v2:
KV .80 steady at that density). Everything else byte-identical to p50-sub.
Usage: mkP50d7sub.py"""
import hashlib
import json
import runpy
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parents[2] / "dist"
BASE, READ = D / "arena-flash-competition-p50-sub", D / "b2-p50d7tc942-ex5x10-v1"
NAME = "arena-flash-competition-p50d7-sub"
OUT = D / NAME
RUNTIME = f"richardcsaky/arc-arena-hero-v2-{NAME.removeprefix('arena-flash-competition-')}-runtime"
OLD_RT = "arc-arena-hero-v2-p50-sub-runtime"
OLD_OV, NEW_OV = "arc-expert-int3-rtn-k256c268-v1", "arc-expert-int3-rtn-p50d7tc942-v1"
OLD_MASK, NEW_MASK = "configs/expert-prune/prune-t50-r15b-nvfp4-v1.json", "configs/expert-prune/prune-t50-r15b-d7-v1.json"
# the weights change: the files the p50d7 read ran for the 41-layer cut (p50 v3 -> p50d7 v2 diff, nothing else)
CODE = ("src/arena/expert_overlay.py", "src/arena/serving.py", "src/arc_infra/expert_prune_patch.py",
        "src/arc_infra/expert_prune_runtime.py", "src/arc_infra/serving_kernels_patch.py")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(BASE, OUT, symlinks=True, ignore=shutil.ignore_patterns("dataset-upload", "__pycache__"))
P50V3, D7V2 = D / "b2-k256c268-kf-ho10x11-9h-v3", D / "b2-p50d7c294-kf-ho10x11-9h-v2"
for rel in CODE:
    assert sha(BASE / rel) == sha(P50V3 / rel), f"p50-sub changed {rel}: not a pure weights swap"
    assert sha(READ / rel) == sha(D7V2 / rel), rel
    shutil.copyfile(READ / rel, OUT / rel)
(OUT / OLD_MASK).unlink()
shutil.copyfile(READ / NEW_MASK, OUT / NEW_MASK)

s = json.loads((BASE / "configs/comparison.json").read_text())
r = json.loads((READ / "configs/comparison.json").read_text())
assert s["server"]["expert_overlay"]["dataset"] == f"richardcsaky/{OLD_OV}" and s["server"]["expert_prune"]["mask"] == OLD_MASK
s["server"]["expert_overlay"] = r["server"]["expert_overlay"]
s["server"]["expert_prune"] = r["server"]["expert_prune"]
s["server"]["dense_kernels"]["int8_weights"] = r["server"]["dense_kernels"]["int8_weights"]
s["fp8_dense"] = r["fp8_dense"]
s["model_datasets"] = r["model_datasets"]
assert s["server"]["expert_prune"]["sha256"] == sha(OUT / NEW_MASK)
assert s["focus_oversubscription"] == r["focus_oversubscription"] and s["server"]["max_num_seqs"] == r["server"]["max_num_seqs"]
s["evidence"]["candidate_p50d7"] = (f"p50-sub with the p50d7 ckpt-942 weights served by {READ.name} "
                                    "(overlay, d7 mask, BF16 dense) and p50-sub's boot stream formula; push + 10-min preview; "
                                    "the maintainer submits")
s["name"] = NAME
(OUT / "configs/comparison.json").write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((OUT / "MANIFEST.json").read_text().replace(BASE.name, NAME))
m["study"] = NAME
m["files"] = {q.relative_to(OUT).as_posix(): sha(q) for q in sorted(OUT.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts
              and q.name not in ("MANIFEST.json", "kernel-metadata.json", "build-manifest.json")}
(OUT / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
runpy.run_path(str(OUT / "bootstrap.py"))["verify"](OUT)

upload = OUT / "dataset-upload"
upload.mkdir()
old_blob, blob_name = f"{BASE.name}.tar.gz.blob", f"{NAME}.tar.gz.blob"
excluded = {"dataset-upload", "experiment.ipynb", "kernel-metadata.json", "build-manifest.json"}
with tarfile.open(upload / blob_name, "w:gz") as archive:
    for path in sorted(OUT.iterdir()):
        if path.name not in excluded:
            archive.add(path, arcname=path.name)
digest = sha(upload / blob_name)
meta_ds = json.loads((BASE / "dataset-upload/dataset-metadata.json").read_text())
meta_ds["id"] = RUNTIME
meta_ds["title"] = RUNTIME.split("/")[1]
meta_ds["description"] = meta_ds["description"].replace(old_blob, blob_name).split(" sha256 ")[0] + f" sha256 {digest}."
(upload / "dataset-metadata.json").write_text(json.dumps(meta_ds, indent=2) + "\n")

old_digest = json.loads((BASE / "build-manifest.json").read_text())["sha256"]
book = json.loads((BASE / "experiment.ipynb").read_text())
reps = [(OLD_RT, RUNTIME.split("/")[1], None), (old_blob, blob_name, 1), (old_digest, digest, 1), (OLD_OV, NEW_OV, 1)]
texts = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in book["cells"]]
for old, new, n in reps:
    got = sum(t.count(old) for t in texts)
    if got == 0 or (n is not None and got != n):
        raise SystemExit(f"notebook: {old[:60]!r} found {got}x, expected {n}")
    texts = [t.replace(old, new) for t in texts]
for bad in ("k256c268", "p50-sub"):
    if any(bad in t for t in texts):
        raise SystemExit(f"notebook still names {bad}")
for t in texts[1:]:
    compile(t, "cell", "exec")
for c, t in zip(book["cells"], texts):
    c["source"] = t
(OUT / "experiment.ipynb").write_text(json.dumps(book, indent=1))

km = json.loads((BASE / "kernel-metadata.json").read_text())
km["id"], km["title"] = f"richardcsaky/arc-{NAME}", f"ARC {NAME}"
swap = {f"richardcsaky/{OLD_RT}": RUNTIME, f"richardcsaky/{OLD_OV}": f"richardcsaky/{NEW_OV}"}
for old in swap:
    if km["dataset_sources"].count(old) != 1:
        raise SystemExit(f"kernel-metadata: {old} not attached once")
km["dataset_sources"] = [swap.get(d, d) for d in km["dataset_sources"]]
if any("keithtyser" in d for d in km["dataset_sources"] + km["model_sources"]):
    raise SystemExit("public copies attached")
(OUT / "kernel-metadata.json").write_text(json.dumps(km, indent=2) + "\n")
bm = json.loads((BASE / "build-manifest.json").read_text())
bm.update(study=s, blob=blob_name, sha256=digest, runtime_dataset=RUNTIME)
(OUT / "build-manifest.json").write_text(json.dumps(bm, indent=2) + "\n")
chk = subprocess.run([sys.executable, str(HERE / "prevcheck.py"), str(OUT), "1200", "800"], capture_output=True, text=True)
print(chk.stdout.strip())
if chk.returncode:
    raise SystemExit("prevcheck failed: the trial would not start")
print(json.dumps({"out": str(OUT), "kernel": km["id"], "runtime": RUNTIME, "blob_sha256": digest,
                  "blob_mb": round((upload / blob_name).stat().st_size / 1e6, 1), "datasets": km["dataset_sources"],
                  "models": km["model_sources"]}, indent=1))
