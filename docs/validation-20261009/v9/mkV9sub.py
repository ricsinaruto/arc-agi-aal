"""v9 Kaggle submission kernel: v87b-sub's kernel (notebook, private copies only) carrying v9's
exact sub tree (b2-v9-p75c268-kf-ho10x11-9h-v1-sub, the tree the 9 h read runs) with the competition roster restored and
a 10-min preview (preview/wall 600, notebook 5700, as v87b-sub-prev600). The roster/competition/preview keys are the only
study differences. Push + preview only; never submit. Usage: mkV9sub.py"""
import hashlib
import json
import runpy
import shutil
import tarfile
from pathlib import Path

D = Path("./.claude/worktrees/v84sub/dist")
BASE, V9 = D / "arena-flash-competition-v87b-sub", D / "b2-v9-p75c268-kf-ho10x11-9h-v1-sub"
NAME = "arena-flash-competition-v9-sub"
OUT = D / NAME
RUNTIME = f"richardcsaky/arc-arena-hero-v2-{NAME.removeprefix('arena-flash-competition-')}-runtime"
OLD_OV, NEW_OV = "arc-expert-int3-requant-dn4g8-t37-all25v71b-v1", "arc-expert-int3-requant-dn4g8-p75c268-all25-v1"
OLD_CAL, NEW_CAL = "arc-fp8-calib-all25-v1", "arc-fp8-calib-p75c268-all25-v1"
OLD_RT = "arc-arena-hero-v2-v87b-sub-runtime"
PREVIEW, NOTEBOOK = 600, 5700


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(V9, OUT, ignore=shutil.ignore_patterns("dataset-upload", "__pycache__"))
for f in ("experiment.ipynb", "kernel-metadata.json", "build-manifest.json"):
    (OUT / f).unlink(missing_ok=True)
base_s = json.loads((BASE / "configs/comparison.json").read_text())
s = json.loads((V9 / "configs/comparison.json").read_text())
for k in ("games", "games_excluded", "roster"):
    s[k] = base_s[k]
assert s["competition"] == {**base_s["competition"]}, "v9 sub competition block differs from v87b-sub's"
s["name"] = NAME
s["evidence"]["candidate_v9"] = s.pop("heldout")
s["evidence"]["candidate_v9"]["submission"] = ("Kaggle kernel of b2-v9-p75c268-kf (push + 10-min "
                                               "preview, no submit); competition roster restored")
s["competition"]["preview_seconds"] = PREVIEW
s["wall_seconds"] = PREVIEW
s["notebook_seconds"] = NOTEBOOK
(OUT / "configs/comparison.json").write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((OUT / "MANIFEST.json").read_text())
m["study"] = NAME
m["files"]["configs/comparison.json"] = sha(OUT / "configs/comparison.json")
(OUT / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
runpy.run_path(str(OUT / "bootstrap.py"))["verify"](OUT)

upload = OUT / "dataset-upload"
upload.mkdir()
blob_name = f"{NAME}.tar.gz.blob"
excluded = {"dataset-upload", "experiment.ipynb", "kernel-metadata.json", "build-manifest.json"}
with tarfile.open(upload / blob_name, "w:gz") as archive:
    for path in sorted(OUT.iterdir()):
        if path.name not in excluded:
            archive.add(path, arcname=path.name)
digest = sha(upload / blob_name)
meta_ds = json.loads((BASE / "dataset-upload/dataset-metadata.json").read_text())
meta_ds["id"] = RUNTIME
meta_ds["description"] = (f"Private benchmark runtime for the arena ({len(s['trials'])} independent trials, {len(s['games'])} "
                          f"public games, {s['wall_seconds']} s per trial) with pinned third-party dependencies; original "
                          f"licenses retained. {blob_name} sha256 {digest}.")
(upload / "dataset-metadata.json").write_text(json.dumps(meta_ds, indent=2) + "\n")

book = json.loads((BASE / "experiment.ipynb").read_text())
reps = [(OLD_RT, RUNTIME.split("/")[1], 5), ("arena-flash-competition-v87b-sub.tar.gz.blob", blob_name, 1),
        (json.loads((BASE / "build-manifest.json").read_text())["sha256"], digest, 1), (OLD_OV, NEW_OV, 1),
        (OLD_CAL, NEW_CAL, 4), ("'--max-seconds', '6900'", f"'--max-seconds', '{NOTEBOOK}'", 1),
        ("1800 s of gameplay", f"{PREVIEW} s of gameplay", 1)]
texts = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in book["cells"]]
for old, new, n in reps:
    got = sum(t.count(old) for t in texts)
    if got != n:
        raise SystemExit(f"notebook: {old!r} found {got}x, expected {n}")
    texts = [t.replace(old, new) for t in texts]
for bad in ("v87b-sub", "t37-all25", "draft-head", OLD_CAL):
    if any(bad in t for t in texts):
        raise SystemExit(f"notebook still names {bad}")
for c, t in zip(book["cells"], texts):
    c["source"] = t
(OUT / "experiment.ipynb").write_text(json.dumps(book, indent=1))

km = json.loads((BASE / "kernel-metadata.json").read_text())
km["id"], km["title"] = f"richardcsaky/arc-{NAME}", f"ARC {NAME}"
src = km["dataset_sources"]
for old in (f"richardcsaky/{OLD_RT}", f"richardcsaky/{OLD_OV}", f"richardcsaky/{OLD_CAL}"):
    if src.count(old) != 1:
        raise SystemExit(f"kernel-metadata: {old} not attached once")
swap = {f"richardcsaky/{OLD_RT}": RUNTIME, f"richardcsaky/{OLD_OV}": f"richardcsaky/{NEW_OV}",
        f"richardcsaky/{OLD_CAL}": f"richardcsaky/{NEW_CAL}"}
km["dataset_sources"] = [swap.get(d, d) for d in src]
if any("keithtyser" in d for d in km["dataset_sources"] + km["model_sources"]):
    raise SystemExit("public copies attached")
(OUT / "kernel-metadata.json").write_text(json.dumps(km, indent=2) + "\n")
bm = json.loads((BASE / "build-manifest.json").read_text())
bm.update(study=s, blob=blob_name, sha256=digest, runtime_dataset=RUNTIME)
(OUT / "build-manifest.json").write_text(json.dumps(bm, indent=2) + "\n")
print(json.dumps({"out": str(OUT), "kernel": km["id"], "runtime": RUNTIME, "blob_sha256": digest,
                  "blob_mb": round((upload / blob_name).stat().st_size / 1e6, 1), "datasets": km["dataset_sources"],
                  "models": km["model_sources"]}, indent=1))
