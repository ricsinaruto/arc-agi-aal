"""p50 Kaggle submission kernel (push + 10-min public preview; the maintainer submits, never an agent):
v87b-sub's kernel carrying the tree p50 v3 runs (b2-k256c268-kf-ho10x11-9h-v3: k256 ckpt-268, mask t50-r15b, overlay
arc-expert-int3-rtn-k256c268-v1, gbps 46.3, boot-formula streams), with the Nebius-only hooks out (as v9's sub tree),
the competition roster, preview 600 / notebook 5,700, and the rerun branch unchanged (32,400 s less setup less 600 s).
Beyond v3: wd_nogiveup + v9robust (no play change while nothing fails) and live_progress prefix hit (log only).
Usage: mkP50sub.py"""
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
BASE, P50 = D / "arena-flash-competition-v87b-sub", D / "b2-k256c268-kf-ho10x11-9h-v3"
V9, V9SUB = D / "b2-v9-p75c268-kf-ho10x11-9h-v1", D / "b2-v9-p75c268-kf-ho10x11-9h-v1-sub"
NAME = "arena-flash-competition-p50-sub"
OUT = D / NAME
RUNTIME = f"richardcsaky/arc-arena-hero-v2-{NAME.removeprefix('arena-flash-competition-')}-runtime"
OLD_OV, NEW_OV = "arc-expert-int3-requant-dn4g8-t37-all25v71b-v1", "arc-expert-int3-rtn-k256c268-v1"
OLD_CAL = "arc-fp8-calib-all25-v1"
OLD_RT = "arc-arena-hero-v2-v87b-sub-runtime"
PREVIEW, NOTEBOOK = 600, 5700
CALIB_START = f"assert Path('/kaggle/input/datasets/richardcsaky/{OLD_CAL}').is_dir(), 'attach the dataset richardcsaky/{OLD_CAL}'\n"
CALIB_END = "_arc_calib_thread.start()\n"
JOIN_START = "_arc_t_join = time.time()\n"
JOIN_END = "print(json.dumps({'t_mark': 'calib_join', 'wait_s': round(time.time() - _arc_t_join, 1), 's': round(time.time() - NOTEBOOK_START_EPOCH, 1)}), flush=True)\n"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def strip_block(texts, start, end):
    out = []
    for t in texts:
        if start in t:
            a = t.index(start)
            t = t[:a] + t[t.index(end, a) + len(end):]
        out.append(t)
    return out


if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(P50, OUT, symlinks=True, ignore=shutil.ignore_patterns("dataset-upload", "__pycache__"))
for f in ("experiment.ipynb", "kernel-metadata.json", "build-manifest.json"):
    (OUT / f).unlink(missing_ok=True)
# v9 kf -> sub: the only code difference; p50's copies must still be v9 kf's
for rel in ("bootstrap_arena.py", "src/arena/competition.py"):
    assert sha(P50 / rel) == sha(V9 / rel), rel
    shutil.copyfile(V9SUB / rel, OUT / rel)
(OUT / "src/arena/kaggle_faithful.py").unlink()
assert not (V9SUB / "src/arena/kaggle_faithful.py").exists()
subprocess.run([sys.executable, str(HERE / "wd_nogiveup.py"), str(OUT)], check=True)
subprocess.run([sys.executable, str(HERE / "v9robust.py"), str(OUT)], check=True)
shutil.copyfile(HERE.parents[2] / "src/arena/live_progress.py", OUT / "src/arena/live_progress.py")

base_s = json.loads((BASE / "configs/comparison.json").read_text())
s = json.loads((P50 / "configs/comparison.json").read_text())
assert s["focus_oversubscription"]["gate_blocks_per_stream"] == 46.3 and s["cache_experiment"]["cpu_cache_gib"] == 0
assert s["priority"]["level_cap_tokens"] == 0 and s["priority"]["park_tokens"] == 300000 and s["priority"]["equal"] is True
for k in ("b200", "kaggle_faithful"):
    s.pop(k)
for k in ("games", "games_excluded", "roster"):
    s[k] = base_s[k]
s["competition"] = dict(base_s["competition"])
s["competition"]["preview_seconds"] = PREVIEW
s["wall_seconds"] = PREVIEW
s["notebook_seconds"] = NOTEBOOK
s["evidence"]["candidate_p50"] = s.pop("heldout")
s["evidence"]["candidate_p50"]["submission"] = ("Kaggle kernel of p50 v3 (push + 10-min preview; the "
                                                "owner submits); competition roster restored")
s["name"] = NAME
text = json.dumps(s, indent=2) + "\n"
(OUT / "configs/comparison.json").write_text(text)
hooks = [k for k in s if "nebius" in k or k in ("research_clones",)] + [k for k in s["competition"] if k in ("rehearsal", "research_clones", "preview_live_score")]
assert not hooks, hooks
m = json.loads((OUT / "MANIFEST.json").read_text().replace(P50.name, NAME))
m["study"] = NAME
m["files"] = {q.relative_to(OUT).as_posix(): sha(q) for q in sorted(OUT.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
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
meta_ds["title"] = RUNTIME.split("/")[1]
meta_ds["description"] = (f"Private benchmark runtime for the arena ({len(s['trials'])} trial, {len(s['games'])} public "
                          f"games, {PREVIEW} s) with pinned third-party dependencies; original licenses retained. "
                          f"{blob_name} sha256 {digest}.")
(upload / "dataset-metadata.json").write_text(json.dumps(meta_ds, indent=2) + "\n")

book = json.loads((BASE / "experiment.ipynb").read_text())
reps = [(OLD_RT, RUNTIME.split("/")[1], 5), ("arena-flash-competition-v87b-sub.tar.gz.blob", blob_name, 1),
        (json.loads((BASE / "build-manifest.json").read_text())["sha256"], digest, 1), (OLD_OV, NEW_OV, 1),
        ("'--max-seconds', '6900'", f"'--max-seconds', '{NOTEBOOK}'", 1), ("1800 s of gameplay", f"{PREVIEW} s of gameplay", 1)]
texts = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in book["cells"]]
assert sum(t.count(CALIB_START) for t in texts) == 1 and sum(t.count(JOIN_START) for t in texts) == 1
texts = strip_block(strip_block(texts, CALIB_START, CALIB_END), JOIN_START, JOIN_END)
for old, new, n in reps:
    got = sum(t.count(old) for t in texts)
    if got != n:
        raise SystemExit(f"notebook: {old[:60]!r} found {got}x, expected {n}")
    texts = [t.replace(old, new) for t in texts]
for bad in ("v87b-sub", "t37-all25", "draft-head", OLD_CAL, "calib", "requant", "dn4g8"):
    if any(bad in t for t in texts):
        raise SystemExit(f"notebook still names {bad}")
for t in texts[1:]:
    compile(t, "cell", "exec")
for c, t in zip(book["cells"], texts):
    c["source"] = t
(OUT / "experiment.ipynb").write_text(json.dumps(book, indent=1))

km = json.loads((BASE / "kernel-metadata.json").read_text())
km["id"], km["title"] = f"richardcsaky/arc-{NAME}", f"ARC {NAME}"
src = km["dataset_sources"]
for old in (f"richardcsaky/{OLD_RT}", f"richardcsaky/{OLD_OV}", f"richardcsaky/{OLD_CAL}"):
    if src.count(old) != 1:
        raise SystemExit(f"kernel-metadata: {old} not attached once")
swap = {f"richardcsaky/{OLD_RT}": RUNTIME, f"richardcsaky/{OLD_OV}": f"richardcsaky/{NEW_OV}"}
km["dataset_sources"] = [swap.get(d, d) for d in src if d != f"richardcsaky/{OLD_CAL}"]
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
