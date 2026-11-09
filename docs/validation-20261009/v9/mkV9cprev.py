"""9 h Kaggle PUBLIC PREVIEW of V9C on c110 (push + preview only, never submit): the
p50-sub kernel byte for byte (tree, boot, attachments, roster, park 300K) with only what V9C needs: overlay + mask of
c893 dn4g8, FP4 down env, gbps 47.9, 9 h budget (as mkV9prev: 33,600 / 31,800), and log-only additions: the diagnostic
sidecar of mkV9bprev plus a 60 s /metrics sample (ms/step, KV, hit). Usage: mkV9cprev.py"""
import ast
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
P50, C = D / "arena-flash-competition-p50-sub", D / "b2-v9c-p75c893-kf-ho10x11-24h-v1"
NAME = "arena-flash-v9c-prev9h"
OUT = D / NAME
RUNTIME = f"richardcsaky/arc-arena-hero-v2-{NAME.removeprefix('arena-flash-')}-runtime"
OLD_RT = "arc-arena-hero-v2-p50-sub-runtime"
OLD_OV, NEW_OV = "arc-expert-int3-rtn-k256c268-v1", "arc-expert-int3-requant-dn4g8-p75c893-tr15-v1"
OLD_MASK, NEW_MASK = "prune-t50-r15b-nvfp4-v1.json", "prune-t75-nodrop-excl5-v1.json"
NOTEBOOK, PREVIEW = 33600, 31800
FP4 = {"ARC_EXPERT_INT3_FP4_DOWN": "1", "ARC_EXPERT_INT3_FP4_LAYERS": "20,23,28,31,32,33,35,40"}

tree = ast.parse((HERE / "mkV9bprev.py").read_text())
SIDECAR = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SIDECAR")
METRICS = r'''def _dmetrics():
    with _dur.urlopen('http://127.0.0.1:1234/metrics', timeout=5) as r:
        text = r.read().decode(errors='replace')
    keep = {}
    for line in text.splitlines():
        if line.startswith('vllm:') and '_bucket' not in line:
            name, val = line.split('{')[0].split(' ')[0], line.rsplit(' ', 1)[-1]
            keep[name] = keep.get(name, 0.0) + float(val)
    with (_DIAG / 'metrics.jsonl').open('a') as fh:
        fh.write(json.dumps({'s': round(time.time() - NOTEBOOK_START_EPOCH, 1), 'm': keep}) + '\n')
def _dloop():
'''
assert SIDECAR.count("def _dloop():\n") == 1 and SIDECAR.count("            _dkvboot()\n") == 1
SIDECAR = SIDECAR.replace("def _dloop():\n", METRICS).replace(
    "            _dkvboot()\n", "            _dkvboot()\n            _dmetrics()\n")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(P50, OUT, symlinks=True, ignore=shutil.ignore_patterns("dataset-upload", "__pycache__"))
for f in ("experiment.ipynb", "kernel-metadata.json", "build-manifest.json"):
    (OUT / f).unlink(missing_ok=True)
(OUT / "configs/expert-prune" / OLD_MASK).unlink()
shutil.copyfile(C / "configs/expert-prune" / NEW_MASK, OUT / "configs/expert-prune" / NEW_MASK)

s = json.loads((P50 / "configs/comparison.json").read_text())
c = json.loads((C / "configs/comparison.json").read_text())
assert s["focus_oversubscription"]["gate_blocks_per_stream"] == 46.3 and s["cache_experiment"]["cpu_cache_gib"] == 0
assert s["priority"]["park_tokens"] == 300000 and s["priority"]["level_cap_tokens"] == 0 and s["priority"]["equal"] is True
assert len(s["games"]) == 20 and s["competition"]["preview_instances"] == 110
s["focus_oversubscription"]["gate_blocks_per_stream"] = c["focus_oversubscription"]["gate_blocks_per_stream"]
assert s["focus_oversubscription"]["gate_blocks_per_stream"] == 47.9
for k in ("expert_overlay", "expert_prune"):
    s["server"][k] = c["server"][k]
assert s["server"]["expert_overlay"]["dataset"].endswith(NEW_OV) and s["server"]["expert_prune"]["mask"].endswith(NEW_MASK)
assert sha(OUT / s["server"]["expert_prune"]["mask"]) == s["server"]["expert_prune"]["sha256"]
s["model_datasets"] = c["model_datasets"]
env = s["trials"][0]["server_env"]
assert not set(FP4) & set(env) and {k: c["trials"][0]["server_env"][k] for k in FP4} == FP4
env.update(FP4)
assert env == c["trials"][0]["server_env"]
s.pop("dn4g8")
s["fp8_dense"]["note"] = c["fp8_dense"]["note"]
# argv[1]: public preview stop only (5 min so v2 can be submitted); wall and envelope stay v1's
s["competition"]["preview_seconds"] = int(sys.argv[1]) if len(sys.argv) > 1 else PREVIEW
s["wall_seconds"] = PREVIEW
s["notebook_seconds"] = NOTEBOOK
s["evidence"]["v9c_preview"] = ("9 h public preview of V9C (c893 dn4g8 INT3 refit, INT8 dense) on "
                                "c110, Kaggle parts as p50-sub; park 300K as p50-sub; push + preview only")
s["name"] = NAME
text = json.dumps(s, indent=2) + "\n"
assert "keithtyser" not in text
(OUT / "configs/comparison.json").write_text(text)
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
meta_ds = json.loads((P50 / "dataset-upload/dataset-metadata.json").read_text())
meta_ds["id"] = RUNTIME
meta_ds["title"] = RUNTIME.split("/")[1]
meta_ds["description"] = (f"Private benchmark runtime for the arena ({len(s['trials'])} trial, {len(s['games'])} public "
                          f"games, {PREVIEW} s) with pinned third-party dependencies; original licenses retained. "
                          f"{blob_name} sha256 {digest}.")
(upload / "dataset-metadata.json").write_text(json.dumps(meta_ds, indent=2) + "\n")

book = json.loads((P50 / "experiment.ipynb").read_text())
reps = [(OLD_RT, RUNTIME.split("/")[1], 5), (f"{P50.name}.tar.gz.blob", blob_name, 1),
        (json.loads((P50 / "build-manifest.json").read_text())["sha256"], digest, 1), (OLD_OV, NEW_OV, 1),
        ("'--max-seconds', '5700'", f"'--max-seconds', '{NOTEBOOK}'", 1), ("600 s of gameplay", f"{PREVIEW} s of gameplay", 1),
        ("result = subprocess.run(command, check=False)\n", SIDECAR.lstrip("\n"), 1),
        ("            'level_cap': progress.get('level_cap'), 'admission_tick': tick}))",
         "            'level_cap': progress.get('level_cap'), 'admission_tick': tick,\n"
         "            'kv_capacity': (arena.get('window_choice') or {}).get('capacity'),\n"
         "            'streams': ((arena.get('window_choice') or {}).get('focus_fit') or {}).get('density_allowed')}))", 1)]
texts = ["".join(c_["source"]) if isinstance(c_["source"], list) else c_["source"] for c_ in book["cells"]]
for old, new, n in reps:
    got = sum(t.count(old) for t in texts)
    if got != n:
        raise SystemExit(f"notebook: {old[:60]!r} found {got}x, expected {n}")
    texts = [t.replace(old, new) for t in texts]
for bad in ("p50-sub", "k256", "t37-all25", "v87b-sub", "calib"):
    if any(bad in t for t in texts):
        raise SystemExit(f"notebook still names {bad}")
for t in texts[1:]:
    compile(t, "cell", "exec")
for c_, t in zip(book["cells"], texts):
    c_["source"] = t
(OUT / "experiment.ipynb").write_text(json.dumps(book, indent=1))

km = json.loads((P50 / "kernel-metadata.json").read_text())
km["id"], km["title"] = f"richardcsaky/arc-{NAME}", f"ARC {NAME}"
swap = {f"richardcsaky/{OLD_RT}": RUNTIME, f"richardcsaky/{OLD_OV}": f"richardcsaky/{NEW_OV}"}
assert all(km["dataset_sources"].count(k) == 1 for k in swap)
km["dataset_sources"] = [swap.get(d, d) for d in km["dataset_sources"]]
if any("keithtyser" in d for d in km["dataset_sources"] + km["model_sources"]):
    raise SystemExit("public copies attached")
(OUT / "kernel-metadata.json").write_text(json.dumps(km, indent=2) + "\n")
bm = json.loads((P50 / "build-manifest.json").read_text())
bm.update(study=s, blob=blob_name, sha256=digest, runtime_dataset=RUNTIME)
(OUT / "build-manifest.json").write_text(json.dumps(bm, indent=2) + "\n")
chk = subprocess.run([sys.executable, str(HERE / "prevcheck.py"), str(OUT), "1200", "800"], capture_output=True, text=True)
print(chk.stdout.strip())
if chk.returncode:
    raise SystemExit("prevcheck failed: the trial would not start")
print(json.dumps({"out": str(OUT), "kernel": km["id"], "runtime": RUNTIME, "blob_sha256": digest,
                  "blob_mb": round((upload / blob_name).stat().st_size / 1e6, 1), "datasets": km["dataset_sources"],
                  "models": km["model_sources"]}, indent=1))
