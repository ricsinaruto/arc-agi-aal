"""p50d7 held-out 10 x 5 read: b2-p50d7c294-kf-ho10x11-9h-v2's tree with only weights and roster
changed: 10 clones of the 5 excl5 games, their prior clones' drain jitter (k005 k006 ... k095 ... k099 of the prior read).
Usage: mkP50d7x5.py TAG DATASET MANIFEST_SHA [VERSION]"""
import hashlib
import json
import runpy
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parents[2] / "dist"
V2 = D / "b2-p50d7c294-kf-ho10x11-9h-v2"
tag, dataset, msha = sys.argv[1:4]
RUN = f"b2-p50d7{tag}-ex5x10-v{sys.argv[4] if len(sys.argv) > 4 else 1}"
OUT = D / RUN
GAMES = ["ft09-0d8bbf25", "vc33-5430563c", "lp85-305b61c3", "tu93-0768757b", "ar25-0c556536"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(V2, OUT, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
p = OUT / "src/arena/duck.py"
t = p.read_text()
a = '            value = trigger_draw(game, index, jitter["low"], jitter["high"], jitter.get("seed", 0))\n'
b = ('            # 10 x 5 read: the prior read\'s clone of this game (k005 + game + 10 x clone), same draws.\n'
     '            prior = f"k{5 + int(game[1:]) % 5 + 10 * (int(game[1:]) // 5):03d}"\n'
     '            value = trigger_draw(prior, index, jitter["low"], jitter["high"], jitter.get("seed", 0))\n')
assert t.count(a) == 1
p.write_text(t.replace(a, b))

# MANIFEST_SHA "late": boot while SFT's export runs (nebius_asset_late); the runner fetches the
# overlay when its asset manifest lands and checks every file against it, so the bundle reads the overlay manifest
# unpinned (logged), as arc2 12f1cd77a
LATE = msha == "late"
if LATE:
    ov = OUT / "src/arena/expert_overlay.py"
    t = ov.read_text()
    a = '''    digest = spec["manifest_sha256"]\n    if not isinstance'''
    b = '''    digest = spec["manifest_sha256"]\n    if digest == "late":  # nebius_asset_late: checked file by file against the export's asset manifest\n        return dict(spec)\n    if not isinstance'''
    c = '''    if hashlib.sha256(raw).hexdigest() != expected_sha256:'''
    d = '''    if expected_sha256 == "late":  # nebius_asset_late: the runner checked every file against the asset manifest\n        print(json.dumps({"event": "expert_overlay_late", "manifest_sha256": hashlib.sha256(raw).hexdigest()}), flush=True)\n    elif hashlib.sha256(raw).hexdigest() != expected_sha256:'''
    assert t.count(a) == 1 and t.count(c) == 1
    ov.write_text(t.replace(a, b).replace(c, d))

s = json.loads((V2 / "configs/comparison.json").read_text().replace(V2.name, RUN))
assert s["games"][5:] == GAMES and s["competition"]["preview_instances"] == 110
s["games_excluded"] = s["games_excluded"] + s["games"][:5]
s["games"] = GAMES
s["roster"] = "10 clones each of the 5 excl5 games (ft09 vc33 lp85 tu93 ar25), order as the prior read"
s["competition"]["preview_instances"] = 50
s["server"]["expert_overlay"].update(dataset=dataset, manifest_sha256=msha)
s["model_datasets"] = [dataset]
s["heldout"].update(overlay=dataset, games=[g.split("-")[0] for g in GAMES])
s["evidence"]["p50d7x5"] = f"10 x 5 read of {tag}, pairs with {V2.name}'s first 10 clones of these games; each clone to 300K or a win (external stop loop)"
s["name"] = RUN
if LATE:  # 42 GiB measured on the jc1059 overlay; wait 1 h for the manifest after boot
    s["nebius_asset_late"] = {f"flash-input/datasets/{dataset}": {"wait_seconds": 3600, "gib": 44}}
(OUT / "configs/comparison.json").write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((V2 / "MANIFEST.json").read_text().replace(V2.name, RUN))
m["files"] = {q.relative_to(OUT).as_posix(): sha(q) for q in sorted(OUT.rglob("*"))
              if q.is_file() and "__pycache__" not in q.parts and q.name not in ("MANIFEST.json", "build-manifest.json")}
(OUT / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
runpy.run_path(str(OUT / "bootstrap.py"))["verify"](OUT)
print(OUT, len(s["games"]), dataset, s["focus_oversubscription"]["gate_blocks_per_stream"])
