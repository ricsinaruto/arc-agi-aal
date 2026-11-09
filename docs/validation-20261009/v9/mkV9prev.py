"""9 h Kaggle PUBLIC PREVIEW of v9-sub: the v9 v1 tree (b2-v9-p75c268-kf-ho10x11-9h-v1-sub: v1
config, no tier, the 11 x 10 held-out roster of the Nebius 9 h reads) in v87b-sub's kernel, plus
- the engine watchdog without give-up (wd_nogiveup.py, v84-sub 3c48401c3);
- v9robust.py: A. OOM victim order (sandboxes +1000, vLLM -1000, trial -900); B. game re-entry
  after a turn-loop error (60 s backoff, waits for a healthy engine, at most 20 per game);
- a 9 h preview: notebook envelope 33,600 s, preview/wall 31,800 (as the Nebius 9 h reads; prevcheck.py gates it);
- a diagnostic sidecar in the notebook (thread, every 300 s, to /kaggle/working/diag/): cgroup memory.current /
  memory.stat / memory.events, /proc/meminfo, per-process RSS (ps, sandboxes included), nvidia-smi, server /health,
  process starts/exits (pid set diff; bootstrap and watchdog restart exit codes from the logs), dmesg tail if readable,
  tails of the trial process log and the vLLM server log, live progress; end.json with the bootstrap exit and the
  trial's end reason.
Push + preview only; never submit. Usage: mkV9prev.py"""
import hashlib
import json
import runpy
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = Path("./.claude/worktrees/v84sub/dist")
BASE, V9 = D / "arena-flash-competition-v87b-sub", D / "b2-v9-p75c268-kf-ho10x11-9h-v1-sub"
NAME = "arena-flash-v9-prev9h"
OUT = D / NAME
RUNTIME = f"richardcsaky/arc-arena-hero-v2-{NAME.removeprefix('arena-flash-')}-runtime"
OLD_OV, NEW_OV = "arc-expert-int3-requant-dn4g8-t37-all25v71b-v1", "arc-expert-int3-requant-dn4g8-p75c268-all25-v1"
OLD_CAL, NEW_CAL = "arc-fp8-calib-all25-v1", "arc-fp8-calib-p75c268-all25-v1"
OLD_RT = "arc-arena-hero-v2-v87b-sub-runtime"
# 33,600 / 31,800 = the Nebius 9 h reads' envelope; 32,400 left the bootstrap 31,795 s at trial start (< wall + 300:
# prev9h v2 skipped its trial at 485 s). prevcheck.py gates every build.
NOTEBOOK, PREVIEW = 33600, 31800

SIDECAR = r'''
# --- diagnostic sidecar (30 Oct): every 300 s to /kaggle/working/diag ---
import threading as _dth, subprocess as _dsp, urllib.request as _dur
_DIAG = Path('/kaggle/working/diag'); _DIAG.mkdir(parents=True, exist_ok=True)
_diag_stop, _diag_pids = _dth.Event(), {}
def _dread(p, n=None):
    try:
        t = Path(p).read_text(errors='replace')
        return t if n is None else t[-n:]
    except Exception as e:
        return f'<{type(e).__name__}>'
def _dcmd(*a, n=20000):
    try:
        return _dsp.run(list(a), capture_output=True, text=True, timeout=60).stdout[-n:]
    except Exception as e:
        return f'<{type(e).__name__}>'
def _dcg():
    for root in ('/sys/fs/cgroup', '/sys/fs/cgroup/memory'):
        if Path(root, 'memory.current').exists() or Path(root, 'memory.usage_in_bytes').exists():
            return {n: _dread(Path(root, n)) for n in ('memory.current', 'memory.max', 'memory.events', 'memory.stat',
                    'memory.usage_in_bytes', 'memory.limit_in_bytes', 'memory.max_usage_in_bytes')
                    if Path(root, n).exists()}
    return {}
def _dhealth():
    try:
        with _dur.urlopen('http://127.0.0.1:1234/health', timeout=5) as r:
            return r.status
    except Exception as e:
        return type(e).__name__
def _dpass(final=False):
    t = time.time()
    ps = _dcmd('ps', '-eo', 'pid,ppid,rss,etimes,stat,comm,args', '--sort=-rss', n=10**6).splitlines()
    pids = {}
    for line in ps[1:]:
        f = line.split(None, 6)
        if len(f) >= 6:
            pids[f[0]] = ' '.join(f[5:])[:200]
    gone = {p: c for p, c in _diag_pids.items() if p not in pids}
    new = {p: c for p, c in pids.items() if p not in _diag_pids}
    _diag_pids.clear(); _diag_pids.update(pids)
    rec = {'t': round(t, 1), 's': round(t - NOTEBOOK_START_EPOCH, 1), 'final': final, 'health': _dhealth(),
           'cgroup': _dcg(), 'meminfo': _dread('/proc/meminfo'), 'ps_top': ps[:60], 'n_procs': len(pids),
           'rss_total_kib': sum(int(l.split()[2]) for l in ps[1:] if len(l.split()) > 2 and l.split()[2].isdigit()),
           'proc_new': new if len(new) < 200 else {'count': len(new)}, 'proc_gone': gone if len(gone) < 200 else {'count': len(gone)},
           'gpu': _dcmd('nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,clocks_throttle_reasons.active', '--format=csv,noheader'),
           'xid': _dcmd('nvidia-smi', '-q', '-d', 'PAGE_RETIREMENT,ROW_REMAPPER', n=3000),
           'dmesg': _dcmd('dmesg', '-T', n=8000), 'df': _dcmd('df', '-h', '/tmp', '/kaggle/working', '/dev/shm')}
    with (_DIAG / 'trace.jsonl').open('a') as fh:
        fh.write(json.dumps(rec) + '\n')
    for src, name, n in (('/tmp/arena/trials/duck-transfer-process.log', 'trial-process.tail', 400000),
                         ('/tmp/arena/serving/vllm-openai-server.log', 'server.tail', 400000),
                         ('/tmp/arena/trials/duck-transfer/progress.json', 'progress.json', None),
                         ('/tmp/arena/live/progress.json', 'live-progress.json', None),
                         ('/tmp/arena/trials/duck-transfer/oom-guard.jsonl', 'oom-guard.jsonl', None)):
        try:
            (_DIAG / name).write_text(_dread(src, n))
        except Exception:
            pass
def _dloop():
    while not _diag_stop.wait(300):
        try:
            _dpass()
        except Exception as e:
            (_DIAG / 'sidecar-errors.txt').open('a').write(f'{time.time()} {e!r}\n')
try:
    _dpass()
except Exception as _e:
    (_DIAG / 'sidecar-errors.txt').open('a').write(f'start {_e!r}\n')
_dth.Thread(target=_dloop, daemon=True, name='arc-diag').start()
result = subprocess.run(command, check=False)
_diag_stop.set()
try:
    _dpass(final=True)
    _end = {'bootstrap_exit': result.returncode, 'elapsed': round(time.time() - NOTEBOOK_START_EPOCH, 1)}
    for _p in (Path('/kaggle/working/arena/trials/duck-transfer/arena.json'), Path('/tmp/arena/trials/duck-transfer/arena.json')):
        if _p.is_file():
            _a = json.loads(_p.read_text())
            _end.update(status=_a.get('status'), end_reason=_a.get('end_reason'), wall_seconds=_a.get('wall_seconds'),
                        score=_a.get('mean_score'), levels=_a.get('levels'), failed_sessions=_a.get('failed_sessions'),
                        game_status={s: sum(1 for g in (_a.get('games') or []) if g.get('status') == s)
                                     for s in {g.get('status') for g in (_a.get('games') or [])}},
                        game_errors=[(g.get('game_id'), (g.get('error') or '')[:200]) for g in (_a.get('games') or []) if g.get('error')][:40])
            break
    _re = {}
    for _g in Path('/tmp/arena/trials/duck-transfer/games').glob('*/log.jsonl'):
        try:
            _n = sum(1 for _l in _g.open(errors='replace') if '"kind":"game_reentry"' in _l)
        except Exception:
            _n = -1
        if _n:
            _re[_g.parent.name] = _n
    _end['game_reentries'] = _re
    (_DIAG / 'end.json').write_text(json.dumps(_end, indent=1))
except Exception as _e:
    (_DIAG / 'sidecar-errors.txt').open('a').write(f'end {_e!r}\n')
'''


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(V9, OUT, ignore=shutil.ignore_patterns("dataset-upload", "__pycache__"))
for f in ("experiment.ipynb", "kernel-metadata.json", "build-manifest.json"):
    (OUT / f).unlink(missing_ok=True)
subprocess.run([sys.executable, str(HERE / "wd_nogiveup.py"), str(OUT)], check=True)
subprocess.run([sys.executable, str(HERE / "v9robust.py"), str(OUT)], check=True)
base_s = json.loads((BASE / "configs/comparison.json").read_text())
s = json.loads((V9 / "configs/comparison.json").read_text())
assert len(s["games"]) == 10 and s["competition"]["preview_instances"] == 110  # the 11 x 10 held-out roster
assert {k: v for k, v in s["competition"].items()} == base_s["competition"], "competition block differs from v87b-sub"
assert s["cache_experiment"]["cpu_cache_gib"] == 0 and s["focus_oversubscription"]["gate_blocks_per_stream"] == 47.9
s["name"] = NAME
s["competition"]["preview_seconds"] = PREVIEW
s["wall_seconds"] = PREVIEW
s["notebook_seconds"] = NOTEBOOK
s["evidence"]["preview_9h"] = ("/ ~09:35 (v2): 9 h public preview of v9-sub (v1 config) on the 11 x 10 "
                               "held-out roster, engine watchdog without give-up, OOM victim order, game re-entry, "
                               "diagnostic sidecar; push + preview only")
(OUT / "configs/comparison.json").write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((OUT / "MANIFEST.json").read_text())
m["study"] = NAME
for rel in ("configs/comparison.json", "src/arena/engine_watchdog.py", "src/arena/trial.py", "src/arena/oom_guard.py",
            "src/duck_transfer/agent.py"):
    m["files"][rel] = sha(OUT / rel)
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
        (OLD_CAL, NEW_CAL, 4), ("'--max-seconds', '6900'", f"'--max-seconds', '{NOTEBOOK}'", 1),
        ("1800 s of gameplay", f"{PREVIEW} s of gameplay", 1),
        ("result = subprocess.run(command, check=False)\n", SIDECAR.lstrip("\n"), 1)]
texts = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in book["cells"]]
for old, new, n in reps:
    got = sum(t.count(old) for t in texts)
    if got != n:
        raise SystemExit(f"notebook: {old[:60]!r} found {got}x, expected {n}")
    texts = [t.replace(old, new) for t in texts]
for bad in ("v87b-sub", "t37-all25", "draft-head", OLD_CAL):
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
swap = {f"richardcsaky/{OLD_RT}": RUNTIME, f"richardcsaky/{OLD_OV}": f"richardcsaky/{NEW_OV}",
        f"richardcsaky/{OLD_CAL}": f"richardcsaky/{NEW_CAL}"}
for old in swap:
    if src.count(old) != 1:
        raise SystemExit(f"kernel-metadata: {old} not attached once")
km["dataset_sources"] = [swap.get(d, d) for d in src]
if any("keithtyser" in d for d in km["dataset_sources"] + km["model_sources"]):
    raise SystemExit("public copies attached")
(OUT / "kernel-metadata.json").write_text(json.dumps(km, indent=2) + "\n")
bm = json.loads((BASE / "build-manifest.json").read_text())
bm.update(study=s, blob=blob_name, sha256=digest, runtime_dataset=RUNTIME)
(OUT / "build-manifest.json").write_text(json.dumps(bm, indent=2) + "\n")
chk = subprocess.run([sys.executable, str(HERE / "prevcheck.py"), str(OUT)], capture_output=True, text=True)
print(chk.stdout.strip())
if chk.returncode:
    raise SystemExit("prevcheck failed: the trial would not start")
print(json.dumps({"out": str(OUT), "kernel": km["id"], "runtime": RUNTIME, "blob_sha256": digest,
                  "blob_mb": round((upload / blob_name).stat().st_size / 1e6, 1), "datasets": km["dataset_sources"],
                  "models": km["model_sources"]}, indent=1))
