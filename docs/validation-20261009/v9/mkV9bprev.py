"""9 h Kaggle PUBLIC PREVIEW of v9b: the v9b sub tree (mkV9b.py sub: v9 harness + kernels serving
p75 ckpt-613 on the AR W4A16 heal-overlay path, BF16 dense, own MTP, wd_nogiveup + v9robust already applied) in
v87b-sub's kernel, plus the AR attachments of the AV2CH kernel (private richardcsaky/arc-w4-weights linked to
/tmp/arc-w4-model; PLE FP8 from the private base copy) and the c613 heal overlay dataset; no expert overlay, no FP8
calibration; the diagnostic sidecar and prevcheck as mkV9prev.py. Push + preview only; never submit. Usage: mkV9bprev.py"""
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
BASE, V9 = D / "arena-flash-competition-v87b-sub", D / "b2-v9b-p75c613-kf-ho10x11-9h-v2-sub"
AV2CH = D / "arena-flash-competition-av2ch-sub"
NAME = "arena-flash-v9b-prev9h"
HEAL_DS = "arc-heal-p75-qad-c613"
OUT = D / NAME
RUNTIME = f"richardcsaky/arc-arena-hero-v2-{NAME.removeprefix('arena-flash-')}-runtime"
OLD_OV, NEW_OV = "arc-expert-int3-requant-dn4g8-t37-all25v71b-v1", "arc-expert-overlay-none-v9b"
OLD_CAL = "arc-fp8-calib-all25-v1"
OLD_RT = "arc-arena-hero-v2-v87b-sub-runtime"
# 31,800 s of play = the Nebius 9 h reads; 32,400 left the bootstrap 31,795 s at trial start (< wall + 300: prev9h v2
# skipped its trial at 485 s). 34,800 (not v9's 33,600): the AR path loads 73 GiB W4 + 28.6 GiB heal overlay + PLE FP8
# from Kaggle inputs with no matching compile cache, so setup up to ~2,580 s still starts the trial (preview session
# limit 12 h; real use = setup + play). prevcheck.py gates every build.
NOTEBOOK, PREVIEW = 34800, 31800

SIDECAR = r'''
# --- diagnostic sidecar (30 Oct): every 300 s to /kaggle/working/diag ---
import threading as _dth, subprocess as _dsp, urllib.request as _dur
_DIAG = Path('/kaggle/working/diag'); _DIAG.mkdir(parents=True, exist_ok=True)
_diag_stop, _diag_pids, _dkv, _dlast = _dth.Event(), {}, {}, [0.0]
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
def _dkvboot():
    if _dkv:
        return
    t = Path('/tmp/arena/trials/duck-transfer')
    try:
        cl, p2 = json.loads((t / 'cache-layout.json').read_text()), json.loads((t / 'phase2-capacity.json').read_text())
    except Exception:
        return
    n = int(cl['num_blocks'])
    _dkv.update(num_blocks=n, gate_capacity=int(n * 0.96), gbps=47.9, density=min(60, int(int(n * 0.96) / 47.9)),
                streams=p2.get('active_limit'), s=round(time.time() - NOTEBOOK_START_EPOCH))
    print(json.dumps({'kv_boot': _dkv}), flush=True)
    (_DIAG / 'kv-boot.json').write_text(json.dumps(_dkv))
def _dloop():
    while not _diag_stop.wait(60):
        try:
            _dkvboot()
            if time.time() - _dlast[0] < 290:
                continue
            _dlast[0] = time.time()
            _dpass()
        except Exception as e:
            (_DIAG / 'sidecar-errors.txt').open('a').write(f'{time.time()} {e!r}\n')
try:
    _dlast[0] = time.time()
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

W4_BLOCK = """w4parts = sorted(Path('/kaggle/input/models/richardcsaky/arc-w4-weights').glob('*/part*/*'), key=lambda p: int(p.parent.name[4:]))
assert len(w4parts) == 2 and all(p.is_dir() for p in w4parts), f'attach the 2 instances of richardcsaky/arc-w4-weights: {w4parts}'
w4 = Path('/tmp/arc-w4-model'); w4.mkdir(exist_ok=False)
for part in w4parts:
    for f in part.iterdir():
        if not f.name.startswith('.'):
            (w4 / f.name).symlink_to(f)
assert (w4 / 'config.json').is_file() and (w4 / 'model.safetensors.index.json').is_file(), 'w4 parts incomplete'
assert Path('/kaggle/input/datasets/richardcsaky/%s/overlay.json').is_file(), 'attach the dataset richardcsaky/%s'
print(json.dumps({'w4_copy': {'files': len(list(w4.iterdir()))}}), flush=True)
""" % (HEAL_DS, HEAL_DS)
CALIB_START = "assert Path('/kaggle/input/datasets/richardcsaky/arc-fp8-calib-all25-v1').is_dir(), 'attach the dataset richardcsaky/arc-fp8-calib-all25-v1'\n"
CALIB_END = "_arc_calib_thread.start()\n"
JOIN_START = "_arc_t_join = time.time()\n"
JOIN_END = "print(json.dumps({'t_mark': 'calib_join', 'wait_s': round(time.time() - _arc_t_join, 1), 's': round(time.time() - NOTEBOOK_START_EPOCH, 1)}), flush=True)\n"
NB_EXTRA = [
    ("print(json.dumps({'private_copy': {'model_files': len(list(model.iterdir())), 'input_root': str(inputs)}}), flush=True)\n",
     "print(json.dumps({'private_copy': {'model_files': len(list(model.iterdir())), 'input_root': str(inputs)}}), flush=True)\n" + W4_BLOCK, 1),
]

PF_START = "def _arc_prefetch():\n    import re as _re\n"
PF_END = "threading.Thread(target=_arc_prefetch, daemon=True, name='arc-prefetch').start()\n"
# v2 (30 Oct, preview v2 load 655 s): v87b's loader-tracking window (v85s design: load order, byte budget under the
# cgroup, DONTNEED behind the loader, runtime blobs first) over the slim AR farm's load list (arena.ar_slim): the
# local base-rest file (not prefetched), the PLE-FP8 shards of /tmp/arc-model the farm indexes, the heal overlay shards
# (~74 GiB). v1/v2 ran AV2CH's plain W4 prefetch: it read 73 GiB of base shards the slim farm no longer loads and left
# the PLE-FP8 and overlay shards cold (18-50 s each at 130-250 MB/s, cgroup 163-171 of 175 GiB).
PF_HEAD_OLD = (
    "    overlay = next((p for p in (Path('/tmp/kaggle-input/datasets/richardcsaky/arc-expert-int3-requant-dn4g8-t37-all25v71b-v1'),) if p.is_dir()), None)\n"
    "    shards = set(json.loads((overlay / 'overlay-manifest.json').read_text())['shards']) if overlay else set()\n"
    "    names = [f.name for f in sorted(model.iterdir()) if f.name.endswith('.safetensors')]\n"
    "    files = [(overlay / n) if n in shards else (model / n).resolve() for n in names]\n"
    "    sizes = [f.stat().st_size for f in files]\n"
    "    wmap = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']\n"
    "    keep = {f for k, f in wmap.items() if any(s in k for s in ('mtp.', 'embed_tokens.', 'lm_head.', 'shared_head.'))}\n")
PF_HEAD_NEW = (
    "    heal = Path('/kaggle/input/datasets/richardcsaky/" + HEAL_DS + "')\n"
    "    hmap = json.loads((heal / 'overlay.json').read_text())['weight_map']\n"
    "    pmap = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']\n"
    "    pfiles = sorted({f for k, f in pmap.items() if k.startswith('model.language_model.layers.1.ple.ple_embedding.ngram_embedding.')})\n"
    "    hfiles = sorted(set(hmap.values()))\n"
    "    # the slim AR farm in vLLM's load order (sorted names): base-rest (local), PLE-FP8, heal overlay\n"
    "    names = ['model-00000-base-rest.safetensors'] + pfiles + ['zz-heal-overlay-' + f for f in hfiles]\n"
    "    files = [None] + [(model / f).resolve() for f in pfiles] + [(heal / f).resolve() for f in hfiles]\n"
    "    sizes = [0] + [f.stat().st_size for f in files[1:]]\n"
    "    keep = {'zz-heal-overlay-' + f for k, f in hmap.items() if any(s in k for s in ('mtp.', 'embed_tokens.', 'lm_head.', 'shared_head.'))}\n")
PF_ITEMS_OLD = "    items = [(i, off, min(chunk, sizes[i] - off)) for i in order for off in range(0, max(sizes[i], 1), chunk)]\n"
PF_ITEMS_NEW = "    items = [(i, off, min(chunk, sizes[i] - off)) for i in order if files[i] is not None for off in range(0, max(sizes[i], 1), chunk)]\n"


def strip_block(texts, start, end):
    out = []
    for t in texts:
        if start in t:
            a = t.index(start); b = t.index(end, a) + len(end)
            t = t[:a] + t[b:]
        out.append(t)
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(V9, OUT, ignore=shutil.ignore_patterns("dataset-upload", "__pycache__"))
# v4: prefix hit (5 min + cumulative) in the live_progress line; the only change from v3
shutil.copyfile(HERE.parents[2] / "src/arena/live_progress.py", OUT / "src/arena/live_progress.py")
_m = json.loads((OUT / "MANIFEST.json").read_text())
_m["files"]["src/arena/live_progress.py"] = sha(OUT / "src/arena/live_progress.py")
(OUT / "MANIFEST.json").write_text(json.dumps(_m, indent=2) + "\n")
for f in ("experiment.ipynb", "kernel-metadata.json", "build-manifest.json"):
    (OUT / f).unlink(missing_ok=True)
assert (OUT / "src/arena/oom_guard.py").is_file() and "restart_backoff" in (OUT / "src/arena/engine_watchdog.py").read_text()
base_s = json.loads((BASE / "configs/comparison.json").read_text())
s = json.loads((V9 / "configs/comparison.json").read_text())
assert len(s["games"]) == 10 and s["competition"]["preview_instances"] == 110  # the 11 x 10 held-out roster
assert {k: v for k, v in s["competition"].items()} == base_s["competition"], "competition block differs from v87b-sub"
assert s["cache_experiment"]["cpu_cache_gib"] == 0 and s["focus_oversubscription"]["gate_blocks_per_stream"] == 47.9
# Kaggle paths of the AR checkpoint (as AV2CH) and of the heal overlay dataset
av = json.loads((AV2CH / "configs/comparison.json").read_text())["research_checkpoint"]
rc = s["research_checkpoint"]
assert rc["repo"] == av["repo"] and rc["revision"] == av["revision"] and rc["ple_fp8"]["files"] == av["ple_fp8"]["files"]
rc.update(path=av["path"], ple_fp8=av["ple_fp8"], skip_dotfiles=True, heal_overlay=f"/kaggle/input/datasets/richardcsaky/{HEAL_DS}")
for k in ("nebius_assets", "nebius_asset_skip", "nebius_boot_disk_gib"):
    s.pop(k, None)
s["name"] = NAME
s["competition"]["preview_seconds"] = PREVIEW
s["wall_seconds"] = PREVIEW
s["notebook_seconds"] = NOTEBOOK
s["evidence"]["preview_9h"] = ("9 h public preview of v9b (p75 ckpt-613 AR INT4 g128 + BF16 dense + own "
                               "MTP on the v9 harness/kernels) on the 11 x 10 held-out roster, engine watchdog without "
                               "give-up, OOM victim order, game re-entry, diagnostic sidecar; push + preview only")
(OUT / "configs/comparison.json").write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((OUT / "MANIFEST.json").read_text())
m["study"] = NAME
for rel in ("configs/comparison.json",):
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
        (json.loads((BASE / "build-manifest.json").read_text())["sha256"], digest, 1),
        ("'--max-seconds', '6900'", f"'--max-seconds', '{NOTEBOOK}'", 1),
        ("1800 s of gameplay", f"{PREVIEW} s of gameplay", 1),
        ("result = subprocess.run(command, check=False)\n", SIDECAR.lstrip("\n"), 1),
        ("            'level_cap': progress.get('level_cap'), 'admission_tick': tick}))",
         "            'level_cap': progress.get('level_cap'), 'admission_tick': tick,\n"
         "            'kv_capacity': (arena.get('window_choice') or {}).get('capacity'),\n"
         "            'streams': ((arena.get('window_choice') or {}).get('focus_fit') or {}).get('density_allowed')}))", 1)] + NB_EXTRA
texts = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in book["cells"]]
assert sum(t.count(CALIB_START) for t in texts) == 1 and sum(t.count(JOIN_START) for t in texts) == 1
texts = strip_block(strip_block(texts, CALIB_START, CALIB_END), JOIN_START, JOIN_END)
REPAIR = ("for _f in sorted(f for f in _ARC_CALIB_SRC.iterdir() if f.is_file()):  # repair anything the thread missed\n"
          "    _arc_calib_copy_one(_f)\n"
          "assert sorted(p.name for p in _ARC_CALIB_DST.iterdir()) == sorted(p.name for p in _ARC_CALIB_SRC.iterdir() if p.is_file())\n")
texts = [x.replace(REPAIR, "") for x in texts]
assert sum(x.count(PF_START) for x in texts) == 1
for old, new in ((PF_HEAD_OLD, PF_HEAD_NEW), (PF_ITEMS_OLD, PF_ITEMS_NEW)):
    assert sum(x.count(old) for x in texts) == 1, old[:80]
    texts = [x.replace(old, new) for x in texts]
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
drop = (f"richardcsaky/{OLD_RT}", f"richardcsaky/{OLD_OV}", f"richardcsaky/{OLD_CAL}")
for old in drop:
    if src.count(old) != 1:
        raise SystemExit(f"kernel-metadata: {old} not attached once")
km["dataset_sources"] = [RUNTIME, f"richardcsaky/{HEAL_DS}"] + [d for d in src if d not in drop]
w4 = [m for m in json.loads((AV2CH / "kernel-metadata.json").read_text())["model_sources"] if "arc-w4-weights" in m]
assert len(w4) == 2, w4
km["model_sources"] = km["model_sources"] + w4
if any("keithtyser" in d for d in km["dataset_sources"] + km["model_sources"]):
    raise SystemExit("public copies attached")
(OUT / "kernel-metadata.json").write_text(json.dumps(km, indent=2) + "\n")
bm = json.loads((BASE / "build-manifest.json").read_text())
bm.update(study=s, blob=blob_name, sha256=digest, runtime_dataset=RUNTIME)
(OUT / "build-manifest.json").write_text(json.dumps(bm, indent=2) + "\n")
# expected setup 600 s: v9-prev9h reached setup-complete at 446 s with a 78 GiB load; the slim AR farm loads ~74 GiB
chk = subprocess.run([sys.executable, str(HERE / "prevcheck.py"), str(OUT), "1200", "600"], capture_output=True, text=True)
print(chk.stdout.strip())
if chk.returncode:
    raise SystemExit("prevcheck failed: the trial would not start")
# loader key-stream dry-run (preview v1 died at weight load): inputs = the AR index and the overlay shard headers
LD = Path("<scratch>/v9b")
chk = subprocess.run([sys.executable, str(HERE / "loaderdry.py"), str(OUT), str(LD / "ar-index.json"), str(LD / "hdr")],
                     capture_output=True, text=True)
print(chk.stdout.strip())
if chk.returncode:
    raise SystemExit("loaderdry failed: the server would not load the weights")
print(json.dumps({"out": str(OUT), "kernel": km["id"], "runtime": RUNTIME, "blob_sha256": digest,
                  "blob_mb": round((upload / blob_name).stat().st_size / 1e6, 1), "datasets": km["dataset_sources"],
                  "models": km["model_sources"]}, indent=1))
