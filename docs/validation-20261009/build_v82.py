"""v8.2: the v8 source and host side with play at the v7.1p state, plus the items of ``V82``.

Base = build_v8.py with --weights=excl5, --off=border-moves,sweep,floor (v7.1p play: no sweep, no border_moves, no reset
offer, priority floor as v7.1p, half-drain, no stagger, no fraction compaction, reply cap 16,384) and its host items on:
state co-tenant, reply ids, keep-alive, the tuned draft head (``--draft-head-overlay=DIR``, required), ``transient``.
Then, on the built study:

prompt fixes    harness_fixes + solvable_line, death_turn_v2, findings_keep, tool_call_followup, stuck_streak;
                hx_harness reflect_skip_death 1 (docs/harness-arm.md 5.7)
no world model  dfr_loop no_world_model
playbook        hx_harness memo_style 2: one model-written playbook, no length limit, in place of note() and the notes
                block, with the memory framing in the system prompt
animation       dfr_harness animation_history: the frames of every animated action stay readable
transient_note  off (``transient`` alone)

argv: output --draft-head-overlay=DIR [--window=130816] [--streams=48] [--gate-margin=0.96] [--density-blocks=2084]
[--wall=N] [--dependencies] and build_v8.py's other options. The drain drops half the context budget at every window. The three long runs (streams from the throughput arm):
--window=65536 --streams=76 | --window=130816 --streams=48 | --window=196608 --streams=34.
[--smoke-trigger=N] (smoke runs: drain at N prompt tokens instead of the window's budget).
[--level-cap=N] [--level-cap-grant=N] (priority.level_cap_tokens, the per-level token cap; off without it).
[--weights=all25 --kaggle] (a submission: the all-25 fit; the Kaggle files are written again from the final study).
[--weights=all25 --all25-test] (a test run of the submission's weights, on explicit approval only;
[--overlay=owner/dataset:sha256] with it: another INT3 overlay under the all-25 mask).
[--v84] (v8.2 with 8-bit FP8 KV and BF16 indexer keys, sabl-K's serving change; the long runs'
equal scheduler with parking at 300,000 generated tokens (docs/validation-20261009/v84_equal.py), no level cap, no tail
fade; streams from the FP8 pool: --streams=27 --density-blocks=2051).
[--runtime-dataset=owner/slug] (with --kaggle: another runtime dataset for the same kernel, so an earlier kernel
version keeps its archive). With a public preview (no --no-preview) the save run prints a result summary.
A margin above 0.98 is written to the built study (``kv_margin``): hero-v2 accepts 0.5..0.98.
"""
import hashlib
import json
import runpy
import sys
from pathlib import Path

ROOT = Path.cwd()
OWN = {"streams": "48", "window": "130816", "gate-margin": "0.96", "density-blocks": "2084", "smoke-trigger": "",
       "level-cap": "", "level-cap-grant": "", "runtime-dataset": "", "av2c": "", "av2c-harness": "",
       "safety": "", "cand": "", "tool-repair": "", "overlay-compose": "", "save-compile-cache": "", "stage-local": "",
       "compile-cache": "", "draft-files": "", "prefetch-uncapped": "", "prefetch-window": "",
       "prefetch-order": "", "prefetch-readers": "", "l1": "", "l1-cpu-gib": "64", "memtrace": "", "l1-shm-fix": "", "dn4g8": "", "graph-sizes": "",
       "profile-at": "", "keep-tokens": ""}
PROFILE_PATCHES = [
    ("src/arena/trial.py", "                stop = self.early_stop.check(wall.elapsed())\n",
     "                _profile_tick(self, wall.elapsed())\n"
     "                stop = self.early_stop.check(wall.elapsed())\n"),
    ("bootstrap.py", '        for name in ("cache-probe",):', '        for name in ("cache-probe", "torch-profile"):'),
]
PROFILE_FIRE = '''

def _profile_tick(trial, elapsed):
    """--profile-at (build_v82): POST /start_profile then /stop_profile on a thread at each due time; never fatal."""
    pending = getattr(trial, "_profile_pending", None)
    if pending is None:
        pending = trial._profile_pending = sorted(float(t) for t in trial.study.get("profile_at") or [])
    while pending and elapsed >= pending[0]:
        at = pending.pop(0)
        url = trial.base_url.rstrip("/").removesuffix("/v1")

        def run(at=at, url=url):
            import urllib.request
            for path in ("/start_profile", "/stop_profile"):
                try:
                    urllib.request.urlopen(urllib.request.Request(url + path, data=b"", method="POST"), timeout=600).read()
                    trial.gate_log({"event": "profile", "path": path, "at": at})
                except Exception as error:
                    trial.gate_log({"event": "profile_error", "path": path, "error": repr(error)[:300]})
                time.sleep(30)

        import threading
        threading.Thread(target=run, daemon=True).start()
'''
ALL25_MASK = "data/expert_prune/prune-t37-int3-requant-all25v71b-v1.json"
# --cand (v8.6): files the candidate's dfr-exact run bundle changes on v8.5p's tree
# 20 Oct (start-up work): a public preview saves the compile / Triton / CuTeDSL caches its start built.
CALIB_LOCAL = "/tmp/arc-fp8-calib"  # local copy of the FP8 calibration dataset (v85f2, 22 Oct)
SAVE_COMPILE_CACHE = """    try:  # preview only: save the compile / Triton / CuTeDSL caches this start built (start-up work, 20 Oct)
        import tarfile as _tarfile
        _out, _sizes = Path('/kaggle/working/compile-cache.tar.gz'), {}
        with _tarfile.open(_out, 'w:gz') as _tar:
            for _root in ('/tmp/qwen38-flash-next-vllm-cache', '/tmp/qwen38-flash-next-vllm-compile-cache'):
                for _f in sorted(Path(_root).rglob('*')) if Path(_root).is_dir() else []:
                    if _f.is_file() and not _f.is_symlink() and _f.stat().st_size < 2**30:
                        _tar.add(_f, arcname=str(_f.relative_to('/tmp')))
                        _top = '/'.join(str(_f.relative_to('/tmp')).split('/')[:3])
                        _sizes[_top] = _sizes.get(_top, 0) + _f.stat().st_size
        print('COMPILE CACHE ' + json.dumps({'bytes': _out.stat().st_size, 'parts': _sizes}))
    except Exception as error:
        print('COMPILE CACHE unavailable: ' + repr(error))
"""
# 20 Oct (Kaggle start-up, first play < 600 s): --stage-local copies the served files from the GCS FUSE inputs to
# local /tmp with 16 threads from the notebook's first second (runtime layers first, then the weights in vLLM's load
# order), dropping each FUSE file's page cache after its copy; each link (runtime dir, /tmp/arc-model, the composed
# overlay) is swapped atomically to the local copy once that copy is whole. The bootstrap starts when the runtime layers
# are local; serving.py waits for the weights marker before the overlay farm is built (ARC_STAGE_WAIT), so vLLM reads
# only local files (lazy mmap, no eager pass). Bytes are identical: output-exact. --compile-cache=DATASET restores the
# compile / Triton / CuTeDSL caches a preview saved (a cache miss only recompiles).
STAGE = """import re as _re, queue as _queue
def _arc_nat(name):
    return [int(t) if t.isdigit() else t for t in _re.split(r'(\\d+)', name)]
_stage_dir = Path('/tmp/arc-stage'); _stage_dir.mkdir(exist_ok=False); (_stage_dir / '.swaps').mkdir()
_stage_marker = _stage_dir / '.weights-done'
os.environ['ARC_STAGE_WAIT'] = str(_stage_marker)
_stage = {'t0': time.time(), 'runtime_done': threading.Event(), 'files': 0, 'bytes': 0, 'errors': [], 'runtime_seconds': None}
_rt_link = inputs / 'datasets' / 'keithtyser' / 'qwen38-flash-next-vllm-nvfp4-runtime-v1'
_rt_src = _rt_link.resolve(); _rt_link.unlink(); _rt_link.mkdir()
_rt_jobs = []
for _f in sorted(_rt_src.iterdir()):
    (_rt_link / _f.name).symlink_to(_f)
    if _f.is_file() and _f.stat().st_size > 2**20:
        _rt_jobs.append((_f, _rt_link / _f.name))
_ov_dir = next((p for p in (Path('/tmp/kaggle-input/datasets/richardcsaky/OVERLAY_SLUG'),) if p.is_dir()), None)
_ov_shards = set(json.loads((_ov_dir / 'overlay-manifest.json').read_text())['shards']) if _ov_dir else set()
_w_jobs = []
for _f in sorted(model.iterdir(), key=lambda p: (not p.name.startswith('model-plefp8'), _arc_nat(p.name))):
    if _f.name.endswith('.safetensors'):
        _link = (_ov_dir / _f.name) if _f.name in _ov_shards else _f
        _w_jobs.append((_link.resolve(), _link))
_stage['total'] = sum(s.stat().st_size for s, _ in _rt_jobs + _w_jobs)
_stage['free'] = os.statvfs('/tmp').f_bavail * os.statvfs('/tmp').f_frsize
def _arc_copy(src, link):
    dst = _stage_dir / link.name
    part = _stage_dir / (link.name + '.part')
    buf = bytearray(16 << 20); view = memoryview(buf)
    with open(src, 'rb', buffering=0) as r, open(part, 'wb', buffering=0) as w:
        while (n := r.readinto(buf)):
            done = 0
            while done < n:
                done += w.write(view[done:n])
        try:
            os.posix_fadvise(r.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
    if part.stat().st_size != src.stat().st_size:
        raise RuntimeError(f'{link.name}: staged size differs')
    os.replace(part, dst)
    swap = _stage_dir / '.swaps' / link.name
    swap.symlink_to(dst)
    os.replace(swap, link)
    return dst.stat().st_size
def _arc_stage():
    if _stage['free'] < _stage['total'] + 20 * 2**30:
        _stage['errors'].append('not enough local disk'); _stage['runtime_done'].set()
        _stage_marker.write_text(json.dumps({'skipped': True})); return
    jobs, lock, left = _queue.Queue(), threading.Lock(), [len(_rt_jobs)]
    for job in _rt_jobs + _w_jobs:
        jobs.put(job)
    def work():
        while True:
            try:
                src, link = jobs.get_nowait()
            except _queue.Empty:
                return
            try:
                size = _arc_copy(src, link)
                with lock:
                    _stage['files'] += 1; _stage['bytes'] += size
            except Exception as error:
                with lock:
                    _stage['errors'].append(f'{link.name}: {error!r}')
            if any(link == l for _, l in _rt_jobs):
                with lock:
                    left[0] -= 1
                    if left[0] == 0:
                        _stage['runtime_seconds'] = round(time.time() - _stage['t0'], 1); _stage['runtime_done'].set()
    threads = [threading.Thread(target=work, daemon=True) for _ in range(16)]
    for t in threads: t.start()
    if not _rt_jobs:
        _stage['runtime_done'].set()
    while any(t.is_alive() for t in threads):
        threads[0].join(30) if threads[0].is_alive() else time.sleep(30)
        s = time.time() - _stage['t0']
        print(json.dumps({'stage_progress': {'files': _stage['files'], 'gib': round(_stage['bytes'] / 2**30, 1), 'of_gib': round(_stage['total'] / 2**30, 1), 'seconds': round(s), 'mib_s': round(_stage['bytes'] / 2**20 / max(s, 1)), 'cgroup': _arc_cgroup()}}), flush=True)
    _stage['runtime_done'].set()
    summary = {'files': _stage['files'], 'of': len(_rt_jobs) + len(_w_jobs), 'gib': round(_stage['bytes'] / 2**30, 1), 'seconds': round(time.time() - _stage['t0'], 1), 'runtime_seconds': _stage['runtime_seconds'], 'errors': _stage['errors'][:5], 'overlay': str(_ov_dir)}
    _stage_marker.write_text(json.dumps(summary))
    print(json.dumps({'stage': summary}), flush=True)
threading.Thread(target=_arc_stage, daemon=True, name='arc-stage').start()
"""
STAGE_GATE = """_stage['runtime_done'].wait(300)
print(json.dumps({'stage_runtime_gate': {'seconds': round(time.time() - NOTEBOOK_START_EPOCH, 1), 'runtime_seconds': _stage['runtime_seconds']}}), flush=True)
"""
RESTORE_CACHE = """def _arc_restore_cache():
    found = next((p for p in (Path('/kaggle/input/datasets/richardcsaky/CACHE_SLUG'), Path('/kaggle/input/CACHE_SLUG')) if p.is_dir()), None)
    if found is None:
        return {'error': 'dataset not attached'}
    started = time.time()
    try:
        archive = next(found.glob('*.tar.gz.blob'))
        with tarfile.open(archive, 'r:gz') as tar:
            names = [m for m in tar.getmembers() if m.name.startswith(('qwen38-flash-next-vllm-cache/', 'qwen38-flash-next-vllm-compile-cache/'))]
            tar.extractall('/tmp', members=names, filter='data')
        return {'files': len(names), 'seconds': round(time.time() - started, 1), 'archive': archive.name}
    except Exception as error:
        return {'error': repr(error)}
print(json.dumps({'compile_cache_restore': _arc_restore_cache()}), flush=True)
"""
# 20 Oct (Kaggle start-up, l45 v1: the MTP draft's eager pass re-read all 206 shards, ~4 min over GCS FUSE): with
# ARC_DRAFT_FILES=1 (set by the notebook) the draft's load reads only the shards whose index entries its
# load_weights maps (_remap_mtp_weight_name); it drops every other name, so the loaded tensors are the same.
DRAFT_FILES_EDITS = """    'edits': [('        loaded_weights = model.load_weights(self.get_all_weights(model_config, model))\\n',"""
DRAFT_FILES_EXTRA = """              ('        yield from self._get_weights_iterator(primary_weights)\\n',
               '        # ARC (Kaggle start-up): with ARC_DRAFT_FILES=1 the MTP draft reads only the shards it maps.\\n'
               '        self._arc_draft_files = (os.environ.get("ARC_DRAFT_FILES") == "1"\\n'
               '                                 and type(model).__name__ == "Qwen3_8FlashNextMTP")\\n'
               '        yield from self._get_weights_iterator(primary_weights)\\n'
               '        self._arc_draft_files = False\\n', 1),
              ('            source.allow_patterns_overrides,\\n        )\\n        if self.load_config.load_format == "npcache":\\n',
               '            source.allow_patterns_overrides,\\n        )\\n'
               '        if getattr(self, "_arc_draft_files", False) and use_safetensors:\\n'
               '            import json as _arc_json\\n'
               '            from vllm.models.qwen3_8_flash_next.nvidia.mtp import _remap_mtp_weight_name\\n'
               '            with open(os.path.join(hf_folder, SAFE_WEIGHTS_INDEX_NAME)) as _arc_f:\\n'
               '                _arc_map = _arc_json.load(_arc_f)["weight_map"]\\n'
               '            _arc_keep = {f for n, f in _arc_map.items() if _remap_mtp_weight_name(n) is not None}\\n'
               '            _arc_all = len(hf_weights_files)\\n'
               '            hf_weights_files = [f for f in hf_weights_files if os.path.basename(f) in _arc_keep]\\n'
               '            logger.info("ARC_DRAFT_FILES %d of %d shards for the MTP draft", len(hf_weights_files), _arc_all)\\n'
               '        try:\\n'
               '            from vllm.model_executor.model_loader.weight_utils import _natural_sort_key as _arc_nsk\\n'
               '        except ImportError:\\n'
               '            _arc_nsk = None\\n'
               '        logger.info("ARC_LOAD_ORDER %s", [os.path.basename(f) for f in sorted(hf_weights_files, key=_arc_nsk)[:5]])\\n'
               '        if self.load_config.load_format == "npcache":\\n', 1),
"""
# 20 Oct (Kaggle start-up, l45 v4: an uncapped prefetch filled the 175 GiB cgroup and the loader read at eviction
# speed; v5: a cgroup-reading gate lagged 16 readers): prefetch in vLLM's load order inside a byte-accounted window
# (started minus dropped <= min(PRE GiB, cgroup high/max - 65 GiB)), drop each file's page cache once the loader has
# passed it (its progress bar in the server log), except the MTP draft's shards; the runtime datasets' cache is
# dropped once the server log appears (extract done). POST is unused. v8: the 10 PLE shards (the cold tail) are read first
# in 256 MiB ranges by all 16 readers, after RUNTIME_WARM has read the runtime layer blobs the same way.
RUNTIME_WARM = """import threading as _th, os as _os
_arc_rt_warm, _arc_rt_warm_done = {}, _th.Event()
def _arc_runtime_warm():
    t0 = time.time()
    root = next((p for p in (Path('/kaggle/input/datasets/richardcsaky/arc-base-runtime-v1'), Path('/kaggle/input/arc-base-runtime-v1')) if p.is_dir()), None)
    blobs = sorted(root.glob('layer-*.tar.gz.blob'), key=lambda p: int(p.name.split('-')[1])) if root else []
    items = [(b, off) for b in blobs for off in range(0, b.stat().st_size, 256 << 20)]
    lock, pos = _th.Lock(), [0]
    def work():
        buffer = bytearray(16 << 20)
        while True:
            with lock:
                if pos[0] >= len(items):
                    return
                b, off = items[pos[0]]; pos[0] += 1
            with open(b, 'rb', buffering=0) as stream:
                stream.seek(off); left = 256 << 20
                while left > 0:
                    got = stream.readinto(memoryview(buffer)[:min(left, len(buffer))])
                    if not got:
                        break
                    left -= got
    threads = [_th.Thread(target=work, daemon=True) for _ in range(16)]
    for t in threads: t.start()
    for t in threads: t.join()
    _arc_rt_warm.update(blobs=len(blobs), gib=round(sum(b.stat().st_size for b in blobs) / 2**30, 1), seconds=round(time.time() - t0, 1), done_s=round(time.time() - NOTEBOOK_START_EPOCH, 1))
    _arc_rt_warm_done.set()
_th.Thread(target=_arc_runtime_warm, daemon=True, name='arc-runtime-warm').start()
"""
PREFETCH_WINDOW = """def _arc_prefetch():
    import re as _re
    started = time.time()
    overlay = next((p for p in (Path('/tmp/kaggle-input/datasets/richardcsaky/OVERLAY_SLUG'),) if p.is_dir()), None)
    shards = set(json.loads((overlay / 'overlay-manifest.json').read_text())['shards']) if overlay else set()
    names = [f.name for f in sorted(model.iterdir()) if f.name.endswith('.safetensors')]
    files = [(overlay / n) if n in shards else (model / n).resolve() for n in names]
    sizes = [f.stat().st_size for f in files]
    wmap = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    keep = {f for k, f in wmap.items() if any(s in k for s in ('mtp.', 'embed_tokens.', 'lm_head.', 'shared_head.'))}
    def _cg(name):
        try:
            v = Path('/sys/fs/cgroup/' + name).read_text().strip()
            return None if v == 'max' else int(v)
        except Exception:
            return None
    high, hard = _cg('memory.high'), _cg('memory.max')
    cap = high or hard or 175 * 2**30
    # Byte-accounted window (v6): bytes started minus bytes dropped, never above B = cap - 53 GiB anon - 12 GiB margin
    budget = min(PRE_GIB * 2**30, cap - 65 * 2**30)
    print(json.dumps({'prefetch_window': {'memory_high_gib': high and round(high / 2**30, 1), 'memory_max_gib': hard and round(hard / 2**30, 1), 'budget_gib': round(budget / 2**30, 1), 'first': names[:5]}}), flush=True)
    ple = [i for i, n in enumerate(names) if n.startswith('model-plefp8')]
    order = ple + [i for i in range(len(names)) if i not in set(ple)]  # v8: the cold PLE tail first, then load order
    chunk = 256 << 20
    items = [(i, off, min(chunk, sizes[i] - off)) for i in order for off in range(0, max(sizes[i], 1), chunk)]
    log, lock, st = Path('/tmp/arena/serving/vllm-openai-server.log'), threading.Lock(), {'next': 0, 'pos': 0, 'loading': False, 'dropped': 0, 'drop_gib': 0.0, 'waits': 0, 'window': 0, 'skipped': 0, 'rt_gib': 0.0, 'files': 0, 'marks': {}}
    begun = [False] * len(files)
    def drop(path):
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            return True
        except (AttributeError, OSError):
            return False
    def work(k):
        buffer = bytearray(16 << 20)
        while True:
            with lock:
                if st['next'] >= len(items):
                    return
                i, off, ln = items[st['next']]
                if off == 0 and i < st['pos']:
                    st['skipped'] += 1; st['next'] += 1; begun[i] = None; continue
                if off == 0 and not begun[i]:
                    if st['window'] + sizes[i] > budget:
                        st['waits'] += 1; i = None
                    else:
                        begun[i] = True; st['window'] += sizes[i]; st['files'] += 1; st['next'] += 1
                else:
                    st['next'] += 1
                    if not begun[i]:
                        continue
            if i is None:
                time.sleep(0.5); continue
            with open(files[i], 'rb', buffering=0) as stream:
                stream.seek(off); left = ln
                while left > 0:
                    got = stream.readinto(memoryview(buffer)[:min(left, len(buffer))])
                    if not got:
                        break
                    left -= got
    def drop_runtime():
        for root in ('/tmp/kaggle-input/datasets/keithtyser', '/kaggle/input/datasets/richardcsaky/arc-arena-hero-v2-v85ql45-sub-runtime',
                     '/kaggle/input/arc-arena-hero-v2-v85ql45-sub-runtime', '/kaggle/input/datasets/richardcsaky/arc-base-patches-v1',
                     '/kaggle/input/datasets/richardcsaky/arc-base-runtime-v1'):
            for dirpath, _, fnames in os.walk(root, followlinks=True):
                for fname in fnames:
                    path = os.path.realpath(os.path.join(dirpath, fname))
                    try:
                        if path.endswith('.safetensors') or not os.path.isfile(path):
                            continue
                        if drop(path):
                            st['rt_gib'] += os.path.getsize(path) / 2**30
                    except OSError:
                        pass
    def follow():
        seen, pattern = 0, _re.compile(r'\\(eager\\):\\s+\\d+% Completed \\| (\\d+)/(\\d+)')
        while True:
            time.sleep(1)
            if not log.is_file():
                continue
            if 'runtime_dropped' not in st['marks']:
                drop_runtime()  # the server log exists: the runtime extract is done
                st['marks']['runtime_dropped'] = {'s': round(time.time() - NOTEBOOK_START_EPOCH), 'gib': round(st['rt_gib'], 1), 'cgroup': _arc_cgroup()}
            with log.open(errors='replace') as stream:
                stream.seek(seen); text = stream.read(); seen = stream.tell()
            if not st['loading'] and 'Loading safetensors checkpoint shards' in text:
                st['loading'] = True; st['marks']['load_start'] = {'s': round(time.time() - NOTEBOOK_START_EPOCH), 'cgroup': _arc_cgroup()}
            for n, total in pattern.findall(text):
                if int(total) == len(files):
                    st['pos'] = max(st['pos'], int(n))
            with lock:
                for j in range(st['dropped'], st['pos']):
                    if begun[j] and names[j] not in keep and drop(files[j]):
                        st['drop_gib'] += sizes[j] / 2**30; st['window'] -= sizes[j]
                    if not begun[j]:
                        begun[j] = None  # passed before its prefetch began: never read
                st['dropped'] = max(st['dropped'], st['pos'])
            if st['pos'] >= len(files):
                st['marks']['load_end'] = {'s': round(time.time() - NOTEBOOK_START_EPOCH), 'cgroup': _arc_cgroup()}
                return
    if '_arc_rt_warm_done' in globals():
        _arc_rt_warm_done.wait(240)  # the runtime blobs first: the bootstrap extracts them from page cache
    threads = [threading.Thread(target=work, args=(k,), daemon=True) for k in range(16)]
    for t in threads: t.start()
    follower = threading.Thread(target=follow, daemon=True); follower.start()
    while follower.is_alive():
        follower.join(30)
        print(json.dumps({'prefetch_progress': {'started': st['files'], 'chunks': st['next'], 'of_chunks': len(items), 'loader': st['pos'], 'of': len(files), 'window_gib': round(st['window'] / 2**30, 1), 'dropped_gib': round(st['drop_gib'], 1), 'waits': st['waits'], 'skipped': st['skipped'], 'seconds': round(time.time() - started), 'cgroup': _arc_cgroup()}}), flush=True)
    print(json.dumps({'prefetch': {'files': st['files'], 'of': len(files), 'runtime_warm': globals().get('_arc_rt_warm'), 'gib': round(sum(sizes) / 2**30, 1), 'dropped_gib': round(st['drop_gib'], 1), 'kept_for_draft': sorted(keep), 'marks': st['marks'], 'seconds': round(time.time() - started, 1)}}), flush=True)
"""
SERVING_WAIT_ANCHOR = "        if expert_overlay is not None:\n            # Expert quantization (28 September)"
SERVING_WAIT = """        if os.environ.get('ARC_STAGE_WAIT'):
            # Kaggle start-up (20 Oct): the notebook stages the served files to local disk; wait for its marker once,
            # before the overlay farm resolves the links (a missing marker only means FUSE reads, as before).
            import time as _time
            _marker, _t0 = Path(os.environ['ARC_STAGE_WAIT']), _time.monotonic()
            while not _marker.exists() and _time.monotonic() - _t0 < float(os.environ.get('ARC_STAGE_WAIT_SECONDS', '900')):
                _time.sleep(1)
            print('ARC_STAGE_WAIT ' + json.dumps({'found': _marker.exists(), 'waited': round(_time.monotonic() - _t0, 1)}),
                  flush=True)
"""
CAND_FILES = ("src/arc_infra/dense_kernels_patch.py", "src/arc_infra/dense_kernels_runtime.py",  # FP8 W8A8 dense
              "src/arena/prefix_tail_runtime.py",  # MTP tail multidrop
              "src/arena/gate.py",  # reply reserve 4,096
              "src/arena/expert_overlay.py")  # strict overlay completeness check
# run-bundle files allowed to differ from the kernel: Kaggle model path, the run's (inactive here) token_stop
CAND_KERNEL_DIFFS = ("src/arc_infra/serving_setup.py", "src/arena/trial.py")
# study keys allowed to differ from the run bundle (Kaggle paths, competition clock, records, run-only diagnostics)
CAND_STUDY_DIFFS = ("/name", "/status", "/wall_seconds", "/notebook_seconds", "/competition/", "/live_progress/interval_seconds",
                    "/live_progress/score_seconds", "/trials[0]/server_env/ARENA_SETUP_TIMEOUTS",
                    "/trials[0]/server_env/ARENA_READY_GUARD", "/priority/park_tokens", "/token_stop", "/b200",
                    "/cache_experiment/prefix_cache/collapse_log", "/v82/level_cap", "/v84", "/v85", "/v85p", "/v86cand",
                    "/probe_retry", "/serving_ablation", "/server_patch", "/study_patch", "/arena_variant/",
                    "/hero_v2/", "/kernel_timeout", "/games", "/roster", "/games_source", "/games_excluded",
                    "/stall_dump_seconds", "/trials[0]/server_env/ARENA_STALL_DUMP",
                    "/trials[0]/server_env/ARC_FP8_CALIB")  # local copy of the same calibration bytes (22 Oct)
V85B_STREAMS = 25  # v8.5b focus target on the FP8 pool
FIXES = ("solvable_line", "death_turn_v2", "findings_keep", "tool_call_followup", "stuck_streak")
SUMMARY = """    try:
        trial = Path('/kaggle/working/arena/trials/duck-transfer')
        arena = json.loads((trial / 'arena.json').read_text())
        engine, progress = arena.get('engine') or {}, ((arena.get('scheduler') or {}).get('priority') or {}).get('progress') or {}
        tick = None
        for name in ('vllm-openai-server.log',):
            if (trial / name).is_file():
                with (trial / name).open(errors='replace') as stream:
                    for text in stream:
                        if 'ARC_ADMIT_TICK_STATS' in text:
                            tick = text[text.index('ARC_ADMIT_TICK_STATS') + 21:].strip()
        print('PREVIEW SUMMARY ' + json.dumps({
            'status': arena.get('status'), 'end_reason': arena.get('end_reason'), 'wall_seconds': arena.get('wall_seconds'),
            'score': arena.get('mean_score'), 'levels': arena.get('levels'), 'games': arena.get('sessions'),
            'actions': arena.get('actions'), 'failed_sessions': arena.get('failed_sessions'),
            'reply_tokens': arena.get('completed_reply_tokens'),
            'tokens_per_second': engine.get('accepted_generation_tokens_per_second'),
            'peak_running': engine.get('peak_running_requests'), 'peak_cache_fraction': engine.get('peak_cache_fraction'),
            'preemptions': engine.get('preemptions'), 'setup_seconds': record.get('setup_seconds'),
            'drains': (arena.get('context_drain') or {}).get('drains'),
            'handovers': {k: progress.get(k) for k in ('handover', 'handover_kept', 'finished')},
            'level_cap': progress.get('level_cap'), 'admission_tick': tick}))
    except Exception as error:  # the summary never fails the notebook
        print('PREVIEW SUMMARY unavailable: ' + repr(error))
"""

CG_SIZES = [128, 160, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096]
CG_PATCHES = [
    ("src/arena/serving.py", "not 1 <= v <= 256 for v in sizes)", "not 1 <= v <= 4096 for v in sizes)"),
    ("src/arena/serving.py", "not 1 <= graph_capture_size <= 256:", "not 1 <= graph_capture_size <= 4096:"),
    ("src/arc_infra/serving_setup.py", "            minimum=0,\n            maximum=256,\n        )",
     "            minimum=0,\n            maximum=4096,\n        )"),
    ("src/arc_infra/serving_setup.py", '"max_cudagraph_capture_size": (0, 256),', '"max_cudagraph_capture_size": (0, 4096),'),
    # The private model copy's parts are linked into one folder by the notebook (arc2 5bfe1806).
    ("src/arc_infra/serving_setup.py", 'MODEL_KAGGLE_PATH = Path(\n    "/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/"\n    "pytorch/radixark-modelopt-fp4/1"\n)',
     '# ARC_MODEL_DIR: the private copy\'s parts, linked into one folder by the notebook (v8.6).\nMODEL_KAGGLE_PATH = Path(os.environ.get("ARC_MODEL_DIR") or (\n    "/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/"\n    "pytorch/radixark-modelopt-fp4/1"\n))'),
]
PRIVATE_FOR = {"keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1": "richardcsaky/arc-base-patches-v1",
               "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1": "richardcsaky/arc-base-runtime-v1"}
# The arc2 kernels' link step (arc2 v29/v30, with the dotfile fix): model parts -> /tmp/arc-model, and a linked input
# root where the private patches/runtime datasets stand in under the public names.
PRIVATE_LINKS = """import os
parts = sorted(Path('/kaggle/input/models/richardcsaky/arc-base-weights').glob('*/part*/*'), key=lambda p: int(p.parent.name[4:]))
assert len(parts) == 9 and all(p.is_dir() for p in parts), f'attach the 9 instances of richardcsaky/arc-base-weights: {parts}'
model = Path('/tmp/arc-model'); model.mkdir(exist_ok=False)
for part in parts:
    for f in part.iterdir():
        link = model / f.name
        if f.name.startswith('.'):
            continue
        if link.is_symlink():
            assert link.stat().st_size == f.stat().st_size, f'{f.name} differs between parts'
            continue
        link.symlink_to(f)
assert (model / 'MODEL_MANIFEST.json').is_file() and (model / 'config.json').is_file(), 'model parts incomplete'
os.environ['ARC_MODEL_DIR'] = str(model)
inputs = Path('/tmp/kaggle-input'); inputs.mkdir(exist_ok=False)
for top in Path('/kaggle/input').iterdir():
    if top.name != 'datasets':
        (inputs / top.name).symlink_to(top)
        continue
    for owner in top.iterdir():
        (inputs / 'datasets' / owner.name).mkdir(parents=True, exist_ok=True)
        for item in owner.iterdir():
            (inputs / 'datasets' / owner.name / item.name).symlink_to(item)
for public, private in (('keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1', 'arc-base-patches-v1'),
                        ('keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1', 'arc-base-runtime-v1')):
    mounted = next((p for p in (Path('/kaggle/input/datasets/richardcsaky') / private, Path('/kaggle/input') / private) if p.is_dir()), None)
    assert mounted is not None, f'attach the dataset richardcsaky/{private}'
    (inputs / 'datasets' / public).parent.mkdir(parents=True, exist_ok=True)
    (inputs / 'datasets' / public).symlink_to(mounted)
print(json.dumps({'private_copy': {'model_files': len(list(model.iterdir())), 'input_root': str(inputs)}}), flush=True)
"""

# AV2C submission (/ 18:40 / 18:55): v8.5b's harness and infrastructure with the serving of a
# dfr-exact AV2C probe bundle (--av2c=DIR): AutoRound W4A16 (private richardcsaky/arc-w4-weights parts 1-2), the base
# checkpoint's FP8 PLE, BF16 dense + keys, FP8 KV, MTP 3, 32 GiB decode offload with the eagle-tail store, no CG.
# Startup cache probe (18 Oct): a client-side transport error (httpx ReadError with the server healthy, gi3m3x v2)
# failed the boot; each probe POST now retries transport errors 3 times with 1/2/4 s backoff (same request, output-exact).
PROBE_RETRY_ANCHOR = "\n\nasync def run(server_dir, study, base_url):\n"
PROBE_RETRY_HELPER = '''

async def _post(client, url, **kwargs):
    """client.post with transport-error retries (startup fragility, 18 Oct): same request, 1/2/4 s backoff."""
    for attempt in range(4):
        try:
            return await client.post(url, **kwargs)
        except httpx.TransportError:
            if attempt == 3:
                raise
            await asyncio.sleep(2 ** attempt)
'''


def patch_probe_retry(path):
    text = path.read_text()
    if "async def _post(client, url" in text:
        return False
    if text.count(PROBE_RETRY_ANCHOR) != 1 or "await client.post(" not in text:
        raise SystemExit(f"cache probe layout changed: {path}")
    text = text.replace("await client.post(", "await _post(client, ")  # call sites first, then the helper
    text = text.replace(PROBE_RETRY_ANCHOR, PROBE_RETRY_HELPER + PROBE_RETRY_ANCHOR)
    path.write_text(text)
    return True

AV2C_FILES = ("bootstrap_arena.py", "src/arena/compatibility.py", "src/arena/cache_probe.py", "src/arena/serving.py",
              "src/arena/research_checkpoint.py", "src/arena/models/intel-autoround-w4a16.json",
              "src/arc_infra/arc_offload_runtime.py", "src/arena/scheduler.py", "src/arena/scheduler_audit.py",
              "src/arena/trial.py", "src/arena/priority.py")
W4_LINKS = """w4parts = sorted(Path('/kaggle/input/models/richardcsaky/arc-w4-weights').glob('*/part*/*'), key=lambda p: int(p.parent.name[4:]))
assert len(w4parts) == 2 and all(p.is_dir() for p in w4parts), f'attach the 2 instances of richardcsaky/arc-w4-weights: {w4parts}'
w4 = Path('/tmp/arc-w4-model'); w4.mkdir(exist_ok=False)
for part in w4parts:
    for f in part.iterdir():
        if not f.name.startswith('.'):
            (w4 / f.name).symlink_to(f)
assert (w4 / 'config.json').is_file() and (w4 / 'model.safetensors.index.json').is_file(), 'w4 parts incomplete'
print(json.dumps({'w4_copy': {'files': len(list(w4.iterdir()))}}), flush=True)
"""
AV2C_PREFETCH = """import threading
def _arc_prefetch():
    started = time.time()
    files = sorted(f.resolve() for f in w4.iterdir() if f.name.endswith('.safetensors'))
    queue, lock, total = iter(files), threading.Lock(), [0, 0]
    def work():
        buffer = bytearray(16 << 20)
        while True:
            with lock:
                item = next(queue, None) if total[0] < 80 * 2**30 else None
                if item is None:
                    return
                total[0] += item.stat().st_size; total[1] += 1
            with open(item, 'rb', buffering=0) as stream:
                while stream.readinto(buffer):
                    pass
    threads = [threading.Thread(target=work, daemon=True) for _ in range(16)]
    for t in threads: t.start()
    while any(t.is_alive() for t in threads):
        threads[0].join(60) if threads[0].is_alive() else time.sleep(60)
        print(json.dumps({'prefetch_progress': {'files_started': total[1], 'gib_started': round(total[0] / 2**30, 1), 'seconds': round(time.time() - started)}}), flush=True)
    print(json.dumps({'prefetch': {'files': total[1], 'of': len(files), 'gib': round(total[0] / 2**30, 1), 'seconds': round(time.time() - started, 1)}}), flush=True)
threading.Thread(target=_arc_prefetch, daemon=True, name='arc-prefetch').start()
"""

# v8.6d: bootstrap's find_games rglob does not descend the symlinks in /tmp/kaggle-input, so find the public games
# under the real /kaggle/input with the same rule. The hidden rerun ignores --games-dir (it plays through the gateway).
GAMES = """_envs = sorted(Path('/kaggle/input').rglob('environment_files'))
_valid = [p for p in _envs if p.is_dir() and len(list(p.glob('*/*.py'))) >= 25] or [p for p in _envs if p.is_dir() and len(list(p.rglob('*.py'))) >= 25]
GAMES_ARGS = ['--games-dir', str(_valid[0])] if len(_valid) == 1 else []
print(json.dumps({'public_games': [str(p) for p in _valid]}), flush=True)
"""

# v8.6b: read the served weights (base files; the expert overlay's shards where it replaces them) with 16 threads from
# the notebook's first minute, so vLLM's single-stream load finds them in the page cache; at most 80 GiB (clean pages,
# reclaimable). The thread is a daemon of the notebook process; its summary line lands in the log.
PREFETCH = """import threading
def _arc_prefetch():
    started = time.time()
    overlay = next((p for p in (Path('/kaggle/input/datasets/richardcsaky/OVERLAY_SLUG'), Path('/kaggle/input/OVERLAY_SLUG')) if p.is_dir()), None)
    shards = set(json.loads((overlay / 'overlay-manifest.json').read_text())['shards']) if overlay else set()
    files = [(overlay / f.name) if f.name in shards else f.resolve() for f in sorted(model.iterdir()) if f.name.endswith('.safetensors')]
    queue, lock, total = iter(files), threading.Lock(), [0, 0]
    def work():
        buffer = bytearray(16 << 20)
        while True:
            with lock:
                item = next(queue, None) if total[0] < 80 * 2**30 else None
                if item is None:
                    return
                total[0] += item.stat().st_size; total[1] += 1
            with open(item, 'rb', buffering=0) as stream:
                while stream.readinto(buffer):
                    pass
    threads = [threading.Thread(target=work, daemon=True) for _ in range(16)]
    for t in threads: t.start()
    while any(t.is_alive() for t in threads):
        threads[0].join(60) if threads[0].is_alive() else time.sleep(60)
        print(json.dumps({'prefetch_progress': {'files_started': total[1], 'gib_started': round(total[0] / 2**30, 1), 'seconds': round(time.time() - started)}}), flush=True)
    print(json.dumps({'prefetch': {'files': total[1], 'of': len(files), 'gib': round(total[0] / 2**30, 1), 'seconds': round(time.time() - started, 1), 'overlay': bool(overlay)}}), flush=True)
threading.Thread(target=_arc_prefetch, daemon=True, name='arc-prefetch').start()
"""

# --l1-shm-fix (25 Oct): one /dev/shm offload region across engine restarts (see the --l1 block)
L1_OFFLOAD_ENGINE_ID = "arc-arena-offload"
L1_SHM_FIX = [
    ("src/arena/serving.py",
     "                'kv_role': 'kv_both',\n                'kv_connector_extra_config':",
     f"                'kv_role': 'kv_both', 'engine_id': '{L1_OFFLOAD_ENGINE_ID}',\n                'kv_connector_extra_config':"),
    ("bootstrap_arena.py",
     '        stop_program = ("from arc_infra.serving import load_recipe; import sys; "\n'
     '                        "load_recipe(sys.argv[1],sys.argv[2]).stop_owned_server()")',
     '        stop_program = ("from arc_infra.serving import load_recipe; import sys, os, glob; "\n'
     '                        "load_recipe(sys.argv[1],sys.argv[2]).stop_owned_server(); "\n'
     '                        "[os.unlink(p) for p in glob.glob(os.environ.get(\'ARENA_SHM_DIR\', \'/dev/shm\') + \'/vllm_offload_*.mmap\')]")'),
]

# v8.6c: the server log's start-up milestones in the kernel log (the bootstrap is silent while vLLM starts).
SERVER_ECHO = """def _arc_cgroup():
    try:
        root = Path('/sys/fs/cgroup')
        if (root / 'memory.max').is_file():
            stat = dict(line.split() for line in (root / 'memory.stat').read_text().splitlines())
            limit = (root / 'memory.max').read_text().strip()
            return {'limit_gib': None if limit == 'max' else round(int(limit) / 2**30, 1),
                    'used_gib': round(int((root / 'memory.current').read_text()) / 2**30, 1),
                    **{k + '_gib': round(int(stat.get(k, 0)) / 2**30, 1) for k in ('anon', 'file', 'shmem')}}
        v1 = root / 'memory'
        stat = dict(line.split() for line in (v1 / 'memory.stat').read_text().splitlines())
        return {'limit_gib': round(int((v1 / 'memory.limit_in_bytes').read_text()) / 2**30, 1),
                'used_gib': round(int((v1 / 'memory.usage_in_bytes').read_text()) / 2**30, 1),
                **{k + '_gib': round(int(stat.get(k, 0)) / 2**30, 1) for k in ('rss', 'cache', 'shmem')}}
    except Exception as error:
        return {'error': type(error).__name__}
def _arc_server_echo():
    log, seen, keys = Path('/tmp/arena/serving/vllm-openai-server.log'), 0, ('took', 'Loading', 'Capturing', 'capturing', 'KV cache', 'Error', 'error', 'Traceback', 'startup complete', 'Compiling', 'Filesystem', 'offload')
    while True:
        time.sleep(60)
        if not log.is_file():
            print(json.dumps({'cgroup': _arc_cgroup()}), flush=True)
            continue
        with log.open(errors='replace') as stream:
            stream.seek(seen); text = stream.read(); seen = stream.tell()
        lines = [l for l in text.splitlines() if any(k in l for k in keys)]
        print(json.dumps({'server_log': {'bytes': seen, 'lines': [l[-200:] for l in lines[-6:]]}, 'cgroup': _arc_cgroup()}), flush=True)
        if any('startup complete' in l for l in lines):
            return
threading.Thread(target=_arc_server_echo, daemon=True, name='arc-server-echo').start()
"""
# 25 Oct (v85l1b hidden-run ERROR): whole-run host-memory trace (cgroup used/anon/file/shmem, /dev/shm offload files)
MEMTRACE = """def _arc_memtrace():
    while True:
        time.sleep(MEMTRACE_S)
        try:
            shm = [(f.name, f.stat().st_size) for f in Path('/dev/shm').iterdir() if f.name.startswith('vllm_offload')]
            free = os.statvfs('/dev/shm'); free = round(free.f_bavail * free.f_frsize / 2**30, 1)
        except Exception as error:
            shm, free = [('error', 0)], repr(error)
        print(json.dumps({'memtrace': {'s': round(time.time() - NOTEBOOK_START_EPOCH), **_arc_cgroup(),
                                       'offload_files': len(shm), 'offload_gib': round(sum(s for _, s in shm) / 2**30, 1), 'shm_free_gib': free}}), flush=True)
threading.Thread(target=_arc_memtrace, daemon=True, name='arc-memtrace').start()
"""

if __name__ == "__main__":
    argv = sys.argv[1:]
    given = {a[2:].split("=", 1)[0]: a.split("=", 1)[1] for a in argv if a.startswith("--") and "=" in a}
    if not given.get("draft-head-overlay"):
        raise SystemExit("v8.2 serves the tuned draft head: --draft-head-overlay=DIR")
    v85b = "--v85b" in argv  # v8.5 + FP8 KV (streams/gate from the FP8 pool) + v8.6d's infrastructure, no CG
    v85p = "--v85p" in argv  # v8.5 serving and harness unchanged on v8.6d's private-copy infrastructure
    infra = v85b or v85p  # v8.6d infrastructure without CG: private copies, prefetch, start limits, games under /kaggle/input
    v85 = "--v85" in argv or v85b or v85p  # v8.4b + BF16 dense projections (as sabl-D) + BF16 indexer keys
    v86 = "--v86" in argv or "--v86b" in argv or "--v86c" in argv or "--v86d" in argv  # v8.5 with dense + keys reverted (= v8.4b) + CG
    v86c = "--v86c" in argv  # v8.6b + the vLLM server log's start-up lines echoed to the kernel log (diagnosis)
    v86d = "--v86d" in argv  # v8.6b + public games found under the real /kaggle/input (rglob skips the symlinked input root)
    v86b = "--v86b" in argv or v86c or v86d  # v8.6 + model prefetch from t=0 + raised start timeouts
    if v85 and v86:
        raise SystemExit("--v85 or --v86, not both")
    v84b = "--v84b" in argv or v85 or v86  # fallback (12 Oct): v8.2 serving unchanged (int4 KV, FP8 indexer keys, 48 streams)
    v84 = "--v84" in argv or v84b
    if v84b and "--v84" in argv or (v85 or v86) and "--v84b" in argv:
        raise SystemExit("--v84 or --v84b, not both")
    if v84 and (given.get("level-cap") or "--kaggle" not in argv):
        raise SystemExit("--v84 is a submission (--kaggle) with the equal scheduler: no --level-cap")
    argv = [a for a in argv if a not in ("--v84", "--v84b", "--v85", "--v85b", "--v85p", "--v86", "--v86b", "--v86c", "--v86d")]
    bad = [a for a in argv if a.startswith(("--off=", "--on=", "--compact", "--stagger", "--version=",
                                             "--completion-cap="))]
    weights = given.get("weights", "excl5")
    held = "--all25-test" in argv  # an owner-ordered test run of a submission's weights (10 October)
    if weights not in ("excl5", "all25") or ("--kaggle" in argv or held) != (weights == "all25") or (
            held and "--kaggle" in argv):
        raise SystemExit("--weights=all25 goes with --kaggle (a submission) or --all25-test (an owner-ordered test "
                         "run); other test runs serve excl5")
    overlay = given.get("overlay")  # with --all25-test: another INT3 overlay under the all-25 mask (the refit)
    if overlay and not held:
        raise SystemExit("--overlay goes with --weights=all25 --all25-test")
    if bad:
        raise SystemExit(f"fixed in v8.2: {bad}")
    values = {**OWN, **{k: v for k, v in given.items() if k in OWN}}
    margin = float(values["gate-margin"])
    if not 0.5 <= margin <= 1.0:
        raise SystemExit("--gate-margin is 0.5..1.0")
    plain = [a for a in argv if not a.startswith(tuple(f"--{k}=" for k in OWN))]
    plain = [a for a in plain if not a.startswith(("--weights=", "--overlay=")) and a not in ("--all25-test", "--v84", "--v84b", "--v85", "--v85b", "--v85p", "--v86", "--v86b", "--v86c", "--v86d")] + [
        *([f"--mask={ALL25_MASK}", f"--overlay={overlay}"] if overlay else [f"--weights={weights}"]),
        "--off=border-moves,sweep,floor"]
    arguments, on = runpy.run_path(str(ROOT / "docs/validation-20261007/build_v8.py"))["plan"](plain)
    if "draft-head" not in on:
        raise SystemExit("the draft head is not on")
    arguments = [a for a in arguments if a != "--version=v8"] + [
        "--version=v8.2", f"--streams={values['streams']}", f"--window={values['window']}",
        f"--density-blocks={values['density-blocks']}"]
    arguments = [a for a in arguments if not a.startswith(("--gate-margin=", "--smoke-trigger=", "--level-cap"))]
    own, sys.argv = sys.argv, ["build_v71.py", *arguments]
    try:
        runpy.run_path(str(ROOT / "docs/validation-20261006/build_v71.py"), run_name="__main__")
    finally:
        sys.argv = own
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
    from build_arena import manifest  # noqa: E402

    from arena import dfr_harness, dfr_loop, hx_harness  # noqa: E402
    from arena.harness_fixes import FIXES as BASE_FIXES, OPTIONAL_FIXES  # noqa: E402

    out = Path(own[1])
    path = out / "configs/comparison.json"
    study = json.loads(path.read_text())
    fixes = study["harness_fixes"]["fixes"]
    if set(FIXES) - set(OPTIONAL_FIXES) or set(fixes) - set(BASE_FIXES) - set(OPTIONAL_FIXES):
        raise SystemExit("harness_fixes names changed")
    study["harness_fixes"]["fixes"] = [*fixes, *(f for f in FIXES if f not in fixes)]
    study["hx_harness"] = hx_harness.settings({**study["hx_harness"], "reflect_skip_death": 1, "memo_style": 2})
    study["dfr_harness"] = dfr_harness.settings({**study["dfr_harness"], "transient": True, "transient_note": False,
                                                 "animation_history": True})
    study["dfr_loop"] = {**study["dfr_loop"], **dfr_loop.settings({**{k: study["dfr_loop"][k] for k in dfr_loop.FLAGS},
                                                                   "no_world_model": True})}
    budget = int(values["window"]) - study["completion_cap"] - 512
    # Half the budget at every window: the 59,392 default is the cap fitted to the 130,816 window.
    study["drain_half"] = {**study["drain_half"], "drain_tokens": budget // 2}
    if margin != study["kv_margin"]:
        if margin <= 0.98 and margin != 0.96:
            raise SystemExit("a margin other than 0.96 at or below 0.98 goes through hero-v2 gate_margin: not wired here")
        study["kv_margin"] = margin
    if values["smoke-trigger"]:
        # Smoke runs only: the drain fires at this many prompt tokens (and drops half of it), so a 20-minute run
        # shows a drain; window, streams and gate stay the bundle's.
        if study["wall_seconds"] > 3600:
            raise SystemExit("--smoke-trigger is for smoke runs (--wall of at most 3,600)")
        study["trim"] = {"budget_tokens": int(values["smoke-trigger"])}
    study["v82"] = {"window": int(values["window"]), "streams": int(values["streams"]), "gate_margin": margin,
                    "prompt_fixes": list(FIXES), "source": "docs/validation-20261009/build_v82.py"}
    if values["level-cap"]:
        from arena import priority  # noqa: E402

        study["priority"] = priority.settings({
            **study["priority"], "level_cap_tokens": int(values["level-cap"]),
            **({"level_cap_grant_tokens": int(values["level-cap-grant"])} if values["level-cap-grant"] else {})})
        study["v82"]["level_cap"] = {k: study["priority"][k] for k in ("level_cap_tokens", "level_cap_grant_tokens")}
    if v84:
        sys.path.insert(0, str(ROOT / "docs/validation-20261009"))
        import v84_equal  # noqa: E402

        v84_equal.apply(out)
        kernels = study["server"]["serving_kernels"]
        if kernels.get("kv_cache_4bit") != "int4" or kernels.get("indexer_fp8") is not True:
            raise SystemExit("v8.4 starts from v8.2's int4 KV and FP8 indexer keys")
        if not v84b:
            kernels.pop("kv_cache_4bit")
            kernels["kv_cache_8bit"] = "fp8"
            kernels["indexer_fp8"] = False
        study["priority"] = {**study["priority"], "equal": True, "park_tokens": 300000, "level_cap_tokens": 0,
                             "tail_fade": False}
        # Streams on the FP8 pool ("scale streams accordingly"): the arena density rule, focus = min(target,
        # floor(0.96 x served blocks) / gate_blocks_per_stream), with v8.2's oversubscription at the FP8 block cost.
        # sabl-K's served layout (RTX PRO 6000, same weights and window): 2,051 blocks of 1,616 tokens, a full
        # 130,816-token window costs 92 blocks (int4: 52 of 2,084), so 20 full windows fit (int4: 36) and the mean
        # request cost over the drain range is 1.711x int4's; v8.2's 41.666 blocks per stream x 1.711 = 71.3, i.e.
        # 27 streams on 1,968 gate blocks (gate density 72.888). sabl-K at 47 slots ran 24 requests at KV 0.79.
        streams = int(values["streams"])
        if v84b and "focus_oversubscription" not in study:
            pass  # v8.5b below: FP8 gate set after v8.5's keys (the plan sets no oversubscription below 48 streams)
        else:
          study["focus_oversubscription"] = study["focus_oversubscription"] if v84b else {
            "focus_slots": streams, "full_windows_fit_served_sablK": 20, "blocks_per_full_window_served_sablK": 92,
            "gate_capacity_projected": int(0.96 * int(values["density-blocks"])),
            "gate_blocks_per_stream": int(0.96 * int(values["density-blocks"])) * 1000 // streams / 1000,
            "min_full_windows": 14, "min_slots": 14,
            "safety": "the exact KV gate (arena.gate): every request reserves cost(prompt + completion cap) blocks and "
                      "the reservations never exceed 0.9 x the served blocks, so requests wait at the gate and none "
                      "can be preempted"}
        if not v84b:
            study.pop("require_focus_windows_fit", None)
            study.pop("focus_fit_fallback", None)
        study["v84"] = {"serving": "v8.2 unchanged: int4 KV, FP8 indexer keys" if v84b else
                        "kv_cache_8bit fp8 in place of int4 KV, BF16 indexer keys (sabl-K)",
                        "scheduler": "priority.equal: fewest generated tokens first; hand-overs at drains, compactions "
                                     "and finished games; parked at 300,000 generated tokens (ranked after every "
                                     "unparked game, fewest tokens first); no level cap, no tail fade",
                        "streams": int(values["streams"]), "density_blocks": int(values["density-blocks"]),
                        "source": "docs/validation-20261009/build_v82.py " + ("--v84b" if v84b else "--v84")}
    if v85:
        # sabl-D (13 Oct, +3.38 [+0.79, +6.39] at 100K): served 1,965 KV blocks (int8 dense: 2,084), so the 48-stream
        # target gates to floor(0.96 x served blocks) / 41.666 (D: 45 focus slots). Streams and gate unchanged.
        study["server"]["dense_kernels"]["int8_weights"] = []
        kernels["indexer_fp8"] = False  # BF16 indexer keys too (sabl-DKb)
        study["v85"] = {"dense": "BF16 dense projections (dense_kernels.int8_weights empty), as sabl-D",
                        "indexer_keys": "BF16 (indexer_fp8 off), as sabl-Kb/DKb",
                        "base": "v8.4b", "source": "docs/validation-20261009/build_v82.py --v85"}
    if infra:
        rel, old, new = CG_PATCHES[-1]  # only the private model path (ARC_MODEL_DIR), not the CG capture bounds
        text = (out / rel).read_text()
        if "ARC_MODEL_DIR" not in new or text.count(old) != 1:
            raise SystemExit(f"v8.5b/p: {rel} model path anchor not found once")
        (out / rel).write_text(text.replace(old, new))
        study["live_progress"] = {**study["live_progress"], "interval_seconds": 300, "score_seconds": 300}
        for trial in study["trials"]:
            env = trial["server_env"]
            if env.get("ARENA_SETUP_TIMEOUTS") != "2700,3600" or env.get("ARENA_READY_GUARD") != "180,3000":
                raise SystemExit("v8.5b/p: start limits changed")
            env.update(ARENA_SETUP_TIMEOUTS="3600,4500", ARENA_READY_GUARD="180,3600")
        study["notebook_seconds"] = study["wall_seconds"] + 5100
    if v85p:
        study["v85p"] = {"base": "v8.5 (serving, harness, streams and gate unchanged)",
                         "infrastructure": "v8.6d: private model copies (arc-base-weights parts 1-9, arc-base-patches-v1 / "
                                           "arc-base-runtime-v1), 16-thread prefetch from t=0, raised start limits, public "
                                           "games under the real /kaggle/input; no CG",
                         "source": "docs/validation-20261009/build_v82.py --v85p"}
    if v85b:
        # FP8 KV in place of int4 (as sabl-Kf/DKf) on v8.5's BF16 dense + BF16 keys, at Kf/DKf's validated gate: the plan
        # stays v8.5's (48-stream oversubscription layout, max_num_seqs 52, graphs to 104) and the focus target drops
        # to what the FP8 pool holds. FP8 blocks hold 1,616 tokens; served pool estimated from the Nebius logs: DKf
        # (BF16 dense, FP8 KV) 1,868 blocks, sabl-K 2,051 with int8 dense; at 72.888 gate blocks per stream the run
        # takes min(25, floor(0.96 x served) / 72.888), i.e. 24-25, and the exact KV gate never reserves past 0.9 x served.
        kernels.pop("kv_cache_4bit")
        kernels["kv_cache_8bit"] = "fp8"
        study["active_limit"] = V85B_STREAMS
        study["focus_oversubscription"].update(focus_slots=V85B_STREAMS, gate_blocks_per_stream=72.888,
                                               min_full_windows=8, min_slots=8)
        study["v85b"] = {"base": "v8.5", "kv": "kv_cache_8bit fp8 in place of int4 KV (BF16 indexer keys kept)",
                         "streams": V85B_STREAMS, "gate_blocks_per_stream": 72.888, "max_num_seqs": study["server"]["max_num_seqs"],
                         "infrastructure": "v8.6d: private model copies (arc-base-weights parts 1-9, arc-base-patches-v1 / "
                                           "arc-base-runtime-v1), 16-thread prefetch from t=0, raised start limits, public "
                                           "games under the real /kaggle/input; no CG",
                         "source": "docs/validation-20261009/build_v82.py --v85b"}
    av2c = values["av2c"]
    if av2c:
        # AV2C serving on v8.5b's harness and infrastructure. The dfr-exact source bundle's scheduler,
        # trial and audit equal v8.5b's plus the cache-budget admission (and an inactive token_stop); serving.py,
        # cache_probe.py and compatibility.py carry the offload connector, decode offload, eagle-tail store, FP8 PLE
        # and load-time pruning on the research checkpoint.
        import shutil

        if not v85b:
            raise SystemExit("--av2c builds on --v85b")
        src = Path(av2c)
        probe = json.loads((src / "configs/comparison.json").read_text())
        if probe["serving_ablation"]["arm"] != "AV2C" or not probe["serving_ablation"].get("ple_fp8"):
            raise SystemExit("--av2c needs a dfr-exact AV2C bundle with --ple-fp8")
        for rel in AV2C_FILES:
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, out / rel)
        # 17 Oct (throughput): the source bundle's PLE prefill bound (--ple-bound: the worker's arc_expert_prune) and
        # engine watchdog (--watchdog: trial restarts a dead server, games wait). Those dfr-exact files are the v8.5b
        # tree's own files plus the patch (same base bytes), so they are copied; the base check refuses anything else.
        extra_files = []
        if probe.get("ple_prefill_bound"):
            extra_files.append("src/arc_infra/expert_prune_runtime.py")
            study["ple_prefill_bound"] = probe["ple_prefill_bound"]
        if probe.get("engine_watchdog"):
            if (probe["engine_watchdog"] or {}).get("test_kill_seconds"):
                raise SystemExit("--av2c: the source bundle's watchdog carries a test kill")
            extra_files += ["src/arena/duck.py", "src/arena/engine_watchdog.py"]
            study["engine_watchdog"] = probe["engine_watchdog"]
        for rel in extra_files:
            if rel != "src/arena/engine_watchdog.py":
                base, patched = (out / rel).read_text(), (src / rel).read_text()
                if not (patched.startswith(base.split("\n    p = the_plan or settings()\n")[0])
                        if rel.endswith("expert_prune_runtime.py") else
                        patched.split("        from . import engine_watchdog as _wd")[0] == base.split(
                            "        for attempt, delay in enumerate((*SERVER_ERROR_DELAYS, None), 1):")[0]):
                    raise SystemExit(f"--av2c: {rel} in the source bundle is not the submission's base plus the patch")
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, out / rel)
        if values["av2c-harness"] == "source":
            # 17 Oct (owner): the source bundle's tool history (no incremental transition views), so the harness
            # equals the evidence run's byte for byte; serving_setup.py keeps the private-copy model path.
            shutil.copyfile(src / "src/duck_transfer/history.py", out / "src/duck_transfer/history.py")
            extra_files.append("src/duck_transfer/history.py")
        elif values["av2c-harness"]:
            raise SystemExit("--av2c-harness takes 'source'")
        server = json.loads(json.dumps(probe["server"]))
        server["draft_head"] = study["server"]["draft_head"]  # the submission's tuned head (its bundle files)
        mask = server["expert_prune"]["mask"]
        if "all25" not in mask:
            raise SystemExit("a submission serves the all-25 mask")
        (out / mask).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / mask, out / mask)
        study["server"] = server
        if server["max_cudagraph_capture_size"] == 4096:  # CG source bundle: the fork's capture bounds too
            for rel, a_old, a_new in CG_PATCHES[2:4]:
                text = (out / rel).read_text()
                if text.count(a_old) != 1:
                    raise SystemExit(f"av2c CG: {rel} anchor not found once")
                (out / rel).write_text(text.replace(a_old, a_new))
        rc = json.loads(json.dumps(probe["research_checkpoint"]))
        rc.update(path="/tmp/arc-w4-model", skip_dotfiles=True)
        rc["ple_fp8"]["root"] = "/tmp/arc-model"  # the private base copy's FP8 PLE files (notebook link step)
        study["research_checkpoint"] = rc
        study["cache_experiment"] = {**study["cache_experiment"], "cpu_cache_gib": probe["cache_experiment"]["cpu_cache_gib"]}
        if probe.get("cache_budget"):
            study["cache_budget"] = probe["cache_budget"]
        study["active_limit"] = probe["active_limit"]
        study["generation_slots"] = server["max_num_seqs"]
        study["focus_oversubscription"] = probe["focus_oversubscription"]
        study["model_datasets"] = [d for d in study["model_datasets"] if "expert-int3" not in d]
        study.pop("stall_dump_seconds", None)
        penv = probe["trials"][0]["server_env"]
        for trial in study["trials"]:
            env = trial["server_env"]
            env.pop("ARENA_STALL_DUMP", None)
            env.update({k: penv[k] for k in ("ARENA_GPU_MEMORY_UTILIZATION", "ARC_PLE_FP8", "ARC_OFFLOAD_EAGLE_TAIL",
                                             "ARC_OFFLOAD_NO_EAGLE_POP") if k in penv})
        study["av2c"] = {"source_bundle": src.name, "serving": probe["serving_ablation"]["server_change"],
                         "mask": mask, "cpu_cache_gib": study["cache_experiment"]["cpu_cache_gib"],
                         "streams_target": study["active_limit"], "cache_budget": study.get("cache_budget"),
                         "model": "private richardcsaky/arc-w4-weights parts 1-2 (AutoRound W4A16, no BF16 PLE) at "
                                  "/tmp/arc-w4-model; FP8 PLE from the private base copy at /tmp/arc-model",
                         "harness_files_from_source": [f for f in extra_files if f.startswith("src/duck_transfer/")],
                         "source": "docs/validation-20261009/build_v82.py --v85b --av2c"}
    if v86:
        # CG (14 Oct, sabl-CG runs): piecewise CUDA graphs up to max_num_batched_tokens (4,096) so steps carrying a
        # prefill over 104 tokens replay graphs (eager: 276-310 ms at 17-42% GPU busy); +7-8% tok/s, KV pool unchanged.
        server = study["server"]
        if server["cudagraph_capture_sizes"][-1] != 104 or server["max_num_batched_tokens"] != 4096:
            raise SystemExit("v8.6: base capture sizes changed")
        server["cudagraph_capture_sizes"] = server["cudagraph_capture_sizes"] + CG_SIZES
        server["max_cudagraph_capture_size"] = 4096
        for rel, old, new in CG_PATCHES:
            text = (out / rel).read_text()
            if text.count(old) != 1:
                raise SystemExit(f"v8.6: {rel} anchor not found once")
            (out / rel).write_text(text.replace(old, new))
        study["live_progress"] = {**study["live_progress"], "interval_seconds": 300, "score_seconds": 300}
        study["v86"] = {"base": "v8.4b (v8.5's BF16 dense and BF16 indexer keys reverted)",
                        "cg": "cudagraph_capture_sizes to 4,096 as the sabl-CG runs (serving.py and the fork's "
                              "serving_setup.py capture bounds 256 -> 4,096)",
                        "model": "private copies: richardcsaky/arc-base-weights parts 1-9 linked to ARC_MODEL_DIR, "
                                 "arc-base-patches-v1 / arc-base-runtime-v1 aliased under the public names",
                        "live_progress": "a status line every 300 s", "source": "docs/validation-20261009/build_v82.py --v86"}
        if v86b:
            # A slow first read of the private model parts must not hit the start limits (v8.6 v1 on Kaggle): server
            # ready within 3,600 s of its launch (ready guard total at its 3,600 maximum), setup total 4,500 s, and a
            # notebook envelope of wall + 5,100 s, so a start of up to ~80 min still leaves the 300 s of play.
            for trial in study["trials"]:
                env = trial["server_env"]
                if env.get("ARENA_SETUP_TIMEOUTS") != "2700,3600" or env.get("ARENA_READY_GUARD") != "180,3000":
                    raise SystemExit("v8.6b: start limits changed")
                env.update(ARENA_SETUP_TIMEOUTS="3600,4500", ARENA_READY_GUARD="180,3600")
            study["notebook_seconds"] = study["wall_seconds"] + 5100
            study["v86"]["prefetch"] = ("the served safetensors (base files, overlay shards over them) read by 16 "
                                        "threads from the notebook's first minute, at most 80 GiB")
            study["v86"]["start_limits"] = {"ARENA_SETUP_TIMEOUTS": "3600,4500", "ARENA_READY_GUARD": "180,3600",
                                            "notebook_seconds": study["notebook_seconds"]}
    if values["safety"]:
        # v8.5p plus the engine watchdog (a dead server restarts, games keep their slots) and the
        # PLE prefill memory bound (bit-identical), from a dfr-exact bundle built with --watchdog --ple-bound.
        # duck.py and expert_prune_runtime.py there are this tree's files plus the patch (pure additions, checked);
        # bootstrap_arena.py and trial.py get only the watchdog lines.
        import difflib
        import shutil

        if not v85p or values["av2c"]:
            raise SystemExit("--safety builds on --v85p alone")
        src = Path(values["safety"])
        probe = json.loads((src / "configs/comparison.json").read_text())
        wd, pb = probe.get("engine_watchdog"), probe.get("ple_prefill_bound")
        if not isinstance(wd, dict) or wd.get("test_kill_seconds") or not pb:
            raise SystemExit("--safety needs a dfr-exact bundle with --watchdog (no test kill) and --ple-bound")
        for rel in ("src/arena/duck.py", "src/arc_infra/expert_prune_runtime.py"):
            base, patched = (out / rel).read_text().splitlines(), (src / rel).read_text().splitlines()
            ops = [op for op in difflib.SequenceMatcher(None, base, patched, autojunk=False).get_opcodes() if op[0] != "equal"]
            if any(op[0] != "insert" for op in ops):
                raise SystemExit(f"--safety: {rel} in the source bundle is not this tree's file plus insertions")
            shutil.copyfile(src / rel, out / rel)
        shutil.copyfile(src / "src/arena/engine_watchdog.py", out / "src/arena/engine_watchdog.py")
        for rel, old, new in (
            ("src/arena/trial.py", "        scheduler.start_all()\n        sampling = asyncio.create_task(monitor.run())\n",
             "        scheduler.start_all()\n        from . import engine_watchdog as _engine_watchdog\n"
             "        _engine_watchdog.start(self.base_url, self.server_dir)\n"
             "        sampling = asyncio.create_task(monitor.run())\n"),
            ("bootstrap_arena.py", "                continue\n            command = [\n",
             "                continue\n"
             "            if study.get(\"engine_watchdog\"):\n"
             "                # Engine watchdog (17 Oct): the trial restarts a dead server with this command (arena.engine_watchdog).\n"
             "                env[\"ARENA_ENGINE_WATCHDOG\"] = json.dumps({\n"
             "                    \"start\": [*serving_command, \"--working\", str(working / \"serving\"), \"--input-root\", args.input_root],\n"
             "                    \"stop\": [python, \"-c\", stop_program, str(working / \"serving\"), args.input_root],\n"
             "                    \"env\": current_env or {},\n"
             "                    \"test_kill_seconds\": (study[\"engine_watchdog\"] if isinstance(study[\"engine_watchdog\"], dict)\n"
             "                                          else {}).get(\"test_kill_seconds\")})\n"
             "            command = [\n")):
            text = (out / rel).read_text()
            if text.count(old) != 1:
                raise SystemExit(f"--safety: {rel} anchor not found once")
            (out / rel).write_text(text.replace(old, new))
            if rel == "bootstrap_arena.py" and (src / rel).read_text().count(new.split("            command = [")[0]) != 1:
                raise SystemExit("--safety: the watchdog block differs from the source bundle's")
        study["engine_watchdog"] = wd
        study["ple_prefill_bound"] = pb
        study["v85p"]["safety"] = {"source_bundle": src.name, "engine_watchdog": wd, "ple_prefill_bound": pb}
    cand = values["cand"]
    if cand:
        # the t50 competition candidate's serving on v8.5p-sub's infrastructure, from its
        # dfr-exact c100 300K run bundle without ucap: FP8 W8A8 dense, MTP 3 + tail multidrop, decode graphs to 4 x
        # seqs, reply reserve 4,096, the t50 all25 mask + its INT3 overlay, 56 streams on 60 seqs. Code: CAND_FILES;
        # study: the run's server block, streams, gate, slots and model datasets (checked against the run below).
        import shutil

        if not v85p or not values["safety"] or av2c:
            raise SystemExit("--cand builds on --v85p --safety")
        src = Path(cand)
        probe = json.loads((src / "configs/comparison.json").read_text())
        server = json.loads(json.dumps(probe["server"]))
        env0 = probe["trials"][0]["server_env"]
        if any(k in env0 for k in ("ARC_UCAP", "ARC_OEA_K0")) or probe.get("token_stop") != 300000:
            raise SystemExit("--cand: the source bundle carries ucap/OEA or is not a 300K run bundle")
        if server["draft_head"] != study["server"]["draft_head"]:
            raise SystemExit("--cand: the source bundle's draft head differs from the submission's")
        mask = server["expert_prune"]["mask"]
        if "all25" not in mask or server["expert_overlay"]["dataset"] not in probe["model_datasets"]:
            raise SystemExit("--cand: a submission serves an all-25 mask with its overlay attached")
        for rel in (*CAND_FILES, mask):
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, out / rel)
        study["server"] = server
        # Calibrated FP8 dense (fp8cq): the run's calibration dataset rides in model_datasets and its
        # server env (ARC_FP8_CALIB is the /kaggle/input/datasets/<owner>/<slug> mount, the same path on Kaggle).
        calib = (probe.get("fp8_dense") or {}).get("calib")
        fp8_env = {k: v for k, v in env0.items() if k.startswith("ARC_FP8_")}
        if calib and (fp8_env.get("ARC_FP8_CALIB") != f"/kaggle/input/datasets/{calib}" or calib not in probe["model_datasets"]):
            raise SystemExit("--cand: the source bundle's FP8 calibration is not its attached dataset")
        if fp8_env and not calib:
            raise SystemExit("--cand: ARC_FP8_* env without fp8_dense.calib")
        if calib:  # served from a local copy made at notebook start (same bytes; the Kaggle mount read took 13 min)
            fp8_env["ARC_FP8_CALIB"] = f"{CALIB_LOCAL}/{calib.split('/')[1]}"
        for trial in study["trials"]:
            trial.setdefault("server_env", {}).update(fp8_env)
        for key in ("active_limit", "focus_oversubscription", "generation_slots", "model_datasets", "fp8_dense",
                    "tail_multidrop", "reply_reserve_tokens"):
            if key in probe:
                study[key] = json.loads(json.dumps(probe[key]))
        study["v86cand"] = {"source_bundle": src.name, "mask": mask, "overlay": server["expert_overlay"],
                           "streams": study["active_limit"], "max_num_seqs": server["max_num_seqs"],
                           "mtp_tokens": server["mtp_tokens"], "graphs_to": server["max_cudagraph_capture_size"],
                           "fp8_w8a8": server["dense_kernels"].get("fp8_w8a8"),
                           "reply_reserve_tokens": probe.get("reply_reserve_tokens"),
                           "files_from_source": list(CAND_FILES),
                           "kaggle_load": "prefetch of every served file (no 80 GiB cap), VLLM_ENGINE_READY_TIMEOUT_S=3000, "
                                          "server start-up milestones echoed to the kernel log",
                           "source": "docs/validation-20261009/build_v82.py --v85p --safety --cand"}
    dfr, hx, pc = study["dfr_harness"], study["hx_harness"], study["cache_experiment"]["prefix_cache"]
    checks = {
        "streams": study["active_limit"] == (V85B_STREAMS if v85b else int(values["streams"])) and study["server"]["max_num_seqs"] == int(values["streams"]) + 4,
        "window": study["windows"] == [int(values["window"])], "cap": study["completion_cap"] == 16384,
        "weights": study["model_datasets"] == [overlay.split(":")[0] if overlay else
            f"richardcsaky/arc-expert-int3-requant-t37-{'all25v71b' if weights == 'all25' else 'excl5'}-v1"]
            and (weights != "all25" or study["server"]["expert_prune"]["mask"].endswith(Path(ALL25_MASK).name)),
        "draft_head": bool((study["server"].get("draft_head") or {}).get("overlay")),
        "host": all(pc.get(k) is True for k in ("reply_ids", "tail_keepalive", "state_cotenant")),
        "dfr": (dfr["transient"], dfr["transient_note"], dfr["sweep"], dfr["border_moves"], dfr["reset_offer"],
                dfr["animation"], dfr["animation_history"]) == (True, False, False, False, False, True, True),
        "hx": (hx["toolbox"], hx["reflect"], hx["reflect_style"], hx["economy"], hx["memo"], hx["memo_style"],
               hx["reflect_skip_death"], hx["no_stuck_notice"]) == (True, True, 2, True, True, 2, 1, False),
        "loop": study["dfr_loop"]["no_world_model"] is True and study["dfr_loop"]["prompt_fix"] is True
                and study["loop_rule"]["action"] == "fresh_context",
        "play": study.get("compaction_keep_fraction") is None and study.get("first_trigger_stagger") is None
                and study["priority"]["floor"] is True and study["drain_half"]["drain_tokens"] == budget // 2,
        "fixes": set(FIXES) <= set(study["harness_fixes"]["fixes"]) and "stuck_notice" in study["harness_fixes"]["fixes"],
        "v84": not v84 or ((study["server"]["serving_kernels"].get("kv_cache_4bit") == "int4"
                            and "kv_cache_8bit" not in study["server"]["serving_kernels"]
                            and study["server"]["serving_kernels"]["indexer_fp8"] is (not v85) if v84b and not v85b else
                            study["server"]["serving_kernels"].get("kv_cache_8bit") == "fp8"
                            and "kv_cache_4bit" not in study["server"]["serving_kernels"]
                            and study["server"]["serving_kernels"]["indexer_fp8"] is False)
                           and study["priority"]["equal"] is True and study["priority"]["park_tokens"] == 300000
                           and study["priority"]["level_cap_tokens"] == 0 and study["priority"]["tail_fade"] is False
                           and "_equal_parked" in (out / "src/arena/scheduler.py").read_text()),
        "v86": not v86 or (study["server"]["cudagraph_capture_sizes"][-1] == 4096
                           and study["server"]["max_cudagraph_capture_size"] == 4096
                           and study["server"]["serving_kernels"]["indexer_fp8"] is True
                           and all(new in (out / rel).read_text() for rel, _, new in CG_PATCHES)),
        "v85b": not v85b or (study["server"]["cudagraph_capture_sizes"][-1] <= 2 * study["server"]["max_num_seqs"]
                             and study["focus_oversubscription"]["focus_slots"] == V85B_STREAMS
                             and study["server"]["serving_kernels"].get("kv_cache_8bit") == "fp8"
                             and "ARC_MODEL_DIR" in (out / "src/arc_infra/serving_setup.py").read_text()
                             and "<= 256 for v in sizes" in (out / "src/arena/serving.py").read_text()),
        "v85p": not v85p or ("ARC_MODEL_DIR" in (out / "src/arc_infra/serving_setup.py").read_text()
                             and study["server"]["serving_kernels"].get("kv_cache_4bit") == "int4"
                             and study["server"]["max_cudagraph_capture_size"] == 104),
        "v85": study["server"]["dense_kernels"]["int8_weights"] == ([] if v85 else
                                                                     ["lm_head", "gdn_in_proj_qkvz", "qsa_qkv", "attn_out"]),
    }
    if av2c:
        kern = study["server"]["serving_kernels"]
        checks.update(streams=study["active_limit"] == study["focus_oversubscription"]["focus_slots"],
                      weights=study["model_datasets"] == [] and "all25" in study["server"]["expert_prune"]["mask"],
                      v84=study["priority"]["equal"] is True and study["priority"]["park_tokens"] == 300000,
                      v85b=((kern.get("kv_cache_8bit") == "fp8" and kern["indexer_fp8"] is False)
                            # G1 class (17 Oct): int4 KV + FP8 indexer keys from the source bundle's KV4/IXF8 levers
                            or (kern.get("kv_cache_4bit") == "int4" and kern["indexer_fp8"] is True))
                           and study["server"]["mtp_tokens"] == 3
                           and "ARC_MODEL_DIR" in (out / "src/arc_infra/serving_setup.py").read_text(),
                      cg=study["server"]["max_cudagraph_capture_size"] == 104 or (
                          # decode graphs (DG, 17 Oct): every decode step graphed, mixed steps eager, within the 256 bound
                          study["server"]["max_num_seqs"] * (1 + study["server"]["mtp_tokens"])
                          <= study["server"]["max_cudagraph_capture_size"] <= 256) or (
                          study["server"]["max_cudagraph_capture_size"] == 4096
                          and all(new in (out / rel).read_text() for rel, _, new in CG_PATCHES[:4])))
    if cand:
        srv = study["server"]
        checks.update(streams=study["active_limit"] == study["focus_oversubscription"]["focus_slots"]
                      and srv["max_num_seqs"] == study["generation_slots"] == study["active_limit"] + 4,
                      weights=study["model_datasets"] == [srv["expert_overlay"]["dataset"]]
                      + ([study["fp8_dense"]["calib"]] if (study.get("fp8_dense") or {}).get("calib") else [])
                      and "all25" in srv["expert_prune"]["mask"],
                      v85p=("ARC_MODEL_DIR" in (out / "src/arc_infra/serving_setup.py").read_text()
                            and srv["serving_kernels"].get("kv_cache_4bit") == "int4"
                            and srv["max_cudagraph_capture_size"] == srv["max_num_seqs"] * (1 + srv["mtp_tokens"]) <= 256
                            and srv["cudagraph_capture_sizes"][-1] == srv["max_cudagraph_capture_size"]))
    if not all(checks.values()):
        raise SystemExit(f"built study is not v8.2: {checks} (model_datasets {study.get('model_datasets')}, "
                         f"fp8_dense {study.get('fp8_dense')})")
    patch_probe_retry(out / "src/arena/cache_probe.py")
    study["probe_retry"] = "transport errors: 3 retries, 1/2/4 s"
    if cand:
        src = Path(cand)
        skip = ("wheels/", "dataset-upload/", "configs/expert-prune/")
        tree = lambda root: {str(f.relative_to(root)) for f in root.rglob("*") if f.is_file()
                             and "__pycache__" not in f.parts and not str(f.relative_to(root)).startswith(skip)}
        own, theirs = tree(out), tree(src)
        meta = {"MANIFEST.json", "build-manifest.json", "configs/comparison.json", "requirements.txt",
                "experiment.ipynb", "kernel-metadata.json"}
        extra = sorted((own ^ theirs) - meta)
        differ = sorted(r for r in (own & theirs) - meta if (out / r).read_bytes() != (src / r).read_bytes())
        if extra or set(differ) - set(CAND_KERNEL_DIFFS):
            raise SystemExit(f"--cand: tree differs from the run bundle: only-one-side {extra}, differ {differ}")
        trial = (src / "src/arena/trial.py").read_text()
        if trial.replace(trial[trial.index("        token_stop = int("):trial.index("        return stop\n") + 20],
                         "").replace("        stop = EarlyStop(", "        return EarlyStop(") != (out / "src/arena/trial.py").read_text():
            raise SystemExit("--cand: trial.py differs from the run bundle's beyond the token_stop block")
        probe = json.loads((src / "configs/comparison.json").read_text())

        def walk(a, b, at, found):
            if isinstance(a, dict) and isinstance(b, dict):
                for k in set(a) | set(b):
                    walk(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{at}/{k}", found)
            elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and a and all(isinstance(x, dict) for x in a):
                for i, (x, y) in enumerate(zip(a, b)):
                    walk(x, y, f"{at}[{i}]", found)
            elif a != b:
                found.append(at)
            return found

        diffs = walk(study, probe, "", [])
        bad = [d for d in diffs if not any(d == k or (k.endswith("/") and d.startswith(k)) or d.startswith(k + "/")
                                           or d.startswith(k + "[") for k in CAND_STUDY_DIFFS)]
        if bad:
            raise SystemExit(f"--cand: study differs from the run bundle beyond Kaggle/competition keys: {bad}")
        study["v86cand"]["study_diffs_vs_run"] = sorted(diffs)
    if "--eager-load" in argv:
        # 19-20 Oct: on Kaggle's GCSFUSE inputs, file cache + the pinned PLE table (~48 GiB) fill the 175 GiB cgroup, reclaim
        # evicts prefetched shards just before the loader reaches them, and lazy mmap faults them back at ~1 MB/s (v86t37
        # v1/v2 stalled). Eager loading reads each shard with one sequential read() (vLLM's --safetensors-load-strategy;
        # every pass, including the MTP draft's).
        rel = out / "src/arc_infra/serving_setup.py"
        text = rel.read_text()
        anchor = '        "--load-format",\n        "safetensors",\n'
        if text.count(anchor) != 1 or "--safetensors-load-strategy" in text:
            raise SystemExit("--eager-load: load-format anchor not found once")
        rel.write_text(text.replace(anchor, anchor + '        "--safetensors-load-strategy",\n        "eager",\n'))
        load = ("vLLM --safetensors-load-strategy eager, notebook prefetch capped at 80 GiB"
                + (" without the plefp8 shards" if "--prefetch-skip-ple" in argv else "") + ", "
                "VLLM_ENGINE_READY_TIMEOUT_S=3000, server start-up milestones and cgroup memory echoed to the kernel log")
        (study["v86cand"] if cand else study)["kaggle_load"] = load
    tool_repair = values["tool-repair"] or None
    if tool_repair:
        # the malformed tool-name repair the l45 mask needs (dfr-exact
        # scripts/dfr_exact/harness/tool_repair.py, as in the l45x5-trep c100 run): drain_half.drop_unknown_tools on.
        # Applied after the --cand tree and study checks against the (repair-free) run bundle.
        import importlib.util

        if not (study.get("drain_half") or {}).get("drop_malformed"):
            raise SystemExit("--tool-repair needs a drain_half with drop_malformed")
        spec = importlib.util.spec_from_file_location("tool_repair", tool_repair)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.apply(out / "src")
        study["drain_half"]["drop_unknown_tools"] = True
        study["tool_repair"] = {"source": Path(tool_repair).name,
                                "sha256": hashlib.sha256(Path(tool_repair).read_bytes()).hexdigest(),
                                "drain_half_key": "drop_unknown_tools"}
    if values["draft-files"]:
        rel = out / "src/arc_infra/expert_prune_patch.py"
        text = rel.read_text()
        close = "               'model)))\\n', 1)],\n}\nTARGETS = (BASE_LOADER, DEFAULT_LOADER)"
        if text.count(close) != 1:
            raise SystemExit("--draft-files: expert_prune_patch DEFAULT_LOADER edits not found once")
        rel.write_text(text.replace(close, close.replace("', 1)],\n}", "', 1),\n" + DRAFT_FILES_EXTRA + "              ],\n}")))
        (study["v86cand"] if cand else study)["kaggle_draft_files"] = "ARC_DRAFT_FILES=1: the MTP draft reads only its shards"
    if values["stage-local"]:
        rel = out / "src/arena/serving.py"
        text = rel.read_text()
        if text.count(SERVING_WAIT_ANCHOR) != 1:
            raise SystemExit("--stage-local: serving.py overlay anchor not found once")
        rel.write_text(text.replace(SERVING_WAIT_ANCHOR, SERVING_WAIT + SERVING_WAIT_ANCHOR))
        (study["v86cand"] if cand else study)["kaggle_stage"] = "local /tmp staging of the served files, ARC_STAGE_WAIT"
        (study["v86cand"] if cand else study)["kaggle_load"] = ("vLLM default (lazy mmap) load of the local copies; no notebook "
            "prefetch; VLLM_ENGINE_READY_TIMEOUT_S=3000, server start-up milestones and cgroup memory echoed to the kernel log")
    if values["l1"]:
        # v85s v3 + the throughput arm's L1 stack (IMPROVEMENTS.md: MTP2 + tail-cache multi-drop
        # + 64 GiB host KV offload with decode/eagle-tail offload + ARC_OFFLOAD_NO_EAGLE_POP=1), ported from the
        # dfr-exact pair L1_BUNDLE,X5_BASE (build_v82_ablation.py --mtp 2 --tail-multidrop --cpu-gib 64
        # --offload-decode --offload-eagle-tail --extra DG): the L1 files replace ours only where ours equal x5's.
        import shutil

        l1, x5 = (Path(x) for x in values["l1"].split(","))
        for rel in ("bootstrap_arena.py", "src/arena/cache_probe.py", "src/arena/compatibility.py",
                    "src/arena/prefix_tail_runtime.py", "src/arena/serving.py"):
            if (out / rel).read_bytes() != (x5 / rel).read_bytes():
                raise SystemExit(f"--l1: {rel} differs from the x5 base; port by hand")
            shutil.copyfile(l1 / rel, out / rel)
        if (out / "src/arc_infra/arc_offload_runtime.py").exists():
            raise SystemExit("--l1: arc_offload_runtime.py already present")
        shutil.copyfile(l1 / "src/arc_infra/arc_offload_runtime.py", out / "src/arc_infra/arc_offload_runtime.py")
        ref = json.loads((l1 / "configs/comparison.json").read_text())
        server = study["server"]
        if server["mtp_tokens"] != 1 or study["cache_experiment"]["cpu_cache_gib"] != 0:
            raise SystemExit("--l1: expected an MTP-1, no-offload base")
        server["mtp_tokens"] = 2
        need = server["max_num_seqs"] * (1 + server["mtp_tokens"])  # decode graphs for every decode step (DG)
        sizes = server["cudagraph_capture_sizes"]
        server["cudagraph_capture_sizes"] = sorted(set(sizes) | set(range(112, need + 16, 16) if need > sizes[-1] else []))
        server["max_cudagraph_capture_size"] = server["cudagraph_capture_sizes"][-1]
        # 64 GiB as measured; v85l1 v1 died silently on Kaggle (empty log, ~5 min into play): the offload is a
        # pre-populated /dev/shm file, non-reclaimable inside the 175 GiB cgroup (v85s v3 play: ~66 GiB vLLM RSS).
        study["cache_experiment"]["cpu_cache_gib"] = int(values["l1-cpu-gib"])
        study["tail_multidrop"] = True
        for trial in study["trials"]:
            trial.setdefault("server_env", {}).update(ARC_OFFLOAD_EAGLE_TAIL="1", ARC_OFFLOAD_NO_EAGLE_POP="1")
        for key in ("mtp_tokens", "cudagraph_capture_sizes", "max_cudagraph_capture_size"):
            if server[key] != ref["server"][key]:
                raise SystemExit(f"--l1: server.{key} {server[key]} != the L1 bundle's {ref['server'][key]}")
        if ref["cache_experiment"]["cpu_cache_gib"] != 64 or not ref.get("tail_multidrop"):
            raise SystemExit("--l1: the L1 bundle is not MTP2 + tail multi-drop + 64 GiB offload")
        study["l1_stack"] = {"source": l1.name, "base": x5.name, "flags": f"mtp 2, tail_multidrop, cpu_cache_gib {values['l1-cpu-gib']}, "
                             "offload decode + eagle tail, ARC_OFFLOAD_NO_EAGLE_POP=1, decode graphs to "
                             f"{server['max_cudagraph_capture_size']}"}
    if values["l1-shm-fix"]:
        # 25 Oct (v85l1b hidden ERROR x2): vLLM names the 32 GiB /dev/shm offload region by a per-start uuid4
        # engine_id and only the creator's close() unlinks it, so each engine-watchdog restart (SIGKILL stop)
        # orphaned 32 GiB of non-reclaimable shmem (Nebius PSFG v2: +31.5 GiB). Fixed engine_id (a restart that
        # finds the file joins it: same size) + the stop program unlinks every vllm_offload_*.mmap after the stop.
        # Without --l1 (no offload region; v85t) only the stop program's unlink applies, and it is inert
        # (the glob finds nothing); the engine_id belongs to the L1 serving.py's offload connector config.
        for rel, old, new in L1_SHM_FIX:
            if not values["l1"] and rel == "src/arena/serving.py":
                continue
            text = (out / rel).read_text()
            if text.count(old) != 1:
                raise SystemExit(f"--l1-shm-fix: anchor not found once in {rel}")
            (out / rel).write_text(text.replace(old, new))
        note = f"engine_id {L1_OFFLOAD_ENGINE_ID}; stop unlinks /dev/shm/vllm_offload_*.mmap"
        if values["l1"]:
            study["l1_stack"]["shm_fix"] = note
        else:
            study["shm_fix"] = "stop unlinks /dev/shm/vllm_offload_*.mmap (no offload region: inert; engine_id is L1-only)"
    if values["graph-sizes"]:
        # CG v3 (throughput, 26 Oct): piecewise graphs over mixed steps up to 512 tokens, exact: the extra capture sizes
        # are the tops of the bitwise row runs common to every served dense GEMM (cginv --stack v87), so a mixed step
        # padded to the next size gives each GEMM an M in the same run as unpadded. Decode graphs stay as served.
        server = study["server"]
        extra = sorted({int(v) for v in values["graph-sizes"].split(",") if v})
        old_max = server["max_cudagraph_capture_size"]
        if not extra or extra[0] <= old_max or extra[-1] > server["max_num_batched_tokens"]:
            raise SystemExit(f"--graph-sizes: sizes above the served {old_max} and <= max_num_batched_tokens")
        server["cudagraph_capture_sizes"] = server["cudagraph_capture_sizes"] + extra
        server["max_cudagraph_capture_size"] = extra[-1]
        if extra[-1] > 256:
            for rel, old, new in CG_PATCHES[:4]:
                text = (out / rel).read_text()
                if text.count(old) != 1:
                    raise SystemExit(f"--graph-sizes: {rel} bound anchor not found once")
                (out / rel).write_text(text.replace(old, new))
        study["cg_v3"] = {"extra_sizes": extra, "decode_graphs_to": old_max,
                          "rule": "extra sizes = bit-exact run tops of the served dense GEMMs (cginv --stack v87)",
                          "source": "docs/validation-20261009/build_v82.py --graph-sizes"}
    if values["profile-at"]:
        # Step profile (throughput, 26 Oct): the server's torch profiler armed (no cost until /start_profile; 3-step
        # delay, 40 steps), fired by the trial loop at these seconds of play; traces land in serving/torch-profile.
        at = sorted(float(v) for v in values["profile-at"].split(",") if v)
        for rel, old, new in PROFILE_PATCHES:
            text = (out / rel).read_text()
            if text.count(old) != 1:
                raise SystemExit(f"--profile-at: {rel} anchor not found once")
            (out / rel).write_text(text.replace(old, new))
        (out / "src/arena/trial.py").write_text((out / "src/arena/trial.py").read_text() + PROFILE_FIRE)
        for trial in study["trials"]:
            trial.setdefault("server_env", {})["ARENA_TORCH_PROFILER"] = "1"
        study["profile_at"] = at
    if values["keep-tokens"]:
        # a drain stops at this many tokens (context_drain keep_tokens) instead of
        # drain_tokens below the budget: fewer, deeper drains (less recompute) and a smaller mean context.
        keep = int(values["keep-tokens"])
        if not 1024 <= keep < budget - int(study["drain_half"]["drain_tokens"]):
            raise SystemExit("--keep-tokens: below the current drain target")
        study["drain_half"] = {**study["drain_half"], "keep_tokens": keep}
    if values["dn4g8"]:
        # dn4g8 = 4-bit (NVFP4) down projections everywhere + NVFP4 gate/up in 8 layers on the served
        # INT3 overlay, from the dfr-exact pair DN4G8_BUNDLE,ITS_BASE (C1: L1 + dn4g8 vs L1). Runtime-only: the overlay's
        # INT3 tensors for those sublayers are ignored and the base NVFP4 ones load.
        import shutil

        # DN4G8_BUNDLE,ITS_BASE,OWNER/DATASET@MANIFEST_SHA256: the served overlay must hold the NVFP4 checkpoint's
        # down_proj (and gate/up of the listed layers): the runtime keeps those tensors as loaded, so on a plain INT3
        # overlay they would serve INT3-grid values (docs/validation-20261009/modal_dn4g8_overlay.py builds one).
        src, base, ov = values["dn4g8"].split(",")
        src, base = Path(src), Path(base)
        dataset, sha = ov.split("@")
        old = study["server"]["expert_overlay"]["dataset"]
        if "dn4g8" not in dataset or len(sha) != 64 or old not in study["model_datasets"]:
            raise SystemExit("--dn4g8: OWNER/DATASET@MANIFEST_SHA256 of a dn4g8 overlay")
        study["server"]["expert_overlay"].update(dataset=dataset, manifest_sha256=sha)
        study["model_datasets"] = [dataset if d == old else d for d in study["model_datasets"]]
        if "v86cand" in study:  # the cand record names the source run's overlay; keep it the served one
            study["v86cand"]["overlay"] = dict(study["server"]["expert_overlay"])
        rel = "src/arc_infra/expert_int3_runtime.py"
        if (out / rel).read_bytes() != (base / rel).read_bytes():
            raise SystemExit(f"--dn4g8: {rel} differs from the dn4g8 bundle's base; port by hand")
        shutil.copyfile(src / rel, out / rel)
        env = json.loads((src / "configs/comparison.json").read_text())["trials"][0]["server_env"]
        keep = {k: env[k] for k in ("ARC_EXPERT_INT3_FP4_DOWN", "ARC_EXPERT_INT3_FP4_LAYERS")}
        if keep["ARC_EXPERT_INT3_FP4_DOWN"] != "1" or not keep["ARC_EXPERT_INT3_FP4_LAYERS"]:
            raise SystemExit("--dn4g8: the source bundle is not dn4g8")
        for trial in study["trials"]:
            trial.setdefault("server_env", {}).update(keep)
        study["dn4g8"] = {"source": src.name, "base": base.name, **keep,
                          "overlay": study["server"]["expert_overlay"]["dataset"]}
    path.write_text(json.dumps(study, indent=2) + "\n")
    identity = json.loads((out / "MANIFEST.json").read_text())
    identity["files"] = manifest(out)
    (out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    runpy.run_path(str(out / "bootstrap.py"))["verify"](out)
    if "--kaggle" in argv:
        # build_v71.py packed the archive before the study changed above: pack it again, with its image pin.
        from build_arena import kaggle_files  # noqa: E402

        before = json.loads((out / "kernel-metadata.json").read_text())
        built = json.loads((out / "build-manifest.json").read_text())["runtime_dataset"]
        runtime = values["runtime-dataset"] or built
        digest = kaggle_files(out, study, smoke=False, runtime_dataset=runtime)
        meta = json.loads((out / "kernel-metadata.json").read_text())
        meta.update({k: before[k] for k in ("docker_image", "docker_image_pinning_type")})
        if not (av2c or cand) and {**meta, "dataset_sources": [built, *meta["dataset_sources"][1:]]} != before:
            raise SystemExit("kernel metadata changed in the second packing")
        if not study["competition"].get("no_preview"):
            # The public preview leaves a readable result in the kernel log; never on the rerun, never fatal.
            book = json.loads((out / "experiment.ipynb").read_text())
            source = book["cells"][2]["source"]
            text = source if isinstance(source, str) else "".join(source)
            line = "    print('Public validation completed. The placeholder selects the notebook for a hidden rerun.')"
            if text.count(line) != 1:
                raise SystemExit("notebook layout changed: no place for the preview summary")
            save = SAVE_COMPILE_CACHE if values["save-compile-cache"] else ""
            book["cells"][2]["source"] = text.replace(line, SUMMARY + save + line)
            (out / "experiment.ipynb").write_text(json.dumps(book, indent=1))
        if v86 or infra:
            # Private model copy: never keithtyser's public inputs.
            meta["dataset_sources"] = [d for d in meta["dataset_sources"] if d not in PRIVATE_FOR] + list(PRIVATE_FOR.values())
            meta["model_sources"] = [f"richardcsaky/arc-base-weights/PyTorch/part{k}/1" for k in range(1, 10)]
            if av2c:
                meta["model_sources"] += [f"richardcsaky/arc-w4-weights/PyTorch/part{k}/1" for k in (1, 2)]
            if any("keithtyser" in d for d in meta["dataset_sources"] + meta["model_sources"]):
                raise SystemExit("v8.6 attaches only private copies")
            book = json.loads((out / "experiment.ipynb").read_text())
            source = book["cells"][1]["source"]
            text = source if isinstance(source, str) else "".join(source)
            anchor = "command = [sys.executable, str(bundle / 'bootstrap_arena.py'), '--bundle', str(bundle), '--working', '/tmp/arena',"
            if text.count(anchor) != 1 or "'--max-seconds'" not in text:
                raise SystemExit("notebook layout changed: no place for the private-copy links")
            prefetch = PREFETCH.replace("OVERLAY_SLUG", study["server"]["expert_overlay"]["dataset"].split("/")[1])
            ready_env = ""
            eager = "--eager-load" in argv
            if cand or eager:
                # Kaggle inputs are GCSFUSE now (vLLM skips its own prefetch there and mmap-loads
                # each shard slowly), so prefetch every served file (no 80 GiB cap; the host has ~180 GB) and give the
                # engine 3,000 s to come up (vLLM's default 600 s killed arc2 v39's frontends); start limits stay above it.
                # 20 Oct: --eager-load applies to any v8.5p-infrastructure build (v8.5q), not only --cand.
                cap = "item = next(queue, None) if total[0] < 80 * 2**30 else None"
                if prefetch.count(cap) != 1:
                    raise SystemExit("--cand: prefetch cap line not found once")
                if not eager:  # with eager loading the 80 GiB cap stays (v86t37 v3)
                    prefetch = prefetch.replace(cap, "item = next(queue, None)")
                if "--prefetch-skip-ple" in argv:
                    # 20 Oct: prefetch only the overlay experts + bf16 (~55 GiB) so file cache + the pinned PLE table
                    # (~48 GiB) + runtime stay under the 175 GiB cgroup; eager streams the 4.7 GiB plefp8 shards.
                    sel = "if f.name.endswith('.safetensors')]"
                    if prefetch.count(sel) != 1:
                        raise SystemExit("--prefetch-skip-ple: prefetch file filter not found once")
                    prefetch = prefetch.replace(sel, "if f.name.endswith('.safetensors') and not f.name.startswith('model-plefp8')]")
                ready_env = "os.environ['VLLM_ENGINE_READY_TIMEOUT_S'] = '3000'\n"
            if eager and not (v86b or infra) or av2c and eager:
                raise SystemExit("--eager-load needs the v8.5p/v8.6b prefetch infrastructure")
            calib = (study.get("fp8_dense") or {}).get("calib")
            calib_check = (f"assert Path('/kaggle/input/datasets/{calib}').is_dir(), 'attach the dataset {calib}'\n"
                           if cand and calib else "")
            calib_join = ""
            if calib_check:
                # v85f v1 preview (21 Oct): the FP8 dense step read the 196 calibration files through the Kaggle input
                # mount at ~3.5 MB/s (13 min of setup). Copy them to local disk in parallel at notebook start (page-cache
                # prefetch alone can be evicted under the cgroup's weight prefetch); the run waits for the copy.
                local = f"{CALIB_LOCAL}/{calib.split('/')[1]}"
                calib_check += (
                    "import os, threading as _arc_th, shutil as _arc_sh, concurrent.futures as _arc_cf\n"
                    f"_ARC_CALIB_SRC, _ARC_CALIB_DST = Path('/kaggle/input/datasets/{calib}'), Path('{local}')\n"
                    "def _arc_calib_copy_one(f):\n"
                    "    dst = _ARC_CALIB_DST / f.name\n"
                    "    if dst.is_file() and dst.stat().st_size == f.stat().st_size:\n"
                    "        return\n"
                    "    tmp = _ARC_CALIB_DST / (f.name + '.part')\n"
                    "    _arc_sh.copyfile(f, tmp)\n"
                    "    os.replace(tmp, dst)\n"
                    "def _arc_calib_copy():\n"
                    "    t0 = time.time()\n"
                    "    _ARC_CALIB_DST.mkdir(parents=True, exist_ok=True)\n"
                    "    files = sorted(f for f in _ARC_CALIB_SRC.iterdir() if f.is_file())\n"
                    "    with _arc_cf.ThreadPoolExecutor(16) as ex:\n"
                    "        list(ex.map(_arc_calib_copy_one, files))\n"
                    "    print(json.dumps({'t_mark': 'calib_copy', 'files': len(files), 'copy_s': round(time.time() - t0, 1), "
                    "'s': round(time.time() - NOTEBOOK_START_EPOCH, 1)}), flush=True)\n"
                    "_arc_calib_thread = _arc_th.Thread(target=_arc_calib_copy, daemon=True, name='arc-calib-copy')\n"
                    "_arc_calib_thread.start()\n")
                calib_join = (
                    "_arc_t_join = time.time()\n"
                    "_arc_calib_thread.join()\n"
                    "for _f in sorted(f for f in _ARC_CALIB_SRC.iterdir() if f.is_file()):  # repair anything the thread missed\n"
                    "    _arc_calib_copy_one(_f)\n"
                    "assert sorted(p.name for p in _ARC_CALIB_DST.iterdir()) == sorted(p.name for p in _ARC_CALIB_SRC.iterdir() if p.is_file())\n"
                    "print(json.dumps({'t_mark': 'calib_join', 'wait_s': round(time.time() - _arc_t_join, 1), "
                    "'s': round(time.time() - NOTEBOOK_START_EPOCH, 1)}), flush=True)\n")
            links = PRIVATE_LINKS + calib_check + ready_env + (W4_LINKS + AV2C_PREFETCH if av2c else prefetch if v86b or infra else "") + (
                SERVER_ECHO if v86c or cand or eager else "") + (GAMES if v86d or infra else "")
            text = text.replace(anchor, links + anchor.replace("'--working', '/tmp/arena',",
                                                                    "'--working', '/tmp/arena', '--input-root', '/tmp/kaggle-input',"
                                                                    + (" *GAMES_ARGS," if v86d or infra else "")))
            if calib_join:
                run_line = "result = subprocess.run(command, check=False)"
                if text.count(run_line) != 1:
                    raise SystemExit("calib copy: run anchor not found once")
                text = text.replace(run_line, calib_join + run_line)
            book["cells"][1]["source"] = text
            (out / "experiment.ipynb").write_text(json.dumps(book, indent=1))
        (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        if values["overlay-compose"]:
            # 20 Oct (Kaggle private quota): the served overlay is composed on Kaggle from v8.5p's t37 all25 dataset and a
            # small delta dataset (shards + composed overlay-manifest.json), one symlink per file, before anything reads it.
            base_ds, delta_ds = values["overlay-compose"].split(",")
            virtual = study["server"]["expert_overlay"]["dataset"]
            if meta["dataset_sources"].count(virtual) != 1:
                raise SystemExit("--overlay-compose: the virtual overlay is not a dataset source once")
            at = meta["dataset_sources"].index(virtual)
            meta["dataset_sources"][at:at + 1] = [base_ds, delta_ds]
            (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
            book = json.loads((out / "experiment.ipynb").read_text())
            source = book["cells"][1]["source"]
            text = source if isinstance(source, str) else "".join(source)
            anchor = "print(json.dumps({'private_copy':"
            v, b, d = (x.split("/")[1] for x in (virtual, base_ds, delta_ds))
            compose = (
                f"def _arc_mount(name):\n"
                f"    found = next((p for p in (Path('/kaggle/input/datasets/richardcsaky') / name, Path('/kaggle/input') / name) if p.is_dir()), None)\n"
                f"    assert found is not None, f'attach the dataset richardcsaky/{{name}}'\n"
                f"    return found\n"
                f"_ov_base, _ov_delta = _arc_mount('{b}'), _arc_mount('{d}')\n"
                f"_ov = inputs / 'datasets' / 'richardcsaky' / '{v}'\n"
                f"_ov.mkdir(parents=True, exist_ok=False)\n"
                f"_ov_manifest = json.loads((_ov_delta / 'overlay-manifest.json').read_text())\n"
                f"for _name in sorted({{f.name for f in _ov_base.iterdir()}} | {{f.name for f in _ov_delta.iterdir()}}):\n"
                f"    if _name in ('overlay-manifest.json', 'build-stats.json', 'verify.json'):\n"
                f"        continue\n"
                f"    (_ov / _name).symlink_to((_ov_delta if (_ov_delta / _name).is_file() else _ov_base) / _name)\n"
                f"(_ov / 'overlay-manifest.json').symlink_to(_ov_delta / 'overlay-manifest.json')\n"
                f"_ov_bad = [n for n, s in _ov_manifest['shards'].items() if not (_ov / n).is_file() or (_ov / n).stat().st_size != s['bytes']]\n"
                f"assert not _ov_bad, f'composed overlay differs from its manifest: {{_ov_bad[:3]}}'\n"
                f"print(json.dumps({{'overlay_compose': {{'dir': str(_ov), 'delta': sorted(n for n in _ov_manifest['shards'] if (_ov_delta / n).is_file())}}}}), flush=True)\n")
            if text.count(anchor) != 1:
                raise SystemExit("--overlay-compose: notebook anchor not found once")
            text = text.replace(anchor, compose + anchor)
            pre = (f"(Path('/kaggle/input/datasets/richardcsaky/{v}'), Path('/kaggle/input/{v}'))")
            if text.count(pre) != 1:
                raise SystemExit("--overlay-compose: prefetch overlay lookup not found once")
            text = text.replace(pre, f"(Path('/tmp/kaggle-input/datasets/richardcsaky/{v}'),)")
            book["cells"][1]["source"] = text
            (out / "experiment.ipynb").write_text(json.dumps(book, indent=1))
            print(json.dumps({"overlay_compose": {"virtual": virtual, "base": base_ds, "delta": delta_ds}}))
        if values["draft-files"] or values["prefetch-uncapped"]:
            book = json.loads((out / "experiment.ipynb").read_text())
            source = book["cells"][1]["source"]
            text = source if isinstance(source, str) else "".join(source)
            if values["draft-files"]:
                env = "os.environ['VLLM_ENGINE_READY_TIMEOUT_S'] = '3000'\n"
                if text.count(env) != 1:
                    raise SystemExit("--draft-files: notebook env anchor not found once")
                text = text.replace(env, env + "os.environ['ARC_DRAFT_FILES'] = '1'\n")
            if values["prefetch-window"]:
                pre, post = values["prefetch-window"].split(",")
                i0, i1 = text.find("def _arc_prefetch():\n"), text.find("threading.Thread(target=_arc_prefetch")
                echo = text.find("def _arc_cgroup():")
                if min(i0, i1, echo) < 0 or not i0 < i1 or "--eager-load" not in argv or values["prefetch-uncapped"]:
                    raise SystemExit("--prefetch-window: prefetch block not found (or no --eager-load, or with --prefetch-uncapped)")
                virtual = study["server"]["expert_overlay"]["dataset"].split("/")[1]
                body = PREFETCH_WINDOW.replace("OVERLAY_SLUG", virtual).replace("PRE_GIB", pre).replace("POST_GIB", post)
                # v85s v2 preview (22 Oct, slow model mount ~0.15 GiB/s): PLE-first held 47.7 of the 60 GiB window
                # for the whole load (12 GiB lookahead, loader read 25 shards cold) and its pages were evicted before
                # the loader reached them (re-read at the tail, 17 s/shard). load: prefetch in load order only.
                if values["prefetch-order"] == "load":
                    ple_first = "    order = ple + [i for i in range(len(names)) if i not in set(ple)]  # v8: the cold PLE tail first, then load order\n"
                    if body.count(ple_first) != 1:
                        raise SystemExit("--prefetch-order: order line not found once")
                    body = body.replace(ple_first, "    order = list(range(len(names)))  # load order (v85s v3)\n")
                elif values["prefetch-order"]:
                    raise SystemExit("--prefetch-order=load only")
                if values["prefetch-readers"]:
                    readers = "for k in range(16)]"
                    if body.count(readers) != 1:
                        raise SystemExit("--prefetch-readers: reader line not found once")
                    body = body.replace(readers, f"for k in range({int(values['prefetch-readers'])})]")
                if values["runtime-dataset"]:  # the runtime blob dataset's cache is dropped too
                    body = body.replace("arc-arena-hero-v2-v85ql45-sub-runtime", values["runtime-dataset"].split("/")[1])
                start_line = "threading.Thread(target=_arc_prefetch, daemon=True, name='arc-prefetch').start()\n"
                text = text[:i0] + text[i1:]  # drop the old body; it starts after the cgroup helper is defined
                text = text.replace(start_line, "")
                anchor_echo = "threading.Thread(target=_arc_server_echo, daemon=True, name='arc-server-echo').start()\n"
                if text.count(anchor_echo) != 1:
                    raise SystemExit("--prefetch-window: server echo anchor not found once")
                text = text.replace(anchor_echo, anchor_echo + body + start_line)
                epoch = "NOTEBOOK_START_EPOCH = time.time()\n"
                if text.count(epoch) != 1:
                    raise SystemExit("--prefetch-window: notebook start anchor not found once")
                text = text.replace(epoch, epoch + RUNTIME_WARM)
                for mark, anchor in (("runtime_untar", "    archive.extractall(bundle, filter='data')\n"),
                                     ("model_links", "os.environ['ARC_MODEL_DIR'] = str(model)\n"),
                                     ("input_links", "            (inputs / 'datasets' / owner.name / item.name).symlink_to(item)\n")):
                    if text.count(anchor) != 1:
                        raise SystemExit(f"--prefetch-window: {mark} anchor not found once")
                    text = text.replace(anchor, anchor + f"print(json.dumps({{'t_mark': '{mark}', 's': round(time.time() - NOTEBOOK_START_EPOCH, 1)}}), flush=True)\n")
            if values["prefetch-uncapped"]:
                cap = "item = next(queue, None) if total[0] < 80 * 2**30 else None"
                ple = " and not f.name.startswith('model-plefp8')]"
                if text.count(cap) != 1:
                    raise SystemExit("--prefetch-uncapped: prefetch cap not found once")
                text = text.replace(cap, "item = next(queue, None)").replace(ple, "]")
            book["cells"][1]["source"] = text
            (out / "experiment.ipynb").write_text(json.dumps(book, indent=1))
        if values["compile-cache"] and not values["stage-local"]:
            # 20 Oct (l45 v8): restore a preview's compile / Triton caches before the bootstrap (a miss recompiles)
            book = json.loads((out / "experiment.ipynb").read_text())
            source = book["cells"][1]["source"]
            text = source if isinstance(source, str) else "".join(source)
            run = "result = subprocess.run(command, check=False)"
            if text.count(run) != 1:
                raise SystemExit("--compile-cache: run anchor not found once")
            cache = values["compile-cache"]
            meta["dataset_sources"].append(cache)
            (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
            text = text.replace(run, RESTORE_CACHE.replace("CACHE_SLUG", cache.split("/")[1]) + run)
            book["cells"][1]["source"] = text
            (out / "experiment.ipynb").write_text(json.dumps(book, indent=1))
        if values["stage-local"]:
            book = json.loads((out / "experiment.ipynb").read_text())
            source = book["cells"][1]["source"]
            text = source if isinstance(source, str) else "".join(source)
            start = "threading.Thread(target=_arc_prefetch, daemon=True, name='arc-prefetch').start()\n"
            run = "result = subprocess.run(command, check=False)"
            echo = "threading.Thread(target=_arc_server_echo, daemon=True, name='arc-server-echo').start()\n"
            if text.count(start) != 1 or text.count(run) != 1 or text.count(echo) != 1 or "--eager-load" in argv:
                raise SystemExit("--stage-local: notebook anchors not found once (or --eager-load given)")
            virtual = study["server"]["expert_overlay"]["dataset"].split("/")[1]
            # the staging thread starts after the cgroup helper exists (server echo block), before the games scan
            text = text.replace(start, "").replace(echo, echo + STAGE.replace("OVERLAY_SLUG", virtual))
            gate = STAGE_GATE
            if values["compile-cache"]:
                cache = values["compile-cache"]
                meta["dataset_sources"].append(cache)
                (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
                gate = RESTORE_CACHE.replace("CACHE_SLUG", cache.split("/")[1]) + gate
            text = text.replace(run, gate + run)
            book["cells"][1]["source"] = text
            (out / "experiment.ipynb").write_text(json.dumps(book, indent=1))
        if values["memtrace"]:
            book = json.loads((out / "experiment.ipynb").read_text())
            source = book["cells"][1]["source"]
            text = source if isinstance(source, str) else "".join(source)
            echo = "threading.Thread(target=_arc_server_echo, daemon=True, name='arc-server-echo').start()\n"
            if text.count(echo) != 1 or "NOTEBOOK_START_EPOCH = time.time()" not in text:
                raise SystemExit("--memtrace: server echo / notebook start anchor not found once")
            text = text.replace(echo, echo + MEMTRACE.replace("MEMTRACE_S", str(int(values["memtrace"]))))
            book["cells"][1]["source"] = text
            (out / "experiment.ipynb").write_text(json.dumps(book, indent=1))
        print(json.dumps({"kaggle": {"kernel": meta["id"], "runtime_dataset": runtime, "blob_sha256": digest}}))
    drain = min(int(study["drain_half"]["drain_tokens"]), budget // 2)
    if study["drain_half"].get("keep_tokens"):
        drain = budget - int(study["drain_half"]["keep_tokens"])
    print(json.dumps({"v82": study["v82"], "max_num_seqs": study["server"]["max_num_seqs"],
                      "graphs": study["server"].get("cudagraph_capture_sizes"), "kv_margin": study["kv_margin"],
                      "context_budget": budget, "drain_keeps": budget - drain, "wall": study["wall_seconds"],
                      "notebook_seconds": study["notebook_seconds"], "games": len(study["games"]),
                      "draft_head": study["server"].get("draft_head"), "gate": study.get("focus_oversubscription")}))
