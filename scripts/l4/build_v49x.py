"""l4 arm bundles over arc2-kaggle-v49 (exact v49 harness and serving bytes; only the named edits).
  rk32: + host KV tier (ArcOffloadingConnector, the files arc2-kaggle-v45off added to v45: option cpu_cache_gib)
        + option reclaim (cache_experiment.reclaim_skipped_blocks, sparse skipped-block reclamation; off on L4 before).
  car:  + options custom_ar (arc2.l4_comm: vLLM's custom all-reduce over PCIe peer access) and nccl_p2p_level.
  dbg:  + CUDA_LAUNCH_BLOCKING=1 in the server env (debug read: names the kernel of A's illegal memory access).
  ple:  + v87b's expert_prune_runtime.py byte for byte (= v49's + install_ple_prefill_bound).
usage: build_v49x.py rk32|car|dbg|ple   ->  dist/arc2-kaggle-v49<variant> (rk32 -> v49rk; the run picks the options)"""
import hashlib, json, os, runpy, shutil, subprocess, sys
from pathlib import Path

ARC2 = Path('./.claude/worktrees/arc2/dist')
HERE = Path(__file__).resolve().parents[2]
V49, V45, V45OFF = ARC2 / 'arc2-kaggle-v49', ARC2 / 'arc2-kaggle-v45', ARC2 / 'arc2-kaggle-v45off'
OFF = ('src/arc2/kaggle.py', 'src/arc2/l4_serving.py', 'src/arena/serving.py', 'src/arena/compatibility.py')
NEW = ('src/arc_infra/arc_offload_runtime.py',)
V87B = Path('./.claude/worktrees/v84sub/dist/arena-flash-competition-v87b-sub')
PRUNE = 'src/arc_infra/expert_prune_runtime.py'
GUARD = '        if options.get("kv4_guard"):\n            env["ARC_KV4_GUARD"] = "1"\n'
DBG = GUARD + '        env["CUDA_LAUNCH_BLOCKING"] = "1"  # l4 debug bundle: synchronous launches name the faulting kernel\n'
EDITS = [  # (old, new) in src/arc2/l4_serving.py, each exactly once
    ('warm_rht=False, context=None, cpu_cache_gib=0):', 'warm_rht=False, context=None, cpu_cache_gib=0, reclaim=False):'),
    ('''                               "reclaim_skipped_blocks": False}
    if prefix_patches:''', '''                               "reclaim_skipped_blocks": bool(reclaim)}  # l4 arm: arena.cache_reclamation
    if prefix_patches:'''),
    ('''                            "cpu_cache_gib": int(cpu_cache_gib or 0),''', '''                            "cpu_cache_gib": int(cpu_cache_gib or 0), "reclaim": bool(reclaim),'''),
    ('"block_tokens", "warmup_rows", "warmup_dump", "warm_rht", "context", "cpu_cache_gib"):',
     '"block_tokens", "warmup_rows", "warmup_dump", "warm_rht", "context", "cpu_cache_gib", "reclaim"):'),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(variant):
    assert variant in ('rk32', 'car', 'dbg', 'dbg2', 'dbg3', 'dbg4', 'dbg5', 'dbg6', 'dbg7', 'dbg8', 'dbg9', 'eag', 'ple'), variant
    for rel in OFF:  # v49 = v45 + loop/tasks; v45off edited only these on v45
        assert sha(V49 / rel) == sha(V45 / rel), rel
    rk = variant == 'rk32'
    out = HERE / 'dist' / ('arc2-kaggle-v49rk' if rk else f'arc2-kaggle-v49{variant}')
    edited = (set(OFF) if rk else {PRUNE} if variant == 'ple' else
              {'src/arc2/l4_serving.py', 'src/arc2/l4_logits.py'} if variant in ('dbg2', 'dbg3', 'dbg4', 'dbg5', 'dbg6', 'dbg7', 'dbg8', 'dbg9', 'eag') else {'src/arc2/l4_serving.py'})
    if variant == 'ple':  # v87b's file must be v49's with lines added only
        new = iter((V87B / PRUNE).read_text().splitlines())
        assert all(any(line == n for n in new) for line in (V49 / PRUNE).read_text().splitlines())
        assert 'def install_ple_prefill_bound' in (V87B / PRUNE).read_text()
    shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    for p in V49.rglob('*'):
        rel = p.relative_to(V49)
        if rel.parts[0] == 'dataset-upload' or '__pycache__' in rel.parts or p.is_dir():
            continue
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        if rel.as_posix() in edited:
            shutil.copyfile(V45OFF / rel if rk else V87B / rel if variant == 'ple' else p, out / rel)
        elif rel.as_posix() in ('configs/comparison.json', 'MANIFEST.json'):
            shutil.copyfile(p, out / rel)
        else:
            os.link(p, out / rel)  # hardlinks: never edit these in place
    path = out / 'src/arc2/l4_serving.py'
    if rk:
        for rel in NEW:
            shutil.copyfile(V45OFF / rel, out / rel)
        text = path.read_text()
        for old, new in EDITS:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
        path.write_text(text)
    elif variant in ('dbg', 'dbg2', 'dbg3', 'dbg4', 'dbg5', 'dbg6', 'dbg7', 'dbg8', 'dbg9', 'eag'):
        text = path.read_text()
        assert text.count(GUARD) == 1
        extra = '        env["ARC2_L4_FLUSH_EVERY"] = "20"  # l4 debug: forced cache release (scripts/l4/flush_hook.py)\n'
        if variant == 'dbg4':
            extra += '        env["ARC2_L4_DRAFT_EAGER"] = "1"  # l4 debug bisect: the draft without CUDA graphs\n'
        if variant == 'eag':  # the draft without CUDA graphs only (a read, not a debug run)
            extra = '        env["ARC2_L4_DRAFT_EAGER"] = "1"  # l4: the draft without CUDA graphs (scripts/l4/flush_hook.py)\n'
            path.write_text(text.replace(GUARD, GUARD + extra))
        elif variant in ('dbg5', 'dbg6', 'dbg7', 'dbg8', 'dbg9'):  # the stale-pointer scan only: no blocking, no forced release
            extra = '        env["ARC2_L4_STALE_SCAN"] = "1"  # l4 debug: freed tensors baked into CUDA graphs\n'
            if variant in ('dbg7', 'dbg8'):
                extra += '        env["ARC2_L4_GRAPH_DUMP"] = "1"  # l4 debug: draft graph DOT + capture allocations\n'
            path.write_text(text.replace(GUARD, GUARD + extra))
        else:
            path.write_text(text.replace(GUARD, DBG + (extra if variant != 'dbg' else '')))
        if variant != 'dbg':
            logits = out / 'src/arc2/l4_logits.py'
            logits.write_text(logits.read_text() + '\n\n' + (HERE / 'scripts/l4/flush_hook.py').read_text())
    elif variant == 'car':
        shutil.copyfile(HERE / 'src/arc2/l4_comm.py', out / 'src/arc2/l4_comm.py')
        subprocess.run([sys.executable, str(HERE / 'scripts/l4/comm_edits.py'), str(path)], check=True)
    path = out / 'configs/comparison.json'
    study = json.loads(path.read_text()); study['name'] = out.name
    path.write_text(json.dumps(study, indent=2) + '\n')
    sys.path[:0] = [str(HERE / 'src'), str(HERE / 'scripts'), str(HERE / 'scripts/arena')]
    from build_arena import manifest
    idn = json.loads((out / 'MANIFEST.json').read_text())
    commit = subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], text=True).strip()
    idn.update(study=out.name, files=manifest(out), git_commit=commit,
               source=('arc2-kaggle-v49 + ' + {'rk32': 'v45off offload files + reclaim option', 'car': 'custom_ar / nccl_p2p_level options (arc2.l4_comm)',
                                         'dbg': 'CUDA_LAUNCH_BLOCKING=1 (debug)', 'dbg2': 'CUDA_LAUNCH_BLOCKING=1 + cache release every 20 steps (debug)', 'dbg3': 'dbg2 + worker traceback to a file (debug)', 'dbg4': 'dbg3 + draft without CUDA graphs (debug bisect)', 'dbg5': 'stale-pointer scan of captured CUDA graph metadata (debug)', 'dbg6': 'dbg5 + allocations freed during the draft capture, by stack (debug)', 'dbg7': 'dbg6 + draft graph DOT dumps and capture-window allocation addresses (debug)', 'dbg8': 'dbg7, one graph, freed default-pool blocks only, rank 0 (debug)', 'dbg9': 'stale scan with storage-expiry check (debug)', 'eag': 'draft run without CUDA graphs (read)', 'ple': 'v87b expert_prune_runtime.py (PLE prefill bound)'}[variant] + f' (l4 {commit[:8]}, scripts/l4/build_v49x.py)'))
    (out / 'MANIFEST.json').write_text(json.dumps(idn, indent=2) + '\n')
    runpy.run_path(str(out / 'bootstrap.py'))['verify'](out)
    print('built', out)


if __name__ == '__main__':
    main(sys.argv[1])
