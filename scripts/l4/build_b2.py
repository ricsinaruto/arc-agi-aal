"""l4 b2 port: arc2's b2 harness on the v49 competition stack with the 32 GiB host KV tier.
Runtime = dist/arc2-kaggle-v49rk (v49 + host-tier offload files) with the harness files from the arc2 branch at HEAD,
as build_v51.py did (no second implementation: the roster set is arc2's built b2 arm, key for key).
Kernel = v49's notebook with that set, cpu_cache_gib 32, and a c240 save run (the bundle's 120 evaluation tasks x 2,
334 outputs, scored) to the 11.75 h end epoch: a long public preview, never a submission.
usage: build_b2.py ARC2_B2_BUNDLE [--admission-from COMPARISON.json] [--run-seconds N] [--save-seconds N] [--c240]
  (arc2's dist/arc2-rtx-c240-hx-b2-*: its configs/comparison.json arm). --admission-from: that study's arc2.admission
  (arc2.admission CacheGate) carried into the options; l4_serving gets an ``admission`` keyword
  (v49's has none). --run-seconds: the notebook's end epoch, both modes (default 42,300, v49's). --save-seconds: the
  save run's play (default: --run-seconds). --c240: the save run plays c240 (else v49's placeholder save run, as every
  submitted version). Kernel: l4-arc2-b2-c240 with --c240, else arc2-kaggle-l4b2 (a submission candidate)."""
import argparse, difflib, hashlib, importlib.util, json, os, shutil, subprocess, sys, tarfile
from pathlib import Path

ARC2 = Path('./.claude/worktrees/arc2')
HERE = Path(__file__).resolve().parents[2]
DIST = HERE / 'dist'
src, name = DIST / 'arc2-kaggle-v49rk', 'arc2-kaggle-l4b2'
HARNESS = tuple(f'src/arc2/{f}.py' for f in ('loop', 'tasks', 'trial', 'conv', 'multi', 'repl', 'repl_worker',
                                             'admission', 'gpu', 'sandbox', 'worker', 'arckit', 'served', 'novelty'))
TIER = {'cpu_cache_gib': 32}


LIVE = [
    ('PROGRESS_SECONDS = 300', 'PROGRESS_SECONDS = 60'),
    ('''                if e.get('event') != 'reply':
                    continue
''', '''                if e.get('event') == 'context_drain':  # l4 v52 live line: drains (trigger T, size after)
                    seen['drains'] = seen.get('drains', 0) + 1
                    seen['last_drain'] = (e.get('trigger_tokens'), e.get('tokens_after'))
                if e.get('event') == 'park':
                    seen['parks'] = seen.get('parks', 0) + 1
                if e.get('event') == 'reopen':  # entries by pass
                    seen.setdefault('passes', {})[str(e.get('pass'))] = seen.setdefault('passes', {}).get(str(e.get('pass')), 0) + 1
                if e.get('event') != 'reply':
                    continue
                seen['submits'] = seen.get('submits', 0) + (e.get('outcome') == 'submit')
'''),
    ('''            f"recompute {seen.get('computed', 0) / max(1, seen.get('gen', 0)):.2f}x | errors {live.get('request_errors', 0)}")''',
     '''            f"recompute {seen.get('computed', 0) / max(1, seen.get('gen', 0)):.2f}x | errors {live.get('request_errors', 0)} | "
            f"submits {seen.get('submits', 0)} | running {(live.get('engine') or {}).get('running')} waiting {(live.get('engine') or {}).get('waiting')} | "
            f"KV {(live.get('engine') or {}).get('kv_cache_fraction')} | prefix hit 30m {(live.get('engine') or {}).get('prefix_hit_rate')} | "
            f"preemptions {(live.get('engine') or {}).get('preemptions')} | drains {seen.get('drains', 0)} parks {seen.get('parks', 0)} last T/after {seen.get('last_drain')} | "
            f"gate admitted {(live.get('gate') or {}).get('admitted')} inflight {(live.get('gate') or {}).get('inflight')} "
            f"waiting {(live.get('gate') or {}).get('waiting')} reserved {(live.get('gate') or {}).get('reserved')}/{(live.get('gate') or {}).get('capacity')} slots {(live.get('gate') or {}).get('slots')} cuts {(live.get('gate') or {}).get('slot_cuts', 0)}")'''),
    # tok/s over a true 5-min window from the server's per-token counter (finished replies only if unreadable)
    ('''    t0, k0 = seen.get('last', (now, tokens))
    recent = (tokens - k0) / max(1.0, now - t0) if now > t0 else 0.0
    seen['last'] = (now, tokens)
''', '''    def total(body, name):
        return sum(float(x.rsplit(' ', 1)[1]) for x in body.splitlines() if x.startswith(name + '{') or x.startswith(name + ' '))
    try:
        import urllib.request
        body = urllib.request.urlopen('http://127.0.0.1:1234/metrics', timeout=3).read().decode()
        counted, source = total(body, 'vllm:generation_tokens_total'), 'engine'
        server = {k: total(body, 'vllm:' + k) for k in ('prompt_tokens_total', 'prefix_cache_hits_total', 'prefix_cache_queries_total',
                                                        'external_prefix_cache_hits_total', 'spec_decode_num_accepted_tokens_total',
                                                        'spec_decode_num_draft_tokens_total')}
        tier = {x.rsplit(' ', 1)[0].removeprefix('vllm:')[:70]: round(float(x.rsplit(' ', 1)[1]) / (2**30 if 'bytes' in x else 1), 2)
                for x in body.splitlines() if x.startswith('vllm:') and 'offload' in x.split('{')[0] and '_bucket' not in x and '_created' not in x}  # cumulative
    except Exception:
        counted, source, server, tier = tokens, 'replies', {}, {}
    window = [p for p in seen.get('window', []) if p[2] == source and now - p[0] <= 330] + [(now, counted, source, server)]
    seen['window'] = window
    t0, k0, _, s0 = next((p for p in window if now - p[0] <= 300), window[-1])
    recent = (counted - k0) / (now - t0) if now > t0 else 0.0
    d = {k: v - s0.get(k, v) for k, v in server.items()}  # 5-min deltas of the server's counters
    asked = d.get('prefix_cache_queries_total') or 0
    gpu_hit = f"{d['prefix_cache_hits_total'] / asked:.3f}" if asked else '?'
    cpu_hit = f"{d['external_prefix_cache_hits_total'] / asked:.3f}" if asked else '?'  # CPU-tier hits as a share of all queried tokens
    both_hit = f"{(d['prefix_cache_hits_total'] + d['external_prefix_cache_hits_total']) / asked:.3f}" if asked else '?'
    redo = (f"{(d['prompt_tokens_total'] - d['prefix_cache_hits_total'] - d['external_prefix_cache_hits_total']) / (counted - k0):.2f}x"
            if server and counted > k0 else '?')
    drafted = d.get('spec_decode_num_draft_tokens_total') or 0
    mtp = f"{d['spec_decode_num_accepted_tokens_total'] / drafted:.3f}" if drafted else '?'
    if 'kv' not in seen:  # KV between the 60 s lines: the 5-min max
        import threading
        seen['kv'] = []
        def sample():
            while True:
                try:
                    text = urllib.request.urlopen('http://127.0.0.1:1234/metrics', timeout=3).read().decode()
                    seen['kv'].append((time.time(), max(float(x.rsplit(' ', 1)[1]) for x in text.splitlines() if x.startswith('vllm:kv_cache_usage_perc'))))
                    del seen['kv'][:-40]
                except Exception:
                    pass
                time.sleep(10)
        threading.Thread(target=sample, daemon=True).start()
    kv5 = [v for w, v in list(seen['kv']) if now - w <= 300]
    kv_read = f"{sum(kv5) / len(kv5):.3f} max {max(kv5):.3f}" if kv5 else '?'
    held = seen.setdefault('held', {})  # blocks per task, from the gate's own log
    for path in Path('/tmp/arena').glob('trials/*/admission.jsonl'):
        with path.open() as stream:
            stream.seek(seen.get(str(path), 0))
            for line in stream:
                if not line.endswith('\\n'):
                    break
                seen[str(path)] = seen.get(str(path), 0) + len(line.encode())
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get('event') in ('admit', 'grow'):
                    held[e.get('game')] = e.get('tokens') or 0
                elif e.get('event') == 'release':
                    held.pop(e.get('game'), None)
    per_task = sum(held.values()) / len(held) if held else 0.0
    seen['peak_mean'] = max(seen.get('peak_mean', 0.0), per_task)
    seen['peak_task'] = max(seen.get('peak_task', 0), max(held.values(), default=0))
    rep = seen.setdefault('rep', [])
    rep.append((now, tokens))
    del rep[:-7]
    r0 = next((p for p in rep if now - p[0] <= 300), rep[-1])
    report5 = (tokens - r0[1]) / (now - r0[0]) if now > r0[0] else 0.0
    marks = {'alloc': 'ARC_CACHE_ALLOCATION_FAILURE', 'shm': 'shared memory broadcast', 'rpc': 'timed out'}
    try:  # the server log's new bytes; a restart moves the file aside and starts it again
        with Path('/tmp/arena/serving/vllm-openai-server.log').open('rb') as stream:
            size = stream.seek(0, 2)
            stream.seek(seen.get('log_at', 0) if size >= seen.get('log_at', 0) else 0)
            fresh = stream.read().decode(errors='replace')
            seen['log_at'] = size
        for k, mark in marks.items():
            seen[k] = seen.get(k, 0) + fresh.count(mark)
    except OSError:
        pass
    restarts = sum(p.read_text(errors='replace').count('restart_begin') for p in root.glob('trial-process-*.log'))
    try:
        mem = {x.split(':')[0]: int(x.split()[1]) for x in open('/proc/meminfo')}
        ram = f"{(mem['MemTotal'] - mem['MemAvailable']) / 2**20:.0f}/{mem['MemTotal'] / 2**20:.0f} GiB"
    except Exception:
        ram = '?'
'''),
    ('''f"recompute {seen.get('computed', 0) / max(1, seen.get('gen', 0)):.2f}x | errors''',
     '''f"5min: hit GPU {gpu_hit} CPU-tier {cpu_hit} both {both_hit} recompute {redo} | host RAM {ram} | "
            f"blocks/task now {per_task:.1f} peak mean {seen['peak_mean']:.1f} peak task {seen['peak_task']} | KV 5min {kv_read} | MTP accept {mtp} | report tok/s 5min {report5:.0f} | tier {tier} | entries by pass {seen.get('passes', {})} | "
            f"engine restarts {restarts} | alloc fails {seen.get('alloc', 0)} shm warns {seen.get('shm', 0)} rpc timeouts {seen.get('rpc', 0)} | "
            f"recompute (replies, all) {seen.get('computed', 0) / max(1, seen.get('gen', 0)):.2f}x | errors'''),
    ('''tok/s {recent:.0f} (last 5 min) /''', '''tok/s {recent:.0f} (5 min, {source}) /'''),
    ('''f\"verified programs {seen.get('verified', 0)} | right''', '''f\"right'''),
]
FAIL_FAST = [('''            if short and time.time() - START > PREVIEW_START_LIMIT and not list(Path('/kaggle/working/arc2').glob('arena/live/*.json')):
                status['error'] =''', '''            server_tail = Path('/kaggle/working/arc2/live-logs/vllm-openai-server.tail.log')
            if (short and server_tail.is_file() and not list(Path('/kaggle/working/arc2').glob('arena/live/*.json'))
                    and 'Engine core initialization failed' in server_tail.read_text(errors='replace')):
                if time.time() - seen.setdefault('start_failed', time.time()) >= 240:  # a retry rewrites the log, which clears this
                    status['error'] = 'engine start failed'
                    print(json.dumps({'start_failed': round(time.time() - START)}), flush=True)
                    process.kill()
                    process.wait()
                    break
            else:
                seen.pop('start_failed', None)
            if short and time.time() - START > PREVIEW_START_LIMIT and not list(Path('/kaggle/working/arc2').glob('arena/live/*.json')):
                status['error'] =''')]
STATE_CHARGE = (  # arena.cache.BlockBudget: kept/superseded Mamba states outside req_to_blocks (arc2 30 Oct; L4 v52c: 6-13 orphan blocks per request)
    "        self.rows = layout['costs']\n",
    "        self.rows = layout['costs']\n"
    "        extra = {charge} * sum(1 for g in layout.get('groups') or () if g.get('type') == 'MambaSpec')  # l4: uncharged Mamba states\n"
    "        if extra:\n"
    "            self.rows = [{{**r, 'blocks': r['blocks'] + extra, 'groups': [*r['groups'], extra]}} for r in self.rows]\n")
OLD_OVERLAY = 'arc-expert-int3-refit-arc2c240-k256-v1'
STEPPROF_DRIVER = [
    ('DRIVER_KEYS = ("drill_restart",', 'DRIVER_KEYS = ("stepprof", "drill_restart",'),
    ('        choice.pop("prefetch_threads", None)\n', '        choice.pop("prefetch_threads", None)\n        choice.pop("stepprof", None)\n'),
    ('        budget = args.play_seconds or float("inf")\n',
     '''        if json.loads(args.options or "{}").get("stepprof"):
            stage("stepprof")
            run_logged([python, "-u", "-m", "arc2.l4_stepprof", "--study", str(study_path), "--server-dir", str(serving),
                        "--output", str(out / "stepprof")], env, out / "stepprof-process.log", 1700)
            import threading
            from . import l4_stepprof
            threading.Thread(target=l4_stepprof.in_play, args=(str(serving), str(out / "stepprof"), 200), daemon=True).start()
        budget = args.play_seconds or float("inf")
'''),
]
STEADY_DRIVER = [
    ('DRIVER_KEYS = ("drill_restart",', 'DRIVER_KEYS = ("steady", "drill_restart",'),
    ('        choice.pop("prefetch_threads", None)\n', '        choice.pop("prefetch_threads", None)\n        choice.pop("steady", None)\n'),
    ('            trial.pop("curve_step", None)\n',
     '            trial.pop("curve_step", None)\n            trial["steady"] = json.loads(args.options)["steady"]\n'),
]
STEADY_TRIAL = ('    served_before = await generated_tokens(client)\n',
                '''    served_before = await generated_tokens(client)
    if trial.get("steady"):  # scripts/l4/l4_steady_src.py: steady-state recorded traffic in place of the play
        from .l4_steady import run_steady

        live = os.environ.get("ARC_VALIDATION_OUTPUT")
        wall = float(trial["wall_seconds"])  # the whole budget: no drain share in a test with no answers to keep
        transport.deadline = started + wall
        report = await run_steady(trial, source, output, transport, gate, arms[0], client, slots=int(study["generation_slots"]),
                                  window=window, started=started, wall=wall, live=Path(live) / "arena" / "live" if live else None,
                                  engine_metrics=engine_metrics)
        await client.aclose()
        admission.close()
        events.close()
        return report
''')
PEAK_DRIVER = ('    def stage(name):\n        record.update(stage=name',
               '''    try:  # every card's memory from the first second: a failed start says by how much
        subprocess.Popen(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits", "-lms", "1000",
                          "-f", str(out / "gpu-mem.csv")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        pass

    def stage(name):
        record.update(stage=name''')
PEAK_LIVE = ('''f"5min: hit GPU {gpu_hit}''',
             '''f"card peak MiB {peak_mem(root)} | 5min: hit GPU {gpu_hit}''')
PEAK_HELPER = ('''    lives = sorted(root.glob('arena/live/*.json'))
    if not lives:
        return f"[{stamp} elapsed] stage {run.get('stage') or 'starting'} (serving start / probe; no play yet)"
''',
               '''    def peak_mem(where):
        top = {}
        try:
            for line in (where / 'gpu-mem.csv').read_text(errors='replace').splitlines():
                card, used = line.split(',')[:2]
                top[card.strip()] = max(top.get(card.strip(), 0), int(used))
        except (OSError, ValueError):
            pass
        return top
    lives = sorted(root.glob('arena/live/*.json'))
    if not lives:
        return f"[{stamp} elapsed] stage {run.get('stage') or 'starting'} (serving start / probe; no play yet) | card peak MiB {peak_mem(root)}"
''')
BPS_FIT = '''    fit = max(24, int(capacity / {bps}))  # l4 v56: streams by the formula on the pool this boot measured
    if fit < int(study["generation_slots"]):
        study.update(generation_slots=fit, active_limit=fit)
    print(json.dumps({{"stream_fit": fit, "gate_blocks": capacity, "blocks_per_stream": {bps}}}), flush=True)
'''
SLOT_CUT = [('        self.capacity = max(1, self.base - cut)\n',
             '        self.guard(engine, now)\n        self.capacity = max(1, self.base - cut)\n'),
            ('    def snapshot(self):\n        return {**super().snapshot(), "admission": "overhang"',
             '''    def guard(self, engine, now):
        """l4 v56: a 6 h run nobody can tune. A preemption, or KV at HIGH or over for SUSTAIN s, takes STEP slots and
        SHARE of the base from the gate, at most once in COOL s, down to FLOOR slots; nothing goes back up."""
        HIGH, SUSTAIN, STEP, SHARE, COOL, FLOOR = 0.93, 60.0, 2, 0.03, 300.0, 26
        guard = self.__dict__.setdefault("guarded", {"preemptions": None, "high_since": None, "pending": None, "next": 0.0, "cuts": 0})
        seen, count = guard["preemptions"], engine.get("preemptions")
        if count is not None:
            guard["preemptions"] = count
            if seen is not None and count > seen:
                guard["pending"] = "preemption"
        if float(engine["kv_cache_fraction"]) < HIGH:
            guard["high_since"] = None
        elif guard["high_since"] is None:
            guard["high_since"] = now
        if guard["high_since"] is not None and now - guard["high_since"] >= SUSTAIN:
            guard["pending"] = guard["pending"] or "kv"
        if guard["pending"] and now >= guard["next"] and self.slots > FLOOR:
            self.slots, self.base = max(FLOOR, self.slots - STEP), int(self.base * (1 - SHARE))
            guard.update(pending=None, next=now + COOL, cuts=guard["cuts"] + 1)
            self.stats["slot_cuts"] = guard["cuts"]
            if self.log:
                self.log({"event": "slot_cut", "slots": self.slots, "base": self.base,
                          "kv": engine["kv_cache_fraction"], "preemptions": count})

    def snapshot(self):
        return {**super().snapshot(), "admission": "overhang"''')]
PLE_QUEUE = ('''            record = l4_ple_pack.apply(''', '''            from arc2 import ple_queue_patch  # the offload connector waits for its staging thread, as on BF16 (base15bf)

            ple_queue_patch.apply(recipe.RUNTIME_ROOT / "usr/local/lib/python3.12/dist-packages", working, 120.0)
            record = l4_ple_pack.apply(''')
V9B = Path('./.claude/worktrees/v84sub/dist/arena-flash-v9b-prev9h/src')  # the AR + heal overlay path, Kaggle-proven on RTX
AR_SERVING = [
    ("if research_checkpoint is not None and (expert_overlay is not None or expert_prune is not None):",
     "if research_checkpoint is not None and expert_overlay is not None:"),
    ("result['research_checkpoint'] = configure_checkpoint(recipe, working, research_checkpoint)",
     "result['research_checkpoint'] = configure_checkpoint(recipe, working, research_checkpoint,\n                                                             expert_prune=expert_prune)"),
    ("    command = recipe.server_command\n",
     '''    _arc_env = getattr(recipe, 'runtime_environment', None)
    if _arc_env is not None:
        # v9b: FP8 PLE table on the AR base, after the recipe's PLE patch
        def _arc_ple_fp8_environment(*args, **kwargs):
            out = _arc_env(*args, **kwargs)
            ple = recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py'
            text = ple.read_text()
            anchor = '    \\"\\"\\"Select global-scale FP8 only for quantized PLE checkpoint shards.\\"\\"\\"\\n'
            if 'ARC_PLE_FP8' not in text:
                if text.count(anchor) != 1:
                    raise RuntimeError('ple_layer.py: FP8 PLE anchor not found once')
                ple.write_text(text.replace(anchor, anchor + "    if __import__('os').environ.get('ARC_PLE_FP8') == '1' and "
                                            "prefix.endswith('ple_embedding.ngram_embedding'):\\n"
                                            "        return Qwen3_8FlashNextPLEFp8EmbeddingMethod()\\n"))
            return out

        recipe.runtime_environment = _arc_ple_fp8_environment

    command = recipe.server_command
'''),
]
V9C = Path('./.claude/worktrees/v84sub/dist/b2-v9c-p75c893-kf-tr15x10-24h-v2/src/arc_infra')
# p50d7: a depth-cut mask (drop_layers) serves only on arc3sub's mkP50d7 files: the cut farm config, the
# drop-capable prune patch/runtime and the NVFP4 gate on the checkpoint's layer types
DEPTH_SRC = Path('./.claude/worktrees/v84sub/dist/b2-p50d7tc942-ex5x10-v1/src')
DEPTH_FILES = ('arena/expert_overlay.py', 'arc_infra/expert_prune_patch.py', 'arc_infra/expert_prune_runtime.py')
DEPTH_SERVING = ("                    expert_prune_sha256=expert_prune_sha256 if expert_prune is not None else None)\n",
                 "                    expert_prune_sha256=expert_prune_sha256 if expert_prune is not None else None,\n"
                 "                    drop_layers=(expert_prune or {}).get('drop_layers'))\n")

DEPTH_GATE = ("""               '            getattr(config, "arc_unpruned_" + field, getattr(config, field, None)) != expected\\n', 1)],\n""",
              """               '            getattr(config, "arc_unpruned_" + field, getattr(config, field, None)) != expected\\n', 1),\n"""
              """              ('        if tuple(getattr(config, "layer_types", ())) != _RADIXARK_NVFP4_LAYER_TYPES:\\n',\n"""
              """               '        # ARC (depth cut): the checkpoint\\'s list.\\n'\n"""
              """               '        if tuple(getattr(config, "arc_unpruned_layer_types", getattr(config, "layer_types", ())))'\n"""
              """               ' != _RADIXARK_NVFP4_LAYER_TYPES:\\n', 1)],\n""")


def depth_cut(out):
    """The mkP50d7 port onto this tree (after kernels() and the overlay edit of serving.py)."""
    for rel in DEPTH_FILES:
        (out / 'src' / rel).unlink()
        shutil.copyfile(DEPTH_SRC / rel, out / 'src' / rel)
    path = out / 'src/arena/serving.py'; text = path.read_text(); path.unlink()
    assert text.count(DEPTH_SERVING[0]) == 1
    path.write_text(text.replace(*DEPTH_SERVING))
    # the exact NVFP4-PLE FP8 gate (vLLM ple_layer.py) pins 48 layer types: read the checkpoint's list (mkP50d7's edit)
    path = out / 'src/arc_infra/serving_kernels_patch.py'; text = path.read_text(); path.unlink()
    assert text.count(DEPTH_GATE[0]) == 1, 'NVFP4 PLE gate edit anchor'
    path.write_text(text.replace(*DEPTH_GATE))

KERNEL_FILES = ('serving_kernels_patch.py', 'qsa_fast_runtime.py', 'stack_kernels_patch.py', 'gdn_defer_runtime.py', 'gdn_defer.cu')


def kernels(out, chosen):
    """v55b: V9C's QSA / GDN files on L4: sm_89 for the deferred GDN library, the flags in l4_study."""
    assert chosen <= {'qsa', 'gdn', 'h2d'}, chosen
    for name in KERNEL_FILES:
        dst = out / 'src/arc_infra' / name
        dst.unlink(missing_ok=True)
        shutil.copyfile(V9C / name, dst)
    stack = out / 'src/arc_infra/stack_kernels_patch.py'
    text = stack.read_text(); stack.unlink()
    old = "'-gencode=arch=compute_120,code=sm_120'"
    assert text.count(old) == 1
    stack.write_text(text.replace(old, "'-gencode=arch=compute_89,code=sm_89'"))
    serving = out / 'src/arc2/l4_serving.py'
    text = serving.read_text(); serving.unlink()
    old = '**({"indexer_fp8": True} if indexer_fp8 else {})})\n'
    assert text.count(old) == 1
    flag = '**({"indexer_fp8": True} if indexer_fp8 else {})' + (', "qsa_fast": True' if 'qsa' in chosen else '') + '})\n'
    pick = {'async_h2d': 'h2d' in chosen, 'gdn_defer': 'gdn' in chosen}
    if any(pick.values()):
        flag += f'    server["stack_kernels"] = {pick!r}\n'
    serving.write_text(text.replace(old, flag))


DRAFT_EDITS = (  # arc2 8588e99e: draft 0..3 on L4
    ('    if draft not in (0, 1):\n        raise ValueError("draft is 0 or 1")\n',
     '    if draft not in (0, 1, 2, 3):\n        raise ValueError("draft is 0..3")\n'),
    ('    rows = streams * (draft + 1)  # a draft of one decodes two positions a stream in a step\n',
     '    rows = streams * (draft + 1)  # a draft of k decodes k + 1 positions a stream in a step\n'
     '    if rows > 256:\n        raise ValueError("streams x (draft + 1) is at most 256 (the arena\'s graph capture limit)")\n'),
    ('(1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160)\n', '(1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 224)\n'),
    ('PREFIX_KEYS_LEFT_OUT[2 * draft:]', 'PREFIX_KEYS_LEFT_OUT[2 * bool(draft):]'),
    ('            if argv[at] != str(KV4_BLOCK_SERVED):\n', '            if int(argv[at]) not in (3136, 3200, 3264):  # kv4_block_tokens at MTP 0-1, 2, 3\n'),
)
SAVE_TRIALS = [  # every 5 min the trial's logs (events, tasks, served-*) to the output: a cancelled session keeps them
    ("""    lives = sorted(root.glob('arena/live/*.json'))
""", """    if now - seen.get('saved_at', 0) >= 300:
        seen['saved_at'] = now
        import shutil
        for src in Path('/tmp/arena').glob('trials/*/*'):
            if src.is_file() and (src.suffix == '.jsonl' or src.name == 'served-meta.json'):
                dst = root / 'saved-trials' / src.parent.name / src.name
                try:
                    if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(src, dst.with_name(dst.name + '.part'))
                        os.replace(dst.with_name(dst.name + '.part'), dst)
                except OSError:
                    pass
    lives = sorted(root.glob('arena/live/*.json'))
"""),
]
L4_ADMISSION = [  # l4_serving.l4_study: an ``admission`` keyword into the study's arc2 block (arc2.trial reads it)
    ('warm_rht=False, context=None, cpu_cache_gib=0, reclaim=False):',
     'warm_rht=False, context=None, cpu_cache_gib=0, reclaim=False, admission=None):'),
    ('    out.update(generation_slots=slots, active_limit=slots, windows=[int(window or served)], name=out["name"] + "-l4x4")\n',
     '    out.update(generation_slots=slots, active_limit=slots, windows=[int(window or served)], name=out["name"] + "-l4x4")\n'
     ' if admission:  # l4 arm: arc2.admission CacheGate (real KV use), not the reserving gate\n'
     '        out["arc2"]["admission"] = dict(admission)\n'),
    ('"block_tokens", "warmup_rows", "warmup_dump", "warm_rht", "context", "cpu_cache_gib", "reclaim"):',
     '"block_tokens", "warmup_rows", "warmup_dump", "warm_rht", "context", "cpu_cache_gib", "reclaim", "admission"):'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref'); ap.add_argument('--admission-from'); ap.add_argument('--run-seconds', type=int, default=42300)
    ap.add_argument('--save-seconds', type=int); ap.add_argument('--c240', action='store_true')
    ap.add_argument('--c120', action='store_true', help='the save run plays the 120 evaluation tasks once (167 outputs)')
    ap.add_argument('--fail-fast', action='store_true', help='a failed engine start ends the kernel 4 min after it, not at the start limit')
    ap.add_argument('--arm-set', help='JSON merged into the ref arm (e.g. the drain block)')
    ap.add_argument('--context', type=int, default=131072, help='served context; non-v49 values patch l4_serving CONTEXTS')
    ap.add_argument('--window', type=int, help='harness window (default: the context)')
    ap.add_argument('--reply-reserve', type=int, help="the gate's reply reserve (v9: measured p95 reply)")
    ap.add_argument('--live', action='store_true', help='kernel log every minute with engine, gate and drain metrics')
    ap.add_argument('--slots', type=int, help='fixed generation slots (formula) in place of the probe fit; patches kaggle.py')
    ap.add_argument('--state-charge', type=int, default=0,
                    help='gate blocks added per Mamba group per request (arena.cache.BlockBudget; arc2: 2)')
    ap.add_argument('--kv-margin', type=float, help="the gate's share of the served blocks (study kv_margin; v49: 0.96)")
    ap.add_argument('--kernels', default='', help='comma list of qsa, gdn, h2d: arc3sub kern-qsa / kern-gdn files from the V9C bundle, patched into l4_study')
    ap.add_argument('--draft', type=int, default=0, help='MTP tokens 0..3 (l4_serving draft, arc2 8588e99e); with --streams for the graph sizes')
    ap.add_argument('--streams', type=int, help='server streams (max_num_seqs); 64 by default')
    ap.add_argument('--int8', choices=('g64', 'v9c'), help="dense INT8 on V9C's four families: Humming g64 or V9C's per-channel Triton")
    ap.add_argument('--api-servers', type=int, help='API server processes (reply_ids records are per process: 1)')
    ap.add_argument('--grow-wait', type=float, help='seconds a task waits to grow with its blocks held')
    ap.add_argument('--utilization', help='gpu memory utilization, e.g. 0.95')
    ap.add_argument('--cotenant', action='store_true', help='state_cotenant kept at MTP > 0 (L4 left it out)')
    ap.add_argument('--ple-queue', action='store_true', help="arc2's ple_queue_patch on the offload connector, 120 s")
    ap.add_argument('--steady', help='conversation snapshot (steady_snapshot.py): steady traffic in place of the play')
    ap.add_argument('--pp2', action='store_true', help='l4pp: PP 2 + TP 2 + EP, MTP 0..3 (pp_edits.py)')
    ap.add_argument('--pp-layers', help='with --pp2: decoder layers per stage, "a,b" summing to 48 less a mask\'s drop_layers (default even); the draft sits on stage 2')
    ap.add_argument('--slot-cut', action='store_true', help='OverhangGate + one-way slot cut on a preemption or KV >= .93 (v56)')
    ap.add_argument('--bps', type=float, help='slots = gate capacity / this, at most --slots, read from the booted pool (v56 PP2TP2)')
    ap.add_argument('--play-early', type=int, help='kaggle.py EARLY_SECONDS: a play that ends inside the trial end drain is not an engine death')
    ap.add_argument('--stepprof', action='store_true', help='step profile between the probe and the play (l4_stepprof_src.py)')
    ap.add_argument('--ar-overlay', help="heal overlay dataset (AR 4-bit experts + BF16 dense, trained format): the Intel AR base on Marlin, v9b's research_checkpoint path")
    ap.add_argument('--ar-sha', help='sha256 of its overlay.json')
    ap.add_argument('--overlay', help='expert + dense overlay dataset name (v55b); default v49/v52 k256 refit')
    ap.add_argument('--overlay-sha', help='sha256 of its overlay-manifest.json')
    ap.add_argument('--mask', help="the overlay's expert-prune json")
    ap.add_argument('--fp4-layers', help="dn4g8 overlays: ARC_EXPERT_INT3_FP4_LAYERS; ships V9C's expert_int3_runtime")
    ap.add_argument('--engine-fixes', action='store_true', help="v9's no-give-up, OOM order, task re-entry (engine_fixes.py)")
    ap.add_argument('--tier', type=int, default=32, help='cpu_cache_gib; 0: v49 serving exactly (base v49, no tier key)')
    ap.add_argument('--name', default=name, help='bundle / kernel name (default arc2-kaggle-l4b2)')
    ap.add_argument('--harness-from-ref', action='store_true',
                    help="harness bytes from the ref bundle (the RTX run's own), not arc2 HEAD")
    a = ap.parse_args()
    assert a.pp2 or not a.pp_layers, '--pp-layers needs --pp2'
    ref = Path(a.ref)
    gate = json.loads(Path(a.admission_from).read_text())['arc2']['admission'] if a.admission_from else None
    arm = dict(json.loads((ref / 'configs/comparison.json').read_text())['trials'][0]['arms'][0])
    arm.pop('name')
    arm = {**arm, **(json.loads(a.arm_set) if a.arm_set else {})}
    out_name = a.name
    files = tuple(h for h in HARNESS if (ref / h).is_file())  # multi.py only from b2 on
    source = ref if a.harness_from_ref else ARC2
    dirty = subprocess.run(['git', '-C', str(ARC2), 'status', '--porcelain', '--', *HARNESS], capture_output=True, text=True)
    assert a.harness_from_ref or dirty.stdout == '', 'arc2 harness files uncommitted: ' + dirty.stdout
    commit = subprocess.run(['git', '-C', str(ARC2), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    for rel in files:  # the RTX bundle and the harness source run the same bytes
        assert (ref / rel).read_bytes() == (source / rel).read_bytes(), f'{rel}: {ref.name} differs from arc2 HEAD'
    base = src if a.tier else ARC2 / 'dist/arc2-kaggle-v49'
    tier = {'cpu_cache_gib': a.tier} if a.tier else {}
    out = DIST / out_name; shutil.rmtree(out, ignore_errors=True); out.mkdir()
    for p in base.rglob('*'):
        rel = p.relative_to(base)
        if rel.parts[0] == 'dataset-upload' or '__pycache__' in rel.parts or p.is_dir():
            continue
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        if rel.as_posix() in files:
            shutil.copyfile(source / rel, out / rel)
        elif rel.as_posix() in ('configs/comparison.json', 'MANIFEST.json'):
            shutil.copyfile(p, out / rel)
        else:
            os.link(p, out / rel)
    for rel in files:
        if not (out / rel).exists():
            shutil.copyfile(source / rel, out / rel)
    if a.slots:  # v49's driver overwrites slots with the probe's second-turn fit - 2; a preset slot count wins
        drv = out / 'src/arc2/kaggle.py'
        text = drv.read_text(); drv.unlink()
        old = '                    choice = {**choice, "slots": max(8, min(int(choice["streams"]), fit - 2))}\n'
        new = ('                    choice = {**choice, "slots": int(choice["slots"]) if choice.get("slots") else '
               'max(8, min(int(choice["streams"]), fit - 2))}  # l4 v52: fixed slots (formula) over the probe fit\n')
        assert text.count(old) == 1
        drv.write_text(text.replace(old, new))
    if a.context != 131072:
        serving = out / 'src/arc2/l4_serving.py'
        text = serving.read_text(); serving.unlink()
        old = 'CONTEXTS = (65536, 98304, 131072)'
        assert text.count(old) == 1
        serving.write_text(text.replace(old, f'CONTEXTS = (65536, 98304, 131072, {a.context})  # l4 v52: v9 context'))
    if gate:
        serving = out / 'src/arc2/l4_serving.py'
        text = serving.read_text(); serving.unlink()  # a hard link to v49rk's bytes: write a new file
        for old, new in L4_ADMISSION:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
        serving.write_text(text)
    if a.state_charge:
        cache = out / 'src/arena/cache.py'; text = cache.read_text(); cache.unlink()
        old, new = STATE_CHARGE
        assert text.count(old) == 1
        cache.write_text(text.replace(old, new.format(charge=int(a.state_charge))))
    if a.kernels:
        kernels(out, set(a.kernels.split(',')))
    if a.draft:
        serving = out / 'src/arc2/l4_serving.py'
        text = serving.read_text(); serving.unlink()
        for old, new in DRAFT_EDITS:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
        serving.write_text(text)
    if a.int8 == 'v9c':
        shutil.copyfile(Path(__file__).with_name('l4_int8v9c_src.py'), out / 'src/arc2/l4_int8v9c.py')
        (out / 'src/arc_infra/dense_kernels_runtime.py').unlink()
        shutil.copyfile(V9C / 'dense_kernels_runtime.py', out / 'src/arc_infra/dense_kernels_runtime.py')
        dense = out / 'src/arc2/l4_dense.py'
        text = dense.read_text(); dense.unlink()
        old = '''            with torch.no_grad():
                report = convert(torch, self.model_runner.get_model(), mode, progress=progress,
                                 families=setting.get("families"))
'''
        new = '''            from . import l4_int8v9c

            with torch.no_grad():
                report = convert(torch, self.model_runner.get_model(), mode, progress=progress,
                                 families=setting.get("families"), tools=(l4_int8v9c.quantize, l4_int8v9c.forward))
            l4_int8v9c.warm()
'''
        assert text.count(old) == 1
        dense.write_text(text.replace(old, new))
    if a.fp4_layers:
        (out / 'src/arc_infra/expert_int3_runtime.py').unlink()
        shutil.copyfile(V9C / 'expert_int3_runtime.py', out / 'src/arc_infra/expert_int3_runtime.py')
        serving = out / 'src/arc2/l4_serving.py'
        text = serving.read_text(); serving.unlink()
        old = '            env.setdefault(INT3_ENV, json.dumps({"group": int(serving_kernels["expert_int3"])}, sort_keys=True))\n'
        assert text.count(old) == 1
        serving.write_text(text.replace(old, old + f'            env.setdefault("ARC_EXPERT_INT3_FP4_DOWN", "1")\n            env.setdefault("ARC_EXPERT_INT3_FP4_LAYERS", {a.fp4_layers!r})\n'))
    if a.engine_fixes:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import engine_fixes
        engine_fixes.apply(out)
    serving = out / 'src/arc2/l4_serving.py'  # keys arc2 reads by settings.get, outside loop.DEFAULTS (base13b/14)
    text = serving.read_text(); serving.unlink(); old = 'unknown = set(roster["set"]) - set(DEFAULTS)\n'
    assert old in text
    serving.write_text(text.replace(old, 'unknown = set(roster["set"]) - set(DEFAULTS) - {"loop_exit", "no_first_line", "pass_clean", "submit_two", "idle_end", "idle_tokens", "one_answer", "park_drain", "reopen_wrong", "second_differs", "submit_final", "fresh_tries", "fresh_interp", "pass_clean", "differs_first", "pick", "idle_any", "reopen_picture", "sandbox_fix"}\n'))
    if a.stepprof:
        shutil.copyfile(Path(__file__).with_name('l4_stepprof_src.py'), out / 'src/arc2/l4_stepprof.py')
        drv = out / 'src/arc2/kaggle.py'; text = drv.read_text(); drv.unlink()
        for old, new in STEPPROF_DRIVER:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
        drv.write_text(text)
        serving = out / 'src/arc2/l4_serving.py'; text = serving.read_text(); serving.unlink()
        old = '**({"profile_rows": [16, 24, 32]} if profile else {}),'
        assert text.count(old) == 1
        serving.write_text(text.replace(old, ''))
    if a.steady:
        shutil.copyfile(Path(__file__).with_name('l4_steady_src.py'), out / 'src/arc2/l4_steady.py')
        shutil.copyfile(a.steady, out / 'configs/l4-steady.json.gz')
        drv = out / 'src/arc2/kaggle.py'; text = drv.read_text(); drv.unlink()
        for old, new in [*STEADY_DRIVER, PEAK_DRIVER]:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
        drv.write_text(text)
        tr = out / 'src/arc2/trial.py'; text = tr.read_text(); tr.unlink()
        assert text.count(STEADY_TRIAL[0]) == 1
        tr.write_text(text.replace(*STEADY_TRIAL))
    if a.cotenant or a.ple_queue:
        serving = out / 'src/arc2/l4_serving.py'; text = serving.read_text(); serving.unlink()
        if a.cotenant:
            old = 'PREFIX_KEYS_LEFT_OUT = ("spec_decode", "tail_spec_acceptance", "state_cotenant", "admission_tick")'
            assert text.count(old) == 1
            text = text.replace(old, 'PREFIX_KEYS_LEFT_OUT = ("spec_decode", "tail_spec_acceptance", "admission_tick")')
        if a.ple_queue:
            assert text.count(PLE_QUEUE[0]) == 1
            text = text.replace(*PLE_QUEUE)
            shutil.copyfile(ARC2 / 'src/arc2/ple_queue_patch.py', out / 'src/arc2/ple_queue_patch.py')
        serving.write_text(text)
    drop = (json.loads(Path(a.mask).read_text()).get('drop_layers') or []) if a.mask else []
    if a.pp2:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import pp_edits
        pp_edits.apply(out, a.pp_layers, 48 - len(drop))
    path = out / 'configs/comparison.json'; study = json.loads(path.read_text()); study['name'] = out_name
    if a.ar_overlay:
        for rel in ('arena/research_checkpoint.py', 'arena/ar_slim.py', 'arena/models/intel-autoround-w4a16.json',
                    'arc_infra/expert_prune_patch.py', 'arc_infra/expert_prune_runtime.py'):
            (out / 'src' / rel).unlink(missing_ok=True)
            shutil.copyfile(V9B / rel, out / 'src' / rel)
        arena = out / 'src/arena/serving.py'; text = arena.read_text(); arena.unlink()
        for old, new in AR_SERVING:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
        arena.write_text(text)
        serving = out / 'src/arc2/l4_serving.py'; text = serving.read_text(); serving.unlink()
        for old, new in (('"moe_backend": "humming", "expert_int3": 16,', '"moe_backend": "marlin",'),
                         ('SERVER_ENV = {"ARENA_GPU_MEMORY_UTILIZATION": "0.92",', 'SERVER_ENV = {"ARC_PLE_FP8": "1", "ARENA_GPU_MEMORY_UTILIZATION": "0.92",')):
            assert text.count(old) == 1, old
            text = text.replace(old, new)
        serving.write_text(text)
        mask = Path(a.mask)
        shutil.copyfile(mask, out / 'configs/expert-prune' / mask.name)
        study['server']['expert_prune'] = {'mask': f'configs/expert-prune/{mask.name}', 'sha256': hashlib.sha256(mask.read_bytes()).hexdigest()}
        study['server'].pop('expert_overlay', None); study['server'].pop('draft_head', None)
        study['research_checkpoint'] = {**json.loads((V9B.parent / 'configs/comparison.json').read_text())['research_checkpoint'],
                                        'heal_overlay': f'/kaggle/input/datasets/richardcsaky/{a.ar_overlay}', 'heal_reload': False}
        study['arc2']['weights'] = {'name': a.ar_overlay, 'replaced': OLD_OVERLAY}
    if a.overlay:
        mask = Path(a.mask)
        shutil.copyfile(mask, out / 'configs/expert-prune' / mask.name)
        study['server']['expert_prune'] = {'mask': f'configs/expert-prune/{mask.name}', 'sha256': hashlib.sha256(mask.read_bytes()).hexdigest()}
        study['server']['expert_overlay'] = {'dataset': 'richardcsaky/' + a.overlay, 'manifest_sha256': a.overlay_sha, 'check': 'size'}
        study['server'].pop('draft_head', None)  # the overlay's dense shards carry the student's own MTP head (as V9C)
        arena = out / 'src/arena/serving.py'; text = arena.read_text(); arena.unlink()
        old = "Path(working) / 'expert-overlay-checkpoint', expert_overlay['manifest_sha256'],"
        assert text.count(old) == 1
        arena.write_text(text.replace(old, "Path(working) / 'expert-overlay-checkpoint', os.environ.get('ARC2_OVERLAY_SHA') or expert_overlay['manifest_sha256'],"))
        study['arc2']['weights'] = {'name': a.overlay, 'replaced': OLD_OVERLAY}
        if drop:
            assert not a.int8, 'p50d7: the exact weights, BF16 dense, no requant'
            depth_cut(out)
            study['arc2']['weights']['drop_layers'] = drop
    if a.kv_margin:
        study['kv_margin'] = a.kv_margin
    if a.bps:
        trial = out / 'src/arc2/trial.py'; text = trial.read_text(); trial.unlink()
        old = '    capacity = budget.capacity\n'
        assert text.count(old) == 1
        trial.write_text(text.replace(old, old + BPS_FIT.format(bps=a.bps)))
    if a.play_early:
        driver = out / 'src/arc2/kaggle.py'; text = driver.read_text(); driver.unlink()
        old = 'EARLY_SECONDS = 180 '
        assert text.count(old) == 1
        driver.write_text(text.replace(old, f'EARLY_SECONDS = {a.play_early} '))
    if a.slot_cut:
        study['arc2']['gate_overhang'] = {'window': 60.0, 'poll': 5.0}
        admission_py = out / 'src/arc2/admission.py'; text = admission_py.read_text(); admission_py.unlink()
        for old, new in SLOT_CUT:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
        admission_py.write_text(text)
    path.write_text(json.dumps(study, indent=2) + '\n')
    sys.path[:0] = [str(ARC2 / 'src'), str(ARC2 / 'scripts'), str(ARC2 / 'scripts/arena')]
    from build_arena import manifest
    idn = json.loads((out / 'MANIFEST.json').read_text())
    idn.update(study=out_name, files=manifest(out))
    idn['source'] = (f'{base.name} bundle + ' + '%s from arc2 %s, arm of %s (scripts/l4/build_b2.py)'
                     % (', '.join(Path(h).name for h in files), (ref.name + "'s") if a.harness_from_ref else commit[:8], ref.name))
    (out / 'MANIFEST.json').write_text(json.dumps(idn, indent=2) + '\n')
    __import__('runpy').run_path(str(out / 'bootstrap.py'))['verify'](out)

    template = ARC2 / 'dist/arc2-kaggle-v49-kernel'
    nb = json.loads((template / 'submission.ipynb').read_text()); old = ''.join(nb['cells'][1]['source'])
    i = old.index("'--options', '") + len("'--options', '"); j = old.index("'", i)
    opts = json.loads(old[i:j])
    assert opts['roster']['set']['cap'] == arm['cap'] == 16384 and opts['context'] == 131072, (opts, arm)
    opts = {**opts, 'context': a.context, **({'window': a.window} if a.window else {}),
            **({'reply_reserve': a.reply_reserve} if a.reply_reserve else {})}
    new = {**opts, 'roster': {**opts['roster'], 'set': arm}, **({'draft': a.draft} if a.draft else {}), **({'dense_int8': ['attention_qkv_gate', 'gdn_qkvz', 'mixer_out', 'output_head']} if a.int8 else {}), **({'streams': a.streams} if a.streams else {}), **tier, **({'slots': a.slots} if a.slots else {}), **({'admission': gate} if gate else {})}
    if a.stepprof:
        new.update(profile=True, stepprof=True, engine_watchdog={'test_kill_seconds': 420})  # the reload drill in the play
    if a.api_servers:
        new['api_servers'] = a.api_servers
    if a.grow_wait is not None:
        new['grow_wait'] = a.grow_wait
    if a.utilization:
        new['utilization'] = a.utilization
    if a.steady:
        new['steady'] = {'file': 'configs/l4-steady.json.gz'}
    if a.pp2:
        new['block_tokens'] = None  # the recipe's 3,136 (3,200 at MTP 2): a GDN state on one of two ranks
    code = old[:i] + json.dumps(new) + old[j:]

    check = ("import json, sys; from arc2 import kaggle, l4_serving, loop, conv, repl, admission, trial; "
             "o = json.loads(sys.argv[1]); kaggle.check_options(o); s = o['roster']['set']; "
             "bad = sorted(set(s) - set(loop.DEFAULTS) - {'loop_exit', 'no_first_line', 'pass_clean', 'submit_two', 'idle_end', 'idle_tokens', 'one_answer', 'park_drain', 'reopen_wrong', 'second_differs', 'submit_final', 'fresh_tries', 'fresh_interp', 'pass_clean', 'differs_first', 'pick', 'idle_any', 'reopen_picture', 'sandbox_fix'}); assert not bad, bad; assert int(s['cap']) in l4_serving.REPLY_CAPS; "
             "c = {k: v for k, v in {**kaggle.OPTIONS, **o}.items() if k not in kaggle.DRIVER_KEYS}; "
             "st = l4_serving.study_for(json.loads(open(sys.argv[2]).read()), c); x = st['arc2']['l4x4']; a = st['trials'][0]['arms'][0]; "
             "g = o.get('admission'); assert st['arc2'].get('admission') == g and 'admission' not in st['trials'][0]; "
             "g and admission.settings(g); "
             "print(json.dumps({'arm': {k: a.get(k) for k in s}, 'window': st['windows'], 'context': x.get('context'), "
             "'streams': x['streams'], 'slots': st['generation_slots'], 'gate': st['arc2'].get('admission'), 'tier': st['cache_experiment']['cpu_cache_gib'], "
             "'prune': st['server']['expert_prune']['sha256'][:8], 'overlay': (st['server'].get('expert_overlay') or {'dataset': (st.get('research_checkpoint') or {}).get('heal_overlay')})['dataset']}))")
    done = subprocess.run([sys.executable, '-c', check, json.dumps(new), str(out / 'configs/comparison.json')],
                          env={**os.environ, 'PYTHONPATH': str(out / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'},
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    res = json.loads(done.stdout)
    assert res['arm'] == arm and res['window'] == [a.window or a.context] and res['context'] == a.context and res['tier'] == a.tier, res
    assert res['gate'] == gate and res['slots'] == (a.slots or 64), res
    assert res['prune'] == (hashlib.sha256(Path(a.mask).read_bytes()).hexdigest()[:8] if a.overlay or a.ar_overlay else '01514464') and res['overlay'].endswith(a.overlay or a.ar_overlay or OLD_OVERLAY), res
    print('options check:', done.stdout.strip())
    (DIST / f'{out_name}-options.json').write_text(json.dumps(new) + '\n')

    # c240 save run: the bundle's 120 evaluation tasks twice (Modal full_main's roster), scored, to the end epoch
    save = 'SAVE_TASKS, SAVE_SECONDS, RUN_SECONDS = 240, 1800, 42300'
    assert code.count(save) == 1
    code = code.replace(save, f'SAVE_TASKS, SAVE_SECONDS, RUN_SECONDS = {120 if a.c120 else 240}, {a.save_seconds or a.run_seconds}, {a.run_seconds}')
    short = """    if short:
        command += ['--tasks', str(SAVE_TASKS), '--play-seconds', str(SAVE_SECONDS)]
        solutions = data / 'arc-agi_training_solutions.json'  # the placeholder's tasks are training tasks
        if solutions.is_file() and set(tests) <= set(json.loads(solutions.read_text())):
            command += ['--solutions', str(solutions)]
"""
    c240 = """    if short:  # l4 b2 preview: the c240 roster (120 evaluation tasks x 2, 334 outputs), scored, to the end epoch
        folder = bundle / 'data/arc2/evaluation'
        names = sorted(p.stem for p in folder.glob('*.json'))
        assert len(names) == 120, len(names)
        roster = {n + ('' if k == 0 else f'_{k + 1}'): json.loads((folder / f'{n}.json').read_text()) for n in names for k in range(COPIES)}
        tests = {n: {'train': t['train'], 'test': [{'input': x['input']} for x in t['test']]} for n, t in roster.items()}
        challenges, solutions = Path('/tmp/c240-challenges.json'), Path('/tmp/c240-solutions.json')
        challenges.write_text(json.dumps(tests))
        solutions.write_text(json.dumps({n: [x['output'] for x in t['test']] for n, t in roster.items()}))
        command[command.index('--challenges') + 1] = str(challenges)
        command += ['--tasks', str(SAVE_TASKS), '--play-seconds', str(SAVE_SECONDS), '--solutions', str(solutions)]
"""
    assert code.count(short) == 1
    if a.live:  # every minute: engine (running/waiting, KV, prefix hit, preemptions), gate, submits and drains
        for old, new in LIVE:
            assert code.count(old) == 1, old[:60]
            code = code.replace(old, new)
    for old, new in SAVE_TRIALS:
        assert code.count(old) == 1, old[:60]
        code = code.replace(old, new)
    if a.steady:
        for old, new in (PEAK_HELPER, *([PEAK_LIVE] if a.live else [])):
            assert code.count(old) == 1, old[:60]
            code = code.replace(old, new)
    if a.c240 or a.c120:
        code = code.replace(short, c240.replace('COPIES', '1' if a.c120 else '2'))
    if a.fail_fast:
        for old, new in FAIL_FAST:
            assert code.count(old) == 1, old[:60]
            code = code.replace(old, new)

    spec = importlib.util.spec_from_file_location('build_arc2', ARC2 / 'docs/validation-20261009/build_arc2.py')
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    for cache in list(out.rglob('__pycache__')):
        shutil.rmtree(cache)
    info = b.kaggle_files(out, 'tool')
    with tarfile.open(out / 'dataset-upload' / f'{out_name}.tar.gz.blob') as archive:
        names = archive.getnames()
        assert not [n for n in names if '__pycache__' in n or n.endswith('.pyc')]
        assert all(any(n.endswith(h) for n in names) for h in files), 'harness file missing from the blob'
        assert any(n.endswith('data/arc2/evaluation/' + sorted(os.listdir(out / 'data/arc2/evaluation'))[0]) for n in names)
    for extra in ('submission.ipynb', 'kernel-metadata.json'):
        (out / extra).unlink()
    V49_BLOB = '21db8b9fbb100a5e5ae84fabf774c2ce4dc953480a62451d134143f6ac1024bb'
    assert code.count(V49_BLOB) == 1
    code = code.replace(V49_BLOB, info['sha256']).replace('arc2-kaggle-v49-runtime', f'{out_name}-runtime').replace(
        'arc2-kaggle-v49.tar.gz.blob', f'{out_name}.tar.gz.blob')
    assert 'arc2-kaggle-v49' not in code
    if a.ar_overlay:
        code = code.replace(OLD_OVERLAY, a.ar_overlay)
        old = """        shards = set(json.loads((overlay / 'overlay-manifest.json').read_text())['shards'])  # served in place of the base's
        fetched = prefetch([overlay / f.name if f.name in shards else f.resolve()
                            for f in sorted(model.iterdir()) if f.name.endswith('.safetensors')])
"""
        assert code.count(old) == 1
        code = code.replace(old, """        fetched = prefetch(sorted(overlay.glob('heal-*.safetensors')) + [f.resolve() for f in sorted(model.iterdir()) if f.name.startswith('model-plefp8-')])
""")
        at = "os.environ['ARC_MODEL_DIR'] = str(model)\n"
        assert code.count(at) == 1
        code = code.replace(at, at + ("    os.environ['ARC_DRAFT_FILES'] = '1'\n" if a.draft else '') + f"""    w4parts = sorted(Path('/kaggle/input/models/richardcsaky/arc-w4-weights').glob('*/part*/*'), key=lambda p: int(p.parent.name[4:]))
    assert len(w4parts) == 2 and all(p.is_dir() for p in w4parts), f'attach the 2 instances of richardcsaky/arc-w4-weights: {{w4parts}}'
    w4 = Path('/tmp/arc-w4-model'); w4.mkdir(exist_ok=False)
    for part in w4parts:
        for f in part.iterdir():
            if not f.name.startswith('.'):
                (w4 / f.name).symlink_to(f)
    assert (w4 / 'config.json').is_file() and (w4 / 'model.safetensors.index.json').is_file(), 'w4 parts incomplete'
    assert hashlib.sha256(Path('/kaggle/input/datasets/richardcsaky/{a.ar_overlay}/overlay.json').read_bytes()).hexdigest() == {a.ar_sha!r}, 'heal overlay manifest'
""")
    if a.overlay:
        code = code.replace(OLD_OVERLAY, a.overlay)
        # the manifest sha travels in the kernel, not the runtime archive, so the sha can land after the runtime upload
        at = "os.environ['ARC_MODEL_DIR'] = str(model)\n"
        assert code.count(at) == 1
        code = code.replace(at, at + f"    os.environ['ARC2_OVERLAY_SHA'] = {a.overlay_sha!r}\n")
    compile(code, 'nb', 'exec'); nb['cells'][1]['source'] = code.splitlines(keepends=True)
    diff = [l for l in difflib.unified_diff(old.splitlines(), code.splitlines(), lineterm='', n=0)
            if l[:1] in '+-' and l[:3] not in ('+++', '---')]
    print('notebook lines changed vs v49:', len(diff))
    kname = out_name + ('-c240' if a.c240 else '-c120' if a.c120 else '')
    k = DIST / f'{kname}-kernel'; shutil.rmtree(k, ignore_errors=True); k.mkdir()
    (k / 'submission.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
    m = json.loads((template / 'kernel-metadata.json').read_text())
    m.update(id='richardcsaky/' + kname, title=kname)
    m['dataset_sources'] = [info['dataset'] if d.endswith('arc2-kaggle-v49-runtime') else d for d in m['dataset_sources']]
    if a.overlay:
        m['dataset_sources'] = [d.replace(OLD_OVERLAY, a.overlay) for d in m['dataset_sources']]
    if a.ar_overlay:
        m['dataset_sources'] = [d.replace(OLD_OVERLAY, a.ar_overlay) for d in m['dataset_sources']]
        m['model_sources'] += [f'richardcsaky/arc-w4-weights/PyTorch/part{n}/1' for n in (1, 2)]
    assert 'keithtyser' not in json.dumps(m) and info['dataset'] in m['dataset_sources']
    (k / 'kernel-metadata.json').write_text(json.dumps(m, indent=2) + '\n')
    print(json.dumps(info, indent=1)); print(m['dataset_sources'])


if __name__ == '__main__':
    main()
