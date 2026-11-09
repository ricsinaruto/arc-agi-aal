"""The arena's vLLM server start: the pinned recipe with our boot patches, flags and checks (``python -m arena.serving``)."""
import argparse
import asyncio
import json
import os
import re
import subprocess
from pathlib import Path

import httpx

from arc_infra.served_route_patch import SERVER_FLAGS as served_route_flags, apply as apply_served_route
from arc_infra import serving_fast_patch
from arc_infra import expert_mask_patch
from arc_infra.serving import BASE_URL, MODEL, load_recipe
from arc_infra.serving_fork import FORK_SETUP, GPU_MEMORY_UTILIZATION, GPU_MEMORY_UTILIZATION_ENV  # noqa: F401
from arc_infra.serving_fork import fork_identity, stage_fork_bundle
from . import dry_boot
from arc_infra.serving_kernels_patch import PLE_OFFLOAD_ENV, ple_target
from arc_infra.serving_kernels_patch import apply as apply_serving_kernels
from arc_infra.serving_kernels_patch import apply_ple as apply_serving_kernels_ple
from arc_infra.serving_kernels_patch import server_environment as serving_kernel_environment
from arc_infra.serving_kernels_patch import server_flags as serving_kernel_flags
from arc_infra.serving_kernels_patch import validate as validate_serving_kernels

from . import runtime_cache
from .cache import BlockBudget
from .cache_patch import apply
from .cache_reclamation import apply as apply_reclamation
from .compatibility import validate_settings
from .prefix_cache_patch import apply as apply_prefix_cache
from .prefix_cache_patch import validate as validate_prefix_cache


def clear_compile_cache(cache_root):
    """Remove vLLM's torch.compile / AOT cache under ``cache_root`` (the serving XDG cache): its key does not see a
    changed quant method or expert layout, so a graph an earlier server left in this container would be reused."""
    import shutil

    removed = []
    for path in (Path(cache_root) / 'vllm' / 'torch_compile_cache',):
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    return removed


# Server knobs a trial's server_env may change without a new bundle (``lab_overrides``): ARENA_READY_GUARD="<after
# capture>,<total>" s (fail fast on a hung warm-up), ARENA_SETUP_TIMEOUTS="<server ready>,<setup total>" s (raises the
# pinned recipe's 1,500 / 1,800 s), ARENA_PROMPT_TOKENS_DETAILS=1 (--enable-prompt-tokens-details). A prefix-match unit
# (--prefix-match-unit: prefix-cache keys every N tokens) divides the 1,600-token blocks and is a multiple of the QSA
# compressor's 4-token group.
PREFIX_MATCH_UNITS = (16, 32, 64, 160, 320, 400, 800)


def ready_guard(recipe, after_capture, total):
    """``recipe.wait_for_server`` that also gives up ``after_capture`` s after CUDA-graph capture or ``total`` s
    after launch (the pinned wait allows 1,500 s; two expert-usage runs hung after capture)."""
    import time

    expected = [recipe.SERVED_MODEL_NAME]

    def wait(identity, setup_deadline):
        started = time.monotonic()
        captured = None
        deadline = min(started + total, setup_deadline)
        while time.monotonic() < deadline:
            if not recipe.alive(int(identity["pid"]), int(identity["start_ticks"])):
                raise RuntimeError(f"vLLM exited before readiness.\n{recipe.tail_log(300)}")
            try:
                models = recipe.request_json(f"{recipe.BASE_URL}/models", timeout=10)
                ids = [row.get("id") for row in (models.get("data") or []) if isinstance(row, dict)]
                if ids == expected:
                    return {"models": models, "ready_seconds": time.monotonic() - started}
            except Exception:  # noqa: BLE001 - not ready yet
                pass
            if captured is None and "Graph capturing finished" in recipe.tail_log(400):
                captured = time.monotonic()
                print(f"ARC_READY_GUARD capture_seen elapsed_s={int(captured - started)}", flush=True)
            if captured is not None and time.monotonic() - captured > after_capture:
                raise TimeoutError(f"ARC ready guard: engine not ready {after_capture}s after CUDA-graph capture "
                                   f"(hung warm-up).\n{recipe.tail_log(300)}")
            time.sleep(5)
        raise TimeoutError(f"ARC ready guard: engine not ready after {total}s.\n{recipe.tail_log(300)}")

    return wait


GRAPH_CAPTURE_MAX = 1152  # 1152 (27 Oct): real-v5b 352 seqs x MTP 3.  MFU-2 tput6 (23 Oct): capture to 1024 cut the unprofiled step 52.0 -> 42.3 ms


def validate_graph_capture_sizes(sizes, graph_capture_size):
    """An explicit CUDA-graph capture list: strictly increasing integers in 1..GRAPH_CAPTURE_MAX whose largest is the
    capture size."""
    if sizes is None:
        return None
    if (not isinstance(sizes, list) or not sizes or any(type(v) is not int or not 1 <= v <= GRAPH_CAPTURE_MAX for v in sizes)
            or sizes != sorted(set(sizes))):
        raise ValueError(f'cudagraph_capture_sizes must be strictly increasing integers in 1..{GRAPH_CAPTURE_MAX}')
    if graph_capture_size is None or sizes[-1] != graph_capture_size:
        raise ValueError('The largest capture size must equal max_cudagraph_capture_size')
    return list(sizes)


def lab_overrides(environ=None):
    """Parse the serving-lab overrides; unknown values fail before the server starts."""
    environ = os.environ if environ is None else environ
    result = {}
    if environ.get('ARENA_READY_GUARD'):
        after, total = (int(v) for v in environ['ARENA_READY_GUARD'].split(','))
        if not 60 <= after <= total <= 3600:
            raise ValueError('ARENA_READY_GUARD is "<after capture>,<total>" seconds, 60 <= after <= total <= 3600')
        result['ready_guard'] = (after, total)
    if environ.get('ARENA_SETUP_TIMEOUTS'):
        ready, total = (int(v) for v in environ['ARENA_SETUP_TIMEOUTS'].split(','))
        if not 1500 <= ready <= total <= 7200 or total < 1800:
            raise ValueError('ARENA_SETUP_TIMEOUTS is "<server ready>,<setup total>" seconds, 1500 <= ready <= total '
                             '<= 7200, total >= 1800')
        result['setup_timeouts'] = (ready, total)
    if environ.get('ARENA_PROMPT_TOKENS_DETAILS'):
        if environ['ARENA_PROMPT_TOKENS_DETAILS'] not in ('0', '1'):
            raise ValueError('ARENA_PROMPT_TOKENS_DETAILS must be 0 or 1')
        result['prompt_tokens_details'] = environ['ARENA_PROMPT_TOKENS_DETAILS'] == '1'
    return result


LORA_DEFAULTS = {'max_rank': 32, 'max_loras': 1, 'max_cpu_loras': 2, 'moe_backend': 'marlin', 'sleep_mode': True,
                 'ple_queue_wait': None,  # removed (rl-min): null only
                 'mtp': False}  # True: keep the MTP draft beside the adapter (the draft has no adapter; 25 Oct check)


def validate_lora(lora):
    """RL step 1 (13 Oct): an unmerged adapter on the served checkpoint. ``max_rank`` (vLLM's ranks), adapter slots,
    the MoE kernel that carries expert LoRA for NVFP4 (Marlin), sleep mode for the same-card trainer."""
    if lora is None:
        return None
    if not isinstance(lora, dict) or set(lora) - set(LORA_DEFAULTS):
        raise ValueError(f'lora is a dict over {sorted(LORA_DEFAULTS)}')
    merged = {**LORA_DEFAULTS, **lora}
    if merged['max_rank'] not in (8, 16, 32, 64, 128, 256):
        raise ValueError('lora.max_rank is one of vLLM\'s ranks (8..256)')
    if type(merged['max_loras']) is not int or not 1 <= merged['max_loras'] <= 8 or type(
            merged['max_cpu_loras']) is not int or merged['max_cpu_loras'] < merged['max_loras']:
        raise ValueError('lora.max_loras is 1..8 and max_cpu_loras at least that')  # 8: joint reads (21 Oct)
    if merged['moe_backend'] not in ('marlin', 'triton') or type(merged['sleep_mode']) is not bool:
        raise ValueError('lora.moe_backend is marlin or triton; lora.sleep_mode is boolean')
    if merged['ple_queue_wait'] is not None:
        raise ValueError('lora.ple_queue_wait is gone (rl-min): null only')
    if type(merged['mtp']) is not bool:
        raise ValueError('lora.mtp is boolean')
    return merged


def lora_flags(argv, lora):
    """The serving command with LoRA on: the MoE backend replaced (the pinned one has no LoRA path), the adapter
    flags, the mixed MoE LoRA format (PEFT 3D expert adapters), sleep mode; no draft model (the MTP head has no
    adapter)."""
    argv = list(argv)
    if '--speculative-config' in argv and not lora.get('mtp'):
        raise ValueError('LoRA serving runs without MTP: launch with mtp_tokens 0 (or lora.mtp true)')
    if '--moe-backend' in argv:
        argv[argv.index('--moe-backend') + 1] = lora['moe_backend']
    else:
        argv.extend(['--moe-backend', lora['moe_backend']])
    for flag in ('--enable-lora', '--max-lora-rank', '--max-loras', '--enable-sleep-mode'):
        if flag in argv:
            raise ValueError(f'The pinned serving command already sets {flag}')
    argv.extend(['--enable-lora', '--max-lora-rank', str(lora['max_rank']), '--max-loras', str(lora['max_loras']),
                 '--max-cpu-loras', str(lora['max_cpu_loras']), '--enable-mixed-moe-lora-format'])
    if lora['sleep_mode']:
        argv.append('--enable-sleep-mode')
    return argv


def lora_environment(env):
    """Runtime adapter loading, the localhost dev endpoints (sleep, wake), and the build arch of the card in use
    (the recipe pins 12.0, the RTX PRO 6000)."""
    env = dict(env)
    env.update(VLLM_SERVER_DEV_MODE='1', VLLM_ALLOW_RUNTIME_LORA_UPDATING='True')
    try:
        cap = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], capture_output=True,
                             text=True, timeout=30).stdout.split()[0]
        env['TORCH_CUDA_ARCH_LIST'] = cap.split('.')[0] + '.0'
    except (OSError, IndexError, subprocess.SubprocessError):
        pass
    return env


def configure(recipe, working, *, cpu_gib=0, graph_capture_size=None, reclaim_skipped_blocks=False,
              serving_kernels=None, ready_guard_seconds=None, prefix_cache=None, prompt_tokens_details=False,
              prefix_match_unit=None,
              graph_capture_sizes=None, setup_timeouts=None, renderer_num_workers=None,
              mm_processor_cache_gb=None, lora=None, served_route=None, serving_fast=None, expert_mask=None):
    if dry_boot.enabled():  # CPU dry boot (arena.dry_boot): GPU inventory and the engine start are stand-ins
        dry_boot.install(recipe, working)
    lora = validate_lora(lora)
    serving_fast = serving_fast_patch.validate(serving_fast)
    if served_route is not None and (type(served_route) is not str or not served_route.startswith('/')):
        raise ValueError('served_route is an absolute directory')
    prefix_cache = validate_prefix_cache(prefix_cache)
    if type(prompt_tokens_details) is not bool:
        raise ValueError('prompt_tokens_details must be boolean')
    if prefix_match_unit is not None and prefix_match_unit not in PREFIX_MATCH_UNITS:
        raise ValueError(f'prefix_match_unit must be one of {PREFIX_MATCH_UNITS}')
    if renderer_num_workers is not None and (type(renderer_num_workers) is not int or not 1 <= renderer_num_workers <= 16):
        raise ValueError('renderer_num_workers is an integer 1..16')
    if mm_processor_cache_gb is not None and (type(mm_processor_cache_gb) not in (int, float)
                                              or not 0 <= mm_processor_cache_gb <= 32):
        raise ValueError('mm_processor_cache_gb is a number of GiB in 0..32 (vLLM default 4)')
    if expert_mask is not None and not (isinstance(expert_mask, tuple) and len(expert_mask) == 2):
        raise ValueError('expert_mask is (mask path, sha256) (arena.serving.bundle_expert_mask)')
    if setup_timeouts is not None:
        # Before recipe.main(): wait_for_server's deadline and main()'s setup deadline read these globals.
        recipe.SERVER_READY_TIMEOUT, recipe.SETUP_TOTAL_TIMEOUT = (int(v) for v in setup_timeouts)
    if ready_guard_seconds is not None:
        recipe.wait_for_server = ready_guard(recipe, *ready_guard_seconds)
    serving_kernels = validate_serving_kernels(serving_kernels)
    if type(reclaim_skipped_blocks) is not bool:
        raise ValueError('reclaim_skipped_blocks must be boolean')
    cpu_gib = int(os.environ.get('ARENA_CPU_CACHE_GIB', cpu_gib))  # per-trial override (RL hero, 24 Oct)
    if type(cpu_gib) is not int or not 0 <= cpu_gib <= 192:  # B300 host: ~500 GiB free beside the trainer's park
        raise ValueError('Arena CPU cache must be 0..192 GiB')
    if graph_capture_size is not None:
        if type(graph_capture_size) is not int or not 1 <= graph_capture_size <= GRAPH_CAPTURE_MAX:
            raise ValueError(f'Arena CUDA graph capture size must be an integer in 1..{GRAPH_CAPTURE_MAX}')
        os.environ[recipe.MAX_CUDAGRAPH_CAPTURE_SIZE_ENV] = str(graph_capture_size)
    graph_capture_sizes = validate_graph_capture_sizes(graph_capture_sizes, graph_capture_size)
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"  # localhost-only cache reset after serving probes
    extract = recipe.verify_and_extract_runtime

    def wrapped_extract(*args, **kwargs):
        cached = os.environ.get(runtime_cache.ENV)  # MFU-2 boot item: the extracted tree from the volume
        result = runtime_cache.restore(recipe, cached, args, kwargs) if cached else None
        if result is None:
            result = extract(*args, **kwargs)
        if dry_boot.enabled():
            dry_boot.mark(working)
        if cpu_gib:  # the hybrid-cache offloading connector (arc_infra.arc_offload_runtime, throughput arm L1)
            target = recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages/vllm/arc_offload.py'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((Path(__file__).parents[1] / 'arc_infra/arc_offload_runtime.py').read_bytes())
            target.with_name('arc_routed_offload.py').write_bytes(  # R3 rows through the host tier (served_route)
                (Path(__file__).parents[1] / 'arc_infra/arc_routed_offload.py').read_bytes())
        apply(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working,
              **({'state_cotenant': True} if prefix_cache and prefix_cache.get('state_cotenant') else {}))
        if reclaim_skipped_blocks:
            apply_reclamation(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working)
        if prefix_cache is not None:
            # Prefix cache v2 (29 September): after the reclamation patch, which edits the same manager file.
            apply_prefix_cache(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working, prefix_cache)
        if serving_kernels is not None:
            # Kernel work, phase 4: the HC fusion and the draft quant method change the compiled graph as well.
            cleared = clear_compile_cache(recipe.CACHE_ROOT)
            identity = apply_serving_kernels(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working,
                                             serving_kernels)
            (Path(working) / 'serving-kernels-start.json').write_text(json.dumps(
                {'identity': identity, 'cleared_compile_cache': cleared}, indent=2))
        if served_route is not None:
            # R3: each finished request's served top-k experts per layer, written under served_route.
            # With the host KV tier (cpu_gib) the routed rows travel with its chunks (arc_routed_offload).
            apply_served_route(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working, served_route,
                               offload=bool(cpu_gib))
        if serving_fast is not None and any(serving_fast.values()):
            # MFU-2 (22 Oct): API front-end overlays (sampled-only logprob text, decoded-image cache); replies unchanged.
            serving_fast_patch.apply(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working,
                                     serving_fast)
        if expert_mask is not None:
            # Hero run (23 Oct): the pruned model as a router-logit mask; expert ids, LoRA and served routes unchanged.
            # Compact mode (MFU-2) allocates only kept experts: a compile cache from an uncompacted start is cleared.
            cleared = clear_compile_cache(recipe.CACHE_ROOT) if os.environ.get(expert_mask_patch.COMPACT_ENV) == '1' else []
            identity = expert_mask_patch.apply(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working,
                                               *expert_mask)
            (Path(working) / 'expert-mask-start.json').write_text(json.dumps(
                {'identity': identity, 'cleared_compile_cache': cleared}, indent=2))
        return result

    recipe.verify_and_extract_runtime = wrapped_extract
    environment = getattr(recipe, 'runtime_environment', None)
    ple_uva = serving_kernels is not None and ple_target(serving_kernels) is not None
    if ple_uva and environment is None:
        raise ValueError('ple_uva needs the recipe runtime_environment')
    if environment is not None and (ple_uva or lora is not None):
        # Serving tweaks (ple_uva): the PLE overlay patches the file the recipe's own PLE patch writes, after the
        # recipe's last check of that file (runtime_environment verifies VLLM_PLE_PATCHED_SHA256; smoke v2 failed
        # when the overlay ran right after patch_ple_layer), and the server runs without the PLE offload process.
        # 8-bit KV smoke: the serving bench runs the QSA microbenchmark in the server's own runtime environment.
        def wrapped_environment(*args, **kwargs):
            env, check = environment(*args, **kwargs)
            if lora is not None:
                env, check = lora_environment(env), {**check, 'lora_environment': True}
            if ple_uva:
                identity = apply_serving_kernels_ple(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages',
                                                     working, serving_kernels)
                (Path(working) / 'serving-kernels-ple-start.json').write_text(
                    json.dumps({'identity': identity}, indent=2))
                env = serving_kernel_environment(serving_kernels, env)
                check = {**check, 'serving_kernels_environment': {PLE_OFFLOAD_ENV: '0'}}
            return env, check

        recipe.runtime_environment = wrapped_environment
    command = recipe.server_command

    def wrapped_command(*args, **kwargs):
        argv = command(*args, **kwargs)
        if cpu_gib:
            argv.extend(['--kv-transfer-config', json.dumps({
                'kv_connector': 'ArcOffloadingConnector', 'kv_connector_module_path': 'vllm.arc_offload',
                'kv_role': 'kv_both',
                'kv_connector_extra_config': {'cpu_bytes_to_use': cpu_gib * 1024**3, 'offload_prompt_only': False,
                                              # R3 rows through the host tier: bound/refused status for the arena
                                              **({'arc_routed_status_dir': str(working)} if served_route else {})},
            }), '--no-disable-hybrid-kv-cache-manager'])
        if serving_kernels is not None:
            spec_tokens = 0
            if '--speculative-config' in argv:  # the draft length sizes the 4-bit block (kv4_block_tokens)
                spec_tokens = int(json.loads(argv[argv.index('--speculative-config') + 1]).get('num_speculative_tokens', 0))
            flags = serving_kernel_flags(serving_kernels, spec_tokens)
            for name, value in zip(flags[::2], flags[1::2]):
                if name in argv:
                    raise ValueError(f'The pinned serving command already sets {name}')
                argv.extend([name, value])
        if prompt_tokens_details and '--enable-prompt-tokens-details' not in argv:
            argv.append('--enable-prompt-tokens-details')
        if served_route is not None:  # R3: vLLM's own routed-experts return, dumped by the served-route overlay
            argv.extend(flag for flag in served_route_flags if flag not in argv)
        if graph_capture_sizes is not None:
            # vLLM's CompilationConfig.cudagraph_capture_sizes; its largest entry equals --max-cudagraph-capture-size.
            if any(flag in argv for flag in ('--compilation-config', '-cc', '-O')):
                raise ValueError('The pinned serving command already sets a compilation config')
            argv.extend(['--compilation-config',
                         json.dumps({'cudagraph_capture_sizes': graph_capture_sizes}, separators=(',', ':'))])
        if prefix_match_unit is not None:
            if '--prefix-match-unit' in argv:
                raise ValueError('The pinned serving command already sets --prefix-match-unit')
            argv.extend(['--prefix-match-unit', str(prefix_match_unit)])
        if renderer_num_workers is not None:
            # B200 (5 October): the API server's chat-template and tokenizer thread pool (vLLM default 1); multimodal
            # processing keeps its own single thread.
            if '--renderer-num-workers' in argv:
                raise ValueError('The pinned serving command already sets --renderer-num-workers')
            argv.extend(['--renderer-num-workers', str(renderer_num_workers)])
        if lora is not None:
            argv = lora_flags(argv, lora)
        if mm_processor_cache_gb is not None:
            # v7 (5 October): the processed-image LRU is duplicated in the API server and the engine core
            # (4 GiB each by default); host RAM on Kaggle is ~78.6 GiB with 47.7 GiB of PLE pinned.
            if '--mm-processor-cache-gb' in argv:
                raise ValueError('The pinned serving command already sets --mm-processor-cache-gb')
            argv.extend(['--mm-processor-cache-gb', str(mm_processor_cache_gb)])
        return argv

    recipe.server_command = wrapped_command
    extra = {'max_cudagraph_capture_size': graph_capture_size, 'cudagraph_capture_sizes': graph_capture_sizes,
             'reclaim_skipped_blocks': reclaim_skipped_blocks or None, 'prefix_cache': prefix_cache,
             'prompt_tokens_details': prompt_tokens_details or None, 'prefix_match_unit': prefix_match_unit,
             'renderer_num_workers': renderer_num_workers, 'mm_processor_cache_gb': mm_processor_cache_gb, 'lora': lora,
             'serving_kernels': serving_kernels,
             'ready_guard_seconds': list(ready_guard_seconds) if ready_guard_seconds is not None else None,
             'expert_mask': {'mask': str(expert_mask[0]), 'sha256': expert_mask[1]} if expert_mask is not None else None,
             'served_route': served_route, 'serving_fast': serving_fast}
    return {'cpu_cache_gib': cpu_gib, 'layout_export': True, 'shared_prefix_discount': False,
            **{k: v for k, v in extra.items() if v is not None}}


CONTEXT = 262144  # the native context every start must admit (no silent downgrade)
KV_LOG_PATTERNS = ("prefix caching", "Prefix cach", "Initial free memory", "Available KV cache memory",
                   "GPU KV cache size", "Maximum concurrency", "Model loading took", "KV cache is needed")


def kv_admission_summary(working):
    """vLLM's memory/KV admission lines from the owned server log, if present."""
    lines = []
    for log in sorted(Path(working).glob("*.log")):
        for line in log.read_text(errors="replace").splitlines():
            if any(pattern in line for pattern in KV_LOG_PATTERNS):
                lines.append(re.sub(r"^\([^)]*\)\s*", "", line).strip())
    return lines[-12:]


def verify_hardware():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True)
    cards = [line for line in output.splitlines() if line.strip()]
    if (len(cards) != 1
            or not (("RTX PRO 6000" in cards[0] and "Blackwell" in cards[0]) or "B200" in cards[0] or "B300" in cards[0])
            or int(cards[0].rsplit(",", 1)[1]) < 95000):
        raise RuntimeError(f"Require exactly one RTX PRO 6000 Blackwell 96 GB, B200 or B300; found {cards}")
    return cards[0]


async def preflight():
    """The started server: /tokenize reports the native context and /v1/models serves MODEL."""
    hardware = verify_hardware()
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Native context admission check."}],
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False, "reasoning_effort": "xhigh"},
            "add_generation_prompt": True}
    async with httpx.AsyncClient(base_url=BASE_URL.rstrip("/") + "/", timeout=None, trust_env=False) as client:
        async with asyncio.timeout(60):
            for attempt in range(3):  # tokenization has no side effect: a dropped connection retries it
                try:
                    response = await client.post(client.base_url.copy_with(path="/tokenize"), json=body)
                    break
                except httpx.TransportError:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(.05 * (attempt + 1))
            if response.is_error:
                raise RuntimeError(f"/tokenize returned {response.status_code}: {response.text[:2000]}")
            data = response.json()
            if data.get("max_model_len") != CONTEXT:
                raise RuntimeError(f"Native context preflight failed: tokenizer reports {data.get('max_model_len')}; "
                                   f"requested {CONTEXT}. No downgrade allowed.")
            response = await client.get("models")
            response.raise_for_status()
            models = response.json()
    served = [m.get("id") for m in models.get("data", [])]
    if MODEL not in served:
        raise RuntimeError(f"Served models {served} do not include {MODEL}")
    return {"hardware": hardware, "native_tokenizer_count": int(data["count"]), "served_models": served,
            "max_model_len": [m.get("max_model_len") for m in models.get("data", [])],
            "serving_setup": fork_identity()}


def verify_sequence_capacity(working, expected):
    """Check the launched command, not just the intended profile, before gameplay."""
    identity = json.loads((Path(working) / 'vllm-server-identity.json').read_text())
    argv = identity.get('argv', [])
    values = [argv[i + 1] for i, item in enumerate(argv[:-1]) if item == '--max-num-seqs']
    if values != [str(expected)]:
        raise RuntimeError(f'Launched vLLM max-num-seqs {values!r} differs from requested {expected}')
    return {'expected_max_num_seqs': expected, 'actual_max_num_seqs': int(values[0]), 'verified': True}


def start(working, input_root, *, kv_cache_memory_bytes=0, generation_slots=8, kv_cache_dtype="auto",
          recipe_hook=None, max_num_batched_tokens=8192, mtp_tokens=3, gpu_memory_utilization=None):
    """The pinned recipe's server start (the fork's serving script), ``recipe_hook`` (configure) before its main."""
    if type(kv_cache_memory_bytes) is not int or kv_cache_memory_bytes < 0:
        raise ValueError("Explicit KV cache bytes must be a non-negative integer")
    # None keeps the fork's 0.96 (a lower share leaves GPU memory outside the profiled KV pool)
    utilization = GPU_MEMORY_UTILIZATION if gpu_memory_utilization is None else gpu_memory_utilization
    if (type(utilization) is not str or re.fullmatch(r"0\.\d{1,3}", utilization) is None
            or not 0.5 <= float(utilization) <= 0.99):
        raise ValueError("gpu_memory_utilization is a decimal string from 0.5 through 0.99")
    if type(mtp_tokens) is not int or mtp_tokens not in (0, 1, 2, 3, 4):
        raise ValueError("Flash serving comparisons support MTP depth 1-4, or 0 (no MTP draft)")
    profile = {"max_model_len": CONTEXT, "max_num_seqs": generation_slots, "prefix_caching": True,
               "speculative_tokens": mtp_tokens, "max_num_batched_tokens": max_num_batched_tokens,
               "kv_cache_dtype": "bfloat16" if kv_cache_dtype == "auto" else kv_cache_dtype,
               "gpu_memory_utilization": utilization, "kv_cache_memory_bytes": kv_cache_memory_bytes}
    working = Path(working)
    working.mkdir(parents=True, exist_ok=True)
    recipe = None
    stage = "hardware"
    try:
        verify_hardware()
        stage = "server_startup"
        os.environ[GPU_MEMORY_UTILIZATION_ENV] = utilization
        bundle = stage_fork_bundle(input_root, working / "fork-bundle")
        recipe = load_recipe(working, input_root, mtp_tokens=mtp_tokens, generation_slots=generation_slots,
                             kv_cache_dtype=kv_cache_dtype, max_num_batched_tokens=max_num_batched_tokens,
                             bundle_dir=bundle, setup_sha256=fork_identity()["fork_sha256"],
                             kv_cache_memory_bytes=kv_cache_memory_bytes, prefix_caching=True)
        if recipe_hook is not None:
            profile["arena_cache"] = recipe_hook(recipe, working)
        recipe.ANALYZER_CONTEXT = CONTEXT
        recipe.main()
        stage = "sequence_capacity"
        sequence_capacity = verify_sequence_capacity(working, generation_slots)
        stage = "native_context"
        result = asyncio.run(preflight())
        result.update(profile=profile, sequence_capacity=sequence_capacity, kv_admission_log=kv_admission_summary(working))
        (working / "native-preflight.json").write_text(json.dumps(result, indent=2))
    except BaseException as exc:
        if recipe is not None:
            recipe.stop_owned_server()
        (working / "native-preflight-failure.json").write_text(json.dumps(
            {"error": str(exc), "stage": stage, "profile": profile, "kv_admission_log": kv_admission_summary(working)},
            indent=2))
        raise



def bundle_expert_mask(spec, bundle):
    """``server.expert_mask`` -> (the bundle file's absolute path, its pinned sha256), checked here."""
    spec = expert_mask_patch.validate(spec)
    if spec is None:
        return None
    path = Path(bundle) / spec['mask']
    expert_mask_patch.load_mask(path, spec['sha256'])
    return str(path), spec['sha256']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--working', required=True)
    p.add_argument('--input-root', required=True)
    p.add_argument('--study', required=True)
    args = p.parse_args()
    study = json.loads(Path(args.study).read_text())
    settings = study['cache_experiment']
    validate_settings(settings)
    # Serving A/B bench (26 September): an explicit KV byte allocation from the trial's server environment;
    # load_recipe resets TAAF_VLLM_KV_CACHE_MEMORY_BYTES from start()'s argument, so it travels separately.
    kv_bytes = int(os.environ.get('ARENA_KV_CACHE_MEMORY_BYTES', '0'))
    # GPU memory headroom: a lower utilization than the fork's 0.96 from the trial's server
    # environment (build_arena gpu_memory_utilization); unset keeps 0.96.
    utilization = os.environ.get('ARENA_GPU_MEMORY_UTILIZATION') or None
    lab = lab_overrides()
    for key in ('expert_prune', 'speculative', 'scheduling_policy', 'cudagraph_metrics', 'api_server_count'):
        if study['server'].get(key):
            raise ValueError(f'server.{key} is gone (rl-min)')
    # Backend capacity and ordinary admission slots are distinct when siblings use spare sequences.
    sequences = study['server'].get('max_num_seqs', study['generation_slots'])
    if type(sequences) is not int or sequences < study['generation_slots']:
        raise ValueError('Server max_num_seqs must cover ordinary generation slots')
    start(args.working, args.input_root, generation_slots=sequences, kv_cache_memory_bytes=kv_bytes,
          gpu_memory_utilization=utilization,
          mtp_tokens=study['server'].get('mtp_tokens', 3),
          kv_cache_dtype=settings['kv_cache_dtype'],
          max_num_batched_tokens=study['server'].get('max_num_batched_tokens', 8192),
          recipe_hook=lambda recipe, working: configure(
              recipe, working, cpu_gib=settings['cpu_cache_gib'],
              graph_capture_size=study['server'].get('max_cudagraph_capture_size'),
              graph_capture_sizes=study['server'].get('cudagraph_capture_sizes'),
              reclaim_skipped_blocks=settings.get('reclaim_skipped_blocks', False),
              serving_kernels=study['server'].get('serving_kernels'),
              served_route=study['server'].get('served_route'),
              serving_fast=study['server'].get('serving_fast'),
              expert_mask=bundle_expert_mask(study['server'].get('expert_mask'), Path(args.study).parent.parent),
              prefix_cache=settings.get('prefix_cache'),
              prompt_tokens_details=lab.get('prompt_tokens_details',
                                            study['server'].get('prompt_tokens_details', False)),
              prefix_match_unit=study['server'].get('prefix_match_unit'),
              renderer_num_workers=study['server'].get('renderer_num_workers'),
              mm_processor_cache_gb=study['server'].get('mm_processor_cache_gb'),
              lora=study['server'].get('lora'),
              ready_guard_seconds=lab.get('ready_guard'),
              setup_timeouts=lab.get('setup_timeouts'),
))
    budget = BlockBudget.load(args.working, study['kv_margin'])
    (Path(args.working) / 'cache-layout.json').write_text(json.dumps(budget.layout, indent=2))


if __name__ == '__main__':
    main()
