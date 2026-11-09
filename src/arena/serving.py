"""Opt-in arena cache serving configuration; native/historical profiles stay unchanged."""
import argparse
import json
import os
import sys
from pathlib import Path

from arc_infra.dense_kernels_patch import apply as apply_dense_kernels
from arc_infra.dense_kernels_patch import clear_compile_cache
from arc_infra.dense_kernels_patch import validate as validate_dense_kernels
from arc_infra.draft_head_patch import DUMP_DIR as DRAFT_DUMP_DIR
from arc_infra.draft_head_patch import apply as apply_draft_head
from arc_infra.draft_head_patch import server_flags as draft_head_flags
from arc_infra.draft_head_patch import validate as validate_draft_head
from arc_infra.expert_capture_patch import apply as apply_expert_capture
from arc_infra.expert_probe_patch import apply as apply_expert_probe
from arc_infra.expert_prune_patch import apply as apply_expert_prune
from arc_infra.expert_prune_patch import kept_summary as expert_prune_kept
from arc_infra.expert_prune_patch import load_mask as load_prune_mask
from arc_infra.expert_prune_patch import validate as validate_expert_prune
from arc_infra.serving_kernels_patch import PLE_OFFLOAD_ENV, ple_target
from arc_infra.serving_kernels_patch import apply as apply_serving_kernels
from arc_infra.serving_kernels_patch import apply_ple as apply_serving_kernels_ple
from arc_infra.serving_kernels_patch import server_environment as serving_kernel_environment
from arc_infra.serving_kernels_patch import server_flags as serving_kernel_flags
from arc_infra.serving_kernels_patch import validate as validate_serving_kernels
from arc_infra.stack_kernels_patch import apply as apply_stack_kernels
from arc_infra.stack_kernels_patch import validate as validate_stack_kernels
from arc_infra.stall_dump_patch import apply as apply_stall_dump
from arc_infra.stall_dump_patch import validate as validate_stall_dump

from . import expert_overlay as expert_overlay_module
from .cache import BlockBudget
from .cache_patch import apply
from .cache_reclamation import apply as apply_reclamation
from .compatibility import validate_settings
from .prefix_cache_patch import apply as apply_prefix_cache
from .prefix_cache_patch import validate as validate_prefix_cache
from .serving_core.serving import start

SPECULATIVE_METHODS = ('ngram_gpu',)  # the pinned script's --async-scheduling rejects the CPU `ngram` proposer


def validate_speculative(speculative):
    """An n-gram (prompt-lookup) speculative config that replaces the pinned script's MTP draft.

    The script emits its own ``--speculative-config`` only for MTP depth > 0, so a study that
    asks for n-gram speculation launches with MTP depth 0 and the arena appends this config.
    """
    if not isinstance(speculative, dict) or set(speculative) != {
            'method', 'num_speculative_tokens', 'prompt_lookup_max', 'prompt_lookup_min'}:
        raise ValueError('Speculative config needs method, num_speculative_tokens, prompt_lookup_max, prompt_lookup_min')
    if speculative['method'] not in SPECULATIVE_METHODS:
        raise ValueError(f'Pinned vLLM async scheduling supports only {SPECULATIVE_METHODS} n-gram speculation')
    tokens, high, low = (speculative[k] for k in ('num_speculative_tokens', 'prompt_lookup_max', 'prompt_lookup_min'))
    if any(type(v) is not int for v in (tokens, high, low)) or not 1 <= tokens <= 8 or not 1 <= low <= high <= 16:
        raise ValueError('Speculative draft length must be 1..8 and 1 <= prompt_lookup_min <= prompt_lookup_max <= 16')
    return dict(speculative)


SCHEDULING_POLICIES = ('priority',)  # vLLM's default is fcfs; only an explicit change is appended


def profiler_config(directory):
    """Torch profiler, created lazily by vLLM on the first /start_profile (no cost until then).

    Skips three steps after the start (the uncached tail of a warmed prefix) and stops itself after
    40 decode iterations; no Python stacks, no front-end (AsyncLLM) profile.
    """
    return {'profiler': 'torch', 'torch_profiler_dir': str(directory), 'ignore_frontend': True,
            'delay_iterations': 3, 'max_iterations': 40, 'torch_profiler_with_stack': False,
            'torch_profiler_use_gzip': True}


# Serving lab (27 September): server knobs a trial's server_env may change without a new bundle.
# load_recipe resets the TAAF_VLLM_* equivalents from start()'s arguments, so they travel separately.
# ARENA_DENSE_KERNELS (kernel work, phase 2): "off" serves the pinned runtime whatever the study's
# server.dense_kernels says (the kernel gate's control reference); a JSON object replaces it.
# ARENA_SERVING_KERNELS (kernel work, phase 4): the same for server.serving_kernels.
# ARENA_STACK_KERNELS (kernel work, phase 5): the same for server.stack_kernels.
# ARENA_EXPERT_OVERLAY (expert quantization, 28 September): "off" or the JSON of server.expert_overlay.
# Expert-usage diagnostics (28 September): ARENA_EXPERT_PROBE ("0"/"1") replaces server.expert_probe;
# ARENA_ENFORCE_EAGER=1 adds --enforce-eager (no torch.compile, no CUDA graphs); ARENA_READY_GUARD="<after
# capture>,<total>" seconds makes the start fail fast when the engine never becomes ready (a hung warm-up).
# ARENA_PREFIX_CACHE (prefix cache v2, 29 September): "off" or the JSON of cache_experiment.prefix_cache.
# ARENA_PROMPT_TOKENS_DETAILS=1 adds --enable-prompt-tokens-details (exact per-request cached tokens in usage).
# ARENA_PREFIX_MATCH_UNIT=N (hero-v2 precheck, 29 September) adds --prefix-match-unit N: prefix-cache keys every N
# tokens instead of the gcd of the prefix-cacheable groups' blocks (1,600 on the hero), which turns on the pinned
# runtime's partial-tail entry at a prompt's last N-token boundary (docs/validation-20260927/hero-v2-precheck.md).
LAB_ENV = ('ARENA_KV_CACHE_DTYPE', 'ARENA_MTP_TOKENS', 'ARENA_MAX_CUDAGRAPH_CAPTURE_SIZE', 'ARENA_PREFIX_CACHING',
           'ARENA_TORCH_PROFILER', 'ARENA_DENSE_KERNELS', 'ARENA_SERVING_KERNELS', 'ARENA_STACK_KERNELS',
           'ARENA_EXPERT_OVERLAY', 'ARENA_EXPERT_PROBE', 'ARENA_ENFORCE_EAGER', 'ARENA_READY_GUARD',
           'ARENA_EXPERT_PRUNE', 'ARENA_STALL_DUMP', 'ARENA_PREFIX_CACHE', 'ARENA_PROMPT_TOKENS_DETAILS',
           'ARENA_PREFIX_MATCH_UNIT', 'ARENA_WARM_PROMPT_LOGPROBS', 'ARENA_SETUP_TIMEOUTS')
# ARENA_SETUP_TIMEOUTS="<server ready>,<setup total>" (competition int3 builds, 30 September): raises the pinned recipe's
# SERVER_READY_TIMEOUT (1,500 s) and SETUP_TOTAL_TIMEOUT (1,800 s), module globals read when main() runs; Kaggle's slow
# weight load plus Humming's kernel build can pass 1,800 s. Only raises (never lowers) them.
# A unit must divide the hero's 1,600-token attention/Mamba blocks and be a multiple of the QSA compressor's 4-token
# group (the pinned CSA+linear check: a hit never lands inside a partially filled compressor state).
PREFIX_MATCH_UNITS = (16, 32, 64, 160, 320, 400, 800)


def ready_guard(recipe, after_capture, total):
    """``recipe.wait_for_server`` that also gives up ``after_capture`` s after CUDA-graph capture or ``total`` s
    after launch (the pinned wait allows 1,500 s; two expert-usage runs hung after capture)."""
    import time

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
                if ids == [recipe.SERVED_MODEL_NAME]:
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


def validate_graph_capture_sizes(sizes, graph_capture_size):
    """An explicit CUDA-graph capture list: strictly increasing integers in 1..256 whose largest is the capture size."""
    if sizes is None:
        return None
    if (not isinstance(sizes, list) or not sizes or any(type(v) is not int or not 1 <= v <= 256 for v in sizes)
            or sizes != sorted(set(sizes))):
        raise ValueError('cudagraph_capture_sizes must be strictly increasing integers in 1..256')
    if graph_capture_size is None or sizes[-1] != graph_capture_size:
        raise ValueError('The largest capture size must equal max_cudagraph_capture_size')
    return list(sizes)


def lab_overrides(environ=None):
    """Parse the serving-lab overrides; unknown values fail before the server starts."""
    environ = os.environ if environ is None else environ
    result = {}
    if environ.get('ARENA_KV_CACHE_DTYPE'):
        # bfloat16: a candidate's explicit BF16 KV (the Holo4 drift reference); the Flash recipe takes auto or fp8.
        if environ['ARENA_KV_CACHE_DTYPE'] not in ('auto', 'fp8', 'bfloat16'):
            raise ValueError('ARENA_KV_CACHE_DTYPE must be auto, fp8 or bfloat16')
        result['kv_cache_dtype'] = environ['ARENA_KV_CACHE_DTYPE']
    if environ.get('ARENA_MTP_TOKENS'):
        result['mtp_tokens'] = int(environ['ARENA_MTP_TOKENS'])
    if environ.get('ARENA_MAX_CUDAGRAPH_CAPTURE_SIZE'):
        result['graph_capture_size'] = int(environ['ARENA_MAX_CUDAGRAPH_CAPTURE_SIZE'])
    if environ.get('ARENA_CUDAGRAPH_CAPTURE_SIZES'):
        # Decode sweep (30 September): an explicit capture list, "default" = vLLM's own ladder.
        value = environ['ARENA_CUDAGRAPH_CAPTURE_SIZES']
        result['graph_capture_sizes'] = None if value == 'default' else [int(v) for v in value.split(',')]
    if environ.get('ARENA_PREFIX_CACHING'):
        if environ['ARENA_PREFIX_CACHING'] not in ('0', '1'):
            raise ValueError('ARENA_PREFIX_CACHING must be 0 or 1')
        result['prefix_caching'] = environ['ARENA_PREFIX_CACHING'] == '1'
    if environ.get('ARENA_TORCH_PROFILER'):
        if environ['ARENA_TORCH_PROFILER'] != '1':
            raise ValueError('ARENA_TORCH_PROFILER must be 1 when set')
        result['torch_profiler'] = True
    if environ.get('ARENA_DENSE_KERNELS'):
        value = environ['ARENA_DENSE_KERNELS']
        result['dense_kernels'] = None if value == 'off' else validate_dense_kernels(json.loads(value))
    if environ.get('ARENA_SERVING_KERNELS'):
        value = environ['ARENA_SERVING_KERNELS']
        result['serving_kernels'] = None if value == 'off' else validate_serving_kernels(json.loads(value))
    for key, name in (('ARENA_EXPERT_PROBE', 'expert_probe'), ('ARENA_ENFORCE_EAGER', 'enforce_eager')):
        if environ.get(key):
            if environ[key] not in ('0', '1'):
                raise ValueError(f'{key} must be 0 or 1')
            result[name] = environ[key] == '1'
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
    if environ.get('ARENA_STACK_KERNELS'):
        value = environ['ARENA_STACK_KERNELS']
        result['stack_kernels'] = None if value == 'off' else validate_stack_kernels(json.loads(value))
    result.update(expert_overlay_module.lab_override(environ))
    if environ.get('ARENA_EXPERT_PRUNE'):
        # Load-time expert pruning: "off" or the JSON of server.expert_prune (a bundle-relative mask and its sha256).
        value = environ['ARENA_EXPERT_PRUNE']
        result['expert_prune'] = None if value == 'off' else validate_expert_prune(json.loads(value))
    if environ.get('ARENA_STALL_DUMP'):
        # Diagnostics: all-thread stack dumps in the pruned worker when one step runs this many seconds (a one-shot
        # timer per step, below vLLM's 300 s execute-model timeout; with server.expert_prune).
        seconds = int(environ['ARENA_STALL_DUMP'])
        if not STALL_DUMP_RANGE[0] <= seconds <= STALL_DUMP_RANGE[1]:
            raise ValueError(f'ARENA_STALL_DUMP is {STALL_DUMP_RANGE[0]}..{STALL_DUMP_RANGE[1]} seconds')
        result['stall_dump_seconds'] = seconds
    if environ.get('ARENA_PREFIX_CACHE'):
        value = environ['ARENA_PREFIX_CACHE']
        result['prefix_cache'] = None if value == 'off' else validate_prefix_cache(json.loads(value))
    if environ.get('ARENA_PROMPT_TOKENS_DETAILS'):
        if environ['ARENA_PROMPT_TOKENS_DETAILS'] not in ('0', '1'):
            raise ValueError('ARENA_PROMPT_TOKENS_DETAILS must be 0 or 1')
        result['prompt_tokens_details'] = environ['ARENA_PROMPT_TOKENS_DETAILS'] == '1'
    if environ.get('ARENA_PREFIX_MATCH_UNIT'):
        unit = int(environ['ARENA_PREFIX_MATCH_UNIT'])
        if unit not in PREFIX_MATCH_UNITS:
            raise ValueError(f'ARENA_PREFIX_MATCH_UNIT must be one of {PREFIX_MATCH_UNITS}')
        result['prefix_match_unit'] = unit
    if environ.get('ARENA_WARM_PROMPT_LOGPROBS'):
        # prefix-unit-int4 (29 September): teacher forcing on warm requests (arena.prompt_logprobs_patch).
        if environ['ARENA_WARM_PROMPT_LOGPROBS'] not in ('0', '1'):
            raise ValueError('ARENA_WARM_PROMPT_LOGPROBS must be 0 or 1')
        result['warm_prompt_logprobs'] = environ['ARENA_WARM_PROMPT_LOGPROBS'] == '1'
    return result


LORA_DEFAULTS = {'max_rank': 32, 'max_loras': 1, 'max_cpu_loras': 2, 'moe_backend': 'marlin', 'sleep_mode': True,
                 'ple_queue_wait': None}  # seconds; arena.ple_queue_patch (15 Oct)


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
    if type(merged['max_loras']) is not int or not 1 <= merged['max_loras'] <= 4 or type(
            merged['max_cpu_loras']) is not int or merged['max_cpu_loras'] < merged['max_loras']:
        raise ValueError('lora.max_loras is 1..4 and max_cpu_loras at least that')
    if merged['moe_backend'] not in ('marlin', 'triton') or type(merged['sleep_mode']) is not bool:
        raise ValueError('lora.moe_backend is marlin or triton; lora.sleep_mode is boolean')
    wait = merged['ple_queue_wait']
    if wait is not None and (type(wait) not in (int, float) or not 0 < wait <= 600):
        raise ValueError('lora.ple_queue_wait is None or seconds in (0, 600]')
    return merged


def lora_flags(argv, lora):
    """The serving command with LoRA on: the MoE backend replaced (the pinned one has no LoRA path), the adapter
    flags, the mixed MoE LoRA format (PEFT 3D expert adapters), sleep mode; no draft model (the MTP head has no
    adapter)."""
    argv = list(argv)
    if '--speculative-config' in argv:
        raise ValueError('LoRA serving runs without MTP: launch with mtp_tokens 0')
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
    import subprocess

    env = dict(env)
    env.update(VLLM_ALLOW_RUNTIME_LORA_UPDATING='True', VLLM_SERVER_DEV_MODE='1')
    try:
        cap = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], capture_output=True,
                             text=True, timeout=30).stdout.split()[0]
        env['TORCH_CUDA_ARCH_LIST'] = cap.split('.')[0] + '.0'
    except (OSError, IndexError, subprocess.SubprocessError):
        pass
    return env


def configure(recipe, working, *, cpu_gib=0, graph_capture_size=None, reclaim_skipped_blocks=False,
              graph_metrics=False, speculative=None, scheduling_policy=None, torch_profiler=False,
              dense_kernels=None, serving_kernels=None, draft_head=None, stack_kernels=None, expert_capture=False,
              expert_overlay=None, input_root='/kaggle/input', expert_probe=False, enforce_eager=False,
              ready_guard_seconds=None, expert_prune=None, expert_prune_sha256=None, prefix_cache=None, prompt_tokens_details=False,
              stall_dump_seconds=None, prefix_match_unit=None, warm_prompt_logprobs=False, research_checkpoint=None,
              graph_capture_sizes=None, setup_timeouts=None, renderer_num_workers=None,
              mm_processor_cache_gb=None, lora=None):
    lora = validate_lora(lora)
    if research_checkpoint is not None and (expert_overlay is not None or expert_prune is not None):
        raise ValueError('A research checkpoint cannot use the base Flash expert overlay or pruning mask')
    if type(warm_prompt_logprobs) is not bool:
        raise ValueError('warm_prompt_logprobs must be boolean')
    prefix_cache = validate_prefix_cache(prefix_cache)
    # Stall-only worker dump without pruning (arc_infra.stall_dump_patch; a pruned server carries it on its mask).
    stall_dump_seconds = validate_stall_dump(stall_dump_seconds)
    if stall_dump_seconds is not None and expert_prune is not None:
        raise ValueError('A pruned server carries the stall dump on its mask (with_stall_dump), not the standalone '
                         'overlay: both patch base_loader.py')
    if type(prompt_tokens_details) is not bool:
        raise ValueError('prompt_tokens_details must be boolean')
    if prefix_match_unit is not None and prefix_match_unit not in PREFIX_MATCH_UNITS:
        raise ValueError(f'prefix_match_unit must be one of {PREFIX_MATCH_UNITS}')
    if renderer_num_workers is not None and (type(renderer_num_workers) is not int or not 1 <= renderer_num_workers <= 16):
        raise ValueError('renderer_num_workers is an integer 1..16')
    if mm_processor_cache_gb is not None and (type(mm_processor_cache_gb) not in (int, float)
                                              or not 0 <= mm_processor_cache_gb <= 32):
        raise ValueError('mm_processor_cache_gb is a number of GiB in 0..32 (vLLM default 4)')
    dense_kernels = validate_dense_kernels(dense_kernels)
    draft_head = validate_draft_head(draft_head)
    expert_overlay = expert_overlay_module.validate(expert_overlay)
    if type(expert_capture) is not bool:
        raise ValueError('expert_capture must be boolean')
    if type(expert_probe) is not bool or type(enforce_eager) is not bool:
        raise ValueError('expert_probe and enforce_eager must be boolean')
    if expert_prune is not None and (not isinstance(expert_prune, dict) or 'pruned' not in expert_prune):
        raise ValueError('expert_prune is a loaded mask (arena.serving.bundle_expert_prune)')
    if expert_probe and expert_capture:
        raise ValueError('expert_probe and expert_capture patch the same pinned MoE runner')
    if ready_guard_seconds is not None:
        recipe.wait_for_server = ready_guard(recipe, *ready_guard_seconds)
    if setup_timeouts is not None:
        # Before recipe.main(): wait_for_server's deadline, main()'s setup deadline and its SIGALRM read these globals.
        recipe.SERVER_READY_TIMEOUT, recipe.SETUP_TOTAL_TIMEOUT = (int(v) for v in setup_timeouts)
    serving_kernels = validate_serving_kernels(serving_kernels)
    stack_kernels = validate_stack_kernels(stack_kernels)
    if type(torch_profiler) is not bool:
        raise ValueError('torch_profiler must be boolean')
    if scheduling_policy is not None and scheduling_policy not in SCHEDULING_POLICIES:
        raise ValueError(f'Arena scheduling policy must be one of {SCHEDULING_POLICIES}')
    if type(graph_metrics) is not bool:
        raise ValueError('CUDA graph metrics must be boolean')
    if speculative is not None:
        speculative = validate_speculative(speculative)
    if type(reclaim_skipped_blocks) is not bool:
        raise ValueError('reclaim_skipped_blocks must be boolean')
    if type(cpu_gib) is not int or not 0 <= cpu_gib <= 32:
        raise ValueError('Arena CPU cache must be 0..32 GiB')
    if graph_capture_size is not None:
        if type(graph_capture_size) is not int or not 1 <= graph_capture_size <= 256:
            raise ValueError('Arena CUDA graph capture size must be an integer in 1..256')
        os.environ[recipe.MAX_CUDAGRAPH_CAPTURE_SIZE_ENV] = str(graph_capture_size)
    graph_capture_sizes = validate_graph_capture_sizes(graph_capture_sizes, graph_capture_size)
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"  # localhost-only cache reset after serving probes
    extract = recipe.verify_and_extract_runtime

    def wrapped_extract(*args, **kwargs):
        result = extract(*args, **kwargs)
        apply(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working,
              **({'state_cotenant': True} if prefix_cache and prefix_cache.get('state_cotenant') else {}))
        if reclaim_skipped_blocks:
            apply_reclamation(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working)
        if prefix_cache is not None:
            # Prefix cache v2 (29 September): after the reclamation patch, which edits the same manager file.
            apply_prefix_cache(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working, prefix_cache)
        if lora is not None and lora.get('ple_queue_wait'):
            from .ple_queue_patch import apply as apply_ple_queue

            apply_ple_queue(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working,
                            lora['ple_queue_wait'])
        if warm_prompt_logprobs:
            from .prompt_logprobs_patch import apply as apply_prompt_logprobs

            apply_prompt_logprobs(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working)
        if dense_kernels is not None:
            # Kernel work, phase 2: the overlay changes the quant methods, which vLLM's AOT compile cache
            # key does not see; drop any compiled graph an earlier (control) server left in this container.
            cleared = clear_compile_cache(recipe.CACHE_ROOT)
            identity = apply_dense_kernels(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working,
                                           dense_kernels)
            (Path(working) / 'dense-kernels-start.json').write_text(json.dumps(
                {'identity': identity, 'cleared_compile_cache': cleared}, indent=2))
        if serving_kernels is not None:
            # Kernel work, phase 4: the HC fusion and the draft quant method change the compiled graph as well.
            cleared = clear_compile_cache(recipe.CACHE_ROOT)
            identity = apply_serving_kernels(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working,
                                             serving_kernels)
            (Path(working) / 'serving-kernels-start.json').write_text(json.dumps(
                {'identity': identity, 'cleared_compile_cache': cleared}, indent=2))
        if stack_kernels is not None:
            # Kernel work, phase 5: attention-metadata host code only (outside the compiled graph).
            identity = apply_stack_kernels(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working,
                                           stack_kernels)
            (Path(working) / 'stack-kernels-start.json').write_text(json.dumps({'identity': identity}, indent=2))
        if draft_head is not None:
            # Draft-head fine-tune (28 September): hidden-state dumps (eager) and/or a fine-tuned MTP head;
            # applied after the serving-kernels overlay, which may also have edited mtp.py.
            if draft_head['dump']:
                os.makedirs(DRAFT_DUMP_DIR, exist_ok=True)
            identity = apply_draft_head(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working,
                                        draft_head)
            (Path(working) / 'draft-head-start.json').write_text(json.dumps({'identity': identity}, indent=2))
        if expert_capture:
            # Expert quantization (28 September): MoE input rows and the served routing, captured inside the MoE
            # custom op (opaque to torch.compile) when switched on through /collective_rpc.
            apply_expert_capture(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working)
        if expert_probe:
            # Expert-usage measurement (28 September): routing statistics and a router mask inside the MoE
            # custom op (opaque to torch.compile), switched through /collective_rpc.
            apply_expert_probe(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working)
        if expert_prune is not None:
            # Load-time expert pruning (28 September): the model is built with fewer experts, so any compiled graph an
            # earlier server left in this container has the wrong shapes.
            cleared = clear_compile_cache(recipe.CACHE_ROOT)
            identity = apply_expert_prune(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working,
                                          expert_prune)
            (Path(working) / 'expert-prune-start.json').write_text(json.dumps(
                {'identity': identity, 'cleared_compile_cache': cleared}, indent=2))
        if stall_dump_seconds is not None:
            # Diagnostics only (no pruning): the loader arms the one-shot per-step dump on the worker class.
            identity = apply_stall_dump(recipe.RUNTIME_ROOT / 'usr/local/lib/python3.12/dist-packages', working,
                                        stall_dump_seconds)
            (Path(working) / 'stall-dump-start.json').write_text(json.dumps({'identity': identity}, indent=2))
        return result

    recipe.verify_and_extract_runtime = wrapped_extract
    environment = getattr(recipe, 'runtime_environment', None)
    ple_uva = serving_kernels is not None and ple_target(serving_kernels) is not None
    kv8 = serving_kernels is not None and bool(serving_kernels.get('kv_cache_8bit'))
    if ple_uva and environment is None:
        raise ValueError('ple_uva needs the recipe runtime_environment')
    if environment is not None and (ple_uva or kv8 or lora is not None):
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
            if kv8:
                from .kv8_checks import record_runtime_env

                record_runtime_env(working, sys.executable, env)
            return env, check

        recipe.runtime_environment = wrapped_environment
    command = recipe.server_command
    overlays = {}  # base model path -> materialised expert-overlay path (one check per server start)

    def wrapped_command(*args, **kwargs):
        argv = command(*args, **kwargs)
        if graph_metrics:
            if '--disable-log-stats' in argv:
                raise ValueError('CUDA graph metrics require enabled log stats')
            argv.append('--cudagraph-metrics')
        if cpu_gib:
            argv.extend(['--kv-transfer-config', json.dumps({
                'kv_connector': 'OffloadingConnector', 'kv_role': 'kv_both',
                'kv_connector_extra_config': {'cpu_bytes_to_use': cpu_gib * 1024**3},
            }), '--no-disable-hybrid-kv-cache-manager'])
        if speculative is not None:
            if '--speculative-config' in argv:
                raise ValueError('n-gram speculation replaces the MTP draft: launch with MTP depth 0')
            argv.extend(['--speculative-config', json.dumps(speculative, separators=(',', ':'))])
        if scheduling_policy is not None:
            # fixed-2x-commit: lane requests carry priority 1 and yield to focus requests (priority 0)
            # when vLLM must preempt. Recorded in vllm-server-identity.json (argv) and the server log.
            if '--scheduling-policy' in argv:
                raise ValueError('The pinned serving command already sets a scheduling policy')
            argv.extend(['--scheduling-policy', scheduling_policy])
        if serving_kernels is not None:
            spec_tokens = 0
            if '--speculative-config' in argv:  # the draft length sizes the 4-bit block (kv4_block_tokens)
                spec_tokens = int(json.loads(argv[argv.index('--speculative-config') + 1]).get('num_speculative_tokens', 0))
            flags = serving_kernel_flags(serving_kernels, spec_tokens)
            for name, value in zip(flags[::2], flags[1::2]):
                if name in argv:
                    raise ValueError(f'The pinned serving command already sets {name}')
                argv.extend([name, value])
        if draft_head is not None:
            for flag in draft_head_flags(draft_head):
                if flag not in argv:
                    argv.append(flag)
        if expert_overlay is not None:
            # Expert quantization (28 September): the checkpoint with replacement routed-expert shards (a symlink
            # farm; same names, dtypes and shapes, so vLLM and its kernels are unchanged).
            from arc_infra.serving import dataset_mount

            position = argv.index('serve') + 1
            if argv[position] not in overlays:
                record = expert_overlay_module.materialise(
                    argv[position], dataset_mount(expert_overlay['dataset'], input_root),
                    Path(working) / 'expert-overlay-checkpoint', expert_overlay['manifest_sha256'],
                    check=expert_overlay.get('check', 'sha256'),
                    expert_prune_sha256=expert_prune_sha256 if expert_prune is not None else None)
                (Path(working) / expert_overlay_module.RECORD).write_text(json.dumps(record, indent=2))
                overlays[argv[position]] = record['dest']
            argv[position] = overlays[argv[position]]
        if torch_profiler:
            if '--profiler-config' in argv:
                raise ValueError('The pinned serving command already sets a profiler config')
            argv.extend(['--profiler-config', json.dumps(profiler_config(Path(working) / 'torch-profile'),
                                                         separators=(',', ':'))])
        if enforce_eager and '--enforce-eager' not in argv:
            argv.append('--enforce-eager')
        if prompt_tokens_details and '--enable-prompt-tokens-details' not in argv:
            argv.append('--enable-prompt-tokens-details')
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
    result = {'cpu_cache_gib': cpu_gib, 'layout_export': True, 'shared_prefix_discount': False}
    if graph_capture_size is not None:
        result['max_cudagraph_capture_size'] = graph_capture_size
    if graph_capture_sizes is not None:
        result['cudagraph_capture_sizes'] = graph_capture_sizes
    if reclaim_skipped_blocks:
        result['reclaim_skipped_blocks'] = True
    if prefix_cache is not None:
        result['prefix_cache'] = prefix_cache
    if prompt_tokens_details:
        result['prompt_tokens_details'] = True
    if warm_prompt_logprobs:
        result['warm_prompt_logprobs'] = True
    if prefix_match_unit is not None:
        result['prefix_match_unit'] = prefix_match_unit
    if renderer_num_workers is not None:
        result['renderer_num_workers'] = renderer_num_workers
    if mm_processor_cache_gb is not None:
        result['mm_processor_cache_gb'] = mm_processor_cache_gb
    if lora is not None:
        result['lora'] = lora
    if graph_metrics:
        result['cudagraph_metrics'] = True
    if speculative is not None:
        result['speculative'] = speculative
    if scheduling_policy is not None:
        result['scheduling_policy'] = scheduling_policy
    if torch_profiler:
        result['torch_profiler'] = profiler_config(Path(working) / 'torch-profile')
    if dense_kernels is not None:
        result['dense_kernels'] = dense_kernels
    if serving_kernels is not None:
        result['serving_kernels'] = serving_kernels
    if expert_probe:
        result['expert_probe'] = True
    if enforce_eager:
        result['enforce_eager'] = True
    if ready_guard_seconds is not None:
        result['ready_guard_seconds'] = list(ready_guard_seconds)
    if expert_prune is not None:
        result['expert_prune'] = {'fraction': expert_prune['fraction'], **expert_prune_kept(expert_prune)}
    if stall_dump_seconds is not None:
        result['stall_dump_seconds'] = stall_dump_seconds
    if stack_kernels is not None:
        result['stack_kernels'] = stack_kernels
    if draft_head is not None:
        result['draft_head'] = draft_head
    if expert_capture:
        result['expert_capture'] = True
    if expert_overlay is not None:
        result['expert_overlay'] = expert_overlay
    if research_checkpoint is not None:
        from .research_checkpoint import configure as configure_checkpoint
        result['research_checkpoint'] = configure_checkpoint(recipe, working, research_checkpoint)
    return result


def bundle_expert_prune(spec, bundle):
    """``server.expert_prune`` -> the loaded mask (its bundle-relative file checked against the pinned sha256)."""
    spec = validate_expert_prune(spec)
    return None if spec is None else load_prune_mask(Path(bundle) / spec['mask'], spec['sha256'])


STALL_DUMP_RANGE = (30, 240)


def with_stall_dump(mask, seconds):
    if seconds is None:
        return mask
    if mask is None:
        raise ValueError('ARENA_STALL_DUMP rides on the expert-prune overlay (server.expert_prune)')
    return {**mask, 'stall_dump_seconds': seconds}


def bundle_draft_head(spec, bundle):
    """A relative ``server.draft_head.overlay.path`` names a directory inside the bundle."""
    if not spec or not spec.get('overlay') or Path(spec['overlay']['path']).is_absolute():
        return spec
    return {**spec, 'overlay': {**spec['overlay'], 'path': str(Path(bundle) / spec['overlay']['path'])}}


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
    prune_spec = lab['expert_prune'] if 'expert_prune' in lab else study['server'].get('expert_prune')
    # Backend capacity and ordinary admission slots are distinct when siblings use spare sequences.
    sequences = study['server'].get('max_num_seqs', study['generation_slots'])
    if type(sequences) is not int or sequences < study['generation_slots']:
        raise ValueError('Server max_num_seqs must cover ordinary generation slots')
    samples = study.get('parallel_samples') or {}
    if samples and sequences != samples['max_num_seqs']:
        raise ValueError('Sampling and server sequence limits disagree')
    start(args.working, args.input_root, generation_slots=sequences, kv_cache_memory_bytes=kv_bytes,
          gpu_memory_utilization=utilization,
          mtp_tokens=lab.get('mtp_tokens', study['server'].get('mtp_tokens', 3)),
          kv_cache_dtype=lab.get('kv_cache_dtype', settings['kv_cache_dtype']),
          max_num_batched_tokens=study['server'].get('max_num_batched_tokens', 8192),
          prefix_caching=lab.get('prefix_caching'),
          recipe_hook=lambda recipe, working: configure(
              recipe, working, cpu_gib=settings['cpu_cache_gib'],
              graph_capture_size=lab.get('graph_capture_size', study['server'].get('max_cudagraph_capture_size')),
              graph_capture_sizes=(lab['graph_capture_sizes'] if 'graph_capture_sizes' in lab
                                   else study['server'].get('cudagraph_capture_sizes')),
              graph_metrics=study['server'].get('cudagraph_metrics', False),
              reclaim_skipped_blocks=settings.get('reclaim_skipped_blocks', False),
              speculative=study['server'].get('speculative'),
              scheduling_policy=study['server'].get('scheduling_policy'),
              torch_profiler=lab.get('torch_profiler', False),
              dense_kernels=lab['dense_kernels'] if 'dense_kernels' in lab else study['server'].get('dense_kernels'),
              serving_kernels=(lab['serving_kernels'] if 'serving_kernels' in lab
                               else study['server'].get('serving_kernels')),
              stack_kernels=(lab['stack_kernels'] if 'stack_kernels' in lab
                             else study['server'].get('stack_kernels')),
              draft_head=bundle_draft_head(study['server'].get('draft_head'), Path(args.study).parent.parent),
              expert_capture=study['server'].get('expert_capture', False),
              expert_overlay=(lab['expert_overlay'] if 'expert_overlay' in lab
                              else study['server'].get('expert_overlay')),
              input_root=args.input_root,
              expert_probe=lab.get('expert_probe', study['server'].get('expert_probe', False)),
              enforce_eager=lab.get('enforce_eager', False),
              prefix_cache=lab['prefix_cache'] if 'prefix_cache' in lab else settings.get('prefix_cache'),
              prompt_tokens_details=lab.get('prompt_tokens_details',
                                            study['server'].get('prompt_tokens_details', False)),
              prefix_match_unit=lab.get('prefix_match_unit', study['server'].get('prefix_match_unit')),
              renderer_num_workers=study['server'].get('renderer_num_workers'),
              mm_processor_cache_gb=study['server'].get('mm_processor_cache_gb'),
              lora=study['server'].get('lora'),
              warm_prompt_logprobs=lab.get('warm_prompt_logprobs', False),
              ready_guard_seconds=lab.get('ready_guard'),
              setup_timeouts=lab.get('setup_timeouts'),
              expert_prune=with_stall_dump(bundle_expert_prune(prune_spec, Path(args.study).parent.parent),
                                           lab.get('stall_dump_seconds')),
              expert_prune_sha256=(prune_spec or {}).get('sha256'),
              research_checkpoint=study.get('research_checkpoint')))
    budget = BlockBudget.load(args.working, study['kv_margin'])
    (Path(args.working) / 'cache-layout.json').write_text(json.dumps(budget.layout, indent=2))


if __name__ == '__main__':
    main()
