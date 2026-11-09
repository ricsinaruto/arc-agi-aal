"""Hash-guarded optional dense-kernel overlay (kernel work, phase 2); pinned sources stay untouched.

Inserts one call at the top of the pinned ``enable_qwen38next_low_latency_gemm`` that hands the model
to ``arc_dense_kernels.enable`` (``dense_kernels_runtime.py``, copied next to it with its settings in
``arc_dense_kernels.json``). Refuses to patch any ``low_latency_gemm.py`` but the pinned one.

Settings (a study's ``server.dense_kernels``; absent means the pinned runtime is used unchanged):
``{"skinny": bool, "int8_weights": [family, ...]}`` with families from ``INT8_FAMILIES``, plus the optional
``"mid_m"`` (kernel work, phase 5): ``true`` tunes the decode-size Triton tiles at load, or a fixed plan
``{"bf16": {"N,K": {"M": [BLOCK_N, BLOCK_K, warps, stages] | null}}, "int8": {...}}`` (``dense_kernels_runtime``).

Kernel work, phase 6 (docs/validation-20260927/kernels-phase6.md), each absent unless set, so the earlier settings,
the patched file and the copied helper stay byte-identical:
``"fused_splitk": true`` (the split-K reduction in the GEMM's last CTA per tile, bitwise equal to the two-launch
path), ``"mid_int8_buckets": [8, 16, 24, 32]`` (INT8 mid-M buckets; phase 5 has 24 and 32) and
``"mid_extra_candidates": [[BLOCK_N, BLOCK_K, warps, stages], ...]`` (more tuner candidates). They are served by
``dense_fused_runtime.py`` (copied to ``arc_dense_fused.py`` only with them), installed by one extra line.

Kernel profile (30 September; docs/validation-20260930/kernel-profile.md), likewise absent unless set:
``"mid_wide": true`` (mid-M buckets 40 / 48 / 56 for both kinds, tuned in the kernel warmup where the fixed plan has
no tile), ``"mid_wide_shapes": [[N, K], ...]`` (extra BF16 shapes on the mid-M path, e.g. the prune12 router
[448, 2560]) and ``"mid_wide_plan"`` (fixed wide tiles, {"bf16"|"int8": {"N,K": {"40": [BLOCK_M, BLOCK_N, BLOCK_K,
warps, stages] | null}}}). They are served by ``dense_wide_runtime.py`` (copied to ``arc_dense_wide.py`` only with
them), installed by its own extra line.
"""

import hashlib
import json
from pathlib import Path

RELATIVE = 'vllm/models/qwen3_8_flash_next/nvidia/low_latency_gemm.py'
ORIGINAL_SHA256 = '46fcbdd4567864fce21a6f18a225d461bc49a459886bd09a11bea476d14f73e6'
ANCHOR = '    if dtype != torch.bfloat16 or not _is_sm103():\n        return\n'
INSERT = '''    # ARC optional dense kernels (skinny GEMM at TP=1 on SM120, weight-only INT8 families).
    from vllm.models.qwen3_8_flash_next.nvidia.arc_dense_kernels import enable as _arc_dense_enable
    if _arc_dense_enable(module, dtype):
        return
'''
HELPER = 'arc_dense_kernels.py'
# Phase 6: installed before ``enable`` only when a phase-6 key is set (the helper is copied only then too).
INSERT_PHASE6 = '''    # ARC dense kernels, phase 6 (fused split-K reduction, INT8 mid-M buckets, tuner candidates).
    from vllm.models.qwen3_8_flash_next.nvidia.arc_dense_fused import install as _arc_dense_fused_install
    _arc_dense_fused_install()
'''
PHASE6_HELPER = 'arc_dense_fused.py'
PHASE6_SOURCE = 'dense_fused_runtime.py'
PHASE6_KEYS = ('fused_splitk', 'mid_int8_buckets', 'mid_extra_candidates')
# Kernel profile (30 September): the wide decode tiles, installed before ``enable`` only with a wide key.
INSERT_WIDE = '''    # ARC dense kernels, wide decode tiles (mid-M buckets 40-56, extra BF16 shapes).
    from vllm.models.qwen3_8_flash_next.nvidia.arc_dense_wide import install as _arc_dense_wide_install
    _arc_dense_wide_install()
'''
WIDE_HELPER = 'arc_dense_wide.py'
WIDE_SOURCE = 'dense_wide_runtime.py'
WIDE_KEYS = ('mid_wide', 'mid_wide_shapes', 'mid_wide_plan', 'mid_wide_buckets')
WIDE_BUCKETS = (40, 48, 56)
# Throughput arm (3 October): MTP-1 verify batches are 2 x streams rows (60-68 at 30-34 streams), past the 56 bucket.
# ``mid_wide_buckets`` replaces the default wide buckets with any increasing subset of these (absent = the default).
WIDE_BUCKETS_ALLOWED = tuple(range(40, 193, 8))  # to 2 x 96 sequences
# The prune12 router: 448 kept experts x hidden 2560, in no mid-M plan (the phase-5 plan lists the unpruned 512).
PRUNED_ROUTER_SHAPES = ([448, 2560],)
SETTINGS = 'arc_dense_kernels.json'
MARKER = 'dense-kernels-patch.json'
# The large projection families where phase 1 measured weight-only 8-bit faster than BF16.
INT8_FAMILIES = ('lm_head', 'gdn_in_proj_qkvz', 'qsa_qkv', 'attn_out')
# The phase-2 A+C setting (docs/validation-20260927/kernels-phase2.md).
PHASE2 = {'skinny': True, 'int8_weights': list(INT8_FAMILIES)}
# Kernel work, phase 5 (docs/validation-20260927/kernels-phase5-stack.md): A+C plus the decode-size ("mid-M") Triton
# tiles. The plan is the one measured by the load-time tuner in the stack smoke (kernel-stack-phase5-v1: two server
# starts, candidates timed against the pinned kernels on real weights, pooled; a tile is kept when >= 3% faster in the
# mean and within the error bound in both); a fixed plan is only error-checked at start-up, so runs are reproducible.
# "N,K" -> {"M bucket": [BLOCK_N, BLOCK_K, warps, stages] or None (the pinned kernel)}.
MID_PLAN_PHASE5 = {
    'bf16': {
        '10240,320': {'8': [64, 128, 4, 3], '16': [64, 128, 4, 3], '24': None, '32': [64, 128, 4, 3]},
        '1280,2560': {'8': None, '16': [16, 256, 4, 3], '24': [16, 256, 4, 3], '32': [16, 256, 4, 3]},
        '2560,2560': {'8': None, '16': [32, 256, 4, 3], '24': [16, 256, 4, 3], '32': [64, 128, 4, 3]},
        '2560,640': {'8': None, '16': [32, 256, 4, 3], '24': [16, 256, 4, 3], '32': [32, 256, 4, 3]},
        '320,10240': {'8': [32, 128, 4, 4], '16': [32, 128, 4, 4], '24': [32, 128, 4, 4], '32': [32, 128, 4, 4]},
        '336,10240': {'8': None, '16': [32, 128, 4, 4], '24': [32, 128, 4, 4], '32': [32, 128, 4, 4]},
        '512,2560': {'8': None, '16': [16, 256, 4, 3], '24': [16, 256, 4, 3], '32': [16, 256, 4, 3]},
        '640,2560': {'8': None, '16': [16, 256, 4, 3], '24': [16, 256, 4, 3], '32': [16, 256, 4, 3]},
        '96,2560': {'8': None, '16': [32, 128, 4, 4], '24': [32, 128, 4, 4], '32': [32, 128, 4, 4]},
    },
    'int8': {
        '13312,2560': {'24': [64, 128, 4, 3], '32': [64, 128, 4, 3]},
        '16384,2560': {'24': [32, 128, 4, 4], '32': [32, 256, 4, 3]},
        '248320,2560': {'24': None, '32': None},
        '2560,6144': {'24': [16, 256, 4, 3], '32': [32, 128, 4, 4]},
    },
}
PHASE5 = {**PHASE2, 'mid_m': MID_PLAN_PHASE5}
# Kernel work, phase 6: the hero's PHASE5 plan with the fused split-K reduction (bitwise equal: no re-tuning needed).
PHASE6_FUSED = {**PHASE5, 'fused_splitk': True}
# Phase 6 tuning run (GPU smoke only): the load-time tuner over every mid-M bucket for both kinds, with the extra
# candidates below, on the fused path; its logged choices become the next fixed plan.
MID_EXTRA_CANDIDATES_PHASE6 = [[16, 128, 4, 4], [32, 64, 4, 4], [64, 64, 4, 4], [16, 512, 4, 2]]
PHASE6_TUNE = {**PHASE2, 'mid_m': True, 'fused_splitk': True, 'mid_int8_buckets': [8, 16, 24, 32],
               'mid_extra_candidates': MID_EXTRA_CANDIDATES_PHASE6}
# Buckets of a mid-M plan: vLLM's decode CUDA-graph sizes above 4 (the served capture sizes are 1, 2, 4, 8, 16, 24, 32).
MID_M_BUCKETS = (8, 16, 24, 32)
MID_KINDS = ('bf16', 'int8')


def validate_mid_plan(plan):
    """A fixed mid-M plan: {"bf16"|"int8": {"N,K": {"M": [BLOCK_N, BLOCK_K, warps, stages] or null}}}."""
    if not isinstance(plan, dict) or not plan or set(plan) - set(MID_KINDS):
        raise ValueError(f'dense_kernels.mid_m plan takes only {MID_KINDS}')
    result = {}
    for kind in MID_KINDS:
        shapes = plan.get(kind, {})
        if not isinstance(shapes, dict):
            raise ValueError(f'dense_kernels.mid_m.{kind} must map "N,K" to buckets')
        out = {}
        for key, buckets in shapes.items():
            parts = key.split(',') if isinstance(key, str) else []
            if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts) or not isinstance(buckets, dict):
                raise ValueError(f'dense_kernels.mid_m.{kind}: bad shape key {key!r}')
            entry = {}
            for m, cfg in buckets.items():
                if not (isinstance(m, str) and m.isdigit() and int(m) in MID_M_BUCKETS):
                    raise ValueError(f'dense_kernels.mid_m.{kind}.{key}: bucket {m!r} not in {MID_M_BUCKETS}')
                if cfg is not None and not (isinstance(cfg, list) and len(cfg) == 4
                                            and all(type(v) is int and v > 0 for v in cfg)):
                    raise ValueError(f'dense_kernels.mid_m.{kind}.{key}.{m}: [BLOCK_N, BLOCK_K, warps, stages] or null')
                entry[m] = None if cfg is None else list(cfg)
            out[key] = {m: entry[m] for m in sorted(entry, key=int)}
        result[kind] = {k: out[k] for k in sorted(out)}
    return result


def validate_phase6(spec, mid):
    """The phase-6 keys of a ``server.dense_kernels`` value, canonical and without the absent/false ones."""
    result = {}
    fused = spec.get('fused_splitk', False)
    if type(fused) is not bool:
        raise ValueError('dense_kernels.fused_splitk must be a boolean')
    if fused:
        result['fused_splitk'] = True
    buckets = spec.get('mid_int8_buckets') or []
    if not isinstance(buckets, list) or any(type(b) is not int or b not in MID_M_BUCKETS for b in buckets) \
            or len(set(buckets)) != len(buckets):
        raise ValueError(f'dense_kernels.mid_int8_buckets must be distinct buckets from {MID_M_BUCKETS}')
    candidates = spec.get('mid_extra_candidates') or []
    if not isinstance(candidates, list) or any(
            not (isinstance(c, list) and len(c) == 4 and all(type(v) is int and v > 0 for v in c)) for c in candidates):
        raise ValueError('dense_kernels.mid_extra_candidates must be [BLOCK_N, BLOCK_K, warps, stages] lists')
    if (buckets or candidates) and not mid:
        raise ValueError('dense_kernels.mid_int8_buckets / mid_extra_candidates extend mid_m; enable it')
    if candidates and mid is not True:
        raise ValueError('dense_kernels.mid_extra_candidates only feed the load-time tuner (mid_m: true)')
    if buckets:
        result['mid_int8_buckets'] = sorted(buckets)
    if candidates:
        result['mid_extra_candidates'] = [list(c) for c in candidates]
    return result


def validate_wide(spec, mid):
    """The wide-tile keys of a ``server.dense_kernels`` value, canonical and without the absent/false ones."""
    on = spec.get('mid_wide', False)
    shapes = spec.get('mid_wide_shapes') or []
    plan = spec.get('mid_wide_plan') or {}
    wide_buckets = spec.get('mid_wide_buckets')
    if type(on) is not bool:
        raise ValueError('dense_kernels.mid_wide must be a boolean')
    if wide_buckets is not None and (not isinstance(wide_buckets, list) or not wide_buckets
                                     or any(type(b) is not int or b not in WIDE_BUCKETS_ALLOWED for b in wide_buckets)
                                     or wide_buckets != sorted(set(wide_buckets))):
        raise ValueError(f'dense_kernels.mid_wide_buckets must be an increasing list from {WIDE_BUCKETS_ALLOWED}')
    allowed = tuple(wide_buckets) if wide_buckets is not None else WIDE_BUCKETS
    if not isinstance(shapes, list) or any(not (isinstance(s, list) and len(s) == 2
                                               and all(type(v) is int and v > 0 for v in s)) for s in shapes):
        raise ValueError('dense_kernels.mid_wide_shapes must be [N, K] lists')
    if not isinstance(plan, dict) or set(plan) - set(MID_KINDS):
        raise ValueError(f'dense_kernels.mid_wide_plan takes only {MID_KINDS}')
    canonical_plan = {}
    for kind in MID_KINDS:
        entries = plan.get(kind) or {}
        if not isinstance(entries, dict):
            raise ValueError(f'dense_kernels.mid_wide_plan.{kind} must map "N,K" to buckets')
        out = {}
        for key, buckets in entries.items():
            parts = key.split(',') if isinstance(key, str) else []
            if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts) or not isinstance(buckets, dict):
                raise ValueError(f'dense_kernels.mid_wide_plan.{kind}: bad shape key {key!r}')
            entry = {}
            for m, cfg in buckets.items():
                if not (isinstance(m, str) and m.isdigit() and int(m) in allowed):
                    raise ValueError(f'dense_kernels.mid_wide_plan.{kind}.{key}: bucket {m!r} not in {allowed}')
                if cfg is not None and not (isinstance(cfg, list) and len(cfg) == 5
                                            and all(type(v) is int and v > 0 for v in cfg) and cfg[0] in (32, 64)):
                    raise ValueError(f'dense_kernels.mid_wide_plan.{kind}.{key}.{m}: [BLOCK_M (32|64), BLOCK_N, '
                                     'BLOCK_K, warps, stages] or null')
                entry[m] = None if cfg is None else list(cfg)
            out[key] = {m: entry[m] for m in sorted(entry, key=int)}
        if out:
            canonical_plan[kind] = {k: out[k] for k in sorted(out)}
    if (shapes or canonical_plan or wide_buckets is not None) and not on:
        raise ValueError('dense_kernels.mid_wide_shapes / mid_wide_plan / mid_wide_buckets extend mid_wide; enable it')
    if on and not mid:
        raise ValueError('dense_kernels.mid_wide extends mid_m; enable it')
    result = {}
    if on:
        result['mid_wide'] = True
    if shapes:
        result['mid_wide_shapes'] = [list(s) for s in sorted({tuple(s) for s in shapes})]
    if canonical_plan:
        result['mid_wide_plan'] = canonical_plan
    if wide_buckets is not None:
        result['mid_wide_buckets'] = list(wide_buckets)
    return result


def wide(spec):
    """True when an (already validated) setting uses a wide-tile key (helper and install line needed)."""
    return any(key in spec for key in WIDE_KEYS)


def validate(spec):
    """Normalize a ``server.dense_kernels`` value; None/absent/all-off means no overlay."""
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) - {'skinny', 'int8_weights', 'mid_m', *PHASE6_KEYS, *WIDE_KEYS}:
        raise ValueError('dense_kernels takes only "skinny" (bool), "int8_weights" (list of families), "mid_m" and '
                         f'the phase-6 keys {PHASE6_KEYS}')
    skinny = spec.get('skinny', False)
    families = spec.get('int8_weights', [])
    mid = spec.get('mid_m', False)
    if type(skinny) is not bool:
        raise ValueError('dense_kernels.skinny must be a boolean')
    if not isinstance(families, list) or any(f not in INT8_FAMILIES for f in families) \
            or len(set(families)) != len(families):
        raise ValueError(f'dense_kernels.int8_weights must be distinct families from {INT8_FAMILIES}')
    if type(mid) is not bool:
        mid = validate_mid_plan(mid)
    phase6 = validate_phase6(spec, mid)
    wide_keys = validate_wide(spec, mid)
    if not skinny and not families:
        if mid or phase6 or wide_keys:
            raise ValueError('dense_kernels.mid_m and the phase-6 keys extend the skinny and INT8 paths; enable one')
        return None
    # Canonical order, so equal settings always serialize identically; an absent or false mid_m is left out, so the
    # phase-2 setting serializes exactly as before.
    result = {'skinny': skinny, 'int8_weights': [f for f in INT8_FAMILIES if f in families]}
    if mid:
        result['mid_m'] = mid
    result.update(phase6)
    result.update(wide_keys)
    return result


def phase6(spec):
    """True when an (already validated) setting uses a phase-6 key (helper and install line needed)."""
    return any(key in spec for key in PHASE6_KEYS)


def patched_source(original, with_phase6=False, with_wide=False):
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError('Dense-kernel overlay rejected an unrecognized vLLM low_latency_gemm.py')
    text = original.decode()
    if text.count(ANCHOR) != 1:
        raise RuntimeError('Dense-kernel overlay anchor is not unique')
    insert = INSERT.replace('    from vllm.models.qwen3_8_flash_next.nvidia.arc_dense_kernels import enable',
                            INSERT_PHASE6 + '    from vllm.models.qwen3_8_flash_next.nvidia.arc_dense_kernels import '
                            'enable', 1) if with_phase6 else INSERT
    if with_wide:
        # After the phase-6 install (it may rebind MID_INT8_BUCKETS / MID_CANDIDATES), before enable.
        insert = insert.replace('    from vllm.models.qwen3_8_flash_next.nvidia.arc_dense_kernels import enable',
                                INSERT_WIDE + '    from vllm.models.qwen3_8_flash_next.nvidia.arc_dense_kernels import '
                                'enable', 1)
    result = text.replace(ANCHOR, insert + ANCHOR)
    compile(result, RELATIVE, 'exec')
    return result.encode()


def apply(site, output, spec):
    """Patch the extracted runtime under ``site`` for ``spec``; the identity goes to ``output``."""
    spec = validate(spec)
    if spec is None:
        raise ValueError('No dense kernels requested; the pinned runtime must stay unpatched')
    target = Path(site) / RELATIVE
    helper = Path(__file__).with_name('dense_kernels_runtime.py').read_bytes()
    extra = Path(__file__).with_name(PHASE6_SOURCE).read_bytes() if phase6(spec) else None
    extra_wide = Path(__file__).with_name(WIDE_SOURCE).read_bytes() if wide(spec) else None
    settings = (json.dumps(spec, sort_keys=True) + '\n').encode()
    original = target.read_bytes()
    marker = Path(output) / MARKER
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
        # A restart in the same working directory re-extracts the pinned runtime first (cache_patch does the
        # same); anything else already patched with this exact overlay is accepted, never re-patched.
        if marker.exists():
            identity = json.loads(marker.read_text())
            if (hashlib.sha256(original).hexdigest() == identity['patched_sha256']
                    and target.with_name(HELPER).read_bytes() == helper
                    and target.with_name(SETTINGS).read_bytes() == settings
                    and (extra is None or (target.with_name(PHASE6_HELPER).is_file()
                                           and target.with_name(PHASE6_HELPER).read_bytes() == extra))
                    and (extra_wide is None or (target.with_name(WIDE_HELPER).is_file()
                                                and target.with_name(WIDE_HELPER).read_bytes() == extra_wide))):
                return identity
        raise RuntimeError('Dense-kernel overlay found a modified low_latency_gemm.py')
    patched = patched_source(original, with_phase6=extra is not None, with_wide=extra_wide is not None)
    target.with_name(HELPER).write_bytes(helper)
    if extra is not None:
        target.with_name(PHASE6_HELPER).write_bytes(extra)
    if extra_wide is not None:
        target.with_name(WIDE_HELPER).write_bytes(extra_wide)
    target.with_name(SETTINGS).write_bytes(settings)
    target.write_bytes(patched)
    identity = {'original_sha256': ORIGINAL_SHA256,
                'patched_sha256': hashlib.sha256(patched).hexdigest(),
                'helper_sha256': hashlib.sha256(helper).hexdigest(),
                'settings': spec}
    if extra is not None:
        identity['phase6_helper_sha256'] = hashlib.sha256(extra).hexdigest()
    if extra_wide is not None:
        identity['wide_helper_sha256'] = hashlib.sha256(extra_wide).hexdigest()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    print('ARC_DENSE_KERNELS_PATCH ' + json.dumps(identity, sort_keys=True), flush=True)
    return identity


def clear_compile_cache(cache_root):
    """Remove vLLM's torch.compile / AOT cache under ``cache_root`` (the serving XDG cache).

    vLLM's AOT cache key (env, config, forward hash) does not see a changed quant method, and its
    source check only covers the files the cached artifact traced. A control server started earlier in
    the same container would otherwise hand its compiled graph (plain BF16 linears) to this server.
    """
    import shutil

    removed = []
    for path in (Path(cache_root) / 'vllm' / 'torch_compile_cache',):
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    return removed
