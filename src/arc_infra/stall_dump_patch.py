"""Hash-guarded stall-only worker stack dump without expert pruning (Holo4-27B, 29 September).

hero-v2's ``stall_dump`` component rides on the expert-prune overlay: ``expert_prune_runtime.prepare_config`` arms
``install_stall_dump`` when the mask carries ``stall_dump_seconds``. A model with no routed experts (Holo4-27B, a
dense Qwen3.8-27B fine-tune) has no mask, so this overlay patches only the pinned ``base_loader.py`` of the prune
overlay (same sha256, same anchor, right before ``initialize_model`` in the worker) to arm the same one-shot
per-step ``faulthandler`` timer on the worker class: all-thread Python stacks to stderr (the server log) only when
one ``execute_model`` / ``sample_tokens`` call runs ``seconds``. Nothing in the model, the loader's weights or the
forward changes. The helper is the prune overlay's own runtime module, copied as ``vllm/arc_stall_dump.py`` (stdlib
at import; its pruning settings are never read here).

The two overlays edit the same file, so they are exclusive: a pruned server keeps the stall dump on its mask
(``arena.serving``).
"""

import hashlib
import json
from pathlib import Path

from .expert_prune_patch import BASE_LOADER

HELPER = 'vllm/arc_stall_dump.py'
MARKER = 'stall-dump-patch.json'
RANGE = (30, 240)
ANCHOR = '        with set_default_torch_dtype(model_config.dtype):\n'


def edit(seconds):
    return ('        # ARC (stall dump, no pruning): stacks when one worker step stalls; diagnostics only.\n'
            '        from vllm.arc_stall_dump import install_stall_dump as _arc_install_stall_dump\n'
            '        from vllm.arc_stall_dump import worker_class as _arc_worker_class\n'
            f'        _arc_install_stall_dump({int(seconds)}, _arc_worker_class(vllm_config), log=lambda text, **kw: '
            "print(text.replace('ARC_EXPERT_PRUNE ', 'ARC_STALL_DUMP ', 1), **kw))\n" + ANCHOR)


def validate(seconds):
    if seconds is None:
        return None
    if type(seconds) is not int or not RANGE[0] <= seconds <= RANGE[1]:
        raise ValueError(f'stall_dump_seconds is {RANGE[0]}..{RANGE[1]} seconds')
    return seconds


def patched_source(original, seconds):
    if hashlib.sha256(original).hexdigest() != BASE_LOADER['sha256']:
        raise RuntimeError(f"Stall-dump overlay rejected an unrecognized {BASE_LOADER['relative']}")
    text = original.decode()
    if text.count(ANCHOR) != 1:
        raise RuntimeError(f"Stall-dump overlay anchor count changed in {BASE_LOADER['relative']}")
    text = text.replace(ANCHOR, edit(seconds))
    compile(text, BASE_LOADER['relative'], 'exec')
    return text.encode()


def apply(site, output, seconds):
    """Patch the extracted runtime under ``site``; identity to ``output``. A restart re-extracts the pinned file
    first; an already patched file is accepted only if it is this exact overlay."""
    seconds = validate(seconds)
    if seconds is None:
        raise ValueError('stall-dump overlay applied without seconds')
    site = Path(site)
    helper = Path(__file__).with_name('expert_prune_runtime.py').read_bytes()
    target = site / BASE_LOADER['relative']
    original = target.read_bytes()
    marker = Path(output) / MARKER
    digest = hashlib.sha256(original).hexdigest()
    if digest == BASE_LOADER['sha256']:
        patched = patched_source(original, seconds)
        (site / HELPER).write_bytes(helper)
        target.write_bytes(patched)
        identity = {'file': BASE_LOADER['relative'], 'original_sha256': digest,
                    'patched_sha256': hashlib.sha256(patched).hexdigest(),
                    'helper_sha256': hashlib.sha256(helper).hexdigest(), 'stall_dump_seconds': seconds}
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(identity, indent=2) + '\n')
        print('ARC_STALL_DUMP_PATCH ' + json.dumps(identity, sort_keys=True), flush=True)
        return identity
    if marker.exists():
        identity = json.loads(marker.read_text())
        if (digest == identity['patched_sha256'] and identity['stall_dump_seconds'] == seconds
                and (site / HELPER).read_bytes() == helper):
            return identity
    raise RuntimeError('Stall-dump overlay found a modified or partially patched pinned base_loader.py')
