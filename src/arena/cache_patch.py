"""Hash-guarded telemetry-only insertion into a disposable extracted vLLM runtime."""
import hashlib
import json
from pathlib import Path

RELATIVE = 'vllm/v1/core/kv_cache_utils.py'
ORIGINAL_SHA256 = 'f6643bad29b7b3cb044482b6736a042afa91a4f43a11f034317a8e8029d3a8a4'
ANCHOR = '    num_tokens, max_concurrency = get_kv_cache_capacity(vllm_config, kv_cache_config)\n'
INSERT = '    from vllm.v1.core.arc_cache_export import emit\n    emit(vllm_config, kv_cache_config)\n'
# state_cotenant (throughput arm, 7 October; prefix_cache option of that name). The CSA+linear layout cuts one main
# tensor per QSA layer and gives every Mamba group one layer per tensor. Served: 13 main tensors (12 QSA layers + the
# draft's), three GDN groups of 12 layers (so tensor 12 is idle in their blocks) and the PLE short-conv layer alone in
# a fourth group: three whole blocks per request (running, snapshot, speculative) for 3 x 184 KB. With the option the
# replicated group joins the first sharded Mamba group that has a spare tensor slot, as a UniformTypeKVCacheSpecs of
# per-layer MambaSpecs (the form the worker's mamba_utils / attn_utils already read per layer; the scheduler sees the
# first member, the GDN spec, as before). Same block table and lifetime, same padded page: the PLE pages are the
# idle tensor-12 pages of GDN group 0's blocks. Bit-exact; fixed blocks per request 13 -> 10 at MTP-1.
FOLD_ANCHOR = '''        groups.extend(
            KVCacheGroupSpec(group_names, padded_spec) for group_names in grouped_names
        )
    return groups
'''
FOLD_REPLACEMENT = '''        groups.extend(
            KVCacheGroupSpec(group_names, padded_spec) for group_names in grouped_names
        )
    return _arc_fold_replicated_mamba(groups, len(main_kv_names))


def _arc_fold_replicated_mamba(groups, num_slots):
    """ARC state_cotenant: fold each replicated Mamba group into a sharded one with spare main_kv tensor slots."""
    import json as _json

    def members(group):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            return dict(spec.kv_cache_specs)
        return {name: spec for name in group.layer_names}

    def policy(spec):
        return (spec.block_size, spec.page_size_bytes, spec.mamba_cache_mode, spec.num_speculative_blocks)

    folded = []
    record = []
    for extra in list(groups):
        spec = extra.kv_cache_spec
        if not (isinstance(spec, MambaSpec) and spec.tp_replicated):
            continue
        for host in groups:
            specs = members(host)
            first = next(iter(specs.values()), None)
            if (host is extra or not isinstance(first, MambaSpec) or first.tp_replicated
                    or len(specs) + len(extra.layer_names) > num_slots or policy(first) != policy(spec)):
                continue
            slot = len(specs)
            specs.update({name: spec for name in extra.layer_names})
            uniform = UniformTypeKVCacheSpecs.from_specs(specs)
            if uniform is None:
                continue
            host.layer_names = list(specs)
            host.kv_cache_spec = uniform
            folded.append(extra)
            record.append({"layers": extra.layer_names, "host_first": host.layer_names[0], "slot": slot})
            break
    if not folded:
        raise ValueError("ARC state_cotenant: no replicated Mamba group fits a sharded group's spare tensor slot")
    print("ARC_STATE_COTENANT " + _json.dumps({"folded": record, "slots": num_slots}), flush=True)
    return [group for group in groups if not any(group is extra for extra in folded)]
'''
LAYOUT_ANCHOR = '''        elif type(member) is FullAttentionSpec:
            compressed_sparse = spec.kv_cache_specs
        else:
            return None
'''
LAYOUT_REPLACEMENT = '''        elif type(member) is FullAttentionSpec:
            compressed_sparse = spec.kv_cache_specs
        elif isinstance(member, MambaSpec):  # ARC state_cotenant: a sharded group with its replicated co-tenant
            mamba_groups.append(group)
        else:
            return None
'''
MANAGER_RELATIVE = 'vllm/v1/core/kv_cache_manager.py'
MANAGER_SHA256 = '2d20c3d98845cfd8d88a2f66b8fc6402ea1fcfbee16879d2364a4a9e45b8fa47'


def patched_manager(original):
    if hashlib.sha256(original).hexdigest() != MANAGER_SHA256:
        raise RuntimeError('Cache diagnostics rejected unrecognized pinned vLLM manager')
    text = original.decode()
    anchors = [
        ('            if required_blocks > self.block_pool.get_num_free_blocks():\n',
         'full_sequence', 'self.block_pool.get_num_free_blocks()', '0', '                '),
        ('        if required_blocks > available_blocks:\n',
         'incremental', 'available_blocks', 'reserved_blocks', '            '),
    ]
    for anchor, stage, available, reserved, indent in anchors:
        if text.count(anchor) != 1:
            raise RuntimeError('Cache diagnostics insertion must have one anchor')
        insertion = (f'{indent}from vllm.v1.core.arc_cache_failure import emit_failure\n'
                     f'{indent}emit_failure(self, request, stage={stage!r},\n'
                     f'{indent}             required=required_blocks, available={available},\n'
                     f'{indent}             reserved={reserved}, watermark=watermark_blocks)\n')
        text = text.replace(anchor, anchor + insertion)
    compile(text, MANAGER_RELATIVE, 'exec')
    return text.encode()


def patched_source(original, state_cotenant=False):
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError('Cache telemetry rejected unrecognized pinned vLLM source')
    text = original.decode()
    if text.count(ANCHOR) != 1:
        raise RuntimeError('Cache telemetry insertion must have one anchor')
    result = text.replace(ANCHOR, INSERT + ANCHOR)
    if state_cotenant:
        for anchor, replacement in ((FOLD_ANCHOR, FOLD_REPLACEMENT), (LAYOUT_ANCHOR, LAYOUT_REPLACEMENT)):
            if result.count(anchor) != 1:
                raise RuntimeError('state_cotenant edit must have one anchor')
            result = result.replace(anchor, replacement)
    compile(result, RELATIVE, 'exec')
    return result.encode()


def apply(site, output, state_cotenant=False):
    target = Path(site) / RELATIVE
    helper = Path(__file__).with_name('cache_export_runtime.py').read_bytes()
    helper_target = target.with_name('arc_cache_export.py')
    marker = Path(output) / 'cache-layout-patch.json'
    original = target.read_bytes()
    manager_target = Path(site) / MANAGER_RELATIVE
    failure_helper = Path(__file__).with_name('cache_failure_runtime.py').read_bytes()
    failure_target = target.with_name('arc_cache_failure.py')
    if marker.exists():
        identity = json.loads(marker.read_text())
        if bool(identity.get('state_cotenant')) != bool(state_cotenant):
            raise RuntimeError('state_cotenant changed within one working directory')
        if (hashlib.sha256(original).hexdigest() == identity['patched_sha256']
                and helper_target.read_bytes() == helper
                and hashlib.sha256(manager_target.read_bytes()).hexdigest() == identity.get('manager_patched_sha256')
                and failure_target.read_bytes() == failure_helper):
            return identity
        if hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256:
            raise RuntimeError('Cache telemetry runtime identity changed')
        # A server restart in the same working directory re-extracts the pinned runtime (serving A/B
        # bench, 26 September): the files are the pinned originals again, so the same patch re-applies.
    patched = patched_source(original, state_cotenant)
    manager_patched = patched_manager(manager_target.read_bytes())
    helper_target.write_bytes(helper)
    failure_target.write_bytes(failure_helper)
    target.write_bytes(patched)
    manager_target.write_bytes(manager_patched)
    identity = {'original_sha256': ORIGINAL_SHA256, **({'state_cotenant': True} if state_cotenant else {}),
                'patched_sha256': hashlib.sha256(patched).hexdigest(),
                'helper_sha256': hashlib.sha256(helper).hexdigest(),
                'manager_original_sha256': MANAGER_SHA256,
                'manager_patched_sha256': hashlib.sha256(manager_patched).hexdigest(),
                'failure_helper_sha256': hashlib.sha256(failure_helper).hexdigest()}
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    return identity
