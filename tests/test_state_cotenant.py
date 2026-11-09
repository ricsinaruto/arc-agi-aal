"""state_cotenant (arena.cache_patch): the PLE short-conv state as a co-tenant of a GDN group's blocks."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from arena import cache_patch
from arena.prefix_cache_patch import validate

PINNED_SITE = Path(os.environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))
PINNED_PYTHON = os.environ.get("ARC_PINNED_PYTHON", sys.executable)  # an interpreter with the runtime's imports
BASE = {"mamba_align_fix": True, "mamba_keep_state": True, "mamba_demote_superseded": True}


def test_option_is_optional_and_normalized_away_when_off():
    assert "state_cotenant" not in validate({**BASE, "state_cotenant": False})
    assert validate({**BASE, "state_cotenant": True})["state_cotenant"] is True
    with pytest.raises(ValueError):
        validate({**BASE, "state_cotenant": 1})


def test_collapse_log_is_an_optional_tail_diagnostic():
    tail = {**BASE, "request_end_tail": True}
    assert validate({**tail, "collapse_log": True})["collapse_log"] is True
    assert "collapse_log" not in validate({**tail, "collapse_log": False})
    with pytest.raises(ValueError):
        validate({**BASE, "collapse_log": True})


def test_marker_refuses_a_changed_option(tmp_path):
    target = tmp_path / cache_patch.RELATIVE
    target.parent.mkdir(parents=True)
    target.write_text("x")
    (tmp_path / "cache-layout-patch.json").write_text('{"patched_sha256": "0"}')
    with pytest.raises(RuntimeError, match="state_cotenant changed"):
        cache_patch.apply(tmp_path, tmp_path, state_cotenant=True)


def test_layout_on_the_pinned_runtime(tmp_path):
    """13 main tensors, three GDN groups of 12 and the PLE: the PLE lands in tensor 12 of GDN group 0, the scheduler
    still sees plain MambaSpec groups and a request's fixed blocks fall from 13 to 10 (MTP-1, align)."""
    original = PINNED_SITE / cache_patch.RELATIVE
    if not original.is_file():
        pytest.skip("Optional pinned runtime (ARC_PINNED_SITE)")
    plain = cache_patch.patched_source(original.read_bytes())
    patched = cache_patch.patched_source(original.read_bytes(), True)
    assert plain != patched and b"_arc_fold_replicated_mamba" not in plain
    script = textwrap.dedent('''
        import json, sys, types
        import torch
        from vllm.v1.kv_cache_interface import (CircularBufferSpec, FullAttentionSpec, MLAAttentionSpec, MambaSpec,
                                                MambaAttentionBackendEnum as E)
        import vllm.v1.core.kv_cache_utils as u
        exec(compile(open(sys.argv[1]).read(), "kv_cache_utils.py", "exec"), u.__dict__)
        stub = types.ModuleType("vllm.model_executor.models.utils")
        stub.extract_layer_index = lambda name, n=1: [int(q) for q in name.split(".") if q.isdigit()][0]
        sys.modules["vllm.model_executor.models.utils"] = stub
        sys.modules["vllm.v1.core.arc_cache_export"] = types.SimpleNamespace(emit=lambda *a: None)
        B, specs = 3136, {}
        for base in [f"language_model.model.layers.{3 + 4 * i}" for i in range(12)] + ["mtp.layers.48"]:
            specs[base + ".self_attn.attn"] = FullAttentionSpec(block_size=B, num_kv_heads=1, head_size=264, dtype=torch.int8)
            specs[base + ".self_attn.indexer.compressed_key_cache"] = MLAAttentionSpec(
                block_size=B, num_kv_heads=1, head_size=128, dtype=torch.bfloat16, compress_ratio=4)
            specs[base + ".self_attn.indexer.raw_key_cache"] = CircularBufferSpec(
                block_size=8, num_kv_heads=1, head_size=132, head_size_v=0, dtype=torch.bfloat16)
        gdn = MambaSpec(block_size=B, shapes=((3, 10240), (48, 128, 128)), dtypes=(torch.bfloat16,) * 2,
                        mamba_type=next(m for m in E if "GDN" in m.name), mamba_cache_mode="align",
                        num_speculative_blocks=1)
        ple = MambaSpec(block_size=B, shapes=((9, 10240),), dtypes=(torch.bfloat16,), mamba_type=E.SHORT_CONV,
                        mamba_cache_mode="align", num_speculative_blocks=1, tp_replicated=True)
        for i in range(48):
            if i % 4 != 3:
                specs[f"language_model.model.layers.{i}.linear_attn"] = gdn
        specs["language_model.model.layers.1.ple"] = ple
        NS = types.SimpleNamespace
        config = NS(model_config=NS(max_model_len=262144, original_max_model_len=262144,
                                    get_num_kv_heads=lambda parallel: 1),
                    cache_config=NS(enable_prefix_caching=True, prefix_match_unit=16, num_gpu_blocks_override=None,
                                    mamba_cache_mode="align", block_size=B),
                    parallel_config=NS(pipeline_parallel_size=1, decode_context_parallel_size=1),
                    scheduler_config=NS(disable_hybrid_kv_cache_manager=False),
                    speculative_config=NS(num_speculative_tokens=1, use_multi_module_mtp=lambda: False),
                    kv_transfer_config=None)
        block = 13 * (1655808 + 200704)
        configs = u.get_kv_cache_configs(config, [specs], [2084 * block])
        worker, sched = configs[0], u.generate_scheduler_kv_cache_config(configs)
        fixed = [-(-g.kv_cache_spec.max_memory_usage_bytes(config) // g.kv_cache_spec.page_size_bytes)
                 for g in sched.kv_cache_groups[1:]]
        print(json.dumps({
            "blocks": worker.num_blocks,
            "worker": [[type(g.kv_cache_spec).__name__, len(g.layer_names)] for g in worker.kv_cache_groups],
            "sched": [[type(g.kv_cache_spec).__name__, len(g.layer_names)] for g in sched.kv_cache_groups],
            "fixed": fixed, "tensor12": worker.kv_cache_tensors[12].shared_by,
            "tensor0": worker.kv_cache_tensors[0].shared_by, "sizes": u.resolve_kv_cache_block_sizes(sched, config),
            "ple_spec": [list(s.shapes[0]), s.page_size_bytes, s.mamba_type.name] if (s := getattr(
                worker.kv_cache_groups[2].kv_cache_spec, "kv_cache_specs", {}).get("language_model.model.layers.1.ple"))
            else None}))
    ''')
    results = {}
    for name, source in (("plain", plain), ("patched", patched)):
        path = tmp_path / f"{name}.py"
        path.write_bytes(source)
        run = subprocess.run([PINNED_PYTHON, "-c", script, str(path)], capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": str(PINNED_SITE)})
        if run.returncode and "ModuleNotFoundError" in run.stderr:
            pytest.skip("The pinned runtime's imports are not installed here")
        assert run.returncode == 0, run.stderr[-2000:]
        results[name] = json.loads(run.stdout.strip().splitlines()[-1])
    plain, patched = results["plain"], results["patched"]
    assert plain["blocks"] == patched["blocks"] == 2084 and plain["sizes"] == patched["sizes"] == [3136, 16]
    assert plain["fixed"] == [1, 3, 3, 3, 3] and patched["fixed"] == [1, 3, 3, 3]
    assert plain["worker"][2:] == [["MambaSpec", 12]] * 3 + [["MambaSpec", 1]]
    assert patched["worker"][2:] == [["UniformTypeKVCacheSpecs", 13], ["MambaSpec", 12], ["MambaSpec", 12]]
    assert patched["sched"][2:] == [["MambaSpec", 13], ["MambaSpec", 12], ["MambaSpec", 12]]
    assert patched["tensor12"] == ["mtp.layers.48.self_attn.attn", "language_model.model.layers.1.ple"]
    assert plain["tensor12"] == ["mtp.layers.48.self_attn.attn"]
    assert "language_model.model.layers.1.ple" in plain["tensor0"] and patched["tensor0"] == plain["tensor0"][:-1]
    assert patched["ple_spec"] == [[9, 10240], 1655808, "SHORT_CONV"] and plain["ple_spec"] is None
