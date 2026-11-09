"""Load-time expert pruning: masks, the patch, the weight filter, config sizing, and a CPU forward that matches the
masked reference measured in docs/validation-20260927/expert-usage.md."""

import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arc_infra import expert_prune_patch as patch

ROOT = Path(__file__).resolve().parents[1]
PINNED_SITE = Path(os.environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))


def load_runtime():
    spec = importlib.util.spec_from_file_location("arc_expert_prune_test",
                                                  ROOT / "src/arc_infra/expert_prune_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rt = load_runtime()


@pytest.mark.parametrize("tag,pruned", [("12", 64), ("25", 128)])
def test_the_shipped_masks_match_the_variants_and_the_v4_measurement(tag, pruned):
    from arena.variants import EXPERT_PRUNE_VARIANTS

    spec = EXPERT_PRUNE_VARIANTS[f"fixed-2x-mtp0-wide-kv8-hero-prune{tag}"]
    path = ROOT / spec["source"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]
    mask = patch.load_mask(path, spec["sha256"])
    assert mask["pruned_per_layer"] == pruned and mask["fraction"] == spec["fraction"]
    assert mask["source_run"] == "arena-expert-usage-v4-modal-20260928"
    p = rt.plan(mask["experts"], mask["pruned"])
    assert p["kept_per_layer"] == 512 - pruned and sorted(p["kept"]) == list(range(48))
    with pytest.raises(RuntimeError, match="sha256"):
        patch.load_mask(path, "0" * 64)


USAGE_V4 = Path("./runs/modal-validation/arena-expert-usage-v4-modal-20260928/"
                "arena-expert-usage-v4-modal-20260928/arena/trials/expert-usage/expert-usage.json")


def test_the_level_masks_are_reproducible_and_well_formed(tmp_path):
    uniform = patch.load_mask(ROOT / "data/expert_prune/prune18.json")
    assert uniform["pruned_per_layer"] == 96 and uniform["fraction"] == 0.1875
    layered = patch.load_mask(ROOT / "data/expert_prune/prune25-layer.json")
    counts = {int(k): v for k, v in layered["pruned_counts"].items()}
    assert layered["pruned_per_layer"] == "per-layer" and layered["pruned_total"] == 6144 == 48 * 128
    assert all(counts[i] == 64 for i in (0, 1, 2)) and min(counts.values()) == 64 and max(counts.values()) == 192
    assert all(c % 8 == 0 for c in counts.values())                   # kept counts stay multiples of 8
    p = rt.plan(layered["experts"], layered["pruned"], min_kept=rt.TOP_K_FLOOR)
    assert p["uniform"] is False and p["kept_per_layer"] == 448 and sum(p["kept_counts"].values()) == 48 * 384
    assert patch.kept_summary(layered) == {"kept_per_layer": 448, "kept_total": 18432, "kept_min": 320}
    assert patch.kept_summary(uniform) == {"kept_per_layer": 416, "kept_total": 48 * 416}
    # The uniform 12.5/25% ranking is the same, so the 18.75% mask nests between them.
    p12, p25 = (patch.load_mask(ROOT / f"data/expert_prune/prune{t}.json")["pruned"] for t in ("12", "25"))
    assert all(set(p12[k]) <= set(uniform["pruned"][k]) <= set(p25[k]) for k in p12)
    from arena.variants import EXPERT_PRUNE_VARIANTS

    spec37 = EXPERT_PRUNE_VARIANTS["fixed-2x-mtp0-wide-kv8-hero-prune37-layer"]
    layered37 = patch.load_mask(ROOT / spec37["source"], spec37["sha256"])
    counts37 = {int(k): v for k, v in layered37["pruned_counts"].items()}
    assert layered37["pruned_total"] == 9216 == 48 * 192 and layered37["fraction"] == spec37["fraction"] == 0.375
    assert all(counts37[i] == 96 for i in (0, 1, 2)) and 96 <= min(counts37.values()) and max(counts37.values()) <= 288
    assert all(c % 8 == 0 for c in counts37.values())
    assert patch.kept_summary(layered37) == {"kept_per_layer": 416, "kept_total": 48 * 320, "kept_min": 224}
    assert rt.plan(layered37["experts"], layered37["pruned"], min_kept=rt.TOP_K_FLOOR)["uniform"] is False
    # The 37.5% mask nests the 25% layer-aware one where it prunes at least as many experts per layer (same ranking).
    assert all(set(layered["pruned"][k]) <= set(layered37["pruned"][k]) for k in layered["pruned"]
               if counts37[int(k)] >= counts[int(k)])
    spec50 = EXPERT_PRUNE_VARIANTS["fixed-2x-mtp0-wide-kv8-hero-prune50-layer"]
    layered50 = patch.load_mask(ROOT / spec50["source"], spec50["sha256"])
    counts50 = {int(k): v for k, v in layered50["pruned_counts"].items()}
    assert layered50["pruned_total"] == 12288 == 48 * 256 and layered50["fraction"] == spec50["fraction"] == 0.5
    assert all(counts50[i] == 128 for i in (0, 1, 2)) and 128 <= min(counts50.values()) and max(counts50.values()) <= 384
    assert all(c % 8 == 0 for c in counts50.values())
    assert patch.kept_summary(layered50)["kept_total"] == 48 * 256
    assert rt.plan(layered50["experts"], layered50["pruned"], min_kept=rt.TOP_K_FLOOR)["uniform"] is False
    # The 50% mask nests the 37.5% one where it prunes at least as many experts per layer (same ranking).
    assert all(set(layered37["pruned"][k]) <= set(layered50["pruned"][k]) for k in layered37["pruned"]
               if counts50[int(k)] >= counts37[int(k)])
    bad = dict(layered, pruned_total=6000)
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="pruned_total"):
        patch.load_mask(tmp_path / "bad.json")
    if not USAGE_V4.is_file():
        pytest.skip("the v4 expert-usage.json is not available")
    spec = importlib.util.spec_from_file_location("masks", ROOT / "scripts/expert_prune_masks.py")
    masks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(masks)
    built = masks.build(json.loads(USAGE_V4.read_text()))
    for name, m in built.items():
        assert (json.dumps(m, indent=1, sort_keys=True) + "\n") == (ROOT / f"data/expert_prune/{name}.json").read_text()
    layer50 = built["prune50-layer"]["heldout_routing_on_pruned"]
    assert layer50["count_mean"] < 0.0989 and layer50["count_max_layer"] < 0.233  # uniform 50%: 9.89% / 23.3%
    layer37 = built["prune37-layer"]["heldout_routing_on_pruned"]
    assert layer37["count_mean"] < 0.0457 and layer37["count_max_layer"] < 0.141  # uniform 37.5%: 4.57% / 14.1%
    layer_aware = built["prune25-layer"]["heldout_routing_on_pruned"]
    uniform25 = patch.load_mask(ROOT / "data/expert_prune/prune25.json")["heldout_routing_on_pruned"]
    assert layer_aware["count_mean"] < uniform25["count_mean"]
    assert layer_aware["count_max_layer"] < uniform25["count_max_layer"]


def test_moe_blocks_are_built_with_their_own_layer_count():
    class Block:
        def __init__(self, vllm_config, prefix=""):
            """Docs."""
            self.n_routed_experts = vllm_config.model_config.hf_text_config.num_experts

    config = SimpleNamespace(num_experts=8, num_hidden_layers=2)
    vllm_config = SimpleNamespace(speculative_config=None, model_config=SimpleNamespace(hf_text_config=config))
    p = rt.plan(8, {"0": [1], "1": [0, 7, 3]})
    logs = []
    rt.prepare_config(vllm_config, p, log=lambda *a, **k: logs.append(a[0]), moe_block=Block)
    assert config.num_experts == 7 and Block.__init__._arc_moe_sizes == {0: 7, 1: 5}
    assert Block.__init__.__doc__ == "Docs."
    assert Block(vllm_config, prefix="model.language_model.layers.1.mlp").n_routed_experts == 5
    assert Block(vllm_config=vllm_config, prefix="model.language_model.layers.0.mlp").n_routed_experts == 7
    assert config.num_experts == 7                                    # restored after each block
    with pytest.raises(RuntimeError, match="no kept count"):
        Block(vllm_config, prefix="mtp.layers.3.mlp")
    assert config.num_experts == 7
    assert rt.install_moe_sizes({0: 7}, Block, log=lambda *a, **k: None) is False     # idempotent
    events = [json.loads(line.split(" ", 1)[1]) for line in logs]
    assert {"per_layer_experts": True, "min": 5, "max": 7, "total": 12}.items() <= events[0].items()
    assert events[-1]["uniform"] is False and events[-1]["kept_total"] == 12
    # A uniform plan leaves the block class alone.
    class Plain:
        def __init__(self, vllm_config, prefix=""):
            pass

    config.num_experts = 8
    rt.prepare_config(vllm_config, rt.plan(8, {"0": [1], "1": [2]}), log=lambda *a, **k: None, moe_block=Plain)
    assert not hasattr(Plain.__init__, "_arc_moe_sizes")


def test_plan_renumbers_kept_experts_in_ascending_order_and_sizes_uneven_layers():
    p = rt.plan(8, {"0": [1, 5], "1": [0, 7]})
    assert p["kept"][0] == [0, 2, 3, 4, 6, 7] and p["remap"][0] == {0: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5}
    assert p["kept"][1] == [1, 2, 3, 4, 5, 6] and p["kept_per_layer"] == 6 and p["uniform"] is True
    uneven = rt.plan(8, {"0": [1], "1": [0, 7, 3]})
    assert uneven["uniform"] is False and uneven["kept_counts"] == {0: 7, 1: 5} and uneven["kept_per_layer"] == 7
    assert uneven["remap"][1] == {1: 0, 2: 1, 4: 2, 5: 3, 6: 4}
    with pytest.raises(ValueError):
        rt.plan(8, {"0": [1, 8]})
    with pytest.raises(ValueError, match="fewer than 10"):
        rt.plan(16, {"0": list(range(7))}, min_kept=rt.TOP_K_FLOOR)


def test_filter_drops_renames_and_slices_only_the_target_layers():
    p = rt.plan(8, {"0": [1, 5], "1": [0, 7]})
    router0 = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
    weights = [
        ("model.language_model.layers.0.mlp.gate.weight", router0),
        ("model.language_model.layers.0.mlp.experts.1.gate_proj.weight", "drop"),
        ("model.language_model.layers.0.mlp.experts.6.down_proj.weight_scale", "e6"),
        ("model.language_model.layers.0.mlp.shared_expert.gate_proj.weight", "shared"),
        ("model.language_model.layers.0.mlp.shared_expert_gate.weight", "shared-gate"),
        ("model.language_model.layers.1.mlp.experts.7.up_proj.input_scale", "drop"),
        ("model.language_model.layers.1.mlp.experts.3.up_proj.weight_scale_2", "e3"),
        ("model.language_model.layers.1.mlp.gate.weight", np.zeros((8, 3))),
        ("mtp.layers.0.mlp.experts.1.gate_proj.weight", "mtp"),
        ("model.language_model.embed_tokens.weight", "embed"),
    ]
    logs = []
    out = list(rt.filter_weights(iter(weights), p, log=lambda *a, **k: logs.append(a[0])))
    names = [n for n, _ in out]
    assert "model.language_model.layers.0.mlp.experts.4.down_proj.weight_scale" in names       # 6 -> rank 4
    assert "model.language_model.layers.1.mlp.experts.2.up_proj.weight_scale_2" in names      # 3 -> rank 2
    assert not any(isinstance(v, str) and v == "drop" for _, v in out)
    for kept in ("shared", "shared-gate", "mtp", "embed"):
        assert any(v is kept for _, v in out)
    router = dict(out)["model.language_model.layers.0.mlp.gate.weight"]
    assert np.array_equal(router, router0[[0, 2, 3, 4, 6, 7]])
    assert json.loads(logs[-1].split(" ", 1)[1]) == {"dropped": 2, "renamed": 2, "routers": 2}
    # A stream with expert tensors but a router missing is refused at its end.
    with pytest.raises(RuntimeError, match="routers"):
        list(rt.filter_weights(iter(weights[:3]), p, log=lambda *a, **k: None))


def test_prepare_config_sets_the_kept_count_once_and_refuses_mtp():
    p = rt.plan(8, {str(i): [0, 1] for i in range(4)})
    config = SimpleNamespace(num_experts=8, num_hidden_layers=4)
    vllm_config = SimpleNamespace(speculative_config=None, model_config=SimpleNamespace(hf_text_config=config))
    rt.prepare_config(vllm_config, p, log=lambda *a, **k: None)
    assert config.num_experts == 6 and config.arc_unpruned_num_experts == 8  # the PLE exact gate reads the latter
    rt.prepare_config(vllm_config, p, log=lambda *a, **k: None)          # idempotent
    assert config.num_experts == 6 and config.arc_unpruned_num_experts == 8
    with pytest.raises(RuntimeError, match="MTP-0"):
        rt.prepare_config(SimpleNamespace(speculative_config=object(), model_config=vllm_config.model_config), p)
    with pytest.raises(RuntimeError, match="expects 8"):
        bad = SimpleNamespace(speculative_config=None, model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_experts=10, num_hidden_layers=4)))
        rt.prepare_config(bad, p, log=lambda *a, **k: None)


def _moe_forward(x, router, gate_up, down, top_k, mask=None):
    """Softmax router, top-k, renormalized weights, SiLU-gated experts (the Qwen3-Next MoE, numpy)."""
    logits = x @ router.T
    if mask is not None:
        logits = logits + mask
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    top = np.argsort(-probs, axis=-1)[:, :top_k]
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        w = probs[t, top[t]] / probs[t, top[t]].sum()
        for weight, e in zip(w, top[t]):
            g, u = np.split(gate_up[e] @ x[t], 2)
            out[t] += weight * (down[e] @ (g / (1 + np.exp(-g)) * u))
    return out, top


def test_a_pruned_layer_loaded_through_the_filter_matches_the_masked_reference():
    rng = np.random.default_rng(0)
    experts, hidden, inter, top_k, layer = 16, 8, 4, 3, 5
    pruned = [2, 3, 9, 11]
    router = rng.normal(size=(experts, hidden))
    gate_up = rng.normal(size=(experts, 2 * inter, hidden))
    down = rng.normal(size=(experts, hidden, inter))
    x = rng.normal(size=(32, hidden))
    mask = np.zeros(experts)
    mask[pruned] = -1.0e4                                            # the v4 probe's additive mask
    reference, reference_top = _moe_forward(x, router, gate_up, down, top_k, mask)
    # A checkpoint in the served naming, streamed through the filter; the layer is rebuilt from what it yields.
    base = f"model.language_model.layers.{layer}.mlp."
    checkpoint = [(base + "gate.weight", router)]
    for e in range(experts):
        checkpoint += [(base + f"experts.{e}.gate_proj.weight", gate_up[e, :inter]),
                       (base + f"experts.{e}.up_proj.weight", gate_up[e, inter:]),
                       (base + f"experts.{e}.down_proj.weight", down[e])]
    p = rt.plan(experts, {str(layer): pruned})
    loaded = dict(rt.filter_weights(iter(checkpoint), p, log=lambda *a, **k: None))
    kept = p["kept_per_layer"]
    new_router = loaded[base + "gate.weight"]
    new_gate_up = np.stack([np.concatenate([loaded[base + f"experts.{i}.gate_proj.weight"],
                                            loaded[base + f"experts.{i}.up_proj.weight"]]) for i in range(kept)])
    new_down = np.stack([loaded[base + f"experts.{i}.down_proj.weight"] for i in range(kept)])
    assert new_router.shape == (experts - len(pruned), hidden) and new_gate_up.shape[0] == kept
    output, top = _moe_forward(x, new_router, new_gate_up, new_down, top_k)
    assert np.allclose(output, reference, rtol=1e-9, atol=1e-9)
    assert np.array_equal(np.array(p["kept"][layer])[top], reference_top)   # the same experts, renumbered
    assert not set(reference_top.ravel()) & set(pruned)


def test_patch_applies_to_the_pinned_loaders_when_available(tmp_path):
    if not all((PINNED_SITE / t["relative"]).is_file() for t in patch.TARGETS):
        pytest.skip("extracted pinned runtime not available (ARC_PINNED_SITE)")
    site = tmp_path / "site"
    for t in patch.TARGETS:
        (site / t["relative"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PINNED_SITE / t["relative"], site / t["relative"])
    mask = patch.load_mask(ROOT / "data/expert_prune/prune12.json")
    identity = patch.apply(site, tmp_path / "out", mask)
    base = (site / patch.BASE_LOADER["relative"]).read_text()
    default = (site / patch.DEFAULT_LOADER["relative"]).read_text()
    assert base.index("_arc_prepare_config(vllm_config)") < base.index("model = initialize_model(")
    assert "_arc_filter_weights(self.get_all_weights(model_config, model))" in default
    assert identity["kept_per_layer"] == 448
    assert json.loads((site / patch.SETTINGS).read_text())["pruned"] == mask["pruned"]
    assert patch.apply(site, tmp_path / "out", mask) == identity                  # restart: accepted, not re-patched


def test_patch_refuses_unknown_files_and_bad_specs(tmp_path):
    site = tmp_path / "site"
    for t in patch.TARGETS:
        (site / t["relative"]).parent.mkdir(parents=True, exist_ok=True)
        (site / t["relative"]).write_text("# not the pinned loader\n")
    with pytest.raises(RuntimeError):
        patch.apply(site, tmp_path / "out", patch.load_mask(ROOT / "data/expert_prune/prune12.json"))
    assert patch.validate(None) is None
    for bad in ({"mask": "x.json"}, {"mask": "/abs.json", "sha256": "0" * 64}, {"mask": "a.json", "sha256": "xyz"}):
        with pytest.raises(ValueError):
            patch.validate(bad)


def test_serving_loads_the_bundled_mask_and_the_build_ships_it(tmp_path):
    from arena import serving
    from arena.variants import EXPERT_PRUNE_VARIANTS

    spec = EXPERT_PRUNE_VARIANTS["fixed-2x-mtp0-wide-kv8-hero-prune25"]
    bundle = tmp_path / "bundle"
    (bundle / spec["mask"]).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / spec["source"], bundle / spec["mask"])
    mask = serving.bundle_expert_prune({"mask": spec["mask"], "sha256": spec["sha256"]}, bundle)
    assert mask["pruned_per_layer"] == 128
    assert serving.bundle_expert_prune(None, bundle) is None
    (bundle / spec["mask"]).write_text((bundle / spec["mask"]).read_text().replace("128", "129", 1))
    with pytest.raises((RuntimeError, ValueError)):
        serving.bundle_expert_prune({"mask": spec["mask"], "sha256": spec["sha256"]}, bundle)


def test_levels_study_runs_the_new_candidates_after_the_reference_with_the_stall_dump(tmp_path):
    spec = importlib.util.spec_from_file_location("builder_levels", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in builder.EXPERT_USAGE_FILES:
        (inputs / name).write_text("{}\n")
    output = tmp_path / "bundle"
    study = builder.expert_prune_check_study({"server": {"mtp_tokens": 0}, "trials": []}, output, inputs,
                                             levels=True)
    assert [t["name"] for t in study["trials"]] == ["prune-ref", "prune-25-layer", "prune-25", "prune-18"]
    assert "ARENA_STALL_DUMP" not in study["trials"][0]["server_env"]
    assert {t["server_env"]["ARENA_STALL_DUMP"] for t in study["trials"][1:]} == {"90"}
    assert all(t["role"] == "candidate" and t["reference"] == "prune-ref" for t in study["trials"][1:])
    assert study["kernel_timeout"] == 3000 and study["expert_prune_check"]["levels"] is True
    for level in builder.EXPERT_PRUNE_LEVELS:
        mask = patch.load_mask(output / level["mask"], level["sha256"])
        assert mask["fraction"] == level["fraction"]
        assert hashlib.sha256((ROOT / level["source"]).read_bytes()).hexdigest() == level["sha256"]
    plain = builder.expert_prune_check_study({"server": {"mtp_tokens": 0}, "trials": []}, tmp_path / "b2", inputs)
    assert [t["name"] for t in plain["trials"]] == ["prune-ref", "prune-12", "prune-25"]
    assert plain["kernel_timeout"] == 2100 and "levels" not in plain["expert_prune_check"]


def test_pruned_hero_focus_slots_follow_the_projected_layout():
    from arena.variants import pruned_hero_focus_slots

    assert pruned_hero_focus_slots(0.0, 25) == 15          # the hero smoke's served layout: 15 full 64K windows
    assert pruned_hero_focus_slots(0.125, 25) == 22
    assert pruned_hero_focus_slots(0.25, 25) == 25 and pruned_hero_focus_slots(0.25, 40) == 29


async def test_prune_check_trials_compare_a_candidate_with_the_reference_file(tmp_path, monkeypatch):
    import random
    import time

    from test_expert_usage import _mock_server

    from arena import expert_usage as eu

    async def idle(*args, **kwargs):
        return {}

    monkeypatch.setattr(eu, "isolate_trial", idle)
    rng = random.Random(3)
    folder = tmp_path / "src" / eu.INPUTS
    folder.mkdir(parents=True)
    for name, count in (("usage.json", 1), ("heldout.json", 3)):
        rows = [{"index": i, "ids": [rng.randrange(30) for _ in range(200)], "assistant_spans": [[5, 150]]}
                for i in range(count)]
        (folder / name).write_text(json.dumps(rows))
    server = tmp_path / "serving"
    server.mkdir()
    (server / "vllm-openai-server.log").write_text(
        "INFO Model loading took 63.41 GiB memory\nGPU KV cache size: 615,000 tokens\n"
        "Using 'MARLIN' NvFp4 MoE backend\nARC_EXPERT_PRUNE {\"num_experts\": 448, \"from\": 512, \"layers\": 48}\n")
    common = dict(source=tmp_path / "src", server_dir=server, base_url="http://x/v1",
                  deadline_epoch=time.time() + 600, transport=_mock_server(), log=lambda *a, **k: None)
    trials = tmp_path / "trials"
    ref = await eu.run_prune_check({"served_model": "m"}, {"name": "prune-ref", "role": "reference"},
                                   output=trials / "prune-ref", **common)
    assert ref["status"] == "complete", ref["errors"]
    assert ref["passes"]["repeat"]["summary"]["all"]["mean_kl"] == pytest.approx(0.0)
    assert ref["server"]["model_loading_gib"] == 63.41 and ref["server"]["marlin"] is True
    assert ref["server"]["kv_cache_tokens"] == 615000 and ref["server"]["expert_prune"][0]["num_experts"] == 448
    cand = await eu.run_prune_check({"served_model": "m"}, {"name": "prune-12", "role": "candidate",
                                                            "reference": "prune-ref"},
                                    output=trials / "prune-12", **common)
    assert cand["status"] == "complete", cand["errors"]
    assert cand["passes"]["candidate"]["summary"]["assistant"]["positions"] > 0


def test_stall_dump_rides_on_the_prune_overlay(tmp_path, monkeypatch):
    import faulthandler

    from arena import serving

    assert serving.lab_overrides({"ARENA_STALL_DUMP": "120"}) == {"stall_dump_seconds": 120}
    with pytest.raises(ValueError):
        serving.lab_overrides({"ARENA_STALL_DUMP": "5"})
    with pytest.raises(ValueError):
        serving.with_stall_dump(None, 120)
    mask = patch.load_mask(ROOT / "data/expert_prune/prune12.json")
    assert serving.with_stall_dump(mask, None) is mask
    armed = serving.with_stall_dump(mask, 120)
    site = tmp_path / "site"
    for t in patch.TARGETS:
        (site / t["relative"]).parent.mkdir(parents=True, exist_ok=True)
        (site / t["relative"]).write_text("x\n")
    fake = tuple({**t, "sha256": hashlib.sha256(b"x\n").hexdigest(), "edits": []} for t in patch.TARGETS)
    monkeypatch.setattr(patch, "TARGETS", fake)
    patch.apply(site, tmp_path / "out", armed)
    assert json.loads((site / patch.SETTINGS).read_text())["stall_dump_seconds"] == 120
    calls = []
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later", lambda: calls.append("cancel"))
    p = rt.plan(8, {str(i): [0, 1] for i in range(4)})
    p["stall_dump_seconds"] = 120
    config = SimpleNamespace(num_experts=8, num_hidden_layers=4)
    worker = type("FakeWorker", (), {"execute_model": lambda self, s: ("ran", s),
                                     "sample_tokens": lambda self, g: ("sampled", g)})
    monkeypatch.setitem(__import__("sys").modules, "arc_fake_worker", SimpleNamespace(FakeWorker=worker))
    vllm_config = SimpleNamespace(speculative_config=None, model_config=SimpleNamespace(hf_text_config=config),
                                  parallel_config=SimpleNamespace(worker_cls="arc_fake_worker.FakeWorker"))
    rt.prepare_config(vllm_config, p, log=lambda *a, **k: None)
    assert calls == []  # nothing armed at load: the timer belongs to steps only
    assert worker().execute_model(3) == ("ran", 3)
    assert calls == [((120,), {"repeat": False, "file": calls[0][1]["file"]}), "cancel"]


def test_stall_dump_is_a_one_shot_timer_per_worker_step(monkeypatch):
    import faulthandler

    calls = []
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **k: calls.append(("arm", a, k["repeat"])))
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later", lambda: calls.append("cancel"))

    class Worker:
        def execute_model(self, scheduler_output):
            """Docs."""
            calls.append("step")
            if scheduler_output == "boom":
                raise RuntimeError("step failed")
            return scheduler_output

        def sample_tokens(self, grammar_output):
            calls.append("sample")
            return grammar_output

    class Sub(Worker):  # a subclass wrapped separately keeps its own guard
        pass

    logs = []
    assert rt.install_stall_dump(90, Worker, log=lambda *a, **k: logs.append(a[0])) == ["execute_model", "sample_tokens"]
    assert rt.install_stall_dump(90, Worker, log=lambda *a, **k: None) == []  # idempotent
    assert "per-step one-shot" in logs[0] and Worker.execute_model.__name__ == "execute_model"
    assert Worker.execute_model.__doc__ == "Docs." and Worker.execute_model._arc_stall_dump == 90
    w = Worker()
    assert w.execute_model("s") == "s" and w.sample_tokens("g") == "g"
    assert calls == [("arm", (90,), False), "step", "cancel", ("arm", (90,), False), "sample", "cancel"]
    calls.clear()
    with pytest.raises(RuntimeError, match="step failed"):
        w.execute_model("boom")
    assert calls == [("arm", (90,), False), "step", "cancel"]  # cancelled on raise too
    assert rt.install_stall_dump(90, Sub, log=lambda *a, **k: None) == []  # inherited guard is already there
    assert rt.install_stall_dump(90, type("NoSample", (), {"execute_model": lambda self: 1}),
                                 log=lambda *a, **k: None) == ["execute_model"]


def test_worker_class_follows_the_resolved_worker_cls(monkeypatch):
    marker = type("W", (), {})
    monkeypatch.setitem(__import__("sys").modules, "arc_fake_worker2", SimpleNamespace(W=marker))
    assert rt.worker_class(SimpleNamespace(parallel_config=SimpleNamespace(worker_cls="arc_fake_worker2.W"))) is marker
    gpu = type("Worker", (), {})
    monkeypatch.setitem(__import__("sys").modules, "vllm.v1.worker.gpu_worker", SimpleNamespace(Worker=gpu))
    for unresolved in ("auto", None):  # falls back to the pinned GPU worker
        assert rt.worker_class(SimpleNamespace(parallel_config=SimpleNamespace(worker_cls=unresolved))) is gpu
    assert rt.worker_class(SimpleNamespace()) is gpu


def test_stall_dump_fires_only_on_a_stalled_step(tmp_path):
    """Real faulthandler in a child process: a fast step then idle time dumps nothing; a stalled step dumps once."""
    import subprocess
    import sys

    script = tmp_path / "child.py"
    script.write_text(f"""
import importlib.util, time
spec = importlib.util.spec_from_file_location("rt", {str(ROOT / "src/arc_infra/expert_prune_runtime.py")!r})
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
class Worker:
    def execute_model(self, seconds):
        time.sleep(seconds)
    def sample_tokens(self, seconds):
        time.sleep(seconds)
rt.install_stall_dump(1, Worker, log=lambda *a, **k: None)
w = Worker()
for _ in range(5):
    w.execute_model(0.05); w.sample_tokens(0.05)
print("IDLE", flush=True)
time.sleep(2.5)
print("STALL", flush=True)
w.sample_tokens(3.5)
print("DONE", flush=True)
""")
    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == ["IDLE", "STALL", "DONE"]
    assert run.stderr.count("Timeout (0:00:01)!") == 1  # one-shot: a 3.5 s stall dumps once, idle never
    assert "in sample_tokens" in run.stderr and "in execute_model" not in run.stderr


def test_hang_study_dumps_only_on_a_stalled_step():
    from arena import serving

    assert serving.lab_overrides({"ARENA_STALL_DUMP": "240"}) == {"stall_dump_seconds": 240}
    with pytest.raises(ValueError, match="30..240"):  # must fire before vLLM's 300 s execute-model timeout
        serving.lab_overrides({"ARENA_STALL_DUMP": "300"})
    spec = importlib.util.spec_from_file_location("builder_hang", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    study = builder.expert_prune_hang_study({"server": {"expert_prune": {"mask": "m", "sha256": "0" * 64},
                                                        "mtp_tokens": 0}, "trials": []})
    assert [t["server_env"]["ARENA_STALL_DUMP"] for t in study["trials"]] == ["90", "90"]
    assert study["expert_prune_hang_test"]["stall_dump_mode"] == "per-step one-shot"
    assert study["trials"][1]["server_env"]["ARENA_KV_CACHE_MEMORY_BYTES"] == "18229248000"


def test_pinned_worker_dispatches_steps_through_the_wrapped_class():
    """Static check on the extracted pinned runtime: the steps the stall dump wraps are Worker methods looked up
    per RPC, and the CUDA platform resolves worker_cls to the class install_stall_dump patches."""
    files = {name: PINNED_SITE / "vllm" / name for name in
             ("v1/worker/gpu_worker.py", "platforms/cuda.py", "v1/executor/multiproc_executor.py",
              "v1/worker/worker_base.py")}
    if not all(path.is_file() for path in files.values()):
        pytest.skip("extracted pinned runtime not available (ARC_PINNED_SITE)")
    worker = files["v1/worker/gpu_worker.py"].read_text()
    body = worker[worker.index("class Worker(WorkerBase):"):]
    assert all(f"    def {name}(" in body for name in rt.STEP_METHODS)
    assert f'parallel_config.worker_cls = "{rt.DEFAULT_WORKER}"' in files["platforms/cuda.py"].read_text()
    assert "func = getattr(self.worker, method)" in files["v1/executor/multiproc_executor.py"].read_text()
    assert "return self.worker.execute_model(scheduler_output)" in files["v1/worker/worker_base.py"].read_text()


def test_pinned_moe_block_reads_the_expert_count_at_construction_only():
    """Static check for the layer-aware overlay: the block install_moe_sizes wraps takes (vllm_config, prefix),
    reads num_experts from the text config in __init__, the Flash-Next subclass forwards by keyword with the
    decoder's ``...layers.L.mlp`` prefix, and each FusedMoE builds its own expert mapping."""
    files = {name: PINNED_SITE / "vllm" / name for name in
             ("model_executor/models/qwen3_next.py", "models/qwen3_8_flash_next/nvidia/model.py",
              "model_executor/layers/fused_moe/routed_experts.py")}
    if not all(path.is_file() for path in files.values()):
        pytest.skip("extracted pinned runtime not available (ARC_PINNED_SITE)")
    assert rt.MOE_BLOCK == "vllm.model_executor.models.qwen3_next.Qwen3NextSparseMoeBlock"
    base = files["model_executor/models/qwen3_next.py"].read_text()
    block = base[base.index("class Qwen3NextSparseMoeBlock(nn.Module):"):]
    block = block[:block.index("\nclass ")]
    init = block[:block.index("    def forward(")]
    assert "    def __init__(self, vllm_config: VllmConfig, prefix: str = \"\"):" in init
    assert "config = vllm_config.model_config.hf_text_config" in init
    assert "self.n_routed_experts = config.num_experts" in init
    assert "config.num_experts" not in block[len(init):]                  # not read again after construction
    flash = files["models/qwen3_8_flash_next/nvidia/model.py"].read_text()
    assert "super().__init__(vllm_config=vllm_config, prefix=prefix)" in flash
    assert 'Qwen3_8FlashNextSparseMoeBlock(\n                vllm_config=vllm_config, prefix=f"{prefix}.mlp"' in flash
    assert rt.MOE_PREFIX.search("model.language_model.layers.17.mlp").group(1) == "17"
    routed = files["model_executor/layers/fused_moe/routed_experts.py"].read_text()
    assert "expert_mapping = self.get_expert_mapping(include_fused=True)" in routed


def test_memory_watch_logs_growth_and_heartbeats_only_when_installed():
    """Throughput arm (3 October): the opt-in ARENA_MEMORY_WATCH step wrapper (diagnostics only)."""
    from types import SimpleNamespace as NS

    from arc_infra import expert_prune_runtime as runtime

    state = {"reserved": 0, "peak": 0}
    cuda = NS(memory_reserved=lambda: state["reserved"], max_memory_allocated=lambda: state["peak"],
              memory_allocated=lambda: state["peak"], mem_get_info=lambda: (1 << 30, 96 << 30))
    torch = NS(cuda=cuda)
    lines, now = [], [0.0]

    def log(line, **_):
        lines.append(line)

    class Worker:
        def execute_model(self, scheduler_output):
            state["reserved"] += scheduler_output.grow
            return "ran"

    assert runtime.install_memory_watch(Worker, log=log, torch_module=torch, clock=lambda: now[0])
    assert not runtime.install_memory_watch(Worker, log=log, torch_module=torch)  # idempotent
    so = NS(grow=0, num_scheduled_tokens={"a": 4096, "b": 1}, total_num_scheduled_tokens=4097, scheduled_new_reqs=[],
            scheduled_encoder_inputs={"a": [0, 1]}, scheduled_spec_decode_tokens={})
    now[0] = 200.0
    assert Worker().execute_model(so) == "ran"   # first call: a heartbeat (no growth)
    so.grow = 512 << 20
    Worker().execute_model(so)                    # +512 MiB: a growth line with the batch
    so.grow = 1 << 20
    now[0] = 210.0
    Worker().execute_model(so)                    # +1 MiB, no heartbeat due: nothing
    import json
    events = [json.loads(line.split(" ", 1)[1]) for line in lines]
    assert [e["event"] for e in events] == ["installed", "heartbeat", "growth"]
    assert events[2]["reserved_delta_mib"] == 512 and events[2]["tokens"] == 4097 and events[2]["encoder_inputs"] == 2
    assert events[2]["max_request_tokens"] == 4096
