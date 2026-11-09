"""ARC_EXPERT_MASK_COMPACT (MFU-2, 24 Oct): masked experts are not allocated; the served path must equal the masked one."""
import ast
import hashlib
import json
import math
import shutil
import sys
import types
from pathlib import Path

import pytest

from arc_infra import expert_mask_patch as P

KEEP = Path(__file__).resolve().parents[2] / "throughput-arm/keep/vllm-full"
needs_pinned = pytest.mark.skipif(not (KEEP / "vllm").is_dir(), reason="pinned vLLM source not on this machine")


def mask_file(tmp_path, experts=512, layers=48, per=192, uneven=False):
    pruned = {str(layer): list(range(layer % 7, layer % 7 + per - (layer == 3 and uneven))) for layer in range(layers)}
    raw = json.dumps({"format": "arc-expert-prune-v1", "experts": experts, "fraction": per / experts,
                      "pruned_per_layer": "per-layer" if uneven else per, "pruned": pruned,
                      **({"pruned_counts": {k: len(v) for k, v in pruned.items()},
                          "pruned_total": sum(len(v) for v in pruned.values())} if uneven else {})}).encode()
    path = tmp_path / "t37.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def runtime(tmp_path, monkeypatch, spec):
    pytest.importorskip("torch")
    worker = types.ModuleType("vllm.v1.worker.worker_base")
    worker.WorkerWrapperBase = type("WorkerWrapperBase", (), {})
    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.worker_base", worker)
    from arc_infra import expert_mask_runtime as R

    settings = tmp_path / "arc_expert_mask.json"
    settings.write_text(json.dumps(spec))
    monkeypatch.setattr(R, "SETTINGS", str(settings))
    monkeypatch.setattr(R, "COMPACT", {"plan": None, "layers": set(), "dropped": 0, "scale_max": 0, "ids_tables": 0})
    return R


# ---- patch ------------------------------------------------------------------------------------------------------
@needs_pinned
def test_every_compact_target_patches_the_pinned_runtime():
    for t in P.COMPACT_TARGETS:
        out = P.compact_source(t, (KEEP / t["relative"]).read_bytes()).decode()
        assert "_arc_" in out and out.count("from vllm.arc_expert_mask import") >= 1
        with pytest.raises(RuntimeError, match="unrecognized"):
            P.compact_source(t, (KEEP / t["relative"]).read_bytes() + b"\n")
    lm = P.compact_source(P.COMPACT_TARGETS[3], (KEEP / P.COMPACT_TARGETS[3]["relative"]).read_bytes()).decode()
    assert lm.count("_arc_sel = _arc_expert_select(module_name, module, expert_start, expert_end)") == 3
    assert "expert_start:expert_end" not in lm
    runner = P.patched_source((KEEP / P.RUNNER["relative"]).read_bytes(), True, False, True).decode()
    assert runner.index("self.router.select_experts(") < runner.index("_arc_compact_ids(self.routed_experts, topk_ids)")
    assert runner.index("_arc_compact_ids(") < runner.index("self.routed_experts.forward_modular(")
    assert "_arc_install_compact()" in runner and "_arc_install_skip_read()" in runner


def site_of(tmp_path):
    site = tmp_path / "site"
    for rel in [P.RUNNER["relative"]] + [t["relative"] for t in P.COMPACT_TARGETS]:
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(KEEP / rel, site / rel)
    return site


@needs_pinned
def test_apply_compact_patches_all_files_and_accepts_only_its_own_restart(tmp_path, monkeypatch):
    site, out = site_of(tmp_path), tmp_path / "out"
    path, digest = mask_file(tmp_path)
    monkeypatch.setenv(P.COMPACT_ENV, "1")
    ident = P.apply(site, out, path, digest)
    assert set(ident["compact"]) == {t["relative"] for t in P.COMPACT_TARGETS}
    assert json.loads((site / P.SETTINGS).read_text())["compact"] is True
    for t in P.COMPACT_TARGETS:
        assert hashlib.sha256((site / t["relative"]).read_bytes()).hexdigest() == ident["compact"][t["relative"]]
    assert P.apply(site, out, path, digest) == ident
    monkeypatch.setenv(P.COMPACT_ENV, "0")  # a restart that changes the mode is refused
    with pytest.raises(RuntimeError, match="modified or partially patched"):
        P.apply(site, out, path, digest)


@needs_pinned
def test_compact_off_leaves_the_other_files_alone(tmp_path, monkeypatch):
    site, out = site_of(tmp_path), tmp_path / "out"
    path, digest = mask_file(tmp_path)
    monkeypatch.delenv(P.COMPACT_ENV, raising=False)
    ident = P.apply(site, out, path, digest)
    assert "compact" not in ident and "compact" not in json.loads((site / P.SETTINGS).read_text())
    for t in P.COMPACT_TARGETS:
        assert (site / t["relative"]).read_bytes() == (KEEP / t["relative"]).read_bytes()


@needs_pinned
def test_compact_refuses_an_uneven_mask_before_writing(tmp_path, monkeypatch):
    site, out = site_of(tmp_path), tmp_path / "out"
    path, digest = mask_file(tmp_path, uneven=True)
    monkeypatch.setenv(P.COMPACT_ENV, "1")
    with pytest.raises(ValueError, match="one pruned count"):
        P.apply(site, out, path, digest)
    assert (site / P.RUNNER["relative"]).read_bytes() == (KEEP / P.RUNNER["relative"]).read_bytes()


# ---- runtime ----------------------------------------------------------------------------------------------------
SPEC = {"experts": 16, "pruned": {"0": [1, 2, 3, 9, 15], "1": [0, 4, 5, 6, 7]}, "label": "t", "compact": True}


def test_plan_layers_and_manager(tmp_path, monkeypatch):
    R = runtime(tmp_path, monkeypatch, SPEC)
    plan = R.compact_plan()
    assert plan["kept"] == 11 and plan["ids"][0] == [0, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]
    assert plan["slots"][1][1] == 0 and plan["slots"][1][8] == 3
    assert R.layer_of("language_model.model.layers.1.mlp.experts") == 1
    assert R.layer_of("model.language_model.layers.12.mlp.experts.3.down_proj.weight") == 12
    assert R.layer_of("model.mtp.layers.0.mlp.experts") is None and R.layer_of("layers.1.mlp.gate") is None
    man = types.SimpleNamespace(global_num_experts=16, use_ep=False, _expert_map=None, num_fused_shared_experts=0,
                                _local_num_experts=16)
    assert R.compact_manager(man, "language_model.model.layers.1.mlp.experts") and man._local_num_experts == 11
    assert man.arc_layer == 1
    other = types.SimpleNamespace(global_num_experts=16, use_ep=False, _expert_map=None, num_fused_shared_experts=0,
                                  _local_num_experts=16)
    assert not R.compact_manager(other, "model.mtp.layers.0.mlp.experts") and other._local_num_experts == 16
    ep = types.SimpleNamespace(global_num_experts=16, use_ep=True, _expert_map=None, num_fused_shared_experts=0)
    with pytest.raises(RuntimeError, match="EP on"):
        R.compact_manager(ep, "language_model.model.layers.0.mlp.experts")


def test_uneven_plan_is_refused(tmp_path, monkeypatch):
    R = runtime(tmp_path, monkeypatch, {**SPEC, "pruned": {"0": [1], "1": [2, 3]}})
    with pytest.raises(RuntimeError, match="one kept count"):
        R.compact_plan()


def test_off_without_the_setting(tmp_path, monkeypatch):
    R = runtime(tmp_path, monkeypatch, {k: v for k, v in SPEC.items() if k != "compact"})
    assert R.compact_plan() is None and R.install_compact(type("X", (), {"weight_loader": lambda *a: 0})) is False
    assert R.moe_shape((11, 4, 4)) == (11, 4, 4) and R.moe_experts(11) == 11 and not R.marlin_compact(11, 16)


def routed_of(R, layer, extra=0):
    man = types.SimpleNamespace(global_num_experts=16, use_ep=False, _expert_map=None, num_fused_shared_experts=extra,
                                _local_num_experts=16 + extra)
    R.compact_manager(man, f"language_model.model.layers.{layer}.mlp.experts")
    return types.SimpleNamespace(expert_map_manager=man, params_dtype=None)


def test_reference_moe_compact_equals_masked(tmp_path, monkeypatch):
    """A plain MoE layer: router + mask + top-k over all 16 experts with every expert allocated, against the same
    router with the kept experts in compact slots and the routed ids remapped. Outputs and routes are identical."""
    torch = pytest.importorskip("torch")
    R = runtime(tmp_path, monkeypatch, SPEC)
    gen = torch.Generator().manual_seed(0)
    T, H, Imid, E, K = 37, 24, 20, 16, 4
    x = torch.randn(T, H, generator=gen)
    router = torch.randn(E, H, generator=gen)
    w1 = torch.randn(E, 2 * Imid, H, generator=gen) / 5
    w2 = torch.randn(E, H, Imid, generator=gen) / 5
    for layer in (0, 1):
        mask = torch.zeros(E)
        mask[SPEC["pruned"][str(layer)]] = -1.0e4
        logits = x @ router.T + mask
        weights, ids = torch.topk(torch.softmax(logits, -1), K)
        weights = weights / weights.sum(-1, keepdim=True)

        def moe(w1_, w2_, ids_):
            out = torch.zeros(T, H)
            for t in range(T):
                for k in range(K):
                    e = int(ids_[t, k])
                    g, u = (w1_[e] @ x[t]).chunk(2)
                    out[t] += weights[t, k] * (w2_[e] @ (torch.nn.functional.silu(g) * u))
            return out

        kept = R.compact_plan()["ids"][layer]
        routed = routed_of(R, layer)
        local = R.compact_ids(routed, ids.to(torch.int32))
        assert local.dtype == torch.int32 and int(local.max()) < len(kept)
        assert torch.equal(torch.tensor(kept)[local.long()], ids)  # the slots point back at the routed experts
        assert torch.equal(moe(w1, w2, ids), moe(w1[kept], w2[kept], local))
    assert R.COMPACT["ids_tables"] == 2


def test_compact_ids_leaves_other_layers_and_maps_fused_shared(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    R = runtime(tmp_path, monkeypatch, SPEC)
    plain = types.SimpleNamespace(expert_map_manager=types.SimpleNamespace())
    ids = torch.tensor([[1, 2]])
    assert R.compact_ids(plain, ids) is ids
    routed = routed_of(R, 0, extra=1)
    assert routed.expert_map_manager._local_num_experts == 12
    assert R.compact_ids(routed, torch.tensor([[0, 16, 14]])).tolist() == [[0, 11, 10]]


def test_loader_slots_and_masked_scale_max(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    R = runtime(tmp_path, monkeypatch, SPEC)
    got = []

    class Routed:
        def __init__(self, layer):
            r = routed_of(R, layer)
            self.expert_map_manager, self.params_dtype = r.expert_map_manager, torch.bfloat16
            self.w13_weight_scale, self.w2_weight_scale, self.w13_weight = object(), object(), object()

        def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
            got.append((weight_name, expert_id))
            return True if return_success else None

    Routed.weight_loader.supports_moe_loading = True
    assert R.install_compact(Routed) and not R.install_compact(Routed)
    m = Routed(0)
    assert getattr(m.weight_loader, "supports_moe_loading", False)
    assert m.weight_loader(m.w13_weight, None, "a", "w1", 4) is None and got[-1] == ("a", 1)  # id 4 -> slot 1
    assert m.weight_loader(m.w13_weight, None, "b", "w1", 1, return_success=True) is True  # masked: dropped
    assert m.weight_loader(m.w13_weight, None, "b", "w1", 1, True) is True
    assert got[-1] == ("a", 1)
    s1 = torch.tensor([0.0, 0.5, 3.0]).to(torch.float8_e4m3fn)
    s2 = torch.tensor([1.0, 7.0]).to(torch.float8_e4m3fn)
    m.weight_loader(m.w13_weight_scale, s1, "s", "w1", 2)
    m.weight_loader(m.w13_weight_scale, s2, "s", "w3", 3)
    m.weight_loader(m.w2_weight_scale, s1, "s", "w2", 9)
    assert m.__dict__["_arc_masked_scale_max"] == {"w13": 7.0, "w2": 3.0} and R.COMPACT["dropped"] == 5
    plain = type("Plain", (Routed,), {})
    p = plain.__new__(plain)
    p.expert_map_manager = types.SimpleNamespace()
    p.weight_loader(None, None, "c", "w1", 9)
    assert got[-1] == ("c", 9)  # an uncompacted layer loads as before


def pinned_scale_factor():
    src = (KEEP / "vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py").read_text()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_nvfp4_compute_scale_factor")
    import torch

    ns = {"torch": torch}
    exec(compile(ast.Module([node], []), "pinned", "exec"), ns)
    return ns["_nvfp4_compute_scale_factor"]


@needs_pinned
def test_marlin_scale_factor_is_the_all_expert_factor(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    R = runtime(tmp_path, monkeypatch, SPEC)
    original = pinned_scale_factor()
    gen = torch.Generator().manual_seed(1)
    for trial in range(40):
        full = (torch.rand(16, 8, 6, generator=gen) * (2.0 ** (trial % 9 - 4))).to(torch.float8_e4m3fn)
        full[0, 0, 0] = 0
        scales = full.to(torch.bfloat16)
        masked = SPEC["pruned"]["0"]
        kept = R.compact_plan()["ids"][0]
        layer = types.SimpleNamespace()
        mx = scales[masked].float()
        mx = mx[mx > 0]
        layer.__dict__["_arc_masked_scale_max"] = {"w13": float(mx.max()) if mx.numel() else 0.0}
        want = original(scales, torch.bfloat16)
        assert R.marlin_scale_factor(layer, "w13", scales[kept], torch.bfloat16, original) == want
        assert R.marlin_scale_factor(types.SimpleNamespace(), "w13", scales, torch.bfloat16, original) == want
    assert R.marlin_scale_factor(layer, "w13", scales[kept], torch.half, original) == 1.0


def test_lora_rows_follow_the_kept_experts(tmp_path, monkeypatch):
    """The 3D PEFT adapter (rank x experts flattened) cut the way the patched _convert_3d_to_2d_moe_lora cuts it."""
    torch = pytest.importorskip("torch")
    R = runtime(tmp_path, monkeypatch, SPEC)
    module = types.SimpleNamespace(use_ep=False, local_num_experts=11, global_num_experts=16, ep_rank=0)
    name = "language_model.model.layers.0.mlp.experts"
    sel = R.expert_select(name, module, 0, 11)
    kept = R.compact_plan()["ids"][0]
    r, H, Imid = 3, 5, 4
    a = torch.arange(16 * r * H, dtype=torch.float32).reshape(16 * r, H)  # (rank * experts, hidden)
    b = torch.arange(2 * Imid * r * 16, dtype=torch.float32).reshape(2 * Imid, r * 16)  # (out, rank * experts)
    a3 = a.reshape(16, -1, H)[sel]
    b3 = b.reshape(2 * Imid, -1, 16)[..., sel].permute(2, 0, 1)
    assert torch.equal(a3, a.reshape(16, r, H)[kept]) and torch.equal(b3[2], b.reshape(2 * Imid, r, 16)[:, :, kept[2]])
    assert R.compacted_module(name, module)
    assert R.expert_select("language_model.model.layers.0.mlp.experts", types.SimpleNamespace(
        use_ep=True, local_num_experts=8, global_num_experts=16), 8, 16) == slice(8, 16)  # EP: vLLM's own slice
    assert R.expert_select("x.mtp.layers.0.mlp.experts", module, 0, 11) == slice(0, 11)


def test_kernel_choices_read_the_checkpoint_count(tmp_path, monkeypatch):
    R = runtime(tmp_path, monkeypatch, SPEC)
    assert R.moe_shape((11, 40, 24)) == (16, 40, 24) and R.moe_shape((16, 40, 24)) == (16, 40, 24)
    assert R.moe_experts(11) == 16 and R.moe_experts(7) == 7
    assert R.marlin_compact(11, 16) and not R.marlin_compact(16, 16)

    def block(M, topk, E, G, patched):  # fused_marlin_moe's block-size pick, as edited by COMPACT_TARGETS[1]
        if G == -1:
            G = E
        else:
            M = M if (patched and R.marlin_compact(E, G)) else math.ceil(M * E / G)
        for bs in [8, 16, 32, 48, 64]:
            if M * topk / (G if patched and R.marlin_compact(E, G) else E) / bs < 0.9:
                break
        return bs

    masked = [block(M, 4, 16, 16, False) for M in range(1, 600)]
    assert [block(M, 4, 11, 16, True) for M in range(1, 600)] == masked
    assert [block(M, 4, 11, 16, False) for M in range(1, 600)] != masked  # unpatched, the pick drifts


def test_status_reports_compact(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    R = runtime(tmp_path, monkeypatch, SPEC)
    monkeypatch.setattr(R, "_STATE", None)
    R.on_router_logits(types.SimpleNamespace(layer_id=0), torch.zeros(2, 16))
    R.compact_ids(routed_of(R, 1), torch.tensor([[0]]))
    import vllm.v1.worker.worker_base as wb

    st = wb.WorkerWrapperBase.arc_expert_mask(None, {"op": "status"})
    assert st["compact"] == {"kept": 11, "layers": 1, "dropped_loads": 0, "scale_max_rows": 0, "slot_tables": 1}


def test_record_scale_max_matches_the_masked_reduction():
    import types

    torch = pytest.importorskip("torch")

    from arc_infra import expert_mask_runtime as r

    def masked(t):
        v = t.to(torch.bfloat16).float()
        v = v[v > 0]
        return float(v.max().item()) if v.numel() else None

    g = torch.Generator().manual_seed(0)
    for shape in [(1024, 160), (2560, 64), (3, 3)]:
        for lo in (-0.5, 0.0, -5.0):
            x = (torch.rand(*shape, generator=g) * 3 + lo).to(torch.float8_e4m3fn)
            routed = types.SimpleNamespace(params_dtype=torch.bfloat16)
            r._record_scale_max(routed, "w13", x)
            assert routed.__dict__.get("_arc_masked_scale_max", {}).get("w13") == masked(x)
    routed = types.SimpleNamespace(params_dtype=torch.bfloat16)
    r._record_scale_max(routed, "w2", torch.tensor([float("nan"), 2.0, -1.0]).to(torch.bfloat16))
    assert routed._arc_masked_scale_max["w2"] == 2.0
