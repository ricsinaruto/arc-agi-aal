"""Expert quantization, phase 3 (29 September; expert-quant.md section 6): act-order GPTQ with static scales, the
pooled-Hessian prior, mask-aware calibration on a pruned server, overlays without the pruned experts, the size-only
overlay check and the 4-bit KV sizing. CPU only; torch parts skip without torch."""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from test_expert_quant import tiny_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_static_act_order_gptq_and_pooled_prior():
    torch = pytest.importorskip("torch")
    from arena import expert_quant_math as qm

    gen = torch.Generator().manual_seed(4)
    experts, n, k, tokens = 2, 32, 128, 3000
    w = (torch.randn(experts, n, k, generator=gen) * 0.02).to(torch.bfloat16)
    scale = torch.rand(k, generator=gen) * 3 + 0.2  # uneven input feature scales: act-order's case
    mix = torch.randn(k, k, generator=gen) * 0.3 + torch.eye(k)
    x = ((torch.randn(tokens, k, generator=gen) @ mix) * scale).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(experts, generator=gen)[:1] for _ in range(tokens)])
    wts = torch.rand(tokens, 1, generator=gen)
    hess, counts = qm.hessians(x, ids, wts, range(experts), experts)
    s2 = qm.scale2_of(w)
    eye = torch.eye(k).expand(experts, k, k).clone()
    p, s = qm.gptq_static(w, eye, s2, act_order=True)
    rp, rs = qm.rtn(w, s2)
    assert torch.equal(p, rp) and torch.equal(s.view(torch.uint8), rs.view(torch.uint8))  # identity Hessian = RTN

    def err(packed, scales):
        d = qm.dequantize(packed, scales, s2)
        total = 0.0
        for e in range(experts):
            rows = x[ids[:, 0] == e].float()
            total += qm.relative_error(rows @ d[e].t(), rows @ w[e].float().t())
        return total / experts

    h = hess.clone()
    qm.damp_hessian(h, counts)
    dynamic = err(*qm.gptq(w, qm.inverse_cholesky_upper(h), s2))
    act = err(*qm.gptq_static(w, h, s2, act_order=True))
    assert act < dynamic < err(rp, rs)
    # pooled prior: an expert without rows gets the layer's pooled moment instead of the identity
    empty = hess.clone()
    empty[1] = 0
    c = counts.clone()
    c[1] = 0
    pooled = qm.pool_hessians(empty, c, 256)
    assert torch.allclose(empty[1], 256 * pooled) and torch.allclose(pooled, hess[0] / counts[0])
    assert qm.remap_pruned_ids(torch.tensor([[0, 2]], dtype=torch.int16), [1, 5, 7]).tolist() == [[1, 7]]


def test_gptq_program_mask_aware_and_omitted_pruned_shards(tmp_path, monkeypatch):
    bench, model, bf16, capture = tiny_checkpoint(tmp_path, monkeypatch, rows=64)
    import numpy as np
    from safetensors import safe_open

    from arena import expert_overlay
    from arena import expert_quant_gptq as gq

    # A pruned server (expert 2 of 4 dropped) routes in the kept ranks 0..2 = experts [0, 1, 3].
    mask = {"format": "arc-expert-prune-v1", "experts": 4, "pruned_per_layer": 1, "pruned": {"3": [2]}}
    mask_path = tmp_path / "mask.json"
    mask_path.write_text(json.dumps(mask))
    rng = np.random.default_rng(0)
    for split in ("calib8", "eval"):
        path = capture / split / "L03.ids"
        ids = np.fromfile(path, dtype=np.int16).reshape(-1, 2)
        ids[:] = np.stack([rng.permutation(3)[:2] for _ in range(ids.shape[0])])
        ids.tofile(path)
    overlay = tmp_path / "overlay"
    assert gq.main(["--calib", str(capture / "calib8"), "--capture", str(capture), "--model-dir", str(model),
                    "--bf16-dir", str(bf16), "--overlay-out", str(overlay), "--out", str(tmp_path / "g.json"),
                    "--layers", "3", "--deadline-seconds", "600", "--eval-rows", "64", "--batch", "2",
                    "--blocksize", "16", "--device", "cpu", "--prior-rows", "16", "--act-order",
                    "--prune-mask", str(mask_path), "--omit-pruned"]) == 0
    layer = json.loads((tmp_path / "g.json").read_text())["layers"]["3"]
    assert layer["experts_gptq"] == 3 and layer["row_counts"][2] == 0
    assert layer["errors"]["gptq_w4a16"]["vs_bf16"] < layer["errors"]["checkpoint_w4a16"]["vs_bf16"]
    with safe_open(str(overlay / "layer.safetensors"), "pt") as fh:
        names = set(fh.keys())
    assert not any(".experts.2." in n for n in names) and any(".experts.3." in n for n in names)
    manifest = json.loads((overlay / "overlay-manifest.json").read_text())
    mask_sha = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    assert manifest["omits_pruned"] and manifest["expert_prune_sha256"] == mask_sha
    manifest["complete"] = True
    (overlay / "overlay-manifest.json").write_text(json.dumps(manifest))
    digest = hashlib.sha256((overlay / "overlay-manifest.json").read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="pruning mask"):
        expert_overlay.materialise(model, overlay, tmp_path / "farm", digest, log=lambda *_: None)
    record = expert_overlay.materialise(model, overlay, tmp_path / "farm", digest, log=lambda *_: None, check="size",
                                        expert_prune_sha256=mask_sha)
    assert record["omits_pruned"] and record["check"] == "size"


def test_overlay_spec_check_option():
    from arena import expert_overlay

    spec = {"dataset": "a/b", "manifest_sha256": "0" * 64}
    assert expert_overlay.validate({**spec, "check": "size"})["check"] == "size"
    assert expert_overlay.validate(spec) == spec
    with pytest.raises(ValueError):
        expert_overlay.validate({**spec, "check": "crc"})


def test_kv4_sizing_reproduces_the_hero_layout():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import kv4_sizing
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    rows = kv4_sizing.table()  # asserts the served kv8 hero layout (860 blocks, 51 per 64K, 15 fit)
    assert rows["none/int4"]["64k"]["fit"] == 23 and rows["none/kv8"]["64k"]["fit"] == 15
    assert rows["prune25/int4"]["128k"]["fit"] == 27


def test_mask_aware_calibrate_build(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_arena_p3", ROOT / "scripts/build_arena.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("calib.json", "eval.json"):
        (inputs / name).write_text("[]")
    (inputs / "selection.json").write_text(json.dumps({"eval_families": ["cn04"], "holdout_excluded": ["sb26"]}))
    output, study = builder.build("c", output=tmp_path / "c", paired_30m=True, expert_quant=inputs,
                                  expert_quant_phase="calibrate", expert_quant_settings=builder.EXPERT_QUANT_V2,
                                  expert_quant_prune="fixed-2x-mtp0-wide-kv8-hero-prune25")
    assert study["server"]["expert_prune"]["mask"] == "configs/expert-prune/prune25.json"
    assert (output / "configs/expert-prune/prune25.json").is_file()
    assert study["expert_quant"]["prior_rows"] == 4096 and study["expert_quant"]["act_order"] is True
    import ast

    tree = ast.parse((ROOT / "scripts/modal_validation.py").read_text())
    ns = {}
    for name in ("validation_timeout", "arena_arguments"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_validation", "exec"), ns)
    timeout = ns["validation_timeout"]("arena", "expert-quant-calibrate-v2")
    assert ns["arena_arguments"]("expert-quant-calibrate-v2") == ["--trial", "expert-quant"]
    assert study["notebook_seconds"] == timeout - 60
    assert (timeout + 60) * (0.000842 + 8 * 0.0000131 + 192 * 0.00000222) * 1.15 <= 4.0 # <=
    with pytest.raises(ValueError):
        builder.build("d", output=tmp_path / "d", paired_30m=True, expert_quant=inputs, expert_quant_phase="calibrate",
                      expert_quant_settings={"bogus": 1})


def test_overlay_farm_is_the_same_without_the_replaced_base_shards(tmp_path):
    """20 Oct: the Nebius runner may skip fetching the base shards an omits-pruned overlay replaces."""
    from arena import expert_overlay

    base, overlay = tmp_path / "base", tmp_path / "overlay"
    base.mkdir()
    overlay.mkdir()
    shards = {"layer-00000-experts-0000-0127.safetensors": b"o" * 7}
    weight_map = {"a.w": "model-bf16-00001.safetensors", "e.w": "layer-00000-experts-0000-0127.safetensors"}
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (base / "model-bf16-00001.safetensors").write_bytes(b"b" * 5)
    (base / "layer-00000-experts-0000-0127.safetensors").write_bytes(b"x" * 9)
    (base / "layer-00000-experts-0000-0127.complete.json").write_text("{}")
    for name, body in shards.items():
        (overlay / name).write_bytes(body)
    mask = "f" * 64
    manifest = {"format": expert_overlay.FORMAT, "complete": True, "omits_pruned": True, "expert_prune_sha256": mask,
                "shards": {n: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in shards.items()}}
    (overlay / expert_overlay.MANIFEST).write_text(json.dumps(manifest))
    digest = hashlib.sha256((overlay / expert_overlay.MANIFEST).read_bytes()).hexdigest()

    def farm(base_dir, dest):
        expert_overlay.materialise(base_dir, overlay, dest, digest, log=lambda *_: None, expert_prune_sha256=mask)
        return {f.name: os.readlink(f) for f in dest.iterdir()}

    full = farm(base, tmp_path / "farm")
    sparse = tmp_path / "sparse"
    shutil.copytree(base, sparse)
    (sparse / "layer-00000-experts-0000-0127.safetensors").unlink()
    assert farm(sparse, tmp_path / "farm-sparse") == {
        name: target.replace(str(base.resolve()), str(sparse.resolve())) for name, target in full.items()}
    assert full["layer-00000-experts-0000-0127.safetensors"] == str(overlay / "layer-00000-experts-0000-0127.safetensors")
