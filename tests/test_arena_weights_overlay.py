"""Weights overlays (the LoRA test's serving path): writing, materialising on a fake checkpoint, the
arena builder flag and the candidate-serving hook."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from arena import weights_overlay as wo

ROOT = Path(__file__).resolve().parents[1]
ORNITH = json.loads((ROOT / "src/arena/models/ornith15-35b-a3b-nvfp4.json").read_text())
IDENTITY = {"repo": ORNITH["repo"], "revision": ORNITH["revision"]}


def tensors_of(seed):
    data = lambda n: bytes((seed * 31 + i) % 251 for i in range(n))  # noqa: E731
    return {
        "a.weight": ("F8_E4M3", [4, 8], data(32)),
        "a.weight_scale": ("F32", [], data(4)),
        "a.input_scale": ("F32", [], data(4)),
        "b.weight": ("U8", [8, 4], data(32)),
        "b.weight_scale": ("F8_E4M3", [8, 2], data(16)),
        "c.norm": ("BF16", [16], data(32)),
    }


def fake_checkpoint(folder):
    """Two shards, an index, a config and the inventory (hashes of every file)."""
    folder.mkdir(parents=True)
    base = tensors_of(1)
    shards = {"model-00001-of-00002.safetensors": ["a.weight", "a.weight_scale", "a.input_scale"],
              "model-00002-of-00002.safetensors": ["b.weight", "b.weight_scale", "c.norm"]}
    for shard, names in shards.items():
        wo.write_safetensors(folder / shard, {n: base[n] for n in names})
    (folder / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {n: s for s, names in shards.items() for n in names}}))
    (folder / "config.json").write_text("{}")
    inventory = {**IDENTITY, "files": {p.name: {"bytes": p.stat().st_size, "sha256": wo.sha256_file(p)}
                                       for p in folder.iterdir()}}
    return base, shards, inventory


def test_write_and_read_safetensors_round_trip(tmp_path):
    tensors = tensors_of(3)
    wo.write_safetensors(tmp_path / "x.safetensors", tensors, metadata={"format": "test"})
    start, header, metadata = wo.read_header(tmp_path / "x.safetensors")
    assert start % 8 == 0 and metadata == {"format": "test"}
    for name, (dtype, shape, data) in tensors.items():
        assert header[name]["dtype"] == dtype and header[name]["shape"] == shape
        assert wo.read_tensor(tmp_path / "x.safetensors", name) == data
    with pytest.raises(ValueError, match="bytes"):
        wo.write_safetensors(tmp_path / "y.safetensors", {"z": ("F32", [2], b"123")})


def test_overlay_materialises_to_the_expected_shards(tmp_path):
    base, shards, inventory = fake_checkpoint(tmp_path / "base")
    new = tensors_of(2)
    replaced = {n: new[n] for n in ("a.weight", "a.weight_scale", "b.weight", "b.weight_scale")}
    manifest = wo.write_overlay(tmp_path / "overlay", tmp_path / "base", replaced, base_identity=IDENTITY,
                                base_files=inventory["files"], extra={"adapter": {"sha256": "ab"}},
                                log=lambda *a: None)
    assert manifest["identical_to_base"] is False and manifest["overlay"]["tensors"] == 4
    assert set(manifest["patched"]) == set(shards) and manifest["adapter"] == {"sha256": "ab"}
    digest = wo.sha256_file(tmp_path / "overlay" / wo.MANIFEST_FILE)
    record = wo.materialise(tmp_path / "base", tmp_path / "overlay", tmp_path / "patched",
                            base_inventory=inventory, expected_manifest_sha256=digest, log=lambda *a: None)
    # the patched shards are exactly what writing the new tensors in place gives
    expected = tmp_path / "expected"
    expected.mkdir()
    merged = {**base, **replaced}
    for shard, names in shards.items():
        wo.write_safetensors(expected / shard, {n: merged[n] for n in names})
        assert (tmp_path / "patched" / shard).read_bytes() == (expected / shard).read_bytes()
        assert not (tmp_path / "patched" / shard).is_symlink()
    assert (tmp_path / "patched/config.json").is_symlink()
    assert (tmp_path / "patched/model.safetensors.index.json").read_text() == \
        (tmp_path / "base/model.safetensors.index.json").read_text()
    assert record["tensors"] == 4 and json.loads((tmp_path / "patched" / wo.RECORD_FILE).read_text())["tensors"] == 4


def test_zero_overlay_is_identical_to_the_base(tmp_path):
    base, shards, inventory = fake_checkpoint(tmp_path / "base")
    manifest = wo.write_overlay(tmp_path / "overlay", tmp_path / "base", {"a.weight": base["a.weight"]},
                                base_identity=IDENTITY, base_files=inventory["files"], log=lambda *a: None)
    shard = "model-00001-of-00002.safetensors"
    assert manifest["identical_to_base"] and manifest["patched"][shard]["sha256"] == inventory["files"][shard]["sha256"]


def test_materialise_rejects_tampering_and_other_bases(tmp_path):
    base, _, inventory = fake_checkpoint(tmp_path / "base")
    new = tensors_of(5)
    wo.write_overlay(tmp_path / "overlay", tmp_path / "base", {"b.weight": new["b.weight"]}, base_identity=IDENTITY,
                     base_files=inventory["files"], log=lambda *a: None)
    kwargs = {"base_inventory": inventory, "log": lambda *a: None}
    with pytest.raises(RuntimeError, match="pinned"):
        wo.materialise(tmp_path / "base", tmp_path / "overlay", tmp_path / "p1", expected_manifest_sha256="0" * 64,
                       **kwargs)
    other = {**inventory, "revision": "0" * 40}
    with pytest.raises(RuntimeError, match="not the served checkpoint"):
        wo.materialise(tmp_path / "base", tmp_path / "overlay", tmp_path / "p2", base_inventory=other,
                       log=lambda *a: None)
    changed = json.loads(json.dumps(inventory))
    changed["files"]["model-00002-of-00002.safetensors"]["sha256"] = "1" * 64
    with pytest.raises(RuntimeError, match="differs from the checkpoint inventory"):
        wo.materialise(tmp_path / "base", tmp_path / "overlay", tmp_path / "p3", base_inventory=changed,
                       log=lambda *a: None)
    overlay = tmp_path / "overlay" / wo.OVERLAY_FILE
    raw = bytearray(overlay.read_bytes())
    raw[-1] ^= 0xFF
    overlay.write_bytes(bytes(raw))
    with pytest.raises(RuntimeError, match="differs from its manifest"):
        wo.materialise(tmp_path / "base", tmp_path / "overlay", tmp_path / "p4", **kwargs)


def test_overlay_must_keep_dtype_shape_and_names(tmp_path):
    base, _, inventory = fake_checkpoint(tmp_path / "base")
    with pytest.raises(ValueError, match="differs from the base"):
        wo.write_overlay(tmp_path / "o1", tmp_path / "base", {"a.weight": ("U8", [4, 8], bytes(32))},
                         base_identity=IDENTITY, base_files=inventory["files"], log=lambda *a: None)
    with pytest.raises(ValueError, match="not in the checkpoint index"):
        wo.write_overlay(tmp_path / "o2", tmp_path / "base", {"zzz": ("U8", [1], b"x")},
                         base_identity=IDENTITY, base_files=inventory["files"], log=lambda *a: None)


def test_materialise_for_study_uses_the_mounted_dataset(tmp_path, monkeypatch):
    base, _, inventory = fake_checkpoint(tmp_path / "base")
    mount = tmp_path / "input/datasets/richardcsaky/arc-ornith-lora-test"
    wo.write_overlay(mount, tmp_path / "base", {"b.weight": tensors_of(7)["b.weight"]}, base_identity=IDENTITY,
                     base_files=inventory["files"], log=lambda *a: None)
    spec = {"dataset": "richardcsaky/arc-ornith-lora-test",
            "manifest_sha256": wo.sha256_file(mount / wo.MANIFEST_FILE)}
    path, record = wo.materialise_for_study(spec, tmp_path / "base", inventory, tmp_path / "input", tmp_path / "work")
    assert path == tmp_path / "work/patched-checkpoint" and record["tensors"] == 1


def test_prepare_overlay_dataset_checks_the_zero_adapter_record(tmp_path):
    spec = importlib.util.spec_from_file_location("prep", ROOT / "scripts/posttrain/prepare_overlay_dataset.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    base, _, inventory = fake_checkpoint(tmp_path / "base")
    extra = {"requant": {"zero_adapter_check": {"tensors": 2, "exact": 2}}, "adapter": {"sha256": "cd"}}
    wo.write_overlay(tmp_path / "overlay", tmp_path / "base", {"b.weight": tensors_of(9)["b.weight"]},
                     base_identity=IDENTITY, base_files=inventory["files"], extra=extra, log=lambda *a: None)
    prep.prepare(tmp_path / "overlay", "richardcsaky/arc-ornith-lora-test", tmp_path / "upload", inventory=inventory)
    metadata = json.loads((tmp_path / "upload/dataset-metadata.json").read_text())
    assert metadata["id"] == "richardcsaky/arc-ornith-lora-test" and "cd" in metadata["description"]
    assert (tmp_path / "upload" / wo.OVERLAY_FILE).read_bytes() == (tmp_path / "overlay" / wo.OVERLAY_FILE).read_bytes()
    extra["requant"]["zero_adapter_check"]["exact"] = 1
    wo.write_overlay(tmp_path / "bad", tmp_path / "base", {"b.weight": tensors_of(9)["b.weight"]},
                     base_identity=IDENTITY, base_files=inventory["files"], extra=extra, log=lambda *a: None)
    with pytest.raises(ValueError, match="zero-adapter"):
        prep.prepare(tmp_path / "bad", "richardcsaky/arc-ornith-lora-test", tmp_path / "upload2", inventory=inventory)


# --- builder flag and serving hook -----------------------------------------------------------------


def overlay_folder(tmp_path, dataset="richardcsaky/arc-ornith-lora-test"):
    """An upload folder whose manifest names the real Ornith inventory (no shards needed for the build)."""
    folder = tmp_path / "overlay-upload"
    folder.mkdir(parents=True)
    shard = "model-00001-of-00003.safetensors"
    manifest = {"format": wo.FORMAT, "base": {**IDENTITY, "files": {shard: ORNITH["files"][shard]},
                                              "index_sha256": ORNITH["files"]["model.safetensors.index.json"]["sha256"]},
                "overlay": {"file": wo.OVERLAY_FILE, "bytes": 1, "sha256": "e" * 64, "tensors": 250},
                "tensors": {}, "patched": {shard: {"bytes": 1, "sha256": "f" * 64}}, "identical_to_base": False,
                "adapter": {"sha256": "a" * 64}}
    (folder / wo.MANIFEST_FILE).write_text(json.dumps(manifest))
    (folder / "dataset-metadata.json").write_text(json.dumps({"id": dataset}))
    return folder


def builder():
    spec = importlib.util.spec_from_file_location("builder", ROOT / "scripts/build_arena.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_pins_the_overlay_and_attaches_its_dataset(tmp_path):
    build = builder()
    folder = overlay_folder(tmp_path)
    output, study = build.build("ornith-overlay-test", output=tmp_path / "bundle", paired_30m=True,
                                candidate_profile="ornith15-35b-a3b-nvfp4", weights_overlay=folder)
    build.kaggle_files(output, study, smoke=False)
    spec = study["weights_overlay"]
    assert spec["dataset"] == "richardcsaky/arc-ornith-lora-test"
    assert spec["manifest_sha256"] == hashlib.sha256((folder / wo.MANIFEST_FILE).read_bytes()).hexdigest()
    assert spec["base"] == f"{IDENTITY['repo']}@{IDENTITY['revision']}" and spec["tensors"] == 250
    metadata = json.loads((output / "kernel-metadata.json").read_text())
    assert metadata["dataset_sources"][-2:] == ["richardcsaky/arc-ornith15-35b-a3b-nvfp4",
                                                "richardcsaky/arc-ornith-lora-test"]
    assert json.loads((output / "configs/comparison.json").read_text())["weights_overlay"] == spec
    assert (output / "src/arena/weights_overlay.py").is_file()
    plain_output, plain = build.build("ornith-plain-test", output=tmp_path / "plain", paired_30m=True,
                                      candidate_profile="ornith15-35b-a3b-nvfp4")
    assert "weights_overlay" not in plain and plain["model_datasets"] == ["richardcsaky/arc-ornith15-35b-a3b-nvfp4"]
    with pytest.raises(ValueError, match="candidate"):
        build.build("bad", output=tmp_path / "bad", paired_30m=True, weights_overlay=folder)
    other = overlay_folder(tmp_path / "other")
    manifest = json.loads((other / wo.MANIFEST_FILE).read_text())
    manifest["base"]["revision"] = "0" * 40
    (other / wo.MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="not the served checkpoint"):
        build.build("bad2", output=tmp_path / "bad2", paired_30m=True, candidate_profile="ornith15-35b-a3b-nvfp4",
                    weights_overlay=other)


def test_candidate_serving_starts_the_server_on_the_patched_checkpoint(tmp_path, monkeypatch):
    from arena import candidate_serving as candidate
    from arena import weights_overlay

    study = json.loads((ROOT / "configs/arena_30m_ornith15.json").read_text())
    study["weights_overlay"] = {"dataset": "richardcsaky/arc-ornith-lora-test", "manifest_sha256": "a" * 64}
    started = []
    recipe = NS(BASE_URL="http://localhost/v1", resolve_runtime_dir=lambda: tmp_path,
                verify_and_extract_runtime=lambda *a, **k: {}, patch_ple_layer=lambda: {},
                resolve_vllm_tuning=lambda: {}, runtime_environment=lambda **k: ({}, {}),
                start_server=lambda model, *a, **k: started.append(model) or {"pid": 1, "start_ticks": 1},
                update_ready_identity=lambda identity: identity, stop_owned_server=lambda: {"stopped": True})
    monkeypatch.setattr(candidate, "verify_hardware", lambda: "test GPU")
    monkeypatch.setattr(candidate, "needed", lambda: False)
    monkeypatch.setattr(candidate, "worker_probe", lambda *a: NS(returncode=0, stdout='{"type": "ready"}', stderr=""))
    monkeypatch.setattr(candidate, "dataset_mount", lambda *a: tmp_path / "base")
    monkeypatch.setattr(candidate, "verify_model", lambda *a, **k: {"verified": True})
    monkeypatch.setattr(candidate, "stage_fork_bundle", lambda *a: tmp_path)
    monkeypatch.setattr(candidate, "load_recipe", lambda *a, **k: recipe)
    monkeypatch.setattr(candidate, "configure", lambda *a, **k: None)
    monkeypatch.setattr(candidate, "configure_candidate", lambda *a: None)
    monkeypatch.setattr(candidate.BlockBudget, "load", lambda *a: NS(layout={}))
    seen = {}

    def materialise(spec, base, inventory, input_root, working):
        seen.update(spec=spec, base=base, repo=inventory["repo"], working=working)
        return tmp_path / "patched", {"tensors": 250}

    monkeypatch.setattr(weights_overlay, "materialise_for_study", materialise)

    async def ready(*a, **k):
        return [{"id": study["served_model"]}]

    async def preflight(config):
        return {"profile": {}}

    monkeypatch.setattr(candidate, "ready", ready)
    monkeypatch.setattr(candidate, "preflight", preflight)
    candidate.start(study, tmp_path, "/inputs")
    assert started == [tmp_path / "patched"] and seen["base"] == tmp_path / "base"
    assert seen["spec"] == study["weights_overlay"] and seen["repo"] == IDENTITY["repo"]
    record = json.loads((tmp_path / "candidate-preflight.json").read_text())
    assert record["status"] == "ready" and record["weights_overlay"] == {"tensors": 250}
