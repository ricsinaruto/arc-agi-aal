"""The relaxed strict overlay check of heldout_bundle.py (A's dense p75 shards ride in the expert overlay)."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
STRICT = Path(os.environ.get("HK_STRICT_OVERLAY", "./.claude/worktrees/v84sub/dist/"
                                                  "v87b-sub-prev600/src/arena/expert_overlay.py"))
pytestmark = pytest.mark.skipif(not STRICT.exists(), reason="v87b's expert_overlay.py not on this machine")

EXPERT = [f"layer-{l:05d}-experts-{i:04d}-0002.safetensors" for l in range(2) for i in range(2)]
DENSE = "model-bf16-00001.safetensors"


def overlay_module(tmp_path):
    spec = importlib.util.spec_from_file_location("hb", HERE / "heldout_bundle.py")
    hb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hb)
    path = tmp_path / "expert_overlay_relaxed.py"
    path.write_text(hb.relaxed_overlay_source(STRICT))
    spec = importlib.util.spec_from_file_location("eo", path)
    eo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eo)
    return eo


def setup(tmp_path, overlay_shards):
    base, ov = tmp_path / "base", tmp_path / "ov"
    base.mkdir()
    ov.mkdir()
    weight_map = {f"t{i}": name for i, name in enumerate(EXPERT + [DENSE])}
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (base / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 2}}))
    for name in EXPERT + [DENSE]:
        (base / name).write_bytes(b"b" * 8)
    shards = {}
    for name in overlay_shards:
        (ov / name).write_bytes(b"o" * 8)
        shards[name] = {"bytes": 8, "sha256": hashlib.sha256(b"o" * 8).hexdigest()}
    raw = json.dumps({"format": "arc-expert-overlay-v1", "shards": shards, "layers": [0, 1], "complete": True}).encode()
    (ov / "overlay-manifest.json").write_bytes(raw)
    return base, ov, hashlib.sha256(raw).hexdigest()


def run(tmp_path, shards, check="sha256", corrupt=()):
    eo = overlay_module(tmp_path)
    base, ov, digest = setup(tmp_path, shards)
    for name in corrupt:  # same size, other bytes
        (ov / name).write_bytes(b"x" * 8)
    return (eo.materialise(base, ov, tmp_path / "farm", digest, log=lambda *_: None, check=check),
            tmp_path / "farm")


def test_expert_shards_only_pass(tmp_path):
    rec, farm = run(tmp_path, EXPERT)
    assert rec["shards"] == 4 and (farm / EXPERT[0]).resolve() == (tmp_path / "ov" / EXPERT[0])


def test_extra_dense_shard_passes(tmp_path):
    rec, farm = run(tmp_path, EXPERT + [DENSE])
    assert rec["shards"] == 5 and (farm / DENSE).resolve() == (tmp_path / "ov" / DENSE)


def test_missing_expert_shard_fails(tmp_path):
    with pytest.raises(RuntimeError, match="partial expert overlay"):
        run(tmp_path, EXPERT[1:] + [DENSE])


def test_extra_shard_not_in_index_fails(tmp_path):
    with pytest.raises(RuntimeError, match="not in the checkpoint index"):
        run(tmp_path, EXPERT + ["model-bf16-00099.safetensors"])


def test_size_check_hashes_extra_shard(tmp_path):
    with pytest.raises(RuntimeError, match="sha256 differs"):
        run(tmp_path, EXPERT + [DENSE], check="size", corrupt=[DENSE])


def test_size_check_skips_expert_hash(tmp_path):
    rec, _ = run(tmp_path, EXPERT + [DENSE], check="size", corrupt=[EXPERT[0]])
    assert rec["check"] == "size" and rec["shards"] == 5
