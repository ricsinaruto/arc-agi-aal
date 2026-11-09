import json
import struct

import pytest

from arena import expert_overlay, expert_strip


def write_safetensors(path, tensors, metadata=None):
    """Minimal safetensors writer: name -> bytes (dtype U8, shape [len])."""
    header, offset, blobs = {}, 0, []
    for name, data in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(data)], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def read_safetensors(path):
    size, header = expert_strip.read_header(path)
    body = path.read_bytes()[8 + size:]
    return {k: body[v["data_offsets"][0]:v["data_offsets"][1]] for k, v in header.items() if k != "__metadata__"}


def test_strip_drops_only_the_mask_experts_and_keeps_bytes_exact(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    tensors = {f"model.language_model.layers.{layer}.mlp.experts.{e}.w{k}": bytes([10 * e + k]) * (7 + e)
               for layer in (0, 1) for e in range(4) for k in (1, 2)}
    tensors["model.language_model.layers.0.mlp.gate.weight"] = b"g" * 16
    shard = "layer-00000-experts-0000-0003.safetensors"
    write_safetensors(src / shard, tensors, {"a": "b"})
    manifest = {"format": expert_overlay.FORMAT, "complete": True,
                "shards": {shard: {"bytes": (src / shard).stat().st_size, "sha256": "x"}}}
    (src / expert_overlay.MANIFEST).write_text(json.dumps(manifest))
    mask = tmp_path / "mask.json"
    mask.write_text(json.dumps({"experts": 4, "pruned": {"0": [1, 3], "1": [0]}}))
    totals = expert_strip.strip_overlay(src, dst, mask, log=lambda *_: None)
    out = read_safetensors(dst / shard)
    dropped = {(0, 1), (0, 3), (1, 0)}
    assert totals["dropped"] == 6 and totals["kept"] == len(tensors) - 6
    assert set(out) == {k for k in tensors if not any(f"layers.{la}.mlp.experts.{e}." in k for la, e in dropped)}
    assert all(out[k] == tensors[k] for k in out)
    size, header = expert_strip.read_header(dst / shard)
    assert header["__metadata__"] == {"a": "b"} and (8 + size) % 8 == 0
    written = json.loads((dst / expert_overlay.MANIFEST).read_text())
    assert written["omits_pruned"] and written["shards"][shard]["bytes"] < manifest["shards"][shard]["bytes"]
    with pytest.raises(RuntimeError):
        expert_strip.strip_overlay(dst, tmp_path / "again", mask)
