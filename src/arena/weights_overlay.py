"""Weights overlays: a served checkpoint with some tensors replaced (the merged and re-quantized LoRA
of ``docs/posttrain/lora-test-plan.md``), shipped as the replaced tensors only.

An overlay directory holds ``overlay.safetensors`` (the replacement tensors, under the checkpoint's own
tensor names) and ``overlay-manifest.json``:

- ``base``: the checkpoint identity (``repo``, ``revision``) and the inventory entry (bytes, sha256) of
  every shard the overlay patches;
- ``overlay``: the overlay file's bytes and sha256;
- ``tensors``: per replaced tensor its shard, dtype, shape and sha256;
- ``patched``: per patched shard the bytes and sha256 after the replacement;
- anything else the producer records (adapter, training, re-quantization checks).

A replacement keeps each tensor's dtype and shape, so a patched shard has the same header and the same
byte layout as the base shard: materialising copies the shard once with the replaced byte ranges
substituted (``stream_patched``) and checks the result against ``patched``. The index
(``model.safetensors.index.json``) is unchanged (same names, same shards). Every other file of the
checkpoint is symlinked. Stdlib only: it runs in the arena bootstrap and in the Modal merge job.
"""

import hashlib
import json
import os
import shutil
import struct
import time
from pathlib import Path

FORMAT = "arc-weights-overlay-v1"
OVERLAY_FILE = "overlay.safetensors"
MANIFEST_FILE = "overlay-manifest.json"
RECORD_FILE = "weights-overlay.json"
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "BF16": 2, "F16": 2, "I16": 2,
               "U16": 2, "F32": 4, "I32": 4, "U32": 4, "F64": 8, "I64": 8, "U64": 8}
CHUNK = 64 << 20
SPACE_MARGIN = 2 << 30


# --- safetensors -----------------------------------------------------------------------------------


def read_header(path):
    """``(data_start, tensors, metadata)``: tensors map names to dtype, shape and data offsets."""
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    metadata = header.pop("__metadata__", None)
    return 8 + size, header, metadata


def tensor_bytes(dtype, shape):
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * DTYPE_BYTES[dtype]


def write_safetensors(path, tensors, metadata=None):
    """Write ``{name: (dtype, shape, data bytes)}`` as a safetensors file (names sorted, data contiguous,
    header padded with spaces to 8 bytes as the reference writer does)."""
    header, offset = {}, 0
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        if len(data) != tensor_bytes(dtype, shape):
            raise ValueError(f"{name}: {len(data)} bytes for {dtype} {list(shape)}")
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        for name in sorted(tensors):
            handle.write(tensors[name][2])


def read_tensor(path, name, header=None):
    data_start, tensors, _ = header or read_header(path)
    start, end = tensors[name]["data_offsets"]
    with open(path, "rb") as handle:
        handle.seek(data_start + start)
        return handle.read(end - start)


def sha256_file(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# --- patch plan and streaming ----------------------------------------------------------------------


def patch_plan(base_dir, overlay_path, weight_map=None):
    """``{shard: [(absolute offset, length, name), ...]}`` sorted by offset. Every overlay tensor must
    exist in the base shard the index names, with the same dtype and shape."""
    base_dir = Path(base_dir)
    weight_map = weight_map or json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    _, overlay, _ = read_header(overlay_path)
    plan, headers = {}, {}
    for name, spec in sorted(overlay.items()):
        shard = weight_map.get(name)
        if shard is None:
            raise ValueError(f"overlay tensor {name} is not in the checkpoint index")
        if shard not in headers:
            headers[shard] = read_header(base_dir / shard)
        data_start, tensors, _ = headers[shard]
        base = tensors.get(name)
        if base is None:
            raise ValueError(f"overlay tensor {name} is not in {shard}")
        if (base["dtype"], list(base["shape"])) != (spec["dtype"], list(spec["shape"])):
            raise ValueError(f"{name}: overlay {spec['dtype']} {spec['shape']} differs from the base "
                             f"{base['dtype']} {base['shape']}")
        start, end = base["data_offsets"]
        plan.setdefault(shard, []).append((data_start + start, end - start, name))
    for entries in plan.values():
        entries.sort()
        for (a, n, _), (b, _, name) in zip(entries, entries[1:]):
            if a + n > b:
                raise ValueError(f"overlapping replacements at {name}")
    return plan


def stream_patched(base_path, replacements, overlay_path, *, out=None, chunk=CHUNK):
    """Stream ``base_path`` with the byte ranges of ``replacements`` (``patch_plan`` entries) taken from
    the overlay file; write to ``out`` when given. Returns ``(bytes, sha256)`` of the patched stream."""
    digest, total = hashlib.sha256(), 0
    overlay_header = read_header(overlay_path)
    sink = open(out, "wb") if out else None
    try:
        with open(base_path, "rb") as base:
            size = os.fstat(base.fileno()).st_size
            position = 0

            def emit(data):
                nonlocal total
                digest.update(data)
                total += len(data)
                if sink:
                    sink.write(data)

            def copy_until(stop):
                nonlocal position
                base.seek(position)
                while position < stop:
                    data = base.read(min(chunk, stop - position))
                    if not data:
                        raise ValueError(f"{base_path} ended at {position}, expected {stop}")
                    emit(data)
                    position += len(data)

            for offset, length, name in replacements:
                copy_until(offset)
                data = read_tensor(overlay_path, name, overlay_header)
                if len(data) != length:
                    raise ValueError(f"{name}: {len(data)} overlay bytes for a {length}-byte slot")
                emit(data)
                position = offset + length
            copy_until(size)
    finally:
        if sink:
            sink.close()
    return total, digest.hexdigest()


# --- producing an overlay --------------------------------------------------------------------------


def write_overlay(out_dir, base_dir, tensors, *, base_identity, base_files, extra=None, log=print):
    """Write ``overlay.safetensors`` with ``tensors`` (``{name: (dtype, shape, bytes)}``) and the manifest,
    computing every patched shard's hash by streaming (nothing is written besides the overlay).
    ``base_files`` is the checkpoint inventory (``{file: {bytes, sha256}}``)."""
    out_dir, base_dir = Path(out_dir), Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / OVERLAY_FILE
    write_safetensors(overlay_path, tensors, metadata={"format": FORMAT})
    plan = patch_plan(base_dir, overlay_path)
    patched, changed = {}, {}
    for shard, entries in sorted(plan.items()):
        started = time.monotonic()
        size, digest = stream_patched(base_dir / shard, entries, overlay_path)
        patched[shard] = {"bytes": size, "sha256": digest, "tensors": len(entries)}
        changed[shard] = digest != base_files[shard]["sha256"]
        log(f"{shard}: {len(entries)} tensors, patched sha256 {digest[:16]} "
            f"({'changed' if changed[shard] else 'identical to the base'}), {time.monotonic() - started:.0f} s")
    manifest = {
        "format": FORMAT,
        "base": {**base_identity, "files": {s: dict(base_files[s]) for s in sorted(plan)},
                 "index_sha256": base_files["model.safetensors.index.json"]["sha256"]},
        "overlay": {"file": OVERLAY_FILE, "bytes": overlay_path.stat().st_size, "sha256": sha256_file(overlay_path),
                    "tensors": len(tensors)},
        "tensors": {name: {"shard": shard, "dtype": tensors[name][0], "shape": list(tensors[name][1]),
                           "sha256": sha256_bytes(tensors[name][2])}
                    for shard, entries in plan.items() for _, _, name in entries},
        "patched": patched,
        "identical_to_base": not any(changed.values()),
        **(extra or {}),
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


# --- materialising (arena bootstrap) ---------------------------------------------------------------


def load_manifest(overlay_dir, expected_sha256=None):
    path = Path(overlay_dir) / MANIFEST_FILE
    raw = path.read_bytes()
    if expected_sha256 and sha256_bytes(raw) != expected_sha256:
        raise RuntimeError(f"{path}: manifest sha256 differs from the pinned {expected_sha256}")
    manifest = json.loads(raw)
    if manifest.get("format") != FORMAT:
        raise RuntimeError(f"{path}: unknown overlay format {manifest.get('format')!r}")
    return manifest


def check_base(manifest, base_inventory):
    """The overlay was made for this checkpoint: identity and every patched shard's inventory entry."""
    base = manifest["base"]
    if (base["repo"], base["revision"]) != (base_inventory["repo"], base_inventory["revision"]):
        raise RuntimeError(f"overlay base {base['repo']}@{base['revision']} is not the served checkpoint")
    for shard, spec in base["files"].items():
        entry = base_inventory["files"].get(shard) or {}
        if (entry.get("bytes"), entry.get("sha256")) != (spec["bytes"], spec["sha256"]):
            raise RuntimeError(f"overlay base shard {shard} differs from the checkpoint inventory")
    if base_inventory["files"]["model.safetensors.index.json"]["sha256"] != base["index_sha256"]:
        raise RuntimeError("overlay base index differs from the checkpoint inventory")
    if set(manifest["patched"]) != set(base["files"]):
        raise RuntimeError("overlay patches shards outside its base inventory")


def materialise(base_dir, overlay_dir, dest, *, base_inventory, expected_manifest_sha256=None, log=print):
    """Build ``dest``: the base checkpoint with the overlay applied (patched shards copied, every other
    file symlinked), each patched shard checked against the manifest. Returns the record."""
    started = time.monotonic()
    base_dir, overlay_dir, dest = Path(base_dir), Path(overlay_dir), Path(dest)
    manifest = load_manifest(overlay_dir, expected_manifest_sha256)
    check_base(manifest, base_inventory)
    overlay_path = overlay_dir / manifest["overlay"]["file"]
    if (overlay_path.stat().st_size, sha256_file(overlay_path)) != (manifest["overlay"]["bytes"],
                                                                     manifest["overlay"]["sha256"]):
        raise RuntimeError(f"{overlay_path}: overlay file differs from its manifest")
    plan = patch_plan(base_dir, overlay_path)
    if set(plan) != set(manifest["patched"]):
        raise RuntimeError("overlay tensors fall in other shards than the manifest records")
    planned = {name for entries in plan.values() for _, _, name in entries}
    if planned != set(manifest["tensors"]):
        raise RuntimeError("overlay file and manifest list different tensors")
    need = sum(manifest["patched"][s]["bytes"] for s in plan)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    free = shutil.disk_usage(dest).free
    if free < need + SPACE_MARGIN:
        raise RuntimeError(f"{dest}: {free / 2**30:.1f} GiB free, the patched shards need {need / 2**30:.1f} GiB")
    for item in sorted(base_dir.iterdir()):
        if item.name not in plan:
            (dest / item.name).symlink_to(item.resolve())
    shards = {}
    for shard, entries in sorted(plan.items()):
        size, digest = stream_patched(base_dir / shard, entries, overlay_path, out=dest / shard)
        expected = manifest["patched"][shard]
        if (size, digest) != (expected["bytes"], expected["sha256"]):
            raise RuntimeError(f"patched {shard} does not match the overlay manifest")
        shards[shard] = {"bytes": size, "sha256": digest, "tensors": len(entries)}
        log(f"weights overlay: {shard} patched ({len(entries)} tensors), sha256 verified")
    record = {"format": FORMAT, "dest": str(dest), "overlay_dir": str(overlay_dir),
              "manifest_sha256": sha256_bytes((overlay_dir / MANIFEST_FILE).read_bytes()),
              "overlay_sha256": manifest["overlay"]["sha256"], "tensors": len(planned), "patched": shards,
              "identical_to_base": manifest.get("identical_to_base"), "free_bytes_before": free,
              "seconds": round(time.monotonic() - started, 1)}
    (dest / RECORD_FILE).write_text(json.dumps(record, indent=1) + "\n")
    return record


def materialise_for_study(spec, base_dir, base_inventory, input_root, working):
    """Arena bootstrap hook: ``spec`` is the study's ``weights_overlay`` entry (``dataset`` and
    ``manifest_sha256``); the patched checkpoint goes to ``working/patched-checkpoint``."""
    from arc_infra.serving import dataset_mount

    overlay_dir = dataset_mount(spec["dataset"], input_root)
    record = materialise(base_dir, overlay_dir, Path(working) / "patched-checkpoint",
                         base_inventory=base_inventory, expected_manifest_sha256=spec["manifest_sha256"])
    return Path(record["dest"]), record
