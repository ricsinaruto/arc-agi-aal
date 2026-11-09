"""Serve the checkpoint with replacement routed-expert shards (28 September; docs/validation-20260927/expert-quant.md).

An expert overlay (``arena.expert_quant_gptq``) is a directory of whole replacement shards, named, laid out and typed
exactly as the served checkpoint's ``layer-000LL-experts-XXXX-YYYY.safetensors`` shards, plus
``overlay-manifest.json`` (``format``, ``shards``: bytes and sha256 per shard, ``complete``). ``materialise`` builds
a symlink farm in the working directory: every checkpoint file links to the base, except the overlay's shards, which
link to the overlay after a size and sha256 check. The index (``model.safetensors.index.json``) is unchanged, so vLLM
loads the same tensor names with no code change; the serving command's model path is the farm.

A study selects it with ``server.expert_overlay = {"dataset": <owner/name>, "manifest_sha256": <sha256>}``; the
serving-lab variable ``ARENA_EXPERT_OVERLAY`` (``off`` or that JSON) overrides it per trial. Stdlib only.
"""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FORMAT = "arc-expert-overlay-v1"
MANIFEST = "overlay-manifest.json"
RECORD = "expert-overlay.json"


CHECKS = ("sha256", "size")


def validate(spec):
    """Normalize ``server.expert_overlay``: None, or a dict with ``dataset`` and ``manifest_sha256`` and optionally
    ``check`` (``sha256``, the default: every shard hashed at start; ``size``: sizes only, for an immutable pinned
    dataset version where the notebook cannot afford reading the shards twice, e.g. Kaggle)."""
    if spec is None:
        return None
    if (not isinstance(spec, dict) or not {"dataset", "manifest_sha256"} <= set(spec)
            or set(spec) - {"dataset", "manifest_sha256", "check"}):
        raise ValueError("expert_overlay takes dataset, manifest_sha256 and optionally check")
    if spec.get("check", "sha256") not in CHECKS:
        raise ValueError(f"expert_overlay.check must be one of {CHECKS}")
    if not isinstance(spec["dataset"], str) or spec["dataset"].count("/") != 1:
        raise ValueError("expert_overlay.dataset must be <owner>/<name>")
    digest = spec["manifest_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("expert_overlay.manifest_sha256 must be a sha256 hex digest")
    return dict(spec)


def lab_override(environ=None):
    """``ARENA_EXPERT_OVERLAY``: absent -> no override (key missing); ``off`` -> None; else the JSON spec."""
    environ = os.environ if environ is None else environ
    value = environ.get("ARENA_EXPERT_OVERLAY")
    if not value:
        return {}
    return {"expert_overlay": None if value == "off" else validate(json.loads(value))}


def sha256_file(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_manifest(overlay_dir, expected_sha256):
    raw = (Path(overlay_dir) / MANIFEST).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError(f"{overlay_dir}/{MANIFEST}: sha256 differs from the pinned {expected_sha256}")
    manifest = json.loads(raw)
    if manifest.get("format") != FORMAT:
        raise RuntimeError(f"unknown expert overlay format {manifest.get('format')!r}")
    if not manifest.get("complete"):
        raise RuntimeError("the expert overlay is incomplete (not every layer was re-quantized)")
    return manifest


def materialise(base_dir, overlay_dir, dest, expected_sha256, *, threads=8, log=print, check="sha256",
                expert_prune_sha256=None):
    """Symlink farm at ``dest`` (base files, overlay shards over them); returns the record. An overlay that omits a
    pruning mask's experts (``omits_pruned``) is smaller than the base shards and is served only with that mask."""
    started = time.monotonic()
    base_dir, overlay_dir, dest = Path(base_dir), Path(overlay_dir), Path(dest)
    manifest = load_manifest(overlay_dir, expected_sha256)
    weight_map = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    shards = manifest["shards"]
    unknown = sorted(set(shards) - set(weight_map.values()))
    if unknown:
        raise RuntimeError(f"overlay shards not in the checkpoint index: {unknown[:5]}")
    omits = bool(manifest.get("omits_pruned"))
    if (omits or manifest.get("expert_prune_sha256")) and manifest.get("expert_prune_sha256") != expert_prune_sha256:
        raise RuntimeError("the expert overlay was calibrated for another pruning mask "
                           f"({manifest.get('expert_prune_sha256')}) than the server's ({expert_prune_sha256})")
    for name, spec in shards.items():
        size = (overlay_dir / name).stat().st_size
        if size != spec["bytes"] or (not omits and size != (base_dir / name).stat().st_size):
            raise RuntimeError(f"{name}: overlay {size} bytes, manifest {spec['bytes']}, base "
                               f"{(base_dir / name).stat().st_size}")

    def check_one(name):
        return name, sha256_file(overlay_dir / name)

    if check == "sha256":
        with ThreadPoolExecutor(threads) as pool:
            for name, digest in pool.map(check_one, sorted(shards)):
                if digest != shards[name]["sha256"]:
                    raise RuntimeError(f"{name}: overlay sha256 differs from its manifest")
    elif check != "size":
        raise ValueError(f"unknown overlay check {check!r}")
    if dest.exists():
        for item in dest.iterdir():
            item.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    for item in sorted(base_dir.iterdir()):
        source = overlay_dir / item.name if item.name in shards else item.resolve()
        (dest / item.name).symlink_to(source)
    record = {"format": FORMAT, "base_dir": str(base_dir), "overlay_dir": str(overlay_dir), "dest": str(dest),
              "manifest_sha256": expected_sha256, "shards": len(shards), "check": check, "omits_pruned": omits,
              "overlay_bytes": sum(s["bytes"] for s in shards.values()),
              "seconds": round(time.monotonic() - started, 1)}
    log("ARC_EXPERT_OVERLAY " + json.dumps(record, sort_keys=True))
    return record
