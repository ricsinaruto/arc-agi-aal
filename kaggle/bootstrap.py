"""Standard-library-only setup executed by the generated Kaggle notebook."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def input_files(root, filename):
    for directory, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".kaggle", ".git", ".cache"}]
        if filename in files:
            yield Path(directory) / filename


def find_wheelhouse(input_root):
    matches = []
    for marker in input_files(input_root, "manifest.json"):
        if (marker.parent / "runtime/requirements.lock").is_file() and (
            marker.parent / "analysis/requirements.lock"
        ).is_file():
            matches.append(marker.parent)
    if len(matches) != 1:
        raise RuntimeError(f"Attach exactly one arc-prolong wheelhouse dataset; found {matches}")
    return matches[0]


def find_model(input_root):
    matches = []
    for index in input_files(input_root, "model.safetensors.index.json"):
        config_path = index.parent / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        text_config = config.get("text_config", config)
        if (
            "Qwen3_5ForConditionalGeneration" in config.get("architectures", [])
            and text_config.get("hidden_size") == 5120
            and config.get("quantization_config", {}).get("quant_method") == "fp8"
        ):
            missing = [
                name
                for name in set(json.loads(index.read_text())["weight_map"].values())
                if not (index.parent / name).is_file()
            ]
            if missing:
                raise RuntimeError(f"Incomplete model dataset: missing {missing}")
            matches.append(index.parent)
    if len(matches) != 1:
        raise RuntimeError(f"Attach one official Qwen3.8-27B-FP8 model; found {matches}")
    return matches[0]


def verify_model(model, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    for name, expected in manifest["files"].items():
        with (model / name).open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        if digest != expected["sha256"]:
            raise RuntimeError(
                f"Model metadata differs from the pinned official checkpoint: {name}"
            )
    full = os.environ.get("ARC_VERIFY_MODEL_WEIGHTS") == "1"
    for name, expected in manifest["weights"].items():
        path = model / name
        if path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"Model shard size mismatch: {name}")
        if full:
            with path.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            if digest != expected["sha256"]:
                raise RuntimeError(f"Model shard checksum mismatch: {name}")
    return {
        "revision": manifest["revision"],
        "metadata_sha256": "verified",
        "weights": "sha256_verified" if full else "sizes_verified",
    }


def verify_solver(source):
    return {"profile": "current"}


def verify_runtime_fingerprints(wheels, fingerprints):
    if fingerprints is None or not Path(fingerprints).exists():
        return
    for name, expected in json.loads(Path(fingerprints).read_text()).items():
        with (wheels / name).open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        if digest != expected:
            raise RuntimeError(f"Runtime dataset differs from the packaged dependency pins: {name}")


def install_environment(dest, wheels):
    # Kaggle's system Python can create venvs but its ensurepip bootstrap fails.
    # pip --python supports an isolated target environment without installing pip there.
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(dest)], check=True, timeout=45
    )
    python = Path(dest) / "bin/python"
    pip = [sys.executable, "-m", "pip", "--isolated", "--python", str(python)]
    subprocess.run(
        [
            *pip,
            "install",
            "--no-index",
            "--no-compile",
            "--disable-pip-version-check",
            "--find-links",
            str(wheels),
            "-r",
            str(Path(wheels) / "requirements.lock"),
        ],
        check=True,
        timeout=300,
    )
    subprocess.run([*pip, "check"], check=True, timeout=30)
    return str(python)


def install_runtime(input_root, working, notebook_start, *, fingerprints=None):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("This tested wheelhouse requires Kaggle's Python 3.12 runtime")
    working = Path(working)
    wheels = find_wheelhouse(input_root)
    verify_runtime_fingerprints(wheels, fingerprints)
    manifest = json.loads((wheels / "manifest.json").read_text())
    # Verify the offline assets before executing any installed package.
    for name, expected in manifest.items():
        path = (wheels / name).resolve()
        if not path.is_relative_to(wheels.resolve()):
            raise ValueError("Invalid wheelhouse manifest path")
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
            raise RuntimeError(f"Wheel checksum mismatch: {name}")
    model = find_model(input_root)
    versions = {}
    for kind in ("runtime", "analysis"):
        dest = working / f"arc-prolong-{kind}"
        versions[kind] = install_environment(dest, wheels / kind)
    record = {
        "install_seconds": time.time() - notebook_start,
        "model_path": str(model),
        "wheelhouse_path": str(wheels),
        "interpreters": versions,
    }
    (working / "arc-prolong-install.json").write_text(json.dumps(record, indent=2))
    return record
