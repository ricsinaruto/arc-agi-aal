"""Bundle helpers, copied into every arena bundle as ``bootstrap.py`` (``bootstrap_arena.py`` imports them):
verified manifest, locked wheel install without the host's pip, evidence collection.

Formerly ``scripts/qwen/bootstrap.py`` (whose own three-trial Qwen comparison entry point was retired on 4 October).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PIP_WHEEL = "pip-25.2-py3-none-any.whl"
PIP_SHA256 = "6d67a2b4e7f14d8b31b8b52648866fa717f45a1eb70e83002f4331d07e953717"


def install_agent_environment(bundle, venv, env):
    """Install locked wheels without depending on the host's ensurepip package."""
    wheel = bundle / "wheels" / PIP_WHEEL
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != PIP_SHA256:
        raise ValueError("Pinned bootstrap pip wheel mismatch")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    python = str(venv / "bin/python")
    # Run pip directly from its verified wheel in the target interpreter. Do not
    # inherit host site-packages or depend on host pip/ensurepip availability.
    installer_env = {**env, "PYTHONPATH": str(wheel), "PIP_CONFIG_FILE": os.devnull}
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--require-hashes",
            "--find-links",
            str(bundle / "wheels"),
            "-r",
            str(bundle / "requirements.txt"),
        ],
        env=installer_env,
        check=True,
    )
    subprocess.run([python, "-m", "pip", "check"], env=installer_env, check=True)
    return python


def collect_evidence(working, output):
    """Copy trial outputs, serving diagnostics and setup report to a persisted directory.

    Installations and serving scratch stay behind so notebook
    outputs hold evidence only; Kaggle otherwise lists thousands of runtime files.
    """
    working, output = Path(working), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (working / "trials").is_dir():
        shutil.copytree(working / "trials", output / "trials", dirs_exist_ok=True)
    if (working / "setup.json").is_file():
        shutil.copyfile(working / "setup.json", output / "setup.json")
    if (working / "serving").is_dir():
        (output / "serving").mkdir(exist_ok=True)
        for source in (working / "serving").iterdir():
            if source.is_file() and source.suffix in {".json", ".log"}:
                shutil.copyfile(source, output / "serving" / source.name)
        for name in ("cache-probe",):
            if (working / "serving" / name).is_dir():
                shutil.copytree(working / "serving" / name, output / "serving" / name, dirs_exist_ok=True)


def verify(bundle):
    identity = json.loads((bundle / "MANIFEST.json").read_text())
    if not identity["dependencies_prepared"]:
        raise RuntimeError("Bundle lacks offline dependencies; rebuild with --dependencies")
    for name, expected in identity["files"].items():
        path = bundle / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Offline source/dependency mismatch: {name}")
    return identity
