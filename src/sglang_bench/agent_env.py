"""The public loop's CPU venv from the attached wheelhouse, exactly as ``flash_bootstrap``
installs it (minus the server). Standard library only; runs under Kaggle's Python."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

PIN_FILES = ("manifest.json", "analysis/requirements.lock", "runtime/requirements.lock")


def find_wheelhouse(input_root="/kaggle/input") -> Path:
    from bootstrap import find_wheelhouse as _find  # the bundled kaggle/bootstrap.py

    return Path(_find(input_root))


def agent_requirements(source: Path) -> list[str]:
    """The list from the bundled ``kaggle/flash_bootstrap.py`` (import-free read)."""
    import ast

    tree = ast.parse((Path(source) / "kaggle/flash_bootstrap.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "AGENT_REQUIREMENTS" for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise RuntimeError("AGENT_REQUIREMENTS not found in kaggle/flash_bootstrap.py")


def install(source, work, *, input_root="/kaggle/input", runner=subprocess.run) -> dict:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Pinned runtime requires Python 3.12")
    source, work = Path(source), Path(work)
    started = time.time()
    work.mkdir(parents=True, exist_ok=True)
    wheels = find_wheelhouse(input_root)
    pins = source / "kaggle/runtime-pins"
    for name in PIN_FILES:
        if (wheels / name).read_bytes() != (pins / name).read_bytes():
            raise RuntimeError(f"Attached CPU wheel metadata changed: {name}")
    agent = work / "agent"
    runner([sys.executable, "-m", "venv", "--without-pip", str(agent)], check=True, timeout=45)
    python = str(agent / "bin/python")
    pip = [sys.executable, "-m", "pip", "--isolated", "--python", python, "install", "--no-index",
           "--no-compile", "--disable-pip-version-check"]
    requirements = agent_requirements(source)
    runner([*pip, "--find-links", str(wheels / "runtime"), *requirements], check=True, timeout=180)
    runner([*pip, "--find-links", str(source / "wheels"), "imageio==2.37.0"], check=True, timeout=120)
    record = {
        "setup_seconds": round(time.time() - started, 1),
        "agent_python": python,
        "cpu_requirements": requirements,
        "wheelhouse": str(wheels),
        "source_sha256": hashlib.sha256((source / "build-manifest.json").read_bytes()).hexdigest(),
    }
    (work / "install.json").write_text(json.dumps(record, indent=2))
    return record
