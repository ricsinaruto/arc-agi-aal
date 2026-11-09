"""Install only CPU agent dependencies, then start the pinned offline Flash-Next server."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from bootstrap import find_wheelhouse, install_environment

from arc_infra.serving import load_recipe

AGENT_REQUIREMENTS = [
    "arc-agi==0.9.9",
    "httpx==0.28.1",
    "numpy==2.3.5",
    "pillow==12.3.0",
    "pyyaml==6.0.3",
    "scipy==1.18.1",
    "requests==2.34.2",
]


def setup_command(source, output, server=None, working="server", python=None):
    """Start the pinned Duck-profile server: MTP 3, eight generation slots.

    ``server`` (from ``study.json``) may raise the context window and KV cache size, turn
    prefix caching on, or (``gpu_memory_utilization`` with ``kv_cache_gib`` 0) run the ARC
    fork with a vLLM-profiled KV cache. ``working`` is the server's directory under ``output``
    (Duck v5b runs one server per trial, sequentially). ``python`` (v5c) runs the setup under
    that interpreter instead of Kaggle's: under ``/usr/bin/python3`` the fork's profiling left
    6.09 GiB of KV at 64K and at 262K alike, while the same recipe under the R/B/C agent venv
    (no system site-packages) left 8.46 GiB.
    """
    command = [
        python or sys.executable, "-m", "arc_infra.setup", "--output", str(Path(output) / working),
        "--mtp-tokens", "3",
    ]
    for key, flag in (("max_model_len", "--max-model-len"), ("kv_cache_gib", "--kv-cache-gib")):
        if server and key in server:
            command += [flag, str(int(server[key]))]
    if server and server.get("prefix_caching"):
        command.append("--prefix-caching")
    if server and server.get("gpu_memory_utilization"):
        command += ["--gpu-memory-utilization", str(server["gpu_memory_utilization"])]
    return command


def start_server(source, output, server=None, working="server", python=None):
    """Run the setup child with Kaggle's Python (as in the validated recipe) or, when the
    server spec says ``"python": "agent"``, with the agent venv (``python``); it returns once
    the detached server is ready. The child configures its own environment only."""
    env = {**os.environ, "PYTHONPATH": str(Path(source) / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    if not (server and server.get("python") == "agent"):
        python = None
    elif python is None:
        raise ValueError("The server spec asks for the agent venv; pass its interpreter")
    command = setup_command(source, output, server, working, python=python)
    subprocess.run(command, env=env, check=True, timeout=1800)
    return command


def prepare(source, output, notebook_start, *, server=None, working="server"):
    """Install the analysis and agent environments, then start the first server.

    ``server`` defaults to ``study.json``'s ``server`` entry (single-server studies); Duck
    v5b passes the first trial's own spec and working directory.
    """
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Pinned runtime requires Python 3.12")
    source, output = Path(source), Path(output)
    work = Path("/tmp/arc-flash-next")
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    wheels = find_wheelhouse("/kaggle/input")
    pins = source / "kaggle/runtime-pins"
    for name in ("manifest.json", "analysis/requirements.lock", "runtime/requirements.lock"):
        if (wheels / name).read_bytes() != (pins / name).read_bytes():
            raise RuntimeError(f"Attached CPU wheel metadata changed: {name}")
    # Analysis has its own small, engine-free environment. The evaluator never
    # imports from the large, specialized vLLM runtime's site-packages.
    manifest = json.loads((pins / "manifest.json").read_text())
    for name, value in manifest.items():
        if name.startswith("analysis/"):
            with (wheels / name).open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            if digest != value["sha256"]:
                raise RuntimeError(f"Analysis wheel mismatch: {name}")
    analysis = install_environment(work / "analysis", wheels / "analysis")
    agent = work / "agent"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(agent)], check=True, timeout=45
    )
    python = str(agent / "bin/python")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "--python",
            python,
            "install",
            "--no-index",
            "--no-compile",
            "--disable-pip-version-check",
            "--find-links",
            str(wheels / "runtime"),
            *AGENT_REQUIREMENTS,
        ],
        check=True,
        timeout=180,
    )
    if server is None:
        study = json.loads((source / "study.json").read_text())
        server = study.get("server")
    start_server(source, output, server, working, python=python)
    record = {
        "setup_seconds": time.time() - notebook_start,
        "analysis_python": analysis,
        "agent_python": python,
        "cpu_requirements": AGENT_REQUIREMENTS,
        "source_sha256": hashlib.sha256((source / "build-manifest.json").read_bytes()).hexdigest(),
    }
    (output / "install.json").write_text(json.dumps(record, indent=2))
    return record


def stop_server(output, working="server"):
    recipe = load_recipe(Path(output) / working)
    result = recipe.stop_owned_server()
    (Path(output) / working / "cleanup.json").write_text(json.dumps(result, indent=2))
    if not result.get("stopped"):
        raise RuntimeError("Owned Flash-Next server did not stop")
