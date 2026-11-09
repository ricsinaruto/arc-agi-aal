"""Assemble the Duck controls bundle from the working tree: duck-transfer and the public loop.

The bundle carries the shared infrastructure, the public Duck packages, the duck-transfer
agent, the control driver, the pinned dependency records and the study definition. A
study either selects games (one session each) or plays a 25/110-session fleet. Nothing is
submitted here.
"""

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("arc_infra", "inference", "taaf", "duck_transfer", "duck_controls")
KAGGLE_FILES = (
    "bootstrap.py",
    "flash_bootstrap.py",
    "runtime-pins/manifest.json",
    "runtime-pins/analysis/requirements.lock",
    "runtime-pins/runtime/requirements.lock",
)
EXTRA_WHEELS = ROOT / "kaggle/runtime-pins/extra-wheels"
DEFAULT_GAMES = ("ft09-0d8bbf25", "lp85-305b61c3", "ar25-0c556536")
DATASET = "richardcsaky/arc-duck-controls"
KAGGLE_INPUTS = [
    DATASET,
    "richardcsaky/arc-prolong-runtime",
    "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
    "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1",
]
MODEL_SOURCE = "keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def build(
    name,
    games=None,
    sessions=None,
    gameplay_seconds=600,
    output=None,
    *,
    max_model_len=65536,
    kv_cache_gib=5,
    reasoning_effort="xhigh",
    prefix_caching=True,
    max_output_tokens=8192,
    gpu_memory_utilization="0.96",
    fork_max_model_len=262144,
):
    """``gpu_memory_utilization`` (v5b) runs the corrected trial on the ARC fork with a
    vLLM-profiled KV cache at that utilization; ``None`` keeps the explicit-KV pinned server
    for both trials. The public control always gets the pinned server (``kv_cache_gib``).
    ``fork_max_model_len`` (v5c) is the fork server's context (262,144 gives the full
    profiled pool; 65,536 lost 2.4 GiB to profiling); the agent's own limit stays
    ``max_model_len`` client-side."""
    if gameplay_seconds not in (600, 900, 1200, 3600):
        raise ValueError("Use a gameplay duration the fleet runner accepts")
    if bool(games) == bool(sessions):
        raise ValueError("Select games or a session count, not both")
    if sessions is not None and sessions not in (25, 110):
        raise ValueError("Fleet studies use 25 or 110 sessions")
    if max_model_len not in (32768, 65536) or kv_cache_gib < 0:
        raise ValueError("Server context must be 32768 or 65536 with a non-negative KV size")
    if sessions is not None and (max_model_len, kv_cache_gib) != (32768, 5):
        raise ValueError("Fleet studies keep the pinned 32K context and 5 GiB KV cache")
    if gpu_memory_utilization is not None and not 0.5 <= float(gpu_memory_utilization) <= 0.99:
        raise ValueError("gpu_memory_utilization must be a fraction between 0.5 and 0.99")
    if fork_max_model_len not in (65536, 262144) or fork_max_model_len < max_model_len:
        raise ValueError("The fork server context must be 65536 or 262144 and cover the agent's limit")
    output = Path(output or ROOT / "dist" / name)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for package in PACKAGES:
        shutil.copytree(ROOT / "src" / package, output / "src" / package, ignore=ignore)
    for name_ in KAGGLE_FILES:
        target = output / "kaggle" / name_
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "kaggle" / name_, target)
    (output / "configs").mkdir()
    config = yaml.safe_load((ROOT / "configs/duck_controls.yaml").read_text())
    config["gameplay_seconds"] = gameplay_seconds
    config["max_model_len"] = max_model_len
    config["reasoning_effort"] = reasoning_effort
    config["max_output_tokens"] = max_output_tokens
    (output / "configs/experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copyfile(
        ROOT / "configs/duck_public_notebook.json", output / "configs/duck_public_notebook.json"
    )
    wheels = output / "wheels"
    wheels.mkdir()
    for wheel in sorted(EXTRA_WHEELS.glob("*.whl")):
        shutil.copyfile(wheel, wheels / wheel.name)
    passes = 1 if games else sessions // 25
    public_server = {
        "max_model_len": max_model_len,
        "kv_cache_gib": kv_cache_gib,
        "prefix_caching": prefix_caching,
    }
    if gpu_memory_utilization is None:
        corrected_server = dict(public_server)
        corrected_text = f"the same pinned server ({kv_cache_gib} GiB KV)"
    else:
        corrected_server = {
            "max_model_len": fork_max_model_len,
            "kv_cache_gib": 0,
            "prefix_caching": prefix_caching,
            "gpu_memory_utilization": str(gpu_memory_utilization),
            # v5c: the fork's setup runs under the agent venv, as the R/B/C server does; under
            # Kaggle's python its profiling left 6.09 GiB (and refused 262K), the venv 8.46 GiB.
            "python": "agent",
        }
        corrected_text = (
            f"the ARC fork of the pinned script (vLLM-profiled KV cache at utilization "
            f"{gpu_memory_utilization}, {fork_max_model_len // 1024}K server context, "
            f"{max_model_len // 1024}K agent limit)"
        )
    study = {
        "name": name,
        "gameplay_seconds": gameplay_seconds,
        # v5b: one server per trial; the second start costs about seven minutes.
        "notebook_seconds": 1800 + 600 + 2 * (gameplay_seconds * passes + 200),
        "trials": [
            {
                "name": "duck-transfer",
                "kind": "corrected",
                "agent_profile": "duck-transfer",
                "source": "src/duck_transfer over the public src/inference package; observation-preserving trimming",
                "server_dir": "server-duck-transfer",
            },
            {
                "name": "duck-public-notebook",
                "kind": "public",
                "source": "wuliao0 Duck Qwen3.8 Anim Base Version 1 loop and solver (src/inference, src/taaf); per-game runtime set to the gameplay seconds",
                "server_dir": "server-duck-public-notebook",
            },
        ],
        "policy": (
            f"One server per trial ({max_model_len // 1024}K context for the agent and the pinned "
            f"control server, prefix caching "
            f"{'on' if prefix_caching else 'off'}), started fresh and stopped before the next: "
            f"duck-transfer on {corrected_text}; the public control on the unmodified pinned "
            f"script ({kv_cache_gib} GiB KV); the public loop keeps a 32K analyzer window and "
            f"uncapped output but samples at the config temperature ({config['temperature']}; its "
            "notebook default is 0.6); fresh environments and scorecards per trial"
        ),
        "servers": {
            "duck-transfer": corrected_server,
            "duck-public-notebook": public_server,
        },
        "corrected": {
            "python_state": "persistent",
            "context_window": max_model_len,
            "server_context": corrected_server["max_model_len"],
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "server": corrected_text,
            "token_estimate": "server tokenizer precheck before each request; image tokens from the PNG geometry, calibrated characters-per-token and count residual, every sample logged",
            "prompt_layout": "per-turn boilerplate in the system prompt; older turns stubbed; one image per request",
            "feedback": "transition receipts, no-change repeat refusal, stuck notices",
            "game_over": "world/goal/action models retained; automatic level reset explained; RESET refused as a level's first action",
            "plumbing": "in-memory runtime state, delta history to the persistent interpreter, memoised segmentation, one prompt-log write per turn",
        },
    }
    if games:
        study["games"] = list(games)
    else:
        study["sessions"] = sessions
    (output / "study.json").write_text(json.dumps(study, indent=2))
    files = {
        p.relative_to(output).as_posix(): sha256(p.read_bytes())
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "build-manifest.json"
    }
    # File hashes let the notebook detect a stale or partial dataset upload.
    (output / "build-manifest.json").write_text(
        json.dumps({"study": study, "files": files}, indent=2)
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in sorted([*files, "build-manifest.json"]):
            info = zipfile.ZipInfo(rel, date_time=(2026, 9, 16, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (output / rel).read_bytes())
    bundle = buffer.getvalue()
    (output / "source.zip").write_bytes(bundle)
    return output, sha256(bundle)


def notebook(study, digest):
    """A Kaggle notebook that runs the bundle from the ``arc-duck-controls`` dataset."""
    import ast

    def cell(kind, source):
        result = {"cell_type": kind, "metadata": {}, "source": source.strip() + "\n"}
        if kind == "code":
            ast.parse(result["source"])
            result.update(execution_count=None, outputs=[])
        return result

    scope = (
        ", ".join(study["games"]) if "games" in study else f"{study['sessions']} public sessions"
    )
    cells = [
        cell(
            "markdown",
            f"""# Duck controls: {study["name"]}
Two sequential {study["gameplay_seconds"]}-second trials on {scope} (concurrent within a trial), each on its own
Duck-profile server (MTP 3, eight slots; fork at {study["servers"]["duck-transfer"]["max_model_len"] // 1024}K, agent limit {study["corrected"]["context_window"] // 1024}K) started fresh and stopped before the next:
duck-transfer on {study["corrected"]["server"]} (persistent Python, reasoning_effort
{study["corrected"]["reasoning_effort"]}, max_tokens {study["corrected"].get("max_output_tokens") or "uncapped"}, v5 prompt layout, transition receipts, GAME_OVER knowledge retained, tokenizer precheck), then the public notebook's own TAAF loop with its solver settings at a 32K analyzer window on the pinned script ({study["servers"]["duck-public-notebook"]["kv_cache_gib"]} GiB KV, prefix caching {"on" if study["servers"]["duck-public-notebook"].get("prefix_caching") else "off"}).
Public benchmark only; no competition submission.""",
        ),
        cell(
            "code",
            f"""
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

NOTEBOOK_START_EPOCH = time.time()
if os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").lower() in {{"1", "true"}}:
    raise RuntimeError("Public controls cannot run as competition submissions")
assert sys.version_info[:2] == (3, 12), "Select the Python 3.12 notebook image"
roots = [Path("/kaggle/input/datasets/{DATASET}"), Path("/kaggle/input/{DATASET.split("/")[1]}")]
mount = next((p for p in roots if p.is_dir()), None)
assert mount is not None, "Attach the {DATASET} dataset"
# Kaggle auto-extracts uploaded .zip files; the .blob suffix keeps the archive intact.
payload = (mount / {study["name"] + ".zip.blob"!r}).read_bytes()
if hashlib.sha256(payload).hexdigest() != {digest!r}:
    raise RuntimeError("Dataset bundle hash mismatch")
SOURCE = Path("/tmp/arc-flash-next/source")
OUTPUT = Path("/kaggle/working/duck-controls")
OUTPUT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        if not (SOURCE / name).resolve().is_relative_to(SOURCE.resolve()):
            raise ValueError("Invalid source archive path")
    archive.extractall(SOURCE)
manifest = json.loads((SOURCE / "build-manifest.json").read_text())
for name, expected in manifest["files"].items():
    if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Source mismatch: {{name}}")
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", ONLY_RESET_LEVELS="true",
                  MPLBACKEND="Agg", PIP_NO_INDEX="1", PYTHONDONTWRITEBYTECODE="1")
os.environ["LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", "")
sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "kaggle")]
from flash_bootstrap import prepare, start_server, stop_server
study = json.loads((SOURCE / "study.json").read_text())
FIRST = study["trials"][0]
try:
    install = prepare(SOURCE, OUTPUT, NOTEBOOK_START_EPOCH,
                      server=study["servers"][FIRST["name"]], working=FIRST["server_dir"])
    subprocess.run([sys.executable, "-m", "pip", "--isolated", "--python", install["agent_python"],
                    "install", "--no-index", "--no-compile", "--disable-pip-version-check",
                    "--find-links", str(SOURCE / "wheels"), "imageio==2.37.0"], check=True, timeout=120)
except BaseException:
    stop_server(OUTPUT, FIRST["server_dir"])
    raise
print(json.dumps(install, indent=2), flush=True)
""",
        ),
        cell(
            "code",
            """
# One server per trial: the first is already up; each later one starts after the previous stopped.
env = {**os.environ, "PYTHONPATH": str(SOURCE / "src")}
failures = []
for index, trial in enumerate(study["trials"]):
    working = trial["server_dir"]
    if index:
        start_server(SOURCE, OUTPUT, study["servers"][trial["name"]], working, python=install["agent_python"])
    command = [install["agent_python"], "-u", "-m", "duck_controls.run", "--source", str(SOURCE),
               "--study", str(SOURCE / "study.json"), "--output", str(OUTPUT),
               "--analysis-python", install["analysis_python"],
               "--notebook-start-epoch", str(NOTEBOOK_START_EPOCH),
               "--trial", trial["name"], "--server-dir", str(OUTPUT / working)]
    process = None
    log_path = OUTPUT / f"benchmark-{trial['name']}.log"
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(command, env=env, start_new_session=True,
                                       stdout=log, stderr=subprocess.STDOUT)
            with log_path.open() as reader:
                while process.poll() is None:
                    print(reader.read(), end="", flush=True)
                    if time.time() > NOTEBOOK_START_EPOCH + study["notebook_seconds"] + 10:
                        raise TimeoutError("Notebook experiment deadline reached")
                    time.sleep(5)
                print(reader.read(), end="", flush=True)
            if process.returncode:
                failures.append(f"{trial['name']}: experiment process exited {process.returncode}")
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        stop_server(OUTPUT, working)
if failures:
    raise RuntimeError("; ".join(failures))
""",
        ),
        cell(
            "code",
            """
report = json.loads((OUTPUT / "study.json").read_text())
for trial in report["trials"]:
    print(json.dumps({"trial": trial["trial"]["name"], "status": trial["status"], "summary": trial.get("summary")})[:4000])
if len(report["trials"]) != len(study["trials"]) or any(t["status"] != "complete" for t in report["trials"]):
    raise RuntimeError("Incomplete controls; inspect saved logs")
""",
        ),
    ]
    for i, item in enumerate(cells):
        item["id"] = f"duck-controls-{i:02d}"
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
    }


def kaggle_files(output, digest, kernel_suffix=""):
    """Write the notebook, kernel metadata and dataset upload folder next to the bundle."""
    study = json.loads((output / "study.json").read_text())
    (output / "experiment.ipynb").write_text(json.dumps(notebook(study, digest), indent=1) + "\n")
    minutes = study["gameplay_seconds"] // 60
    kernel = {
        "id": f"richardcsaky/arc-{study['name']}{kernel_suffix}",
        # Kaggle derives the slug from the title; keep it equal to the id's slug.
        "title": f"ARC {study['name']}{kernel_suffix}",
        "code_file": "experiment.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": KAGGLE_INPUTS,
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": [MODEL_SOURCE],
        "kernel_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(kernel, indent=2) + "\n")
    upload = output / "dataset-upload"
    upload.mkdir(exist_ok=True)
    # Kaggle datasets extract uploaded .zip files in place; a .blob suffix is served verbatim.
    shutil.copyfile(output / "source.zip", upload / f"{study['name']}.zip.blob")
    (upload / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": DATASET,
                "title": "ARC Duck controls source bundles",
                "licenses": [{"name": "other"}],
                "description": "Private benchmark bundles: duck-transfer over the public Duck package (MIT), "
                "the public notebook's TAAF loop and the control driver. "
                f"{study['name']} sha256 {digest}; {minutes}-minute gameplay.",
            },
            indent=2,
        )
        + "\n"
    )
    return kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="duck-controls-3game-900s-v5b")
    parser.add_argument("--kernel-suffix", default="", help="Kaggle kernel id suffix, e.g. -r2")
    parser.add_argument("--game", action="append", help="Selected public game (repeatable)")
    parser.add_argument("--sessions", type=int, choices=(25, 110), help="Full public fleet instead")
    parser.add_argument("--gameplay-seconds", type=int, default=900)
    parser.add_argument("--max-model-len", type=int, choices=(32768, 65536), default=65536)
    parser.add_argument("--kv-cache-gib", type=int, default=5, help="Explicit KV cache of the pinned (public control) server")
    parser.add_argument(
        "--gpu-memory-utilization", default="0.96",
        help="Corrected trial: ARC fork with a profiled KV cache at this utilization; 'pinned' keeps the explicit-KV server",
    )
    parser.add_argument("--reasoning-effort", choices=("xhigh", "medium", "low"), default="xhigh")
    parser.add_argument("--no-prefix-caching", action="store_true", help="Keep the pinned prefix-caching-off server")
    parser.add_argument("--max-output-tokens", type=int, default=8192, help="0 keeps the native uncapped output")
    parser.add_argument(
        "--fork-max-model-len", type=int, choices=(65536, 262144), default=262144,
        help="Fork server context (v5c: 262144 for the full profiled pool; the agent limit stays --max-model-len)",
    )
    args = parser.parse_args()
    games = None if args.sessions else (args.game or DEFAULT_GAMES)
    output, digest = build(
        args.name,
        games,
        args.sessions,
        args.gameplay_seconds,
        max_model_len=args.max_model_len,
        kv_cache_gib=args.kv_cache_gib,
        reasoning_effort=args.reasoning_effort,
        prefix_caching=not args.no_prefix_caching,
        fork_max_model_len=args.fork_max_model_len,
        max_output_tokens=args.max_output_tokens,
        gpu_memory_utilization=None if args.gpu_memory_utilization == "pinned" else args.gpu_memory_utilization,
    )
    kaggle_files(output, digest, args.kernel_suffix)
    print(json.dumps({"output": str(output), "source_bundle_sha256": digest}))


if __name__ == "__main__":
    main()
