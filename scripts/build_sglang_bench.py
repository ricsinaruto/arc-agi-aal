"""Assemble the SGLang benchmark bundle: the public Duck loop, the bench driver and a study.

Separate from ``build_duck_controls.py``: no vLLM serving code is copied. The bundle is
uploaded as ``richardcsaky/arc-sglang-bench`` and run by the generated notebook next to the
``arc-sglang-runtime`` dataset (the Pennyroyal image's ``/opt`` tree). Nothing is submitted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sglang_bench import launch, profile  # noqa: E402

FULL_PACKAGES = ("inference", "taaf", "sglang_bench")
ARC_INFRA_FILES = ("__init__.py", "arcade.py", "server.py", "metrics.py", "serving.py")
DUCK_CONTROL_FILES = ("__init__.py", "public_trial.py")
KAGGLE_FILES = (
    "bootstrap.py",
    "flash_bootstrap.py",
    "runtime-pins/manifest.json",
    "runtime-pins/analysis/requirements.lock",
    "runtime-pins/runtime/requirements.lock",
)
EXTRA_WHEELS = ROOT / "kaggle/runtime-pins/extra-wheels"
DATASET = "richardcsaky/arc-sglang-bench"
RUNTIME_DATASET = "richardcsaky/arc-sglang-runtime"
WHEELHOUSE_DATASET = "richardcsaky/arc-prolong-runtime"
KAGGLE_INPUTS = [DATASET, RUNTIME_DATASET, WHEELHOUSE_DATASET]
MODEL_SOURCE = "keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"
SWEEP_NOTEBOOK_SECONDS = 8400
SMOKE_NOTEBOOK_SECONDS = 6600  # push with --timeout 7200: the first boot took > 25 min


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def copy_tree(src, dst):
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def study_definition(name, *, smoke, points, game_seconds, context_window, online_fp8):
    if smoke:
        point = profile.smoke_point()
        point.update(seconds=120, probes=True)
        selected = [point]
        mode, seconds = "smoke", SMOKE_NOTEBOOK_SECONDS
    else:
        selected = profile.sweep_points(online_fp8=online_fp8)
        if points:
            selected = [p for p in selected if p["name"] in points]
            if len(selected) != len(points):
                raise ValueError(f"Unknown points {sorted(set(points) - {p['name'] for p in selected})}")
        mode, seconds = "sweep", SWEEP_NOTEBOOK_SECONDS
    return {
        "name": name,
        "mode": mode,
        "game_seconds": game_seconds,
        "grace_seconds": 90,
        "analyzer_context_window": context_window,
        "online_fp8": online_fp8,
        "notebook_seconds": seconds,
        "recipe": {
            "image": launch.RECIPE_IMAGE,
            "script": launch.RECIPE_SCRIPT,
            "dropped": launch.DROPPED_RECIPE_FLAGS,
            "dropped_env": launch.DROPPED_RECIPE_ENV,
            "host": "127.0.0.1",
            "cc": "host gcc (Kaggle: gcc 11) instead of gcc-15",
            "nvcc": "CUDA 13.3 subset shipped in arc-sglang-runtime",
        },
        "public_loop": "duck_controls/public_trial.py with the notebook's solver settings; analyzer window raised to "
        f"{context_window}; LOCAL_ANALYZER_MODEL_ID pennyroyal; provider vllm (keeps top_k and chat_template_kwargs)",
        "points": selected,
    }


def build(name, *, smoke=False, points=None, game_seconds=900, context_window=131_072, online_fp8=True, output=None, duck_transfer=False):
    output = Path(output or ROOT / "dist" / name)
    if output.exists():
        shutil.rmtree(output)
    (output / "src").mkdir(parents=True)
    packages = (*FULL_PACKAGES, "arc_infra", "duck_transfer") if duck_transfer else FULL_PACKAGES
    for package in packages:
        copy_tree(ROOT / "src" / package, output / "src" / package)
    partial = [("duck_controls", DUCK_CONTROL_FILES)]
    if not duck_transfer:
        partial.append(("arc_infra", ARC_INFRA_FILES))
    for package, files in partial:
        (output / "src" / package).mkdir()
        for name_ in files:
            shutil.copyfile(ROOT / "src" / package / name_, output / "src" / package / name_)
    (output / "configs").mkdir()
    shutil.copyfile(ROOT / "configs/duck_public_notebook.json", output / "configs/duck_public_notebook.json")
    for rel in KAGGLE_FILES:
        target = output / "kaggle" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "kaggle" / rel, target)
    (output / "wheels").mkdir()
    for wheel in sorted(EXTRA_WHEELS.glob("imageio-*.whl")):
        shutil.copyfile(wheel, output / "wheels" / wheel.name)
    study = study_definition(name, smoke=smoke, points=points, game_seconds=game_seconds,
                             context_window=context_window, online_fp8=online_fp8)
    if duck_transfer:
        if smoke or points or not online_fp8:
            raise ValueError("Duck-transfer uses its fixed three-game recipe; do not combine profile overrides")
        study.update(mode="duck-transfer", game_seconds=900, analyzer_context_window=65536,
                     notebook_seconds=2040, startup_timeout_seconds=600,
                     agent_config="configs/duck_transfer_sglang.yaml",
                     comparison_reference="docs/three-game-v3-20260917/kaggle-duck-v5c-r2-result.json",
                     public_loop="Current Duck-transfer; v5c-r2 settings plus corrected tokenizer accounting",
                     points=[{"name": "duck-transfer", "harness": "duck-transfer", "seconds": 900,
                              "games": list(profile.CORE_GAMES), "profile": profile.RECIPE_DEFAULT.to_dict()}])
        shutil.copyfile(ROOT / "configs/duck_transfer_sglang.yaml", output / "configs/duck_transfer_sglang.yaml")
    (output / "study.json").write_text(json.dumps(study, indent=2) + "\n")
    files = {}
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        files[str(path.relative_to(output))] = sha256(path.read_bytes())
    (output / "build-manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for rel in sorted([*files, "build-manifest.json"]):
            info = zipfile.ZipInfo(rel, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (output / rel).read_bytes())
    (output / "source.zip").write_bytes(buffer.getvalue())
    digest = sha256(buffer.getvalue())
    kaggle_files(output, digest)
    return output, digest


def notebook(study, digest, *, launch_overrides=None):
    import ast

    def cell(kind, source):
        result = {"cell_type": kind, "metadata": {}, "source": source.strip() + "\n"}
        if kind == "code":
            ast.parse(result["source"])
            result.update(execution_count=None, outputs=[])
        return result

    points = "; ".join(f"{p['name']}: {p['profile']['admitted']} admitted, games {', '.join(p['games'])}"
                       for p in study["points"])
    harness = "Duck-transfer (established three-game settings)" if study["mode"] == "duck-transfer" else "Public Duck notebook TAAF loop"
    cells = [
        cell("markdown", f"""# SGLang benchmark: {study["name"]}
{harness} at a {study["analyzer_context_window"] // 1024}K analyzer window against the
Pennyroyal SGLang build ({study["recipe"]["image"]}), one fresh server per point ({study["game_seconds"]} s per game).
Recipe `{study["recipe"]["script"]}` reproduced as an explicit argv; dropped: {", ".join(study["recipe"]["dropped"][:1])} and the
NIXL host-cache flags. Points: {points}. Public benchmark only; no competition submission."""),
        cell("code", f"""
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
    raise RuntimeError("Benchmarks cannot run as competition submissions")
assert sys.version_info[:2] == (3, 12), "Select the Python 3.12 notebook image"
roots = [Path("/kaggle/input/datasets/{DATASET}"), Path("/kaggle/input/{DATASET.split("/")[1]}")]
mount = next((p for p in roots if p.is_dir()), None)
assert mount is not None, "Attach the {DATASET} dataset"
payload = (mount / {study["name"] + ".zip.blob"!r}).read_bytes()
if hashlib.sha256(payload).hexdigest() != {digest!r}:
    raise RuntimeError("Dataset bundle hash mismatch")
SOURCE = Path("/tmp/arc-sglang/source")
OUTPUT = Path("/kaggle/working/sglang-bench")
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
sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "kaggle")]
from sglang_bench import agent_env, launch, runtime
from sglang_bench.profile import RECIPE_DEFAULT

study = json.loads((SOURCE / "study.json").read_text())
import threading
prefetch = {{}}
prefetcher = threading.Thread(target=lambda: prefetch.update(runtime.prefetch_model()), daemon=True)
prefetcher.start()  # warms the page cache with the checkpoint while the runtime extracts
mount_rt = runtime.locate()
rt_manifest = runtime.verify_manifest(mount_rt)
extraction = runtime.extract(mount_rt, rt_manifest)
print(json.dumps(extraction), flush=True)
extraction["cudart"] = runtime.link_cudart()
extraction["cuda_driver"] = runtime.link_cuda_driver()
env = launch.recipe_env(RECIPE_DEFAULT, cache_base=runtime.SCRATCH / "cache", cuda_home=runtime.CUDA_HOME,
                        repo_root=runtime.REPO_ROOT, scratch=runtime.SCRATCH / "tmp")
check = runtime.check(env)
check["model"] = runtime.verify_model()
check["extraction"] = extraction
prefetcher.join()
check["prefetch"] = prefetch
(OUTPUT / "runtime-check.json").write_text(json.dumps(check, indent=2, default=str))
print(json.dumps(check.get("runtime")), flush=True)
if check["problems"]:
    raise RuntimeError("Runtime check failed: " + "; ".join(check["problems"]))
install = agent_env.install(SOURCE, runtime.SCRATCH)
print(json.dumps(install, indent=2), flush=True)
"""),
        cell("code", """
command = [install["agent_python"], "-u", "-m", "sglang_bench.bench", "--source", str(SOURCE),
           "--study", str(SOURCE / "study.json"), "--output", str(OUTPUT),
           "--model-dir", str(runtime.MODEL_DIR), "--notebook-start-epoch", str(NOTEBOOK_START_EPOCH)]
env = {**os.environ, "PYTHONPATH": str(SOURCE / "src")}
process = None
log_path = OUTPUT / "bench.log"
try:
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        with log_path.open() as reader:
            while process.poll() is None:
                print(reader.read(), end="", flush=True)
                if time.time() > NOTEBOOK_START_EPOCH + study["notebook_seconds"] + 10:
                    raise TimeoutError("Notebook experiment deadline reached")
                time.sleep(5)
            print(reader.read(), end="", flush=True)
finally:
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    for identity in OUTPUT.glob("*/server-identity.json"):
        pid = json.loads(identity.read_text()).get("pid")
        if pid:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
"""),
        cell("code", """
from sglang_bench.report import render

report = json.loads((OUTPUT / "study.json").read_text())
print(render(report))
if report.get("status") != "complete" or len(report["points"]) != len(study["points"]):
    raise RuntimeError("Incomplete benchmark; inspect saved logs")
"""),
    ]
    for i, item in enumerate(cells):
        item["id"] = f"sglang-bench-{i:02d}"
    if launch_overrides:
        if set(launch_overrides) != {"notebook_seconds", "startup_timeout_seconds"}:
            raise ValueError("Frozen repeats may override only startup and notebook time limits")
        marker = 'study = json.loads((SOURCE / "study.json").read_text())'
        cells[1]["source"] = cells[1]["source"].replace(
            marker, marker + f"\nstudy.update({launch_overrides!r})\n"
            '(OUTPUT / "launch-study.json").write_text(json.dumps(study, indent=2))', 1)
        cells[2]["source"] = cells[2]["source"].replace(
            'str(SOURCE / "study.json")', 'str(OUTPUT / "launch-study.json")')
        cells[0]["source"] += (
            f"\nFrozen-source repeat: readiness allowance {launch_overrides['startup_timeout_seconds']} s; "
            f"notebook allowance {launch_overrides['notebook_seconds']} s. Gameplay unchanged.\n")
        for item in cells:
            if item['cell_type'] == 'code':
                ast.parse(item['source'])
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}}


def kaggle_files(output, digest, kernel_suffix=""):
    study = json.loads((output / "study.json").read_text())
    (output / "experiment.ipynb").write_text(json.dumps(notebook(study, digest), indent=1) + "\n")
    kernel = {
        "id": f"richardcsaky/arc-{study['name']}{kernel_suffix}",
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
    shutil.copyfile(output / "source.zip", upload / f"{study['name']}.zip.blob")
    (upload / "dataset-metadata.json").write_text(json.dumps({
        "id": DATASET,
        "title": "ARC SGLang benchmark source bundles",
        "licenses": [{"name": "other"}],
        "description": "Private benchmark bundles: the public Duck notebook's TAAF loop (MIT) driven against the "
        f"Pennyroyal SGLang build. {study['name']} sha256 {digest}; mode {study['mode']}.",
    }, indent=2) + "\n")
    return kernel


def prepare_repeat(source, name, *, notebook_seconds, startup_seconds, kernel_timeout, output=None):
    """Preserve a verified frozen ZIP; alter only the notebook's launch time allowances."""
    source = Path(source).resolve()
    output = Path(output or ROOT / "dist" / name).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not 0 < startup_seconds < notebook_seconds < kernel_timeout:
        raise ValueError("Require startup < notebook < kernel timeout")
    study = json.loads((source / 'study.json').read_text())
    # Match the frozen driver's startup + warmup + gameplay + cleanup admission.
    from sglang_bench.bench import DRAIN_ALLOWANCE
    needed = startup_seconds + 60 + max(p['seconds'] for p in study['points']) + 90 + DRAIN_ALLOWANCE
    if notebook_seconds < needed + 600:
        raise ValueError("Notebook envelope must also reserve 600 s for extraction/prefetch")
    manifest = json.loads((source / 'build-manifest.json').read_text())
    with zipfile.ZipFile(source / 'source.zip') as archive:
        if json.loads(archive.read('build-manifest.json')) != manifest:
            raise ValueError('ZIP manifest differs from frozen bundle')
        for rel, expected in manifest['files'].items():
            if sha256((source / rel).read_bytes()) != expected or sha256(archive.read(rel)) != expected:
                raise ValueError(f'Frozen source mismatch: {rel}')
    shutil.copytree(source, output)
    digest = sha256((output / 'source.zip').read_bytes())
    overrides = {'notebook_seconds': notebook_seconds, 'startup_timeout_seconds': startup_seconds}
    (output / 'experiment.ipynb').write_text(json.dumps(notebook(study, digest, launch_overrides=overrides), indent=1)+'\n')
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    metadata.update(id=f'richardcsaky/arc-{name}', title=f'ARC {name}')
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (output / 'repeat-manifest.json').write_text(json.dumps({
        'frozen_bundle': str(source), 'source_zip_sha256': digest,
        'source_files_verified': len(manifest['files']), 'launch_overrides': overrides,
        'kernel_timeout_seconds': kernel_timeout,
    }, indent=2)+'\n')
    return output, digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="sglang-bench-3g-128k")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--duck-transfer", action="store_true")
    parser.add_argument("--point", action="append", choices=["c3", "c4", "c6"])
    parser.add_argument("--game-seconds", type=int, default=900)
    parser.add_argument("--context-window", type=int, default=131_072)
    parser.add_argument("--no-online-fp8", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--repeat-from", help="Verified frozen bundle; preserve its ZIP and gameplay settings")
    parser.add_argument("--notebook-seconds", type=int, default=5340)
    parser.add_argument("--startup-seconds", type=int, default=2700)
    parser.add_argument("--kernel-timeout", type=int, default=5400)
    args = parser.parse_args(argv)
    if args.repeat_from:
        if args.smoke or args.point or args.duck_transfer or args.no_online_fp8 or args.game_seconds != 900 or args.context_window != 131_072:
            parser.error("Frozen repeat cannot change experiment settings")
        output, digest = prepare_repeat(args.repeat_from, args.name, output=args.output,
                                       notebook_seconds=args.notebook_seconds, startup_seconds=args.startup_seconds,
                                       kernel_timeout=args.kernel_timeout)
    else:
        output, digest = build(args.name, smoke=args.smoke, points=args.point, game_seconds=args.game_seconds,
                               context_window=args.context_window, online_fp8=not args.no_online_fp8, output=args.output, duck_transfer=args.duck_transfer)
    print(json.dumps({"output": str(output), "sha256": digest}))


if __name__ == "__main__":
    main()
