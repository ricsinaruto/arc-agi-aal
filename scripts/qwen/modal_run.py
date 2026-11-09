"""Prepared Modal launcher. Importing this file does not launch a benchmark.

Only invoke after benchmark authorization and budget review. Existing assets are
read-only; the run gets its own output directory and one GPU.
"""

import os
import shutil
import tempfile
from pathlib import Path

import modal

BUNDLE = (
    Path(os.environ.get("ARC_QWEN_BUNDLE", "dist/qwen-minimal")).resolve()
    if modal.is_local()
    else Path("/opt/qwen-minimal")
)
app = modal.App("arc-qwen-minimal", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libseccomp2",
        "binutils",
        "libglib2.0-0",
        "libgl1",
        "libnuma1",
        "libgomp1",
        "build-essential",
        "ninja-build",
    )
    .add_local_dir(str(BUNDLE), "/opt/qwen-minimal", copy=True)
    .add_local_file(__file__, "/root/modal_run.py", copy=True)
)


def collect_artifacts(working, output):
    """Persist evidence only; venv, Node and serving scratch stay on local disk."""
    working, output = Path(working), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (working / "trials").exists():
        shutil.copytree(working / "trials", output / "trials", dirs_exist_ok=True)
    for filename in ("setup.json",):
        if (working / filename).is_file():
            shutil.copyfile(working / filename, output / filename)
    if (working / "serving").exists():
        (output / "serving").mkdir(exist_ok=True)
        for source in (working / "serving").iterdir():
            if source.is_file() and source.suffix in {".json", ".log"}:
                shutil.copyfile(source, output / "serving" / source.name)


@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    cpu=(0.125, 8),
    memory=(128, 196608),
    timeout=int(os.environ.get("ARC_QWEN_MODAL_TIMEOUT", "900")),
    retries=0,
    block_network=True,
    max_containers=1,
    volumes={"/assets": assets, "/runs": runs},
)
def benchmark(run_id: str, gameplay_seconds: int = 60, max_seconds: int = 840):
    import subprocess
    import sys

    # Exercise the Kaggle-compatible path during the pre-Kaggle GPU smoke.
    os.environ["ARC_FORCE_CHROOT"] = "1"
    output = Path("/runs/qwen-minimal") / run_id
    output.mkdir(parents=True, exist_ok=False)
    root = Path(tempfile.mkdtemp(prefix="arc-qwen-smoke-")) / "working"
    if not Path("/assets/flash-input/READY.json").is_file():
        raise RuntimeError("The existing offline Flash assets have not finished preparation")
    # The pinned recipe uses the competition model's canonical Kaggle mount path.
    Path("/kaggle").mkdir(exist_ok=True)
    Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    os.environ["LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", "")
    try:
        subprocess.run(
            [
                sys.executable,
                "/opt/qwen-minimal/bootstrap.py",
                "--working",
                str(root),
                "--input-root",
                "/kaggle/input",
                "--games-dir",
                "/assets/games",
                "--gameplay-seconds",
                str(gameplay_seconds),
                "--max-seconds",
                str(max_seconds),
            ],
            check=True,
        )
    finally:
        collect_artifacts(root, output)
        runs.commit()
    return str(output)


@app.local_entrypoint()
def main(run_id: str, gameplay_seconds: int = 60, max_seconds: int = 840):
    call = benchmark.spawn(run_id, gameplay_seconds, max_seconds)
    print({"function_call_id": call.object_id, "run_id": run_id})
