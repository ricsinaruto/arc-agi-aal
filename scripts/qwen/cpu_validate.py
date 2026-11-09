"""Run actual Linux CPU invariants on Modal; no GPU or game assets are attached."""

import json
import shutil
import tempfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/app")
app = modal.App("arc-qwen-cpu-validation", include_source=False)


def cpu_image():
    # Snapshot only required files. Excluding bytecode explicitly avoids source
    # mount races with simultaneous local tests and never uploads credentials.
    context = Path(tempfile.mkdtemp(prefix="arc-qwen-cpu-build-"))
    shutil.copyfile(ROOT / "pyproject.toml", context / "pyproject.toml")
    shutil.copytree(
        ROOT / "src", context / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copytree(ROOT / "pi", context / "pi", ignore=shutil.ignore_patterns("node_modules"))
    (context / "tests").mkdir()
    for pattern in (
        "conftest.py",
        "test_qwen*.py",
        "test_prolong_qwen.py",
        "test_schema_qwen.py",
        "test_segmentation.py",
    ):
        for test in (ROOT / "tests").glob(pattern):
            shutil.copyfile(test, context / "tests" / test.name)
    shutil.copyfile(ROOT / "scripts/qwen/Dockerfile", context / "Dockerfile")
    (context / "scripts/qwen").mkdir(parents=True)
    for name in ("bootstrap.py", "modal_run.py"):
        shutil.copyfile(ROOT / "scripts/qwen" / name, context / "scripts/qwen" / name)
    return (
        modal.Image.from_dockerfile(str(context / "Dockerfile"), context_dir=str(context))
        .add_local_file(
            str(ROOT / "scripts/qwen/bootstrap.py"), "/app/scripts/qwen/bootstrap.py", copy=True
        )
        .add_local_file(__file__, "/root/cpu_validate.py", copy=True)
    )


image = cpu_image() if modal.is_local() else None


@app.function(
    image=image, timeout=300, cpu=(0.125, 2), memory=(128, 4096), retries=0, block_network=True
)
def validate(chroot: bool = False):
    import os
    import platform
    import subprocess
    import time

    if chroot:
        os.environ["ARC_FORCE_CHROOT"] = "1"
    start = time.monotonic()
    result = subprocess.run(
        ["python", "-m", "pytest", "tests", "-q"],
        cwd="/app",
        capture_output=True,
        text=True,
        timeout=240,
    )
    return {
        "forced_chroot": chroot,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "seconds": time.monotonic() - start,
        "kernel": platform.release(),
    }


@app.local_entrypoint()
def main(output: str = "runs/qwen-cpu-validation.json", chroot: bool = False):
    result = validate.remote(chroot)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["returncode"]:
        raise RuntimeError("CPU invariants failed; see report")
