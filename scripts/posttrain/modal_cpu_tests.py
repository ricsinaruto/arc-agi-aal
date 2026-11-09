"""Heal unit tests on Modal CPU (no local torch): ``modal run scripts/posttrain/modal_cpu_tests.py --tests a.py,b.py``
(test file names under tests/; scripts/posttrain and src are importable as in the repo)."""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE
app = modal.App("sft-heal-cpu-tests")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("rclone")
         .pip_install("torch==2.14.0", "transformers==5.18.0", "safetensors", "pillow", "torchvision==0.29.0",
                      "accelerate", "numpy", "pytest", "modal", "httpx", "pyyaml", "ruff", extra_index_url="https://download.pytorch.org/whl/cpu"))
if modal.is_local():
    image = (image.add_local_dir(ROOT / "scripts" / "posttrain", "/opt/t/scripts/posttrain", copy=True,
                                 ignore=["*.pt", "*.safetensors", "__pycache__"])
             .add_local_dir(ROOT / "tests", "/opt/t/tests", copy=True, ignore=["__pycache__"])
             .add_local_dir(ROOT / "src", "/opt/t/src", copy=True, ignore=["__pycache__"]))


@app.function(image=image, timeout=3600)
def run(tests: str):
    import subprocess
    files = [f"/opt/t/tests/{t}" for t in tests.split(",") if t]
    r = subprocess.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", *files], capture_output=True,
                       text=True, cwd="/opt/t", env={"PYTHONPATH": "/opt/t/src:/opt/t/scripts/posttrain", "PATH": "/usr/local/bin:/usr/bin:/bin"})
    print(r.stdout[-6000:], r.stderr[-2000:])
    return r.returncode


@app.local_entrypoint()
def main(tests: str):
    code = run.remote(tests)
    print("TESTS", "OK" if code == 0 else f"FAILED ({code})")
