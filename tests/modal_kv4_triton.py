"""The 4-bit KV kernel tests (E1, 26 Oct) under the Triton interpreter, on a Modal CPU job. The CI image has CPU torch
and no Triton, so the kernel tests skip there; this job adds Triton and the pinned vLLM files the checks read
(ARC_PINNED_SITE: int4_per_token_head.py, the QSA qsa.py).

  python tests/modal_kv4_triton.py [--tree PATH] [--pinned DIR] [-- PYTEST ARGS]

--tree: the checkout to test (default this one; its tracked files as they are on disk). --pinned: the pristine pinned
vLLM tree holding vllm/ (default ~/arc-shared/pinned-vllm). Prints pytest's output; exit status is pytest's. The job
ends itself: pytest's own timeout per test, and a 45-minute function ceiling.
"""
import argparse
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import modal

TORCH = "2.14.0"  # the CI image's CPU torch (runops.ciimage on main)
TRITON = "3.8.0"  # the version the kv4 CPU checks were validated on (3.6.0: the interpreter fails the split-K loop)
TESTS = ["tests/test_kv4_kernels.py", "tests/test_kv4_b300.py", "tests/test_dry_boot.py"]
PINNED = ("vllm/v1/attention/ops/int4_per_token_head.py", "vllm/models/qwen3_8_flash_next/nvidia/qsa.py")
KEEP = ("src/", "tests/", "launch/", "scripts/", "pyproject.toml")
W = "/w/repo"

app = modal.App("arc-kv4-triton")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(f"torch=={TORCH}", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install(f"triton=={TRITON}", "numpy>=2.2,<3", "pytest>=8,<10", "pytest-timeout>=2.3,<3", "httpx>=0.28,<0.29",
                      "pyyaml>=6,<7", "pydantic>=2,<3", "requests>=2.32,<3"))


@app.function(image=image, cpu=(8, 16), timeout=45 * 60)  # timeout: a crash ceiling only
def run(bundle: bytes, args: list) -> dict:
    import os

    os.makedirs(W, exist_ok=True)
    tarfile.open(fileobj=io.BytesIO(bundle)).extractall(W)
    env = {**os.environ, "PYTHONPATH": f"{W}/src", "TRITON_INTERPRET": "1", "ARC_PINNED_SITE": f"{W}/pinned"}
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--timeout", "900",
                        *args], cwd=W, env=env, capture_output=True, text=True)
    return {"rc": p.returncode, "log": p.stdout + p.stderr}


def pack(tree, pinned):
    tree = Path(tree)
    names = subprocess.run(["git", "-C", str(tree), "ls-files"], capture_output=True, text=True, check=True).stdout
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tar:
        for name in names.splitlines():
            if name.startswith(KEEP) and (tree / name).is_file():
                tar.add(tree / name, arcname=name)
        for rel in PINNED:
            if (Path(pinned) / rel).is_file():
                tar.add(Path(pinned) / rel, arcname=f"pinned/{rel}")
    return out.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--pinned", default=str(Path.home() / "arc-shared/pinned-vllm"))
    ap.add_argument("pytest_args", nargs="*")
    a = ap.parse_args()
    data = pack(a.tree, a.pinned)
    with modal.enable_output(), app.run():
        r = run.remote(data, a.pytest_args or TESTS)
    print(r["log"])
    return r["rc"]


if __name__ == "__main__":
    raise SystemExit(main())
