"""The Pennyroyal runtime dataset on Kaggle: locate, verify, extract, check.

Standard library only: this runs under Kaggle's Python before any venv exists. The dataset
``richardcsaky/arc-sglang-runtime`` carries the image's ``/opt/pennyroyal`` (fork checkout
and uv venv), ``/opt/python`` (the venv's interpreter) and a CUDA 13.3 toolchain subset as
one gzip tar stream split into ``.blob`` parts, plus ``runtime-manifest.json``. It is
extracted to the same absolute paths so the venv's links and ``pyvenv.cfg`` stay valid.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

DATASET = "richardcsaky/arc-sglang-runtime"
MANIFEST_NAME = "runtime-manifest.json"
SCHEMA = 1
REPO_ROOT = Path("/opt/pennyroyal")
PYTHON_ROOT = Path("/opt/python")
CUDA_HOME = Path("/usr/local/cuda-13.3")
MODEL_DIR = Path("/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1")
SCRATCH = Path("/tmp/arc-sglang")
EXPECTED_PYTHON = "3.12.13"
MIN_FREE_BYTES = 30 * 1024**3
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"


def dataset_mount(ref: str, input_root="/kaggle/input") -> Path:
    root = Path(input_root)
    matches = [p for p in (root / "datasets" / ref, root / ref.split("/")[1]) if p.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one mounted dataset for {ref}; found {matches}")
    return matches[0]


def locate(input_root="/kaggle/input") -> Path:
    return dataset_mount(DATASET, input_root)


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify_manifest(mount: Path, *, full_hashes=True) -> dict:
    manifest = json.loads((Path(mount) / MANIFEST_NAME).read_text())
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError(f"Unsupported runtime manifest schema {manifest.get('schema')}")
    for key in ("image", "index_digest", "parts", "roots", "stream_sha256"):
        if key not in manifest:
            raise RuntimeError(f"Runtime manifest lacks {key}")
    for part in manifest["parts"]:
        path = Path(mount) / part["file"]
        if not path.is_file():
            raise RuntimeError(f"Runtime part missing: {part['file']}")
        if path.stat().st_size != part["size"]:
            raise RuntimeError(f"Runtime part size differs: {part['file']}")
        if full_hashes and sha256_file(path) != part["sha256"]:
            raise RuntimeError(f"Runtime part hash differs: {part['file']}")
    return manifest


def free_bytes(path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def extract(mount: Path, manifest: dict, *, root="/", replace=False, min_free=MIN_FREE_BYTES) -> dict:
    """``cat parts | tar -xzf - -C root``; refuses existing roots unless ``replace``."""
    root = Path(root)
    started = time.time()
    existing = [r for r in manifest["roots"] if (root / r).exists()]
    if existing and not replace:
        raise RuntimeError(f"Runtime roots already exist under {root}: {existing}")
    for r in existing:
        shutil.rmtree(root / r)
    for r in manifest["roots"]:
        (root / r).parent.mkdir(parents=True, exist_ok=True)
    before = free_bytes(root)
    if before < min_free:
        raise RuntimeError(f"Only {before / 1024**3:.1f} GiB free under {root}; need {min_free / 1024**3:.0f}")
    parts = [str(Path(mount) / p["file"]) for p in manifest["parts"]]
    cat = subprocess.Popen(["cat", *parts], stdout=subprocess.PIPE)
    tar = subprocess.run(
        ["tar", "-xzf", "-", "-C", str(root), "--no-same-owner", "--exclude=__pycache__", "--exclude=*.pyc"],
        stdin=cat.stdout, capture_output=True, text=True,
    )
    cat.stdout.close()
    cat.wait()
    if tar.returncode or cat.returncode:
        raise RuntimeError(f"Runtime extraction failed: cat={cat.returncode} tar={tar.returncode} {tar.stderr[-2000:]}")
    record = {
        "extract_seconds": round(time.time() - started, 1),
        "roots": [str(root / r) for r in manifest["roots"]],
        "free_bytes_before": before,
        "free_bytes_after": free_bytes(root),
    }
    python = root / REPO_ROOT.relative_to("/") / ".venv/bin/python"
    if str(root) == "/":
        version = subprocess.run([str(python), "--version"], capture_output=True, text=True, timeout=30)
        record["python_version"] = version.stdout.strip() or version.stderr.strip()
        if EXPECTED_PYTHON not in record["python_version"]:
            raise RuntimeError(f"Runtime python is {record['python_version']!r}, expected {EXPECTED_PYTHON}")
    return record


def relocate(root: Path, *, old_prefix="/opt", new_prefix=None) -> dict:
    """Rewrite absolute ``/opt`` references when the runtime cannot live at ``/opt``."""
    root = Path(root)
    new_prefix = new_prefix or str(root / "opt")
    repo = root / "opt/pennyroyal"
    venv = repo / ".venv"
    changed = []
    cfg = venv / "pyvenv.cfg"
    if cfg.is_file():
        text = cfg.read_text()
        cfg.write_text(text.replace(old_prefix, new_prefix))
        changed.append(str(cfg))
    for link in (venv / "bin").glob("python*"):
        if link.is_symlink():
            target = os.readlink(link)
            if target.startswith(old_prefix):
                link.unlink()
                link.symlink_to(target.replace(old_prefix, new_prefix, 1))
                changed.append(str(link))
    site = venv / "lib/python3.12/site-packages"
    for path in list(site.glob("*.pth")) + list(site.glob("__editable__*.py")):
        text = path.read_text()
        if old_prefix in text:
            path.write_text(text.replace(old_prefix, new_prefix))
            changed.append(str(path))
    for script in (venv / "bin").iterdir():
        if script.is_file() and not script.is_symlink():
            head = script.read_bytes()[:200]
            if head.startswith(b"#!") and old_prefix.encode() in head:
                data = script.read_bytes()
                script.write_bytes(data.replace(old_prefix.encode(), new_prefix.encode(), 1))
                changed.append(str(script))
    return {"old_prefix": old_prefix, "new_prefix": new_prefix, "changed": changed}


CHECK_SCRIPT = r"""
import json, os, platform, sys
record = {"python": sys.version.split()[0], "executable": sys.executable, "glibc": platform.libc_ver()[1]}
import torch
record["torch"] = torch.__version__
record["torch_cuda"] = torch.version.cuda
record["cuda_available"] = torch.cuda.is_available()
if record["cuda_available"]:
    record["device_name"] = torch.cuda.get_device_name(0)
    record["capability"] = list(torch.cuda.get_device_capability(0))
    record["total_memory_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    x = torch.ones(1024, device="cuda") * 2
    record["cuda_kernel_ok"] = float(x.sum().item()) == 2048.0
import flashinfer
record["flashinfer"] = getattr(flashinfer, "__version__", None)
import sgl_kernel
record["sgl_kernel"] = getattr(sgl_kernel, "__version__", None)
import sglang
record["sglang_file"] = sglang.__file__
record["sglang_version"] = getattr(sglang, "__version__", None)
from sglang.srt.server_args import ServerArgs  # noqa: F401
record["nixl_imported"] = "nixl" in sys.modules
print(json.dumps(record))
"""


def _run(command, env, timeout=120) -> dict:
    try:
        done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
        return {"returncode": done.returncode, "stdout": done.stdout[-2000:], "stderr": done.stderr[-2000:]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def host_facts() -> dict:
    facts = {"kaggle_python": sys.version.split()[0]}
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        facts["rlimit_memlock"] = [soft, hard]
    except (ValueError, OSError):
        pass
    for label, path in (("root", "/"), ("tmp", "/tmp"), ("shm", "/dev/shm"), ("working", "/kaggle/working")):
        try:
            usage = shutil.disk_usage(path)
            facts[f"disk_{label}"] = {"total": usage.total, "free": usage.free}
        except OSError:
            pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal", "MemAvailable")):
                key, value = line.split(":", 1)
                facts[key] = int(value.split()[0]) * 1024
    except OSError:
        pass
    return facts


def check(env: dict, *, python=None, cuda_home=CUDA_HOME, cc="/usr/bin/gcc") -> dict:
    """Import the runtime under the launch environment and record the toolchain."""
    python = str(python or REPO_ROOT / ".venv/bin/python")
    record = {"host": host_facts()}
    probe = _run([python, "-c", CHECK_SCRIPT], env, timeout=300)
    record["probe"] = probe
    try:
        record["runtime"] = json.loads(probe["stdout"].strip().splitlines()[-1])
    except (KeyError, IndexError, ValueError):
        record["runtime"] = None
    record["nvcc"] = _run([str(Path(cuda_home) / "bin/nvcc"), "--version"], env)
    record["gcc"] = _run([cc, "--version"], env)
    runtime = record["runtime"] or {}
    problems = []
    if not runtime.get("cuda_available"):
        problems.append("CUDA unavailable under the runtime python")
    if runtime.get("nixl_imported"):
        problems.append("nixl was imported; the Fedora-built library must stay unused")
    if runtime and runtime.get("python") != EXPECTED_PYTHON:
        problems.append(f"runtime python {runtime.get('python')} != {EXPECTED_PYTHON}")
    record["problems"] = problems
    return record


def prefetch_model(model_dir=MODEL_DIR, *, threads=8, block=16 << 20, log=print) -> dict:
    """Read every checkpoint file into the page cache (vLLM's loader does the same with 8
    threads and 16 MiB blocks). Kaggle's NFS input served SGLang's loader at 4-38 s per
    600 MB shard cold; warm reads take about 1.7 s. The host has 180 GB of RAM for the
    126 GB checkpoint. Returns bytes read and seconds."""
    import concurrent.futures
    import time

    files = sorted(p for p in Path(model_dir).iterdir() if p.is_file())
    started = time.monotonic()

    def read(path):
        total = 0
        with path.open("rb", buffering=0) as handle:
            while True:
                chunk = handle.read(block)
                if not chunk:
                    return total
                total += len(chunk)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        sizes = list(pool.map(read, files))
    record = {"files": len(files), "bytes": sum(sizes), "seconds": round(time.monotonic() - started, 1)}
    log(f"[prefetch] {record}")
    return record


CUDART_CANDIDATES = (
    "opt/pennyroyal/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13",
)


def link_cudart(root="/", cuda_home=CUDA_HOME, candidates=CUDART_CANDIDATES, log=print) -> dict:
    """Give the JIT linker a shared cudart under ``<cuda_home>/lib64``.

    The packaged CUDA subset carries only the static cudart, but SGLang's JIT modules link
    with ``-L<cuda_home>/lib64 -lcudart`` (smoke 2 died on ``cannot find -lcudart``). The
    venv's ``nvidia/cu13`` wheel ships ``libcudart.so.13``; symlink it as ``libcudart.so``.
    """
    root = Path(root)
    cuda = root / str(cuda_home).lstrip("/")
    lib64 = cuda / "lib64"
    source = next((root / c for c in candidates if (root / c).is_file()), None)
    if source is None:
        raise RuntimeError(f"No shared libcudart under {root} in {candidates}")
    lib64.mkdir(parents=True, exist_ok=True)
    links = {}
    for name in ("libcudart.so", "libcudart.so.13"):
        target = lib64 / name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(source)
        links[name] = str(source)
    log(f"[cudart] linked {links}")
    return {"lib64": str(lib64), "links": links}


CUDA_DRIVER_CANDIDATES = (
    "usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "usr/lib/x86_64-linux-gnu/libcuda.so",
    "usr/local/nvidia/lib64/libcuda.so.1",
    "usr/lib64/libcuda.so.1",
    "usr/local/cuda/lib64/stubs/libcuda.so",
)


def find_cuda_driver(root="/", candidates=CUDA_DRIVER_CANDIDATES):
    """The driver's ``libcuda.so.1`` under ``root``: a candidate path, else (real root only)
    the loader's own answer from ``ldconfig -p``; ``None`` when neither finds it."""
    root = Path(root)
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path
    if root != Path("/"):
        return None
    try:
        listing = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, check=False).stdout
    except OSError:
        return None
    for line in listing.splitlines():
        name, _, target = line.strip().partition(" => ")
        if name.startswith("libcuda.so.1") and target and Path(target.strip()).is_file():
            return Path(target.strip())
    return None


def link_cuda_driver(root="/", cuda_home=CUDA_HOME, candidates=CUDA_DRIVER_CANDIDATES, log=print) -> dict:
    """Give the JIT linker a ``libcuda.so`` under ``<cuda_home>/lib64/stubs`` and ``lib64``.

    flashinfer's XQA module links with ``-L<cuda_home>/lib64/stubs -lcuda`` (smoke 3 died at
    CUDA-graph capture on ``cannot find -lcuda``); the packaged CUDA subset carries no stub, so
    the box's own driver library stands in for it (the kernel runs on that driver anyway).
    """
    root = Path(root)
    source = find_cuda_driver(root, candidates)
    if source is None:
        raise RuntimeError(f"No libcuda.so.1 under {root} in {candidates} or ldconfig")
    links = {}
    for folder in ("lib64/stubs", "lib64"):
        directory = root / str(cuda_home).lstrip("/") / folder
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "libcuda.so"
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(source)
        links[str(target)] = str(source)
    log(f"[cuda-driver] linked {links}")
    return {"source": str(source), "links": links}


def verify_model(model_dir=MODEL_DIR) -> dict:
    model_dir = Path(model_dir)
    record = {"model_dir": str(model_dir), "files": {}}
    for name in ("config.json", "model.safetensors.index.json", "chat_template.jinja", "tokenizer.json"):
        record["files"][name] = (model_dir / name).is_file()
    if not all(record["files"].values()):
        raise RuntimeError(f"Model files missing under {model_dir}: {record['files']}")
    record["tokenizer_sha256"] = sha256_file(model_dir / "tokenizer.json")
    record["tokenizer_matches_recipe"] = record["tokenizer_sha256"] == TOKENIZER_SHA256
    manifest = model_dir / "MODEL_MANIFEST.json"
    if manifest.is_file():
        try:
            record["model_manifest"] = json.loads(manifest.read_text())
        except ValueError:
            pass
    return record
