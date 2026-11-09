"""Build offline source/dependency assets locally. Does not submit any cloud job."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from bootstrap import PIP_SHA256, PIP_WHEEL

ROOT = Path(__file__).resolve().parents[2]


def manifest(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.json"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist/qwen-minimal")
    parser.add_argument(
        "--study",
        default="configs/qwen_minimal_games.json",
        help="Study definition copied to configs/comparison.json (trials A/B/C or P)",
    )
    parser.add_argument(
        "--dependencies",
        action="store_true",
        help="Download locked Linux x86_64 CPU wheels, Node and Pi dependencies",
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("src", "configs", "pi"):
        (output / name).mkdir(exist_ok=True)
    for package in (
        "qwen_baselines",
        "prolong_qwen",
        "schema_qwen",
        "arc_infra",
        "inference",
    ):
        target = output / "src" / package
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            ROOT / "src" / package, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in ("package.json", "package-lock.json", ".node-version", "agent.mjs", "transport.mjs"):
        shutil.copyfile(ROOT / "pi" / name, output / "pi" / name)
    study = json.loads((ROOT / args.study).read_text())
    shutil.copyfile(ROOT / args.study, output / "configs/comparison.json")
    for name in ("bootstrap.py", "modal_run.py"):
        shutil.copyfile(ROOT / "scripts/qwen" / name, output / name)
    notebook = study.get("notebook", "arc_qwen_minimal_games.ipynb")
    metadata = study.get("kernel_metadata", "qwen-minimal-kernel-metadata.json")
    shutil.copyfile(ROOT / "kaggle" / notebook, output / notebook)
    shutil.copyfile(ROOT / "kaggle" / metadata, output / "kernel-metadata.json")
    (output / "qwen-minimal-kernel-metadata.json").unlink(missing_ok=True)
    subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--no-editable",
            "--no-emit-project",
            "--output-file",
            str(output / "requirements.txt"),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if args.dependencies:
        wheels = output / "wheels"
        wheels.mkdir(exist_ok=True)
        pip_wheel = wheels / PIP_WHEEL
        pip_url = (
            "https://files.pythonhosted.org/packages/b7/3f/945ef7ab14dc4f9d7f40288d2df998d1837ee0888ec3659c813487572faa/"
            + PIP_WHEEL
        )
        if (
            not pip_wheel.exists()
            or hashlib.sha256(pip_wheel.read_bytes()).hexdigest() != PIP_SHA256
        ):
            urllib.request.urlretrieve(pip_url, pip_wheel)
        if hashlib.sha256(pip_wheel.read_bytes()).hexdigest() != PIP_SHA256:
            raise ValueError("Pinned bootstrap pip wheel mismatch")
        subprocess.run(
            [
                "uv",
                "run",
                "--with",
                "pip",
                "python",
                "-m",
                "pip",
                "download",
                "--require-hashes",
                "--only-binary=:all:",
                "--python-version",
                "312",
                "--platform",
                "manylinux_2_28_x86_64",
                "--platform",
                "manylinux2014_x86_64",
                "--implementation",
                "cp",
                "--abi",
                "cp312",
                "-r",
                str(output / "requirements.txt"),
                "-d",
                str(wheels),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                "npm",
                "ci",
                "--ignore-scripts",
                "--os=linux",
                "--cpu=x64",
                "--prefix",
                str(output / "pi"),
            ],
            check=True,
        )
        filename = "node-v22.19.0-linux-x64.tar.xz"
        base = "https://nodejs.org/dist/v22.19.0/"
        sums = urllib.request.urlopen(base + "SHASUMS256.txt", timeout=60).read().decode()
        expected = next(
            line.split()[0] for line in sums.splitlines() if line.endswith("  " + filename)
        )
        target = output / filename
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            urllib.request.urlretrieve(base + filename, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned Node distribution hash mismatch")
        with tarfile.open(target) as archive:
            archive.extractall(output / "node", filter="data")
        (output / "NODE_SHA256.txt").write_text(expected + "  " + filename + "\n")
    identity = {
        "files": manifest(output),
        "source": "current working tree; no frozen historical agent archive",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dependencies_prepared": args.dependencies,
        "gpu_benchmarks_launched": False,
    }
    (output / "MANIFEST.json").write_text(json.dumps(identity, indent=2))
    print(f"Prepared {output}; {len(identity['files'])} hashed files; no GPU jobs submitted")


if __name__ == "__main__":
    main()
