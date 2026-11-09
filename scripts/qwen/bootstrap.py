"""Offline Kaggle/Modal entry point for the prepared three-trial comparison."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PIP_WHEEL = "pip-25.2-py3-none-any.whl"
PIP_SHA256 = "6d67a2b4e7f14d8b31b8b52648866fa717f45a1eb70e83002f4331d07e953717"


def install_agent_environment(bundle, venv, env):
    """Install locked wheels without depending on the host's ensurepip package."""
    wheel = bundle / "wheels" / PIP_WHEEL
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != PIP_SHA256:
        raise ValueError("Pinned bootstrap pip wheel mismatch")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    python = str(venv / "bin/python")
    # Run pip directly from its verified wheel in the target interpreter. Do not
    # inherit host site-packages or depend on host pip/ensurepip availability.
    installer_env = {**env, "PYTHONPATH": str(wheel), "PIP_CONFIG_FILE": os.devnull}
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--require-hashes",
            "--find-links",
            str(bundle / "wheels"),
            "-r",
            str(bundle / "requirements.txt"),
        ],
        env=installer_env,
        check=True,
    )
    subprocess.run([python, "-m", "pip", "check"], env=installer_env, check=True)
    return python


def collect_evidence(working, output):
    """Copy trial outputs, serving diagnostics and setup report to a persisted directory.

    Installations, staged Node/Pi trees and serving scratch stay behind so notebook
    outputs hold evidence only; Kaggle otherwise lists thousands of runtime files.
    """
    working, output = Path(working), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (working / "trials").is_dir():
        shutil.copytree(working / "trials", output / "trials", dirs_exist_ok=True)
    if (working / "setup.json").is_file():
        shutil.copyfile(working / "setup.json", output / "setup.json")
    if (working / "serving").is_dir():
        (output / "serving").mkdir(exist_ok=True)
        for source in (working / "serving").iterdir():
            if source.is_file() and source.suffix in {".json", ".log"}:
                shutil.copyfile(source, output / "serving" / source.name)
        for name in ("cache-probe", "native-action-preflight"):
            if (working / "serving" / name).is_dir():
                shutil.copytree(working / "serving" / name, output / "serving" / name, dirs_exist_ok=True)


def verify(bundle):
    identity = json.loads((bundle / "MANIFEST.json").read_text())
    if not identity["dependencies_prepared"]:
        raise RuntimeError("Bundle lacks offline dependencies; rebuild with --dependencies")
    for name, expected in identity["files"].items():
        path = bundle / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Offline source/dependency mismatch: {name}")
    return identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--working", default="/kaggle/working/qwen-minimal")
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--games-dir")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--gameplay-seconds", type=int, help="Explicit smoke-test duration override"
    )
    parser.add_argument("--max-seconds", type=int, default=5340)
    parser.add_argument(
        "--output", help="Persist only trial evidence, serving diagnostics and setup here"
    )
    args = parser.parse_args()
    bundle, working = Path(args.bundle).resolve(), Path(args.working).resolve()
    started = time.time()
    identity = verify(bundle)
    study = json.loads((bundle / "configs/comparison.json").read_text())
    if args.gameplay_seconds is not None:
        if not 1 <= args.gameplay_seconds <= 900:
            raise ValueError("Smoke gameplay must be between 1 and 900 seconds")
        study["gameplay_seconds"] = args.gameplay_seconds
    if args.max_seconds < 120:
        raise ValueError("Startup/run envelope must include cleanup time")
    if study["trials"] not in (["A", "B", "C"], ["B", "C"], ["P"], ["R"], ["S"]) or (
        study["serving_profile"] != "qwen-minimal-native"
    ):
        raise ValueError("Unexpected comparison definition")
    if args.verify_only:
        print(json.dumps({"verified_files": len(identity["files"]), "gpu_started": False}))
        return
    working.mkdir(parents=True, exist_ok=False)
    env = {
        **os.environ,
        "PYTHONPATH": str(bundle / "src"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PIP_NO_INDEX": "1",
        "MPLBACKEND": "Agg",
        "ONLY_RESET_LEVELS": "true",
    }
    venv = working / "agent-venv"
    python = install_agent_environment(bundle, venv, env)
    # Dataset mounts need not preserve executable modes or permit executable maps.
    # Stage the pinned Node distribution and native Pi dependencies onto working storage.
    with tarfile.open(bundle / "node-v22.19.0-linux-x64.tar.xz") as archive:
        archive.extractall(working / "node", filter="data")
    shutil.copytree(bundle / "pi", working / "pi", symlinks=True)
    try:
        # Import the new package only after its isolated CPU dependencies are installed.
        subprocess.run(
            [
                python,
                "-m",
                "qwen_baselines.cli",
                "serve-native",
                "--working",
                str(working / "serving"),
                "--input-root",
                args.input_root,
            ],
            env=env,
            check=True,
            timeout=max(1, started + args.max_seconds - 60 - time.time()),
        )
        (working / "setup.json").write_text(
            json.dumps({"setup_seconds": time.time() - started, "bundle": identity}, indent=2)
        )
        games = args.games_dir
        if not games:
            games = subprocess.check_output(
                [
                    python,
                    "-c",
                    "from arc_infra.arcade import find_games; import sys; print(find_games(sys.argv[1]))",
                    args.input_root,
                ],
                env=env,
                text=True,
            ).strip()
        subprocess.run(
            [
                python,
                "-m",
                "qwen_baselines.cli",
                "run",
                "--baseline",
                *(["all"] if study["trials"] == ["A", "B", "C"] else study["trials"]),
                "--seconds",
                str(study["gameplay_seconds"]),
                "--games",
                *study["games"],
                "--games-dir",
                games,
                "--output",
                str(working / "trials"),
                "--pi-dir",
                str(working / "pi"),
                "--node",
                str(working / "node/node-v22.19.0-linux-x64/bin/node"),
            ],
            env=env,
            check=True,
            timeout=max(1, started + args.max_seconds - 60 - time.time()),
        )
    finally:
        # Persist evidence first: a slow or failed server stop must not lose the trials.
        if args.output:
            collect_evidence(working, args.output)
        # The pinned recipe only stops its own recorded server identity.
        program = "from arc_infra.serving import load_recipe; import sys; load_recipe(sys.argv[1],sys.argv[2]).stop_owned_server()"
        try:
            subprocess.run(
                [python, "-c", program, str(working / "serving"), args.input_root],
                env=env,
                check=False,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            print(json.dumps({"event": "server_stop_timeout", "seconds": 180}), flush=True)
        if args.output:
            collect_evidence(working, args.output)  # refresh serving logs after the stop


if __name__ == "__main__":
    main()
