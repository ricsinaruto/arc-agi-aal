"""Replay check of scripted points as a Modal CPU job (8 CPU, 16 GiB, no GPU, network blocked in the function).
REPLAY_BUNDLE=<bundle dir> REPLAY_POINTS=<points.json> REPLAY_GAMES=<environment_files> \
  modal run modal_replay.py --out <new empty dir> [--concurrency 6] [--arm-study '{"handover": {"mode": "suggest"}}']
Writes <out>/replay.log and <out>/replay.json, and with the suggest arm <out>/prompts.json (data from the container:
read with python -I)."""
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("arc-hx7-replay-check")
image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    bundle, points, games = (Path(os.environ[k]).resolve() for k in ("REPLAY_BUNDLE", "REPLAY_POINTS", "REPLAY_GAMES"))
    image = (image.pip_install_from_requirements(str(bundle / "requirements.txt"))
             .add_local_dir(str(bundle), "/job/bundle", copy=True, ignore=["wheels/**", "dataset-upload/**"])
             .add_local_dir(str(games), "/job/games", copy=True)
             .add_local_file(str(points), "/job/points.json", copy=True)
             .add_local_file(str(HERE / "replay_check_len.py"), "/job/replay_check_len.py", copy=True))


@app.function(image=image, cpu=8, memory=16384, timeout=3600, retries=0, block_network=True)
def replay(concurrency: int, arm_study: str = "{}"):
    import subprocess
    done = subprocess.run([sys.executable, "/job/replay_check_len.py", "/job/bundle/src", "/job/bundle", "/tmp/out",
                           "/job/games", "/job/points.json", str(concurrency), arm_study], capture_output=True, text=True,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    report = Path("/tmp/out/replay.json")
    prompts = Path("/tmp/out/prompts.json")
    return {"returncode": done.returncode, "stdout": done.stdout[-400000:], "stderr": done.stderr[-20000:],
            "replay": report.read_text() if report.is_file() else None,
            "prompts": prompts.read_text() if prompts.is_file() else None}


@app.local_entrypoint()
def main(out: str, concurrency: int = 6, arm_study: str = "{}"):
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    result = replay.remote(concurrency, arm_study)
    (target / "replay.log").write_text(result["stdout"] + "\n--- stderr ---\n" + result["stderr"])
    if result["replay"]:
        (target / "replay.json").write_text(result["replay"])
    if result.get("prompts"):
        (target / "prompts.json").write_text(result["prompts"])
    print("returncode", result["returncode"])
    print(result["stdout"].strip().splitlines()[-1] if result["stdout"].strip() else result["stderr"][-2000:])
