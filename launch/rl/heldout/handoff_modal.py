"""Held-out read hand-off (RL redesign stage F): one step's adapter from the training volume to the Nebius bucket, on a
Modal CPU container (no local copy). Called by launch/rl/heldout/hread.py:

    modal run --detach launch/rl/heldout/handoff_modal.py --src validation/<run>/hread/s0008 \
        --prefix flash-input/models/arc-rl/hread/<run>/s0008 --bucket B --endpoint E --region R

Checks every file against the trainer-side manifest (arena.hread: size + sha256, digest), rclone-copies the folder to
assets/<prefix>, compares remote sizes, then writes assets-manifests/<slug>.json in nebius_mirror's format (the VM
checks every fetched file against it). Prints one JSON line {"prefix", "digest", "files", "bytes", "draft", ...}."""
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("arc-rl-hread-handoff")  # include_source=False lost this module in the container (s0016, 29 Oct)
VOLUME = os.environ.get("ARC_HREAD_VOLUME") or "arc-prolong-runs"  # the training run's (hread.train_volume)
runs = modal.Volume.from_name(VOLUME)
CORE = HERE.parents[2] / "src/arena/hread.py" if modal.is_local() else Path("/root/hread_core.py")  # /root: no parents[2]
image = (modal.Image.debian_slim(python_version="3.12").apt_install("rclone")
         .add_local_file(str(CORE), "/root/hread_core.py", copy=True))


@app.function(image=image, volumes={"/runs": runs.read_only()}, secrets=[modal.Secret.from_name("nebius-s3")],
              timeout=45 * 60, retries=0, cpu=(0.125, 8))
def publish(src_rel: str, prefix: str, bucket: str, endpoint: str, region: str, volume: str = "arc-prolong-runs"):
    sys.path.insert(0, "/root")
    import hread_core as H

    os.environ.update(RCLONE_CONFIG_ARC_TYPE="s3", RCLONE_CONFIG_ARC_PROVIDER="Other", RCLONE_CONFIG_ARC_ENV_AUTH="true",
                      RCLONE_CONFIG_ARC_ENDPOINT=endpoint, RCLONE_CONFIG_ARC_REGION=region,
                      RCLONE_CONFIG_ARC_NO_CHECK_BUCKET="true")

    def put(local, remote, single=False):
        cmd = ["rclone", "copyto" if single else "copy", local, remote]
        if not single:
            cmd += ["--transfers", "8", "--s3-upload-concurrency", "4", "--s3-chunk-size", "16M", "--retries", "5",
                    "--timeout", "2m", "--low-level-retries", "20"]
        subprocess.run(cmd, check=True)

    def ls(remote):
        out = subprocess.run(["rclone", "lsjson", "-R", "--files-only", remote], check=True, capture_output=True,
                             text=True).stdout
        return {item["Path"]: item["Size"] for item in json.loads(out)}

    return H.publish_folder(Path("/runs") / src_rel, prefix, f"arc:{bucket}", put, ls, source=f"modal:{volume}")


@app.local_entrypoint()
def main(src: str, prefix: str, bucket: str, endpoint: str, region: str):
    print("HREAD_HANDOFF " + json.dumps(publish.remote(src, prefix, bucket, endpoint, region, VOLUME)), flush=True)
