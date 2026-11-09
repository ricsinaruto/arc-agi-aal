"""One-time mirror of arc-prolong-assets prefixes to the Nebius bucket (Modal CPU job; `nebius_validation.py
mirror-plan` prints the command for a bundle).

    uv run --extra modal modal run --detach scripts/nebius_mirror.py --bucket B --endpoint https://storage.R.nebius.cloud \
        --region R --prefixes flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4,games,...

Needs the Modal secret ``nebius-s3`` (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY of the bucket's service account).
One container per prefix: size + sha256 of every file read from the volume, ``rclone copy`` to assets/<prefix>,
remote sizes compared, and only then assets-manifests/<slug>.json (the launcher treats a manifest as "mirrored";
the VM checks every file against it). Re-running skips files already there. per container hour.

``--source dfr-exact`` (11 October): the prefixes come from the ``arc-dfr-exact`` volume (dfranzen's Kaggle inputs in the
Kaggle layout), mounted at /assets/flash-input, so ``flash-input/models/dfranzen/...`` in the bucket is
``/kaggle/input/models/dfranzen/...`` on the VM. Prefixes there that the assets volume also holds (``games``) keep
coming from the assets volume. One container, one prefix after the other, so ``--detach`` keeps the whole job alive.
"""

import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

app = modal.App("arc-nebius-mirror", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
# arc-dfr-exact is mounted only with ARC_MIRROR_DFR=1 (27 Oct: the volume is gone from env main, and an unconditional
# mount failed every assets mirror at app start)
DFR = os.environ.get("ARC_MIRROR_DFR") == "1"
dfr_inputs = modal.Volume.from_name("arc-dfr-exact", version=2) if DFR else None
image = modal.Image.debian_slim(python_version="3.12").apt_install("rclone").add_local_file(
    __file__, "/root/nebius_mirror.py", copy=True)


def asset_slug(prefix):
    return prefix.strip("/").replace("/", "--")


def sha256_file(path, chunk=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb", buffering=0) as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root, prefix, workers=8, source="modal:arc-prolong-assets"):
    base = Path(root) / prefix
    files = sorted(p for p in base.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no files under {base}")
    with ThreadPoolExecutor(workers) as pool:
        digests = list(pool.map(sha256_file, files))
    entries = {str(p.relative_to(base)): {"size": p.stat().st_size, "sha256": d} for p, d in zip(files, digests)}
    return {"prefix": prefix, "source": source, "created_epoch": round(time.time()),
            "total_bytes": sum(e["size"] for e in entries.values()), "files": entries}


def _mirror(prefix, bucket, endpoint, region, source="modal:arc-prolong-assets"):
    os.environ.update(RCLONE_CONFIG_ARC_TYPE="s3", RCLONE_CONFIG_ARC_PROVIDER="Other",
                      RCLONE_CONFIG_ARC_ENV_AUTH="true", RCLONE_CONFIG_ARC_ENDPOINT=endpoint,
                      RCLONE_CONFIG_ARC_REGION=region, RCLONE_CONFIG_ARC_NO_CHECK_BUCKET="true")
    started = time.time()
    manifest = build_manifest("/assets", prefix, source=source)
    hashed = time.time()
    remote = f"arc:{bucket}/assets/{prefix}"
    subprocess.run(["rclone", "copy", f"/assets/{prefix}", remote, "--copy-links", "--transfers", "8",
                    "--checkers", "16", "--s3-upload-concurrency", "8", "--s3-chunk-size", "64M", "--retries", "5",
                    "--stats", "60s", "--stats-one-line"], check=True)
    listing = json.loads(subprocess.run(["rclone", "lsjson", "-R", "--files-only", remote], check=True,
                                        capture_output=True, text=True).stdout)
    sizes = {item["Path"]: item["Size"] for item in listing}
    wrong = [rel for rel, meta in manifest["files"].items() if sizes.get(rel) != meta["size"]]
    if wrong:
        raise RuntimeError(f"{prefix}: {len(wrong)} remote files differ in size, e.g. {wrong[:3]}")
    path = Path("/tmp") / f"{asset_slug(prefix)}.json"
    path.write_text(json.dumps(manifest, indent=1))
    subprocess.run(["rclone", "copyto", str(path), f"arc:{bucket}/assets-manifests/{path.name}"], check=True)
    return {"prefix": prefix, "files": len(manifest["files"]), "gib": round(manifest["total_bytes"] / 2 ** 30, 2),
            "hash_seconds": round(hashed - started), "upload_seconds": round(time.time() - hashed)}


@app.function(image=image, timeout=2 * 3600, retries=0, max_containers=8,
              volumes={"/assets": assets.read_only()}, secrets=[modal.Secret.from_name("nebius-s3")])
def mirror(prefix: str, bucket: str, endpoint: str, region: str):
    return _mirror(prefix, bucket, endpoint, region)


if DFR:
    @app.function(image=image, timeout=4 * 3600, retries=0, max_containers=1,
                  volumes={"/assets/flash-input": dfr_inputs.read_only()}, secrets=[modal.Secret.from_name("nebius-s3")])
    def mirror_dfr(prefixes: list, bucket: str, endpoint: str, region: str):
        rows = []
        for prefix in prefixes:
            if not prefix.startswith("flash-input/"):
                raise ValueError(f"{prefix}: the arc-dfr-exact volume is mounted at flash-input/")
            rows.append(_mirror(prefix, bucket, endpoint, region, source="modal:arc-dfr-exact"))
            print(json.dumps(rows[-1]), flush=True)
        return rows


@app.local_entrypoint()
def main(bucket: str, endpoint: str, region: str, prefixes: str, source: str = "assets"):
    wanted = [p.strip("/") for p in prefixes.split(",") if p.strip("/")]
    if source == "dfr-exact":
        if not DFR:
            raise SystemExit("--source dfr-exact needs ARC_MIRROR_DFR=1 (and the arc-dfr-exact volume)")
        call = mirror_dfr.spawn(wanted, bucket, endpoint, region)
        return print(json.dumps({"function_call_id": call.object_id, "prefixes": wanted}))
    if source != "assets":
        raise ValueError("--source is assets or dfr-exact")
    for result in mirror.starmap([(p, bucket, endpoint, region) for p in wanted]):
        print(json.dumps(result))
