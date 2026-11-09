"""Live Modal GPUs by app, from Modal itself (registered or not): one JSON object on stdout,
{app_id: {"name": app name, "gpus": {"B300": 2}, "containers": n}}. Run with the repo venv (needs the modal package).
Used by gpu_reconcile.py and records.py so an app without a queue row still counts by its real GPU type."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

MODAL = str(Path(sys.executable).parent / "modal")  # the venv running this script


async def live():
    from modal.client import _Client
    from modal_proto import api_pb2

    out = subprocess.run([MODAL, "container", "list", "--json"], capture_output=True, text=True, timeout=60, check=True)
    client = await _Client.from_env()
    apps = {}

    async def one(c):
        info = (await client.stub.TaskGetInfo(api_pb2.TaskGetInfoRequest(task_id=c["container_id"]))).info
        kind = info.gpu_config.gpu_type or info.gpu_type or "CPU"
        a = apps.setdefault(c["app_id"], {"name": c.get("app_name"), "gpus": {}, "containers": 0})
        a["containers"] += 1
        if kind.upper() != "CPU":
            a["gpus"][kind] = a["gpus"].get(kind, 0) + max(1, info.gpu_config.count or 1)

    await asyncio.gather(*(one(c) for c in json.loads(out.stdout)))
    return apps


if __name__ == "__main__":
    try:
        print(json.dumps(asyncio.run(live())))
    except Exception as e:  # callers fall back to the queue rows
        print(json.dumps({"error": repr(e)[:300]}))
        sys.exit(1)
