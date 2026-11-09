"""Modal CPU fan-out for ``blocks_window_replay.py`` ("prioritize running on modal cpu-s for heavier
stuff, not local"). No GPU.

Each job is one ``blocks_window_replay.main`` call (its own argv: traces, windows, streams, pool blocks, margin, output
cap, seeds, ``--lockstep``); a container runs its batch of jobs in a fork pool. The recorded traces (run A, run B and
the first three-block run L: ``arena.json``, ``cache-layout.json``, ``admission.jsonl`` of each) and the served
prune25 cost table are baked into the image; a job names them as ``A=@A`` / ``--layout @layout`` and the container
substitutes its own paths. The entrypoint writes every job's result JSON (the script's own ``--out`` format) under
``--out-dir`` as ``<key>.json``.

Usage (from a worktree; the ledger is the main checkout's):
  ARC_VALIDATION_ROOT=. uv run --extra modal modal run --detach \\
      scripts/arena_analysis/modal_window_replay.py --jobs jobs.json --run-id sim-window-replay-v1-modal-20260930 \\
      --out-dir docs/validation-20260930/window-replay

``jobs.json``: ``[{"key": "...", "args": ["--run", "A=@A", "--layout", "@layout", "--blocks", "1630", ...]}, ...]``.
The entrypoint reserves the bound (containers x (timeout + 60) x the CPU/memory rates x 1.15) in docs/spend.json under
an exclusive lock and reconciles it with the measured container seconds afterwards.
"""

import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LOCAL = modal.is_local()
REPO = HERE.parents[1] if LOCAL else HERE
ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or REPO).resolve()  # the ledger's checkout
CPU = 8
MEMORY_MIB = 8192
TIMEOUT = 900
MAX_CONTAINERS = 20
REMOTE = "/root/wr"
RUNS = Path("./runs")
TRACES = {
    "A": RUNS / "modal-validation/arena-hero-v2-int4-streams-v1-modal-20260929/arena-hero-v2-int4-streams-v1-modal-20260929/arena/trials/duck-transfer",
    "B": RUNS / "modal-validation/arena-hero-v2-128k-streams-v1-modal-20260929/arena-hero-v2-128k-streams-v1-modal-20260929/arena/trials/duck-transfer",
    "L": RUNS / "modal-validation/arena-hero-v2-blocks-v1-modal-20260929/arena-hero-v2-blocks-v1-modal-20260929/arena/trials/duck-transfer",
}
# The prune25 v2 startup's cost table (num_blocks is overridden by every job's --blocks; pruning leaves the KV spec).
LAYOUT = RUNS / "campaign-20260930/arena-campaign-blocks2-prune25-9h-v2-modal-20260930/arena/serving/cache-layout.json"
# Another cost table (e.g. the MTP-1 prune25 pilot's, docs/validation-20260930/prune25-mtp1-cache-layout.json).
LAYOUT = Path(os.environ.get("ARC_REPLAY_LAYOUT") or LAYOUT)
AUTHORIZATION = ("'prioritize running on modal cpu-s for heavier stuff, not local' ("
                 "coordinator): window/gate replays for the nine-hour restart set on Modal CPUs, no GPU")

app = modal.App("arc-window-replay")
image = modal.Image.debian_slim(python_version="3.12")
if LOCAL:
    image = (image
             .add_local_file(str(HERE / "blocks_window_replay.py"), f"{REMOTE}/scripts/arena_analysis/blocks_window_replay.py")
             .add_local_file(str(HERE / "gate_replay.py"), f"{REMOTE}/scripts/arena_analysis/gate_replay.py")
             .add_local_file(str(REPO / "src/arena/cache.py"), f"{REMOTE}/src/arena/cache.py")
             .add_local_file(str(LAYOUT), f"{REMOTE}/layout.json"))
    for name, trace in TRACES.items():
        for f in ("arena.json", "cache-layout.json", "admission.jsonl"):
            image = image.add_local_file(str(trace / f), f"{REMOTE}/traces/{name}/{f}")


def _remote_args(args):
    out = []
    for a in args:
        if a == "@layout":
            a = f"{REMOTE}/layout.json"
        elif "=@" in a:
            name, ref = a.split("=@", 1)
            a = f"{name}={REMOTE}/traces/{ref}"
        out.append(a)
    return out


def _job(item):
    import contextlib
    import importlib.util
    import io
    import tempfile

    spec = importlib.util.spec_from_file_location(
        "blocks_window_replay", f"{REMOTE}/scripts/arena_analysis/blocks_window_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = Path(tempfile.mkdtemp()) / "out.json"
    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        mod.main(_remote_args(item["args"]) + ["--out", str(out)])
    return item["key"], json.loads(out.read_text()), time.time() - t


@app.function(image=image, cpu=float(CPU), memory=MEMORY_MIB, timeout=TIMEOUT, retries=0,
              max_containers=MAX_CONTAINERS, block_network=True)
def run_batch(items):
    import multiprocessing as mp

    open(f"{REMOTE}/src/arena/__init__.py", "a").close()
    t0 = time.time()
    with mp.get_context("fork").Pool(min(CPU, len(items))) as pool:
        out = list(pool.imap_unordered(_job, sorted(items, key=lambda x: -x.get("cost", 1.0))))
    return {"results": out, "seconds": time.time() - t0}


def cost(item):
    """Relative run time: seeds x streams x (pools x floors) for a steady job, seeds x streams for lockstep."""
    a = item["args"]
    seeds = int(a[a.index("--seeds") + 1]) if "--seeds" in a else 10
    streams = sum(int(x) for x in a[a.index("--streams") + 1:] if x.isdigit()) if "--streams" in a else 37
    lockstep = "--lockstep" in a
    return seeds * streams * (1 if lockstep else 4)


def batches(items, n):
    bins, load = [[] for _ in range(n)], [0.0] * n
    for item in sorted(items, key=lambda x: -x["cost"]):
        i = min(range(n), key=lambda j: load[j])
        bins[i].append(item)
        load[i] += item["cost"]
    return [b for b in bins if b]


def _per_second(data):
    r = data["rates_per_second"]
    return CPU * r["physical_cpu_core"] + MEMORY_MIB / 1024 * r["memory_gib"]


def reserve(run_id, containers, note, max_bound):
    import fcntl

    path = ROOT / "docs/spend.json"
    with path.open("r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        data = json.load(file)
        bound = containers * (TIMEOUT + 60) * _per_second(data) * 1.15
        if bound > max_bound:
            raise RuntimeError(f"bound ${bound:.2f} exceeds this launch's limit ${max_bound:.2f}")
        from arc_infra.runops.records import row_cost
        committed = data["infrastructure_reserve_usd"] + sum(row_cost(x)[0] for x in data["runs"])
        if committed + bound > data["cap_usd"]:
            raise RuntimeError(f"Spend cap: ${committed:.2f} + ${bound:.2f} required")
        if any(x["run_id"] == run_id for x in data["runs"]):
            raise ValueError("Run already reserved; inspect it instead of relaunching")
        data["runs"].append({
            "run_id": run_id, "state": "reserved", "kind": "simulator", "charged_bound_usd": bound,
            "max_seconds": TIMEOUT, "gpu": None, "cpu_ceiling": CPU, "memory_ceiling_gib": MEMORY_MIB / 1024,
            "containers": containers, "authorization": AUTHORIZATION, "note": note,
        })
        file.seek(0)
        json.dump(data, file, indent=1)
        file.write("\n")
        file.truncate()
    return bound


def reconcile(run_id, container_seconds, wall_seconds, app_id):
    import fcntl

    path = ROOT / "docs/spend.json"
    with path.open("r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        data = json.load(file)
        per = _per_second(data)
        for x in data["runs"]:
            if x["run_id"] == run_id:
                estimate = sum(container_seconds) * per
                x.update(state="complete_pending_metering", app_id=app_id,
                         observed_container_seconds=round(sum(container_seconds), 1),
                         observed_wall_seconds=round(wall_seconds, 1), containers_used=len(container_seconds),
                         estimated_cost_usd=round(estimate, 4),
                         charged_bound_usd=round(max(estimate * 1.15, 0.01), 4))
        file.seek(0)
        json.dump(data, file, indent=1)
        file.write("\n")
        file.truncate()


@app.local_entrypoint()
def main(jobs: str, run_id: str, out_dir: str, containers: int = MAX_CONTAINERS, max_bound: float = 3.0):
    items = json.loads(Path(jobs).read_text())
    keys = [i["key"] for i in items]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate job keys")
    for item in items:
        item["cost"] = cost(item)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = batches(items, min(containers, max(1, (len(items) + CPU - 1) // CPU)))
    print(f"{len(items)} jobs, {len(groups)} containers x {CPU} CPU", flush=True)
    bound = reserve(run_id, len(groups), f"window replay: {len(items)} jobs ({Path(jobs).name})", max_bound)
    print(f"reserved ${bound:.3f} for {run_id}; app id {getattr(app, 'app_id', None)}", flush=True)
    t, seconds, done = time.time(), [], 0
    for batch in run_batch.map(groups, order_outputs=False):
        seconds.append(batch["seconds"])
        for key, result, dt in batch["results"]:
            (out / f"{key}.json").write_text(json.dumps(result, indent=1) + "\n")
            done += 1
        print(f"  batch: {len(batch['results'])} jobs in {batch['seconds']:.0f} s; {done}/{len(items)}", flush=True)
    wall = time.time() - t
    reconcile(run_id, seconds, wall, getattr(app, "app_id", None))
    print(f"wall {wall:.0f} s; container seconds {sum(seconds):.0f}", flush=True)
