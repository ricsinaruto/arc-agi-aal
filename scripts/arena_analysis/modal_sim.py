"""Modal CPU fan-out for the calibrated scheduler simulator ("run the simulator on
Modal"). NOT LAUNCHED AND UNTESTED: the 28 September studies ran on the local pool (streams-9h.md §8-9); a first
launch needs a small smoke (one stage, few runs) and the user's own go-ahead for the spend.

The local stages (``depthwise_sim.py``, ``streams_9h.py``) run one process per policy on this machine. This app runs
the same (config, policy) jobs on Modal CPU containers: every job is ``schedule_sim.run_one`` with the job's own seed,
so a job gives the same scores as the local pool. Containers take batches of jobs and run them in a fork pool of
``CPU`` processes; the entrypoint gathers the results and writes the same ``<name>.config.json`` / ``.result.json``
files as the local path, so the report stages are unchanged. No GPU.

Usage (from the worktree; the spend ledger is the main checkout's):
  ARC_VALIDATION_ROOT=. uv run --extra modal modal run scripts/arena_analysis/modal_sim.py \\
      --study depthwise --stages validate,compare --world w1 --run-id sim-depthwise-v1-modal-20260928 \\
      --attempts <attempts-now.jsonl>

Before the launch the entrypoint reserves the bound (containers x timeout x (CPU cores + memory) at the ledger's
rates x 1.15) in docs/spend.json under an exclusive lock; afterwards it records the containers' measured seconds.
"""

import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# In the container this module is /root/modal_sim.py: the repository paths exist only locally (29 September: the
# first launch crashed at import on ``HERE.parents[1]``).
LOCAL = modal.is_local()
REPO = HERE.parents[1] if LOCAL else HERE
ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or REPO).resolve()  # the ledger's checkout
CPU = 8
MEMORY_MIB = 6144
TIMEOUT = 900
MAX_CONTAINERS = 20
REMOTE = "/root/sim"

app = modal.App("arc-sim")  # the entrypoint module is mounted by Modal; the simulator sources are added below
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.5.3", "scipy==1.18.1")
if LOCAL:
    image = (
        image
        .add_local_dir(str(HERE), f"{REMOTE}/scripts", ignore=["__pycache__", "*.pyc"])
        .add_local_file(str(REPO / "src/arena/gittins.py"), f"{REMOTE}/pkg/arena/gittins.py")
        .add_local_file(str(REPO / "src/arena/hazard_index.py"), f"{REMOTE}/pkg/arena/hazard_index.py")
        .add_local_dir(str(REPO / "docs/validation-20260927/c110-path"), f"{REMOTE}/data/c110-path",
                       ignore=["*.result.json", "*.config.json"])
        .add_local_dir(str(REPO / "docs/validation-20260927/depthwise-sim"), f"{REMOTE}/data/depthwise-sim",
                       ignore=["*.result.json", "*.config.json", "summary.json"])
    )


def _argv_value(flag):
    """``--attempts <path>`` from the ``modal run`` command line (the image is defined at import, before the
    entrypoint parses its arguments)."""
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


_ATTEMPTS = os.environ.get("ARC_SIM_ATTEMPTS") or _argv_value("--attempts")
if LOCAL and _ATTEMPTS and Path(_ATTEMPTS).is_file():
    image = image.add_local_file(_ATTEMPTS, f"{REMOTE}/data/attempts.jsonl")


def _remote_path(path):
    """Local data paths in a config -> their place in the image."""
    p = str(path)
    for local, remote in ((REPO / "docs/validation-20260927/c110-path", f"{REMOTE}/data/c110-path"),
                          (REPO / "docs/validation-20260927/depthwise-sim", f"{REMOTE}/data/depthwise-sim")):
        if p.startswith(str(local)):
            return remote + p[len(str(local)):]
    raise ValueError(f"no image path for {p}")


_WORLDS = {}


def _job(item):
    import schedule_sim as sim

    key = item["world_key"]
    world = _WORLDS[key]
    world.fine_start = item["fine_start"]
    games = sim.ROSTERS[item["roster"]]
    t = time.time()
    res = sim.run_one(world, item["policy"], item["kw"], item["skw"], item["runs"], item["seed"], games, item["shared"])
    return item["name"], item["label"], res, time.time() - t


@app.function(image=image, cpu=float(CPU), memory=MEMORY_MIB, timeout=TIMEOUT, retries=0,
              max_containers=MAX_CONTAINERS, block_network=True)
def run_batch(items):
    import multiprocessing as mp

    sys.path[:0] = [f"{REMOTE}/scripts", f"{REMOTE}/pkg"]
    open(f"{REMOTE}/pkg/arena/__init__.py", "a").close()
    import schedule_sim as sim

    t0 = time.time()
    for item in items:
        key = item["world_key"]
        if key not in _WORLDS:
            attempts, world, extra, exclude = json.loads(key)
            _WORLDS[key] = sim.World.fit(sim.load_rows(attempts, tuple(extra), tuple(exclude)), shape=world)
    fit_seconds = time.time() - t0
    order = sorted(items, key=lambda x: -x["cost"])  # long jobs first
    with mp.get_context("fork").Pool(min(CPU, len(items))) as pool:
        out = list(pool.imap_unordered(_job, order))
    return {"results": out, "seconds": time.time() - t0, "fit_seconds": fit_seconds}


def flatten(configs):
    """(name, config) -> jobs. ``cost``: a relative run-time estimate for batching (runs x wall hours x slots)."""
    items = []
    for name, cfg in configs.items():
        extra = [_remote_path(p) for p in cfg.get("extra_attempts", ())]
        key = json.dumps([f"{REMOTE}/data/attempts.jsonl", cfg.get("world", "pooled"), extra,
                          list(cfg.get("exclude_runs", ()))])
        for label, spec in cfg["policies"].items():
            skw = {**cfg.get("sim", {}), **(spec[2] if len(spec) > 2 else {})}
            skw.setdefault("slots", 2)
            hours = skw.get("wall", 3600.0) / 3600.0
            index = spec[0] in ("index", "known-rate", "known-game", "oracle")
            cost = cfg.get("runs", 20) * hours * (4.0 if index else 1.0) * (1 + skw.get("slots", 6) / 12)
            items.append({"name": name, "label": label, "policy": spec[0], "kw": spec[1], "skw": skw,
                          "runs": cfg.get("runs", 20), "seed": cfg.get("seed", 0), "roster": str(cfg.get("roster", "23")),
                          "shared": cfg.get("shared", False), "fine_start": bool(cfg.get("fine_start", False)),
                          "world_key": key, "cost": cost})
    return items


def batches(items, n):
    """Greedy longest-first packing of jobs into ``n`` batches of about equal cost."""
    bins = [[] for _ in range(n)]
    load = [0.0] * n
    for item in sorted(items, key=lambda x: -x["cost"]):
        i = min(range(n), key=lambda j: load[j])
        bins[i].append(item)
        load[i] += item["cost"] / CPU
    return [b for b in bins if b]


def rate_per_container_second():
    data = json.loads((ROOT / "docs/spend.json").read_text())
    r = data["rates_per_second"]
    return CPU * r["physical_cpu_core"] + MEMORY_MIB / 1024 * r["memory_gib"]


AUTHORIZATION = ("run the simulator on Modal CPUs (no GPU); both simulator studies <= in "
                 "total")
AUTH_HERO_V2 = ("hero-v2 scheduler simulation on Modal CPUs (no GPU); smoke "
                "<=, each launch <= study total <=; Modal ")
AUTH_MTP3_KV4 = ("simulator on Modal CPUs (no GPU) for hero-v2 with phase 2 "
                 "restarted on MTP-3 and int4 KV at 38 x 64K; in ledger bounds; Modal ")


def reserve(run_id, containers, note, authorization=AUTHORIZATION, max_bound=None):
    """Append a reservation to docs/spend.json under an exclusive lock (as the other Modal launchers)."""
    import fcntl

    path = ROOT / "docs/spend.json"
    with path.open("r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        data = json.load(file)
        r = data["rates_per_second"]
        per = CPU * r["physical_cpu_core"] + MEMORY_MIB / 1024 * r["memory_gib"]
        bound = containers * (TIMEOUT + 60) * per * 1.15
        if max_bound is not None and bound > max_bound:
            raise RuntimeError(f"bound ${bound:.2f} exceeds this launch's limit ${max_bound:.2f}; use fewer containers")
        committed = data["infrastructure_reserve_usd"] + sum(x["charged_bound_usd"] for x in data["runs"])
        if committed + bound > data["cap_usd"]:
            raise RuntimeError(f"Spend cap: ${committed:.2f} + ${bound:.2f} required")
        if any(x["run_id"] == run_id for x in data["runs"]):
            raise ValueError("Run already reserved; inspect it instead of relaunching")
        data["runs"].append({
            "run_id": run_id, "state": "reserved", "kind": "simulator", "charged_bound_usd": bound,
            "max_seconds": TIMEOUT, "gpu": None, "cpu_ceiling": CPU, "memory_ceiling_gib": MEMORY_MIB / 1024,
            "containers": containers,
            "authorization": authorization,
            "note": note,
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
        r = data["rates_per_second"]
        per = CPU * r["physical_cpu_core"] + MEMORY_MIB / 1024 * r["memory_gib"]
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
def main(study: str, stages: str, run_id: str, attempts: str, world: str = "w1", runs: int = 0,
         containers: int = MAX_CONTAINERS, out_dir: str = "", max_bound: float = 3.0):
    sys.path.insert(0, str(HERE))
    if not (_ATTEMPTS and Path(_ATTEMPTS).is_file()):
        raise SystemExit("pass --attempts <file> (or set ARC_SIM_ATTEMPTS): it is baked into the image")
    authorization = AUTHORIZATION
    import c110_path as cp

    share = cp.opening_share()
    if study == "depthwise":
        import depthwise_sim as mod
        configs = {}
        for st in stages.split(","):
            configs.update(mod.with_world(mod.STAGES[st](runs, share), world))
        writer = mod.write
        out = Path(out_dir or mod.OUT)
    elif study == "streams":
        import streams_9h as mod
        stage_fns = {"kv8_nine": mod.kv8_nine, "kv8_hour": mod.kv8_hour, "kv8_robust": mod.kv8_robust,
                     "nine": mod.nine, "hour": mod.hour}
        configs = {}
        for st in stages.split(","):
            configs.update(stage_fns[st](runs, share))

        def writer(name, config, res, out_dir):
            summary = {label: {**cp.summarise(r), "scores": r["scores"]} for label, r in res.items()}
            Path(out_dir, f"{name}.config.json").write_text(json.dumps(config, indent=1))
            Path(out_dir, f"{name}.result.json").write_text(json.dumps(summary, indent=1))
        out = Path(out_dir or mod.OUT)
    elif study == "hero_v2":
        import sim_hero_v2 as mod
        configs = {}
        for st in stages.split(","):
            configs.update(mod.STAGES[st](runs, share))
        writer = mod.write
        out = Path(out_dir or mod.OUT)
        authorization = AUTH_HERO_V2
    elif study == "mtp3_kv4":
        import sim_mtp3_kv4 as mod
        configs = {}
        for st in stages.split(","):
            configs.update(mod.STAGES[st](runs, share))
        writer = __import__("sim_hero_v2").write
        out = Path(out_dir or mod.OUT)
        authorization = AUTH_MTP3_KV4
    else:
        raise SystemExit(f"unknown study {study}")
    out.mkdir(parents=True, exist_ok=True)
    items = flatten(configs)
    groups = batches(items, min(containers, max(1, (len(items) + CPU - 1) // CPU)))
    print(f"{len(configs)} configs, {len(items)} jobs, {len(groups)} containers x {CPU} CPU", flush=True)
    bound = reserve(run_id, len(groups), f"{study}: {stages} ({world}), {len(items)} jobs", authorization, max_bound)
    print(f"reserved ${bound:.3f} for {run_id}", flush=True)
    app_id = getattr(app, "app_id", None)
    print(f"app id {app_id}", flush=True)
    t = time.time()
    results, seconds = {}, []
    for batch in run_batch.map(groups, order_outputs=False):
        seconds.append(batch["seconds"])
        for name, label, res, dt in batch["results"]:
            results.setdefault(name, {})[label] = res
        print(f"  batch done: {len(batch['results'])} jobs in {batch['seconds']:.0f} s "
              f"(fit {batch['fit_seconds']:.0f} s); {sum(len(v) for v in results.values())}/{len(items)}", flush=True)
    wall = time.time() - t
    for name, cfg in configs.items():
        res = {label: results[name][label] for label in cfg["policies"]}
        print(f"== {name}", flush=True)
        writer(name, cfg, res, out)
    reconcile(run_id, seconds, wall, app_id)
    per = rate_per_container_second()
    print(f"wall {wall:.0f} s; container seconds {sum(seconds):.0f}; estimated ${sum(seconds) * per:.3f}", flush=True)
