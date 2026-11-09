"""dfranzen's public Kaggle notebook, unchanged, on Modal for one hour on the 110-clone roster.

User: "can we run dfranzen's exact notebook on modal for 1 hour to see how throughput compares please". The notebook is
"ARC-AGI-3 Milestone 2 Solution" (https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution, public score
27.89), pulled 1 October: ``scripts/dfr_exact/arc-agi-3-milestone-2-solution.ipynb`` (sha256 7b76c194...).

Everything his notebook reads lives in the Modal Volume ``arc-dfr-exact`` in the Kaggle layout and is mounted at
``/kaggle/input``, so every path in the notebook is unchanged:

- ``datasets/dfranzen/pennyroyal-v253`` (his SGLang wheelhouse, from Kaggle through the authorized wrapper),
- ``datasets/dfranzen/taaf-kaggle-source-bundle-copy`` (the Tufa Duck source bundle, from Kaggle),
- ``models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1`` (Intel's AutoRound W4A16 from
  Hugging Face, with his 22-shard PLE reshard rebuilt from his ``ple_reshard_manifest.json`` and checked against its
  sha256s; small files from Kaggle),
- ``models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1`` (from Kaggle; vocab.json from HF),
- ``competitions/arc-prize-2026-arc-agi-3`` (arc_agi wheels and public environment files, from Kaggle).

``docs/validation-20261004/dfranzen-exact/inputs.json`` lists every file with its Kaggle size and sha256.

Functions (no Kaggle credentials ever reach Modal):

    uv run --extra modal modal run --detach scripts/modal_dfranzen_exact.py::stage_main     # CPU: HF fetch + reshard
    uv run --extra modal modal run scripts/modal_dfranzen_exact.py::selftest_main            # CPU: image/gateway/cells
    ARC_VALIDATION_ROOT=. uv run --extra modal modal run --detach \
        scripts/modal_dfranzen_exact.py::main --run-id dfranzen-exact-v1-modal-20261004     # the GPU hour

The GPU run starts the local competition gateway (110 cyclic clones, port 8002), executes the notebook in an IPython
kernel with ``KAGGLE_IS_COMPETITION_RERUN=1`` (his competition branch: 10 priority streams over all games) and
``ARC_BASE_URL`` pointing at that gateway, and changes one line of the notebook: ``max_runtime_s_per_game = 532*60`` ->
``60*60`` (the one-hour wall; his own 25-minute demo branch scales the same knob). Monitoring sidecars (never read by
the notebook): the gateway's scorecard snapshots and an SGLang ``/metrics`` scrape every 15 s.
"""

import json
import os
import time
from pathlib import Path

import modal

ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or Path(__file__).resolve().parents[1]).resolve()
REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "scripts/dfr_exact"
RECORDS = "docs/validation-20261004"
INPUTS_JSON = REPO / RECORDS / "dfranzen-exact/inputs.json"
NOTEBOOK = "arc-agi-3-milestone-2-solution.ipynb"
NOTEBOOK_SHA256 = "7b76c194b478faa0309b01e5fba05840f1314da7f288e05e5f4b09972e4f9b4c"
WALL_LINE_OLD = "        bm.solver.max_runtime_s_per_game = 532*60\n"
WALL_LINE_NEW = " bm.solver.max_runtime_s_per_game = 60*60  # Modal plumbing: one-hour wall, was 532*60\n"
TIMEOUT = 7800  # like competition-preview-1h: setup + the 3,600 s wall + teardown
CPU = 16
MEMORY_MIB = 181248  # Kaggle's RTX PRO 6000 machine reported 176.9 GiB RAM
GATEWAY_PORT = 8002  # his SGLang takes 127.0.0.1:8001; Kaggle's gateway is a separate host (gateway:8001)
# The 110-clone roster of our dfr-combined previews: k000..k109 = these 20 games cyclically, in this order.
ROSTER = ["tn36-ef4dde99", "lf52-271a04aa", "cn04-2fe56bfb", "bp35-0a0ad940", "wa30-ee6fef47", "r11l-495a7899",
          "sp80-589a99af", "m0r0-492f87ba", "ka59-38d34dbb", "sc25-635fd71a", "sk48-d8078629", "dc22-fdcac232",
          "cd82-fb555c5d", "g50t-5849a774", "ls20-9607627b", "re86-8af5384d", "s5i5-18d95033", "sb26-7fbdac44",
          "su15-1944f8ab", "tr87-cd924810"]
HF_INTEL = ("Intel/Qwen3.8-Flash-Next-W4A16-AutoRound", "4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897")
HF_ALBUCINO = ("albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE", "c6af9fee7ad2cb8ef44b534a99b8c1b81fb0f6da")

app = modal.App("arc-dfr-exact", include_source=False)
inputs = modal.Volume.from_name("arc-dfr-exact", version=2)
runs = modal.Volume.from_name("arc-prolong-runs")

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "build-essential", "ninja-build", "libnuma1", "libgomp1", "libglib2.0-0", "libgl1",
                      "procps", "curl")
         # The Kaggle Python image's scientific stack that the Duck harness imports, plus the notebook runner.
         # IPython is pinned below 9.5: from 9.5 on %%writefile strips whitespace-only lines, which corrupts the
         # harness patch of his cell 2 (git apply fails on prompts.py); Kaggle's kernel writes it intact.
         .pip_install("numpy", "scipy", "matplotlib", "pillow", "imageio", "requests", "pandas", "pyarrow",
                      "python-dotenv", "ipython==8.37.0", "ipykernel==6.29.5", "nbclient", "nbformat")
         .add_local_dir(str(HERE), "/opt/dfr-exact", copy=True)
         .add_local_file(str(INPUTS_JSON), "/opt/dfr-exact/inputs.json", copy=True)
         .add_local_file(__file__, "/root/modal_dfranzen_exact.py", copy=True))
stage_image = (modal.Image.debian_slim(python_version="3.12")
               .pip_install("huggingface_hub>=0.34", "hf_xet")
               .env({"HF_XET_HIGH_PERFORMANCE": "1"})
               .add_local_file(str(INPUTS_JSON), "/opt/inputs.json", copy=True)
               .add_local_file(__file__, "/root/modal_dfranzen_exact.py", copy=True))


def _rate(cpu, memory_gib=192):
    return 0.000842 + cpu * 0.0000131 + memory_gib * 0.00000222


def reserve(run_id, seconds, *, cpu, gpu=True, kind="arena", authorization):
    """The ledger entry (docs/spend.json in the main checkout), as scripts/modal_validation.py reserve()."""
    import fcntl

    path = ROOT / "docs/spend.json"
    with path.open("r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        data = json.load(file)
        rate = _rate(cpu) if gpu else cpu * 0.0000131 + 32 * 0.00000222
        bound = (seconds + 60) * rate * 1.15
        committed = data["infrastructure_reserve_usd"] + sum(r["charged_bound_usd"] for r in data["runs"])
        if committed + bound > data["cap_usd"]:
            raise RuntimeError(f"Spend cap: ${committed:.2f} + ${bound:.2f} required")
        if any(r["run_id"] == run_id for r in data["runs"]):
            raise ValueError("Run already reserved; inspect saved call instead of relaunching")
        data["runs"].append({
            "run_id": run_id, "state": "reserved", "kind": kind, "charged_bound_usd": bound, "max_seconds": seconds,
            "gpu": "RTX-PRO-6000" if gpu else None, "cpu_ceiling": cpu, "memory_ceiling_gib": 192 if gpu else 32,
            "authorization": authorization,
        })
        file.seek(0)
        json.dump(data, file, indent=1)
        file.write("\n")
        file.truncate()
    return bound


# ---------------------------------------------------------------------------------------------------- staging (CPU)

@app.function(image=stage_image, cpu=8, memory=(2048, 32768), timeout=4 * 3600, retries=0, max_containers=1,
              volumes={"/vol": inputs})
def stage():
    """Fetch the Hugging Face weights into the Kaggle layout, rebuild the PLE reshard, unzip the wheelhouse, verify."""
    import hashlib
    import shutil
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download

    manifest = json.loads(Path("/opt/inputs.json").read_text())
    vol = Path("/vol")
    target = vol / manifest["paths"]["target"]
    draft = vol / manifest["paths"]["draft"]
    staging = vol / "_staging"
    report = {"started": time.time(), "steps": []}

    def log(**row):
        row["t"] = round(time.time() - report["started"], 1)
        report["steps"].append(row)
        print(json.dumps(row), flush=True)

    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                block = f.read(64 << 20)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()

    # 1. Hugging Face downloads (only files not already present with the right size).
    jobs = []
    for name, entry in manifest["target"].items():
        if entry["source"] == "hf":
            jobs.append((HF_INTEL, name, target, entry["size"]))
    jobs.append((HF_INTEL, "model-00016-of-00017.safetensors", staging / "intel", manifest["hf_source_ple"]["size"]))
    for name, entry in manifest["draft"].items():
        if entry["source"] == "hf":
            jobs.append((HF_ALBUCINO, f"runtime/mtp-int4-g32/{name}", staging / "albucino", entry["size"]))

    def fetch(job):
        (repo, revision), name, folder, size = job
        dest = folder / name
        if dest.is_file() and dest.stat().st_size == size:
            return name, "present"
        folder.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        hf_hub_download(repo, name, revision=revision, local_dir=str(folder))
        if dest.stat().st_size != size:
            raise RuntimeError(f"{name}: {dest.stat().st_size} bytes, expected {size}")
        return name, f"{time.time() - t0:.0f}s"

    with ThreadPoolExecutor(4) as pool:
        for name, how in pool.map(fetch, jobs):
            log(step="download", file=name, result=how)
    shutil.rmtree(target / ".cache", ignore_errors=True)
    for name, entry in manifest["draft"].items():
        if entry["source"] == "hf":
            src = staging / "albucino/runtime/mtp-int4-g32" / name
            if src.exists():
                shutil.move(str(src), str(draft / name))
    inputs.commit()

    # 2. Hashes of the linked weights and the PLE source against the HF LFS oids (the same-size Kaggle files).
    checks = [(target / n, e["sha256"]) for n, e in manifest["target"].items()
              if e["source"] in ("hf", "kaggle") and e.get("sha256") and n.endswith(".safetensors")]
    checks += [(draft / n, e["sha256"]) for n, e in manifest["draft"].items() if e.get("sha256")]
    source = staging / "intel/model-00016-of-00017.safetensors"
    checks.append((source, manifest["hf_source_ple"]["sha256"]))
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda c: (c[0], c[1], sha256(c[0])), checks))
    bad = [str(p) for p, want, got in results if want != got]
    log(step="hash_inputs", files=len(results), mismatches=bad)
    if bad:
        raise RuntimeError(f"hash mismatch: {bad}")

    # 3. The PLE reshard: safetensors serialization of each manifest shard, byte ranges copied from the source.
    ple_manifest = json.loads((target / "ple_reshard_manifest.json").read_text())
    with open(source, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    base = 8 + n
    meta = header.pop("__metadata__", None)

    def serialize(names, metadata, order):
        names = sorted(names) if order == "sorted" else list(names)
        offset, body = 0, {}
        if metadata is not None:
            body["__metadata__"] = metadata
        spans = []
        for name in names:
            t = header[name]
            size = t["data_offsets"][1] - t["data_offsets"][0]
            body[name] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [offset, offset + size]}
            spans.append((base + t["data_offsets"][0], size))
            offset += size
        text = json.dumps(body, separators=(",", ":")).encode()
        text += b" " * ((8 - len(text) % 8) % 8)
        return len(text).to_bytes(8, "little") + text, spans, 8 + len(text) + offset

    def write_shard(shard):
        dest = target / shard["file"]
        if dest.is_file() and dest.stat().st_size == shard["size_bytes"] and sha256(dest) == shard["sha256"]:
            return shard["file"], "present", None
        tried = []
        for metadata in (meta, {"format": "pt"}, None):
            for order in ("sorted", "manifest"):
                head, spans, total = serialize(shard["tensors"], metadata, order)
                if total != shard["size_bytes"]:
                    tried.append((metadata, order, "size"))
                    continue
                h = hashlib.sha256(head)
                with open(source, "rb") as src, open(dest, "wb") as out:
                    out.write(head)
                    for start, size in spans:
                        src.seek(start)
                        left = size
                        while left:
                            block = src.read(min(left, 64 << 20))
                            out.write(block)
                            h.update(block)
                            left -= len(block)
                if h.hexdigest() == shard["sha256"]:
                    return shard["file"], "written", {"metadata": metadata, "order": order}
                tried.append((metadata, order, "sha"))
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{shard['file']}: no serialization matched {tried}")

    with ThreadPoolExecutor(4) as pool:
        for name, how, variant in pool.map(write_shard, ple_manifest["shards"]):
            log(step="reshard", file=name, result=how, variant=variant)
    inputs.commit()

    # 4. The wheelhouse (Kaggle's own archive of dfranzen/pennyroyal-v253; zipfile checks every CRC).
    wheel_dir = vol / manifest["paths"]["wheelhouse"]
    archive = staging / "pennyroyal-v253.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(wheel_dir)
        log(step="unzip_wheelhouse", files=len(manifest["wheelhouse_zip"]))

    # 5. Every input against the manifest (sizes; no extra files in the model folders).
    problems = []
    for key, folder in (("target", target), ("draft", draft), ("bundle", vol / manifest["paths"]["bundle"]),
                        ("competition", vol / manifest["paths"]["competition"]),
                        ("wheelhouse_zip", wheel_dir)):
        for name, entry in manifest[key].items():
            p = folder / name
            if not p.is_file() or p.stat().st_size != entry["size"]:
                problems.append(f"{key}:{name}")
    for key, folder in (("target", target), ("draft", draft)):
        extra = sorted(str(p.relative_to(folder)) for p in folder.rglob("*")
                       if p.is_file() and str(p.relative_to(folder)) not in manifest[key])
        if extra:
            problems.append(f"{key} extra files: {extra}")
    log(step="verify", problems=problems)
    if problems:
        raise RuntimeError(f"staging problems: {problems}")
    # The HF PLE source and the archive are no longer needed (the reshard is the Kaggle layout).
    shutil.rmtree(staging, ignore_errors=True)
    report["finished"] = time.time()
    (vol / "STAGED.json").write_text(json.dumps(report, indent=1))
    inputs.commit()
    return report


@app.local_entrypoint()
def stage_main(run_id: str = "dfranzen-exact-stage-v1-modal-20261004"):
    bound = reserve(run_id, 4 * 3600, cpu=8, gpu=False, kind="cpu-staging",
                    authorization="dfranzen's exact notebook on Modal for 1 hour (plus any Modal CPU "
                                  "staging jobs it needs): HF weight fetch + PLE reshard + wheelhouse unzip")
    record = ROOT / RECORDS / f"modal-{run_id}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    data = {"state": "submission_intent", "run_id": run_id, "kind": "cpu-staging", "charged_bound_usd": bound}
    record.write_text(json.dumps(data, indent=2))
    call = stage.spawn()
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))


# ------------------------------------------------------------------------------------------------- helpers (remote)

def _verify_inputs():
    manifest = json.loads(Path("/opt/dfr-exact/inputs.json").read_text())
    vol = Path("/kaggle/input")
    problems = []
    for key, rel in (("target", "target"), ("draft", "draft"), ("bundle", "bundle"),
                     ("competition", "competition"), ("wheelhouse_zip", "wheelhouse")):
        folder = vol / manifest["paths"][rel]
        for name, entry in manifest[key].items():
            p = folder / name
            if not p.is_file() or p.stat().st_size != entry["size"]:
                problems.append(f"{key}:{name}")
    return problems


def _gateway(out, log):
    import subprocess
    import sys
    import urllib.request

    venv = Path("/tmp/gateway-venv")
    wheels = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    subprocess.run([str(venv / "bin/pip"), "install", "-q", "--no-index", "--find-links", wheels, "arc-agi"],
                   check=True)
    proc = subprocess.Popen(
        [str(venv / "bin/python"), "-u", "/opt/dfr-exact/gateway.py", "--environments-dir",
         "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files", "--games", json.dumps(ROSTER),
         "--total", "110", "--port", str(GATEWAY_PORT), "--out", str(out), "--interval", "30"],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        env={**os.environ, "RECORDINGS_DIR": "/tmp/gateway-recordings"})
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{GATEWAY_PORT}/api/games", timeout=2) as r:
                if r.status == 200 and len(json.loads(r.read())) == 110:
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError("gateway exited")
        time.sleep(1)
    raise RuntimeError("gateway not ready")


def _notebook(path):
    import hashlib

    raw = Path("/opt/dfr-exact", NOTEBOOK).read_bytes()
    if hashlib.sha256(raw).hexdigest() != NOTEBOOK_SHA256:
        raise RuntimeError("notebook hash mismatch")
    nb = json.loads(raw)
    for c in nb["cells"]:  # nbformat allows a string or a list of lines
        if isinstance(c["source"], str):
            c["source"] = c["source"].splitlines(True)
    hits = [(i, j) for i, c in enumerate(nb["cells"]) for j, line in enumerate(c["source"]) if line == WALL_LINE_OLD]
    if len(hits) != 1:
        raise RuntimeError(f"expected one wall line, found {hits}")
    i, j = hits[0]
    nb["cells"][i]["source"][j] = WALL_LINE_NEW
    Path(path).write_text(json.dumps(nb, indent=1))
    return {"cell": i, "line": j, "old": WALL_LINE_OLD.strip(), "new": WALL_LINE_NEW.strip()}


def _kernel_env():
    env = dict(os.environ)
    env.update(KAGGLE_IS_COMPETITION_RERUN="1", ARC_BASE_URL=f"http://127.0.0.1:{GATEWAY_PORT}/",
               PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    return env


# -------------------------------------------------------------------------------------------------- self-test (CPU)

@app.function(image=image, cpu=4, memory=(1024, 16384), timeout=1800, retries=0, max_containers=1,
              block_network=True, volumes={"/kaggle/input": inputs})
def selftest():
    """No GPU: inputs present, the gateway serves 110 clones and plays, and the notebook's CPU cells run (patch,
    pip install of arc-agi, source path, unpickling the benchmark, the solver overrides in the competition branch)."""
    import subprocess

    result = {"input_problems": _verify_inputs()}
    out = Path("/tmp/selftest")
    out.mkdir(parents=True, exist_ok=True)
    Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
    with (out / "gateway.log").open("wb") as log:
        gw = _gateway(out / "gateway", log)
    result["notebook_change"] = _notebook(out / "nb.ipynb")
    extra = (
        "import arc_agi, taaf.game_api\n"
        "spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.COMPETITION, "
        "arc_base_url=os.environ['ARC_BASE_URL'], environments_dir='')\n"
        "arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.COMPETITION, "
        "arc_base_url=spec.arc_base_url, environments_dir='')\n"
        "ids = [e.game_id for e in arcade.available_environments]\n"
        "print('GATEWAY_GAMES', len(ids), ids[:3])\n"
        "print('SOLVER', bm.solver.concurrency, bm.solver.max_runtime_s_per_game, bm.solver.save_request_logs, "
        "TRUE_SUBMISSION, os.environ.get('ARC3_MAX_ACTIVE_STREAMS'))\n"
        "import importlib; [importlib.import_module(m) for m in ('inference.framework.solver', "
        "'inference.agent.tool_agent', 'inference.agent.priority_scheduler', 'inference.utils.frame_diff', "
        "'inference.utils.animation')]\n"
        "print('IMPORTS_OK')\n"
        "card = arcade.open_scorecard()\n"
        "env = arcade.make('k000', scorecard_id=card)\n"
        "print('MAKE_OK', type(env).__name__)\n")
    proc = subprocess.run(["python", "/opt/dfr-exact/run_notebook.py", str(out / "nb.ipynb"),
                           str(out / "nb-out.ipynb"), "--cells", "2,4,8,10,14,16", "--extra", extra],
                          env=_kernel_env(), cwd="/kaggle/working", capture_output=True, text=True, timeout=1200)
    result["notebook_exit"] = proc.returncode
    result["notebook_tail"] = proc.stdout[-6000:]
    time.sleep(35)
    gw.terminate()
    result["gateway_log"] = (out / "gateway.log").read_text()[-3000:]
    snaps = (out / "gateway/scorecards.jsonl")
    rows = snaps.read_text().splitlines() if snaps.exists() else []
    result["scorecard_rows"] = [r[:600] for r in rows[-2:]]
    result["git"] = subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip()
    result["gxx"] = subprocess.run(["bash", "-lc", "ls /usr/bin/g++* /usr/bin/gcc*"], capture_output=True,
                                   text=True).stdout.split()
    return result


@app.local_entrypoint()
def selftest_main():
    print(json.dumps(selftest.remote(), indent=1))


# ------------------------------------------------------------------------------------------------- the GPU hour

@app.function(image=image, gpu="RTX-PRO-6000", cpu=(0.125, CPU), memory=(1024, MEMORY_MIB), timeout=TIMEOUT,
              retries=0, max_containers=1, scaledown_window=2, block_network=True,
              volumes={"/kaggle/input": inputs, "/runs": runs})
def run(run_id: str):
    import re
    import shutil
    import signal
    import subprocess
    import threading
    import urllib.request

    started = time.time()
    out = Path("/runs/validation") / run_id
    if out.exists():
        out.rename(out.with_name(f"{run_id}-lost-{int(started)}"))
    out.mkdir(parents=True)
    identity = {"run_id": run_id, "started_epoch": started, "gpu": "RTX-PRO-6000", "cpu_ceiling": CPU,
                "memory_ceiling_mib": MEMORY_MIB, "function_timeout": TIMEOUT, "image": "debian-slim/python3.12"}
    stop = threading.Event()
    procs = {}
    try:
        problems = _verify_inputs()
        identity["input_problems"] = problems
        if problems:
            raise RuntimeError(f"inputs: {problems[:10]}")
        Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
        gw_log = (out / "gateway.log").open("wb")
        procs["gateway"] = _gateway(out / "gateway", gw_log)
        identity["gateway_ready_seconds"] = round(time.time() - started, 1)
        identity["notebook_change"] = _notebook(out / "notebook-input.ipynb")

        def scrape():
            # SGLang's Prometheus endpoint (--enable-metrics): every non-bucket sample, every 15 s.
            with (out / "metrics.jsonl").open("a") as f:
                while not stop.wait(15):
                    try:
                        with urllib.request.urlopen("http://127.0.0.1:8001/metrics", timeout=5) as r:
                            text = r.read().decode()
                    except Exception:
                        continue
                    sample = {}
                    for line in text.splitlines():
                        if line.startswith("#") or "_bucket{" in line:
                            continue
                        key, _, value = line.rpartition(" ")
                        try:
                            sample[key] = float(value)
                        except ValueError:
                            pass
                    f.write(json.dumps({"epoch": time.time(), "m": sample}) + "\n")
                    f.flush()

        def committer():
            while not stop.wait(120):
                try:
                    serve = Path("/kaggle/working/serve.log")
                    if serve.exists():
                        shutil.copyfile(serve, out / "serve.log")
                    runs.commit()
                except Exception as exc:
                    print(f"interim commit failed: {exc}", flush=True)

        def progress():
            # One line a minute: the gateway's latest score and the engine's generated tok/s.
            last = None
            while not stop.wait(60):
                try:
                    rows = (out / "gateway/scorecards.jsonl").read_text().splitlines()
                    snap = json.loads(rows[-1]) if rows else {}
                    m = (out / "metrics.jsonl").read_text().splitlines()
                    tps = None
                    if len(m) >= 5:
                        a, b = json.loads(m[-5]), json.loads(m[-1])
                        key = next((k for k in b["m"] if k.startswith("sglang:generation_tokens_total")), None)
                        if key and key in a["m"]:
                            tps = round((b["m"][key] - a["m"][key]) / (b["epoch"] - a["epoch"]), 1)
                    line = {"event": "progress", "elapsed": round(time.time() - started), "score_pct":
                            round(snap["score_mean_110"], 3) if "score_mean_110" in snap else None,
                            "levels": snap.get("levels"), "actions": snap.get("actions"),
                            "games_played": snap.get("games_played"), "gen_tok_s_60s": tps}
                    if line != last:
                        print(json.dumps(line), flush=True)
                    last = line
                except Exception:
                    pass

        threads = [threading.Thread(target=f, daemon=True) for f in (scrape, committer, progress)]
        for t in threads:
            t.start()
        identity["notebook_start_seconds"] = round(time.time() - started, 1)
        nb_log = (out / "notebook.log").open("wb")
        procs["notebook"] = subprocess.Popen(
            ["python", "-u", "/opt/dfr-exact/run_notebook.py", str(out / "notebook-input.ipynb"),
             str(out / "notebook-executed.ipynb")],
            env=_kernel_env(), cwd="/kaggle/working", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True)

        def pump():
            for line in iter(procs["notebook"].stdout.readline, b""):
                nb_log.write(line)
                nb_log.flush()
                print(line.decode(errors="replace").rstrip(), flush=True)

        pumper = threading.Thread(target=pump, daemon=True)
        pumper.start()
        try:
            procs["notebook"].wait(timeout=max(60, started + TIMEOUT - 300 - time.time()))
        except subprocess.TimeoutExpired:
            identity["hard_stop"] = "notebook still running 300 s before the function timeout"
            os.killpg(procs["notebook"].pid, signal.SIGTERM)
            procs["notebook"].wait(timeout=60)
        identity["notebook_exit"] = procs["notebook"].returncode
        identity["notebook_end_seconds"] = round(time.time() - started, 1)
        pumper.join(timeout=10)
        time.sleep(35)  # one more gateway snapshot after the close
    except BaseException as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop.set()
        for name, p in procs.items():
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:
                    pass
        subprocess.run(["pkill", "-9", "-f", "sglang"], check=False)
        working = Path("/kaggle/working")
        if working.exists():
            shutil.copytree(working, out / "working", dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.pkl"))
        ready = None
        try:
            text = (out / "notebook.log").read_text(errors="replace")
            m = re.search(r"READY after (\d+)s", text)
            ready = int(m.group(1)) if m else None
        except Exception:
            pass
        identity["server_ready_after_launch_seconds"] = ready
        identity["elapsed_seconds"] = round(time.time() - started, 1)
        (out / "execution.json").write_text(json.dumps(identity, indent=2))
        runs.commit()
    return str(out)


@app.local_entrypoint()
def main(run_id: str = ""):
    if not run_id:
        raise ValueError("--run-id is required")
    bound = reserve(run_id, TIMEOUT, cpu=CPU,
                    authorization="\"can we run dfranzen's exact notebook on modal for 1 hour to see "
                                  "how throughput compares please\" (one Modal GPU run, plus CPU staging)")
    record = ROOT / RECORDS / f"modal-{run_id}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    data = {"state": "submission_intent", "run_id": run_id, "kind": "dfranzen-exact", "test": "notebook-1h",
            "charged_bound_usd": bound, "platform_policy": "Modal only",
            "notebook": f"scripts/dfr_exact/{NOTEBOOK}", "notebook_sha256": NOTEBOOK_SHA256,
            "volume": "arc-dfr-exact", "gateway_port": GATEWAY_PORT, "roster": ROSTER, "cpu": CPU,
            "memory_mib": MEMORY_MIB, "timeout": TIMEOUT}
    record.write_text(json.dumps(data, indent=2))
    call = run.spawn(run_id)
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))
