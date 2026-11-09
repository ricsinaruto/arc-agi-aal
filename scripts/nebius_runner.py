"""VM-side half of scripts/nebius_validation.py (stdlib only; root on the Nebius VM, started by the rendered bootstrap).

The counterpart of ``modal_validation.benchmark``: the same paths (/opt/arc-validation, /assets, /kaggle/input,
/runs/validation/<run_id>), the same bootstrap command and environment, the same stop-request and stop-rule watchers.
Differences: assets are fetched from the bucket and checked against their manifests, the output folder is copied to
the bucket every ``sync_seconds`` instead of a volume commit, the bootstrap runs in a network namespace with loopback
only (Modal's block_network), and SIGTERM (the 60 s preemption notice) writes an ``interrupted`` final marker.
With ``spec["checkpoint"]`` the trial checkpoints (src/arena/checkpoint.py): complete generations go to the bucket
every sync, the notice asks for one more, and a relaunch (``attempt`` > 1) restores the trial folders before the
bootstrap so it resumes.
This process never deletes the VM: its systemd unit's ExecStopPost and the deadline timer do.
"""

import fnmatch
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REMOTE = "arc"  # rclone remote from RCLONE_CONFIG_ARC_* (env_auth: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
ROOT = Path(os.environ.get("ARC_NEBIUS_ROOT", "/"))  # "/" on the VM (Modal's paths); a temp folder in the tests
SOURCE = ROOT / "opt/arc-validation"
ASSETS = ROOT / "assets"
RUNS = ROOT / "runs/validation"
KAGGLE = ROOT / "kaggle"
STATE = ROOT / "opt/arc"  # launch files and manifests, placed by the bootstrap script
WORKING = ROOT / "tmp/arena"
INPUTS = ROOT / "opt/arc-inputs"  # cpujob inputs (spec ``inputs``)
BAKED = STATE / "baked-assets.json"  # a bake's record of the assets on its disk (spec mode "bake" / "use_baked")
BAKE_KEEP = ROOT / "run/arc/bake-keep"  # tmpfs: arc-self-delete powers off instead of deleting (bake only)
STOP_FILE = "stop-request"
STOP_LOCAL = str(ROOT / "tmp/arena-stop-request")
STOP_POLL_SECONDS = 30
CONTROL_FILE = "control.json"  # live knobs (arc2 wall_seconds): bucket arena/live/control.json -> $ARENA_CONTROL
CONTROL_LOCAL = str(ROOT / "tmp/arena-control.json")
KNOBS_FILE = "knobs.json"  # live admission knobs (arc3sub 30 Oct: active_limit): arena/live/knobs.json -> $ARENA_KNOBS
KNOBS_LOCAL = str(ROOT / "tmp/arena-knobs.json")
SECRET_ENV_PREFIXES = ("AWS_", "RCLONE_", "NEBIUS_")
INTERRUPT_BUDGET = 45  # of the 60 s between SIGTERM and the stop
CHECKPOINT_WAIT = 14  # of them: for the checkpoint asked for at the notice (the trial holds replies for <= 8 s)
IMDS = "http://metadata.nebius.internal/v1"


ISOLATE = ("ip link set lo up && { ip link add arc0 type dummy && ip addr add <ip>/30 dev arc0 "
           "&& ip link set arc0 up && ip route add default dev arc0; } 2>/dev/null; exec \"$@\"")


LATE_POLL_SECONDS = 20  # spec asset_late: how often the bucket is asked for the late asset's manifest
SIGNALS_PREFIX = "assets-signals"  # <name>.failed.json there: the export of a late asset failed (the wait ends at once)
LATE_WATCH_SECONDS = 3600  # after a late asset is fetched, its failed signal is still watched this long (SFT byte checks)


class Interrupted(BaseException):
    pass


def asset_slug(prefix):
    return prefix.strip("/").replace("/", "--")


def job_environment(base, spec, output):
    """The Modal container's environment for the bootstrap, without this VM's credentials."""
    env = {k: v for k, v in base.items() if not k.startswith(SECRET_ENV_PREFIXES)}
    env.update(PYTHONPATH=str(SOURCE / "src"), PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PIP_NO_INDEX="1", MPLBACKEND="Agg", ONLY_RESET_LEVELS="true", PYTHONDONTWRITEBYTECODE="1",
               ARC_VALIDATION_KIND=spec["kind"], ARC_VALIDATION_TEST=spec["test"],
               ARC_VALIDATION_OUTPUT=str(output), ARC_FORCE_CHROOT="1", ARENA_STOP_REQUEST=STOP_LOCAL,
               ARENA_CONTROL=CONTROL_LOCAL, ARENA_KNOBS=KNOBS_LOCAL)
    env["LIBRARY_PATH"] = ":".join(["/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu",
                                    *filter(None, [base.get("LIBRARY_PATH")])])
    if spec.get("inputs"):
        env["ARC_INPUTS"] = str(INPUTS)
    checkpoint = spec.get("checkpoint")
    if checkpoint:
        env.update(ARC_CHECKPOINT_SECONDS=str(checkpoint["seconds"]), ARC_CHECKPOINT_HOLD="8",
                   ARC_CHECKPOINT_ATTEMPT=str(spec.get("attempt", 1)))
        if spec.get("attempt", 1) > 1:
            env["ARC_CHECKPOINT_RESUME"] = "1"
    return env


IMAGE_CONTAINER = "arc-job"


def image_command(spec, output):
    """Backend ``image`` (24 Oct, dots3 / mmrtx: serving stacks this image cannot hold): the bundle's entry script in
    a digest-pinned docker image on the VM's GPU, host network (pip, model pulls, a loopback server). /w = the fetched
    assets (read-only), /out = <output>/out (synced to the bucket with the rest), /opt/arc = the bundle (read-only)."""
    out = Path(output) / "out"
    out.mkdir(parents=True, exist_ok=True)
    env = {"ARC_RUN_ID": spec.get("run_id", ""), "ARC_OUT": "/out", "ARC_W": "/w",
           "ARC_TIMEOUT_SECONDS": str(spec["timeout"] - 60), **(spec.get("image_env") or {})}
    return ["docker", "run", "--rm", "--name", IMAGE_CONTAINER, "--gpus", "all", "--network", "host", "--ipc", "host",
            "-v", f"{ASSETS}:/w:ro", "-v", f"{out}:/out", "-v", f"{SOURCE}:/opt/arc:ro",
            *[a for k, v in sorted(env.items()) for a in ("-e", f"{k}={v}")],
            "--entrypoint", "bash", spec["image"], f"/opt/arc/{spec['script']}", *spec["arguments"]]


def prepare_docker(image, run=subprocess.run):
    """docker + the NVIDIA container runtime (installed when the boot image lacks them), then the pinned image."""
    report, began = {}, time.time()
    if not shutil.which("docker"):
        run(["/bin/sh", "-c", "apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q docker.io"],
            check=True, timeout=900)
        report["installed_docker"] = True
    probe = run(["docker", "info", "--format", "{{json .Runtimes}}"], capture_output=True, text=True, timeout=60)
    if "nvidia" not in (probe.stdout or ""):
        run(["/bin/sh", "-c", NVIDIA_TOOLKIT], check=True, timeout=900)
        report["installed_nvidia_toolkit"] = True
    pulled = time.time()
    run(["docker", "pull", "-q", image], check=True, timeout=3600)
    report.update(setup_seconds=round(pulled - began, 1), pull_seconds=round(time.time() - pulled, 1), image=image)
    return report


NVIDIA_TOOLKIT = (
    "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o "
    "/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -fsSL "
    "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed "
    "'s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > "
    "/etc/apt/sources.list.d/nvidia-container-toolkit.list && apt-get update -q && DEBIAN_FRONTEND=noninteractive "
    "apt-get install -y -q nvidia-container-toolkit && nvidia-ctk runtime configure --runtime=docker && "
    "systemctl restart docker")


def job_command(spec, output, python=None):
    if spec.get("image"):
        return image_command(spec, output)
    command = [python or sys.executable, str(SOURCE / spec["script"]), "--bundle", str(SOURCE),
               "--working", str(WORKING), "--output", str(Path(output) / "arena"),
               "--games-dir", str(ASSETS / "games"), "--max-seconds", str(spec["timeout"] - 60), *spec["arguments"]]
    if spec.get("block_network", True):
        # Modal block_network=True: a fresh network namespace. Loopback for the server (127.0.0.1:1234) and, like
        # Modal's container, an interface with a default route that leads nowhere (best effort).
        return ["unshare", "--net", "--", "/bin/sh", "-c", ISOLATE, "sh", *command]
    return command


def sha256_file(path, chunk=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb", buffering=0) as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def skip_list(manifest, patterns):
    """Manifest paths matching the spec's ``asset_skip`` patterns (files the run never reads: not fetched or checked).
    A pattern that matches nothing is a stale spec and an error."""
    files = manifest["files"]
    for pattern in patterns:
        if not any(fnmatch.fnmatchcase(rel, pattern) for rel in files):
            raise ValueError(f"asset_skip pattern {pattern!r} matches no file of {manifest['prefix']}")
    return sorted(rel for rel in files if any(fnmatch.fnmatchcase(rel, pattern) for pattern in patterns))


def verify_manifest(root, manifest, mode="sha256", workers=16, skip=()):
    """Every manifest file (but ``skip``) is present under ``root`` with its size (and sha256); raises on the first
    difference."""
    root = Path(root)
    skip = set(skip)
    files = {rel: meta for rel, meta in manifest["files"].items() if rel not in skip}
    started = time.time()
    for rel, meta in files.items():
        path = root / rel
        if not path.is_file() or path.stat().st_size != meta["size"]:
            raise ValueError(f"asset size mismatch: {manifest['prefix']}/{rel}")
    if mode == "sha256":
        # Largest first, so the pool is not left with one long tail; hashlib releases the GIL.
        order = sorted(files, key=lambda rel: -files[rel]["size"])
        with ThreadPoolExecutor(workers) as pool:
            for rel, digest in zip(order, pool.map(lambda rel: sha256_file(root / rel), order)):
                if digest != files[rel]["sha256"]:
                    raise ValueError(f"asset sha256 mismatch: {manifest['prefix']}/{rel}")
    elif mode != "size":
        raise ValueError(f"unknown verify mode {mode}")
    return {"prefix": manifest["prefix"], "files": len(files), "bytes": sum(m["size"] for m in files.values()),
            "mode": mode, "seconds": round(time.time() - started, 1),
            **({"skipped_files": len(skip), "skipped_bytes": sum(manifest["files"][rel]["size"] for rel in skip)}
               if skip else {})}


# -- boot calibration (VMs come in speeds ~10% apart; no read recorded which it got). One process of its own
# before the assets (nothing else on the VM yet), hard-limited, and a failure only logs. stdlib + the driver's libcuda.
CALIBRATE_SECONDS = 150
CALIBRATE_SCHEMA = 1
CALIB_PTX = b"""
.version 7.0
.target sm_80
.address_size 64
.entry calib (.param .u64 p_a, .param .u32 p_n, .param .u32 p_passes)
{
    .reg .u32 %tx, %bx, %bd, %i, %n, %passes, %p;
    .reg .u64 %a, %off, %addr;
    .reg .f32 %x;
    .reg .pred %stop;
    ld.param.u64 %a, [p_a];
    ld.param.u32 %n, [p_n];
    ld.param.u32 %passes, [p_passes];
    mov.u32 %tx, %tid.x;
    mov.u32 %bx, %ctaid.x;
    mov.u32 %bd, %ntid.x;
    mul.lo.u32 %i, %bx, %bd;
    add.u32 %i, %i, %tx;
    setp.ge.u32 %stop, %i, %n;
    @%stop bra END;
    cvt.u64.u32 %off, %i;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %a, %off;
    ld.global.f32 %x, [%addr];
    mov.u32 %p, 0;
LOOP:
    setp.ge.u32 %stop, %p, %passes;
    @%stop bra DONE;
    fma.rn.f32 %x, %x, 0f3F800001, 0f33D6BF95;
    add.u32 %p, %p, 1;
    bra LOOP;
DONE:
    st.global.f32 [%addr], %x;
END:
    ret;
}
\x00"""


def spread(values):
    """(best rate, relative spread (best - worst) / best) of a bench's repeats. Best, as STREAM does: interference only
    slows a repeat. The spread is the reader's noise floor for that number."""
    best = max(values)
    return best, round((best - min(values)) / best, 4) if best else None


def command_text(argv, timeout=20):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def calib_host_facts():
    cpu = {k.strip(): v.strip() for k, _, v in (x.partition(":") for x in command_text(["lscpu"]).splitlines())}
    mem = [x.split(":", 1)[1].strip() for x in command_text(["dmidecode", "-t", "17"]).splitlines()
           if x.strip().startswith(("Speed:", "Configured Memory Speed:"))]
    dmi = {}
    for name in ("sys_vendor", "product_name", "product_uuid", "product_serial", "board_serial", "chassis_serial", "bios_version"):
        try:
            dmi[name] = Path(f"/sys/class/dmi/id/{name}").read_text().strip()
        except OSError:
            dmi[name] = None
    meminfo = Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else ""
    total = next((int(x.split()[1]) for x in meminfo.splitlines() if x.startswith("MemTotal:")), 0)
    return {"cpu_model": cpu.get("Model name"), "cpu_count": os.cpu_count(), "sockets": cpu.get("Socket(s)"),
            "threads_per_core": cpu.get("Thread(s) per core"), "cpu_max_mhz": cpu.get("CPU max MHz"), "l3": cpu.get("L3 cache"),
            "numa_nodes": cpu.get("NUMA node(s)"), "hypervisor": cpu.get("Hypervisor vendor"), "memory_speed": sorted(set(mem)),
            "mem_total_gib": round(total / 2 ** 20, 1), "kernel": command_text(["uname", "-r"]), "hostid": command_text(["hostid"]),
            "dmi": dmi}


GPU_FIELDS = ("name", "uuid", "serial", "vbios_version", "driver_version", "pci.bus_id", "pcie.link.gen.max", "pcie.link.width.max",
              "pcie.link.gen.current", "pcie.link.width.current", "memory.total", "power.limit", "clocks.max.sm", "clocks.max.mem",
              "clocks.sm", "clocks.mem", "temperature.gpu", "clocks_event_reasons.active")


def calib_gpu_facts():
    text = command_text(["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader"])
    return [dict(zip(GPU_FIELDS, (x.strip() for x in line.split(",")))) for line in text.splitlines()]


def calib_bench_host(per_thread=1 << 29, repeats=5, loops=5_000_000):
    """All-core and one-core memory copy (GB/s) and a fixed pure-Python loop (Mops/s): the host side of serving."""
    import ctypes
    import mmap
    threads = os.cpu_count() or 1
    free = next((int(x.split()[1]) << 10 for x in Path("/proc/meminfo").read_text().splitlines()
                 if x.startswith("MemAvailable:")), 8 << 30) if Path("/proc/meminfo").exists() else 8 << 30
    size = min(per_thread, max(1 << 24, free // 10 // threads // 2))
    maps = [mmap.mmap(-1, 2 * size) for _ in range(threads)]
    addr = [ctypes.addressof(ctypes.c_char.from_buffer(m)) for m in maps]
    for a in addr:
        ctypes.memset(a, 1, 2 * size)  # the first touch is page faults, not memory speed

    def round_of(n):
        ready = threading.Barrier(n + 1)

        def work(a):
            ready.wait()
            ctypes.memmove(a + size, a, size)
        pool = [threading.Thread(target=work, args=(a,)) for a in addr[:n]]
        for t in pool:
            t.start()
        ready.wait()
        began = time.perf_counter()
        for t in pool:
            t.join()
        return n * size / (time.perf_counter() - began) / 1e9

    every, one = spread([round_of(threads) for _ in range(repeats)]), spread([round_of(1) for _ in range(repeats)])
    gc_loop = []
    for _ in range(3):
        began, x = time.perf_counter(), 0
        for i in range(loops):
            x = (x * 31 + i) & 0xFFFFFFFF
        gc_loop.append(loops / (time.perf_counter() - began) / 1e6)
    py = spread(gc_loop)
    return {"host_mem_copy_gbs": round(every[0], 2), "host_mem_copy_spread": every[1], "host_mem_copy_1t_gbs": round(one[0], 2),
            "host_mem_copy_1t_spread": one[1], "host_py_mops": round(py[0], 2), "host_py_spread": py[1], "host_py_check": x,
            "host_threads": threads, "host_bytes_per_thread": size}


def calib_bench_gpu(cuda=None, core=(1 << 24, 1 << 16), mem_elements=1 << 31, repeats=7):
    """One PTX kernel, no model and no toolkit (the driver compiles it): ``gpu_core_gflops`` many fused multiply-adds on a
    small array (core clock), ``gpu_mem_gbs`` one read + write of each element of a large one (memory), plus the
    driver's fill and a host-to-card copy (PCIe)."""
    import ctypes
    from ctypes import byref, c_char_p, c_float, c_int, c_size_t, c_uint, c_ulonglong, c_void_p
    cu = cuda or ctypes.CDLL("libcuda.so.1")

    def ck(name, *args):
        code = getattr(cu, name)(*args)
        if code:
            raise RuntimeError(f"{name}: CUDA error {code}")

    ck("cuInit", c_uint(0))
    device, context, module, kernel = c_int(0), c_void_p(), c_void_p(), c_void_p()
    ck("cuDeviceGet", byref(device), c_int(0))
    ck("cuCtxCreate_v2", byref(context), c_uint(0), device)
    out = {}

    def launch(base, n, passes):
        args = [c_ulonglong(base), c_uint(n), c_uint(passes)]
        params = (c_void_p * 3)(*[ctypes.addressof(a) for a in args])
        began = time.perf_counter()
        ck("cuLaunchKernel", kernel, c_uint((n + 255) // 256), c_uint(1), c_uint(1), c_uint(256), c_uint(1), c_uint(1), c_uint(0),
           None, params, None)
        ck("cuCtxSynchronize")
        return time.perf_counter() - began

    big, small = c_ulonglong(0), c_ulonglong(0)
    ck("cuMemAlloc_v2", byref(big), c_size_t(mem_elements * 4))
    ck("cuMemAlloc_v2", byref(small), c_size_t(core[0] * 4))

    def fill(base, n):
        began = time.perf_counter()
        ck("cuMemsetD32_v2", c_ulonglong(base), c_uint(0x3F800000), c_size_t(n))  # 1.0f
        ck("cuCtxSynchronize")
        return n * 4 / (time.perf_counter() - began) / 1e9

    filled = spread([fill(big.value, mem_elements) for _ in range(repeats)][1:])
    out.update(gpu_fill_gbs=round(filled[0], 1), gpu_fill_spread=filled[1])
    import mmap
    copy_bytes = min(2 << 30, mem_elements * 4)
    host = mmap.mmap(-1, copy_bytes)
    source = ctypes.c_char.from_buffer(host)
    ctypes.memset(ctypes.addressof(source), 1, copy_bytes)

    def to_card():
        began = time.perf_counter()
        ck("cuMemcpyHtoD_v2", c_ulonglong(big.value), byref(source), c_size_t(copy_bytes))
        return copy_bytes / (time.perf_counter() - began) / 1e9

    copied = spread([to_card() for _ in range(repeats)][1:])
    out.update(gpu_h2d_gbs=round(copied[0], 2), gpu_h2d_spread=copied[1])
    ck("cuModuleLoadData", byref(module), c_char_p(CALIB_PTX))
    ck("cuModuleGetFunction", byref(kernel), module, b"calib")
    ck("cuMemsetD32_v2", c_ulonglong(small.value), c_uint(0x3F800000), c_size_t(core[0]))
    ck("cuCtxSynchronize")
    fast = spread([core[0] * core[1] / launch(small.value, *core) / 1e9 for _ in range(repeats)][1:])
    check = c_float(0.0)
    ck("cuMemcpyDtoH_v2", byref(check), c_ulonglong(small.value), c_size_t(4))
    if not check.value > 1.0:
        raise RuntimeError(f"kernel left {check.value}: it did not run")
    ck("cuMemsetD32_v2", c_ulonglong(big.value), c_uint(0x3F800000), c_size_t(mem_elements))
    ck("cuCtxSynchronize")
    wide = spread([2 * mem_elements * 4 / launch(big.value, mem_elements, 1) / 1e9 for _ in range(repeats)][1:])
    out.update(gpu_core_gflops=round(fast[0], 1), gpu_core_spread=fast[1], gpu_mem_gbs=round(wide[0], 1), gpu_mem_spread=wide[1])
    return out


def calibrate(path):
    """Writes ``path`` after each stage, so a stage that hangs (the runner kills it) leaves the ones before it."""
    began = time.time()
    data = {"schema": CALIBRATE_SCHEMA, "started_epoch": round(began, 1), "bench": {}, "errors": []}

    def save():
        data["seconds"] = round(time.time() - began, 1)
        Path(f"{path}.tmp").write_text(json.dumps(data, indent=2) + "\n")
        os.replace(f"{path}.tmp", path)

    for key, stage in (("host", calib_host_facts), ("gpus", calib_gpu_facts), ("bench", calib_bench_host), ("bench", calib_bench_gpu)):
        started = time.time()
        try:
            got = stage()
            if key == "bench":
                data["bench"].update(got)
            else:
                data[key] = got
        except Exception as exc:  # noqa: BLE001 - one stage's failure is a line in the file
            data["errors"].append(f"{stage.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
        data.setdefault("stage_seconds", {})[stage.__name__] = round(time.time() - started, 1)
        save()
    gpus = data.get("gpus") or []
    if gpus:
        data["gpus_end"] = [{k: g.get(k) for k in ("clocks.sm", "clocks.mem", "temperature.gpu", "clocks_event_reasons.active")}
                            for g in calib_gpu_facts()]
        save()
    return data


def mirror_live_file(rclone, remote, local, present):
    """One poll of a live knob file: a new remote version replaces the local file atomically (tmp, then rename; an
    unchanged one is left alone), a deleted remote file removes the local one. A failed copy keeps the last good file."""
    if not present:
        Path(local).unlink(missing_ok=True)
        return
    tmp = f"{local}.tmp"
    try:
        rclone("copyto", remote, tmp, timeout=60, check=False)
        new = Path(tmp).read_bytes()
    except (subprocess.SubprocessError, OSError):
        Path(tmp).unlink(missing_ok=True)
        return
    try:
        same = Path(local).read_bytes() == new
    except OSError:
        same = False
    if same:
        Path(tmp).unlink(missing_ok=True)
    else:
        os.replace(tmp, local)


def checkpoints(trials):
    """``{trial: newest complete generation's seq, elapsed and epoch}`` under a trials folder."""
    found = {}
    for manifest in sorted(Path(trials).glob("*/checkpoint/gen-*/manifest.json")):
        if manifest.parent.name.endswith(".tmp"):
            continue
        try:
            data = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        found[manifest.parents[2].name] = {key: data.get(key) for key in ("seq", "elapsed_seconds", "epoch", "bytes")}
    return found


def stop_rule_reason(rule, score, usd):
    """modal_validation.target_watcher's rule: the live score reached the target, or the dollars the limit."""
    if score is not None and score >= rule["score_target"]:
        return f"score {score:.2f} >= target {rule['score_target']}"
    if rule.get("usd_limit") and usd >= rule["usd_limit"]:
        return f"${usd:.2f} >= usd_limit {rule['usd_limit']}"
    return None


def imds(path, timeout=5):
    import urllib.request

    try:
        request = urllib.request.Request(f"{IMDS}/{path}", headers={"Metadata": "true"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode()
    except Exception as exc:  # noqa: BLE001 - metadata is evidence only
        return f"unavailable: {type(exc).__name__}"


class Runner:
    def __init__(self, spec):
        self.spec = spec
        self.run_id = spec["run_id"]
        self.output = RUNS / self.run_id
        self.instance_id = spec.get("instance_id")
        self.remote_runs = f"{REMOTE}:{spec['bucket']}/{spec['runs_prefix']}/validation/{self.run_id}"
        self.started = time.time()
        self.phase, self.history = "boot", []
        self.interrupted, self.finishing = None, False
        self.process = None
        self.late_error = None
        self.manifest_digests = {}
        try:  # an image made by a bake (spec use_baked); absent on the stock image
            self.baked = json.loads(BAKED.read_text()) if spec.get("use_baked") and BAKED.is_file() else {}
        except (OSError, ValueError):
            self.baked = {}
        self.stop = threading.Event()
        self.sync_lock = threading.Lock()
        self.checkpoint_lock = threading.Lock()  # its own: the notice's upload never waits for an evidence pass

    # -- bucket ---------------------------------------------------------------------------------------------------
    def rclone(self, *args, timeout=None, check=True):
        return subprocess.run(["rclone", *args, "--retries", "3", "--low-level-retries", "10", "--stats", "0"],
                              check=check, timeout=timeout, capture_output=True, text=True)

    def sync(self, *, full, timeout=None):
        """Copy the output folder to the bucket (same layout as the runs volume). Interim passes skip big files."""
        with self.sync_lock:
            limits = [] if full else ["--max-size", "64M"]
            try:
                self.rclone("copy", str(self.output), self.remote_runs, "--update", "--use-server-modtime",
                            "--transfers", "8", "--checkers", "16", "--exclude", "exchange-in/**", *limits,
                            timeout=timeout)
                return True
            except (subprocess.SubprocessError, OSError) as exc:  # a failed pass must not end the run
                print(f"sync failed: {type(exc).__name__}: {str(getattr(exc, 'stderr', exc))[:300]}", flush=True)
                return False

    def exchange_in(self):
        """spec ``exchange_in``: other runs' output folders copied into <output>/exchange-in/<prefix> each sync."""
        for prefix in self.spec.get("exchange_in") or []:
            target = self.output / "exchange-in" / prefix
            target.mkdir(parents=True, exist_ok=True)
            try:
                self.rclone("copy", f"{REMOTE}:{self.spec['bucket']}/{self.spec['runs_prefix']}/validation/{prefix}",
                            str(target), "--update", "--transfers", "8", "--checkers", "16", timeout=300)
            except (subprocess.SubprocessError, OSError) as exc:
                print(f"exchange copy failed ({prefix}): {type(exc).__name__}", flush=True)

    def put_json(self, name, payload, timeout=30):
        path = self.output / "nebius" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        try:
            self.rclone("copyto", str(path), f"{self.remote_runs}/nebius/{name}", timeout=timeout)
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"marker upload failed ({name}): {type(exc).__name__}", flush=True)

    def launch_cancelled(self):
        """The launcher's cancel marker for this launch attempt, or None."""
        try:
            done = self.rclone("cat", f"{self.remote_runs}/nebius/cancelled.json", timeout=60, check=False)
            marker = json.loads(done.stdout) if done.returncode == 0 and done.stdout.strip() else None
        except (subprocess.SubprocessError, OSError, ValueError):
            return None
        if marker and marker.get("launch_attempt") in (None, self.spec.get("launch_attempt")):
            return marker
        return None

    def set_phase(self, phase, **extra):
        self.phase = phase
        self.history.append({"phase": phase, "epoch": round(time.time(), 1), **extra})
        print(json.dumps({"event": "nebius_phase", **self.history[-1]}), flush=True)
        self.put_json("status.json", {"run_id": self.run_id, "phase": phase, "epoch": round(time.time(), 1),
                                      "started_epoch": self.started, "deadline_epoch": self.spec["deadline_epoch"],
                                      "instance_id": self.instance(),
                                      "history": self.history})

    def instance(self):
        if not self.instance_id:
            self.instance_id = imds("instance-data/id")
        return self.instance_id

    # -- threads --------------------------------------------------------------------------------------------------
    def syncer(self):
        cycle = 0
        while not self.stop.wait(self.spec.get("sync_seconds", 60)):
            cycle += 1
            self.write_diag()
            self.sync(full=cycle % 10 == 0, timeout=600)
            self.exchange_in()
            if self.spec.get("checkpoint"):
                self.sync_checkpoints(timeout=600)
            if cycle % 5 == 0 and (WORKING / "trials").is_dir():
                # The trial's working files reach the output folder only at its end; keep the small ones in the
                # bucket so a preempted run leaves its progress and logs behind.
                try:
                    self.rclone("copy", str(WORKING / "trials"), f"{self.remote_runs}/arena/working/trials",
                                "--update", "--use-server-modtime", "--max-size", "32M", "--transfers", "8",
                                "--exclude", "*/checkpoint/**", timeout=300)
                except (subprocess.SubprocessError, OSError):
                    pass

    def write_diag(self):
        """nebius-diag.txt in the output (synced each minute): disk, memory, kernel log and runner log tails, so a VM
        that vanishes without a final marker (AR v1/v2, 13 Oct: gone during the asset fetch) leaves evidence."""
        parts = []
        for title, argv in (("df", ["df", "-h", "/", "/assets"]), ("free", ["free", "-g"]),
                            ("dmesg", ["sh", "-c", "dmesg -T 2>/dev/null | tail -n 40"]),
                            ("runner.log", ["sh", "-c", "tail -n 40 /var/log/arc/runner.log 2>/dev/null"]),
                            ("top", ["sh", "-c", "ps -eo pid,rss,pcpu,comm --sort=-rss | head -n 12"])):
            try:
                out = subprocess.run(argv, capture_output=True, text=True, timeout=20).stdout
            except (subprocess.SubprocessError, OSError) as exc:
                out = f"{type(exc).__name__}"
            parts.append(f"== {title}\n{out}")
        try:
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / "nebius-diag.txt").write_text(f"epoch {time.time():.0f} phase {self.phase}\n" + "\n".join(parts))
        except OSError:
            pass

    # -- checkpoints (src/arena/checkpoint.py) -----------------------------------------------------------------------
    def sync_checkpoints(self, timeout):
        """Each trial's complete generations to the bucket (the ones it dropped leave it too); the newest per trial."""
        with self.checkpoint_lock:
            for folder in sorted((WORKING / "trials").glob("*/checkpoint")):
                try:
                    self.rclone("sync", str(folder),
                                f"{self.remote_runs}/arena/working/trials/{folder.parent.name}/checkpoint",
                                "--exclude", "*.tmp/**", "--exclude", "request", "--transfers", "8",
                                "--s3-upload-concurrency", "8", "--s3-chunk-size", "64M", timeout=timeout)
                except (subprocess.SubprocessError, OSError) as exc:
                    print(f"checkpoint sync failed: {type(exc).__name__}", flush=True)
                    return None
        newest = checkpoints(WORKING / "trials")
        done = sorted(p.parent.name for p in (WORKING / "trials").glob("*/execution.json"))
        if (newest, done) != getattr(self, "uploaded", None):
            self.uploaded = (newest, done)
            # What a relaunch can continue from: the launcher resumes only when this lists something.
            self.put_json("checkpoint.json", {"epoch": round(time.time(), 1), "attempt": self.spec.get("attempt", 1),
                                              "trials": newest, "done": done})
        return newest

    def final_checkpoint(self):
        """The preemption notice: ask every running trial for a checkpoint now and wait briefly for it."""
        folders = [f for f in (WORKING / "trials").glob("*/checkpoint") if not (f.parent / "execution.json").exists()]
        before = checkpoints(WORKING / "trials")
        for folder in folders:
            (folder / "request").touch()
        deadline = time.time() + CHECKPOINT_WAIT
        while time.time() < deadline and any((folder / "request").exists() for folder in folders):
            time.sleep(0.25)
        after = checkpoints(WORKING / "trials")
        return {"asked": len(folders), "written": sorted(t for t in after if after[t] != before.get(t)),
                "seconds": round(CHECKPOINT_WAIT - max(0.0, deadline - time.time()), 1)}

    def restore_working(self):
        """A relaunch: the trial folders the earlier machine left in the bucket, for the bootstrap to resume from."""
        (WORKING / "trials").mkdir(parents=True, exist_ok=True)
        self.rclone("copy", f"{self.remote_runs}/arena/working/trials", str(WORKING / "trials"), "--transfers", "16",
                    "--multi-thread-streams", "8", timeout=1800)
        return checkpoints(WORKING / "trials")

    def calibrate_vm(self):
        """The VM's identity and speed, before anything else runs on it. Never blocks or fails the run:
        CALIBRATE_SECONDS is a hard limit and any failure is one line in nebius/calibration.json."""
        if not shutil.which("nvidia-smi"):
            return None
        path, started, process = self.output / "nebius/calibration.json", time.time(), None
        data = {"schema": CALIBRATE_SCHEMA, "started_epoch": round(started, 1), "bench": {}, "errors": []}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--calibrate", str(path)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
            try:
                stderr = process.communicate(timeout=self.spec.get("calibrate_seconds", CALIBRATE_SECONDS))[1]
                if process.returncode:
                    data["errors"].append(f"exit {process.returncode}: {stderr.decode(errors='replace')[-300:]}")
            except subprocess.TimeoutExpired:
                data["errors"].append(f"killed at the {self.spec.get('calibrate_seconds', CALIBRATE_SECONDS)} s limit")
            if path.exists():
                data = {**json.loads(path.read_text()), "errors": json.loads(path.read_text()).get("errors", []) + data["errors"]}
        except Exception as exc:  # noqa: BLE001 - the bench never costs the run
            data["errors"].append(f"{type(exc).__name__}: {str(exc)[:300]}")
        finally:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                process.wait()
        try:
            data.update(run_id=self.run_id, attempt=self.spec.get("attempt", 1), instance=self.instance(),
                        boot_seconds=round(time.time() - started, 1))
            self.put_json("calibration.json", data)
            print(json.dumps({"event": "calibration", "seconds": data["boot_seconds"], "bench": data.get("bench"),
                              "errors": data["errors"]}), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"calibration not saved: {type(exc).__name__}", flush=True)
        return {"seconds": data.get("boot_seconds"), "errors": len(data.get("errors") or [])}

    def stop_watcher(self):
        # scripts/nebius_validation.py stop RUN_ID puts validation/<run>/arena/live/stop-request in the bucket.
        folder = f"{self.remote_runs}/arena/live"
        while not self.stop.wait(self.spec.get("stop_poll_seconds", STOP_POLL_SECONDS)):
            try:
                names = set(self.rclone("lsf", folder, timeout=60, check=False).stdout.split())
            except (subprocess.SubprocessError, OSError):
                continue
            if CONTROL_FILE in names:  # copied as is; the trial reads it when its mtime changes
                try:
                    self.rclone("copyto", f"{folder}/{CONTROL_FILE}", CONTROL_LOCAL, timeout=60, check=False)
                except (subprocess.SubprocessError, OSError):
                    pass
            mirror_live_file(self.rclone, f"{folder}/{KNOBS_FILE}", KNOBS_LOCAL, KNOBS_FILE in names)
            if STOP_FILE in names or (self.output / "arena/live" / STOP_FILE).exists():
                Path(STOP_LOCAL).write_text(json.dumps({"source": f"{folder}/{STOP_FILE}",
                                                        "epoch": round(time.time(), 1)}) + "\n")
                print(json.dumps({"event": "stop_request_seen", "path": f"{folder}/{STOP_FILE}"}), flush=True)
                if self.spec.get("image"):  # the container cannot see the local marker: a graceful stop (SIGTERM, 60 s)
                    subprocess.run(["docker", "stop", "-t", "60", IMAGE_CONTAINER], capture_output=True, timeout=120)
                return

    def target_watcher(self, bootstrap_started):
        rule = self.spec["stop_rule"]
        progress = self.output / "arena/live/progress.json"
        while not self.stop.wait(60):
            score = None
            try:
                score = ((json.loads(progress.read_text()).get("live") or {}).get("score") or {}).get("score")
            except (OSError, ValueError):
                pass
            why = stop_rule_reason(rule, score, (time.time() - bootstrap_started) * self.spec["rate_per_second"])
            if why:
                Path(STOP_LOCAL).write_text(json.dumps({"source": "b200 stop rule", "reason": why,
                                                        "epoch": round(time.time(), 1)}) + "\n")
                print(json.dumps({"event": "b200_stop_rule", "reason": why}), flush=True)
                return

    # -- steps ----------------------------------------------------------------------------------------------------
    def wait_late(self, prefix, manifest_path):
        """spec ``asset_late`` (boot the read VM while the export runs): the asset's manifest, which the
        export writes last, is polled for until ``wait_seconds`` after the VM's start. A failed-export signal or the
        ceiling fails the run (the VM then ends as after any failure); the fetch and its check follow as usual."""
        spec, late = self.spec, self.spec["asset_late"][prefix]
        deadline = self.started + float(late["wait_seconds"])
        poll = float(late.get("poll_seconds") or LATE_POLL_SECONDS)
        name, began = prefix.rstrip("/").rsplit("/", 1)[-1], time.time()
        print(json.dumps({"event": "asset_late_wait", "prefix": prefix, "deadline_epoch": round(deadline, 1)}), flush=True)
        while True:
            got = self.rclone("copyto", f"{REMOTE}:{spec['bucket']}/{spec['manifests_prefix']}/{manifest_path.name}",
                              str(manifest_path), timeout=120, check=False)
            if got.returncode == 0 and manifest_path.is_file():
                waited = round(time.time() - began, 1)
                self.identity.setdefault("asset_waits", {})[prefix] = waited
                print(json.dumps({"event": "asset_late_ready", "prefix": prefix, "waited_seconds": waited}), flush=True)
                return
            failed = self.rclone("cat", f"{REMOTE}:{spec['bucket']}/{SIGNALS_PREFIX}/{name}.failed.json", timeout=60,
                                 check=False)
            if failed.returncode == 0 and failed.stdout.strip():
                raise RuntimeError(f"late asset {prefix}: the export failed: {failed.stdout.strip()[:500]}")
            if time.time() + poll > deadline:
                raise RuntimeError(f"late asset {prefix}: no manifest within {late['wait_seconds']} s of the VM start")
            time.sleep(poll)

    def late_watch(self, prefixes):
        """SFT's byte checks can fail after the manifest landed: its failed signal stops the run until
        LATE_WATCH_SECONDS after the last late asset was fetched."""
        spec, end = self.spec, time.time() + LATE_WATCH_SECONDS
        poll = 3 * min(float(spec["asset_late"][p].get("poll_seconds") or LATE_POLL_SECONDS) for p in prefixes)
        while time.time() < end and not self.stop.is_set():
            for prefix in prefixes:
                name = prefix.rstrip("/").rsplit("/", 1)[-1]
                got = self.rclone("cat", f"{REMOTE}:{spec['bucket']}/{SIGNALS_PREFIX}/{name}.failed.json", timeout=60,
                                  check=False)
                if got.returncode == 0 and got.stdout.strip():
                    self.late_error = f"late asset {prefix}: the export failed: {got.stdout.strip()[:500]}"
                    print(json.dumps({"event": "asset_failed", "prefix": prefix, "error": self.late_error}), flush=True)
                    if self.process is not None and self.process.poll() is None:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    return
            self.stop.wait(poll)

    def fetch_assets(self):
        """Every asset; late ones (spec asset_late) on a thread beside the rest: each is fetched the moment its manifest
        lands (2 Nov: the 1293 overlay was ready 7 min before the base fetch ended), and all are checked before return."""
        assets, late = self.spec["assets"], self.spec.get("asset_late") or {}
        if not late:
            return self.fetch_some(assets)
        box = {}

        def worker():
            try:
                box["report"] = self.fetch_some([p for p in assets if p in late])
            except BaseException as exc:  # noqa: BLE001 - raised on the main thread below
                box["error"] = exc
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        report = self.fetch_some([p for p in assets if p not in late])
        if thread.is_alive():
            self.set_phase("asset_wait", prefixes=sorted(late))
            thread.join()
            self.set_phase("assets")
        if "error" in box:
            raise box["error"]
        threading.Thread(target=self.late_watch, args=(list(late),), daemon=True).start()
        return report + box["report"]

    def fetch_some(self, prefixes):
        spec, report, late = self.spec, [], self.spec.get("asset_late") or {}
        for prefix in prefixes:
            manifest_path = STATE / "manifests" / f"{asset_slug(prefix)}.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            if prefix in late:
                self.wait_late(prefix, manifest_path)
            self.rclone("copyto", f"{REMOTE}:{spec['bucket']}/{spec['manifests_prefix']}/{manifest_path.name}",
                        str(manifest_path), timeout=120)
            manifest = json.loads(manifest_path.read_text())
            if manifest["prefix"] != prefix:
                raise ValueError(f"manifest {manifest_path.name} is for {manifest['prefix']}")
            target = ASSETS / prefix
            skip = skip_list(manifest, (spec.get("asset_skip") or {}).get(prefix, []))
            digest = sha256_file(manifest_path)
            self.manifest_digests[prefix] = {"manifest_sha256": digest, "skip": sorted(skip)}
            if spec.get("use_baked") and self.baked.get(prefix) == self.manifest_digests[prefix] and target.is_dir():
                # on the image already: sizes only, no fetch, no sha256 (the bake verified them)
                result = verify_manifest(target, manifest, "size", skip=skip)
                result.update(fetch_seconds=0.0, baked=True)
                report.append(result)
                print(json.dumps({"event": "asset_ready", **result}), flush=True)
                continue
            excludes = []
            if skip:  # exact paths (no glob characters in our asset names), anchored at the prefix root
                if any(set(rel) & set("*?[]{}\\") for rel in skip):
                    raise ValueError(f"asset_skip path with a filter character in {prefix}")
                listing = STATE / "manifests" / f"{asset_slug(prefix)}.skip"
                listing.write_text("".join(f"/{rel}\n" for rel in skip))
                excludes = ["--exclude-from", str(listing)]
            started = time.time()
            self.rclone("copy", f"{REMOTE}:{spec['bucket']}/{spec['assets_prefix']}/{prefix}", str(target),
                        "--transfers", "16", "--checkers", "32", "--multi-thread-streams", "8",
                        "--multi-thread-cutoff", "256M", "--buffer-size", "64M", *excludes)
            present = [rel for rel in skip if (target / rel).exists()]
            if present:
                raise ValueError(f"asset_skip: {len(present)} skipped files fetched anyway, e.g. {present[0]}")
            result = verify_manifest(target, manifest, spec.get("verify", "sha256"), skip=skip)
            result["fetch_seconds"] = round(time.time() - started - result["seconds"], 1)
            report.append(result)
            print(json.dumps({"event": "asset_ready", **result}), flush=True)
        return report

    def fetch_inputs(self):
        """spec ``inputs`` (CPU jobs, ``cpujob``): ``{"runs": <launch file>, "assets": <launch file>}``, each a list of
        bucket keys relative to runs/validation or to the assets prefix, copied to INPUTS/runs and INPUTS/assets before
        the bundle starts (its own network can stay closed; the keys never reach it)."""
        report = {}
        for part, prefix in (("runs", f"{self.spec['runs_prefix']}/validation"), ("assets", self.spec["assets_prefix"])):
            name = self.spec["inputs"].get(part)
            if not name:
                continue
            started = time.time()
            done = self.rclone("copy", f"{REMOTE}:{self.spec['bucket']}/{prefix}", str(INPUTS / part),
                               "--files-from-raw", str(STATE / "launch" / name), "--no-traverse", "--transfers", "32",
                               "--checkers", "64", check=False)
            files = sum(1 for p in (INPUTS / part).rglob("*") if p.is_file())
            report[part] = {"listed": sum(1 for line in open(STATE / "launch" / name) if line.strip()), "files": files,
                            "seconds": round(time.time() - started, 1), "rclone_exit": done.returncode}
            print(json.dumps({"event": "inputs_ready", "part": part, **report[part]}), flush=True)
        return report

    def extract_bundle(self):
        archive = STATE / "launch/bundle.tar.gz"
        if sha256_file(archive) != self.spec["bundle_sha256"]:
            raise ValueError("bundle archive sha256 mismatch")
        SOURCE.mkdir(parents=True, exist_ok=False)
        with tarfile.open(archive) as tar:
            tar.extractall(SOURCE, filter="data")

    def preflight(self):
        """Cheap CPU-VM check of everything a run depends on except the GPU and the assets."""
        probe = self.output / "nebius/preflight-probe.txt"
        probe.write_text(f"{self.run_id} {time.time()}\n")
        self.rclone("copyto", str(probe), f"{self.remote_runs}/nebius/preflight-probe.txt", timeout=60)
        listed = self.rclone("lsf", f"{REMOTE}:{self.spec['bucket']}/{self.spec['manifests_prefix']}/",
                             timeout=60, check=False)
        isolated = subprocess.run(job_command({**self.spec, "script": "x", "arguments": [], "timeout": 60}, self.output)
                                  [:7] + ["/bin/sh", "-c", "ip -brief addr; ip route; "
                                          "! curl -s -m 4 -o /dev/null https://nebius.com "
                                          "&& ! curl -s -m 4 -o /dev/null -H 'Metadata: true' "
                                          "http://metadata.nebius.internal/v1/instance-data/id"],
                                  capture_output=True, text=True)
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True) if shutil.which("nvidia-smi") else None
        return {"bucket_write": True, "manifests": listed.stdout.split(), "network_isolation": isolated.returncode == 0,
                "isolated_interfaces": isolated.stdout.strip(),
                "python": sys.version.split()[0], "python_h": Path("/usr/include/python3.12/Python.h").is_file(),
                "nebius_cli": subprocess.run(["/usr/local/bin/arc-nebius", "iam", "whoami", "--format", "json"],
                                             capture_output=True, text=True).returncode == 0,
                "tools": {name: shutil.which(name) for name in ("gcc", "g++", "ld", "ninja", "unshare", "ip")},
                "gpu": smi.stdout.strip() if smi else None, "instance": imds("instance-data")}

    def controller(self):
        """scripts/nebius_validation.py ``watch --controller`` until it leaves (idle) or the deadline deletes the VM."""
        self.set_phase("controller")
        env = {**os.environ, "ARC_NEBIUS_CONFIG": str(STATE / "launch/config.json")}
        with (self.output / "controller.log").open("ab") as log:
            self.process = subprocess.Popen(
                [sys.executable, "-u", str(STATE / "launch/nebius_validation.py"), "watch", "--controller",
                 "--interval", str(self.spec.get("interval", 60))],
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = self.process.wait()
        self.identity["exit_code"] = code
        if code:
            raise RuntimeError(f"controller exited {code}; inspect controller.log")

    def run_bootstrap(self):
        spec, output = self.spec, self.output
        remaining = spec["deadline_epoch"] - time.time() - spec["finalize_allowance"]
        if remaining < spec["timeout"]:
            raise RuntimeError(f"setup used its allowance: {remaining:.0f} s left for a {spec['timeout']} s run")
        if self.late_error:
            raise RuntimeError(self.late_error)
        KAGGLE.mkdir(parents=True, exist_ok=True)
        (KAGGLE / "input").symlink_to(ASSETS / "flash-input", target_is_directory=True)
        command = job_command(spec, output)
        env = job_environment(os.environ, spec, output)
        started = time.time()
        self.identity.update(bootstrap_started_epoch=started, command=command)
        self.set_phase("bootstrap")

        def pump(pipe, log):
            for line in iter(pipe.readline, b""):
                log.write(line)
                log.flush()
                sys.stdout.buffer.write(line)
                sys.stdout.flush()

        with (output / "bootstrap.log").open("wb") as log:
            self.process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            start_new_session=True)
            threads = [threading.Thread(target=pump, args=(self.process.stdout, log), daemon=True),
                       threading.Thread(target=self.stop_watcher, daemon=True)]
            if spec.get("stop_rule"):
                threads.append(threading.Thread(target=self.target_watcher, args=(started,), daemon=True))
            for thread in threads:
                thread.start()
            try:
                self.process.wait(timeout=max(1, started + spec["timeout"] - 20 - time.time()))
            finally:
                if self.interrupted and spec.get("checkpoint") and self.process.poll() is None:
                    self.identity["final_checkpoint"] = self.final_checkpoint()  # before the trial is killed
                if self.process.poll() is None:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
                threads[0].join(timeout=10)
        self.identity["exit_code"] = self.process.returncode
        if self.late_error:  # the bootstrap was stopped for it: this is the error
            raise RuntimeError(self.late_error)
        if self.process.returncode:
            raise RuntimeError(f"Bootstrap exited {self.process.returncode}; inspect {output}/bootstrap.log")

    # -- lifecycle ------------------------------------------------------------------------------------------------
    def on_signal(self, signum, frame):
        if self.finishing or self.interrupted:  # a signal during the final sync must not abort it
            return
        self.interrupted = {"signal": signal.Signals(signum).name, "epoch": round(time.time(), 1), "phase": self.phase}
        raise Interrupted()

    def execute(self):
        spec = self.spec
        self.output.mkdir(parents=True, exist_ok=True)
        self.identity = {"run_id": self.run_id, "kind": spec["kind"], "test": spec["test"], "platform": "nebius",
                         "started_epoch": self.started, "gpu": spec["gpu"], "function_timeout": spec["timeout"],
                         "image": f"nebius:{spec.get('image_family')}", "attempt": spec.get("attempt", 1)}
        state = "failed"
        signal.signal(signal.SIGTERM, self.on_signal)
        signal.signal(signal.SIGINT, self.on_signal)
        syncer = threading.Thread(target=self.syncer, daemon=True)
        cancelled = self.launch_cancelled()
        if cancelled:
            # The launcher gave this VM up (create timeout / no capacity) and recorded no execution: never run, and
            # leave the folder's markers alone (a later attempt may share it). The unit ends and deletes the VM.
            self.put_json("cancelled-boot.json", {"epoch": round(time.time(), 1), "instance_id": self.instance(),
                                                  "marker": cancelled})
            print(json.dumps({"event": "nebius_cancelled", "marker": cancelled}), flush=True)
            return 1
        try:
            self.set_phase("boot")
            syncer.start()
            if spec.get("mode") == "preflight":
                self.identity["preflight"] = self.preflight()
                state = "preflight_ok"
            elif spec.get("mode") == "controller":
                self.controller()
                state = "complete"
            elif spec.get("mode") == "bake":
                self.set_phase("assets")
                self.identity["assets"] = self.fetch_assets()
                BAKED.write_text(json.dumps(self.manifest_digests, indent=1, sort_keys=True))
                BAKE_KEEP.parent.mkdir(parents=True, exist_ok=True)
                BAKE_KEEP.write_text(f"{self.run_id}\n")
                os.sync()
                state = "baked"
            else:
                self.identity["calibration"] = self.calibrate_vm()
                self.set_phase("assets")
                self.identity["assets"] = self.fetch_assets()
                if spec.get("inputs"):
                    self.identity["inputs"] = self.fetch_inputs()
                self.extract_bundle()
                if spec.get("image"):
                    self.set_phase("image")
                    self.identity["docker"] = prepare_docker(spec["image"])
                if spec.get("checkpoint") and spec.get("attempt", 1) > 1:
                    self.identity["restored"] = self.restore_working()
                self.run_bootstrap()
                state = "complete"
        except Interrupted:
            state = "interrupted"
        except subprocess.TimeoutExpired as exc:
            state, self.identity["error"] = "timeout", f"TimeoutExpired: {exc}"
        except BaseException as exc:  # noqa: BLE001 - every failure is recorded, then the unit ends and the VM goes
            self.identity["error"] = f"{type(exc).__name__}: {str(getattr(exc, 'stderr', None) or exc)[:2000]}"
        finally:
            self.finish(state)
        return 0 if state in ("complete", "preflight_ok", "baked") else 1

    def finish(self, state):
        budget_end = time.time() + (INTERRUPT_BUDGET if self.interrupted else self.spec["finalize_allowance"] - 30)
        left = lambda: max(5, budget_end - time.time())  # noqa: E731
        self.finishing = True
        self.stop.set()
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
        if self.spec.get("image") and shutil.which("docker"):  # killing the client leaves the container running
            subprocess.run(["docker", "rm", "-f", IMAGE_CONTAINER], capture_output=True, timeout=60)
        self.identity.update(elapsed_seconds=time.time() - self.started, ended_epoch=time.time())
        final = {"state": state, **{k: self.identity.get(k) for k in (
            "run_id", "exit_code", "error", "started_epoch", "bootstrap_started_epoch", "ended_epoch",
            "elapsed_seconds", "attempt", "restored", "final_checkpoint")}}
        if self.spec.get("checkpoint") and (WORKING / "trials").is_dir():
            # A checkpoint the trial could not continue: the launcher restarts from zero instead of giving up.
            final["resume_failed"] = any("ResumeError" in p.read_text(errors="replace")
                                         for p in (WORKING / "trials").glob("*/exception.txt"))
            # What a relaunch would resume from: the checkpoint first (it is the work), then the marker.
            final["checkpoint"] = self.sync_checkpoints(timeout=min(20, left()) if self.interrupted else 600)
        if self.interrupted:
            # Preemption sends SIGTERM 60 s before the stop: the marker next, evidence with what is left.
            final["interrupted"] = {**self.interrupted, "instance_events": imds("instance-events", timeout=3)}
            self.put_json("final.json", final, timeout=15)
        serving = WORKING / "serving"
        if serving.exists():
            dest = self.output / "arena/serving"
            dest.mkdir(parents=True, exist_ok=True)
            for f in serving.iterdir():
                if f.is_file() and f.suffix in {".json", ".log"}:
                    shutil.copyfile(f, dest / f.name)
        (self.output / "execution.json").write_text(json.dumps(self.identity, indent=2, default=str))
        for log in ("/var/log/arc/bootstrap.log", "/var/log/arc/runner.log", "/var/log/cloud-init-output.log"):
            if Path(log).is_file():
                shutil.copyfile(log, self.output / "nebius" / Path(log).name)
        if self.interrupted and (WORKING / "trials").is_dir():
            try:
                self.rclone("copy", str(WORKING / "trials"), f"{self.remote_runs}/arena/working/trials", "--update",
                            "--use-server-modtime", "--max-size", "256M", "--transfers", "16",
                            "--exclude", "*/checkpoint/**", timeout=left() / 2)
            except (subprocess.SubprocessError, OSError):
                pass
        final["synced"] = self.sync(full=True, timeout=left())
        self.put_json("final.json", final, timeout=20)


def main(argv=None):
    args = argv or sys.argv[1:]
    if args[0] == "--calibrate":
        calibrate(args[1])
        return 0
    spec = json.loads(Path(args[0]).read_text())
    return Runner(spec).execute()


if __name__ == "__main__":
    raise SystemExit(main())
