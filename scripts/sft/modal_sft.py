"""Modal entry for the SFT trainer (trunk/sft): one B300 training run, and its CPU preflight on the same image.

    modal run --detach scripts/sft/modal_sft.py --config launch/sft/<run>.json             # training
    modal run --detach scripts/sft/modal_sft.py --config launch/sft/<run>.json --pf  # CPU preflight, spawned; result in .runops/<run>/preflight.json
    modal run scripts/sft/modal_sft.py --config launch/sft/<run>.json --dry  # typed-config dry run only (launch gate)
    modal run scripts/sft/modal_sft.py --tests tests/test_sft_grids.py[,...]  # pytest on CPU, Triton interpreted

The config holds ``run`` and ``train`` (train.Config fields; paths on the volumes: /runs = arc-prolong-runs,
/assets = arc-prolong-assets). Live knobs: /runs/validation/<run>/train/knobs.json.
"""
import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if modal.is_local() else HERE  # the container holds this file at /root
AXOLOTL_IMAGE = "axolotlai/axolotl@sha256:af1ff77a35170b794d5fd3d01d97d69fe11ba9a098bca43eabe0849ff3d2b010"
PY = "/workspace/axolotl-venv/bin/python"
PIP = ("--no-deps triton==3.6.0",)  # the selected-attention kernel's Triton (as the heal trainer)

app = modal.App("arc-sft", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
# per-run volumes: launch.sh sets ARC_SFT_VOLUME (arc-run-<run>, v2, at /runs) and
# ARC_SFT_INPUT_VOLUMES (read-only at /in/<vol>/); the compile cache then lives on the shared v2 arc-cache at /cache.
# Unset: arc-prolong-runs at /runs, as before. The image env carries both so the container's volume objects match.
SHARED_VOLUME = "arc-prolong-runs"
RUN_VOLUME = os.environ.get("ARC_SFT_VOLUME") or SHARED_VOLUME
PER_RUN = RUN_VOLUME != SHARED_VOLUME
INPUT_VOLUMES = [v for v in (os.environ.get("ARC_SFT_INPUT_VOLUMES") or "").split(",") if v]
CACHE_ROOT = "/cache" if PER_RUN else "/runs"
runs = modal.Volume.from_name(RUN_VOLUME, **({"create_if_missing": True, "version": 2} if PER_RUN else {}))
cache_vol = modal.Volume.from_name("arc-cache", create_if_missing=True, version=2) if PER_RUN else None
VOLUMES = {"/runs": runs, "/assets": assets, **({"/cache": cache_vol} if PER_RUN else {}),
           **{f"/in/{v}": modal.Volume.from_name(v).read_only() for v in INPUT_VOLUMES}}
VOL_ENV = {"ARC_SFT_VOLUME": RUN_VOLUME, "ARC_SFT_INPUT_VOLUMES": ",".join(INPUT_VOLUMES)} if PER_RUN else {}


def commit_volumes():
    runs.commit()
    if cache_vol is not None:
        cache_vol.commit()


image = (modal.Image.from_registry(AXOLOTL_IMAGE, add_python="3.12").entrypoint([])
         .apt_install("build-essential").env({"PYTHONUNBUFFERED": "1"}))
for command in PIP:
    image = image.run_commands(f"uv pip install --python {PY} " + " ".join(f"'{x}'" for x in command.split()))
image = (image.add_local_dir(str(HERE), "/opt/sft/scripts/sft", copy=True)
         .add_local_dir(str(ROOT / "tests"), "/opt/sft/tests", copy=True)
         .add_local_dir(str(ROOT / "src/arc_infra"), "/opt/sft/src/arc_infra", copy=True)
         .add_local_file(__file__, "/root/modal_sft.py", copy=True))
if VOL_ENV:
    image = image.env(VOL_ENV)
# the preflight's image: the shared CPU image (torch/transformers as the trainer's, no Triton/fla, which need a GPU
# driver) plus the trainer image's torchao, so the tiny run steps the real 8-bit AdamW
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src") if modal.is_local() else "/opt/sft/src")
from arc_infra.runops.ciimage import image as ci_image  # noqa: E402

pf_image = (ci_image(extra=("torchao==0.18.0",)).add_local_dir(str(HERE), "/opt/sft/scripts/sft", copy=True)
            .add_local_dir(str(ROOT / "tests"), "/opt/sft/tests", copy=True)
            .add_local_dir(str(ROOT / "launch/sft"), "/opt/sft/launch/sft", copy=True)
            .add_local_dir(str(ROOT / "src/arc_infra"), "/opt/sft/src/arc_infra", copy=True)
            .add_local_file(__file__, "/root/modal_sft.py", copy=True))
if VOL_ENV:
    pf_image = pf_image.env(VOL_ENV)
# --tests: the preflight's image plus Triton, its kernels run under TRITON_INTERPRET=1 (the served kv4 writer, INT8
# dequant and the fused int3 grid in tests/test_sft_grids.py)
tests_image = (ci_image(extra=("torchao==0.18.0", "triton==3.6.0"))
               .add_local_dir(str(HERE), "/opt/sft/scripts/sft", copy=True)
               .add_local_dir(str(ROOT / "tests"), "/opt/sft/tests", copy=True)
               .add_local_dir(str(ROOT / "launch/sft"), "/opt/sft/launch/sft", copy=True)
               .add_local_dir(str(ROOT / "src/arc_infra"), "/opt/sft/src/arc_infra", copy=True)
               .add_local_file(__file__, "/root/modal_sft.py", copy=True))


def _run(code, *args):
    import subprocess
    import sys

    env = {"PYTHONPATH": "/opt/sft/scripts/sft:/opt/sft/src", "PATH": "/usr/local/bin:/usr/bin:/bin"}
    return subprocess.run([sys.executable, "-c", code, *args], env=env, cwd="/opt/sft", text=True, capture_output=True)


TRAIN = """
import json, sys
import train
import vol
from arc_infra.runops.restart import RestartGuard
commit, reload, read = vol.client()  # the parent's volume ops (no modal in the trainer's venv)
spec = json.loads(sys.argv[1])
cfg = train.Config(**spec["train"])
guard = RestartGuard("/runs", spec["run"], commit=commit, policy=spec.get("restart_policy", "resume"))  # a preemption
# with a finalised checkpoint resumes (k256 main lost 47 steps to a 30 Oct preemption); anything else stays blocked
kw = dict(refresh=reload, commit=commit, guard=guard, knobs=lambda: read(cfg.out + "/knobs.json"))
if int(spec.get("gpus", 1)) > 1:  # one process per card (shard.py); rank 0 here keeps the volume bridge and guard
    import shard
    shard.launch(cfg, int(spec["gpus"]), **kw)
else:
    train.train(cfg, device="cuda:0", **kw)
"""


def run_train(spec, env):
    """The trainer child (TRAIN) with this container's volume ops served to it."""
    if "/opt/sft/scripts/sft" not in sys.path:
        sys.path.insert(0, "/opt/sft/scripts/sft")
    import vol

    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", **env}  # v5: 52 GiB reserved but unallocated at OOM
    return vol.run_child([PY, "-c", TRAIN, json.dumps(spec)], env, vol.modal_ops(runs), cwd="/opt/sft")


def compile_cache(root=None):
    """Inductor / Triton caches (torchao's per-shape optimizer graphs, the Triton kernels and their autotune picks) on
    the volume, one folder for every trainer code: both caches key entries by source, shapes and torch/Triton version
    (the image is pinned by digest), so a code change no longer makes the next boot compile and autotune cold."""
    base = f"{root or CACHE_ROOT + '/compile-cache'}/sft-shared"
    return {"TORCHINDUCTOR_CACHE_DIR": f"{base}/inductor", "TRITON_CACHE_DIR": f"{base}/triton",
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1", "TORCHINDUCTOR_AUTOGRAD_CACHE": "1", "TRITON_CACHE_AUTOTUNING": "1"}


LOCAL_CACHE = "/tmp/compile-cache"  # 28 Oct: step 1 spent 7.5 min loading Triton launchers from the degraded volume


def sync_cache(src, dst, limit_s=600, workers=32):
    """Copy the files of ``src`` that ``dst`` lacks (compile caches are content-keyed; entries are never rewritten).
    Train runs read and write the cache on local disk: copied in before the trainer starts, new entries copied back
    after it ends. Parallel (the degraded volume serves ~7 files/s to one reader); ``complete`` is False when the time
    limit stopped it (28 Oct lr probe: a partial copy left the Triton autotune picks behind, the GDN backward autotuned
    at peak memory and a config's launch ran out of memory)."""
    import shutil
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    if not Path(src).is_dir():
        return {"files": 0, "bytes": 0, "seconds": 0.0, "src": str(src), "missing": True, "complete": True}

    def one(f):
        if time.time() - t0 > limit_s:
            return None
        g = Path(dst) / f.relative_to(src)
        if not f.is_file() or (g.exists() and g.stat().st_size == f.stat().st_size):
            return 0
        g.parent.mkdir(parents=True, exist_ok=True)
        tmp = g.with_name(g.name + f".part{threading.get_ident()}")
        shutil.copyfile(f, tmp)
        tmp.replace(g)
        return f.stat().st_size

    files = sorted(Path(src).rglob("*"))
    with ThreadPoolExecutor(max(1, workers)) as pool:
        got = list(pool.map(one, files))
    done = [x for x in got if x]
    return {"files": len(done), "bytes": sum(done), "seconds": round(time.time() - t0, 1), "src": str(src),
            "complete": all(x is not None for x in got)}


def cache_tar_in(tar, dst):
    """Extract the shared cache tar (one volume file) into ``dst``; None when there is none yet."""
    import tarfile
    import time

    t0 = time.time()
    if not Path(tar).is_file():
        return None
    Path(dst).mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as archive:
        files = sum(m.isfile() for m in archive.getmembers())
        archive.extractall(dst, filter="data")
    return {"files": files, "bytes": Path(tar).stat().st_size, "seconds": round(time.time() - t0, 1),
            "src": str(tar), "complete": True}


def cache_tar_out(src, tar, before=0):
    """The local cache back to the volume as one tar (28 Oct: copying entries back added ~5.6K inodes/h to a volume at
    82% of its inode limit); skipped when the run added no entries. Last writer wins (entries are content-keyed)."""
    import os
    import tarfile

    files = [f for f in Path(src).rglob("*") if f.is_file()] if Path(src).is_dir() else []
    if len(files) <= before:
        return {"files": len(files), "written": False}
    tmp = Path(tar).with_name(Path(tar).name + f".part{os.getpid()}")
    Path(tar).parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tmp, "w") as archive:
        archive.add(src, arcname=".")
    tmp.replace(tar)
    return {"files": len(files), "written": True, "bytes": Path(tar).stat().st_size}


def cache_env(shared, local, limit_s=600):
    """The compile-cache env for a run: the local copy when the copy-in completed (the shared tar, else the legacy
    folder), else the volume cache itself (slow launcher loads, but every autotune pick present)."""
    got = cache_tar_in(Path(shared).with_name(Path(shared).name + ".tar"), Path(local) / "sft-shared")
    if got is None:
        got = sync_cache(shared, Path(local) / "sft-shared", limit_s)
    use_local = got["complete"]
    return (compile_cache(local) if use_local else compile_cache(str(Path(shared).parent))), use_local, got


# cpu=(8, 16) (p75 main: qat_refresh 8 -> 24 s and the forward up to 1.6x slower on a busy host with
# the default CPU share; the refresh is thousands of small launches)
# memory: token request with a ceiling; without one the cgroup limit is the placement's (1024 or 305.8 GiB), and 305.8
# leaves no room for the 70 GiB async snapshot (kd-c613 legs: inline, ~10 min stall an hour)
@app.function(image=image, gpu="B300", cpu=(8, 16), memory=(4096, 360 * 1024), timeout=24 * 3600, retries=0,
              volumes=VOLUMES)
def train_run(spec: dict):
    """The run (timeout = crash ceiling; training ends on its step count). Output streams to the Modal log."""
    _train(spec)


# a 360 GiB host-RAM request for the Kahan run (K1's BF16 comp states were OOM-killed above
# the default request); a spec selects it with "host_mem_gib": 360. No other run reserves RAM.
@app.function(image=image, gpu="B300", memory=360 * 1024, timeout=24 * 3600, retries=0,
              volumes=VOLUMES)
def train_run_mem360(spec: dict):
    _train(spec)


# N-GPU runs (spec "gpus": N, shard.py): one container, N B300s, one trainer process per card. CPU and RAM are token
# requests with ceilings: CPUs up to N x 16; host RAM up to HOST_GIB_PER_GPU per card (the
# n-gram table is shared; paged optimizer states and checkpoint snapshots are per rank).
HOST_GIB_PER_GPU = 360


@app.function(image=image, gpu="B300:2", cpu=(0.125, 32), memory=(4096, 2 * HOST_GIB_PER_GPU * 1024), timeout=24 * 3600,
              retries=0, volumes=VOLUMES)
def train_run_g2(spec: dict):
    _train(spec)


@app.function(image=image, gpu="B300:4", cpu=(0.125, 64), memory=(4096, 4 * HOST_GIB_PER_GPU * 1024), timeout=24 * 3600,
              retries=0, volumes=VOLUMES)
def train_run_g4(spec: dict):
    _train(spec)


@app.function(image=image, gpu="B300:8", cpu=(0.125, 128), memory=(4096, 8 * HOST_GIB_PER_GPU * 1024), timeout=24 * 3600,
              retries=0, volumes=VOLUMES)
def train_run_g8(spec: dict):
    _train(spec)


TRAIN_RUNS = {2: train_run_g2, 4: train_run_g4, 8: train_run_g8}


def _train(spec):
    import time

    env = {"PYTHONPATH": "/opt/sft/scripts/sft:/opt/sft/src", "PATH": "/workspace/axolotl-venv/bin:/usr/bin:/bin",
           "PYTHONUNBUFFERED": "1", "ARC_RESTART_ACK": spec.get("ack", ""), "ARC_T0": str(time.time()),
           "ARC_LAUNCH_ID": modal.current_function_call_id() or ""}  # the guard's call id: no modal in the child
    shared = Path(CACHE_ROOT) / "compile-cache" / "sft-shared"
    if spec.get("cache_seed") and not shared.exists():  # a smoke's warm cache (V1 volumes copy no folders server-side)
        import shutil

        shutil.copytree(spec["cache_seed"], shared)
    cache, local, got = cache_env(shared, LOCAL_CACHE)
    env.update(cache)
    print("COMPILE_CACHE_IN " + json.dumps({**got, "local": local}), flush=True)
    r = run_train(spec, env)
    if local:
        print("COMPILE_CACHE_OUT " + json.dumps(cache_tar_out(Path(LOCAL_CACHE) / "sft-shared", f"{shared}.tar",
                                                             got["files"])), flush=True)
    commit_volumes()
    if r.returncode:
        raise RuntimeError(f"trainer exited {r.returncode}")


DRYRUN = """
import json, sys
from pathlib import Path
import train
spec = json.loads(sys.argv[1]); cfg = train.Config(**spec["train"]); out = {}
student = Path(cfg.student); index = json.loads((student / "model.safetensors.index.json").read_text())["weight_map"]
config = json.loads((student / "config.json").read_text())
out["student_cut"] = config.get("student_cut", {}).get("source_layers") is not None
out["mtp_tensors"] = sum(k.startswith("mtp.") for k in index)
for part, root in (("train", Path(cfg.data)), ("heldout", Path(cfg.heldout_data or cfg.data))):
    every = json.loads((root / "index.json").read_text())[part]["files"]
    out_pairs = {tuple(x) for x in cfg.exclude_pairs}
    out_files = set(json.loads(Path(cfg.exclude_files).read_text())) if cfg.exclude_files and part == "train" else set()
    kept = [e for e in every if (e.get("run"), e.get("game")) not in out_pairs and e["file"] not in out_files]
    files = [e for e in kept if cfg.loss != "kl" or (root / "teacher" / e["file"]).is_file()]  # kl: scored only
    missing = [e["file"] for e in files if not (root / e["file"]).is_file()]
    out[part] = {"examples": len(files), "reply_tokens": sum(e["reply_tokens"] for e in files), "missing": len(missing),
                 "excluded": len(every) - len(kept), "unscored_skipped": len(kept) - len(files), "games": sorted({e.get("game") for e in files})}
    if part == "train":
        train_index = {"examples": files}
out["heldout"]["eval_examples"] = min(cfg.eval_examples, out["heldout"]["examples"])
steps_data = out["train"]["reply_tokens"] / cfg.batch_reply_tokens
out["epochs_at_total_steps"] = round(cfg.total_steps / max(steps_data, 1e-9), 2)
latest = train.latest(cfg.out); out["resume_from"] = str(latest) if latest else None
init_ok = True
if not latest and cfg.init_ckpt and not (Path(cfg.init_ckpt) / "state.json").is_file():
    out["init_ckpt_missing"], init_ok = cfg.init_ckpt, False  # a later leg's checkpoint not written yet
elif not latest and cfg.init_ckpt:  # resume from another run's checkpoint: its step, cursor and first batch
    st = json.loads((Path(cfg.init_ckpt) / "state.json").read_text())
    st["cursor"] = [0, 0] if cfg.init_fresh_data else st["cursor"]
    names = train.next_batch(cfg, train_index, tuple(st["cursor"]))[0]
    seen = set(train.order(train_index, cfg.seed, 0)[:st["cursor"][1]]) if st["cursor"][0] == 0 else None
    out.update(resume_from=cfg.init_ckpt, resume_step=st["step"], resume_cursor=st["cursor"],
               first_batch=len(names), first_batch_seen=None if seen is None else sum(n in seen for n in names))
    last = cfg.stop_step or cfg.total_steps  # eval-only from a checkpoint (eval_resume, stop == its step): no steps
    init_ok = ((Path(cfg.init_ckpt) / "optim.safetensors").is_file() or (Path(cfg.init_ckpt) / "shard.json").is_file()) and (last > st["step"] or (cfg.eval_resume and last == st["step"]))
# an ARC-3 SFT training run reads the ARC-2 held-out slice at its start, middle and end
trains = (cfg.stop_step or cfg.total_steps) > (out.get("resume_step") or 0) and not spec.get("arc2_exempt")
arc2_ok = not trains
if cfg.arc2_heldout_data:
    aroot = Path(cfg.arc2_heldout_data); afiles = json.loads((aroot / "index.json").read_text())["heldout"]["files"]
    out["arc2_heldout"] = {"examples": len(afiles), "tokens": sum(e["tokens"] for e in afiles), "tasks": len({e["game"] for e in afiles}),
                           "unscored": sum(not (aroot / "teacher" / e["file"]).is_file() for e in afiles), "steps": cfg.arc2_heldout_steps}
    arc2_ok = out["arc2_heldout"]["unscored"] == 0 and afiles and (len(cfg.arc2_heldout_steps) >= 2 or cfg.arc2_heldout_every > 0 or not trains)
out["arc2_rule"] = bool(arc2_ok)
gpus = int(spec.get("gpus", 1))  # a sharded run (shard.py): every rank holds >= 1 expert per layer
out["gpus"] = gpus
out["experts"] = config.get("text_config", config).get("num_experts")
gpus_ok = gpus == 1 or (gpus in (2, 4, 8) and (out["experts"] or 0) >= gpus and not cfg.channel_fisher
                        and not cfg.eval_dump and not cfg.kahan and (cfg.reduce_dtype == "fp32" or gpus == 2))
try:  # the predicted per-rank card peak against the measured one-GPU p75 reference (memest.py; headers only)
    import memest
    ex = train_index["examples"]  # the warm-up's example (train.warm_window): longest total, largest window
    w_total = max(int(e.get("tokens") or cfg.window) for e in ex)
    w_tail = max(min(int(e.get("tail") or cfg.window), cfg.window, int(e.get("tokens") or cfg.window)) for e in ex)
    out["mem"] = memest.fit(cfg.student, gpus, cfg.keep_layers, cfg.mem_budget_gib, fake_quant=cfg.fake_quant,
                            grid=cfg.qat_grid, paged=cfg.paged_states, depth=cfg.paged_depth, tokens=w_total, window=w_tail)
    out["predicted_peak_gib"] = out["mem"]["predicted_peak_gib"]
except Exception as error:  # noqa: BLE001 - a prediction, not a gate
    out["mem_error"] = repr(error)[:300]
out["ok"] = (out["student_cut"] and out["mtp_tensors"] > 0 and not out["train"]["missing"] and not out["heldout"]["missing"]
             and out["train"]["examples"] > 0 and out["heldout"]["examples"] > 0 and init_ok and gpus_ok and out["arc2_rule"])
print("DRYRUN " + json.dumps(out))
"""


SCOREDRY = """
import json, sys
from pathlib import Path
from transformers import AutoTokenizer
import teacher
spec = json.loads(sys.argv[1]); s = spec["score"]; out = {}
root, model = Path(s["data"]), Path(s["teacher"])
out["teacher_files"] = len(list(model.glob("*.safetensors")))
tok = AutoTokenizer.from_pretrained(str(model))
out["forced"] = {t: tok.convert_tokens_to_ids(t) for t in s["forced"]}
entries = teacher.order(root, s.get("priority"))
missing = [e["file"] for _, e in entries if not (root / e["file"]).is_file()]
done = [e for _, e in entries if (root / "teacher" / e["file"]).is_file()]
for part in ("heldout", "train"):
    es = [e for sp, e in entries if sp == part]
    out[part] = {"examples": len(es), "tokens": sum(int(e["tokens"]) for e in es)}
out["shards"] = [sum(int(e["tokens"]) for _, e in entries[i::s["cards"]]) for i in range(s["cards"])]
out.update(missing=len(missing), already_scored=len(done))
out["ok"] = (out["teacher_files"] > 0 and not missing and len(entries) > 0
             and all(isinstance(i, int) and i != tok.unk_token_id for i in out["forced"].values()))
print("DRYRUN " + json.dumps(out))
"""


# No CPU reservation: 8-16 reserved CPUs (leg2c, 26 Oct) left the 2-card host launch rate as low as without them
@app.function(image=image, gpu="B300", timeout=int(os.environ.get("SFT_SCORE_TIMEOUT", 3 * 3600)), retries=0, volumes=VOLUMES)
def score_run(spec: dict, shard: int):
    """One card of the teacher scoring pass (teacher.py): BF16 teacher with the fast paths, shard i of ``cards`` of the
    priority-ordered pool, each example written and committed when done; ends when its share is done or at
    ``deadline_s`` of scoring (timeout = crash ceiling)."""
    s = spec["score"]
    env = {"PYTHONPATH": "/opt/sft/scripts/sft:/opt/sft/src", "PATH": "/workspace/axolotl-venv/bin:/usr/bin:/bin",
           "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    shared = Path(CACHE_ROOT) / "compile-cache" / "sft-shared"
    cache, local, got = cache_env(shared, LOCAL_CACHE)
    env.update(cache)
    print("COMPILE_CACHE_IN " + json.dumps({**got, "local": local}), flush=True)
    args = [PY, "/opt/sft/scripts/sft/teacher.py", "--model", s["teacher"], "--data", s["data"], "--k", str(s["k"]),
            "--forced", *s["forced"], "--shard", f"{shard}/{s['cards']}", "--deadline-s", str(s["deadline_s"]),
            "--out", f"{s['out']}/shard{shard}", "--fast"]
    if s.get("dest"):  # a leg: files under <dest>/teacher, a size sweep first
        # one scratch dest per card: a card's --merge moves only its own files (another container's view is stale)
        args += ["--dest", s["dest"] if s["cards"] == 1 else f"{s['dest']}/shard{shard}"]
    if s.get("sizes"):
        args += ["--sizes", s["sizes"], "--per-size-s", str(s.get("per_size_s", 75))]
    if s.get("profile_at"):
        args += ["--profile-at", str(s["profile_at"])]
    if s.get("row_graph"):
        args += ["--row-graph"]
    if s.get("ab_blocks"):
        args += ["--ab-blocks", s["ab_blocks"], "--ab-block-s", str(s.get("ab_block_s", 120))]
    if s.get("files"):
        args += ["--files", ",".join(s["files"])]
    if s.get("merge"):
        args += ["--merge"]
    if s.get("priority"):
        args += ["--priority", s["priority"]]
    if s.get("local_io"):
        args += ["--local-io", "--local-io-ahead", str(s.get("local_io_ahead", 4))]
    if s.get("stage_inline"):
        args += ["--stage-inline"]
    if s.get("sanity_rec_lp_min") is not None:
        args += ["--sanity-rec-lp-min", str(s["sanity_rec_lp_min"])]
    if s.get("slow_host_floor"):
        args += ["--slow-host-floor", str(s["slow_host_floor"]), "--slow-host-after", str(s.get("slow_host_after", 10))]
    if "/opt/sft/scripts/sft" not in sys.path:
        sys.path.insert(0, "/opt/sft/scripts/sft")
    import vol

    # the whole log on the volume: sft-score-rlp-v1's SIGABRTs left no cause (Modal keeps the last 85 lines)
    r = vol.run_child(args, {**env, "PYTHONFAULTHANDLER": "1"}, vol.modal_ops(runs), cwd="/opt/sft",
                      log_path=f"{s['out']}/shard{shard}/child.log")
    if local:
        print("COMPILE_CACHE_OUT " + json.dumps(cache_tar_out(Path(LOCAL_CACHE) / "sft-shared", f"{shared}.tar",
                                                             got["files"])), flush=True)
    commit_volumes()
    if r.returncode:
        raise RuntimeError(f"scorer {shard} exited {r.returncode}")


LAYERS = """
import json, sys
import layers, train
spec = json.loads(sys.argv[1]); s = spec["layers"]
fast = s.get("fast", True)  # the trainer's kernels (sft-layers-p75-v1: the reference GDN/conv paths ran > 23 min)
model, _ = train.load_model(s["student"], "cuda:0", fast, fast_load=fast, sel_cache=fast, gdn_fast=fast, gdn_keep=fast,
                            hyper_fast=fast, union=fast)
model.eval()
extra = {}
if s.get("trained_format"):
    import qat
    info = layers.trained_format(model, s.get("qat_grid", "int4"))
    print("TRAINED_FORMAT", {k: v for k, v in info.items() if k != "calls"}, flush=True)
    extra = {"weight": qat.int8_rows(model.get_output_embeddings().weight.detach()), "calls": info["calls"]}
data = layers.windows_data(s["windows"], "cuda:0", tokens=s.get("tokens", 4096), scored=s.get("scored", 2048))
layers.run(model, data, s["out"], depth=s.get("depth", 8), lazy=s.get("lazy", 6), cost_full=s.get("cost_full", 2.0),
           sets=s.get("sets"), **extra)
"""

LAYERSDRY = """
import json, sys
from pathlib import Path
spec = json.loads(sys.argv[1]); s = spec["layers"]; out = {}
student = Path(s["student"]); config = json.loads((student / "config.json").read_text())
text = config.get("text_config", config)
out["layers"] = text.get("num_hidden_layers"); out["types"] = text.get("layer_types")
w = json.loads(Path(s["windows"]).read_text())["windows"]
out["windows"] = sum(x["index"] % 4 == 1 for x in w)
out["out_exists"] = (Path(s["out"]) / "layers.json").exists()
names = set(range(out["layers"] or 0))
out["ok"] = (bool(out["layers"]) and out["windows"] > 0 and not out["out_exists"]
             and all(int(i) in names for m in (s.get("sets") or {}).values() for i in m) and 0 < s.get("depth", 8) < out["layers"])
print("DRYRUN " + json.dumps(out))
"""


@app.function(image=image, gpu="B300", timeout=50 * 60, retries=0, volumes=VOLUMES)
def layers_run(spec: dict):
    """Greedy depth-cut layer choice (layers.py) on one student; <out>/layers.json and events.jsonl on the volume."""
    env = {"PYTHONPATH": "/opt/sft/scripts/sft:/opt/sft/src", "PATH": "/workspace/axolotl-venv/bin:/usr/bin:/bin",
           "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    import subprocess

    Path(spec["layers"]["out"]).mkdir(parents=True, exist_ok=True)
    with open(Path(spec["layers"]["out"]) / "layers.log", "w") as log:
        r = subprocess.run([PY, "-c", LAYERS, json.dumps(spec)], env=env, cwd="/opt/sft", stdout=log,
                           stderr=subprocess.STDOUT, text=True)
    commit_volumes()
    if r.returncode:
        raise RuntimeError(f"layers exited {r.returncode}")


@app.function(image=pf_image, cpu=(8, 16), timeout=45 * 60, retries=0, volumes=VOLUMES)
def preflight(spec: dict, dry_only: bool = False):
    """CPU preflight (shared CPU image + torchao): the run's typed config against the volumes (student, data, resume point),
    then the goldens and the trainer test (tiny student: start, checkpoint, resume reproducing the next step, MTP
    isolation) with the image's torchao 8-bit AdamW."""
    import subprocess

    dry = _run(SCOREDRY if "score" in spec else LAYERSDRY if "layers" in spec else DRYRUN, json.dumps(spec))
    line = next((l for l in dry.stdout.splitlines() if l.startswith("DRYRUN ")), None)
    result = {"dryrun": json.loads(line[7:]) if line else {"ok": False, "error": (dry.stdout + dry.stderr)[-2000:]}}
    if dry_only:  # the blocking launch gate (<=5 min); the full preflight runs beside the run
        return result["dryrun"]
    import sys

    t = subprocess.run([sys.executable, "-c", "import torchao; print(torchao.__version__)"], capture_output=True, text=True)
    result["torchao"] = t.stdout.strip() or t.stderr.strip()[-300:]
    r = subprocess.run([sys.executable, "-m", "pytest", "-v", "-rfE", "-p", "no:cacheprovider", "tests/golden", "tests/test_sft_train.py",
                        "tests/test_sft_speed.py", "tests/test_sft_qat.py", "tests/test_sft_grids.py",
                        "tests/test_sft_export.py", "tests/test_sft_layers.py", "tests/test_sft_ngpu.py"],
                       cwd="/opt/sft", text=True, capture_output=True,
                       env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp"})
    lines = (r.stdout + r.stderr).strip().splitlines()
    result["tests"] = lines[-1:] + [l[:300] for l in lines if l.startswith(("FAILED", "ERROR", "E   "))][:30]
    if r.returncode:  # an aborted process leaves no summary: the tail shows the test it died in
        result["tail"] = [l[:300] for l in lines[-25:]]
    result["ok"] = bool(result["dryrun"].get("ok")) and r.returncode == 0 and bool(t.stdout.strip())
    path = Path(f"/runs/.runops/{spec['run']}/preflight.json")  # the result, whatever becomes of the client
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    runs.commit()
    return result


@app.function(image=tests_image, cpu=(0.125, 16), timeout=45 * 60, retries=0)
def pytest_cpu(args: list):
    """pytest (CPU, Triton interpreted) with ``args`` as its command line from /opt/sft: output streams to the log;
    returns per-test results, the summary and the failures."""
    import subprocess
    import sys

    p = subprocess.Popen([sys.executable, "-m", "pytest", "-v", "-rfE", "-p", "no:cacheprovider", *args], cwd="/opt/sft",
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "TRITON_INTERPRET": "1"})
    lines = []
    for x in p.stdout:
        print(x, end="", flush=True)
        lines.append(x.rstrip("\n"))
    code = p.wait()
    keep = [x[:300] for x in lines if any(k in x for k in (" PASSED", " FAILED", " SKIPPED", " ERROR"))
            or x.startswith(("E   ", "FAILED"))]
    return {"ok": code == 0, "lines": keep + lines[-3:], "tail": [x[:300] for x in lines[-60:]] if code else []}


@app.function(image=image, gpu="B300", timeout=45 * 60, retries=0, volumes=VOLUMES)
def smoke_run(spec: dict):
    """smoke.py on one B300: single-layer kernel checks (own Triton cache), then trainer boot 1 (cold compile cache) and
    boot 2 (resumes boot 1's checkpoint, warm cache, ``smoke.boot2`` overrides), the one-card teacher speed if time
    allows, then the bars -> <out>/smoke.json. ``smoke.phases`` picks a subset (default kernels, boots, teacher)."""
    import subprocess
    import time

    began = time.time()
    out = spec["train"]["out"]
    cache = compile_cache(f"{CACHE_ROOT}/compile-cache/{spec['run']}-{int(time.time())}")
    env = {"PYTHONPATH": "/opt/sft/scripts/sft:/opt/sft/src", "PATH": "/workspace/axolotl-venv/bin:/usr/bin:/bin",
           "PYTHONUNBUFFERED": "1", "ARC_RESTART_ACK": spec.get("ack", "")}
    kj = f"{out}/kernels.json"
    smoke = "/opt/sft/scripts/sft/smoke.py"
    phases = spec["smoke"].get("phases", ["kernels", "boots", "teacher"])
    if "kernels" in phases:
        subprocess.run([PY, smoke, "kernels", json.dumps({**spec, "kernels_json": kj})], cwd="/opt/sft",
                       env={**env, "TRITON_CACHE_DIR": "/tmp/triton-kernels"})
        commit_volumes()
    tri = Path(cache["TRITON_CACHE_DIR"])

    def files():
        return sum(1 for _ in tri.rglob("*")) if tri.exists() else 0

    counts = []
    for boot, extra in ((1, {}), (2, spec["smoke"]["boot2"])) if "boots" in phases else ():
        before = files()
        sub = {**spec, "run": f"{spec['run']}-boot{boot}", "train": {**spec["train"], **extra}}
        r = run_train(sub, {**env, **cache, "ARC_T0": str(time.time())})
        commit_volumes()
        counts.append(files() - before)
        if r.returncode:
            break
    if "teacher" in phases and time.time() - began < spec["smoke"]["teacher_after_s"]:  # the job stays in its 30 min
        subprocess.run([PY, smoke, "teacher", json.dumps({**spec, "teacher_json": f"{out}/teacher.json"})],
                       cwd="/opt/sft", env=env)
    subprocess.run([PY, smoke, "report", json.dumps({"out": out, "kernels_json": kj, "triton": counts})], cwd="/opt/sft",
                   env=env)
    commit_volumes()


@app.local_entrypoint()
def main(config: str = "", pf: bool = False, smoke: bool = False, dry: bool = False, tests: str = ""):
    if tests:  # CPU tests: --tests "tests/a.py,tests/b.py" or a pytest command line "-q -s tests/a.py -k x"
        import shlex

        print("TESTS " + json.dumps(pytest_cpu.remote(shlex.split(tests.replace(",", " "))), indent=1))
        return
    spec = json.loads(Path(config).read_text())
    sys.path.insert(0, str(HERE))
    import vol

    want, inputs, problems = vol.volume_plan(spec)  # the record's volume and the mounts agree (launch.sh sets the env)
    if problems or (want, inputs) != (RUN_VOLUME, INPUT_VOLUMES):
        raise SystemExit(f"VOLUME MISMATCH {spec['run']}: config wants {want} + inputs {inputs}, this process mounts "
                         f"{RUN_VOLUME} + {INPUT_VOLUMES} (launch through launch/sft/launch.sh){problems or ''}")
    if os.environ.get("ARC_RESTART_ACK"):  # a relaunch after a stop or death names its cause
        spec["ack"] = os.environ["ARC_RESTART_ACK"]
    if dry:
        print("DRYRUN " + json.dumps(preflight.remote(spec, True)))
        return
    if pf:
        # spawned: two held clients timed out on v1c with no result (poll <volume>:.runops/<run>/preflight.json)
        print("PREFLIGHT_CALL " + preflight.spawn(spec).object_id, flush=True)
        return
    # GPU runs are spawned, never held: a held .remote()/.starmap() is cancelled by Modal when this client's connection
    # drops, even with --detach (26 Oct 18:15 and 18:49: util-v3, arc2 r7/r8). The call ids are printed for polling
    # by id (modal.FunctionCall.from_id); the app lives on with --detach.
    if "score" in spec:
        calls = [score_run.spawn(spec, i) for i in spec["score"].get("shards") or range(spec["score"]["cards"])]
    elif "layers" in spec:
        calls = [layers_run.spawn(spec)]
    else:
        mem, gpus = spec.get("host_mem_gib"), int(spec.get("gpus", 1))
        if mem not in (None, 360):
            raise SystemExit(f"host_mem_gib {mem}: only 360 (K1b) has a function")
        if gpus != 1 and (gpus not in TRAIN_RUNS or smoke or mem):
            raise SystemExit(f"gpus {gpus}: 1, or {sorted(TRAIN_RUNS)} for a training run (no smoke / host_mem_gib)")
        calls = [(smoke_run if smoke else train_run_mem360 if mem else TRAIN_RUNS[gpus] if gpus > 1
                  else train_run).spawn(spec)]
    print("LAUNCH " + json.dumps({"run": spec["run"], "calls": [c.object_id for c in calls]}), flush=True)
