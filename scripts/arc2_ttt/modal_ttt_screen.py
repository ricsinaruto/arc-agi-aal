"""arc2 TTT screen on Modal (decisions ea22abe69: 1 B300 <= 1 h; CPU gate first).

Model folder: the base NVFP4 checkpoint with the arc2 k256 INT3 overlay as complete shards
(``modal_fill_overlay.py``; ``arena.expert_overlay.materialise``) + the k256 router mask, the BF16 dense weights as
served. Code: ``ttt_screen.py`` (worker, Axolotl venv) over the RL trainer's loader pinned at rl0 ``RL_SHA`` and the
restart guard pinned at main ``RUNOPS_SHA`` (``git archive`` at image build, into dist/ttt-screen-code).

  modal run scripts/arc2_ttt/modal_ttt_screen.py --run-id R --parity P --mode dry        # local typed-config check
  modal run --detach scripts/arc2_ttt/modal_ttt_screen.py --run-id R --parity P --mode preflight   # CPU, tiny model
  modal run --detach scripts/arc2_ttt/modal_ttt_screen.py --run-id R --parity P --mode screen      # 1 B300

P: the RTX parity reference (``arc2.nll_parity`` parity.json). Outputs: /runs/validation/R/ on arc-prolong-runs."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal
import modal.exception
try:  # grpclib's stream end on a dropped connection (pf11, 26 Oct)
    from grpclib.exceptions import StreamTerminatedError as _StreamTerminated
except ImportError:  # pragma: no cover
    _StreamTerminated = OSError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[min(1, len(HERE.parents) - 1)]
RL_SHA = "28a838dfe"       # rl0: flash_qlora_job.load, rl1_trainer.install_expert_mask
RUNOPS_SHA = "ed704e65d"   # main: arc_infra.runops.RestartGuard
AXOLOTL_IMAGE = "axolotlai/axolotl@sha256:af1ff77a35170b794d5fd3d01d97d69fe11ba9a098bca43eabe0849ff3d2b010"  # = RL
RL_PIP = ("quack-kernels==0.6.1 nvidia-cutlass-dsl==4.6.0 apache-tvm-ffi>=0.1.10,<0.2", "--no-deps triton==3.6.0")
VENV = "/workspace/axolotl-venv/bin/python"
BASE = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
OVERLAY = "/assets/flash-input/datasets/richardcsaky/arc2-ttt-k256-int3-full-v1"
OVERLAY_SHA = "a453705817d8d6108ff7852708322d2465f7a2f50173208202d178d6bb03a84a"  # modal_fill_overlay.py, 26 Oct
MASK_SHA = "01514464af09bdd8bf5419758f0fa56a00c389a65438f4fba0392691e093633a"
BUNDLE = ROOT / "dist/arc2-rtx-c240-hx-base18-v3b"
CODE = ROOT / "dist/ttt-screen-code"
# r7: the 28 miss tasks only, 1 lr 3e-4, 1 epoch, packed at r6's measured batch 8 /
# 582.6 tok/s (no bench), sized in-run on the time left (stop if the 28 do not fit); pass = >= 6/37 exact alone
SETTINGS = {"lrs": [3e-4], "epochs": 1, "augments": 4, "max_tokens": 16384, "rank": 16, "reply_tasks": 1,
            "warmup": True, "kinds": ["miss"], "fixed": {"batch_size": 8, "rate": 582.6}, "pass_needs_nll": False}
SCREEN_TIMEOUT = 3600  # crash ceiling = the 1 h; the worker stops taking tasks at DEADLINE_S
DEADLINE_S = 3300      # r7: no new task after 55 min (largest miss task ~170 s at r6's pace)
END_MARGIN_S = 150     # the sizing counts to the timeout less this (summary, commit)
LAYERS = 1  # preflight real-weights model: layer 0 (GDN, FP32 router, NVFP4 routed + shared experts); 4 layers with the PLE table went past 160 GiB host (pf4)


def build_code():
    """dist/ttt-screen-code: rl1/ (rl0 scripts/posttrain), src/arc_infra (main), src/arc2, ttt worker, roster tasks."""
    import shutil

    if not (ROOT / ".git").exists() and not (ROOT / "src").exists():
        return
    shutil.rmtree(CODE, ignore_errors=True)
    (CODE / "rl1").mkdir(parents=True)
    for sha, path, dest in ((RL_SHA, "scripts/posttrain", CODE / "rl1"), (RUNOPS_SHA, "src/arc_infra", CODE / "src")):
        tar = subprocess.run(["git", "-C", str(ROOT), "archive", sha, path], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(CODE)], input=tar, check=True)
        if path == "scripts/posttrain":
            for f in (CODE / path).iterdir():
                f.rename(dest / f.name)
            shutil.rmtree(CODE / "scripts")
    keep = {"__init__.py", "runops"}
    for f in list((CODE / "src/arc_infra").iterdir()):
        if f.name not in keep:
            shutil.rmtree(f) if f.is_dir() else f.unlink()
    shutil.copytree(ROOT / "src/arc2", CODE / "src/arc2", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(ROOT / "src/arena/expert_overlay.py", CODE / "expert_overlay.py")
    for f in ("ttt_core.py", "ttt_screen.py", "screen_tasks.json"):
        shutil.copyfile(HERE / f, CODE / f)
    shutil.copyfile(BUNDLE / "configs/expert-prune/prune-arc2-k256-v1.json", CODE / "mask.json")
    roster = json.loads((HERE / "screen_tasks.json").read_text())
    (CODE / "tasks").mkdir()
    for t in set(roster["miss"]) | set(roster["hit"]) | set(roster["parity"]):
        shutil.copyfile(BUNDLE / f"data/arc2/evaluation/{t}.json", CODE / f"tasks/{t}.json")


if modal.is_local():
    build_code()

app = modal.App("arc2-ttt-screen")
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.from_registry(AXOLOTL_IMAGE, add_python="3.12").entrypoint([])
         .apt_install("libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1", "build-essential",
                      "ninja-build")
         .env({"PYTHONUNBUFFERED": "1"}))
for command in RL_PIP:
    image = image.run_commands(f"uv pip install --python {VENV} " + " ".join(f"'{x}'" for x in command.split()))
image = image.add_local_dir(str(CODE), "/w", ignore=["**/__pycache__/**"])


HUB_PROBE = """
from kernels import get_kernel
try:
    get_kernel("kernels-community/activation", version=1)
    print("HUB_OK loaded")
except Exception as exc:  # no CUDA build to load on CPU: the fetch itself is what the probe checks
    text = f"{type(exc).__name__}: {exc}"
    bad = "offline" in text.lower() or "not available in the local cache" in text
    print(("HUB_FAIL " if bad else "HUB_OK fetched; load: ") + text[:600])
"""


def header(path):
    import struct

    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        raw = fh.read(n)
    return raw


def folder(farm_dir):
    """materialise the farm (base + full overlay shards) and check it: every index key in its shard's header, every
    overlay shard's header byte-identical to the base shard's, the mask the overlay's."""
    import hashlib
    import importlib.util

    spec = importlib.util.spec_from_file_location("expert_overlay", "/w/expert_overlay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mask_sha = hashlib.sha256(Path("/w/mask.json").read_bytes()).hexdigest()
    assert mask_sha == MASK_SHA, f"mask sha {mask_sha}"
    record = mod.materialise(BASE, OVERLAY, Path(farm_dir), OVERLAY_SHA, check="size", expert_prune_sha256=MASK_SHA,
                             log=lambda *_: None)
    index = json.loads((Path(farm_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {}
    for key, shard in index.items():
        shards.setdefault(shard, []).append(key)
    missing, header_diff = [], []
    manifest = json.loads((Path(OVERLAY) / "overlay-manifest.json").read_text())
    for shard, keys in shards.items():
        raw = header(Path(farm_dir) / shard)
        have = set(json.loads(raw))
        missing += [k for k in keys if k not in have][:3]
        if shard in manifest["shards"] and raw != header(Path(BASE) / shard):
            header_diff.append(shard)
    out = {"record": record, "shards": len(shards), "keys": len(index), "missing": missing[:10],
           "overlay_header_diff": header_diff[:10], "overlay_omits_pruned": manifest.get("omits_pruned"),
           "pass": not missing and not header_diff and not manifest.get("omits_pruned")}
    print("FOLDER " + json.dumps(out), flush=True)
    return out


def truncated(farm_dir, dest, layers=LAYERS):
    """A real-weights copy of the farm with the first ``layers`` decoder layers: config, index and mask cut, every
    other file a link. The loader reads only the kept tensors (real dtypes: NVFP4 experts, FP32 routers, BF16 dense,
    FP8 n-gram table)."""
    import re
    import shutil

    dest = Path(dest)
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for item in Path(farm_dir).iterdir():
        if item.name not in ("config.json", "model.safetensors.index.json"):
            (dest / item.name).symlink_to(item.resolve())
    config = json.loads((Path(farm_dir) / "config.json").read_text())
    text = config["text_config"]
    text["num_hidden_layers"] = layers
    text["layer_types"] = text["layer_types"][:layers]
    text["ple_layer_ids"] = [i for i in text.get("ple_layer_ids", []) if i < layers]
    (dest / "config.json").write_text(json.dumps(config, indent=1))
    index = json.loads((Path(farm_dir) / "model.safetensors.index.json").read_text())
    keep = {}
    for key, shard in index["weight_map"].items():
        m = re.match(r"model\.language_model\.layers\.(\d+)\.", key)
        if m is None or int(m.group(1)) < layers:
            keep[key] = shard
    index["weight_map"] = keep
    (dest / "model.safetensors.index.json").write_text(json.dumps(index))
    mask = json.loads(Path("/w/mask.json").read_text())
    mask["pruned"] = {k: v for k, v in mask["pruned"].items() if int(k) < layers}
    (dest.parent / "mask-truncated.json").write_text(json.dumps(mask))
    return {"layers": layers, "keys": len(keep), "layer_types": text["layer_types"],
            "ple_layer_ids": text["ple_layer_ids"]}


def metrics_ok(path):
    """The watcher's row format ({"t", "m": {key: number}}) with the plan's keys present (r8's rows were flat)."""
    try:
        rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return False
    keys = {k for r in rows for k, v in (r.get("m") or {}).items() if isinstance(v, (int, float))}
    return bool(rows) and all(isinstance(r.get("m"), dict) for r in rows) and \
        {"ttt.tasks_done", "ttt.train_tok_s"} <= keys


def execute(run_id, parity_text, mode, skip=None):
    sys.path[:0] = ["/w/src"]
    from arc_infra.runops.restart import RestartGuard

    out = Path("/runs/validation") / run_id
    out.mkdir(parents=True, exist_ok=True)
    guard = RestartGuard("/runs", run_id, policy="stop", commit=runs.commit)
    with guard.run() as start:
        print("RUNOPS " + json.dumps({"attempt": start.attempt, "restart": start.restart}), flush=True)
        began = time.time()
        (out / "parity-reference.json").write_text(parity_text)
        check = folder("/tmp/farm")
        (out / "folder.json").write_text(json.dumps(check, indent=1))
        runs.commit()
        if not check["pass"]:
            raise RuntimeError(f"model folder check failed: {check}")
        cfg = {"mode": "flash" if mode == "screen" else "tiny", "model_dir": "/tmp/farm", "mask": "/w/mask.json",
               "tasks_dir": "/w/tasks", "roster": "/w/screen_tasks.json", "parity": str(out / "parity-reference.json"),
               "out": str(out), "deadline_epoch": began + DEADLINE_S,
               "end_epoch": began + SCREEN_TIMEOUT - END_MARGIN_S, **SETTINGS, "skip": list(skip or [])}
        if mode == "preflight":
            cfg.update(limit=2, reply_tasks=1, epochs=1, kinds=["miss", "hit"])
        if cfg["skip"]:  # a follow-up/resume run never replies (r9: task 1's generate took >14 min of B300)
            cfg["reply_tasks"] = 0
        (out / "config.json").write_text(json.dumps(cfg, indent=1))
        stop = threading.Event()

        def committer():
            while not stop.wait(60):
                try:
                    runs.commit()
                except Exception as exc:  # noqa: BLE001
                    print(f"commit failed: {exc!r}", flush=True)

        threading.Thread(target=committer, daemon=True).start()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHONPATH", "VIRTUAL_ENV"))}
        # hub online, as the RL trainer: the kernels plugin fetches kernels-community/* at load (screen 1 died offline)
        env.update(PYTHONPATH="/w/rl1:/w/src:/w", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="0",
                   AXOLOTL_SONICMOE_NVFP4_BACKEND="dequant", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        if mode == "preflight":  # the Flash load's hub fetches (kernels plugin), with the worker's own env
            offline = subprocess.run([VENV, "-c", HUB_PROBE], env={**env, "HF_HUB_OFFLINE": "1"}, capture_output=True,
                                     text=True, timeout=600)  # screen 1's failure, reproduced: must fail
            probe = subprocess.run([VENV, "-c", HUB_PROBE], env=env, capture_output=True, text=True, timeout=600)
            hub = {"exit": probe.returncode, "tail": (probe.stdout + probe.stderr)[-1500:],
                   "offline_tail": (offline.stdout + offline.stderr)[-600:]}
            (out / "hub.json").write_text(json.dumps(hub, indent=1))
            print("HUB " + json.dumps(hub), flush=True)
        def worker(config_path, log_name):
            with (out / log_name).open("w") as log:
                proc = subprocess.Popen([VENV, "-u", "/w/ttt_screen.py", "--config", str(config_path)], env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return proc.wait()

        real = None
        if mode == "preflight":  # the real weights' dtypes and kernels on CPU: forward, backward, generate
            info = truncated("/tmp/farm", "/tmp/farm4/model")
            rcfg = {**cfg, "mode": "real-cpu", "model_dir": "/tmp/farm4/model", "mask": "/tmp/farm4/mask-truncated.json",
                    "out": str(out / "real"), "limit": 1, "augments": 1, "lrs": cfg["lrs"][:1], "reply_tasks": 1}
            (out / "real").mkdir(exist_ok=True)
            (out / "real/config.json").write_text(json.dumps(rcfg, indent=1))
            rcode = worker(out / "real/config.json", "real/worker.log")
            block = json.loads((out / "real/real-block.json").read_text()) if (out / "real/real-block.json").exists() \
                else {}
            real = {"truncated": info, "exit": rcode, **block}
            (out / "real.json").write_text(json.dumps(real, indent=1, default=str))
            print("REAL " + json.dumps(real, default=str), flush=True)
        code = worker(out / "config.json", "worker.log")
        stop.set()
        result = {"run_id": run_id, "mode": mode, "exit": code, "elapsed_s": round(time.time() - began, 1),
                  "folder_pass": check["pass"]}
        for name in ("summary.json", "parity.json", "loaded.json"):
            if (out / name).exists():
                result[name.split(".")[0]] = json.loads((out / name).read_text())
        if mode == "preflight":
            rows = [json.loads(x) for x in (out / "tasks.jsonl").read_text().splitlines()] if \
                (out / "tasks.jsonl").exists() else []
            result["checks"] = {
                "worker_exit_0": code == 0, "tasks_2": len(rows) == 2,
                "both_arms_read": all(set(q["arms"]) == {f"{lr:g}" for lr in SETTINGS["lrs"]}
                                      for r in rows for q in r["queries"]),
                "reply_ran": (not rows[0].get("reply")) if cfg["skip"] else
                bool(rows) and all("error" not in v for v in rows[0].get("reply", {"x": {"error": 1}}).values()),
                "followup_replies_off": (not cfg["skip"]) or cfg["reply_tasks"] == 0,
                "parity_ids_same": bool((result.get("parity") or {}).get("rows")) and all(r["same_ids"] for r in result["parity"]["rows"]),
                "summary": "summary" in result, "metrics": (out / "metrics.jsonl").exists(),
                "metrics_runops_format": metrics_ok(out / "metrics.jsonl"),
                "mixed_dtype_reproduced": "dtype" in str((result.get("loaded") or {}).get("plain_forward_mixed_dtype")),
                "real_block_exit_0": bool(real) and real["exit"] == 0, "real_block_pass": bool(real) and (bool(real.get("pass")) or (
                    "requires CUDA" in str(real.get("error")) and bool(real.get("ids_same")))),  # router passed; experts are CUDA-only (pf6)
                "real_block_plain_forward_fails": bool(real) and "dtype" in str(real.get("plain_forward")),
                "warmup_go": (out / "warmup.json").exists() and json.loads((out / "warmup.json").read_text())["decision"] == "go",
                "batched_train": bool(rows) and all(v["steps"] > 0 for r in rows for v in r["train"].values()),
                "hub_fetch": "HUB_OK" in json.loads((out / "hub.json").read_text())["tail"],
                "hub_offline_reproduced": "HUB_FAIL" in json.loads((out / "hub.json").read_text())["offline_tail"]}
            result["pass"] = all(result["checks"].values())
        (out / "result.json").write_text(json.dumps(result, indent=1, default=str))
        runs.commit()
        print("RESULT " + json.dumps({k: v for k, v in result.items() if k not in ("parity", "loaded")}, default=str),
              flush=True)
        print("END", flush=True)
        if code != 0:
            raise RuntimeError(f"worker exit {code}")
        return result


@app.function(image=image, volumes={"/assets": assets, "/runs": runs}, cpu=(8, 16), memory=(2048, 240 * 1024),
              timeout=2700, retries=0)
def preflight(run_id: str, parity_text: str, skip: list = None):
    return execute(run_id, parity_text, "preflight", skip)


@app.function(image=image, volumes={"/assets": assets, "/runs": runs}, gpu="B300", cpu=(0.125, 16),
              timeout=SCREEN_TIMEOUT, retries=0, max_containers=1)
def screen(run_id: str, parity_text: str, skip: list = None):
    return execute(run_id, parity_text, "screen", skip)


def dry(parity_text):
    """Typed-config dry run (local): roster, tasks, parity rows, mask, pinned shas, code dir, worker compiles."""
    import hashlib
    import py_compile

    roster = json.loads((HERE / "screen_tasks.json").read_text())
    par = json.loads(parity_text)["rows"] if parity_text else []
    checks = {
        "overlay_sha_pinned": len(OVERLAY_SHA) == 64,
        "mask_sha": hashlib.sha256((CODE / "mask.json").read_bytes()).hexdigest() == MASK_SHA,
        "roster_28_10": (len(roster["miss"]), len(roster["hit"])) == (28, 10),
        "miss_outputs_37": sum(len(v) for v in roster["miss_outputs"].values()) == 37,
        "tasks_present": all((CODE / f"tasks/{t}.json").exists() for t in roster["miss"] + roster["hit"]),
        "parity_rows_5": [r["task"] for r in par] == roster["parity"],
        "parity_lengths": all(len(r["logprobs"]) == len(r["full_ids"]) - len(r["prompt_ids"]) > 0 for r in par),
        "rl_code": all((CODE / f"rl1/{f}").exists() for f in ("flash_qlora_job.py", "rl1_trainer.py", "rl_job.py")),
        "runops": (CODE / "src/arc_infra/runops/restart.py").exists(),
        "lrs_epochs": SETTINGS["epochs"] >= 1 and all(0 < lr < 1e-2 for lr in SETTINGS["lrs"]),
        "timeout_le_1h": SCREEN_TIMEOUT <= 3600 and DEADLINE_S < SCREEN_TIMEOUT,
    }
    for f in ("ttt_core.py", "ttt_screen.py", "rl1/flash_qlora_job.py", "rl1/rl1_trainer.py"):
        try:
            py_compile.compile(str(CODE / f), doraise=True)
            checks[f"compile_{f}"] = True
        except py_compile.PyCompileError:
            checks[f"compile_{f}"] = False
    sys.path[:0] = [str(CODE / "src")]
    from arc2 import ttt_data

    counts = []
    for t in roster["miss"] + roster["hit"]:
        task = json.loads((CODE / f"tasks/{t}.json").read_text())
        s = ttt_data.samples(task, t, augments=SETTINGS["augments"], max_tokens=10 ** 9)
        counts.append((len(s), sum(x["tokens"] for x in s)))
    checks["samples_every_task"] = all(n > 0 for n, _ in counts)
    est = sum(tok for _, tok in counts) * SETTINGS["epochs"] * len(SETTINGS["lrs"])
    return {"pass": all(checks.values()), "checks": checks,
            "estimate": {"train_tokens_chars3": est, "samples": sum(n for n, _ in counts),
                         "max_sample_tokens_chars3": max(tok for _, tok in counts)}}

def wait_spawned(fn, *args):
    """spawn + wait, re-attaching by call id: a client holding .remote() cancels the call when its connection drops
    (r7/r8 26 Oct, pf-temp05b 27 Oct)."""
    import time as _time
    call = fn.spawn(*args)
    print("CALL " + call.object_id, flush=True)
    misses = 0
    while True:
        try:
            return call.get(timeout=60)
        except (TimeoutError, modal.exception.TimeoutError):
            continue
        except (OSError, modal.exception.ConnectionError, _StreamTerminated) as exc:  # a dropped connection: re-attach by id
            misses += 1
            if misses > 30:
                raise
            print("reattach after " + type(exc).__name__, flush=True)
            _time.sleep(20)
            call = modal.FunctionCall.from_id(call.object_id)


@app.local_entrypoint()
def main(run_id: str, parity: str, mode: str = "dry", skip_rows: str = ""):
    text = Path(parity).read_text() if parity else ""
    # skip_rows: a finished run's tasks.jsonl; its tasks are not trained again (a follow-up on the unread ones)
    skip = [json.loads(x)["task"] for x in Path(skip_rows).read_text().splitlines() if x.strip()] if skip_rows else []
    if skip:
        print(f"skipping {len(skip)} tasks already read: {skip}")
    if mode == "dry":
        print(json.dumps(dry(text), indent=1))
    elif mode == "preflight":
        print(json.dumps(wait_spawned(preflight, run_id, text, skip), default=str)[:6000])
    elif mode == "screen":
        if "--detach" not in sys.argv and "-d" not in sys.argv:  # a GPU screen must outlive its local client
            raise SystemExit("refusing a non-detached screen launch: use modal run --detach")
        # spawn, not remote: r7/r8 lost their input when this machine's connection to Modal dropped (a client
        # holding .remote() cancels the call); a spawned call in a detached app outlives the client
        call = screen.spawn(run_id, text, skip)
        print("CALL " + json.dumps({"run_id": run_id, "call_id": call.object_id}), flush=True)
    else:
        raise SystemExit(f"mode {mode}")
