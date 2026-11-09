"""``draft-head`` trial: dump the served target's MTP input rows, stop the server, train the draft head.

Draft-head fine-tune (docs/validation-20260927/draft-head-finetune.md). The server
runs the pinned stack with ``server.draft_head.dump`` (``arc_infra.draft_head_patch``: eager execution,
a Python hook that appends the target's pre-final-mixer multi-stream rows of each forward while a
control file exists). This trial:

1. sends each bundled sequence (``draft-head/data/{eval,train}.npz``, built by
   ``scripts/draft_head_data.py``) as a token-id prompt with ``max_tokens`` 1 after resetting the prefix
   cache, so the server prefills every position once; checks that the dumped rows cover positions
   0..L-1 and files them under ``<dump dir>/<split>/<index>.target.bin``. The held-out games go first
   (the first few also dump the draft's own step-1 rows, to check the training re-implementation);
   training sequences follow until the dump budget, the disk or the data runs out;
2. stops the vLLM server (freeing the GPU) exactly as the bootstrap does;
3. runs ``arena.draft_head_train`` in the serving runtime's Python (torch) with the trial's deadline.

The dumps stay on the container's disk (``/tmp``), never in the trial folder (the trial folder is
copied to the runs volume); the trainer's report and the exported overlay are.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from arc_infra.serving import MODEL

WIDTH = 4 * 2560  # hc_count x hidden_size: the MTP input rows
STOP_PROGRAM = ("from arc_infra.serving import load_recipe; import sys; "
                "load_recipe(sys.argv[1],sys.argv[2]).stop_owned_server()")
# splits: dump order (evaluation splits first, so a dump budget only shortens training); draft_splits: splits whose
# every sequence also dumps the served draft's step-1 rows (7 October); model_dir: the trainer's checkpoint (the base
# checkpoint when the server runs an expert-overlay copy: the MTP tensors, embedding and head are the same).
DEFAULTS = {"dump_dir": "/tmp/arc-draft-dump", "dump_seconds": 900, "min_free_gib": 12,
            "verify_sequences": 2, "trainer": {}, "splits": ["eval", "train"], "draft_splits": [], "model_dir": None,
            "post_dump_seconds": 900}


def event(name, **fields):
    print(json.dumps({"event": f"draft_head_{name}", "t": round(time.time(), 1), **fields}), flush=True)


def settle(path, need, timeout=60.0):
    """Wait until ``path`` holds at least ``need`` bytes and stopped growing."""
    deadline, last = time.monotonic() + timeout, -1
    while time.monotonic() < deadline:
        size = path.stat().st_size if path.exists() else 0
        if size >= need and size == last:
            return size
        last = size
        time.sleep(0.25)
    return path.stat().st_size if path.exists() else 0


def file_rows(dump, kind, dest, length, width):
    """Move ``current.<kind>.*`` to ``dest.<kind>.*`` with rows ordered by position 0..length-1 (the first
    row seen for each position). Returns the number of positions covered."""
    pos_path, bin_path = dump / f"current.{kind}.pos", dump / f"current.{kind}.bin"
    pos = np.fromfile(pos_path, dtype=np.int32)
    rows = bin_path.stat().st_size // (2 * width)
    if rows != len(pos):
        raise RuntimeError(f"{kind}: {rows} rows for {len(pos)} positions")
    if kind == "draft":  # kept as recorded (positions alongside), used for a check only
        shutil.move(pos_path, f"{dest}.draft.pos")
        shutil.move(bin_path, f"{dest}.draft.bin")
        return len(set(pos.tolist()))
    if len(pos) == length and np.array_equal(pos, np.arange(length, dtype=np.int32)):
        shutil.move(bin_path, f"{dest}.target.bin")
        pos_path.unlink()
        return length
    first = {}
    for row, p in enumerate(pos.tolist()):
        if 0 <= p < length:
            first.setdefault(p, row)
    if len(first) != length:
        raise RuntimeError(f"target rows cover {len(first)} of {length} positions")
    data = np.memmap(bin_path, dtype=np.int16, mode="r", shape=(rows, width))
    ordered = np.asarray([first[p] for p in range(length)])
    data[ordered].tofile(f"{dest}.target.bin")
    del data
    bin_path.unlink()
    pos_path.unlink()
    return length


def dump_split(client, base_url, split, data_dir, dump, settings, deadline, record):
    arrays = np.load(data_dir / f"{split}.npz")
    tokens, offsets = arrays["tokens"], arrays["offsets"]
    folder = dump / split
    folder.mkdir(parents=True, exist_ok=True)
    index = {}
    for i in range(len(offsets) - 1):
        if time.time() > deadline:
            record["stopped"][split] = "dump budget"
            break
        free = shutil.disk_usage(dump).free / 2**30
        length = int(offsets[i + 1] - offsets[i])
        if free - length * WIDTH * 2 / 2**30 < settings["min_free_gib"]:
            record["stopped"][split] = f"disk ({free:.1f} GiB free)"
            break
        for stale in dump.glob("current.*"):
            stale.unlink()
        verify = split in settings["draft_splits"] or (split == "eval" and i < settings["verify_sequences"])
        started = time.monotonic()
        try:
            for _ in range(20):  # a cached prefix (the shared system prompt) would skip its rows
                reset = client.post(base_url.removesuffix("/v1") + "/reset_prefix_cache")
                reset.raise_for_status()
                if reset.json().get("success", True):
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError("prefix cache reset did not succeed")
            (dump / "ACTIVE").touch()
            if verify:
                (dump / "ACTIVE_DRAFT").touch()
            response = client.post(base_url + "/completions", json={
                "model": MODEL, "prompt": tokens[offsets[i]:offsets[i + 1]].tolist(), "max_tokens": 1,
                "temperature": 0.0})
            response.raise_for_status()
            settle(dump / "current.target.bin", length * WIDTH * 2)
            if verify:
                settle(dump / "current.draft.bin", 1)
            (dump / "ACTIVE").unlink()
            (dump / "ACTIVE_DRAFT").unlink(missing_ok=True)
            time.sleep(0.3)
            covered = file_rows(dump, "target", folder / f"{i:05d}", length, WIDTH)
            entry = {"ok": True, "tokens": length, "seconds": round(time.monotonic() - started, 2),
                     "prompt_tokens": response.json().get("usage", {}).get("prompt_tokens"), "covered": covered}
            if verify and (dump / "current.draft.pos").exists():
                entry["draft_positions"] = file_rows(dump, "draft", folder / f"{i:05d}", length, 2560)
        except Exception as exc:  # noqa: BLE001 - one bad sequence must not end the dump
            (dump / "ACTIVE").unlink(missing_ok=True)
            (dump / "ACTIVE_DRAFT").unlink(missing_ok=True)
            entry = {"ok": False, "tokens": length, "error": f"{type(exc).__name__}: {exc}"[:500]}
        index[str(i)] = entry
        record["dumped"][split] = sum(1 for v in index.values() if v["ok"])
        record["tokens"][split] = sum(v["tokens"] for v in index.values() if v["ok"])
        if i % 10 == 0 or not entry["ok"]:
            event("dump", split=split, index=i, **{k: v for k, v in entry.items() if k != "ok"}, ok=entry["ok"])
    (folder / "index.json").write_text(json.dumps(index, indent=1))
    return index


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.split()[0])
    except Exception:  # noqa: BLE001 - no nvidia-smi: assume free
        return 0


def wait_gpu_free(limit_mib=8192, timeout=90):
    deadline = time.monotonic() + timeout
    used = gpu_used_mib()
    while used > limit_mib and time.monotonic() < deadline:
        time.sleep(3)
        used = gpu_used_mib()
    return used


def runtime_env(server_dir, source):
    env = dict(os.environ)
    saved = json.loads((Path(server_dir) / "serving-environment.json").read_text())
    for key in ("PYTHONPATH", "PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDACXX", "PYTORCH_CUDA_ALLOC_CONF",
                "TORCH_CUDA_ARCH_LIST"):
        if saved.get(key):
            env[key] = saved[key]
    env["PYTHONPATH"] = os.pathsep.join([env.get("PYTHONPATH", ""), str(Path(source) / "src")]).strip(os.pathsep)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # no PLE IPC once the server is gone
    return env


def model_dir(server_dir):
    identity = Path(server_dir) / "vllm-server-identity.json"
    if identity.is_file():
        argv = json.loads(identity.read_text()).get("argv") or []
        if "serve" in argv:
            return argv[argv.index("serve") + 1]
    return "/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"


def run_draft_head(study, trial, *, source, output, server_dir, base_url, deadline_epoch):
    settings = {**DEFAULTS, **(study.get("draft_head") or {})}
    data_dir = Path(source) / settings["data"]
    dump = Path(settings["dump_dir"])
    dump.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record = {"kind": "draft-head", "status": "running", "settings": settings, "dumped": {}, "tokens": {},
              "stopped": {}, "errors": [], "disk_free_gib": round(shutil.disk_usage(dump).free / 2**30, 1)}
    path = Path(output) / "draft-head.json"

    def save():
        path.write_text(json.dumps(record, indent=1))

    try:
        dump_deadline = min(started + settings["dump_seconds"], deadline_epoch - settings["post_dump_seconds"])
        with httpx.Client(timeout=900, trust_env=False) as client:
            for split in settings["splits"]:
                if not (data_dir / f"{split}.npz").is_file():
                    continue
                index = dump_split(client, base_url, split, data_dir, dump, settings, dump_deadline, record)
                record[f"{split}_failures"] = [v.get("error") for v in index.values() if not v["ok"]][:5]
                save()
        record["dump_seconds"] = round(time.time() - started, 1)
        rate = sum(record["tokens"].values()) / max(1.0, record["dump_seconds"])
        record["dump_tokens_per_second"] = round(rate, 1)
        record["disk_free_gib_after_dump"] = round(shutil.disk_usage(dump).free / 2**30, 1)
        event("dumped", seconds=record["dump_seconds"], dumped=record["dumped"], tokens=record["tokens"],
              tokens_per_second=record["dump_tokens_per_second"], stopped=record["stopped"])
        save()
        stop = subprocess.run([sys.executable, "-c", STOP_PROGRAM, str(server_dir), "/kaggle/input"],
                              env={**os.environ, "PYTHONPATH": str(Path(source) / "src")}, capture_output=True,
                              text=True, timeout=240)
        record["server_stop"] = {"returncode": stop.returncode, "tail": (stop.stdout + stop.stderr)[-1500:],
                                 "gpu_used_mib": wait_gpu_free()}
        if record["server_stop"]["gpu_used_mib"] > 8192:
            # Our own container: a server the owned-stop left behind would starve the trainer.
            subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], check=False)
            record["server_stop"]["killed_leftovers"] = True
            record["server_stop"]["gpu_used_mib_after_kill"] = wait_gpu_free()
        event("server_stopped", **{k: v for k, v in record["server_stop"].items() if k != "tail"})
        save()
        trainer_out = Path(output) / "trainer"
        command = [sys.executable, "-u", "-m", "arena.draft_head_train", "--model-dir",
                   settings["model_dir"] or model_dir(server_dir),
                   "--data-dir", str(data_dir), "--dump-dir", str(dump), "--out", str(trainer_out),
                   "--deadline-epoch", str(deadline_epoch - 60)]
        for key, value in settings["trainer"].items():
            flag = "--" + key.replace("_", "-")
            if key in ("draft_vocab", "init_overlay") and value and not os.path.isabs(str(value)):
                value = str(Path(source) / value)
            if value is True:
                command.append(flag)
            elif isinstance(value, list):
                command += [flag, *map(str, value)]
            elif value is not False and value is not None:
                command += [flag, str(value)]
        record["trainer_command"] = command
        save()
        with (Path(output) / "trainer.log").open("wb") as log:
            process = subprocess.Popen(command, env=runtime_env(server_dir, source), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)
            for line in iter(process.stdout.readline, b""):
                log.write(line)
                log.flush()
                if line.startswith(b'{"event": "draft_head'):
                    sys.stdout.buffer.write(line)
                    sys.stdout.flush()
            code = process.wait(timeout=max(60, deadline_epoch - time.time()))
        record["trainer_returncode"] = code
        report = trainer_out / "draft-head-train.json"
        if report.is_file():
            data = json.loads(report.read_text())
            record["result"] = {k: data.get(k) for k in ("served_step1", "eval_before", "eval_after", "selected_step",
                                                         "train_steps_done", "seconds_per_step", "export", "evals",
                                                         "curve", "stopped", "init_overlay", "oom_skips",
                                                         "load_seconds", "sequences")}
        record["status"] = "complete" if code == 0 else "complete_with_errors"
        if code:
            record["errors"].append(f"trainer exited {code}; see trainer.log")
    except Exception as exc:  # noqa: BLE001 - recorded, the trial reports an error status
        record["status"] = "error"
        record["errors"].append(f"{type(exc).__name__}: {exc}"[:1000])
    finally:
        for stale in ("ACTIVE", "ACTIVE_DRAFT"):
            (dump / stale).unlink(missing_ok=True)
        record["elapsed_seconds"] = round(time.time() - started, 1)
        save()
    return record
