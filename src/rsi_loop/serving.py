"""The live serving stack and its switches (docs/rsi.md 1: the proposer moves its running stack to a new build only
after that build passes a smoke test, with automatic rollback).

A stack is a candidate's ``serving/`` folder: ``server.json`` (the study's ``server`` block), optional ``study.json``
(other top-level study keys), ``env.json`` (server environment) and ``overlay/`` (put first on the server's
PYTHONPATH, so any module of vLLM or of our serving patches can be replaced). A switch stops the server, optionally
runs the proposer's GPU job (weights work: quantization, calibration on practice data, pruning) while the GPU is free,
starts the new stack the way the bundle's bootstrap does (``arena.serving --study``), and runs the smoke test; any
failure restarts the previous stack. Timer and meters live in the loop, outside the server.

Who runs it: gen 0's stack (its ``serving/`` hash) as the root bootstrap does; any other stack, which runs the
proposer's server code (overlay, environment, study keys), as ``rsiserve`` in a private mount namespace: GPU devices,
the model inputs, the bundle's serving sources, its own /tmp (``/rsi/serve/tmp``, the runtime unpacks there) with the
server's working folder bound in, the candidate's ``serving/`` read-only at /serve/candidate, loopback only; the loop's
folders, state, checkpoint, outputs, launch files and every game folder are hidden.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

STOP_PROGRAM = ("from arc_infra.serving import load_recipe; import sys; "
                "load_recipe(sys.argv[1],sys.argv[2]).stop_owned_server()")


class StackError(RuntimeError):
    pass


def _under_tmp(path):
    return str(path) == "/tmp" or str(path).startswith("/tmp/")


def _ids(user):
    import pwd
    entry = pwd.getpwnam(user)
    return entry.pw_uid, entry.pw_gid


def clip(text, head=200, tail=1300):
    """An error's start and its end (a traceback's cause is at the end; v14's was cut off at 500 characters)."""
    return text if len(text) <= head + tail else text[:head] + " [...] " + text[-tail:]


def _tail(path, size=1500):
    try:
        return Path(path).read_text(errors="replace")[-size:]
    except OSError:
        return ""


SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AWS_", "RCLONE_", "NEBIUS_", "KAGGLE_")
SERVER_TMP_CACHES = ("qwen38-flash-next-vllm-cache", "qwen38-flash-next-vllm-compile-cache")


def fit_slots(study):
    """arena.serving refuses ``server.max_num_seqs`` below ``generation_slots`` (the arena's admission slots, unused by
    the loop): a server block that lowers its sequences lowers the slots with it."""
    seqs = (study.get("server") or {}).get("max_num_seqs")
    if type(seqs) is int and type(study.get("generation_slots")) is int and study["generation_slots"] > seqs:
        study["generation_slots"] = seqs
    return study


class Stacks:
    def __init__(self, bundle, server_dir, base_url, served_model, work, input_root="/kaggle/input", log=print,
                 runtime=None, trusted_hash=None):
        self.bundle, self.server_dir, self.base_url = Path(bundle), Path(server_dir), base_url.rstrip("/")
        self.served, self.work, self.input_root, self.log = served_model, Path(work), input_root, log
        self.base_study = json.loads((self.bundle / "configs/comparison.json").read_text())
        self.work.mkdir(parents=True, exist_ok=True)
        self.runtime, self.trusted_hash = runtime, trusted_hash
        self.force_unprivileged = False  # the boot probe starts gen 0's stack the unprivileged way too

    def trusted(self, serving_dir):
        from .candidates import serving_hash
        return (not self.force_unprivileged and self.trusted_hash is not None
                and serving_hash(Path(serving_dir).parent) == self.trusted_hash)

    def study_for(self, serving_dir):
        from .candidates import read_inside

        serving_dir = Path(serving_dir)
        study = json.loads(json.dumps(self.base_study))
        if (serving_dir / "study.json").exists():
            study.update(json.loads(read_inside(serving_dir, "study.json")))
        study["server"] = json.loads(read_inside(serving_dir, "server.json"))
        return fit_slots(study)

    def env_for(self, serving_dir):
        serving_dir = Path(serving_dir)
        env = dict(os.environ)
        if (serving_dir / "env.json").exists():
            from .candidates import read_inside
            env.update({str(k): str(v) for k, v in json.loads(read_inside(serving_dir, "env.json")).items()})
        return env  # the trusted (gen 0) stack has no overlay

    def stop(self):
        """Stop the running stack. An unprivileged one (its working folder belongs to rsiserve) is killed as a user
        and everything it could write is wiped (its /tmp, HOME, the working folder): nothing it saw, held-out games
        included, outlives it. The root stop program never reads files that stack wrote."""
        from . import sandbox

        unprivileged = self.runtime is not None and self.server_dir.exists() and \
            self.server_dir.stat().st_uid == _ids(sandbox.SERVE)[0]
        if not unprivileged:
            subprocess.run([sys.executable, "-c", STOP_PROGRAM, str(self.server_dir), self.input_root],
                           check=False, timeout=300, capture_output=True)
        if self.runtime is None:
            return
        for _ in range(60):
            subprocess.run(["pkill", "-KILL", "-u", sandbox.SERVE], check=False)
            if subprocess.run(["pgrep", "-u", sandbox.SERVE], capture_output=True).returncode != 0:
                break
            time.sleep(1)
        if unprivileged:
            logs = self.work / f"server-logs-{int(time.time())}"
            logs.mkdir(parents=True, exist_ok=True)
            from .candidates import copy_regular
            for name in ("vllm-openai-server.log", "native-preflight-failure.json"):
                try:  # files rsiserve wrote: never through a link it planted
                    copy_regular(self.server_dir / name, logs / name)
                except (OSError, ValueError):
                    pass
            for folder in (self.runtime.root / "serve/tmp", self.runtime.root / "serve/home", self.server_dir):
                shutil.rmtree(folder, ignore_errors=True)
                folder.mkdir(parents=True, exist_ok=True)
                os.chmod(folder, 0o700)

    def start(self, serving_dir, tag, timeout=3600):
        if self.runtime is not None and not self.trusted(serving_dir):
            return self.start_unprivileged(serving_dir, tag, timeout)
        study_path = self.work / f"study-{tag}.json"
        study_path.write_text(json.dumps(self.study_for(serving_dir), indent=1))
        log = self.work / f"start-{tag}.log"
        with log.open("ab") as stream:
            done = subprocess.run([sys.executable, "-m", "arena.serving", "--study", str(study_path), "--working",
                                   str(self.server_dir), "--input-root", self.input_root],
                                  env=self.env_for(serving_dir), stdout=stream, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False)
        if done.returncode != 0:
            raise StackError(f"start {tag}: exit {done.returncode} (log {log})")

    def unprivileged_argv(self, serving_dir, program, env_extra=None):
        """``program`` as rsiserve in the serving namespace (also used by the isolation probe)."""
        from . import sandbox

        rt = self.runtime
        tmp = rt.root / "serve/tmp"
        self.server_dir.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not any(m in k.upper() for m in SECRET_MARKERS)}
        env.update(HOME="/serve/home", TMPDIR="/tmp", USER=sandbox.SERVE, ARC_FORCE_CHROOT="0",
                   ARENA_SKIP_WORKER_PROBE="1")
        if (Path(serving_dir) / "env.json").exists():
            from .candidates import read_inside
            env.update({str(k): str(v) for k, v in json.loads(read_inside(serving_dir, "env.json")).items()})
        # The serving sources the proposer sees (bundle rsi/serving-src: arc_infra without harness code, the arena
        # modules the server imports); the rest of the bundle (games, points, dataset copies) stays hidden.
        # rsi_boot's sitecustomize puts the candidate's overlay in front module by module (any single vLLM or
        # serving module can be replaced; the rest stays the runtime's); RSI_PYTHON_BOOT keeps it first on the
        # vLLM process's path too (serving_setup puts the runtime's packages ahead of the inherited PYTHONPATH)
        env["PYTHONPATH"] = os.pathsep.join(["/serve/src/rsi_boot", "/serve/src"])
        env["RSI_PYTHON_BOOT"] = "/serve/src/rsi_boot"
        if (Path(serving_dir) / "overlay").is_dir():
            env["RSI_OVERLAY"] = "/serve/candidate/overlay"
        env.update(env_extra or {})
        tmp_binds = [(self.server_dir, self.server_dir.relative_to("/tmp"), False)] if _under_tmp(self.server_dir) \
            else []
        if _under_tmp(Path(sys.prefix)):
            tmp_binds.append((Path(sys.prefix), Path(sys.prefix).relative_to("/tmp"), True))
        binds = [(Path(serving_dir), "/serve/candidate", True), (rt.root / "serve/home", "/serve/home", False),
                 (rt.bundle / "rsi/serving-src", "/serve/src", True)]
        if not _under_tmp(self.server_dir):
            binds.append((self.server_dir, self.server_dir, False))
        # the server reads the model and runtime inputs (/kaggle/input -> /assets/flash-input), nothing else there
        hide = [p for p in rt.hidden() if p not in ("/kaggle", "/assets")]
        hide += [p for p in ("/assets/games",) if Path(p).is_dir()]
        hide += [str(p) for p in (Path(self.input_root).parent / "working",) if p.is_dir()]
        return sandbox.command(program, sandbox.SERVE, None, env, cwd="/tmp", extra_binds=binds, hide=hide, tmp=tmp,
                               tmp_binds=tmp_binds, shm_size="16g", net="host", pid=False,
                               memory=rt.memory("server"))  # no PID namespace: the server outlives its starter

    def start_unprivileged(self, serving_dir, tag, timeout):
        from . import sandbox

        rt = self.runtime
        tmp = rt.root / "serve/tmp"
        for name in SERVER_TMP_CACHES:  # warm caches from the bootstrap's start (copied once, then the server's)
            source, target = Path("/tmp") / name, tmp / name
            if source.is_dir() and not target.exists():
                subprocess.run(["cp", "-a", str(source), str(target)], check=False)
        self.server_dir.mkdir(parents=True, exist_ok=True)
        for folder in (tmp, rt.root / "serve/home", self.server_dir):
            subprocess.run(["chown", "-R", f"{sandbox.SERVE}:{sandbox.SERVE}", str(folder)], check=True)
        study_name = f"rsi-study-{tag}.json"
        (tmp / study_name).write_text(json.dumps(self.study_for(serving_dir), indent=1))
        os.chown(tmp / study_name, *_ids(sandbox.SERVE))
        argv = self.unprivileged_argv(serving_dir, [sys.executable, "-m", "arena.serving", "--study",
                                                    f"/tmp/{study_name}", "--working", str(self.server_dir),
                                                    "--input-root", self.input_root])
        log = self.work / f"start-{tag}.log"
        with log.open("ab") as stream:
            stream.write(f"unprivileged start as {sandbox.SERVE}\n".encode())
            stream.flush()
            done = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        if done.returncode != 0:
            raise StackError(f"start {tag} (unprivileged): exit {done.returncode}: {_tail(log)}")

    async def smoke(self, timeout=600):
        """Two requests through the server: a tool call and a plain answer; both must come back well formed."""
        tools = [{"type": "function", "function": {"name": "python", "description": "Run Python code",
                                                   "parameters": {"type": "object", "properties": {
                                                       "code": {"type": "string"}}, "required": ["code"]}}}]
        asks = [{"messages": [{"role": "user", "content": "Use the python tool to compute 17*23."}], "tools": tools},
                {"messages": [{"role": "user", "content": "Reply with the single word: ready"}]}]
        out = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as http:
            for ask in asks:
                started = time.time()
                async with http.post(f"{self.base_url}/chat/completions",
                                     json={"model": self.served, "max_tokens": 2048, "temperature": 0.6, **ask,
                                           "chat_template_kwargs": {"reasoning_effort": "low"}}) as response:
                    body = await response.json()
                if response.status != 200:
                    raise StackError(f"smoke: status {response.status}: {str(body)[:300]}")
                message = body["choices"][0]["message"]
                if not (message.get("content") or message.get("tool_calls")):
                    raise StackError("smoke: empty reply")
                out.append({"s": round(time.time() - started, 2), "tool_calls": len(message.get("tool_calls") or []),
                            "completion_tokens": body.get("usage", {}).get("completion_tokens")})
        if not out[0]["tool_calls"]:
            raise StackError("smoke: no tool call parsed")
        return out

    async def switch(self, new_dir, new_tag, old_dir, old_tag, gpu_job=None, fallback=None):
        """Stop, optional GPU job, start the new stack and smoke it; on any failure restart the old one, and when
        that fails too, ``fallback`` ((dir, tag): gen 0's trusted stack). Returns a record: ``ok``, ``live``
        ("new", "old" or "fallback"), seconds, errors. Raises only when nothing comes back."""
        record = {"from": old_tag, "to": new_tag, "started": time.time()}
        self.stop()
        try:
            if gpu_job is not None:
                record["gpu_job"] = await gpu_job()
            await asyncio.to_thread(self.start, new_dir, new_tag)
            record["smoke"] = await self.smoke()
            record.update(ok=True, live="new")
        except Exception as exc:  # noqa: BLE001 - every failure rolls back
            record.update(ok=False, error=clip(f"{type(exc).__name__}: {exc}"))
            self.log({"event": "rsi_stack_rollback", **record})
            self.force_unprivileged = False  # the way back is the ordinary start (root for gen 0's stack)
            ways = [(old_dir, old_tag, "old")] + ([(*fallback, "fallback")] if fallback and
                                                   str(fallback[1]) != str(old_tag) else [])
            for index, (folder, tag, which) in enumerate(ways):
                try:
                    self.stop()
                    await asyncio.to_thread(self.start, folder, f"{tag}-back-{int(time.time())}")
                    record["rollback_smoke"] = await self.smoke()
                    record["live"] = which
                    break
                except Exception as back:  # noqa: BLE001
                    record[f"rollback_error_{which}"] = f"{type(back).__name__}: {back}"[:500]
                    self.log({"event": "rsi_stack_rollback_failed", "to": tag, "error": record[f"rollback_error_{which}"]})
                    if index == len(ways) - 1:
                        raise
        record["seconds"] = round(time.time() - record["started"], 1)
        return record
