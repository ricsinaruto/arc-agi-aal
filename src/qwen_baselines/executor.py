"""Pi operation adapters use isolated Linux children, never host shell/file tools."""

import asyncio
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

from . import segmentation
from .jail import Jail, needed
from .output_filter import suppress_dump
from .prompts import PI_API
from .sandbox import kill_group

NO_NUMPY = (
    "raise ImportError('numpy is not available: this sandbox provides only the Python "
    "standard library')\n"
)


class Executor:
    def __init__(
        self,
        bridge,
        workspace,
        *,
        api=True,
        evidence=True,
        scientific=True,
        bash_seconds=30,
        bash_max_seconds=30,
        suppress_dumps=False,
    ):
        self.bridge, self.workspace = bridge, Path(workspace).resolve()
        self.suppress_dumps = suppress_dumps
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.bash_seconds, self.bash_max_seconds = bash_seconds, bash_max_seconds
        self.api = self.workspace.parent / "api"
        self.api.mkdir(exist_ok=True)
        if api:
            shutil.copyfile(Path(__file__).with_name("arc_api.py"), self.api / "arc_api.py")
            (self.workspace / "ARC_API.md").write_text(PI_API)
        # Duck's segmentation helper is importable from every bash Python (C and P).
        segmentation.write(self.api / "segmentation.py")
        if not scientific:
            # The api directory heads PYTHONPATH in every bash process, so this stub shadows
            # the runtime's numpy for model code on both isolation paths (PRO-LONG upstream
            # sandboxes provided the standard library only).
            (self.api / "numpy").mkdir(exist_ok=True)
            (self.api / "numpy" / "__init__.py").write_text(NO_NUMPY)
        # Evidence must be outside the writable tree; symlink only grants read access.
        link = self.workspace / "evidence"
        if evidence and not link.exists():
            link.symlink_to(bridge.archive.root / "evidence", target_is_directory=True)
        self.lock = asyncio.Lock()
        self.process = None
        self.jail = None

    async def prepare(self):
        if self.jail is None and needed():
            self.jail = await asyncio.to_thread(
                Jail, self.workspace, [self.bridge.archive.root / "evidence", self.api]
            )

    async def rpc(self, sock):
        reader, writer = await asyncio.open_connection(sock=sock)
        try:
            while line := await reader.readline():
                try:
                    request = json.loads(line)
                    if request["method"] == "step":
                        result = await self.bridge.step(
                            request["action"], request.get("x"), request.get("y")
                        )
                        if request.get("grid"):
                            result = self.bridge.archive.observation()
                    elif request["method"] == "observe":
                        result = (
                            self.bridge.archive.observation()
                            if request.get("grid")
                            else self.bridge.archive.metadata()
                        )
                    elif request["method"] == "transition":
                        result = self.bridge.archive.metadata().get("transition")
                    else:
                        raise ValueError("Unknown game API request")
                    if self.jail:
                        self.jail.sync_evidence()
                    reply = {"result": result}
                except Exception as exc:
                    reply = {"error": str(exc)}
                writer.write((json.dumps(reply) + "\n").encode())
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def execute(self, operation, **args):
        async with self.lock:
            await self.prepare()
            if self.jail:
                self.jail.sync_in()
            if self.bridge.clock.remaining() <= 0:
                raise TimeoutError("Gameplay deadline reached")
            if self.bridge.stopped:
                raise RuntimeError("Skipped after boundary; await the next model decision")
            if operation != "bash":
                path = Path(args["path"])
                if not path.is_absolute():
                    path = self.workspace / path
                resolved = path.resolve()
                allowed = [self.workspace]
                if operation in {"read", "access"}:
                    allowed.append((self.bridge.archive.root / "evidence").resolve())
                if not any(resolved.is_relative_to(root) for root in allowed):
                    raise PermissionError("Path outside this game's permitted files")
                args["path"] = str(resolved)
            host, child = socket.socketpair()
            host.setblocking(False)
            process = None
            task = None
            before = self.bridge.archive.count()
            call = self.bridge.archive.begin_call(json.dumps({"operation": operation, **args}))
            started = time.monotonic()
            output = []
            try:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-I",
                    "-u",
                    str(Path(__file__).with_name("shell_child.py")),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    # Host interpreter startup noise (Kaggle's sitecustomize error, locale
                    # warnings) must never reach tool output: read results are base64.
                    stderr=asyncio.subprocess.PIPE,
                    pass_fds=(child.fileno(),),
                    start_new_session=True,
                    env={
                        **({"ARC_CHROOT": str(self.jail.root)} if self.jail else {}),
                        "PATH": os.defpath,
                        "ARC_RPC_FD": str(child.fileno()),
                        "OPENBLAS_NUM_THREADS": "1",
                    },
                )
                self.process = process
                child.close()
                task = asyncio.create_task(self.rpc(host))
                cfg = {
                    "workspace": str(self.workspace),
                    "evidence": str(self.bridge.archive.root / "evidence"),
                    "api": str(self.api),
                    "operation": operation,
                    **args,
                }
                process.stdin.write((json.dumps(cfg) + "\n").encode())
                await process.stdin.drain()
                process.stdin.close()
                noise = asyncio.create_task(process.stderr.read())
                async with asyncio.timeout(
                    min(
                        self.bash_max_seconds,
                        self.bridge.clock.remaining(),
                        args.get("timeout") or self.bash_seconds,
                    )
                ):
                    while chunk := await process.stdout.read(16384):
                        text = chunk.decode(errors="replace")
                        self.bridge.archive.output(call, text)
                        output.append(text)
                    await process.wait()
                    stderr = (await noise).decode(errors="replace")
                if stderr:
                    self.bridge.archive.event("tool_stderr", call_id=call, text=stderr[:4000])
                result = {"exitCode": process.returncode, "output": "".join(output)}
                if operation == "bash" and self.suppress_dumps:
                    result["output"], info = suppress_dump(result["output"])
                    if info:
                        result["output_suppressed"] = info
                        self.bridge.archive.event("output_suppressed", call_id=call, **info)
                if operation != "bash" and process.returncode:
                    raise RuntimeError(result["output"] + stderr)
                self.bridge.archive.end_call(call, status="complete", exit_code=process.returncode)
                return result
            except BaseException as exc:
                self.bridge.archive.end_call(
                    call,
                    status="error",
                    error=str(exc),
                    observations=list(range(before, self.bridge.archive.count())),
                )
                raise
            finally:
                self.bridge.archive.event(
                    "tool_time", call_id=call, seconds=time.monotonic() - started
                )
                kill_group(process)
                if process:
                    await process.wait()
                if self.jail:
                    self.jail.sync_out()
                if task:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                host.close()
                child.close()
                self.process = None
