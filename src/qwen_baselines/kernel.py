"""Parent-owned archive and per-call wall-clock supervision for B."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from . import segmentation
from .jail import Jail, needed
from .sandbox import kill_group


class Kernel:
    def __init__(self, bridge, config):
        self.bridge, self.config, self.process = bridge, config, None

    async def start(self):
        jail = await asyncio.to_thread(Jail) if needed() else None
        root = Path(__file__).parent
        helper = segmentation.write(self.bridge.archive.root / "segmentation.py")
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-u",
            str(root / "worker.py"),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=512 * 1024,
            env={
                **({"ARC_CHROOT": str(jail.root)} if jail else {}),
                "PATH": os.defpath,
                "ARC_SANDBOX_MODULE": str(root / "sandbox.py"),
                "ARC_SEGMENTATION_SOURCE": str(helper),
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        async with asyncio.timeout(30):
            line = await self.process.stdout.readline()
            if not line or json.loads(line).get("type") != "ready":
                error = await self.process.stderr.read()
                raise RuntimeError(f"Persistent worker sandbox failed: {error.decode()}")

    async def dispatch(self, request):
        archive = self.bridge.archive
        method = request["method"]
        if method == "observe":
            return archive.observation(request.get("index", -1))
        if method == "transition":
            return archive.metadata().get("transition")
        if method == "step":
            await self.bridge.step(request["action"], request.get("x"), request.get("y"))
            return archive.observation()
        name = request.get("name")
        if name not in {"history", "transitions", "transcript"}:
            raise ValueError("Unknown evidence sequence")
        if method == "length":
            return (
                archive.count("calls")
                if name == "transcript"
                else archive.count() - (name == "transitions")
            )
        if method == "item":
            return {
                "history": archive.observation,
                "transitions": archive.transition,
                "transcript": archive.transcript,
            }[name](request["index"])
        raise ValueError("Unknown RPC method")

    async def execute(self, code):
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        archive = self.bridge.archive
        call = archive.begin_call(code)
        started = time.monotonic()
        before = archive.count()
        restarted, error = False, None
        try:
            if self.process is None:
                async with asyncio.timeout(
                    min(self.config.python_seconds, self.bridge.clock.remaining())
                ):
                    await self.start()
            self.process.stdin.write(
                (json.dumps({"type": "execute", "code": code}) + "\n").encode()
            )
            await self.process.stdin.drain()
            async with asyncio.timeout(
                min(
                    self.config.python_seconds - (time.monotonic() - started),
                    self.bridge.clock.remaining(),
                )
            ):
                while True:
                    line = await self.process.stdout.readline()
                    if not line:
                        raise RuntimeError("Persistent worker exited")
                    response = json.loads(line)
                    if response["type"] == "output":
                        archive.output(call, response["text"])
                    elif response["type"] == "rpc":
                        try:
                            packet = {"result": await self.dispatch(response)}
                        except Exception as exc:
                            packet = {"error": str(exc), "error_type": type(exc).__name__}
                        self.process.stdin.write((json.dumps(packet) + "\n").encode())
                        await self.process.stdin.drain()
                    elif response["type"] == "done":
                        error = response.get("error")
                        break
        except (TimeoutError, RuntimeError, BrokenPipeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}. Python namespace lost; archive access will be restored. Code was not replayed."
            archive.output(call, error)
            restarted = True
            await self.close()
        except asyncio.CancelledError:
            archive.end_call(
                call,
                status="cancelled",
                namespace_lost=True,
                error="Gameplay deadline; completed actions remain recorded",
            )
            await self.close()
            raise
        finally:
            archive.event("python_time", call_id=call, seconds=time.monotonic() - started)
        archive.end_call(
            call,
            status="error" if error else "complete",
            error=error,
            namespace_lost=restarted,
            observations=list(range(before, archive.count())),
            receipts=[archive.metadata(index) for index in range(before, archive.count())],
        )
        return archive.transcript(call)

    async def close(self):
        if self.process:
            kill_group(self.process)
            await self.process.wait()
            self.process = None
