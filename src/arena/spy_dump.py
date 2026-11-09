"""One-shot py-spy dumps of a worker, one process per sample with a hard kill (kprofH3, 30 Oct). `py-spy record` hung on
the worker under turns twice, SIGINT included, losing the point; here a hang costs one sample and nothing raises."""
import asyncio
import json
import os
import signal
import time
from pathlib import Path


class SpyDump:
    def __init__(self, exe, pid, output, seconds, interval=0.5, timeouts=(5.0, 20.0)):
        self.argv = [str(exe), "dump", "--pid", str(pid)]
        self.pid, self.output, self.seconds, self.interval = int(pid), Path(output), float(seconds), float(interval)
        self.timeouts = timeouts  # nonblocking, blocking
        self.counts = {"ok": 0, "timeout": 0, "failed": 0}
        self.task = None

    def start(self):
        self.task = asyncio.create_task(self._loop())
        return self

    async def _one(self, handle, n):
        blocking = n % 2 == 1  # both modes in one run: nonblocking never pauses, blocking cannot race the frames
        argv = self.argv + ([] if blocking else ["--nonblocking"])
        started = time.monotonic()
        row = {"n": n, "blocking": blocking, "t": round(time.time(), 3)}
        process = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                       stderr=asyncio.subprocess.STDOUT)
        try:
            got, _ = await asyncio.wait_for(process.communicate(), self.timeouts[blocking])
            key = "ok" if process.returncode == 0 else "failed"
            row.update(rc=process.returncode, text=got.decode(errors="replace"))
        except (TimeoutError, asyncio.CancelledError) as exc:
            key = "timeout"
            process.kill()
            await process.wait()
            if blocking:  # a killed tracer can leave the attach stop behind
                try:
                    os.kill(self.pid, signal.SIGCONT)
                except OSError:
                    pass
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            row["seconds"] = round(time.monotonic() - started, 3)
        self.counts[key] += 1
        row["status"] = key
        handle.write(json.dumps(row) + "\n")
        handle.flush()

    async def _loop(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        end, n = time.monotonic() + self.seconds, 0
        with open(self.output, "a") as handle:
            while time.monotonic() < end:
                next_at = time.monotonic() + self.interval
                try:
                    await self._one(handle, n)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - diagnostics only
                    self.counts["failed"] += 1
                    handle.write(json.dumps({"n": n, "status": "failed", "error": repr(exc)[:200]}) + "\n")
                n += 1
                await asyncio.sleep(max(0.0, next_at - time.monotonic()))

    async def finish(self, grace=30.0):
        """Never raises: the point's timing survives whatever the sampler did."""
        try:
            await asyncio.wait_for(asyncio.shield(self.task), self.seconds + grace)
        except BaseException as exc:  # noqa: BLE001 - diagnostics only
            self.counts["end_error"] = repr(exc)[:200]
            self.task.cancel()
            try:
                await asyncio.wait_for(asyncio.gather(self.task, return_exceptions=True), 10)
            except BaseException as again:  # noqa: BLE001
                self.counts["cancel_error"] = repr(again)[:200]
            if isinstance(exc, asyncio.CancelledError):
                raise
        return dict(self.counts)
