"""v9's engine fixes (v84-sub 3c48401c3, 599bc25e1) ported to the arc2 L4 bundle; no play change while nothing fails.

A. No give-up: restarts continue past MAX_RESTARTS with a backoff, HEALTHY stays clear, requests wait to the wall.
B. OOM victim order: arena/oom_guard.py, started beside the watchdog (interpreters +1000, vLLM -1000, trial -900).
C. Task re-entry: a request lost to a server failure does not count against errors_per_task, and any other exception
   in a task's turn re-queues the task, each at most REENTRY_MAX times, after REENTRY_SECONDS and a healthy server.
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REENTRY = """REENTRY_MAX, REENTRY_SECONDS = 20, 60.0  # task re-entry (v9robust): a failed turn waits and continues


async def reenter(run, exc, left, record):
    \"\"\"True when the task may go on after ``exc``: waited REENTRY_SECONDS, then for a healthy server.\"\"\"
    from arena import engine_watchdog as wd

    done = getattr(run, "reentries", 0)
    if done >= REENTRY_MAX or left() <= REENTRY_SECONDS + 30:
        return False
    run.reentries = done + 1
    record({"event": "task_reentry", "task": run.id, "reentry": done + 1, "error": f"{type(exc).__name__}: {exc}"[:400]})
    await asyncio.sleep(REENTRY_SECONDS)
    while not wd.HEALTHY.is_set() and left() > 30:
        await asyncio.sleep(5)
    return left() > 30


def server_failed(exc):
    \"\"\"A request lost to the server (5xx, connection), not to the request itself.\"\"\"
    from arena import engine_watchdog as wd

    if not wd.enabled():
        return False
    response = getattr(exc, "response", None)
    return isinstance(exc, httpx.TransportError) or (response is not None and response.status_code >= 500)


"""
EDITS = {
    "src/arena/engine_watchdog.py": [
        ('''                if STATE["restarts"] >= MAX_RESTARTS:
                    _log("gave_up", restarts=STATE["restarts"])
                    HEALTHY.set()  # games fall back to their ordinary retries
                    return
''', '''                if STATE["restarts"] >= MAX_RESTARTS:  # no give-up: requests wait, restarts continue with a backoff
                    pause = min(600.0, 60.0 * (STATE["restarts"] - MAX_RESTARTS + 1))
                    _log("restart_backoff", restarts=STATE["restarts"], seconds=pause)
                    time.sleep(pause)
                    if _healthy(root):
                        continue
'''),
        ('''                if attempt > 5 or not await settle(OUTAGE_WAIT):
''', '''                if (attempt > 5 and HEALTHY.is_set()) or not await settle(86400.0):
'''),
    ],
    "src/arc2/trial.py": [
        ('''                    if (engine_watchdog.enabled() and outages < 5
                            and await engine_watchdog.settle(min(engine_watchdog.OUTAGE_WAIT, self.left() - 1))
''', '''                    if (engine_watchdog.enabled()
                            and await engine_watchdog.settle(self.left() - 1)
'''),
        ('''    engine_watchdog.start(base_url, server_dir)  # armed by arc2.kaggle (ARENA_ENGINE_WATCHDOG), else a no-op
''', '''    engine_watchdog.start(base_url, server_dir)  # armed by arc2.kaggle (ARENA_ENGINE_WATCHDOG), else a no-op
    from arena import oom_guard

    oom_guard.start(output)
'''),
        ('''            except Exception as exc:  # noqa: BLE001 - one task\'s failure never ends the trial
                failures.append(f"{runs[n].id}: {type(exc).__name__}: {exc}"[:300])
                runs[n].done = "failed"
''', '''            except Exception as exc:  # noqa: BLE001 - one task\'s failure never ends the trial
                if await reenter(runs[n], exc, lambda: deadline - time.monotonic(), engine.record):
                    queue.put_nowait(engine.queue_item(runs[n], n) if hasattr(engine, "queue_item")
                                     else (engine.priority(runs[n]), n))
                    continue
                failures.append(f"{runs[n].id}: {type(exc).__name__}: {exc}"[:300])
                runs[n].done = "failed"
'''),
        ('''class Transport''', REENTRY + '''class Transport'''),
    ],
    "src/arc2/conv.py": [
        ('''                if self.deadline is not None and time.monotonic() >= self.deadline - 5:
                    return
                run.errors += 1
                self.record({"event": "request_error", "task": run.id, "error": f"{type(exc).__name__}: {exc}"[:400]})
''', '''                if self.deadline is not None and time.monotonic() >= self.deadline - 5:
                    return
                from .trial import reenter, server_failed

                if server_failed(exc) and await reenter(run, exc, lambda: self.deadline - time.monotonic()
                                                        if self.deadline is not None else 86400.0, self.record):
                    continue
                run.errors += 1
                self.record({"event": "request_error", "task": run.id, "error": f"{type(exc).__name__}: {exc}"[:400]})
'''),
    ],
}


def apply(out):
    """Patch the bundle tree ``out`` (hard-linked files are replaced, not written through)."""
    for rel, edits in EDITS.items():
        path = Path(out) / rel
        text = path.read_text()
        for old, new in edits:
            assert text.count(old) == 1, (rel, old[:70])
            text = text.replace(old, new)
        compile(text, rel, "exec")
        path.unlink()
        path.write_text(text)
    shutil.copyfile(HERE / "oom_guard_src.py", Path(out) / "src/arena/oom_guard.py")
