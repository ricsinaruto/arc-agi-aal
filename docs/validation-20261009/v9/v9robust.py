"""Engine-death robustness for the v9 9 h preview (A + B, on top of wd_nogiveup.py):

A. OOM victim order: arena/oom_guard.py (from oom_guard_src.py), started by the trial beside the engine watchdog:
   sandboxes oom_score_adj +1000, vLLM -1000, trial -900. No play change unless the host runs out of memory.
B. Game re-entry: a game whose turn loop raises (server errors past the retries, a failed analyzer turn, a step
   error) no longer ends; it logs ``game_reentry``, waits REENTRY_SECONDS (and for the engine watchdog to report
   healthy), then continues the same loop on the same env, session and conversation, at most REENTRY_MAX times.
   Time-outs, cancellation and the wall end as before; a hung analyzer worker (still running 35 s after the error)
   ends the game as before. No play change while nothing fails.
Usage: v9robust.py BUNDLE_ROOT (holding src/); edits in place, idempotence refused (anchors must match once)."""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

REENTER = '''    # Game re-entry (30 Oct, v9 9 h preview; docs/validation-20261009/v9/v9robust.py).
    REENTRY_MAX = 20
    REENTRY_SECONDS = 60.0

    async def _arena_reenter(self, exc, done):
        """True when the turn loop may continue after ``exc`` (the ``done``-th re-entry so far)."""
        if done >= self.REENTRY_MAX or self.remaining() <= self.REENTRY_SECONDS + 30:
            return False
        worker, self.worker = self.worker, None
        if worker is not None and not worker.done():
            try:
                await asyncio.wait_for(asyncio.shield(worker), timeout=35)
            except Exception:
                pass
            if not worker.done():
                self.worker = worker  # a hung analyzer: the game ends as before (drain_worker cleans up)
                return False
        self.backend_error = None
        self.workspace.append("game_reentry", step=self.steps, reentry=done + 1,
                              error=f"{type(exc).__name__}: {exc}"[:500], remaining=round(self.remaining(), 1))
        await asyncio.sleep(max(0.0, min(self.REENTRY_SECONDS, self.remaining() - 30)))
        try:
            from arena import engine_watchdog as _wd

            while not _wd.HEALTHY.is_set() and self.remaining() > 30 and not self.stopping.is_set():
                await asyncio.sleep(5)
        except ImportError:
            pass
        return self.remaining() > 30 and not self.stopping.is_set()

    async def run(self):
        self.started = time.monotonic()
'''

LOOP_START = ("            await self.before_first_turn()\n"
              "            while self.remaining() > 0 and self.steps < self.config.max_actions:\n")
LOOP_END = ('        except (TimeoutError, asyncio.CancelledError):\n'
            '            self.status = "time_limit"\n')


def patch_agent(path):
    text = path.read_text()
    head = "    async def run(self):\n        self.started = time.monotonic()\n"
    for anchor in (head, LOOP_START, LOOP_END):
        if text.count(anchor) != 1:
            raise SystemExit(f"{path.name}: anchor not found once: {anchor[:60]!r}")
    a = text.index(LOOP_START) + len("            await self.before_first_turn()\n")
    b = text.index(LOOP_END)
    block = text[a:b]
    if "self.status = \"time_limit\" if self.remaining() <= 0 else \"action_limit\"" not in block:
        raise SystemExit("turn loop does not end with its else clause")
    inner = "".join(("        " + line) if line.strip() else line for line in block.splitlines(keepends=True))
    wrapped = ("            reentries = 0\n"
               "            while True:\n"
               "                try:\n"
               + inner +
               "                    break\n"
               "                except (TimeoutError, asyncio.CancelledError):\n"
               "                    raise\n"
               "                except Exception as exc:\n"
               "                    if not await self._arena_reenter(exc, reentries):\n"
               "                        raise\n"
               "                    reentries += 1\n")
    text = text[:a] + wrapped + text[b:]
    text = text.replace(head, REENTER, 1)
    compile(text, str(path), "exec")
    path.write_text(text)


def patch_trial(path):
    text = path.read_text()
    old = "        _engine_watchdog.start(self.base_url, self.server_dir)\n"
    if text.count(old) != 1:
        raise SystemExit("trial.py: watchdog start anchor not found once")
    text = text.replace(old, old + "        from . import oom_guard as _oom_guard\n\n"
                                   "        _oom_guard.start(self.output)\n", 1)
    compile(text, str(path), "exec")
    path.write_text(text)


def main(root):
    src = Path(root) / "src"
    if (src / "arena/oom_guard.py").exists():
        raise SystemExit("already patched")
    shutil.copyfile(HERE / "oom_guard_src.py", src / "arena/oom_guard.py")
    patch_trial(src / "arena/trial.py")
    patch_agent(src / "duck_transfer/agent.py")
    print("patched", root)


if __name__ == "__main__":
    main(sys.argv[1])
