"""Engine watchdog without give-up (30 Oct, v9-sub early end): an engine death or failed restart must never end the
trial early; games wait for the server to the deadline (no gameplay change: the same requests are re-sent).

Before: after MAX_RESTARTS (4) restarts the watchdog logged ``gave_up``, set HEALTHY and returned, so every request
fell back to 4 attempts and raised; each game's ``backend_error`` then ended it (status error) and the trial ended
when all games had. The httpx patch also raised after 5 transport errors or a 1,800 s outage.
After: restarts continue with a backoff (min(600, 60 x restarts past MAX_RESTARTS) s between attempts, logged
``restart_backoff``); HEALTHY stays clear while the server is down, so ``with_server_retries`` keeps waiting
(``settle(remaining - 30)``); the httpx patch waits out any outage (its 5-attempt bound applies only while healthy).
Usage: wd_nogiveup.py BUNDLE_SRC_ROOT (the folder holding src/arena/engine_watchdog.py); edits in place."""
import sys
from pathlib import Path

EDITS = [
    ('            if fails >= FAILS:\n'
     '                if STATE["restarts"] >= MAX_RESTARTS:\n'
     '                    _log("gave_up", restarts=STATE["restarts"])\n'
     '                    HEALTHY.set()  # games fall back to their ordinary retries\n'
     '                    return\n'
     '                STATE["restarts"] += 1\n',
     '            if fails >= FAILS:\n'
     '                if STATE["restarts"] >= MAX_RESTARTS:\n'
     '                    # No give-up (30 Oct): games wait (HEALTHY stays clear) and restarts continue with a backoff.\n'
     '                    pause = min(600.0, 60.0 * (STATE["restarts"] - MAX_RESTARTS + 1))\n'
     '                    _log("restart_backoff", restarts=STATE["restarts"], seconds=pause)\n'
     '                    time.sleep(pause)\n'
     '                    if _healthy(root):\n'
     '                        continue\n'
     '                STATE["restarts"] += 1\n'),
    ('                if attempt > 5 or not await settle(1800.0):\n'
     '                    raise\n',
     '                if (attempt > 5 and HEALTHY.is_set()) or not await settle(86400.0):\n'
     '                    raise\n'),
]

root = Path(sys.argv[1])
p = root / "src/arena/engine_watchdog.py"
text = p.read_text()
for old, new in EDITS:
    if text.count(old) != 1:
        raise SystemExit(f"engine_watchdog.py: anchor not found once: {old[:70]!r}")
    text = text.replace(old, new)
p.write_text(text)
print("patched", p)
