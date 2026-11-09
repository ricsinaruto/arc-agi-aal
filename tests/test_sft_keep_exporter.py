"""launch/sft/keep_exporter.sh: a crashed dashboard exporter restarts and alerts; a finished or stopped one does not."""
import os
import subprocess
from pathlib import Path

KEEP = Path(__file__).resolve().parents[1] / "launch/sft/keep_exporter.sh"


def keep(tmp_path, body):
    exporter = tmp_path / "fake.sh"
    exporter.write_text(f"n=$(cat {tmp_path}/n 2>/dev/null || echo 0); echo $((n + 1)) > {tmp_path}/n; echo run $n\n{body}\n")
    python = tmp_path / "py"
    python.write_text('#!/bin/bash\nshift\nexec bash "$@"\n')  # drops -u; the exporter is a shell fake
    python.chmod(0o755)
    done = subprocess.run(["bash", str(KEEP), str(python), str(exporter), "r1", str(tmp_path)], timeout=20,
                          env={**os.environ, "KEEP_EXPORTER_WAIT": "0"})
    log = tmp_path / "watch.log"
    return done.returncode, int((tmp_path / "n").read_text()), log.read_text() if log.exists() else ""


def test_crash_restarts_and_alerts_then_a_clean_end_stops(tmp_path):
    rc, runs, log = keep(tmp_path, '[ "$n" -ge 2 ] && exit 0; echo "ModuleNotFoundError: mfu"; exit 1')
    assert (rc, runs) == (0, 3)
    assert log.count("ALERT r1 [exporter] exit 1") == 2 and "ModuleNotFoundError: mfu" in log


def test_clean_end_is_quiet(tmp_path):
    assert keep(tmp_path, "exit 0") == (0, 1, "")


def test_term_is_a_stop_not_a_crash(tmp_path):
    assert keep(tmp_path, "kill -TERM $$") == (0, 1, "")


def test_quick_crashes_give_up(tmp_path):
    rc, runs, log = keep(tmp_path, "exit 3")
    assert (rc, runs) == (1, 5) and "gave up after 5 quick crashes" in log
