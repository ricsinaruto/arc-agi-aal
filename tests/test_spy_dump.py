"""spy_dump: hangs and failures cost one sample, kill the sampler process, and never raise."""
import asyncio
import json
import os
import stat

from arena.spy_dump import SpyDump


def fake(tmp_path, body):
    exe = tmp_path / "py-spy"
    exe.write_text("#!/bin/bash\n" + body + "\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def run(exe, tmp_path, seconds=1.0, **kw):
    out = tmp_path / "dumps.jsonl"

    async def go():
        spy = SpyDump(exe, os.getpid(), out, seconds, interval=0.1, timeouts=(0.3, 0.3), **kw).start()
        return await spy.finish(grace=5)

    counts = asyncio.run(go())
    return counts, [json.loads(line) for line in out.read_text().splitlines()]


def test_ok_alternates_modes(tmp_path):
    counts, rows = run(fake(tmp_path, 'echo "$*"; echo Thread 1 MainThread'), tmp_path, seconds=1.2)
    assert counts["ok"] >= 2 and counts["timeout"] == counts["failed"] == 0
    assert [r["blocking"] for r in rows[:2]] == [False, True]
    assert "--nonblocking" in rows[0]["text"] and "--nonblocking" not in rows[1]["text"]


def test_hang_costs_one_sample_and_is_killed(tmp_path):
    marker = tmp_path / "pids"
    counts, rows = run(fake(tmp_path, f'echo $$ >> {marker}; exec sleep 600'), tmp_path)
    assert counts["timeout"] >= 2 and counts["ok"] == 0 and "end_error" not in counts
    for pid in marker.read_text().split():
        try:
            os.kill(int(pid), 0)
            raise AssertionError(f"sampler {pid} still alive")
        except ProcessLookupError:
            pass


def test_hang_only_in_blocking_mode(tmp_path):
    counts, _ = run(fake(tmp_path, 'case "$*" in *--nonblocking*) echo ok;; *) exec sleep 600;; esac'), tmp_path)
    assert counts["ok"] >= 1 and counts["timeout"] >= 1


def test_bad_exit_and_missing_exe_never_raise(tmp_path):
    counts, _ = run(fake(tmp_path, "exit 3"), tmp_path, seconds=0.3)
    assert counts["failed"] >= 1
    counts, rows = run(tmp_path / "nope", tmp_path, seconds=0.3)
    assert counts["failed"] >= 1 and rows[0]["status"] == "failed"


def test_finish_survives_a_stuck_loop(tmp_path):
    async def go():
        spy = SpyDump(fake(tmp_path, "echo ok"), os.getpid(), tmp_path / "d.jsonl", 0.1)

        async def stuck():
            await asyncio.sleep(600)

        spy.task = asyncio.create_task(stuck())
        return await spy.finish(grace=0.2)

    assert "end_error" in asyncio.run(go())
