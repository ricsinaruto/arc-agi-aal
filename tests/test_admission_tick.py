"""admission_tick (prefix_cache_patch): waiting requests are admitted together, once per tick."""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from arena import prefix_cache_patch as patch

ROOT = Path(__file__).resolve().parents[1]
PINNED_SITE = Path(os.environ.get("ARC_PINNED_SITE", "/tmp/arc-pinned-runtime"))
PINNED_PYTHON = os.environ.get("ARC_PINNED_PYTHON", sys.executable)  # an interpreter with the runtime's imports
BASE = {"mamba_align_fix": True, "mamba_keep_state": True, "mamba_demote_superseded": True}
SCHEDULER = PINNED_SITE / patch.SCHEDULER_RELATIVE
pinned = pytest.mark.skipif(not SCHEDULER.is_file(), reason="extracted pinned runtime not available (ARC_PINNED_SITE)")


def test_option_is_seconds_and_absent_means_off():
    assert patch.validate(BASE) == BASE
    assert patch.validate({**BASE, "admission_tick": 2})["admission_tick"] == 2.0
    assert patch.validate({**BASE, "admission_tick": 0.5, "request_end_tail": True})["request_end_tail"] is True
    for bad in (True, False, 0, -1, 30.5, "2"):
        with pytest.raises(ValueError):
            patch.validate({**BASE, "admission_tick": bad})
    with pytest.raises(ValueError, match="mamba_align_fix"):
        patch.validate({"mamba_align_fix": False, "mamba_keep_state": True, "mamba_demote_superseded": True,
                        "admission_tick": 2})


@pinned
def test_patch_is_three_inserts_and_off_leaves_the_scheduler_text_alone():
    source = SCHEDULER.read_bytes()
    off = patch.patched_scheduler(source, True)
    assert patch.patched_scheduler(source, True, None) == off and b"_arc_admit" not in off
    on = patch.patched_scheduler(source, True, 2).decode()
    assert on.count("self._arc_admit_at = arc_now + 2.0") == 1 and on.count("if not arc_hold and not preempted_reqs") == 1
    assert len(on.splitlines()) - len(off.decode().splitlines()) == 31
    # no lookup before the hold: the waiting loop is not entered on a held step
    assert on.index("arc_hold = bool(") < on.index("request_queue = self._select_waiting_queue_for_scheduling()")


def replay(tmp_path, name, patches, arguments):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    sim = str(ROOT / "scripts/arena_analysis/prefix_cache_sim.py")
    out = tmp_path / name
    subprocess.run([PINNED_PYTHON, sim, "overlay", "--runtime", str(PINNED_SITE), "--out", str(out), *patches],
                   check=True, env=env, capture_output=True)
    done = subprocess.run([PINNED_PYTHON, sim, "run", *arguments], env={**env, "PYTHONPATH": str(out)},
                          capture_output=True, text=True)
    rows = [ast.literal_eval(line) for line in done.stdout.splitlines() if line.startswith("{")]
    assert rows and "steps" in rows[-1], done.stderr[-2000:]
    return rows[-1]


STACK = ["align", "keep", "demote", "tail", "keepboundary", "keepalive"]
# 48 streams with long replies and short tools at 45 ms per step: a 2 s tick is 44.4 steps.
LONG = ["--block", "64", "--batched", "164", "--cc-block", "4", "--max-seqs", "52", "--games", "48", "--requests", "3",
        "--blocks", "9000", "--gap", "20", "--step-seconds", "0.045", "--system", "166", "--new-abs", "20", "60",
        "--reply-abs", "600", "800", "--demote-min", "256"]


@pinned
def test_starts_share_a_step_wait_at_most_a_tick_and_cache_results_do_not_change(tmp_path):
    base = replay(tmp_path, "base", STACK, LONG)
    tick = replay(tmp_path, "tick", [*STACK, "tick", "--tick-seconds", "2.0"], LONG)
    assert base["admissions"] == tick["admissions"] == 144 and tick["tick_admitted"] == 144
    assert tick["prompt_steps_beside_decode"] <= 0.65 * base["prompt_steps_beside_decode"]
    assert base["wait_steps_max"] == 0 and 0 < tick["follow_wait_steps_max"] <= 48  # 44.4 + the opening's steps
    assert tick["wait_steps_mean"] <= 30 and tick["tick_openings"] < tick["tick_admitted"]
    for key in ("wrong_state", "leaked_blocks", "free_queue_dups"):
        assert base.get(key, 0) == tick.get(key, 0) == 0
    assert tick["prefix_hit_tokens"] == base["prefix_hit_tokens"] and tick["shortfall_tokens"] == base["shortfall_tokens"]
    assert tick["steps"] <= 1.06 * base["steps"]


@pinned
def test_a_continuous_arrival_stream_is_served_every_tick(tmp_path):
    # Short replies and no tool gap: a request arrives on almost every step.
    busy = LONG[:LONG.index("--requests")] + ["--requests", "12", "--blocks", "9000", "--gap", "0", "--step-seconds",
                                               "0.045", "--system", "166", "--new-abs", "20", "60", "--reply-abs",
                                               "30", "60", "--demote-min", "256"]
    tick = replay(tmp_path, "tick", [*STACK, "tick", "--tick-seconds", "2.0"], busy)
    # an opening of 40 requests x 40 tokens takes about 10 steps of the 164-token budget
    assert tick["admissions"] == 48 * 12 and tick["follow_wait_steps_max"] <= 44 + 30
    assert tick["tick_max_group"] >= 3 and tick.get("wrong_state", 0) == 0 and tick["leaked_blocks"] == 0


@pinned
def test_an_engine_without_a_decoding_stream_admits_at_once(tmp_path):
    lone = ["--block", "64", "--batched", "164", "--cc-block", "4", "--games", "1", "--requests", "6", "--blocks", "400",
            "--gap", "3", "--step-seconds", "0.045", "--system", "166", "--new-abs", "20", "60", "--reply-abs", "30",
            "60", "--demote-min", "256"]
    tick = replay(tmp_path, "tick", [*STACK, "tick", "--tick-seconds", "2.0"], lone)
    assert tick["admissions"] == 6 and tick["wait_steps_max"] == 0 and tick["tick_held_steps"] == 0
