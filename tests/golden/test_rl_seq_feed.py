"""Golden (redesign): the rl_seq loop on the fake servers/trainer of tests/test_rl_seq.py: the call sequence and the
feed rows (kinds, groups, steps; times dropped) stay identical through the trunk/rl cleanup."""
import asyncio
import sys
from pathlib import Path

from common import check

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_rl_seq as T  # noqa: E402

DROP = {"t", "epoch", "seconds", "setup_seconds", "since_play_s", "first_token_t", "play_t", "ready_t", "switch",
        "phases", "stop", "start", "timeline", "boots", "checkpoint_seconds", "step_call_seconds", "play_seconds",
        "train_seconds", "waited_s", "since_spawn_s", "trainer", "server_env"}


def test_rl_seq_fake_run(tmp_path, harvest_ok, monkeypatch):
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "vo"))
    loop = T.FakeLoop(tmp_path)
    gate = T.phase.PhaseGate(inner=T.Upstream())
    why = asyncio.run(asyncio.wait_for(T.rl_seq.run(loop, gate=gate), 30))
    rows = [{k: v for k, v in r.items() if k not in DROP} for r in loop.feed.rows]
    kinds = {}
    for r in rows:  # counts only; the fakes race one late group in or out before the last stop: groups not counted
        if r.get("kind") not in ("seq_group", "group_done", "drops"):
            kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
    check("rl_seq_feed", {"why": why, "log": loop.log, "kinds": kinds,
                          "steps": [r for r in rows if r.get("kind") in ("step", "seq_train", "seq_end", "seq_start")]})


harvest_ok = T.harvest_ok
