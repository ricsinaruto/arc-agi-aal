"""The pre-launch timeline check (arena.timeline) on the three test-slot jobs lost to sizing on 29 September, with
their measured phases: it rejects each of them, and passes the corrected kv8-prefill-check-v2."""

import pytest

from arena import timeline


def study(notebook, *trials):
    return {"notebook_seconds": notebook, "wall_seconds": 120,
            "trials": [{"name": n, "kind": k, "wall_seconds": w, "server_env": e} for n, k, w, e in trials]}


def test_kernel_phase6_v1_would_have_been_refused():
    # Both trials ran past wall 120 + 600 s: each bench needed ~900 s (the mixed-step profile was cut off).
    s = study(3150, ("hero", "serving-bench", 120, {}), ("p6", "serving-bench", 120, {"X": "1"}))
    with pytest.raises(ValueError, match="hero needs"):
        timeline.check(s, setup_seconds=500, restart_seconds=425, run_seconds={"hero": 900, "p6": 900})


def test_kernel_phase6b_v1_would_have_been_refused():
    # The 1,200 s wall needed 1,680 s left after a ~665 s setup; 1,615 s were.
    s = study(2280, ("p6b", "serving-bench", 1200, {"X": "1"}))
    with pytest.raises(ValueError, match="p6b starts"):
        timeline.check(s, setup_seconds=665, restart_seconds=0, run_seconds={"p6b": 900}, margin=1.0)


def test_kv8_prefill_check_v1_would_have_been_refused():
    # The reference needed 822 s (x 1.15 = 945) against wall 240 + 600 = 840 s.
    s = study(2340, ("kv8p-ref", "kernel-quality", 240, {}), ("kv8p-bf16", "kernel-quality", 180, {"X": "1"}))
    with pytest.raises(ValueError, match="kv8p-ref needs"):
        timeline.check(s, setup_seconds=665, restart_seconds=370, run_seconds={"kv8p-ref": 822, "kv8p-bf16": 400})


def test_restart_only_when_the_server_changes_and_the_end_check():
    same = study(2000, ("a", "serving-bench", 60, {}), ("b", "serving-bench", 60, {}))
    plan = timeline.check(same, setup_seconds=500, restart_seconds=400, run_seconds={"a": 300, "b": 300}, margin=1.0)
    assert plan[1]["start"] == plan[0]["end"]  # no restart between identical servers
    # The end check matters when the teardown exceeds the bootstrap's 120 s reserve (the timeout already keeps a run
    # inside the notebook minus 120 s).
    with pytest.raises(ValueError, match="past the notebook"):
        timeline.check(study(1300, ("a", "serving-bench", 60, {})), setup_seconds=600, restart_seconds=0,
                       run_seconds={"a": 500}, margin=1.0, teardown_seconds=300)
    with pytest.raises(ValueError, match="no run estimate"):
        timeline.check(same, setup_seconds=1, restart_seconds=1, run_seconds={"a": 1})
    # Exempt kinds size their own wall and skip the start check (arena.run).
    exempt = study(1000, ("lb", "level-bench", 5000, {}))
    timeline.check(exempt, setup_seconds=100, restart_seconds=0, run_seconds={"lb": 300})
