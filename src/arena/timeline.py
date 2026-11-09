"""Pre-launch check of a multi-trial study's timeline against its own start and timeout rules (29 September).

Three test-slot jobs were lost to sizing: kernel-phase6-v1 (both trials past their timeout), kernel-phase6b-v1 (the
trial refused to start) and kv8-prefill-check-v1 (reference cut off, gate skipped). The rules are the harness's:

- ``scripts/arena/bootstrap.py`` gives each trial ``min(budget, wall + TRIAL_GRACE)`` seconds, where the budget is
  what is left of the notebook minus 120 s;
- ``arena.run`` starts a non-competition trial only with ``wall + START_MARGIN`` seconds of the notebook left
  (level-bench, expert-quant, expert-usage, server-check and prune-quality trials are exempt);
- the first trial starts after the container and server setup, a later trial with other server settings after a
  restart.

``check`` takes per-phase estimates (measured seconds from an earlier run of the same kind) and requires every trial
to start and finish with ``margin`` (default 15%) on every phase. It returns the plan; a violation raises ValueError
naming the trial, so the build fails before any GPU time is spent.
"""

TRIAL_GRACE = 600    # scripts/arena/bootstrap.py
START_MARGIN = 480   # arena.run: hard_end - now >= wall + 480
BOOTSTRAP_RESERVE = 120  # the bootstrap's budget = notebook left - 120 s
EXEMPT_KINDS = ("level-bench", "expert-quant", "expert-usage", "server-check", "prune-quality")


def check(study, *, setup_seconds, restart_seconds, run_seconds, margin=1.15, teardown_seconds=120):
    """``run_seconds``: {trial name: estimated seconds of its measurements}. A trial restarts the server when its
    ``server_env`` differs from the previous trial's. Returns [{trial, start, end, timeout, remaining_at_start}]."""
    notebook = float(study["notebook_seconds"])
    clock = setup_seconds * margin
    previous_env = None
    plan = []
    for index, trial in enumerate(study["trials"]):
        name = trial["name"]
        if name not in run_seconds:
            raise ValueError(f"timeline: no run estimate for trial {name}")
        env = trial.get("server_env") or {}
        if index and env != previous_env:
            clock += restart_seconds * margin
        previous_env = env
        wall = float(trial.get("wall_seconds", study.get("wall_seconds", 0)))
        stop = study.get("early_stop") or {}
        if stop.get("always") and str(trial.get("kind", "")).startswith("duck"):
            wall = min(wall, float(stop["at_seconds"]))  # an unconditional run stop (the bootstrap's budget rule)
        remaining = notebook - clock
        if trial.get("kind") not in EXEMPT_KINDS and remaining < wall + START_MARGIN:
            raise ValueError(f"timeline: {name} starts at {clock:.0f} s with {remaining:.0f} s left; it needs wall "
                             f"{wall:.0f} + {START_MARGIN} s (with {margin:.2f}x margins)")
        timeout = min(remaining - BOOTSTRAP_RESERVE, wall + TRIAL_GRACE)
        needed = run_seconds[name] * margin
        if needed > timeout:
            raise ValueError(f"timeline: {name} needs {needed:.0f} s ({run_seconds[name]:.0f} x {margin:.2f}) but its "
                             f"timeout is {timeout:.0f} s (wall {wall:.0f} + {TRIAL_GRACE}, or the budget left)")
        plan.append({"trial": name, "start": round(clock), "end": round(clock + needed), "timeout": round(timeout),
                     "remaining_at_start": round(remaining)})
        clock += needed
    if clock + teardown_seconds > notebook:
        raise ValueError(f"timeline: the trials end at {clock:.0f} s; with {teardown_seconds} s of teardown that is past "
                         f"the notebook's {notebook:.0f} s")
    return plan
