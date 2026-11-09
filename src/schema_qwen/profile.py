"""Baseline S settings (Schema on Qwen3.8 via the Pi coding agent)."""

S_DEFAULTS = {
    "compaction_tokens": 65536,
    # The v4 A/B/C mechanisms (object view, transition summary, dump suppression, no-change
    # repeat rejection, per-response action cap) are not part of this faithful port.
    "object_view": False,
    "transition_summary": False,
    "suppress_grid_dumps": False,
    "reject_no_change_repeats": False,
    "actions_per_turn": None,
    # Backtests and BFS run as bash commands; upstream had no per-command timeout.
    "bash_seconds": 120,
    "bash_max_seconds": 600,
    # The world-model sandbox pre-injects numpy.
    "scientific_python": True,
    # Tool turns per Pi prompt (one Schema turn), as for baseline P.
    "call_turns": 50,
    # open-schema's default --max-actions.
    "max_actions": 3000,
}
