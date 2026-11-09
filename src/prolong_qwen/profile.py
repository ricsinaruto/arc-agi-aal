"""Baseline P settings that differ from the shared A/B/C defaults."""

P_DEFAULTS = {
    # Pi compaction trigger; upstream coding CLIs compacted near their ~250K windows.
    # 128K on the first run (16 September); 96K afterwards so three games share the KV pool.
    "compaction_tokens": 65536,
    # The v4 A/B/C mechanisms (object view, transition summary, dump suppression, no-change
    # repeat rejection, per-response action cap) are not part of this faithful port.
    "object_view": False,
    "transition_summary": False,
    "suppress_grid_dumps": False,
    "reject_no_change_repeats": False,
    "actions_per_turn": None,
    # Claude Code's default Bash timeout and its largest model-requested value.
    "bash_seconds": 120,
    "bash_max_seconds": 600,
    # Upstream sandboxes provided only the Python standard library.
    "scientific_python": False,
}
