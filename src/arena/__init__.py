"""Arena: four agents on the 25 public games under one 900 s wall clock with a scheduler.

Two games are served at a time after an opening phase; admission and parking follow per-level
time limits derived from the score's level weights; every context is append-only and compacted
at the cap. The 3-game study paths (duck_controls, qwen_baselines.runner) are untouched.
"""
