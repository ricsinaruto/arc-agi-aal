"""Arena: the Duck agents on the public (or cloned / hidden) games under one wall clock with a scheduler.

Games are admitted to a fixed number of streams; admission, parking and handover follow the study's budget policy
(level budgets, or the priority scheduler with handover at context evictions); every context is append-only between
compactions / drains. ``arena.serving`` starts the vLLM server.
"""
