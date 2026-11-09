"""Arena: the Duck agents on the public (or cloned / hidden) games under one wall clock with a scheduler.

Games are admitted to a fixed number of streams; admission, parking and handover follow the study's budget policy
(level budgets, or the priority scheduler with handover at context evictions); every context is append-only between
compactions / drains. ``arena.serving_core`` holds the native vLLM start-up (formerly ``qwen_baselines``).
"""

import os

if os.environ.get("ARC_CHECKPOINT_RESUME") == "1":
    # A resumed trial continues the checkpoint's monotonic clock (arena.resume_clock); before any arena module loads.
    from . import resume_clock

    resume_clock.install()
