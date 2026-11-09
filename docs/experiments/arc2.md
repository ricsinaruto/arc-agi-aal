# ARC-AGI-2 harness and settings, 13 Oct – 3 Nov

**Bottom line:**
- The best hidden score is **73.19 (v52)**.
- The best harness found since then is **base15e256**: one answer, then a reopen asking for a different second answer. About 20 later designs tied it and none beat it.
- The final candidate, **v57**, is base15e256 plus correctness fixes. It leads every alternative on the 240-task roster.

Reads run on one RTX PRO 6000. A read is the score curve against generated tokens, compared at matched tokens. Builds for Kaggle are in `scripts/l4/build_b2.py` (branch `l4`).

### 1. Settings that raised the score
Each was measured against its base at matched tokens on the 240-task roster:

| Setting | Effect |
|---|---|
| INT3 refit experts + int4 KV cache | 193 vs 181 correct, and 14–17% faster |
| 224 px grid images in the prompt | +13.8 on the score curve |
| 16K reply cap | +9 at 9.26M tokens |
| drain_half (drop the oldest whole blocks to ~57K under budget) | keeps long conversations within context |
| 256K context window | best curve (base15e256) |

### 2. The best harness: one answer, then a reopen (base15e256)
- **What:**
  1. The model works the task and submits one answer, told to submit only when 100% certain.
  2. Later, the task is reopened with its full context, and the model must give a second answer that differs from the first.
  3. Settings: 262,144 window, xhigh effort, 16K reply cap, 224 px images, 64 slots.
- **Result:** on the 120-task roster it scored 144 correct at 10.2M tokens, with area under the curve 58.5, the best of about 25 designs. Several close variants tied it within noise: base15, base15b, base20, base22.
- **Run:** `arc2-rtx-c120-hx-base15e256-cpu32-p1-v3b-nebius-20261102`.

### 3. Correctness fixes (free)
- **What:** sandbox fixes, tool-call salvage and restoring renamed helper functions.
- **Result:** errors fell from 42 to 0, and 52 empty tool calls were salvaged, with no extra tokens. Every new build should include them.
- **Where:** `reopen_picture`, `sandbox_fix` and `restore_names` in the base22 bundle's `src/arc2` (branch `arc2`).

### 4. v57: the final candidate, read on the competition-sized roster
- **What:** base15e256 plus the fixes, on the 240-task roster. This roster is the size of the hidden set, and the 120-task roster overstates it by 2–6 points.
- **Result:**
  - v57 led every alternative at matched tokens: 233 correct at 10M tokens (69.9%), against 150–220 for the others.
  - It finished at 272/334 (81.3%) at 19.6M tokens.
  - 99% of tasks had a first answer by 13.1M tokens.
- **Run:** `arc2-rtx-c240-hx-v57240-cpu32-p1-v3b-nebius-20261102`.

### 5. Hidden scores that moved

| Build | Change | Hidden |
|---|---|---|
| v37 | 64K window, 224 px images, refit experts | 65.6 |
| v49 | + ARC-AGI-2 hint, review, reject known answers | 66.8 |
| **v52** | BASE12 harness (fresh tries, two guesses), 152K context, drain jitter | **73.19** |

A single submission has an sd of about 3.8, so smaller differences are noise.
