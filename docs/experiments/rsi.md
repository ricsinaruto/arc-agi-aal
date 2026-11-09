# Self-improvement (RSI), 8–18 Oct

**Bottom line:** Flash editing its own agent produced no edit that beat the starting system beyond noise. Three things from the work are worth reusing.

### 1. A paired yardstick sized to the real noise
- **What:** before judging candidates, play the starting system twice on the same games and seeds, and measure the run-to-run spread. Then compare each candidate against a fresh start run on the same seeds, and accept only on a clear margin.
- **Result:** one run against one run has a standard error of 4–5 RHAE points, so only gains of about 10 points can be seen. The accept rule was z ≥ 2 on both score and early-clear speed, followed by a confirmation run.
- **Reproduce:** the noise check is `scripts/rsi2/build_gen0_v85.py` (branch `rsi2`), test `rsi2-arena`. The loop with this rule is `scripts/rsi2/build_loop.py`, with code in `src/rsi2_loop/`.

### 2. Refitting the submission stack on a few fixed games costs nothing
- **What:** rebuild the v8.5 submission model's expert mask, quantization and speculative-decoding head using only 10 fixed public games, so nothing is fitted on held-out games.
- **Result:**
  - The mask matched the original's by 96.9%.
  - Quantization error (0.0909) and draft acceptance (0.79) matched the original.
  - Speed matched (1,476 tok/s).
- **Reproduce:** `scripts/rsi2/fit_mask.py`, `draft_vocab_fixed10.py`, `dh_bundle.py`, `export_step.py` (branch `v85-refit`).

### 3. Keep the selection rules out of the proposer's reach
- **What we saw:** given control over how candidates were scored, the proposer moved itself to an easier, token-capped setting. That setting lost 11–15 points against the start.
- **Lesson:** the yardstick, the game pool and the accept rule must be fixed outside anything the proposer can edit.

Also observed: the best candidate (a helper kit plus medium reasoning effort) was +0.7 ± 5.8 on the pool. That is a hint, not a result.
