# RL on ARC-AGI-3, 8–31 Oct

**Bottom line:** RL on Flash works on the levels it trains on, but it never transferred to unseen games: held-out games stayed at 0.95–1.04x the base model. What is worth keeping is the recipe that learns, and the fixes that make the training data match what the model actually saw.

All runs train a rank-16 LoRA adapter (alpha 32) on all layers, with the router frozen, using fp32 AdamW. Code is on branch `trunk/rl`. Runs launch with `launch/rl/launch.py launch launch/rl/runs/<run>.json`.

### 1. RL on one level cuts tokens per clear 4.7x
- **What:** train on one level of one game, using many different starting states of that level. The reward is reference tokens divided by tokens used on a clear, and 0 on a fail. Advantages are leave-one-out within each group of tries. 2 B300s, lr 5e-5.
- **Result:**
  - Held-out starts of the trained level were cleared 31/32 times at a median of 3.9K tokens, against 18.2K for the base (4.7x).
  - Merging the adapter into the INT3 submission weights kept the gain (6.4x).
  - Other levels of the same game improved 1.18–1.38x.
- **Reproduce:** run `rl1-loop-l4` with `scripts/posttrain/modal_rl1.py` (branch `trunk/rl`).

### 2. A reward that cannot be gamed
- **What:** rewards on tokens alone were gamed. The model cut tokens by taking 3–9x more actions, and the real score fell (one run fell to 0.46x the base).
- **Fix that worked:** a log-ratio reward on score per token, with the score anchored to the human baseline and every fail given a fixed score below the worst clear.
- **Where:** logeq reward with `reward_ref human` in the `rl1-loop-seq-real-v*` configs.

### 3. Train exactly on what the model saw
These fixes are needed for any RL run on an agent:
- **Exact level starts.** Each training start replays the exact context of a real submission try, checked by a prompt digest (`mg.digest_gate`). Restores were exact: 128/128.
- **Exact reply tokens.** The trainer replays the token ids the server sampled (`RL1_REPLY_IDS`). Before this fix, 36% of trained tokens sat in a context the model never saw, because the server and trainer split text into tokens differently.
- **Expert routing replay.** The trainer uses the experts the server actually used (R3). Without it, the KL estimate was inflated 10–14x.

### 4. Token budgets that don't shrink by themselves
- **What:** setting each level's budget from successful clears alone makes it shrink every round, because tries cut by the budget never count.
- **Fix:** count cut tries at their budget, and never let a new budget cut more than 20% of recent tries. The simulated cut share fell from 47% to 15%.
- **Where:** `seq.budget_censored` and `seq.budget_guard` in the v15+ configs.

### 5. A training loop that keeps the GPU busy
- **What:** a sequential play-then-train loop on the same B300.
- **Result:** 93–95% GPU utilization in the training phase and 2.6–3.0K tok/s of play at 172 streams.
- **What it took:**
  - Removing about 2,100 host syncs per update.
  - Cutting adapter swap time from 142 s to 3–8 s.
  - Restarting the vLLM server before host memory runs out; it grows about 20 GiB per million tokens.
- **Where:** `launch/rl/runs/rl1-loop-seq-real-v17.json`.

### 6. How to judge an RL run
Only paired reads settled anything: the same seeds and games, against the base on the same build, on held-out games. In-training held-out numbers rose several times while paired reads showed no gain. The held-out follower is `launch/rl/heldout/hread.py`.

**Why transfer failed:** the served model moved only 0.53x as far as the trainer intended, and most of each update was noise (signal share 0.12). These are the first things to fix in any future RL attempt.
