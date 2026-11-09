# ARC-AGI-3 submissions, 12 Oct – 2 Nov

**Bottom line:**
- The best hidden score is **45.41 (v8.5p)**.
- The faster distilled students (p50-sub 44.41, then p50d7-sub at +29% tok/s) keep that level of play while generating more tokens.
- What moved the score was serving precision and a correct stream count. Raw speed alone did not.

Target hardware: one RTX PRO 6000 with about 180 GB of host RAM, for 9 hours. Builds are pushed with:

```
scripts/kaggle_cli.py kernels push -p dist/arena-flash-competition-<name>-sub
```

### 1. v8.5p: precision where it matters (hidden 45.41, best)
- **What:**
  - 37.5% of experts pruned, with INT3 experts.
  - **BF16** dense layers and attention-indexer keys.
  - int4 KV cache, a bounded prefill, and an engine watchdog.
- **Result:** BF16 dense plus BF16 keys took the hidden score from 37.6 (v8) to 41.7 (v8.5). The rest of v8.5p took it to 45.41.
- **Build:** `build_v82.py --v85p` (branch v84-sub).

### 2. Startup under 600 s
- **What:** prefetch the weight files in the order the loader reads them. The 10 cold FP8 shards were the main cost.
- **Shared-memory fix:** delete the shared-memory offload file whenever the watchdog restarts the engine. Orphaned files crashed one kernel.
- **Result:** startup went from about 1,900 s to 400–600 s.
- **Where:** the v8 startup path in the v84-sub builders, flag `--l1-shm-fix`.

### 3. Stream-count rule
- **What:** at boot, read the KV pool and set:

  ```
  streams = min(slots, floor(gate_capacity / blocks_per_stream))
  ```

  Use about 46–48 blocks per stream.
- **Result:** KV use stays near 0.8 and the prefix-cache hit rate at or above 0.95. Below about 44 blocks per stream the hit rate collapses and throughput halves.
- **Where:** `focus_oversubscription.gate_blocks_per_stream` in the bundle's `configs/comparison.json`, written by the builders in `docs/validation-20261009/` (branch `v84-sub`).

### 4. v8.7b: better requantization (hidden 44.64)
- **What:** requantize the down-projection everywhere, plus gate and up in 6 layers, at 4-bit with group size 8.
- **Result:** reply KL fell from 0.118 to 0.098.
- **Build:** the v87b builder in `docs/validation-20261009/` (branch `v84-sub`).

### 5. p50-sub: the 50%-pruned student (hidden 44.41)
- **What:** the distilled p50 ckpt-268 (see `sft-opd.md`), with INT3 experts, INT8 dense layers, int4 KV and 46 streams.
- **Result:**
  - On 9 h held-out games it scored 63.72 and 656 levels, with 0 preemptions.
  - At 40M tokens it reached 645 levels, against 573 for the p75 build.
- **Build:** `mkP50sub.py`.

### 6. p50d7-sub: 7 layers fewer, same play, faster
- **What:** p50-sub with the 41-layer ckpt-942 weights and the stream rule above, giving 52 streams.
- **Result:**
  - The preview ran at about 2,700 tok/s against p50-sub's 1,900.
  - In the first hour it played 213 levels against 169, with the same levels per token.
- **Build:** `mkP50d7sub.py`.

### 7. Which local read predicts the hidden score
The held-out read (10 games x 11 copies, 9 h) ranked hidden scores correctly. The public-game roster (c110, copies of the 20 hardest public games) did not, for builds fitted on those games: V9C scored higher there but 36.88 hidden, against p50's 44.41. Compare builds at matched tokens, not at the end of the run.
