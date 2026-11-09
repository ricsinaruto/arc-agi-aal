# Pruning plus distillation (SFT), 15 Oct – 2 Nov

**Bottom line:** remove half to three quarters of Flash's experts, then heal the smaller model by distilling it from the full model. The healed model plays like the original while running faster, and it is what the later ARC-AGI-3 submissions use. On-policy distillation, removing many layers, and SFT on the model's own plays did not work.

Training runs on B300s through branch `trunk/sft`:

```
bash launch/sft/launch.sh launch/sft/<run>.json
```

Exports for serving use `scripts/sft/modal_export.py` (dense layers) and `scripts/sft/modal_int3_export.py` (INT3 experts).

### 1. The distillation recipe
- **Loss:** match the full BF16 model's top-131 next-token probabilities, plus a tail bucket, on the reply tokens of good recorded plays.
- **Served format:** train with the int3/int4 rounding switched on (QAD), so the model learns the format it will be served in.
- **Learning rate:** constant 3e-6, decayed to near zero over the last ~50 steps. 1e-5 was worse in three tests.
- **Batch:** about 250K reply tokens per step, with 49–65K token windows.
- **Why it works:**
  - Pruning causes about 98% of the damage; quantization adds little.
  - Most of the healing happens in the first ~200 steps. After that, new data helps slowly, and repeated data is worth about 28% of new data.
  - The final decay carries the play gain: the decayed checkpoint beat the same run before decay.
- **Config:** `launch/sft/sft-p75-qad-p75-v1.json`.

### 2. 75% of experts removed (p75: 128 of 512 kept)
- **Result:**
  - Held-out KL to the full model went from 0.282 to 0.124.
  - Over 9 hours on held-out games it tied the 37.5%-pruned submission model (59.57 vs 59.82), because it runs 61 streams instead of 41.
  - On ARC-AGI-2, ckpt-1293 tied the base student (150 vs 144).
- **Configs:** `launch/sft/sft-p75-qad-p75-v1.json`, then `sft-p75-rest-c268-v1*.json`, `sft-p75-kd-c613-v1*.json`, and for ARC-AGI-2 `sft-p75-a2r2t10-c1047-v1.json`.

### 3. 50% of experts removed (p50: 256 kept)
- **Result:**
  - Held-out KL went from 0.121 to 0.078 in 231 steps.
  - Its ckpt-268 became the p50-sub ARC-AGI-3 submission (hidden 44.41) and the ARC-AGI-2 base student.
- **Config:** `launch/sft/sft-k256-qad-v1b.json`.

### 4. p50 with 7 of 48 layers removed (p50d7)
- **What:** drop the 7 layers that hurt least, chosen greedily one at a time, then heal.
- **Selected-data pass:** a pass on the 20% most token-efficient segments of each level, with the learning rate decayed at the end.
- **Result:**
  - Plays like p50 per token at 29% higher throughput.
  - The selected-data pass gave ckpt-942: +1.34 levels per game on held-out games against the earlier checkpoint, and 18% fewer tokens per action. This is the p50d7-sub submission.
  - On ARC-AGI-2, one pass on the best tries (ranked by correct answers per token) took the score from 125 to 142.
- **Mask and configs:** the mask is `scripts/sft/prune/prune-t50-r15b-d7-v1.json`. Configs:
  - `launch/sft/sft-p50d7-kd-c647-v1.json`
  - `sft-p50d7-top20-c731-v1.json`
  - `sft-p50d7-a2r2t-c942-v3.json` (ARC-AGI-2)

### 5. Choose checkpoints by playing them, not by KL
Held-out KL is good for catching broken checkpoints, but it does not rank good ones. One ARC-AGI-2 pass lowered KL by 3% while the score fell from 142 to 127. Every checkpoint that shipped was picked by a play read on held-out games.
