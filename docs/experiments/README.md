# What worked, 5 Oct – 4 Nov 2026

One file per area. Each lists only the experiments that worked or gave a lesson worth keeping, with the key numbers and where to find the code to rerun them. Dead ends are left out.

| File | Area |
|---|---|
| [rsi.md](rsi.md) | Self-improvement loop: nothing beat the start, but the yardstick and the refit are reusable |
| [rl.md](rl.md) | RL on ARC-AGI-3: works on trained levels, plus the fixes any future RL run needs |
| [sft-opd.md](sft-opd.md) | Pruning plus distillation: the recipe behind the faster submission models |
| [arc2.md](arc2.md) | ARC-AGI-2 harness and settings that raised the score |
| [arc3-sub.md](arc3-sub.md) | ARC-AGI-3 submission builds and serving rules |
| [l4.md](l4.md) | ARC-AGI-2 serving on 4xL4 |

Smaller lines of work (Dots3, MiniMax, world-model guidance, other harness ideas, DeepSeek, Flint) produced nothing that shipped.

Terms:
- **Flash:** Qwen3.8-Flash-Next, the base model.
- **Roster:** the set of games or tasks a run plays.
  - **c100 / c110:** the 20 hardest public ARC-AGI-3 games, each copied 5–6 times (100 or 110 games in all).
  - **ho10x11:** 10 held-out ARC-AGI-3 games x 11 copies.
  - **excl5:** 5 held-out ARC-AGI-3 games.
  - **c120 / c240:** ARC-AGI-2 rosters of 120 and 240 tasks.
- **RHAE:** the ARC-AGI-3 per-level score. **Score per token:** score divided by generated tokens.
- **Code** is cited by path and branch.
