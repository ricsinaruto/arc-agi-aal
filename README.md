# arc-agi-aal

Code for our ARC Prize 2026 entries: an agent on Qwen3.8-Flash-Next ("Flash") for ARC-AGI-3 (one RTX PRO 6000,
9 h) and for ARC-AGI-2 (four L4 GPUs, 12 h), with the serving stack, the post-training work (RL, prune-then-heal
distillation, self-improvement) and the tooling around them.

**Results:** ARC-AGI-3: 8th at the milestones, 16th in the final standings. ARC-AGI-2: 26th in the final standings.

- What worked, with the numbers and the code to rerun it: [docs/experiments/](docs/experiments/README.md).
- Training practice notes from the literature that the training runs were checked against:
  [docs/posttrain-practice.md](docs/posttrain-practice.md).
- Public Kaggle notebooks: ARC-AGI-3
  [v85p-sub](https://www.kaggle.com/code/richardcsaky/arc-arena-flash-competition-v85p-sub) and
  [p50-sub](https://www.kaggle.com/code/richardcsaky/arc-arena-flash-competition-p50-sub); ARC-AGI-2
  [l4v52-c240](https://www.kaggle.com/code/richardcsaky/arc2-kaggle-l4v52-c240),
  [l4v56k-c120](https://www.kaggle.com/code/richardcsaky/arc2-kaggle-l4v56k-c120) and
  [l4v57s44-c120](https://www.kaggle.com/code/richardcsaky/arc2-kaggle-l4v57s44-c120).

## What we did

One open model, Flash, served by our vLLM fork, plays every game or task under a fixed wall clock. A score is
tokens per game times score per token, so most of the work went into getting more tokens out of the hardware
without losing score per token.

| Submission | Score | What it runs |
|---|---|---|
| ARC-AGI-3 v85p-sub | 45.41 | Flash with 37.5% of experts pruned, 3-bit experts, BF16 dense layers, 4-bit KV cache |
| ARC-AGI-3 p50-sub | 44.41 | 50%-pruned student healed by distillation, 8-bit dense layers |
| ARC-AGI-2 l4v52-c240 | 73.19 | Flash with half its experts, 3-bit experts, 4-bit KV cache on 4xL4; re-solve passes with two guesses each |

- **Agent (ARC-AGI-3).** A port of Tufa Labs' public Duck harness with features from dfranzen's public Milestone 2
  notebook. We added a token-economy prompt, a model-written playbook shown after each context cut, a step-back
  prompt for stuck levels, and a scheduler that gives a free slot to the game with the fewest tokens so far.
- **Serving.** Routed experts pruned by router usage and stored in 3 bits, a 4-bit KV cache, prefix caching, and the
  number of games at once set at boot from the measured KV pool. Precision mattered more than raw speed: BF16 dense
  layers raised the hidden score from 37.6 to 41.7, and the 4-bit KV cache plus an engine watchdog to 45.41.
- **Pruning plus distillation.** Unhealed, pruning past 37.5% of experts breaks tool calls. Forward-KL distillation from
  the full model on recorded plays healed a 50%-pruned student to within noise of the shipped model and let a
  75%-pruned student tie it over 9 hours. A student healed on ARC-AGI-3 plays also solved ARC-AGI-2 tasks as well as
  our ARC-AGI-2-tuned pruned model.
- **RL.** LoRA RL on real-try contexts with GSPO cut tokens per clear 4.7x on trained levels and 1.1–1.4x on
  unseen levels of the same game, but did not transfer to unseen games.
- **ARC-AGI-2.** The ARC-AGI-3 scheduler and context handling run all three ARC-AGI-2 submissions. The model gets the
  grids, a Python interpreter and a `submit` call. Tries behave like independent draws, and asking for a different
  second answer beats checking answers against the training pairs.
- **Did not help:** on-policy distillation (the student learned to end replies mid-reasoning), a self-improvement loop
  (no edit beat the run-to-run noise), ~20 harness aids, other harnesses and models, and smarter schedulers.

[docs/experiments/](docs/experiments/README.md) has the numbers, settings and code paths for each line of work.

## Layout

- `src/duck_transfer`, `src/arena`: the ARC-AGI-3 agent and the arena that plays many games under one wall clock
  with a scheduler.
- `src/arc_infra`, `src/arena/serving_core`: the pinned vLLM fork patches (expert pruning masks, INT3 experts, KV
  gate, prefix caching, draft head) and the shared run library.
- `src/inference`, `src/taaf`: the public Duck and TAAF code the agent started from.
- `src/posttrain`, `scripts/posttrain`: early post-training code (episodes, rejection sampling, LoRA).
- `scripts/`: model preparation (expert usage, pruning masks, 3-bit requantization), Modal jobs, benchmarks and
  analyses (`scripts/arena_analysis`).
- `tools/ops`: launch and monitoring tooling: run records, CPU launch gate, live watcher, dashboards, Modal and
  Nebius reconciliation.
- `data/`: small inputs the code reads: expert pruning masks (`data/expert_prune`), level-bench start points,
  kernel-quality prompts, training-game rosters, and the early post-training episodes and games.
- `kaggle/`: notebook bootstrap code and pinned runtime versions.

Each line of work lives on its own branch; `main` holds the shared code.

| Branch | Contents |
|---|---|
| `v84-sub` | ARC-AGI-3 submission builders (`docs/validation-20261009/`) |
| `trunk/sft` | Pruning plus distillation, the p50/p75 students (`launch/sft/`, `scripts/sft/`) |
| `trunk/rl` | RL on ARC-AGI-3 (`launch/rl/`, `scripts/posttrain/modal_rl1.py`) |
| `trunk/pth`, `sft-heal`, `sft-opd` | Earlier prune-then-heal and on-policy distillation |
| `arc2` | ARC-AGI-2 harness (`src/arc2`) |
| `l4`, `l4kv`, `l4pp` | ARC-AGI-2 serving on 4xL4 (`scripts/l4/build_b2.py`), KV pool and pipeline-parallel variants |
| `v85-refit` | Refit of the v8.5 stack's pruning mask and draft head (`scripts/rsi2/`), throughput benchmarks |
| `tp-heldout` | The submission stack refit on training games only, for clean held-out reads; exact reproduction of dfranzen's notebook (`scripts/dfr_exact/`) |
| `rsi`, `rsi-loop`, `rsi2` | Self-improvement: first hand-made attempt, the first automated loop, its rewrite |
| `nebius`, `nebius-opd` | Launcher for rented RTX PRO 6000 machines (`scripts/nebius_validation.py`) |
| `envs-trunk-data` | Generated training games |
| `prime2` | A Prime agent harness for ARC-AGI-2 on Flash (unfinished) |

## Using this repo

```sh
uv sync --extra dev --extra modal
uv run --extra dev --extra modal pytest -q
uv run ruff check src tests scripts
```

- **Rerun a submission:** the easiest path is the public Kaggle notebooks above. p50-sub reruns as is, with code,
  pinned wheels and weights attached. v85p-sub's 3-bit expert weights were deleted, so it needs a rebuild.
- **Rebuild a submission or a training run:** check out its branch and follow the "Reproduce" line in the matching
  [docs/experiments/](docs/experiments/README.md) file, which names the script, config and branch.
- **Compute:** GPU jobs run on Modal (`modal`) or on rented RTX PRO 6000 machines (`scripts/nebius_validation.py`,
  `tools/ops/nebius-config.example.json`); both need your own accounts. Some launchers name our private Kaggle
  datasets and storage volumes; point them at your own copies. Model weights are not included.

## License

Apache-2.0 (see [LICENSE](LICENSE)). Third-party code and data included here are listed in [NOTICE](NOTICE).

## Acknowledgments

We thank Protocol Labs Neuro for compute support. The ARC-AGI-3 agent builds on Tufa Labs' public Duck harness and
dfranzen's public ARC-AGI-3 Milestone 2 notebook (Apache-2.0).
