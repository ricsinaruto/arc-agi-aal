# Post-training practice: sourced notes (9 October 2026, second pass)

Check every training design against section 13. Source marks (section 15): `[read]` the author downloaded the text and found the numbers in it (S33, S34 through the fetch tool's summary). Second pass: all first-pass numbers were rechecked in the papers' own text (one correction, the Nebius task count), eight gaps were searched again (section 9), and section 10 is new.

## 1. Data

- **Small SFT sets work when targets carry what the student lacks.** 491 teacher trajectories: 32B model 7.0% → 20.6% on SWE-bench Verified, no saturation (S1). 6,548 own successes: 72B model 11.4% → 20.5%, where the base failed on format (S2). *For us:* Flash has no format deficit and no teacher.
- **A second round on own successes stalls or hurts.** Adding 868 own successes: 15.3% → 8.7% (S1). *For us:* run 2's null is the published norm.
- **Uncapped successes bias toward easy tasks.** Best cap is 2 per task: 9.7% against 9.3% uncapped (S1). *For us:* 79% of run-2 tokens came from levels cleared 80%+.
- **Own rollouts transferred once, at breadth.** Paprika: 20 samples per task, fewest-turn success against a random worse attempt; 17,181 trajectories and 5,260 preference pairs over 10 task groups; +47% relative; 9 of 10 left-out groups improve (S3). *For us:* our recipe at 200 times our traces. The RL pool's 387 wins and 301 losses are unused pairs.
- **Corrections on the student's own states beat more demonstrations.** An expert takes over mid-rollout: 3,000 demonstrations 36.0%, 961 corrections plus 961 demonstrations 39.0% (32B) (S36). DAgger-style mixing: SFT 22.9% → 27.3% (4B); GRPO alone 11.6% (S37). With a tenth of the data: 11.4% → 19.0% (S49). All three use a stronger model as labeller. *For us:* our labellers are search and best-of-N from a restart.
- **Recovery data without a teacher.** Agent-R: tree search with the model's own policy splices a failed prefix onto a successful continuation; 5,500-8,000 such trajectories per task; average 70.71 against 62.36 for appending the success after the whole failure (8B) (S4). Hindsight relabelling of failures: +7.6-11.4 over success-only SFT (S38). *For us:* exact restarts give this directly: restart at a stuck state, sample N continuations, keep the cheapest that clears.
- **General data in the mix.** Agent data at 20%: held-out 1.40 against 0.87 agent-only (S5). *For us:* no replay used.



## 2. Reasoning traces

- **Reasoning SFT teaches structure, not content.** 17K traces: +40 points AIME24 (32B); corrupting half the digits costs 3.3 points, shuffling steps 13.3 (S6). *For us:* pick traces by structure (a probe that tests a rule before adopting it), not by outcome.
- **Repetition beats breadth for teacher traces; validation loss misleads.** 128 passes over 400 traces beat one pass over 51,200 by 12-26 points at equal updates. Gains stop when training token accuracy saturates (98% at 16 passes); validation loss rises throughout (S8). *For us:* run 2 was selected and stopped on held-out loss after two passes.
- **Answer-only targets collapse the thinking mode.** Qwen3-8B fine-tuned on chemistry answers behind empty think tags: on GSM8K valid reasoning 98.4% → 60.2%, accuracy 95.3 → 79.5. With the think region masked from the loss: 99.8% and 96.5 (S39). *For us:* human-action rows and no-think training keep the empty block out of the loss, and every adapter gets a thinking-mode check.
- **Mixing modes.** Qwen3 fusion cost thinking-mode AIME'24 83.8 → 81.9 (S9). Reasoning leaks into no-think; what helps: more data, think and no-think answers from different questions, moderately more no-think data, reasoning trained first; no-think length 1,085 → 585 tokens (S40). AdaptThink forces half the samples to skip thinking, bonus 0.05: 53% fewer tokens, +2.4 accuracy (S10). *For us:* the published target is "skip when easy".
- **Multi-step inference does not distil into no-think.** About 7% against 52.8% (S11). *For us:* expect cheaper execution, not better rule inference.
- **Token cost enters as a budget.** Zero reward beyond a limit tightened 4K → 3K → 2K: 8,763 → 4,494 tokens, 85.6 → 82.4 (S12).



## 3. Adaptation method

- **MLP layers matter most.** Attention-only rank 256 loses to MLP-only rank 128 at equal parameters, also on Qwen3-30B-A3B with one LoRA per expert (S13); same in S14. *For us:* our 372 matrices leave out every routed expert, 97% of the weights.
- **For MoE, train the experts the task routes to.** ESFT: 5-15% of experts per layer (1.4B of 15.7B), router and the rest frozen: 50.2 against 51.0 full and 44.9 LoRA; general ability 60.6 against 58.8 full (S15). *For us:* measure routing concentration on game play; train those experts. Section 4 gives the merge constraint.
- **LoRA equals full fine-tuning at our data size.** Same loss on small and medium sets; RL equal at rank 1 (S13). Rank 256 0.498 against 0.497 full; rank 16 0.358 (S14). *For us:* full fine-tuning cannot fix a data problem.
- **The learning rate is swept.** LoRA optimum is 10x full fine-tuning, 15x under 100 updates, within 2x across ranks 4-512 (S13). Sweep 1e-5 to 5e-4, take the highest stable, alpha = 2r (S14). *For us:* every run used one unswept rate.



## 4. Quantization

- **Rounding erases updates smaller than the step.** 4-bit rounding restored 83% of unlearned knowledge against 21% at full precision; 8-bit behaved like full precision (S16). *For us:* matches our 8-bit (no cost) and expert-adapter (no movement) results.
- **A naive merge into NVFP4 deletes the adapter.** Dequantise, add the LoRA, requantise: Llama-8B 93.0 → 55.6; Qwen3-30B-A3B -23.3 points; MXFP4 MoEs (gpt-oss-120B) only -0.8 to -2.6. A grid-aware adapter merges exactly, on the grid it trained for (S17). *For us:* our experts are NVFP4. An expert adapter needs grid-aware training, or experts kept BF16 until a distillation step.
- **Recover a quantised build by distilling, not fine-tuning.** NVFP4 student against its BF16 teacher: AIME25 63.5 BF16, 58.7 post-training, 46.1 quantisation-aware fine-tuning, 62.0 distilled; about 0.3B tokens; lr 1e-6 to 1e-5; off-domain data works (S18). *For us:* the path to the submission build after a BF16-side gain.
- **Calibrate on the target domain.** GPTQ at 3 bits: 28.3 → 38.1, BF16 48.7 (S19). *For us:* recalibrate the INT3 build on game play.



## 5. Hyperparameters and dose

- Recipes: section 12. LoRA 5e-4 at rank 64 (S1), 1e-4 at rank 128 (S20); full 5e-6 to 1e-4; 1 to 128 passes; warm-up 5-10% (S7, S8) or none (S13).
- **Tiny data, high rank, large effect on the trained task.** Rank-128 LoRA on at most 250 examples per task: 18.3% → 47.1% (S20). 2025 winners: rank 512 for 175,000 steps offline, per-task LoRA at test time (S46); full fine-tune then rank-256 test-time LoRA (S45). *For us:* a trained-starts test should show a large effect if the pipeline works.
- **Dose is measured in behaviour.** KL from the base on the new task predicts forgetting (R² 0.96; 0.71 on LLMs); weight-change L1 0.34, spectral norm 0.58 (S41). Training token accuracy marks completion (S8). *For us:* drop weight-norm targets; use probe KL and training token accuracy.
- **Long contexts.** Full backward over 131K by context parallelism on 128 GPUs (S2). LongStraw caches the prompt state without gradient and trains a 27B model at 2M tokens in 97.5 GB per GPU; it does not measure the bias (S42). *For us:* our truncated window has a published twin and no published validation.



## 6. RL for LLM agents

- **Scale.** S2: over 100 iterations of 100-300 problems × 10 rollouts, 20.5% → 39.0%. DeepSWE: 4,500 tasks, 200 updates, 64 H100 for six days, 23% → 42% (S21). Qwen3: 170 updates (S9).
- **Information.** Policy gradient gives O(1) bits per episode, about 1,000x less per token than SFT (S13). *For us:* per-decision credit is the only way up without more rollouts.
- **Credit.** S2 puts one advantage on every token and names it as noise. Values from restarts (9 continuations per state) reach PPO's peak in up to 9x fewer updates; learned critics are near chance (S22).
- **A fixed pool with step-level credit has worked.** DigiRL offline only, 1,552 trajectories: 14.6% → 62.8%, filtered cloning 50.7%; with online data 71.9% (S23). Replay saves up to 40% of compute (S50); data 256 updates stale matches on-policy inside a trust region (S26). *For us:* that base was weak (14.6%); ours clears 80% of pool attempts.
- **Negatives off-policy.** The negative term is unbounded and collapses the policy. Fix: positives as plain weighted likelihood, negatives with the ratio truncated to [0, 1]; best baseline leaves 10-20% effective positives; PPO clipping stalls off-policy (S24). Positive-only 56.1 against GRPO 56.3; dropping all-right and all-wrong groups 56.4 (S25). *For us:* the signed arm's drift is the predicted failure.
- **Settings.** lr 1e-6 to 2e-6 full, 128-512 trajectories per update (S2, S10, S25); clip-higher, no KL, no std normalisation, leave-one-out (S21). Reward: binary success plus a linear step penalty; top-k and min-p in the sampler degraded training within 5-10 iterations (S2).
- **RL generalises where SFT memorises.** Rule variants: RL +3.5 and +11.0, SFT -8.1 and -79.5 (S27).



## 7. Distillation and amplification

- **Search plus hindsight relabelling.** SOAR, four rounds: 32B 25.25% → 44.37% on ARC (S28). *For us:* a failed attempt is a verified example of "this sequence produces this board".
- **World-model tasks help as a warm-up, not alone.** Own alternative actions at expert states, next state as target, then imitation: WebShop 47.3 → 58.6; out-of-domain ALFWorld 63.3 → 78.1; an eighth of the demonstrations beats full imitation (S29). Next-state prediction learned by RL, then policy RL: ALFWorld 87.9 against 80.4 when it was learned by SFT; that SFT alone left the agent at 2.8 (S43). *For us:* simulator-labelled side tasks come first, policy data after, with a play check between.
- **On-policy distillation is cheapest when a teacher exists.** 74.4 against 67.6 for RL at a tenth of the GPU hours (S9). *For us:* Flash plus search rescues 3% of never-cleared branches.
- **Reverse-only OPD mode-seeks on our pruned student.** c7h (top-128 sampled reverse + forward KD 0.25, lr 7e-6), step 16 vs D: sampled reverse -23% but teacher-H forward KL +22%, reverse +16%. GKD (Agarwal+ 2024) mixes forward and reverse divergences to counter this. Our 0.25 forward weight is from probe P1-y (teacher-H fwd 1.04-1.05x D at 6 steps, lr 5e-6); c7i tests 0.5.
- **Cross-tokenizer.** 69.2 against 64.6 for SFT, original 71.2 (S30); moot under the no-foreign-text rule.



## 8. Generalisation

- **Breadth beats depth.** 1, 4, 16 training environments: each step better on 50 held-out ones; 400 environments +3.37 against +0.49 for 3x the compute on the old set (S44). 0 → 2,000 environments: 18.4% → 38.5%, largest steps at 10 → 100 → 500 (S47). 700 trajectories from 4, 25, 50, 100 repositories: log-linear (S48). Up to 10,000 levels close the gap (S31). NVARC: models "quickly overfit to the provided 1000 training puzzles"; it won with 103K synthetic puzzles (S45). *For us:* 18 starts cannot transfer; the 163 gated games are the lever.
- **Exploration is trainable and transfers.** S3. Oracle exploration data from an easy task lifted a harder one 34.0 → 48.4 (S32). *For us:* "wrong rule after one observation" is an exploration failure.
- **ARC-AGI-3.** Preview winner: a CNN trained online, 12.58% (S51). The 2026 reports are inference-time world models on the public 25: Twin 93.3 against 61.1 for the same model in a standard harness; milestone-1 winners are prompt-only (S52, S34). The official leaderboard discounts agents "trained on many synthetically generated ARC-AGI-3 lookalike environments" (S51, section 4.3.1: the Foundation's own verified leaderboard, not Kaggle's). Kaggle rules, read 11 Oct: section 6 allows external data and models that are public or "reasonably accessible to all" at "minimal cost"; nothing on synthetic or lookalike environments; winners must open-source code and describe data.
- **Power.** Paired, clustered errors; a 3-point effect at 80% power needs about 969 items; clustered error was 3x naive (S35). *For us:* 88 attempts see 11 points.



## 9. Pruning recovery

- **Dense pruning needs tens of billions of tokens.** Minitron: Llama-3.1-8B -> 4B (half the depth or width) recovered by distillation on 94B tokens; depth pruning recovers worse (GSM8K 16.8 depth vs 41.2 width) (S53). Sheared-LLaMA: 7B -> 1.3B/2.7B with 0.4B pruning + 50B continued-pretraining tokens (S54).
- **MoE expert pruning is near-lossless only to ~50%.** REAP (router-weighted expert activation pruning) keeps 97.6% coding / 96.7% SWE-Bench on Qwen3-Coder-480B at 50% of experts, one-shot (S55); NAEE prunes Mixtral 8x7B experts post-training from a small calibration set (S56).
- *For us:* p75d4 drops 75% of experts plus 4 of 48 layers, past the one-shot regime. Our heals used ~10M reply tokens, ~10^4 below general recovery; the 21 Oct dose fit (held-out reply KL -0.03 per doubling, KL ~ T^-0.18) puts a base-like reply KL (~0.1) at ~0.2-1.2B task tokens.

## 12. Scale check


| Run                        | Data                             | Updates × batch, lr             | Trained         | Result              |
| -------------------------- | -------------------------------- | ------------------------------- | --------------- | ------------------- |
| s1, 32B (S7)               | 1,000 teacher traces             | 315 × 16, 1e-5                  | all             | MATH500 93.0        |
| SWE-Gym, 32B (S1)          | 491 teacher trajectories         | 5 passes × 8, 1e-4              | all             | 7.0 → 20.6%         |
| Nebius SFT, 72B (S2)       | 6,548 own successes              | 50 × 64, 5e-6                   | all             | 11.4 → 20.5%        |
| Nebius RL, 72B (S2)        | 7,249 tasks, 10 rollouts each    | 100+ iterations × 128-256, 1e-6 | all             | 20.5 → 39.0%        |
| Qwen3 RL, 235B (S9)        | 3,995 queries                    | 170                             | all             | 70.1 → 85.1         |
| Paprika, 8B (S3)           | 17,181 trajectories, 5,260 pairs | batch 32, 1e-6                  | not stated      | +47% relative       |
| DigiRL offline, 1.3B (S23) | 1,552 trajectories               | not stated                      | all             | 14.6 → 62.8%        |
| NVARC, 4B (S45)            | 103K puzzles, 3.2M samples       | 32 H100 for 27 h                | all             | 24.03%, first (S33) |
| Run 2                      | 1,884 own replies, 1.94M tokens  | 158 × 4, 2e-4                   | LoRA r64, 134M  | 46 against 42 of 88 |
| RL pilot                   | 240 of 1,504 episodes            | 60 × 4, 1e-4                    | LoRA r16, 33.5M | no separation       |
| D2                         | 85 own wins, 2.2M tokens         | 4 passes, 5e-4                  | LoRA r256, 536M | pending             |
| No-think RL                | 255 windows                      | 85 × 3, 2e-4                    | LoRA r32, 67M   | pending             |


Sources rarely report tokens. Adapter sizes scale from 33.5M at rank 16.



```markdown
`Findings`
1. Answer-only targets can collapse thinking. On a Qwen3-8B, valid reasoning on another task fell from 98.4% to 60.2%; keeping the think region out of the loss held it at 99.8%. This bears directly on the no-think RL loop and any no-thinking SFT items. Our training now masks the think block and add a thinking-mode read-out.
2. Weight-change norm is a poor dose measure (R² 0.34 against 0.96 for a KL probe). D2 and the RL loop now stop on training token accuracy and probe KL.
3. Corrections on the student's own states beat more demonstrations (36.0% → 39.0% in one study; SFT 22.9% → 27.3% in another).
4. Breadth beats depth: 400 environments gave +3.37 against +0.49 for three times the compute.
5. A naive LoRA merge into the 4-bit format deleted the adapter in one report (−23.3 points). That is a problem for the later submission build.
```

## 13. Pre-launch checklist

1. Do the targets hold behaviour the base produces under half the time in that state, measured?
2. Has this trainer overfit a small set and shown it in play on the trained items?
3. Are collection, training and evaluation on one build and one sampler?
4. Are task-routed experts trained, or is the reason for leaving them out written down?
5. Was the learning rate picked from three or more short probes?
6. Is the stop rule behavioural (training token accuracy, probe KL, play), not held-out loss or weight norm?
7. Does the test have 80% power for the smallest effect worth having, on starts away from ceiling?
8. Does a read-out separate better from worse behaviour?
9. Is there a per-start cap, with under a fifth of tokens from starts cleared over 80%?
10. RL: is credit per decision, with ratio-truncated negatives?
11. Is the run one point of a three-point curve, or labelled a pipeline test?
12. Is thinking-mode behaviour checked after training, and the path to the quantised build costed?
13. Is the positive control measured on the served model (top-1 on the trained chains through the play path), not only in the trainer? (added 10 Oct: run 2 was served at one eighth strength; D2's served accuracy was never measured)
14. Does a damage probe generate full sampled replies on untrained contexts (length, share at the reply cap), with a hard stop near 0.02 nat of drift? (added 10 Oct: D2 drifted 0.091 nat and its 32-token probe could not see replies running to the cap) Tested 10 Oct: a full-reply probe on contexts the base wrote did not flag D2 either (0 of 160 capped replies, base 0); the runaway needs the closed loop. Until a probe is validated, the gate is a short closed-loop play test plus the untrained-text loss (D2: +0.145, bar not yet set).
15. RL: is the weight pushed up and pushed down balanced within each update, and confirmed on the trained windows? (added 10 Oct: 80-92% of trained tokens were weighted down)



## 15. Sources

- S1 Pan+ 2024, *Training Software Engineering Agents and Verifiers with SWE-Gym*, arxiv.org/abs/2412.21139 [read]
- S2 Golubev+ 2025, *Training Long-Context, Multi-Turn Software Engineering Agents with Reinforcement Learning*, arxiv.org/abs/2508.03501 [read]
- S3 Tajwar+ 2025, *Training a Generally Curious Agent*, arxiv.org/abs/2502.17543 [read]
- S4 Yuan+ 2025, *Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training*, arxiv.org/abs/2501.11425 [read]
- S5 Zeng+ 2023, *AgentTuning*, arxiv.org/abs/2310.12823 [read]
- S6 Li+ 2025, *LLMs Can Easily Learn to Reason from Demonstrations*, arxiv.org/abs/2502.07374 [read]
- S7 Muennighoff+ 2025, *s1: Simple test-time scaling*, arxiv.org/abs/2501.19393 [read]
- S8 Kopiczko+ 2026, *Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning*, arxiv.org/abs/2602.11149 [read]
- S9 Qwen Team 2025, *Qwen3 Technical Report*, arxiv.org/abs/2505.09388 [read]
- S10 Zhang+ 2025, *AdaptThink*, arxiv.org/abs/2505.13417 [read]
- S11 Yu+ 2024, *Distilling System 2 into System 1*, arxiv.org/abs/2407.06023 [read]
- S12 Hou+ 2025, *ThinkPrune*, arxiv.org/abs/2504.01296 [read]
- S13 Schulman+ 2025, *LoRA Without Regret*, thinkingmachines.ai/blog/lora [read]
- S14 Biderman+ 2024, *LoRA Learns Less and Forgets Less*, arxiv.org/abs/2405.09673 [read]
- S15 Wang+ 2024, *Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models*, arxiv.org/abs/2407.01906 [read]
- S16 Zhang+ 2024, *Catastrophic Failure of LLM Unlearning via Quantization*, arxiv.org/abs/2410.16454 [read]
- S17 Li+ 2026, *Scale-QLoRA: Code-Invariant Adapter Merging for Native 4-bit Microscaling LLMs*, arxiv.org/abs/2609.04526 [read]
- S18 Xin+ 2026, *Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery*, arxiv.org/abs/2601.20088 [read]
- S19 Liu+ 2025, *Quantization Hurts Reasoning?*, arxiv.org/abs/2504.04823 [read]
- S20 Akyürek+ 2024, *The Surprising Effectiveness of Test-Time Training for Few-Shot Learning*, arxiv.org/abs/2411.07279 [read]
- S21 Together AI 2025, *DeepSWE*, together.ai/blog/deepswe [read]
- S22 Kazemnejad+ 2024, *VinePPO*, arxiv.org/abs/2410.01679 [read]
- S23 Bai+ 2024, *DigiRL*, arxiv.org/abs/2406.11896 [read]
- S24 Le Roux+ 2025, *Tapered Off-Policy REINFORCE*, arxiv.org/abs/2503.14286 [read]
- S25 Xiong+ 2025, *A Minimalist Approach to LLM Reasoning*, arxiv.org/abs/2504.11343 [read]
- S26 Zheng+ 2025, *Prosperity before Collapse*, arxiv.org/abs/2510.01161 [read]
- S27 Chu+ 2025, *SFT Memorizes, RL Generalizes*, arxiv.org/abs/2501.17161 [read]
- S28 Pourcel+ 2025, *Self-Improving Language Models for Evolutionary Program Synthesis*, arxiv.org/abs/2507.14172 [read]
- S29 Zhang+ 2025, *Agent Learning via Early Experience*, arxiv.org/abs/2510.08558 [read]
- S30 Minixhofer+ 2025, *Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching*, arxiv.org/abs/2503.20083 [read]
- S31 Cobbe+ 2020, *Leveraging Procedural Generation to Benchmark Reinforcement Learning*, arxiv.org/abs/1912.01588 [read]
- S32 Nie+ 2024, *EVOLvE*, arxiv.org/abs/2410.06238 [read]
- S33 ARC Prize 2025, *ARC Prize 2025 Results and Analysis*, arcprize.org/blog/arc-prize-2025-results-analysis [read]
- S34 ARC Prize 2026, *ARC Prize 2026: ARC-AGI-3 Milestone Prize #1*, arcprize.org/blog/arc-prize-2026-milestone-1 [read]
- S35 Miller 2024, *Adding Error Bars to Evals*, arxiv.org/abs/2411.00640 [read]
- S36 Lauffer+ 2025, *Imitation Learning for Multi-turn LM Agents via On-policy Expert Corrections*, arxiv.org/abs/2512.14895 [read]
- S37 Li+ 2026, *Revisiting DAgger in the Era of LLM-Agents*, arxiv.org/abs/2605.12913 [read]
- S38 Ding+ 2026, *AgentHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling*, arxiv.org/abs/2603.21357 [read]
- S39 Twist+ 2026, *Reasoning-Trace Collapse: Evaluating the Loss of Explicit Reasoning During Fine-Tuning*, arxiv.org/abs/2605.21127 [read]
- S40 Wang+ 2025, *Demystifying Hybrid Thinking: Can LLMs Truly Switch Between Think and No-Think?*, arxiv.org/abs/2510.12680 [read]
- S41 Shenfeld+ 2025, *RL's Razor: Why Online Reinforcement Learning Forgets Less*, arxiv.org/abs/2509.04259 [read]
- S42 Zhou+ 2026, *LongStraw: Long-Context RL Beyond 2M Tokens under a Fixed GPU Budget*, arxiv.org/abs/2607.14952 [read]
- S43 Yu+ 2026, *Reinforcement World Model Learning for LLM-based Agents*, arxiv.org/abs/2602.05842 [read]
- S44 Zeng+ 2025, *RLVE: Scaling Up Reinforcement Learning for Language Models with Adaptive Verifiable Environments*, arxiv.org/abs/2511.07317 [read]
- S45 Sorokin and Puget 2025, *NVARC solution to ARC-AGI-2 2025*, github.com/1ytic/NVARC [read]
- S46 Franzen+ 2025, *The ARChitects - Technical Report*, lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects [read]
- S47 Dong+ 2026, *Agent-World: Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence*, arxiv.org/abs/2604.18292 [read]
- S48 Yang+ 2025, *SWE-smith: Scaling Data for Software Engineering Agents*, arxiv.org/abs/2504.21798 [read]
- S49 Ye+ 2026, *A Few Teacher Steps Go a Long Way*, arxiv.org/abs/2607.04574 [read]
- S50 Arnal+ 2026, *Efficient RL Training for LLMs with Experience Replay*, arxiv.org/abs/2604.08706 [read]
- S51 ARC Prize Foundation 2026, *ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence*, arxiv.org/abs/2603.24621 [read]
- S52 Skoutnev+ 2026, *Twin: Playing an Unknown Game with a Test-Time Digital Twin*, arxiv.org/abs/2608.14490 [read]
- S53 Muralidharan/Sreenivas+ 2024, *LLM Pruning and Distillation in Practice: The Minitron Approach*, arxiv.org/abs/2408.11796 [read: tokens, depth vs width table]
- S54 Xia+ 2024, *Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning*, arxiv.org/abs/2310.06694 [read: token budget]
- S55 Lasby+ 2025, *REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression*, arxiv.org/abs/2510.13999 [read: summary, blog figures]
- S56 Lu+ 2024, *Not All Experts are Equal: Efficient Expert Pruning and Skipping for MoE LLMs*, arxiv.org/abs/2402.14800 [read: method, calibration ablation]

