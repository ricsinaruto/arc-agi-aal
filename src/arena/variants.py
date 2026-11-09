"""Single-factor arena studies with explicit agent or serving interventions."""

import copy

from arc_infra.model_templates import allowed_efforts

VARIANTS = ("control", "phase2-four", "context-64k", "grant-180s", "graph-128",
            "cache-reclaim", "reclaim-mtp1", "reclaim-context-64k",
            "reclaim-phase2-four", "reclaim-grant-180s", "reclaim-graph-128",
            "reclaim-grant-2t1", "reclaim-effort-medium", "reclaim-64k-keep3", "reclaim-regrant-2x",
            "reclaim-fixed-2x", "fixed-2x-64k-four", "fixed-2x-effort-medium", "fixed-2x-effort-low",
            "reclaim-mtp4", "fixed-2x-mtp4", "fixed-2x-t1cap-120", "fixed-2x-switch-3",
            "fixed-2x-pool", "fixed-2x-80k-three", "fixed-2x-stall-open360", "reclaim-ngram", "reclaim-ngram8", "fixed-2x-notes", "fixed-2x-memory",
            "fixed-2x-pool2-cap", "fixed-2x-pool2-timed-novelty", "fixed-2x-progress", "fixed-2x-notes-live", "fixed-2x-escalate",
            "fixed-2x-snapshot", "fixed-2x-progress-novelty", "fixed-2x-reopen-pause", "fixed-2x-reopen-run",
            "fixed-2x-slots-1", "fixed-2x-slots-4", "fixed-2x-slots-8", "fixed-2x-budget-1x", "fixed-2x-budget-4x", "fixed-2x-llm-rank", "fixed-2x-procedure", "fixed-2x-thompson", "fixed-2x-coverage",
            "fixed-2x-explorer", "fixed-2x-token-budget", "fixed-2x-index", "fixed-2x-index-lane", "fixed-2x-gittins-broad", "fixed-2x-gittins-c110", "fixed-2x-no-think", "fixed-2x-adaptive-think", "fixed-2x-worldmodel", "fixed-2x-commit", "fixed-2x-commit-queue", "fixed-2x-explorer-start",
            "fixed-2x-explorer-start-silent", "fixed-2x-critical", "fixed-2x-play", "fixed-2x-harness-v2",
            "fixed-2x-mtp0-wide", "fixed-2x-level-probe",
            "fixed-2x-gittins-c110-wide", "fixed-2x-mtp0-wide-kernels", "fixed-2x-deep-explore",
            "fixed-2x-mtp3-wide-kernels", "fixed-2x-depthwise", "fixed-2x-depthwise-mtp3",
            "fixed-2x-mtp3-wide-tweaks", "fixed-2x-depthwise-mtp3-tweaks",
            "fixed-2x-mtp3-wide-tweaks-nople", "fixed-2x-depthwise-mtp3-tweaks-nople",
            "fixed-2x-depthwise-mtp3-fixed-order",
            "fixed-2x-mtp3-wide-kernels-kv8", "fixed-2x-mtp0-wide-kernels-kv8",
            "fixed-2x-mtp3-wide-stack", "fixed-2x-depthwise-mtp3-stack", "fixed-2x-mtp0-wide-kv8-stack",
            "fixed-2x-depthwise-mtp3-w4a16", "fixed-2x-mtp0-wide-kv8-hero",
            "fixed-2x-mtp0-wide-kv8-hero-21x42k", "fixed-2x-mtp0-wide-kv8-hero-uva", "fixed-2x-mtp0-wide-kv8-hero-6x64k",
            "fixed-2x-mtp0-wide-kv8-hero-prune12", "fixed-2x-mtp0-wide-kv8-hero-prune25",
            "fixed-2x-mtp0-wide-kv8-hero-prefix",
            "fixed-2x-depthwise-mtp3-gptq-arc", "fixed-2x-depthwise-mtp3-gptq-arc-w4a16", "hero-v2-rank",
            "fixed-2x-mtp0-wide-kv8-hero-p6",
            "fixed-2x-mtp0-wide-kv8-hero-gptq-arc", "fixed-2x-mtp0-wide-kv8-hero-prune12-gptq-arc",
            "fixed-2x-mtp0-wide-kv8-hero-prune25-gptq-arc")

# Serving-only: vLLM prompt-lookup (n-gram) speculation in place of the MTP head, for an agent
# loop that re-emits code and board text. The pinned serving script launches with
# --async-scheduling, which the pinned vLLM only allows with the GPU n-gram proposer
# (`ngram_gpu`; the CPU `ngram` method fails config validation), so the draft length and the
# lookup window are the study knobs and every other server flag stays the MTP-3 control's.
NGRAM_SPECULATION = {"method": "ngram_gpu", "num_speculative_tokens": 3,
                     "prompt_lookup_max": 4, "prompt_lookup_min": 2}
# Long prompt-lookup draft for replayed transcripts (25 September): when the model re-emits
# code or observations already in its context, an eight-token draft accepts whole spans.
NGRAM8_SPECULATION = {"method": "ngram_gpu", "num_speculative_tokens": 8,
                      "prompt_lookup_max": 6, "prompt_lookup_min": 2}

# Focus slots -> window (26 September): the largest candidate window with slots * window inside
# the 0.9 margin of the Flash server's 291,900-token pool.
SLOT_WINDOWS = {1: 131072, 4: 65536, 8: 32768}

# Throughput arm (27 September; serving lab v1 and the level-bench calibration): MTP-0 serving (speculative
# decoding off) drops the 5 GiB MTP head and shrinks the fixed hybrid state per request from 22 blocks to 10
# half-size blocks: 751 server blocks (675 at the 0.9 margin) and 91 per full 64K request, so seven fit.
MTP0_WIDE = {"mtp_tokens": 0, "active_limit": 6, "window": 65536}

# Composed variants whose vLLM server differs from the fixed-2x base's.
SERVING_CHANGED = ("fixed-2x-mtp4", "fixed-2x-commit", "fixed-2x-commit-queue", "fixed-2x-mtp0-wide",
                   "fixed-2x-gittins-c110-wide", "fixed-2x-mtp0-wide-kernels", "fixed-2x-deep-explore",
                   "fixed-2x-mtp3-wide-kernels", "fixed-2x-depthwise", "fixed-2x-depthwise-mtp3",
                   "fixed-2x-mtp3-wide-tweaks", "fixed-2x-depthwise-mtp3-tweaks",
                   "fixed-2x-mtp3-wide-tweaks-nople", "fixed-2x-depthwise-mtp3-tweaks-nople",
            "fixed-2x-depthwise-mtp3-fixed-order",
            "fixed-2x-mtp3-wide-kernels-kv8", "fixed-2x-mtp0-wide-kernels-kv8",
            "fixed-2x-mtp3-wide-stack", "fixed-2x-depthwise-mtp3-stack", "fixed-2x-mtp0-wide-kv8-stack",
            "fixed-2x-depthwise-mtp3-w4a16", "fixed-2x-mtp0-wide-kv8-hero",
            "fixed-2x-mtp0-wide-kv8-hero-21x42k", "fixed-2x-mtp0-wide-kv8-hero-uva", "fixed-2x-mtp0-wide-kv8-hero-6x64k",
            "fixed-2x-mtp0-wide-kv8-hero-prune12", "fixed-2x-mtp0-wide-kv8-hero-prune25",
            "fixed-2x-mtp0-wide-kv8-hero-prefix",
            "fixed-2x-depthwise-mtp3-gptq-arc", "fixed-2x-depthwise-mtp3-gptq-arc-w4a16", "hero-v2-rank",
            "fixed-2x-mtp0-wide-kv8-hero-p6",
            "fixed-2x-mtp0-wide-kv8-hero-gptq-arc", "fixed-2x-mtp0-wide-kv8-hero-prune12-gptq-arc",
            "fixed-2x-mtp0-wide-kv8-hero-prune25-gptq-arc")

# 8-bit QSA KV cache (28 September, docs/validation-20260927/kv-cache-8bit.md): the phase-4 serving with the main KV
# cache in int8 (per-token-head fp32 scales) holds a 64K window in 62 blocks of 1,632-1,664 tokens instead of 101 of
# 832, so the same 14.7 GiB pool fits nine full windows instead of six (served layout in the doc). The arm runs
# KV8_FOCUS_SLOTS focus games; require_focus_windows_fit refuses to start on a server that fits fewer.
KV8_MODE = "int8"
KV8_FOCUS_SLOTS = 9
# The kv8 smoke (28 September) served 661 blocks of 1,664 tokens, 61 per 64K window, 594 at the gate: nine fit. It also
# decoded nine streams outside the full CUDA graphs: MTP-3 verifies 4 tokens per stream, so 9 x 4 = 36 exceeds the
# default capture size 32 (the 64K x 9 point ran at 17.9 iterations/s against 29.6 at six streams). The arm captures
# up to 48 tokens (sizes 40 and 48 added); this setting is not yet measured on the server.
KV8_GRAPH_CAPTURE_SIZE = 48
# The MTP-0 kv8 arm (kernels-v2 plus the top-k-first sampler, the int8 KV cache and the most 64K
# streams that fit; the opening switch stays at kernels-v2's six clears). MTP-0 already serves the bf16 recurrent state
# and kernels-v2 already ran with the A+C-freed memory in the pool (875 blocks of 800 tokens, 16.5 GiB). With int8 KV
# that pool is ~845 blocks of 1,584 tokens, 760 at the 0.9 gate, 51 per 64K window (42 attention + 9 fixed): 14.9, so
# 14 fit (CPU sizing, scripts/kv8_sizing.py; require_focus_windows_fit checks the served layout). One token per stream
# per step keeps 14 streams and the opening's 25 sequences inside the default CUDA-graph capture size (32).
MTP0_KV8_FOCUS_SLOTS = 14
HERO_21_SLOTS, HERO_21_WINDOW = 21, 42496  # fixed-2x-mtp0-wide-kv8-hero-21x42k
HERO_6_SLOTS = 6  # fixed-2x-mtp0-wide-kv8-hero-6x64k: kernels-v2's six focus games
# Load-time expert pruning on the hero (docs/validation-20260927/expert-usage.md): the least-used experts per layer by
# gate-weight mass on the recorded usage set (run arena-expert-usage-v4-modal-20260928), dropped at load
# (arc_infra.expert_prune_patch). 12.5% and 25% passed the 1.25x-noise bar on held-out games. Each pruned expert
# frees 2.765 MB in each of 48 layers. Focus slots are projected on the hero smoke's served layout (860 blocks,
# 51 per 64K window, the 0.9 gate; 60 blocks per 1.18 GiB as in the hero-uva projection), capped by the 25
# sequences the server admits; require_focus_windows_fit checks the served layout at the trial start.
# Prefix cache v2 (29 September, docs/validation-20260927/prefix-cache-v2.md; arena.prefix_cache_patch): the hero
# serving with the Mamba chunk alignment fix (prefill chunks end on the 1,600-token Mamba block instead of the QSA
# ring's 4-token block, so each request writes the state at its prompt's last block boundary and no cached block holds
# a chunk-end state under a later boundary's hash), the newest boundary state kept with its request until it ends, and
# superseded states (never resumed from) put at the front of the free queue instead of the MRU end.
HERO_PREFIX_CACHE = {"mamba_align_fix": True, "mamba_keep_state": True, "mamba_demote_superseded": True}
EXPERT_BYTES, MOE_LAYERS, ROUTED_EXPERTS = 2.765e6, 48, 512
HERO_BLOCKS, HERO_WINDOW_BLOCKS, HERO_BLOCK_BYTES = 860, 51, 1.18 * 2**30 / 60
EXPERT_PRUNE_VARIANTS = {
    "fixed-2x-mtp0-wide-kv8-hero-prune25-layer": {
        "fraction": 0.25, "source": "data/expert_prune/prune25-layer.json", "mask": "configs/expert-prune/prune25-layer.json",
        "sha256": "6ea3dfd2373f64d09f71eea1dd3939011100917226368f61a31ee07eb4cca8cc", "served_blocks": 1667},
    "fixed-2x-mtp0-wide-kv8-hero-prune12": {
        "fraction": 0.125, "source": "data/expert_prune/prune12.json", "mask": "configs/expert-prune/prune12.json",
        "sha256": "e625b8b6b54e23d311b3a46a7d4573231569db4b8011da1beefe08babd093193", "served_blocks": 1245},
    "fixed-2x-mtp0-wide-kv8-hero-prune25": {
        "fraction": 0.25, "source": "data/expert_prune/prune25.json", "mask": "configs/expert-prune/prune25.json",
        "sha256": "1aea34ddaacea5bbad339323259a24a164e9c3033e6d34790ead2bac05969a09", "served_blocks": 1667},
    # Layer-aware 37.5% (the blocks-2 campaign arm blocks2-prune37). Not measured on a GPU:
    # served_blocks is projected on the prune25 check's 1,667 int8 blocks + the 3,072 further experts' bytes
    # (3,072 x 2.7648 MB = 8.49 GB / 21.20 MB per int8 block = +400).
    "fixed-2x-mtp0-wide-kv8-hero-prune37-layer": {
        "fraction": 0.375, "source": "data/expert_prune/prune37-layer.json", "mask": "configs/expert-prune/prune37-layer.json",
        "sha256": "6b4f18b651cf8dc52dbf9f6e2f986eba73fbb159798f49817f4f53cfc36ca2df", "served_blocks": 2067},
    # Layer-aware 50% ("blocks-2 + plans, pruned 50%, 3-bit experts"). Not measured on a GPU:
    # served_blocks is projected on the prune25 check's 1,667 int8 blocks + the 6,144 further experts' bytes
    # (6,144 x 2,764,800 B = 16.99 GB / 21,196,800 B per int8 block = +801).
    "fixed-2x-mtp0-wide-kv8-hero-prune50-layer": {
        "fraction": 0.5, "source": "data/expert_prune/prune50-layer.json", "mask": "configs/expert-prune/prune50-layer.json",
        "sha256": "e2c395c875a30aef633aff326aee2642f3100f141501eac51c67d7684b8165b7", "served_blocks": 2468},
    # Tuning-family masks (throughput arm phase 5, 4 October; LOG section 18). served_blocks projected on prune12's
    # 1,245 measured int8 blocks + the further experts' NVFP4 bytes (2,764,800 B / 21,196,800 B per int8 block).
    "fixed-2x-mtp0-wide-kv8-hero-prune-t12": {
        "fraction": 0.125, "source": "data/expert_prune/prune-t12.json", "mask": "configs/expert-prune/prune-t12.json",
        "sha256": "e30f6a81f27534155958f86e2b4ce30b37f7e728294fdd8bcad4fbc1105143de", "served_blocks": 1245},
    "fixed-2x-mtp0-wide-kv8-hero-prune-t25": {
        "fraction": 0.25, "source": "data/expert_prune/prune-t25.json", "mask": "configs/expert-prune/prune-t25.json",
        "sha256": "fed40a2af7ddd6b4490eb7fd078cb2292736cc11aff73dd723bcff4343355ef1", "served_blocks": 1645},
    "fixed-2x-mtp0-wide-kv8-hero-prune-t37": {
        "fraction": 0.375, "source": "data/expert_prune/prune-t37.json", "mask": "configs/expert-prune/prune-t37.json",
        "sha256": "95ca64d4043600139655305ef75ef802b4f6fe22a0ac5003f956215d6b2efb81", "served_blocks": 2046},
    "fixed-2x-mtp0-wide-kv8-hero-prune-t37-all": {  # v7 submission refit: ranked on all 25 public families
        "fraction": 0.375, "source": "data/expert_prune/prune-t37-all.json",
        "mask": "configs/expert-prune/prune-t37-all.json",
        "sha256": "529677a0c3e3768f783e4831c3a591a1f3ba9cfab59c1cc3c976c65dd2cd3fb3", "served_blocks": 2046},
    "fixed-2x-mtp0-wide-kv8-hero-prune-t37-bal": {  # balanced all-25 refit: 380K tokens per family
        "fraction": 0.375, "source": "data/expert_prune/prune-t37-bal.json",
        "mask": "configs/expert-prune/prune-t37-bal.json",
        "sha256": "bf60c595c369c6925f241ac61b8d76ca46f70f4b1ffd88ecc77549e1abc151fd", "served_blocks": 2046},
    "fixed-2x-mtp0-wide-kv8-hero-prune-t37-fixed10": {  # RSI gen 0: ranked on the 10 fixed games only
        "fraction": 0.375, "source": "data/expert_prune/prune-t37-int3-requant-fixed10-v1.json",
        "mask": "configs/expert-prune/prune-t37-int3-requant-fixed10-v1.json",
        "sha256": "ea368a44fce2f0e46c127d1808324076ea58d0be6a2881533fb528a814ff7632", "served_blocks": 2046},
    "fixed-2x-mtp0-wide-kv8-hero-prune-t50": {
        "fraction": 0.5, "source": "data/expert_prune/prune-t50.json", "mask": "configs/expert-prune/prune-t50.json",
        "sha256": "7ffab6db4b8224707a715f2a8e84fc2db2f2ce56ec12850c47868bbd76d180f3", "served_blocks": 2447},
}


def pruned_hero_focus_slots(fraction, max_sequences):
    """64K windows that fit the hero layout with ``fraction`` of the routed experts dropped, capped by the sequences."""
    freed = fraction * ROUTED_EXPERTS * MOE_LAYERS * EXPERT_BYTES
    blocks = HERO_BLOCKS + int(freed // HERO_BLOCK_BYTES)
    return min(int(0.9 * blocks) // HERO_WINDOW_BLOCKS, max_sequences)


MTP0_KV8_SERVING = {"mamba_ssm_cache_dtype": None, "moe_backend": None, "draft_experts": None, "hc_fusion": False,
                    "topk_sampler": True, "kv_cache_8bit": KV8_MODE}

# ARC expert quantization (docs/validation-20260927/expert-quant.md) -> the composed variant whose
# study it changes in its server block only. gptq-arc serves the routed experts re-quantized by GPTQ on ARC
# activations (replacement shards in the served NVFP4 format, arena.expert_overlay); w4a16 serves the unchanged NVFP4
# experts through vLLM's Marlin kernel (BF16 activations) instead of the FlashInfer CUTLASS W4A4 path.
EXPERT_QUANT_SOURCE = {"fixed-2x-depthwise-mtp3-gptq-arc": "fixed-2x-depthwise-mtp3",
                       "fixed-2x-depthwise-mtp3-w4a16": "fixed-2x-depthwise-mtp3",
                       "fixed-2x-depthwise-mtp3-gptq-arc-w4a16": "fixed-2x-depthwise-mtp3",
                       # Phase 3 (29 September): the hero serves Marlin W4A16 already, so the overlay alone brings the
                       # GPTQ-ARC experts (held-out MoE error 12.3% -> 9.3%). With load-time pruning the filter drops
                       # the pruned experts from the overlay's shards as from the checkpoint's.
                       "fixed-2x-mtp0-wide-kv8-hero-gptq-arc": "fixed-2x-mtp0-wide-kv8-hero",
                       "fixed-2x-mtp0-wide-kv8-hero-prune12-gptq-arc": "fixed-2x-mtp0-wide-kv8-hero-prune12",
                       "fixed-2x-mtp0-wide-kv8-hero-prune25-gptq-arc": "fixed-2x-mtp0-wide-kv8-hero-prune25"}
# The stage-2 overlay (arena.expert_quant_gptq, run arena-expert-quant-calibrate-v2-modal-20260928): 192 replacement
# shards (63.3 GiB) in the assets volume at flash-input/datasets/richardcsaky/arc-expert-gptq-arc-v1 (Modal only).
EXPERT_OVERLAY_GPTQ_ARC = {"dataset": "richardcsaky/arc-expert-gptq-arc-v1",
                           "manifest_sha256": "97c0caa9fa17649d0108ad1df03a1d0ce36d1c815c778f0cecd9bad9bddd0d00"}
# 3-bit routed experts (30 September; docs/validation-20260930/expert-3bit.md), by group size: GPTQ-ARC v1 re-rounded to
# the int3 grid inside the NVFP4 container (arena.expert_3bit_overlay; research-expert3bit-overlay-v1/v2/v3-20260930; 67,987,279,488 bytes), 192
# shards of the checkpoint's sizes in the assets volume (Modal only). Served on the Humming MoE with
# serving_kernels.expert_int3 (the uint3 repack at load). "size": the manifest's sha256 is pinned and every shard's
# size is checked at start; each shard's sha256 was checked when the manifest was written (with the int3-grid check).
EXPERT_OVERLAY_INT3 = {16: {"dataset": "richardcsaky/arc-expert-int3-g16-v1",
                            "manifest_sha256": "7fca7969efba1d510c59b8430cfe3c357b54f9213ff2f5c58dd24ec4179cc944",
                            "check": "size"}}


# Depthwise scheduling -> the composed variant whose serving (server
# block, window, six slots, full-window fit check) it runs on. A depthwise arm on another serving is one more entry
# here plus its name in VARIANTS, SERVING_CHANGED and COMPOSED (e.g. "fixed-2x-depthwise-mtp3":
# "fixed-2x-mtp3-wide-kernels"); ``_depthwise_variant`` checks that the source serves the depthwise slots and window.
DEPTHWISE_SERVING = {"fixed-2x-depthwise": "fixed-2x-mtp0-wide-kernels",
                     "fixed-2x-depthwise-mtp3": "fixed-2x-mtp3-wide-kernels",
                     "fixed-2x-depthwise-mtp3-tweaks": "fixed-2x-mtp3-wide-tweaks",
                     "fixed-2x-depthwise-mtp3-tweaks-nople": "fixed-2x-mtp3-wide-tweaks-nople",
                     "fixed-2x-depthwise-mtp3-fixed-order": "fixed-2x-mtp3-wide-kernels",
                     "fixed-2x-depthwise-mtp3-stack": "fixed-2x-mtp3-wide-stack"}
# Continuous ranking (``arena.rank``, docs/validation-20260927/sim-hero-v2.md) -> the composed
# variant whose serving (server block, window, focus slots, full-window fit check) it runs on. hero-v2 (hero + the
# host-memory tweaks + expert pruning + ARC GPTQ experts + token budgets) is not composed on this branch yet; the arm
# runs on the 25%-pruned hero, which serves the simulator's recommended 25 x 64K (max_num_seqs 25). When the hero-v2
# variant lands, point this entry at it (``_rank_variant`` checks the source).
RANK_SERVING = {"hero-v2-rank": "fixed-2x-mtp0-wide-kv8-hero-prune25"}
# Depthwise arms that refill in the fixed-2x steady-state order (``depthwise.backlog``) instead of depth first.
DEPTHWISE_FIXED_ORDER = ("fixed-2x-depthwise-mtp3-fixed-order",)

# Follow-ups on the fixed-2x base (24 September evening): one change each, compared with
# the fixed-2x 25-game sanity trial, never with the incremental-clock runs.
COMPOSED = {
    "hero-v2-rank": "continuous ranking on the 25%-pruned hero serving "
                    "(fixed-2x-mtp0-wide-kv8-hero-prune25: 25 focus games at 64K): no opening and no level budgets; "
                    "every 60 s every game is ranked by its next level's value, (levels + 1)^1.5 / (N(N+1)/2), times "
                    "its recency-weighted clears per completion token (each clear weighted exp(-tokens since it / H), "
                    "H = the run's median tokens per cleared level; prior 0.25 clears per 5K tokens); free slots take "
                    "the top games and a waiting game replaces the worst focus game past 10K tokens of tenure while it "
                    "scores more than 1.05x as much (constants tuned in the simulator on held-out halves of the games)",
    "fixed-2x-harness-v2": "harness v2 (failure-signature study, 26 September, edits E1-E5 each behind its own flag): effect-aware receipts, refusals and stuck count that separate timer/counter bars, mid-action animations and scrolls from gameplay; a nudge after three requests without an action; an action legend in the prompt; a new-level notice of objects and colours no earlier level showed; a compaction handoff with a verified action/effect table, a refuted list and conventions",
    "fixed-2x-coverage": "exploration coverage helper (paper notes 2.6 item 9): a pre-loaded coverage() lists the board objects no click has touched and the key actions not yet tried on the level, with change counts; the prompt asks for it when a level stalls",
    "fixed-2x-explorer-start": "probe arm: the model-free explorer takes each game's level 1 before the first model request and the model receives its demonstration",
    "fixed-2x-explorer-start-silent": "probe arm: the model-free explorer takes each game's level 1 before the first model request; no demonstration reaches the model",
    "fixed-2x-play": "action economy: a pre-loaded play([...]) runs a planned sequence and stops at the first action that changes nothing, a level change or a game over; the prompt asks for whole predicted sequences in one call and single actions only to test unverified hypotheses",
    "fixed-2x-critical": "Self-Harness rule (paper notes 2.6 item 1): a python-tool exception carries 'CRITICAL: the last step failed; fix it within 1-2 steps'",
    "fixed-2x-index": "hazard-index slot allocation after the opening: no level budgets; slots go to the games with the highest expected score per token over the next 20K tokens (fitted clear probability from the tokens already spent on the level x the level's weight plus discounted later levels), with a 15K-token quantum and a 25% hysteresis",
    "fixed-2x-index-lane": "fixed-2x-index plus a work-conserving background lane: games outside the two focus slots (level-0 and parked games) send requests on the cache the focus games leave free, at most two at a time, fewest completion tokens on the level first; each focus game keeps a guard for its next request (latest prompt + 2 x cap, clipped to the window) so it never waits on a lane request, and a lane clear re-scores the game for the next index decision",
    "fixed-2x-no-think": "reasoning off: the chat template's enable_thinking=False (an empty think block) on every gameplay request; everything else is the fixed-2x base that scored 7.18% on the hidden set",
    "fixed-2x-adaptive-think": "adaptive thinking (arena.adaptive): xhigh on a game's first request, after a request without an action, after a surprise (no board change, level change, game over), after a compaction and every fourth request; thinking off otherwise (right after actions that changed the board)",
    "fixed-2x-level-probe": "per-level snapshot probe: the fixed-2x-snapshot probe (palette, 20 largest objects, each "
                            "simple available action once in sequence with its transition receipt) plus one click on "
                            "each of the three largest non-background objects, then one RESET to the level's start "
                            "board, run at game start and again at the start of every later level (up to 5 + 3 + 1 "
                            "actions per level); the latest level's block replaces the carried snapshot message and "
                            "is shown once in the first user turn on the level",
    "fixed-2x-worldmodel": "executable world model at level transitions: skills check_sim (replays every recorded transition of the game through the model's sim), plan (BFS in the model) and run_plan (executes, stops at the first surprise), plus a consolidation notice in the first user turn after each level clear asking for read_state/sim, a check_sim pass, then plan/run_plan on the new level",
    "fixed-2x-gittins-broad": "broad opening plus a Bayesian Gittins-style index: the opening also waits until every game has 15K completion tokens (cap 1,500 s); after it, slots go to the best expected score per token over plans 'play up to j more levels, giving up on a level after c tokens', from a hierarchical hazard model (pooled token-band hazard x a per-game factor learned from all the game's levels in this run x a per-level factor); 6K-token quantum, 15% hysteresis (scripts/arena_analysis/schedule_sim.py)",
    "fixed-2x-gittins-c110": "Gittins index for the 110-clone one-hour roster: the two-clear opening with an absolute cap (not rescaled by the clone roster); after it slots go to the Gittins-style index on the within-level hazard shape (the clear rate inside a level rises with tokens; pooled over all public games) with k, ke and beta fitted jointly to it, the game-factor prior's scale estimated in-run from every game's record, and a 4x hysteresis (a waiting game must score five times the worst focus game); 6K-token quantum (docs/validation-20260927/scheduler-c110.md)",
    "fixed-2x-commit-queue": "fixed-2x-commit, but a lane clear takes only a free slot or one held by a focus game past its stall cap; otherwise it waits in a queue (earliest clear first) for the next freed slot",
    "fixed-2x-commit": "momentum commitment with a preemptible lane: after the opening two focus slots go to games that clear levels (a lane clear is promoted at once, replacing the focus game past its stall cap or with the most tokens on its level) and a focus game keeps its slot up to clamp(2.5 x its median tokens per cleared level, 25K, 60K) tokens on the current level; every other game plays in a lane of up to four requests (level-1 games under 25K tokens first, then demoted deep games, fewest tokens first) admitted on an estimate of the cache in use (92% of the pool), with vLLM priority scheduling (focus 0, lane 1) so that only lane requests are preempted",
    "fixed-2x-token-budget": "budget audit fixes (a+b): budgets and T1 on the token clock (1.65 s + completion tokens at the two-stream 104 tok/s, no context term, plus tool seconds) so opening contention no longer inflates T1, and T1 clipped to [0.5, 2] x the median T1 of the games with a clear",
    "fixed-2x-explorer": "model-free graph explorer on waiting games: after the opening switch a CPU explorer (one click per object, every key, breadth-first walks to untried actions, level resets) takes each parked level-0 game while its agent waits for a slot, up to 20,000 actions; a clear flags the game for readmission and the model receives the shortest winning sequence and the region it changed",
    "fixed-2x-thompson": "Thompson backlog: waiting games ordered by a draw from Beta(1 + levels cleared, 1 + admissions that ended without a clear), damped by actions since the last new board (parked clears still first); level-0 games no longer re-enter in roster order",
    "fixed-2x-procedure": "structured stuck procedure: the stuck notice (same cadence) becomes a checklist (evidence, hypotheses CONFIRMED/REFUTED/UNTESTED, bad execution vs bad idea, one probe with a stated expectation) plus a collapse detector (>=80% of the last 30 actions on one action or click region)",
    "fixed-2x-llm-rank": "LLM-ranked backlog: at the opening switch and every 180 s of phase 2 the served model ranks the waiting games from an evidence table (levels, T1, busy seconds, actions, distinct boards, actions since a new board, admissions); admissions follow its order (budgets, parking and parked clears first unchanged)",
    "fixed-2x-slots-1": "one phase-2 focus game; the opening ends at the first level-1 clear (phase 2 waits until every slot has a game)",
    "fixed-2x-slots-4": "four phase-2 focus games at a 64K window (four windows fit the cache); the opening ends at the fourth distinct level-1 clear",
    "fixed-2x-slots-8": "eight phase-2 focus games at a 32K window (eight windows fit the cache, the server's eight sequences); the opening ends at the eighth distinct level-1 clear",
    "fixed-2x-budget-1x": "level budgets 1*L*T1 instead of 2*L*T1 (focus games rotate twice as often)",
    "fixed-2x-budget-4x": "level budgets 4*L*T1 instead of 2*L*T1 (focus games keep their slot twice as long)",
    "fixed-2x-64k-four": "64K window keeping the last three observations verbatim, four phase-2 games",
    "fixed-2x-effort-medium": "reasoning effort medium",
    "fixed-2x-effort-low": "reasoning effort low",
    "fixed-2x-mtp4": "MTP depth 4 (the pinned serving script's maximum) at one to two phase-2 streams",
    "fixed-2x-t1cap-120": "T1 enters the 2*L*T1 budgets capped at 120 busy seconds (audition winners with slow first clears no longer earn 500-850 s level-2 budgets)",
    "fixed-2x-switch-3": "the opening ends at the third distinct level-1 clear instead of the second",
    "fixed-2x-pool": "audition pool: a free phase-2 slot runs the top tied backlog games (up to four) in parallel "
                     "until one clears; budgets and T1 on the normalized clock (tokens at the two-stream rate, "
                     "context-weighted, plus tool time); stalled auditions park",
    "fixed-2x-stall-open360": "a focus game parks (and drops one place) after 15 consecutive actions without a "
                              "board its level has not shown; the opening ends at the second clear or at 360 s, "
                              "whichever comes first",
    "fixed-2x-pool2-cap": "audition pool of two per free slot on the normalized clock, with the 360 s opening cap "
                          "and focus stall parking after 15 actions without a board new to the level (the pool-v2, "
                          "stall-open360-v1/v2 follow-up: which games hold the focus slots early decides the run)",
    "fixed-2x-pool2-timed-novelty": "fixed-2x-pool2-cap plus a 120 s (normalized) time limit per audition and the "
                                    "level-0 backlog ordered by opening evidence: within a tie on readmissions and "
                                    "levels the game that has shown the most distinct boards is admitted first",
    "fixed-2x-progress": "learning-progress backlog order (ideas.md point 2, maths model): the backlog is ranked "
                         "by estimated marginal score per busy second, (levels+1)/(N(N+1)/2) * (levels+0.5)/(busy+90) "
                         "/ (1+stalled_actions/10), highest first; fixed-2x budgets unchanged",
    "fixed-2x-progress-novelty": "learning-progress backlog order, second maths model: value = (levels+0.5+distinct "
                                 "boards on the level/(actions on the level+3))/(busy+90) / (1+stalled_actions/10), no "
                                 "level-share weight (progress v2 ordered the level-0 backlog by fewest total levels and "
                                 "never reached the fast eight-level games)",
    "fixed-2x-reopen-pause": "reopened phase 1 (user design): when a focus slot frees, every backlog game tied at the "
                             "top of the order (readmissions, levels) plays at once with no budget; the first to clear a "
                             "level takes the slot, the others park with their order unchanged; the surviving focus game "
                             "is paused meanwhile and resumes with its budget",
    "fixed-2x-reopen-run": "reopened phase 1 as fixed-2x-reopen-pause, but the surviving focus game keeps playing",
    "fixed-2x-notes-live": "fixed-2x-notes with the notebook also injected at a game's next level clear and next "
                           "compaction (not only at re-admission), so the focus games receive the other games' notes",
    "fixed-2x-escalate": "executable-model escalation (Tycho / Schema): from the second stuck notice on a level the "
                         "notice demands state/transition/goal functions in the persistent interpreter, verified by "
                         "retrodicting the level's recorded transitions, then a searched plan abandoned at the first "
                         "misprediction",
    "fixed-2x-snapshot": "game-start snapshot (Meta-Harness environment bootstrap): before the first model call a "
                         "deterministic probe reads the start board (palette, 20 largest objects) and takes each simple "
                         "available action once in sequence, recording its transition receipt, then restores the board "
                         "with RESET; the block is carried as the first user message for the whole game",
    "fixed-2x-80k-three": "three phase-2 focus games at an 80K window (three windows fit the cache), keeping the "
                          "last three observations verbatim across compactions",
    "fixed-2x-notes": "cross-game notes: at every level clear and park one tool-less request (1,024-token cap, thinking off) appends up to ten general lines to a per-trial notebook, injected from the next admission on as the first user message (2,000-token cap, oldest lines dropped)",
    "fixed-2x-gittins-c110-wide": "combination (27 September, c110 results): fixed-2x-mtp0-wide serving and six focus games at 64K, scheduled by fixed-2x-gittins-c110 (within-level Gittins index, in-run prior scale, hysteresis 4, two-clear opening capped at an absolute 1,200 s) with the index's per-stream rate at 75 tok/s; the simulator's best six-slot policy",
    "fixed-2x-mtp0-wide-kernels-kv8": "kernels-v2 (fixed-2x-mtp0-wide-kernels: MTP-0, A+C dense kernels, six-clear "
                                      "opening switch) plus the top-k-first sampler and the int8 QSA KV cache "
                                      "(server.serving_kernels), with 14 focus games at 64K after the switch (the most "
                                      "that fit; the trial refuses to start on a server that fits fewer); slots beyond "
                                      "the cleared games fill from the backlog in fixed-2x order",
    "fixed-2x-mtp0-wide-kernels": "fixed-2x-mtp0-wide plus the phase-2 dense kernels (server.dense_kernels): the "
                                  "pinned CuTe skinny GEMM for BF16 projections at M <= 16 on SM120, and weight-only "
                                  "INT8 (per-channel, quantized at load) for lm_head, GDN in_proj_qkvz, QSA qkv and "
                                  "the attention output projections; everything else as fixed-2x-mtp0-wide",
    "fixed-2x-mtp3-wide-kernels": "admission route (kernel work, phase 4): the fixed-2x-mtp0-wide scheduling (six focus "
                                  "games at a 64K window, six-clear opening, the trial refuses to start unless six full "
                                  "windows fit) on MTP-3 serving with the phase-2 dense kernels and the phase-4 serving "
                                  "overlay (server.serving_kernels): bf16 GDN recurrent state, NvFp4 MTP draft experts on "
                                  "b12x, plus the fusions that passed the phase-4 microbenchmark",
    "fixed-2x-mtp3-wide-kernels-kv8": "8-bit KV (28 September): fixed-2x-mtp3-wide-kernels with the QSA main KV cache in "
                                      "int8 (per-token-head fp32 scales, server.serving_kernels.kv_cache_8bit) and as many "
                                      "focus games at a 64K window as the served cache fits (KV8_FOCUS_SLOTS; the opening "
                                      "ends at that many distinct level-1 clears); the trial refuses to start unless every "
                                      "focus game's full window fits",
    "fixed-2x-deep-explore": "depth-first scheduling with a wide exploration rotation on the "
                             "fixed-2x-mtp0-wide-kernels serving (MTP-0 plus the phase-2 dense kernels): no opening; five "
                             "exploration slots at a 32K window rotate the games on an 18K-completion-token quantum (never "
                             "played first, then fewest tokens; 60K on level 1 without a clear goes last); the first game to "
                             "clear a level takes the deep slot (128K window) and keeps playing level after level while its "
                             "tokens on the current level stay within max(25K, 2.5 x the median tokens of its cleared "
                             "levels); past that it is demoted to the rotation (its context compacted to 32K) only when "
                             "an exploration clear waits in the promotion queue (most recent clear first); the trial "
                             "refuses to start unless 1 x 128K + 8 x 32K full windows fit the served cache",
    "fixed-2x-mtp3-wide-stack":"fixed-2x-mtp3-wide-kernels plus the phase-5 kernel stack (numerics-preserving): the "
                                "PLE short-conv metadata copies made asynchronous (server.stack_kernels.async_h2d: no "
                                "per-step stream sync before input preparation) and the decode-size Triton tiles "
                                "(server.dense_kernels.mid_m: small BF16 projections and INT8 families at M = 8-32, "
                                "the plan measured by the stack smoke, error-checked at start-up)",
    "fixed-2x-depthwise-mtp3-stack": "fixed-2x-depthwise-mtp3 on the fixed-2x-mtp3-wide-stack serving (the phase-4 "
                                     "serving plus the phase-5 kernel stack); scheduling unchanged",
    "fixed-2x-mtp0-wide-kv8-stack": "fixed-2x-mtp0-wide-kernels-kv8 plus the phase-5 decode-size Triton tiles "
                                    "(server.dense_kernels.mid_m; the MTP-only async_h2d item is also on, a no-op "
                                    "without speculative decoding); scheduling unchanged",
    "fixed-2x-depthwise-mtp3-fixed-order": "fixed-2x-depthwise-mtp3 with the fixed-2x steady-state refill order (user, "
                                           "28 September: after the slots fill, schedule as fixed-2x does after its "
                                           "opening): a clear that landed while parked first, then fewest admissions, "
                                           "then most levels, then roster order, so a parked depth game waits behind "
                                           "every game admitted fewer times instead of coming back ahead of all "
                                           "level-0 games",
    "fixed-2x-mtp0-wide-kv8-hero-21x42k": "the hero arm (fixed-2x-mtp0-wide-kv8-hero) with 21 focus games at a 42,496-token "
                                          "window instead of 14 at 64K (the most streams that fit at the largest window "
                                          "on the served layout); six-clear switch and everything else unchanged",
    "fixed-2x-mtp0-wide-kv8-hero-6x64k": "the hero serving (fixed-2x-mtp0-wide-kv8-hero) with kernels-v2's six focus "
                                         "games at 64K instead of 14: per-stream throughput for depth; everything else "
                                         "the hero's",
    "fixed-2x-mtp0-wide-kv8-hero-prune12": "the hero arm (fixed-2x-mtp0-wide-kv8-hero) with 12.5% of the routed "
                                           "experts (64 per layer, the least used on recorded arena traffic) dropped "
                                           "at load: the router and the MoE are built with 448 experts; the freed 7.9 "
                                           "GiB goes to the KV pool, 21 focus games at 64K (served layout); six-clear "
                                           "switch and everything else unchanged",
    "fixed-2x-mtp0-wide-kv8-hero-prune25": "the hero arm (fixed-2x-mtp0-wide-kv8-hero) with 25% of the routed experts "
                                           "(128 per layer) dropped at load: 384 experts per layer, 15.8 GiB more KV, "
                                           "25 focus games at 64K (29 projected, capped by the 25 sequences); "
                                           "six-clear switch and everything else unchanged",
    "fixed-2x-mtp0-wide-kv8-hero-p6": "the hero-uva arm (fixed-2x-mtp0-wide-kv8-hero-uva: PLE table and input "
                                      "embedding in host memory over UVA) with the phase-6 fused split-K reduction "
                                      "(server.dense_kernels.fused_splitk: the dense Triton GEMMs reduce their split-K "
                                      "partials in the last CTA per tile, bitwise equal to the two-launch path); the "
                                      "phase-5 tile plan and scheduling unchanged",
    "fixed-2x-mtp0-wide-kv8-hero-prefix": "the hero arm (fixed-2x-mtp0-wide-kv8-hero) with the prefix-cache v2 serving "
                                          "patches (cache_experiment.prefix_cache): prefill chunks aligned to the "
                                          "1,600-token Mamba block, so every request leaves its prompt-end state for "
                                          "the next request of the game (the pinned split aligned to the QSA ring's "
                                          "4-token block), the newest boundary state kept with its request until it "
                                          "ends, and superseded states reused first instead of displacing other "
                                          "games' prefixes; scheduling unchanged",
    "fixed-2x-mtp0-wide-kv8-hero-uva": "the hero arm (fixed-2x-mtp0-wide-kv8-hero) with the host-memory serving tweaks "
                                       "(docs/validation-20260927/serving-tweaks.md): the PLE n-gram table read by the "
                                       "GPU from page-locked host memory over UVA (no CPU offload process) and the "
                                       "input embedding in host memory; no draft head change (MTP-0)",
    "fixed-2x-mtp0-wide-kv8-hero": "hero serving (28 September):fixed-2x-mtp0-wide-kernels-kv8 (MTP-0, A+C dense "
                                   "kernels, top-k-first sampler, int8 KV, 14 focus games at 64K, six-clear switch) "
                                   "plus the phase-5 decode-size Triton tiles (fixed measured plan, tuned after KV "
                                   "sizing) and the target's NVFP4 experts on vLLM's Marlin W4A16 kernel; scheduling "
                                   "unchanged",
    "fixed-2x-depthwise-mtp3-gptq-arc": "fixed-2x-depthwise-mtp3 with the routed experts re-quantized on ARC "
                                        "activations: GPTQ onto the same NVFP4 grid from the BF16 source (Hessians "
                                        "from calibration games' MoE inputs; holdout games excluded), served as "
                                        "drop-in replacement shards (same format, size and CUTLASS W4A4 kernels)",
    "fixed-2x-mtp0-wide-kv8-hero-gptq-arc": "the hero arm (Marlin W4A16 experts) with the routed experts re-quantized "
                                            "on ARC activations (GPTQ-ARC v1 overlay, same format and kernels)",
    "fixed-2x-mtp0-wide-kv8-hero-prune12-gptq-arc": "fixed-2x-mtp0-wide-kv8-hero-prune12 with the GPTQ-ARC v1 experts",
    "fixed-2x-mtp0-wide-kv8-hero-prune25-gptq-arc": "fixed-2x-mtp0-wide-kv8-hero-prune25 with the GPTQ-ARC v1 experts",
    "fixed-2x-depthwise-mtp3-gptq-arc-w4a16": "fixed-2x-depthwise-mtp3-gptq-arc served on vLLM's Marlin weight-only "
                                              "kernel (W4A16): the ARC GPTQ experts with BF16 activations",
    "fixed-2x-depthwise-mtp3-w4a16": "fixed-2x-depthwise-mtp3 with the target's NVFP4 experts on vLLM's Marlin "
                                     "weight-only kernel (W4A16: BF16 activations, no activation FP4 rounding) instead "
                                     "of FlashInfer CUTLASS W4A4; same weights and memory",
    "fixed-2x-depthwise-mtp3": "fixed-2x-depthwise scheduling on the fixed-2x-mtp3-wide-kernels serving (MTP-3 with six "
                               "full 64K windows: bf16 SSM state, NvFp4 draft experts, top-k sampler, dense kernels)",
    "fixed-2x-mtp3-wide-tweaks": "fixed-2x-mtp3-wide-kernels plus the serving tweaks ("
                                 "docs/validation-20260927/serving-tweaks.md): the MTP draft head over a fixed "
                                 "16,384-token subset (exact output, draft-only), the PLE n-gram table read by the GPU "
                                 "from page-locked host memory over UVA (no CPU offload round trip), and the input "
                                 "embedding in host memory; the vision tower stays (boards are sent as images)",
    "fixed-2x-mtp3-wide-tweaks-nople": "fixed-2x-mtp3-wide-tweaks without the PLE host read: "
                                       "the 16K draft head and the host-resident input embedding; the PLE stays on "
                                       "the CPU offload process",
    "fixed-2x-depthwise-mtp3-tweaks-nople": "fixed-2x-depthwise scheduling on the fixed-2x-mtp3-wide-tweaks-nople "
                                            "serving",
    "fixed-2x-depthwise-mtp3-tweaks": "fixed-2x-depthwise scheduling on the fixed-2x-mtp3-wide-tweaks serving "
                                      "(fixed-2x-depthwise-mtp3 plus the serving tweaks)",
    "fixed-2x-depthwise": "depth-first slots with fixed-2x budgets on the "
                          "fixed-2x-mtp0-wide-kernels serving (MTP-0 plus the phase-2 dense kernels, six full 64K "
                          "windows required to fit): no opening; level-0 games rotate through any free slot at a 32K "
                          "window on an 18K-completion-token turn (never played first, then fewest tokens); a game "
                          "that clears level 1 keeps its slot at 64K under fixed-2x budgets on the busy clock "
                          "(2 x (k+1) x T1 for level k -> k+1, granted in full at every clear); an exhausted depth "
                          "game parks only if a game waits (the level-0 rotation counts); a freed slot goes to parked "
                          "clears, then parked depth games by fastest level-1 clear (smallest T1), then the level-0 "
                          "rotation; re-admission: the full fixed-2x allowance again",
    "fixed-2x-mtp0-wide": "throughput arm: MTP-0 serving (speculative decoding off; 675 gate blocks, 91 per full 64K request, so seven full windows fit) and six phase-2 focus games at a 64K window; the opening ends at the sixth distinct level-1 clear; the trial refuses to start unless six full windows fit the served cache",
    "fixed-2x-memory": "PRO-LONG persistent memory block: every six actions one tool-less request (2,048-token cap, thinking off) rewrites a structured world-model block kept as the first user message and verbatim across compaction",
}


def apply_variant(study, name):
    if name not in VARIANTS:
        raise ValueError(f"Unknown arena variant: {name}")
    if name in COMPOSED:
        return _composed_variant(study, name)
    if name == "graph-128" and study.get("backend", "vllm") != "vllm":
        raise ValueError("CUDA graph capture comparison requires vLLM")
    if name == "cache-reclaim" and study.get("backend", "vllm") != "vllm":
        raise ValueError("Sparse cache reclamation comparison requires vLLM")
    if name in ('reclaim-mtp1', 'reclaim-mtp4') and (study.get('backend', 'vllm') != 'vllm' or study.get('candidate_profile')):
        raise ValueError('MTP depth comparison requires the Flash vLLM profile')
    if name in ('reclaim-ngram', 'reclaim-ngram8') and (study.get('backend', 'vllm') != 'vllm' or study.get('candidate_profile')):
        raise ValueError('Speculation method comparison requires the Flash vLLM profile')
    if name == 'reclaim-context-64k' and (study.get('backend', 'vllm') != 'vllm' or study.get('candidate_profile')):
        raise ValueError('Reclaimed context comparison requires the Flash vLLM profile')
    if name in ('reclaim-phase2-four', 'reclaim-grant-180s', 'reclaim-grant-2t1', 'reclaim-effort-medium',
                'reclaim-64k-keep3', 'reclaim-regrant-2x', 'reclaim-fixed-2x') and (
            study.get('backend', 'vllm') != 'vllm' or study.get('candidate_profile')):
        raise ValueError('Reclaimed scheduling comparison requires the Flash vLLM profile')
    if name == 'reclaim-graph-128' and (
            study.get('backend', 'vllm') != 'vllm' or study.get('candidate_profile')):
        raise ValueError('Reclaimed graph comparison requires the Flash vLLM profile')
    if (len(study["games"]) + len(study.get("games_excluded", [])) != 25 or study["wall_seconds"] != 1800
            or [t["kind"] for t in study["trials"]] not in (["duck-transfer", "pi"], ["pi-rlm"])):
        raise ValueError("Variants require both Duck/Pi (or the Pi RLM trial), 25 games and 1800 seconds each")
    if (study["active_limit"] != 2 or study["windows"] != [131072]
            or study.get("max_level_grant_seconds") is not None
            or study.get("max_level_grant_t1_multiple") is not None
            or study.get("regrant_multiple") is not None
            or "arena_variant" in study):
        raise ValueError("Variants must start from the unmodified paired control")
    result = copy.deepcopy(study)
    if name == "phase2-four":
        result["active_limit"] = 4
        changed = {"active_limit": {"before": 2, "after": 4}}
    elif name == "context-64k":
        result["windows"] = [65536]
        if "cache_experiment" in result:
            result["cache_experiment"]["compaction_changed"] = True
        changed = {"windows": {"before": [131072], "after": [65536]}}
    elif name == "grant-180s":
        result["max_level_grant_seconds"] = 180
        result["level_budget_rule"] += "; cap each steady-state allocation at 180 active seconds"
        changed = {"max_level_grant_seconds": {"before": None, "after": 180}}
    elif name == "graph-128":
        before = result["server"].get("max_cudagraph_capture_size", 32)
        if before != 32:
            raise ValueError("Graph comparison requires the historical 32-token capture control")
        result["server"]["max_cudagraph_capture_size"] = 128
        changed = {"server.max_cudagraph_capture_size": {"before": before, "after": 128}}
    elif name in ("cache-reclaim", "reclaim-mtp1", "reclaim-context-64k",
                  "reclaim-phase2-four", "reclaim-grant-180s", "reclaim-graph-128",
                  "reclaim-grant-2t1", "reclaim-effort-medium", "reclaim-64k-keep3", "reclaim-regrant-2x",
                  "reclaim-fixed-2x", "reclaim-mtp4", "reclaim-ngram", "reclaim-ngram8"):
        if result.get('cache_experiment', {}).get('reclaim_skipped_blocks', False) is not False:
            raise ValueError('Cache reclamation comparison requires an unpatched control')
        result.setdefault('cache_experiment', {})['reclaim_skipped_blocks'] = True
        changed = {'cache_experiment.reclaim_skipped_blocks': {'before': False, 'after': True}}
        if name in ('reclaim-mtp1', 'reclaim-mtp4'):
            if result['server'].get('mtp_tokens') != 3:
                raise ValueError('MTP depth comparison requires the MTP-3 control')
            depth = 1 if name == 'reclaim-mtp1' else 4
            result['server']['mtp_tokens'] = depth
            changed['server.mtp_tokens'] = {'before': 3, 'after': depth}
        elif name in ('reclaim-ngram', 'reclaim-ngram8'):
            # MTP off (the pinned script then emits no --speculative-config) and the arena's
            # serving hook appends the n-gram config; with the 3-token draft the lookahead
            # matches the MTP-3 control, so cache admission is unchanged (ngram8 adds five).
            if result['server'].get('mtp_tokens') != 3 or result['server'].get('speculative') is not None:
                raise ValueError('Speculation method comparison requires the MTP-3 control')
            spec = NGRAM8_SPECULATION if name == 'reclaim-ngram8' else NGRAM_SPECULATION
            result['server']['mtp_tokens'] = 0
            result['server']['speculative'] = dict(spec)
            changed['server.mtp_tokens'] = {'before': 3, 'after': 0}
            changed['server.speculative'] = {'before': None, 'after': dict(spec)}
        elif name == 'reclaim-context-64k':
            result['windows'] = [65536]
            result['cache_experiment']['compaction_changed'] = True
            changed['windows'] = {'before': [131072], 'after': [65536]}
        elif name == 'reclaim-phase2-four':
            result['active_limit'] = 4
            changed['active_limit'] = {'before': 2, 'after': 4}
        elif name == 'reclaim-grant-180s':
            result['max_level_grant_seconds'] = 180
            result['level_budget_rule'] += '; cap each steady-state allocation at 180 active seconds'
            changed['max_level_grant_seconds'] = {'before': None, 'after': 180}
            changed['level_budget_rule'] = {
                'before': study['level_budget_rule'], 'after': result['level_budget_rule']}
        elif name == 'reclaim-grant-2t1':
            # Renewable allocations: at most 2*T1 (inference seconds) per admission of a game
            # that has cleared a level; exhausted games rotate through the backlog. Tests the
            # user's scheduling-timing question without the fixed 180 s cap's T1 blindness.
            result['max_level_grant_t1_multiple'] = 2
            result['level_budget_rule'] += '; cap each steady-state allocation at 2*T1 (renewable)'
            changed['max_level_grant_t1_multiple'] = {'before': None, 'after': 2}
            changed['level_budget_rule'] = {
                'before': study['level_budget_rule'], 'after': result['level_budget_rule']}
        elif name == 'reclaim-regrant-2x':
            # User (24 September): re-admission grants of 2x the seconds already spent instead
            # of 1.5x. Under the inference clock the control's level-0 regrants were only
            # 29-76 s (one to three xhigh replies); the floor (+30 s) and the (k+1)*T1 level
            # budgets are unchanged.
            result['regrant_multiple'] = 2
            result['level_budget_rule'] = result['level_budget_rule'].replace(
                'regrant max(1.5*spent, spent+30)', 'regrant max(2*spent, spent+30)')
            if result['level_budget_rule'] == study['level_budget_rule']:
                raise ValueError('Regrant comparison requires the max(1.5*spent, spent+30) control rule')
            changed['regrant_multiple'] = {'before': None, 'after': 2}
            changed['level_budget_rule'] = {
                'before': study['level_budget_rule'], 'after': result['level_budget_rule']}
        elif name == 'reclaim-fixed-2x':
            # evening: budgets 2*L*T1 fixed per admission, the
            # level-0 first-park spend for games without a clear, backlog by fewest
            # admissions then most levels, parked clears always first, and a budget clock
            # that charges tool execution as well as admitted inference.
            result['budget_clock'] = 'busy'
            result['level_budget_policy'] = 'fixed-2x'
            result['backlog_policy'] = 'readmissions'
            result['prioritize_parked_progress'] = True
            for key, after in (('budget_clock', 'busy'), ('level_budget_policy', 'fixed-2x'),
                               ('backlog_policy', 'readmissions'), ('prioritize_parked_progress', True)):
                changed[key] = {'before': study.get(key), 'after': after}
        elif name == 'reclaim-64k-keep3':
            # Compaction retention at the window where compaction actually fires: the last
            # three observations and their tool rounds survive each summary verbatim.
            result['windows'] = [65536]
            result['cache_experiment']['compaction_changed'] = True
            result['compaction_keep_observations'] = 3
            changed['windows'] = {'before': [131072], 'after': [65536]}
            changed['cache_experiment.compaction_changed'] = {'before': False, 'after': True}
            changed['compaction_keep_observations'] = {'before': None, 'after': 3}
        elif name == 'reclaim-effort-medium':
            # The Flash template accepts only xhigh, medium and low: `arc-arena-duck-30m-
            # inference-effort-high-v1` (24 September) failed every game's first /tokenize
            # call with "Unexpected reasoning effort high". Validate against the served
            # model's template before a bundle is built.
            if result.get('reasoning_effort') != 'xhigh':
                raise ValueError('Reasoning-effort comparison requires the xhigh control')
            served = result.get('served_model') or 'Qwen/Qwen3.8-Flash-Next-NVFP4'
            if 'medium' not in allowed_efforts(served):
                raise ValueError(f'{served} template does not accept reasoning effort medium')
            result['reasoning_effort'] = 'medium'
            changed['reasoning_effort'] = {'before': 'xhigh', 'after': 'medium'}
        elif name == 'reclaim-graph-128':
            before = result['server'].get('max_cudagraph_capture_size', 32)
            if before != 32:
                raise ValueError('Graph comparison requires the historical 32-token capture control')
            result['server']['max_cudagraph_capture_size'] = 128
            changed['server.max_cudagraph_capture_size'] = {'before': before, 'after': 128}
    else:
        changed = {}
    result["arena_variant"] = {"name": name, "changed": changed,
                               "checkpoint_and_serving_unchanged": name not in (
                                   "graph-128", "cache-reclaim", "reclaim-mtp1", "reclaim-mtp4", "reclaim-ngram", "reclaim-ngram8",
                                   "reclaim-context-64k", "reclaim-phase2-four", "reclaim-grant-180s",
                                   "reclaim-graph-128")}
    if name in ('reclaim-mtp1', 'reclaim-mtp4'):
        result['arena_variant']['reference_variant'] = 'cache-reclaim'
        result['arena_variant']['reference_difference'] = {'server.mtp_tokens': changed['server.mtp_tokens']}
    elif name in ('reclaim-ngram', 'reclaim-ngram8'):
        result['arena_variant']['reference_variant'] = 'cache-reclaim'
        result['arena_variant']['reference_difference'] = {
            'server.mtp_tokens': changed['server.mtp_tokens'],
            'server.speculative': changed['server.speculative']}
        result['arena_variant']['reference_checkpoint_and_serving_unchanged'] = False
    elif name == 'reclaim-context-64k':
        result['arena_variant']['reference_variant'] = 'cache-reclaim'
        result['arena_variant']['reference_difference'] = {
            'windows': {'before': [131072], 'after': [65536]},
            'cache_experiment.compaction_changed': {'before': False, 'after': True},
        }
        result['arena_variant']['reference_checkpoint_and_serving_unchanged'] = True
    elif name in ('reclaim-phase2-four', 'reclaim-grant-180s', 'reclaim-grant-2t1', 'reclaim-effort-medium',
                  'reclaim-64k-keep3', 'reclaim-regrant-2x', 'reclaim-fixed-2x'):
        result['arena_variant']['reference_variant'] = 'cache-reclaim'
        result['arena_variant']['reference_difference'] = {
            k: v for k, v in changed.items() if k != 'cache_experiment.reclaim_skipped_blocks'}
        result['arena_variant']['reference_checkpoint_and_serving_unchanged'] = True
    elif name == 'reclaim-graph-128':
        result['arena_variant']['reference_variant'] = 'cache-reclaim'
        result['arena_variant']['reference_difference'] = {
            'server.max_cudagraph_capture_size': {'before': 32, 'after': 128}}
        result['arena_variant']['reference_checkpoint_and_serving_unchanged'] = False
    return result


def _depthwise_variant(study, serving, name=None):
    """``fixed-2x-depthwise`` on the serving of the composed variant ``serving``: that variant's study (server block,
    window, slot count and the full-window fit requirement) with the depthwise scheduler in place of its level-budget
    policy. The source must serve ``depthwise.slots`` full ``deep_window`` requests (``require_focus_windows_fit``)
    on the fixed-2x readmissions base with hybrid-block admission. Returns the study and its reference difference."""
    from .depthwise import DEFAULTS as DEPTHWISE

    source = _composed_variant(study, serving)
    if (source["windows"] != [DEPTHWISE["deep_window"]] or source["active_limit"] != DEPTHWISE["slots"]
            or source.get("require_focus_windows_fit") is not True or source.get("level_budget_policy") != "fixed-2x"
            or source.get("backlog_policy") != "readmissions" or source.get("cache_admission") != "hybrid_blocks"
            or source.get("budget_clock") != "busy"):
        raise ValueError(f"The depthwise arm needs a serving source with {DEPTHWISE['slots']} fixed-2x slots at "
                         f"{DEPTHWISE['deep_window']} tokens, the full-window fit check and hybrid-block admission; "
                         f"{serving} is not one")
    result = copy.deepcopy(source)
    result["level_budget_policy"] = "depthwise"
    # User (28 September): a parked depth game re-enters with the full fixed-2x allowance (the fixed-2x code's
    # re-admission), not the rest of its level budget; the level-0 rotation may stop once depth games fill the slots.
    result["depthwise"] = {**DEPTHWISE, "regrant": "full"}
    if name in DEPTHWISE_FIXED_ORDER:
        result["depthwise"]["backlog"] = "fixed-2x"
    result["depthwise_serving_source"] = serving
    changed = {**source["arena_variant"]["reference_difference"],
               "level_budget_policy": {"before": "fixed-2x", "after": "depthwise"},
               "depthwise": {"before": None, "after": dict(result["depthwise"])}}
    return result, changed


def _rank_variant(study, serving):
    """The continuous ranking (``arena.rank``) on the serving of the composed variant ``serving``: that variant's study
    with ``level_budget_policy="rank"`` and ``rank`` settings whose slots and window are the source's focus slots and
    window. The source must be a fixed-2x readmissions study with one window, hybrid-block admission and the
    full-window fit check. Returns the study and its reference difference."""
    from .rank import DEFAULTS as RANK

    source = _composed_variant(study, serving)
    if (source.get("level_budget_policy") != "fixed-2x" or source.get("backlog_policy") != "readmissions"
            or source.get("cache_admission") != "hybrid_blocks" or source.get("require_focus_windows_fit") is not True
            or len(source["windows"]) != 1 or not isinstance(source.get("active_limit"), int)):
        raise ValueError(f"The rank arm needs a fixed-2x serving source with one window, a fixed focus-slot count, the "
                         f"full-window fit check and hybrid-block admission; {serving} is not one")
    result = copy.deepcopy(source)
    result["level_budget_policy"] = "rank"
    result["rank"] = {**RANK, "slots": source["active_limit"], "window": source["windows"][0]}
    result["rank_serving_source"] = serving
    changed = {**source["arena_variant"]["reference_difference"],
               "level_budget_policy": {"before": "fixed-2x", "after": "rank"},
               "rank": {"before": None, "after": dict(result["rank"])}}
    return result, changed


def _expert_quant_variant(study, name):
    """The source variant's study (``EXPERT_QUANT_SOURCE``) with one server change: the ARC GPTQ expert overlay, or
    the Marlin W4A16 MoE backend. Returns the study and its difference from the fixed-2x reference."""
    source_name = EXPERT_QUANT_SOURCE[name]
    source = _composed_variant(study, source_name)
    result = copy.deepcopy(source)
    changed = dict(source["arena_variant"]["reference_difference"])
    if "-gptq-arc" in name:
        if not EXPERT_OVERLAY_GPTQ_ARC["manifest_sha256"] or "expert_overlay" in result["server"]:
            raise ValueError("The ARC GPTQ expert overlay is not pinned yet (arena.variants.EXPERT_OVERLAY_GPTQ_ARC)")
        result["server"]["expert_overlay"] = dict(EXPERT_OVERLAY_GPTQ_ARC)
        changed["server.expert_overlay"] = {"before": None, "after": dict(EXPERT_OVERLAY_GPTQ_ARC)}
    if name.endswith("-w4a16"):
        kernels = copy.deepcopy(result["server"].get("serving_kernels"))
        if not kernels or kernels.get("moe_backend") is not None:
            raise ValueError("The W4A16 arm needs a serving-kernels source on the default CUTLASS MoE backend")
        kernels["moe_backend"] = "marlin"
        result["server"]["serving_kernels"] = kernels
        changed["server.serving_kernels"] = {"before": changed.get("server.serving_kernels", {}).get("after"),
                                             "after": copy.deepcopy(kernels)}
    result["expert_quant_serving_source"] = source_name
    return result, changed


def _composed_variant(study, name):
    from arc_infra.model_templates import allowed_efforts

    base = apply_variant(study, "reclaim-fixed-2x")
    result = copy.deepcopy(base)
    changed = {}
    if name == "fixed-2x-64k-four":
        result["windows"] = [65536]
        result["cache_experiment"]["compaction_changed"] = True
        result["compaction_keep_observations"] = 3
        result["active_limit"] = 4
        changed = {"windows": {"before": [131072], "after": [65536]},
                   "cache_experiment.compaction_changed": {"before": False, "after": True},
                   "compaction_keep_observations": {"before": None, "after": 3},
                   "active_limit": {"before": 2, "after": 4}}
    elif name == "fixed-2x-t1cap-120":
        # Scheduling-timing variant (25 September): in five fixed-2x runs the second phase-2 slot's
        # audition winners had T1 of 108-213 busy seconds (their whole pre-clear spend) and so earned
        # 500-850 s for level 2, mostly without clearing it; the opening focus games had T1 27-94 s.
        result["t1_cap_seconds"] = 120
        changed = {"t1_cap_seconds": {"before": None, "after": 120}}
    elif name in ("fixed-2x-notes", "fixed-2x-notes-live"):
        # Harness addition (25 September, ideas 4.1): general knowledge carried between games.
        # notes v1/v2: the notebook reached only re-admitted level-0 games (the focus games never
        # re-admit); ``live`` refreshes the injected message at the next clear or compaction too.
        result["cross_game_notes"] = {"lines_per_request": 10, "request_cap_tokens": 1024,
                                      "inject_cap_tokens": 2000}
        if name == "fixed-2x-notes-live":
            result["cross_game_notes"]["refresh"] = "live"
        changed = {"cross_game_notes": {"before": None, "after": dict(result["cross_game_notes"])}}
    elif name == "fixed-2x-memory":
        # Harness addition (25 September, ideas 4.2): PRO-LONG's persistent memory block.
        result["memory_block"] = {"every_actions": 6, "request_cap_tokens": 2048}
        changed = {"memory_block": {"before": None, "after": dict(result["memory_block"])}}
    elif name == "fixed-2x-switch-3":
        if result.get("opening_switch_level1_games") != 2:
            raise ValueError("Opening-trigger comparison requires the two-clear control")
        result["opening_switch_level1_games"] = 3
        changed = {"opening_switch_level1_games": {"before": 2, "after": 3}}
    elif name == "fixed-2x-pool":
        # User decisions (25 September): the second slot's serial level-0 auditions (42-70% of phase 2,
        # 6-9 of ~20 tied games reached, in roster order) become one parallel pool; the normalized clock
        # keeps T1 and budgets contention-invariant, so the 120 s T1 cap is not needed.
        if result.get("budget_clock") != "busy" or result.get("t1_cap_seconds") is not None:
            raise ValueError("Pool comparison requires the busy-clock fixed-2x control without a T1 cap")
        result["budget_clock"] = "normalized"
        result["cost_normalization"] = {"rate_tokens_per_second": 104.0, "context_weight": 0.25,
                                        "context_reference": 131072, "request_overhead_seconds": 1.65}
        result["audition_pool"] = 4
        result["pool_stall_actions"] = 12
        result["pool_action_limit"] = 60
        result["level_budget_rule"] += ("; normalized clock: 1.65 s + completion tokens / 104 tok/s * "
                                        "(1 + 0.25 * prompt / 128K) + tool seconds; level-0 backlog games "
                                        "audition in parallel pools of four per free slot")
        changed = {"budget_clock": {"before": "busy", "after": "normalized"},
                   "cost_normalization": {"before": None, "after": result["cost_normalization"]},
                   "audition_pool": {"before": None, "after": 4},
                   "pool_stall_actions": {"before": None, "after": 12},
                   "pool_action_limit": {"before": None, "after": 60}}
    elif name in ("fixed-2x-pool2-cap", "fixed-2x-pool2-timed-novelty"):
        # 25 September afternoon: pool v2 opened its first pool only at 862 s and stall-open360's two samples
        # (8.36%, 3.61%) differed only in which games held the focus slots at the 360 s switch. Combined: the cap
        # frees a slot early, a pool of two auditions the tied level-0 backlog in parallel (stalled auditions
        # rotate out after 12 no-novel actions), and a stalled focus game gives its slot back.
        if result.get("budget_clock") != "busy" or result.get("t1_cap_seconds") is not None:
            raise ValueError("Pool comparison requires the busy-clock fixed-2x control without a T1 cap")
        result["budget_clock"] = "normalized"
        result["cost_normalization"] = {"rate_tokens_per_second": 104.0, "context_weight": 0.25,
                                        "context_reference": 131072, "request_overhead_seconds": 1.65}
        result["audition_pool"] = 2
        result["pool_stall_actions"] = 12
        result["pool_action_limit"] = 60
        result["focus_stall_actions"] = 15
        result["opening_max_wall_seconds"] = 360
        result["level_budget_rule"] += ("; normalized clock: 1.65 s + completion tokens / 104 tok/s * "
                                        "(1 + 0.25 * prompt / 128K) + tool seconds; level-0 backlog games "
                                        "audition in parallel pools of two per free slot; opening capped at 360 s; "
                                        "focus games park after 15 actions without a novel board")
        changed = {"budget_clock": {"before": "busy", "after": "normalized"},
                   "cost_normalization": {"before": None, "after": result["cost_normalization"]},
                   "audition_pool": {"before": None, "after": 2},
                   "pool_stall_actions": {"before": None, "after": 12},
                   "pool_action_limit": {"before": None, "after": 60},
                   "focus_stall_actions": {"before": None, "after": 15},
                   "opening_max_wall_seconds": {"before": None, "after": 360}}
        if name == "fixed-2x-pool2-timed-novelty":
            # stall-open360 v2: the slot the cap freed walked the level-0 roster in order (tn36, lf52, cn04,
            # bp35, wa30, ~90 s each) and reached lp85, which then cleared five levels, only at 829 s.
            # pool2-cap v1 (0.25%): auditions bounded by actions alone held pool places for 500+ s at xhigh
            # (cn04 539 s without a clear, stall or the 60-action limit); only five games auditioned.
            result["level0_order"] = "novelty"
            result["pool_time_limit_seconds"] = 120
            result["level_budget_rule"] += ("; level-0 ties ordered by distinct boards shown, most first; "
                                            "auditions park after 120 normalized seconds")
            changed["level0_order"] = {"before": "roster", "after": "novelty"}
            changed["pool_time_limit_seconds"] = {"before": None, "after": 120}
    elif name == "fixed-2x-progress":
        # ideas.md point 2 (25 September): "rank games based on learning progress computed from levels
        # solved, tokens used, actions taken; always choose games with maximal learning progress".
        # Replaces the readmissions order (fewest admissions, most levels, roster) with a value estimate.
        if result.get("backlog_policy") != "readmissions":
            raise ValueError("Progress-order comparison requires the readmissions-order fixed-2x control")
        result["backlog_policy"] = "progress"
        result["level_budget_rule"] += ("; backlog by estimated marginal score per busy second: "
                                        "(levels+1)/(N(N+1)/2) * (levels+0.5)/(busy+90) / (1+stalled/10)")
        changed = {"backlog_policy": {"before": "readmissions", "after": "progress"}}
    elif name == "fixed-2x-progress-novelty":
        if result.get("backlog_policy") != "readmissions":
            raise ValueError("Progress-order comparison requires the readmissions-order fixed-2x control")
        result["backlog_policy"] = "progress"
        result["progress_model"] = "novelty"
        result["level_budget_rule"] += ("; backlog by estimated clears per busy second: "
                                        "(levels+0.5+distinct/(actions+3))/(busy+90) / (1+stalled/10), no level-share weight")
        changed = {"backlog_policy": {"before": "readmissions", "after": "progress"},
                   "progress_model": {"before": None, "after": "novelty"}}
    elif name in ("fixed-2x-reopen-pause", "fixed-2x-reopen-run"):
        if result.get("backlog_policy") != "readmissions":
            raise ValueError("The reopened phase 1 requires the readmissions-order fixed-2x control")
        mode = name.rsplit("-", 1)[1]
        result["reopen_phase1"] = mode
        result["level_budget_rule"] += (f"; a freed slot reopens phase 1 for the top tie group until a clear "
                                        f"(surviving focus game: {mode})")
        changed = {"reopen_phase1": {"before": None, "after": mode}}
    elif name == "fixed-2x-snapshot":
        # Extension from public harnesses (25 September, ideas 5): Meta-Harness's one clean
        # harness win on a weak model was a fixed environment bootstrap handed over as context.
        result["game_snapshot"] = {"probe": True, "objects": 20}
        changed = {"game_snapshot": {"before": None, "after": {"probe": True, "objects": 20}}}
    elif name == "fixed-2x-level-probe":
        # User (27 September): "it would be worth starting every level for every game with this deterministic
        # trying of actions" -- the game-start probe repeated after every level clear, with three clicks.
        if result.get("level_budget_policy") != "fixed-2x" or result.get("game_snapshot"):
            raise ValueError("Level-probe comparison requires the fixed-2x control without a game snapshot")
        settings = {"probe": True, "objects": 20, "every_level": True, "clicks": 3}
        result["game_snapshot"] = dict(settings)
        changed = {"game_snapshot": {"before": None, "after": dict(settings)}}
    elif name == "fixed-2x-escalate":
        # Extension from public harnesses (25 September, ideas 5): the heavy executable-model
        # mechanisms never engaged as a default loop with local models (harness review, 17
        # September); here they replace the stuck notice once ordinary probing has failed twice.
        result["model_escalation"] = {"after_notices": 2}
        changed = {"model_escalation": {"before": None, "after": {"after_notices": 2}}}
    elif name == "fixed-2x-stall-open360":
        # sb26 held a focus slot in four of six fixed-2x runs (level 1 in 27-47 s, then nothing) for its
        # whole 110-190 s budget; the opening itself ran 247-581 s at five streams before phase 2.
        result["focus_stall_actions"] = 15
        result["opening_max_wall_seconds"] = 360
        changed = {"focus_stall_actions": {"before": None, "after": 15},
                   "opening_max_wall_seconds": {"before": None, "after": 360}}
    elif name == "fixed-2x-80k-three":
        # Three focus streams: the card decodes 290-320 tok/s at three or four streams against ~230 at two,
        # and 3 * 80K fits the 292K pool; completed games peaked at 101-105K prompt tokens, so the
        # smaller window costs some compactions (11 at 64K, 8.8% of inference).
        result["windows"] = [81920]
        result["cache_experiment"]["compaction_changed"] = True
        result["compaction_keep_observations"] = 3
        result["active_limit"] = 3
        changed = {"windows": {"before": [131072], "after": [81920]},
                   "cache_experiment.compaction_changed": {"before": False, "after": True},
                   "compaction_keep_observations": {"before": None, "after": 3},
                   "active_limit": {"before": 2, "after": 3}}
    elif name.startswith("fixed-2x-slots-"):
        # ideas.md (26 September): 1, 2, 4, 8 focus slots with the window fitted to the cache
        # (slots * window <= 0.9 * 291,900 tokens), and phase 2 starts only once every slot can be
        # given a game that has cleared level 1 (the switch fires at the N-th distinct clear).
        if result.get("opening_switch_level1_games") != 2 or result["active_limit"] != 2:
            raise ValueError("Slot comparison requires the two-slot, two-clear control")
        slots = int(name.rsplit("-", 1)[1])
        window = SLOT_WINDOWS[slots]
        result["active_limit"] = slots
        result["opening_switch_level1_games"] = slots
        changed = {"active_limit": {"before": 2, "after": slots},
                   "opening_switch_level1_games": {"before": 2, "after": slots}}
        if window != 131072:
            result["windows"] = [window]
            result["cache_experiment"]["compaction_changed"] = True
            changed["windows"] = {"before": [131072], "after": [window]}
            changed["cache_experiment.compaction_changed"] = {"before": False, "after": True}
    elif name == "fixed-2x-mtp0-wide":
        # Throughput arm (27 September): on MTP-3 a request pins 22 of 199 usable blocks, so the base runs two
        # focus games at ~200-300 tok/s. On MTP-0 the lab fitted 7 x 64K at the 0.9 margin, and the level bench
        # decoded 458 tok/s at 16 streams through this block gate with no preemptions. The gate reads the MTP-0
        # server's own layout (lookahead 0); ``require_focus_windows_fit`` stops the trial before gameplay
        # unless the six focus games' full windows fit that layout.
        if (result["server"].get("mtp_tokens") != 3 or result["server"].get("speculative") is not None
                or result.get("opening_switch_level1_games") != 2 or result["active_limit"] != 2
                or result["windows"] != [131072]):
            raise ValueError("The MTP-0 wide arm requires the MTP-3, two-slot, two-clear, 128K control")
        slots, window = MTP0_WIDE["active_limit"], MTP0_WIDE["window"]
        result["server"]["mtp_tokens"] = MTP0_WIDE["mtp_tokens"]
        result["active_limit"] = slots
        result["opening_switch_level1_games"] = slots
        result["windows"] = [window]
        result["cache_experiment"]["compaction_changed"] = True
        result["require_focus_windows_fit"] = True
        changed = {"server.mtp_tokens": {"before": 3, "after": MTP0_WIDE["mtp_tokens"]},
                   "active_limit": {"before": 2, "after": slots},
                   "opening_switch_level1_games": {"before": 2, "after": slots},
                   "windows": {"before": [131072], "after": [window]},
                   "cache_experiment.compaction_changed": {"before": False, "after": True},
                   "require_focus_windows_fit": {"before": None, "after": True}}
    elif name in ("fixed-2x-budget-1x", "fixed-2x-budget-4x"):
        # ideas.md (26 September): timing changes to the 2*L*T1 level budgets.
        multiple = {"fixed-2x-budget-1x": 1, "fixed-2x-budget-4x": 4}[name]
        result["level_budget_multiple"] = multiple
        changed = {"level_budget_multiple": {"before": 2, "after": multiple}}
    elif name == "fixed-2x-llm-rank":
        # ideas.md point 4 (26 September): "LLM based, e.g. every 1-5 minutes check evidence and select
        # game ranking". The paper notes found no support for LLM-judged scheduling; this tests it.
        if result.get("backlog_policy") != "readmissions":
            raise ValueError("LLM ranking comparison requires the readmissions-order fixed-2x control")
        result["backlog_policy"] = "llm"
        result["llm_backlog"] = {"interval_seconds": 180, "cap_tokens": 768, "thinking": False}
        changed = {"backlog_policy": {"before": "readmissions", "after": "llm"},
                   "llm_backlog": {"before": None, "after": dict(result["llm_backlog"])}}
    elif name == "fixed-2x-procedure":
        # Paper notes 2.6 (Self-Harness rules, HyperAgents checklists, AIRA2 execution check, collapse
        # detector), 26 September: one surface changed, the stuck notice's text.
        result["stuck_procedure"] = {"collapse_window": 30, "collapse_share": 0.8}
        changed = {"stuck_procedure": {"before": None, "after": dict(result["stuck_procedure"])}}
    elif name == "fixed-2x-thompson":
        # Paper notes 2.4 item 1 (SPaCe), 26 September: in the one-hour base runs the late deep games
        # (cn04, tu93, vc33, ka59) got their first clear on a re-admission, and level-0 re-admission
        # follows roster order; posterior sampling spreads it and demotes games that stall.
        if result.get("backlog_policy") != "readmissions":
            raise ValueError("Thompson comparison requires the readmissions-order fixed-2x control")
        result["backlog_policy"] = "thompson"
        result["thompson_seed"] = 0
        changed = {"backlog_policy": {"before": "readmissions", "after": "thompson"},
                   "thompson_seed": {"before": None, "after": 0}}
    elif name == "fixed-2x-coverage":
        # Paper notes 2.6 item 9 (AutoEnv: agents inferred rules but could not search), 26 September:
        # 17 of 23 games never clear level 1 in the one-hour base runs; the helper rides on the
        # harness-directory mechanism (one skill file and one prompt line).
        result["duck_harness"] = "src/arena/harnesses/exploration"
        changed = {"duck_harness": {"before": None, "after": "src/arena/harnesses/exploration"}}
    elif name == "fixed-2x-play":
        # 26 September analysis: ~850 completion tokens per action and 1.7-3.2 actions per call in the focus
        # games; the prompt only asks for "a reliable short sequence". One harness surface: a skill + one line.
        result["duck_harness"] = "src/arena/harnesses/play"
        changed = {"duck_harness": {"before": None, "after": "src/arena/harnesses/play"}}
    elif name == "fixed-2x-harness-v2":
        # docs/validation-20260926/failure-signatures.md §4: five single-surface edits, all on; each has its own
        # flag in ``harness_v2`` so that ablations switch one off. The scheduler is the fixed-2x base's.
        from .harness_v2 import DEFAULTS

        result["duck_harness"] = "src/arena/harnesses/v2"
        result["harness_v2"] = dict(DEFAULTS)
        changed = {"duck_harness": {"before": None, "after": "src/arena/harnesses/v2"},
                   "harness_v2": {"before": None, "after": dict(DEFAULTS)}}
    elif name == "fixed-2x-critical":
        result["critical_errors"] = True
        changed = {"critical_errors": {"before": None, "after": True}}
    elif name in ("fixed-2x-index", "fixed-2x-index-lane"):
        # Budget audit (26 September, user: deep on promising games, open the search when stuck):
        # level-1 attempts get 5-12K tokens while clears continue at 0.19-0.25 per 10K up to 40K,
        # and deep levels hold slots past 80K tokens where the rate is 0.04 per 10K.
        if result.get("level_budget_policy") != "fixed-2x" or result.get("backlog_policy") != "readmissions":
            raise ValueError("Index comparison requires the fixed-2x control")
        from .hazard_index import DEFAULTS

        result["level_budget_policy"] = "index"
        result["index_scheduler"] = dict(DEFAULTS)
        changed = {"level_budget_policy": {"before": "fixed-2x", "after": "index"},
                   "index_scheduler": {"before": None, "after": dict(DEFAULTS)}}
        if name == "fixed-2x-index-lane":
            # Index-v2 (26 September): sb26 kept its slot and cleared 8/8, but lf52, cn04, tu93 and re86
            # sat at level 0 on 4-14K tokens all hour while phase 2 ran two streams on 45% of the cache
            # (a 4-stream decode costs little more than two). Leftover capacity serves the waiting games.
            from .gate import LANE_DEFAULTS

            result["background_lane"] = dict(LANE_DEFAULTS)
            changed["background_lane"] = {"before": None, "after": dict(LANE_DEFAULTS)}
    elif name == "fixed-2x-no-think":
        # User (27 September): a simple token-saving test on the hidden-set base. Reasoning is ~850 completion
        # tokens per action; effort low/medium lost clears at 30 minutes, but no run has turned it off.
        if result.get("level_budget_policy") != "fixed-2x" or result.get("enable_thinking", True) is not True:
            raise ValueError("No-think comparison requires the fixed-2x control with thinking on")
        result["enable_thinking"] = False
        changed = {"enable_thinking": {"before": True, "after": False}}
    elif name == "fixed-2x-adaptive-think":
        # User (27 September): more actions per request. 55-60% of gameplay requests execute no action and
        # hold 50-60% of the tokens; medium effort gave 510 tokens per action against ~850 in the probes.
        if result.get("level_budget_policy") != "fixed-2x" or result.get("adaptive_thinking"):
            raise ValueError("Adaptive-thinking comparison requires the fixed-2x control")
        from .adaptive import DEFAULTS

        result["adaptive_thinking"] = dict(DEFAULTS)
        changed = {"adaptive_thinking": {"before": None, "after": dict(DEFAULTS)}}
    elif name == "fixed-2x-worldmodel":
        # User (27 September, "we can try the idea regarding deep levels"): deep levels cost as much as level 1.
        if result.get("level_budget_policy") != "fixed-2x" or result.get("duck_harness"):
            raise ValueError("World-model comparison requires the fixed-2x control without a harness directory")
        result["duck_harness"] = "src/arena/harnesses/worldmodel"
        result["worldmodel_notice"] = True
        changed = {"duck_harness": {"before": None, "after": "src/arena/harnesses/worldmodel"},
                   "worldmodel_notice": {"before": None, "after": True}}
    elif name == "fixed-2x-gittins-broad":
        # Trace-driven simulator (27 September, docs/validation-20260927/schedule-sim.md): the index and commit
        # policies reach 4-5% against ~8.6% for a scheduler that knows each level's clear rate; at the two-clear
        # switch 21 games hold ~5K tokens (warm-up band). Broad opening + this index: 6.25 +- 0.35 vs 4.16.
        if result.get("level_budget_policy") != "fixed-2x" or result.get("backlog_policy") != "readmissions":
            raise ValueError("Gittins comparison requires the fixed-2x control")
        from .gittins import BROAD

        result["level_budget_policy"] = "index"
        result["index_scheduler"] = dict(BROAD)  # frozen: the recorded gittins-broad runs
        result["opening_min_tokens"] = 15000
        result["opening_max_wall_seconds"] = 1500
        changed = {"level_budget_policy": {"before": "fixed-2x", "after": "index"},
                   "index_scheduler": {"before": None, "after": dict(BROAD)},
                   "opening_min_tokens": {"before": None, "after": 15000},
                   "opening_max_wall_seconds": {"before": None, "after": 1500}}
    elif name == "fixed-2x-mtp0-wide-kernels":
        # Kernel work, phase 2 ("go ahead with A+C"): the only difference from the recorded
        # c110 mtp0-wide run is the dense-GEMM kernels (docs/validation-20260927/kernels-phase2.md).
        from arc_infra.dense_kernels_patch import PHASE2

        wide = _composed_variant(study, "fixed-2x-mtp0-wide")
        if "dense_kernels" in wide["server"]:
            raise ValueError("The kernel arm requires the unpatched mtp0-wide serving")
        result = copy.deepcopy(wide)
        result["server"]["dense_kernels"] = copy.deepcopy(PHASE2)
        changed = {**wide["arena_variant"]["reference_difference"],
                   "server.dense_kernels": {"before": None, "after": copy.deepcopy(PHASE2)}}
    elif name == "fixed-2x-mtp3-wide-kernels":
        # Kernel work, phase 4 ("6 games of 64K with drafting back on" + "the fusions you
        # mention"): mtp0-wide's scheduling, but speculative decoding stays on. Six 64K windows fit MTP-3 once the
        # recurrent state is bf16, the A+C-freed 2.4 GiB is not pinned away and the draft experts are NvFp4
        # (docs/validation-20260927/kernels-phase4.md); require_focus_windows_fit checks it on the served layout.
        from arc_infra.dense_kernels_patch import PHASE2
        from arc_infra.serving_kernels_patch import PHASE4, PHASE4_MTP_TOKENS

        wide = _composed_variant(study, "fixed-2x-mtp0-wide")
        if "dense_kernels" in wide["server"] or "serving_kernels" in wide["server"]:
            raise ValueError("The phase-4 arm requires the unpatched mtp0-wide serving")
        result = copy.deepcopy(wide)
        result["server"]["mtp_tokens"] = PHASE4_MTP_TOKENS
        result["server"]["dense_kernels"] = copy.deepcopy(PHASE2)
        result["server"]["serving_kernels"] = copy.deepcopy(PHASE4)
        changed = {k: v for k, v in wide["arena_variant"]["reference_difference"].items() if k != "server.mtp_tokens"}
        if PHASE4_MTP_TOKENS != 3:
            changed["server.mtp_tokens"] = {"before": 3, "after": PHASE4_MTP_TOKENS}
        changed.update({"server.dense_kernels": {"before": None, "after": copy.deepcopy(PHASE2)},
                        "server.serving_kernels": {"before": None, "after": copy.deepcopy(PHASE4)}})
    elif name in ("fixed-2x-mtp3-wide-tweaks", "fixed-2x-mtp3-wide-tweaks-nople"):
        # Serving tweaks: the phase-4 arm with the draft head over a 16K-token subset, the PLE
        # table read over UVA and the input embedding in host memory (docs/validation-20260927/serving-tweaks.md);
        # -nople leaves the PLE on the CPU offload process (user: the PLE host read stays out of the next hour).
        # Only server.serving_kernels differs from fixed-2x-mtp3-wide-kernels.
        from arc_infra.serving_kernels_patch import PHASE4, TWEAKS, TWEAKS_NO_PLE

        tweaks = TWEAKS_NO_PLE if name.endswith("-nople") else TWEAKS
        kernels = _composed_variant(study, "fixed-2x-mtp3-wide-kernels")
        if kernels["server"].get("serving_kernels") != PHASE4:
            raise ValueError("The serving-tweaks arm requires the phase-4 serving arm")
        result = copy.deepcopy(kernels)
        result["server"]["serving_kernels"] = copy.deepcopy(tweaks)
        changed = {**kernels["arena_variant"]["reference_difference"],
                   "server.serving_kernels": {"before": None, "after": copy.deepcopy(tweaks)}}
    elif name == "fixed-2x-mtp3-wide-stack":
        # Kernel work, phase 5 ("small kernel optimizations that then could be stacked"): the
        # phase-4 arm plus numerics-preserving items only (docs/validation-20260927/kernels-phase5-stack.md).
        from arc_infra.dense_kernels_patch import PHASE5 as DENSE_PHASE5
        from arc_infra.stack_kernels_patch import PHASE5 as STACK_PHASE5

        base_arm = _composed_variant(study, "fixed-2x-mtp3-wide-kernels")
        if "stack_kernels" in base_arm["server"] or "mid_m" in base_arm["server"]["dense_kernels"]:
            raise ValueError("The stack arm extends the unmodified phase-4 serving")
        result = copy.deepcopy(base_arm)
        result["server"]["dense_kernels"] = copy.deepcopy(DENSE_PHASE5)
        result["server"]["stack_kernels"] = copy.deepcopy(STACK_PHASE5)
        changed = {**base_arm["arena_variant"]["reference_difference"],
                   "server.dense_kernels": {"before": None, "after": copy.deepcopy(DENSE_PHASE5)},
                   "server.stack_kernels": {"before": None, "after": copy.deepcopy(STACK_PHASE5)}}
    elif name == "fixed-2x-mtp0-wide-kv8-hero":
        # Hero serving: the MTP-0 kv8 stack arm with the target's NVFP4 experts on vLLM's
        # Marlin W4A16 kernel (expert quantization stage 1: activation-FP4 error removed, 0.46-0.80x the CUTLASS time).
        stack_arm = _composed_variant(study, "fixed-2x-mtp0-wide-kv8-stack")
        kernels = copy.deepcopy(stack_arm["server"]["serving_kernels"])
        if kernels.get("moe_backend") is not None or stack_arm["server"].get("mtp_tokens") != 0:
            raise ValueError("The hero arm extends the MTP-0 kv8 stack arm on the default CUTLASS MoE backend")
        kernels["moe_backend"] = "marlin"
        result = copy.deepcopy(stack_arm)
        result["server"]["serving_kernels"] = kernels
        changed = copy.deepcopy(stack_arm["arena_variant"]["reference_difference"])
        changed["server.serving_kernels"] = {"before": None, "after": copy.deepcopy(kernels)}
    elif name == "fixed-2x-mtp0-wide-kv8-hero-21x42k":
        # Matched stream-count test (evening: "a matched 21 x 42k 1-hour run"): the hero arm with
        # 21 focus games at a 42,496-token window instead of 14 at 64K. On the hero smoke's served layout (860 blocks,
        # 774 at the 0.9 gate) a 42,496 + 256-token request costs 36 blocks, so 21 fit (42,752 is the largest window
        # that does); the calibrated simulator puts 21 x ~42K at +4.3 (9 h) over 14 x 64K without a window penalty
        # (break-even multiplier 0.67; docs/validation-20260927/streams-9h.md section 8). Everything else is the hero's.
        hero = _composed_variant(study, "fixed-2x-mtp0-wide-kv8-hero")
        if hero["active_limit"] != MTP0_KV8_FOCUS_SLOTS or hero["windows"] != [65536]:
            raise ValueError("The 21 x 42K arm requires the 14 x 64K hero arm")
        result = copy.deepcopy(hero)
        result["active_limit"] = HERO_21_SLOTS
        result["windows"] = [HERO_21_WINDOW]
        changed = copy.deepcopy(hero["arena_variant"]["reference_difference"])
        changed["active_limit"]["after"] = HERO_21_SLOTS
        changed["windows"] = {"before": [131072], "after": [HERO_21_WINDOW]}
    elif name == "fixed-2x-mtp0-wide-kv8-hero-6x64k":
        # Depth test (28 September, after the hero preview scored 0.72% with 20 first clears but nothing past a second
        # level, against kernels-v2's 1.42% carried by one 7-level sb26 clone): the hero serving with kernels-v2's six
        # focus games at 64K, so each focus game gets the per-stream rate of six streams instead of fourteen.
        hero = _composed_variant(study, "fixed-2x-mtp0-wide-kv8-hero")
        if hero["active_limit"] != MTP0_KV8_FOCUS_SLOTS or hero["windows"] != [65536]:
            raise ValueError("The 6 x 64K arm requires the 14 x 64K hero arm")
        result = copy.deepcopy(hero)
        result["active_limit"] = HERO_6_SLOTS
        changed = copy.deepcopy(hero["arena_variant"]["reference_difference"])
        changed["active_limit"]["after"] = HERO_6_SLOTS
    elif name == "fixed-2x-mtp0-wide-kv8-stack":
        # Kernel work, phase 5 on the MTP-0 kv8 arm (evening: the main candidate): the
        # target-side stack items; without speculative decoding only mid_m acts (M = 8 at six streams, 16 at 14).
        from arc_infra.dense_kernels_patch import PHASE5 as DENSE_PHASE5
        from arc_infra.stack_kernels_patch import PHASE5 as STACK_PHASE5

        base_arm = _composed_variant(study, "fixed-2x-mtp0-wide-kernels-kv8")
        if "stack_kernels" in base_arm["server"] or "mid_m" in base_arm["server"]["dense_kernels"]:
            raise ValueError("The MTP-0 stack arm extends the unmodified MTP-0 kv8 serving")
        result = copy.deepcopy(base_arm)
        result["server"]["dense_kernels"] = copy.deepcopy(DENSE_PHASE5)
        result["server"]["stack_kernels"] = copy.deepcopy(STACK_PHASE5)
        changed = {**copy.deepcopy(base_arm["arena_variant"]["reference_difference"]),
                   "server.dense_kernels": {"before": None, "after": copy.deepcopy(DENSE_PHASE5)},
                   "server.stack_kernels": {"before": None, "after": copy.deepcopy(STACK_PHASE5)}}
    elif name == "fixed-2x-mtp3-wide-kernels-kv8":
        # 8-bit QSA KV cache: the phase-4 arm with the main KV in int8,
        # which frees about a third of every 64K window's blocks; the freed capacity becomes more focus games.
        kernels = _composed_variant(study, "fixed-2x-mtp3-wide-kernels")
        if kernels["server"]["serving_kernels"].get("kv_cache_8bit") or kernels["active_limit"] != MTP0_WIDE["active_limit"]:
            raise ValueError("The kv8 arm requires the phase-4 arm with BF16 KV and six focus games")
        result = copy.deepcopy(kernels)
        result["server"]["serving_kernels"]["kv_cache_8bit"] = KV8_MODE
        result["server"]["max_cudagraph_capture_size"] = KV8_GRAPH_CAPTURE_SIZE
        result["active_limit"] = KV8_FOCUS_SLOTS
        result["opening_switch_level1_games"] = KV8_FOCUS_SLOTS
        changed = copy.deepcopy(kernels["arena_variant"]["reference_difference"])
        changed["server.serving_kernels"]["after"] = copy.deepcopy(result["server"]["serving_kernels"])
        changed["active_limit"]["after"] = KV8_FOCUS_SLOTS
        changed["opening_switch_level1_games"]["after"] = KV8_FOCUS_SLOTS
        changed["server.max_cudagraph_capture_size"] = {"before": kernels["server"].get("max_cudagraph_capture_size", 32),
                                                        "after": KV8_GRAPH_CAPTURE_SIZE}
    elif name == "fixed-2x-mtp0-wide-kernels-kv8":
        kernels = _composed_variant(study, "fixed-2x-mtp0-wide-kernels")
        if ("serving_kernels" in kernels["server"] or kernels["server"].get("mtp_tokens") != 0
                or kernels["active_limit"] != MTP0_WIDE["active_limit"] or kernels["opening_switch_level1_games"] != 6):
            raise ValueError("The MTP-0 kv8 arm requires the kernels-v2 arm (MTP-0, six slots, six-clear switch)")
        result = copy.deepcopy(kernels)
        result["server"]["serving_kernels"] = dict(MTP0_KV8_SERVING)
        result["active_limit"] = MTP0_KV8_FOCUS_SLOTS  # the opening switch stays at six clears (user)
        changed = copy.deepcopy(kernels["arena_variant"]["reference_difference"])
        changed["server.serving_kernels"] = {"before": None, "after": dict(MTP0_KV8_SERVING)}
        changed["active_limit"]["after"] = MTP0_KV8_FOCUS_SLOTS
    elif name in EXPERT_PRUNE_VARIANTS:
        # Load-time expert pruning: hero serving plus server.expert_prune; the freed
        # memory becomes focus games at 64K. Only server.expert_prune and active_limit change.
        spec = EXPERT_PRUNE_VARIANTS[name]
        hero = _composed_variant(study, "fixed-2x-mtp0-wide-kv8-hero")
        if (hero["server"].get("mtp_tokens") != 0 or "expert_prune" in hero["server"]
                or hero["active_limit"] != MTP0_KV8_FOCUS_SLOTS or hero["windows"] != [65536]):
            raise ValueError("The pruned hero arms extend the MTP-0 14 x 64K hero serving")
        # The prune GPU check (arena-expert-prune-check-v1-modal-20260929) served 1,245 / 1,667 blocks (7.59 / 15.88 GiB
        # freed, 0.32 GiB under the arithmetic for prune12): the served layout sets the slots, not the projection.
        slots = min(int(0.9 * spec["served_blocks"]) // HERO_WINDOW_BLOCKS, hero["server"]["max_num_seqs"])
        result = copy.deepcopy(hero)
        result["server"]["expert_prune"] = {"mask": spec["mask"], "sha256": spec["sha256"]}
        result["active_limit"] = slots
        changed = copy.deepcopy(hero["arena_variant"]["reference_difference"])
        changed["server.expert_prune"] = {"before": None, "after": dict(result["server"]["expert_prune"])}
        changed["active_limit"]["after"] = slots
    elif name == "fixed-2x-mtp0-wide-kv8-hero-prefix":
        # Prefix cache v2 ("implement the improvements to the evicted prefixes issue, and the state
        # lag"): only cache_experiment.prefix_cache changes (two hash-guarded runtime patches, arena.prefix_cache_patch).
        hero = _composed_variant(study, "fixed-2x-mtp0-wide-kv8-hero")
        if hero["server"].get("mtp_tokens") != 0 or "prefix_cache" in hero["cache_experiment"]:
            raise ValueError("The prefix-cache arm extends the MTP-0 hero serving (keep-state needs MTP-0)")
        result = copy.deepcopy(hero)
        result["cache_experiment"]["prefix_cache"] = dict(HERO_PREFIX_CACHE)
        changed = copy.deepcopy(hero["arena_variant"]["reference_difference"])
        changed["cache_experiment.prefix_cache"] = {"before": None, "after": dict(HERO_PREFIX_CACHE)}
    elif name == "fixed-2x-mtp0-wide-kv8-hero-uva":
        # Host-memory serving tweaks on the hero: the PLE table
        # and the input embedding in page-locked host memory read over UVA (no PLE offload process, 1.18 GiB of GPU
        # memory back to the KV pool). No draft_vocab: MTP-0 has no draft head. Only server.serving_kernels changes.
        # Projected on the hero smoke's layout (860 blocks, 774 at the gate, 51 per 64K window): +60 blocks, 827 at
        # the gate, 16 windows (a 17th needs 867); the focus slots stay the hero's 14.
        hero = _composed_variant(study, "fixed-2x-mtp0-wide-kv8-hero")
        kernels = copy.deepcopy(hero["server"]["serving_kernels"])
        if hero["server"].get("mtp_tokens") != 0 or kernels.get("ple_uva") or kernels.get("embed_host"):
            raise ValueError("The hero-uva arm extends the MTP-0 hero serving without host-memory tweaks")
        kernels.update(ple_uva=True, embed_host=True)
        result = copy.deepcopy(hero)
        result["server"]["serving_kernels"] = kernels
        changed = copy.deepcopy(hero["arena_variant"]["reference_difference"])
        changed["server.serving_kernels"] = {"before": None, "after": copy.deepcopy(kernels)}
    elif name == "fixed-2x-mtp0-wide-kv8-hero-p6":
        # Kernel work, phase 6 (docs/validation-20260927/kernels-phase6.md): the numerics-preserving items on the hero,
        # all bitwise or byte-equal to the hero's served path: the UVA host tables (hero-uva) and the fused split-K
        # reduction on the unchanged phase-5 tile plan. Only server.serving_kernels and server.dense_kernels change.
        from arc_infra.dense_kernels_patch import PHASE5 as DENSE_PHASE5
        from arc_infra.dense_kernels_patch import PHASE6_FUSED

        uva = _composed_variant(study, "fixed-2x-mtp0-wide-kv8-hero-uva")
        if uva["server"]["dense_kernels"] != DENSE_PHASE5:
            raise ValueError("The phase-6 arm extends the hero-uva arm on the phase-5 dense setting")
        result = copy.deepcopy(uva)
        result["server"]["dense_kernels"] = copy.deepcopy(PHASE6_FUSED)
        changed = copy.deepcopy(uva["arena_variant"]["reference_difference"])
        changed["server.dense_kernels"] = {"before": None, "after": copy.deepcopy(PHASE6_FUSED)}
    elif name == "fixed-2x-deep-explore":
        # User design (28 September, ideas.md): "more streams at lower context + depthwise scheduling: always keep playing
        # the first game that clears in 1 slot, then rotate the others in the other slots to find the next game to clear
        # a level and so on", with mixed context sizes. Serving is the kernels arm's (MTP-0 + dense kernels); the
        # scheduler is ``arena.deep_explore`` (docs/validation-20260927/deep-explore.md).
        from arc_infra.dense_kernels_patch import PHASE2

        from .deep_explore import DEFAULTS as DEEP_EXPLORE

        if (result["server"].get("mtp_tokens") != 3 or result["server"].get("speculative") is not None
                or "dense_kernels" in result["server"] or result["active_limit"] != 2
                or result["windows"] != [131072] or result.get("level_budget_policy") != "fixed-2x"
                or result.get("backlog_policy") != "readmissions" or result.get("cache_admission") != "hybrid_blocks"):
            raise ValueError("The deep-explore arm requires the MTP-3, two-slot, 128K fixed-2x hybrid-block control")
        # User (28 September): one deep slot and five exploration slots (fewer, faster streams than the 1 + 8 default;
        # the calibrated simulator favours fewer streams at the same aggregate rate).
        DEEP_EXPLORE = {**DEEP_EXPLORE, "explore_slots": 5}
        slots = DEEP_EXPLORE["deep_slots"] + DEEP_EXPLORE["explore_slots"]
        result["server"]["mtp_tokens"] = MTP0_WIDE["mtp_tokens"]
        result["server"]["dense_kernels"] = copy.deepcopy(PHASE2)
        result["level_budget_policy"] = "deep-explore"
        result["deep_explore"] = dict(DEEP_EXPLORE)
        result["active_limit"] = slots
        result["windows"] = [DEEP_EXPLORE["deep_window"]]  # the largest request; exploration slots compact at 32K
        result["cache_experiment"]["compaction_changed"] = True
        changed = {"server.mtp_tokens": {"before": 3, "after": MTP0_WIDE["mtp_tokens"]},
                   "server.dense_kernels": {"before": None, "after": copy.deepcopy(PHASE2)},
                   "level_budget_policy": {"before": "fixed-2x", "after": "deep-explore"},
                   "deep_explore": {"before": None, "after": dict(DEEP_EXPLORE)},
                   "active_limit": {"before": 2, "after": slots},
                   "cache_experiment.compaction_changed": {"before": False, "after": True}}
    elif name in DEPTHWISE_SERVING:
        result, changed = _depthwise_variant(study, DEPTHWISE_SERVING[name], name)
    elif name in EXPERT_QUANT_SOURCE:
        result, changed = _expert_quant_variant(study, name)
    elif name in RANK_SERVING:
        result, changed = _rank_variant(study, RANK_SERVING[name])
    elif name == "fixed-2x-gittins-c110-wide":
        # Combination (27 September): on the 110-clone hour mtp0-wide raised completion tokens 51% (1.40M) but spent
        # them on first levels across more games (0.72%), while gittins-c110 committed to deep games on two slots
        # (0.84%). The simulator's best six-slot policy is the c110 index with the two-clear opening (2.08 against
        # 1.84 with the six-clear switch and 1.20-1.35 for fixed-2x); per-stream decode at six streams is ~75 tok/s.
        wide = _composed_variant(study, "fixed-2x-mtp0-wide")
        index = _composed_variant(study, "fixed-2x-gittins-c110")
        result = copy.deepcopy(wide)
        for key in ("level_budget_policy", "index_scheduler", "opening_max_wall_seconds", "opening_cap_absolute",
                    "level_budget_rule", "backlog_priority"):
            result[key] = copy.deepcopy(index[key])
        result["index_scheduler"]["stream_tokens_per_second"] = 75.0
        result["opening_switch_level1_games"] = 2
        changed = {**wide["arena_variant"]["reference_difference"], **index["arena_variant"]["reference_difference"],
                   "index_scheduler": {"before": None, "after": dict(result["index_scheduler"])},
                   "opening_switch_level1_games": {"before": 2, "after": 2}}
    elif name == "fixed-2x-gittins-c110":
        # 110 cyclic clones of 20 games, one hour, two slots (docs/validation-20260927/scheduler-c110.md). In the
        # trace-driven simulator with the within-level world, 110 games x even 3K tokens is a third of the hour, and
        # a broad opening buys no information (a level barely clears in its first 5K tokens). What pays is the
        # two-clear opening (its first clears are the fast, deep games) and then commitment: the Gittins index on
        # the within-level shape with an in-run prior scale and a 4x hysteresis (a waiting game must score five
        # times the worst focus game).
        if result.get("level_budget_policy") != "fixed-2x" or result.get("backlog_policy") != "readmissions":
            raise ValueError("Gittins comparison requires the fixed-2x control")
        from .gittins import C110, C110_OPENING_MAX_WALL_SECONDS

        result["level_budget_policy"] = "index"
        result["index_scheduler"] = dict(C110)
        result["opening_max_wall_seconds"] = C110_OPENING_MAX_WALL_SECONDS
        # The cap is stated for the 110-instance roster; arena.competition must not rescale it.
        result["opening_cap_absolute"] = True
        changed = {"level_budget_policy": {"before": "fixed-2x", "after": "index"},
                   "index_scheduler": {"before": None, "after": dict(C110)},
                   "opening_max_wall_seconds": {"before": None, "after": C110_OPENING_MAX_WALL_SECONDS},
                   "opening_cap_absolute": {"before": None, "after": True}}
    elif name in ("fixed-2x-commit", "fixed-2x-commit-queue"):
        # Clear-rate curve (26 September, 1,088 level attempts in 29 one-hour runs): clears per 10K tokens
        # peak at 10-25K tokens on a level and fall past 50K; ten games per run stay under 10K on level 1
        # while stuck deep games hold the focus slots (re86 299K) and fast games are not committed to after
        # a lane clear (ar25 in index-lane-v3). Phase 2 kept KV use at 45% mean with two streams.
        if result.get("level_budget_policy") != "fixed-2x" or result.get("backlog_policy") != "readmissions":
            raise ValueError("Commit comparison requires the fixed-2x control")
        if (result.get("backend", "vllm") != "vllm" or result.get("candidate_profile")
                or result.get("cache_admission") != "hybrid_blocks" or "scheduling_policy" in result["server"]):
            raise ValueError("Commit comparison requires the Flash vLLM hybrid-block profile (arena.serving)")
        from .commit import COMMIT_DEFAULTS

        result["level_budget_policy"] = "commit"
        result["commit"] = dict(COMMIT_DEFAULTS)
        if name == "fixed-2x-commit-queue":
            # commit-v1 (4.66%): lane clears replaced ar25 and sb26 right after their level-1 clears; a lane
            # clear now takes only a free or stalled slot and otherwise waits in a queue.
            result["commit"]["replace_unstalled"] = False
        # Serving change (user-approved for throughput): lane requests are preemptible under vLLM priority
        # scheduling; arena.serving appends --scheduling-policy priority to the pinned command.
        result["server"]["scheduling_policy"] = "priority"
        changed = {"level_budget_policy": {"before": "fixed-2x", "after": "commit"},
                   "commit": {"before": None, "after": dict(COMMIT_DEFAULTS)},
                   "server.scheduling_policy": {"before": None, "after": "priority"}}
    elif name == "fixed-2x-token-budget":
        # Budget audit (26 September, user: try a+b): a busy second buys 66 tokens in the opening and
        # 108 in phase 2, T1 accumulates over level-0 admissions (sp80 662 s) and easy first levels
        # give short budgets later (sb26 29 s, ar25 74 s parked after 444 s on level 2).
        if result.get("budget_clock") != "busy" or result.get("t1_cap_seconds") is not None:
            raise ValueError("Token-budget comparison requires the busy-clock fixed-2x control without a T1 cap")
        result["budget_clock"] = "normalized"
        result["cost_normalization"] = {"rate_tokens_per_second": 104.0, "context_weight": 0.0,
                                        "context_reference": 131072, "request_overhead_seconds": 1.65}
        result["t1_clip"] = [0.5, 2.0]
        changed = {"budget_clock": {"before": "busy", "after": "normalized"},
                   "cost_normalization": {"before": None, "after": dict(result["cost_normalization"])},
                   "t1_clip": {"before": None, "after": [0.5, 2.0]}}
    elif name in ("fixed-2x-explorer-start", "fixed-2x-explorer-start-silent"):
        # Probe A/B (26 September): the value of the explorer's demonstration for level 2 and beyond,
        # with the explorer's own clear held equal in both arms.
        report = name == "fixed-2x-explorer-start"
        result["explorer"] = {"at_start": True, "report": report, "max_level": -1, "max_actions": 20000,
                              "max_seconds": 90, "min_remaining_seconds": 120}
        changed = {"explorer": {"before": None, "after": dict(result["explorer"])}}
    elif name == "fixed-2x-explorer":
        # 26 September (ideas.md goal: more useful actions per unit of compute): waiting games get
        # 3-28 actions all hour while their environments step at 2,000-4,000 actions/s on a CPU; offline
        # the explorer cleared level 1 of eight roster games in 20,000 actions with short winning paths.
        result["explorer"] = {"max_level": 0, "max_actions": 20000, "max_seconds": 90,
                              "min_remaining_seconds": 120}
        changed = {"explorer": {"before": None, "after": dict(result["explorer"])}}
    elif name == "fixed-2x-mtp4":
        # Serving-only change: a deeper MTP draft where phase 2 runs one or two streams and the
        # GPU is otherwise idle; the pinned serving script accepts at most four draft tokens.
        if result["server"].get("mtp_tokens") != 3:
            raise ValueError("MTP depth comparison requires the MTP-3 control")
        result["server"]["mtp_tokens"] = 4
        changed = {"server.mtp_tokens": {"before": 3, "after": 4}}
    else:
        effort = name.rsplit("-", 1)[1]
        served = result.get("served_model") or "Qwen/Qwen3.8-Flash-Next-NVFP4"
        if result.get("reasoning_effort") != "xhigh" or effort not in allowed_efforts(served):
            raise ValueError(f"{served} reasoning-effort comparison requires the xhigh control and a template-accepted effort")
        result["reasoning_effort"] = effort
        changed = {"reasoning_effort": {"before": "xhigh", "after": effort}}
    result["arena_variant"] = {
        "name": name, "description": COMPOSED[name],
        "changed": {**base["arena_variant"]["changed"], **changed},
        "checkpoint_and_serving_unchanged": name not in SERVING_CHANGED,
        "reference_variant": "reclaim-fixed-2x", "reference_difference": changed,
        "reference_checkpoint_and_serving_unchanged": name not in SERVING_CHANGED,
    }
    return result
