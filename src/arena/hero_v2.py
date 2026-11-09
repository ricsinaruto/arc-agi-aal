"""hero-v2: the hero arm plus every validated improvement, each one opt-in.

User: "Run new hero (with token budgets: hero-v2) with 64k context plus all improvements below for 2 hours ...
remember to set number of streams to the max possible": serving tweaks, all kernel improvements, expert pruning and our
custom quantization. ``compose`` starts from ``fixed-2x-mtp0-wide-kv8-hero`` (MTP-0, A+C dense kernels + phase-5
tiles, top-k-first sampler, int8 QSA KV, Marlin W4A16 experts, 14 x 64K, six-clear switch) and switches components
on one at a time. Every component names what it changes; ``study["hero_v2"]`` records which are on, which are off and
why, the projected served layout and the stream count; ``arena_variant.reference_difference`` is the change against
the hero.

Components (``COMPONENTS``; ``status()`` says which the current tree can build):

- ``token_budgets``: level budgets in completion tokens (``budget_clock = "tokens"``); needs the token-clock scheduler.
- ``serving_tweaks``: the PLE table and the input embedding in page-locked host memory over UVA (``ple_uva``,
  ``embed_host``; the hero-uva arm); frees ~1.27 GB of GPU memory, about +60 KV blocks.
- ``kernels_phase6``: the phase-6 dense-kernel plan (fused split-K reduction, bitwise equal to phase 5).
- ``kv8_prefill``: the BF16 prefill path for the int8 QSA cache (``serving_kernels.kv8_prefill = "bf16"``; kernel
  work, phase 7): prefill-chunk rows attend over dequantized pages with the pinned BF16 kernel, decode keeps the int8
  kernel and the cache stays int8 (capacity unchanged). Off until its GPU validation passes.
- ``expert_prune``: a load-time pruning mask by level (``prune12``, ``prune25``; masks pinned in
  ``arena.variants.EXPERT_PRUNE_VARIANTS``); the freed memory becomes KV blocks.
- ``gptq_arc``: the ARC-GPTQ routed-expert overlay (drop-in NVFP4 shards; Modal assets volume only). With pruning the
  loader reads the overlay's shards (the symlink farm) and streams them through the prune filter, which drops and
  renumbers experts by tensor name; the overlay shards carry the checkpoint's per-expert names, so the same filter
  covers them (``overlay_prune_plan`` checks a manifest against a mask; tests/test_hero_v2.py on synthetic shards).
- ``topk_sampler``: keep (default) or drop the top-k-first sampler (drop it if the sampler check finds that it
  changes the sampling distribution).
- ``stall_dump``: the stall-only worker stack dump (one-shot timer per step, N seconds); rides on the prune overlay.
- ``prefix_cache``: prefix cache v2 (``cache_experiment.prefix_cache``, arena.prefix_cache_patch): prefill chunks
  aligned to the Mamba block (the state-lag fix), the newest boundary state kept with its request and superseded
  states demoted in the free queue (docs/validation-20260927/prefix-cache-v2.md); off until its GPU smoke passes.
- ``live_progress``: periodic progress (score, levels, tokens, throughput) readable from outside during the run.
- ``compaction_thinking``: the Duck compaction summary mode (``arena.compaction.COMPACTION_THINKING``); default
  "prefilled-empty": the gameplay prefix plus a pre-filled empty think block, so the summary
  reuses the cached context without reasoning ("gameplay": thinking at the gameplay effort, the hero's).
- ``reasoning_history``: how the Duck agents re-send earlier replies (``arc_infra.config.reasoning_history``); default
  "verbatim" ("yes to the round-trip fix"): the server's reasoning and content strings with only
  ``<think>`` tags removed, so the next prompt continues the generated tokens and the prefix cache reuses the reply
  ("normalized": the public loop's blank-line removal, the hero's).
- ``harness_fixes``: the harness fixes of the hero-v2 2 h audit ("bug fixes + inventory";
  ``arena.harness_fixes``): exact-count compaction trigger, summary-room guard, post-compaction inventory, cap-hit
  follow-up, active-time stuck notice, sandbox modules and timeout interrupt, prompt text. Off by default (every
  earlier build plays without them); true switches every sub-fix on, a list selects some; study ``harness_fixes``
  records the sub-fixes and thresholds. The opt-in ``summary_headroom`` and ``inventory_imports``
  (``arena.harness_fixes.OPTIONAL_FIXES``, 30 September) go in a list only: true keeps runs A and B's eight.

Follow-up runs (docs/validation-20260927/hero-v2-next-runs.md):

- ``kv_cache``: "int8" (the hero's QSA cache) or "int4" (``serving_kernels.kv_cache_4bit``, 3,136-token blocks, MTP-0
  only; docs/validation-20260927/kv-cache-4bit.md). The int4 pool is the int8 pool's bytes re-cut into int4 blocks.
- ``window``: 65,536 (the hero's) or 131,072; the compaction threshold and the gate follow the window as in every
  arena run (the Duck agents compact against ``game.window``; the gate reserves cost(prompt + cap) blocks).
- ``budget_policy``: "fixed-2x" (the hero's) or "rank" (``arena.rank``, the ``hero-v2-rank`` settings: 60 s tick,
  10K-token tenure, 5% swap margin, no opening, no level budgets) with the focus slots as the ranking's slots, or
  "priority" (``arena.priority``, dfranzen's priority gate at our stream count): the focus slots are
  the gate's slots, unstarted games first in roster order, handovers only at context evictions (summary compaction or
  drain) to the best-scored waiting game, no opening, no level budgets, no retirement. The optional ``priority``
  component is a dict of overrides of ``arena.priority.DEFAULTS`` (their submitted settings); it needs
  ``backlog_fix`` off (the fixed-2x backlog order does not apply).
- ``focus_slots``: a target focus count in place of "the most full windows that fit". Without oversubscription the
  trial runs min(target, the served fit) (``focus_fit_fallback``); ``focus_oversubscription`` runs the target even when
  fewer full windows fit: no preemption is possible because the exact KV gate (``arena.gate``: cost(prompt + cap)
  blocks per request, at most 0.9 x the served blocks in all) holds requests until the pool has room.
  ``max_num_seqs`` is then max(25, focus + 2) and the CUDA-graph capture size rises above 32 when it must.
- ``backlog_fix``: the fixed-2x backlog order of run 2 (``arena.scheduler`` ``backlog_key`` "progress-fruitless":
  progress since the last admission, then the fewest fruitless tenures on the current level, then more levels, then
  ``level0_order`` "tokens" (the most gameplay tokens on level 0) for level-0 games, then roster order); a same-tick
  re-admission after a park is logged "regrant_same_tick".
- ``prompt_tokens_details``: ``--enable-prompt-tokens-details`` (exact cached tokens in every usage; logging only).
- ``prefix_match_unit``: ``--prefix-match-unit N`` (prefix-hit granularity inside a block; off until its warm/cold check).
- ``spare_sequences``: ``max_num_seqs`` = focus + this (default 2, runs A/B); 4 for the competition run-B build at 32
  streams (prefix-unit-int4.md section 4: with four spares the waits for a sequence slot vanish).
- ``density_blocks``: the served pool (in the served KV's blocks) on which an oversubscribed focus target was validated
  by the gate replay. The startup density is then floor(0.9 x density_blocks) / focus: a pool of at least that size
  runs the full target, a smaller one proportionally fewer streams (default: the projected pool).

Three-block run (``hero-v2-blocks``, docs/validation-20260927/hero-v2-blocks.md):

- ``budget_policy`` "blocks" (``arena.blocks``): the three-block schedule with its settings in ``blocks`` (a dict of
  overrides of ``arena.blocks.DEFAULTS``; the slots are the focus streams); no opening, no level budgets.
- ``window``: also any multiple of 1,024 (a cost-table row) from 65,536 to 131,072 (the three-block window is chosen
  by the gate replay).
- ``gate_margin``: the KV gate's cap as a share of the served blocks (study ``kv_margin``; default 0.90, every earlier
  build). 0.96 for the next three-block build (MTP-0: every request is already charged whole
  blocks; hero-v2-blocks.md gives the engine's peak use against the pool in runs A, B, 1, hero-v2 and the first blocks
  run). The projected gate, the fit and the density follow it.
- ``graph_capture``: an explicit decode CUDA-graph capture size (``--max-cudagraph-capture-size``; default: the next
  multiple of 8 above 32 when the sequences need it). vLLM captures the sizes 1, 2, 4, 8, 16, ... up to it, so a size
  that is not a multiple of 8 captures up to the multiple of 8 below it.
Opt-in harness features (30 September; docs/validation-20260927/parallel-samples-plans.md), ``OPTIONAL_COMPONENTS``:
recorded in the study only when requested, so every build that does not name them (runs A and B, the competition
run-B build) is byte-for-byte unchanged.

- ``parallel_samples``: true or a dict over ``arena.parallel_samples.DEFAULTS`` (k 4, min_levels 2, max_num_seqs 48):
  each eligible gameplay request has up to k samples; the first tool reply wins and losers abort. Siblings reserve
  full contexts by default; prefix warm-up and measurement-only loser completion are optional. Ordinary admission
  slots stay unchanged; the backend limit and decode graph capture increase for sibling-only capacity.
  Needs prefix_cache and the plain gate: budget_policy fixed-2x, or blocks (campaign arm
  blocks3-samples2; the three-block scheduler admits through the same plain KVGate).
- ``action_plans``: true adds ``run_plan`` to the Duck sandbox and one paragraph to the system prompt.
- ``drain_half``: true or a dict over ``arena.context_drain.DEFAULTS`` (dfranzen's "drop half" context
  management in place of the summary compaction): at the compaction trigger the oldest whole history blocks are dropped
  until the request is ``drain_tokens`` (58K, at most half the budget) below the budget, with no summary request; a
  turn cap (150 assistant messages, drained by 30, coalesced with the token drain), 3,072-token tool outputs with
  head/tail middle truncation, malformed tool calls kept out of history, and a forced drain after a context-length
  rejection. The system prompt and the latest observation are never dropped; reasoning stays verbatim; the persistent
  python state is untouched.
- ``dfr_harness``: true or a dict of booleans over ``arena.dfr_harness.FLAGS`` (vision, guards, actions, level_prompt,
  frame_diff, animation): harness features ported from dfranzen's ARC-AGI-3 Milestone 2 notebook
  (Apache-2.0) into the corrected Duck agent; recorded only when on, so every other build is byte-for-byte unchanged.
- ``mtp_tokens``: 1 serves the MTP-1 speculative draft (campaign arm
  blocks3-plans-prune25-40-mtp1; docs/validation-20260930/spec-decode.md section 5): ``server.mtp_tokens`` 1 with the
  pilot-validated draft stack (BF16 GDN state, NvFp4 draft experts, the 16K draft vocabulary), int4 KV only (the
  3,136-token block still holds the MTP-1 conv state), 4 more fixed state blocks per request (13: the 4 GDN groups
  hold 2 + k state pages each), blocks of 13 QSA-layer pages (the MTP layer), and the pilot's CUDA-graph ladder
  [1, 2, 4, 8, 16, 24, 30, 32, 36, 40, 44] streams x (k + 1) tokens up to max_num_seqs (``graph_capture`` must be
  (k + 1) x max_num_seqs). The gate charges the draft tokens through the served layout's lookahead (ARC_CACHE_LAYOUT)
  and prefix cache v2's keep/demote switch themselves off under speculation.

Streams: the most full windows that fit the projected served layout at the 0.9 gate (51 blocks per 64K window on the
hero layout, by the served cost table's 1,024-token rows), capped by the server's sequences; ``max_num_seqs`` rises above the hero's 25 only when more windows fit, up
to the MTP-0 decode CUDA-graph capture size (32: one token per stream per step, so every decode batch stays in the
captured graphs). ``focus_fit_fallback`` lets a slightly smaller served pool run with what fits (at least 14)
instead of refusing after the setup; ``served_blocks`` (measured by the hero-v2 smoke) replaces the projection.
"""

import copy
import re

NAME = "fixed-2x-mtp0-wide-kv8-hero-v2"
BASE = "fixed-2x-mtp0-wide-kv8-hero"
WALL_SECONDS = 7200
# Served layout of the hero smoke (docs/validation-20260927/hero-smoke.md): 860 blocks, 51 per 64K window, 0.9 gate.
HERO_BLOCKS, WINDOW_BLOCKS, MARGIN, WINDOW = 860, 51, 0.9, 65536
# serving-tweaks.md: the host-resident embedding alone frees 1.27 GB, +60 blocks of 21.2 MB on the hero layout.
UVA_BLOCKS = 60
SEQ_FLOOR = 25   # the hero's max_num_seqs (and generation slots)
SEQ_CAP = 32     # arc_infra.serving SERVER_ENV capture size 32; MTP-0 decodes one token per stream per step
# Explicit focus targets (follow-up runs): max_num_seqs = focus + SEQ_SPARE (hero-v2: 23 focus, 25 sequences), up to
# SEQ_CAP_RAISED; above SEQ_CAP the decode CUDA graphs are captured up to the next multiple of 8 (the pinned capture
# sizes step by 8 above 8). Decode batches above 32 take the pinned dense path (the phase-5 mid-M tiles cover 8..32).
SEQ_SPARE = 2
SEQ_CAP_RAISED = 40
# 60 / 64 until 30 September; raised for campaign arm blocks2-plans-prune50-int3 (user: "however many streams fit with a
# 96k context size": 76 streams, 80 sequences, graphs to 80). vLLM (serving_setup) accepts up to 256.
FOCUS_RANGE = (14, 92)
SPARE_RANGE = (2, 8)
SEQ_LIMIT = 96  # arc_infra.serving: generation slots 25..96
# Served geometry (docs/validation-20260927/kv-cache-4bit.md section 4, scripts/kv4_sizing.py hero_v2_table): per
# QSA layer and token, the main KV page (int8 1,040 B, int4 528 B) plus the BF16 compressed indexer keys (64 B), on 12
# layers; blocks of 1,600 (int8, the indexer backend's rule) or 3,136 tokens (int4, KV4_BLOCK_TOKENS); 9 fixed blocks
# per request (the raw-key ring and 4 GDN groups x 2 state pages).
KV_MODES = {"int8": {"bytes_per_token": 1040, "block_tokens": 1600},
            "int4": {"bytes_per_token": 528, "block_tokens": 3136}}
QSA_LAYERS, INDEXER_BYTES, FIXED_BLOCKS = 12, 64, 9
# The served cost table (arena.cache_export_runtime) has one row per 1,024 tokens; the gate charges the first row at
# or above the request, so a full window costs its 1,024-rounded length (131,328 -> 132,096 tokens: 43 int4 blocks).
COST_ROW_TOKENS = 1024
FRESH_OVERHEAD = 256  # arena.capacity.FRESH_OVERHEAD_TOKENS
# 130,816: 128K minus 256, so a full request (window + 256) is exactly 131,072 tokens,
# the cost row 131,072 -> 42 int4 attention blocks + 9 = 51, and 22 full windows fit 1,143 gate blocks.
WINDOWS = (65536, 130816, 131072)
# The three-block run (29 September): any multiple of 1,024 between 64K and 128K, chosen by the gate replay.
WINDOW_STEP = 1024  # the served cost table's row
# windows above 131,072 where they fit; a request is window + 256, within max_model_len 262,144.
WINDOW_MAX = 261120
BUDGET_POLICIES = ("fixed-2x", "rank", "blocks", "priority")
FIT_FALLBACK_MIN = 14  # the hero's own 14 focus games
PRUNE_LEVELS = {"prune50-layer": "fixed-2x-mtp0-wide-kv8-hero-prune50-layer", "prune37-layer": "fixed-2x-mtp0-wide-kv8-hero-prune37-layer", "prune25-layer": "fixed-2x-mtp0-wide-kv8-hero-prune25-layer", "prune12": "fixed-2x-mtp0-wide-kv8-hero-prune12", "prune25": "fixed-2x-mtp0-wide-kv8-hero-prune25"}
LIVE_PROGRESS = {"interval_seconds": 60, "score_seconds": 300}
STALL_DUMP_SECONDS = 90

DEFAULT_COMPONENTS = {
    "token_budgets": True,
    "serving_tweaks": True,
    # Off: kernel-phase6-v1 measured no gain for the fused split-K reduction
    # (docs/validation-20260927/kernels-phase6.md section 6); selectable, never on by default.
    "kernels_phase6": False,
    # Off until its GPU validation (kernel gate + injected-prefill bench) passes (kernels-phase7.md).
    "kv8_prefill": False,
    "expert_prune": "prune12",
    "gptq_arc": True,
    "topk_sampler": True,
    "stall_dump": STALL_DUMP_SECONDS,
    "prefix_cache": False,
    # Prefix cache v3 (29 September, prefix-cache-v3.md): off until the operator and the user wire them into a run.
    "request_end_tail": False,
    "keep_alive": False,
    "live_progress": True,
    "compaction_thinking": "prefilled-empty",
    "reasoning_history": "verbatim",
    # Follow-up runs (29 September); the defaults are hero-v2's own settings.
    "kv_cache": "int8",
    "window": WINDOW,
    "budget_policy": "fixed-2x",
    "focus_slots": None,
    "focus_oversubscription": False,
    "prompt_tokens_details": False,
    "prefix_match_unit": None,
    "backlog_fix": False,
    "harness_fixes": False,
    "spare_sequences": SEQ_SPARE,
    "density_blocks": None,
    # Three-block run (29 September): the blocks settings and an explicit graph capture size.
    "blocks": None,
    "graph_capture": None,
    "gate_margin": MARGIN,
}
# Opt-in harness features (30 September): absent from ``requested`` / ``components`` unless requested and on.
OPTIONAL_COMPONENTS = {"parallel_samples": None, "action_plans": False, "completion_cap": None, "level_token_allowance": None,
                       "mtp_tokens": None, "expert_int3": None, "priority": None, "drain_half": None,
                       "dfr_harness": None}
# 3-bit routed experts (30 September; docs/validation-20260930/expert-3bit.md): the group sizes with a built overlay.
EXPERT_INT3_GROUPS = (16,)
# MTP-1 serving (30 September; spec-decode.md section 5, the arena-spec-sweep-p25-mtp1-v3 pilot on the prune25 pool): the
# draft stack, the decode-graph ladder in streams (each x (k + 1) verify tokens) and the fixed blocks per draft token.
MTP_DEPTHS = (1,)
MTP_KERNELS = {"mamba_ssm_cache_dtype": "bfloat16", "draft_experts": "nvfp4", "draft_vocab": "arena16k"}
MTP_LADDER = (1, 2, 4, 8, 16, 24, 30, 32, 36, 40, 44)
MTP_FIXED_BLOCKS_PER_TOKEN = 4  # 4 GDN groups x one more state page per draft token


class ComponentUnavailable(ValueError):
    """A requested component whose code is not in this tree (not merged yet)."""


def _token_clock():
    from arena.clock import ActiveClock

    return "tokens" in getattr(ActiveClock, "BUDGET_CLOCKS", ())


def _phase6_plan():
    from arc_infra import dense_kernels_patch as dense

    for attr in ("PHASE6", "PHASE6_FUSED"):
        if hasattr(dense, attr):
            return attr, copy.deepcopy(getattr(dense, attr))
    return None, None


def _prune_spec(level):
    from arena.variants import EXPERT_PRUNE_VARIANTS

    variant = PRUNE_LEVELS.get(level)
    if variant is None or variant not in EXPERT_PRUNE_VARIANTS:
        return None
    return EXPERT_PRUNE_VARIANTS[variant]


def status():
    """``{component: {"available": bool, "note": str}}`` for this tree."""
    phase6_attr, _ = _phase6_plan()
    return {
        "completion_cap": {"available": True, "note": "output cap and compaction reserve"},
        "level_token_allowance": {"available": True, "note": "independent per-level gameplay token allowance"},
        "token_budgets": {"available": _token_clock(),
                          "note": "token-clock scheduler (arena.clock budget_clock 'tokens')"
                                  + ("" if _token_clock() else ": not merged")},
        "serving_tweaks": {"available": True, "note": "ple_uva + embed_host (hero-uva arm); not yet smoked on MTP-0"},
        "kernels_phase6": {"available": phase6_attr is not None,
                           "note": f"arc_infra.dense_kernels_patch.{phase6_attr}" if phase6_attr
                           else "phase-6 dense-kernel plan: not merged"},
        "kv8_prefill": {"available": True, "note": "serving_kernels.kv8_prefill 'bf16' (arc_kv8_prefill); int8 KV"},
        "expert_prune": {"available": True, "note": f"levels {sorted(k for k in PRUNE_LEVELS if _prune_spec(k))}"},
        "gptq_arc": {"available": True, "note": "GPTQ-ARC overlay (Modal assets volume only)"},
        "topk_sampler": {"available": True, "note": "keep or drop the top-k-first sampler"},
        "stall_dump": {"available": True, "note": "stall-only worker dump; needs expert_prune"},
        "prefix_cache": {"available": True, "note": "arena.prefix_cache_patch (align fix, keep, demote); needs MTP-0; "
                                                    "GPU smoke prefix-cache-smoke"},
        "request_end_tail": {"available": True, "note": "prefix cache v3 (b): a finished request's exact end is a "
                                                        "prefix-cache entry (arena.prefix_tail_runtime); needs "
                                                        "prefix_cache and MTP-0"},
        "keep_alive": {"available": True, "note": "prefix cache v3 (a): max_tokens=1 pings keep the next parked games' "
                                                  "prefixes warm (arena.keep_alive); plain gate only"},
        "live_progress": {"available": True, "note": "progress.json + score peek, echoed by the bootstrap"},
        "compaction_thinking": {"available": True, "note": "Duck compaction summary mode (arena.compaction)"},
        "reasoning_history": {"available": True,
                              "note": "Duck history reasoning re-sent verbatim (arc_infra.config.reasoning_history)"},
        "kv_cache": {"available": True, "note": "int8 or int4 QSA main KV (serving_kernels.kv_cache_4bit, MTP-0)"},
        "window": {"available": True, "note": "64K or 128K windows"},
        "budget_policy": {"available": True, "note": "fixed-2x or the continuous ranking (arena.rank)"},
        "focus_slots": {"available": True, "note": "explicit focus target (default: the most full windows that fit)"},
        "focus_oversubscription": {"available": True,
                                   "note": "more focus games than full windows; the exact KV gate provides safety"},
        "prompt_tokens_details": {"available": True, "note": "--enable-prompt-tokens-details (logging only)"},
        "backlog_fix": {"available": True, "note": "backlog_key progress-fruitless + level0_order tokens"},
        "prefix_match_unit": {"available": True, "note": "--prefix-match-unit N (pending its warm/cold check)"},
        "harness_fixes": {"available": True, "note": "hero-v2 audit harness fixes (arena.harness_fixes)"},
        "spare_sequences": {"available": True, "note": "max_num_seqs = focus + spare (2 or more)"},
        "density_blocks": {"available": True, "note": "the replay-validated served pool of an oversubscribed target"},
        "blocks": {"available": True, "note": "three-block schedule settings (arena.blocks; budget_policy blocks)"},
        "graph_capture": {"available": True, "note": "explicit --max-cudagraph-capture-size"},
        "gate_margin": {"available": True, "note": "the KV gate cap as a share of the served blocks (kv_margin)"},
        "parallel_samples": {"available": True, "note": "k samples per gameplay request on deep games "
                                                        "(arena.parallel_samples); spare gate capacity only"},
        "action_plans": {"available": True, "note": "run_plan in the Duck sandbox + one prompt paragraph "
                                                    "(arena.action_plans)"},
        "mtp_tokens": {"available": True, "note": "MTP-1 speculative draft on int4 KV (spec-decode.md section 5)"},
        "expert_int3": {"available": True, "note": "int3 g16 routed experts on Humming (expert-3bit.md; Modal only)"},
        "priority": {"available": True, "note": "priority-gate settings (arena.priority; budget_policy priority)"},
        "drain_half": {"available": True, "note": "dfranzen's drop-half context drain in place of the summary "
                                                  "compaction (arena.context_drain)"},
        "dfr_harness": {"available": True, "note": "ported dfranzen harness features in the corrected Duck agent "
                                                   "(arena.dfr_harness)"},
    }


def normalize(components=None):
    """Defaults overlaid with ``components``; values validated (no availability check)."""
    result = dict(DEFAULT_COMPONENTS)
    for key, value in (components or {}).items():
        if key not in DEFAULT_COMPONENTS and key not in OPTIONAL_COMPONENTS:
            raise ValueError(f"Unknown hero-v2 component {key!r}; known: "
                             f"{sorted([*DEFAULT_COMPONENTS, *OPTIONAL_COMPONENTS])}")
        result[key] = value
    for key in ("completion_cap", "level_token_allowance"):
        value = result.get(key)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{key} must be a positive integer")
    if result.get("completion_cap", 16384) >= result["window"] // 2:
        raise ValueError("completion cap must leave room for input")
    if "parallel_samples" in result:
        from .parallel_samples import settings as parallel_settings

        spec = parallel_settings(result["parallel_samples"])
        if spec is None:
            del result["parallel_samples"]  # off: not recorded (earlier builds stay byte-for-byte)
        else:
            result["parallel_samples"] = spec
    if "action_plans" in result:
        from .action_plans import settings as plans_settings

        if plans_settings(result["action_plans"]):
            result["action_plans"] = True
        else:
            del result["action_plans"]
    if "drain_half" in result:
        from .context_drain import settings as drain_settings

        spec = drain_settings(result["drain_half"])
        if spec is None:
            del result["drain_half"]  # off: not recorded (earlier builds stay byte-for-byte)
        else:
            result["drain_half"] = spec
    if "dfr_harness" in result:
        from .dfr_harness import settings as dfr_settings

        block = dfr_settings(result["dfr_harness"])  # validates the flags
        if block is None:
            del result["dfr_harness"]  # off: not recorded (earlier builds stay byte-for-byte)
        else:
            result["dfr_harness"] = block
    if "mtp_tokens" in result:
        depth = result["mtp_tokens"]
        if depth in (None, 0):
            del result["mtp_tokens"]  # MTP-0: not recorded (earlier builds stay byte-for-byte)
        elif type(depth) is not int or depth not in MTP_DEPTHS:
            raise ValueError(f"mtp_tokens is 0 or one of {MTP_DEPTHS} (the measured MTP-1 pilot)")
        elif result["kv_cache"] != "int4" or result.get("parallel_samples") is not None:
            raise ValueError("mtp_tokens is validated on int4 KV without parallel samples")
    if "expert_int3" in result:
        group = result["expert_int3"]
        if group is None:
            del result["expert_int3"]  # off: not recorded (earlier builds stay byte-for-byte)
        elif type(group) is not int or group not in EXPERT_INT3_GROUPS:
            raise ValueError(f"expert_int3 is null or one of {EXPERT_INT3_GROUPS} (the built int3 overlays)")
        elif result["gptq_arc"] is not True:
            raise ValueError("expert_int3 re-rounds the GPTQ-ARC v1 experts: it needs gptq_arc")
    for key in ("token_budgets", "serving_tweaks", "kernels_phase6", "kv8_prefill", "gptq_arc", "topk_sampler", "prefix_cache",
                "live_progress", "focus_oversubscription", "prompt_tokens_details", "backlog_fix", "request_end_tail"):
        if type(result[key]) is not bool:
            raise ValueError(f"hero-v2 component {key} is on/off (true/false)")
    if result["request_end_tail"] and not result["prefix_cache"]:
        raise ValueError("request_end_tail extends prefix cache v2: it needs prefix_cache")
    from arena.keep_alive import settings as keep_alive_settings

    keep_alive_settings(result["keep_alive"])  # validates
    from arc_infra.config import REASONING_HISTORY_MODES
    from arena.compaction import COMPACTION_THINKING

    if result["reasoning_history"] not in REASONING_HISTORY_MODES:
        raise ValueError(f"reasoning_history is one of {REASONING_HISTORY_MODES}")
    if result["compaction_thinking"] not in COMPACTION_THINKING:
        raise ValueError(f"compaction_thinking is one of {COMPACTION_THINKING}")
    from arena.harness_fixes import settings as harness_fixes_settings

    fixes = result["harness_fixes"]
    if not (type(fixes) is bool or isinstance(fixes, list)):
        raise ValueError("harness_fixes is on/off (true/false) or a list of sub-fixes (arena.harness_fixes.FIXES)")
    if fixes == []:
        result["harness_fixes"] = False
    else:
        harness_fixes_settings(fixes)  # validates the names
    if result["expert_prune"] not in (None, False, *PRUNE_LEVELS):
        raise ValueError(f"expert_prune is false or one of {sorted(PRUNE_LEVELS)}")
    if result["expert_prune"] is False:
        result["expert_prune"] = None
    dump = result["stall_dump"]
    if dump is False:
        result["stall_dump"] = dump = None
    if dump is not None and (type(dump) is not int or not 30 <= dump <= 240):
        raise ValueError("stall_dump is false or 30..240 seconds")
    if dump is not None and result["expert_prune"] is None:
        raise ValueError("stall_dump rides on the expert-prune overlay: switch expert_prune on or stall_dump off")
    if result["kv_cache"] not in KV_MODES:
        raise ValueError(f"kv_cache is one of {sorted(KV_MODES)}")
    window = result["window"]
    if window not in WINDOWS and not (type(window) is int and 40960 <= window <= WINDOW_MAX and window % WINDOW_STEP == 0):
        raise ValueError(f"window is one of {WINDOWS} or a multiple of {WINDOW_STEP} from 40960 to {WINDOW_MAX}")
    margin = result["gate_margin"]
    if type(margin) not in (int, float) or not 0.5 <= margin <= 0.98:
        raise ValueError("gate_margin is the KV gate's share of the served blocks, 0.5..0.98")
    graphs = result["graph_capture"]
    if graphs is not None and (type(graphs) is not int or not SEQ_CAP < graphs <= 256):
        raise ValueError(f"graph_capture is null or an integer above {SEQ_CAP} (at most 256)")
    if (result["blocks"] is not None) != (result["budget_policy"] == "blocks"):
        raise ValueError("blocks settings go with budget_policy 'blocks'")
    if result["blocks"] is not None and not isinstance(result["blocks"], dict):
        raise ValueError("blocks is a dict of arena.blocks settings")
    if result["budget_policy"] not in BUDGET_POLICIES:
        raise ValueError(f"budget_policy is one of {BUDGET_POLICIES}")
    if result["budget_policy"] == "priority":
        # Priority gate (1 October): recorded with its validated settings; earlier builds never carry the key.
        from .priority import settings as priority_settings

        raw = result.get("priority")
        if raw is not None and not isinstance(raw, dict):
            raise ValueError("priority is a dict of arena.priority settings")
        result["priority"] = priority_settings(raw)
    elif result.get("priority") is not None:
        raise ValueError("priority settings go with budget_policy 'priority'")
    else:
        result.pop("priority", None)
    focus = result["focus_slots"]
    if focus is not None and (type(focus) is not int or not FOCUS_RANGE[0] <= focus <= FOCUS_RANGE[1]):
        raise ValueError(f"focus_slots is null or an integer in {FOCUS_RANGE[0]}..{FOCUS_RANGE[1]}")
    if result["focus_oversubscription"] and focus is None:
        raise ValueError("focus_oversubscription needs an explicit focus_slots target")
    spare = result["spare_sequences"]
    if type(spare) is not int or not SPARE_RANGE[0] <= spare <= SPARE_RANGE[1]:
        raise ValueError(f"spare_sequences is an integer in {SPARE_RANGE[0]}..{SPARE_RANGE[1]}")
    if spare != SEQ_SPARE and focus is None:
        raise ValueError("spare_sequences applies to an explicit focus_slots target")
    if focus is not None and focus + spare > SEQ_LIMIT:
        raise ValueError(f"focus_slots + spare_sequences is at most {SEQ_LIMIT} (the served sequence limit)")
    density = result["density_blocks"]
    if density is not None and (type(density) is not int or density < WINDOW_BLOCKS
                                or not result["focus_oversubscription"]):
        raise ValueError("density_blocks is the validated served pool (blocks) of an oversubscribed focus target")
    unit = result["prefix_match_unit"]
    if unit is not None:
        block = KV_MODES[result["kv_cache"]]["block_tokens"]
        if type(unit) is not int or not 16 <= unit < block or block % unit or unit % 16:
            raise ValueError(f"prefix_match_unit is null or a multiple of 16 dividing the {block}-token KV block")
    return result


def cost_blocks(kv, window, mtp=0):
    """Blocks the gate charges one full-window request (window + 256 tokens, + ``mtp`` draft tokens of lookahead) on the
    ``kv`` layout; MTP-k adds 4 fixed state blocks per draft token."""
    block = KV_MODES[kv]["block_tokens"]
    tokens = -(-(int(window) + FRESH_OVERHEAD + mtp) // COST_ROW_TOKENS) * COST_ROW_TOKENS
    return -(-tokens // block) + FIXED_BLOCKS + MTP_FIXED_BLOCKS_PER_TOKEN * mtp


def block_bytes(kv, mtp=0):
    """One block's bytes; with an MTP draft every block id holds a 13th QSA-layer page (the MTP layer)."""
    mode = KV_MODES[kv]
    return (QSA_LAYERS + (1 if mtp else 0)) * mode["block_tokens"] * (mode["bytes_per_token"] + INDEXER_BYTES)


def served_layout(components, served_blocks=None):
    """Projected (or measured) served layout and the stream count for ``components`` (normalized).

    ``served_blocks`` is the measured int8 pool (the hero-v2 smoke); with ``kv_cache`` "int4" its bytes are re-cut into
    int4 blocks (the KV dtype changes no weight or activation, kv-cache-4bit.md section 5.3)."""
    if served_blocks is not None:
        if type(served_blocks) is not int or served_blocks < WINDOW_BLOCKS:
            raise ValueError("served_blocks is the measured block count of the served pool")
        blocks, source = served_blocks, "measured (hero-v2 smoke)"
    else:
        blocks, parts = HERO_BLOCKS, [f"hero smoke {HERO_BLOCKS}"]
        level = components["expert_prune"]
        if level is not None:
            # The prune GPU check's served pools (with the PLE offload process and the GPU embedding).
            blocks = _prune_spec(level)["served_blocks"]
            parts = [f"{level} served {blocks}"]
        if components["serving_tweaks"]:
            blocks += UVA_BLOCKS
            parts.append(f"+{UVA_BLOCKS} host embedding (projected)")
        source = "projected: " + ", ".join(parts)
    kv, window = components["kv_cache"], components["window"]
    mtp = components.get("mtp_tokens") or 0
    int8_blocks = blocks
    if kv != "int8":
        blocks = int8_blocks * block_bytes("int8") // block_bytes(kv, mtp)
        source += f"; {kv}: the int8 pool's bytes in {KV_MODES[kv]['block_tokens']:,}-token blocks (projected)"
        if mtp:
            source += f" of {QSA_LAYERS + 1} QSA-layer pages (MTP-{mtp})"
    per_window = cost_blocks(kv, window, mtp)
    margin = components.get("gate_margin", MARGIN)
    capacity = int(margin * blocks)
    fit = capacity // per_window
    target = components["focus_slots"]
    oversubscribed = bool(target is not None and target > fit and components["focus_oversubscription"])
    if target is None:
        sequences = max(SEQ_FLOOR, min(fit, SEQ_CAP))
        active = min(fit, sequences)
        rule = ("active_limit = min(floor(0.9 x blocks / per-window blocks), max_num_seqs); "
                "max_num_seqs = clamp(windows, 25, 32)")
    else:
        if target > fit + 1 and not components["focus_oversubscription"]:
            raise ValueError(f"focus_slots {target} is {target - fit} more than the {fit} full {window}-token windows "
                             "the projected layout fits; ask for focus_oversubscription or a smaller target")
        active = target
        spare = components.get("spare_sequences", SEQ_SPARE)
        sequences = max(SEQ_FLOOR, target + spare)
        rule = (f"explicit focus target {target}; max_num_seqs = max(25, focus + {spare}); "
                + ("oversubscribed: the exact KV gate holds the requests the pool cannot take" if oversubscribed
                   else "the trial runs min(target, the served full-window fit) (focus_fit_fallback)"))
    capture = -(-sequences // 8) * 8 if sequences > SEQ_CAP else None
    if components.get("graph_capture") is not None:
        if components["graph_capture"] < sequences * (mtp + 1):
            raise ValueError(f"graph_capture {components['graph_capture']} is below max_num_seqs {sequences} x "
                             f"{mtp + 1} tokens per step")
        capture = components["graph_capture"]
    elif mtp:
        raise ValueError("mtp_tokens needs an explicit graph_capture of (k + 1) x max_num_seqs")
    density_blocks = components.get("density_blocks")
    density_capacity = int(margin * density_blocks) if density_blocks is not None else capacity
    return {"density_blocks": density_blocks, "density_gate_capacity": density_capacity, "blocks": blocks, "source": source, "gate_capacity": capacity, "window": window, "kv_cache": kv,
            "block_tokens": KV_MODES[kv]["block_tokens"], "int8_blocks": int8_blocks,
            "blocks_per_window": per_window, "windows_fit": fit, "max_num_seqs": sequences,
            "active_limit": active, "focus_target": target, "oversubscribed": oversubscribed,
            "graph_capture_size": capture, "rule": rule,
            **({"mtp_tokens": mtp, "fixed_blocks": FIXED_BLOCKS + MTP_FIXED_BLOCKS_PER_TOKEN * mtp} if mtp else {})}


def compose(hero, components=None, *, served_blocks=None):
    """The hero-v2 study from the hero study (``apply_variant(control, BASE)``) and the selected components."""
    if (hero.get("arena_variant") or {}).get("name") != BASE:
        raise ValueError(f"hero-v2 composes on the {BASE} study")
    if hero["server"].get("mtp_tokens") != 0 or hero["windows"] != [WINDOW] or "expert_prune" in hero["server"]:
        raise ValueError("hero-v2 needs the MTP-0 64K hero serving without pruning")
    components = dict(components or {})
    # Optional {component: why} recorded with the component (e.g. why one is switched off for a run).
    reasons = components.pop("reasons", None) or {}
    if not isinstance(reasons, dict) or set(reasons) - set(DEFAULT_COMPONENTS) - set(OPTIONAL_COMPONENTS) or not all(
            isinstance(v, str) and v for v in reasons.values()):
        raise ValueError("hero-v2 reasons is {component: non-empty text}")
    selected = normalize(components)
    if set(reasons) & set(OPTIONAL_COMPONENTS) - set(selected):
        raise ValueError("hero-v2 reasons name an optional component that is off (it is not recorded)")
    available = status()
    wanted = [k for k, v in selected.items() if v not in (None, False)]
    missing = [k for k in wanted if not available[k]["available"]]
    if missing:
        raise ComponentUnavailable("hero-v2 components not in this tree: "
                                   + "; ".join(f"{k} ({available[k]['note']})" for k in missing))
    result = copy.deepcopy(hero)
    server = result["server"]
    kernels = server["serving_kernels"]
    changed = {}
    record = {}

    def note(key, on, detail):
        record[key] = {"on": bool(on), **detail}

    if selected["token_budgets"]:
        from arena.scheduler import token_budget_settings

        before = result.get("budget_clock")
        result["budget_clock"] = "tokens"
        result.pop("cost_normalization", None)
        # The same block every token-clock build records (build_arena adds the rule text).
        result["token_budget"] = token_budget_settings()
        changed["budget_clock"] = {"before": before, "after": "tokens"}
        changed["token_budget"] = {"before": None, "after": copy.deepcopy(result["token_budget"])}
    note("token_budgets", selected["token_budgets"], {"budget_clock": result.get("budget_clock")})

    for key in ("completion_cap", "level_token_allowance"):
        if selected.get(key) is not None:
            changed[key] = {"before": result.get(key), "after": selected[key]}
            result[key] = selected[key]
            note(key, True, {"value": selected[key]})

    before_kernels = copy.deepcopy(kernels)
    if selected["serving_tweaks"]:
        kernels.update(ple_uva=True, embed_host=True)
    note("serving_tweaks", selected["serving_tweaks"], {"ple_uva": bool(kernels.get("ple_uva")),
                                                        "embed_host": bool(kernels.get("embed_host"))})
    if not selected["topk_sampler"]:
        kernels["topk_sampler"] = False
    note("topk_sampler", kernels.get("topk_sampler", False), {})
    if selected["kv8_prefill"]:
        if kernels.get("kv_cache_8bit") != "int8":
            raise ValueError("kv8_prefill needs the hero's int8 QSA KV cache")
        kernels["kv8_prefill"] = "bf16"
    note("kv8_prefill", selected["kv8_prefill"], {"mode": kernels.get("kv8_prefill")})
    kv = selected["kv_cache"]
    if kv == "int4":
        # The 4-bit QSA main KV (kv-cache-4bit.md): exclusive with the hero's int8 cache; --block-size 3136 (MTP-0).
        from arc_infra.serving_kernels_patch import KV4_BLOCK_TOKENS

        if kernels.get("kv_cache_8bit") != "int8" or server.get("mtp_tokens") != 0:
            raise ValueError("kv_cache int4 replaces the hero's int8 QSA cache on the MTP-0 serving")
        kernels.pop("kv_cache_8bit")
        kernels["kv_cache_4bit"] = "int4"
        if KV4_BLOCK_TOKENS["int4"] != KV_MODES["int4"]["block_tokens"]:
            raise ValueError("arena.hero_v2.KV_MODES and serving_kernels_patch.KV4_BLOCK_TOKENS disagree")
    note("kv_cache", kv != "int8", {"mode": kv, "block_tokens": KV_MODES[kv]["block_tokens"],
                                    "serving_kernels": "kv_cache_4bit int4" if kv == "int4" else "kv_cache_8bit int8"})
    depth = selected.get("mtp_tokens")
    if depth:
        # MTP-1 (30 September, spec-decode.md section 5): the pilot's draft stack; the 4-bit block follows the draft
        # length (serving_kernels_patch.kv4_block_tokens: 3,136 tokens still hold the MTP-1 conv state).
        from arc_infra.serving_kernels_patch import kv4_block_tokens

        if kv4_block_tokens(kv, depth) != KV_MODES[kv]["block_tokens"]:
            raise ValueError(f"MTP-{depth} changes the {kv} block size; only MTP-1 on 3,136-token blocks is measured")
        if any(kernels.get(k) not in (None, v) for k, v in MTP_KERNELS.items()):
            raise ValueError("the hero serving kernels disagree with the MTP draft stack")
        kernels.update(MTP_KERNELS)
    int3 = selected.get("expert_int3")
    if int3:
        # 3-bit routed experts (30 September, expert-3bit.md): the int3 g16 overlay (GPTQ-ARC v1 re-rounded inside the
        # NVFP4 container) on vLLM's Humming MoE, which re-encodes it as uint3 at load, bit-exact
        # (serving_kernels.expert_int3). Replaces Marlin W4A16; the MTP draft's own NvFp4 b12x experts are unchanged.
        if kernels.get("moe_backend") != "marlin":
            raise ValueError("expert_int3 replaces the hero's Marlin W4A16 routed experts")
        kernels.update(moe_backend="humming", expert_int3=int3)
    if kernels != before_kernels:
        from arc_infra.serving_kernels_patch import validate as validate_serving_kernels

        validate_serving_kernels(kernels)
        changed["server.serving_kernels"] = {"before": before_kernels, "after": copy.deepcopy(kernels)}

    if selected["kernels_phase6"]:
        attr, plan = _phase6_plan()
        changed["server.dense_kernels"] = {"before": copy.deepcopy(server["dense_kernels"]), "after": plan}
        server["dense_kernels"] = plan
        note("kernels_phase6", True, {"plan": f"arc_infra.dense_kernels_patch.{attr}"})
    else:
        note("kernels_phase6", False, {"plan": "phase 5 (the hero's)"})

    level = selected["expert_prune"]
    if level is not None:
        spec = _prune_spec(level)
        server["expert_prune"] = {"mask": spec["mask"], "sha256": spec["sha256"]}
        changed["server.expert_prune"] = {"before": None, "after": dict(server["expert_prune"])}
        note("expert_prune", True, {"level": level, "fraction": spec["fraction"], "mask": spec["mask"],
                                    "served_blocks_prune_check": spec["served_blocks"]})
    else:
        note("expert_prune", False, {})

    if selected["gptq_arc"] and int3:
        from arena.variants import EXPERT_OVERLAY_INT3

        overlay = dict(EXPERT_OVERLAY_INT3[int3])
        server["expert_overlay"] = overlay
        changed["server.expert_overlay"] = {"before": None, "after": dict(overlay)}
        note("gptq_arc", True, {"dataset": overlay["dataset"], "platform": "modal-only",
                                "format": f"int3 g{int3} re-rounded from GPTQ-ARC v1 (NVFP4 container)",
                                "composition": "overlay shards (symlink farm), then the prune filter"
                                if level is not None else "overlay shards only"})
        note("expert_int3", True, {"group": int3, "moe_backend": "humming", "dataset": overlay["dataset"],
                                   "manifest_sha256": overlay["manifest_sha256"], "check": overlay.get("check"),
                                   "draft_experts": kernels.get("draft_experts")})
    elif selected["gptq_arc"]:
        from arena.variants import EXPERT_OVERLAY_GPTQ_ARC

        if kernels.get("moe_backend") != "marlin":
            raise ValueError("gptq_arc is validated on the Marlin W4A16 experts (the hero's moe_backend)")
        server["expert_overlay"] = dict(EXPERT_OVERLAY_GPTQ_ARC)
        changed["server.expert_overlay"] = {"before": None, "after": dict(EXPERT_OVERLAY_GPTQ_ARC)}
        note("gptq_arc", True, {"dataset": EXPERT_OVERLAY_GPTQ_ARC["dataset"], "platform": "modal-only",
                                "composition": "overlay shards (symlink farm), then the prune filter"
                                if level is not None else "overlay shards only"})
    else:
        note("gptq_arc", False, {})

    note("stall_dump", selected["stall_dump"] is not None, {"seconds": selected["stall_dump"]})
    if selected["stall_dump"] is not None:
        result["stall_dump_seconds"] = selected["stall_dump"]  # applied to every trial by scripts/build_arena.py
    if selected["prefix_cache"]:
        from arena.variants import HERO_PREFIX_CACHE

        patches = dict(HERO_PREFIX_CACHE)
        if selected.get("request_end_tail"):
            patches["request_end_tail"] = True
        result["cache_experiment"]["prefix_cache"] = patches
        changed["cache_experiment.prefix_cache"] = {"before": None, "after": dict(patches)}
        note("prefix_cache", True, dict(HERO_PREFIX_CACHE))
    else:
        note("prefix_cache", False, {"reason": "off by default until the prefix-cache GPU smoke passes"})
    if "request_end_tail" in selected:
        note("request_end_tail", bool(selected["request_end_tail"]),
             {"option": "cache_experiment.prefix_cache.request_end_tail"} if selected["request_end_tail"] else {})
    if "keep_alive" in selected:
        from arena.keep_alive import settings as keep_alive_settings

        keep = keep_alive_settings(selected["keep_alive"])
        if keep is not None:
            result["cache_experiment"]["keep_alive"] = keep
            changed["cache_experiment.keep_alive"] = {"before": None, "after": dict(keep)}
        note("keep_alive", keep is not None, dict(keep) if keep is not None else {})

    if selected["live_progress"]:
        result["live_progress"] = dict(LIVE_PROGRESS)
        changed["live_progress"] = {"before": None, "after": dict(LIVE_PROGRESS)}
    note("live_progress", selected["live_progress"], dict(LIVE_PROGRESS) if selected["live_progress"] else {})

    mode = selected["compaction_thinking"]
    result["compaction_thinking"] = mode  # scripts/build_arena.py writes the default "gameplay" as no key
    changed["compaction_thinking"] = {"before": hero.get("compaction_thinking", "gameplay"), "after": mode}
    note("compaction_thinking", mode == "prefilled-empty", {"mode": mode})
    history = selected["reasoning_history"]
    result["reasoning_history"] = history  # scripts/build_arena.py writes "normalized" as no key
    changed["reasoning_history"] = {"before": hero.get("reasoning_history", "normalized"), "after": history}
    note("reasoning_history", history == "verbatim", {"mode": history})
    if selected["harness_fixes"]:
        from arena.harness_fixes import settings as harness_fixes_settings

        block = harness_fixes_settings(selected["harness_fixes"])
        result["harness_fixes"] = block  # arena.trial passes it to every Duck agent
        changed["harness_fixes"] = {"before": None, "after": copy.deepcopy(block)}
        note("harness_fixes", True, copy.deepcopy(block))
    else:
        note("harness_fixes", False, {})

    window = selected["window"]
    if window != WINDOW:
        result["windows"] = [window]
        changed["windows"] = {"before": list(hero["windows"]), "after": [window]}
    note("window", window != WINDOW, {"tokens": window})
    if selected["prompt_tokens_details"]:
        server["prompt_tokens_details"] = True
        changed["server.prompt_tokens_details"] = {"before": None, "after": True}
    note("prompt_tokens_details", selected["prompt_tokens_details"], {"flag": "--enable-prompt-tokens-details"})
    unit = selected["prefix_match_unit"]
    if unit is not None:
        server["prefix_match_unit"] = unit
        changed["server.prefix_match_unit"] = {"before": None, "after": unit}
    note("prefix_match_unit", unit is not None, {"tokens": unit} if unit is not None else {
        "reason": "off: pending its warm/cold correctness check with prefix cache v2 and the user's go"})

    layout = served_layout(selected, served_blocks)
    before = {"active_limit": result["active_limit"], "max_num_seqs": server["max_num_seqs"],
              "generation_slots": result["generation_slots"]}
    result["active_limit"] = layout["active_limit"]
    server["max_num_seqs"] = layout["max_num_seqs"]
    result["generation_slots"] = layout["max_num_seqs"]
    for key, value in (("active_limit", layout["active_limit"]), ("server.max_num_seqs", layout["max_num_seqs"]),
                       ("generation_slots", layout["max_num_seqs"])):
        old = before[key.split(".")[-1]]
        if old != value:
            changed[key] = {"before": old, "after": value}
    if layout["graph_capture_size"] is not None:
        # More sequences than the pinned 32-token capture: MTP-0 decodes one token per sequence, so a decode batch of
        # every running sequence needs graphs up to max_num_seqs (the next multiple of 8).
        changed["server.max_cudagraph_capture_size"] = {"before": server.get("max_cudagraph_capture_size", SEQ_CAP),
                                                        "after": layout["graph_capture_size"]}
        server["max_cudagraph_capture_size"] = layout["graph_capture_size"]
    if depth:
        # MTP-k decodes k + 1 tokens per sequence per step: the pilot's ladder (streams) x (k + 1), up to max_num_seqs.
        streams = [n for n in MTP_LADDER if n < layout["max_num_seqs"]] + [layout["max_num_seqs"]]
        sizes = [n * (depth + 1) for n in streams]
        if sizes[-1] != server["max_cudagraph_capture_size"]:
            raise ValueError(f"MTP-{depth}: graph_capture must be {sizes[-1]} ({depth + 1} x max_num_seqs "
                             f"{layout['max_num_seqs']})")
        changed["server.mtp_tokens"] = {"before": server.get("mtp_tokens"), "after": depth}
        changed["server.cudagraph_capture_sizes"] = {"before": server.get("cudagraph_capture_sizes"), "after": sizes}
        server["mtp_tokens"] = depth
        server["cudagraph_capture_sizes"] = sizes
        note("mtp_tokens", True, {
            "tokens": depth, "serving_kernels": dict(MTP_KERNELS), "cudagraph_capture_sizes": sizes,
            "fixed_blocks_per_request": layout["fixed_blocks"],
            "evidence": "docs/validation-20260930/spec-decode.md section 5 (arena-spec-sweep-p25-mtp1-v3: pool 1,438 "
                        "blocks at 44 sequences / graphs to 88, acceptance 0.78-0.82, MTP-1/MTP-0 1.33-1.34x "
                        "generated tok/s at 24-40 streams, verify-path drift within the MTP-0 noise)",
            "prefix_cache": "keep/demote switch themselves off under speculation (the chunk-alignment fix stays)",
            "gate": "the served layout's lookahead (k) charges the draft tokens; the Duck client's lookahead is k"})
    if layout["oversubscribed"]:
        # More focus games than full windows fit: no fit check and no fallback; the exact KV gate
        # holds a request until the pool has room for its prompt + completion cap, so no request can be preempted.
        result["require_focus_windows_fit"] = False
        result["focus_oversubscription"] = {
            "focus_slots": layout["active_limit"], "full_windows_fit_projected": layout["windows_fit"],
            "blocks_per_full_window": layout["blocks_per_window"], "gate_capacity_projected": layout["gate_capacity"],
            # The startup keeps this density on the served pool: focus = min(target, floor(served gate blocks / it)),
            # at least min_full_windows full windows (arena.capacity.competition_focus_fit).
            # density_blocks (competition run B, 29 September): the replay-validated pool, e.g. 32 on >= 1,268 blocks.
            # Rounded down, so the validated pool itself always allows the full target.
            "gate_blocks_per_stream": (round(layout["gate_capacity"] / layout["active_limit"], 3)
                                       if layout["density_blocks"] is None else
                                       (layout["density_gate_capacity"] * 1000 // layout["active_limit"]) / 1000),
            "min_full_windows": FIT_FALLBACK_MIN, "min_slots": FIT_FALLBACK_MIN,
            "safety": "the exact KV gate (arena.gate): every request reserves cost(prompt + completion cap) blocks and "
                      "the reservations never exceed 0.9 x the served blocks, so requests wait at the gate and none "
                      "can be preempted"}
        changed["focus_oversubscription"] = {"before": None, "after": copy.deepcopy(result["focus_oversubscription"])}
    else:
        result["require_focus_windows_fit"] = True
        result["focus_fit_fallback"] = {"min_slots": FIT_FALLBACK_MIN}
        changed["focus_fit_fallback"] = {"before": None, "after": {"min_slots": FIT_FALLBACK_MIN}}
    note("focus_slots", layout["focus_target"] is not None, {
        "target": layout["focus_target"], "active_limit": layout["active_limit"],
        "windows_fit_projected": layout["windows_fit"]})
    note("focus_oversubscription", layout["oversubscribed"], {"requested": selected["focus_oversubscription"]})

    if selected["backlog_fix"]:
        if result.get("level_budget_policy") != "fixed-2x" or selected["budget_policy"] != "fixed-2x":
            raise ValueError("backlog_fix changes the fixed-2x readmissions backlog")
        result["level0_order"] = "tokens"
        result["backlog_key"] = "progress-fruitless"
        changed["level0_order"] = {"before": hero.get("level0_order", "roster"), "after": "tokens"}
        changed["backlog_key"] = {"before": None, "after": "progress-fruitless"}
    note("backlog_fix", selected["backlog_fix"], {"level0_order": result.get("level0_order", "roster"),
                                                  "backlog_key": result.get("backlog_key")})

    spec = selected.get("parallel_samples")
    if spec is not None:
        # parallel_samples (30 September): siblings need server sequences beyond the focus streams + spares, and every
        # decode batch of up to max_num_seqs needs its CUDA graph (MTP-0: one token per sequence per step). The
        # generation slots (the gate's regular requests) stay the focus layout's, so regular requests run exactly as
        # without the feature; the gate gives the siblings the difference (KVGate background_slots, arena.trial).
        from .parallel_samples import graph_capture_size

        if not selected["prefix_cache"] or selected["budget_policy"] not in ("fixed-2x", "blocks"):
            raise ValueError("parallel_samples needs prefix_cache (siblings share the pinged prefix) and the "
                             "plain gate (budget_policy fixed-2x or blocks)")
        if spec["max_num_seqs"] < server["max_num_seqs"]:
            raise ValueError(f"parallel_samples.max_num_seqs {spec['max_num_seqs']} is below the "
                             f"{server['max_num_seqs']} sequences the focus streams already use")
        capture = graph_capture_size(spec["max_num_seqs"])
        for key, old, new in (("server.max_num_seqs", before["max_num_seqs"], spec["max_num_seqs"]),
                              ("server.max_cudagraph_capture_size", hero["server"].get("max_cudagraph_capture_size",
                                                                                          SEQ_CAP), capture)):
            if old != new:
                changed[key] = {"before": old, "after": new}
            else:
                changed.pop(key, None)
        server["max_num_seqs"] = spec["max_num_seqs"]
        server["max_cudagraph_capture_size"] = capture
        result["parallel_samples"] = copy.deepcopy(spec)
        changed["parallel_samples"] = {"before": None, "after": copy.deepcopy(spec)}
        note("parallel_samples", True, {**spec, "max_cudagraph_capture_size": capture,
                                        "generation_slots": result["generation_slots"],
                                        "background_slots": spec["max_num_seqs"] - result["generation_slots"]})
    if selected.get("drain_half"):
        block = copy.deepcopy(selected["drain_half"])
        result["drain_half"] = block  # arena.trial passes it to every Duck agent
        changed["drain_half"] = {"before": None, "after": copy.deepcopy(block)}
        note("drain_half", True, {**copy.deepcopy(block), "replaces": "summary compaction (arena.compaction)",
                                  "source": "dfranzen ARC-AGI-3 milestone-2 notebook (Apache-2.0)"})
    if selected.get("action_plans"):
        result["action_plans"] = True
        changed["action_plans"] = {"before": None, "after": True}
        note("action_plans", True, {"helper": "run_plan (duck_transfer.sandbox.with_plans)",
                                    "prompt": "arena.action_plans.PROMPT"})
    if selected.get("dfr_harness"):
        from .dfr_harness import describe as dfr_describe

        block = copy.deepcopy(selected["dfr_harness"])
        result["dfr_harness"] = block  # arena.trial passes it to every Duck agent
        changed["dfr_harness"] = {"before": None, "after": copy.deepcopy(block)}
        note("dfr_harness", True, dfr_describe(block))

    policy = selected["budget_policy"]
    if policy == "blocks":
        # The three-block schedule: no opening, no level budgets; a block is the
        # stream count.
        from .blocks import settings as blocks_settings

        if result.get("level_budget_policy") != "fixed-2x" or result.get("backlog_policy") != "readmissions" \
                or result.get("cache_admission") != "hybrid_blocks" or selected["backlog_fix"]:
            raise ValueError("budget_policy blocks replaces the hero's fixed-2x readmissions scheduler (no backlog_fix)")
        cfg = blocks_settings({**selected["blocks"], "slots": selected["blocks"].get("slots", layout["active_limit"])})
        if cfg["slots"] != layout["active_limit"]:
            raise ValueError(f"blocks need {cfg['slots']} streams; the focus target is {layout['active_limit']}")
        result["level_budget_policy"] = "blocks"
        result["blocks"] = {k: cfg[k] for k in ("wall_seconds", "blocks", "slots", "roster_games",
                                                "level0_budget_tokens", "tenure_unit", "tenure_tokens",
                                                "stagger_seconds")}
        if cfg.get("level_token_allowance") is not None:
            result["blocks"]["level_token_allowance"] = cfg["level_token_allowance"]
        if cfg["partition"] != "even":
            result["blocks"]["partition"] = cfg["partition"]  # recorded only when requested: other builds unchanged
        changed["level_budget_policy"] = {"before": "fixed-2x", "after": "blocks"}
        changed["blocks"] = {"before": None, "after": dict(result["blocks"])}
    if policy == "priority":
        # dfranzen's priority gate: the focus streams are its slots; no opening, no
        # level budgets, no retirement; handovers only at context evictions.
        if result.get("level_budget_policy") != "fixed-2x" or result.get("backlog_policy") != "readmissions" \
                or result.get("cache_admission") != "hybrid_blocks" or selected["backlog_fix"]:
            raise ValueError("budget_policy priority replaces the hero's fixed-2x readmissions scheduler "
                             "(no backlog_fix)")
        result["level_budget_policy"] = "priority"
        result["priority"] = dict(selected["priority"])
        before_rule = result.get("level_budget_rule")
        result["level_budget_rule"] = (
            f"priority gate (dfranzen's milestone-2 scheduler, arena.priority): {layout['active_limit']} "
            "slots; never-run games take freed slots in roster order (a first window each); a game keeps its slot until "
            "its next context eviction (summary compaction or drain), then it is re-scored (level x (25/(25+actions on "
            "level))^2, score-normalized, plus the remaining-levels tail 8/7/5/0 faded over the last 20% of the wall, "
            "times the 0.25/0.75 action/token half-life chance at 115 actions / 62,000 tokens on the level) and the best "
            "waiting game takes the slot; no opening, no level budgets, no retirement")
        changed["level_budget_rule"] = {"before": before_rule, "after": result["level_budget_rule"]}
        changed["level_budget_policy"] = {"before": "fixed-2x", "after": "priority"}
        changed["priority"] = {"before": None, "after": dict(result["priority"])}
    if policy == "rank":
        # The hero-v2-rank arm (arena.variants._rank_variant) on hero-v2's serving: no opening, no level budgets.
        from .rank import DEFAULTS as RANK

        if result.get("level_budget_policy") != "fixed-2x" or result.get("backlog_policy") != "readmissions" \
                or result.get("cache_admission") != "hybrid_blocks":
            raise ValueError("budget_policy rank replaces the hero's fixed-2x readmissions scheduler")
        result["level_budget_policy"] = "rank"
        result["rank"] = {**RANK, "slots": layout["active_limit"], "window": window}
        changed["level_budget_policy"] = {"before": "fixed-2x", "after": "rank"}
        changed["rank"] = {"before": None, "after": dict(result["rank"])}
    note("budget_policy", policy != "fixed-2x", {"policy": policy, **(
        {"settings": dict(result["rank"])} if policy == "rank" else {}), **(
        {"settings": dict(result["blocks"])} if policy == "blocks" else {}), **(
        {"settings": dict(result["priority"]), "slots": layout["active_limit"]} if policy == "priority" else {})})
    # The three-block run's components are recorded when used (earlier builds' records stay byte for byte).
    if policy == "blocks":
        note("blocks", True, dict(result["blocks"]))
    if policy == "priority":
        note("priority", True, dict(result["priority"]))
    if selected["graph_capture"] is not None:
        note("graph_capture", True, {"size": selected["graph_capture"]})
    if selected["gate_margin"] != MARGIN:
        before = result.get("kv_margin")
        result["kv_margin"] = float(selected["gate_margin"])
        changed["kv_margin"] = {"before": before, "after": result["kv_margin"]}
        note("gate_margin", True, {"kv_margin": result["kv_margin"]})

    for key, text in reasons.items():
        record[key]["reason"] = text
    against_control = {**hero["arena_variant"]["changed"], **changed}
    if "server.mtp_tokens" in changed and "server.mtp_tokens" in hero["arena_variant"]["changed"]:
        # Against the paired control (MTP-3): its own depth before, this build's after.
        against_control["server.mtp_tokens"] = {
            "before": hero["arena_variant"]["changed"]["server.mtp_tokens"]["before"],
            "after": changed["server.mtp_tokens"]["after"]}
    if "windows" in changed and "windows" in hero["arena_variant"]["changed"]:
        # Against the paired control: back to its own window (128K) is no change; another window keeps its before.
        control_windows = hero["arena_variant"]["changed"]["windows"]["before"]
        if control_windows == [window]:
            against_control.pop("windows")
        else:
            against_control["windows"] = {"before": control_windows, "after": [window]}
    on = sorted(k for k, v in record.items() if v["on"])
    requested = {k: v for k, v in selected.items()
                 if k not in ("blocks", "graph_capture", "gate_margin") or v != DEFAULT_COMPONENTS[k]}
    result["hero_v2"] = {"components": record, "requested": requested, "on": on,
                         "off": sorted(k for k, v in record.items() if not v["on"]),
                         "layout": layout, "base": BASE, "wall_seconds_target": WALL_SECONDS}
    result["arena_variant"] = {
        "name": NAME,
        "description": (f"hero-v2: {BASE} plus {', '.join(on)}; "
                        f"{layout['active_limit']} focus games at {window // 1024}K {kv} KV ({layout['source']}), "
                        f"max_num_seqs {layout['max_num_seqs']}"),
        "changed": against_control,
        "checkpoint_and_serving_unchanged": False,
        "reference_variant": BASE, "reference_difference": changed,
        "reference_checkpoint_and_serving_unchanged": False,
        "base_reference": copy.deepcopy({k: hero["arena_variant"][k]
                                         for k in ("reference_variant", "reference_difference")}),
    }
    return result


# --- hero-v2 on a candidate checkpoint (Holo4-27B, 29 September) ------------------------------------------------
#
# "test Holo4-27B with the hero-v2 arena setup. Drop kernel-specific improvements that don't apply
# to this model, and fit as many streams as possible at 64K contexts" (NVFP4 weights, FP8 KV). ``compose_candidate``
# runs ``compose`` with every Flash-Next serving component off, keeps the arena-side components (and prefix cache v2,
# which patches generic vLLM Mamba code), then moves the study onto the candidate's reviewed checkpoint and serving
# (arena.candidate_profiles, arena.candidate_serving): the hero's Flash-only server keys are removed and recorded with
# why each does not apply. The stream count follows the candidate's projected (or measured) served layout.

CANDIDATE_NAME = "holo4-27b-hero-v2"
CANDIDATE_DEFAULTS = {
    "token_budgets": True,
    "stall_dump": STALL_DUMP_SECONDS,  # the standalone overlay (arc_infra.stall_dump_patch): no pruning needed
    "prefix_cache": True,
    "live_progress": True,
    "compaction_thinking": "prefilled-empty",
    "reasoning_history": "verbatim",
}
# hero-v2 serving that does not apply to a dense Qwen3.8-27B checkpoint on stock vLLM kernels (component or server key:
# why). Each is absent from the candidate's server block.
CANDIDATE_DROPPED = {
    "dense_kernels": "A+C dense kernels and the phase-5 mid-M tiles: Triton/int8 plans for Flash-Next's GEMM shapes "
                     "(hidden 2,560, QSA, 512 experts); Holo4's projections run the stock compressed-tensors kernels "
                     "(FP8 attention/GDN projections, NVFP4 W4A16 MLP)",
    "kernels_phase6": "the phase-6 fused split-K plan extends the Flash-Next phase-5 tiles (off in hero-v2 as well)",
    "kv_cache_8bit": "int8 QSA KV is a change to Flash-Next's QSA sparse-attention backend; Holo4 has plain full "
                     "attention, so the stock FP8 KV cache (--kv-cache-dtype fp8, the checkpoint's calibrated "
                     "k/v scales) on Triton attention",
    "moe_backend": "Marlin W4A16 routed experts: Holo4 is dense (no routed experts)",
    "expert_prune": "load-time expert pruning: no routed experts to prune",
    "gptq_arc": "the ARC-GPTQ overlay replaces Flash-Next routed-expert shards",
    "serving_tweaks": "ple_uva / embed_host move Flash-Next's per-layer-embedding table and input embedding to host "
                      "memory; Holo4 has no PLE",
    "topk_sampler": "the top-k-first sampler patch never runs on gameplay batches (hero-v2 behaviour check)",
    "hc_fusion": "hyper-connection fusion: a Flash-Next layer type",
    "draft_vocab": "the MTP draft's reduced vocabulary: Holo4 serves MTP-0",
    "mtp": "MTP-0: the config declares one MTP layer but the checkpoint ships no MTP weights",
    "stack_kernels": "async_h2d patches Flash-Next's PLE short-conv attention metadata (short_conv_attn.py)",
    "mamba_ssm_cache_dtype": "BF16 recurrent state (about +8% windows) is not drift-checked; off (fp32 state, the "
                             "checkpoint's mamba_ssm_dtype)",
}
# Projected served layouts at the hero's utilization (0.96) on the RTX PRO 6000, and the blocks one full 64K focus window
# needs (window + 256 tokens of attention blocks plus the fixed recurrent-state blocks, MTP-0 align mode).
CANDIDATE_LAYOUTS = {
    "holo4-27b-nvfp4": {
        # Measured by the candidate smoke v2 (arena-holo4-27b-smoke-v2-modal-20260929, FP8 KV on Triton attention at
        # 0.96): 66.81 GiB of KV in 1,396 blocks of 1,568 tokens (pages of 3,211,264 B: the GDN state page, which the
        # FP8 attention page is rounded up to; 16 layers per group). The projection before it was 1,360 blocks of 1,600.
        "blocks": 1396, "blocks_per_window": 49, "block_tokens": 1568,
        "source": "measured (Holo4 smoke v2: 66.81 GiB, 2,103,178 KV tokens)",
        "window_rule": "the served cost of a full focus window (65,536 + 256 tokens): ceil(65,792 / 1,568) = 43 "
                       "attention blocks + 3 GDN groups x 2 state blocks (MTP-0 align mode) = 49",
    },
}
CANDIDATE_FIT_FALLBACK_MIN = 14
IDENTITY_KEYS = ("candidate_profile", "served_model", "model_revision", "model_datasets", "model_sources",
                 "model_template_notes", "serving_profile", "startup_timeout_seconds", "observation_mode")
SERVER_KEYS = ("max_model_len", "prefix_caching", "mtp_tokens", "max_num_seqs", "gpu_memory_utilization",
               "max_num_batched_tokens")


def normalize_candidate(components=None):
    """``CANDIDATE_DEFAULTS`` overlaid with ``components``; a Flash-only component may only be switched off."""
    result = dict(CANDIDATE_DEFAULTS)
    for key, value in (components or {}).items():
        if key in CANDIDATE_DROPPED:
            if value not in (None, False):
                raise ComponentUnavailable(f"{key} does not apply to a candidate checkpoint: {CANDIDATE_DROPPED[key]}")
            continue
        if key not in CANDIDATE_DEFAULTS:
            raise ValueError(f"Unknown candidate hero-v2 component {key!r}; known: {sorted(CANDIDATE_DEFAULTS)}")
        result[key] = value
    for key in ("token_budgets", "prefix_cache", "live_progress"):
        if type(result[key]) is not bool:
            raise ValueError(f"hero-v2 component {key} is on/off (true/false)")
    from arc_infra.stall_dump_patch import validate as validate_stall_dump

    if result["stall_dump"] is False:
        result["stall_dump"] = None
    validate_stall_dump(result["stall_dump"])
    return result


def candidate_layout(profile, served_blocks=None, blocks_per_window=None):
    """The candidate's projected (or measured) served layout and its stream count at 64K."""
    if profile not in CANDIDATE_LAYOUTS:
        raise ValueError(f"No served-layout projection for candidate {profile!r}")
    spec = CANDIDATE_LAYOUTS[profile]
    window_blocks = spec["blocks_per_window"] if blocks_per_window is None else blocks_per_window
    if type(window_blocks) is not int or window_blocks <= 0:
        raise ValueError("blocks_per_window is the measured cost of one 64K focus window")
    if served_blocks is not None:
        if type(served_blocks) is not int or served_blocks < window_blocks:
            raise ValueError("served_blocks is the measured block count of the served pool")
        blocks, source = served_blocks, "measured (candidate hero-v2 smoke)"
    else:
        blocks, source = spec["blocks"], spec["source"]
    capacity = int(MARGIN * blocks)
    fit = capacity // window_blocks
    sequences = max(SEQ_FLOOR, min(fit, SEQ_CAP))
    return {"blocks": blocks, "source": source, "gate_capacity": capacity, "window": WINDOW,
            "blocks_per_window": window_blocks, "block_tokens": spec["block_tokens"],
            "window_rule": spec["window_rule"] if blocks_per_window is None else "measured cost table",
            "windows_fit": fit, "max_num_seqs": sequences, "active_limit": min(fit, sequences),
            "rule": f"active_limit = min(floor(0.9 x blocks / {window_blocks}), max_num_seqs); "
                    "max_num_seqs = clamp(windows, 25, 32)"}


def compose_candidate(hero, candidate, components=None, *, served_blocks=None, blocks_per_window=None):
    """hero-v2 on a candidate checkpoint: ``hero`` is the Flash hero study (``apply_variant(control, BASE)``),
    ``candidate`` the candidate's paired-30m config (``configs/arena_30m_<name>.json``)."""
    from .candidate_profiles import PROFILES

    name = candidate.get("candidate_profile")
    if name not in PROFILES:
        raise ValueError("compose_candidate needs a reviewed candidate config (candidate_profile)")
    profile = PROFILES[name]
    if (candidate.get("served_model"), candidate.get("model_revision")) != (profile.model, profile.revision):
        raise ValueError("The candidate config does not match its pinned checkpoint identity")
    if profile.mtp_tokens != 0 or profile.effort != hero["reasoning_effort"]:
        raise ValueError("hero-v2 on a candidate needs its MTP-0 serving and the hero's reasoning effort")
    components = dict(components or {})
    reasons = components.pop("reasons", None) or {}
    if not isinstance(reasons, dict) or not all(isinstance(v, str) and v for v in reasons.values()):
        raise ValueError("hero-v2 reasons is {component: non-empty text}")
    selected = normalize_candidate(components)
    # The Flash composition with every Flash-Next serving component off: the arena-side keys come out exactly as
    # hero-v2 builds them (token clock, live progress, compaction, history, prefix cache v2, focus-fit fallback).
    arena_side = ("token_budgets", "prefix_cache", "live_progress", "compaction_thinking", "reasoning_history")
    flash = {"serving_tweaks": False, "kernels_phase6": False, "expert_prune": False, "gptq_arc": False,
             "stall_dump": False, **{k: selected[k] for k in arena_side}}
    result = compose(hero, flash)
    record = {k: v for k, v in result["hero_v2"]["components"].items() if k in arena_side}
    if result.get("harness_fixes"):
        raise ValueError("hero-v2 on a candidate keeps the hero-v2 2 h run's harness (harness_fixes off)")
    record["harness_fixes"] = {"on": False, "reason": "off: the Holo4 arm keeps the hero-v2 2 h run's harness for "
                                                      "comparability"}
    for key in IDENTITY_KEYS:
        result.pop(key, None)
        if key in candidate:
            result[key] = copy.deepcopy(candidate[key])
    for key in ("max_model_len", "prefix_caching", "gpu_memory_utilization", "max_num_batched_tokens"):
        if candidate["server"][key] != hero["server"][key]:
            raise ValueError(f"The candidate config's server.{key} differs from the hero's")
    flash_server = result["server"]
    dropped_server = sorted(k for k in flash_server if k not in SERVER_KEYS)
    layout = candidate_layout(name, served_blocks, blocks_per_window)
    result["server"] = {k: flash_server[k] for k in SERVER_KEYS}
    result["server"]["max_num_seqs"] = layout["max_num_seqs"]
    result["active_limit"] = layout["active_limit"]
    result["generation_slots"] = layout["max_num_seqs"]
    result["require_focus_windows_fit"] = True
    result["focus_fit_fallback"] = {"min_slots": CANDIDATE_FIT_FALLBACK_MIN}
    args = dict(zip(profile.extra_args[::2], profile.extra_args[1::2]))
    kv = args.get("--kv-cache-dtype", "auto")
    result["cache_experiment"]["kv_cache_dtype"] = kv
    result["candidate_serving"] = {"attention_backend": args.get("--attention-backend"), "kv_cache_dtype": kv,
                                   "quantization": profile.cli_quantization, "extra_args": list(profile.extra_args),
                                   "mtp_tokens": 0}
    result.pop("stall_dump_seconds", None)
    if selected["stall_dump"] is not None:
        result["stall_dump_seconds"] = selected["stall_dump"]  # applied to every trial by scripts/build_arena.py
    record["stall_dump"] = {"on": selected["stall_dump"] is not None, "seconds": selected["stall_dump"],
                            "overlay": "arc_infra.stall_dump_patch (standalone; no expert-prune mask)"}
    for key, why in CANDIDATE_DROPPED.items():
        record[key] = {"on": False, "dropped": True, "reason": why}
    for key, text in reasons.items():
        if key not in record:
            raise ValueError(f"hero-v2 reason for an unknown component {key!r}")
        record[key]["reason"] = text
    on = sorted(k for k, v in record.items() if v["on"])
    changed = {
        "served_model": {"before": hero.get("served_model", "Qwen/Qwen3.8-Flash-Next-NVFP4"), "after": profile.model},
        "model_revision": {"before": hero.get("model_revision"), "after": profile.revision},
        "server": {"before": {k: flash_server[k] for k in SERVER_KEYS}, "after": copy.deepcopy(result["server"]),
                   "removed": dropped_server},
        "candidate_serving": {"before": None, "after": copy.deepcopy(result["candidate_serving"])},
        "active_limit": {"before": hero["active_limit"], "after": layout["active_limit"]},
        "generation_slots": {"before": hero["generation_slots"], "after": layout["max_num_seqs"]},
        "cache_experiment.kv_cache_dtype": {"before": hero["cache_experiment"].get("kv_cache_dtype"), "after": kv},
    }
    result["hero_v2"] = {"components": record, "requested": {**selected, "candidate": name}, "on": on,
                         "off": sorted(k for k, v in record.items() if not v["on"]), "layout": layout,
                         "base": BASE, "reference": NAME, "candidate": name, "wall_seconds_target": 3600,
                         "dropped_server_keys": dropped_server}
    result["arena_variant"] = {
        "name": CANDIDATE_NAME,
        "description": (f"hero-v2 on {profile.model}: the hero-v2 arena ({', '.join(on)}) with "
                        f"the Flash-Next serving components dropped; {layout['active_limit']} focus games at 64K "
                        f"({'measured' if served_blocks is not None else 'projected'} {layout['blocks']} blocks, "
                        f"{layout['blocks_per_window']} per window), max_num_seqs {layout['max_num_seqs']}, "
                        f"{args.get('--attention-backend')} attention, {kv} KV"),
        "changed": {**result["arena_variant"]["changed"], **changed},
        "checkpoint_and_serving_unchanged": False,
        "reference_variant": NAME, "reference_difference": changed,
        "reference_checkpoint_and_serving_unchanged": False,
        "base_reference": copy.deepcopy(result["arena_variant"]["base_reference"]),
    }
    return result


# --- GPTQ overlay x pruning -------------------------------------------------------------------------------------

EXPERT_NAME = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$")


def overlay_prune_plan(weight_map, overlay_shards, pruned, experts=512):
    """How the loader combines an expert overlay with a pruning mask.

    ``weight_map``: the checkpoint index (tensor -> shard); ``overlay_shards``: the overlay manifest's shard names;
    ``pruned``: the mask's ``{layer: [expert ids]}``. The overlay replaces whole shards (same tensor names), so every
    expert tensor of an overlay shard must be a per-expert name the prune filter matches; a fused or unknown expert
    tensor would reach the pruned model unrenamed. Returns counts per layer of overlay tensors kept (renumbered) and
    dropped; raises on a name the filter would not see.
    """
    overlay = set(overlay_shards)
    unknown = sorted(overlay - set(weight_map.values()))
    if unknown:
        raise ValueError(f"overlay shards not in the checkpoint index: {unknown[:3]}")
    drop = {int(layer): {int(e) for e in ids} for layer, ids in pruned.items()}
    layers = {}
    for name, shard in weight_map.items():
        if shard not in overlay or not name.startswith("model.language_model.layers.") or ".mlp.experts" not in name:
            continue  # shared experts, routers, norms and the MTP draft pass the filter unchanged
        match = EXPERT_NAME.match(name)
        if match is None:
            raise ValueError(f"{name}: an overlay expert tensor the prune filter does not rename")
        layer, expert = int(match.group(1)), int(match.group(2))
        if layer not in drop or not 0 <= expert < experts:
            raise ValueError(f"{name}: no pruning plan for layer {layer} or expert {expert} out of range")
        row = layers.setdefault(layer, {"kept": 0, "dropped": 0})
        row["dropped" if expert in drop[layer] else "kept"] += 1
    return layers
