"""campaign. Historical studies remain unchanged."""

WALL = 32400
TIMEOUT = 36000
FIXES = ["exact_trigger", "summary_guard", "inventory", "cap_followup", "stuck_notice",
         "sandbox", "timeout_interrupt", "prompt_text"]
ARMS = ("samples", "blocks3-plans", "blocks2", "b31", "blocks2-prune25", "blocks4-plans", "runb64-plans", "runb93-plans",
        "blocks3-plans-prune25-40", "blocks2-prune37", "blocks3-samples2",
        "comp-prune25", "blocks4-plans-prune25", "blocks3-plans-prune37-40", "blocks2-plans-prune25",
        "blocks2-plans-prune37", "blocks3-samples2-prune25", "blocks3-plans-prune25-40-mtp1",
        "blocks3-plans-prune25-40-mtp1-int3", "blocks3-plans-prune25-40-int3", "blocks2-plans-prune50-int3",
        "blocks2-plans-prune37-int3")
PLANS_OFF_TWO_BLOCK = ("blocks2", "blocks2-prune25", "blocks2-prune37")
# The one arm authorized to serve a speculative draft ("create and run an 8th modal 9-hour run for
# the MTP-1 version for the 3-block 40 stream, 25% pruned variant"); every other arm stays MTP-0.
MTP_ARMS = {"blocks3-plans-prune25-40-mtp1": 1, "blocks3-plans-prune25-40-mtp1-int3": 1}
# The one arm authorized to serve 3-bit routed experts ("create and run a 9th modal 9-hour run for
# the 3-bit expert version for the MTP-1 3-block 40 stream, 25% pruned variant. i.e. i want to stack both improvements
# for this run"): the int3 g16 overlay on the Humming MoE (group size per arm).
# "launch a new run: blocks-2 + plans, pruned 50%, 3-bit experts, however many streams fit with a
# 96k context size" (blocks2-plans-prune50-int3).
INT3_ARMS = {"blocks3-plans-prune25-40-mtp1-int3": 16, "blocks3-plans-prune25-40-int3": 16,
             "blocks2-plans-prune50-int3": 16, "blocks2-plans-prune37-int3": 16}


def components(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown campaign arm: {arm}")
    out = dict(kernels_phase6=False, prefix_cache=True, kv_cache="int4",
               focus_oversubscription=True, prompt_tokens_details=True,
               window=130816, focus_slots=24, harness_fixes=list(FIXES), backlog_fix=True)
    if arm == "samples":
        out.update(parallel_samples=True, action_plans=True)
        return out
    out.update(prefix_match_unit=16, gate_margin=.96, spare_sequences=4,
               harness_fixes=FIXES + ["summary_headroom"], density_blocks=1260)
    if arm in ("runb64-plans", "runb93-plans"):
        # Submitted competition B policy, without the newer independent level allowances.
        # 46 streams pass steady-residency and readmission replay; 48 exceeds the p90 wait bar.
        out.update(window=65536, focus_slots=46, graph_capture=56, action_plans=True)
        if arm == "runb93-plans":
            out.update(window=93184, focus_slots=40, graph_capture=48)
        return out
    if arm == "comp-prune25":
        # the submitted competition Run B study (31 x 130,816, fixed-2x first-clear grants,
        # progress-fruitless backlog, no plans, no level allowance) with layer-aware 25% pruning and the newer cache
        # improvements above. Run B's 2 h trace replayed on the 1,630-block pruned floor at .96: 43 streams pass
        # (late half .35% lost, p90 0 at 20 and 40 seeds); 44 is marginal (late p90 1.98 s / 2.07 s); 45 fails.
        out.update(focus_slots=43, expert_prune="prune25-layer", density_blocks=1630)  # 47 sequences, graphs to 48
        return out
    if arm == "b31":
        out.update(focus_slots=31, level_token_allowance=90000)
        return out
    two, four = arm.startswith("blocks2"), arm.startswith("blocks4")
    slots = 55 if two else 28 if four else 37
    out.update(backlog_fix=False, budget_policy="blocks", focus_slots=slots,
               window=49152 if two else 131072 if four else 93184,
               graph_capture=64 if two else None if four else 48,
               blocks=dict(wall_seconds=WALL, blocks=2 if two else 4 if four else 3, slots=slots,
                           stagger_seconds=0, level_token_allowance=90000))
    if arm in PLANS_OFF_TWO_BLOCK:
        out["completion_cap"] = 12288
    else:
        # every other blocks arm is based on the hero blocks3-plans run (plans on, 16,384 cap).
        out["action_plans"] = True
    if arm == "blocks2-prune25":
        # ~8.29 GiB extra / 22.26 MB per int4 pool block ~= 399 extra blocks.
        # Keep the fixed recurrent overhead: 55 * 9 blocks cannot scale with context.
        # First 64-graph startup measured 1,647 blocks (not the projected 1,650 minimum).
        # Replayed at 1,630, retaining 17 blocks of profiling variability and all 55 streams.
        out.update(expert_prune="prune25-layer", window=68608, density_blocks=1630)
    if arm == "blocks4-plans-prune25":
        # blocks4-plans + layer-aware 25% pruning, 28 streams, the largest window that passes the replay
        # (190,464 on the 1,630 floor; lockstep L 0.67%, p90 0; 191,488 fails; restart-window-replay.json).
        out.update(expert_prune="prune25-layer", window=190464, density_blocks=1630)
    if arm == "blocks2-plans-prune25":
        # the hero base with 2 blocks, 55 x 68,608, 25% pruning, plans on, 16K cap (replay 0.12%, p90 0).
        out.update(expert_prune="prune25-layer", window=68608, density_blocks=1630)
    if arm == "blocks2-plans-prune37":
        # the hero base with 2 blocks, 55 x 93,184, 37.5% pruning (estimated pool, floor 2,008), plans on,
        # 16K cap (replay 0.13%, p90 0).
        out.update(expert_prune="prune37-layer", window=93184, density_blocks=2008)
    if arm == "blocks2-plans-prune50-int3":
        # "launch a new run: blocks-2 + plans, pruned 50%, 3-bit experts, however many streams fit
        # with a 96k context size". blocks2-plans-prune37 (plans on, 16K cap, gate .96, prefix unit 16, 90K x level
        # allowances, stagger 0, int4 KV, MTP-0) with layer-aware 50% pruning (prune50-layer: 12,288 experts, 128..384
        # per layer, layers 0-2 at 128), the int3 g16 experts on Humming (as blocks3-plans-prune25-40-int3; the prune
        # filter applies after the overlay) and the window 98,304 (the user's "96k"). The pool is ESTIMATED, not
        # measured: the measured MTP-0 prune25 pool (1,647) + the 6,144 further pruned experts' NVFP4 bytes (6,144 x
        # 2,764,800 B / 22,278,144 B per int4 block = 762.5) + the 12,288 kept experts' int3 savings (12,288 x 614,400 B
        # = 338.9) = 2,748 (the running int3 arm's 6,046,224 served tokens ~ 2,145 blocks realized 98% of its 508-block
        # int3 estimate). The floor keeps 90 blocks (1.87 GiB) of that estimate's uncertainty: 2,658. A smaller served
        # pool refuses before gameplay (a block is the stream count). 76 streams is the most whose steady (A/B/L x
        # all/late x floors 0/8,800) and lockstep block-start (A/B/L x floors 0/8,800) replays at 98,304 on 2,658 blocks
        # and a .96 margin keep <= 1% stream time waiting, p90 wait <= 2 s and residency within the pool (worst:
        # lockstep L floor 0, 0.95%, p90 0); 78 fails (1.36%, p90 2.4 s, residency above the pool 0.4%) (docs/
        # validation-20260930/prune50-int3-capacity.json). 80 sequences, graphs to 80; blocks of 76 in roster order
        # (76 / 34; block 2's 42 spare slots take old refills from its start).
        out.update(expert_prune="prune50-layer", expert_int3=16, window=98304, density_blocks=2658, focus_slots=76,
                   graph_capture=80)
        out["blocks"].update(slots=76, partition="streams")
    if arm == "blocks2-plans-prune37-int3":
        # "start this on modal too: blocks-2, 37.5% pruning, 3-bit, 96k context as many streams as
        # fit (same as the kaggle preview we are pushing soon)". The prune50-int3 arm above with prune37-layer, matching
        # the Kaggle build arena-flash-competition-blocks2-p37-int3-v1 exactly (60 streams, 64 sequences, graphs to 64,
        # floor 2,340, blocks of 60 / 50). The pool is ESTIMATED: the prune37 floor (2,008) + the 15,360 kept experts'
        # int3 savings (15,360 x 614,400 B / 22,278,144 B per int4 block = 423.6) = 2,432; floor 2,340 (the Kaggle
        # build's; 2,432 - 90 = 2,342). The replay at 98,304 on 2,340 blocks and a .96 margin (steady A/B/L x all/late
        # x floors 0/8,800, lockstep A/B/L x floors 0/8,800) passes up to 66 streams (worst lockstep L floor 0: 60
        # 0.02%, 66 0.70%); 68 fails (1.10%) (docs/validation-20260930/prune37-int3-capacity.json). 60 is the Kaggle
        # build's count (its stream limit was 60 / 64 sequences).
        out.update(expert_prune="prune37-layer", expert_int3=16, window=98304, density_blocks=2340, focus_slots=60,
                   graph_capture=64)
        out["blocks"].update(slots=60, partition="streams")
    if arm == "blocks3-plans-prune37-40":
        # blocks3-plans-prune25-40 with 37.5% pruning; 162,816 is the largest passing window on the 2,008
        # floor (lockstep L 0.98%; 163,840 fails at 1.07%).
        out.update(expert_prune="prune37-layer", window=162816, density_blocks=2008, focus_slots=40)
        out["blocks"].update(slots=40, partition="streams")
    if arm.startswith("blocks3-plans-prune25-40"):
        # the running blocks3-plans arm with the layer-aware 25% pruning and 40 streams (44
        # sequences, graphs to 48). Blocks of 40 in roster order (40 / 40 / 30). The pool floor is blocks2-prune25's
        # validated 1,630 (measured 1,647-1,669 at 59 sequences / 64 graphs). 123,904 is the largest window whose
        # steady (A/B/L, all/late) and lockstep block-start (A/B/L) replays at 40 streams on 1,630 blocks and a .96
        # margin keep <= 1% stream time waiting, p90 wait <= 2 s and residency within the pool
        # (docs/validation-20260930/prune25-40-capacity.json).
        out.update(expert_prune="prune25-layer", window=123904, density_blocks=1630, focus_slots=40)
        out["blocks"].update(slots=40, partition="streams")
    if arm.startswith("blocks3-plans-prune25-40-mtp1"):
        # blocks3-plans-prune25-40 with the MTP-1 draft (spec-decode.md section 5): 40 streams, 44
        # sequences, graphs [1..44] streams x 2 tokens (to 88). Measured MTP-1 prune25 pool 1,438 blocks (13 fixed
        # blocks per request, 13 QSA-layer pages per block); floor 1,420 (18 blocks of start-up variability).
        # 84,992 is the largest window whose steady (A/B/L x all/late x floors 0/8,800) and lockstep block-start
        # (A/B/L x floors 0/8,800) replays at 40 streams on the MTP-1 cost table at 1,420 blocks and a .96 margin keep
        # <= 1% stream time waiting, p90 wait <= 2 s and residency within the pool, with MTP-0 and MTP-1 (x0.77) holds;
        # 86,016 fails (lockstep L 1.01%, residency above the pool 0.04%) (docs/validation-20260930/prune25-40-mtp1-capacity.json).
        out.update(mtp_tokens=1, window=84992, density_blocks=1420, graph_capture=88)
    if arm == "blocks3-plans-prune25-40-mtp1-int3":
        # the MTP-1 arm above with 3-bit routed experts (expert-3bit.md): the int3 g16 overlay
        # (GPTQ-ARC v1 re-rounded to int3 inside the NVFP4 container, arc-expert-int3-g16-v1) on vLLM's Humming MoE,
        # repacked to uint3 at load (3.5 against Marlin's 4.5 bits per weight). The MTP draft keeps its NvFp4 b12x
        # experts (BF16 in the checkpoint, not in the overlay). The pool is ESTIMATED, not measured: the measured MTP-1
        # prune25 pool (1,438) + the 18,432 served experts' saved bytes (18,432 x 614,400 B = 10.55 GiB / 24,134,656 B
        # per MTP-1 int4 block = 469.2) = 1,907. The floor keeps 47 blocks (1.06 GiB) of that estimate's
        # uncertainty (the 18 blocks of start-up variability + Humming workspace and activation-peak differences):
        # 1,860. A smaller served pool refuses before gameplay. 131,072 is the largest window whose steady (A/B/L x
        # all/late x floors 0/8,800) and lockstep block-start (A/B/L x floors 0/8,800) replays at 40 streams on the MTP-1
        # cost table at 1,860 blocks and a .96 margin keep <= 1% stream time waiting, p90 wait <= 2 s and residency
        # within the pool (worst: lockstep L floor 0, 0.92%, p90 0.06 s); 132,096 fails (lockstep L 1.03%)
        # (docs/validation-20260930/prune25-40-mtp1-int3-capacity.json).
        out.update(expert_int3=16, window=131072, density_blocks=1860)
    if arm == "blocks3-plans-prune25-40-int3":
        # "cancel the mtp-1 run. start the 9-hour 3-bit run without mtp asap." The running MTP-0
        # blocks3-plans-prune25-40 arm (44 sequences, graphs to 48, 9 fixed blocks) with the int3 g16 experts on
        # Humming (as the MTP-1 int3 arm above). The pool is ESTIMATED, not measured: the measured MTP-0 prune25 pool
        # (1,647, the lowest start) + the 18,432 served experts' saved bytes (11,324,620,800 B / 22,278,144 B per int4
        # block = 508.3) = 2,155. The floor keeps 55 blocks (1.14 GiB) of that estimate's uncertainty: 2,100. A
        # smaller served pool refuses before gameplay. 172,032 is the largest window whose steady (A/B/L x all/late
        # x floors 0/8,800) and lockstep (A/B/L x floors 0/8,800) replays at 40 streams on 2,100 blocks and a .96
        # margin pass (worst lockstep L floor 0: 0.98%, p90 0, residency within the pool); 173,056 fails (1.04%)
        # (docs/validation-20260930/prune25-40-int3-capacity.json).
        out.update(expert_int3=16, window=172032, density_blocks=2100)
    if arm == "blocks2-prune37":
        # blocks2-prune25 with layer-aware 37.5% pruning (prune37-layer: 9,216 experts, 96..288
        # per layer, layers 0-2 at 96), 55 streams, the window scaled to the pool. The pool is ESTIMATED, not measured:
        # the measured prune25-layer pool at 59 sequences / 64 graphs (1,647) + the 3,072 further pruned experts'
        # bytes (3,072 x 2,764,800 B / 22,278,144 B per int4 block = 381.3) = 2,028. The floor keeps 20 blocks of
        # that estimate's uncertainty: 2,008. A smaller served pool refuses before gameplay (a block is the stream
        # count). 106,496 is the largest window whose steady (A/B/L, all/late, floors 0/8,800) and lockstep
        # block-start (A/B/L) replays at 55 streams on 2,008 blocks and a .96 margin keep <= 1% stream time
        # waiting, p90 wait <= 2 s and residency within the pool (docs/validation-20260930/prune37-55-capacity.json).
        out.update(expert_prune="prune37-layer", window=106496, density_blocks=2008)
    if arm.startswith("blocks3-samples2"):
        # the running blocks3-plans arm plus the corrected parallel sampler (9e2f104) with k = 2,
        # siblings only on spare capacity (the gate's background slots / try_acquire: nothing waiting, one full window
        # still fitting; siblings yield to any waiting regular request), max_num_seqs 48 (graphs to 48), no prefix
        # warm-up, no late siblings, no loser completion, min_levels 2 (the samples arm's). Regular streams: 38, the most
        # at 93,184 whose steady (A/B x floors 0/8,800 x all/late) and lockstep block-start (A/B/L) replays on the
        # measured 48-sequence prune12 pool (1,268 blocks, samples-9h-v1) at a .96 margin keep <= 1% stream time
        # waiting, p90 <= 2 s and zero residency above the pool (39 fails: 1.49% / p90 3.2 s; on the older 1,260 floor
        # 38 fails, 1.05%). Blocks of 38 in roster order (38 / 38 / 34); 42 regular + 6 sibling-only sequences
        # (docs/validation-20260930/blocks3-samples2-capacity.json).
        out.update(focus_slots=38, density_blocks=1268,
                   parallel_samples={"k": 2, "min_levels": 2, "max_num_seqs": 48, "prefix_warmup": False,
                                     "late_admission_seconds": 0.0, "observe_every": 0})
        out["blocks"].update(slots=38, partition="streams")
        if arm == "blocks3-samples2-prune25":
            out.update(expert_prune="prune25-layer", density_blocks=1630)
    return out


def finalize(study, arm):
    from .timeline import check

    study["campaign"] = {"arm": arm, "report_interval_seconds": 1800,
                         "manual_stop": True, "authorization_date": "2026-09-30"}
    study.update(notebook_seconds=TIMEOUT - 60, kernel_timeout=TIMEOUT)
    study["live_progress"] = {"interval_seconds": 60, "score_seconds": 300}
    if study.get("blocks_plan"):
        study["blocks_plan"]["min_pool_blocks"] = components(arm)["density_blocks"]
    plan = check(study, setup_seconds=1500, restart_seconds=0,
                 run_seconds={"duck-transfer": WALL + 600}, margin=1.0, teardown_seconds=300)
    study["timeline"] = {"notebook_seconds": TIMEOUT - 60, "setup_seconds": 1500,
                         "trial_tail_seconds": 600, "teardown_seconds": 300, "plan": plan}
    validate(study)
    return study


def validate(study):
    arm = study.get("campaign", {}).get("arm")
    expected = components(arm)
    actual = study["hero_v2"]["requested"]
    for key, value in expected.items():
        if key == "parallel_samples":
            value = __import__("arena.parallel_samples", fromlist=["settings"]).settings(value)
        if actual.get(key) != value:
            raise ValueError(f"Campaign {arm}: {key} differs from authorized configuration")
    if (study["wall_seconds"] != WALL or study.get("early_stop")
            or study["competition"]["preview_instances"] != 110
            or not study["competition"].get("research_clones")
            or [t["kind"] for t in study["trials"]] != ["duck-transfer"]
            or study["budget_clock"] != "tokens"
            or study["server"]["mtp_tokens"] != MTP_ARMS.get(arm, 0)):
        raise ValueError("Campaign requires nine gameplay hours, 110 public clones, Duck, MTP0 (MTP-1 for the "
                         "authorized MTP arm only) and token clocks")
    if arm in MTP_ARMS:
        from .hero_v2 import MTP_KERNELS

        server = study["server"]
        sizes = server.get("cudagraph_capture_sizes") or []
        if (any(server["serving_kernels"].get(k) != v for k, v in MTP_KERNELS.items())
                or server["serving_kernels"].get("kv_cache_4bit") != "int4"
                or not sizes or sizes[-1] != server.get("max_cudagraph_capture_size")
                or sizes[-1] != (MTP_ARMS[arm] + 1) * server["max_num_seqs"]):
            raise ValueError(f"Campaign {arm}: the MTP-{MTP_ARMS[arm]} draft stack or its CUDA graphs differ from the "
                             "validated pilot")
    kernels = study["server"].get("serving_kernels") or {}
    if arm in INT3_ARMS:
        from .variants import EXPERT_OVERLAY_INT3

        if (kernels.get("moe_backend") != "humming" or kernels.get("expert_int3") != INT3_ARMS[arm]
                or study["server"].get("expert_overlay") != EXPERT_OVERLAY_INT3[INT3_ARMS[arm]]):
            raise ValueError(f"Campaign {arm}: the int3 overlay, the Humming backend or its repack differ from the "
                             "authorized configuration")
    elif kernels.get("moe_backend") == "humming" or kernels.get("expert_int3"):
        raise ValueError(f"Campaign {arm}: 3-bit experts are authorized for the int3 arm only")
    if arm in PLANS_OFF_TWO_BLOCK and study.get("action_plans"):
        raise ValueError("Plans are disabled for both two-block arms")
    return True
