"""Flash on four L4s (sm_89, 22 GiB each): the arena's server start with a tensor-parallel switch.

    python -m arc2.l4_serving serve --study <bundle>/configs/comparison-l4.json --working W --input-root /kaggle/input
    python -m arc2.l4_serving serve-candidate ...   (a dense candidate checkpoint, e.g. Qwen3.8-27B FP8)
    python -m arc2.l4_serving load --study ... --source <bundle> --output W/trials/load --server-dir W
    arc2.l4_serving.probe(...): before the load, plain completions and the trial's first turns at rising concurrency

``serve`` in place of ``python -m arena.serving``; ``load`` runs the study's roster trial without ``arena.run``,
whose record starts with the one-RTX hardware line (``arc_infra.server.hardware``). Nothing in the shared serving code is edited: this module replaces
attributes of the loaded recipe (``arc_infra/serving_setup.py``, imported per start) and of the arena's start-up
modules in this process only, so the RTX PRO 6000 path is the same bytes and the same behaviour.

What changes for the L4s, and why:
- hardware checks: four L4s instead of one RTX PRO 6000 (``serving_core.serving.verify_hardware``, the recipe's
  ``gpu_inventory``);
- the serving environment: all four cards visible, architecture list 8.9; ``ARC_EXPERT_INT3`` set here, because the
  arena only passes it on when ``ple_uva`` is on (the UVA table read is off: the fork's CPU offload process serves
  the lookup table once for all tensor-parallel ranks, ``vllm/v1/ple_offload/connector.py``);
- the command: ``--tensor-parallel-size 4``; context 65,536 (a full 262,144 window has to fit every rank's cache),
  and the arena's own check after readiness asks for that context instead of 262,144;
- the worker snapshot after readiness may not match the one-GPU expectations: recorded, not fatal;
- the four workers each print a cache layout; if they differ the smallest pool is used.

The first profile (``l4_study``) keeps what the cards need and drops what was tuned or compiled for sm_120:
pruned 3-bit experts on Humming (required: the unpruned 4-bit model leaves no cache), no draft (its experts run on
the sm_120 b12x kernel), BF16 cache unless asked (the 4-bit cache is the second step), none of the dense, serving
or host-copy kernel overlays, no prefix-cache v2 patches. Stdlib only at import: the Modal driver imports it too.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CARDS, CARD_NAME, ARCH, CONTEXT = 4, "L4", "8.9", 65536
CONTEXTS = (65536, 98304, 131072)  # served context options (``context``); the model's own is 262,144
PROFILE = {"gpu": "4 x NVIDIA L4", "tensor_parallel_size": CARDS, "torch_cuda_arch_list": ARCH,
           "max_model_len": CONTEXT, "ple": "CPU offload process (one table for all ranks)"}
# Start 1 (9 October) was ready 14 minutes after the job began; 25 minutes leave room for a second start in one job.
# Start 3: each further first turn prefilling at once took about 210 MiB of every card (21.27 GiB at 1, 21.89 at 4,
# 22.53 of 22.56 at 7); four at once is the most that was clearly inside the card at 0.90.
PREFILL = {"slots": 4, "tokens_per_second": 1500}
# Limits as the competition bundles' (scripts/build_arena.py): Kaggle's weight load from its input disks and
# Humming's kernel build can pass 1,800 s (ARC-AGI-3 kernels: 1,525-3,000 s of setup); Modal is ready in about 1,100.
SERVER_ENV = {"ARENA_GPU_MEMORY_UTILIZATION": "0.92", "ARENA_READY_GUARD": "300,3000",
              "ARENA_SETUP_TIMEOUTS": "2700,3600"}
LOG_FACTS = ("Model loading took", "Available KV cache memory", "GPU KV cache size", "Maximum concurrency",
             "Initial free memory", "KV cache is needed", "Free memory on device", "all-reduce backends",
             "PleOffload: spawning", "Graph capturing finished", "init engine")
LOG_ERRORS = ("Traceback", "Error", "error:", "CUDA out of memory", "OutOfMemory", "raise ", "Exception",
              "not supported", "NotImplemented", "Killed", "terminated", "assert")
# What ends a served engine (start 2, 10 October: out of card memory in the first batched prefill, then a dead
# engine and refused connections). The first lines matter; thousands of traceback lines follow them.
LOG_DEATH = ("OutOfMemoryError", "memory allocation failed", "hit an exception", "EngineDeadError", "Killed",
             "CUDA error", "Segmentation", "died unexpectedly")
READY_LINE = "Application startup complete"
PROBE_LEVELS = (1, 4, 8, 12, 16, 24, 32, 48, 64, 96)
DECODE_LEVELS = (1, 2, 4, 8, 12, 16, 24, 32)
ROSTER_SECONDS, ROSTER_TASKS, ROSTER_STEP = 43200, 240, 2500  # the competition run; the curve's tokens per task
ALLOCATOR_RETRY = "memory allocation failed with OOM"
# the two draft keys (kept with ``draft``) and the two that change the cache layout or the admission clock
PREFIX_KEYS_LEFT_OUT = ("spec_decode", "tail_spec_acceptance", "state_cotenant", "admission_tick")
REPLY_CAPS = (32768, 16384, 8192)  # the loop's reply cap and the shorter arm: a request reserves its prompt plus the cap
# The 4-bit block (``serving_kernels_patch.KV4_BLOCK_TOKENS``, 3,136) is sized for one rank's GDN state on one card
# (1,634,304 B). Over four ranks a GDN state is 2 x (10,240 / 4 x (3 + draft) + 48 / 4 x 128 x 128) = 408,576 B
# (MTP-0) in a 827,904 B page (264 B a token: one KV head a rank): every fixed state block is half padding, and the
# 9 fixed blocks a request (ring 1 + 4 state groups x 2) cost 100 MiB a rank. The smallest multiple of 64 tokens
# whose page holds the state is 1,600: same bytes a token, half the bytes a fixed block.
KV4_BYTES_PER_TOKEN_RANK = 264
KV4_BLOCK_SERVED = 3136


def state_block_tokens(draft=0):
    """The smallest 4-bit block (tokens, multiple of 64) whose page on one of the four ranks holds a GDN state."""
    page = 2 * (10240 // CARDS * (3 + int(draft)) + 48 // CARDS * 128 * 128)
    return 64 * -(-page // (64 * KV4_BYTES_PER_TOKEN_RANK))


def l4_study(study, *, streams=32, kv="bf16", tasks=48, wall=420, reply_cap=1024, utilization="0.92", slots=None,
             text_only=False, prefix_caching=True, inventory=False, expert_parallel=False, roster=None,
             prefill_slots=None, prefix_patches=False, sticky=False, finite_check=False, kv4_guard=False,
             hold=False, ple_pack=False, ple_drop=False, draft=0, lean_graphs=False, probe_decode=None, profile=False,
             window=None, embed_host=False, indexer_fp8=False, grow_wait=None, dense_int8=False, tower_off=False, tower_lean=False, api_servers=0,
             prefill_chunk=None, reply_reserve=None, logits_topk=None, block_tokens=None, warmup_rows=None,
             warmup_dump=None, warm_rht=False, context=None, custom_ar=False, nccl_p2p_level=None, cpu_cache_gib=0):
    """The bundle's study with the first L4 profile and one short roster trial as load (module docstring).
    ``roster`` {"arm": name, "rates": [130, 180]}: in place of the load, the bundle's roster trial under that one
    arm, scored at those marks (tok/s over 12 h for 240 tasks, scaled to these tasks) and stopped at the largest.
    ``prefill_slots``: requests inside their prefill at once, when fewer than ``PREFILL``'s.
    ``sticky``: a task keeps its worker until its attempt ends (``arc2.trial``), so a turn follows the last one
    within seconds; in roster order the pause was 97 s at least and every freed block had been reused.
    ``hold`` (with ``sticky``): it also keeps its cache blocks at the gate between those turns (``arc2.trial``).
    ``ple_pack``: the PLE layer's prefill convolution in groups of bounded padding (``arc2.l4_ple_pack``); the
    pinned call pads every prefilling row to the step's longest, the allocation that ended the engines at 0.85.
    ``ple_drop``: the PLE offload process drops each PLE shard from the page cache once copied
    (``arc2.l4_ple_drop``; v41's 563 s load: the table's copy evicted the ranks' unread shards under the cgroup).
    ``grow_wait``: seconds an attempt waits with its blocks held for more (``arc2.trial``; 0: it gives them up at
    once, as before the hold fix). ``embed_host``: each rank's share of the input embedding (0.30 GiB) in page-locked host memory, read through
    UVA (``arc_infra.serving_kernels_runtime.host_embedding``; not yet started on four ranks). ``indexer_fp8``: the
    indexer's compressed keys in float8 (smaller blocks; it is the path of the L4 garbage finding, so its gate is
    the scored garbage check with the guard on). ``window``: the loop's context window under the served 65,536 (an attempt ends where it is full; the server
    is the same). ``lean_graphs``: no decode graphs under 8 rows (a play runs 15 or more). ``probe_decode``: the probe's decode
    ladder (default ``DECODE_LEVELS``; the top level is always run). ``profile``: the server's torch profiler
    (``ARENA_TORCH_PROFILER``) and a 40-step trace at the ladder's 16, 24 and 32 rows and in the first turns.
    ``dense_int8``: the six large projections in int8 on Humming's dense GEMM, quantized after the load
    (``arc2.l4_dense``; 5 ms of a decode step and 1.1 GiB a rank on the bench); a list keeps it to those families.
    ``tower_off``: the vision tower's weights dropped after the load and no encoder output held by the profile run
    (``arc2.l4_tower``, ``--skip-mm-profiling``; not ``text_only``'s flag, whose starts hung). ``prefill_chunk``:
    tokens a step may prefill (the server's 4,096 when unset): the 80 MiB activations of a full chunk are what the
    cards ran out of at 0.93, and the profile run sizes the cache after them.
    ``draft`` 1: the MTP draft of one token with the bundle's tuned head, the two draft keys of the prefix patches
    and graphs for two rows a stream. Its experts as the checkpoint has them (BF16, 4.69 GiB over the ranks: the
    NvFp4 draft experts are b12x's, sm_120) and its logits over the whole vocabulary (the 16K draft vocabulary,
    ``serving_kernels.draft_vocab``, refuses a head split over ranks).
    ``prefix_patches``: the bundle's prefix-cache patches (``arena.prefix_cache_patch``) without the keys that
    belong to a draft or change the cache layout. Without them the pinned runtime hits nothing across the turns
    of a conversation (arc2-l4-score-v1: 0 of 3.6M prompt tokens; chunks end off the state's block boundary).
    ``reply_cap``: replies short enough to finish inside the wall at a few tokens per second per stream.
    ``slots``: replies the load keeps open, when fewer than the server's ``streams``. ``text_only``: no vision
    tower on the ranks (``--language-model-only``). ``prefix_caching`` off: one state block per group instead of
    two, every turn prefilled whole. ``inventory``: ``arc2.l4_inventory`` in the workers; ``finite_check``:
    ``arc2.l4_finite`` (where a non-finite value first appears in the routed experts). ``kv4_guard``: the 4-bit
    cache's rotation guard (``arc_infra.kv4_runtime``, ``ARC_KV4_GUARD``). ``block_tokens`` (4-bit cache): the
    served block in place of the one-card 3,136 (``state_block_tokens``: 1,600 holds a rank's GDN state).
    ``context``: the served context (``CONTEXTS``), 65,536 by default; ``window`` follows it unless set."""
    slots = int(slots or streams)
    if not 1 <= slots <= streams:
        raise ValueError("slots is 1..streams")
    if kv not in ("bf16", "int4") or not (streams in (8, 12, 16) or 25 <= streams <= 96):
        raise ValueError("kv is bf16 or int4; streams 8, 12, 16 or 25..96 (arc_infra.serving.load_recipe)")
    out = json.loads(json.dumps(study))
    server = out["server"]
    draft = int(draft)
    if draft not in (0, 1, 2, 3):  # MTP 3 (reopened on L4; arc3 RTX +8% over MTP 1)
        raise ValueError("draft is 0..3")
    served = int(context or CONTEXT)
    if served not in CONTEXTS:
        raise ValueError(f"context is one of {CONTEXTS}")
    if window is not None and not 16384 <= int(window) <= served:
        raise ValueError("window is 16,384..the served context")
    if prefill_chunk is not None and not 512 <= int(prefill_chunk) <= 8192:
        raise ValueError("prefill_chunk is 512..8,192 tokens")
    if reply_reserve is not None and not 1024 <= int(reply_reserve) <= 16384:
        raise ValueError("reply_reserve is 1,024..16,384 tokens")
    if nccl_p2p_level not in (None, "PIX", "PXB", "PHB", "NODE", "SYS"):
        raise ValueError("nccl_p2p_level is one of PIX, PXB, PHB, NODE, SYS")
    if logits_topk and not 32 <= int(logits_topk) <= 512:
        raise ValueError("logits_topk is 32..512 (the play samples with top_k 20)")
    if block_tokens is not None and (kv != "int4" or int(block_tokens) % 64
                                     or not state_block_tokens(draft) <= int(block_tokens) <= KV4_BLOCK_SERVED):
        raise ValueError(f"block_tokens: 4-bit cache, a multiple of 64 in {state_block_tokens(draft)}..3,136")
    for key in ("dense_kernels", "stack_kernels", *(() if draft else ("draft_head",))):
        server.pop(key, None)
    rows = streams * (draft + 1)  # a draft of k decodes k + 1 positions a stream in a step
    if rows > 256:
        raise ValueError("streams x (draft + 1) is at most 256 (the arena's graph capture limit)")
    server.update(
        mtp_tokens=draft, max_num_seqs=streams, max_cudagraph_capture_size=rows,
        cudagraph_capture_sizes=sorted({n for n in (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 224)
                                        if (8 if lean_graphs else 1) <= n < rows} | {rows}),
        serving_kernels={"mamba_ssm_cache_dtype": "bfloat16", "moe_backend": "humming", "expert_int3": 16,
                         **({"kv_cache_4bit": "int4"} if kv == "int4" else {}),
                         **({"embed_host": True} if embed_host else {}),
                         **({"indexer_fp8": True} if indexer_fp8 else {})})
    if not 0 <= int(cpu_cache_gib or 0) <= 64:
        raise ValueError("cpu_cache_gib is 0..64 (host RAM for evicted KV, arc_offload)")
    out["cache_experiment"] = {"variant": "blocks", "cpu_cache_gib": int(cpu_cache_gib or 0), "kv_cache_dtype": "auto",
                               "compaction_changed": True, "probe_seconds_limit": 270,
                               "reclaim_skipped_blocks": False}
    if prefix_patches:
        out["cache_experiment"]["prefix_cache"] = {
            key: value for key, value in study["cache_experiment"]["prefix_cache"].items()
            if key not in PREFIX_KEYS_LEFT_OUT[2 * bool(draft):]}
    out.update(generation_slots=slots, active_limit=slots, windows=[int(window or served)], name=out["name"] + "-l4x4")
    names = out["trials"][0]["tasks"][:tasks]
    if len(names) < slots and not roster:  # a roster trial's tasks are the play's (arc2.kaggle), not these 48
        raise ValueError("one reply per task is open at a time: the load needs at least as many tasks as streams")
    first = out["trials"][0]
    out["trials"] = [{"name": "load", "kind": "arc2", "competition": False,
                      "server_env": {**SERVER_ENV, "ARENA_GPU_MEMORY_UTILIZATION": str(utilization),
                                     **({} if prefix_caching else {"ARENA_PREFIX_CACHING": "0"}),
                                     **({"ARENA_TORCH_PROFILER": "1"} if profile else {})},
                      "wall_seconds": wall, "split": first["split"], "tasks": names,
                      "arms": [{"name": "load", "cap": reply_cap}], "marks": {"never": 10**12}}]
    if roster:
        arm = next(a for a in first["arms"] if a["name"] == roster["arm"])
        if roster.get("set"):  # loop settings over the bundle's arm (arc2.loop.DEFAULTS keys), e.g. carry
            from .loop import DEFAULTS

            unknown = set(roster["set"]) - set(DEFAULTS)
            if unknown:
                raise ValueError(f"roster set: not loop settings: {sorted(unknown)}")
            arm = {**arm, **roster["set"], "name": roster.get("name") or arm["name"] + "-set"}
        rates = sorted(int(r) for r in roster["rates"])
        out["trials"][0].update(arms=[arm], curve_step=ROSTER_STEP * len(names),
                                marks={str(r): r * ROSTER_SECONDS * len(names) // ROSTER_TASKS for r in rates},
                                min_verified_tasks=int(first.get("min_verified_tasks", 0)))
    out["arc2"] = {**out["arc2"], "cache_layout": "smallest", "sticky_attempts": bool(sticky),
                   **({} if probe_decode is None else {"probe_decode": [int(n) for n in probe_decode]}),
                   **({"profile_rows": [16, 24, 32]} if profile else {}),
                   "hold_attempt_blocks": bool(hold and sticky),
                   **({} if grow_wait is None else {"grow_wait": float(grow_wait)}),
                   **({} if reply_reserve is None else {"reply_reserve": int(reply_reserve)}),
                   "prefill": {**PREFILL, "slots": int(prefill_slots or PREFILL["slots"])},
                   "l4x4": {**PROFILE, "kv": kv, "streams": streams, "slots": slots,
                            "gpu_memory_utilization": str(utilization), "text_only": bool(text_only),
                            "prefix_caching": bool(prefix_caching), "inventory": bool(inventory),
                            "expert_parallel": bool(expert_parallel), "roster": roster,
                            "prefix_patches": bool(prefix_patches),
                            "finite_check": finite_check if finite_check in ("moe", "attention") else bool(finite_check),
                            "kv4_guard": bool(kv4_guard), "ple_pack": bool(ple_pack), "ple_drop": bool(ple_drop),
                            "draft": draft,
                            "lean_graphs": bool(lean_graphs), "profile": bool(profile),
                            "window": int(window or served), "context": served, "embed_host": bool(embed_host),
                            "indexer_fp8": bool(indexer_fp8), "tower_off": bool(tower_off), "tower_lean": bool(tower_lean), "api_servers": int(api_servers or 0),
                            "logits_topk": int(logits_topk or 0), "warmup_rows": int(warmup_rows or 0),
                            "warmup_dump": warmup_dump or None, "warm_rht": bool(warm_rht),
                            "custom_ar": bool(custom_ar), "nccl_p2p_level": nccl_p2p_level,
                            "cpu_cache_gib": int(cpu_cache_gib or 0),
                            "block_tokens": None if block_tokens is None else int(block_tokens),
                            "prefill_chunk": None if prefill_chunk is None else int(prefill_chunk),
                            "dense_int8": sorted(dense_int8) if isinstance(dense_int8, (list, tuple)) else bool(dense_int8)}}
    return out


def candidate_study(study, name, *, streams=16, tasks=48, wall=420, reply_cap=1024, utilization="0.90", slots=None,
                    profile=None, inventory=False):
    """The L4 load study on a reviewed dense checkpoint (``arena.candidate_profiles``, e.g. qwen38-27b-fp8) in
    place of Flash: the profile's own quantization and draft tokens, BF16 cache, 64K context, and only the server
    keys a candidate start reads (``arena.candidate_serving``). ``profile``: its model, revision, effort and
    mtp_tokens, where ``arena`` is not importable (the Modal driver)."""
    profile = profile or candidate_profile(name)
    out = l4_study(study, streams=streams, kv="bf16", tasks=tasks, wall=wall, reply_cap=reply_cap,
                   utilization=utilization, slots=slots, inventory=inventory)
    out["server"] = {"max_model_len": CONTEXT, "prefix_caching": True, "mtp_tokens": profile["mtp_tokens"],
                     "max_num_seqs": streams, "gpu_memory_utilization": str(utilization),
                     "max_num_batched_tokens": out["server"]["max_num_batched_tokens"],
                     # a draft of k tokens decodes k + 1 positions per stream per step (arena.hero_v2); with
                     # graphs to ``streams`` only, the 27B step doubled between 8 and 12 streams (q27-serve-v2)
                     "max_cudagraph_capture_size": min(256, streams * (profile["mtp_tokens"] + 1))}
    out.update(candidate_profile=name, served_model=profile["model"], model_revision=profile["revision"],
               reasoning_effort=profile["effort"], startup_timeout_seconds=1500, serving_slots=streams,
               cache_experiment={"reclaim_skipped_blocks": False})
    # A candidate start takes the ready guard only; its memory share and start-up limit come from the study.
    out["trials"][0]["server_env"] = {"ARENA_READY_GUARD": SERVER_ENV["ARENA_READY_GUARD"]}
    out["arc2"]["l4x4"] = {**out["arc2"]["l4x4"], "model": name, "mtp_tokens": profile["mtp_tokens"]}
    out["arc2"].pop("prefill")  # the 27B's cards did not move with first turns at once (20.07-20.12 GiB at 4-16)
    return out


def candidate_profile(name):
    from arena.candidate_profiles import PROFILES

    profile = PROFILES[name]
    manifest = json.loads(profile.manifest.read_text())
    return {"model": profile.model, "revision": profile.revision, "effort": profile.effort,
            "mtp_tokens": profile.mtp_tokens, "dataset": manifest["dataset"]}


def study_for(study, choice):
    """``l4_study`` or, with a ``model`` in the choice, ``candidate_study``."""
    choice = dict(choice)
    name = choice.pop("model", None)
    if not name:
        return l4_study(study, **choice)
    for key in ("kv", "text_only", "prefix_caching", "expert_parallel", "roster", "prefill_slots",
                "prefix_patches", "sticky", "finite_check", "kv4_guard", "hold", "ple_pack", "ple_drop", "draft",
                "lean_graphs", "probe_decode", "profile", "window", "embed_host", "indexer_fp8", "grow_wait",
                "dense_int8", "tower_off", "tower_lean", "api_servers", "prefill_chunk", "reply_reserve", "logits_topk",
                "block_tokens", "warmup_rows", "warmup_dump", "warm_rht", "context", "custom_ar", "nccl_p2p_level", "cpu_cache_gib"):  # Flash's
        choice.pop(key, None)
    return candidate_study(study, name, **choice)


def fallback(choice, passed):
    """The next start after the engine died in the probe with ``passed`` requests at once still answered: 0.03 less
    of each card, and no more slots than were answered (the recipe takes 8, 12, 16 or 25..96). Not lower than
    0.87: below it the 4-bit pool is smaller than one 64K window and the server refuses to start."""
    if choice.get("expert_parallel"):  # a pool of several GiB: less of each card and fewer prefills at once
        if passed < 1:
            return None
        return {**choice, "utilization": f"{max(0.80, float(choice['utilization']) - 0.03):.2f}",
                "prefill_slots": max(1, min(passed, PREFILL["slots"]))}
    if passed < 1:  # not a matter of concurrency: the plainest profile that has started (start 1), fewest slots
        return {**choice, "kv": "bf16", "streams": 8, "slots": None, "utilization": "0.92"}
    streams = max((n for n in (8, 12, 16) if n <= passed), default=8)
    return {**choice, "streams": streams, "slots": min(passed, streams),
            "utilization": f"{max(0.87, float(choice['utilization']) - 0.03):.2f}"}


def start_fallback(choice):
    """The next start after one that did not come up: without the host embedding and with the vision tower (both
    starts that hung after graph capture had the embedding in host memory; profile-v2 and package-v17), else 0.03
    less of each card above 0.90, else the same start again."""
    if choice.get("text_only") or choice.get("tower_off") or choice.get("embed_host"):
        return {**choice, "text_only": False, "tower_off": False, "embed_host": False}
    if float(choice.get("utilization") or 0) > 0.905:
        return {**choice, "utilization": f"{float(choice['utilization']) - 0.03:.2f}"}
    return choice


def free_cards(wait=20.0, marks=("vllm", "agent-venv", "pleoffload")):
    """Kill the serving processes that outlive the recipe's stop (a hung start's workers: package-v17's second start
    found 1.23 GiB free) and wait until the cards are released; the pids killed. Only this container's own vLLM
    processes, found in /proc, never an ancestor: under Modal nvidia-smi's compute pids are not this namespace's, and
    killing them took the runner down (roster-v21b, both containers, exit 137 right after failure-1)."""
    import signal
    import time

    ancestors, pid = set(), os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) <= 1 or int(entry.name) in ancestors:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").lower()
        except OSError:
            continue
        if any(mark in command for mark in marks):  # vLLM retitles its workers
            pids.append(int(entry.name))
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    end = time.time() + wait
    while pids and time.time() < end:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=False).stdout.split()
        if out and all(x.strip().isdigit() and int(x) < 1024 for x in out):
            break
        time.sleep(1)
    return sorted(pids)


def cards():
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                                    "--format=csv,noheader,nounits"], text=True, timeout=60)
    return [line.strip() for line in rows.splitlines() if line.strip()]


def hardware():
    found = cards()
    if len(found) != CARDS or any(CARD_NAME not in row or int(row.split(",")[1]) < 22000 for row in found):
        raise RuntimeError(f"Require {CARDS} x {CARD_NAME}; found {found}")
    return "; ".join(found)


def meminfo():
    rows = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in ("MemTotal", "MemAvailable", "Shmem", "Cached", "Mlocked", "Unevictable"):
            rows[name] = int(value.split()[0]) * 1024
    return rows


def head_counts(config_path, ranks=CARDS):
    """Whether every head count of the model divides over ``ranks`` (the fork's own start-up checks, read ahead)."""
    text = json.loads(Path(config_path).read_text())
    text = text.get("text_config", text)
    keys = ("num_attention_heads", "num_key_value_heads", "linear_num_key_heads", "linear_num_value_heads",
            "indexer_n_heads", "indexer_kv_heads", "moe_intermediate_size", "shared_expert_intermediate_size")
    rows = {key: text.get(key) for key in keys}
    # qsa.py: heads must divide; KV heads divide, or (when fewer than the ranks) the ranks divide by them.
    ok = {key: value is not None and (value % ranks == 0 or (key.endswith("kv_heads") or key == "num_key_value_heads")
                                      and ranks % value == 0) for key, value in rows.items()}
    return {"ranks": ranks, "values": rows, "divides": ok, "all": all(ok.values())}


OPTIONS_ENV = "ARC2_L4_OPTIONS"  # the study's l4x4 block, for the recipe changes below (set by serve)
ENV_DUMP = "ARC2_L4_ENV_DUMP"  # a file for the server's environment (the NCCL diagnosis runs in it after the start)
PLUGIN = "arc2_l4_inventory-0.1.dist-info"
PLUGINS = {"inventory": "arc2_l4_inventory = arc2.l4_inventory:register",  # option -> vLLM general plugin
           "text_only": "arc2_l4_stacks = arc2.l4_stacks:register",
           "finite_check": "arc2_l4_finite = arc2.l4_finite:register",
           "dense_int8": "arc2_l4_dense = arc2.l4_dense:register",
           "tower_off": "arc2_l4_tower = arc2.l4_tower:register",
           "logits_topk": "arc2_l4_logits = arc2.l4_logits:register",
           "warmup_rows": "arc2_l4_warmup = arc2.l4_warmup:register",
           "warm_rht": "arc2_l4_warmup = arc2.l4_warmup:register",
           "custom_ar": "arc2_l4_comm = arc2.l4_comm:register",  # l4 arm: all-reduce over peer access
           "kv4_guard": "arc2_l4_finite = arc2.l4_finite:register"}  # for a bundle whose helper predates the guard


def dump_stacks(root=None, wait=3.0):
    """SIGUSR1 to the server (``root``, its pid) and every Python descendant, or to this process's descendants
    (``arc2.l4_stacks`` prints each one's stacks to the log); the pids signalled. For a start that is being given
    up: a process without the handler ends on the signal."""
    import signal
    import time

    parents = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                parents[int(entry.name)] = (int(fields[1]), (entry / "cmdline").read_bytes())
            except (OSError, IndexError, ValueError):
                continue
    found, front = [root] if root else [], [root or os.getpid()]
    while front:
        pid = front.pop()
        for child, (parent, command) in parents.items():
            if parent == pid:
                front.append(child)
                if b"python" in command or b"VLLM" in command:  # the engine and the workers rename themselves
                    found.append(child)
    for pid in found:
        try:
            os.kill(pid, signal.SIGUSR1)
        except OSError:
            pass
    time.sleep(wait)
    return sorted(found)


def plugin_files(options):
    """The ``.dist-info`` files that make vLLM load the plugins the options ask for (none: no files)."""
    lines = list(dict.fromkeys(entry for key, entry in PLUGINS.items() if options.get(key)))
    return {"METADATA": "Metadata-Version: 2.1\nName: arc2-l4-inventory\nVersion: 0.1\n",
            "entry_points.txt": "[vllm.general_plugins]\n" + "\n".join(lines) + "\n"} if lines else {}


def patch_recipe(recipe, working, serving_kernels):
    """The L4 changes on one loaded recipe module (module docstring)."""
    working = Path(working)
    options = json.loads(os.environ.get(OPTIONS_ENV) or "{}")
    if options.get("prefill_chunk"):  # here, not in the study: the loader takes 4,096 or 8,192 only, and the
        os.environ[recipe.MAX_NUM_BATCHED_TOKENS_ENV] = str(int(options["prefill_chunk"]))  # recipe reads this in main

    def gpu_inventory():
        rows, memory = hardware(), meminfo()
        if memory.get("MemAvailable", 0) < recipe.MIN_HOST_AVAILABLE_BYTES:
            raise RuntimeError(f"Host memory is too small for the PLE CPU offload: {memory}")
        return {"gpu_rows": rows.split("; "), "meminfo_bytes": memory,
                "host_mem_available_bytes": memory["MemAvailable"]}

    environment = recipe.runtime_environment

    def runtime_environment(*args, **kwargs):
        env, check = environment(*args, **kwargs)
        if env.get("CUDA_VISIBLE_DEVICES") != "0" or env.get("TORCH_CUDA_ARCH_LIST") != "12.0":
            raise RuntimeError("the recipe's one-card environment changed; re-read arc2.l4_serving")
        env = {**env, "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(CARDS)), "TORCH_CUDA_ARCH_LIST": ARCH}
        # NCCL's host buffers without cuMem: on Modal's 610.57.04 hosts (gVisor) importing one fails (Cuda failure
        # 801) and the first all-reduce dies; without it the reduction is right at the same latency (nccl610-v2)
        env.setdefault("NCCL_CUMEM_HOST_ENABLE", "0")
        if (serving_kernels or {}).get("expert_int3"):
            from arc_infra.expert_int3_patch import ENV as INT3_ENV

            env.setdefault(INT3_ENV, json.dumps({"group": int(serving_kernels["expert_int3"])}, sort_keys=True))
        files = plugin_files(options)
        if files:  # vLLM general plugins, found through a .dist-info on the path
            folder = working / "l4-plugin" / PLUGIN
            folder.mkdir(parents=True, exist_ok=True)
            for name, text in files.items():
                (folder / name).write_text(text)
        if options.get("nccl_p2p_level"):  # l4 arm: Kaggle's L4 pairs are PHB with peer access; NCCL used SHM
            env["NCCL_P2P_LEVEL"] = str(options["nccl_p2p_level"])
        if options.get("custom_ar"):  # arc2.l4_comm (plugin): vLLM's custom all-reduce on four PCIe cards
            env["ARC2_L4_CUSTOM_AR"] = "1"
        if options.get("kv4_guard"):
            env["ARC_KV4_GUARD"] = "1"
        if options.get("ple_drop"):  # read by the edit server_command applies (arc2.l4_ple_drop)
            from arc2.l4_ple_drop import ENV as PLE_DROP_ENV, PATTERN as PLE_DROP_PATTERN

            env[PLE_DROP_ENV] = PLE_DROP_PATTERN
        if files:
            if options.get("finite_check"):  # the site arc2.l4_finite checks in the workers
                env["ARC2_FINITE_CHECK"] = "attention" if options["finite_check"] == "attention" else "moe"
            if options.get("dense_int8"):  # arc2.l4_dense: the large projections in 8 bits after the load
                env["ARC2_L4_DENSE"] = json.dumps({"mode": "serve", **(
                    {"families": options["dense_int8"]} if isinstance(options["dense_int8"], list) else {})})
            if options.get("tower_off"):
                env["ARC2_L4_TOWER"] = "off"
            if options.get("logits_topk"):  # arc2.l4_logits: the ranks' top k gathered, not the whole logits
                env["ARC2_L4_LOGITS_TOPK"] = str(int(options["logits_topk"]))
            if options.get("warmup_rows"):  # arc2.l4_warmup: the post-capture eager warm-up capped, its steps timed
                env["ARC2_L4_WARMUP_ROWS"] = str(int(options["warmup_rows"]))
            if options.get("warm_rht"):  # arc2.l4_warmup: the 4-bit rotation eagerly before graph capture
                env["ARC2_L4_WARM_RHT"] = "1"
            if options.get("warmup_dump"):  # test starts only: "every,until" seconds of stack dumps
                env["ARC2_L4_WARMUP_ROWS_DUMP"] = str(options["warmup_dump"])
            env["PYTHONPATH"] = os.pathsep.join(
                [*filter(None, [env.get("PYTHONPATH")]), str(folder.parent), str(Path(__file__).resolve().parents[1])])
        if os.environ.get(ENV_DUMP):  # container-local: the whole environment, never among the evidence
            Path(os.environ[ENV_DUMP]).write_text(json.dumps(env))
        return env, {**check, "l4x4": PROFILE}

    command = recipe.server_command

    def server_command(*args, **kwargs):
        argv = command(*args, **kwargs)
        at = argv.index("--tensor-parallel-size") + 1
        if argv[at] != "1":
            raise RuntimeError("the recipe's tensor-parallel size changed; re-read arc2.l4_serving")
        argv[at] = str(CARDS)
        if options.get("text_only"):
            argv.append("--language-model-only")
        elif options.get("tower_off"):  # arc2.l4_tower drops the weights; the profile run holds no encoder output
            argv.append("--skip-mm-profiling")
        elif options.get("tower_lean"):  # the tower stays (images served); its profile-run encoder output is not
            argv.append("--skip-mm-profiling")  # reserved from the cache (small images: image_side 224)
        if options.get("api_servers", 0) > 1:  # request rendering (chat template, tokenizer, multimodal processor)
            argv += ["--api-server-count", str(int(options["api_servers"]))]  # in N processes, not one thread
        if options.get("expert_parallel"):  # whole experts per rank: the tensor-parallel shards load padded
            argv.append("--enable-expert-parallel")
        if options.get("block_tokens"):  # the 4-bit flag's one-card block (serving_kernels_patch.server_flags)
            at = argv.index("--block-size") + 1
            from arc_infra.serving_kernels_patch import kv4_block_tokens  # 3,136; 3,264 with a 3-token draft

            if argv[at] != str(kv4_block_tokens("int4", int(options.get("draft") or 0))):
                raise RuntimeError(f"the 4-bit block is {argv[at]}, not the patch's own; re-read arc2.l4_serving")
            argv[at] = str(int(options["block_tokens"]))
        if options.get("ple_pack"):  # here: after the recipe's own PLE patch, which wants the stock file
            from arc2 import l4_ple_pack

            record = l4_ple_pack.apply(recipe.RUNTIME_ROOT / "usr/local/lib/python3.12/dist-packages", working)
            print("ARC2_PLE_PACK " + json.dumps(record), flush=True)
        if options.get("ple_drop"):  # a speed fix only: a runtime it does not fit starts without it
            from arc2 import l4_ple_drop

            try:
                record = l4_ple_drop.apply(recipe.RUNTIME_ROOT / "usr/local/lib/python3.12/dist-packages", working)
            except (OSError, RuntimeError, SyntaxError) as exc:
                record = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            print("ARC2_PLE_DROP_PATCH " + json.dumps(record), flush=True)
        return argv

    if options.get("text_only") or options.get("tower_off"):
        # No vision tower on the cards: the recipe's readiness check with an image is not asked (its record says
        # so), and a start given up prints every server process's stacks first (the one text-only start so far,
        # arc2-l4-serve-v4, hung in the workers after graph capture and left no trace of where).
        asked, waited = recipe.chat, recipe.wait_for_server

        def chat(payload, *args, **kwargs):
            if any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])
                   for m in payload.get("messages") or []):
                return {"choices": [{"message": {"content": "red (not asked: no vision tower on these cards)"},
                                     "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}, 0.0
            return asked(payload, *args, **kwargs)

        def wait_for_server(identity, *args, **kwargs):
            try:
                return waited(identity, *args, **kwargs)
            except BaseException:
                print("ARC2_STACKS " + json.dumps(dump_stacks(int(identity["pid"]))), flush=True)
                raise

        recipe.chat, recipe.wait_for_server = chat, wait_for_server

    snapshot = recipe.update_ready_identity

    def update_ready_identity(identity):
        try:
            return snapshot(identity)
        except RuntimeError as exc:  # the one-GPU process expectations; the server is up
            (working / "l4-identity-note.json").write_text(json.dumps({"error": str(exc)[:2000]}, indent=1))
            return identity

    main = recipe.main

    def recipe_main():
        recipe.ANALYZER_CONTEXT = served_context()  # serving_core.serving.start sets 262,144 just before
        return main()

    recipe.gpu_inventory, recipe.runtime_environment = gpu_inventory, runtime_environment
    recipe.server_command, recipe.update_ready_identity, recipe.main = server_command, update_ready_identity, recipe_main


def served_context():
    """The served context: the study's ``context`` (``OPTIONS_ENV``, set by serve) or 65,536."""
    return int(json.loads(os.environ.get(OPTIONS_ENV) or "{}").get("context") or CONTEXT)


def install():
    """Route this process's server start through the L4 changes."""
    import arena.serving as arena_serving
    import arena.serving_core.serving as core

    core.verify_hardware = hardware
    preflight = core.preflight

    async def preflight_l4(config=None):  # the arena's default asks the tokenizer for the 262,144 context
        from arena.serving_core.config import Config

        return await preflight(config or Config(context_window=served_context(), history_tokens=None, compaction_tokens=None))

    core.preflight = preflight_l4
    configure = arena_serving.configure

    def configure_l4(recipe, working, **kwargs):
        result = configure(recipe, working, **kwargs)
        patch_recipe(recipe, working, kwargs.get("serving_kernels"))
        return {**result, "l4x4": PROFILE}

    arena_serving.configure = configure_l4
    return arena_serving


def log_lines(server_dir, patterns, limit):
    lines = []
    for log in sorted(Path(server_dir).glob("*.log")):
        for line in log.read_text(errors="replace").splitlines():
            if any(pattern in line for pattern in patterns):
                lines.append(line.strip()[:400])
    return lines[-limit:]


def log_first(server_dir, patterns=LOG_DEATH, limit=24, after=READY_LINE):
    """The first lines after readiness that match: the cause, before the lines it causes."""
    lines = []
    for log in sorted(Path(server_dir).glob("*.log")):
        ready = False
        for line in log.read_text(errors="replace").splitlines():
            ready = ready or after in line
            if ready and len(lines) < limit and any(pattern in line for pattern in patterns):
                lines.append(line.strip()[:1200])
    return lines


RETRY_DELAYS = (1, 2, 4)  # a reset or refused connection is sent again (ARC-3 v85p cache_probe; arc2 sched-p1, 18 Oct)


def post(url, payload=None, timeout=60, method=None):
    """One request, no proxy: status (None when no reply came), the whole body, an error's too, and the seconds.
    A connection reset / refused / closed without a reply is retried after 1, 2 and 4 s; a timeout is not."""
    import http.client
    import socket
    import time
    import urllib.error
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = None if payload is None else json.dumps(payload).encode()
    begun = time.monotonic()
    for attempt in range(len(RETRY_DELAYS) + 1):
        request = urllib.request.Request(url, data, {"Content-Type": "application/json"},
                                         method=method or ("GET" if payload is None else "POST"))
        try:
            with opener.open(request, timeout=timeout) as response:
                status, body = response.status, response.read().decode(errors="replace")
        except urllib.error.HTTPError as exc:
            status, body = exc.code, exc.read().decode(errors="replace")
        except Exception as exc:  # noqa: BLE001 - refused, reset, timed out: the text is the record
            status, body = None, f"{type(exc).__name__}: {exc}"
            reason = getattr(exc, "reason", exc)
            transient = isinstance(reason, (ConnectionError, http.client.RemoteDisconnected,
                                            http.client.IncompleteRead)) and not isinstance(reason, socket.timeout)
            if transient and attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])
                continue
        break
    return {"status": status, "body": body, "seconds": round(time.monotonic() - begun, 2),
            **({"retries": attempt} if attempt else {})}


def metric(text, name):
    return sum(float(line.rsplit(" ", 1)[1]) for line in text.splitlines() if line.startswith(name))


def decode_steps(base_url, model, fields, levels, *, tokens=384, level_seconds=150, used=cards, profile=(),
                 ask=None, window=(0.25, 0.9)):
    """The decode step at each stream count with no prefill in the window: ``level`` requests of about 30 prompt
    tokens generate ``tokens`` each; the server's generated-token counter is read four times a second, and the rate
    is taken between a quarter and nine tenths of the level's tokens, while every request is still decoding.
    ``profile``: levels run again after the ladder with the server's torch profiler started once every request
    decodes (it skips 3 steps and stops itself after 40); the row carries ``profiled`` and its epochs.
    ``ask``: request ``i``'s messages in place of the short prompt, and ``window`` the share of the level's tokens
    the rate is taken between (``arc2.l4_depth``: long prompts served from the prefix cache, so a later window)."""
    import threading
    import time

    from . import tasks

    root, chat = base_url.rstrip("/").rsplit("/v1", 1)[0], base_url.rstrip("/") + "/chat/completions"

    def generated():
        reply = post(root + "/metrics", timeout=10)
        if reply["status"] != 200:
            return None, None
        return metric(reply["body"], "vllm:generation_tokens_total"), metric(reply["body"], "vllm:num_requests_running")

    rows = []
    again = sorted({n for n in profile if n < levels[-1]} | {levels[-1]}) if profile else []
    for level, traced in [(n, False) for n in levels] + [(n, True) for n in again]:
        replies, samples, peak = [None] * level, [], None
        start, _ = generated()
        trace = None

        def one(i):
            messages = ask(i) if ask else [{"role": "user", "content": (
                f"Write the integers from {1000 * (i + 1) + 1} upward, one per line, without stopping and without "
                "any other text.")}]
            replies[i] = post(chat, tasks.chat_request(model, messages, tokens, **fields), timeout=level_seconds)

        threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(level)]
        t0 = time.monotonic()
        for thread in threads:
            thread.start()
        ticks = 0
        while any(thread.is_alive() for thread in threads):
            now, running = generated()
            if now is not None and start is not None:
                samples.append((time.monotonic() - t0, now - start, running))
                if traced and trace is None and now - start >= 0.15 * level * tokens:
                    trace = {"started_epoch": time.time(), "start": post(root + "/start_profile", method="POST",
                                                                         timeout=60)["status"]}
            ticks += 1
            if ticks % 16 == 0:
                try:
                    cards_now = [int(row.split(",")[2]) for row in used()]
                    peak = cards_now if peak is None else [max(a, b) for a, b in zip(peak, cards_now)]
                except Exception:  # noqa: BLE001 - a reading
                    pass
            time.sleep(0.25)
        good = [reply for reply in replies if reply and reply["status"] == 200]
        usage = [json.loads(reply["body"]).get("usage") or {} for reply in good]
        if trace is not None:
            trace.update(stop=post(root + "/stop_profile", method="POST", timeout=300)["status"],
                         stopped_epoch=time.time())
        row = {"at_once": level, "answered": len(good), "seconds": round(time.monotonic() - t0, 1),
               **({"profiled": trace} if traced else {}),
               "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage),
               "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage), "card_used_mib_peak": peak,
               "cached_tokens": sum((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0 for u in usage),
               "finish": sorted({json.loads(reply["body"])["choices"][0].get("finish_reason") for reply in good})}
        total = level * tokens
        span = [sample for sample in samples if window[0] * total <= sample[1] <= window[1] * total]
        if len(span) >= 2 and span[-1][0] > span[0][0]:
            rate = (span[-1][1] - span[0][1]) / (span[-1][0] - span[0][0])
            running = min(sample[2] for sample in span)
            # with a draft a step yields more than one token: step_ms is then the time per token of a stream
            row.update(window_seconds=round(span[-1][0] - span[0][0], 2), tokens_per_second=round(rate, 1),
                       stream_tokens_per_second=round(rate / max(1, min(level, running or level)), 2),
                       running_min=running, step_ms=round(1000 * min(level, running or level) / rate, 2) if rate else None)
        if len(good) < level:
            row["errors"] = [{**reply, "body": reply["body"][:4000]} for reply in replies
                             if reply and reply["status"] != 200][:2]
        rows.append(row)
        if len(good) < level:
            break
    return rows


def probe(base_url, study, source, *, server_dir=None, budget=None, budget_seconds=360, level_seconds=180,
          reply_tokens=256, used=cards):
    """Before the load, so a start ends with tokens or with the server's own error text: one bare completion, one
    with the trial's request fields, then the load trial's first turns (the trial's exact bodies but for
    ``max_tokens``) at 1, 4, 8, ... at once, from a cold prefix cache each time, up to ``top``: the load's slots,
    or fewer when the cache ``budget`` (the trial gate's) admits fewer of the largest first turns. Every answer
    that is not a 200 is kept whole. ``passed``: the most requests at once that were all answered."""
    import threading
    import time

    from . import tasks

    begun, spec, trial = time.monotonic(), study["arc2"], study["trials"][0]
    root, chat = base_url.rstrip("/").rsplit("/v1", 1)[0], base_url.rstrip("/") + "/chat/completions"
    out = {"passed": 0, "top": None, "levels": [], "stopped": None}

    def kept(reply):
        if reply["status"] != 200:
            return {**reply, "body": reply["body"][:60000]}
        try:
            data = json.loads(reply["body"])
            message = data["choices"][0]["message"]
            return {"status": 200, "seconds": reply["seconds"], "usage": data.get("usage") or {},
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "text": ((message.get("reasoning") or message.get("reasoning_content") or "")
                             + (message.get("content") or ""))[:300]}
        except Exception as exc:  # noqa: BLE001 - a 200 that is not a completion is not an answer
            return {"status": "200 unreadable", "error": repr(exc), "seconds": reply["seconds"],
                    "body": reply["body"][:60000]}

    def finish(reason):
        out.update(stopped=reason, health=post(root + "/health", timeout=20)["status"],
                   seconds=round(time.monotonic() - begun, 1),
                   server_errors=log_first(server_dir) if server_dir else [])
        return out

    listing = post(base_url.rstrip("/") + "/models", timeout=30)
    if listing["status"] != 200:
        out["models"] = listing
        return finish("no model listing")
    model = json.loads(listing["body"])["data"][0]["id"]
    fields = {"temperature": (spec.get("settings") or {}).get("temperature") or float(study["temperature"]),
              "effort": (spec.get("settings") or {}).get("effort") or study["reasoning_effort"]}
    hello = [{"role": "user", "content": "Reply with the single word: ready"}]
    # The first request also pays the kernels compiled at first use.
    out["bare"] = kept(post(chat, {"model": model, "messages": hello, "max_tokens": 48}, timeout=240))
    if out["bare"]["status"] != 200:
        return finish("the bare completion failed")
    out["plain"] = kept(post(chat, tasks.chat_request(model, hello, 48, **fields), timeout=120))
    if out["plain"]["status"] != 200:
        return finish("the completion with the trial's request fields failed")
    # Decode steps first: short prompts cannot run a card out of memory, the first turns below can.
    served = int(spec.get("l4x4", {}).get("streams") or study["generation_slots"])
    fits = served if budget is None else max(1, int(budget.capacity // budget.cost(64 + 384)))
    ladder = DECODE_LEVELS if spec.get("probe_decode") is None else spec["probe_decode"]
    out["decode"] = decode_steps(base_url, model, fields,
                                 sorted({n for n in ladder if n < min(served, fits)} | {min(served, fits)}),
                                 used=used, profile=spec.get("profile_rows") or ())
    if out["decode"] and out["decode"][-1]["answered"] < out["decode"][-1]["at_once"]:
        out["passed"] = out["decode"][-2]["at_once"] if len(out["decode"]) > 1 else 0
        return finish(f"decode at {out['decode'][-1]['at_once']} at once failed")
    task_map = tasks.load(Path(source) / spec["data"] / trial["split"], trial["tasks"])
    bodies = [tasks.chat_request(model, [{"role": "system", "content": tasks.SYSTEM},
                                         {"role": "user", "content": tasks.task_prompt(task_map[name])}],
                                 reply_tokens, **fields) for name in trial["tasks"]]
    top, at = int(study["generation_slots"]), 0
    if budget is not None:  # what the trial's gate would hold open of the largest first turns
        costs, counts = [], []
        for body in bodies:
            count = post(root + "/tokenize", {key: body[key] for key in ("model", "messages", "chat_template_kwargs")},
                         timeout=60)
            if count["status"] != 200:
                out["tokenize"] = count
                return finish("the token count failed")
            counts.append(int(json.loads(count["body"])["count"]))
            costs.append(budget.cost(counts[-1] + reply_tokens))
        fit, held = 0, 0
        for cost in sorted(costs, reverse=True):
            if held + cost > budget.capacity:
                break
            fit, held = fit + 1, held + cost
        median, largest = sorted(counts)[len(counts) // 2], max(counts)
        out["gate"] = {"capacity_blocks": budget.capacity, "first_turns_that_fit": fit,
                       "blocks_per_first_turn": [min(costs), sorted(costs)[len(costs) // 2], max(costs)],
                       "first_turn_tokens": {"median": median, "largest": largest},
                       # the real loop: a request reserves its prompt plus the reply cap; a second turn re-sends
                       # the first prompt and a full first reply
                       "real_loop_streams": {str(cap): {
                           "blocks_per_request": budget.cost(median + cap),
                           "median_first_turn": int(budget.capacity // budget.cost(median + cap)),
                           "largest_first_turn": int(budget.capacity // budget.cost(largest + cap)),
                           "second_turn": int(budget.capacity // budget.cost(
                               min(int(budget.layout["max_model_len"]) - int(budget.layout["lookahead"]), median + 2 * cap))),
                           # with ``reply_reserve``: the second turn holds its prompt and the reserve, not the cap
                           **({"second_turn_reserved": int(budget.capacity // budget.cost(min(
                               int(budget.layout["max_model_len"]) - int(budget.layout["lookahead"]),
                               median + cap + min(cap, int(spec["reply_reserve"])))))}
                              if spec.get("reply_reserve") else {})}
                           for cap in REPLY_CAPS}}
        top = max(1, min(top, fit))
    if spec.get("prefill"):  # the load never has more first turns in prefill than this
        top = min(top, int(spec["prefill"]["slots"]))
    out["top"] = top
    for level in sorted({n for n in PROBE_LEVELS if n < top} | {top}):
        if time.monotonic() - begun > budget_seconds:
            return finish(f"out of time before {level} at once")
        reset = post(root + "/reset_prefix_cache", method="POST", timeout=60)["status"]
        if spec.get("profile_rows") and level == min(4, top):  # the prefill step, traced once
            out["prefill_profile"] = {"at_once": level, "started_epoch": time.time(),
                                      "start": post(root + "/start_profile", method="POST", timeout=60)["status"]}
        replies, peak, t0 = [None] * level, None, time.monotonic()

        def one(i, body):
            replies[i] = post(chat, body, timeout=level_seconds)

        threads = [threading.Thread(target=one, args=(i, bodies[(at + i) % len(bodies)]), daemon=True)
                   for i in range(level)]
        at += level
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            try:
                now = [int(row.split(",")[2]) for row in used()]
                peak = now if peak is None else [max(a, b) for a, b in zip(peak, now)]
            except Exception:  # noqa: BLE001 - a reading
                pass
            time.sleep(2)
        seconds, rows = time.monotonic() - t0, [kept(reply) for reply in replies]
        if "prefill_profile" in out and "stop" not in out["prefill_profile"]:
            out["prefill_profile"].update(stop=post(root + "/stop_profile", method="POST", timeout=300)["status"],
                                          stopped_epoch=time.time())
        good = [row for row in rows if row["status"] == 200]
        generated = sum(row["usage"].get("completion_tokens", 0) for row in good)
        prompts = [row["usage"].get("prompt_tokens", 0) for row in good]
        out["levels"].append({
            "at_once": level, "answered": len(good), "seconds": round(seconds, 1), "prefix_cache_reset": reset,
            "prompt_tokens": sum(prompts), "prompt_tokens_longest": max(prompts, default=0),
            "completion_tokens": generated, "tokens_per_second": round(generated / max(1e-9, seconds), 1),
            "card_used_mib_peak": peak, "statuses": sorted({str(row["status"]) for row in rows}),
            "errors": [row for row in rows if row["status"] != 200][:3]})
        if len(good) < level:
            return finish(f"{level} at once: {level - len(good)} not answered")
        out["passed"] = level
    post(root + "/reset_prefix_cache", method="POST", timeout=60)  # the load starts cold
    return finish(None)


def server_facts(server_dir, window=CONTEXT):
    """What a started server says about itself: load, cache, streams that fit ``window`` tokens."""
    server_dir, facts = Path(server_dir), {}
    facts["log"] = log_lines(server_dir, LOG_FACTS, 40)
    layouts = []
    for log in sorted(server_dir.glob("*.log")):
        for line in log.read_text(errors="replace").splitlines():
            if "ARC_CACHE_LAYOUT " in line:
                layouts.append(json.JSONDecoder().raw_decode(line.split("ARC_CACHE_LAYOUT ", 1)[1])[0])
    found = []
    for log in sorted(server_dir.glob("*.log")):
        for line in log.read_text(errors="replace").splitlines():
            if "ARC2_L4_INVENTORY" in line:
                tag, _, text = line.partition("ARC2_L4_INVENTORY")
                try:
                    found.append({"process": tag.strip()[:40], **({"error": text} if text.startswith("_ERROR") else
                                                                 json.JSONDecoder().raw_decode(text.strip())[0])})
                except ValueError:
                    found.append({"process": tag.strip()[:40], "unread": text[:300]})
    if found:
        facts["inventory"] = found
    dense = []
    for log in sorted(server_dir.glob("*.log")):
        for line in log.read_text(errors="replace").splitlines():
            if "ARC2_DENSE8" in line:
                tag, _, text = line.partition("ARC2_DENSE8")
                try:
                    dense.append({"process": tag.strip()[:40], **({"error": text} if text.startswith("_ERROR") else
                                                                 json.JSONDecoder().raw_decode(text.strip())[0])})
                except ValueError:
                    dense.append({"process": tag.strip()[:40], "unread": text[:300]})
    if dense:
        facts["dense_int8"] = dense
    if layouts:
        layout = min(layouts, key=lambda x: x["num_blocks"])
        cost = next((row["blocks"] for row in layout["costs"] if row["tokens"] >= window), None)
        facts["cache"] = {"layouts": len(layouts), "num_blocks": sorted({x["num_blocks"] for x in layouts}),
                          "blocks_at": {str(row["tokens"]): [row["blocks"], row.get("groups")]
                                        for row in layout["costs"] if row["tokens"] in (1, 4096, 8192, 16384, 24576,
                                                                                        32768, 65536, 131072)},
                          "blocks_per_window": cost, "max_model_len": layout["max_model_len"],
                          "streams_that_fit": layout["num_blocks"] // cost if cost else None}
    for name in ("native-preflight.json", "vllm-setup-provenance.json"):
        path = server_dir / name
        if path.is_file():
            data = json.loads(path.read_text())
            if name.startswith("native"):
                facts["preflight"] = {k: data.get(k) for k in ("hardware", "served_models", "max_model_len",
                                                               "kv_admission_log")}
            else:
                facts["setup"] = {"setup_elapsed_seconds": data.get("setup_elapsed_seconds"),
                                  "ready_seconds": (data.get("readiness") or {}).get("ready_seconds"),
                                  "argv": (data.get("server_identity") or {}).get("argv")}
    return facts


def failure(server_dir, stage, error, process_log=None):
    """One record of a failed start: the stage, the recipe's own failure files, the log's error lines and tail."""
    server_dir, out = Path(server_dir), {"stage": stage, "error": str(error)[:3000]}
    for name, keys in (("native-preflight-failure.json", ("stage", "error")),
                       ("vllm-setup-failure.json", ("error_type", "error", "traceback"))):
        path = server_dir / name
        if path.is_file():
            data = json.loads(path.read_text())
            out[name] = {key: str(data.get(key))[-4000:] for key in keys}
    out["first_errors_after_ready"] = log_first(server_dir)
    out["log_errors"] = log_lines(server_dir, LOG_ERRORS, 60)
    out["log_facts"] = log_lines(server_dir, LOG_FACTS, 20)
    log = server_dir / "vllm-openai-server.log"
    if log.is_file():
        out["server_log_tail"] = log.read_text(errors="replace").splitlines()[-80:]
    if process_log and Path(process_log).is_file():
        out["process_log_tail"] = Path(process_log).read_text(errors="replace").splitlines()[-60:]
    try:
        out["cards"], out["meminfo"] = cards(), meminfo()
    except Exception as exc:  # noqa: BLE001 - the record matters more than this line
        out["cards_error"] = repr(exc)
    return out


def steady_rate(events_path, low=0.15, high=0.6):
    """Generated tokens per second between ``low`` and ``high`` of the trial, from the replies' own intervals
    (replies still open at the end are not in the record, so the tail is left out)."""
    from .loop import token_clock

    replies = [row for row in map(json.loads, Path(events_path).read_text().splitlines())
               if row.get("event") == "reply"]
    if not replies:
        return None
    start, end = min(r["t0"] for r in replies), max(r["t1"] for r in replies)
    a, b = start + low * (end - start), start + high * (end - start)
    clock = token_clock(replies)
    return {"tokens_per_second": (clock(b) - clock(a)) / max(1e-9, b - a), "window_seconds": [a - start, b - start],
            "replies": len(replies), "span_seconds": end - start,
            "open_streams_mean": sum(min(r["t1"], b) - max(r["t0"], a) for r in replies
                                     if r["t1"] > a and r["t0"] < b) / max(1e-9, b - a)}


def prompt_cache(events_path):
    """Prompt tokens sent and the share the server took from its prefix cache, from the replies' usage."""
    replies = [row for row in map(json.loads, Path(events_path).read_text().splitlines())
               if row.get("event") == "reply"]
    known = [r for r in replies if r.get("cached_tokens") is not None]
    sent, cached = sum(r["prompt_tokens"] for r in known), sum(r["cached_tokens"] for r in known)
    later = [r for r in known if r["attempt"] > 1 or r["reply"] > 1]  # a first turn has nothing to hit
    generated = sum(r["completion_tokens"] for r in known)
    return {"replies": len(replies), "replies_with_usage": len(known), "prompt_tokens": sent, "cached_tokens": cached,
            "uncached_tokens": sent - cached, "cached_share": cached / sent if sent else None,
            "later_turns": {"replies": len(later), "prompt_tokens": sum(r["prompt_tokens"] for r in later),
                            "cached_tokens": sum(r["cached_tokens"] for r in later)},
            "uncached_per_generated": (sent - cached) / generated if generated else None}


def allocator_retries(server_dir, after=READY_LINE):
    """The allocator's out-of-memory warnings after readiness (a retry, not a crash): count, first and last line."""
    lines = []
    for log in sorted(Path(server_dir).glob("*.log")):
        ready = False
        for line in log.read_text(errors="replace").splitlines():
            ready = ready or after in line
            if ready and ALLOCATOR_RETRY in line:
                lines.append(line.strip()[:300])
    return {"lines": len(lines), "first": lines[:2], "last": lines[-1:]}


def within(wide, deep):
    """Experts ``deep`` prunes beyond ``wide`` when it prunes every expert ``wide`` does (two ``pruned`` maps of
    mask files, layer -> ids); None when it does not."""
    extra = 0
    for layer, ids in wide.items():
        a, b = {int(e) for e in ids}, {int(e) for e in deep.get(layer, ())}
        if not a <= b:
            return None
        extra += len(b - a)
    return extra


def nest_overlay(study, bundle):
    """A mask that prunes more inside the mask the expert overlay was fit for, served from the same shards
    (``arc2.nested_mask``: the overlay's mask, bundle-relative, and its sha256; ``server.expert_prune`` is the
    deeper one). The overlay omits its mask's experts under their own ids, so every expert the deeper mask keeps
    is in it; the arena's loader takes an overlay only with the mask in its manifest, so it is shown that one
    once both files are read and the nesting holds. Returns the experts pruned beyond the overlay's mask."""
    held = study["arc2"].get("nested_mask")
    if not held:
        return None
    import arena.expert_overlay as overlay
    from arc_infra.expert_prune_patch import load_mask

    served = study["server"]["expert_prune"]
    extra = within(load_mask(Path(bundle) / held["mask"], held["sha256"])["pruned"],
                   load_mask(Path(bundle) / served["mask"], served["sha256"])["pruned"])
    if not extra:
        raise RuntimeError("arc2.nested_mask: the served mask is not a deeper mask inside the overlay's")
    materialise = overlay.materialise

    def materialise_nested(*args, expert_prune_sha256=None, **kwargs):
        if expert_prune_sha256 != served["sha256"]:
            return materialise(*args, expert_prune_sha256=expert_prune_sha256, **kwargs)
        record = materialise(*args, expert_prune_sha256=held["sha256"], **kwargs)
        return {**record, "nested_mask": {"overlay_mask_sha256": held["sha256"], "served_mask_sha256": served["sha256"],
                                          "experts_pruned_beyond_the_overlay": extra}}

    overlay.materialise = materialise_nested
    return extra


def serve(args):
    study = json.loads(Path(args.study).read_text())
    if study["arc2"].get("l4x4", {}).get("tensor_parallel_size") != CARDS:
        raise SystemExit("not an L4 study (arc2.l4_serving.l4_study)")
    arena_serving = install()
    nest_overlay(study, Path(args.study).parent.parent)
    os.environ[OPTIONS_ENV] = json.dumps(study["arc2"]["l4x4"])
    sys.argv = [sys.argv[0], "--working", args.working, "--input-root", args.input_root, "--study", args.study]
    try:
        arena_serving.main()
    except RuntimeError as exc:
        if "cache layouts disagree" not in str(exc):
            raise
        # The server is up; the ranks' pools differ. The trial takes the smallest (arc2.trial.block_budget).
        from .trial import block_budget

        budget = block_budget(args.working, study["kv_margin"], smallest=True)
        (Path(args.working) / "cache-layout.json").write_text(json.dumps(budget.layout, indent=2))
        (Path(args.working) / "l4-layout-note.json").write_text(json.dumps({"note": str(exc)}, indent=1))


def serve_candidate(args):
    """``python -m arena.candidate_serving`` through the L4 changes: four cards, tensor parallel 4, 64K context."""
    import dataclasses

    study = json.loads(Path(args.study).read_text())
    if study["arc2"].get("l4x4", {}).get("model") != study.get("candidate_profile"):
        raise SystemExit("not an L4 candidate study (arc2.l4_serving.candidate_study)")
    os.environ[OPTIONS_ENV] = json.dumps(study["arc2"]["l4x4"])
    install()  # before the import below: it binds the arena's configure, hardware check and readiness check
    import arena.candidate_serving as candidates

    from .trial import block_budget

    from_study = candidates.from_study
    candidates.from_study = lambda s: dataclasses.replace(from_study(s), native_context=CONTEXT)

    class Budget:  # the ranks' pools may differ: the smallest, as the trial takes it
        @staticmethod
        def load(working, margin):
            return block_budget(working, float(margin), smallest=True)

    candidates.BlockBudget = Budget
    config = candidates.Config  # its default history and compaction sizes belong to the 262,144 window
    candidates.Config = lambda **kw: config(**{"history_tokens": None, "compaction_tokens": None, **kw})
    # The server takes the study's slot count; the load may keep fewer replies open.
    candidates.start({**study, "generation_slots": study["serving_slots"]}, args.working, args.input_root)


def load(args):
    import asyncio
    import time

    from .trial import run_arc2

    study = json.loads(Path(args.study).read_text())
    trial = study["trials"][0]
    report = asyncio.run(run_arc2(study, trial, source=args.source, output=args.output, server_dir=args.server_dir,
                                  base_url=args.base_url, deadline_epoch=time.time() + float(trial["wall_seconds"]),
                                  python=sys.executable))
    print(json.dumps({"event": "arc2_l4_load", "status": report["status"], "errors": report.get("task_errors")}),
          flush=True)
    raise SystemExit(0 if report["status"] == "complete" else 1)


def degenerate(text, completion_tokens, content_chars=None):
    """A reply that is not text: "empty" (it stopped within a few tokens and said nothing) or "loop" (its tail
    is one short pattern repeated: zlib leaves under 5% of 3,000 characters, under 20% of a short reply)."""
    import zlib

    text = text or ""
    if completion_tokens is not None and completion_tokens <= 30 and not (content_chars if content_chars is not None
                                                                         else len(text.strip())):
        return "empty"
    tail = text[-3000:].encode()
    if len(tail) < 300:
        return None
    return "loop" if len(zlib.compress(tail)) / len(tail) < (0.05 if len(tail) >= 2500 else 0.2) else None


def reply_text(body):
    """A chat completion's usage, message and verdict of ``degenerate`` on reasoning + content."""
    data = json.loads(body)
    used, message = data.get("usage") or {}, data["choices"][0]["message"]
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    content = message.get("content") or ""
    kind = degenerate(reasoning + content, used.get("completion_tokens"),
                      len(content.strip()) + len(reasoning.strip()))
    return used, message, kind


def finite_report(server_dir, limit=30):
    """The server log's ``ARC2_FINITE`` lines (``arc2.l4_finite``) and the cache check's NaN lines, by worker:
    events in time order (the first is where a non-finite value appeared), odd expert tensors at load, the last
    summary of each worker."""
    import math
    import re

    path = Path(server_dir) / "vllm-openai-server.log"
    events, load, summaries, errors, workers, installed = [], [], {}, [], set(), {}
    nan = {"lines": 0, "nan": 0, "first_layer": None}
    for line in path.read_text(errors="replace").splitlines() if path.is_file() else ():
        worker = (re.match(r"\((\w+)", line) or [None, "?"])[1]
        if "ARC2_FINITE_ERROR" in line:
            errors.append(f"{worker}: {line.split('ARC2_FINITE_ERROR', 1)[1].strip()[:300]}")
        elif "ARC2_FINITE " in line:
            try:
                row = json.loads(line[line.index("{"):])
            except ValueError:
                continue
            workers.add(worker)
            if row["kind"] in ("moe", "input", "attention", "query"):
                events.append({"worker": worker, **row})
            elif row["kind"] == "summary":
                summaries[worker] = row
            elif row["kind"] in ("installed", "guard"):
                installed.setdefault(worker, {}).update(row)
            elif row["kind"] == "load":
                odd = {name: t for name, t in row["tensors"].items()
                       if t.get("nonfinite") or t.get("error") or (t.get("zero") and name in ("w1", "w2"))}
                if odd or row["layer"] == 0:
                    load.append({"worker": worker, "layer": row["layer"], "odd": bool(odd),
                                 "tensors": odd or row["tensors"]})
        elif "ARC_KV4_ROUNDTRIP" in line:
            try:
                row = json.loads(line[line.index("{"):])
            except ValueError:
                continue
            nan["lines"] += 1
            value = row.get("k_peak_to_rms")
            if isinstance(value, float) and math.isnan(value):
                nan["nan"] += 1
                first = nan["first_layer"]
                nan["first_layer"] = row["layer_order"] if first is None else min(first, row["layer_order"])
    events.sort(key=lambda row: row["t"])
    moe = [row for row in events if row["kind"] in ("moe", "attention")]
    return {"workers": sorted(workers), "installed": installed, "moe_events": len(moe),
            "input_events": len(events) - len(moe),
            "first": events[0] if events else None, "events": events[:limit], "load": load[:limit],
            "summaries": summaries, "errors": errors[:10], "kv_roundtrip": nan}


def garbage_probe(base_url, study, source, *, phases=((20, 600),), tokens=200, turns=None, replies=None):
    """Garbage replies under prefill load: real first turns, and second turns whose first reply is a stored sound
    one (``turns``: {task: {"reasoning", "content"}}), so every request prefills its whole prompt. ``phases``:
    (streams, seconds) in order. Each reply is cut at ``tokens`` and read by ``degenerate``; a row per phase and
    turn: requests, prompt tokens, the degenerate ones by kind, errors. The degenerate replies' times and tails are
    kept for the server log; ``replies``: a JSONL file that takes every reply whole (training tasks)."""
    import threading
    import time

    from . import tasks

    spec, trial = study["arc2"], study["trials"][0]
    chat = base_url.rstrip("/") + "/chat/completions"
    model = json.loads(post(base_url.rstrip("/") + "/models", timeout=60)["body"])["data"][0]["id"]
    fields = {"temperature": (spec.get("settings") or {}).get("temperature") or float(study["temperature"]),
              "effort": (spec.get("settings") or {}).get("effort") or study["reasoning_effort"]}
    task_map = tasks.load(Path(source) / spec["data"] / trial["split"], trial["tasks"])
    stored = json.loads(Path(turns).read_text()) if turns else {}
    items = []
    for name in trial["tasks"]:
        first = [{"role": "system", "content": tasks.SYSTEM},
                 {"role": "user", "content": tasks.task_prompt(task_map[name])}]
        items.append((name, 1, first))
        if name in stored:
            entry = {"role": "assistant", "content": stored[name].get("content") or ""}
            if stored[name].get("reasoning"):
                entry["reasoning"] = stored[name]["reasoning"]
            items.append((name, 2, [*first, entry, {"role": "user", "content": tasks.TRUNCATED}]))
    warm = post(chat, {"model": model, "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                       "max_tokens": 16}, timeout=240)  # the kernels compiled at first use
    lock, cursor, rows, found, out = threading.Lock(), [0], {}, [], []
    whole = open(replies, "a") if replies else None

    def worker(phase, streams, deadline):
        while time.time() < deadline:
            with lock:
                name, turn, messages = items[cursor[0] % len(items)]
                cursor[0] += 1
            begun = time.time()
            reply = post(chat, tasks.chat_request(model, messages, tokens, **fields), timeout=900)
            error = None if reply["status"] == 200 else f"{reply['status']}: {reply['body'][:300]}"
            if error is None:
                try:
                    used, message, kind = reply_text(reply["body"])
                except Exception as exc:  # noqa: BLE001 - a body that is not a completion
                    error = repr(exc)[:300]
            with lock:
                row = rows.setdefault((phase, turn), {"phase": phase, "streams": streams, "turn": turn, "requests": 0,
                                                      "prompt_tokens": 0, "degenerate": {}, "errors": 0})
                if error is not None:
                    row["errors"] += 1
                    row.setdefault("first_error", error)
            if error is not None:
                time.sleep(2)
                continue
            with lock:
                if whole is not None:
                    choice = json.loads(reply["body"])["choices"][0]
                    whole.write(json.dumps({
                        "phase": phase, "task": name, "turn": turn, "sent": round(begun, 2),
                        "answered": round(time.time(), 2), "prompt_tokens": used.get("prompt_tokens"),
                        "completion_tokens": used.get("completion_tokens"), "finish": choice.get("finish_reason"),
                        "kind": kind, "reasoning": message.get("reasoning") or message.get("reasoning_content") or "",
                        "content": message.get("content") or ""}) + "\n")
                    whole.flush()
                row["requests"] += 1
                row["prompt_tokens"] += used.get("prompt_tokens") or 0
                if kind:
                    row["degenerate"][kind] = row["degenerate"].get(kind, 0) + 1
                    text = (message.get("reasoning") or message.get("reasoning_content") or "") + (message.get("content") or "")
                    found.append({"phase": phase, "task": name, "turn": turn, "kind": kind, "sent": round(begun, 2),
                                  "answered": round(time.time(), 2), "prompt_tokens": used.get("prompt_tokens"),
                                  "completion_tokens": used.get("completion_tokens"), "tail": text[-160:]})

    for phase, (streams, seconds) in enumerate(phases, 1):
        begun = time.time()
        threads = [threading.Thread(target=worker, args=(phase, streams, begun + seconds), daemon=True)
                   for _ in range(streams)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(seconds + 900)
        out.append({"phase": phase, "streams": streams, "seconds": round(time.time() - begun)})
    table = [rows[key] for key in sorted(rows)]
    return {"phases": out, "tokens": tokens, "second_turns": len(stored), "warm_status": warm["status"],
            "rows": table, "requests": sum(r["requests"] for r in table),
            "prompt_tokens": sum(r["prompt_tokens"] for r in table),
            "degenerate": sum(sum(r["degenerate"].values()) for r in table),
            "errors": sum(r["errors"] for r in table), "found": found[:60]}


def garbage_probe_command(args):
    study = json.loads(Path(args.study).read_text())
    phases = tuple(tuple(int(x) for x in part.split(":")) for part in args.phases.split(","))
    found = garbage_probe(args.base_url, study, args.source, phases=phases, turns=args.turns, tokens=args.tokens,
                          replies=args.replies)
    Path(args.output).write_text(json.dumps(found, indent=1))
    print(json.dumps({"event": "arc2_l4_garbage_probe", **{k: found[k] for k in (
        "requests", "prompt_tokens", "degenerate", "errors")}}), flush=True)


def cache_probe(base_url, study, source, *, gaps=(2, 30, 90, 240), background=20, background_tokens=4096,
                first_tokens=512, second_tokens=160):
    """Does the second turn of a conversation hit the prefix cache, and after how long a pause? Two-turn
    conversations on real task prompts: turn 1 (cut at ``first_tokens``), a pause, then turn 2 = the same messages
    plus the reply and the loop's follow-up. One pair on the idle engine, then two pairs per pause in ``gaps``
    while ``background`` other conversations keep generating (replies of ``background_tokens``, a new task each
    time), which is what recycles freed blocks in the roster loop. A row: the pause and turn 2's prompt, cached
    and uncached tokens. Every reply's text is read (``degenerate``): the probe passes only when the short pauses
    are served from the cache and no reply, a background one included, is garbage; cached tokens alone say
    nothing about what a cached turn then writes."""
    import threading
    import time

    from . import tasks

    spec, trial = study["arc2"], study["trials"][0]
    root, chat = base_url.rstrip("/").rsplit("/v1", 1)[0], base_url.rstrip("/") + "/chat/completions"
    model = json.loads(post(base_url.rstrip("/") + "/models", timeout=60)["body"])["data"][0]["id"]
    fields = {"temperature": (spec.get("settings") or {}).get("temperature") or float(study["temperature"]),
              "effort": (spec.get("settings") or {}).get("effort") or study["reasoning_effort"]}
    task_map = tasks.load(Path(source) / spec["data"] / trial["split"], trial["tasks"])
    names, rows, lock, stop = list(trial["tasks"]), [], threading.Lock(), threading.Event()
    text_check = {"replies": 0, "degenerate": 0, "kinds": {}}

    def read(kind):
        with lock:
            text_check["replies"] += 1
            if kind:
                text_check["degenerate"] += 1
                text_check["kinds"][kind] = text_check["kinds"].get(kind, 0) + 1
        return kind

    post(chat, {"model": model, "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                "max_tokens": 16}, timeout=240)  # the kernels compiled at first use

    def first(name):
        return [{"role": "system", "content": tasks.SYSTEM}, {"role": "user", "content": tasks.task_prompt(task_map[name])}]

    def usage(reply):
        used, message, kind = reply_text(reply["body"])
        return used, message, (used.get("prompt_tokens_details") or {}).get("cached_tokens"), read(kind)

    def pair(name, gap, label):
        row = {"task": name, "gap_seconds": gap, "background": label}
        try:
            messages = first(name)
            one = post(chat, tasks.chat_request(model, messages, first_tokens, **fields), timeout=600)
            used, message, cached, kind = usage(one)
            row.update(turn1={"prompt_tokens": used.get("prompt_tokens"), "cached_tokens": cached,
                              "completion_tokens": used.get("completion_tokens"), "seconds": one["seconds"],
                              "degenerate": kind})
            entry = {"role": "assistant", "content": message.get("content") or ""}
            if message.get("reasoning") or message.get("reasoning_content"):
                entry["reasoning"] = message.get("reasoning") or message.get("reasoning_content")
            time.sleep(gap)
            two = post(chat, tasks.chat_request(model, [*messages, entry, {"role": "user", "content": tasks.TRUNCATED}],
                                                second_tokens, **fields), timeout=600)
            used, _, cached, kind = usage(two)
            row.update(turn2={"prompt_tokens": used.get("prompt_tokens"), "cached_tokens": cached,
                              "uncached_tokens": None if cached is None else used.get("prompt_tokens") - cached,
                              "completion_tokens": used.get("completion_tokens"), "seconds": two["seconds"],
                              "degenerate": kind})
        except Exception as exc:  # noqa: BLE001 - a row with its error
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        with lock:
            rows.append(row)

    quiet = 1 + 2 * len(gaps)  # the pairs' tasks: no background conversation shares their prompts
    if len(names) < quiet + 2:
        raise ValueError("the cache probe needs more tasks than its pairs")

    def busy(k):
        at = k
        while not stop.is_set():
            reply = post(chat, tasks.chat_request(model, first(names[at % (len(names) - quiet)]), background_tokens,
                                                  **fields), timeout=900)
            try:
                read(reply_text(reply["body"])[2])
            except Exception:  # noqa: BLE001 - a request the stop cut, or an error body
                pass
            at += background

    post(root + "/reset_prefix_cache", method="POST", timeout=60)
    pair(names[-1], 2, 0)
    workers = [threading.Thread(target=busy, args=(k,), daemon=True) for k in range(background)]
    for worker in workers:
        worker.start()
    time.sleep(45)  # the background's first turns are in and decoding
    spare = names[::-1][1:]  # tasks the background reaches last
    pairs = [threading.Thread(target=pair, args=(spare[i], gap, background), daemon=True)
             for i, gap in enumerate(g for g in gaps for _ in range(2))]
    for thread in pairs:
        thread.start()
    for thread in pairs:
        thread.join(max(gaps) + 600)
    stop.set()
    text = post(root + "/metrics", timeout=30)["body"]
    short = [r for r in rows if r["gap_seconds"] <= 30]
    cached = bool(short) and all((r.get("turn2") or {}).get("cached_tokens") is not None
                                 and 2 * r["turn2"]["cached_tokens"] >= r["turn2"]["prompt_tokens"] for r in short)
    verdict = {"cached_within_30s": cached, "text_ok": not text_check["degenerate"],
               "errors": sum("error" in r for r in rows)}
    verdict["passed"] = cached and verdict["text_ok"] and not verdict["errors"]
    return {"gaps": list(gaps), "background": background, "background_tokens": background_tokens,
            "verdict": verdict, "text": text_check,
            "rows": sorted(rows, key=lambda r: (r["background"], r["gap_seconds"])),
            "server": {"prefix_cache_queries": metric(text, "vllm:prefix_cache_queries_total"),
                       "prefix_cache_hits": metric(text, "vllm:prefix_cache_hits_total")}}


def cache_probe_command(args):
    study = json.loads(Path(args.study).read_text())
    found = cache_probe(args.base_url, study, args.source)
    Path(args.output).write_text(json.dumps(found, indent=1))
    print(json.dumps({"event": "arc2_l4_cache_probe", "verdict": found["verdict"], "text": found["text"], "rows": [
        [r["background"], r["gap_seconds"], (r.get("turn2") or {}).get("prompt_tokens"),
         (r.get("turn2") or {}).get("cached_tokens"), r.get("error")] for r in found["rows"]]}), flush=True)


def probe_command(args):
    from .trial import block_budget

    study = json.loads(Path(args.study).read_text())
    budget = block_budget(args.server_dir, float(study["kv_margin"]), smallest=True)
    found = probe(args.base_url, study, args.source, server_dir=args.server_dir, budget=budget,
                  budget_seconds=args.budget_seconds)
    Path(args.output).write_text(json.dumps(found, indent=1))
    print(json.dumps({"event": "arc2_l4_probe", **{key: found.get(key) for key in (
        "passed", "top", "stopped", "health", "gate")}}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("serve", "serve-candidate", "load", "probe", "cache-probe",
                                            "garbage-probe"))
    parser.add_argument("--phases", default="20:600")
    parser.add_argument("--turns")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--replies")
    parser.add_argument("--budget-seconds", type=float, default=360)
    parser.add_argument("--study", required=True)
    parser.add_argument("--working")
    parser.add_argument("--input-root")
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--server-dir")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    args = parser.parse_args()
    {"serve": serve, "serve-candidate": serve_candidate, "load": load, "probe": probe_command,
     "cache-probe": cache_probe_command, "garbage-probe": garbage_probe_command}[args.command](args)


if __name__ == "__main__":
    main()
