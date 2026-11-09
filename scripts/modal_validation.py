"""Budgeted Modal smoke validation, independent of Kaggle queue state.

The arena uses its offline bundle; SGLang uses the digest-pinned original Pennyroyal
image rather than re-extracting that image on Kaggle. Neither path changes gameplay.
Set ARC_VALIDATION_KIND and ARC_VALIDATION_BUNDLE before invoking Modal.
"""

import json
import os
import time
from pathlib import Path

import modal

# The ledger (docs/spend.json) and launch records live under ROOT; ARC_VALIDATION_ROOT points a launch from a
# git worktree at the main checkout's ledger (27 September: one ledger for every session).
ROOT = Path(os.environ.get("ARC_VALIDATION_ROOT") or Path(__file__).resolve().parents[1]).resolve()
KIND = os.environ.get("ARC_VALIDATION_KIND", "arena")
if KIND not in {"arena", "sglang", "vllm"}:
    raise ValueError("Expected arena, sglang or vllm")
TEST = os.environ.get("ARC_VALIDATION_TEST", "smoke")


HX4_TRANCHE_USD = 200.0  # harness arm, 7 October tranche (run ids hx4-)


def hx4_study():
    bundle = (Path(os.environ.get("ARC_VALIDATION_BUNDLE", "")).resolve() if modal.is_local()
              else Path("/opt/arc-validation"))
    return json.loads((bundle / "configs/comparison.json").read_text())


def validation_timeout(kind, test):
    if kind == "arena" and test in ("hx4-level", "hx4-quality", "rl0-level", "rl0-play"):
        # Stuck-level bench (harness arm, 7 October): the bundle's notebook_seconds (setup + bench wall) + 300 s.
        # hx4-quality: a copy of a throughput-arm quality-protocol bundle (quality_eval version 2, DEV rep4) with
        # harness flags changed; same rule as that arm's quality-fixed-replicated (notebook + 300 s, 8 CPUs).
        return min(14400, int(hx4_study()["notebook_seconds"]) + 300)
    choices = {
        ("vllm", "duck-transfer-mxfp8"): 2100,
        ("arena", "smoke"): 1680,
        ("arena", "campaign-9h"): 36000,
        # Offline test of a Kaggle submission archive (no Kaggle GPU quota, so no public preview on
        # Kaggle): the unmodified competition bundle on its saved-version branch, i.e. setup (raised int3 timeouts:
        # ready <= 2,700 s, setup <= 3,600 s), the 300 s public preview on 110 clones and teardown. Bound
        # (4,500 + 60) s at the ledger rate x 1.15 =.
        ("arena", "competition-preview"): 4500,
        # The same with the default one-hour public preview (the dfr-combined bundle's Kaggle start
        # timed out; test the identical bundle on Modal): the bundle's notebook_seconds 7,500 + 300 s.
        ("arena", "competition-preview-1h"): 7800,
        # Four-hour preview (4 h runs of v3 and of the 256K + compaction variant): the bundle's
        # notebook_seconds (3,900 + 14,400 for int3 builds) + 300 s.
        ("arena", "competition-preview-4h"): 18600,
        # Eight-hour preview (v6 for 8 h): notebook_seconds (3,900 + 28,800) + 300 s.
        ("arena", "competition-preview-8h"): 33000,
        ("arena", "research-model"): 3600,
        ("arena", "full-duck-transfer"): 2100,
        ("arena", "cache-duck-transfer"): 2700,
        ("arena", "paired-30m"): 8100,
        # Duck-only 25-game / 1,800 s arena (setup ~700 s, probes ≤ 300 s, gameplay 1,800 s,
        # teardown); the corrected-clock control after the 10-minute Kaggle queue rule.
        ("arena", "duck-30m"): 4500,
        # The same with 3,600 s of gameplay (23-game roster, one hour).
        ("arena", "duck-60m"): 6300,
        # Short probes: a few games for 900 s beside the one-hour Kaggle runs.
        ("arena", "duck-probe"): 2700,
        # One-hour arena on 110 cyclic clones (20 games x 5-6 clones); setup for 110
        # interpreters is longer than for 23 games.
        ("arena", "duck-110"): 7200,
        # hero-v2: the 110-clone arena for 7,200 s on the dedicated hero-v2 GPU slot. The one-hour
        # hero hours ran ~550 s beside their 3,600 s of play; the GPTQ overlay check adds ~190 s; 9,600 s leaves
        # ~1,700 s for setup before the bootstrap's trial budget (wall + 300 s) would skip the trial..
        ("arena", "hero-v2-2h"): 9600,
        # hero-v2 follow-ups (docs/validation-20260927/hero-v2-next-runs.md), the same envelope on the
        # hero-v2 slot: run 1 at 128K windows on the int4 attention KV, run 2 at 64K with 30 streams under the exact
        # gate and the backlog fix. Both stop gracefully at 1 h when their score beats hero-v2's 1.68% (early_stop), so
        # the reconciled then;.
        ("arena", "hero-v2-128k-int4"): 9600,
        ("arena", "hero-v2-streams"): 9600,
        # (two parallel 2 h runs, no early stop; a third GPU as a temporary exception): run A = int4
        # KV at 64K with 38 oversubscribed focus streams (max_num_seqs 40, graphs to 40); run B = int4 KV at 130,816
        # with 24 oversubscribed streams; both with harness_fixes and the progress-fruitless backlog. ~705 s setup +
        # 7,200 s + teardown in 9,600 s (arena.timeline recorded by the build).
        ("arena", "hero-v2-int4-streams"): 9600,
        ("arena", "hero-v2-128k-streams"): 9600,
        # hero-v2-blocks: the three-block scheduler with a 30,600 s scheduler wall, stopped
        # unconditionally at 7,200 s of play (early_stop always); run B's envelope: ~705 s setup + 7,200 s + teardown.
        ("arena", "hero-v2-blocks"): 9600,
        # Run B + parallel_samples + action_plans (docs/validation-20260927/parallel-samples-plans.md):
        # run B's envelope (~705 s setup + 7,200 s + teardown) with max_num_seqs 48 / graphs to 48 for the siblings.
        ("arena", "hero-v2-samples-plans"): 9600,
        # hero-v2 smoke: one server start with every chosen component (~705 s to the first trial: the hero hours took
        # ~515 s, the overlay check ~190 s), the check trial (~200 s), a 900 s soak on the clone roster (the bootstrap
        # skips it unless it starts by ~1,080 s). (arena.hero_v2, scripts/build_arena.py --hero-v2-smoke).
        ("arena", "hero-v2-smoke"): 2460,
        # hero-v2 pre-check ("Yes, both"; test slot): one start of hero-v2's serving at
        # max_num_seqs 32 with --prefix-match-unit 16 (~665-705 s to the trial), the decode / mixed stream sweep
        # (~330 s), the teacher-forced partial-hit check (~100 s) and the prefix replay (~90 s); the bench stops at
        # the end - 180 s. Bound (1,560 + 60) s x ledger rate x 1.15 =.
        ("arena", "hero-v2-precheck"): 1560,
        # prefix-unit-int4 ("let's test the 3 priority items"; test slot, <=): one start of the
        # int4 hero-v2 serving at 48 sequences / graphs to 48 with --prefix-match-unit 16 (~650 s), a 48-stream decode
        # check (~80 s), the teacher-forced Latin square (~400 s) and the unit-aware replay (~100 s); the bench stops
        # at the end - 180 s. Bound (1,740 + 60) s x ledger rate x 1.15 =.
        ("arena", "prefix-unit-int4"): 1740,
        # v2: the teacher-forced square alone (v1's warm arms were lost to a client length check); setup ~600 s, the
        # square ~200 s. Bound (1,440 + 60) s x ledger rate x 1.15 =.
        ("arena", "prefix-unit-int4-tf"): 1440,
        # Decode sweep: the blocks3 campaign serving at 56 sequences with a
        # finer CUDA-graph ladder (~600 s setup, ~20 min of points). Bound (2,400 + 60) s x ledger rate x 1.15.
        ("arena", "decode-sweep"): 2400,
        # Speculative-decoding pilot ("do the speculative decoding analysis fast"; research pilot
        # <= 1 h, <=): one MTP arm (0 / 1 / 2) of the blocks3 serving on recorded arena contexts (~600-900 s
        # setup, ~10 min of points). Bound (2,400 + 60) s x ledger rate x 1.15 =.
        ("arena", "spec-sweep"): 2400,
        # Kernel work, phase 2: the kernel quality gate (control reference, patched
        # restart, ~20 minutes with both server starts) before the same one-hour 110-clone arena.
        ("arena", "duck-110-kernels"): 7800,
        # Serving-only throughput grid (setup ~700 s, probes, ~20 min of fixed requests).
        ("arena", "bench"): 3600,
        # Kernel work, phase 4: one server start of the MTP-3 admission arm and a short
        # grid; the user's for the job.
        ("arena", "kernel-smoke"): 1500,
        # Serving tweaks: the same one-server smoke on the tweaks arm (16K draft head, PLE
        # read over UVA, host-resident embedding); the user's.
        ("arena", "tweaks-smoke"): 1500,
        # Kernel work, phase 5 (the stack, <= of GPU): the MTP-3 stack server (host-copy
        # item toggled at run time on the same points) and the MTP-0 kv8 stack server restarted in place.
        ("arena", "kernel-stack"): 2400,
        # Draft-head fine-tune: one eager server start, hidden-state dumps, then the
        # trainer on the same GPU (scripts/build_arena.py DRAFT_HEAD_*; pilot).
        ("arena", "draft-head"): 3200,
        # 8-bit QSA KV cache: one server start of the kv8 arm, the
        # served layout, drift against the recorded BF16-KV reference, decode points and the QSA microbenchmark;
        # the user's for the job ((1500 + 60) s at the ledger rate is).
        ("arena", "kv8-smoke"): 1500,
        # Hero smoke: one MTP-0 server start of the hero arm (layout, drift, 1/6/14-stream
        # decode, profiles at 1 and 14 streams); ((1500 + 60) s at the ledger rate is).
        ("arena", "hero-smoke"): 1500,
        # Kernel work, phase 6: the hero (control) and the phase-6 arm (UVA host
        # tables + fused split-K) restarted in place: decode at 1 / 6 / 14 streams, mixed decode + injected prefill at
        # 14 and 21, drift on both, decode and mixed-step profiles, and the kernel microbenchmarks on the second
        # server; bound (3300 + 60) s at the ledger rate =.
        ("arena", "kernel-phase6"): 3300,
        # Part 2 (v1's trials hit the per-trial timeout in the mixed-step profile): one p6 server, the mixed-step
        # profiles and the microbenchmarks; bound (2400 + 60) s at the ledger rate =.
        ("arena", "kernel-phase6b"): 2400,
        # Prefix cache v2 smoke (29 September, not yet approved): two hero server starts (pinned and patched prefix
        # cache) with warm/cold correctness and a 24-game replay under a cut pool; (2700 + 60) s is.
        ("arena", "prefix-cache-smoke"): 2700,
        # Prefix cache v3 smoke (item 3, test slot): hero-v2 serving with the request-end
        # tail, int4 KV then int8 KV restarted in place (~705 s setup, ~280 s of checks each, a >= 614 s restart;
        # arena.timeline checked at build). Bound (2,460 + 60) s x ledger rate x 1.15 =.
        ("arena", "prefix-cache-v3-smoke"): 2460,
        # Its recheck: the teacher-forced tail check alone at int4, one server start (~705 s
        # setup, ~200 s of checks). Bound (1,460 + 60) s x ledger rate x 1.15 =.
        ("arena", "prefix-cache-v3-recheck"): 1460,
        # Serving A/B bench: eight server starts and bench passes in one container.
        ("arena", "bench-ab"): 6600,
        # Offline harness evolution: setup, a baseline 30-minute round, up to five propose/smoke/
        # evaluate rounds and two held-out rounds.
        ("arena", "evolve"): 28800,
        # One-round rehearsal of the evolution driver on 300 s evaluations (an
        # 8-hour evolution with no visibility is not acceptable); every driver path in under 2 hours.
        ("arena", "evolve-rehearsal"): 7200,
        # Level bench (27 September): setup (~70 s weights, ~8-minute server start, cache probe) plus a
        # bench wall the bench sizes to what is left (~40 minutes of attempts), drain and teardown.
        ("arena", "level-bench"): 3600,
        # Expert-usage measurement: one server start with the probe overlay
        # (~480 s of setup), the usage pass and the held-out passes until the deadline. v2:
        # 2,100 s so that all four masks fit;. v3: the eager measurement
        # arm plus two server-start diagnostic arms;.
        ("arena", "expert-usage"): 2760,
        # v4: the eager measurement arm alone (v3 showed it starts);.
        ("arena", "expert-usage-measure"): 1920,
        # Load-time expert pruning GPU check (not yet approved): three hero server starts (unpruned, 12.5%, 25%)
        # with held-out teacher forcing;.
        ("arena", "expert-prune-check"): 2100,
        ("arena", "expert-prune-levels"): 3000,
        # The prune12 hang test (not yet approved): two prune12 server arms sweeping every KV block id;.
        ("arena", "expert-prune-hang"): 2400,
        # ARC expert quantization, stage 1: one server start with the capture overlay
        # (~710 s of setup), ~100K tokens of capture, then the microbenchmark after the server stops.
        ("arena", "expert-quant-bench"): 1200,
        # Stage 2 (<= for the stage): capture on every layer, then GPTQ of the 48 layers; and the end-to-end check
        # (served reference plus report-only gates; the study's kernel_timeout sizes it).
        ("arena", "expert-quant-calibrate"): 3300,
        # Phase 3 (29 September; <=): the v2 recalibration (4x calibration data, pooled prior, act-order).
        ("arena", "expert-quant-calibrate-v2"): 2400,
        ("arena", "expert-quant-eval"): 6300,
        # Long-context drift test ("yes, to the long-context drift test", <=): kernels-v2
        # reference + repeat, the hero serving and int8 KV alone on 39-44K-token contexts; three server starts
        # (~650 + 2 x ~550 s) and ~550 s of teacher forcing. Bound (3,300 + 60) s x ledger rate x 1.15 =.
        ("arena", "drift-long"): 3300,
        # 4-bit attention KV drift test: hero-v2's serving (int8 KV) as the
        # reference with its repeat, then int4 and k8v4 KV (only the KV mode changes) on the held-out ARC requests and
        # the 39-44K drift inputs; three server starts. Bound (3,000 + 60) s x ledger rate x 1.15 =.
        ("arena", "kv4-drift"): 3000,
        # Holo4-27B on the hero-v2 arena setup: the candidate smoke (BF16-KV reference and
        # FP8-KV drift on pinned pools, the arena server's check and a 600 s clone-roster soak; three server starts; the
        # soak must start by ~1,620 s), bound (2,700 + 60) s x ledger rate x 1.15 =,; then the
        # one-hour 110-clone arena: setup, the 3,600 s wall, the 600 s cleanup reserve (the trial must start by
        # ~1,020 s), bound (5,280 + 60) s x rate x 1.15 =,.
        ("arena", "holo-smoke"): 2700,
        ("arena", "holo-60m"): 5280,
        # kv8 prefill check (kernel work, phase 7; <= test slot): hero-v2's serving as the
        # reference + repeat, then the same with serving_kernels.kv8_prefill "bf16" (report-only gate), with the
        # injected-prefill bench at 14 / 21 / 23 streams on both; two server starts. Bound (2,400 + 60) s =.
        ("arena", "kv8-prefill-check"): 2400,
        # Holo4 soak (soak first, then decide on the hour): one start (604 s measured) and a 720 s
        # clone-roster soak; build_arena.candidate_soak_seconds(720) = 1,920 s. Bound (1,920 + 60) s x rate x 1.15 =
        #, (~1,550 s of app lifetime).
        ("arena", "holo-soak"): 1920,
        ("sglang", "smoke"): 3600,
        ("sglang", "duck-transfer"): 2100,
    }
    if (kind, test) not in choices:
        raise ValueError(f"Unsupported validation kind/test: {kind}/{test}")
    return choices[kind, test]


def arena_arguments(test):
    if test == "research-model":
        return []
    if test == "campaign-9h":
        return ["--trial", "duck-transfer"]
    if test in ("competition-preview", "competition-preview-1h", "competition-preview-4h", "competition-preview-8h"):
        return ["--trial", "duck-transfer"]
    if test == "smoke":
        return ["--smoke"]
    if test in {"full-duck-transfer", "cache-duck-transfer", "duck-30m", "duck-60m", "duck-probe", "duck-110",
                "hero-v2-2h", "hero-v2-128k-int4", "hero-v2-streams", "hero-v2-int4-streams",
                "hero-v2-128k-streams", "hero-v2-blocks", "hero-v2-samples-plans"}:
        return ["--trial", "duck-transfer"]
    if test == "hero-v2-smoke":
        return []  # the check trial, then the soak on the same server
    if test == "holo-smoke":
        return []  # the BF16-KV reference, the FP8-KV gate, the check, then the soak (three server starts)
    if test in ("holo-60m", "holo-soak"):
        return ["--trial", "duck-transfer"]
    if test in ("prefix-unit-int4", "prefix-unit-int4-tf"):
        return ["--trial", "prefix-int4"]
    if test == "decode-sweep":
        return ["--trial", "sweep"]
    if test == "spec-sweep":
        return ["--trial", "spec"]
    if test == "hero-v2-precheck":
        return ["--trial", "precheck"]
    if test == "duck-110-kernels":
        return []  # kernel-ref, kernel-gate, then duck-transfer (skipped unless the gate passed)
    if test in ("bench", "kernel-smoke", "tweaks-smoke", "kv8-smoke"):
        return ["--trial", "serving-bench"]
    if test in ("expert-usage", "expert-usage-measure", "expert-prune-check", "expert-prune-hang",
                "expert-prune-levels"):
        return []  # the measurement arm, then the server-start diagnostic arms
    if test == "hero-smoke":
        return ["--trial", "hero"]
    if test == "kernel-stack":
        return []  # the MTP-3 stack arm, then the MTP-0 kv8 stack arm (server restart in place)
    if test == "kernel-phase6":
        return []  # the hero (control), then the phase-6 arm (server restart in place)
    if test == "kernel-phase6b":
        return ["--trial", "p6b"]
    if test == "prefix-cache-smoke":
        return []  # the pinned-cache arm, then the patched arm (server restart in place)
    if test == "prefix-cache-v3-smoke":
        return []  # the int4-KV arm, then the int8-KV arm (server restart in place)
    if test == "prefix-cache-v3-recheck":
        return ["--trial", "tail-int4"]
    if test == "draft-head":
        return ["--trial", "draft-head"]
    if test in ("expert-quant-bench", "expert-quant-calibrate", "expert-quant-calibrate-v2"):
        return ["--trial", "expert-quant"]
    if test in ("expert-quant-eval", "drift-long", "kv4-drift", "kv8-prefill-check"):
        return []
    if test == "bench-ab":
        return []
    if test == "paired-30m":
        return []  # both agents, each receives the full independent 1800 seconds
    if test in ("evolve", "evolve-rehearsal"):
        return ["--evolve"]
    if test in ("level-bench", "hx4-level"):
        return ["--trial", "level-bench"]
    if test == "hx4-quality":
        return ["--trial", "duck-transfer"]
    if test == "rl0-level":  # RL restart step 0 (11 October): the level bench, then the training-cost probe
        return ["--trial", "level-bench", "--trial", "train-probe"]
    if test == "rl0-play":  # RL step 0 on a model too large for the one-GPU training probe (12 October): play only
        return ["--trial", "level-bench"]
    raise ValueError(f"Unsupported arena test: {test}")


TIMEOUT = validation_timeout(KIND, TEST)
# Container CPU ceiling (cores). 8 everywhere, 16 for the oversubscribed follow-ups (29 September): 38 / 24 concurrent
# Duck streams on 110 sandboxes beside the vLLM API server and engine, against 23-25 in every earlier run; no host CPU
# telemetry exists to show 8 is enough, and the ceiling adds to a 2 h bound.
CPU_CEILING = 16 if TEST in ("hero-v2-int4-streams", "hero-v2-128k-streams", "hero-v2-blocks", "hx4-level") else 8
if TEST in ("hero-v2-samples-plans", "campaign-9h"):
    CPU_CEILING = 16  # run B's ceiling: 24 focus streams (+ their sibling requests) on 110 sandboxes
# Manual graceful stop (arena.early_stop): the runs-volume request and the local file the trial checks.
STOP_FILE = "stop-request"
STOP_LOCAL = "/tmp/arena-stop-request"
STOP_POLL_SECONDS = 30
BUNDLE = (Path(os.environ.get("ARC_VALIDATION_BUNDLE", "dist/arena-25g-900s-v2")).resolve()
          if modal.is_local() else Path("/opt/arc-validation"))
PENNYROYAL = "ghcr.io/jpezzulli/sglang-rtxpro6000@sha256:7103728d3d5568e7fe56d684c179632dae5ccd62df48d20807e1cc779867b00c"
app = modal.App(f"arc-{KIND}-validation", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
if KIND in {"arena", "vllm"}:
    image = modal.Image.debian_slim(python_version="3.12").apt_install(
        "libseccomp2", "binutils", "libglib2.0-0", "libgl1", "libnuma1", "libgomp1",
        "build-essential", "ninja-build",
    )
else:
    image = modal.Image.from_registry(PENNYROYAL, add_python="3.12").entrypoint([]).run_commands(
        # The original recipe pins GCC 15; Fedora's unversioned compiler may be newer.
        "test -x /usr/bin/gcc-15 && test -x /usr/bin/g++-15",
        "ln -sf /usr/bin/gcc-15 /usr/bin/gcc && ln -sf /usr/bin/g++-15 /usr/bin/g++",
        "/usr/bin/gcc --version",
    )
image = image.env({"ARC_VALIDATION_KIND": KIND, "ARC_VALIDATION_TEST": TEST}).add_local_dir(
    str(BUNDLE), "/opt/arc-validation", copy=True,
    ignore=["dataset-upload/**", "*.ipynb", "source.zip"],
).add_local_file(__file__, "/root/modal_validation.py", copy=True)


def reserve(run_id, kind, seconds, *, test="smoke"):
    import fcntl

    path = ROOT / "docs/spend.json"
    with path.open("r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        data = json.load(file)
        if test in ('research-model', 'spec-sweep'):
            if sum(r.get('kind') == 'research-pilot' and r.get('state') == 'reserved'
                   for r in data['runs']) >= 4:
                raise RuntimeError('Four research GPUs reserved; reconcile before launching another')
            kind = 'research-pilot'
        cpu = globals().get("CPU_CEILING", 8)  # the tests exec reserve() alone
        rate = 0.000842 + cpu * 0.0000131 + 192 * 0.00000222
        bound = (seconds + 60) * rate * 1.15
        from arc_infra.runops.records import row_cost
        committed = data["infrastructure_reserve_usd"] + sum(row_cost(r)[0] for r in data["runs"])
        if committed + bound > data["cap_usd"]:
            raise RuntimeError(f"Spend cap: ${committed:.2f} + ${bound:.2f} required")
        if any(r["run_id"] == run_id for r in data["runs"]):
            raise ValueError("Run already reserved; inspect saved call instead of relaunching")
        if test in ("hx4-level", "hx4-quality"):
            if not run_id.startswith(("hx4-", "hx5-", "hx6-", "hx7-")):
                raise ValueError("hx4 runs use hx4- run ids")
            spent = sum(r["charged_bound_usd"] for r in data["runs"] if r["run_id"].startswith(("hx4-", "hx5-", "hx6-", "hx7-")))
            if spent + bound > HX4_TRANCHE_USD:
                raise RuntimeError(f"hx4 tranche: ${spent:.2f} + ${bound:.2f} exceeds ${HX4_TRANCHE_USD:.0f}")
        data["runs"].append({
            "run_id": run_id, "state": "reserved", "kind": kind,
            "charged_bound_usd": bound, "max_seconds": seconds,
            "gpu": "RTX-PRO-6000", "cpu_ceiling": cpu, "memory_ceiling_gib": 192,
            "authorization": (
                "harness arm restart (hx4), 1 RTX PRO 6000 at a time, <= of new reservations"
                if test in ("hx4-level", "hx4-quality") else
                "up to four research GPUs, pilot <= one hour including setup, "
                if test == "research-model" else
                "seven nine-hour Modal benchmarks plus user-requested additional nine-hour arms, cumulative cap"
                if test == "campaign-9h" else
                "User authorized paired arena comparisons; Kaggle-first with 30-minute queue fallback"
                if test == 'paired-30m'
                else "AGENTS.md 24 September: Kaggle first; switch to Modal after 10 minutes queued"
                if test in ('duck-30m', 'duck-60m', 'bench', 'bench-ab')
                else "10-15 minute Modal probes in parallel with the one-hour Kaggle runs"
                if test == 'duck-probe'
                else "level-bench calibration on Modal (one Modal GPU job at a time)"
                if test == 'level-bench'
                else "evening: 110-game one-hour arena runs on Modal (one GPU job at a time)"
                if test == 'duck-110'
                else "hero-v2 on its own dedicated Modal GPU (lock /tmp/arc-gpu-lock2), beside the "
                     "one GPU for tests; smoke <= then the 2-hour 110-clone arena; go from the operator"
                if test in ('hero-v2-smoke', 'hero-v2-2h')
                else "hero-v2 follow-up runs on the hero-v2 slot (/tmp/arc-gpu-lock2), run 1 then "
                     "run 2 in sequence after the analysis, on explicit approval; 2 hours each, stopped gracefully "
                     "at 1 hour if the score beats hero-v2's 1.68%"
                if test in ('hero-v2-128k-int4', 'hero-v2-streams')
                else "two parallel 2-hour hero-v2 follow-ups without early stop, run A on the "
                     "hero-v2 slot (/tmp/arc-gpu-lock2) and run B on a third GPU (/tmp/arc-gpu-lock3, temporary user "
                     "exception); launch as soon as built and checked"
                if test in ('hero-v2-int4-streams', 'hero-v2-128k-streams')
                else "the three-block scheduler (hero-v2-blocks) on the c110 roster, 37 streams, "
                     "a 30,600 s scheduler wall stopped at 7,200 s; hero-v2 slot (/tmp/arc-gpu-lock2); launch once the "
                     "checks pass"
                if test == 'hero-v2-blocks'
                else "run B's configuration plus the opt-in parallel_samples and action_plans harness "
                     "features, one 2-hour c110 arena on a free Modal GPU slot (test slot /tmp/arc-gpu-lock or the "
                     "hero-v2 slot /tmp/arc-gpu-lock2, whichever frees first), after the CPU tests"
                if test == 'hero-v2-samples-plans'
                else "test Holo4-27B (Hcompany/Holo4-27B-NVFP4, NVFP4 + FP8 KV) with the hero-v2 "
                     "arena setup on the test slot (/tmp/arc-gpu-lock): smoke then one 1-hour c110 arena "
                     " if it passes"
                if test in ('holo-smoke', 'holo-60m', 'holo-soak')
                else "kernel phase 2 (A+C) quality gate, then the one-hour 110-clone arena "
                     "if it passes; one Modal GPU job, "
                if test == 'duck-110-kernels'
                else "kernel phase 4 server smoke of the MTP-3 admission arm (6 x 64K with "
                     "drafting + fusions); one Modal GPU job, "
                if test == 'kernel-smoke'
                else "expert-usage measurement (routing capture, pruned-router KL/NLL on "
                     "held-out games); v3 server-start diagnostics <= plus the measurement "
                     "<= "
                if test in ('expert-usage', 'expert-usage-measure')
                else "load-time expert pruning GPU check (held-out KL of the 12.5% and "
                     "25% pruned hero against the unpruned hero, and the freed memory); one Modal GPU job"
                if test == 'expert-prune-check'
                else "measure more pruned levels (layer-aware 25%, uniform 25%, 18.75%) on "
                     "the served hero against its repeat noise; one Modal GPU job, (coordinator)"
                if test == 'expert-prune-levels'
                else "prune12 hang test (block-id sweep on the pruned hero, served pool "
                     "against the unpruned hero's pool); one Modal GPU job"
                if test == 'expert-prune-hang'
                else "serving-tweaks smoke (16K MTP draft head, PLE rows over UVA, host-resident "
                     "input embedding; the vision tower stays); one Modal GPU job, "
                if test == 'tweaks-smoke'
                else "small stackable kernel optimizations (phase 5); one Modal GPU job, "
                     " (coordinator)"
                if test == 'kernel-stack'
                else "hero-configuration profiling smoke; one Modal GPU job, "
                     ""
                if test == 'hero-smoke'
                else "kernel phase 6 (hero vs UVA + fused split-K, mixed prefill, "
                     "microbenchmarks); one Modal GPU job, "
                if test in ('kernel-phase6', 'kernel-phase6b')
                else "prefix-cache v2 smoke (Mamba chunk alignment, keep and demote patches "
                     "against the pinned cache on the hero serving); one Modal GPU job, "
                if test == 'prefix-cache-smoke'
                else "(user: \"let's test the 3 priority items\", item 3): prefix-cache v3 "
                     "smoke (request-end tail entries: teacher-forced warm/cold check and replay with exact cached "
                     "tokens on hero-v2 serving, int4 then int8 KV); one test-slot Modal GPU job, stated "
                     "before launch"
                if test == 'prefix-cache-v3-smoke'
                else "(via coordinator): prefix-cache v3 recheck (the teacher-forced request-end "
                     "tail check alone at int4, per-prefix entry cap 128); one test-slot Modal GPU job, "
                     "stated before launch"
                if test == 'prefix-cache-v3-recheck'
                else "draft-head (MTP) fine-tune; pilot <= Modal one GPU job "
                     "at a time"
                if test == 'draft-head'
                else "8-bit QSA KV cache smoke (served layout, drift, decode, QSA "
                     "microbenchmark); one Modal GPU job, "
                if test == 'kv8-smoke'
                else "ARC expert re-quantization, stage 1 microbenchmark (capture + CUTLASS/"
                     "Marlin/BF16 errors on real ARC activations); one Modal GPU job, "
                if test == 'expert-quant-bench'
                else "ARC expert re-quantization, stage 2 (GPTQ on ARC activations; end-to-end "
                     "check); one Modal GPU job at a time, stage "
                if test in ('expert-quant-calibrate', 'expert-quant-eval', 'expert-quant-calibrate-v2')
                else "long-context drift test (kernels-v2 reference + repeat, hero serving, int8 KV "
                     "alone on 39-44K-token c110 contexts); one test-slot Modal GPU job, "
                if test == 'drift-long'
                else "4-bit attention KV cache drift test (hero-v2 serving with int8 KV as the "
                     "reference + repeat, int4 and k8v4 KV as report-only gates); one test-slot Modal GPU job, "
                     ""
                if test == 'kv4-drift'
                else "(\"Yes, both\", via coordinator): hero-v2 serving pre-check (pool at "
                     "max_num_seqs 32, decode / mixed stream sweep 16-32, --prefix-match-unit 16 warm/cold and "
                     "replay); one test-slot Modal GPU job, (stated before launch)"
                if test == 'hero-v2-precheck'
                else "(\"let's test the 3 priority items\", via coordinator): --prefix-match-unit 16 "
                     "on the int4 layout (teacher-forced Latin square, replay) and the pool at 48 sequences / graphs "
                     "to 48; one test-slot Modal GPU job, <= (stated before launch)"
                if test == 'prefix-unit-int4'
                else "prefix-unit-int4 v2 (the teacher-forced square alone; v1 lost its warm arms to a client length "
                     "check): one test-slot Modal GPU job, stated before launch (coordinator go)"
                if test == 'prefix-unit-int4-tf'
                else "(\"do the 20-minute gpu sweep first\"; \"i allow the 3 items\"): decode sweep on "
                     "the blocks3 campaign serving at 24-56 streams with a finer CUDA-graph ladder; one research GPU "
                     "(<= 1 h including setup)"
                if test == 'decode-sweep'
                else "(\"do the speculative decoding analysis fast\"): speculative-decoding pilot, "
                     "one MTP arm of the blocks3 serving on recorded arena contexts; research GPU (<= 1 h including "
                     "setup, <= up to three pilots)"
                if test == 'spec-sweep'
                else "(via coordinator): kv8 BF16 prefill path check (hero-v2 reference + "
                     "repeat, kv8_prefill gate, injected-prefill bench); one test-slot Modal GPU job, "
                if test == 'kv8-prefill-check'
                else "User requested Modal-only validation on 22 September 2026"),
        })
        file.seek(0)
        json.dump(data, file, indent=1)
        file.write("\n")
        file.truncate()
    return bound


@app.function(
    image=image, gpu="RTX-PRO-6000", cpu=(0.125, CPU_CEILING), memory=(128, 196608),
    timeout=TIMEOUT, retries=0, max_containers=1, scaledown_window=2,
    block_network=True, volumes={"/assets": assets, "/runs": runs},
)
def benchmark(run_id: str):
    import hashlib
    import shutil
    import subprocess
    import sys

    source = Path("/opt/arc-validation")
    output = Path("/runs/validation") / run_id
    if output.exists():
        # Modal re-runs an input whose container was lost (25 September: memory-v1's first container
        # died during the wheel install and the re-run failed here); keep the lost attempt aside.
        output.rename(output.with_name(f"{run_id}-lost-{int(time.time())}"))
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    os.environ["ARC_VALIDATION_OUTPUT"] = str(output)  # where a trial may persist large evidence directly
    identity = {"run_id": run_id, "kind": KIND, "started_epoch": started,
                "gpu": "RTX-PRO-6000", "function_timeout": TIMEOUT,
                "image": PENNYROYAL if KIND == "sglang" else "debian-slim/python3.12"}
    Path("/kaggle").mkdir(exist_ok=True)
    Path("/kaggle/input").symlink_to("/assets/flash-input", target_is_directory=True)
    os.environ.update(PYTHONPATH=str(source / "src"), PYTHONNOUSERSITE="1",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PIP_NO_INDEX="1",
                      MPLBACKEND="Agg", ONLY_RESET_LEVELS="true", PYTHONDONTWRITEBYTECODE="1")
    os.environ["LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", "")
    process = None
    try:
        if KIND == "vllm":
            os.environ["ARC_FORCE_CHROOT"] = "1"
            command = [sys.executable, str(source / "bootstrap_mxfp8.py"), "--bundle", str(source),
                       "--working", "/tmp/mxfp8", "--output", str(output / "mxfp8"),
                       "--games-dir", "/assets/games", "--max-seconds", str(TIMEOUT - 60)]
        elif KIND == "arena":
            os.environ["ARC_FORCE_CHROOT"] = "1"
            os.environ["ARENA_STOP_REQUEST"] = STOP_LOCAL  # arena.early_stop: the trial's manual stop file
            command = [sys.executable, str(source / "bootstrap_arena.py"), "--bundle", str(source),
                       "--working", "/tmp/arena", "--output", str(output / "arena"),
                       "--games-dir", "/assets/games", "--max-seconds", str(TIMEOUT - 60),
                       *arena_arguments(TEST)]
            if TEST in {"full-duck-transfer", "cache-duck-transfer"}:
                study = json.loads((source / "configs/comparison.json").read_text())
                if len(set(study["games"])) != 25 or study["wall_seconds"] != 900:
                    raise ValueError("Full arena must contain 25 distinct games and 900 seconds")
            if TEST in ('duck-30m', 'duck-60m'):
                study = json.loads((source / 'configs/comparison.json').read_text())
                if (len(set(study['games'])) + len(study.get('games_excluded', [])) != 25
                        or study['wall_seconds'] != (1800 if TEST == 'duck-30m' else 3600)
                        or study['trials'][0] != {'name': 'duck-transfer', 'kind': 'duck-transfer'}
                        or study.get('backend', 'vllm') != 'vllm'):
                    raise ValueError('Duck arena fallback requires 25 games, the test wall, Duck-transfer and vLLM')
            if TEST in ('competition-preview', 'competition-preview-1h', 'competition-preview-4h', 'competition-preview-8h'):
                study = json.loads((source / 'configs/comparison.json').read_text())
                clones = study.get('competition') or {}
                if (not clones or clones.get('research_clones') or clones.get('preview_instances') != 110
                        or study['wall_seconds'] != clones.get('preview_seconds') or not study.get('hero_v2')
                        or [t['name'] for t in study['trials']] != ['duck-transfer']):
                    raise ValueError('The competition preview test runs an unmodified submission bundle (110 public '
                                     'clones for its preview seconds, the Duck-transfer trial only)')
            if TEST == 'duck-110':
                study = json.loads((source / 'configs/comparison.json').read_text())
                clones = study.get('competition') or {}
                if (not clones.get('research_clones') or clones.get('preview_instances') != 110
                        or len(set(study['games'])) + len(study.get('games_excluded', [])) != 25
                        or study['wall_seconds'] != 3600 or clones.get('preview_seconds') != 3600
                        or study['trials'][0] != {'name': 'duck-transfer', 'kind': 'duck-transfer'}):
                    raise ValueError('The 110-clone arena requires a research clone roster of 110 instances for 3,600 s')
            if TEST in ('hero-v2-2h', 'hero-v2-smoke'):
                study = json.loads((source / 'configs/comparison.json').read_text())
                clones = study.get('competition') or {}
                hero = study.get('hero_v2') or {}
                names = [t['name'] for t in study['trials']]
                if (study.get('arena_variant', {}).get('name') != 'fixed-2x-mtp0-wide-kv8-hero-v2' or not hero
                        or not clones.get('research_clones') or clones.get('preview_instances') != 110
                        or clones.get('preview_seconds') != study['wall_seconds'] or not study.get('live_progress')
                        or study['server'].get('mtp_tokens') != 0 or study['windows'] != [65536]
                        or hero.get('requested', {}).get('kv_cache', 'int8') != 'int8'):
                    raise ValueError('hero-v2 needs the hero-v2 study on the 110-clone roster with live progress')
                if TEST == 'hero-v2-2h' and (study['wall_seconds'] != 7200 or 'smoke' in hero
                                             or names[0] != 'duck-transfer'):
                    raise ValueError('The hero-v2 run is the 7,200 s arena (not the smoke bundle)')
                if TEST == 'hero-v2-smoke' and (names != ['hero-v2-check', 'duck-transfer'] or 'smoke' not in hero
                                                or study['wall_seconds'] > 1200
                                                or not (source / study['throughput_bench']['quality']['reference']).is_file()):
                    raise ValueError('The hero-v2 smoke needs the check trial, a soak of at most 1,200 s and its reference')
            if TEST in ('hero-v2-int4-streams', 'hero-v2-128k-streams'):
                # Runs A and B: int4 KV, oversubscribed focus (more streams than full windows fit)
                # under the exact gate, harness_fixes (all), the progress-fruitless backlog, fixed-2x tokens, 7,200 s
                # with no early stop; max_num_seqs = focus + 2 and decode graphs captured to it.
                study = json.loads((source / 'configs/comparison.json').read_text())
                clones = study.get('competition') or {}
                hero = study.get('hero_v2') or {}
                kernels = study['server'].get('serving_kernels') or {}
                over = study.get('focus_oversubscription') or {}
                fixes = (study.get('harness_fixes') or {}).get('fixes') or []
                seqs = study['server'].get('max_num_seqs', 0)
                if (study.get('arena_variant', {}).get('name') != 'fixed-2x-mtp0-wide-kv8-hero-v2' or not hero
                        or 'smoke' in hero or not clones.get('research_clones')
                        or clones.get('preview_instances') != 110 or study['wall_seconds'] != 7200
                        or clones.get('preview_seconds') != 7200 or not study.get('live_progress')
                        or study['server'].get('mtp_tokens') != 0
                        or [t['name'] for t in study['trials']] != ['duck-transfer']
                        or study.get('early_stop') or study.get('level_budget_policy') != 'fixed-2x'
                        or study.get('budget_clock') != 'tokens' or kernels.get('kv_cache_4bit') != 'int4'
                        or kernels.get('kv_cache_8bit') or len(fixes) != 8
                        or study.get('backlog_key') != 'progress-fruitless' or study.get('level0_order') != 'tokens'
                        or not study['server'].get('prompt_tokens_details')
                        or not (study.get('cache_experiment') or {}).get('prefix_cache')
                        or not over or over.get('focus_slots') != study['active_limit']
                        or not over['focus_slots'] > over.get('full_windows_fit_projected', 10 ** 6)
                        or study.get('require_focus_windows_fit') or seqs != study['active_limit'] + 2
                        or study['generation_slots'] != seqs
                        or study['server'].get('max_cudagraph_capture_size', 32) < seqs
                        or not (study.get('timeline') or {}).get('plan')):
                    raise ValueError('Runs A/B need the hero-v2 clone arena for 7,200 s without early stop: int4 KV, '
                                     'oversubscribed focus (> full windows) at max_num_seqs = focus + 2 within the '
                                     'captured graphs, harness_fixes (all 8), the progress-fruitless backlog, '
                                     'fixed-2x tokens and a recorded timeline')
                if TEST == 'hero-v2-int4-streams' and study['windows'] != [65536]:
                    raise ValueError('Run A is the 64K window')
                if TEST == 'hero-v2-128k-streams' and study['windows'] != [130816]:
                    raise ValueError('Run B is the 130,816-token window')
            if TEST == 'hero-v2-blocks':
                # hero-v2-blocks: the three-block scheduler (37 / 37 / 36 of the 110 clones, 37
                # streams, 10,200 s blocks of a 30,600 s wall, 90K-token level-0 retirement), stopped unconditionally at
                # 7,200 s; run B's harness (8 sub-fixes) + summary_headroom (not inventory_imports); hero-v2 serving on
                # int4 KV with --prefix-match-unit 16, max_num_seqs 41 = streams + 4, graphs to 41; block 1 = every
                # source game once + 17 distinct second clones (user requirement).
                study = json.loads((source / 'configs/comparison.json').read_text())
                clones = study.get('competition') or {}
                hero = study.get('hero_v2') or {}
                kernels = study['server'].get('serving_kernels') or {}
                over = study.get('focus_oversubscription') or {}
                fixes = (study.get('harness_fixes') or {}).get('fixes') or []
                blocks = study.get('blocks') or {}
                plan = study.get('blocks_plan') or {}
                early = study.get('early_stop') or {}
                seqs = study['server'].get('max_num_seqs', 0)
                games = study['games']
                sources = [games[i % len(games)] for i in range(37)]
                counts = {g: sources.count(g) for g in set(sources)}
                expected_fixes = ['exact_trigger', 'summary_guard', 'inventory', 'cap_followup', 'stuck_notice',
                                  'sandbox', 'timeout_interrupt', 'prompt_text', 'summary_headroom']
                prefix = (study.get('cache_experiment') or {}).get('prefix_cache') or {}
                if (study.get('arena_variant', {}).get('name') != 'fixed-2x-mtp0-wide-kv8-hero-v2' or not hero
                        or 'smoke' in hero or not clones.get('research_clones')
                        or clones.get('preview_instances') != 110 or study['wall_seconds'] != 30600
                        or clones.get('preview_seconds') != 30600 or not study.get('live_progress')
                        or study['server'].get('mtp_tokens') != 0
                        or [t['name'] for t in study['trials']] != ['duck-transfer']
                        or early != {'at_seconds': 7200, 'score_above': None, 'retry_seconds': 300, 'always': True}
                        or study.get('level_budget_policy') != 'blocks'
                        or {k: v for k, v in blocks.items() if k != 'stagger_seconds'} != {
                            'wall_seconds': 30600, 'blocks': 3, 'slots': 37, 'roster_games': 110,
                            'level0_budget_tokens': 90000, 'tenure_unit': 'wall_seconds', 'tenure_tokens': None}
                        or blocks.get('stagger_seconds', 0) not in (0, 600)
                        or study.get('kv_margin') not in (0.9, 0.96)
                        or plan.get('block_sizes') != [37, 37, 36] or plan.get('block_seconds') != 10200
                        or plan.get('run_stop_seconds') != 7200
                        or len(games) != 20 or len(counts) != 20 or max(counts.values()) != 2
                        or sum(c == 2 for c in counts.values()) != 17
                        or plan.get('block1', {}).get('sources') != sources
                        or study.get('budget_clock') != 'tokens' or kernels.get('kv_cache_4bit') != 'int4'
                        or kernels.get('kv_cache_8bit') or fixes != expected_fixes
                        or study.get('backlog_key') or study.get('level0_order') not in (None, 'roster')
                        or not study['server'].get('prompt_tokens_details')
                        or study['server'].get('prefix_match_unit') != 16
                        or not prefix or prefix.get('request_end_tail')
                        or (study.get('cache_experiment') or {}).get('keep_alive')
                        or study['active_limit'] != 37 or over.get('focus_slots') != 37
                        or not over.get('gate_blocks_per_stream')
                        or over['gate_blocks_per_stream'] * 37 > int(study.get('kv_margin', 0.9) * 1260)
                        or study.get('require_focus_windows_fit') or seqs != 41 or study['generation_slots'] != 41
                        or study['server'].get('max_cudagraph_capture_size') != 41
                        or len(study['windows']) != 1 or not 81920 <= study['windows'][0] <= 131072
                        or study['windows'][0] % 1024
                        or not (study.get('timeline') or {}).get('plan')
                        or study.get('reasoning_history') != 'verbatim'
                        or study.get('compaction_thinking') != 'prefilled-empty'
                        or study.get('completion_cap') != 16384):
                    raise ValueError('hero-v2-blocks needs the three-block study (37 / 37 / 36 of 110 clones, 37 '
                                     'streams, 30,600 s scheduler wall, 90K level-0 budget, unconditional stop at 7,200 '
                                     's, block 1 = 20 sources + 17 second clones), int4 KV with prefix-match-unit 16, '
                                     'max_num_seqs 41 and graphs to 41, run B harness fixes + summary_headroom, a '
                                     'window of at least 80K and a recorded timeline')
            if TEST == 'campaign-9h':
                import sys
                sys.path.insert(0, str(source / 'src'))
                from arena.campaign import validate as validate_campaign
                validate_campaign(json.loads((source / 'configs/comparison.json').read_text()))
            if TEST == 'hero-v2-samples-plans':
                # Run B + the two opt-in harness features: run B's
                # configuration exactly (int4 KV, 24 oversubscribed focus streams at 130,816, fixed-2x tokens, the
                # six-clear switch, harness_fixes (8), progress-fruitless backlog, 7,200 s, no early stop), plus
                # parallel_samples (k 4, min_levels 2) and action_plans, with max_num_seqs 48 / graphs to 48 for the
                # siblings; nothing else (no prefix-match unit, no summary_headroom, no request-end tail).
                study = json.loads((source / 'configs/comparison.json').read_text())
                clones = study.get('competition') or {}
                hero = study.get('hero_v2') or {}
                kernels = study['server'].get('serving_kernels') or {}
                over = study.get('focus_oversubscription') or {}
                fixes = (study.get('harness_fixes') or {}).get('fixes') or []
                samples = study.get('parallel_samples') or {}
                prefix = (study.get('cache_experiment') or {}).get('prefix_cache') or {}
                seqs = study['server'].get('max_num_seqs', 0)
                if (study.get('arena_variant', {}).get('name') != 'fixed-2x-mtp0-wide-kv8-hero-v2' or not hero
                        or 'smoke' in hero or not clones.get('research_clones')
                        or clones.get('preview_instances') != 110 or study['wall_seconds'] != 7200
                        or clones.get('preview_seconds') != 7200 or not study.get('live_progress')
                        or study['server'].get('mtp_tokens') != 0 or study['windows'] != [130816]
                        or [t['name'] for t in study['trials']] != ['duck-transfer']
                        or study.get('early_stop') or study.get('level_budget_policy') != 'fixed-2x'
                        or study.get('budget_clock') != 'tokens' or kernels.get('kv_cache_4bit') != 'int4'
                        or kernels.get('kv_cache_8bit') or fixes != ['exact_trigger', 'summary_guard', 'inventory',
                                                                     'cap_followup', 'stuck_notice', 'sandbox',
                                                                     'timeout_interrupt', 'prompt_text']
                        or study.get('backlog_key') != 'progress-fruitless' or study.get('level0_order') != 'tokens'
                        or not study['server'].get('prompt_tokens_details') or not prefix
                        or prefix.get('request_end_tail') or study['server'].get('prefix_match_unit')
                        or (study.get('cache_experiment') or {}).get('keep_alive')
                        or study['active_limit'] != 24 or not over or over.get('focus_slots') != 24
                        or not over['focus_slots'] > over.get('full_windows_fit_projected', 10 ** 6)
                        or study.get('require_focus_windows_fit')
                        or samples.get('k') != 4 or samples.get('min_levels') != 2
                        or seqs != samples.get('max_num_seqs') or seqs != 48
                        or study['generation_slots'] != study['active_limit'] + 2
                        or study['server'].get('max_cudagraph_capture_size', 32) < seqs
                        or study.get('action_plans') is not True
                        or not (study.get('timeline') or {}).get('plan')):
                    raise ValueError('hero-v2-samples-plans needs run B exactly (int4 KV, 24 oversubscribed focus '
                                     'streams at 130,816, fixed-2x tokens, harness_fixes (8), progress-fruitless, '
                                     '7,200 s, no early stop, prefix cache v2 without the tail / unit / keep-alive) '
                                     'plus parallel_samples (k 4, min_levels 2) at max_num_seqs 48 within the '
                                     'captured graphs and action_plans, with a recorded timeline')
            if TEST in ('hero-v2-128k-int4', 'hero-v2-streams'):
                # hero-v2 follow-ups: run 1 = int4 KV at 128K windows, fixed-2x, the hero-v2
                # harness; run 2 = int8 KV at 64K, 30 oversubscribed focus streams at max_num_seqs 32, the backlog fix.
                # Both: the 7,200 s clone arena with the 1 h early stop above hero-v2's 1.68%.
                study = json.loads((source / 'configs/comparison.json').read_text())
                clones = study.get('competition') or {}
                hero = study.get('hero_v2') or {}
                requested = hero.get('requested') or {}
                kernels = study['server'].get('serving_kernels') or {}
                early = study.get('early_stop') or {}
                if (study.get('arena_variant', {}).get('name') != 'fixed-2x-mtp0-wide-kv8-hero-v2' or not hero
                        or 'smoke' in hero or not clones.get('research_clones')
                        or clones.get('preview_instances') != 110 or study['wall_seconds'] != 7200
                        or clones.get('preview_seconds') != 7200 or not study.get('live_progress')
                        or study['server'].get('mtp_tokens') != 0
                        or [t['name'] for t in study['trials']] != ['duck-transfer']
                        or early.get('at_seconds') != 3600 or early.get('score_above') is None
                        or study.get('level_budget_policy') != 'fixed-2x' or study.get('budget_clock') != 'tokens'):
                    raise ValueError('hero-v2 follow-ups need the hero-v2 clone arena for 7,200 s with live progress, '
                                     'fixed-2x token budgets and the 1 h early stop')
                if TEST == 'hero-v2-128k-int4' and (
                        study['windows'] not in ([131072], [130816]) or kernels.get('kv_cache_4bit') != 'int4'
                        or kernels.get('kv_cache_8bit') or requested.get('backlog_fix') or study.get('harness_fixes')):
                    raise ValueError('Run 1 is hero-v2 at 128K windows on the int4 attention KV, hero-v2 harness')
                if TEST == 'hero-v2-streams' and (
                        study['windows'] != [65536] or kernels.get('kv_cache_8bit') != 'int8'
                        or not study.get('focus_oversubscription') or study['server'].get('max_num_seqs', 0) > 32):
                    raise ValueError('Run 2 is hero-v2 at 64K on int8 KV with oversubscribed focus streams, '
                                     'max_num_seqs <= 32 (the captured decode graphs)')
            if TEST in ('holo-smoke', 'holo-60m', 'holo-soak'):
                study = json.loads((source / 'configs/comparison.json').read_text())
                clones = study.get('competition') or {}
                hero = study.get('hero_v2') or {}
                names = [t['name'] for t in study['trials']]
                if (study.get('arena_variant', {}).get('name') != 'holo4-27b-hero-v2'
                        or study.get('candidate_profile') != 'holo4-27b-nvfp4' or not hero
                        or not clones.get('research_clones') or clones.get('preview_instances') != 110
                        or clones.get('preview_seconds') != study['wall_seconds'] or not study.get('live_progress')
                        or study['server'].get('mtp_tokens') != 0 or study['windows'] != [65536]):
                    raise ValueError('The Holo4 tests need the candidate hero-v2 study on the 110-clone roster')
                if TEST == 'holo-60m' and (study['wall_seconds'] != 3600 or 'smoke' in hero or 'soak' in hero
                                           or names != ['duck-transfer']
                                           or not {'ARENA_PROMPT_TOKENS_DETAILS', 'ARENA_STALL_DUMP',
                                                   'ARENA_READY_GUARD'} <= set(study['trials'][0].get('server_env', {}))
                                           or not study['cache_experiment'].get('prefix_cache')):
                    raise ValueError('The Holo4 run is the 3,600 s arena (not the smoke bundle)')
                if study.get('harness_fixes') or hero['components'].get('harness_fixes', {}).get('on') is not False:
                    raise ValueError('The Holo4 arm keeps the hero-v2 2 h harness: harness_fixes must be off')
                if TEST == 'holo-soak' and (names != ['duck-transfer'] or 'soak' not in hero or study['wall_seconds'] > 900
                                            or hero['soak']['function_timeout'] != TIMEOUT):
                    raise ValueError('The Holo4 soak is the one-trial soak bundle sized for this test timeout')
                if TEST == 'holo-smoke' and (names != ['kv-bf16-ref', 'kv-fp8', 'holo-check', 'duck-transfer']
                                             or 'smoke' not in hero or study['wall_seconds'] > 900
                                             or not (source / 'configs/kernel-quality/prompts.json').is_file()
                                             or not (source / study['throughput_bench']['transcript']['file']).is_file()):
                    raise ValueError('The Holo4 smoke needs the drift pair, the check, a soak of at most 900 s and inputs')
            if TEST in ('prefix-unit-int4', 'prefix-unit-int4-tf'):
                study = json.loads((source / 'configs/comparison.json').read_text())
                bench = study.get('throughput_bench') or {}
                trial = study['trials'][0] if len(study['trials']) == 1 else {}
                server = study['server']
                if ('prefix_unit_int4' not in (study.get('hero_v2') or {})
                        or trial.get('name') != 'prefix-int4' or trial.get('kind') != 'serving-bench'
                        or trial.get('server_env', {}).get('ARENA_WARM_PROMPT_LOGPROBS') != '1'
                        or (server.get('serving_kernels') or {}).get('kv_cache_4bit') != 'int4'
                        or server.get('prefix_match_unit') != 16 or server.get('max_num_seqs') != 48
                        or server.get('max_cudagraph_capture_size') != 48 or study.get('generation_slots') != 48
                        or not (study.get('cache_experiment') or {}).get('prefix_cache')
                        or not bench.get('partial_tf')
                        or bool(bench.get('prefix_replay')) != (TEST == 'prefix-unit-int4')
                        or not (source / bench['partial_tf']['file']).is_file()
                        or not (source / bench['transcript']['file']).is_file()
                        or study['notebook_seconds'] > TIMEOUT - 60):
                    raise ValueError('prefix-unit-int4 needs the one-trial int4 serving bench at 48 sequences with the '
                                     'prefix cache, --prefix-match-unit 16 and the warm prompt-logprobs patch')
            if TEST == 'decode-sweep':
                study = json.loads((source / 'configs/comparison.json').read_text())
                bench = study.get('throughput_bench') or {}
                trial = study['trials'][0] if len(study['trials']) == 1 else {}
                server = study['server']
                if ('decode_sweep' not in study or trial.get('name') != 'sweep' or trial.get('kind') != 'serving-bench'
                        or (server.get('serving_kernels') or {}).get('kv_cache_4bit') != 'int4'
                        or server.get('prefix_match_unit') != 16 or server.get('max_num_seqs') != 56
                        or server.get('max_cudagraph_capture_size') != 56
                        or (server.get('cudagraph_capture_sizes') or [None])[-1] != 56
                        or not bench.get('stream_sweep') or study['notebook_seconds'] > TIMEOUT - 60):
                    raise ValueError('decode-sweep needs the one-trial int4 serving bench at 56 sequences with the '
                                     'finer CUDA-graph ladder and --prefix-match-unit 16')
            if TEST == 'spec-sweep':
                study = json.loads((source / 'configs/comparison.json').read_text())
                bench = study.get('throughput_bench') or {}
                server = study['server']
                pilot = study.get('spec_sweep_pilot') or {}
                if (study['trials'] != [{**study['trials'][0], 'name': 'spec', 'kind': 'serving-bench'}]
                        or len(study['trials']) != 1 or not bench.get('spec_sweep')
                        or server.get('mtp_tokens') not in (0, 1, 2) or pilot.get('mtp_tokens') != server['mtp_tokens']
                        or (server.get('serving_kernels') or {}).get('kv_cache_4bit') != 'int4'
                        or server.get('prefix_match_unit') != 16 or server.get('max_num_seqs') != 44
                        or server.get('max_cudagraph_capture_size') != 44 * (server['mtp_tokens'] + 1)
                        or not (source / bench['spec_sweep']['contexts']).is_file()
                        or study['notebook_seconds'] > TIMEOUT - 60):
                    raise ValueError('spec-sweep needs the one-trial spec bench on the int4 blocks3 serving at 44 '
                                     'sequences, graphs to 44 x (MTP + 1) and its recorded contexts')
            if TEST == 'hero-v2-precheck':
                study = json.loads((source / 'configs/comparison.json').read_text())
                bench = study.get('throughput_bench') or {}
                trial = study['trials'][0] if len(study['trials']) == 1 else {}
                if (study.get('arena_variant', {}).get('name') != 'fixed-2x-mtp0-wide-kv8-hero-v2'
                        or 'precheck' not in (study.get('hero_v2') or {})
                        or trial.get('name') != 'precheck' or trial.get('kind') != 'serving-bench'
                        or trial.get('server_env', {}).get('ARENA_PREFIX_MATCH_UNIT') != '16'
                        or study['server'].get('max_num_seqs') != 32 or study.get('generation_slots') != 32
                        or not (study.get('cache_experiment') or {}).get('prefix_cache')
                        or not bench.get('stream_sweep') or not bench.get('partial_check')
                        or not bench.get('prefix_replay')
                        or not (source / bench['partial_check']['file']).is_file()
                        or not (source / bench['transcript']['file']).is_file()):
                    raise ValueError('The hero-v2 pre-check needs the one-trial serving bench on hero-v2 at 32 '
                                     'sequences with the prefix cache and --prefix-match-unit 16')
            if TEST == 'duck-110-kernels':
                study = json.loads((source / 'configs/comparison.json').read_text())
                clones = study.get('competition') or {}
                kinds =[(t['name'], t['kind']) for t in study['trials']]
                if (not study.get('kernel_gate') or not study['server'].get('dense_kernels')
                        or kinds != [('kernel-ref', 'kernel-quality'), ('kernel-gate', 'kernel-quality'),
                                     ('duck-transfer', 'duck-transfer')]
                        or study['trials'][2].get('requires') != 'kernel-gate'
                        or not clones.get('research_clones') or clones.get('preview_instances') != 110
                        or study['wall_seconds'] != 3600):
                    raise ValueError('The kernel arena requires the kernel gate before a gated 110-clone hour')
            if TEST == 'duck-probe':
                study = json.loads((source / 'configs/comparison.json').read_text())
                if (not study.get('probe') or study['wall_seconds'] > 1200
                        or study['trials'][0] != {'name': 'duck-transfer', 'kind': 'duck-transfer'}):
                    raise ValueError('Duck probe requires a probe study of at most 1,200 s on Duck-transfer')
            if TEST in ('evolve', 'evolve-rehearsal'):
                study = json.loads((source / 'configs/comparison.json').read_text())
                if not study.get('evolution') or study.get('backend', 'vllm') != 'vllm':
                    raise ValueError('Evolution fallback requires an evolution study on vLLM')
                if TEST == 'evolve-rehearsal' and (study['evolution']['rounds'] != 1
                                                   or study['evolution']['round_wall_seconds'] > 300):
                    raise ValueError('The evolution rehearsal runs one round on evaluations of at most 300 s')
            if TEST == 'level-bench':
                study = json.loads((source / 'configs/comparison.json').read_text())
                if (study['trials'] != [{'name': 'level-bench', 'kind': 'level-bench'}] or not study.get('level_bench')
                        or not (source / study['level_bench']['start_points']).is_file()):
                    raise ValueError('The level-bench test requires the level-bench study and its start points')
            if TEST == 'hx4-quality':
                study = json.loads((source / 'configs/comparison.json').read_text())
                quality = study.get('quality_eval') or {}
                if ([(t['name'], t['kind']) for t in study['trials']] != [('duck-transfer', 'duck-transfer')]
                        or not run_id.startswith(('hx4-', 'hx5-', 'hx6-', 'hx7-')) or quality.get('version') != 2
                        or quality.get('split') != 'dev10'
                        or (study.get('competition') or {}).get('preview_seconds') != study.get('wall_seconds')
                        or study.get('notebook_seconds', 10 ** 6) > 12900):
                    raise ValueError('hx4-quality needs a DEV quality-protocol bundle (quality_eval version 2, one '
                                     'Duck trial, notebook <= 12,900 s) and an hx4- run id')
            if TEST == 'hx4-level':
                study = json.loads((source / 'configs/comparison.json').read_text())
                bench = study.get('level_bench') or {}
                if ([(t['name'], t['kind']) for t in study['trials']] != [('level-bench', 'level-bench')]
                        or not run_id.startswith(('hx4-', 'hx5-', 'hx6-', 'hx7-')) or not (source / bench.get('start_points', '-')).is_file()
                        or bench.get('concurrency', 10 ** 6) > study['active_limit']):
                    raise ValueError('hx4-level needs a build_hx4.py bundle (level-bench trial, start points, '
                                     'no more workers than KV-gate streams) and an hx4- run id')
            if TEST == 'rl0-level':
                study = json.loads((source / 'configs/comparison.json').read_text())
                bench = study.get('level_bench') or {}
                if ([(t['name'], t['kind']) for t in study['trials']] != [('level-bench', 'level-bench'),
                                                                         ('train-probe', 'train-probe')]
                        or not run_id.startswith('rl0-') or study.get('candidate_profile') != 'qwen35-4b'
                        or not (source / bench.get('start_points', '-')).is_file()
                        or bench.get('concurrency', 10 ** 6) > study['active_limit']
                        or study.get('notebook_seconds', 10 ** 6) > 23400):
                    raise ValueError('rl0-level needs a build_rl0.py bundle (Qwen3.5-4B, level-bench then train-probe, '
                                     'notebook <= 23,400 s) and an rl0- run id')
            if TEST == 'rl0-play':
                study = json.loads((source / 'configs/comparison.json').read_text())
                bench = study.get('level_bench') or {}
                if ([(t['name'], t['kind']) for t in study['trials']] != [('level-bench', 'level-bench')]
                        or not run_id.startswith('rl0-')
                        or study.get('candidate_profile') not in ('qwen35-4b', 'qwen36-35b-a3b-fp8')
                        or not (source / bench.get('start_points', '-')).is_file()
                        or bench.get('concurrency', 10 ** 6) > study['active_limit']
                        or study.get('notebook_seconds', 10 ** 6) > 23400):
                    raise ValueError('rl0-play needs a build_rl0.py bundle (a reviewed RL candidate, level-bench only, '
                                     'notebook <= 23,400 s) and an rl0- run id')
            if TEST == 'bench-ab':
                study = json.loads((source / 'configs/comparison.json').read_text())
                if not study.get('serving_ab') or {t['kind'] for t in study['trials']} != {'serving-bench'}:
                    raise ValueError('Serving A/B fallback requires the serving A/B bench study')
            if TEST in ('kernel-smoke', 'tweaks-smoke'):
                study = json.loads((source / 'configs/comparison.json').read_text())
                if (study['trials'] != [{'name': 'serving-bench', 'kind': 'serving-bench'}]
                        or not study.get('throughput_bench') or not study['server'].get('serving_kernels')
                        or study.get('backend', 'vllm') != 'vllm'):
                    raise ValueError('The kernel smoke requires the serving bench on the phase-4 serving arm')
                if TEST == 'tweaks-smoke' and not {'draft_vocab', 'ple_uva', 'embed_host'} <= set(
                        study['server']['serving_kernels']):
                    raise ValueError('The tweaks smoke requires the serving-tweaks arm')
            if TEST == 'expert-prune-hang':
                study = json.loads((source / 'configs/comparison.json').read_text())
                if ([t['name'] for t in study['trials']] != ['hang-pool1244', 'hang-pool860']
                        or study.get('arena_variant', {}).get('name') != 'fixed-2x-mtp0-wide-kv8-hero-prune12'
                        or not (source / study['server']['expert_prune']['mask']).is_file()):
                    raise ValueError('The prune hang test requires the two-arm study on the prune12 hero')
            if TEST in ('expert-prune-check', 'expert-prune-levels'):
                study = json.loads((source / 'configs/comparison.json').read_text())
                if bool(study['expert_prune_check'].get('levels')) != (TEST == 'expert-prune-levels'):
                    raise ValueError('The expert-prune levels test and the levels study go together')
                if ({t['kind'] for t in study['trials']} != {'prune-quality'} or study['trials'][0].get('role') != 'reference'
                        or study.get('arena_variant', {}).get('name') != 'fixed-2x-mtp0-wide-kv8-hero'
                        or study['server'].get('mtp_tokens') != 0 or study['server'].get('expert_prune')
                        or not all((source / v['mask']).is_file()
                                   for v in study['expert_prune_check']['variants'].values())):
                    raise ValueError('The expert-prune check requires the prune-quality study on the hero arm')
            if TEST in ('expert-usage', 'expert-usage-measure'):
                study = json.loads((source / 'configs/comparison.json').read_text())
                if (study['trials'][0]['kind'] != 'expert-usage'
                        or {t['kind'] for t in study['trials'][1:]} - {'server-check'}
                        or study['server'].get('expert_probe') is not True or study['server'].get('mtp_tokens') != 0
                        or not (source / 'configs/expert-usage/usage.json').is_file()
                        or study.get('backend', 'vllm') != 'vllm'):
                    raise ValueError('The expert-usage test requires the expert-usage study on the probe overlay')
            if TEST == 'hero-smoke':
                study = json.loads((source / 'configs/comparison.json').read_text())
                bench = study.get('throughput_bench') or {}
                if (study['trials'] != [{'name': 'hero', 'kind': 'serving-bench',
                                         'server_env': {'ARENA_TORCH_PROFILER': '1'}}]
                        or study.get('arena_variant', {}).get('name') != 'fixed-2x-mtp0-wide-kv8-hero'
                        or (study['server'].get('serving_kernels') or {}).get('moe_backend') != 'marlin'
                        or not bench.get('quality') or not (source / bench['quality']['reference']).is_file()
                        or study.get('backend', 'vllm') != 'vllm'):
                    raise ValueError('The hero smoke requires the serving bench on the hero arm with its reference')
            if TEST == 'kernel-phase6':
                study = json.loads((source / 'configs/comparison.json').read_text())
                trials = study['trials']
                if ([(t['name'], t['kind']) for t in trials] != [('hero', 'serving-bench'), ('p6', 'serving-bench')]
                        or study.get('arena_variant', {}).get('name') != 'fixed-2x-mtp0-wide-kv8-hero'
                        or not study.get('phase6_smoke')
                        or not (study.get('throughput_bench') or {}).get('mixed_points')
                        or not (source / study['throughput_bench']['quality']['reference']).is_file()
                        or 'ARENA_DENSE_KERNELS' in trials[0].get('server_env', {})
                        or json.loads(trials[1]['server_env']['ARENA_DENSE_KERNELS']).get('fused_splitk') is not True
                        or not json.loads(trials[1]['server_env']['ARENA_SERVING_KERNELS']).get('ple_uva')
                        or study.get('backend', 'vllm') != 'vllm'):
                    raise ValueError('The phase-6 smoke requires the hero control then the phase-6 arm on the hero study')
            if TEST == 'kernel-phase6b':
                study = json.loads((source / 'configs/comparison.json').read_text())
                trials = study['trials']
                if ([(t['name'], t['kind']) for t in trials] != [('p6b', 'serving-bench')]
                        or (study.get('phase6_smoke') or {}).get('part') != 2
                        or not (study.get('throughput_bench') or {}).get('phase6_microbench')
                        or json.loads(trials[0]['server_env']['ARENA_DENSE_KERNELS']).get('fused_splitk') is not True
                        or study.get('backend', 'vllm') != 'vllm'):
                    raise ValueError('The phase-6 part 2 requires the single p6b trial with the microbenchmarks')
            if TEST == 'prefix-cache-smoke':
                study = json.loads((source / 'configs/comparison.json').read_text())
                bench = study.get('throughput_bench') or {}
                arms = [t.get('name') for t in study['trials']]
                if (arms != ['pinned', 'prefix'] or {t['kind'] for t in study['trials']} != {'serving-bench'}
                        or study.get('arena_variant', {}).get('name') != 'fixed-2x-mtp0-wide-kv8-hero-prefix'
                        or study['trials'][0]['server_env'].get('ARENA_PREFIX_CACHE') != 'off'
                        or not bench.get('prefix_check') or not bench.get('prefix_replay')
                        or not (source / bench['transcript']['file']).is_file()
                        or study.get('backend', 'vllm') != 'vllm'):
                    raise ValueError('The prefix-cache smoke requires the two-arm serving bench on the hero prefix arm')
            if TEST == 'prefix-cache-v3-recheck':
                study = json.loads((source / 'configs/comparison.json').read_text())
                bench = study.get('throughput_bench') or {}
                if ([t.get('name') for t in study['trials']] != ['tail-int4']
                        or not ((study.get('cache_experiment') or {}).get('prefix_cache') or {}).get('request_end_tail')
                        or not bench.get('tail_check') or bench.get('prefix_replay')
                        or not (source / bench['tail_check']['file']).is_file()
                        or b'MAX_PER_PARENT = 128' not in (source / 'src/arena/prefix_tail_runtime.py').read_bytes()):
                    raise ValueError('The prefix-cache v3 recheck is the int4 tail check alone with the 128-entry cap')
            if TEST == 'prefix-cache-v3-smoke':
                study = json.loads((source / 'configs/comparison.json').read_text())
                bench = study.get('throughput_bench') or {}
                arms = [t.get('name') for t in study['trials']]
                if (arms != ['tail-int4', 'tail-int8'] or {t['kind'] for t in study['trials']} != {'serving-bench'}
                        or not ((study.get('cache_experiment') or {}).get('prefix_cache') or {}).get('request_end_tail')
                        or 'ARENA_SERVING_KERNELS' not in study['trials'][1]['server_env']
                        or not bench.get('tail_check') or not bench.get('prefix_replay')
                        or not (source / bench['tail_check']['file']).is_file()
                        or not (source / bench['transcript']['file']).is_file()
                        or study.get('backend', 'vllm') != 'vllm'):
                    raise ValueError('The prefix-cache v3 smoke requires the int4/int8 tail-check serving bench on '
                                     'hero-v2 with request_end_tail')
            if TEST == 'kernel-stack':
                study = json.loads((source / 'configs/comparison.json').read_text())
                if ([(t['name'], t['kind']) for t in study['trials']]
                        != [('stack-mtp3', 'serving-bench'), ('stack-mtp0-kv8', 'serving-bench')]
                        or not study.get('stack_smoke') or not study.get('throughput_bench', {}).get('h2d_toggle')
                        or study['server'].get('stack_kernels') != {'async_h2d': 'toggle'}
                        or not study['server'].get('serving_kernels')
                        or study['trials'][1].get('server_env', {}).get('ARENA_MTP_TOKENS') != '0'
                        or study.get('backend', 'vllm') != 'vllm'):
                    raise ValueError('The stack smoke requires the MTP-3 stack arm (toggle) then the MTP-0 kv8 stack arm')
            if TEST == 'draft-head':
                study = json.loads((source / 'configs/comparison.json').read_text())
                if (study['trials'] != [{'name': 'draft-head', 'kind': 'draft-head'}]
                        or not study['server'].get('draft_head', {}).get('dump')
                        or not (source / study['draft_head']['data'] / 'train.npz').is_file()):
                    raise ValueError('The draft-head test requires the draft-head study and its data')
            if TEST == 'kv8-smoke':
                study = json.loads((source / 'configs/comparison.json').read_text())
                bench = study.get('throughput_bench') or {}
                if (study['trials'] != [{'name': 'serving-bench', 'kind': 'serving-bench'}]
                        or not (study['server'].get('serving_kernels') or {}).get('kv_cache_8bit')
                        or not bench.get('quality') or not (source / bench['quality']['reference']).is_file()
                        or study.get('backend', 'vllm') != 'vllm'):
                    raise ValueError('The kv8 smoke requires the serving bench on the kv8 arm with its reference')
            if TEST == 'expert-quant-bench':
                study = json.loads((source / 'configs/comparison.json').read_text())
                if ([(t['name'], t['kind'], t.get('phase')) for t in study['trials']]
                        != [('expert-quant', 'expert-quant', 'bench')]
                        or study['server'].get('expert_capture') is not True or study['server'].get('mtp_tokens') != 0
                        or not (source / 'configs/expert-quant/stage1.json').is_file()
                        or study.get('backend', 'vllm') != 'vllm'):
                    raise ValueError('The expert-quant bench requires the stage-1 study on the capture overlay')
                if not Path(study['expert_quant']['bf16_dir'], 'model.safetensors.index.json').is_file():
                    raise FileNotFoundError('The BF16 expert shards are missing (scripts/modal_expert_bf16.py)')
            if TEST in ('expert-quant-calibrate', 'expert-quant-calibrate-v2'):
                study = json.loads((source / 'configs/comparison.json').read_text())
                if ([(t['name'], t['kind'], t.get('phase')) for t in study['trials']]
                        != [('expert-quant', 'expert-quant', 'calibrate')]
                        or study['server'].get('expert_capture') is not True or study['server'].get('mtp_tokens') != 0
                        or not (source / 'configs/expert-quant/calib.json').is_file()):
                    raise ValueError('The expert-quant calibration requires the stage-2 study on the capture overlay')
                bf16 = Path(study['expert_quant']['bf16_dir'])
                index = json.loads((bf16 / 'model.safetensors.index.json').read_text())['weight_map']
                missing = sorted({index[f'model.language_model.layers.{i}.mlp.experts.{p}'] for i in range(48)
                                  for p in ('gate_up_proj', 'down_proj')} - {p.name for p in bf16.iterdir()})
                if missing:
                    raise FileNotFoundError(f'BF16 expert shards missing: {missing[:4]} ({len(missing)})')
            if TEST == 'expert-quant-eval':
                study = json.loads((source / 'configs/comparison.json').read_text())
                if ({t['kind'] for t in study['trials']} != {'kernel-quality'} or study['trials'][0]['name'] != 'eq-ref'
                        or study.get('expert_quant', {}).get('phase') != 'evaluate'):
                    raise ValueError('The expert-quant evaluation requires the kernel-quality study')
                if study['notebook_seconds'] > TIMEOUT - 60:
                    raise ValueError('The expert-quant evaluation study needs more time than the Modal test gives')
            if TEST == 'drift-long':
                study = json.loads((source / 'configs/comparison.json').read_text())
                if ([t['name'] for t in study['trials']] != ['dl-ref', 'dl-hero', 'dl-kv8']
                        or {t['kind'] for t in study['trials']} != {'kernel-quality'} or 'drift_long' not in study
                        or study['server'].get('mtp_tokens') != 0):
                    raise ValueError('The drift test requires the drift-long kernel-quality study')
                if study['notebook_seconds'] > TIMEOUT - 60:
                    raise ValueError('The drift study needs more time than the Modal test gives')
            if TEST == 'kv4-drift':
                study = json.loads((source / 'configs/comparison.json').read_text())
                if ([t['name'] for t in study['trials']] != ['kv4-ref', 'kv4-int4', 'kv4-k8v4']
                        or {t['kind'] for t in study['trials']} != {'kernel-quality'} or 'kv4_drift' not in study
                        or study['server'].get('mtp_tokens') != 0):
                    raise ValueError('The kv4 drift test requires the kv4-drift kernel-quality study')
                if study['notebook_seconds'] > TIMEOUT - 60:
                    raise ValueError('The kv4 drift study needs more time than the Modal test gives')
            if TEST == 'kv8-prefill-check':
                study = json.loads((source / 'configs/comparison.json').read_text())
                trials = study['trials']
                if ([t['name'] for t in trials] != ['kv8p-ref', 'kv8p-bf16']
                        or {t['kind'] for t in trials} != {'kernel-quality'} or 'kv8_prefill_check' not in study
                        or study['server'].get('mtp_tokens') != 0
                        or 'kv8_prefill' in study['server']['serving_kernels']
                        or json.loads(trials[1]['server_env']['ARENA_SERVING_KERNELS']).get('kv8_prefill') != 'bf16'):
                    raise ValueError('The kv8 prefill check requires the two-trial kernel-quality study')
                if study['notebook_seconds'] > TIMEOUT - 60:
                    raise ValueError('The kv8 prefill check needs more time than the Modal test gives')
            if TEST == 'bench':
                study = json.loads((source / 'configs/comparison.json').read_text())
                if (study['trials'] != [{'name': 'serving-bench', 'kind': 'serving-bench'}]
                        or not study.get('throughput_bench') or study.get('backend', 'vllm') != 'vllm'):
                    raise ValueError('Serving benchmark fallback requires the serving-bench study on vLLM')
            if TEST == 'paired-30m':
                study = json.loads((source / 'configs/comparison.json').read_text())
                if (len(set(study['games'])) != 25 or study['wall_seconds'] != 1800
                        or study['trials'] != [{'name': 'duck-transfer', 'kind': 'duck-transfer'},
                                               {'name': 'qwen-c', 'kind': 'pi'}]
                        or study.get('backend', 'vllm') != 'vllm'):
                    raise ValueError('Paired fallback requires both agents, 25 games, 1800 seconds and vLLM')
        else:
            manifest = json.loads((source / "build-manifest.json").read_text())
            for rel, digest in manifest["files"].items():
                if hashlib.sha256((source / rel).read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f"Bundle hash mismatch: {rel}")
            sys.path[:0] = [str(source / "src"), str(source / "kaggle")]
            from sglang_bench import agent_env, launch, runtime
            from sglang_bench.profile import RECIPE_DEFAULT

            # The full image already ships cudart and CUDA linker stubs. The link
            # repairs are only needed for Kaggle's intentionally reduced archive.
            env = launch.recipe_env(RECIPE_DEFAULT, cache_base=runtime.SCRATCH / "cache",
                                    cuda_home=runtime.CUDA_HOME, repo_root=runtime.REPO_ROOT,
                                    scratch=runtime.SCRATCH / "tmp")
            check = runtime.check(env)
            check["model"] = runtime.verify_model()
            (output / "runtime-check.json").write_text(json.dumps(check, indent=2, default=str))
            if check["problems"]:
                raise RuntimeError(str(check["problems"]))
            install = agent_env.install(source, runtime.SCRATCH)
            study = json.loads((source / "study.json").read_text())
            study["notebook_seconds"] = TIMEOUT - 60
            if study["mode"] != TEST:
                raise ValueError("Bundle study does not match the selected validation test")
            study_path = output / "study-definition.json"
            study_path.write_text(json.dumps(study, indent=2))
            command = [install["agent_python"], "-u", "-m", "sglang_bench.bench",
                       "--source", str(source), "--study", str(study_path), "--output", str(output),
                       "--games-dir", "/assets/games", "--notebook-start-epoch", str(started)]
        # Live visibility (no Modal run had live logs): the bootstrap's output is teed to
        # the container's stdout (`modal app logs <app>` streams it) and the runs volume is committed every
        # two minutes, so progress files can be read mid-run with `modal volume get arc-prolong-runs`.
        import threading

        def pump(pipe, log):
            for line in iter(pipe.readline, b""):
                log.write(line)
                log.flush()
                sys.stdout.buffer.write(line)
                sys.stdout.flush()

        def committer(stop):
            while not stop.wait(120):
                try:
                    runs.commit()
                except Exception as exc:  # a failed interim commit must not end the run
                    print(f"interim volume commit failed: {exc}", flush=True)

        def stop_watcher(stop):
            # Manual graceful stop (29 September, arena.early_stop): `scripts/arena/request_stop.py RUN_ID` puts
            # validation/<run>/arena/live/stop-request on the runs volume. This container holds open files on that
            # volume, so it cannot reload it; the listing comes from the Modal API instead. The trial ends at its next
            # tick when the local file exists, and every evidence file is written as at a wall end.
            folder, reported = f"validation/{run_id}/arena/live", False
            while not stop.wait(STOP_POLL_SECONDS):
                try:
                    names = {Path(entry.path).name for entry in runs.listdir(folder)}
                except Exception as exc:  # no live folder yet, or the API is unavailable: keep polling
                    if not reported:
                        print(f"stop-request watcher: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                        reported = True
                    continue
                if STOP_FILE in names or (output / "arena/live" / STOP_FILE).exists():
                    Path(STOP_LOCAL).write_text(json.dumps({"source": f"{folder}/{STOP_FILE}",
                                                            "epoch": round(time.time(), 1)}) + "\n")
                    print(json.dumps({"event": "stop_request_seen", "path": f"{folder}/{STOP_FILE}"}), flush=True)
                    return

        with (output / "bootstrap.log").open("wb") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            stop = threading.Event()
            threads = [threading.Thread(target=pump, args=(process.stdout, log), daemon=True),
                       threading.Thread(target=committer, args=(stop,), daemon=True)]
            if KIND == "arena":
                threads.append(threading.Thread(target=stop_watcher, args=(stop,), daemon=True))
            for thread in threads:
                thread.start()
            try:
                process.wait(timeout=max(1, started + TIMEOUT - 20 - time.time()))
            finally:
                stop.set()
                threads[0].join(timeout=10)
        identity["exit_code"] = process.returncode
        if process.returncode:
            raise RuntimeError(f"Bootstrap exited {process.returncode}; inspect {output}/bootstrap.log")
    except BaseException as exc:
        identity["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if process is not None and process.poll() is None:
            import signal
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        serving_work = Path("/tmp/mxfp8" if KIND == "vllm" else "/tmp/arena")
        if KIND in {"arena", "vllm"} and (serving_work / "serving").exists():
            dest = output / ("mxfp8" if KIND == "vllm" else "arena") / "serving"
            dest.mkdir(parents=True, exist_ok=True)
            for f in (serving_work / "serving").iterdir():
                if f.is_file() and f.suffix in {".json", ".log"}:
                    shutil.copyfile(f, dest / f.name)
        identity["elapsed_seconds"] = time.time() - started
        (output / "execution.json").write_text(json.dumps(identity, indent=2))
        runs.commit()
        if TEST in ("expert-quant-calibrate", "expert-quant-calibrate-v2"):
            assets.commit()  # the replacement shards (the assets volume, served from /kaggle/input)
    return str(output)


@app.function(image=image, cpu=1, memory=2048, timeout=300, retries=0, block_network=True)
def probe():
    """CPU-only dry run of the image (29 September: three test-slot jobs were lost to sizing or packaging): this
    launcher imports in the container, the bundle is in place and every Python source of it compiles, and the study
    names its trials. No GPU, no ledger entry."""
    import importlib
    import py_compile

    importlib.import_module("modal_validation")
    source = Path("/opt/arc-validation")
    failed = []
    count = 0
    for path in sorted((source / "src").rglob("*.py")):
        count += 1
        try:
            py_compile.compile(str(path), doraise=True, cfile="/tmp/probe.pyc")
        except py_compile.PyCompileError as exc:
            failed.append(f"{path}: {exc.msg[:200]}")
    study = json.loads((source / "configs/comparison.json").read_text())
    return {"test": TEST, "bundle_files": sum(1 for _ in source.rglob("*")), "python_sources": count,
            "compile_failures": failed, "trials": [t["name"] for t in study["trials"]],
            "notebook_seconds": study.get("notebook_seconds"), "timeout": TIMEOUT}


@app.local_entrypoint()
def main(run_id: str = "", probe_only: bool = False):
    if probe_only:
        result = probe.remote()
        print(json.dumps({"probe": result}))
        if result["compile_failures"]:
            raise RuntimeError(f"probe: {len(result['compile_failures'])} bundle sources do not compile")
        return
    if not run_id:
        raise ValueError("--run-id is required")
    bound = reserve(run_id, KIND, TIMEOUT, test=TEST)
    folder = {"paired-30m": "docs/validation-20260923", "duck-30m": "docs/validation-20260924",
              "bench": "docs/validation-20260924", "evolve": "docs/validation-20260924", "evolve-rehearsal": "docs/validation-20260926", "bench-ab": "docs/validation-20260927",
              "duck-60m": "docs/validation-20260926", "duck-probe": "docs/validation-20260926",
              "level-bench": "docs/validation-20260927",
              "duck-110": "docs/validation-20260927",
              "duck-110-kernels": "docs/validation-20260927",
              "kernel-smoke": "docs/validation-20260927",
              "tweaks-smoke": "docs/validation-20260927",
              "kernel-stack": "docs/validation-20260927",
              "hero-smoke": "docs/validation-20260927",
              "kernel-phase6": "docs/validation-20260927",
              "kernel-phase6b": "docs/validation-20260927",
              "prefix-cache-smoke": "docs/validation-20260927",
              "prefix-cache-v3-smoke": "docs/validation-20260927",
              "prefix-cache-v3-recheck": "docs/validation-20260927",
              "draft-head": "docs/validation-20260927",
              "kv8-smoke": "docs/validation-20260927",
              "expert-quant-bench": "docs/validation-20260927", "expert-quant-calibrate": "docs/validation-20260927",
              "expert-quant-eval": "docs/validation-20260927", "expert-quant-calibrate-v2": "docs/validation-20260927",
              "drift-long": "docs/validation-20260927",
              "kv4-drift": "docs/validation-20260927",
              "kv8-prefill-check": "docs/validation-20260927",
              "expert-usage": "docs/validation-20260927",
              "expert-usage-measure": "docs/validation-20260927",
              "expert-prune-check": "docs/validation-20260927",
              "expert-prune-levels": "docs/validation-20260927",
              "expert-prune-hang": "docs/validation-20260927",
              "hero-v2-smoke": "docs/validation-20260927", "hero-v2-2h": "docs/validation-20260927",
              "hero-v2-128k-int4": "docs/validation-20260927", "hero-v2-streams": "docs/validation-20260927",
              "hero-v2-int4-streams": "docs/validation-20260927", "hero-v2-128k-streams": "docs/validation-20260927",
              "hero-v2-blocks": "docs/validation-20260927",
              "hero-v2-samples-plans": "docs/validation-20260927",
              "holo-smoke": "docs/validation-20260927", "holo-60m": "docs/validation-20260927",
              "hero-v2-precheck": "docs/validation-20260927", "prefix-unit-int4": "docs/validation-20260927",
              "prefix-unit-int4-tf": "docs/validation-20260927",
              "holo-soak": "docs/validation-20260927",
              }.get(TEST, "docs/validation-20260922")
    if TEST in ("campaign-9h", "research-model", "decode-sweep", "spec-sweep"):
        folder = "docs/validation-20260930"
    if TEST in ("hx4-level", "hx4-quality"):
        folder = "docs/validation-20261007"
    if TEST in ("competition-preview", "competition-preview-1h", "competition-preview-4h", "competition-preview-8h"):
        folder = "docs/validation-20261001"
    record = ROOT / folder / f"modal-{run_id}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    data = {"state": "submission_intent", "run_id": run_id, "kind": KIND, "test": TEST,
            "charged_bound_usd": bound,
            "platform_policy": ("Modal only: offline test of a Kaggle submission archive (no Kaggle GPU quota)" if TEST in ("competition-preview", "competition-preview-1h", "competition-preview-4h", "competition-preview-8h") else "Modal only: seven authorized nine-hour benchmarks plus user-requested additional nine-hour arms and <=4 one-hour research pilots" if TEST == "campaign-9h" else "kaggle-first; 30-minute queue fallback" if TEST == 'paired-30m'
                                else "kaggle-first; 10-minute queue fallback" if TEST in ('duck-30m', 'duck-60m', 'bench', 'bench-ab')
                                else "modal probe beside the Kaggle runs" if TEST == 'duck-probe'
                                else "modal-only; one Modal GPU job at a time" if TEST in ('level-bench', 'duck-110', 'duck-110-kernels', 'kernel-smoke', 'tweaks-smoke', 'draft-head', 'kv8-smoke', 'kernel-stack', 'hero-smoke', 'prefix-cache-smoke', 'kernel-phase6', 'kernel-phase6b', 'expert-quant-bench',
                                                                                                          'expert-quant-calibrate', 'expert-quant-eval', 'expert-quant-calibrate-v2',
                                                                                                           'expert-usage', 'expert-usage-measure', 'expert-prune-check', 'expert-prune-levels',
                                                                                                           'expert-prune-hang')
                                else "modal-only; the test GPU slot beside the hero-v2 slot"
                                if TEST in ('drift-long', 'kv4-drift', 'holo-smoke', 'holo-60m', 'hero-v2-precheck', 'kv8-prefill-check', 'holo-soak', 'prefix-unit-int4', 'prefix-unit-int4-tf',
                                            'prefix-cache-v3-smoke', 'prefix-cache-v3-recheck')
                                else "modal-only; dedicated hero-v2 GPU slot beside the test slot"
                                if TEST in ('hero-v2-smoke', 'hero-v2-2h', 'hero-v2-128k-int4', 'hero-v2-streams')
                                else "modal-only; runs A/B in parallel, hero-v2 slot and a third GPU (user exception, "
                                     "29 September)"
                                if TEST in ('hero-v2-int4-streams', 'hero-v2-128k-streams')
                                else "modal-only; dedicated hero-v2 GPU slot (/tmp/arc-gpu-lock2)"
                                if TEST == 'hero-v2-blocks'
                                else "modal-only; one free GPU slot (test or hero-v2), run B + parallel samples + "
                                     "action plans"
                                if TEST == 'hero-v2-samples-plans'
                                else "modal-only"),
            "bundle": str(BUNDLE)}
    if TEST == "hero-v2-blocks":
        # The block plan with the block-1 source list (user requirement: 20 sources + 17 second clones).
        study = json.loads((BUNDLE / "configs/comparison.json").read_text())
        plan = study["blocks_plan"]
        data["blocks_plan"] = {"block_sizes": plan["block_sizes"], "block_seconds": plan["block_seconds"],
                               "run_stop_seconds": plan["run_stop_seconds"], "block1_sources": plan["block1"]["sources"],
                               "block1_second_clones": plan["block1"]["second_clones"],
                               "window": study["windows"][0], "streams": study["active_limit"],
                               "max_num_seqs": study["server"]["max_num_seqs"]}
    record.write_text(json.dumps(data, indent=2))
    call = benchmark.spawn(run_id)
    data.update(state="submitted", function_call_id=call.object_id)
    record.write_text(json.dumps(data, indent=2))
    print(json.dumps(data))
