"""Throughput arm (3 October): MTP-1 vs MTP-0 decode at the competition's long contexts, wide decode tiles on the MTP-1
verify batches, and the verify-path drift, on the dfr-combined v3 serving (int3 Humming, prune12, int4 KV, 0.92).

One container, three serving-bench trials (the bootstrap restarts the server when a trial's ``server_env`` differs):

1. ``mtp1``: the tp-mtp1-dfr serving as built (MTP-1, BF16 GDN state, NvFp4 draft experts, 16K draft vocab, prefix
   cache v2 + ``spec_decode``, 34 sequences, graphs [2..68]);
2. ``mtp1-wide``: the same + ``ARENA_DENSE_KERNELS`` = the served dense setting + ``mid_wide`` (buckets 40-72, the
   prune12 router shape; tuned in the kernel warmup, error-checked against FP32): the verify batches are 48-68 rows,
   past the phase-5 plan's 32;
3. ``mtp0``: ``ARENA_MTP_TOKENS`` 0 with the v3 serving kernels and prefix cache (no draft stack, MTP-0 graphs to
   34): the same contexts at MTP-0.

Each trial runs ``arena.spec_bench``: every context prefilled once, decode points at 24 / 30 / 34 streams (clean
windows: every stream running, none waiting, no prompt tokens computed for 5 consecutive 1-s samples, then 30 s),
then the decode-vs-teacher-forced drift check on the 16 shortest contexts (256 sampled tokens each).

Contexts: real dfr-combined v3 requests (``scripts/spec_decode_inputs.py --trial-dir``: the latest logged request of
each game, cut at an assistant turn, rendered with the served template and tokenized; no images: the image tokens
are ~400 each), spread over [low, high] tokens.

Pre-stated reading (before launch): MTP-1 / MTP-0 generated tok/s at equal streams (expected 1.25-1.35 at 18K in the
30 Sept pilots; long contexts add indexer work per verify token); wide tiles pass when the MTP-1 30-stream step is
>= 0.5 ms faster than ``mtp1`` and the drift summary (mean KL, top-1, |delta logprob|) is within 1.25 x ``mtp1``'s.

    uv run --extra dev python scripts/arena/build_tp_spec_long.py --contexts <spec-contexts.json> --out dist/<name>
"""
import argparse
import json
import pathlib
import runpy
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from build_arena import build, manifest  # noqa: E402

from arena.campaign import components  # noqa: E402
from arena.variants import HERO_PREFIX_CACHE  # noqa: E402

STREAMS, SPARE, FLOOR = 30, 4, 1390
SPEC = {
    "contexts": "bench/spec-contexts.json",
    "decode_counts": [24, 30, 34],
    "output_tokens": 3072,
    "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 20},
    "settle_seconds": 240, "quiet_samples": 5, "window_seconds": 30, "sample_seconds": 1.0,
    "drift": {"prompts": 16, "output_tokens": 256, "top_logprobs": 10, "teacher_concurrency": 4},
}
WIDE_BUCKETS = [40, 48, 56, 64, 72]
MTP0_GRAPHS = [1, 2, 4, 8, 16, 24, 32, 34]
TRIAL_WALL = 1500  # bootstrap kills a trial process at wall + 600 s
TEACHER_FILE = "bench/tf-sequences.json"
PROFILE_SPEC = {**SPEC, "decode_counts": [24, 32], "drift": None, "profile_streams": [32], "profile_seconds": 6,
                "profile_prefill_seconds": 14}
TAIL_SPEC = {**SPEC, "decode_counts": [], "drift": None,
             "mtp_tail_check": {"block": 3136, "sequences": 6, "min_tokens": 23000, "top_logprobs": 20,
                                "max_tokens": 96, "stop_min": 24, "primer_offset": 1500}}
# v3 (4 October): more power for the decode-ended condition (12 sequences, 24 probes each, shorter stop search).
TAIL_SPEC_V3 = {**TAIL_SPEC, "mtp_tail_check": {
    **TAIL_SPEC["mtp_tail_check"], "sequences": 12, "max_tokens": 160, "stop_min": 8,
    "probe_offsets": [1, 2, 3, 5, 7, 10, 13, 16, 20, 25, 31, 40, 50, 64, 80, 100, 125, 150, 190, 230, 300, 380, 470, 600]}}
TAIL_SPEC_V4 = {**TAIL_SPEC_V3, "mtp_tail_check": {**TAIL_SPEC_V3["mtp_tail_check"], "continue_greedy": True,
                                                   "probe_offsets": [2, 3, 5, 7, 10, 13, 16, 20, 25, 31, 40, 50, 64,
                                                                     80, 100, 125, 150, 190, 230, 300, 380, 470, 600]}}
TIMEOUT = 4800


def hero_components():
    hero = components("comp-prune25")
    hero.update(expert_prune="prune12", expert_int3=16, window=130816, focus_slots=STREAMS, density_blocks=FLOOR,
                graph_capture=2 * (STREAMS + SPARE), backlog_fix=False, budget_policy="priority",
                dfr_harness=True, drain_half=True, mtp_tokens=1, mtp_prefix_cache=True)
    return hero


def bench_study(study, contexts, arms=("mtp1", "mtp1-wide", "mtp0"), teacher=None, profile=False, tail=False):
    study = json.loads(json.dumps(study))
    server = study["server"]
    if server.get("mtp_tokens") != 1 or not (study["cache_experiment"]["prefix_cache"] or {}).get("spec_decode"):
        raise ValueError("the long-context spec bench builds on the MTP-1 + spec_decode serving")
    env = {}
    for trial in study.get("trials") or []:
        env.update(trial.get("server_env") or {})
    dense = dict(server["dense_kernels"])
    wide = {**dense, "mid_wide": True, "mid_wide_shapes": [[448, 2560]], "mid_wide_buckets": WIDE_BUCKETS}
    from arc_infra.dense_kernels_patch import validate as validate_dense

    validate_dense(wide)
    kernels_mtp0 = {k: v for k, v in server["serving_kernels"].items()
                    if k not in ("mamba_ssm_cache_dtype", "draft_experts", "draft_vocab")}
    prefix_mtp0 = dict(HERO_PREFIX_CACHE)
    kernels_fp8 = {**server["serving_kernels"], "indexer_fp8": True}
    from arc_infra.serving_kernels_patch import validate as validate_kernels

    validate_kernels(kernels_fp8)
    trials = [
        {"name": "mtp1", "kind": "serving-bench", "competition": False, "wall_seconds": TRIAL_WALL,
         "server_env": dict(env)},
        {"name": "mtp1-wide", "kind": "serving-bench", "competition": False, "wall_seconds": TRIAL_WALL,
         "server_env": {**env, "ARENA_DENSE_KERNELS": json.dumps(wide)}},
        {"name": "mtp0", "kind": "serving-bench", "competition": False, "wall_seconds": TRIAL_WALL,
         "server_env": {**env, "ARENA_MTP_TOKENS": "0", "ARENA_CUDAGRAPH_CAPTURE_SIZES": ",".join(map(str, MTP0_GRAPHS)),
                        "ARENA_MAX_CUDAGRAPH_CAPTURE_SIZE": str(MTP0_GRAPHS[-1]),
                        "ARENA_SERVING_KERNELS": json.dumps(kernels_mtp0),
                        "ARENA_PREFIX_CACHE": json.dumps(prefix_mtp0)}},
        # Throughput arm (3 October, second bench): float8 indexer keys on top of the wide tiles.
        {"name": "mtp1-wide-fp8", "kind": "serving-bench", "competition": False, "wall_seconds": TRIAL_WALL,
         "server_env": {**env, "ARENA_DENSE_KERNELS": json.dumps(wide),
                        "ARENA_SERVING_KERNELS": json.dumps(kernels_fp8)}},
    ]
    # Teacher-forced indexer-key comparison (3 October): the same two servers, fixed sequences only.
    trials += [{**trials[1], "name": "tf-bf16"}, {**trials[3], "name": "tf-fp8"}]
    # Kernel profile of the v4 serving (4 October): MTP-1 + wide tiles + float8 indexer keys, torch profiler on.
    trials.append({**trials[3], "name": "prof", "server_env": {**trials[3]["server_env"], "ARENA_TORCH_PROFILER": "1"}})
    # MTP-2 on the v4 serving (4 October): the same kernels, draft length 2 (4-bit block 3,200 tokens, 17 fixed blocks),
    # verify graphs 3 x the MTP-0 ladder.
    trials.append({**trials[3], "name": "mtp2-wide-fp8", "server_env": {
        **trials[3]["server_env"], "ARENA_MTP_TOKENS": "2",
        "ARENA_DENSE_KERNELS": json.dumps({**wide, "mid_wide_buckets": [40, 48, 56, 64, 72, 80, 88, 96]}),
        "ARENA_CUDAGRAPH_CAPTURE_SIZES": ",".join(str(3 * n) for n in MTP0_GRAPHS),
        "ARENA_MAX_CUDAGRAPH_CAPTURE_SIZE": str(3 * MTP0_GRAPHS[-1])}})
    # MTP request-end tail (4 October, phase 3): the v4 serving + prefix cache v3 request_end_tail with SPEC_TAIL;
    # the tail stats line after every registration.
    tail_cache = {**study["cache_experiment"]["prefix_cache"], "request_end_tail": True}
    trials.append({**trials[3], "name": "tail-mtp", "server_env": {
        **trials[3]["server_env"], "ARENA_PREFIX_CACHE": json.dumps(tail_cache), "ARC_TAIL_LOG_EVERY": "1"}})
    # Stream curve (4 October, phase 3): the v4 serving at 32-48 streams (a --max-seqs 48 build: verify
    # graphs to 96 rows, wide buckets to 96).
    trials.append({**trials[3], "name": "curve", "server_env": {
        **trials[3]["server_env"],
        "ARENA_DENSE_KERNELS": json.dumps({**wide, "mid_wide_buckets": [40, 48, 56, 64, 72, 80, 88, 96]})}})
    # Phase 5 (5 October): + tail_spec_acceptance (tails for finishes without an in-flight step) + keep-boundary.
    accept_cache = {**tail_cache, "tail_keep_boundary": True, "tail_spec_acceptance": True}
    trials.append({**trials[3], "name": "tail-accept", "server_env": {
        **trials[3]["server_env"], "ARENA_PREFIX_CACHE": json.dumps(accept_cache), "ARC_TAIL_LOG_EVERY": "1"}})
    by_name = {t["name"]: t for t in trials}
    study["trials"] = [by_name[a] for a in arms]
    study["throughput_bench"] = {"prompt_tokens": [], "concurrency": [], "output_tokens": 64, "shared_prefix": None,
                                 "points": [], "sanity": (study.get("throughput_bench") or {}).get("sanity"),
                                 "spec_sweep": ({"v3": TAIL_SPEC_V3, "v4": TAIL_SPEC_V4}.get(tail, TAIL_SPEC)) if tail else PROFILE_SPEC if profile else dict(SPEC)
                                 if teacher is None else {
                                     **SPEC, "decode_counts": [], "drift": None,
                                     "teacher_force": {"sequences": TEACHER_FILE, "top_logprobs": 10,
                                                       "positions": 512}}}
    data = json.loads(pathlib.Path(contexts).read_text())
    study["tp_spec_long"] = {
        "arms": [t["name"] for t in study["trials"]], "wide_buckets": WIDE_BUCKETS,
        "contexts": {"source": data["source"], "count": data["count"], "mean_tokens": data["mean_tokens"],
                     "low": data["low"], "high": data["high"], "sha256": data["contexts_sha256"]},
        "pass_bar": {"wide": "mtp1-wide 30-stream step <= mtp1 - 0.5 ms; drift summary within 1.25 x mtp1's",
                     "mtp1": "reported: MTP-1 / MTP-0 generated tok/s at 24 / 30 / 34 streams"}}
    study.pop("blocks_plan", None)
    study.update(notebook_seconds=TIMEOUT - 60, kernel_timeout=TIMEOUT, wall_seconds=TRIAL_WALL)
    return study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dependencies", action="store_true")
    parser.add_argument("--arms", default="mtp1,mtp1-wide,mtp0", help="comma-separated trial order")
    parser.add_argument("--teacher-force", default=None, help="sequences JSON for the teacher-forced arms")
    parser.add_argument("--profile", action="store_true", help="torch-profiler spec (prefill + 32-stream decode)")
    parser.add_argument("--tail", nargs="?", const="v2", default=None,
                        help="MTP request-end tail check (arena.prefix_replay); --tail v3 = 12 sequences x 24 probes")
    parser.add_argument("--max-seqs", type=int, default=None, help="server max_num_seqs (stream curve, bench only)")
    parser.add_argument("--decode-counts", default=None, help="comma-separated decode points (stream curve)")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    output, study = build(out.name, output=out, dependencies=args.dependencies, paired_30m=True,
                          arena_variant="fixed-2x-mtp0-wide-kv8-hero", hero_v2=hero_components(),
                          hero_v2_served_blocks=1839 - 192, exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"],
                          gpu_memory_utilization="0.92", clone_instances=110)
    study = bench_study(study, args.contexts, tuple(args.arms.split(",")), teacher=args.teacher_force,
                        profile=args.profile, tail=args.tail)
    if args.max_seqs:
        # Bench-only stream curve: more sequences and MTP-1 verify graphs to 2 x max_num_seqs rows.
        server = study["server"]
        server["max_num_seqs"] = args.max_seqs
        study["generation_slots"] = args.max_seqs  # the bench's KVGate admits at most generation_slots streams
        server["cudagraph_capture_sizes"] = sorted({*server["cudagraph_capture_sizes"],
                                                    *range(48, 2 * args.max_seqs + 1, 16), 2 * args.max_seqs})
        for key in ("max_cudagraph_capture_size",):
            if key in server:
                server[key] = 2 * args.max_seqs
    if args.decode_counts:
        sweep = study["throughput_bench"]["spec_sweep"]
        sweep.update(decode_counts=[int(n) for n in args.decode_counts.split(",")], drift=None)
    (output / "bench").mkdir(exist_ok=True)
    shutil.copyfile(args.contexts, output / SPEC["contexts"])
    if args.teacher_force:
        shutil.copyfile(args.teacher_force, output / TEACHER_FILE)
    (output / "configs/comparison.json").write_text(json.dumps(study, indent=1) + "\n")
    identity = json.loads((output / "MANIFEST.json").read_text())
    identity["files"] = manifest(output)
    (output / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    runpy.run_path(str(output / "bootstrap.py"))["verify"](output)
    print(json.dumps({"output": str(output), "trials": [t["name"] for t in study["trials"]],
                      "server": {k: study["server"].get(k) for k in ("mtp_tokens", "max_num_seqs",
                                                                       "cudagraph_capture_sizes")},
                      "contexts": study["tp_spec_long"]["contexts"]}))


if __name__ == "__main__":
    main()
