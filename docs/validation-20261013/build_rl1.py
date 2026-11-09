"""RL step 1: the v8.2 level bench on Flash's own NVFP4 checkpoint with an
unmerged LoRA, one trial ``rl-loop`` (``arena.rl_loop``) that plays rounds of sb26, trains on the same B300
(``scripts/posttrain/rl1_trainer.py`` in the Axolotl image) and evaluates on lp85.

Built from ``docs/validation-20261007/build_hx4.py --serving=v82`` (harness keys untouched); then only the serving
side changes: no INT3 overlay, no expert mask, no draft head, no MTP, no int8 dense / SM120 serving kernels; BF16 KV;
LoRA rank 32 with Marlin experts and sleep mode, a 32 GiB KV pool. Every changed key is recorded in ``study["rl1"]``.

argv: output --mode=probe|loop --draft-head-overlay=DIR (build_v82 requires one; removed from this bundle)
[--notebook=N] (bootstrap seconds; probe 10,800, loop 45,000) [--rounds=25] [--lr=6e-5] [--round0-from=PATH] [--round0-play-from=PATH]
[--concurrency=32] [--window=24576] [--probe-updates=4] [--loop-lora] [--tis-cap=C] [--ple-queue-wait=S] [--rl=JSON] [--rank=32] [--optimizer=adamw8bit|adamw32] [--fit-trials=JSON --first-fit
--select-window=24576]
"""
import hashlib
import json
import runpy
import shutil
import sys
from pathlib import Path

ROOT = Path.cwd()
POINTS = "data/stuck_bench/rl1_points.json"  # sb26 and lp85, fresh games from level 1 (easy10_through.json)
PREFIX_CACHE = ("mamba_align_fix", "mamba_keep_state", "mamba_demote_superseded")
# KV pool by bytes (32 GiB, about 1.28M tokens): the serving test's server used 20 GiB above its 0.5 share at 32
# streams, and the resident trainer holds about 105 GiB; the fork accepts no utilization below 0.5
SERVER_ENV = {"ARENA_KV_CACHE_MEMORY_BYTES": str(32 * 1024**3), "ARENA_GPU_MEMORY_UTILIZATION": "0.5",
              "ARENA_PROMPT_TOKENS_DETAILS": "1",
              "ARENA_READY_GUARD": "180,3000", "ARENA_SETUP_TIMEOUTS": "2700,3600"}
DROPPED = ("hero_v2", "arena_variant", "focus_oversubscription", "serving_profile")  # serving descriptors of the INT3 build
DROPPED_SERVER = ("dense_kernels", "serving_kernels", "stack_kernels", "expert_prune", "expert_overlay", "draft_head")
TRAINER_FILES = ("scripts/posttrain/rl1_trainer.py", "scripts/posttrain/flash_qlora_job.py",
                 "scripts/posttrain/rl_job.py", "src/posttrain/rl_rollouts.py")


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    mode = arg("mode", "probe")
    if mode not in ("probe", "loop", "restore", "final"):
        raise SystemExit("--mode is probe or loop")
    notebook = int(arg("notebook", "10800" if mode == "probe" else "45000"))
    concurrency = int(arg("concurrency", "32"))  # 32 attempts per round
    own = sys.argv[:]
    sys.argv = ["build_hx4.py", str(out), "--serving=v82", f"--draft-head-overlay={arg('draft-head-overlay')}",
                f"--points={POINTS}", "--budget=30000", "--passes=16", f"--concurrency={concurrency}", "--wall=3600",
                f"--setup={notebook - 3600}", "--split=all", "--window=130816"]
    if concurrency > 48:  # the two-card loop: the server card alone holds the KV pool for every attempt at once
        sys.argv.append(f'--study={{"active_limit": {concurrency}}}')
    try:
        runpy.run_path(str(ROOT / "docs/validation-20261007/build_hx4.py"), run_name="__main__")
    finally:
        sys.argv = own
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
    from build_arena import manifest  # noqa: E402

    from arena import level_bench, rl_loop  # noqa: E402
    from arena.serving import validate_lora, validate_prefix_cache  # noqa: E402

    path = out / "configs/comparison.json"
    study = json.loads(path.read_text())
    if "v82" not in study or study["windows"] != [130816] or study["completion_cap"] != 16384:
        raise SystemExit("not a v8.2 bench study")
    before = json.loads(json.dumps(study))
    server = study["server"]
    seqs = concurrency + 4
    study["server"] = {
        **{k: v for k, v in server.items() if k not in DROPPED_SERVER},
        "mtp_tokens": 0, "max_num_seqs": seqs, "gpu_memory_utilization": SERVER_ENV["ARENA_GPU_MEMORY_UTILIZATION"],
        "max_num_batched_tokens": 8192, "max_cudagraph_capture_size": 32,
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32], "lora": validate_lora(
            {"max_rank": int(arg("rank", "32")), "ple_queue_wait": float(arg("ple-queue-wait", "0")) or None,
             "max_loras": int(arg("max-loras", "1")), "max_cpu_loras": int(arg("max-loras", "1")) + 1})}
    if arg("serving-kernels", None):  # e.g. {"ple_uva": true}: the PLE table read through UVA, no offload process
        from arc_infra.serving_kernels_patch import validate as validate_serving_kernels

        study["server"]["serving_kernels"] = validate_serving_kernels(json.loads(arg("serving-kernels", None)))
    if concurrency > 32:  # the two-card loop plays every attempt at once: decode graphs up to that batch
        top = max(64, -(-seqs // 8) * 8)  # every batch the server can run is a captured size (no eager decode)
        study["server"].update(max_cudagraph_capture_size=top,
                               cudagraph_capture_sizes=[1, 2, 4, 8, 16] + list(range(24, top + 1, 8)))
    cache = study["cache_experiment"]
    study["cache_experiment"] = {**cache, "kv_cache_dtype": "auto", "reclaim_skipped_blocks": False,
                                 "prefix_cache": {k: True for k in PREFIX_CACHE}}
    validate_prefix_cache(study["cache_experiment"]["prefix_cache"])
    for key in DROPPED:
        study.pop(key, None)
    study["served_model"] = "Qwen/Qwen3.8-Flash-Next-NVFP4"
    study["model_datasets"] = []
    points = out / "configs/rl1_points.json"
    shutil.copyfile(ROOT / POINTS, points)
    study["level_bench"] = level_bench.settings({
        **study["level_bench"], "serving": "rl1-flash-nvfp4-lora", "split": "all", "games": ["sb26"],
        "start_points": "configs/rl1_points.json", "token_budget": 30000, "passes": 16, "concurrency": concurrency,
        "rollout": {"top_logprobs": 20, "token_ids": True}, "attempt_wall_seconds": 5000, "min_wall_seconds": 300,
        "action_cap": None, "token_budget_by_point": {"sb26-game": 30000, "lp85-game": 40000}})
    rl = {"mode": mode, "rounds": int(arg("rounds", "25")), "lr": float(arg("lr", "6e-5")),
          "concurrency": concurrency, "train_attempts": concurrency, "round0_from": arg("round0-from"),
          "round0_play_from": arg("round0-play-from"),
          "trainer_script": "rl1/rl1_trainer.py", "rank": int(arg("rank", "32")),
          "optimizer": arg("optimizer", "adamw8bit"), "window": int(arg("window", "24576")),
          "probe_updates": int(arg("probe-updates", "4")), "loop_lora": "--loop-lora" in sys.argv,
          "tis_cap": float(arg("tis-cap", "0"))}
    if arg("limits"):  # owner-set stop rules, e.g. {"pass_reward": 6.0}
        rl["limits"] = json.loads(arg("limits"))
    if arg("rl"):  # further rl_loop settings, e.g. {"train_attempts": 64, "update_mode": "signed"}
        rl.update(json.loads(arg("rl")))
    if arg("kv-gib"):  # a larger play KV pool; the awake server's size follows (32 GiB pool: 148 GiB awake)
        SERVER_ENV["ARENA_KV_CACHE_MEMORY_BYTES"] = str(int(arg("kv-gib")) * 1024**3)
        rl["limits"] = {**rl.get("limits", {}), "awake_gib": 148.0 + int(arg("kv-gib")) - 32}
    if arg("fit-trials"):  # [[mode, window], ...]; with --first-fit a descending list per mode stops at a fit
        rl["fit"] = {"trials": json.loads(arg("fit-trials")), "first_fit": "--first-fit" in sys.argv,
                     "select_window": int(arg("select-window", "24576"))}
    study["rl_loop"] = rl_loop.settings(rl)
    study["generation_slots"] = seqs  # serving.main: max_num_seqs must cover the ordinary slots
    # rl1-probe-v7 was cut at the study's 3,600 s + grace: the loop's wall is the notebook (the bootstrap clamps it
    # to what is left); each bench stops at the loop's deadline, each attempt at attempt_wall_seconds
    study["trials"] = [{"name": "rl-loop", "kind": "rl-loop", "server_env": dict(SERVER_ENV), "wall_seconds": notebook}]
    study["wall_seconds"], study["notebook_seconds"] = notebook, notebook
    study["bootstrap_reserves"] = {"end_seconds": 120, "trial_seconds": 300}
    (out / "rl1").mkdir(exist_ok=True)
    for name in TRAINER_FILES:
        shutil.copyfile(ROOT / name, out / "rl1" / Path(name).name)
    study["rl1"] = {
        "source": "docs/validation-20261013/build_rl1.py", "base": "v8.2 level bench (build_hx4.py --serving=v82)",
        "model": {"before": "Qwen/Qwen3.8-Flash-Next-NVFP4 + INT3 requant expert overlay, 37.5% expert mask",
                  "after": "Qwen/Qwen3.8-Flash-Next-NVFP4 (own checkpoint) + unmerged LoRA 'policy'"},
        "server": {"removed": sorted(k for k in before["server"] if k not in study["server"]),
                   "before": {k: before["server"].get(k) for k in ("mtp_tokens", "max_num_seqs", "gpu_memory_utilization",
                                                                    "max_cudagraph_capture_size")},
                   "after": {k: study["server"].get(k) for k in ("mtp_tokens", "max_num_seqs", "gpu_memory_utilization",
                                                                   "max_cudagraph_capture_size", "lora")}},
        "prefix_cache": {"before": sorted(k for k, v in cache["prefix_cache"].items() if v), "after": list(PREFIX_CACHE)},
        "server_env": {"before": before["trials"][0].get("server_env"), "after": SERVER_ENV},
        "harness_keys_changed": [], "dropped_study_keys": [k for k in DROPPED if k in before],
    }
    harness = ("harness_fixes", "drain_half", "hx_harness", "dfr_harness", "dfr_loop", "loop_rule", "priority",
               "windows", "completion_cap", "temperature", "reasoning_history", "compaction_thinking", "token_budget",
               "context_policy", "v82", "reasoning_effort")
    study["rl1"]["harness_keys_changed"] = [k for k in harness if study.get(k) != before.get(k)]
    if study["rl1"]["harness_keys_changed"] or concurrency > study["active_limit"]:
        raise SystemExit(f"study check failed: {study['rl1']['harness_keys_changed']}")
    for folder in ("draft-head", "configs/expert-prune", "dataset-upload"):
        shutil.rmtree(out / folder, ignore_errors=True)
    study["name"] = out.name
    path.write_text(json.dumps(study, indent=2) + "\n")
    identity = json.loads((out / "MANIFEST.json").read_text())
    identity["files"] = manifest(out)
    (out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    runpy.run_path(str(out / "bootstrap.py"))["verify"](out)
    print(json.dumps({"output": str(out), "manifest_sha256": hashlib.sha256((out / "MANIFEST.json").read_bytes()).hexdigest(),
                      "mode": mode, "server": study["server"], "notebook_seconds": notebook,
                      "rl_loop": {k: study["rl_loop"][k] for k in ("mode", "rounds", "lr", "rank", "window")}}))
