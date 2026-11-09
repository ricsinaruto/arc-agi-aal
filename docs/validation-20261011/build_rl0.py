"""RL restart, step 0: the v8.2 level bench on Qwen/Qwen3.5-4B (BF16, pinned vLLM fork, Flash's
chat template, thinking on; a short thinking-off arm on the level-1 starts), then the training-cost probe. A
measurement: no reward, no update.

It runs ``docs/validation-20261007/build_hx4.py --serving=v82`` unchanged over the DEV level start points
(``data/rl0/dev_l123.json``; level 1 only by default: ), then moves the built study onto the candidate serving path (``arena.candidate_serving``)
and records every key it changed under ``study["rl0"]``. The harness keys are not touched.

Token cap ("the cap based on average flash clears times 2", from Flash's uncapped DEV plays): per
start, max(20,000, 2 x Flash's mean generated tokens to clear that level of that game), rounded to 1,000, from
``--reference`` (main's tools/ops/bench_reference.json: 80 plain v8.2 whole-game attempts, 8 per DEV game, 400K budget,
tokens at each level clear).

argv: output --draft-head-overlay=DIR (build_v82.py requires one; it is removed from this bundle) --reference=FILE
[--cap-floor=20000] [--levels=1] [--budget=54000] (a start the reference has no clear for)
Every live reply is stored with its request, sampled token ids and log-probabilities (``level_bench.rollout``): the
chosen game's attempts are the untrained model's first round for step 1.
[--passes=8] [--concurrency=48] [--wall=4500] (bench) [--probe-wall=720] [--notebook=6300]
[--nothink-passes=4] (attempts per level-1 start of the thinking-off arm; 0: no such arm)
[--window=130816] [--kv-tokens=N] [--stream-margin=0.96] (v5, harness window 65,536, served at the
native context, streams = floor(margin x the measured KV pool / window) so every stream's full context fits the cache at once;
N = the step-0 server's "GPU KV cache size" line)
The launcher's timeout is notebook + 300 s and the bootstrap gets timeout - 60 s: with the defaults the server
setup and the bench's wind-down have 1,140 s before the probe would be skipped.
"""
import hashlib
import json
import runpy
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path.cwd()
POINTS = "data/rl0/dev_l123.json"
THROUGH = "data/stuck_bench/dev_through10.json"  # the play-through points of Flash's reference plays
PROFILE = "qwen35-4b"
FLA = ("fla_core-0.5.2-py3-none-any.whl", "5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761",
       "https://files.pythonhosted.org/packages/2d/ed/dfe19c4da779957eb6a42a26812f9b4e2280bf757a17a71933ff59ffcb98/"
       "fla_core-0.5.2-py3-none-any.whl")
PREFIX_CACHE = ("mamba_align_fix", "mamba_keep_state", "mamba_demote_superseded")  # the set the Holo4 candidate ran
# Qwen3.6 model card samplers (thinking: general tasks; non-thinking: instruct), sent on every request
SAMPLING = {"think": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                      "repetition_penalty": 1.0},
            "nothink": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                        "repetition_penalty": 1.0}}
SERVER_KEYS = ("max_model_len", "prefix_caching", "gpu_memory_utilization", "max_num_batched_tokens", "max_num_seqs")
SERVER_ENV = {"ARENA_STALL_DUMP": "90", "ARENA_PROMPT_TOKENS_DETAILS": "1"}
DROPPED = ("hero_v2", "arena_variant", "focus_oversubscription", "serving_profile")
# With every earlier reasoning block kept, an attempt's requests run to the drain point (113,920 tokens of the 130,816
# window): the ladder ends at the window.
PROBE = {"label_fraction": 0.5, "head_chunk": 2048, "variants": [
    {"storage": "bf16", "lengths": [2048, 32768, 65536, 98304, 131072], "repeat_below": 0, "share": 0.5},
    {"storage": "fp32", "lengths": [8192, 32768, 65536, 98304, 131072], "repeat_below": 0, "share": 1.0}]}


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


def fla_wheel():
    name, digest, url = FLA
    cache = ROOT / "dist/.cache" / name
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            cache.write_bytes(response.read())
    if hashlib.sha256(cache.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"{name}: sha256 differs from the pin")
    return cache


if __name__ == "__main__":
    out = Path(sys.argv[1])
    wall, probe_wall, notebook = int(arg("wall", "4500")), int(arg("probe-wall", "720")), int(arg("notebook", "6300"))
    window = int(arg("window", "130816"))
    concurrency = int(arg("concurrency", "48"))
    if arg("kv-tokens"):
        concurrency = int(float(arg("stream-margin", "0.96")) * int(arg("kv-tokens")) // window)
    own = sys.argv[:]
    sys.argv = ["build_hx4.py", str(out), "--serving=v82", f"--draft-head-overlay={arg('draft-head-overlay')}",
                f"--points={arg('through-points', THROUGH) if arg('arm') else POINTS}", f"--budget={arg('budget', '54000')}", f"--passes={arg('passes', '8')}",
                f"--concurrency={concurrency}", f"--wall={wall}", f"--setup={notebook - wall}", f"--split={arg('split', 'dev')}",
                f"--window={window}"]
    try:
        runpy.run_path(str(ROOT / "docs/validation-20261007/build_hx4.py"), run_name="__main__")
    finally:
        sys.argv = own
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
    from build_arena import manifest  # noqa: E402

    from arena import candidate_profiles, level_bench, train_probe  # noqa: E402
    from arena.candidate_serving import CANDIDATE_LAB  # noqa: E402
    from arena.serving import lab_overrides, validate_prefix_cache  # noqa: E402

    profile = candidate_profiles.PROFILES[arg("profile", PROFILE)]
    inventory = json.loads(profile.manifest.read_text())
    path = out / "configs/comparison.json"
    study = json.loads(path.read_text())
    if "v82" not in study or study["windows"] != [window] or study["completion_cap"] != 16384:
        raise SystemExit("not a v8.2 bench study")
    before = json.loads(json.dumps(study))
    study.update(candidate_profile=profile.name, served_model=profile.model, model_revision=profile.revision,
                 model_datasets=[inventory["dataset"]], model_sources=[], startup_timeout_seconds=1800,
                 reasoning_effort=profile.effort, enable_thinking=True)
    server = study["server"]
    # The candidate path serves the native context only (candidate_serving.start); the KV pool is set by memory.
    served = profile.native_context
    # v8.2's own rule: max_num_seqs = streams + 4 (the bench's concurrency; v8.2's build is run at its 48 streams)
    seqs = concurrency + 4 if (arg("kv-tokens") or arg("arm")) else server["max_num_seqs"]
    study["server"] = {**{k: server[k] for k in SERVER_KEYS}, "max_model_len": served, "max_num_seqs": seqs,
                       "mtp_tokens": profile.mtp_tokens,
                       "max_cudagraph_capture_size": 8 * -(-seqs * (1 + profile.mtp_tokens) // 8)}
    cache = study["cache_experiment"]
    prefix_cache = PREFIX_CACHE + (("spec_decode",) if profile.mtp_tokens else ())  # keep/demote under MTP-k
    study["cache_experiment"] = {**cache, "kv_cache_dtype": "auto", "prefix_cache": {k: True for k in prefix_cache}}
    for key in DROPPED:
        study.pop(key, None)
    study["trials"] = [{"name": "level-bench", "kind": "level-bench", "server_env": dict(SERVER_ENV)}]
    if probe_wall:  # --probe-wall=0: play only (launcher kind rl0-play)
        study["trials"].append({"name": "train-probe", "kind": "train-probe", "server_env": dict(SERVER_ENV),
                                "wall_seconds": probe_wall})
    nothink = int(arg("nothink-passes", "4"))
    arms = [{"name": "think"}, {"name": "nothink", "thinking": False, "levels": [1], "passes": nothink}] if nothink else None
    # Play-through: --arm=think|nothink, one arm per run; each instance a fresh game from level 1 played
    # to --budget generated tokens, levels continuing in the same game; no level, action or per-level time cap.
    through = arg("arm")
    if through not in (None, "think", "nothink"):
        raise SystemExit("--arm is think or nothink")
    if through:
        arms = [{"name": "nothink", "thinking": False}] if through == "nothink" else None
    sampling = SAMPLING[through] if through and arg("sampling", "card") == "card" else None
    reference_path = Path(arg("reference"))
    reference, floor = json.loads(reference_path.read_text()), int(arg("cap-floor", "20000"))
    levels = sorted(int(n) for n in arg("levels", "1").split(","))
    points = out / "configs/stuck_points.json"
    kept = json.loads(points.read_text())
    kept["points"] = [p for p in kept["points"] if p["level"] in levels and bool(p.get("through")) == bool(through)]
    games = arg("games")  # comma-separated 4-letter game ids; default: every game of the points file
    if games:
        want = games.split(",")
        kept["points"] = [p for p in kept["points"] if p["game"][:4] in want]
        if sorted({p["game"][:4] for p in kept["points"]}) != sorted(want):
            raise SystemExit(f"--games not all in the points file: {want}")
    points.write_text(json.dumps(kept, separators=(",", ":")) + "\n")
    study["hx4"]["points_sha256"] = hashlib.sha256(points.read_bytes()).hexdigest()
    caps, means = {}, {}
    for point in [] if through else kept["points"]:
        marks = [a["marks"] for a in reference["points"][point["game"][:4] + "-game"]["attempts"]]
        level = point["level"]
        used = [m[level - 1][0] - (m[level - 2][0] if level > 1 else 0) for m in marks if len(m) >= level]
        if used:
            means[point["id"]] = {"clears": len(used), "mean": round(sum(used) / len(used))}
            caps[point["id"]] = max(floor, int(round(2 * sum(used) / len(used), -3)))
    study["level_bench"] = level_bench.settings({**study["level_bench"], "serving": f"rl0-{profile.name}",
                                                 "action_cap": None if through else {"multiple": 5, "floor": 100},
                                                 "arms": arms, "sampling": sampling,
                                                 "level_token_cap": int(arg("level-cap")) if arg("level-cap") else None,
                                                 "token_budget_by_point": caps or None,
                                                 "rollout": {"top_logprobs": 5, "token_ids": True}})
    study["wall_seconds"], study["notebook_seconds"] = wall, notebook
    study["bootstrap_reserves"] = {"end_seconds": 60, "trial_seconds": 120}
    if probe_wall:
        wheel = fla_wheel()
        (out / "train-probe").mkdir(exist_ok=True)
        shutil.copyfile(wheel, out / "train-probe" / wheel.name)
        study["train_probe"] = train_probe.settings({**PROBE, "wheels": [f"train-probe/{wheel.name}"]})
    study["rl0"] = {
        "source": "docs/validation-20261011/build_rl0.py", "base": "v8.2 level bench (build_hx4.py --serving=v82)",
        "harness_keys_changed": [],
        "changed": {
            "model": {"before": "Qwen/Qwen3.8-Flash-Next-NVFP4 + INT3 requant expert overlay, 37.5% expert mask",
                      "after": f"{profile.model}@{profile.revision} ({profile.quant_method or 'BF16'})"},
            "chat_template": ({"file": profile.chat_template[0], "sha256": profile.chat_template[1],
                               "note": "Flash's (Qwen3.8) template in place of the checkpoint's own"}
                              if profile.chat_template else
                              {"file": "checkpoint chat_template.jinja", "kwargs": "preserve_thinking=True (earlier "
                               "reasoning kept, as the harness keeps it); no effort line"}),
            "sampler": {"before": {"temperature": study["temperature"]}, "after": sampling,
                        "source": "model card (thinking: general tasks; non-thinking)"},
            "moe_backend": " ".join(profile.extra_args) or "vLLM default (DeepGEMM)",
            "level_token_cap": int(arg("level-cap")) if arg("level-cap") else None,
            "server": {"before": {k: before["server"].get(k) for k in ("mtp_tokens", "max_cudagraph_capture_size")},
                       "after": {k: study["server"][k] for k in ("mtp_tokens", "max_cudagraph_capture_size")},
                       "removed": sorted(k for k in before["server"] if k not in study["server"])},
            "prefix_cache": {"before": sorted(k for k, v in cache["prefix_cache"].items() if v),
                             "after": list(prefix_cache)},
            "server_env": {"before": before["trials"][0].get("server_env"), "after": SERVER_ENV},
            "level_bench.action_cap": {"before": None, "after": study["level_bench"]["action_cap"]},
            "level_bench.arms": arms,
            "level_bench.rollout": study["level_bench"]["rollout"],
            "level_bench.token_budget_by_point": {
                "rule": f"max({floor}, 2 x Flash's mean generated tokens to clear the level), rounded to 1,000",
                "reference": {"file": reference_path.name, "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                              "name": reference.get("name"), "budget": reference.get("budget"), "runs": reference.get("runs")},
                "flash": means, "caps": caps, "tokens_per_pass": sum(caps.values())},
            "levels": levels,
            "play_through": {"arm": through, "points": arg("through-points", THROUGH),
                             "games": sorted({p["game"][:4] for p in kept["points"]}), "token_budget": int(arg("budget", "54000"))} if through else None,
            "streams": {"window": window, "served_max_model_len": served, "streams": concurrency,
                        "kv_tokens": int(arg("kv-tokens")) if arg("kv-tokens") else None,
                        "rule": "floor(margin x KV pool tokens / window)" if arg("kv-tokens") else "fixed"},
            "dropped_study_keys": [k for k in DROPPED if k in before],
            "dropped_files": ["draft-head/", "configs/expert-prune/"],
        },
        "unchanged": {k: study[k] for k in ("windows", "completion_cap", "temperature", "kv_margin", "active_limit",
                                            "generation_slots", "reasoning_history", "loop_rule")},
    }
    harness = ("harness_fixes", "drain_half", "hx_harness", "dfr_harness", "dfr_loop", "loop_rule", "priority",
               "windows", "completion_cap", "temperature", "reasoning_history", "compaction_thinking", "token_budget",
               "context_policy", "v82", "reasoning_effort")
    study["rl0"]["harness_keys_changed"] = [k for k in harness if study.get(k) != before.get(k)]
    # the model's own template has no effort dial: reasoning_effort follows the profile ('native')
    allowed = {"reasoning_effort"} if profile.effort == "native" else set()
    # the candidate path's own preconditions, before any GPU time
    candidate_profiles.from_study(study)
    validate_prefix_cache(study["cache_experiment"]["prefix_cache"])
    for trial in study["trials"]:
        extra = set(lab_overrides(trial["server_env"])) - CANDIDATE_LAB
        if extra:
            raise SystemExit(f"server_env keys a candidate server refuses: {sorted(extra)}")
    if (set(study["rl0"]["harness_keys_changed"]) - allowed or study["server"]["max_model_len"] != profile.native_context
            or study["level_bench"]["concurrency"] > study["active_limit"]
            or notebook - wall - probe_wall + 60 < 1100 or notebook > 23400):
        raise SystemExit(f"study check failed: {study['rl0']['harness_keys_changed']}")
    for folder in ("draft-head", "configs/expert-prune", "dataset-upload"):
        shutil.rmtree(out / folder, ignore_errors=True)
    study["name"] = out.name
    path.write_text(json.dumps(study, indent=2) + "\n")
    identity = json.loads((out / "MANIFEST.json").read_text())
    identity["files"] = manifest(out)
    (out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    runpy.run_path(str(out / "bootstrap.py"))["verify"](out)
    print(json.dumps({"output": str(out), "manifest_sha256": hashlib.sha256((out / "MANIFEST.json").read_bytes()).hexdigest(),
                      "model": profile.model, "datasets": study["model_datasets"], "server": study["server"],
                      "trials": [(t["name"], t.get("wall_seconds", wall)) for t in study["trials"]],
                      "attempts": {a: sum(1 for w in level_bench.plan(
                          level_bench.load_points(out / study["level_bench"]["start_points"], study["level_bench"]),
                          study["level_bench"]["passes"], 0, arms) if w[2]["name"] == a)
                          for a in ([x["name"] for x in arms] if arms else ["base"])},
                      "notebook_seconds": notebook, "level_bench": {k: study["level_bench"][k] for k in (
                          "token_budget", "passes", "concurrency", "window", "action_cap")},
                      "caps": caps, "tokens_per_pass": sum(caps.values()),
                      "points_sha256": study["hx4"]["points_sha256"]}))
