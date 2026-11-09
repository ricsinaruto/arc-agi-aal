"""Build the arena bundle: source, configs, locked wheels; the Kaggle notebook, kernel metadata and
dataset upload. No GPU job is submitted.

``scripts/arena/bundle_bootstrap.py`` (bundled as ``bootstrap.py``) holds the verify/install/evidence helpers and
``scripts/arena/bootstrap.py`` (bundled as ``bootstrap_arena.py``) is the in-notebook entry point.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from arena.compaction import COMPACTION_THINKING, DEFAULT_COMPACTION_THINKING
from arena.scheduler import token_budget_settings
from arena.variants import VARIANTS, apply_variant

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("arena", "arc_infra", "inference", "taaf", "duck_transfer")
# Token budgets ("It should definitely be token based to make it comparable with
# different number of running streams. Make this change and it should apply to all new runs."): every new build's
# level budgets count completion tokens instead of busy / inference seconds (arena.clock, arena.scheduler;
# docs/validation-20260927/token-budgets.md). ``--budget-clock time`` keeps the clock the configuration and variant
# name, which is how every bundle before 29 September was built.
BUDGET_CLOCK_CHOICES = ("tokens", "time")
TOKEN_BUDGET = token_budget_settings()  # unit, purposes, floor_tokens, reference rate (shared with arena.hero_v2)
TOKEN_CLOCK_TEXT = ("completion tokens of the game's gameplay replies, reasoning included (compaction summaries and "
                    "notes/memory asides, tool execution, queue, tokenization and latency excluded); the 30 s floor "
                    "is 1,920 tokens (30 s at 64 tokens per busy second, one focus game at six streams)")
# Studies whose studied factor is a seconds knob stay on their recorded clock.
SECONDS_KNOBS = ("max_level_grant_seconds", "t1_cap_seconds", "pool_time_limit_seconds")
# Compaction summary mode (reuse the gameplay prompt cache without reasoning, the default
# for all new runs): the Duck agents' summary continues a pre-filled empty think block after the gameplay prefix
# (arena.compaction). ``--compaction-thinking gameplay`` (thinking at the gameplay effort, 28 September) or ``off``
# (tool-less, thinking-off) rebuilds an older study.
COMPACTION_KINDS = ("duck-transfer", "duck-public", "level-bench")
DEFAULT_COMPACTION_MODE = "prefilled-empty"


def with_compaction_thinking(study, requested=None):
    """``study`` with its compaction summary mode (arena.compaction.COMPACTION_THINKING) for the Duck agents.

    New builds default to "prefilled-empty"; a component that already chose a mode
    (hero-v2 ``compaction_thinking``) keeps it, and ``requested`` (``--compaction-thinking``) may not contradict it.
    "gameplay", the mode of every study built before the key, is written as no key, so such a rebuild's
    configuration is byte-identical to one built before 29 September."""
    if not {t.get("kind") for t in study.get("trials", [])} & set(COMPACTION_KINDS):
        if requested not in (None, DEFAULT_COMPACTION_MODE):
            raise ValueError("--compaction-thinking applies to the Duck agents; this study has no Duck trial")
        return study
    chosen = study.get("compaction_thinking")
    if chosen is not None and requested is not None and requested != chosen:
        raise ValueError(f"This study's components chose compaction_thinking {chosen!r}; --compaction-thinking "
                         f"{requested!r} contradicts it (change the component instead)")
    mode = requested or chosen or DEFAULT_COMPACTION_MODE
    if mode not in COMPACTION_THINKING:
        raise ValueError(f"compaction_thinking must be one of {COMPACTION_THINKING}")
    if mode == DEFAULT_COMPACTION_THINKING:
        study.pop("compaction_thinking", None)
    else:
        study["compaction_thinking"] = mode
    return study


# Reasoning history ("yes to the round-trip fix", prefix-cache-v2.md): the Duck agents
# re-send the server's reasoning and content strings verbatim (only <think> tags removed), the default for new builds;
# ``--reasoning-history normalized`` (the public loop: blank lines removed) rebuilds an older study.
DEFAULT_REASONING_HISTORY = "verbatim"


def with_reasoning_history(study, requested=None):
    """``study`` with its Duck reasoning-history mode (arc_infra.config.REASONING_HISTORY_MODES).

    New builds default to "verbatim"; a component that already chose a mode (hero-v2 ``reasoning_history``) keeps it,
    and ``requested`` (``--reasoning-history``) may not contradict it. "normalized", the behaviour of every study built
    before the key, is written as no key, so such a rebuild's configuration is byte-identical to an older one."""
    from arc_infra.config import REASONING_HISTORY_MODES

    if not {t.get("kind") for t in study.get("trials", [])} & set(COMPACTION_KINDS):
        if requested not in (None, DEFAULT_REASONING_HISTORY):
            raise ValueError("--reasoning-history applies to the Duck agents; this study has no Duck trial")
        return study
    chosen = study.get("reasoning_history")
    if chosen is not None and requested is not None and requested != chosen:
        raise ValueError(f"This study's components chose reasoning_history {chosen!r}; --reasoning-history "
                         f"{requested!r} contradicts it (change the component instead)")
    mode = requested or chosen or DEFAULT_REASONING_HISTORY
    if mode not in REASONING_HISTORY_MODES:
        raise ValueError(f"reasoning_history must be one of {REASONING_HISTORY_MODES}")
    if mode == "normalized":
        study.pop("reasoning_history", None)
    else:
        study["reasoning_history"] = mode
    return study


def token_clock_exclusion(study):
    """Why a study keeps its time clock under ``--budget-clock tokens`` (None: it takes the token clock)."""
    if study.get("budget_clock") == "normalized":
        return "the variant's own normalized clock (equivalent seconds charged from tokens) is its studied factor"
    if study.get("hero_v2") and not study["hero_v2"]["requested"].get("token_budgets", True):
        return "hero-v2 was built with its token_budgets component off"
    knobs = [k for k in SECONDS_KNOBS if study.get(k) is not None]
    if knobs:
        return f"the variant's studied factor is defined in seconds ({', '.join(knobs)})"
    return None
THROUGHPUT_GRID = {"prompt_tokens": [8192, 32768, 65536], "concurrency": [1, 2, 4, 8], "output_tokens": 512,
                   "shared_prefix": {"prompt_tokens": 32768, "concurrency": 8, "tail_tokens": 256}}
TRANSCRIPT_GRID = {"prompt_tokens": [], "concurrency": [], "output_tokens": 512, "shared_prefix": None,
                   "transcript": {"prompt_tokens": [8192, 32768, 65536, 98304], "concurrency": [1, 2, 3]}}
# Serving A/B bench (26 September, throughput survey): behaviour-preserving server knobs of the pinned
# serving script, each benched against the control in the same container (control, candidates, repeated).
SERVING_AB = {
    "control": {},
    "index-share": {"TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION": "1"},  # MTP draft steps reuse step 0's sparse top-k
    # flashinfer_b12x dropped (bench v2, 26 September): the Flash checkpoint has unquantized MoE layers and
    # vLLM rejects the backend for them.
    "kv-11gib": {"ARENA_KV_CACHE_MEMORY_BYTES": str(11 * 1024**3)},  # explicit KV bytes above the profiled 9.1 GiB
}
# Serving lab v1 (27 September): why decode plateaus near 320 tok/s. One server start per arm in one
# container, control first, tail dropped by the notebook budget. Each arm carries its own grid; the
# control also torch-profiles a short decode at 8 and 1 streams. Findings in the pinned vLLM source:
# the QSA attention backend rejects an FP8 KV cache (arm kept as a recorded startup check), and the
# FLASHINFER_TRTLLM / CUTEDSL NvFp4 MoE kernels are SM100-family only, so the MoE arm forces the only
# other SM120 path, Marlin (W4A16), through vLLM's VLLM_TEST_FORCE_FP8_MARLIN switch.
LAB_FULL_GRID = {"prompt_tokens": [4096, 16384], "concurrency": [1, 2, 4, 8, 16, 24], "output_tokens": 512,
                 "shared_prefix": None, "transcript": None, "points": [[65536, 1], [65536, 2]]}
LAB_HIGH_GRID = {"prompt_tokens": [4096, 16384], "concurrency": [1, 8, 16, 24], "output_tokens": 512,
                 "shared_prefix": None, "transcript": None, "points": []}
LAB_PROFILE = {"prompt_tokens": 2048, "concurrency": [8, 1], "output_tokens": 96}
SERVING_LAB = (
    ("control", {"ARENA_TORCH_PROFILER": "1"}, {**LAB_FULL_GRID, "profile": LAB_PROFILE}, 900),
    ("kv-fp8", {"ARENA_KV_CACHE_DTYPE": "fp8"}, LAB_FULL_GRID, 900),
    ("mtp0", {"ARENA_MTP_TOKENS": "0"}, LAB_HIGH_GRID, 600),
    ("mtp1", {"ARENA_MTP_TOKENS": "1"}, LAB_HIGH_GRID, 600),
    ("moe-marlin", {"VLLM_TEST_FORCE_FP8_MARLIN": "1"}, LAB_HIGH_GRID, 600),
    ("graph-128", {"ARENA_MAX_CUDAGRAPH_CAPTURE_SIZE": "128"}, LAB_FULL_GRID, 900),
    # Diagnostic only (the arena needs prefix caching): without it the hybrid Mamba state costs one
    # block per group instead of 2 + draft tokens, the upper bound on concurrent streams.
    ("mtp0-noprefix", {"ARENA_MTP_TOKENS": "0", "ARENA_PREFIX_CACHING": "0"}, LAB_HIGH_GRID, 600),
    ("control-repeat", {}, LAB_FULL_GRID, 900),
)
# Kernel work, phase 4: the server smoke of the MTP-3 admission arm. Decode at 1 and 6 streams on 16K prompts
# and on full 64K windows (64,512 prompt + 512 completion + 256 overhead), a replay of a recorded transcript
# for arena-like MTP acceptance, and one greedy sanity answer.
KERNEL_SMOKE_VARIANTS = ("fixed-2x-mtp3-wide-kernels", "fixed-2x-mtp3-wide-tweaks", "fixed-2x-mtp3-wide-stack")
KERNEL_SMOKE_GRID = {"prompt_tokens": [], "concurrency": [], "output_tokens": 512, "shared_prefix": None,
                     "points": [[16384, 1], [16384, 6], [64512, 1], [64512, 6]],
                     "transcript_points": {"prompt_tokens": [16384], "concurrency": [1, 6]},
                     "sanity": [{"prompt": "What is 17 * 23? Reply with the number only.", "expect": "391",
                                 "max_tokens": 32},
                                {"prompt": "Name the capital of France in one word.", "expect": "Paris",
                                 "max_tokens": 16}],
                     # Compaction shares the gameplay prefix via tool_choice "none" (main a1e350d): check that the
                     # served template still renders the tool list under it.
                     "tool_choice_probe": True}
# Kernel work, phase 5: the stack smoke, two server starts (<=). (1) The phase-4 MTP-3 arm plus the stack, with the
# host-copy item switchable at run time (stack_kernels.async_h2d "toggle"): every point of the phase-4 smoke's grid
# (replayed 16K transcript, 16K and 64K synthetic, at 1 and 6 streams) runs first with blocking copies (the pinned
# behaviour) and then asynchronous ones, with the same prompts, then a torch-profiled 16K decode at 1 and 6 streams.
# (2) The MTP-0 kv8 arm (fixed-2x-mtp0-wide-kernels-kv8, the main candidate) plus the stack, restarted in place, at 1,
# 6 and 14 streams on 16K and 64K prompts with a profiled 14-stream decode. The mid-M tiles are measured per shape by
# the load-time tuner in both servers (ARC_DENSE_MID), against the pinned kernels on the same weights.
STACK_SMOKE_VARIANTS = ("fixed-2x-mtp3-wide-stack",)
STACK_SMOKE_GRID = {"prompt_tokens": [], "concurrency": [], "output_tokens": 512, "shared_prefix": None,
                    "points": [[16384, 1], [16384, 6], [64512, 1], [64512, 6]],
                    "transcript_points": {"prompt_tokens": [16384], "concurrency": [1, 6]},
                    "sanity": KERNEL_SMOKE_GRID["sanity"],
                    "h2d_toggle": {"order": ["blocking", "async"]},
                    "profile": {"prompt_tokens": 16384, "concurrency": [1, 6], "output_tokens": 96}}
STACK_MTP0_GRID = {"prompt_tokens": [], "concurrency": [], "output_tokens": 512, "shared_prefix": None,
                   "transcript": None, "h2d_toggle": None, "sanity": KERNEL_SMOKE_GRID["sanity"],
                   "points": [[16384, 1], [16384, 6], [16384, 14], [64512, 1], [64512, 6], [64512, 14]],
                   "profile": {"prompt_tokens": 16384, "concurrency": [14], "output_tokens": 64}}
# 8-bit QSA KV cache (28 September, docs/validation-20260927/kv-cache-8bit.md): one server start of the kv8 arm. The
# served layout and focus fit; greedy sanity; greedy and teacher-forced drift on the exact token ids of the recorded
# phase-2 A+C gate trial (BF16 KV, same dense kernels) plus a teacher-forced self-repeat; decode at 1 and 6 streams
# and at the new fit on full 64K windows (the phase-4 smoke's points); the QSA kernel microbenchmark (BF16 against
# int8 / fp8) in the server's runtime.
KV8_SMOKE_VARIANTS = ("fixed-2x-mtp3-wide-kernels-kv8",)
KV8_REFERENCE_NAME = "configs/kernel-quality/reference-ac-bf16.json"
KV8_REFERENCE_SOURCE = ("runs/modal-validation/arena-duck-60m-c110-kernels-v1-modal-20260928/arena/trials/kernel-gate/"
                        "kernel-quality.json")
KV8_SMOKE_GRID = {"prompt_tokens": [], "concurrency": [], "output_tokens": 512, "shared_prefix": None,
                  "transcript": None, "points": [[64512, 9], [64512, 1], [16384, 1], [64512, 6]],
                  "sanity": KERNEL_SMOKE_GRID["sanity"], "tool_choice_probe": True,
                  "quality": {"reference": KV8_REFERENCE_NAME, "repeat_teacher": True}, "kv8_microbench": True}
# Hero smoke: the hero arm (MTP-0 kv8 stack + Marlin W4A16 experts), one server start.
# Drift uses the kv8 smoke's check (the recorded phase-2 A+C gate trial, BF16 KV, same dense kernels, plus a
# teacher-forced self-repeat); decode at 1 / 6 / 14 streams on the replayed transcript and 16K / 64K synthetic prompts
# (the 16K and 64K synthetic points match the MTP-0 kv8 stack smoke's); torch profiles at 1 and 14 streams.
HERO_SMOKE_VARIANTS = ("fixed-2x-mtp0-wide-kv8-hero",)
HERO_SMOKE_GRID = {"prompt_tokens": [], "concurrency": [], "output_tokens": 512, "shared_prefix": None,
                   "transcript": {"prompt_tokens": [16384], "concurrency": [1, 6, 14]},
                   "points": [[16384, 1], [16384, 6], [16384, 14], [64512, 1], [64512, 6], [64512, 14]],
                   "sanity": KERNEL_SMOKE_GRID["sanity"],
                   "quality": {"reference": KV8_REFERENCE_NAME, "repeat_teacher": True},
                   "profile": {"prompt_tokens": 16384, "concurrency": [1, 14], "output_tokens": 64}}

# Kernel work, phase 6: the hero (control) and the phase-6 arm (hero-p6: UVA host tables and
# the fused split-K reduction) in one job, restarted in place. Decode at 1 / 6 / 14 streams on 64K synthetic prompts
# (the hero smoke's points), then mixed points shaped on the recorded phase 2 (phase2_gap.py; kernels-phase6.md): the
# streams at the arena's mean contexts (14 x 24K, 21 x 16K) decode 2,048 forced tokens while prefill requests of
# ~5K fresh tokens on a cached prefix (a new board image each) arrive at the recorded request rates (0.34 and 0.45/s).
# Drift on both servers against the recorded phase-2 A+C gate (the kv8 smoke's check, no self-repeat: the arms are
# compared with each other), torch profiles at 14 and 21 streams plus one mixed step at 14, and on the phase-6 server
# the kernel microbenchmarks (dense tiles two-launch vs fused, QSA launch profiles, W4 error on the real weights).
PHASE6_SMOKE_VARIANTS = ("fixed-2x-mtp0-wide-kv8-hero",)
PHASE6_INJECT_14 = {"prefix_tokens": 20480, "tail_tokens": 4800, "rate": 0.34}
PHASE6_INJECT_21 = {"prefix_tokens": 12288, "tail_tokens": 5500, "rate": 0.45}
PHASE6_SMOKE_GRID = {"prompt_tokens": [], "concurrency": [], "output_tokens": 512, "shared_prefix": None,
                     "transcript": None, "points": [[64512, 1], [64512, 6], [64512, 14]],
                     "mixed_points": [{"prompt_tokens": 24576, "concurrency": 14, "inject": PHASE6_INJECT_14,
                                       "output_tokens": 2048},
                                      {"prompt_tokens": 16384, "concurrency": 21, "inject": PHASE6_INJECT_21,
                                       "output_tokens": 2048}],
                     "sanity": KERNEL_SMOKE_GRID["sanity"],
                     "quality": {"reference": KV8_REFERENCE_NAME, "repeat_teacher": False},
                     "profile": {"prompt_tokens": 16384, "concurrency": [14, 21], "output_tokens": 64,
                                 "inject": {**PHASE6_INJECT_14, "concurrency": [14]}}}
PHASE6_MICROBENCH = {"parts": ["dense", "qsa", "w4"], "budget_seconds": 300, "timeout": 1200}
# Part 2 (after kernel-phase6-v1, whose two trials hit the per-trial timeout of wall + 600 s in the mixed-step profile):
# one p6 server, the mixed-step profiles at 14 and 21 streams and the microbenchmarks, with a per-trial wall that
# covers them (timeout = wall + 600 s).
PHASE6_PART2_GRID = {"prompt_tokens": [], "concurrency": [], "output_tokens": 512, "shared_prefix": None,
                     "transcript": None, "points": [], "sanity": KERNEL_SMOKE_GRID["sanity"],
                     "profile": {"prompt_tokens": 16384, "concurrency": [], "output_tokens": 64,
                                 "inject": {**PHASE6_INJECT_14, "concurrency": [14, 21]}},
                     "phase6_microbench": dict(PHASE6_MICROBENCH)}
# 900 s (v1 used 1,200 and was refused: arena.run starts a trial only with wall + 480 s left, and the Modal container's
# setup took ~665 s of the 2,340 s notebook, leaving ~1,675 < 1,680). The trial's own timeout (wall + 600 = 1,500 s)
# still covers the two profiles (~150 s) and the microbenchmark's 1,200 s.
PHASE6_PART2_WALL = 900


def phase6_trials(config_name):
    """The two arms: the study's hero serving, then hero-p6's dense and serving kernels as server overrides."""
    arm = apply_variant(json.loads((ROOT / "configs" / config_name).read_text()),
                        "fixed-2x-mtp0-wide-kv8-hero-p6")["server"]
    env = {"ARENA_TORCH_PROFILER": "1", "ARENA_DENSE_KERNELS": json.dumps(arm["dense_kernels"]),
           "ARENA_SERVING_KERNELS": json.dumps(arm["serving_kernels"])}
    return [{"name": "hero", "kind": "serving-bench", "server_env": {"ARENA_TORCH_PROFILER": "1"}},
            {"name": "p6", "kind": "serving-bench", "server_env": env,
             "throughput_bench": {"phase6_microbench": dict(PHASE6_MICROBENCH)}}], arm


# Prefix cache v2 smoke (29 September; docs/validation-20260927/prefix-cache-v2.md): two server starts of the hero
# serving in one container, the pinned prefix cache (control) then the three patches (arena.prefix_cache_patch), each
# with exact per-request cached tokens and a KV pool cut to PREFIX_SMOKE_POOL_BLOCKS so that the replay's 24 games do
# not all fit (opening-like eviction pressure). Per arm: greedy sanity, the warm/cold correctness check on cuts of a
# ~15K-token transcript prompt, then the 24-game x 6-request replay at 8 concurrent requests with live replies.
PREFIX_SMOKE_VARIANTS = ("fixed-2x-mtp0-wide-kv8-hero-prefix",)
PREFIX_SMOKE_POOL_BLOCKS = 420  # of the hero's 860; HERO_KV_POOL_BYTES / 860 per block
PREFIX_SMOKE_GRID = {"prompt_tokens": [], "concurrency": [], "output_tokens": 64, "shared_prefix": None,
                     "points": [], "sanity": KERNEL_SMOKE_GRID["sanity"],
                     "prefix_check": {"prompt_chars": 48000, "cut_fractions": [0.2, 0.33, 0.66, 0.9],
                                      "output_tokens": 24},
                     "prefix_replay": {"games": 24, "requests": 6, "concurrency": 8, "max_tokens": 1200,
                                       "temperature": 0.7, "system_chars": 12000, "first_chars": 6000,
                                       "tool_chars": 1200}}


def quality_reference(output, grid, reference):
    """Bundle the kernel-quality inputs and the recorded phase-2 A+C gate trial for a bench ``quality`` check."""
    target = output / "configs/kernel-quality"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("prompts.json", "gsm8k.jsonl"):
        shutil.copyfile(ROOT / KERNEL_QUALITY_INPUTS / name, target / name)
    recorded = json.loads(Path(reference).read_text())
    if recorded.get("role") != "gate" or recorded["server"].get("dense_kernels") is None:
        raise ValueError("The drift reference is the recorded phase-2 A+C gate trial (BF16 KV)")
    trimmed = {k: recorded[k] for k in ("role", "trial", "server_env", "server", "greedy", "teacher_forced")}
    (output / grid["quality"]["reference"]).write_text(json.dumps(trimmed))
    grid["quality"]["reference_sha256"] = hashlib.sha256((output / grid["quality"]["reference"]).read_bytes()).hexdigest()
    grid["quality"]["inputs_sha256"] = hashlib.sha256((target / "prompts.json").read_bytes()).hexdigest()


CONFIGS = ("configs/arena.json", "configs/duck_public_notebook.json", "configs/duck_controls.yaml")
DATASET = "richardcsaky/arc-arena-runtime"
# Offline harness evolution rosters (25 September): 14 dev games and 9 held-out games from the
# 23-game roster without ft09 and lp85; both halves hold games with recorded clears (dev: ar25,
# tu93, vc33, wa30, tn36, lf52, r11l, cn04, bp35; held-out: sb26, re86, s5i5, su15).
EVOLVE_DEV_GAMES = ("ar25", "tu93", "vc33", "wa30", "tn36", "lf52", "cn04", "bp35", "r11l", "sp80", "m0r0", "ka59", "sc25", "sk48")
EVOLVE_HOLDOUT_GAMES = ("sb26", "re86", "s5i5", "su15", "dc22", "cd82", "g50t", "ls20", "tr87")

KAGGLE_INPUTS = [
    DATASET,
    "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1",
    "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1",
]
MODEL_SOURCE = "keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"
PIP_WHEEL = "pip-25.2-py3-none-any.whl"
PIP_SHA256 = "6d67a2b4e7f14d8b31b8b52648866fa717f45a1eb70e83002f4331d07e953717"
PIP_URL = "https://files.pythonhosted.org/packages/b7/3f/945ef7ab14dc4f9d7f40288d2df998d1837ee0888ec3659c813487572faa/"


def manifest(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.json" and p.suffix != ".ipynb"
        and p.name not in ("kernel-metadata.json", "build-manifest.json") and "dataset-upload" not in p.parts
    }


def download_dependencies(output):
    """Locked Linux x86_64 CPU wheels and the pinned bootstrap pip."""
    wheels = output / "wheels"
    wheels.mkdir(exist_ok=True)
    pip_wheel = wheels / PIP_WHEEL
    if not pip_wheel.exists() or hashlib.sha256(pip_wheel.read_bytes()).hexdigest() != PIP_SHA256:
        urllib.request.urlretrieve(PIP_URL + PIP_WHEEL, pip_wheel)
    if hashlib.sha256(pip_wheel.read_bytes()).hexdigest() != PIP_SHA256:
        raise ValueError("Pinned bootstrap pip wheel mismatch")
    subprocess.run(
        ["uv", "run", "--with", "pip", "python", "-m", "pip", "download", "--require-hashes",
         "--only-binary=:all:", "--python-version", "312", "--platform", "manylinux_2_28_x86_64",
         "--platform", "manylinux2014_x86_64", "--implementation", "cp", "--abi", "cp312",
         "-r", str(output / "requirements.txt"), "-d", str(wheels)],
        cwd=ROOT, check=True,
    )


# Draft-head fine-tune (docs/validation-20260927/draft-head-finetune.md): one server start
# of the pinned stack (MTP-3, eager, hidden-state dump hook), dump of the bundled sequences, then the trainer
# on the same GPU. Timing (pilot): setup ~12 min, dumps <= DRAFT_HEAD_DUMP_SECONDS, training until the trial
# deadline minus the final eval/export reserve.
DRAFT_HEAD_WALL_SECONDS = 1560        # trial process: killed by the bootstrap at wall + 600 s
DRAFT_HEAD_NOTEBOOK_SECONDS = 3140    # the Modal test's timeout (3,200 s) minus 60
DRAFT_HEAD_DUMP_SECONDS = 480
DRAFT_HEAD_TRAINER = {"train_steps": 300, "lr": 5e-5, "warmup": 20, "max_anchors": 6144, "eval_every": 100,
                      "eval_subset": 8, "steps": 3, "step_weights": [1.0, 0.8, 0.6], "reserve_seconds": 420}


def draft_head_study(study, output, data_dir):
    """The serving-bench profile turned into the draft-head job: data copied into the bundle."""
    data_dir = Path(data_dir)
    target = output / "draft-head/data"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("train.npz", "eval.npz", "manifest.json"):
        shutil.copyfile(data_dir / name, target / name)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    from posttrain.split import HOLDOUT
    leaked = set(manifest["splits"]["train"]["games"]) & set(HOLDOUT)
    if leaked:
        raise ValueError(f"holdout games in the draft-head training split: {sorted(leaked)}")
    study = json.loads(json.dumps(study))
    study["trials"] = [{"name": "draft-head", "kind": "draft-head"}]
    study.pop("throughput_bench", None)
    study["server"]["draft_head"] = {"dump": True, "overlay": None}
    study["draft_head"] = {"data": "draft-head/data", "dump_seconds": DRAFT_HEAD_DUMP_SECONDS,
                           "verify_sequences": 2, "min_free_gib": 12, "trainer": dict(DRAFT_HEAD_TRAINER),
                           "splits": {k: {kk: v[kk] for kk in ("sequences", "tokens", "generated_tokens", "calls",
                                                                  "games")}
                                      for k, v in manifest["splits"].items()}}
    study.update(wall_seconds=DRAFT_HEAD_WALL_SECONDS, notebook_seconds=DRAFT_HEAD_NOTEBOOK_SECONDS,
                 kernel_timeout=DRAFT_HEAD_NOTEBOOK_SECONDS + 60,
                 harness_selection="draft-head fine-tune (no agent)")
    return study


def draft_head_overlay_study(study, output, overlay_dir):
    """Serve a fine-tuned draft head (``arena.draft_head_train`` export): the overlay is copied into the bundle
    and applied at load by ``arc_infra.draft_head_patch`` (MTP tensors only; any serving arm)."""
    overlay_dir = Path(overlay_dir)
    target = output / "draft-head/overlay"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("overlay.safetensors", "overlay-manifest.json"):
        shutil.copyfile(overlay_dir / name, target / name)
    manifest = (target / "overlay-manifest.json").read_bytes()
    study = json.loads(json.dumps(study))
    study["server"]["draft_head"] = {"dump": False, "overlay": {
        "path": "draft-head/overlay", "sha256": hashlib.sha256(manifest).hexdigest()}}
    return study


def build(name, *, smoke=False, output=None, dependencies=False, cache_variant=None, paired_30m=False, sglang=False, candidate_profile=None, arena_variant=None, capacity_phase2=False, competition=False, prioritize_parked_progress=False, graph_metrics=False, opening_burst_requests=1, throughput_bench=False, rerun_rehearsal=False, bench_transcript=None, exclude_games=(), evolve=False, wall_seconds=None, serving_ab=False, probe_games=(), generated_games=None, evolve_rounds=None, preview_seconds=3600, preview_instances=110, clone_instances=None, sanity_trial=True, rehearsal_seconds=6000, no_preview=False, weights_overlay=None, serving_lab=False, llamacpp=None, llamacpp_slots=None,
          llamacpp_kv_unified=False, llamacpp_serve_check=None, level_bench=None, kernel_gate=False,
          kernel_smoke=False, draft_head=None, draft_head_overlay=None, kv8_smoke=False,
          kv8_reference=None, expert_quant=None, expert_quant_phase=None, expert_quant_candidates=None,
          expert_quant_settings=None, expert_quant_prune=None,
          hero_smoke=False, expert_usage=None, expert_usage_measure_only=False, expert_prune_check=None,
          expert_prune_levels=False,
          expert_prune_hang_test=False, stall_dump=None, gpu_memory_utilization=None, hero_v2=None, hero_v2_served_blocks=None,
          hero_v2_smoke=False, hero_v2_reference=None, budget_clock="tokens", compaction_thinking=None,
          prefix_cache_smoke=False, reasoning_history=None, phase6_smoke=False,
          phase6_part2=False, drift_long=None, kv4_drift=None, hero_v2_window_blocks=None,
          hero_v2_drift_inputs=None, hero_v2_transcript=None, hero_v2_precheck=None, prefix_unit_int4=None,
          prefix_unit_int4_tf_only=False, kv8_prefill_check=None, early_stop=None, hero_v2_soak=False,
          decode_sweep=None,
          prefix_cache_v3_smoke=None, prefix_cache_v3_recheck=False, campaign_arm=None,
          preview_live_score=False):
    if draft_head is not None and (not throughput_bench or kernel_smoke or serving_ab or serving_lab
                                   or bench_transcript is not None or arena_variant is not None):
        raise ValueError('The draft-head job builds on the plain serving-bench profile (--paired-30m '
                         '--throughput-bench, no variant)')
    if clone_instances is not None and (competition or not paired_30m or smoke or evolve or probe_games or generated_games
                                        or level_bench is not None or throughput_bench
                                        or not 1 <= int(clone_instances) <= 110):
        raise ValueError('--clone-instances is a 1-110 instance research roster on the Duck arena (not competition)')
    if rerun_rehearsal and not competition:
        raise ValueError('The hidden-rerun rehearsal is a competition build')
    if no_preview and not competition:
        raise ValueError('no_preview is a competition build option')
    if preview_live_score and not competition:
        raise ValueError('preview_live_score is a competition build option')
    if level_bench is not None and (not paired_30m or smoke or competition or throughput_bench or evolve or probe_games
                                    or generated_games or cache_variant
                                    or wall_seconds is not None or exclude_games):
        raise ValueError('The level bench builds on the 30-minute Duck arena profile (e.g. --arena-variant '
                         'reclaim-fixed-2x) with no roster, wall or other trial option')
    if type(opening_burst_requests) is not int or opening_burst_requests not in (1, 3):
        raise ValueError('Opening comparison supports FIFO or three-request bursts')
    if throughput_bench and (not paired_30m or smoke or sglang or competition
                             or candidate_profile or cache_variant or capacity_phase2
                             or prioritize_parked_progress or opening_burst_requests != 1
                             or arena_variant not in (None, 'cache-reclaim', 'reclaim-mtp1', 'reclaim-mtp4',
                                                      'reclaim-ngram', 'reclaim-ngram8', 'reclaim-graph-128',
                                                      *KERNEL_SMOKE_VARIANTS, *KV8_SMOKE_VARIANTS, *HERO_SMOKE_VARIANTS,
                                                      *PREFIX_SMOKE_VARIANTS)):
        raise ValueError('The serving benchmark runs on the Flash vLLM arena profile (optionally cache-reclaim) alone')
    if bool(kernel_smoke) != (throughput_bench and arena_variant in KERNEL_SMOKE_VARIANTS):
        raise ValueError('The phase-4 kernel smoke is the serving benchmark on the phase-4 serving arm, and only it')
    if bool(kv8_smoke) != (throughput_bench and arena_variant in KV8_SMOKE_VARIANTS):
        raise ValueError('The 8-bit KV smoke is the serving benchmark on the kv8 arm, and only it')
    if kv8_smoke and (bench_transcript is not None or serving_ab or serving_lab or not kv8_reference):
        raise ValueError('The 8-bit KV smoke takes the recorded BF16 reference (--kv8-reference) and no other bench option')
    if bool(hero_smoke or phase6_smoke) != (throughput_bench and arena_variant in HERO_SMOKE_VARIANTS):
        raise ValueError('The hero and phase-6 smokes are the serving benchmark on the hero arm, and only it')
    if phase6_smoke and (hero_smoke or bench_transcript is not None or serving_ab or serving_lab
                         or not (kv8_reference or phase6_part2) or kernel_smoke or kv8_smoke):
        raise ValueError('The phase-6 smoke takes the recorded reference (--kv8-reference) and no other bench option')
    if phase6_part2 and not phase6_smoke:
        raise ValueError('--phase6-part2 extends --phase6-smoke')
    if hero_smoke and (bench_transcript is None or serving_ab or serving_lab or not kv8_reference or kernel_smoke):
        raise ValueError('The hero smoke takes --bench-transcript and the recorded reference (--kv8-reference)')
    if bool(prefix_cache_smoke) != (throughput_bench and arena_variant in PREFIX_SMOKE_VARIANTS):
        raise ValueError('The prefix-cache smoke is the serving benchmark on the hero prefix arm, and only it')
    if prefix_cache_smoke and (bench_transcript is None or serving_ab or serving_lab or kv8_reference or kernel_smoke):
        raise ValueError('The prefix-cache smoke takes --bench-transcript and no other bench option')
    if opening_burst_requests != 1 and (not paired_30m or smoke or prioritize_parked_progress):
        raise ValueError('Opening affinity requires a standalone full Duck-transfer research study')
    if type(graph_metrics) is not bool or (graph_metrics and (not paired_30m or smoke or sglang or competition)):
        raise ValueError('CUDA graph metrics require a full vLLM arena research study')
    if prioritize_parked_progress and (not paired_30m or smoke or cache_variant
                                      or arena_variant not in (None, 'control', 'cache-reclaim')):
        raise ValueError('Parked-progress comparison requires a full arena control without another scheduling variant')
    if competition and (not paired_30m or smoke or sglang or candidate_profile
                        or capacity_phase2 or cache_variant
                        or arena_variant not in (None, 'control', 'cache-reclaim', 'reclaim-fixed-2x', 'fixed-2x-stall-open360', 'fixed-2x-index-lane', 'fixed-2x-worldmodel', 'fixed-2x-mtp0-wide-kernels', 'fixed-2x-mtp0-wide-kv8-hero', 'fixed-2x-mtp0-wide-kv8-hero-6x64k')):
        raise ValueError('Competition requires the fixed Flash Duck-transfer control, cache-reclaim, reclaim-fixed-2x, fixed-2x-stall-open360, fixed-2x-index-lane, fixed-2x-worldmodel or fixed-2x-mtp0-wide-kv8-hero variant')
    if capacity_phase2 and (not paired_30m or arena_variant not in (None, 'graph-128') or cache_variant or smoke):
        raise ValueError('Cache-sized phase 2 requires a full model comparison; only the graph-128 serving variant may be combined')
    if arena_variant is not None and (not paired_30m or smoke or cache_variant):
        raise ValueError('Arena variants require the independent paired 30-minute study')
    if arena_variant is not None and arena_variant not in VARIANTS:
        raise ValueError('Unknown arena variant')
    if sglang and arena_variant == 'graph-128':
        raise ValueError('CUDA graph capture comparison requires vLLM')
    if sglang and arena_variant == 'cache-reclaim':
        raise ValueError('Sparse cache reclamation comparison requires vLLM')
    # The other screened models' configs were retired on 4 October (cleanup); Ornith stays for the paused
    # post-training arm (M0 overlay evaluation).
    candidate_configs = {'ornith15-35b-a3b-nvfp4': 'arena_30m_ornith15.json'}
    if candidate_profile and (not paired_30m or sglang):
        raise ValueError('Candidates require the paired 30-minute vLLM profile')
    if candidate_profile and candidate_profile not in candidate_configs:
        raise ValueError('Unreviewed candidate profile')
    if sglang and not paired_30m:
        raise ValueError("SGLang requires the paired 30-minute profile")
    if paired_30m and (smoke or cache_variant is not None):
        raise ValueError("The paired 30-minute profile is a separate full study")
    if cache_variant in {"cpu", "fp8"}:
        from arena.compatibility import validate_settings
        validate_settings({"kv_cache_dtype": "fp8" if cache_variant == "fp8" else "auto", "cpu_cache_gib": 16})
    output = Path(output or ROOT / "dist" / name).resolve()
    if output.exists():
        for child in output.iterdir():
            if child.name != "wheels":
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    output.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for package in (*PACKAGES, *(["sglang_bench"] if sglang else [])):
        target = output / "src" / package
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(ROOT / "src" / package, target, ignore=ignore)
    (output / "configs").mkdir(exist_ok=True)
    config_name = candidate_configs[candidate_profile] if candidate_profile else "arena_30m_sglang.json" if sglang else "arena_30m.json" if paired_30m else "arena.json"
    study = json.loads((ROOT / "configs" / config_name).read_text())
    hero_v2_candidate = bool(candidate_profile) and (hero_v2 is not None or hero_v2_smoke
                                                     or hero_v2_served_blocks is not None)
    if arena_variant is not None and not hero_v2_candidate:
        study = apply_variant(study, arena_variant)
    if hero_v2_window_blocks is not None and not hero_v2_candidate:
        raise ValueError('--hero-v2-window-blocks is the measured window cost of a candidate hero-v2 build')
    if hero_v2 is not None or hero_v2_smoke or hero_v2_served_blocks is not None:
        # hero-v2: the hero plus the selected components (arena.hero_v2).
        from arena import hero_v2 as hero_v2_module

        # The roster is either the research clone roster (--clone-instances) or, for a competition submission (user, 29
        # September: "please create a competition submission with run B"), the competition gateway: the public
        # preview's cyclic instances and gateway enumeration on the hidden rerun.
        if (hero_v2 is None or arena_variant != hero_v2_module.BASE or not paired_30m or smoke
                or throughput_bench or (clone_instances is None) == (not competition)):
            raise ValueError('hero-v2 builds on the hero variant with the clone roster or the competition gateway '
                             f'(--arena-30m --arena-variant {hero_v2_module.BASE} --hero-v2 JSON, and '
                             '--clone-instances N or --competition)')
        if competition and (hero_v2_candidate or hero_v2_smoke or hero_v2_precheck is not None or hero_v2_soak
                            or early_stop is not None or prefix_unit_int4 is not None
                            or decode_sweep is not None):
            raise ValueError('A hero-v2 competition build is the arena submission only (no candidate, smoke, '
                             'pre-check, soak, early stop or bench)')
        if hero_v2_candidate:
            # hero-v2 on a candidate checkpoint (Holo4-27B, 29 September): the Flash hero study gives the arena side,
            # the candidate config the checkpoint and serving (arena.hero_v2.compose_candidate).
            if weights_overlay is not None or capacity_phase2:
                raise ValueError('hero-v2 on a candidate takes no other candidate option')
            hero = apply_variant(json.loads((ROOT / "configs/arena_30m.json").read_text()), arena_variant)
            study = hero_v2_module.compose_candidate(hero, study, hero_v2, served_blocks=hero_v2_served_blocks,
                                                     blocks_per_window=hero_v2_window_blocks)
        else:
            study = hero_v2_module.compose(study, hero_v2, served_blocks=hero_v2_served_blocks)
    if graph_metrics:
        study['server']['cudagraph_metrics'] = True
    if weights_overlay is not None:
        study['weights_overlay'] = overlay_spec(weights_overlay, candidate_profile)
        study['model_datasets'] = [*study.get('model_datasets', []), study['weights_overlay']['dataset']]
    if prioritize_parked_progress:
        study['prioritize_parked_progress'] = True
    if capacity_phase2:
        study['phase2_admission'] = 'cache_capacity'
        study['active_limit'] = 'resolved_from_cache'
    if paired_30m:
        # The paired configs' second trial (the retired Pi baseline C, "qwen-c") is always dropped.
        study['harness_selection'] = 'duck-transfer default'
        study['trials'] = [t for t in study['trials'] if t['kind'] == 'duck-transfer']
        # Remove the second 30-minute trial and its 10-minute cleanup allowance.
        # Keep the model-specific setup budget and first-trial cleanup intact.
        study['notebook_seconds'] -= 2400
        study['kernel_timeout'] -= 2400
    if cache_variant is not None:
        if cache_variant not in {"blocks", "cpu", "fp8"} or smoke:
            raise ValueError("Cache variants require a full 25-game Duck-transfer study")
        study.update(cache_admission="hybrid_blocks", generation_slots=25,
                     trials=[{"name": "duck-transfer", "kind": "duck-transfer"}],
                     notebook_seconds=2640, kernel_timeout=2700)
        study["server"]["max_num_seqs"] = 25
        # The 8192-token probe exhausted temporary PLE prefill memory despite
        # safe persistent-cache admission. Preserve context and reduce chunks.
        study["server"]["max_num_batched_tokens"] = 4096
        study["cache_experiment"] = {
            "variant": cache_variant, "cpu_cache_gib": 0 if cache_variant == "blocks" else 16,
            "kv_cache_dtype": "fp8" if cache_variant == "fp8" else "auto",
            "compaction_changed": False, "probe_seconds_limit": 270,
        }
    if bench_transcript is not None and not throughput_bench:
        raise ValueError('A replay transcript belongs to the serving benchmark')
    if throughput_bench:
        # Serving-only ceiling on the arena's exact server profile: no games are played.
        study["trials"] = [{"name": "serving-bench", "kind": "serving-bench"}]
        study["throughput_bench"] = dict(THROUGHPUT_GRID)
        if bench_transcript is not None:
            # Replay grid (25 September): a recorded Duck transcript as the prompt, so the
            # decode pass generates against real context (code and observations the model
            # re-emits) instead of synthetic word salad; 1-3 streams, up to a 96K prompt.
            text = Path(bench_transcript).read_text()
            (output / "bench").mkdir(parents=True, exist_ok=True)
            (output / "bench/transcript.txt").write_text(text)
            study["throughput_bench"] = dict(TRANSCRIPT_GRID, transcript={
                **TRANSCRIPT_GRID["transcript"], "file": "bench/transcript.txt",
                "source": str(bench_transcript), "chars": len(text),
                "sha256": hashlib.sha256(text.encode()).hexdigest()})
        study.update(wall_seconds=1500, notebook_seconds=3600, kernel_timeout=3900,
                     harness_selection="serving benchmark (no agent)")
        if kernel_smoke:
            # Kernel work, phase 4: one server start of the phase-4 arm (layout, acceptance, decode at 1 and 6
            # streams on 16K and full-window prompts, a greedy sanity answer); replay points when a transcript
            # is given. The trial carries the arm's scheduling keys only for the recorded focus-window fit.
            grid = {**{k: v for k, v in KERNEL_SMOKE_GRID.items() if k != "transcript_points"}, "transcript": None}
            if bench_transcript is not None:
                grid["transcript"] = {**study["throughput_bench"]["transcript"],
                                      **KERNEL_SMOKE_GRID["transcript_points"]}
            study["throughput_bench"] = grid
            # Setup takes ~710 s (smoke v1); the bench runs to hard_end - 420 s. wall_seconds only feeds the
            # bootstrap (budget >= wall + 300) and run.py (hard_end - now >= wall + 480) start checks.
            study.update(wall_seconds=120, notebook_seconds=1440, kernel_timeout=1480)
            if arena_variant in STACK_SMOKE_VARIANTS:
                if bench_transcript is None:
                    raise ValueError("The stack smoke replays an arena transcript (--bench-transcript)")
                from arc_infra.stack_kernels_patch import TOGGLE_FLAG

                grid = {**{k: v for k, v in STACK_SMOKE_GRID.items() if k != "transcript_points"},
                        "transcript": {**study["throughput_bench"]["transcript"],
                                       **STACK_SMOKE_GRID["transcript_points"]}}
                grid["h2d_toggle"] = {**grid["h2d_toggle"], "flag": TOGGLE_FLAG}
                study["throughput_bench"] = grid
                study["server"]["stack_kernels"] = {"async_h2d": "toggle"}
                mtp0 = apply_variant(json.loads((ROOT / "configs" / config_name).read_text()),
                                "fixed-2x-mtp0-wide-kv8-stack")["server"]
                mtp0_env = {"ARENA_TORCH_PROFILER": "1", "ARENA_MTP_TOKENS": str(mtp0["mtp_tokens"]),
                            "ARENA_DENSE_KERNELS": json.dumps(mtp0["dense_kernels"]),
                            "ARENA_SERVING_KERNELS": json.dumps(mtp0["serving_kernels"]),
                            "ARENA_STACK_KERNELS": json.dumps(mtp0["stack_kernels"])}
                study["trials"] = [{"name": "stack-mtp3", "kind": "serving-bench",
                                    "server_env": {"ARENA_TORCH_PROFILER": "1"}},
                                   {"name": "stack-mtp0-kv8", "kind": "serving-bench", "server_env": mtp0_env,
                                    "throughput_bench": json.loads(json.dumps(STACK_MTP0_GRID))}]
                study["stack_smoke"] = {"arms": ["stack-mtp3", "stack-mtp0-kv8"], "mtp0_server": mtp0,
                                        "h2d_toggle_flag": TOGGLE_FLAG}
                # Two server starts (~500-600 s each, both with the mid-M tuning) and two ~300 s benches; the MTP-0
                # trial must start with >= wall + 480 s left.
                study.update(wall_seconds=120, notebook_seconds=2340, kernel_timeout=2380)
        if hero_smoke:
            # Hero smoke: one server start of the hero arm; served layout, sanity, drift
            # against the recorded phase-2 A+C gate trial and its repeat noise (the kv8 smoke's check), decode at 1 / 6
            # / 14 streams on the replayed transcript and on 16K / 64K synthetic prompts, profiles at 1 and 14.
            grid = json.loads(json.dumps(HERO_SMOKE_GRID))
            grid["transcript"] = {**study["throughput_bench"]["transcript"], **HERO_SMOKE_GRID["transcript"]}
            study["throughput_bench"] = grid
            quality_reference(output, study["throughput_bench"], kv8_reference)
            study["trials"] = [{"name": "hero", "kind": "serving-bench", "server_env": {"ARENA_TORCH_PROFILER": "1"}}]
            # Setup ~600 s (MTP-0 start); the bench (~9 points, drift, two profiles) ~450 s; <= at 1,500 s.
            study.update(wall_seconds=120, notebook_seconds=1440, kernel_timeout=1480)
        if phase6_smoke and phase6_part2:
            study["throughput_bench"] = json.loads(json.dumps(PHASE6_PART2_GRID))
            trials, arm = phase6_trials(config_name)
            study["trials"] = [{"name": "p6b", "kind": "serving-bench", "server_env": trials[1]["server_env"],
                                "wall_seconds": PHASE6_PART2_WALL}]
            study["phase6_smoke"] = {"arms": ["p6b"], "part": 2, "p6_variant": "fixed-2x-mtp0-wide-kv8-hero-p6",
                                     "p6_server": arm}
            # Setup ~450 s (the second server's start in v1 took 425 s), the profiles ~200 s, the microbenchmarks
            # <= 1,200 s; the trial's timeout is wall + 600 s.
            study.update(wall_seconds=120, notebook_seconds=2280, kernel_timeout=2330)
        elif phase6_smoke:
            study["throughput_bench"] = json.loads(json.dumps(PHASE6_SMOKE_GRID))
            quality_reference(output, study["throughput_bench"], kv8_reference)
            study["trials"], arm = phase6_trials(config_name)
            study["phase6_smoke"] = {"arms": ["hero", "p6"], "p6_variant": "fixed-2x-mtp0-wide-kv8-hero-p6",
                                     "p6_server": arm}
            # Two server starts (~400 s each; the second pins the 51 GB PLE table) and two ~600-700 s benches, then the
            # ~15-minute microbenchmark on the second; the second trial must start with >= wall + 480 s left.
            study.update(wall_seconds=120, notebook_seconds=3150, kernel_timeout=3200)
        if prefix_cache_smoke:
            from arena.variants import HERO_PREFIX_CACHE

            grid = json.loads(json.dumps(PREFIX_SMOKE_GRID))
            # The transcript feeds the checks' prompts only: no transcript decode points.
            grid["transcript"] = {**study["throughput_bench"]["transcript"], "prompt_tokens": [], "concurrency": []}
            study["throughput_bench"] = grid
            pool = str(HERO_KV_POOL_BYTES * PREFIX_SMOKE_POOL_BLOCKS // 860)
            common = {"ARENA_PROMPT_TOKENS_DETAILS": "1", "ARENA_KV_CACHE_MEMORY_BYTES": pool}
            study["trials"] = [
                {"name": "pinned", "kind": "serving-bench", "server_env": {**common, "ARENA_PREFIX_CACHE": "off"}},
                {"name": "prefix", "kind": "serving-bench",
                 "server_env": {**common, "ARENA_PREFIX_CACHE": json.dumps(HERO_PREFIX_CACHE)}}]
            study["prefix_cache_smoke"] = {"arms": ["pinned", "prefix"], "patches": dict(HERO_PREFIX_CACHE),
                                           "pool_blocks": PREFIX_SMOKE_POOL_BLOCKS, "pool_bytes": int(pool)}
            # Two server starts (~550 s each) and two ~500 s checks; <= at the 2,700 s ledger bound.
            study.update(wall_seconds=120, notebook_seconds=2580, kernel_timeout=2640)
        if kv8_smoke:
            study["throughput_bench"] = json.loads(json.dumps(KV8_SMOKE_GRID))
            target = output / "configs/kernel-quality"
            target.mkdir(parents=True, exist_ok=True)
            for name in ("prompts.json", "gsm8k.jsonl"):
                shutil.copyfile(ROOT / KERNEL_QUALITY_INPUTS / name, target / name)
            recorded = json.loads(Path(kv8_reference).read_text())
            if recorded.get("role") != "gate" or recorded["server"].get("dense_kernels") is None:
                raise ValueError("The kv8 reference is the recorded phase-2 A+C gate trial (BF16 KV)")
            trimmed = {k: recorded[k] for k in ("role", "trial", "server_env", "server", "greedy", "teacher_forced")}
            (output / KV8_REFERENCE_NAME).write_text(json.dumps(trimmed))
            study["throughput_bench"]["quality"]["reference_sha256"] = hashlib.sha256(
                (output / KV8_REFERENCE_NAME).read_bytes()).hexdigest()
            study["throughput_bench"]["quality"]["inputs_sha256"] = hashlib.sha256(
                (target / "prompts.json").read_bytes()).hexdigest()
            # Same envelope as the phase-4 smoke (setup ~710 s; the bench runs to hard_end - 120 s).
            study.update(wall_seconds=120, notebook_seconds=1440, kernel_timeout=1480)
        if serving_ab:
            # Two rounds of control and every candidate; the server restarts in place between trials.
            study["trials"] = [{"name": f"bench-{label}-{rep}", "kind": "serving-bench", "server_env": dict(knobs)}
                               for rep in (1, 2) for label, knobs in SERVING_AB.items()]
            study["serving_ab"] = {"candidates": {k: dict(v) for k, v in SERVING_AB.items()}, "rounds": 2}
            study.update(notebook_seconds=6600, kernel_timeout=6900)
        if serving_lab:
            if serving_ab or bench_transcript is not None or arena_variant != "cache-reclaim":
                raise ValueError("The serving lab runs its own arms on the cache-reclaim profile")
            study["trials"] = [{"name": f"lab-{label}", "kind": "serving-bench", "server_env": dict(env),
                                "throughput_bench": dict(grid), "wall_seconds": wall}
                               for label, env, grid, wall in SERVING_LAB]
            study["serving_ab"] = {"candidates": {label: dict(env) for label, env, _, _ in SERVING_LAB}, "rounds": 1,
                                   "lab": "serving-lab-v1"}
            study.update(notebook_seconds=6600, kernel_timeout=6900)
    elif serving_ab or serving_lab:
        raise ValueError("--serving-ab needs --throughput-bench")
    study["opening_burst_requests"] = opening_burst_requests
    # A variant may select the busy clock and the fixed-2x policy; everything else is the
    # inference-clock control. Composed after the variant so the recorded text matches the
    # scheduler's actual knobs (the regrant-2x v1 study text still said 1.5x while it ran with 2x).
    # A component may already have put the study on the token clock (hero-v2 ``token_budgets``); that is an explicit
    # request, so --budget-clock time refuses it instead of silently reverting.
    requested_tokens = study.get("budget_clock") == "tokens"
    if requested_tokens and budget_clock != "tokens":
        raise ValueError("This study asks for token budgets (e.g. hero-v2 token_budgets); --budget-clock time cannot "
                         "rebuild it on a time clock (turn the component off instead)")
    study["budget_clock"] = (study.get("budget_clock")
                             if study.get("budget_clock") in ("inference", "busy", "normalized") else "inference")
    if (study["budget_clock"] == "normalized") != bool(study.get("cost_normalization")):
        raise ValueError("The normalized clock and cost_normalization go together")
    clock_text = {"busy": "admitted model-request seconds plus tool execution; queue and tokenization excluded",
                  "normalized": "equivalent seconds charged from completion tokens (context-weighted) plus tool "
                                "execution; latency, queue and tokenization excluded",
                  }.get(study["budget_clock"], "admitted model-request seconds; queue, tokenization and tools excluded")
    if study["budget_clock"] == "normalized" and not (study["cost_normalization"].get("context_weight") or 0):
        clock_text = clock_text.replace(" (context-weighted)", "")
    if budget_clock not in BUDGET_CLOCK_CHOICES:
        raise ValueError(f"budget_clock must be one of {BUDGET_CLOCK_CHOICES}")
    if budget_clock == "tokens":
        excluded = token_clock_exclusion(study)
        if excluded is None:
            study["budget_clock"] = "tokens"
            study["token_budget"] = {**TOKEN_BUDGET, "purposes": list(TOKEN_BUDGET["purposes"])}
            clock_text = TOKEN_CLOCK_TEXT
        elif requested_tokens:
            raise ValueError(f"This study asks for token budgets but cannot run them: {excluded}")
        else:
            study["budget_clock_note"] = f"time clock kept: {excluded}"
    floor = "1,920 tokens" if study["budget_clock"] == "tokens" else "30 s"
    plus = "1920" if study["budget_clock"] == "tokens" else "30"
    if study.get("level_budget_policy") == "index" and study["index_scheduler"].get("model") == "gittins":
        cfg = study["index_scheduler"]
        rule = ("no level budgets after the opening: slots go to the best expected score per token over plans 'play up "
                "to j more levels, giving up on a level after c tokens' (c in "
                f"{', '.join(f'{c / 1000:g}K' for c in cfg['caps'])}) under a hierarchical hazard model: "
                f"{'within-level' if cfg.get('shape') == 'within' else 'pooled'} "
                f"token-band hazard x a per-game factor Gamma({cfg['k']:g}, {cfg['k']:g}"
                f"{' / s, s estimated in-run from every game' if cfg.get('prior_scale') == 'run' else ''}) learned "
                f"from all the game's levels in this run (power {cfg['beta']:g} on deeper levels) x a per-level factor "
                f"Gamma({cfg['ke']:g}, {cfg['ke']:g}); quantum {cfg['quantum_tokens']} tokens, hysteresis "
                f"{cfg['hysteresis']:g}")
        if study.get("opening_min_tokens"):
            rule += (f"; the opening also waits until every game has {study['opening_min_tokens']} completion tokens "
                     f"(cap {study.get('opening_max_wall_seconds')} s)")
        elif study.get("opening_max_wall_seconds"):
            rule += (f"; the opening ends at the second level-1 clear or at {study['opening_max_wall_seconds']} s"
                     f"{' (absolute, not rescaled by the clone roster)' if study.get('opening_cap_absolute') else ''}")
        study["backlog_priority"] = "Gittins-style index (arena.gittins)"
    elif study.get("level_budget_policy") == "index":
        cfg = study["index_scheduler"]
        rule = ("no level budgets after the opening: slots go to the highest expected score per token over the next "
                "s tokens, s = the run's median tokens per level clear; clear rate = a per-class level fitted online "
                f"from this run x a relative shape ({cfg['shape']}); level weight plus later levels discounted by the "
                f"fitted deep-clear chance; quantum {cfg['quantum_scale']:g}s, hysteresis {cfg['hysteresis']:g}")
        study["backlog_priority"] = "hazard index (arena.hazard_index)"
        if study.get("background_lane"):
            lane = study["background_lane"]
            rule += (f"; background lane: games outside the focus slots request on leftover cache, at most "
                     f"{lane['max_inflight']} at once, fewest completion tokens on the level first, never into a "
                     f"focus game's guard (latest prompt + {lane['guard_caps']:g} x cap, clipped to the window)")
    elif study.get("level_budget_policy") == "commit":
        cfg = study["commit"]
        rule = (f"no level budgets after the opening: {cfg['focus_slots']} focus slots go to games that clear a level "
                f"(a lane clear is promoted at once, replacing the focus game past its stall cap, else the one with the "
                f"most tokens on its level); a focus game keeps its slot while its tokens on the level stay within "
                f"clamp({cfg['stall_multiple']:g} x its median tokens per cleared level, {cfg['stall_min_tokens']}, "
                f"{cfg['stall_max_tokens']}), then moves to the lane until it clears; free slots go to the lane game with "
                f"the most levels, then fewest tokens on its level; lane: at most {cfg['lane_max_inflight']} requests, "
                f"level-1 games under {cfg['level1_floor_tokens']} tokens first, then deep games, fewest tokens on the "
                f"level first, admitted while estimated cache use + prompt + p{cfg['lane_headroom_quantile'] * 100:g} "
                f"completion <= {cfg['lane_kv_fraction']:g} x pool; vLLM priority scheduling (focus 0, lane 1)")
        study["backlog_priority"] = "momentum commitment (arena.commit)"
    elif study.get("level_budget_policy") == "deep-explore":
        cfg = study["deep_explore"]
        rule = (f"no opening and no level budgets: {cfg['explore_slots']} exploration slots ({cfg['explore_window']:,}-token "
                f"window) rotate the games on a {cfg['explore_quantum_tokens']:,}-completion-token quantum (never played "
                f"first, then fewest total tokens; {cfg['explore_max_tokens']:,} tokens on level 1 without a clear goes "
                f"last); the first clear takes a free deep slot ({cfg['deep_slots']}, {cfg['deep_window']:,}-token window) "
                f"and plays level after level while its tokens on the current level stay within "
                f"max({cfg['deep_min_tokens']:,}, {cfg['deep_multiple']:g} x the median tokens of its cleared levels); past "
                f"that it is demoted to the rotation (compacted to the exploration window) only when a promotion "
                f"candidate waits; an exploration clear keeps its slot to the end of its quantum, then queues for the "
                f"deep slot (most recent clear first, then fewest tokens on its cleared levels)")
        study["backlog_priority"] = "deep/explore rotation (arena.deep_explore)"
    elif study.get("level_budget_policy") == "depthwise":
        cfg = study["depthwise"]
        unit = "completion tokens" if study["budget_clock"] == "tokens" else "seconds"
        regrant = (f"the rest of the level allowance if any is left, else max(1.5*spent, spent+{plus}) with spent = "
                   f"{unit} on the level" if cfg["regrant"] == "spent" else "the full level allowance again")
        rule = (f"no opening: {cfg['slots']} slots from the release; level-0 games rotate through any free slot at a "
                f"{cfg['level0_window']:,}-token window on a {cfg['level0_quantum_tokens']:,}-completion-token turn "
                f"(never played first in roster order, then fewest completion tokens); a game that clears level 1 keeps "
                f"its slot at the {cfg['deep_window']:,}-token window with fixed-2x budgets: 2*(k+1)*T1 for level "
                f"k -> k+1, granted in full at every clear; an exhausted depth game parks only while another game "
                f"waits (the level-0 rotation counts); re-admission grant: {regrant}; " + clock_text)
        study["backlog_priority"] = ("parked clears first, then parked depth games by fastest level-1 clear (smallest "
                                     "T1), then level-0 games never played (roster order), then fewest completion "
                                     "tokens (arena.depthwise)")
    elif study.get("level_budget_policy") == "blocks":
        from arena.blocks import rule_text
        from arena.blocks import settings as blocks_settings

        rule = rule_text(blocks_settings(study["blocks"]))
        study["backlog_priority"] = ("never-run games in roster order, then parked old games by most levels and fewest "
                                     "gameplay completion tokens (arena.blocks)")
    elif study.get("level_budget_policy") == "rank":
        cfg = study["rank"]
        h = (f"{cfg['horizon_rel']:g} x the run's median completion tokens per cleared level (floor "
             f"{cfg['horizon_floor_tokens']:,}; {cfg['horizon_tokens']:,} before the first clear)"
             if cfg["horizon_rel"] is not None else f"{cfg['horizon_tokens']:,} tokens")
        rule = (f"no opening and no level budgets: {cfg['slots']} focus games at {cfg['window']:,} tokens from the release; "
                f"every {cfg['tick_seconds']:g} s every game is ranked by (levels + 1)^{cfg['depth_power']:g} / "
                f"(N(N+1)/2) x ({cfg['prior_clears']:g} + sum over its clears of exp(-tokens since the clear / H)) / "
                f"({cfg['prior_tokens']:,} + H(1 - exp(-its tokens / H))), H = {h} "
                f"({'every clear' if cfg['clears'] == 'all' else 'the latest clear only'}); free slots take the top games, "
                f"then a waiting game replaces the worst focus game that has generated {cfg['tenure_tokens']:,} tokens "
                f"since its admission while it scores more than {1 + cfg['hysteresis']:g}x as much")
        study["backlog_priority"] = "continuous token/clear ranking (arena.rank)"
    elif study.get("level_budget_policy") == "fixed-2x":
        multiple = study.get("level_budget_multiple", 2)
        rule = (f"{multiple:g}*L*T1 for playing level L, granted in full at every clear and re-admission; level-0 "
                f"re-admission = the game's spend at its first park (floor {floor}); " + clock_text)
        if study.get("t1_cap_seconds") is not None:
            rule += f"; T1 capped at {study['t1_cap_seconds']:g} s in the level budgets"
        if study.get("t1_clip") is not None:
            low, high = study["t1_clip"]
            rule += f"; T1 clipped to [{low:g}, {high:g}] x the median T1 of the games with a clear"
        study["backlog_priority"] = "parked clears first, then fewest admissions, then most levels, then roster order"
        if study.get("backlog_key") == "progress-fruitless":
            # hero-v2 run 2.
            study["backlog_priority"] = (
                "a level cleared since the last admission first (in the tenure or while parked), then the fewest "
                "fruitless tenures on the current level (steady-state parks without a clear, reset at every clear), "
                "then most levels, then " + ("the most gameplay tokens on level 0 for level-0 games, then "
                                             if study.get("level0_order") == "tokens" else "")
                + "roster order")
        if study.get("backlog_policy") == "thompson":
            study["backlog_priority"] = ("parked clears first, then a Thompson draw Beta(1 + levels, 1 + fruitless "
                                         "admissions) damped by actions since a new board")
        if study.get("backlog_policy") == "llm":
            study["backlog_priority"] = ("parked clears first, then the served model's latest ranking of the waiting games "
                                         f"(every {study['llm_backlog']['interval_seconds']} s of phase 2), then fewest "
                                         "admissions, most levels, roster order")
    else:
        rule = f"(k+1)*T1; regrant max({study.get('regrant_multiple', 1.5):g}*spent, spent+{plus}); " + clock_text
        if study.get("max_level_grant_seconds") is not None:
            rule += f"; cap each steady-state allocation at {study['max_level_grant_seconds']:g} active seconds"
        if study.get("max_level_grant_t1_multiple") is not None:
            rule += f"; cap each steady-state allocation at {study['max_level_grant_t1_multiple']:g}*T1 (renewable)"
        study["backlog_priority"] = ("fewest completion tokens, then roster order" if study["budget_clock"] == "tokens"
                                     else "fewest inference seconds, then roster order")
    if study["budget_clock"] == "tokens" and study.get("level_budget_policy") in ("index", "commit", "deep-explore"):
        # No level budgets; the token clock only sets T1 and the opening switch's keep order (fastest first clear).
        rule += "; T1 and the opening-switch keep order in completion tokens (gameplay replies)"
    if study["budget_clock"] == "tokens" and study.get("level_budget_policy") == "blocks":
        rule += ("; the level-0 budget counts gameplay completion tokens on the token budget clock (no level budgets; T1 "
                 "records only)")
    if study["budget_clock"] == "tokens" and study.get("level_budget_policy") == "rank":
        # Rank has no opening and no level budgets and ranks on every request's tokens: the clock only sets the
        # recorded T1 / budget fields.
        rule += "; the budget clock does not enter the ranking (records only: T1 in gameplay completion tokens)"
    if study["budget_clock"] == "tokens" and study.get("backlog_policy") == "progress":
        rule += "; the progress order's 90 s prior is 5,760 tokens (three floors)"
    study["level_budget_rule"] = rule
    if study["budget_clock"] == "tokens":
        study["token_budget"]["rule"] = rule
    if probe_games:
        # Short probes (10-15 minute Modal probes beside the one-hour Kaggle runs):
        # a fixed set of games plays concurrently with no phase switch, and the metric is level clears
        # per completion token on those games, not the score.
        if exclude_games or not paired_30m or smoke or competition or evolve or wall_seconds is None:
            raise ValueError("--probe-games needs the 30-minute Duck arena with --wall-seconds and no other roster option")
        wanted = [str(g).strip()[:4] for g in probe_games if str(g).strip()]
        present = {g[:4] for g in study["games"]}
        if not wanted or set(wanted) - present:
            raise ValueError(f"Unknown probe games: {sorted(set(wanted) - present)}")
        exclude_games = [g for g in sorted(present) if g not in wanted]
        study["probe"] = {"games": wanted, "metric": "level clears per 10K completion tokens per game",
                          "opening_switch": "never (all probe games play concurrently for the whole wall)"}
        study["opening_switch_level1_games"] = len(wanted) + 1
    if generated_games:
        # Teacher runs: Flash plays the post-training arm's generated games
        # (posttrain.games) under this harness, all concurrently for a short wall, so its replies and
        # env-verified level clears become training data. The bundle carries the games'
        # environment_files only, never their notes (hidden rules and solutions).
        if probe_games or exclude_games or not paired_30m or smoke or competition or evolve or wall_seconds is None:
            raise ValueError("--generated-games needs the 30-minute Duck arena with --wall-seconds and no other roster option")
        from posttrain.split import is_generated
        source = Path(generated_games).resolve() / "environment_files"
        ids = sorted(json.loads(p.read_text())["game_id"] for p in source.glob("*/*/metadata.json"))
        if not ids or not all(is_generated(g) for g in ids) or len({g[:4] for g in ids}) != len(ids):
            raise ValueError(f"--generated-games needs one version of each qNNN game, found {ids}")
        shutil.copytree(source, output / "generated_games/environment_files", ignore=ignore)
        study["games"], study["games_excluded"] = ids, []
        study["generated_games"] = {"dir": "generated_games/environment_files", "games": ids}
        study["probe"] = {"games": [g[:4] for g in ids], "metric": "level clears per 10K completion tokens per game",
                          "opening_switch": "never (all generated games play concurrently for the whole wall)"}
        study["opening_switch_level1_games"] = len(ids) + 1
        study["roster"] = f"{len(ids)} generated games (posttrain.games)"
    if exclude_games:
        # Roster study (25 September, user decision): ft09 and lp85 are solved in most 30-minute
        # runs and their opening luck explains two thirds of the score variance; a roster without
        # them measures the changes that get other games through level 1.
        # Competition builds (a 30-minute public preview on the 23 games): the roster
        # shapes only the public preview; the hidden rerun enumerates the gateway.
        if not paired_30m or smoke:
            raise ValueError("--exclude-games applies to the 30-minute arena only (not smoke)")
        prefixes = tuple(str(g).strip()[:4] for g in exclude_games if str(g).strip())
        kept = [g for g in study["games"] if not g.startswith(prefixes)]
        dropped = [g for g in study["games"] if g.startswith(prefixes)]
        if len(dropped) != len(prefixes):
            raise ValueError(f"Unknown games to exclude: {sorted(set(prefixes) - {g[:4] for g in dropped})}")
        study["games"], study["games_excluded"] = kept, dropped
        study["roster"] = f"{len(kept)} public games (excluded: {', '.join(g[:4] for g in dropped)})"
    if evolve:
        # Offline harness evolution: the local model is the proposer,
        # the arena score at the round wall is the gate, dev and held-out rosters are fixed splits of
        # the roster without ft09 and lp85 (the games solved in most runs).
        if not paired_30m or smoke or competition or arena_variant != "reclaim-fixed-2x":
            raise ValueError("--evolve requires the 30-minute arena on the reclaim-fixed-2x variant")
        by_prefix = {g[:4]: g for g in study["games"]}
        missing = [g for g in (*EVOLVE_DEV_GAMES, *EVOLVE_HOLDOUT_GAMES) if g not in by_prefix]
        if missing:
            raise ValueError(f"Evolution rosters name games outside the roster: {missing}")
        study["evolution"] = {
            "dev_games": [by_prefix[g] for g in EVOLVE_DEV_GAMES],
            "holdout_games": [by_prefix[g] for g in EVOLVE_HOLDOUT_GAMES],
            "rounds": 5, "round_wall_seconds": 1800, "eval_grace_seconds": 480,
            "smoke": {"games": 3, "wall_seconds": 150},
            "proposer": {"tool_turns": 40, "seconds": 1200, "cap_tokens": 8192, "reasoning_effort": "xhigh",
                         "result_chars": 8000, "context_chars": 300000, "temperature": 0.7},
            "holdout": True,
            "gate": "candidate score at the round wall > incumbent (ties: clears); co-metrics recorded",
        }
        study["notebook_seconds"] = 28800
        study["kernel_timeout"] = 28800
        if evolve_rounds is not None:
            # Rehearsal (an 8-hour evolution with no visibility is not acceptable):
            # fewer rounds on a short wall run every driver path (baseline, proposer, validation,
            # smoke, candidate evaluation, gate, held-out pair) in about 1.5 hours.
            study["evolution"]["rounds"] = int(evolve_rounds)
    if wall_seconds is not None and wall_seconds != study["wall_seconds"]:
        # One-hour arena: every run on the 23-game roster gets 3,600 s of
        # gameplay; the notebook and kernel limits grow by the same amount.
        if not paired_30m or smoke or competition or throughput_bench:
            raise ValueError("--wall-seconds applies to the Duck-transfer arena only")
        delta = int(wall_seconds) - int(study["wall_seconds"])
        study["wall_seconds"] = int(wall_seconds)
        if evolve:
            # One-hour evolution rounds (26 September) run on Kaggle within its 12-hour session limit;
            # the driver skips rounds that no longer fit (about four at 3,600 s plus the held-out pair).
            study["evolution"]["round_wall_seconds"] = int(wall_seconds)
            study["notebook_seconds"] = 41400
            study["kernel_timeout"] = 43000
        else:
            study["notebook_seconds"] += delta
            study["kernel_timeout"] += delta
    if level_bench is not None:
        study = level_bench_study(study, output, level_bench, candidate_profile=candidate_profile)
    if llamacpp is not None:
        study = llamacpp_study(study, llamacpp, probe=bool(probe_games), smoke=smoke, sglang=sglang,
                               candidate=candidate_profile, competition=competition,
                               bench=throughput_bench, evolve=evolve,
                               slots=llamacpp_slots, kv_unified=llamacpp_kv_unified,
                               serve_check=llamacpp_serve_check)
    study["name"] = name
    study["smoke_active"] = bool(smoke)
    if clone_instances is not None:
        # Research roster of cyclic clones (a 110-game one-hour arena built from 20 games)
        # through the competition preview's local gateway (arena.competition.open_arcade); the instances play
        # under one scorecard exactly as in a public preview. Not a submission build.
        study['competition'] = {
            'preview_instances': int(clone_instances), 'preview_seconds': int(study['wall_seconds']),
            'submission_seconds': 32400, 'cleanup_seconds': 600, 'max_actions_per_game': 100000,
            'roster': f"{int(clone_instances)} cyclic clones of the {len(study['games'])} roster games",
            'research_clones': True,
        }
        study['roster'] = study['competition']['roster']
    if competition:
        study['competition'] = {
            'preview_instances': 110, 'preview_seconds': 3600,
            'submission_seconds': 32400, 'cleanup_seconds': 600,
            'max_actions_per_game': 100000,
            'roster': 'cyclic public clones in preview; gateway enumeration on hidden rerun',
        }
        # Public preview: a 25-game, 30-minute sanity trial first, then
        # the 110-instance hour; the hidden rerun skips the sanity trial (bootstrap) and plays
        # the gateway roster only.
        study['trials'] = [
            {"name": "sanity-25g", "kind": "duck-transfer", "competition": False, "wall_seconds": 1800},
            {"name": "duck-transfer", "kind": "duck-transfer"},
        ]
        study.update(wall_seconds=3600, notebook_seconds=7500, kernel_timeout=32400)
        if study.get('require_focus_windows_fit'):
            # Competition (the hero serving as a submission): a served cache that fits fewer
            # full focus windows than the study names runs with what fits (at least kernels-v2's six) instead of
            # refusing to start (arena.capacity.competition_focus_fit).
            study['focus_fit_fallback'] = {'min_slots': 6}
        overlay = study['server'].get('expert_overlay')
        if overlay is not None:
            # hero-v2 competition: the GPTQ-ARC overlay is the private Kaggle dataset of the same
            # name, attached to the kernel. A dataset version is immutable and the manifest (sha256 pinned here) pins
            # every shard's sha256, so the notebook checks sizes only instead of hashing 63 GiB from /kaggle/input
            # before vLLM reads the same bytes (expert-quant.md section 6.3); the served bytes are unchanged.
            study['server']['expert_overlay'] = {**overlay, 'check': 'size'}
            if overlay['dataset'] not in study.get('model_datasets', []):
                study['model_datasets'] = [*study.get('model_datasets', []), overlay['dataset']]
        if preview_seconds != 3600 or not sanity_trial:
            # Short public preview (a 15-minute 110-instance check before
            # submitting); the hidden rerun is unchanged (it plays for the whole envelope).
            # Up to 8 h (four- and eight-hour Modal runs of the competition bundle; a Kaggle saved
            # version of 28,800 s plus setup still fits the 12 h kernel limit).
            if not 300 <= int(preview_seconds) <= 28800:
                raise ValueError('The public preview must be 300-28800 seconds')
            study['competition']['preview_seconds'] = int(preview_seconds)
            # The bootstrap budgets each preview trial by the study wall (lane-v1 preview skipped its
            # trial: 1,780 s left < 3,600 + 300); the hidden rerun recomputes its wall from the envelope.
            study['wall_seconds'] = int(preview_seconds)
            if not sanity_trial:
                study['trials'] = [t for t in study['trials'] if t['name'] != 'sanity-25g']
            # Setup allowance 2,700 s: the lane-v1 preview spent 915 s starting vLLM and 1,240 s before its trial.
            study['notebook_seconds'] = 2700 + int(preview_seconds) + (1800 if sanity_trial else 0)
        if preview_instances != 110:
            # Public preview with one instance per roster game (the 23-game roster
            # for 30 minutes, comparable with the 23-game arena runs); the hidden rerun is unchanged.
            if not 1 <= int(preview_instances) <= 110:
                raise ValueError('The public preview must have 1-110 instances')
            study['competition']['preview_instances'] = int(preview_instances)
        if preview_live_score:
            # Live score in the public preview's progress (4 October: Modal preview tests of competition bundles had
            # score None): the controller reads the local gateway's open scorecard in-process; never on the rerun.
            if no_preview:
                raise ValueError('preview_live_score needs the public preview')
            study['competition']['preview_live_score'] = True
        if no_preview:
            # No public preview (out of Kaggle GPU quota; the build is tested offline by a Modal
            # rerun rehearsal instead). Outside the hidden rerun the notebook only verifies and unpacks the runtime
            # archive and writes the placeholder that selects it for a rerun (about a minute, no server, no games);
            # the hidden rerun's branch is unchanged.
            study['competition']['no_preview'] = True
        if rerun_rehearsal:
            # Exercise the hidden-rerun branch (rerun flag, skipped sanity trial, rerun envelope and
            # tolerance, gateway enumeration, no placeholder parquet) on the public cyclic gateway
            # in an ordinary kernel, so a submission never spends a daily quota on an untested path.
            # The only difference from a real rerun is the gateway host and roster.
            # The envelope is a build option (a short rehearsal saves Kaggle quota);
            # gameplay is what remains after setup and the cleanup reserve.
            if not 2400 <= int(rehearsal_seconds) <= 6000:
                raise ValueError('The rerun rehearsal envelope must be 2400-6000 seconds')
            study['competition'].update(rehearsal=True, submission_seconds=int(rehearsal_seconds))
            study.update(kernel_timeout=int(rehearsal_seconds) + 600)
    if hero_v2_precheck is not None:
        if hero_v2_smoke:
            raise ValueError("--hero-v2-precheck and --hero-v2-smoke are separate bundles")
        study = hero_v2_precheck_study(study, output, *hero_v2_precheck)
    if prefix_unit_int4 is not None:
        if hero_v2_smoke or hero_v2_precheck is not None:
            raise ValueError("--prefix-unit-int4 is its own bundle")
        study = prefix_unit_int4_study(study, output, *prefix_unit_int4, tf_only=prefix_unit_int4_tf_only)
    if decode_sweep is not None:
        if hero_v2_smoke or hero_v2_precheck is not None or prefix_unit_int4 is not None:
            raise ValueError("--decode-sweep is its own bundle")
        study = decode_sweep_study(study, output, *decode_sweep)
    if prefix_cache_v3_smoke is not None:
        if hero_v2_smoke or hero_v2_precheck is not None:
            raise ValueError("--prefix-cache-v3-smoke is its own bundle")
        study = prefix_cache_v3_smoke_study(study, output, *prefix_cache_v3_smoke, recheck=prefix_cache_v3_recheck)
    elif prefix_cache_v3_recheck:
        raise ValueError("--prefix-cache-v3-recheck narrows --prefix-cache-v3-smoke")
    if hero_v2_soak:
        if hero_v2_smoke or not study.get("candidate_profile"):
            raise ValueError("--hero-v2-soak is the candidate hero-v2 soak (not with --hero-v2-smoke)")
        study = candidate_soak_study(study)
    if hero_v2_smoke and study.get("candidate_profile"):
        if hero_v2_reference is not None:
            raise ValueError("A candidate hero-v2 smoke measures its own BF16-KV reference (--hero-v2-drift-inputs)")
        study = candidate_smoke_study(study, output, hero_v2_drift_inputs, hero_v2_transcript)
    elif hero_v2_smoke:
        study = hero_v2_smoke_study(study, output, hero_v2_reference)
    elif hero_v2_reference is not None:
        raise ValueError("--hero-v2-reference is the hero-v2 smoke's drift reference")
    if not hero_v2_smoke and (hero_v2_drift_inputs is not None or hero_v2_transcript is not None):
        raise ValueError("--hero-v2-drift-inputs and --hero-v2-transcript belong to the candidate hero-v2 smoke")
    if kernel_gate:
        study = kernel_gate_study(study, output)
    if draft_head is not None:
        study = draft_head_study(study, output, draft_head)
    if draft_head_overlay is not None:
        if draft_head is not None:
            raise ValueError("--draft-head-overlay serves a trained head; it is not a training job option")
        study = draft_head_overlay_study(study, output, draft_head_overlay)
    if expert_quant is not None:
        if (not paired_30m or smoke or arena_variant is not None or throughput_bench or competition
                or candidate_profile or sglang or kernel_gate or level_bench is not None or evolve):
            raise ValueError('The expert-quantization trials build on the plain Flash 30-minute arena profile alone')
        if expert_quant_phase == "evaluate":
            study = expert_quant_eval_study(study, output, expert_quant, expert_quant_candidates or {})
        else:
            study = expert_quant_study(study, output, expert_quant, expert_quant_phase,
                                       overrides=expert_quant_settings, prune_variant=expert_quant_prune)
    if drift_long is not None:
        if (not paired_30m or smoke or arena_variant is not None or throughput_bench or competition
                or candidate_profile or sglang or kernel_gate or level_bench is not None or evolve
                or expert_quant is not None):
            raise ValueError('The drift test builds on the plain Flash 30-minute arena profile alone')
        study = drift_long_study(study, output, drift_long)
    if kv4_drift is not None:
        if (not paired_30m or smoke or arena_variant is not None or throughput_bench or competition
                or candidate_profile or sglang or kernel_gate or level_bench is not None or evolve
                or expert_quant is not None or drift_long is not None):
            raise ValueError('The 4-bit KV drift test builds on the plain Flash 30-minute arena profile alone')
        study = kv4_drift_study(study, output, kv4_drift)
    if kv8_prefill_check is not None:
        if (not paired_30m or throughput_bench or arena_variant is not None or kv4_drift is not None
                or drift_long is not None or expert_quant is not None):
            raise ValueError('The kv8 prefill check builds on the plain Flash 30-minute arena profile alone')
        study = kv8_prefill_study(study, output, kv8_prefill_check)
    if stall_dump is None and study.get('stall_dump_seconds'):
        stall_dump = study['stall_dump_seconds']  # hero-v2's stall_dump component
    if stall_dump is not None:
        # Diagnostics (prune12 hang, 29 September): all-thread stack dumps in the pruned worker when one step
        # stalls for N seconds, on every trial's server (arena.serving ARENA_STALL_DUMP; needs server.expert_prune).
        # A candidate server arms it without pruning (arc_infra.stall_dump_patch, arena.candidate_serving).
        if (not (study['server'].get('expert_prune') or study.get('candidate_profile'))
                or not 30 <= int(stall_dump) <= 240):
            raise ValueError('--stall-dump needs a pruned-hero variant (or a candidate server) and 30..240 seconds')
        study['trials'] = [{**t, 'server_env': {**(t.get('server_env') or {}), 'ARENA_STALL_DUMP': str(int(stall_dump))}}
                           for t in study['trials']]
    if gpu_memory_utilization is not None:
        # GPU memory headroom (the dfr-combined bundle's 640 px images ran the vision encoder out of
        # memory at the fork's 0.96 on Modal): every trial's server profiles its KV pool at this share instead
        # (arena.serving ARENA_GPU_MEMORY_UTILIZATION); the pool shrinks by the difference, the rest stays free.
        value = gpu_memory_utilization
        if not study.get('hero_v2') or type(value) is not str or re.fullmatch(r'0\.\d{1,3}', value) is None or not 0.5 <= float(value) < 0.96:
            raise ValueError('gpu_memory_utilization is a hero-v2 option: a decimal string from 0.5 to below 0.96')
        study['server']['gpu_memory_utilization'] = value
        study['trials'] = [{**t, 'server_env': {**(t.get('server_env') or {}), 'ARENA_GPU_MEMORY_UTILIZATION': value}}
                           for t in study['trials']]
    if study.get('hero_v2'):
        # A hung server start costs the ready guard's 900 s, not the whole reservation (every trial: no restart).
        # Competition builds (29 September): Kaggle reads the weights and the GPTQ-ARC overlay from its dataset mounts at
        # ~4 s per shard (206 shards, ~15 min), so the 900 s total killed the first preview 20 s before the load ended;
        # they keep the 180 s after-capture check with a 3,000 s total.
        guard = HERO_V2_COMPETITION_READY_GUARD if competition else HERO_V2_READY_GUARD
        if study['server'].get('mtp_tokens') and not competition:
            guard = HERO_V2_MTP_READY_GUARD
        if (study['server'].get('serving_kernels') or {}).get('expert_int3') and not competition:
            guard = HERO_V2_HUMMING_READY_GUARD
        study['trials'] = [{**t, 'server_env': {**(t.get('server_env') or {}), 'ARENA_READY_GUARD': guard}}
                           for t in study['trials']]
        if competition and (study['server'].get('serving_kernels') or {}).get('expert_int3'):
            # Competition int3 builds (30 September): Kaggle's weight load (1,030-1,375 s to CUDA-graph capture with
            # Marlin) plus Humming's kernel build can pass the pinned recipe's 1,800 s setup ceiling, so its server-ready
            # and setup totals rise to 2,700 / 3,600 s (arena.serving ARENA_SETUP_TIMEOUTS); the ready guard's
            # after-capture check stays. The preview's notebook budget grows with it (setup allowance 3,900 s).
            study['trials'] = [{**t, 'server_env': {**t['server_env'], 'ARENA_SETUP_TIMEOUTS': COMPETITION_SETUP_TIMEOUTS}}
                               for t in study['trials']]
            study['notebook_seconds'] = max(study['notebook_seconds'],
                                            3900 + int(study['competition']['preview_seconds']))
    if study.get('hero_v2') and study.get('candidate_profile'):
        # Candidate hero-v2 (Holo4): exact cached tokens in every reply's usage (the prefix hit
        # rate per request; usage reporting only) on the arena trials too, as in the soak.
        study['trials'] = [{**t, 'server_env': {**(t.get('server_env') or {}), 'ARENA_PROMPT_TOKENS_DETAILS': '1'}}
                           if t['kind'] == 'duck-transfer' else t for t in study['trials']]
    if expert_prune_hang_test:
        if (not paired_30m or smoke or arena_variant != 'fixed-2x-mtp0-wide-kv8-hero-prune12' or throughput_bench
                or competition or expert_usage is not None or expert_prune_check is not None):
            raise ValueError('The prune hang test builds on the prune12 hero variant alone (--arena-30m '
                             '--arena-variant fixed-2x-mtp0-wide-kv8-hero-prune12)')
        study = expert_prune_hang_study(study)
    if expert_prune_check is not None:
        if (not paired_30m or smoke or arena_variant != 'fixed-2x-mtp0-wide-kv8-hero' or throughput_bench
                or competition or expert_usage is not None or kernel_gate):
            raise ValueError('The expert-prune check builds on the hero variant alone (--arena-30m '
                             '--arena-variant fixed-2x-mtp0-wide-kv8-hero)')
        study = expert_prune_check_study(study, output, expert_prune_check, levels=expert_prune_levels)
    if expert_usage is not None:
        if (not paired_30m or smoke or arena_variant is not None or throughput_bench or competition
                or candidate_profile or sglang or kernel_gate or level_bench is not None or evolve):
            raise ValueError('The expert-usage measurement builds on the plain Flash 30-minute arena profile alone')
        study = expert_usage_study(study, output, expert_usage, measure_only=expert_usage_measure_only)
    if study["server"].get("expert_prune"):
        # Load-time expert pruning (arena.variants.EXPERT_PRUNE_VARIANTS): the mask ships in the bundle, pinned by sha256.
        from arena.variants import EXPERT_PRUNE_VARIANTS

        prune = study["server"]["expert_prune"]
        source = next(v["source"] for v in EXPERT_PRUNE_VARIANTS.values() if v["mask"] == prune["mask"])
        target = output / prune["mask"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != prune["sha256"]:
            raise ValueError(f"{source}: sha256 differs from the variant's pinned mask")
    if study.get("hero_v2") and study["hero_v2"]["requested"]["token_budgets"] and study["budget_clock"] != "tokens":
        raise ValueError("hero-v2 asked for token budgets but the built study runs on the "
                         f"{study['budget_clock']} clock")
    study = with_compaction_thinking(study, compaction_thinking)
    study = with_reasoning_history(study, reasoning_history)
    if early_stop is not None:
        # Early stop ("stop after 1 hour if score is higher than hero-v2 @ 1 hour";
        # arena.early_stop): the arena trial ends gracefully at its first score read at or after at_seconds when the
        # score is above score_above; a stop-request file ends it the same way.
        from arena.early_stop import settings as early_stop_settings

        if not {t.get("kind") for t in study["trials"]} & {"duck-transfer", "duck-public"}:
            raise ValueError("--early-stop applies to the Duck arena trials")
        study["early_stop"] = early_stop_settings(early_stop, study["wall_seconds"])
    if study.get("level_budget_policy") == "blocks":
        # hero-v2-blocks: the scheduler's wall is the competition-length play wall and the run stops
        # unconditionally earlier (early_stop always); a block is the stream count, block 1 = every source game once
        # plus distinct second clones (user requirement), checked on the clone roster's cyclic order.
        from arena.blocks import check_block1, composition
        from arena.blocks import settings as blocks_settings

        cfg = blocks_settings(study["blocks"])
        stop = study.get("early_stop") or {}
        clones = study.get("competition") or {}
        if competition:
            # The competition gateway (the blocks3-plans-prune25-40 submission): the scheduler's
            # wall is the submission's play time, set at runtime from the envelope, the measured setup and the cleanup
            # reserve (arena.competition.resolve_blocks); the build's wall is its nominal value (Kaggle setup ~1,160 s)
            # for the plan below. The public preview plays block 1 of that wall for preview_seconds.
            envelope = clones["submission_seconds"] - clones["cleanup_seconds"]
            if (not clones["preview_seconds"] < float(cfg["wall_seconds"]) < envelope or stop
                    or clones.get("preview_instances") != cfg["roster_games"] or study["active_limit"] != cfg["slots"]):
                raise ValueError("competition blocks need a nominal wall between the preview and the envelope less "
                                 "cleanup, no early stop, the preview roster of roster_games instances and the slots "
                                 "as the focus streams")
            stop = {"at_seconds": clones["preview_seconds"], "always": True}
            clones["blocks_wall"] = {"nominal_seconds": cfg["wall_seconds"],
                                     "rule": "runtime: submission_seconds from the notebook start - measured setup - "
                                             "cleanup_seconds; preview: that projected wall, stopped at "
                                             "preview_seconds (arena.competition.resolve_blocks)"}
        elif (float(cfg["wall_seconds"]) != float(study["wall_seconds"]) or (stop and not stop.get("always"))
                or not clones.get("research_clones") or clones.get("preview_instances") != cfg["roster_games"]
                or study["active_limit"] != cfg["slots"]):
            raise ValueError("blocks need the study wall as their wall, an unconditional early stop, the clone roster "
                             "of roster_games instances and the slots as the focus streams")
        sources = [study["games"][i % len(study["games"])] for i in range(cfg["roster_games"])]
        study["blocks_plan"] = {"block_seconds": cfg["block_seconds"], "block_sizes": cfg["block_sizes"],
                                "run_stop_seconds": stop.get("at_seconds", study["wall_seconds"]),
                                # The window replay's pool (hero-v2-blocks.md): the trial refuses a smaller served pool
                                # before gameplay rather than run 37 streams on less cache than was replayed.
                                "min_pool_blocks": 1260 if not competition else
                                study["hero_v2"]["requested"].get("density_blocks") or 1260,
                                "block1": check_block1(sources, cfg),
                                "blocks": [{k: b[k] for k in ("block", "games", "distinct_sources", "second_clones")}
                                           for b in composition(sources, cfg)],
                                "roster_rule": "cyclic clones k000..k109 of the roster games in roster order "
                                               "(arena.competition.open_arcade public-roster.json)"}
        if cfg["partition"] == "streams":
            # every block runs all the streams (40 / 40 / 40 at 110 games): the last block's spare
            # slots take the best old games (old_refill: most levels, then fewest gameplay tokens).
            study["blocks_plan"]["streams_per_block"] = [cfg["slots"]] * cfg["blocks"]
            study["blocks_plan"]["spare_slots_for_old_refills"] = [cfg["slots"] - n for n in cfg["block_sizes"]]
    if study.get("hero_v2") and "smoke" not in study["hero_v2"] and (
            study["wall_seconds"] == 7200 or (study.get("early_stop") or {}).get("always")) \
            and study.get("focus_oversubscription"):
        # The 2 h follow-ups: the Modal tests' 9,600 s (the bootstrap gets 9,540)
        # against the measured setup (hero-v2 2 h: ~705 s to the trial; run 1: ~630 s) x 1.5, the 7,200 s wall plus
        # 300 s of drain, scorecard close and report, and 300 s of server stop and evidence; the build fails otherwise.
        from arena.timeline import check as check_timeline

        plan = check_timeline({**study, "notebook_seconds": 9540}, setup_seconds=round(705 * 1.5), restart_seconds=0,
                              run_seconds={t["name"]: 7200 + 300 for t in study["trials"]}, margin=1.0,
                              teardown_seconds=300)
        study["timeline"] = {"notebook_seconds": 9540, "setup_seconds": round(705 * 1.5), "trial_tail_seconds": 300,
                             "teardown_seconds": 300, "plan": plan}
    if campaign_arm is not None:
        from arena.campaign import finalize
        study = finalize(study, campaign_arm)
    (output / "configs/comparison.json").write_text(json.dumps(study, indent=2) + "\n")
    for config in CONFIGS[1:]:
        shutil.copyfile(ROOT / config, output / "configs" / Path(config).name)
    shutil.copyfile(ROOT / "scripts/arena/bundle_bootstrap.py", output / "bootstrap.py")
    shutil.copyfile(ROOT / "scripts/arena/bootstrap.py", output / "bootstrap_arena.py")
    subprocess.run(
        ["uv", "export", "--frozen", "--no-dev", "--no-editable", "--no-emit-project",
         "--output-file", str(output / "requirements.txt")],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
    )
    if dependencies:
        download_dependencies(output)
    identity = {
        "files": manifest(output),
        "source": "current working tree",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dependencies_prepared": dependencies,
        "gpu_benchmarks_launched": False,
        "study": name,
        "smoke": bool(smoke),
    }
    (output / "MANIFEST.json").write_text(json.dumps(identity, indent=2))
    return output, study


# Kernel work, phase 2: a quality gate before an arena trial on patched serving.
KERNEL_QUALITY_INPUTS = "data/kernel_quality"
# The recorded c110 mtp0-wide control's profiled KV pool (arena-duck-60m-c110-mtp0-wide-v1, cache-layout.json:
# 751 blocks). The patched server holds ~3.3 GB less weight; pinning the pool keeps the arena's admission
# identical to the recorded run, so the kernels are the only difference.
KERNEL_GATE_KV_BYTES = {0: 15226675200}
KERNEL_GATE_BENCH = {"prompt_tokens": [4096, 16384], "concurrency": [1, 6], "output_tokens": 512,
                     "shared_prefix": None, "transcript": None, "points": [],
                     "profile": {"prompt_tokens": 2048, "concurrency": [1], "output_tokens": 96}}


def kernel_gate_study(study, output):
    """Prepend the kernel quality gate to the study's trials (``arena.kernel_gate``).

    ``kernel-ref`` serves the pinned runtime (ARENA_DENSE_KERNELS=off) with the arena's profile and records
    the reference; ``kernel-gate`` restarts on the study's ``server.dense_kernels`` with the KV pool pinned to
    the recorded control's and compares; every original trial then ``requires`` the gate and runs on that
    same server (no restart). Both quality servers arm the torch profiler for one short profiled point."""
    from arena.kernel_gate import GREEDY, GSM8K, TEACHER, THRESHOLDS

    study = json.loads(json.dumps(study))
    spec = study["server"].get("dense_kernels")
    mtp = study["server"].get("mtp_tokens", 3)
    if not spec:
        raise ValueError("--kernel-gate needs an arena variant with server.dense_kernels")
    if mtp not in KERNEL_GATE_KV_BYTES or study["server"].get("speculative") is not None:
        raise ValueError("The kernel gate's pinned KV pool is recorded for the MTP-0 profile only")
    if any(t["kind"] != "duck-transfer" or t.get("server_env") for t in study["trials"]):
        raise ValueError("The kernel gate precedes plain Duck-transfer trials")
    target = output / "configs/kernel-quality"
    target.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in ("prompts.json", "gsm8k.jsonl"):
        shutil.copyfile(ROOT / KERNEL_QUALITY_INPUTS / name, target / name)
        digests[name] = hashlib.sha256((target / name).read_bytes()).hexdigest()
    profiled = {"ARENA_TORCH_PROFILER": "1"}
    patched = {"ARENA_KV_CACHE_MEMORY_BYTES": str(KERNEL_GATE_KV_BYTES[mtp]), **profiled}
    study["trials"] = [
        {"name": "kernel-ref", "kind": "kernel-quality", "role": "reference", "competition": False,
         "wall_seconds": 900, "server_env": {"ARENA_DENSE_KERNELS": "off", **profiled},
         "bench": dict(KERNEL_GATE_BENCH)},
        {"name": "kernel-gate", "kind": "kernel-quality", "role": "gate", "reference": "kernel-ref",
         "competition": False, "wall_seconds": 900, "server_env": dict(patched), "bench": dict(KERNEL_GATE_BENCH)},
        *[{**t, "requires": "kernel-gate", "server_env": dict(patched)} for t in study["trials"]],
    ]
    study["kernel_gate"] = {
        "dense_kernels": spec, "thresholds": dict(THRESHOLDS), "greedy": dict(GREEDY), "teacher_forced": dict(TEACHER),
        "gsm8k": dict(GSM8K), "inputs": {"folder": "configs/kernel-quality", "source": KERNEL_QUALITY_INPUTS,
                                         "sha256": digests},
        "kv_cache_memory_bytes": KERNEL_GATE_KV_BYTES[mtp],
        "arena_server": "the gate's server (patched, KV pool pinned, torch profiler armed and idle)",
    }
    # Two server starts (~500 s each on Modal) and two quality passes (~600 s each) before the arena.
    study["notebook_seconds"] += 2700
    study["kernel_timeout"] += 2700
    return study


# Expert-usage measurement (docs/validation-20260927/expert-usage.md). MTP off (the draft
# layer's MoE would share the probe's layer ids), and an explicit 7.5 GiB KV pool (308K MTP-0 tokens, above the
# 262,144-token max_model_len): prompt_logprobs on a 4,096-token chunk needs ~7-9 GiB of logits transients that
# the profiled pool (utilization 0.96, ~14.2 GiB of KV) would not leave free.
EXPERT_USAGE_KV_BYTES = 8053063680
EXPERT_USAGE_FILES = ("usage.json", "heldout.json", "selection.json")
EXPERT_USAGE_WALL = 1150


def expert_usage_study(study, output, inputs, measure_only=False):
    """One ``expert-usage`` trial (``arena.expert_usage``) on the probe overlay; no games are played."""
    from arena.expert_usage import SETTINGS

    study = json.loads(json.dumps(study))
    target = output / "configs/expert-usage"
    target.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in EXPERT_USAGE_FILES:
        shutil.copyfile(Path(inputs) / name, target / name)
        digests[name] = hashlib.sha256((target / name).read_bytes()).hexdigest()
    selection = json.loads((target / "selection.json").read_text())
    study["server"].update(mtp_tokens=0, expert_probe=True)
    # v3: v1 and v2 hung in vLLM's post-capture decode warm-up (CUDA-graph replay).
    # The measurement runs first on arm (c), probe + 7.5 GiB pool with --enforce-eager (no graphs; the workload is
    # all prefill), for at most EXPERT_USAGE_WALL s. Then two server-start arms find the cause: (b) no probe on
    # the 7.5 GiB pool, (a) the probe on the profiled pool, both with CUDA graphs. Every start is inside the trial
    # loop (a failed start costs only its trial), and the ready guard fails a start 180 s after graph capture
    # (or 720 s after launch) instead of the pinned 1,500 s.
    guard = {"ARENA_READY_GUARD": "180,720"}
    kv = {"ARENA_KV_CACHE_MEMORY_BYTES": str(EXPERT_USAGE_KV_BYTES)}
    study["start_in_trial_loop"] = True
    study["trials"] = [
        {"name": "expert-usage", "kind": "expert-usage", "competition": False, "wall_seconds": EXPERT_USAGE_WALL,
         "server_env": {**kv, **guard, "ARENA_EXPERT_PROBE": "1", "ARENA_ENFORCE_EAGER": "1"}},
        {"name": "diag-b-noprobe-kv7.5", "kind": "server-check", "competition": False, "wall_seconds": 60,
         "server_env": {**kv, **guard, "ARENA_EXPERT_PROBE": "0"}},
        {"name": "diag-a-probe-profiled", "kind": "server-check", "competition": False, "wall_seconds": 60,
         "server_env": {**guard, "ARENA_EXPERT_PROBE": "1"}},
    ]
    study["expert_usage"] = {
        "settings": dict(SETTINGS), "kv_cache_memory_bytes": EXPERT_USAGE_KV_BYTES,
        "inputs": {"folder": "configs/expert-usage", "source": str(inputs), "sha256": digests,
                   **{split: {k: selection[split][k] for k in ("sequences", "tokens", "assistant_tokens", "families")}
                      for split in ("usage", "heldout")}},
    }
    if measure_only:
        # v4: v3 showed arm (c) starts; the measurement alone runs until the notebook deadline.
        study["trials"] = [{**study["trials"][0], "wall_seconds": 3600}]
        study["expert_usage"]["arms"] = "measurement only (probe, 7.5 GiB pool, --enforce-eager)"
        study.update(wall_seconds=120, notebook_seconds=1860, kernel_timeout=1920,
                     harness_selection="expert-usage measurement (no agent)")
        return study
    # Setup ~180 s, arm (c) start ~400 s, the measurement <= EXPERT_USAGE_WALL, two diagnostic starts <= ~480 s each.
    study.update(wall_seconds=120, notebook_seconds=2700, kernel_timeout=2760,
                 harness_selection="expert-usage measurement (no agent)")
    return study


# ARC expert re-quantization (docs/validation-20260927/expert-quant.md). MTP off (the draft
# layer's MoE would share layer ids with the capture), the pinned runtime otherwise (no dense or serving overlay).
EXPERT_QUANT_BF16 = "/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540"
EXPERT_QUANT_PHASES = {
    # Stage 1: capture 6 layers on ~100K tokens of recorded sequences (every 2nd row), stop the server, then the
    # microbenchmark in the pinned runtime (arena.expert_quant_bench); layers in priority order.
    "bench": {"inputs": ("stage1.json", "selection.json"),
              "settings": {"layers": [20, 0, 47, 10, 30, 40], "stride": 2, "concurrency": 2, "eval_rows": 4096,
                           "rtn_search": True, "bf16_dir": EXPERT_QUANT_BF16, "gpu_reserve_seconds": 200},
              "seconds": 1200},
    # Stage 2: capture every MoE layer on the calibration sequences (E4M3 rows, every 6th row: ~120K rows, ~15 GB)
    # and on the held-out sequences (BF16, every 16th row, capped), stop the server, then GPTQ over the 48 layers
    # (arena.expert_quant_gptq) writing replacement shards into the assets volume.
    "calibrate": {"inputs": ("calib.json", "eval.json", "selection.json"),
                  "settings": {"calib_stride": 6, "eval_stride": 16, "eval_max_rows": 12288, "concurrency": 3,
                               "eval_rows": 4096, "percdamp": 0.01, "min_rows": 64, "batch": 128, "scale_search": [1.0],
                               "bf16_dir": EXPERT_QUANT_BF16, "gpu_reserve_seconds": 1500,
                               "overlay_dir": "/assets/flash-input/datasets/richardcsaky/arc-expert-gptq-arc-v1"},
                  "seconds": 3300},
}
# Stage 2 end to end: kernel-quality trials (arena.kernel_gate) on held-out arena requests; the served checkpoint is
# the reference, each candidate a report-only gate after a server restart. MTP off; the KV pool is pinned so that
# prompt_logprobs on 4,096-token chunks fits (as the expert-usage measurement).
EXPERT_QUANT_KV_BYTES = 8053063680
EXPERT_QUANT_QUALITY = {"gsm8k_pass_seconds": 240, "gsm8k_repeat": False, "report_only": True}


def expert_quant_eval_study(study, output, inputs, candidates):
    """The end-to-end check: ``eq-ref`` on the served checkpoint, then one report-only gate per candidate
    (``{name: server_env}``) against it (``arena.kernel_gate``); no games are played."""
    study = json.loads(json.dumps(study))
    target = output / "configs/kernel-quality"
    target.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, source in (("prompts.json", Path(inputs) / "quality/prompts.json"),
                         ("gsm8k.jsonl", ROOT / KERNEL_QUALITY_INPUTS / "gsm8k.jsonl")):
        shutil.copyfile(source, target / name)
        digests[name] = hashlib.sha256((target / name).read_bytes()).hexdigest()
    pinned = {"ARENA_KV_CACHE_MEMORY_BYTES": str(EXPERT_QUANT_KV_BYTES)}
    study["server"].update(mtp_tokens=0)
    study["trials"] = [{"name": "eq-ref", "kind": "kernel-quality", "role": "reference", "competition": False,
                        "wall_seconds": 900, "server_env": dict(pinned), "quality": dict(EXPERT_QUANT_QUALITY)}]
    for name, env in candidates.items():
        study["trials"].append({"name": name, "kind": "kernel-quality", "role": "gate", "reference": "eq-ref",
                                "competition": False, "wall_seconds": 600, "server_env": {**pinned, **env},
                                "quality": dict(EXPERT_QUANT_QUALITY)})
    study["expert_quant"] = {"phase": "evaluate", "kv_cache_memory_bytes": EXPERT_QUANT_KV_BYTES,
                             "quality": dict(EXPERT_QUANT_QUALITY), "candidates": candidates,
                             "inputs": {"folder": "configs/kernel-quality", "source": str(inputs), "sha256": digests}}
    seconds = 900 + 1500 * len(candidates) + 900
    study.update(wall_seconds=60, notebook_seconds=seconds - 60, kernel_timeout=seconds,
                 harness_selection="expert quantization end-to-end check (no agent)")
    return study


# Long-context drift test (<=; docs/validation-20260927/hero-behavior.md, Test A): the
# kernels-v2 serving as reference (teacher forcing at concurrency 1, repeat at 4), then the full hero serving and the
# int8 KV cache alone as report-only gates, on c110 contexts of 39-44K tokens (scripts/drift_long_inputs.py). No
# greedy or GSM8K passes; MTP off; the KV pool pinned as in the expert-quant check (prompt_logprobs transients).
DRIFT_LONG_QUALITY = {"greedy": False, "gsm8k": False, "report_only": True}
DRIFT_LONG_KV8_ONLY = {"mamba_ssm_cache_dtype": None, "moe_backend": None, "draft_experts": None, "hc_fusion": False,
                       "topk_sampler": False, "kv_cache_8bit": "int8"}
# Setup + first start ~650 s, the reference ~250 s, two restarts ~550 s each, two gates ~150 s each (the expert-quant
# check: 161K teacher-forced positions in 55 s, restarts 486-499 s); the Modal test "drift-long" allows 3,300 s.
DRIFT_LONG_SECONDS = 3200


def drift_long_serving():
    """``(reference, candidates)`` server_env blocks from the recorded arms' variants (identical to the recorded
    kernels-v2 and hero-6x64k server blocks)."""
    base = json.loads((ROOT / "configs/arena_30m.json").read_text())
    kernels = apply_variant(json.loads(json.dumps(base)), "fixed-2x-mtp0-wide-kernels")["server"]
    hero = apply_variant(json.loads(json.dumps(base)), "fixed-2x-mtp0-wide-kv8-hero")["server"]
    if kernels.get("serving_kernels") or kernels.get("stack_kernels") or kernels["mtp_tokens"] or hero["mtp_tokens"]:
        raise ValueError("The drift test expects the MTP-0 kernels-v2 arm without serving or stack kernels")
    reference = {"ARENA_DENSE_KERNELS": json.dumps(kernels["dense_kernels"]), "ARENA_SERVING_KERNELS": "off",
                 "ARENA_STACK_KERNELS": "off"}
    candidates = {
        "dl-hero": {"ARENA_DENSE_KERNELS": json.dumps(hero["dense_kernels"]),
                    "ARENA_SERVING_KERNELS": json.dumps(hero["serving_kernels"]),
                    "ARENA_STACK_KERNELS": json.dumps(hero["stack_kernels"])},
        "dl-kv8": {"ARENA_DENSE_KERNELS": json.dumps(kernels["dense_kernels"]),
                   "ARENA_SERVING_KERNELS": json.dumps(DRIFT_LONG_KV8_ONLY), "ARENA_STACK_KERNELS": "off"},
    }
    return reference, candidates


def drift_long_study(study, output, inputs):
    """``dl-ref`` (kernels-v2 serving), then ``dl-hero`` and ``dl-kv8`` as report-only gates (arena.kernel_gate)."""
    study = json.loads(json.dumps(study))
    target = output / "configs/kernel-quality"
    target.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, source in (("prompts.json", Path(inputs) / "quality/prompts.json"),
                         ("gsm8k.jsonl", ROOT / KERNEL_QUALITY_INPUTS / "gsm8k.jsonl")):
        shutil.copyfile(source, target / name)
        digests[name] = hashlib.sha256((target / name).read_bytes()).hexdigest()
    prompts = json.loads((target / "prompts.json").read_text())
    if prompts.get("greedy") or not prompts.get("teacher_forced"):
        raise ValueError("The drift inputs are teacher-forced sequences only (scripts/drift_long_inputs.py)")
    pinned = {"ARENA_KV_CACHE_MEMORY_BYTES": str(EXPERT_QUANT_KV_BYTES)}
    reference, candidates = drift_long_serving()
    study["server"].update(mtp_tokens=0)
    study["trials"] = [{"name": "dl-ref", "kind": "kernel-quality", "role": "reference", "competition": False,
                        "wall_seconds": 600, "server_env": {**pinned, **reference}, "quality": dict(DRIFT_LONG_QUALITY)}]
    for name, env in candidates.items():
        study["trials"].append({"name": name, "kind": "kernel-quality", "role": "gate", "reference": "dl-ref",
                                "competition": False, "wall_seconds": 400, "server_env": {**pinned, **env},
                                "quality": dict(DRIFT_LONG_QUALITY)})
    study["drift_long"] = {"kv_cache_memory_bytes": EXPERT_QUANT_KV_BYTES, "quality": dict(DRIFT_LONG_QUALITY),
                           "reference": reference, "candidates": candidates,
                           "teacher_forced": len(prompts["teacher_forced"]),
                           "positions": sum(len(t["ids"]) - 1 for t in prompts["teacher_forced"]),
                           "inputs": {"folder": "configs/kernel-quality", "source": str(inputs), "sha256": digests}}
    study.update(wall_seconds=60, notebook_seconds=DRIFT_LONG_SECONDS - 60, kernel_timeout=DRIFT_LONG_SECONDS,
                 harness_selection="long-context drift test (no agent)")
    return study


# 4-bit attention KV drift test (docs/validation-20260927/kv-cache-4bit.md): hero-v2's serving
# (prune12, serving tweaks, GPTQ-ARC overlay, Marlin experts, int8 QSA KV; MTP-0) as the reference with its own
# repeat as the noise floor, then the same serving with only the QSA KV mode changed (int4, then k8v4) as report-only
# gates. Inputs: the held-out ARC requests (expert-quant inputs-v2: 10 teacher-forced sequences of 8.7-24K tokens,
# 10 greedy prompts) and the long-context drift sequences (7 of 39-44K tokens). Greedy on, GSM8K off; the KV pool
# pinned as in the expert-quant check (prompt_logprobs transients); a short decode bench on each server.
KV4_DRIFT_MODES = ("int4", "k8v4")
KV4_DRIFT_QUALITY = {"greedy": True, "gsm8k": False, "report_only": True}
KV4_DRIFT_BENCH = {"prompt_tokens": [], "concurrency": [], "output_tokens": 256, "shared_prefix": None,
                   "transcript": None, "points": [[1024, 1], [32768, 6]]}
KV4_DRIFT_SERVED_BLOCKS = 1335  # hero-v2 smoke v2 (hero-v2.md)
# Test A (three starts, 299K teacher-forced positions twice on the reference, once per gate) took 1,850 s; this adds
# the prune/overlay load (size check only), 161K more positions per pass, greedy passes and a short bench per server:
# ~2,300 s expected. The Modal test "kv4-drift" allows 3,000 s.
KV4_DRIFT_SECONDS = 2940


def kv4_drift_prompts(sources):
    """One kernel-quality prompts file from several input folders (``<folder>/quality/prompts.json``): indices
    renumbered by source (source k from 1000 k), ``source_folder`` recorded on each entry."""
    merged = {"greedy": [], "teacher_forced": []}
    for k, folder in enumerate(sources):
        data = json.loads((Path(folder) / "quality/prompts.json").read_text())
        for key in merged:
            for entry in data.get(key) or []:
                merged[key].append({**entry, "index": 1000 * k + int(entry["index"]), "source_index": entry["index"],
                                    "source_folder": str(folder)})
    if not merged["teacher_forced"] or not all(t.get("ids") for t in merged["teacher_forced"]):
        raise ValueError("The kv4 drift inputs are token-id sequences (expert-quant / drift-long inputs)")
    if any("prompt_ids" not in g for g in merged["greedy"]):
        raise ValueError("The kv4 drift greedy prompts are token ids")
    return merged


def kv4_drift_serving():
    """(hero-v2 server block, {trial: ARENA_SERVING_KERNELS}) for the reference and the 4-bit candidates."""
    from arena.hero_v2 import BASE, compose

    base = json.loads((ROOT / "configs/arena_30m.json").read_text())
    server = compose(apply_variant(base, BASE), {}, served_blocks=KV4_DRIFT_SERVED_BLOCKS)["server"]
    kernels = server["serving_kernels"]
    if server["mtp_tokens"] != 0 or kernels.get("kv_cache_8bit") != "int8" or not server.get("expert_overlay") \
            or not server.get("expert_prune") or not (kernels.get("ple_uva") and kernels.get("embed_host")):
        raise ValueError("The kv4 drift test expects the hero-v2 serving (MTP-0, int8 KV, overlay, prune, tweaks)")
    candidates = {}
    for mode in KV4_DRIFT_MODES:
        spec = {k: v for k, v in kernels.items() if k != "kv_cache_8bit"}
        candidates[f"kv4-{mode}"] = {**spec, "kv_cache_4bit": mode}
    return server, candidates


def kv4_drift_study(study, output, sources):
    """``kv4-ref`` (hero-v2 serving, int8 KV), then ``kv4-int4`` and ``kv4-k8v4`` (report-only gates)."""
    study = json.loads(json.dumps(study))
    sources = [Path(s) for s in (sources.split(",") if isinstance(sources, str) else sources)]
    target = output / "configs/kernel-quality"
    target.mkdir(parents=True, exist_ok=True)
    prompts = kv4_drift_prompts(sources)
    (target / "prompts.json").write_text(json.dumps(prompts))
    shutil.copyfile(ROOT / KERNEL_QUALITY_INPUTS / "gsm8k.jsonl", target / "gsm8k.jsonl")
    digests = {name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in ("prompts.json", "gsm8k.jsonl")}
    server, candidates = kv4_drift_serving()
    # The overlay's shard sizes are checked at each of the three starts instead of their sha256 (~190 s per start,
    # hero-v2 smoke): the same shards the hero-v2 runs verified on this volume; the served weights are unchanged.
    server["expert_overlay"] = {**server["expert_overlay"], "check": "size"}
    study["server"] = server
    shared = {"ARENA_KV_CACHE_MEMORY_BYTES": str(EXPERT_QUANT_KV_BYTES), "ARENA_READY_GUARD": HERO_V2_READY_GUARD,
              "ARENA_STALL_DUMP": "90"}
    study["trials"] = [{"name": "kv4-ref", "kind": "kernel-quality", "role": "reference", "competition": False,
                        "wall_seconds": 900, "server_env": dict(shared), "quality": dict(KV4_DRIFT_QUALITY),
                        "bench": dict(KV4_DRIFT_BENCH)}]
    for name, kernels in candidates.items():
        # v1: a 600 s gate wall (+ the bootstrap's 300 s margin) skipped the k8v4 gate with ~530 s left; the int4
        # gate's measurements took ~260 s.
        study["trials"].append({"name": name, "kind": "kernel-quality", "role": "gate", "reference": "kv4-ref",
                                "competition": False, "wall_seconds": 360,
                                "server_env": {**shared, "ARENA_SERVING_KERNELS": json.dumps(kernels)},
                                "quality": dict(KV4_DRIFT_QUALITY), "bench": dict(KV4_DRIFT_BENCH)})
    study["kv4_drift"] = {
        "kv_cache_memory_bytes": EXPERT_QUANT_KV_BYTES, "quality": dict(KV4_DRIFT_QUALITY),
        "bench": dict(KV4_DRIFT_BENCH), "reference_serving_kernels": server["serving_kernels"],
        "candidates": candidates, "served_blocks": KV4_DRIFT_SERVED_BLOCKS,
        "pass_bar": "per context bin: KL and top-1 disagreement <= 1.25 x the reference repeat's; recorded-token "
                    "NLL change CI not clearly positive",
        "teacher_forced": len(prompts["teacher_forced"]), "greedy": len(prompts["greedy"]),
        "positions": sum(len(t["ids"]) - 1 for t in prompts["teacher_forced"]),
        "inputs": {"folder": "configs/kernel-quality", "sources": [str(s) for s in sources], "sha256": digests}}
    study.update(wall_seconds=60, notebook_seconds=KV4_DRIFT_SECONDS - 60, kernel_timeout=KV4_DRIFT_SECONDS,
                 harness_selection="4-bit attention KV drift test (no agent)")
    return study


# kv8 prefill check (kernel work, phase 7; <=; docs/validation-20260927/kernels-phase7.md):
# hero-v2's serving (int8 QSA KV) as the reference with its own repeat as the noise floor, then the same serving with
# serving_kernels.kv8_prefill = "bf16" as a report-only gate (the 1.25 x bar per context bin, as the kv4 drift test).
# Teacher-forced scoring prefills whole sequences, so it runs the changed path; greedy is off (decode is unchanged).
# The same pinned KV pool as the kv4 test (prompt_logprobs transients); on it, the injected-prefill bench at 14 / 21 /
# 23 streams (the recorded phase-2 prefill shapes) on both servers. The 23 x 64K fit comes from each start's logged
# peak activation (the option changes no cache page or flag; tests/test_kv8_prefill_plumbing.py).
KV8P_QUALITY = {"greedy": False, "gsm8k": False, "report_only": True}
# v2 shapes (v1: 14 x 24K / 21 x 16K / 23 x 16K with 20K / 12K cached prefixes overflowed the pinned 8 GB pool,
# 341 blocks of 1,515 tokens at the gate: every injection re-prefilled most of its prefix, ~6K prefill tok/s against
# the arena's 1.6-2.5K, a 64% "tax"). Here streams and injector together use <= 50% of the gate capacity, so the shared
# prefix stays cached and each injection prefills only its tail (the arena's ~5K uncached tokens per request).
KV8P_POOL_BLOCKS, KV8P_BLOCK_TOKENS = 341, 1515
# measure at 23 and 30 streams (run 2: 30 streams, max_num_seqs 32) with hero-v2's phase-2
# prefill load, ~1,170 uncached tok/s (hero-v2-analysis/throughput.md): 0.23 requests/s of 5,100 fresh tokens.
KV8P_SEQUENCES = 32
KV8P_INJECT = {"prefix_tokens": 8192, "tail_tokens": 5100, "rate": 0.23}
KV8P_BENCH = {"prompt_tokens": [], "concurrency": [], "output_tokens": 512, "shared_prefix": None, "transcript": None,
              "points": [],
              "mixed_points": [{"prompt_tokens": 8192, "concurrency": 23, "output_tokens": 512, "inject": KV8P_INJECT},
                               {"prompt_tokens": 6144, "concurrency": 30, "output_tokens": 512, "inject": KV8P_INJECT}]}
# Phase estimates (seconds) from v1's measured phases: setup 510 s ready + ~155 s; teacher-forced 170 s and its repeat
# 114 s; restart 370 s. v1's bench points took 163-191 s (prefill 33-37, decode 47-52 for 1,024 tokens, mixed 82-102
# with the pool-overflow re-prefills); v2's points prefill a third of the tokens and force 512 tokens: ~90 s each.
KV8P_PHASES = {"setup": 665, "restart": 370, "run": {"kv8p-ref": 170 + 114 + 180, "kv8p-bf16": 170 + 180}}
KV8P_SECONDS = 2340
KV8P_WALLS = {"reference": 60, "gate": 60}


def kv8_prefill_study(study, output, sources):
    """``kv8p-ref`` (hero-v2 serving), then ``kv8p-bf16`` (the same with kv8_prefill "bf16"), report-only gate."""
    study = json.loads(json.dumps(study))
    sources = [Path(s) for s in (sources.split(",") if isinstance(sources, str) else sources)]
    target = output / "configs/kernel-quality"
    target.mkdir(parents=True, exist_ok=True)
    prompts = kv4_drift_prompts(sources)
    prompts["greedy"] = []
    (target / "prompts.json").write_text(json.dumps(prompts))
    shutil.copyfile(ROOT / KERNEL_QUALITY_INPUTS / "gsm8k.jsonl", target / "gsm8k.jsonl")
    digests = {name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in ("prompts.json", "gsm8k.jsonl")}
    server, _ = kv4_drift_serving()
    server["expert_overlay"] = {**server["expert_overlay"], "check": "size"}
    study["server"] = server
    candidate = {**server["serving_kernels"], "kv8_prefill": "bf16"}
    from arc_infra.serving_kernels_patch import validate as validate_serving_kernels

    if validate_serving_kernels(candidate) != candidate:
        raise ValueError("kv8_prefill candidate is not a canonical serving_kernels setting")
    shared = {"ARENA_KV_CACHE_MEMORY_BYTES": str(EXPERT_QUANT_KV_BYTES), "ARENA_READY_GUARD": HERO_V2_READY_GUARD,
              "ARENA_STALL_DUMP": "90"}
    study["trials"] = [
        {"name": "kv8p-ref", "kind": "kernel-quality", "role": "reference", "competition": False,
         "wall_seconds": KV8P_WALLS["reference"], "server_env": dict(shared), "quality": dict(KV8P_QUALITY),
         "bench": json.loads(json.dumps(KV8P_BENCH))},
        {"name": "kv8p-bf16", "kind": "kernel-quality", "role": "gate", "reference": "kv8p-ref", "competition": False,
         "wall_seconds": KV8P_WALLS["gate"],
         "server_env": {**shared, "ARENA_SERVING_KERNELS": json.dumps(candidate)},
         "quality": dict(KV8P_QUALITY), "bench": json.loads(json.dumps(KV8P_BENCH))}]
    study["server"]["max_num_seqs"] = KV8P_SEQUENCES
    study["generation_slots"] = KV8P_SEQUENCES  # the bench's gate slots: 30 streams + the injector
    study["kv8_prefill_check"] = {
        "kv_cache_memory_bytes": EXPERT_QUANT_KV_BYTES, "quality": dict(KV8P_QUALITY), "bench": KV8P_BENCH,
        "reference_serving_kernels": server["serving_kernels"], "candidate": candidate,
        "served_blocks": KV4_DRIFT_SERVED_BLOCKS,
        "pass_bar": "per context bin: KL and top-1 disagreement <= 1.25 x the reference repeat's; recorded-token "
                    "NLL change CI not clearly positive",
        "fit": "23 x 64K on hero-v2's measured 1,335 blocks, less the peak-activation difference between the starts",
        "teacher_forced": len(prompts["teacher_forced"]),
        "positions": sum(len(t["ids"]) - 1 for t in prompts["teacher_forced"]),
        "inputs": {"folder": "configs/kernel-quality", "sources": [str(s) for s in sources], "sha256": digests}}
    study.update(wall_seconds=60, notebook_seconds=KV8P_SECONDS, kernel_timeout=KV8P_SECONDS + 50,
                 harness_selection="kv8 prefill path check (no agent)")
    # Every trial starts and finishes with >= 15% margin on every phase, or the build fails (arena.timeline).
    from arena.timeline import check as check_timeline

    study["kv8_prefill_check"]["timeline"] = check_timeline(
        study, setup_seconds=KV8P_PHASES["setup"], restart_seconds=KV8P_PHASES["restart"],
        run_seconds=KV8P_PHASES["run"])
    return study


# Phase 3 (29 September; expert-quant.md section 6): the stage-2 recipe on the 4x calibration set (inputs-v2: 2.90M
# tokens, v1's held-out set), every 8th row, every expert quantized with a pooled-Hessian prior of 4,096 rows (the CPU
# study's best at 21.9K rows was 1,024-4,096) instead of the < 64-row fallback, act-order with static scales. The
# Modal test expert-quant-calibrate-v2 gives it 2,400 s.
EXPERT_QUANT_V2 = {"prior_rows": 4096, "act_order": True, "calib_stride": 8, "seconds": 2400,
                   "gpu_reserve_seconds": 1100,
                   "overlay_dir": "/assets/flash-input/datasets/richardcsaky/arc-expert-gptq-arc-v2"}


def expert_quant_study(study, output, inputs, phase, *, overrides=None, prune_variant=None):
    """One ``expert-quant`` trial (``arena.expert_quant``) on the capture overlay; no games are played. ``overrides``
    replace phase settings (e.g. ``EXPERT_QUANT_V2``); ``prune_variant`` serves the capture with that variant's
    load-time pruning mask, so the Hessians see the pruned routing (mask-aware calibration)."""
    if phase not in EXPERT_QUANT_PHASES:
        raise ValueError(f"--expert-quant-phase must be one of {sorted(EXPERT_QUANT_PHASES)}")
    spec = json.loads(json.dumps(EXPERT_QUANT_PHASES[phase]))
    overrides = dict(overrides or {})
    if "seconds" in overrides:
        spec["seconds"] = int(overrides.pop("seconds"))
    unknown = set(overrides) - set(spec["settings"]) - {"prior_rows", "act_order", "omit_pruned"}
    if unknown:
        raise ValueError(f"unknown expert-quant settings {sorted(unknown)}")
    spec["settings"].update(overrides)
    study = json.loads(json.dumps(study))
    target = output / "configs/expert-quant"
    target.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in spec["inputs"]:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(inputs) / name, target / name)
        digests[name] = hashlib.sha256((target / name).read_bytes()).hexdigest()
    selection = json.loads((target / "selection.json").read_text())
    study["server"].update(mtp_tokens=0, expert_capture=True)
    if prune_variant is not None:
        from arena.variants import EXPERT_PRUNE_VARIANTS

        if phase != "calibrate" or prune_variant not in EXPERT_PRUNE_VARIANTS:
            raise ValueError("mask-aware calibration takes the calibrate phase and a pruned hero variant")
        mask = EXPERT_PRUNE_VARIANTS[prune_variant]
        study["server"]["expert_prune"] = {"mask": mask["mask"], "sha256": mask["sha256"]}
    study["trials"] = [{"name": "expert-quant", "kind": "expert-quant", "phase": phase, "competition": False,
                        "wall_seconds": 60}]
    study["expert_quant"] = {**spec["settings"], "phase": phase,
                             "inputs": {"folder": "configs/expert-quant", "source": str(inputs), "sha256": digests,
                                        "eval_families": selection["eval_families"],
                                        "holdout_excluded": selection["holdout_excluded"]}}
    study.update(wall_seconds=60, notebook_seconds=spec["seconds"] - 60, kernel_timeout=spec["seconds"],
                 harness_selection="expert quantization (no agent)")
    return study


# Load-time expert pruning GPU check (docs/validation-20260927/expert-usage.md section 5):
# the hero serving three times in one container (unpruned reference + repeat, then the 12.5% and 25% pruned
# overlays), CUDA graphs on, an explicit 7.5 GiB KV pool (the prompt_logprobs transients), the v4 held-out set.
EXPERT_PRUNE_CHECK_GUARD = "180,900"
# More pruned levels ("measure more pruned experts"), in run order, the most informative first:
# the layer-aware 25% (same 6,144 experts and memory as the uniform 25%, 64..192 per layer, layers 0-2 at 64), the
# uniform 25%, then the uniform 18.75% fallback. Masks from scripts/expert_prune_masks.py; each candidate server
# carries the stall-only stack dump (90 s per step).
EXPERT_PRUNE_LEVELS = (
    {"trial": "prune-25-layer", "fraction": 0.25, "source": "data/expert_prune/prune25-layer.json",
     "mask": "configs/expert-prune/prune25-layer.json",
     "sha256": "6ea3dfd2373f64d09f71eea1dd3939011100917226368f61a31ee07eb4cca8cc"},
    {"trial": "prune-25", "fraction": 0.25, "source": "data/expert_prune/prune25.json",
     "mask": "configs/expert-prune/prune25.json",
     "sha256": "1aea34ddaacea5bbad339323259a24a164e9c3033e6d34790ead2bac05969a09"},
    {"trial": "prune-18", "fraction": 0.1875, "source": "data/expert_prune/prune18.json",
     "mask": "configs/expert-prune/prune18.json",
     "sha256": "7b4e15277715adc129760920523961f4abf9c02ee41d15938b5e5ea65749a8ff"},
)
EXPERT_PRUNE_LEVELS_STALL_SECONDS = 90
# The plain (non-levels) check keeps its two served variants; later-registered masks do not join it.
EXPERT_PRUNE_CHECK_VARIANTS = ("fixed-2x-mtp0-wide-kv8-hero-prune12", "fixed-2x-mtp0-wide-kv8-hero-prune25")


def expert_prune_check_study(study, output, inputs, levels=False):
    """``prune-quality`` trials (arena.expert_usage.run_prune_check) on the hero variant; no games are played.
    ``levels``: the EXPERT_PRUNE_LEVELS candidates (with the stall-only dump) instead of the two served variants."""
    from arena.expert_usage import SETTINGS
    from arena.variants import EXPERT_PRUNE_VARIANTS

    study = json.loads(json.dumps(study))
    if study["server"].get("mtp_tokens") != 0 or study["server"].get("expert_prune"):
        raise ValueError("The expert-prune check needs the unpruned MTP-0 hero serving")
    target = output / "configs/expert-usage"
    target.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in EXPERT_USAGE_FILES:
        shutil.copyfile(Path(inputs) / name, target / name)
        digests[name] = hashlib.sha256((target / name).read_bytes()).hexdigest()
    env = {"ARENA_KV_CACHE_MEMORY_BYTES": str(EXPERT_USAGE_KV_BYTES), "ARENA_READY_GUARD": EXPERT_PRUNE_CHECK_GUARD}
    trials = [{"name": "prune-ref", "kind": "prune-quality", "role": "reference", "competition": False,
               "wall_seconds": 480, "server_env": dict(env)}]
    candidates = ({spec["trial"]: spec for spec in EXPERT_PRUNE_LEVELS} if levels else
                  {f"prune-{name.rsplit('prune', 1)[1]}": spec for name, spec in EXPERT_PRUNE_VARIANTS.items()
                   if name in EXPERT_PRUNE_CHECK_VARIANTS})
    extra = {"ARENA_STALL_DUMP": str(EXPERT_PRUNE_LEVELS_STALL_SECONDS)} if levels else {}
    for trial, spec in candidates.items():
        mask = output / spec["mask"]
        mask.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / spec["source"], mask)
        if hashlib.sha256(mask.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"{spec['source']}: sha256 differs from the pinned mask")
        trials.append({"name": trial, "kind": "prune-quality", "role": "candidate",
                       "reference": "prune-ref", "competition": False, "wall_seconds": 300,
                       "server_env": {**env, **extra, "ARENA_EXPERT_PRUNE": json.dumps(
                           {"mask": spec["mask"], "sha256": spec["sha256"]}, separators=(",", ":"))}})
    study["trials"] = trials
    study["start_in_trial_loop"] = True
    study["expert_prune_check"] = {
        "settings": {k: SETTINGS[k] for k in ("heldout_prompt_logprobs", "reference_concurrency",
                                              "repeat_concurrency", "mask_concurrency")},
        "kv_cache_memory_bytes": EXPERT_USAGE_KV_BYTES, "ready_guard": EXPERT_PRUNE_CHECK_GUARD,
        "variants": {k: {kk: v[kk] for kk in ("fraction", "mask", "sha256")} for k, v in candidates.items()},
        "inputs": {"folder": "configs/expert-usage", "source": str(inputs), "sha256": digests}}
    if levels:
        study["expert_prune_check"].update(levels=True, stall_dump_seconds=EXPERT_PRUNE_LEVELS_STALL_SECONDS,
                                           order=list(candidates))
        # v1 timeline: setup + first start 562 s, reference + repeat 186 s, then ~400-540 s per start and ~90 s
        # per candidate pass: 562 + 186 + 3 x ~590 = ~2,520 s; 3,000 s leaves one slow start of margin.
        study.update(wall_seconds=120, notebook_seconds=2940, kernel_timeout=3000,
                     harness_selection="expert-prune levels check (no agent)")
        return study
    # Setup ~180 s; three server starts (~400 s each, graphs on); passes: reference + repeat ~210 s, 2 x ~100 s.
    study.update(wall_seconds=120, notebook_seconds=2040, kernel_timeout=2100,
                 harness_selection="expert-prune GPU check (no agent)")
    return study


# The prune12 c110 hang (29 September; docs/validation-20260927/expert-usage.md section 6): the pruned hero's KV pool
# is 1,244 blocks against the unpruned hero's 860, and the hung step was handing out block ids up to 1,159, a
# range no earlier Flash run reached. Two serving-bench arms on the prune12 server sweep the whole free-block queue
# (4 points x 25 distinct 32K prompts, each prefilled then decoded 512 tokens: ~540 fresh blocks per point, so
# ~2,200 allocations cycle every id); (a) the served 1,244-block pool, (b) the pool pinned to the unpruned hero's
# 860 blocks (18,229,248,000 bytes, hero-21x42k cache-layout.json). A hang in (a) but not (b) points at block ids
# above ~860 (not the 448-expert model); both hang or neither: the pool is not it. A worker step that runs 90 s
# dumps every thread's stack once (a hung worker's wait). v1 dumped every 120 s on a timer and the dump itself
# crashed the busy worker (SIGSEGV) in both arms; v2 dumps only on a stalled step.
EXPERT_PRUNE_HANG_GRID = {"prompt_tokens": [], "concurrency": [], "output_tokens": 512, "shared_prefix": None,
                          "transcript": None, "points": [[32768, 25]] * 4}
HERO_KV_POOL_BYTES = 18229248000
EXPERT_PRUNE_HANG_STALL_SECONDS = 90


def expert_prune_hang_study(study):
    """Two serving-bench arms on the prune12 hero server (the served pool, then the unpruned hero's pool)."""
    study = json.loads(json.dumps(study))
    if not study["server"].get("expert_prune") or study["server"].get("mtp_tokens") != 0:
        raise ValueError("The prune hang test needs the pruned MTP-0 hero serving")
    env = {"ARENA_READY_GUARD": EXPERT_PRUNE_CHECK_GUARD, "ARENA_STALL_DUMP": str(EXPERT_PRUNE_HANG_STALL_SECONDS)}
    study["trials"] = [
        {"name": "hang-pool1244", "kind": "serving-bench", "competition": False, "wall_seconds": 60,
         "server_env": dict(env), "throughput_bench": dict(EXPERT_PRUNE_HANG_GRID)},
        {"name": "hang-pool860", "kind": "serving-bench", "competition": False, "wall_seconds": 60,
         "server_env": {**env, "ARENA_KV_CACHE_MEMORY_BYTES": str(HERO_KV_POOL_BYTES)},
         "throughput_bench": dict(EXPERT_PRUNE_HANG_GRID)},
    ]
    study["throughput_bench"] = dict(EXPERT_PRUNE_HANG_GRID)
    study["start_in_trial_loop"] = True
    study["expert_prune_hang_test"] = {"grid": dict(EXPERT_PRUNE_HANG_GRID), "pinned_pool_bytes": HERO_KV_POOL_BYTES,
                                       "stall_dump_seconds": EXPERT_PRUNE_HANG_STALL_SECONDS,
                                       "stall_dump_mode": "per-step one-shot"}
    # Setup ~180 s; two server starts (~540 s, ~400 s); four points per arm (~90 s prefill + ~30 s decode each).
    study.update(wall_seconds=60, notebook_seconds=2340, kernel_timeout=2400,
                 harness_selection="expert-prune hang test (no agent)")
    return study


# hero-v2 smoke (29 September; docs/validation-20260927/hero-v2.md): one server start with every chosen component on
# the hero-v2 GPU slot. Trial 1 (serving bench, no agent): served layout and focus fit, greedy sanity, teacher-forced
# drift against the recorded hero smoke (same token ids; the hero's own repeat noise is in that record) and one
# full-load point of every focus window at once (64,512-token prompts at the stream count). Trial 2: a short soak on
# the 2 h run's clone roster (real arena traffic: images, request churn, 25-32 running) with the stall-only dump on;
# the opening switch is forced at HERO_V2_SMOKE_SWITCH_SECONDS so that phase 2 runs too (the prune12 hour hung 80 s
# after its switch, ~20 minutes into play).
HERO_V2_REFERENCE_NAME = "configs/kernel-quality/reference-hero-v1.json"
HERO_V2_SMOKE_SWITCH_SECONDS = 540
HERO_V2_SMOKE_CLEANUP_SECONDS = 240
HERO_V2_READY_GUARD = "180,900"  # give up 180 s after graph capture or 900 s after launch (arena.serving.ready_guard)
# MTP-1 (30 September): the draft load and the 2x verify graphs lengthen the start; the spec-sweep pilots' 1,200 s total.
HERO_V2_MTP_READY_GUARD = "180,1200"
# 3-bit experts on Humming (30 September, expert-3bit.md): Humming JIT-compiles its MoE kernels (nvcc) during the
# profile run and graph capture, on top of the MTP start, with no measured start yet: 420 s after capture and the
# pinned recipe's own 1,800 s setup ceiling (serving_setup.SETUP_TOTAL_TIMEOUT, which binds first) as the total.
HERO_V2_HUMMING_READY_GUARD = "420,1800"
COMPETITION_SETUP_TIMEOUTS = "2700,3600"  # competition int3: pinned server-ready / setup totals
HERO_V2_COMPETITION_READY_GUARD = "180,3000"  # Kaggle: ~15 min weight load from the dataset mounts


def hero_v2_reference(output, reference):
    """The hero smoke's recorded drift rows (``trials/hero/kv8-quality.json``) as a kernel-quality reference file."""
    path = Path(reference)
    recorded = json.loads(path.read_text())
    record = json.loads((path.parents[1] / "study.json").read_text())
    trial = next(t for t in record["trials"] if t["trial"]["name"] == path.parent.name)
    server = trial["report"]["server"]
    if (server.get("mtp_tokens") != 0 or (server.get("serving_kernels") or {}).get("moe_backend") != "marlin"
            or not server.get("dense_kernels") or not recorded.get("teacher_forced", {}).get("rows")):
        raise ValueError("The hero-v2 drift reference is the hero smoke's kv8-quality.json (MTP-0, Marlin experts)")
    trimmed = {"role": "gate", "trial": {**trial["trial"], "record": str(path)},
               "server_env": trial["trial"].get("server_env") or {}, "server": server,
               "greedy": recorded["greedy"], "teacher_forced": recorded["teacher_forced"],
               "recorded_summary": recorded.get("summary")}
    target = output / "configs/kernel-quality"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("prompts.json", "gsm8k.jsonl"):
        shutil.copyfile(ROOT / KERNEL_QUALITY_INPUTS / name, target / name)
    (output / HERO_V2_REFERENCE_NAME).write_text(json.dumps(trimmed))
    return {"reference": HERO_V2_REFERENCE_NAME, "repeat_teacher": True,
            "reference_sha256": hashlib.sha256((output / HERO_V2_REFERENCE_NAME).read_bytes()).hexdigest(),
            "inputs_sha256": hashlib.sha256((target / "prompts.json").read_bytes()).hexdigest(),
            "recorded": {"vs_phase2_reference": (recorded.get("summary") or {}).get("vs_reference", {})
                         .get("teacher_forced"),
                         "self_repeat": (recorded.get("summary") or {}).get("self_repeat", {}).get("teacher_forced")}}


def hero_v2_smoke_study(study, output, reference):
    study = json.loads(json.dumps(study))
    clones = study.get("competition") or {}
    if not study.get("hero_v2") or not clones.get("research_clones") or reference is None:
        raise ValueError("The hero-v2 smoke builds on a hero-v2 clone-roster study with --hero-v2-reference")
    soak = int(study["wall_seconds"])
    if not 300 <= soak <= 1200:
        raise ValueError("The hero-v2 smoke soak (--wall-seconds) is 300..1200 s")
    active = int(study["active_limit"])
    study["throughput_bench"] = {"prompt_tokens": [], "concurrency": [], "output_tokens": 512, "shared_prefix": None,
                                 "transcript": None, "points": [[64512, active]],
                                 "sanity": KERNEL_SMOKE_GRID["sanity"], "quality": hero_v2_reference(output, reference)}
    study["trials"] = [{"name": "hero-v2-check", "kind": "serving-bench", "competition": False, "wall_seconds": 60},
                       {"name": "duck-transfer", "kind": "duck-transfer"}]
    switch = min(HERO_V2_SMOKE_SWITCH_SECONDS, soak // 2)
    study["opening_max_wall_seconds"] = switch
    study["opening_cap_absolute"] = True
    study["competition"]["cleanup_seconds"] = HERO_V2_SMOKE_CLEANUP_SECONDS
    study["hero_v2"]["smoke"] = {
        "check": "served layout / focus fit, greedy sanity, teacher-forced drift vs the hero smoke, "
                 f"{active} x 64,512-token prompts at once",
        "soak_seconds": soak, "opening_switch_forced_at_seconds": switch,
        "cleanup_seconds": HERO_V2_SMOKE_CLEANUP_SECONDS, "roster": study["roster"],
        "differs_from_the_run": ["wall_seconds", "opening_max_wall_seconds", "opening_cap_absolute",
                                 "competition.cleanup_seconds", "trials", "throughput_bench"]}
    # ~705 s to the first trial (the hero hours: ~515 s; the overlay sha256 check ~190 s), check ~200 s, soak, cleanup.
    study.update(notebook_seconds=2400, kernel_timeout=2460)
    return study


# Candidate hero-v2 smoke (Holo4-27B, 29 September; docs/validation-20260927/holo4-27b.md), one container on the test
# slot, three server starts:
# 1. ``kv-bf16-ref`` (kernel-quality reference): the candidate's serving with BF16 KV on the same attention backend,
#    the KV pool pinned for the prompt_logprobs transients; teacher forcing on the held-out ARC requests and its repeat
#    at another concurrency (the noise floor);
# 2. ``kv-fp8`` (report-only gate): the same with the served FP8 KV, on the reference's token ids;
# 3. ``holo-check`` (serving bench) on the arena's own server (FP8 KV, the full pool at 0.96): served layout and focus
#    fit, greedy sanity, the tool_choice probe, prefix cache v2 warm vs cold on a recorded transcript (exact cached
#    tokens), and every focus window at once (64,512-token prompts);
# 4. ``duck-transfer``: a short soak on the run's 110-clone roster on that server (the compaction preflight runs at its
#    start), the opening switch forced halfway.
# Pass bar for the drift (stated before the run): per context bin, KL and top-1 disagreement of FP8 against BF16 at
# most 1.25 x the BF16 repeat's, recorded-token NLL change not clearly positive (scripts/kv4_drift_analyze.py).
CANDIDATE_SMOKE_KV_BYTES = 20 * 2**30  # one 262,144-token BF16 request needs 16.32 GiB (smoke v1 refused 16 GiB)
CANDIDATE_SMOKE_QUALITY = {"greedy": False, "gsm8k": False, "report_only": True}
CANDIDATE_SMOKE_CLEANUP_SECONDS = 240
CANDIDATE_SMOKE_GRID = {"prompt_tokens": [], "concurrency": [], "output_tokens": 64, "shared_prefix": None,
                        "sanity": KERNEL_SMOKE_GRID["sanity"], "tool_choice_probe": True,
                        "prefix_check": dict(PREFIX_SMOKE_GRID["prefix_check"])}
# Setup ~150 s, the reference start ~450 s and passes ~250 s, two restarts ~250 s each, the gate ~120 s, the check
# ~250 s, the soak and its cleanup; the Modal test "holo-smoke" allows 2,700 s (the soak must start by ~1,620 s).
CANDIDATE_SMOKE_SECONDS = 2700


# Candidate hero-v2 soak (Holo4-27B, "soak first, then decide on the hour"): one start of the built
# arena server (no restarts), then a short clone-roster arena at full streams with exact cached-token usage; the
# compaction preflight runs at the trial start and the opening switch is forced halfway so phase 2 (every focus stream)
# runs too. Setup to the first trial measured 604 s (holo4 smoke v2: agent venv, runtime extraction, a cold compile);
# the bootstrap starts the trial only while (function timeout - 60 - 120 - 300 - wall) s remain, so the timeout is
# sized for that start plus CANDIDATE_SOAK_SETUP_MARGIN.
CANDIDATE_SOAK_SETUP_SECONDS = 604
CANDIDATE_SOAK_SETUP_MARGIN = 116
CANDIDATE_SOAK_CLEANUP_SECONDS = 240
BOOTSTRAP_RESERVE_SECONDS = 60 + 120 + 300  # Modal max_seconds, the bootstrap's budget reserve, its trial margin


def candidate_soak_seconds(wall):
    """The Modal timeout that lets the soak start after the measured setup plus its margin."""
    return BOOTSTRAP_RESERVE_SECONDS + int(wall) + CANDIDATE_SOAK_SETUP_SECONDS + CANDIDATE_SOAK_SETUP_MARGIN


def candidate_soak_study(study):
    study = json.loads(json.dumps(study))
    clones = study.get("competition") or {}
    if not study.get("hero_v2") or not study.get("candidate_profile") or not clones.get("research_clones"):
        raise ValueError("The candidate soak builds on a candidate hero-v2 clone-roster study")
    wall = int(study["wall_seconds"])
    if not 600 <= wall <= 900:
        raise ValueError("The candidate soak (--wall-seconds) is 600..900 s")
    study["trials"] = [{"name": "duck-transfer", "kind": "duck-transfer",
                        "server_env": {"ARENA_PROMPT_TOKENS_DETAILS": "1"}}]
    switch = wall // 2
    study["opening_max_wall_seconds"] = switch
    study["opening_cap_absolute"] = True
    study["competition"]["cleanup_seconds"] = CANDIDATE_SOAK_CLEANUP_SECONDS
    seconds = candidate_soak_seconds(wall)
    study["hero_v2"]["soak"] = {
        "soak_seconds": wall, "opening_switch_forced_at_seconds": switch,
        "cleanup_seconds": CANDIDATE_SOAK_CLEANUP_SECONDS, "roster": study["roster"],
        "function_timeout": seconds, "trial_start_deadline_seconds": seconds - BOOTSTRAP_RESERVE_SECONDS - wall,
        "measured_setup_seconds": CANDIDATE_SOAK_SETUP_SECONDS,
        "differs_from_the_run": ["wall_seconds", "opening_max_wall_seconds", "opening_cap_absolute",
                                 "competition.cleanup_seconds", "ARENA_PROMPT_TOKENS_DETAILS"]}
    study.update(notebook_seconds=seconds - 60, kernel_timeout=seconds)
    return study


def drift_input_prompts(specs):
    """Teacher-forced token-id sequences from ``<folder>[:k]`` specs (the first k sequences of that folder's
    ``quality/prompts.json``; all when k is left out), merged as in the kv4 drift test."""
    specs = [str(s) for s in (specs.split(",") if isinstance(specs, str) else specs)]
    sources, limits = [], []
    for spec in specs:
        folder, _, limit = spec.partition(":")
        sources.append(Path(folder))
        limits.append(int(limit) if limit else None)
    merged = kv4_drift_prompts(sources)
    kept = []
    for folder, limit in zip(sources, limits):
        rows = [t for t in merged["teacher_forced"] if t["source_folder"] == str(folder)]
        kept.extend(rows if limit is None else rows[:limit])
    return {"greedy": merged["greedy"], "teacher_forced": kept}, specs


def candidate_smoke_study(study, output, drift_inputs, transcript):
    from arena.candidate_profiles import PROFILES

    study = json.loads(json.dumps(study))
    clones = study.get("competition") or {}
    if not study.get("hero_v2") or not clones.get("research_clones") or not drift_inputs or not transcript:
        raise ValueError("The candidate hero-v2 smoke builds on a candidate hero-v2 clone-roster study with "
                         "--hero-v2-drift-inputs and --hero-v2-transcript")
    profile = PROFILES[study["candidate_profile"]]
    if "--kv-cache-dtype" not in profile.extra_args:
        raise ValueError("The candidate smoke's drift arm changes the pinned --kv-cache-dtype")
    soak = int(study["wall_seconds"])
    if not 300 <= soak <= 900:
        raise ValueError("The candidate hero-v2 smoke soak (--wall-seconds) is 300..900 s")
    target = output / "configs/kernel-quality"
    target.mkdir(parents=True, exist_ok=True)
    prompts, sources = drift_input_prompts(drift_inputs)
    (target / "prompts.json").write_text(json.dumps(prompts))
    shutil.copyfile(ROOT / KERNEL_QUALITY_INPUTS / "gsm8k.jsonl", target / "gsm8k.jsonl")
    digests = {name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in ("prompts.json", "gsm8k.jsonl")}
    text = Path(transcript).read_text()
    (output / "bench").mkdir(parents=True, exist_ok=True)
    (output / "bench/transcript.txt").write_text(text)
    active = int(study["active_limit"])
    grid = json.loads(json.dumps(CANDIDATE_SMOKE_GRID))
    grid["points"] = [[64512, active]]
    grid["transcript"] = {"prompt_tokens": [], "concurrency": [], "file": "bench/transcript.txt",
                          "source": str(transcript), "chars": len(text),
                          "sha256": hashlib.sha256(text.encode()).hexdigest()}
    study["throughput_bench"] = grid
    served_kv = dict(zip(profile.extra_args[::2], profile.extra_args[1::2]))["--kv-cache-dtype"]
    pinned = {"ARENA_KV_CACHE_MEMORY_BYTES": str(CANDIDATE_SMOKE_KV_BYTES)}
    arena_env = {"ARENA_PROMPT_TOKENS_DETAILS": "1"}  # exact cached tokens for the prefix check (usage only)
    study["trials"] = [
        {"name": "kv-bf16-ref", "kind": "kernel-quality", "role": "reference", "competition": False,
         "wall_seconds": 420, "server_env": {**pinned, "ARENA_KV_CACHE_DTYPE": "bfloat16"},
         "quality": dict(CANDIDATE_SMOKE_QUALITY)},
        {"name": f"kv-{served_kv}", "kind": "kernel-quality", "role": "gate", "reference": "kv-bf16-ref",
         "competition": False, "wall_seconds": 120, "server_env": dict(pinned),
         "quality": dict(CANDIDATE_SMOKE_QUALITY)},
        {"name": "holo-check", "kind": "serving-bench", "competition": False, "wall_seconds": 60,
         "server_env": dict(arena_env)},
        {"name": "duck-transfer", "kind": "duck-transfer", "server_env": dict(arena_env)},
    ]
    switch = soak // 2
    study["opening_max_wall_seconds"] = switch
    study["opening_cap_absolute"] = True
    study["competition"]["cleanup_seconds"] = CANDIDATE_SMOKE_CLEANUP_SECONDS
    study["hero_v2"]["smoke"] = {
        "drift": {"reference": "kv-bf16-ref (BF16 KV, same attention backend, KV pool pinned)",
                  "candidate": f"kv-{served_kv}", "kv_cache_memory_bytes": CANDIDATE_SMOKE_KV_BYTES,
                  "quality": dict(CANDIDATE_SMOKE_QUALITY),
                  "pass_bar": "per context bin, all and assistant positions: KL and top-1 disagreement <= 1.25 x the "
                              "BF16 reference repeat's; recorded-token NLL change CI not clearly positive",
                  "teacher_forced": len(prompts["teacher_forced"]),
                  "positions": sum(len(t["ids"]) - 1 for t in prompts["teacher_forced"]),
                  "inputs": {"folder": "configs/kernel-quality", "sources": sources, "sha256": digests}},
        "check": f"served layout / focus fit, greedy sanity, tool_choice probe, prefix-cache warm vs cold, {active} x "
                 "64,512-token prompts at once",
        "soak_seconds": soak, "opening_switch_forced_at_seconds": switch,
        "cleanup_seconds": CANDIDATE_SMOKE_CLEANUP_SECONDS, "roster": study["roster"],
        "differs_from_the_run": ["wall_seconds", "opening_max_wall_seconds", "opening_cap_absolute",
                                 "competition.cleanup_seconds", "trials", "throughput_bench",
                                 "ARENA_PROMPT_TOKENS_DETAILS"]}
    study.update(notebook_seconds=CANDIDATE_SMOKE_SECONDS - 60, kernel_timeout=CANDIDATE_SMOKE_SECONDS)
    return study


# hero-v2 pre-check ("Yes, both"; docs/validation-20260927/hero-v2-precheck.md): one server start
# of hero-v2's exact serving at max_num_seqs 32 with --prefix-match-unit 16 and --enable-prompt-tokens-details (neither
# changes the GPU memory plan or the decode step), then the decode / mixed stream sweep, the partial-hit correctness
# check and the prefix-cache smoke's replay (arena.hero_v2_precheck).
# Prompt lengths: the (i + 0.5) / 32 quantiles of arena-hero-v2-2h-v1's 2,658 phase-2 prompts (requests.json, started
# after the switch + 30 s; p50 24,782, p90 40,579).
HERO_V2_PRECHECK_LENGTHS = [6379, 8050, 10274, 12360, 14199, 15502, 16393, 17423, 18273, 19157, 20007, 20743, 21593,
                            22439, 23327, 24236, 25358, 26390, 27400, 28529, 29538, 30799, 32188, 33471, 34753, 36077,
                            37532, 38789, 40126, 41592, 43579, 45944]
HERO_V2_PRECHECK_SEQUENCES = "configs/hero-v2-precheck/sequences.json"
HERO_V2_PRECHECK = {
    "max_num_seqs": 32, "active_limit": 30, "prefix_match_unit": 16, "trial_wall_seconds": 120,
    "stream_sweep": {"lengths": HERO_V2_PRECHECK_LENGTHS, "decode_counts": [16, 23, 26, 29, 32],
                     # 30 = the gate replay's recommended focus streams: with max_num_seqs 32 the injected requests
                     # need free slots, so the mixed passes stop at 30.
                     "mixed_counts": [23, 26, 29, 30], "output_tokens": 1400, "window_seconds": 20,
                     "settle_seconds": 90, "inject_lead_seconds": 4, "calibrate_tokens": 24576,
                     # ~0.45 requests/s x (1,600 fresh + ~800 block remainder + image) ~ 1,100-1,200 uncached tok/s,
                     # the recorded phase-2 rate (hero-v2-analysis/throughput.md).
                     "inject": {"prefix_tokens": 24576, "tail_tokens": 1600, "rate": 0.45}},
    "partial_check": {"file": HERO_V2_PRECHECK_SEQUENCES, "block": 1600, "unit": 16, "probes": 24, "top_logprobs": 20},
    "prefix_replay": {"games": 24, "requests": 6, "concurrency": 8, "max_tokens": 1200, "temperature": 0.7,
                      "system_chars": 12000, "first_chars": 6000, "tool_chars": 1200, "unit": 16},
}


def hero_v2_precheck_study(study, output, transcript, sequences_source):
    """The hero-v2 study (``--hero-v2`` on the clone roster) turned into one serving-bench trial (see above)."""
    study = json.loads(json.dumps(study))
    if not study.get("hero_v2") or not (study.get("competition") or {}).get("research_clones"):
        raise ValueError("The hero-v2 pre-check builds on a hero-v2 clone-roster study (--hero-v2 --clone-instances)")
    spec = HERO_V2_PRECHECK
    text = Path(transcript).read_text()
    (output / "bench").mkdir(parents=True, exist_ok=True)
    (output / "bench/transcript.txt").write_text(text)
    # Two recorded arena contexts (token ids) for the teacher-forced check: the long-context drift inputs.
    rows = json.loads(Path(sequences_source).read_text())["teacher_forced"][:2]
    need = 7 * spec["partial_check"]["block"] + 64
    sequences = [row["ids"][:need] for row in rows]
    if len(sequences) != 2 or any(len(s) < need for s in sequences):
        raise ValueError(f"The pre-check needs two recorded sequences of at least {need} tokens")
    target = output / HERO_V2_PRECHECK_SEQUENCES
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"source": str(sequences_source), "rows": [row.get("source") for row in rows],
                                  "sequences": sequences}))
    before = {"max_num_seqs": study["server"]["max_num_seqs"], "generation_slots": study["generation_slots"],
              "active_limit": study["active_limit"]}
    study["server"]["max_num_seqs"] = spec["max_num_seqs"]
    study["generation_slots"] = spec["max_num_seqs"]
    study["active_limit"] = spec["active_limit"]  # recorded focus-fit check only; the bench plays no game
    study["throughput_bench"] = {
        "prompt_tokens": [], "concurrency": [], "output_tokens": 64, "shared_prefix": None, "points": [],
        "transcript": {"prompt_tokens": [], "concurrency": [], "file": "bench/transcript.txt",
                       "source": str(transcript), "chars": len(text),
                       "sha256": hashlib.sha256(text.encode()).hexdigest()},
        "sanity": KERNEL_SMOKE_GRID["sanity"],
        "stream_sweep": json.loads(json.dumps(spec["stream_sweep"])),
        "partial_check": {**spec["partial_check"], "sha256": hashlib.sha256(target.read_bytes()).hexdigest()},
        "prefix_replay": dict(spec["prefix_replay"])}
    study["trials"] = [{"name": "precheck", "kind": "serving-bench", "competition": False,
                        "wall_seconds": spec["trial_wall_seconds"],
                        "server_env": {"ARENA_PREFIX_MATCH_UNIT": str(spec["prefix_match_unit"]),
                                       "ARENA_PROMPT_TOKENS_DETAILS": "1"}}]
    study["hero_v2"]["precheck"] = {
        "server_changes": {"max_num_seqs": {"before": before["max_num_seqs"], "after": spec["max_num_seqs"]},
                           "generation_slots": {"before": before["generation_slots"], "after": spec["max_num_seqs"]},
                           "prefix_match_unit": {"before": "gcd of the prefix-cacheable blocks (1,600)",
                                                 "after": spec["prefix_match_unit"]},
                           "enable_prompt_tokens_details": {"before": False, "after": True}},
        "active_limit_recorded": {"before": before["active_limit"], "after": spec["active_limit"]},
        "checks": ["served layout at max_num_seqs 32", "decode and mixed stream sweep", "partial-hit correctness",
                   "prefix replay at unit 16"]}
    # ~665-705 s to the first trial; the bench runs to the Modal test's end - 180 s (its deadline); the trial's own
    # timeout is wall + 600 s. The Modal timeout (scripts/modal_validation.py) sets the envelope.
    study.update(notebook_seconds=1500, kernel_timeout=1560, wall_seconds=spec["trial_wall_seconds"])
    return study


# prefix-unit-int4 ("let's test the 3 priority items"; docs/validation-20260927/prefix-unit-int4.md):
# one start of the int4 hero-v2 serving (runs A/B) with --prefix-match-unit 16, at max_num_seqs 48 with CUDA graphs
# captured to 48 (the served pool for streams + 4), and the warm prompt-logprobs patch; then a 48-stream decode check,
# the teacher-forced Latin-square test (arena.partial_tf) and the unit-aware replay at 3,136-token blocks.
PREFIX_UNIT_INT4_SEQUENCES = "configs/prefix-unit-int4/sequences.json"
PREFIX_UNIT_INT4 = {
    "max_num_seqs": 48, "graph_capture_size": 48, "trial_wall_seconds": 120,
    "stream_sweep": {"lengths": [2048] * 64, "decode_counts": [32, 40, 48], "mixed_counts": [], "output_tokens": 1000,
                     "window_seconds": 15, "settle_seconds": 60, "calibrate_tokens": 2048},
    "partial_tf": {"file": PREFIX_UNIT_INT4_SEQUENCES, "block": 3136, "unit": 16, "block_index": 2,
                   "offsets": [768, 1536, 2304], "tail_tokens": 1024, "top_logprobs": 4, "lanes": 8,
                   "bootstrap": 4000, "bar": 1.25, "near_positions": 256},
    "prefix_replay": {"games": 24, "requests": 6, "concurrency": 8, "max_tokens": 1200, "temperature": 0.7,
                      "system_chars": 12000, "first_chars": 6000, "tool_chars": 1200, "unit": 16},
}


def prefix_unit_int4_study(study, output, transcript, sequences_source, tf_only=False):
    """The int4 hero-v2 study (``--hero-v2`` with kv_cache int4, prefix_match_unit 16, prompt_tokens_details) turned into
    one serving-bench trial (see above)."""
    study = json.loads(json.dumps(study))
    server = study["server"]
    if (not study.get("hero_v2") or (server.get("serving_kernels") or {}).get("kv_cache_4bit") != "int4"
            or server.get("prefix_match_unit") != 16 or not server.get("prompt_tokens_details")
            or not (study.get("cache_experiment") or {}).get("prefix_cache")):
        raise ValueError("prefix-unit-int4 builds on hero-v2 with kv_cache int4, prefix_cache, prefix_match_unit 16 "
                         "and prompt_tokens_details")
    spec = PREFIX_UNIT_INT4
    text = Path(transcript).read_text()
    (output / "bench").mkdir(parents=True, exist_ok=True)
    (output / "bench/transcript.txt").write_text(text)
    tf = spec["partial_tf"]
    need = (tf["block_index"] + 1) * tf["block"] + tf["tail_tokens"]
    rows = [row for row in json.loads(Path(sequences_source).read_text())["teacher_forced"] if len(row["ids"]) >= need]
    if len(rows) < 20:
        raise ValueError(f"prefix-unit-int4 needs >= 20 recorded contexts of >= {need} tokens, got {len(rows)}")
    target = output / PREFIX_UNIT_INT4_SEQUENCES
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"source": str(sequences_source), "rows": [row.get("source") for row in rows],
                                  "families": [row.get("family") for row in rows],
                                  "sequences": [row["ids"][:need] for row in rows]}))
    before = {"max_num_seqs": server["max_num_seqs"], "generation_slots": study["generation_slots"],
              "max_cudagraph_capture_size": server.get("max_cudagraph_capture_size")}
    server["max_num_seqs"] = study["generation_slots"] = spec["max_num_seqs"]
    server["max_cudagraph_capture_size"] = spec["graph_capture_size"]
    study["throughput_bench"] = {
        "prompt_tokens": [], "concurrency": [], "output_tokens": 64, "shared_prefix": None, "points": [],
        "transcript": {"prompt_tokens": [], "concurrency": [], "file": "bench/transcript.txt",
                       "source": str(transcript), "chars": len(text),
                       "sha256": hashlib.sha256(text.encode()).hexdigest()},
        "sanity": KERNEL_SMOKE_GRID["sanity"],
        "stream_sweep": json.loads(json.dumps(spec["stream_sweep"])),
        "partial_tf": {**tf, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()},
        "prefix_replay": dict(spec["prefix_replay"])}
    if tf_only:
        # v2 (29 September): v1 measured the pool, the 48-stream decode and the replay, but its warm arms were lost
        # to a length check (the served list of a warm request keeps only the computed rows); v2 reruns the square.
        for key in ("stream_sweep", "prefix_replay"):
            study["throughput_bench"].pop(key)
    study["trials"] = [{"name": "prefix-int4", "kind": "serving-bench", "competition": False,
                        "wall_seconds": spec["trial_wall_seconds"],
                        "server_env": {"ARENA_WARM_PROMPT_LOGPROBS": "1"}}]
    study["hero_v2"]["prefix_unit_int4"] = {
        "server_changes": {k: {"before": before[k], "after": spec["graph_capture_size" if k.startswith("max_cuda")
                                                                  else "max_num_seqs"]} for k in before},
        "warm_prompt_logprobs": "arena.prompt_logprobs_patch (vllm_xargs arc_read_prefix_cache)",
        "contexts": len(rows), "tf_only": bool(tf_only)}
    if tf_only:
        study.update(notebook_seconds=1380, kernel_timeout=1440, wall_seconds=spec["trial_wall_seconds"])
    else:
        study.update(notebook_seconds=1680, kernel_timeout=1740, wall_seconds=spec["trial_wall_seconds"])
    return study


# Decode sweep (docs/validation-20260930/throughput-context-frontier.md section 6,
# experiment 2): the blocks3 campaign serving exactly (int4 KV, prefix-match-unit 16, prune12 + GPTQ-ARC + Marlin,
# ple_uva / embed_host, prefix cache v2, arena sampling) at max_num_seqs 56 with CUDA graphs captured on a finer
# ladder (every 4 from 32 to 56). One serving-bench trial: decode points at 24-56 streams on recorded-length contexts,
# greedy against arena sampling at 32 / 40 / 48, the longest contexts at 24 / 36, one mixed point at 40, and
# nvidia-smi free memory at every point. Research GPU (<= 1 h including setup).
DECODE_SWEEP_GRAPH_SIZES = [1, 2, 4, 8, 16, 24, 32, 36, 40, 44, 48, 52, 56]
DECODE_SWEEP = {
    "max_num_seqs": 56, "graph_capture_size": 56, "trial_wall_seconds": 120, "output_tokens": 1024,
    "length_scale": 0.8, "length_floor": 8192, "streams_max": 56, "pool_blocks_assumed": 1240,
    "stream_sweep": {"decode_counts": [24, 32, 33, 36, 37, 40, 41, 44, 48, 52, 56], "mixed_counts": [40],
                     "output_tokens": 1024, "window_seconds": 15, "settle_seconds": 60, "inject_lead_seconds": 4,
                     "calibrate_tokens": 24576,
                     "inject": {"prefix_tokens": 24576, "tail_tokens": 1600, "rate": 0.45},
                     "variants": [
                         {"label": "greedy", "streams": [32, 40, 48],
                          "overrides": {"temperature": 0.0, "top_p": 1.0, "top_k": -1}},
                         {"label": "longest", "streams": [24, 36], "subset": "longest"}]},
}


def decode_sweep_lengths(requests_path, count, scale, floor, since=1200.0):
    """``count`` prompt lengths: the (i + 0.5) / count quantiles of a recorded run's gameplay prompts (after
    ``since`` seconds of its monotonic clock), scaled by ``scale`` (so that ``count`` of them fit the pool) and
    floored at ``floor``."""
    rows = json.loads(Path(requests_path).read_text())
    start = min(r["started_monotonic"] for r in rows)
    prompts = sorted(int((r.get("usage") or {}).get("prompt_tokens") or 0) for r in rows
                     if r.get("status") == 200 and r["started_monotonic"] - start >= since)
    prompts = [p for p in prompts if p > 0]
    if len(prompts) < 10 * count:
        raise ValueError("decode sweep: too few recorded prompts")
    return [max(floor, int(scale * prompts[int((i + 0.5) / count * len(prompts))])) for i in range(count)]


def decode_sweep_study(study, output, transcript, requests_path):
    """The blocks3 campaign study (``--hero-v2`` with the blocks3-plans components) turned into the decode-sweep
    serving bench above."""
    import math

    study = json.loads(json.dumps(study))
    server = study["server"]
    if (not study.get("hero_v2") or (server.get("serving_kernels") or {}).get("kv_cache_4bit") != "int4"
            or server.get("prefix_match_unit") != 16 or not server.get("prompt_tokens_details")
            or not (study.get("cache_experiment") or {}).get("prefix_cache") or server.get("mtp_tokens") != 0):
        raise ValueError("The decode sweep builds on the hero-v2 int4 / prefix-match-unit 16 / MTP-0 campaign serving")
    spec = DECODE_SWEEP
    text = Path(transcript).read_text()
    (output / "bench").mkdir(parents=True, exist_ok=True)
    (output / "bench/transcript.txt").write_text(text)
    lengths = decode_sweep_lengths(requests_path, spec["streams_max"], spec["length_scale"], spec["length_floor"])
    out = spec["output_tokens"]
    # All streams_max prompts must be in flight together under the exact gate (3,136-token int4 blocks, 9 fixed).
    need = sum(math.ceil((n + out) / 3136) + 9 for n in lengths)
    if need > int(study["kv_margin"] * spec["pool_blocks_assumed"]):
        raise ValueError(f"decode sweep: {need} blocks at {spec['streams_max']} streams do not fit the gate")
    before = {"max_num_seqs": server["max_num_seqs"], "generation_slots": study["generation_slots"],
              "max_cudagraph_capture_size": server.get("max_cudagraph_capture_size"),
              "cudagraph_capture_sizes": server.get("cudagraph_capture_sizes")}
    server["max_num_seqs"] = study["generation_slots"] = spec["max_num_seqs"]
    server["max_cudagraph_capture_size"] = spec["graph_capture_size"]
    server["cudagraph_capture_sizes"] = list(DECODE_SWEEP_GRAPH_SIZES)
    sweep = json.loads(json.dumps(spec["stream_sweep"]))
    sweep["lengths"] = lengths
    study["throughput_bench"] = {
        "prompt_tokens": [], "concurrency": [], "output_tokens": 64, "shared_prefix": None, "points": [],
        "transcript": {"prompt_tokens": [], "concurrency": [], "file": "bench/transcript.txt",
                       "source": str(transcript), "chars": len(text),
                       "sha256": hashlib.sha256(text.encode()).hexdigest()},
        "sanity": KERNEL_SMOKE_GRID["sanity"],
        "stream_sweep": sweep}
    env = {}
    for trial in study.get("trials") or []:
        env.update(trial.get("server_env") or {})
    study["trials"] = [{"name": "sweep", "kind": "serving-bench", "competition": False,
                        "wall_seconds": spec["trial_wall_seconds"], "server_env": env}]
    study["decode_sweep"] = {
        "server_changes": {"max_num_seqs": {"before": before["max_num_seqs"], "after": spec["max_num_seqs"]},
                           "generation_slots": {"before": before["generation_slots"], "after": spec["max_num_seqs"]},
                           "max_cudagraph_capture_size": {"before": before["max_cudagraph_capture_size"],
                                                          "after": spec["graph_capture_size"]},
                           "cudagraph_capture_sizes": {"before": before["cudagraph_capture_sizes"] or "vLLM default",
                                                       "after": list(DECODE_SWEEP_GRAPH_SIZES)}},
        "lengths_source": str(requests_path), "length_scale": spec["length_scale"],
        "blocks_at_max_streams": need, "mean_prompt_tokens": round(sum(lengths) / len(lengths))}
    study.pop("blocks_plan", None)
    # ~600 s to the first trial (prefix-unit-int4 v1: 594 s), the bench to the Modal test's end - 120 s.
    study.update(notebook_seconds=2340, kernel_timeout=2400, wall_seconds=spec["trial_wall_seconds"])
    return study


# Prefix cache v3 smoke ("test the 3 priority items", item 3; docs/validation-20260927/
# prefix-cache-v3.md): hero-v2's serving built with --hero-v2 '{"prefix_cache": true, "request_end_tail": true,
# "kv_cache": "int4", "prompt_tokens_details": true, ...}' as two serving-bench trials in one container: the int4-KV
# server of the build, then the int8-KV server restarted in place (ARENA_SERVING_KERNELS: the hero's kv_cache_8bit).
# Each runs the sanity rows, the teacher-forced request-end tail check (arena.prefix_replay.tail_check) and the
# 24-game replay with exact cached tokens (--enable-prompt-tokens-details).
PREFIX_V3_SEQUENCES = "configs/prefix-cache-v3/sequences.json"
PREFIX_V3_SMOKE = {
    "trial_wall_seconds": 60,
    "tail_check": {"file": PREFIX_V3_SEQUENCES, "probes": 16, "top_logprobs": 20},
    "prefix_replay": {"games": 24, "requests": 5, "concurrency": 8, "max_tokens": 1200, "temperature": 0.7,
                      "system_chars": 12000, "first_chars": 6000, "tool_chars": 1200},
    # arena.timeline estimates: hero-v2 setup to the first trial ~705 s (the pre-check's 665-705 s), an in-place
    # restart >= 614 s, each trial's checks ~280 s.
    "timeline": {"setup_seconds": 705, "restart_seconds": 614, "run_seconds": 280},
    # The recheck: the teacher-forced tail check alone on the int4 server, after the smoke's
    # landing failure (the per-prefix entry cap of 4, now 128). v1's int4 tail check + sanity took ~150 s of its 280.
    "recheck_timeline": {"setup_seconds": 705, "restart_seconds": 614, "run_seconds": 200},
}


def prefix_cache_v3_smoke_study(study, output, transcript, sequences_source, recheck=False):
    """The hero-v2 study with the request-end tail turned into the two-trial serving bench above; ``recheck``: the
    int4 trial with the sanity rows and the tail check only."""
    from arc_infra.serving_kernels_patch import KV4_BLOCK_TOKENS
    from arc_infra.serving_kernels_patch import validate as validate_serving_kernels
    from arena import timeline

    study = json.loads(json.dumps(study))
    prefix = (study.get("cache_experiment") or {}).get("prefix_cache") or {}
    kernels = study["server"].get("serving_kernels") or {}
    if (not study.get("hero_v2") or not prefix.get("request_end_tail") or kernels.get("kv_cache_4bit") != "int4"
            or study["server"].get("mtp_tokens") != 0):
        raise ValueError("The prefix-cache v3 smoke builds on hero-v2 with prefix_cache + request_end_tail at int4 KV "
                         "(--hero-v2 '{\"prefix_cache\": true, \"request_end_tail\": true, \"kv_cache\": \"int4\"}')")
    spec = PREFIX_V3_SMOKE
    text = Path(transcript).read_text()
    (output / "bench").mkdir(parents=True, exist_ok=True)
    (output / "bench/transcript.txt").write_text(text)
    rows = json.loads(Path(sequences_source).read_text())["teacher_forced"][:2]
    need = 7 * KV4_BLOCK_TOKENS["int4"] + 65  # the int4 block; the int8 trial's 1,600-token blocks need less
    sequences = [row["ids"][:need] for row in rows]
    if len(sequences) != 2 or any(len(seq) < need for seq in sequences):
        raise ValueError(f"The tail check needs two recorded sequences of at least {need} tokens")
    target = output / PREFIX_V3_SEQUENCES
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"source": str(sequences_source), "rows": [row.get("source") for row in rows],
                                  "sequences": sequences}))
    int8 = {k: v for k, v in kernels.items() if k != "kv_cache_4bit"}
    int8["kv_cache_8bit"] = "int8"
    validate_serving_kernels(int8)
    common = {"ARENA_PROMPT_TOKENS_DETAILS": "1"}
    study["throughput_bench"] = {
        "prompt_tokens": [], "concurrency": [], "output_tokens": 64, "shared_prefix": None, "points": [],
        "transcript": {"prompt_tokens": [], "concurrency": [], "file": "bench/transcript.txt",
                       "source": str(transcript), "chars": len(text),
                       "sha256": hashlib.sha256(text.encode()).hexdigest()},
        "sanity": KERNEL_SMOKE_GRID["sanity"],
        "tail_check": {**spec["tail_check"], "sha256": hashlib.sha256(target.read_bytes()).hexdigest()},
        "prefix_replay": dict(spec["prefix_replay"])}
    wall = spec["trial_wall_seconds"]
    study["trials"] = [
        {"name": "tail-int4", "kind": "serving-bench", "competition": False, "wall_seconds": wall,
         "server_env": dict(common)},
        {"name": "tail-int8", "kind": "serving-bench", "competition": False, "wall_seconds": wall,
         "server_env": {**common, "ARENA_SERVING_KERNELS": json.dumps(int8, sort_keys=True)}}]
    study["prefix_cache_v3_smoke"] = {"arms": ["tail-int4", "tail-int8"], "patches": dict(prefix),
                                      "int8_serving_kernels": int8, "checks": ["sanity", "tail_check", "prefix_replay"]}
    # Setup + int4 checks + restart + int8 checks with 15% margins and the int8 trial's start rule (wall + 480 s left).
    study.update(notebook_seconds=2400, kernel_timeout=2460, wall_seconds=wall)
    times = spec["timeline"]
    if recheck:
        study["trials"] = study["trials"][:1]
        del study["throughput_bench"]["prefix_replay"]
        study["prefix_cache_v3_smoke"].update(arms=["tail-int4"], checks=["sanity", "tail_check"], recheck=True)
        study.update(notebook_seconds=1400, kernel_timeout=1460)
        times = spec["recheck_timeline"]
    study["prefix_cache_v3_smoke"]["timeline"] = timeline.check(
        study, setup_seconds=times["setup_seconds"], restart_seconds=times["restart_seconds"],
        run_seconds={t["name"]: times["run_seconds"] for t in study["trials"]})
    return study


LEVEL_BENCH_POINTS = "data/level_bench/start_points.json"
LEVEL_BENCH_DEFAULTS = {"wall_seconds": 2400, "mtp_tokens": 0}


def level_bench_study(study, output, options, *, candidate_profile=None):
    """The level bench (27 September, ``arena.level_bench``): one ``level-bench`` trial on the fixed-2x base
    agent (the study as built so far), the committed start points in the bundle, and the serving profile.

    ``options``: ``wall_seconds`` (bench wall; the bench shortens it to what the envelope leaves),
    ``mtp_tokens`` (Flash vLLM only: 0 = speculative decoding off, 3 = the arena's MTP-3), and any
    ``arena.level_bench.DEFAULTS`` key (split, concurrency, window, token_budget, passes, ...)."""
    from arena.level_bench import settings

    options = {**LEVEL_BENCH_DEFAULTS, **options}
    wall, mtp = int(options.pop("wall_seconds")), options.pop("mtp_tokens")
    if not 600 <= wall <= 7200:
        raise ValueError("The level-bench wall must be 600-7200 seconds")
    study = json.loads(json.dumps(study))
    if study.get("competition") or study.get("evolution") or study.get("probe"):
        raise ValueError("The level bench is its own study")
    serving = options.get("serving")
    if candidate_profile or study.get("backend") == "sglang":
        if mtp not in (None, "native"):
            raise ValueError("--bench-mtp applies to the Flash vLLM profile only (candidates keep their own serving)")
        serving = serving or f"{candidate_profile or study.get('backend')} native"
    else:
        if mtp not in (0, 3) or study["server"].get("speculative") is not None:
            raise ValueError("The Flash level bench runs MTP-0 (speculative decoding off) or the arena's MTP-3")
        before = study["server"].get("mtp_tokens", 3)
        study["server"]["mtp_tokens"] = int(mtp)
        study.setdefault("level_bench_changes", {})["server.mtp_tokens"] = {"before": before, "after": int(mtp)}
        serving = serving or ("mtp0 (speculative decoding off)" if mtp == 0 else "mtp3")
    cfg = settings({**options, "serving": serving})
    source = ROOT / LEVEL_BENCH_POINTS
    points = json.loads(source.read_text())
    (output / "configs").mkdir(exist_ok=True)
    shutil.copyfile(source, output / cfg["start_points"])
    selected = [p for p in points["points"] if cfg["split"] == "all" or p["split"] == cfg["split"]]
    if cfg["games"]:
        selected = [p for p in selected if p["game"][:4] in {g[:4] for g in cfg["games"]}]
    if not selected:
        raise ValueError("No level-bench start points selected")
    study.update(
        trials=[{"name": "level-bench", "kind": "level-bench"}], wall_seconds=wall,
        # Setup (install, ~8-minute server start, cache probe) ~900 s plus drain and teardown ~300 s.
        notebook_seconds=wall + 1200, kernel_timeout=wall + 1500,
        harness_selection="duck-transfer (level bench)",
        level_bench=cfg,  # every setting explicit (re-validated by the bench)
        level_bench_points={"source": LEVEL_BENCH_POINTS, "built": points.get("built"), "selected": len(selected),
                            "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
    )
    return study


def llamacpp_study(study, name, *, probe, smoke, sglang, candidate, competition, bench, evolve,
                   slots=None, kv_unified=False, serve_check=None):
    """A Duck-transfer probe on llama-server (model screens, 27 September): the profile's model,
    per-slot window, slots, sampling and effort replace the Flash vLLM serving keys; the probe roster,
    wall, scheduler variant, completion cap and harness stay as built."""
    from arena.llamacpp_profiles import profile

    if not probe or smoke or sglang or candidate or competition or bench or evolve:
        raise ValueError("--llamacpp builds a Duck-transfer --probe-games study only")
    prof = dict(profile(name))
    # GLM follow-up: a unified KV cache (the path the glm5next PR fixed) or one slot
    # (the only path the first serve check showed correct); the serve check can be trimmed for a probe.
    if slots is not None:
        if type(slots) is not int or not 1 <= slots <= 8:
            raise ValueError("llama.cpp slots must be 1..8")
        prof["slots"] = slots
    study = json.loads(json.dumps(study))
    for key in ("cache_admission", "cache_experiment", "isolate_trial_cache"):
        study.pop(key, None)
    study.update(
        backend="llamacpp",
        llamacpp={"profile": name, "repo": prof["repo"], "files": prof["files"], "llama_cpp": prof["llama_cpp"],
                  "server_args": prof["server_args"], "kv_unified": bool(kv_unified),
                  **({"serve_check": dict(serve_check)} if serve_check else {}),
                  **({"serve_check_extra": prof["serve_check_extra"]} if prof.get("serve_check_extra") else {})},
        served_model=prof["served_model"], model_revision=f"{prof['repo']}:{prof['model_file']}",
        serving_profile=f"llamacpp-{name}", windows=[prof["window"]], generation_slots=prof["slots"],
        temperature=prof["sampling"]["temperature"], reasoning_effort=prof["reasoning_effort"],
        sampling={"top_p": prof["sampling"]["top_p"], "top_k": prof["sampling"]["top_k"]},
        server={"backend": "llamacpp", "max_model_len": prof["window"], "slots": prof["slots"],
                "context": prof["slots"] * prof["window"], "prefix_caching": True, "mtp_tokens": 0,
                "kv_unified": bool(kv_unified)},
        observations="letter-coded boards (text-only)" if prof["text_only"] else "board images",
    )
    if study["completion_cap"] >= prof["window"] // 2:
        raise ValueError("The completion cap must leave at least half of the per-slot window for the prompt")
    return study


def overlay_spec(folder, candidate_profile):
    """The study entry of a weights overlay (LoRA test): the prepared upload folder holds the Kaggle
    dataset metadata and ``overlay-manifest.json``; the manifest sha256 is pinned in the study and checked
    in the kernel before the patched shards are built (``arena.weights_overlay``)."""
    from arena.candidate_profiles import PROFILES
    from arena.weights_overlay import MANIFEST_FILE, check_base, load_manifest, sha256_bytes

    if candidate_profile not in PROFILES:
        raise ValueError('A weights overlay needs a reviewed candidate checkpoint (--candidate)')
    folder = Path(folder)
    dataset = validate_runtime_dataset(json.loads((folder / 'dataset-metadata.json').read_text())['id'])
    manifest = load_manifest(folder)
    check_base(manifest, json.loads(PROFILES[candidate_profile].manifest.read_text()))
    return {'dataset': dataset, 'manifest_sha256': sha256_bytes((folder / MANIFEST_FILE).read_bytes()),
            'overlay_sha256': manifest['overlay']['sha256'], 'tensors': manifest['overlay']['tensors'],
            'base': f"{manifest['base']['repo']}@{manifest['base']['revision']}",
            'identical_to_base': manifest.get('identical_to_base'),
            'adapter_sha256': (manifest.get('adapter') or {}).get('sha256')}


def cell(kind, source, index):
    result = {"cell_type": kind, "id": f"arena-{index:02d}", "metadata": {}, "source": source}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def validate_runtime_dataset(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", value):
        raise ValueError("Runtime dataset must be an owner/slug reference")
    return value


def notebook(study, blob_name, digest, *, smoke, runtime_dataset=DATASET):
    validate_runtime_dataset(runtime_dataset)
    seconds = study["smoke"]["notebook_seconds"] if smoke else study["notebook_seconds"]
    wall = study["smoke"]["wall_seconds"] if smoke else study["wall_seconds"]
    games = study["smoke"]["games"] if smoke else len(study["games"])
    trials = ", ".join(t["name"] for t in study["trials"])
    windows = study['windows']
    context_label = (f"{windows[0]:,}-token agent context" if len(windows) == 1 else
                     f"candidate agent contexts {', '.join(f'{w:,}' for w in windows)} tokens")
    context_note = (
        f"{context_label}; exact token admission within the measured SGLang pool"
        if study.get("backend") == "sglang" else
        f"{context_label}; measured hybrid cache blocks control admission up to {study['generation_slots']} requests"
        if study.get("cache_admission") == "hybrid_blocks" else
        f"{context_label}; startup admission checks the KV pool at margin {study['kv_margin']}"
    )
    phase2_note = ('cache-sized phase-2 concurrency (resolved per model at startup)'
                   if study.get('phase2_admission') == 'cache_capacity'
                   else (f"{study['deep_explore']['deep_slots']} deep + {study['deep_explore']['explore_slots']} "
                         "exploration slots from the release")
                   if study.get('deep_explore')
                   else (f"{study['depthwise']['slots']} slots from the release (level-0 rotation, depth games keep "
                         "their slot)")
                   if study.get('depthwise')
                   else (f"{study['rank']['slots']} focus slots from the release, refilled from a continuous "
                         f"token/clear ranking every {study['rank']['tick_seconds']:g} s")
                   if study.get('rank')
                   else f"{study['active_limit']} active games after two reach level 1")
    cells = [
        cell("markdown", f"""# Arena: {study["name"]}
{len(study["trials"])} sequential trials ({trials}) on {games} {'generated' if study.get('generated_games') else 'public'} games under one {wall}-second wall clock
(all games start together; {phase2_note}; per-level time limits (k+1)×T1
in active seconds; {context_note};
completion cap {study["completion_cap"]}, reasoning {study["reasoning_effort"]}, temperature {study["temperature"]},
append-only contexts compacted by the model at the cap). One {study.get('backend', 'vllm')} server
at {study['server']['max_model_len']:,} with prefix caching; independent trials reset the cache.
Public benchmark only; no competition submission.""", 0),
        cell("code", f"""from pathlib import Path
import hashlib, json, subprocess, sys, tarfile, time
assert sys.version_info[:2] == (3, 12), 'Select the Python 3.12 notebook image'
NOTEBOOK_START_EPOCH = time.time()
roots = [Path('/kaggle/input/datasets/{runtime_dataset}'), Path('/kaggle/input/{runtime_dataset.split("/")[1]}')]
mount = next((p for p in roots if p.is_dir()), None)
assert mount is not None, 'Attach the prepared {runtime_dataset} dataset'
blob = mount / '{blob_name}'
with blob.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == '{digest}', 'Runtime archive differs from reviewed build'
bundle = Path('/tmp/arena-runtime')
bundle.mkdir(exist_ok=False)
with tarfile.open(blob, 'r:gz') as archive:
    archive.extractall(bundle, filter='data')
command = [sys.executable, str(bundle / 'bootstrap_arena.py'), '--bundle', str(bundle), '--working', '/tmp/arena',
           '--output', '/kaggle/working/arena',{" '--evolve'," if study.get("evolution") else ""} '--max-seconds', '{seconds}'{", '--smoke'" if smoke else ""}]
result = subprocess.run(command, check=False)
print(json.dumps({{'bootstrap_exit': result.returncode, 'elapsed': time.time() - NOTEBOOK_START_EPOCH}}))""", 1),
        cell("code", """import json
from pathlib import Path
report_path = Path('/kaggle/working/arena/trials/study.json')
if not report_path.is_file():
    probe_path = Path('/kaggle/working/arena/serving/cache-probe/result.json')
    probe = json.loads(probe_path.read_text()) if probe_path.is_file() else {}
    raise RuntimeError(f"Arena stopped before gameplay: bootstrap_exit={result.returncode}; "
                       f"probe_error={probe.get('error')}; inspect saved serving logs")
record = json.loads(report_path.read_text())
for row in record.get('comparison', []):
    print(json.dumps(row))
incomplete = [t['trial']['name'] for t in record['trials'] if t.get('status') != 'complete']
preempted = [t['trial']['name'] for t in record['trials'] if not (t.get('report') or {}).get('preemption_free', False)]
if incomplete or preempted or len(record['trials']) != len(record['study']['trials']):
    raise RuntimeError(f'Arena incomplete: incomplete={incomplete} preempted={preempted}; inspect saved logs')""", 2),
    ]
    if study.get('competition'):
        rehearsal = bool(study['competition'].get('rehearsal'))
        preview = study['competition']['preview_seconds']
        cells[0]['source'] = f'''# Flash Duck-transfer arena competition candidate
Public validation: {study['competition']['preview_instances']} independent cyclic replicas of the {len(study['games'])} public games,
{preview} s of gameplay{' after a 25-game sanity trial' if any(t['name'] == 'sanity-25g' for t in study['trials']) else ''}. Competition rerun: discover the hidden gateway roster,
nine hours total including setup and a ten-minute cleanup reserve.
One RTX PRO 6000 Blackwell 96 GB; fixed winning model/agent/scheduler settings.
Public replicas test scaling and stability, not hidden-game generalization.
The gateway records actions and creates the real submission artifact.'''
        if rehearsal:
            cells[0]['source'] += f'''

**Hidden-rerun rehearsal**: this kernel sets `KAGGLE_IS_COMPETITION_RERUN=1` itself and follows the
rerun branch end to end (sanity trial skipped, {study['competition']['submission_seconds']} s envelope,
rerun tolerance, no placeholder parquet) against the public cyclic gateway. Not a submission.'''
        seconds = str(study['notebook_seconds'])
        assert f"'--max-seconds', '{seconds}']" in cells[1]['source']
        cells[1]['source'] = cells[1]['source'].replace(
            f"'--max-seconds', '{seconds}']",
            f"'--max-seconds', '{seconds}', '--notebook-start-epoch', str(NOTEBOOK_START_EPOCH)]")
        if rehearsal:
            cells[1]['source'] = cells[1]['source'].replace(
                "NOTEBOOK_START_EPOCH = time.time()",
                "NOTEBOOK_START_EPOCH = time.time()\nimport os\nos.environ['KAGGLE_IS_COMPETITION_RERUN'] = '1'  # rehearsal of the hidden-rerun branch")
        cells[2]['source'] = f'''import json, os
from pathlib import Path
EXPECTED_PREVIEW_TRIALS = {len(study['trials'])}
rerun = os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').lower() in ('1', 'true')
report_path = Path('/kaggle/working/arena/trials/study.json')
if not report_path.is_file():
    raise RuntimeError(f'Arena stopped before final report: bootstrap_exit={{result.returncode}}')
record = json.loads(report_path.read_text())
acceptable = {{'complete', 'complete_with_preemptions', 'error'}} if rerun else {{'complete'}}
trials = record.get('trials', [])
expected = 1 if rerun else EXPECTED_PREVIEW_TRIALS
# On the hidden rerun the gateway scorecard is already closed: only a missing or unacceptable
# gateway trial fails the notebook, never the bootstrap exit code or the preview-only checks.
if len(trials) != expected or any(t.get('status') not in acceptable for t in trials) or (result.returncode and not rerun):
    raise RuntimeError('Arena validation failed; inspect saved runtime and trial reports')
for row in record.get('comparison', []):
    print(json.dumps(row))
if not rerun:
    import pandas as pd
    pd.DataFrame([['1_0', '1', True, 1]],
                 columns=['row_id', 'game_id', 'end_of_game', 'score']).to_parquet(
                     '/kaggle/working/submission.parquet', index=False)
    print('Public validation completed. The placeholder selects the notebook for a hidden rerun.')
else:
    print('Hidden gateway scorecard closed; the competition gateway produces the submission artifact.')
'''
    if (study.get('competition') or {}).get('no_preview'):
        # No public preview (build option no_preview): outside the hidden rerun the archive is verified and unpacked,
        # the bootstrap is skipped and the placeholder is written; on the rerun every line runs as before.
        run = "result = subprocess.run(command, check=False)"
        assert cells[1]['source'].count(run) == 1
        cells[1]['source'] = cells[1]['source'].replace(run, """import os
RERUN = os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').lower() in ('1', 'true')
if RERUN:
    result = subprocess.run(command, check=False)
else:
    result = subprocess.CompletedProcess(command, 0)
    print('No public preview: runtime archive verified and unpacked; only the hidden rerun serves the model.')""")
        start = "report_path = Path('/kaggle/working/arena/trials/study.json')\n"
        head, found, tail = cells[2]['source'].partition(start)
        checks, found2, rest = tail.partition("if not rerun:\n")
        assert found and found2
        import textwrap
        cells[2]['source'] = (head + "if rerun:\n" + textwrap.indent(start + checks, '    ') + found2
                              + rest.replace("print('Public validation completed. The placeholder selects the notebook for a hidden rerun.')",
                                             "print('No public preview. The placeholder selects the notebook for a hidden rerun.')"))
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def kaggle_files(output, study, *, smoke, kernel_suffix="", runtime_dataset=DATASET):
    validate_runtime_dataset(runtime_dataset)
    upload = output / "dataset-upload"
    upload.mkdir(exist_ok=True)
    blob_name = f"{study['name']}.tar.gz.blob"
    blob = upload / blob_name
    excluded = {"dataset-upload", "experiment.ipynb", "kernel-metadata.json", "build-manifest.json"}
    with tarfile.open(blob, "w:gz") as archive:
        for path in sorted(output.iterdir()):
            if path.name in excluded:
                continue
            archive.add(path, arcname=path.name)
    digest = hashlib.sha256(blob.read_bytes()).hexdigest()
    (upload / "dataset-metadata.json").write_text(json.dumps({
        "id": runtime_dataset,
        "title": "ARC arena offline runtime",
        "licenses": [{"name": "other"}],
        "description": f"Private benchmark runtime for the arena ({len(study['trials'])} independent trials, "
                       f"{len(study['games'])} public games, {study['wall_seconds']} s per trial) "
                       f"with pinned third-party dependencies; original licenses retained. {blob_name} sha256 {digest}.",
    }, indent=2) + "\n")
    (output / "experiment.ipynb").write_text(json.dumps(
        notebook(study, blob_name, digest, smoke=smoke, runtime_dataset=runtime_dataset), indent=1))
    (output / "kernel-metadata.json").write_text(json.dumps({
        "id": f"richardcsaky/arc-{study['name']}{kernel_suffix}",
        "title": f"ARC {study['name']}{kernel_suffix}",
        "code_file": "experiment.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": ([runtime_dataset, "richardcsaky/arc-sglang-runtime"] if study.get("backend") == "sglang"
                            else [runtime_dataset, *KAGGLE_INPUTS[1:], *study.get('model_datasets', [])]),
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": study.get('model_sources', []) if study.get('candidate_profile') else [MODEL_SOURCE],
        "kernel_sources": [],
    }, indent=2) + "\n")
    (output / "build-manifest.json").write_text(json.dumps({
        "study": study, "blob": blob_name, "sha256": digest, "runtime_dataset": runtime_dataset,
        "kernel_timeout": study["smoke"]["kernel_timeout"] if smoke else study["kernel_timeout"],
    }, indent=2) + "\n")
    return digest


def level_bench_options(args):
    """``build(level_bench=...)`` from the CLI: None unless ``--level-bench``."""
    if not args.level_bench:
        return None
    options = {"wall_seconds": args.bench_wall_seconds, "split": args.bench_split,
               "mtp_tokens": None if (args.candidate or args.sglang) else args.bench_mtp}
    for key in ("concurrency", "window", "token_budget", "passes"):
        value = getattr(args, f"bench_{key}")
        if value is not None:
            options[key] = value
    games = [g.strip()[:4] for g in args.bench_games.split(",") if g.strip()]
    if games:
        options["games"] = games
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="arena-25g-900s-v1")
    parser.add_argument("--smoke", action="store_true", help="3 games, 120 s; kernel id gets -smoke")
    parser.add_argument("--output")
    parser.add_argument("--cache-variant", choices=["blocks", "cpu", "fp8"])
    parser.add_argument("--arena-30m", "--paired-30m", dest="paired_30m", action="store_true",
                        help="25 games, 1800 s: Duck-transfer by default (legacy --paired-30m alias retained)")
    parser.add_argument("--sglang", action="store_true", help="Arena on the pinned Pennyroyal SGLang recipe")
    parser.add_argument("--arena-variant", choices=VARIANTS,
                        help="Control or one-factor concurrency/context/allocation/CUDA graph comparison")
    parser.add_argument("--candidate", choices=['ornith15-35b-a3b-nvfp4'],
                        help="Arena using a reviewed checkpoint-specific serving profile")
    parser.add_argument("--dependencies", action="store_true", help="Download the locked wheels")
    parser.add_argument('--competition', action='store_true',
                        help='Flash Duck: one-hour 110-game preview and nine-hour hidden gateway rerun')
    parser.add_argument('--prioritize-parked-progress', action='store_true',
                        help='One priority resumption after a parked game clears a new level; no active-game preemption')
    parser.add_argument("--fixed-phase2-control", action="store_true",
                        help="Explicit historical fixed-concurrency control; new paired model comparisons default to cache-sized phase 2")
    parser.add_argument('--cudagraph-metrics', action='store_true',
                        help='Log actual graph dispatch shapes/modes; enable equally in both comparison arms')
    parser.add_argument("--opening-burst-requests", type=int, choices=[1, 3], default=1,
                        help="Opening-only continuation affinity; one means FIFO")
    parser.add_argument("--throughput-bench", action="store_true",
                        help="Serving-only throughput grid on the arena server profile; no gameplay")
    parser.add_argument("--bench-transcript", default=None,
                        help="Recorded Duck transcript replayed as the benchmark prompt (replaces the synthetic grid)")
    parser.add_argument('--rerun-rehearsal', action='store_true',
                        help='Competition build that forces the hidden-rerun branch on the public gateway (not a submission)')
    parser.add_argument("--evolve", action="store_true", help="Offline harness evolution study (Modal, 8 h)")
    parser.add_argument("--evolve-rounds", type=int, default=None,
                        help="Evolution rounds (default 5); a rehearsal uses 1 on a short --wall-seconds")
    parser.add_argument("--probe-games", default="",
                        help="Comma-separated games for a short probe: only these play, concurrently, with no phase switch")
    parser.add_argument("--generated-games", default=None,
                        help="Folder with environment_files of generated qNNN games (data/posttrain/games): only these "
                             "play, concurrently, with no phase switch (teacher runs)")
    parser.add_argument("--preview-seconds", type=int, default=3600,
                        help="Competition builds: seconds of the 110-instance public preview (the hidden rerun is unchanged)")
    parser.add_argument("--preview-instances", type=int, default=110,
                        help="Competition builds: instances in the public preview (cyclic over the roster; the hidden rerun is unchanged)")
    parser.add_argument("--clone-instances", type=int, default=None,
                        help="Research arena on N cyclic clones of the roster through the local competition gateway (not a submission)")
    parser.add_argument("--rehearsal-seconds", type=int, default=6000,
                        help="Rerun rehearsal: the envelope that stands in for the 9-hour submission")
    parser.add_argument("--no-sanity-trial", action="store_true",
                        help="Competition builds: skip the 25-game sanity trial in the public preview")
    parser.add_argument("--exclude-games", default="",
                        help="Comma-separated game ids (4-char prefixes) dropped from the 30-minute roster, e.g. ft09,lp85")
    parser.add_argument("--serving-ab", action="store_true",
                        help="Serving bench over control and the SERVING_AB server knobs, restarted in one container")
    parser.add_argument("--serving-lab", action="store_true",
                        help="Serving lab: one server start per SERVING_LAB arm with its own grid (control profiled)")
    parser.add_argument("--wall-seconds", type=int, default=None,
                        help="Gameplay seconds for the Duck-transfer arena (default: the config's 1800), e.g. 3600")
    parser.add_argument("--weights-overlay", default=None,
                        help="Prepared weights-overlay upload folder (dataset-metadata.json + overlay-manifest.json): "
                             "the candidate serves its checkpoint with the overlay's tensors (LoRA test)")
    parser.add_argument("--level-bench", action="store_true",
                        help="Level bench (arena.level_bench): many concurrent single-level attempts from the committed "
                             "start points instead of the 23-game hour; combine with --arena-30m --arena-variant "
                             "reclaim-fixed-2x for the base agent")
    parser.add_argument("--bench-mtp", type=int, choices=[0, 3], default=0,
                        help="Level bench serving on Flash vLLM: 0 = speculative decoding off (default), 3 = MTP-3")
    parser.add_argument("--bench-wall-seconds", type=int, default=LEVEL_BENCH_DEFAULTS["wall_seconds"])
    parser.add_argument("--bench-split", choices=["dev", "holdout", "all"], default="dev",
                        help="Start points: dev (default); holdout is the separate confirmation set")
    parser.add_argument("--bench-concurrency", type=int, default=None)
    parser.add_argument("--bench-window", type=int, default=None)
    parser.add_argument("--bench-token-budget", type=int, default=None)
    parser.add_argument("--bench-passes", type=int, default=None)
    parser.add_argument("--bench-games", default="", help="Comma-separated game prefixes (default: every game in the split)")
    parser.add_argument("--llamacpp", default=None,
                        help="Model screen on llama-server (arena.llamacpp_profiles): a --probe-games Duck study")
    parser.add_argument("--llamacpp-slots", type=int, default=None, help="llama-server slots (default: profile)")
    parser.add_argument("--llamacpp-kv-unified", action="store_true", help="One KV buffer shared by the slots")
    parser.add_argument("--llamacpp-serve-check", default=None,
                        help='JSON serve-check settings, e.g. {"depths": [8192], "streams": [1]}')
    parser.add_argument("--kernel-gate", action="store_true",
                        help="Kernel quality gate (arena.kernel_gate) before the arena trial; needs a variant with "
                             "server.dense_kernels, e.g. fixed-2x-mtp0-wide-kernels")
    parser.add_argument("--kernel-smoke", action="store_true",
                        help="Kernel work, phase 4: server smoke of the phase-4 arm (with --throughput-bench "
                             "--arena-variant fixed-2x-mtp3-wide-kernels, or fixed-2x-mtp3-wide-tweaks for the "
                             "serving tweaks; optional --bench-transcript)")
    parser.add_argument("--draft-head", default=None,
                        help="Draft-head fine-tune job (with --paired-30m --throughput-bench): the directory "
                             "written by scripts/draft_head_data.py")
    parser.add_argument("--draft-head-overlay", default=None,
                        help="Serve a fine-tuned MTP draft head: the overlay directory exported by "
                             "arena.draft_head_train (any Flash vLLM arm, e.g. the kernel smoke)")
    parser.add_argument("--kv8-smoke", action="store_true",
                        help="8-bit KV smoke: one server start of the kv8 arm (with --throughput-bench --arena-variant "
                             "fixed-2x-mtp3-wide-kernels-kv8 --kv8-reference <recorded A+C gate kernel-quality.json>)")
    parser.add_argument("--kv8-reference", default=None,
                        help=f"The recorded BF16-KV reference (main checkout: {KV8_REFERENCE_SOURCE})")
    parser.add_argument("--phase6-smoke", action="store_true",
                        help="kernel work, phase 6: hero control and the phase-6 arm, mixed prefill, microbenchmarks "
                             "(with --throughput-bench --arena-variant fixed-2x-mtp0-wide-kv8-hero --kv8-reference)")
    parser.add_argument("--phase6-part2", action="store_true",
                        help="with --phase6-smoke: one p6 server, mixed-step profiles and the microbenchmarks only")
    parser.add_argument("--prefix-cache-smoke", action="store_true",
                        help="Prefix cache v2 smoke: the pinned and the patched hero prefix cache in one container "
                             "(with --throughput-bench --arena-variant fixed-2x-mtp0-wide-kv8-hero-prefix "
                             "--bench-transcript <file>)")
    parser.add_argument("--hero-smoke", action="store_true",
                        help="Hero smoke: one server start of fixed-2x-mtp0-wide-kv8-hero (with --throughput-bench "
                             "--bench-transcript <file> --kv8-reference <recorded A+C gate kernel-quality.json>)")
    parser.add_argument("--expert-quant", default=None,
                        help="Expert quantization: a folder from scripts/expert_quant_inputs.py (with --paired-30m)")
    parser.add_argument("--expert-quant-phase", default="bench", choices=sorted([*EXPERT_QUANT_PHASES, "evaluate"]))
    parser.add_argument("--expert-quant-settings", default=None,
                        help='calibrate: JSON overrides of the phase settings, or "v2" (build_arena.EXPERT_QUANT_V2)')
    parser.add_argument("--expert-quant-prune", default=None,
                        help="calibrate: capture on this pruned hero variant's mask (mask-aware GPTQ)")
    parser.add_argument("--expert-quant-candidates", default=None,
                        help='evaluate: JSON {trial name: server_env}, e.g. {"eq-w4a16": {"ARENA_SERVING_KERNELS": '
                             '"{\\"moe_backend\\": \\"marlin\\"}"}}')
    parser.add_argument("--drift-long", default=None,
                        help="Long-context drift test (with --paired-30m): a folder from scripts/drift_long_inputs.py")
    parser.add_argument("--kv8-prefill-check", default=None,
                        help="kernel work, phase 7: hero-v2 vs hero-v2 + kv8_prefill (drift gate + injected-prefill "
                             "bench); comma-separated input folders as --kv4-drift")
    parser.add_argument("--kv4-drift", default=None,
                        help="4-bit KV drift test (with --paired-30m): comma-separated input folders with "
                             "quality/prompts.json (expert-quant inputs-v2, drift-long inputs-v1)")
    parser.add_argument("--expert-usage", default=None,
                        help="Expert-usage measurement: a folder from scripts/expert_usage_inputs.py (with --arena-30m)")
    parser.add_argument("--stall-dump", type=int, default=None,
                        help="Pruned-hero diagnostics: all-thread stack dumps when one worker step runs N (30..240) seconds")
    parser.add_argument("--budget-clock", choices=BUDGET_CLOCK_CHOICES, default="tokens",
                        help="Level-budget unit: completion tokens (default since 29 September) or the time clock the "
                             "configuration and variant name (busy / inference / normalized seconds), to reproduce a "
                             "run built before the token clock")
    parser.add_argument("--reasoning-history", choices=("normalized", "verbatim"), default=None,
                        help="Duck history reasoning/content: verbatim (default since 29 September: the server's "
                             "string, <think> tags removed only, so the prefix cache reuses the reply) or normalized "
                             "(the public loop: blank lines removed), to reproduce a run built before the key")
    parser.add_argument("--compaction-thinking", choices=COMPACTION_THINKING, default=None,
                        help="Duck compaction summary: prefilled-empty (default since 29 September: the gameplay "
                             "prefix, no reasoning), gameplay (thinking at the gameplay effort, 28 September) or off "
                             "(tool-less, thinking-off), to reproduce a run built before the key")
    parser.add_argument("--expert-prune-hang-test", action="store_true",
                        help="The prune12 hang test (with --arena-30m --arena-variant fixed-2x-mtp0-wide-kv8-hero-prune12)")
    parser.add_argument("--expert-prune-levels", action="store_true",
                        help="With --expert-prune-check: the layer-aware 25%%, uniform 25%% and 18.75%% candidates")
    parser.add_argument("--expert-prune-check", default=None,
                        help="Load-time expert pruning GPU check (with --arena-30m --arena-variant fixed-2x-mtp0-wide-kv8-hero): the expert-usage inputs folder")
    parser.add_argument("--expert-usage-measure-only", action="store_true",
                        help="With --expert-usage: the eager measurement arm alone (Modal test expert-usage-measure)")
    parser.add_argument("--hero-v2", default=None,
                        help="hero-v2 components as JSON over arena.hero_v2.DEFAULT_COMPONENTS ('{}' = defaults); "
                             "needs --arena-variant fixed-2x-mtp0-wide-kv8-hero and --clone-instances")
    parser.add_argument("--hero-v2-served-blocks", type=int, default=None,
                        help="measured served KV blocks (the hero-v2 smoke) in place of the projected layout")
    parser.add_argument("--hero-v2-smoke", action="store_true",
                        help="hero-v2 smoke: fit/greedy/drift check, then a short clone-roster soak (--wall-seconds)")
    parser.add_argument("--hero-v2-reference", default=None,
                        help="the hero smoke's recorded kv8-quality.json (drift reference of the hero-v2 smoke)")
    parser.add_argument("--early-stop", default=None,
                        help='graceful early stop, e.g. \'{"at_seconds": 3600, "score_above": 1.68}\' (arena.early_stop)')
    parser.add_argument("--hero-v2-window-blocks", type=int, default=None,
                        help="candidate hero-v2: the measured blocks of one 64K focus window (the smoke's cost table)")
    parser.add_argument("--hero-v2-drift-inputs", default=None,
                        help="candidate hero-v2 smoke: <folder>[:k],... with quality/prompts.json token-id sequences "
                             "(the first k of each) for the BF16- vs served-KV drift")
    parser.add_argument("--hero-v2-soak", action="store_true",
                        help="candidate hero-v2: one server start, then a short clone-roster soak (--wall-seconds)")
    parser.add_argument("--hero-v2-transcript", default=None,
                        help="candidate hero-v2 smoke: a recorded duck-transcript.txt for the prefix-cache check")
    parser.add_argument("--prefix-unit-int4", nargs=2, default=None, metavar=("TRANSCRIPT", "SEQUENCES"),
                        help="prefix-unit-int4 check (with --hero-v2 on the int4 clone roster, prefix_match_unit 16): a "
                             "recorded duck-transcript.txt for the replay and a drift-long style prompts.json of >= 20 "
                             "recorded contexts for the teacher-forced test")
    parser.add_argument("--decode-sweep", nargs=2, default=None, metavar=("TRANSCRIPT", "REQUESTS"),
                        help="decode sweep (with --hero-v2 blocks3 campaign serving, 30 September): a recorded "
                             "duck-transcript.txt and a recorded run's requests.json for the prompt lengths")
    parser.add_argument("--prefix-unit-int4-tf-only", action="store_true",
                        help="with --prefix-unit-int4: the teacher-forced square alone (v2; Modal test prefix-unit-int4-tf)")
    parser.add_argument("--hero-v2-precheck", nargs=2, default=None, metavar=("TRANSCRIPT", "SEQUENCES"),
                        help="hero-v2 pre-check (with --hero-v2 on the clone roster): a recorded duck-transcript.txt "
                             "for the replay and the drift-long quality/prompts.json for the partial-hit check")
    parser.add_argument("--prefix-cache-v3-smoke", nargs=2, default=None, metavar=("TRANSCRIPT", "SEQUENCES"),
                        help="prefix cache v3 smoke (with --hero-v2 prefix_cache + request_end_tail at int4 KV): a "
                             "recorded duck-transcript.txt for the replay and the drift-long quality/prompts.json for "
                             "the teacher-forced tail check")
    parser.add_argument("--prefix-cache-v3-recheck", action="store_true",
                        help="with --prefix-cache-v3-smoke: the int4 trial with the tail check only (one server start)")
    parser.add_argument("--kernel-suffix", default="")
    parser.add_argument("--runtime-dataset", default=DATASET, type=validate_runtime_dataset,
                        help="Private runtime dataset; use a small separate dataset when version quota is limited")
    args = parser.parse_args()
    name = args.name + ("-smoke" if args.smoke and not args.name.endswith("-smoke") else "")
    output, study = build(name, smoke=args.smoke, output=args.output, dependencies=args.dependencies,
                          cache_variant=args.cache_variant, paired_30m=args.paired_30m, sglang=args.sglang,
                          candidate_profile=args.candidate, arena_variant=args.arena_variant,
                          competition=args.competition,
                          prioritize_parked_progress=args.prioritize_parked_progress,
                          graph_metrics=args.cudagraph_metrics,
                          opening_burst_requests=args.opening_burst_requests,
                          throughput_bench=args.throughput_bench, rerun_rehearsal=args.rerun_rehearsal, bench_transcript=args.bench_transcript,
        exclude_games=[g for g in args.exclude_games.split(',') if g],
        evolve=args.evolve, wall_seconds=args.wall_seconds, serving_ab=args.serving_ab,
        probe_games=[g for g in args.probe_games.split(',') if g], generated_games=args.generated_games,
        evolve_rounds=args.evolve_rounds, preview_seconds=args.preview_seconds, preview_instances=args.preview_instances, clone_instances=args.clone_instances, sanity_trial=not args.no_sanity_trial, rehearsal_seconds=args.rehearsal_seconds,
        weights_overlay=args.weights_overlay, serving_lab=args.serving_lab, llamacpp=args.llamacpp,
        llamacpp_slots=args.llamacpp_slots, llamacpp_kv_unified=args.llamacpp_kv_unified,
        llamacpp_serve_check=json.loads(args.llamacpp_serve_check) if args.llamacpp_serve_check else None,
        level_bench=level_bench_options(args), kernel_gate=args.kernel_gate, kernel_smoke=args.kernel_smoke,
        draft_head=args.draft_head, draft_head_overlay=args.draft_head_overlay,
        kv8_smoke=args.kv8_smoke, kv8_reference=args.kv8_reference, hero_smoke=args.hero_smoke, phase6_smoke=args.phase6_smoke, phase6_part2=args.phase6_part2,
                          expert_quant=args.expert_quant, expert_quant_phase=args.expert_quant_phase,
                          expert_quant_settings=(EXPERT_QUANT_V2 if args.expert_quant_settings == "v2"
                                                 else json.loads(args.expert_quant_settings)
                                                 if args.expert_quant_settings else None),
                          expert_quant_prune=args.expert_quant_prune,
                          expert_quant_candidates=(
                              json.loads(Path(args.expert_quant_candidates[1:]).read_text()
                                         if args.expert_quant_candidates.startswith("@")
                                         else args.expert_quant_candidates)
                              if args.expert_quant_candidates else None),
                          expert_usage=args.expert_usage,
                          expert_usage_measure_only=args.expert_usage_measure_only,
                          expert_prune_check=args.expert_prune_check, expert_prune_levels=args.expert_prune_levels,
                          expert_prune_hang_test=args.expert_prune_hang_test, stall_dump=args.stall_dump,
                          prefix_cache_smoke=args.prefix_cache_smoke, reasoning_history=args.reasoning_history,
                          hero_v2=json.loads(args.hero_v2) if args.hero_v2 is not None else None,
                          hero_v2_served_blocks=args.hero_v2_served_blocks, hero_v2_smoke=args.hero_v2_smoke,
                          hero_v2_reference=args.hero_v2_reference, hero_v2_window_blocks=args.hero_v2_window_blocks,
                          hero_v2_drift_inputs=args.hero_v2_drift_inputs, hero_v2_transcript=args.hero_v2_transcript,
                          hero_v2_precheck=args.hero_v2_precheck, prefix_unit_int4=args.prefix_unit_int4,
                          prefix_unit_int4_tf_only=args.prefix_unit_int4_tf_only,
                          decode_sweep=args.decode_sweep,
                          prefix_cache_v3_smoke=args.prefix_cache_v3_smoke,
                          prefix_cache_v3_recheck=args.prefix_cache_v3_recheck,
                          hero_v2_soak=args.hero_v2_soak,
                          budget_clock=args.budget_clock, compaction_thinking=args.compaction_thinking,
                          drift_long=args.drift_long, kv4_drift=args.kv4_drift, kv8_prefill_check=args.kv8_prefill_check,
                          early_stop=json.loads(args.early_stop) if args.early_stop else None,
                          capacity_phase2=(args.paired_30m and not args.smoke and not args.cache_variant
                                           and not args.throughput_bench and not args.expert_quant and not args.drift_long
                                           and not args.kv4_drift
                                           and not args.expert_usage and not args.expert_prune_check
                                           and not args.expert_prune_hang_test
                                           and (args.arena_variant is None
                                                or (args.arena_variant == 'graph-128' and args.candidate))
                                           and not args.fixed_phase2_control
                                           and not args.competition))
    digest = kaggle_files(output, study, smoke=args.smoke, kernel_suffix=args.kernel_suffix,
                          runtime_dataset=args.runtime_dataset)
    print(json.dumps({"output": str(output), "sha256": digest, "dependencies": args.dependencies}))


if __name__ == "__main__":
    main()
