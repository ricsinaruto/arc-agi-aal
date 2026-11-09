"""One typed config per RL run: launch/rl/runs/<run>.json holds the whole study, the code's source
commit, the bundle's data files, the launch (backend, env, resume chain, knobs), the preflight, the watcher plan and
the record fields. Strict: no defaults, no unknown keys (runops.launchcfg). build.py turns it into dist/<bundle>/,
launch.py launches it.

Study values may be {"$file": "launch/rl/data/X.json", "sha256": "..."}: the builder inlines that file (big data such
as mg.sets), checking its hash."""
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[1]
MAIN = Path(os.environ.get("ARC_MAIN_CHECKOUT", "."))
if not any(Path(p).resolve() == (MAIN / "src").resolve() for p in sys.path if p):  # records live in the main checkout
    sys.path.insert(0, str(MAIN / "src"))
from arc_infra.runops import launchcfg  # noqa: E402


@dataclass(frozen=True)
class Seed:
    dest: str      # path in the bundle
    path: str      # local file or dir (relative to the checkout, or absolute)
    sha256: str    # sha256 of the file, or build.tree_sha256 of the dir


@dataclass(frozen=True)
class Launch:
    backend: str                 # "modal" (modal_rl1.py run) or "nebius" (nlaunch.sh)
    kind: str                    # the record's kind (rl1-loop, arena)
    env: dict[str, str]          # modal: ARC_RL1_* launch env (STALL, TIMEOUT, GATE_ALONGSIDE, OVERRIDES, ...)
    resume_from: tuple[str, ...]  # earlier legs, volume-relative (ARC_RL1_RESUME_FROM)
    knobs: dict[str, Any] | None  # the live knob file put before launch (launch/rl/put_knobs.py)
    nebius: dict[str, str] | None  # nebius: {"test": "b200-run", "reason": "..."}
    queue_next: str               # the queue row's `next`
    allowance_exempt: str         # owner-approved reason outside the arm's GPU allowance (records.py reserve); "" = counted


@dataclass(frozen=True)
class RLConfig(launchcfg.LaunchConfig):
    bundle: str                   # dist/<bundle>/ (also modal_rl1's run id for preflights)
    source: str                   # commit the code comes from (clean, pushed, CI-green)
    packages: tuple[str, ...]     # src/<package> trees, whole, by git archive
    files: dict[str, str]         # bundle path -> repo path at `source` (trainer files, bootstraps, repo data)
    seeds: tuple[Seed, ...]       # data that is not in git at `source`
    study: dict[str, Any]         # configs/comparison.json, whole
    launch: Launch
    preflight: dict[str, dict[str, str]]  # CPU preflight legs: tag -> env (ARC_RL1_PREFLIGHT_*, ARC_RL1_PATCH_CHECK, STALL)
    watch: dict[str, Any]         # tools/ops/watch.py plan (run/app_id/launched filled at launch)


def load(path):
    return launchcfg.load(RLConfig, path)


def bundle_roster(cfg):
    return list(cfg.roster.ids)


PREFLIGHT_STUB = ("ARC_RL1_PREFLIGHT_CORPUS", "ARC_RL1_PREFLIGHT_SERVING", "ARC_RL1_PREFLIGHT_SPEED", "ARC_RL1_STALL")


NEBIUS_BOOT_SECONDS = 1500  # a Nebius RTX boot to server ready (hread-v6 ~700 s, cold ~1,300 s)
PENDING = "PENDING"  # a sha256 placeholder: an input another arm still delivers (e.g. the held-out read's t37 refit)


def pending(cfg):
    """Placeholders the config still carries ([] = every input pinned). build.py and launch.py refuse a pending config;
    checks() does not, so a launch-ready config can be checked in before its last input lands."""
    out = [f"seed {s.dest}" for s in cfg.seeds if str(s.sha256).startswith(PENDING)]
    mask = ((cfg.study.get("server") or {}).get("expert_mask") or {})
    if str(mask.get("sha256", "")).startswith(PENDING):
        out.append("study.server.expert_mask")
    return out


def serving_kernels_patch():
    """This checkout's arc_infra.serving_kernels_patch (by path: the main venv's arc_infra is the main checkout's)."""
    import importlib.util

    if "_rl_skp" not in globals():
        spec = importlib.util.spec_from_file_location("_rl_skp", CHECKOUT / "src/arc_infra/serving_kernels_patch.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        globals()["_rl_skp"] = module
    return globals()["_rl_skp"]


def draft_checks(study):
    """Stage D (mg.draft_weight > 0): the trained draft LoRA is served only with MTP on and ARC_DRAFT_LORA=1 on every
    trial's server (rl_seq refuses the latter at boot; here before launch), including each redesign card's env."""
    mg = (study.get("rl_loop") or {}).get("mg") or {}
    if not float(mg.get("draft_weight") or 0) > 0:
        return []
    out = []
    if int((study.get("server") or {}).get("mtp_tokens") or 0) < 1:
        out.append("mg.draft_weight > 0 with study.server.mtp_tokens < 1: the trained draft is never used")
    for t in study.get("trials") or []:
        if (t.get("server_env") or {}).get("ARC_DRAFT_LORA") != "1":
            out.append(f"mg.draft_weight > 0: trial {t.get('name')} server_env lacks ARC_DRAFT_LORA=1 (the draft is "
                       "not merged at load)")
    redesign = mg.get("redesign") if isinstance(mg.get("redesign"), dict) else {}
    for card, env in (redesign.get("card_env") or {}).items():
        if "ARC_DRAFT_LORA" in (env or {}) and (env or {}).get("ARC_DRAFT_LORA") != "1":
            out.append(f"mg.draft_weight > 0: mg.redesign.card_env {card} sets ARC_DRAFT_LORA "
                       f"{env['ARC_DRAFT_LORA']!r}")
    return out


TRIAL_SLACK = 900  # bootstrap TRIAL_GRACE 600 + its 300-s end reserve
CACHE_DIRS = ("compile-cache", "boot-cache", "trainer-cache", "runtime-cache", "runtime-cache-preflight")


def _strings(v):
    if isinstance(v, str):
        yield from (x.strip() for x in v.split(","))
    elif isinstance(v, Mapping):
        for x in v.values():
            yield from _strings(x)
    elif isinstance(v, Sequence):
        for x in v:
            yield from _strings(x)


def volume_checks(cfg):
    """Per-run volumes: launch.env ARC_RL1_VOLUME "run" mounts the run's own v2 volume at /runs, the shared
    caches at /cache and other runs' outputs read-only at /in/<volume>/. Such a config names caches as /cache/<dir> and
    other runs' folders as /in/<volume>/validation/<run>/... (`records.py volume RUN`); /runs/validation/ and relative
    validation/ paths are this run's own (<run>, <run>-*, or its preflight legs)."""
    env = cfg.launch.env
    if "ARC_RL1_INPUT_VOLUMES" in env:
        return ["launch.env.ARC_RL1_INPUT_VOLUMES: set by launch.py from the config's /in/<volume>/ paths"]
    if env.get("ARC_RL1_VOLUME") is None:
        return []
    if env["ARC_RL1_VOLUME"] != "run":
        return [f"launch.env.ARC_RL1_VOLUME {env['ARC_RL1_VOLUME']!r}: \"run\" (the run's own volume) or absent"]
    out = []
    for s in _strings([cfg.study, dict(env), list(cfg.launch.resume_from), cfg.preflight]):
        rel = s[len("/runs/"):] if s.startswith("/runs/") else s if s.startswith("validation/") else None
        if rel is None:
            continue
        head, _, rest = rel.partition("/")
        if head in CACHE_DIRS:
            out.append(f"{s}: a cache on a per-run volume is /cache/{rel}")
        elif head == "validation":
            other = rest.split("/", 1)[0]
            if other and other != "preflight" and other != cfg.run and not other.startswith(cfg.run + "-"):
                out.append(f"{s}: another run's folder; on a per-run volume it is /in/<its volume>/{rel} "
                           f"(records.py volume {other})")
    return sorted(set(out))



def checks(cfg):
    """RL-specific problems ([] = fine)."""
    out = []
    if cfg.launch.backend not in ("modal", "nebius"):
        out.append(f"launch.backend {cfg.launch.backend!r}: modal or nebius")
    if cfg.launch.backend == "nebius" and not (cfg.launch.nebius or {}).get("test"):
        out.append("launch.nebius.test missing")
    if cfg.launch.backend == "modal" and not cfg.run.startswith("rl1-loop-"):
        out.append(f"run {cfg.run}: modal_rl1 run ids start with rl1-loop-")
    if not cfg.bundle.startswith("rl1-loop-"):
        out.append(f"bundle {cfg.bundle}: its preflight runs as a modal_rl1 run id (rl1-loop-...)")
    if len(cfg.source) != 40 or any(c not in "0123456789abcdef" for c in cfg.source):
        out.append(f"source {cfg.source!r}: the full 40-character commit sha")
    if cfg.study.get("name") is None:
        out.append("study.name missing")
    for k in ("run", "app_id", "launched"):
        if k in cfg.watch:
            out.append(f"watch.{k}: filled at launch, not in the config")
    for k in ("progress_key", "expect"):
        if k not in cfg.watch:
            out.append(f"watch.{k} missing")
    bad = [k for k in cfg.launch.env if not k.startswith(("ARC_RL1_", "ARC_RESTART_ACK"))]
    if bad or "ARC_RL1_BUNDLE" in cfg.launch.env or "ARC_RL1_RESUME_FROM" in cfg.launch.env:
        out.append(f"launch.env: ARC_RL1_* only, without BUNDLE / RESUME_FROM (set from bundle / resume_from): {bad}")
    trials = [t.get("name") for t in cfg.study.get("trials") or []]
    boot = (cfg.study.get("hread") or {}).get("boot_seconds")  # the stage-F template and later (earlier runs: as run)
    if cfg.launch.backend == "nebius" and boot is not None:  # hspeed-a v1 (26 Oct): notebook_seconds counts from the
        boot = max(float(boot), NEBIUS_BOOT_SECONDS)  # bootstrap start, so the boot comes out of the play window
        note = float(cfg.study.get("notebook_seconds") or 0)
        for t in cfg.study.get("trials") or []:
            wall = float(t.get("wall_seconds", cfg.study.get("wall_seconds")) or 0)
            if note < wall + boot:
                out.append(f"study.notebook_seconds {note:.0f} < trial {t.get('name')} wall {wall:.0f} + boot {boot:.0f}: "
                           "the server start comes out of the play window")
        if float(cfg.watch.get("boot_s") or 0) < boot:
            out.append(f"watch.boot_s {cfg.watch.get('boot_s')} < boot allowance {boot:.0f}: the watcher would expect "
                       "play during the boot")
        if cfg.max_seconds < note + 300:
            out.append(f"max_seconds {cfg.max_seconds} < notebook_seconds {note:.0f} + 300 (the Nebius timeout)")
    if cfg.launch.backend == "modal":  # G-long v3 (27 Oct): trial wall 1800 + the bootstrap's 600-s grace killed a
        for t in cfg.study.get("trials") or []:  # 90-min run's play at 40 min; the trial wall must not end it early
            wall = float(t.get("wall_seconds", cfg.study.get("wall_seconds")) or 0)
            if wall < cfg.max_seconds - TRIAL_SLACK:
                out.append(f"trial {t.get('name')} wall_seconds {wall:.0f} < max_seconds {cfg.max_seconds} - "
                           f"{TRIAL_SLACK}: the bootstrap kills the trial at its wall + 600 s, before the run's end")
    server = cfg.study.get("server") or {}
    if server.get("serving_kernels"):  # the serving flags the boot builds (kvgolden-int4 26 Oct: refused on the VM)
        try:
            spec = serving_kernels_patch().validate(server["serving_kernels"])
            if spec:
                serving_kernels_patch().server_flags(spec, int(server.get("mtp_tokens") or 0))
        except Exception as exc:  # noqa: BLE001 - the boot's own refusal, reported before launch
            out.append(f"study.server.serving_kernels: {type(exc).__name__}: {exc}")
    out += volume_checks(cfg)
    absent = [n for n in (cfg.study.get("chain") or {}).get("trials") or [] if n not in trials]
    if absent:  # nebius_validation refuses it at launch (kvgolden 26 Oct); the bootstrap would on the VM
        out.append(f"study.chain.trials {absent}: not a study trial {trials}")
    for tag, leg in cfg.preflight.items():
        miss = [k for k in PREFLIGHT_STUB if k not in leg]
        if miss:
            out.append(f"preflight.{tag}: stub env missing {miss} (corpus, serving, speed, stall as pf-hero.sh set them)")
        if leg.get("ARC_RL1_PATCH_CHECK") != "1" and leg.get("ARC_RL1_PREFLIGHT_TRIAL", "rl-loop") not in trials:
            out.append(f"preflight.{tag}: ARC_RL1_PREFLIGHT_TRIAL {leg.get('ARC_RL1_PREFLIGHT_TRIAL', 'rl-loop')!r} "
                       f"is not a study trial {trials}")
    rl_loop = cfg.study.get("rl_loop") or {}
    out += draft_checks(cfg.study)
    if (rl_loop.get("mg") or {}).get("redesign"):  # the lr schedule, on this checkout's rl_seq
        import json as _json
        import subprocess as _sp

        got = _sp.run([sys.executable, "-c", "import json, sys; from arena import rl_seq; "
                       "print(json.dumps(rl_seq.schedule_check(json.loads(sys.stdin.read()))))"],
                      input=_json.dumps(rl_loop), capture_output=True, text=True,
                      env={**os.environ, "PYTHONPATH": str(CHECKOUT / "src")}, cwd=str(CHECKOUT))
        try:
            out += _json.loads(got.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            out.append(f"lr schedule check did not run: {(got.stderr or got.stdout)[-300:]}")
    return out
