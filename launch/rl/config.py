"""One typed config per RL run: launch/rl/runs/<run>.json holds the whole study, the code's source
commit, the bundle's data files, the launch (backend, env, resume chain, knobs), the preflight, the watcher plan and
the record fields. Strict: no defaults, no unknown keys (runops.launchcfg). build.py turns it into dist/<bundle>/,
launch.py launches it.

Study values may be {"$file": "launch/rl/data/X.json", "sha256": "..."}: the builder inlines that file (big data such
as mg.sets), checking its hash."""
import os
import sys
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
    env: dict[str, str]          # modal: ARC_RL1_* launch env (STALL, TIMEOUT, GATE_ALONGSIDE, MAX_ROUNDS, OVERRIDES, ...)
    resume_from: tuple[str, ...]  # earlier legs, volume-relative (ARC_RL1_RESUME_FROM)
    knobs: dict[str, Any] | None  # the live knob file put before launch (launch/rl/put_knobs.py)
    nebius: dict[str, str] | None  # nebius: {"test": "b200-run", "reason": "..."}
    queue_next: str               # the queue row's `next`


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


def checks(cfg):
    """RL-specific problems ([] = fine)."""
    out = []
    if cfg.launch.backend not in ("modal", "nebius"):
        out.append(f"launch.backend {cfg.launch.backend!r}: modal or nebius")
    if cfg.launch.backend == "nebius" and not (cfg.launch.nebius or {}).get("test"):
        out.append("launch.nebius.test missing")
    if cfg.launch.backend == "modal" and not cfg.run.startswith(("rl1-loop-", "rl1-screen-", "rl1-final-", "rl1-restore-")):
        out.append(f"run {cfg.run}: modal_rl1 run ids start with rl1-loop-/rl1-screen-/rl1-final-/rl1-restore-")
    if not cfg.bundle.startswith(("rl1-loop-", "rl1-screen-", "rl1-final-", "rl1-restore-")):
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
    for tag, leg in cfg.preflight.items():
        miss = [k for k in PREFLIGHT_STUB if k not in leg]
        if miss:
            out.append(f"preflight.{tag}: stub env missing {miss} (corpus, serving, speed, stall as pf-hero.sh set them)")
        if leg.get("ARC_RL1_PATCH_CHECK") != "1" and leg.get("ARC_RL1_PREFLIGHT_TRIAL", "rl-loop") not in trials:
            out.append(f"preflight.{tag}: ARC_RL1_PREFLIGHT_TRIAL {leg.get('ARC_RL1_PREFLIGHT_TRIAL', 'rl-loop')!r} "
                       f"is not a study trial {trials}")
    return out
