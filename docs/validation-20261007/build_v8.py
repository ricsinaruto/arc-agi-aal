"""v8: v7.1p (``docs/validation-20261006/build_v71.py --restore-v6d``: the v7.1 harness on INT3 experts,
37.5% pruned, Humming, 44 streams) with every item of ``V8`` on. ``--off=NAME[,NAME]`` switches items off:

weights         the all-25 refit (mask + INT3 overlay, ``WEIGHTS["all25"]``); off = the excl5 set v7.1p served
draft-head      the draft head fine-tuned for the served weights (``DRAFT_HEADS``, or ``--draft-head-overlay=DIR``)
floor           scheduler priority floor off (``priority.floor`` false: float priorities, arena.priority)
state-cotenant  PLE state in the idle tensor slot of GDN group 0's blocks (``prefix_cache.state_cotenant``)
reply-ids       a re-sent reply keeps its generated token ids (``prefix_cache.reply_ids``)
keepalive       a game idle past 5 s (long python call) keeps its cached blocks fresh (``prefix_cache.tail_keepalive``)
transient, border-moves, sweep   three hx4 ``dfr_harness`` opt-in flags

Not in v8, ``--on=NAME`` adds it: reset-offer (hx4 ``dfr_harness.reset_offer``; removed from v8: no
more clears and twice the actions on the step-limit game in the reset bench, and a RESET as a level's first action
restarts the whole game where the gateway lacks ONLY_RESET_LEVELS).

argv: output [--off=NAME,...] [--on=NAME,...] [--weights=all25|excl5] [--mask=FILE --overlay=owner/dataset:sha256] (another set; no
default draft head) [--draft-head-overlay=DIR] [--wall=N] [--dependencies] and build_v71.py's other options, passed on
(``--restore-v6d`` and ``--version=v8`` are always set). Refused: --compact, --compact-full-prefix, --stagger and more
than 44 streams (rejected for v8). Prints build_v71.py's JSON summary.
Diagnostics, not a v8 item: [--collapse-log] (``prefix_cache.collapse_log``: one ARC_TAIL_COLLAPSE line per
full-prefix miss; the engine keeps each finished reply's token ids for it). For test runs; refused with --kaggle.
"""
import json
import runpy
import sys
from pathlib import Path

ROOT = Path.cwd()
WEIGHTS = {  # set -> (mask, overlay dataset:manifest sha256)
    "excl5": ("data/expert_prune/prune-t37-int3-requant-excl5-v1.json",
              "richardcsaky/arc-expert-int3-requant-t37-excl5-v1:"
              "24f9a90f190171daadacfd6a7487df37a4e2d552480f497fe5fdc46c246c7714"),
    # All-25 refit on v7.1 traffic, build b (throughput arm; chosen 7 October).
    "all25": ("data/expert_prune/prune-t37-int3-requant-all25v71b-v1.json",
              "richardcsaky/arc-expert-int3-requant-t37-all25v71b-v1:"
              "17161a50dabe4c93360079fb32a147d20e3a5a601be6e678804ae26bc5f2d099"),
}
DRAFT_HEADS = {}  # weights set -> draft-head overlay directory (arena.draft_head_train export)


def _prefix(key):
    return lambda study: (study["cache_experiment"]["prefix_cache"] or {}).get(key) is True


def _dfr(flag):
    return lambda study: (study.get("dfr_harness") or {}).get(flag) is True


# item -> (build_v71.py arguments that switch it on, the built study shows it on). ``weights`` and ``draft-head`` take
# their arguments from the tables above (see ``plan``).
V8 = {
    "weights": ([], None),
    "draft-head": ([], lambda study: bool((study["server"].get("draft_head") or {}).get("overlay"))),
    "floor": (["--priority-floor=off"], lambda study: study["priority"]["floor"] is False),
    "state-cotenant": (["--state-cotenant"], _prefix("state_cotenant")),
    "reply-ids": (["--reply-ids"], _prefix("reply_ids")),
    "keepalive": (["--keepalive"], _prefix("tail_keepalive")),
    "transient": (["--dfr=transient"], _dfr("transient")),
    "border-moves": (["--dfr=border_moves"], _dfr("border_moves")),
    "sweep": (["--dfr=sweep"], _dfr("sweep")),
}
OPT_IN = {"reset-offer": (["--dfr=reset_offer"], _dfr("reset_offer"))}  # same shape; on only with --on
ITEMS = {**V8, **OPT_IN}
OWN = ("--off=", "--on=", "--weights=")
REFUSED = ("--compact", "--stagger")  # --compact-full-prefix included


def plan(argv):
    """``(build_v71.py argv, items on)`` for build_v8.py's ``argv`` (without the script name)."""
    def arg(name, default=None):
        return next((a.split("=", 1)[1] for a in argv if a.startswith(f"--{name}=")), default)

    off = [name for name in (arg("off") or "").split(",") if name]
    if set(off) - set(V8):
        raise SystemExit(f"--off takes {', '.join(V8)}")
    bad = [a for a in argv if a.startswith(REFUSED)]
    if bad or int(arg("streams", "44")) > 44:
        raise SystemExit(f"not for v8: {bad or '--streams above 44'}")
    if "--collapse-log" in argv and "--kaggle" in argv:
        raise SystemExit("--collapse-log is test-run diagnostics: not in a Kaggle build")
    extra = [name for name in (arg("on") or "").split(",") if name]
    if set(extra) - set(OPT_IN):
        raise SystemExit(f"--on takes {', '.join(OPT_IN)}")
    on = [name for name in V8 if name not in off] + extra
    passed = [a for a in argv if not a.startswith(OWN)]
    if arg("mask") or arg("overlay"):
        chosen = None  # an explicit set (build_v71.py checks the pair)
        if "weights" in off or arg("weights"):
            raise SystemExit("--mask/--overlay name the weights: no --weights and no --off=weights with them")
    else:
        chosen = "excl5" if "weights" in off else arg("weights", "all25")
        if chosen not in WEIGHTS:
            raise SystemExit(f"weights set {chosen} is not declared in WEIGHTS: --mask=... --overlay=..., or "
                             f"--weights={'|'.join(WEIGHTS)}")
        passed += [f"--mask={WEIGHTS[chosen][0]}", f"--overlay={WEIGHTS[chosen][1]}"]
    if "draft-head" in off:
        if arg("draft-head-overlay"):
            raise SystemExit("--draft-head-overlay with --off=draft-head")
    elif not arg("draft-head-overlay"):
        if chosen in DRAFT_HEADS:
            passed.append(f"--draft-head-overlay={DRAFT_HEADS[chosen]}")
        else:
            on.remove("draft-head")  # no head declared for these weights: the stock draft head
    for name in on:
        passed += ITEMS[name][0]
    return [*passed, "--restore-v6d", "--version=v8"], on


if __name__ == "__main__":
    arguments, on = plan(sys.argv[1:])
    own, sys.argv = sys.argv, ["build_v71.py", *arguments]
    try:
        runpy.run_path(str(ROOT / "docs/validation-20261006/build_v71.py"), run_name="__main__")
    finally:
        sys.argv = own
    study = json.loads((Path(own[1]) / "configs/comparison.json").read_text())
    wrong = [name for name, (_, shown) in ITEMS.items() if shown is not None and shown(study) != (name in on)]
    if wrong:
        raise SystemExit(f"built study disagrees with the v8 switches: {wrong}")
