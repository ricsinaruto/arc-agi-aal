"""v8.1b: v8.1 (build_v8.py --off=border-moves) at 88 streams on a 49,152-token window, 90% fraction
compaction with a no-think summary (4,096 cap, "prefilled-empty"), first compaction staggered from 0.5, reply cap
12,288. Window: 88 x cost(49,152 + 1) = 88 x 26 blocks = 2,288, v8's gate pressure (44 x 52) on the measured 2,084-block
pool (kv_margin 0.96 -> 2,000); the startup rule drops streams one at a time on a smaller pool.

argv: output and build_v8.py's options (--weights=all25|excl5, --draft-head-overlay=DIR, --off=..., --kaggle, --wall=N,
--dependencies ...). --streams / --window / --density-blocks override the v8.1b values (fallback builds).
v8.1b2: --streams=30 --window=196608 --compact-full-prefix --compact-summary-cap=12288
--off=sweep,floor (the summary request on the whole cached context; sweep and priority floor as v7.1p).
v8.1b3: v8.1b2 at --streams=36 (36 x 74 = 2,664 full-window blocks on the 2,000-block gate;
the user accepts the looser bar: requests wait at the gate when the contexts fill).
"""
import json
import runpy
import sys
from pathlib import Path

ROOT = Path.cwd()
V81B = {"streams": "88", "window": "49152", "density-blocks": "2084", "stagger": "0.5", "completion-cap": "12288",
        "compact-thinking": "prefilled-empty", "compact-summary-cap": "4096"}

if __name__ == "__main__":
    argv = sys.argv[1:]
    given = {a[2:].split("=", 1)[0]: a.split("=", 1)[1] for a in argv if a.startswith("--") and "=" in a}
    values = {**V81B, **{k: v for k, v in given.items() if k in V81B}}
    full = "--compact-full-prefix" in argv
    plain = [a for a in argv if not a.startswith(tuple(f"--{k}=" for k in V81B))
             and a not in ("--compact", "--compact-full-prefix")]
    off = sorted({"border-moves", *(n for n in given.get("off", "").split(",") if n)})
    plain = [a for a in plain if not a.startswith("--off=")] + [f"--off={','.join(off)}"]
    arguments, on = runpy.run_path(str(ROOT / "docs/validation-20261007/build_v8.py"))["plan"](plain)
    arguments = [a for a in arguments if a != "--version=v8"] + ["--version=v8.1b", "--compact",
                                                                  *(["--compact-full-prefix"] if full else []),
                                                                  *(f"--{k}={v}" for k, v in values.items())]
    own, sys.argv = sys.argv, ["build_v71.py", *arguments]
    try:
        runpy.run_path(str(ROOT / "docs/validation-20261006/build_v71.py"), run_name="__main__")
    finally:
        sys.argv = own
    study = json.loads((Path(own[1]) / "configs/comparison.json").read_text())
    dfr = study["dfr_harness"]
    checks = {"streams": study["active_limit"] == int(values["streams"]),
              "window": study["windows"] == [int(values["window"])],
              "cap": study["completion_cap"] == int(values["completion-cap"]),
              "compaction": (study["compaction_keep_fraction"], study["compaction_summary_cap"],
                             study["compaction_thinking"], study.get("compaction_full_prefix")) ==
                            (0.1, int(values["compact-summary-cap"]), values["compact-thinking"], True if full else None),
              "stagger": study["first_trigger_stagger"] == {"min_fraction": float(values["stagger"])},
              "harness": (dfr["transient"], dfr["sweep"], dfr["border_moves"], dfr["reset_offer"]) ==
                         (True, "sweep" not in off, False, False),
              "floor": study["priority"]["floor"] is ("floor" in off)}
    if not all(checks.values()):
        raise SystemExit(f"built study is not v8.1b: {checks}")
    print(json.dumps({"v81b": values, "completion_cap": study["completion_cap"],
                      "gate": study["focus_oversubscription"], "kv_margin": study.get("kv_margin")}))
