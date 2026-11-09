"""v8.2: the v8 source and host side with play at the v7.1p state, plus the items of ``V82``.

Base = build_v8.py with --weights=excl5, --off=border-moves,sweep,floor (v7.1p play: no sweep, no border_moves, no reset
offer, priority floor as v7.1p, half-drain, no stagger, no fraction compaction, reply cap 16,384) and its host items on:
state co-tenant, reply ids, keep-alive, the tuned draft head (``--draft-head-overlay=DIR``, required), ``transient``.
Then, on the built study:

prompt fixes    harness_fixes + solvable_line, death_turn_v2, findings_keep, tool_call_followup, stuck_streak;
                hx_harness reflect_skip_death 1 (docs/harness-arm.md 5.7)
no world model  dfr_loop no_world_model
playbook        hx_harness memo_style 2: one model-written playbook, no length limit, in place of note() and the notes
                block, with the memory framing in the system prompt
animation       dfr_harness animation_history: the frames of every animated action stay readable
transient_note  off (``transient`` alone)

argv: output --draft-head-overlay=DIR [--window=130816] [--streams=48] [--gate-margin=0.96] [--density-blocks=2084]
[--wall=N] [--dependencies] and build_v8.py's other options. The drain drops half the context budget at every window. The three long runs (streams from the throughput arm):
--window=65536 --streams=76 | --window=130816 --streams=48 | --window=196608 --streams=34.
[--smoke-trigger=N] (smoke runs: drain at N prompt tokens instead of the window's budget).
[--level-cap=N] [--level-cap-grant=N] (priority.level_cap_tokens, the per-level token cap; off without it).
A margin above 0.98 is written to the built study (``kv_margin``): hero-v2 accepts 0.5..0.98.
"""
import json
import runpy
import sys
from pathlib import Path

ROOT = Path.cwd()
OWN = {"streams": "48", "window": "130816", "gate-margin": "0.96", "density-blocks": "2084", "smoke-trigger": "",
       "level-cap": "", "level-cap-grant": ""}
FIXES = ("solvable_line", "death_turn_v2", "findings_keep", "tool_call_followup", "stuck_streak")

if __name__ == "__main__":
    argv = sys.argv[1:]
    given = {a[2:].split("=", 1)[0]: a.split("=", 1)[1] for a in argv if a.startswith("--") and "=" in a}
    if not given.get("draft-head-overlay"):
        raise SystemExit("v8.2 serves the tuned draft head: --draft-head-overlay=DIR")
    bad = [a for a in argv if a.startswith(("--off=", "--on=", "--weights=", "--compact", "--stagger", "--version=",
                                             "--completion-cap=", "--kaggle"))]
    if bad:
        raise SystemExit(f"fixed in v8.2: {bad}")
    values = {**OWN, **{k: v for k, v in given.items() if k in OWN}}
    margin = float(values["gate-margin"])
    if not 0.5 <= margin <= 1.0:
        raise SystemExit("--gate-margin is 0.5..1.0")
    plain = [a for a in argv if not a.startswith(tuple(f"--{k}=" for k in OWN))]
    plain += ["--weights=excl5", "--off=border-moves,sweep,floor"]
    arguments, on = runpy.run_path(str(ROOT / "docs/validation-20261007/build_v8.py"))["plan"](plain)
    if "draft-head" not in on:
        raise SystemExit("the draft head is not on")
    arguments = [a for a in arguments if a != "--version=v8"] + [
        "--version=v8.2", f"--streams={values['streams']}", f"--window={values['window']}",
        f"--density-blocks={values['density-blocks']}"]
    arguments = [a for a in arguments if not a.startswith(("--gate-margin=", "--smoke-trigger=", "--level-cap"))]
    own, sys.argv = sys.argv, ["build_v71.py", *arguments]
    try:
        runpy.run_path(str(ROOT / "docs/validation-20261006/build_v71.py"), run_name="__main__")
    finally:
        sys.argv = own
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
    from build_arena import manifest  # noqa: E402

    from arena import dfr_harness, dfr_loop, hx_harness  # noqa: E402
    from arena.harness_fixes import FIXES as BASE_FIXES, OPTIONAL_FIXES  # noqa: E402

    out = Path(own[1])
    path = out / "configs/comparison.json"
    study = json.loads(path.read_text())
    fixes = study["harness_fixes"]["fixes"]
    if set(FIXES) - set(OPTIONAL_FIXES) or set(fixes) - set(BASE_FIXES) - set(OPTIONAL_FIXES):
        raise SystemExit("harness_fixes names changed")
    study["harness_fixes"]["fixes"] = [*fixes, *(f for f in FIXES if f not in fixes)]
    study["hx_harness"] = hx_harness.settings({**study["hx_harness"], "reflect_skip_death": 1, "memo_style": 2})
    study["dfr_harness"] = dfr_harness.settings({**study["dfr_harness"], "transient": True, "transient_note": False,
                                                 "animation_history": True})
    study["dfr_loop"] = {**study["dfr_loop"], **dfr_loop.settings({**{k: study["dfr_loop"][k] for k in dfr_loop.FLAGS},
                                                                   "no_world_model": True})}
    budget = int(values["window"]) - study["completion_cap"] - 512
    # Half the budget at every window: the 59,392 default is the cap fitted to the 130,816 window.
    study["drain_half"] = {**study["drain_half"], "drain_tokens": budget // 2}
    if margin != study["kv_margin"]:
        if margin <= 0.98 and margin != 0.96:
            raise SystemExit("a margin other than 0.96 at or below 0.98 goes through hero-v2 gate_margin: not wired here")
        study["kv_margin"] = margin
    if values["smoke-trigger"]:
        # Smoke runs only: the drain fires at this many prompt tokens (and drops half of it), so a 20-minute run
        # shows a drain; window, streams and gate stay the bundle's.
        if study["wall_seconds"] > 3600:
            raise SystemExit("--smoke-trigger is for smoke runs (--wall of at most 3,600)")
        study["trim"] = {"budget_tokens": int(values["smoke-trigger"])}
    study["v82"] = {"window": int(values["window"]), "streams": int(values["streams"]), "gate_margin": margin,
                    "prompt_fixes": list(FIXES), "source": "docs/validation-20261009/build_v82.py"}
    if values["level-cap"]:
        from arena import priority  # noqa: E402

        study["priority"] = priority.settings({
            **study["priority"], "level_cap_tokens": int(values["level-cap"]),
            **({"level_cap_grant_tokens": int(values["level-cap-grant"])} if values["level-cap-grant"] else {})})
        study["v82"]["level_cap"] = {k: study["priority"][k] for k in ("level_cap_tokens", "level_cap_grant_tokens")}
    dfr, hx, pc = study["dfr_harness"], study["hx_harness"], study["cache_experiment"]["prefix_cache"]
    checks = {
        "streams": study["active_limit"] == int(values["streams"]) and study["server"]["max_num_seqs"] == int(values["streams"]) + 4,
        "window": study["windows"] == [int(values["window"])], "cap": study["completion_cap"] == 16384,
        "weights": study["model_datasets"] == ["richardcsaky/arc-expert-int3-requant-t37-excl5-v1"],
        "draft_head": bool((study["server"].get("draft_head") or {}).get("overlay")),
        "host": all(pc.get(k) is True for k in ("reply_ids", "tail_keepalive", "state_cotenant")),
        "dfr": (dfr["transient"], dfr["transient_note"], dfr["sweep"], dfr["border_moves"], dfr["reset_offer"],
                dfr["animation"], dfr["animation_history"]) == (True, False, False, False, False, True, True),
        "hx": (hx["toolbox"], hx["reflect"], hx["reflect_style"], hx["economy"], hx["memo"], hx["memo_style"],
               hx["reflect_skip_death"], hx["no_stuck_notice"]) == (True, True, 2, True, True, 2, 1, False),
        "loop": study["dfr_loop"]["no_world_model"] is True and study["dfr_loop"]["prompt_fix"] is True
                and study["loop_rule"]["action"] == "fresh_context",
        "play": study.get("compaction_keep_fraction") is None and study.get("first_trigger_stagger") is None
                and study["priority"]["floor"] is True and study["drain_half"]["drain_tokens"] == budget // 2,
        "fixes": set(FIXES) <= set(study["harness_fixes"]["fixes"]) and "stuck_notice" in study["harness_fixes"]["fixes"],
    }
    if not all(checks.values()):
        raise SystemExit(f"built study is not v8.2: {checks}")
    path.write_text(json.dumps(study, indent=2) + "\n")
    identity = json.loads((out / "MANIFEST.json").read_text())
    identity["files"] = manifest(out)
    (out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
    runpy.run_path(str(out / "bootstrap.py"))["verify"](out)
    drain = min(int(study["drain_half"]["drain_tokens"]), budget // 2)
    print(json.dumps({"v82": study["v82"], "max_num_seqs": study["server"]["max_num_seqs"],
                      "graphs": study["server"].get("cudagraph_capture_sizes"), "kv_margin": study["kv_margin"],
                      "context_budget": budget, "drain_keeps": budget - drain, "wall": study["wall_seconds"],
                      "notebook_seconds": study["notebook_seconds"], "games": len(study["games"]),
                      "draft_head": study["server"].get("draft_head"), "gate": study.get("focus_oversubscription")}))
