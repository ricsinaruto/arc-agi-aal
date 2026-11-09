"""CPU dry run of the combined dfranzen build through the real arena trial (no GPU runs).

    uv run --extra dev python scripts/arena/dfr_dry_run.py [--games 8] [--slots 3] [--window 40960] [--wall 240]

The combined competition study (comp-prune25 + prune12 + int3, dfr_harness, drain_half, budget_policy priority; the
build in docs/validation-20261001/competition-dfr-combined.md) runs ``arena.trial.Trial`` on a few real offline games
against the fake vLLM server of ``samples_dry_run.py`` (its block ledger on the measured 1,776-block pool). To reach the
eviction paths in minutes the dry run shrinks the window (drain trigger = window - completion cap - 512) and the slot
count (so games queue and hand over); everything else is the built study. The fake replies are python tool calls that
act on one or several actions (batches exercise the no-op stop and the latches), use UNDO, call ``frame_diff`` or
inspect. Fails on an engine-ledger or ``max_num_seqs`` violation, task errors, leftover gate reservations, no drain, no
handover, or a missing feature total.
"""

import argparse
import asyncio
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
spec = importlib.util.spec_from_file_location("samples_dry_run", Path(__file__).with_name("samples_dry_run.py"))
dry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dry)
dry.POOL_BLOCKS = 1776  # the pool the Modal offline test measured for this serving (fixed2x-p12-int3)


class CombinedEngine(dry.FakeEngine):
    hx = False  # --hx: some replies use the hx_harness sandbox calls (probe, register_observer, register_tool)

    def reply_code(self, text):
        import re

        match = re.search(r"Valid actions right now: ([^\n]+?)\.", text)
        names = [n.strip() for n in (match.group(1).split(",") if match else ["UP"]) if n.strip()]

        def one():
            name = self.rng.choice(names)
            if name == "MOUSE":
                return {"action": "MOUSE", "row": self.rng.randrange(64), "col": self.rng.randrange(64)}
            return name
        roll = self.rng.random()
        if self.hx and roll < 0.15:
            self.stats["hx_replies"] = self.stats.get("hx_replies", 0) + 1
            return self.rng.choice([
                f"print(probe([{one()!r}, {one()!r}]))",
                "register_observer(lambda b, a: 'step %s' % a.step, note='step')\nprint('ok')",
                "def cells(f):\n    return f.shape\nregister_tool(cells, note='shape')\nprint(cells(current_frame))",
                "print(explore(max_actions=12)) if 'explore' in dir() else print('no explore')",
                f"print(autopilot(lambda f: {one()!r}, max_actions=8)) if 'autopilot' in dir() else print(0)",
                "note('goal', 'unknown yet') if 'note' in dir() else None\nprint(1)",
                "print(register_sim(lambda r, a: r)) if 'register_sim' in dir() else print(2)"])
        if roll < 0.35:
            self.stats["run_plan_replies"] += 1  # here: multi-action batches
            return f"print(action({[one() for _ in range(self.rng.randint(2, 5))]!r}))"
        if roll < 0.7:
            self.stats["action_replies"] += 1
            return f"print(action([{one()!r}]))"
        if roll < 0.8:
            return "print(frame_diff(previous_frame, current_frame) if 'previous_frame' in dir() else 'no previous')"
        self.stats["inspect_replies"] += 1
        return "print(current_frame.shape)"


def combined_study(tmp, v5=False, hx=None, loop=None):
    spec = importlib.util.spec_from_file_location("build_arena_dry", ROOT / "scripts/build_arena.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    from arena.campaign import components

    hero = components("comp-prune25")
    hero.update(expert_prune="prune12", expert_int3=16, window=130816, focus_slots=40, density_blocks=1700,
                graph_capture=48, backlog_fix=False, budget_policy="priority", dfr_harness=True, drain_half=True)
    if v5:  # build_comp_dfr_v5_4h.py: dfr_loop and his 12,288-token reply cap
        hero.update(dfr_loop=True, completion_cap=12288)
    if hx:
        hero["hx_harness"] = hx
    if loop:
        hero["dfr_loop"] = loop
    build.build("arena-flash-competition-dfr-combined-dry", output=tmp / "bundle", paired_30m=True,
                arena_variant="fixed-2x-mtp0-wide-kv8-hero", hero_v2=hero, hero_v2_served_blocks=1839,
                exclude_games=["ft09", "lp85", "ar25", "vc33", "tu93"], opening_burst_requests=3,
                competition=True, preview_seconds=300, preview_instances=110, sanity_trial=False, no_preview=True)
    return json.loads((tmp / "bundle/configs/comparison.json").read_text())


async def main_async(args):
    from arc_infra.arcade import make_arcade
    from arena import trial as trial_module
    from arena.trial import Trial

    gates = []

    class RecordedGate(trial_module.KVGate):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            gates.append(self)

    trial_module.KVGate = RecordedGate
    tmp = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix="dfr-dry-"))
    tmp.mkdir(parents=True, exist_ok=True)
    study = combined_study(tmp, v5=args.v5, hx=json.loads(args.hx) if args.hx else None,
                           loop=json.loads(args.loop) if args.loop else None)
    arc = make_arcade("offline", dry.GAMES_DIR, tmp / "recordings")
    ids = sorted(g.game_id for g in arc.get_environments())
    study["games"] = ids[: args.games]
    study["wall_seconds"] = float(args.wall)
    study.pop("competition", None)
    study["active_limit"] = args.slots
    study["windows"] = [args.window]
    dry.FakeEngine.margin = float(study.get("kv_margin", 0.9))
    server_dir = tmp / "server"
    server_dir.mkdir(exist_ok=True)
    (server_dir / "server.log").write_text("ARC_CACHE_LAYOUT " + json.dumps(dry.layout()) + "\n")
    engine = CombinedEngine(seed=args.seed, seqs=study["server"]["max_num_seqs"])
    engine.hx = bool(args.hx)
    runner, port = await dry.serve(engine)
    try:
        trial = Trial(study, study["trials"][0], source=ROOT, games_dir=dry.GAMES_DIR, output=tmp / "trial",
                      server_dir=server_dir, base_url=f"http://127.0.0.1:{port}/v1", arcade=arc,
                      log=lambda *a, **k: None)
        report = await trial.run()
        await asyncio.sleep(1.5)
    finally:
        await runner.cleanup()
    rows = report.get("rows") or report.get("games") or []
    drains = sum(((r.get("context_drain") or {}).get("drains") or 0) for r in rows)
    dfr = [r.get("dfr_harness") for r in rows if r.get("dfr_harness")]
    loop = report.get("dfr_loop") or {}
    if (args.v5 or args.loop) and not loop:
        from arena.dfr_loop import aggregate as loop_aggregate

        loop = loop_aggregate([r.get("dfr_loop") for r in rows])
    priority = report.get("priority") or (report.get("scheduler") or {}).get("priority") or {}
    result = {"output": str(tmp), "end_reason": report.get("end_reason"), "games": len(study["games"]),
              "slots": args.slots, "window": args.window,
              "engine": {**engine.stats, "peak_running": engine.peak_running, "peak_physical": engine.peak_physical,
                         "gate_capacity": int(engine.margin * dry.POOL_BLOCKS), "violations": engine.violations[:5]},
              "drains": drains, "context_drain_rows": [r.get("context_drain") for r in rows][:3],
              "dfr_harness_rows": dfr[:3], "dfr_loop": loop, "priority": priority,
              "levels": sum(r.get("levels") or 0 for r in rows), "actions": sum(r.get("actions") or 0 for r in rows),
              "task_errors": report.get("task_errors"),
              "game_errors": [r.get("error") for r in rows if r.get("error")]}
    problems = []
    if engine.violations:
        problems.append("engine ledger exceeded the gate capacity or max_num_seqs")
    if not drains:
        problems.append("no context drain happened")
    counts = priority.get("counts") or priority
    if not (counts.get("handover") or counts.get("handover_kept")):
        problems.append("no priority handover was recorded")
    if not dfr:
        problems.append("no dfr_harness totals in the rows")
    if args.v5 and not (loop.get("yields") and loop.get("echoes") and loop.get("stale_openers")):
        problems.append("dfr_loop: no yield, stale opener or echo")
    hx_rows = [r.get("hx_harness") for r in rows if r.get("hx_harness")]
    result["hx_harness_rows"] = hx_rows[:3]
    if args.hx and not hx_rows:
        problems.append("no hx_harness totals in the rows")
    if not gates or gates[0].reserved or gates[0].inflight:
        problems.append("gate reservations left after the run")
    if report.get("task_errors"):
        problems.append("task errors")
    result["problems"] = problems
    (tmp / "dry-run.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str))
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--window", type=int, default=40960)
    ap.add_argument("--wall", type=float, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output")
    ap.add_argument("--v5", action="store_true", help="the v5 study (dfr_loop, 12,288-token cap)")
    ap.add_argument("--hx", help="hx_harness JSON (harness arm) for the study")
    ap.add_argument("--loop", help="dfr_loop JSON (a dict over arena.dfr_loop.FLAGS) for the study")
    args = ap.parse_args()
    from arena import hero_v2

    hero_v2._token_clock = lambda: True
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
