#!/usr/bin/env python3
"""arc3 — open-schema's CLI replication of the [schema] ARC-AGI-3 harness, on our game bridge.

The original harness drove a coding agent (Claude Code for the Claude runs, codex exec for
the GPT runs) with four bespoke tools plus a per-turn prompt. open-schema (brayan07, MIT)
turned that tool surface into a CLI so any agent that can run shell commands and edit files
can play; here the Pi coding agent runs it through bash inside the sandbox. Baseline S
keeps the subcommands, prompts, event schema and model contract; the environment is the
parent process's game (``--env rpc``), git snapshots and the HTML timeline are dropped.

    init      start a run (creates workdir; the opening frame is already recorded)
    observe   the turn prompt: state, legal actions, notes.md, current grid
    backtest  check world_model.py against every recorded transition
    bfs       plan inside the model (only meaningful when backtest is green)
    commit    execute actions; each step is checked against the model's
              prediction — ONE mismatch voids the rest of the plan
    history   inspect recorded transitions
    score     RHAE against a human baseline csv
    finish    emit run_finished (WIN does this automatically)

Everything is recorded to events.jsonl in the released-trace schema.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agi3lib import backtest as bt  # noqa: E402
from agi3lib import bfs as bfs_mod  # noqa: E402
from agi3lib import render, score  # noqa: E402
from agi3lib.env import make_env  # noqa: E402
from agi3lib.events import (  # noqa: E402
    EventLog,
    history_summary,
    transitions_from_events,
)
from agi3lib.model import ModelError, WorldModel, find_model  # noqa: E402


class _Tee(io.TextIOBase):
    def __init__(self, *sinks):
        self.sinks = sinks

    def write(self, text):
        for s in self.sinks:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.sinks:
            s.flush()


def _instrumented(args):
    """Run a subcommand, recording it as tool_started/tool_finished events
    (the original harness logged every tool call the same way) and
    snapshotting the workdir afterwards."""
    run = Run(args.workdir)
    if not run.cfg:  # no run to record against — execute plainly
        return args.fn(args)
    call_id = f"id-{uuid.uuid4().hex[:16]}"
    shown = {k: v for k, v in sorted(vars(args).items())
             if k not in ("fn", "cmd", "workdir") and v is not None}
    run.log.emit("tool_started", turn=run.state.get("turn", 0),
                 call_id=call_id, name=args.cmd, args=shown)
    buf = io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
            args.fn(args)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        raise
    finally:
        # a fresh EventLog re-reads the file, so seq stays consistent with
        # whatever events the command itself appended
        out = buf.getvalue()
        if len(out) > 6000:  # keep head AND tail: plan lines live up top
            out = out[:3000] + "\n... [output truncated] ...\n" + out[-3000:]
        EventLog(os.path.join(args.workdir, "events.jsonl")).emit(
            "tool_finished", turn=run.state.get("turn", 0),
            call_id=call_id, name=args.cmd,
            output=out, is_error=code != 0)


MISPREDICT_MSG = (
    "world model MISPREDICTED the step just taken (action {action}); the "
    "rest of the committed plan was dropped. Run the backtest to see the "
    "mismatch and fix the model before planning again."
)


class Run:
    """A run directory: run.json + state.json + events.jsonl + notes.md."""

    def __init__(self, workdir):
        self.workdir = workdir
        self.cfg = self._read("run.json")
        self.state = self._read("state.json")
        self.log = EventLog(os.path.join(workdir, "events.jsonl"))

    def _read(self, name):
        try:
            with open(os.path.join(self.workdir, name), encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save(self):
        for name, obj in (("run.json", self.cfg), ("state.json", self.state)):
            with open(os.path.join(self.workdir, name), "w",
                      encoding="utf-8") as f:
                json.dump(obj, f, indent=1)

    def env(self):
        history = [(t.action, t.x, t.y) for t in self.transitions()]
        return make_env(self.cfg, self.state.get("env_state"),
                        history=history)

    def record_frame(self, frame):
        self.state["frame"] = {
            "grid": frame.grid, "state": frame.state, "level": frame.level,
            "win_levels": frame.win_levels, "legal": frame.legal,
        }

    def transitions(self):
        return transitions_from_events(self.log.read())

    def model(self):
        path = find_model(self.workdir, self.cfg.get("model"))
        return WorldModel(path) if path else None


def _require_run(args):
    run = Run(args.workdir)
    if not run.cfg:
        sys.exit(f"no run.json in {args.workdir}; run `arc3.py init` first")
    return run


def cmd_init(args):
    os.makedirs(args.workdir, exist_ok=True)
    run = Run(args.workdir)
    if run.cfg and not args.resume:
        sys.exit(f"{args.workdir} already holds a run; use --resume")
    if not run.cfg:
        run.cfg = {"env": args.env, "game_id": args.game,
                   "max_actions": args.max_actions, "agent": args.agent}
    env = run.env()
    # The bridge already performed the opening RESET (observation 0); read it instead of
    # spending an action, as upstream's env.reset() would.
    frame = env.current() if hasattr(env, "current") else env.reset()
    prior = len(run.transitions())
    run.log.emit(
        "run_started", game_id=run.cfg.get("game_id") or args.env,
        provider="cli", model=args.agent, max_actions=args.max_actions,
        win_levels=frame.win_levels, workdir=args.workdir,
        resumed=bool(args.resume), resumed_transitions=prior,
    )
    run.state["turn"] = run.state.get("turn", 0)
    run.state["env_state"] = env.state_dict()
    run.record_frame(frame)
    # Record the opening frame so the first transition has a before-grid
    # even if the agent commits without observing first.
    run.log.emit(
        "turn_started", turn=run.state["turn"], env_step=prior,
        state=frame.state, level=frame.level, win_levels=frame.win_levels,
        legal=frame.legal, grid=frame.grid,
    )
    run.save()
    notes = os.path.join(args.workdir, "notes.md")
    if not os.path.exists(notes):
        with open(notes, "w", encoding="utf-8") as f:
            f.write(NOTES_TEMPLATE)
    print(f"run started: env={args.env} game={run.cfg.get('game_id')} "
          f"state={frame.state} level {frame.level}/{frame.win_levels}")
    print("Next: arc3.py observe")


def cmd_observe(args):
    run = _require_run(args)
    fr = run.state.get("frame")
    if not fr:
        sys.exit("no frame recorded; init failed?")
    run.state["turn"] = run.state.get("turn", 0) + 1
    trans = run.transitions()
    run.log.emit(
        "turn_started", turn=run.state["turn"], env_step=len(trans),
        state=fr["state"], level=fr["level"], win_levels=fr["win_levels"],
        legal=fr["legal"], grid=fr["grid"],
    )
    run.save()
    model_path = find_model(run.workdir, run.cfg.get("model"))
    model_line = (os.path.basename(model_path) if model_path
                  else "NONE yet")
    print(f"State: {fr['state']} | level {fr['level']}/{fr['win_levels']}")
    print(f"Legal actions: {fr['legal']}  "
          "(action 6 is a click: also give x,y in 0..63)")
    print(f"World model: {model_line}; history: {len(trans)} transitions.")
    print(f"Workdir (read/write): {run.workdir}")
    notes_path = os.path.join(run.workdir, "notes.md")
    if os.path.exists(notes_path):
        print("\nYour notes (notes.md — keep it concise, prune stale "
              "entries):")
        with open(notes_path, encoding="utf-8") as f:
            print(f.read().rstrip())
    print("\nCurrent grid:")
    print(render.render(fr["grid"]))


def cmd_backtest(args):
    run = _require_run(args)
    model = run.model()
    if model is None:
        sys.exit("no world_model*.py in the workdir yet")
    indices = ([int(i) for i in args.indices.split(",")]
               if args.indices else None)
    checked, mismatches, skipped, scope = bt.run_backtest(
        model, run.transitions(), start=args.start, end=args.end,
        level=args.level, indices=indices)
    print(bt.format_report(checked, mismatches, skipped, scope,
                           max_details=args.max_details))
    sys.exit(1 if mismatches else 0)


def cmd_bfs(args):
    run = _require_run(args)
    model = run.model()
    if model is None:
        sys.exit("no world_model*.py in the workdir yet")
    fr = run.state["frame"]
    actions = ([int(a) for a in args.actions.split(",")] if args.actions
               else [a for a in fr["legal"] if a not in (0, 6)])
    clicks = []
    if args.clicks:
        for pair in args.clicks.split(";"):
            x, y = pair.split(",")
            clicks.append((int(x), int(y)))
    result = bfs_mod.run_bfs(
        model, fr["grid"], fr["level"], actions, clicks=clicks,
        target=args.target, max_depth=args.max_depth,
        max_nodes=args.max_nodes)
    print(bfs_mod.format_report(result, actions, len(clicks)))
    sys.exit(0 if result.plan else 1)


def _flags(prev_level, frame):
    return {
        "level_up": frame.level > prev_level,
        "dead": frame.state == "GAME_OVER",
        "win": frame.state == "WIN",
    }


def cmd_commit(args):
    run = _require_run(args)
    plan = json.loads(args.actions)
    if not isinstance(plan, list) or not plan:
        sys.exit("--actions must be a non-empty JSON list")
    trans = run.transitions()
    budget = run.cfg.get("max_actions") or 0
    if budget and len(trans) + len(plan) > budget:
        sys.exit(f"plan would exceed max_actions={budget} "
                 f"({len(trans)} used)")
    env = run.env()
    fr = run.state["frame"]
    model = run.model()
    mstate = None
    if model is not None:
        try:
            mstate = model.init_state(fr["grid"], level=fr["level"])
        except ModelError as exc:
            print(f"note: model unusable for prediction checks: {exc}")
            model = None
    turn = run.state.get("turn", 0)
    run.log.emit("turn_committed", turn=turn,
                 plan=[[s.get("action"), s.get("x"), s.get("y")]
                       for s in plan],
                 reason=args.reason or "")
    step_index = len(trans)
    grid, level = fr["grid"], fr["level"]
    for n, stepspec in enumerate(plan):
        action = stepspec["action"]
        x, y = stepspec.get("x"), stepspec.get("y")
        pred = None
        if model is not None and action != 0:
            try:
                pred = model.predict(mstate, grid, action, x, y, level=level)
                mstate = pred[2]
            except ModelError as exc:
                print(f"note: prediction failed ({exc}); executing unchecked")
                pred = None
        frame = env.act(action, x, y)
        flags = _flags(level, frame)
        run.log.emit(
            "action_taken", turn=turn, step_index=step_index, action=action,
            x=x, y=y, level_up=flags["level_up"], dead=flags["dead"],
            win=flags["win"], state=frame.state, level=frame.level,
            grid=frame.grid,
        )
        step_index += 1
        click = f"({x},{y})" if action == 6 else ""
        print(f"step {n}: action={action}{click} -> state={frame.state} "
              f"level={frame.level}"
              + (" LEVEL_UP" if flags["level_up"] else "")
              + (" DEAD" if flags["dead"] else "")
              + (" WIN" if flags["win"] else ""))
        mismatch = None
        if pred is not None:
            pred_grid, pred_flags, _ = pred
            if pred_flags != flags:
                mismatch = (f"flags: predicted {pred_flags}, "
                            f"actual {flags}")
            elif (not (flags["dead"] or flags["win"])
                    and pred_grid != frame.grid):
                mismatch = render.diff_summary(frame.grid, pred_grid)
        run.record_frame(frame)
        run.state["env_state"] = env.state_dict()
        grid, level = frame.grid, frame.level
        if flags["level_up"] and model is not None:
            try:  # model state is rebuilt at every level entry
                mstate = model.init_state(grid, level=level)
            except ModelError:
                model = None
        if mismatch and pred is not None:
            run.log.emit(
                "model_mispredicted", turn=turn, step_index=step_index - 1,
                surprise=MISPREDICT_MSG.format(action=action),
                predicted=pred[0])
            run.save()
            print("\nMISPREDICTED — plan voided "
                  f"({len(plan) - n - 1} step(s) dropped).")
            print(mismatch)
            if frame.state == "WIN":
                # the WIN step itself often mispredicts (the terminal frame
                # is unknowable) — the game is still over; record it
                _emit_finished(run, frame)
                print("\n...but that step WON the game. Run finished.")
                return
            print("Fix the model (the counterexample is transition "
                  f"#{step_index - 1}), get the backtest green, then plan "
                  "again.")
            sys.exit(3)
        if frame.state == "WIN":
            run.save()
            _emit_finished(run, frame)
            print("\nWIN — run finished.")
            return
        if frame.state == "GAME_OVER":
            run.save()
            print("\nGAME_OVER — only RESET (action 0) is legal now"
                  + (f"; {len(plan) - n - 1} step(s) dropped."
                     if n + 1 < len(plan) else "."))
            return
    run.save()
    print(f"\nCommitted {len(plan)} action(s). Re-observe before planning "
          "the next turn.")


def _emit_finished(run, frame):
    trans = run.transitions()
    run.log.emit(
        "run_finished", state=frame.state, levels=frame.level,
        win_levels=frame.win_levels, actions=len(trans),
        transitions=len(trans),
        has_world_model=find_model(run.workdir,
                                   run.cfg.get("model")) is not None,
    )


def cmd_history(args):
    run = _require_run(args)
    trans = run.transitions()
    print(history_summary(trans))
    if args.indices:
        wanted = [int(i) for i in args.indices.split(",")]
    elif args.last:
        wanted = [t.index for t in trans[-args.last:]]
    else:
        return
    subset = [t for t in trans if t.index in set(wanted)]
    print(f"showing indices {wanted} -> {len(subset)} steps; "
          f"detail={args.detail}:")
    for t in subset:
        click = f"({t.x},{t.y})" if t.action == 6 else ""
        n_changed = ("n/a" if t.before is None
                     else len(render.diff_cells(t.before, t.after)))
        flags = [k for k, v in (("level_up", t.level_up), ("dead", t.dead),
                                ("win", t.win)) if v]
        print(f"#{t.index} action={t.action}{click}; {n_changed} cells "
              f"changed; state={t.state}; level={t.level}; flags={flags}")
        if args.detail == "full":
            if t.before is not None:
                print("  before:")
                print("  " + render.render(t.before).replace("\n", "\n  "))
            print("  after:")
            print("  " + render.render(t.after).replace("\n", "\n  "))


def cmd_score(args):
    run = _require_run(args)
    summary = score.summarize_events(run.log.read())
    print(f"state={summary.state} levels_completed="
          f"{len(summary.completed_actions)} "
          f"level_actions={list(summary.completed_actions)} "
          f"total_actions={summary.total_actions}")
    if not args.baseline:
        print("(no --baseline csv given; RHAE not computed)")
        return
    baselines = score.load_baselines(args.baseline)
    game_id = run.cfg.get("game_id") or ""
    if game_id not in baselines:
        sys.exit(f"game_id {game_id!r} not in {args.baseline}")
    _, human = baselines[game_id]
    value, per_level = score.rhae(summary.completed_actions, human)
    print(f"RHAE = {value:.2f}%  per-level="
          f"{[round(s, 2) for s in per_level]}  human={list(human)}")


def cmd_finish(args):
    run = _require_run(args)
    fr = run.state.get("frame") or {}

    class F:
        state = fr.get("state", "NOT_FINISHED")
        level = fr.get("level", 0)
        win_levels = fr.get("win_levels", 0)

    _emit_finished(run, F)
    print(f"run_finished emitted (state={F.state}).")


NOTES_TEMPLATE = """\
# Notes — your living scratchpad (shown to you every observe).
# Keep it CONCISE; edit and PRUNE stale entries as you learn.

## Action semantics (confirmed / guessed)

## Current level

## Hypotheses to test

## Confirmed facts
"""


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="arc3 — open replication of the [schema] ARC-AGI-3 "
                    "harness as a CLI",
        epilog="""the loop:

  1. init      arc3.py --workdir W init --env rpc --game ls20 --agent me
  2. observe   see the current frame, legal actions, and your notes
  3. model     write W/world_model.py:
                 init_state(entry_grid) -> state
                 predict(state, grid, action, x, y) -> (grid', info, state')
               with info flags level_up / dead / win
               (or stateless: step(grid, action, x, y) -> (grid', info))
  4. backtest  must be green over ALL recorded history before planning
  5. bfs       search for a plan INSIDE the model (costs no real actions)
  6. commit    execute; one misprediction voids the rest of the plan and
               hands you the counterexample -> back to 3
  7. repeat 2-6 until WIN

W/notes.md is your scratchpad, shown by every observe. All state lives in
the workdir; every command is a separate process, safe to interleave with
file edits.""")
    p.add_argument("--workdir", default=os.environ.get("ARC3_WORKDIR", "."),
                   help="run directory (default: $ARC3_WORKDIR or .)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="start (or resume) a run")
    sp.add_argument("--env", choices=["toy", "rpc"], default="toy")
    sp.add_argument("--game", default=None, help="game name/id (label only)")
    sp.add_argument("--max-actions", type=int, default=3000)
    sp.add_argument("--agent", default="unspecified-agent",
                    help="label for who is playing (model/harness name)")
    sp.add_argument("--resume", action="store_true")
    sp.set_defaults(fn=cmd_init)

    sp = sub.add_parser("observe", help="the turn prompt")
    sp.set_defaults(fn=cmd_observe)

    sp = sub.add_parser("backtest", help="check the model against history")
    sp.add_argument("--start", type=int, default=None)
    sp.add_argument("--end", type=int, default=None)
    sp.add_argument("--level", type=int, default=None)
    sp.add_argument("--indices", default=None)
    sp.add_argument("--max-details", type=int, default=3)
    sp.set_defaults(fn=cmd_backtest)

    sp = sub.add_parser("bfs", help="plan inside the model")
    sp.add_argument("--target", choices=["advance", "win"], default="advance")
    sp.add_argument("--clicks", default=None, help='"x,y;x,y" candidates')
    sp.add_argument("--actions", default=None, help="override action set")
    sp.add_argument("--max-depth", type=int, default=12)
    sp.add_argument("--max-nodes", type=int, default=20000)
    sp.set_defaults(fn=cmd_bfs)

    sp = sub.add_parser("commit", help="execute actions with model checking")
    sp.add_argument("--actions", required=True,
                    help='JSON: [{"action":3},{"action":6,"x":1,"y":2}]')
    sp.add_argument("--reason", default="")
    sp.set_defaults(fn=cmd_commit)

    sp = sub.add_parser("history", help="inspect recorded transitions")
    sp.add_argument("--last", type=int, default=None)
    sp.add_argument("--indices", default=None)
    sp.add_argument("--detail", choices=["summary", "full"],
                    default="summary")
    sp.set_defaults(fn=cmd_history)

    sp = sub.add_parser("score", help="RHAE for this run")
    sp.add_argument("--baseline", default=None,
                    help="baseline_actions.csv path")
    sp.set_defaults(fn=cmd_score)

    sp = sub.add_parser("finish", help="emit run_finished")
    sp.set_defaults(fn=cmd_finish)

    args = p.parse_args()
    if args.cmd == "init":
        args.fn(args)  # init logs run_started
    else:
        _instrumented(args)


if __name__ == "__main__":
    main()
