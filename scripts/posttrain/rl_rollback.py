"""Checked rollback of a loop run for a restart (rollbacks go through this tool, never hand edits).

usage: rl_rollback.py RUN_REL rNN [--lr X] [--kl-target X] [--apply] [--state FILE --feed FILE (offline check)]
       rl_rollback.py RUN_REL sNNNN [--apply] [--state v7_state.json --feed FILE [--snap FILE] [--files a,b] (offline)]

v7 (mg.async, step ids sNNNN): the resume point is step NNNN's own snapshot (state/sNNNN/v7_state.json) or, without
one, the current v7_state cut back (lineage to NNNN, ready / running groups and open reads of later versions dropped).
Either way new group ids continue after the current count and every later group is closed "rolled_back", so no group
id (and no kept try under partial/v7/<gid>) of the abandoned branch is reused. Checks: state/sNNNN holds the trainer
checkpoint, the feed's step NNNN row exists with a finite grad norm and version NNNN. --apply keeps the current state
as v7_state.sMMMM.json and writes the new one; the restart resumes from RUN_REL as usual.

The resume point is state rNN's own snapshot (state/rNN/mg_state.json, v5 runs) or, for older runs, the current
mg_state rebuilt from the run's feed: accepted updates and steps of later rounds removed, lr / ramp / dose as they stood
after round NN. Checks: the state's files exist, round NN's update was accepted, its probes' KL to base is recorded
(the in-run state check compares against it). --apply keeps the current mg_state as mg_state.rMM.json and writes the
new one; the restart (ARC_RL1_RESUME_FROM=RUN_REL) then needs no preflight (settings/state only).
"""
import argparse
import json
import sys

VOLUME = "arc-prolong-runs"


def rounds_of(feed_text):
    rows = {}
    for line in feed_text.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") == "round" and isinstance(e.get("round"), int):
            rows[e["round"]] = e
    return rows


def rebuild(state, rows, to):
    """mg_state at state r<to> from the newer ``state`` and the feed's round rows; (new state, problems)."""
    problems, row = [], rows.get(to)
    if row is None or row.get("accepted_updates") is None:
        return None, [f"round {to}: no accepted update in the feed"]
    if int(state.get("played_by", state["round"])) <= to:
        problems.append(f"r{to:02d} is not earlier than the served state (played_by {state.get('played_by')})")
    later = [r for n, r in sorted(rows.items()) if n > to and r.get("accepted_updates") is not None]
    new = dict(state)
    new.update(round=to, played_by=to, served=True, accepted=int(row["accepted_updates"]),
               steps_done=int(state.get("steps_done") or 0) - sum(len((r.get("update") or {}).get("steps") or []) for r in later),
               checkpoint=str(state["checkpoint"]).rsplit("/", 1)[0] + f"/r{to:02d}")
    lr = [c["to"] for c in row.get("lr_control") or [] if c.get("to")]
    nxt = rows.get(to + 1, {}).get("kl_step") or {}
    new["lr"] = float(lr[-1]) if lr else float(nxt["lr"]) if nxt.get("lr") else state.get("lr")
    ramp = row.get("kl_ramp")
    if ramp:
        counts = [tuple(int(x) for x in str(v.get("clears", "")).split("/")) for v in (row.get("train") or {}).values()
                  if "/" in str(v.get("clears", ""))]
        new["kl_ramp"] = {"target": ramp["next_target"], "safe": ramp["safe"], "hold": ramp["hold"],
                          "clears": [sum(c for c, _ in counts), sum(n for _, n in counts)] if counts else None}
        new["kl_target"] = ramp["next_target"]
    step = row.get("kl_step") or {}
    if step.get("excess") is not None and (step.get("lr_effective") or step.get("lr")):
        new["last_dose"] = [float(step.get("lr_effective") or step["lr"]), float(step["excess"])]
    for key in ("kl_trips", "step_rollbacks", "behaviour_trips", "success_streak"):  # counters of the dropped rounds
        new[key] = 0 if any(n > to for n in rows) else state.get(key, 0)
    new["fixed"] = {k: v for k, v in (state.get("fixed") or {}).items() if int(k) <= to}
    probes = {n: p.get("kl_base") for n, p in ((row.get("update") or {}).get("probes") or {}).items()}
    if not probes:
        problems.append(f"round {to}: no recorded probe KL (the in-run state check would skip)")
    if new["steps_done"] < 0:
        problems.append(f"steps_done {new['steps_done']} < 0")
    return new, problems


def steps_of(feed_text):
    rows = {}
    for line in feed_text.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") == "step" and isinstance(e.get("step"), int):
            rows[e["step"]] = e
    return rows


def rollback_v7(state, feed_text, to, snap=None, files=None):
    """v7 state at step ``to`` (version ``to``) from its snapshot or the current state; (new state, problems)."""
    import math

    problems = []
    if files is not None and not ({"master.safetensors", "optimizer.pt"} <= set(files) or "state.json" in set(files)):
        problems.append(f"state/s{to:04d}: no checkpoint ({sorted(files)})")  # state.json: the fake trainer's
    row = steps_of(feed_text).get(to)
    if row is None:
        problems.append(f"step {to}: no step row in the feed")
    elif int(row.get("version", -1)) != to or not (isinstance(row.get("grad_norm"), (int, float))
                                                    and math.isfinite(row["grad_norm"])):
        problems.append(f"step {to}: version {row.get('version')} grad_norm {row.get('grad_norm')}")
    cur = int(state["step"])
    if to >= cur:
        problems.append(f"s{to:04d} is not earlier than the current step {cur}")
    if to not in state["book"]["lineage"]:
        problems.append(f"version {to} not in the current lineage {state['book']['lineage']}")
    if problems:
        return None, problems
    if snap is not None:
        new = json.loads(json.dumps(snap))
        if int(new["step"]) != to or int(new["book"]["current"]) != to:
            return None, [f"snapshot is step {new['step']} version {new['book']['current']}, not {to}"]
        source = "snapshot"
    else:  # the current state cut back: groups and reads of later versions leave, nothing else is known
        new = json.loads(json.dumps(state))
        lin = [v for v in new["book"]["lineage"] if v <= to]
        new["book"].update(lineage=lin, current=to, served=to, pending=None)
        new["buffer"]["ready"] = [g for g in new["buffer"]["ready"] if int(g["version"]) <= to]
        new["running"] = [g for g in new["running"] if int(g["version"]) <= to]
        new["reads"]["open"] = {k: r for k, r in new["reads"]["open"].items() if int(r["version"]) <= to}
        new["step"] = to
        new["adapters"] = {k: p for k, p in new["adapters"].items() if int(k) <= to}
        source = "current state, cut back"
    new["checkpoint"] = str(state["checkpoint"]).rsplit("/", 1)[0] + f"/s{to:04d}"
    keep = {g["gid"] for g in new["buffer"]["ready"]} | {g["gid"] for g in new["running"]}
    closed = dict(state["buffer"]["closed"])
    closed.update(new["buffer"]["closed"])
    made = int(state["made"]["n"])
    for n in range(1, made + 1):  # every group id the abandoned branch made or trained is closed for good
        gid = f"g{n:05d}"
        if gid not in keep and closed.get(gid) not in ("consumed", "dropped"):
            closed[gid] = "rolled_back"
    for gid, why in list(closed.items()):  # consumed after `to` by the abandoned branch: its update is gone, the
        if why == "consumed" and gid not in new["buffer"]["closed"]:  # tries are lost to the new branch
            closed[gid] = "rolled_back"
    for gid in keep:  # a group the restored state still holds plays or trains again
        closed.pop(gid, None)
    new["buffer"]["closed"] = closed
    new["made"] = {**new["made"], "n": made}  # new ids continue: no kept try of a dropped group is ever reused
    new["since_resume"] = 0
    new["rollback_note"] = f"rl_rollback.py: s{to:04d} from {source} (was step {cur})"
    return new, []


def main_v7(a, to):
    vol = None
    if a.state and a.feed:
        state, feed = json.load(open(a.state)), open(a.feed).read()
        snap = json.load(open(a.snap)) if a.snap else None
        files = a.files.split(",") if a.files else None
    else:
        import modal

        vol = modal.Volume.from_name(VOLUME)
        read = lambda p: b"".join(vol.read_file(f"{a.run}/{p}")).decode()
        state, feed = json.loads(read("v7_state.json")), read("arena/trials/rl-loop/rl/feed.jsonl")
        try:
            files = {e.path.rsplit("/", 1)[-1] for e in vol.listdir(f"{a.run}/state/s{to:04d}")}
        except modal.exception.NotFoundError:
            files = set()
        snap = json.loads(read(f"state/s{to:04d}/v7_state.json")) if "v7_state.json" in files else None
    if a.lr is not None or a.kl_target is not None:
        print(json.dumps({"ok": False, "problems": ["v7: lr / KL target are bundle settings, not state"]}, indent=1))
        sys.exit(1)
    new, problems = rollback_v7(state, feed, to, snap, files)
    if problems:
        print(json.dumps({"ok": False, "problems": problems}, indent=1))
        sys.exit(1)
    show = lambda d: {"step": d["step"], "checkpoint": d["checkpoint"], "lineage": d["book"]["lineage"][-4:],
                      "ready": [g["gid"] for g in d["buffer"]["ready"]], "running": [g["gid"] for g in d["running"]],
                      "made": d["made"]["n"], "open_reads": sorted(d["reads"]["open"])}
    print(json.dumps({"ok": True, "note": new["rollback_note"], "before": show(state), "after": show(new)}, indent=1))
    if a.apply and vol is not None:
        keep = f"v7_state.s{int(state['step']):04d}.json"
        with vol.batch_upload(force=True) as up:
            if keep not in {e.path.rsplit("/", 1)[-1] for e in vol.listdir(a.run)}:
                up.put_file(_tmp(json.dumps(state)), f"{a.run}/{keep}")
            up.put_file(_tmp(json.dumps(new)), f"{a.run}/v7_state.json")
        print(f"applied: {a.run}/v7_state.json (previous kept as {keep})")
    elif a.apply and a.out:
        open(a.out, "w").write(json.dumps(new))
        print(f"written: {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("to")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--kl-target", type=float)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--state")
    ap.add_argument("--feed")
    ap.add_argument("--snap")  # v7 offline: state/sNNNN/v7_state.json
    ap.add_argument("--files")  # v7 offline: state/sNNNN's file names, comma-separated
    ap.add_argument("--out")  # v7 offline with --apply: where the new state goes
    a = ap.parse_args()
    if a.to.startswith("s"):
        return main_v7(a, int(a.to[1:]))
    to = int(a.to.lstrip("r"))
    vol = None
    if a.state and a.feed:
        state, feed, files, snap = json.load(open(a.state)), open(a.feed).read(), None, None
    else:
        import modal

        vol = modal.Volume.from_name(VOLUME)
        read = lambda p: b"".join(vol.read_file(f"{a.run}/{p}")).decode()
        state, feed = json.loads(read("mg_state.json")), read("arena/trials/rl-loop/rl/feed.jsonl")
        try:
            files = {e.path.rsplit("/", 1)[-1] for e in vol.listdir(f"{a.run}/state/r{to:02d}")}
        except modal.exception.NotFoundError:
            files = set()
        snap = json.loads(read(f"state/r{to:02d}/mg_state.json")) if "mg_state.json" in files else None
    problems = []
    if files is not None and not {"master.safetensors", "optimizer.pt"} <= files:
        problems.append(f"state/r{to:02d}: files {sorted(files)}")
    served = int(state.get("played_by", state["round"]))
    if to == served:  # no rollback: the restart's settings only (lr / KL target) on the current state
        new, source = dict(state), "current"
        if a.lr is None and a.kl_target is None:
            problems.append("same state and no setting given")
    elif snap is not None:
        new = dict(snap)
        if int(new.get("played_by", -1)) != to:
            problems.append(f"snapshot played_by {new.get('played_by')}")
        source = "snapshot"
    else:
        new, more = rebuild(state, rounds_of(feed), to)
        problems += more
        source = "feed"
    if new is None or problems:
        print(json.dumps({"ok": False, "problems": problems}, indent=1))
        sys.exit(1)
    if a.lr is not None:
        new["lr"] = a.lr
    if a.kl_target is not None:
        new["kl_target"] = a.kl_target
        if new.get("kl_ramp"):
            new["kl_ramp"] = {**new["kl_ramp"], "target": a.kl_target}
    new["rollback_note"] = f"rl_rollback.py: r{to:02d} from {source} (was round {state['round']})"
    show = lambda d: {k: d.get(k) for k in ("round", "played_by", "checkpoint", "accepted", "steps_done", "lr",
                                            "kl_target", "kl_ramp", "last_dose")}
    print(json.dumps({"ok": True, "source": source, "before": show(state), "after": show(new)}, indent=1))
    if a.apply and vol is not None:
        keep = f"mg_state.r{int(state['round']):02d}.json"
        with vol.batch_upload(force=True) as up:
            if keep not in {e.path.rsplit("/", 1)[-1] for e in vol.listdir(a.run)}:
                up.put_file(_tmp(json.dumps(state)), f"{a.run}/{keep}")
            up.put_file(_tmp(json.dumps(new)), f"{a.run}/mg_state.json")
        print(f"applied: {a.run}/mg_state.json (previous kept as {keep})")


def _tmp(text):
    import tempfile

    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    f.write(text)
    f.close()
    return f.name


if __name__ == "__main__":
    main()
