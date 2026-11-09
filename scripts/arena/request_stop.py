"""Ask a running Modal arena run to stop gracefully (arena.early_stop; hero-v2 follow-ups, 29 September).

    ARC_VALIDATION_ROOT=. uv run --extra modal python scripts/arena/request_stop.py RUN_ID \
        --reason "..." [--preflight-tag TAG] [--dry-run]

Writes ``validation/<RUN_ID>/arena/live/stop-request`` on the runs volume. The run's wrapper
(scripts/modal_validation.py ``stop_watcher``) lists that folder every 30 s through the Modal API and mirrors the
request to the trial's local stop file; the trial then ends at its next tick as at a normal wall end (scheduler wall
end, scorecard close, arena.json, evidence, server stop, volume commit). Never cancel the app instead: the evidence
reaches the volume only at the end. Use this tool, not a bare ``modal volume put``: it refuses a target no live
attempt reads (a -lost- folder, a wrong id or tag, a finished attempt) and reads the request back after the put. A
request put just before a preemption restart is carried to the resumed attempt (modal_rl1.carry_live).
"""

import argparse
import io
import json
import re
import time

VOLUME = "arc-prolong-runs"
LOST = re.compile(r"-lost-\d+$")


def remote_base(run_id, preflight_tag=None):
    """The live attempt's folder: validation/<id> (arena, RL runs) or validation/preflight/<id><tag> (CPU preflights).
    A preemption restart renames the dead attempt to <base>-lost-<epoch> and runs again at <base>, so <base> is always
    the live attempt."""
    return f"validation/preflight/{run_id}{preflight_tag}" if preflight_tag is not None else f"validation/{run_id}"


def remote_path(run_id, preflight_tag=None):
    return f"{remote_base(run_id, preflight_tag)}/arena/live/stop-request"


def check_target(base, names):
    """Refuses a stop that no live attempt would read: a dead attempt's -lost- folder, a folder that does not exist
    (wrong id or tag), or a finished attempt (its modal-identity.json is written at the end). ``names(rel)`` lists a
    folder's entry names, raising for a missing one."""
    if LOST.search(base.rstrip("/").split("/")[-1]):
        raise SystemExit(f"refused: {base} is a dead attempt kept aside; the live attempt is {LOST.sub('', base)}")
    try:
        entries = set(names(base))
    except Exception as exc:
        raise SystemExit(f"refused: {base} does not exist ({type(exc).__name__}); check the run id and preflight tag")
    if "modal-identity.json" in entries:
        raise SystemExit(f"refused: {base} has finished (modal-identity.json present)")
    return f"{base}/arena/live/stop-request"


def main(argv=None, volume=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--preflight-tag", default=None, help="a CPU preflight leg: validation/preflight/<id><tag>")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    payload = json.dumps({"reason": args.reason, "epoch": round(time.time(), 1)}) + "\n"
    base = remote_base(args.run_id, args.preflight_tag)
    if volume is None:
        import modal

        volume = modal.Volume.from_name(VOLUME)
    names = lambda rel: [e.path.rstrip("/").split("/")[-1] for e in volume.listdir(rel)]
    target = check_target(base, names)
    if args.dry_run:
        print(json.dumps({"volume": VOLUME, "path": target, "payload": payload.strip(), "dry_run": True}))
        return target
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(payload.encode()), target)
    if "stop-request" not in names(f"{base}/arena/live"):  # read back: the request is where the live attempt looks
        raise SystemExit(f"stop-request not found at {target} after the put")
    print(json.dumps({"volume": VOLUME, "path": target, "requested": True, "confirmed": True}))
    return target


if __name__ == "__main__":
    main()
