"""Ask a running Modal arena run to stop gracefully (arena.early_stop; hero-v2 follow-ups, 29 September).

    ARC_VALIDATION_ROOT=. uv run --extra modal python scripts/arena/request_stop.py RUN_ID \
        --reason "..." [--dry-run]

Writes ``validation/<RUN_ID>/arena/live/stop-request`` on the runs volume. The run's wrapper
(scripts/modal_validation.py ``stop_watcher``) lists that folder every 30 s through the Modal API and mirrors the
request to the trial's local stop file; the trial then ends at its next tick as at a normal wall end (scheduler wall
end, scorecard close, arena.json, evidence, server stop, volume commit). Never cancel the app instead: the evidence
reaches the volume only at the end. Equivalent CLI:
``modal volume put arc-prolong-runs <file> validation/<RUN_ID>/arena/live/stop-request``.
"""

import argparse
import io
import json
import time

VOLUME = "arc-prolong-runs"


def remote_path(run_id):
    return f"validation/{run_id}/arena/live/stop-request"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    payload = json.dumps({"reason": args.reason, "epoch": round(time.time(), 1)}) + "\n"
    target = remote_path(args.run_id)
    if args.dry_run:
        print(json.dumps({"volume": VOLUME, "path": target, "payload": payload.strip(), "dry_run": True}))
        return
    import modal

    volume = modal.Volume.from_name(VOLUME)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(payload.encode()), target)
    print(json.dumps({"volume": VOLUME, "path": target, "requested": True}))


if __name__ == "__main__":
    main()
