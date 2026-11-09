"""Poll one LoRA-test Modal run by the call id in its launch record; on completion download its
outputs from the runs volume (``arc-prolong-runs:/lora/<run id>``) to ``runs/modal-lora/<run id>``.

Usage:
    uv run --extra modal python scripts/posttrain/poll_modal_lora.py RUN_ID [--once] [--interval 120]
        [--horizon SECONDS] [--no-download]

It never stops apps or relaunches. Reconcile ``docs/spend.json`` to the measured seconds afterwards.
"""

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "docs/validation-20260926"


def stamp():
    return datetime.datetime.now(datetime.UTC).strftime("%H:%M:%S")


def load_record(run_id):
    path = RECORDS / f"modal-{run_id}.json"
    record = json.loads(path.read_text())
    if record.get("kind") != "lora" or not record.get("function_call_id"):
        raise SystemExit(f"{path}: not a submitted LoRA launch record (no function_call_id)")
    return path, record


def download(run_id, dest_root=ROOT / "runs/modal-lora"):
    dest_root.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["uv", "run", "--extra", "modal", "modal", "volume", "get", "--force", "arc-prolong-runs",
                             f"lora/{run_id}", str(dest_root)], capture_output=True, text=True, cwd=ROOT)
    print(stamp(), "volume get exit", result.returncode, (result.stdout + result.stderr)[-600:], flush=True)
    return dest_root / run_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=None, help="stop polling after this many seconds")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)
    path, record = load_record(args.run_id)
    import modal

    call = modal.FunctionCall.from_id(record["function_call_id"])
    horizon = args.horizon or record.get("spec", {}).get("seconds", 10800) + 1800
    started, state = time.time(), None
    while True:
        try:
            result = call.get(timeout=0)
            print(stamp(), "DONE", json.dumps(result, default=str)[:2000], flush=True)
            state = "done"
            record.update(state="done", result=result)
            break
        except TimeoutError:
            print(stamp(), "running", round((time.time() - started) / 60), "min", flush=True)
        except (ConnectionError, OSError) as exc:
            print(stamp(), "transient", type(exc).__name__, str(exc)[:200], flush=True)
        except Exception as exc:  # noqa: BLE001 - the remote exception type is not importable locally
            if "Deadline exceeded" in str(exc) or "nodename nor servname" in str(exc):
                print(stamp(), "transient", type(exc).__name__, str(exc)[:200], flush=True)
            else:
                print(stamp(), "ERROR", type(exc).__name__, str(exc)[:1000], flush=True)
                state = "error"
                record.update(state="error", error=f"{type(exc).__name__}: {str(exc)[:1000]}")
                break
        if args.once or time.time() - started > horizon:
            state = "polling"
            break
        time.sleep(args.interval)
    if state in ("done", "error"):
        record["finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        path.write_text(json.dumps(record, indent=2, default=str))
        if not args.no_download:
            local = download(args.run_id)
            print(stamp(), "outputs in", local, flush=True)
    print("STATE", state, "- reconcile docs/spend.json to the measured seconds", flush=True)
    return 0 if state != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
