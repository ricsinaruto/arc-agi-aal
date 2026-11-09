"""Persist 30-minute comparison reports and immediate failure notices; never alter jobs."""
import argparse
import concurrent.futures
import datetime
import json
import time
from pathlib import Path

from watch_live import launch_record, call_state, VOLUME


def completed_progress(report):
    """Use the closed scorecard, not a five-minute-old live score, after finalization."""
    games = report.get("games", [])
    elapsed = report["wall_seconds"]
    engine = report.get("engine", {})
    return {"elapsed_seconds": elapsed, "phase": "final", "levels": report["levels"],
            "live": {"score": {"score": report["mean_score_equal_weight_games"],
                               "at_elapsed_seconds": elapsed},
                     "actions": report["actions"],
                     "compactions": sum(g.get("compactions", 0) for g in games),
                     "compaction_seconds": sum(g.get("compaction_seconds", 0) for g in games),
                     "errors": report.get("task_errors", []),
                     "engine": {"tokens_per_second_mean": engine.get("accepted_generation_tokens_per_second"),
                                "preemptions": engine.get("preemptions")}},
            "final_status": report["status"], "end_reason": report.get("end_reason"),
            "scheduler_audit_valid": report.get("scheduler_audit", {}).get("valid")}


def snapshot(run_id):
    import modal
    _, record = launch_record(run_id)
    state = call_state(modal.FunctionCall.from_id(record["function_call_id"]))
    try:
        raw = b"".join(modal.Volume.from_name(VOLUME).read_file(
            f"validation/{run_id}/arena/live/progress.json"))
        progress = json.loads(raw)
    except Exception as exc:
        progress = {"read_error": f"{type(exc).__name__}: {str(exc)[:150]}"}
    if state == "done":
        try:
            raw = b"".join(modal.Volume.from_name(VOLUME).read_file(
                f"validation/{run_id}/arena/trials/duck-transfer/arena.json"))
            progress = completed_progress(json.loads(raw))
        except Exception as exc:
            progress["final_report_error"] = f"{type(exc).__name__}: {str(exc)[:150]}"
    return {"run_id": run_id, "state": state, "progress": progress}


def render(rows, stamp):
    lines = [f"# Campaign report: {stamp}", "",
             "Scores use all 110 roster entries. Each elapsed time is that run's own gameplay clock.", "",
             "| Run | State | Minutes | Score (%) | Clears | Actions | Engine tok/s | Selected gameplay tok/s | KV | Preemptions |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    details = []
    for row in rows:
        p = row["progress"]
        live = p.get("live") or {}
        engine = live.get("engine") or {}
        elapsed = p.get("elapsed_seconds", 0)
        score = (live.get("score") or {}).get("score")
        useful = live.get("gameplay_tokens")
        lines.append(f"| {row['run_id']} | {row['state']} | {elapsed / 60:.1f} | "
                     f"{score if score is not None else 'pending'} | {p.get('levels', '-')} | "
                     f"{live.get('actions', '-')} | {engine.get('tokens_per_second_mean', '-')} | "
                     f"{round(useful / elapsed, 1) if useful is not None and elapsed else '-'} | "
                     f"{engine.get('kv_cache_fraction', '-')} | {engine.get('preemptions', '-')} |")
        details.append(f"- {row['run_id']}: phase={p.get('phase', 'setup')}; "
                       f"started={live.get('started_games', '-')}/110; "
                       f"compactions={live.get('compactions', '-')}; "
                       f"compaction seconds={live.get('compaction_seconds', '-')}; "
                       f"score sampled at={(live.get('score') or {}).get('at_elapsed_seconds', '-')}s.")
        if live.get("errors") or p.get("read_error"):
            details.append(f"  Diagnostic: {live.get('errors') or p.get('read_error')}")
        if p.get("final_status"):
            details.append(f"  Final status={p['final_status']}; end={p.get('end_reason')}; "
                           f"scheduler audit valid={p.get('scheduler_audit_valid')}.")
        if p.get("final_report_error"):
            details.append(f"  Final report unavailable; values above are the last live snapshot: {p['final_report_error']}")
    return "\n".join(lines + ["", "Selected gameplay tokens exclude compactions and losing samples; they do not measure action quality.", "", *details]) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="+")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--interval", type=int, default=1800)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    due, previous = 0, {}
    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(args.run_ids))) as pool:
            rows = list(pool.map(snapshot, args.run_ids))
        states = {r['run_id']: r['state'] for r in rows}
        changed = any(s != 'running' and previous.get(k) != s for k, s in states.items())
        done = all(s != 'running' for s in states.values())
        if time.monotonic() >= due or changed or done or args.once:
            stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
            (args.output / f"{stamp}.json").write_text(json.dumps(rows, indent=2))
            text = render(rows, stamp)
            (args.output / f"{stamp}.md").write_text(text)
            (args.output / "latest.md").write_text(text)
            print(text, flush=True)
            due = time.monotonic() + args.interval
        previous = states
        if done or args.once:
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
