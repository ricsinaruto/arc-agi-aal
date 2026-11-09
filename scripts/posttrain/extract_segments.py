"""Extract verified level-clear segments from the recorded Duck-transfer arena runs (CPU only).

Usage: uv run python scripts/posttrain/extract_segments.py [--source dev|teacher] [--out DIR]
           [--workers 8] [--no-images] [--limit N] [--runs PATTERN ...]

``--source dev`` (default) walks every teacher run (Qwen3.8-Flash-Next-NVFP4 at xhigh) under runs/
and keeps the DEV games of posttrain.split (out: data/posttrain/segments). ``--source teacher`` walks
the generated-game teacher runs (``posttrain.segments.TEACHER_RUNS``; adding a run is one line there)
and keeps the generated games (out: data/posttrain/teacher). Each cleared level's turns are rebuilt
as an SFT chat transcript with the full context the teacher saw, the recorded actions are replayed in
the local engine (environment_files/, or the generated-games store) and the script writes:

* ``segments.jsonl``: verified, unexcluded segments of training games, one per distinct action
  sequence per game level (``sequence_count`` counts the identical copies);
* ``eval_segments.jsonl`` (``--source teacher``): the same for the held-out generated mechanics
  (``split.GENERATED_HOLDOUT``), evaluation only;
* ``rejected.jsonl``: every other segment, without messages, with its verification or exclusion
  reason;
* ``stats.json``: per game and level counts, actions vs baseline, scores and tokens.

See ``posttrain.segments`` for the log schema and what the transcripts can and cannot reproduce.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from posttrain import segments  # noqa: E402


def work(args):
    game_dir, info, environments_dir, images = args
    try:
        return segments.process_game(game_dir, info, environments_dir=environments_dir, images=images)
    except Exception as exc:  # noqa: BLE001 - one broken log never stops the extraction
        return {"segments": [], "prompt_check": {}, "error": f"{game_dir}: {type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--source", choices=("dev", "teacher"), default="dev")
    parser.add_argument("--out", default=None,
                        help="default data/posttrain/segments (dev) or data/posttrain/teacher (teacher)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-images", action="store_true",
                        help="omit the per-turn board image parts (the text label stays)")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N games")
    parser.add_argument("--runs", nargs="*", default=None,
                        help="glob patterns relative to --root (default: posttrain.segments.RUN_PATTERNS "
                             "for dev, TEACHER_RUNS for teacher)")
    args = parser.parse_args()
    root = Path(args.root)
    teacher = args.source == "teacher"
    out = Path(args.out or ROOT / ("data/posttrain/teacher" if teacher else "data/posttrain/segments"))
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    patterns = tuple(args.runs) if args.runs else (segments.TEACHER_RUNS if teacher else segments.RUN_PATTERNS)
    games, skipped = segments.teacher_games(root, patterns, games="generated" if teacher else "dev")
    if args.limit:
        games = games[:args.limit]
    environments_dir = str(segments.environment_dir(root))
    jobs = [(str(game_dir), info, environments_dir, not args.no_images) for game_dir, info in games]
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result.get("error"):
                errors.append(result["error"])
            results.append(result)
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} games", flush=True)
    results.sort(key=lambda r: r.get("game_id") or "")
    kept, rejected, stats, held = segments.finalize(results, skipped, segments.tools_schema(), with_eval=True)
    if held and not teacher:
        raise SystemExit(f"held-out generated games in a dev extraction: {sorted({r['game_id'] for r in held})}")
    stats["source"] = args.source
    stats["run_patterns"] = list(patterns)
    stats["errors"] = errors
    stats["runs"] = sorted({info["run"] for _, info in games})
    stats["request_caps"] = {info["run"]: [info.get("request_cap"), info.get("request_cap_source")]
                             for _, info in games}
    stats["images"] = not args.no_images
    stats["seconds"] = round(time.monotonic() - started, 1)
    with open(out / "segments.jsonl", "w") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if teacher:
        with open(out / "eval_segments.jsonl", "w") as handle:
            for record in held:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(out / "rejected.jsonl", "w") as handle:
        for record in rejected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    totals = stats["totals"]
    print(f"{len(games)} games -> {totals['segments']} segments, {totals['verified']} verified, "
          f"{totals['usable_verified']} usable, {totals['unique_verified']} unique; "
          f"{len(errors)} errors; {stats['seconds']} s")


if __name__ == "__main__":
    main()
