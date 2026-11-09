"""Answer the pending relay turn (posttrain.relay) and optionally wait for the next one.

  uv run python scripts/posttrain/relay_answer.py OUT --show            # print the pending view
  uv run python scripts/posttrain/relay_answer.py OUT --reasoning-file r.txt --code-file c.py [--wait]
  uv run python scripts/posttrain/relay_answer.py OUT --reasoning "..." --code "..." [--content "..."] [--wait]

With --wait it blocks until the next turn is pending (or the game is done) and prints its view.
"""

import argparse
import json
import sys
import time
from pathlib import Path


def pending(relay):
    marker = relay / "PENDING"
    return int(marker.read_text()) if marker.exists() else None


def show(relay, timeout):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if (relay / "DONE").exists():
            print("GAME DONE", (relay / "DONE").read_text().strip())
            return
        turn = pending(relay)
        if turn is not None and not (relay / f"turn-{turn:04d}.response.json").exists():
            print((relay / f"turn-{turn:04d}.view.txt").read_text())
            return
        time.sleep(0.5)
    print("no pending turn yet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)
    for name in ("reasoning", "content", "code"):
        parser.add_argument(f"--{name}")
        parser.add_argument(f"--{name}-file")
    args = parser.parse_args()
    relay = Path(args.out) / "relay"
    if args.show:
        show(relay, args.timeout)
        return
    turn = pending(relay)
    if turn is None:
        sys.exit("no pending turn")
    answer = {}
    for name in ("reasoning", "content", "code"):
        value = getattr(args, name)
        path = getattr(args, f"{name}_file")
        if path:
            value = Path(path).read_text()
        if value is not None:
            answer[name] = value
    if not answer.get("reasoning"):
        sys.exit("--reasoning is required (the student learns from it)")
    (relay / f"turn-{turn:04d}.response.json").write_text(json.dumps(answer, ensure_ascii=False, indent=1))
    print(f"answered turn {turn}")
    if args.wait:
        time.sleep(1.0)
        show(relay, args.timeout)


if __name__ == "__main__":
    main()
