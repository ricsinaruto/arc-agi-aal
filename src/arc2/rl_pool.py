"""The RL task pool (docs/arc2-arm.md, "RL with a verifiable reward"): ARC tasks with held-out pairs, never a task we
score on. ``python -m arc2.rl_pool --raw DIR --exclude-ids h240.json --out pool.jsonl``.

Sources under ``--raw``: ``ARC-AGI/data/{training,evaluation}`` (ARC-AGI-1) and ``ARC-AGI-2/data/{training,evaluation}``.
Excluded: every ARC-AGI-2 evaluation task (c240 is built from them), the ids in ``--exclude-ids`` (h240), and any task
that shares a grid (input or output of any pair) with an excluded task. Duplicates across sources (ARC-AGI-2 training
repeats 767 ARC-AGI-1 ids, some edited) are kept once: by id, the ARC-AGI-2 version; then by the hash of all grids.

A pool row: ``{"id", "source", "train": [...], "holdout": [...]}``; ``holdout`` pairs (the task's test pairs, with their
outputs) are never shown to the model and decide the reward (``reward``)."""

import argparse
import hashlib
import json
from pathlib import Path

SOURCES = (("arc2-train", "ARC-AGI-2/data/training"), ("arc1-train", "ARC-AGI/data/training"),
           ("arc1-eval", "ARC-AGI/data/evaluation"))  # ARC-AGI-2 first: an ARC-AGI-1 task it reuses (by id) is kept once
EXCLUDED_SOURCE = "ARC-AGI-2/data/evaluation"


def grid_hash(grid):
    return hashlib.sha256(json.dumps(grid, separators=(",", ":")).encode()).hexdigest()[:20]


def task_grids(task):
    for pair in task["train"] + task["test"]:
        yield pair["input"]
        if "output" in pair:
            yield pair["output"]


def build(raw, exclude_ids=()):
    raw = Path(raw)
    excluded = {}
    for path in sorted((raw / EXCLUDED_SOURCE).glob("*.json")):
        excluded[path.stem] = json.loads(path.read_text())
    for source, folder in SOURCES:  # h240 ids are ARC-AGI-2 training tasks
        for path in sorted((raw / folder).glob("*.json")):
            if path.stem in exclude_ids:
                excluded[path.stem] = json.loads(path.read_text())
    banned = {grid_hash(g) for task in excluded.values() for g in task_grids(task) if len(g) * len(g[0]) > 4}
    rows, seen, dropped, ids = [], set(), {"shared_grid": 0, "duplicate": 0, "excluded_id": 0}, set()
    for source, folder in SOURCES:
        for path in sorted((raw / folder).glob("*.json")):
            if path.stem in ids:
                dropped["duplicate"] += 1
                continue
            ids.add(path.stem)
            if path.stem in excluded:
                dropped["excluded_id"] += 1
                continue
            task = json.loads(path.read_text())
            hashes = [grid_hash(g) for g in task_grids(task)]
            if any(h in banned for g, h in zip(task_grids(task), hashes) if len(g) * len(g[0]) > 4):
                dropped["shared_grid"] += 1
                continue
            key = hashlib.sha256("".join(hashes).encode()).hexdigest()
            if key in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(key)
            rows.append({"id": f"{source}:{path.stem}", "source": source, "train": task["train"],
                         "holdout": task["test"]})
    return rows, {"rows": len(rows), "excluded_tasks": len(excluded), "dropped": dropped,
                  "by_source": {s: sum(r["source"] == s for r in rows) for s, _ in SOURCES}}


def reward(row, program_outputs):
    """1.0 when a program's outputs on (shown training inputs + held-out inputs), in that order, match every one
    exactly; else 0.0. ``program_outputs``: ``judge``-style list (None or a dict for an invalid output)."""
    expected = [p["output"] for p in row["train"]] + [p["output"] for p in row["holdout"]]
    return float(len(program_outputs) == len(expected) and all(o == e for o, e in zip(program_outputs, expected)))


def model_view(row):
    """The task as the harness shows it: training pairs, test inputs only."""
    return {"train": row["train"], "test": [{"input": p["input"]} for p in row["holdout"]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--exclude-ids", help="JSON list of task ids (h240)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    ids = set(json.loads(Path(args.exclude_ids).read_text())) if args.exclude_ids else set()
    rows, stats = build(args.raw, ids)
    with open(args.out, "w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
