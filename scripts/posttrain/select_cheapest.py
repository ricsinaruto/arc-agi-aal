"""Cheapest-clear corpus from an on-policy collection (SFT replan, 17 Oct): per level and restored start, the ``k``
cheapest clears by tokens to clear; starts split into train and validation (whole starts, never mixed).

    python select_cheapest.py OUT LEVEL_FOLDER [LEVEL_FOLDER ...] [--k 2] [--train 6] [--val 2]

Each LEVEL_FOLDER holds a gate play's ``level-bench.json`` and ``attempts/<label>/rollout.jsonl``. OUT gets
``train/<name>/`` and ``val/<name>/`` (rollout.jsonl + meta.json), the layout ``rl_sft.sft_specs`` reads, and
``selection.json``."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from arena.rl_single import clear_rhae, clear_tokens  # noqa: E402


def select(rows, k=2, n_train=6, n_val=2, seed=0):
    """``rows``: one level's tries (level-bench attempts). Returns [(split, row, tokens)]: per start (``point``) the
    ``k`` cheapest clears; starts in a seeded order, the first ``n_train`` train, the next ``n_val`` validation. A
    start without a clear is skipped (the next start takes its place)."""
    by = {}
    for r in rows:
        t = clear_tokens(r)
        if t is not None and r.get("outcome") not in ("error", "replay_mismatch"):
            by.setdefault(str(r["point"]), []).append((t, r["label"], r))
    starts = sorted(by)
    random.Random(seed).shuffle(starts)
    out = []
    for i, start in enumerate(starts[: n_train + n_val]):
        split = "train" if i < n_train else "val"
        for t, _, r in sorted(by[start], key=lambda x: (x[0], x[1]))[:k]:
            out.append((split, r, t))
    return out


def scripted(row):
    """Requests replayed from the recording before a restored try's own (0 for a fresh level-1 try)."""
    restore = row.get("restore") or {}
    return int(restore.get("scripted") or restore.get("request") or 0)


def credit_rollout(src, dest, history):
    """Copy a rollout with reply weights: the replayed history ``w`` 0 (no-gradient prefix under credited_only), the
    try's own replies ``w`` 1. Without weights the trainer would train every reply, the recorded game's included."""
    with open(src) as fin, open(dest, "w") as fout:
        for line in fin:
            row = json.loads(line)
            row["w"] = 0.0 if int(row["n"]) < history else 1.0
            fout.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--train", type=int, default=6)
    ap.add_argument("--val", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    summary = {"k": a.k, "train": a.train, "val": a.val, "levels": {}}
    levels = {}  # several folders of one level (disjoint start slices) are pooled
    for folder in map(Path, a.folders):
        bench, partial = folder / "level-bench.json", folder / "level-bench.partial.jsonl"
        rows = (json.loads(bench.read_text())["attempts"] if bench.exists()
                else [json.loads(line) for line in open(partial)])  # a stopped job's finished tries
        for r in rows:
            levels.setdefault(str(r["level"]), []).append({**r, "_folder": str(folder)})
    for level, rows in sorted(levels.items()):
        picked = select(rows, a.k, a.train, a.val, a.seed)
        clears = sorted(t for t in (clear_tokens(r) for r in rows) if t is not None)
        summary["levels"][level] = {"tries": len(rows), "clears": len(clears),
                                    "median_clear": clears[len(clears) // 2] if clears else None,
                                    "picked": {s: [t for s2, _, t in picked if s2 == s] for s in ("train", "val")}}
        for split, r, t in picked:
            folder = Path(r["_folder"])
            name = f"L{level}-{folder.name}-{r['label']}"
            dest = out / split / name
            dest.mkdir(parents=True, exist_ok=True)
            credit_rollout(folder / "attempts" / r["label"] / "rollout.jsonl", dest / "rollout.jsonl", scripted(r))
            meta = {"name": name, "weight": 1.0, "level": int(level), "start": str(r["point"]), "tokens_to_clear": t,
                    "actions": r.get("actions"), "rhae": clear_rhae(r), "source": str(folder)}
            (dest / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "selection.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
