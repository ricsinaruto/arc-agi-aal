"""SFT replan pieces: cheapest-clear selection by start (scripts/posttrain/select_cheapest.py) and the validation
early stop (arena.rl_sft.EarlyStop)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/posttrain"))
import select_cheapest as sc  # noqa: E402
from arena.rl_sft import EarlyStop  # noqa: E402


def row(label, point, tokens, outcome="cleared"):
    return {"label": label, "point": point, "outcome": outcome, "level": 2,
            "level_rows": [{"tokens": tokens, "score": 115.0}] if outcome == "cleared" else []}


def test_two_cheapest_clears_per_start_and_whole_start_splits():
    rows = []
    for s in range(5):
        for i, t in enumerate([900, 300, 500, 100 + s, 700]):
            rows.append(row(f"a{s}{i}", f"p{s}", t))
        rows.append(row(f"f{s}", f"p{s}", 0, outcome="budget"))  # unfinished: never picked
    picked = sc.select(rows, k=2, n_train=3, n_val=2, seed=0)
    assert len(picked) == 10
    by = {}
    for split, r, t in picked:
        by.setdefault(r["point"], set()).add(split)
        assert t in (300, 100 + int(r["point"][1:]))  # the two cheapest of that start
    assert all(len(v) == 1 for v in by.values())  # a start is train or validation, never both
    assert sum(1 for v in by.values() if v == {"train"}) == 3 and sum(1 for v in by.values() if v == {"val"}) == 2


def test_starts_without_clears_are_skipped_and_errors_ignored():
    rows = [row("a", "p0", 0, outcome="budget"), row("b", "p1", 400), row("c", "p1", 200, outcome="replay_mismatch"),
            row("d", "p2", 300)]
    picked = sc.select(rows, k=2, n_train=1, n_val=1, seed=1)
    assert sorted(r["label"] for _, r, _ in picked) == ["b", "d"]


def test_early_stop_keeps_the_best_and_stops_after_patience():
    e = EarlyStop(2)
    assert [e.see(s, v) for s, v in [(4, 0.50), (8, 0.40), (12, 0.41), (16, 0.39), (20, 0.40), (24, 0.42)]] == \
        ["best", "best", "wait", "best", "wait", "stop"]
    assert (e.best, e.best_step) == (0.39, 16)


def test_credit_marks_the_replayed_history_no_grad(tmp_path):
    src = tmp_path / "r.jsonl"
    src.write_text("".join(json.dumps({"n": i, "usage": {"completion_tokens": 10}}) + "\n" for i in range(5)))
    sc.credit_rollout(src, tmp_path / "out.jsonl", sc.scripted({"restore": {"scripted": 3, "request": 3}}))
    assert [json.loads(l)["w"] for l in open(tmp_path / "out.jsonl")] == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert sc.scripted({"restore": None}) == 0  # a fresh level-1 try: every reply is its own
