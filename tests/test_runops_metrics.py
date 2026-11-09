import json

from arc_infra.runops import metrics


def test_writer_rows_rounding_cost_monotone_and_commit(tmp_path):
    calls = []
    w = metrics.MetricsWriter(tmp_path / "r.jsonl", commit=lambda: calls.append(1), commit_s=0)
    w.emit({"infra.gen_tok_s": 1234.56789, "skip": None, "flag": True, "infra.cost_usd": 5.0}, t=1)
    w.emit({"infra.cost_usd": 4.0}, t=2)  # a lower sample never steps the total back
    assert w.emit({}) is None
    w.alert("stall", **{"infra.running": 0})
    rows = metrics.read_rows(tmp_path / "r.jsonl")
    assert rows[0] == {"t": 1.0, "m": {"infra.gen_tok_s": 1234.57, "infra.cost_usd": 5.0}}
    assert rows[1]["m"]["infra.cost_usd"] == 5.0
    assert rows[2]["alert"] == "stall" and len(calls) == 3
    w2 = metrics.MetricsWriter(tmp_path / "r.jsonl")  # a restarted exporter keeps the floor
    w2.emit({"infra.cost_usd": 1.0}, t=3)
    assert metrics.read_rows(tmp_path / "r.jsonl")[-1]["m"]["infra.cost_usd"] == 5.0


def test_meta_link_stale(tmp_path):
    metrics.meta("a", arm="rl", what="leg a", cards=2, root=tmp_path)
    metrics.link("b", "a", stage="v6b", root=tmp_path)
    a = json.loads((tmp_path / "a.meta.json").read_text())
    b = json.loads((tmp_path / "b.meta.json").read_text())
    assert a["status"] == "ended"
    assert b == {"run": "b", "status": "live", "continues": ["a"], "stage": "v6b", "arm": "rl", "what": "leg a",
                 "cards": 2}
    metrics.meta("b", continues=["a"], root=tmp_path)  # never duplicated
    assert json.loads((tmp_path / "b.meta.json").read_text())["continues"] == ["a"]
    metrics.MetricsWriter(tmp_path / "b.jsonl").emit({"x": 1}, t=100)
    assert metrics.stale("b", now=200, limit=1800, root=tmp_path) is None
    assert metrics.stale("b", now=2000, limit=1800, root=tmp_path) == 1900
    assert metrics.stale("a", now=99999, root=tmp_path) is None  # ended
