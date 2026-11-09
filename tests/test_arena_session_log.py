"""v7.1 buffered Workspace log."""

import json

from arc_infra.session import Workspace


def test_the_log_is_buffered_and_flushed_on_interval_finish_and_flush(tmp_path, monkeypatch):
    monkeypatch.setattr(Workspace, "FLUSH_SECONDS", 3600)
    monkeypatch.setattr(Workspace, "CURRENT_SECONDS", 3600)
    ws = Workspace(tmp_path / "g")
    ws.append("first")  # the first event flushes (the clock starts at 0)
    ws.append("a", x=1)
    assert [json.loads(x)["kind"] for x in ws.log.read_text().splitlines()] == ["first"]
    ws.append("finished", status="won")
    assert [json.loads(x)["kind"] for x in ws.log.read_text().splitlines()] == ["first", "a", "finished"]

    class Obs:
        def __init__(self, n):
            self.n = n

        def payload(self):
            return {"n": self.n}

    ws.observe(1, Obs(1))
    ws.observe(2, Obs(2))
    assert json.loads((tmp_path / "g/current.json").read_text())["n"] == 1
    Workspace.flush_all()
    assert json.loads((tmp_path / "g/current.json").read_text())["n"] == 2
    assert ws.log.read_text().count('"observation"') == 2


def test_the_heap_keeper_freezes_on_its_interval_and_records_gen2_pauses():
    import gc

    from arena.heap import HeapKeeper

    keeper = HeapKeeper(600)
    try:
        keeper.tick(10)
        assert keeper.stats["freezes"] == 0
        keeper.tick(600)
        assert keeper.stats["freezes"] == 1 and keeper.stats["frozen"] == gc.get_freeze_count() > 0
        keeper.tick(900)
        assert keeper.stats["freezes"] == 1
        gc.collect()
        assert keeper.summary()["gen2"] >= 1
    finally:
        keeper.close()
        gc.unfreeze()
