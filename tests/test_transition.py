"""Object view and transition summaries (pure, host-side)."""

import io
import pickle
import threading

import numpy as np
import pytest

from arc_infra.objects import component_count, object_view, segment_host
from arc_infra.transition import summarize


def test_segment_frames_survive_partial_pipe_reads_and_writes():
    from arc_infra.segment_worker import _read, _write

    class FragmentedPipe(io.BytesIO):
        def read(self, size=-1):
            return super().read(min(size, 7))

        def write(self, data):
            return super().write(data[:13])

    pipe = FragmentedPipe()
    replies = [("ok", {"payload": "x" * 70_000}), ("error", "worker failure")]
    for reply in replies:
        _write(pipe, reply)
    pipe.seek(0)
    assert [_read(pipe), _read(pipe)] == replies
    with pytest.raises(EOFError, match="segment worker closed"):
        _read(pipe)


@pytest.mark.parametrize("cut", [0, 2, -1])
def test_segment_pipe_truncation_is_not_a_valid_reply(cut):
    from arc_infra.segment_worker import HEADER, _read

    data = pickle.dumps(("ok", {"nodes": []}))
    with pytest.raises(EOFError, match="segment worker closed"):
        _read(io.BytesIO((HEADER.pack(len(data)) + data)[:cut]))


def test_segment_fallback_is_reported_and_sticks(monkeypatch, caplog):
    from arc_infra import transition

    class BrokenPool:
        calls = 0

        def segment(self, board, limit):
            self.calls += 1
            raise EOFError("segment worker closed")

    pool = BrokenPool()
    monkeypatch.setattr(transition, "SEGMENT_WORKERS", 1)
    monkeypatch.setattr(transition, "_POOL", {"pool": pool, "failed": None})
    grid = np.zeros((64, 64), dtype=np.uint8)
    first = transition.segment_offloaded(grid, limit=200)
    assert transition.segment_offloaded(grid, limit=200) == first
    assert pool.calls == 1
    assert "Segmentation pool failed; using in-process fallback: EOFError" in caplog.text


def test_segment_worker_failure_wakes_waiting_callers(monkeypatch):
    from arc_infra import segment_worker

    entered, release = threading.Event(), threading.Event()

    class FailedWorker:
        stdin = io.BytesIO()
        stdout = None

        def kill(self):
            pass

    def fail_read(stream):
        entered.set()
        assert release.wait(2)
        raise EOFError("worker exited")

    pool = segment_worker.SegmentPool(1)
    monkeypatch.setattr(pool, "_spawn", FailedWorker)
    monkeypatch.setattr(segment_worker, "_read", fail_read)
    waiting = threading.Event()
    wait = pool.lock.wait

    def note_wait(*args, **kwargs):
        waiting.set()
        return wait(*args, **kwargs)

    monkeypatch.setattr(pool.lock, "wait", note_wait)
    errors = []

    def call():
        try:
            pool.segment(np.zeros((64, 64), dtype=np.uint8), 200)
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=call, daemon=True)
    second = threading.Thread(target=call, daemon=True)
    try:
        first.start()
        assert entered.wait(2)
        second.start()
        assert waiting.wait(2)
        release.set()
        first.join(2)
        second.join(2)
        assert not first.is_alive() and not second.is_alive()
        assert {type(error) for error in errors} == {EOFError, RuntimeError}
        assert all("worker exited" in str(error) for error in errors)
        with pytest.raises(RuntimeError, match="worker exited"):
            pool._take()
    finally:
        release.set()
        pool.close()


def board(*blocks, fill=0):
    grid = np.full((64, 64), fill, dtype=np.uint8)
    for colour, r0, c0, r1, c1 in blocks:
        grid[r0:r1, c0:c1] = colour
    return grid


def test_object_view_lists_largest_objects_and_skips_noise():
    grid = board((5, 10, 10, 14, 14), (5, 30, 30, 34, 34), (7, 2, 40, 3, 60))
    objects = segment_host(grid)
    assert component_count(grid) == 4 and len(objects["nodes"]) == 4
    view = object_view(objects, limit=2)
    lines = view.split("\n")
    assert lines[0] == "objects: 4 (by colour: 0x1 5x2 7x1)"
    assert lines[1].startswith("#0 c=0 px=4044 bbox=[0, 0, 64, 64]") and "kids=" in lines[1]
    assert lines[2].startswith("#") and "c=7 px=20 bbox=[2, 40, 3, 60]" in lines[2]
    assert lines[3] == "... 2 smaller objects omitted" and "ACTION6" in lines[4]
    checker = (np.indices((64, 64)).sum(0) % 2).astype(np.uint8)
    skipped = segment_host(checker)
    assert skipped["skipped"] == 4096 and "too many to list" in object_view(skipped)
    assert object_view(None) == "objects: unavailable"


def test_transition_summary_matches_objects_by_hash():
    before = board((5, 10, 10, 14, 14), (7, 2, 40, 3, 60))
    after = board((5, 20, 10, 24, 14), (9, 50, 50, 52, 52))
    summary = summarize(
        before, after, before_objects=segment_host(before), after_objects=segment_host(after), cells=2
    )
    assert summary["changed_cells"] == 16 + 16 + 20 + 4
    assert summary["change_bbox"] == [2, 10, 52, 60] and len(summary["cells"]) == 2
    objects = summary["objects"]
    assert objects["counts"] == {"appeared": 1, "disappeared": 1, "moved": 1}
    assert objects["background_reshaped"]  # the all-0 region changed shape, not listed
    assert objects["moved"][0]["from"] == [10, 10, 14, 14]
    assert objects["moved"][0]["bbox"] == [20, 10, 24, 14] and objects["moved"][0]["color"] == 5
    assert objects["disappeared"][0]["color"] == 7 and objects["appeared"][0]["color"] == 9
    assert summarize(None, after)["note"] == "initial observation"
    same = summarize(after, after, before_objects=segment_host(after), after_objects=segment_host(after))
    assert same["changed_cells"] == 0 and same["change_bbox"] is None
    assert same["objects"]["counts"] == {"appeared": 0, "disappeared": 0, "moved": 0}
    assert not same["objects"]["background_reshaped"]
    assert "objects" not in summarize(before, after, before_objects={"skipped": 5000}, after_objects=segment_host(after))
