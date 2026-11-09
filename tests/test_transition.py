"""Object view, transition summaries and grid-dump suppression (pure, host-side)."""

import numpy as np

from qwen_baselines.objects import component_count, object_view, segment_host
from qwen_baselines.output_filter import suppress_dump
from qwen_baselines.transition import summarize


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


def test_grid_dumps_are_suppressed_and_long_output_truncated():
    grid = board((5, 10, 10, 14, 14))
    hex_rows = "\n".join("".join("0123456789ABCDEF"[v] for v in row) for row in grid)
    filtered, info = suppress_dump("before\n" + hex_rows + "\nafter")
    assert filtered == "before\n[grid dump suppressed: 64 rows; read small crops instead]\nafter"
    assert info == {"suppressed_rows": 64}
    numpy_rows = np.array2string(grid, max_line_width=400, threshold=10**6)
    assert suppress_dump(numpy_rows)[1] == {"suppressed_rows": 64}
    list_rows = repr(grid.tolist()).replace("], [", "],\n[")
    assert suppress_dump(list_rows)[1] == {"suppressed_rows": 64}
    crop = "\n".join("".join("0123456789ABCDEF"[v] for v in row[8:16]) for row in grid[8:16])
    assert suppress_dump(crop) == (crop, None)
    short_rows = "\n".join(["0" * 64] * 10)
    assert suppress_dump(short_rows) == (short_rows, None)
    text, info = suppress_dump("x" * 7000)
    assert text.startswith("x" * 4000) and text.endswith("read small crops instead]")
    assert info == {"truncated_chars": 7000}
    assert suppress_dump("hello\n4096\n") == ("hello\n4096\n", None)
