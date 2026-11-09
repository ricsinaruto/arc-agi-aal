"""Duck's segmentation helper staged for the Qwen sandboxes."""

import numpy as np
import pytest

from qwen_baselines import segmentation


@pytest.fixture
def segment(tmp_path):
    path = segmentation.write(tmp_path / "segmentation.py")
    namespace = {"__name__": "segmentation"}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace["segment"]


def test_staged_source_is_the_public_duck_module(tmp_path):
    public = segmentation.SOURCE.read_text()
    assert segmentation.SOURCE.parts[-3:] == ("inference", "utils", "segmentation.py")
    text = segmentation.write(tmp_path / "s.py").read_text()
    assert public in text
    assert text.startswith('"""Board segmentation helper') and "def segment_layer" in text
    assert "import hashlib" in text and "numpy" not in text.split("def segment_layer")[1]


def test_segment_int_board_nodes_bbox_children_and_hashes(segment):
    board = np.zeros((64, 64), dtype=np.uint8)
    board[10:16, 20:26] = 3  # a 6x6 tile
    board[12:14, 22:24] = 5  # a 2x2 dot inside it
    board[40:46, 50:56] = 3  # an identical tile elsewhere
    result = segment(board)
    nodes = result["nodes"]
    assert [n["id"] for n in nodes] == list(range(len(nodes)))
    tiles = [n for n in nodes if n["color"] == 3]
    dot = next(n for n in nodes if n["color"] == 5)
    assert dot["bbox"] == [12, 22, 14, 24] and dot["pixels"] == 4
    assert [t["bbox"] for t in tiles] == [[10, 20, 16, 26], [40, 50, 46, 56]]
    assert tiles[0]["pixels"] == 32 and tiles[1]["pixels"] == 36
    assert tiles[0]["children"] == [dot["id"]] and tiles[1]["children"] == []
    assert tiles[0]["hash"] != tiles[1]["hash"]
    same = segment(np.roll(board, 5, axis=1))
    pieces = lambda result: {n["hash"] for n in result["nodes"] if n["color"]}  # noqa: E731
    assert pieces(same) == pieces(result) and len(pieces(result)) == 3
    assert [tiles[0]["id"], dot["id"]] in result["adjacency_list"]
    assert all(isinstance(p, list) and len(p) == 2 for n in nodes for p in n["boundary"])
    assert segment(board.tolist())["nodes"][1]["bbox"] == tiles[0]["bbox"]


def test_segment_ascii_board_and_validation(segment):
    rows = ["...." for _ in range(4)]
    rows[1] = ".##."
    rows[2] = ".##."
    result = segment(rows)
    block = next(n for n in result["nodes"] if n["color"] == "#")
    assert block["bbox"] == [1, 1, 3, 3] and block["pixels"] == 4
    assert segment("\n".join(rows) + "\n")["nodes"][1]["color"] == "#"
    with pytest.raises(ValueError, match="equal length"):
        segment(["ab", "a"])
    with pytest.raises(ValueError, match="0..15"):
        segment([[16]])
    with pytest.raises(ValueError, match="non-empty"):
        segment([])
