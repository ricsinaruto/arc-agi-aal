"""Exact observations, lossless grid encoding and validated actions."""

import hashlib

import numpy as np
import pytest
from conftest import observation, raw

from arc_infra.perception import Observation, board_hash, decode_grid, encode_grid
from arc_infra.plans import Action, integer


def test_grid_encoding_is_lossless_and_hash_tracks_content():
    board = np.full((64, 64), 5, dtype=np.uint8)
    board[3:5, 10:13] = 0
    assert np.array_equal(decode_grid(encode_grid(board)), board)
    after = board.copy()
    after[9, 20] = 14
    assert board_hash(board) != board_hash(after) and board_hash(board) == board_hash(board.copy())


@pytest.mark.parametrize("shape", [(0, 0), (1, 1), (31, 66), (32, 64), (64, 64), (100, 100)])
def test_board_hash_preserves_existing_digests(shape):
    board = np.arange(shape[0] * shape[1], dtype=np.uint8).reshape(shape)
    for value in (board, board.T, board[:, ::2], board.tolist(), board.astype(np.int64)):
        expected = hashlib.blake2s(np.asarray(value, dtype=np.uint8).tobytes(), digest_size=12).hexdigest()
        assert board_hash(value) == expected


def test_observation_uses_settled_animation_and_copies_arrays():
    obs = observation()
    obs.frames.append(np.full((64, 64), 3, dtype=np.uint8))
    converted = Observation.from_raw(raw(obs))
    obs.frames[-1][0, 0] = 8
    assert converted.board[0, 0] == 3
    assert len(converted.frames) == 2
    assert converted.available == ["ACTION1", "ACTION4", "ACTION6"]
    assert converted.payload()["frames"][1][0][0] == "3"


@pytest.mark.parametrize("value", [-1, 64, True, 3.5, "3"])
def test_click_coordinates_are_validated(value):
    with pytest.raises(ValueError):
        integer(value, 0, 63, "col")
    assert Action("ACTION6", 3, 4).payload() == {"action": "ACTION6", "x": 3, "y": 4}
    assert Action("ACTION1").payload() == {"action": "ACTION1"}
