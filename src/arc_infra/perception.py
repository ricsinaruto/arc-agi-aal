"""Exact, game-agnostic observations; all coordinates use x=column, y=row."""

import hashlib
from dataclasses import dataclass

import numpy as np

PALETTE = [
    "FFFFFF",
    "CCCCCC",
    "999999",
    "666666",
    "333333",
    "000000",
    "E53AA3",
    "FF7BCC",
    "F93C31",
    "1E93FF",
    "88D8F1",
    "FFDC00",
    "FF851B",
    "921231",
    "4FCC30",
    "A356D6",
]
HEX = "0123456789abcdef"


@dataclass
class Observation:
    frames: list[np.ndarray]
    levels: int
    total_levels: int
    state: str
    available: list[str]
    full_reset: bool = False

    @property
    def board(self):
        return self.frames[-1]

    @classmethod
    def from_raw(cls, raw, previous=None):
        if raw is None:
            raise RuntimeError("Environment returned no observation; action outcome is unknown")
        frames = [np.array(frame, dtype=np.uint8, copy=True) for frame in raw.frame]
        if not frames and previous is not None:
            frames = [previous.board.copy()]
        if not frames or any(f.ndim != 2 or f.shape != (64, 64) for f in frames):
            raise ValueError("Expected one or more 64x64 ARC frames")
        if any(np.any(f > 15) for f in frames):
            raise ValueError("ARC frame contains an invalid color")
        return cls(
            frames,
            int(raw.levels_completed),
            int(raw.win_levels),
            getattr(raw.state, "name", str(raw.state)),
            ["RESET" if int(a) == 0 else f"ACTION{int(a)}" for a in raw.available_actions],
            bool(raw.full_reset),
        )

    def payload(self):
        return {
            "frames": [encode_grid(f) for f in self.frames],
            "levels": self.levels,
            "total_levels": self.total_levels,
            "state": self.state,
            "available": self.available,
            "full_reset": self.full_reset,
        }


def encode_grid(board):
    return ["".join(HEX[int(c)] for c in row) for row in board]


def decode_grid(rows):
    return np.array([[int(c, 16) for c in row] for row in rows], dtype=np.uint8)


def board_hash(board):
    return hashlib.blake2s(np.asarray(board, dtype=np.uint8).tobytes(), digest_size=12).hexdigest()
