"""Exact, game-agnostic observations; all coordinates use x=column, y=row."""

import hashlib
from dataclasses import dataclass

import numpy as np

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


_HEX_TABLE = bytes.maketrans(bytes(range(16)), HEX.encode())


def encode_grid(board):
    """Rows as hex strings (v7.1: one table lookup per board instead of a python loop per cell)."""
    try:
        array = np.asarray(board)
        if array.ndim == 2 and array.size and array.dtype.kind in "iu" and 0 <= array.min() and array.max() < 16:
            data = array.astype(np.uint8).tobytes().translate(_HEX_TABLE).decode()
            width = array.shape[1]
            return [data[i:i + width] for i in range(0, len(data), width)]
    except (TypeError, ValueError):
        pass
    return ["".join(HEX[int(c)] for c in row) for row in board]


def board_hash(board):
    data = np.asarray(board, dtype=np.uint8).tobytes()
    if len(data) > 4096:
        return hashlib.blake2s(data, digest_size=12).hexdigest()
    # A 4 KiB board releases the GIL when hashed at once; reacquiring it under
    # rendering contention costs milliseconds for a hash that takes microseconds.
    digest = hashlib.blake2s(data[:2047], digest_size=12)
    for offset in range(2047, len(data), 2047):
        digest.update(data[offset:offset + 2047])
    return digest.hexdigest()
