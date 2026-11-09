"""The game's frames and history (``DuckAgent.duck_history``) and the state file written at start and end."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference.utils.grid_utils import format_grid_ascii


@dataclass(frozen=True)
class Frame:
    grid: tuple[tuple[int, ...], ...]
    step: int
    level: int

    @property
    def shape(self) -> tuple[int, int]:
        rows = len(self.grid)
        cols = max((len(row) for row in self.grid), default=0)
        return rows, cols

    @property
    def ascii(self) -> str:
        return format_grid_ascii(self.grid)

    def __str__(self) -> str:
        rows, cols = self.shape
        return (
            f"Level: {self.level}\n"
            f"Step: {self.step}\n"
            f"Grid shape: {rows} x {cols}\n"
            f"Grid contents:\n{self.ascii}"
        )


@dataclass(frozen=True)
class HistoryEntry:
    action: str
    frame: Frame


def frame_to_payload(frame: Frame | None) -> dict[str, Any] | None:
    if frame is None:
        return None
    return {
        "grid": [list(row) for row in frame.grid],
        "step": frame.step,
        "level": frame.level,
    }


def history_entry_to_payload(entry: HistoryEntry) -> dict[str, Any]:
    return {
        "action": entry.action,
        "frame": frame_to_payload(entry.frame),
    }


def write_runtime_state(
    path: Path,
    *,
    current_frame: Frame | None,
    history: list[HistoryEntry],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    # A long game must not duplicate every grid and a pretty-printed JSON string at shutdown.
    with tmp_path.open("w", encoding="utf-8", buffering=1 << 16) as output:
        output.write('{"current_frame":')
        output.write(json.dumps(frame_to_payload(current_frame), separators=(",", ":")))
        output.write(',"history":[')
        for index, entry in enumerate(history):
            if index:
                output.write(",")
            output.write(json.dumps(history_entry_to_payload(entry), separators=(",", ":")))
        output.write("]}")
    tmp_path.replace(path)
