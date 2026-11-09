"""Comparable planning horizons; independent of actual benchmark deadlines."""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class CompetitionClock:
    started: float
    sessions: int

    def __post_init__(self):
        if type(self.sessions) is not int or self.sessions < 1:
            raise ValueError("Fleet size must be a positive integer")

    def snapshot(self, now):
        elapsed = max(0.0, now - self.started) * 110 / self.sessions
        return {
            "competition_equivalent_elapsed_seconds": round(elapsed, 1),
            "competition_equivalent_remaining_seconds": round(max(0.0, 32400 - elapsed), 1),
        }

    def metadata(self):
        return {
            "policy": "competition_equivalent_v1",
            "reference_budget_seconds": 32400,
            "reference_sessions": 110,
            "fleet_sessions": self.sessions,
            "elapsed_multiplier": 110 / self.sessions,
            "origin": "shared fleet gameplay start; setup excluded",
            "actual_deadline": "enforced independently by the runner",
        }

    def annotate(self, messages, now):
        """Add fresh request-only feedback without changing persistent history/images."""
        note = (
            "\n\n[RUNTIME PLANNING BUDGET — current for this request]\n"
            "This is a normalized nine-hour, 110-game competition-equivalent planning "
            "budget. Elapsed time advances by 110 / fleet size times actual gameplay "
            "time, including inference and queueing. It is not an environment timer "
            "or the execution cost of an action. Setup is excluded. A benchmark can "
            "stop earlier; its real cutoff is enforced separately.\n"
            + json.dumps(self.snapshot(now), sort_keys=True)
        )
        result = list(messages)
        # Keep reusable prefixes and assistant/tool ordering. The analyzer never
        # stores this note, so the next request cannot inherit a stale snapshot.
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") == "user":
                message = dict(result[i])
                content = message.get("content")
                message["content"] = (
                    [*content, {"type": "text", "text": note}]
                    if isinstance(content, list) else str(content or "") + note
                )
                result[i] = message
                return result
        raise ValueError("Duck request has no user observation for timing feedback")
