"""One serialized, authoritative action boundary per game."""

import asyncio

import numpy as np
from arcengine import GameAction

from arc_infra.perception import board_hash

from . import objects as objects_
from . import transition


class Bridge:
    def __init__(
        self,
        env,
        archive,
        clock,
        label="",
        *,
        boundaries=True,
        objects=False,
        actions_per_turn=None,
        reject_repeats=False,
    ):
        self.env, self.archive, self.clock = env, archive, clock
        self.label = label
        # PRO-LONG (baseline P) keeps draining its plan across level clears, resets and
        # GAME_OVER (after the automatic RESET), so its runner manages boundaries itself.
        self.boundaries = boundaries
        # v4 A/B/C mechanisms: segmented transition summaries in every observation, a
        # per-response action cap and rejection of repeats that produced no board change.
        self.objects = objects
        self.actions_per_turn = actions_per_turn
        self.reject_repeats = reject_repeats
        self.turn_actions = 0
        self.no_change = set()
        self.lock = asyncio.Lock()
        self.actions, self.attempt = 0, 0
        self.stopped = False
        self.unknown = False
        self.clears = []

    def repeat_key(self, action, x, y):
        return (board_hash(self.archive.frame()), action, x, y)

    def record(self, raw, action=None, *, initial=False):
        if raw is None:
            raise RuntimeError("Action outcome unknown: no observation returned")
        frames = [np.array(f, copy=True) for f in raw.frame]
        if not frames or any(
            f.shape != (64, 64) or np.any(f > 15) or np.any(f < 0) for f in frames
        ):
            raise ValueError("Expected nonempty 64x64 frames with color IDs 0..15")
        frames = [f.astype(np.uint8) for f in frames]
        previous = self.archive.metadata() if self.archive.count() else None
        levels = int(raw.levels_completed)
        state = getattr(raw.state, "name", str(raw.state))
        reset = action is not None and action["action"] == "RESET"
        if reset:
            self.attempt += 1
        clear = previous is not None and levels > previous["levels_completed"]
        outcome = {
            "executed": not initial,
            "level_completed": clear,
            "game_over": state == "GAME_OVER",
            "done": state == "WIN",
            "reset": reset,
            "board_changed": None
            if previous is None
            else not np.array_equal(self.archive.frame(), frames[-1]),
        }
        if clear:
            self.clears.append(
                {
                    "levels_completed": levels,
                    "seconds": self.clock.elapsed(),
                    "actions": self.actions,
                }
            )
        extra = {}
        if self.objects:
            after_objects = objects_.segment_host(frames[-1])
            extra["transition"] = transition.summarize(
                self.archive.frame() if previous is not None else None,
                frames[-1],
                before_objects=self.archive.objects() if previous is not None else None,
                after_objects=after_objects,
            )
        repeat = False
        if previous is not None and action is not None and not reset:
            key = self.repeat_key(action["action"], action.get("x"), action.get("y"))
            repeat = key in self.no_change
            if clear or outcome["game_over"]:
                self.no_change.clear()
            elif outcome["board_changed"] is False:
                self.no_change.add(key)
        elif reset:
            self.no_change.clear()
        if self.objects:
            extra["transition"].update(
                board_changed=outcome["board_changed"], level_completed=clear, no_change_repeat=repeat
            )
        meta = self.archive.append(
            {
                "action_id": None if initial else self.actions,
                "action": action,
                "previous_observation_id": previous["observation_id"] if previous else None,
                "status": state,
                "full_reset": bool(getattr(raw, "full_reset", False)),
                "legal_actions": [
                    f"ACTION{int(a)}" if int(a) else "RESET" for a in raw.available_actions
                ],
                "levels_completed": levels,
                "win_levels": int(raw.win_levels),
                "actions_taken": self.actions,
                "attempt": self.attempt,
                "outcome": outcome,
                **outcome,
                "elapsed_seconds": self.clock.elapsed(),
                **extra,
            },
            frames,
        )
        if self.objects:
            self.archive.objects_cache[meta["index"]] = after_objects
        self.stopped = self.boundaries and (
            clear or outcome["game_over"] or outcome["done"] or reset
        )
        if not initial:
            self.progress(action, meta, clear)
        return meta

    def progress(self, action, meta, clear):
        """Live progress line for notebook/console logs, like the Duck harness prints."""
        name = action["action"]
        if name == "ACTION6":
            name += f"(x={action['x']},y={action['y']})"
        flags = []
        if clear:
            flags.append("LEVEL CLEARED")
        if meta["done"]:
            flags.append("GAME WON")
        if meta["game_over"]:
            flags.append("GAME OVER")
        if meta["reset"]:
            flags.append("reset")
        print(
            f"[{self.label}] action {self.actions} {name} -> levels "
            f"{meta['levels_completed']}/{meta['win_levels']} {meta['status']} "
            f"elapsed {meta['elapsed_seconds']:.1f}s"
            + (" *** " + ", ".join(flags) if flags else ""),
            flush=True,
        )

    async def initialize(self):
        # Arcade.make already performs RESET. Never charge an extra initial action.
        self.record(self.env.observation_space, initial=True)

    async def begin_turn(self):
        if self.unknown:
            raise RuntimeError("Previous action outcome is unknown; automatic replay prohibited")
        self.stopped = False
        self.turn_actions = 0
        if self.archive.metadata()["status"] == "GAME_OVER":
            await self.step("RESET", automatic=True)
            self.stopped = False

    async def step(self, action, x=None, y=None, *, automatic=False):
        async with self.lock:
            if self.clock.start is None or self.clock.remaining() <= 0:
                raise TimeoutError("Gameplay deadline reached or not started")
            if self.stopped or self.unknown:
                raise RuntimeError("Execution stopped at a boundary; await a new model turn")
            current = self.archive.metadata()
            if current["status"] == "WIN":
                raise RuntimeError("Game already complete")
            if action != "RESET" and action not in current["legal_actions"]:
                raise ValueError(f"Unavailable action: {action}")
            if action == "ACTION6":
                if any(type(v) is not int or not 0 <= v < 64 for v in (x, y)):
                    raise ValueError("ACTION6 requires integer x=column,y=row in 0..63")
            elif x is not None or y is not None:
                raise ValueError("Coordinates apply only to ACTION6")
            args = {"action": action, **({"x": x, "y": y} if action == "ACTION6" else {})}
            if not automatic and self.actions_per_turn and self.turn_actions >= self.actions_per_turn:
                self.archive.event("action_rejected", reason="turn_cap", action=args)
                raise ValueError(
                    f"At most {self.actions_per_turn} actions per response; "
                    "await the next model decision"
                )
            if self.reject_repeats and action != "RESET" and self.repeat_key(action, x, y) in self.no_change:
                self.archive.event("action_rejected", reason="no_change_repeat", action=args)
                where = f" at (x={x},y={y})" if action == "ACTION6" else ""
                raise ValueError(
                    f"{action}{where} already produced no board change from this board; "
                    "choose a different action"
                )
            if not automatic:
                self.turn_actions += 1
            self.archive.event(
                "action_issued", action=args, automatic=automatic, seconds=self.clock.elapsed()
            )
            call = self.env.reset if action == "RESET" else self.env.step
            positional = (
                ()
                if action == "RESET"
                else (GameAction[action], {"x": x, "y": y} if action == "ACTION6" else {})
            )
            task = asyncio.create_task(asyncio.to_thread(call, *positional))
            cancelled = False
            try:
                try:
                    raw = await asyncio.shield(task)
                except asyncio.CancelledError:
                    raw = await task  # Commit an already-issued action before teardown.
                    cancelled = True
                self.actions += 1
                result = self.record(raw, args)
            except Exception as exc:
                self.unknown = self.stopped = True
                self.archive.event("action_unknown", action=args, error=str(exc))
                print(f"[{self.label}] action outcome unknown: {exc}", flush=True)
                raise
            if cancelled:
                raise asyncio.CancelledError
            return result
