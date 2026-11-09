"""Equal scheduler in dfranzen's harness, replacing his priority gate.

His gate (``tool_agent._PriorityGate``) admits ``ARC3_MAX_ACTIVE_STREAMS`` games and hands a game's stream over only
where his trimmer has just invalidated its prefix (``_maybe_handover``) or when the game ends (``release``). This keeps
those two hand-over points and replaces the choice of who gets a free stream:

- the waiting game with the fewest generated tokens so far (unstarted 0; ties in his queue order, game index first);
- a game is parked once it has generated ``park_tokens`` in total: it gets no stream while an unparked game waits,
  and parked games come back fewest tokens first only when no unparked game is waiting;
- won or ended games leave (his ``release``).

Generated tokens per game are counted from his own usage accounting (``_accumulate_usage_tokens``) under the game's
thread name (``game:<id>``; one worker thread per game for its life), optionally one row per counted usage
(``requests_path``). Every admission is logged with its reason.
"""

import json
import os
import threading
import time
from pathlib import Path

UNTRIMMED_BASE = 2_000_000  # his _PRIORITY_UNTRIMMED_BASE: acquire(BASE - queue position)


def game_name():
    name = threading.current_thread().name
    return name if name.startswith("game:") else f"thread:{threading.get_ident()}"


class EqualGate:
    def __init__(self, slots, park_tokens, log_path=None, state_path=None, clock=time.time):
        self._cond = threading.Condition()
        self.slots, self.park_tokens = max(1, int(slots)), int(park_tokens)
        self._free = self.slots
        self.tokens, self.position, self.waiting = {}, {}, {}
        self.active, self.finished, self.parked_logged = set(), set(), set()
        self.counts = {"admissions": 0, "start": 0, "drain": 0, "drain_kept": 0, "finish": 0, "parked_admissions": 0}
        self.log_path, self.state_path, self.clock = log_path, state_path, clock
        self._state_written = 0.0
        self._seq = 0

    # his gate's interface
    def configure_clock(self, started_at, deadline):
        pass

    def _snapshot_priority(self, snapshot, now):
        return 0

    def add_tokens(self, game, count):
        if count > 0:
            with self._cond:
                self.tokens[game] = self.tokens.get(game, 0) + int(count)

    def acquire(self, priority):
        game = game_name()
        with self._cond:
            self.position[game] = UNTRIMMED_BASE - int(priority)
            self.tokens.setdefault(game, 0)
            self._wait(game, "start")

    def handover(self, priority, snapshot=None):
        game = game_name()
        with self._cond:
            self.active.discard(game)
            self._free += 1
            if self.parked(game) and game not in self.parked_logged:
                self.parked_logged.add(game)
                self._log({"event": "parked", "game": game, "tokens": self.tokens.get(game, 0)})
            self._wait(game, "drain")

    def release(self):
        game = game_name()
        with self._cond:
            self.active.discard(game)
            self.finished.add(game)
            self._free += 1
            self._pump(game, "finish")
            self._write_state(force=True)

    def _wait(self, game, reason):
        self._seq += 1
        self.waiting[game] = self._seq
        self._pump(game, reason)
        while game not in self.active:
            self._cond.wait()

    def parked(self, game):
        return self.tokens.get(game, 0) >= self.park_tokens

    def _pump(self, trigger, reason):
        woke = False
        while self._free > 0 and self.waiting:
            unparked = [g for g in self.waiting if not self.parked(g)]
            pool = unparked or list(self.waiting)
            chosen = min(pool, key=lambda g: (self.tokens.get(g, 0), self.position.get(g, 0), self.waiting[g]))
            del self.waiting[chosen]
            self.active.add(chosen)
            self._free -= 1
            woke = True
            kept = chosen == trigger
            self.counts["admissions"] += 1
            self.counts[reason] += 1
            self.counts["drain_kept"] += int(reason == "drain" and kept)
            self.counts["parked_admissions"] += int(self.parked(chosen))
            self._log({"event": "admit", "reason": reason, "trigger": trigger,
                       "trigger_tokens": self.tokens.get(trigger, 0), "game": chosen,
                       "game_tokens": self.tokens.get(chosen, 0), "kept": kept, "parked": self.parked(chosen),
                       "waiting": len(self.waiting), "waiting_unparked": len(unparked) - (chosen in unparked),
                       "active": len(self.active), "finished": len(self.finished)})
        if woke:
            self._cond.notify_all()
        self._write_state()

    def _log(self, row):
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"epoch": round(self.clock(), 3), **row}) + "\n")

    def summary(self):
        tokens = sorted(self.tokens.values())
        return {"slots": self.slots, "park_tokens": self.park_tokens, "active": len(self.active),
                "waiting": len(self.waiting), "finished": len(self.finished),
                "parked": sum(1 for g in self.tokens if self.parked(g)),
                "started": sum(1 for g in self.tokens if g in self.active or self.tokens[g] > 0 or g in self.finished),
                "tokens_min": tokens[0] if tokens else 0, "tokens_median": tokens[len(tokens) // 2] if tokens else 0,
                "tokens_max": tokens[-1] if tokens else 0, **self.counts}

    def _write_state(self, force=False):
        now = self.clock()
        if not self.state_path or (not force and now - self._state_written < 30):
            return
        self._state_written = now
        state = {"epoch": round(now, 1), **self.summary(), "games": dict(sorted(self.tokens.items())),
                 "active_games": sorted(self.active)}
        Path(self.state_path).write_text(json.dumps(state, indent=1))


def name_game_threads(solver):
    """Each of his worker threads plays one game for its life (``HarnessSolver._play_one``): name it ``game:<id>``.
    Same marker as dfr_gate.name_game_threads, so the two never wrap twice."""
    cls = solver.HarnessSolver
    if getattr(cls._play_one, "_b2d_named", False):
        return
    original = cls._play_one

    def _play_one(self, game, *args, **kwargs):
        try:
            threading.current_thread().name = f"game:{game.game_run.game_id}"
        except Exception:  # noqa: BLE001 - naming only
            pass
        return original(self, game, *args, **kwargs)

    _play_one._b2d_named = True
    cls._play_one = _play_one


def install(config_path=None):
    """In his process, after his sources are importable and before ``bm.run``."""
    cfg = json.loads(Path(config_path or os.environ["DFR_EQUAL_CONFIG"]).read_text())
    from inference.agent import tool_agent
    import inference.framework.solver as solver

    slots = tool_agent._max_active_streams()
    if slots <= 0 or tool_agent._PRIORITY_GATE is not None:
        raise RuntimeError(f"equal gate: needs his stream gate on and not yet built (slots {slots})")
    gate = EqualGate(slots, cfg["park_tokens"], cfg.get("log_path"), cfg.get("state_path"))
    owner = next(c for c in vars(tool_agent).values()
                 if isinstance(c, type) and "_accumulate_usage_tokens" in vars(c))
    original = owner._accumulate_usage_tokens
    requests_path, log_lock = cfg.get("requests_path"), threading.Lock()  # one row per usage his agent counts

    def _accumulate_usage_tokens(self, usage, *args, **kwargs):
        before = int(getattr(self, "_session_generated_tokens", 0) or 0)
        result = original(self, usage, *args, **kwargs)
        done = int(getattr(self, "_session_generated_tokens", 0) or 0) - before
        gate.add_tokens(game_name(), done)
        if requests_path and isinstance(usage, dict):
            details = usage.get("prompt_tokens_details") or {}
            row = {"epoch": round(time.time(), 3), "game": game_name(), "prompt_tokens": usage.get("prompt_tokens"),
                   "cached_tokens": details.get("cached_tokens"), "completion_tokens": done,
                   "summary": not kwargs.get("count_toward_turn", True)}
            with log_lock, open(requests_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        return result

    owner._accumulate_usage_tokens = _accumulate_usage_tokens
    name_game_threads(solver)
    tool_agent._PRIORITY_GATE = gate
    print(json.dumps({"event": "dfr_equal_gate", "slots": slots, "park_tokens": gate.park_tokens,
                      "usage_hook": f"{owner.__name__}._accumulate_usage_tokens"}), flush=True)
    return gate
