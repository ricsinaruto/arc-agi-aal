"""Checkpoint and resume of a Duck-transfer arena trial on preemptible compute (``scripts/nebius_validation.py``).

Off unless ``ARC_CHECKPOINT_SECONDS`` is set (the Nebius runner sets it): without it no object here is created and
every hook in the trial, the agents and the sandbox is one ``is None`` test.

The cut. A game can restart only where its analyzer thread waits for a gameplay reply: the thread is blocked in the
bridge, the conversation is the list it passed, the interpreter and the environment are idle. A checkpoint is taken on
the event loop in one step when every unfinished game is there (``safe``); after ``patience_seconds`` arriving replies
are held for at most ``hold_seconds`` so the games still executing a tool call catch up. Requests in flight at the cut
are not part of the checkpoint: a resumed game sends its request again.

What is written (``gen-NNNNNN/``): ``state.pkl.z`` (one pickle of the wall clock, game records and clocks, scheduler,
gate, agents, analyzers and each game's open conversation), ``actions.json`` (every environment call with the board
digest it returned) and ``manifest.json``. Objects that belong to the process (event-loop primitives, threads, files,
clients, callables) are stored as the path they were found at and taken from the new process on restore; arena objects
are updated in place, so the code that holds them keeps working.

Resume. The new process builds the trial as usual; environments replay ``actions.json`` when they are made (digest
checked per call); every game runs to its first request and waits there; then the state is laid over the fresh
objects, the interpreters replay their recorded inputs, each analyzer's conversation is replaced in place and the
monotonic clock is set to the checkpoint instant (``arena.resume_clock``), so the play clock, budgets and stored
timestamps continue. Not restored: requests in flight (sent again), the server's KV cache (cold: every game prefills
its context once), GPU monitor samples and live-progress state (new process), an interpreter whose replay diverges or
runs past ``REPLAY_BUDGET`` (fresh interpreter, the model is told), and three analyzer locals of the open turn (its
request count, turn start time and transcript buffer)."""

import asyncio
import dis
import enum
import functools
import hashlib
import io
import json
import os
import pickle
import shutil
import sys
import threading
import time
import types
import zlib
from collections import deque
from pathlib import Path

from arc_infra.perception import board_hash

from . import resume_clock

UNSUPPORTED = ("explorer", "parallel_samples", "llm_backlog", "cross_game_notes", "memory_block")
OUR = frozenset({"arena", "duck_transfer", "arc_infra", "inference"})
LIVE_ROOTS = frozenset({
    "asyncio", "_asyncio", "threading", "_thread", "concurrent", "subprocess", "queue", "_queue", "socket", "_socket",
    "ssl", "_ssl", "selectors", "select", "io", "_io", "tempfile", "weakref", "_weakref", "_weakrefset", "contextvars",
    "_contextvars", "logging", "httpx", "httpcore", "anyio", "sniffio", "h11", "requests", "urllib3", "multiprocessing",
    "sqlite3", "_sqlite3", "arc_agi", "contextlib", "signal"})
LIVE_TYPES = (types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.ModuleType, types.GeneratorType,
              types.CoroutineType, types.AsyncGeneratorType, types.FrameType, types.CodeType, types.MethodWrapperType,
              types.WrapperDescriptorType, types.MethodDescriptorType, functools.partial)
ATOMS = frozenset({str, bytes, int, float, bool, type(None), complex, bytearray})
KEYS = frozenset({str, int, bool, type(None)})
LIMIT = 512  # containers above this are data: stored by value, not searched for process objects
FRESH = frozenset({"Trial", "LiveProgress", "GpuMonitor", "ArenaSglangMonitor"})  # arena objects of the new process
REPLAY_BUDGET = 300.0
RESET_NOTE = ("The run was interrupted and resumed; this game's python interpreter could not be rebuilt and was "
              "restarted before this call. Redefine the helpers and variables you still need.")


class _Gone:
    """A process object the new process does not have (an attribute holding it is dropped)."""

    __slots__ = ()

    def __bool__(self):
        return False

    def __reduce__(self):
        return (_gone, ())


GONE = _Gone()


def _gone():
    return GONE


class ResumeError(RuntimeError):
    """The checkpoint cannot continue this trial; the attempt must not play on."""


# --- environment record / replay ---------------------------------------------------------------------------------

def frame_digest(raw):
    if raw is None:
        return "none"
    frames = raw.frame
    board = board_hash(frames[-1]) if len(frames) else "-"
    return f"{int(raw.levels_completed)}:{getattr(raw.state, 'name', raw.state)}:{board}"


class RecordingEnv:
    """The environment with every call logged as ``[action, data, digest]``."""

    _checkpoint_live = True

    def __init__(self, env, log):
        self._env, self._log = env, log

    def step(self, action, data=None, **kwargs):
        raw = self._env.step(action, data, **kwargs)
        self._log.append([getattr(action, "name", str(action)), data or None, frame_digest(raw)])
        return raw

    def reset(self):
        raw = self._env.reset()
        self._log.append(["RESET", None, frame_digest(raw)])
        return raw

    def __getattr__(self, name):
        return getattr(self._env, name)


class RecordingArcade:
    """The arcade whose environments log their calls; with ``replay`` a new environment first repeats them."""

    _checkpoint_live = True

    def __init__(self, arc, logs, replay=None):
        self._arc, self.logs, self.replay = arc, logs, dict(replay or {})

    def make(self, game_id, **kwargs):
        env = self._arc.make(game_id, **kwargs)
        if env is None:
            return None
        log = self.logs.setdefault(game_id, [])
        prior = self.replay.pop(game_id, None)
        if prior:
            from arcengine import GameAction

            for index, (name, data, digest) in enumerate(prior):
                raw = env.reset() if name == "RESET" else env.step(GameAction[name], data or {})
                if frame_digest(raw) != digest:
                    raise ResumeError(f"{game_id}: replayed action {index + 1} of {len(prior)} gave another board")
            log.extend(prior)
        return RecordingEnv(env, log)

    def __getattr__(self, name):
        return getattr(self._arc, name)


# --- interpreter record / replay ---------------------------------------------------------------------------------

class SandboxRecorder:
    """What the live interpreter was sent since it started (``StatefulSandbox.recorder``) and the kinds of its
    answers, enough to rebuild its state in a new child."""

    def __init__(self):
        self.spawned()

    def spawned(self):
        self.sent, self.seen, self.interrupts, self.requests = [], [], {}, 0

    def note_sent(self, payload):
        self.requests += payload.get("type", "code") in ("code", "inventory")
        self.sent.append(zlib.compress(json.dumps(payload, ensure_ascii=False).encode(), 1))

    def note_received(self, message):
        kind = str(message.get("type", "")).strip()
        self.seen.append((kind, json.dumps(message.get("actions"), sort_keys=True) if kind == "action" else None))

    def note_interrupt(self, timeout_seconds):
        self.interrupts[self.requests] = float(timeout_seconds)


def replay_sandbox(sandbox, budget=REPLAY_BUDGET):
    """Rebuild ``sandbox``'s interpreter from its recorder; ``{"reset": reason}`` when it had to start empty."""
    recorder = sandbox.recorder
    if recorder is None or not recorder.sent:
        return {"requests": 0}
    sent, seen, interrupts, synced = list(recorder.sent), deque(recorder.seen), dict(recorder.interrupts), sandbox.synced
    sandbox.recorder = None
    started = time.perf_counter()
    report = {"requests": 0, "inexact": 0}

    def fail(reason):
        sandbox._discard("resume")
        recorder.spawned()
        sandbox.recorder = recorder
        return {**report, "reset": reason, "seconds": round(time.perf_counter() - started, 3)}

    def line(index):
        return json.loads(zlib.decompress(sent[index])) if index < len(sent) else None

    try:
        with sandbox.lock:
            sandbox._spawn()
            sandbox._send({**line(0), "sandbox_cwd": sandbox.tempdir.name})
            index = report["requests"] = 1
            interrupted, deadline = False, None
            if 1 in interrupts:
                deadline = time.perf_counter() + interrupts[1]
            while True:
                if time.perf_counter() - started > budget:
                    return fail("replay_budget")
                try:
                    raw = sandbox.lines.get(timeout=0.5)
                except Exception:  # noqa: BLE001 - queue.Empty
                    if deadline is not None and time.perf_counter() >= deadline:
                        sandbox._interrupt()  # the recorded call was interrupted at its timeout
                        interrupted = True
                    continue
                if raw is None:
                    return fail("exit")
                message = json.loads(raw)
                kind = str(message.get("type", "")).strip()
                expected = seen.popleft() if seen else (None, None)
                following = line(index)
                if kind in ("action", "animation"):
                    if (following is None or following.get("type") not in (f"{kind}_result", f"{kind}_error")
                            or expected != (kind, json.dumps(message.get("actions"), sort_keys=True)
                                            if kind == "action" else None)):
                        return fail(f"{kind}_diverged")
                    sandbox._send(following)
                    index += 1
                    continue
                if kind not in ("final", "error", "inventory") or expected[0] in ("action", "animation"):
                    return fail("ended_early" if kind in ("final", "error", "inventory") else "unknown_message")
                if expected[0] != kind or interrupted:
                    report["inexact"] += 1  # same request, another outcome (a timing-dependent or interrupted call)
                if following is None:
                    break
                if following.get("type") not in ("code", "inventory"):
                    return fail("request_diverged")
                sandbox._send(following)
                index += 1
                report["requests"] += 1
                interrupted = False
                deadline = (time.perf_counter() + interrupts[report["requests"]]
                            if report["requests"] in interrupts else None)
    except (OSError, ValueError) as exc:
        return fail(type(exc).__name__)
    sandbox.synced, sandbox.recorder = synced, recorder
    return {**report, "seconds": round(time.perf_counter() - started, 3)}


# --- generic state capture / overlay ---------------------------------------------------------------------------

def _root(kind):
    return (getattr(kind, "__module__", None) or "").split(".", 1)[0]


def _name(kind):
    return f"{kind.__module__}:{kind.__qualname__}"


def _importable(obj):
    found = sys.modules.get(getattr(obj, "__module__", None) or "")
    try:
        for part in obj.__qualname__.split("."):
            found = getattr(found, part)
    except AttributeError:
        return False
    return found is obj


def is_live(obj):
    kind = type(obj)
    return (getattr(kind, "_checkpoint_live", False) or isinstance(obj, LIVE_TYPES) or _root(kind) in LIVE_ROOTS
            or kind.__name__ in FRESH)


def _is_data(value, depth=4):
    if type(value) in ATOMS:
        return True
    if depth and type(value) in (list, tuple, set, frozenset, dict) and len(value) <= LIMIT:
        items = [*value.keys(), *value.values()] if type(value) is dict else value
        return all(_is_data(item, depth - 1) for item in items)
    return False


@functools.cache
def _rebound(code):
    """The closure variables ``code`` (or a function nested in it) assigns: state a closure keeps between calls."""
    names = {i.argval for i in dis.get_instructions(code) if i.opname == "STORE_DEREF"}
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _rebound(const)
    return frozenset(names)


def _cells(function):
    """``{name: value}`` of the plain-data closure variables ``function`` rebinds (the only closure state restored)."""
    state = {}
    for name, cell in zip(function.__code__.co_freevars, function.__closure__ or (), strict=True):
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if name in _rebound(function.__code__) and _is_data(value):
            state[name] = value
    return state


def is_anchor(obj):
    kind = type(obj)
    if _root(kind) not in OUR or isinstance(obj, (type, enum.Enum, BaseException)) or not hasattr(obj, "__dict__"):
        return False
    params = getattr(kind, "__dataclass_params__", None)
    return not (params is not None and params.frozen)


def paths_of(roots):
    """``id -> path`` of every arena object and process object reachable through attributes and small containers."""
    found, seen, queue = {}, set(), deque([((), roots)])
    while queue:
        path, obj = queue.popleft()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if is_live(obj):
            found[id(obj)] = path
            if type(obj) is not types.FunctionType or not obj.__closure__:
                continue
            children = []  # what a stored closure holds is reachable too (a tracker or cache only it knows)
            for index, cell in enumerate(obj.__closure__):
                try:
                    children.append(("c", index, cell.cell_contents))
                except ValueError:
                    pass
        elif is_anchor(obj):
            found[id(obj)] = path
            children = [("a", key, value) for key, value in vars(obj).items()]
        elif type(obj) is types.SimpleNamespace:
            children = [("a", key, value) for key, value in vars(obj).items()]
        elif isinstance(obj, (list, tuple, deque)):
            if len(obj) > LIMIT and len(path) > 1:
                continue
            children = [("i", index, value) for index, value in enumerate(obj)]
        elif isinstance(obj, dict):
            if len(obj) > LIMIT:
                continue
            children = [("k", key, value) for key, value in obj.items() if type(key) in KEYS]
        else:
            continue
        for step, key, value in children:
            if type(value) not in ATOMS:
                queue.append((path + ((step, key),), value))
    return found


class _Load:
    def __init__(self, roots):
        self.roots, self.pending, self.events, self.cells = roots, [], [], []
        self.report = {"matched": 0, "created": {}, "unresolved": {}}


_LOAD = None


def _resolve(path):
    if path is None:
        return GONE
    obj = _LOAD.roots
    try:
        for step, key in path:
            obj = getattr(obj, key) if step == "a" else obj.__closure__[key].cell_contents if step == "c" else obj[key]
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        return GONE
    return obj


def _count(table, key):
    table[key] = table.get(key, 0) + 1


def _anchor(path, cls, name):
    fresh = _resolve(path)
    if fresh is not GONE and _name(type(fresh)) == name:
        _LOAD.report["matched"] += 1
        return fresh
    if cls is None:
        _count(_LOAD.report["unresolved"], name)
        return types.SimpleNamespace()
    _count(_LOAD.report["created"], name)
    return cls.__new__(cls)


def _set_state(obj, state):
    _LOAD.pending.append((obj, state))


def _live(path, name, is_set):
    fresh = _resolve(path)
    if fresh is GONE:
        _count(_LOAD.report["unresolved"], name)
    elif is_set is not None and hasattr(fresh, "is_set"):
        _LOAD.events.append((fresh, is_set))
    return fresh


def _function(path, name, cells):
    fresh = _resolve(path)
    if fresh is GONE:
        _count(_LOAD.report["unresolved"], name)
    elif cells and isinstance(fresh, types.FunctionType):
        _LOAD.cells.append((fresh, cells))
    return fresh


def _method(path, owner, attribute, name):
    fresh = _resolve(path)
    if fresh is not GONE:
        return fresh
    if owner is GONE or not hasattr(owner, attribute):
        _count(_LOAD.report["unresolved"], name)
        return GONE
    return getattr(owner, attribute)


def _missing(name):
    _count(_LOAD.report["unresolved"], name)
    return GONE


class _Pickler(pickle.Pickler):
    def __init__(self, file, paths, report):
        super().__init__(file, protocol=5)
        self.paths, self.report, self.plain = paths, report, {}

    def reducer_override(self, obj):
        kind = type(obj)
        path = self.paths.get(id(obj))
        if isinstance(obj, type):
            return NotImplemented if _importable(obj) else (_missing, (_name(obj),))
        if is_live(obj):
            if path is None and isinstance(obj, (types.FunctionType, types.BuiltinFunctionType)) and _importable(obj):
                return NotImplemented
            if path is not None and type(obj) is types.FunctionType and obj.__closure__:
                return (_function, (path, _name(kind), _cells(obj)))
            if isinstance(obj, types.MethodType):
                owner = obj.__self__ if id(obj.__self__) in self.paths else GONE
                return (_method, (path, owner, obj.__name__, _name(kind)))
            if path is None:
                _count(self.report["unresolved"], _name(kind))
            return (_live, (path, _name(kind), obj.is_set() if isinstance(obj, (asyncio.Event, threading.Event)) else None))
        if is_anchor(obj):
            if path is None:
                return NotImplemented if _importable(kind) else (_missing, (_name(kind),))
            self.report["anchors"] += 1
            return (_anchor, (path, kind if _importable(kind) else None, _name(kind)), dict(vars(obj)), None, None,
                    _set_state)
        plain = self.plain.get(kind)
        if plain is None:
            try:
                obj.__reduce_ex__(5)
                plain = True
            except Exception:  # noqa: BLE001 - any object the process cannot hand over
                plain = False
            self.plain[kind] = plain
        if not plain:
            if path is None:
                _count(self.report["unresolved"], _name(kind))
            return (_live, (path, _name(kind), None))
        return NotImplemented


def capture(roots):
    """``(bytes, report)``: one pickle of ``roots`` (a dict)."""
    report = {"anchors": 0, "unresolved": {}}
    buffer = io.BytesIO()
    _Pickler(buffer, paths_of(roots), report).dump(roots)
    return buffer.getvalue(), report


def restore(roots, data):
    """Lay ``data`` over the fresh ``roots``; returns ``(loaded roots, report)``."""
    global _LOAD
    load = _LOAD = _Load(roots)
    try:
        loaded = pickle.loads(data)
    finally:
        _LOAD = None
    for obj, state in load.pending:
        target = obj.__dict__
        target.clear()
        target.update((key, value) for key, value in state.items() if value is not GONE)
    for event, is_set in load.events:
        event.set() if is_set else event.clear()
    for function, cells in load.cells:
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or (), strict=True):
            if name in cells:
                cell.cell_contents = cells[name]
    return loaded, load.report


# --- the trial's checkpointer ------------------------------------------------------------------------------------

def generations(folder):
    return sorted(p for p in Path(folder).glob("gen-*") if p.is_dir() and not p.name.endswith(".tmp"))


def latest(folder):
    """``(path, manifest)`` of the newest complete generation whose state matches its manifest, or None."""
    for path in reversed(generations(folder)):
        try:
            manifest = json.loads((path / "manifest.json").read_text())
            packed = (path / "state.pkl.z").read_bytes()
            json.loads((path / "actions.json").read_text())
        except (OSError, ValueError):
            continue
        if len(packed) == manifest.get("bytes") and hashlib.sha256(packed).hexdigest() == manifest.get("sha256"):
            return path, manifest
    return None


def played_seconds(output, environ=None):
    """Play-clock seconds a resumed trial has behind it (0 unless this process resumes a checkpoint): what its wall
    no longer needs."""
    environ = os.environ if environ is None else environ
    if environ.get("ARC_CHECKPOINT_RESUME") != "1" or not environ.get("ARC_CHECKPOINT_SECONDS"):
        return 0.0
    source = latest(Path(output) / "checkpoint")
    return float(source[1]["elapsed_seconds"]) if source else 0.0


def game_digest(agent, messages=None):
    """The comparable state of one game (manifest row; the fidelity test compares two of them)."""
    def sha(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]

    clock, analyzer = agent.game.clock, agent._analyzer
    row = {"steps": agent.steps, "turns": agent.turns, "status": agent.status, "resets": agent.resets,
           "levels": agent.obs.levels if agent.obs else 0, "state": agent.obs.state if agent.obs else None,
           "board": board_hash(agent.obs.board) if agent.obs else None,
           "prompt_tokens": agent.prompt_tokens, "completion_tokens": agent.completion_tokens,
           "budget": clock.budget, "budget_base": clock.budget_base, "budget_tokens": clock.budget_tokens(),
           "game_state": agent.game.state, "game_levels": agent.game.levels,
           "sandbox_calls": agent.sandbox.calls, "sandbox_restarts": agent.sandbox.restarts,
           "history": None if analyzer is None else [len(analyzer._history_messages), sha(analyzer._history_messages)]}
    if messages is not None:
        row["messages"] = [len(messages), sha(messages)]
    return row


def from_env(study, kind, output, log=None, environ=None):
    """The trial's ``Checkpointer``, or None when checkpointing is off or the study has a feature it cannot cut."""
    environ = os.environ if environ is None else environ
    seconds = environ.get("ARC_CHECKPOINT_SECONDS")
    if not seconds:
        return None
    source = latest(Path(output) / "checkpoint") if environ.get("ARC_CHECKPOINT_RESUME") == "1" else None
    resume = source is not None
    if environ.get("ARC_CHECKPOINT_RESUME") == "1" and not resume and (Path(output) / "progress.jsonl").exists():
        raise ResumeError("A partial trial without a usable checkpoint cannot continue in its folder")
    blocked = [key for key in UNSUPPORTED if study.get(key)] + ([] if kind == "duck-transfer" else [f"kind:{kind}"])
    if blocked:
        if resume:
            raise ResumeError(f"Checkpoint resume is not supported with {blocked}")
        if log:
            log({"event": "checkpoint_off", "unsupported": blocked})
        return None
    return Checkpointer({
        "interval_seconds": float(seconds), "resume": resume,
        "patience_seconds": float(environ.get("ARC_CHECKPOINT_PATIENCE", 45)),
        "hold_seconds": float(environ.get("ARC_CHECKPOINT_HOLD", 20)),
        "attempt": int(environ.get("ARC_CHECKPOINT_ATTEMPT", 1)), "keep": 2, "dir": Path(output) / "checkpoint"},
        log, source)


class Checkpointer:
    _checkpoint_live = True

    def __init__(self, cfg, log=None, source=None):
        self.cfg, self.log = cfg, log or (lambda entry: None)
        self.dir = Path(cfg["dir"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.request = self.dir / "request"  # the runner touches it on a preemption notice: checkpoint now
        self.source = source if cfg["resume"] else None
        self.resuming = self.source is not None
        self.seq = max([int(p.name[4:]) for p in generations(self.dir)], default=0)
        self.logs, self.replay, self.segments = {}, {}, []
        if self.resuming:
            if not resume_clock.installed():
                raise ResumeError("Resume needs arena.resume_clock installed before the arena modules load")
            path, manifest = self.source
            self.replay = json.loads((path / "actions.json").read_text())
            self.segments = list(manifest.get("segments") or [])
        self.segment = {"attempt": cfg["attempt"], "kind": "resumed" if self.resuming else "fresh",
                        "started_epoch": round(time.time(), 1)}
        self.segments.append(self.segment)
        self.at_chat, self.arrived, self.gates, self.ended = {}, {}, {}, set()
        self.roots = self.tasks = self.card = self.open = self.arrival = None
        self.next_at = self.due_since = self.hold_since = self.writer = None
        self.stats = {"written": 0, "skipped": 0, "holds": 0, "held_seconds": 0.0, "last": None}

    def arcade(self, arc):
        return RecordingArcade(arc, self.logs, self.replay)

    def bind(self, roots, *, card, tasks):
        self.roots, self.card, self.tasks = roots, card, tasks
        self.open, self.arrival = asyncio.Event(), asyncio.Event()
        self.open.set()
        for agent in roots["agents"]:
            agent.checkpoints = self
            agent.sandbox.recorder = SandboxRecorder()
            self.gates[agent] = asyncio.Event()

    # --- the agents' side ---

    async def chat(self, agent, messages, tools):
        if self.resuming and agent not in self.arrived:
            self.arrived[agent] = messages
            self.arrival.set()
            await self.gates[agent].wait()
            if agent in self.ended:
                raise TimeoutError("The game had finished before the checkpoint")
        self.at_chat[agent] = messages
        try:
            if self.due_since is not None and self.safe():
                self.snapshot()
            return await agent._arena_chat(messages, tools)
        finally:
            self.at_chat.pop(agent, None)

    async def hold(self):
        """A gameplay reply waits here while a checkpoint is being cut (never when none is)."""
        if not self.open.is_set():
            await self.open.wait()

    # --- writing ---

    def safe(self):
        return all(task.done() or (agent in self.at_chat and agent.request_purpose is None)
                   for agent, task in zip(self.roots["agents"], self.tasks, strict=True))

    def tick(self):
        """Once per scheduler tick: cut a checkpoint when one is due and every game can restart."""
        now = time.monotonic()
        if self.writer is not None and not self.writer.is_alive():
            self.writer = None
        if self.next_at is None:
            self.next_at = now + self.cfg["interval_seconds"]
        forced = self.request.exists()
        if self.writer is not None or self.resuming or not (forced or now >= self.next_at):
            return
        if self.due_since is None:
            self.due_since = now
        if self.safe():
            return self.snapshot()
        if self.open.is_set():
            if forced or now - self.due_since >= self.cfg["patience_seconds"]:
                self.open.clear()
                self.hold_since = now
                self.stats["holds"] += 1
        elif now - self.hold_since >= self.cfg["hold_seconds"]:
            self.stats["skipped"] += 1
            self.log({"event": "checkpoint_skipped", "unsafe": sum(not self._safe(a, t) for a, t in
                                                                   zip(self.roots["agents"], self.tasks, strict=True))})
            self._release(now)
            self.next_at = now if forced else now + min(60.0, self.cfg["interval_seconds"])

    def _safe(self, agent, task):
        return task.done() or (agent in self.at_chat and agent.request_purpose is None)

    def _release(self, now):
        if not self.open.is_set():
            self.stats["held_seconds"] += now - self.hold_since
            self.open.set()
        self.due_since = self.hold_since = None

    def snapshot(self):
        started, now = time.perf_counter(), time.monotonic()
        agents, wall = self.roots["agents"], self.roots["wall"]
        messages = {agent.game_id: self.at_chat[agent] for agent in agents if agent in self.at_chat}
        data, report = capture({**self.roots, "messages": messages})
        self.seq += 1
        inflight = sum(agent.game.clock._inference_started is not None for agent in agents)
        manifest = {
            "seq": self.seq, "epoch": round(time.time(), 3), "monotonic": now,
            "elapsed_seconds": round(wall.elapsed(), 3), "wall_seconds": wall.seconds, "card": self.card,
            "games": {agent.game_id: {**game_digest(agent, messages.get(agent.game_id)),
                                      "finished": agent.game_id not in messages} for agent in agents},
            "inflight_at_cut": inflight, "capture": report, "segments": self.segments,
            "capture_seconds": round(time.perf_counter() - started, 3),
            "held_seconds": round(now - self.hold_since, 3) if self.hold_since is not None else 0.0,
        }
        actions = {game_id: list(log) for game_id, log in self.logs.items()}
        forced = self.request.exists()
        self._release(now)
        self.next_at = now + self.cfg["interval_seconds"]
        self.writer = threading.Thread(target=self._write, args=(self.seq, data, manifest, actions, forced), daemon=True)
        self.writer.start()

    def _write(self, seq, data, manifest, actions, forced):
        started = time.perf_counter()
        final, temp = self.dir / f"gen-{seq:06d}", self.dir / f"gen-{seq:06d}.tmp"
        shutil.rmtree(temp, ignore_errors=True)
        temp.mkdir(parents=True)
        packed = zlib.compress(data, 1)
        (temp / "state.pkl.z").write_bytes(packed)
        (temp / "actions.json").write_text(json.dumps(actions))
        manifest.update(bytes=len(packed), raw_bytes=len(data), sha256=hashlib.sha256(packed).hexdigest(),
                        write_seconds=round(time.perf_counter() - started, 3))
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=1))
        temp.rename(final)
        for old in generations(self.dir)[:-self.cfg["keep"]]:
            shutil.rmtree(old, ignore_errors=True)
        if forced:
            self.request.unlink(missing_ok=True)
        self.stats["written"] += 1
        self.stats["last"] = {key: manifest[key] for key in ("seq", "elapsed_seconds", "bytes", "raw_bytes",
                                                             "capture_seconds", "write_seconds", "held_seconds",
                                                             "inflight_at_cut")}
        self.log({"event": "checkpoint", "forced": forced, **self.stats["last"]})

    # --- resuming ---

    async def resume(self, *, deadline_epoch=None, ready_timeout=3600.0):
        """Called where a fresh trial releases its clocks: wait for every game at its first request, lay the
        checkpoint over the trial and let the games continue."""
        path, manifest = self.source
        agents, cut = self.roots["agents"], manifest["games"]
        if [agent.game_id for agent in agents] != list(cut):
            raise ResumeError("The checkpoint is of another game list")
        started, limit = time.perf_counter(), time.monotonic() + ready_timeout
        while any(agent not in self.arrived and not task.done() for agent, task in zip(agents, self.tasks, strict=True)):
            if time.monotonic() > limit:
                raise ResumeError("Games did not reach their first request")
            self.arrival.clear()
            try:
                await asyncio.wait_for(self.arrival.wait(), timeout=1.0)
            except TimeoutError:
                pass
        # A game that had finished ends here too, before its state is restored; any other game must be waiting.
        self.ended = {agent for agent in agents if cut[agent.game_id]["finished"]}
        for agent in self.ended:
            self.gates[agent].set()
        await asyncio.gather(*(task for agent, task in zip(agents, self.tasks, strict=True) if agent in self.ended),
                             return_exceptions=True)
        failed = [agent.game_id for agent, task in zip(agents, self.tasks, strict=True)
                  if agent not in self.ended and task.done()]
        if failed:
            errors = {agent.game_id: agent.error for agent in agents if agent.game_id in failed}
            raise ResumeError(f"Games ended before the restore: {errors}")
        requests = agents[0].client.client.requests
        loaded, report = restore(self.roots, zlib.decompress((path / "state.pkl.z").read_bytes()))
        resume_clock.set(manifest["monotonic"])
        discarded = self._settle(loaded, requests)
        playing = [agent for agent in agents if agent not in self.ended]
        sandboxes = await asyncio.gather(*(asyncio.to_thread(replay_sandbox, agent.sandbox) for agent in playing))
        problems = []
        for agent, outcome in zip(playing, sandboxes, strict=True):
            if outcome.get("reset"):
                agent.ckpt_reset_note = RESET_NOTE
            saved = loaded["messages"][agent.game_id]
            self.arrived[agent][:] = saved
            if not _patch_turn(agent._analyzer):
                problems.append(f"{agent.game_id}: open turn not found")
            row = game_digest(agent, saved)
            row["sandbox_restarts"] = cut[agent.game_id]["sandbox_restarts"]  # a reset replay is reported, not an error
            if row != {key: cut[agent.game_id][key] for key in row}:
                problems.append(f"{agent.game_id}: restored state differs from the manifest")
            if frame_digest(agent.env.observation_space).rsplit(":", 1)[1] not in ("-", row["board"]):
                problems.append(f"{agent.game_id}: replayed board differs from the restored observation")
        if problems:
            raise ResumeError("; ".join(problems[:5]))
        wall = self.roots["wall"]
        resume_clock.set(manifest["monotonic"])  # the restore itself costs no play time
        if deadline_epoch is not None:
            wall.seconds = min(wall.seconds, wall.elapsed() + max(0.0, deadline_epoch - time.time()))
        self.segment.update(
            from_checkpoint=manifest["seq"], resumed_at_elapsed_seconds=manifest["elapsed_seconds"],
            checkpoint_age_seconds=round(time.time() - manifest["epoch"], 1), discarded_inflight=discarded,
            restore_seconds=round(time.perf_counter() - started, 3), restore=report,
            reprefill_tokens=sum(int(agent.last_prompt_tokens or 0) for agent in playing),
            sandbox={"replayed": sum(1 for o in sandboxes if o["requests"] and not o.get("reset")),
                     "reset": sorted(a.game_id for a, o in zip(playing, sandboxes, strict=True) if o.get("reset")),
                     "inexact_calls": sum(o.get("inexact", 0) for o in sandboxes),
                     "max_seconds": max((o.get("seconds", 0.0) for o in sandboxes), default=0.0)})
        self.log({"event": "resumed", **{k: v for k, v in self.segment.items() if k != "restore"}})
        self.resuming = False
        for agent in playing:
            self.gates[agent].set()

    def _settle(self, loaded, requests):
        """Close what was open at the cut: requests in flight and the gate reservations they held."""
        gate, discarded = self.roots["gate"], 0
        requests[:] = loaded["agents"][0].client.client.requests
        for record in requests:
            if "seconds" not in record:
                record.update(error="discarded_at_checkpoint", seconds=None)
        for agent in self.roots["agents"]:
            agent.client.client.requests = requests
            agent.card, agent.admitted_count, agent.cache_ping_request = self.card, None, None
            slot = agent.client.slots
            if slot._stack and agent.game.requests:
                agent.game.requests[-1]["discarded_at_checkpoint"] = True
            slot._stack.clear()
            for attempt, *_ in slot._timing_stack:
                attempt["status"] = "discarded_at_checkpoint"
            slot._timing_stack.clear()
            for attempt in agent.game.admission_attempts[-1:]:
                if attempt.get("status") == "waiting":
                    attempt["status"] = "discarded_at_checkpoint"
            clock = agent.game.clock
            for start, total, intervals in (("_inference_started", "_inference_total", clock.inference_intervals),
                                            ("_tool_started", "_tool_total", clock.tool_intervals)):
                began = getattr(clock, start)
                if began is None:
                    continue
                ended = max(began, clock._inference_wall())
                setattr(clock, total, getattr(clock, total) + ended - began)
                intervals.append({"started_wall_seconds": began, "ended_wall_seconds": ended, "purpose": "gameplay",
                                  "status": "discarded_at_checkpoint"})
                setattr(clock, start, None)
                discarded += start == "_inference_started"
            clock._tool_depth = 0
        gate.queue.clear()
        gate._prompt_of.clear()
        gate.inflight = gate.reserved = gate.background_inflight = 0
        if gate.lane is not None:
            gate.lane_queue.clear()
            gate._focus_since.clear()
            gate.held.clear()
            gate.lane_inflight = 0
            if gate.commit:
                gate.lane_reserved = 0
                gate.flights.clear()
        return discarded

    def rewind(self, path):
        """Drop the progress rows written after the checkpoint by the interrupted segment."""
        if not self.source or not Path(path).is_file():
            return
        elapsed, kept = self.source[1]["elapsed_seconds"], []
        for line in Path(path).read_text().splitlines():
            try:
                if json.loads(line).get("elapsed_seconds", 0) <= elapsed:
                    kept.append(line)
            except ValueError:
                continue
        Path(path).write_text("".join(line + "\n" for line in kept))

    def summary(self):
        """arena.json ``checkpoint``: which kind of attempt this report is, and what checkpointing cost."""
        return {"kind": "resumed" if any(s["kind"] == "resumed" for s in self.segments) else "fresh",
                "attempt": self.cfg["attempt"], "segments": self.segments, "interval_seconds": self.cfg["interval_seconds"],
                **self.stats}


def _patch_turn(analyzer):
    """The open ``analyze`` call of a resumed game keeps the history it would roll back to in a local list."""
    for frame in sys._current_frames().values():
        while frame is not None:
            if frame.f_code.co_name == "analyze" and frame.f_locals.get("self") is analyzer:
                previous = frame.f_locals.get("previous_history_messages")
                if isinstance(previous, list):
                    previous[:] = analyzer._history_messages
                    return True
            frame = frame.f_back
    return False
