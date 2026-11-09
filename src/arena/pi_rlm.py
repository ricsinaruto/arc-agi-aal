"""Arena kind ``pi-rlm``: a Pi-driven agent in the style of Prime Intellect's Prime Agent (the
"RLM harness", arXiv 2608.23552). Generic by construction: the model gets the environment as an
object in a persistent Python kernel, a shared harness of notes and skills it refines from
evidence, and control of its own reasoning effort; no Duck procedures, segmentation library or
mechanics hints.

Pieces:

- ``Harness``: ``<trial output>/harness/`` with ``notes.md`` (general knowledge, bounded at
  ``notes_cap_tokens``, oldest lines dropped), ``skills/*.py`` (executable procedures, imported
  as ``skills.<name>`` into every game's kernel), ``log.jsonl`` (every edit with game, trigger,
  wall seconds and a before-copy path under ``versions/``).
- ``RlmKernel``: the host side of ``rlm_worker.py``, one sandboxed persistent Python process
  per game. ``env.step`` inside the kernel is an RPC the host serves by stepping the
  ``ArenaBridge`` (clock, archive, level reports, ``game.note_action``) exactly as the other
  kinds do; observations are sent back as compact records and stay in the kernel.
- ``RlmExecutor``: Pi's read/write/edit/bash operations; paths under ``./harness`` (a symlink
  in the workspace) are served on the host with versioning and logging, everything else goes
  through the ordinary sandboxed child.
- ``RlmRequestLayer``: the arena request layer with a per-request reasoning-effort override
  read from ``./effort`` in the workspace.
- ``ArenaRlmAgent``: the Pi session. Gameplay prompts and refine prompts are sequential Pi
  prompts on one session: a refine trigger (level clear, or ``stall_actions`` consecutive
  actions without a board state new to the level, tracked here independently of the
  scheduler) interrupts the running gameplay turn at its next request boundary (the same
  mechanism that parks a game), the refine prompt runs with a tool-turn cap enforced at the
  same ``admit`` hook, and the neutral continuation resumes play. A refine that becomes due
  while the game is parked runs first after the resume.
"""

import ast
import asyncio
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

from arc_infra.model_templates import allowed_efforts, template_kwargs, text_only
from arc_infra.perception import board_hash
from qwen_baselines import sandbox as sandbox_module
from qwen_baselines.evidence import png
from qwen_baselines.executor import Executor
from qwen_baselines.jail import Jail, needed
from qwen_baselines.sandbox import kill_group

from .pi import ArenaPiAgent, ArenaRequestLayer, Parked

RLM_DEFAULTS = {
    "stall_actions": 12,          # consecutive actions without a board new to the level
    "refine_tool_turns": 3,       # tool calls admitted in one refine prompt
    "notes_cap_tokens": 2000,     # harness/notes.md bound (and the injection cap)
    "python_seconds": 30,         # default python-tool call limit
    "python_max_seconds": 120,    # the largest model-requested timeout
    "output_chars": 4000,         # tool-visible output cap per python call
}
STORED_OUTPUT_CHARS = 1_000_000   # archived output per python call; beyond it chunks are dropped
SKILL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
INJECTION_LABEL = (
    "[Shared harness] Notes and skills kept across the games of this run (harness/notes.md, "
    "harness/skills/): general knowledge written from evidence in other games, not facts about "
    "this game. Use what the evidence here supports.\n"
)
REFINE_PROMPT = (
    "[Refine turn: {trigger}] The game is paused for this turn (env.step is refused). Update the "
    "shared harness with MINIMAL, evidence-backed edits that would help an agent starting a "
    "different game of this suite:\n"
    "- harness/notes.md: append or replace lines of GENERAL knowledge (mechanics patterns, "
    "kernel/tool tricks that saved actions or time, mistakes to avoid). The file is bounded at "
    "{cap} tokens; oldest lines are dropped first.\n"
    "- harness/skills/<name>.py: create, update or delete an executable procedure (a module "
    "with a docstring whose first line says what it does; to delete one, write an empty file). "
    "Skills are imported as skills.<name> in every game's kernel.\n"
    "Nothing game-specific: no game IDs, coordinates, colours, object names or level numbers of "
    "this board. Use the write and edit tools (read first if unsure); at most {turns} tool calls "
    "are admitted in this turn, then play resumes. Say in one line why each edit is worth "
    "keeping. If nothing is worth keeping, reply NONE and make no edit.\n\n"
    "Current harness:\n{harness}"
)
SYSTEM_PROMPT = """You are an autonomous agent playing an unfamiliar grid game through a persistent Python kernel.

Primary objective: clear as many levels as you can; later levels weigh more in the score.
Secondary objective: use as few game actions as practical. Infer the rules and the goal from
observations and action outcomes: nothing about the mechanics is given. Build your own
strategy, probes, helpers and macros; keep what the evidence supports.

The game is the `env` object in the python tool:
- env.step(action, x=None, y=None) -> dict: execute one action and return the observation
  record: action, levels, win_levels, status (NOT_FINISHED / GAME_OVER / WIN), legal_actions,
  level_completed, game_over, done, reset, board_changed, actions_taken, elapsed_seconds,
  frames (the animation frames of the action, settled last) and frame (the settled board, a
  read-only 64x64 uint8 numpy array; frame[y, x] is a colour ID 0..15).
- env.reset() (RESET the level; counts as an action), env.frame(), env.ascii() (64 rows of
  hex digits 0-F), env.legal_actions(), env.levels(), env.win_levels(), env.state() (the
  latest record without frames), env.history() (every observation of this game, initial
  state first; it lives in the kernel, not in this conversation), env.last().
- Actions are the names in env.legal_actions(): ACTION1..ACTION7 and RESET. ACTION6 takes
  x=column, y=row in 0..63; the others take no coordinates. Coordinates are zero-based.
- A level clear (levels increases), a RESET, GAME_OVER or WIN stops further actions in the
  current call and response; your next response may continue. GAME_OVER is reset by the
  runner before your next response and the reset receipt is reported. An action that already
  produced no board change from the same board is rejected as an error: choose another.
- board_changed reports a difference, not proof of progress. Frames are ordered: animation
  first, settled board last.
{legend}
The python tool is a persistent kernel: variables, functions and imports survive across
calls, levels and resets (a hard timeout restarts it and says so; env.history() is restored,
your variables are not). Use it as your memory and your laboratory: keep observations and
hypotheses in variables, write helpers, diff boards, enumerate or search over your model of
the game, then execute the chosen sequence in one call. Print selectively: the tool returns
at most {output_chars} characters per call and everything you print costs context. Never
print a whole board unless you need it. `np` (numpy) and `skills` are predefined; the
standard library's pure modules are importable. The kernel has no filesystem: use the read,
write, edit and bash tools for files in your workspace (`./`, persistent for this game).
{images}
Shared harness (`./harness`, shared by every game of this run, injected at every resume):
- harness/notes.md: general knowledge, bounded at {notes_cap} tokens (oldest lines dropped).
- harness/skills/<name>.py: executable procedures, imported as skills.<name> in every game's
  kernel; skills.names() lists them. A skill must compile; write an empty file to delete it.
- Refine turns: at every level clear and after {stall} consecutive actions without a board
  state new to the level, the running turn is cut at its next request and you receive one
  refine prompt (at most {refine_turns} tool calls) asking for minimal, evidence-backed
  edits with a one-line reason each; nothing game-specific. Every harness write is versioned
  and logged; you may read the harness at any time.

Reasoning effort: you control it. Write one of {efforts} to `./effort` (a file in your
workspace); it applies from your next request until you change it (the default is
{default_effort}; an invalid value is ignored). Plan, diagnose and search at high effort;
execute known sequences and routine checks at low effort.

Runtime: every request carries the actual gameplay clock as a tool response: the shared
solve target and remaining seconds, this game's active, inference and tool seconds and, in
the steady state, the level budget granted for the current level and its remainder. This
game shares the server with other games under one {seconds:g}-second wall clock: the
scheduler may park a game between requests (the running turn is cut and play resumes
later with the board unchanged) and admits it again on a fresh level budget. Waiting for
inference and tools consumes the clock; setup does not. Time and actions both matter:
probe with single discriminating actions while the mechanics are uncertain, then stop
probing and execute.

A turn ends when you stop calling tools; play then resumes with a neutral continuation.
"""


def estimate_tokens(text, chars_per_token=3.0):
    return int(len(text) / chars_per_token) + 1


def rlm_settings(settings=None):
    """The study's ``rlm`` block over the defaults; unknown keys are rejected."""
    merged = dict(RLM_DEFAULTS)
    for key, value in (settings or {}).items():
        if key not in RLM_DEFAULTS:
            raise ValueError(f"Unknown rlm setting: {key}")
        merged[key] = type(RLM_DEFAULTS[key])(value)
    if merged["python_seconds"] <= 0 or merged["python_max_seconds"] < merged["python_seconds"]:
        raise ValueError("python_seconds must be positive and at most python_max_seconds")
    if min(merged["stall_actions"], merged["refine_tool_turns"], merged["notes_cap_tokens"],
           merged["output_chars"]) < 1:
        raise ValueError("rlm counts must be positive")
    return merged


class Harness:
    """The trial's shared harness directory: notes, skills, the edit log and before-copies."""

    def __init__(self, root, *, notes_cap_tokens=2000):
        self.root = Path(root).resolve()
        self.notes_cap_tokens = int(notes_cap_tokens)
        self.skills_dir = self.root / "skills"
        self.versions = self.root / "versions"
        self.log_path = self.root / "log.jsonl"
        self.notes_path = self.root / "notes.md"
        for folder in (self.root, self.skills_dir, self.versions):
            folder.mkdir(parents=True, exist_ok=True)
        if not self.notes_path.exists():
            self.notes_path.write_text("")
        self.skills_version = 0
        self.edits = 0
        self.notes_lines_added = 0
        self.notes_lines_dropped = 0
        self.skills_created = self.skills_updated = self.skills_deleted = 0
        self.rejected = 0

    # --- reading ------------------------------------------------------------------------------

    def notes_lines(self):
        return [line for line in self.notes_path.read_text().splitlines() if line.strip()]

    def fit_notes(self, lines):
        """``(kept, tokens)``: the newest lines that fit the cap (oldest dropped)."""
        kept, total = [], 0
        for line in reversed(lines):
            cost = estimate_tokens(line + "\n")
            if total + cost > self.notes_cap_tokens:
                break
            kept.append(line)
            total += cost
        kept.reverse()
        return kept, total

    def skills(self):
        return {path.stem: path.read_text() for path in sorted(self.skills_dir.glob("*.py"))
                if SKILL_NAME.match(path.stem)}

    def skills_index(self):
        """``[(name, first docstring line)]``."""
        index = []
        for name, source in self.skills().items():
            try:
                doc = ast.get_docstring(ast.parse(source)) or ""
            except SyntaxError:
                doc = "(syntax error)"
            index.append((name, doc.strip().splitlines()[0] if doc.strip() else ""))
        return index

    def injection(self):
        """``(text, tokens, lines, skills)``: the labelled harness block for a prompt."""
        kept, _ = self.fit_notes(self.notes_lines())
        index = self.skills_index()
        text = INJECTION_LABEL
        text += f"Notes ({len(kept)} lines, oldest first):\n" if kept else "Notes: (none yet)\n"
        text += "".join(f"{line}\n" for line in kept)
        text += "Skills (import as skills.<name> in the python tool):\n" if index else "Skills: (none yet)\n"
        text += "".join(f"- {name}: {doc}\n" if doc else f"- {name}\n" for name, doc in index)
        return text, estimate_tokens(text), len(kept), len(index)

    def relative(self, path):
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.root):
            raise PermissionError("Path outside the shared harness")
        return resolved.relative_to(self.root)

    def read(self, path):
        rel = self.relative(path)
        target = self.root / rel
        if not target.is_file():
            raise FileNotFoundError(f"No such harness file: {rel}")
        return target.read_bytes()

    def exists(self, path):
        return (self.root / self.relative(path)).exists()

    # --- writing ------------------------------------------------------------------------------

    def _version(self, rel, previous):
        stem = str(rel).replace("/", "__")
        number = len(list(self.versions.glob(f"{stem}.*"))) + 1
        copy = self.versions / f"{stem}.{number:04d}"
        copy.write_text(previous)
        return str(copy.relative_to(self.root))

    def write(self, path, content, *, game, trigger, wall_seconds):
        """Write ``notes.md`` or ``skills/<name>.py`` (empty content deletes a skill); the
        previous content is copied under ``versions/`` and the edit is appended to the log.
        Returns the log entry."""
        rel = self.relative(path)
        content = "" if content is None else str(content)
        target = self.root / rel
        previous = target.read_text() if target.is_file() else None
        entry = {"epoch": time.time(), "game": game, "trigger": trigger,
                 "wall_seconds": round(float(wall_seconds), 3), "file": str(rel),
                 "before": None if previous is None else self._version(rel, previous)}
        if rel == Path("notes.md"):
            lines = [line for line in content.splitlines() if line.strip()]
            kept, tokens = self.fit_notes(lines)
            old = set(self.notes_lines())
            added = sum(line not in old for line in kept)
            target.write_text("".join(f"{line}\n" for line in kept))
            entry.update(change="updated", lines=len(kept), lines_added=added,
                         lines_dropped=len(lines) - len(kept), tokens=tokens)
            self.notes_lines_added += added
            self.notes_lines_dropped += len(lines) - len(kept)
        elif rel.parent == Path("skills") and rel.suffix == ".py" and SKILL_NAME.match(rel.stem):
            if not content.strip():
                if previous is None:
                    raise FileNotFoundError(f"No such skill to delete: {rel}")
                target.unlink()
                entry.update(change="deleted")
                self.skills_deleted += 1
            else:
                try:
                    compile(content, f"skills/{rel.name}", "exec")
                except SyntaxError as exc:
                    self.rejected += 1
                    raise ValueError(f"Skill not saved, it does not compile: {exc}") from exc
                target.write_text(content)
                entry.update(change="created" if previous is None else "updated",
                             lines=len(content.splitlines()))
                if previous is None:
                    self.skills_created += 1
                else:
                    self.skills_updated += 1
            self.skills_version += 1
        else:
            self.rejected += 1
            raise PermissionError("Only harness/notes.md and harness/skills/<name>.py are writable")
        self.edits += 1
        with self.log_path.open("a") as log:
            log.write(json.dumps(entry) + "\n")
        return entry

    def trim_notes(self):
        """Drop the oldest notes lines beyond the cap (files carried from an earlier run)."""
        lines = self.notes_lines()
        kept, _ = self.fit_notes(lines)
        if len(kept) != len(lines):
            self.notes_path.write_text("".join(f"{line}\n" for line in kept))
        return len(lines) - len(kept)

    def summary(self):
        return {
            "path": str(self.root), "notes_lines": len(self.notes_lines()),
            "notes_cap_tokens": self.notes_cap_tokens, "skills": [n for n, _ in self.skills_index()],
            "edits": self.edits, "notes_lines_added": self.notes_lines_added,
            "notes_lines_dropped": self.notes_lines_dropped, "skills_created": self.skills_created,
            "skills_updated": self.skills_updated, "skills_deleted": self.skills_deleted,
            "rejected_writes": self.rejected,
        }


class RlmKernel:
    """Host side of ``rlm_worker.py``: start, execute with a wall-clock limit, serve RPCs."""

    def __init__(self, agent, *, seconds, max_seconds):
        self.agent = agent
        self.bridge = agent.bridge
        self.seconds, self.max_seconds = float(seconds), float(max_seconds)
        self.process = None
        self.loaded_skills_version = None
        self.restarts = 0
        self.calls = 0

    def record(self, index=-1):
        """One observation as the kernel receives it: metadata plus base64 frames."""
        archive = self.bridge.archive
        meta = archive.metadata(index)
        frames = [base64.b64encode(archive.frame(meta["index"], i).tobytes()).decode()
                  for i in range(meta["frame_count"])]
        outcome = meta.get("outcome", {})
        return {
            "index": meta["index"], "observation_id": meta["observation_id"],
            "action": meta.get("action"), "levels": meta["levels_completed"],
            "win_levels": meta["win_levels"], "status": meta["status"],
            "legal_actions": list(meta["legal_actions"]),
            "level_completed": outcome.get("level_completed", False),
            "game_over": outcome.get("game_over", False), "done": outcome.get("done", False),
            "reset": outcome.get("reset", False), "board_changed": outcome.get("board_changed"),
            "actions_taken": meta["actions_taken"], "attempt": meta.get("attempt", 0),
            "elapsed_seconds": meta.get("elapsed_seconds"), "frames": frames,
        }

    async def dispatch(self, request):
        method = request["method"]
        if method == "history":
            return [self.record(i) for i in range(self.bridge.archive.count())]
        if method == "step":
            meta = await self.agent.env_step(request["action"], request.get("x"), request.get("y"))
            return self.record(meta["index"])
        raise ValueError("Unknown RPC method")

    async def _serve(self, deadline_seconds):
        """Read worker messages until ``done``; RPCs are served; returns the done message."""
        async with asyncio.timeout(deadline_seconds):
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    raise RuntimeError("Persistent kernel exited")
                message = json.loads(line)
                if message["type"] == "rpc":
                    try:
                        packet = {"result": await self.dispatch(message)}
                    except Exception as exc:  # noqa: BLE001 - the model sees the error text
                        packet = {"error": f"{type(exc).__name__}: {exc}"}
                    self.process.stdin.write((json.dumps(packet) + "\n").encode())
                    await self.process.stdin.drain()
                elif message["type"] in ("done", "ready"):
                    return message
                elif message["type"] == "output":
                    self.on_output(message["text"])
                else:
                    raise RuntimeError(f"Unexpected kernel message: {message['type']}")

    def on_output(self, text):  # replaced per call
        pass

    async def start(self):
        jail = await asyncio.to_thread(Jail) if needed() else None
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-u", str(Path(__file__).with_name("rlm_worker.py")),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True, limit=512 * 1024,
            env={
                **({"ARC_CHROOT": str(jail.root)} if jail else {}),
                "PATH": os.defpath,
                "ARC_SANDBOX_MODULE": str(Path(sandbox_module.__file__)),
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        try:
            ready = await self._serve(60)
        except (TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            error = await self.process.stderr.read() if self.process.returncode is not None else b""
            await self.close()
            raise RuntimeError(f"Persistent kernel failed to start: {exc}: {error.decode()[-2000:]}") from exc
        if ready.get("type") != "ready":
            await self.close()
            raise RuntimeError("Persistent kernel did not report ready")
        self.loaded_skills_version = None
        self.bridge.archive.event("rlm_kernel_started", sandbox=ready.get("sandbox"), restarts=self.restarts)

    async def send(self, packet):
        self.process.stdin.write((json.dumps(packet) + "\n").encode())
        await self.process.stdin.drain()

    async def sync_skills(self, harness):
        """Load the harness skills into the kernel when they changed; import errors by name."""
        if self.process is None or self.loaded_skills_version == harness.skills_version:
            return {}
        version = harness.skills_version
        await self.send({"type": "skills", "sources": harness.skills()})
        done = await self._serve(min(self.seconds, max(0.1, self.bridge.clock.remaining())))
        self.loaded_skills_version = version
        errors = done.get("skills") or {}
        if errors:
            self.bridge.archive.event("rlm_skills_import_error", errors=errors)
        return errors

    async def execute(self, code, timeout=None, *, harness=None):
        """Run ``code``; returns the archived transcript plus ``skills_errors``."""
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        limit = self.seconds if not timeout else min(self.max_seconds, max(0.1, float(timeout)))
        archive = self.bridge.archive
        call = archive.begin_call(code)
        started = time.monotonic()
        before = archive.count()
        restarted, error, skills_errors = False, None, {}
        stored = {"chars": 0, "dropped": 0}

        def on_output(text):
            if stored["chars"] < STORED_OUTPUT_CHARS:
                archive.output(call, text[: STORED_OUTPUT_CHARS - stored["chars"]])
                stored["chars"] += len(text)
            else:
                stored["dropped"] += len(text)

        try:
            if self.process is None:
                async with asyncio.timeout(min(limit, max(0.1, self.bridge.clock.remaining()))):
                    await self.start()
            self.on_output = on_output  # import-time prints of a skill land in this call too
            if harness is not None:
                skills_errors = await self.sync_skills(harness)
            self.calls += 1
            await self.send({"type": "execute", "code": code})
            done = await self._serve(min(limit - (time.monotonic() - started),
                                         max(0.1, self.bridge.clock.remaining())))
            error = done.get("error")
        except (TimeoutError, RuntimeError, BrokenPipeError, ValueError) as exc:
            error = (f"{type(exc).__name__}: {exc}. The kernel was restarted: your variables are "
                     "lost (env.history() is restored); completed actions remain recorded. "
                     "Code was not replayed.")
            archive.output(call, error)
            restarted = True
            self.restarts += 1
            await self.close()
        except asyncio.CancelledError:
            archive.end_call(call, status="cancelled", namespace_lost=True,
                             error="Gameplay deadline; completed actions remain recorded")
            await self.close()
            raise
        finally:
            self.on_output = lambda text: None
            archive.event("python_time", call_id=call, seconds=time.monotonic() - started)
        archive.end_call(
            call, status="error" if error else "complete", error=error, namespace_lost=restarted,
            observations=list(range(before, archive.count())), dropped_output_chars=stored["dropped"],
        )
        return {**archive.transcript(call), "skills_errors": skills_errors,
                "dropped_output_chars": stored["dropped"]}

    async def close(self):
        if self.process:
            kill_group(self.process)
            await self.process.wait()
            self.process = None


class RlmExecutor(Executor):
    """Pi's file and shell operations; ``./harness`` paths are served on the host."""

    def __init__(self, agent, workspace, harness, *, suppress_dumps):
        super().__init__(agent.bridge, workspace, api=False, evidence=True, suppress_dumps=suppress_dumps)
        self.agent, self.harness = agent, harness
        link = self.workspace / "harness"
        if not link.is_symlink() and not link.exists():
            link.symlink_to(harness.root, target_is_directory=True)

    def harness_path(self, args):
        path = Path(args["path"])
        if not path.is_absolute():
            path = self.workspace / path
        resolved = path.resolve()
        return resolved if resolved.is_relative_to(self.harness.root) else None

    async def execute(self, operation, **args):
        target = self.harness_path(args) if operation != "bash" else None
        if target is None:
            return await super().execute(operation, **args)
        if self.bridge.clock.remaining() <= 0:
            raise TimeoutError("Gameplay deadline reached")
        rel = self.harness.relative(target)
        archive = self.bridge.archive
        call = archive.begin_call(json.dumps({"operation": operation, "path": str(target), "harness": True}))
        try:
            output = ""
            if operation == "read":
                output = base64.b64encode(self.harness.read(target)).decode() + "\n"
            elif operation == "access":
                if not self.harness.exists(target):
                    raise FileNotFoundError(f"No such harness file: {rel}")
            elif operation == "mkdir":
                if rel not in (Path("."), Path("skills")):
                    raise PermissionError("Only harness/skills exists; new directories are not allowed")
            elif operation == "write":
                entry = self.harness.write(
                    target, args.get("content", ""), game=self.agent.game.game_id,
                    trigger=self.agent.edit_trigger(),
                    wall_seconds=self.agent.game.clock.global_clock.elapsed(),
                )
                self.agent.record_edit(entry)
            else:
                raise ValueError("Unknown tool operation")
            archive.end_call(call, status="complete", exit_code=0)
            return {"exitCode": 0, "output": output}
        except BaseException as exc:
            archive.end_call(call, status="error", error=str(exc))
            raise


class RlmRequestLayer(ArenaRequestLayer):
    """The arena request layer with the model-chosen reasoning effort for the next request."""

    effort = None

    def template_settings(self, *, summary=False):
        return template_kwargs(self.config.model, {
            "enable_thinking": not summary,
            "preserve_thinking": self.config.preserve_thinking,
            "reasoning_effort": self.effort or self.config.reasoning_effort,
        })


class RefineCut(TimeoutError):
    """Raised inside the gateway to cut a turn for a due refine prompt or at its tool cap."""


class ArenaRlmAgent(ArenaPiAgent):
    TOOLS = ("read", "write", "edit", "bash", "python")

    def __init__(self, bridge, layer, *, pi_dir, node="node", game, harness, settings=None):
        self.harness = harness
        self.settings = rlm_settings(settings)
        super().__init__(bridge, layer, pi_dir=pi_dir, node=node, game=game)
        self.kernel = RlmKernel(self, seconds=self.settings["python_seconds"],
                                max_seconds=self.settings["python_max_seconds"])
        self.python_lock = asyncio.Lock()
        self.mode = "gameplay"          # or "refine"
        self.refine_due = None          # the trigger text while a refine prompt is pending
        self.refine_id = 0
        self.cuts = 0                   # turns cut for a refine (due or at its tool cap)
        self.seen_boards = set()
        self.stalled = 0
        self.level_seen = None
        self.effort_source = "default"
        self.invalid_effort_seen = None
        self.stats = {
            "refine_turns": 0, "refine_complete": 0, "refine_cut": 0, "refine_parked": 0,
            "refine_errors": 0, "clear_triggers": 0, "stall_triggers": 0, "harness_edits": 0,
            "notes_lines_added": 0, "skills_created": 0, "skills_updated": 0, "skills_deleted": 0,
            "injections": 0, "injected_tokens": 0, "python_calls": 0, "python_restarts": 0,
            "python_actions": 0, "effort_requests": {}, "effort_invalid": 0,
        }

    # --- construction ---------------------------------------------------------------------

    def make_executor(self):
        return RlmExecutor(self, self.bridge.archive.root / "workspace", self.harness,
                           suppress_dumps=self.layer.config.suppress_grid_dumps)

    def system_prompt(self):
        cfg = self.layer.config
        model = cfg.model
        efforts = allowed_efforts(model)
        if text_only(model):
            images = ("This model reads boards as text: env.ascii() and env.frame() are your views of "
                      "the board; no images are available.\n")
            legend = "Cell values 0..15 are colour IDs; env.ascii() writes them as hex digits 0-F."
        else:
            images = ("The initial board is shown once as an image. Every observation's frames are "
                      "also exported under ./evidence/<observation_id>/ as frame-NNN.png (view with "
                      "the read tool) and .npy grids; the kernel's arrays are the same data.\n")
            legend = ("Images show the 64x64 board; cell values are colour IDs 0..15 (env.ascii() writes "
                      "them as hex digits 0-F).")
        return SYSTEM_PROMPT.format(
            legend=legend, images=images, output_chars=f"{self.settings['output_chars']:,}",
            notes_cap=f"{self.settings['notes_cap_tokens']:,}", stall=self.settings["stall_actions"],
            refine_turns=self.settings["refine_tool_turns"],
            efforts=", ".join(efforts) if len(efforts) > 1 else efforts[0],
            default_effort=cfg.reasoning_effort, seconds=cfg.gameplay_seconds,
        )

    # --- harness injection ------------------------------------------------------------------

    def harness_block(self):
        text, tokens, lines, skills = self.harness.injection()
        self.stats["injections"] += 1
        self.stats["injected_tokens"] += tokens
        self.bridge.archive.event("rlm_inject", tokens=tokens, notes_lines=lines, skills=skills)
        return text

    def initial_prompt(self):
        text = "Initial observation: " + json.dumps(self.bridge.archive.metadata())
        return text + "\n\n" + self.harness_block()

    def continuation(self):
        text = "Continue when ready. The game remains available."
        if self.resumed_after is not None:
            text = (f"Play resumed after a pause of {self.resumed_after:.0f} s; the board is "
                    "unchanged.\n" + text)
        return text + "\n\n" + self.harness_block()

    def refine_prompt(self, trigger):
        harness, tokens, lines, skills = self.harness.injection()
        self.bridge.archive.event("rlm_inject", tokens=tokens, notes_lines=lines, skills=skills,
                                  refine=self.refine_id)
        self.stats["injections"] += 1
        self.stats["injected_tokens"] += tokens
        return REFINE_PROMPT.format(trigger=trigger, cap=f"{self.settings['notes_cap_tokens']:,}",
                                    turns=self.settings["refine_tool_turns"], harness=harness)

    # --- harness edits --------------------------------------------------------------------

    def edit_trigger(self):
        return f"refine:{self.refine_id}" if self.mode == "refine" else "gameplay"

    def record_edit(self, entry):
        self.stats["harness_edits"] += 1
        self.stats["notes_lines_added"] += entry.get("lines_added", 0)
        if entry["file"].startswith("skills/"):
            self.stats[{"created": "skills_created", "updated": "skills_updated",
                        "deleted": "skills_deleted"}[entry["change"]]] += 1
        self.bridge.archive.event("rlm_harness_edit", **entry)

    # --- reasoning effort -----------------------------------------------------------------

    def effort_for_request(self):
        """``(effort, source)`` from ``./effort`` in the workspace, else the study's effort."""
        cfg = self.layer.config
        try:
            raw = (self.executor.workspace / "effort").read_text()
        except OSError:
            return cfg.reasoning_effort, "default"
        value = raw.strip().split()[0].lower() if raw.strip() else ""
        if not value:
            return cfg.reasoning_effort, "default"
        if value in allowed_efforts(cfg.model):
            return value, "file"
        if value != self.invalid_effort_seen:
            self.invalid_effort_seen = value
            self.stats["effort_invalid"] += 1
            self.bridge.archive.event("rlm_effort_invalid", value=value[:40],
                                      allowed=list(allowed_efforts(cfg.model)))
        return cfg.reasoning_effort, "default_invalid_file"

    async def gateway(self, request):
        self.layer.effort, self.effort_source = self.effort_for_request()
        return await super().gateway(request)

    # --- request boundaries ---------------------------------------------------------------

    async def admit(self):
        self.turns += 1
        if self.game.clock.parked:
            self.parked_turns += 1
            self.bridge.archive.event("arena_parked_request", turns=self.turns, mode=self.mode)
            await self.interrupt()
            raise Parked("Game parked at a request boundary")
        if self.mode == "refine":
            if self.turns > self.settings["refine_tool_turns"] + 1:
                self.cuts += 1
                self.bridge.archive.event("rlm_refine_cap", refine=self.refine_id, turns=self.turns)
                await self.interrupt()
                raise RefineCut("Refine turn tool cap reached")
        elif self.refine_due is not None:
            self.cuts += 1
            self.bridge.archive.event("rlm_refine_interrupt", trigger=self.refine_due, turns=self.turns)
            await self.interrupt()
            raise RefineCut("Refine turn due; the gameplay turn is cut")
        effort = self.layer.effort or self.layer.config.reasoning_effort
        self.stats["effort_requests"][effort] = self.stats["effort_requests"].get(effort, 0) + 1
        self.bridge.archive.event("rlm_effort", effort=effort, source=self.effort_source,
                                  mode=self.mode, turns=self.turns)

    async def begin_turn(self):
        if self.mode == "refine":
            return  # no automatic RESET and no boundary reset while play is paused
        await super().begin_turn()

    # --- the environment ------------------------------------------------------------------

    async def env_step(self, action, x, y):
        """``env.step`` from the kernel: the bridge (clock, archive, level reports) plus the
        adapter's own board-identity tracking for the stall trigger."""
        if self.mode == "refine":
            raise RuntimeError("Refine turn: the game is paused; no actions until play resumes")
        archive = self.bridge.archive
        before = archive.metadata()["levels_completed"]
        meta = await self.bridge.step(action, x, y)
        self.stats["python_actions"] += 1
        key = (meta["levels_completed"], board_hash(archive.frame()))
        self.game.note_action(key)
        if meta["levels_completed"] > before:
            self.seen_boards = {key}
            self.stalled = 0
            self.stats["clear_triggers"] += 1
            self.refine_due = self.refine_due or (
                f"level {meta['levels_completed']} of {meta['win_levels']} cleared after "
                f"{self.bridge.actions} actions")
        elif key in self.seen_boards:
            self.stalled += 1
            if self.stalled >= self.settings["stall_actions"]:
                self.stats["stall_triggers"] += 1
                self.stalled = 0
                self.refine_due = self.refine_due or (
                    f"{self.settings['stall_actions']} consecutive actions without a board state "
                    "new to this level")
        else:
            self.seen_boards.add(key)
            self.stalled = 0
        return meta

    async def python_call(self, code, timeout=None):
        if self.bridge.clock.remaining() <= 0:
            raise TimeoutError("Gameplay deadline reached")
        async with self.python_lock:
            with self.game.clock.tool("python"):
                result = await self.kernel.execute(code, timeout, harness=self.harness)
        self.stats["python_calls"] += 1
        self.stats["python_restarts"] = self.kernel.restarts
        output = result["output"]
        cap = self.settings["output_chars"]
        text = output
        if len(output) > cap:
            text = output[:cap] + (f"\n[output truncated at {cap:,} characters; {len(output):,} were "
                                   "produced. Print less or keep results in variables.]")
        notes = []
        for name, error in result["skills_errors"].items():
            notes.append(f"[skills] {name}.py failed to import: {error.strip().splitlines()[-1]}")
        executed = len(result.get("observations") or [])
        if executed:
            last = self.bridge.archive.metadata()
            notes.append(f"[env] {executed} action(s) executed in this call; levels "
                         f"{last['levels_completed']}/{last['win_levels']}, status {last['status']}, "
                         f"{last['actions_taken']} actions taken in this game.")
        if notes:
            text = (text + "\n" if text and not text.endswith("\n") else text) + "\n".join(notes)
        return {"output": text, "details": {
            "call_id": result["call_id"], "status": result["status"], "actions": executed,
            "namespace_lost": result.get("namespace_lost", False), "chars": len(output)}}

    async def handle_rpc(self, message):
        if message.get("operation") != "python":
            return await super().handle_rpc(message)
        try:
            result = await self.python_call(message["code"], message.get("timeout"))
            await self.send({"type": "reply", "id": message["id"], "result": result})
        except Exception as exc:  # noqa: BLE001 - the model sees the error text
            await self.send({"type": "reply", "id": message["id"], "error": f"{type(exc).__name__}: {exc}"})

    # --- prompts --------------------------------------------------------------------------

    async def refine_turn(self):
        """One refine prompt; returns ``complete``, ``cut``, ``parked`` or ``error``."""
        trigger, self.refine_due = self.refine_due, None
        self.refine_id += 1
        self.mode = "refine"
        self.turns, self.text, self.interrupting = 0, "", False
        parked_before, cuts_before = self.parked_turns, self.cuts
        edits_before = self.stats["harness_edits"]
        self.stats["refine_turns"] += 1
        self.bridge.archive.event("rlm_refine_start", refine=self.refine_id, trigger=trigger,
                                  actions=self.bridge.actions)
        try:
            await self.send({"type": "prompt", "text": self.refine_prompt(trigger)})
            await self.receive_until("done", raise_model_errors=False)
        finally:
            self.mode = "gameplay"
        if self.parked_turns > parked_before:
            outcome = "parked"
        elif self.cuts > cuts_before:
            outcome = "cut"
        elif self.last_model_error:
            outcome = "error"
        else:
            outcome = "complete"
        self.stats[{"complete": "refine_complete", "cut": "refine_cut", "parked": "refine_parked",
                    "error": "refine_errors"}[outcome]] += 1
        self.bridge.archive.event("rlm_refine_end", refine=self.refine_id, trigger=trigger,
                                  outcome=outcome, edits=self.stats["harness_edits"] - edits_before,
                                  tool_turns=max(0, self.turns - 1), reply=self.text[:2000])
        return outcome

    async def run(self):
        """``ArenaPiAgent.run`` with refine prompts sequenced between gameplay prompts."""
        initial = True
        errors = 0
        clock = self.game.clock
        self.resumed_after = None
        try:
            self.seen_boards = {(self.bridge.archive.metadata()["levels_completed"],
                                 board_hash(self.bridge.archive.frame()))}
            while (
                clock.global_clock.remaining() > 0
                and self.bridge.archive.metadata()["status"] != "WIN"
            ):
                await clock.wait_active()
                paused = clock.parked_seconds() - self._parked_at_prompt
                if paused > 0:
                    self.resumed_after = paused
                if clock.global_clock.remaining() <= 0:
                    break
                self._parked_at_prompt = clock.parked_seconds()
                if self.refine_due is not None:
                    outcome = await self.refine_turn()
                    if outcome == "error":
                        errors += 1
                        self.bridge.archive.event("pi_turn_error", error=self.last_model_error,
                                                  consecutive=errors, refine=self.refine_id)
                        if errors >= 3:
                            raise RuntimeError(self.last_model_error)
                    elif outcome == "complete":
                        errors = 0
                    continue
                text = self.initial_prompt() if initial else self.continuation()
                packet = {"type": "prompt", "text": text}
                self.turns, self.text = 0, ""
                self.interrupting = False
                self.mode = "gameplay"
                parked_before, cuts_before = self.parked_turns, self.cuts
                if initial and not text_only(self.layer.config.model):
                    packet["images"] = [{
                        "type": "image", "mimeType": "image/png",
                        "data": base64.b64encode(png(self.bridge.archive.frame())).decode(),
                    }]
                self.resumed_after = None
                await self.send(packet)
                await self.receive_until("done", raise_model_errors=False)
                if self.parked_turns > parked_before:
                    self.bridge.archive.event("arena_parked_turn", turns=self.turns)
                elif self.cuts > cuts_before:
                    self.bridge.archive.event("rlm_turn_cut", turns=self.turns, trigger=self.refine_due)
                elif self.last_model_error:
                    errors += 1
                    self.bridge.archive.event("pi_turn_error", error=self.last_model_error, consecutive=errors)
                    if errors >= 3:
                        raise RuntimeError(self.last_model_error)
                else:
                    errors = 0
                if not initial:
                    self.bridge.archive.event("neutral_continuation")
                initial = False
        except (TimeoutError, asyncio.CancelledError):
            self.bridge.archive.event("deadline", seconds=self.bridge.clock.elapsed())
        finally:
            await self.close()

    def rlm_row(self):
        return {**self.stats, "python_restarts": self.kernel.restarts, "turns_cut": self.cuts}

    async def close(self):
        await self.kernel.close()
        await super().close()
