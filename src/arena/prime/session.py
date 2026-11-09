"""Prime Agent's scored ARC-AGI-3 session (prime-agent v0.3.3, one session per game) on the arena's Duck agent.

The request loop stays the Duck ``ToolAgent.analyze`` (transport, KV gate, token ledger); what the model reads and the
context policy are Prime's: its system prompt, the continual harness (``rlm.harness``, ``refine``, ``compact`` in the
kernel), the serialized auto-refine checkpoint before each model step and Pi's compaction. Arm N keeps the native game
interface (per-turn observation, ``action()``, guards); arm F plays through their broker client with their guidance.
Record: docs/prime-arm.md."""

import ast
import asyncio
import concurrent.futures
import copy
import dataclasses
import json
import re
import statistics
import types

import httpx

from duck_transfer.agent import ContextLengthRejected, GameplayRequestExpired

from . import texts
from .store import CONTENT_LIMIT, HarnessStore, compact_text, parse_proposal, parse_review

ARMS = ("N", "F")
DEFAULTS = {
    "arm": "N",
    "compaction": "prime",            # "native": the study's own context policy (drain_half), refinement unchanged
    "state_in_system": False,         # True: v0.3.3's form (an applied edit rebuilds the system prompt's state block)
    "reserve_tokens": 16384,          # Pi: summary caps are 0.8x (history) and 0.5x (turn prefix) of it
    "keep_recent_tokens": 20000,
    "summary_room_tokens": 8192,      # (ours) below this room the summary request covers the cut prefix only
    "summary_thinking": "gameplay",   # Pi: the session's thinking level; "prefilled-empty": none
    "refine": True,
    "refine_turns": 25,
    "refine_cooldown_tokens": 36000,  # (ours) their 20 min cooldown in generated tokens
    "review_tokens": 1024,
    "plan_tokens": 4096,
    "output_bytes": 51200,            # F: tool output tail
    "output_lines": 2000,
    "batch": 20,                      # F: actions per act()
    # N on a scripted level-bench point: plain v8.2 while the recorded replies replay; at the first live request the
    # session takes over as at a compaction (summary, newest 20K kept, playbook seeded as memories, system switched)
    "takeover": False,
    # (ours, off in run 1) what a v8.2 cut carries, under their compaction: entries in full in the harness block, and
    # the harness-written inventory of the kernel's live definitions after the summary
    "digest_full": False,
    "inventory": False,
    # (ours, run 2) "level_clear": review and planner once after each cleared level instead of every refine_turns
    # replies and after each compaction (no cooldown); notices False: an automatic refinement adds no message, its
    # edits reach the model in the next harness block
    "refine_at": "interval",
    "notices": True,
    # RSI loop candidates (docs/rsi-arm.md): text the candidate adds or changes; empty = the Prime arm's session
    "extra_system": "",               # appended to the system prompt's project context
    "seed_entries": [],               # [{"kind": "prompt" | "memory" | "skill", "title", "content"}], in the system prompt
    "text_overrides": {},             # {name in texts.py: text}: per-session replacement (placeholders kept)
    "native_edits": [],               # [{"old", "new"}] on the native (v8.2) system text; each old must occur once
}
OVERRIDABLE = ("HEAD", "IPYTHON_CONTROL_PROMPT", "REFINE_PARAGRAPH", "HARNESS_STATE_HEAD", "AUTONOMOUS_CONTINUATION_PROMPT",
               "SUMMARIZATION_SYSTEM_PROMPT", "SUMMARIZATION_PROMPT", "UPDATE_SUMMARIZATION_PROMPT",
               "TURN_PREFIX_SUMMARIZATION_PROMPT", "KERNEL_PERSIST_SUMMARY_NOTE", "REFINEMENT_SYSTEM_PROMPT",
               "AUTO_REFINE_REVIEW_SYSTEM_PROMPT", "REFINE_SCOPE_LOCAL", "REFINE_PLAN_TAIL", "REFINE_REVIEW_TAIL",
               "AUTO_REFINE_INSTRUCTIONS", "SKILLS_HEAD", "SKILL_REFINE_DESCRIPTION", "SKILL_REFINE_MD",
               "SKILL_COMPACT_DESCRIPTION", "SKILL_COMPACT_MD")
TEXT_LIMIT = 20000
SEED_ENTRIES = 40
# (ours) Once, in the takeover's summary message: a native cut carries the playbook in full, so the takeover does too.
TAKEOVER_NOTE = (
    "\n\n[Session change] From this message on the continual harness replaces your playbook: `playbook()` is retired. "
    "Record what must outlive the context with `rlm.harness.create_memory(title, content)` and "
    "`await refine.run()`; read entries in full with `rlm.harness.list('memory')`.")
TAKEOVER_PLAYBOOK = (" Your playbook is stored as the memory entries `playbook_01` to `playbook_{count:02d}` (path "
                     "`playbook`). The playbook as you last wrote it:\n\n{text}")
DEADLINE = (TimeoutError, asyncio.CancelledError, concurrent.futures.CancelledError, GameplayRequestExpired)
HEADINGS = ("## Goal", "## Constraints & Preferences", "## Progress", "## Key Decisions", "## Next Steps",
            "## Critical Context")
HARNESS_DIGEST_PREFIX = ("[harness-digest]\n\nThe persistent memories produced across this session so far:\n\n"
                         "<harness_state>\n")  # v0.10.0 (763094e1) session_engine/messages.rs
HARNESS_DIGEST_SUFFIX = "\n</harness_state>"
MODULES = ("bisect, collections, copy, fractions, functools, hashlib, heapq, itertools, json, math, operator, random, "
           "re, statistics, string, textwrap, time, traceback")
# Our words: what differs from the kernel their prompt describes.
ENVIRONMENT = (
    "Session environment. The one tool is named `python`: it is the persistent kernel that the text below calls "
    f"IPython. It is a restricted Python (standard library only: {MODULES}; no numpy, no files, no network and no "
    "shell, so `%%bash`, `%` magics and `!cmd` do not exist) and a call is stopped after 30 seconds. Top-level `await` "
    "works and the value of a call's last expression is shown, as in IPython. `rlm.harness`, `refine` and `compact` "
    "work as described; `rlm(...)` sub-agents are disabled in this session and harness entries are local to this game."
)
DELIVERY = (" Changes to harness entries reach you as `[auto-refinement]` or `[self-refinement]` messages, and the "
            "full list is sent again as a `[harness-digest]` message after every compaction.")
DELIVERY_SILENT = (" Automatic refinement runs after each cleared level; the list of harness entries is sent as a "
                   "`[harness-digest]` message after every cut of the conversation.")
F_WORKSPACE = (" `/workspace/AGENTS.md` is the Project Context section of this prompt and "
               "`/workspace/fixed_broker_client.py` is the module `fixed_broker_client` "
               "(`import fixed_broker_client as game`); there are no files to read.")
F_TOOLS = [{"type": "function", "function": {
    "name": "python",
    "description": "Execute Python code in the persistent kernel. Variables, imports and helpers persist across "
                   "calls.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to run."}},
                   "required": ["code"]}}}]
F_SENTINEL = "[prime-turn]"
PUBLIC_FOLLOWUPS = ("You have not acted yet. Investigate first. ", "You did not call a tool. We detected ")
PREVIOUS_POINTER = ("<previous-summary>\n(the summary in the message above that begins \"The conversation history "
                    "before this point was compacted into the following summary:\")\n</previous-summary>\n\n")
SUMMARY_RETRY = "\n\nYour previous reply contained no text. Write the summary now as plain text, starting with '{}'."
MAGIC_CODE = ('raise RuntimeError("Shell cells and IPython magics (%%bash, %magic, !cmd) are not available: this '
              'kernel is a restricted Python without files or a shell.")')


def settings(value):
    """The study's ``prime`` block with defaults, or None when it is off."""
    if not value:
        return None
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError(f"prime: a mapping with keys from {sorted(DEFAULTS)}")
    block = {**DEFAULTS, **value}
    if block["arm"] not in ARMS:
        raise ValueError(f"prime arm must be one of {ARMS}")
    if block["compaction"] not in ("prime", "native"):
        raise ValueError("prime compaction is 'prime' or 'native'")
    if block["summary_thinking"] not in ("gameplay", "prefilled-empty"):
        raise ValueError("prime summary_thinking is 'gameplay' or 'prefilled-empty'")
    if block["arm"] == "F" and block["compaction"] != "prime":
        raise ValueError("prime arm F has no native context policy")
    if block["refine_at"] not in ("interval", "level_clear"):
        raise ValueError("prime refine_at is 'interval' or 'level_clear'")
    if block["takeover"] and (block["arm"] != "N" or block["compaction"] != "prime" or block["state_in_system"]):
        raise ValueError("prime takeover is arm N with its own compaction and the harness block in messages")
    candidate_checks(block)
    return block


def placeholders(text):
    return sorted(set(re.findall(r"\{[A-Za-z_]*\}", text)))


def candidate_checks(block):
    """The RSI candidate keys: types, sizes, and overrides that keep each text's format placeholders."""
    if not isinstance(block["extra_system"], str) or len(block["extra_system"]) > TEXT_LIMIT:
        raise ValueError(f"prime extra_system: a string of at most {TEXT_LIMIT} characters")
    seeds = block["seed_entries"]
    if (not isinstance(seeds, list) or len(seeds) > SEED_ENTRIES or any(
            not isinstance(s, dict) or set(s) != {"kind", "title", "content"} or s["kind"] not in ("prompt", "memory", "skill")
            or not isinstance(s["title"], str) or not isinstance(s["content"], str) or len(s["content"]) > 4000
            or len(s["title"]) > 120 for s in seeds)):
        raise ValueError("prime seed_entries: up to 40 {kind: prompt|memory|skill, title, content (<= 4,000 chars)}")
    overrides = block["text_overrides"]
    if not isinstance(overrides, dict) or set(overrides) - set(OVERRIDABLE):
        raise ValueError(f"prime text_overrides: keys from {OVERRIDABLE}")
    for name, value in overrides.items():
        if not isinstance(value, str) or not value.strip() or len(value) > TEXT_LIMIT:
            raise ValueError(f"prime text_overrides.{name}: a non-empty string of at most {TEXT_LIMIT} characters")
        if placeholders(value) != placeholders(getattr(texts, name)):
            raise ValueError(f"prime text_overrides.{name} must keep the placeholders {placeholders(getattr(texts, name))}")
    edits = block["native_edits"]
    if (not isinstance(edits, list) or len(edits) > 20 or any(
            not isinstance(e, dict) or set(e) != {"old", "new"} or not isinstance(e["old"], str) or not e["old"]
            or not isinstance(e["new"], str) or len(e["new"]) > TEXT_LIMIT for e in edits)):
        raise ValueError("prime native_edits: up to 20 {old (non-empty), new}")


def role(message):
    return str(message.get("role", "")).strip()


# --- the kernel side ---------------------------------------------------------------------------------------------------

ANCHOR = "    request = initial\n    while True:\n"
# Inside the sandbox child's ``main()`` before the request loop (after every native block): ``rlm``, ``refine`` and
# ``compact`` as pre-imported modules. Calls go to the host over the sandbox's ``animation`` request channel.
CHILD_BLOCK = '''
    def _prime_call(op, **args):
        interrupt = globals().get("_INTERRUPT")
        armed = bool(interrupt and interrupt["armed"])
        if interrupt:
            interrupt["armed"] = False
        try:
            _send({"type": "animation", "request": {"prime": op, "args": _json_safe(args)}})
            reply = _recv()
        finally:
            if interrupt:
                interrupt["armed"] = armed
        payload = reply.get("animation") if reply.get("type") == "animation_result" else None
        if not isinstance(payload, dict):
            raise RuntimeError("The session host did not answer this call.")
        if not payload.get("ok"):
            raise RuntimeError(str(payload.get("error") or "call refused"))
        return payload.get("value")

    def _prime_await(value):
        if type(value).__name__ == "coroutine":
            try:
                value.send(None)
            except StopIteration as stop:
                return stop.value
            value.close()
            raise RuntimeError("This kernel has no event loop: only harness, refine and compact calls can be awaited.")
        return value

    def _prime_display(value):
        if value is None or (action_results and value is action_results[-1]):
            return  # an action's receipt is the tool result already
        runtime_globals["_"] = value
        print(repr(value))

    def _prime_local(fields):
        if fields.pop("global_", False):
            raise RuntimeError("Global harness entries are not available in this session; entries are local.")
        fields.pop("scope", None)
        return fields

    class _PrimeHarness(object):
        def create(self, kind, title, content, **fields):
            return _prime_call("harness.create", kind=kind, title=title, content=content, **_prime_local(fields))

        def upsert(self, kind, title, content, **fields):
            return _prime_call("harness.upsert", kind=kind, title=title, content=content, **_prime_local(fields))

        def update(self, kind, id, title, content, **fields):
            return _prime_call("harness.update", kind=kind, id=id, title=title, content=content,
                               **_prime_local(fields))

        def delete(self, kind, id, **fields):
            _prime_local(fields)
            return _prime_call("harness.delete", kind=kind, id=id)

        def get(self, kind, id, **fields):
            return _prime_call("harness.get", kind=kind, id=id)

        def list(self, kind=None, **fields):
            return _prime_call("harness.list", kind=kind)

        def overview(self, max_entries_per_kind=20, **fields):
            return _prime_call("harness.overview", max_entries_per_kind=max_entries_per_kind)

        def snapshot(self, **fields):
            return _prime_call("harness.list", kind=None)

        def record_refinement(self, trigger, changes, evidence="", outcome="", id=None, **fields):
            return _prime_call("harness.record_refinement", trigger=trigger, changes=changes, evidence=evidence,
                               outcome=outcome, id=id)

    def _prime_kind(kind, path):
        def create(self, title, content, **fields):
            fields.setdefault("path", path)
            return self.create(kind, title, content, **fields)

        def update(self, id, title, content, **fields):
            return self.update(kind, id, title, content, **fields)

        def delete(self, id, **fields):
            return self.delete(kind, id, **fields)

        return create, update, delete

    for _prime_name, _prime_kind_name, _prime_path in (("memory", "memory", "general"), ("prompt_note", "prompt", "policy"),
                                                       ("skill", "skill", "general"), ("subagent", "subagent", "general")):
        _prime_c, _prime_u, _prime_d = _prime_kind(_prime_kind_name, _prime_path)
        setattr(_PrimeHarness, "create_" + _prime_name, _prime_c)
        setattr(_PrimeHarness, "update_" + _prime_name, _prime_u)
        setattr(_PrimeHarness, "delete_" + _prime_name, _prime_d)

    class _PrimeRlm(object):
        def __init__(self):
            self.harness = _PrimeHarness()

        def get_harness_state(self):
            return self.harness

        def __call__(self, *args, **kwargs):
            raise RuntimeError("Sub-agents are disabled in this session: rlm(...) cannot be called (rlm.harness works).")

    class _PrimeRefine(object):
        def run(self, instructions=None, global_=False, **fields):
            if global_:
                return {"scheduled": False, "reason": "global refinement is not available in this session"}
            return _prime_call("refine.run", instructions=instructions)

        def status(self):
            return _prime_call("refine.status")

    class _PrimeCompact(object):
        def run(self, instructions=None, **fields):
            return _prime_call("compact.run", instructions=instructions)

        def status(self):
            return _prime_call("compact.status")

    _PRIME_MODULES = {"rlm": _PrimeRlm(), "refine": _PrimeRefine(), "compact": _PrimeCompact()}
    _PRIME_RETIRED = []

    def _prime_retired(*args, **kwargs):
        raise RuntimeError("playbook() is retired in this session: its text is now in the harness memories "
                           "(rlm.harness.list('memory')); record new facts with rlm.harness.create_memory(title, content).")

    def _prime_takeover():
        _PRIME_RETIRED.append(True)
        for _retired in ("playbook", "won", "keep_rule"):
            if _retired in runtime_globals:
                runtime_globals[_retired] = _prime_retired

    _prime_refresh_all = _refresh_state

    def _refresh_state(state_payload):
        _prime_refresh_all(state_payload)
        if _PRIME_RETIRED:
            _prime_takeover()

    runtime_globals["_prime_takeover"] = _prime_takeover
__PRIME_F__
    _prime_import = runtime_globals["__builtins__"]["__import__"]

    def _prime_safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in _PRIME_MODULES and not level:
            return _PRIME_MODULES[name]
        return _prime_import(name, globals, locals, fromlist, level)

    runtime_globals["__builtins__"]["__import__"] = _prime_safe_import
    runtime_globals.update(_PRIME_MODULES)
    runtime_globals["_prime_await"] = _prime_await
    runtime_globals["_prime_display"] = _prime_display
    try:
        _RUNTIME_NAMES.update(_PRIME_MODULES)
        _RUNTIME_NAMES.update(("_prime_await", "_prime_display", "_prime_takeover", "_"))
    except NameError:
        pass

'''
# Arm F: their fixed broker client (arc-agi-3-prime-agent 398d4dd6 ``broker_client.py`` / ``arc3_local_broker.py``) over
# the native step path; the native runtime names are taken out of the kernel.
CHILD_F = '''
    _PRIME_ACTION = runtime_globals.pop("action")
    _PRIME_HIDDEN = ("current_frame", "latest_frame", "history", "transitions", "last_transition", "previous_frame",
                     "last_action_frame", "last_action", "valid_actions", "last_action_result")
    _prime_refresh = _refresh_state

    def _refresh_state(state_payload):
        _prime_refresh(state_payload)
        for _hidden in _PRIME_HIDDEN:
            runtime_globals.pop(_hidden, None)

    class _PrimeBroker(object):
        """Client for exactly one fixed independent ARC broker endpoint."""

        def observe(self):
            return _prime_call("broker.read", which="observe")

        def status(self):
            return _prime_call("broker.read", which="status")

        def act(self, actions):
            begun = _prime_call("broker.begin", actions=actions)
            if begun.get("observation") is not None:
                return begun["observation"]
            for item in actions:
                name = item.get("name") if isinstance(item, dict) else None
                if name not in ("RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"):
                    raise RuntimeError("ProtocolError: bad action %r" % (name,))
                request = {"action": name}
                if name == "ACTION6":
                    data = item.get("data") if isinstance(item.get("data"), dict) else {}
                    if not all(isinstance(data.get(k), int) and 0 <= data[k] <= 63 for k in ("x", "y")):
                        raise RuntimeError("ProtocolError: ACTION6 needs integer x,y in 0..63")
                    request["row"], request["col"] = data["y"], data["x"]
                result = _PRIME_ACTION([request])
                if not result.get("executed"):
                    raise RuntimeError("ProtocolError: " + str(
                        result.get("error") or result.get("stop_detail") or result.get("stop_reason")
                        or "action refused"))
                if (result.get("game_over") or result.get("done") or result.get("run_complete")
                        or result.get("level_completed")):
                    break
            return _prime_call("broker.end")

    _PRIME_MODULES["fixed_broker_client"] = _PrimeBroker()
'''


def with_prime(bootstrap, block):
    """``bootstrap`` with the kernel block before the request loop."""
    if "_PRIME_MODULES" in bootstrap:
        raise ValueError("prime: the kernel block is already installed")
    if bootstrap.count(ANCHOR) != 1:
        raise ValueError("prime: pinned Duck sandbox loop changed")
    child = CHILD_BLOCK.replace("__PRIME_F__", CHILD_F if block["arm"] == "F" else "")
    return bootstrap.replace(ANCHOR, child.strip("\n") + "\n\n" + ANCHOR, 1)


def rewrite_code(code, prefix=b""):
    """``(code, note)``: a cell as their kernel runs it (``prefix``: a statement put in front on the first line). Every ``await X`` becomes ``_prime_await(X)`` (the kernel has
    no event loop; harness calls return at once) and a trailing expression is displayed. A shell cell or a magic
    becomes an error; a cell that does not parse is left to the tool's own syntax report."""
    try:
        tree = compile(code, "<python_tool>", "exec", flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    except (SyntaxError, ValueError):
        first = next((line.strip() for line in code.split("\n") if line.strip()), "")
        return (MAGIC_CODE, "magic") if first.startswith(("%", "!")) else (code, None)
    raw = code.encode()
    starts, total = [], 0
    for line in raw.split(b"\n"):
        starts.append(total)
        total += len(line) + 1

    def at(line, column):
        return starts[line - 1] + column

    edits, awaits = [], 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Await):
            awaits += 1
            edits.append((at(node.lineno, node.col_offset), 5, b"_prime_await("))
            edits.append((at(node.end_lineno, node.end_col_offset), 0, b")"))
    last = tree.body[-1] if tree.body else None
    display = isinstance(last, ast.Expr) and not code.rstrip().endswith(";")
    if display:
        edits.append((at(last.lineno, last.col_offset), 0, b"_prime_display(("))  # a bare tuple is one value
        edits.append((at(last.end_lineno, last.end_col_offset), 0, b"))"))
    if prefix:
        edits.append((0, 0, prefix))
    if not edits:
        return code, None
    # Right to left; at one offset the replacement goes in before the insertion that must end up in front of it.
    for offset, length, text in sorted(edits, key=lambda e: (-e[0], -e[1])):
        raw = raw[:offset] + text + raw[offset + length:]
    out = raw.decode()
    try:
        compile(out, "<python_tool>", "exec")
    except (SyntaxError, ValueError):
        return code, "rewrite_failed"
    return out, "await" if awaits else "display"


# --- the host side ----------------------------------------------------------------------------------------------------

class PrimeSession:
    def __init__(self, owner, block):
        self.owner, self.cfg = owner, block
        self.texts = types.SimpleNamespace(**{name: getattr(texts, name) for name in dir(texts) if name.isupper()})
        for name, value in (block.get("text_overrides") or {}).items():
            setattr(self.texts, name, value)
        self.store = HarnessStore()
        owner.sandbox.bootstrap = with_prime(owner.sandbox.bootstrap, block)
        self.inner_handler = owner.sandbox.animation_handler
        owner.sandbox.animation_handler = self.handle
        self.replies = 0               # gameplay replies since the last review
        self.review_tokens_at = None   # generated tokens at the last review
        self.pending_review = None     # "compact": a review owed after a compaction
        self.levels_refined = 0        # refine_at level_clear: levels cleared at the last automatic review
        self.requested_refine = None   # {"instructions"} from refine.run()
        self.requested_compact = None  # {"instructions"} from compact.run()
        self.in_flight = False
        self.previous_summary = None
        self.last_count = None
        self.inside = False            # compaction "native": the study's own trim is running under ours
        self.phase = 0                 # F: the client's bootstrap phase
        self.act_start = None          # F: steps at the running act()
        self.first_message = True
        self.system_head = self.system_tail = None
        self.active = not block["takeover"]   # False: a recorded script is replaying under the native agent
        self.native_off = None                # takeover: the native system text without the playbook paragraphs
        self.retired = False                  # takeover: every cell from here on retires the kernel's playbook()
        self.took_over = None
        self.stats = {
            "replies": 0, "continuations": 0, "rewrites": {}, "kernel_calls": {},
            "reviews": 0, "reviews_approved": 0, "reviews_failed": 0, "plans": 0, "plans_failed": 0,
            "refinements": 0, "self_refinements": 0, "edits_applied": 0, "edits_failed": 0, "notices": 0,
            "edit_marks": [],          # [generated tokens, edits applied] per refinement
            "compactions": 0, "compactions_requested": 0, "digests": 0, "summaries": [],
            "tail_truncations": 0,
        }
        if block["arm"] == "F":
            self.stats.update(acts=0, act_sizes=[], act_actions=0, acting_replies=0, plan_replies=0, resets=0)

    def tokens(self):
        return int(self.owner.game.clock.completion_tokens())

    def event(self, kind, **data):
        self.owner.workspace.append(kind, step=self.owner.steps, **data)

    # --- prompt ---------------------------------------------------------------------------------------------------

    def skills_section(self):
        lines = [self.texts.SKILLS_HEAD.strip("\n"), "", "<available_skills>"]
        bodies = []
        for name, description, body in (("refine", self.texts.SKILL_REFINE_DESCRIPTION, self.texts.SKILL_REFINE_MD),
                                        ("compact", self.texts.SKILL_COMPACT_DESCRIPTION, self.texts.SKILL_COMPACT_MD)):
            lines += ["  <skill>", f"    <name>{name}</name>", "    <type>python</type>",
                      f"    <python_import>{name}</python_import>", f"    <description>{description}</description>",
                      f"    <location>skills/{name}/SKILL.md (printed below)</location>", "  </skill>"]
            bodies += ["", f"## skills/{name}/SKILL.md", "", body.strip()]
        return "\n".join([*lines, "</available_skills>", *bodies])

    def system_prompt(self, native):
        """Their prompt in their order: head, kernel, refinement, harness state, project context, skills. The project
        context is their ``AGENTS.md`` (F) or the native system prompt (N)."""
        arm = self.cfg["arm"]
        environment = (ENVIRONMENT + (F_WORKSPACE if arm == "F" else "")
                       + ("" if self.cfg["state_in_system"] else DELIVERY if self.cfg["notices"] else DELIVERY_SILENT))
        installed = "`refine`, `compact`" + (", `fixed_broker_client`" if arm == "F" else "")
        self.system_head = "\n\n".join([
            self.texts.HEAD, environment, f"Installed Python skill modules (pre-imported): {installed}.",
            self.texts.IPYTHON_CONTROL_PROMPT, self.texts.REFINE_PARAGRAPH])
        context = self.texts.ARC_AGENTS_MD if arm == "F" else native
        self.native_edits_applied = []
        for edit in self.cfg.get("native_edits") or []:
            hit = context.count(edit["old"]) == 1
            if hit:
                context = context.replace(edit["old"], edit["new"])
            self.native_edits_applied.append(hit)
        extra = self.candidate_section()
        self.system_tail = ("# Project Context\n\nProject-specific instructions and guidelines:\n\n## AGENTS.md\n\n"
                            + context.strip() + ("\n\n" + extra if extra else "") + "\n\n" + self.skills_section())
        return self.render_system()

    def candidate_section(self):
        """The candidate's added text and inherited entries (RSI loop); empty for the Prime arm's own runs."""
        parts = []
        if self.cfg.get("extra_system"):
            parts.append(self.cfg["extra_system"].strip())
        seeds = self.cfg.get("seed_entries") or []
        if seeds:
            parts.append("## Inherited harness entries\n\nWritten before this game started; they are not in "
                         "`rlm.harness` and may be wrong for this game.\n\n" + "\n\n".join(
                             f"### [{s['kind']}] {s['title']}\n{s['content'].strip()}" for s in seeds))
        return "\n\n".join(parts)

    def render_system(self):
        state = self.store.prompt_block() if self.cfg["state_in_system"] else self.texts.HARNESS_STATE_HEAD
        return "\n\n".join([self.system_head, state, self.system_tail])

    def bind(self, analyzer):
        """On the finished native analyzer: the system prompt, and arm F's own turn, tool and tool output."""
        if not self.active and getattr(self.owner, "bench_script", None) is None:
            self.active = True  # not a scripted point: nothing to take over from
        if not self.active:
            self.native_off = self.without_playbook(analyzer._system_prompt)
            self.event("prime_session", arm=self.cfg["arm"], takeover="pending",
                       settings={k: v for k, v in self.cfg.items()})
            return
        analyzer._system_prompt = self.system_prompt(analyzer._system_prompt)
        self.event("prime_session", arm=self.cfg["arm"], system_chars=len(analyzer._system_prompt),
                   settings={k: v for k, v in self.cfg.items()})
        if self.cfg["arm"] != "F":
            return
        analyzer._build_user_message = self.f_user_message
        analyzer._tools = lambda state_path: copy.deepcopy(F_TOOLS)
        analyzer._tool_output_chars = int(self.cfg["output_bytes"])
        analyzer._trim_tool_text = self.tail_text
        dispatch = analyzer._dispatch_tool

        def dispatch_text(state_path, name, arguments):
            result = dispatch(state_path, name, arguments)
            return dataclasses.replace(result, content=self.tool_text(result.content))

        analyzer._dispatch_tool = dispatch_text

    # --- arm F: one long turn, plain-text tool output ---------------------------------------------------------------

    def f_user_message(self, user_prompt, current_frame):
        if self.first_message:
            self.first_message = False
            return {"role": "user", "content": self.texts.ARC_GAME_PROMPT}
        return {"role": "user", "content": F_SENTINEL}

    def tail_text(self, text):
        """Their tool output cut: the last ``output_lines`` lines and ``output_bytes`` bytes."""
        lines = text.split("\n")
        cut = len(lines) > self.cfg["output_lines"]
        if cut:
            text = "\n".join(lines[-self.cfg["output_lines"]:])
        data = text.encode()
        if len(data) > self.cfg["output_bytes"]:
            text, cut = data[-self.cfg["output_bytes"]:].decode(errors="ignore"), True
        return text, cut

    def tool_text(self, content):
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            return content
        if not isinstance(payload, dict):
            return content
        parts = []
        if payload.get("truncated"):
            parts.append(f"[output truncated: the last {self.cfg['output_lines']} lines / "
                         f"{self.cfg['output_bytes'] // 1024} KB are shown]")
        for key in ("stdout", "error", "note", "critical"):
            value = payload.get(key)
            if value:
                parts.append(str(value).rstrip("\n"))
        return "\n".join(parts) or "(no output)"

    def tail_rewrites(self, messages):
        """The not-yet-sent last message: F's turn sentinel is dropped (the session is one turn) and the public
        no-tool follow-up becomes their autonomous continuation prompt."""
        if self.cfg["arm"] != "F" or not messages:
            return messages
        last = messages[-1]
        content = last.get("content")
        if role(last) != "user" or not isinstance(content, str):
            return messages
        if content == F_SENTINEL:
            return list(messages[:-1])
        if content.startswith(PUBLIC_FOLLOWUPS):
            self.stats["continuations"] += 1
            return [*messages[:-1], {"role": "user", "content": self.texts.AUTONOMOUS_CONTINUATION_PROMPT}]
        return messages

    # --- kernel calls ------------------------------------------------------------------------------------------------

    def handle(self, request):
        """The sandbox's ``animation`` channel: a ``prime`` request is ours; never raises (the child would only
        hear "not available")."""
        if "prime" not in request:
            if self.inner_handler is None:
                raise RuntimeError("no animation handler")
            return self.inner_handler(request)
        op, args = str(request.get("prime")), dict(request.get("args") or {})
        calls = self.stats["kernel_calls"]
        calls[op] = calls.get(op, 0) + 1
        try:
            return {"ok": True, "value": self.dispatch(op, args)}
        except Exception as exc:  # noqa: BLE001 - the cell gets the error text
            plain = type(exc) in (RuntimeError, ValueError)
            return {"ok": False, "error": (str(exc) if plain else f"{type(exc).__name__}: {exc}")[:600]}

    def dispatch(self, op, args):
        store = self.store
        if op in ("harness.create", "harness.update", "harness.upsert"):
            fields = {k: args.get(k) for k in ("path", "reference", "arguments", "metadata") if args.get(k) is not None}
            if op == "harness.update":
                entry = store.update(args["kind"], args["id"], args["title"], args["content"], **fields)
            elif op == "harness.create":
                entry = store.create(args["kind"], args["title"], args["content"], id=args.get("id"), **fields)
            else:
                entry = store._upsert(args["kind"], args["title"], args["content"],
                                      id=store._bare(args.get("id")), **fields)
            self.event("prime_harness_write", op=op, entry=f"{entry['kind']}:{entry['id']}", version=entry["version"],
                       chars=len(entry["content"]))
            return entry
        if op == "harness.delete":
            found = store.delete(args["kind"], args["id"])
            self.event("prime_harness_write", op=op, entry=f"{args['kind']}:{args['id']}", found=found)
            return found
        if op == "harness.get":
            return store.get(args["kind"], args["id"])
        if op == "harness.list":
            return store.list(args.get("kind"))
        if op == "harness.overview":
            return store.overview(int(args.get("max_entries_per_kind") or 20))
        if op == "harness.record_refinement":
            return store.record_refinement(args.get("trigger", ""), args.get("changes") or [],
                                           evidence=args.get("evidence") or "", outcome=args.get("outcome") or "",
                                           id=args.get("id"))
        if op == "refine.run":
            self.requested_refine = {"instructions": args.get("instructions")}
            return {"scheduled": True}
        if op == "refine.status":
            return {"pending": self.requested_refine is not None, "in_flight": self.in_flight}
        if op == "compact.run":
            if self.cfg["compaction"] != "prime":
                return {"scheduled": False, "reason": "the host manages the context in this session"}
            self.requested_compact = {"instructions": args.get("instructions")}
            return {"scheduled": True}
        if op == "compact.status":
            tokens, window = self.last_count, int(self.owner.window)
            return {"tokens": tokens, "context_window": window,
                    "percent": None if tokens is None else round(100.0 * tokens / window, 1),
                    "scheduled": self.requested_compact is not None}
        if op.startswith("broker.") and self.cfg["arm"] == "F":
            return self.broker(op, args)
        raise ValueError(f"unknown operation {op!r}")

    # --- arm F: the broker ---------------------------------------------------------------------------------------------

    def observation(self):
        owner = self.owner
        obs, limit = owner.obs, int(owner.config.max_actions)
        terminal = "WIN" if obs.state == "WIN" else "ACTION_CAP" if owner.steps >= limit else "ACTIVE"
        return {"game_id": owner.game_id, "state": obs.state, "levels_completed": int(obs.levels),
                "available_actions": [0 if name == "RESET" else int(name[6:]) for name in obs.available],
                "frame": [frame.tolist() for frame in obs.frames], "actions_taken": int(owner.steps),
                "actions_remaining": max(0, limit - int(owner.steps)), "terminal": terminal}

    def broker(self, op, args):
        owner = self.owner
        if op == "broker.read":
            if self.phase == 0 and args.get("which") != "observe":
                raise RuntimeError("first game operation must be observe")
            if self.phase in (1, 2):
                raise RuntimeError("immediate single genuine non-RESET action required before another game operation")
            value = self.observation()
            self.phase = 1 if self.phase == 0 else self.phase
            return value
        if op == "broker.begin":
            actions = args.get("actions")
            if self.phase == 0:
                raise RuntimeError("first game operation must be observe")
            single = (isinstance(actions, list) and len(actions) == 1 and isinstance(actions[0], dict)
                      and set(actions[0]) == {"name", "data"} and isinstance(actions[0].get("name"), str)
                      and actions[0]["name"] != "RESET" and isinstance(actions[0].get("data"), dict))
            if self.phase in (1, 2) and not single:
                raise RuntimeError("immediate single genuine non-RESET action required before another game operation")
            if not isinstance(actions, list) or not 1 <= len(actions) <= self.cfg["batch"]:
                raise RuntimeError(f"ProtocolError: act takes 1..{self.cfg['batch']} actions")
            value = self.observation()
            if value["terminal"] != "ACTIVE":
                self.advance(value)
                return {"observation": value}
            self.act_start = (int(owner.steps), len(actions))
            return {"observation": None}
        if op == "broker.end":
            if owner.obs.state == "GAME_OVER" and owner.steps < int(owner.config.max_actions):
                # Their broker: a game-over consumes one counted RESET and the batch stops.
                owner.on_loop(self.game_over_reset)
                self.stats["resets"] += 1
            value = self.observation()
            start, requested = self.act_start or (int(owner.steps), 0)
            self.act_start = None
            self.stats["acts"] += 1
            self.stats["act_sizes"].append(requested)
            self.stats["act_actions"] += int(owner.steps) - start
            self.advance(value)
            return value
        raise ValueError(f"unknown operation {op!r}")

    def advance(self, value):
        if self.phase == 1:
            self.phase = 2 if value.get("terminal") == "ACTIVE" else 3
        elif self.phase == 2:
            self.phase = 3

    async def game_over_reset(self):
        """The run loop's reset after a game over (``DuckAgent.run``), taken inside the act that died."""
        from arc_infra.plans import Action

        owner = self.owner
        game_over = owner.last_game_over or {"step": owner.steps, "actions": []}
        await owner.step(Action("RESET"))
        owner.write_duck_state("RESET")
        owner.auto_reset = {**game_over, "reset_step": owner.steps}
        owner.level_start_step = owner.steps
        owner.stuck.reset_level()
        owner.workspace.append("auto_reset", step=owner.steps, game_over_step=game_over.get("step"),
                               actions=game_over.get("actions"), source="prime_broker")

    # --- per reply / per cell -----------------------------------------------------------------------------------------

    def note_reply(self, reply):
        """One gameplay reply (event loop)."""
        if not self.active:
            return
        self.replies += 1
        self.stats["replies"] += 1
        usage = reply.get("usage") or {}
        self.last_count = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
        if self.cfg["arm"] != "F":
            return
        message = reply.get("message") or {}
        calls = message.get("tool_calls") or []
        if any("act(" in str((call.get("function") or {}).get("arguments") or "") for call in calls):
            self.stats["acting_replies"] += 1
            text = " ".join(str(message.get(key) or "") for key in ("content", "reasoning_content", "reasoning"))
            self.stats["plan_replies"] += "[PLAN]" in text

    def rewrite(self, arguments):
        code = (arguments or {}).get("code")
        if not self.active or not isinstance(code, str) or not code.strip():
            return arguments  # a replayed cell runs as recorded
        # Every cell after a takeover retires playbook() again (a restarted kernel comes back with it); a cell that
        # opens with a compound statement cannot take the prefix on its first line and runs without it.
        out, note = rewrite_code(code, b"_prime_takeover();" if self.retired else b"")
        if self.retired and note == "rewrite_failed":
            out, note = rewrite_code(code)
        if note:
            self.stats["rewrites"][note] = self.stats["rewrites"].get(note, 0) + 1
        return arguments if out is code else {**arguments, "code": out}

    # --- the checkpoint before each model step --------------------------------------------------------------------------

    def trim(self, analyzer, messages, tools, extra_safety_tokens=0):
        """``CompactionMixin._trim_messages_for_context`` under Prime (analyzer thread): the rewrites of the unsent
        last message, the refine checkpoint, then the threshold compaction."""
        owner = self.owner
        if not self.active:
            script = getattr(owner, "bench_script", None)
            if script is not None and owner.bench_script_at < len(script["replies"]):
                return self.native(analyzer, messages, tools, extra_safety_tokens)
            return self.takeover(analyzer, messages, tools, extra_safety_tokens)
        if self.cfg["compaction"] == "native":
            return self.trim_native(analyzer, messages, tools, extra_safety_tokens)
        messages = self.tail_rewrites(owner.before_trim(messages))
        if len(messages) < 2:
            return messages
        measure = owner.trigger_tokens
        force = extra_safety_tokens > 0
        budget = max(1, analyzer._context_budget_tokens - max(0, extra_safety_tokens))
        first = getattr(owner, "first_trigger_fraction", None)
        if first is not None and not getattr(owner, "first_trigger_done", False) and not force:
            budget = max(1, int(budget * first))
        if getattr(owner, "lrule_pending", None) is not None:
            fresh = owner.loop_fresh_context(messages, tools, measure)
            if fresh is not None:
                return self.with_digest(fresh)
        count = measure(messages, tools)
        if force:
            count = max(count, int(getattr(owner, "rejected_prompt_tokens", 0) or 0))
            owner.rejected_prompt_tokens = None
        self.last_count = int(count)
        over = force or count > budget
        reason = self.refine_due()
        if reason is not None and over and count + self.cfg["plan_tokens"] + 4096 > owner.window:
            self.pending_review = self.pending_review or (None if reason == "requested" else reason)
        elif reason is not None:
            refined = self.refine(analyzer, messages, reason)
            if refined is not messages:
                messages = refined
                count = measure(messages, tools)
                over = force or count > budget
        if over or self.requested_compact is not None:
            messages = self.compact(messages, tools, measure, budget, count, force)
        return messages

    def native(self, analyzer, messages, tools, extra_safety_tokens):
        """The study's own context policy on ``messages``."""
        if self.inside:
            raise RuntimeError("prime: nested trim")
        self.inside = True
        try:
            return analyzer._trim_messages_for_context(messages, tools=tools, extra_safety_tokens=extra_safety_tokens)
        finally:
            self.inside = False

    def trim_native(self, analyzer, messages, tools, extra_safety_tokens):
        owner = self.owner
        reason = self.refine_due()
        if reason is not None and len(messages) > 1:
            messages = self.refine(analyzer, owner.before_trim(messages), reason)
        result = self.native(analyzer, messages, tools, extra_safety_tokens)
        if result is not messages and len(result) < len(messages):
            self.pending_review = "compact"
            result = self.with_digest(result)
        return result

    # --- takeover at a stuck point ----------------------------------------------------------------------------------------

    def without_playbook(self, native):
        """The native system text as arm N has it from the start (``hx_harness.memo`` off)."""
        from .. import hx_harness as hx_module

        hx = getattr(self.owner, "hx", None)
        if hx is None or not hx["memo"]:
            return native
        if hx["memo_style"] != 2:
            raise ValueError("prime takeover knows the v8.2 playbook (hx_harness memo_style 2)")
        for text in (hx_module.KEEP_RULES_SYSTEM, hx_module.KEEP_SYSTEM, hx_module.PLAYBOOK_OPEN_SYSTEM):
            for variant in (text, text.rstrip("\n")):
                if native.count(variant) == 1:
                    native = native.replace(variant, "", 1)
                    break
            else:
                if text is hx_module.PLAYBOOK_OPEN_SYSTEM:
                    raise ValueError("prime takeover: pinned playbook paragraph of the system prompt changed")
        return native

    def seed(self, playbook):
        """The playbook as memory entries (path ``playbook``): one per paragraph, or per line of a single block."""
        blocks = [b.strip() for b in re.split(r"\n\s*\n", playbook) if b.strip()]
        if len(blocks) == 1:
            blocks = [line.strip() for line in blocks[0].split("\n") if line.strip()]
        if len(blocks) > SEED_ENTRIES:
            blocks = [*blocks[:SEED_ENTRIES - 1], "\n".join(blocks[SEED_ENTRIES - 1:])]
        for index, block in enumerate(blocks, 1):
            title = compact_text(block.split("\n")[0].lstrip("-*# "), 60) or f"playbook {index}"
            self.store._upsert("memory", title, block, id=f"playbook_{index:02d}", path="playbook", source="seed")
        return len(blocks)

    def takeover(self, analyzer, messages, tools, extra_safety_tokens):
        """The first live request of a scripted point (analyzer thread): the replayed conversation is compacted their
        way under the new system prompt, the playbook becomes memory entries and the kernel's ``playbook()`` retires."""
        owner = self.owner
        if getattr(owner, "bench_script_end", None) is None:
            owner.on_loop(owner.bench_script_finish)  # the replay check, before a summary is paid for
        if getattr(owner, "bench_outcome", None) is not None:
            return self.native(analyzer, messages, tools, extra_safety_tokens)  # the gate ends the attempt
        messages = owner.before_trim(messages)
        self.active = True
        playbook = str((owner.sandbox.hx_state or {}).get("playbook") or "")
        seeded = self.seed(playbook)
        system = {**messages[0], "content": self.system_prompt(self.native_off)}
        analyzer._system_prompt = system["content"]
        if getattr(owner, "hx", None) is not None and owner.hx["memo"]:
            owner.hx = {**owner.hx, "memo": False}  # no playbook lines in the turns from here on
            self.retired = True
        measure = owner.trigger_tokens
        budget = max(1, analyzer._context_budget_tokens - max(0, extra_safety_tokens))
        count = measure(messages, tools)
        result = self.compact(messages, tools, measure, budget, count, True, system=system)
        if result is messages:  # nothing to summarise: the system prompt and the harness block only
            result = self.with_digest([system, *messages[1:]])
        note = TAKEOVER_NOTE + (TAKEOVER_PLAYBOOK.format(count=seeded, text=playbook.strip()) if seeded else "")
        result = [result[0], {**result[1], "content": result[1]["content"].rstrip("\n") + note}, *result[2:]]
        self.replies, self.review_tokens_at = 0, None
        self.took_over = {"completion_tokens": self.tokens(), "steps": int(owner.steps), "levels": int(owner.obs.levels),
                          "playbook_chars": len(playbook), "seeded": seeded, "tokens_before": int(count),
                          "tokens_after": int(measure(result, tools)), "messages_before": len(messages),
                          "messages_after": len(result)}
        self.event("prime_takeover", **self.took_over)
        return result

    def with_digest(self, messages):
        """After a native context cut (a drain, the loop rule's fresh context): the harness block in front."""
        self.previous_summary = None
        if self.cfg["state_in_system"] or len(messages) < 2:
            return messages
        self.stats["digests"] += 1
        return [messages[0], {"role": "user", "content": self.digest().rstrip("\n")}, *messages[1:]]

    def digest(self):
        entries = self.store.entries_block(full=bool(self.cfg["digest_full"]))
        if self.cfg["digest_full"]:  # their head line calls the entries compact summaries; here they are whole
            entries = "The entries below are given in full.\n" + entries
        return HARNESS_DIGEST_PREFIX + entries + HARNESS_DIGEST_SUFFIX + "\n\n"

    # --- refinement ---------------------------------------------------------------------------------------------------

    def refine_due(self):
        if self.requested_refine is not None:
            return "requested"
        if not self.cfg["refine"]:
            return None
        if self.cfg["refine_at"] == "level_clear":
            return "level_clear" if self.levels_now() > self.levels_refined else None
        if (self.review_tokens_at is not None
                and self.tokens() - self.review_tokens_at < self.cfg["refine_cooldown_tokens"]):
            return None
        if self.pending_review:
            return self.pending_review
        return "turn_interval" if self.replies >= self.cfg["refine_turns"] else None

    def levels_now(self):
        obs = getattr(self.owner, "obs", None)
        return int(getattr(obs, "levels", 0) or 0)

    def prefill_ok(self):
        from ..compaction import PREFILL_TEMPLATE_MODELS

        config = self.owner.config
        return config.model in PREFILL_TEMPLATE_MODELS and getattr(config, "serving_backend", "vllm") == "vllm"

    def aside(self, messages, prompt, *, cap, floor, purpose, prefill):
        """One request on the whole context plus ``prompt`` (it resumes from the cached context), tools shown but not
        callable; without reasoning where the template can be prefilled."""
        owner = self.owner
        kwargs = dict(cap=cap, floor=floor, purpose="summary", record_purpose=purpose)
        try:
            text, _, _, _ = owner.on_loop(owner.aside_request, messages, prompt, gameplay_prefix=True,
                                          prefill=prefill if self.prefill_ok() else None, **kwargs)
        except httpx.HTTPStatusError as exc:
            self.event("prime_prefix_refused", purpose=purpose, status=exc.response.status_code)
            text, _, _, _ = owner.on_loop(owner.aside_request, messages, prompt, **kwargs)
        return text

    @staticmethod
    def parsed(parse, text, prefill):
        """``parse(text)``; a reply that restarted the object after the prefill is read without it."""
        try:
            return parse(text)
        except ValueError:
            if not text.startswith(prefill):
                raise
            return parse(text[len(prefill):])

    def review_prompt(self, reason, turns):
        return "\n\n".join([
            self.texts.AUTO_REFINE_REVIEW_SYSTEM_PROMPT,
            f"<trigger>\n{reason}; {turns} assistant turns since last auto-refine review\n</trigger>",
            f"<current_harness_state>\n{self.store.overview_for_plan()}\n</current_harness_state>",
            f"<refinement_history>\n{self.store.history_for_plan()}\n</refinement_history>",
            "<conversation>\n(the messages above)\n</conversation>",
            self.texts.REFINE_REVIEW_TAIL])

    def plan_prompt(self, instructions):
        return "\n\n".join(filter(None, [
            self.texts.REFINEMENT_SYSTEM_PROMPT,
            f"<current_harness_state>\n{self.store.overview_for_plan()}\n</current_harness_state>",
            f"<refinement_history>\n{self.store.history_for_plan()}\n</refinement_history>",
            "<conversation>\n(the messages above)\n</conversation>",
            f"<scope_policy>\n{self.texts.REFINE_SCOPE_LOCAL}\n</scope_policy>",
            f"<user_refine_instructions>\n{instructions}\n</user_refine_instructions>" if instructions else "",
            self.texts.REFINE_PLAN_TAIL]))

    def refine(self, analyzer, messages, reason):
        """The review gate (automatic triggers), the planner, the edits; ``messages`` with the notice (or the rebuilt
        system prompt), or ``messages`` itself when nothing was applied."""
        cfg, store = self.cfg, self.store
        request, self.requested_refine = self.requested_refine, None
        source = "auto" if request is None else "self"
        turns = self.replies
        self.in_flight = True
        try:
            if request is None:
                self.replies, self.pending_review = 0, None
                self.levels_refined = self.levels_now()
                try:
                    prefill = '{"shouldRefine":'
                    review = self.parsed(parse_review, self.aside(
                        messages, self.review_prompt(reason, turns), cap=cfg["review_tokens"], floor=256,
                        purpose="refine_review", prefill=prefill), prefill)
                except DEADLINE:
                    raise
                except Exception as exc:  # noqa: BLE001 - a failed review never ends the game
                    self.review_tokens_at = self.tokens()
                    self.stats["reviews_failed"] += 1
                    self.event("prime_refine_failed", stage="review", reason=reason,
                               error=f"{type(exc).__name__}: {exc}"[:300])
                    return messages
                self.review_tokens_at = self.tokens()
                self.stats["reviews"] += 1
                self.stats["reviews_approved"] += review["shouldRefine"]
                self.event("prime_refine_review", reason=reason, turns=turns, approved=review["shouldRefine"],
                           rationale=review["rationale"][:300], completion_tokens=self.review_tokens_at)
                if not review["shouldRefine"]:
                    return messages
                detail = f"\nReviewer instructions: {review['instructions']}" if review["instructions"] else ""
                instructions = self.texts.AUTO_REFINE_INSTRUCTIONS.format(reason=reason, rationale=review["rationale"],
                                                                     detail=detail)
            else:
                instructions = request.get("instructions")
            self.stats["plans"] += 1
            try:
                prefill = '{"summary":'
                proposal = self.parsed(parse_proposal, self.aside(
                    messages, self.plan_prompt(instructions), cap=cfg["plan_tokens"], floor=1024,
                    purpose="refine_plan", prefill=prefill), prefill)
            except DEADLINE:
                raise
            except Exception as exc:  # noqa: BLE001
                self.stats["plans_failed"] += 1
                self.event("prime_refine_failed", stage="plan", reason=reason,
                           error=f"{type(exc).__name__}: {exc}"[:300])
                return messages
            result = store.apply(proposal)
        finally:
            self.in_flight = False
        applied = [edit for edit in result["appliedEdits"] if edit["applied"]]
        failed = [edit for edit in result["appliedEdits"] if not edit["applied"]]
        self.stats["refinements"] += 1
        self.stats["self_refinements"] += source == "self"
        self.stats["edits_applied"] += len(applied)
        self.stats["edits_failed"] += len(failed)
        self.stats["edit_marks"].append([self.tokens(), len(applied)])
        self.event("prime_refine", id=result["id"], source=source, reason=reason, summary=result["summary"][:300],
                   applied=[f"{e['action']} {e['kind']}:{e['id']}" for e in applied],
                   failed=[f"{e['action']} {e['kind']}:{e['id']}: {e.get('error')}" for e in failed],
                   completion_tokens=self.tokens())
        if not applied:
            return messages
        if cfg["state_in_system"]:
            analyzer._system_prompt = self.render_system()
            return [{**messages[0], "content": analyzer._system_prompt}, *messages[1:]]
        if source == "auto" and not cfg["notices"]:
            return messages
        lines = [compact_text(result["summary"], CONTENT_LIMIT)]
        for edit in applied:
            entry = edit.get("after") or edit.get("before") or {}
            lines.append(f"- {edit['action']} {edit['kind']} [{entry.get('scope', 'local')}:{edit['id']}] "
                         f"{entry.get('title', edit['id'])}: {compact_text(entry.get('content', ''), CONTENT_LIMIT)}")
        notice = {"role": "user", "content": f"[{source}-refinement]\n\n" + "\n".join(lines)}
        self.stats["notices"] += 1
        # Before a last message that has not been sent (the turn's observation, a follow-up); after a tool result.
        if role(messages[-1]) == "user":
            return [*messages[:-1], notice, messages[-1]]
        return [*messages, notice]

    # --- compaction ---------------------------------------------------------------------------------------------------

    @staticmethod
    def is_summary(message):
        content = message.get("content")
        return role(message) == "user" and isinstance(content, str) and (
            content.startswith(HARNESS_DIGEST_PREFIX) or content.startswith(texts.COMPACTION_SUMMARY_PREFIX))

    def message_tokens(self, message):
        return int(self.owner._analyzer._estimate_request_input_tokens([message]))

    def cut_index(self, messages, start):
        """Pi's ``findCutPoint`` on ``messages[start:]``: walk back until ``keep_recent_tokens`` are kept, then the
        first user or assistant message at or after that point (never a tool result). Ours: with none after it (the
        newest message is a tool result larger than the keep), the last one before it."""
        points = [i for i in range(start, len(messages)) if role(messages[i]) in ("user", "assistant")]
        if not points:
            return None
        total, cut = 0, points[0]
        for index in range(len(messages) - 1, start - 1, -1):
            total += self.message_tokens(messages[index])
            if total >= self.cfg["keep_recent_tokens"]:
                later = [p for p in points if p >= index]
                cut = later[0] if later else points[-1]
                break
        return cut

    def summary_request(self, source, prompt, cap, opener):
        """``(text, usage, seconds, attempts, fallback)``; as the native summary, a game is never ended by one."""
        from ..compaction import mechanical_summary

        owner = self.owner
        prefill = opener if self.cfg["summary_thinking"] == "prefilled-empty" and self.prefill_ok() else None
        kwargs = dict(cap=cap, floor=2048, purpose="summary", record_purpose="compaction")
        text, data, seconds, attempts, prefix = "", {}, 0.0, 1, True
        try:
            try:
                text, data, seconds, _ = owner.on_loop(owner.aside_request, source, prompt, gameplay_prefix=True,
                                                       prefill=prefill, **kwargs)
            except httpx.HTTPStatusError as exc:
                self.event("compaction_prefix_refused", status=exc.response.status_code)
                prefill, prefix = None, False
                text, data, seconds, _ = owner.on_loop(owner.aside_request, source, prompt, **kwargs)
            if not text:
                self.event("compaction_empty", attempt=attempts, usage=data.get("usage", {}))
                text, data, more, _ = owner.on_loop(owner.aside_request, source, prompt + SUMMARY_RETRY.format(opener),
                                                    gameplay_prefix=prefix, prefill=prefill, **kwargs)
                attempts, seconds = 2, seconds + more
        except ContextLengthRejected as exc:
            self.event("compaction_rejected", attempt=attempts, detail=str(exc)[:500])
        fallback = not text
        if fallback:
            obs = owner.obs
            text = mechanical_summary(steps=owner.steps, levels=obs.levels if obs else 0,
                                      total_levels=getattr(obs, "total_levels", None) if obs else None)
        return text, data.get("usage", {}), seconds, attempts, fallback

    def compact(self, messages, tools, measure, budget, count, force, system=None):
        from ..context_drain import latest_observation, middle_truncate
        from ..duck import SUMMARY_SCOPE_NOTE

        owner, cfg = self.owner, self.cfg
        request, self.requested_compact = self.requested_compact, None
        over = force or count > budget
        start = 2 if len(messages) > 2 and self.is_summary(messages[1]) else 1
        previous = self.previous_summary if start == 2 else None
        cut = self.cut_index(messages, start)
        if cut is None or cut <= start:
            points = [i for i in range(start + 1, len(messages)) if role(messages[i]) in ("user", "assistant")]
            if not over or not points:
                return messages  # nothing to summarise yet
            cut = points[-1]
        turn = max((i for i in range(start, cut) if role(messages[i]) == "user"), default=None)
        # Pi summarises a split turn as "No prior history." plus the turn's prefix when nothing precedes the turn.
        prefix_only = role(messages[cut]) == "assistant" and turn == start and previous is None
        if prefix_only:
            body, cap, opener, kind = self.texts.TURN_PREFIX_SUMMARIZATION_PROMPT, cfg["reserve_tokens"] // 2, \
                "## Original Request", "turn_prefix"
        else:
            body = (self.texts.SUMMARIZATION_PROMPT if previous is None
                    else PREVIOUS_POINTER + self.texts.UPDATE_SUMMARIZATION_PROMPT)
            body += "\n\n" + self.texts.KERNEL_PERSIST_SUMMARY_NOTE
            cap, opener, kind = int(0.8 * cfg["reserve_tokens"]), "## Goal", "history" if previous is None else "update"
        prompt = self.texts.SUMMARIZATION_SYSTEM_PROMPT + "\n\n" + body
        if request and request.get("instructions"):
            prompt += f"\n\nAdditional focus: {request['instructions']}"
        # A request that ends inside the cached context recomputes all of it, so the summary reads the whole context
        # and is told which messages stay; without room for it, the cut prefix alone.
        whole = count + cfg["summary_room_tokens"] <= owner.window
        source = messages if whole else list(messages[:cut])
        if whole:
            prompt += SUMMARY_SCOPE_NOTE.format(messages=len(messages) - cut)
        summary, usage, seconds, attempts, fallback = self.summary_request(source, prompt, cap, opener)
        summary = summary.strip()
        if prefix_only and not fallback:
            summary = f"No prior history.\n\n---\n\n**Turn Context (split turn):**\n\n{summary}"
        tail = list(messages[cut:])
        if owner.dfr is not None and owner.dfr["vision"]:
            current = latest_observation(tail, getattr(owner, "is_observation", None))
            if current is not None:
                tail = [*owner.dfr_history(tail[:current]), *tail[current:]]
        head = self.texts.COMPACTION_SUMMARY_PREFIX + summary + self.texts.COMPACTION_SUMMARY_SUFFIX
        if not cfg["state_in_system"]:
            head = self.digest() + head
            self.stats["digests"] += 1
        if cfg["inventory"] and owner.fix("inventory"):
            head = head.rstrip("\n") + "\n\n" + owner.on_loop(owner.inventory_block)
        result = [system or messages[0], {"role": "user", "content": head}, *tail]
        after = measure(result, tools)
        rounds = 0
        while after > budget and rounds < 6:
            # (ours) A tail that alone exceeds the budget: its largest tool result is cut in the middle.
            sizes = [(len(m["content"]), i) for i, m in enumerate(result) if i > 1 and role(m) == "tool"
                     and isinstance(m.get("content"), str)]
            if not sizes or max(sizes)[0] < 2000:
                break
            size, index = max(sizes)
            result[index] = {**result[index], "content": middle_truncate(result[index]["content"], size // 2)[0]}
            self.stats["tail_truncations"] += 1
            after, rounds = measure(result, tools), rounds + 1
        headings = sum(1 for heading in HEADINGS if heading in summary)
        self.previous_summary = summary
        self.pending_review = "compact"
        self.stats["compactions"] += 1
        self.stats["compactions_requested"] += request is not None and not over
        self.stats["summaries"].append({"kind": kind, "headings": headings, "chars": len(summary),
                                        "fallback": fallback, "completion_tokens": usage.get("completion_tokens")})
        owner.context_compactions += 1
        owner.compaction_calls += 1
        owner.game.compactions += 1
        owner.game.compaction_seconds += seconds
        owner.first_trigger_done = True
        getattr(owner.game, "note_compaction", lambda: None)()
        self.event("compaction", estimate=int(count), summary_chars=len(summary), seconds=round(seconds, 3),
                   attempts=attempts, mechanical_fallback=fallback, usage=usage, thinking=cfg["summary_thinking"],
                   prime={"kind": kind, "headings": headings, "whole_context": whole, "cut": cut,
                          "kept_messages": len(tail), "summarised_messages": cut - start, "tokens_after": int(after),
                          "requested": request is not None, "tail_truncations": rounds,
                          "entries": self.store.count()})
        return result

    # --- the attempt's record -----------------------------------------------------------------------------------------

    def summary(self):
        stats = dict(self.stats)
        marks = stats.pop("edit_marks")
        summaries = stats["summaries"]
        full = [s for s in summaries if s["kind"] != "turn_prefix" and not s["fallback"]]
        base = (self.took_over or {}).get("completion_tokens", 0)  # a takeover's 200K count from its first live request
        row = {"arm": self.cfg["arm"], "compaction": self.cfg["compaction"], **stats,
               "edits_applied_by_200k": sum(n for tokens, n in marks if tokens <= 200000 + base),
               "summaries_six_headings": sum(1 for s in full if s["headings"] == len(HEADINGS)),
               "summaries_structured": len(full), "store": self.store.stats()}
        if self.cfg["takeover"]:
            row["takeover"] = self.took_over
        if self.cfg["arm"] == "F":
            sizes = row.pop("act_sizes")
            row["act_size_median"] = statistics.median(sizes) if sizes else None
        return row
