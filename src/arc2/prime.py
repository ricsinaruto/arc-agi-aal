"""ARC-AGI-2 roster trial played by the stock Prime Agent v0.3.3 binary (prime2 arm; 2026: "a full,
faithful Prime Agent based harness for ARC-AGI-2 with Flash").

v4 ("same submit() approach as conv1, closing the task; same prompts and initial inputs as
conv1; arc2-specific guidance and answer workflow"): Prime's loop, ipython tool, persistent kernel, compaction and
auto-refine stay; conv1's (``arc2.conv``) system prompt replaces Prime's (``--system-prompt``; the tool is named
``ipython`` there), the first message is conv1's (the task as text, then the pictures), the kernel holds conv1's
globals (``train``, ``test_inputs``, ``np``, ``submit``, ``program_N``; ``player/startup.py``), and when Prime's agent
stops without a tool call the harness answers as conv1 does: a ```python block defining ``transform`` is run on the
training pairs (the exact differences, the program's number, the ledger of distinct readings), otherwise conv1's
nothing / length-limit message. No ``--autonomous``: these replies take the place of Prime's continuation prompt. A
valid ``submit()`` closes the entry's session and frees its place; the gate's hot set follows conv1's hold rule (a
session keeps its place until it compacts or ends; the next is the open entry with the fewest tokens). Read-out as
conv1: an entry's guesses at a mark are ``loop.guesses`` over its submit and verified programs by then.

v1-v3 (as shipped, autonomous):

An ``arc2`` trial with a ``prime`` block runs here instead of ``arc2.loop``. Per roster entry (task x copy) one stock
Prime Agent rpc session (PrimeIntellect-ai/prime-agent bc6894c8, MIT; ``--mode rpc --autonomous``), as the prime arm's
``arena.prime_binary`` ran it on Nebius (13 Oct): their system prompt (recursion and subagents allowed), IPython
kernel, harness state, compaction and auto-refine as shipped. The ARC-2 side: ``/workspace/task.json`` (training pairs
and test inputs, no outputs), ``/workspace/arc_task.py`` (``load()``, ``submit()`` -> ``answer.json``) and the task
prompt in ``prime/player``. Each session runs as its own user in its own mount, pid and network namespaces
(``prime_sandbox``): no network but the model proxy, the bundle (its task data holds the test outputs), the trial
folder and the server hidden.

Every model request of every session (gameplay, compaction summaries, refinement, rlm children) passes ``Proxy``: the
exact prompt count from the server tokenizer, the study window (an overflow gets the server's context-length error,
which Pi answers with a compaction), base18's sampling where Pi sends none, and ``PriorityGate``: at most ``slots``
requests in flight and the served KV blocks, the next request going to the entry with the fewest completion tokens
plus a delay (base18's roster rule: ``confirm_delay`` once the entry has a valid answer, ``settled_delay`` once the
agent asked to continue with one). Sessions never end by themselves (autonomous limits raised); the roster stops
admitting at its largest mark, the replies in flight finish, and the read-out scores each entry's latest valid
``answer.json`` against the roster's token clock (``loop.token_clock``) at every mark.
"""

import asyncio
import contextlib
import heapq
import itertools
import json
import os
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from . import tasks
from .loop import guesses, token_clock

RUNTIME = Path(os.environ.get("PRIME_RUNTIME", "/opt/prime-runtime"))
KERNEL_PACKAGES = ("ipykernel", "nest-asyncio", "tyro", "numpy", "dill", "requests", "httpx", "pyyaml", "tomli",
                   "python-dotenv", "pandas", "scipy", "beautifulsoup4", "lxml", "pydantic", "prime-agent-runtime")
PIP_WHEEL = "pip-25.2-py3-none-any.whl"
PROXY_PORT = 18080
SUMMARY_SYSTEM = "You are a context summarization assistant."
REFINE_SYSTEMS = ("You are Prime Agent's /refine continual harness subsystem.",
                  "You are Prime Agent's automatic /refine review gate.")
# core/autonomous.ts DEFAULT_AUTONOMOUS_CONTINUATION_PROMPT (v0.3.3)
CONTINUATION = ("No human input is available in autonomous mode. Continue working until the host evaluator, verifier, "
                "or configured autonomous limits stop the run. If you were asking the user a question, make a "
                "reasonable assumption and verify it. If you believe you are blocked, prove it with host-observable "
                "evidence, preserve that evidence, and keep looking for safe progress while budget remains. Do not end "
                "the session yourself; the verifier/evaluator decides completion when configured gates pass.")
KEEP = {"agent_start", "agent_end", "turn_start", "turn_end", "compaction_start", "compaction_end", "refine_complete",
        "refine_failed", "tool_execution_start", "tool_execution_end", "auto_retry_start", "auto_retry_end", "response",
        "error", "extension_error", "message_end"}


def _ipython(text):
    """conv1's wording with Prime's tool name (its interpreter tool is ``ipython``; code fences keep ```python)."""
    return text.replace("`python`", "`ipython`")


CONV_SYSTEM = _ipython(tasks.CONV_SYSTEM)
CONV_NOTHING = _ipython(tasks.CONV_NOTHING)
CONV_CUT = _ipython(tasks.CONV_CUT)
DEFAULTS = {
    "sandbox": True,              # own uid + mount/pid/net namespaces (prime_sandbox); off only in tests / Modal CPU
    "thinking": "xhigh",
    "top_p": 0.95, "top_k": 20,   # base18's (arc2.tasks.chat_request); temperature is the study's
    "min_reply_tokens": 1024,     # below this room in the window the request gets the context-length error
    "watchdog_seconds": 900,      # nothing in the proxy this long with a tool running: rpc abort
    "idle_seconds": 60,           # (autonomous only) a session idle after agent_end this long: the continuation prompt
    "conv": True,                 # v4: conv1's prompts, kernel globals, answer workflow and closing submit (no --autonomous)
    "images": True, "image_side": 224,  # conv1's arm (images True, image_side 224, no change panel)
    "call_seconds": 30,           # conv1's per-call limit in the kernel
    "reply_delay_seconds": 1.0,   # agent stopped this long (no compaction, no request) -> the harness's reply
    "restarts": 10,               # a session that exits is resumed with its conversation (--continue) at most this often
    "session_memory_gib": 4,      # cgroup ceiling per session (node + kernel)
    "start_stagger_seconds": 0.25,
    "answer_poll_seconds": 2.0,
    "confirm_delay": 20000, "settled_delay": 100000,  # base18's arc2 settings
    "stop_grace_seconds": 900,    # after the last mark / a stop request: replies in flight finish within this
    "hot_limit": 49,              # PriorityGate hot set (0: plain fewest-tokens rotation, run 1); conv1: 49 held tasks
    "quantum_tokens": 0,          # completion tokens an entry keeps its hot place for (v2-v3 40000); 0: conv1's hold
    "release_on_compaction": True,  # conv1's hold: a session that compacts gives its place up
    "keep_cut_replies": True,     # v5: a reply cut at the cap stays in the conversation (CutReplies)
    "hot_idle_seconds": 120,
    # Requests between admission and their first streamed token, at most this many at once (run 2, 27 Oct: the hot
    # set's cache hits put ~20 short prefills in one step beside a 3,136-token chunk and the PLE short-conv batched
    # prefill ran the card out of memory, the engine died; run 1's misses were long prefills, 1-2 a step). 0: off.
    "prefill_slots": 4,
    "server_dead_seconds": 180,   # no answer from the model server this long: the trial ends (no engine watchdog here)
}
PROGRESS_SECONDS = 60


def settings(trial, study):
    cfg = {**DEFAULTS, **(trial.get("prime") or {})}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"prime: unknown keys {sorted(unknown)}")
    settings_ = (study.get("arc2") or {}).get("settings") or {}
    for key in ("confirm_delay", "settled_delay"):  # the study's roster delays unless the prime block sets its own
        if key not in (trial.get("prime") or {}) and key in settings_:
            cfg[key] = settings_[key]
    return cfg


def purpose_of(payload):
    messages = payload.get("messages") or []
    system = messages[0].get("content") if messages and messages[0].get("role") in ("system", "developer") else ""
    if isinstance(system, list):
        system = " ".join(p.get("text", "") for p in system if isinstance(p, dict))
    system = str(system or "")
    if system.startswith(SUMMARY_SYSTEM):
        return "summary"
    if system.startswith(REFINE_SYSTEMS):
        return "refine"
    return "gameplay"


def last_user_text(payload):
    for message in reversed(payload.get("messages") or []):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            return str(content or "")
        if message.get("role") == "assistant":
            return ""
    return ""


def usage_of(tail):
    """Usage and finish reason from the end of an SSE stream (or a JSON body)."""
    text = tail.decode("utf-8", "replace")
    usage = finish = None
    items = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")] or [text]
    for item in items:
        if not item or item == "[DONE]":
            continue
        try:
            data = json.loads(item)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        usage = data.get("usage") or usage
        for choice in data.get("choices") or []:
            finish = choice.get("finish_reason") or finish
    return usage, finish


def stream_message(body):
    """The assistant message of a streamed reply: (reasoning text, its delta field, content text, any tool call)."""
    reasoning, field, content, tools = [], None, [], False
    for line in body.decode("utf-8", "replace").splitlines():
        if not line.startswith("data:"):
            continue
        try:
            data = json.loads(line[5:].strip())
        except ValueError:
            continue
        for choice in (data.get("choices") or []) if isinstance(data, dict) else []:
            delta = choice.get("delta") or {}
            for key in ("reasoning_content", "reasoning"):
                if delta.get(key):
                    reasoning.append(delta[key])
                    field = field or key
            if delta.get("content"):
                content.append(delta["content"])
            tools = tools or bool(delta.get("tool_calls"))
    return "".join(reasoning), field, "".join(content), tools


def message_key(message):
    return json.dumps(message, sort_keys=True)[-4000:]


class CutReplies:
    """v5 (27 Oct, v4's cause analysis): a reply cut at the completion cap is all reasoning, no text and no tool call,
    and Pi's openai-completions converter skips an assistant message with neither (`if (!hasContent &&
    !assistantMsg.tool_calls) continue`): the next request lost the whole cut reply (v4: 65 of 65; prompt +42 tokens
    where conv1 keeps +16,435), so the model was told to go on from where it was with no record of it. Each cut
    reply is put back, as conv1 keeps it (content "" + the reasoning in the field the server streamed), at its place in
    every later gameplay request: after the message it followed, checked by position and content, until a compaction
    removes that message."""

    def __init__(self):
        self.kept = []  # {"pos", "anchor", "message"} in conversation order

    def restore(self, messages):
        """Insert the kept cut replies into Pi's ``messages`` (in place); drops those whose anchor is gone."""
        alive, inserted = [], 0
        for item in self.kept:
            pos = item["pos"]
            if pos <= len(messages) and pos >= 1 and message_key(messages[pos - 1]) == item["anchor"]:
                if not (pos < len(messages) and messages[pos] == item["message"]):
                    messages.insert(pos, item["message"])
                    inserted += 1
                alive.append(item)
        self.kept = alive
        return inserted

    def keep(self, messages, body):
        """After a reply to ``messages`` (as sent, cut replies restored) ended at the cap with only reasoning."""
        reasoning, field, content, tools = stream_message(body)
        if tools or content.strip() or not reasoning.strip():
            return False
        self.kept.append({"pos": len(messages), "anchor": message_key(messages[-1]),
                          "message": {"role": "assistant", "content": "", field or "reasoning_content": reasoning}})
        return True


# --- answers --------------------------------------------------------------------------------------------------------

def parse_answer(data, n_tests):
    """``answer.json`` -> per test input a list of at most two valid grids (None where the entry gave none valid)."""
    rows = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) != n_tests:
        return None
    out = []
    for cands in rows:
        if tasks.valid(cands):
            cands = [cands]
        good = [c for c in (cands if isinstance(cands, list) else [])[:2] if tasks.valid(c)]
        out.append(good or None)
    return out if any(out) else None


def right_outputs(answer, task):
    """Test outputs of ``task`` that ``answer`` (parse_answer's form) has right among its two candidates."""
    if not answer:
        return 0
    return sum(bool(cands) and any(c == t["output"] for c in cands) for cands, t in zip(answer, task["test"]))


# --- the gate -------------------------------------------------------------------------------------------------------

class PriorityGate:
    """At most ``slots`` requests in flight and ``capacity`` served KV units reserved.

    Hot set (run 1, 27 Oct: rotating every request over 240 sessions by fewest tokens left each conversation 60-100 s
    at the gate, its cached prefix was evicted, 32% of prompt tokens cached against base18's 94%): at most
    ``hot_limit`` entries are hot. A hot entry's requests go first, in arrival order, so its conversation stays in
    the server's cache across its short tool turns. An entry leaves the hot set after ``quantum`` completion tokens
    since it entered or ``hot_idle`` s without a request; then its next request waits in the cold queue by key
    (fewest tokens + delay) and enters the hot set when there is room. ``hot_limit`` 0: the plain priority queue."""

    def __init__(self, capacity, slots, cost=None, log=None, hot_limit=0, quantum=0, hot_idle=120.0, now=None):
        self.capacity, self.slots, self.cost = int(capacity), int(slots), cost or (lambda tokens: int(tokens))
        self.log, self.now = log, now or time.monotonic
        self.heap, self.seq = [], itertools.count()
        self.hotq = []  # FIFO of hot requests: (seq, future, units, since, entry)
        self.hot = {}  # entry -> {"spent": completion tokens since entering, "last": last request end or start}
        self.hot_limit, self.quantum, self.hot_idle = int(hot_limit), int(quantum), float(hot_idle)
        self.reserved = self.inflight = 0
        self.closed = False
        self.peak_reserved = self.peak_inflight = self.peak_waiting = self.admitted = 0
        self.hot_admitted = self.cold_admitted = self.dropped = self.idle_evictions = self.compaction_releases = 0
        self.wait_seconds = 0.0

    def units(self, tokens):
        return self.cost(int(tokens))

    async def acquire(self, key, tokens, entry=None):
        units = self.units(tokens)
        if units > self.capacity:
            raise ValueError(f"one request needs {units} units; the gate holds {self.capacity}")
        future = asyncio.get_running_loop().create_future()
        since = self.now()
        if entry is not None and entry in self.hot:
            self.hot[entry]["last"] = since
            self.hotq.append((next(self.seq), future, units, since, entry))
        else:
            heapq.heappush(self.heap, (float(key), next(self.seq), future, units, since, entry))
        self.peak_waiting = max(self.peak_waiting, len(self.heap) + len(self.hotq))
        self.pump()
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self.release(units)  # granted as it was cancelled
            raise
        return units

    def charged(self, entry, completion):
        """A reply of ``entry`` ended: its tokens count against its quantum."""
        row = self.hot.get(entry)
        if row is not None:
            row["spent"] += int(completion)
            row["last"] = self.now()
            if self.quantum and row["spent"] >= self.quantum:
                del self.hot[entry]
        self.pump()

    def drop(self, entry):
        """``entry`` leaves the hot set now (its request failed, its session exited or ended)."""
        if self.hot.pop(entry, None) is not None:
            self.dropped += 1
        self.pump()

    def release_hot(self, entry):
        """``entry`` compacted: conv1's hold rule, its place goes to the open entry with the fewest tokens."""
        if self.hot.pop(entry, None) is not None:
            self.compaction_releases += 1
        self.pump()

    def active(self, entry):
        return entry in {row[4] for row in self.hotq} or getattr(entry, "streaming", 0)

    def expire(self):
        now = self.now()
        for entry, row in list(self.hot.items()):
            if not self.active(entry) and now - row["last"] >= self.hot_idle:
                del self.hot[entry]
        # Never idle with work waiting: nothing in flight, no hot request waiting and cold requests waiting -> the
        # longest-quiet hot entry gives its place up.
        if self.inflight == 0 and not self.hotq and self.heap and len(self.hot) >= self.hot_limit:
            quiet = [(row["last"], id(e), e) for e, row in self.hot.items() if not self.active(e)]
            if quiet:
                del self.hot[min(quiet)[2]]
                self.idle_evictions += 1

    def grant(self, future, units, since):
        self.inflight += 1
        self.reserved += units
        self.admitted += 1
        self.wait_seconds += self.now() - since
        self.peak_inflight = max(self.peak_inflight, self.inflight)
        self.peak_reserved = max(self.peak_reserved, self.reserved)
        future.set_result(True)

    def pump(self):
        if self.closed:
            return
        if self.hot_limit:
            self.expire()
        while self.hotq:
            _, future, units, since, entry = self.hotq[0]
            if future.cancelled():
                self.hotq.pop(0)
                continue
            if self.inflight >= self.slots or self.reserved + units > self.capacity:
                return
            self.hotq.pop(0)
            self.hot_admitted += 1
            self.grant(future, units, since)
        while self.heap:
            key, _, future, units, since, entry = self.heap[0]
            if future.cancelled():
                heapq.heappop(self.heap)
                continue
            if self.inflight >= self.slots or self.reserved + units > self.capacity:
                return
            if self.hot_limit and entry is not None and len(self.hot) >= self.hot_limit:
                return
            heapq.heappop(self.heap)
            if self.hot_limit and entry is not None:
                self.hot[entry] = {"spent": 0, "last": self.now()}
            self.cold_admitted += 1
            self.grant(future, units, since)

    def release(self, units):
        self.inflight -= 1
        self.reserved -= int(units)
        self.pump()

    def close(self):
        self.closed = True

    def snapshot(self):
        return {"slots": self.slots, "capacity": self.capacity, "inflight": self.inflight, "reserved": self.reserved,
                "waiting": sum(not row[2].cancelled() for row in self.heap) + len(self.hotq),
                "hot": len(self.hot), "hot_waiting": len(self.hotq), "hot_admitted": self.hot_admitted,
                "cold_admitted": self.cold_admitted, "admitted": self.admitted, "dropped": self.dropped,
                "idle_evictions": self.idle_evictions, "compaction_releases": self.compaction_releases,
                "peak_inflight": self.peak_inflight, "peak_reserved": self.peak_reserved,
                "peak_waiting": self.peak_waiting, "closed": self.closed,
                "mean_wait_seconds": round(self.wait_seconds / max(1, self.admitted), 2)}


# --- runtime --------------------------------------------------------------------------------------------------------

def prepare_runtime(bundle, users, log):
    """Node, the Prime Agent build and the IPython kernel's venv under RUNTIME; one user per entry."""
    source = Path(bundle) / "prime"
    RUNTIME.mkdir(parents=True, exist_ok=True)
    os.chmod(RUNTIME, 0o755)
    started = time.time()
    node = RUNTIME / "node"
    if not (node / "bin/node").exists():
        staging = RUNTIME / "node-staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        with tarfile.open(next((source / "runtime").glob("node-*-linux-x64.tar.xz"))) as tar:
            tar.extractall(staging, filter="tar")
        next(staging.iterdir()).rename(node)
        shutil.rmtree(staging)
    agent = RUNTIME / "prime-agent"
    if not (agent / ".installed").exists():
        shutil.rmtree(agent, ignore_errors=True)
        agent.mkdir()
        with tarfile.open(source / "prime-agent.tar.gz") as tar:
            tar.extractall(agent, filter="tar")
        (agent / ".installed").write_text("ok\n")
    kernel = RUNTIME / "kernel"
    if not (kernel / ".installed").exists():
        shutil.rmtree(kernel, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(kernel)], check=True)
        env = {**os.environ, "PYTHONPATH": str(source / "runtime" / PIP_WHEEL), "PIP_CONFIG_FILE": os.devnull,
               "PIP_NO_INDEX": "1"}
        done = subprocess.run([str(kernel / "bin/python"), "-m", "pip", "install", "--no-index", "--find-links",
                               str(source / "runtime/wheelhouse"), *KERNEL_PACKAGES], env=env, capture_output=True,
                              text=True)
        if done.returncode:
            raise RuntimeError(f"kernel venv install failed: {done.stderr[-2000:]}")
        (kernel / ".installed").write_text("ok\n")
    subprocess.run(["chmod", "-R", "a+rX", str(RUNTIME)], check=True)
    if users:
        from . import prime_sandbox as sandbox

        for user in users:
            sandbox.ensure_user(user)
    probe = subprocess.run([str(node / "bin/node"), "--version"], capture_output=True, text=True, check=True)
    clean = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    check = subprocess.run([str(kernel / "bin/python"), "-c", "import numpy, scipy, ipykernel, rlm; "
                            "print(numpy.__version__)"], capture_output=True, text=True, env=clean)
    if check.returncode:
        raise RuntimeError(f"kernel venv check failed: {check.stderr[-1500:]}")
    info = {"event": "prime_runtime", "seconds": round(time.time() - started, 1), "node": probe.stdout.strip(),
            "numpy": check.stdout.strip(), "users": len(users), "nproc": os.cpu_count()}
    log(info)
    return info


def hidden_paths(*extra):
    """What a session must not see (covered by an empty tmpfs)."""
    paths = ["/kaggle", "/root", "/runs", "/opt/arc", "/run/arc", "/assets", "/tmp/arena", *map(str, extra)]
    for item in os.environ.get("PRIME_HIDE", "").split(":"):
        if item:
            paths.append(item)
    return [p for p in dict.fromkeys(paths) if p and Path(p).is_dir()]


ISOLATION_PROBE = r"""
import json, os, socket, sys
leak = sys.argv[1]
def connect(host, port):
    s = socket.socket(); s.settimeout(3)
    try:
        return s.connect_ex((host, port))
    except OSError as exc:
        return str(exc)
out = {"task_data_visible": os.path.exists(leak), "egress": connect("<ip>", 80),
       "proxy": connect("127.0.0.1", int(sys.argv[2])), "uid": os.getuid(),
       "workspace": sorted(os.listdir("/workspace"))}
print("ISOLATION " + json.dumps(out))
"""


async def isolation_check(ctx, user, workspace, leak):
    """One sandboxed probe as a session user, before any session: the task data (with test outputs) must be
    invisible, the network closed, the proxy reachable."""
    from . import prime_sandbox as sandbox

    sandbox.own(workspace, user)
    command = sandbox.command([str(RUNTIME / "kernel/bin/python"), "-I", "-c", ISOLATION_PROBE, str(leak),
                               str(ctx.proxy_port)], user, workspace, {"LANG": "C.UTF-8"}, hide=ctx.hide, net="proxy",
                              proxy=(ctx.proxy.socket_dir, ctx.proxy_port), python=str(RUNTIME / "kernel/bin/python"))
    proc = await asyncio.create_subprocess_exec(*command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = await asyncio.wait_for(proc.communicate(), 120)
    text = out.decode("utf-8", "replace")
    row = next((json.loads(line.split("ISOLATION ", 1)[1]) for line in text.splitlines() if "ISOLATION " in line),
               None)
    ok = bool(row) and not row["task_data_visible"] and row["egress"] != 0 and row["proxy"] == 0 and row["uid"] != 0
    return {"ok": ok, "probe": row, "output": text[-2000:]}


# --- model proxy ----------------------------------------------------------------------------------------------------

class Proxy:
    """The sessions' only route to the model server: a unix socket (forwarded to 127.0.0.1:18080 inside each
    sandbox) or, without the sandbox, 127.0.0.1:18080 itself. The bearer key names the entry."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.base = ctx.base_url.rstrip("/")  # .../v1
        self.root = self.base.removesuffix("/v1")
        self.keys = {}
        self.runner = self.client = None
        self.socket_dir = None

    def register(self, entry):
        key = secrets.token_hex(16)
        self.keys[key] = entry
        return key

    async def start(self, socket_dir, tcp):
        from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

        self.client = ClientSession(timeout=ClientTimeout(total=None, sock_read=3600),
                                    connector=TCPConnector(limit=512))
        app = web.Application(client_max_size=512 * 1024 * 1024)
        app.router.add_route("*", "/{tail:.*}", self.handle)
        self.runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
        await self.runner.setup()
        if socket_dir is not None:  # the sandboxes' route (bound into each at /proxy)
            self.socket_dir = Path(socket_dir)
            self.socket_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self.socket_dir, 0o755)
            sock = self.socket_dir / "proxy.sock"
            sock.unlink(missing_ok=True)
            await web.UnixSite(self.runner, str(sock)).start()
            os.chmod(sock, 0o666)
        if tcp:
            await web.TCPSite(self.runner, "127.0.0.1", self.ctx.proxy_port).start()

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()
        if self.client:
            await self.client.close()

    async def count(self, payload):
        body = {"model": payload["model"], "messages": payload["messages"],
                "chat_template_kwargs": payload.get("chat_template_kwargs") or {}}
        if payload.get("tools"):
            body["tools"] = payload["tools"]
        try:
            async with self.client.post(self.root + "/tokenize", json=body) as response:
                text = await response.read()
                status = response.status
        except Exception as exc:
            self.ctx.server_failed(f"tokenize: {type(exc).__name__}: {exc}"[:500])
            raise
        if status != 200:
            if status >= 500:
                self.ctx.server_failed(f"tokenize {status}")
            raise RuntimeError(f"tokenize {status}: {text[:300]!r}")
        return int(json.loads(text)["count"])

    def translate(self, payload):
        """Pi's OpenAI request as base18 sends its own: reasoning_effort into the chat template's kwargs, base18's
        sampling where Pi sends none, usage in the stream, the served model's id."""
        cfg = self.ctx.cfg
        effort = payload.pop("reasoning_effort", None)
        kwargs = payload.setdefault("chat_template_kwargs", {})
        if effort is not None:
            kwargs.setdefault("reasoning_effort", effort)
            kwargs.setdefault("enable_thinking", effort not in ("none", "off"))
        payload.setdefault("temperature", self.ctx.temperature)
        payload.setdefault("top_p", cfg["top_p"])
        payload.setdefault("top_k", cfg["top_k"])
        payload["model"] = self.ctx.served
        if payload.get("stream"):
            payload.setdefault("stream_options", {})["include_usage"] = True
        if payload.get("max_completion_tokens") and not payload.get("max_tokens"):
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        payload["max_tokens"] = min(int(payload.get("max_tokens") or self.ctx.cap), self.ctx.cap)
        return payload

    async def handle(self, request):
        from aiohttp import web

        key = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        entry = self.keys.get(key)
        if entry is None:
            return web.json_response({"error": {"message": "unknown key"}}, status=401)
        path = "/" + request.match_info["tail"].lstrip("/")
        if (request.method, path) != ("POST", "/v1/chat/completions"):
            entry.stats["refused_path"] += 1
            return web.json_response({"error": {"message": f"{request.method} {path} is not served here"}},
                                     status=403)
        payload = self.translate(json.loads(await request.read()))
        if not payload.get("stream"):
            entry.stats["refused_path"] += 1
            return web.json_response({"error": {"message": "only streamed chat completions are served here"}},
                                     status=400)
        response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream",
                                                           "Cache-Control": "no-cache"})
        await response.prepare(request)
        waiting = asyncio.Event()
        waiting.set()

        async def heartbeat():  # Node's fetch drops a body silent for 300 s; a request can wait long at the gate
            while waiting.is_set():
                await asyncio.sleep(15)
                if waiting.is_set():
                    await response.write(b": waiting\n\n")

        beat = asyncio.create_task(heartbeat())
        entry.pending += 1
        try:
            await self.forward(entry, payload, response, waiting, beat)
        except asyncio.CancelledError:
            raise
        except ConnectionResetError:  # the session went away (closed after its submit): nothing to tell it
            entry.stats["client_gone"] += 1
            return response
        except Exception as exc:  # noqa: BLE001 - the session sees a server error, the run goes on
            entry.stats["errors"] += 1
            entry.log("proxy_error", error=f"{type(exc).__name__}: {exc}"[:500])
            with contextlib.suppress(Exception):
                await sse_error(response, f"proxy error: {type(exc).__name__}: {exc}"[:500], 500)
        finally:
            entry.pending -= 1
            entry.touch()
            waiting.clear()
            beat.cancel()
            with contextlib.suppress(BaseException):
                await beat
        with contextlib.suppress(Exception):
            await response.write_eof()
        return response

    async def forward(self, entry, payload, response, waiting, beat):
        ctx, cfg = self.ctx, self.ctx.cfg
        purpose = purpose_of(payload)
        if entry.conv and entry.submitted is None:
            entry.poll_answer()
        if entry.submitted is not None:  # the puzzle is finished: the session is being closed
            entry.stats["refused_after_submit"] += 1
            await sse_error(response, "The puzzle is finished.")
            return
        if entry.answer is not None and last_user_text(payload).strip() == CONTINUATION:
            entry.settled = True  # the agent stopped with an answer and is asked to go on
        keep_cut = cfg.get("keep_cut_replies") and purpose == "gameplay"
        if keep_cut:
            entry.stats["cut_restored"] += entry.cut.restore(payload["messages"])
        count = await self.count(payload)
        room = ctx.window - count
        requested = payload["max_tokens"]
        if room < min(requested, cfg["min_reply_tokens"]):
            entry.stats["overflow_refused"] += 1
            entry.log("overflow", prompt_tokens=count, max_tokens=requested, purpose=purpose)
            await sse_error(response, f"This model's maximum context length is {ctx.window} tokens. However, you "
                                      f"requested {count + requested} tokens ({count} in the messages, {requested} in "
                                      "the completion). Please reduce the length of the messages or completion.")
            return
        if requested > room:
            payload["max_tokens"] = room
            entry.stats["cap_fitted"] += 1
        t_queue = time.monotonic()
        units = await ctx.gate.acquire(entry.priority(), count + payload["max_tokens"], entry=entry)
        tail, started, body = b"", time.monotonic(), []
        flight = object()
        ctx.inflight[flight] = started
        prefilling, failed, down, cancelled = False, None, False, False
        try:
            waiting.clear()
            beat.cancel()
            with contextlib.suppress(BaseException):
                await beat
            if ctx.prefill is not None:
                await ctx.prefill.acquire()
                prefilling = True
            entry.streaming += 1
            try:
                async with self.client.post(self.base + "/chat/completions", json=payload) as upstream:
                    if upstream.status != 200:
                        text = (await upstream.read()).decode("utf-8", "replace")
                        try:
                            message = json.loads(text).get("error", {}).get("message") or text
                        except (ValueError, AttributeError):
                            message = text
                        failed = f"{upstream.status}: {message}"[:500]
                        down = upstream.status >= 500
                        entry.stats["errors"] += 1
                        entry.log("upstream_error", status=upstream.status, message=str(message)[:500],
                                  purpose=purpose)
                        await sse_error(response, str(message)[:2000], upstream.status)
                        return
                    async for chunk in upstream.content.iter_any():
                        if prefilling:  # the first streamed bytes: this request's prefill is over
                            ctx.prefill.release()
                            prefilling = False
                        await response.write(chunk)
                        tail = (tail + chunk)[-65536:]
                        if keep_cut:
                            body.append(chunk)
                ctx.server_ok()
            except asyncio.CancelledError:
                cancelled = True
                raise
            except Exception as exc:
                failed, down = f"{type(exc).__name__}: {exc}"[:500], True
                raise
            finally:
                if prefilling:
                    ctx.prefill.release()
                entry.streaming -= 1
                usage, finish = usage_of(tail)
                if keep_cut and finish == "length" and failed is None and not cancelled:
                    entry.stats["cut_kept"] += entry.cut.keep(payload["messages"], b"".join(body))
                if usage:
                    ctx.charge(entry, purpose, usage, finish, count, t_queue, started, time.monotonic())
                elif failed is None and not cancelled:
                    failed, down = "stream ended without usage", True
                if not usage:
                    entry.stats["uncharged"] += 1
        finally:
            ctx.inflight.pop(flight, None)
            if down:
                ctx.server_failed(failed)
            if failed is not None or cancelled:
                ctx.gate.drop(entry)  # a failed request gives its hot place up at once
            ctx.gate.release(units)


async def sse_error(response, message, status=400):
    """An error the OpenAI SDK raises from the stream (``data.error``), so Pi sees the server's message."""
    body = {"error": {"message": message, "type": "BadRequestError" if status < 500 else "InternalServerError",
                      "code": status}}
    await response.write(b"data: " + json.dumps(body).encode() + b"\n\n")


# --- one roster entry -----------------------------------------------------------------------------------------------

class Entry:
    def __init__(self, ctx, entry_id, task, user):
        self.ctx, self.id, self.task, self.user = ctx, entry_id, task, user
        self.dir = ctx.output / "sessions" / entry_id.replace("#", "_")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.logfile = (self.dir / "log.jsonl").open("a", buffering=1)
        self.spent = self.prompt_tokens = 0
        self.replies, self.answers = [], []  # answers: {"t": monotonic, "answer": parse_answer(...)}
        self.answer, self.settled = None, False
        self.answer_mtime = None
        self.pending = self.streaming = 0
        self.last_activity = time.monotonic()
        self.process = self.reader = None
        self.key = None
        self.status, self.error = "waiting", None
        self.events, self.tool_open, self.last_event = {}, 0, None
        self.busy = set()  # "agent" between agent_start and agent_end, "compaction" inside one
        self.conv = bool(ctx.cfg.get("conv"))
        self.cut = CutReplies()
        self.candidates, self.programs, self.submitted = [], 0, None  # conv1's (arc2.conv); submitted: monotonic
        self.last_assistant, self.agent_ends, self.handled, self.stopped_at = None, 0, 0, None
        self.stats = {"requests": {}, "completion_tokens": {}, "prompt_tokens": {}, "errors": 0, "overflow_refused": 0,
                      "cap_fitted": 0, "count_mismatch": 0, "max_count_error": 0, "uncharged": 0, "refused_path": 0,
                      "watchdog_aborts": 0, "continuations": 0, "restarts": 0, "session_exits": [],
                      "answer_writes": 0, "answer_invalid": 0, "programs": 0, "verified_programs": 0,
                      "harness_replies": {}, "refused_after_submit": 0, "compaction_releases": 0,
                      "cut_kept": 0, "cut_restored": 0, "client_gone": 0}

    def log(self, event, **row):
        self.logfile.write(json.dumps({"t": round(time.time(), 3), "event": event, **row}) + "\n")

    def touch(self):
        self.last_activity = time.monotonic()

    def priority(self):
        cfg = self.ctx.cfg
        delay = cfg["settled_delay"] if self.settled else cfg["confirm_delay"] if self.answer is not None else 0
        return self.spent + (delay or 0)

    def poll_answer(self):
        path = self.dir / "session/workspace/answer.json"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if mtime == self.answer_mtime:
            return
        self.answer_mtime = mtime
        self.stats["answer_writes"] += 1
        if self.conv:
            return self.take_submit(path, mtime)
        try:
            answer = parse_answer(json.loads(path.read_text()), len(self.task["test"]))
        except (OSError, ValueError):
            answer = None
        if answer is None:
            self.stats["answer_invalid"] += 1
            self.log("answer_invalid")
            return
        when = time.monotonic() - max(0.0, time.time() - mtime)
        self.answers.append({"t": when, "answer": answer})
        self.answer = answer
        self.log("answer", right=right_outputs(answer, self.task), of=len(self.task["test"]),
                 tokens=self.spent)

    # -- conv1's answer workflow (arc2.conv.ConvEngine.take / check_submit / take_submit / ledger) --
    def take_submit(self, path, mtime):
        try:
            data = json.loads(path.read_text())
            raw = data["guesses"]
        except (OSError, ValueError, KeyError, TypeError):
            raw = None
        n = len(self.task["test"])
        ok = (isinstance(raw, list) and len(raw) == 2 and raw[0] is not None
              and all(g is None or (isinstance(g, list) and len(g) == n and all(tasks.valid(x) for x in g))
                      for g in raw))
        if not ok:  # the kernel's submit() checks first; a file that fails here was not written by it
            self.stats["answer_invalid"] += 1
            self.log("answer_invalid")
            return
        when = time.monotonic() - max(0.0, time.time() - mtime)
        verified = {json.dumps(c["tests"]) for c in self.candidates if c.get("verified")}
        for rank, guess in enumerate(raw, 1):
            if guess is None:
                continue
            self.candidates.append({"code": f"submit-{rank}", "attempt": 1, "reply": len(self.replies),
                                    "tokens_at": self.spent, "t": when, "passed": 0, "pairs": 0, "accuracy": 0.0,
                                    "verified": json.dumps(guess) in verified, "tests": guess, "submitted": rank})
        self.submitted = when
        self.answer = self.current_answer()
        self.answers.append({"t": when, "answer": self.answer})
        self.log("answer", right=right_outputs(self.answer, self.task), of=n, tokens=self.spent, submitted=True)

    def current_answer(self, pool=None):
        """parse_answer's form from conv1's ranking (``loop.guesses``) over ``pool`` (default: every candidate)."""
        picked = guesses(self.candidates if pool is None else pool, [t["input"] for t in self.task["test"]])
        return [[grid for grid, _ in row] or None for row in picked]

    def ledger(self):
        readings = {}
        for c in self.candidates:
            if c.get("verified") and c.get("program"):
                readings.setdefault(json.dumps(c["tests"]), []).append(c["program"])
        if not readings:
            return "No program passes every training pair yet."
        lines = [f"Programs that pass every training pair give {len(readings)} distinct test output"
                 f"{'s' if len(readings) > 1 else ''}:"]
        for n, (_, programs) in enumerate(readings.items()):
            lines.append(f"- reading {chr(65 + n)}: program{'s' if len(programs) > 1 else ''} "
                         + ", ".join(str(p) for p in programs))
        return "\n".join(lines)

    async def check_program(self, code):
        """conv1's reply to a ```python block that defines ``transform``: run on every training and test input,
        judged cell by cell; numbered and callable as ``program_N`` in the kernel; the ledger of readings."""
        self.programs += 1
        n = self.programs
        self.stats["programs"] += 1
        inputs = [p["input"] for p in self.task["train"]] + [t["input"] for t in self.task["test"]]
        result = await self.ctx.execute(code, inputs)
        verdict = tasks.judge(self.task, result)
        tests = verdict["tests"]
        earlier = {json.dumps(c["tests"]): c["program"] for c in self.candidates
                   if c.get("verified") and c.get("program")}
        self.candidates.append({
            "code": __import__("hashlib").sha256(code.encode()).hexdigest()[:16], "attempt": 1,
            "reply": len(self.replies), "tokens_at": self.spent, "t": time.monotonic(), "passed": verdict["passed"],
            "pairs": verdict["pairs"], "accuracy": verdict["accuracy"], "verified": verdict["verified"],
            "tests": tests, "program": n})
        if not result.get("error"):
            folder = self.dir / "session/home/.arc/programs"
            (folder / f"program_{n}.py.tmp").write_text(code)
            os.chmod(folder / f"program_{n}.py.tmp", 0o644)
            os.replace(folder / f"program_{n}.py.tmp", folder / f"program_{n}.py")
        if verdict["verified"]:
            self.stats["verified_programs"] += 1
            same = earlier.get(json.dumps(tests))
            lines = [f"Program {n} passes every training pair (`program_{n}` in python)."]
            lines.append(f"Its test output is the same as program {same}'s." if same else
                         "Its test output is a new reading:\n" + "\n\n".join(
                             tasks.text(g) if tasks.valid(g) else tasks.describe(g) for g in tests))
            text = "\n".join(lines)
        else:
            text = f"Program {n}: " + tasks.feedback(self.task, result, verdict).replace(
                tasks.FIX_LINE, "Fix the rule and send the corrected program, or keep exploring in python.")
        self.log("program", n=n, passed=verdict["passed"], pairs=verdict["pairs"], verified=verdict["verified"],
                 error=bool(result.get("error")), tokens=self.spent)
        return text + "\n\n" + self.ledger()

    async def harness_reply(self):
        """Prime's agent stopped without a tool call: conv1's next user message."""
        last = self.last_assistant or {}
        code = tasks.extract_code(last.get("text") or "")
        if code is not None:
            kind, text = "program", await self.check_program(code)
        elif last.get("stop") == "length":
            kind, text = "cut", CONV_CUT
        else:
            kind, text = "nothing", CONV_NOTHING
        self.stats["harness_replies"][kind] = self.stats["harness_replies"].get(kind, 0) + 1
        self.log("harness_reply", kind=kind, stop=last.get("stop"))
        await self.send({"id": f"h-{self.agent_ends}", "type": "prompt", "message": text,
                         "streamingBehavior": "followUp"})
        self.touch()

    def first_message(self):
        """conv1's first user message (ConvEngine.start): the task as text, then its pictures. Pi's rpc prompt takes
        one text and a list of images (appended after it), so conv1's label before each picture goes into one line
        naming them in order."""
        cfg = self.ctx.cfg
        text = tasks.task_prompt(self.task)
        if not cfg["images"]:
            return text, []
        parts = tasks.image_parts(self.task, int(cfg["image_side"]), False)
        labels = [p["text"].strip().rstrip(":") for p in parts[1:] if p["type"] == "text"]
        text += parts[0]["text"] + "\nThe pictures, in order: " + ", ".join(labels) + "."
        images = []
        for part in parts:
            if part["type"] == "image_url":
                head, data = part["image_url"]["url"].split(",", 1)
                images.append({"type": "image", "data": data, "mimeType": head[len("data:"):].split(";")[0]})
        return text, images

    # -- the session --
    def models_json(self):
        ctx = self.ctx
        return {"providers": {"arena": {
            "baseUrl": f"http://127.0.0.1:{ctx.proxy_port}/v1", "apiKey": "PRIME_PROXY_KEY",
            "api": "openai-completions",
            "models": [{"id": ctx.served, "name": "flash", "reasoning": True,
                        "thinkingLevelMap": {"off": "none", "minimal": "low", "low": "low", "medium": "medium",
                                             "high": "xhigh", "xhigh": "xhigh"},
                        "input": ["text", "image"] if ctx.cfg.get("images") else ["text"], "contextWindow": int(ctx.window), "maxTokens": int(ctx.cap),
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                        "compat": {"supportsDeveloperRole": False, "supportsStore": False,
                                   "supportsReasoningEffort": True}}]}}}

    def prepare_dirs(self):
        root = self.dir / "session"
        ws, home = root / "workspace", root / "home"
        shutil.rmtree(root, ignore_errors=True)
        ws.mkdir(parents=True)
        home.mkdir(parents=True)
        visible = {"train": self.task["train"], "test": [{"input": t["input"]} for t in self.task["test"]]}
        if self.conv:  # conv1: no files; the kernel's globals come from ~/.arc (startup file)
            arc = home / ".arc"
            (arc / "programs").mkdir(parents=True)
            sandboxed = self.ctx.cfg["sandbox"]
            answer = "/workspace/answer.json" if sandboxed else str(ws / "answer.json")
            (arc / "task.json").write_text(json.dumps({**visible, "answer_path": answer,
                                                       "call_seconds": int(self.ctx.cfg["call_seconds"])}) + "\n")
            (arc / "system.md").write_text(CONV_SYSTEM)
            startup = home / ".ipython/profile_default/startup"
            startup.mkdir(parents=True)
            shutil.copyfile(self.ctx.player / "startup.py", startup / "00-arc.py")
        else:
            shutil.copytree(self.ctx.player / "workspace", ws, dirs_exist_ok=True)
            (ws / "task.json").write_text(json.dumps(visible) + "\n")
        agent = home / ".prime/agent"
        agent.mkdir(parents=True)
        (agent / "models.json").write_text(json.dumps(self.models_json(), indent=1) + "\n")
        return ws, home

    def command(self, resume):
        ctx, sandboxed = self.ctx, self.ctx.cfg["sandbox"]
        ws, home = self.dir / "session/workspace", self.dir / "session/home"
        env = {"HOME": "/home/session" if sandboxed else str(home), "USER": self.user if sandboxed else "prime",
               "LANG": "C.UTF-8", "PATH": f"{RUNTIME}/node/bin:{RUNTIME}/kernel/bin:/usr/local/bin:/usr/bin:/bin",
               "PRIME_PROXY_KEY": self.key, "PRIME_AGENT_KERNEL_PYTHON": f"{RUNTIME}/kernel/bin/python",
               "PI_OFFLINE": "1", "PRIME_AGENT_INTERNAL_LEGACY_OWNED_WORKER_FRONTEND": "1"}
        argv = [f"{RUNTIME}/node/bin/node", f"{RUNTIME}/prime-agent/packages/coding-agent/dist/bundle/cli.js",
                "--mode", "rpc", "--provider", "arena", "--model", "flash", "--thinking", ctx.cfg["thinking"]]
        if self.conv:  # conv1's system prompt; conv1's replies take the autonomous continuation's place
            argv += ["--system-prompt", ("/home/session" if sandboxed else str(home)) + "/.arc/system.md"]
        else:
            argv += ["--autonomous", "--autonomous-max-continuations", "1000000", "--autonomous-max-turns",
                     "1000000000", "--autonomous-max-tokens", "1000000000", "--autonomous-timeout-ms", "1000000000"]
        if resume:
            argv.append("--continue")
        drop = "^\\{\"type\":\"(message_update|tool_execution_update)\""
        shell = " ".join(shlex.quote(a) for a in argv) + f" 2>&1 | grep --line-buffered -v -E {shlex.quote(drop)}"
        if not sandboxed:
            return ["/bin/bash", "-c", shell], str(ws), env
        from . import prime_sandbox as sandbox

        for path in (ws, home):
            sandbox.own(path, self.user)
        command = sandbox.command(["/bin/bash", "-c", "set -o pipefail 2>/dev/null; " + shell], self.user, ws, env,
                                  extra_binds=[(home, "/home/session", False)], hide=ctx.hide, net="proxy",
                                  proxy=(ctx.proxy.socket_dir, ctx.proxy_port),
                                  python=str(RUNTIME / "kernel/bin/python"),
                                  memory=int(ctx.cfg["session_memory_gib"]) << 30)
        return command, None, None

    async def start_session(self, resume=False):
        if not resume:
            self.prepare_dirs()
        else:  # the killed the lease (Pi reclaims stale ones too)
            shutil.rmtree(self.dir / "session/home/.prime/agent/session-leases", ignore_errors=True)
            # the dead process's open agent run / tool cell never ends: a new process starts idle (preflight v4 C/D:
            # a stale tool_open kept the harness from answering the resumed session for 900 s)
            self.busy.clear()
            self.tool_open = 0
            self.handled = self.agent_ends
        if self.key is None:
            self.key = self.ctx.proxy.register(self)
        command, cwd, env = self.command(resume)
        self.process = await asyncio.create_subprocess_exec(
            *command, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True, limit=1 << 28)
        self.reader = asyncio.create_task(self.read_events(self.process.stdout))
        if self.conv:
            command = {"type": "prompt", "message": CONV_NOTHING} if resume else dict(
                zip(("message", "images"), self.first_message()), type="prompt")
        else:
            command = {"type": "prompt", "message": self.ctx.prompt if not resume else CONTINUATION}
        await self.send({"id": "1" if not resume else f"resume-{self.stats['restarts']}", **command})
        self.log("session_start", resume=resume, pid=self.process.pid)
        self.touch()

    async def send(self, command):
        if self.process is None or self.process.stdin is None:
            return
        try:
            self.process.stdin.write(json.dumps(command).encode() + b"\n")
            await self.process.stdin.drain()
        except (ConnectionError, BrokenPipeError):
            pass

    async def read_events(self, stream):
        log = (self.dir / "rpc-events.jsonl").open("a")
        try:
            while True:
                try:
                    line = await stream.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    line = (await stream.read(1 << 20))[:4096] + b"\n"
                if not line:
                    return
                try:
                    event = json.loads(line)
                except ValueError:
                    log.write(json.dumps({"t": round(time.time(), 3), "text": line[:2000].decode("utf-8", "replace")})
                              + "\n")
                    continue
                kind = event.get("type") if isinstance(event, dict) else None
                if kind not in KEEP:
                    continue
                self.events[kind] = self.events.get(kind, 0) + 1
                self.last_event = kind
                if kind in ("agent_start", "compaction_start"):
                    self.busy.add(kind.split("_")[0])
                elif kind in ("agent_end", "compaction_end"):
                    self.busy.discard(kind.split("_")[0])
                if kind == "agent_end":
                    self.agent_ends += 1
                    self.stopped_at = time.monotonic()
                elif kind == "compaction_end" and self.ctx.cfg.get("release_on_compaction"):
                    self.stats["compaction_releases"] += 1
                    self.ctx.gate.release_hot(self)
                if kind == "tool_execution_start":
                    self.tool_open += 1
                elif kind == "tool_execution_end":
                    self.tool_open = max(0, self.tool_open - 1)
                    self.poll_answer()
                row = {"t": round(time.time(), 3), "type": kind}
                if kind == "compaction_end":
                    result = event.get("result") or {}
                    row.update(reason=event.get("reason"), tokens_before=result.get("tokensBefore"),
                               summary_chars=len(result.get("summary") or ""), error=event.get("errorMessage"),
                               aborted=event.get("aborted"))
                elif kind in ("refine_complete", "refine_failed"):
                    result = event.get("result") or {}
                    row.update(error=str(event.get("error"))[:300] if event.get("error") else None,
                               summary=str(result.get("summary") or "")[:300], edits=len(result.get("edits") or []))
                elif kind == "message_end":
                    message = event.get("message") or {}
                    if message.get("role") != "assistant":
                        continue
                    text = "".join(p.get("text", "") for p in message.get("content") or []
                                   if isinstance(p, dict) and p.get("type") == "text")
                    self.last_assistant = {"text": text, "stop": message.get("stopReason")}
                    row.update(stop=message.get("stopReason"), error=str(message.get("errorMessage") or "")[:300] or None)
                elif kind == "tool_execution_start":
                    args = event.get("args") or {}
                    code = str(args.get("code") or "")
                    row.update(tool=event.get("toolName"), chars=len(code), rlm="rlm" in code,
                               agents="agent" in code)
                elif kind in ("compaction_start", "agent_end", "auto_retry_start", "auto_retry_end", "error",
                              "extension_error", "response"):
                    row.update({k: (str(v)[:300] if not isinstance(v, (int, float, bool)) else v)
                                for k, v in event.items() if k not in ("type", "messages", "message")})
                log.write(json.dumps(row) + "\n")
                log.flush()
        finally:
            log.close()

    async def stop_session(self):
        if self.process is not None and self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
        if self.ctx.cfg["sandbox"]:
            await asyncio.to_thread(subprocess.run, ["pkill", "-KILL", "-u", self.user], check=False)
        if self.process is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.process.wait(), 30)
        if self.reader is not None:
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(self.reader, 10)

    async def watchdog(self):
        """A tool cell running with nothing in the proxy for ``watchdog_seconds`` (their kernel has no cell timeout):
        rpc abort. A session whose agent is not running (after agent_end, also when an automatic compaction ended
        the run) for ``idle_seconds``: the stock continuation prompt."""
        cfg, now = self.ctx.cfg, time.monotonic()
        if self.pending or self.process is None or self.ctx.gate.closed:
            return
        quiet = now - self.last_activity
        if self.conv and not self.busy and not self.tool_open:
            if self.handled < self.agent_ends and now - (self.stopped_at or now) >= float(cfg["reply_delay_seconds"]):
                self.handled = self.agent_ends
                await self.harness_reply()
            elif quiet >= float(cfg["watchdog_seconds"]):  # stopped with nothing to answer (lost event): go on
                self.stats["continuations"] += 1
                self.log("continue", last_event=self.last_event)
                await self.send({"id": f"cont-{self.stats['continuations']}", "type": "prompt",
                                 "message": CONV_NOTHING, "streamingBehavior": "followUp"})
                self.touch()
        elif not self.busy and not self.tool_open:
            if quiet >= float(cfg["idle_seconds"]):
                self.stats["continuations"] += 1
                if self.answer is not None:
                    self.settled = True
                self.log("continue", last_event=self.last_event)
                await self.send({"id": f"cont-{self.stats['continuations']}", "type": "prompt",
                                 "message": CONTINUATION})
                self.touch()
        elif quiet >= float(cfg["watchdog_seconds"]):
            self.stats["watchdog_aborts"] += 1
            self.log("abort", last_event=self.last_event, tool_open=self.tool_open)
            await self.send({"id": f"abort-{self.stats['watchdog_aborts']}", "type": "abort"})
            self.last_activity = now - float(cfg["watchdog_seconds"]) + 120

    async def run(self, start_at):
        ctx = self.ctx
        await asyncio.sleep(max(0.0, start_at - time.monotonic()))
        if ctx.ending.is_set():
            self.status = "not_started"
            return
        try:
            await self.start_session()
            self.status = "playing"
            while not ctx.ending.is_set():
                try:
                    await asyncio.wait_for(asyncio.shield(self.process.wait()), timeout=float(ctx.cfg["answer_poll_seconds"]))
                except asyncio.TimeoutError:
                    self.poll_answer()
                    if self.submitted is not None:  # conv1: submitting ends the puzzle; the place goes to the next
                        self.status = "submitted"
                        self.log("closed", tokens=self.spent)
                        self.ctx.gate.drop(self)
                        return
                    await self.watchdog()
                    continue
                self.poll_answer()
                self.ctx.gate.drop(self)
                if self.submitted is not None:
                    self.status = "submitted"
                    return
                self.stats["session_exits"].append({"code": self.process.returncode, "tokens": self.spent})
                self.log("session_exit", code=self.process.returncode)
                if ctx.ending.is_set():
                    break
                if self.stats["restarts"] >= int(ctx.cfg["restarts"]):
                    self.status = "session_exit"
                    return
                self.stats["restarts"] += 1
                await self.start_session(resume=True)
            if self.status == "playing":
                self.status = "ended"
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one entry's failure never ends the trial
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"[:500]
            ctx.failures.append(f"{self.id}: {self.error}")
        finally:
            self.poll_answer()
            await asyncio.shield(self.stop_session())

    def row(self, clock=None):
        stats = self.stats
        return {"task": self.id, "spent": self.spent, "prompt_tokens": self.prompt_tokens, "status": self.status,
                "error": self.error, "right": right_outputs(self.answer, self.task), "of": len(self.task["test"]),
                "answers": [{"t": a["t"], "clock": clock(a["t"]) if clock else None,
                             "right": right_outputs(a["answer"], self.task), "answer": a["answer"]}
                            for a in self.answers],
                "settled": self.settled, "rpc_events": dict(self.events), **{k: stats[k] for k in stats}}


# --- read-out -------------------------------------------------------------------------------------------------------

def read_out(entries, marks, *, step=None, cut=None):
    """Score against the roster's total completion tokens: an entry's answer at a mark is its latest valid
    ``answer.json`` written by the time the roster had generated that many tokens (``loop.token_clock``)."""
    clock = token_clock([x for e in entries for x in e.replies])
    total = clock(float("inf"))
    exact = total if cut is None else clock(cut)
    outputs = sum(len(e.task["test"]) for e in entries)
    timed = [(e, [(clock(a["t"]), a["answer"]) for a in e.answers]) for e in entries]
    pools = [(e, [(clock(c["t"]), c) for c in getattr(e, "candidates", ())]) for e in entries]

    def row(tokens):
        right = answered = 0
        for (e, rows), (_, cands) in zip(timed, pools):
            if getattr(e, "conv", False):  # conv1's read-out: loop.guesses over the submit and the programs by then
                pool = [c for at, c in cands if at <= tokens + 1e-6]
                answer = e.current_answer(pool)
                answered += any(c.get("submitted") for c in pool)
                right += right_outputs(answer, e.task)
                continue
            answer = None
            for at, ans in rows:
                if at <= tokens + 1e-6:
                    answer = ans
            right += right_outputs(answer, e.task)
            answered += answer is not None
        return {"tokens": tokens, "tokens_per_task": tokens / max(1, len(entries)), "reached": tokens <= exact + 1e-6,
                "right": right, "score_outputs": 100 * right / max(1, outputs), "entries_answered": answered}
    out = {"tokens": total, "exact_through": exact, "marks": {name: row(t) for name, t in marks.items()}}
    if step:
        out["curve"] = [row(step * k) for k in range(1, int(max(total, *marks.values()) // step) + 1)]
    return out, clock


def host_stats():
    """Host RAM, load and CPU count (mon.py hostline keys); {} where /proc is missing."""
    try:
        info = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
        load1 = float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return {}
    total, available = info.get("MemTotal", 0) / 2 ** 20, info.get("MemAvailable", 0) / 2 ** 20
    vcpus = os.cpu_count() or 1
    return {"mem_total_gib": round(total, 1), "mem_used_gib": round(total - available, 1), "load1": load1,
            "vcpus": vcpus, "cpu_busy": round(min(1.0, load1 / vcpus), 3)}


def totals(entries):
    out = {"requests": {}, "completion_tokens": {}, "prompt_tokens": {}, "rpc_events": {}}
    for e in entries:
        for name in ("requests", "completion_tokens", "prompt_tokens"):
            for key, value in e.stats[name].items():
                out[name][key] = out[name].get(key, 0) + value
        for key, value in e.events.items():
            out["rpc_events"][key] = out["rpc_events"].get(key, 0) + value
        for name in ("errors", "overflow_refused", "cap_fitted", "count_mismatch", "uncharged", "refused_path",
                     "watchdog_aborts", "continuations", "restarts", "answer_writes", "answer_invalid", "programs",
                     "verified_programs", "refused_after_submit", "compaction_releases", "cut_kept", "cut_restored",
                     "client_gone"):
            out[name] = out.get(name, 0) + int(e.stats[name])
        for key, value in e.stats["harness_replies"].items():
            out.setdefault("harness_replies", {})[key] = out.get("harness_replies", {}).get(key, 0) + value
    out["entries_restarted"] = sum(e.stats["restarts"] > 0 for e in entries)
    out["entries_settled"] = sum(e.settled for e in entries)
    out["entries_submitted"] = sum(e.submitted is not None for e in entries)
    out["entries_started"] = sum(e.spent > 0 for e in entries)
    out["statuses"] = {s: sum(e.status == s for e in entries) for s in sorted({e.status for e in entries})}
    return out


# --- the trial ------------------------------------------------------------------------------------------------------

class Context:
    down_since = None
    server_failures = 0
    last_failure = None

    def server_failed(self, why):
        self.server_failures += 1
        self.last_failure = why
        if self.down_since is None:
            self.down_since = time.monotonic()

    def server_ok(self):
        self.down_since = None

    def charge(self, entry, purpose, usage, finish, count, t_queue, t0, t1):
        prompt, completion = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        entry.spent += completion
        entry.prompt_tokens += prompt
        self.spent += completion
        for name, value in (("requests", 1), ("completion_tokens", completion), ("prompt_tokens", prompt)):
            entry.stats[name][purpose] = entry.stats[name].get(purpose, 0) + value
        if prompt and prompt != count:
            entry.stats["count_mismatch"] += 1
            entry.stats["max_count_error"] = max(entry.stats["max_count_error"], abs(prompt - count))
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        row = {"task": entry.id, "purpose": purpose, "prompt_tokens": prompt, "completion_tokens": completion,
               "cached_tokens": cached, "finish": finish, "t0": t0, "t1": t1, "wait": round(t0 - t_queue, 3)}
        entry.replies.append(row)
        self.gate.charged(entry, completion)
        self.events.write(json.dumps({"t": round(t1 - self.started, 2), **row}) + "\n")
        if self.stop is not None and self.spent >= self.stop and not self.gate.closed:
            self.gate.close()  # no new reply; those in flight finish (the token clock stays exact)
            self.stopped_at = time.monotonic()


async def run_prime(study, trial, *, source, output, server_dir, base_url, deadline_epoch):
    import httpx

    from .gpu import GpuSampler
    from .trial import DRAIN_SECONDS, block_budget, generated_tokens

    from arena.early_stop import stop_paths

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = study["arc2"]
    cfg = settings(trial, study)
    left = deadline_epoch - time.time()
    wall = min(float(trial["wall_seconds"]), left - min(1.1 * DRAIN_SECONDS, 0.25 * left))
    task_map = tasks.load(Path(source) / spec["data"] / trial["split"], trial["tasks"])
    copies = int(trial.get("copies", 1))
    ids = [(f"{i}#{k + 1}" if copies > 1 else i, i) for i in trial["tasks"] for k in range(copies)]
    marks = {name: float(tokens) for name, tokens in (trial.get("marks") or {}).items()}
    ctx = Context()
    ctx.cfg, ctx.output, ctx.base_url = cfg, output, base_url
    ctx.started = time.monotonic()
    ctx.spent, ctx.stop, ctx.stopped_at = 0, max(marks.values(), default=None), None
    ctx.failures, ctx.inflight = [], {}
    ctx.ending = asyncio.Event()
    ctx.temperature = float(study["temperature"])
    ctx.prefill = asyncio.Semaphore(int(cfg["prefill_slots"])) if int(cfg["prefill_slots"]) > 0 else None
    ctx.cap = int(study["completion_cap"])
    ctx.proxy_port = PROXY_PORT
    budget = block_budget(server_dir, float(study["kv_margin"]))
    ctx.window = min(int(study["windows"][0]), int(budget.layout["max_model_len"]) - int(budget.layout["lookahead"]))
    admission = (output / "admission.jsonl").open("a", buffering=1)
    ctx.gate = PriorityGate(budget.capacity, int(study["active_limit"]), cost=budget.cost,
                            log=lambda row: admission.write(json.dumps(row) + "\n"), hot_limit=cfg["hot_limit"],
                            quantum=cfg["quantum_tokens"], hot_idle=cfg["hot_idle_seconds"])
    ctx.events = (output / "events.jsonl").open("a", buffering=1)
    if cfg["conv"]:  # conv1's program runs (arc2.trial.run_arc2: sandbox self-test, at most 8 at once)
        from . import sandbox

        check = await asyncio.to_thread(sandbox.self_test, None)
        if not check["ok"]:
            check = {**await asyncio.to_thread(sandbox.self_test, None, strict=False), "first": check}
        (output / "sandbox-check.json").write_text(json.dumps(check, indent=1))
        if not check["ok"]:
            raise RuntimeError("sandbox self-test failed")
        runner = asyncio.Semaphore(8)

        async def execute(code, inputs):
            async with runner:
                return await asyncio.to_thread(sandbox.run, code, inputs, strict=check["strict"])

        ctx.execute = execute
    ctx.player = Path(source) / "prime/player"
    ctx.prompt = (ctx.player / "task-prompt.txt").read_text().strip()
    ctx.hide = hidden_paths(source, output, server_dir)
    log = (output / "prime.jsonl").open("a", buffering=1)

    def note(row):
        log.write(json.dumps({"t": round(time.monotonic() - ctx.started, 2), **row}) + "\n")

    async with httpx.AsyncClient(base_url=base_url.rstrip("/") + "/", trust_env=False) as client:
        listing = await client.get("models", timeout=30)
        listing.raise_for_status()
        ctx.served = listing.json()["data"][0]["id"]
        served_before = await generated_tokens(client)
    users = [f"pa2u{n:03d}" for n in range(len(ids))] if cfg["sandbox"] else []
    runtime = await asyncio.to_thread(prepare_runtime, source, users, note)
    ctx.proxy = Proxy(ctx)
    await ctx.proxy.start(Path("/run/prime-proxy") if cfg["sandbox"] else None, tcp=not cfg["sandbox"])
    entries = [Entry(ctx, name, task_map[i], users[n] if users else "prime") for n, (name, i) in enumerate(ids)]
    if cfg["sandbox"]:
        probe_ws = output / "isolation-workspace"
        probe_ws.mkdir(exist_ok=True)
        leak = Path(source) / spec["data"] / trial["split"] / f"{trial['tasks'][0]}.json"
        isolation = await isolation_check(ctx, users[0], probe_ws, leak.resolve())
        (output / "isolation.json").write_text(json.dumps(isolation, indent=1))
        note({"event": "isolation", **isolation})
        if not isolation["ok"]:
            await ctx.proxy.stop()
            raise RuntimeError(f"session isolation check failed: {json.dumps(isolation)[:1500]}")
    note({"event": "prime_trial", "entries": len(entries), "window": ctx.window, "cap": ctx.cap, "served": ctx.served,
          "gate": ctx.gate.snapshot(), "settings": cfg, "hide": ctx.hide})
    gpu = GpuSampler()
    gpu.mark("play")
    stop_files, stopping = stop_paths(output), {}
    live = os.environ.get("ARC_VALIDATION_OUTPUT")
    live = Path(live) / "arena" / "live" if live else None

    def snapshot(final=False):
        cut = min(ctx.inflight.values(), default=None)  # the earliest reply in flight: marks past it are not exact
        roster, clock = read_out(entries, marks, step=trial.get("curve_step") if final else None,
                                 cut=None if final else cut)
        elapsed = time.monotonic() - ctx.started
        last = max(marks, key=marks.get) if marks else None
        reached = [m for m, r in roster["marks"].items() if r["reached"]]
        now_row = {"right": sum(right_outputs(e.current_answer() if e.conv else e.answer, e.task) for e in entries)}
        test_outputs = sum(len(e.task["test"]) for e in entries)
        return {"trial": trial["name"], "kind": "arc2-prime", "elapsed_seconds": round(elapsed, 1),
                "wall_seconds": wall, "gate": ctx.gate.snapshot(), "failures": len(ctx.failures),
                "request_errors": sum(e.stats["errors"] for e in entries), "arm_tokens": {"prime": ctx.spent},
                "completion_tokens": ctx.spent, "tokens_per_second": ctx.spent / max(1.0, elapsed),
                "test_outputs": test_outputs,
                "final": {"score_outputs": 100 * now_row["right"] / max(1, test_outputs), **now_row},
                "roster": roster, "last_mark": last, "marks_reached": reached, "gpu": gpu.summary(),
                "stop_request": stopping or None, "host": host_stats(),
                "sessions_live": sum(e.process is not None and e.process.returncode is None for e in entries),
                "prompt_cached_share": round(sum(x["cached_tokens"] or 0 for e in entries for x in e.replies)
                                             / max(1, sum(x["prompt_tokens"] for e in entries for x in e.replies)), 3),
                "prime": totals(entries)}, clock

    def publish():
        row, _ = snapshot()
        text = json.dumps(row, indent=1)
        (output / "progress.json").write_text(text)
        if live is not None:
            live.mkdir(parents=True, exist_ok=True)
            (live / f"arc2-{trial['name']}.json").write_text(text)

    stagger = float(cfg["start_stagger_seconds"])
    jobs = [asyncio.create_task(e.run(ctx.started + n * stagger)) for n, e in enumerate(entries)]
    end = ctx.started + max(1.0, wall)
    reason = "wall"
    last_publish = 0.0
    try:
        while True:
            now = time.monotonic()
            if now - last_publish >= PROGRESS_SECONDS:
                publish()
                last_publish = now
            if not stopping and any(p.exists() for p in stop_files):
                stopping.update(seen_seconds=round(now - ctx.started, 1), tokens=ctx.spent)
                ctx.gate.close()
                ctx.stopped_at = now
            if ctx.gate.closed and ctx.stopped_at is not None:
                inflight = sum(e.streaming for e in entries)
                if inflight == 0 or now - ctx.stopped_at >= float(cfg["stop_grace_seconds"]):
                    reason = "stop_request" if stopping else "last_mark"
                    break
            if now >= end:
                break
            if ctx.down_since is not None and now - ctx.down_since >= float(cfg["server_dead_seconds"]):
                reason = "server_dead"
                ctx.failures.append(f"model server unanswered {now - ctx.down_since:.0f} s: {ctx.last_failure}")
                break
            if all(j.done() for j in jobs):
                reason = "sessions_ended"
                break
            await asyncio.sleep(1.0)
    finally:
        ctx.ending.set()
        ctx.gate.close()
        for job in jobs:
            job.cancel()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.gather(*jobs, return_exceptions=True), timeout=DRAIN_SECONDS / 2)
        for e in entries:  # a session cancelled mid-stop
            if e.process is not None and e.process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(e.process.pid, signal.SIGKILL)
        await ctx.proxy.stop()
    async with httpx.AsyncClient(base_url=base_url.rstrip("/") + "/", trust_env=False) as client:
        served = await generated_tokens(client)
    report, clock = snapshot(final=True)
    report.update(status="complete", end_reason=reason, runtime=runtime, settings=cfg, window=ctx.window,
                  completion_cap=ctx.cap, streams=ctx.gate.slots, marks=marks, stop_tokens=ctx.stop, copies=copies,
                  task_errors=ctx.failures[:20], play_seconds=round(time.monotonic() - ctx.started, 1),
                  replies=sum(len(e.replies) for e in entries))
    if served is not None and served_before is not None:
        report["server_generated_tokens"] = served - served_before
    answered = sum(e.answer is not None for e in entries)
    healthy = (report["replies"] > 0 and answered > 0 and reason != "server_dead"
               and len(ctx.failures) <= max(3, len(entries) // 20))
    report["status"] = "complete" if healthy else "error"
    if not healthy:
        report["task_errors"] = ctx.failures[:20] or [f"{report['replies']} replies, {answered} entries answered"]
    with (output / "tasks.jsonl").open("w") as stream:
        for e in entries:
            stream.write(json.dumps(e.row(clock)) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=1))
    text = json.dumps(report, indent=1)
    (output / "progress.json").write_text(text)
    if live is not None:
        live.mkdir(parents=True, exist_ok=True)
        (live / f"arc2-{trial['name']}.json").write_text(text)
    for e in entries:
        e.logfile.close()
    ctx.events.close()
    admission.close()
    log.close()
    return report
