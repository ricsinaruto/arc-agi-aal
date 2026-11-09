"""The proposer (docs/rsi.md 1): a second Prime Agent session on the same Flash, with code execution on its own
workspace, and the root side of its requests. Adapted from rsi_loop.proposer (v27) for the arena harness.

Its workspace (``/workspace`` in its namespace; real folder ``/rsi/prop/ws``, user ``rsiprop``):

    AGENTS.md, rsi.py         from the live proposer setup (``meta``)
    meta/                     read-only copy of the live setup
    archive/<id>/             every candidate: tree/, meta.json, scores.json (pool scores per mark only),
                              practice/<run>/ (full traces of its practice plays)
    src/                      read-only: the serving runtime (vLLM, our serving patches)
    work/                     its own: forks, notes, tools (persists across sessions)
    events.jsonl              what the loop tells it

One session at a time; each ends at the setup's ``session_seconds`` or when the agent stops. A self-update must pass
a check session first; three quick failures under it restore the previous setup.
"""
from __future__ import annotations

import asyncio
import json
import os
import pwd
import secrets
import shutil
import signal
import subprocess
import time
from pathlib import Path

from . import candidates as cand
from . import sandbox
from .runtime import RUNTIME

SELF_CHECK = ("Self-check of a new proposer setup: in Python run `import rsi; print(rsi.status()['budget'])`, "
              "then stop.")
QUICK_FAIL_SECONDS = 300
IDLE_END = 60

LOOP_FACTS = """

## Loop facts (added by the loop to every session; not part of your setup)

- The agent is scored in the ARC-AGI-3 competition on Kaggle: 110 hidden game instances, played offline on one
  RTX PRO 6000 within 9 h (~273 s of the GPU per game, ~400K generated tokens per instance at ~1,470 tok/s).
  Per game, a cleared level scores min(115, 100 x (baseline actions / agent actions)^2), levels weighted by their
  number, the game capped at 100; the final score is the mean over games (RHAE). The pool only selects; the loop
  measures the best candidates on held-out games you never see.
- Every pool evaluation plays all pool games at once from fresh starts, on one GPU, and is raced against gen 0's
  per-game scores at three budget marks (13% / 39% / 100% of the competition slice). It stops at a mark where it
  trails gen 0 by more than the noise allows, and is accepted at the last mark only when it beats gen 0 by more than
  twice the noise. The marks, the bars, the pool and the scorer are fixed.
- An evaluation holds the GPU for up to ~2.2 h (29 games x 273 s); your session runs next to it on the same server.
"""


def slim_line(line):
    if b'"message_update"' not in line[:200]:
        return line
    try:
        event = json.loads(line)
    except ValueError:
        return line
    event.pop("message", None)
    inner = event.get("assistantMessageEvent")
    if isinstance(inner, dict):
        inner.pop("partial", None)
        inner.pop("message", None)
    return (json.dumps(event, separators=(",", ":")) + "\n").encode()


async def slim_copy(stream, path):
    with open(path, "ab") as out:
        while True:
            try:
                line = await stream.readline()
            except (ValueError, asyncio.LimitOverrunError):
                line = (await stream.read(1 << 20))[:4096] + b"\n"
            if not line:
                return
            out.write(slim_line(line))
            out.flush()


def session_ended(log_path):
    """The agent's run has ended (rpc mode then idles): its newest agent_end comes after its newest turn."""
    with log_path.open("rb") as handle:
        handle.seek(max(0, log_path.stat().st_size - (1 << 20)))
        tail = handle.read().decode("utf-8", "replace")
    end, start, pending = -1, -1, set()
    for i, line in enumerate(tail.splitlines()):
        if line.startswith('{"type":"agent_end"'):
            end, pending = i, set()
        elif line.startswith(('{"type":"agent_start"', '{"type":"turn_start"', '{"type":"message_start"')):
            start = i
        else:
            for kind in ("compaction", "auto_retry"):
                if line.startswith(f'{{"type":"{kind}_start"'):
                    pending.add(kind)
                elif line.startswith(f'{{"type":"{kind}_end"'):
                    pending.discard(kind)
    return end > start and not pending


EXTRA_FACTS = ""  # set by the loop (two-VM evaluation)


def meta_file(live, name):
    data = (live / name).read_bytes()
    if name != "AGENTS.md":
        return data
    return data.rstrip(b"\n") + b"\n" + LOOP_FACTS.encode() + EXTRA_FACTS.encode()


class ProposerHost:
    def __init__(self, loop):
        self.loop = loop
        self.root = loop.root
        self.ws, self.home = self.root / "prop/ws", self.root / "prop/home"
        self.process = self.session = self.server = None
        self.paused = False
        self.end_requested = False
        self.user = pwd.getpwnam(sandbox.PROPOSER)

    @property
    def state(self):
        return self.loop.state

    async def setup(self):
        meta = self.state["meta"]
        target = self.root / "meta/m0000"
        if not target.exists():
            shutil.copytree(self.loop.source / "rsi/proposer/meta", target)
            self.loop.runtime.seal(target)
        if meta["live"] is None:
            meta["items"]["m0000"] = {"created": time.time(), "status": "live", "parent": None}
            meta["live"] = "m0000"
        subprocess.run(["pkill", "-KILL", "-u", sandbox.PROPOSER], check=False)
        self.ws.mkdir(parents=True, exist_ok=True)
        for part in ("work", ".rsi", "archive", "meta", "src"):
            if (self.ws / part).is_symlink() or ((self.ws / part).exists() and not (self.ws / part).is_dir()):
                cand.remove_entry(self.ws / part)
            (self.ws / part).mkdir(parents=True, exist_ok=True)
        self.home.mkdir(parents=True, exist_ok=True)
        live = self.root / "meta" / meta["live"]
        if not os.path.lexists(self.home / ".prime") and (live / "home").is_dir():
            for item in (live / "home").iterdir():
                cand.remove_entry(self.home / item.name)
                if item.is_dir() and not item.is_symlink():
                    shutil.copytree(item, self.home / item.name, symlinks=True)
                else:
                    shutil.copy2(item, self.home / item.name, follow_symlinks=False)
        self.install_meta_files(live)
        docs = self.loop.source / "rsi/proposer/docs"
        if docs.is_dir():
            cand.remove_entry(self.ws / "docs")
            shutil.copytree(docs, self.ws / "docs")
        for path in (self.ws, self.home):
            sandbox.own(path, sandbox.PROPOSER, 0o700)
        sock = self.ws / ".rsi/sock"
        if os.path.lexists(sock):
            cand.remove_entry(sock)
        self.server = await asyncio.start_unix_server(self.handle, path=str(sock))
        os.chown(sock, self.user.pw_uid, self.user.pw_gid)
        os.chmod(sock, 0o600)

    def install_meta_files(self, live):
        for name in ("AGENTS.md", "rsi.py"):
            if (live / name).is_file() and not (live / name).is_symlink():
                cand.write_untrusted(self.ws, name, meta_file(live, name), owner=(self.user.pw_uid, self.user.pw_gid))

    def settings(self, mid=None):
        live = self.root / "meta" / (mid or self.state["meta"]["live"])
        return json.loads((live / "settings.json").read_text())

    def src_views(self):
        views = []
        if (self.loop.source / "rsi/serving-src").is_dir():
            views.append(("serving", self.loop.source / "rsi/serving-src"))
        site = Path("/tmp/qwen38-flash-next-vllm-runtime/usr/local/lib/python3.12/dist-packages")
        if (site / "vllm").is_dir():
            views.append(("vllm", site / "vllm"))
            views.append(("python-packages", site))
        return views

    def argv(self, mid, key, prompt, user=sandbox.PROPOSER, ws=None, home=None):
        live = self.root / "meta" / mid
        run = json.loads((live / "run.json").read_text())
        argv = [a.replace("{prompt}", prompt) for a in run["argv"]]
        ws, home = ws or self.ws, home or self.home
        env = {"HOME": "/home/session", "USER": user, "LANG": "C.UTF-8",
               "PATH": f"{RUNTIME}/node/bin:{RUNTIME}/propkernel/bin:/usr/local/bin:/usr/bin:/bin",
               "RSI_PROXY_URL": f"http://127.0.0.1:{self.loop.proxy.port}/v1", "RSI_PROXY_KEY": key,
               "RSI_MODEL": self.loop.served, **{str(k): str(v) for k, v in (run.get("env") or {}).items()}}
        binds = [(home, "/home/session", False), (self.root / "archive", "/workspace/archive", True),
                 (live, "/workspace/meta", True)]
        for name, source in self.src_views():
            binds.append((source, f"/workspace/src/{name}", True))
        return sandbox.command(argv, user, ws, env, extra_binds=binds, hide=self.loop.runtime.hidden(),
                               net="proxy", proxy=self.loop.runtime.proxy_access(), tmp_size="16g", shm_size="2g",
                               memory=self.loop.runtime.memory("check" if user == sandbox.CHECK else "proposer"))

    async def start(self, mid=None, prompt=None, wall=None, tag="session", user=sandbox.PROPOSER, ws=None, home=None):
        mid = mid or self.state["meta"]["live"]
        settings = self.settings(mid)
        prompt = prompt or (self.root / "meta" / mid / "prompt.txt").read_text()
        key = secrets.token_hex(16)
        sid = self.loop.new_id("p")
        self.loop.proxy.register(key, sid, "loop", role="proposer")
        log_dir = self.root / "root/proposer" / sid
        log_dir.mkdir(parents=True, exist_ok=True)
        argv = self.argv(mid, key, prompt, user=user, ws=ws, home=home)
        process = await asyncio.create_subprocess_exec(*argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                       stderr=subprocess.STDOUT, start_new_session=True, limit=1 << 26)
        process.copier = asyncio.ensure_future(slim_copy(process.stdout, log_dir / "session.log"))
        run = json.loads((self.root / "meta" / mid / "run.json").read_text())
        if not any("{prompt}" in a for a in run["argv"]):  # rpc mode: the prompt is a command on stdin
            process.stdin.write(json.dumps({"id": "1", "type": "prompt", "message": prompt}).encode() + b"\n")
            await process.stdin.drain()
        record = {"id": sid, "meta": mid, "tag": tag, "started": time.time(), "key": key,
                  "wall": float(wall or settings.get("session_seconds", 7200)), "submitted": 0}
        self.loop.event_log({"event": "rsi2_proposer_start", "id": sid, "meta": mid, "tag": tag})
        return process, record

    def end(self, process, record, reason, user=sandbox.PROPOSER):
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        subprocess.run(["pkill", "-KILL", "-u", user], check=False)
        self.loop.proxy.forget(record.pop("key"))
        totals = self.loop.proxy.session_totals(record["id"])
        record.update(ended=time.time(), reason=reason, exit=process.returncode,
                      tokens=totals["completion_tokens"], requests=totals["requests"], non_flash=totals["non_flash"])
        return record

    async def tick(self, stop=False):
        if self.process is not None:
            record = self.session
            elapsed = time.time() - record["started"]
            log = self.root / "root/proposer" / record["id"] / "session.log"
            size = log.stat().st_size if log.exists() else 0
            if size != record.get("log_size"):
                record["log_size"], record["quiet_since"] = size, time.time()
            proxy = self.loop.proxy
            idle = (time.time() - max(record.get("quiet_since", time.time()),
                                      proxy.last_activity.get(record["id"], 0)) > IDLE_END and size
                    and not proxy.inflight[record["id"]] and session_ended(log))
            if (idle and self.loop.cfg.get("proposer_continue") and record["tag"] == "session" and not stop
                    and not self.paused and not self.end_requested and self.process.returncode is None
                    and elapsed < record["wall"] - 600 and await self.nudge(record, elapsed)):
                return
            reason = "exit" if self.process.returncode is not None else (
                "wall" if elapsed > record["wall"] else "stop" if stop or self.paused else "stopped" if idle else
                "requested" if self.end_requested else None)
            if reason:
                self.end_requested = False
                done = self.end(self.process, record, reason)
                self.state["proposer"]["sessions"].append(done)
                self.loop.event_log({"event": "rsi2_proposer_session", **done})
                self.process = self.session = None
                self.check_rollback()
                self.loop.save("proposer")
            return
        if stop or self.paused:
            return
        self.process, self.session = await self.start()

    async def nudge(self, record, elapsed):
        """(1a, 17 Oct) The agent ended its run with session time left: one more prompt instead of ending the session.
        Not after three nudges in a row that each produced under 2K generated tokens (a broken tool loop)."""
        tokens = self.loop.proxy.session_totals(record["id"])["completion_tokens"]
        history = record.setdefault("nudges", [])
        if len(history) >= 3 and all(b - a < 2000 for a, b in zip(history[-4:-1], history[-3:])):
            return False
        history.append(tokens)
        left = round((record["wall"] - elapsed) / 60)
        message = (f"Your session is not over: about {left} minutes remain. Continue the work in /workspace/AGENTS.md "
                   "(rsi.status(), your notes, the next candidate or practice play). If a tool keeps failing, write "
                   "what fails in /workspace/work/notes.md and work around it. Call rsi.end_session() only when you "
                   "have nothing useful left to do in this session.")
        try:
            self.process.stdin.write(json.dumps({"id": f"n{len(history)}", "type": "prompt",
                                                 "message": message}).encode() + b"\n")
            await self.process.stdin.drain()
        except (ConnectionError, RuntimeError, AttributeError):
            return False
        record["quiet_since"] = time.time()
        self.loop.event_log({"event": "rsi2_proposer_nudge", "id": record["id"], "n": len(history),
                             "minutes_left": left, "tokens": tokens})
        return True

    async def stop(self):
        self.paused = True
        if self.process is not None:
            await self.tick(stop=True)
        if self.server is not None:
            self.server.close()

    def check_rollback(self):
        meta = self.state["meta"]
        live = meta["live"]
        recent = [s for s in self.state["proposer"]["sessions"] if s["meta"] == live and s["tag"] == "session"
                  and s.get("reason") not in ("stop", "requested")][-3:]
        if len(recent) < 3 or meta["items"][live].get("parent") is None:
            return
        if all(s["ended"] - s["started"] < QUICK_FAIL_SECONDS and not s.get("submitted") for s in recent):
            previous = meta["items"][live]["parent"]
            meta["items"][live]["status"] = "rolled back"
            meta["live"] = previous
            self.install_meta_files(self.root / "meta" / previous)
            self.notify({"event": "self_update_rolled_back", "from": live, "to": previous})
            self.loop.event_log({"event": "rsi2_meta_rollback", "from": live, "to": previous})

    def notify(self, row):
        try:
            cand.write_untrusted(self.ws, "events.jsonl", (json.dumps({"time": time.time(), **row}) + "\n").encode(),
                                 owner=(self.user.pw_uid, self.user.pw_gid), append=True)
        except (OSError, ValueError) as exc:
            self.loop.event_log({"event": "rsi2_notify_failed", "error": f"{type(exc).__name__}: {exc}"[:200]})

    # ----- requests ----------------------------------------------------------------------------------------------
    async def handle(self, reader, writer):
        try:
            request = json.loads(await reader.readline())
            op = request.pop("op")
            handler = getattr(self, f"op_{op}", None)
            reply = ({"ok": False, "error": f"unknown op {op}"} if handler is None
                     else {"ok": True, "value": await handler(**request)})
        except Exception as exc:  # noqa: BLE001 - the error goes back to the proposer
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
        try:
            writer.write(json.dumps(reply, default=str).encode() + b"\n")
            await writer.drain()
            writer.close()
        except ConnectionError:
            pass

    def work_rel(self, path):
        path = str(path)
        if not path.startswith("/workspace/work/"):
            raise ValueError("path: a folder under /workspace/work")
        rel = path.removeprefix("/workspace/").rstrip("/")
        cand._parts(rel)
        return rel

    def work(self, name):
        if not isinstance(name, str) or not name or "/" in name or name.startswith("."):
            raise ValueError("name: a plain folder name under /workspace/work")
        return self.ws / "work" / name

    async def op_status(self):
        return self.loop.status_view()

    async def op_fork(self, parent, name):
        if parent not in self.state["candidates"] or not self.loop.candidate_dir(parent).is_dir():
            raise ValueError(f"no candidate tree {parent}")
        self.work(name)
        staging = self.root / "prop/.staging" / secrets.token_hex(6)
        staging.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(staging.parent, 0o700)
        try:
            await asyncio.to_thread(cand.copy_tree, self.loop.candidate_dir(parent), staging, link_big=True,
                                    owner=(self.user.pw_uid, self.user.pw_gid))
            (staging / ".rsi-parent").write_text(parent + "\n")
            await asyncio.to_thread(sandbox.own, staging, sandbox.PROPOSER, 0o700)
            cand.move_into(staging, self.ws, "work", name)
        except (OSError, ValueError) as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise ValueError(f"/workspace/work/{name}: {exc}") from None
        return {"path": f"/workspace/work/{name}"}

    async def op_submit(self, name, note=""):
        self.work(name)
        cid = self.loop.new_id("c")
        target = self.loop.candidate_dir(cid)
        try:
            await asyncio.to_thread(cand.copy_untrusted, self.ws, f"work/{name}", target, link_root_readonly=True)
        except (OSError, ValueError) as exc:
            shutil.rmtree(target.parent, ignore_errors=True)
            raise ValueError(f"/workspace/work/{name}: a plain folder, not a link ({type(exc).__name__})") from None
        links = cand.escaping_links(target)
        if links:
            shutil.rmtree(target.parent, ignore_errors=True)
            raise ValueError(f"symbolic links out of the candidate: {links[:5]} (links may point only inside it)")
        self.loop.runtime.seal(target.parent)
        parent = None
        if (target / ".rsi-parent").is_file() and not (target / ".rsi-parent").is_symlink():
            parent = cand.read_inside(target, ".rsi-parent").strip()
            parent = parent if parent in self.state["candidates"] else None
        item = self.loop.register_candidate(cid, target, parent or "c0000", str(note), "proposer",
                                            meta=self.state["meta"]["live"])
        if self.session is not None:
            self.session["submitted"] = self.session.get("submitted", 0) + 1
        self.loop.save("submit")
        return {"candidate": cid, "status": item["status"], "problems": item["problems"],
                "queue_position": self.loop.queue_position(cid)}

    async def op_withdraw(self, candidate=None, job=None):
        """Remove queued (not yet running) work: one job by id (``job``, practice or evaluation), or all of a
        candidate's queued evaluation and practice (``candidate``)."""
        return self.loop.withdraw(candidate, job)

    async def op_practice(self, candidate, games=None, n=4, seconds=None):
        return self.loop.request_practice(candidate, games, n, seconds)

    async def op_end_session(self):
        self.end_requested = True
        return {"ending": True, "next_setup": self.state["meta"]["live"]}

    async def op_meta_history(self, diff_lines=400):
        import difflib

        out = []
        items = self.state["meta"]["items"]
        for mid, item in sorted(items.items(), key=lambda kv: kv[1].get("created", 0)):
            parent, changed, budget = item.get("parent"), {}, max(0, min(int(diff_lines), 5000))
            here, there = self.root / "meta" / mid, (self.root / "meta" / parent) if parent else None
            if there is not None and here.is_dir() and there.is_dir():
                names = sorted({p.relative_to(here).as_posix() for p in here.rglob("*") if p.is_file()}
                               | {p.relative_to(there).as_posix() for p in there.rglob("*") if p.is_file()})
                for name in names:
                    a = (there / name).read_text(errors="replace").splitlines() if (there / name).is_file() else []
                    b = (here / name).read_text(errors="replace").splitlines() if (here / name).is_file() else []
                    if a != b:
                        diff = list(difflib.unified_diff(a, b, f"{parent}/{name}", f"{mid}/{name}", lineterm="", n=1))
                        changed[name] = "\n".join(diff[:budget])
                        budget = max(0, budget - len(diff))
            out.append({"meta": mid, "parent": parent, "status": item.get("status"), "note": item.get("note", ""),
                        "changed": changed,
                        "candidates": {c: {"status": v["status"], "verdict": v.get("verdict")}
                                       for c, v in self.state["candidates"].items() if v.get("meta") == mid}})
        return out

    async def op_self_update(self, path, note=""):
        rel = self.work_rel(path)
        mid = self.loop.new_id("m")
        target = self.root / "meta" / mid
        required = ("AGENTS.md", "prompt.txt", "settings.json", "run.json")

        def refuse(why):
            shutil.rmtree(target, ignore_errors=True)
            raise ValueError(why)

        try:
            await asyncio.to_thread(cand.copy_untrusted, self.ws, rel, target)
        except (OSError, ValueError) as exc:
            refuse(f"path: a plain folder under /workspace/work, not a link ({type(exc).__name__})")
        missing = [r for r in required if not (target / r).is_file() or (target / r).is_symlink()]
        if missing:
            refuse(f"missing (or links): {missing}")
        if cand.escaping_links(target):
            refuse("no symbolic links out of the setup")
        try:
            run = json.loads((target / "run.json").read_text())
            settings = json.loads((target / "settings.json").read_text())
            if not (isinstance(run, dict) and isinstance(run.get("argv"), list) and run["argv"]
                    and all(isinstance(a, str) for a in run["argv"]) and isinstance(run.get("env", {}), dict)
                    and isinstance(settings, dict)):
                raise ValueError("run.json needs argv (strings) and env (object); settings.json an object")
        except (OSError, ValueError) as exc:
            refuse(f"{type(exc).__name__}: {exc}"[:300])
        self.loop.runtime.seal(target)
        meta = self.state["meta"]
        meta["items"][mid] = {"created": time.time(), "status": "checking", "parent": meta["live"],
                              "note": str(note)[:2000]}
        task = asyncio.create_task(self.self_check(mid))
        self.tasks = getattr(self, "tasks", set())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return {"meta": mid, "status": "checking: a short session under the new setup; see events.jsonl"}

    async def self_check(self, mid):
        check = self.root / "prop/check" / mid
        shutil.rmtree(check, ignore_errors=True)
        ws, home = check / "ws", check / "home"
        for part in (".rsi", "work", "archive", "meta", "src"):
            (ws / part).mkdir(parents=True, exist_ok=True)
        source = self.root / "meta" / mid
        for name in ("AGENTS.md", "rsi.py"):
            if (source / name).is_file() and not (source / name).is_symlink():
                (ws / name).write_bytes(meta_file(source, name))
        if (source / "home").is_dir():
            shutil.copytree(source / "home", home, dirs_exist_ok=True, symlinks=True)
        home.mkdir(parents=True, exist_ok=True)
        # the model provider config is loop plumbing (proxy URL and key come from the env): a setup without it
        # gets gen 0's, so the check tests the setup, not whether its copy kept a hidden file
        provider = Path(".prime/agent/models.json")
        if not (home / provider).is_file():
            for origin in (self.root / "meta/m0000/home", self.home):
                if (origin / provider).is_file() and not (origin / provider).is_symlink():
                    (home / provider).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(origin / provider, home / provider)
                    break
        for path in (ws, home):
            sandbox.own(path, sandbox.CHECK, 0o700)
        os.chmod(check, 0o711)
        calls = []

        async def handle(reader, writer):
            try:
                request = json.loads(await reader.readline())
                if request.get("op") == "status":
                    calls.append(time.time())
                    reply = {"ok": True, "value": await self.op_status()}
                else:
                    reply = {"ok": False, "error": "self-check session: only status"}
            except Exception as exc:  # noqa: BLE001
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
            writer.write(json.dumps(reply, default=str).encode() + b"\n")
            await writer.drain()
            writer.close()

        sock = ws / ".rsi/sock"
        server = await asyncio.start_unix_server(handle, path=str(sock))
        entry = pwd.getpwnam(sandbox.CHECK)
        os.chown(sock, entry.pw_uid, entry.pw_gid)
        os.chmod(sock, 0o600)
        process, record = await self.start(mid, prompt=SELF_CHECK, wall=900, tag="self-check", user=sandbox.CHECK,
                                           ws=ws, home=home)
        try:
            while process.returncode is None and time.time() - record["started"] < record["wall"]:
                await asyncio.sleep(5)
                if calls and self.loop.proxy.session_totals(record["id"])["requests"] > 0:
                    break
        finally:
            done = self.end(process, record, "check", user=sandbox.CHECK)
            server.close()
        ok = bool(calls) and done["requests"] > 0
        done["check"] = {"status_calls": len(calls), "model_requests": done["requests"]}
        self.state["proposer"]["sessions"].append(done)
        meta = self.state["meta"]
        meta["items"][mid]["status"] = "live" if ok else "failed check"
        if ok:
            meta["items"][meta["live"]]["status"] = "previous"
            meta["live"] = mid
            self.install_meta_files(self.root / "meta" / mid)
        why = None
        if not ok:
            try:
                tail = (self.root / "root/proposer" / record["id"] / "session.log").read_bytes()[-1500:]
                why = tail.decode("utf-8", "replace")
            except OSError:
                why = None
        self.notify({"event": "self_update", "meta": mid, "ok": ok, "check": done["check"],
                     **({"session_log_tail": why} if why is not None else {})})
        self.loop.event_log({"event": "rsi2_self_update", "meta": mid, "ok": ok})
        self.loop.save("self_update")
