"""The proposer (docs/rsi.md 1): a second Prime Agent session on the same Flash, with code execution on its own
workspace, and the root side of its requests.

Its workspace (``/workspace`` in its namespace; real folder ``/rsi/prop/ws``, user ``rsiprop``):

    AGENTS.md, rsi.py         from the live proposer setup (``meta``)
    meta/                     read-only copy of the live setup (guidance, prompt, policy.py, settings.json, run.json)
    archive/<id>/             every candidate: tree/, meta.json, scores.json (pool scores only), practice/ (traces)
    src/                      read-only: the serving sources (vLLM, our serving patches) and the base weights
    work/                     its own: forks, notes, scratch (persists across sessions)
    events.jsonl              what the loop tells it (switch results, rejections, finished evaluations)

Everything else of the loop is hidden from its namespace. One session at a time; each ends at the setup's
``session_seconds`` or when the agent stops, and the next one starts fresh (its notes and HOME persist). A setup
installed by ``self_update`` must pass a check session first; if the next sessions under it keep failing, the previous
setup comes back (rollback).
"""
from __future__ import annotations

import asyncio
import json
import os
import pwd
import re
import secrets
import shutil
import signal
import subprocess
import time
from pathlib import Path

from . import candidates as cand
from . import sandbox
from .attempt import IDLE_END, RUNTIME, Attempt, slim_copy
from .feedback import LOOP_FACTS, feedback, meta_history, server_metrics

SELF_CHECK = ("Self-check of a new proposer setup: in Python run `import rsi; print(rsi.status()['budget'])`, "
              "then stop.")
QUICK_FAIL_SECONDS = 300


# no advice on how much to change per candidate (the inherited gen 0 sentence is replaced on install)
ONE_CHANGE = re.compile(rb"A\s+candidate with one change tells you what that change is worth;\s+several changes at once"
                        rb"\s+are allowed\.")
NEUTRAL = (b"How much to change per candidate, and when, is your call: fewer changes are easier to attribute, more "
           b"move faster.")


def meta_file(live, name):
    """A setup file as the session gets it: AGENTS.md with the loop's fixed facts after it."""
    data = (live / name).read_bytes()
    if name != "AGENTS.md":
        return data
    data = ONE_CHANGE.sub(NEUTRAL, data)
    return data.rstrip(b"\n") + b"\n\n" + LOOP_FACTS.lstrip("\n").encode()


class ProposerHost:
    def __init__(self, loop):
        self.loop = loop
        self.root = loop.root
        self.ws, self.home = self.root / "prop/ws", self.root / "prop/home"
        self.process = None
        self.session = None
        self.paused = False
        self.server = None
        self.user = pwd.getpwnam(sandbox.PROPOSER)

    @property
    def state(self):
        return self.loop.state

    # ----- setup -----------------------------------------------------------------------------------------------
    async def setup(self):
        meta = self.state["meta"]
        if meta["live"] is None:
            target = self.root / "meta/m0000"
            if not target.exists():
                shutil.copytree(self.loop.source / "rsi/gen0/meta", target)
            self.loop.runtime.seal(target)
            meta["items"]["m0000"] = {"created": time.time(), "status": "live", "parent": None}
            meta["live"] = "m0000"
        # nothing of the proposer runs now (a leftover process would race the root's writes below); the folders may
        # hold links it planted in an earlier leg: the root replaces whatever is at a path before writing there
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
        docs = self.loop.source / "rsi/gen0/docs"
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

    # ----- sessions ----------------------------------------------------------------------------------------------
    def argv(self, mid, key, prompt, user=sandbox.PROPOSER, ws=None, home=None, program=None):
        live = self.root / "meta" / mid
        run = json.loads((live / "run.json").read_text())
        agent = self.loop.candidate_dir(run.get("agent_from", "c0000")) / "player"
        argv = program or [a.replace("{prompt}", prompt) for a in run["argv"]]
        ws, home = ws or self.ws, home or self.home
        env = {"HOME": "/home/session", "USER": user, "LANG": "C.UTF-8",
               "PATH": f"{RUNTIME}/node/bin:{RUNTIME}/propkernel/bin:/usr/local/bin:/usr/bin:/bin",
               "RSI_PROXY_URL": f"http://127.0.0.1:{self.loop.proxy.port}/v1", "RSI_PROXY_KEY": key,
               "RSI_MODEL": self.loop.served, **{str(k): str(v) for k, v in (run.get("env") or {}).items()}}
        binds = [(home, "/home/session", False), (agent, "/candidate", True),
                 (self.root / "archive", "/workspace/archive", True), (live, "/workspace/meta", True)]
        for name, source in self.src_views():
            binds.append((source, f"/workspace/src/{name}", True))
        return sandbox.command(argv, user, ws, env, extra_binds=binds, hide=self.loop.runtime.hidden(),
                               net="proxy", proxy=self.loop.runtime.proxy_access(), tmp_size="16g", shm_size="2g",
                               memory=self.loop.runtime.memory("check" if user == sandbox.CHECK else "proposer",
                                                               self.loop.concurrency()))

    def src_views(self):
        views = []
        serving_src = self.loop.source / "rsi/serving-src"
        if serving_src.is_dir():
            views.append(("serving", serving_src))
        # the full installed serving runtime (vLLM and every package it runs with), as gen 0's root start unpacked
        # it; a proposer-built stack runs the same files, with its overlay in front module by module
        site = Path("/tmp/qwen38-flash-next-vllm-runtime/usr/local/lib/python3.12/dist-packages")
        if (site / "vllm").is_dir():
            views.append(("vllm", site / "vllm"))
            views.append(("python-packages", site))
        identity = Path(self.loop.stacks.server_dir) / "vllm-server-identity.json"
        if getattr(self, "weights", None) is None and identity.is_file() and not identity.is_symlink() and \
                identity.stat().st_uid == 0:  # written by gen 0's root start only (never a proposer stack's file)
            argv = json.loads(identity.read_text()).get("argv", [])
            self.weights = next((a for a in argv if a.startswith("/") and Path(a, "config.json").exists()), None)
        if getattr(self, "weights", None):
            views.append(("weights", Path(self.weights)))
        return views

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
                    and not proxy.inflight[record["id"]]
                    and Attempt._ended(log))  # rpc mode idles after agent_end: the session is over
            reason = "exit" if self.process.returncode is not None else (
                "wall" if elapsed > record["wall"] else "stop" if stop or self.paused else "stopped" if idle else
                "requested" if getattr(self, "end_requested", False) else None)
            if reason:
                self.end_requested = False
                done = self.end(self.process, record, reason)
                self.state["proposer"]["sessions"].append(done)
                self.loop.event_log({"event": "rsi_proposer_session", **done})
                self.process = self.session = None
                if reason != "stop":  # a session the loop paused says nothing about the stack
                    self.loop.probation_note(session=done)
                self.check_rollback()
                self.loop.save("proposer")
            return
        if stop or self.paused or self.state.get("switching"):
            return
        self.process, self.session = await self.start()

    async def pause(self):
        self.paused = True
        await self.tick()

    def resume(self):
        self.paused = False

    async def stop(self):
        self.paused = True
        if self.process is not None:
            await self.tick(stop=True)
        if self.server is not None:
            self.server.close()

    def check_rollback(self):
        """Three quick failures in a row (exit or crash within QUICK_FAIL_SECONDS with no submission) under a
        self-updated setup restore the previous one."""
        meta = self.state["meta"]
        live = meta["live"]
        recent = [s for s in self.state["proposer"]["sessions"] if s["meta"] == live and s["tag"] == "session"
                  and s.get("reason") not in ("stop", "requested")][-3:]  # paused or ended on request: no failure
        if len(recent) < 3 or meta["items"][live].get("parent") is None:
            return
        if all(s["ended"] - s["started"] < QUICK_FAIL_SECONDS and not s.get("submitted") for s in recent):
            previous = meta["items"][live]["parent"]
            meta["items"][live]["status"] = "rolled back"
            meta["live"] = previous
            self.install_meta_files(self.root / "meta" / previous)
            self.notify({"event": "self_update_rolled_back", "from": live, "to": previous})
            self.loop.event_log({"event": "rsi_meta_rollback", "from": live, "to": previous})

    def notify(self, row):
        try:
            cand.write_untrusted(self.ws, "events.jsonl", (json.dumps({"time": time.time(), **row}) + "\n").encode(),
                                 owner=(self.user.pw_uid, self.user.pw_gid), append=True)
        except (OSError, ValueError) as exc:
            self.loop.event_log({"event": "rsi_notify_failed", "error": f"{type(exc).__name__}: {exc}"[:200]})

    # ----- policy and GPU jobs (run as rsiprop) -----------------------------------------------------------------
    async def policy(self, view, meta_dir=None):
        """``(decision, error)`` from ``meta_dir``'s (default: the live setup's) policy.py, run as rsiprop without
        network; the decision is validated by the loop (``check_decision``)."""
        folder = meta_dir or self.root / "meta" / self.state["meta"]["live"]
        if not (folder / "policy.py").is_file():
            return None, "no policy.py"
        argv = sandbox.command([f"{RUNTIME}/kernel/bin/python", "-I", "/workspace/meta/policy.py"], sandbox.PROPOSER,
                               self.ws, {"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"},
                               extra_binds=[(folder, "/workspace/meta", True)], hide=self.loop.runtime.hidden(),
                               net="none", memory=self.loop.runtime.memory("policy"))
        process = None
        try:
            process = await asyncio.create_subprocess_exec(*argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                           stderr=subprocess.PIPE, start_new_session=True)
            out, err = await asyncio.wait_for(process.communicate(json.dumps(view).encode()), timeout=120)
            if process.returncode != 0:
                return None, err.decode(errors="replace")[-500:]
            lines = out.decode(errors="replace").strip().splitlines()
            if not lines:
                return None, "policy printed nothing"
            return json.loads(lines[-1]), None
        except Exception as exc:  # noqa: BLE001
            if process is not None and process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return None, f"{type(exc).__name__}: {exc}"[:500]

    async def test_policy(self, folder):
        """A self-update's policy.py on the loop's current view: it must give a valid decision."""
        decision, error = await self.policy(self.loop.view(), meta_dir=folder)
        if error:
            return error[-300:]
        try:
            self.loop.check_decision(decision, self.loop.view())
        except ValueError as exc:
            return str(exc)[:300]
        return None

    async def gpu_job(self, spec):
        """The proposer's Python script with the GPU free (the server is stopped), as rsiprop, no network,
        time-capped, under ``gpu_guard`` (no gradient training). ``spec``: {"script": "/workspace/work/.../job.py",
        "args": [...], "minutes": N}. PyTorch and the rest of the serving runtime's packages are on its path."""
        script, args = str(spec.get("script") or ""), spec.get("args") or []
        minutes = min(float(spec.get("minutes", 30)), float(self.settings().get("gpu_job_max_minutes", 120)))
        if not script.startswith("/workspace/work/") or not script.endswith(".py") or not (
                isinstance(args, list) and all(isinstance(a, str) for a in args)):
            return {"ok": False, "error": "gpu_job needs script (a .py file under /workspace/work) and args (strings)"}
        rel = script.removeprefix("/workspace/")
        try:
            body = cand.read_untrusted(self.ws, rel)  # kept for the audit, read without following a link
        except (OSError, ValueError) as exc:
            return {"ok": False, "error": f"gpu_job script: {type(exc).__name__}: {exc}"[:300]}
        site = Path("/tmp/qwen38-flash-next-vllm-runtime/usr/local")  # gen 0's runtime, unpacked by the root start
        env = {"HOME": "/home/session", "PATH": f"{RUNTIME}/kernel/bin:/usr/local/bin:/usr/bin:/bin",
               "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
               "RSI_GPU_SITE": "/gpu/usr-local/lib/python3.12/dist-packages",
               "LD_LIBRARY_PATH": "/gpu/usr-local/cuda-13.0/lib64:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu"}
        binds = [(self.home, "/home/session", False), (self.root / "archive", "/workspace/archive", True),
                 (RUNTIME / "gpujob", "/gpu/guard", True)]
        if site.is_dir():
            binds.append((site, "/gpu/usr-local", True))
        for name, source in self.src_views():
            binds.append((source, f"/workspace/src/{name}", True))
        python = shutil.which("python3.12") or "/usr/bin/python3"
        argv = sandbox.command([python, "-I", "/gpu/guard/gpu_guard.py", script, *args], sandbox.PROPOSER, self.ws,
                               env, cwd=str(Path(script).parent), extra_binds=binds, hide=self.loop.runtime.hidden(),
                               net="none", shm_size="16g", tmp_size="32g",
                               memory=self.loop.runtime.memory("gpu_job"))
        stamp = f"{int(time.time())}-{secrets.token_hex(3)}"
        log = self.root / "root/proposer/gpu-jobs" / f"{stamp}.log"  # root's file; a copy goes to the workspace
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with log.open("ab") as stream:
            process = await asyncio.create_subprocess_exec(*argv, stdout=stream, stderr=subprocess.STDOUT,
                                                           start_new_session=True)
            try:
                await asyncio.wait_for(process.wait(), timeout=minutes * 60)
            except asyncio.TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        subprocess.run(["pkill", "-KILL", "-u", sandbox.PROPOSER], check=False)  # nothing of the job outlives it
        name = f"work/gpu-job-{stamp}.log"
        try:
            with log.open("rb") as handle:
                cand.write_untrusted(self.ws, name, handle.read(64 * 1024 ** 2),
                                     owner=(self.user.pw_uid, self.user.pw_gid))
        except (OSError, ValueError):
            pass
        audit = self.loop.ckpt / "gpu-jobs" / stamp  # the script and log, for the no-training audit
        audit.mkdir(parents=True, exist_ok=True)
        (audit / "script.py").write_bytes(body)
        shutil.copy2(log, audit / "job.log")
        (audit / "spec.json").write_text(json.dumps({"script": script, "args": args, "minutes": minutes,
                                                      "exit": process.returncode}))
        return {"ok": process.returncode == 0, "exit": process.returncode, "seconds": round(time.time() - started),
                "log": f"/workspace/{name}"}

    # ----- requests ----------------------------------------------------------------------------------------------
    async def handle(self, reader, writer):
        try:
            line = await reader.readline()
            request = json.loads(line)
            op = request.pop("op")
            handler = getattr(self, f"op_{op}", None)
            if handler is None:
                reply = {"ok": False, "error": f"unknown op {op}"}
            else:
                reply = {"ok": True, "value": await handler(**request)}
        except Exception as exc:  # noqa: BLE001 - the error goes back to the proposer
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
        try:
            writer.write(json.dumps(reply, default=str).encode() + b"\n")
            await writer.drain()
            writer.close()
        except ConnectionError:
            pass  # the session ended while its request ran

    def work_rel(self, path):
        """``/workspace/work/...`` as a path relative to the workspace (no ``..``; resolved by descriptors later)."""
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
        loop = self.loop
        from .policy import standing

        cands = {c: v for c, v in self.state["candidates"].items() if v["pool"]}
        stand = standing({c: v for c, v in cands.items() if v["status"] not in ("rejected", "failed")})
        m = self.state["meters"]
        budget = {"loop_gpu_hours": round(m["loop_s"] / 3600, 3),
                  "approved_loop_gpu_hours": loop.cfg.get("budget_loop_gpu_hours"),
                  "remaining_loop_gpu_hours": round(float(loop.cfg.get("budget_loop_gpu_hours") or 0)
                                                    - m["loop_s"] / 3600, 3),
                  "scoring_gpus": 1 + len(loop.fresh_scorers()),
                  "scoring_vm_gpu_hours": round(m.get("remote_s", 0.0) / 3600, 3),
                  "round_cap_gpu_hours": loop.cfg.get("round_cap_gpu_hours"),
                  "this_machine_leg_ends_in_hours": round((loop.deadline - time.time()) / 3600, 2),
                  "deadline_in_hours": round((loop.deadline - time.time()) / 3600, 2),
                  "ram_ceiling_gib_per_player": round((loop.runtime.memory("player", loop.concurrency()) or 0)
                                                      / 2 ** 30, 1)}
        return {"budget": budget, "live_stack": self.state["stack"]["live"],
                "suggested_parent": self.state.get("suggested_parent"),
                "queue": {k: sum(1 for j in self.state["jobs"] if j["kind"] == k)
                          for k in ("smoke", "pool", "practice")},
                "running_games": sum(1 for j, _ in loop.running.values() if not j["kind"].startswith("heldout")),
                "slots": loop.scoring_slots(), "slots_this_gpu": loop.slots(), "pending_switch": self.state.get("pending_switch"),
                "candidates": {c: {"status": v["status"], "parent": v["parent"], "note": v["note"][:200],
                                   "problems": v.get("problems"), "pool_games": len(v["pool"]),
                                   "effect": stand.get(c, {}).get("effect"), "se": stand.get(c, {}).get("se"),
                                   "raw_mean": stand.get(c, {}).get("raw_mean"),
                                   "live": v["serving_hash"] == self.state["stack"]["live_hash"],
                                   "stack_reverted": loop.stack_reverted(c),
                                   "fixed_scored": sum(1 for g in loop.fixed_pool if g in v["pool"])}
                               for c, v in self.state["candidates"].items()},
                "pool_games": self.state["pool_games"], "practice_games": self.state["practice_games"],
                "base_pool_games": list(loop.base_pool), "policy_best": self.state.get("policy_best"),
                "fixed_pool_games": list(loop.fixed_pool),
                "policy_fallback": self.state.get("policy_fallback"),  # your policy failed; gen 0's decided
                "stack_probation": self.state["stack"].get("probation"),
                "practice_rates": {g: [round(r["levels"] / max(1, r["total_levels"]), 3) for r in rows]
                                   for g, rows in self.state["practice"].items()},
                "meta_live": self.state["meta"]["live"], "concurrency_now": loop.concurrency(),
                "unprivileged_serving": self.state.get("unprivileged_serving"),
                "time": {**feedback(self.state, loop.cfg, loop.concurrency(), loop.scoring_slots(),
                                    float(self.settings().get("session_seconds", 7200)), time.time()),
                         "server_now": await self.server_now()}}

    async def server_now(self):
        """The serving stack's own counters now: running/waiting requests, KV use and capacity, preemptions, and
        output tok/s since the previous status call (60 s to 1 h apart)."""
        try:
            async with self.loop.proxy.client.get(self.loop.proxy.upstream + "/metrics", timeout=5) as response:
                now = server_metrics(await response.text())
        except Exception as exc:  # noqa: BLE001 - a status without the server's view
            return {"error": type(exc).__name__}
        now["t"] = time.time()
        base = getattr(self, "metrics_base", None)  # the sample the recent rate is measured from
        if base is None or now["t"] - base["t"] > 3600 or now.get("generation_tokens_total") is None:
            self.metrics_base = now
        elif now["t"] - base["t"] >= 60:
            self.recent_rate = round((now["generation_tokens_total"] - base["generation_tokens_total"])
                                     / (now["t"] - base["t"]), 1)
            self.metrics_base = now
        if getattr(self, "recent_rate", None) is not None:
            now["output_tok_per_s_recent"] = self.recent_rate
        if self.loop.proxy.gate is not None:
            now["gate"] = self.loop.proxy.gate.view()
        return {k: v for k, v in now.items() if k != "t"}

    async def op_fork(self, parent, name):
        if parent not in self.state["candidates"] or not self.loop.candidate_dir(parent).is_dir():
            raise ValueError(f"no candidate tree {parent}")
        self.work(name)
        # built in a root-only staging folder, then renamed into work/ (reached without following a link)
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

    async def op_release(self, candidate):
        """A candidate submitted with ``hold=True`` becomes eligible for pool evaluation."""
        item = self.state["candidates"].get(candidate)
        if item is None or item["status"] != "held":
            raise ValueError(f"{candidate} is not a held candidate")
        item["status"] = "new"
        await self.loop.run_policy()
        self.loop.save("release")
        return {"candidate": candidate, "status": item["status"]}

    async def op_submit(self, name, note="", hold=False):
        self.work(name)
        if not isinstance(hold, bool):
            raise ValueError("hold: true or false")
        cid = self.loop.new_id("c")
        target = self.loop.candidate_dir(cid)
        # copied through descriptors only (no link followed, a top-level link refused); everything below is checked on
        # the root-owned copy, which nothing can change any more
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
        item = self.loop.register_candidate(cid, target, parent or self.state.get("suggested_parent"), str(note),
                                            "proposer", hold=hold)
        if self.session is not None:
            self.session["submitted"] = self.session.get("submitted", 0) + 1
        self.loop.save("submit")
        return {"candidate": cid, "status": item["status"], "problems": item["problems"],
                "live": item["serving_hash"] == self.state["stack"]["live_hash"]}

    async def op_adopt_stack(self, candidate, gpu_job=None):
        item = self.state["candidates"].get(candidate)
        if item is None or item["status"] == "rejected":
            raise ValueError(f"{candidate} is not an archived, accepted candidate")
        if self.state.get("pending_switch") or self.state["stack"].get("return_to"):
            raise ValueError("the stack is busy (a switch is pending); retry later")
        if item["serving_hash"] == self.state["stack"]["live_hash"]:
            raise ValueError(f"{candidate}'s serving/ is the live stack already")
        if gpu_job is not None and not (isinstance(gpu_job, dict) and str(gpu_job.get("script", "")).startswith(
                "/workspace/work/") and str(gpu_job["script"]).endswith(".py")):
            raise ValueError('gpu_job: {"script": "/workspace/work/.../job.py", "args": [...], "minutes": N}')
        self.loop.request_switch(candidate, gpu_job)
        return {"queued": True, "drain_seconds": self.loop.cfg["drain_seconds"]}

    async def op_practice(self, candidate, games=None, n=2, seconds=None):
        if candidate not in self.state["candidates"]:
            raise ValueError(f"no candidate {candidate}")
        if seconds is not None and (not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
                                    or not 60 <= seconds <= 1e6):
            raise ValueError("seconds: a number, at least 60 (capped at the candidate's game wall)")
        from .policy import practice_ranking
        import random

        known = self.state["practice_games"]
        if games:
            unknown = [g for g in games if g not in known]
            if unknown:
                raise ValueError(f"not practice games: {unknown[:5]}")
        else:
            games = practice_ranking(self.state["practice"], known, random.Random(self.state["seq"]))[:max(1, int(n))]
        extra = {"seconds": float(seconds)} if seconds is not None else {}
        jobs = [self.loop.enqueue({"kind": "practice", "candidate": candidate, "game": g, "seed": 0, **extra})["id"]
                for g in games[:20]]
        self.loop.save("practice")
        return {"queued": jobs, "games": games[:20]}

    async def op_add_game(self, path, role="practice"):
        return await self.loop.runtime_add_game(self, path, role)

    async def op_end_session(self):
        """End this session now (after this reply); the next starts at once under the live setup."""
        self.end_requested = True
        return {"ending": True, "next_setup": self.state["meta"]["live"]}

    async def op_meta_history(self, diff_lines=400):
        """Every setup version: parent, note, status, files changed against its parent (unified diff), its sessions
        and the candidates submitted under it with their standing."""
        return meta_history(self.root / "meta", self.state, int(diff_lines))

    async def op_self_update(self, path, note=""):
        rel = self.work_rel(path)
        mid = self.loop.new_id("m")
        target = self.root / "meta" / mid
        required = ("AGENTS.md", "prompt.txt", "policy.py", "settings.json", "run.json")

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
        links = cand.escaping_links(target)
        if links:
            refuse(f"no symbolic links out of the setup: {links[:5]}")
        try:
            run = json.loads((target / "run.json").read_text())
            settings = json.loads((target / "settings.json").read_text())
            if not (isinstance(run, dict) and isinstance(run.get("argv"), list) and run["argv"]
                    and all(isinstance(a, str) for a in run["argv"]) and isinstance(run.get("env", {}), dict)
                    and isinstance(settings, dict)):
                raise ValueError("run.json needs argv (strings) and env (object); settings.json an object")
            agent_from = run.get("agent_from", "c0000")
        except (OSError, ValueError) as exc:
            refuse(f"{type(exc).__name__}: {exc}"[:300])
        if not isinstance(agent_from, str) or self.state["candidates"].get(agent_from, {}).get("status") in (
                None, "rejected", "smoke", "unrestorable"):
            refuse(f"run.json agent_from: {agent_from!r} is not an archived candidate that passed its smoke")
        self.loop.runtime.seal(target)
        problem = await self.test_policy(target)  # the new policy.py on the loop's current view
        if problem:
            refuse(f"policy.py: {problem}")
        meta = self.state["meta"]
        meta["items"][mid] = {"created": time.time(), "status": "checking", "parent": meta["live"],
                              "note": str(note)[:2000]}
        task = asyncio.create_task(self.self_check(mid))
        self.tasks = getattr(self, "tasks", set())
        self.tasks.add(task)  # held until done (the event loop keeps only weak references)
        task.add_done_callback(self.tasks.discard)
        return {"meta": mid, "status": "checking: a short session under the new setup; see events.jsonl"}

    async def self_check(self, mid):
        """A short session under the new setup, as its own user (``rsichk``) in a fresh workspace with a socket of its
        own that answers ``status`` only. It passes when that socket got a ``status`` call and the session made a
        model request through the proxy: both observed by the root, out of the main session's reach."""
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
        self.notify({"event": "self_update", "meta": mid, "ok": ok})
        self.loop.event_log({"event": "rsi_self_update", "meta": mid, "ok": ok})
        self.loop.save("self_update")
