"""One game played by one candidate (fixed yardstick side: broker, timer, scorer), and the host runtime the sessions
run on (users, network rules, node and kernel environments, locked game folders).

An attempt: a fresh copy of the candidate's ``player/workspace`` at /workspace and ``player/home`` as HOME, the broker
(``rsi_loop.broker``, game code as ``rsigame``) behind /broker/broker.sock, then the candidate's ``player/run.json``
command as the slot's user in a private mount namespace where the loop's own folders, the bundle, the game folders
and the outputs are hidden. The wall clock (slots x Kaggle runtime / 110) starts when the session starts and ends it;
a WIN ends it after a short grace. Score: the official RHAE of the broker's record against the point's baselines.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from . import sandbox
from .score import level_actions_from_record, rhae

RUNTIME = Path("/opt/rsi-runtime")
WIN_GRACE = 30
# Prime Agent's kernel check (packages/coding-agent/src/core/kernel/bootstrap.ts) imports all of these.
KERNEL_PACKAGES = ("ipykernel", "nest-asyncio", "tyro", "numpy", "dill", "requests", "httpx", "pyyaml", "tomli",
                   "python-dotenv", "pandas", "scipy", "beautifulsoup4", "lxml", "pydantic", "prime-agent-runtime")
IDLE_END = 60  # an rpc session idles after agent_end; this long without new output ends the game
# A token-capped (held-out) game has no time slice: a session that stopped calling the model and writing output for
# this long ends ("idle"); 14 Oct, v22: a player stalled after a compaction and held the exclusive machine 2.5 h.
IDLE_MODEL = 900
KAGGLE_RAM = 180 * 10 ** 9
KAGGLE_RESERVED = 72 * 2 ** 30


def slim_line(line):
    """A Prime Agent rpc event as the session log keeps it: a streaming update carries the whole message so far
    (``message`` and ``assistantMessageEvent.partial``), so logging it as is grows quadratically (v15, 13 Oct: 1.7 GB
    per player-hour); the delta alone is kept. Other events pass unchanged (``message_end`` has the full message)."""
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
    """Copy a session's stdout into its log, line by line, through ``slim_line``."""
    with open(path, "ab") as out:
        while True:
            try:
                line = await stream.readline()
            except (ValueError, asyncio.LimitOverrunError):  # a line beyond the limit: keep its start, go on
                line = (await stream.read(1 << 20))[:4096] + b"\n"
            if not line:
                return
            out.write(slim_line(line))
            out.flush()
SLOTS = 96  # play slots at most (concurrency 97 with the proposer); players use ~0.4 GB of RAM each
BROKER_PACKAGES = ("arc-agi", "arcengine", "numpy")


class Runtime:
    """Host setup shared by every session (run once per loop start; idempotent)."""

    def __init__(self, root, bundle, games_dir, cfg, proxy, log, ckpt):
        self.root, self.bundle, self.games_dir, self.cfg, self.proxy, self.log = (
            Path(root), Path(bundle), games_dir, cfg, proxy, log)
        self.ckpt = Path(ckpt)
        self.sessions = [sandbox.PROPOSER, sandbox.CHECK] + [sandbox.slot_user(i) for i in range(SLOTS)]
        self.games = [sandbox.VET] + [sandbox.game_user(i) for i in range(SLOTS)]
        self.users = self.sessions + self.games + [sandbox.SERVE]
        self.folders = {}  # games set -> {game_id: its folder under environment_files}

    def hidden(self, keep=()):
        """Paths a sandboxed process must not see (mounted over with an empty tmpfs in its namespace): the loop's
        folders, the bundle, the checkpoint and trial output (also under the private /tmp), the run's output root,
        launch files, game folders. ``keep``: paths this process needs (the broker's library)."""
        paths = [self.root, self.bundle, RUNTIME / "lib", self.ckpt, self.ckpt.parent, "/kaggle", "/root", "/runs",
                 "/opt/arc", "/run/arc", "/tmp/arena", "/assets"]  # /assets: VM inputs (models, runtime, games)
        if self.games_dir:
            paths.append(self.games_dir)
        for extra in os.environ.get("RSI_HIDE", "").split(":"):
            if extra:
                paths.append(extra)
        keep = {str(k) for k in keep}
        return [str(p) for p in dict.fromkeys(map(str, paths))
                if str(p) not in keep and Path(p).exists() and Path(p).is_dir()]

    def memory(self, role, concurrency=14):
        """Memory ceilings (bytes) per sandbox, from the VM's RAM (218 GB on the RTX PRO 6000 preset): the server
        keeps 55% (its 47.7 GiB PLE table is page-locked host memory), the play sessions share 30% at the running
        concurrency (the proposer counts twice), the root loop keeps the rest; a GPU job (server stopped) 60%."""
        total = sandbox.mem_total()
        if not total:
            return None
        # the competition host (one RTX PRO 6000) has ~180 GB of host memory; serving and the harness keep 72 GiB (the 47.7 GiB
        # page-locked PLE table, the engine, the game brokers); the rest is shared by the C sessions (the proposer
        # counts twice: C + 1 shares). 6.4 GiB per player at C = 14.
        share = max((KAGGLE_RAM - KAGGLE_RESERVED) / (max(2, int(concurrency)) + 1), 1 << 30)
        return int({"player": share, "proposer": 2 * share, "check": share, "game": 2 << 30, "vet": 2 << 30,
                    "policy": 1 << 30, "gpu_job": 0.60 * total, "server": 0.55 * total}[role])

    def proxy_access(self):
        return Path(self.proxy.unix_path).parent, self.proxy.port

    def lock_games(self):
        """Game folders: root-owned and read-only; /rsi/games itself root-only (each broker or vet gets its own game
        folder bound into its namespace, nothing else)."""
        games = self.root / "games"
        subprocess.run(["chown", "-R", "root:root", str(games)], check=True)
        subprocess.run(["chmod", "-R", "a+rX,a-w", str(games)], check=True)
        os.chmod(games, 0o700)
        self.folders = {}

    def game_folder(self, games_set, game_id):
        """``environment_files/<folder>`` of one game (the folder holding its version folders)."""
        index = self.folders.get(games_set)
        if index is None:
            index = {}
            for meta in (self.root / "games" / games_set / "environment_files").glob("*/*/metadata.json"):
                try:
                    index[json.loads(meta.read_text())["game_id"]] = meta.parent.parent
                except (OSError, ValueError, KeyError):
                    continue
            self.folders[games_set] = index
        if game_id in index:
            return index[game_id]
        matches = [path for gid, path in index.items() if gid.startswith(game_id)]
        if len(set(matches)) != 1:
            raise RuntimeError(f"{game_id}: {len(set(matches))} folders in {games_set}")
        return matches[0]

    def prepare(self):
        for name in ("root", "games", "archive", "meta", "prop", "slots", "sock", "serve"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o711)
        os.chmod(self.root / "root", 0o700)
        for user in self.users:
            sandbox.ensure_user(user)
        try:  # the model server's own port: reachable through the proxy only
            from arc_infra.serving_setup import PORT
        except Exception:  # noqa: BLE001 - tests without the serving stack
            PORT = int(os.environ.get("RSI_SERVER_PORT", "1234"))
        sandbox.close_network(self.sessions, [PORT])
        sandbox.close_network(self.games, [PORT, self.cfg.get("proxy_port", 18080)])
        sandbox.close_network([sandbox.SERVE], [53])  # no DNS either (the resolver would carry data out)
        subprocess.run(["usermod", "-a", "-G", sandbox.PROPOSER, sandbox.CHECK], check=True)  # reads the archive
        for name in ("tmp", "home"):
            folder = self.root / "serve" / name
            folder.mkdir(parents=True, exist_ok=True)
            sandbox.own(folder, sandbox.SERVE, 0o700)
        for folder in (self.ckpt, self.ckpt.parent):
            os.chmod(folder, 0o700)
        RUNTIME.mkdir(parents=True, exist_ok=True)
        os.chmod(RUNTIME, 0o755)
        node = RUNTIME / "node"
        if not (node / "bin/node").exists():
            archive = next((self.bundle / "rsi/runtime").glob("node-*-linux-*.tar.xz"))
            staging = RUNTIME / "node-staging"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir()
            subprocess.run(["tar", "-xJf", str(archive), "-C", str(staging)], check=True)
            next(staging.iterdir()).rename(node)
            shutil.rmtree(staging)
        for name, packages, wheels in (("kernel", KERNEL_PACKAGES, self.bundle / "rsi/runtime/wheelhouse"),
                                       ("broker", BROKER_PACKAGES, self.bundle / "wheels"),
                                       # the proposer's kernel: the player's packages plus the ARC engine, so it can
                                       # run and test games it writes (rsi.add_game)
                                       ("propkernel", KERNEL_PACKAGES + ("arc-agi", "arcengine"),
                                        self.bundle / "wheels")):
            venv = RUNTIME / name
            if not (venv / ".installed").exists():  # a failed install leaves no marker and is redone
                shutil.rmtree(venv, ignore_errors=True)
                subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
                subprocess.run([str(venv / "bin/python"), "-m", "pip", "install", "--no-index", "--find-links",
                                str(wheels), "--find-links", str(self.bundle / "rsi/runtime/wheelhouse"), *packages],
                               check=True, capture_output=True)
                (venv / ".installed").write_text("ok\n")
        lib = RUNTIME / "lib"
        shutil.rmtree(lib, ignore_errors=True)
        shutil.copytree(self.bundle / "src", lib, ignore=shutil.ignore_patterns("__pycache__"))
        gamelib = RUNTIME / "gamelib"  # all a broker or a vet sees of our code: two standalone scripts
        shutil.rmtree(gamelib, ignore_errors=True)
        gamelib.mkdir()
        for name in ("broker.py", "vet.py"):
            shutil.copy2(self.bundle / "src/rsi_loop" / name, gamelib / name)
        gpujob = RUNTIME / "gpujob"
        gpujob.mkdir(exist_ok=True)
        shutil.copy2(self.bundle / "src/rsi_loop/gpu_guard.py", gpujob / "gpu_guard.py")
        subprocess.run(["chmod", "-R", "a+rX", str(RUNTIME)], check=True)
        games = self.root / "games"
        for role in ("pool", "practice", "heldout_a", "heldout_b"):
            source = self.bundle / "rsi/games" / role
            if source.is_dir() and not (games / role).exists():
                shutil.copytree(source, games / role)
        for base in (self.ckpt, self.bundle / "configs/rsi_resume"):
            for role in ("pool", "practice"):  # games the proposer wrote, from a checkpoint
                added = base / "games" / role / "environment_files"
                if added.is_dir():
                    shutil.copytree(added, games / role / "environment_files", dirs_exist_ok=True)
        self.lock_games()
        for locked in [self.bundle / "rsi/games", self.bundle / "generated_games", self.games_dir]:
            if locked and Path(locked).exists():
                subprocess.run(["chmod", "-R", "go-rwx", str(locked)], check=False)

    def install_gen0(self, tree):
        """Gen 0's candidate tree from the bundle (Prime Agent v0.3.3 build, the stock ARC runner files)."""
        source = self.bundle / "rsi/gen0"
        tree.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source / "tree", tree, dirs_exist_ok=True)
        with tarfile.open(source / "prime-agent.tar.gz") as tar:
            tar.extractall(tree / "player/prime-agent", filter="tar")
        self.seal(tree.parent)

    def seal(self, folder):
        """Archive copies: root-owned and read-only. Contents are world-readable (a player reaches only its own
        candidate's ``player/``, bound into its namespace); the folder itself is root:rsiprop 0750."""
        import pwd

        gid = pwd.getpwnam(sandbox.PROPOSER).pw_gid
        subprocess.run(["chown", "-R", f"root:{gid}", str(folder)], check=True)
        subprocess.run(["chmod", "-R", "a+rX,a-w", str(folder)], check=True)
        os.chmod(folder, 0o750)

    def restore(self, ckpt, resume, state):
        """Candidate trees from the checkpoint mirror (node_modules linked from gen 0's build)."""
        mirror = next((p / "archive" for p in (Path(ckpt), Path(resume)) if (p / "archive").is_dir()), None)
        gen0 = self.root / "archive/c0000/tree"
        if not gen0.exists():
            self.install_gen0(gen0)
        for base in (Path(ckpt), Path(resume)):  # proposer setups, workspace and HOME
            if (base / "meta").is_dir():
                for item in sorted((base / "meta").iterdir()):
                    if not (self.root / "meta" / item.name).exists():
                        shutil.copytree(item, self.root / "meta" / item.name, symlinks=True)
                        self.seal(self.root / "meta" / item.name)
            for part, target in (("prop/work", self.root / "prop/ws/work"), ("prop/home", self.root / "prop/home")):
                if (base / part).is_dir() and not target.exists():
                    shutil.copytree(base / part, target, symlinks=True)
            for item in (state.get("big_files") or {}).get("work") or []:
                if not (self.root / "prop/ws/work" / item["path"]).exists():
                    self.restore_big("work", item["path"], self.root / "prop/ws/work")
            modules = gen0 / "player/prime-agent/node_modules"
            for fork in sorted((self.root / "prop/ws/work").glob("*/player/prime-agent")):
                if any(p.is_symlink() for p in (fork, fork.parent, fork.parent.parent)):
                    continue  # a link the proposer planted: never written through
                if modules.exists() and not os.path.lexists(fork / "node_modules"):  # mirrors leave node_modules out
                    from .candidates import copy_tree
                    copy_tree(modules, fork / "node_modules", link_big=True)
            if (base / "prop/events.jsonl").is_file() and not (self.root / "prop/ws/events.jsonl").exists():
                (self.root / "prop/ws").mkdir(parents=True, exist_ok=True)
                shutil.copy2(base / "prop/events.jsonl", self.root / "prop/ws/events.jsonl")
        if mirror is None:
            return
        if (mirror / "c0000/practice").is_dir():  # gen 0's tree comes from the bundle, its practice traces from here
            shutil.copytree(mirror / "c0000/practice", self.root / "archive/c0000/practice", symlinks=True,
                            dirs_exist_ok=True)
        for cid in state["candidates"]:
            target = self.root / "archive" / cid
            if cid == "c0000" or (target / "tree").exists():
                continue
            if (mirror / cid).is_dir():
                shutil.copytree(mirror / cid, target, symlinks=True, dirs_exist_ok=True)
            for item in state["candidates"][cid].get("not_checkpointed") or []:
                self.restore_big(f"archive/{cid}", item["path"], target)
            missing = [f["path"] for f in state["candidates"][cid].get("not_checkpointed") or []
                       if not (target / f["path"]).exists()] + ([] if (target / "tree").is_dir() else ["tree"])
            if missing:  # files over 512 MB are not checkpointed: this candidate cannot be played again
                state["candidates"][cid].update(status="unrestorable", problems=[f"not checkpointed: {missing[:5]}"])
                state["jobs"] = [j for j in state["jobs"] if j["candidate"] != cid]
            if not (target / "tree").is_dir():
                if target.is_dir():
                    self.seal(target)
                continue
            modules = gen0 / "player/prime-agent/node_modules"
            mine = target / "tree/player/prime-agent/node_modules"
            if modules.exists() and not mine.exists():
                from .candidates import copy_tree
                copy_tree(modules, mine, link_big=True)
            self.seal(target)

    def restore_big(self, area, rel, target_root):
        """A big file of an earlier leg from ``big_inputs`` (its run output's rsi-big folder, attached as an input)."""
        from .candidates import _parts, open_dir

        for base in self.cfg.get("big_inputs") or []:
            source = Path(base) / area / rel
            if source.is_file() and not source.is_symlink():
                parts = _parts(rel)
                try:  # the target folder may be the proposer's (links it planted are never followed)
                    dfd = open_dir(target_root, os.path.join(*parts[:-1]) if len(parts) > 1 else "", create=True)
                except (OSError, ValueError):
                    return False
                try:
                    fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dfd)
                except OSError:
                    os.close(dfd)
                    return False
                os.close(dfd)
                with os.fdopen(fd, "wb") as writer, source.open("rb") as reader:
                    shutil.copyfileobj(reader, writer, 1 << 24)
                return True
        return False

    def egress(self):
        try:
            return sandbox.egress_counts(self.users)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)[:200]}


class Attempt:
    def __init__(self, runtime, job, point, tree, wall, meter, games_set, concurrency=14, token_cap=None):
        self.rt, self.job, self.point, self.tree = runtime, job, point, Path(tree)
        self.concurrency, self.token_cap = concurrency, token_cap
        self.wall, self.meter, self.set = float(wall), meter, games_set
        self.user = sandbox.slot_user(job["slot"])
        self.game_user = sandbox.game_user(job["slot"])
        self.dir = runtime.root / "root/attempts" / job["id"]
        self.slot = runtime.root / "slots" / f"{job['slot']:02d}"
        self.sock = runtime.root / "sock" / job["id"]

    def _fresh(self):
        shutil.rmtree(self.slot, ignore_errors=True)
        for part in ("workspace", "home", "state"):
            (self.slot / part).mkdir(parents=True)
        # links stay links (the root never reads through a candidate's link; inside the session they resolve in the
        # session's own view, where the loop's files do not exist)
        shutil.copytree(self.tree / "player/workspace", self.slot / "workspace", dirs_exist_ok=True, symlinks=True)
        if (self.tree / "player/home").is_dir() and not (self.tree / "player/home").is_symlink():
            shutil.copytree(self.tree / "player/home", self.slot / "home", dirs_exist_ok=True, symlinks=True)
        sandbox.own(self.slot, self.user, 0o700)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sock.mkdir(parents=True, exist_ok=True)
        import pwd
        game = pwd.getpwnam(self.game_user)
        os.chown(self.sock, game.pw_uid, pwd.getpwnam(self.user).pw_gid)
        os.chmod(self.sock, 0o750)
        record = self.sock / "record.jsonl"
        record.touch()
        os.chown(record, game.pw_uid, game.pw_gid)
        os.chmod(record, 0o644)

    def broker_argv(self, program=None):
        """The broker (or ``program``, the isolation probe) as this slot's game user: private mount and network
        namespaces with only its own game folder (read-only) and the slot's socket folder."""
        folder = self.rt.game_folder(self.set, self.point["game"])
        env = {"HOME": "/tmp", "PATH": "/usr/bin:/bin"}
        argv = program or [str(RUNTIME / "broker/bin/python"), "-I", str(RUNTIME / "gamelib/broker.py"),
                           "--game", self.point["game"], "--corpus", "/game/environment_files",
                           "--socket", "/broker/broker.sock", "--record", "/broker/record.jsonl",
                           *(["--levels", str(int(self.point["level_cap"]))] if self.point.get("level_cap") else [])]
        binds = [(folder, f"/game/environment_files/{folder.name}", True), (self.sock, "/broker", False)]
        return sandbox.command(argv, self.game_user, None, env, cwd="/tmp", extra_binds=binds, hide=self.rt.hidden(),
                               tmp_size="1g", shm_size="256m", net="none", memory=self.rt.memory("game"))

    async def _broker(self):
        log = (self.dir / "broker.log").open("ab")
        process = await asyncio.create_subprocess_exec(*self.broker_argv(), stdout=log, stderr=subprocess.STDOUT,
                                                       start_new_session=True)
        for _ in range(300):
            if (self.sock / "broker.sock").exists():
                return process
            if process.returncode is not None:
                break
            await asyncio.sleep(0.1)
        raise RuntimeError(f"broker did not start: {(self.dir / 'broker.log').read_text()[-500:]}")

    def _session_argv(self, key, program=None):
        from .candidates import read_inside
        run = json.loads(read_inside(self.tree, "player/run.json"))
        prompt = read_inside(self.tree, "player/game-prompt.txt") if program is None else ""
        self.stdin_prompt = None if any("{prompt}" in a for a in run["argv"]) else prompt
        argv = program or [a.replace("{prompt}", prompt) for a in run["argv"]]
        env = {"HOME": "/home/session", "USER": self.user, "LANG": "C.UTF-8",
               "PATH": f"/candidate/bin:{RUNTIME}/node/bin:{RUNTIME}/kernel/bin:/usr/local/bin:/usr/bin:/bin",
               "ARC3_GAME_BROKER_SOCKET": "/broker/broker.sock", "ARC3_STATE": "/state",
               "RSI_PROXY_URL": f"http://127.0.0.1:{self.rt.proxy.port}/v1", "RSI_PROXY_KEY": key,
               "RSI_MODEL": self.rt.proxy.served, **{str(k): str(v) for k, v in (run.get("env") or {}).items()}}
        binds = [(self.slot / "home", "/home/session", False), (self.slot / "state", "/state", False),
                 (self.sock, "/broker", False), (self.tree / "player", "/candidate", True)]
        return sandbox.command(argv, self.user, self.slot / "workspace", env, extra_binds=binds,
                               hide=self.rt.hidden(), net="proxy", proxy=self.rt.proxy_access(),
                               memory=self.rt.memory("player", self.concurrency))

    async def run(self):
        self._fresh()
        broker = await self._broker()
        key = secrets.token_hex(16)
        session = self.job["id"]
        self.rt.proxy.register(key, session, self.meter, candidate=self.job["candidate"], token_cap=self.token_cap)
        started = time.time()
        outcome, error, process, copier = "time", None, None, None
        try:
            process = await asyncio.create_subprocess_exec(*self._session_argv(key), stdin=subprocess.PIPE,
                                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                           start_new_session=True, limit=1 << 26)
            copier = asyncio.ensure_future(slim_copy(process.stdout, self.dir / "session.log"))
            if self.stdin_prompt is not None:
                process.stdin.write(json.dumps({"id": "1", "type": "prompt", "message": self.stdin_prompt}).encode()
                                    + b"\n")
                await process.stdin.drain()
            won_at = None
            log_path, size, quiet_since = self.dir / "session.log", 0, time.time()
            while True:
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                    outcome = "exit"
                    break
                except asyncio.TimeoutError:
                    pass
                elapsed = time.time() - started
                if won_at is None and self._won():
                    won_at = time.time()
                if won_at and time.time() - won_at > WIN_GRACE:
                    outcome = "win"
                    break
                if elapsed > self.wall:
                    outcome = "time"
                    break
                if self.token_cap and self.rt.proxy.session_totals(session)["completion_tokens"] >= self.token_cap:
                    outcome = "tokens"  # held-out format: the game ends at its token cap
                    break
                inflight = self.rt.proxy.inflight[session]
                if not inflight and time.time() - max(
                        quiet_since, self.rt.proxy.last_activity.get(session, started)) > IDLE_MODEL:
                    outcome = "idle"  # 15 min without a model call or output (any game since 14 Oct evening)
                    break
                now_size = log_path.stat().st_size
                try:  # game actions count as activity too (a tool playing a long scripted run); lstat, no link followed
                    now_size += os.lstat(self.sock / "record.jsonl").st_size
                except OSError:
                    pass
                if now_size != size:
                    size, quiet_since = now_size, time.time()
                elif not inflight and time.time() - max(quiet_since, self.rt.proxy.last_activity.get(
                        session, started)) > IDLE_END and self._ended(log_path):
                    outcome = "stopped"
                    break
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001
            outcome, error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            seconds = time.time() - started
            for proc in (process, broker):
                if proc is not None and proc.returncode is None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            for user in (self.user, self.game_user):
                subprocess.run(["pkill", "-KILL", "-u", user], check=False)
            if copier is not None:
                try:
                    await asyncio.wait_for(copier, timeout=10)
                except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                    copier.cancel()
            self.rt.proxy.forget(key)
            if outcome == "cancelled":
                self._keep_trace()
        # the game's user owns the socket folder: never follow a link it put in place of the record
        from .candidates import copy_regular
        try:
            copy_regular(self.sock / "record.jsonl", self.dir / "record.jsonl")
        except (OSError, ValueError):
            (self.dir / "record.jsonl").write_text("")
        shutil.rmtree(self.sock, ignore_errors=True)
        rows = [json.loads(line) for line in (self.dir / "record.jsonl").read_text().splitlines() if line.strip()]
        actions = level_actions_from_record(rows)
        if self._won(rows) and outcome == "exit":
            outcome = "win"
        totals = self.rt.proxy.session_totals(session)
        infra = outcome == "error" or (totals["errors"] > 0 and totals["requests"] == totals["errors"]
                                       and self._server_down())
        baselines = self.point.get("baselines")  # held-out B has none yet: levels cleared only
        row = {"score": rhae(baselines, actions) if baselines else None, "levels": len(actions),
               "level_actions": actions, "total_levels": int(self.point["total_levels"]), "actions": sum(1 for r in rows if "n" in r),
               "outcome": outcome, "error": error, "seconds": round(seconds, 1), "wall": self.wall, "token_cap": self.token_cap,
               "tokens": totals["completion_tokens"], "prompt_tokens": totals["prompt_tokens"],
               "model_requests": totals["requests"], "request_errors": totals["errors"],
               "non_flash": totals["non_flash"], "infrastructure": bool(infra)}
        if infra:
            row["score"] = None
        self._keep_trace()
        return row

    @staticmethod
    def _ended(log_path):
        """The agent's run has ended (rpc mode then idles): its newest agent_end comes after its newest turn. Prime emits
        compaction and refine events after agent_end, so the last line alone misses the end (14 Oct: gen 0 games that
        hit its autonomous limits idled to the wall)."""
        with log_path.open("rb") as handle:
            handle.seek(max(0, log_path.stat().st_size - (1 << 20)))
            tail = handle.read().decode("utf-8", "replace")
        end, start, pending = -1, -1, set()
        for i, line in enumerate(tail.splitlines()):
            if line.startswith('{"type":"agent_end"'):
                end, pending = i, set()
            elif line.startswith(('{"type":"agent_start"', '{"type":"turn_start"', '{"type":"message_start"')):
                start = i
            else:  # 14 Oct: a compaction (or retry) after agent_end is followed by an autonomous continuation
                for kind in ("compaction", "auto_retry"):
                    if line.startswith(f'{{"type":"{kind}_start"'):
                        pending.add(kind)
                    elif line.startswith(f'{{"type":"{kind}_end"'):
                        pending.discard(kind)
        return end > start and not pending

    def _won(self, rows=None):
        if rows is None:
            try:
                fd = os.open(self.sock / "record.jsonl", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, "rb") as handle:
                    handle.seek(max(0, os.fstat(handle.fileno()).st_size - 4000))
                    tail = handle.read().decode("utf-8", "replace")
            except OSError:
                return False
            return '"state": "WIN"' in tail
        return any(r.get("state") == "WIN" for r in rows)

    def _server_down(self):
        import urllib.request
        try:
            with urllib.request.urlopen(self.rt.proxy.upstream + "/health", timeout=5) as response:
                return response.status != 200
        except Exception:  # noqa: BLE001
            return True

    def _keep_trace(self):
        """Session files and logs into the attempt folder (root-only); practice and smoke traces also go to the
        proposer's archive view."""
        keep = self.dir / "trace"
        shutil.rmtree(keep, ignore_errors=True)
        try:  # the player's HOME: read through descriptors, no link followed (it wrote that folder)
            from .candidates import copy_untrusted
            copy_untrusted(self.slot, "home", keep / "home", skip=("node_modules",), limit=256 * 1024 ** 2)
        except Exception:  # noqa: BLE001
            pass
        for name in ("session.log", "record.jsonl", "broker.log"):
            if (self.dir / name).exists():
                shutil.copy2(self.dir / name, keep / name) if keep.exists() else None
        if self.job["kind"] in ("practice", "smoke"):
            tail = "-smoke" if self.job["kind"] == "smoke" else ""
            name = f"{self.point['game']}-{self.job['id']}{tail}"
            view = self.rt.root / "archive" / self.job["candidate"] / "practice" / name
            shutil.rmtree(view, ignore_errors=True)
            shutil.copytree(keep, view, symlinks=True)
            self.rt.seal(view)
