"""Host setup for the loop (run once per leg; idempotent): users, network rules, the proposer's runtime (node, its
kernel, Prime Agent), the fixed library copy, locked game folders. Adapted from rsi_loop.attempt.Runtime."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from . import sandbox

RUNTIME = Path("/opt/rsi-runtime")
# v27's list (prime-agent-runtime, module rlm, is the kernel side of Prime Agent: l2 proposer died on its absence) + the engine
KERNEL_PACKAGES = ("ipykernel", "nest-asyncio", "tyro", "numpy", "dill", "requests", "httpx", "pyyaml", "tomli",
                   "python-dotenv", "pandas", "scipy", "beautifulsoup4", "lxml", "pydantic", "prime-agent-runtime",
                   "arc-agi", "arcengine")
EVAL = "rsieval"


class Runtime:
    def __init__(self, root, bundle, games_dir, cfg, proxy, ckpt):
        self.root, self.bundle, self.games_dir, self.cfg, self.proxy = (Path(root), Path(bundle), games_dir, cfg,
                                                                        proxy)
        self.ckpt = Path(ckpt)
        self.lib = RUNTIME / "lib"
        self.users = [sandbox.PROPOSER, sandbox.CHECK, sandbox.SERVE, EVAL]

    def hidden(self, keep=()):
        paths = [self.root, self.bundle, self.lib, self.ckpt, self.ckpt.parent, "/kaggle", "/root", "/runs",
                 "/opt/arc", "/run/arc", "/tmp/arena", "/assets"]
        if self.games_dir:
            paths.append(self.games_dir)
        keep = {str(k) for k in keep}
        return [str(p) for p in dict.fromkeys(map(str, paths))
                if str(p) not in keep and Path(p).exists() and Path(p).is_dir()]

    def memory(self, role, concurrency=14):
        total = sandbox.mem_total()
        if not total:
            return None
        return int({"proposer": 24 << 30, "check": 12 << 30, "policy": 1 << 30, "eval": 0.30 * total,
                    "gpu_job": 0.60 * total, "server": 0.55 * total}.get(role, 8 << 30))

    def proxy_access(self):
        return Path(self.proxy.unix_path).parent, self.proxy.port

    def allow_gateway(self, port):
        """The proposer and check sessions run in their own network namespace (loopback + the proxy only), so a
        gateway port on the host loopback is out of their reach; the evaluation user reaches it under its rules."""

    def forbid_gateway(self, port):
        pass

    def prepare(self, worker=False):
        """``worker``: an evaluation worker VM (no proposer): no node, kernels or Prime Agent."""
        for name in ("root", "games", "archive", "meta", "prop", "eval", "sock", "serve", "psock"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o711)
        os.chmod(self.root / "root", 0o700)
        for user in self.users:
            sandbox.ensure_user(user)
        try:
            from arc_infra.serving_setup import PORT
        except Exception:  # noqa: BLE001 - tests without the serving stack
            PORT = int(os.environ.get("RSI_SERVER_PORT", "1234"))
        sandbox.close_network([sandbox.PROPOSER, sandbox.CHECK], [PORT])
        sandbox.close_network([EVAL], [self.cfg.get("proxy_port", 18080)])
        sandbox.close_network([sandbox.SERVE], [53])
        subprocess.run(["usermod", "-a", "-G", sandbox.PROPOSER, sandbox.CHECK], check=True)
        for name in ("tmp", "home"):
            folder = self.root / "serve" / name
            folder.mkdir(parents=True, exist_ok=True)
            sandbox.own(folder, sandbox.SERVE, 0o700)
        for folder in (self.ckpt, self.ckpt.parent):
            os.chmod(folder, 0o700)
        RUNTIME.mkdir(parents=True, exist_ok=True)
        os.chmod(RUNTIME, 0o755)
        if worker:
            return self.prepare_shared()
        node = RUNTIME / "node"
        if not (node / "bin/node").exists():
            archive = next((self.bundle / "rsi/runtime").glob("node-*-linux-*.tar.xz"))
            staging = RUNTIME / "node-staging"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir()
            subprocess.run(["tar", "-xJf", str(archive), "-C", str(staging)], check=True)
            next(staging.iterdir()).rename(node)
            shutil.rmtree(staging)
        for name in ("kernel", "propkernel"):
            venv = RUNTIME / name
            if not (venv / ".installed").exists():
                shutil.rmtree(venv, ignore_errors=True)
                subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
                subprocess.run([str(venv / "bin/python"), "-m", "pip", "install", "--no-index", "--find-links",
                                str(self.bundle / "wheels"), "--find-links", str(self.bundle / "rsi/runtime/wheelhouse"),
                                *KERNEL_PACKAGES], check=True, capture_output=True)
                subprocess.run([str(venv / "bin/python"), "-c", "import rlm, ipykernel"], check=True)
                (venv / ".installed").write_text("ok\n")
        agent = RUNTIME / "prime-agent"
        if not (agent / "packages/coding-agent/dist/bundle/cli.js").exists():
            shutil.rmtree(agent, ignore_errors=True)
            agent.mkdir()
            with tarfile.open(self.bundle / "rsi/proposer/prime-agent.tar.gz") as tar:
                tar.extractall(agent, filter="tar")
        self.prepare_shared()

    def prepare_shared(self):
        shutil.rmtree(self.lib, ignore_errors=True)
        shutil.copytree(self.bundle / "src", self.lib, ignore=shutil.ignore_patterns("__pycache__"))
        gpujob = RUNTIME / "gpujob"
        gpujob.mkdir(exist_ok=True)
        shutil.copy2(self.bundle / "src/rsi2_loop/gpu_guard.py", gpujob / "gpu_guard.py")
        subprocess.run(["chmod", "-R", "a+rX", str(RUNTIME)], check=True)
        games = self.root / "games"
        if not (games / "pool").exists():
            shutil.copytree(self.games_dir, games / "pool/environment_files")
        if not (games / "practice").exists() and (self.bundle / "rsi/games/practice").is_dir():
            shutil.copytree(self.bundle / "rsi/games/practice", games / "practice")
        subprocess.run(["chown", "-R", "root:root", str(games)], check=True)
        subprocess.run(["chmod", "-R", "a+rX,a-w", str(games)], check=True)
        os.chmod(games, 0o700)
        for locked in (self.bundle / "rsi/games", self.bundle / "generated_games", self.games_dir):
            if locked and Path(locked).exists():
                subprocess.run(["chmod", "-R", "go-rwx", str(locked)], check=False)

    def seal(self, folder):
        import pwd

        gid = pwd.getpwnam(sandbox.PROPOSER).pw_gid
        subprocess.run(["chown", "-R", f"root:{gid}", str(folder)], check=True)
        subprocess.run(["chmod", "-R", "a+rX,a-w", str(folder)], check=True)
        os.chmod(folder, 0o750)

    def egress(self):
        try:
            return sandbox.egress_counts(self.users)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)[:200]}
