"""Parent-owned chroot staging for hosts that block Landlock and namespaces.

Only explicit runtime files and this game's evidence/workspace enter a jail.
The child drops to uid/gid 65534 before model code. Model code never runs in
this staging process. Immutable files are linked incrementally, not rewritten.
"""

import ctypes
import functools
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

_LOCK = threading.Lock()
_RUNTIME = None


def needed():
    return os.environ.get("ARC_FORCE_CHROOT") == "1" or _platform_needs_chroot()


@functools.lru_cache(maxsize=1)
def _platform_needs_chroot():
    if sys.platform != "linux":
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(444, 0, 0, 1) >= 3:
        return False
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import ctypes,sys;sys.exit(ctypes.CDLL(None).unshare(0x10000000)!=0)",
        ],
        capture_output=True,
        timeout=10,
    )
    return probe.returncode != 0


def _remove(path):
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)


def _tree(source, target, *, writable=False, prune=False):
    """Synchronize without following user symlinks or recopying existing inodes."""
    source, target = Path(source), Path(target)
    if source.is_symlink():
        value = os.readlink(source)
        if target.is_symlink() and os.readlink(target) == value:
            return
        if target.exists() or target.is_symlink():
            _remove(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(value)
        return
    if source.is_dir():
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            _remove(target)
        target.mkdir(parents=True, exist_ok=True)
        if prune:
            for entry in target.iterdir():
                if not (source / entry.name).exists() and not (source / entry.name).is_symlink():
                    _remove(entry)
        for entry in source.iterdir():
            _tree(entry, target / entry.name, writable=writable, prune=prune)
        target.chmod(0o700 if writable else 0o555)
        os.chown(target, 65534 if writable else 0, 65534 if writable else 0)
        return
    if source.is_char_device() and source.stat().st_rdev == os.makedev(1, 3):
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.mknod(target, stat.S_IFCHR | 0o666, os.makedev(1, 3))
            target.chmod(0o666)
        return
    if not source.is_file():
        raise ValueError(f"Only regular files/directories/symlinks may be staged: {source}")
    if target.exists() and not target.is_symlink() and source.samefile(target):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        _remove(target)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    target.chmod((source.stat().st_mode & 0o111) | (0o600 if writable else 0o444))
    os.chown(target, 65534 if writable else 0, 65534 if writable else 0)


def _runtime():
    global _RUNTIME
    with _LOCK:
        if _RUNTIME is not None:
            return _RUNTIME
        if os.geteuid() != 0:
            raise RuntimeError(
                "Host blocks namespace isolation and lacks root for chroot isolation"
            )
        from .sandbox import runtime_paths

        root = Path(tempfile.mkdtemp(prefix="arc-runtime-jail-"))
        # Copy runtime sources once; never chmod/chown hardlinks to host system files.
        broad = {"/usr/bin", "/bin", "/lib", "/lib64", "/usr/lib"}
        sources = [Path(p) for p in runtime_paths() if p not in broad and not p.startswith("/dev/")]
        commands = "bash ls cat head tail grep sed awk sort uniq wc cut find xargs echo printf cp mv rm mkdir touch ln pwd sleep true false uname stat date env timeout tee tr diff base64 sha256sum".split()
        sources += [Path(shutil.which(command)) for command in commands if shutil.which(command)]
        sources += [Path(sys.executable)]
        binaries = []
        for source in sources:
            target = root / str(source).lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True, symlinks=False)
                binaries.extend(p for p in source.rglob("*") if p.is_file() and ".so" in p.name)
            elif source.exists():
                shutil.copyfile(source, target)
                shutil.copymode(source, target)
                binaries.append(source)
        # Coding agents call `python3`; venv interpreters are staged under their own name.
        executable = Path(sys.executable)
        for alias in ("python", "python3"):
            link = root / str(executable.parent / alias).lstrip("/")
            if not link.exists() and not link.is_symlink():
                link.symlink_to(executable.name)
        # Shared libraries are explicit dependency closure, not all /usr/lib.
        for offset in range(0, len(binaries), 64):
            result = subprocess.run(
                ["ldd", *map(str, binaries[offset : offset + 64])],
                capture_output=True,
                text=True,
                timeout=30,
            )
            for name in set(re.findall(r"(/[^\s:]+)\s+\(", result.stdout)):
                source = Path(name)
                if source.is_file():
                    target = root / name.lstrip("/")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        shutil.copyfile(source, target)
                        shutil.copymode(source, target)
        # libseccomp is loaded dynamically rather than linked to Python.
        result = subprocess.run(
            ["/sbin/ldconfig", "-p"], capture_output=True, text=True, check=True
        )
        lib = next(
            (
                line.split("=> ")[1]
                for line in result.stdout.splitlines()
                if "libseccomp.so.2 " in line
            ),
            None,
        )
        if not lib:
            raise RuntimeError("libseccomp.so.2 unavailable for chroot isolation")
        target = root / lib.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(lib, target)
        # Exact loader aliases must exist; no host configuration hooks are inherited.
        for alias in ["/lib", "/lib64", "/bin", "/sbin"]:
            source = Path(alias)
            target = root / alias.lstrip("/")
            if source.is_symlink() and not target.exists():
                target.symlink_to(os.readlink(source))
        for hook in root.rglob("sitecustomize.py"):
            hook.write_text("# No inherited host customization in isolated model tools.\n")
        for file in root.rglob("*"):
            if not file.is_symlink():
                file.chmod(0o555 if file.is_dir() else (0o444 | (file.stat().st_mode & 0o111)))
        (root / "dev").mkdir(exist_ok=True)
        os.mknod(root / "dev/null", stat.S_IFCHR | 0o666, os.makedev(1, 3))
        (root / "dev/null").chmod(0o666)
        root.chmod(0o755)
        _RUNTIME = root
        return root


class Jail:
    def __init__(self, workspace=None, reads=()):
        self.workspace = Path(workspace) if workspace else None
        self.reads = [Path(p) for p in reads]
        runtime = _runtime()
        if workspace is None:
            self.root = runtime
        else:
            self.root = Path(tempfile.mkdtemp(prefix="arc-game-jail-"))
            _tree(runtime, self.root)
            self.sync_in()
        self.root.chmod(0o755)

    def inside(self, path):
        return self.root / str(Path(path)).lstrip("/")

    def sync_in(self):
        for path in self.reads:
            if path.exists():
                _tree(path, self.inside(path))
        if self.workspace:
            _tree(self.workspace, self.inside(self.workspace), writable=True, prune=True)

    def sync_evidence(self):
        for path in self.reads:
            if path.exists():
                _tree(path, self.inside(path))

    def sync_out(self):
        if self.workspace:
            _tree(self.inside(self.workspace), self.workspace, writable=True, prune=True)
