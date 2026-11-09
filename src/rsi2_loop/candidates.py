"""Candidate trees and the archive (docs/rsi.md 1: every variant goes into an archive; edits go into a candidate copy).

A candidate is one directory, the whole agent the proposer may change:

    player/                 what one game session runs (Prime Agent's ARC setup in gen 0)
      prime-agent/          Prime Agent source + build (v0.3.3 in gen 0); node_modules hard-linked read-only
      workspace/            copied into every game's /workspace (AGENTS.md, fixed_broker_client.py, guard, ...)
      home/                 the session's HOME (.prime/agent: settings, models.json, skills, memory)
      game-prompt.txt       the one user message
      run.json              {"argv": [...], "env": {...}, "concurrency": N, "game_seconds": S, "game_token_cap": T}
    serving/                the serving stack: server.json (the study's server block), env.json, overlay/ (Python
                            modules ahead of the installed ones), weights/ (files the server block points at)
    notes.md                free text

The archive keeps an immutable copy per candidate (root-owned, read-only); ``fork`` gives the proposer a writable
copy in which files over ``LINK_BYTES`` and everything under node_modules are read-only hard links (replace, do not
edit in place). ``serving_hash`` decides whether a candidate runs on the live stack.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

LINK_BYTES = 32 * 1024 * 1024
SIZE_CAP = 150 * 1024 ** 3   # one candidate's own files (weights included): a Kaggle dataset set, not a hard proof
REQUIRED = ("player/run.json", "player/game-prompt.txt", "player/workspace", "serving/server.json")


def _linked(rel: Path, size: int) -> bool:
    return size > LINK_BYTES or "node_modules" in rel.parts


def escaping_links(root) -> list[str]:
    """Symbolic links under ``root`` that resolve outside it (through any chain of links): a candidate or a game
    folder may hold links only within itself (npm's workspace links), never to the loop's files."""
    root = os.path.realpath(root)
    out = []
    for folder, dirs, files in os.walk(root):
        for name in dirs + files:
            path = os.path.join(folder, name)
            if os.path.islink(path):
                target = os.path.realpath(path)
                if os.path.isabs(os.readlink(path)) or not (target == root or target.startswith(root + os.sep)):
                    out.append(os.path.relpath(path, root))
    return out


def copy_regular(src, dst, *args, **kwargs):
    """``shutil.copytree``'s copy function for trees the proposer wrote: regular files only, opened without following
    a link and without blocking on a fifo (a swapped-in link, fifo or device is refused)."""
    import stat

    fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"not a regular file: {src}")
        os.set_blocking(fd, True)
        with os.fdopen(fd, "rb", closefd=False) as reader, open(dst, "wb") as writer:
            shutil.copyfileobj(reader, writer, 1 << 20)
        os.chmod(dst, info.st_mode & 0o777)
    finally:
        os.close(fd)
    return dst


def read_inside(root, rel) -> str:
    """Text of ``root/rel`` read by the root: the file must resolve inside ``root`` and is opened without following
    a final link."""
    root = os.path.realpath(root)
    path = os.path.join(root, rel)
    real = os.path.realpath(path)
    if not real.startswith(root + os.sep):
        raise ValueError(f"{rel} resolves outside its folder")
    fd = os.open(real, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, encoding="utf-8") as handle:
        return handle.read()


# ----- trees a sandboxed user writes (the proposer's workspace, a player's slot) --------------------------------------
# The root never resolves a path inside such a tree: every component is opened relative to its parent's descriptor
# with O_NOFOLLOW, so a link (planted before, or swapped in while the root works) is never followed; links are copied
# as links. ``base`` itself is a root-controlled folder.

DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _parts(rel):
    parts = [p for p in Path(rel).parts if p not in ("", ".")]
    if any(p in ("..", "/") for p in parts):
        raise ValueError(f"bad relative path: {rel}")
    return parts


def open_dir(base, rel="", create=False):
    """Descriptor of the folder ``base/rel``, reached without following a link below ``base`` (missing folders made
    when ``create``)."""
    fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in _parts(rel):
            if create:
                try:
                    os.mkdir(part, 0o755, dir_fd=fd)
                except FileExistsError:
                    pass
            before = os.stat(part, dir_fd=fd, follow_symlinks=False)
            nxt = os.open(part, DIR_FLAGS, dir_fd=fd)
            after = os.fstat(nxt)
            if not (stat.S_ISDIR(before.st_mode) and (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)):
                os.close(nxt)  # a link, or swapped since (some kernels, gVisor among them, follow despite O_NOFOLLOW)
                raise OSError(errno.ELOOP, f"not a plain folder: {part}")
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_untrusted(base, rel, limit=64 * 1024 ** 2) -> bytes:
    """Bytes of the regular file ``base/rel`` (no link followed anywhere below ``base``)."""
    import stat

    parts = _parts(rel)
    dfd = open_dir(base, os.path.join(*parts[:-1]) if len(parts) > 1 else "")
    try:
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
    finally:
        os.close(dfd)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"not a regular file: {rel}")
        os.set_blocking(fd, True)
        return handle.read(limit)


def write_untrusted(base, rel, data: bytes, owner=None, append=False, mode=0o644):
    """Write (or append to) ``base/rel`` without following a link: a link or fifo in the way is replaced."""
    import stat

    parts = _parts(rel)
    dfd = open_dir(base, os.path.join(*parts[:-1]) if len(parts) > 1 else "", create=True)
    name = parts[-1]
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | (os.O_APPEND if append else os.O_EXCL)
        if not append:
            try:
                os.unlink(name, dir_fd=dfd)
            except FileNotFoundError:
                pass
        try:
            fd = os.open(name, flags, mode, dir_fd=dfd)
        except OSError:  # append onto a link or fifo: start the file again
            os.unlink(name, dir_fd=dfd)
            fd = os.open(name, flags | os.O_EXCL, mode, dir_fd=dfd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError(f"not a regular file: {rel}")
            os.set_blocking(fd, True)
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            if owner is not None:
                os.fchown(fd, *owner)
        finally:
            os.close(fd)
    finally:
        os.close(dfd)


def copy_untrusted(base, rel, target, *, link_root_readonly=False, skip=(), limit=None, left_out=None,
                   changed_only=False):
    """Copy the tree ``base/rel`` to ``target`` (a root-owned place) as root, reading only through descriptors:
    folders are entered without following links (``os.fwalk``), files opened with O_NOFOLLOW and copied only when
    regular, links copied as links (never resolved). ``link_root_readonly``: files already root-owned and read-only
    (hard links from the archive) are hard-linked, not copied. ``limit``: bigger files are left out (listed in
    ``left_out``); ``changed_only``: files whose size and mtime match the copy are skipped (checkpoint mirrors).
    Returns the number of files written."""
    import errno
    import stat

    target = Path(target)
    copied = 0
    top = open_dir(base, rel)
    try:
        for root, dirs, files, dfd in os.fwalk(".", dir_fd=top, follow_symlinks=False):
            out = target / root
            out.mkdir(parents=True, exist_ok=True)
            for name in list(dirs):
                if name in skip:
                    dirs.remove(name)
                elif stat.S_ISLNK(os.stat(name, dir_fd=dfd, follow_symlinks=False).st_mode):
                    dirs.remove(name)
                    files.append(name)
            for name in files:
                dst = out / name
                try:
                    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
                except OSError as exc:
                    if exc.errno == errno.ELOOP:  # a link: copied as a link
                        link = os.readlink(name, dir_fd=dfd)
                        if not (changed_only and os.path.islink(dst) and os.readlink(dst) == link):
                            if os.path.lexists(dst):
                                os.unlink(dst)
                            os.symlink(link, dst)
                    continue
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode):
                        continue  # fifos, sockets, devices: not part of a tree
                    relpath = os.path.normpath(os.path.join(root, name))
                    if limit is not None and info.st_size > limit:
                        if left_out is not None:
                            left_out.append({"path": relpath, "bytes": info.st_size})
                        continue
                    if changed_only:
                        try:
                            old = os.stat(dst, follow_symlinks=False)
                            if stat.S_ISREG(old.st_mode) and old.st_size == info.st_size and \
                                    int(old.st_mtime) == int(info.st_mtime):
                                continue
                        except OSError:
                            pass
                    if os.path.lexists(dst):
                        os.unlink(dst)
                    if link_root_readonly and info.st_uid == 0 and not info.st_mode & 0o222:
                        try:
                            os.link(name, dst, src_dir_fd=dfd, follow_symlinks=False)
                            if os.stat(dst, follow_symlinks=False).st_ino == info.st_ino:
                                copied += 1
                                continue
                            os.unlink(dst)
                        except OSError:
                            pass
                    os.set_blocking(fd, True)
                    with open(dst, "wb") as writer:
                        while True:
                            chunk = os.read(fd, 1 << 20)
                            if not chunk:
                                break
                            writer.write(chunk)
                    os.chmod(dst, info.st_mode & 0o777)
                    os.utime(dst, (info.st_atime, info.st_mtime))
                    copied += 1
                finally:
                    os.close(fd)
    finally:
        os.close(top)
    return copied


def move_into(staging, base, rel, name):
    """Rename the root-made folder ``staging`` to ``base/rel/name`` (``rel`` reached without following links; an
    existing entry there is an error)."""
    dfd = open_dir(base, rel)
    try:
        try:
            os.stat(name, dir_fd=dfd, follow_symlinks=False)
            raise ValueError(f"{name} exists")
        except FileNotFoundError:
            pass
        os.rename(staging, name, dst_dir_fd=dfd)
    finally:
        os.close(dfd)


def remove_entry(path):
    """Remove whatever is at ``path`` (link, file or folder) without following a link."""
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def copy_tree(source: Path, target: Path, *, link_big: bool, owner=None, read_only=False):
    """Copy ``source`` to ``target``: big files and node_modules as hard links when ``link_big`` (both sides then share
    a read-only inode), the rest as real copies."""
    source, target = Path(source), Path(target)
    for root, dirs, files in os.walk(source):
        rel = Path(root).relative_to(source)
        (target / rel).mkdir(parents=True, exist_ok=True)
        for name in [d for d in dirs if (Path(root) / d).is_symlink()]:
            dirs.remove(name)  # a link to a folder stays a link (os.walk would make it an empty folder)
            os.symlink(os.readlink(Path(root) / name), target / rel / name)
        for name in files:
            src, dst = Path(root) / name, target / rel / name
            if src.is_symlink():
                os.symlink(os.readlink(src), dst)
                continue
            size = src.stat().st_size
            if link_big and _linked(rel / name, size):
                os.chmod(src, src.stat().st_mode & 0o555)
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)
                if read_only:
                    os.chmod(dst, dst.stat().st_mode & 0o555)
    if owner is not None:
        for root, dirs, files in os.walk(target):
            for name in [root, *(os.path.join(root, d) for d in dirs), *(os.path.join(root, f) for f in files)]:
                if not (os.path.isfile(name) and os.stat(name).st_nlink > 1):
                    os.lchown(name, owner[0], owner[1])


def tree_hash(path: Path, rel: str) -> str:
    """sha256 over the relative paths, sizes and contents of ``path/rel`` (contents of files under 1 GB; inode identity
    and size above, since big files are immutable hard links)."""
    digest = hashlib.sha256()
    base = Path(path) / rel
    if not base.exists():
        return "absent"
    for root, dirs, files in sorted(os.walk(base)):
        dirs.sort()
        for name in sorted(files + [d for d in dirs if (Path(root) / d).is_symlink()]):
            item = Path(root) / name
            stat = item.lstat()
            digest.update(str(item.relative_to(base)).encode() + b"\0" + str(stat.st_size).encode())
            if item.is_symlink():
                digest.update(os.readlink(item).encode())
            elif stat.st_size < 1024 ** 3:
                digest.update(hashlib.sha256(item.read_bytes()).digest())
            else:
                digest.update(f"{stat.st_dev}:{stat.st_ino}".encode())
    return digest.hexdigest()


def serving_hash(path: Path) -> str:
    return tree_hash(path, "serving")


GAME_SECONDS_MAX = 6 * 3600
GAME_TOKENS_MAX = 4_000_000


def check(path: Path) -> list[str]:
    """Problems that stop a candidate before any GPU time: layout, run.json shape, size (docs/rsi.md 1: it must still
    package as an offline Kaggle submission on one RTX Pro 6000; offline play is enforced by the sandbox, the one-GPU
    boot by the stack switch's smoke test)."""
    path = Path(path)
    problems = [f"missing {rel}" for rel in REQUIRED if not (path / rel).exists()]
    links = escaping_links(path)
    if links:
        return problems + [f"symbolic links out of the candidate: {links[:5]}"]
    try:
        run = json.loads(read_inside(path, "player/run.json"))
        if not (isinstance(run.get("argv"), list) and run["argv"] and all(isinstance(a, str) for a in run["argv"])):
            problems.append("player/run.json: argv must be a non-empty list of strings")
        if not isinstance(run.get("env", {}), dict):
            problems.append("player/run.json: env must be an object")
        if not (isinstance(run.get("concurrency", 14), int) and 2 <= run.get("concurrency", 14) <= 97):
            problems.append("player/run.json: concurrency must be 2..97 (play slots + the proposer; 96 slots at most)")
        # the candidate sets its own pool/practice game length (wall seconds and/or a generated-token cap)
        seconds, tokens = run.get("game_seconds"), run.get("game_token_cap")
        if seconds is not None and not (isinstance(seconds, (int, float)) and 60 <= seconds <= GAME_SECONDS_MAX):
            problems.append(f"player/run.json: game_seconds must be 60..{GAME_SECONDS_MAX} (or absent: C x 294.5 s)")
        if tokens is not None and not (isinstance(tokens, int) and 10_000 <= tokens <= GAME_TOKENS_MAX):
            problems.append(f"player/run.json: game_token_cap must be an integer 10000..{GAME_TOKENS_MAX} (or absent)")
    except (OSError, ValueError) as exc:
        problems.append(f"player/run.json: {exc}")
    try:
        if not isinstance(json.loads(read_inside(path, "serving/server.json")), dict):
            problems.append("serving/server.json must be an object")
    except (OSError, ValueError) as exc:
        problems.append(f"serving/server.json: {exc}")
    # docs/rsi.md 1: the player's own /refine stays on (in-game learning is the only kind Kaggle allows). Checked in
    # the settings files Prime Agent reads (global HOME and the game's working folder); a code change to Prime Agent
    # itself is not checked here.
    for rel in ("player/home/.prime/agent/settings.json", "player/workspace/.prime/agent/settings.json"):
        if (path / rel).is_file():
            try:
                settings = json.loads(read_inside(path, rel))
                refine = settings.get("autoRefine") if isinstance(settings, dict) else None
                if isinstance(refine, dict) and refine.get("enabled") is False:
                    problems.append(f"{rel}: autoRefine.enabled false (the player's /refine stays on)")
            except (OSError, ValueError) as exc:
                problems.append(f"{rel}: {exc}")
    total = 0
    seen = set()
    for root, _, files in os.walk(path):
        for name in files:
            stat = (Path(root) / name).lstat()
            if (stat.st_dev, stat.st_ino) not in seen:
                seen.add((stat.st_dev, stat.st_ino))
                total += stat.st_size
    if total > SIZE_CAP:
        problems.append(f"size {total / 1024 ** 3:.1f} GB over {SIZE_CAP / 1024 ** 3:.0f} GB")
    return problems
