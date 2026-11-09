"""Sandboxing of the sessions (players and the proposer) on the loop VM, enforced from root.

- One unprivileged uid per play slot (``rsip00``...) and one for the proposer (``rsiprop``); the broker runs game code as
  ``rsigame``. Loop data (state, meters, pool and held-out games and results, other sessions' folders) is root- or
  owner-only (mode 0700), so a session reads only its own folder and what is shared with it on purpose.
- Network: loopback only, and on loopback not the model server's own port (all model traffic goes through the metering
  proxy). Everything else is rejected and counted (``egress_counts``).
- Each session sees its folder at ``/workspace`` (a private mount namespace), as Prime Agent's runner expects.
"""
from __future__ import annotations

import os
import pwd
import secrets
import shlex
import shutil
import subprocess
from pathlib import Path

PROPOSER = "rsiprop"
CHECK = "rsichk"     # a new proposer setup's check session (apart from the main session's user)
SERVE = "rsiserve"   # a serving stack that is not gen 0's (it runs the proposer's server code)
VET = "rsigvet"      # vets of games the proposer writes
PRIVATE = ("/tmp", "/var/tmp", "/dev/shm")


def slot_user(slot):
    return f"rsip{slot:02d}"


def game_user(slot):
    """The game code of one slot's broker (one user per slot: no broker reaches another)."""
    return f"rsig{slot:02d}"


def ensure_user(name):
    try:
        return pwd.getpwnam(name)
    except KeyError:
        subprocess.run(["useradd", "--system", "--no-create-home", "--shell", "/usr/sbin/nologin", name], check=True)
        return pwd.getpwnam(name)


def _rules(uid, blocked_ports):
    rules = [["-o", "lo", "-p", proto, "--dport", str(port), "-j", "REJECT"] for port in blocked_ports
             for proto in ("tcp", "udp")]
    return rules + [["-o", "lo", "-j", "ACCEPT"], ["-j", "REJECT"]]


def close_network(users, blocked_ports):
    """Loopback only (minus ``blocked_ports``) for each user, IPv4 and IPv6; raises when the rules cannot be set."""
    for tool in ("iptables", "ip6tables"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} missing: sessions cannot be kept off the network")
        for name in users:
            uid = str(ensure_user(name).pw_uid)
            for rule in _rules(uid, blocked_ports):
                spec = ["OUTPUT", "-m", "owner", "--uid-owner", uid, *rule]
                if subprocess.run([tool, "-C", *spec], capture_output=True).returncode != 0:
                    subprocess.run([tool, "-A", *spec], check=True)


def egress_counts(users):
    """Rejected outbound packets per user (non-loopback REJECT rule counters, IPv4 + IPv6)."""
    uids = {str(pwd.getpwnam(name).pw_uid): name for name in users}
    out = dict.fromkeys(users, 0)
    for tool in ("iptables", "ip6tables"):
        listing = subprocess.run([tool, "-L", "OUTPUT", "-v", "-x", "-n"], capture_output=True, text=True).stdout
        for line in listing.splitlines():
            parts = line.split()
            if len(parts) > 3 and parts[2] == "REJECT" and "owner" in line and " lo " not in f" {line} ":
                uid = line.split("UID match")[-1].split()[0] if "UID match" in line else None
                if uid in uids and "dpt:" not in line:
                    out[uids[uid]] += int(parts[0])
    return out


def own(path, user, mode=0o700):
    """``user`` owns ``path`` and can write everything in it (copies from the read-only archive keep a-w modes)."""
    entry = pwd.getpwnam(user)
    for root, dirs, files in os.walk(path):
        for name in [root, *(os.path.join(root, d) for d in dirs), *(os.path.join(root, f) for f in files)]:
            if os.path.islink(name):
                os.lchown(name, entry.pw_uid, entry.pw_gid)
                continue
            if os.stat(name).st_nlink > 1 and not os.path.isdir(name):
                continue  # a hard link into the archive stays root's and read-only
            os.lchown(name, entry.pw_uid, entry.pw_gid)
            os.chmod(name, os.stat(name).st_mode | (0o700 if os.path.isdir(name) else 0o600))
    os.chmod(path, mode)


def _under(path, roots):
    return any(str(path) == r or str(path).startswith(r.rstrip("/") + "/") for r in roots)


FORWARDER = """
import asyncio, sys
SOCK, PORT = sys.argv[1], int(sys.argv[2])
async def pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass
async def handle(reader, writer):
    try:
        up_reader, up_writer = await asyncio.open_unix_connection(SOCK)
    except Exception:
        writer.close()
        return
    await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))
async def main():
    server = await asyncio.start_server(handle, "127.0.0.1", PORT)
    async with server:
        await server.serve_forever()
asyncio.run(main())
"""
PROXY_SOCKET_DIR = "/proxy"


def mem_total():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def systemd_scopes():
    return shutil.which("systemd-run") is not None and Path("/run/systemd/system").is_dir()


def limited(argv, name, memory):
    """``argv`` in its own transient systemd scope with a memory ceiling (cgroup v2 MemoryMax, no swap): a session
    that runs away is killed alone, never the server. Without systemd (the test container) the ceiling is absent."""
    if not memory or not systemd_scopes():
        return argv
    return ["systemd-run", "--scope", "--quiet", "--collect", f"--unit=rsi-{name}-{secrets.token_hex(4)}",
            "-p", f"MemoryMax={int(memory)}", "-p", "MemorySwapMax=0", "--", *argv]


def command(argv, user, workspace, env, cwd="/workspace", extra_binds=(), hide=(), tmp=None, tmp_binds=(),
            shm_size="1g", tmp_size="4g", net="none", proxy=None, python="/opt/rsi-runtime/kernel/bin/python",
            pid=True, memory=None):
    """argv run as ``user`` in its own mount and IPC namespaces:

    1. private /tmp, /var/tmp, /dev/shm and /dev/mqueue: empty tmpfs, or ``tmp`` (a host folder) at /tmp with
       ``tmp_binds`` ((source, path under /tmp, read_only)) bound into it first;
    2. ``workspace`` (if any) at /workspace, then ``extra_binds`` ((source, target, read_only));
    3. every path in ``hide`` covered by an empty read-only tmpfs (paths under the private folders are gone already);
    4. every other mount read-only (the host's world-writable folders, /run/lock and the like, carry nothing between
       sandboxes).

    ``net``: "none" (own network namespace, nothing but its own loopback), "proxy" (the same, plus 127.0.0.1:<port>
    forwarded to the metering proxy's unix socket; ``proxy`` = (socket folder, port)), "host" (the host's loopback,
    under the iptables owner rules). No session shares a loopback, a socket namespace or IPC with another.
    Returns the root-side argv for ``asyncio.create_subprocess_exec``."""
    q = shlex.quote
    lines = ["set -e"]
    writable = ["/tmp", "/var/tmp", "/dev/shm", "/dev/mqueue"]
    # 1. binds first (their sources may live under the host's /tmp, which the private /tmp covers next)
    if workspace is not None:
        lines += ["mkdir -p /workspace", f"mount --bind {q(str(workspace))} /workspace"]
        writable.append("/workspace")
    binds = list(extra_binds)
    if net == "proxy":
        binds.append((proxy[0], PROXY_SOCKET_DIR, False))
    for source, target, read_only in binds:
        lines.append(f"mkdir -p {q(str(target))} 2>/dev/null || true")
        lines.append(f"mount --bind {q(str(source))} {q(str(target))}")
        if read_only:
            lines.append(f"mount -o remount,bind,ro {q(str(target))}")
        else:
            writable.append(str(target))
    # 2. private /tmp (or the given folder with its own binds), /var/tmp, /dev/shm, /dev/mqueue
    if tmp is not None:
        for source, rel, read_only in tmp_binds:
            target = f"{str(tmp).rstrip('/')}/{str(rel).strip('/')}"
            lines.append(f"mkdir -p {q(target)}")
            lines.append(f"mount --bind {q(str(source))} {q(target)}")
            if read_only:
                lines.append(f"mount -o remount,bind,ro {q(target)}")
        lines.append(f"mount --rbind {q(str(tmp))} /tmp")
    else:
        lines.append(f"mount -t tmpfs -o size={tmp_size},mode=1777,nosuid,nodev rsi-tmp /tmp")
    lines.append("mount -t tmpfs -o size=1g,mode=1777,nosuid,nodev rsi-vartmp /var/tmp")
    lines.append(f"mount -t tmpfs -o size={shm_size},mode=1777,nosuid,nodev rsi-shm /dev/shm")
    lines.append("mount -t mqueue rsi-mqueue /dev/mqueue 2>/dev/null || true")
    hide = [str(p) for p in hide]
    for path in hide:
        # 13 Oct: v13's player probe died mounting /assets/games after /assets was already covered (no mount point)
        if any(other != path and _under(path, [other]) for other in hide):
            continue
        if not _under(path, PRIVATE):
            lines.append(f"mount -t tmpfs -o ro,size=4k,mode=755 rsi-hidden {q(str(path))}")
    keep = "|".join(q(w) for w in writable) + "|" + "|".join(q(w.rstrip("/") + "/") + "*" for w in writable)
    lines.append("for m in $(awk '{print }' /proc/self/mountinfo); do case \"$m\" in *\\\\*|/proc|/proc/*|/sys|"
                 f"/sys/*|/dev|/dev/pts|/dev/pts/*|{keep}) continue;; esac; "
                 "mount -o remount,bind,ro \"$m\" 2>/dev/null || true; done")
    drop = (f"setpriv --reuid={user} --regid={user} --init-groups --no-new-privs --inh-caps=-all "
            f"--bounding-set=-all --")
    if net in ("none", "proxy"):
        lines.append(f"{shutil.which('ip') or '/usr/sbin/ip'} link set lo up")
    if net == "proxy":
        lines.append(f"{drop} {q(python)} -I -c {q(FORWARDER)} {PROXY_SOCKET_DIR}/proxy.sock {int(proxy[1])} "
                     "</dev/null >/dev/null 2>&1 &")
        lines.append("sleep 0.3")
    exports = " ".join(f"{k}={q(str(v))}" for k, v in env.items())
    lines.append(f"cd {q(cwd)}")
    lines.append(f"exec env -i {exports} {drop} " + " ".join(q(str(a)) for a in argv))
    flags = ["--mount", "--ipc", *(["--net"] if net in ("none", "proxy") else []),
             *(["--pid", "--fork", "--kill-child=SIGKILL", "--mount-proc"] if pid else [])]
    return limited(["unshare", *flags, "--propagation", "private", "--", "/bin/sh", "-c", "\n".join(lines)], user,
                   memory)
