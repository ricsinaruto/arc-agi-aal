"""Fail-closed Linux filesystem isolation and seccomp process/network sandbox.

Landlock is preferred; isolated user/mount/network namespaces provide the
compatible fallback on gVisor hosts. The parent retains the
environment; children only inherit a private RPC descriptor and permitted files.
"""

import ctypes
import ctypes.util
import os
import platform
import signal
import sys
from pathlib import Path


def restrict(read=(), write=()):
    if sys.platform != "linux":
        raise RuntimeError(
            "Agent execution requires the Linux sandbox; use the CPU container locally"
        )
    if root := os.environ.pop("ARC_CHROOT", None):
        cwd = os.getcwd()
        os.chroot(root)
        os.chdir(cwd if Path(cwd).exists() else "/")
        libc = ctypes.CDLL(None, use_errno=True)
        for cap in range(64):
            if libc.prctl(24, cap, 0, 0, 0) and ctypes.get_errno() != 22:
                raise OSError(ctypes.get_errno(), "chroot capability bounding set")
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
        # Some container runtimes populate inheritable capabilities. setuid
        # clears effective/permitted but does not necessarily clear inheritable.
        header = (ctypes.c_uint32 * 2)(0x20080522, 0)
        data = (ctypes.c_uint32 * 6)()
        if libc.capset(header, data):
            raise OSError(ctypes.get_errno(), "chroot clear capabilities")
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0):
            raise OSError(ctypes.get_errno(), "chroot no_new_privs")
        _seccomp()
        return "chroot"
    libc = ctypes.CDLL(None, use_errno=True)
    # These syscall numbers are shared by supported x86_64 and aarch64 Linux.
    if platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("Unsupported sandbox architecture")
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 3:
        _namespace_restrict(read, write)
        _seccomp()
        return "namespaces"
    handled = (1 << 15) - 1  # Includes REFER and TRUNCATE.

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    attr = Ruleset(handled)
    fd = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "Landlock ruleset")
    try:
        import sysconfig

        directories = [sysconfig.get_path("stdlib"), sysconfig.get_path("purelib")]
        for path, access in (
            [(p, 1 << 3) for p in directories]
            + [(p, (1 << 0) | (1 << 2) | (1 << 3)) for p in read]
            + [(p, handled) for p in [*write, "/dev/null"]]
        ):
            path = Path(path).resolve()
            if not path.exists():
                continue
            if not path.is_dir():
                access &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
            handle = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(access, handle)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), f"Landlock path {path}")
            finally:
                os.close(handle)
        if libc.prctl(38, 1, 0, 0, 0) or libc.syscall(446, fd, 0):
            raise OSError(ctypes.get_errno(), "Landlock enforcement")
    finally:
        os.close(fd)
    _seccomp()
    return "landlock"


def _seccomp():
    seccomp = ctypes.CDLL(ctypes.util.find_library("seccomp") or "libseccomp.so.2")
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = seccomp.seccomp_init(0x7FFF0000)  # ALLOW, then deny capability acquisition/escape.
    if not ctx:
        raise RuntimeError("seccomp initialization failed")
    try:
        for name in (
            "socket",
            "socketpair",
            "connect",
            "bind",
            "listen",
            "accept",
            "accept4",
            "ptrace",
            "process_vm_readv",
            "process_vm_writev",
            "pidfd_getfd",
            "pidfd_send_signal",
            "kill",
            "tkill",
            "tgkill",
            "setsid",
            "setpgid",
            "unshare",
            "setns",
            "mount",
            "chroot",
            "pivot_root",
            "move_mount",
            "open_tree",
            "fsopen",
            "fsmount",
            "mount_setattr",
            "capset",
            "umount2",
            "bpf",
            "keyctl",
            "open_by_handle_at",
            "io_uring_setup",
        ):
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and seccomp.seccomp_rule_add(ctx, 0x50000 | 1, number, 0):
                raise RuntimeError(f"seccomp rule failed: {name}")
        if seccomp.seccomp_load(ctx):
            raise RuntimeError("seccomp enforcement failed")
    finally:
        seccomp.seccomp_release(ctx)


def _namespace_restrict(read, write):
    """Construct an empty filesystem with explicit binds, then discard capabilities.

    Called only by the trusted child before executing model code. If any step
    fails, the child exits without running user code. No external network or
    host /proc mount is made visible.
    """
    import tempfile

    libc = ctypes.CDLL(None, use_errno=True)
    libc.mount.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_ulong,
        ctypes.c_char_p,
    ]

    def check(result, operation):
        if result != 0:
            raise OSError(ctypes.get_errno(), "Namespace sandbox: " + operation)

    uid, gid = os.getuid(), os.getgid()
    cwd = Path.cwd()
    # Resolve all inputs before changing namespaces/root. Only trusted parent
    # paths enter this function; the model cannot add mounts.
    entries = {}
    for path in read:
        p = Path(path).resolve()
        if p.exists():
            entries[str(p)] = False
    for path in write:
        p = Path(path).resolve()
        if p.exists():
            entries[str(p)] = True
    entries["/dev/null"] = True
    aliases = {}
    for path in ["/bin", "/sbin", "/lib", "/lib64", sys.executable]:
        p = Path(path)
        if p.exists() and str(p) != str(p.resolve()):
            aliases[str(p)] = str(p.resolve())
    jail = Path(tempfile.mkdtemp(prefix="arc-sandbox-"))
    check(libc.unshare(0x10000000), "create user namespace")
    Path("/proc/self/setgroups").write_text("deny")
    Path("/proc/self/uid_map").write_text(f"0 {uid} 1")
    Path("/proc/self/gid_map").write_text(f"0 {gid} 1")
    check(libc.unshare(0x00020000 | 0x40000000), "create mount/network namespaces")
    check(libc.mount(None, b"/", None, 16384 | 262144, None), "private mounts")
    check(libc.mount(b"tmpfs", os.fsencode(jail), b"tmpfs", 2, b"size=16m"), "empty root")
    # Ancestors first, so more-specific mounts can override a parent's policy.
    for source, writable in sorted(entries.items(), key=lambda item: len(Path(item[0]).parts)):
        target = jail / source.lstrip("/")
        if Path(source).is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.touch()
        check(
            libc.mount(os.fsencode(source), os.fsencode(target), None, 4096, None), "bind " + source
        )
        flags = 4096 | 32 | 2 | (0 if writable else 1)
        check(libc.mount(None, os.fsencode(target), None, flags, None), "protect " + source)
    for alias, target in aliases.items():
        dest = jail / alias.lstrip("/")
        if not dest.exists() and not dest.is_symlink():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(target)
    check(libc.mount(None, os.fsencode(jail), None, 32 | 1 | 2, None), "read-only root")
    os.chdir(jail)
    os.chroot(".")
    os.chdir(str(cwd) if any(cwd.is_relative_to(Path(p)) for p in entries) else "/")
    check(libc.prctl(38, 1, 0, 0, 0), "no new privileges")
    # An empty bounding set plus no_new_privs prevents capability regain on
    # exec, including for uid 0. gVisor does not support all securebits flags.
    for capability in range(64):
        result = libc.prctl(24, capability, 0, 0, 0)
        if result and ctypes.get_errno() != 22:
            check(result, "drop capability bounding set")

    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [
            ("effective", ctypes.c_uint32),
            ("permitted", ctypes.c_uint32),
            ("inheritable", ctypes.c_uint32),
        ]

    check(libc.capset(ctypes.byref(Header(0x20080522, 0)), (Data * 2)()), "clear capabilities")


def runtime_paths():
    """Allow standard runtime files, never the entire dependency/repository tree."""
    import sysconfig

    import numpy

    stdlib = Path(sysconfig.get_path("stdlib"))
    paths = [p for p in stdlib.iterdir() if p.name not in {"site-packages", "dist-packages"}]
    np_path = Path(numpy.__file__).parent
    return [
        str(p)
        for p in [
            *paths,
            np_path,
            np_path.parent / "numpy.libs",
            Path(sys.executable).resolve(),
            Path(sys.prefix) / "pyvenv.cfg",
            Path("/usr/bin"),
            Path("/bin"),
            Path("/lib"),
            Path("/lib64"),
            Path("/usr/lib"),
            Path("/usr/local/lib/libpython3.12.so.1.0"),
            Path("/etc/ld.so.cache"),
            Path("/dev/null"),
            Path("/dev/urandom"),
        ]
    ]


def kill_group(process):
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
