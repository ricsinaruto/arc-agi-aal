"""Single-flight volume reads (28 Oct: 11 concurrent `modal volume get/ls` from watch, export and mon on this machine,
several on the same feed, slowed every read to minutes). One fetch per (volume, path) at a time across the processes
of this machine: a caller that finds one running skips this pass and keeps what it has. 28 Oct: one fetcher per file:
``get`` refreshes the machine-wide mirror (export_rl), readers take the mirror while it is fresh (``read``).
CLI for shell polls: ``python -m arc_infra.runops.volfetch VOL PATH [MAX_AGE_S]`` prints the file."""
import fcntl
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

LOCKS = Path(os.environ.get("ARC_VOLFETCH_LOCKS") or "/tmp/arc-volfetch")
MODAL = str(Path(__file__).resolve().parents[3] / ".venv/bin/modal")


SHARED = "arc-prolong-runs"


def route(vol, path):
    """one volume per run: a read of validation/<folder>/... on the shared volume goes to the volume the
    run's record names (records.volume_of; <folder> may be <run>-attempt<N>). Other paths and volumes stay as given."""
    if vol != SHARED:
        return vol
    parts = path.lstrip("/").split("/")
    if len(parts) < 2 or not parts[1] or parts[0] != "validation" or parts[1] == "preflight":
        return vol
    try:
        import re
        from arc_infra.runops import records as R
        return R.volume_of(re.sub(r"-attempt\d+$", "", parts[1]))
    except Exception:
        return vol


def mirror(vol, path):
    """The machine-wide local copy of (vol, path) (vol routed: one copy per real volume)."""
    return LOCKS / "mirror" / route(vol, path) / path


EXEC = LOCKS / "exec.json"  # {"<volume path prefix>": "<app id>"}: read those paths inside the live container


def exec_app(vol, path):
    """The app whose container holds ``path`` live (28 Oct: arc-prolong-runs gets took >50 s while a run's own files
    sit on its mount at /runs/<path>; set with ``python -m arc_infra.runops.volfetch --exec PREFIX APP``)."""
    if vol != SHARED and not vol.startswith("arc-run-"):  # the run's own volume is mounted at /runs in its container
        return None
    try:
        table = __import__("json").loads(EXEC.read_text())
    except (OSError, ValueError):
        table = {}
    return next((app for prefix, app in table.items() if path.lstrip("/").startswith(prefix.strip("/"))), None) or record_app(path)


def record_app(path):
    """validation/<folder>/...: the app of the running record whose current attempt writes <folder> (28 Oct 14:13: the
    volume view of real-v11's sampler was 9 min stale while the container's file was current), else None."""
    parts = path.lstrip("/").split("/")
    if len(parts) < 3 or parts[0] != "validation":
        return None
    try:
        import re
        from arc_infra.runops import records as R
        run = re.sub(r"-attempt\d+$", "", parts[1])
        rec = R.get(run)
        if not rec or R.state_of(rec) != "running" or R.out_id(run, rec=rec) != parts[1]:
            return None
        app = str(((rec.get("ledger") or [{}])[-1].get("row") or {}).get("app_id") or "")  # the current attempt's app only
        return app if app.startswith("ap-") else None
    except Exception:
        return None


def exec_read(app, path, timeout=60, modal=None):
    """``cat /runs/<path>`` inside the app's running container (None when it has none or the read fails)."""
    import json

    listed = run([modal or MODAL, "container", "list", "--json"], timeout=timeout)
    try:
        box = next((c["container_id"] for c in json.loads(listed or "[]") if c.get("app_id") == app), None)
    except ValueError:
        box = None
    if not box:
        return None
    # a bare `cat` exits before Modal's exec stream drains: stdout stopped at 8,192 bytes (28 Oct: SFT's infra rows froze at
    # 0.35 h, real-v10's sampler showed only its first minutes). Size first, then the bytes, then a pause; a short read is
    # a failed read.
    out = run([modal or MODAL, "container", "exec", box, "--", "sh", "-c",
               'n=$(wc -c < "") || exit 1; echo "$n"; cat ""; rc=$?; sleep 1; exit $rc', "_", "/runs/" + path.lstrip("/")],
              timeout=timeout)
    if out is None or "\n" not in out:
        return None
    size, body = out.split("\n", 1)
    try:
        return body if len(body.encode()) >= int(size) else None
    except ValueError:
        return None


def get(vol, path, timeout=120, modal=None):
    """Refresh the mirror of (vol, path) with one `modal volume get` (or a read inside the live container: EXEC);
    True when it did. False when the read failed or another process holds this path's fetch (the mirror keeps its
    last good copy)."""
    vol = route(vol, path)
    local = mirror(vol, path)
    with single_flight(vol, path) as mine:
        if not mine:
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        part = local.with_name(f"{local.name}.{os.getpid()}.part")
        app = exec_app(vol, path)
        if app:
            got = exec_read(app, path, timeout=min(timeout, 60), modal=modal)
            if got:  # an empty read (a file mid-rewrite) never replaces a good copy
                part.write_text(got)
                os.replace(part, local)
                return True
        ok = run([modal or MODAL, "volume", "get", "--force", vol, path, str(part)], timeout=timeout) is not None
        if ok and part.is_file():
            os.replace(part, local)  # readers never see a half-written file
            return True
        part.unlink(missing_ok=True)
        return False


def read(vol, path, max_age=None):
    """Text of the mirror, or None when it is missing or (with ``max_age``) refreshed more than max_age s ago."""
    local = mirror(vol, path)
    try:
        if max_age is not None and time.time() - local.stat().st_mtime > max_age:
            return None
        return local.read_text(errors="replace")
    except OSError:
        return None


def text(vol, path, max_age=150, timeout=120, modal=None):
    """The mirror while fresh, else one fetch into it (single-flight); a skipped or failed fetch returns the last good
    copy, or "" when there is none."""
    fresh = read(vol, path, max_age)
    if fresh is not None:
        return fresh
    get(vol, path, timeout=timeout, modal=modal)
    return read(vol, path) or ""


@contextmanager
def single_flight(vol, path):
    """Yields True while this caller holds the (vol, path) fetch, False when another process holds it."""
    LOCKS.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{vol}:{path}".encode()).hexdigest()[:16]
    handle = open(LOCKS / f"{key}.lock", "w")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True
    finally:
        handle.close()  # releases the lock


def run(argv, timeout=120):
    """stdout of ``argv`` (exit 0) or None. Its own process group, killed whole on a timeout: a `uv run modal` killed
    alone left the modal child reading after the lock was released, so the next pass stacked a second fetch."""
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    except OSError:
        return None
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:  # a grandchild that left the group may still hold the pipe: never wait on it with the lock held
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        return None
    return out if proc.returncode == 0 else None


def copy(vol, path, local, max_age=300, timeout=300, modal=None):
    """``local`` = the mirror while fresh, else after one fetch into it; a skipped or failed fetch copies the last good
    copy. False when there is no copy at all."""
    if read(vol, path, max_age) is None:
        get(vol, path, timeout=timeout, modal=modal)
    src = mirror(vol, path)
    if not src.is_file():
        return False
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, local)
    return True


if __name__ == "__main__":
    if sys.argv[1] == "--exec":  # --exec PREFIX APP (APP "-" removes PREFIX)
        import json

        try:
            table = json.loads(EXEC.read_text())
        except (OSError, ValueError):
            table = {}
        if sys.argv[3] == "-":
            table.pop(sys.argv[2], None)
        else:
            table[sys.argv[2]] = sys.argv[3]
        LOCKS.mkdir(parents=True, exist_ok=True)
        EXEC.write_text(json.dumps(table, indent=1))
        print(json.dumps(table))
        sys.exit(0)
    vol_, path_ = sys.argv[1:3]
    sys.stdout.write(text(vol_, path_, max_age=float(sys.argv[3]) if len(sys.argv) > 3 else 150))
