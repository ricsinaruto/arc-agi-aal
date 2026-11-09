"""The proposer's handle on the loop: requests to the root controller over /workspace/.rsi/sock (see AGENTS.md)."""
import json
import socket

SOCKET = "/workspace/.rsi/sock"


def _call(op, **args):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(600)
        s.connect(SOCKET)
        s.sendall(json.dumps({"op": op, **args}).encode() + b"\n")
        raw = b""
        while not raw.endswith(b"\n"):
            chunk = s.recv(1 << 20)
            if not chunk:
                raise RuntimeError("loop controller disconnected")
            raw += chunk
    reply = json.loads(raw)
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error", "refused"))
    return reply.get("value")


def status():
    return _call("status")


def fork(parent, name):
    return _call("fork", parent=parent, name=name)


def submit(name, note="", hold=False):
    """hold=True: archived and smoke-tested, but not evaluated on the pool until release(candidate)."""
    return _call("submit", name=name, note=note, hold=hold)


def release(candidate):
    return _call("release", candidate=candidate)


def adopt_stack(candidate, gpu_job=None):
    return _call("adopt_stack", candidate=candidate, gpu_job=gpu_job)


def practice(candidate, games=None, n=2, seconds=None):
    """seconds: this play's wall (at least 60, at most the candidate's game wall); None: the game wall."""
    return _call("practice", candidate=candidate, games=games, n=n, seconds=seconds)


def add_game(path, role="practice"):
    return _call("add_game", path=str(path), role=role)


def self_update(path, note=""):
    """Installs a copy of /workspace/meta/ for your NEXT session; note: why (kept in meta_history)."""
    return _call("self_update", path=str(path), note=note)


def end_session():
    """End this session now; the next one starts at once under the live setup."""
    return _call("end_session")


def meta_history(diff_lines=400):
    """Every setup version: note, status, diff against its parent, its sessions, candidates made under it."""
    return _call("meta_history", diff_lines=diff_lines)
