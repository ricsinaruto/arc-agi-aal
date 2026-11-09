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


def submit(name, note=""):
    return _call("submit", name=name, note=note)


def withdraw(candidate=None, job=None):
    return _call("withdraw", candidate=candidate, job=job)


def practice(candidate, games=None, n=4, seconds=None):
    return _call("practice", candidate=candidate, games=games, n=n, seconds=seconds)


def self_update(path, note=""):
    return _call("self_update", path=str(path), note=note)


def end_session():
    return _call("end_session")


def meta_history(diff_lines=400):
    return _call("meta_history", diff_lines=diff_lines)
