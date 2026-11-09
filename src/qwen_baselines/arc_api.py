"""Public synchronous client copied into C's read-only API directory."""

import fcntl
import json
import os
import socket


def _call(method, **args):
    # Separate opens serialize concurrent clients sharing the inherited descriptor.
    with open(".arc-rpc-lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sock = socket.socket(fileno=os.dup(int(os.environ["ARC_RPC_FD"])))
        with sock, sock.makefile("rwb", buffering=0) as channel:
            channel.write((json.dumps({"method": method, **args}) + "\n").encode())
            reply = json.loads(channel.readline())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["result"]


def observe(grid=False):
    """Metadata of the latest observation; ``grid=True`` adds ``grid`` and ``frames``."""
    return _call("observe", grid=grid)


def step(action, x=None, y=None, grid=False):
    return _call("step", action=action, x=x, y=y, grid=grid)


def transition():
    """The latest action's transition summary (changed cells, objects moved/appeared...)."""
    return _call("transition")
