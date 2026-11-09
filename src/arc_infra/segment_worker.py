"""Host segmentation off the fleet's GIL (v7.1, 6 October).

``segment_host`` is pure python (~0.1-0.7 s a board under a busy fleet) and ran for every executed action in a thread
of the arena process, holding the GIL that 110 games' threads and the event loop share. ``SegmentPool`` runs it in a
few worker processes (``python -m arc_infra.segment_worker``: length-prefixed pickles over pipes); same function, same
result; the caller's thread waits on a pipe with the GIL released. A broken pool falls back to in-process for good.
"""

import os
import pickle
import queue
import struct
import subprocess
import sys
import threading

HEADER = struct.Struct(">I")


def _read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise EOFError("segment worker closed")
        data.extend(chunk)
    return data


def _read(stream):
    head = _read_exact(stream, HEADER.size)
    size = HEADER.unpack(head)[0]
    return pickle.loads(_read_exact(stream, size))


def _write(stream, value):
    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    remaining = memoryview(HEADER.pack(len(data)) + data)
    while remaining:
        written = stream.write(remaining)
        if not written:
            raise EOFError("segment worker closed")
        remaining = remaining[written:]
    stream.flush()


class SegmentPool:
    def __init__(self, workers):
        self.workers = workers
        self.idle = queue.Queue()
        self.started = 0
        self.lock = threading.Condition()
        self.failed = None
        self.calls = 0

    def _spawn(self):
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
        return subprocess.Popen([sys.executable, "-m", "arc_infra.segment_worker"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, bufsize=0)

    def _take(self):
        with self.lock:
            while True:
                if self.failed:
                    raise RuntimeError(self.failed)
                try:
                    return self.idle.get_nowait()
                except queue.Empty:
                    pass
                if self.started < self.workers:
                    try:
                        process = self._spawn()
                    except BaseException as exc:
                        self.failed = f"{type(exc).__name__}: {exc}"
                        self.lock.notify_all()
                        raise
                    self.started += 1
                    return process
                self.lock.wait()

    def segment(self, board, limit):
        import numpy as np

        process = self._take()
        try:
            _write(process.stdin, (np.asarray(board, dtype=np.uint8), int(limit)))
            kind, value = _read(process.stdout)
        except BaseException as exc:
            with self.lock:
                self.started -= 1
                self.failed = f"{type(exc).__name__}: {exc}"
                self.lock.notify_all()
            process.kill()
            raise
        with self.lock:
            retired = bool(self.failed)
            if not retired:
                self.idle.put(process)
                self.lock.notify()
            self.calls += 1
        if retired:
            process.kill()
        if kind == "error":
            raise RuntimeError(value)
        return value

    def close(self):
        with self.lock:
            self.failed = self.failed or "segment pool closed"
            self.lock.notify_all()
        while True:
            try:
                process = self.idle.get_nowait()
            except queue.Empty:
                return
            process.kill()


def compact(objects):
    """Node boundaries as tuples of (row, col) tuples (same values; v7.1: the cached segmentations of 110 games were
    ~1M GC-tracked two-element lists, most of every gen-2 pause)."""
    for node in objects.get("nodes") or ():
        if isinstance(node.get("boundary"), list):
            node["boundary"] = tuple(map(tuple, node["boundary"]))
    return objects


def main():
    from arc_infra.objects import segment_host

    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        try:
            board, limit = _read(stdin)
        except EOFError:
            return
        try:
            _write(stdout, ("ok", compact(segment_host(board, limit=limit))))
        except Exception as exc:  # noqa: BLE001 - reported to the host, which falls back
            _write(stdout, ("error", f"{type(exc).__name__}: {exc}"[:300]))


if __name__ == "__main__":
    main()
