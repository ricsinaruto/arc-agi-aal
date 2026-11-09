"""Volume ops for the trainer child. The trainer runs in the image's axolotl venv, which has no ``modal``, so the Modal
function (parent) serves the child's requests over a pipe pair (``ARC_VOL_FDS``): commit, reload, and read (a
committed volume file's text straight from the server; a reload fails while any file on the volume is open, so live
knobs are read this way)."""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path


def run_child(args, env, ops, cwd=None, log_path=None, log_commit_s=60.0):
    """Run ``args`` with ``ops`` ({name: fn(*args) -> JSON-able or None}) served to its ``client()``; returns the
    CompletedProcess. ``log_path``: the child's stdout + stderr also go line by line to that file, committed every
    ``log_commit_s`` and at the child's end, so a crash keeps its whole log (Modal keeps only the last lines)."""
    req_r, req_w = os.pipe()
    ack_r, ack_w = os.pipe()
    pipe = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0) if log_path else {}
    proc = subprocess.Popen(args, env={**env, "ARC_VOL_FDS": f"{req_w},{ack_r}"}, pass_fds=(req_w, ack_r), cwd=cwd, **pipe)
    os.close(req_w)
    os.close(ack_r)
    lock = threading.Lock()  # a commit at a time: the child's and the log's
    if log_path:
        ops = {**ops, "commit": lambda *a, inner=ops["commit"]: _locked(lock, inner, *a)}
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        ended = threading.Event()

        def append(data):  # open per write: a file left open on the volume fails reloads
            with open(log_path, "ab") as f:
                f.write(data)

        def tee():  # a helper process can outlive the child and hold the pipe: joined with a bound below
            for line in proc.stdout:
                append(line)
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()

        def save():
            while not ended.wait(log_commit_s):
                _save_log(ops)

        teeing, saving = threading.Thread(target=tee, daemon=True), threading.Thread(target=save, daemon=True)
        teeing.start()
        saving.start()

    def serve():
        with os.fdopen(req_r) as req, os.fdopen(ack_w, "w", buffering=1) as ack:
            for line in req:
                try:
                    op, *a = json.loads(line)
                    reply = {"ok": True, "value": ops[op](*a)}
                except Exception as error:  # noqa: BLE001 - raised again in the child, where the caller decides
                    reply = {"ok": False, "error": repr(error)[:500]}
                try:
                    ack.write(json.dumps(reply) + "\n")
                except BrokenPipeError:
                    return

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    code = proc.wait()
    thread.join(timeout=60)
    if log_path:
        ended.set()
        teeing.join(timeout=30)
        saving.join(timeout=60)
        append(f"\nCHILD EXIT {code}\n".encode())
        _save_log(ops)
    return subprocess.CompletedProcess(args, code)


def _locked(lock, fn, *a):
    with lock:
        return fn(*a)


def _save_log(ops):
    try:
        ops["commit"]()
    except Exception as error:  # noqa: BLE001 - a failed log commit must not touch the child
        print(f"LOG COMMIT FAILED {error!r}", flush=True)


def modal_ops(volume, mount="/runs"):
    """The parent's ops on a Modal volume mounted at ``mount``; read takes a container path, None if absent."""
    def read(path):
        rel = os.path.relpath(path, mount)
        try:
            return b"".join(volume.read_file(rel)).decode()
        except FileNotFoundError:
            return None
        except Exception as error:  # noqa: BLE001 - modal's not-found error type varies by client version
            if "not found" in str(error).lower() or "no such file" in str(error).lower():
                return None
            raise

    return {"commit": volume.commit, "reload": volume.reload, "read": read}


def client():
    """(commit, reload, read) callables for the child; thread-safe; outside ``run_child`` commit and reload are no-ops
    and read reads the local file."""
    fds = os.environ.get("ARC_VOL_FDS")
    if not fds:
        def local(path):
            try:
                with open(path) as f:
                    return f.read()
            except FileNotFoundError:
                return None

        return (lambda: None), (lambda: None), local
    w, r = fds.split(",")
    req, ack = os.fdopen(int(w), "w", buffering=1), os.fdopen(int(r))
    lock = threading.Lock()

    def ask(op, *args):
        with lock:
            req.write(json.dumps([op, *args]) + "\n")
            line = ack.readline()
        reply = json.loads(line) if line else {"ok": False, "error": "no reply"}
        if not reply["ok"]:
            raise RuntimeError(f"volume {op}: {reply['error']}")
        return reply["value"]

    return (lambda: ask("commit")), (lambda: ask("reload")), (lambda path: ask("read", path))


# --- per-run volumes (one v2 volume per run) ---------------------------------------------------------
# A config with "volume": "run" mounts arc-run-<run> at /runs (its own output, validation/<run>/...), the shared v2
# arc-cache at /cache (compile caches) and every volume it names as /in/<volume>/... read-only (sft-data, another run's
# checkpoint). launch/sft/launch.sh exports the result as ARC_SFT_VOLUME / ARC_SFT_INPUT_VOLUMES; modal_sft refuses a
# config whose plan differs from that env. A config without "volume" keeps today's mounts (arc-prolong-runs at /runs).
SHARED_VOLUME = "arc-prolong-runs"
CACHE_DIRS = ("compile-cache",)


def run_volume_name(run):
    """arc-run-<run> within Modal's name rules (as main's runops.records.run_volume_name)."""
    import hashlib
    import re
    base = "arc-run-" + re.sub(r"[^A-Za-z0-9._-]", "-", run)
    return base if len(base) <= 64 else base[:55] + "-" + hashlib.sha1(run.encode()).hexdigest()[:8]


def _strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _strings(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _strings(x)


def volume_plan(spec):
    """(run volume, sorted input volumes, problems) of a config (its "what" is prose and not read)."""
    import re
    body = {k: v for k, v in spec.items() if k != "what"}
    mode = spec.get("volume")
    if mode is None:
        return SHARED_VOLUME, [], []
    if mode != "run":
        return SHARED_VOLUME, [], [f'volume {mode!r}: "run" (the run\'s own volume) or absent']
    run, probs = spec.get("volume_run") or spec["run"], []  # volume_run: legs of one job share the first run's volume
    for s in _strings(body):
        if not s.startswith("/runs/"):
            continue
        rel = s[len("/runs/"):]
        head, _, rest = rel.partition("/")
        if head in CACHE_DIRS:
            probs.append(f"{s}: a cache on a per-run volume is /cache/{rel}")
        elif not (head == "validation" and (rest.split("/")[0] == run or rest.split("/")[0].startswith(run + "-"))):
            probs.append(f"{s}: not this run's folder; on a per-run volume it is /in/<its volume>/{rel} "
                         f"(sft-data and older runs: /in/{SHARED_VOLUME}/{rel}; records.py volume RUN)")
    inputs = sorted({m for s in _strings(body) for m in re.findall(r"^/in/([A-Za-z0-9._-]+)/", s)})
    return run_volume_name(run), inputs, sorted(set(probs))


if __name__ == "__main__":  # launch.sh: `python vol.py env CONFIG` -> shell exports, or the problems and exit 1
    import sys
    if sys.argv[1:2] != ["env"] or len(sys.argv) != 3:
        raise SystemExit("usage: vol.py env CONFIG")
    vol, ins, problems = volume_plan(json.load(open(sys.argv[2])))
    if problems:
        print("VOLUME PLAN REFUSED:\n  " + "\n  ".join(problems), file=sys.stderr)
        raise SystemExit(1)
    print(f"ARC_SFT_VOLUME={vol}\nARC_SFT_INPUT_VOLUMES={','.join(ins)}")
