"""Boot timeline (stage H, 26 Oct): launch to first generated token, by phase, for the feed's ``seq_boot`` row (the
10-min check reads it) and ``seq_first_token``.

Zero is the container's start (ARC_RL1_STARTED_EPOCH, set by modal_rl1.execute). Sources: the launcher's
``bootstrap.log`` (prefetch, prebuilt venv, runtime cache, boot cache rows), the serving setup's provenance
(setup start, runtime extraction) and the vLLM server log (its own phase lines and timestamps). Every field is
optional: a missing source leaves its keys out, never fails the loop."""
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

# "(Worker pid=1545) INFO 10-26 10:18:52 [default_loader.py:430] ..." (vLLM) and
# "2026-10-26 10:22:39,202 - INFO - autotuner.py:852 - ..." (FlashInfer)
VLLM_TS = re.compile(r"(?:INFO|WARNING|ERROR|DEBUG) (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) ")
FI_TS = re.compile(r"(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d),\d+ - ")
LAST = {  # seconds vLLM reports itself (last occurrence)
    "engine_init_s": r"init engine \(profile, create kv cache, warmup model\) took ([0-9.]+) s",
    "graphs_s": r"Graph capturing finished in ([0-9.]+) secs",
    "model_load_s": r"Model loading took [0-9.]+ GiB memory and ([0-9.]+) seconds",
}
EVERY = {  # every occurrence, in order (target then draft; backbone then eagle head)
    "weights_s": r"Loading weights took ([0-9.]+) seconds",
    "dynamo_s": r"Dynamo bytecode transform time: ([0-9.]+) s",
    "compile_s": r"torch.compile took ([0-9.]+) s in total",
    "warmup_s": r"Initial profiling/warmup run took ([0-9.]+) s",
}
MARKS = {  # first line carrying the text: its time from the server's first log line
    "engine_core": "Initializing a V1 LLM engine",
    "load_start": "Loading model from scratch",
    "autotune_start": "Autotuning process starts",
    "autotune_end": "Autotuning process ends",
    "capture_start": "Capturing CUDA graphs",
    "api_ready": "Starting vLLM server on",
}


def stamp(line, year):
    """Epoch (UTC) of a log line, or None."""
    m = VLLM_TS.search(line)
    if m:
        mo, d, h, mi, s = map(int, m.groups())
        return datetime(year, mo, d, h, mi, s, tzinfo=timezone.utc).timestamp()
    m = FI_TS.search(line)
    if m:
        y, mo, d, h, mi, s = map(int, m.groups())
        return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc).timestamp()
    return None


def server_phases(text, zero=None):
    """vLLM's phase seconds from its log; with ``zero`` (an epoch) also ``start_t`` / ``ready_t`` against it."""
    out = {}
    for key, pattern in LAST.items():
        found = re.findall(pattern, text)
        if found:
            out[key] = float(found[-1])
    for key, pattern in EVERY.items():
        found = re.findall(pattern, text)
        if found:
            out[key] = [float(v) for v in found]
    year = datetime.fromtimestamp(zero or time.time(), timezone.utc).year
    first, marks = None, {}
    for line in text.splitlines():
        t = stamp(line, year)
        if t is None:
            continue
        if first is None:
            first = t
        for key, needle in MARKS.items():
            if key not in marks and needle in line:
                marks[key] = t
    if first is not None:
        out.update({f"{k}_at": round(v - first, 1) for k, v in marks.items()})
        if "autotune_start" in marks and "autotune_end" in marks:
            out["autotune_s"] = round(marks["autotune_end"] - marks["autotune_start"], 1)
        if zero:
            out["start_t"] = round(first - zero, 1)
            if "api_ready" in marks:
                out["ready_t"] = round(marks["api_ready"] - zero, 1)
    out["aot_loaded"] = "Directly load" in text or "load the compiled graph" in text
    return out


def events(text, names):
    """The last JSON row of each event name in a launcher log (lines that are JSON objects with "event")."""
    rows = {}
    for line in text.splitlines():
        line = line.strip()
        if not (line.startswith("{") and '"event"' in line):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("event") in names:
            rows[row["event"]] = {k: v for k, v in row.items() if k != "event"}
    return rows


def read(path):
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


LAUNCH_EVENTS = ("weights_prefetched", "prebuilt_venv", "boot_cache_restore", "boot_cache_carry", "host_plan")


def timeline(server_dir, environ=None):
    """The boot timeline so far: seconds from the container's start (``*_t``) and per phase (``*_s``)."""
    environ = os.environ if environ is None else environ
    zero = float(environ.get("ARC_RL1_STARTED_EPOCH") or 0) or None
    out = {"started_epoch": zero}
    server_dir = Path(server_dir)
    output = environ.get("ARC_VALIDATION_OUTPUT")
    launcher = read(Path(output) / "bootstrap.log") if output else ""
    if launcher:
        out.update(events(launcher, LAUNCH_EVENTS))
        runtime = re.findall(r"ARC_RUNTIME_CACHE_RESTORED (\{.*\})", launcher)
        if runtime:
            try:
                out["runtime_cache_s"] = json.loads(runtime[-1]).get("seconds")
            except ValueError:
                pass
    try:
        prov = json.loads(read(server_dir / "vllm-setup-provenance.json") or "{}")
    except ValueError:
        prov = {}
    if zero and prov.get("setup_started_epoch"):
        out["setup_t"] = round(float(prov["setup_started_epoch"]) - zero, 1)
    if zero and prov.get("completed_epoch"):
        out["setup_done_t"] = round(float(prov["completed_epoch"]) - zero, 1)
    log = read(server_dir / "vllm-openai-server.log")
    if log:
        out["server1"] = server_phases(log, zero)
    return out


def extra_phases(folders, zero=None):
    """Each extra server's phases (its own vllm.log)."""
    return [server_phases(read(Path(f) / "vllm.log"), zero) for f in folders]


def since_launch(environ=None):
    environ = os.environ if environ is None else environ
    zero = float(environ.get("ARC_RL1_STARTED_EPOCH") or 0)
    return round(time.time() - zero, 1) if zero else None


async def first_token(urls, baseline, timeout=900.0, poll=2.0, get=None):
    """Seconds until any server's generated-token counter passes its baseline (None on timeout). ``baseline``:
    {url: tokens} read when play began (the cache probe and warm-ups generate before it)."""
    import asyncio

    began = time.time()
    while time.time() - began < timeout:
        for url in urls:
            tokens = await generated(url, get)
            if tokens is not None and tokens > baseline.get(url, 0.0):
                return round(time.time() - began, 1)
        await asyncio.sleep(poll)
    return None


async def generated(url, get=None):
    """vllm:generation_tokens_total of one server (its /v1 base or root), or None."""
    root = url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        if get is None:
            import httpx

            async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                text = (await client.get(root + "/metrics")).text
        else:
            text = await get(root + "/metrics")
    except Exception:  # noqa: BLE001 - no metrics: no reading
        return None
    total = None
    for line in text.splitlines():
        if line.startswith("vllm:generation_tokens_total"):
            total = (total or 0.0) + float(line.rsplit(" ", 1)[1])
    return total
