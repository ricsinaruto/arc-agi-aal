"""R3-through-offload replay check (RL, 26 Oct; ``arc_infra.arc_routed_offload``).

On a server with ``served_route``, the host KV tier (``ARENA_CPU_CACHE_GIB``) and ``ARC_SERVED_ROUTE_FULL=1`` (each
record holds every row, cached ones too), recorded hero requests (the kv-dump probes: v6 game requests at 8K-100K
context) go three times, one at a time, one token each:

A  each probe: its record is the rows the server returned; its chunks are stored in host RAM;
S  after a GPU prefix-cache reset, each probe with its first message changed and ``max_offload_tokens`` 0 (computed
   fresh, host RAM untouched): the freed GPU blocks' slots now hold other rows;
B  after another GPU reset (host tier kept), each probe again, a GPU reset before each: its prefix comes from host RAM.

Pass: every B request with a host hit (``prompt_tokens_details.cached_tokens``) returns exactly A's rows on its cached
positions; at least ``min_hits`` probes hit; S's rows differ from A's on most positions (stale slots would fail).
Run at boot by rl_seq (``mg.redesign.r3_check``: ``boot_check``); writes r3-offload.json.
"""
import concurrent.futures
import contextlib
import json
import time
from pathlib import Path

import httpx

from .served_route import read

BARS = {"min_hits": 6, "scrub_diff": 0.5}
STATUS_GLOB = "r3-offload-*.json"  # arc_routed_offload.write_status, in the server's working dir
ALERT = "ARC_ROUTED_OFFLOAD_ALERT"
SERVER_LOG = "vllm-openai-server.log"
FULL_FLAG = "full.on"  # arc_infra.served_route_runtime.FULL_FLAG


# ---------------------------------------------------------------------------- the server's bound state (any R3 run)
def offload_on(study, trial=None):
    """served_route with the host KV tier: the routed rows must travel with the host chunks."""
    server = (study or {}).get("server") or {}
    env = (trial or {}).get("server_env") or {}
    gib = env.get("ARENA_CPU_CACHE_GIB", ((study or {}).get("cache_experiment") or {}).get("cpu_cache_gib") or 0)
    try:
        return bool(server.get("served_route")) and int(gib) > 0
    except (TypeError, ValueError):
        return False


def server_status(server_dirs):
    """The engine cores' status files and the logs' ALERT lines under ``server_dirs``."""
    files, alerts, stub = [], [], False
    for d in server_dirs:
        d = Path(d)
        for p in sorted(d.glob(STATUS_GLOB)):
            try:
                files.append({"file": p.name, **json.loads(p.read_text())})
            except (OSError, ValueError):
                continue
        log = d / SERVER_LOG
        if log.is_file():
            text = log.read_text(errors="replace")
            stub = stub or "STUB_CORPUS" in text  # the CPU legs' replay stub: no engine, no connector
            alerts += [line.strip()[:400] for line in text.splitlines() if ALERT in line]
    refused = {}
    for f in files:
        for k, v in f.items():
            if k.startswith("refused_") and isinstance(v, (int, float)):
                refused[k[len("refused_"):]] = refused.get(k[len("refused_"):], 0) + v
    bound = None if not files else float(all(f.get("bound") for f in files))
    return {"files": files, "bound": bound, "refused": refused, "alerts": alerts, "stub": stub,
            "restored_tokens": sum(f.get("restored_tokens", 0) for f in files),
            "saved": sum(f.get("saved", 0) for f in files)}


def status_metrics(st):
    """Exporter keys (train-metrics rows): infra.r3_offload_bound 0/1, refusals by reason, restored tokens."""
    m = {"infra.r3_offload_bound": st["bound"] if st["bound"] is not None else 0.0,
         "infra.r3_offload_restored_tokens": float(st["restored_tokens"]), "infra.r3_offload_saved": float(st["saved"])}
    m.update({f"infra.r3_offload_refused.{k}": float(v) for k, v in st["refused"].items()})
    return m


def status_alert(st):
    """None when bound (or a stub server); else the alert text: unbound, an ALERT line, or no status at all."""
    if st["stub"]:
        return None
    why = [f.get("why") for f in st["files"] if not f.get("bound") and f.get("why")]
    if st["alerts"] or why:
        return "R3 offload unbound, every host hit recomputed: " + "; ".join(why or st["alerts"])[:300]
    if not st["files"]:
        return "R3 offload: no status from the server (rows hook not installed?): host hits recomputed"
    if not st["bound"]:
        return "R3 offload not bound: host hits recomputed"
    return None


def watch(study, trial, server_dirs, *, emit, metrics_path, wait_s=120.0, poll=5.0, sleep=time.sleep):
    """After play starts (rl_seq) or a check ends (kv-score): the bound state as a train-metrics row (the watcher's
    metrics_remote) and, when not bound, a feed ``alert`` (``emit("alert", why=...)``: the run's verdict fails and the
    row's alert fires watch.py's run-alert rule). Waits up to ``wait_s`` for the status file (written at the first
    schedule). Returns the status (None when the host tier is off)."""
    if not offload_on(study, trial):
        return None
    end = time.time() + wait_s
    st = server_status(server_dirs)
    while not st["files"] and not st["alerts"] and not st["stub"] and time.time() < end:
        sleep(poll)
        st = server_status(server_dirs)
    alert = status_alert(st)
    row = {"t": time.time(), "m": status_metrics(st), **({"alert": alert} if alert else {})}
    try:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass
    if alert:
        emit("alert", why=alert, r3_offload={k: st[k] for k in ("bound", "refused", "alerts")})
    return {**st, "alert": alert}


@contextlib.contextmanager
def full_records(route_dir):
    """Full records (cached rows too) for requests that finish while held (served_route_runtime reads the flag)."""
    flag = Path(route_dir) / FULL_FLAG
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("r3 check\n")
    try:
        yield
    finally:
        flag.unlink(missing_ok=True)


def closing(directory):
    return {p.name[:-3] for p in Path(directory).glob("*.rt")}


def wait_record(directory, before, timeout=60.0, api=None):
    """The record a request left: by its API id (records are ``<API id>-<8 characters>``; servers checked at once
    share the directory), else the one new record (requests go one at a time)."""
    end = time.time() + timeout
    while time.time() < end:
        new = ({p.name[:-3] for p in Path(directory).glob(f"{api}-*.rt")} if api else closing(directory) - before)
        if new:
            rid = sorted(new)[0]
            settle = time.time() + 5
            while not (Path(directory) / f"{rid}.json").is_file() and time.time() < settle:
                time.sleep(0.05)  # the runtime's closing file (the stub writes none)
            return rid, read(Path(directory) / f"{rid}.rt")
        time.sleep(0.05)
    return None, None


def scrubbed(body):
    messages = [dict(m) for m in body["messages"]]
    first = messages[0]
    if isinstance(first.get("content"), str):
        first["content"] = "[scrub] " + first["content"]
    else:
        first["content"] = [{"type": "text", "text": "[scrub] "}] + list(first.get("content") or [])
    return {**body, "messages": messages}


def compare(a, b, upto):
    """Rows of ``b`` vs ``a`` on positions [0, upto): equal flag and mismatching row count."""
    if a is None or b is None:
        return {"equal": False, "mismatch_rows": None}
    n = min(upto, a["hi"], b["hi"])
    same = (a["ids"][:n] == b["ids"][:n]).all(axis=(1, 2)) & a["covered"][:n] & b["covered"][:n]
    return {"equal": bool(n == upto and same.all()), "mismatch_rows": int(n - same.sum()), "rows": int(n)}


def diff_share(a, b):
    if a is None or b is None:
        return None
    n = min(a["hi"], b["hi"])
    return float(((a["ids"][:n] != b["ids"][:n]).reshape(n, -1).any(axis=1)).mean()) if n else None


def run_check(spec, *, base_url, model, output, route_dir, log=print, transport=None, bars=BARS, cleanup=False,
              flag=True):
    """``flag``: set the route dir's full-record flag around the requests (False: the caller holds it, e.g. servers
    checked at once). ``spec["probes_max"]``: the first n probes only."""
    if not route_dir:
        raise ValueError("r3_offload needs study.server.served_route")
    probes = [json.loads(line) for line in open(spec["r3_offload"])][:spec.get("probes_max") or None]
    root = base_url.rstrip("/").removesuffix("/v1")
    began = time.time()
    with httpx.Client(base_url=root + "/", timeout=1800, trust_env=False, transport=transport) as http:
        def reset(external=False):
            for _ in range(30):
                r = http.post("reset_prefix_cache", params={"reset_external": "true" if external else "false"})
                r.raise_for_status()
                if r.json().get("success") is True:
                    return
                time.sleep(1)
            raise RuntimeError("prefix cache reset refused")

        def send(body, no_store=False):
            body = {"model": model, **body, "max_tokens": 1, "temperature": 0.0}
            if no_store:
                body["kv_transfer_params"] = {"max_offload_tokens": 0}
            before = closing(route_dir)
            r = http.post("v1/chat/completions", json=body)
            if r.status_code == 400 and no_store:  # a front end without kv_transfer_params: S stores too
                body.pop("kv_transfer_params")
                r = http.post("v1/chat/completions", json=body)
            r.raise_for_status()
            reply = r.json()
            usage = reply.get("usage") or {}
            rid, rows = wait_record(route_dir, before, float(spec.get("record_wait_s", 60)), api=reply.get("id"))
            return {"rid": rid, "rows": rows, "prompt": usage.get("prompt_tokens"),
                    "cached": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)}

        def body_of(p):
            return {k: v for k, v in (("messages", p["body"]["messages"]),
                                      ("chat_template_kwargs", p["body"].get("chat_template_kwargs")),
                                      ("tools", p["body"].get("tools"))) if v is not None}

        held = full_records(route_dir) if flag else contextlib.nullcontext()
        with held:
            reset(external=True)
            first = [send(body_of(p)) for p in probes]
            time.sleep(float(spec.get("settle_s", 3)))  # the last stores finish (the engine steps while it has push work)
            reset()
            scrub = [send(scrubbed(body_of(p)), no_store=True) for p in probes]
            again = []
            for p in probes:
                reset()
                again.append(send(body_of(p)))
            if cleanup:  # an in-run check: its records are no play data, and the caches start empty
                reset(external=True)
    if cleanup:
        for item in first + scrub + again:
            for suffix in (".rt", ".mm.json", ".json"):
                if item["rid"]:
                    (Path(route_dir) / f"{item['rid']}{suffix}").unlink(missing_ok=True)
    rows, hits, ok = [], 0, True
    for p, a, s, b in zip(probes, first, scrub, again):
        full = a["rows"] is not None and a["rows"]["lo"] == 0
        row = {"label": p["label"], "prompt": b["prompt"], "a_cached": a["cached"], "cached": b["cached"],
               "full_record": full, "scrub_diff": diff_share(a["rows"], s["rows"]),
               "restored": compare(a["rows"], b["rows"], b["cached"])}
        if a["rows"] is not None and b["rows"] is not None:  # info: positions B computed itself, vs A's
            n = min(a["rows"]["hi"], b["rows"]["hi"])
            c = min(b["cached"], n)
            same = (a["rows"]["ids"][c:n] == b["rows"]["ids"][c:n]).all(axis=(1, 2))
            row["recompute_agree"] = float(same.mean()) if n > c else None
        if b["cached"] > 0:
            hits += 1
            ok = ok and row["restored"]["equal"]
        ok = ok and full
        rows.append(row)
        log(json.dumps({"event": "kv_score_r3", **row}, default=str), flush=True)
    diffs = sorted(r["scrub_diff"] for r in rows if r["scrub_diff"] is not None)
    median = diffs[len(diffs) // 2] if diffs else None
    ok = bool(ok and hits >= int(spec.get("min_hits", bars["min_hits"]))
              and median is not None and median >= bars["scrub_diff"])
    result = {"ok": ok, "probes": len(rows), "hits": hits, "scrub_diff_median": median, "bars": bars,
              "seconds": round(time.time() - began, 1), "per_probe": rows}
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "r3-offload.json").write_text(json.dumps(result, indent=1, default=str))
    return result


# ---------------------------------------------------------------------------- rl_seq boot stage (mg.redesign.r3_check)
def boot_check(servers, spec, folder, *, model, route_dir, status_dirs, log=None, transport=None):
    """Before play (H5, 26 Oct): the A/S/B replay on every server at once (each request's record found by its API
    id in the shared route dir; the full-record flag held here until every server is done; each server removes only
    its own records and resets only its own caches), then the bound state from the status files. ``spec``:
    {"probes": path, "min_hits": 6, "probes_max": 6}. Row: ok = every server's check ok and infra.r3_offload_bound
    == 1; a replay stub (CPU legs) runs the path and is never judged. Never raises; a failed row is the caller's
    feed alert. ``transport``: a list (one per server) or one for all (tests)."""
    folder = Path(folder)
    began = time.time()
    row = {"servers": {}, "errors": {}}
    stub = server_status(status_dirs)["stub"]
    names = [getattr(s, "name", None) or f"server{i + 1}" for i, s in enumerate(servers)]
    transports = transport if isinstance(transport, (list, tuple)) else [transport] * len(servers)
    check = {"r3_offload": spec["probes"], **({"record_wait_s": 5} if stub else {}),  # CPU legs
             **{k: v for k, v in spec.items() if k != "probes"}}

    def one(i):
        return run_check(check, base_url=servers[i].root + "/v1", model=model, output=folder / names[i],
                         route_dir=route_dir, log=log or (lambda *a, **k: None), transport=transports[i],
                         cleanup=True, flag=False)

    try:
        if not route_dir:
            raise ValueError("r3_check needs study.server.served_route")
        with full_records(route_dir), concurrent.futures.ThreadPoolExecutor(max(1, len(servers))) as pool:
            jobs = {pool.submit(one, i): name for i, name in enumerate(names)}
            for job, name in jobs.items():
                try:
                    got = job.result()
                    row["servers"][name] = {k: got[k] for k in ("ok", "probes", "hits", "scrub_diff_median")}
                except Exception as exc:  # noqa: BLE001 - recorded
                    row["errors"][name] = f"{type(exc).__name__}: {exc}"[:300]
    except Exception as exc:  # noqa: BLE001 - recorded
        row["errors"]["check"] = f"{type(exc).__name__}: {exc}"[:300]
    st = server_status(status_dirs)
    row.update(bound=st["bound"], refused=st["refused"], stub=stub or st["stub"],
               metrics=status_metrics(st), seconds=round(time.time() - began, 1))
    checked = bool(row["servers"]) and not row["errors"] and all(v["ok"] for v in row["servers"].values())
    row["ok"] = True if row["stub"] else bool(checked and st["bound"] == 1)
    if not row["ok"]:
        row["why"] = ("r3_check: " + ("; ".join(f"{k}: {v}" for k, v in row["errors"].items()) or
                                      f"bound {st['bound']}, servers " + json.dumps(row["servers"])))[:400]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "verdict.json").write_text(json.dumps(row, indent=1, default=str))
    return row
