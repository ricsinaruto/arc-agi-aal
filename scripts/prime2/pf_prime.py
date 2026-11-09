"""prime2 launch gate on Modal CPU: the exact bundle's study and trial through run_arc2 (-> arc2.prime), as arena.run
calls it, on linux with the bundle's own Prime Agent build, node and kernel wheelhouse (prepare_runtime), against the
scripted server (tests/prime2_fake_model.py) and the served cache layout of a real base18 run. Sandbox off (gVisor
refuses namespaces; the isolation probe runs inside the real run before any session).
  A  ENTRIES entries of the roster to a token mark: every entry answers, compaction + refine + rlm child requests,
     generated = charged, read-out at the marks;
  B  a stop request after ~200 replies: ends by the request, report and tasks.jsonl written;
  C  one session's process killed mid-run: resumed with --continue, the entry answers again;
  D  failing requests (EngineDead 500 every 7th) + a killed session, hot set smaller than the roster, prefill limit;
  E  a dead server ends the trial.
v4 bundles (prime.conv): the fake plays conv1's workflow (tests/prime2_fake_model.py CONV_STEPS); an entry "answers"
by submit(), which closes its session; A runs until every session has closed and checks conv1's first message
(the pictures), the harness's replies and the read-out of the submits.
usage: PF_BUNDLE=DIR modal run --detach scripts/prime2/pf_prime.py --layout-run RUN"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import modal
import modal.exception

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/")
app = modal.App("prime2-preflight")
BUNDLE = Path(os.environ.get("PF_BUNDLE", "/bundle")).resolve()
ENTRIES_TASKS = 12  # x the study's copies
image = (modal.Image.debian_slim(python_version="3.12").apt_install("xz-utils", "procps")
         .pip_install("numpy==2.2.6", "scipy==1.15.3", "httpx==0.28.1", "aiohttp==3.14.3"))
if modal.is_local():
    image = (image.add_local_dir(str(BUNDLE), "/bundle", ignore=["wheels", "draft-head", "**/__pycache__"])
             .add_local_file(str(ROOT / "tests/prime2_fake_model.py"), "/pf/prime2_fake_model.py"))


@app.function(image=image, timeout=2700, retries=0, cpu=(8, 16), nonpreemptible=True)
def phases(layout_line, deadline=None):
    import asyncio
    import signal
    import socket
    import sys
    import threading

    if deadline is not None and time.time() > deadline - 2400:
        return {"pass": False, "checks": {"restart_within_ceiling": False}}
    sys.path[:0] = ["/bundle/src", "/pf"]
    os.environ.pop("ARENA_STOP_REQUEST", None)
    os.environ["ARC_REAL_NVIDIA_SMI"] = "true"
    import prime2_fake_model as fm
    from arc2 import prime
    from arc2.trial import run_arc2

    study = json.loads(Path("/bundle/configs/comparison.json").read_text())
    base = study["trials"][0]
    work = Path("/tmp/pf")
    server = work / "server"
    server.mkdir(parents=True, exist_ok=True)
    (server / "server.log").write_text(layout_line + "\n")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    conv = prime.settings(base, study)["conv"]
    cut = conv and prime.settings(base, study).get("keep_cut_replies", False)  # v5: one cut reply per entry in A
    fake = fm.serve_in_thread(port, conv=conv, cut=cut)
    url = f"http://127.0.0.1:{port}/v1"
    checks, info = {}, {}
    began = time.time()
    tasks = base["tasks"][:ENTRIES_TASKS]
    over = {"sandbox": False, "start_stagger_seconds": 0.2, "idle_seconds": 5}
    done = (lambda r: r["status"] == "submitted") if conv else (lambda r: bool(r["answers"]))
    until = 10 ** 9 if conv else None  # conv: sessions close by themselves

    def play(name, mark, stop_after=None, kill_after=None, url=url, extra=None):
        out = work / name
        trial = {**base, "tasks": tasks, "marks": {"half": mark // 2, "all": mark}, "wall_seconds": 900,
                 "curve_step": mark // 4, "prime": {**base["prime"], **over, **(extra or {})}}
        if stop_after:
            def stopper():
                while True:
                    ev = out / "events.jsonl"
                    if ev.exists() and len(ev.read_text().splitlines()) >= stop_after:
                        (out / "stop-request").write_text("{}")
                        return
                    time.sleep(0.2)
            threading.Thread(target=stopper, daemon=True).start()
        killed = {}
        if kill_after:
            def killer():
                while True:
                    ev = out / "events.jsonl"
                    if ev.exists() and len(ev.read_text().splitlines()) >= kill_after:
                        logs = sorted((out / "sessions").glob("*/log.jsonl"))
                        for log in logs:
                            rows = [json.loads(x) for x in log.read_text().splitlines()]
                            pid = next((r["pid"] for r in rows if r["event"] == "session_start"), None)
                            if pid:
                                os.killpg(pid, signal.SIGKILL)
                                killed.update(entry=log.parent.name, pid=pid)
                                return
                    time.sleep(0.2)
            threading.Thread(target=killer, daemon=True).start()
        t0 = time.time()
        report = asyncio.run(run_arc2(study, trial, source="/bundle", output=out, server_dir=server, base_url=url,
                                      deadline_epoch=time.time() + 1500))
        rows = [json.loads(x) for x in (out / "tasks.jsonl").read_text().splitlines()]
        return report, rows, round(time.time() - t0, 1), killed

    entries = len(tasks) * int(base.get("copies", 1))
    # A
    report, rows, seconds, _ = play("A", until or 400 * entries)
    p = report["prime"]
    info["A"] = {"seconds": seconds, "status": report["status"], "end": report["end_reason"], "entries": len(rows),
                 "answered": sum(done(r) for r in rows), "tokens": report["completion_tokens"],
                 "server_generated": report.get("server_generated_tokens"), "requests": p["requests"],
                 "rpc_events": p["rpc_events"], "marks": {k: v["right"] for k, v in report["roster"]["marks"].items()},
                 "runtime": report["runtime"], "settled": p["entries_settled"], "errors": p["errors"]}
    kinds = {r["kind"] for r in fake.requests}
    checks["A_complete"] = report["status"] == "complete" and report["end_reason"] == (
        "sessions_ended" if conv else "last_mark")
    checks["A_all_answered"] = all(done(r) for r in rows) and len(rows) == entries
    checks["A_generated_eq_charged"] = report.get("server_generated_tokens") == report["completion_tokens"] or (
        abs((report.get("server_generated_tokens") or 0) - report["completion_tokens"]) <= 20 * 3)
    if conv:
        task_map = {t: json.loads((Path("/bundle") / study["arc2"]["data"] / base["split"] / f"{t}.json").read_text())
                    for t in tasks}
        pictures = sorted(len(t["train"]) + len(t["test"]) for t in task_map.values() for _ in range(2))
        info["A"].update(harness_replies=p.get("harness_replies"), first_requests=len(fake.first_requests),
                         compaction_releases=report["gate"]["compaction_releases"], verified=p["verified_programs"])
        checks["A_conv_first_message"] = (sorted(n for _, n in fake.first_requests) == pictures and all(
            s.startswith(prime.CONV_SYSTEM[:200]) for s, _ in fake.first_requests))
        checks["A_conv_workflow"] = (all(r["harness_replies"].get("nothing", 0) >= 1 and r["harness_replies"].get(
            "program", 0) == 1 for r in rows) and report["gate"]["compaction_releases"] >= entries // 2
            and p["entries_submitted"] == entries and report["sessions_live"] == 0)
        if cut:
            info["A"].update(cuts=fake.cuts, cut_kept=p.get("cut_kept"), cut_restored=p.get("cut_restored"))
            checks["A_cut_reply_kept"] = (fake.cuts == entries and p.get("cut_kept") == entries
                                          and p.get("cut_restored", 0) >= entries)
        checks["A_compaction_refine_child"] = ({"summary"} <= kinds
                                               and p["rpc_events"].get("compaction_end", 0) >= entries // 2)
        checks["A_marks_exact"] = True
    else:
        checks["A_compaction_refine_child"] = ({"summary", "child", "refine"} <= kinds
                                               and p["rpc_events"].get("compaction_end", 0) >= entries // 2)
    checks["A_sampling"] = all((r["temperature"], r["top_p"], r["top_k"]) == (study["temperature"], 0.95, 20)
                               and r["max_tokens"] <= study["completion_cap"] for r in fake.requests)
    checks.setdefault("A_marks_exact", all(v["reached"] for v in report["roster"]["marks"].values()))
    checks["A_live_keys"] = all(k in report for k in ("completion_tokens", "tokens_per_second", "gate", "gpu",
                                                      "final", "test_outputs", "arm_tokens"))
    # B
    rb, rowsb, sb, _ = play("B", 10 ** 9, stop_after=100 if conv else 200)
    info["B"] = {"seconds": sb, "status": rb["status"], "end": rb["end_reason"], "tokens": rb["completion_tokens"]}
    checks["B_stopped"] = rb["end_reason"] == "stop_request" and len(rowsb) == entries and sb < 600
    # C
    rc, rowsc, sc, killed = play("C", until or 400 * entries, kill_after=60)
    hit = next((r for r in rowsc if r["task"].replace("#", "_") == killed.get("entry")), None)
    info["C"] = {"seconds": sc, "killed": killed, "restarts": rc["prime"]["restarts"], "end": rc["end_reason"],
                 "killed_row": {k: hit[k] for k in ("restarts", "session_exits", "status", "spent")} if hit else None}
    checks["C_resumed"] = bool(hit) and hit["restarts"] == 1 and rc["status"] == "complete" and done(hit)
    # D: run 2's classes: failing requests (EngineDead 500 every 7th) + a killed session, hot set smaller than the
    # roster, prefill limit; E: a dead server ends the trial
    def fake_at(**kw):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            p = probe.getsockname()[1]
        return fm.serve_in_thread(p, **kw), f"http://127.0.0.1:{p}/v1"
    fd, url_d = fake_at(fail_every=7, first_delay=0.05, conv=conv)
    rd, rowsd, sd, killed_d = play("D", until or 250 * entries, kill_after=60, url=url_d,
                                   extra={"hot_limit": 6, "quantum_tokens": 300, "hot_idle_seconds": 5,
                                          "prefill_slots": 2, **({"quantum_tokens": 0} if conv else {})})
    info["D"] = {"seconds": sd, "end": rd["end_reason"], "gate": rd["gate"], "fake_failed": fd.failed,
                 "prefill_max": fd.prefill_max, "answered": sum(done(r) for r in rowsd),
                 "unanswered": [{k: r[k] for k in ("task", "spent", "restarts", "status", "errors", "requests")}
                                for r in rowsd if not done(r)], "killed": killed_d,
                 "spent_min_max": [min(r["spent"] for r in rowsd), max(r["spent"] for r in rowsd)]}
    checks["D_faults_survived"] = (rd["end_reason"] == ("sessions_ended" if conv else "last_mark")
                                   and all(done(r) for r in rowsd)
                                   and fd.failed > 5 and rd["gate"]["dropped"] >= fd.failed // 2
                                   and fd.prefill_max <= 2 and rd["gate"]["cold_admitted"] > 6)
    fe, url_e = fake_at(down_at=40, conv=conv)
    re_, _, se, _ = play("E", 10 ** 9, url=url_e, extra={"server_dead_seconds": 10})
    info["E"] = {"seconds": se, "end": re_["end_reason"], "status": re_["status"]}
    checks["E_dead_server_ends"] = re_["end_reason"] == "server_dead" and re_["status"] == "error" and se < 300
    info["seconds"] = round(time.time() - began, 1)
    out = {"pass": all(checks.values()), "checks": checks, "info": info}
    print("PF_RESULT " + json.dumps(out), flush=True)
    return out


@app.local_entrypoint()
def main(layout_run: str):
    sha = hashlib.sha256((BUNDLE / "MANIFEST.json").read_bytes()).hexdigest()
    log = subprocess.run(["rclone", "cat", f"nebius-arc:arc-runs-bucket/runs/validation/{layout_run}/arena/trials/c240/"
                          "vllm-openai-server.log"], capture_output=True, text=True).stdout
    layout = next(line for line in log.splitlines() if "ARC_CACHE_LAYOUT" in line)
    layout = layout[layout.index("ARC_CACHE_LAYOUT"):]
    call = phases.spawn(layout, time.time() + 2700)  # spawn + re-attach: a held .remote() dies with the client
    print("CALL", call.object_id, flush=True)
    result, misses = None, 0
    while result is None:
        try:
            result = call.get(timeout=60)
        except (TimeoutError, modal.exception.TimeoutError):
            continue
        except (OSError, modal.exception.ConnectionError) as exc:
            misses += 1
            if misses > 30:
                raise
            print("reattach after", type(exc).__name__, flush=True)
            time.sleep(20)
            call = modal.FunctionCall.from_id(call.object_id)
    record = {**result, "bundle": str(BUNDLE), "manifest_sha256": sha, "layout_from": layout_run, "epoch": time.time()}
    print(json.dumps(record, indent=1)[:6000])
    path = Path(os.environ.get("PF_RECORDS", "/tmp")) / f"preflight-prime2-{sha[:16]}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(path)
