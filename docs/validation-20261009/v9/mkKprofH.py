"""kprofH: v9's HOST side, profiled with py-spy. The kprofP tree (exact v9 v1 serving + the
bench-only range/prefill points) plus:
- tools/py-spy (0.4.1 manylinux wheel, sha256 6a80ec05...) and a "pyspy" point option: py-spy record (raw collapsed
  stacks, idle frames, threads, 200 Hz) of every vLLM process (API server, engine core, worker) over the window;
- a "turns" point: play-like turns on N streams (each: send the context, max_tokens R, append the reply and a short
  user message, resend), so every request re-renders, re-tokenizes and re-hashes a long, mostly cached prompt.
Points: PFH64x4 (cold prefill of 4 x ~64K, py-spy over the prefill), D44 (pure decode, 44 x ~64K, py-spy 20 s),
TURNS44 (44 streams of turns, py-spy 60 s after a 120 s warm-up).
Usage: mkKprofH.py CONTEXTS PYSPY_BINARY"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

D = Path("./.claude/worktrees/v84sub/dist")
BASE = D / "b2-kprofP-v9-v1"
contexts, spy_bin = Path(sys.argv[1]), Path(sys.argv[2])
SPY_SHA8 = "e7c2de2d"  # the binary inside py_spy-0.4.1 manylinux x86_64 (wheel sha256 6a80ec05...)
assert hashlib.sha256(spy_bin.read_bytes()).hexdigest()[:8] == SPY_SHA8, "py-spy binary differs from the checked one"
out = D / "b2-kprofH-v9-v1"
if out.exists():
    shutil.rmtree(out)
shutil.copytree(BASE, out, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "dataset-upload"))
(out / "tools").mkdir(exist_ok=True)
shutil.copyfile(spy_bin, out / "tools/py-spy")
(out / "tools/py-spy").chmod(0o755)

kp = out / "src/arena/kernel_profile.py"
t = kp.read_text()
MEASURE = '''    async def measure(point):
        bodies, counts = sets[point["content"]]
'''
SPY = '''    async def spy_start(label, seconds):
        # kprofH (30 Oct): py-spy of every serving process over the window (raw collapsed stacks, idle frames kept)
        exe = Path(__file__).resolve().parents[2] / "tools/py-spy"
        os.chmod(exe, 0o755)
        out_dir = Path(bench.output) / f"pyspy-{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        procs = []
        for role, pids in serving_processes().items():
            for pid in pids[:4]:
                p = await asyncio.create_subprocess_exec(
                    str(exe), "record", "--pid", str(pid), "--duration", str(int(seconds)), "--rate", "200",
                    "--format", "raw", "--output", str(out_dir / f"{role}-{pid}.txt"), "--nonblocking", "--idle",
                    "--threads", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                procs.append((role, pid, p))
        return procs

    async def spy_end(procs, record):
        res = {}
        for role, pid, p in procs:
            try:
                got, _ = await asyncio.wait_for(p.communicate(), 120)
                res[f"{role}-{pid}"] = {"rc": p.returncode, "tail": got.decode(errors="replace")[-300:]}
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                res[f"{role}-{pid}"] = {"error": repr(exc)[:200]}
        record["pyspy"] = res

''' + MEASURE
PF_OLD = '''            tasks = [asyncio.create_task(one(i, b)) for i, b in enumerate(requests)]
            await asyncio.sleep(float(point.get("lead_seconds", 3)))
'''
PF_NEW = '''            spy = await spy_start(point["label"], float(point["pyspy"])) if point.get("pyspy") else None
            tasks = [asyncio.create_task(one(i, b)) for i, b in enumerate(requests)]
            await asyncio.sleep(float(point.get("lead_seconds", 3)))
'''
PF_END_OLD = '''            await asyncio.gather(*tasks)
            record["prefill_seconds"] = round(time.monotonic() - started, 3)
'''
PF_END_NEW = '''            await asyncio.gather(*tasks)
            record["prefill_seconds"] = round(time.monotonic() - started, 3)
            if spy is not None:
                await spy_end(spy, record)
'''
TASKS_OLD = '''        tasks = [asyncio.create_task(one(i, b)) for i, b in enumerate(requests)]
        stop, injector, injected, inject_tasks = asyncio.Event(), None, [], []
'''
TASKS_NEW = '''        if point.get("turns"):  # kprofH (30 Oct): play-like turns (re-render, re-tokenize, re-hash a long prompt)
            tcfg = point["turns"]
            filler = " ".join(["look at the board and plan the next move with care"] * int(tcfg.get("user_repeat", 30)))
            turn_stats = record.setdefault("turn_stats", {"turns": 0, "reply_tokens": 0, "errors": 0})

            async def turns(i, body):
                msgs = list(body["messages"])
                base = {k: v for k, v in body.items() if k != "messages"}
                base["max_tokens"] = int(tcfg.get("max_tokens", 800))
                grown = 0
                while True:
                    reservation = await gate.acquire(f"kprof-{point['label']}-{i}", sizes[i] + grown + 2048)
                    try:
                        r = await client.post(bench.base_url + "/chat/completions", json={**base, "messages": msgs})
                    finally:
                        gate.release(reservation)
                    if r.status_code != 200:
                        turn_stats["errors"] += 1
                        record.setdefault("failures", []).append(r.text[:300])
                        await asyncio.sleep(1)
                        continue
                    d = r.json()
                    m = d["choices"][0]["message"]
                    text = m.get("content") or (m.get("reasoning_content") or "")[:4000] or "ok"
                    msgs = msgs + [{"role": "assistant", "content": text}, {"role": "user", "content": filler}]
                    grown += int(d["usage"]["completion_tokens"]) + 400
                    turn_stats["turns"] += 1
                    turn_stats["reply_tokens"] += int(d["usage"]["completion_tokens"])

            tasks = [asyncio.create_task(turns(i, bodies[c])) for i, c in enumerate(chosen)]
        else:
            tasks = [asyncio.create_task(one(i, b)) for i, b in enumerate(requests)]
        stop, injector, injected, inject_tasks = asyncio.Event(), None, [], []
'''
SETTLE_OLD = '''                m = now[1]
                steady = (m.get("vllm:num_requests_running", 0) >= streams
'''
SETTLE_NEW = '''                m = now[1]
                if point.get("turns") and time.monotonic() - started >= float(point["turns"].get("warm_seconds", 120)):
                    break  # turns never reach pure decode: a fixed warm-up instead
                steady = (m.get("vllm:num_requests_running", 0) >= streams
'''
WIN_OLD = '''                first = await sample()
                cpu_first = snapshot_cpu(roles)
'''
WIN_NEW = '''                spy = (await spy_start(point["label"], float(point.get("window_seconds", spec.get("window_seconds", 15))))
                       if point.get("pyspy") else None)
                first = await sample()
                cpu_first = snapshot_cpu(roles)
'''
WIN_END_OLD = '''                last = samples[-1]
                cpu_last = snapshot_cpu(roles)
'''
WIN_END_NEW = '''                last = samples[-1]
                cpu_last = snapshot_cpu(roles)
                if spy is not None:
                    await spy_end(spy, record)
'''
for old, new in ((MEASURE, SPY), (PF_OLD, PF_NEW), (PF_END_OLD, PF_END_NEW), (TASKS_OLD, TASKS_NEW),
                 (SETTLE_OLD, SETTLE_NEW), (WIN_OLD, WIN_NEW), (WIN_END_OLD, WIN_END_NEW)):
    assert t.count(old) == 1, old[:70]
    t = t.replace(old, new)
compile(t, "kernel_profile.py", "exec")
kp.write_text(t)

p = out / "configs/comparison.json"
s = json.loads(p.read_text())
shutil.copyfile(contexts, out / "configs/bench/kprof-contexts.json")
points = [{"label": "PFH64x4", "content": "real", "prefill": True, "streams": 4, "subset": "range:0:4", "lead_seconds": 3,
           "pyspy": 50},
          {"label": "D44", "content": "real", "streams": 44, "subset": "range:0:44", "pyspy": True, "window_seconds": 20},
          {"label": "TURNS44", "content": "real", "streams": 44, "subset": "range:0:44", "pyspy": True,
           "window_seconds": 60, "settle_seconds": 600,
           "turns": {"max_tokens": 800, "user_repeat": 30, "warm_seconds": 120}}]
kpro = s["throughput_bench"]["kernel_profile"]
kpro["points"] = points
s["name"] = out.name
s["notebook_seconds"] = 7200
s["wall_seconds"] = 3600
s["trials"][0]["wall_seconds"] = 3600
s["b200"]["stop"]["wall_seconds"] = 7200
s["roster"] = "host-side profile v9: no games; cold prefill, pure decode and play-like turns under py-spy"
s.setdefault("evidence", {})["kprofH"] = "from b2-kprofP-v9-v1: py-spy + turns points (bench-only patch)"
p.write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((out / "MANIFEST.json").read_text())
m["study"] = out.name
m["files"] = {q.relative_to(out).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(out.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(out / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
print(out)
