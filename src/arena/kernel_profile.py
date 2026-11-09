"""Kernel-level profile and real-vs-synthetic step pilot (30 September; docs/validation-20260930/kernel-profile.md).

One server start of the blocks3 campaign serving (the decode-sweep build: 56 sequences, graph ladder every 4 from 32
to 56, torch profiler configured). The serving bench then measures a list of points. A point is n concurrent forced
decodes (``min_tokens = max_tokens``, ``ignore_eos``) on one prompt set:

- ``real``: recorded Duck contexts (``bench/kernel-profile-contexts.json``, ``scripts/kernel_profile_inputs.py``) sent
  as the harness sends them: ``/v1/chat/completions`` with the python tool, ``tool_choice`` auto, temperature 0.7,
  top-p 0.95, top-k 20, no seed, non-streaming;
- ``synthetic``: the serving bench's Duck-shaped word-salad prompts with a board image and a per-request seed (the
  decode sweep's bodies), each calibrated to the token count of the real context at the same index.

Every point waits for **pure decode** (all n running, none waiting, and scheduled tokens per step equal to generated
tokens per step over the last second: no prefill chunk in flight), then reads a window of engine counters (ms per
step), the CPU time of the engine-core, worker and API-server processes per step, the cgroup throttling and host
steal time, and nvidia-smi clocks / power / temperature. A point can add a load during its window (CPU burners, a
``/tokenize`` flood at the harness's rate, streaming consumption, a prefill injector) and a torch-profiler trace of
~40 steps after its window. The decode sweep (arena-decode-sweep-v1) settled on the prompt-token counter, which
moves only when a prefill completes, so its windows could hold running prefill chunks; the pure-decode test here
uses ``iteration_tokens`` (scheduled) against generated tokens instead and records any residue.
"""

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

from .hero_v2_precheck import COUNTERS, Sampler, wait_for_idle, window_stats
from .isolation import isolate_trial
from .throughput_bench import PROFILE_DIR, load_trace, request_body, summarize_trace

CONTEXTS = "bench/kernel-profile-contexts.json"
SMI_FIELDS = ("name", "uuid", "serial", "vbios_version", "pci.bus_id", "power.limit", "enforced.power.limit",
              "power.max_limit", "clocks.max.sm", "clocks.max.mem", "clocks.sm", "clocks.mem", "power.draw",
              "temperature.gpu", "clocks_event_reasons.active", "utilization.gpu", "pstate")
CGROUP_STAT = ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu,cpuacct/cpu.stat", "/sys/fs/cgroup/cpu/cpu.stat")
TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


# ----------------------------------------------------------------------------- host / process / GPU readings
def classify_process(cmdline, comm):
    """Role of a serving process from its /proc cmdline and comm (vLLM sets VLLM::EngineCore / VLLM::Worker titles)."""
    text = f"{comm} {cmdline}"
    if "EngineCore" in text:
        return "engine_core"
    if "Worker" in text and "VLLM" in text:
        return "worker"
    if "vllm.entrypoints" in cmdline and " serve" in cmdline:
        return "api_server"
    return None


def serving_processes(proc="/proc"):
    """{role: [pid]} of the vLLM processes visible in this container."""
    roles = {}
    for entry in Path(proc).iterdir() if Path(proc).is_dir() else []:
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            comm = (entry / "comm").read_text().strip()
        except OSError:
            continue
        role = classify_process(cmdline, comm)
        if role:
            roles.setdefault(role, []).append(int(entry.name))
    return roles


def process_cpu_seconds(pid, proc="/proc"):
    """utime + stime of a process (all its threads), seconds; None when it is gone."""
    try:
        text = (Path(proc) / str(pid) / "stat").read_text()
    except OSError:
        return None
    fields = text[text.rindex(")") + 2:].split()  # after "pid (comm) "
    return (int(fields[11]) + int(fields[12])) / TICK


def cgroup_cpu():
    """cgroup CPU accounting: usage and throttling counters (v2 names; v1 keys kept as they are)."""
    for path in CGROUP_STAT:
        try:
            text = Path(path).read_text()
        except OSError:
            continue
        values = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                values[parts[0]] = int(parts[1])
        return {"path": path, **values}
    return None


def host_cpu(proc="/proc"):
    """The aggregate cpu line of /proc/stat (ticks): user nice system idle iowait irq softirq steal."""
    try:
        first = Path(proc, "stat").read_text().splitlines()[0].split()
    except OSError:
        return None
    names = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
    return {n: int(v) for n, v in zip(names, first[1:9])}


def snapshot_cpu(roles):
    return {"epoch": time.time(), "cgroup": cgroup_cpu(), "host": host_cpu(),
            "processes": {role: sum(v for v in (process_cpu_seconds(p) for p in pids) if v is not None)
                          for role, pids in roles.items()}}


def cpu_delta(first, last, steps):
    """CPU per engine step by process role, cgroup throttling and host shares between two ``snapshot_cpu`` reads."""
    seconds = last["epoch"] - first["epoch"]
    out = {"seconds": round(seconds, 3)}
    out["process_cpu_ms_per_step"] = {
        role: round(1000 * (last["processes"][role] - first["processes"].get(role, 0.0)) / steps, 3)
        for role in last["processes"]} if steps else None
    out["process_cpu_share"] = {
        role: round((last["processes"][role] - first["processes"].get(role, 0.0)) / seconds, 3)
        for role in last["processes"]} if seconds > 0 else None
    a, b = first.get("cgroup") or {}, last.get("cgroup") or {}
    if a and b:
        out["cgroup"] = {k: b[k] - a[k] for k in ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec")
                         if k in a and k in b}
        if "usage_usec" in out["cgroup"] and seconds > 0:
            out["cgroup"]["cores_used"] = round(out["cgroup"]["usage_usec"] / 1e6 / seconds, 3)
    h1, h2 = first.get("host") or {}, last.get("host") or {}
    if h1 and h2:
        delta = {k: h2[k] - h1[k] for k in h1}
        total = sum(delta.values())
        if total > 0:
            out["host_share"] = {k: round(v / total, 4) for k, v in delta.items()}
    return out


async def run_command(*argv, timeout=15):
    proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.DEVNULL)
    out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    return proc.returncode, out.decode(errors="replace")


async def smi(fields=SMI_FIELDS):
    """One nvidia-smi read of ``fields`` (each field alone when the combined query is rejected); None on no GPU."""
    try:
        code, out = await run_command("nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader")
        if code == 0 and out.strip():
            values = [v.strip() for v in out.splitlines()[0].split(",")]
            if len(values) == len(fields):
                return {"epoch": round(time.time(), 1), **dict(zip(fields, values))}
        result = {"epoch": round(time.time(), 1)}
        for field in fields:
            code, out = await run_command("nvidia-smi", "--query-gpu=" + field, "--format=csv,noheader")
            result[field] = out.strip().splitlines()[0] if code == 0 and out.strip() else None
        return result
    except Exception:  # noqa: BLE001 - diagnostics only
        return None


async def host_info():
    info = {"cpu_count": os.cpu_count(), "python": sys.version.split()[0]}
    try:
        info["affinity"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    for path, key in (("/sys/fs/cgroup/cpu.max", "cgroup_cpu_max"), ("/proc/loadavg", "loadavg")):
        try:
            info[key] = Path(path).read_text().strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info["gpu"] = await smi()
    return info


# ----------------------------------------------------------------------------- request bodies
def chat_body(context, *, model, effort, temperature, template_kwargs):
    """A recorded context as the Duck harness sends it (non-streaming, tools, tool_choice auto, no seed)."""
    return {"model": model, "messages": context["messages"], "tools": context["tools"], "tool_choice": "auto",
            "temperature": temperature, "top_p": 0.95, "top_k": 20, "max_tokens": 1,
            "chat_template_kwargs": template_kwargs(model, {"enable_thinking": True, "reasoning_effort": effort})}


def forced(body, out, overrides=None, drop=()):
    body = {**body, "max_tokens": out, "min_tokens": out, "ignore_eos": True, **(overrides or {})}
    for key in drop:
        body.pop(key, None)
    return body


def chosen_indices(counts, streams, subset):
    """``first``: indices 0..n-1 (the long set first, then the short set); ``longest``: the n longest prompts."""
    if streams > len(counts):
        raise ValueError(f"{streams} streams but {len(counts)} prompts")
    if subset == "first":
        return list(range(streams))
    if subset == "longest":
        return sorted(sorted(range(len(counts)), key=lambda i: -counts[i])[:streams])
    raise ValueError(f"unknown subset {subset!r}")


def gate_blocks(counts, out, block_tokens, fixed):
    return sum(-(-(n + out) // block_tokens) + fixed for n in counts)


def pure_decode(previous, now, tolerance=0):
    """True when the engine scheduled only decode tokens between two samples (and did schedule some)."""
    steps = now[1].get(COUNTERS["steps"], 0) - previous[1].get(COUNTERS["steps"], 0)
    scheduled = now[1].get(COUNTERS["scheduled_tokens"], 0) - previous[1].get(COUNTERS["scheduled_tokens"], 0)
    generated = now[1].get(COUNTERS["generated"], 0) - previous[1].get(COUNTERS["generated"], 0)
    return steps > 0 and generated > 0 and scheduled - generated <= tolerance


# ----------------------------------------------------------------------------- loads
class CpuBurners:
    """``count`` spinning Python processes for the window (CPU contention)."""

    def __init__(self, count):
        self.count, self.procs = int(count), []

    async def __aenter__(self):
        for _ in range(self.count):
            self.procs.append(await asyncio.create_subprocess_exec(sys.executable, "-c", "while True: pass"))
        return self

    async def __aexit__(self, *exc):
        for p in self.procs:
            try:
                p.kill()
            except ProcessLookupError:
                pass
        await asyncio.gather(*(p.wait() for p in self.procs), return_exceptions=True)


class TokenizeFlood:
    """``/tokenize`` of real bodies at ``rate`` per second (the harness counts each prompt 2-3 times per request)."""

    def __init__(self, client, root, bodies, rate):
        self.client, self.root, self.bodies, self.rate = client, root, bodies, float(rate)
        self.sent = self.done = 0
        self.seconds = 0.0
        self.stop, self.task, self.inflight = asyncio.Event(), None, set()

    async def one(self, body):
        started = time.monotonic()
        try:
            r = await self.client.post(self.root + "/tokenize", json={k: body[k] for k in (
                "model", "messages", "tools", "chat_template_kwargs") if k in body})
            if r.status_code == 200:
                self.done += 1
                self.seconds += time.monotonic() - started
        except Exception:  # noqa: BLE001 - load generator only
            pass

    async def loop(self):
        interval, next_at, i = 1.0 / self.rate, time.monotonic(), 0
        while not self.stop.is_set():
            task = asyncio.create_task(self.one(self.bodies[i % len(self.bodies)]))
            self.inflight.add(task)
            task.add_done_callback(self.inflight.discard)
            self.sent, i = self.sent + 1, i + 1
            next_at += interval
            try:
                await asyncio.wait_for(self.stop.wait(), max(0.0, next_at - time.monotonic()))
            except TimeoutError:
                pass

    async def __aenter__(self):
        self.task = asyncio.create_task(self.loop())
        return self

    async def __aexit__(self, *exc):
        self.stop.set()
        await asyncio.gather(self.task, *self.inflight, return_exceptions=True)

    def summary(self):
        return {"rate": self.rate, "sent": self.sent, "completed": self.done,
                "mean_seconds": round(self.seconds / self.done, 4) if self.done else None}


class NullLoad:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


# ----------------------------------------------------------------------------- the pilot
async def kernel_profile(bench, client, gate, budget, monitor, spec, *, save=lambda: None):
    from arc_infra.model_templates import template_kwargs

    result = {"spec": spec, "points": [], "errors": []}
    root = bench.base_url.removesuffix("/v1")
    sample = Sampler(client, bench.base_url)
    out = int(spec["output_tokens"])
    tick = float(spec.get("sample_seconds", 0.5))
    result["host"] = await host_info()
    roles = serving_processes()
    result["processes"] = roles
    path = (bench.source / spec["contexts_file"]) if bench.source else Path(spec["contexts_file"])
    contexts = json.loads(path.read_text())["contexts"]
    real = [chat_body(c, model=bench.model, effort=bench.effort, temperature=bench.temperature,
                      template_kwargs=template_kwargs) for c in contexts]

    async def count_body(body):
        r = await client.post(root + "/tokenize", json={k: body[k] for k in (
            "model", "messages", "tools", "chat_template_kwargs") if k in body})
        r.raise_for_status()
        return int(r.json()["count"])

    real_counts = [await count_body(b) for b in real]
    result["real_prompt_tokens"] = real_counts
    synth_n = int(spec.get("synthetic_streams", 40))
    reference = int(spec.get("calibrate_tokens", 24576))
    words_ref = await bench.words_for(client, reference)
    tokens_ref = bench.calibration[reference]["tokens"]
    synth, synth_counts = [], []
    for i in range(synth_n):
        words = max(64, int(words_ref * real_counts[i] / tokens_ref))
        body = request_body(6_000_000 + i, words, model=bench.model, output_tokens=1,
                            reasoning_effort=bench.effort, temperature=bench.temperature)
        synth.append(body)
        synth_counts.append(await bench.count(client, body))
    result["synthetic_prompt_tokens"] = synth_counts
    sets = {"real": (real, real_counts), "synthetic": (synth, synth_counts)}
    result["cache_reset"] = await isolate_trial(bench.base_url)
    save()
    folder = Path(bench.server_dir) / PROFILE_DIR

    async def profile(label, seconds):
        before = set(folder.rglob("*")) if folder.is_dir() else set()
        record = {}
        response = await client.post(root + "/start_profile")
        if response.status_code != 200:
            raise RuntimeError(f"start_profile {response.status_code}: {response.text[:200]}")
        record["started_epoch"] = time.time()
        await asyncio.sleep(seconds)
        response = await client.post(root + "/stop_profile")
        if response.status_code != 200:
            raise RuntimeError(f"stop_profile {response.status_code}: {response.text[:200]}")
        record["stopped_epoch"] = time.time()
        destination = bench.output / f"profile-{label}"
        destination.mkdir(parents=True, exist_ok=True)
        record["traces"] = []
        for trace in await bench.new_traces(folder, before):
            target = destination / trace.name
            shutil.move(str(trace), target)
            entry = {"file": f"profile-{label}/{target.name}", "bytes": target.stat().st_size}
            try:
                entry["summary"] = summarize_trace(load_trace(target))
            except Exception as exc:  # noqa: BLE001 - keep the raw trace
                entry["summary_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            record["traces"].append(entry)
        for table in folder.glob("profiler_out_*.txt") if folder.is_dir() else []:
            shutil.move(str(table), destination / table.name)
        return record

    async def measure(point):
        bodies, counts = sets[point["content"]]
        streams = int(point["streams"])
        chosen = chosen_indices(counts, streams, point.get("subset", "first"))
        record = {"label": point["label"], "content": point["content"], "streams": streams,
                  "subset": point.get("subset", "first"), "overrides": point.get("overrides") or {},
                  "drop": point.get("drop") or [], "load": point.get("load"), "stream": bool(point.get("stream")),
                  "mean_prompt_tokens": round(sum(counts[i] for i in chosen) / streams, 1),
                  "max_prompt_tokens": max(counts[i] for i in chosen)}
        requests = [forced(bodies[i], out, point.get("overrides"), point.get("drop") or ()) for i in chosen]
        if point.get("stream"):
            requests = [{**b, "stream": True, "stream_options": {"include_usage": True}} for b in requests]
        sizes = [counts[i] for i in chosen]
        finished, statuses, chunk_times = asyncio.Event(), [], {}

        async def one(i, body):
            reservation = await gate.acquire(f"kprof-{point['label']}-{i}", sizes[i] + out)
            try:
                if body.get("stream"):
                    async with client.stream("POST", bench.base_url + "/chat/completions", json=body) as response:
                        statuses.append(response.status_code)
                        times = chunk_times.setdefault(i, [])
                        async for line in response.aiter_lines():
                            if line.startswith("data:"):
                                times.append(time.time())
                else:
                    response = await client.post(bench.base_url + "/chat/completions", json=body)
                    statuses.append(response.status_code)
                    if response.status_code != 200:
                        record.setdefault("failures", []).append(response.text[:300])
            finally:
                gate.release(reservation)
                finished.set()

        tasks = [asyncio.create_task(one(i, b)) for i, b in enumerate(requests)]
        stop, injector, injected, inject_tasks = asyncio.Event(), None, [], []
        try:
            started = time.monotonic()
            deadline = started + float(point.get("settle_seconds", spec.get("settle_seconds", 90)))
            settle_first = await sample()
            previous, quiet_since = settle_first, None
            while True:
                await asyncio.sleep(tick)
                now = await sample()
                m = now[1]
                steady = (m.get("vllm:num_requests_running", 0) >= streams
                          and not m.get("vllm:num_requests_waiting", 0))
                if point.get("inject") is None and not pure_decode(previous, now):
                    steady = False
                quiet_since = (quiet_since or now[0]) if steady else None
                if quiet_since is not None and now[0] - quiet_since >= float(spec.get("quiet_seconds", 1.0)):
                    break
                if finished.is_set():
                    raise RuntimeError(f"{point['label']}: a request finished before steady decode "
                                       f"(statuses {statuses[:3]}, {record.get('failures', [''])[0][:200]})")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{point['label']}: no pure decode within the settle time (running "
                                       f"{m.get('vllm:num_requests_running')}, waiting "
                                       f"{m.get('vllm:num_requests_waiting')})")
                previous = now
            settle = {k: now[1].get(name, 0.0) - settle_first[1].get(name, 0.0)
                      for k, name in (("uncached", COUNTERS["uncached"]), ("cached", COUNTERS["cached"]))}
            record["settle"] = {"seconds": round(time.monotonic() - started, 2), **settle}
            if point.get("inject") is not None:
                make = await bench.injector_bodies(client, point["inject"], 7_000_000)
                record["inject_warm"] = await bench.run_pass("inject-warm", client, gate, budget, [make(0)], monitor)
                interval = 1.0 / float(point["inject"]["rate"])

                async def run_injector():
                    index, next_at = 0, time.monotonic()
                    while not stop.is_set():
                        index += 1
                        inject_tasks.append(asyncio.create_task(
                            bench.inject_one(client, gate, make(index), index, injected)))
                        next_at += interval
                        try:
                            await asyncio.wait_for(stop.wait(), max(0.0, next_at - time.monotonic()))
                        except TimeoutError:
                            pass

                injector = asyncio.create_task(run_injector())
                await asyncio.sleep(float(spec.get("inject_lead_seconds", 4)))
            load = point.get("load") or {}
            if load.get("cpu_burners"):
                count = load["cpu_burners"]
                count = (result["host"].get("affinity") or os.cpu_count() or 8) if count == "all" else int(count)
                context = CpuBurners(count)
            elif load.get("tokenize_rate"):
                context = TokenizeFlood(client, root, real, load["tokenize_rate"])
            else:
                context = NullLoad()
            async with context:
                if not isinstance(context, NullLoad):
                    await asyncio.sleep(float(spec.get("load_lead_seconds", 2)))
                first = await sample()
                cpu_first = snapshot_cpu(roles)
                window = float(point.get("window_seconds", spec.get("window_seconds", 15)))
                samples, smi_mid = [first], None
                while samples[-1][0] - first[0] < window and not finished.is_set():
                    await asyncio.sleep(tick)
                    samples.append(await sample())
                    if smi_mid is None and samples[-1][0] - first[0] >= window / 2:
                        smi_mid = await smi(("clocks.sm", "clocks.mem", "power.draw", "temperature.gpu",
                                             "clocks_event_reasons.active", "utilization.gpu", "pstate"))
                last = samples[-1]
                cpu_last = snapshot_cpu(roles)
            if isinstance(context, TokenizeFlood):
                record["tokenize_load"] = context.summary()
            if isinstance(context, CpuBurners):
                record["cpu_burners"] = context.count
            stats = window_stats(first, last, streams)
            record.update(stats)
            if stats.get("valid"):
                record["prefill_tokens_per_step"] = round(stats["scheduled_tokens_per_step"]
                                                          - stats["generated_per_step"], 2)
                record["pure_decode"] = record["prefill_tokens_per_step"] <= 0.01
                record["cpu"] = cpu_delta(cpu_first, cpu_last, stats["steps"])
            record["running_min"] = min(s[1].get("vllm:num_requests_running", 0) for s in samples)
            record["ended_early"] = finished.is_set()
            record["smi"] = smi_mid
            if point.get("stream"):
                gaps = [b - a for times in chunk_times.values() for a, b in zip(times, times[1:])
                        if first[0] <= a and b <= last[0]]
                record["client_chunk_gap_ms"] = round(1000 * sum(gaps) / len(gaps), 3) if gaps else None
                record["client_chunks_in_window"] = len(gaps)
            if point.get("profile") and not finished.is_set():
                try:
                    record["profile"] = await profile(point["label"], float(point.get("profile_seconds", 4)))
                except Exception as exc:  # noqa: BLE001 - a failed profile keeps the window
                    record["profile_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                    result["errors"].append(f"profile {point['label']}: {record['profile_error']}")
        finally:
            stop.set()
            if injector is not None:
                await asyncio.gather(injector, return_exceptions=True)
            for task in tasks:
                task.cancel()  # client disconnect: vLLM aborts the forced requests
            await asyncio.gather(*tasks, *inject_tasks, return_exceptions=True)
        if point.get("inject") is not None:
            ok = [r for r in injected if r.get("status") == 200]
            record["inject"] = {**point["inject"], "sent": len(injected), "completed": len(ok),
                                "mean_seconds": round(sum(r["seconds"] for r in ok) / len(ok), 3) if ok else None}
        record["gpu_after"] = await smi(("clocks.sm", "power.draw", "temperature.gpu", "memory.free"))
        await wait_for_idle(sample)
        return record

    for point in spec["points"]:
        remaining = bench.remaining()
        need = float(point.get("min_remaining_seconds", 120))
        if remaining is not None and remaining < need:
            result["errors"].append(f"{point['label']}: skipped, {remaining:.0f} s left")
            continue
        bench.log(json.dumps({"event": "kprof_point", "label": point["label"]}), flush=True)
        try:
            result["points"].append(await measure(point))
        except Exception as exc:  # noqa: BLE001 - one bad point must not lose the pilot
            result["errors"].append(f"{point['label']}: {type(exc).__name__}: {str(exc)[:400]}")
            try:
                await wait_for_idle(sample)
            except Exception:  # noqa: BLE001
                pass
        save()
    result["host_end"] = {"gpu": await smi()}
    return result
