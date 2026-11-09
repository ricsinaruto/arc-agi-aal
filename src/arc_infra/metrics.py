"""vLLM counter and GPU headroom sampling for fleet runs."""

import asyncio
import csv
import io
import json
import os
import re
import time
from pathlib import Path

import httpx

# llama-server ``--metrics`` under the vLLM names the fleet reads (model screens, 27 September).
# llama-server has no preemption counter: a request that fits its slot's fixed window never
# loses its cache, so reports on this backend carry no ``preemptions``.
LLAMACPP_ALIASES = {
    "llamacpp:requests_processing": "vllm:num_requests_running",
    "llamacpp:requests_deferred": "vllm:num_requests_waiting",
    "llamacpp:tokens_predicted_total": "vllm:generation_tokens_total",
    "llamacpp:prompt_tokens_total": "vllm:prompt_tokens_by_source_total:local_compute",
    "llamacpp:prompt_tokens_cached_total": "vllm:prompt_tokens_by_source_total:local_cache_hit",
}


def generation_counter_metadata(metrics):
    """Pinned llama.cpp flushes generation counts on slot reset, not every decode step."""
    if "llamacpp:tokens_predicted_total" in metrics:
        return {"generation_counter_updates": "on_slot_reset",
                "generation_counter_includes_inflight": False}
    return {}


def parse_metrics(text):
    values = {}
    for line in text.splitlines():
        match = re.fullmatch(r"((?:vllm|llamacpp):[\w:]+)(\{[^}]*\})?\s+([-+\d.eE]+)", line)
        if not match:
            continue
        name, labels, value = match.groups()
        if name.startswith("llamacpp:"):
            values[name] = float(value)
            if name in LLAMACPP_ALIASES:
                values[LLAMACPP_ALIASES[name]] = float(value)
            continue
        if name.endswith("_created") or name.endswith("_bucket"):
            continue
        if name == "vllm:prompt_tokens_by_source_total":
            source = re.search(r'source="([^"]+)"', labels or "")
            if source:
                name += ":" + source[1]
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            position = re.search(r'position="([^"]+)"', labels or "")
            if position:
                name += ":" + position[1]
        values[name] = values.get(name, 0.0) + float(value)
    return values


class Monitor:
    def __init__(self, output, base_url="http://127.0.0.1:8000"):
        self.output = Path(output)
        self.metrics_url = base_url.removesuffix("/v1").rstrip("/") + "/metrics"
        self.start = time.monotonic()
        self.samples = []
        self.errors = []
        self.on_sample = None

    async def sample(self):
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.get(self.metrics_url, timeout=3)
                response.raise_for_status()
            metrics = parse_metrics(response.text)
            item = {"elapsed_seconds": time.monotonic() - self.start, "metrics": metrics}
            self.samples.append(item)
            with (self.output / "telemetry.jsonl").open("a") as file:
                file.write(json.dumps(item) + "\n")
            (self.output / "vllm-metrics.txt").write_text(response.text)
            if self.on_sample is not None:
                self.on_sample(item)
        except httpx.HTTPError as exc:
            self.errors.append(
                {"elapsed_seconds": time.monotonic() - self.start, "error": type(exc).__name__}
            )
            if self.on_sample is not None:
                self.on_sample(None)

    async def run(self):
        # httpx/anyio can turn a cancel that lands in a timed-out request into an HTTPError, which sample() swallows;
        # the loop then ran on and held the bench's finish forever (rl1-loop-scc-chk2, 21 Oct)
        task = asyncio.current_task()
        while not task.cancelling():
            await asyncio.sleep(5)
            await self.sample()

    def summary(self):
        if len(self.samples) < 2:
            return {"valid": False, "errors": self.errors}
        first, last = self.samples[0], self.samples[-1]
        # The first sample is collected before gameplay begins.
        elapsed = last["elapsed_seconds"]
        counters = {
            "accepted_generation_tokens": "vllm:generation_tokens_total",
            "preemptions": "vllm:num_preemptions_total",
            "computed_prompt_tokens": "vllm:prompt_tokens_by_source_total:local_compute",
            "cached_prompt_tokens": "vllm:prompt_tokens_by_source_total:local_cache_hit",
        }
        result = {"valid": True, "sampled_seconds": elapsed, "errors": self.errors,
                  **generation_counter_metadata(last["metrics"])}
        for label, key in counters.items():
            if key in first["metrics"] and key in last["metrics"]:
                result[label] = last["metrics"][key] - first["metrics"][key]
        if "accepted_generation_tokens" not in result or elapsed <= 0:
            result["valid"] = False
        else:
            result["accepted_generation_tokens_per_second"] = (
                result["accepted_generation_tokens"] / elapsed
            )
        for label, key in {
            "peak_cache_fraction": "vllm:kv_cache_usage_perc",
            "peak_running_requests": "vllm:num_requests_running",
            "peak_waiting_requests": "vllm:num_requests_waiting",
        }.items():
            values = [s["metrics"][key] for s in self.samples if key in s["metrics"]]
            if values:
                result[label] = max(values)
        return result


GPU_FIELDS = (
    "index", "memory.total", "memory.used", "memory.free", "utilization.gpu", "power.draw"
)


def parse_gpu_rows(output):
    rows = []
    for values in csv.reader(io.StringIO(output)):
        if len(values) != len(GPU_FIELDS):
            raise ValueError("Unexpected nvidia-smi field count")
        row = {}
        for field, value in zip(GPU_FIELDS, values, strict=True):
            try:
                row[field] = float(value.strip())
            except ValueError:
                row[field] = None
        rows.append(row)
    if not rows:
        raise ValueError("nvidia-smi returned no GPU rows")
    return rows


def cpu_snapshot(proc="/proc"):
    """Host CPU accounting for a telemetry row (throughput arm, 7 October): cgroup usage and throttling, load, the
    aggregate /proc/stat line, and cumulative CPU seconds by process role (vLLM engine core, worker, API server, every
    other process summed) with the four busiest threads of each vLLM role. Counters: difference two rows."""
    tick = os.sysconf("SC_CLK_TCK")
    out = {"epoch": time.time(), "cpu_count": os.cpu_count()}
    for path in ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu,cpuacct/cpu.stat", "/sys/fs/cgroup/cpu/cpu.stat"):
        try:
            lines = Path(path).read_text().splitlines()
        except OSError:
            continue
        out["cgroup"] = {q[0]: int(q[1]) for q in (line.split() for line in lines) if len(q) == 2 and q[1].isdigit()}
        break
    for path, key in (("/sys/fs/cgroup/cpu.max", "cgroup_cpu_max"), (f"{proc}/loadavg", "loadavg")):
        try:
            out[key] = Path(path).read_text().strip()
        except OSError:
            pass
    try:
        first = Path(proc, "stat").read_text().splitlines()[0].split()
        out["host_ticks"] = dict(zip(("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"),
                                     (int(v) for v in first[1:9])))
    except (OSError, ValueError, IndexError):
        pass

    def seconds(path):
        text = path.read_text()
        fields = text[text.rindex(")") + 2:].split()  # after "pid (comm) "
        return (int(fields[11]) + int(fields[12])) / tick

    roles, counts, threads = {}, {}, {}
    for entry in Path(proc).iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            text = (entry / "comm").read_text().strip() + " " + cmdline
            used = seconds(entry / "stat")
            if "EngineCore" in text:
                role = "engine_core"
            elif "Worker" in text and "VLLM" in text:
                role = "worker"
            elif "vllm.entrypoints" in cmdline and " serve" in cmdline:
                role = "api_server"
            else:
                role = "other"
            if role != "other":
                per = {}
                for task in (entry / "task").iterdir():
                    try:
                        per[task.name] = seconds(task / "stat")
                    except (OSError, ValueError, IndexError):
                        pass
                top = sorted(per.items(), key=lambda item: -item[1])[:4]
                threads.setdefault(role, {}).update({f"{entry.name}/{tid}": round(v, 2) for tid, v in top})
        except (OSError, ValueError, IndexError):
            continue
        roles[role] = roles.get(role, 0.0) + used
        counts[role] = counts.get(role, 0) + 1
    out["process_cpu_seconds"] = {role: round(v, 2) for role, v in roles.items()}
    out["process_counts"] = counts
    out["thread_cpu_seconds"] = threads
    return out


CPU_SNAPSHOT_S = float(os.environ.get("ARC_CPU_SNAPSHOT_S", "120"))  # process-wide sampling cap; 0 = off
_cpu_shared = {"row": None, "task": None}


async def shared_cpu_snapshot(min_s=None):
    """cpu_snapshot for telemetry rows, off the event loop and at most once per ``min_s`` per process: every monitor
    reuses the last row (its ``epoch`` dates it). rl1-loop-hero-v6l 25 Oct: one GpuMonitor per running level bench
    each walked /proc on the loop every 30 s, 43% of the arena loop's CPU and 10-30 s loop stalls. None when off."""
    min_s = CPU_SNAPSHOT_S if min_s is None else float(min_s)
    if min_s <= 0:
        return None
    row = _cpu_shared["row"]
    if row is not None and time.time() - row["epoch"] < min_s:
        return row
    task = _cpu_shared["task"]
    if task is None or task.done() or task.get_loop() is not asyncio.get_running_loop():
        task = _cpu_shared["task"] = asyncio.ensure_future(asyncio.to_thread(cpu_snapshot))
    row = await asyncio.shield(task)
    _cpu_shared["row"] = row
    return row


class GpuMonitor(Monitor):
    """The engine monitor plus GPU headroom samples from nvidia-smi."""

    async def sample_gpu(self):
        record = {"elapsed_seconds": time.monotonic() - self.start}
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS),
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            output, _ = await asyncio.wait_for(process.communicate(), timeout=3)
            if process.returncode:
                raise RuntimeError(f"nvidia-smi exited {process.returncode}")
            record["gpus"] = parse_gpu_rows(output.decode())
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.communicate()
        try:
            cpu = await shared_cpu_snapshot()
            if cpu is not None:
                record["cpu"] = cpu
        except Exception as exc:  # telemetry only: never a reason to lose the GPU row
            record["cpu_error"] = f"{type(exc).__name__}: {exc}"
        record["sample_seconds"] = time.monotonic() - self.start - record["elapsed_seconds"]
        record["units"] = {"memory": "MiB", "utilization": "percent", "power": "watts"}
        with (self.output / "vllm-gpu-memory.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")

    async def run_gpu(self):
        task = asyncio.current_task()
        while not task.cancelling():
            await self.sample_gpu()
            await asyncio.sleep(30)

    async def run(self):
        tasks = [asyncio.create_task(super().run()), asyncio.create_task(self.run_gpu())]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
