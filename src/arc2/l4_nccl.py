"""What NCCL can do on a host where the server's first all-reduce fails (Modal hosts with NVIDIA driver 610.57.04:
"unhandled cuda error" in the warm-up reduction of ``PyNcclCommunicator``, all four ranks, seconds after the
workers start). ``python -m arc2.l4_nccl --out file.json --logs dir`` in the serving runtime's environment, the
cards free: every ``CASES`` setting under the server's own path (vLLM's communicator on a gloo group) and under
torch.distributed's, each a fresh set of four processes with NCCL's log in a file. Per case and path: whether the
first reduction came back right, the latency of a decode step's reduction (16 and 32 rows x 2,560 BF16), and from
NCCL's log the warnings (the failing CUDA call) and the lines that name the transport.

A first reduction is where NCCL connects its ranks, so the settings are the transports: cuMem handles for
peer-to-peer buffers (off: the older IPC handles), peer-to-peer off (shared host memory), both off (sockets).
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from .l4_allreduce import HIDDEN, RANKS, timed

CASES = {"base": {},
         "cumem_off": {"NCCL_CUMEM_ENABLE": "0"},
         "cumem_host_off": {"NCCL_CUMEM_HOST_ENABLE": "0"},
         "cumem_both_off": {"NCCL_CUMEM_ENABLE": "0", "NCCL_CUMEM_HOST_ENABLE": "0"},
         "eager_connect": {"NCCL_RUNTIME_CONNECT": "0"},
         "no_p2p": {"NCCL_P2P_DISABLE": "1"},
         "no_p2p_cumem_off": {"NCCL_P2P_DISABLE": "1", "NCCL_CUMEM_ENABLE": "0", "NCCL_CUMEM_HOST_ENABLE": "0"},
         "sockets": {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}}
PATHS = ("pynccl", "torch")  # the server's path first
ROWS = (16, 32)
KEEP = re.compile(r"via |NCCL version|cuMem|CUMEM|P2P|SHM|NET/|Cuda|cuda|driver|Driver|failed|error", re.I)


def worker(rank, port, path, env, log_file, calls, queue):
    os.environ.update({"NCCL_DEBUG": "INFO", "NCCL_DEBUG_FILE": log_file, **env})  # before NCCL starts
    out = {}
    try:
        import torch
        import torch.distributed as dist

        if path == "cpu":  # the self-test: the processes and the queue, no card
            dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=RANKS)
            x = torch.ones(4, HIDDEN)
            dist.all_reduce(x)
            out = {"sum_ok": bool((x == RANKS).all().item())}
        else:
            device, dtype = torch.device(f"cuda:{rank}"), torch.bfloat16
            if path == "pynccl":  # as vllm.distributed.parallel_state: the communicator on a non-NCCL group
                dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=RANKS)
                from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

                comm = PyNcclCommunicator(group=dist.group.WORLD, device=rank)  # its warm-up is the failing call
                if comm.disabled:
                    raise RuntimeError("PyNcclCommunicator is disabled: no NCCL library")
                out["nccl"] = str(comm.nccl.ncclGetVersion())
                reduce = comm.all_reduce
            else:
                dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=RANKS)
                out["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))

                def reduce(x):
                    y = x.clone()
                    dist.all_reduce(y)
                    return y

                try:
                    reduce(torch.ones(1, HIDDEN, dtype=dtype, device=device))
                except Exception:  # noqa: BLE001 - a build without BF16 reductions: same bytes in half precision
                    dtype = torch.float16
            first = reduce(torch.ones(4, HIDDEN, dtype=dtype, device=device))
            torch.cuda.synchronize(device)
            out.update(sum_ok=bool((first == RANKS).all().item()), dtype=str(dtype))
            for rows in ROWS:
                x = torch.ones(rows, HIDDEN, dtype=dtype, device=device)
                out[str(rows)] = timed(lambda x=x: reduce(x), device, calls)
    except BaseException as exc:  # noqa: BLE001 - the record says what failed
        out["error"] = repr(exc)[:500]
    queue.put((rank, out))
    queue.close()
    queue.join_thread()
    os._exit(0)  # no teardown: a communicator that failed can hang in it


def excerpt(texts, warnings=15, facts=25):
    """From NCCL's log files of one start: the distinct warnings and the lines that name what it used."""
    seen = {"warn": [], "info": []}
    for text in texts:
        for line in text.splitlines():
            found = re.search(r"NCCL (WARN|INFO) (.*)", line)
            if not found or (found.group(1) == "INFO" and not KEEP.search(found.group(2))):
                continue
            kind, key = found.group(1).lower(), re.sub(r"0x[0-9a-f]+|\d+", "#", found.group(2))
            if all(key != k for k, _ in seen[kind]):
                seen[kind].append((key, found.group(2)[:300]))
    return {"warn": [text for _, text in seen["warn"][:warnings]], "info": [text for _, text in seen["info"][:facts]]}


def measure(context, port, path, env, logs, name, calls, wait):
    queue, pattern = context.Queue(), str(Path(logs) / f"{name}-{path}.%h.%p.log")
    jobs = [context.Process(target=worker, args=(rank, port, path, env, pattern, calls, queue))
            for rank in range(RANKS)]
    begun = time.time()
    for job in jobs:
        job.start()
    rows = {}
    try:
        for _ in jobs:
            rank, out = queue.get(timeout=max(5, wait - (time.time() - begun)))
            rows[str(rank)] = out
    except Exception as exc:  # noqa: BLE001 - a rank that never answered (hung or killed)
        rows["missing"] = repr(exc)[:200]
    for job in jobs:
        job.join(timeout=5)
        if job.is_alive():
            job.kill()
    ranks = [rows.get(str(rank)) or {} for rank in range(RANKS)]
    texts = [p.read_text(errors="replace") for p in sorted(Path(logs).glob(f"{name}-{path}.*.log"))]
    return {"ok": all(r.get("sum_ok") and "error" not in r for r in ranks), "seconds": round(time.time() - begun, 1),
            "errors": sorted({r["error"] for r in ranks if "error" in r})[:3], "missing": rows.get("missing"),
            "nccl": ranks[0].get("nccl"), "dtype": ranks[0].get("dtype"),
            "median_us": {str(n): ranks[0][str(n)]["median_us"] for n in ROWS if str(n) in ranks[0]},
            "log": excerpt(texts)}


def host():
    out = {}
    for name, command in (("cards", ["nvidia-smi", "--query-gpu=index,name,driver_version,pci.bus_id",
                                    "--format=csv,noheader"]),
                          ("libraries", ["sh", "-c", "ldconfig -p | grep -E 'libcuda|libnvidia-ml|libnccl' | head -12"])):
        try:
            out[name] = subprocess.run(command, capture_output=True, text=True, timeout=20).stdout[-1500:]
        except Exception as exc:  # noqa: BLE001 - a reading
            out[name] = repr(exc)[:200]
    for name, path in (("driver", "/proc/driver/nvidia/version"), ("kernel", "/proc/version")):
        try:
            out[name] = Path(path).read_text()[:300]
        except OSError as exc:
            out[name] = repr(exc)[:120]
    out["nccl_environment"] = {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "VLLM_NCCL"))}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--calls", type=int, default=300)
    parser.add_argument("--port", type=int, default=29617)
    parser.add_argument("--wait", type=int, default=75)
    parser.add_argument("--budget", type=int, default=900, help="seconds; cases left when it is spent are named")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    import torch
    import torch.multiprocessing as mp

    context, begun = mp.get_context("spawn"), time.time()
    Path(args.logs).mkdir(parents=True, exist_ok=True)
    record = {"host": host(), "torch": torch.__version__, "cuda": torch.version.cuda, "cases": {}, "skipped": []}
    paths = ("cpu",) if args.self_test else PATHS
    if not args.self_test:
        try:
            record["peer_access"] = [[i == j or bool(torch.cuda.can_device_access_peer(i, j)) for j in range(RANKS)]
                                     for i in range(RANKS)]
        except Exception as exc:  # noqa: BLE001 - a reading
            record["peer_access"] = repr(exc)[:200]
    for k, (name, env) in enumerate(CASES.items()):
        for j, path in enumerate(paths):
            if time.time() - begun > args.budget:
                record["skipped"].append(f"{name}/{path}")
                continue
            found = measure(context, args.port + 2 * k + j, path, env, args.logs, name, args.calls, args.wait)
            record["cases"].setdefault(name, {"env": env})[path] = found
            with open(args.out, "w") as stream:  # whole after every start
                json.dump(record, stream, indent=1)
            print("ARC2_L4_NCCL " + json.dumps({"case": name, "path": path, "ok": found["ok"],
                                                "median_us": found["median_us"], "errors": found["errors"][:1],
                                                "warn": found["log"]["warn"][:2]}), flush=True)
    record["serves"] = sorted(name for name, row in record["cases"].items() if (row.get(paths[0]) or {}).get("ok"))
    with open(args.out, "w") as stream:
        json.dump(record, stream, indent=1)
    print("ARC2_L4_NCCL_SERVES " + json.dumps(record["serves"]), flush=True)


if __name__ == "__main__":
    main()
