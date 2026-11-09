"""All-reduce latency between the cards, as a tensor-parallel decode step pays it (about two per layer on a
``rows x 2,560`` BF16 tensor): ``python -m arc2.l4_allreduce --out file.json`` in the serving runtime's
environment, after the server has stopped. torch.distributed's NCCL backend (the server's PYNCCL path calls the same
library); each call is followed by a synchronize, so the number is the latency a step waits for.

Beside it, for the question whether a faster path exists on these hosts: peer access between every pair of cards
and their PCIe links, the same reduction under other NCCL settings (``VARIANTS``; one more with NCCL's own log,
which names the transport), and per rank the floor of two other paths for a tensor this small: a round trip
through host memory and a copy to the next card.
"""

import argparse
import json
import time

HIDDEN, RANKS = 2560, 4
GROUP = 128  # ``--q8``: elements per int8 scale


def q8_all_reduce(x, dist, ranks=RANKS, group=GROUP):
    """Two-shot all-reduce in 8 bits (``--q8``): each rank's tensor is cut into one chunk per rank, quantized per
    ``group`` elements (absmax / 127, the scale stored as 4 int8 bytes of a float32 beside its group), sent with one
    all-to-all, summed in float32 by the chunk's owner, requantized and gathered with one all-gather. About half the
    bytes of NCCL's BF16 ring; two NCCL calls instead of one."""
    import torch

    flat = x.reshape(-1)
    size = flat.numel()
    unit = ranks * group
    padded = -(-size // unit) * unit
    if padded != size:
        flat = torch.nn.functional.pad(flat, (0, padded - size))

    def pack(v):  # [..., group] float -> [..., group + 4] int8 (values, then the float32 scale's bytes)
        scale = v.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
        q = torch.round(v / scale).to(torch.int8)
        return torch.cat([q, scale.to(torch.float32).view(torch.int8)], -1)

    def unpack(b):
        return b[..., :group].to(torch.float32) * b[..., group:].contiguous().view(torch.float32)

    send = pack(flat.view(ranks, -1, group).float())
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    mine = pack(unpack(recv).sum(0))
    every = torch.empty((ranks * mine.shape[0], mine.shape[1]), dtype=torch.int8, device=x.device)
    dist.all_gather_into_tensor(every, mine)
    return unpack(every).reshape(-1)[:size].view_as(x).to(x.dtype)


VARIANTS = {"proto_ll": {"NCCL_PROTO": "LL"}, "proto_ll128": {"NCCL_PROTO": "LL128"}, "tree": {"NCCL_ALGO": "Tree"},
            "no_p2p": {"NCCL_P2P_DISABLE": "1"}, "log": {"NCCL_DEBUG": "INFO"}}


def timed(step, device, calls):
    import torch

    for _ in range(50):
        step()
    torch.cuda.synchronize(device)
    times = []
    for _ in range(calls):
        begun = time.perf_counter()
        step()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - begun)
    times.sort()
    return {"median_us": round(times[len(times) // 2] * 1e6, 1), "mean_us": round(sum(times) / len(times) * 1e6, 1),
            "p90_us": round(times[int(len(times) * 0.9)] * 1e6, 1)}


def worker(rank, port, rows_list, calls, queue, env=None, floors=False, q8=False):
    import os

    os.environ.update(env or {})  # before NCCL starts
    import torch
    import torch.distributed as dist

    try:
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=RANKS)
        device = torch.device(f"cuda:{rank}")
        out, dtype = {}, torch.bfloat16
        try:
            dist.all_reduce(torch.ones(1, HIDDEN, dtype=dtype, device=device))
        except Exception:  # noqa: BLE001 - a build without BF16 reductions: same bytes in half precision
            dtype = torch.float16
        out["dtype"] = str(dtype)
        for rows in rows_list:
            x = torch.ones(rows, HIDDEN, dtype=dtype, device=device)
            out[str(rows)] = timed(lambda x=x: dist.all_reduce(x), device, calls)
            if q8:  # the 8-bit two-shot path on the same tensor, and its error against the exact sum
                torch.manual_seed(rank)
                v = torch.randn(rows, HIDDEN, device=device).to(dtype)
                exact = v.float().clone()
                dist.all_reduce(exact)
                got = q8_all_reduce(v, dist).float()
                out[str(rows)]["q8"] = timed(lambda v=v: q8_all_reduce(v, dist), device, calls)
                out[str(rows)]["q8_rel_err"] = round(float((got - exact).norm() / exact.norm()), 5)
            if floors:  # what a path of our own could not beat: host memory and back; the next card and back
                host = torch.empty(rows, HIDDEN, dtype=dtype).pin_memory()
                out[str(rows)]["host_round_trip"] = timed(lambda x=x, host=host: x.copy_(host.copy_(x)), device, calls)
                peer = torch.device(f"cuda:{(rank + 1) % RANKS}")
                there = torch.empty(rows, HIDDEN, dtype=dtype, device=peer)

                def hop(x=x, there=there, peer=peer):
                    there.copy_(x)
                    torch.cuda.synchronize(peer)
                    x.copy_(there)

                out[str(rows)]["peer_round_trip"] = timed(hop, device, calls)
        queue.put((rank, out))
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001 - the record says what failed
        queue.put((rank, {"error": repr(exc)[:500]}))


def arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--port", type=int, default=29517)
    parser.add_argument("--variants", action="store_true")
    parser.add_argument("--q8", action="store_true", help="also the 8-bit two-shot path, at prefill sizes")
    return parser.parse_args(argv)


def main():
    args = arguments()
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")

    def measure(port, calls, env=None, floors=False, wait=120, q8=False):
        queue = context.Queue()
        rows_list = (1, 16, 48, 256, 1600, 3200) if q8 else (1, 8, 16, 32)
        jobs = [context.Process(target=worker, args=(rank, port, rows_list, calls, queue, env, floors, q8))
                for rank in range(RANKS)]
        for job in jobs:
            job.start()
        rows = {}
        try:
            for _ in jobs:
                rank, out = queue.get(timeout=wait)
                rows[str(rank)] = out
        except Exception as exc:  # noqa: BLE001 - a rank that never answered
            rows["error"] = repr(exc)[:300]
        for job in jobs:
            job.join(timeout=10)
            if job.is_alive():
                job.kill()
        return rows

    def save(record):
        with open(args.out, "w") as stream:
            json.dump(record, stream, indent=1)

    rows = measure(args.port, args.calls, floors=args.variants, q8=args.q8, wait=600 if args.q8 else 120)
    record = {"backend": "torch.distributed nccl", "hidden": HIDDEN, "ranks": RANKS, "calls": args.calls,
              "by_rank": rows, "rank0": rows.get("0")}
    save(record)
    if args.variants:  # each is its own NCCL start; the record is whole after every one
        import subprocess

        import torch

        record["peer_access"] = [[i == j or bool(torch.cuda.can_device_access_peer(i, j)) for j in range(RANKS)]
                                 for i in range(RANKS)]
        for name, command in (("links", ["nvidia-smi", "--query-gpu=index,pci.bus_id,pcie.link.gen.current,"
                                         "pcie.link.width.current", "--format=csv,noheader"]),
                              ("topology", ["nvidia-smi", "topo", "-m"])):
            try:
                record[name] = subprocess.run(command, capture_output=True, text=True, timeout=20).stdout[-3000:]
            except Exception as exc:  # noqa: BLE001 - a reading
                record[name] = repr(exc)[:200]
        record["variants"] = {}
        for k, (name, env) in enumerate(VARIANTS.items()):
            found = measure(args.port + 1 + k, 10 if name == "log" else 300, env, wait=45)
            record["variants"][name] = {"env": env, "rank0": found.get("0"), "error": found.get("error")}
            save(record)
    print("ARC2_L4_ALLREDUCE " + json.dumps(record.get("rank0")), flush=True)


if __name__ == "__main__":
    main()
