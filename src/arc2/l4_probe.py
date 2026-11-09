"""L4 port probe (ARC-AGI-2 arm, 9 October): what the pinned serving runtime does on an L4, without a server.

Run inside the extracted runtime (its torch 2.13 + cu130 and vLLM fork). Every step is independent and recorded:

- ``torch``: versions, the wheel's architecture list, the devices, a BF16 matmul on each (does cu130 run on sm_89 with
  this host's driver at all).
- ``bandwidth``: read rate of a 6 GiB tensor on one card, GB/s (decode on the RTX PRO 6000 is bound by it: 1.53 TB/s).
- ``gemm``: BF16 GEMMs at decode shapes (rows x 2,560 against 2,560 / 10,240 / the 248,320-row output head).
- ``source``: the fork's model and MoE files with every line on tensor / pipeline parallelism, MoE backends and
  capability checks, so the multi-GPU path can be chosen from its own code (no copy of the fork exists in the repo).

    python -m arc2.l4_probe --site <runtime dist-packages> --out probe.json
"""

import argparse
import json
import os
import re
import time
import traceback

PATTERN = re.compile(
    r"tensor_parallel|pipeline_parallel|get_tensor_model_parallel|get_pp_group|SupportsPP|PPMissingLayer|make_layers|"
    r"tp_size|tp_rank|world_size|all_reduce|moe_backend|humming|marlin|cutlass|b12x|get_device_capability|"
    r"sm_?1[02]\d|sm_?[89]\d|\(1[02], \d\)|compute_capability|device_map|cuda:\d", re.I)
PATHS = ("qwen3_8_flash_next", "fused_moe", "humming_utils", "parallel_utils")
ROWS = (16, 32, 56)


def step(result, name, call):
    started = time.time()
    try:
        result[name] = call()
    except Exception as exc:  # noqa: BLE001 - every step is recorded, none ends the probe
        result[name] = {"error": f"{type(exc).__name__}: {str(exc)[:600]}", "traceback": traceback.format_exc()[-1500:]}
    result.setdefault("seconds", {})[name] = round(time.time() - started, 1)
    print(json.dumps({"event": "l4_probe", "step": name, "seconds": result["seconds"][name],
                      "error": (result[name] or {}).get("error") if isinstance(result[name], dict) else None}), flush=True)


def torch_facts():
    import torch

    out = {"torch": torch.__version__, "cuda": torch.version.cuda, "arch_list": torch.cuda.get_arch_list(),
           "devices": torch.cuda.device_count(), "cards": []}
    for index in range(out["devices"]):
        props = torch.cuda.get_device_properties(index)
        x = torch.randn(256, 2560, device=f"cuda:{index}", dtype=torch.bfloat16)
        y = (x @ x.t()).float().sum().item()
        out["cards"].append({"name": props.name, "capability": list(torch.cuda.get_device_capability(index)),
                             "total_gib": round(props.total_memory / 2**30, 2), "processors": props.multi_processor_count,
                             "matmul_finite": y == y})
    for module in ("vllm", "humming", "triton", "flashinfer"):
        try:
            out[module] = getattr(__import__(module), "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            out[module] = f"unavailable: {type(exc).__name__}: {str(exc)[:200]}"
    return out


def timed(torch, call, repeats=5):
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return min(samples)


def bandwidth():
    import torch

    x = torch.ones(6 * 2**30 // 8, dtype=torch.int64, device="cuda:0")
    seconds = timed(torch, lambda: x.sum())
    return {"read_gib": 6, "seconds": round(seconds, 4), "read_gb_per_second": round(6 * 2**30 / 1e9 / seconds, 1)}


def gemm():
    import torch

    def one(n):
        w = torch.randn(n, 2560, device="cuda:0", dtype=torch.bfloat16)
        row = {"gb": round(w.numel() * 2 / 1e9, 3)}
        for m in ROWS:
            x = torch.randn(m, 2560, device="cuda:0", dtype=torch.bfloat16)
            row[f"rows{m}_ms"] = round(1000 * timed(torch, lambda x=x: x @ w.t()), 3)
        return row

    out = {}
    for n in (2560, 10240, 248320):
        out[f"{n}x2560"] = one(n)
        torch.cuda.empty_cache()
    return out


def source(site):
    root = os.path.join(site, "vllm")
    files, hits, kept = [], [], 0
    for folder, _, names in os.walk(root):
        for name in sorted(names):
            path = os.path.join(folder, name)
            rel = os.path.relpath(path, site)
            if not name.endswith(".py") or not any(part in rel for part in PATHS):
                continue
            lines = open(path, errors="replace").read().splitlines()
            found = [(n, line.strip()[:220]) for n, line in enumerate(lines, 1) if PATTERN.search(line)]
            files.append({"file": rel, "lines": len(lines), "matches": len(found)})
            for n, line in found[:60]:
                if kept < 1500:
                    hits.append(f"{rel}:{n}: {line}")
                    kept += 1
    return {"files": files, "hits": hits}


DUMP = ("vllm/models/qwen3_8_flash_next/nvidia/", "vllm/models/qwen3_8_flash_next/common/",
        "vllm/model_executor/layers/fused_moe/layer.py", "vllm/model_executor/layers/fused_moe/config.py",
        "vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py",
        "vllm/model_executor/layers/fused_moe/oracle/nvfp4.py", "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py",
        "vllm/model_executor/layers/quantization/utils/humming_utils.py")


def dump(site, folder):
    """The whole text of the model's own files and the expert-layer glue, for reading how they split over cards."""
    import shutil

    copied = []
    for base, _, names in os.walk(os.path.join(site, "vllm")):
        for name in sorted(names):
            rel = os.path.relpath(os.path.join(base, name), site)
            if name.endswith(".py") and rel.startswith(DUMP):
                target = os.path.join(folder, rel)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copyfile(os.path.join(site, rel), target)
                copied.append(rel)
    return {"files": len(copied)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dump", help="folder for the full text of the model's files")
    args = parser.parse_args(argv)
    result = {}
    for name, call in (("torch", torch_facts), ("source", lambda: source(args.site)), ("bandwidth", bandwidth),
                       ("gemm", gemm), *([("dump", lambda: dump(args.site, args.dump))] if args.dump else [])):
        step(result, name, call)
        with open(args.out, "w") as stream:
            json.dump(result, stream, indent=1)
    return result


if __name__ == "__main__":
    main()
