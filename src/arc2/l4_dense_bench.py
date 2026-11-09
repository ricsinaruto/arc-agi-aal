"""Dense projections of a decode step on one L4, BF16 against weights stored in fewer bits (11 October).

The step profile (``arc2-l4-profile-v2``) has 16.9 ms of a 51 ms step in 776 BF16 matmuls whose time does not
move with the rows: each reads its whole weight once (6.9 MB in 24 us, 21 MB in 97 us, the 318 MB output head in
1.4 ms: the card's 220-290 GB/s), about 2.9 GiB a step and rank. Fewer calls would read the same bytes; fewer
bytes would not. This times, in the serving runtime without a server, one step's worth of each projection (the
shapes a rank holds at TP 4, ``vllm/models/qwen3_8_flash_next/nvidia/low_latency_gemm.py``) as a CUDA-graph replay:

- ``bf16``: ``torch.nn.functional.linear``, the served path (its sum should come out near the profile's);
- Humming's dense GEMM (the library of the 3-bit experts; vLLM's ``HummingLinearMethod`` serves a linear layer
  through it and can quantize BF16 weights at load) on the same weights rounded to each schema of ``SCHEMAS``.

Weights are random (the time is the bytes'); the error against the float32 product is a sanity check of the
path, not a quality figure: that is the routing screen's and a rehearsal's. Every call of a step reads its own copy
of the weight, as every layer does: one weight replayed (``arc2-l4-dense-v1``) is served from the card's cache and
came out at 6.2 ms a step where the profile has 16.9 (the 303 MiB head, too large for it, agreed: 1.4 ms).

    python -m arc2.l4_dense_bench --out dense.json [--rows 16,24,32] [--replays 30]
"""

import argparse
import json
import time
import traceback

# name, outputs, inputs, calls a step (48 layers: 36 with the GDN mixer, 12 with attention; two hyper-connection
# mixers a layer)
SHAPES = (("hc_down_inject", 336, 10240, 96), ("hc_up", 10240, 320, 96), ("gdn_qkvz", 4096, 2560, 36),
          ("mixer_out", 2560, 1536, 48), ("gdn_ba", 24, 2560, 36), ("attention_qkv_gate", 3584, 2560, 12),
          ("indexer_qk", 640, 2560, 12), ("shared_expert_gate_up", 320, 2560, 48), ("output_head", 62080, 2560, 1))
# name, weight type, scale type, inputs a scale, zero point
# (64 inputs a scale divides every shape's inputs: 128 does not divide the 320 of ``hc_up``, and gave garbage there;
# a zero point needs outputs in multiples of 32, which ``hc_down_inject`` and ``gdn_ba`` are not: v1's six errors)
SCHEMAS = (("int8_g64", "int8", "bfloat16", 64, False), ("fp8_e4m3", "float8e4m3", None, 0, False),
           ("uint4_zp_g64", "uint4", "bfloat16", 64, True))
COMPUTE = json.dumps({"use_batch_invariant": False, "use_f16_accum": False, "gemm_type": "dense"})


def step_bytes(bits=16):
    """Weight bytes one step reads at ``bits`` a weight."""
    return sum(n * k * calls for _, n, k, calls in SHAPES) * bits // 8


def log(row):
    print("ARC2_L4_DENSE " + json.dumps(row, default=str), flush=True)


def schema_of(name, weight, scale, group, zero_point):
    from humming import dtypes
    from humming.config.enum import WeightScaleType
    from humming.schema import HummingWeightSchema

    if scale is None:
        return HummingWeightSchema(b_dtype=getattr(dtypes, weight))
    return HummingWeightSchema(b_dtype=getattr(dtypes, weight), bs_dtype=getattr(dtypes, scale),
                               weight_scale_group_size=group, weight_scale_type=WeightScaleType.GROUP,
                               has_zero_point=zero_point)


def humming_layer(torch, schema, weight, copies=1):
    """The forward of one layer on Humming's dense GEMM (``forward(x, i)`` reads copy ``i`` of its tensors) and the
    bytes one copy holds."""
    from humming.forward import humming_forward
    from humming.schema import HummingInputSchema
    from humming.transform import prepare_layer_config, transform_humming_tensors

    tensors = schema.quant_tensor(weight, schema, torch.bfloat16)
    config = prepare_layer_config(shape_n=weight.shape[0], shape_k=weight.shape[1], weight_schema=schema,
                                  input_schema=HummingInputSchema(), pad_n_to_multiple=256, pad_k_to_multiple=128,
                                  has_bias=False, torch_dtype=torch.bfloat16)
    tensors = transform_humming_tensors(config, tensors)
    locks = torch.zeros(1024, dtype=torch.int32, device=weight.device)
    held = sum(t.numel() * t.element_size() for t in tensors.values() if torch.is_tensor(t))
    sets = [tensors, *({k: v.clone() if torch.is_tensor(v) else v for k, v in tensors.items()}
                       for _ in range(copies - 1))]

    def forward(x, i=0):
        t = sets[i % copies]
        out = humming_forward(config, inputs=x, weight=t["weight"], weight_scale=t.get("weight_scale"),
                              zero_point=t.get("zero_point"), bias=None, weight_scale_2=t.get("weight_scale_2"),
                              locks=locks, compute_config=COMPUTE)
        return out[:, :weight.shape[0]]

    return forward, held


def replay_ms(torch, forward, x, calls, replays):
    """Milliseconds of ``calls`` forwards as one CUDA-graph replay (the server's decode step is one graph); the
    eager time when the capture fails, and which it was."""
    for i in range(min(calls, 3)):
        forward(x, i)
    torch.cuda.synchronize()
    kind = "graph"
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(calls):
                forward(x, i)
        run = graph.replay
    except Exception:  # noqa: BLE001 - a kernel that cannot be captured is still timed
        kind = "eager"

        def run():
            for i in range(calls):
                forward(x, i)
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    times = []
    for _ in range(replays):
        begun = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - begun)
    times.sort()
    return round(times[len(times) // 2] * 1e3, 3), kind


def measure(rows_list, replays, out_path, distinct=True):
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda")
    results = {"env": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                       "capability": list(torch.cuda.get_device_capability())},
               "step_gib_bf16": round(step_bytes() / 2**30, 3), "distinct_weights": bool(distinct), "shapes": {},
               "errors": [], "totals": {}}
    try:
        import humming

        results["env"]["humming"] = getattr(humming, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        results["env"]["humming"] = f"unavailable: {type(exc).__name__}: {exc}"

    def flush():
        with open(out_path + ".part", "w") as stream:
            json.dump(results, stream, indent=1, default=str)
        import os

        os.replace(out_path + ".part", out_path)

    generator = torch.Generator(device="cpu").manual_seed(0)
    for name, n, k, calls in SHAPES:
        weight = (torch.randn((n, k), generator=generator) * 0.02).to(device=device, dtype=torch.bfloat16)
        entry = results["shapes"][name] = {"outputs": n, "inputs": k, "calls": calls,
                                           "bf16_mib": round(n * k * 2 / 2**20, 2), "variants": {}}
        copies = calls if distinct else 1
        own = [weight, *(weight.clone() for _ in range(copies - 1))]
        paths = {"bf16": (lambda x, i=0, own=own: F.linear(x, own[i % len(own)]), n * k * 2)}
        for label, *spec in SCHEMAS:
            try:
                paths[label] = humming_layer(torch, schema_of(label, *spec), weight, copies)
            except Exception as exc:  # noqa: BLE001 - a schema this build does not take
                results["errors"].append({"shape": name, "variant": label, "error": f"{type(exc).__name__}: {exc}"[:400],
                                          "traceback": traceback.format_exc()[-1200:]})
        for label, (forward, held) in paths.items():
            row = entry["variants"][label] = {"mib": round(held / 2**20, 2), "rows": {}}
            try:
                probe = torch.randn((32, k), generator=generator).to(device=device, dtype=torch.bfloat16)
                exact = probe.float() @ weight.float().T
                row["relative_error"] = float(((forward(probe).float() - exact).norm() / exact.norm()).item())
                for rows in rows_list:
                    x = torch.randn((rows, k), generator=generator).to(device=device, dtype=torch.bfloat16)
                    ms, kind = replay_ms(torch, forward, x, calls, replays)
                    row["rows"][str(rows)] = {"ms_a_step": ms, "us_a_call": round(ms * 1e3 / calls, 1), "timed": kind}
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"[:400]
                results["errors"].append({"shape": name, "variant": label, "error": row["error"],
                                          "traceback": traceback.format_exc()[-1200:]})
            log({"shape": name, "variant": label, **row})
        del weight, own, paths
        torch.cuda.empty_cache()
        flush()
    for label in ("bf16", *(s[0] for s in SCHEMAS)):  # a step's dense time: only where every shape was timed
        for rows in rows_list:
            parts = [results["shapes"][name]["variants"].get(label, {}).get("rows", {}).get(str(rows), {}).get("ms_a_step")
                     for name, *_ in SHAPES]
            if all(p is not None for p in parts):
                results["totals"].setdefault(label, {})[str(rows)] = round(sum(parts), 2)
        if label in results["totals"]:
            results["totals"][label]["gib_a_step"] = round(sum(
                results["shapes"][name]["variants"][label]["mib"] * calls for name, _, _, calls in SHAPES) / 1024, 3)
    log({"totals": results["totals"], "errors": len(results["errors"])})
    flush()
    return results


def arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", default="16,24,32")
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--one-weight", action="store_true", help="v1: every call on the same copy (cache-warm)")
    return parser.parse_args(argv)


def main():
    args = arguments()
    measure([int(r) for r in args.rows.split(",")], args.replays, args.out, distinct=not args.one_weight)


if __name__ == "__main__":
    main()
