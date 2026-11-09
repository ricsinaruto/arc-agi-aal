"""CPU feasibility of MTP-k speculative decoding on the blocks3 serving (30 September; spec-decode.md).

Geometry (pinned vLLM; kv8_sizing.py / kv4_sizing.py rules, checked against the served blocks3 layout):
- GDN state page = conv 10,240 x (3 + k) x 2 B + recurrent 48 x 128 x 128 x 2 B (BF16 state; gated_delta_net_state_shape
  adds one conv row per speculative token).
- int4 block = smallest multiple of 64 tokens with block x 528 B >= page (kv4_block_tokens); int8 uses the indexer's
  1,024 B/token rule (16-token multiples).
- A block id spans every QSA layer: 12, plus the MTP layer's QSA when k > 0 (block bytes x 13 / 12).
- Per request: cdiv(tokens + k, block) attention blocks + 1 raw-key ring + 4 mamba groups (3 GDN + PLE) x (2 + k)
  state pages (MambaSpec.max_memory_usage_bytes, align mode).
- Pool: the served blocks3 plan (1,269 blocks x 22,278,144 B = 28.27 GB, graphs to 48) minus the MTP overhead
  (NvFp4 draft experts ~1.2 GiB + the MTP layer's attention/dense weights and larger graphs; 1.5 GiB central,
  1.0-2.5 GiB range) -- replaced by the pilot's measured num_blocks when available.
- Concurrency at window W: the frontier replay's 37 x 93K cell holds ~41 mean in flight at the 0.96 gate on MTP-0,
  i.e. ~20.7 attention blocks (65K tokens) per stream at 93K; scaled linearly in W (the resident-context share).
"""
import argparse
import json
from pathlib import Path

SERVED_BLOCKS, SERVED_BLOCK_BYTES = 1269, 22278144
GIB = 1 << 30
GATE = 0.96
ATTN_TOKENS_PER_STREAM_93K = 20.7 * 3136  # frontier replay: 41 in flight x 93K on the MTP-0 pool
TAX = {49152: 0.354, 65536: 0.24, 81920: 0.168, 93184: 0.14, 110592: 0.111, 130816: 0.089}


def page(k):
    return 10240 * (3 + k) * 2 + 48 * 128 * 128 * 2


def geometry(k, kv="int4"):
    per_token = {"int4": 528, "int8": 1040}[kv]
    if kv == "int4":
        block = 64 * -(-page(k) // (64 * per_token))
    else:
        block = 16 * -(-page(k) // (16 * 1024))
    compressed = block // 4 * 128 * 2
    layers = 12 + (k > 0)
    return {"block_tokens": block, "gdn_page": page(k), "block_bytes": layers * (block * per_token + compressed),
            "fixed_blocks": 1 + 4 * (2 + k), "layers": layers}


def pool_blocks(k, overhead_gib, kv="int4"):
    g = geometry(k, kv)
    served = SERVED_BLOCKS * SERVED_BLOCK_BYTES
    return int((served - (overhead_gib * GIB if k else 0)) // g["block_bytes"])


def streams_at(window, gate_blocks, g):
    attention = ATTN_TOKENS_PER_STREAM_93K * window / 93184 / g["block_tokens"]
    return gate_blocks / (g["fixed_blocks"] + attention)


def window_at(streams, gate_blocks, g):
    attention = gate_blocks / streams - g["fixed_blocks"]
    return max(0.0, attention * g["block_tokens"] / ATTN_TOKENS_PER_STREAM_93K * 93184)


def tax(window):
    points = sorted(TAX)
    if window <= points[0]:
        return TAX[points[0]]
    for a, b in zip(points, points[1:]):
        if window <= b:
            return TAX[a] + (TAX[b] - TAX[a]) * (window - a) / (b - a)
    return TAX[points[-1]]


def table(measured=None):
    rows = {}
    for kv in ("int4", "int8"):
        for k in (0, 1, 2, 3):
            g = geometry(k, kv)
            for label, overhead in (("central", 1.5), ("low", 1.0), ("high", 2.5)):
                if kv == "int8" and label != "central":
                    continue
                blocks = pool_blocks(k, overhead, kv)
                if kv == "int4" and measured and k in measured and label == "central":
                    blocks = measured[k]
                    label = "measured"
                gate = int(GATE * blocks)
                full = -(-(93184 + 256 + k) // g["block_tokens"]) + g["fixed_blocks"]
                rows[f"{kv}/mtp{k}/{label}"] = {
                    **g, "pool_blocks": blocks, "gate_blocks": gate, "blocks_per_93k_request": full,
                    "full_93k_windows": gate // full,
                    "state_bytes_per_request_mb": round(4 * (2 + k) * g["block_bytes"] / 1e6, 1),
                    "streams_at": {str(w): round(streams_at(w, gate, g), 1) for w in (65536, 81920, 93184)},
                    "window_at": {str(n): int(window_at(n, gate, g)) for n in (24, 31, 32, 37, 40)}}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--measured", default=None, help='JSON {"1": blocks, "2": blocks} from the pilots')
    args = ap.parse_args()
    measured = {int(k): v for k, v in json.loads(args.measured).items()} if args.measured else None
    rows = table(measured)
    print("| arm | block | fixed | pool | gate | per 93K | full 93K | streams @64K/82K/93K | window @24/31/32/37/40 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for name, r in rows.items():
        print(f"| {name} | {r['block_tokens']} | {r['fixed_blocks']} | {r['pool_blocks']} | {r['gate_blocks']} | "
              f"{r['blocks_per_93k_request']} | {r['full_93k_windows']} | "
              f"{' / '.join(str(v) for v in r['streams_at'].values())} | "
              f"{' / '.join(str(v // 1000) + 'K' for v in r['window_at'].values())} |")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
