"""4-bit QSA main KV cache: blocks per window and fits on the hero layout (CPU sizing, 29 September).

Extends ``scripts/kv8_sizing.py`` (same block rule and request cost, which reproduce the served layouts) with the
pinned vLLM's sub-byte per-token-head layouts (``TritonAttentionBackend.customize_spec``):

- ``int4``: ``int4_per_token_head`` (two 4-bit codes per byte after a random Hadamard rotation, an fp32 scale per
  (token, head, K|V) with the zero point in its low mantissa bits): (128 + 128) B + 2 x 4 B = 264 B per token and head;
- ``int4g32``: the same codes with an fp16 scale per 32 channels (a finer grid, not in the pinned vLLM):
  (128 + 128) B + 2 x 8 x 2 B = 288 B;
- ``k8v4``: 8-bit keys, 4-bit values (the pinned ``turboquant_k8v4`` bit budget): (256 + 128) B + 2 x 4 B = 392 B.

The hero pool is the served one (hero-smoke.md: MTP-0, int8 KV, 860 blocks, 51 per 64K window, 15 fit); the pruned
pools add the load-time pruning's measured savings (expert-usage.md: 7.59 GiB at 12.5%, 15.88 GiB at 25%).
``max_num_seqs`` (25) caps the served focus slots.

hero-v2 (``hero_v2_table``, 29 September; kv-cache-4bit.md): the served pool is measured (hero-v2 smoke v2: 1,335
blocks of 1,600 tokens at int8 with prune12 and the serving tweaks, 51 per 64K, 23 fit), and the block sizes are the
served ones: kv8's block rule sees 1,024 B per token (the indexer's backend: 1,600 tokens), and the 4-bit modes set
``--block-size`` (``serving_kernels_patch.KV4_BLOCK_TOKENS``: int4 3,136, k8v4 2,112, the smallest multiples of 64
whose main-KV page holds the MTP-0 GDN state page, which the CSA+linear layout requires).

    uv run python scripts/kv4_sizing.py [--json docs/validation-20260927/expert-quant/kv4-sizing.json]
    uv run python scripts/kv4_sizing.py --hero-v2 [--json docs/validation-20260927/kv-cache-4bit-sizing.json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kv8_sizing as kv8  # noqa: E402

GIB = 1 << 30
MAX_NUM_SEQS = 25
PER_TOKEN = {"bf16": 2 * 1024, "kv8": 2 * (512 + 8), "k8v4": 2 * (384 + 8), "int4g32": 2 * (256 + 32),
             "int4": 2 * (256 + 8)}
HERO = {"blocks": 860, "per_64k": 51, "fit_64k": 15}
PRUNE_SAVINGS = {"none": 0.0, "prune12.5": 7.59, "prune25": 15.88}


def geometry(per_token, ns=0, state_bytes=2):
    page = kv8.gdn_page(ns, state_bytes)
    block = 16 * -(-page // (16 * per_token))
    layers = 12 + (ns > 0)
    compressed = block // kv8.RATIO * kv8.INDEXER_DIM * 2
    return {"block_tokens": block, "block_bytes": layers * (block * per_token + compressed)}


def fit(pool_bytes, per_token, window, ns=0):
    g = geometry(per_token, ns)
    blocks = int(pool_bytes // g["block_bytes"])
    gate = int(blocks * kv8.GATE)
    request = window + 256
    cost = -(-(request + ns) // g["block_tokens"]) + 1 + 4 * (2 + ns)
    return {"block_tokens": g["block_tokens"], "blocks": blocks, "gate": gate, "per_window": cost,
            "fit": gate // cost, "served_fit": min(gate // cost, MAX_NUM_SEQS),
            "pool_tokens": blocks * g["block_tokens"]}


def table():
    hero_pool = HERO["blocks"] * geometry(PER_TOKEN["kv8"])["block_bytes"]
    check = fit(hero_pool, PER_TOKEN["kv8"], 65536)
    assert (check["blocks"], check["per_window"], check["fit"]) == (HERO["blocks"], HERO["per_64k"], HERO["fit_64k"])
    rows = {}
    for prune, gib in PRUNE_SAVINGS.items():
        pool = hero_pool + gib * GIB
        for kv, per_token in PER_TOKEN.items():
            rows[f"{prune}/{kv}"] = {"pool_gib": round(pool / GIB, 2), "kv": kv, "prune": prune,
                                     "bytes_per_token_layer": per_token,
                                     "64k": fit(pool, per_token, 65536), "128k": fit(pool, per_token, 131072)}
    return rows


# hero-v2 smoke v2 (hero-v2.md): the served int8 pool and its geometry.
HERO_V2 = {"blocks": 1335, "block_tokens": 1600, "per_64k": 51, "fit_64k": 23, "max_num_seqs": 25, "seq_cap": 32}
SERVED_BLOCK_TOKENS = {"kv8": 1600, "int4": 3136, "k8v4": 2112}
HERO_V2_KV = ("kv8", "k8v4", "int4")


def served_geometry(kv):
    block = SERVED_BLOCK_TOKENS[kv]
    assert block * PER_TOKEN[kv] >= kv8.gdn_page(0, 2), kv  # the CSA+linear main-KV page holds the GDN state page
    compressed = block // kv8.RATIO * kv8.INDEXER_DIM * 2
    return {"block_tokens": block, "block_bytes": 12 * (block * PER_TOKEN[kv] + compressed)}


def served_fit(pool_bytes, kv, window):
    g = served_geometry(kv)
    blocks = int(pool_bytes // g["block_bytes"])
    gate = int(blocks * kv8.GATE)
    cost = -(-(window + 256) // g["block_tokens"]) + 1 + 4 * 2
    fit_ = gate // cost
    return {**g, "blocks": blocks, "gate": gate, "per_window": cost, "fit": fit_,
            "served_fit": min(fit_, HERO_V2["max_num_seqs"]), "capture_cap_fit": min(fit_, HERO_V2["seq_cap"]),
            "pool_tokens": blocks * g["block_tokens"]}


def hero_v2_table(pool_bytes=None):
    """{kv: {"64k": ..., "128k": ...}} on the hero-v2 served pool (or ``pool_bytes``, e.g. a pinned test pool)."""
    pool = HERO_V2["blocks"] * served_geometry("kv8")["block_bytes"] if pool_bytes is None else pool_bytes
    check = served_fit(pool, "kv8", 65536)
    if pool_bytes is None:
        assert (check["blocks"], check["block_tokens"], check["per_window"], check["fit"]) == (
            HERO_V2["blocks"], HERO_V2["block_tokens"], HERO_V2["per_64k"], HERO_V2["fit_64k"]), check
    return {kv: {"pool_gib": round(pool / GIB, 3), "bytes_per_token_layer": PER_TOKEN[kv],
                 "64k": served_fit(pool, kv, 65536), "128k": served_fit(pool, kv, 131072)} for kv in HERO_V2_KV}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--hero-v2", action="store_true")
    args = ap.parse_args()
    if args.hero_v2:
        rows = hero_v2_table()
        print("| KV | B/token/layer | Block tokens | Block MB | Blocks | Gate | Pool tokens | Per 64K | 64K fit "
              "(cap 25 / 32) | Per 128K | 128K fit |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for kv, row in rows.items():
            a, b = row["64k"], row["128k"]
            print(f"| {kv} | {row['bytes_per_token_layer']} | {a['block_tokens']:,} | {a['block_bytes'] / 1e6:.2f} | "
                  f"{a['blocks']:,} | {a['gate']:,} | {a['pool_tokens']:,} | {a['per_window']} | {a['fit']} "
                  f"({a['served_fit']} / {a['capture_cap_fit']}) | {b['per_window']} | {b['fit']} |")
        if args.json:
            Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")
        return
    rows = table()
    print("| Experts | KV | B/token/layer | Block tokens | Blocks | Pool tokens | Per 64K | 64K fit (served, cap 25) | "
          "Per 128K | 128K fit |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows.values():
        a, b = row["64k"], row["128k"]
        print(f"| {row['prune']} | {row['kv']} | {row['bytes_per_token_layer']} | {a['block_tokens']} | {a['blocks']} | "
              f"{a['pool_tokens']:,} | {a['per_window']} | {a['fit']} ({a['served_fit']}) | {b['per_window']} | "
              f"{b['fit']} |")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
