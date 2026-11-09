"""8-bit QSA main KV cache: blocks per 64K request and 64K fit (CPU sizing, 28 September).

Geometry from the pinned vLLM (runtime layer 24):
- ``platforms/interface.py`` ``_align_hybrid_block_size``: block = 16 * cdiv(GDN page, 16 * attention bytes per
  token per layer), with the attention bytes from ``backend.customize_spec`` of the first non-SSM backend.
- ``kv_cache_utils._get_kv_cache_groups_csa_linear`` / ``_CSALinearTensorLayout``: one block id spans every QSA
  layer (12, plus the MTP layer on MTP-k) and costs ``main_kv_page + compressed_page`` per layer; the GDN and PLE
  state pages are padded to the main-KV page and alias the main-KV tensors.
- A request costs cdiv(tokens + lookahead, block) attention blocks + 1 raw-key ring block + 4 mamba groups x
  (2 + draft tokens) state pages (align mode, prefix caching on). The arena admits at 0.9 of the blocks.

Pools are the served KV bytes: phase-4 MTP-3 (691 blocks of 832 tokens, 14.7 GiB), the MTP-0 control pinned pool
(751 blocks of 800, KERNEL_GATE_KV_BYTES) and the MTP-0 A+C pool unpinned (+2.4 GiB, phase 3).
"""

import argparse
import json
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs/validation-20260927"
KV_HEADS, HEAD_DIM, INDEXER_DIM, RATIO = 2, 256, 128, 4
GATE = 0.9
REQUEST = 65536 + 256  # a full 64K window plus the arena's fresh-clock/template overhead
GIB = 1 << 30


def gdn_page(ns, state_bytes):
    """GDN state page: conv 10240 x (3 + ns) x 2 B + recurrent 48 x 128 x 128 x state bytes."""
    return 10240 * (3 + ns) * 2 + 48 * 128 * 128 * state_bytes


def per_token(kv):
    """Main-KV bytes per token per QSA layer (both KV heads, K and V): BF16, or 8-bit with inline fp32 scales."""
    if kv == "bf16":
        return KV_HEADS * 2 * HEAD_DIM * 2
    if kv == "8bit":  # customize_spec: (256 + 256) x 1 B + 2 x 4 B per head
        return KV_HEADS * (2 * HEAD_DIM + 8)
    if kv == "8bit-nopad":  # what the block-size rule sees if the first non-SSM backend is the indexer's identity
        return KV_HEADS * 2 * HEAD_DIM
    raise ValueError(kv)


def geometry(ns, state_bytes, kv, rule=None):
    rule = rule or kv
    page = gdn_page(ns, state_bytes)
    block = 16 * -(-page // (16 * per_token(rule)))
    layers = 12 + (ns > 0)
    main = block * per_token(kv)
    compressed = block // RATIO * INDEXER_DIM * 2
    return {"block_tokens": block, "gdn_page": page, "main_kv_page": main, "compressed_page": compressed,
            "block_bytes": layers * (main + compressed), "layers": layers}


def fit(pool_bytes, ns, state_bytes, kv, rule=None, request=REQUEST):
    g = geometry(ns, state_bytes, kv, rule)
    blocks = int(pool_bytes // g["block_bytes"])
    gate = int(blocks * GATE)
    attention = -(-(request + ns) // g["block_tokens"])
    fixed = 1 + 4 * (2 + ns)
    cost = attention + fixed
    return {**g, "pool_gib": round(pool_bytes / GIB, 3), "blocks": blocks, "gate_blocks": gate,
            "attention_blocks": attention, "fixed_blocks": fixed, "blocks_per_64k": cost,
            "fit_64k": gate // cost, "fit_exact": round(gate / cost, 2), "spare_blocks": gate - (gate // cost) * cost,
            "pool_tokens_per_request_slot": g["block_tokens"] * attention}


def pools():
    phase4 = geometry(3, 2, "bf16")
    mtp0 = geometry(0, 2, "bf16")
    return {
        # Served (kernels-phase4.md): 691 blocks x 13 x (832 x 2048 + 208 x 256) bytes = 14.7 GiB.
        "mtp3_phase4": {"ns": 3, "state_bytes": 2, "pool_bytes": 691 * phase4["block_bytes"],
                        "served": {"blocks": 691, "block_tokens": 832, "gate_blocks": 621, "blocks_per_64k": 101,
                                   "fit_64k": 6}},
        # Served (serving lab mtp0, c110 mtp0-wide, KERNEL_GATE_KV_BYTES): 751 blocks of 800 tokens.
        "mtp0_pinned": {"ns": 0, "state_bytes": 2, "pool_bytes": 751 * mtp0["block_bytes"],
                        "served": {"blocks": 751, "block_tokens": 800, "gate_blocks": 675, "blocks_per_64k": 92,
                                   "fit_64k": 7}},
        # Phase 3: the A+C server frees 2.4 GiB that phase 2 pinned away.
        "mtp0_ac_unpinned": {"ns": 0, "state_bytes": 2, "pool_bytes": 751 * mtp0["block_bytes"] + 2.4 * GIB,
                             "served": None},
    }


def table():
    rows = {}
    for name, pool in pools().items():
        row = {"served": pool["served"]}
        for label, kv, rule in (("bf16", "bf16", None), ("kv8", "8bit", None), ("kv8_rule_nopad", "8bit", "8bit-nopad")):
            row[label] = fit(pool["pool_bytes"], pool["ns"], pool["state_bytes"], kv, rule)
        row["gain_64k"] = row["kv8"]["fit_64k"] / row["bf16"]["fit_64k"] - 1
        rows[name] = row
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DOC / "kv-cache-8bit-sizing.json"))
    args = ap.parse_args()
    rows = table()
    print("| Pool | KV | Block tokens | Blocks | Gate | Per 64K (attn + fixed) | 64K fit | Spare |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, row in rows.items():
        for label in ("bf16", "kv8", "kv8_rule_nopad"):
            f = row[label]
            print(f"| {name} | {label} | {f['block_tokens']} | {f['blocks']} | {f['gate_blocks']} | "
                  f"{f['blocks_per_64k']} ({f['attention_blocks']} + {f['fixed_blocks']}) | {f['fit_64k']} "
                  f"({f['fit_exact']}) | {f['spare_blocks']} |")
    for name, row in rows.items():
        served = row["served"]
        if served:
            b = row["bf16"]
            assert (b["block_tokens"], b["blocks"], b["gate_blocks"], b["fit_64k"]) == (
                served["block_tokens"], served["blocks"], served["gate_blocks"], served["fit_64k"]), (name, b)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
