#!/usr/bin/env python3
"""Tables for docs/validation-20260927/expert-usage.md from a trial's expert-usage.json (CPU only).

Usage: uv run python scripts/expert_usage_analyze.py RUN_DIR [--out docs/validation-20260927/expert-usage.json]
"""

import argparse
import glob
import json
import math
import os

import numpy as np

EXPERTS = 512
EXPERT_BYTES = 2.765e6        # one NvFp4 routed expert (gate_up 1280 x 2560 + down 2560 x 640, with scales)
MOE_LAYERS = 48
# MTP-3 phase-4 layout (docs/validation-20260927/kernels-phase4.md): 691 blocks of 21.8 MiB, 101 per 64K
# window, the 0.9 admission gate.
BLOCKS, BLOCK_BYTES, WINDOW_BLOCKS, GATE = 691, 21.8 * 2**20, 101, 0.9
# The MTP-0 hero layout (fixed-2x-mtp0-wide-kernels-kv8: A+C dense kernels, int8 KV; arena.variants): a 16.5 GiB pool
# of ~845 blocks of 1,584 tokens, 51 per 64K window at the 0.9 gate (14 windows today).
HERO_BLOCKS, HERO_BLOCK_BYTES, HERO_WINDOW_BLOCKS = 845, 16.5 * 2**30 / 845, 51
FETCH_US = 70.0               # one expert over PCIe (~40 GB/s)
TOP_K = 10


def windows(fraction, blocks=BLOCKS, block_bytes=BLOCK_BYTES, window_blocks=WINDOW_BLOCKS):
    freed = fraction * EXPERTS * MOE_LAYERS * EXPERT_BYTES
    extra = int(freed // block_bytes)
    return freed, extra, int(GATE * (blocks + extra) // window_blocks)


def hero_windows(fraction):
    return windows(fraction, HERO_BLOCKS, HERO_BLOCK_BYTES, HERO_WINDOW_BLOCKS)


def coverage_rows(mass):
    mass = np.asarray(mass, dtype=np.float64)
    ordered = -np.sort(-mass, axis=1)
    cum = np.cumsum(ordered, axis=1) / ordered.sum(axis=1, keepdims=True)
    return {f: np.array([int(np.searchsorted(row, f - 1e-12) + 1) for row in cum]) for f in (0.5, 0.9, 0.99, 0.999)}


def bootstrap_delta(per_sequence, key_ref, key_new, key_n, reps=4000, seed=0):
    """Position-weighted mean delta and a 95% interval, resampling sequences."""
    ref = np.array([r[key_ref] for r in per_sequence])
    new = np.array([r[key_new] for r in per_sequence])
    n = np.array([r[key_n] for r in per_sequence], dtype=float)
    rng = np.random.default_rng(seed)
    point = (new.sum() - ref.sum()) / n.sum()
    draws = []
    for _ in range(reps):
        i = rng.integers(0, len(n), len(n))
        draws.append((new[i].sum() - ref[i].sum()) / n[i].sum())
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--out")
    args = parser.parse_args()
    path = (glob.glob(os.path.join(args.run, "**/expert-usage/expert-usage.json"), recursive=True) or [args.run])[0]
    data = json.load(open(path))
    usage, held = data["usage"], data["heldout"]
    layers = usage["dump_layers"]
    summary = {"source": path, "status": data.get("status"), "errors": data.get("errors"),
               "skipped": data.get("skipped"), "layers": len(layers)}

    tokens = np.asarray(usage["tokens"])
    count = np.asarray(usage["count"])
    weight = np.asarray(usage["weight"])
    print(f"usage tokens per layer {tokens.min():.0f}-{tokens.max():.0f}; selections/token "
          f"{(count.sum(1) / tokens).mean():.3f}; weight/token {(weight.sum(1) / tokens).mean():.4f}")
    print("usage timing", usage["timing"], "NLL", {k: v for k, v in usage["nll"].items()})
    for label, stats in (("usage", usage), ("held-out", held)):
        for key in ("count", "weight"):
            cov = coverage_rows(stats[key])
            row = {str(f): {"min": int(v.min()), "median": float(np.median(v)), "max": int(v.max()),
                            "mean": float(v.mean())} for f, v in cov.items()}
            summary.setdefault("coverage", {}).setdefault(label, {})[key] = row
            print(f"{label:8s} {key:6s} " + "  ".join(
                f"{f}: {r['min']}/{r['median']:.0f}/{r['max']} (mean {r['mean']:.0f})" for f, r in row.items()))
    cov_w = coverage_rows(usage["weight"])
    cov_c = coverage_rows(usage["count"])
    summary["per_layer"] = [{"layer": layer, **{f"w{f}": int(cov_w[f][i]) for f in cov_w},
                             **{f"c{f}": int(cov_c[f][i]) for f in cov_c},
                             "min_share": float(count[i].min() / count[i].sum()),
                             "unused": int((count[i] == 0).sum())} for i, layer in enumerate(layers)]
    uniform = 1 / EXPERTS
    shares = count / count.sum(axis=1, keepdims=True)
    summary["balance"] = {"max_share_over_uniform": float((shares.max(1) / uniform).mean()),
                          "min_share_over_uniform": float((shares.min(1) / uniform).mean()),
                          "unused_experts_total": int((count == 0).sum()),
                          "entropy_bits_mean": float((-(shares * np.log2(np.where(shares > 0, shares, 1)))).sum(1).mean())}
    print("balance", summary["balance"])

    # Transfer of the usage ranking to held-out games: held-out routing mass on each mask's pruned experts.
    pruned_mass = held.get("pruned_mass", {})
    rows = []
    for fraction in sorted(pruned_mass, key=float):
        f = float(fraction)
        freed, extra, fit = windows(f)
        _, hero_extra, hero_fit = hero_windows(f)
        c = pruned_mass[fraction]["count"]
        w = pruned_mass[fraction]["weight"]
        fetch_ms = c["mean"] * MOE_LAYERS * TOP_K * FETCH_US / 1000
        rows.append({"fraction": f, "freed_gib": freed / 2**30, "extra_blocks": extra, "windows_64k": fit,
                     "hero_extra_blocks": hero_extra, "hero_windows_64k": hero_fit,
                     "heldout_count_mass": c["mean"], "heldout_count_mass_max_layer": c["max"],
                     "heldout_weight_mass": w["mean"], "heldout_weight_mass_max_layer": w["max"],
                     "host_fetch_ms_per_token": fetch_ms})
    summary["pruning_memory"] = rows
    for r in rows:
        print(f"{r['fraction']:.3f}: {r['freed_gib']:.1f} GiB, +{r['extra_blocks']} blocks, {r['windows_64k']} windows (hero +{r['hero_extra_blocks']}, {r['hero_windows_64k']}); "
              f"held-out mass on pruned: count {100 * r['heldout_count_mass']:.3f}% (max layer "
              f"{100 * r['heldout_count_mass_max_layer']:.2f}%), weight {100 * r['heldout_weight_mass']:.3f}%; "
              f"host fetch {r['host_fetch_ms_per_token']:.2f} ms/token")

    ref = data["passes"].get("reference", {})
    print("reference", ref.get("timing"), ref.get("nll"))
    curve = []
    for name, entry in data["passes"].items():
        if name == "reference":
            continue
        s = entry["summary"]
        row = {"pass": name, "fraction": entry.get("fraction"), "seconds": entry["timing"]["seconds"],
               "failed": len(entry["timing"]["failed"])}
        for scope in ("all", "assistant"):
            if scope in s:
                row[scope] = s[scope]
        seqs = entry["per_sequence"]
        row["delta_nll_ci"] = {
            "all": bootstrap_delta(seqs, "nll_ref_sum", "nll_sum", "positions"),
            "assistant": bootstrap_delta(seqs, "assistant_nll_ref_sum", "assistant_nll_sum", "assistant_positions")}
        kl_seq = np.array([r["assistant_kl_sum"] / max(r["assistant_positions"], 1) for r in seqs])
        row["assistant_kl_per_sequence"] = {"min": float(kl_seq.min()), "median": float(np.median(kl_seq)),
                                            "max": float(kl_seq.max())}
        curve.append(row)
    noise = next((r for r in curve if r["pass"] == "repeat"), None)
    for r in curve:
        for scope in ("all", "assistant"):
            if noise and scope in r and scope in noise and noise[scope]["mean_kl"]:
                r[scope]["kl_over_noise"] = r[scope]["mean_kl"] / noise[scope]["mean_kl"]
        a = r.get("assistant", {})
        d = r["delta_nll_ci"]["assistant"]
        print(f"{r['pass']:12s} assistant: KL {a.get('mean_kl', float('nan')):.4f} "
              f"({a.get('kl_over_noise', float('nan')):.2f}x noise) p50 {a.get('p50_kl', math.nan):.4f} "
              f"p99 {a.get('p99_kl', math.nan):.3f} top1 {100 * a.get('top1_agreement', math.nan):.2f}% "
              f"out-of-top20 {100 * a.get('top1_outside_ref_top20', math.nan):.3f}% dNLL {d[0]:+.4f} "
              f"[{d[1]:+.4f}, {d[2]:+.4f}] ({r['seconds']} s)")
        a = r.get("all", {})
        d = r["delta_nll_ci"]["all"]
        print(f"{'':12s} all:       KL {a.get('mean_kl', math.nan):.4f} ({a.get('kl_over_noise', math.nan):.2f}x) "
              f"top1 {100 * a.get('top1_agreement', math.nan):.2f}% dNLL {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}] "
              f"NLL ref {a.get('nll_ref', math.nan):.4f}")
    summary["curve"] = curve
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
            f.write("\n")


if __name__ == "__main__":
    main()
