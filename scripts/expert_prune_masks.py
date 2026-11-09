"""Expert-prune masks from the v4 usage measurement (docs/validation-20260927/expert-usage.md).

Ranking (as in ``arena.expert_usage.masks``): per layer, the least usage-set gate-weight mass first, routing count
breaking ties. Writes ``data/expert_prune/``:

- ``prune18.json``: 96 of 512 per layer (18.75%), uniform;
- ``prune25-layer.json``: layer-aware 25%, 6,144 experts in total (128 per layer on average). Each layer's mass
  is normalized to 1 and the smallest shares are pruned greedily in blocks of 8 experts (each layer's next 8 by
  rank, costed by their summed share), so concentrated layers (a thin tail) give up more experts than flat ones.
  Bounds: 64..192 per layer, and layers 0-2 are held to 64 (12.5%). Every kept count is a multiple of 8, like the
  uniform masks, so no MoE kernel sees an expert count it has not been run with modulo 8.
- ``prune37-layer.json`` (the blocks-2 campaign arm with 37.5% pruning): the same method at
  9,216 experts (192 per layer on average), the bounds scaled with the average (64..192 at 128 -> 96..288 at 192) and
  layers 0-2 held to 96 (18.75%). Kept counts 224..416, multiples of 8. Its per-layer held-out routing on the pruned
  set is recorded in the mask (``heldout_count_on_pruned_per_layer``).
- ``prune50-layer.json`` ("blocks-2 + plans, pruned 50%, 3-bit experts"): the same method at
  12,288 experts (256 per layer on average), the bounds scaled with the average (64..192 at 128 -> 128..384 at 256)
  and layers 0-2 held to 128 (25%). Kept counts 128..384, multiples of 8, with the per-layer held-out routing.

The existing uniform masks (prune12, prune25) are re-derived and checked, not rewritten.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERTS = 512
SOURCE_RUN = "arena-expert-usage-v4-modal-20260928"
RANK_BY = "usage-set gate-weight mass (renormalized top-10), routing count breaks ties"
LAYER_AWARE = {"total": 6144, "low": 64, "high": 192, "caps": {0: 64, 1: 64, 2: 64}, "step": 8}
LAYER_AWARE_37 = {"total": 9216, "low": 96, "high": 288, "caps": {0: 96, 1: 96, 2: 96}, "step": 8}
LAYER_AWARE_50 = {"total": 12288, "low": 128, "high": 384, "caps": {0: 128, 1: 128, 2: 128}, "step": 8}


def ranking(weight, count):
    return np.stack([np.lexsort((count[i], weight[i])) for i in range(weight.shape[0])])


def layer_aware_counts(weight, order, total, low, high, caps, step=1):
    """Pruned count per layer: the ``total`` smallest per-layer-normalized shares, in blocks of ``step``, within
    the bounds."""
    share = weight / weight.sum(1, keepdims=True)
    ranked = np.take_along_axis(share, order, 1)
    layers = weight.shape[0]
    limit = np.array([caps.get(i, high) for i in range(layers)])
    counts = np.minimum(np.full(layers, low), limit)
    if counts.sum() > total or limit.sum() < total or any(v % step for v in (total, low, *limit)):
        raise ValueError("the bounds cannot reach the total in blocks of the step")
    while counts.sum() < total:
        open_layers = np.flatnonzero(counts < limit)
        nxt = [ranked[i, counts[i]:counts[i] + step].sum() for i in open_layers]
        counts[open_layers[int(np.argmin(nxt))]] += step
    return counts


def pruned_mass(order, counts, weight, count):
    """Mean and max over layers of the routing (count, weight) share the pruned experts carried unmasked."""
    out = {}
    for key, stat in (("count", count), ("weight", weight)):
        share = np.take_along_axis(stat / stat.sum(1, keepdims=True), order, 1)
        lost = np.array([share[i, :counts[i]].sum() for i in range(len(counts))])
        out[key] = {"mean": float(lost.mean()), "max": float(lost.max())}
    return out


def pruned_per_layer(order, counts, stat):
    """Each layer's held-out routing share (``stat``: count or weight) on its pruned experts."""
    share = np.take_along_axis(stat / stat.sum(1, keepdims=True), order, 1)
    return [float(share[i, :counts[i]].sum()) for i in range(len(counts))]


def mask(order, counts, heldout, fraction, extra):
    pruned = {str(i): sorted(int(e) for e in order[i, :counts[i]]) for i in range(len(counts))}
    mass = pruned_mass(order, counts, *heldout)
    return {"format": "arc-expert-prune-v1", "experts": EXPERTS, "fraction": fraction, "rank_by": RANK_BY,
            "source_run": SOURCE_RUN, "source_doc": "docs/validation-20260927/expert-usage.md",
            "heldout_routing_on_pruned": {"count_mean": mass["count"]["mean"], "count_max_layer": mass["count"]["max"],
                                          "weight_mean": mass["weight"]["mean"]},
            "pruned": pruned, **extra}


def build(usage):
    u, h = usage["usage"], usage["heldout"]
    weight, count = (np.asarray(u[k], dtype=np.float64) for k in ("weight", "count"))
    heldout = tuple(np.asarray(h[k], dtype=np.float64) for k in ("weight", "count"))
    order = ranking(weight, count)
    layers = weight.shape[0]
    for fraction, name in ((0.125, "prune12"), (0.25, "prune25")):
        m = int(round(fraction * EXPERTS))
        want = usage["masks"][str(fraction)]
        if any(sorted(order[i, :m].tolist()) != sorted(want[str(i)]) for i in range(layers)):
            raise RuntimeError(f"{name}: the ranking no longer reproduces the v4 mask")
    uniform18 = np.full(layers, 96)
    counts = layer_aware_counts(weight, order, LAYER_AWARE["total"], LAYER_AWARE["low"], LAYER_AWARE["high"],
                                LAYER_AWARE["caps"], LAYER_AWARE["step"])
    counts37 = layer_aware_counts(weight, order, LAYER_AWARE_37["total"], LAYER_AWARE_37["low"],
                                  LAYER_AWARE_37["high"], LAYER_AWARE_37["caps"], LAYER_AWARE_37["step"])
    counts50 = layer_aware_counts(weight, order, LAYER_AWARE_50["total"], LAYER_AWARE_50["low"],
                                  LAYER_AWARE_50["high"], LAYER_AWARE_50["caps"], LAYER_AWARE_50["step"])
    return {
        "prune18": mask(order, uniform18, heldout, 0.1875, {"pruned_per_layer": 96}),
        "prune50-layer": mask(order, counts50, heldout, 0.5, {
            "pruned_per_layer": "per-layer", "pruned_total": int(counts50.sum()),
            "pruned_counts": {str(i): int(c) for i, c in enumerate(counts50)},
            "heldout_count_on_pruned_per_layer": {str(i): round(v, 6) for i, v in
                                                  enumerate(pruned_per_layer(order, counts50, heldout[1]))},
            "allocation": ("layer-aware: the smallest per-layer-normalized usage shares across all layers, in blocks "
                           "of 8, 128..384 per layer, layers 0-2 held to 128")}),
        "prune37-layer": mask(order, counts37, heldout, 0.375, {
            "pruned_per_layer": "per-layer", "pruned_total": int(counts37.sum()),
            "pruned_counts": {str(i): int(c) for i, c in enumerate(counts37)},
            "heldout_count_on_pruned_per_layer": {str(i): round(v, 6) for i, v in
                                                  enumerate(pruned_per_layer(order, counts37, heldout[1]))},
            "allocation": ("layer-aware: the smallest per-layer-normalized usage shares across all layers, in blocks "
                           "of 8, 96..288 per layer, layers 0-2 held to 96")}),
        "prune25-layer": mask(order, counts, heldout, 0.25, {
            "pruned_per_layer": "per-layer", "pruned_total": int(counts.sum()),
            "pruned_counts": {str(i): int(c) for i, c in enumerate(counts)},
            "allocation": ("layer-aware: the smallest per-layer-normalized usage shares across all layers, in blocks "
                           "of 8, 64..192 per layer, layers 0-2 held to 64")}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("usage", help="expert-usage.json of the v4 run")
    parser.add_argument("--output", default=str(ROOT / "data/expert_prune"))
    args = parser.parse_args()
    masks = build(json.loads(Path(args.usage).read_text()))
    for name, m in masks.items():
        path = Path(args.output) / f"{name}.json"
        path.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
        print(name, m.get("pruned_total", 48 * m["pruned_per_layer"] if isinstance(m["pruned_per_layer"], int) else 0),
              f"held-out routing on pruned: {100 * m['heldout_routing_on_pruned']['count_mean']:.3f}% mean, "
              f"{100 * m['heldout_routing_on_pruned']['count_max_layer']:.3f}% max layer")


if __name__ == "__main__":
    main()
