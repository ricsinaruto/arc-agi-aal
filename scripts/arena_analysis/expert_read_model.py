"""Throughput arm (4 October, phase 3): expected distinct routed experts read per decode step, from the per-layer
usage quantiles of the 28 September usage run (docs/validation-20260927/expert-usage.json; no game-specific data).

Per layer the routing shares are interpolated from the count quantiles (experts covering 50 / 90 / 99 / 99.9% of
selections; equal shares within each segment), the least-used experts are pruned, each kept expert's per-token
inclusion is q = min(1, k x share / kept mass) and a step of T tokens reads it with probability 1 - (1 - q)^T.
Tokens are taken as independent (MTP verify rows of one stream are correlated, so absolute counts run ~10-20% below
the profile's 280-340; the ratios are what the estimate is for).

    uv run --extra dev python scripts/arena_analysis/expert_read_model.py
"""
import json
import pathlib

USAGE = pathlib.Path(__file__).resolve().parents[2] / "docs/validation-20260927/expert-usage.json"


def shares(layer, experts=512):
    used = experts - layer["unused"]
    points = [(0, 0.0), (layer["c0.5"], .5), (layer["c0.9"], .9), (layer["c0.99"], .99), (layer["c0.999"], .999),
              (used, 1.0)]
    out = []
    for (a, ca), (b, cb) in zip(points, points[1:]):
        out += [(cb - ca) / (b - a)] * (b - a) if b > a else []
    return out + [0.0] * (experts - len(out))


def read_probability(layer, keep, tokens, k=10):
    kept = shares(layer)[:keep]
    mass = sum(kept)
    return [1 - (1 - min(1.0, k * s / mass)) ** tokens for s in kept]


def main():
    layers = json.loads(USAGE.read_text())["per_layer"]
    report = {}
    for tokens in (48, 64, 80):
        row = {f"keep{keep}": sum(sum(read_probability(L, keep, tokens)) for L in layers) / len(layers)
               for keep in (512, 448, 416, 384)}
        row["keep448_top8"] = sum(sum(read_probability(L, 448, tokens, 8)) for L in layers) / len(layers)
        base = row["keep448"]
        row["vs_prune12"] = {name: round(1 - value / base, 4) for name, value in row.items() if name != "keep448"}
        report[f"T{tokens}"] = {name: (round(v, 1) if isinstance(v, float) else v) for name, v in row.items()}
    tail = {}
    for cut in (192, 256, 320):
        fractions = []
        for L in layers:
            p = read_probability(L, 448, 64)
            fractions.append(sum(p[cut:]) / sum(p))
        tail[f"ranked_ge_{cut}"] = round(sum(fractions) / len(fractions), 4)
    report["read_share_of_least_used_T64"] = tail
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
