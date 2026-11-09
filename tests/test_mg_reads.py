import math
import random

from arena import mg_reads, rl_loop


def arm(rate, tokens, rhae, n, rng):
    return [(rhae, tokens, True) if rng.random() < rate else (0.0, tokens * 2, False) for _ in range(n)]


def test_cheap_level_no_longer_dominates_the_pooled_read():
    rng = random.Random(1)
    meta = {"g1-L1": {"game": "g1", "level": 1, "source": "a"}, "g1-L2": {"game": "g1", "level": 2, "source": "a"},
            "g2-L1": {"game": "g2", "level": 1, "source": "b"}, "g3-L1": {"game": "g3", "level": 1, "source": "b"}}
    base = {"g1-L1": arm(0.6, 30000, 0.6, 16, rng), "g1-L2": arm(1.0, 300, 0.4, 16, rng),
            "g2-L1": arm(0.6, 30000, 0.6, 16, rng), "g3-L1": arm(0.6, 30000, 0.6, 16, rng)}
    # the policy lifts the cheap level's RHAE 2.5x and spends 25% more tokens on every expensive level: the pooled
    # RHAE/10K rises (sc25-style), the per-game read falls
    cur = {"g1-L1": [(r, int(1.25 * t), c) for r, t, c in base["g1-L1"]],
           "g1-L2": [(1.0 if c else 0.0, t, c) for r, t, c in base["g1-L2"]],
           "g2-L1": [(r, int(1.25 * t), c) for r, t, c in base["g2-L1"]],
           "g3-L1": [(r, int(1.25 * t), c) for r, t, c in base["g3-L1"]]}
    pooled = lambda d: sum(r for v in d.values() for r, _, _ in v) / sum(t for v in d.values() for _, t, _ in v)
    assert pooled(cur) > 1.05 * pooled(base)
    out = mg_reads.norm_read(cur, base, meta, n_boot=200)
    assert out["all"]["ratio"] < 1.0 and out["all"]["games"] == 3
    assert out["level:L2"]["ratio"] > 2.0 and out["level:L1"]["ratio"] < 0.85
    assert out["source:b"]["games"] == 2 and out["all"]["ci"][0] <= out["all"]["ratio"] <= out["all"]["ci"][1]


def test_identical_arms_read_one_and_thin_sets_shrink():
    rng = random.Random(2)
    meta = {f"g{i}-L1": {"game": f"g{i}", "level": 1} for i in range(6)}
    base = {n: arm(0.5, 20000, 0.8, 12, rng) for n in meta}
    out = mg_reads.norm_read(base, base, meta, n_boot=200)
    assert out["all"]["ratio"] == 1.0 and out["all"]["clear_delta"] == 0.0
    assert out["all"]["ci"][0] < 1.0 < out["all"]["ci"][1]
    none = {n: [(0.0, 40000, False)] * 2 for n in meta}  # no clears in either arm: finite, near 1
    got = mg_reads.norm_read(none, none, meta, n_boot=50)
    assert math.isfinite(got["all"]["ratio"])


def test_compact_roundtrip_and_key_check():
    ts = [(0.75, 1200, True), (0.0, 5000, False)]
    assert mg_reads.expand(mg_reads.compact(ts)) == ts
    rl_loop.check_mg({"read_norm": True})
    rl_loop.check_mg({"read_norm": {"lam": 1.0, "n_boot": 100}})
    try:
        rl_loop.check_mg({"read_norm": {"lamb": 1.0}})
        raise AssertionError("unknown read_norm key accepted")
    except ValueError:
        pass
