from arena import admission_sim


def study(**mg):
    base = {"active_tries": 80, "admit_stagger_s": 50, "active_kv_fill": 0.65, "kv_project": {"group_prefix": True},
            "timeshare": {"active_tries": 128}, "v7": {"in_flight": 400, "in_flight_mode": "unfinished", "ramp_s": 30,
                                                       "max_ready": 8}}
    base.update(mg)
    return {"rl_loop": {"mg": base}, "server": {"max_model_len": 131072}}


TRACE = {"tries": [{"try": f"a{i}", "group": f"g{i % 5}", "reqs": [[6000 + 2000 * k, 800, 0] for k in range(15)]}
                   for i in range(40)]}


def test_gate_passes_the_hero_admission_and_catches_a_starved_one():
    """The v6f settings fill server 1 by 10 min; a group cap that starves admission (v6d: planned counting at
    in_flight 110 held ~46 live) or a stagger that serialises a bench fails the same gate."""
    rows = admission_sim.simulate(study(), TRACE, minutes=12, pool_tokens=5_400_000)
    ok, msg = admission_sim.check(rows, 60, 0.75)
    assert ok, msg
    starved = admission_sim.simulate(study(v7={"in_flight": 24, "in_flight_mode": "planned", "ramp_s": 30,
                                               "max_ready": 8}), TRACE, minutes=12, pool_tokens=5_400_000)
    assert not admission_sim.check(starved, 60, 0.75)[0]


READS = {"tries": [{"try": f"r{i}", "group": f"r{i}", "reqs": [[20000 + 3000 * k, 1500, 0] for k in range(40)]}
                   for i in range(10)]}


def test_read_heavy_mix_still_admits_training_groups():
    """v6h 24 Oct: 12 long held-out read benches outranked every group (oldest_first) and filled the projected-KV
    gate; no training group started for 15 min. With v7.read_kv_share the reads hold read_share of the gate."""
    v7 = {"in_flight": 150, "in_flight_mode": "unfinished", "ramp_s": 30, "max_ready": 8, "read_share": 0.25}
    kw = dict(minutes=12, pool_tokens=5_400_000, reads=12, read_tries=4, read_trace=READS)
    old = admission_sim.simulate(study(admit_stagger_s=3, active_kv_fill=0.75, timeshare={}, v7=v7), TRACE, **kw)
    new = admission_sim.simulate(study(admit_stagger_s=3, active_kv_fill=0.75, timeshare={},
                                       v7={**v7, "read_kv_share": True}), TRACE, **kw)
    at = lambda rows, m: next(r for r in rows if r["minute"] == m)
    assert at(old, 10)["group_live"] <= 3, at(old, 10)  # the v6h starvation (projected 0.74, KV 0.37)
    assert at(new, 10)["group_live"] >= 24 and at(new, 10)["read_live"] > 0, at(new, 10)
    assert at(new, 10)["read_live"] < at(old, 10)["read_live"]
