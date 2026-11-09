"""mg.engine_admission: new tries pause on the engine's own KV use, recent preemptions and prefix-hit drops."""
from arena import level_bench
from arena.level_bench import EngineAdmission, SharedSlots, engine_sample

PAGE = """# HELP vllm:kv_cache_usage_perc x
vllm:kv_cache_usage_perc{engine="0",model_name="m"} %s
vllm:num_preemptions_total{engine="0",model_name="m"} %s
vllm:prefix_cache_queries_total{engine="0",model_name="m"} %s
vllm:prefix_cache_hits_total{engine="0",model_name="m"} %s
"""


def s(kv, pre=0, q=0, h=0):
    return {"kv": kv, "preempt": pre, "queries": q, "hits": h}


def test_engine_sample_parses_and_sums_label_sets():
    text = PAGE % (0.5, 3, 1000, 900) + 'vllm:num_preemptions_total{engine="1"} 2\n'
    assert engine_sample(text) == {"kv": 0.5, "preempt": 5.0, "queries": 1000.0, "hits": 900.0}
    assert engine_sample("vllm:num_requests_running 3\n") is None  # no KV gauge: no signal


def test_kv_and_preemption_pause_and_clear():
    rule = EngineAdmission(kv_max=0.85, preempt_s=60)
    assert rule.update(0, s(0.5)) is None
    assert rule.update(5, s(0.9)) == "kv"
    assert rule.update(10, s(0.6, pre=1)) == "preempt"  # a new preemption
    assert rule.update(65, s(0.6, pre=1)) == "preempt"  # within 60 s of it
    assert rule.update(75, s(0.6, pre=1)) is None
    assert rule.update(80, None) is None  # no signal never pauses


def test_prefix_hit_drop_pauses_against_the_running_median():
    rule = EngineAdmission(hit_s=60, hit_drop=0.05)
    q = h = 0
    for t in range(0, 600, 10):  # a steady 94% hit rate
        q, h = q + 1000, h + 940
        assert rule.update(t, s(0.5, q=q, h=h)) is None
    for t in range(600, 700, 10):  # 80%: below the median minus 5 points once the 60-s window is mostly new
        q, h = q + 1000, h + 800
        why = rule.update(t, s(0.5, q=q, h=h))
    assert why == "hit"


def test_shared_slots_pause_never_deadlocks_an_idle_pool():
    pool = SharedSlots(4)
    pool.want("b", 0, 3)
    pool.paused_why = "kv"
    assert pool.take("b")  # nothing running: one try starts anyway
    assert not pool.take("b") and pool.engine_waits == 1
    pool.paused_why = None
    assert pool.take("b")


def test_key_is_known():
    from arena import rl_loop
    assert "engine_admission" in rl_loop.MG_KEYS
    assert level_bench.METRIC_KEYS[0] == "vllm:kv_cache_usage_perc"


def test_hit_rule_waits_for_kv():
    """23 Oct: a hit-rate drop at low KV is new tries' fresh prompts, not eviction: no pause under hit_kv."""
    rule = EngineAdmission(hit_s=60, hit_drop=0.05)
    q = h = 0
    for t in range(0, 600, 10):
        q, h = q + 1000, h + 940
        rule.update(t, s(0.2, q=q, h=h))
    for t in range(600, 700, 10):
        q, h = q + 1000, h + 800
        why = rule.update(t, s(0.2, q=q, h=h))
    assert why is None
