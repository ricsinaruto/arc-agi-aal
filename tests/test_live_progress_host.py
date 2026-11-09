from arena import live_progress


def test_host_memory_reads_meminfo_cgroup_and_vllm_rss(tmp_path):
    proc, cgroup = tmp_path / "proc", tmp_path / "cg"
    proc.mkdir()
    cgroup.mkdir()
    (proc / "meminfo").write_text("MemTotal: 104857600 kB\nMemFree: 1 kB\nMemAvailable: 52428800 kB\n")
    (cgroup / "memory.peak").write_text(str(60 * 1024**3) + "\n")
    for pid, cmd, rss in (("10", b"python\0-m\0vllm.entrypoints.cli.main\0serve", 4 * 1024**2),
                          ("11", b"VLLM::EngineCore", 50 * 1024**2), ("12", b"python\0agent.py", 9 * 1024**2)):
        (proc / pid).mkdir()
        (proc / pid / "cmdline").write_bytes(cmd)
        (proc / pid / "status").write_text(f"Name:\tx\nVmRSS:\t{rss} kB\n")
    out = live_progress.host_memory(proc, cgroup)
    assert out == {"used_gib": 50.0, "available_gib": 50.0, "cgroup_peak_gib": 60.0, "vllm_rss_gib": 54.0,
                   "vllm_processes": 2}
    assert live_progress.host_memory(tmp_path / "missing", cgroup) is None


def _sample(t, queries, hits, external=0):
    return {"elapsed_seconds": t, "metrics": {"vllm:prefix_cache_queries_total": queries,
                                              "vllm:prefix_cache_hits_total": hits,
                                              "vllm:external_prefix_cache_hits_total": external}}


def test_prefix_hit_is_windowed_and_counts_both_tiers():
    # the first 100 s missed everything; the 5-min window sees only the later samples
    samples = [_sample(0, 0, 0), _sample(100, 1000, 0), _sample(200, 2000, 900), _sample(500, 5000, 3600, 200)]
    engine = live_progress.engine_snapshot(samples)
    assert engine["prefix_hit_recent"] == round((3600 + 200 - 900) / 3000, 4)
    assert engine["prefix_hit_mean"] == round((3600 + 200) / 5000, 4)
    line = live_progress.summary_line({"live": {"engine": engine}})
    assert line["prefix_hit"] == engine["prefix_hit_recent"] and line["prefix_hit_mean"] == engine["prefix_hit_mean"]


def test_prefix_hit_absent_without_queries():
    assert live_progress.engine_snapshot([_sample(0, 0, 0), _sample(10, 0, 0)])["prefix_hit_recent"] is None
    assert live_progress.summary_line({"live": {"engine": {}}})["prefix_hit"] is None
    # counters missing (an older server, a failed scrape): null, never an exception
    engine = live_progress.engine_snapshot([{"elapsed_seconds": 0, "metrics": {}}, {"elapsed_seconds": 9, "metrics": {}}])
    assert engine["prefix_hit_recent"] is None and engine["prefix_hit_mean"] is None


def test_prefix_hit_on_real_vllm_metrics(tmp_path):
    # the live v9b 24 h run's /metrics text (30 Oct), through the monitor's own parser
    from arc_infra.metrics import parse_metrics

    text = ("vllm:prefix_cache_queries_total{engine=\"0\",model_name=\"Qwen/Qwen3.8-Flash-Next-NVFP4\"} 1.16891691e+08\n"
            "vllm:prefix_cache_hits_total{engine=\"0\",model_name=\"Qwen/Qwen3.8-Flash-Next-NVFP4\"} 1.14022493e+08\n"
            "vllm:external_prefix_cache_hits_total{engine=\"0\",model_name=\"Qwen/Qwen3.8-Flash-Next-NVFP4\"} 0.0\n")
    metrics = parse_metrics(text)
    assert live_progress.prefix_hit({live_progress.PREFIX_QUERIES: 0.0}, metrics) == 0.9755
