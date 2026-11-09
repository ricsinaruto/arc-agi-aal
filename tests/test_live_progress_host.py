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
