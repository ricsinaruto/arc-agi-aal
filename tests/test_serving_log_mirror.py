import time
from pathlib import Path
from types import SimpleNamespace


def test_serving_log_mirror_copies_both_servers_tails(tmp_path, monkeypatch):
    """v6h/v6i: server 2 died twice with no log on the volume; both logs' tails land in serving-logs/."""
    from arena import rl_single

    cls = next(v for v in vars(rl_single).values() if isinstance(v, type) and hasattr(v, "serving_log_mirror"))
    host = cls.__new__(cls)
    work, two = tmp_path / "work", tmp_path / "work" / "server2"
    two.mkdir(parents=True)
    (work / "vllm-openai-server.log").write_text("s1 boot\n" * 10)
    (two / "vllm.log").write_text("x" * 100 + "s2 died: CUDA error\n")
    host.server2 = SimpleNamespace(folder=two)
    monkeypatch.setenv("ARC_VALIDATION_OUTPUT", str(tmp_path / "out"))
    monkeypatch.setenv("TAAF_KAGGLE_WORKING_DIR", str(work))
    dest = host.serving_log_mirror(every_s=0.02, tail_bytes=30)
    try:
        for _ in range(200):
            if (dest / "server2.log").is_file() and (dest / "server1.log").is_file():
                break
            time.sleep(0.02)
        assert (dest / "server2.log").read_text().endswith("s2 died: CUDA error\n")
        assert len((dest / "server2.log").read_bytes()) == 30 and "s1 boot" in (dest / "server1.log").read_text()
        assert host.serving_log_mirror() is None  # one mirror per loop
    finally:
        host._log_mirror.stop.set()
