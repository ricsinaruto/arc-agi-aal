"""The runops sampler on N cards (H2 26 Oct: vLLM metrics came from :1234 only)."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/posttrain"))

METRICS = """# HELP vllm:num_requests_running x
vllm:num_requests_running{{engine="0",model_name="m"}} {running}
vllm:num_requests_waiting{{engine="0",model_name="m"}} {waiting}
vllm:kv_cache_usage_perc{{engine="0",model_name="m"}} {kv}
vllm:generation_tokens_total{{engine="0",model_name="m"}} {gen}
vllm:num_preemptions_total{{engine="0",model_name="m"}} 0.0
vllm:prompt_tokens_total{{engine="0",model_name="m"}} 99.0
"""


def serve(text):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = text.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_live_sample_reports_every_server_and_the_totals(tmp_path, monkeypatch):
    m = pytest.importorskip("modal_rl1")
    one = serve(METRICS.format(running=5, waiting=1, kv=0.02, gen=1017))
    two = serve(METRICS.format(running=118, waiting=0, kv=0.30, gen=90000))
    monkeypatch.setenv("ARC_REAL_NVIDIA_SMI", "/nonexistent-smi")
    try:
        ports = [one.server_address[1], two.server_address[1], 1]  # the third: no server (asleep, not up yet)
        m.live_sample(tmp_path / "trial", tmp_path / "live", 0.0, ports=ports)
    finally:
        one.shutdown()
        two.shutdown()
    row = json.loads((tmp_path / "live/sampler.jsonl").read_text().splitlines()[-1])
    per = row["vllm_servers"]
    assert set(per) == {str(ports[0]), str(ports[1])}
    assert per[str(ports[1])]["num_requests_running"] == 118 and per[str(ports[0])]["generation_tokens_total"] == 1017
    assert row["vllm"]["num_requests_running"] == 123 and row["vllm"]["generation_tokens_total"] == 91017
    assert row["vllm"]["kv_cache_usage_perc"] == pytest.approx(0.16)  # the mean, not a sum past 1
    assert "prompt_tokens_total" not in row["vllm"] and {"epoch", "sample_s", "commit_s"} <= set(row)


def test_server_ports_follow_second_server():
    m = pytest.importorskip("modal_rl1")
    assert m.server_ports(1) == [1234]
    assert m.server_ports(2) == [1234, 1236] and m.server_ports(4) == [1234, 1236, 1237, 1238]
