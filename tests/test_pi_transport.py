import shutil
import subprocess
from pathlib import Path

import pytest


def test_pi_stream_transport_with_slow_reader():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node required for transport test")
    root = Path(__file__).resolve().parents[1]
    subprocess.run([node, str(root / "tests/pi_transport.test.mjs")], check=True,
                   cwd=root, capture_output=True, text=True, timeout=30)
