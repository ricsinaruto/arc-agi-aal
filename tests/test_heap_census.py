import json
import os
import subprocess
import venv

import pytest

from arc_infra import heap_census

BUSY = """
import copy, time
from collections import OrderedDict
x = {"a": (1, 2, (3, 4)), "b": [OrderedDict(k=(i, str(i))) for i in range(200)], "c": {i: (i, i) for i in range(500)}}
keep = []
end = time.time() + {secs}
n = 0
while time.time() < end:
    y = copy.deepcopy(x); n += 1
    t = tuple(i for i in range(50))
    keep.append(bytearray(4096))
print("survived", n)
"""


def venv_python(tmp_path):
    env_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(env_dir)
    python = env_dir / "bin" / "python"
    assert (heap_census.install(python) / "arc_heap_census.pth").is_file()
    return python


def test_census_rows_from_a_venv_process(tmp_path):
    """install() puts the module + .pth in the venv; a process there with HEAP_CENSUS_DIR writes rows (with
    HEAP_CENSUS_TRACE the top allocation sites, naming the retaining line), one without it writes none."""
    python, out = venv_python(tmp_path), tmp_path / "heap"
    env = {**os.environ, "HEAP_CENSUS_DIR": str(out), "HEAP_CENSUS_EVERY_S": "0.5", "HEAP_CENSUS_FIRST_S": "0.5",
           "HEAP_CENSUS_TRACE": "3", "HEAP_CENSUS_TRIM": "1"}
    subprocess.run([str(python), "-c", BUSY.replace("{secs}", "2.5")], env=env, check=True, timeout=60)
    rows = [json.loads(l) for f in out.glob("heap-*.jsonl") for l in f.read_text().splitlines()]
    assert rows and all(r["traced_gib"] >= 0 and r["lines"] and r["stacks"] for r in rows)
    assert any("<string>" in site for r in rows for _, _, site in r["lines"])  # the keep.append line
    assert all("rss_after_trim_gib" in r or "trim_error" in r for r in rows)  # glibc: trimmed; elsewhere: noted
    quiet = tmp_path / "quiet"
    subprocess.run([str(python), "-c", "pass"], env={**os.environ, "HEAP_CENSUS_DIR": ""}, check=True, timeout=60)
    assert not quiet.exists() and heap_census.install_for(["x"], {}) is None


def test_census_never_breaks_a_busy_main_thread(tmp_path):
    """Heap legs A/B 27 Oct: the gc-walk census crashed vLLM's API server at start-up (SystemError
    tupleobject.c:909 in a deepcopy). The census thread at 50 passes/s beside deepcopy and tuple building: no error."""
    python = venv_python(tmp_path)
    env = {**os.environ, "HEAP_CENSUS_DIR": str(tmp_path / "heap"), "HEAP_CENSUS_EVERY_S": "0.02",
           "HEAP_CENSUS_FIRST_S": "0", "HEAP_CENSUS_TRACE": "3"}
    r = subprocess.run([str(python), "-c", BUSY.replace("{secs}", "8")], env=env, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0 and "survived" in r.stdout, r.stderr[-1500:]


def test_census_trace_after_a_delay(tmp_path):
    """HEAP_CENSUS_TRACE_AFTER_S: no tracing at start-up (boot at full speed); the first pass starts it, later rows
    carry the allocation sites."""
    python, out = venv_python(tmp_path), tmp_path / "heap"
    env = {**os.environ, "HEAP_CENSUS_DIR": str(out), "HEAP_CENSUS_EVERY_S": "0.5", "HEAP_CENSUS_TRACE": "3",
           "HEAP_CENSUS_TRACE_AFTER_S": "0.5"}
    code = "import tracemalloc, time; assert not tracemalloc.is_tracing(); " + BUSY.replace("{secs}", "3")
    subprocess.run([str(python), "-c", code], env=env, check=True, timeout=60)
    rows = [json.loads(line) for f in out.glob("heap-*.jsonl") for line in f.read_text().splitlines()]
    assert rows[0].get("trace_started") and any(r.get("lines") for r in rows[1:])


@pytest.mark.skipif(not os.path.exists("/proc/self/cmdline"), reason="needs /proc (Linux)")
def test_census_traces_only_matching_processes(tmp_path):
    """HEAP_CENSUS_TRACE_MATCH (heap legs D 28 Oct: trace the vLLM worker only; a traced API server stalled): a
    process whose cmdline lacks the string writes RSS rows untraced; one with it is traced and reports tracing's own
    memory. Every row names its process (cmd)."""
    python = venv_python(tmp_path)
    for tag, match in (("no", "VLLM::Worker"), ("yes", "survived")):
        out = tmp_path / tag
        env = {**os.environ, "HEAP_CENSUS_DIR": str(out), "HEAP_CENSUS_EVERY_S": "0.5", "HEAP_CENSUS_TRACE": "1",
               "HEAP_CENSUS_TRACE_AFTER_S": "0.5", "HEAP_CENSUS_TRACE_MATCH": match}
        # 4 s: >= 2 traced rows after the 0.5 s delay on a loaded CI box (CI 28 Oct: 2.5 s left one)
        subprocess.run([str(python), "-c", BUSY.replace("{secs}", "4")], env=env, check=True, timeout=60)
        rows = [json.loads(line) for f in out.glob("heap-*.jsonl") for line in f.read_text().splitlines()]
        assert rows and all(r["cmd"] for r in rows)  # the process title (truncated to 80 chars)
        traced = [r for r in rows if r.get("trace_started") or r.get("lines")]
        if tag == "no":
            assert not traced
        else:
            assert rows[0].get("trace_started") and any(r.get("trace_own_gib", -1) >= 0 for r in rows[1:])
