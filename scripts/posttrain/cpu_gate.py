"""CPU gate before a B300 hero launch, on the exact bundle: (1) admission sim
(arena.admission_sim) on the bundle's config and a KV trace: planned live tries and server-1 KV at 10 min;
(2) the start -> step -> kill/stop -> resume cycle of the fake-server loop tests, run against the bundle's own src.
The full CPU preflight legs (D/S/R) still run beside the GPU run.

    python scripts/posttrain/cpu_gate.py BUNDLE --live N --kv-max X [--trace data/kv_trace/hero-v6de.json]"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CYCLE = ["tests/test_rl_async_loop.py::test_resume_continues_without_training_a_group_twice",
         "tests/test_rl_async_loop.py::test_resume_after_a_lost_container_rebuilds_the_current_adapter",
         "tests/test_rl_async_loop.py::test_stop_request_ends_at_a_step_boundary_and_keeps_the_adapter",
         "tests/test_resume_mid_try.py", "tests/test_runops_restart.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--trace", type=Path, default=ROOT / "data/kv_trace/hero-v6de.json")
    ap.add_argument("--live", type=int, required=True)
    ap.add_argument("--kv-max", type=float, required=True)
    ap.add_argument("--pool-tokens", type=int, default=5_400_000)
    a = ap.parse_args()
    began = time.time()
    env = {**os.environ, "PYTHONPATH": f"{a.bundle.resolve()}/src:{ROOT}/tests"}
    sim = subprocess.run([sys.executable, str(ROOT / "src/arena/admission_sim.py"), str(a.bundle), "--trace", str(a.trace),
                          "--live", str(a.live), "--kv-max", str(a.kv_max), "--pool-tokens", str(a.pool_tokens)],
                         env=env, cwd=ROOT, capture_output=True, text=True, timeout=120)
    print(sim.stdout.strip().splitlines()[-1] if sim.stdout.strip() else sim.stderr[-500:])
    cyc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", *CYCLE], env=env, cwd=ROOT,
                         capture_output=True, text=True, timeout=240)
    print("cycle:", (cyc.stdout.strip().splitlines() or ["no output"])[-1])
    ok = sim.returncode == 0 and cyc.returncode == 0
    print(f"cpu gate {'PASS' if ok else 'FAIL'} in {time.time() - began:.0f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
