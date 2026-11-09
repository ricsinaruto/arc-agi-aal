"""CI gate for PTH bundle builds (a red CI blocks the build): the source checkout's HEAD must be
CI-green (tools/ops/ci.py require; runs the suite on Modal CPU when no result exists, ~5 min). ARC_CI_SKIP="<reason>"
skips it with the reason printed (owner-approved emergencies only)."""
import os
import subprocess
import sys

ROOT = "."


def require(worktree):
    if os.environ.get("ARC_CI_SKIP"):
        return print(f"CI GATE SKIPPED for {worktree}: {os.environ['ARC_CI_SKIP']}", file=sys.stderr)
    r = subprocess.run([f"{ROOT}/.venv/bin/python", f"{ROOT}/tools/ops/ci.py", "require", "HEAD"], cwd=str(worktree))
    if r.returncode:
        sys.exit(f"CI RED for {worktree} HEAD: bundle build blocked (fix the tests or set ARC_CI_SKIP=\"<reason>\")")
