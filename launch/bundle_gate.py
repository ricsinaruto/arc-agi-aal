"""CI gate for every launch/rl and launch/pth bundle builder (a red CI blocks the build). Runs the main
checkout's `tools/ops/ci.py gate` in the SOURCE checkout: uncommitted src/ scripts/ files, an unpushed sha or a red
sha block the build (no result yet: the suite runs on Modal CPU, ~5 min). ARC_CI_SKIP="<reason>" overrides it; the
override is logged (runs/ops/ci/overrides.jsonl) and shows on the dashboard."""
import os
import subprocess
import sys

ROOT = os.environ.get("ARC_MAIN_CHECKOUT", ".")


def require(worktree, builder, ref="HEAD"):
    """ref: the commit the bundle's code comes from (launch/rl/build.py builds from a config's source commit)."""
    r = subprocess.run([f"{ROOT}/.venv/bin/python", f"{ROOT}/tools/ops/ci.py", "gate", "--builder", builder, ref],
                       cwd=str(worktree))
    if r.returncode:
        sys.exit(f"CI GATE: {builder} build from {worktree} blocked (see above)")
