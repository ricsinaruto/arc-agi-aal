"""mkbundle.py kl RUN JOBS_JSON SECONDS ASSETS | taskread RUN "ARGS" SECONDS ASSETS -> $MMRTX_SCRATCH/bundles/RUN (Nebius image kind).
entry.sh embeds the script and the job list (image_bundle copies only the entry)."""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = Path(".")
IMAGE = "vllm/vllm-openai@sha256:f5ec69891fb7b855762f40598eea1aa5cad6e4f2f0c2a2f81d270e2bdcd93632"
kind, run, jobs, seconds, assets = sys.argv[1:6]
assert kind in ("kl", "taskread")
if kind == "taskread":
    files = {n: (HERE / n).read_text() for n in ("taskread.py", "taskread_harness.py")}
    assert not any("PYEOF" in t for t in files.values()) and "'" not in jobs
    writes = "".join(f"cat > /tmp/mm/{n} <<'PYEOF'\n{t}\nPYEOF\n" for n, t in files.items())
    entry = f"""#!/bin/bash
set -euo pipefail
export MMRTX_ROOT=/w/mmrtx-models PYTHONUNBUFFERED=1
mkdir -p /tmp/mm
{writes}nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd /tmp/mm && python3 taskread.py {jobs}
"""
else:
  json.loads(jobs)
  src = (HERE / "kl_score.py").read_text()
  assert "PYEOF" not in src and "JEOF" not in jobs
  entry = f"""#!/bin/bash
set -euo pipefail
export MMRTX_ROOT=/w/mmrtx-models PYTHONUNBUFFERED=1
mkdir -p /tmp/mm
cat > /tmp/mm/kl_score.py <<'PYEOF'
{src}
PYEOF
cat > /tmp/mm/jobs.json <<'JEOF'
{jobs}
JEOF
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd /tmp/mm && python3 kl_score.py nebius "$(cat jobs.json)" mmrtx-ref-v2 sets-v2
"""
OUT = Path(__import__("os").environ.get("MMRTX_SCRATCH", "/tmp/mmrtx"))
OUT.mkdir(parents=True, exist_ok=True)
tmp = OUT / f"entry-{run}.sh"
tmp.write_text(entry)
dest = OUT / "bundles" / run
dest.parent.mkdir(exist_ok=True)
out = subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / ".claude/worktrees/nebius/scripts/nebius_validation.py"),
                      "image-bundle", "--dest", str(dest), "--script", str(tmp), "--image", IMAGE, "--seconds", seconds,
                      "--assets", assets], capture_output=True, text=True, cwd=REPO)
print(out.stdout[-2000:], out.stderr[-2000:])
