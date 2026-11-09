"""Launch the eval recording (24 fresh re86 games) when the first of collections L2-L4 has ended."""
import os, subprocess, time
MV = "./.venv/bin/modal"; M = "~/arc-sft-data/mon"
runs = [f"sft-heldout-collect-re86-L{l}-v1" for l in (2, 3, 4)]
def ended(run):
    return subprocess.run([MV, "volume", "get", "--force", "arc-prolong-runs", f"/validation/{run}/modal-identity.json", f"{M}/{run}-id.json"],
                          capture_output=True).returncode == 0 and '"exit_code": null' not in open(f"{M}/{run}-id.json").read() and '"exit_code"' in open(f"{M}/{run}-id.json").read()
t0 = time.time()
while time.time() - t0 < 2 * 3600 and not any(ended(r) for r in runs):
    time.sleep(60)
time.sleep(120)
env = dict(os.environ, ARC_VALIDATION_ROOT=".", ARC_RL1_BUNDLE="dist/sft-evalrec-re86",
           ARC_RL1_AUTHORIZATION="re86 L1-L4 single-game SFT plan approved; eval recording")
r = subprocess.run([MV, "run", "--detach", "scripts/posttrain/modal_rl1.py", "--run-id", "sft-heldout-evalrec-re86-v1"],
                   cwd="./.claude/worktrees/sft-heal", env=env, capture_output=True, text=True)
call = [w for w in r.stdout.split('"') if w.startswith("fc-")]
print(f"LAUNCHED sft-heldout-evalrec-re86-v1 {call[:1]} rc={r.returncode} {r.stderr[-300:] if r.returncode else ''} at {time.strftime('%H:%M', time.gmtime())}", flush=True)
