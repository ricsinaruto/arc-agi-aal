"""The no-foreground-wait PreToolUse hook: blocks foreground waits over 60 s for every agent, also one
whose cwd is an arm worktree (the hook reads only the tool input, never the cwd)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude/hooks/no_foreground_wait.py"
S = "sl" + "eep"  # spelled apart so the hook does not block commands that edit this file


def run(tool_input, cwd):
    import os

    d = {"tool_name": "Bash", "tool_input": tool_input, "cwd": str(cwd), "agent_id": "test"}
    return subprocess.run([sys.executable, str(HOOK)], input=json.dumps(d), capture_output=True, text=True, cwd=cwd,
                          env={**os.environ, "ARC_HOOK_LOG": str(cwd / "hook-wait.log")})


@pytest.mark.parametrize("where", ["", ".claude/worktrees/rl-async"])
@pytest.mark.parametrize("tool_input,blocked", [
    ({"command": "ls"}, False),
    ({"command": "pytest -q", "timeout": 60000}, False),
    ({"command": "pytest -q", "timeout": 120000}, True),
    ({"command": "pytest -q", "timeout": 600000, "run_in_background": True}, False),
    ({"command": f"{S} 30 && ls"}, False),
    ({"command": f"{S} 61"}, True),
    ({"command": f"{S} 2m; ls"}, True),
    ({"command": f"until [ -s f ]; do {S} 5; done"}, True),
    ({"command": f"for f in a b; do echo $f; done; {S} 5"}, False),
    ({"command": f"until [ -s f ]; do {S} 5; done", "run_in_background": True}, False),
    # quoted text and heredoc bodies are data, not commands (false positives on commit messages / agent text)
    ({"command": f"git commit -m 'waiter: {S} 120 then check' -- a.py"}, False),
    ({"command": f'echo "do x; {S} 5; done"'}, False),
    ({"command": f"cat > notes.md <<'EOF'\nuntil ok; do {S} 5; done\n{S} 300\nEOF\nls"}, False),
    ({"command": f'sh -c "{S} 120"'}, True),
    ({"command": f"bash <<EOF\n{S} 120\nEOF"}, True),
    ({"command": f"echo 'x' && {S} 90"}, True),
    ({"command": f'echo "$({S} 120)"'}, True),  # command substitution inside double quotes runs
    ({"command": f'x="`{S} 120`"'}, True),
    ({"command": f'bash -lc "{S} 120"'}, True),
    ({"command": f"sh -ec '{S} 120'"}, True),
    ({"command": f"cat <<EOF | bash\n{S} 120\nEOF"}, True),
    ({"command": f'eval "{S} 120"'}, True),
    ({"command": f'ssh vm "{S} 120"'}, True),  # the remote wait still blocks this call
    ({"command": f"cat > run.sh <<'EOF'\n{S} 300\nEOF"}, False),  # writing a script is not running it
])
def test_blocks_foreground_waits_from_any_cwd(tmp_path, where, tool_input, blocked):
    cwd = tmp_path / where
    cwd.mkdir(parents=True, exist_ok=True)
    r = run(tool_input, cwd)
    assert r.returncode == (2 if blocked else 0), r.stderr
    assert ("Blocked: foreground wait" in r.stderr) == blocked
    assert (cwd / "hook-wait.log").exists() == blocked  # each block is logged
