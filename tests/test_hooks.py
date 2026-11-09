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
])
def test_blocks_foreground_waits_from_any_cwd(tmp_path, where, tool_input, blocked):
    cwd = tmp_path / where
    cwd.mkdir(parents=True, exist_ok=True)
    r = run(tool_input, cwd)
    assert r.returncode == (2 if blocked else 0), r.stderr
    assert ("Blocked: foreground wait" in r.stderr) == blocked
    assert (cwd / "hook-wait.log").exists() == blocked  # each block is logged
