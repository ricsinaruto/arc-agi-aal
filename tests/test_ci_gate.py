"""ci.py gate (bundle builders): uncommitted source or a red/unpushed sha blocks the build; ARC_CI_SKIP
overrides it only with a worded reason, and every override is logged."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ops"))
import ci  # noqa: E402


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "src-checkout"
    (r / "src/pkg").mkdir(parents=True)
    (r / "src/pkg/a.py").write_text("x = 1\n")
    (r / "notes.txt").write_text("n\n")
    for c in (["init", "-q"], ["add", "-A"], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c"]):
        subprocess.run(["git", "-C", str(r), *c], check=True)
    monkeypatch.setattr(ci, "OUT", tmp_path / "ci")
    monkeypatch.setattr(ci, "OVERRIDES", tmp_path / "ci/overrides.jsonl")
    calls = []
    monkeypatch.setattr(ci, "require", lambda sha, dirs=(), cwd=None: calls.append(sha) or list(VERDICT))
    VERDICT.clear()
    return r, calls


VERDICT = []


def test_green_clean_checkout_goes(repo):
    r, calls = repo
    assert ci.gate("rl-hero", cwd=r, env={}) == []
    assert calls == [ci.git("rev-parse", "HEAD", cwd=r)]


def test_red_sha_blocks(repo):
    r, calls = repo
    VERDICT.append("abc is red: 1 failed")
    assert ci.gate("pth-heal", cwd=r, env={}) == ["abc is red: 1 failed"]


@pytest.mark.parametrize("change", ["modify", "untracked"])
def test_uncommitted_source_blocks_without_running_ci(repo, change):
    r, calls = repo
    (r / "src/pkg" / ("a.py" if change == "modify" else "b.py")).write_text("x = 2\n")
    probs = ci.gate("rl-hero", cwd=r, env={})
    assert len(probs) == 1 and "uncommitted" in probs[0] and "src/pkg/" in probs[0]
    assert calls == []


def test_non_code_changes_do_not_block(repo):
    r, _ = repo
    (r / "notes.txt").write_text("changed\n")
    assert ci.gate("rl-hero", cwd=r, env={}) == []


def test_override_needs_a_reason_and_is_logged(repo):
    r, calls = repo
    VERDICT.append("red")
    (r / "src/pkg/a.py").write_text("x = 3\n")
    assert "reason in words" in ci.gate("pth-heal", cwd=r, env={"ARC_CI_SKIP": "1"})[0]
    assert not ci.OVERRIDES.exists()
    t0 = time.time()
    assert ci.gate("pth-heal", cwd=r, env={"ARC_CI_SKIP": "flaky arena test, fix in flight"}) == []
    assert calls == []
    row = json.loads(ci.OVERRIDES.read_text())
    assert row["builder"] == "pth-heal" and row["reason"].startswith("")
    assert row["sha"] == ci.git("rev-parse", "HEAD", cwd=r) and row["cwd"] == str(r.resolve())
    assert [x["builder"] for x in ci.overrides(t0 - 1)] == ["pth-heal"]
    assert ci.overrides(time.time() + 1) == []


def test_cli_exits_1_when_blocked(repo, monkeypatch, capsys):
    r, _ = repo
    (r / "src/pkg/a.py").write_text("x = 4\n")
    monkeypatch.chdir(r)
    monkeypatch.delenv("ARC_CI_SKIP", raising=False)
    monkeypatch.setattr(sys, "argv", ["ci.py", "gate", "--builder", "rl-hero"])
    with pytest.raises(SystemExit) as e:
        ci.main()
    assert e.value.code == 1
    assert "rl-hero bundle build blocked" in capsys.readouterr().err
