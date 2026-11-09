import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("put_knobs", Path(__file__).parents[1] / "launch/rl/put_knobs.py")
pk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pk)


def test_put_fails_loudly_on_missing_file_failed_put_and_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(pk, "MODAL", "/nonexistent/.venv/bin/modal")  # no records/modal in CI: started() must not raise
    assert pk.started("r") is False and pk.out_of("r") == "r"
    with pytest.raises(FileNotFoundError):  # v6h: the launcher pointed at a file that did not exist
        pk.put("r", tmp_path / "nope.json")
    k = tmp_path / "k.json"
    k.write_text(json.dumps({"admit_stagger_s": 3}))
    bad = lambda *a, **kw: subprocess.CompletedProcess(a, 1, "", "No such file")
    with pytest.raises(SystemExit, match="PUT FAILED"):
        pk.put("r", k, sh=bad, exists=lambda r: True, is_started=lambda r: False)
    ok = lambda *a, **kw: subprocess.CompletedProcess(a, 0, "", "")
    other = lambda path, dest: Path(dest).write_text(json.dumps({"admit_stagger_s": 12})) or True
    with pytest.raises(SystemExit, match="MISMATCH"):
        pk.put("r", k, sh=ok, get=other, exists=lambda r: True, is_started=lambda r: False)
    same = lambda path, dest: Path(dest).write_text(k.read_text()) or True
    assert pk.put("r", k, sh=ok, get=same, exists=lambda r: True, is_started=lambda r: False) == {"admit_stagger_s": 3}


def test_check_alerts_when_the_run_did_not_read_the_requested_knobs():
    want = {"admit_stagger_s": 3}
    start = {"kind": "v7_start", "epoch": 100.0}
    assert pk.check([], want, 120, 1000) == "pending"  # not started
    assert pk.check([start], want, 120, 150) == "pending"
    assert pk.check([start], want, 120, 300) == "run read nothing"
    assert pk.check([start, {"kind": "knobs_file", "keys": None, "missing": True}], want, 120, 300) == "run read no file"
    assert pk.check([start, {"kind": "knobs_file", "keys": {"admit_stagger_s": 12}}], want, 120, 300).startswith("run read {")
    assert pk.check([start, {"kind": "knobs_file", "keys": want}], want, 120, 300) is None


def test_verify_exits_nonzero_with_alert(tmp_path, capsys):
    k = tmp_path / "k.json"
    k.write_text(json.dumps({"a": 1}))
    feed = "\n".join(json.dumps(r) for r in [{"kind": "v7_start", "epoch": 0.0}, {"kind": "knobs_file", "keys": None, "missing": True}])
    get = lambda path, dest: Path(dest).write_text(feed) or True
    assert pk.verify("r", k, within=10, get=get, clock=lambda: 1000.0, sleep=lambda s: None) == 1
    assert "ALERT r [knobs]" in capsys.readouterr().out


def test_put_refuses_a_folder_the_run_does_not_have(tmp_path):
    """v6k 25 Oct: attempt 2 writes validation/<run>-attempt2/; a put to validation/<run>/ was never read."""
    k = tmp_path / "k.json"
    k.write_text(json.dumps({"a": 1}))
    never = lambda *a, **kw: pytest.fail("no put into a missing folder")
    with pytest.raises(SystemExit, match="TARGET MISSING"):
        pk.put("r", k, sh=never, exists=lambda r: False, is_started=lambda r: True)
    # v6l: launchers put before `modal run`; a reserved, not yet started attempt has no folder yet and is allowed
    ok = lambda *a, **kw: subprocess.CompletedProcess(a, 0, "", "")
    same = lambda path, dest: Path(dest).write_text(k.read_text()) or True
    assert pk.put("r", k, sh=ok, get=same, exists=lambda r: False, is_started=lambda r: False) == {"a": 1}
