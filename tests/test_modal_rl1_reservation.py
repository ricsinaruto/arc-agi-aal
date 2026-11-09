"""modal_rl1's launch gate needs the run record's open reservation (24 Oct: a GPU launch went ahead after a printed
records refusal), and attempt N > 1 of a record launches under its own receipt / volume id (<run>-attemptN)."""
import importlib.util
import json
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def modal_rl1():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("modal_rl1_res", ROOT / "scripts/posttrain/modal_rl1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cli(rc=0, out="", err=""):
    return lambda *a: types.SimpleNamespace(returncode=rc, stdout=out, stderr=err)


def test_refused_or_missing_reservation_stops_the_launch(monkeypatch):
    M = modal_rl1()
    monkeypatch.setattr(M, "_records_cli", cli(1, err="mfu-x: no reservation (records.py reserve first)"))
    with pytest.raises(SystemExit, match="launch gate: mfu-x: no reservation"):
        M.gate_reservation("mfu-x")
    monkeypatch.setattr(M, "_records_cli", cli(0, out="mfu-x: reserved, launch may go"))
    assert M.gate_reservation("mfu-x") == "reserved"


def test_attempt_n_gets_its_own_id(monkeypatch):
    M = modal_rl1()
    monkeypatch.setattr(M, "_records_cli", cli(0, out=json.dumps({"run": "r", "attempt": 3})))
    assert M.record_attempt("r") == 3
    monkeypatch.setattr(M, "_records_cli", cli(0, out="null"))
    assert M.record_attempt("r") == 1
    monkeypatch.setattr(M, "record_attempt", lambda run: 2)
    monkeypatch.setattr(M, "launch_env_errors", lambda env: [])
    monkeypatch.setattr(M, "launch_rl", lambda run_id, *a: run_id)  # stands in for the launch path
    named = []
    monkeypatch.setattr(M, "record_app", lambda run: named.append(run))  # the record gets the app id after a launch
    assert M.main("rl1-loop-x") == "rl1-loop-x-attempt2" and M.RECORD_ID == "rl1-loop-x" and named == ["rl1-loop-x"]
    seen = []
    monkeypatch.setattr(M, "_records_cli", lambda *a: seen.append(a) or types.SimpleNamespace(returncode=0, stdout="",
                                                                                              stderr=""))
    M.gate_reservation(M.RECORD_ID or "rl1-loop-x-attempt2")
    assert seen == [("require", "rl1-loop-x")]  # the gate asks about the record, not the attempt id
    M.RECORD_ID = None
    assert M.main("rl1-loop-y", preflight_run=True) == "rl1-loop-y"  # preflights keep their id
    assert named == ["rl1-loop-x"]  # and do not name the main run's app
