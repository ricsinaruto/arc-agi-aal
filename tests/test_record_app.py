"""25 Oct: PTH's sft-stream-check-eosA-v1 went live with no app_id on its record (gpu_reconcile: unrecorded B300)."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("modal")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("modal_rl1_rec", ROOT / "scripts/posttrain/modal_rl1.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_record_app_writes_the_app_id_through_records():
    calls = []
    ok = lambda *a: calls.append(a) or subprocess.CompletedProcess(a, 0, "", "")
    assert M.record_app("sft-stream-check-x", app_id="ap-abc", cli=ok) == "ap-abc"
    assert calls == [("ledger", "sft-stream-check-x", "--set", 'app_id="ap-abc"')]


def test_record_app_fails_loudly():
    bad = lambda *a: subprocess.CompletedProcess(a, 1, "", "GateError: sft-x: no ledger row (records.py reserve first)")
    with pytest.raises(SystemExit, match="APP_ID NOT RECORDED sft-x .*no ledger row"):
        M.record_app("sft-x", app_id="ap-abc", cli=bad)
    with pytest.raises(SystemExit, match="APP_ID NOT RECORDED"):
        M.record_app("sft-x", app_id=None, cli=lambda *a: pytest.fail("no write without an app id"))
