import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "qwen_bootstrap", Path(__file__).resolve().parents[1] / "scripts/qwen/bootstrap.py"
)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def test_installer_bypasses_ensurepip_and_host_packages(tmp_path, monkeypatch):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel = wheels / bootstrap.PIP_WHEEL
    wheel.write_bytes(b"test wheel")
    monkeypatch.setattr(bootstrap, "PIP_SHA256", hashlib.sha256(wheel.read_bytes()).hexdigest())
    calls = []
    monkeypatch.setattr(
        bootstrap.subprocess, "run", lambda argv, **kwargs: calls.append((argv, kwargs))
    )
    python = bootstrap.install_agent_environment(
        tmp_path, tmp_path / "venv", {"PYTHONPATH": "/host/code"}
    )
    assert "--without-pip" in calls[0][0]
    assert len(calls) == 3
    assert calls[1][0][0] == python == calls[2][0][0]
    assert calls[1][1]["env"]["PYTHONPATH"] == str(wheel)
    assert calls[1][1]["env"]["PIP_CONFIG_FILE"] == bootstrap.os.devnull
    assert "--require-hashes" in calls[1][0] and "--no-index" in calls[1][0]
    assert calls[2][0][-1] == "check"
    assert all(kwargs["check"] for _, kwargs in calls)


def test_bad_bootstrap_wheel_fails_before_any_process(tmp_path, monkeypatch):
    (tmp_path / "wheels").mkdir()
    (tmp_path / "wheels" / bootstrap.PIP_WHEEL).write_bytes(b"corrupt")
    monkeypatch.setattr(
        bootstrap.subprocess, "run", lambda *a, **k: pytest.fail("Must reject before executing")
    )
    with pytest.raises(ValueError, match="pip wheel mismatch"):
        bootstrap.install_agent_environment(tmp_path, tmp_path / "venv", {})


def test_collect_evidence_persists_only_evidence(tmp_path):
    working, output = tmp_path / "working", tmp_path / "persist"
    for name in [
        "agent-venv/lib/big.so",
        "node/bin/node",
        "pi/node_modules/x.js",
        "serving/runtime/big.whl",
        "serving/fork-bundle/serving_setup.py",
        "serving/cache-probe/result.json",
        "serving/native-action-preflight/result.json",
        "serving/native-action-preflight/evidence/evidence.sqlite",
        "serving/native-preflight.json",
        "serving/vllm-openai-server.log",
        "trials/C/lp85/evidence.sqlite",
        "trials/C/report.json",
        "trials/comparison.json",
        "setup.json",
    ]:
        path = working / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    bootstrap.collect_evidence(working, output)
    assert {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()} == {
        "serving/cache-probe/result.json",
        "serving/native-action-preflight/result.json",
        "serving/native-action-preflight/evidence/evidence.sqlite",
        "serving/native-preflight.json",
        "serving/vllm-openai-server.log",
        "trials/C/lp85/evidence.sqlite",
        "trials/C/report.json",
        "trials/comparison.json",
        "setup.json",
    }
    bootstrap.collect_evidence(tmp_path / "never", output)  # safe before the working tree exists
