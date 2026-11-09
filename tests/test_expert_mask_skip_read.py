"""ARC_EXPERT_MASK_SKIP_READ (mfu2-maskskip 6d66efa43, ported to the redesign 26 Oct): masked experts' packed weights
are never read; scales, the MTP head's experts and every other tensor are."""
import json
import types

import pytest

from arc_infra import expert_mask_patch as P

MASK = {"experts": 8, "pruned": {"0": [1, 2], "3": [7]}}


def test_settings_carry_skip_read_only_when_on():
    off = json.loads(P.settings_of(MASK, "m.json"))
    on = json.loads(P.settings_of(MASK, "m.json", skip_read=True))
    assert "skip_read" not in off and on["skip_read"] is True
    assert P.SKIP_TAIL.strip().endswith("_arc_install_skip_read()")


def test_skip_read_wraps_should_skip_weight(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from arc_infra import expert_mask_runtime as R

    settings = tmp_path / "s.json"
    settings.write_bytes(P.settings_of(MASK, "m.json", skip_read=True))
    monkeypatch.setattr(R, "SETTINGS", str(settings))
    monkeypatch.setattr(R, "SKIP", {"installed": False, "skipped": 0})
    wu = types.SimpleNamespace(should_skip_weight=lambda name, ids: name.endswith(".bias"))
    assert R.install_skip_read(wu) and not R.install_skip_read(wu)  # once
    pre = "model.language_model.layers"
    skip = wu.should_skip_weight
    assert skip(f"{pre}.0.mlp.experts.1.gate_proj.weight", None)
    assert skip(f"{pre}.3.mlp.experts.7.down_proj.weight", None)
    assert not skip(f"{pre}.0.mlp.experts.1.gate_proj.weight_scale", None)  # scales are read
    assert not skip(f"{pre}.0.mlp.experts.3.up_proj.weight", None)  # kept expert
    assert not skip("model.language_model.mtp.layers.0.mlp.experts.1.gate_proj.weight", None)  # the draft's
    assert skip(f"{pre}.0.x.bias", None)  # the original rule still applies
    assert R.SKIP["skipped"] == 2 and R.SKIP["expected"] == 9


def test_skip_read_off_installs_nothing(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from arc_infra import expert_mask_runtime as R

    settings = tmp_path / "s.json"
    settings.write_bytes(P.settings_of(MASK, "m.json"))
    monkeypatch.setattr(R, "SETTINGS", str(settings))
    monkeypatch.setattr(R, "SKIP", {"installed": False, "skipped": 0})
    wu = types.SimpleNamespace(should_skip_weight=lambda name, ids: False)
    assert not R.install_skip_read(wu)
