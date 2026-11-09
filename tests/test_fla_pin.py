"""fla_pin (MFU-1 gdnrace-b1): only chunk_fwd_kernel_o's BK=BV=64 config is kept (through jit wrappers), its cache
is cleared, a missing config raises, RL1_GDN_PIN_O=0 is a no-op, and the teacher/trainer entry points call it."""
import sys
import types
from pathlib import Path

import pytest

P = Path(__file__).resolve().parents[1] / "scripts" / "posttrain"
sys.path.insert(0, str(P))
import fla_pin  # noqa: E402


class Cfg:
    def __init__(self, bk, bv, w):
        self.kwargs = {"BK": bk, "BV": bv}
        self.num_warps = w


def fake(configs):
    at = types.SimpleNamespace(configs=configs, cache={("k",): configs[-1]})
    return types.SimpleNamespace(chunk_fwd_kernel_o=types.SimpleNamespace(fn=at)), at


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(fla_pin, "STATE", {"done": False, "config": None, "saved": None})


def test_keeps_only_bk_bv_64_and_clears_the_cache():
    mod, at = fake([Cfg(32, 32, 4), Cfg(64, 64, 4), Cfg(64, 64, 8), Cfg(128, 128, 8)])
    assert fla_pin.pin(mod) == {"BK": 64, "BV": 64}
    assert len(at.configs) == 1 and at.configs[0].kwargs == {"BK": 64, "BV": 64} and at.cache == {}


def test_missing_config_raises_and_off_is_a_noop(monkeypatch):
    mod, _ = fake([Cfg(128, 128, 8)])
    with pytest.raises(RuntimeError, match="BK=BV=64"):
        fla_pin.pin(mod)
    monkeypatch.setenv("RL1_GDN_PIN_O", "0")
    mod, at = fake([Cfg(64, 64, 4), Cfg(128, 128, 8)])
    assert fla_pin.pin(mod) is None and len(at.configs) == 2


def test_entry_points_pin():
    assert "fla_pin.pin()" in (P / "heal_kd.py").read_text() and "fla_pin.py" in (P / "heal_preflight.py").read_text()
    assert 'self.rec.event("fla_pinned", config=fla_pin.pin())' in (P / "rl1_trainer.py").read_text()


def test_unpin_restores_every_config():
    configs = [Cfg(64, 64, 4), Cfg(128, 128, 8)]
    mod, at = fake(list(configs))
    fla_pin.pin(mod)
    assert len(at.configs) == 1
    at.cache[("k",)] = at.configs[0]
    assert fla_pin.unpin() is None and at.configs == configs and at.cache == {} and not fla_pin.STATE["done"]


def test_trainer_diagnosis_knobs():
    src = (P / "rl1_trainer.py").read_text()
    assert 'if "chain_lock" in cmd:' in src and 'if "fla_pin" in cmd:' in src
    assert 'if cmd.get("verify_exit", True):' in src  # the default still exits on a failed verify
