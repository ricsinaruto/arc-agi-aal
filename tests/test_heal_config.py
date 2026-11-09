"""launch/pth/heal_config.py (typed-config pilot for PTH's OPD rounds): the c8 config reproduces its documented launch
line, and the PTH checks catch naming, start-point, auth, env-pair, launcher-drift and incomplete-model problems."""
import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("heal_config", ROOT / "launch/pth/heal_config.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
C8 = ROOT / "launch/pth/configs/sft-heal-round-opdS-c8.json"
# sft.md Now (25 Oct), the c8 launch line; ACK / AUTH_SET / WHAT_SET are filled at launch
DOC = ("DSTART=1 NAME_SET=opdS-c8 HBUNDLE_SET=dist/sft-heal-round-v97 PB_SET=sft4x-opdS-g1-v27 "
       "PLAY_SET=b2-sft4x-opdS-g1-v33,b2-sft4x-opdS-g1-v8,b2-sft4x-opdS-g1-v9,b2-sft4x-opdS-g1-v10,b2-sft4x-opdS-g1-v14 "
       "HLR_SET=3.5e-6 MAX_AGE_SET=12 VKL=0.052 XE_SET=HEAL_FORCE_IDS=248046+248069+248058,HEAL_THINK_EOS_MASK=248046+248069 "
       "XT_SET=HEAL_FORCE_IDS=248046+248069+248058,HEAL_EVICT_CLOSING_FIRST=1 REUSE_PF=1")


def c8(**kw):
    return dataclasses.replace(H.launchcfg.load(H.HealConfig, C8), **kw)


def test_c8_env_is_the_documented_launch_line():
    e = H.env(c8())
    assert {k: v for k, v in e.items() if k not in ("ACK", "AUTH_SET", "WHAT_SET")} == \
        dict(x.split("=", 1) for x in DOC.split())


def test_strict_load_refuses_unknown_and_missing_keys(tmp_path):
    d = json.loads(C8.read_text())
    for bad in ({**d, "hlr_typo": 1}, {k: v for k, v in d.items() if k != "vkl"}):
        (tmp_path / "c.json").write_text(json.dumps(bad))
        with pytest.raises(H.launchcfg.ConfigError):
            H.launchcfg.load(H.HealConfig, tmp_path / "c.json")


@pytest.fixture
def trees(tmp_path):
    heal, prune = tmp_path / "sft-heal", tmp_path / "sft-prune"
    (heal / "dist/sft-heal-round-v97/rl1").mkdir(parents=True)
    (heal / "launch/pth").mkdir(parents=True)
    (heal / "launch/pth/launch_c7k.sh").write_text("x HCARDS=4 HTRANKS=3\n")
    (heal / "launch/pth/opd_stream_c.sh").write_text("TO=${TO:-25200} DUR=1\n")
    (prune / "dist/sft4x-opdS-g1-v27/configs").mkdir(parents=True)
    (prune / "dist/sft4x-opdS-g1-v27/configs/comparison.json").write_text(json.dumps({"games": ["ft09-0d8b", "ls20-96"]}))
    return heal, prune


def run_checks(cfg, trees, names=(), index=None):
    return H.checks(cfg, *trees, ls=lambda spec: list(names), index=lambda spec: index)


def test_filled_c8_passes_and_each_fault_is_named(trees):
    ok = c8(authorization="", ack="restart from D after c7r stop")
    assert run_checks(ok, trees) == []
    faults = {
        "run sft-heal-round-x != sft-heal-round-opdS-c8": dict(run="sft-heal-round-x"),
        "exactly one start point": dict(ck=6),
        "authorization: not filled in": dict(authorization="PENDING owner GO"),
        "env pair 'HEAL_A=1,HEAL_B=2'": dict(xe=("HEAL_A=1,HEAL_B=2",)),
        "gpu_count 3 but launch_c7k.sh sets HCARDS=4": dict(gpu_count=3),
        "max_seconds 9000 but opd_stream_c.sh's crash ceiling TO is 25200": dict(max_seconds=9000),
        "trainer bundle": dict(hbundle="dist/none"),
    }
    for want, change in faults.items():
        probs = run_checks(dataclasses.replace(ok, **change), trees)
        assert any(want in p for p in probs), (want, probs)


def test_model_copies_must_be_complete(trees):
    ok = c8(authorization="a", ack="b", data=("arc-prolong-assets:m",))
    assert run_checks(ok, trees, names=["model.safetensors.index.json", "a.safetensors"], index={"a.safetensors"}) == []
    assert "1 of 2 indexed shards missing" in run_checks(ok, trees, ["model.safetensors.index.json", "a.safetensors"],
                                                         {"a.safetensors", "b.safetensors"})[0]
    part = ["model-00001-of-00131.safetensors", "model-00002-of-00131.safetensors"]
    assert "2 of 131 model shards and no index" in run_checks(ok, trees, part)[0]


def test_bundle_roster_drops_the_hash(trees, monkeypatch):
    monkeypatch.setattr(H, "PRUNE", trees[1])
    assert H.bundle_roster(c8()) == ["ft09", "ls20"]
