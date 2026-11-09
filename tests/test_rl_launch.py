"""launch/rl: one typed config per run, one builder from a commit, one launcher. The checked-in configs
load strictly and pass their checks; the builder's hashing, $file inlining and manifest; the launcher refuses a bundle
not built from the config and composes the modal_rl1 env from the config alone."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "launch/rl"))
import build as B  # noqa: E402
import config as C  # noqa: E402
import launch as L  # noqa: E402

CONFIGS = sorted((ROOT / "launch/rl/runs").glob("*.json"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_checked_in_configs_load_and_pass_checks(path):
    cfg = C.load(path)
    assert cfg.run == path.stem and C.checks(cfg) == []
    assert len(cfg.source) == 40 and cfg.files and cfg.seeds
    for s in cfg.seeds:  # in-repo seeds match their pins (the large stores live outside the repo)
        p = B.local(s.path)
        if p.is_relative_to(ROOT):
            assert B.tree_sha256(p) == s.sha256, s.path
    sets = cfg.study["rl_loop"]["mg"]["sets"]
    assert set(sets) == {"$file", "sha256"} and B.sha256_file(B.local(sets["$file"])) == sets["sha256"]


def test_strict_config_refuses_unknown_and_missing_keys(tmp_path):
    d = json.loads(CONFIGS[0].read_text())
    for bad in ({**d, "typo": 1}, {k: v for k, v in d.items() if k != "source"},
                {**d, "launch": {**d["launch"], "extra": 1}}):
        (tmp_path / "c.json").write_text(json.dumps(bad))
        with pytest.raises(C.launchcfg.ConfigError):
            C.load(tmp_path / "c.json")


def test_checks_name_each_fault():
    cfg = C.load(CONFIGS[0])
    import dataclasses
    bad = dataclasses.replace(cfg, launch=dataclasses.replace(cfg.launch, backend="k8s", env={"ARC_RL1_BUNDLE": "x"}),
                              watch={"run": "x"})
    probs = " | ".join(C.checks(bad))
    for want in ("launch.backend", "watch.run", "watch.progress_key", "launch.env"):
        assert want in probs, (want, probs)


def test_preflight_legs_carry_the_stub_env_and_a_study_trial():
    cfg = C.load(ROOT / "launch/rl/runs/b2-rl-hread-s0017-v2.json")
    import dataclasses
    leg = dict(cfg.preflight["-pf"])
    assert leg["ARC_RL1_PREFLIGHT_TRIAL"] == "heldout-read"
    del leg["ARC_RL1_PREFLIGHT_CORPUS"]
    probs = " | ".join(C.checks(dataclasses.replace(cfg, preflight={"-pf": leg})))
    assert "ARC_RL1_PREFLIGHT_CORPUS" in probs
    no_trial = {k: v for k, v in cfg.preflight["-pf"].items() if k != "ARC_RL1_PREFLIGHT_TRIAL"}
    assert "'rl-loop' is not a study trial" in " | ".join(C.checks(dataclasses.replace(cfg, preflight={"x": no_trial})))


def test_tree_hash_ignores_pycache_and_sees_content(tmp_path):
    (tmp_path / "a/__pycache__").mkdir(parents=True)
    (tmp_path / "a/x.py").write_text("x = 1\n")
    h = B.tree_sha256(tmp_path / "a")
    (tmp_path / "a/__pycache__/x.cpython-312.pyc").write_bytes(b"junk")
    assert B.tree_sha256(tmp_path / "a") == h
    (tmp_path / "a/x.py").write_text("x = 2\n")
    assert B.tree_sha256(tmp_path / "a") != h


def test_inline_checks_the_hash(tmp_path, monkeypatch):
    f = tmp_path / "sets.json"
    f.write_text(json.dumps([{"name": "s1"}]))
    ref = {"$file": str(f), "sha256": B.sha256_file(f)}
    assert B.inline({"mg": {"sets": ref, "k": 1}}) == {"mg": {"sets": [{"name": "s1"}], "k": 1}}
    with pytest.raises(SystemExit, match="sha256"):
        B.inline({"sets": {**ref, "sha256": "0" * 64}})


def test_manifest_skips_notebooks_and_itself(tmp_path):
    for n in ("MANIFEST.json", "experiment.ipynb", "kernel-metadata.json", "src/a.py"):
        (tmp_path / n).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / n).write_text(n)
    assert list(B.manifest(tmp_path)) == ["src/a.py"]


def test_launcher_refuses_a_bundle_not_built_from_this_config(tmp_path, monkeypatch):
    cfg = C.load(CONFIGS[0])
    monkeypatch.setattr(B, "DIST", tmp_path)
    assert any("not built" in p for p in L.check(cfg, CONFIGS[0], gate=False))
    (tmp_path / cfg.bundle).mkdir()
    (tmp_path / cfg.bundle / "MANIFEST.json").write_text(json.dumps({"spec_sha256": "x", "git_commit": cfg.source}))
    monkeypatch.setattr(C.launchcfg, "dry_run", lambda *a, **k: [])
    monkeypatch.setattr(L, "bundle_roster", lambda cfg: list(cfg.roster.ids))
    probs = L.check(cfg, CONFIGS[0], gate=False)
    assert probs == [f"bundle {tmp_path / cfg.bundle} was not built from this config (rebuild: launch/rl/build.py "
                     f"{CONFIGS[0]})"], probs


def test_modal_env_comes_from_the_config(monkeypatch, tmp_path):
    hero = C.load(ROOT / "launch/rl/runs/rl1-loop-hero-v6m2.json")
    monkeypatch.setattr(B, "DIST", tmp_path)
    e = L.modal_env(hero, {"ARC_RL1_PREFLIGHT_TAG": "PC"})
    assert e["ARC_RL1_BUNDLE"] == str(tmp_path / "rl1-loop-hero-v6m2") and e["ARC_RL1_PREFLIGHT_TAG"] == "PC"
    assert e["ARC_RL1_RESUME_FROM"].split(",")[-1] == "validation/rl1-loop-hero-v6m" and e["ARC_RL1_STALL"] == "1200"
    sys.path.insert(0, str(ROOT / "scripts/posttrain"))
    import ast
    tree = ast.parse((ROOT / "scripts/posttrain/modal_rl1.py").read_text())
    allowed = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and getattr(n.targets[0], "id", "") == "LAUNCH_ENV")
    assert {k for k in e if not k.startswith("ARC_RL1_PREFLIGHT_")} <= allowed  # nothing the launcher would drop


def test_store_seed_is_fetched_on_a_miss_and_hash_checked(tmp_path, monkeypatch):
    src = tmp_path / "remote/games-x"
    (src / "g1").mkdir(parents=True)
    (src / "g1/g1.py").write_text("x = 1\n")
    h = B.tree_sha256(src)
    monkeypatch.setenv("ARC_RL_DATA", str(tmp_path / "store"))
    calls = []

    def get(remote, into):  # stands in for `modal volume get arc-prolong-assets rl-seeds/<name> <into>`
        calls.append(remote)
        import shutil
        shutil.copytree(src, Path(into) / Path(remote).name)

    got = B.fetch("~/arc-rl-data/games-x", h, get=get)
    assert got == tmp_path / "store/games-x" and B.tree_sha256(got) == h and calls == ["rl-seeds/games-x"]
    assert B.fetch("~/arc-rl-data/games-x", h, get=get) == got and len(calls) == 1  # present: no second fetch
    with pytest.raises(SystemExit, match="fetched sha256"):
        B.fetch("~/arc-rl-data/games-y", "0" * 64, get=lambda r, i: get("x/games-x", i) or (Path(i) / "games-x").rename(Path(i) / "games-y"))


def test_spec_hash_ignores_launch_and_watch_fields():
    import dataclasses
    cfg = C.load(CONFIGS[0])
    h = B.spec_sha256(cfg)
    assert B.spec_sha256(dataclasses.replace(cfg, watch={**cfg.watch, "cadence_s": 1}, what="other")) == h
    assert B.spec_sha256(dataclasses.replace(cfg, source="0" * 40)) != h
