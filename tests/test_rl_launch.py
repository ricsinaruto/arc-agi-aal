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
    assert "sets" not in (cfg.study.get("rl_loop") or {}).get("mg", {})  # rl_seq builds its sets from the backlog (28 Oct
    # cut); a held-out read config plays level-bench only and has no rl_loop


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


def test_checks_run_the_boots_server_flags(tmp_path):
    d = json.loads((ROOT / "launch/rl/runs/b2-rl-hread-rd-v1.json").read_text())
    d["study"]["server"]["serving_kernels"]["mamba_ssm_cache_dtype"] = None  # as kvgolden-int4-v1 failed on the VM (26 Oct)
    (tmp_path / "c.json").write_text(json.dumps(d))
    probs = " | ".join(C.checks(C.load(tmp_path / "c.json")))
    assert 'mamba_ssm_cache_dtype "bfloat16"' in probs, probs


def test_preflight_legs_carry_the_stub_env_and_a_study_trial():
    cfg = C.load(ROOT / "launch/rl/runs/b2-rl-hread-rd-v1.json")
    import dataclasses
    leg = dict(cfg.preflight["-pf"])
    assert leg["ARC_RL1_PREFLIGHT_TRIAL"] == "heldout-read"
    del leg["ARC_RL1_PREFLIGHT_CORPUS"]
    probs = " | ".join(C.checks(dataclasses.replace(cfg, preflight={"-pf": leg})))
    assert "ARC_RL1_PREFLIGHT_CORPUS" in probs
    no_trial = {k: v for k, v in cfg.preflight["-pf"].items() if k != "ARC_RL1_PREFLIGHT_TRIAL"}
    assert "'rl-loop' is not a study trial" in " | ".join(C.checks(dataclasses.replace(cfg, preflight={"x": no_trial})))


def test_preflight_legs_carry_the_launch_knobs():  # real-v2 N1 (27 Oct): no put -> 120 slots = the back-off floor
    cfg = C.load(ROOT / "launch/rl/runs/rl1-loop-seq-real-v8.json")
    env = L.preflight_env(cfg, "N1")
    assert json.loads(env["ARC_RL1_PREFLIGHT_KNOBS"]) == dict(cfg.launch.knobs) == {"seq": {"active_tries": 280,
                                                                                          "streams": 560}}
    assert env["ARC_RL1_PREFLIGHT_PREEMPT_AFTER"] == "1000" and env["ARC_RL1_PREFLIGHT_TAG"] == "N1"

def test_nebius_play_window_leaves_room_for_the_boot():
    import dataclasses

    cfg = C.load(ROOT / "launch/rl/runs/b2-rl-hread-rd-v1.json")
    assert C.checks(cfg) == []
    study = {**cfg.study, "notebook_seconds": 900}  # hspeed-a v1: the ~11 min boot left 192 s of a 900 s window
    probs = " | ".join(C.checks(dataclasses.replace(cfg, study=study)))
    assert "the server start comes out of the play window" in probs
    probs = " | ".join(C.checks(dataclasses.replace(cfg, watch={**cfg.watch, "boot_s": 600}, max_seconds=2000)))
    assert "watch.boot_s 600" in probs and "max_seconds 2000" in probs


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
    path = next(p for p in CONFIGS if not C.pending(C.load(p)))  # a pending input is its own refusal (test_hread)
    cfg = C.load(path)
    monkeypatch.setattr(B, "DIST", tmp_path)
    assert any("not built" in p for p in L.check(cfg, path, gate=False))
    (tmp_path / cfg.bundle).mkdir()
    (tmp_path / cfg.bundle / "MANIFEST.json").write_text(json.dumps({"spec_sha256": "x", "git_commit": cfg.source}))
    monkeypatch.setattr(C.launchcfg, "dry_run", lambda *a, **k: [])
    monkeypatch.setattr(L, "bundle_roster", lambda cfg: list(cfg.roster.ids))
    monkeypatch.setattr(L, "r3_problems", lambda cfg, roots=None: [])  # its own test below
    probs = L.check(cfg, path, gate=False)
    assert probs == [f"bundle {tmp_path / cfg.bundle} was not built from this config (rebuild: launch/rl/build.py "
                     f"{path})"], probs


def test_modal_env_comes_from_the_config(monkeypatch, tmp_path):
    run = C.load(ROOT / "launch/rl/runs/rl1-loop-seq-real-v8.json")
    monkeypatch.setattr(B, "DIST", tmp_path)
    e = L.modal_env(run, {"ARC_RL1_PREFLIGHT_TAG": "PC"})
    assert e["ARC_RL1_BUNDLE"] == str(tmp_path / "rl1-loop-seq-real-v8") and e["ARC_RL1_PREFLIGHT_TAG"] == "PC"
    assert e["ARC_RL1_STALL"] == "1200" and e["ARC_RL1_TIMEOUT"] == "86400"
    sys.path.insert(0, str(ROOT / "scripts/posttrain"))
    import ast
    tree = ast.parse((ROOT / "scripts/posttrain/modal_rl1.py").read_text())
    allowed = {k for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in
               ("LAUNCH_ENV", "READ_KEYS") for k in ast.literal_eval(n.value)}  # READ_KEYS: ARC_RESTART_ACK
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


def test_build_refuses_a_bundle_missing_a_flat_import(tmp_path):
    """T3 (26 Oct): rl1_trainer imported rl1_dist, which config.files did not ship."""
    (tmp_path / "rl1").mkdir()
    (tmp_path / "rl1/rl1_trainer.py").write_text("import os\nimport rl1_dist\nfrom rl1_cp import x\nimport rl1_gdn\n")
    (tmp_path / "rl1/rl1_gdn.py").write_text("")
    local = {"rl1_trainer", "rl1_dist", "rl1_cp", "rl1_gdn"}
    assert B.missing_imports(tmp_path, local) == ["rl1/rl1_trainer.py imports rl1_cp", "rl1/rl1_trainer.py imports rl1_dist"]
    (tmp_path / "rl1/rl1_dist.py").write_text("")
    (tmp_path / "rl1/rl1_cp.py").write_text("")
    assert B.missing_imports(tmp_path, local) == []


def test_modal_watch_plan_reads_the_run_sampler(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MAIN", tmp_path)
    started = []
    monkeypatch.setattr(L, "background", lambda argv, log: started.append(argv))
    import dataclasses
    cfg = C.load(ROOT / "launch/rl/runs/rl1-loop-seq-real-v8.json")
    cfg = dataclasses.replace(cfg, watch={k: v for k, v in cfg.watch.items() if k != "export"})  # the default exporter
    plan = json.loads(L.watch(cfg, "ap-x", 1).read_text())
    # every leg gets its dashboard exporter, configured or not (dl2-v3 ran without one, 26 Oct)
    assert len(started) == 1 and started[0][2].endswith("tools/ops/export_rl.py") and started[0][3] == cfg.run
    assert f"--passes={tmp_path / 'runs/ops/export'}" in started[0] and "export" not in plan
    assert plan["arena"]["sampler"] == "modal:arc-prolong-runs:validation/rl1-loop-seq-real-v8/live/sampler.jsonl"
    assert plan["arena"]["cpulog"].endswith("/util/cpulog.txt") and plan["follow_record"] is True
    assert plan["arena"]["feed"].endswith("validation/rl1-loop-seq-real-v8/arena/trials/rl-loop/rl/feed.jsonl")
    # a resumed leg's exporter carries its parent's target metric until its first step (v15 29 Oct)
    started.clear()
    v15 = C.load(ROOT / "launch/rl/runs/rl1-loop-seq-real-v15.json")
    v15 = dataclasses.replace(v15, watch={k: v for k, v in v15.watch.items() if k != "export"})
    L.watch(v15, "ap-y", 1)
    assert "--continues=rl1-loop-seq-real-v14" in started[0]
    lost = dataclasses.replace(v15, launch=dataclasses.replace(v15.launch, resume_from=(
        "validation/rl1-loop-seq-real-v12-attempt2",)))
    L.watch(lost, "ap-z", 1)
    assert "--continues=rl1-loop-seq-real-v12" in started[1]


def test_a_trained_draft_needs_mtp_and_the_merge_on_every_trial():
    study = json.loads((ROOT / "launch/rl/runs/rl1-loop-seq-real-v8.json").read_text())["study"]
    assert C.draft_checks(study) == []  # draft off: nothing asked
    study["rl_loop"]["mg"]["draft_weight"] = 1.0
    probs = " | ".join(C.draft_checks(study))
    assert "lacks ARC_DRAFT_LORA=1" in probs and "mtp_tokens" not in probs  # real-v8 serves MTP 2
    for t in study["trials"]:
        t["server_env"]["ARC_DRAFT_LORA"] = "1"
    assert C.draft_checks(study) == []
    study["server"]["mtp_tokens"] = 0
    study["rl_loop"]["mg"]["redesign"]["card_env"] = {"1": {"ARC_DRAFT_LORA": None}}
    probs = " | ".join(C.draft_checks(study))
    assert "mtp_tokens < 1" in probs and "card_env 1 sets ARC_DRAFT_LORA None" in probs



def test_checks_refuse_a_trial_wall_that_ends_the_run_early():
    """G-long v3 (27 Oct): max_seconds 5400 with trial wall 1800 -> the bootstrap killed play at wall + 600 s."""
    import dataclasses
    cfg = next(C.load(p) for p in CONFIGS if C.load(p).launch.backend == "modal")
    trials = [{**t, "wall_seconds": 1800} for t in cfg.study.get("trials") or [{"name": "rl-loop"}]]
    bad = dataclasses.replace(cfg, max_seconds=5400, study={**cfg.study, "trials": trials})
    assert any("wall_seconds 1800 < max_seconds 5400" in p for p in C.checks(bad))
    good = dataclasses.replace(bad, study={**cfg.study, "trials": [{**t, "wall_seconds": 5400} for t in trials]})
    assert not any("wall_seconds" in p for p in C.checks(good))


def test_kill_resume_leg_cancels_then_resubmits_with_an_ack(tmp_path, monkeypatch):
    """NK (kill/resume a standard leg): attempt 0 is cancelled at +KILL_S, its record kept as -a0, the
    same tag resubmitted with an ARC_RESTART_ACK; the kill key never reaches the container; the verdict waits for a1."""
    import dataclasses

    cfg = C.load(CONFIGS[0])
    leg = {**next(iter(cfg.preflight.values())), L.KILL: "390"}
    cfg = dataclasses.replace(cfg, preflight={"NKt": leg})
    monkeypatch.setattr(L.C, "MAIN", tmp_path)
    (tmp_path / "docs/validation-rl").mkdir(parents=True)
    subs, slept, cancelled = [], [], []

    def submit(c, tag, extra=None, log_suffix=""):
        subs.append(dict(extra or {}))
        L.leg_record(c, tag).write_text(json.dumps({"function_call_id": f"fc-{len(subs)}"}))
        return f"ap-{len(subs)}"

    monkeypatch.setattr(L, "submit", submit)
    env = L.preflight_env(cfg, "NKt")
    assert L.KILL not in env and env["ARC_RL1_PREFLIGHT_MAX_STEPS"] == "4" and env["ARC_RL1_PREFLIGHT_RESTORE_CHECK"] == "1"
    printed = []
    monkeypatch.setattr(L, "leg_call", lambda c, t: ("running", None))
    monkeypatch.setattr(L, "leg_finished", lambda c, t: False)
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    L.verdict(cfg, ["NKt"])
    assert "attempt 1 not submitted" in printed[-1]
    # attempt 0 kills itself at its first checkpoint (no Modal cancel, which can run the finally)
    gone = iter([False, False, True])
    L.nk(cfg, "NKt", sleep=slept.append, cancel=cancelled.append, step=lambda: 1, ended=lambda fc: next(gone),
         identity=lambda: False)
    assert slept == [20.0, 20.0, 60] and cancelled == []
    assert subs[0] == {"ARC_RL1_PREFLIGHT_SELF_KILL": "1"} and "planned kill/resume" in subs[1]["ARC_RESTART_ACK"]
    assert "ARC_RL1_PREFLIGHT_SELF_KILL" not in subs[1]
    assert json.loads(L.leg_record(cfg, "NKt", "-a0").read_text())["function_call_id"] == "fc-1"
    assert json.loads(L.leg_record(cfg, "NKt").read_text())["function_call_id"] == "fc-2"
    monkeypatch.setattr(L, "LOGS", tmp_path)
    log = tmp_path / f"{cfg.bundle}NKt-nk.log"
    monkeypatch.setattr("builtins.print", lambda *a, **k: (printed.append(" ".join(map(str, a))),
                                                            log.open("a").write(" ".join(map(str, a)) + "\n")))
    # NKc (29 Oct): attempt 0 ended (it wrote its identity) instead of dying: no second attempt, the verdict says so
    subs.clear(), slept.clear(), cancelled.clear()
    L.nk(cfg, "NKt", sleep=slept.append, cancel=cancelled.append, step=lambda: 1, ended=lambda fc: True,
         identity=lambda: True)
    assert len(subs) == 1 and cancelled == []
    L.verdict(cfg, ["NKt"])
    assert "attempt 0 ended instead of dying" in printed[-1]
    # no checkpoint within the ceiling: attempt 0 cancelled, no second attempt, the verdict fails
    subs.clear(), slept.clear(), cancelled.clear()
    L.nk(cfg, "NKt", sleep=slept.append, cancel=cancelled.append, step=lambda: 0, ended=lambda fc: False,
         identity=lambda: False)
    assert len(subs) == 1 and cancelled == ["fc-1"] and sum(slept) >= 390
    L.leg_record(cfg, "NKt", "-a0").unlink(missing_ok=True)
    L.verdict(cfg, ["NKt"])
    assert "fail (kill/resume: no checkpoint" in printed[-1]


def test_r3_gate_refuses_a_route_less_resume_and_r3_off_training():
    """real-v9 28 Oct: a resume of real-v6/v7 (built before R3 in training) trained route-less groups."""
    import dataclasses

    cfg = C.load(L.HERE / "runs" / "rl1-loop-seq-real-v10.json")
    assert cfg.launch.resume_from == ("validation/rl1-loop-seq-real-v10",)  # attempt 2 resumes its own R3-on attempt 1
    assert L.r3_problems(cfg, ancestor=lambda a, b: True) == []  # (CI's shallow clone has no history to walk)
    probs = L.r3_problems(cfg, ["validation/rl1-loop-seq-real-v9"], ancestor=lambda a, b: False)
    assert len(probs) == 1 and "before R3 in training" in probs[0]
    assert L.r3_problems(cfg, ["validation/rl1-loop-seq-real-v9"], ancestor=lambda a, b: True) == []
    assert "no checked-in config" in L.r3_problems(cfg, ["validation/rl1-loop-seq-real-v6"])[0]
    # an attempt or preempted folder is judged by its run's config (real-v12 resumes real-v11-attempt3's groups)
    for folder in ("validation/rl1-loop-seq-real-v9-attempt3", "validation/rl1-loop-seq-real-v9-attempt2-lost-17"):
        assert L.r3_problems(cfg, [folder], ancestor=lambda a, b: True) == []
    raw = json.loads((L.HERE / "runs" / "rl1-loop-seq-real-v10.json").read_text())
    raw["study"]["rl_loop"]["mg"]["served_route"] = False
    assert "must be on" in L.r3_problems(dataclasses.replace(cfg, study=raw["study"]))[0]


def test_put_knobs_retries_a_failed_read_back(tmp_path):
    import put_knobs as P
    k = tmp_path / "k.json"
    k.write_text(json.dumps({"seq": {"streams": 304}}))
    ok = type("R", (), {"returncode": 0})()
    calls = []

    def get(path, dest, fails=[1, 1]):  # fails twice (volume under load), then returns the put
        calls.append(path)
        if fails:
            fails.pop()
            return False
        Path(dest).write_text(k.read_text())
        return True

    P.put("r", k, sh=lambda *a, **kw: ok, get=get, is_started=lambda r: False)
    assert len(calls) == 3
    with pytest.raises(SystemExit, match="READ-BACK FAILED"):
        P.put("r", k, sh=lambda *a, **kw: ok, get=lambda p, d: False, is_started=lambda r: False)


def test_per_run_volume_config_paths_env_and_knobs(monkeypatch, tmp_path):
    """Per-run volumes: ARC_RL1_VOLUME "run" -> arc-run-<run> at /runs, caches at /cache, other runs'
    folders at /in/<volume>/ (the config says so; the check refuses a /runs path into another run or a cache)."""
    import dataclasses
    from arc_infra.runops import records
    cfg = C.load(ROOT / "launch/rl/runs/rl1-loop-seq-real-v11.json")
    assert C.volume_checks(cfg) == [] and L.run_volume(cfg) == "arc-prolong-runs"
    monkeypatch.setattr(B, "DIST", tmp_path)
    assert "ARC_RL1_VOLUME" not in L.modal_env(cfg) and "ARC_RL1_INPUT_VOLUMES" not in L.modal_env(cfg)
    per = dataclasses.replace(cfg, launch=dataclasses.replace(cfg.launch, env={**cfg.launch.env, "ARC_RL1_VOLUME": "run"}))
    probs = " | ".join(C.volume_checks(per))
    assert "/runs/boot-cache/rd: a cache on a per-run volume is /cache/boot-cache/rd" in probs
    assert "/runs/validation/rl1-loop-seq-t4-v2/adapters/v0001: another run's folder" in probs
    assert "validation/rl1-screen-seed-v2/arena/trials/rl-loop/rl: another run's folder" in probs
    assert "real-v11-census" not in probs  # the run's own folders stay at /runs
    raw = json.dumps({"study": per.study, "env": per.launch.env, "pf": per.preflight}, default=dict)
    raw = raw.replace('"/runs/boot-cache/', '"/cache/boot-cache/').replace(
        "/runs/validation/rl1-loop-seq-t4-v2", "/in/arc-prolong-runs/validation/rl1-loop-seq-t4-v2")
    raw = __import__("re").sub(r'(?<=[",])validation/rl1-screen', "/in/arc-prolong-runs/validation/rl1-screen", raw)
    d = json.loads(raw)
    fixed = dataclasses.replace(per, study=d["study"], preflight=d["pf"],
                                launch=dataclasses.replace(per.launch, env=d["env"]))
    assert C.volume_checks(fixed) == []
    e = L.modal_env(fixed)
    assert e["ARC_RL1_VOLUME"] == records.run_volume_name(cfg.run) == "arc-run-rl1-loop-seq-real-v11"
    assert e["ARC_RL1_INPUT_VOLUMES"] == "arc-prolong-runs"
    bad = dataclasses.replace(fixed, launch=dataclasses.replace(fixed.launch, env={**fixed.launch.env,
                                                                                   "ARC_RL1_INPUT_VOLUMES": "x"}))
    assert "set by launch.py" in C.volume_checks(bad)[0]
    import put_knobs as P
    k = tmp_path / "k.json"
    k.write_text(json.dumps({"seq": {"streams": 304}}))
    calls = []
    ok = type("R", (), {"returncode": 0})()
    def get(path, dest, volume="arc-prolong-runs"):
        calls.append(volume)
        Path(dest).write_text(k.read_text())
        return True

    monkeypatch.setattr(P, "volume_get", get)
    P.put("r", k, sh=lambda cmd, **kw: calls.append(cmd[4]) or ok, is_started=lambda r: False, vol_of=lambda r: "arc-run-r")
    assert calls == ["arc-run-r", "arc-run-r"]


def test_resume_paths_from_another_runs_volume():
    from arena.rl_loop import Loop
    tag = Loop.mg_tag
    spec = {"level": 3}
    assert tag(spec, "/in/arc-prolong-runs/validation/rl1-x/a/b", 2) == tag(spec, "/runs/validation/rl1-x/a/b", 2)
    m = pytest.importorskip("modal_rl1")
    assert m.vabs("validation/x/y") == "/runs/validation/x/y" and m.vabs("/in/v/validation/x") == "/in/v/validation/x"
    assert str(m.vrel("/in/arc-prolong-runs/validation/preflight/p")) == "validation/preflight/p"
    assert str(m.vrel("/runs/validation/preflight/p")) == "validation/preflight/p"


def test_verdict_judges_a_leg_whose_function_ended_while_its_app_lingers(tmp_path, monkeypatch):
    """NKd 29 Oct: the function ended (identity + in-container verdict on the volume) but its detached app still read
    running: the verdict judges it instead of waiting."""
    import dataclasses

    cfg = C.load(CONFIGS[0])
    cfg = dataclasses.replace(cfg, preflight={"N1t": {"ARC_RL1_PREFLIGHT_SPEED": "0"}})  # an rl-loop leg
    monkeypatch.setattr(L, "leg_call", lambda c, t: ("running", None))
    ran, printed = [], []
    monkeypatch.setattr(L, "LOGS", tmp_path)
    monkeypatch.setattr(L, "sh", lambda cmd, log=None, env=None, check=True: (
        ran.append(cmd), Path(log).write_text('"verdict": "pass"'), type("R", (), {"returncode": 0})())[-1])
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    monkeypatch.setattr(L, "leg_finished", lambda c, t: False)
    L.verdict(cfg, ["N1t"])
    assert "running" in printed[-1] and not ran
    monkeypatch.setattr(L, "leg_finished", lambda c, t: True)
    L.verdict(cfg, ["N1t"])
    assert ran and ": pass" in printed[-1]


def test_relaunch_refuses_a_dropped_live_knob():
    """v17 a9 (30 Oct): ckpt_budget_gib 230 was set live on a8; a9's launch put only the config's knobs and U22 ran at
    the settings' 240. A live knob of the resumed attempt missing from launch.knobs is a drop; a changed value is not."""
    import put_knobs as P

    prev = {"seq": {"minibatches": 1, "lr_scale": 1.0, "ckpt_budget_gib": 230}}
    assert P.dropped(prev, {"seq": {"minibatches": 1, "lr_scale": 1.0}}) == {"seq.ckpt_budget_gib": 230}
    assert P.dropped(prev, {"seq": {"minibatches": 2, "lr_scale": 1.0, "ckpt_budget_gib": 230, "epochs": 1}}) == {}
    assert P.dropped(None, {"seq": {}}) == {}
