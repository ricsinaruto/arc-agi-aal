import importlib.util, json, sys
from pathlib import Path

import pytest

modal = pytest.importorskip("modal")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("modal_rl1_gate", ROOT / "scripts/posttrain/modal_rl1.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def base():
    return {"name": "a", "rl_loop": {"lr": 1.5e-5, "mg": {"kl_target": 0.02, "lr_dose": False,
                                                          "kl_ramp": {"start": 0.01, "cap": 0.05},
                                                          "behaviour_guard": {"clear_drop": 0.15, "actions_rise": None}}}}


def test_value_changes_are_settings_and_switches_are_code():
    new = json.loads(json.dumps(base()))
    new["name"] = "b"; new["rl_loop"]["lr"] = 2e-5; new["rl_loop"]["mg"]["kl_ramp"]["cap"] = 0.04
    assert M.settings_diff(base(), new)[1] == []
    for edit in (lambda m: m.update(lr_dose=True), lambda m: m.update(harvest={"levels": [2]}),
                 lambda m: m.pop("kl_ramp"), lambda m: m["behaviour_guard"].update(actions_rise=0.3)):
        new = json.loads(json.dumps(base()))
        edit(new["rl_loop"]["mg"])
        assert M.settings_diff(base(), new)[1], edit


def test_settings_only_gate_needs_same_code(tmp_path):
    def bundle(name, cfg, code):
        d = tmp_path / name; (d / "configs").mkdir(parents=True)
        (d / "configs/comparison.json").write_text(json.dumps(cfg))
        (d / "MANIFEST.json").write_text(json.dumps({"files": {"configs/comparison.json": name, "src/x.py": code}}))
        return d
    import hashlib
    old = bundle("old", base(), "c1")
    receipt = tmp_path / "preflight-old.json"
    receipt.write_text(json.dumps({"verdict": "pass", "bundle": str(old),
                                   "manifest_sha256": hashlib.sha256((old / "MANIFEST.json").read_bytes()).hexdigest()}))
    cfg = base(); cfg["rl_loop"]["lr"] = 3e-5
    assert M.settings_only_gate(bundle("new", cfg, "c1"), [receipt]).startswith("settings-only")
    assert M.settings_only_gate(bundle("code", cfg, "c2"), [receipt]) is None
    cfg["rl_loop"]["mg"]["lr_dose"] = True
    assert M.settings_only_gate(bundle("switch", cfg, "c1"), [receipt]) is None


def test_pack_reads_retry_on_corrupt_copy_and_checksum_mismatch(tmp_path):
    import shutil
    import tarfile

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("x" * 5000)
    good = tmp_path / "good.tar.gz"
    with tarfile.open(good, "w:gz") as tar:
        tar.add(src / "a.txt", arcname="a.txt")
    want = M.sha256_file(good)
    bad = good.read_bytes()
    bad = bad[:-12] + bytes(b ^ 0xFF for b in bad[-12:-8]) + bad[-8:]  # gzip trailer CRC no longer matches

    def fetch_from(seq):
        def fetch(n):
            local = tmp_path / f"copy{n}.tar.gz"
            local.write_bytes(seq[n]) if isinstance(seq[n], bytes) else shutil.copyfile(seq[n], local)
            return local
        return fetch

    target = tmp_path / "t"
    assert M.extract_checked(fetch_from([bad, good]), target) == 2  # CRC caught by the read itself
    assert (target / "a.txt").read_text() == "x" * 5000
    assert M.extract_checked(fetch_from([bad, bad, good]), target, want) == 3
    with pytest.raises(RuntimeError, match="unreadable after 3"):
        M.extract_checked(fetch_from([bad, bad, bad]), target, want)
    assert not target.exists()


def test_judge_reads_any_tree_and_the_pack_keeps_the_judgement_outside(tmp_path):
    """In-app verdict: the same checks on the container's local tree; the judged file stays beside
    the identity, outside the pack."""
    base = "validation/preflight/rl1-loop-x-pf1"
    root = tmp_path / base
    rl = root / "arena/trials/rl-loop/rl"
    (rl / "r01-sc25-L3").mkdir(parents=True)
    (root / "arena/bootstrap-results.json").write_text(json.dumps({"rl-loop": "complete"}))
    feed = [{"kind": "round", "round": 1, "update": {"ok": True}, "swap": {"load_seconds": 1.0}},
            {"kind": "checkpoint", "ok": True}, {"kind": "final", "stop": "max_rounds"}]
    (rl / "feed.jsonl").write_text("\n".join(json.dumps(e) for e in feed) + "\n")
    (rl / "r01-sc25-L3/level-bench.json").write_text(json.dumps({"attempts": [
        {"label": "a", "restore": {"ok": True, "prompt_digest": {"same": True}}}]}))
    (root / "arena/serving").mkdir(parents=True)
    (root / "arena/serving/vllm-openai-server.log").write_text('x STUB {"errors": 0, "exact": 3} KV_STUB {"peak": 0.0, "failures": 0}\n')
    (root / "bootstrap.log").write_text("ok\n")
    read = lambda r: (tmp_path / r).read_text() if (tmp_path / r).is_file() else None
    names = lambda r: [p.name for p in (tmp_path / r).iterdir()]
    got = M.judge(base, read, names, "", {"rl_loop": {"mg": {}}}, {"exit_code": 0, "elapsed_seconds": 12.0})
    assert got["verdict"] == "pass" and list(got) == ["verdict", "checks", "restore", "stub", "feed_kinds", "elapsed_seconds"]
    assert got["restore"] == {"starts": 1, "bad": [], "state_only": 0} and got["elapsed_seconds"] == 12.0
    assert M.judge(base, read, names, "", {"rl_loop": {"mg": {}}}, {"exit_code": 1})["verdict"] == "fail"
    (root / M.JUDGED).write_text(json.dumps(got))
    (root / "modal-identity.json").write_text("{}")
    M.pack(root)
    assert sorted(p.name for p in root.iterdir()) == sorted([M.JUDGED, "modal-identity.json", M.PACK, M.PACK + ".sha256"])


def test_screen_judge_accepts_levels_no_recording_reached_but_not_an_unusable_recording(tmp_path):
    """Seed screen (envs 22 Oct): a record-fed set without starts is fine; a failed recording is not."""
    base = "validation/preflight/rl1-screen-x-pf1"
    root = tmp_path / base
    rl = root / "arena/trials/rl-loop/rl"
    (rl / "s-L2").mkdir(parents=True)
    (root / "arena/bootstrap-results.json").write_text(json.dumps({"rl-loop": "complete"}))
    (rl / "s-L2/level-bench.json").write_text(json.dumps({"attempts": [
        {"label": "a", "restore": {"ok": True, "prompt_digest": {"same": True}}}]}))
    (rl / "screen.json").write_text(json.dumps({"sets": {"g-L2": {"tries": 2, "folder": "/x/s-L2"}},
                                                "missing": ["g-L3"]}))
    (root / "arena/serving").mkdir(parents=True)
    (root / "arena/serving/vllm-openai-server.log").write_text('x STUB {"errors": 0, "exact": 3}\n')
    (root / "bootstrap.log").write_text("ok\n")
    read = lambda r: (tmp_path / r).read_text() if (tmp_path / r).is_file() else None
    names = lambda r: [p.name for p in (tmp_path / r).iterdir()]
    study = {"rl_loop": {"mg": {"sets": [{"name": "g-L2", "level": 2, "sources": ["record:g"]},
                                         {"name": "g-L3", "level": 3, "sources": ["record:g"]},
                                         {"name": "g-L1", "level": 1, "sources": ["record:g"]}]}}}

    def verdict(feed):
        (rl / "feed.jsonl").write_text("\n".join(json.dumps(e) for e in feed + [{"kind": "final"}]) + "\n")
        return M.judge(base, read, names, "", study, {"exit_code": 0})

    ok = {"kind": "mg_record", "game": "g", "play": {"usable": 1}}
    got = verdict([ok, {"kind": "mg_set_missing", "name": "g-L3", "error": "no starts"}])
    assert got["verdict"] == "pass" and got["restore"]["unreached"] == 1 and got["restore"]["unreached_sets"] == ["g-L3"]
    none = verdict([{"kind": "mg_set_missing", "name": "g-L3", "error": "no starts"}])  # the recording path broken
    assert none["verdict"] == "fail" and none["checks"]["every_record_usable"] is False
    (rl / "screen.json").write_text(json.dumps({"sets": {"g-L2": {"tries": 2, "folder": "/x/s-L2"}}, "missing": ["g-L1"]}))
    l1 = verdict([ok, {"kind": "mg_set_missing", "name": "g-L1", "error": "no starts"}])
    assert l1["checks"]["every_set_played"] is False  # an L1 set is never "unreached"
    (rl / "screen.json").write_text(json.dumps({"sets": {"g-L2": {"tries": 2, "folder": "/x/s-L2"}}, "missing": ["g-L3"]}))
    assert verdict([ok])["checks"]["every_set_played"] is False  # missing for another reason
    bad = verdict([{**ok, "play": {"usable": 0}}, {"kind": "mg_set_missing", "name": "g-L3", "error": "no starts"}])
    assert bad["verdict"] == "fail" and bad["checks"]["every_record_usable"] is False


def test_reconcile_verdict():
    ok = "B300+B200 live 6/8, RTX VMs live 3/8\n"
    assert M.reconcile_verdict(ok, 2, 8, "") == "B300+B200 live 6/8 + 2"
    with pytest.raises(SystemExit, match="> cap 8"):
        M.reconcile_verdict(ok, 3, 8, "")
    stale = ok + "ISSUE queue row rl1-loop-m6-v6x-r4 says running but app ap-x is not live\n"
    assert M.reconcile_verdict(stale, 2, 8, "validation/a,validation/rl1-loop-m6-v6x-r4")  # chained death relaunch
    with pytest.raises(SystemExit, match="says running"):
        M.reconcile_verdict(stale, 2, 8, "validation/rl1-loop-m6-v6x-r3")
    with pytest.raises(SystemExit, match="no queue row"):
        M.reconcile_verdict(ok + "ISSUE modal app ap-y (z) has no queue row\n", 1, 8, "")
    with pytest.raises(SystemExit, match="no count"):
        M.reconcile_verdict("Traceback", 1, 8, "")


def test_launch_env_errors_name_dropped_variables():
    """23 Oct: ARC_RL1_EVAL_ADAPTERS in the launch env never reached the run."""
    env = {"ARC_RL1_BUNDLE": "dist/x", "ARC_RL1_PREFLIGHT_TAG": "-pf", "ARC_RL1_EVAL_ADAPTERS": "{}", "ARC_RL1_TYPO": "1",
           "OTHER": "1"}
    bad = M.launch_env_errors(env)
    assert len(bad) == 2 and "ARC_RL1_OVERRIDES" in bad[0] and bad[1].startswith("ARC_RL1_TYPO")
    assert M.launch_env_errors({"ARC_RL1_BUNDLE": "x", "ARC_RL1_OVERRIDES": "{}"}) == []


def test_judge_passes_alert_only_k3_and_tis_rows_but_not_other_alerts(tmp_path):
    """alert-only dosing: k3-over and TIS alerts report (the step was taken); any other alert fails."""
    base = "validation/preflight/rl1-loop-x-pf2"
    root = tmp_path / base
    rl = root / "arena/trials/rl-loop/rl"
    (rl / "r01-sc25-L3").mkdir(parents=True)
    (root / "arena/bootstrap-results.json").write_text(json.dumps({"rl-loop": "complete"}))
    (rl / "r01-sc25-L3/level-bench.json").write_text(json.dumps({"attempts": [
        {"label": "a", "restore": {"ok": True, "prompt_digest": {"same": True}}}]}))
    (root / "arena/serving").mkdir(parents=True)
    (root / "arena/serving/vllm-openai-server.log").write_text('x STUB {"errors": 0, "exact": 3} KV_STUB {"peak": 0.0, "failures": 0}\n')
    (root / "bootstrap.log").write_text("ok\n")
    read = lambda r: (tmp_path / r).read_text() if (tmp_path / r).is_file() else None
    names = lambda r: [p.name for p in (tmp_path / r).iterdir()]
    head = [{"kind": "round", "round": 1, "update": {"ok": True}, "swap": {"load_seconds": 1.0}}, {"kind": "checkpoint", "ok": True}]

    def verdict(alerts):
        feed = head + [{"kind": "alert", "why": w} for w in alerts] + [{"kind": "final", "stop": "max_rounds"}]
        (rl / "feed.jsonl").write_text("\n".join(json.dumps(e) for e in feed) + "\n")
        return M.judge(base, read, names, "", {"rl_loop": {"mg": {}}}, {"exit_code": 0})["verdict"]

    assert verdict(["k3 over stop (alert only, step taken)", "TIS: token share outside [0.5, 2] over the alert line"]) == "pass"
    assert verdict(["k3 over target (alert only, step taken)", "k3 measure failed (step kept)"]) == "fail"
    assert verdict(["time share failed; off for the rest of the run"]) == "fail"
    assert verdict(["window fallback 32768 (alert only)"]) == "pass"




def test_v7_judge_checks_steps_state_reads_and_mode_b(tmp_path):
    """v7 (mg.async): step rows with consecutive versions and finite grad norms, the last step's state saved, a read
    once read_every steps ran, and Mode B blocks that moved tries (L1e: Mode B served 0 tries)."""
    base = "validation/preflight/rl1-loop-v7-pf1"
    root = tmp_path / base
    rl = root / "arena/trials/rl-loop/rl"
    (rl / "g00001-sc25-L3").mkdir(parents=True)
    (root / "state/s0002").mkdir(parents=True)
    (root / "arena/bootstrap-results.json").write_text(json.dumps({"rl-loop": "complete"}))
    (rl / "g00001-sc25-L3/level-bench.json").write_text(json.dumps({"attempts": [
        {"label": "a", "restore": {"ok": True, "prompt_digest": {"same": True}}}]}))
    (root / "arena/serving").mkdir(parents=True)
    (root / "arena/serving/vllm-openai-server.log").write_text('x STUB {"errors": 0, "exact": 3} KV_STUB {"peak": 0.0}\n')
    (root / "bootstrap.log").write_text("ok\n")
    read = lambda r: (tmp_path / r).read_text() if (tmp_path / r).is_file() else None
    names = lambda r: [p.name for p in (tmp_path / r).iterdir()]
    study = {"rl_loop": {"mg": {"async": True, "v7": {"read_every": 8}}}}

    def run(served, grad=0.01, read_row=None):
        feed = [{"kind": "v7_start", "cfg": {"read_every": 2, "card2": {"on": True}}},
                {"kind": "step", "step": 1, "version": 1, "grad_norm": grad, "load_seconds": 1.0},
                {"kind": "block_off", "served": served}, {"kind": "step", "step": 2, "version": 2, "grad_norm": 0.02},
                read_row or {"kind": "read", "step": 2}, {"kind": "v7_end", "why": "max_steps"}]
        (rl / "feed.jsonl").write_text("\n".join(json.dumps(e) for e in feed) + "\n")
        return M.judge(base, read, names, "", study, {"exit_code": 0})

    got = run(3)
    assert got["verdict"] == "pass" and got["checks"]["reads_landed"] and got["checks"]["mode_b_played"], got
    assert not run(0)["checks"]["mode_b_played"] and run(0)["verdict"] == "fail"
    # a manual stop (stop leg) cuts the open reads instead of waiting on them (58d786bb7): still a pass
    assert run(3, read_row={"kind": "reads_cut", "n": 1, "open": [2]})["checks"]["reads_landed"]
    assert not run(3, read_row={"kind": "group_done"})["checks"]["reads_landed"]
    assert not run(3, grad=float("nan"))["checks"]["steps_ok"]
    (rl / "feed.jsonl").write_text((rl / "feed.jsonl").read_text() + json.dumps(
        {"kind": "alert", "why": "k3 over 0.05 (alert only)"}) + "\n" + json.dumps({"kind": "alert", "why": "TIS: token share"}) + "\n")
    assert M.judge(base, read, names, "", study, {"exit_code": 0})["checks"]["feed_final_no_error"]
    (rl / "feed.jsonl").write_text((rl / "feed.jsonl").read_text() + json.dumps({"kind": "alert", "why": "v7: accumulate failed"}) + "\n")
    assert not M.judge(base, read, names, "", study, {"exit_code": 0})["checks"]["feed_final_no_error"]


def test_pack_preempted_before_it_is_whole_leaves_the_tree_readable(tmp_path, monkeypatch):
    """xv7d pfA (envs): a preemption mid-pack left a half-written pack.tar.gz and the restart failed on its CRC. The
    pack is written aside and renamed once whole; until then the dir reads as it is."""
    root = tmp_path / "runs/validation/preflight/x-pfA"
    (root / "state/s0001").mkdir(parents=True)
    (root / "v7_state.json").write_text("{}")
    (root / "modal-identity.json").write_text("{}")

    def preempted(path):
        raise SystemExit("preempted")

    monkeypatch.setattr(M, "sha256_file", preempted)
    with pytest.raises(SystemExit):
        M.pack(root)
    assert not (root / M.PACK).exists() and (root / "v7_state.json").is_file() and (root / "state/s0001").is_dir()
    assert M.unpacked(str(root)) == str(root)
    monkeypatch.undo()
    M.pack(root)  # the next pack replaces the leftover part
    assert sorted(p.name for p in root.iterdir()) == sorted(["modal-identity.json", M.PACK, M.PACK + ".sha256"])


def test_v7_judge_reads_a_preempted_run_as_the_union_of_its_attempts(tmp_path):
    """hero-v5 A2 (24 Oct): preempted after its 10 steps; the resumed attempt went straight to v7_end and its own feed
    held no step row, so the judge failed steps / checkpoints / adapter / Mode B. The earlier attempt's feed and state
    (identity preempt_resume, the renamed -lost dir) count too."""
    base, lost = "validation/preflight/rl1-loop-v7-pfA", "validation/preflight/rl1-loop-v7-pfA-lost-1"
    for where in (base, lost):
        (tmp_path / where / "arena/trials/rl-loop/rl").mkdir(parents=True)
        (tmp_path / where / "arena/serving").mkdir(parents=True)
    (tmp_path / lost / "state/s0002").mkdir(parents=True)
    (tmp_path / base / "state").mkdir(parents=True)
    (tmp_path / base / "arena/bootstrap-results.json").write_text(json.dumps({"rl-loop": "complete"}))
    (tmp_path / base / "arena/serving/vllm-openai-server.log").write_text('x STUB {"errors": 0, "exact": 3} KV_STUB {"peak": 0.0}\n')
    (tmp_path / base / "bootstrap.log").write_text("ok\n")
    (tmp_path / base / "arena/trials/rl-loop/rl/g00003-sc25-L3").mkdir()
    (tmp_path / base / "arena/trials/rl-loop/rl/g00003-sc25-L3/level-bench.json").write_text(json.dumps({"attempts": [
        {"label": "a", "restore": {"ok": True, "prompt_digest": {"same": True}}}]}))
    first = [{"kind": "v7_start", "cfg": {"read_every": 2, "card2": {"on": True}}},
             {"kind": "step", "step": 1, "version": 1, "grad_norm": 0.01, "load_seconds": 1.0},
             {"kind": "block_off", "served": 3}, {"kind": "step", "step": 2, "version": 2, "grad_norm": 0.02},
             {"kind": "read", "step": 2}]
    second = [{"kind": "v7_start", "resume": True, "step": 2, "version": 2, "cfg": {"read_every": 2, "card2": {"on": True}}},
              {"kind": "read_lost", "step": 2}, {"kind": "v7_end", "why": "max_steps"}]
    (tmp_path / lost / "arena/trials/rl-loop/rl/feed.jsonl").write_text("\n".join(json.dumps(e) for e in first) + "\n")
    (tmp_path / base / "arena/trials/rl-loop/rl/feed.jsonl").write_text("\n".join(json.dumps(e) for e in second) + "\n")
    read = lambda r: (tmp_path / r).read_text() if (tmp_path / r).is_file() else None
    names = lambda r: [p.name for p in (tmp_path / r).iterdir()]
    study = {"rl_loop": {"mg": {"async": True, "v7": {"read_every": 8}}}}
    alone = M.judge(base, read, names, "", study, {"exit_code": 0})
    assert alone["verdict"] == "fail" and not alone["checks"]["steps_ok"]
    got = M.judge(base, read, names, "", study, {"exit_code": 0, "preempt_resume": f"/runs/{lost}"})
    assert got["verdict"] == "pass", got
    assert got["checks"]["steps_ok"] and got["checks"]["checkpoints_saved"] and got["checks"]["mode_b_played"]



def test_a_stop_put_before_a_preemption_restart_reaches_the_resumed_attempt(tmp_path):
    """hero-v5 S4 (24 Oct): the stop-request was put at the right path 80 s before a preemption restart; the restart
    renamed the attempt to -lost-<epoch> with the request inside and carried only knobs.json, so the resumed attempt ran
    on. Every live request now carries over."""
    lost, out = tmp_path / "pfS4-lost-1792818044", tmp_path / "pfS4"
    (lost / "arena/live").mkdir(parents=True)
    (lost / "arena/live/stop-request").write_text('{"reason": "S4"}')
    (lost / "arena/live/knobs.json").write_text("{}")
    out.mkdir()
    assert M.carry_live(lost, out) == ["knobs.json", "stop-request"]
    assert (out / "arena/live/stop-request").read_text() == '{"reason": "S4"}'
    assert M.carry_live(None, out) == [] and M.carry_live(tmp_path / "none", out) == []


class FakeVolume:
    """The runs volume over a local folder: listdir raises for a missing folder; batch_upload writes files."""

    def __init__(self, root):
        self.root = root

    def listdir(self, rel):
        path = self.root / rel
        if not path.is_dir():
            raise FileNotFoundError(rel)
        return [type("E", (), {"path": f"{rel}/{p.name}"})() for p in path.iterdir()]

    def batch_upload(self, force=False):
        root = self.root

        class Batch:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def put_file(self, data, rel):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_bytes(data.read())

        return Batch()


def test_stop_tool_writes_to_the_live_attempt_of_a_resumed_run_and_refuses_otherwise(tmp_path):
    spec2 = importlib.util.spec_from_file_location("request_stop", ROOT / "scripts/arena/request_stop.py")
    rs = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(rs)
    base = tmp_path / "validation/preflight/rl1-loop-hero-v5S4"
    (tmp_path / "validation/preflight/rl1-loop-hero-v5S4-lost-1792818044/arena/live").mkdir(parents=True)
    (base / "arena").mkdir(parents=True)
    vol = FakeVolume(tmp_path)
    got = rs.main(["rl1-loop-hero-v5", "--reason", "t", "--preflight-tag", "S4"], volume=vol)
    assert got == "validation/preflight/rl1-loop-hero-v5S4/arena/live/stop-request" and (tmp_path / got).is_file()
    assert not (tmp_path / "validation/preflight/rl1-loop-hero-v5S4-lost-1792818044/arena/live/stop-request").exists()
    with pytest.raises(SystemExit, match="dead attempt"):  # the resumed run's -lost- id names a dead attempt
        rs.main(["rl1-loop-hero-v5", "--reason", "t", "--preflight-tag", "S4-lost-1792818044"], volume=vol)
    with pytest.raises(SystemExit, match="does not exist"):  # wrong tag / id
        rs.main(["rl1-loop-hero-v5", "--reason", "t", "--preflight-tag", "S9"], volume=vol)
    with pytest.raises(SystemExit, match="does not exist"):  # a preflight addressed as a main run
        rs.main(["rl1-loop-hero-v5S4", "--reason", "t"], volume=vol)
    (base / "modal-identity.json").write_text("{}")
    with pytest.raises(SystemExit, match="finished"):
        rs.main(["rl1-loop-hero-v5", "--reason", "t", "--preflight-tag", "S4"], volume=vol)
