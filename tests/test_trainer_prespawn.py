import json, subprocess, sys, time

from arena import trainer_prespawn as tp


def pre(tmp_path, command, env):
    folder = tmp_path / "trainer"; (folder / "cmd").mkdir(parents=True)
    p = subprocess.Popen(command, env=env, start_new_session=True)
    (folder / tp.RECORD).write_text(json.dumps({"pid": p.pid, "epoch": time.time(), "command": command, "env": env,
                                                "log": str(tmp_path / "pre.log")}))
    return folder, p


CMD = [sys.executable, "-c", "import time; time.sleep(60)"]


def test_exact_match_is_adopted(tmp_path):
    folder, p = pre(tmp_path, CMD, {"A": "1"})
    pid, row = tp.take(folder, CMD, {"A": "1"}, tmp_path / "loop.log")
    assert pid == p.pid and row["adopted"] and row["env_diff"] == [] and not row["log_equal"]
    a = tp.Adopted(pid)
    assert a.poll() is None
    a.terminate(); p.wait(10)  # the test is the parent here: reap, then the adopted view reads it gone
    assert a.wait(5) == -1


def test_mismatch_is_stopped_and_cleaned(tmp_path):
    folder, p = pre(tmp_path, CMD, {"A": "1"})
    (folder / "alive.json").write_text('{"state": "ready"}')
    (folder / "cmd" / "0000.json").write_text("{}")
    pid, row = tp.take(folder, CMD, {"A": "2"})
    assert pid is None and row["why"] == "environment differs" and row["env_diff"] == ["A"]
    assert p.wait(10) is not None  # stopped
    assert not (folder / "alive.json").exists() and not list((folder / "cmd").iterdir()) and not (folder / tp.RECORD).exists()
    folder2, p2 = pre(tmp_path / "b", CMD, {})
    pid, row = tp.take(folder2, CMD + ["x"], {})
    assert pid is None and row["why"] == "command differs" and row["command_diff"] == [len(CMD), len(CMD) + 1]
    p2.wait(10)


def test_no_record_and_dead(tmp_path):
    assert tp.take(tmp_path, CMD, {}) == (None, None)
    folder, p = pre(tmp_path, [sys.executable, "-c", "pass"], {})
    p.wait(10)
    pid, row = tp.take(folder, [sys.executable, "-c", "pass"], {})
    assert pid is None and row["why"] == "pre-spawned trainer not alive"


def test_wanted_only_lazy_dual_loop():
    base = {"rl_loop": {"dual": True, "mode": "loop", "mg": {"timeshare": {"lazy_boot": True}}}}
    assert tp.wanted(base)
    for edit in (lambda s: s["rl_loop"].update(dual=False), lambda s: s["rl_loop"]["mg"]["timeshare"].pop("lazy_boot"),
                 lambda s: s["rl_loop"]["mg"].update(lr_probe={"x": 1}), lambda s: s["rl_loop"].update(mode="screen")):
        s = json.loads(json.dumps(base)); edit(s)
        assert not tp.wanted(s)
