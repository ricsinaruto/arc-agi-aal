"""Single-flight volume reads: a second reader of the same path skips while the first holds it."""
from arc_infra.runops import volfetch


def test_single_flight_skips_a_second_reader_of_the_same_path(tmp_path, monkeypatch):
    monkeypatch.setattr(volfetch, "LOCKS", tmp_path)
    with volfetch.single_flight("v", "a/feed.jsonl") as first:
        with volfetch.single_flight("v", "a/feed.jsonl") as second, volfetch.single_flight("v", "b") as other:
            assert first and not second and other  # same path skips; another path runs
    with volfetch.single_flight("v", "a/feed.jsonl") as again:
        assert again  # released at exit


def test_run_kills_the_whole_group_on_a_timeout(tmp_path, monkeypatch):
    """A fake slow 'modal' that spawns a child (as `uv run modal` does): after the timeout the whole group is gone and
    the path's lock is free again."""
    import os
    import time

    monkeypatch.setattr(volfetch, "LOCKS", tmp_path)
    pidfile = tmp_path / "child.pid"
    fake = tmp_path / "modal"
    fake.write_text(f"#!/bin/sh\nsleep 60 &\necho $! > {pidfile}\nwait\n")
    fake.chmod(0o755)
    with volfetch.single_flight("v", "p") as mine:
        began = time.time()
        assert mine and volfetch.run([str(fake), "volume", "get"], timeout=2) is None
        assert time.time() - began < 10
    child = int(pidfile.read_text())
    for _ in range(50):  # reaped by init after the kill; give it a moment
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        raise AssertionError(f"child {child} outlived the timeout")
    with volfetch.single_flight("v", "p") as again:
        assert again
    assert volfetch.run(["/bin/echo", "ok"]) == "ok\n" and volfetch.run(["/bin/false"]) is None
    assert volfetch.run(["/nonexistent/modal"]) is None


def _fake_modal(tmp_path, body, calls):
    """A fake `modal volume get --force VOL PATH DEST` that writes ``body`` to DEST and logs each call."""
    fake = tmp_path / "modal"
    fake.write_text(f'#!/bin/sh\necho "$@" >> {calls}\nprintf %s "{body}" > ""\n')
    fake.chmod(0o755)
    return str(fake)


def test_one_fetcher_fills_the_mirror_and_readers_take_it_while_fresh(tmp_path, monkeypatch):
    """28 Oct: watch, export and mon each fetched live/sampler.jsonl per pass (30-70 s a get). export_rl's get fills the
    machine-wide mirror; a reader within max_age reads it with no fetch; a stale mirror is refetched once."""
    import os
    import time

    monkeypatch.setattr(volfetch, "LOCKS", tmp_path / "locks")
    calls = tmp_path / "calls.txt"
    modal = _fake_modal(tmp_path, "row1", calls)
    assert volfetch.read("v", "r/live/sampler.jsonl") is None
    assert volfetch.get("v", "r/live/sampler.jsonl", modal=modal)
    assert volfetch.read("v", "r/live/sampler.jsonl", max_age=300) == "row1"
    assert volfetch.text("v", "r/live/sampler.jsonl", modal=modal) == "row1"
    assert len(calls.read_text().splitlines()) == 1  # the fresh mirror was read, not fetched
    assert not list(volfetch.mirror("v", "r/live").glob("*.part"))
    old = time.time() - 1000
    os.utime(volfetch.mirror("v", "r/live/sampler.jsonl"), (old, old))
    assert volfetch.read("v", "r/live/sampler.jsonl", max_age=300) is None
    modal = _fake_modal(tmp_path, "row2", calls)
    assert volfetch.text("v", "r/live/sampler.jsonl", modal=modal) == "row2"
    assert len(calls.read_text().splitlines()) == 2
    with volfetch.single_flight("v", "r/live/sampler.jsonl"):  # another process fetching: last good copy, no fetch
        os.utime(volfetch.mirror("v", "r/live/sampler.jsonl"), (old, old))
        assert not volfetch.get("v", "r/live/sampler.jsonl", modal=modal)
        assert volfetch.text("v", "r/live/sampler.jsonl", modal=modal) == "row2"
    assert len(calls.read_text().splitlines()) == 2
    assert not volfetch.get("v", "missing", modal="/bin/false") and volfetch.read("v", "missing") is None


def test_copy_serves_the_fresh_mirror_and_the_last_good_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(volfetch, "LOCKS", tmp_path / "locks")
    calls = tmp_path / "calls.txt"
    modal = _fake_modal(tmp_path, "p1", calls)
    assert volfetch.copy("v", "r/progress.json", tmp_path / "a/out.json", modal=modal)
    assert (tmp_path / "a/out.json").read_text() == "p1"
    assert volfetch.copy("v", "r/progress.json", tmp_path / "b.json", modal="/bin/false")  # fresh: no fetch
    assert len(calls.read_text().splitlines()) == 1
    assert not volfetch.copy("v", "nothing", tmp_path / "c.json", modal="/bin/false")


def test_exec_read_rejects_a_short_stream(tmp_path, monkeypatch):
    """28 Oct: `modal container exec ... cat` returned the first 8,192 bytes; exec_read checks the size it printed."""
    fake = tmp_path / "modal"
    body = "x" * 20000
    fake.write_text("#!/bin/sh\nif [ \"$2\" = list ]; then echo '[{\"app_id\": \"ap-1\", \"container_id\": \"ta-1\"}]'; "
                    f"else printf '20000\\n'; printf '%s' \"$(head -c ${{SHORT:-20000}} {tmp_path}/body)\"; fi\n")
    fake.chmod(0o755)
    (tmp_path / "body").write_text(body)
    assert volfetch.exec_read("ap-1", "validation/r/x", modal=str(fake)) == body
    monkeypatch.setenv("SHORT", "8192")
    assert volfetch.exec_read("ap-1", "validation/r/x", modal=str(fake)) is None


def test_exec_app_from_running_record(tmp_path, monkeypatch):
    """28 Oct: a running record's current attempt folder is read inside its app's container (the volume view lagged 9 min)."""
    from arc_infra.runops import records as R
    from arc_infra.runops import volfetch as vf

    monkeypatch.setattr(vf, "EXEC", tmp_path / "exec.json")
    recs = {"r1": {"run": "r1", "state": "running", "attempt": 2, "attempts": [{"n": 1}, {"n": 2}],
                   "ledger": [{"row": {"app_id": "ap-old"}}, {"row": {"app_id": "ap-new"}}]},
            "r2": {"run": "r2", "state": "ended", "ledger": [{"row": {"app_id": "ap-x"}}]}}
    monkeypatch.setattr(R, "get", lambda run, root=None: recs.get(run))
    assert vf.exec_app("arc-prolong-runs", "validation/r1-attempt2/live/sampler.jsonl") == "ap-new"
    assert vf.exec_app("arc-prolong-runs", "validation/r1/live/sampler.jsonl") is None  # attempt 1's folder: not live
    assert vf.exec_app("arc-prolong-runs", "validation/r2/live/x") is None
    assert vf.exec_app("arc-prolong-assets", "validation/r1-attempt2/live/x") is None
