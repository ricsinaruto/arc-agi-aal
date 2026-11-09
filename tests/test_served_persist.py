import json, os, time

from arena import served_route as sr
from arena import replay_stub as rs


def rec(folder, rid, n=64):
    rs.served_record(folder, rid, list(range(n)), "policy")


def test_mirror_copies_settled_records_once_and_restore_fills_missing(tmp_path):
    import tarfile

    live, mir = tmp_path / "live", tmp_path / "mirror"
    rec(live, "chatcmpl-a"); rec(live, "chatcmpl-b")
    old = time.time() - 120
    for f in live.iterdir():
        os.utime(f, (old, old))
    rec(live, "chatcmpl-c")  # still being written (fresh)
    seen = {}
    got = sr.mirror(live, mir, seen)
    tars = sorted(mir.glob("pass-*.tar"))
    assert got["files"] == 4 and len(tars) == 1 and not list(mir.glob("*.part"))
    with tarfile.open(tars[0]) as tar:
        assert sorted(n for n in tar.getnames() if n.endswith(".rt")) == ["chatcmpl-a-stub0000.rt", "chatcmpl-b-stub0000.rt"]
    assert sr.mirror(live, mir, seen)["files"] == 0 and len(list(mir.glob("pass-*.tar"))) == 1  # nothing new: no tar
    assert (mir / "layers.json").exists()
    assert abs(tars[0].stat().st_mtime - old) < 1  # the tar's mtime is its newest member's: prune cuts it alike
    fresh = tmp_path / "next"
    rec(fresh, "chatcmpl-z")
    got = sr.restore([mir], fresh)
    assert got["files"] == 4 and got["roots"] == 1
    assert abs((fresh / "chatcmpl-a-stub0000.rt").stat().st_mtime - old) < 1
    store = sr.Store(fresh, scratch=True)
    assert store.record("chatcmpl-a") is not None and store.record("chatcmpl-z") is not None
    assert sr.restore([mir], fresh)["files"] == 0


def test_mirror_tar_pruned_only_when_all_members_old(tmp_path):
    live, mir = tmp_path / "live", tmp_path / "mirror"
    rec(live, "chatcmpl-a")
    t0 = time.time() - 600
    for f in live.iterdir():
        os.utime(f, (t0, t0))
    sr.mirror(live, mir, {})
    rec(live, "chatcmpl-b")
    t1 = time.time() - 120
    for f in live.glob("chatcmpl-b*"):
        os.utime(f, (t1, t1))
    seen = {f.name: f.stat().st_mtime for f in live.glob("chatcmpl-a*")}
    sr.mirror(live, mir, seen)
    assert len(list(mir.glob("pass-*.tar"))) == 2
    got = sr.prune(mir, t0 + 60)  # between the two passes: only the first tar goes
    assert got["files"] == 1 and got["kept"] == 1
    fresh = tmp_path / "fresh"
    sr.restore([mir], fresh)
    assert sorted(p.name for p in fresh.glob("*.rt")) == ["chatcmpl-b-stub0000.rt"]


def test_restore_reads_loose_mirrors_too(tmp_path):
    old, dst = tmp_path / "old", tmp_path / "dst"  # a pre-23-Oct mirror: loose request files
    rec(old, "chatcmpl-l")
    assert sr.restore([old], dst)["files"] == 3 and (dst / "chatcmpl-l-stub0000.rt").exists()


def test_restore_latest_root_wins(tmp_path):
    a, b, dst = tmp_path / "a", tmp_path / "b", tmp_path / "dst"
    rec(a, "chatcmpl-x", 64); rec(b, "chatcmpl-x", 70)
    sr.restore([a, b], dst)
    assert sr.read(dst / "chatcmpl-x-stub0000.rt")["hi"] == 70


def test_fake_trainer_counts_coverage(tmp_path):
    from arena.fake_trainer import Fake

    served = tmp_path / "served"
    rec(served, "chatcmpl-hit")
    roll = []
    for rid in ("chatcmpl-hit", "chatcmpl-miss"):
        p = tmp_path / f"{rid}.jsonl"
        p.write_text(json.dumps({"n": 0, "rid": rid, "usage": {"completion_tokens": 3}}) + "\n")
        roll.append({"rollout": str(p), "served_route": str(served), "name": rid})
    f = Fake(folder=tmp_path)
    for spec in roll:
        f.served_count(spec)
    assert f.served == {"windows": 1, "no_record": 1}
