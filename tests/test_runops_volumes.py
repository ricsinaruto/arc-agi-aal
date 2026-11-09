"""Per-run volumes: the run's own volume at /runs, other runs' outputs read-only at /in/<volume>."""
from arc_infra.runops import records as R
from arc_infra.runops import volumes as V


def _recs(monkeypatch, table):
    monkeypatch.setattr(R, "get", lambda run, root=None: table.get(run))


def test_own_volume_and_read_only_inputs(monkeypatch):
    _recs(monkeypatch, {"rl1-loop-seq-real-v12": {"volume": "arc-run-rl1-loop-seq-real-v12"},
                        "rl1-screen-x": {"volume": "arc-run-rl1-screen-x"}})
    ins = ["/runs/validation/rl1-screen-seed-v2/arena/trials/rl-loop/rl", "/runs/validation/rl1-screen-x-attempt2/partial",
           "/runs/validation/rl1-loop-seq-real-v12/state/s0002", "/runs/compile-cache/sft-shared", "/tmp/local"]
    mounts, path = V.plan("rl1-loop-seq-real-v12", ins)
    assert mounts == {"/runs": "arc-run-rl1-loop-seq-real-v12", "/cache": "arc-cache",
                      "/in/arc-prolong-runs": "arc-prolong-runs", "/in/arc-run-rl1-screen-x": "arc-run-rl1-screen-x"}
    assert path(ins[0]) == "/in/arc-prolong-runs/validation/rl1-screen-seed-v2/arena/trials/rl-loop/rl"
    assert path(ins[1]) == "/in/arc-run-rl1-screen-x/validation/rl1-screen-x-attempt2/partial"
    assert path(ins[2]) == ins[2] and path(ins[4]) == "/tmp/local"
    assert path(ins[3]) == "/cache/compile-cache/sft-shared" and V.cache_root(mounts) == "/cache"
    env = V.rewrite_env({"ARC_BOOT_CACHE_IN": "/runs/boot-cache/rd", "N": 3,
                         "ARC_RL1_RESUME": "/runs/validation/rl1-screen-x-attempt2/partial,/runs/validation/rl1-loop-seq-real-v12/s"}, path)
    assert env == {"ARC_BOOT_CACHE_IN": "/cache/boot-cache/rd", "N": 3,
                   "ARC_RL1_RESUME": "/in/arc-run-rl1-screen-x/validation/rl1-screen-x-attempt2/partial,/runs/validation/rl1-loop-seq-real-v12/s"}


def test_old_run_on_shared_volume_is_unchanged(monkeypatch):
    _recs(monkeypatch, {})
    ins = ["/runs/validation/rl1-screen-seed-v2/arena", "/runs/compile-cache/x"]
    mounts, path = V.plan("rl1-loop-old", ins)
    assert mounts == {"/runs": "arc-prolong-runs"} and [path(p) for p in ins] == ins and V.cache_root(mounts) == "/runs"


def test_modal_mounts_create_per_run_v2_and_read_only_inputs(monkeypatch):
    import sys, types
    calls = []

    class Vol:
        def __init__(self, name, **kw):
            self.name, self.kw, self.ro = name, kw, False

        @classmethod
        def from_name(cls, name, **kw):
            calls.append((name, kw))
            return cls(name, **kw)

        def read_only(self):
            self.ro = True
            return self

    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(Volume=Vol))
    out = V.modal_mounts({"/runs": "arc-run-x", "/cache": "arc-cache", "/in/arc-prolong-runs": "arc-prolong-runs"})
    assert out["/runs"].kw == {"create_if_missing": True, "version": 2} and not out["/runs"].ro
    assert out["/cache"].kw == {"create_if_missing": True, "version": 2} and not out["/cache"].ro
    assert out["/in/arc-prolong-runs"].ro and out["/in/arc-prolong-runs"].kw == {}
    assert V.modal_mounts({"/runs": "arc-prolong-runs"})["/runs"].kw == {}


def test_tools_route_to_the_records_volume(monkeypatch):
    import importlib.util
    import pathlib
    from arc_infra.runops import launchcfg, volfetch
    _recs(monkeypatch, {"rl1-x": {"volume": "arc-run-rl1-x"}})
    assert volfetch.route("arc-prolong-runs", "validation/rl1-x") == "arc-run-rl1-x"
    assert volfetch.route("arc-prolong-runs", "validation/rl1-x-attempt3/arena") == "arc-run-rl1-x"
    assert volfetch.route("arc-prolong-runs", "validation/rl1-old/arena") == "arc-prolong-runs"
    seen = []
    monkeypatch.setattr(launchcfg.subprocess, "run", lambda cmd, **kw: seen.append(cmd) or type("R", (), {
        "returncode": 0, "stdout": '[{"Type": "dir", "Size": "1"}]', "stderr": ""})())
    assert launchcfg.check_data(["arc-prolong-runs:validation/rl1-x/arena"]) == []
    assert seen[0][4] == "arc-run-rl1-x"
    p = pathlib.Path(__file__).resolve().parents[1] / "scripts/arena/request_stop.py"
    spec = importlib.util.spec_from_file_location("request_stop", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.volume_for(m.remote_path("rl1-x")) == "arc-run-rl1-x"
