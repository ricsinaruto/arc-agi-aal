"""Trainer ops of the single-card start-up cuts (MFU-2 23 Oct): base_export / base_import (cut c) and prepin (cut d)."""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
import rl1_trainer  # noqa: E402

T = rl1_trainer.Trainer


def ref(seed):
    g = torch.Generator().manual_seed(seed)
    return {"actor": torch.randn(7, generator=g), "valid": torch.rand(7, generator=g) > 0.3,
            "ids": torch.randint(0, 50, (7, 20), generator=g), "logp": torch.randn(7, 21, generator=g)}


def test_base_export_import_round_trip_sets_base_previous_and_last(tmp_path):
    a = SimpleNamespace(references={"p0": {"base": ref(0), "previous": ref(9)}, "p1": {"base": ref(1)}, "x": {}})
    path = tmp_path / "c" / "k.pt"
    out = T.op_base_export(a, {"path": str(path), "key": "k", "names": ["p0", "p1"]})
    assert out["windows"] == 2 and path.is_file() and not list(path.parent.glob("*.tmp"))
    b = SimpleNamespace(references={"p0": {"previous": ref(5)}})
    got = T.op_base_import(b, {"path": str(path), "key": "k", "names": ["p1", "p0"]})
    assert got["names"] == ["p0", "p1"]
    for name in ("p0", "p1"):
        for slot in ("base", "previous", "last"):
            for k, v in a.references[name]["base"].items():
                assert torch.equal(b.references[name][slot][k], v)


def test_base_import_keeps_references_on_the_host_like_score(tmp_path, monkeypatch):
    a = SimpleNamespace(references={"p0": {"base": ref(0)}})
    path = tmp_path / "k.pt"
    T.op_base_export(a, {"path": str(path), "key": "k", "names": ["p0"]})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # v6j2: imported refs went to cuda:0
    b = SimpleNamespace(references={})
    T.op_base_import(b, {"path": str(path), "key": "k", "names": ["p0"]})
    assert all(v.device.type == "cpu" for slot in ("base", "previous", "last") for v in b.references["p0"][slot].values())


def test_base_drop_removes_only_the_named_references():
    a = SimpleNamespace(references={"p0": {"base": ref(0)}, "p1": {"base": ref(1)}})
    assert T.op_base_drop(a, {"names": ["p0", "p9"]}) == {"dropped": 1}
    assert set(a.references) == {"p1"}


def test_base_import_refuses_another_key_or_window_set(tmp_path):
    a = SimpleNamespace(references={"p0": {"base": ref(0)}})
    path = tmp_path / "k.pt"
    T.op_base_export(a, {"path": str(path), "key": "k", "names": ["p0"]})
    with pytest.raises(ValueError):
        T.op_base_import(SimpleNamespace(references={}), {"path": str(path), "key": "other", "names": ["p0"]})
    with pytest.raises(ValueError):
        T.op_base_import(SimpleNamespace(references={}), {"path": str(path), "key": "k", "names": ["p0", "p1"]})
    with pytest.raises(ValueError):  # export of a window without a base reference
        T.op_base_export(a, {"path": str(path), "key": "k", "names": ["p0", "p9"]})


def test_prepin_allocates_missing_buffers_on_a_thread_and_park_waits(monkeypatch):
    real = torch.empty
    started, release = threading.Event(), threading.Event()

    def empty(*a, pin_memory=False, **k):  # CPU build: no pinned memory; the thread waits so the join is observable
        if pin_memory:
            started.set()
            release.wait(5)
        return real(*a, **k)

    monkeypatch.setattr(torch, "empty", empty)

    class Cudaish:  # a CUDA tensor's shape, dtype and is_cuda
        def __init__(self, t):
            self.is_cuda, self.shape, self.dtype = True, t.shape, t.dtype

    p, base = SimpleNamespace(data=Cudaish(torch.zeros(3, 4))), Cudaish(torch.zeros(5, dtype=torch.bfloat16))
    held_key = ("param", "w")
    me = SimpleNamespace(pinned={("param", "already"): real(1)}, held=lambda: [(held_key, p, None)],
                         base_tensors=lambda: [base])
    out = T.op_prepin(me, {"base": True})
    assert out["started"] and out["tensors"] == 2
    assert started.wait(5)
    assert T.op_prepin(me) == {"already": True}
    release.set()
    done = T.prepin_join(me)
    assert done["error"] is None and done["seconds"] is not None
    assert me.pinned[held_key].shape == (3, 4) and me.pinned[("base", id(base))].dtype == torch.bfloat16


# --- cut (b): host load with the cold load's layout record ----------------------------------------------------------

class Tiny(torch.nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.a = torch.nn.Linear(6, 4)
        self.register_buffer("b", torch.arange(5.0))
        self.scale = torch.rand(3, generator=g)  # a plain tensor attribute (a quantised weight's scales)
        with torch.no_grad():
            self.a.weight.copy_(torch.randn(4, 6, generator=g))
            self.a.bias.copy_(torch.randn(4, generator=g))


class Rec:
    def __init__(self):
        self.events = []

    def event(self, name, **kw):
        self.events.append((name, kw))


def fake_load(monkeypatch, models):
    calls = []

    def load(rec, model_dir, out, sequence_len):
        calls.append(dict(rl1_trainer.fq.DEVICE_MAP))
        m = models.pop(0)
        if isinstance(m, Exception):
            raise m
        return m, None, {}

    monkeypatch.setattr(rl1_trainer.fq, "load", load)
    return calls


@pytest.fixture(autouse=True)
def device_map():
    yield
    rl1_trainer.fq.DEVICE_MAP = {"": 0}


def test_cold_load_writes_the_layout_record_once(tmp_path, monkeypatch):
    record = tmp_path / "layout.json"
    monkeypatch.setenv("RL1_LAYOUT_RECORD", str(record))
    monkeypatch.delenv("RL1_HOST_LOAD", raising=False)
    calls = fake_load(monkeypatch, [Tiny(), Tiny(1)])
    _, _, loaded = rl1_trainer.load_model(Rec(), "m", tmp_path)
    assert loaded["layout_record"] == str(record) and calls == [{"": 0}]
    first = record.read_text()
    rl1_trainer.load_model(Rec(), "m", tmp_path)
    assert record.read_text() == first  # an existing record is kept


def test_host_load_passes_on_the_same_layout_and_restores_threads(tmp_path, monkeypatch):
    record = tmp_path / "layout.json"
    record.write_text(rl1_trainer.json.dumps(rl1_trainer.layout_digest(Tiny())))
    monkeypatch.setenv("RL1_LAYOUT_RECORD", str(record))
    monkeypatch.setenv("RL1_HOST_LOAD", "1")
    monkeypatch.setenv("RL1_HOST_LOAD_THREADS", "2")
    threads = torch.get_num_threads()
    seen = {}

    def load(rec, model_dir, out, sequence_len):
        seen.update(map=dict(rl1_trainer.fq.DEVICE_MAP), threads=torch.get_num_threads())
        return Tiny(), None, {}

    monkeypatch.setattr(rl1_trainer.fq, "load", load)
    rec = Rec()
    _, _, loaded = rl1_trainer.load_model(rec, "m", tmp_path)
    assert seen == {"map": {"": "cpu"}, "threads": 2} and torch.get_num_threads() == threads
    assert not loaded["host_load"].get("fallback") and rec.events[0][0] == "host_load"


@pytest.mark.parametrize("change", ["value", "attr", "forward", "missing_record", "load_error"])
def test_host_load_falls_back_to_the_cold_load(tmp_path, monkeypatch, change):
    record = tmp_path / "layout.json"
    if change != "missing_record":
        record.write_text(rl1_trainer.json.dumps(rl1_trainer.layout_digest(Tiny())))
    monkeypatch.setenv("RL1_LAYOUT_RECORD", str(record))
    monkeypatch.setenv("RL1_HOST_LOAD", "1")
    host = Tiny()
    if change == "value":
        with torch.no_grad():
            host.a.weight.view(-1)[0] += 1.0
    elif change == "attr":
        host.scale = host.scale + 0.5
    elif change == "forward":
        host.a.forward = lambda x: x
    first_model = RuntimeError("cpu load failed") if change == "load_error" else host
    calls = fake_load(monkeypatch, [first_model, Tiny()])
    rec = Rec()
    _, _, loaded = rl1_trainer.load_model(rec, "m", tmp_path)
    expect = [{"": 0}] if change == "missing_record" else [{"": "cpu"}, {"": 0}]
    assert calls == expect and loaded["host_load"]["fallback"] is True and loaded["host_load"]["error"]
    assert rl1_trainer.fq.DEVICE_MAP == {"": 0}


def test_move_to_layout_moves_buffers_and_attributes_and_leaves_matching_tensors(monkeypatch):
    m = Tiny()
    want = rl1_trainer.layout_digest(m)
    for t in want["tensors"]:  # parameters stay (a CPU build has no second device for .data); the rest to meta
        if t[1] != "param":
            t[4] = "meta"
    moved = rl1_trainer.move_to_layout(m, want)
    assert m.a.weight.device.type == "cpu" and m.b.device.type == "meta" and m.scale.device.type == "meta"
    assert moved == sum(x.numel() * x.element_size() for x in (Tiny().b, Tiny().scale))
