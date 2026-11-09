"""runops.ckpt: bounded-ring streaming save, tree round trip, checksums, staged copy, killed save, torch-zip fallback;
runops.hostguard: readings and the 1.2x check."""
import gc
import json
import os

import pytest

torch = pytest.importorskip("torch")

from arc_infra.runops import ckpt  # noqa: E402
from arc_infra.runops.hostguard import HostGuard  # noqa: E402
from arc_infra.runops.metrics import MetricsWriter, read_rows  # noqa: E402


def state():
    torch.manual_seed(0)
    m = torch.nn.Linear(37, 11)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    m(torch.randn(3, 37)).sum().backward()
    opt.step()
    big = torch.randn(300_000)  # > one 1 MiB slot: crosses chunks
    return {"params": {k: v.detach().clone() for k, v in m.state_dict().items()}, "optim": opt.state_dict(),
            "rows": {"m": torch.randn(5, 3).to(torch.bfloat16), "ids": torch.arange(7), "mask": torch.tensor([True, False]),
                     "odd": torch.arange(13, dtype=torch.uint8), "empty": torch.empty(0, 4), "scalar": torch.tensor(2.5)},
            "big": big, "view": big[1:200:3], "shape": torch.Size([2, 3]), "meta": {"step": 12, 3: "int key", "t": (1, 2.0)},
            "nan": float("nan")}


def small_ring():
    return ckpt.Ring(slots=2, slot_mib=1)


def test_round_trip_exact_and_checksums(tmp_path):
    s = state()
    h = ckpt.save(s, tmp_path / "a.pt", ring=small_ring())
    assert h.nbytes > 1 << 20 and (tmp_path / "a.pt").is_file() and not (tmp_path / "a.pt.tmp").exists()
    assert ckpt.compare(tmp_path / "a.pt", s) == []
    assert ckpt.verify(tmp_path / "a.pt") == []
    back = ckpt.load(tmp_path / "a.pt")
    assert back["meta"] == {"step": 12, 3: "int key", "t": (1, 2.0)} and back["shape"] == torch.Size([2, 3])
    assert back["optim"]["state"][0]["step"].item() == s["optim"]["state"][0]["step"].item()
    assert back["optim"]["param_groups"][0]["lr"] == 1e-3
    assert torch.equal(back["view"], s["view"])
    gc.collect()  # the mmap stays alive with its tensors
    assert torch.equal(back["big"], s["big"])
    assert ckpt.load(tmp_path / "a.pt", mmap_=False)["rows"]["m"].dtype == torch.bfloat16


def test_safetensors_reads_the_tensors(tmp_path):
    from safetensors import safe_open

    s = {"w": torch.randn(4, 4), "b": torch.arange(3)}
    ckpt.save(s, tmp_path / "s.safetensors", ring=small_ring())
    with safe_open(str(tmp_path / "s.safetensors"), "pt") as f:
        got = {k: f.get_tensor(k) for k in f.keys()}
    assert sorted(torch.equal(v, s["w"]) or torch.equal(v, s["b"]) for v in got.values()) == [True, True]


def test_corruption_is_found(tmp_path):
    s = state()
    ckpt.save(s, tmp_path / "a.pt", ring=small_ring())
    with open(tmp_path / "a.pt", "r+b") as f:
        f.seek(-100, os.SEEK_END)
        f.write(b"\x01")
    assert ckpt.verify(tmp_path / "a.pt") != []
    assert ckpt.compare(tmp_path / "a.pt", s) != []


def test_staged_copy_and_after(tmp_path):
    calls = []
    h = ckpt.save(state(), tmp_path / "vol" / "a.pt", stage_dir=tmp_path / "local", ring=small_ring(),
                  after=lambda: calls.append(1))
    h.wait()
    assert (tmp_path / "vol" / "a.pt").is_file() and not (tmp_path / "local" / "a.pt").exists()
    assert calls == [1] and ckpt.verify(tmp_path / "vol" / "a.pt") == []


def test_killed_save_leaves_no_file(tmp_path, monkeypatch):
    s = state()
    real = ckpt._Sum.add
    calls = {"n": 0}

    def dying(self, chunk, start):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("killed mid-save")
        return real(self, chunk, start)

    monkeypatch.setattr(ckpt._Sum, "add", dying)
    with pytest.raises(RuntimeError):
        ckpt.save(s, tmp_path / "a.pt", ring=small_ring())
    assert not (tmp_path / "a.pt").exists()  # only the .tmp: readers never see a partial file


def test_ring_is_reused_and_bounded(tmp_path):
    r = small_ring()
    ckpt.save(state(), tmp_path / "a.pt", ring=r)
    first = sorted(id(x) for x in list(r.free.queue))
    ckpt.save(state(), tmp_path / "b.pt", ring=r)
    assert sorted(id(x) for x in list(r.free.queue)) == first and r.host_gib == 2 / 1024


def test_torch_zip_fallback(tmp_path):
    torch.save({"x": torch.arange(3)}, tmp_path / "z.pt")
    assert not ckpt.is_runops(tmp_path / "z.pt")
    assert torch.equal(ckpt.load(tmp_path / "z.pt")["x"], torch.arange(3))


def test_hostguard(tmp_path):
    w = MetricsWriter(tmp_path / "m.jsonl")
    readings = iter([{"total_gib": 1024.0, "avail_gib": 600.0}, {"total_gib": 1024.0, "avail_gib": 144.2},
                     {"total_gib": 1024.0, "avail_gib": 600.0}])
    logs = []
    g = HostGuard(w, sample=lambda: next(readings), log=logs.append)
    g.read()
    assert not g.check(443, "ckpt-1 host copy")  # c7b: 144.2 < 1.2 x 443
    assert g.check(184 / 3, "ckpt-1 rank shard")
    rows = read_rows(tmp_path / "m.jsonl")
    assert any("ckpt-1 host copy needs 443.0 GiB" in (r.get("alert") or "") for r in rows)
    assert rows[1]["m"]["infra.next_alloc_gib"] == 443.0 and g.min_avail == 144.2
    assert logs and logs[0].startswith("RUNOPS HOST-RAM")


def test_header_is_json_with_tree(tmp_path):
    ckpt.save({"a": torch.ones(2)}, tmp_path / "h.pt", ring=small_ring())
    with open(tmp_path / "h.pt", "rb") as f:
        n, meta, entries = ckpt._read_header(f)
    assert meta["format"] == ckpt.FORMAT and json.loads(meta["tree"]) == {"a": {"__t__": "t0"}}
    assert n % 8 == 0 and list(entries) == ["t0"]
