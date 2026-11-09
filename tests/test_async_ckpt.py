"""HEAL_ASYNC_CKPT (c4's 260 s saves): host copy then a background writer, one save in flight,
progress.json written last and atomically; a save killed mid-write leaves no progress.json, so the preemption
resume (modal_rl1.preempt_retry) takes the previous complete checkpoint, and resume loads the saved state."""
import ast
import json
import os
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
SRC = Path(__file__).resolve().parents[1] / "scripts" / "posttrain"
sys.path.insert(0, str(SRC))
import sys as _sys  # trunk 24 Oct: PTH's trainer copies (scripts/posttrain/pth) shadow RL's for this module
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "posttrain" / "pth"))
for _m in ("rl1_trainer", "flash_qlora_job", "rl1_sparse"):
    _sys.modules.pop(_m, None)
import heal_kd  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def gloo():
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)
    yield


class Rows:
    def __init__(self):
        self.s = {"buf": torch.arange(4.0)}

    def state_dict(self):
        return self.s


def setup(seed=0):
    torch.manual_seed(seed)
    m = torch.nn.Linear(8, 4)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    m(torch.randn(3, 8)).sum().backward()
    opt.step()
    return m, opt


def save(tmp, m, opt, step):
    args = types.SimpleNamespace(student="s")
    heal_kd.checkpoint_async(m, opt, Rows(), args, tmp, 0, step=step, log=None,
                             progress={"step": step, "consumed": ["a"]})


def preempt_retry():
    tree = ast.parse((SRC / "modal_rl1.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "preempt_retry")
    ns = {"json": json, "time": __import__("time")}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal_rl1.py", "exec"), ns)
    return ns["preempt_retry"]


def test_save_resume_and_one_in_flight(tmp_path):
    m, opt = setup()
    save(tmp_path, m, opt, 2)
    first = heal_kd.CKPT_THREAD.get("thread")
    save(tmp_path, m, opt, 4)  # joins the step-2 writer first: one save in flight
    assert first is not None and not first.is_alive()
    heal_kd.wait_checkpoint()
    ck = tmp_path / "ckpt-4"
    assert (ck / "progress.json").is_file() and not (ck / "progress.json.tmp").exists()
    assert not (tmp_path / "ckpt-2").exists()  # older removed only after the new one is complete
    assert json.loads((ck / "progress.json").read_text())["sharded"] is True
    m2, _ = setup(seed=1)
    assert heal_kd.load_local_params(m2, ck, 0) == 2
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a.float(), b.float())
    st = torch.load(ck / "optim-rank0.pt", weights_only=False)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    opt2.load_state_dict(st["adam"])
    assert all(torch.equal(opt.state_dict()["state"][i]["exp_avg"], opt2.state_dict()["state"][i]["exp_avg"])
               for i in opt.state_dict()["state"])


def test_killed_save_is_never_resumed(tmp_path, monkeypatch):
    out = tmp_path / "run"
    (out / "train").mkdir(parents=True)
    m, opt = setup()
    save(out / "train", m, opt, 6)
    heal_kd.wait_checkpoint()
    real = torch.save

    def dies(obj, path, *a, **k):  # the writer dies mid-save (a preemption: the params shard never lands)
        if "params-rank" in str(path):
            raise OSError("killed")
        return real(obj, path, *a, **k)

    monkeypatch.setattr(heal_kd.torch, "save", dies)
    save(out / "train", m, opt, 12)
    with pytest.raises(RuntimeError, match="async checkpoint failed"):
        heal_kd.wait_checkpoint()
    monkeypatch.setattr(heal_kd.torch, "save", real)
    assert not (out / "train/ckpt-12/progress.json").exists() and (out / "train/ckpt-6/progress.json").is_file()
    got = preempt_retry()(out)  # Modal re-runs the input: the newest COMPLETE checkpoint
    assert got["step0"] == 6 and got["resume"].endswith("train/ckpt-6")


def _files(folder):
    return {f: torch.load(folder / f, weights_only=False) for f in ("optim-rank0.pt", "params-rank0.pt")}


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def test_reused_buffers_save_what_the_plain_copy_saves(tmp_path, monkeypatch):
    """HEAL_CKPT_PINNED=1 (reused host buffers; pinned on CUDA): each checkpoint's files equal the plain async
    path's, the buffers are refilled in place by the next one, and the earlier files keep their own step's state."""
    m, opt = setup()
    save(tmp_path / "plain2", m, opt, 2)
    heal_kd.wait_checkpoint()
    monkeypatch.setenv("HEAL_CKPT_PINNED", "1")
    monkeypatch.setenv("HEAL_CKPT_PIN_HEADROOM", "0")
    heal_kd.CKPT_BUFS.clear()
    heal_kd.CKPT_PIN.clear()
    save(tmp_path / "pin2", m, opt, 2)
    heal_kd.wait_checkpoint()
    ptrs = {k: v.data_ptr() for k, v in heal_kd.CKPT_BUFS.items()}
    m(torch.randn(3, 8)).sum().backward()
    opt.step()
    save(tmp_path / "pin4", m, opt, 4)
    heal_kd.wait_checkpoint()
    assert ptrs and {k: v.data_ptr() for k, v in heal_kd.CKPT_BUFS.items()} == ptrs  # reused, not reallocated
    monkeypatch.delenv("HEAL_CKPT_PINNED")
    save(tmp_path / "plain4", m, opt, 4)
    heal_kd.wait_checkpoint()
    p2, q2 = _files(tmp_path / "pin2/ckpt-2"), _files(tmp_path / "plain2/ckpt-2")
    p4, q4 = _files(tmp_path / "pin4/ckpt-4"), _files(tmp_path / "plain4/ckpt-4")
    assert _same(p2, q2) and _same(p4, q4)
    assert not _same(p2["params-rank0.pt"], p4["params-rank0.pt"])  # step 2's files kept step 2's weights


def test_pinning_needs_host_headroom(tmp_path, monkeypatch):
    """The first pinned checkpoint pins only when >= HEAL_CKPT_PIN_HEADROOM of host RAM stays free after every
    rank's buffers; otherwise the plain copy for the whole run (and the files are the same)."""
    m, opt = setup()
    monkeypatch.setenv("HEAL_CKPT_PINNED", "1")
    for headroom, want in (("0.0", True), ("1.0", False)):
        heal_kd.CKPT_PIN.clear()
        heal_kd.CKPT_BUFS.clear()
        monkeypatch.setenv("HEAL_CKPT_PIN_HEADROOM", headroom)
        save(tmp_path / headroom, m, opt, 2)
        heal_kd.wait_checkpoint()
        assert heal_kd.CKPT_PIN["ok"] is want and bool(heal_kd.CKPT_BUFS) is want
        assert (tmp_path / headroom / "ckpt-2/progress.json").is_file()
    assert _same(_files(tmp_path / "0.0/ckpt-2"), _files(tmp_path / "1.0/ckpt-2"))
    heal_kd.CKPT_PIN.clear()


def test_checkpoint_skips_when_the_host_is_low(tmp_path, monkeypatch):
    """c7b 24 Oct: below 1.2 x need + 10% of host RAM, every rank skips the copy (event ckpt_skip) instead of
    thrashing the container; above it, the checkpoint is written."""
    m, opt = setup()
    monkeypatch.delenv("HEAL_CKPT_PINNED", raising=False)
    for avail, want in ((5.0, False), (500.0, True)):
        monkeypatch.setattr(heal_kd, "host_mem_gib", lambda a=avail: (a, 100.0))
        (tmp_path / str(avail)).mkdir()
        heal_kd.checkpoint_async(m, opt, Rows(), types.SimpleNamespace(student="s"), tmp_path / str(avail), 0,
                                 step=1, log=tmp_path / str(avail), progress={"step": 1, "consumed": []})
        heal_kd.wait_checkpoint()
        assert (tmp_path / str(avail) / "ckpt-1/progress.json").is_file() is want
        skips = [json.loads(x) for x in (tmp_path / str(avail) / "events.jsonl").read_text().splitlines()
                 if '"ckpt_skip"' in x] if (tmp_path / str(avail) / "events.jsonl").exists() else []
        assert bool(skips) is not want and all(x["floor_gib"] >= 10.0 for x in skips)


class Experts(torch.nn.Module):
    """A fused expert tensor (dim 0 = experts) beside a dense one."""

    def __init__(self):
        super().__init__()
        self.experts = torch.nn.Parameter(torch.randn(6, 4, 3))
        self.dense = torch.nn.Parameter(torch.randn(5))

    def forward(self, x):
        return (self.experts.sum(0) @ x).sum() + self.dense.sum()


def test_rows_only_expert_checkpoint_round_trip(tmp_path, monkeypatch):
    """c7b (24 Oct): a checkpoint keeps only RowAdam's selected experts of a fused tensor (with their global ids);
    a restart from the starting weights plus the checkpoint equals the trained weights; the reload check's join and
    the reshard move them by global id."""
    import heal_reload_check
    import heal_reshard

    monkeypatch.delenv("HEAL_CKPT_PINNED", raising=False)
    torch.manual_seed(0)
    m = Experts()
    start = {k: v.detach().clone() for k, v in m.state_dict().items()}
    rows = heal_kd.RowAdam([(m.experts, [1, 4])])
    opt = torch.optim.AdamW([m.dense], lr=1e-2)
    for _ in range(2):
        m(torch.randn(3)).backward()
        rows.mask_grads()
        rows.step(1e-2)
        opt.step()
        rows.zero_grad()
        opt.zero_grad()
    assert not torch.equal(m.experts[1], start["experts"][1]) and torch.equal(m.experts[0], start["experts"][0])
    args = types.SimpleNamespace(student="s")
    heal_kd.checkpoint_async(m, opt, rows, args, tmp_path, 0, step=2, log=None, progress={"step": 2, "consumed": []})
    heal_kd.wait_checkpoint()
    ck = tmp_path / "ckpt-2"
    f = torch.load(ck / "params-rank0.pt", weights_only=True)
    assert f["experts"].shape == (2, 4, 3) and f["experts::experts"].tolist() == [1, 4]
    m2 = Experts()
    m2.load_state_dict(start)
    assert heal_kd.load_local_params(m2, ck, 0) == 2
    assert all(torch.equal(a, b) for a, b in zip(m.parameters(), m2.parameters()))
    joined = heal_kd.join_params([f])
    assert sorted(joined["experts"]) == [1, 4] and torch.equal(joined["experts"][4], m.experts[4].detach())
    # reshard to 2 ranks (6 experts: 0-2 | 3-5): expert 1 stays on rank 0, expert 4 moves to rank 1 as local id 1
    (ck / "optim-rank0.pt").exists()
    dst = tmp_path / "w2"
    heal_reshard.reshard(ck, dst, 2, 6)
    a, b = (torch.load(dst / f"params-rank{q}.pt", weights_only=True) for q in range(2))
    assert a["experts::experts"].tolist() == [1] and b["experts::experts"].tolist() == [4]
    assert torch.equal(b["experts"][0], m.experts[4].detach())
    monkeypatch.setenv("HEAL_CKPT_ALL_EXPERTS", "1")  # the old full format stays available
    assert heal_kd.local_params(m, rows=rows)["experts"].shape == (6, 4, 3)
    assert heal_reload_check is not None
