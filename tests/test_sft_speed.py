"""trunk/sft speed paths on CPU with the tiny student: the async checkpoint, the data feed and the warm start leave
every step and checkpoint byte equal; parallel reads load the same weights; the optimizer state round-trips; the
selection cache is bitwise and the selected attention matches the model's; the GDN weight-grad sums match autograd."""
import dataclasses
import json
import math
import sys
import threading
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts/sft"), str(ROOT / "tests/golden"), str(ROOT / "src"), str(ROOT / "tests")]
import common  # noqa: E402
import mtp  # noqa: E402
import train  # noqa: E402
from test_sft_train import data, opt  # noqa: E402

SLOW_OFF = dict(async_ckpt=False, prefetch=0, vision_ahead=False, warm_start=False, fast_load=False,
                ple_table=False)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sftspeed")
    common.tiny_student_on_disk(tmp / "student")
    index = {}
    data(tmp / "data", "train", 4, 1, index)
    data(tmp / "data", "heldout", 2, 2, index)
    return tmp


def cfg(tmp, out, **kw):
    return train.Config(student=str(tmp / "student"), data=str(tmp / "data"), out=str(out), lr=1e-3, warmup=1, rewarm=1,
                        total_steps=3, batch_reply_tokens=150, window=128, prefix_chunk=32, loss_chunk=48,
                        ckpt_steps=2, eval_steps=3, eval_examples=2, fast_attention=False, **kw)


def rows(out, kind):
    return [r for r in map(json.loads, (out / "events.jsonl").read_text().splitlines()) if r.get("event") == kind]


def test_speed_paths_exact(setup):
    """Speed paths on (default) vs all off: the same steps, evals and checkpoint bytes."""
    tmp = setup
    on, off = tmp / "on", tmp / "off"
    train.train(cfg(tmp, on, metrics_seconds=0.05), device="cpu", make_opt=opt)
    train.train(cfg(tmp, off, **SLOW_OFF), device="cpu", make_opt=opt)
    keys = ("loss", "grad_norm", "mtp_kl", "reply_tokens", "examples")
    assert [[r[k] for k in keys] for r in rows(on, "step")] == [[r[k] for k in keys] for r in rows(off, "step")]
    assert [r["heldout_loss"] for r in rows(on, "eval")] == [r["heldout_loss"] for r in rows(off, "eval")]
    for step in (2, 3):
        a, b = on / f"ckpt-{step}", off / f"ckpt-{step}"
        files = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file() and p.suffix == ".safetensors")
        assert files and files == sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file() and p.suffix == ".safetensors")
        assert all((a / f).read_bytes() == (b / f).read_bytes() for f in files)
    assert not list(on.glob(".ckpt-*"))
    boot = rows(on, "boot")
    assert {"load", "optimizer", "data", "warm_window", "eval0"} <= set(boot[0]["seconds"]) and boot[1]["first_step"] > 0
    assert boot[0]["warm"]["window"] == 128 and boot[0]["warm"]["tokens"] == 200
    m = [json.loads(x) for x in (on / "metrics.jsonl").read_text().splitlines()]
    assert [r["m"]["train.step"] for r in m if "train.loss" in r["m"]] == [1, 2, 3]
    assert any("train.heldout_loss" in r["m"] for r in m)


def test_profile_and_step_split(setup, tmp_path):
    """profile=True: one example profiled before step 1 (gradients dropped: the steps equal a run without it), its
    trace reduced by a separate process; every step row carries the step split, util samples and token counts."""
    import time

    on, off = tmp_path / "prof", tmp_path / "plain"
    train.train(cfg(setup, on, profile=True, util_every=0.05), device="cpu", make_opt=opt)
    train.train(cfg(setup, off), device="cpu", make_opt=opt)
    keys = ("loss", "grad_norm", "mtp_kl", "reply_tokens")
    assert [[r[k] for k in keys] for r in rows(on, "step")] == [[r[k] for k in keys] for r in rows(off, "step")]
    for _ in range(600):  # the reduction runs on in its own process
        got = json.loads((on / "profile.json").read_text())
        if "flops_hw_matmul" in got:
            break
        time.sleep(0.1)
    assert got["wall_s"] > 0 and got["flops_matmul"] > 0 and 0 < got["window"] <= min(128, got["tokens"])
    assert "aten::mm" in got["flops_by_op"] or "aten::addmm" in got["flops_by_op"]
    assert rows(on, "profile")[0]["wall_s"] == got["wall_s"]
    for r in rows(on, "step"):
        assert {"examples_s", "update_s", "gap_s", "util_n", "vram_gib", "gap_util", "window_tokens", "alloc"} <= set(r)
        assert {"next_batch", "clip", "opt", "prepare"} <= set(r["update_split"])
        assert r["ctx_tokens"] >= r["window_tokens"] > 0
        assert r["examples_s"] + r["update_s"] == pytest.approx(r["seconds"], abs=0.11)


def test_ple_rows_exact(setup):
    """The n-gram rows looked up once per example on the host (ple_table) give the same loss sums and gradients as
    the model's own lookup per call, with a prefix cache (cut > 0) and the window's checkpoint replays; the plan's
    counts match the labels."""
    import ple

    model, head = train.load_model(setup / "student", "cpu", fast_attention=False)
    box, _ = mtp.hc_capture(model)
    ple.install(model)
    assert ple.modules(model)
    ex = train.load_example(setup / "data/train/e0.pt")
    got = []
    for on in (False, True):
        model.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)
        c = dataclasses.replace(cfg(setup, setup / "x", ple_table=on), window=64, prefix_chunk=24, loss_chunk=16)
        plan = train.loss_plan(c, model, feed_prepare(ex), torch.device("cpu"), False)
        assert plan["cut"] > 0 and (plan["ple"] is not None) == on
        loss = train.host(*train.example_loss(model, head, box, ex, c))
        got.append((loss, {n: p.grad.clone() for n, p in list(model.named_parameters()) + list(head.named_parameters())
                           if p.grad is not None}))
    (l0, g0), (l1, g1) = got
    assert l0 == l1 and g0.keys() == g1.keys() and all(torch.equal(g0[k], g1[k]) for k in g0)
    total = ex["input_ids"].shape[-1]
    assert l0[1] == int((ex["labels"].reshape(-1)[total - 64 + 1:] != -100).sum())


def feed_prepare(ex):
    import feed

    return feed.prepare(ex)


def test_prof_reduce_gaps():
    """Busy share, idle gaps by phase / overlapping CUDA call / op at the gap's end, and matmul FLOPs (with_flops'
    elementwise counts apart; grouped GEMMs from their dims) from a synthetic chrome trace."""
    import prof

    def ev(cat, name, ts, dur, **args):
        return {"cat": cat, "name": name, "ts": ts, "dur": dur, "args": args}

    evs = [ev("kernel", "k1", 0, 1000), ev("kernel", "k2", 3000, 1000), ev("kernel", "k3", 4000, 500),
           ev("kernel", "k4", 4510, 490),  # a 10 us gap: below min_gap
           ev("user_annotation", "sft:prefix", 0, 5000),
           ev("cuda_runtime", "cudaStreamSynchronize", 900, 300), ev("cuda_driver", "cuMemMap", 1300, 1500),
           ev("cpu_op", "aten::nonzero", 800, 2200), ev("cpu_op", "aten::mm", 2950, 40, **{"Input Dims": [[5, 2], [2, 5]]}),
           ev("cpu_op", "aten::mul", 10, 5, flops=7),
           ev("cpu_op", "aten::_grouped_mm", 20, 5, **{"Input Dims": [[8, 4], [2, 4, 3]]})]
    got = prof.reduce(evs, wall_s=0.005)
    assert got["busy_s"] == pytest.approx(0.00299) and got["kernel_span_s"] == pytest.approx(0.005)
    assert got["gaps"] == 1 and got["gap_s_by_size"][">=1000us"] == pytest.approx(0.002)
    assert got["idle_by_phase"][0] == {"name": "sft:prefix", "gaps": 1, "s": 0.002}
    calls = {r["name"]: r["s"] for r in got["idle_overlapping_calls"]}
    assert calls == {"cudaStreamSynchronize": 0.0002, "cuMemMap": 0.0015}
    assert got["idle_by_op_at_end"][0]["name"] == "aten::nonzero"
    assert got["flops_matmul"] == 100 and got["flops_grouped_mm"] == 2 * 8 * 4 * 3
    assert got["flops_hw_matmul"] == 100 + 192 and got["flops_by_op"]["aten::mul"] == 7


def test_prefetch_ranges_upload_compact(setup, monkeypatch):
    """parallel_reads with many small ranges per file and an upload device (CPU here) loads the same weights as
    transformers' own path; compact gives every viewed tensor its own storage, values unchanged."""
    import modelio
    from transformers import AutoModelForMultimodalLM

    import transformers.modeling_utils as mu

    path = setup / "student"
    calls = []
    monkeypatch.setattr(mu, "caching_allocator_warmup", lambda *a, **k: calls.append(1))
    with modelio.parallel_reads(path, threads=8, device="cpu", piece=4096) as how:
        a = AutoModelForMultimodalLM.from_pretrained(str(path), dtype=torch.bfloat16, device_map={"": "cpu"})
    assert not calls  # v7 OOM: the warm-up must not reserve the model beside the uploaded files
    b = AutoModelForMultimodalLM.from_pretrained(str(path), dtype=torch.bfloat16, device_map={"": "cpu"})
    assert calls  # outside parallel_reads from_pretrained does call the hook (the test sees the real path)
    assert how["mode"] == "parallel" and {"read_wall_s", "read_thread_s", "upload_s", "kept_host"} <= set(how)
    before = {k: v.clone() for k, v in a.state_dict().items()}
    assert modelio.compact(a) > 0
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) and torch.equal(sa[k], before[k]) for k in sa)
    assert all(t.untyped_storage().nbytes() == t.numel() * t.element_size() for t in a.parameters())


def test_prefetch_reads_only_mapped_tensors(setup, tmp_path):
    """A file holding tensors the index does not map (the base files' uncut experts) is read only where the mapped
    tensors lie; the model loads the same weights."""
    import shutil

    import modelio
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForMultimodalLM

    path = tmp_path / "student"
    shutil.copytree(setup / "student", path)
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    f = sorted(set(index.values()))[0]
    tensors = load_file(str(path / f))
    names = sorted(tensors)
    junk = {f"junk.{i}": torch.randn(1024, 1024) for i in range(4)}  # unmapped, 16 MiB (beyond the 1 MiB merge gap)
    save_file({**{n: tensors[n] for n in names[: len(names) // 2]}, **junk,
               **{n: tensors[n] for n in names[len(names) // 2:]}}, str(path / f))
    size = sum((path / g).stat().st_size for g in set(index.values()))
    with modelio.parallel_reads(path, threads=4, piece=4096) as how:
        a = AutoModelForMultimodalLM.from_pretrained(str(path), dtype=torch.bfloat16, device_map={"": "cpu"})
    b = AutoModelForMultimodalLM.from_pretrained(str(path), dtype=torch.bfloat16, device_map={"": "cpu"})
    assert how["read_bytes"] <= size - (16 << 20)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_upload_room_guard(monkeypatch):
    """A file goes to the GPU only with its size + 16 GiB free; otherwise it stays on the host."""
    import modelio

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (20 << 30, 280 << 30))
    assert modelio.room(10 << 30, "cpu")
    assert modelio.room(4 << 30, "cuda:0") and not modelio.room(5 << 30, "cuda:0")


def test_parallel_reads_same_weights(setup):
    a, _ = train.load_model(setup / "student", "cpu", fast_attention=False, fast_load=True)
    assert train.LOAD["mode"] == "parallel" and train.LOAD["read_gib"] >= 0 and train.LOAD["read_thread_s"] >= 0
    assert {"read_wall_s", "read_wait_s", "from_pretrained_s", "mtp_s"} <= set(train.LOAD)
    b, _ = train.load_model(setup / "student", "cpu", fast_attention=False, fast_load=False)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_optimizer_roundtrip(setup, tmp_path):
    """The checkpoint's optimizer files (torchao 8-bit states by their inner tensors) load back to the same state."""
    import ckpt

    model, head = train.load_model(setup / "student", "cpu", fast_attention=False)
    params = train.trainable(model, head)
    o = opt(params, 1e-3)
    for p in params:
        p.grad = torch.randn_like(p) * 1e-2
    o.step()
    c = cfg(setup, tmp_path / "rt")
    (tmp_path / "rt").mkdir()
    saver = ckpt.Saver(c, async_=True)
    saver.save(model, head, o, 1, (0, 1))
    saver.wait()
    o2 = opt(params, 1e-3)
    ckpt.load_optimizer(o2, ckpt.latest(c.out))
    s1, s2 = o.state_dict()["state"], o2.state_dict()["state"]

    def flat(t):
        if type(t) is not torch.Tensor and hasattr(t, "__tensor_flatten__"):
            return [x for a in t.__tensor_flatten__()[0] for x in flat(getattr(t, a))]
        return [t]

    assert s1.keys() == s2.keys()
    for pid in s1:
        for k in s1[pid]:
            assert all(torch.equal(x, y) for x, y in zip(flat(s1[pid][k]), flat(s2[pid][k]))), (pid, k)


@pytest.fixture
def selected():
    """The attention fast path with the torch SelectedAttention (CPU); the model's classes restored after."""
    import attention
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    yield attention
    for cls, flag in ((m.Qwen4ExpTextQSAIndexer, "_rl1_vectorized"), (m.Qwen4ExpTextAttention, "_rl1_selected")):
        if getattr(cls, flag, False):
            cls.forward = cls._rl1_original
            setattr(cls, flag, False)
    attention.SPARSE["mode"], attention.SEL["on"] = "masked", False


def grads_of(setup, ex, c):
    model, head = train.load_model(setup / "student", "cpu", fast_attention=False)
    box, _ = mtp.hc_capture(model)
    loss = train.example_loss(model, head, box, ex, c)
    return loss, {n: p.grad.clone() for n, p in list(model.named_parameters()) + list(head.named_parameters())
                  if p.grad is not None}


def test_empty_window_example_is_a_counted_zero_pass(setup):
    """No reply row in the window (capped labels cleared, v1c step 746): no raise, counted, zero grads on every trained
    parameter that a real example reaches, so a rank's collectives and the step stay as they were."""
    ex = train.load_example(setup / "data/train/e0.pt")
    cleared = {**ex, "labels": torch.full_like(ex["labels"], -100)}
    c = cfg(setup, setup / "x")
    before = train.EMPTY["n"]
    (_, n, _, _, rows, _), grads = grads_of(setup, cleared, c)
    assert train.EMPTY["n"] == before + 1 and n == 0 and rows == 0
    assert grads and all(g.abs().sum() == 0 for g in grads.values())
    _, real = grads_of(setup, ex, c)
    assert train.EMPTY["n"] == before + 1 and any(g.abs().sum() > 0 for g in real.values())


def test_selected_attention_and_sel_cache(setup, selected):
    ex = train.load_example(setup / "data/train/e0.pt")
    c = cfg(setup, setup / "x")
    base_loss, base = grads_of(setup, ex, c)
    selected.install(mode="selected", sel_cache=False)
    loss0, g0 = grads_of(setup, ex, c)
    calls = selected.SPARSE["calls"]
    selected.SEL["on"] = True
    loss1, g1 = grads_of(setup, ex, c)
    assert calls > 0 and selected.SEL["hits"] > 0
    assert loss0 == loss1 and g0.keys() == g1.keys() and all(torch.equal(g0[k], g1[k]) for k in g0)
    assert abs(loss0[0] - base_loss[0]) <= 2e-2 * abs(base_loss[0])  # BF16 model, other attention arithmetic
    num = sum(float((g0[k].float() - base[k].float()).norm() ** 2) for k in base) ** 0.5
    den = sum(float(base[k].float().norm() ** 2) for k in base) ** 0.5
    assert num / den < 5e-2


def stub_triton():
    """Enough of triton to import gdn.py on CPU (its kernels are never called here)."""
    if "triton" in sys.modules:
        return
    tr, tl = types.ModuleType("triton"), types.ModuleType("triton.language")
    tr.jit = lambda f=None, **kw: f if f is not None else (lambda g: g)
    tr.cdiv = lambda a, b: -(-a // b)
    tl.constexpr = object
    tr.language = tl
    sys.modules.update({"triton": tr, "triton.language": tl})


def test_gdn_weight_grads():
    """conv_weight_grad / norm_weight_grad against autograd of the model's ops (fp32 inputs: same sums)."""
    try:
        import triton  # noqa: F401
    except ImportError:
        stub_triton()
    import gdn

    g = torch.Generator().manual_seed(0)
    B, T, C = 2, 11, 6
    x = torch.randn(B, T, C, generator=g)
    prev = torch.randn(B, C, 5, generator=g)
    w = torch.randn(C, 1, 4, generator=g, requires_grad=True)
    for p in (None, prev):
        full = torch.cat([p[..., -3:] if p is not None else torch.zeros(B, C, 3), x.transpose(1, 2)], dim=-1)
        out = torch.nn.functional.conv1d(full, w, groups=C)  # [B, C, T]: out[t] = sum_j w[j] full[t + j]
        d = torch.randn(out.shape, generator=g)
        (gw,) = torch.autograd.grad((out * d).sum(), w)
        got = gdn.conv_weight_grad(d.transpose(1, 2).reshape(B * T, C), x.reshape(B * T, C), p, B, T, C)
        assert torch.allclose(got, gw, atol=1e-4, rtol=1e-4)
    N, nh, D, eps = 7, 3, 8, 1e-6
    o, z = torch.randn(N, nh, D, generator=g), torch.randn(N, nh * D, generator=g)
    nw = torch.randn(D, generator=g, requires_grad=True)
    h = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)
    out = (nw * h.to(torch.bfloat16).float()) * torch.sigmoid(z.view(N, nh, D))
    d = torch.randn(out.shape, generator=g)
    (gn,) = torch.autograd.grad((out * d).sum(), nw)
    got = gdn.norm_weight_grad(o, z, d.reshape(N, nh * D), eps)
    assert torch.allclose(got, gn, rtol=2e-2, atol=2e-2)  # BF16 rounding of the gated grad, as the kernel's


def test_smoke_attention_check(setup, selected):
    """smoke.attention_check on CPU (the torch SelectedAttention standing in for the Triton kernel) in a window where
    the indexer drops blocks: within the smoke's bars on the trainer's SDPA reference; the bare-config eager
    reference (26 Oct GPU smoke: cos 0.973, grad cos 0.774) fails them."""
    import smoke
    from transformers import AutoConfig

    text = AutoConfig.from_pretrained(setup / "student").text_config
    full = next(i for i, t in enumerate(text.layer_types) if t != "linear_attention")
    att = smoke.layer(setup / "student", text, full, device="cpu")
    selected.install(mode="selected")
    T = 512
    assert text.indexer_budget * 2 <= T  # selection active: late queries see well under their causal keys

    def check(impl):
        return smoke.attention_check(att, text, T, torch.Generator().manual_seed(1), device="cpu", fast="selected",
                                     impl=impl)

    got = check("sdpa")
    assert got["calls"] > 0 and not got["grads_missing"], got
    assert got["fwd_cos"] >= 0.9999 and got["fwd_rel"] <= 0.02 and got["grad_cos_min"] >= 0.999, got
    bad = check(None)
    assert bad["fwd_cos"] < 0.9999 or bad["grad_cos_min"] < 0.999, bad


TRAIN_STUB = '''
import threading
class Config:
    def __init__(self, **kw):
        self.out = kw["out"]
def train(cfg, device, refresh, commit, guard, knobs):
    held = open(__file__)  # a file held open, as the trainer's metrics / checkpoint writer / data feed do
    threads = [threading.Thread(target=lambda: [commit() for _ in range(20)]) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    try:
        refresh()
    except RuntimeError as e:
        assert "open files" in str(e)
    assert knobs() == '{"lr": 2e-06}', knobs()
    held.close()
'''


def test_train_child_volume_bridge(tmp_path):
    """modal_sft's TRAIN child runs without modal (the trainer venv has none: 26 Oct smoke, both boots died on
    import) and its commits / reloads, from several threads, reach the parent's volume ops; a failed op raises in
    the child."""
    import ast

    import vol

    tree = ast.parse((ROOT / "scripts/sft/modal_sft.py").read_text())
    code = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "TRAIN" for t in n.targets))
    stubs = tmp_path / "stubs"
    (stubs / "arc_infra/runops").mkdir(parents=True)
    (stubs / "modal.py").write_text("raise ImportError('no modal in the trainer venv')\n")
    (stubs / "arc_infra/__init__.py").write_text("")
    (stubs / "arc_infra/runops/__init__.py").write_text("")
    (stubs / "arc_infra/runops/restart.py").write_text(
        "class RestartGuard:\n    def __init__(self, root, run, commit, **kw):\n        commit()\n")
    (stubs / "train.py").write_text(TRAIN_STUB)
    env = {"PYTHONPATH": f"{stubs}:{ROOT / 'scripts/sft'}", "PATH": "/usr/bin:/bin"}
    args = [sys.executable, "-c", code, json.dumps({"run": "x", "train": {"out": "/runs/x"}})]
    seen = {"commit": 0, "reload": 0, "read": []}

    def reload():
        seen["reload"] += 1
        raise RuntimeError("there are open files preventing the operation")  # Modal's reload with a file open

    ops = {"commit": lambda: seen.__setitem__("commit", seen["commit"] + 1), "reload": reload,
           "read": lambda path: seen["read"].append(path) or '{"lr": 2e-06}'}
    assert vol.run_child(args, env, ops).returncode == 0
    assert seen == {"commit": 61, "reload": 1, "read": ["/runs/x/knobs.json"]}

    def boom():
        raise OSError("boom")

    assert vol.run_child(args, env, {**ops, "commit": boom}).returncode == 1


def test_run_child_log_keeps_a_crash(tmp_path):
    """Stdout, stderr and the abort's faulthandler stack land in the log file, committed before run_child returns."""
    import vol

    log = tmp_path / "shard0" / "child.log"
    commits = []
    code = "import os, sys, time\nprint('row', flush=True)\nprint('native message', file=sys.stderr, flush=True)\ntime.sleep(0.3)\nos.abort()"
    env = {"PATH": "/usr/bin:/bin", "PYTHONFAULTHANDLER": "1", "PYTHONUNBUFFERED": "1"}
    r = vol.run_child([sys.executable, "-c", code], env, {"commit": lambda: commits.append(1)}, log_path=log, log_commit_s=0.1)
    text = log.read_text()
    assert r.returncode == -6 and len(commits) >= 2
    assert "row" in text and "native message" in text and "Fatal Python error: Aborted" in text
    assert text.endswith("CHILD EXIT -6\n")


def test_crash_keeps_the_checkpoint_write_in_flight(tmp_path):
    """The write thread is a daemon: at a crash in the main thread it finishes before the interpreter ends."""
    import subprocess

    done = tmp_path / "written"
    code = f"""
import sys, threading, time, types
sys.path.insert(0, {str(ROOT / "scripts/sft")!r})
import ckpt
s = ckpt.Saver(types.SimpleNamespace(), lambda: None, True, lambda **k: None)
s.thread = threading.Thread(target=lambda: (time.sleep(1), open({str(done)!r}, "w").close()), daemon=True)
s.thread.start()
raise RuntimeError("crash")
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 1 and "crash" in r.stderr and done.exists()


def test_failed_reload_is_not_fatal_knobs_live(setup):
    """A volume reload that fails (open files on the volume: 26 Oct smoke v4 died on its first) is skipped and
    logged; the run steps on, and the knobs still arrive through ``knobs`` (the parent's read of the committed file)."""
    out = setup / "reload"

    def refresh():
        raise RuntimeError("there are open files preventing the operation")

    train.train(cfg(setup, out, boundary_reload=True, **SLOW_OFF), device="cpu", make_opt=opt, refresh=refresh,
                knobs=lambda: json.dumps({"lr": 1.25e-4}))
    assert [r["step"] for r in rows(out, "step")] == [1, 2, 3]
    lrs = [r["lr"] for r in rows(out, "step")]  # knobs are read on the boundary thread: one step late at most
    assert lrs[-1] == 1.25e-4 and lrs[1:] == [1.25e-4] * 2 or lrs[2:] == [1.25e-4]
    assert rows(out, "reload_skipped")[0]["count"] == 1


def test_no_boundary_reload_by_default_and_load_retry(setup, monkeypatch, tmp_path):
    """28 Oct restart (ENOENT on an existing teacher file while a reload swapped the mount): no step-boundary reload
    unless asked, and a file not found is retried before the loader fails."""
    import feed

    out = setup / "noreload"
    calls = []
    train.train(cfg(setup, out, **SLOW_OFF), device="cpu", make_opt=opt, refresh=lambda: calls.append(1),
                knobs=lambda: json.dumps({"lr": 1.25e-4}))
    assert [r["step"] for r in rows(out, "step")] == [1, 2, 3] and not calls
    path = tmp_path / "ex.pt"
    torch.save({"input_ids": torch.arange(4, dtype=torch.int32)}, path)
    real, seen = torch.load, []

    def flaky(p, **kw):
        seen.append(p)
        if len(seen) < 3:
            raise FileNotFoundError(p)
        return real(p, **kw)

    monkeypatch.setattr(feed.torch, "load", flaky)
    monkeypatch.setitem(feed.RETRY, "sleep", 0.0)
    assert feed.load_example(path)["input_ids"].shape == (1, 4) and len(seen) == 3
    seen.clear()
    monkeypatch.setitem(feed.RETRY, "tries", 2)
    with pytest.raises(FileNotFoundError):
        feed.load_example(path)


def test_stale_stop_knob_is_ignored(setup):
    """27 Oct: attempt 2 of sft-p75-qad-v2 stopped at its first checkpoint on the {"stop": true} left from the hold that
    stopped attempt 1. A stop already in the file at start is logged and ignored; a new value still stops the run."""
    out = setup / "stale"
    train.train(cfg(setup, out, **SLOW_OFF), device="cpu", make_opt=opt, knobs=lambda: json.dumps({"stop": True}))
    assert [r["step"] for r in rows(out, "step")] == [1, 2, 3] and rows(out, "stale_stop_ignored")
    seen = {}
    assert train.read_knobs(out, seen, lambda: json.dumps({"stop": True})) == {"stop": True}  # first sight (start)
    assert train.read_knobs(out, seen, lambda: json.dumps({"stop": True})) == {}  # the same value again: stale
    assert train.read_knobs(out, seen, lambda: json.dumps({"stop": "now"})) == {"stop": True}  # a new value stops
    assert train.read_knobs(out, seen, lambda: json.dumps({"batch_reply_tokens": 300})) == {"batch": 300}
    assert train.read_knobs(out, seen, lambda: json.dumps({"batch_reply_tokens": 300})) == {}


def test_plain_causal_in_pieces():
    """plain_causal row by row in pieces (no [L, keys] temporaries: 26 Oct smoke v5 asked 43 GiB for one) gives the
    whole-mask answer for causal, prefix-causal, float and non-causal masks, and its largest single allocation stays
    near one piece."""
    import attention

    def whole(mask):
        v = mask if mask.dtype == torch.bool else mask == 0
        c = v[0, 0].sum(-1)
        return c if torch.equal(v[0, 0], torch.arange(v.shape[-1])[None, :] < c[:, None]) else None

    L, K = 256, 1024
    causal = (torch.arange(K)[None, :] < torch.arange(K - L + 1, K + 1)[:, None])[None, None]
    flt = torch.where(causal, 0.0, torch.finfo(torch.float32).min)
    holed = causal.clone()
    holed[0, 0, 200, 3] = False
    old = attention.CAUSAL["piece"]
    try:
        attention.CAUSAL["piece"] = 4096
        for mask in (causal, flt, holed, causal[:, :, :1]):
            attention.reset()
            got, ref = attention.plain_causal(mask), whole(mask)
            assert (got is None) == (ref is None) and (got is None or torch.equal(got, ref))
        attention.reset()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True) as prof:
            attention.plain_causal(causal)
        biggest = max(e.cpu_memory_usage for e in prof.events())
        assert biggest <= 4 * 4096 + 8 * L, biggest  # a piece of int32 / bool rows, not L x K x 8 B
        attention.reset()
        mark = attention.causal_marker(L, K, "cpu")  # the MTP head's stand-in: same counts, no [L, K] mask
        assert mark.numel() == 1 and torch.equal(attention.plain_causal(mark), whole(causal))
    finally:
        attention.CAUSAL["piece"] = old
        attention.reset()


def test_write_safetensors_parallel_ranges(tmp_path):
    """The checkpoint writer's positional range writes (small pieces, several threads) read back exactly."""
    import ckpt
    from safetensors import safe_open

    g = torch.Generator().manual_seed(0)
    items = [("a", torch.randn(1000, 7, generator=g).to(torch.bfloat16)), ("b", torch.zeros(0)),
             ("c", torch.randint(0, 255, (5003,), generator=g, dtype=torch.uint8)), ("d", torch.randn(3, 3).t())]
    ckpt.write_safetensors(tmp_path / "x.safetensors", items, threads=4, piece=1000)
    with safe_open(str(tmp_path / "x.safetensors"), "pt") as f:
        assert all(torch.equal(f.get_tensor(n), t.contiguous()) for n, t in items)


def test_adamw4bit_states_and_step(setup):
    """optim_bits=4 builds torchao's AdamW4bit; init_states creates its states before step 1 (about half the 8-bit
    bytes) and a step runs on them."""
    pytest.importorskip("torchao")
    model, head = train.load_model(setup / "student", "cpu", fast_attention=False)
    params = train.trainable(model, head)
    o8, o4 = train.adamw8bit(params, 1e-3), train.adamw8bit(params, 1e-3, 4)
    g8, g4 = train.init_states(o8), train.init_states(o4)
    assert type(o4).__name__ == "AdamW4bit" and g4 <= g8 and all(len(o4.state[p]) == 3 for p in params)
    for p in params:
        p.grad = torch.randn_like(p) * 1e-2
    o4.step()
    assert all(torch.isfinite(p).all() for p in params)


def test_paged_states_bit_equal(monkeypatch):
    """paged.check: host-paged states give the on-card optimizer's weights and state bytes after 3 steps (CPU: the
    paged step works on copies, so a missing write-back would differ); 8- and 4-bit."""
    pytest.importorskip("torchao")
    import paged

    for bits in (8, 4):
        got = paged.check(lambda ps: train.adamw8bit(ps, 1e-3, bits), "cpu", shapes=((256, 256), (512, 256)))
        assert got["ok"], (bits, got)


def _kahan_run(kahan, dtype, lr, steps):
    import paged

    p = torch.nn.Parameter(torch.ones(256, 256, dtype=dtype))
    opt = train.adamw8bit([p], lr)
    paged.Pager(opt, "cpu", kahan=kahan).install()
    for _ in range(steps):
        p.grad = torch.ones(256, 256, dtype=dtype)
        opt.step()
    st = opt.state[p]
    assert ("comp" in st) == kahan
    if kahan:
        assert st["comp"].dtype is torch.int8  # 1 byte per weight on the host
        return p.detach(), p.detach().float() + st["comp"].float() * paged.ulp(p.detach()) / 254
    return p.detach(), p.detach().float()


def test_kahan_accumulates_sub_ulp_updates():
    """kahan=True: updates far below half a BF16 ulp (2e-5 vs 2e-3 at 1.0) add up as with FP32 weights (weight +
    comp), where round-to-nearest BF16 would never move."""
    pytest.importorskip("torchao")
    _, ref = _kahan_run(False, torch.float32, 2e-5, 200)
    w, got = _kahan_run(True, torch.bfloat16, 2e-5, 200)
    assert abs(float(ref.mean()) - (1 - 200 * 2e-5)) < 2e-4
    assert abs(float(got.mean()) - float(ref.mean())) < 2e-5 and (got - ref).abs().max() < 1e-3
    assert w.dtype is torch.bfloat16 and float(w.float().max()) < 1.0  # the BF16 weight itself moved


def test_kahan_keeps_updates_below_comp_resolution():
    """K1b (27 Oct): a round-to-nearest int8 comp dropped every update under ulp/508 (norm weights at lr 3e-6).
    0.0005 ulp per step (0.13 comp units) must still add up in expectation: 400 steps = 0.2 ulp at 1.0."""
    pytest.importorskip("torchao")
    lr = 0.0005 * 2.0 ** -7
    _, ref = _kahan_run(False, torch.float32, lr, 400)
    _, got = _kahan_run(True, torch.bfloat16, lr, 400)
    moved = 1 - float(ref.mean())
    assert abs(moved - 400 * lr) < 0.05 * 400 * lr
    assert abs((1 - float(got.mean())) - moved) < 0.05 * moved


def test_bf16_ulp():
    import paged

    w = torch.tensor([1.0, 0.5, -3.0, 0.0, 0.02], dtype=torch.bfloat16)
    want = torch.tensor([2.0 ** -7, 2.0 ** -8, 2.0 ** -6, 2.0 ** -133, 2.0 ** -13])
    assert torch.equal(paged.ulp(w), want)


def test_kahan_train_checkpoint_resume(setup, tmp_path, monkeypatch):
    """kahan through the trainer: steps finite, comp states in the checkpoint, a resume continues from them."""
    pytest.importorskip("torchao")
    import ckpt
    import paged

    monkeypatch.setattr(paged, "MIN_NUMEL", 1)
    out = tmp_path / "k"
    train.train(cfg(setup, out, paged_states=True, kahan=True, stop_step=2, **SLOW_OFF), device="cpu",
                make_opt=train.adamw8bit)
    steps = rows(out, "step")
    assert len(steps) == 2 and all(math.isfinite(r["loss"]) for r in steps)
    model, head = train.load_model(setup / "student", "cpu", fast_attention=False)
    o = train.adamw8bit(train.trainable(model, head), 1e-3)
    paged.Pager(o, "cpu", kahan=True).install()
    ckpt.load_optimizer(o, out / "ckpt-2")
    bf = [p for g in o.param_groups for p in g["params"] if p.dtype is torch.bfloat16 and len(o.state[p])]
    assert bf and all("comp" in o.state[p] for p in bf)
    assert any(float(o.state[p]["comp"].float().abs().max()) > 0 for p in bf)
    train.train(cfg(setup, out, paged_states=True, kahan=True, **SLOW_OFF), device="cpu",
                make_opt=train.adamw8bit)
    assert [r["step"] for r in rows(out, "step")] == [1, 2, 3]
    with pytest.raises(ValueError):
        train.train(cfg(setup, tmp_path / "x", kahan=True, **SLOW_OFF), device="cpu", make_opt=train.adamw8bit)


def test_paged_train_same_steps_and_checkpoint(setup, tmp_path, monkeypatch):
    """paged_states=True: the same step rows and checkpoint bytes as the on-card states; a loaded optimizer is paged
    out again with the same values."""
    pytest.importorskip("torchao")
    import ckpt
    import paged

    monkeypatch.setattr(paged, "MIN_NUMEL", 1)  # the tiny student's tensors are all small: page everything
    on, off = tmp_path / "paged", tmp_path / "card"
    train.train(cfg(setup, on, paged_states=True, **SLOW_OFF), device="cpu", make_opt=train.adamw8bit)
    train.train(cfg(setup, off, **SLOW_OFF), device="cpu", make_opt=train.adamw8bit)
    keys = ("loss", "grad_norm", "mtp_kl", "reply_tokens")
    assert [[r[k] for k in keys] for r in rows(on, "step")] == [[r[k] for k in keys] for r in rows(off, "step")]
    assert all("paged" in r["update_split"] for r in rows(on, "step"))
    a, b = on / "ckpt-2", off / "ckpt-2"
    files = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file() and p.suffix in (".safetensors", ".pt"))
    assert files and all((a / f).read_bytes() == (b / f).read_bytes() for f in files if f.suffix == ".safetensors")
    model, head = train.load_model(setup / "student", "cpu", fast_attention=False)
    o = train.adamw8bit(train.trainable(model, head), 1e-3)
    pager = paged.Pager(o, "cpu").install()
    ckpt.load_optimizer(o, a)
    before = {id(p): [x.clone() for v in st.values() for x in paged._leaves(v)] for p, st in o.state.items()}
    pager.page_out()
    after = {id(p): [x for v in st.values() for x in paged._leaves(v)] for p, st in o.state.items()}
    assert all(torch.equal(x, y) for k in before for x, y in zip(before[k], after[k]))


def test_paged_async_checkpoint_writes_host_states_direct(setup, tmp_path, monkeypatch):
    """Async checkpoints with paged states: the host states are written from their own storage (not the snapshot
    buffer), and the next optimizer step waits for that file: with the optim write held back 1 s, the checkpoint
    bytes and steps still equal the on-card, inline run's; resuming from it gives the same last step."""
    pytest.importorskip("torchao")
    import time

    import ckpt
    import paged

    monkeypatch.setattr(paged, "MIN_NUMEL", 1)
    plain = ckpt.write_safetensors

    def slow(path, items, *a, **k):
        if Path(path).name == "optim.safetensors":
            time.sleep(1.0)  # the next step reaches before_update first
        return plain(path, items, *a, **k)

    monkeypatch.setattr(ckpt, "write_safetensors", slow)
    on, off = tmp_path / "paged", tmp_path / "card"
    train.train(cfg(setup, on, paged_states=True, **dict(SLOW_OFF, async_ckpt=True)), device="cpu",
                make_opt=train.adamw8bit)
    train.train(cfg(setup, off, **SLOW_OFF), device="cpu", make_opt=train.adamw8bit)
    keys = ("loss", "grad_norm", "mtp_kl", "reply_tokens")
    assert [[r[k] for k in keys] for r in rows(on, "step")] == [[r[k] for k in keys] for r in rows(off, "step")]
    written = rows(on, "ckpt_written")
    assert written and all(r["direct_n"] > 0 for r in written)
    a, b = on / "ckpt-2", off / "ckpt-2"
    files = sorted(p.relative_to(a) for p in a.rglob("*.safetensors"))
    assert files and all((a / f).read_bytes() == (b / f).read_bytes() for f in files)
    again = tmp_path / "again"
    import shutil

    shutil.copytree(on, again)
    shutil.rmtree(again / "ckpt-3")
    (again / "events.jsonl").write_text("")
    train.train(cfg(setup, again, paged_states=True, **dict(SLOW_OFF, async_ckpt=True)), device="cpu",
                make_opt=train.adamw8bit)
    assert [[r[k] for k in keys] for r in rows(again, "step")] == [[r[k] for k in keys] for r in rows(on, "step")][2:]


def test_state_subclasses_cacheable():
    """cacheable_states: torchao's 8/4-bit states get a stable compile-cache hash that separates sign and shape."""
    pytest.importorskip("torchao")
    from torchao.optim.subclass_8bit import OptimState8bit

    train.cacheable_states()
    a, b = OptimState8bit.zeros((256, 256), True, 256), OptimState8bit.zeros((256, 256), False, 256)
    c = OptimState8bit.zeros((512, 256), True, 256)
    assert a._stable_hash_for_caching() == OptimState8bit.zeros((256, 256), True, 256)._stable_hash_for_caching()
    assert len({a._stable_hash_for_caching(), b._stable_hash_for_caching(), c._stable_hash_for_caching()}) == 3


def test_keep_layers_same_steps(setup, tmp_path):
    """keep_layers: the last trunk layers skip the checkpoint recompute; the steps are unchanged."""
    on, off = tmp_path / "keep", tmp_path / "ckpt"
    train.train(cfg(setup, on, keep_layers=1, **SLOW_OFF), device="cpu", make_opt=opt)
    train.train(cfg(setup, off, **SLOW_OFF), device="cpu", make_opt=opt)
    keys = ("loss", "grad_norm", "mtp_kl", "reply_tokens")
    assert [[r[k] for k in keys] for r in rows(on, "step")] == [[r[k] for k in keys] for r in rows(off, "step")]
    assert rows(on, "start")[0]["load"]["kept_layers"] == 1


def test_gpu_launches_are_spawned():
    """modal_sft's entrypoint spawns every GPU function (a held .remote()/.starmap()/.map() dies with the client's
    connection, 26 Oct); only the CPU preflight and the CPU pytest entry (--tests) may be held."""
    import ast

    tree = ast.parse((ROOT / "scripts/sft/modal_sft.py").read_text())
    main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
    held = [ast.unparse(n) for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("remote", "starmap", "map", "remote_gen")
            and ast.unparse(n.func.value) not in ("preflight", "pytest_cpu")]
    assert not held, held
    assert any(isinstance(n, ast.Attribute) and n.attr == "spawn" for n in ast.walk(main))


@pytest.mark.parametrize("error", [lambda: torch.cuda.OutOfMemoryError("test"),
                                   lambda: RuntimeError("Triton Error [CUDA]: out of memory")])
def test_keep_layers_step_down_on_warm_oom(setup, tmp_path, monkeypatch, error):
    """A warm-up that runs out of memory with kept layers retries with one fewer, down to 0 (then memory_stop); a
    Triton launch OOM (a RuntimeError, 28 Oct lr probe) counts as one."""
    real, calls = train.warm_window, []

    def warm(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise error()
        return real(*a, **k)

    monkeypatch.setattr(train, "warm_window", warm)
    out = tmp_path / "down"
    slow = {k: v for k, v in SLOW_OFF.items() if k != "warm_start"}
    train.train(cfg(setup, out, keep_layers=2, warm_start=True, **slow), device="cpu", make_opt=opt)
    assert [r["keep_layers"] for r in rows(out, "keep_layers_down")] == [1] and len(calls) == 2
    assert rows(out, "boot")[0]["warm"]["kept_layers"] == 1


def test_index_check_counts_and_gates(setup, selected):
    """attention.index_check runs a pass with the fp32 picks, counts the TF32 picks found among them (on CPU TF32 is a
    no-op: all match) and keeps TF32 on only at >= the floor; the pass's grads are those of the plain fp32 indexer."""
    ex = train.load_example(setup / "data/train/e0.pt")
    c = cfg(setup, setup / "x")
    selected.install(mode="selected", sel_cache=False)
    loss0, g0 = grads_of(setup, ex, c)
    box = {}
    got = selected.index_check(lambda: box.update(r=grads_of(setup, ex, c)))
    assert got["picked"] > 0 and got["match"] == 1.0 and got["on"] and selected.INDEX["tf32"]
    loss1, g1 = box["r"]
    assert loss0 == loss1 and all(torch.equal(g0[k], g1[k]) for k in g0)
    assert selected.index_check(lambda: grads_of(setup, ex, c), floor=1.5)["on"] is False
    selected.INDEX["tf32"] = False


def test_short_probe_skips_the_snapshot_buffer(setup):
    """28 Oct p75 lr probes: an 8-step probe without stop_ckpt pinned a ~240 GB snapshot buffer it never used (205-240 s
    of boundary gap at steps 2-3). With no checkpoint possible the buffer is skipped; with one possible it is made."""
    c = dataclasses.replace(cfg(setup, setup / "x", ckpt_seconds=3600, stop_step=8, stop_ckpt=False), ckpt_steps=0,
                            total_steps=231)
    assert not train.ckpt_possible(c, 1, 110.0, 120.0)
    assert train.ckpt_possible(c, 1, 600.0, 120.0)  # slow steps: the time-based checkpoint can fall inside
    assert train.ckpt_possible(dataclasses.replace(c, stop_ckpt=True), 1, 110.0, 120.0)
    assert train.ckpt_possible(dataclasses.replace(c, stop_step=0), 1, 110.0, 120.0)  # the final checkpoint
    assert train.ckpt_possible(dataclasses.replace(c, ckpt_steps=4), 1, 110.0, 120.0)
    assert not train.ckpt_possible(dataclasses.replace(c, ckpt_steps=16), 1, 110.0, 120.0)
    short, full = setup / "short", setup / "full"
    train.train(dataclasses.replace(cfg(setup, short, **dict(SLOW_OFF, async_ckpt=True, stop_step=2, stop_ckpt=False)),
                                    ckpt_steps=0), device="cpu", make_opt=opt)
    assert [r["step"] for r in rows(short, "step")] == [1, 2] and not rows(short, "ckpt_buffer")
    assert not rows(short, "ckpt")
    train.train(cfg(setup, full, **dict(SLOW_OFF, async_ckpt=True)), device="cpu", make_opt=opt)
    assert rows(full, "ckpt_buffer") and rows(full, "ckpt")


def test_loop_never_waits_on_the_volume(setup):
    """28 Oct p75 lr probes: with 3 runs and 2 preflights committing to one volume, the training loop waited 205-240 s
    at steps 2-3 (its metrics row committed inline; its events opened volume files). Rows now go to a local spool that
    the boundary and metrics threads copy to <out> and commit: a 1 s commit never shows in a step's gap, and the
    volume copy ends with every row, once and in order."""
    import time as _time

    commits = []

    def slow_commit():
        commits.append(threading.current_thread().name)
        _time.sleep(1.0)

    out = setup / "slowvol"
    train.train(dataclasses.replace(cfg(setup, out, **SLOW_OFF), ckpt_steps=0), device="cpu", make_opt=opt,
                commit=slow_commit)
    steps = rows(out, "step")
    assert [r["step"] for r in steps] == [1, 2, 3]
    assert all(r["gap_s"] < 0.5 for r in steps), [r["gap_s"] for r in steps]
    assert any(name == "sft-volume" for name in commits)  # the boundary thread commits
    local = train.SPOOL[str(out)]["local"]
    assert (local / "events.jsonl").read_text() == (out / "events.jsonl").read_text()
    assert (local / "metrics.jsonl").read_text() == (out / "metrics.jsonl").read_text()
    assert rows(out, "done") and len(rows(out, "start")) == 1


def _cache_fns():
    import ast

    tree = ast.parse((ROOT / "scripts/sft/modal_sft.py").read_text())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("compile_cache", "sync_cache", "cache_env", "cache_tar_in", "cache_tar_out")]
    ns = {"Path": Path, "json": json}
    exec(compile(ast.Module(fns, []), "modal_sft_cache", "exec"), ns)
    return ns


def test_cache_copy_incomplete_falls_back_to_volume(tmp_path):
    """28 Oct lr probe: the copy-in hit its 600 s limit on the degraded volume, the Triton autotune picks stayed
    behind and the GDN backward autotuned at peak memory (launch OOM). A stopped copy reports complete False and the
    run reads the volume cache; a full copy runs on the local one."""
    ns = _cache_fns()
    shared = tmp_path / "runs/compile-cache/sft-shared"
    for i in range(40):
        f = shared / ("inductor" if i % 2 else "triton") / f"k{i}" / "e.bin"
        f.parent.mkdir(parents=True)
        f.write_bytes(bytes([i]) * (i + 1))
    env, local, got = ns["cache_env"](shared, tmp_path / "local", limit_s=-1)
    assert not local and not got["complete"] and env["TRITON_CACHE_DIR"] == f"{shared}/triton"
    env, local, got = ns["cache_env"](shared, tmp_path / "local", limit_s=600)
    assert local and got["complete"] and got["files"] == 40
    assert env["TRITON_CACHE_DIR"] == f"{tmp_path / 'local'}/sft-shared/triton"
    assert all((tmp_path / "local/sft-shared" / f.relative_to(shared)).read_bytes() == f.read_bytes()
               for f in shared.rglob("*") if f.is_file())
    assert not list((tmp_path / "local").rglob("*.part*"))
    again = ns["sync_cache"](shared, tmp_path / "local/sft-shared")
    assert again["complete"] and again["files"] == 0


def test_cache_shared_tar_round_trip(tmp_path):
    """Compile cache back to the volume as one tar (inode budget): the legacy folder seeds the first run, its new
    entries land in sft-shared.tar, the next run extracts the tar; a run that added nothing writes no tar."""
    ns = _cache_fns()
    shared = tmp_path / "runs/compile-cache/sft-shared"
    for i in range(5):
        f = shared / "triton" / f"k{i}" / "e.bin"
        f.parent.mkdir(parents=True)
        f.write_bytes(bytes([i]) * (i + 1))
    local = tmp_path / "local"
    env, use_local, got = ns["cache_env"](shared, local)
    assert use_local and got["files"] == 5
    assert not ns["cache_tar_out"](local / "sft-shared", f"{shared}.tar", got["files"])["written"]
    (local / "sft-shared/inductor/new.bin").parent.mkdir(parents=True)
    (local / "sft-shared/inductor/new.bin").write_bytes(b"x")
    out = ns["cache_tar_out"](local / "sft-shared", f"{shared}.tar", got["files"])
    assert out["written"] and out["files"] == 6 and not list(shared.parent.glob("*.part*"))
    env, use_local, got = ns["cache_env"](shared, tmp_path / "local2")
    assert use_local and got["files"] == 6 and got["src"].endswith("sft-shared.tar")
    assert (tmp_path / "local2/sft-shared/inductor/new.bin").read_bytes() == b"x"


def test_curriculum_order_is_fixed_and_one_pass():
    """curriculum: the index's ``order`` (build.py priority.json, kept examples only) is the batch order every epoch,
    no shuffle and no longest-first; next_batch walks it from the cursor."""
    ex = [{"file": f"train/f{i}.pt", "reply_tokens": 10, "tokens": 100 + i} for i in range(6)]
    index = {"examples": ex, "order": ["train/f3.pt", "train/f0.pt", "train/f5.pt", "train/f1.pt"]}
    cfg = types.SimpleNamespace(seed=0, repeat_examples=0, batch_reply_tokens=20)
    assert train.pool(cfg, index, 0) == train.pool(cfg, index, 1) == index["order"]
    names, tokens, cur = train.next_batch(cfg, index, (0, 0))
    assert names == ["train/f3.pt", "train/f0.pt"] and tokens == 20 and cur == (0, 2)
    names, _, cur = train.next_batch(cfg, index, cur)
    assert names == ["train/f5.pt", "train/f1.pt"] and cur == (0, 4)
    assert train.next_batch(cfg, index, cur)[0] == ["train/f3.pt", "train/f0.pt"]  # a second pass repeats the order


def test_preemption_auto_resume_from_finalised_checkpoint(setup, tmp_path, monkeypatch):
    """The SFT training path under policy "resume" (30 Oct: k256 main preempted with its hourly checkpoint unfinalised):
    a preemption (signal recorded, attempt left open) with a finalised checkpoint resumes on Modal's auto-restart
    (same call id) from that checkpoint, never from an unfinalised .ckpt-N; no checkpoint, policy "stop", or a death
    without a signal stay blocked."""
    import shutil

    import ckpt
    from arc_infra.runops.restart import RestartBlocked, RestartGuard

    state, out = tmp_path / "state", tmp_path / "r"

    def guard(run="r", policy="resume"):
        return RestartGuard(state, run, policy=policy, call_id="call-1", beat_s=3600)

    g1 = guard()
    monkeypatch.setattr(g1, "end", lambda status="ok": None)  # the attempt dies: no clean end
    train.train(cfg(setup, out, stop_step=2, **SLOW_OFF), device="cpu", make_opt=opt, guard=g1)
    g1.record_preemption("test")
    assert (out / "ckpt-2" / "state.json").exists()
    shutil.copytree(out / "ckpt-2", out / ".ckpt-3")  # a save in flight at the preemption (state.json included)
    s = json.loads((out / ".ckpt-3" / "state.json").read_text())
    (out / ".ckpt-3" / "state.json").write_text(json.dumps({**s, "step": 3}))
    assert ckpt.latest(out) == out / "ckpt-2"
    train.train(cfg(setup, out, **SLOW_OFF), device="cpu", make_opt=opt, guard=guard())
    assert rows(out, "start")[-1]["resume"] == str(out / "ckpt-2")
    assert [r["step"] for r in rows(out, "step")] == [1, 2, 3]
    inc = json.loads((state / ".runops" / "r" / "attempts.json").read_text())["incidents"]
    assert len(inc) == 1 and inc[0]["resumed"] and inc[0]["cause"].startswith("preemption")

    for name, policy, ckpt_steps, signal_ in (("none", "resume", 0, True), ("stop", "stop", 2, True),
                                              ("nosig", "resume", 2, False)):
        o = tmp_path / name
        g = guard(name, policy)
        monkeypatch.setattr(g, "end", lambda status="ok": None)
        c = dataclasses.replace(cfg(setup, o, stop_step=2, stop_ckpt=bool(ckpt_steps), **SLOW_OFF),
                                ckpt_steps=ckpt_steps or 99)
        train.train(c, device="cpu", make_opt=opt, guard=g)
        if signal_:
            g.record_preemption("test")
        with pytest.raises(RestartBlocked):
            train.train(cfg(setup, o, **SLOW_OFF), device="cpu", make_opt=opt, guard=guard(name, policy))


def test_checkpoint_commit_survives_a_refused_commit(monkeypatch):
    """A commit refused with "too many layers" waits and retries; any other error still surfaces."""
    import ckpt

    calls, rows = [], []

    def commit():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("volume commit: too many layers in volume. Please wait and retry.")

    saver = ckpt.Saver(types.SimpleNamespace(), commit, False, log=lambda **row: rows.append(row))
    saver.commit_retrying(delay=0.0)
    assert len(calls) == 3 and [r["attempt"] for r in rows] == [0, 1]
    saver.commit = lambda: (_ for _ in ()).throw(RuntimeError("other"))
    with pytest.raises(RuntimeError, match="other"):
        saver.commit_retrying(delay=0.0)


def test_batch_plan_step_order():
    plan = train.BatchPlan(100, 5)  # resumed at step 5: the next batch is step 6's
    plan.set(200, 8)
    assert [plan.take() for _ in range(4)] == [100, 100, 200, 200]


def test_batch_knob_flag_off_changes_nothing(setup):
    out = setup / "off"
    train.train(cfg(setup, out, **SLOW_OFF), device="cpu", make_opt=opt, knobs=lambda: json.dumps({"batch_reply_tokens": 10 ** 6}))
    assert not [r for r in rows(out, "batch_knob")]


def test_saver_copies_direct_states_so_the_step_does_not_wait_for_the_write(tmp_path, monkeypatch):
    """v3 2 Nov: the hourly async checkpoint held the next optimizer step 161 s while the paged host states were written
    to the volume. With host room the writer copies them first and the step waits only for that copy; without room
    it waits for the write as before. Either way the file holds the states as they were at the save."""
    import time

    import ckpt
    from safetensors.torch import load_file

    student = tmp_path / "student"
    student.mkdir()
    (student / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    host, w = torch.arange(1000, dtype=torch.float32), torch.ones(10)
    plain = ckpt.write_safetensors

    def slow(path, items, *a, **k):
        if Path(path).name == "optim.safetensors":
            time.sleep(1.5)
        return plain(path, items, *a, **k)

    monkeypatch.setattr(ckpt, "write_safetensors", slow)
    for room in (True, False):
        out = tmp_path / f"out-{room}"
        out.mkdir()
        s = ckpt.Saver(types.SimpleNamespace(out=str(out), student=str(student), keep_ckpts=2), async_=True)
        s.direct = lambda t: t is host
        s._collect = lambda *a: ([("w", w)], [("state", host)], {"skeleton": 1})
        s._copy_room = lambda n, room=room: room
        want = host.clone()
        s.save(None, None, None, 1, (0, 0))
        began = time.time()
        s.before_update()
        waited = time.time() - began
        host.add_(1)  # the optimizer step
        s.wait()
        assert torch.equal(load_file(str(out / "ckpt-1" / "optim.safetensors"))["state"], want)
        assert (waited < 1.0) if room else (waited >= 1.0), (room, waited)
        host.sub_(1)
