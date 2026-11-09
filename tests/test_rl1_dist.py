"""rl1_dist (RL C2): the trainer on 3 gloo ranks through its own command channel (rank 0 serves the cmd dir): a
data-parallel + sequence-parallel accumulate (one rank with no data-parallel window), the step, park / unpark /
checkpoint / exit; gradients, loss and entropy against one process running every window. A follower's failure is
rank 0's answer, not a hang. Tiny Flash-shaped model (real qwen4_exp classes, peft LoRA), fp32 on CPU."""
import contextlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
pytest.importorskip("peft")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "posttrain"))

from arc_infra.runops.multirank import run_ranks  # noqa: E402

V, H = 96, 32
TARGETS = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a",
           "linear_attn.out_proj", "shared_expert.up_proj", "shared_expert.down_proj", "shared_expert.gate_proj",
           "q_proj", "k_proj", "v_proj", "o_proj"]
# name -> (window tokens, tail); "big" (tail 170 > fit 150) runs sequence-parallel, "a" and "b" data-parallel (LPT:
# a -> rank 0, b -> rank 1, rank 2 none)
WINDOWS = {"big": (190, 170), "a": (120, 60), "b": (100, 100)}
FIT = 150
LR = 1e-3


def window(name):
    end, tail = WINDOWS[name]
    seed = sum(map(ord, name))
    ids = torch.randint(2, V, (1, end), generator=torch.Generator().manual_seed(seed))
    ids[0, ::37] = 1  # eos: n-gram resets
    g = torch.Generator().manual_seed(seed + 1)
    positions = torch.sort(torch.randperm(tail - 1, generator=g)[:min(40, tail - 1)] + end - tail + 1).values
    n = len(positions)
    arrays = {"positions": positions, "support": torch.full((n, 4), -1), "weights": torch.linspace(-1, 2, n),
              "reference": -torch.rand(n, generator=g) * 3, "reply": torch.arange(n) // 7}
    return {"name": name, "inputs": {"input_ids": ids}, "tail": tail, "arrays": arrays, "temperature": 0.7,
            "argmax": False, "engine_raw": [float("nan")] * n}


def specs():
    return [{"name": n, "w": 1.0, "est_tokens": e, "est_tail": t} for n, (e, t) in WINDOWS.items()]


ISOLATED = {"depth": 0}


@contextlib.contextmanager
def isolated():
    """trainer() patches module globals (rl1_trainer SPARSE/TIES modes, fq.positions_and_images, torch.cuda.synchronize,
    the attention forward); a test that calls it in its own process runs inside this, which puts them back. Without it
    a later test in the same worker ran against the stubs (CI 41a5b554a: host_index KeyError 'position_ids')."""
    import rl1_trainer as rt
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    # every modeling class's own attributes: the trainer's patches replace methods and set install flags
    # (_rl1_selected, _rl1_route, _rl1_vectorized); a forward put back under a flag left set would never re-patch
    classes = [c for c in vars(modeling).values() if isinstance(c, type) and c.__module__ == modeling.__name__]
    snap = {c: dict(vars(c)) for c in classes}
    kept = (rt.SPARSE["mode"], rt.TIES["mode"], rt.fq.positions_and_images, torch.cuda.synchronize, rt.CARD,
            dict(rt.ROUTE))
    ISOLATED["depth"] += 1
    try:
        yield
    finally:
        ISOLATED["depth"] -= 1
        rt.SPARSE["mode"], rt.TIES["mode"], rt.fq.positions_and_images, torch.cuda.synchronize, rt.CARD, route = kept
        rt.ROUTE.clear(), rt.ROUTE.update(route)  # rl1_tiny's served Tiny turns R3 on
        for c, before in snap.items():
            for name in [n for n in vars(c) if n not in before]:
                delattr(c, name)
            for name, value in before.items():
                if vars(c).get(name) is not value and name not in ("__dict__", "__weakref__"):
                    setattr(c, name, value)


def isolate(test):
    """A test that builds trainer() in its own process: run it inside isolated()."""
    import functools

    @functools.wraps(test)
    def wrapped(*args, **kwargs):
        with isolated():
            return test(*args, **kwargs)

    return wrapped


def trainer(folder, fail_rank=None):
    """A Trainer without the GPU load: the tiny model with LoRA, the trainer's own ops; window encoding replaced by
    the fixed windows above. Only in a spawned rank (run_ranks) or inside isolated(): it patches module globals."""
    import multiprocessing

    if multiprocessing.parent_process() is None and not ISOLATED["depth"]:
        raise RuntimeError("test_rl1_dist.trainer() patches module globals: call it in a spawned rank or inside "
                           "isolated()")
    from peft import LoraConfig, get_peft_model
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    import rl1_dist
    import rl1_encode
    import rl1_memory
    import rl1_trainer as rt

    torch.set_num_threads(1)
    torch.cuda.synchronize = lambda *a, **k: None
    rt.fq.positions_and_images = lambda model, inputs, **kw: (lambda start, stop: {})
    rt.selected_attention()
    rt.SPARSE["mode"], rt.TIES["mode"] = "selected", "index"
    lin, full = "linear_attention", "full_attention"
    config = Qwen4ExpTextConfig(
        vocab_size=V, hidden_size=H, num_hidden_layers=4, layer_types=[lin, lin, full, lin],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, partial_rotary_factor=0.25,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        moe_intermediate_size=16, shared_expert_intermediate_size=16, num_experts=8, num_experts_per_tok=2,
        hc_count=4, hc_lowrank=8, ple_embed_dim=32, ple_layer_ids=[2], ngram_vocab_size_base=101,
        split_ngram_parts=4, indexer_n_heads=2, indexer_head_dim=8, indexer_kv_heads=1, indexer_budget=8,
        indexer_compress_ratio=4, eos_token_id=1, bos_token_id=1, pad_token_id=None, output_gate_type="sigmoid")
    torch.manual_seed(0)
    base = modeling.Qwen4ExpForCausalLM(config).float()
    with torch.no_grad():
        for param in base.parameters():
            if param.dim() > 1:
                param.normal_(0, 0.08)
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TARGETS))
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=g) * 0.05)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rt.set_checkpointing(base, "full")
    model.train()
    t = rt.Trainer.__new__(rt.Trainer)
    t.dist = rl1_dist.init()
    t.args = SimpleNamespace(tis_cap=2.0, window=4096, optimizer="adamw32", lr=LR, model=None)
    t.dir = Path(folder)
    (t.dir / "cmd").mkdir(parents=True, exist_ok=True)
    (t.dir / "done").mkdir(parents=True, exist_ok=True)
    t.model, t.policy, t.loaded = model, {}, {}
    t.rec = SimpleNamespace(event=lambda *a, **k: None)
    t.head = rl1_memory.head_capture(model)
    t.prefix_chunk = t.path_chunk = 64
    t.snapshots, t.snap_keep = True, 3
    t.encoders, t.window_cache, t.references, t.updates = None, rl1_encode.Cache(), {}, 0
    t.parked, t.pinned = False, {}
    t.params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    t.optimizer = t.make_optimizer(LR)
    t.expand_cmd = lambda cmd: [dict(a) for a in cmd["attempts"]]
    rank = t.dist.rank if t.dist is not None else 0

    def feed(specs_, cmd, **kw):
        todo = list(specs_)

        def nxt():
            spec = todo.pop(0)
            if spec["name"] == "b" and rank == fail_rank:
                raise RuntimeError(f"injected failure on rank {rank}")
            return window(spec["name"])
        return nxt, (lambda: None), {"waits": 0.0}
    t.window_feed = feed
    ran = []
    t.op_park = lambda cmd=None: ran.append("park") or {"parked": True}
    t.op_unpark = lambda cmd=None: ran.append("unpark") or {"unparked": True}
    t.ran = ran
    return t


def grads_of(t):
    return [g.numpy() for g in t.ref["grads"]]


def reference():
    """One process (no RL1_DIST) running every window; the module settings it changes are put back after."""
    import tempfile

    with isolated(), tempfile.TemporaryDirectory() as folder:
        t = trainer(folder)
        acc = t.op_accumulate({"attempts": specs(), "lr": LR, "first": True})
        assert acc["ok"], acc
        step = t.op_step({"lr": LR, "ref": "keep"})
        assert step["ok"], step
        return grads_of(t), step


COMMANDS = [{"op": "accumulate", "lr": LR, "first": True, "fit_tokens": FIT},
            {"op": "step", "lr": LR, "ref": "keep", "dist_digest": True},
            {"op": "park"}, {"op": "unpark"}, {"op": "checkpoint"}, {"op": "exit"}]


def serve_rank(rank, world, arg):
    """One torchrun-style rank: the trainer's own serve loop over the cmd dir (rank 0 reads and answers)."""
    folder, fail_rank = arg
    os.environ["RL1_DIST"] = "1"
    os.environ["RL1_DIST_POLL_S"] = "0.05"
    t = trainer(folder, fail_rank)
    t.serve()
    return {"ran": t.ran, "grads": grads_of(t) if getattr(t, "ref", None) else None,
            "params": [p.detach().numpy().copy() for _, p in t.params[:4]]}


def write_cmds(folder):
    (Path(folder) / "cmd").mkdir(parents=True, exist_ok=True)
    for i, cmd in enumerate(COMMANDS):
        cmd = {**cmd, "attempts": specs()} if cmd["op"] == "accumulate" else dict(cmd)
        if cmd["op"] == "checkpoint":
            cmd["path"] = str(Path(folder) / "ckpt")
        (Path(folder) / "cmd" / f"{i:04d}.json").write_text(json.dumps(cmd))


def answers(folder):
    return [json.loads((Path(folder) / "done" / f"{i:04d}.json").read_text()) for i in range(len(COMMANDS))]


@pytest.mark.multirank
def test_dp_cp_accumulate_and_step_match_one_process(tmp_path):
    import numpy as np

    ref_grads, ref_step = reference()
    write_cmds(tmp_path)
    out = run_ranks(serve_rank, world=3, inputs=[(str(tmp_path), None)] * 3, timeout=240)
    acc, step, park, unpark, ckpt, done = answers(tmp_path)
    assert acc["ok"] and step["ok"], (acc.get("error"), step.get("error"))
    assert [r["dist_plan"] for r in acc["ranks"]] == [{"cp": 1, "dp": 1, "load": [120, 100, 0]},
                                                      {"cp": 1, "dp": 1, "load": [120, 100, 0]},
                                                      {"cp": 1, "dp": 0, "load": [120, 100, 0]}]
    assert sorted(w["name"] for w in acc["windows"]) == ["a", "b", "big"]
    assert acc["trained_tokens"] == ref_step["trained_tokens"]
    assert step["trained_tokens"] == ref_step["trained_tokens"] and step["dist_digest"]["equal"]
    assert abs(step["loss"] - ref_step["loss"]) <= 1e-5 * abs(ref_step["loss"])
    assert abs(step["grad_norm"] - ref_step["grad_norm"]) <= 1e-5 * ref_step["grad_norm"]
    assert abs(step["entropy"] - ref_step["entropy"]) <= 1e-5 * ref_step["entropy"]
    worst = 0.0
    for rank_out in out:
        assert len(rank_out["grads"]) == len(ref_grads)
        for got, want in zip(rank_out["grads"], ref_grads):
            scale = float(np.abs(want).max()) or 1.0
            worst = max(worst, float(np.abs(got - want).max()) / scale)
        for a, b in zip(rank_out["params"], out[0]["params"]):
            assert np.array_equal(a, b)  # the same step on every rank
    assert worst <= 1e-4, worst
    print(f"dist vs one process: LoRA grad max rel err {worst:.2e}, loss {step['loss']} vs {ref_step['loss']}")
    assert [r["ran"] for r in out] == [["park", "unpark"]] * 3  # park / unpark on every rank
    assert ckpt["ok"] and (tmp_path / "ckpt" / "master.safetensors").exists()
    assert [r["skipped"] for r in ckpt["ranks"]] == [False, True, True]  # checkpoint on rank 0 only
    assert done["ok"] and [r["skipped"] for r in done["ranks"]] == [False, False, False]


@pytest.mark.multirank
def test_follower_failure_is_rank0_answer(tmp_path):
    write_cmds(tmp_path)
    out = run_ranks(serve_rank, world=3, inputs=[(str(tmp_path), 1)] * 3, timeout=240)
    acc, step, *_, done = answers(tmp_path)
    assert not acc["ok"] and "rank 1" in acc["error"] and "injected failure on rank 1" in acc["error"]
    assert set(acc["rank_errors"]) == {"0", "1", "2"}  # (JSON keys) every rank stopped at the agreement, none waits in a collective
    assert not step["ok"] and done["ok"]  # the next ops still answer; exit stops every rank
    assert len(out) == 3


def test_plan_longest_first_cp_first():
    import rl1_dist

    est = [(500, 500), (100, 100), (300, 300), (90, 90), (80, 80)]
    p = rl1_dist.plan([{}] * 5, 3, 2, 400, 4096, estimates=est)
    assert p["cp"] == [0] and p["dp"] == {0: [2], 1: [1], 2: [3, 4]} and p["mine"] == [0, 3, 4]
    assert rl1_dist.plan([{}] * 2, 3, 2, 400, 4096, estimates=est[1:3])["mine"] == []  # an empty rank


def test_merge_sums_and_errors():
    import rl1_dist

    rs = [{"ok": True, "windows": [{"name": "x"}], "trained_tokens": 3, "seconds": 1.0},
          {"ok": False, "error": "boom", "windows": [], "trained_tokens": 2, "seconds": 2.0},
          {"ok": True, "skipped": "rank 0 only"}]
    m = rl1_dist.merge({"op": "accumulate"}, rs)
    assert not m["ok"] and m["error"] == "rank 1: boom" and m["trained_tokens"] == 5 and m["seconds"] == 2.0
    assert m["windows"] == [{"name": "x"}] and m["ranks"][2]["skipped"]
    reduced = rl1_dist.merge({"op": "step"}, [{**r, "dist_reduced": True, "ok": True} for r in rs[:2]])
    assert reduced["trained_tokens"] == 3 and reduced["windows"] == [{"name": "x"}]


def test_whole_segment_window_holds_the_first_trained_tokens_logit(monkeypatch):
    """T1a 26 Oct: margin 0 put the cut at the first trained token, whose logit sits one before it ("a trained token
    lies outside the gradient window"); whole segments now cut one earlier."""
    import rl1_encode
    import rl_job

    ids = torch.arange(20).view(1, -1)

    def encode(capture):
        return {"input_ids": ids.clone(), "labels": torch.full_like(ids, -100)}, 12

    _, tail, _ = rl_job.window_example(encode, [{"rollout": {"w": 1.0}}], 0, window=1 << 20, margin=1, floor=0,
                                       credited_only="scripted")
    assert (12 - (20 - tail) - 1) == 0  # keep of the first trained position
    seen = {}
    monkeypatch.setattr(rl_job, "attempt_example", lambda *a, **k: seen.update(k))
    monkeypatch.setattr(rl1_encode, "read_rows", lambda spec, cached: [])
    rl1_encode.encode_window(None, None, 32768, {"whole_segment": True, "margin": 0}, device="cpu")
    assert seen["margin"] == 1 and seen["floor"] == 0


def test_whole_segment_spec():
    import rl1_encode
    import rl1_trainer as rt

    spec = rt.whole_segment({"name": "t", "w": 0.5, "window": 8192, "credited_only": True, "margin": 1024})
    assert spec["whole"] and spec["whole_segment"] and spec["margin"] == 1 and spec["floor"] == 0
    assert spec["window"] == rt.WHOLE_SEGMENT_WINDOW and not spec["credited_only"]
    assert rl1_encode.reply_weight(spec, {"scripted": True}) is None  # the restored history ends the walk
    assert rl1_encode.reply_weight({"w": 0.5}, {"scripted": True}) == 0.0  # off: unchanged
    assert rl1_encode.reply_weight(spec, {"n": 3}) == 0.5


def test_trainer_command_torchrun():
    from arena import rl_loop

    cfg = rl_loop.DEFAULTS | {"mg": {"trainer_cards": 4}}
    command, env = rl_loop.trainer_command(cfg, "/src", "/work")
    assert command[1:5] == ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4"]
    assert command[5].endswith("rl1_trainer.py") and env["RL1_DIST"] == "1"
    assert env["RL1_ENCODE_PROCS"] == str(rl_loop.FAST["encode_procs"])
    one, env1 = rl_loop.trainer_command(rl_loop.DEFAULTS | {"mg": {}}, "/src", "/work")
    assert one[1] == "-u" and "RL1_DIST" not in env1


def warm_rank(rank, world, folder):
    """serve_rank with RL1_NCCL_WARM: the warm-up collective issued on a thread at the end of the load (as
    Trainer.__init__), then the same accumulate (one CP window, rank 2 without a DP window) and step."""
    os.environ["RL1_NCCL_WARM"] = "1"
    os.environ["RL1_DIST"] = "1"
    os.environ["RL1_DIST_POLL_S"] = "0.05"
    t = trainer(folder)
    started = t.dist.warm("cpu")
    t.serve()
    return {"started": started, "warmed": t.dist.warmed, "grads": grads_of(t) if getattr(t, "ref", None) else None}


@pytest.mark.multirank
def test_nccl_warm_runs_first_and_changes_nothing(tmp_path):
    import numpy as np

    ref_grads, ref_step = reference()
    write_cmds(tmp_path)
    out = run_ranks(warm_rank, world=3, inputs=[str(tmp_path)] * 3, timeout=240)
    acc, step, *_ = answers(tmp_path)
    assert acc["ok"] and step["ok"], (acc.get("error"), step.get("error"))
    assert all(o["started"] for o in out) and all(o["warmed"] and not o["warmed"]["error"] for o in out)
    assert step["dist_collectives_before"] >= 1 and step["nccl_warm"]["error"] is None
    assert set(step["step_parts"]) >= {"windows", "reduce_sums", "reduce_grads", "objects", "clip", "optimizer"}
    assert step["trained_tokens"] == ref_step["trained_tokens"] and step["dist_digest"]["equal"]
    for rank_out in out:
        for got, want in zip(rank_out["grads"], ref_grads):
            assert float(np.abs(got - want).max()) <= 1e-4 * (float(np.abs(want).max()) or 1.0)


@isolate
def test_async_adapter_save_serves_the_stepped_adapter(tmp_path, monkeypatch):
    """mg.async_adapter_save: the step returns before the files are written; once ``save_join`` returns, the folder
    holds exactly the stepped adapter (the same bytes as the synchronous save; the params' BF16 copy). A failed write
    leaves only ``.partial`` and raises at the join."""
    import time

    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file

    import rl1_trainer as rt

    monkeypatch.delenv("RL1_DIST", raising=False)
    kept = (rt.SPARSE["mode"], rt.TIES["mode"], rt.fq.positions_and_images, torch.cuda.synchronize)
    try:
        t = trainer(tmp_path / "t")
        assert t.op_accumulate({"attempts": specs(), "lr": LR, "first": True})["ok"]
        step = t.op_step({"lr": LR, "save": str(tmp_path / "v1"), "save_async": True})
        assert step["ok"] and step["adapter"]["pending"] and step["adapter"]["tensors"] == len(t.params)
        done = t.op_save_join({})
        assert done["pending"] and done["path"] == str(tmp_path / "v1") and not done.get("error")
        assert not (tmp_path / "v1.partial").exists() and t.op_save_join({}) == {"pending": False}
        sync = t.save_adapter(tmp_path / "sync")
        assert "pending" not in sync
        names = ("adapter_model.safetensors", "adapter_config.json", "adapter-tensors.json")
        for name in names:
            assert (tmp_path / "v1" / name).read_bytes() == (tmp_path / "sync" / name).read_bytes(), name
        served = load_file(str(tmp_path / "v1" / "adapter_model.safetensors"))
        want = {k: v.detach().to(torch.bfloat16) for k, v in get_peft_model_state_dict(t.model).items()}
        assert set(served) == set(want) and all(torch.equal(served[k], want[k]) for k in want)

        def broken(folder, state):
            time.sleep(0.2)
            raise OSError("disk full (test)")

        t.write_adapter = broken
        assert t.save_adapter(tmp_path / "v2", background=True)["pending"]
        with pytest.raises(RuntimeError, match="disk full"):
            t.save_join()
        assert not (tmp_path / "v2").exists() and (tmp_path / "v2.partial").exists()  # never named complete
        assert {"accumulate", "step", "load_checkpoint", "exit"} <= set(rt.SAVE_JOIN_OPS)
    finally:
        rt.SPARSE["mode"], rt.TIES["mode"], rt.fq.positions_and_images, torch.cuda.synchronize = kept


def test_nccl_warm_off_without_the_flag(monkeypatch):
    import rl1_dist

    monkeypatch.delenv("RL1_NCCL_WARM", raising=False)
    d = rl1_dist.Dist.__new__(rl1_dist.Dist)
    d.collectives, d.warming, d.warmed = 0, None, None
    assert d.warm("cpu") is False and d.ready() is None


# minibatches: an update's windows in chunks, one accumulate (first=True) + one optimizer step each;
# chunk 2 is one window, so two of the three ranks get no window in it (an empty rank)
MB_CHUNKS = [["a", "b"], ["big"], ["a"]]


def mb_commands():
    out = []
    for names in MB_CHUNKS:
        out.append({"op": "accumulate", "lr": LR, "first": True, "fit_tokens": FIT,
                    "attempts": [s for s in specs() if s["name"] in names]})
        out.append({"op": "step", "lr": LR, "ref": "keep", "dist_digest": True})
    return out + [{"op": "exit"}]


def mb_reference():
    import tempfile

    with isolated(), tempfile.TemporaryDirectory() as folder:
        t = trainer(folder)
        steps = []
        for cmd in mb_commands()[:-1]:
            got = (t.op_accumulate if cmd["op"] == "accumulate" else t.op_step)(dict(cmd))
            assert got["ok"], got
            if cmd["op"] == "step":
                steps.append(got)
        return steps, [p.detach().numpy().copy() for _, p in t.params[:4]]


def mb_rank(rank, world, folder):
    os.environ["RL1_DIST"] = "1"
    os.environ["RL1_DIST_POLL_S"] = "0.05"
    t = trainer(folder)
    t.serve()
    return {"params": [p.detach().numpy().copy() for _, p in t.params[:4]]}


@pytest.mark.multirank
def test_minibatch_steps_on_three_ranks_match_one_process(tmp_path):
    """mg.redesign.minibatches on rl1_dist: 3 accumulate + step pairs (asymmetric chunks, an empty rank in chunk 2)
    give each step's loss / grad norm and the final LoRA params of one process running the same sequence; every rank
    holds the same params."""
    import numpy as np

    ref_steps, ref_params = mb_reference()
    cmds = mb_commands()
    (tmp_path / "cmd").mkdir(parents=True, exist_ok=True)
    for i, cmd in enumerate(cmds):
        (tmp_path / "cmd" / f"{i:04d}.json").write_text(json.dumps(cmd))
    out = run_ranks(mb_rank, world=3, inputs=[str(tmp_path)] * 3, timeout=300)
    done = [json.loads((tmp_path / "done" / f"{i:04d}.json").read_text()) for i in range(len(cmds))]
    steps = [d for c, d in zip(cmds, done) if c["op"] == "step"]
    assert all(d["ok"] for d in done), [d.get("error") for d in done]
    loads = [r["dist_plan"]["dp"] for r in done[2]["ranks"]]  # chunk 2: one window ("big", sequence-parallel)
    assert len(loads) == 3
    for got, want in zip(steps, ref_steps):
        assert got["dist_digest"]["equal"] and got["trained_tokens"] == want["trained_tokens"]
        assert abs(got["loss"] - want["loss"]) <= 1e-5 * max(abs(want["loss"]), 1e-12)
        assert abs(got["grad_norm"] - want["grad_norm"]) <= 1e-5 * want["grad_norm"]
    for rank_out in out:
        for a, b in zip(rank_out["params"], out[0]["params"]):
            assert np.array_equal(a, b)
    worst = max(float(np.abs(a - b).max()) / (float(np.abs(b).max()) or 1.0) for a, b in zip(out[0]["params"], ref_params))
    assert worst <= 1e-4, worst
