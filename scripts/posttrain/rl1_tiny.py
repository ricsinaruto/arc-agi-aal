"""A tiny Flash-shaped rl1_trainer (the real qwen4_exp classes at 4 layers, peft LoRA, fp32 on CPU) answering the
loop's commands through the trainer's own ``run_op``: (1), no restart from a trainer-code bug again.
The CI tests drive it through rl_seq.train_step (``Tiny.call``); the CPU preflight runs it behind the stand-in
(arena.fake_trainer, ARC_PREFLIGHT_TINY_TRAINER=1) in its own process (``serve``, on the trainer's python, which has
torch) on the loop's own commands. Windows are synthetic, keyed by the
spec's name; with ``served`` every window resolves its routes through a real served_route store on disk.

Patches module globals of rl1_trainer (fq.positions_and_images, the attention forward, torch.cuda.synchronize): use
it in its own process or inside test_rl1_dist.isolated()."""
import json
import os
import sys
import traceback
import zlib
from pathlib import Path
from types import SimpleNamespace

V, H = 96, 32
TARGETS = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a",
           "linear_attn.out_proj", "shared_expert.up_proj", "shared_expert.down_proj", "shared_expert.gate_proj",
           "q_proj", "k_proj", "v_proj", "o_proj"]
LR = 1e-3
LAYERS = [0, 1, 2, 3]  # every decoder layer of the tiny model routes (MoE)
K = 2  # num_experts_per_tok
EXPERTS = 8


# trainer arguments whose op_paths branch loads a Triton kernel module (rl1_gdn, rl1_sparse2, rl1_hyper, rl1_dequant,
# rl1_experts, rl1_topk) or needs a card (ngram_pinned: the pinned host n-gram lookup): off on CPU; every other
# argument of the run is sent as the run sends it
KERNEL_ARGS = ("gdn_fast", "gdn_keep", "prefix_nograd", "hyper_fast", "hyper_xb", "fast_dequant", "fused_experts",
               "union_attention", "fast_topk", "fast_topk_wide", "ngram_pinned")


def cpu_args(args):
    """A run's trainer arguments (rl_loop.Loop.v7_trainer_args) for the tiny CPU trainer: Triton kernels off, the
    prefix chunk at the tiny model's 64."""
    return {**{k: v for k, v in args.items() if k not in KERNEL_ARGS}, "prefix_chunk": 64}


def shape_of(name):
    """A spec name's (window tokens, tail): fixed per name, 80-200 tokens, tail 40 to all of it."""
    h = zlib.crc32(str(name).encode())
    end = 80 + h % 121
    return end, 40 + (h >> 8) % (end - 39)


def window(name, end=None, tail=None):
    """The synthetic window of a spec name (test_rl1_dist's layout): ids, ~40 trained positions in the tail, signed
    weights, an engine reference, replies of 7 tokens."""
    import torch

    if end is None:
        end, tail = shape_of(name)
    seed = sum(map(ord, str(name)))
    ids = torch.randint(2, V, (1, end), generator=torch.Generator().manual_seed(seed))
    ids[0, ::37] = 1  # eos: n-gram resets
    g = torch.Generator().manual_seed(seed + 1)
    positions = torch.sort(torch.randperm(tail - 1, generator=g)[:min(40, tail - 1)] + end - tail + 1).values
    n = len(positions)
    arrays = {"positions": positions, "support": torch.full((n, 4), -1), "weights": torch.linspace(-1, 2, n),
              "reference": -torch.rand(n, generator=g) * 3, "reply": torch.arange(n) // 7}
    return {"name": name, "inputs": {"input_ids": ids}, "tail": tail, "arrays": arrays, "temperature": 0.7,
            "argmax": False, "engine_raw": [float("nan")] * n}


def write_record(folder, rid, ids):
    """One engine request file <rid>-xxxxxxxx.rt (served_route's format): a record from position 0 over ``ids``."""
    import numpy as np

    t = np.asarray(ids, dtype=np.int32)
    picks = np.stack([(np.arange(len(t))[:, None] + np.arange(K)[None, :] + layer) % EXPERTS
                      for layer in range(len(LAYERS))], axis=1).astype(np.int16)  # [T, L, K], distinct per row
    head = np.array([0, len(t), len(LAYERS), K], dtype=np.int32).tobytes()
    (Path(folder) / f"{rid}-abcdefgh.rt").write_bytes(head + t.tobytes() + picks.tobytes())
    (Path(folder) / f"{rid}-abcdefgh.mm.json").write_text(json.dumps({"lora": "policy"}))


def build(folder, feed=None):
    """The tiny Trainer (rl1_trainer.Trainer without its GPU load); ``feed`` replaces window_feed (default: the
    synthetic windows by name)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    import rl1_dist
    import rl1_encode
    import rl1_memory
    import rl1_trainer as rt

    torch.set_num_threads(1)
    torch.cuda.synchronize = lambda *a, **k: None
    if not torch.cuda.is_available():
        rt.CARD = "cpu"  # served routes and other card tensors land on the host
    rt.fq.positions_and_images = lambda model, inputs, **kw: (lambda start, stop: {})
    rt.selected_attention()
    rt.SPARSE["mode"], rt.TIES["mode"] = "selected", "index"
    lin, full = "linear_attention", "full_attention"
    config = Qwen4ExpTextConfig(
        vocab_size=V, hidden_size=H, num_hidden_layers=4, layer_types=[lin, lin, full, lin],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, partial_rotary_factor=0.25,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        moe_intermediate_size=16, shared_expert_intermediate_size=16, num_experts=EXPERTS, num_experts_per_tok=K,
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
    t.encode_errors = []

    def plain(specs_, cmd, **kw):
        todo = list(specs_)
        return (lambda: window(todo.pop(0)["name"])), (lambda: None), feed_stats(len(todo))

    t.window_feed = feed or plain
    return t


def feed_stats(n):
    """window_feed's counters as the real feed returns them (the call's ``encode`` row reads them)."""
    return {"waits": 0.0, "encoded": n, "cache_hits": 0, "held": 0, "ahead_hits": 0, "ahead_sent": 0}


class Tiny:
    """The tiny trainer behind the loop's command interface: ``call(op, **cmd)`` runs ``run_op`` (the serve loop's
    error capture, save joins, unpark, op logging) on rl1_trainer's own ops. Specs need only "name" (and "w"; 1.0 if
    absent). ``served``: R3 on, a record per window in ``folder``/served-route. ``trainer.log``: every op's event."""

    PASS = ("accumulate", "old_pass", "step", "update", "score", "checkpoint", "checkpoint_wait", "load_checkpoint",
            "park", "unpark", "reset", "route", "save_join", "save", "exit")

    def __init__(self, folder, served=False):
        import rl1_trainer as rt

        self.folder = Path(folder)
        self.route = self.folder / "served-route"
        self.route.mkdir(parents=True, exist_ok=True)
        (self.route / "layers.json").write_text(json.dumps({"layers": LAYERS}))
        self.rows, self.index, self.log = {}, 0, self.folder / "trainer.log"
        self.t = build(self.folder / "t", feed=self.feed)
        self.t.rec = SimpleNamespace(event=self.event)
        self.served = served
        if served:
            assert self.t.op_route({"router": "served"})["router"] == "served"
        self.rt = rt

    def event(self, kind, **kw):
        with self.log.open("a") as f:
            f.write(json.dumps({"event": kind, **kw}, default=str) + "\n")

    def feed(self, specs_, cmd, **kw):
        todo = list(specs_)

        def nxt():
            spec = todo.pop(0)
            made = window(spec["name"])
            made["key"] = self.rt.window_key(spec, made)
            if not self.served:
                return made
            rid = f"chatcmpl-{abs(hash(spec['name'])) % 10**8}"
            if spec["name"] not in self.rows:
                write_record(self.route, rid, made["inputs"]["input_ids"][0].tolist())
                self.rows[spec["name"]] = [{"n": 0, "rid": rid, "keep": 0, "messages": [{}]}]
            made["replies"] = [{"request": 0}]
            return self.t.finish_window({**spec, "served_route": str(self.route)}, made, self.rows[spec["name"]])
        return nxt, (lambda: None), feed_stats(len(todo))

    @staticmethod
    def specs_of(attempts):
        return [{**a, "name": a.get("name") or Path(str(a.get("rollout") or "x")).parent.name or "x",
                 "w": float(a.get("w") if a.get("w") is not None else 1.0)} for a in attempts or []]

    def call(self, op, **cmd):
        if op not in self.PASS:
            raise ValueError(f"tiny trainer: op {op} not taken")
        cmd = {**cmd, "op": op}
        if "attempts" in cmd:
            cmd["attempts"] = self.specs_of(cmd["attempts"])
        cmd.pop("ahead", None)  # the encoders' look-ahead: the synthetic feed encodes nothing
        cmd.setdefault("lr", LR)
        result = self.t.run_op(cmd, self.index)
        self.index += 1
        if op == "route" and result.get("ok"):  # the loop turned R3 on: the next windows resolve through the store
            self.served = cmd.get("router") == "served"
        return result

    async def acall(self, op, **cmd):
        """The loop's ``trainer.call`` for tests: a failed op raises with the trainer's own traceback."""
        result = self.call(op, **cmd)
        if not result.get("ok") and result.get("traceback"):
            raise RuntimeError(f"tiny trainer {op}: {result.get('error')}\n{result['traceback']}")
        return result


def serve(root):
    """The stand-in's child: one JSON op per stdin line, one JSON answer per stdout line. Stdout is kept for the answers
    (import and trainer prints go to stderr). An op that raises answers ok False with its traceback."""
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    # the trainer's venv has fla, whose GDN kernels need a card (N1t: Triton "0 active drivers"): the modeling code's
    # reference path, as in CI, whose image has no fla. None: `import fla` raises and find_spec answers None.
    sys.modules["fla"] = None
    try:
        import served_route  # noqa: F401 - the bundle's copy beside rl1_encode
    except ImportError:
        from arena import served_route

        sys.modules["served_route"] = served_route
    tiny = Tiny(root)
    for line in sys.stdin:
        req = json.loads(line)
        try:
            answer = tiny.call(req["op"], **cpu_args(req["args"]))
        except Exception as error:  # noqa: BLE001 - the command's answer
            answer = {"ok": False, "error": repr(error)[:2000], "traceback": traceback.format_exc()[-4000:]}
        out.write(json.dumps(answer, default=str) + "\n")
        if req["op"] == "exit":
            return


if __name__ == "__main__":
    serve(sys.argv[2])
