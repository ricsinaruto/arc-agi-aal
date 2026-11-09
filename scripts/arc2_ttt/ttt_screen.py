"""TTT screen worker: per-task LoRA on Flash with the competition weights, run by
``modal_ttt_screen.py`` in the Axolotl venv. ``--config`` JSON: mode (flash | tiny), model_dir, mask, tasks_dir,
roster, parity, out, lrs, epochs, augments, max_tokens, rank, deadline_epoch, reply_tasks, limit.

1. Load: flash = the RL trainer's load (rl0 ``flash_qlora_job.load``, NVFP4 container, routed experts frozen, LoRA on
   GDN / attention / shared expert) + the n-gram table scale + the k256 router mask; tiny = a 2-layer random Qwen3
   with the same tokenizer (CPU preflight: every path but the Flash load).
2. Parity: the served log-probs of 5 true answers (``arc2.nll_parity`` on the RTX) against this forward on the same
   token ids; the trainer's own encode must give the served ids. flash: a fail stops the run (exit 3).
3. Per task: base read, then per lr a fresh adapter trained on ``arc2.ttt_data.samples`` and the adapted read
   (teacher-forced NLL + greedy-exact of every test output, view 0); a greedy reply on the first ``reply_tasks``.
4. summary.json: the pre-registered bars per lr arm.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import ttt_core as tc  # noqa: E402

PARITY_MEAN, PARITY_TASK = 0.02, 0.05  # nats/token: mean |delta| over all answer tokens; per task
PASS_MISS, FAIL_MISS, NLL_SHARE, CONTROL_LOST = 6, 3, 0.70, 1  # of 37 miss outputs (screen_tasks.json)


def gpu():
    try:
        line = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        util, used = (float(x) for x in line.split(","))
        return {"gpu.util": util, "gpu.mem_used_gib": round(used / 1024, 1)}
    except Exception:  # noqa: BLE001 - CPU preflight
        return {}


class Live:
    def __init__(self, out):
        self.path, self.state, self.lock = Path(out) / "metrics.jsonl", {"ttt.tasks_done": 0, "ttt.phase": "load"}, \
            threading.Lock()
        self.stop = threading.Event()
        threading.Thread(target=self.beat, daemon=True).start()

    def write(self, **extra):
        with self.lock:
            self.state.update(extra)
            flat = {**self.state, **gpu()}  # the runops row format: numbers under "m", text as a note (r8 watcher)
            row = {"t": round(time.time(), 1), "m": {k: v for k, v in flat.items() if isinstance(v, (int, float))},
                   "note": " ".join(f"{k}={v}" for k, v in flat.items() if not isinstance(v, (int, float)))}
            with self.path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")

    def beat(self):
        while not self.stop.wait(30):
            self.write()


def load_flash(cfg, out):
    import flash_qlora_job as fq
    import rl1_trainer as rt
    import transformers.modeling_utils as modeling_utils

    fq.set_adapter(int(cfg["rank"]), False)
    fq.NVFP4.update(on=True, experts=False, backend="sonicmoe")
    fq.CUT_CROSS_ENTROPY["on"] = False
    modeling_utils.caching_allocator_warmup = lambda *a, **k: None  # as rl1_trainer: no BF16-size pre-reservation
    rec = fq.Recorder(Path(out) / "trainer")
    model, tok, loaded = fq.load(rec, cfg["model_dir"], Path(out) / "trainer", sequence_len=int(cfg["max_tokens"]) + 64)
    loaded = {k: loaded.get(k) for k in ("seconds", "trainable", "params_total", "adapter_groups", "linear_layers",
                                         "experts_module", "peft", "gpu_allocated_gib")}
    loaded["table_scale"] = fq.apply_table_scale(rec, model, cfg["model_dir"])
    loaded["expert_mask"] = rt.install_expert_mask(model, cfg["mask"])
    from axolotl.integrations.kernels.libs.sonicmoe import set_merge_aware_enabled

    set_merge_aware_enabled(False)  # plain LoRA over the dequantised base, as the server computes
    if not hasattr(tok, "apply_chat_template") or getattr(tok, "chat_template", None) is None:
        from transformers import AutoProcessor

        tok = getattr(tok, "tokenizer", tok)
        tok.chat_template = AutoProcessor.from_pretrained(cfg["model_dir"]).chat_template
    return model, tok, loaded


def load_tiny(cfg):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(cfg["model_dir"])
    if getattr(tok, "chat_template", None) is None:
        from transformers import AutoProcessor

        tok.chat_template = AutoProcessor.from_pretrained(cfg["model_dir"]).chat_template
    conf = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                       max_position_embeddings=int(cfg["max_tokens"]) + 64)
    base = Qwen3ForCausalLM(conf).to(torch.bfloat16)
    base.model.layers[0].mlp.gate_proj.float()  # an FP32 weight in a BF16 model, as Flash's routers (screen r2)
    model = get_peft_model(base, LoraConfig(r=int(cfg["rank"]), lora_alpha=2 * int(cfg["rank"]),
                                                              lora_dropout=0.0,
                                                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # FP32 adapter, as the Flash load
    try:  # screen r2's failure class, reproduced: a plain forward over mixed BF16/FP32 weights fails
        model(input_ids=torch.tensor([[1, 2, 3]]))
        mixed = "no error"
    except RuntimeError as exc:
        mixed = f"RuntimeError: {exc}"[:200]
    return model, tok, {"tiny": True, "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                        "plain_forward_mixed_dtype": mixed}


def real_block(model, tok, cfg, loaded):
    """Layer 0's MoE block with the real weights (FP32 router + k256 mask, NVFP4 routed experts, LoRA'd shared expert):
    forward under the worker's autocast, backward to the adapter; dtypes of every tensor group; the trainer's encode on
    a real task."""
    import torch

    from arc2 import ttt_data

    dtypes = {}
    for name, p in model.named_parameters():
        group = ("router" if name.endswith("mlp.gate.weight") else "lora" if "lora_" in name else
                 "experts" if ".mlp.experts." in name else "other")
        dtypes.setdefault(group, set()).add(str(p.dtype))
    mlp = next(m for n, m in model.named_modules() if n.endswith("layers.0.mlp"))
    hidden = int(model.config.text_config.hidden_size) if hasattr(model.config, "text_config") else \
        int(model.config.hidden_size)
    x = torch.randn(1, 64, hidden, dtype=torch.bfloat16)
    out = {"loaded_mask": loaded.get("expert_mask"), "dtypes": {k: sorted(v) for k, v in dtypes.items()}}
    try:
        with tc.autocast(model):
            y = mlp(x)
        y = y[0] if isinstance(y, tuple) else y
        loss = y.float().pow(2).mean()
        loss.backward()
        grads = [p.grad for n, p in model.named_parameters() if "lora_" in n and "layers.0.mlp" in n]
        out.update(forward_ok=bool(torch.isfinite(y).all()), out_dtype=str(y.dtype),
                   lora_grads=len(grads), grads_finite=all(g is not None and bool(torch.isfinite(g).all())
                                                           for g in grads))
    except Exception as exc:  # noqa: BLE001 - recorded
        out["error"] = f"{type(exc).__name__}: {exc}"[:500]
    try:
        with torch.no_grad():
            mlp(x)
        out["plain_forward"] = "no error"
    except Exception as exc:  # noqa: BLE001 - screen r2's class without autocast
        out["plain_forward"] = f"{type(exc).__name__}: {exc}"[:200]
    rows = json.loads(Path(cfg["parity"]).read_text())["rows"]
    task = json.loads((Path(cfg["tasks_dir"]) / f"{rows[0]['task']}.json").read_text())
    q = ttt_data.queries(task, rows[0]["task"])[0]
    mine = tc.encode(tok, q["messages"], q["target"])
    out["ids_same"] = mine is not None and mine[0] == rows[0]["full_ids"]
    out["pass"] = bool(out.get("forward_ok") and out.get("grads_finite") and out.get("lora_grads") and out["ids_same"])
    print("REAL_BLOCK " + json.dumps(out, default=str), flush=True)
    return out


def load(cfg, out):
    """The model for ``cfg["mode"]``: flash (the B300 screen), real-cpu (the preflight's real-weights block) or tiny.
    r3 ran parity on the tiny model after the Flash load (a dangling else): the class is now checked."""
    mode = cfg["mode"]
    if mode == "real-cpu":  # the Flash load on the host (as rl1_trainer's host load), no CUDA calls
        import torch
        import flash_qlora_job as fq

        fq.DEVICE_MAP = {"": "cpu"}
        torch.cuda.synchronize = lambda *a, **k: None
        torch.cuda.memory_allocated = torch.cuda.memory_reserved = lambda *a, **k: 0
        config = fq.config
        fq.config = lambda *a, **k: {**config(*a, **k), "bf16": True}  # "auto" picks FP16 on a CPU host
    if mode in ("flash", "real-cpu"):
        model, tok, loaded = load_flash(cfg, out)
    elif mode == "tiny":
        model, tok, loaded = load_tiny(cfg)
    else:
        raise ValueError(f"mode {mode!r}")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    loaded["model_class"] = type(base).__name__
    want = "Qwen3ForCausalLM" if mode == "tiny" else "Qwen4Exp"
    if want not in loaded["model_class"]:
        raise RuntimeError(f"mode {mode}: loaded {loaded['model_class']}, want {want}")
    print("MODEL " + json.dumps({"mode": mode, "class": loaded["model_class"]}), flush=True)
    return model, tok, loaded


def checkpointing(model):
    try:
        if not getattr(model, "is_gradient_checkpointing", False):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        return True
    except Exception as exc:  # noqa: BLE001 - recorded
        return f"{type(exc).__name__}: {exc}"[:200]


def parity(model, tok, cfg, ttt_data, task_of):
    rows = json.loads(Path(cfg["parity"]).read_text())["rows"]
    out, deltas = [], []
    for row in rows:
        q = ttt_data.queries(task_of(row["task"]), row["task"])[0]
        mine = tc.encode(tok, q["messages"], q["target"])
        n = len(row["prompt_ids"])
        labels = [-100] * n + row["full_ids"][n:]
        got = tc.read(model, [(row["full_ids"], labels)], detail=True)[0]
        d = [abs(a - b) for a, b in zip(got["logprobs"], row["logprobs"])]
        deltas += d
        out.append({"task": row["task"], "same_ids": mine is not None and mine[0] == row["full_ids"],
                    "tokens": len(d), "mean_abs": round(sum(d) / len(d), 5), "max_abs": round(max(d), 4),
                    "nll_trainer": got["nll"], "nll_served": round(row["nll"], 5)})
    mean = sum(deltas) / max(1, len(deltas))
    # gate = ids + the mean bar (after r4: the per-task outlier fits the served INT4 KV; base and
    # adapted are both read on the trainer); the per-task bar is recorded only
    ok = bool(out) and all(r["same_ids"] for r in out) and mean <= PARITY_MEAN
    return {"pass": ok, "mean_abs": round(mean, 5), "tokens": len(deltas), "rows": out,
            "per_task_pass": all(r["mean_abs"] <= PARITY_TASK for r in out),
            "bars": {"mean": PARITY_MEAN, "task": PARITY_TASK, "same_ids": True, "gate": "ids + mean"}}


def pad_id(tok):
    for v in (getattr(tok, "pad_token_id", None), getattr(tok, "eos_token_id", None)):
        if v is not None:
            return int(v)
    return 0


def warmup(model, tok, cfg, roster, ttt_data, task_of, sizes=(1, 2, 4, 8)):
    """r6 warm-up: packed loss = the single-sample loss (|d| < 1e-3, else stop);
    real tok/s and peak memory per batch size on the first task's samples; the batch with the best real tok/s under
    90% of the card; then the largest of (2 lrs) > (1 lr = 3e-4) whose estimated time for the whole roster fits the
    time left before the deadline (load, parity and this leg already spent), else stop (no trimming here)."""
    import torch

    plan = [x for x in order(roster, cfg.get("kinds", ("miss", "hit")))
            if x[0] not in set(cfg.get("skip") or ())][:int(cfg.get("limit") or 10 ** 6)]
    t0 = plan[0][0]
    enc = [e for e in (tc.encode(tok, x["messages"], x["target"], int(cfg["max_tokens"]))
                       for x in ttt_data.samples(task_of(t0), t0, augments=int(cfg["augments"]), max_tokens=10 ** 9)) if e]
    tc.reset(model, seed=0)
    pid = pad_id(tok)
    with torch.no_grad():
        single = float(tc.loss_of(model, *enc[0]))
        packed = float(tc.batch_loss(model, [enc[0]], pid)[0])
        pair = float(tc.batch_loss(model, [enc[0], enc[1 % len(enc)]], pid)[0])
    out = {"task": t0, "samples": len(enc), "equivalence": {"single": round(single, 5), "packed": round(packed, 5),
                                                             "pair": round(pair, 5)}}
    if abs(single - packed) >= 1e-3:
        return {**out, "decision": "stop", "why": f"packed loss {packed} != single {single}"}
    bench = {}
    fixed = cfg.get("fixed")  # r7: r6's measured batch and pace, no bench
    for size in ([] if fixed else sizes):
        try:
            bench[size] = tc.bench(model, enc, [size], pid)[size]
        except torch.cuda.OutOfMemoryError as exc:
            bench[size] = {"oom": str(exc)[:200]}
            torch.cuda.empty_cache()
            break
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if torch.cuda.is_available() else None
    ok = {k: v for k, v in bench.items() if "real_tok_s" in v and (total is None or v["peak_gib"] <= 0.9 * total)}
    if fixed:
        size, rate, epochs = int(fixed["batch_size"]), float(fixed["rate"]), int(cfg["epochs"])
    elif not ok:
        return {**out, "bench": bench, "decision": "stop", "why": "no batch size fits"}
    else:
        size = max(ok, key=lambda k: ok[k]["real_tok_s"])
        rate = ok[size]["real_tok_s"]
        epochs = 2 if size >= 4 else 1
    tokens, reads = 0, 0
    for t, _ in plan:
        task = task_of(t)
        tokens += sum(x["tokens"] for x in ttt_data.samples(task, t, augments=int(cfg["augments"]), max_tokens=10 ** 9))
        reads += sum(len(q["messages"][1]["content"]) + len(q.get("target", "")) for q in ttt_data.queries(task, t))
    left = float(cfg.get("end_epoch") or cfg["deadline_epoch"]) - time.time()  # to the run's own end (r7)
    options = ([{"lrs": cfg["lrs"], "epochs": epochs}] if fixed else
               [{"lrs": cfg["lrs"], "epochs": epochs}, {"lrs": [3e-4], "epochs": epochs}, {"lrs": [3e-4], "epochs": 1}])
    est = []
    for o in options:
        arms = len(o["lrs"])
        secs = tokens * o["epochs"] * arms / rate + reads * (1 + arms) / (3 * rate)  # a read ~ a third of fwd+bwd
        est.append(round(secs))
        if secs <= left:
            chosen = {**o, "batch_size": size}
            return {**out, "bench": bench, "rate": rate, "left_s": round(left), "estimates_s": est,
                    "plan_task_s": round(secs / len(plan), 1), "chosen": chosen, "decision": "go",
                    "roster_tokens": tokens}
    return {**out, "bench": bench, "rate": rate, "left_s": round(left), "estimates_s": est, "roster_tokens": tokens,
            "decision": "stop", "why": "even 1 lr x 1 epoch does not fit the time left"}


def order(roster, kinds=("miss", "hit")):
    tasks = [(t, k) for k in kinds for t in roster[k]]
    return sorted(tasks, key=lambda x: hashlib.sha256(("ttt-order-v1" + x[0]).encode()).hexdigest())


def summary(rows, roster, lrs, needs_nll=True):
    miss_out = {t: set(v) for t, v in roster["miss_outputs"].items()}
    arms = {}
    for lr in lrs:
        key = f"{lr:g}"
        miss_exact = sum(1 for r in rows for q in r["queries"] if q["index"] in miss_out.get(r["task"], ())
                         and q["arms"][key]["exact"])
        base_miss_exact = sum(1 for r in rows for q in r["queries"] if q["index"] in miss_out.get(r["task"], ())
                              and q["base"]["exact"])
        lower = [sum(q["arms"][key]["nll"] for q in r["queries"]) < sum(q["base"]["nll"] for q in r["queries"])
                 for r in rows if r["queries"]]
        lost = sum(1 for r in rows if r["kind"] == "hit" for q in r["queries"]
                   if q["base"]["exact"] and not q["arms"][key]["exact"])
        gained_hit = sum(1 for r in rows if r["kind"] == "hit" for q in r["queries"]
                         if q["arms"][key]["exact"] and not q["base"]["exact"])
        share = sum(lower) / max(1, len(lower))
        verdict = ("pass" if miss_exact >= PASS_MISS and (not needs_nll or share >= NLL_SHARE) and lost <= CONTROL_LOST
                   else "fail" if miss_exact < FAIL_MISS else "inconclusive")
        arms[key] = {"miss_exact": miss_exact, "base_miss_exact": base_miss_exact, "nll_lower_share": round(share, 3),
                     "controls_lost": lost, "controls_gained": gained_hit, "verdict": verdict}
    read = {(r["task"], q["index"]) for r in rows for q in r["queries"]}
    return {"tasks_read": len(rows), "miss_outputs_read": sum(1 for t, ks in miss_out.items() for k in ks
                                                               if (t, k) in read),
            "miss_outputs_total": sum(len(v) for v in miss_out.values()), "arms": arms,
            "bars": {"pass_miss": PASS_MISS, "fail_miss": FAIL_MISS, "nll_share": NLL_SHARE,
                     "controls_lost": CONTROL_LOST}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = json.loads(Path(ap.parse_args().config).read_text())
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    from arc2 import ttt_data

    live = Live(out)
    began = time.time()
    model, tok, loaded = load(cfg, out)
    if cfg["mode"] == "real-cpu":  # the GDN layers run fla's Triton kernels (GPU only): check the MoE block
        (out / "real-block.json").write_text(json.dumps(real_block(model, tok, cfg, loaded), indent=1, default=str))
        print("END", flush=True)
        return
    loaded["checkpointing"] = checkpointing(model)
    loaded["load_s"] = round(time.time() - began, 1)
    (out / "loaded.json").write_text(json.dumps(loaded, indent=1, default=str))
    live.write(**{"ttt.phase": "parity", "ttt.load_s": loaded["load_s"]})

    def task_of(t):
        return json.loads((Path(cfg["tasks_dir"]) / f"{t}.json").read_text())

    par = parity(model, tok, cfg, ttt_data, task_of) if cfg.get("parity") else None
    (out / "parity.json").write_text(json.dumps(par, indent=1))
    print("PARITY " + json.dumps({k: v for k, v in (par or {}).items() if k != "rows"}), flush=True)
    if cfg["mode"] == "flash" and not (par and par["pass"]):
        live.write(**{"ttt.phase": "parity_failed"})
        print("END parity failed", flush=True)
        sys.exit(3)
    roster = json.loads(Path(cfg["roster"]).read_text())
    if cfg.get("warmup"):
        live.write(**{"ttt.phase": "warmup"})
        warm = warmup(model, tok, cfg, roster, ttt_data, task_of)
        (out / "warmup.json").write_text(json.dumps(warm, indent=1, default=str))
        print("WARMUP " + json.dumps(warm, default=str), flush=True)
        if warm["decision"] == "stop":
            live.write(**{"ttt.phase": "warmup_stop"})
            print("END warm-up: " + warm["why"], flush=True)
            sys.exit(4)
        cfg.update(warm["chosen"])
        live.write(**{"ttt.plan_task_s": warm["plan_task_s"], "ttt.batch": cfg["batch_size"]})
    lrs = [float(x) for x in cfg["lrs"]]
    rows, tokens_trained, train_s = [], 0, 0.0
    results = (out / "tasks.jsonl").open("a")
    plan = [x for x in order(roster, cfg.get("kinds", ("miss", "hit")))  # a follow-up run skips tasks read elsewhere
            if x[0] not in set(cfg.get("skip") or ())][:int(cfg.get("limit") or 10 ** 6)]
    stopped = None
    for i, (t, kind) in enumerate(plan):
        if time.time() > float(cfg["deadline_epoch"]):
            stopped = f"deadline before task {i} of {len(plan)}"
            break
        live.write(**{"ttt.phase": "task", "ttt.task": t})
        task = task_of(t)
        enc = [e for e in (tc.encode(tok, s["messages"], s["target"], int(cfg["max_tokens"]))
                           for s in ttt_data.samples(task, t, augments=int(cfg["augments"]), max_tokens=10 ** 9)) if e]
        qs = ttt_data.queries(task, t)
        qenc = [tc.encode(tok, q["messages"], q["target"]) for q in qs]
        tc.reset(model, seed=0)
        base = tc.read(model, qenc)
        row = {"task": t, "kind": kind, "samples": len(enc), "sample_tokens": sum(len(e[0]) for e in enc),
               "queries": [{"index": q["test_index"], "tokens": len(e[0]), "base": b} for q, e, b in zip(qs, qenc, base)],
               "train": {}}
        for lr in lrs:
            key = f"{lr:g}"
            tc.reset(model, seed=0)
            t0 = time.time()
            losses = tc.train(model, enc, epochs=int(cfg["epochs"]), lr=lr, batch_size=int(cfg.get("batch_size", 1)),
                              pad_id=pad_id(tok)) if enc else []
            secs = time.time() - t0
            tokens_trained += row["sample_tokens"] * int(cfg["epochs"])
            train_s += secs
            adapted = tc.read(model, qenc)
            for q, a in zip(row["queries"], adapted):
                q.setdefault("arms", {})[key] = a
            row["train"][key] = {"steps": len(losses), "seconds": round(secs, 1),
                                 "loss_first": round(losses[0], 4) if losses else None,
                                 "loss_last": round(losses[-1], 4) if losses else None,
                                 "loss_last_epoch_mean": round(sum(losses[-len(enc):]) / len(enc), 4) if enc else None}
            if i < int(cfg.get("reply_tasks", 0)):
                try:
                    r = tc.reply(model, *qenc[0])
                    row.setdefault("reply", {})[key] = {**{k: v for k, v in r.items() if k != "tokens"},
                                                        "teacher_forced_exact": adapted[0]["exact"]}
                except Exception as exc:  # noqa: BLE001 - recorded
                    row.setdefault("reply", {})[key] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        rows.append(row)
        results.write(json.dumps(row) + "\n")
        results.flush()
        live.write(**{"ttt.tasks_done": len(rows), "ttt.train_tok_s": round(tokens_trained / max(train_s, 1e-6), 1)})
        print(json.dumps({"task": t, "kind": kind, "samples": len(enc), "train": row["train"],
                          "base_nll": [q["base"]["nll"] for q in row["queries"]],
                          "arms": {k: [q["arms"][k]["nll"] for q in row["queries"]] for k in row["train"]},
                          "exact": {k: [q["arms"][k]["exact"] for q in row["queries"]] for k in row["train"]}}),
              flush=True)
    result = {"summary": summary(rows, roster, lrs, bool(cfg.get("pass_needs_nll", True))), "parity": {k: v for k, v in (par or {}).items() if k != "rows"},
              "stopped": stopped, "tasks_planned": len(plan), "train_tokens": tokens_trained,
              "train_tok_s": round(tokens_trained / max(train_s, 1e-6), 1), "elapsed_s": round(time.time() - began, 1),
              "settings": {k: cfg[k] for k in ("lrs", "epochs", "augments", "max_tokens", "rank")}}
    (out / "summary.json").write_text(json.dumps(result, indent=1))
    live.write(**{"ttt.phase": "done"})
    live.stop.set()
    print("SUMMARY " + json.dumps(result), flush=True)
    print("END", flush=True)


if __name__ == "__main__":
    main()
