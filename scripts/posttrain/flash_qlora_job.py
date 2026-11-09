"""Flash QLoRA trainer jobs (post-training pilot, 7 October), run inside the pinned Axolotl image by
``scripts/posttrain/modal_flash_qlora.py`` with the image's own Python (``/workspace/axolotl-venv``).

The model is loaded exactly as ``axolotl train`` would load it (``axolotl.train.setup_model_and_tokenizer`` on the
config below: routed experts NF4 on load, n-gram table in host RAM, LoRA on attention, Gated DeltaNet and the shared
expert; routed experts, router, QSA indexer, PLE, vision and MTP frozen), then measured with plain forward/backward
steps, so one load answers every sequence length.

- ``check`` (a small GPU): environment and imports, read speed of the checkpoint volume, and the whole ``smoke``
  procedure on a tiny random Qwen4-Exp model (the cheap test of this file before the B200 run);
- ``smoke`` (one B200): load time and memory, then a ladder of sequence lengths, each with two steps: finite loss,
  seconds, tokens/s, peak memory, gradient norm per adapter group; the ladder stops at the first length that does not
  fit. Then six optimizer steps on one sequence (the loss must fall) and the adapter saved. With ``--windows``, the
  cached-prefix step (``window_step``) for contexts that do not fit whole.
- ``parity`` (CPU): the tensors the adapter touches are stored in BF16 in the served checkpoint; this checks they are
  bit-identical to the training checkpoint's, which is what makes a zero adapter export exact.
- ``adapter`` (one B200) and ``export`` (CPU): the G2 pre-check. A short adapter on DEV-game text with its gain on
  other DEV games measured in the trainer, then the adapter merged into the served BF16 shards as an overlay folder;
  the served side is ``scripts/posttrain/build_nll_bench.py``. ``overlay`` (CPU) compares every tensor of the
  overlay's own shards with the served checkpoint.
- ``train`` (one B200): the adapter itself, one pass over a captured corpus (``build_corpus.py``) with the
  cached-prefix step, self-checked stop rules, checkpoints and live numbers in ``progress.json``.
- ``multi`` (several GPUs, the layers split after the load): what trains whole, and the cached-prefix gradient
  against the whole-context one on captured requests.
- ``contexts`` (a small GPU): captured served requests (``arena.level_bench`` fork capture) as training examples:
  token counts against the server's, reply-only labels, and images across the cached prefix's chunks on a tiny model.

Every event is one JSON line on stdout and in ``<out>/events.jsonl``; ``<out>/result.json`` is rewritten after each.
"""

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

TARGET_MODULES = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a",
                  "linear_attn.out_proj", "shared_expert.up_proj", "shared_expert.down_proj", "shared_expert.gate_proj",
                  "q_proj", "k_proj", "v_proj", "o_proj"]
GROUPS = (("gdn", "linear_attn."), ("attention", "self_attn."), ("shared_expert", "shared_expert."))
TARGET_TENSOR = re.compile(r"^model\.language_model\.layers\.\d+\.(linear_attn\.(in_proj_(qkv|z|b|a)|out_proj)"
                           r"|mlp\.shared_expert\.(up|down|gate)_proj|self_attn\.[qkvo]_proj)\.weight$")
# adapted layers the locked build runs through its INT8 dense kernels (server.dense_kernels.int8_weights)
INT8_SERVED = re.compile(r"linear_attn\.(in_proj_(qkv|z)|out_proj)|self_attn\.[qkvo]_proj")
START = time.time()


class Recorder:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.result = {"events": []}

    def event(self, kind, **data):
        row = {"t": round(time.time() - START, 1), "event": kind, **data}
        print(json.dumps(row), flush=True)
        with (self.out / "events.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")
        return row

    def save(self, **data):
        self.result.update(data)
        (self.out / "result.json").write_text(json.dumps({k: v for k, v in self.result.items() if k != "events"},
                                                         indent=1))


def group_of(name):
    return next((group for group, mark in GROUPS if mark in name), "other")


def config(model_dir, out, *, sequence_len):
    """The Axolotl config: ``examples/qwen3.8-flash-next/vision-qlora.yaml`` with the pilot's adapter (rank 16,
    alpha 32, no LoRA on the routed experts)."""
    settings = {
        "base_model": str(model_dir), "processor_type": "AutoProcessor",
        "plugins": ["axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"] * CUT_CROSS_ENTROPY["on"],
        "load_in_8bit": False, "load_in_4bit": True, "quantize_moe_experts": True, "ple_cpu_offload": True,
        "skip_prepare_dataset": True, "remove_unused_columns": False, "sample_packing": False,
        "datasets": [{"path": str(Path(out) / "unused.jsonl"), "type": "chat_template"}],
        "val_set_size": 0.0, "output_dir": str(Path(out) / "axolotl"),
        "adapter": "qlora", "sequence_len": sequence_len, "pad_to_sequence_len": False,
        "lora_r": ADAPTER["rank"], "lora_alpha": ADAPTER["alpha"], "peft_use_rslora": ADAPTER["rslora"],
        "lora_dropout": 0, "lora_target_modules": TARGET_MODULES,
        "lora_mlp_kernel": False, "lora_qkv_kernel": False, "lora_o_kernel": False,
        "gradient_accumulation_steps": 1, "micro_batch_size": 1, "num_epochs": 1, "optimizer": "adamw_torch_fused",
        "lr_scheduler": "cosine", "learning_rate": 2e-5, "bf16": "auto", "tf32": True,
        "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "attn_implementation": "sdpa", "warmup_ratio": 0.05, "saves_per_epoch": 1, "weight_decay": 0.0,
        # --bf16-linear: bitsandbytes converts no linear layer, so everything but the routed experts (NF4 by
        # quantize_moe_experts) is the served BF16 weight; the G2 pre-check put the trainer's base 0.095 NLL
        # above the served one with NF4 linear layers
        **({"bnb_config_kwargs": {"llm_int8_skip_modules": ["model", "lm_head"]}} if BF16_LINEAR["on"] else {}),
    }
    if NVFP4["on"]:
        # the served checkpoint itself: Axolotl's nvfp4-lora.yaml. The routed experts stay NVFP4
        # (SonicMoE; the dequant backend is W4A16 like the serving kernels), everything else is its BF16 weight.
        for key in ("load_in_4bit", "load_in_8bit", "quantize_moe_experts", "bnb_config_kwargs"):
            settings.pop(key, None)
        # Axolotl allows ple_cpu_offload only with bitsandbytes (it is about accelerate's placement, which this job
        # does not use); keep_table_in_host_ram() already leaves the table in host RAM at load
        settings.update(adapter="lora", strict=False, device_map={"": 0}, ple_cpu_offload=False,
                        **{f"use_{NVFP4['backend']}": True},
                        nvfp4_merge_aware=bool(NVFP4["experts"]),
                        plugins=settings["plugins"] + ["axolotl.integrations.kernels.KernelsPlugin"])
        if NVFP4["experts"]:  # adapters on the routed experts, trained through the NVFP4 quantizer
            settings["lora_target_parameters"] = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    return settings


def set_adapter(rank, rslora):
    """Rank and scaling of the adapter. The update is scaled by 2 either way: alpha = 2 x rank, or with rank-stabilized
    scaling (alpha / sqrt(rank)) alpha = 2 x sqrt(rank), which keeps the update's size from shrinking as the rank grows."""
    ADAPTER.update(rank=rank, rslora=bool(rslora), alpha=2 * rank if not rslora else round(2 * rank ** 0.5, 4))


def host_gib():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1)


def read_bytes():
    try:
        return int(next(line for line in Path("/proc/self/io").read_text().splitlines()
                        if line.startswith("rchar")).split()[1])
    except Exception:
        return 0


def heartbeat(rec, stop, label, seconds=30):
    import torch

    def run():
        last = read_bytes()
        while not stop.wait(seconds):
            now = read_bytes()
            rec.event("heartbeat", phase=label, gpu_gib=round(torch.cuda.memory_allocated() / 2 ** 30, 1),
                      host_peak_gib=host_gib(), read_gib=round(now / 2 ** 30, 1),
                      read_gib_s=round((now - last) / 2 ** 30 / seconds, 2))
            last = now

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def environment(rec):
    import importlib

    import torch

    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "axolotl", "fla", "cut_cross_entropy",
                 "torchvision", "triton"):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "present")
        except Exception as error:  # the report is the point
            versions[name] = f"MISSING {type(error).__name__}: {error}"
    importlib.import_module("transformers.models.qwen4_exp.modeling_qwen4_exp")
    gpu = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = {"name": torch.cuda.get_device_name(0), "total_gib": round(total / 2 ** 30, 1),
               "free_gib": round(free / 2 ** 30, 1), "capability": list(torch.cuda.get_device_capability(0))}
    memory = {line.split(":")[0]: int(line.split()[1]) // 2 ** 20 for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith(("MemTotal", "MemAvailable"))}
    return rec.event("environment", versions=versions, cuda=torch.cuda.is_available(), gpu=gpu, host_gib=memory,
                     cpus=os.cpu_count())


def keep_table_in_host_ram():
    """transformers joins the n-gram table's shards on CPU and then moves the result to the device map's device.
    Axolotl's single-GPU map makes that the GPU (check v2: the table on cuda:0 and Axolotl's warning that
    ``ple_cpu_offload`` found nothing in host RAM), which is 95 GiB a B200 does not have during the load."""
    import transformers.core_model_loading as loading

    original = loading.get_device
    if getattr(original, "table_in_host_ram", False):
        return

    def get_device(device_map, name, *args, **kwargs):
        if "ngram_embedding." in name:
            return "cpu"
        return original(device_map, name, *args, **kwargs)

    get_device.table_in_host_ram = True
    loading.get_device = get_device


def spread(model, gpus):
    """The decoder layers in equal runs over ``gpus`` devices, after a one-device load: a multi-device map makes
    accelerate carry the host-RAM n-gram table to a GPU at every forward. A layer with a PLE table stays on the first
    device. Each layer gets its inputs on its own device; the last stage brings the hidden state back."""
    import torch

    core = (model.get_base_model() if hasattr(model, "get_base_model") else model).model.language_model
    per = -(-len(core.layers) // gpus)

    def mover(device):
        def move(value):
            if torch.is_tensor(value):
                return value.to(device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if type(value) in (list, tuple):
                return type(value)(move(item) for item in value)
            return value  # the cache: its entries are made on the layer's device

        def hook(module, args, kwargs):
            # the layer's kernels (Triton / CUTLASS / custom ops) launch on the current device's stream; without this
            # they ran on cuda:0's stream over cuda:1 tensors and the next cross-device copy raced them (illegal memory
            # access without CUDA_LAUNCH_BLOCKING; teacher check v3, 20 Oct)
            torch.cuda.set_device(device)
            return move(args), move(kwargs)

        return hook

    placed = []
    for index, layer in enumerate(core.layers):
        device = f"cuda:{0 if getattr(layer, 'ple', None) is not None else index // per}"
        if device != "cuda:0":
            layer.to(device)
            for module in layer.modules():  # bitsandbytes keeps a parametrized weight's state outside the module
                state = getattr(module, "quant_state", None)
                if state is not None and hasattr(state, "to"):
                    state.to(device)
        layer.register_forward_pre_hook(mover(device), with_kwargs=True)
        placed.append(int(device[-1]))
    core.hyper_connection_mixer.register_forward_pre_hook(mover("cuda:0"), with_kwargs=True)
    torch.cuda.empty_cache()
    return placed


def device_gib(peak=True):
    import torch

    read = torch.cuda.max_memory_allocated if peak else torch.cuda.memory_allocated
    return [round(read(i) / 2 ** 30, 2) for i in range(torch.cuda.device_count())]


def reset_peaks():
    import torch

    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)


def load(rec, model_dir, out, *, sequence_len):
    import torch
    import yaml
    from axolotl.cli.config import load_cfg
    from axolotl.train import setup_model_and_tokenizer

    path = Path(out) / "config.yaml"
    path.write_text(yaml.safe_dump(config(model_dir, out, sequence_len=sequence_len), sort_keys=False))
    cfg = load_cfg(str(path))
    keep_table_in_host_ram()
    stop = threading.Event()
    heartbeat(rec, stop, "load")
    began = time.time()
    try:
        model, tokenizer, peft_config, processor = setup_model_and_tokenizer(cfg)
    finally:
        stop.set()
    torch.cuda.synchronize()
    groups, devices, trainable, total = {}, {}, 0, 0
    for name, param in model.named_parameters():
        total += param.numel()
        devices[str(param.device)] = devices.get(str(param.device), 0) + param.numel()
        if param.requires_grad:
            trainable += param.numel()
            groups.setdefault(group_of(name), {"tensors": 0, "params": 0})
            groups[group_of(name)]["tensors"] += 1
            groups[group_of(name)]["params"] += param.numel()
    table = [str(p.device) for n, p in model.named_parameters() if n.endswith("ngram_embedding.weight")]
    experts = next(((n, type(m).__name__) for n, m in model.named_modules() if n.endswith("mlp.experts")), None)
    kinds = {}
    for name, module in model.named_modules():  # which linear layers bitsandbytes holds in 4 bits
        if isinstance(module, torch.nn.Linear) and "lora_" not in name:
            kinds[type(module).__name__] = kinds.get(type(module).__name__, 0) + 1
    expert_casts = cast_expert_inputs(model)
    row = rec.event("loaded", seconds=round(time.time() - began, 1), linear_layers=kinds, expert_casts=expert_casts,
                    gpu_allocated_gib=round(torch.cuda.memory_allocated() / 2 ** 30, 2),
                    gpu_reserved_gib=round(torch.cuda.memory_reserved() / 2 ** 30, 2), host_peak_gib=host_gib(),
                    params_by_device=devices, params_total=total, trainable=trainable, adapter_groups=groups,
                    ngram_table_on=table, experts_module=experts,
                    peft=None if peft_config is None else {"r": peft_config.r, "alpha": peft_config.lora_alpha})
    return model, tokenizer, row


def cast_expert_inputs(model):
    """BF16 teacher (Stage A smoke, 20 Oct): a float32 residual stream reaches the routed experts and SonicMoE's BF16
    grouped GEMM refuses it ("Type must match: Float32 != BFloat16"); NVFP4 experts cast inside their kernel. A
    pre-hook casts the experts' float32 hidden states to BF16 when the expert weights are BF16 (no packed 4-bit or
    FP8 tensor in the module); routing weights and indices pass unchanged. Returns the number of hooked modules."""
    import torch

    packed = {torch.uint8, torch.int8, getattr(torch, "float8_e4m3fn", torch.uint8)}

    def pre(module, args, kwargs):
        if args and torch.is_tensor(args[0]) and args[0].dtype == torch.float32:
            args = (args[0].to(torch.bfloat16), *args[1:])
        hidden = kwargs.get("hidden_states")
        if torch.is_tensor(hidden) and hidden.dtype == torch.float32:
            kwargs = {**kwargs, "hidden_states": hidden.to(torch.bfloat16)}
        return args, kwargs

    hooked = 0
    modules = dict(model.named_modules())
    for name, module in modules.items():
        if not name.endswith("mlp.experts"):
            continue
        dtypes = {p.dtype for p in module.parameters()}
        if torch.bfloat16 in dtypes and not dtypes & packed:
            # the MoE block too: the kernels plugin's SonicMoE forward may read the expert weights without calling
            # the experts module
            for target in (module, modules.get(name.rsplit(".", 1)[0])):
                if target is not None:
                    target.register_forward_pre_hook(pre, with_kwargs=True)
            hooked += 1
    return hooked


BF16_LINEAR = {"on": False}
CUT_CROSS_ENTROPY = {"on": True}  # off: the plain forward, which calls the model's loss_function with the logits
NVFP4 = {"on": False, "experts": False, "backend": "sonicmoe"}  # --nvfp4 / --expert-lora / --moe-backend
TEXT = {"path": None, "ids": None}  # --text: a real context (replies and tool outputs of one DEV game)


def token_ids(tokenizer, length, *, salt=0):
    """Tokens of this length. With ``--text``, the start of that text: a model that has seen the file's own source
    once copies its repeats (loss 0 from 16K tokens on), which is fine for memory and speed and useless for comparing
    gradients. Otherwise this file's own source, repeated with a counter line."""
    if TEXT["path"]:
        if TEXT["ids"] is None:
            TEXT["ids"] = tokenizer(Path(TEXT["path"]).read_text(), add_special_tokens=False)["input_ids"]
        if len(TEXT["ids"]) < length:
            raise ValueError(f"the text holds {len(TEXT['ids'])} tokens, {length} asked")
        return TEXT["ids"][:length]
    source = Path(__file__).read_text()
    pieces, count = [], 0
    while count < length:
        ids = tokenizer(f"# part {salt}.{len(pieces)}\n{source}", add_special_tokens=False)["input_ids"]
        pieces.append(ids)
        count += len(ids)
    return [token for ids in pieces for token in ids][:length]


def batch(tokenizer, length, *, salt=0, supervised=0.5):
    import torch

    ids = torch.tensor([token_ids(tokenizer, length, salt=salt)], device="cuda:0")
    labels = ids.clone()
    labels[:, : int(length * (1 - supervised))] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def step(model, inputs):
    """One forward and backward; returns the loss and timings. The caller owns zero_grad and the optimizer."""
    import torch

    torch.cuda.synchronize()
    began = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**inputs).loss
    torch.cuda.synchronize()
    forward = time.time() - began
    loss.backward()
    torch.cuda.synchronize()
    return float(loss.detach().float()), forward, time.time() - began - forward


def gradients(model):
    import torch

    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = groups.setdefault(group_of(name), {"tensors": 0, "no_grad": 0, "nonfinite": 0, "zero": 0, "sq": 0.0})
        group["tensors"] += 1
        if param.grad is None:
            group["no_grad"] += 1
            continue
        grad = param.grad.detach().float()
        if not bool(torch.isfinite(grad).all()):
            group["nonfinite"] += 1
            continue
        square = float((grad * grad).sum())
        group["sq"] += square
        group["zero"] += square == 0.0
    return {name: {"tensors": g["tensors"], "no_grad": g["no_grad"], "nonfinite": g["nonfinite"],
                   "zero": int(g["zero"]), "norm": round(g["sq"] ** 0.5, 6)} for name, g in groups.items()}


def ladder(rec, model, tokenizer, lengths, *, deadline):
    import torch

    rows, last = [], None
    for length in lengths:
        if last is not None:
            projected = 2.2 * last["seconds"] * (length / last["tokens"]) ** 2
            if time.time() + projected > deadline:
                rows.append(rec.event("rung_skipped", tokens=length, reason="time",
                                      projected_seconds=round(projected)))
                break
        row = {"tokens": length}
        try:
            inputs = batch(tokenizer, length, salt=length)
            for attempt in range(2):  # the first step also compiles kernels
                model.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                loss, forward, backward = step(model, inputs)
                row.update(loss=round(loss, 4), finite=loss == loss and abs(loss) != float("inf"),
                           forward_seconds=round(forward, 2), backward_seconds=round(backward, 2),
                           seconds=round(forward + backward, 2), tokens_per_second=round(length / (forward + backward)),
                           peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                           peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2 ** 30, 2),
                           host_peak_gib=host_gib(), steps=attempt + 1)
                if attempt == 0:
                    row["first_step_seconds"] = round(forward + backward, 2)
            row["gradients"] = gradients(model)
            row["fits"] = True
        except torch.OutOfMemoryError as error:
            row.update(fits=False, error=str(error)[:300],
                       peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
        model.zero_grad(set_to_none=True)
        inputs = None
        torch.cuda.empty_cache()
        rows.append(rec.event("rung", **row))
        rec.save(ladder=rows)
        if not row["fits"]:
            break
        last = row
    return rows


def overfit(rec, model, tokenizer, *, length, steps=6, lr=2e-4):
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    inputs = batch(tokenizer, length, salt=-1)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = step(model, inputs)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        losses.append(round(loss, 4))
    model.zero_grad(set_to_none=True)
    return rec.event("overfit", tokens=length, lr=lr, losses=losses, falls=losses[-1] < losses[0])


def read_only(cache):
    """Make the cache's update calls pure for the window pass: each returns what the layer needs (the prefix state
    joined with the new states) and stores nothing. A gradient-checkpointed layer runs twice, and the stock updates
    append or overwrite in place, so the second run would see its own first run."""
    import torch

    layers = cache.layers

    def update(key_states, value_states, layer_idx, *args, **kwargs):
        layer = layers[layer_idx]
        return torch.cat([layer.keys, key_states], dim=-2), torch.cat([layer.values, value_states], dim=-2)

    def update_indexer(states, layer_idx):
        return torch.cat([layers[layer_idx].indexer_keys, states], dim=1)

    def update_conv_state(conv_states, layer_idx, state_idx=0, **kwargs):
        layer = layers[layer_idx]
        if not layer.has_previous_state[state_idx]:
            raise RuntimeError(f"layer {layer_idx} has no prefix state {state_idx}")
        return torch.cat([layer.conv_states[state_idx], conv_states], dim=-1)

    def update_recurrent_state(states, layer_idx, state_idx=0, **kwargs):
        return states

    cache.update, cache.update_indexer = update, update_indexer
    cache.update_conv_state, cache.update_recurrent_state = update_conv_state, update_recurrent_state
    return cache


def served_messages(messages):
    """Chat messages as the served template gets them from vLLM: tool arguments parsed, ``reasoning`` under the
    template's name, images as placeholders (returned apart, in order)."""
    import base64
    import io

    from PIL import Image

    out, images = [], []
    for message in messages:
        message = {key: value for key, value in message.items() if value is not None}
        if isinstance(message.get("content"), list):
            parts = []
            for part in message["content"]:
                if part.get("type") == "image_url":
                    raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                    images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                    part = {"type": "image"}
                parts.append(part)
            message["content"] = parts
        if message.get("tool_calls"):
            message["tool_calls"] = [{**call, "function": {**call["function"], "arguments": json.loads(
                call["function"]["arguments"])}} for call in message["tool_calls"]]
        if "reasoning" in message:
            message["reasoning_content"] = message.pop("reasoning")
        out.append(message)
    return out, images


def encode_request(processor, capture):
    """A captured request (``arena.level_bench`` fork capture: the served payload and the recorded reply) as one
    training example: the prompt exactly as the server tokenized it, then the reply; the loss is on the reply only
    (through its end-of-turn token). ``checks`` compares the counts with the server's own."""
    import torch

    payload, tokenizer = capture["payload"], processor.tokenizer
    settings = dict(tools=payload.get("tools"), tokenize=False, **(payload.get("chat_template_kwargs") or {}))
    messages, images = served_messages(payload["messages"])
    reply, _ = served_messages([{**capture["reply"], "role": "assistant"}])
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **settings)
    whole = tokenizer.apply_chat_template(messages + reply, add_generation_prompt=False, **settings)
    end = "<|im_end|>"
    if not whole.startswith(prompt) or not whole.rstrip("\n").endswith(end):
        raise ValueError("the reply does not render as a continuation of the served prompt")
    whole = whole[: whole.rindex(end) + len(end)]

    def encode(text):
        return processor(text=[text], images=images or None, return_tensors="pt")

    head, inputs = encode(prompt), encode(whole)
    count, ids = head["input_ids"].shape[1], inputs["input_ids"]
    labels = ids.clone()
    labels[:, :count] = -100
    usage = capture.get("usage") or {}
    checks = {"request": capture.get("request"), "prompt_tokens": count, "served_prompt_tokens": capture["prompt_tokens"],
              "reply_tokens": ids.shape[1] - count, "served_reply_tokens": usage.get("completion_tokens"),
              "images": len(images), "image_tokens": int((ids == processor.image_token_id).sum()),
              "prefix_kept": bool(torch.equal(ids[:, :count], head["input_ids"]))}
    return {**inputs, "labels": labels}, checks


def chain_examples(encode, captures, *, window, margin=8192, floor=16384):
    """Training examples of one chain (``build_corpus.chain_ids``: targets of one game whose served prompts extend
    one another), from its end: the last target's prompt and reply, with the loss also on every earlier reply of
    the chain that starts inside the gradient ``window``. Such a reply sits in the later prompt exactly as it was
    generated; that is checked token by token against its own encoding, and a reply that fails it, or lies outside
    the window, ends the next example instead. ``encode(capture)`` gives ``(inputs, prompt tokens)`` or None for a
    target that is left out. Returns ``[(inputs, tail, targets)]``; ``tail`` is the gradient window of the
    example: its replies plus ``margin`` tokens of context, at least ``floor``, at most ``window``."""
    rows = [(capture, done) for capture, done in ((c, encode(c)) for c in captures) if done is not None]
    out = []
    while rows:
        _, (inputs, start) = rows.pop()
        ids, labels = inputs["input_ids"], inputs["labels"]
        total, first, targets = ids.shape[1], start, 1
        if total - start + 1 > window:  # the reply alone is longer than the window: left out, never truncated
            continue
        while rows:
            earlier_ids, earlier_start = rows[-1][1][0]["input_ids"], rows[-1][1][1]
            end = earlier_ids.shape[1]
            if total - earlier_start + 1 > window or end > first or not bool((ids[:, :end] == earlier_ids).all()):
                break
            labels[:, earlier_start:end] = ids[:, earlier_start:end]
            first, targets = earlier_start, targets + 1
            rows.pop()
        out.append((inputs, min(total, window, max(floor, total - first + margin)), targets))
    return out


def positions_and_images(model, inputs, host_index=False):
    """What a pass over a slice of the sequence needs besides its tokens: M-RoPE positions of the whole sequence
    (the model's own incremental rule only knows the images of its first call) and the image features, computed once
    (the vision tower is frozen). ``piece(start, stop)`` gives the keyword arguments of that slice; an image may
    straddle slices. ``host_index``: the per-token image count on the host (one read here, none per slice)."""
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    core = (model.get_base_model() if hasattr(model, "get_base_model") else model).model
    ids = inputs["input_ids"]
    if inputs.get("pixel_values") is None:
        positions = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(3, ids.shape[0], -1)
        return lambda start, stop: {"position_ids": positions[:, :, start:stop]}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        features = torch.cat(core.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"],
                                                     return_dict=True).pooler_output, dim=0)
    positions, _ = core.get_rope_index(ids, mm_token_type_ids=inputs["mm_token_type_ids"],
                                       image_grid_thw=inputs["image_grid_thw"])
    before = torch.nn.functional.pad((ids[0] == core.config.image_token_id).cumsum(0), (1, 0))
    if host_index:
        before = before.cpu()

    def piece(start, stop):
        extra = {"position_ids": positions[:, :, start:stop]}
        first, last = int(before[start]), int(before[stop])
        if last > first:
            extra["mm_encoder_outputs"] = {"image": BaseModelOutputWithPooling(pooler_output=(features[first:last],))}
        return extra

    return piece


def window_step(model, inputs, *, window, chunk, backward=True):
    """Long contexts on one GPU: the tokens before the last ``window`` go through the model without gradients, in
    chunks, into its cache (as serving does); the window is then one forward and backward against that cache. The
    gradient does not flow into the prefix's representations. Returns loss, prefix, forward and backward seconds."""
    import torch
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling

    modeling.Qwen4ExpTextDecoderLayer._can_checkpoint_with_cache = True  # the window pass only reads the cache
    ids, labels = inputs["input_ids"], inputs["labels"]
    total = ids.shape[1]
    cut = total - window
    piece = positions_and_images(model, inputs)
    torch.cuda.synchronize()
    began, cache = time.time(), None
    model.eval()  # in training mode a checkpointed layer drops the cache it is asked to fill
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, cut, chunk):
                stop = min(start + chunk, cut)
                output = model(input_ids=ids[:, start:stop], attention_mask=torch.ones_like(ids[:, :stop]),
                               past_key_values=cache, use_cache=True, **piece(start, stop))
                cache = output.past_key_values
                del output
    finally:
        model.train()
    if cache.get_seq_length() != cut:
        raise RuntimeError(f"the cache holds {cache.get_seq_length()} tokens, not the {cut} of the prefix")
    torch.cuda.synchronize()
    prefix = time.time() - began
    began = time.time()
    if not backward:  # evaluation: the tail without gradients, the loss only
        model.eval()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids[:, cut:], attention_mask=torch.ones_like(ids), labels=labels[:, cut:],
                             past_key_values=read_only(cache), use_cache=False, **piece(cut, total)).loss
        finally:
            model.train()
        return float(loss.float()), prefix, time.time() - began, 0.0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids=ids[:, cut:], attention_mask=torch.ones_like(ids), labels=labels[:, cut:],
                     past_key_values=read_only(cache), use_cache=False, **piece(cut, total)).loss
    torch.cuda.synchronize()
    forward = time.time() - began
    loss.backward()
    torch.cuda.synchronize()
    return float(loss.detach().float()), prefix, forward, time.time() - began - forward


def flat_gradients(model):
    import torch

    groups = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            groups.setdefault(group_of(name), []).append(param.grad.detach().float().flatten().cpu())
    return {name: torch.cat(parts) for name, parts in groups.items()}


def windows(rec, model, tokenizer, *, compare, pairs, chunk, deadline):
    """The window step against the plain step at a length that fits whole (same loss expected; the gradient cosine
    says what cutting the gradient at the window costs), then (context, window) pairs beyond what fits whole."""
    import torch

    comparisons = []
    for total, window in compare:
        inputs = batch(tokenizer, total, salt=total + 1)
        inputs["labels"][:, : total - window + 1] = -100  # the window's first token has no target in the window pass
        model.zero_grad(set_to_none=True)
        full_loss, _, _ = step(model, inputs)
        full = flat_gradients(model)
        model.zero_grad(set_to_none=True)
        loss, _, _, _ = window_step(model, inputs, window=window, chunk=chunk)
        cut = flat_gradients(model)
        model.zero_grad(set_to_none=True)
        cosine = {name: round(float(torch.nn.functional.cosine_similarity(full[name], cut[name], dim=0)), 4)
                  for name in full}
        ratio = {name: round(float(cut[name].norm() / full[name].norm().clamp_min(1e-30)), 4) for name in full}
        same = abs(loss - full_loss) <= 0.02 * max(abs(full_loss), 1e-3) + 2e-3
        comparisons.append(rec.event(
            "window_compare", tokens=total, window=window, loss_full=round(full_loss, 5), loss_window=round(loss, 5),
            same_loss=same, gradient_cosine=cosine, norm_ratio=ratio,
            full_norm={name: round(float(full[name].norm()), 6) for name in full}))
        rec.save(window_compare=comparisons)
        del full, cut, inputs
        torch.cuda.empty_cache()
    compared = {"same_loss": all(row["same_loss"] for row in comparisons)}
    rows = []
    for total, window in pairs:
        if time.time() > deadline:
            rows.append(rec.event("window_skipped", tokens=total, window=window, reason="time"))
            break
        row = {"tokens": total, "window": window, "chunk": chunk}
        try:
            inputs = batch(tokenizer, total, salt=total + window, supervised=window / total)
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            loss, prefix, forward, backward = window_step(model, inputs, window=window, chunk=chunk)
            seconds = prefix + forward + backward
            row.update(loss=round(loss, 4), finite=loss == loss and abs(loss) != float("inf"),
                       prefix_seconds=round(prefix, 2), forward_seconds=round(forward, 2),
                       backward_seconds=round(backward, 2), seconds=round(seconds, 2),
                       context_tokens_per_second=round(total / seconds),
                       peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                       gradients=gradients(model), fits=True)
        except torch.OutOfMemoryError as error:
            row.update(fits=False, error=str(error)[:300],
                       peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
        model.zero_grad(set_to_none=True)
        inputs = None
        torch.cuda.empty_cache()
        rows.append(rec.event("window", **row))
        rec.save(windows=rows)
    return compared, rows


def run_smoke(rec, model_dir, out, lengths, *, deadline, overfit_tokens, window_plan=None):
    import torch

    model, tokenizer, loaded = load(rec, model_dir, out, sequence_len=max(lengths))
    rec.save(loaded=loaded)
    model.train()
    rows = ladder(rec, model, tokenizer, lengths, deadline=deadline - 180)
    fit = [row["tokens"] for row in rows if row.get("fits")]
    learned = overfit(rec, model, tokenizer, length=min(overfit_tokens, max(fit))) if fit else None
    rec.save(overfit=learned)
    windowed = None
    if window_plan and fit:
        compared, cached = windows(rec, model, tokenizer, deadline=deadline - 120, **window_plan)
        windowed = compared["same_loss"] and (not window_plan["pairs"] or any(row.get("fits") for row in cached)) and all(
            row["finite"] and all(g["no_grad"] == 0 and g["nonfinite"] == 0 and g["norm"] > 0
                                  for g in row["gradients"].values()) for row in cached if row.get("fits"))
    began = time.time()
    model.save_pretrained(str(Path(out) / "adapter"))
    files = {p.name: p.stat().st_size for p in (Path(out) / "adapter").iterdir() if p.is_file()}
    saved = rec.event("adapter_saved", seconds=round(time.time() - began, 1), files=files)
    groups = [row["gradients"] for row in rows if row.get("fits")]
    clean = all(g["no_grad"] == 0 and g["nonfinite"] == 0 and g["norm"] > 0 for row in groups for g in row.values())
    passed = bool(fit) and clean and all(row["finite"] for row in rows if row.get("fits")) and bool(
        learned and learned["falls"]) and set(loaded["adapter_groups"]) == {g for g, _ in GROUPS} and loaded[
            "ngram_table_on"] == ["cpu"] and windowed is not False
    rec.save(saved=saved, largest_fit_tokens=max(fit) if fit else 0, gradients_clean=clean, passed=passed,
             windows_passed=windowed,
             seconds=round(time.time() - START, 1), gpu=torch.cuda.get_device_name(0))
    return passed


def tail_nll(model, sequences, tail):
    """Mean negative log-likelihood of the last ``tail`` tokens of each sequence (teacher forcing)."""
    import torch

    model.eval()
    values = []
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for ids in sequences:
                tokens = torch.tensor([ids], device="cuda:0")
                labels = tokens.clone()
                labels[:, : len(ids) - tail] = -100
                values.append(float(model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                          labels=labels).loss.float()))
    finally:
        model.train()
    return values


def run_adapter(rec, model_dir, out, data_path, *, steps, lr, tokens, tail, deadline):
    """G2 pre-check, trainer side: a short adapter on DEV-game text and its gain on other DEV games' text, measured
    against the trainer's own (NF4) weights. Writes the adapter and the evaluation sequences as token ids
    (``eval-prompts.json``, the input of the served check: ``arena.kernel_gate`` teacher forcing)."""
    import torch

    data = json.loads(Path(data_path).read_text())
    model, tokenizer, loaded = load(rec, model_dir, out, sequence_len=tokens)
    rec.save(loaded=loaded)

    def encode(texts):
        rows = [tokenizer(text, add_special_tokens=False)["input_ids"][:tokens] for text in texts]
        short = [len(row) for row in rows if len(row) < tokens]
        if short:
            raise ValueError(f"{len(short)} texts are shorter than {tokens} tokens (shortest {min(short)})")
        return rows

    train, held = encode(data["train"]), encode(data["eval"])
    (Path(out) / "eval-prompts.json").write_text(json.dumps({"greedy": [], "teacher_forced": [
        {"index": i, "ids": ids, "assistant_spans": [[tokens - tail, tokens]]} for i, ids in enumerate(held)]}))
    model.train()
    base = tail_nll(model, held, tail)
    rec.event("eval", stage="base", mean=round(sum(base) / len(base), 5), sequences=len(base))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    losses = []
    for index in range(steps):
        if time.time() > deadline - 150:
            rec.event("train_stopped", reason="time", steps=index)
            break
        tokens_in = torch.tensor([train[index % len(train)]], device="cuda:0")
        labels = tokens_in.clone()
        labels[:, :64] = -100
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = step(model, {"input_ids": tokens_in, "attention_mask": torch.ones_like(tokens_in),
                                  "labels": labels})
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        losses.append(round(loss, 4))
        if index % 10 == 9:
            rec.event("train", steps=index + 1, last_ten=round(sum(losses[-10:]) / 10, 4))
    model.zero_grad(set_to_none=True)
    tuned = tail_nll(model, held, tail) if losses else base
    gain = sum(base) / len(base) - sum(tuned) / len(tuned)
    rec.event("eval", stage="adapter", mean=round(sum(tuned) / len(tuned), 5), gain=round(gain, 5),
              improved=sum(b > t for b, t in zip(base, tuned)))
    model.save_pretrained(str(Path(out) / "adapter"))
    passed = len(losses) >= min(steps, 20) and (gain > 0 or steps < 20)  # --steps 0: the base NLL only
    rec.save(trainer={"steps": len(losses), "lr": lr, "tokens": tokens, "tail": tail, "train_losses": losses,
                      "base_nll": base, "adapter_nll": tuned, "gain": gain, "games": data.get("games")},
             passed=passed, seconds=round(time.time() - START, 1))
    return passed


def run_export(rec, adapter_dir, served_dir, experts_dir, out_dir):
    """Merge an adapter into the served checkpoint's BF16 shards and write an expert-overlay folder that serves
    them beside the given expert overlay's shards (``arena.expert_overlay``: whole replacement shards, one
    manifest). A zero adapter is checked to reproduce every tensor, and the shard file, bit for bit."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    adapter_dir, served_dir, experts_dir, out_dir = map(Path, (adapter_dir, served_dir, experts_dir, out_dir))
    settings = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = settings["lora_alpha"] / settings["r"]
    lora = load_file(str(adapter_dir / "adapter_model.safetensors"))
    pairs = {}
    for key in lora:
        if ".lora_A." in key:
            name = key.split("base_model.model.", 1)[1].replace(".lora_A.weight", ".weight")
            pairs[name] = (lora[key].float(), lora[key.replace(".lora_A.", ".lora_B.")].float())
    index = json.loads((served_dir / "model.safetensors.index.json").read_text())["weight_map"]
    unknown = [name for name in pairs if name not in index or not TARGET_TENSOR.match(name)]
    if unknown or len(pairs) != 372:
        raise RuntimeError(f"{len(pairs)} adapted tensors, {len(unknown)} not in the served checkpoint: {unknown[:3]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stats, shards, zero = {}, {}, {"tensors": 0, "identical": 0, "files": 0, "files_identical": 0}

    def digest(path):
        with open(path, "rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()

    def bits(tensor):
        return tensor.contiguous().view(torch.int16) if tensor.dtype == torch.bfloat16 else tensor

    for shard in sorted({index[name] for name in pairs}):
        with safe_open(str(served_dir / shard), framework="pt") as handle:
            metadata = handle.metadata()
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
        # a zero adapter: the same writer on the unchanged tensors must give the served file back
        save_file(tensors, str(out_dir / shard), metadata=metadata)
        zero["files"] += 1
        zero["files_identical"] += digest(out_dir / shard) == digest(served_dir / shard)
        for name in [n for n in tensors if n in pairs]:
            weight = tensors[name]
            if weight.dtype != torch.bfloat16:
                raise RuntimeError(f"{name} is {weight.dtype} in the served checkpoint")
            a, b = pairs[name]
            delta = scale * (b @ a)
            zero["tensors"] += 1
            zero["identical"] += bool(torch.equal(bits((weight.float() + 0 * delta).to(torch.bfloat16)), bits(weight)))
            merged = (weight.float() + delta).to(torch.bfloat16)
            realized = merged.float() - weight.float()
            group = stats.setdefault(group_of(name.replace("mlp.", "")), {
                "tensors": 0, "delta_sq": 0.0, "weight_sq": 0.0, "realized_sq": 0.0, "dot": 0.0, "changed": 0,
                "elements": 0})
            group["tensors"] += 1
            group["delta_sq"] += float((delta * delta).sum())
            group["weight_sq"] += float((weight.float() ** 2).sum())
            group["realized_sq"] += float((realized * realized).sum())
            group["dot"] += float((realized * delta).sum())
            group["changed"] += int((bits(merged) != bits(weight)).sum())
            group["elements"] += weight.numel()
            tensors[name] = merged
        save_file(tensors, str(out_dir / shard), metadata=metadata)
        shards[shard] = {"bytes": (out_dir / shard).stat().st_size, "sha256": digest(out_dir / shard)}
        rec.event("shard", name=shard, tensors=len(tensors), merged=sum(n in pairs for n in tensors),
                  bytes=shards[shard]["bytes"], base_bytes=(served_dir / shard).stat().st_size)
        del tensors
    merge = {name: {"tensors": g["tensors"],
                    "delta_over_weight": round((g["delta_sq"] / g["weight_sq"]) ** 0.5, 6),
                    "elements_changed": round(g["changed"] / g["elements"], 4),
                    "realized_over_delta": round((g["realized_sq"] / max(g["delta_sq"], 1e-30)) ** 0.5, 4),
                    "realized_cosine": round(g["dot"] / max((g["realized_sq"] * g["delta_sq"]) ** 0.5, 1e-30), 4)}
             for name, g in stats.items()}
    # --experts none: the merged shards alone, over the served checkpoint's own experts (plain NVFP4 profile)
    base = ({"format": "arc-expert-overlay-v1", "complete": True, "shards": {}} if experts_dir.name == "none"
            else json.loads((experts_dir / "overlay-manifest.json").read_text()))
    links = "symlink"
    for name in base["shards"]:
        target = out_dir / name
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(os.path.relpath(experts_dir / name, out_dir))
            if target.stat().st_size != base["shards"][name]["bytes"]:
                raise OSError("the link does not resolve to the expert shard")
        except OSError:
            links = "copy"
            if target.is_symlink():
                target.unlink()
            shutil.copyfile(experts_dir / name, target)
    manifest = {**base, "shards": {**base["shards"], **shards}, "adapter": {"merge": merge, "scale": scale}}
    raw = json.dumps(manifest, sort_keys=True).encode()
    (out_dir / "overlay-manifest.json").write_bytes(raw)
    passed = zero["identical"] == zero["tensors"] == 372
    rec.event("export", zero_adapter=zero, merge=merge, expert_shards=len(base["shards"]), links=links,
              manifest_sha256=hashlib.sha256(raw).hexdigest(), folder=str(out_dir))
    rec.save(export={"zero_adapter": zero, "merge": merge, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                     "folder": str(out_dir), "links": links, "shards": shards}, passed=passed,
             seconds=round(time.time() - START, 1))
    return passed


def run_contexts(rec, model_dir, out, data_path, *, served_dir=None):
    """Trainer inputs from captured requests (one JSON per line). Every request is encoded with the real processor
    and its counts compared with the server's; then, on a tiny random model, the cached-prefix step on the shortest
    example with several images against the plain step (same loss: positions and images across prefix chunks)."""
    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(served_dir or model_dir))  # the served template and tokenizer
    rows, examples = [], []
    for line in Path(data_path).read_text().splitlines():
        capture = json.loads(line)
        try:
            inputs, checks = encode_request(processor, capture)
        except Exception as error:  # the report is the point
            rows.append(rec.event("context", request=capture.get("request"),
                                  error=f"{type(error).__name__}: {error}"[:400]))
            continue
        rows.append(rec.event("context", **checks))
        examples.append((inputs, checks))
    encoded = [row for row in rows if "error" not in row]
    summary = {"requests": len(rows), "encoded": len(encoded),
               "prompt_exact": sum(r["prompt_tokens"] == r["served_prompt_tokens"] for r in encoded),
               "prompt_diff": sorted({r["prompt_tokens"] - r["served_prompt_tokens"] for r in encoded}),
               "reply_diff": sorted({r["reply_tokens"] - (r["served_reply_tokens"] or 0) for r in encoded}),
               "prefix_kept": sum(r["prefix_kept"] for r in encoded)}
    rec.save(contexts=rows, summary=summary)
    several = [(i, c) for i, c in examples if c["images"] >= 2 and i["input_ids"].shape[1] <= 16384]
    compared = None
    if several and torch.cuda.is_available():
        inputs, checks = min(several, key=lambda item: item[0]["input_ids"].shape[1])
        model, _, _ = load(rec, tiny_model(model_dir, out), out, sequence_len=16384)
        model.train()
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        total = inputs["input_ids"].shape[1]
        window = min(total - 1000, max(1024, checks["reply_tokens"] + 300))
        model.zero_grad(set_to_none=True)
        full_loss, _, _ = step(model, inputs)
        full = flat_gradients(model)
        model.zero_grad(set_to_none=True)
        loss, _, _, _ = window_step(model, inputs, window=window, chunk=1500)
        cut = flat_gradients(model)
        model.zero_grad(set_to_none=True)
        image = (inputs["input_ids"][0] == processor.image_token_id).tolist()
        edges = list(range(1500, total - window, 1500)) + [total - window]
        compared = rec.event(
            "images_across_chunks", request=checks["request"], tokens=total, window=window, images=checks["images"],
            edges_inside_an_image=sum(image[e - 1] and image[e] for e in edges), loss_full=round(full_loss, 5),
            loss_window=round(loss, 5), same_loss=abs(loss - full_loss) <= 0.02 * max(abs(full_loss), 1e-3) + 2e-3,
            gradient_cosine={name: round(float(torch.nn.functional.cosine_similarity(full[name], cut[name], dim=0)), 4)
                             for name in full})
    # a replayed drain note can differ from the recorded one by a few tokens (seen: up to 3)
    passed = bool(encoded) and summary["encoded"] == summary["requests"] == summary["prefix_kept"] and max(
        map(abs, summary["prompt_diff"])) <= 8 and bool(compared and compared["same_loss"])
    rec.save(images_across_chunks=compared, passed=passed, seconds=round(time.time() - START, 1))
    return passed


ADAPTER = {"rank": 16, "alpha": 32, "rslora": False}
SLOW_SECONDS = 93     # 1.5 x the measured 62 s per example
MEMORY_SHARE = 0.95


def reply_loss(model, inputs, *, tail, backward):
    """One example: loss on its reply, by the plain step when it fits in the tail, else by the cached-prefix step."""
    import torch

    total = inputs["input_ids"].shape[1]
    if total <= tail:
        if backward:
            return step(model, inputs)[0]
        model.eval()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                return float(model(**inputs).loss.float())
        finally:
            model.train()
    return window_step(model, inputs, window=tail, chunk=16384, backward=backward)[0]


def generation_probe(model, processor, captures, *, new_tokens):
    """Free generation on a few selection contexts with short recorded replies: does the reply end, does it hold a
    complete tool call, how much of it repeats. A proxy for the served behaviour gates (the completion cap cannot be
    reached here)."""
    import torch

    rows = []
    model.eval()
    try:
        for capture in captures:
            inputs, checks = encode_request(processor, capture)
            count, total = checks["prompt_tokens"], inputs["input_ids"].shape[1]
            prompt = {key: (value[:, :count] if value.dim() == 2 and value.shape[1] == total else value).to("cuda:0")
                      for key, value in inputs.items() if key != "labels"}
            began = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(**prompt, max_new_tokens=new_tokens, do_sample=False, use_cache=True,
                                     max_time=150)
            new = out[0, count:].tolist()
            text = processor.tokenizer.decode(new, skip_special_tokens=False)
            grams = [tuple(new[i:i + 6]) for i in range(max(0, len(new) - 5))]
            rows.append({"request": checks["request"], "new_tokens": len(new), "seconds": round(time.time() - began, 1),
                         "ended": "<|im_end|>" in text, "closed_think": "</think>" in text,
                         "tool_call": bool(re.search(r"<tool_call>\s*<function=\w+>.*</function>\s*</tool_call>", text,
                                                     re.S)),
                         "repeat": round(1 - len(set(grams)) / max(len(grams), 1), 3)})
    finally:
        model.train()
    return {"rows": rows, "ended": sum(r["ended"] for r in rows), "tool_call": sum(r["tool_call"] for r in rows),
            "closed_think": sum(r["closed_think"] for r in rows),
            "repeat": round(sum(r["repeat"] for r in rows) / max(len(rows), 1), 3)}


def example_maker(processor, train, *, tail, phases):
    """``encoded(capture)``: one served request as tensors on the GPU, or None when it is not the served request or
    does not fit (never truncated); ``examples(window, rows)``: the training examples of one unit."""

    def encoded(capture, pair=False, device="cuda:0", slack=0.0):
        """``slack``: allowed prompt-count gap as a share of the prompt (streaming OPD, 23 Oct: the student play's
        server counts +1..+18 tokens over the re-encoding at 9-57K context, 0.03%; 8 tokens dropped every request
        past ~40K). The teacher and the trainer both use the re-encoded ids."""
        inputs, checks = encode_request(processor, capture)
        total = inputs["input_ids"].shape[1]
        limit = max(8, int(slack * checks["prompt_tokens"]))
        if abs(checks["prompt_tokens"] - checks["served_prompt_tokens"]) > limit or total > 131072 or (
                not pair and checks["reply_tokens"] + 2 > tail):
            return None
        if device is not None:  # None: stay in host memory (a producer thread must not touch CUDA: stream capture)
            inputs = {key: value.to(device) for key, value in inputs.items()}
        return (inputs, checks["prompt_tokens"]) if pair else inputs

    def examples(window, rows):
        captures = [json.loads(train[i]) for i in rows]
        if not phases:
            inputs = encoded(captures[0])
            return [] if inputs is None else [(inputs, window, 1)]
        return chain_examples(lambda capture: encoded(capture, pair=True), captures, window=window)

    return encoded, examples


_STREAM_ENC = {}


def stream_encode_init(model_dir, slack, window):
    """A ``heal_stream.ProcessEncoder`` worker: the processor once, host only."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoProcessor

    torch.set_num_threads(1)
    encoded, _ = example_maker(AutoProcessor.from_pretrained(model_dir), [], tail=window, phases=[window])
    _STREAM_ENC.update(encoded=encoded, slack=slack)


def stream_encode(capture):
    """The stream teacher's encode in a worker: ``encoded(capture, pair=True)`` on the host."""
    return _STREAM_ENC["encoded"](capture, pair=True, device=None, slack=_STREAM_ENC["slack"])


def unit_gradients(model, params, made):
    """The examples of one unit on this process's model: their gradients summed in the parameters' ``.grad`` and
    what happened. An example that runs out of memory voids the whole unit (its partial gradients are dropped)."""
    import math

    import torch

    row = {"examples": 0, "targets": 0, "losses": [], "seconds": [], "expected": [], "tokens": [], "tails": [],
           "peak_gib": 0.0, "out_of_memory": 0, "dropped": int(not made), "nonfinite": False}
    for param in params:
        param.grad = None
    while made:
        inputs, tail, targets = made.pop()
        total = int(inputs["input_ids"].shape[1])
        torch.cuda.reset_peak_memory_stats()
        began = time.time()
        try:
            loss = reply_loss(model, inputs, tail=tail, backward=True)
        except torch.OutOfMemoryError:
            del inputs
            made.clear()
            for param in params:
                param.grad = None
            torch.cuda.empty_cache()
            row.update(examples=0, targets=0, losses=[], seconds=[], expected=[], out_of_memory=1)
            break
        del inputs
        if not math.isfinite(loss):
            row["nonfinite"] = True
            break
        row["examples"] += 1
        row["targets"] += targets
        row["losses"].append(loss)
        row["seconds"].append(time.time() - began)
        row["expected"].append(expected_seconds(min(tail, total)))
        row["tokens"].append(total)
        row["tails"].append(min(tail, total))
        row["peak_gib"] = max(row["peak_gib"], torch.cuda.max_memory_allocated() / 2 ** 30)
    return row


class Replicas:
    """Data-parallel replicas: one worker process per further GPU (``CUDA_VISIBLE_DEVICES``), each with its own
    copy of the model. Per update the trainer writes the adapter's weights and one unit per worker to shared
    memory, every process computes its unit's gradients, and the trainer adds the workers' gradients to its own.
    Workers are started one after another: a load peaks at ~250 GiB of host memory."""

    def __init__(self, rec, count, command, out):
        self.rec, self.number, self.workers = rec, 0, []
        devices = (os.environ.get("ARC_REPLICA_DEVICES") or ",".join(map(str, range(count)))).split(",")
        self.root = Path(os.environ.get("ARC_REPLICA_DIR") or "/dev/shm") / f"replicas-{os.getpid()}"
        for index in range(1, count):
            folder = self.root / str(index)
            folder.mkdir(parents=True, exist_ok=True)
            log = open(Path(out) / f"replica-{index}.log", "a")  # noqa: SIM115 - held for the worker's life
            process = subprocess.Popen(
                [*command, "--out", str(folder)], stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": devices[index]})
            self.workers.append((index, folder, process))
            self.wait(folder / "ready.json", process, 3600, f"replica {index} did not load")
            rec.event("replica", index=index, device=devices[index], **json.loads((folder / "ready.json").read_text()))

    @staticmethod
    def wait(path, process, seconds, what):
        limit = time.time() + seconds
        while not path.is_file():
            if process.poll() is not None or time.time() > limit:
                raise RuntimeError(f"{what} (exit {process.poll()})")
            time.sleep(0.2)

    def send(self, params, units):
        import torch

        self.number += 1
        weights = self.root / f"weights-{self.number}.pt"
        torch.save([param.detach().cpu() for param in params], str(weights))
        for (_, folder, _), unit in zip(self.workers, units):
            (folder / "task.tmp").write_text(json.dumps({"number": self.number, "weights": str(weights),
                                                         "window": unit[0], "rows": unit[1]}))
            (folder / "task.tmp").rename(folder / f"task-{self.number}.json")
        self.sent = len(units)

    def collect(self, params):
        import torch

        rows = []
        for index, folder, process in self.workers[:self.sent]:
            done = folder / f"done-{self.number}.json"
            self.wait(done, process, 3600, f"replica {index} gave no gradients for update {self.number}")
            row = json.loads(done.read_text())
            if row["examples"]:
                for param, grad in zip(params, torch.load(str(folder / f"grads-{self.number}.pt"), weights_only=True)):
                    if grad is not None:
                        grad = grad.to(param.device, param.dtype)
                        param.grad = grad if param.grad is None else param.grad + grad
            for name in (f"done-{self.number}.json", f"grads-{self.number}.pt"):
                (folder / name).unlink(missing_ok=True)
            rows.append(row)
        (self.root / f"weights-{self.number}.pt").unlink(missing_ok=True)
        return rows

    def close(self):
        for _, folder, process in self.workers:
            (folder / "stop").write_text("")
        for _, _, process in self.workers:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
        shutil.rmtree(self.root, ignore_errors=True)


def run_worker(rec, model_dir, folder, corpus, *, served_dir, tail, phases):
    """A replica: load the model, then for every task take the adapter's weights, compute the unit's gradients and
    write them back; leave at the ``stop`` file or when the trainer is gone."""
    import torch
    from transformers import AutoProcessor

    folder = Path(folder)
    processor = AutoProcessor.from_pretrained(str(served_dir or model_dir))
    train = (Path(corpus) / "train.jsonl").read_text().splitlines()
    model, _, loaded = load(rec, model_dir, folder, sequence_len=131072)
    if NVFP4["on"]:
        apply_table_scale(rec, model, model_dir)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    _, examples = example_maker(processor, train, tail=tail, phases=phases)
    (folder / "ready.json").write_text(json.dumps({"load_seconds": loaded["seconds"], "trainable": loaded["trainable"],
                                                   "gpu_gib": loaded["gpu_allocated_gib"]}))
    number, parent = 0, os.getppid()
    while not (folder / "stop").is_file() and os.getppid() == parent:
        path = folder / f"task-{number + 1}.json"
        if not path.is_file():
            later = sorted(folder.glob("task-*.json"), key=lambda item: int(item.stem.split("-")[1]))
            if later:  # a restarted trainer counts from where it is
                number = int(later[0].stem.split("-")[1]) - 1
            else:
                time.sleep(0.1)
            continue
        number += 1
        task = json.loads(path.read_text())
        with torch.no_grad():
            for param, value in zip(params, torch.load(task["weights"], weights_only=True)):
                param.copy_(value.to(param.device, param.dtype))
        row = unit_gradients(model, params, examples(task["window"], task["rows"]))
        if row["examples"]:
            torch.save([None if p.grad is None else p.grad.detach().cpu() for p in params],
                       str(folder / f"grads-{number}.pt"))
        for param in params:
            param.grad = None
        torch.cuda.empty_cache()  # nothing is held between updates (two replicas on one GPU in the tiny-model check)
        (folder / "done.tmp").write_text(json.dumps(row))
        (folder / "done.tmp").rename(folder / f"done-{number}.json")
        path.unlink()
    return True


def expected_seconds(tail):
    """Seconds of one step by its gradient window (B300 and B200 alike; sft-qlora-nvfp4-b300-v7: 16K 90 s, 24K 146,
    32K 171, 48K 227)."""
    return 35 + 4.1 * tail / 1024


def train_units(metas, *, tail, limit, phases, mix):
    """The run's units in order, ``(gradient window, row numbers)``. Without ``phases``: one pass, one reply per
    unit (the first adapter). With ``phases`` (windows, shortest first): one pass over every chain per phase, each
    chain cut into examples of at most that phase's window; from the second phase on a share ``mix`` of the chains
    keeps an earlier phase's window, so lengths stay mixed while the longest grows."""
    import random

    if not phases:
        order = list(range(len(metas)))
        random.Random(0).shuffle(order)
        return [(tail, [i]) for i in (order[:limit] if limit else order)]
    chains = {}
    for index, meta in enumerate(metas):
        chains.setdefault(meta.get("chain") or f"row-{index}", []).append((meta.get("chain_position", 0), index))
    names = sorted(chains)[:limit or None]
    units = []
    for phase, window in enumerate(phases):
        rng = random.Random(phase)
        order = list(names)
        rng.shuffle(order)
        for name in order:
            chosen = rng.choice(phases[:phase]) if phase and rng.random() < mix else window
            units.append((chosen, [index for _, index in sorted(chains[name])]))
    return units


def run_train(rec, model_dir, out, corpus, *, served_dir, tail, lr, deadline, limit, eval_every, checkpoint_every,
              probe_tokens, stop_after=0, phases=(), mix=0.25, select_limit=0, replicas=1, worker_command=()):
    """The first adapter (option A): one pass over ``corpus/train.jsonl`` (captured served requests, loss on the
    reply), batch 1, AdamW with warmup and cosine decay. Live numbers go to ``progress.json`` after every step;
    a checkpoint every ``checkpoint_every`` examples; selection NLL (``select.jsonl``) every ``eval_every``. The run
    stops itself on the rules below and says why (``stopped``). A restarted job continues from its last checkpoint
    with the time it has already used taken off its budget."""
    import math

    import torch
    from transformers import AutoProcessor

    out = Path(out)
    processor = AutoProcessor.from_pretrained(str(served_dir or model_dir))
    train = (Path(corpus) / "train.jsonl").read_text().splitlines()
    select = [json.loads(line) for line in (Path(corpus) / "select.jsonl").read_text().splitlines()]
    order = train_units([json.loads(line)["meta"] for line in train], tail=tail, limit=limit, phases=phases, mix=mix)
    steps = len(order)
    if select_limit and len(select) > select_limit:  # an even draw per game, the same rows at every evaluation
        by_game = {}
        for capture in sorted(select, key=lambda c: (c["meta"]["game"], c["meta"]["k"], c["request"])):
            by_game.setdefault(capture["meta"]["game"][:4], []).append(capture)
        share = max(1, select_limit // len(by_game))
        select = [rows[(i * len(rows)) // min(share, len(rows))] for rows in by_game.values()
                  for i in range(min(share, len(rows)))]
    probe_rows = [i for _, rows in order[:len(order) // max(1, len(phases))][-8:] for i in rows[-1:]]
    probe_train = [json.loads(train[i]) for i in probe_rows]  # seen last in the first pass: unseen at the 10% check
    short = sorted((c for c in select if c["prompt_tokens"] <= 24000), key=lambda c: c["usage"]["completion_tokens"])[:6]
    model, _, loaded = load(rec, model_dir, out, sequence_len=131072)
    if NVFP4["on"]:  # the served checkpoint stores the n-gram table's scale apart (run_nvfp4)
        loaded = {**loaded, "table_scale": apply_table_scale(rec, model, model_dir)}
    rec.save(loaded=loaded, plan={"steps": steps, "tail": tail, "lr": lr, "select": len(select),
                                  "eval_every": eval_every, "checkpoint_every": checkpoint_every,
                                  "phases": list(phases), "mix": mix, "adapter": dict(ADAPTER),
                                  "replicas": replicas})
    limit_gib = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warmup = max(1, round(0.05 * steps))

    def rate(index):
        if index < warmup:
            return (index + 1) / warmup
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (index - warmup) / max(1, steps - warmup)))

    state = {"step": 0, "next": 0, "steps": steps, "stopped": None, "dropped": 0, "out_of_memory": 0, "losses": [],
             "seconds": [], "expected": [], "updates": 0, "targets": 0, "peak_gib": 0.0, "train_probe": {},
             "select": [], "generation": {}, "checkpoints": [],
             "wall": [], "used_seconds": 0.0, "used_before": 0.0, "process_start": START, "restarts": 0}
    reserve = min(900, 0.1 * (deadline - START))  # seconds kept for the last selection pass and the save
    latest = out / "ckpt" / "latest.json"
    if latest.is_file():  # a restarted job
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        folder = out / "ckpt" / json.loads(latest.read_text())["label"]
        saved = torch.load(str(folder / "trainer.pt"), map_location="cuda:0", weights_only=False)
        set_peft_model_state_dict(model, load_file(str(folder / "adapter_model.safetensors")))
        optimizer.load_state_dict(saved["optimizer"])
        state = {**saved["state"], "restarts": saved["state"]["restarts"] + 1, "stopped": None}
        if state["process_start"] != START:  # another process used part of the budget
            state.update(used_before=state["used_seconds"], process_start=START)
            deadline -= state["used_before"]
        rec.event("resumed", checkpoint=folder.name, next=state["next"], used_seconds=round(state["used_seconds"]))
    used_before = state["used_before"]

    def progress():
        losses, seconds = state["losses"], state["seconds"]
        mean = lambda values: round(sum(values) / len(values), 4) if values else None  # noqa: E731
        left = (steps - state["next"]) * (sum(state["wall"][-50:]) / max(len(state["wall"][-50:]), 1))
        (out / "progress.json").write_text(json.dumps({
            "step": state["step"], "steps": steps, "updates": state["updates"], "targets": state["targets"],
            "elapsed_seconds": round(used_before + time.time() - START),
            "restarts": state["restarts"],
            "loss_first_20": mean(losses[:20]), "loss_last_20": mean(losses[-20:]),
            "seconds_per_example_last_20": mean(seconds[-20:]), "peak_gib": state["peak_gib"],
            "limit_gib": round(limit_gib, 1), "dropped": state["dropped"], "out_of_memory": state["out_of_memory"],
            "train_probe": state["train_probe"], "select": state["select"], "generation": state["generation"],
            "checkpoints": state["checkpoints"][-4:], "eta_seconds": round(left), "stopped": state["stopped"]},
            indent=1))

    encoded, examples = example_maker(processor, train, tail=tail, phases=phases)

    def nll(captures):
        values = []
        for capture in captures:
            inputs = encoded(capture)
            if inputs is not None:
                values.append(reply_loss(model, inputs, tail=tail, backward=False))
            del inputs
        torch.cuda.empty_cache()
        return values

    def checkpoint(label):
        folder = out / "ckpt" / label
        model.save_pretrained(str(folder))
        if label not in state["checkpoints"]:
            state["checkpoints"].append(label)
        state["used_seconds"] = used_before + time.time() - START
        torch.save({"optimizer": optimizer.state_dict(), "state": state}, str(folder / "trainer.pt"))
        latest.write_text(json.dumps({"label": label, "next": state["next"]}))
        for old in state["checkpoints"][:-2]:  # keep the last two and every selection checkpoint
            if not old.startswith("select-") and (out / "ckpt" / old).is_dir():
                shutil.rmtree(out / "ckpt" / old)
        return label

    def select_eval():
        values = nll(select)
        games = {}
        for capture, value in zip(select, values):
            games.setdefault(capture["meta"]["game"][:4], []).append(value)
        row = {"step": state["step"], "mean": round(sum(sum(v) / len(v) for v in games.values()) / len(games), 5),
               "games": {game: round(sum(v) / len(v), 5) for game, v in games.items()}, "examples": len(values),
               "checkpoint": f"select-{state['step']:04d}"}
        state["select"] = [r for r in state["select"] if r["step"] != row["step"]] + [row]
        checkpoint(row["checkpoint"])
        rec.event("select", **row)
        return row

    def stop(reason):
        state["stopped"] = reason
        rec.event("stopped", reason=reason, step=state["step"])

    if not state["select"]:
        base_probe = nll(probe_train)
        state["train_probe"] = {"base": round(sum(base_probe) / len(base_probe), 5)}
        try:
            state["generation"]["base"] = generation_probe(model, processor, short, new_tokens=probe_tokens)
        except Exception as error:  # the probe is new on this model: its failure is reported, not hidden
            state["generation"]["base"] = {"error": f"{type(error).__name__}: {error}"[:300]}
        select_eval()
        rec.event("base", train_probe=state["train_probe"], select=state["select"][0],
                  generation={k: v for k, v in state["generation"]["base"].items() if k != "rows"})
    base_select = state["select"][0]
    progress()
    workers = Replicas(rec, replicas, worker_command, out) if replicas > 1 else None
    index = state["next"]
    while index < steps and not state["stopped"]:
        if stop_after and state["step"] >= stop_after:  # a test of the restart path
            checkpoint(f"step-{state['step']:04d}")
            progress()
            if workers:
                workers.close()
            return None
        if time.time() > deadline - reserve:
            stop("time")
            break
        began_wall, before = time.time(), state["step"]
        batch = order[index:index + max(1, replicas)]  # one unit per replica; one update from all their examples
        for group in optimizer.param_groups:
            group["lr"] = lr * rate(index)
        if workers:
            workers.send(params, batch[1:])
        rows = [unit_gradients(model, params, examples(*batch[0]))]
        if workers:
            try:
                rows += workers.collect(params)
            except RuntimeError as error:
                stop(str(error))
                break
        index += len(batch)
        state["next"] = index
        count = sum(row["examples"] for row in rows)
        state["dropped"] += sum(row["dropped"] for row in rows)
        state["out_of_memory"] += sum(row["out_of_memory"] for row in rows)
        if any(row["nonfinite"] for row in rows):
            stop(f"non-finite loss at unit {index}")
            break
        if state["out_of_memory"] > 3:
            stop("out of memory on more than three units")
            break
        if state["dropped"] > max(3, 0.05 * steps):
            stop("more than 5% of the units hold no served request")
            break
        if not count:
            continue
        for param in params:
            if param.grad is not None:
                param.grad /= count
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        if not math.isfinite(norm):
            stop(f"non-finite gradient at unit {index}")
            break
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        peak = max(row["peak_gib"] for row in rows)
        for key in ("losses", "seconds", "expected"):
            state[key].extend(value for row in rows for value in row[key])
        state["updates"] += 1
        state["targets"] += sum(row["targets"] for row in rows)
        state["step"] = index
        state["wall"].append((time.time() - began_wall) / len(batch))
        state["peak_gib"] = round(max(state["peak_gib"], peak), 1)
        rec.event("step", step=index, examples=count, targets=sum(row["targets"] for row in rows),
                  tokens=[t for row in rows for t in row["tokens"]], tails=[t for row in rows for t in row["tails"]],
                  loss=round(sum(v for row in rows for v in row["losses"]) / count, 4),
                  seconds=round(time.time() - began_wall, 1), peak_gib=round(peak, 1), grad_norm=round(norm, 4),
                  lr=round(lr * rate(index - 1), 8))

        def reached(mark):  # the unit count passed ``mark`` in this update (an update covers one unit per replica)
            return before < mark <= state["step"]

        def every(period):
            return before // period != state["step"] // period

        if peak > MEMORY_SHARE * limit_gib:
            stop(f"GPU memory {peak:.1f} GiB within 5% of the {limit_gib:.1f} GiB limit")
        recent, allowed = state["seconds"][-20:], state["expected"][-20:]
        if phases and len(recent) == 20 and sum(recent) > 1.5 * sum(allowed):
            stop(f"{sum(recent) / 20:.0f} s per example over the last 20, above 1.5 x the expected {sum(allowed) / 20:.0f}")
        if not phases and len(recent) == 20 and sum(recent) / 20 > SLOW_SECONDS:
            stop(f"{sum(recent) / 20:.0f} s per example over the last 20, above {SLOW_SECONDS}")
        losses = state["losses"]
        if len(losses) >= 100 and sum(losses[-40:]) / 40 > sum(losses[:40]) / 40 + 0.3:
            stop("training loss of the last 40 examples above the first 40 by more than 0.3")
        if every(checkpoint_every):
            checkpoint(f"step-{state['step']:04d}")
        if reached(math.ceil(0.1 * steps)):
            now = nll(probe_train)
            state["train_probe"]["at_10_percent"] = round(sum(now) / len(now), 5)
            rec.event("train_probe", **state["train_probe"])
            if state["train_probe"]["at_10_percent"] >= state["train_probe"]["base"]:
                stop("train loss on the fixed probe is not below its starting value after 10% of the steps")
        if every(eval_every) and state["step"] < steps:
            row = select_eval()
            if row["mean"] > base_select["mean"] + 0.02:
                stop("selection NLL above its starting value by more than 0.02")
        if reached(steps // 2):
            try:
                half = generation_probe(model, processor, short, new_tokens=probe_tokens)
            except Exception as error:
                half = {"error": f"{type(error).__name__}: {error}"[:300]}
            state["generation"]["half"] = half
            rec.event("generation", **{k: v for k, v in half.items() if k != "rows"})
            base = state["generation"]["base"]
            if "error" not in half and "error" not in base and (
                    half["tool_call"] <= base["tool_call"] - 2 or half["ended"] <= base["ended"] - 2
                    or half["repeat"] > base["repeat"] + 0.15):
                stop("generation probe worse than base: fewer finished replies or tool calls, or more repetition")
        progress()
    if workers:
        workers.close()
    final = select_eval()
    best = min(state["select"], key=lambda row: row["mean"])
    shutil.copytree(out / "ckpt" / best["checkpoint"], out / "adapter", dirs_exist_ok=True)
    passed = state["stopped"] in (None, "time") and final["mean"] < base_select["mean"]
    progress()
    rec.save(train={k: v for k, v in state.items() if k not in ("losses", "seconds", "wall", "expected")}, best=best,
             loss_first_20=sum(state["losses"][:20]) / max(len(state["losses"][:20]), 1),
             loss_last_20=sum(state["losses"][-20:]) / max(len(state["losses"][-20:]), 1),
             passed=passed, seconds=round(time.time() - START, 1))
    return passed


def run_multi(rec, model_dir, out, data_path, *, served_dir, gpus, plan, chunk, deadline):
    """Several GPUs, layers split: what context trains whole, and how the cached-prefix gradient compares with the
    whole-context one on captured served requests (loss on the reply only). ``plan``: ``[(request, [tails])]``; each
    request gets a plain step (the reference when it fits, else its longest tail is) and one window step per tail."""
    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(served_dir or model_dir))
    captures = {}
    for line in Path(data_path).read_text().splitlines():
        capture = json.loads(line)
        captures[capture["request"]] = capture
    model, tokenizer, loaded = load(rec, model_dir, out, sequence_len=131072)
    placed = spread(model, gpus)
    rec.save(loaded=loaded, placed=rec.event("spread", layers_per_device=[placed.count(i) for i in range(gpus)],
                                             device_gib=device_gib(peak=False)))
    model.train()
    began = time.time()
    model.zero_grad(set_to_none=True)
    loss, _, _ = step(model, batch(tokenizer, 4096))
    rec.event("first_step", tokens=4096, seconds=round(time.time() - began, 1), loss=round(loss, 4))
    rows, rate = [], None  # rate: seconds per token of the last plain step, for the time check

    def measured(label, run, tokens):
        row = {"step": label, "tokens": tokens}
        model.zero_grad(set_to_none=True)
        reset_peaks()
        began = time.time()
        try:
            row["loss"] = round(run(), 5)
            row.update(seconds=round(time.time() - began, 1), fits=True, peak_gib=device_gib(),
                       gradients=gradients(model))
            flat = flat_gradients(model)
        except torch.OutOfMemoryError as error:
            row.update(fits=False, peak_gib=device_gib(), error=str(error)[:200])
            flat = None
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return row, flat

    for request, tails in plan:
        inputs, checks = encode_request(processor, captures[request])
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        total = inputs["input_ids"].shape[1]
        if rate and time.time() + 1.3 * rate * total > deadline:
            rows.append(rec.event("skipped", request=request, tokens=total, reason="time"))
            continue
        full, reference = measured("plain", lambda: step(model, inputs)[0], total)
        if full["fits"]:
            rate = full["seconds"] / total
        steps, flats = [full], {}
        for tail in tails:  # in the plan's order: the first windowed step of a process took 205 s (32K tail)
            if tail >= total or tail < checks["reply_tokens"] + 2 or time.time() + 60 > deadline:
                continue
            row, flat = measured(f"tail {tail}", lambda tail=tail: window_step(model, inputs, window=tail,
                                                                               chunk=chunk)[0], total)
            row["tail"] = tail
            steps.append(row)
            if flat is not None:
                flats[tail] = flat
        if reference is None and flats:
            reference = flats[max(flats)]
            full["reference"] = f"tail {max(flats)}"
        for row in steps[1:]:
            flat = flats.get(row["tail"])
            if flat is not None and reference is not None:
                row["gradient_cosine"] = {name: round(float(torch.nn.functional.cosine_similarity(
                    reference[name], flat[name], dim=0)), 4) for name in reference}
                row["norm_ratio"] = {name: round(float(flat[name].norm() / reference[name].norm().clamp_min(1e-30)), 4)
                                     for name in reference}
        rows.append(rec.event("request", request=request, tokens=total, reply_tokens=checks["reply_tokens"],
                              images=checks["images"], prompt_exact=checks["prompt_tokens"] == checks[
                                  "served_prompt_tokens"], steps=steps))
        rec.save(requests=rows)
        reference = flats = None
        torch.cuda.empty_cache()
    done = [row for row in rows if row.get("steps")]
    passed = bool(done) and all(any(step_row.get("fits") for step_row in row["steps"]) for row in done) and all(
        abs(step_row["loss"] - row["steps"][0]["loss"]) <= 0.02 * abs(row["steps"][0]["loss"]) + 2e-3
        for row in done if row["steps"][0].get("fits") for step_row in row["steps"][1:] if step_row.get("fits"))
    rec.save(requests=rows, passed=passed, seconds=round(time.time() - START, 1), gpus=gpus)
    return passed


def apply_table_scale(rec, model, model_dir):
    """The served NVFP4 checkpoint stores the n-gram table in FP8 with a ``weight_scale`` the stock loader drops
    (its load report lists the key as unexpected; the serving recipe patches its own loader for it). Without the
    scale the trainer's base scores at chance (span NLL 11.9, ``sft-qlora-nvfp4-b200-v6``)."""
    import torch
    from safetensors import safe_open

    weight_map = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    done = []
    for key, shard in weight_map.items():
        if not key.endswith("ngram_embedding.weight_scale"):
            continue
        with safe_open(str(Path(model_dir) / shard), framework="pt") as file:
            scale = file.get_tensor(key).float()
        name = key[:-len("_scale")]
        param = next(p for n, p in model.named_parameters() if n.endswith(name))
        stored = sorted(k for k in weight_map if k.startswith(name))[:3]  # the table is stored in pieces
        before = float(param.data[:65536].float().abs().mean())
        step = 1 << 22
        with torch.no_grad():
            for start in range(0, param.shape[0], step):  # in place, in pieces: the table is 51B values
                piece = param.data[start:start + step]
                rows = scale[start:start + step] if scale.dim() and scale.shape[0] == param.shape[0] else scale
                piece.copy_((piece.float() * rows).to(piece.dtype))
        done.append(rec.event("table_scale", key=key, scale_shape=list(scale.shape), stored_keys=stored,
                              scale_mean=float(scale.mean()), table_dtype=str(param.dtype), device=str(param.device),
                              abs_mean_before=round(before, 6),
                              abs_mean_after=round(float(param.data[:65536].float().abs().mean()), 6)))
    return done


def span_nll(model, rows):
    """Served-NLL prompt rows (``build_nll_bench``: token ids, text only, ``assistant_spans``): the trainer's mean
    negative log-likelihood of the span tokens, the number the server reports for the same row."""
    import torch

    values = []
    model.eval()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                tokens = torch.tensor([row["ids"]], device="cuda:0")
                labels = torch.full_like(tokens, -100)
                for first, last in row["assistant_spans"]:
                    labels[:, first:last] = tokens[:, first:last]
                values.append(round(float(model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                                labels=labels).loss.float()), 5))
    finally:
        model.train()
    return values


def run_nvfp4(rec, model_dir, out, corpus, data_path, *, gpus, deadline, steps=0):
    """Training against the served NVFP4 checkpoint: does it load, what does a step cost, and does
    the trainer score a served-NLL row as the engine does. Stages, each recorded as it ends: load on one GPU; span
    NLL of ``data_path`` rows; reply-loss steps on three captured requests (short, middle, longest) at growing
    tails on one GPU; the same on ``gpus`` GPUs with the layers split. A step that does not fit is a result."""
    import traceback

    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_dir))
    rows = json.loads(Path(data_path).read_text())["teacher_forced"] if data_path else []
    sizes = []
    with (Path(corpus) / "train.jsonl").open() as file:
        for number, line in enumerate(file):
            sizes.append((json.loads(line)["prompt_tokens"], number))
    sizes.sort()
    wanted = {"short": min(sizes, key=lambda s: abs(s[0] - 20000))[1],
              "middle": min(sizes, key=lambda s: abs(s[0] - 64000))[1], "long": sizes[-1][1]}
    captures = {}
    with (Path(corpus) / "train.jsonl").open() as file:
        for number, line in enumerate(file):
            for label, index in wanted.items():
                if index == number:
                    captures[label] = json.loads(line)
    model, _, loaded = load(rec, model_dir, out, sequence_len=131072)
    kinds = {}
    for name, param in model.named_parameters():
        if ".mlp.experts." in name:
            key = f"{name.split('.mlp.experts.')[1]} {type(param.data).__name__} {tuple(param.shape)} grad={param.requires_grad}"
            kinds[key] = kinds.get(key, 0) + 1
    rec.save(loaded=loaded, expert_parameters=rec.event("experts", kinds=kinds), device_gib=device_gib(peak=False))
    model.train()
    result = {"steps": [], "table_scale": apply_table_scale(rec, model, model_dir)}

    def stage(label, run):
        try:
            return run()
        except Exception as error:  # a stage that fails is recorded; the next one still runs
            result.setdefault("errors", []).append(rec.event("stage_error", stage=label, error=repr(error)[:600],
                                                             trace=traceback.format_exc()[-2500:]))
            torch.cuda.empty_cache()
            return None

    def nll_stage():
        began = time.time()
        result["span_nll"] = span_nll(model, rows)
        result["span_nll_seconds"] = round(time.time() - began, 1)
        rec.event("span_nll", mean=round(sum(result["span_nll"]) / len(result["span_nll"]), 5), rows=len(rows),
                  seconds=result["span_nll_seconds"])

    def measured(label, name, tail, devices):
        inputs, checks = encode_request(processor, captures[name])
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        total = inputs["input_ids"].shape[1]
        row = {"step": label, "request": name, "tokens": total, "tail": tail, "gpus": devices,
               "reply_tokens": checks["reply_tokens"]}
        model.zero_grad(set_to_none=True)
        reset_peaks()
        began = time.time()
        try:
            if tail is None or tail >= total:
                row["loss"] = round(step(model, inputs)[0], 5)
            else:
                loss, prefix, forward, backward = window_step(model, inputs, window=tail, chunk=16384)
                row.update(loss=round(loss, 5), prefix_seconds=round(prefix, 1), forward_seconds=round(forward, 1),
                           backward_seconds=round(backward, 1))
            row.update(seconds=round(time.time() - began, 1), fits=True, peak_gib=device_gib(),
                       gradients=gradients(model))
        except torch.OutOfMemoryError as error:
            row.update(fits=False, seconds=round(time.time() - began, 1), peak_gib=device_gib(), error=str(error)[:200])
        model.zero_grad(set_to_none=True)
        del inputs
        torch.cuda.empty_cache()
        result["steps"].append(rec.event("step", **row))
        rec.save(**result)
        return row

    lengths = {name: capture["prompt_tokens"] for name, capture in captures.items()}

    def ladder_stage(devices, plan):
        slowest, too_big = 0.0, 10 ** 9  # too_big: the smallest gradient window that did not fit
        for name, tail in plan:
            cost = min(tail or lengths[name], lengths[name])
            if cost >= too_big:
                rec.event("skipped", request=name, tail=tail, reason="a smaller window did not fit")
                continue
            if time.time() + max(120.0, 1.5 * slowest) > deadline:
                rec.event("skipped", request=name, tail=tail, reason="time")
                continue
            row = stage(f"{devices} GPU {name} {tail}", lambda: measured(f"{devices} GPU", name, tail, devices))
            if row and row.get("fits"):
                slowest = max(slowest, row["seconds"])
            elif row:
                too_big = min(too_big, cost)

    if rows:
        stage("span nll", nll_stage)
    rec.save(**result)
    if max(result.get("span_nll") or [0]) > 2:  # v6, 8 October: 11.9 with the table's scale dropped at load
        rec.event("refused", reason="base span NLL above 2 on a probe row: not the served model")
        rec.save(**result, passed=False, seconds=round(time.time() - START, 1))
        return False
    # one GPU: the cached-prefix step of the first adapter, then longer tails; the whole short request
    ladder_stage(1, [("short", None), ("middle", 16384), ("long", 16384), ("long", 16384), ("long", 24576),
                     ("long", 32768), ("middle", 32768), ("long", 49152), ("middle", None), ("long", 65536),
                     ("long", 98304), ("long", None)][:steps or None])
    if gpus > 1 and time.time() + 300 < deadline:
        placed = stage("spread", lambda: spread(model, gpus))
        if placed:
            rec.event("spread", layers_per_device=[placed.count(i) for i in range(gpus)],
                      device_gib=device_gib(peak=False))
            ladder_stage(gpus, [("long", 16384), ("long", 32768), ("long", 49152), ("middle", None), ("long", 65536),
                                ("long", None)])
    fits = [row for row in result["steps"] if row.get("fits")]
    passed = bool(fits) and (not rows or "span_nll" in result)
    rec.save(**result, passed=passed, seconds=round(time.time() - START, 1))
    return passed


def run_overlay_check(rec, overlay_dir, served_dir):
    """Every tensor of an exported overlay's own (not linked) shards against the served checkpoint: the adapted ones
    may differ, all others must be bit-identical."""
    import torch
    from safetensors import safe_open

    counts = {"shards": 0, "adapted": 0, "adapted_changed": 0, "other": 0, "other_identical": 0, "missing": 0}
    int8 = {}

    def int8_rows(weight):  # arc_infra.dense_kernels_runtime.quantize_rows, dequantized
        weight = weight.float()
        scale = (weight.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-12)
        return torch.round(weight / scale).clamp_(-127, 127) * scale

    for path in sorted(Path(overlay_dir).glob("model-*.safetensors")):
        if path.is_symlink():
            continue
        counts["shards"] += 1
        with safe_open(str(path), framework="pt") as ours, safe_open(str(Path(served_dir) / path.name),
                                                                     framework="pt") as theirs:
            counts["missing"] += len(set(ours.keys()) ^ set(theirs.keys()))
            for name in sorted(set(ours.keys()) & set(theirs.keys())):
                a, b = ours.get_tensor(name), theirs.get_tensor(name)
                same = a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(
                    a.view(torch.int16) if a.dtype == torch.bfloat16 else a,
                    b.view(torch.int16) if b.dtype == torch.bfloat16 else b))
                if TARGET_TENSOR.match(name):
                    counts["adapted"] += 1
                    counts["adapted_changed"] += not same
                    if INT8_SERVED.search(name):  # what the served INT8 kernel keeps of the merged change
                        delta, kept = a.float() - b.float(), int8_rows(a) - int8_rows(b)
                        group = int8.setdefault(group_of(name), {"tensors": 0, "delta_sq": 0.0, "kept_sq": 0.0,
                                                                 "dot": 0.0, "step_sq": 0.0, "weight_sq": 0.0})
                        group["tensors"] += 1
                        group["delta_sq"] += float((delta * delta).sum())
                        group["kept_sq"] += float((kept * kept).sum())
                        group["dot"] += float((kept * delta).sum())
                        step = b.float().abs().amax(dim=1) / 127
                        group["step_sq"] += float((step * step).sum()) * b.shape[1]
                        group["weight_sq"] += float((b.float() ** 2).sum())
                else:
                    counts["other"] += 1
                    counts["other_identical"] += same
    passed = counts["missing"] == 0 and counts["other"] == counts["other_identical"] and counts["adapted"] == 372
    # projection: the kept change along the intended one (1 = all of it); noise: the rest, in units of the change
    served = {name: {"tensors": g["tensors"], "delta_over_weight": round((g["delta_sq"] / g["weight_sq"]) ** 0.5, 5),
                     "step_over_weight": round((g["step_sq"] / g["weight_sq"]) ** 0.5, 5),
                     "projection": round(g["dot"] / g["delta_sq"], 4),
                     "cosine": round(g["dot"] / max((g["kept_sq"] * g["delta_sq"]) ** 0.5, 1e-30), 4),
                     "noise_over_delta": round((max(g["kept_sq"] - g["dot"] ** 2 / g["delta_sq"], 0.0)
                                                / g["delta_sq"]) ** 0.5, 3)} for name, g in int8.items()}
    rec.event("overlay_check", **counts, int8_served=served)
    rec.save(overlay_check={**counts, "int8_served": served}, passed=passed, seconds=round(time.time() - START, 1))
    return passed


def tiny_model(real_dir, out):
    """A random Qwen4-Exp with the real layer pattern at toy sizes, saved with the real tokenizer and processor."""
    import torch
    from transformers import AutoConfig, AutoModelForMultimodalLM

    data = json.loads((Path(real_dir) / "config.json").read_text())
    data["text_config"].update(
        hidden_size=64, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=2, num_key_value_heads=1, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, num_experts=8, num_experts_per_tok=2, hc_lowrank=8, ple_embed_dim=64,
        ngram_vocab_size_base=5000, split_ngram_parts=4, indexer_n_heads=2, indexer_budget=64)
    data["vision_config"].update(depth=2, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=64)
    target = Path(out) / "tiny-model"
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text(json.dumps(data))
    torch.manual_seed(0)
    model = AutoModelForMultimodalLM.from_config(AutoConfig.from_pretrained(str(target))).to(torch.bfloat16)
    model.save_pretrained(str(target))
    for path in Path(real_dir).iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name not in (
                "config.json", "model.safetensors.index.json", "generation_config.json") and not (target / path.name).exists():
            shutil.copy(path, target / path.name)
    return target


def read_speed(rec, model_dir, *, shards=2):
    files = sorted(Path(model_dir).glob("model-*.safetensors"))[40:40 + shards]
    began, count = time.time(), 0
    for path in files:
        with path.open("rb") as file:
            while chunk := file.read(64 * 2 ** 20):
                count += len(chunk)
    seconds = time.time() - began
    total = sum(p.stat().st_size for p in Path(model_dir).glob("model-*.safetensors"))
    return rec.event("volume_read", files=len(files), gib=round(count / 2 ** 30, 2), seconds=round(seconds, 1),
                     gib_per_second=round(count / 2 ** 30 / seconds, 2), checkpoint_gib=round(total / 2 ** 30, 1),
                     projected_read_minutes=round(total / max(count / seconds, 1) / 60, 1))


def run_parity(rec, bf16_dir, served_dir):
    """Adapted tensors: served checkpoint against the training checkpoint, byte for byte."""
    from safetensors import safe_open

    def tensors(folder):
        index = json.loads((Path(folder) / "model.safetensors.index.json").read_text())["weight_map"]
        return {name: shard for name, shard in index.items() if TARGET_TENSOR.match(name)}

    train, served = tensors(bf16_dir), tensors(served_dir)
    missing = sorted(set(train) ^ set(served))
    handles, different, groups, count = {}, [], {}, 0

    def digest(folder, shard, name):
        import torch

        key = (str(folder), shard)
        if key not in handles:
            handles[key] = safe_open(str(Path(folder) / shard), framework="pt")
        tensor = handles[key].get_tensor(name).contiguous()
        raw = tensor.view(torch.int16) if tensor.dtype == torch.bfloat16 else tensor
        return str(tensor.dtype), tuple(tensor.shape), hashlib.sha256(raw.numpy().tobytes()).hexdigest()

    for name in sorted(set(train) & set(served)):
        a, b = digest(bf16_dir, train[name], name), digest(served_dir, served[name], name)
        group = groups.setdefault(group_of(name.replace("mlp.", "")), {"tensors": 0, "identical": 0, "dtypes": set()})
        group["tensors"] += 1
        group["identical"] += a == b
        group["dtypes"].add(f"{a[0]}/{b[0]}")
        if a != b:
            different.append({"name": name, "train": a[:2], "served": b[:2]})
        count += 1
        if count % 60 == 0:
            rec.event("parity_progress", tensors=count, different=len(different))
    groups = {name: {**g, "dtypes": sorted(g["dtypes"])} for name, g in groups.items()}
    passed = not missing and not different and count > 0
    rec.event("parity", tensors=count, different=len(different), only_in_one=len(missing), groups=groups)
    rec.save(parity={"tensors": count, "different": different[:20], "only_in_one": missing[:20], "groups": groups},
             passed=passed, seconds=round(time.time() - START, 1))
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "smoke", "parity", "adapter", "export", "contexts", "overlay", "nvfp4",
                                         "multi", "train", "worker"])
    parser.add_argument("--corpus", help="train: folder with train.jsonl and select.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="train: at most this many examples")
    parser.add_argument("--eval-every", type=int, default=125)
    parser.add_argument("--stop-after", type=int, default=0, help="train: leave after this many steps (restart test)")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--probe-tokens", type=int, default=384)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--plan", default="", help="multi: request:tail,tail;request:... (captured request numbers)")
    parser.add_argument("--tiny", action="store_true", help="multi: the tiny random model (a test of the code)")
    parser.add_argument("--bf16-linear", action="store_true", help="keep the linear layers in BF16 (not NF4)")
    parser.add_argument("--nvfp4", action="store_true", help="--model is the served NVFP4 checkpoint (SonicMoE)")
    parser.add_argument("--expert-lora", action="store_true", help="nvfp4: merge-aware adapters on routed experts")
    parser.add_argument("--moe-backend", default="sonicmoe", choices=["sonicmoe", "scattermoe"],
                        help="nvfp4: Axolotl's expert kernels (merge-aware expert adapters need sonicmoe)")
    parser.add_argument("--data", help="adapter: {train: [text], eval: [text]} of DEV-game contexts")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--tail", type=int, default=4096)
    parser.add_argument("--adapter", help="export: the adapter folder")
    parser.add_argument("--experts", help="export: the expert overlay folder served beside the merged shards")
    parser.add_argument("--overlay", help="export: the overlay folder to write")
    parser.add_argument("--model", required=True)
    parser.add_argument("--served")
    parser.add_argument("--out", required=True)
    parser.add_argument("--lengths", default="8192,16384,32768,49152,65536,98304,131072")
    parser.add_argument("--seconds", type=int, default=2400, help="the job's own time budget")
    parser.add_argument("--windows", default="", help="context:window pairs for the cached-prefix step, e.g. "
                                                      "65536:16384,114688:16384")
    parser.add_argument("--compare", default="32768:16384", help="context:window pairs compared with a plain step")
    parser.add_argument("--text", help="a real context to draw tokens from instead of this file's repeated source")
    parser.add_argument("--chunk", type=int, default=16384, help="tokens per no-gradient prefix forward")
    parser.add_argument("--phases", default="", help="train: gradient windows of the curriculum's passes, shortest "
                        "first (16384,32768,49152); each pass covers every chain, all its replies in one example")
    parser.add_argument("--mix", type=float, default=0.25, help="train: share of chains on an earlier phase's window")
    parser.add_argument("--select-limit", type=int, default=0, help="train: selection rows per evaluation (0: all)")
    parser.add_argument("--replicas", type=int, default=1, help="train: GPUs, one model copy and one unit each per "
                        "update (data parallel)")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--rslora", action="store_true", help="rank-stabilized scaling (alpha / sqrt(rank))")
    args = parser.parse_args()
    set_adapter(args.rank, args.rslora)
    BF16_LINEAR["on"] = args.bf16_linear
    NVFP4.update(on=args.nvfp4 or args.mode == "nvfp4", experts=args.expert_lora, backend=args.moe_backend)
    rec = Recorder(args.out)
    deadline = START + args.seconds
    if args.mode == "parity":
        passed = run_parity(rec, args.model, args.served)
    elif args.mode == "export":
        passed = run_export(rec, args.adapter, args.served, args.experts, args.overlay)
    elif args.mode == "overlay":
        passed = run_overlay_check(rec, args.overlay, args.served)
    elif args.mode == "train":
        rec.save(environment=environment(rec))
        model_dir = tiny_model(args.model, args.out) if args.tiny else args.model
        worker = [sys.executable, "-u", __file__, "worker", "--model", str(model_dir), "--corpus", args.corpus,
                  "--tail", str(args.tail), "--phases", args.phases, "--rank", str(args.rank), "--moe-backend",
                  args.moe_backend, *(["--served", args.served] if args.served else []),
                  *(flag for flag, on in (("--rslora", args.rslora), ("--nvfp4", args.nvfp4),
                                          ("--bf16-linear", args.bf16_linear)) if on)]
        for stop_after in ([args.stop_after] if args.stop_after else []) + [0]:  # the second pass is the restart
            passed = run_train(rec, model_dir, args.out, args.corpus, served_dir=args.served, tail=args.tail,
                               lr=args.lr, deadline=deadline, limit=args.limit, eval_every=args.eval_every,
                               checkpoint_every=args.checkpoint_every, probe_tokens=args.probe_tokens,
                               stop_after=stop_after, phases=[int(v) for v in args.phases.split(",") if v],
                               mix=args.mix, select_limit=args.select_limit, replicas=args.replicas,
                               worker_command=worker)
    elif args.mode == "worker":
        passed = run_worker(rec, args.model, args.out, args.corpus, served_dir=args.served, tail=args.tail,
                            phases=[int(v) for v in args.phases.split(",") if v])
    elif args.mode == "nvfp4":
        rec.save(environment=environment(rec))
        passed = run_nvfp4(rec, args.model, args.out, args.corpus, args.data, gpus=args.gpus, deadline=deadline - 60,
                           steps=args.limit)
    elif args.mode == "multi":
        rec.save(environment=environment(rec))
        plan = [(int(item.split(":")[0]), [int(v) for v in item.split(":")[1].split(",") if v])
                for item in args.plan.split(";") if item]
        passed = run_multi(rec, tiny_model(args.model, args.out) if args.tiny else args.model, args.out, args.data,
                           served_dir=args.served, gpus=args.gpus, plan=plan, chunk=args.chunk, deadline=deadline - 30)
    elif args.mode == "contexts":
        rec.save(environment=environment(rec))
        passed = run_contexts(rec, args.model, args.out, args.data, served_dir=args.served)
    elif args.mode == "adapter":
        rec.save(environment=environment(rec))
        passed = run_adapter(rec, args.model, args.out, args.data, steps=args.steps, lr=args.lr, tokens=args.tokens,
                             tail=args.tail, deadline=deadline)
    else:
        rec.save(environment=environment(rec))
        if args.mode == "check" and args.data:  # the adapter mode on the tiny model
            passed = run_adapter(rec, tiny_model(args.model, args.out), args.out, args.data, steps=3, lr=args.lr,
                                 tokens=2048, tail=1024, deadline=deadline)
        elif args.mode == "check":
            rec.save(volume_read=read_speed(rec, args.model))
            passed = run_smoke(rec, tiny_model(args.model, args.out), args.out, [512, 4096, 16384], deadline=deadline,
                               overfit_tokens=2048, window_plan={"compare": [(4096, 1024)], "chunk": 1000, "pairs": [
                                   (8192, 1024), (16384, 2048)]})
        else:
            pair = lambda text: tuple(int(v) for v in text.split(":"))  # noqa: E731
            TEXT["path"] = args.text
            plan = {"compare": [pair(item) for item in args.compare.split(",")], "chunk": args.chunk,
                    "pairs": [pair(item) for item in args.windows.split(",") if item]} if (
                        args.windows or args.text) else None
            passed = run_smoke(rec, args.model, args.out, [int(v) for v in args.lengths.split(",")],
                               deadline=deadline, overfit_tokens=8192, window_plan=plan)
    rec.event("done", passed=passed)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
