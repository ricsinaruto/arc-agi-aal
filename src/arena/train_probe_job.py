"""Training-cost probe job (RL restart step 0, 11 October): what one full-fine-tuning step costs at a context length.

Runs in the serving runtime's Python (torch + transformers) after the server has stopped; standalone (stdlib, torch,
transformers only). For each length of the ladder: one sequence of random token ids, a forward pass to the last hidden
state with per-layer gradient checkpointing, the loss on ``label_fraction`` of the positions through the LM head in
chunks (the hidden state's gradient is accumulated chunk by chunk, so the vocabulary-sized logits of only one chunk
exist at a time), the backward pass, and one AdamW step over every language-model weight (vision tower frozen).
Reported per length: seconds of each part and the peak GPU memory. The ladder stops at the first length that does not
fit or would pass the deadline. Nothing is saved but the report.

``--storage bf16``: weights, gradients and AdamW state in BF16 (the checkpoint's dtype).
``--storage fp32``: FP32 weights and optimizer state, BF16 autocast for the passes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def event(name, **fields):
    print(json.dumps({"event": "train_probe_" + name, **fields}), flush=True)


def versions():
    import importlib
    import importlib.metadata as metadata

    found = {}
    for package in ("torch", "transformers", "accelerate", "safetensors", "einops", "triton", "fla-core",
                    "flash-linear-attention"):
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            found[package] = None
    try:
        importlib.import_module("fla.ops.gated_delta_rule")
        found["fla_importable"] = True
    except Exception as exc:  # noqa: BLE001 - recorded: the linear-attention layers then run the reference torch path
        found["fla_importable"] = f"{type(exc).__name__}: {exc}"[:300]
    return found


def load(model_dir, storage, device):
    import torch
    import transformers

    config = json.loads((Path(model_dir) / "config.json").read_text())
    cls = getattr(transformers, config["architectures"][0])
    dtype = torch.bfloat16 if storage == "bf16" else torch.float32
    try:
        model = cls.from_pretrained(str(model_dir), dtype=dtype)
    except TypeError:
        model = cls.from_pretrained(str(model_dir), torch_dtype=dtype)
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    visual = getattr(model.model, "visual", None)
    if visual is not None:
        visual.requires_grad_(False)
    return model


def step(model, optimizer, ids, *, storage, label_fraction, chunk, sync):
    """One training step on ``ids`` [1, L]; returns the seconds of each part and the loss."""
    import torch
    import torch.nn.functional as F

    device = ids.device
    # CPU (the dry run): plain FP32, BF16 matmuls are too slow there
    cast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=storage == "fp32" and device.type == "cuda")
    labelled = torch.arange(ids.shape[1] - 1, device=device)
    labelled = labelled[labelled % 1000 < round(1000 * label_fraction)]  # deterministic share of the positions
    times, start = {}, time.perf_counter()
    with cast:
        hidden = model.model(input_ids=ids).last_hidden_state[0]
    sync()
    times["forward"] = time.perf_counter() - start
    start = time.perf_counter()
    detached = hidden.detach().requires_grad_(True)
    total = 0.0
    for at in range(0, len(labelled), chunk):
        rows = labelled[at:at + chunk]
        with cast:
            logits = model.lm_head(detached[rows])
        loss = F.cross_entropy(logits.float(), ids[0, rows + 1], reduction="sum") / len(labelled)
        loss.backward()
        total += loss.item()
    sync()
    times["head"] = time.perf_counter() - start
    start = time.perf_counter()
    hidden.backward(detached.grad)
    sync()
    times["backward"] = time.perf_counter() - start
    start = time.perf_counter()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sync()
    times["optimizer"] = time.perf_counter() - start
    return times, total, len(labelled)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--storage", choices=("bf16", "fp32"), required=True)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--repeat-below", type=int, default=0, help="a second timed step at lengths up to this")
    parser.add_argument("--label-fraction", type=float, default=0.5)
    parser.add_argument("--head-chunk", type=int, default=2048)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    import torch

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    report = {"storage": args.storage, "lengths": args.lengths, "label_fraction": args.label_fraction,
              "head_chunk": args.head_chunk, "versions": versions(), "rows": [], "stopped": None,
              "gradient_checkpointing": True, "optimizer": "AdamW (fused on GPU), every language-model weight"}

    def save():
        out.write_text(json.dumps(report, indent=1))

    save()
    started = time.perf_counter()
    model = load(args.model_dir, args.storage, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    report.update(load_seconds=round(time.perf_counter() - started, 1),
                  parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in trainable),
                  gpu=torch.cuda.get_device_name(0) if cuda else None,
                  gpu_total_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if cuda else None,
                  weights_gib=round(torch.cuda.memory_allocated() / 2**30, 2) if cuda else None)
    try:
        optimizer = torch.optim.AdamW(trainable, lr=1e-6, weight_decay=0.0, fused=cuda)
    except (RuntimeError, TypeError, ValueError) as exc:
        report["optimizer"] += f"; not fused ({type(exc).__name__})"
        optimizer = torch.optim.AdamW(trainable, lr=1e-6, weight_decay=0.0)
    event("loaded", **{k: report[k] for k in ("storage", "load_seconds", "parameters", "trainable_parameters",
                                               "weights_gib", "versions")})
    save()
    generator = torch.Generator().manual_seed(0)
    vocabulary = min(model.config.text_config.vocab_size, 150000)
    per_token = None
    for index, length in enumerate(args.lengths):
        left = args.deadline_epoch - time.time()
        steps = 2 + (length <= args.repeat_below) if index == 0 else 1 + (length <= args.repeat_below)
        if per_token is not None and steps * per_token * length * 1.3 > left or left < 20:
            report["stopped"] = {"length": length, "reason": "deadline", "seconds_left": round(left, 1)}
            break
        ids = torch.randint(1000, vocabulary, (1, length), generator=generator).to(device)
        row = {"length": length, "steps": []}
        try:
            for number in range(steps):
                if cuda:
                    torch.cuda.reset_peak_memory_stats()
                times, loss, labelled = step(model, optimizer, ids, storage=args.storage,
                                             label_fraction=args.label_fraction, chunk=args.head_chunk, sync=sync)
                entry = {**{k: round(v, 3) for k, v in times.items()}, "seconds": round(sum(times.values()), 3),
                         "loss": round(loss, 4), "labelled": labelled, "warmup": index == 0 and number == 0,
                         "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None,
                         "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2) if cuda else None}
                row["steps"].append(entry)
            timed = [s for s in row["steps"] if not s["warmup"]]
            row.update(seconds=min(s["seconds"] for s in timed),
                       tokens_per_second=round(length / min(s["seconds"] for s in timed), 1),
                       peak_allocated_gib=max((s["peak_allocated_gib"] or 0) for s in row["steps"]) if cuda else None)
            per_token = row["seconds"] / length
        except torch.OutOfMemoryError as exc:
            row["out_of_memory"] = str(exc)[:300]
            report["stopped"] = {"length": length, "reason": "out_of_memory"}
        report["rows"].append(row)
        event("length", storage=args.storage, **{k: v for k, v in row.items() if k != "steps"})
        save()
        del ids
        if report["stopped"]:
            break
    fits = [r["length"] for r in report["rows"] if "out_of_memory" not in r]
    report["largest_length_run"] = max(fits) if fits else None
    report["seconds"] = round(time.perf_counter() - started, 1)
    save()
    event("done", storage=args.storage, largest_length_run=report["largest_length_run"], stopped=report["stopped"],
          seconds=report["seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
