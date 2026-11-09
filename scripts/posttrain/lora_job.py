"""The Ornith-1.5-35B-A3B LoRA test jobs (``docs/posttrain/lora-test-plan.md``), run inside the Modal
container of ``scripts/modal_lora.py`` (and, on a tiny random model, by the CPU tests):

- ``fetch``: download the BF16 training checkpoint and the served NVFP4 checkpoint at their pinned
  revisions into the assets volume and verify every file (sha256 for LFS files, the git blob id for
  the small ones);
- ``smoke``: load, kernel checks (Gated-DeltaNet FLA vs the torch reference, forward and gradients;
  SDPA without the math backend), a few optimizer steps at 32K and 16K (tokens/s, peak memory, finite
  loss, gradients reaching the linear-attention LoRA weights), held-out loss on a few packs, and a
  projection of the training time;
- ``train``: M0 (r 32, alpha 64, dropout 0.05, lr 1e-4, 8 warmup steps then cosine to 10%, 2 epochs,
  accumulation 2, clip 1.0, BF16 weights with FP32 LoRA weights, gradient checkpointing), held-out
  loss before training and every half epoch, the adapter saved at every evaluation and at the end;
- ``merge``: merge the adapter into the adapted BF16 tensors, re-quantize exactly those tensors to the
  served encoding (``posttrain.requant``), check that re-quantizing the unchanged BF16 tensors
  reproduces every served tensor bit for bit (the zero-adapter check), and write the overlay
  (``arena.weights_overlay``).

Every job prints one JSON line per event (``modal app logs`` shows progress) and writes its files under
its output folder. The heavy imports (torch, transformers, peft) happen inside the functions.
"""

import base64
import hashlib
import io
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parents[1] / "src", HERE / "src"):  # repo layout, then the Modal image layout
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from arena import weights_overlay  # noqa: E402
from posttrain import requant  # noqa: E402

BF16 = {"repo": "ornith-ai/Ornith-1.5-35B-A3B", "revision": "10fbf86fed7ecee4a061f8b499a618f46001cac1"}
NVFP4 = {"repo": "ornith-ai/Ornith-1.5-35B-A3B-NVFP4", "revision": "94e431d9cc47fa1986a7a1a4e9a80f7f118b03aa"}
FETCH_PATTERNS = ["*.safetensors", "*.json", "*.jinja", "merges.txt", "vocab.json"]


@dataclass
class TrainConfig:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    lr: float = 1e-4
    warmup_steps: int = 8
    min_lr_ratio: float = 0.1
    epochs: int = 2
    grad_accum: int = 2
    clip: float = 1.0
    weight_decay: float = 0.0
    seed: int = 0
    ce_chunk: int = 4096
    evals_per_epoch: int = 2
    train_probe: int = 8
    experts_implementation: str = "grouped_mm"
    target_regex: str = requant.TARGET_REGEX


class Log:
    """JSON lines to stdout (flushed, for ``modal app logs``) and to a file."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.started = time.monotonic()

    def __call__(self, event, **fields):
        row = {"event": event, "t": round(time.monotonic() - self.started, 1), **fields}
        line = json.dumps(row, default=str)
        print(line, flush=True)
        if self.path:
            with open(self.path, "a") as handle:
                handle.write(line + "\n")
        return row


# --- files -----------------------------------------------------------------------------------------


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify_inventory(folder, inventory, names=None):
    """Check ``folder`` against ``{file: {bytes, sha256 | blob_id}}``; returns the checked names."""
    from concurrent.futures import ThreadPoolExecutor

    folder = Path(folder)
    names = sorted(names or inventory)

    def check(name):
        spec, path = inventory[name], folder / name
        if not path.is_file() or path.stat().st_size != spec["bytes"]:
            raise RuntimeError(f"{path}: missing or not {spec['bytes']} bytes")
        if spec.get("sha256"):
            if weights_overlay.sha256_file(path) != spec["sha256"]:
                raise RuntimeError(f"{path}: sha256 mismatch")
        elif git_blob_sha1(path) != spec["blob_id"]:
            raise RuntimeError(f"{path}: git blob id mismatch")
        return name

    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(check, names))


def fetch(root, *, bf16_inventory, nvfp4_inventory, log):
    """Download both checkpoints (skipped when a verified copy is present) and verify them."""
    from huggingface_hub import snapshot_download

    root = Path(root)
    out = {}
    for label, pin, inventory in (("bf16", BF16, bf16_inventory), ("nvfp4", NVFP4, nvfp4_inventory)):
        folder = checkpoint_dir(root, label)
        marker = folder / ".verified.json"
        wanted = sorted(n for n in inventory["files"] if n.endswith(tuple(p.lstrip("*") for p in FETCH_PATTERNS)))
        if marker.is_file() and json.loads(marker.read_text()).get("files") == wanted:
            log("fetch_cached", checkpoint=label, folder=str(folder))
            out[label] = str(folder)
            continue
        started = time.monotonic()
        snapshot_download(pin["repo"], revision=pin["revision"], local_dir=folder, allow_patterns=FETCH_PATTERNS,
                          max_workers=16)
        log("fetch_downloaded", checkpoint=label, seconds=round(time.monotonic() - started),
            gb=round(sum((folder / n).stat().st_size for n in wanted) / 1e9, 2))
        verify_inventory(folder, inventory["files"], wanted)
        marker.write_text(json.dumps({"files": wanted, "repo": pin["repo"], "revision": pin["revision"]}))
        log("fetch_verified", checkpoint=label, files=len(wanted), seconds=round(time.monotonic() - started))
        out[label] = str(folder)
    return out


def checkpoint_dir(root, label):
    pin = BF16 if label == "bf16" else NVFP4
    return Path(root) / f"ornith-{label}-{pin['revision'][:8]}"


def load_records(path, expected_sha256=None):
    path = Path(path)
    if expected_sha256 and weights_overlay.sha256_file(path) != expected_sha256:
        raise RuntimeError(f"{path}: sha256 differs from the M0 manifest")
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_m0(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    train = load_records(folder / "train.jsonl", manifest["files"]["train.jsonl"]["sha256"])
    evaluation = load_records(folder / "eval.jsonl", manifest["files"]["eval.jsonl"]["sha256"])
    return manifest, train, evaluation


def truncate_record(record, max_tokens, image_token_id, image_tokens=64):
    """A copy cut after the last whole target inside ``max_tokens`` (None when no target fits)."""
    spans = [s for s in record["loss_spans"] if s[1] <= max_tokens]
    if not spans:
        return None
    cut = spans[-1][1]
    ids = record["input_ids"][:cut]
    n_images = ids.count(image_token_id)
    if n_images % image_tokens:
        raise ValueError(f"{record['id']}: the cut splits an image")
    return {**record, "input_ids": ids, "loss_spans": spans, "images": record["images"][:n_images // image_tokens],
            "tokens": cut, "loss_tokens": sum(e - s for s, e in spans)}


# --- schedule --------------------------------------------------------------------------------------


def lr_at(step, total_steps, cfg):
    """Linear warmup over ``warmup_steps`` then cosine from ``lr`` to ``min_lr_ratio * lr``; ``step`` counts
    optimizer steps from 0."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def epoch_order(n, epoch, seed):
    order = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(order)
    return order


def plan_steps(n_records, cfg):
    """Optimizer steps per epoch and the evaluation points (after these optimizer steps, 1-based)."""
    per_epoch = math.ceil(n_records / cfg.grad_accum)
    total = per_epoch * cfg.epochs
    points = sorted({round(per_epoch * (e + (k + 1) / cfg.evals_per_epoch))
                     for e in range(cfg.epochs) for k in range(cfg.evals_per_epoch)})
    return per_epoch, total, [p for p in points if p > 0]


# --- model -----------------------------------------------------------------------------------------


def environment():
    import importlib.metadata as md

    import torch

    info = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available()}
    for package in ("transformers", "peft", "accelerate", "safetensors", "flash-linear-attention", "fla-core",
                    "triton", "torchvision", "tokenizers", "huggingface-hub"):
        try:
            info[package] = md.version(package)
        except md.PackageNotFoundError:
            info[package] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, gpu_memory_gib=round(props.total_memory / 2**30, 1),
                    capability=f"{props.major}.{props.minor}")
    return info


def load_model(model_dir, cfg, *, device="cuda", log):
    """BF16 model (MTP not loaded), frozen, gradient checkpointing on, LoRA on the plan's targets."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoImageProcessor, AutoModelForImageTextToText

    started = time.monotonic()
    kwargs = {"dtype": torch.bfloat16, "attn_implementation": "sdpa",
              "experts_implementation": cfg.experts_implementation}
    if device != "cpu":
        kwargs["device_map"] = {"": device}
    model = AutoModelForImageTextToText.from_pretrained(model_dir, **kwargs)
    model.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout, bias="none",
                      target_modules=cfg.target_regex)
    model = get_peft_model(model, lora)
    model.train()  # from_pretrained returns eval mode: no gradient checkpointing, no LoRA dropout
    processor = AutoImageProcessor.from_pretrained(model_dir)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    groups = lora_groups(trainable)
    checkpointing = sorted({getattr(m, "gradient_checkpointing", None) for m in model.modules()
                            if type(m).__name__.endswith("DecoderLayer")} - {None})
    info = {"seconds": round(time.monotonic() - started, 1), "training_mode": model.training,
            "decoder_gradient_checkpointing": checkpointing,
            "trainable_parameters": sum(p.numel() for p in trainable.values()),
            "trainable_dtype": sorted({str(p.dtype) for p in trainable.values()}),
            "lora_modules": {g: len(v) // 2 for g, v in groups.items()},
            "experts_implementation": getattr(model.get_base_model().config, "_experts_implementation", None),
            "image_processor": type(processor).__name__}
    if device != "cpu":
        info["memory_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    log("model_loaded", **info)
    return model, processor, info


def lora_groups(named):
    """LoRA parameters by target group (full attention, linear attention, shared expert)."""
    groups = {"full_attention": [], "linear_attention": [], "shared_expert": [], "other": []}
    for name in named:
        group = ("full_attention" if ".self_attn." in name else "linear_attention" if ".linear_attn." in name
                 else "shared_expert" if ".shared_expert." in name else "other")
        groups[group].append(name)
    return {g: v for g, v in groups.items() if v}


def image_tensors(record, processor):
    from PIL import Image

    if not record["images"]:
        return None, None
    images = [Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
              for url in record["images"]]
    batch = processor(images=images, return_tensors="pt")
    return batch["pixel_values"], batch["image_grid_thw"]


def prepare(record, processor, image_token_id, merge_size, device):
    import torch

    ids = torch.tensor(record["input_ids"], dtype=torch.long)[None]
    pixel_values, grid = image_tensors(record, processor)
    if grid is not None:
        per_image = (grid.prod(-1) // (merge_size ** 2)).tolist()
        if sum(per_image) != int((ids == image_token_id).sum()):
            raise ValueError(f"{record['id']}: {per_image} image features for "
                             f"{int((ids == image_token_id).sum())} image tokens")
    positions = torch.cat([torch.arange(s, e) for s, e in record["loss_spans"]])
    if int(positions.min()) < 1:
        raise ValueError(f"{record['id']}: a target at position 0 has no context")
    batch = {"input_ids": ids.to(device), "mm_token_type_ids": (ids == image_token_id).int().to(device),
             "targets": positions.to(device)}
    if grid is not None:
        batch.update(pixel_values=pixel_values.to(device), image_grid_thw=grid.to(device))
    return batch


def _ce_chunk(lm_head, hidden, labels):
    import torch.nn.functional as F

    return F.cross_entropy(lm_head(hidden).float(), labels, reduction="sum")


def sequence_loss(model, batch, ce_chunk):
    """Summed cross-entropy over the target tokens and their count. Logits exist for the target
    positions only, one chunk at a time (each chunk recomputed in backward)."""
    import torch
    from torch.utils.checkpoint import checkpoint

    hf = model.get_base_model()
    kwargs = {k: batch[k] for k in ("pixel_values", "image_grid_thw") if k in batch}
    out = hf.model(input_ids=batch["input_ids"], mm_token_type_ids=batch["mm_token_type_ids"], use_cache=False,
                   **kwargs)
    hidden = out.last_hidden_state[0]
    targets = batch["targets"]
    labels = batch["input_ids"][0, targets]
    states = hidden[targets - 1]
    total = torch.zeros((), device=hidden.device, dtype=torch.float32)
    for start in range(0, len(targets), ce_chunk):
        part = slice(start, start + ce_chunk)
        if torch.is_grad_enabled():
            total = total + checkpoint(_ce_chunk, hf.lm_head, states[part], labels[part], use_reentrant=False)
        else:
            total = total + _ce_chunk(hf.lm_head, states[part], labels[part])
    return total, len(targets)


def sdpa_context(device):
    """SDPA without the math backend on GPU (the math path needs T^2 memory at head_dim 256)."""
    import contextlib

    if device == "cpu":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])


def evaluate(model, records, processor, ids, *, cfg, device, log, label):
    import torch

    model.eval()
    started, total, count, rows = time.monotonic(), 0.0, 0, []
    with torch.no_grad(), sdpa_context(device):
        for record in records:
            batch = prepare(record, processor, ids["image"], ids["merge"], device)
            loss, n = sequence_loss(model, batch, cfg.ce_chunk)
            total += float(loss)
            count += n
            rows.append({"id": record["id"], "nll": round(float(loss) / n, 5), "tokens": n})
    model.train()
    result = {"label": label, "nll": total / max(1, count), "loss_tokens": count, "sequences": len(records),
              "seconds": round(time.monotonic() - started, 1), "per_sequence": rows}
    log("eval", **{k: v for k, v in result.items() if k != "per_sequence"})
    return result


def peak_gib(device):
    import torch

    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if device != "cpu" else None


def optimizer_for(model, cfg, device):
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay,
                             fused=device != "cpu"), params


def grad_norms(model):
    import torch

    norms = {}
    for group, names in lora_groups({n: p for n, p in model.named_parameters() if p.requires_grad}).items():
        params = dict(model.named_parameters())
        grads = [params[n].grad for n in names if params[n].grad is not None]
        norms[group] = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads])))\
            if grads else 0.0
    return norms


def token_ids(model):
    config = model.get_base_model().config
    return {"image": config.image_token_id, "merge": config.vision_config.spatial_merge_size}


# --- kernel checks ---------------------------------------------------------------------------------


def gdn_kernel_check(device, *, seq=512, heads=4, dim=64, log):
    """FLA's chunk_gated_delta_rule against the transformers torch reference: outputs and input
    gradients on a small random problem (in float32 on the reference side)."""
    import inspect

    import torch
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling

    fast = modeling.torch_chunk_gated_delta_rule
    reference = inspect.unwrap(fast)
    try:
        import fla  # noqa: F401
        fla_available = True
    except Exception:  # noqa: BLE001
        fla_available = False
    torch.manual_seed(0)
    q = torch.randn(1, seq, heads, dim, device=device)
    k = torch.randn(1, seq, heads, dim, device=device)
    v = torch.randn(1, seq, heads, dim, device=device)
    g = -torch.rand(1, seq, heads, device=device)
    beta = torch.rand(1, seq, heads, device=device).sigmoid()
    results = {}
    for name, fn, dtype in (("fast", fast, torch.bfloat16), ("reference", reference, torch.float32)):
        inputs = [t.clone().to(dtype).requires_grad_(True) for t in (q, k, v)]
        out, _ = fn(*inputs, g=g.clone(), beta=beta.to(dtype), initial_state=None, output_final_state=False,
                    use_qk_l2norm_in_kernel=True)
        out.float().square().sum().backward()
        results[name] = (out.float().detach(), [t.grad.float() for t in inputs])

    def rel(a, b):
        return float((a - b).norm() / b.norm().clamp_min(1e-12))

    check = {"fla_available": fla_available,
             "out_rel_err": rel(results["fast"][0], results["reference"][0]),
             "grad_rel_err": [rel(a, b) for a, b in zip(results["fast"][1], results["reference"][1])]}
    # on GPU the fast path must be FLA's kernel (the torch reference is >10x slower and needs more memory)
    check["passed"] = (check["out_rel_err"] < 0.02 and max(check["grad_rel_err"]) < 0.05
                       and (fla_available or device == "cpu"))
    log("gdn_kernel_check", **check)
    return check


# --- train / smoke ---------------------------------------------------------------------------------


def run_steps(model, optimizer, params, records, processor, ids, *, cfg, device, log, total_steps, step0=0,
              deadline=None, on_step=None, evaluate_at=(), label="train"):
    """Optimizer steps over ``records`` in order, ``cfg.grad_accum`` sequences per step; the loss is the
    mean over the step's target tokens. Returns the step rows."""
    import torch

    rows, step = [], step0
    groups = [records[i:i + cfg.grad_accum] for i in range(0, len(records), cfg.grad_accum)]
    for group in groups:
        if deadline is not None and time.monotonic() > deadline:
            log("deadline_stop", step=step)
            break
        started = time.monotonic()
        if device != "cpu":
            torch.cuda.reset_peak_memory_stats()
        lr = lr_at(step, total_steps, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        step_targets = sum(r["loss_tokens"] for r in group)
        loss_sum, tokens = 0.0, 0
        with sdpa_context(device):
            for record in group:
                batch = prepare(record, processor, ids["image"], ids["merge"], device)
                loss, _n = sequence_loss(model, batch, cfg.ce_chunk)
                (loss / step_targets).backward()
                loss_sum += float(loss.detach())
                tokens += record["tokens"]
        norms = grad_norms(model)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.clip))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        seconds = time.monotonic() - started
        row = {"step": step, "of": total_steps, "loss": loss_sum / step_targets, "lr": lr, "grad_norm": grad_norm,
               "grad_norm_by_group": norms, "tokens": tokens, "target_tokens": step_targets,
               "seconds": round(seconds, 2), "tokens_per_second": round(tokens / seconds, 1),
               "peak_gib": peak_gib(device), "ids": [r["id"] for r in group], "phase": label}
        if not math.isfinite(row["loss"]) or not math.isfinite(grad_norm):
            log("non_finite", **row)
            raise FloatingPointError(f"non-finite loss or gradient at step {step}")
        rows.append(log("step", **row))
        if on_step:
            on_step(step)
    return rows


def smoke(model_dir, m0_dir, out_dir, *, cfg=None, device="cuda", steps=5, lengths=(32768, 16384), eval_packs=10,
          log=None, commit=lambda: None, max_seconds=None):
    """Load, kernel checks, ``steps`` optimizer steps per length, held-out loss on ``eval_packs`` packs.
    With ``max_seconds``, later phases shrink so the results are written before the function timeout."""
    started = time.monotonic()
    deadline = None if max_seconds is None else started + max_seconds - 150
    cfg = cfg or TrainConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = log or Log(out / "smoke-log.jsonl")
    result = {"environment": environment(), "config": asdict(cfg)}
    log("environment", **result["environment"])
    manifest, train, evaluation = load_m0(m0_dir)
    model, processor, result["load"] = load_model(model_dir, cfg, device=device, log=log)
    ids = token_ids(model)
    if ids["image"] != manifest["image_token_id"]:
        raise RuntimeError("the M0 image token differs from the model's")
    result["gdn_kernel_check"] = gdn_kernel_check(device, log=log)
    optimizer, params = optimizer_for(model, cfg, device)
    longest = sorted(train, key=lambda r: -r["tokens"])
    result["steps"] = {}
    for length in lengths:
        chosen = [r for r in (truncate_record(x, length, ids["image"]) for x in longest) if r][:steps]
        if not chosen:
            result["steps"][str(length)] = {"error": f"no training record has a whole target within {length} tokens"}
            log("smoke_length_skipped", length=length)
            continue
        rows = run_steps(model, optimizer, params, chosen, processor, ids, cfg=TrainConfig(**{**asdict(cfg),
                         "grad_accum": 1}), device=device, log=log, total_steps=100, label=f"smoke-{length}",
                         deadline=deadline)
        if not rows:
            result["steps"][str(length)] = {"error": "deadline before the first step"}
            continue
        timed = rows[1:] or rows  # the first step compiles kernels
        result["steps"][str(length)] = {
            "rows": rows, "tokens_per_second": sum(r["tokens"] for r in timed) / sum(r["seconds"] for r in timed),
            "peak_gib": max(r["peak_gib"] or 0 for r in rows), "max_tokens": max(r["tokens"] for r in rows),
            "linear_attention_grad": all(r["grad_norm_by_group"].get("linear_attention", 0) > 0 for r in rows),
            "loss_finite": all(math.isfinite(r["loss"]) for r in rows)}
        (out / "smoke.json").write_text(json.dumps(result, indent=1, default=str))
        commit()
    if deadline is not None and time.monotonic() > deadline - 60:
        eval_packs = min(eval_packs, 3)
    result["eval"] = evaluate(model, evaluation[:eval_packs], processor, ids, cfg=cfg, device=device, log=log,
                              label="heldout-smoke")
    measured = [s for s in result["steps"].values() if "tokens_per_second" in s]
    rate = measured[0]["tokens_per_second"] if measured else float("nan")
    eval_rate = result["eval"]["loss_tokens"] and (sum(r["tokens"] for r in evaluation[:eval_packs])
                                                   / max(result["eval"]["seconds"], 1e-6))
    train_tokens = sum(r["tokens"] for r in train) * cfg.epochs
    eval_tokens = sum(r["tokens"] for r in evaluation) * (1 + cfg.epochs * cfg.evals_per_epoch) \
        + sum(r["tokens"] for r in train[:cfg.train_probe]) * 2
    result["projection"] = {"train_tokens": train_tokens, "train_seconds": round(train_tokens / rate),
                            "eval_tokens": eval_tokens, "eval_seconds": round(eval_tokens / max(eval_rate, 1e-6)),
                            "load_seconds": result["load"]["seconds"]}
    result["projection"]["total_seconds"] = sum(result["projection"][k] for k in ("train_seconds", "eval_seconds",
                                                                                   "load_seconds")) + 300
    result["passed"] = bool(result["gdn_kernel_check"]["passed"] and len(measured) == len(lengths) and all(
        s["loss_finite"] and s["linear_attention_grad"] for s in measured) and math.isfinite(result["eval"]["nll"]))
    log("smoke_done", passed=result["passed"], projection=result["projection"],
        tokens_per_second={k: round(v["tokens_per_second"]) for k, v in result["steps"].items() if v in measured},
        peak_gib={k: v["peak_gib"] for k, v in result["steps"].items() if v in measured})
    (out / "smoke.json").write_text(json.dumps(result, indent=1, default=str))
    commit()
    return result


def save_adapter(model, folder, extra):
    folder = Path(folder)
    model.save_pretrained(folder)
    files = {p.name: weights_overlay.sha256_file(p) for p in sorted(folder.iterdir()) if p.is_file()}
    (folder / "adapter-info.json").write_text(json.dumps({**extra, "files": files}, indent=1, default=str))
    return files


def train(model_dir, m0_dir, out_dir, *, cfg=None, device="cuda", max_seconds=None, log=None,
          commit=lambda: None, limit_records=None):
    """M0 training with held-out loss before training and at every evaluation point."""
    import torch

    cfg = cfg or TrainConfig()
    started = time.monotonic()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = log or Log(out / "train-log.jsonl")
    torch.manual_seed(cfg.seed)
    result = {"environment": environment(), "config": asdict(cfg)}
    log("environment", **result["environment"])
    manifest, train_records, evaluation = load_m0(m0_dir)
    if limit_records:
        train_records, evaluation = train_records[:limit_records], evaluation[:max(1, limit_records // 2)]
    result["data"] = {"manifest": manifest, "train_sequences": len(train_records),
                      "train_tokens": sum(r["tokens"] for r in train_records),
                      "train_loss_tokens": sum(r["loss_tokens"] for r in train_records),
                      "eval_sequences": len(evaluation)}
    model, processor, result["load"] = load_model(model_dir, cfg, device=device, log=log)
    ids = token_ids(model)
    probe = train_records[:cfg.train_probe]
    result["evals"] = [evaluate(model, evaluation, processor, ids, cfg=cfg, device=device, log=log, label="heldout-before"),
                       evaluate(model, probe, processor, ids, cfg=cfg, device=device, log=log, label="train-probe-before")]
    eval_seconds = result["evals"][0]["seconds"] + result["evals"][1]["seconds"]
    optimizer, params = optimizer_for(model, cfg, device)
    per_epoch, total_steps, points = plan_steps(len(train_records), cfg)
    result["plan"] = {"steps_per_epoch": per_epoch, "total_steps": total_steps, "eval_points": points}
    log("plan", **result["plan"])
    # stop training early enough for the last evaluation and the save
    deadline = None if max_seconds is None else started + max_seconds - 2 * eval_seconds - 300
    state = {"stopped_early": False}
    result["steps"] = []

    def checkpoint_at(step):
        if step not in points:
            return
        evaluation_row = evaluate(model, evaluation, processor, ids, cfg=cfg, device=device, log=log,
                                  label=f"heldout-step{step}")
        result["evals"].append(evaluation_row)
        save_adapter(model, out / f"adapter-step{step}", {"step": step, "heldout_nll": evaluation_row["nll"]})
        (out / "train.json").write_text(json.dumps(result, indent=1, default=str))
        commit()

    step = 0
    for epoch in range(cfg.epochs):
        order = [train_records[i] for i in epoch_order(len(train_records), epoch, cfg.seed)]
        rows = run_steps(model, optimizer, params, order, processor, ids, cfg=cfg, device=device, log=log,
                         total_steps=total_steps, step0=step, deadline=deadline, on_step=checkpoint_at,
                         label=f"epoch{epoch + 1}")
        result["steps"] += rows
        step += len(rows)
        if len(rows) < math.ceil(len(order) / cfg.grad_accum):
            state["stopped_early"] = True
            break
    if step not in points:  # an early stop: evaluate and save where training ended
        result["evals"].append(evaluate(model, evaluation, processor, ids, cfg=cfg, device=device, log=log,
                                        label=f"heldout-step{step}"))
    result["evals"].append(evaluate(model, probe, processor, ids, cfg=cfg, device=device, log=log,
                                    label="train-probe-after"))
    heldout = [e for e in result["evals"] if e["label"].startswith("heldout-")]
    result["summary"] = {
        "steps": step, "stopped_early": state["stopped_early"],
        "heldout_nll_before": heldout[0]["nll"], "heldout_nll_after": heldout[-1]["nll"],
        "heldout_nll_best": min(e["nll"] for e in heldout),
        "best_label": min(heldout, key=lambda e: e["nll"])["label"],
        "train_probe_nll_before": result["evals"][1]["nll"], "train_probe_nll_after": result["evals"][-1]["nll"],
        "tokens_per_second": (sum(r["tokens"] for r in result["steps"]) / sum(r["seconds"] for r in result["steps"])
                              if result["steps"] else None),
        "peak_gib": max((r["peak_gib"] or 0 for r in result["steps"]), default=None),
        "seconds": round(time.monotonic() - started)}
    files = save_adapter(model, out / "adapter", {"step": step, "summary": result["summary"],
                                                  "config": asdict(cfg), "m0_manifest": manifest})
    result["adapter"] = {"folder": str(out / "adapter"), "files": files}
    log("train_done", **result["summary"])
    (out / "train.json").write_text(json.dumps(result, indent=1, default=str))
    commit()
    return result


# --- merge -----------------------------------------------------------------------------------------


def adapter_scaling(config):
    if config.get("use_rslora") or config.get("use_dora"):
        raise ValueError("rsLoRA and DoRA adapters are not supported by the merge")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError("per-module ranks are not supported by the merge")
    return config["lora_alpha"] / config["r"]


def merge(adapter_dir, bf16_dir, nvfp4_dir, out_dir, *, nvfp4_inventory, zero_adapter=False, log=None,
          commit=lambda: None, modules_limit=None):
    """Merge, re-quantize the adapted tensors, run the zero-adapter check and write the overlay."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    started = time.monotonic()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = log or Log(out / "merge-log.jsonl")
    adapter_dir, bf16_dir, nvfp4_dir = Path(adapter_dir), Path(bf16_dir), Path(nvfp4_dir)
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    scaling = adapter_scaling(config)
    state = load_file(adapter_dir / "adapter_model.safetensors")
    pairs = requant.adapter_pairs(state)
    bf16_map = json.loads((bf16_dir / "model.safetensors.index.json").read_text())["weight_map"]
    nv_map = json.loads((nvfp4_dir / "model.safetensors.index.json").read_text())["weight_map"]
    modules = requant.target_modules(bf16_map)
    if modules_limit:
        modules = modules[:modules_limit]
    if sorted(requant.target_modules(nv_map)) != sorted(requant.target_modules(bf16_map)):
        raise RuntimeError("the BF16 and NVFP4 checkpoints adapt different modules")
    if not modules_limit and set(pairs) != set(modules):
        raise RuntimeError(f"adapter covers {len(pairs)} modules, the plan targets {len(modules)}: "
                           f"{sorted(set(pairs) ^ set(modules))[:5]}")
    formats = requant.quant_formats(json.loads((nvfp4_dir / "hf_quant_config.json").read_text()), modules)
    handles = {}

    def tensor(folder, index, name):
        shard = index[name]
        key = (str(folder), shard)
        if key not in handles:
            handles[key] = safe_open(str(folder / shard), framework="pt")
        return handles[key].get_tensor(name)

    by_layer = {}
    for module in modules:
        by_layer.setdefault(int(module.split(".")[3]), []).append(module)
    tensors, zero_rows, change_rows = {}, [], []
    for layer, members in sorted(by_layer.items()):
        base = {m: tensor(bf16_dir, bf16_map, f"{m}.weight") for m in members}
        served = {}
        for m in members:
            for name in requant.served_tensor_names(m, formats[m]):
                served[name] = tensor(nvfp4_dir, nv_map, name)
        # zero-adapter check: the unchanged BF16 tensors re-quantize to the served bytes
        for name, value in requant.requantize(base, formats).items():
            ref = requant.tensor_record(served[name])
            mine = requant.tensor_record(value)
            zero_rows.append({"tensor": name, "exact": ref == mine,
                              "byte_equal_fraction": _byte_fraction(ref[2], mine[2])})
        if zero_adapter:
            merged = base
        else:
            merged = {}
            for m in members:
                a = state[pairs[m]["A"]]
                b = state[pairs[m]["B"]]
                merged[m] = requant.merge(base[m], a, b, scaling)
                delta = (merged[m].float() - base[m].float())
                change_rows.append({"module": m, "delta_rel_norm": float(delta.norm() / base[m].float().norm()),
                                    "bf16_changed_fraction": float((merged[m] != base[m]).float().mean())})
        for name, value in requant.requantize(merged, formats).items():
            record = requant.tensor_record(value)
            tensors[name] = record
            if not zero_adapter and name.endswith(".weight"):
                ref = requant.tensor_record(served[name])
                change_rows[_index(change_rows, name)]["served_code_changed_fraction"] = \
                    1 - _byte_fraction(ref[2], record[2])
        log("merge_layer", layer=layer, modules=len(members),
            zero_exact=sum(r["exact"] for r in zero_rows[-sum(len(requant.served_tensor_names(m, formats[m]))
                                                               for m in members):]))
    zero = {"tensors": len(zero_rows), "exact": sum(r["exact"] for r in zero_rows),
            "min_byte_equal_fraction": min(r["byte_equal_fraction"] for r in zero_rows),
            "not_exact": [r for r in zero_rows if not r["exact"]][:20]}
    log("zero_adapter_check", **{k: v for k, v in zero.items() if k != "not_exact"})
    summary = None
    if change_rows:
        summary = {"modules": len(change_rows),
                   "delta_rel_norm_mean": sum(r["delta_rel_norm"] for r in change_rows) / len(change_rows),
                   "delta_rel_norm_max": max(r["delta_rel_norm"] for r in change_rows),
                   "served_code_changed_fraction_mean": sum(r.get("served_code_changed_fraction", 0)
                                                            for r in change_rows) / len(change_rows)}
        for group in ("self_attn", "linear_attn", "shared_expert"):
            rows = [r for r in change_rows if f".{group}." in r["module"]]
            if rows:
                summary[f"served_code_changed_fraction_{group}"] = sum(
                    r.get("served_code_changed_fraction", 0) for r in rows) / len(rows)
        log("merge_changes", **summary)
    adapter_files = {p.name: weights_overlay.sha256_file(p) for p in sorted(adapter_dir.iterdir()) if p.is_file()}
    extra = {"adapter": {"sha256": adapter_files.get("adapter_model.safetensors"), "files": adapter_files,
                         "config": {k: config.get(k) for k in ("r", "lora_alpha", "lora_dropout", "target_modules")},
                         "scaling": scaling, "zero_adapter": zero_adapter},
             "requant": {"method": "posttrain.requant (bit-exact ModelOpt 0.45 FP8 / W4A16 NVFP4 formulas)",
                         "input_scale": "kept from the served checkpoint", "zero_adapter_check": zero,
                         "changes": summary, "per_module": change_rows},
             "producer": {"bf16": BF16, "seconds": round(time.monotonic() - started)}}
    if (adapter_dir / "adapter-info.json").is_file():
        extra["training"] = json.loads((adapter_dir / "adapter-info.json").read_text()).get("summary")
    overlay_dir = out / "overlay"
    manifest = weights_overlay.write_overlay(overlay_dir, nvfp4_dir, tensors, base_identity=NVFP4,
                                             base_files=nvfp4_inventory["files"], extra=extra,
                                             log=lambda m: log("overlay", message=m))
    result = {"overlay_dir": str(overlay_dir), "overlay_sha256": manifest["overlay"]["sha256"],
              "manifest_sha256": weights_overlay.sha256_file(overlay_dir / weights_overlay.MANIFEST_FILE),
              "tensors": manifest["overlay"]["tensors"], "identical_to_base": manifest["identical_to_base"],
              "zero_adapter_exact": f"{zero['exact']}/{zero['tensors']}", "changes": summary,
              "seconds": round(time.monotonic() - started)}
    log("merge_done", **result)
    (out / "merge.json").write_text(json.dumps(result, indent=1) + "\n")
    commit()
    return result


def _byte_fraction(a, b):
    if len(a) != len(b):
        return 0.0
    if a == b:
        return 1.0
    import numpy as np

    return float((np.frombuffer(a, np.uint8) == np.frombuffer(b, np.uint8)).mean())


def _index(rows, tensor_name):
    module = re.sub(r"\.weight$", "", tensor_name)
    for i in range(len(rows) - 1, -1, -1):
        if rows[i]["module"] == module:
            return i
    raise KeyError(module)
