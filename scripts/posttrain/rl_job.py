"""RL arm jobs (8 October), run inside the pinned Axolotl image beside ``flash_qlora_job.py``, whose loader,
cached-prefix step, request encoder and recorder they use (``scripts/posttrain/modal_flash_qlora.py``, modes ``rl*``).

- ``train``: the RL update (docs/rl-plan.md section 1) from one arm's training set (``posttrain.rl_rollouts``):
  per sampled token the clipped ratio of the new policy to the one that generated it, at the actor's temperature
  and inside the actor's sampling set, weighted by the attempt's signed weight; before the first update engine and
  trainer are compared token by token; a fixed probe set measures how far the policy has moved. ``--tiny``: the same
  loop on a tiny random model.
- ``experts`` (one B300): what quantisation-aware adapters on the routed experts cost and what their merge is.
  Axolotl's ``nvfp4_merge_aware`` trains ``Q(dequant(base) + B A)`` with a straight-through gradient and merges to
  the same quantiser. Stages, each recorded as it ends, a failed one does not stop the next:
  load with the expert adapters; span NLL of a few served-NLL rows with the quantised forward off and on (a zero
  adapter must not change the model); gradient steps at 16K and 32K windows on the longest captured request
  (seconds, memory, which experts get gradient); optimizer steps on the expert adapters alone at two learning
  rates (does the loss move through the quantiser); the adapter saved; a step with the adapter factors in BF16;
  Axolotl's merge of a zero and of the trained adapter on the expert shards of a few layers, compared tensor by
  tensor with the served ones (format, which experts changed, seconds per layer on the GPU and on CPU).
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flash_qlora_job as fq  # noqa: E402

EXPERT_MARK = ".experts."
EXPERT_KEY = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(\w+)$")


def expert_group():
    """Expert adapters as their own group in the trainer's reports (``mlp.shared_expert.`` does not match)."""
    if not any(name == "experts" for name, _ in fq.GROUPS):
        fq.GROUPS = (("experts", EXPERT_MARK), *fq.GROUPS)


def per_expert_square(name, grad, experts):
    """Squared gradient norm per expert of one expert-adapter factor, or None for another tensor. PEFT's layout for
    a weight ``[E, dim1, dim2]``: ``lora_A`` is ``[r E, dim2]`` with the expert outermost in the rows, ``lora_B``
    is ``[dim1, r E]`` with the expert innermost in the columns."""
    if "lora_A" in name and grad.dim() == 2 and grad.shape[0] % experts == 0:
        return grad.float().reshape(experts, -1).pow(2).sum(1)
    if "lora_B" in name and grad.dim() == 2 and grad.shape[1] % experts == 0:
        return grad.float().reshape(grad.shape[0], -1, experts).pow(2).sum((0, 1))
    return None


def expert_coverage(model, experts):
    """Which routed experts the last backward reached: ``{stack: [layers, experts] squared gradient norms}`` for the
    decoder (``language_model``) and any other stack that carries expert adapters (``mtp``)."""
    import torch

    stacks = {}
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None or EXPERT_MARK not in name or ".layers." not in name:
            continue
        square = per_expert_square(name, param.grad.detach(), experts)
        if square is None:
            continue
        head, rest = name.split(".layers.", 1)
        layers = stacks.setdefault(head.rsplit(".", 1)[-1], {})
        layer = int(rest.split(".", 1)[0])
        layers[layer] = layers.get(layer, 0) + square.cpu()
    return {stack: torch.stack([layers[i] for i in sorted(layers)]) for stack, layers in stacks.items()}


def coverage_summary(squares):
    """Per layer: experts with a gradient, and how few hold half and 90% of the squared norm."""
    import torch

    reached = (squares > 0).sum(1)
    ordered = squares.sort(dim=1, descending=True).values.cumsum(1)
    total = ordered[:, -1:].clamp_min(1e-30)
    half = (ordered < 0.5 * total).sum(1) + 1
    most = (ordered < 0.9 * total).sum(1) + 1
    mean = lambda values: round(float(values.float().mean()), 1)  # noqa: E731
    return {"layers": int(squares.shape[0]), "experts": int(squares.shape[1]),
            "with_gradient": {"mean": mean(reached), "min": int(reached.min()), "max": int(reached.max())},
            "holding_half_of_the_norm": mean(half), "holding_90_percent": mean(most),
            "with_gradient_per_layer": reached.tolist(),
            "share_of_all": round(float((squares > 0).float().mean()), 4),
            "norm": round(float(torch.sqrt(squares.sum())), 6)}


def overlap(first, second):
    """Experts reached by both of two backward passes over those reached by either."""
    a, b = first > 0, second > 0
    return round(float((a & b).sum()) / max(float((a | b).sum()), 1.0), 4)


def file_sha256(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def raw_bytes(tensor):
    import torch

    return tensor.contiguous().reshape(-1).view(torch.uint8)


def compare_shard(base_path, merged_path):
    """A merged expert shard against the served one, tensor by tensor: same names, types and shapes (a plain NVFP4
    shard the engine reads as it reads the base), and what changed. ``weight`` holds two FP4 codes per byte."""
    import torch
    from safetensors import safe_open

    kinds, changed_experts, wrong = {}, set(), []
    with safe_open(str(base_path), framework="pt") as base, safe_open(str(merged_path), framework="pt") as merged:
        names, theirs = set(base.keys()), set(merged.keys())
        for key in sorted(names & theirs):
            a, b = base.get_tensor(key), merged.get_tensor(key)
            kind = key.rsplit(".", 1)[-1]
            row = kinds.setdefault(kind, {"tensors": 0, "identical": 0, "elements": 0, "changed": 0})
            row["tensors"] += 1
            if a.dtype != b.dtype or a.shape != b.shape:
                wrong.append({"key": key, "base": [str(a.dtype), list(a.shape)], "merged": [str(b.dtype), list(b.shape)]})
                continue
            x, y = raw_bytes(a), raw_bytes(b)
            if kind == "weight" and a.dtype == torch.uint8:
                diff = x ^ y
                count, elements = int(((diff & 15) != 0).sum() + ((diff >> 4) != 0).sum()), 2 * x.numel()
            else:
                width = a.element_size()
                count = int((x.view(-1, width) != y.view(-1, width)).any(1).sum())
                elements = a.numel()
            row["identical"] += count == 0
            row["elements"] += elements
            row["changed"] += count
            match = EXPERT_KEY.search(key)
            if count and match and kind == "weight":
                changed_experts.add(int(match.group(2)))
        same_metadata = base.metadata() == merged.metadata()
    return {"only_in_base": len(names - theirs), "only_in_merged": len(theirs - names), "wrong_type": wrong[:5],
            "wrong_types": len(wrong), "same_metadata": same_metadata, "kinds": kinds,
            "experts_with_changed_codes": sorted(changed_experts)}


def restore_shard(merged_path, base_path, dest):
    """The merged shard written as the served one is (its key order and header metadata), under the served name:
    what an overlay folder holds. Returns bytes and whether the size equals the base file's."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(str(base_path), framework="pt") as base:
        order, metadata = list(base.keys()), base.metadata()
    with safe_open(str(merged_path), framework="pt") as merged:
        tensors = {key: merged.get_tensor(key) for key in order}
    save_file(tensors, str(dest), metadata=metadata)
    size = Path(dest).stat().st_size
    return {"bytes": size, "same_size": size == Path(base_path).stat().st_size,
            "same_file": file_sha256(dest) == file_sha256(base_path)}


def expert_shards(served_dir, layers=None):
    """``{layer: [shard names]}`` of the routed-expert shards (``layer-000LL-experts-XXXX-YYYY.safetensors``)."""
    index = json.loads((Path(served_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    found = {}
    for key, shard in index.items():
        match = EXPERT_KEY.search(key)
        if match and shard.startswith("layer-") and (layers is None or int(match.group(1)) in layers):
            found.setdefault(int(match.group(1)), set()).add(shard)
    return {layer: sorted(shards) for layer, shards in sorted(found.items())}


def merge_view(served_dir, shards, folder):
    """A folder Axolotl's merge reads as a checkpoint: the served folder's small files and the given expert shards,
    linked under names its shard search finds (``model*.safetensors``; the served ones start with ``layer-``)."""
    served_dir, folder = Path(served_dir), Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for path in served_dir.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and not path.name.endswith(".index.json"):
            (folder / path.name).symlink_to(path)
    for shard in shards:
        (folder / f"model-{shard}").symlink_to(served_dir / shard)
    return folder


def merge_experts(rec, label, adapter_dir, served_dir, layers, work, *, device):
    """Axolotl's format-preserving merge of the adapter's expert factors into the expert shards of ``layers``."""
    from axolotl.cli.utils.lora_merge import merge_lora_sharded_efficient

    by_layer = expert_shards(served_dir, layers)
    shards = [shard for names in by_layer.values() for shard in names]
    view, out = merge_view(served_dir, shards, Path(work) / f"{label}-view"), Path(work) / f"{label}-merged"
    if out.exists():
        shutil.rmtree(out)
    began = time.time()
    error = None
    try:
        merge_lora_sharded_efficient(base_model_path=view, lora_adapter_path=Path(adapter_dir), output_path=out,
                                     device=device)
    except Exception as failure:  # a partly written folder is still compared
        error = f"{type(failure).__name__}: {failure}"[:600]
    seconds = time.time() - began
    rows, restored = {}, {}
    for layer, names in by_layer.items():
        for shard in names:
            merged = out / f"model-{shard}"
            if not merged.is_file():
                rows[shard] = {"missing": True}
                continue
            rows[shard] = compare_shard(Path(served_dir) / shard, merged)
            if not restored and not rows[shard]["only_in_base"] and not rows[shard]["wrong_types"]:
                restored = {"shard": shard, **restore_shard(merged, Path(served_dir) / shard, Path(work) / shard)}
                (Path(work) / shard).unlink()
    done = [row for row in rows.values() if not row.get("missing")]
    kinds = {}
    for row in done:
        for kind, counts in row["kinds"].items():
            total = kinds.setdefault(kind, {"tensors": 0, "identical": 0, "elements": 0, "changed": 0})
            for key in total:
                total[key] += counts[key]
    for counts in kinds.values():
        counts["changed_share"] = round(counts["changed"] / max(counts["elements"], 1), 6)
    per_layer = {layer: len({expert for shard in names for expert in rows[shard].get("experts_with_changed_codes", [])})
                 for layer, names in by_layer.items()}
    summary = {"label": label, "device": device, "layers": sorted(by_layer), "shards": len(shards),
               "seconds": round(seconds, 1), "seconds_per_layer": round(seconds / max(len(by_layer), 1), 1),
               "error": error, "written": len(done),
               "bytes": sum((out / f"model-{shard}").stat().st_size for shard in shards
                            if (out / f"model-{shard}").is_file()),
               "only_in_base": sum(row["only_in_base"] for row in done),
               "only_in_merged": sum(row["only_in_merged"] for row in done),
               "wrong_types": sum(row["wrong_types"] for row in done),
               "wrong_type": next((row["wrong_type"] for row in done if row["wrong_type"]), []),
               "same_metadata": all(row["same_metadata"] for row in done), "kinds": kinds,
               "experts_with_changed_codes_per_layer": per_layer, "restored": restored}
    shutil.rmtree(out, ignore_errors=True)
    return rec.event("merge", **summary)


def run_experts(rec, model_dir, out, corpus, data_path, *, deadline, steps, rates, rows_limit, merge_layers, work):
    import torch
    from axolotl.integrations.kernels.libs.sonicmoe import (
        merge_aware_enabled,
        set_merge_aware_enabled,
    )
    from axolotl.integrations.kernels.merge_aware_callback import write_merge_aware_metadata
    from transformers import AutoProcessor

    out, work = Path(out), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    expert_group()
    processor = AutoProcessor.from_pretrained(str(model_dir))
    rows = (json.loads(Path(data_path).read_text())["teacher_forced"] if data_path else [])[:rows_limit]
    sizes = []
    with (Path(corpus) / "train.jsonl").open() as file:
        for number, line in enumerate(file):
            sizes.append((json.loads(line)["prompt_tokens"], number))
    wanted = {"short": min(sizes, key=lambda s: abs(s[0] - 20000))[1], "long": max(sizes)[1]}
    captures = {}
    with (Path(corpus) / "train.jsonl").open() as file:
        for number, line in enumerate(file):
            for label, index in wanted.items():
                if index == number:
                    captures[label] = json.loads(line)
    model, _, loaded = fq.load(rec, model_dir, out, sequence_len=131072)
    result = {"loaded": loaded, "steps": [], "errors": []}
    expert_params = [(name, param) for name, param in model.named_parameters()
                     if param.requires_grad and EXPERT_MARK in name]
    experts = model.config.text_config.num_experts if hasattr(model.config, "text_config") else 512
    kinds = {}
    for name, param in expert_params:
        key = f"{'lora_A' if 'lora_A' in name else 'lora_B'} {tuple(param.shape)} {param.dtype}"
        kinds[key] = kinds.get(key, 0) + 1
    result["expert_adapters"] = rec.event(
        "expert_adapters", tensors=len(expert_params), params=sum(p.numel() for _, p in expert_params),
        kinds=kinds, names=[name for name, _ in expert_params[:4]], experts=experts,
        outside_decoder=sum("language_model" not in name for name, _ in expert_params),
        merge_aware_at_load=merge_aware_enabled(), rank=fq.ADAPTER["rank"],
        gpu_gib=fq.device_gib(peak=False))
    result["table_scale"] = fq.apply_table_scale(rec, model, model_dir)
    model.train()
    rec.save(**result)

    def stage(label, run, needs=0.0):
        if time.time() + needs > deadline:
            result["errors"].append(rec.event("stage_skipped", stage=label, reason="time"))
            return None
        try:
            value = run()
        except Exception as error:  # a stage that fails is recorded; the next one still runs
            result["errors"].append(rec.event("stage_error", stage=label, error=repr(error)[:600],
                                              trace=traceback.format_exc()[-2500:]))
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            value = None
        rec.save(**result)
        return value

    def nll(label, quantised):
        set_merge_aware_enabled(quantised)
        began = time.time()
        try:
            values = fq.span_nll(model, rows)
        finally:
            set_merge_aware_enabled(True)
        result.setdefault("span_nll", {})[label] = values
        rec.event("span_nll", label=label, quantised_forward=quantised, mean=round(sum(values) / len(values), 5),
                  values=values, seconds=round(time.time() - began, 1))

    seen = {}

    def measured(label, name, tail):
        inputs, checks = fq.encode_request(processor, captures[name])
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        total = inputs["input_ids"].shape[1]
        row = {"step": label, "request": name, "tokens": total, "tail": tail, "reply_tokens": checks["reply_tokens"]}
        model.zero_grad(set_to_none=True)
        fq.reset_peaks()
        began = time.time()
        try:
            if tail >= total:
                row["loss"] = round(fq.step(model, inputs)[0], 5)
            else:
                loss, prefix, forward, backward = fq.window_step(model, inputs, window=tail, chunk=16384)
                row.update(loss=round(loss, 5), prefix_seconds=round(prefix, 1), forward_seconds=round(forward, 1),
                           backward_seconds=round(backward, 1))
            row.update(seconds=round(time.time() - began, 1), fits=True, peak_gib=fq.device_gib(),
                       gradients=fq.gradients(model))
            for stack, squares in expert_coverage(model, experts).items():
                row.setdefault("coverage", {})[stack] = coverage_summary(squares)
                if stack == "language_model":
                    seen[label] = squares
        except torch.OutOfMemoryError as error:
            row.update(fits=False, seconds=round(time.time() - began, 1), peak_gib=fq.device_gib(),
                       error=str(error)[:200])
        model.zero_grad(set_to_none=True)
        del inputs
        torch.cuda.empty_cache()
        result["steps"].append(rec.event("step", **row))
        return row

    set_merge_aware_enabled(True)
    if rows:
        stage("span nll, plain forward", lambda: nll("zero_adapter_plain", False))
        stage("span nll, quantised forward", lambda: nll("zero_adapter_quantised", True))
    stage("long 16K first", lambda: measured("long 16K, first step of the process", "long", 16384))
    slow = max([row["seconds"] for row in result["steps"] if row.get("fits")] or [300.0])
    again = stage("long 16K", lambda: measured("long 16K", "long", 16384), needs=1.2 * slow)
    if again and again.get("fits"):
        slow = again["seconds"]  # the first step of a process also compiles kernels
    stage("long 32K", lambda: measured("long 32K", "long", 32768), needs=2.5 * slow)
    stage("short 16K", lambda: measured("short 16K", "short", 16384), needs=1.2 * slow)
    if "long 16K" in seen and "short 16K" in seen:
        result["coverage_overlap"] = rec.event(
            "coverage_overlap", long_16k_and_short=overlap(seen["long 16K"], seen["short 16K"]),
            long_16k_and_32k=overlap(seen["long 16K"], seen["long 32K"]) if "long 32K" in seen else None)

    def save(folder):
        began = time.time()
        model.save_pretrained(str(folder))
        stamped = write_merge_aware_metadata(folder)
        files = {path.name: path.stat().st_size for path in Path(folder).iterdir() if path.is_file()}
        return rec.event("adapter_saved", folder=str(folder), seconds=round(time.time() - began, 1), files=files,
                         stamped=stamped)

    result["zero_adapter"] = stage("save zero adapter", lambda: save(work / "zero-adapter"))

    def optimise():
        """The expert adapters alone (the dense ones are not stepped), so a falling loss is theirs."""
        params = [param for _, param in expert_params]
        optimizer = torch.optim.AdamW(params, lr=rates[0], weight_decay=0.0)
        inputs, _ = fq.encode_request(processor, captures["short"])
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        losses, seconds = [], []
        fq.reset_peaks()
        for rate in rates:  # the loss of a step is measured before its update
            for group in optimizer.param_groups:
                group["lr"] = rate
            for _ in range(steps):
                if time.time() + 1.5 * max(seconds or [slow]) > deadline:
                    break
                began = time.time()
                model.zero_grad(set_to_none=True)
                losses.append(round(fq.reply_loss(model, inputs, tail=16384, backward=True), 5))
                norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                optimizer.step()
                seconds.append(round(time.time() - began, 1))
                rec.event("optimizer_step", lr=rate, loss=losses[-1], grad_norm=round(norm, 5), seconds=seconds[-1])
        model.zero_grad(set_to_none=True)
        after = fq.reply_loss(model, inputs, tail=16384, backward=False) if losses else None
        moved = sum(float(param.detach().float().pow(2).sum()) for name, param in expert_params if "lora_B" in name)
        del optimizer, inputs
        torch.cuda.empty_cache()
        return rec.event("optimised", rates=rates, steps_per_rate=steps, losses=losses,
                         loss_after=None if after is None else round(after, 5), seconds=seconds,
                         peak_gib=fq.device_gib(), lora_b_norm=round(moved ** 0.5, 5))

    result["optimised"] = stage("optimizer steps", optimise, needs=1.5 * slow)
    if rows and result["optimised"]:
        stage("span nll after the steps", lambda: nll("trained_quantised", True), needs=60)
    result["adapter"] = stage("save adapter", lambda: save(work / "adapter"), needs=120)

    def half_precision():
        """The adapter factors in BF16: Axolotl's grouped products need the factors in the activations' type and
        fall back to a loop over the experts otherwise."""
        for _, param in expert_params:
            param.data = param.data.to(torch.bfloat16)
        return measured("long 16K, BF16 expert adapters", "long", 16384)

    if any(param.dtype != torch.bfloat16 for _, param in expert_params):
        stage("long 16K, BF16 expert adapters", half_precision, needs=1.5 * slow)
    if result["zero_adapter"]:
        result["merge_zero"] = stage("merge zero adapter", lambda: merge_experts(
            rec, "zero", work / "zero-adapter", model_dir, merge_layers, work, device="cuda"), needs=240)
    if result["adapter"]:
        result["merge_trained"] = stage("merge trained adapter", lambda: merge_experts(
            rec, "trained", work / "adapter", model_dir, merge_layers, work, device="cuda"), needs=240)
        gpu = (result.get("merge_trained") or {}).get("seconds_per_layer") or 120
        result["merge_cpu"] = stage("merge on CPU", lambda: merge_experts(
            rec, "cpu", work / "adapter", model_dir, merge_layers[:1], work, device="cpu"), needs=max(600, 20 * gpu))
        stage("adapter to the run folder", lambda: shutil.copytree(work / "adapter", out / "adapter",
                                                                    dirs_exist_ok=True), needs=150)
    fits = [row for row in result["steps"] if row.get("fits")]
    passed = bool(fits) and bool(result.get("merge_trained")) and not (result.get("merge_trained") or {}).get("error")
    rec.save(**result, passed=passed, seconds=round(time.time() - fq.START, 1))
    return passed


# --- RL update (docs/rl-plan.md section 1) ----------------------------------------------------------------------

CLIP = (0.8, 1.25)


def rollouts():
    """``posttrain.rl_rollouts``: beside this file in the image, under ``src`` in the repository."""
    try:
        import rl_rollouts
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from posttrain import rl_rollouts
    return rl_rollouts


def actor_logprobs(rows, tokens, support, temperature, at=None):
    """``(log-probability of each sampled token under the actor, valid)``. ``rows`` [n, V]: the logits that predict
    the tokens (or, with ``at`` [n], all logits [T, V] and the rows to read: only the needed entries are copied,
    a window's rows are 8 GB at 16K targets); ``support`` [n, K]: the actor's sampling set per position (-1
    padding), inside which the tempered distribution is renormalised, as the actor's sampler does; a row without a
    set (-1 first) uses the whole vocabulary. A sampled token outside its set is not valid (left out of the update)."""
    import torch

    at = torch.arange(rows.shape[0], device=rows.device) if at is None else at
    out = torch.zeros(at.shape[0], dtype=torch.float32, device=rows.device)
    valid = torch.ones(at.shape[0], dtype=torch.bool, device=rows.device)
    whole = support[:, 0] < 0
    if bool(whole.any()):
        out[whole] = torch.log_softmax(rows[at[whole]].float() / temperature, dim=-1).gather(1, tokens[whole, None])[:, 0]
    if bool((~whole).any()):
        ids = support[~whole]
        scores = (rows[at[~whole][:, None], ids.clamp(min=0)].float() / temperature).masked_fill(ids < 0, float("-inf"))
        hit = ids == tokens[~whole, None]
        out[~whole] = scores.gather(1, hit.float().argmax(dim=1, keepdim=True))[:, 0] - scores.logsumexp(dim=1)
        valid[~whole] = hit.any(dim=1)
    return out, valid


def raw_logprobs(rows, tokens, chunk=1024, at=None):
    """Log-probability of each token over the whole vocabulary at temperature 1 (what the engine reports)."""
    import torch

    at = torch.arange(rows.shape[0], device=rows.device) if at is None else at
    with torch.no_grad():
        return torch.cat([torch.log_softmax(rows[at[i:i + chunk]].float(), dim=-1).gather(1, tokens[i:i + chunk, None])[:, 0]
                          for i in range(0, at.shape[0], chunk)]) if at.shape[0] else rows.new_zeros(0).float()


class RatioLoss:
    """The model's ``loss_function`` for one attempt's window: per sampled token ``ratio = new / reference``
    probability under the actor's policy (temperature, sampling set), objective ``min(ratio w, clip(ratio) w)``
    with the attempt's signed weight ``w``, mean over the window's tokens. A token without a reference counts with
    ratio 1 (the plain weighted log-likelihood gradient). ``measure``: no update, the per-token log-probabilities
    are kept in ``out`` (alignment with the engine, the trainer's own reference, the divergence probe)."""

    def __init__(self, temperature=0.7, clip=CLIP):
        self.temperature, self.clip = float(temperature), clip
        self.arrays, self.weight, self.offset, self.measure, self.out = None, 1.0, 0.0, False, None

    def use(self, arrays, *, weight=1.0, offset=0.0, measure=False):
        """``arrays["weights"]`` (credit per reply): a weight per token, less ``offset`` (the update's baseline shift),
        times ``weight``; ``arrays["reply"]``: which reply each token belongs to."""
        self.arrays, self.weight, self.offset, self.measure, self.out = arrays, float(weight), float(offset), measure, None

    def __call__(self, logits, labels, vocab_size=None, **kwargs):
        import torch

        arrays, labels = self.arrays, labels.to(logits.device)
        where = (labels[0, 1:] != -100).nonzero().squeeze(-1) + 1  # targets inside this pass (the window)
        tokens, rows, at = labels[0, where], logits[0], where - 1
        place = where + (arrays["total"] - labels.shape[-1])       # their positions in the whole sequence
        index = torch.searchsorted(arrays["positions"], place)
        if index.numel() and (int(index.max()) >= arrays["positions"].numel()
                              or not bool((arrays["positions"][index] == place).all())):
            raise RuntimeError("a labelled position has no rollout entry")
        actor, valid = actor_logprobs(rows, tokens, arrays["support"][index], self.temperature, at=at)
        if self.measure:
            self.out = {"index": index.cpu(), "actor": actor.detach().cpu(), "valid": valid.cpu(),
                        "raw": raw_logprobs(rows, tokens, at=at).cpu()}
            return -actor[valid].mean()
        reference = arrays["reference"][index]
        reference = torch.where(torch.isnan(reference), actor.detach(), reference)
        change = (actor - reference)[valid]
        ratio = change.exp()
        low, high = self.clip
        if "weights" in arrays:
            # Credit per reply, tapered off-policy REINFORCE (Le Roux et al. 2025): a reply above the baseline is a
            # plain weighted likelihood term; one below it is weighted by its own probability ratio (the whole
            # reply's, new over generating policy) truncated to [0, 1], so its push fades once it is less likely.
            weight = ((arrays["weights"][index][valid] - self.offset) * self.weight).to(ratio.dtype)
            reply = arrays["reply"][index][valid]
            whole = torch.zeros(int(arrays["reply"].max()) + 1, dtype=torch.float32, device=ratio.device)
            whole.index_add_(0, reply, change.detach().float())
            taper = torch.where(weight > 0, torch.ones_like(weight), whole[reply].exp().clamp(max=1.0).to(weight.dtype))
            objective = taper * weight * actor[valid]
            still = taper > 0.5
        else:
            weight = torch.full_like(ratio, self.weight)
            objective = torch.minimum(ratio * weight, ratio.clamp(low, high) * weight)
            still = torch.where(weight > 0, ratio.detach() < high, ratio.detach() > low)  # tokens the update still moves
        moved, up, down = change.detach(), weight > 0, weight < 0
        self.out = {"tokens": int(valid.sum()), "left_out": int((~valid).sum()),
                    "mean_log_ratio": float(moved.mean()) if change.numel() else 0.0,
                    "moving_share": float(still.float().mean()) if change.numel() else 0.0}
        if "weights" in arrays:  # what the earlier updates did to tokens they push up and to tokens they push down
            self.out.update(up_tokens=int(up.sum()), down_tokens=int(down.sum()),
                            up_log_ratio=float(moved[up].mean()) if bool(up.any()) else None,
                            down_log_ratio=float(moved[down].mean()) if bool(down.any()) else None,
                            up_mass=float(weight[up].sum()), down_mass=float(-weight[down].sum()),
                            taper=float(taper[down].mean()) if bool(down.any()) else None)
        if not change.numel():
            raise RuntimeError("the window holds no usable target")
        return -objective.mean()


def int8_rows(weight, chunk=8192):
    """A weight as the served 8-bit dense kernels hold it (``arc_infra.dense_kernels_runtime.quantize_rows``: per
    output channel, scale = amax / 127, round, clamp to +-127), dequantized into the weight's own type."""
    import torch

    out = torch.empty_like(weight)
    for start in range(0, weight.shape[0], chunk):
        rows = weight[start:start + chunk].float()
        scale = (rows.abs().amax(dim=1) / 127.0).clamp(min=1e-12)
        out[start:start + chunk] = (torch.round(rows / scale[:, None]).clamp_(-127, 127) * scale[:, None]).to(weight.dtype)
    return out


def serve_int8(model, keep=None):
    """Round the trainer's copies of the weights the served build runs through its 8-bit kernels (lm_head, the
    attention and Gated DeltaNet projections) as the engine rounds them at load, so the trainer's forward is the
    served one's. The adapters sit on top: the engine later rounds the merged weight, the trainer the base.
    ``keep``: a dict that receives the unrounded weights (``restore_weights`` puts them back)."""
    import torch

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    head = base.get_output_embeddings()
    tied = head is not None and head.weight is base.get_input_embeddings().weight
    counts, change = {}, 0.0
    for name, module in base.named_modules():
        is_head = module is head
        if "lora_" in name or not isinstance(module, torch.nn.Linear) or not (is_head or fq.INT8_SERVED.search(name)):
            continue
        with torch.no_grad():
            if keep is not None:
                keep[name] = module.weight.data.to("cpu", copy=True)
            rounded = int8_rows(module.weight.data)
            change = max(change, float((rounded.float() - module.weight.data.float()).norm()
                                       / module.weight.data.float().norm().clamp(min=1e-12)))
            if is_head and tied:  # the engine's input embedding stays BF16
                module.weight = torch.nn.Parameter(rounded, requires_grad=False)
            else:
                module.weight.data.copy_(rounded)
        kind = "lm_head" if is_head else fq.group_of(name)
        counts[kind] = counts.get(kind, 0) + 1
    return {"layers": counts, "head_untied": bool(tied), "largest_relative_change": round(change, 5)}


def plan_windows(captures, window):
    """Capture indexes at which a gradient window ends, from the served counts alone (last first): a window holds
    the replies before its last one while they fit and the prompt keeps extending (a drain starts a new one)."""
    ends, index = [], len(captures) - 1
    while index >= 0:
        last = captures[index]
        total = last["prompt_tokens"] + last["usage"]["completion_tokens"]
        ends.append(index)
        first, index = last["prompt_tokens"], index - 1
        while index >= 0:
            earlier = captures[index]
            stop = earlier["prompt_tokens"] + earlier["usage"]["completion_tokens"]
            if total - earlier["prompt_tokens"] + 1 > window or stop > first:
                break
            first, index = earlier["prompt_tokens"], index - 1
    return ends


def window_mass(captures, end, window):
    """The credit (sum of |weight| x tokens) of the replies a window ending at ``end`` holds (``plan_windows`` counts)."""
    return _mass(captures, end, window)


def heaviest_end(captures, ends, window):
    """The window end whose replies carry the most credit (sum of |weight| x tokens), from the served counts: the
    replies before the end while they fit, as ``plan_windows`` counts them."""
    return max(ends, key=lambda end: _mass(captures, end, window))


def _mass(captures, end, window):
    last = captures[end]
    total, first, index, found = last["prompt_tokens"] + last["usage"]["completion_tokens"], last["prompt_tokens"], end, 0.0
    while index >= 0:
        row = captures[index]
        stop = row["prompt_tokens"] + row["usage"]["completion_tokens"]
        if index != end and (total - row["prompt_tokens"] + 1 > window or stop > first):
            break
        found += abs(row["rollout"].get("w") or 0.0) * row["usage"]["completion_tokens"]
        first, index = row["prompt_tokens"], index - 1
    return found


def window_example(encode, captures, end, *, window, margin=8192, floor=16384, credited_only=False):
    """The training example that ends with capture ``end`` (``flash_qlora_job.chain_examples`` for one window, with
    where each reply sits): ``(inputs, tail, [(capture index, start, stop)])`` or None when the reply does not fit.
    An earlier reply is included when it starts inside the window and sits in the later prompt token for token;
    ``credited_only``: not a reply of weight 0 (a restored attempt's scripted history stays in the no-gradient
    prefix, so the window is the credited span plus ``margin``)."""
    done = encode(captures[end])
    if done is None:
        return None
    inputs, start = done
    ids, labels = inputs["input_ids"], inputs["labels"]
    total, first, spans, index = ids.shape[1], start, [(end, start, ids.shape[1])], end - 1
    if total - start + 1 > window:
        return None
    while index >= 0:
        if credited_only and not (captures[index]["rollout"].get("w") or 0.0):
            break
        earlier = encode(captures[index])
        if earlier is None:
            break
        earlier_ids, earlier_start = earlier[0]["input_ids"], earlier[1]
        stop = earlier_ids.shape[1]
        if total - earlier_start + 1 > window or stop > first or not bool((ids[:, :stop] == earlier_ids).all()):
            break
        labels[:, earlier_start:stop] = ids[:, earlier_start:stop]
        spans.append((index, earlier_start, stop))
        first, index = earlier_start, index - 1
    return inputs, min(total, window, max(floor, total - first + margin)), spans[::-1]


def span_arrays(rollout, reply_ids):
    """Per token of one re-encoded reply: ``(reference, support, engine raw log-probability, aligned)``. The engine's
    values are used up to the first token where its sampled ids differ from the reply's (the text of a sampled
    sequence can re-encode differently: 4 of 273 replies in the rollout smoke); ``aligned``: no such token (one
    token of slack at the end)."""
    count, tokens = len(reply_ids), rollout.get("tokens") or []
    shared = next((i for i, (a, b) in enumerate(zip(tokens, reply_ids)) if a != b), min(count, len(tokens)))
    aligned = shared > 0 and shared == min(count, len(tokens)) and abs(count - len(tokens)) <= 1
    nothing = float("nan")
    reference, support, raw = [nothing] * count, [[] for _ in range(count)], [nothing] * count
    if shared:
        raw[:shared] = [float(v) for v in (rollout.get("logprobs") or [])[:shared]] or raw[:shared]
        for i, (ids, value) in enumerate(zip(rollout.get("support") or [], rollout.get("reference") or [])):
            if i < shared:
                support[i] = [int(t) for t in ids]
                reference[i] = nothing if value is None else float(value)
    return reference, support, raw, aligned


def credited_ends(attempt, window):
    """Every planned window end of an attempt whose replies carry weight (whole-segment training: one window per
    end, together covering every weighted reply that fits a window)."""
    captures = rollouts().captures(attempt["replies"])
    captures = [c for c in captures if c["prompt_tokens"] and (c["usage"] or {}).get("completion_tokens")]
    return [end for end in plan_windows(captures, window) if _mass(captures, end, window) > 0] if captures else []


def attempt_example(encode, attempt, *, window, rng, device, think_end=None, end=None, floor=16384, margin=8192,
                    credited_only=False):
    """One window of an attempt (drawn uniformly among its windows) as tensors, with the per-token arrays the loss
    needs. ``windows``: how many the attempt has (its weight is multiplied by it)."""
    import torch

    captures = rollouts().captures(attempt["replies"])
    captures = [c for c in captures if c["prompt_tokens"] and (c["usage"] or {}).get("completion_tokens")]
    ends = plan_windows(captures, window) if captures else []
    credit = any(c["rollout"].get("w") is not None for c in captures)
    # credit per reply: the fullest, heaviest window, ending at any reply (a planned window can be a short remainder)
    if end is None:
        end = heaviest_end(captures, range(len(captures)), window) if credit else (rng.choice(ends) if ends else None)
    made = (window_example(encode, captures, end, window=window, floor=floor, margin=margin, credited_only=credited_only)
            if ends else None)
    if made is None:
        return None
    inputs, tail, spans = made
    ids = inputs["input_ids"][0].tolist()
    positions, reference, support, raw, rows, after_thinking, weights, owner = [], [], [], [], [], [], [], []
    for index, start, stop in spans:
        ref, sup, engine, aligned = span_arrays(captures[index]["rollout"], ids[start:stop])
        positions += range(start, stop)
        reply = ids[start:stop]
        cut = reply.index(think_end) + 1 if think_end in reply else 0  # no thinking block: all of it is the answer
        after_thinking += [i >= cut for i in range(len(reply))]
        reference, support, raw = reference + ref, support + sup, raw + engine
        digest = captures[index]["rollout"].get("prompt_digest")
        weights += [float(captures[index]["rollout"].get("w") or 0.0)] * (stop - start)
        owner += [len(rows)] * (stop - start)
        rows.append({"request": captures[index]["request"], "start": start, "stop": stop, "aligned": aligned,
                     **({"w": captures[index]["rollout"].get("w")} if credit else {}),
                     "prompt_same": None if not digest else digest == rollouts().ids_digest(ids[:start]),
                     "outside": sum(v is None for v in (captures[index]["rollout"].get("reference") or []))})
    width = max(1, max(len(row) for row in support))
    table = torch.full((len(support), width), -1, dtype=torch.long)
    for i, row in enumerate(support):
        table[i, :len(row)] = torch.tensor(row, dtype=torch.long)
    arrays = {"total": len(ids), "positions": torch.tensor(positions, dtype=torch.long, device=device),
              "support": table.to(device), "reference": torch.tensor(reference, dtype=torch.float32, device=device)}
    if credit:
        arrays["weights"] = torch.tensor(weights, dtype=torch.float32, device=device)
        arrays["reply"] = torch.tensor(owner, dtype=torch.long, device=device)
    return {"inputs": inputs, "tail": tail, "arrays": arrays, "engine_raw": raw, "replies": rows, "windows": len(ends),
            "tokens": len(ids), "targets": len(positions), "after_thinking": after_thinking}


def restore_weights(model, kept):
    """Put back the weights ``serve_int8`` rounded."""
    import torch

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    with torch.no_grad():
        for name, module in base.named_modules():
            if name in kept:
                module.weight.data.copy_(kept[name].to(module.weight.device))
    return len(kept)


def baseline_shift(windows, share):
    """The amount to take off every token weight of an update so that the weights above zero hold ``share`` of the
    update's absolute weight (each window counted as the mean over its tokens). Le Roux et al. 2025: the best
    baseline leaves 10-20% effective positives, not half. ``share`` None: the mean (a centred update)."""
    import torch

    windows = [w.float() for w in windows if w.numel()]
    if not windows:
        return 0.0
    if share is None:
        return sum(float(w.mean()) for w in windows) / len(windows)

    def positive(shift):
        up = sum(float((w - shift).clamp(min=0).mean()) for w in windows)
        down = sum(float((shift - w).clamp(min=0).mean()) for w in windows)
        return up / (up + down) if up + down > 0 else 0.0

    low, high = min(float(w.min()) for w in windows), max(float(w.max()) for w in windows)
    for _ in range(50):  # the positive share falls as the shift rises
        middle = (low + high) / 2
        low, high = (middle, high) if positive(middle) > share else (low, middle)
    return (low + high) / 2


def separation(before, after):
    """Did replies that did better than their peers become more likely than replies that did worse, on attempts
    that were never trained? ``before`` / ``after``: per attempt ``{"actor": log-probability per token (nan: not
    scored), "w": the token's reply advantage}``. Mean shift per token of the better replies less that of the worse
    ones; the standard error leaves one attempt out at a time."""
    import torch

    rows = []
    for label, old in before.items():
        new = after.get(label)
        if new is None:
            continue
        keep = torch.isfinite(old["actor"]) & torch.isfinite(new["actor"])
        shift, weight = (new["actor"] - old["actor"])[keep], old["w"][keep]
        rows.append((float(shift[weight > 0].sum()), int((weight > 0).sum()),
                     float(shift[weight < 0].sum()), int((weight < 0).sum())))

    def value(part):
        up, down = sum(r[1] for r in part), sum(r[3] for r in part)
        return (sum(r[0] for r in part) / up, sum(r[2] for r in part) / down) if up and down else None

    whole = value(rows)
    if whole is None:
        return {"attempts": len(rows)}
    left = [value(rows[:i] + rows[i + 1:]) for i in range(len(rows))]
    left = [a - b for a, b in (v for v in left if v)]
    mean = sum(left) / len(left) if left else 0.0
    error = ((len(left) - 1) / len(left) * sum((v - mean) ** 2 for v in left)) ** 0.5 if len(left) > 1 else 0.0
    gap = whole[0] - whole[1]
    return {"attempts": len(rows), "better": round(whole[0], 5), "worse": round(whole[1], 5),
            "separation": round(gap, 5), "se": round(error, 5), "z": round(gap / error, 2) if error else None,
            "better_tokens": sum(r[1] for r in rows), "worse_tokens": sum(r[3] for r in rows)}


class Swapped:
    """The adapter's tensors replaced by a saved state for the length of a ``with`` block (the policy that generated
    this round's rollouts; with every ``lora_B`` at zero, the base), then put back. Plain copies: nothing about the
    adapter's gradient settings changes."""

    def __init__(self, model, state):
        self.params = {name: param for name, param in model.named_parameters() if name in state}
        self.state, self.kept = state, None

    def __enter__(self):
        import torch

        with torch.no_grad():
            self.kept = {name: param.detach().clone() for name, param in self.params.items()}
            for name, param in self.params.items():
                param.copy_(self.state[name].to(param.device, param.dtype))
        return self

    def __exit__(self, *exc):
        import torch

        with torch.no_grad():
            for name, param in self.params.items():
                param.copy_(self.kept[name])
        self.kept = None
        return False


def adapter_state(model, zero=False):
    """The adapter's tensors on the CPU; ``zero``: with every ``lora_B`` at zero (the base policy)."""
    return {name: (param.detach().cpu() * 0 if zero and "lora_B" in name else param.detach().cpu().clone())
            for name, param in model.named_parameters() if "lora_" in name}


class Ahead:
    """Examples built on a worker thread while the GPU runs the ones before them (the GPU otherwise waits for the
    tokenizer and the image processor between passes)."""

    def __init__(self, build, attempts, depth):
        from concurrent.futures import ThreadPoolExecutor

        self.build, self.items, self.depth, self.next, self.queue = build, list(attempts), max(1, depth), 0, []
        self.pool = ThreadPoolExecutor(1)
        self.fill()

    def fill(self):
        while len(self.queue) < self.depth and self.next < len(self.items):
            self.queue.append((self.items[self.next], self.pool.submit(self.build, self.items[self.next])))
            self.next += 1

    def left(self):
        return len(self.queue) + len(self.items) - self.next

    def take(self):
        attempt, future = self.queue.pop(0)
        self.fill()
        return attempt, future.result()

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


class GpuMeter:
    """GPU utilization and memory sampled every few seconds (nvidia-smi): ``read`` gives the means since the last
    read."""

    def __init__(self, seconds=5.0):
        import threading

        self.seconds, self.rows, self.lock, self.stop = seconds, [], threading.Lock(), threading.Event()
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        import subprocess

        while not self.stop.wait(self.seconds):
            try:
                text = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                       "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True,
                                      timeout=20).stdout.strip().splitlines()[0]
                busy, used = (float(part) for part in text.split(","))
            except Exception:  # noqa: BLE001 - a missed sample
                continue
            with self.lock:
                self.rows.append((busy, used))

    def read(self):
        with self.lock:
            rows, self.rows = self.rows, []
        if not rows:
            return {"samples": 0}
        return {"samples": len(rows), "utilization": round(sum(r[0] for r in rows) / len(rows), 1),
                "above_95": round(sum(r[0] >= 95 for r in rows) / len(rows), 3),
                "memory_gib": round(max(r[1] for r in rows) / 1024, 1)}


def serve_merged(model, adapter="default"):
    """Fold the adapter into the base weights as the export does (float32 sum, stored in the weight's own type) and
    zero the adapter, so the forward is the exported overlay's. Per group: the change against the weight and what
    the storage rounding keeps of it."""
    import torch

    stats = {}
    for name, module in model.named_modules():
        if adapter not in (getattr(module, "lora_B", None) or {}):
            continue
        with torch.no_grad():
            weight = module.base_layer.weight
            delta = module.scaling[adapter] * (module.lora_B[adapter].weight.float() @ module.lora_A[adapter].weight.float())
            merged = (weight.data.float() + delta).to(weight.dtype)
            realized = merged.float() - weight.data.float()
            group = stats.setdefault(fq.group_of(name), {"tensors": 0, "delta_sq": 0.0, "weight_sq": 0.0,
                                                         "realized_sq": 0.0, "dot": 0.0})
            group["tensors"] += 1
            group["delta_sq"] += float((delta * delta).sum())
            group["weight_sq"] += float((weight.data.float() ** 2).sum())
            group["realized_sq"] += float((realized * realized).sum())
            group["dot"] += float((realized * delta).sum())
            weight.data.copy_(merged)
            module.lora_B[adapter].weight.zero_()
    return {name: {"tensors": g["tensors"], "delta_over_weight": round((g["delta_sq"] / g["weight_sq"]) ** 0.5, 6),
                   "realized_cosine": round(g["dot"] / max((g["realized_sq"] * g["delta_sq"]) ** 0.5, 1e-30), 4)}
            for name, g in stats.items()}


def measured(model, loss, example):
    """The example scored without an update: per labelled token of the window its position index, the trainer's
    actor and raw log-probabilities."""
    loss.use(example["arrays"], measure=True)
    fq.reply_loss(model, example["inputs"], tail=example["tail"], backward=False)
    out, loss.out = loss.out, None
    if out is None:  # rl-check-v1: Axolotl's cut-cross-entropy patch computes its own loss inside forward
        raise RuntimeError("the model's forward did not call the RL loss (a patched forward computes its own)")
    return out


def alignment(example, out):
    """Engine against trainer on one window, per sampled token: raw log-probability (whole vocabulary, temperature
    1) and the actor's (temperature and sampling set). Differences are engine minus trainer."""
    import torch

    index = out["index"]
    engine_raw = torch.tensor(example["engine_raw"], dtype=torch.float32)[index]
    engine_actor = example["arrays"]["reference"].cpu()[index]

    def stats(engine, trainer, keep):
        keep = keep & torch.isfinite(engine)
        diff = (engine - trainer)[keep]
        if not diff.numel():
            return {"tokens": 0}
        return {"tokens": int(diff.numel()), "mean": round(float(diff.mean()), 5),
                "mean_abs": round(float(diff.abs().mean()), 5),
                "beyond_clip": round(float((diff.abs() > 0.18).float().mean()), 5),
                "largest": round(float(diff.abs().max()), 4), "engine_mean": round(float(engine[keep].mean()), 5)}

    everything = torch.ones_like(out["valid"])
    answer = torch.tensor(example.get("after_thinking") or [False] * len(example["engine_raw"]))[index]
    return {"raw": stats(engine_raw, out["raw"], everything),
            "raw_thinking": stats(engine_raw, out["raw"], ~answer), "raw_answer": stats(engine_raw, out["raw"], answer),
            "actor": stats(engine_actor, out["actor"], out["valid"]),
            "tokens": {"index": index.tolist(), "engine_raw": [round(v, 4) for v in engine_raw.tolist()],
                       "trainer_raw": [round(v, 4) for v in out["raw"].tolist()],
                       "engine_actor": [round(v, 4) for v in engine_actor.tolist()],
                       "trainer_actor": [round(v, 4) for v in out["actor"].tolist()]}}


def divergence(before, after):
    """Mean of log(reference / new) over the probe tokens: positive when the policy has moved away from the one
    that generated them."""
    total = count = 0.0
    for old, new in zip(before, after):
        keep = old["valid"] & new["valid"]
        total += float((old["actor"] - new["actor"])[keep].sum())
        count += int(keep.sum())
    return total / count if count else 0.0


def run_train(rec, model_dir, out, data_path, *, served_dir, window, lr, updates, per_update, reference, probe_every,
              limit, deadline, align, recheck, seed, save_every=5, resume=None, int8=False, score_only=False,
              stages=(), warmup=0, positive_share=0.15, held_every=0, stop_z=3.0, new_round=False):
    """Updates from one arm's training set (``rl_rollouts.training_set``): ``per_update`` backward passes per
    optimizer update. Before the first update the probe windows are scored (the divergence reference) and, with
    ``align`` or the trainer's reference, every training window too: engine and trainer are compared token by token
    and the trainer's values can serve as the reference (``reference``: engine | trainer | none). Stops when the
    probe divergence passes ``limit``. ``recheck``: trained windows scored again at the end. State (adapter,
    optimizer, counters, the probe's base scores) is saved every ``save_every`` updates; a restarted run continues
    from it, and ``resume`` names another run's folder to continue from. ``int8``: the base as the served 8-bit
    kernels hold it (``serve_int8``). ``score_only``: every window scored against the engine, no update.
    ``stages`` (with ``score_only`` and ``resume``): that run's adapter scored on this file once per stage, each on
    top of the one before: base (no adapter), base8 (no adapter, the 8-bit kernels' rounding of the base), adapter
    (as trained; on the rounded base after base8), merged (folded into the stored weights, as the export), int8
    (the 8-bit kernels' rounding of the result). A training set with ``credit: reply`` carries a weight per reply:
    every update shifts them by one baseline (``positive_share``) and the loss is the tapered form of ``RatioLoss``."""
    import math
    import random

    import torch
    from transformers import AutoProcessor

    out = Path(out)
    data = json.loads(Path(data_path).read_text())
    processor = AutoProcessor.from_pretrained(str(served_dir or model_dir))
    fq.CUT_CROSS_ENTROPY["on"] = False  # its patched forward never calls loss_function; this loss needs the logits
    model, _, loaded = fq.load(rec, model_dir, out, sequence_len=131072)
    if fq.NVFP4["on"]:
        loaded = {**loaded, "table_scale": fq.apply_table_scale(rec, model, model_dir)}
    if int8 and not stages:
        loaded = {**loaded, "int8": rec.event("int8", **serve_int8(model))}
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    sampling = next((r["sampling"] for a in data["attempts"] for r in a["replies"] if r.get("sampling")), {})
    loss = RatioLoss(temperature=sampling.get("temperature") or 0.7)
    base.loss_function = loss
    encoded, _ = fq.example_maker(processor, [], tail=window, phases=[window])
    train = [a for a in data["attempts"] if a["role"] == "train"]
    if score_only:
        updates, align = 0, True
    else:
        train = train[: updates * per_update]
    probes = [a for a in data["attempts"] if a["role"] == "probe"]
    held = [a for a in data["attempts"] if a["role"] == "held"] if held_every else []
    result = {"loaded": loaded, "plan": {"arm": data.get("arm"), "passes": len(train), "updates": updates,
                                         "per_update": per_update, "window": window, "lr": lr, "reference": reference,
                                         "probes": len(probes), "reported": data.get("reported"), "clip": list(CLIP),
                                         "temperature": loss.temperature, "adapter": dict(fq.ADAPTER)},
              "alignment": [], "passes": [], "probe": [], "updates": [], "errors": [], "stopped": None, "restarts": 0}
    rec.save(**result)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    think_end = processor.tokenizer.convert_tokens_to_ids("</think>")
    own, before = {}, {}  # the trainer's own scores before the first update: the reference, and the recheck's

    def example(attempt):
        """Built when needed (a round holds hundreds of windows); the same window of an attempt every time."""
        made = attempt_example(lambda capture: encoded(capture, pair=True), attempt, window=window,
                               rng=random.Random(f"{seed}:{attempt['label']}"), device="cuda:0", think_end=think_end)
        if made is None:
            if not any(row.get("label") == attempt["label"] for row in result["errors"]):
                result["errors"].append({"label": attempt["label"], "error": "no window fits"})
        elif reference == "none":
            made["arrays"]["reference"] = torch.full_like(made["arrays"]["reference"], float("nan"))
        elif attempt["label"] in own:
            made["arrays"]["reference"] = own[attempt["label"]].to(made["arrays"]["reference"].device)
        return made

    def score(attempt, made, role, stage=None):
        began = time.time()
        got = measured(model, loss, made)
        row = alignment(made, got)
        tokens = row.pop("tokens")
        tag = {"stage": stage} if stage else {}
        first = 0
        for reply in made["replies"]:  # per reply: its scored tokens and the sum of their log-probabilities
            inside = (got["index"] >= first) & (got["index"] < first + reply["stop"] - reply["start"]) & got["valid"]
            reply["scored"], reply["actor"] = int(inside.sum()), round(float(got["actor"][inside].sum()), 4)
            first += reply["stop"] - reply["start"]
        with (out / "alignment-tokens.jsonl").open("a") as file:
            file.write(json.dumps({"label": attempt["label"], "role": role, **tag, "replies": made["replies"],
                                   **tokens}) + "\n")
        result["alignment"].append({"label": attempt["label"], "role": role, **tag, "context": made["tokens"],
                                    "tail": made["tail"], "replies": made["replies"], **row,
                                    "seconds": round(time.time() - began, 1)})
        rec.event("alignment", label=attempt["label"], **tag, raw=row["raw"], actor=row["actor"])
        return got

    def probe(label):
        scores = [measured(model, loss, made) for _, made in probe_set]
        row = {"at": label, "divergence": round(divergence(probe_base, scores), 6),
               "tokens": sum(int(s["valid"].sum()) for s in scores)}
        if probe_zero:  # a later round: the drift from the base as well as from the round's starting adapter
            row["from_base"] = round(divergence(probe_zero, scores), 6)
        result["probe"].append(row)
        rec.event("probe", **row)
        return row

    def held_scores():
        """Per never-trained attempt: the log-probability of every token of its window, and the token's advantage."""
        found, ahead = {}, Ahead(example, held, 2)
        while ahead.left():
            attempt, made = ahead.take()
            if made is None or "weights" not in made["arrays"]:
                continue
            got = measured(model, loss, made)
            actor = torch.full((made["arrays"]["positions"].numel(),), float("nan"))
            actor[got["index"][got["valid"]]] = got["actor"][got["valid"]]
            found[attempt["label"]] = {"actor": actor, "w": made["arrays"]["weights"].cpu()}
            del made
        ahead.close()
        return found

    def checkpoint():
        folder = out / "ckpt" / f"update-{updates_done:04d}"
        model.save_pretrained(str(folder))
        torch.save({"optimizer": optimizer.state_dict(), "updates_done": updates_done, "passes": passes,
                    "result": result, "probe_base": probe_base, "own": own, "held_base": held_base,
                    "start": start, "probe_zero": probe_zero}, str(folder / "trainer.pt"))
        (out / "ckpt" / "latest.json").write_text(json.dumps({"label": folder.name, "updates": updates_done}))
        for old in sorted(path for path in (out / "ckpt").iterdir() if path.is_dir())[:-2]:
            shutil.rmtree(old)

    updates_done = passes = saved_at = 0
    if score_only and stages:  # another run's adapter on this file, as trained and as the served build holds it
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        folder = Path(resume) / "adapter"
        if not folder.is_dir():  # a run that has not finished: its last saved state
            folder = Path(resume) / "ckpt" / json.loads((Path(resume) / "ckpt" / "latest.json").read_text())["label"]
        kept, adapted = {}, False  # the unrounded weights of a base8 stage; whether the adapter is in yet
        result["stages"] = []
        meter = GpuMeter()
        for stage in stages:  # base and base8 come before the adapter is loaded
            if stage not in ("base", "base8") and not adapted:
                set_peft_model_state_dict(model, load_file(str(folder / "adapter_model.safetensors")))
                rec.event("adapter_loaded", folder=str(folder))
                adapted = True
            if stage == "merged" and kept:  # an adapter trained on the rounded base: the export merges the stored one
                restore_weights(model, kept)
                kept = {}
            change = {"base": lambda: {}, "base8": lambda: serve_int8(model, kept), "adapter": lambda: {},
                      "merged": lambda: serve_merged(model), "int8": lambda: serve_int8(model)}[stage]()
            result["stages"].append(rec.event("stage", stage=stage, **{"change": change} if change else {}))
            ahead = Ahead(example, probes + train, 2)
            while ahead.left():
                if time.time() > deadline:
                    result["stopped"] = "time"
                    break
                attempt, made = ahead.take()
                if made is not None:
                    score(attempt, made, attempt["role"], stage)
                del made
            ahead.close()
            torch.cuda.empty_cache()
            result["stages"][-1]["gpu"] = meter.read()
            rec.save(**result)
        done = len(result["alignment"]) == len(stages) * len(probes + train)
        rec.save(**result, updates_done=0, passed=done, seconds=round(time.time() - fq.START, 1))
        return done
    probe_set = [(a, made) for a, made in ((a, example(a)) for a in probes) if made is not None]
    source = Path(resume) if resume and not (out / "ckpt" / "latest.json").is_file() else out  # own state first
    start, probe_zero = None, None
    look = 0.0  # seconds of one probe-and-held look
    if new_round and source != out:
        # A new round on rollouts of the adapter another run trained: that adapter and its optimizer state, new
        # counters, and everything scored afresh by the policy that generated this round's data.
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        folder = source / "ckpt" / json.loads((source / "ckpt" / "latest.json").read_text())["label"]
        saved = torch.load(str(folder / "trainer.pt"), map_location="cpu", weights_only=False)
        final = source / "adapter" if (source / "adapter" / "adapter_model.safetensors").is_file() else folder
        set_peft_model_state_dict(model, load_file(str(final / "adapter_model.safetensors")))
        optimizer.load_state_dict(saved["optimizer"])  # a finished run's last saved state is its final one
        rec.event("round", adapter=str(final), earlier_updates=saved["updates_done"])
        with Swapped(model, adapter_state(model, zero=True)):
            probe_zero = [measured(model, loss, made) for _, made in probe_set]
        source = out / "none"
    if (source / "ckpt" / "latest.json").is_file():  # a restarted run, or one continued under another id
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        folder = source / "ckpt" / json.loads((source / "ckpt" / "latest.json").read_text())["label"]
        saved = torch.load(str(folder / "trainer.pt"), map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, load_file(str(folder / "adapter_model.safetensors")))
        optimizer.load_state_dict(saved["optimizer"])
        updates_done, passes, probe_base, own = (saved[k] for k in ("updates_done", "passes", "probe_base", "own"))
        held_base = saved.get("held_base") or {}
        start, probe_zero = saved.get("start"), saved.get("probe_zero")
        saved_at = updates_done
        result = {**saved["result"], "plan": result["plan"], "loaded": loaded, "stopped": None,
                  "restarts": saved["result"].get("restarts", 0) + 1}
        rec.event("resumed", checkpoint=str(folder), updates_done=updates_done, passes=passes)
    else:
        # Everything scored before the first update is scored by the policy that generated it.
        began = time.time()
        probe_base = [score(attempt, made, "probe") for attempt, made in probe_set]
        held_base = held_scores() if held else {}
        look = time.time() - began
        if align or reference == "trainer":
            ahead = Ahead(example, train, 2)
            for index in range(len(train)):
                attempt, made = ahead.take()
                if made is None:
                    continue
                got = score(attempt, made, "train")
                if index < recheck:
                    before[attempt["label"]] = got
                if reference == "trainer":
                    mine = torch.full(made["arrays"]["reference"].shape, float("nan"))
                    mine[got["index"]] = got["actor"]
                    own[attempt["label"]] = mine
                del made
            ahead.close()
            torch.cuda.empty_cache()
    if start is None:  # the adapter this run starts from: the policy that generated its rollouts
        start = adapter_state(model)
    rec.save(**result)
    credit = data.get("credit") == "reply"
    meter = GpuMeter()
    longest = 0.0
    ahead = Ahead(example, train[passes:], per_update)  # the next update's examples, while this one's passes run
    while updates_done < updates and passes < len(train) and not result["stopped"]:
        # the clock: room for one more update, its look if it has one, and the closing look (the last update's look
        # is the closing one)
        coming = updates_done + 1
        looks = 1 + (coming != updates and bool((probe_set and coming % probe_every == 0) or (
            held_base and held_every and coming % held_every == 0)))
        if time.time() + 1.15 * longest + 1.15 * looks * look > deadline:
            result["stopped"] = "time"
            break
        cycle = time.time()
        for param in params:
            param.grad = None
        count = 0
        batch = [ahead.take() for _ in range(min(per_update, ahead.left()))]
        passes += len(batch)
        batch = [(attempt, made) for attempt, made in batch if made is not None]
        # credit per reply: one baseline shift per update, leaving ``positive_share`` of its weight above zero
        centre = baseline_shift([made["arrays"]["weights"] for _, made in batch], positive_share) if credit else 0.0
        for group in optimizer.param_groups:  # linear warm-up, then the constant rate
            group["lr"] = lr * min(1.0, (updates_done + 1) / max(1, warmup))
        if reference == "start" and batch:  # the generating policy's values: the windows under the run's first adapter
            with Swapped(model, start):
                for _, made in batch:
                    got = measured(model, loss, made)
                    mine = torch.full(made["arrays"]["reference"].shape, float("nan"))
                    mine[got["index"]] = got["actor"]
                    made["arrays"]["reference"] = mine.to(made["arrays"]["reference"].device)
        for attempt, made in batch:
            torch.cuda.reset_peak_memory_stats()
            began, tail, value = time.time(), made["tail"], None
            for trial in range(2):  # out of memory: once more with a shorter gradient window
                torch.cuda.reset_peak_memory_stats()  # the peak of the pass that ran, not of a failed one
                try:
                    loss.use(made["arrays"], weight=1.0 if credit else attempt["weight"] * made["windows"], offset=centre)
                    value = fq.reply_loss(model, made["inputs"], tail=tail, backward=True)
                    if loss.out is None:
                        raise RuntimeError("the model's forward did not call the RL loss")
                    break
                except torch.OutOfMemoryError:
                    result["errors"].append({"label": attempt["label"], "error": "out of memory", "tail": tail,
                                             "retried": trial == 0})
                    torch.cuda.empty_cache()
                    tail = max(16384, int(tail * 0.85))
            if value is None:
                continue
            row = {"label": attempt["label"], "weight": attempt["weight"], "windows": made["windows"],
                   "context": made["tokens"], "tail": tail, "targets": made["targets"],
                   "loss": round(value, 5), **loss.out, "seconds": round(time.time() - began, 1),
                   "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1), "after_updates": updates_done,
                   **({"centre": round(centre, 5)} if credit else {})}
            if not math.isfinite(value):
                result["stopped"] = f"non-finite loss on {attempt['label']}"
                break
            result["passes"].append(row)
            rec.event("pass", **row)
            count += 1
        del batch
        if reference == "start" and updates_done >= 2 and count and all(
                row["mean_log_ratio"] == 0.0 for row in result["passes"][-count:]):
            result["stopped"] = "the reference equals the adapted model: the starting adapter was not swapped in"
        if result["stopped"] or not count:
            continue
        for param in params:
            if param.grad is not None:
                param.grad /= count
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        if not math.isfinite(norm):
            result["stopped"] = "non-finite gradient"
            break
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates_done += 1
        result["updates"].append({"update": updates_done, "passes": count, "grad_norm": round(norm, 4),
                                  "lr": optimizer.param_groups[0]["lr"], "gpu": meter.read()})
        if updates_done == 1 or updates_done % 10 == 0:
            rec.event("sizing", update=updates_done, **result["updates"][-1]["gpu"],
                      peak_gib=max(row["peak_gib"] for row in result["passes"][-count:]),
                      total_gib=round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1))
        longest, cycle = max(longest, time.time() - cycle), time.time()
        if probe_set and (updates_done % probe_every == 0 or updates_done == updates):
            row = probe(f"update {updates_done}")
            if row["divergence"] > limit:
                result["stopped"] = f"probe divergence {row['divergence']:.4f} nat per token above {limit}"
        if held_base and (updates_done % held_every == 0 or updates_done == updates or result["stopped"]):
            row = {"at": updates_done, **separation(held_base, held_scores())}
            result.setdefault("separation", []).append(row)
            rec.event("separation", **row)
            if row.get("z") is not None and row["z"] >= stop_z and not result["stopped"]:
                result["stopped"] = f"separation {row['separation']:.4f} nat per token, {row['z']} standard errors"
        look = max(look, time.time() - cycle)
        if updates_done % save_every == 0 or updates_done == updates or result["stopped"]:
            checkpoint()
            saved_at = updates_done
        rec.save(**result, updates_done=updates_done)
    if held_base and updates_done and (result.get("separation") or [{}])[-1].get("at") != updates_done:
        result.setdefault("separation", []).append({"at": updates_done, **separation(held_base, held_scores())})
        rec.event("separation", **result["separation"][-1])
    if updates_done != saved_at:  # stopped by the clock between two saves: the next run continues from here
        checkpoint()
    moved = []
    for attempt in train[:recheck] if updates_done else []:  # did the updates move the trained tokens their way
        made = example(attempt)
        if made is None:
            continue
        got, old = measured(model, loss, made), before.get(attempt["label"])
        if old is None:  # against the reference the update used
            keep = got["valid"] & torch.isfinite(made["arrays"]["reference"].cpu()[got["index"]])
            change = (got["actor"] - made["arrays"]["reference"].cpu()[got["index"]])[keep]
        else:
            keep = got["valid"] & old["valid"]
            change = (got["actor"] - old["actor"])[keep]
        moved.append({"label": attempt["label"], "weight": attempt["weight"], "tokens": int(keep.sum()),
                      "against": "reference" if old is None else "trainer before",
                      "mean_log_ratio": round(float(change.mean()), 5) if change.numel() else None})
    result["recheck"] = moved
    if updates_done:
        model.save_pretrained(str(out / "adapter"))
    seconds = [row["seconds"] for row in result["passes"]]
    hard = [row for row in result["errors"] if not row.get("retried") and row.get("error") != "no window fits"]
    planned = result["stopped"] in (None, "time") or str(result["stopped"]).startswith(("separation", "probe divergence"))
    passed = (updates_done > 0 or (score_only and bool(result["alignment"]))) and planned and not hard
    rec.save(**result, updates_done=updates_done, passes_done=len(result["passes"]),
             seconds_per_pass=round(sum(seconds) / len(seconds), 1) if seconds else None,
             peak_gib=max((row["peak_gib"] for row in result["passes"]), default=None),
             passed=passed, seconds=round(time.time() - fq.START, 1))
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["experts", "train"])
    parser.add_argument("--model", required=True, help="the served NVFP4 checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--corpus", help="experts: folder with train.jsonl (captured served requests)")
    parser.add_argument("--data", help="experts: served-NLL prompt rows; train: rl_rollouts.training_set as JSON")
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--rslora", action="store_true")
    parser.add_argument("--steps", type=int, default=3, help="optimizer steps on the expert adapters per rate")
    parser.add_argument("--rates", default="1e-4,1e-3", help="learning rates of those steps, in order")
    parser.add_argument("--rows", type=int, default=4, help="served-NLL rows scored")
    parser.add_argument("--merge-layers", default="0,24", help="layers whose expert shards are merged")
    parser.add_argument("--work", default="/tmp/rl-work", help="local scratch (adapters, merged shards)")
    parser.add_argument("--served", help="train: the served checkpoint's folder (processor)")
    parser.add_argument("--tiny", action="store_true", help="train: a tiny random model (a test of the code)")
    parser.add_argument("--window", type=int, default=32768, help="train: gradient window, tokens")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--per-update", type=int, default=4, help="train: backward passes per optimizer update")
    parser.add_argument("--reference", default="engine", choices=["engine", "trainer", "start", "none"],
                        help="train: the ratio's reference. trainer: every training window scored before the first "
                             "update; start: each update's windows scored just before its passes under the adapter "
                             "the run started from (the same values, no scoring phase up front)")
    parser.add_argument("--round", action="store_true",
                        help="train, with --continue-from: a new round on fresh rollouts of that run's adapter (its "
                             "adapter and optimizer, new counters, probes and held attempts scored afresh)")
    parser.add_argument("--warmup", type=int, default=0, help="train: updates over which the rate rises linearly")
    parser.add_argument("--held-every", type=int, default=0,
                        help="train: updates between scorings of the never-trained attempts (role held): better "
                             "replies against worse ones")
    parser.add_argument("--stop-z", type=float, default=3.0,
                        help="train: stop when that separation is this many standard errors above zero")
    parser.add_argument("--positive-share", type=float, default=0.15,
                        help="train, credit per reply: share of an update's weight left above the baseline (0: the mean)")
    parser.add_argument("--probe-every", type=int, default=10, help="train: updates between divergence probes")
    parser.add_argument("--limit", type=float, default=0.02, help="train: probe divergence that stops the arm, nat")
    parser.add_argument("--align", action="store_true", help="train: score every training window before the updates")
    parser.add_argument("--recheck", type=int, default=0, help="train: trained windows scored again at the end")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5, help="train: updates between saved states")
    parser.add_argument("--continue-from", help="train: another run's output folder whose saved state is continued")
    parser.add_argument("--int8", action="store_true", help="train: the base as the served 8-bit kernels hold it")
    parser.add_argument("--score-only", action="store_true", help="train: score every window, no update")
    parser.add_argument("--stages", default="", help="train, with --score-only and --continue-from: that run's "
                                                     "adapter scored per stage, e.g. adapter,merged,int8")
    args = parser.parse_args()
    fq.set_adapter(args.rank, args.rslora)
    rec = fq.Recorder(args.out)
    rec.save(environment=fq.environment(rec))
    if args.mode == "train":
        fq.NVFP4.update(on=not args.tiny, experts=False)
        fq.BF16_LINEAR["on"] = args.tiny
        passed = run_train(rec, fq.tiny_model(args.model, args.out) if args.tiny else args.model, args.out, args.data,
                           served_dir=args.served, window=args.window, lr=args.lr, updates=args.updates,
                           per_update=args.per_update, reference=args.reference, probe_every=args.probe_every,
                           limit=args.limit, deadline=fq.START + args.seconds - 300, align=args.align,
                           recheck=args.recheck, seed=args.seed, save_every=args.save_every,
                           resume=args.continue_from, int8=args.int8, score_only=args.score_only,
                           stages=[s for s in args.stages.split(",") if s], warmup=args.warmup,
                           positive_share=args.positive_share if args.positive_share > 0 else None,
                           held_every=args.held_every, stop_z=args.stop_z, new_round=args.round)
    else:
        fq.NVFP4.update(on=True, experts=True, backend="sonicmoe")
        passed = run_experts(rec, args.model, args.out, args.corpus, args.data, deadline=fq.START + args.seconds - 60,
                             steps=args.steps, rates=[float(v) for v in args.rates.split(",")], rows_limit=args.rows,
                             merge_layers=[int(v) for v in args.merge_layers.split(",") if v], work=args.work)
    rec.event("done", passed=passed)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
