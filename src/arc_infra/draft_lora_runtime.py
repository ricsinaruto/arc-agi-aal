"""The served MTP draft's LoRA (RL stage D), installed as vllm/arc_draft_lora.py by draft_lora_patch (ARC_DRAFT_LORA=1).

vLLM never LoRA-wraps the draft, so the trainer's draft factors (rl1_draft.export: draft_lora.safetensors +
draft_lora.json beside adapter_model.safetensors) are merged into the draft's BF16 dense weights in place:
- read(lora_path) at _load_adapter: the factors on CPU (None without draft_lora.json);
- activate(lora_model) when vLLM activates an adapter: restore the base rows, then add alpha / r x B A (exact on a
  reload under a new id: always base + delta, never delta on delta);
- deactivate(lora_id) / deactivate_all(): the base back.
In-place copy_ into the same tensors: CUDA graphs and torch.compile keep their addresses. Base copies live on the host
(only the touched params, taken at the first merge). One served adapter (max_loras 1), TP 1."""

import hashlib
import json
from pathlib import Path

import torch

FORMAT = "arc-draft-lora-v1"
SHARD_ORDER = {"qkv_proj": ("q", "k", "v"), "gate_up_proj": (0, 1)}
CHUNK = 2048  # rows per merge slice (bounds the fp32 transient on the card)
_STATE = {"model": None, "base": {}, "active": None, "warned": False}


def say(tag, **row):
    print(tag + " " + json.dumps(row, sort_keys=True, default=str), flush=True)


def key(name, part):
    return f"base_model.model.mtp.{name}.lora_{part}.weight"


def register_draft(model):
    """The speculator's draft model, once loaded (V2 runner: Speculator.load_model)."""
    _STATE.update(model=model, base={}, active=None)
    say("ARC_DRAFT_LORA_DRAFT", params=sum(1 for _ in model.named_parameters()))


def read(lora_path):
    folder = Path(lora_path)
    if not (folder / "draft_lora.json").is_file():
        return None
    meta = json.loads((folder / "draft_lora.json").read_text())
    if meta.get("format") != FORMAT:
        raise ValueError(f"draft LoRA {folder}: format {meta.get('format')!r} != {FORMAT}")
    from safetensors.torch import load_file

    tensors = load_file(str(folder / "draft_lora.safetensors"), device="cpu")
    want = {key(name, part) for name in meta["modules"] for part in "AB"}
    if set(tensors) != want:
        raise ValueError(f"draft LoRA {folder}: {len(tensors)} tensors, {len(want)} expected")
    return {"meta": meta, "tensors": tensors}


def plan(weights, tensors, meta):
    """[(module, weight key, first row)] with every bound checked before anything is written."""
    sizes = {(s["vllm"], s["shard"]): tensors[key(n, "B")].shape[0] for n, s in meta["modules"].items()}
    out = []
    for name, spec in meta["modules"].items():
        param, shard = spec["vllm"], spec["shard"]
        w = weights[param + ".weight"]
        a, b = tensors[key(name, "A")], tensors[key(name, "B")]
        start = 0
        if shard is not None:
            order = SHARD_ORDER[param.rsplit(".", 1)[-1]]
            start = sum(sizes[(param, s)] for s in order[:order.index(shard)])
        if start + b.shape[0] > w.shape[0] or a.shape[1] != w.shape[1] or a.shape[0] != b.shape[1]:
            raise ValueError(f"draft LoRA {name}: B {tuple(b.shape)} A {tuple(a.shape)} at row {start} of {param} "
                             f"{tuple(w.shape)}")
        out.append((name, param + ".weight", start))
    return out


def merge(weights, tensors, meta, sign=1.0):
    """Add (``sign`` -1: remove) sign x alpha / r x B A of every module into ``weights`` (vLLM draft parameter name ->
    its weight, e.g. ``model.layers.0.self_attn.qkv_proj.weight``), each at its shard's rows (TP 1: q | k | v and
    gate | up, sizes from the factors). In place, in row slices; returns the parameter names touched."""
    scale = float(meta["lora_alpha"]) / float(meta["r"])
    touched = []
    for name, wkey, start in plan(weights, tensors, meta):
        w = weights[wkey]
        a = tensors[key(name, "A")].to(w.device, torch.float32)
        b = tensors[key(name, "B")].to(w.device, torch.float32)
        for lo in range(0, b.shape[0], CHUNK):
            hi = min(lo + CHUNK, b.shape[0])
            rows = w[start + lo:start + hi]
            rows.copy_((rows.float() + sign * scale * (b[lo:hi] @ a)).to(w.dtype))
        touched.append(wkey[:-len(".weight")])
    return sorted(set(touched))


def _weights(names):
    """Weight key -> the live draft parameter's tensor (a wrapper's prefix is allowed: suffix match, unique)."""
    params = dict(_STATE["model"].named_parameters())
    out = {}
    for wkey in names:
        hits = [p for k, p in params.items() if k == wkey or k.endswith("." + wkey)]
        if len(hits) != 1:
            raise ValueError(f"draft LoRA: {len(hits)} draft parameters match {wkey}")
        if not hits[0].dtype.is_floating_point or hits[0].element_size() < 2:
            raise ValueError(f"draft LoRA: {wkey} is {hits[0].dtype}, not a dense float weight")
        out[wkey] = hits[0].data
    return out


def _restore():
    if not _STATE["base"]:
        return
    live = _weights(sorted(_STATE["base"]))
    for wkey, saved in _STATE["base"].items():
        live[wkey].copy_(saved)


def digest():
    """sha256 (16 hex) over the touched draft weights as served (GPU check: same adapter -> same digest)."""
    h = hashlib.sha256()
    for wkey, w in sorted(_weights(sorted(_STATE["base"])).items()):
        h.update(wkey.encode())
        h.update(w.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def activate(lora_model):
    draft = getattr(lora_model, "_arc_draft", None)
    if _STATE["model"] is None:
        if draft is not None and not _STATE["warned"]:
            _STATE["warned"] = True
            say("ARC_DRAFT_LORA_NO_DRAFT", id=lora_model.id)  # no speculator registered: the factors go unused
        return
    with torch.no_grad():
        _restore()
        _STATE["active"] = None
        if draft is None:
            return
        weights = _weights(sorted({s["vllm"] + ".weight" for s in draft["meta"]["modules"].values()}))
        plan(weights, draft["tensors"], draft["meta"])
        for wkey, w in weights.items():
            if wkey not in _STATE["base"]:
                _STATE["base"][wkey] = w.detach().to("cpu", copy=True)
        touched = merge(weights, draft["tensors"], draft["meta"])
        _STATE["active"] = lora_model.id
    say("ARC_DRAFT_LORA_MERGED", id=lora_model.id, modules=len(draft["meta"]["modules"]), params=touched,
        digest=digest())


def deactivate(lora_id):
    if _STATE["active"] is not None and _STATE["active"] == lora_id:
        with torch.no_grad():
            _restore()
        _STATE["active"] = None
        say("ARC_DRAFT_LORA_RESTORED", id=lora_id)


def deactivate_all():
    if _STATE["active"] is not None:
        deactivate(_STATE["active"])
