"""What a tensor-parallel rank holds after the model load, by module group: one ``ARC2_L4_INVENTORY {json}`` line
per worker in the server log.

Loaded as a vLLM general plugin (entry point group ``vllm.general_plugins``; ``arc2.l4_serving`` writes the
``.dist-info`` folder and adds it to the server's PYTHONPATH when the study asks for the inventory). It only reads:
the worker's ``load_model`` is wrapped to print after it returns, and nothing here may raise into the server.

Three views, because routed-expert weights need not be registered parameters:
- ``named``: registered parameters and buffers, grouped by name with indices collapsed;
- ``other``: every other tensor on a card that the garbage collector knows, by dtype and shape;
- ``schemas``: the weight formats of objects that carry a ``b_dtype`` (Humming's layers: uint3 after the 3-bit
  repack, float4e2m1 when it did not apply).
"""

import json
import re
import sys


def collapse(name, keep=6):
    """``language_model.model.layers.17.mlp.experts.w13`` -> ``language_model.model.layers.*.mlp.experts``."""
    parts = [re.sub(r"^\d+$", "*", part) for part in name.split(".")]
    return ".".join(parts[:keep])


def storage_key(tensor):
    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:  # noqa: BLE001 - a tensor without storage (meta, sparse): counted by identity
        return ("id", id(tensor))


def nbytes(tensor):
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:  # noqa: BLE001
        return int(tensor.numel()) * int(tensor.element_size())


def inventory(model, tensors, objects=(), limit=28):
    """``model``: named_parameters / named_buffers; ``tensors``: every live tensor; ``objects``: live objects to
    look at for weight formats. Storages are counted once; sizes are the storages' (a view counts its base)."""
    seen, named, other, by_device = set(), {}, {}, {}

    def count(tensor):
        key = storage_key(tensor)
        if key in seen:
            return 0
        seen.add(key)
        size = nbytes(tensor)
        device = str(tensor.device)
        by_device[device] = by_device.get(device, 0) + size
        return size

    for kind, rows in (("param", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in rows:
            size = count(tensor)
            if size:
                key = f"{collapse(name)} [{kind} {str(tensor.dtype).replace('torch.', '')} {tensor.device.type}]"
                named[key] = named.get(key, 0) + size
    named_total = sum(named.values())
    for tensor in tensors:
        if tensor.device.type != "cuda":
            continue
        size = count(tensor)
        if size:
            key = f"{str(tensor.dtype).replace('torch.', '')} {list(tensor.shape)}"
            entry = other.setdefault(key, [0, 0])
            entry[0] += size
            entry[1] += 1
    schemas = {}
    for item in objects:
        try:
            if not (getattr(type(item), "__module__", None) or "").startswith("humming"):
                continue  # hasattr on arbitrary objects can run lazy imports
            if hasattr(item, "b_dtype") and hasattr(item, "weight_scale_group_size"):
                key = f"{type(item).__name__} b_dtype={item.b_dtype} group={item.weight_scale_group_size}"
                schemas[key] = schemas.get(key, 0) + 1
        except Exception:  # noqa: BLE001 - a property that raises
            pass

    def top(rows, size=lambda value: value):
        return sorted(rows.items(), key=lambda row: -size(row[1]))[:limit]

    gib = 2**30
    return {"named_gib": round(named_total / gib, 3),
            "other_gib": round(sum(v[0] for v in other.values()) / gib, 3),
            "by_device_gib": {k: round(v / gib, 3) for k, v in sorted(by_device.items())},
            "named": [[key, round(value / gib, 4)] for key, value in top(named)],
            "other": [[key, round(value[0] / gib, 4), value[1]] for key, value in top(other, lambda v: v[0])],
            "schemas": dict(sorted(schemas.items(), key=lambda row: -row[1])[:12])}


def report(worker):
    import gc
    import os

    import torch

    model = worker.model_runner.get_model()
    objects = gc.get_objects()
    out = inventory(model, [item for item in objects if isinstance(item, torch.Tensor)], objects)
    out.update(pid=os.getpid(), rank=getattr(worker, "rank", None), model=type(model).__name__,
               expert_int3=os.environ.get("ARC_EXPERT_INT3"))
    if torch.cuda.is_available():
        out["allocated_gib"] = round(torch.cuda.memory_allocated() / 2**30, 3)
        out["reserved_gib"] = round(torch.cuda.memory_reserved() / 2**30, 3)
    return out


def register():
    """The plugin entry point: every vLLM process calls it; only a worker being initialised is touched."""
    try:
        frame, inside = sys._getframe(1), False
        while frame is not None and not inside:
            inside = frame.f_code.co_name == "init_worker"
            frame = frame.f_back
        if not inside:
            return
        from vllm.v1.worker import gpu_worker

        original = gpu_worker.Worker.load_model
        if getattr(original, "arc2_inventory", False):
            return

        def load_model(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            try:
                print("ARC2_L4_INVENTORY " + json.dumps(report(self)), flush=True)
            except Exception as exc:  # noqa: BLE001 - a reading, never the server
                print("ARC2_L4_INVENTORY_ERROR " + json.dumps(repr(exc)[:400]), flush=True)
            return result

        load_model.arc2_inventory = True
        gpu_worker.Worker.load_model = load_model
    except Exception as exc:  # noqa: BLE001
        print("ARC2_L4_INVENTORY_ERROR " + json.dumps(repr(exc)[:400]), flush=True)
