"""The vision tower's weights off the cards without ``--language-model-only`` (11 October).

The ARC-AGI-2 loop sends no image. ``--language-model-only`` also strips the multi-axis rotary positions and takes
other paths in the model, and both starts with it hung in the workers' warm-up. This keeps the model as loaded:
``arc2.l4_serving`` adds ``--skip-mm-profiling`` (the profile run then holds no encoder output: about 0.45 GiB a
rank that sizes the cache) and loads this plugin, which after the worker's load replaces ``model.visual`` (0.22 GiB
a rank) by a module that refuses to run, and lowers the runner's weight memory by what was freed. The recipe's
readiness check with an image is answered locally (``arc2.l4_serving``). One ``ARC2_TOWER {json}`` line a worker.
"""

import json
import os
import sys

ENV = "ARC2_L4_TOWER"


def drop(torch, model, name="visual"):
    """Replace ``model.<name>`` by a refusing module; the report (bytes on a card before, what was freed)."""
    import gc

    tower = getattr(model, name, None)
    if not isinstance(tower, torch.nn.Module) or not any(True for _ in tower.parameters()):
        return {"dropped": False, "reason": f"no {name} with weights ({type(tower).__name__})"}
    tensors = [*tower.parameters(), *tower.buffers()]
    held = sum(t.numel() * t.element_size() for t in tensors if t.is_cuda)
    card = next((t.device for t in tensors if t.is_cuda), None)
    before = torch.cuda.memory_allocated(card) if card is not None else 0

    class Dropped(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError("the vision tower was dropped on these cards (arc2.l4_tower): text only")

    kind = type(tower).__name__
    setattr(model, name, Dropped())
    del tower, tensors
    gc.collect()
    if card is not None:
        torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated(card) if card is not None else 0
    return {"dropped": True, "module": kind, "held_gib": round(held / 2**30, 3), "freed_bytes": before - after}


def register():
    """The plugin entry point: only a worker being initialised is touched, and only with ``ARC2_L4_TOWER`` set."""
    try:
        if os.environ.get(ENV) != "off":
            return
        frame, inside = sys._getframe(1), False
        while frame is not None and not inside:
            inside = frame.f_code.co_name == "init_worker"
            frame = frame.f_back
        if not inside:
            return
        from vllm.v1.worker import gpu_worker

        original = gpu_worker.Worker.load_model
        if getattr(original, "arc2_tower", False):
            return
    except Exception as exc:  # noqa: BLE001
        print("ARC2_TOWER_ERROR " + json.dumps(repr(exc)[:400]), flush=True)
        return

    def load_model(self, *args, **kwargs):
        import torch

        result = original(self, *args, **kwargs)
        try:
            report = drop(torch, self.model_runner.get_model())
            runner = self.model_runner
            if report.get("freed_bytes", 0) > 0 and hasattr(runner, "model_memory_usage"):
                runner.model_memory_usage -= report["freed_bytes"]  # what sizes the cache
                report["weights_gib"] = round(runner.model_memory_usage / 2**30, 3)
            report.update(pid=os.getpid(), rank=getattr(self, "rank", None))
            print("ARC2_TOWER " + json.dumps(report), flush=True)
        except Exception as exc:  # noqa: BLE001 - the tower stays: the start goes on as before
            print("ARC2_TOWER_ERROR " + json.dumps(repr(exc)[:400]), flush=True)
        return result

    load_model.arc2_tower = True
    gpu_worker.Worker.load_model = load_model
