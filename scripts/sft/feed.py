"""The SFT trainer's data feed: the files of the next steps read on a loader thread (``prefetch`` steps ahead, the
same batches and cursor as reading them inline), and each example staged on a side stream one example ahead of its
pass: pinned, moved to the device, its M-RoPE positions and image features computed (``vision_ahead``: the frozen
vision tower's run, as ``positions_and_images`` does it inline), and its ``plan`` (train.loss_plan: host-side index
work and PLE rows). The pass waits on the staging event. The pass reads nothing back, so the card may still be using
an earlier example's memory when the host lets it go: each staging first waits (on its stream) for the work queued
before it was submitted, so a freed block of the side stream is reused only after its last reader ran."""
import contextlib
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

import modelio


def prepare(ex):
    """A build.py example (1-D int32 rows) as model inputs: [1, T] long ids, labels and token types."""
    out = dict(ex)
    for k in ("input_ids", "labels", "mm_token_type_ids"):
        if torch.is_tensor(out.get(k)):
            out[k] = (out[k].unsqueeze(0) if out[k].dim() == 1 else out[k]).long()
    return out


RETRY = {"tries": 4, "sleep": 5.0}  # 28 Oct restart: a volume reload made an existing file vanish for a moment


def _load(path):
    for i in range(RETRY["tries"]):
        try:
            return torch.load(path, weights_only=False)
        except FileNotFoundError:
            if i == RETRY["tries"] - 1:
                raise
            time.sleep(RETRY["sleep"] * (i + 1))


def load_example(path, teacher=None):
    """A build.py example; ``teacher``: its teacher.py file (the KL loss's top-k targets) as ``ex["teacher"]``. A file
    not found is retried (transient: a Modal volume reload swaps the mount's view)."""
    ex = prepare(_load(path))
    if teacher is not None:
        ex["teacher"] = _load(teacher)
    return ex


def move(v, device, cuda):
    if isinstance(v, dict):
        return {k: move(x, device, cuda) for k, x in v.items()}
    if not torch.is_tensor(v):
        return v
    return v.pin_memory().to(device, non_blocking=True) if cuda else v.to(device)


def stage(model, ex, device, vision=True, stream=None, plan=None, after=None):
    """An example on ``device``: {"ex": inputs there, "piece": positions_and_images (None unless ``vision``),
    "plan": ``plan(host example)`` (its tensors moved on the same stream), "event": staged on ``stream``}. ``after``:
    an event of the consuming stream the staging waits for first."""
    ex = prepare(ex)
    cuda = torch.device(device).type == "cuda"
    with torch.no_grad(), (torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()):
        if after is not None and stream is not None:
            stream.wait_event(after)
        made_plan = plan(ex) if plan is not None else None
        moved = {k: move(v, device, cuda) for k, v in ex.items()}
        piece = modelio.positions_and_images(model, moved, host_index=True) if vision else None
        event = None
        if stream is not None:
            event = torch.cuda.Event()
            event.record(stream)
    return {"ex": moved, "piece": piece, "plan": made_plan, "event": event}


def ready(made):
    if made.get("event") is not None:
        torch.cuda.current_stream().wait_event(made["event"])
    return made


class Stager:
    """Stages examples one ahead on a worker thread (and side stream on CUDA)."""

    def __init__(self, model, device, vision, plan=None):
        self.model, self.device, self.vision, self.plan = model, device, vision, plan
        self.stream = torch.cuda.Stream(device) if torch.device(device).type == "cuda" else None
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="sft-stage")

    def submit(self, ex):
        after = None
        if self.stream is not None:  # everything queued so far (the examples whose memory may be let go) first
            after = torch.cuda.Event()
            after.record(torch.cuda.current_stream(self.device))
        return self.pool.submit(stage, self.model, ex, self.device, self.vision, self.stream, self.plan, after)

    def prime(self, ex):
        """Start staging ``ex`` now (the next step's first example, during the optimizer step)."""
        self.primed = (ex, self.submit(ex))

    def run(self, examples):
        """Staged examples in order, the next one staging while the current one runs."""
        examples = list(examples)
        primed, pending = getattr(self, "primed", None), None
        if examples and primed is not None and primed[0] is examples[0]:
            pending, self.primed = primed[1], None
        elif examples:
            pending = self.submit(examples[0])
        for i in range(len(examples)):
            made = pending.result()
            pending = self.submit(examples[i + 1]) if i + 1 < len(examples) else None
            yield ready(made)

    def close(self):
        self.pool.shutdown(wait=True)


class Loader(threading.Thread):
    """Batches ``next_batch`` gives from ``cursor`` on, loaded ``depth`` steps ahead: (names, reply, cursor after,
    host examples, reply tokens counted from the labels). ``pick``: the names this process loads (a sharded run's
    rank share; names, examples and reply tokens are then this rank's)."""

    def __init__(self, next_batch, root, cursor, depth, teacher=False, pick=None):
        super().__init__(daemon=True, name="sft-loader")
        self.next_batch, self.root, self.cursor, self.teacher = next_batch, Path(root), cursor, teacher
        self.pick = pick
        self.q, self.stop = queue.Queue(maxsize=max(1, depth)), threading.Event()
        self.calm = threading.Event()  # cleared while the trainer wants the host to itself (the paged optimizer step)
        self.calm.set()

    def run(self):
        try:
            while not self.stop.is_set():
                names, reply, self.cursor = self.next_batch(self.cursor)
                if self.pick is not None:
                    names = self.pick(names)
                exs = []
                for f in names:
                    self.calm.wait()
                    exs.append(load_example(self.root / f, self.root / "teacher" / f if self.teacher else None))
                n_step = sum(int((e["labels"][0, 1:] != -100).sum()) for e in exs)
                while not self.stop.is_set():
                    try:
                        self.q.put((names, reply, self.cursor, exs, n_step), timeout=1.0)
                        break
                    except queue.Full:
                        continue
        except BaseException as error:  # noqa: BLE001 - raised in the trainer by get()
            self.q.put(error)

    def get(self):
        got = self.q.get()
        if isinstance(got, BaseException):
            raise RuntimeError("data loader failed") from got
        return got

    def close(self):
        self.stop.set()
