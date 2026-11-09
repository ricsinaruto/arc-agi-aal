"""RL redesign stage C2 (26 Oct): ``rl1_trainer`` on N GPUs, one process per GPU (torchrun), behind ``RL1_DIST=1``.
Off (the default), nothing here runs and the trainer is the single-GPU one.

- Command channel: rank 0 serves the cmd dir as before; each cmd is broadcast on a gloo control group, every rank
  that runs the op runs it, rank 0 gathers the per-rank results, merges them and answers. While idle, rank 0
  broadcasts a heartbeat every ``HEARTBEAT_S`` (the followers' wait stays inside the control group's timeout).
  Ops on every rank: ``ALL_RANKS``; the rest (save, checkpoint, scores, references) on rank 0 only (the adapter
  and optimizer state are identical on every rank; ``dist_digest`` checks that at a step).
- accumulate / update: the call's windows (after the trainer's own selection and order) are planned here: windows
  whose estimated gradient tail exceeds ``fit_tokens`` (cmd / trainer_fast knob, default 65,536) run sequence-parallel
  over the whole world (``rl1_cp``; every rank, the same order, first), the rest data-parallel by greedy
  longest-first assignment on the estimated tokens. A rank with no windows still joins every collective.
- step: the LoRA gradients and the token / loss / entropy sums are all-reduced (sum) before the divisor, clip and
  optimizer step, which then run identically on every rank.
- Failures: a rank's window failure is agreed on (control group) before the next collective, so every rank raises
  and rank 0 answers with the failing rank's error; a dead follower breaks rank 0's next control collective, which
  rank 0 answers as an error before it exits. A failure inside a CP window's collectives waits for the compute
  group's timeout (``RL1_DIST_TIMEOUT_S``)."""

import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

ALL_RANKS = {"reset", "accumulate", "step", "update", "park", "unpark", "load_checkpoint", "route", "exit"}
HEARTBEAT_S = 30.0
FIT_TOKENS = 65536


class DistError(RuntimeError):
    pass


class Dist:
    def __init__(self, defer_device=False):
        """``defer_device`` (RL1_PRELOAD_HOST): no ``set_device`` here (it creates the card's context, which waits for
        the GPU gate: H3 sat ~450 s on it before the host load began); flash_qlora_job.load sets the device right after
        the gate, before any tensor or collective touches the card (NCCL builds its communicator lazily)."""
        import torch
        import torch.distributed as dist

        self.local = int(os.environ.get("LOCAL_RANK", "0"))
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            if backend == "nccl" and not defer_device:
                torch.cuda.set_device(self.local)
            dist.init_process_group(backend, timeout=timedelta(seconds=float(os.environ.get("RL1_DIST_TIMEOUT_S")
                                                                             or 1800)))
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.backend = dist.get_backend()
        self.device = f"cuda:{self.local}" if self.backend == "nccl" else "cpu"
        self.collectives = 0  # compute-group collectives issued (the first pays the NCCL communicator setup)
        self.warming, self.warmed = None, None  # RL1_NCCL_WARM: the warm-up thread, then its row
        self.ctl = dist.new_group(backend="gloo", timeout=timedelta(
            seconds=float(os.environ.get("RL1_DIST_CTL_TIMEOUT_S") or 7200)))

    # ---- control group (objects) ----
    def bcast(self, obj):
        import torch.distributed as dist

        box = [obj]
        dist.broadcast_object_list(box, src=0, group=self.ctl)
        return box[0]

    def gather(self, obj):
        import torch.distributed as dist

        out = [None] * self.world if self.rank == 0 else None
        dist.gather_object(obj, out, dst=0, group=self.ctl)
        return out

    def all_objects(self, obj):
        import torch.distributed as dist

        out = [None] * self.world
        dist.all_gather_object(out, obj, group=self.ctl)
        return out

    def agree(self, error=None, where=""):
        """Every rank's error (or None); raises the same DistError on every rank if any rank failed."""
        errors = self.all_objects(None if error is None else repr(error)[:1500])
        bad = [(r, e) for r, e in enumerate(errors) if e]
        if bad:
            raise DistError(f"{where}: " + "; ".join(f"rank {r}: {e}" for r, e in bad))

    # ---- compute group (tensors) ----
    def warm(self, device):
        """RL1_NCCL_WARM=1: the compute group's first collective (a 1-element all-reduce: the NCCL communicator setup)
        on a thread, beside the first accumulate's windows (D1 26 Oct: the step, the run's first NCCL collective, sat
        22.6 s at ~0% util before its optimizer). Every compute-group collective joins it first (``ready``), so the
        warm-up is the first collective on every rank. Returns whether it started."""
        import threading

        if os.environ.get("RL1_NCCL_WARM") != "1" or self.collectives or self.warming is not None:
            return False
        import torch
        import torch.distributed as dist

        state = {"began": time.time()}

        def run():
            try:
                if self.backend == "nccl":
                    torch.cuda.set_device(device)
                t = torch.ones(1, device=device)
                dist.all_reduce(t)
                state["sum"] = float(t.item())
            except BaseException as error:  # noqa: BLE001 - raised by ready()
                state["error"] = repr(error)[:500]
            state["seconds"] = round(time.time() - state["began"], 3)

        self.warming = (threading.Thread(target=run, name="nccl-warm", daemon=True), state)
        self.collectives += 1
        self.warming[0].start()
        return True

    def ready(self):
        """The warm-up collective done (no-op without one); its row in ``warmed``."""
        if self.warming is None:
            return self.warmed
        thread, state = self.warming
        waited = time.time()
        thread.join()
        self.warming = None
        self.warmed = {"seconds": state.get("seconds"), "join_wait_s": round(time.time() - waited, 3),
                       "error": state.get("error")}
        if state.get("error") or state.get("sum") != float(self.world):
            raise DistError(f"NCCL warm-up: {state.get('error') or state.get('sum')}")
        return self.warmed

    def reduce_grads(self, params, bucket=1 << 27):
        """Sum every parameter's gradient over the ranks (a gradient missing on some ranks counts as zero; one
        missing on every rank stays None). Buckets of ``bucket`` elements per dtype."""
        import torch
        import torch.distributed as dist

        if not params:
            return 0
        self.ready()
        device = params[0].device
        have = torch.tensor([p.grad is not None for p in params], dtype=torch.float32, device=device)
        dist.all_reduce(have, op=dist.ReduceOp.MAX)
        self.collectives += 1
        todo = []
        for p, h in zip(params, have.tolist()):
            if h:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                todo.append(p.grad)
        moved = 0
        by_dtype = {}
        for g in todo:
            by_dtype.setdefault(g.dtype, []).append(g)
        for grads in by_dtype.values():
            group, size = [], 0
            for g in grads + [None]:
                if g is not None and (size + g.numel() <= bucket or not group):
                    group.append(g)
                    size += g.numel()
                    continue
                if group:
                    flat = torch.cat([x.reshape(-1) for x in group])
                    dist.all_reduce(flat)
                    start = 0
                    for x in group:
                        x.copy_(flat[start:start + x.numel()].view_as(x))
                        start += x.numel()
                    moved += flat.numel() * flat.element_size()
                group, size = ([g], g.numel()) if g is not None else ([], 0)
        return moved

    def reduce_sums(self, values, device):
        """[float] summed over the ranks (float64)."""
        import torch
        import torch.distributed as dist

        self.ready()
        t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=device)
        dist.all_reduce(t)
        self.collectives += 1
        return t.tolist()

    def digest(self, params):
        """Cross-rank check that the adapter is identical: per parameter float64 sum and sum of squares, compared
        bitwise across ranks. ``{"equal": bool, "differ": [names]}``."""
        import torch

        with torch.no_grad():
            mine = torch.stack([torch.stack([p.detach().double().sum(), p.detach().double().square().sum()])
                                for _, p in params]).cpu()
        every = self.all_objects(mine)
        differ = sorted({params[i][0] for other in every[1:] for i in
                         torch.nonzero((other != every[0]).any(dim=1)).flatten().tolist()})
        return {"equal": not differ, "differ": differ[:10], "params": len(params)}


def init(defer_device=False):
    """A ``Dist`` when ``RL1_DIST=1`` (torchrun environment), else None."""
    if os.environ.get("RL1_DIST") != "1":
        return None
    return Dist(defer_device=defer_device)


# ------------------------------------------------------------------------------------------------ window plan
def estimate(spec, default_window):
    """(tokens, tail) a spec's window will have, from the served counts (``rl_job.window_example``'s walk on prompt /
    completion tokens) without encoding it; a spec may carry ``est_tokens`` / ``est_tail`` (tests)."""
    if "est_tokens" in spec:
        return int(spec["est_tokens"]), int(spec.get("est_tail", spec["est_tokens"]))
    import random

    import rl1_encode
    import rl_job

    rows = rl1_encode.read_rows(spec, cached=True)
    caps = rl_job.rollouts().captures(rows, **({"gameplay": False} if spec.get("whole_segment") else {}))
    caps = [c for c in caps if c["prompt_tokens"] and (c["usage"] or {}).get("completion_tokens")]
    window = int(spec.get("window") or default_window)
    if not caps:
        return 0, 0
    end = spec.get("end")
    if end is None:
        credit = any(c["rollout"].get("w") is not None for c in caps)
        ends = rl_job.plan_windows(caps, window)
        end = (rl_job.heaviest_end(caps, range(len(caps)), window) if credit
               else random.Random(0).choice(ends) if ends else None)
        if end is None:
            return 0, 0
    last = caps[end]
    total = last["prompt_tokens"] + last["usage"]["completion_tokens"]
    first, index = last["prompt_tokens"], end - 1
    while index >= 0:
        row = caps[index]
        if spec.get("whole_segment"):
            if row["rollout"].get("w") is None:
                break
        elif spec.get("credited_only") and not (row["rollout"].get("w") or 0.0):
            break
        stop = row["prompt_tokens"] + row["usage"]["completion_tokens"]
        if total - row["prompt_tokens"] + 1 > window or stop > first:
            break
        first, index = row["prompt_tokens"], index - 1
    floor, margin = int(spec.get("floor") or 16384), int(spec.get("margin") or 8192)
    if spec.get("whole_segment"):
        floor, margin = int(spec.get("floor") or 0), int(spec.get("margin") or 0)
    return total, min(total, window, max(floor, total - first + margin))


def plan(specs, world, rank, fit_tokens, default_window, estimates=None):
    """``{"cp": [i], "dp": {rank: [i]}, "mine": [i], "est": [(tokens, tail)], "load": [tokens per rank]}``: windows
    with tail > ``fit_tokens`` sequence-parallel (all ranks, first, in the call's order), the rest by greedy
    longest-first on estimated tokens (ties: the earlier window, the lower rank); each rank's share in call order."""
    est = estimates or [estimate(s, default_window) for s in specs]
    cp = [i for i, (_, tail) in enumerate(est) if tail > fit_tokens]
    load, dp = [0] * world, {r: [] for r in range(world)}
    for i in sorted((i for i in range(len(specs)) if i not in set(cp)), key=lambda i: (-est[i][0], i)):
        r = min(range(world), key=lambda r: (load[r], r))
        dp[r].append(i)
        load[r] += est[i][0]
    for r in dp:
        dp[r].sort()
    return {"cp": cp, "dp": dp, "mine": cp + dp[rank], "est": est, "load": load}


# ------------------------------------------------------------------------------------------------ serve
def runs_here(op, rank):
    return rank == 0 or op in ALL_RANKS


SUM_KEYS = ("trained_tokens", "acc_tokens", "accumulated_windows")
MAX_KEYS = ("train_seconds", "torch_peak_gib", "card_peak_gib", "seconds")


def merge(cmd, results):
    """Rank 0's answer from every rank's result."""
    merged = dict(results[0])
    ran = [(r, x) for r, x in enumerate(results) if not x.get("skipped")]
    bad = {r: x.get("error") or x.get("stop") or "not ok" for r, x in ran if not x.get("ok", True)}
    if bad:
        merged["ok"] = False
        merged["error"] = "; ".join(f"rank {r}: {e}" for r, e in bad.items())[:4000]
        merged["rank_errors"] = bad
        first = min(bad)
        if results[first].get("traceback"):
            merged["traceback"] = results[first]["traceback"]
    if len(ran) > 1 and not merged.get("dist_reduced"):
        for key in ("windows", "errors"):
            if all(isinstance(x.get(key), list) for _, x in ran):
                merged[key] = [w for _, x in ran for w in x[key]]
        for key in SUM_KEYS:
            if all(isinstance(x.get(key), (int, float)) and not isinstance(x.get(key), bool) for _, x in ran):
                merged[key] = sum(x[key] for _, x in ran)
    for key in MAX_KEYS:
        vals = [x.get(key) for _, x in ran if isinstance(x.get(key), (int, float))]
        if vals:
            merged[key] = max(vals)
    merged["ranks"] = [{"rank": r, "ok": x.get("ok", True), "skipped": bool(x.get("skipped")),
                        **{k: x.get(k) for k in ("seconds", "train_seconds", "trained_tokens", "torch_peak_gib",
                                                 "memory", "dist_plan", "step_parts", "nccl_warm")
                           if k in x},
                        **({"windows": len(x["windows"])} if isinstance(x.get("windows"), list) else {})}
                       for r, x in enumerate(results)]
    return merged


def serve(trainer):
    """``Trainer.serve`` for RL1_DIST: rank 0 reads the cmd dir and answers; every rank runs its share."""
    d = trainer.dist
    index, alive, last = 0, trainer.dir / "alive.json", time.time()
    if d.rank == 0:
        alive.write_text(json.dumps({"state": "ready", "loaded": trainer.loaded, "time": time.time(),
                                     "world": d.world}, default=str))
    while True:
        cmd = None
        if d.rank == 0:
            path = trainer.dir / "cmd" / f"{index:04d}.json"
            if path.exists():
                cmd = json.loads(path.read_text())
                alive.write_text(json.dumps({"state": "busy", "op": cmd["op"], "n": index, "time": time.time()}))
            else:
                alive.write_text(json.dumps({"state": "idle", "next": index, "time": time.time(),
                                             "loaded": trainer.loaded}, default=str))
                if time.time() - last < HEARTBEAT_S:
                    poll_early(trainer)
                    time.sleep(min(1.0, float(os.environ.get("RL1_DIST_POLL_S") or 1.0)))
                    continue
                cmd = {"op": "_idle"}
        try:
            cmd = d.bcast(cmd)
        except Exception as error:  # noqa: BLE001 - a follower is gone: answer (rank 0) and stop
            if d.rank == 0 and cmd.get("op") != "_idle":
                answer(trainer, index, {"ok": False, "error": f"dist control broadcast failed: {error!r}"[:2000]})
            raise
        last = time.time()
        if cmd["op"] == "_idle":
            poll_early(trainer)  # every rank: its share of the next update's first encodes (trainer_fast.early_ahead)
            continue
        result = (trainer.run_op(cmd, index) if runs_here(cmd["op"], d.rank)
                  else {"ok": True, "skipped": "rank 0 only"})
        try:
            results = d.gather(result)
        except Exception as error:  # noqa: BLE001
            if d.rank == 0:
                answer(trainer, index, {**result, "ok": False,
                                        "error": f"dist result gather failed (a rank is gone): {error!r}"[:2000]})
            raise
        if d.rank == 0:
            answer(trainer, index, merge(cmd, results))
        index += 1
        if cmd["op"] == "exit":
            return


def poll_early(trainer):
    """The trainer module's early-ahead hook (rl1_trainer.early_poll), when this process has one."""
    poll = getattr(sys.modules.get(type(trainer).__module__), "early_poll", None)
    if poll is not None:
        poll()


def answer(trainer, index, result):
    folder = trainer.dir / "done"
    tmp = folder / f"{index:04d}.json.tmp"
    tmp.write_text(json.dumps(result, default=str))
    tmp.rename(folder / f"{index:04d}.json")


def recorder_dir(base, rank):
    """Each rank's own records (rank 0 keeps the single-GPU folder)."""
    return Path(base) if rank == 0 else Path(str(base) + f"-rank{rank}")
