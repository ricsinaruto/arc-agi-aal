"""Dynamic stream count (throughput E1): more concurrent games while the KV cache has room.

Copied into a bundle as src/arena/dynamic_streams.py by nebius_kaggle_faithful.py --dynamic-streams. The trial loop
calls ``step`` after each scheduler tick. Every ``every_seconds`` it reads vLLM's kv_cache_usage_perc from the
monitor's latest sample and moves ``scheduler.active_limit`` by one inside [lo, hi]:
  up   when usage < up_kv and nothing waits at the KV gate (the gate's reservations still bound every admission, so a
       higher limit never over-admits; requests wait instead of being preempted);
  down when usage > down_kv. A lowered limit never stops a game: the scheduler releases a game only at its next
       context eviction (drain), without a replacement (scheduler patch ``dynamic_release``).
Which game is admitted next is the scheduler's own order (equal priority); only the count moves.
"""
import math

KV = "vllm:kv_cache_usage_perc"


def settings(raw):
    cfg = {"every_seconds": 30.0, **raw}
    lo, hi, up, down = int(cfg["lo"]), int(cfg["hi"]), float(cfg["up_kv"]), float(cfg["down_kv"])
    if not (1 <= lo <= hi and 0.0 < up < down < 1.0 and cfg["every_seconds"] > 0):
        raise ValueError(f"dynamic_streams: need 1 <= lo <= hi and 0 < up_kv < down_kv < 1: {raw}")
    return {**cfg, "lo": lo, "hi": hi, "up_kv": up, "down_kv": down, "every_seconds": float(cfg["every_seconds"])}


class DynamicStreams:
    def __init__(self, raw, log=None):
        self.cfg = settings(raw)
        self.log = log
        self.last = -math.inf
        self.started = False
        self.changes = {"up": 0, "down": 0}
        self.max_limit = self.cfg["lo"]

    def _note(self, **fields):
        if self.log is not None:
            self.log({"event": "dynamic_streams", **fields})

    def step(self, elapsed, samples, scheduler, gate):
        cfg = self.cfg
        if not self.started:  # the run starts at lo whatever the boot layout chose (it sized for max_num_seqs)
            self.started = True
            boot = scheduler.active_limit
            scheduler.active_limit = cfg["lo"]
            self._note(what="start", boot_limit=boot, limit=cfg["lo"], **{k: cfg[k] for k in ("hi", "up_kv", "down_kv")})
        if elapsed - self.last < cfg["every_seconds"] or not samples:
            return None
        sample = samples[-1]
        kv = (sample.get("metrics") or {}).get(KV)
        age = elapsed - float(sample.get("elapsed_seconds", -math.inf))
        if kv is None or age > 2 * cfg["every_seconds"]:  # no fresh engine reading: hold
            return None
        self.last = elapsed
        limit, waiting = scheduler.active_limit, len(getattr(gate, "queue", ()))
        new = limit
        if kv > cfg["down_kv"] and limit > cfg["lo"]:
            new = limit - 1
        elif kv < cfg["up_kv"] and waiting == 0 and limit < cfg["hi"]:
            new = limit + 1
        if new != limit:
            scheduler.active_limit = new
            self.changes["up" if new > limit else "down"] += 1
            self.max_limit = max(self.max_limit, new)
            self._note(what="up" if new > limit else "down", limit=new, kv=round(float(kv), 4), waiting=waiting,
                       active=len(scheduler.active()), elapsed=round(elapsed, 1))
        return new

    def summary(self):
        return {"cfg": self.cfg, "changes": dict(self.changes), "max_limit": self.max_limit}
