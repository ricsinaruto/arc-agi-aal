"""fla 0.5.2 / Triton 3.6 on B300 (MFU-1 gdnrace-b1, f00e49669): chunk_fwd_kernel_o's BK=BV=128 autotune config now
and then writes one wrong 16x128 tile (|err| 0.02-0.05), so the GDN forward is neither repeatable nor exact; the
BK=BV=64 config repeats bit-exactly. pin() keeps only that config (RL1_GDN_PIN_O=0: off). No fla (CPU): no-op."""
import os

STATE = {"done": False, "config": None, "saved": None}


def pin(module=None):
    """Returns the kept config's kwargs, or None when fla is absent or the pin is off."""
    if STATE["done"]:
        return STATE["config"]
    if os.environ.get("RL1_GDN_PIN_O", "1") != "1":
        return None
    if module is None:
        try:
            from fla.ops.common import chunk_o as module
        except Exception:  # noqa: BLE001 - CPU images without fla / Triton
            return None
    at = module.chunk_fwd_kernel_o
    while not hasattr(at, "configs"):
        at = at.fn
    keep = [c for c in at.configs if c.kwargs.get("BK") == 64 and c.kwargs.get("BV") == 64]
    if not keep:
        raise RuntimeError(f"fla_pin: no BK=BV=64 config in {[str(c) for c in at.configs]}")
    STATE["saved"] = (at, list(at.configs))
    at.configs = keep[:1]
    at.cache.clear()
    STATE.update(done=True, config=dict(keep[0].kwargs))
    return STATE["config"]


def unpin():
    """Restores the autotuner's configs (diagnosis only); returns None."""
    if STATE.get("saved"):
        at, configs = STATE["saved"]
        at.configs = configs
        at.cache.clear()
    STATE.update(done=False, config=None, saved=None)
    return None
