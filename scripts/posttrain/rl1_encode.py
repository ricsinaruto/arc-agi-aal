"""Window encoding for ``rl1_trainer`` (MFU stage W, 22 Oct): the CPU half of ``Trainer.attempt_window`` (rollout rows,
chat template, tokens, images) as a plain function, so encoder processes can run it beside the trainer (the GIL held
one encode thread to 25-35% of the host's cores in the GSPO old pass), and host-side window tensors that the old pass
keeps for its update steps.

A window encoded here on the host and moved to the card holds the same tensors as one encoded on the card: the
encode is integer token work and the arrays are built from Python lists; only the device differs."""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

STATE = {}  # an encoder process's processor-backed ``encoded`` and settings (``init``)


def reply_weight(spec, row):
    """A reply's credit: 0 for a restored prefix (None under ``whole_segment``: the window's walk stops there), else
    ``spec["reply_w"][n]`` (mg6 per-turn credit) or the try's ``w``."""
    if row.get("scripted"):
        return None if spec.get("whole_segment") else 0.0
    per = spec.get("reply_w") or {}
    return float(per[str(row.get("n"))]) if str(row.get("n")) in per else float(spec.get("w", 1.0))


ROWS, ROWS_KEEP = {}, 2  # host path: compacted rows of the last rollouts read (a try's windows read its file once)


def read_rows(spec, cached=False):
    """The attempt's rollout rows as the trainer reads them (compacted; each row's credit ``w``). ``cached``: the
    compacted rows of a file read before (same size and mtime) are reused (json + compaction were ~2/3 of a
    window's encode in the stage-W bench); nothing downstream mutates them, each call gets its own row dicts."""
    import rl_job

    compact = rl_job.rollouts().compact
    key = None
    if cached:
        stat = Path(spec["rollout"]).stat()
        key = (str(spec["rollout"]), stat.st_size, stat.st_mtime_ns)
        if key in ROWS:
            ROWS[key] = ROWS.pop(key)  # most recent last
            return [{**row, "w": reply_weight(spec, row)} for row in ROWS[key]]
    rows, base = [], []
    with open(spec["rollout"]) as file:
        for line in file:
            row = json.loads(line)
            if row.get("top") is not None:
                row = compact(row)
            if key is not None:
                base.append(dict(row))
            # a restored attempt's scripted prefix (arena.restore) is context only: weight 0, never trained
            row["w"] = reply_weight(spec, row)
            rows.append(row)
    if key is not None:
        ROWS[key] = base
        while len(ROWS) > ROWS_KEEP:
            ROWS.pop(next(iter(ROWS)))
    return rows


def encode_window(encoded, think_end, default_window, spec, seed=0, device="cuda:0", cached=False):
    """``rl_job.attempt_example`` on the spec's rows with ``encoded(capture, pair=True)``; ``name`` and
    ``temperature`` set. Returns ``(made or None, rows)``."""
    import rl_job

    rows = read_rows(spec, cached)
    whole = bool(spec.get("whole_segment"))  # RL C2 (4a): the drain segment whole, every model output trained
    made = rl_job.attempt_example(lambda capture: encoded(capture, pair=True), {"replies": rows},
                                  window=int(spec.get("window") or default_window), rng=random.Random(seed), device=device,
                                  think_end=think_end, end=spec.get("end"),
                                  floor=0 if whole else int(spec.get("floor") or 16384),
                                  margin=1 if whole else int(spec.get("margin") or 8192),  # 1: the logit before the first trained token
                                  credited_only="scripted" if whole else bool(spec.get("credited_only")),
                                  end_request=spec.get("end_request"),
                                  **({"gameplay": False} if whole else {}))
    if made is not None:
        made["name"] = spec.get("name") or Path(spec["rollout"]).parent.name
        made["temperature"] = float(next((r["sampling"]["temperature"] for r in rows if r.get("sampling")), 0.7))
    return made, rows


def processor_encoder(model, window):
    """The trainer's encoder on the host: ``(encoded, think_end)`` from the checkpoint's processor."""
    from transformers import AutoProcessor

    import flash_qlora_job as fq

    processor = AutoProcessor.from_pretrained(model)
    encoded, _ = fq.example_maker(processor, [], tail=window, phases=[window], device=None)
    return encoded, processor.tokenizer.convert_tokens_to_ids("</think>")


def init(model, window, loader=None, shm=True, barrier=None):
    """Encoder process start: one processor per process (``loader(model, window)`` for tests). ``shm``: windows go
    back as torch tensors (torch moves them to shared memory: one copy, nothing through the pipe); else as numpy
    arrays pickled through the pipe (the stage-W bench: ~0.6 s per 270 MiB window in the trainer's pool thread).
    ``barrier``: the pool's processes meet in ``warm`` (one warm-up each)."""
    import torch

    torch.set_num_threads(1)  # one encode per process; the processes are the parallelism
    STATE["encoded"], STATE["think_end"] = (loader or processor_encoder)(model, window)
    STATE["window"], STATE["shm"], STATE["barrier"] = int(window), bool(shm), barrier


def warm(nbytes=1 << 22, timeout=300.0):
    """RL1_ENCODE_WARM: an encoder process's first result through the pool before any window. The return path's
    first use (shared memory or pickle) costs ~3 s per process (Modal CPU, D1 rollouts: first window 3.7-5.4 s, the
    same window again 0.7-0.8 s; after this warm-up the first window took 0.93 s). With the pool's barrier every
    process takes exactly one warm-up (a process that finished its own would otherwise take another's)."""
    import torch

    barrier = STATE.get("barrier")
    met = None
    if barrier is not None:
        try:
            barrier.wait(timeout)
            met = True
        except Exception:  # noqa: BLE001 - a broken or timed-out barrier: this process warms anyway
            met = False
    made = {"inputs": {"input_ids": torch.zeros(nbytes // 8, dtype=torch.int64)}, "arrays": {}, "pid": os.getpid(),
            "met": met}
    return made if STATE.get("shm", True) else pack(made)


def window(spec, seed=0):
    """An encoder process's window: host tensors (``shm``) or ``pack``ed arrays; None: no window fits. A spec with
    ``served_in_encoder`` (R3 on, set by the trainer's feed) also gets its served-route table here, from the rows this
    process already read (MFU-1 m1h1: resolving in the trainer's feed thread re-parsed the rollout and held the GIL,
    stalling the main thread ~2.4-2.9 s per try)."""
    made, rows = encode_window(STATE["encoded"], STATE["think_end"], STATE["window"], spec, seed, device="cpu", cached=True)
    if made is not None and spec.get("mrope"):
        made["mrope"] = host_mrope(made["inputs"], spec["mrope"])
    if made is not None and spec.get("served_in_encoder") and spec.get("served_route"):
        import torch

        table, audit, layers = served_table(spec, rows, made, made["inputs"]["input_ids"][0].numpy())
        made["served_host"] = None if table is None else torch.from_numpy(table)
        made["served_host_audit"], made["served_layers"] = audit, layers
    return made if STATE["shm"] else pack(made)


def host_mrope(inputs, how):
    """MFU-1 (23 Oct, host_mrope): the window's M-RoPE positions on the host, by the model's own ``get_rope_index``
    (``how["cls"]``, the trainer model's core class) on CPU tensors: integer index work, the same result as on the
    card, done here instead of on the trainer's main thread (~0.1-0.2 s of GPU idle per window: a .tolist() of the
    whole sequence, a groupby over it and three .item() syncs per image). None for a window without images."""
    import functools
    import importlib
    import types

    if inputs.get("pixel_values") is None:
        return None
    module, _, name = how["cls"].rpartition(".")
    cls = getattr(importlib.import_module(module), name)
    shim = types.SimpleNamespace(config=types.SimpleNamespace(
        vision_config=types.SimpleNamespace(spatial_merge_size=int(how["merge"]))))
    shim.get_vision_position_ids = functools.partial(cls.get_vision_position_ids, shim)
    positions, _ = cls.get_rope_index(shim, inputs["input_ids"], mm_token_type_ids=inputs.get("mm_token_type_ids"),
                                      image_grid_thw=inputs.get("image_grid_thw"))
    return positions


def expand_one(spec, default_window):
    """``Trainer.expand`` of one spec (a spec with ``whole`` becomes one spec per credited window end); a plain
    function so the trainer's encoder processes can parse the tries' rollouts (~0.3-0.5 s of JSON each) in parallel."""
    import rl_job

    if not spec.get("whole"):
        return [spec]
    rows = []
    with open(spec["rollout"]) as file:
        for line in file:
            row = json.loads(line)
            row["w"] = reply_weight(spec, row)
            rows.append(row)
    ends = rl_job.credited_ends({"replies": rows}, int(spec.get("window") or default_window),
                                **({"gameplay": False} if spec.get("whole_segment") else {}))
    name = spec.get("name") or Path(spec["rollout"]).parent.name
    return [{**spec, "whole": False, "end": end, "name": f"{name}#{i}", "first_window": end == min(ends)}
            for i, end in enumerate(ends)]


SERVED = {}  # served-route stores by directory (one per process)


def served_table(spec, rows, made, target, stores=None):
    """R3: the play server's routing for a window (``arena.served_route``): ``(int16 [T, layers, k] or None, audit,
    the store's layers)``; the end request's own record, its append-only chain, then from-scratch requests. ``target``:
    the window's token ids (numpy). Used by the trainer's feed and by the encoder processes."""
    import served_route as sr

    stores = SERVED if stores is None else stores
    store = stores.get(spec["served_route"])
    if store is None:
        store = stores[spec["served_route"]] = sr.Store(spec["served_route"], scratch=True)
    else:
        store.refresh()  # kept across updates: requests written and pruned since
    by_n = {r.get("n"): r for r in rows}
    end = by_n.get(made["replies"][-1]["request"])
    own = store.record(end.get("rid")) if end else None
    if own is None:
        return None, {"error": "no record of the end request"}, store.layers
    ordered = sorted((r for r in rows if r.get("n") is not None), key=lambda r: r["n"])
    totals, chain = {}, []
    for r in ordered:
        totals[r["n"]] = int(r.get("keep") or 0) + len(r.get("messages") or [])
    k = ordered.index(end)
    while k > 0 and int(ordered[k].get("keep") or 0) >= totals[ordered[k - 1]["n"]]:
        k -= 1
        chain.append(store.record(ordered[k].get("rid")))
    table, audit = sr.resolve(target, own, chain, store.scratch_for(own, target), store.donors(own, target))
    audit["chain_requests"] = len(chain)
    return table, audit, store.layers


def shm_free_gib():
    import shutil

    try:
        return shutil.disk_usage("/dev/shm").free / 2**30
    except OSError:
        return 0.0


def pack(made):
    """Tensors as numpy arrays for the trip to the trainer: a plain pickle through the pool's pipe (no /dev/shm)."""
    import torch

    def walk(value):
        if isinstance(value, torch.Tensor):
            return Packed(value.numpy())
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    return None if made is None else walk(made)


def unpack(made):
    import torch

    def walk(value):
        if isinstance(value, Packed):
            return torch.from_numpy(value.array)
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    return None if made is None else walk(made)


class Packed:
    __slots__ = ("array",)

    def __init__(self, array):
        self.array = array

    def __getstate__(self):
        return (self.array,)

    def __setstate__(self, state):
        self.array = state[0]


def tensors(made):
    """Every tensor of a window dict (nested dicts and lists)."""
    import torch

    out = []

    def walk(value):
        if isinstance(value, torch.Tensor):
            out.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)

    walk(made)
    return out


HOST_KEYS = ("input_ids", "mm_token_type_ids", "image_grid_thw", "pixel_values")


def to_device(made, device, mrope=None):
    """A copy of a host window with its ``inputs`` and ``arrays`` tensors on ``device`` (the host copy is untouched:
    the update adds ``old`` and OPD keys to the copy's ``arrays``). U2 (27 Oct): ``host_inputs`` keeps the host
    tensors the pass reads indices from (no card read for them), ``support_host`` the chunked head's row splits;
    ``mrope`` (``host_mrope``'s ``how``): a window with images and no encoder-made positions gets them here."""
    import torch

    def move(d):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}

    out = {**made, "inputs": move(made["inputs"]), "arrays": move(made["arrays"])}
    host = {k: v for k in HOST_KEYS if isinstance(v := made["inputs"].get(k), torch.Tensor) and v.device.type == "cpu"}
    if host:
        out["host_inputs"] = host
    if made.get("mrope") is None and mrope and made["inputs"].get("pixel_values") is not None and len(host) == 4:
        made = {**made, "mrope": host_mrope(made["inputs"], mrope)}
    if isinstance(made.get("mrope"), torch.Tensor):
        out["mrope"] = made["mrope"].to(device)
    support = made["arrays"].get("support")
    if isinstance(support, torch.Tensor) and support.device.type == "cpu":  # the chunked head's row splits (U2)
        out["support_host"] = support
    return out


def spec_key(spec, seed=0):
    """What a window depends on: the spec, the seed and the rollout file (size, mtime)."""
    stat = Path(spec["rollout"]).stat()
    return json.dumps([spec, seed, stat.st_size, stat.st_mtime_ns], sort_keys=True, default=str)


class Cache:
    """The trainer's window_cache: host windows by ``spec_key``. Nothing fills it since the GSPO old pass's removal
    (every lookup misses); kept for the trainer's lookups and its summary row."""

    def __init__(self, cap_gib=0.0):
        self.cap = int(float(cap_gib) * 2**30)
        self.rows, self.bytes = {}, 0
        self.stats = {"kept": 0, "over_cap": 0, "hits": 0, "misses": 0, "peak_gib": 0.0}

    def clear(self):
        self.rows.clear()
        self.bytes = 0

    def pop(self, key):
        row = self.rows.pop(key, None)
        if row is None:
            self.stats["misses"] += 1
            return None
        self.bytes -= row[1]
        self.stats["hits"] += 1
        return row[0]

    def summary(self):
        return {**self.stats, "windows": len(self.rows), "gib": round(self.bytes / 2**30, 2), "cap_gib": round(self.cap / 2**30, 2)}
