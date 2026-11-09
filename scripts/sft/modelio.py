"""Model plumbing for the SFT trainer (from flash_qlora_job): the n-gram table kept in host RAM at load, parallel
whole-file reads for the load, M-RoPE positions and image features per slice, and a read-only view of a prefix cache
for the window pass."""
import contextlib
import json
import os
import struct
import threading

ST_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64", "I64": "int64", "I32": "int32",
             "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool", "F8_E4M3": "float8_e4m3fn",
             "F8_E5M2": "float8_e5m2"}


STATS_LOCK = threading.Lock()  # the read split's counters, shared by every file's read
UPLOAD_RESERVE = 16 << 30  # a file goes to the GPU only with this much left free after it


def room(nbytes, device):
    """True when ``device`` has ``nbytes`` + UPLOAD_RESERVE free (always on a non-CUDA device); a file without room
    stays in host RAM and the loader moves its tensors one by one."""
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        return True
    free, _ = torch.cuda.mem_get_info(device)
    return free >= nbytes + UPLOAD_RESERVE


class _Prefetch:
    """One safetensors file read whole as ``piece``-byte ranges on the pool, all files in flight from the start (the
    loader opens them one by one; v5 read ~2 files at a time at 1.6 GiB/s); then, given a ``device``, sent there in one
    copy so the loader's per-tensor ``.to`` is a no-op. ``result()`` gives the buffer."""

    def __init__(self, pool, path, stats, device=None, piece=256 << 20, names=None, gap=1 << 20):
        from concurrent.futures import Future

        import torch

        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n))
        self.extra = meta.pop("__metadata__", None)
        if names is not None:  # only the tensors the index maps to this file (v8: whole base files, ~104 GiB)
            meta = {k: v for k, v in meta.items() if k in names}
        self.start, self.stats, self.path = 8 + n, stats, path
        if device is not None and any("ngram_embedding." in k for k in meta):
            device = None  # the n-gram table stays in host RAM (modelio.table_in_host_ram)
        self.device, self.done, self.lock = device, Future(), threading.Lock()
        spans = []  # file spans holding the wanted tensors, neighbours closer than ``gap`` merged
        for lo, hi in sorted(tuple(v["data_offsets"]) for v in meta.values()):
            if spans and lo - spans[-1][1] <= gap:
                spans[-1][1] = max(spans[-1][1], hi)
            else:
                spans.append([lo, hi])
        base, at = {}, 0  # each span's place in the buffer (64-byte aligned, so in-span alignment is the file's)
        for lo, hi in spans:
            base[lo], at = at, at + (hi - lo + 63) // 64 * 64
        self.meta = {}
        for k, v in meta.items():
            lo, hi = v["data_offsets"]
            s0 = max(s for s in base if s <= lo)
            self.meta[k] = {**v, "data_offsets": [base[s0] + lo - s0, base[s0] + hi - s0]}
        self.buf = torch.empty(at, dtype=torch.uint8)
        ranges = [(base[lo] + a - lo, a, min(a + piece, hi)) for lo, hi in spans for a in range(lo, hi, piece)]
        ranges = ranges or [(0, 0, 0)]
        self.left = len(ranges)
        for to, a, b in ranges:
            pool.submit(self._read, to, a, b)

    def _read(self, to, a, b):
        import time

        began = time.time()
        try:
            view, got = memoryview(self.buf.numpy())[to:to + b - a], 0
            with open(self.path, "rb", buffering=0) as f:
                while got < b - a:
                    k = os.preadv(f.fileno(), [view[got:]], self.start + a + got)
                    if not k:
                        raise OSError(f"short read of {self.path}")
                    got += k
        except BaseException as error:  # noqa: BLE001 - the loader raises it from result()
            if not self.done.done():
                self.done.set_exception(error)
            return
        end = time.time()
        with STATS_LOCK:
            st = self.stats
            st["first"] = min(st.get("first", began), began)
            st["last"] = max(st.get("last", end), end)
            st["thread_s"] = st.get("thread_s", 0.0) + end - began
            st["bytes"] = st.get("bytes", 0) + b - a
        with self.lock:
            self.left -= 1
            last = self.left == 0
        if last:
            buf = self.buf
            if self.device is not None and room(buf.numel(), self.device):
                t = time.time()
                buf = buf.to(self.device)
                with STATS_LOCK:
                    self.stats["upload_s"] = self.stats.get("upload_s", 0.0) + time.time() - t
                    self.stats["upload_last"] = max(self.stats.get("upload_last", 0.0), time.time())
            elif self.device is not None:
                with STATS_LOCK:
                    self.stats["kept_host"] = self.stats.get("kept_host", 0) + 1
            self.buf = None
            self.done.set_result(buf)

    def result(self):
        return self.done.result()


class _File:
    """Stands in for ``safe_open(path)`` in from_pretrained: the header, and tensors as views of the prefetched whole
    file (the same bytes, on the prefetch's device); the buffer is let go once each tensor was taken."""

    def __init__(self, fetch):
        self.fetch, self.meta, self.extra, self.stats = fetch, fetch.meta, fetch.extra, fetch.stats
        self.lock, self.buf, self.left = threading.Lock(), None, set(fetch.meta)

    def keys(self):
        return sorted(self.meta)

    def metadata(self):
        return self.extra

    def get_slice(self, name):
        return _Slice(self, name)

    def get_tensor(self, name):
        import time

        import torch

        info = self.meta[name]
        dtype = getattr(torch, ST_DTYPES[info["dtype"]])
        with self.lock:
            if self.buf is None:
                t = time.time()
                self.buf = self.fetch.result()
                with STATS_LOCK:
                    self.stats["wait_s"] = self.stats.get("wait_s", 0.0) + time.time() - t
            buf = self.buf
            self.left.discard(name)
            if not self.left:
                self.buf = None
        lo, hi = info["data_offsets"]
        raw = buf[lo:hi]
        if lo % torch.empty((), dtype=dtype).element_size():
            raw = raw.clone()
        return raw.view(dtype).reshape(info["shape"])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Slice:
    def __init__(self, file, name):
        self.file, self.name = file, name

    def get_shape(self):
        return list(self.file.meta[self.name]["shape"])

    def get_dtype(self):
        return self.file.meta[self.name]["dtype"]

    def __getitem__(self, index):
        return self.file.get_tensor(self.name)[index]


@contextlib.contextmanager
def parallel_reads(folder, threads=32, margin_gib=64, device=None, piece=256 << 20):
    """from_pretrained's safetensors opens (transformers 5.18: one mmap per file, tensors materialized in key order by
    4 threads with page faults on the volume) served from whole-file prefetches: every file's ranges read on
    ``threads`` threads from the start, each file sent to ``device`` in one copy when given (the n-gram files stay in
    host RAM). Off (yields "mmap") when the host lacks the files' size + ``margin_gib`` of free memory. On exit the
    yielded dict gains the read split: read wall (first start to last end), summed thread seconds, GiB, upload seconds
    (summed) and the loader's wait on reads."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import transformers.modeling_utils as mu

    folder = Path(folder)
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted(set(index.values()))
    size = sum(os.path.getsize(folder / f) for f in files)
    try:
        from arc_infra.runops import hostmem

        avail = hostmem.sample().get("avail_gib")
    except ImportError:
        avail = None
    if avail is not None and avail < size / 2**30 + margin_gib:
        yield {"mode": "mmap", "gib": round(size / 2**30, 1), "avail_gib": avail}
        return
    pool, original, stats, t0 = ThreadPoolExecutor(threads, thread_name_prefix="sft-read"), mu.safe_open, {}, time.time()
    wanted = {}
    for name, f in index.items():
        wanted.setdefault(f, set()).add(name)
    fetches = {str(folder / f): _Prefetch(pool, folder / f, stats, device, piece, wanted[f]) for f in files}

    def opened(path, *args, **kwargs):
        fetch = fetches.get(str(path))
        return _File(fetch) if fetch is not None else original(path, *args, **kwargs)

    # v7: from_pretrained's caching_allocator_warmup reserved the whole model (156 GiB) on top of the files already
    # sent to the GPU; the files are the parameters' storage, so the warm-up only adds a second copy
    warmup = mu.caching_allocator_warmup
    mu.safe_open = opened
    if device is not None:
        mu.caching_allocator_warmup = lambda *a, **k: None
    how = {"mode": "parallel", "gib": round(size / 2**30, 1), "threads": threads, "device": str(device)}
    try:
        yield how
    finally:
        mu.safe_open, mu.caching_allocator_warmup = original, warmup
        pool.shutdown(wait=True)  # a file the loader never opened is still read; nothing runs on after the load
        if stats.get("bytes"):
            how.update(read_wall_s=round(stats["last"] - t0, 1), read_thread_s=round(stats["thread_s"], 1),
                       read_gib=round(stats["bytes"] / 2**30, 1), read_bytes=stats["bytes"], read_wait_s=round(stats.get("wait_s", 0.0), 1),
                       upload_s=round(stats.get("upload_s", 0.0), 1), kept_host=stats.get("kept_host", 0))


def compact(model):
    """Parameters and buffers that are views of a larger storage (a prefetched file holding other tensors, e.g. the
    MTP block's, which the main model skips) get storage of their own, one shared storage at a time (peak: one file)."""
    import torch

    groups = {}
    for mod in model.modules():
        for kind, table in (("p", mod._parameters), ("b", mod._buffers)):
            for name, t in table.items():
                if t is None or t.device.type == "meta":
                    continue
                if t.untyped_storage().nbytes() > t.numel() * t.element_size():
                    groups.setdefault(t.untyped_storage().data_ptr(), []).append((mod, kind, name))
    moved = 0
    for members in groups.values():
        for mod, kind, name in members:
            t = (mod._parameters if kind == "p" else mod._buffers)[name]
            if kind == "p":
                t.data = t.data.clone()
            else:
                mod._buffers[name] = t.clone()
            moved += 1
    if moved and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return moved


def table_in_host_ram():
    """transformers joins the n-gram table's shards on CPU and then moves the result to the device map's device.
    Axolotl's single-GPU map makes that the GPU (check v2: the table on cuda:0 and Axolotl's warning that
    ``ple_cpu_offload`` found nothing in host RAM), which is 95 GiB a B200 does not have during the load."""
    import transformers.core_model_loading as loading

    original = loading.get_device
    if getattr(original, "table_in_host_ram", False):
        return

    def get_device(device_map, name, *args, **kwargs):
        if "ngram_embedding." in name:
            return "cpu"
        return original(device_map, name, *args, **kwargs)

    get_device.table_in_host_ram = True
    loading.get_device = get_device


def positions_and_images(model, inputs, host_index=False):
    """What a pass over a slice of the sequence needs besides its tokens: M-RoPE positions of the whole sequence
    (the model's own incremental rule only knows the images of its first call) and the image features, computed once
    (the vision tower is frozen). ``piece(start, stop)`` gives the keyword arguments of that slice; an image may
    straddle slices. ``host_index``: the per-token image count on the host (one read here, none per slice)."""
    import torch
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    core = (model.get_base_model() if hasattr(model, "get_base_model") else model).model
    ids = inputs["input_ids"]
    if inputs.get("pixel_values") is None:
        positions = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(3, ids.shape[0], -1)
        return lambda start, stop: {"position_ids": positions[:, :, start:stop]}
    with torch.no_grad(), torch.autocast(ids.device.type, dtype=torch.bfloat16):
        features = torch.cat(core.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"],
                                                     return_dict=True).pooler_output, dim=0)
    positions, _ = core.get_rope_index(ids, mm_token_type_ids=inputs["mm_token_type_ids"],
                                       image_grid_thw=inputs["image_grid_thw"])
    before = torch.nn.functional.pad((ids[0] == core.config.image_token_id).cumsum(0), (1, 0))
    if host_index:
        before = before.cpu()

    def piece(start, stop):
        extra = {"position_ids": positions[:, :, start:stop]}
        first, last = int(before[start]), int(before[stop])
        if last > first:
            extra["mm_encoder_outputs"] = {"image": BaseModelOutputWithPooling(pooler_output=(features[first:last],))}
        return extra

    return piece


def read_only(cache):
    """Make the cache's update calls pure for the window pass: each returns what the layer needs (the prefix state
    joined with the new states) and stores nothing. A gradient-checkpointed layer runs twice, and the stock updates
    append or overwrite in place, so the second run would see its own first run."""
    import torch

    layers = cache.layers

    def update(key_states, value_states, layer_idx, *args, **kwargs):
        layer = layers[layer_idx]
        return torch.cat([layer.keys, key_states], dim=-2), torch.cat([layer.values, value_states], dim=-2)

    def update_indexer(states, layer_idx):
        return torch.cat([layers[layer_idx].indexer_keys, states], dim=1)

    def update_conv_state(conv_states, layer_idx, state_idx=0, **kwargs):
        layer = layers[layer_idx]
        if not layer.has_previous_state[state_idx]:
            raise RuntimeError(f"layer {layer_idx} has no prefix state {state_idx}")
        return torch.cat([layer.conv_states[state_idx], conv_states], dim=-1)

    def update_recurrent_state(states, layer_idx, state_idx=0, **kwargs):
        return states

    cache.update, cache.update_indexer = update, update_indexer
    cache.update_conv_state, cache.update_recurrent_state = update_conv_state, update_recurrent_state
    return cache
