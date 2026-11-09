"""ARC_PINNED_LOAD=1 (boot cut, MFU-2 24 Oct): the safetensors weight iterator (lazy strategy) reads each file's
selected tensors with parallel preads into pinned host chunks and yields views of them, instead of views of the
file's mmap. RTX cProfile (b2-mfu2-boot-v1 p): Tensor.copy_ was 77 of 134 s of the load (94K pageable copies from
mmap pages); from pinned memory the copy is a DMA. Same names, order, dtypes, shapes and bytes as the mmap path;
skipped weights are never read. Tensors over 1 GiB (the PLE host table) and any tensor whose dtype
or alignment the chunk path cannot view keep the mmap view. Anchor-matched; an unknown file keeps the original
iterator (ARC_PINNED_LOAD_PATCH_SKIPPED).

ARC_DRAFT_LOAD_FILTER=1 (stage H, 26 Oct; with ARC_PINNED_LOAD=1): the MTP draft's load reads only the tensors its own
``load_weights`` keeps (``_remap_mtp_weight_name(name) is not None`` of the draft model's module), not the whole
126 GiB checkpoint again (T3 boot: 47.7 s for 296,475 tensors, all but the MTP head dropped by that same remap). Exact:
the dropped names never reached the model. ``default_loader.load_weights`` sets the filter per load (None for any
other model); the pinned iterator reads it once, at its first item.

ARC_PINNED_LOAD_FIXED=1: every chunk buffer is allocated at the full chunk size (one size class), so torch's caching
host allocator reuses freed chunks instead of pinning new ones per span size (T3: 191 host allocations, 82.7 s of
allocation summed over the readers). Same bytes; GPU-unmeasured."""

import hashlib
import json
import os
from pathlib import Path

ENV = "ARC_PINNED_LOAD"
RELATIVE = "vllm/model_executor/model_loader/weight_utils.py"
ANCHOR = "    leftover_state_dict: dict[str, torch.Tensor] = {}\n    for st_file in tqdm(\n"
BRANCH = ("    if safetensors_load_strategy in (None, \"lazy\") and os.environ.get(\"ARC_PINNED_LOAD\") == \"1\":"
          "  # ARC pinned_load\n"
          "        yield from _arc_pinned_iterator(sorted_files, local_expert_ids)\n"
          "        return\n")

SOURCE = '''

_ARC_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64", "I8": "int8",  # ARC pinned_load
               "I16": "int16", "I32": "int32", "I64": "int64", "U8": "uint8", "BOOL": "bool",
               "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2", "U16": "uint16", "U32": "uint32",
               "U64": "uint64", "F8_E8M0": "float8_e8m0fnu"}


def _arc_pinned_plan(st_file, names, chunk, limit):
    """Chunks (byte spans of the data section, offset order, gaps <= 1 MiB merged) for the selected tensors that fit
    the chunk path; the rest (too large, unknown dtype) come back in ``direct``."""
    import struct

    with open(st_file, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    base = 8 + size
    entries, direct = {}, set()
    for name in names:
        info = header.get(name)
        dtype = getattr(torch, _ARC_DTYPES.get(info["dtype"], "") if info else "", None)
        begin, end = info["data_offsets"] if info else (0, 0)
        if dtype is None or end - begin > limit or (end - begin) % dtype.itemsize:
            direct.add(name)
        else:
            entries[name] = (base + begin, base + end, dtype, tuple(info["shape"]))
    spans, current = [], None
    for name in sorted(entries, key=lambda n: entries[n][0]):
        begin, end = entries[name][:2]
        if current and begin - current[1] <= (1 << 20) and max(end, current[1]) - current[0] <= chunk:
            current[1] = max(end, current[1])
            current[2].append(name)
        else:
            current = [begin, end, [name]]
            spans.append(current)
    return entries, spans, direct


_ARC_NAME_FILTER = None  # ARC draft_load_filter: set per load by default_loader.load_weights


def _arc_pinned_read(st_file, begin, end, stats, fixed=0):
    pin = torch.cuda.is_available()
    t0 = time.perf_counter()
    if fixed and end - begin <= fixed:  # ARC_PINNED_LOAD_FIXED: one size class, reused by the caching host allocator
        buf = torch.empty(fixed, dtype=torch.uint8, pin_memory=pin)[:end - begin]
    else:
        buf = torch.empty(end - begin, dtype=torch.uint8, pin_memory=pin)
    t1 = time.perf_counter()
    view, done = memoryview(buf.numpy()), 0
    fd = os.open(st_file, os.O_RDONLY)
    try:
        while done < end - begin:
            got = os.preadv(fd, [view[done:]], begin + done)
            if got <= 0:
                raise OSError(f"short read {st_file} at {begin + done}")
            done += got
    finally:
        os.close(fd)
    stats["alloc_s"] += t1 - t0  # summed over reader threads
    stats["pread_s"] += time.perf_counter() - t1
    return buf


def _arc_pinned_iterator(sorted_files, local_expert_ids):
    """The lazy iterator's (name, tensor) stream with chunk tensors (64 MiB) read ahead (one file, 8 preads) into
    pinned memory; torch's caching host allocator reuses a chunk once every view of it is gone. Tensors up to 64 KiB
    are copied out, so one a loader keeps holds no chunk."""
    import concurrent.futures

    chunk, limit, threads, small, ahead_files = 64 << 20, 1024 << 20, 8, 64 << 10, 1
    stats = {"files": 0, "chunk_tensors": 0, "direct_tensors": 0, "chunk_bytes": 0, "wait_s": 0.0, "alloc_s": 0.0,
             "pread_s": 0.0, "cloned": 0}
    keep = _ARC_NAME_FILTER  # the draft's own name filter for this load (ARC_DRAFT_LOAD_FILTER), else None
    fixed = chunk if os.environ.get("ARC_PINNED_LOAD_FIXED") == "1" else 0
    if keep is not None:
        stats["name_filter"] = getattr(keep, "__name__", "filter")
        stats["filtered_out"] = 0
    started = time.perf_counter()

    def plan(st_file):
        with safe_open(st_file, framework="pt") as f:
            names = [n for n in f.keys() if not should_skip_weight(n, local_expert_ids)]  # noqa: SIM118
        if keep is not None:
            kept = [n for n in names if keep(n)]
            stats["filtered_out"] += len(names) - len(kept)
            names = kept
        entries, spans, direct = _arc_pinned_plan(st_file, names, chunk, limit)
        futures = {}
        for span in spans:
            future = pool.submit(_arc_pinned_read, st_file, span[0], span[1], stats, fixed)
            for name in span[2]:
                futures[name] = (future, span[0])
        return st_file, names, entries, futures, direct

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        ahead = [plan(p) for p in sorted_files[:1 + ahead_files]]
        index = 1 + ahead_files
        while ahead:
            st_file, names, entries, futures, direct = ahead.pop(0)
            if index < len(sorted_files):
                ahead.append(plan(sorted_files[index]))
                index += 1
            stats["files"] += 1
            if not names:  # nothing kept from this file (draft filter): its header was all that was read
                continue
            with safe_open(st_file, framework="pt") as f:
                for name in names:
                    if name in direct:
                        stats["direct_tensors"] += 1
                        yield name, f.get_tensor(name)
                        continue
                    future, span_begin = futures.pop(name)
                    begin, end, dtype, shape = entries[name]
                    waited = time.perf_counter()
                    buf = future.result()
                    stats["wait_s"] += time.perf_counter() - waited
                    try:
                        tensor = buf[begin - span_begin:end - span_begin].view(dtype).reshape(shape)
                    except RuntimeError:  # an offset the dtype view cannot take: the mmap view instead
                        stats["direct_tensors"] += 1
                        yield name, f.get_tensor(name)
                        continue
                    del buf
                    if end - begin <= small:  # a small tensor a loader may keep must not hold its whole chunk
                        tensor = tensor.clone()
                        stats["cloned"] += 1
                    stats["chunk_tensors"] += 1
                    stats["chunk_bytes"] += end - begin
                    yield name, tensor
    host = getattr(torch.cuda, "host_memory_stats", None)
    if host is not None and torch.cuda.is_available():
        got = host()
        stats["host"] = {k: got[k] for k in ("allocations.peak", "allocated_bytes.peak", "num_host_alloc",
                                             "host_alloc_time.total") if k in got}
    empty = getattr(torch._C, "_host_emptyCache", None)
    if empty is not None and torch.cuda.is_available():
        empty()
    stats["seconds"] = round(time.perf_counter() - started, 2)
    for key in ("wait_s", "alloc_s", "pread_s"):
        stats[key] = round(stats[key], 2)
    print("ARC_PINNED_LOAD " + json.dumps(stats, sort_keys=True), flush=True)
'''


LOADER_RELATIVE = "vllm/model_executor/model_loader/default_loader.py"
LOADER_ANCHOR = "        self._init_ep_weight_filter(model_config)\n"
LOADER_BRANCH = "        _arc_set_name_filter(model)  # ARC draft_load_filter\n"
LOADER_SOURCE = '''


def _arc_set_name_filter(model):  # ARC draft_load_filter
    """ARC_DRAFT_LOAD_FILTER=1 with ARC_PINNED_LOAD=1: an MTP draft's load reads only the names its module's
    _remap_mtp_weight_name keeps; any other model's load reads everything (the filter is set on every load)."""
    import sys

    from vllm.model_executor.model_loader import weight_utils as _arc_wu

    keep = None
    if (os.environ.get("ARC_DRAFT_LOAD_FILTER") == "1" and os.environ.get("ARC_PINNED_LOAD") == "1"
            and type(model).__name__.endswith("MTP")):
        remap = getattr(sys.modules.get(type(model).__module__), "_remap_mtp_weight_name", None)
        if remap is not None:
            def keep(name, _remap=remap):
                return _remap(name) is not None
            keep.__name__ = type(model).__module__ + "._remap_mtp_weight_name"
    if hasattr(_arc_wu, "_ARC_NAME_FILTER"):
        _arc_wu._ARC_NAME_FILTER = keep
'''


def patched_loader(original):
    text = original.decode()
    if "ARC draft_load_filter" in text:
        raise RuntimeError("draft_load_filter patch: file already carries the edit")
    if text.count(LOADER_ANCHOR) != 1:
        raise RuntimeError("draft_load_filter anchor is not unique")
    text = text.replace(LOADER_ANCHOR, LOADER_BRANCH + LOADER_ANCHOR).rstrip("\n") + "\n" + LOADER_SOURCE
    compile(text, LOADER_RELATIVE, "exec")
    return text.encode()


def patched_source(original):
    text = original.decode()
    if "ARC pinned_load" in text:
        raise RuntimeError("pinned_load patch: file already carries the edit")
    if text.count(ANCHOR) != 1:
        raise RuntimeError("pinned_load anchor is not unique")
    text = text.replace(ANCHOR, BRANCH + ANCHOR).rstrip("\n") + "\n" + SOURCE
    compile(text, RELATIVE, "exec")
    return text.encode()


def apply(site, output):
    """Never raises: an unrecognized file keeps the mmap iterator. Marker pinned-load.json (not *-patch.json: no
    compiled code changes, the boot-cache key ignores it)."""
    target = Path(site) / RELATIVE
    original = target.read_bytes()
    marker = Path(output) / "pinned-load.json"
    if b"ARC pinned_load" in original:
        return json.loads(marker.read_text()) if marker.exists() else {"applied_earlier": True}
    try:
        patched = patched_source(original)
    except Exception as error:  # noqa: BLE001 - a boot cut, never a reason to fail a boot
        print("ARC_PINNED_LOAD_PATCH_SKIPPED " + json.dumps({"error": str(error)[:200]}), flush=True)
        return None
    loader = None
    if os.environ.get("ARC_DRAFT_LOAD_FILTER") == "1":  # the draft's filtered read (exact; see the module doc)
        loader_path = Path(site) / LOADER_RELATIVE
        try:
            loader_original = loader_path.read_bytes()
            loader = (loader_path, loader_original, patched_loader(loader_original))
        except Exception as error:  # noqa: BLE001 - the draft reads everything, as before
            print("ARC_DRAFT_LOAD_FILTER_SKIPPED " + json.dumps({"error": str(error)[:200]}), flush=True)
    target.write_bytes(patched)
    identity = {"original_sha256": hashlib.sha256(original).hexdigest(),
                "patched_sha256": hashlib.sha256(patched).hexdigest()}
    if loader is not None:
        loader[0].write_bytes(loader[2])
        identity["draft_load_filter"] = {"original_sha256": hashlib.sha256(loader[1]).hexdigest(),
                                         "patched_sha256": hashlib.sha256(loader[2]).hexdigest()}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    print("ARC_PINNED_LOAD_PATCH " + json.dumps(identity), flush=True)
    return identity
