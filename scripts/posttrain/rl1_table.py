"""RL1_SHARED_TABLE=1 (26 Oct, H4/H5: the 2-rank preload was skipped, need 1,034 GiB vs 1,015-1,020 room): the PLE n-gram
table (51.2e9 BF16 values, 95.4 GiB) built once per launch straight from the checkpoint's FP8 shards, scaled with
``apply_table_scale``'s arithmetic, into one read-only shared mapping that every local trainer rank uses as its
``ngram_embedding.weight``. transformers leaves the table on meta (``meta_loading``): no shard read, none of the
loader's BF16 pieces plus their concatenation (~190 GiB of the 392 GiB per-rank load peak).

- Local rank 0 builds ``<dir>/rl1-table-<launch>`` (``.part`` until complete, then renamed); the other local ranks wait
  for that name, or its ``.failed`` marker, or ``RL1_SHARED_TABLE_WAIT_S``, then map it. Every mapping is PROT_READ:
  a write faults (the process dies), it never copies. Once every local rank is done with the name, it is unlinked
  (the pages live while mapped).
- ``<launch>``: the torchrun agent's pid under LOCAL_WORLD_SIZE > 1 (each launch builds its own; a container restart
  starts empty); the build removes files of launches whose process is gone.
- Fallback: a rank whose shared table failed builds its own with the same code (own file, read-only; anonymous memory
  if no file fits). A checkpoint without the sharded table: ``start`` returns off and the load is today's."""

import json
import mmap
import os
import re
import threading
import time
from pathlib import Path

ENV = "RL1_SHARED_TABLE"
DIR_ENV = "RL1_SHARED_TABLE_DIR"  # default /dev/shm
WAIT_ENV = "RL1_SHARED_TABLE_WAIT_S"
PREFIX = "rl1-table-"
SHARD = re.compile(r"ngram_embedding\.shard_(\d+)\.weight$")
STATE = {}  # this process's: on, plan, thread, result, table, error
PENDING = {}  # meta_loading: the next materialisation is a table shard
ST_DTYPES = {"F8_E4M3": "float8_e4m3fn", "BF16": "bfloat16", "F16": "float16", "F32": "float32"}


def wanted(environ=None):
    return (os.environ if environ is None else environ).get(ENV) == "1"


def plan(model_dir):
    """The table's shards in index order (shard_<i>, as transformers concatenates them) with their row offsets, the
    column count and the scale's key; raises ValueError if the checkpoint has no complete sharded table."""
    from safetensors import safe_open

    weight_map = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    shards, scale, name = {}, None, None
    for key, file in weight_map.items():
        match = SHARD.search(key)
        if match:
            shards[int(match.group(1))] = (key, file)
            name = key[:match.start()] + "ngram_embedding.weight"
        elif key.endswith("ngram_embedding.weight_scale"):
            scale = {"key": key, "file": file}
    if not shards or sorted(shards) != list(range(len(shards))):
        raise ValueError(f"shared table: shards {sorted(shards)[:5]}... are not shard_0..shard_{len(shards) - 1}")
    out, rows, cols = [], 0, None
    for index in range(len(shards)):
        key, file = shards[index]
        with safe_open(str(Path(model_dir) / file), framework="pt") as handle:
            piece = handle.get_slice(key)
            shape, dtype = piece.get_shape(), piece.get_dtype()
        if len(shape) != 2 or (cols is not None and shape[1] != cols):
            raise ValueError(f"shared table: {key} has shape {shape}")
        cols = shape[1]
        out.append({"key": key, "file": file, "start": rows, "rows": shape[0], "dtype": dtype})
        rows += shape[0]
    return {"name": name, "shards": out, "rows": rows, "cols": cols, "scale": scale,
            "bytes": rows * cols * 2, "dtype": "bfloat16"}


def launch_name(environ=None):
    environ = os.environ if environ is None else environ
    ranks = int(environ.get("LOCAL_WORLD_SIZE") or 1)
    return f"{PREFIX}{os.getppid() if ranks > 1 else os.getpid()}"


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def remove_stale(directory, keep):
    """Files of launches whose process is gone (a crashed trainer's table would hold 95 GiB of tmpfs)."""
    removed = []
    for path in Path(directory).glob(PREFIX + "*"):
        match = re.match(re.escape(PREFIX) + r"(\d+)", path.name)
        if not match or path.name.startswith(keep):
            continue
        if not alive(int(match.group(1))):
            try:
                path.unlink()
                removed.append(path.name)
            except FileNotFoundError:
                pass
    return removed


def reserve(fd, size):
    """The file's whole size, allocated now: a full tmpfs then fails here (ENOSPC), not as SIGBUS mid-copy."""
    try:
        os.posix_fallocate(fd, 0, size)
    except (AttributeError, OSError) as error:
        if isinstance(error, OSError) and error.errno not in (95, 22):  # EOPNOTSUPP / EINVAL: no fallocate here
            raise
        stat = os.fstatvfs(fd)
        if stat.f_bavail * stat.f_frsize < size:
            raise OSError(28, f"shared table: {size} bytes, {stat.f_bavail * stat.f_frsize} free") from None
        os.ftruncate(fd, size)


def fill(out, spec, model_dir, workers=8, step=1 << 20):
    """``out`` [rows, cols] BF16 = the shards in order, FP8 -> BF16 (the loader's cast) then, with a scale,
    ``(piece.float() * scale_rows).to(bfloat16)`` (apply_table_scale)."""
    from concurrent.futures import ThreadPoolExecutor

    import torch
    from safetensors import safe_open

    scale = None
    if spec["scale"]:
        with safe_open(str(Path(model_dir) / spec["scale"]["file"]), framework="pt") as handle:
            scale = handle.get_tensor(spec["scale"]["key"]).float()
    per_row = scale is not None and scale.dim() > 0 and scale.shape[0] == spec["rows"]

    def one(shard):
        with safe_open(str(Path(model_dir) / shard["file"]), framework="pt") as handle:
            source = handle.get_tensor(shard["key"])
        for begin in range(0, shard["rows"], step):
            piece = source[begin:begin + step].to(torch.bfloat16)
            at = shard["start"] + begin
            if scale is not None:
                piece = (piece.float() * (scale[at:at + piece.shape[0]] if per_row else scale)).to(torch.bfloat16)
            out[at:at + piece.shape[0]].copy_(piece)
        return shard["rows"]

    with ThreadPoolExecutor(max(1, workers)) as pool:
        done = sum(pool.map(one, spec["shards"]))
    if done != spec["rows"]:
        raise RuntimeError(f"shared table: {done} rows filled of {spec['rows']}")


def map_read_only(path, spec):
    import warnings

    import torch

    fd = os.open(path, os.O_RDONLY)
    try:
        if os.fstat(fd).st_size != spec["bytes"]:
            raise ValueError(f"shared table: {path} has {os.fstat(fd).st_size} bytes, expected {spec['bytes']}")
        mapped = mmap.mmap(fd, spec["bytes"], flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
    finally:
        os.close(fd)
    with warnings.catch_warnings():  # "the given buffer is not writable": intended
        warnings.simplefilter("ignore")
        table = torch.frombuffer(mapped, dtype=torch.bfloat16)
    STATE.setdefault("mappings", []).append(mapped)
    return table.view(spec["rows"], spec["cols"])


def build_file(path, spec, model_dir, workers=8):
    """``path`` complete (``.part`` while filling, renamed at the end)."""
    import torch

    part = Path(str(path) + ".part")
    fd = os.open(part, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        reserve(fd, spec["bytes"])
        mapped = mmap.mmap(fd, spec["bytes"], flags=mmap.MAP_SHARED)
        out = torch.frombuffer(mapped, dtype=torch.bfloat16).view(spec["rows"], spec["cols"])
        fill(out, spec, model_dir, workers=workers)
        del out
        try:
            mapped.close()
        except BufferError:  # a view still alive: the mapping closes with it
            pass
    except BaseException:
        os.close(fd)
        part.unlink(missing_ok=True)
        raise
    os.close(fd)
    os.rename(part, path)


def own(spec, model_dir, directory, workers=8):
    """This rank's own table: a private read-only file (unlinked once mapped), else anonymous memory."""
    import torch

    path = Path(directory) / f"{launch_name()}.own{os.getpid()}"
    try:
        build_file(path, spec, model_dir, workers=workers)
        table = map_read_only(path, spec)
        path.unlink(missing_ok=True)
        return table, "own", True
    except OSError as error:
        STATE["own_file_error"] = repr(error)[:300]
    table = torch.empty(spec["rows"], spec["cols"], dtype=torch.bfloat16)
    fill(table, spec, model_dir, workers=workers)
    return table, "anon", False


def done_with(base, local, ranks):
    """This rank no longer needs the name; the last of the ranks unlinks it (and the markers)."""
    Path(f"{base}.done{local}").touch()
    if all(Path(f"{base}.done{r}").exists() for r in range(ranks)):
        for path in [Path(base), Path(f"{base}.failed")] + [Path(f"{base}.done{r}") for r in range(ranks)]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def obtain(spec, model_dir, environ=None, workers=8, poll=0.5):
    """(table, row): the build (local rank 0) or the wait and map (the others); either falls back to its own."""
    environ = os.environ if environ is None else environ
    directory = environ.get(DIR_ENV) or "/dev/shm"
    ranks, local = int(environ.get("LOCAL_WORLD_SIZE") or 1), int(environ.get("LOCAL_RANK") or 0)
    base = Path(directory) / launch_name(environ)
    row = {"path": str(base), "ranks": ranks, "local_rank": local, "gib": round(spec["bytes"] / 2 ** 30, 2)}
    began = time.time()
    table = None
    if local == 0:
        row["stale_removed"] = remove_stale(directory, base.name)
        try:
            build_file(base, spec, model_dir, workers=workers)
            table, row["source"] = map_read_only(base, spec), "built"
        except Exception as error:  # noqa: BLE001 - the others build their own; so does this rank
            row["error"] = repr(error)[:500]
            Path(f"{base}.failed").write_text(row["error"])
    else:
        deadline = began + float(environ.get(WAIT_ENV) or 1800)
        while time.time() < deadline and not base.exists() and not Path(f"{base}.failed").exists():
            time.sleep(poll)
        if base.exists():
            try:
                table, row["source"] = map_read_only(base, spec), "mapped"
            except Exception as error:  # noqa: BLE001
                row["error"] = repr(error)[:500]
        else:
            row["error"] = "builder failed" if Path(f"{base}.failed").exists() else "wait timed out"
    row["wait_or_build_s"] = round(time.time() - began, 1)
    if ranks > 1:
        done_with(base, local, ranks)
    else:
        base.unlink(missing_ok=True)
    read_only = table is not None
    if table is None:
        at = time.time()
        table, row["source"], read_only = own(spec, model_dir, directory, workers=workers)
        row["own_s"] = round(time.time() - at, 1)
    row["read_only"] = read_only
    return table, row


def sample_digest(table, blocks=256, rows=64):
    """sha256 over ``blocks`` evenly spaced blocks of ``rows`` rows (and the shape): cheap, equal across ranks."""
    import hashlib

    import torch

    digest = hashlib.sha256(f"{list(table.shape)}{table.dtype}".encode())
    total = table.shape[0]
    for index in range(blocks):
        at = min(total - rows, index * max(1, (total - rows) // max(1, blocks - 1))) if total > rows else 0
        digest.update(table[at:at + rows].contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()[:16]


def start(model_dir, environ=None, workers=8):
    """At trainer start (before the load): the plan now (off on a checkpoint without the sharded table), the build or
    wait on a thread beside the load. Returns whether the table is shared (then ``fq.load(table_meta=True)``)."""
    environ = os.environ if environ is None else environ
    STATE.clear()
    STATE["on"] = False
    if not wanted(environ):
        return False
    try:
        spec = plan(model_dir)
    except (OSError, ValueError, KeyError) as error:
        STATE["off_reason"] = repr(error)[:300]
        print(json.dumps({"event": "shared_table_off", "reason": STATE["off_reason"]}), flush=True)
        return False
    STATE.update(on=True, plan=spec, began=time.time())

    def run():
        try:
            STATE["table"], STATE["row"] = obtain(spec, model_dir, environ, workers=workers)
        except BaseException as error:  # noqa: BLE001 - raised at attach
            STATE["error"] = error

    STATE["thread"] = threading.Thread(target=run, name="shared-table", daemon=True)
    STATE["thread"].start()
    return True


def meta_loading():
    """transformers 5.x: the table's shards become empty meta tensors at materialisation (its ``force_cpu`` converter
    would otherwise read and cast all 128 shards and concatenate them on the host), and the concatenated table stays
    on meta. Installed after the load's other ``get_device`` patches."""
    import torch
    import transformers.core_model_loading as loading

    get_device, spawn = loading.get_device, loading.spawn_materialize
    if getattr(get_device, "table_meta", False):
        return

    def table_device(device_map, name, *args, **kwargs):
        if "ngram_embedding." in name:
            if kwargs.get("valid_torch_device") or (args and args[0]):  # the materialisation's placement call
                PENDING["key"] = name
            return "meta"
        PENDING.pop("key", None)
        return get_device(device_map, name, *args, **kwargs)

    def table_spawn(thread_pool, tensor, device=None, dtype=None, *args, **kwargs):
        key = PENDING.pop("key", None)
        if key is None or not hasattr(tensor, "get_shape"):
            return spawn(thread_pool, tensor, device, dtype, *args, **kwargs)
        kind = getattr(torch, ST_DTYPES.get(tensor.get_dtype(), "bfloat16"))
        value = torch.empty(tensor.get_shape(), dtype=dtype or kind, device="meta")
        PENDING["shards"] = PENDING.get("shards", 0) + 1
        if thread_pool is None:
            return lambda: value
        from concurrent.futures import Future

        future = Future()
        future.set_result(value)
        return future

    for name in ("host_preload", "table_in_host_ram"):
        setattr(table_device, name, getattr(get_device, name, False))
    table_device.table_meta = True
    loading.get_device, loading.spawn_materialize = table_device, table_spawn


def attach(rec, model):
    """The shared (or own) table as the model's ``ngram_embedding.weight`` (frozen); checks that the table is the only
    tensor the load left on meta. Returns the event row."""
    import torch

    thread = STATE.get("thread")
    waited = time.time()
    if thread is not None:
        thread.join()
    if STATE.get("error") is not None:
        raise RuntimeError(f"shared table: {STATE['error']!r}") from STATE["error"]
    spec, table = STATE["plan"], STATE["table"]
    found = [(n, m) for n, m in model.named_modules()
             if isinstance(getattr(m, "weight", None), torch.Tensor) and n.endswith("ngram_embedding")]
    if len(found) != 1:
        raise RuntimeError(f"shared table: {len(found)} ngram_embedding modules")
    name, module = found[0]
    old = module.weight
    if tuple(old.shape) != tuple(table.shape) or old.dtype != table.dtype:
        raise RuntimeError(f"shared table: model {tuple(old.shape)} {old.dtype}, built {tuple(table.shape)} "
                           f"{table.dtype}")
    loaded_by_transformers = old.device.type != "meta"
    module.weight = torch.nn.Parameter(table, requires_grad=False)
    del old
    left = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.device.type == "meta"]
    if left:
        raise RuntimeError(f"shared table: still on meta after the attach: {left[:5]}")
    STATE["ptr"] = table.data_ptr()
    row = {**STATE.get("row", {}), "param": name + ".weight", "shards_skipped": PENDING.get("shards", 0),
           "loaded_by_transformers": loaded_by_transformers, "join_wait_s": round(time.time() - waited, 1),
           "since_start_s": round(time.time() - STATE["began"], 1), "digest": sample_digest(table)}
    if STATE.get("own_file_error"):
        row["own_file_error"] = STATE["own_file_error"]
    if rec is not None:
        rec.event("shared_table", **row)
    return row


def guard(model, optimizer=None, where=""):
    """The attached table is still the mapping, frozen, and in no optimizer group (a replaced ``.data`` or a trainable
    table would silently diverge from the other ranks)."""
    if model is None or not STATE.get("on") or "ptr" not in STATE:
        return None
    tables = [(n, p) for n, p in model.named_parameters() if n.endswith("ngram_embedding.weight")]
    if len(tables) != 1:
        raise RuntimeError(f"shared table ({where}): {len(tables)} table parameters")
    name, param = tables[0]
    problems = []
    if param.requires_grad:
        problems.append("requires_grad")
    if param.data_ptr() != STATE["ptr"]:
        problems.append("no longer the shared mapping")
    if optimizer is not None and any(p is param for group in optimizer.param_groups for p in group["params"]):
        problems.append("in an optimizer group")
    if problems:
        raise RuntimeError(f"shared table ({where}): {name}: " + ", ".join(problems))
    return {"param": name, "ok": True}
