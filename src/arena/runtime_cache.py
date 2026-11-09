"""MFU-2 boot item (22 Oct): the pinned vLLM runtime tree as the fast-mode extraction leaves it (before any patch),
kept as plain tar parts on the volume and restored in parallel in place of the layer-by-layer gzip extraction (v6d-r2
card 1: 115 s). Opt-in: ``ARC_RUNTIME_CACHE=<dir>`` (modal_rl1, bundle configs/boot.json ``runtime_cache``).

Layout ``<dir>/<key>/``: ``part-00.tar`` (directories and symlinks, extracted first), ``part-01..NN.tar`` (regular
files, balanced by size, extracted in parallel), ``result.json`` (the extraction's own return), ``tree.json`` (the
tree digest the parts were checked against before ``DONE`` was written). The key covers the runtime manifest's hash,
the pinned layer and whiteout tables and the extraction mode; full-hash (non-fast) extractions are never cached.
"""

import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENV = "ARC_RUNTIME_CACHE"
VERSION = 1


def cache_key(recipe, runtime_dir, full_layer_hashes, scan_extracted_caches):
    """None for a mode that is not cached; else the key of this runtime and mode."""
    if full_layer_hashes or scan_extracted_caches:
        return None
    manifest = Path(runtime_dir) / "runtime-manifest.json"
    if not manifest.is_file():
        return None
    row = {"version": VERSION, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
           "expected_manifest": recipe.VLLM_RUNTIME_MANIFEST_SHA256,
           "layers": {str(k): list(v) for k, v in sorted(recipe.EXPECTED_RUNTIME_LAYERS.items())},
           "whiteouts": {str(k): list(v) for k, v in sorted(recipe.FAST_RUNTIME_WHITEOUTS.items())}}
    if row["manifest_sha256"] != row["expected_manifest"]:
        return None
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:24]


def call_args(args, kwargs):
    """(runtime_dir, full_layer_hashes, scan_extracted_caches) of a verify_and_extract_runtime call."""
    runtime_dir = args[0] if args else kwargs["runtime_dir"]
    return Path(runtime_dir), kwargs.get("full_layer_hashes", True), kwargs.get("scan_extracted_caches", True)


def tree_digest(root, threads=16):
    """Digest of a tree: every entry's relative path, type, mode, symlink target and file content."""
    root = Path(root)
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            entries.append(Path(dirpath) / name)
    entries.sort()

    def one(path):
        st = path.lstat()
        rel = str(path.relative_to(root))
        if path.is_symlink():
            return f"L|{rel}|{os.readlink(path)}"
        if path.is_dir():
            return f"D|{rel}|{st.st_mode:o}"
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 22):
                h.update(chunk)
        return f"F|{rel}|{st.st_mode:o}|{st.st_size}|{h.hexdigest()}"

    with ThreadPoolExecutor(threads) as pool:
        lines = list(pool.map(one, entries))
    total = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    return {"entries": len(lines), "files": sum(1 for x in lines if x[0] == "F"), "digest": total}


def _extract_parts(folder, root, workers=8):
    parts = sorted(Path(folder).glob("part-*.tar"))
    first, rest = parts[0], parts[1:]
    tar = ["tar", "-xf", None, "-C", str(root), "--no-same-owner"]
    subprocess.run([*tar[:2], str(first), *tar[3:]], check=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda p: subprocess.run([*tar[:2], str(p), *tar[3:]], check=True), rest))
    return len(parts)


def restore(recipe, cache_dir, args, kwargs):
    """The extraction's return after restoring the cached tree into ``recipe.RUNTIME_ROOT``; None (extract as usual)
    when this mode is not cached or the cache has no finished entry for this runtime."""
    runtime_dir, full, scan = call_args(args, kwargs)
    key = cache_key(recipe, runtime_dir, full, scan)
    folder = Path(cache_dir) / str(key)
    if key is None or not (folder / "DONE").is_file():
        return None
    began = time.monotonic()
    root = Path(recipe.RUNTIME_ROOT)
    try:
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        parts = _extract_parts(folder, root)
        result = json.loads((folder / "result.json").read_text())
    except Exception as exc:  # noqa: BLE001 - a broken cache entry: the usual extraction (it clears the root first)
        print(f"ARC_RUNTIME_CACHE_FAILED {type(exc).__name__}: {exc}"[:2000], flush=True)
        return None
    result["extract_seconds"] = time.monotonic() - began
    result["runtime_cache"] = {"key": key, "parts": parts, "seconds": round(result["extract_seconds"], 1)}
    print("ARC_RUNTIME_CACHE_RESTORED " + json.dumps(result["runtime_cache"]), flush=True)
    return result


def build(recipe, runtime_dir, cache_dir, work="/tmp/arc-runtime-cache", parts=8):
    """Extract the runtime afresh into ``work`` with the recipe's own fast-mode extraction, write the parts, restore
    them into a second tree and write ``DONE`` only when both trees have the same digest. Returns a record."""
    key = cache_key(recipe, runtime_dir, False, False)
    if key is None:
        return {"built": False, "why": "runtime manifest not the pinned one"}
    folder = Path(cache_dir) / key
    if (folder / "DONE").is_file():
        return {"built": False, "why": "exists", "key": key}
    began = time.time()
    work = Path(work)
    shutil.rmtree(work, ignore_errors=True)
    built, check = work / "tree", work / "check"
    saved_root = recipe.RUNTIME_ROOT
    recipe.RUNTIME_ROOT = built  # the extraction and its whiteout pass both read this module global
    try:
        result = recipe.verify_and_extract_runtime(Path(runtime_dir), full_layer_hashes=False,
                                                   scan_extracted_caches=False)
    finally:
        recipe.RUNTIME_ROOT = saved_root
    result.pop("extract_seconds", None)
    dirs, files = [], []
    for dirpath, dirnames, filenames in os.walk(built):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            rel = str(path.relative_to(built))
            if path.is_symlink() or path.is_dir():
                dirs.append(rel)
            else:
                files.append((path.lstat().st_size, rel))
    groups = [[] for _ in range(parts)]
    loads = [0] * parts
    for size, rel in sorted(files, reverse=True):
        i = loads.index(min(loads))
        groups[i].append(rel)
        loads[i] += size
    tmp = folder.with_name(f"{key}.{os.getpid()}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    lists = []
    for i, names in enumerate([sorted(dirs)] + groups):
        listing = work / f"list-{i:02d}.txt"
        listing.write_text("\n".join(names) + ("\n" if names else ""))
        lists.append((i, listing))

    def pack(item):  # tarfile: hard links stay links only inside one part (its inode table is per archive)
        import tarfile

        i, listing = item
        with tarfile.open(tmp / f"part-{i:02d}.tar", "w", format=tarfile.GNU_FORMAT, bufsize=1 << 20) as tar:
            for rel in filter(None, listing.read_text().split("\n")):
                tar.add(built / rel, arcname=rel, recursive=False)

    with ThreadPoolExecutor(parts) as pool:
        list(pool.map(pack, lists))
    (tmp / "result.json").write_text(json.dumps(result, indent=1))
    check.mkdir(parents=True)
    _extract_parts(tmp, check)
    want, got = tree_digest(built), tree_digest(check)
    record = {"key": key, "built_tree": want, "restored_tree": got, "bytes": sum(loads),
              "seconds": round(time.time() - began, 1)}
    (tmp / "tree.json").write_text(json.dumps(record, indent=1))
    if want != got:
        record["built"] = False
        record["why"] = "restored tree differs"
        return record
    (tmp / "DONE").write_text(json.dumps({"epoch": time.time()}))
    if folder.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        os.replace(tmp, folder)
    shutil.rmtree(work, ignore_errors=True)
    record["built"] = True
    return record


def pinned_recipe(input_root="/kaggle/input"):
    """(a private copy of the serving script the arena runs, arc_infra/serving_setup.py, the fork serving_core stages
    and loads; its runtime dataset dir). Its import-time paths (bundle, working dir, setup env) point at a scratch dir
    for the import only: the build uses only the extraction."""
    import importlib.util
    import tempfile

    from arc_infra.serving import RUNTIME_DATASET, dataset_mount

    path = Path(__file__).resolve().parents[1] / "arc_infra" / "serving_setup.py"
    scratch = Path(tempfile.mkdtemp(prefix="arc-rtcache-"))
    names = ("TAAF_KAGGLE_BUNDLE_DIR", "TAAF_KAGGLE_WORKING_DIR", "TAAF_KAGGLE_SETUP_ENV")
    saved = {n: os.environ.get(n) for n in names}
    os.environ.update({n: str(scratch / n.lower()) for n in names if saved[n] is None})
    try:
        spec = importlib.util.spec_from_file_location("arc_rtcache_serving_setup", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for n, v in saved.items():
            if v is None:
                os.environ.pop(n, None)
    return module, dataset_mount(RUNTIME_DATASET, Path(input_root))
