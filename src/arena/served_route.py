"""Reader side of the served-routing capture (R3, ``arc_infra.served_route_runtime`` writes it).

A server request's ``<engine id>.rt`` holds the positions that request computed: per position the token fed and the
top-k experts of every MoE layer. Positions it took from the prefix cache were computed by an earlier request, so a
window's full routing is assembled from several records:

1. the end request's own record;
2. earlier requests of the same append-only chain (same try, no drain between; latest first), on positions where
   their fed token equals the window's token, up to the first position where they disagree;
3. requests computed from scratch (record starts at 0) with the same LoRA, on the common token prefix, as long as
   every image starting in that prefix has the same offset, length and content hash as the end request's;
4. any other request whose record starts at s > 0 after the same context: its ``prefix_hash`` (of the tokens before
   s) equals the window's, same LoRA and images before s (e.g. tries sharing a restored start: the first one computed
   the long prefix after a short system-prompt cache hit, so its record starts there, not at 0).

Positions none of them covers (e.g. the last sampled token, which is never fed) keep the trainer's own routing.
"""

import json
import os
import time
from pathlib import Path

import numpy as np

HEAD = 4


def read(path):
    """``{"lo", "hi", "tokens" int32[hi] (-1 not covered), "ids" int16[hi, L, k] (-1), "covered" bool[hi]}``; a
    later record for a position wins."""
    raw = Path(path).read_bytes()
    records, at = [], 0
    while at + 16 <= len(raw):
        start, n, layers, k = (int(v) for v in np.frombuffer(raw, dtype=np.int32, count=HEAD, offset=at))
        at += 16
        tokens = np.frombuffer(raw, dtype=np.int32, count=n, offset=at)
        at += 4 * n
        size = n * layers * k
        if at + 2 * size > len(raw):
            break  # a record still being written
        ids = np.frombuffer(raw, dtype=np.int16, count=size, offset=at).reshape(n, layers, k)
        at += 2 * size
        records.append((start, tokens, ids))
    if not records:
        return None
    layers, k = records[0][2].shape[1:]
    hi = max(s + len(t) for s, t, _ in records)
    out = {"lo": min(s for s, _, _ in records), "hi": hi, "tokens": np.full(hi, -1, dtype=np.int32),
           "ids": np.full((hi, layers, k), -1, dtype=np.int16), "covered": np.zeros(hi, dtype=bool)}
    for start, tokens, ids in records:
        out["tokens"][start:start + len(tokens)] = tokens
        out["ids"][start:start + len(tokens)] = ids
        out["covered"][start:start + len(tokens)] = True
    return out



def mask_hits(directory, mask, max_files=None):
    """Routed ids in ``directory``'s .rt files that fall in ``mask``'s pruned experts (an expert_mask json: "pruned"
    {layer: [expert]}); columns map to layers by layers.json. A masked trainer replaying routes with hits dies on
    "expert compaction: a pruned expert was routed" (rl1-prof-m1v1: ovf3 routes were served unmasked, 8% hits)."""
    directory = Path(directory)
    layers = json.loads((directory / "layers.json").read_text())["layers"]
    pruned = {int(k): np.asarray(v, dtype=np.int16) for k, v in mask["pruned"].items()}
    files = sorted(directory.glob("*.rt"))[: max_files or None]
    out = {"files": len(files), "routed": 0, "hits": 0, "by_layer": {}}
    for path in files:
        got = read(path)
        if got is None:
            continue
        ids = got["ids"][got["covered"]]
        for col, layer in enumerate(layers[: ids.shape[1]]):
            x = ids[:, col, :]
            x = x[x >= 0]
            out["routed"] += int(x.size)
            if int(layer) in pruned:
                n = int(np.isin(x, pruned[int(layer)]).sum())
                if n:
                    out["hits"] += n
                    out["by_layer"][int(layer)] = out["by_layer"].get(int(layer), 0) + n
    return out

KEY = 32  # tokens that index the from-scratch requests
HASH_BASE = 0x9E3779B97F4A7C15  # same as arc_infra.served_route_runtime (this file is copied alone into rl1/)


def prefix_hashes(tokens):
    """``H`` (uint64 [T + 1]): H[0] = 0, H[i + 1] = H[i] * B + tokens[i] + 1 mod 2^64 (served_route_runtime's)."""
    t = np.asarray(tokens, dtype=np.int64).astype(np.uint64) + np.uint64(1)
    with np.errstate(over="ignore"):
        up = np.cumprod(np.full(len(t), HASH_BASE, dtype=np.uint64))
        down = np.cumprod(np.full(len(t), pow(HASH_BASE, -1, 1 << 64), dtype=np.uint64))
        out = np.zeros(len(t) + 1, dtype=np.uint64)
        out[1:] = up * np.cumsum(t * down, dtype=np.uint64)
    return out


class Store:
    """One served-route directory: engine request files by the API id they start with (``chatcmpl-...``);
    ``scratch``: also index the requests whose first record starts at position 0 by their first ``KEY`` tokens."""

    def __init__(self, directory, scratch=False, cache=256):
        self.dir = Path(directory)
        self.files, self.index, self.cache, self.limit, self.heads = {}, {}, {}, cache, {}
        self.by_hash = {}  # (record start, prefix hash) -> API ids
        for name in os.listdir(self.dir):
            if name.endswith(".rt"):
                api = name[:-3].rsplit("-", 1)[0]  # the engine id is the API id plus "-" and 8 random characters
                self.files.setdefault(api, []).append(name[:-3])
                if scratch:
                    with open(self.dir / name, "rb") as f:
                        head = f.read(16 + 4 * KEY)
                    if len(head) >= 16:
                        start, n = (int(v) for v in np.frombuffer(head, dtype=np.int32, count=2))
                        if start == 0 and n >= KEY and len(head) == 16 + 4 * KEY:
                            key = np.frombuffer(head, dtype=np.int32, count=KEY, offset=16).tobytes()
                            self.index.setdefault(key, []).append(api)
                        elif start > 0:
                            meta = self.dir / f"{name[:-3]}.mm.json"
                            try:
                                value = json.loads(meta.read_text()).get("prefix_hash")
                            except (OSError, ValueError):
                                value = None
                            if value:
                                self.by_hash.setdefault((start, int(value, 16)), []).append(api)
        meta = self.dir / "layers.json"
        self.layers = json.loads(meta.read_text())["layers"] if meta.exists() else None

    def meta(self, rid):
        name = self.engine_id(rid)
        path = self.dir / f"{name}.mm.json" if name else None
        return json.loads(path.read_text()) if path is not None and path.exists() else None

    def head(self, rid):
        """(start, fed tokens) of the request's first record, read without its routing ids (donor ranking)."""
        if rid in self.heads:
            return self.heads[rid]
        name = self.engine_id(rid)
        out = None
        if name is not None:
            try:
                with open(self.dir / f"{name}.rt", "rb") as f:
                    start, n = (int(v) for v in np.frombuffer(f.read(16), dtype=np.int32, count=2))
                    out = (start, np.frombuffer(f.read(4 * n), dtype=np.int32))
            except (OSError, ValueError):
                out = None
        if len(self.heads) >= 8 * self.limit:
            self.heads.pop(next(iter(self.heads)))
        self.heads[rid] = out
        return out

    def match(self, rid, target):
        """Positions past the record's start where its fed tokens equal the target's (0 without a record)."""
        head = self.head(rid)
        if head is None:
            return 0
        start, tokens = head
        top = min(len(target), start + len(tokens))
        if top <= start:
            return 0
        diff = np.nonzero(tokens[:top - start] != target[start:top])[0]
        return int(diff[0]) if diff.size else top - start

    def scratch_for(self, own, target, keep=2):
        """Up to ``keep`` from-scratch records sharing the target's first tokens: the longest usable prefix (token
        match, cut at the first image the two requests do not share) first (21 Oct: ranking by images alone kept
        arbitrary ones)."""
        target = np.asarray(target, dtype=np.int32)
        if len(target) < KEY or own.get("meta") is None:
            return []
        ranked = []
        for rid in self.index.get(target[:KEY].tobytes(), []):
            meta = self.meta(rid)
            if meta is None or meta.get("lora") != own["meta"].get("lora"):
                continue
            ranked.append((min(self.match(rid, target), image_cut(meta, own["meta"], len(target))), rid))
        ranked.sort(reverse=True)
        return [r for r in (self.record(rid) for _, rid in ranked[:keep]) if r is not None]

    def donors(self, own, target, keep=2):
        """Records that start at s > 0 after the window's own context (prefix hash of ``target[:s]``), same LoRA and
        images before s; up to ``keep`` per start, earliest start first."""
        target = np.asarray(target, dtype=np.int32)
        meta = own.get("meta")
        if meta is None or not self.by_hash:
            return []
        hashes = prefix_hashes(target)
        out = []
        for start in sorted({s for s, _ in self.by_hash if s < len(target)}):
            # 21 Oct: many records share a short context (the system prompt's cache hit); rank them by how far their
            # fed tokens match the target past the start (q2r-v2 took the first two: 53-67% of window tokens resolved)
            ranked = sorted(((self.match(rid, target), rid) for rid in self.by_hash.get((start, int(hashes[start])), [])),
                            reverse=True)
            found = 0
            for length, rid in ranked:
                if length <= 0 or found >= keep:
                    break
                rec = self.record(rid)
                if rec is None or rec["lo"] != start or rec.get("meta", {}).get("lora") != meta.get("lora"):
                    continue
                if images_before(rec.get("meta"), start) != images_before(meta, start):
                    continue
                out.append(rec)
                found += 1
        return out

    def engine_id(self, rid):
        names = self.files.get(rid) or []
        return names[0] if len(names) == 1 else None

    def record(self, rid):
        """The request's record and metadata, or None."""
        if rid in self.cache:
            return self.cache[rid]
        name = self.engine_id(rid)
        out = None
        if name is not None:
            out = read(self.dir / f"{name}.rt")
            meta = self.dir / f"{name}.mm.json"
            if out is not None:
                out["meta"] = json.loads(meta.read_text()) if meta.exists() else {}
        if len(self.cache) >= self.limit:
            self.cache.pop(next(iter(self.cache)))
        self.cache[rid] = out
        return out


def first_mismatch(a, b, lo, hi):
    """First index in [lo, hi) where a and b differ, else hi."""
    if hi <= lo:
        return lo
    diff = np.nonzero(a[lo:hi] != b[lo:hi])[0]
    return lo + int(diff[0]) if diff.size else hi


def images_before(meta, stop):
    return sorted(tuple(m) for m in (meta or {}).get("mm", []) if m[0] < stop)


def image_cut(meta_a, meta_b, stop):
    """``stop`` lowered to the first image (offset, length, content hash) that the two requests do not share."""
    a, b = images_before(meta_a, stop), images_before(meta_b, stop)
    for x, y in zip(a, b):
        if x != y:
            return min(x[0], y[0])
    if len(a) != len(b):
        return (a if len(a) > len(b) else b)[min(len(a), len(b))][0]
    return stop


def resolve(target, own, chain=(), scratch=(), donors=()):
    """The routing for a window whose token ids are ``target`` (int array [T]): ``(ids int16 [T, L, k] with -1 where
    unresolved, audit)``. ``own``: the end request's record; ``chain``: records of earlier requests of the same
    append-only chain, latest first; ``scratch``: records computed from position 0 (other tries of the round)."""
    target = np.asarray(target, dtype=np.int32)
    total = len(target)
    layers, k = own["ids"].shape[1:]
    out = np.full((total, layers, k), -1, dtype=np.int16)
    done = np.zeros(total, dtype=bool)
    audit = {"tokens": total, "own": 0, "own_mismatch": 0, "chain": 0, "scratch": 0, "donor": 0}

    def take(rec, lo, hi, key):
        hi = min(hi, total, rec["hi"])
        if hi <= lo:
            return
        sel = np.zeros(total, dtype=bool)
        sel[lo:hi] = rec["covered"][lo:hi] & ~done[lo:hi] & (rec["tokens"][lo:hi] == target[lo:hi])
        out[sel] = rec["ids"][np.nonzero(sel)[0]]
        done[sel] = True
        audit[key] += int(sel.sum())

    hi = min(total, own["hi"])
    audit["own_mismatch"] = int((own["covered"][:hi] & (own["tokens"][:hi] != target[:hi])).sum())
    take(own, 0, hi, "own")
    for rec in chain:
        if rec is None:
            continue
        top = min(total, rec["hi"])
        stop = first_mismatch(rec["tokens"], target, rec["lo"], top)
        take(rec, rec["lo"], stop, "chain")
    own_meta = own.get("meta")
    for rec in scratch if own_meta is not None else ():
        if rec is None or rec["lo"] != 0 or rec.get("meta") is None or done.all():
            continue
        if rec["meta"].get("lora") != own_meta.get("lora"):
            continue
        top = min(total, rec["hi"])
        stop = image_cut(rec["meta"], own_meta, first_mismatch(rec["tokens"], target, 0, top))
        take(rec, 0, stop, "scratch")
    for rec in donors if own_meta is not None else ():  # same context before rec["lo"] (Store.donors checked it)
        if rec is None or rec.get("meta") is None or done.all():
            continue
        top = min(total, rec["hi"])
        stop = image_cut(rec["meta"], own_meta, first_mismatch(rec["tokens"], target, rec["lo"], top))
        take(rec, rec["lo"], stop, "donor")
    audit["unresolved"] = int(total - done.sum())
    return out, audit


def prune(directory, before):
    """Delete request files last written before ``before`` (epoch seconds): ``{"files", "bytes", "kept"}``."""
    out = {"files": 0, "bytes": 0, "kept": 0}
    for path in Path(directory).iterdir():
        if path.name in ("layers.json", "status.json") or not path.is_file():
            continue
        stat = path.stat()
        if stat.st_mtime < before:
            path.unlink(missing_ok=True)
            out["files"] += 1
            out["bytes"] += stat.st_size
        else:
            out["kept"] += 1
    return out


def status(directory):
    """The capture's live state: request files on disk and the server side's ``status.json`` counters."""
    out = {"rt": 0, "bytes": 0}
    path = Path(directory)
    if not path.is_dir():
        return {**out, "missing": True}
    for item in path.iterdir():
        if item.name.endswith(".rt"):
            out["rt"] += 1
            out["bytes"] += item.stat().st_size
    try:
        out["server"] = json.loads((path / "status.json").read_text())
    except (OSError, ValueError):
        out["server"] = None
    return out


SIDE = ("layers.json", "status.json")


def mirror(src, dst, seen, settle=30.0):
    """(c) persistence, 22 Oct: the capture directory's finished request files (``.rt`` / ``.mm.json`` unchanged for
    ``settle`` s) go to ``dst`` on the runs volume, so a restart's first update and its reused tries train on served
    routes (r2-r4: every resume's first update had none). One uncompressed tar per pass (23 Oct: per-request files
    took ~14K volume inodes a round on a 500K-inode volume), its mtime the newest member's, so ``prune`` cuts it
    only once every member is older than the mark. ``seen``: {name: mtime} already mirrored (updated).
    Returns {"files", "bytes"} (members)."""
    import shutil
    import tarfile

    src, dst, out, now = Path(src), Path(dst), {"files": 0, "bytes": 0}, time.time()
    if not src.is_dir():
        return out
    dst.mkdir(parents=True, exist_ok=True)
    batch = []
    for entry in os.scandir(src):
        if entry.name in SIDE or not entry.is_file() or not entry.name.endswith((".rt", ".mm.json")):
            continue
        st = entry.stat()
        if now - st.st_mtime < settle or seen.get(entry.name) == st.st_mtime:
            continue
        batch.append((entry.path, entry.name, st))
    if batch:
        name = f"pass-{int(now * 1000)}.tar"
        part = dst / (name + ".part")
        with tarfile.open(part, "w") as tar:
            for path, member, _ in batch:
                tar.add(path, arcname=member)
        newest = max(st.st_mtime for _, _, st in batch)
        os.utime(part, (newest, newest))
        os.replace(part, dst / name)  # a reader (restore) never sees a half-written tar
        for _, member, st in batch:
            seen[member] = st.st_mtime
            out["files"] += 1
            out["bytes"] += st.st_size
    if not (dst / "layers.json").exists() and (src / "layers.json").exists():
        shutil.copy2(src / "layers.json", dst / "layers.json")
    return out


def restore(roots, dst):
    """Copy every root's mirrored request files missing from ``dst`` (the live capture directory; the latest root's
    copy of a name wins, roots in resume order): loose files (mirrors before 23 Oct) and ``pass-*.tar`` members.
    {"files", "bytes", "roots"}. ``layers.json`` only when ``dst`` has none."""
    import shutil
    import tarfile

    dst, out = Path(dst), {"files": 0, "bytes": 0, "roots": 0}
    dst.mkdir(parents=True, exist_ok=True)
    have = set(os.listdir(dst))
    for root in reversed([Path(r) for r in roots]):
        if not root.is_dir():
            continue
        out["roots"] += 1
        tars = []
        for entry in os.scandir(root):
            if not entry.is_file():
                continue
            if entry.name.startswith("pass-") and entry.name.endswith(".tar"):
                tars.append(entry.path)
                continue
            if entry.name in have or entry.name.endswith(".part") or entry.name == "status.json":
                continue
            if entry.name == "layers.json" and (dst / "layers.json").exists():
                continue
            shutil.copy2(entry.path, dst / entry.name)
            have.add(entry.name)
            out["files"] += 1
            out["bytes"] += entry.stat().st_size
        for path in sorted(tars, reverse=True):  # a name in two passes: the later pass's copy
            with tarfile.open(path) as tar:
                for member in tar.getmembers():
                    name = member.name
                    if not member.isfile() or name in have or "/" in name or name.startswith("."):
                        continue
                    tar.extract(member, dst, filter="data")
                    have.add(name)
                    out["files"] += 1
                    out["bytes"] += member.size
    return out

