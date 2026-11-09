"""Per-step weight deltas for the serving student (PTH streaming OPD, 22 Oct).

After each optimizer step the trainer writes what the RTX server needs to serve the new weights in place
(``arc_heal_reload``): every trained dense tensor in BF16, the trained (ESFT-selected) routed experts as AutoRound
int4 (``int4_rtn``, heal_export's rule), the full 512-row routers (kept rows from the student; pruned rows zero, the
server's prune filter drops them), all under the served checkpoint's names (uncut layer and expert numbering). The
MTP head is not synced (a draft never changes what is sampled). Files go to ``<root>/v<step>/`` with ``manifest.json``
written last; an uploader thread copies finished versions to the bucket and keeps the newest few locally.
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import torch

LAYER = re.compile(r"^(model\.language_model\.layers\.)(\d+)(\..*)$")
FUSED = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
ROUTER = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")
FROZEN = (".ple.", "model.visual.", "model.language_model.embed_tokens", "lm_head")


def int4_rtn(w, group=128, zero=8, search=20):
    """heal_export.int4_rtn on ``w``'s own device: symmetric int4 [out, in] -> AutoRound GPTQ (qweight [in/8, out],
    qzeros [in/group, out/8] holding zero - 1, fp16 scales [in/group, out]); ``search`` shrunken scales per group."""
    dev = w.device
    wt = w.float().t().contiguous()
    inp, out = wt.shape
    g = wt.reshape(inp // group, group, out)
    idx = g.abs().argmax(1, keepdim=True)
    peak = g.gather(1, idx).squeeze(1)
    base = torch.where(peak == 0, torch.full_like(peak, 1e-10), -peak / zero)

    def quant(scale):
        q = (g / scale.unsqueeze(1)).round().clamp(-zero, 15 - zero)
        return q, ((q * scale.unsqueeze(1) - g) ** 2).sum(1)

    scale = base
    q, err = quant(scale)
    for f in torch.linspace(1.0, 0.8, search + 1)[1:].tolist() if search else []:
        q2, e2 = quant(base * f)
        better = e2 < err
        scale, err = torch.where(better, base * f, scale), torch.where(better, e2, err)
    scale16 = scale.to(torch.float16)
    q = (g / scale16.float().unsqueeze(1)).round().clamp(-zero, 15 - zero)
    codes = (q + zero).reshape(inp, out).to(torch.int32)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=dev)
    qweight = (codes.reshape(inp // 8, 8, out) << shifts.view(1, -1, 1)).sum(1, dtype=torch.int32)
    z = torch.full((inp // group, out), zero - 1, dtype=torch.int32, device=dev)
    qzeros = (z.reshape(inp // group, out // 8, 8) << shifts.view(1, 1, -1)).sum(2, dtype=torch.int32)
    return qweight, qzeros, scale16


def int4_rtn_rows(ws, group=128, zero=8, search=20, chunk=32):
    """``int4_rtn`` on every row of ``ws`` [E, out, in] at once (MFU-1, HEAL_DELTA_FAST): [E, ...] qweight, qzeros,
    scales, each the per-row result (``chunk`` rows per pass: transient memory)."""
    outs = [[], [], []]
    for a in range(0, ws.shape[0], chunk):
        w = ws[a:a + chunk]
        dev, e = w.device, w.shape[0]
        wt = w.float().transpose(1, 2).contiguous()
        inp, out = wt.shape[1:]
        g = wt.reshape(e, inp // group, group, out)
        idx = g.abs().argmax(2, keepdim=True)
        peak = g.gather(2, idx).squeeze(2)
        base = torch.where(peak == 0, torch.full_like(peak, 1e-10), -peak / zero)

        def quant(scale):
            q = (g / scale.unsqueeze(2)).round().clamp(-zero, 15 - zero)
            return q, ((q * scale.unsqueeze(2) - g) ** 2).sum(2)

        scale = base
        q, err = quant(scale)
        for f in torch.linspace(1.0, 0.8, search + 1)[1:].tolist() if search else []:
            q2, e2 = quant(base * f)
            better = e2 < err
            scale, err = torch.where(better, base * f, scale), torch.where(better, e2, err)
        scale16 = scale.to(torch.float16)
        q = (g / scale16.float().unsqueeze(2)).round().clamp(-zero, 15 - zero)
        codes = (q + zero).reshape(e, inp, out).to(torch.int32)
        shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=dev)
        outs[0].append((codes.reshape(e, inp // 8, 8, out) << shifts.view(1, 1, -1, 1)).sum(2, dtype=torch.int32))
        z = torch.full((e, inp // group, out), zero - 1, dtype=torch.int32, device=dev)
        outs[1].append((z.reshape(e, inp // group, out // 8, 8) << shifts.view(1, 1, 1, -1)).sum(3, dtype=torch.int32))
        outs[2].append(scale16)
    return [torch.cat(x) for x in outs]


def delta_fast():
    return os.environ.get("HEAL_DELTA_FAST") == "1"


def skipped(name):
    """Names ``entries`` writes nothing for (MTP, frozen): the fast export does not gather them."""
    return name.startswith("mtp.") or any(f in name for f in FROZEN)


class Cut:
    """The student's cut: served (uncut) layer of each student layer, original expert ids of each served layer."""

    def __init__(self, student_config, prune_file=None):
        cfg = json.loads(Path(student_config).read_text()) if not isinstance(student_config, dict) else student_config
        cut = cfg["student_cut"]
        self.keep = list(cut["source_layers"])
        self.experts = None
        path = prune_file or cut.get("prune_file")
        if path:
            path = Path(path)
            if not path.is_file():  # the cut's recorded path is the cutting container's: look next to this file
                path = Path(__file__).with_name(path.name if path.name.startswith("heal_") else f"heal_{path.name}")
            spec = json.loads(path.read_text())
            self.n_experts = int(spec["experts"])
            self.experts = {int(layer): [e for e in range(spec["experts"]) if e not in set(pruned)]
                            for layer, pruned in spec["pruned"].items()}

    def served(self, name):
        """(served name prefix, old layer) of a student layer tensor name, else (name, None)."""
        m = LAYER.match(name)
        if not m:
            return name, None
        old = self.keep[int(m.group(2))]
        return f"{m.group(1)}{old}{m.group(3)}", old


def entries(name, full, cut, rows=None, search=20, picked=False):
    """(served name, host tensor) pairs of one gathered trained tensor; ``rows``: the expert rows that train (None:
    every row) for fused expert tensors. Empty for frozen and MTP tensors. ``picked``: ``full`` holds only ``rows``
    (in that order). HEAL_DELTA_FAST: a fused tensor's rows quantized in one batched pass, one copy to the host per
    output (bit-identical files)."""
    if name.startswith("mtp.") or any(f in name for f in FROZEN):
        return []
    fu, ro = FUSED.match(name), ROUTER.match(name)
    served, old = cut.served(name)
    if fu and delta_fast():
        base = served.rsplit(".mlp.experts.", 1)[0]
        ids = list(range(full.shape[0])) if rows is None else list(rows)
        if not ids:
            return []
        sub = full if picked or rows is None else full[torch.tensor(ids, device=full.device)]
        if fu.group(2) == "down_proj":
            parts = (("down_proj", sub),)
        else:
            half = sub.shape[1] // 2
            parts = (("gate_proj", sub[:, :half]), ("up_proj", sub[:, half:]))
        host = {}
        for kind, w in parts:
            qw, qz, sc = int4_rtn_rows(w, group=min(128, w.shape[2]), search=search)
            host[kind] = (qw.cpu(), qz.cpu(), sc.cpu())
        out = []
        for j, e in enumerate(ids):
            orig = cut.experts[old][e] if cut.experts else e
            key = f"{base}.mlp.experts.{orig}"
            for kind, _ in parts:
                qw, qz, sc = host[kind]
                out += [(f"{key}.{kind}.qweight", qw[j].clone()), (f"{key}.{kind}.qzeros", qz[j].clone()),
                        (f"{key}.{kind}.scales", sc[j].clone())]
        return out
    if fu:
        if picked:
            raise ValueError("picked rows need HEAL_DELTA_FAST")
        base = served.rsplit(".mlp.experts.", 1)[0]
        out = []
        for e in (range(full.shape[0]) if rows is None else rows):
            orig = cut.experts[old][e] if cut.experts else e
            key = f"{base}.mlp.experts.{orig}"
            if fu.group(2) == "down_proj":
                mats = ((f"{key}.down_proj", full[e]),)
            else:
                half = full.shape[1] // 2
                mats = ((f"{key}.gate_proj", full[e][:half]), (f"{key}.up_proj", full[e][half:]))
            for k, w in mats:
                qw, qz, sc = int4_rtn(w, group=min(128, w.shape[1]), search=search)  # 128 (served); tiny models: less
                out += [(f"{k}.qweight", qw.cpu()), (f"{k}.qzeros", qz.cpu()), (f"{k}.scales", sc.cpu())]
        return out
    if ro and cut.experts:
        row = torch.zeros(cut.n_experts, full.shape[1], dtype=torch.bfloat16)
        row[torch.tensor(cut.experts[old])] = full.to(torch.bfloat16).cpu()
        return [(served, row)]
    return [(served, full.to(torch.bfloat16).cpu().contiguous())]


def applied(path, generation, timeout=60):
    """The play's weights-follow.jsonl (rclone path) has an 'applied' row of the generation."""
    r = subprocess.run(["rclone", "cat", path], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        return False
    for line in r.stdout.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("kind") == "applied" and row.get("generation") == generation:
            return True
    return False


def play_confirm(path, timeout=60):
    """``confirm`` for DeltaExporter: one play's weights-follow.jsonl."""
    return lambda generation: applied(path, generation, timeout)


def live_confirm(listfile, initial=(), timeout=60, check=applied):
    """``confirm`` whose plays are read live (24 Oct): ``listfile`` holds a JSON list of weights-follow paths, rewritten
    by the round's puller from ``<run>/stream/knobs.json``; ``initial`` while it is unreadable. True once every listed
    play has applied the generation. A path confirmed for a generation stays so (no rclone call, no full version on a
    failed read); a new path turns full versions back on until it applies one (``live``: polled after every write)."""
    done = {}

    def confirm(generation):
        try:
            paths = json.loads(Path(listfile).read_text())
        except (OSError, ValueError):
            paths = list(initial)
        ok = True
        for path in paths:
            if done.get(path) != generation:
                if check(path, generation, timeout):
                    done[path] = generation
                else:
                    ok = False
        return ok
    confirm.live = True
    return confirm


class DeltaExporter:
    """``export(runner, rows, step)`` on every rank (collective gathers); rank 0 writes ``<root>/v<step>`` in a
    background thread and an uploader copies it to ``remote`` (rclone ``remote``/v<step>/, manifest last)."""

    def __init__(self, root, cut, rank, *, remote=None, keep=3, files=32, search=20, log=None, generation=None,
                 base="start", confirm=None):
        """``generation`` (default: now in ms) is new on every trainer start or resume; ``base``: "start" (this trainer
        began from the student the play loads) or "resume" (a checkpoint). Versions carry every expert row (``full``)
        until the play has applied one of this generation (``confirm(generation)``, polled by rank 0's writer after each
        upload: the play's weights-follow.jsonl), then only the rows trained since the start. Without ``confirm``:
        the start export, and every version of a resumed trainer (the play can switch to any of them)."""
        self.root, self.cut, self.rank, self.remote = Path(root), cut, rank, remote
        self.keep, self.files, self.search, self.log = keep, files, search, log
        self.generation, self.base = int(generation if generation is not None else time.time() * 1000), base
        self.confirm, self.confirmed = confirm, False
        self.writer = None
        self.uploads = []
        self.ever = {}  # id(param) -> every expert row trained so far: each version stands alone (one skipped is no loss)
        self.error = None
        self.root.mkdir(parents=True, exist_ok=True)

    def wait(self):
        if self.writer is not None:
            self.writer.join()
            self.writer = None
        if self.error is not None:
            raise RuntimeError("delta export failed") from self.error

    def busy(self):
        """Rank 0's writer (save + upload) still on the previous version, agreed on every rank (a collective)."""
        import torch
        import torch.distributed as dist

        mine = 1.0 if self.rank == 0 and self.writer is not None and self.writer.is_alive() else 0.0
        if not dist.is_initialized():
            return bool(mine)
        flag = torch.tensor([mine], device="cuda" if torch.cuda.is_available() else "cpu")
        dist.broadcast(flag, 0)
        return bool(flag.item())

    def export(self, runner, rows, step, force=False, first=False):
        """Skip-to-newest (c6, 23 Oct: a 228 s upload held the next step's export): while the previous version is
        still being written or uploaded, this step exports nothing (every rank agrees; only its trained rows are gathered); the next free step
        exports the newest weights (each version holds every trainable tensor, so a skipped one loses nothing).
        ``force``: wait for it instead (the start export). ``first``: the start export, every expert row (as is every
        version of a resumed trainer)."""
        import torch.distributed as dist

        began = time.time()
        trained = {}  # id(param) -> global expert rows trained so far, this step's too (fused expert tensors)
        if rows is not None:  # gathered on skipped steps as well: the next version carries the rows they trained
            for it in rows.items:
                mine = (it["rows"] + it["start"]).tolist()
                every = [None] * dist.get_world_size() if dist.is_initialized() else [mine]
                if dist.is_initialized():
                    dist.all_gather_object(every, mine)
                rows_now = self.ever.setdefault(id(it["p"]), set())
                rows_now.update(r for got in every for r in got)
                trained[id(it["p"])] = sorted(rows_now)
        if not force and self.busy():
            if self.error is not None:
                raise RuntimeError("delta export failed") from self.error
            return {"step": step, "skipped": True}
        self.wait()  # one version in flight
        full = first or (not self.confirmed if self.confirm is not None else self.base != "start")  # rank 0's view
        if dist.is_initialized():  # c7f 24 Oct: only rank 0 polls the confirm; HEAL_DELTA_FAST's gathers differ by
            flag = torch.tensor([int(full)], dtype=torch.int64,  # full / delta, so ranks 1-2 (still "full") hung
                                device=f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
            dist.broadcast(flag, 0)  # every rank takes rank 0's view
            full = bool(flag.item())
        host = []
        fast = delta_fast()
        for key, p in runner.named_parameters():
            if not p.requires_grad:
                continue
            name = key[len("model."):] if key.startswith("model.") else key
            if fast and skipped(name):  # MTP / frozen: entries() writes nothing (the same on every rank)
                continue
            if fast and FUSED.match(name) and not full and hasattr(p, "full_tensor") and id(p) in trained:
                got = self._gather_rows(p, trained[id(p)])  # only the rows trained so far, to rank 0
                if self.rank == 0:
                    host += entries(name, got, self.cut, rows=trained[id(p)], search=self.search, picked=True)
                del got
                continue
            if fast and hasattr(p, "full_tensor") and not FUSED.match(name):  # dense/router: written as BF16, so
                tensor = p.detach().to(torch.bfloat16).full_tensor()  # cast the shard first: half the bytes, same values
            elif hasattr(p, "full_tensor"):
                tensor = p.full_tensor()  # collective: every rank calls it
            else:  # an unsharded FSDP parameter (the root group after a no-grad forward: start reads, step 0): its
                # storage is FSDP's all-gather buffer, freed at the next reshard while the writer reads it (c7 build)
                tensor = p.data.detach().clone() if self.rank == 0 else None
            if self.rank == 0:
                host += entries(name, tensor.detach(), self.cut,
                                rows=trained.get(id(p)) if FUSED.match(name) and not full else None, search=self.search)
            del tensor
        gathered = round(time.time() - began, 2)
        if self.rank != 0:
            return {"step": step, "gather_seconds": gathered}
        meta = {"generation": self.generation, "base": self.base, "full": full, "first": first}
        self.writer = threading.Thread(target=self._write, args=(step, host, began, meta), daemon=False,
                                       name=f"delta-v{step}")
        self.writer.start()
        return {"step": step, "gather_seconds": gathered, "tensors": len(host), **meta,
                "bytes": sum(t.numel() * t.element_size() for _, t in host)}

    @staticmethod
    def _gather_rows(p, rows):
        """Rows ``rows`` (global, sorted) of a dim-0-sharded DTensor ``p`` on rank 0, in that order (None elsewhere):
        each rank's own rows padded to the largest count, one all_gather (a collective: every rank calls it)."""
        import torch.distributed as dist
        from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

        local = p.to_local()
        world, rank = dist.get_world_size(), dist.get_rank()
        _, offset = compute_local_shape_and_global_offset(p.shape, p.device_mesh, p.placements)
        box = torch.tensor([int(offset[0]), int(local.shape[0])], dtype=torch.int64, device=local.device)
        every = [torch.empty_like(box) for _ in range(world)]
        dist.all_gather(every, box)  # every rank's (start, rows) of the shard
        bounds = [tuple(int(v) for v in x.tolist()) for x in every]
        per = [[e - a for e in rows if a <= e < a + n] for a, n in bounds]
        width = max(1, max(len(x) for x in per))
        mine = per[rank]
        buf = local.new_zeros((width, *local.shape[1:]))
        if mine:
            buf[:len(mine)] = local[torch.tensor(mine, device=local.device)]
        parts = [torch.empty_like(buf) for _ in range(world)]
        dist.all_gather(parts, buf)
        if rank != 0:
            return None
        order = sorted(range(world), key=lambda r: bounds[r][0])  # global row order (rows is sorted)
        return torch.cat([parts[r][:len(per[r])] for r in order])

    def _write(self, step, host, began, meta=None):
        from safetensors.torch import save_file as _save

        def save_file(buf, path):  # a failing shard names its bad tensors (c7 build, 24 Oct)
            try:
                _save(buf, path)
            except Exception as exc:
                bad = []
                for k, t in buf.items():
                    try:
                        _save({k: t}, path + ".probe")
                    except Exception as e2:  # noqa: BLE001
                        bad.append(f"{k} shape {tuple(t.shape)} {t.dtype} stride {t.stride()} offset "
                                   f"{t.storage_offset()} contiguous {t.is_contiguous()} storage "
                                   f"{t.untyped_storage().nbytes()}: {e2}")
                raise RuntimeError(f"shard {path}: {exc}; bad tensors: {bad[:5]}") from exc

        try:
            time.sleep(float(os.environ.get("HEAL_DELTA_TEST_DELAY", "0")))  # tests: a slow upload, so steps skip
            folder = self.root / f"v{step}"
            tmp = self.root / f".v{step}.partial"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            meta = meta or {}
            tag = f"g{meta['generation']}-" if meta.get("generation") is not None else ""  # a re-used v<step> folder
            # (resumed trainer) never pairs a new manifest with an old shard of the same name and size
            total = sum(t.numel() * t.element_size() for _, t in host)
            per = max(1, total // self.files + 1)
            names, buf, size, part = {}, {}, 0, 0
            for name, t in host:
                buf[name] = t
                size += t.numel() * t.element_size()
                if size >= per:
                    part += 1
                    save_file(buf, str(tmp / f"delta-{tag}{part:03d}.safetensors"))
                    names.update({k: f"delta-{tag}{part:03d}.safetensors" for k in buf})
                    buf, size = {}, 0
            if buf:
                part += 1
                save_file(buf, str(tmp / f"delta-{tag}{part:03d}.safetensors"))
                names.update({k: f"delta-{tag}{part:03d}.safetensors" for k in buf})
            files = sorted(set(names.values()))
            manifest = {"version": step, "files": files, "sizes": {f: (tmp / f).stat().st_size for f in files},
                        "tensors": len(names), "bytes": total,
                        "written_epoch": time.time(), "export_seconds": round(time.time() - began, 1), **meta}
            (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
            shutil.rmtree(folder, ignore_errors=True)
            tmp.rename(folder)
            row = {"event": "delta_written", **manifest}
            if self.remote:
                t0 = time.time()
                dst = f"{self.remote.rstrip('/')}/v{step}"
                for args in (["--exclude", "manifest.json"], ["--include", "manifest.json"]):  # manifest last
                    subprocess.run(["rclone", "copy", str(folder), dst, "--transfers", str(self.files),
                                    "--s3-upload-concurrency", "16", "--s3-chunk-size", "32M", "--retries", "5", *args],
                                   check=True, capture_output=True, text=True)
                row["upload_seconds"] = round(time.time() - t0, 1)
                self.uploads.append(step)
                for old in self.uploads[:-self.keep]:  # the server follows the newest: older versions leave the bucket
                    subprocess.run(["rclone", "purge", f"{self.remote.rstrip('/')}/v{old}"], capture_output=True, text=True)
                self.uploads = self.uploads[-self.keep:]
            if self.confirm is not None and (not self.confirmed or getattr(self.confirm, "live", False)):
                try:
                    self.confirmed = bool(self.confirm(self.generation))
                except Exception:  # noqa: BLE001 - unreadable: stay full (safe)
                    self.confirmed = False
                row["play_confirmed"] = self.confirmed
            versions = sorted(int(p.name[1:]) for p in self.root.glob("v*") if p.name[1:].isdigit())
            for old in versions[:-self.keep]:
                shutil.rmtree(self.root / f"v{old}", ignore_errors=True)
            if self.log is not None:
                self.log(row)
        except BaseException as exc:  # noqa: BLE001 - raised by the next wait()
            self.error = exc
