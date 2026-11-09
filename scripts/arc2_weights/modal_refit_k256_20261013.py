"""Scratch (9 October): refit-only INT3 expert overlay (level 0.0 = all 640 neurons kept), CPU only.
Experts with fewer than 1,280 calibration rows keep the lock's bytes. Derived from modal_nfit.py:


Per layer and kept expert of the DEV lock: the hidden neurons to keep are chosen in rounds (least cost with the others
refit), moved to the first K positions and the rest zeroed. Gate/up rows are the lock's INT3 rows (bytes moved, same
scale_2); down = least-squares refit of the kept columns toward the BF16 expert output on the DEV calibration capture,
re-rounded to the INT3 grid with the overlay's rule. Experts with too few rows keep the W4 source's columns unfit.
The container, names, shapes and mask are the lock's, so the overlay is served unchanged (the zeros still cost time).
One overlay per level; the manifest is written last, after every layer passed ``verify_shard`` structurally.

  modal run --detach modal_refit.py --run-id tp-refit-dev10-v1-20261009 --spec refit-spec.json
"""
import json
import os
from pathlib import Path

import modal

ARM = Path("./.claude/worktrees/throughput-arm")
LEDGER = Path("./docs/spend.json")
LAYER_TIMEOUT = int(os.environ.get("ARC_NFIT_LAYER_TIMEOUT", "900"))
TIMEOUT = int(os.environ.get("ARC_NFIT_TIMEOUT", "2400"))
CPU, RAM_GIB, PARALLEL = 8, 32, 16
MODEL_DIR = "/assets/flash-input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"
OVERLAY_ROOT = "/assets/flash-input/datasets/richardcsaky"

app = modal.App("arc2-refit-20261013", include_source=False)
assets = modal.Volume.from_name("arc-prolong-assets")
runs = modal.Volume.from_name("arc-prolong-runs")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.9.1", "safetensors", "numpy", index_url="https://download.pytorch.org/whl/cpu",
                      extra_index_url="https://pypi.org/simple")
         .add_local_dir(str(ARM / "src"), "/opt/src", copy=True, ignore=["**/__pycache__/**"])
         .add_local_file(__file__, "/root/modal_refit.py", copy=True))


def width_of(level, inter=640):
    return inter - int(round(inter * level / 16)) * 16


@app.function(image=image, cpu=CPU, memory=RAM_GIB * 1024, timeout=LAYER_TIMEOUT, retries=0, max_containers=PARALLEL,
              volumes={"/assets": assets, "/runs": runs})
def fit_layer(layer: int, spec: dict):
    import hashlib
    import sys
    import time

    sys.path.insert(0, "/opt/src")
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from arena import expert_3bit_math as tm
    from arena import expert_3bit_overlay as ov
    from arena import expert_quant_bench as B  # noqa: N812
    from arena import expert_quant_math as qm

    started = time.time()
    torch.set_num_threads(CPU)
    E, H, INTER, TOPK = B.E, B.H, B.INTER, B.TOPK  # noqa: N806
    fp8 = torch.float8_e4m3fn
    silu = torch.nn.functional.silu
    levels = spec["levels"]
    widths = {p: width_of(p, INTER) for p in levels}
    kept = sorted(set(range(E)) - set(spec["pruned"][str(layer)]))
    index = json.loads((Path(MODEL_DIR) / "model.safetensors.index.json").read_text())["weight_map"]
    bf16_index = json.loads((Path(spec["bf16_dir"]) / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp.experts."
    ref = {}
    for proj in ("gate_up_proj", "down_proj"):
        with safe_open(str(Path(spec["bf16_dir"]) / bf16_index[prefix + proj]), framework="pt", device="cpu") as fh:
            ref[proj] = fh.get_tensor(prefix + proj)
    # The lock's shards of this layer, whole (they are rewritten).
    shards = sorted({index[f"{prefix}{e}.gate_proj.weight"] for e in range(E)})
    lock, meta = {}, {}
    for s in shards:
        with safe_open(str(Path(spec["lock"]) / s), framework="pt", device="cpu") as fh:
            meta[s] = fh.metadata()
            lock[s] = {name: fh.get_tensor(name) for name in fh.keys()}
    present = sorted(int(n.split(".mlp.experts.")[1].split(".")[0]) for s in shards for n in lock[s]
                     if n.startswith(prefix) and n.endswith(".gate_proj.weight"))
    if present != kept:
        raise ValueError(f"layer {layer}: the lock holds {len(present)} experts, the mask keeps {len(kept)}")
    w4 = {}

    def lt(e, proj, kind):
        name = f"{prefix}{e}.{proj}.{kind}"
        return lock[index[name]][name]

    def u8(t):
        return t.view(torch.uint8) if t.dtype == fp8 else t

    def w4_tensor(name):
        s = index[name]
        if s not in w4:
            w4[s] = safe_open(str(Path(spec["w4"]) / s), framework="pt", device="cpu").__enter__()
        return w4[s].get_tensor(name)

    def down_w4(e):
        return qm.dequantize(u8(w4_tensor(f"{prefix}{e}.down_proj.weight")).unsqueeze(0),
                             u8(w4_tensor(f"{prefix}{e}.down_proj.weight_scale")).view(fp8).unsqueeze(0),
                             w4_tensor(f"{prefix}{e}.down_proj.weight_scale_2").reshape(1).float())[0]

    def act(x, gate_up):
        h = x @ gate_up.t()
        return silu(h[:, :INTER]) * h[:, INTER:]

    # Eval rows (the moeeval pick of the lock fit's eval capture).
    root = Path(spec["eval_dir"])
    rows = int(json.loads((root / "meta.json").read_text())["rows"][str(layer)])
    x = torch.from_numpy(np.fromfile(root / f"L{layer:02d}.x", dtype=np.int16).reshape(rows, H)).view(torch.bfloat16)
    ids = torch.from_numpy(np.fromfile(root / f"L{layer:02d}.ids", dtype=np.int16).reshape(rows, TOPK).astype(np.int32))
    w = torch.from_numpy(np.fromfile(root / f"L{layer:02d}.w", dtype=np.float32).reshape(rows, TOPK))
    ids = qm.remap_pruned_ids(ids, kept)
    n = min(spec["eval_rows"], rows)
    pick = torch.linspace(0, rows - 1, n).round().long()
    xe = x[pick].float()
    e_token, e_weight, e_bounds = qm.routed_rows(ids[pick].contiguous(), w[pick].contiguous(), range(E), E)
    del x, ids, w
    # Calibration rows (fp8 with row scales; ids are ranks in the capture server's kept experts).
    croot = Path(spec["calib_dir"])
    crow = int(json.loads((croot / "meta.json").read_text())["rows"][str(layer)])
    step = int(spec.get("calib_stride", 1))
    q = np.ascontiguousarray(np.fromfile(croot / f"L{layer:02d}.x", dtype=np.uint8).reshape(crow, H)[::step])
    s = np.fromfile(croot / f"L{layer:02d}.s", dtype=np.float32, count=crow)[::step].copy()
    cid = torch.from_numpy(np.fromfile(croot / f"L{layer:02d}.ids", dtype=np.int16).reshape(crow, TOPK)
                           .astype(np.int32))[::step].contiguous()
    cw = torch.from_numpy(np.fromfile(croot / f"L{layer:02d}.w", dtype=np.float32).reshape(crow, TOPK))[::step].contiguous()
    ckept = sorted(set(range(E)) - set(spec["calib_pruned"][str(layer)]))
    if int(cid.max()) >= len(ckept):
        raise ValueError(f"calib ids reach {int(cid.max())}, mask keeps {len(ckept)}")
    cid = qm.remap_pruned_ids(cid, ckept)
    c_token, c_weight, c_bounds = qm.routed_rows(cid, cw, range(E), E)
    big = (torch.from_numpy(q), torch.from_numpy(s))
    gen = torch.Generator().manual_seed(layer)
    base = torch.zeros((n, H), dtype=torch.float32)
    y = {a: torch.zeros((n, H), dtype=torch.float32) for a in ["lock", *[str(p) for p in levels]]}
    new = {p: {} for p in levels}  # level -> tensor name -> replacement
    stats = {"fit_rows": [], "ls": 0, "ranked_unfit": 0, "norm_only": 0, "eval_routed": 0, "requant_checks": 0}
    steps = sorted({INTER, 544, 464, 400, 352, 320, 288, 256, 224, 192, 160, 128, *widths.values()}, reverse=True)
    for e in kept:
        g_ref, d_ref = ref["gate_up_proj"][e].float(), ref["down_proj"][e].float()
        gw = torch.cat([u8(lt(e, "gate_proj", "weight")), u8(lt(e, "up_proj", "weight"))])
        gs = torch.cat([u8(lt(e, "gate_proj", "weight_scale")), u8(lt(e, "up_proj", "weight_scale"))])
        s13 = lt(e, "gate_proj", "weight_scale_2").reshape(1).float()
        g_lock = qm.dequantize(gw.unsqueeze(0), gs.view(fp8).unsqueeze(0), s13)[0]
        d_w4 = down_w4(e)
        lo, hi = c_bounds[e], c_bounds[e + 1]
        sel = torch.arange(lo, hi)
        if hi - lo > spec["max_rows"]:
            sel = lo + torch.randperm(hi - lo, generator=gen)[:spec["max_rows"]].sort().values
        rows_e = len(sel)
        stats["fit_rows"].append(rows_e)
        norm = d_w4.pow(2).sum(0)
        chosen, fits = {}, {}
        if rows_e >= 2 * INTER:
            t = c_token[sel]
            xf = big[0][t].view(fp8).float() * big[1][t].unsqueeze(1)
            wf = c_weight[sel].unsqueeze(1)
            hw = act(xf, g_lock) * wf
            gram = hw.t() @ hw
            cross = (hw.t() @ (act(xf, g_ref) * wf)) @ d_ref.t()  # sum w^2 h y_ref^T  [INTER, H]
            del xf, hw
            lam = spec["damp"] * float(gram.diagonal().mean())
            live = torch.arange(INTER)
            for width in steps:
                if len(live) > width:
                    sub_inv = torch.linalg.inv(gram[live][:, live] + lam * torch.eye(len(live)))
                    cost = (sub_inv @ cross[live]).pow(2).sum(1) / sub_inv.diagonal()
                    live = live[cost.argsort(descending=True)[:width].sort().values]
                chosen[width] = live
            for p, k in widths.items():
                idx = chosen[k]
                fits[p] = torch.linalg.solve(gram[idx][:, idx] + lam * torch.eye(k), cross[idx]).t().contiguous()
            stats["ls"] += 1
        else:
            if rows_e >= spec["min_rows"]:
                t = c_token[sel]
                hw = act(big[0][t].view(fp8).float() * big[1][t].unsqueeze(1), g_lock) * c_weight[sel].unsqueeze(1)
                order = (hw.pow(2).sum(0) * norm).argsort(descending=True)
                stats["ranked_unfit"] += 1
            else:
                order = norm.argsort(descending=True)
                stats["norm_only"] += 1
            for p, k in widths.items():
                chosen[k] = order[:k].sort().values
                fits[p] = d_w4[:, chosen[k]].contiguous()
        a, b = e_bounds[e], e_bounds[e + 1]
        routed = a != b
        if routed:
            stats["eval_routed"] += 1
            t, wt = e_token[a:b], e_weight[a:b].unsqueeze(1)
            xr = xe[t]
            base.index_add_(0, t, (act(xr, g_ref) @ d_ref.t()) * wt)
            d_lock = qm.dequantize(u8(lt(e, "down_proj", "weight")).unsqueeze(0),
                                   u8(lt(e, "down_proj", "weight_scale")).view(fp8).unsqueeze(0),
                                   lt(e, "down_proj", "weight_scale_2").reshape(1).float())[0]
            h_lock = act(xr, g_lock)
            y["lock"].index_add_(0, t, (h_lock @ d_lock.t()) * wt)
        for p, k in widths.items():
            if p in fits and rows_e < 2 * INTER:  # refit only: too few rows, the lock's tensors stay
                if routed:
                    y[str(p)].index_add_(0, t, (h_lock @ d_lock.t()) * wt)
                continue
            keep = chosen[k]
            gone = torch.ones(INTER, dtype=torch.bool)
            gone[keep] = False
            gone = gone.nonzero().squeeze(1)
            # gate/up: kept rows first, the others' codes zero (their block scales stay, finite)
            g_new, gs_new = torch.zeros_like(gw), torch.empty_like(gs)
            for off in (0, INTER):
                g_new[off:off + k] = gw[off + keep]
                gs_new[off:off + k] = gs[off + keep]
                gs_new[off + k:off + INTER] = gs[off + gone]
            d = torch.zeros((1, H, INTER), dtype=torch.float32)
            d[0, :, :k] = fits[p]
            n2 = tm.scale2_of(d, "int3")
            dq, sc = tm.rtn(d, n2, "int3", spec["group"], ov.SEARCH)
            packed, block = tm.to_nvfp4(dq, sc, n2, spec["group"])
            block = torch.where(block.float() == 0, torch.ones((), dtype=torch.float32), block.float()).to(fp8)
            repl = {("gate_proj", "weight"): g_new[:INTER], ("up_proj", "weight"): g_new[INTER:],
                    ("gate_proj", "weight_scale"): gs_new[:INTER], ("up_proj", "weight_scale"): gs_new[INTER:],
                    ("down_proj", "weight"): packed[0], ("down_proj", "weight_scale"): block[0],
                    ("down_proj", "weight_scale_2"): n2.float()}
            for (proj, kind), value in repl.items():
                original = lt(e, proj, kind)
                value = value.contiguous().clone()
                if kind == "weight_scale_2":
                    value = value.reshape(original.shape).to(original.dtype)
                elif value.dtype != original.dtype:
                    value = value.view(original.dtype)
                if value.dtype != original.dtype or value.shape != original.shape:
                    raise RuntimeError(f"L{layer} e{e} {proj}.{kind}: {value.dtype} {tuple(value.shape)}")
                new[p][f"{prefix}{e}.{proj}.{kind}"] = value
            stats["changed"] = stats.get("changed", 0) + 1
            if routed:  # the error is read from the bytes that are written
                g_out = qm.dequantize(g_new.unsqueeze(0), gs_new.view(fp8).unsqueeze(0), s13)[0]
                d_out = qm.dequantize(u8(new[p][f"{prefix}{e}.down_proj.weight"]).unsqueeze(0),
                                      u8(new[p][f"{prefix}{e}.down_proj.weight_scale"]).view(fp8).unsqueeze(0),
                                      new[p][f"{prefix}{e}.down_proj.weight_scale_2"].reshape(1).float())[0]
                out = act(xr, g_out) @ d_out.t()
                y[str(p)].index_add_(0, t, out * wt)
                if stats["requant_checks"] < 6:  # the moved bytes give the same output as the kept subset
                    direct = h_lock[:, keep] @ dq[0, :, :k].t()
                    if not float((out - direct).norm()) <= 1e-4 * float(direct.norm()):
                        raise RuntimeError(f"L{layer} e{e} level {p}: written tensors differ from the kept subset")
                    stats["requant_checks"] += 1
    for fh in w4.values():
        fh.__exit__(None, None, None)
    del big, ref
    written = {}
    for p in levels:
        out_dir = Path(OVERLAY_ROOT) / spec["names"][str(p)]
        out_dir.mkdir(parents=True, exist_ok=True)
        written[str(p)] = {}
        for sname in shards:
            tensors = dict(lock[sname])
            for name in tensors:
                if name in new[p]:
                    tensors[name] = new[p][name]
            path = out_dir / sname
            save_file(tensors, str(path) + ".tmp", metadata=meta[sname])
            os.replace(str(path) + ".tmp", path)
            ov.verify_shard(torch, str(Path(spec["lock"]) / sname), str(path), float("inf"))
            size = path.stat().st_size
            if size != (Path(spec["lock"]) / sname).stat().st_size:
                raise RuntimeError(f"{sname}: {size} bytes differ from the lock shard")
            with path.open("rb") as fh:
                written[str(p)][sname] = {"bytes": size, "sha256": hashlib.file_digest(fh, "sha256").hexdigest(),
                                          "base_bytes": size}
    assets.commit()
    fit_rows = sorted(stats.pop("fit_rows"))
    return {"layer": layer, "seconds": round(time.time() - started), "kept": len(kept), **stats,
            "fit_rows_min_p10_median": [fit_rows[0], fit_rows[len(fit_rows) // 10], fit_rows[len(fit_rows) // 2]],
            "errors": {a: qm.relative_error(v, base) for a, v in y.items()}, "shards": written}


@app.function(image=image, cpu=2, memory=4096, timeout=TIMEOUT, retries=0, max_containers=1,
              volumes={"/assets": assets, "/runs": runs})
def nfit(run_id: str, spec: dict):
    import hashlib
    import time
    import traceback

    started = time.time()
    rec_dir = Path("/runs/research") / run_id
    rec_dir.mkdir(parents=True, exist_ok=True)
    record = {"run_id": run_id, "kind": "expert-refit", "spec": {k: v for k, v in spec.items() if "pruned" not in k}}
    out = {}
    try:
        for p in spec["levels"]:
            if (Path(OVERLAY_ROOT) / spec["names"][str(p)] / "overlay-manifest.json").exists():
                raise RuntimeError(f"{spec['names'][str(p)]} already has a manifest")
        for row in fit_layer.map(spec["layers"], kwargs={"spec": spec}, order_outputs=False, return_exceptions=True):
            if isinstance(row, BaseException):
                record.setdefault("layer_errors", []).append(repr(row)[:2000])
                continue
            out[str(row["layer"])] = row
            print(json.dumps({"t": round(time.time() - started), "done": len(out),
                              **{k: v for k, v in row.items() if k != "shards"}}), flush=True)
            (rec_dir / "nfit.json").write_text(json.dumps(out, indent=1, sort_keys=True))
            runs.commit()
        if record.get("layer_errors"):
            raise RuntimeError(f"{len(record['layer_errors'])} layers failed")
        mean = {a: sum(r["errors"][a] for r in out.values()) / len(out) for a in next(iter(out.values()))["errors"]}
        record["mean_error"] = mean
        if spec.get("manifest") and len(out) == 48:
            assets.reload()
            src = json.loads((Path(spec["lock"]) / "overlay-manifest.json").read_text())
            record["manifests"] = {}
            for p in spec["levels"]:
                out_dir = Path(OVERLAY_ROOT) / spec["names"][str(p)]
                shards = {}
                for row in out.values():
                    shards.update(row["shards"][str(p)])
                if sorted(shards) != sorted(src["shards"]):
                    raise RuntimeError(f"level {p}: {len(shards)} shards, the lock has {len(src['shards'])}")
                for name, info in shards.items():
                    if (out_dir / name).stat().st_size != info["bytes"]:
                        raise RuntimeError(f"{name}: size on the volume differs")
                manifest = {**src, "shards": shards,
                            "settings": {**src["settings"], "source_dir": spec["lock"],
                                         "neuron_prune": {"level": p, "width": width_of(p), "down_source": spec["w4"],
                                                          "calib_dir": spec["calib_dir"], "damp": spec["damp"],
                                                          "method": "least squares to BF16, all neurons kept (refit only)",
                                                          "mean_error": mean[str(p)], "lock_error": mean["lock"]}},
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                raw = json.dumps(manifest, indent=1, sort_keys=True).encode()
                (out_dir / "overlay-manifest.json").write_bytes(raw)
                record["manifests"][str(p)] = {"name": spec["names"][str(p)],
                                               "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                                               "bytes": sum(v["bytes"] for v in shards.values())}
            assets.commit()
    except BaseException as exc:  # noqa: BLE001
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
    record["elapsed_seconds"] = time.time() - started
    (rec_dir / "execution.json").write_text(json.dumps(record, indent=2))
    runs.commit()
    return record


def reserve(run_id, layers):
    import fcntl

    bound = ((LAYER_TIMEOUT + 60) * (CPU * 0.0000131 + RAM_GIB * 0.00000222) * layers
             + (TIMEOUT + 60) * (2 * 0.0000131 + 4 * 0.00000222)) * 1.15
    with LEDGER.open("r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        d = json.load(f)
        if any(r["run_id"] == run_id for r in d["runs"]):
            raise ValueError("Already reserved")
        used = d["infrastructure_reserve_usd"] + sum(r["charged_bound_usd"] for r in d["runs"])
        if used + bound > d["cap_usd"]:
            raise RuntimeError(f"Cap: {used}+{bound}>{d['cap_usd']}")
        d["runs"].append({"run_id": run_id, "state": "reserved", "kind": "research-preparation",
                          "charged_bound_usd": bound, "max_seconds": TIMEOUT, "gpu": None,
                          "cpu_ceiling": CPU * min(layers, PARALLEL) + 2,
                          "memory_ceiling_gib": RAM_GIB * min(layers, PARALLEL) + 4,
                          "authorization": "redo the expert pruning, 3-bit requant and refit on all 120 public ARC-2 eval tasks (ARC-AGI-2 arm, run ids arc2-)",
                          "research_kind": "expert-refit"})
        f.seek(0)
        json.dump(d, f, indent=1)
        f.write("\n")
        f.truncate()
    return bound


@app.local_entrypoint()
def main(run_id: str, spec: str):
    data = json.loads(Path(spec).read_text())
    bound = reserve(run_id, len(data["layers"]))
    call = nfit.spawn(run_id, data)
    record = {"run_id": run_id, "kind": "expert-refit", "function_call_id": call.object_id,
              "charged_bound_usd": bound, "timeout_seconds": TIMEOUT, "state": "submitted", "spec": spec}
    Path(spec).with_name(f"modal-{run_id}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))
