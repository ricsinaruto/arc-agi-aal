"""Reshard a heal_kd resume folder to another world size (rounds 2+ on 4 B300 from a 2-rank round, 22 Oct).

heal_kd keeps per-rank state: ``optim-rank<r>.pt`` = {"adam": AdamW state (FSDP2 dim-0 DTensor shards), "rows":
RowAdam state (selected expert rows of this rank's experts, moments per row)} and, in a mid-round checkpoint,
``params-rank<r>.pt`` (dim-0 shards of the trainable tensors). This joins every rank's shards and splits them for the
new world (torch.chunk, FSDP2's split), the Adam moments as plain tensors (heal_kd wraps them on the new mesh at load);
expert rows move to the rank that owns their expert. Every tensor is checked: the new shards rejoin to the old whole.

    python heal_reshard.py SRC DST WORLD NUM_EXPERTS     (CPU; torch of the trainer: the files hold DTensors)
"""
import json
import shutil
import sys
from pathlib import Path


def _local(t):
    return t.to_local() if hasattr(t, "to_local") else t


def _split(full, world):
    parts = list(full.chunk(world, 0))
    if len(parts) != world:  # torch.chunk returns fewer pieces when dim 0 < world: FSDP2 pads with empty shards
        parts += [full[:0]] * (world - len(parts))
    return [p.clone() for p in parts]


def _starts(n, world):
    """Each rank's first dim-0 index under torch.chunk / FSDP2 (ceil(n / world) per rank, the last ones short or
    empty), plus n."""
    size = -(-n // world)
    return [min(r * size, n) for r in range(world + 1)]


def reshard(src, dst, world, num_experts):
    import torch

    src, dst = Path(src), Path(dst)
    old = sorted(src.glob("optim-rank*.pt"), key=lambda f: int(f.stem.split("rank")[1]))
    w0 = len(old)
    if not w0 or [int(f.stem.split("rank")[1]) for f in old] != list(range(w0)):
        raise SystemExit(f"{src}: optimizer shards {[f.name for f in old]}")
    states = [torch.load(f, map_location="cpu", weights_only=False) for f in old]
    report = {"src": str(src), "dst": str(dst), "world_from": w0, "world_to": world, "num_experts": num_experts}

    # AdamW: same param indices on every rank; step counters equal
    adam0 = states[0]["adam"]
    keys = sorted(adam0["state"])
    if any(sorted(s["adam"]["state"]) != keys for s in states):
        raise SystemExit("ranks hold Adam state for different parameters")
    new_adam = [{"state": {}, "param_groups": json.loads(json.dumps(adam0["param_groups"]))} for _ in range(world)]
    checked = 0
    for i in keys:
        per = [s["adam"]["state"][i] for s in states]
        for q in range(world):
            new_adam[q]["state"][i] = {}
        for k in per[0]:
            vals = [p[k] for p in per]
            if k == "step" or not isinstance(vals[0], torch.Tensor) or vals[0].dim() == 0:
                if any(not torch.equal(torch.as_tensor(v), torch.as_tensor(vals[0])) for v in vals):
                    raise SystemExit(f"adam state {i}.{k} differs across ranks")
                for q in range(world):
                    new_adam[q]["state"][i][k] = vals[0].clone() if isinstance(vals[0], torch.Tensor) else vals[0]
                continue
            full = torch.cat([_local(v) for v in vals], 0)
            # a DTensor carries its global shape; an asynchronous checkpoint's local shards name theirs beside them
            want = tuple(vals[0].shape) if hasattr(vals[0], "to_local") else \
                tuple((states[0].get("adam_global_shapes") or {}).get(i, {}).get(k) or ())
            if not want:
                raise SystemExit(f"adam state {i}.{k}: local shards without a recorded global shape")
            if want != tuple(full.shape):
                raise SystemExit(f"adam state {i}.{k}: shards join to {tuple(full.shape)}, global {want}")
            parts = _split(full, world)
            if not torch.equal(torch.cat(parts, 0), full):
                raise SystemExit(f"adam state {i}.{k}: resplit does not rejoin")
            for q in range(world):
                new_adam[q]["state"][i][k] = parts[q]
            checked += 1
    report["adam_tensors"] = checked

    # RowAdam: rows are local expert indices of each rank's slice; move each row to its expert's new owner
    rows0 = [s["rows"] for s in states]
    t = rows0[0]["t"]
    if any(r["t"] != t for r in rows0) or len({len(r["rows"]) for r in rows0}) != 1:
        raise SystemExit("RowAdam step or tensor count differs across ranks")
    old_start, new_start = _starts(num_experts, w0), _starts(num_experts, world)  # uneven splits (128 over 3)
    new_rows = [{"t": t, "rows": [], "m": [], "v": []} for _ in range(world)]
    moved = 0
    for j in range(len(rows0[0]["rows"])):
        g = torch.cat([r["rows"][j].long() + old_start[rank] for rank, r in enumerate(rows0)])
        m = torch.cat([r["m"][j] for r in rows0])
        v = torch.cat([r["v"][j] for r in rows0])
        if len(set(g.tolist())) != len(g) or (len(g) and int(g.max()) >= num_experts):
            raise SystemExit(f"rows tensor {j}: bad expert ids {g.tolist()[:8]}")
        back = []
        for q in range(world):
            sel = (g >= new_start[q]) & (g < new_start[q + 1])
            new_rows[q]["rows"].append((g[sel] - new_start[q]).to(rows0[0]["rows"][j].dtype))
            new_rows[q]["m"].append(m[sel].clone())
            new_rows[q]["v"].append(v[sel].clone())
            back.append(g[sel])
        if sorted(torch.cat(back).tolist()) != sorted(g.tolist()):
            raise SystemExit(f"rows tensor {j}: experts lost in the move")
        moved += len(g)
    report["expert_rows"] = moved

    dst.mkdir(parents=True, exist_ok=True)
    for q in range(world):
        torch.save({"adam": new_adam[q], "rows": new_rows[q]}, dst / f"optim-rank{q}.pt")

    # sharded trainable tensors (mid-round checkpoints)
    params = sorted(src.glob("params-rank*.pt"), key=lambda f: int(f.stem.split("rank")[1]))
    if params:
        if len(params) != w0:
            raise SystemExit(f"{len(params)} params shards for {w0} optimizer shards")
        ps = [torch.load(f, map_location="cpu", weights_only=True) for f in params]
        out = [{} for _ in range(world)]
        starts = _starts(num_experts, world)
        for name in ps[0]:
            if name.endswith("::experts"):
                continue
            if name + "::experts" in ps[0]:  # rows-only expert tensor (c7b 24 Oct): global ids move to their owners
                g = torch.cat([p[name + "::experts"].long() for p in ps])
                vals = torch.cat([p[name] for p in ps], 0)
                for q in range(world):
                    sel = (g >= starts[q]) & (g < starts[q + 1])
                    out[q][name], out[q][name + "::experts"] = vals[sel].clone(), g[sel].clone()
                if sum(int(o[name].shape[0]) for o in out) != len(g):
                    raise SystemExit(f"params {name}: experts lost in the move")
                continue
            full = torch.cat([p[name] for p in ps], 0)
            parts = _split(full, world)
            if not torch.equal(torch.cat(parts, 0), full):
                raise SystemExit(f"params {name}: resplit does not rejoin")
            for q in range(world):
                out[q][name] = parts[q]
        for q in range(world):
            torch.save(out[q], dst / f"params-rank{q}.pt")
        report["param_tensors"] = len([k for k in ps[0] if not k.endswith("::experts")])
    for name in ("progress.json", "behavior.pt", "routing.pt"):  # world-independent; progress records the new world
        if (src / name).exists():
            if name == "progress.json":
                p = json.loads((src / name).read_text())
                if "world" in p:
                    p["world"] = world
                (dst / name).write_text(json.dumps(p))
            else:
                shutil.copyfile(src / name, dst / name)
    (dst / "reshard.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    print(json.dumps(reshard(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))))
