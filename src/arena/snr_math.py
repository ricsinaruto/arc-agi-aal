"""arc2 RL gradient-SNR and LoRA-scope diagnostic, the tensor side: runs under the trainer's
python (torch, safetensors) on rl1_trainer checkpoints (op_checkpoint: master.safetensors, the fp32 adapter;
optimizer.pt, {"optimizer": AdamW state_dict, "updates", "names"}). One AdamW step from an empty state (betas b1/b2,
eps, no weight decay) leaves exp_avg = (1 - b1) g, g the clipped gradient, and moves each element by -lr g / (|g| + eps).

  extract START CKPT OUT --lr --clip --grad-norm   a call's unclipped gradient (bf16, START's name order), step checks
  analyze SPEC OUT                                 halves / task cosines, signal shares, the (b) scope directions
  state START DIRS SCOPE D OUT                     START + D x SCOPE's unit direction, START's optimizer state
  fake init|nudge|update|norm|info ...             the preflight stand-in trainer's checkpoints, same format
Each command prints one JSON line."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

GROUPS = (("attention", "self_attn"), ("gdn", "linear_attn"), ("shared_expert", "shared_expert"), ("experts", "experts"))
SCOPES = ("all", "attention", "gdn", "shared_expert")


def group_of(name):
    return next((group for group, key in GROUPS if f".{key}." in name), "other")


def layer_of(name):
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1]) if "layers" in parts else -1


def load_state(folder):
    import torch
    from safetensors.torch import load_file

    folder = Path(folder)
    return (load_file(str(folder / "master.safetensors")),
            torch.load(str(folder / "optimizer.pt"), map_location="cpu", weights_only=False))


def flat(tensors, names):
    import torch

    return torch.cat([tensors[n].reshape(-1).float() for n in names])


def cos(a, b):  # float64 sums: float32 dots over 33.5M elements drift ~0.25% of |a||b| (snr-v1 repeat cos read 1.0025)
    a, b = a.double(), b.double()
    den = float(a.norm()) * float(b.norm())
    return float(a.dot(b)) / den if den else 0.0


def extract(start, ckpt, out, lr, clip, grad_norm):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    start, ckpt = Path(start), Path(ckpt)
    s_state = torch.load(str(start / "optimizer.pt"), map_location="cpu", weights_only=False, mmap=True)
    state = torch.load(str(ckpt / "optimizer.pt"), map_location="cpu", weights_only=False, mmap=True)
    names = s_state["names"]
    if state["names"] != names:
        raise ValueError("checkpoint names differ from the start state's")
    opt = state["optimizer"]
    group = opt["param_groups"][0]
    (b1, b2), eps = group["betas"], group["eps"]
    if group.get("weight_decay"):
        raise ValueError(f"weight decay {group['weight_decay']}: the one-step identity does not hold")
    coef = min(1.0, clip / (grad_norm + 1e-6)) if clip else 1.0
    sums = dict.fromkeys(("err", "exp", "verr", "v", "norm", "delta"), 0.0)
    steps, groups, parts, nonzero = set(), {}, [], 0
    with safe_open(str(start / "master.safetensors"), "pt") as fs, safe_open(str(ckpt / "master.safetensors"), "pt") as fc:
        for i, n in enumerate(names):  # per tensor: float64 sums, no whole-vector temporaries
            new, old = fc.get_tensor(n).double().reshape(-1), fs.get_tensor(n).double().reshape(-1)
            row = opt["state"].get(i) or {}
            m = row["exp_avg"].reshape(-1).double() if "exp_avg" in row else torch.zeros_like(new)
            v = row["exp_avg_sq"].reshape(-1).double() if "exp_avg_sq" in row else torch.zeros_like(new)
            if "step" in row:
                steps.add(float(row["step"]))
            g = m / (1 - b1)  # the clipped gradient
            expected = -(lr / (1 - b1)) * m / (v.sqrt() / math.sqrt(1 - b2) + eps)
            delta = new - old
            true = g / coef
            sums["err"] += float((delta - expected).pow(2).sum())
            sums["exp"] += float(expected.pow(2).sum())
            sums["verr"] += float((v - (1 - b2) * g * g).pow(2).sum())
            sums["v"] += float(v.pow(2).sum())
            sums["delta"] += float(delta.pow(2).sum())
            norm2 = float(true.pow(2).sum())
            sums["norm"] += norm2
            key = f"{group_of(n)}.{'A' if 'lora_A' in n else 'B'}"
            groups[key] = groups.get(key, 0.0) + norm2
            nonzero += int((true != 0).sum())
            parts.append(true.to(torch.bfloat16))
    flat_g = torch.cat(parts)
    del parts
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_file({"g": flat_g.contiguous()}, str(out))
    return {"step_rel_err": math.sqrt(sums["err"]) / max(math.sqrt(sums["exp"]), 1e-30),
            "v_rel_err": math.sqrt(sums["verr"]) / max(math.sqrt(sums["v"]), 1e-30),
            "steps": sorted(steps), "updates": int(state.get("updates", -1)), "coef": coef,
            "norm": math.sqrt(sums["norm"]), "nonzero": nonzero, "elements": int(flat_g.numel()),
            "update_norm": math.sqrt(sums["delta"]), "norm2_by_group": {k: groups[k] for k in sorted(groups)}}


def analyze(spec_path, out_path):
    """arc2 snr-v1's report in Gram form (rl1's adapter: ~2.8e9 elements): every reported quantity but the Adam-step
    ones is a dot product of linear combinations of the calls' gradients, so one pass over the tensors accumulates
    G = X X^T (float64) overall, per group, per layer and on the B factors, and the halves' elementwise Adam-step stats.
    Memory: the calls' bf16 gradients and one tensor's span in float64."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    spec = json.loads(Path(spec_path).read_text())
    start = Path(spec["start"])
    names = torch.load(str(start / "optimizer.pt"), map_location="cpu", weights_only=False, mmap=True)["names"]
    with safe_open(str(start / "master.safetensors"), "pt") as f:
        sizes = [math.prod(f.get_slice(n).get_shape()) for n in names]
    total, eps = sum(sizes), float(spec.get("eps", 1e-8))
    calls, rep = spec["calls"], spec.get("repeat")
    rows = [c["name"] for c in calls] + (["~repeat"] if rep else [])
    index = {name: i for i, name in enumerate(rows)}
    grads = [load_file(c["grad"])["g"] for c in calls] + ([load_file(rep["grad"])["g"]] if rep else [])
    if any(g.numel() != total for g in grads):
        raise ValueError("a gradient's length differs from the start state's")
    n = len(rows)

    def coef(chosen):  # combine(): the calls' token- (or try-) weighted mean, as weights over the rows
        v = torch.zeros(n, dtype=torch.float64)
        weight = sum(c["divisor"] for c in chosen)
        for c in chosen:
            v[index[c["name"]]] = c["divisor"] / weight
        return v

    paired = {t for t in {c["task"] for c in calls} if {c["half"] for c in calls if c["task"] == t} == {0, 1}}
    split = [c for c in calls if c["task"] in paired]
    a0, a1 = coef([c for c in split if c["half"] == 0]), coef([c for c in split if c["half"] == 1])
    full, whole = coef(calls), coef(split)
    gram = torch.zeros(n, n, dtype=torch.float64)
    by_group = {g: torch.zeros(n, n, dtype=torch.float64) for g, _ in GROUPS}
    b_gram = torch.zeros(n, n, dtype=torch.float64)
    by_layer = {}
    adam = dict.fromkeys(("dot", "na", "nb"), 0.0)
    agree = support = 0
    offset = 0
    for name, k in zip(names, sizes):
        x = torch.stack([g[offset:offset + k] for g in grads]).double()
        part = x @ x.T
        gram += part
        group = group_of(name)
        if group in by_group:
            by_group[group] += part
        if "lora_B" in name:
            b_gram += part
        layer = layer_of(name)
        by_layer[layer] = by_layer.get(layer, torch.zeros(n, n, dtype=torch.float64)) + part
        h0, h1 = a0 @ x, a1 @ x
        ua, ub = h0 / (h0.abs() + eps), h1 / (h1.abs() + eps)
        adam["dot"] += float(ua.dot(ub))
        adam["na"] += float(ua.dot(ua))
        adam["nb"] += float(ub.dot(ub))
        sup = (h0.abs() >= eps) & (h1.abs() >= eps)
        agree += int((torch.sign(h0[sup]) == torch.sign(h1[sup])).sum())
        support += int(sup.sum())
        offset += k
        del x, part, h0, h1, ua, ub, sup

    def q(a, b, g=gram):
        return float(a @ g @ b)

    def gcos(a, b, g=gram):
        den = math.sqrt(max(q(a, a, g), 0.0) * max(q(b, b, g), 0.0))
        return q(a, b, g) / den if den else 0.0

    split2 = q(whole, whole)

    def split_row(g):  # signal share: <h0, h1> / |ref|^2 (ref = the split calls' combined gradient)
        w2 = q(whole, whole, g)
        return {"cos": round(gcos(a0, a1, g), 6), "signal_share": round(q(a0, a1, g) / w2, 6) if w2 else None,
                "share_of_norm2": round(w2 / split2, 6) if split2 else None}

    def unit(name):
        v = torch.zeros(n, dtype=torch.float64)
        v[index[name]] = 1.0
        return v

    report = {"elements": total, "b_elements": sum(k for name, k in zip(names, sizes) if "lora_B" in name),
              "unpaired": sorted({c["task"] for c in calls} - paired),
              "halves": {**split_row(gram), "dot": q(a0, a1), "norm2": [q(a0, a0), q(a1, a1)], "split_norm2": split2,
                         "full_norm2": q(full, full), "cos_split_to_full": round(gcos(whole, full), 6),
                         "episodes": [sum(c["episodes"] for c in split if c["half"] == h) for h in (0, 1)],
                         "tokens": [sum(c["divisor"] for c in split if c["half"] == h) for h in (0, 1)],
                         "adam_cos": round(adam["dot"] / math.sqrt(adam["na"] * adam["nb"]), 6) if adam["na"] * adam["nb"] else None,
                         "sign_agree": round(agree / support, 6) if support else None,
                         "groups": {g: split_row(m) for g, m in by_group.items() if float(m.trace())},
                         "b_only": split_row(b_gram),
                         "layers": [{"layer": layer, **split_row(m)} for layer, m in sorted(by_layer.items())]}}
    within, across = [], []
    for i, a in enumerate(calls):
        for b in calls[i + 1:]:
            ua_, ub_ = unit(a["name"]), unit(b["name"])
            row = {"pair": [a["name"], b["name"]], "cos": round(gcos(ua_, ub_), 6), "dot": q(ua_, ub_),
                   "episodes": [a["episodes"], b["episodes"]]}
            (within if a["task"] == b["task"] else across).append(row)
    report["within_task"] = within
    report["across_task"] = {"pairs": len(across), "cos_mean": round(sum(r["cos"] for r in across) / max(1, len(across)), 6),
                             "dot_mean": sum(r["dot"] for r in across) / max(1, len(across)), "rows": across}
    report["within_task_summary"] = {"pairs": len(within), "cos_mean": round(sum(r["cos"] for r in within) / max(1, len(within)), 6),
                                     "dot_mean": sum(r["dot"] for r in within) / max(1, len(within))}
    tasks = sorted({c["task"] for c in calls})
    report["tasks"] = {}
    for t in tasks:
        mine, rest = coef([c for c in calls if c["task"] == t]), [c for c in calls if c["task"] != t]
        report["tasks"][t] = {"norm2": q(mine, mine), "cos_to_rest": round(gcos(mine, coef(rest)), 6) if rest else None}
    if rep:
        r, o = unit("~repeat"), unit(rep["of"])
        diff = r - o
        report["repeat"] = {"of": rep["of"], "cos": round(gcos(r, o), 8),
                            "rel_diff": math.sqrt(max(q(diff, diff), 0.0)) / max(math.sqrt(max(q(o, o), 0.0)), 1e-30)}
    report["gram"] = {"rows": rows, "matrix": [[float(v) for v in line] for line in gram]}
    Path(out_path).write_text(json.dumps(report, indent=1))
    return {"halves": {k: v for k, v in report["halves"].items() if k != "layers"},
            "within_task_summary": report["within_task_summary"],
            "across_task": {k: v for k, v in report["across_task"].items() if k != "rows"},
            "repeat": report.get("repeat"), "tasks": report["tasks"]}


def make_state(start, directions, scope, distance, out):
    from safetensors import safe_open
    from safetensors.torch import save_file

    s_master, s_state = load_state(start)
    names = s_state["names"]
    with safe_open(str(directions), "pt") as f:
        d = f.get_tensor(scope).float()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    new, offset, moved = {}, 0, 0.0
    for n in names:
        t = s_master[n].float()
        k = t.numel()
        step = distance * d[offset:offset + k].view(t.shape)
        new[n] = (t + step).contiguous()
        moved += float(step.pow(2).sum())
        offset += k
    save_file(new, str(out / "master.safetensors"))
    shutil.copyfile(Path(start) / "optimizer.pt", out / "optimizer.pt")
    return {"path": str(out), "scope": scope, "D": distance, "norm": math.sqrt(moved)}


def sha(folder):
    return hashlib.sha256((Path(folder) / "master.safetensors").read_bytes()).hexdigest()


# ---- preflight stand-in (arc2.preflight's trainer in tensor mode): a small adapter with the real name pattern ----

FAKE = (("linear_attn.in_proj_qkv", 12, 16), ("self_attn.q_proj", 12, 12), ("mlp.shared_expert.gate_proj", 12, 10))


def fake_layout():
    out = []
    for layer in range(2):
        for module, d_in, d_out in FAKE:
            stem = f"base_model.model.model.language_model.layers.{layer}.{module}"
            out += [(f"{stem}.lora_A.default.weight", (4, d_in)), (f"{stem}.lora_B.default.weight", (d_out, 4))]
    return out


def save_fake(folder, names, params, optimizer, updates):
    import torch
    from safetensors.torch import save_file

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_file({n: p.detach().float().contiguous() for n, p in zip(names, params)}, str(folder / "master.safetensors"))
    torch.save({"optimizer": optimizer.state_dict(), "updates": updates, "names": names}, str(folder / "optimizer.pt"))


def noise(key, shape):
    import torch

    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def fake(args):
    import torch

    def adamw(params):
        return torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    if args.action == "init":
        names = [n for n, _ in fake_layout()]
        params = [torch.nn.Parameter(noise("init" + n, s) * 0.05 if "lora_A" in n else torch.zeros(s)) for n, s in fake_layout()]
        save_fake(args.out, names, params, adamw(params), 0)
        return {"path": args.out, "adapter_norm": math.sqrt(sum(float(p.detach().pow(2).sum()) for p in params))}
    master, state = load_state(args.state)
    names = state["names"]
    if args.action == "info":
        return {"updates": int(state["updates"]), "adapter_norm": math.sqrt(sum(float(t.pow(2).sum()) for t in master.values()))}
    if args.action == "norm":
        ref, _ = load_state(args.ref)
        return {"distance": math.sqrt(sum(float((master[n] - ref[n]).pow(2).sum()) for n in names))}
    params = [torch.nn.Parameter(master[n].clone()) for n in names]
    opt = adamw(params)
    opt.load_state_dict(state["optimizer"])
    if args.action == "nudge":
        count = 0
        with torch.no_grad():
            for n, p in zip(names, params):
                if "lora_B" in n and not bool(p.any()):
                    p.fill_(args.value)
                    count += 1
        save_fake(args.out, names, params, opt, int(state["updates"]))
        return {"nudged": count, "value": args.value}
    # update: each attempt's gradient = a shared direction + its task's + its own noise (B factors; A factors get none,
    # as at B ~ 0), weighted by w, averaged over 100 tokens an attempt as the trainer divides by trained tokens
    attempts = json.loads(args.attempts)
    for group in opt.param_groups:
        group["lr"] = args.lr
    for n, p in zip(names, params):
        g = torch.zeros_like(p)
        if "lora_B" in n:
            for a in attempts:
                task = a["name"].split(":")[-2] if a["name"].count(":") >= 2 else a["name"]
                g += float(a["w"]) * (noise("shared" + n, p.shape) + 0.8 * noise(task + n, p.shape) + 2.0 * noise(a["name"] + n, p.shape))
            g /= max(1, len(attempts))
        p.grad = g * 1e-3
    norm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
    before = [p.detach().clone() for p in params]
    opt.step()
    change = math.sqrt(sum(float((p.detach() - b).pow(2).sum()) for p, b in zip(params, before)))
    save_fake(args.out, names, params, opt, int(state["updates"]) + 1)
    return {"grad_norm": norm, "trained_tokens": 100 * len(attempts), "loss": -0.01, "update_norm": change,
            "adapter_norm": math.sqrt(sum(float(p.detach().pow(2).sum()) for p in params))}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract")
    p.add_argument("start"), p.add_argument("ckpt"), p.add_argument("out")
    p.add_argument("--lr", type=float, required=True), p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--grad-norm", type=float, required=True)
    p = sub.add_parser("analyze")
    p.add_argument("spec"), p.add_argument("out")
    p = sub.add_parser("state")
    p.add_argument("start"), p.add_argument("directions"), p.add_argument("scope"), p.add_argument("distance", type=float)
    p.add_argument("out")
    p = sub.add_parser("sha")
    p.add_argument("folder")
    p = sub.add_parser("fake")
    p.add_argument("action", choices=("init", "nudge", "update", "norm", "info"))
    p.add_argument("--state"), p.add_argument("--out"), p.add_argument("--ref"), p.add_argument("--attempts", default="[]")
    p.add_argument("--lr", type=float, default=1e-4), p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--value", type=float, default=1e-12)
    args = parser.parse_args()
    if args.command == "extract":
        out = extract(args.start, args.ckpt, args.out, args.lr, args.clip, args.grad_norm)
    elif args.command == "analyze":
        out = analyze(args.spec, args.out)
    elif args.command == "state":
        out = make_state(args.start, args.directions, args.scope, args.distance, args.out)
    elif args.command == "sha":
        out = {"sha256": sha(args.folder)}
    else:
        out = fake(args)
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
