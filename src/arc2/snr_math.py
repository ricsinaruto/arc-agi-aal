"""arc2 RL gradient-SNR and LoRA-scope diagnostic, the tensor side: runs under the trainer's
python (torch, safetensors) on rl1_trainer checkpoints (op_checkpoint: master.safetensors, the fp32 adapter;
optimizer.pt, {"optimizer": AdamW state_dict, "updates", "names"}). One AdamW step from an empty state (betas b1/b2,
eps, no weight decay) leaves exp_avg = (1 - b1) g, g the clipped gradient, and moves each element by -lr g / (|g| + eps).

  extract START CKPT OUT --lr --clip --grad-norm   a call's unclipped gradient (bf16, START's name order), step checks
  analyze SPEC OUT                                 halves / task cosines, signal shares, the (b) scope directions
  gram SPEC OUT                                    per-episode gradients as Gram matrices per (group, layer) block
  combine SPEC OUT                                 bank directions: weighted gradient sums -> first AdamW steps, unit norm
  state START DIRS SCOPE D OUT                     START + D x SCOPE's unit direction, START's optimizer state
  fake init|nudge|update|norm|info ...             the preflight stand-in trainer's checkpoints, same format
Each command prints one JSON line."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

GROUPS = (("attention", "self_attn"), ("gdn", "linear_attn"), ("shared_expert", "shared_expert"))
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
    from safetensors.torch import save_file

    s_master, s_state = load_state(start)
    master, state = load_state(ckpt)
    names = s_state["names"]
    if state["names"] != names:
        raise ValueError("checkpoint names differ from the start state's")
    opt = state["optimizer"]
    group = opt["param_groups"][0]
    (b1, b2), eps = group["betas"], group["eps"]
    if group.get("weight_decay"):
        raise ValueError(f"weight decay {group['weight_decay']}: the one-step identity does not hold")
    m, v, steps = [], [], set()
    for i, n in enumerate(names):
        row = opt["state"].get(i) or {}
        zero = torch.zeros(master[n].numel())
        m.append(row["exp_avg"].reshape(-1).float() if "exp_avg" in row else zero)
        v.append(row["exp_avg_sq"].reshape(-1).float() if "exp_avg_sq" in row else zero)
        if "step" in row:
            steps.add(float(row["step"]))
    m, v = torch.cat(m), torch.cat(v)
    g = m / (1 - b1)  # the clipped gradient
    delta = flat(master, names) - flat(s_master, names)
    expected = -(lr / (1 - b1)) * m / (v.sqrt() / math.sqrt(1 - b2) + eps)
    coef = min(1.0, clip / (grad_norm + 1e-6)) if clip else 1.0
    true = g / coef
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_file({"g": true.to(torch.bfloat16).contiguous()}, str(out))
    groups = {}
    offset = 0
    for n in names:
        k = master[n].numel()
        part = true[offset:offset + k]
        key = f"{group_of(n)}.{'A' if 'lora_A' in n else 'B'}"
        groups[key] = groups.get(key, 0.0) + float(part.dot(part))
        offset += k
    return {"step_rel_err": float((delta - expected).norm()) / max(float(expected.norm()), 1e-30),
            "v_rel_err": float((v - (1 - b2) * g * g).norm()) / max(float(v.norm()), 1e-30),
            "steps": sorted(steps), "updates": int(state.get("updates", -1)), "coef": coef,
            "norm": float(true.norm()), "nonzero": int((true != 0).sum()), "elements": int(true.numel()),
            "update_norm": float(delta.norm()), "norm2_by_group": {k: groups[k] for k in sorted(groups)}}


def layout(names, master):
    import torch

    total = sum(master[n].numel() for n in names)
    masks = {g: torch.zeros(total, dtype=torch.bool) for g, _ in GROUPS}
    b_mask, layers, offset = torch.zeros(total, dtype=torch.bool), {}, 0
    for n in names:
        k = master[n].numel()
        group = group_of(n)
        if group in masks:
            masks[group][offset:offset + k] = True
        if "lora_B" in n:
            b_mask[offset:offset + k] = True
        layers.setdefault(layer_of(n), []).append((offset, offset + k))
        offset += k
    return total, masks, b_mask, layers


def analyze(spec_path, out_path):
    import torch
    from safetensors.torch import load_file, save_file

    spec = json.loads(Path(spec_path).read_text())
    s_master, s_state = load_state(spec["start"])
    names = s_state["names"]
    total, masks, b_mask, layers = layout(names, s_master)
    eps = float(spec.get("eps", 1e-8))
    calls = spec["calls"]
    grads = {c["name"]: load_file(c["grad"])["g"].double() for c in calls}  # float64 sums throughout
    if any(g.numel() != total for g in grads.values()):
        raise ValueError("a gradient's length differs from the start state's")

    def combine(chosen):
        weight = sum(c["divisor"] for c in chosen)
        out = torch.zeros(total, dtype=torch.float64)
        for c in chosen:
            out += grads[c["name"]] * (c["divisor"] / weight)
        return out

    # halves over the tasks with both halves (a one-episode task cannot be split; it stays in the full update)
    paired = {t for t in {c["task"] for c in calls} if {c["half"] for c in calls if c["task"] == t} == {0, 1}}
    split = [c for c in calls if c["task"] in paired]
    halves = [combine([c for c in split if c["half"] == h]) for h in (0, 1)]
    full = combine(calls)
    whole = combine(split)
    h0, h1 = halves
    full2 = float(full.dot(full))

    split2 = float(whole.dot(whole))

    def split_row(a, b, ref, mask=None):  # signal share: <a, b> / |ref|^2 (ref = (a + b) / 2 by tokens)
        if mask is not None:
            a, b, ref = a[mask], b[mask], ref[mask]
        w2 = float(ref.dot(ref))
        return {"cos": round(cos(a, b), 6), "signal_share": round(float(a.dot(b)) / w2, 6) if w2 else None,
                "share_of_norm2": round(w2 / split2, 6) if split2 else None}

    ua, ub = h0 / (h0.abs() + eps), h1 / (h1.abs() + eps)
    support = (h0.abs() >= eps) & (h1.abs() >= eps)
    report = {"elements": total, "b_elements": int(b_mask.sum()), "unpaired": sorted({c["task"] for c in calls} - paired),
              "halves": {**split_row(h0, h1, whole), "dot": float(h0.dot(h1)), "norm2": [float(h0.dot(h0)), float(h1.dot(h1))],
                         "split_norm2": float(whole.dot(whole)), "full_norm2": full2,
                         "cos_split_to_full": round(cos(whole, full), 6),
                         "episodes": [sum(c["episodes"] for c in split if c["half"] == h) for h in (0, 1)],
                         "tokens": [sum(c["divisor"] for c in split if c["half"] == h) for h in (0, 1)],
                         "adam_cos": round(cos(ua, ub), 6),
                         "sign_agree": round(float((torch.sign(h0[support]) == torch.sign(h1[support])).float().mean()), 6)
                         if bool(support.any()) else None,
                         "groups": {g: split_row(h0, h1, whole, m) for g, m in masks.items()},
                         "b_only": split_row(h0, h1, whole, b_mask)}}
    rows = []
    for layer, spans in sorted(layers.items()):  # slices are views: sums per span, no index copies
        dot = sum(float(h0[a:b].dot(h1[a:b])) for a, b in spans)
        n0 = sum(float(h0[a:b].dot(h0[a:b])) for a, b in spans)
        n1 = sum(float(h1[a:b].dot(h1[a:b])) for a, b in spans)
        w2 = sum(float(whole[a:b].dot(whole[a:b])) for a, b in spans)
        rows.append({"layer": layer, "cos": round(dot / math.sqrt(n0 * n1), 6) if n0 * n1 else 0.0,
                     "signal_share": round(dot / w2, 6) if w2 else None,
                     "share_of_norm2": round(w2 / split2, 6) if split2 else None})
    report["halves"]["layers"] = rows
    # within a task (its two halves) against across tasks: E<g_c, g_c'> is the shared mean's square either way, the
    # within-task pair adds the task's own direction
    within, across = [], []
    for i, a in enumerate(calls):
        for b in calls[i + 1:]:
            row = {"pair": [a["name"], b["name"]], "cos": round(cos(grads[a["name"]], grads[b["name"]]), 6),
                   "dot": float(grads[a["name"]].dot(grads[b["name"]])), "episodes": [a["episodes"], b["episodes"]]}
            (within if a["task"] == b["task"] else across).append(row)
    tasks = sorted({c["task"] for c in calls})
    task_grads = {t: combine([c for c in calls if c["task"] == t]) for t in tasks}
    report["within_task"] = within
    report["across_task"] = {"pairs": len(across), "cos_mean": round(sum(r["cos"] for r in across) / max(1, len(across)), 6),
                             "dot_mean": sum(r["dot"] for r in across) / max(1, len(across)),
                             "rows": across}
    report["within_task_summary"] = {"pairs": len(within), "cos_mean": round(sum(r["cos"] for r in within) / max(1, len(within)), 6),
                                     "dot_mean": sum(r["dot"] for r in within) / max(1, len(within))}
    report["tasks"] = {t: {"norm2": float(task_grads[t].dot(task_grads[t])),
                           "cos_to_rest": round(cos(task_grads[t], combine([c for c in calls if c["task"] != t])), 6)}
                       for t in tasks}
    rep = spec.get("repeat")
    if rep:
        g_r, g_o = load_file(rep["grad"])["g"].double(), grads[rep["of"]]
        report["repeat"] = {"of": rep["of"], "cos": round(cos(g_r, g_o), 8),
                            "rel_diff": float((g_r - g_o).norm()) / max(float(g_o.norm()), 1e-30)}
    # (b): the full update's first AdamW step (per unit lr), restricted to a scope, unit norm; a random sign control on
    # the same support; the raw gradient (SGD) direction
    step = -(full / (full.abs() + eps))
    directions, scopes = {}, {}
    for scope in SCOPES:
        mask = torch.ones(total, dtype=torch.bool) if scope == "all" else masks[scope]
        d = torch.where(mask, step, torch.zeros(()))
        n = float(d.norm())
        if not n:
            continue
        directions[scope] = d / n
        scopes[scope] = {"elements": int(mask.sum()), "active": int((d.abs() >= 0.5).sum()), "lr_per_unit_D": 1.0 / n}
    active = step.abs() >= 0.5
    gen = torch.Generator().manual_seed(int(spec.get("seed", 0)))
    rand = (torch.randint(0, 2, (total,), generator=gen) * 2 - 1).double() * active
    directions["random"] = rand / rand.norm()
    scopes["random"] = {"elements": int(active.sum()), "active": int(active.sum()),
                        "lr_per_unit_D": 1.0 / float(rand.norm()), "cos_to_all": round(cos(rand, step), 6)}
    directions["sgd"] = -full / full.norm()
    scopes["sgd"] = {"elements": total, "cos_to_all": round(cos(-full, step), 6)}
    report["scopes"] = scopes
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in directions.items()}, str(spec["directions"]))
    Path(out_path).write_text(json.dumps(report, indent=1))
    return {"halves": {k: v for k, v in report["halves"].items() if k != "layers"},
            "within_task_summary": report["within_task_summary"],
            "across_task": {k: v for k, v in report["across_task"].items() if k != "rows"},
            "repeat": report.get("repeat"), "scopes": scopes, "tasks": report["tasks"]}


def gram(spec_path, out_path):
    """Per-episode gradients (``bank`` mode: one AdamW step per episode, each its own call) as Gram matrices per
    (group, layer) block in float64: every estimator, reward and scope comparison is a quadratic form of these
    (arc2's RL plan, 20 Oct). Also each episode's norm^2 and the repeat call's cosine with its original."""
    import torch
    from safetensors.torch import load_file

    spec = json.loads(Path(spec_path).read_text())
    s_master, s_state = load_state(spec["start"])
    names = s_state["names"]
    blocks, offset = {}, 0
    for n in names:
        k = s_master[n].numel()
        if "lora_B" in n:  # A's first-step gradient is ~0 at B = 1e-12 (snr-v1: 1e-24 against 1e-4)
            blocks.setdefault(f"{group_of(n)}|{layer_of(n)}", []).append((offset, offset + k))
        offset += k
    eps = spec["episodes"]
    rows = torch.stack([load_file(e["grad"])["g"].reshape(-1) for e in eps])  # n x total, bf16
    out = {"n": len(eps), "episodes": [e["name"] for e in eps], "elements": int(rows.shape[1]), "blocks": {}}
    for key, ranges in sorted(blocks.items()):
        idx = torch.cat([torch.arange(a, b) for a, b in ranges])
        x = rows[:, idx].double()
        out["blocks"][key] = (x @ x.T).tolist()
    total = sum(torch.tensor(v, dtype=torch.float64) for v in out["blocks"].values())
    out["norm2"] = total.diagonal().tolist()
    rep = spec.get("repeat")
    if rep:
        a = load_file(rep["grad"])["g"].reshape(-1)
        b = rows[[e["name"] for e in eps].index(rep["of"])]
        out["repeat"] = {"of": rep["of"], "cos": cos(a, b)}
    Path(out_path).write_text(json.dumps(out))
    return {"n": out["n"], "blocks": len(out["blocks"]), "repeat": out.get("repeat"),
            "norm2_mean": float(total.diagonal().mean())}


def scope_mask(scope, total, masks, layers):
    """all, a group (attention / gdn / shared_expert), a layer band La-b, or se+La-b (shared expert in that band)."""
    import torch

    if scope == "all":
        return torch.ones(total, dtype=torch.bool)
    if scope in masks:
        return masks[scope]
    band = scope[4:] if scope.startswith("se+L") else scope[1:] if scope.startswith("L") else None
    if band is None:
        raise ValueError(f"unknown scope {scope}")
    lo, hi = map(int, band.split("-"))
    mask = torch.zeros(total, dtype=torch.bool)
    for layer, spans in layers.items():
        if lo <= layer <= hi:
            for a, b in spans:
                mask[a:b] = True
    return mask & masks["shared_expert"] if scope.startswith("se+") else mask


def combine(spec_path, out_path):
    """Bank directions (arc2 RL plan step 3): per named direction X = sum_i w_i g_i over the bank's per-episode
    gradients (w_i: the estimator's advantage, times the episode's tokens for a token-sum update), its first AdamW step
    -X / (|X| + eps) restricted to a scope, unit norm (make_state's format); ``sgd:<name>`` the raw direction -X; and
    ``random`` random signs on ``random_of``'s active support. Gradients are read once each (memory: one per direction)."""
    import torch
    from safetensors.torch import load_file, save_file

    spec = json.loads(Path(spec_path).read_text())
    s_master, s_state = load_state(spec["start"])
    names = s_state["names"]
    total, masks, _b_mask, layers = layout(names, s_master)
    eps = float(spec.get("eps", 1e-8))
    dirs = spec["directions"]
    sums = {d: torch.zeros(total, dtype=torch.float64) for d in dirs}
    for episode in sorted({e for d in dirs.values() for e in d["weights"]}):
        g = load_file(spec["grads"][episode])["g"].double().reshape(-1)
        if g.numel() != total:
            raise ValueError(f"{episode}: gradient length {g.numel()} != {total}")
        for d, row in dirs.items():
            w = float(row["weights"].get(episode, 0.0))
            if w:
                sums[d] += w * g
    out, info = {}, {}
    for d, row in dirs.items():
        mask = scope_mask(row.get("scope", "all"), total, masks, layers)
        x = sums[d]
        step = torch.where(mask, -(x / (x.abs() + eps)), torch.zeros((), dtype=torch.float64))
        n, nx = float(step.norm()), float(x[mask].norm())
        if not n or not nx:
            raise ValueError(f"direction {d} is empty on scope {row.get('scope', 'all')}")
        out[d] = step / n
        out[f"sgd:{d}"] = torch.where(mask, -x, torch.zeros((), dtype=torch.float64)) / nx
        info[d] = {"scope": row.get("scope", "all"), "episodes": sum(1 for w in row["weights"].values() if w),
                   "elements": int(mask.sum()), "active": int((step.abs() >= 0.5).sum()), "lr_per_unit_D": 1.0 / n,
                   "grad_norm": nx}
    ref = spec.get("random_of")
    if ref:
        active = out[ref].abs() * (1.0 / info[ref]["lr_per_unit_D"]) >= 0.5
        gen = torch.Generator().manual_seed(int(spec.get("seed", 0)))
        rand = (torch.randint(0, 2, (total,), generator=gen) * 2 - 1).double() * active
        out["random"] = rand / rand.norm()
        info["random"] = {"of": ref, "active": int(active.sum()), "lr_per_unit_D": 1.0 / float(rand.norm()),
                          "cos_to_ref": round(cos(out["random"], out[ref]), 6)}
    keys = [d for d in dirs]
    info["cos"] = {f"{a}|{b}": round(cos(sums[a], sums[b]), 6) for i, a in enumerate(keys) for b in keys[i + 1:]}
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}, str(spec["directions_out"]))
    Path(out_path).write_text(json.dumps(info, indent=1))
    return info


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
                own = a["name"] + ("|answer" if a.get("credit") == "answer" else "")  # answer credit: another own part
                g += float(a["w"]) * (noise("shared" + n, p.shape) + 0.8 * noise(task + n, p.shape) + 2.0 * noise(own + n, p.shape))
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
    p = sub.add_parser("gram")
    p.add_argument("spec"), p.add_argument("out")
    p = sub.add_parser("combine")
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
    elif args.command == "gram":
        out = gram(args.spec, args.out)
    elif args.command == "combine":
        out = combine(args.spec, args.out)
    elif args.command == "state":
        out = make_state(args.start, args.directions, args.scope, args.distance, args.out)
    elif args.command == "sha":
        out = {"sha256": sha(args.folder)}
    else:
        out = fake(args)
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
