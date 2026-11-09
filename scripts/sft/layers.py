"""Greedy depth-cut layer choice on one student (port of sft-heal heal_kd.run_layers, greedy drop on
p75). Each decoder layer can be skipped at run time (identity); KL(reference || cut) on the assistant positions of the
excl5 recorded-play windows (reference = the uncut student's top-20 + tail mass; fit rule: excl5 only). One-shot
single-layer scores, then greedy per objective: drop the best layer, re-score the ``lazy`` best remaining candidates on
the cut model, repeat to ``depth``. Objectives: "kl" (KL increase) and "kl_per_cost" (per unit of layer cost, full
attention ``cost_full`` vs GDN 1). Also the KL of given ``sets`` ({name: [layers]}). Writes <out>/layers.json and
<out>/events.jsonl."""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def event(out, **row):
    with open(Path(out) / "events.jsonl", "a") as f:
        f.write(json.dumps({"t": round(time.time(), 1), **row}) + "\n")


def windows_data(path, device, tokens=4096, scored=2048, every=4, which=1):
    """The excl5 windows (heal_kd's selection: index % every == which), the last ``tokens`` ids, the last ``scored``
    rows scored (row t scores token t + 1), assistant spans as a mask."""
    data = []
    for w in json.loads(Path(path).read_text())["windows"]:
        if w["index"] % every != which:
            continue
        ids = torch.tensor(w["ids"][-tokens:], device=device)[None]
        start = max(0, ids.shape[1] - scored)
        asst = torch.zeros(ids.shape[1], dtype=torch.bool)
        off = len(w["ids"]) - ids.shape[1]
        for a, b in w["assistant_spans"]:
            asst[max(0, a - off):max(0, b - off)] = True
        data.append({"ids": ids, "start": start, "asst": asst[start + 1:].to(device)})
    return data


class Skipper:
    """Decoder layers in ``skip`` return their input (the residual stream passes through)."""

    def __init__(self, layers):
        self.skip = set()
        for i, layer in enumerate(layers):
            self._wrap(i, layer)

    def _wrap(self, i, layer):
        orig = layer.forward

        def fwd(hidden_states, *a, **k):
            if i in self.skip:
                return hidden_states
            return orig(hidden_states, *a, **k)
        layer.forward = fwd


def trained_format(model, grid="int4"):
    """The student as trained: experts on the QAT grid (``grid``: "int4" or "int3served"), INT8 dense, int4 KV
    (eval-only) in every forward; the INT8 lm_head (``qat.int8_rows``) goes to ``run``'s ``weight``."""
    import attention
    import qat

    qat.QAT["grid"] = grid
    attention.KVQ["mode"] = "kv4"
    names = qat.install_dense(model)
    qat.install(model)
    mod = qat.routed(model)[0]
    qat.fuse([(getattr(mod, n).detach(), qat.SCALES[id(mod)][n]) for n in qat.NAMES])
    return {"dense_modules": len(names), "fused": bool(qat.QAT["fused"]),
            "calls": lambda: {"kv": attention.KVQ["calls"], "experts": qat.QAT["calls"], "int8": qat.DENSE["calls"]}}


def run(model, data, out, *, depth=8, lazy=6, cost_full=2.0, sets=None, objectives=("kl", "kl_per_cost"), weight=None, calls=None):
    """``model``: an HF multimodal LM (model.model.language_model.layers, model.lm_head) or text LM (model.model.layers).
    ``weight``: the head's weight read instead of lm_head's; ``calls``: () -> counters, written to the report."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    core = model.model
    lm = getattr(core, "language_model", core)
    layers = lm.layers
    cfg = getattr(model.config, "text_config", model.config)
    types = list(getattr(cfg, "layer_types", None) or ["full_attention"] * len(layers))
    head = weight if weight is not None else model.get_output_embeddings().weight
    skipper = Skipper(layers)

    def logp(d):
        h = lm(input_ids=d["ids"], use_cache=False).last_hidden_state[0, d["start"]:-1]
        return torch.log_softmax(F.linear(h, head).float(), dim=-1)

    with torch.no_grad():
        for d in data:
            lp = logp(d)
            top, idx = lp.topk(20, dim=-1)
            d["top"], d["idx"] = top, idx
            d["tail"] = torch.log1p(-top.exp().sum(-1).clamp(max=1 - 1e-6))
            del lp

    def score():
        tot = n = ta = na = 0.0
        with torch.no_grad():
            for d in data:
                lp = logp(d)
                s_top = lp.gather(-1, d["idx"])
                s_tail = torch.log1p(-s_top.exp().sum(-1).clamp(max=1 - 1e-6))
                kl = (d["top"].exp() * (d["top"] - s_top)).sum(-1) + d["tail"].exp() * (d["tail"] - s_tail)
                tot, n = tot + float(kl.sum()), n + kl.numel()
                ta, na = ta + float(kl[d["asst"]].sum()), na + int(d["asst"].sum())
                del lp
        return {"all": round(tot / max(n, 1), 5), "asst": round(ta / max(na, 1), 5)}

    cost = {i: (cost_full if types[i] != "linear_attention" else 1.0) for i in range(len(layers))}
    began = time.time()
    report = {"windows": len(data), "positions": sum(int(d["asst"].numel()) for d in data), "types": types,
              "cost_full": cost_full, "layers": len(layers)}
    skip = skipper.skip
    skip.clear()
    report["uncut"] = score()  # ~0: the reference itself
    event(out, event="layers_uncut", uncut=report["uncut"], seconds=round(time.time() - began, 1))
    single = {}
    for i in range(len(layers)):
        skip.clear()
        skip.add(i)
        single[i] = score()
        if i % 8 == 7:
            event(out, event="layers_progress", scored=i + 1, seconds=round(time.time() - began, 1))
    skip.clear()
    report["single"] = single
    report["single_seconds"] = round(time.time() - began, 1)
    event(out, event="layers_single", order=sorted(single, key=lambda i: single[i]["asst"]), single=single)
    report["sets"] = {}
    for name, members in (sets or {}).items():
        skip.clear()
        skip.update(int(x) for x in members)
        report["sets"][name] = {"layers": sorted(skip), "kl": score()}
        event(out, event="layers_set", name=name, **report["sets"][name])
    for objective in objectives:
        skip.clear()
        chosen, base_kl, last, trail = [], 0.0, {i: single[i]["asst"] for i in single}, []
        div = (lambda i: cost[i]) if objective == "kl_per_cost" else (lambda i: 1.0)
        while len(chosen) < depth:
            rank = sorted((i for i in last if i not in skip), key=lambda i: (last[i] - base_kl) / div(i))
            cands = rank[:lazy]
            fresh = {}
            for i in cands:
                skip.add(i)
                fresh[i] = score()
                skip.discard(i)
                last[i] = fresh[i]["asst"]
            best = min(cands, key=lambda i: (fresh[i]["asst"] - base_kl) / div(i))
            skip.add(best)
            chosen.append(best)
            base_kl = fresh[best]["asst"]
            trail.append({"n": len(chosen), "layer": best, "type": types[best], "kl": fresh[best],
                          "cost": sum(cost[i] for i in chosen)})
            event(out, event="layers_greedy", objective=objective, **trail[-1], seconds=round(time.time() - began, 1))
        report[f"greedy_{objective}"] = trail
        report[f"set_{objective}"] = sorted(chosen)
    skip.clear()
    report["seconds"] = round(time.time() - began, 1)
    report["calls"] = calls() if calls else None
    (out / "layers.json").write_text(json.dumps(report, indent=1))
    event(out, event="done", seconds=report["seconds"])
    return report
