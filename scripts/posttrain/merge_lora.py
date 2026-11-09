"""Deployment path for RL adapters (merge read, 19 Oct): the pure parts of ``modal_merge_int3.py`` and the read's
pre-registered verdicts (``merge_read.py``), testable without Modal.

PEFT 0.21.1 ParamWrapper (the trainer's): lora_A [r E, in] expert-major rows, lora_B [out, r E] expert-minor columns;
dW_e = scale * B[:, :, e] @ A[e] in the parameter's [E, out, in] layout (= the BF16 experts' gate_up [E, 2I, H] and
down [E, H, I])."""
import json
import os
import random
import statistics

FLASH = {"E": 512, "H": 2560, "INTER": 640, "R": 16}


def adapter_scale(cfg):
    return cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])


def expert_delta(adapter, layer, torch, shapes=FLASH):
    """(dW gate_up [E, 2I, H], dW down [E, H, I]) float32 of one layer's experts, and the adapter's scale."""
    from safetensors import safe_open

    E, H, INTER, R = shapes["E"], shapes["H"], shapes["INTER"], shapes["R"]
    scale = adapter_scale(json.load(open(os.path.join(adapter, "adapter_config.json"))))
    base = f"base_model.model.model.language_model.layers.{layer}.mlp.experts."
    out = []
    with safe_open(os.path.join(adapter, "adapter_model.safetensors"), framework="pt", device="cpu") as fh:
        for stem, (o, i) in ((base + "base_layer.", (2 * INTER, H)), (base, (H, INTER))):
            a, b = fh.get_tensor(stem + "lora_A.weight").float(), fh.get_tensor(stem + "lora_B.weight").float()
            if tuple(a.shape) != (R * E, i) or tuple(b.shape) != (o, R * E):
                raise RuntimeError(f"L{layer} {stem}: lora shapes {tuple(a.shape)} {tuple(b.shape)}")
            out.append(torch.einsum("ore,eri->eoi", b.reshape(o, R, E), a.reshape(E, R, i)) * scale)
    return out[0], out[1], scale


def kept_of(mask, experts=FLASH["E"]):
    """{layer: kept expert ids} of a prune mask ({"pruned": {layer: [ids]}})."""
    return {int(l): sorted(set(range(experts)) - {int(e) for e in ids}) for l, ids in mask["pruned"].items()}


def l2_try(row):
    """(entered, cleared, L2 tokens) of one level-bench try at a lead point: completion_tokens count from the L2 entry;
    a lead_out try (L1 not cleared) never entered L2."""
    if row.get("outcome") == "lead_out" or not row.get("lead_tokens"):
        return False, False, 0
    if row.get("cleared"):
        return True, True, int(row["tokens_to_clear"])
    return True, False, int(row["completion_tokens"])


def per_clear(items):
    """L2 tokens over tries that entered L2 (a censored try counts its spend) / L2 clears."""
    entered = [x for x in items if x[0]]
    clears = sum(1 for x in entered if x[1])
    return sum(x[2] for x in entered) / clears if clears else float("inf")


def ratio_ci(a, b, reps=10000, seed=0):
    """per_clear(a) / per_clear(b) with a 95% bootstrap CI (tries resampled within arm)."""
    rng, vals = random.Random(seed), []
    for _ in range(reps):
        ra, rb = per_clear(rng.choices(a, k=len(a))), per_clear(rng.choices(b, k=len(b)))
        if rb and rb != float("inf") and ra != float("inf"):
            vals.append(ra / rb)
    vals.sort()
    return round(per_clear(a) / per_clear(b), 2), (round(vals[int(0.025 * len(vals))], 2), round(vals[int(0.975 * len(vals))], 2))


def summary(raw):
    items = [l2_try(r) for r in raw]
    entered = [x for x in items if x[0]]
    return {"tries": len(raw), "errors": sum(1 for r in raw if r.get("error")), "entered_L2": len(entered),
            "L2_clears": sum(1 for x in entered if x[1]), "L2_tokens_per_clear": round(per_clear(items)),
            "L2_median_tokens": statistics.median([x[2] for x in entered]) if entered else None,
            "L1_median_tokens": statistics.median([r["lead_tokens"] for r in raw if r.get("lead_tokens") is not None] or [0]),
            "L2_median_actions_cleared": statistics.median([r["actions"] for r in raw if r.get("cleared")] or [0]),
            "outcomes": {o: sum(1 for r in raw if r.get("outcome") == o) for o in sorted({r.get("outcome") for r in raw})}}


def verdicts(raw, path_bar=2.0):
    """Pre-registered (19 Oct): path works = zero / merged tokens per clear >= ``path_bar``; chain OK = the zero /
    shipped 95% CI contains 1. ``raw``: {arm: level-bench attempt rows} with arms among shipped, zero, merged."""
    items = {k: [l2_try(r) for r in v] for k, v in raw.items()}
    out = {k: summary(v) for k, v in raw.items()}
    if "zero" in items and "merged" in items:
        out["zero_over_merged"] = ratio_ci(items["zero"], items["merged"])
        out["path_works"] = out["zero_over_merged"][0] >= path_bar
    if "zero" in items and "shipped" in items:
        out["zero_over_shipped"] = ratio_ci(items["zero"], items["shipped"])
        lo, hi = out["zero_over_shipped"][1]
        out["chain_ok"] = lo <= 1.0 <= hi
    return out
