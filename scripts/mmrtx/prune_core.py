"""Expert pruning shared by the 1-RTX fit arms (dots3, mmrtx; generalised from mmrtx's prune.py): keep each MoE
layer's top experts by routed mass on the FIT sets only, renumber them 0..K-1, slice the router rows and the routing
bias, write the expert count into the config. Tensors are byte copies (any stored format: FP8 + scales, NVFP4, ...).

A model is described by a Spec: expert / router tensor-name regexes (group 1 = layer, group 2 = expert, group 3 = rest
for experts; group 1 = layer for router tensors), the renamed expert name, the config key path of the expert count,
and the layers to prune (others are copied untouched)."""
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Spec:
    expert: str
    router: str
    expert_name: str  # format(layer=, expert=, rest=)
    count_key: tuple  # config path, e.g. ("n_routed_experts",) or ("text_config", "num_local_experts")
    layers: tuple
    drop: str = ""  # tensors to leave out entirely (e.g. a vision tower), regex


DOTS3 = Spec(expert=r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$",
             router=r"^model\.layers\.(\d+)\.mlp\.gate\.(weight|e_score_correction_bias)$",
             expert_name="model.layers.{layer}.mlp.experts.{expert}.{rest}", count_key=("n_routed_experts",),
             layers=tuple(range(1, 46)))
MINIMAX_M3 = Spec(expert=r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(.+)$",
                  router=r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.(gate\.weight|e_score_correction_bias)$",
                  expert_name="language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{rest}",
                  count_key=("text_config", "num_local_experts"), layers=tuple(range(3, 60)))


def fit_mask(mass, layers, k):
    """mass: [L, E] summed fit-set routed mass. k: int or {layer: int}. -> ({layer: sorted kept ids}, info)."""
    import torch
    keep, kept = {}, []
    for layer in layers:
        kk = k[layer] if isinstance(k, dict) else k
        order = mass[layer].double().argsort(descending=True)[:kk]
        keep[layer] = sorted(int(e) for e in order)
        kept.append(float(mass[layer, keep[layer]].sum() / mass[layer].sum().clamp_min(1e-12)))
    t = torch.tensor(kept)
    return keep, {"kept_mass_mean": round(float(t.mean()), 4), "kept_mass_min": round(float(t.min()), 4)}


def prune(src, dst, keep, spec, k_config):
    """Write the pruned checkpoint; returns total bytes. ``k_config``: the expert count written to the config (all
    pruned layers must keep the same count for stock loaders)."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    ex, ro, drop = re.compile(spec.expert), re.compile(spec.router), re.compile(spec.drop) if spec.drop else None
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map, total = {}, 0
    for shard in sorted(set(index["weight_map"].values())):
        out = {}
        with safe_open(str(src / shard), framework="pt") as f:
            meta = f.metadata()
            for name in f.keys():
                if drop and drop.match(name):
                    continue
                m = ex.match(name)
                if m and int(m.group(1)) in keep:
                    layer, e = int(m.group(1)), int(m.group(2))
                    if e in keep[layer]:
                        out[spec.expert_name.format(layer=layer, expert=keep[layer].index(e), rest=m.group(3))] = \
                            f.get_tensor(name)
                    continue
                t = f.get_tensor(name)
                r = ro.match(name)
                if r and int(r.group(1)) in keep:
                    t = t[keep[int(r.group(1))]].contiguous()
                out[name] = t
        if out:
            save_file(out, str(dst / shard), metadata=meta)
        for name, t in out.items():
            weight_map[name] = shard
            total += t.numel() * t.element_size()
        print(f"{shard}: {len(out)} tensors", flush=True)
    (dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total},
                                                                  "weight_map": weight_map}, indent=1))
    cfg = json.loads((src / "config.json").read_text())
    node = cfg
    for key in spec.count_key[:-1]:
        node = node[key]
    node[spec.count_key[-1]] = int(k_config)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") and p.name not in ("model.safetensors.index.json",
                                                                                   "config.json"):
            shutil.copy(p, dst / p.name)
        elif p.is_dir():  # e.g. MiniMax's dspark/ draft head
            shutil.copytree(p, dst / p.name, dirs_exist_ok=True)
    return total
