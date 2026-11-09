"""Loader key-stream dry-run of an AR heal-overlay bundle (30 Oct: v9b preview v1 died at load with "expert pruning
saw 96 of 48 routers": vLLM reads every tensor of every indexed file, so a router the heal overlay replaces streams
twice, base shard then zz-heal-overlay shard, and v9's expert_prune_runtime counted tensors, not names).
Replays the stream the worker sees: the AR checkpoint's files in load order (sorted names: model-*, then
zz-heal-overlay-*) with every key each file holds (base: the AR index; overlay: the shard headers), through the
bundle's own filter_weights with the bundle's mask, on stand-in tensors of the real router shapes.
Usage: loaderdry.py BUNDLE AR_INDEX_JSON HEADER_DIR (HEADER_DIR: <shard>.json safetensors headers of the overlay)."""
import importlib.util
import json
import re
import sys
from pathlib import Path


EXPERT = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")


class Stub:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def __getitem__(self, rows):
        return Stub((len(rows),) + self.shape[1:])


def run(bundle, ar_index, header_dir):
    bundle = Path(bundle)
    spec = importlib.util.spec_from_file_location("eprt", bundle / "src/arc_infra/expert_prune_runtime.py")
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    study = json.loads((bundle / "configs/comparison.json").read_text())
    mask = json.loads((bundle / study["server"]["expert_prune"]["mask"]).read_text())
    pruned, lmap = mask["pruned"], None
    if mask.get("drop_layers"):  # as expert_prune_runtime.settings
        lmap = rt.layer_map(mask["drop_layers"], len(pruned))
        pruned = {str(lmap[int(k)]): v for k, v in pruned.items() if int(k) in lmap}
    plan = rt.plan(int(mask["experts"]), pruned, min_kept=rt.TOP_K_FLOOR)
    plan["layer_map"] = lmap
    files = {}
    for key, f in json.loads(Path(ar_index).read_text())["weight_map"].items():
        files.setdefault(f, []).append((key, [int(mask["experts"]), 2560]))
    for h in sorted(Path(header_dir).glob("*.json")):
        # expert_overlay (NVFP4 base): same shard names, the overlay's file replaces the base's in the symlink farm
        name = h.name.removesuffix(".json") if study["server"].get("expert_overlay") else "zz-heal-overlay-" + h.name.removesuffix(".json")
        files[name] = [(k, v["shape"]) for k, v in json.loads(h.read_text()).items() if k != "__metadata__"]

    if (study.get("research_checkpoint") or {}).get("slim_base"):
        # arena.ar_slim: pruned base experts leave the index, overlaid keys stream from the overlay only, the base
        # tensors still indexed stream once from the local rest file; base shards are not read
        over = {k for f, ks in files.items() if f.startswith("zz-heal-overlay-") for k, _ in ks}
        pruned = {int(k): {int(e) for e in v} for k, v in mask["pruned"].items()}
        rest = []
        for f in [f for f in files if f.startswith("model-0") or f.startswith("model_extra")]:
            for key, shape in files.pop(f):
                m = EXPERT.match(key)
                if key in over or (m and int(m.group(2)) in pruned.get(int(m.group(1)), ())):
                    continue
                rest.append((key, shape))
        files["model-00000-base-rest.safetensors"] = rest

    def stream():
        for f in sorted(files):
            for key, shape in files[f]:
                yield key, Stub(shape)

    out = list(rt.filter_weights(stream(), plan, log=lambda *a, **k: None))
    experts = {n for n, _ in out if ".mlp.experts." in n and n.startswith("model.language_model.")}
    names = [n for n, _ in out]
    if (study.get("research_checkpoint") or {}).get("slim_base") and len(names) != len(set(names)):
        raise RuntimeError(f"{len(names) - len(set(names))} tensors stream twice")
    return {"tensors_out": len(out), "kept_expert_tensors": len(experts), "files": len(files),
            "streamed": sum(len(v) for v in files.values())}


if __name__ == "__main__":
    try:
        r = run(*sys.argv[1:4])
    except Exception as exc:  # noqa: BLE001
        print(f"LOADERDRY FAILED {Path(sys.argv[1]).name}: {type(exc).__name__}: {exc}")
        sys.exit(1)
    print(f"LOADERDRY OK {Path(sys.argv[1]).name}: {json.dumps(r)}")
