"""p50d7 9 h held-out read: b2-k256c268-kf-ho10x11-9h-v3's tree and roster; the diff is the
weights and what they force: overlay + mask, BF16 dense (no INT8 at load), mask drop_layers. The d16 path's prune files
carry the load filter; expert_overlay learns a cut (kept shards only, config.json cut in the farm).
Usage: mkP50d7.py STEP DATASET MANIFEST_SHA MASK.json [VERSION]"""
import hashlib
import json
import runpy
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parents[2] / "dist"
V3 = D / "b2-k256c268-kf-ho10x11-9h-v3"
step, dataset, msha, mask = sys.argv[1:5]
RUN = f"b2-p50d7c{step}-kf-ho10x11-9h-v{sys.argv[5] if len(sys.argv) > 5 else 1}"
OUT = D / RUN
DROP = [2, 9, 10, 11, 17, 21, 44]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sub(rel, *pairs):
    p = OUT / rel
    t = p.read_text()
    for a, b in pairs:
        assert t.count(a) == 1, a
        t = t.replace(a, b)
    p.write_text(t)


if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(V3, OUT, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
shutil.copyfile(D / "b2-kprofC-cut-v3/src/arc_infra/expert_prune_patch.py", OUT / "src/arc_infra/expert_prune_patch.py")
shutil.copyfile(D / "b2-v9b-p75c613-int8-check-v1/src/arc_infra/expert_prune_runtime.py",
                OUT / "src/arc_infra/expert_prune_runtime.py")
sub("src/arena/expert_overlay.py",
    ('def materialise(base_dir, overlay_dir, dest, expected_sha256, *, threads=8, log=print, check="sha256",\n'
     '                expert_prune_sha256=None):',
     'def cut_config(config, drop):\n'
     '    """config.json of the depth-cut model; every PLE layer keeps its index (ple_layer_ids are 1-based)."""\n'
     '    tc = config["text_config"]\n'
     '    keep = [i for i in range(int(tc["num_hidden_layers"])) if i not in set(drop)]\n'
     '    for x in tc.get("ple_layer_ids") or []:\n'
     '        if keep[int(x) - 1] != int(x) - 1:\n'
     '            raise ValueError("depth cut: a PLE layer moved")\n'
     '    tc["layer_types"] = [tc["layer_types"][k] for k in keep]\n'
     '    tc["num_hidden_layers"] = len(keep)\n'
     '    return config\n\n\n'
     'def materialise(base_dir, overlay_dir, dest, expected_sha256, *, threads=8, log=print, check="sha256",\n'
     '                expert_prune_sha256=None, drop_layers=()):'),
    ('    _expected = {f for f in set(weight_map.values()) if "-experts-" in f\n'
     '                 and (m := _re.match(r"layer-0*(\\d+)-", f)) and int(m.group(1)) < _layers}\n',
     '    drop = sorted(int(x) for x in drop_layers or ())\n'
     '    _keep = [i for i in range(_layers) if i not in set(drop)]\n'
     '    _expected = {f for f in set(weight_map.values()) if "-experts-" in f\n'
     '                 and (m := _re.match(r"layer-0*(\\d+)-", f)) and int(m.group(1)) in _keep}\n'),
    ('or sorted(manifest.get("layers") or []) != list(range(_layers))):',
     'or sorted(manifest.get("layers") or []) != _keep):'),
    ('    record = {"format": FORMAT, "base_dir": str(base_dir),',
     '    if drop:\n'
     '        (dest / "config.json").unlink()\n'
     '        (dest / "config.json").write_text(json.dumps(cut_config(json.loads((base_dir / "config.json").read_text()), drop),\n'
     '                                                     indent=2))\n'
     '    record = {"format": FORMAT, "drop_layers": drop, "base_dir": str(base_dir),'))
sub("src/arena/serving.py",
    ("                    expert_prune_sha256=expert_prune_sha256 if expert_prune is not None else None)\n",
     "                    expert_prune_sha256=expert_prune_sha256 if expert_prune is not None else None,\n"
     "                    drop_layers=(expert_prune or {}).get('drop_layers'))\n"))
# base15e, 1 Nov: the exact NVFP4-PLE FP8 gate pins 48 layers and the layer_types tuple; a cut fell to the stock loader
# ("no parameter ngram_embedding.weight_scale"). The gate reads the checkpoint's values, as for num_experts.
sub("src/arc_infra/expert_prune_runtime.py",
    ('    config.arc_unpruned_num_experts = current\n',
     '    config.arc_unpruned_num_experts = current\n'
     '    if p.get("layer_map"):\n'
     '        config.arc_unpruned_num_hidden_layers = p["checkpoint_layers"]\n'
     '        config.arc_unpruned_layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention"\n'
     '                                           for i in range(p["checkpoint_layers"])]\n'),
    ('        _PLAN["layer_map"] = lmap\n',
     '        _PLAN["layer_map"] = lmap\n'
     '        _PLAN["checkpoint_layers"] = len(raw["pruned"])\n'))
sub("src/arc_infra/serving_kernels_patch.py",
    ("""               '            getattr(config, "arc_unpruned_" + field, getattr(config, field, None)) != expected\\n', 1)],\n""",
     """               '            getattr(config, "arc_unpruned_" + field, getattr(config, field, None)) != expected\\n', 1),\n"""
     """              ('        if tuple(getattr(config, "layer_types", ())) != _RADIXARK_NVFP4_LAYER_TYPES:\\n',\n"""
     """               '        # ARC (depth cut): the checkpoint\\'s list.\\n'\n"""
     """               '        if tuple(getattr(config, "arc_unpruned_layer_types", getattr(config, "layer_types", ())))'\n"""
     """               ' != _RADIXARK_NVFP4_LAYER_TYPES:\\n', 1)],\n"""))

s = json.loads((V3 / "configs/comparison.json").read_text().replace(V3.name, RUN))
m = json.loads(Path(mask).read_text())
assert sorted(m["drop_layers"]) == DROP and sorted(int(k) for k in m["pruned"]) == list(range(48))
assert m["experts"] == 512 and {512 - len(v) for v in m["pruned"].values()} == {256}
old = OUT / s["server"]["expert_prune"]["mask"]
new = old.with_name(Path(mask).name)
old.unlink()
shutil.copyfile(mask, new)
s["server"]["expert_prune"].update(mask=new.relative_to(OUT).as_posix(), sha256=sha(new))
s["server"]["expert_overlay"].update(dataset=dataset, manifest_sha256=msha)
s["model_datasets"] = [dataset]
assert s["server"]["dense_kernels"]["int8_weights"] and s["server"]["mtp_tokens"] == 2
s["server"]["dense_kernels"]["int8_weights"] = []
s["fp8_dense"]["note"] = "p50d7: BF16 dense as trained, no INT8 at load, no FP8"
s["heldout"].update(overlay=dataset, mask=new.name)
s["evidence"]["p50d7"] = f"ckpt-{step} of sft-p50d7, 41 layers, pairs with {V3.name}; Nebius only"
s["name"] = RUN
(OUT / "configs/comparison.json").write_text(json.dumps(s, indent=2) + "\n")
b = json.loads((V3 / "MANIFEST.json").read_text().replace(V3.name, RUN))
b["files"] = {q.relative_to(OUT).as_posix(): sha(q) for q in sorted(OUT.rglob("*"))
              if q.is_file() and "__pycache__" not in q.parts and q.name not in ("MANIFEST.json", "build-manifest.json")}
(OUT / "MANIFEST.json").write_text(json.dumps(b, indent=2) + "\n")
runpy.run_path(str(OUT / "bootstrap.py"))["verify"](OUT)
print(OUT, len(s["games"]), dataset, s["focus_oversubscription"]["gate_blocks_per_stream"], new.name)
