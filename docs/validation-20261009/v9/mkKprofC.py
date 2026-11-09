"""Extreme p75 cut bench: 50 streams held at ~100K, decode step and pool size, anchor vs cut, same
bench on two VMs of one preset. Usage: mkKprofC.py CONTEXTS anchor | mkKprofC.py CONTEXTS cut MODEL_EDITS.json
anchor: the v9b INT8 check tree (p75 c613). cut: same, with SFT's artifact applied by MODEL_EDITS (nested keys merged
into the study); no game plays, so an unhealed model is fine. Output 16,000: at 32,768 the anchor's gate fits only 46."""
import hashlib, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve()
D = HERE.parents[3] / "dist"
SRC = D / "b2-v9b-p75c613-int8-check-v1"
contexts, arm = Path(sys.argv[1]), sys.argv[2]
OUT = D / f"b2-kprofC-{arm}-{'v3' if arm == 'cut' else 'v1'}"
shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(SRC, OUT, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "dataset-upload"))
p = OUT / "configs/comparison.json"
s = json.loads(p.read_text().replace(SRC.name, OUT.name))
assert s["server"]["dense_kernels"]["int8_weights"] and s["trials"][0]["kind"] == "duck-transfer"


def merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            merge(a[k], v)
        else:
            a[k] = v


if arm == "cut":
    edits = json.loads(Path(sys.argv[3]).read_text())
    merge(s, edits)
    mask = Path(sys.argv[3]).with_name(Path(edits["server"]["expert_prune"]["mask"]).name)
    assert hashlib.sha256(mask.read_bytes()).hexdigest() == edits["server"]["expert_prune"]["sha256"]
    shutil.copyfile(mask, OUT / edits["server"]["expert_prune"]["mask"])
    (OUT / "configs/expert-prune/prune-t75-nodrop-excl5-v1.json").unlink()
    # the v9 tree's patch drops the mask's drop_layers (kprofC cut v1 died at load: plan 48 layers, model 32)
    # an unhealed model can answer empty; bench requests force ignore_eos (cut v2 died in the probe gate)
    cp = OUT / "src/arena/cache_probe.py"
    t, A = cp.read_text(), "                    if not reply['choices'][0]['message'].get('content'):\n"
    assert t.count(A) == 1
    cp.write_text(t.replace(A, "                    if not study.get('probe_allow_empty') and not reply['choices'][0]['message'].get('content'):\n"))
    ep = OUT / "src/arc_infra/expert_prune_patch.py"
    t, A = ep.read_text(), "    if mask.get('stall_dump_seconds'):\n        settings['stall_dump_seconds']"
    assert t.count(A) == 1 and "drop_layers" not in t
    ep.write_text(t.replace(A, "    if mask.get('drop_layers'):\n        settings['drop_layers'] = sorted(int(x) for x in mask['drop_layers'])  # depth cut\n" + A))
(OUT / "configs/bench").mkdir(parents=True, exist_ok=True)
shutil.copyfile(contexts, OUT / "configs/bench/kprof-contexts.json")
assert len(json.loads(contexts.read_text())["contexts"]) == 50
env = {**s["trials"][0]["server_env"], "ARENA_TORCH_PROFILER": "1"}
s["trials"] = [{"name": "kprof", "kind": "serving-bench", "competition": False, "wall_seconds": 3000, "server_env": env}]
s["throughput_bench"] = {"prompt_tokens": [], "concurrency": [], "shared_prefix": None, "output_tokens": 16000,
                         "kernel_profile": {"contexts_file": "configs/bench/kprof-contexts.json", "output_tokens": 16000,
                                            "settle_seconds": 1800, "window_seconds": 60, "profile_seconds": 4,
                                            "synthetic_streams": 48,
                                            "points": [{"label": "C100x50", "content": "real", "streams": 50,
                                                        "subset": "first", "profile": True}]}}
s["wall_seconds"] = 3000
s["chain"] = {"trials": ["kprof"]}
s["notebook_seconds"] = 5700  # setup (~2,450 s) + settle <= 1,800 + window + margin
s["kaggle_faithful"]["rerun"] = False
s["b200"]["stop"]["wall_seconds"] = 5700
s["roster"] = f"kernel profile, extreme cut bench, {arm}: no games; 50 forced decodes on real ~100K v9-harness contexts"
s.setdefault("evidence", {})["kprofC"] = f"from {SRC.name}: serving-bench kernel_profile only, arm {arm}"
p.write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((OUT / "MANIFEST.json").read_text().replace(SRC.name, OUT.name))
m["study"] = OUT.name
m["files"] = {q.relative_to(OUT).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(OUT.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(OUT / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
print(OUT)
