"""Layers-only decode bench: p75 c613 INT8 dense, kprofC's 50 ~100K contexts, base / d8 / d16, graph
list off and on, base repeated, one VM. Arms differ by server_env: ARENA_RC_OVERRIDES (new, bundle only: research_checkpoint
keys), ARENA_EXPERT_PRUNE, ARENA_CUDAGRAPH_CAPTURE_SIZES. bf16 = dense format arm, kprofD lost it. Output 14,000 and kv_margin .995 in every arm: kprofD bf16 at output 12,000 and .96
had its 50th stream held at the gate (49 x 49 = 2,401 of 2,442) until a stream ended; output must outlast prefill + window
(~11,600 tokens on the first stream). Gate only; real KV stays under the pool. Steplog in every arm, no drains.
At 50 streams decode is 150 tokens: the old list rounds it to 160 (PIECEWISE), the list of multiples of 3 to 162 (FULL)."""
import hashlib, json, re, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve()
D = HERE.parents[3] / "dist"
SPLIT = sys.argv[1]  # two VMs, each with its own base; off: base, bf16, d8, d16, base repeat; on: base, d8, d16
SRC, OUT = D / "b2-kprofC-anchor-v1", D / f"b2-kprofE-{SPLIT}-v1"
SIZES = [3, 6, 9, 18, 33, 48, 60, 66, 72, 81, 90, 96, 105, 114, 129, 144, 162, 177, 192]
WALL = 1500
shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(SRC, OUT, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "dataset-upload"))
p = OUT / "configs/comparison.json"
s = json.loads(p.read_text().replace(SRC.name, OUT.name))
assert s["server"]["max_cudagraph_capture_size"] == SIZES[-1] and s["server"]["mtp_tokens"] == 2
assert 150 not in s["server"]["cudagraph_capture_sizes"] and 160 in s["server"]["cudagraph_capture_sizes"] and 162 in SIZES


def sub(path, a, b):
    t = path.read_text()
    assert t.count(a) == 1, a
    path.write_text(t.replace(a, b))


# research_checkpoint keys per trial
sub(OUT / "src/arena/research_checkpoint.py", "def configure(recipe, working, spec, expert_prune=None):\n",
    'def configure(recipe, working, spec, expert_prune=None):\n'
    '    spec = {**spec, **json.loads(__import__("os").environ.get("ARENA_RC_OVERRIDES") or "{}")}\n')
# mkKprofC cut's two edits: the mask's drop_layers reach the worker; unhealed replies can be empty
sub(OUT / "src/arc_infra/expert_prune_patch.py", "    if mask.get('stall_dump_seconds'):\n        settings['stall_dump_seconds']",
    "    if mask.get('drop_layers'):\n        settings['drop_layers'] = sorted(int(x) for x in mask['drop_layers'])  # depth cut\n"
    "    if mask.get('stall_dump_seconds'):\n        settings['stall_dump_seconds']")
sub(OUT / "src/arena/cache_probe.py", "                    if not reply['choices'][0]['message'].get('content'):\n",
    "                    if not study.get('probe_allow_empty') and not reply['choices'][0]['message'].get('content'):\n")
s["probe_allow_empty"] = True
# steplog as kprofR; each arm's rows moved out at its end
g = OUT / "src/arc_infra/gdn_defer_runtime.py"
t, A = g.read_text(), '    install_runner(GPUModelRunnerV2, "finish_requests")\n'
assert t.count(A) == 1 and "install_steplog" not in t
t = t.replace(A, A + "    install_steplog()\n") + "\n\n" + (HERE.parent / "steplog_runtime.py").read_text().split('"""', 2)[2].lstrip("\n")
compile(t, g.name, "exec")
g.write_text(t)
kp = next(q for q in OUT.rglob("kernel_profile.py"))
sub(kp, '    result["host_end"] = {"gpu": await smi()}\n',
    '    if Path("/tmp/arc_steplog.jsonl").is_file():\n'
    '        shutil.copyfile("/tmp/arc_steplog.jsonl", Path(bench.output) / "steplog-window.jsonl")\n'
    '        open("/tmp/arc_steplog.jsonl", "w").close()\n    result["host_end"] = {"gpu": await smi()}\n')

masks = {}
for name in ("d8", "d16"):
    q = HERE.parent / f"expert-prune-t75-{name}.json"
    (OUT / "configs/expert-prune" / q.name).write_bytes(q.read_bytes())
    masks[name] = {"mask": f"configs/expert-prune/{q.name}", "sha256": hashlib.sha256(q.read_bytes()).hexdigest()}
env = {**s["trials"][0]["server_env"], "ARC_STEPLOG": "/tmp/arc_steplog.jsonl", "ARC_STEPLOG_SYNC": "1000000000"}


def arm(name, cut, listed, dense=None):
    e = dict(env)
    if dense:
        e["ARENA_DENSE_KERNELS"] = json.dumps(dense, sort_keys=True)
    if cut:
        e["ARENA_EXPERT_PRUNE"] = json.dumps(masks[cut], sort_keys=True)
        drop = json.loads((OUT / masks[cut]["mask"]).read_text())["drop_layers"]
        e["ARENA_RC_OVERRIDES"] = json.dumps({"drop_layers": drop, "slim_base": False,
                                              "heal_overlay": f"/assets/flash-input/models/arc-heal/p75-{cut}-c613"}, sort_keys=True)
    if listed:
        e["ARENA_CUDAGRAPH_CAPTURE_SIZES"] = ",".join(map(str, SIZES))
    return {"name": name, "kind": "serving-bench", "competition": False, "wall_seconds": WALL, "server_env": e}


if SPLIT == "off":
    s["trials"] = [arm(*a) for a in (("base", None, 0), ("bf16", None, 0, {**s["server"]["dense_kernels"], "int8_weights": []}),
                                     ("d8", "d8", 0), ("d16", "d16", 0), ("baserep", None, 0))]
    s["trials"][-1]["server_env"]["ARC_STEPLOG_REP"] = "1"
else:
    s["trials"] = [arm(*a) for a in (("base", None, 1), ("d8", "d8", 1), ("d16", "d16", 1))]
n = len(s["trials"])
s["chain"] = {"trials": [t["name"] for t in s["trials"]], "server_restarts": n - 1}
tb = s["throughput_bench"]
tb["output_tokens"] = tb["kernel_profile"]["output_tokens"] = 14000
s["kv_margin"] = 0.995
tb["kernel_profile"]["settle_seconds"] = 1200
s["wall_seconds"] = WALL
s["notebook_seconds"] = s["b200"]["stop"]["wall_seconds"] = 2450 + n * WALL + (n - 1) * 1130
s["nebius_assets"] = s["nebius_assets"] + [f"flash-input/models/arc-heal/p75-{c}-c613" for c in masks]
s["roster"] = f"layers-only bench, list {SPLIT}: no games; 50 forced decodes on real ~100K contexts, one VM"
s["evidence"]["kprofE"] = f"from {SRC.name}: trials, ARENA_RC_OVERRIDES, drop-layer and probe edits as cut, steplog, output 14,000, kv_margin .995"
p.write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((OUT / "MANIFEST.json").read_text().replace(SRC.name, OUT.name))
m["study"] = OUT.name
m["files"] = {q.relative_to(OUT).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(OUT.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(OUT / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
print(OUT)
