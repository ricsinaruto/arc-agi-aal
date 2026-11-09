"""kprofL: the decode profile at long contexts, 48 x ~64K, 48 x ~100K and 32 x ~100K real
v9-harness contexts (kprofL-contexts.json: 48 ~64K first, then 48 ~100K; mkctxL.py), output 32,768 forced tokens and a
1,800 s settle so every stream reaches pure decode before the window (rL32 at 3,072 tokens / 240 s found none).
Otherwise as below.

Matched-batch decode profile, v9 vs v8.7b: the exact 9 h kf trees with one
serving-bench trial running arena.kernel_profile on real v9-harness gameplay contexts (SFT a2 records): forced decodes
at 16/32/48/64 streams (~26K prompts), 32 streams at ~90K, a mixed 48-stream point (prefill injector at the arena's
rate), a synthetic 48 point; pure-decode windows (ms/step, tokens/step, CPU per step per process) and a torch trace
(~40 steps) per real point. Arms: v9 (the v1 tree as played), v87b (the ref tree, max_num_seqs 64 / graphs to 192 so
64 streams fit), v9k0 (v9 with qsa_fast and gdn_defer off: the kernels' own share). Usage: mkKprof.py ARM CONTEXTS"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

D = Path("./.claude/worktrees/v84sub/dist")
V9, REF = "b2-v9-p75c268-kf-ho10x11-9h-v1", "b2-v9ref-v87b-kf-ho10x11-9h-v1"
arm, contexts = sys.argv[1], Path(sys.argv[2])
src = {"v9": V9, "v9k0": V9, "v87b": REF}[arm]
out = D / f"b2-kprofL-{arm}-v1"
if out.exists():
    shutil.rmtree(out)
shutil.copytree(D / src, out, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "dataset-upload"))
p = out / "configs/comparison.json"
s = json.loads(p.read_text())
v9 = json.loads((D / V9 / "configs/comparison.json").read_text())
assert len(s["trials"]) == 1 and s["trials"][0]["kind"] == "duck-transfer" and s["server"]["mtp_tokens"] == 2
s["name"] = out.name
srv = s["server"]
if arm == "v87b":
    for k in ("max_num_seqs", "cudagraph_capture_sizes", "max_cudagraph_capture_size"):
        srv[k] = v9["server"][k]
    s["generation_slots"] = v9["generation_slots"]
if arm == "v9k0":
    assert srv["serving_kernels"]["qsa_fast"] is True and srv["stack_kernels"]["gdn_defer"] is True
    srv["serving_kernels"]["qsa_fast"] = False
    srv["stack_kernels"]["gdn_defer"] = False
(out / "configs/bench").mkdir(parents=True, exist_ok=True)
shutil.copyfile(contexts, out / "configs/bench/kprof-contexts.json")
n_ctx = len(json.loads(contexts.read_text())["contexts"])
assert n_ctx == 96, n_ctx
env = {**s["trials"][0]["server_env"], "ARENA_TORCH_PROFILER": "1"}
points = [{"label": "L64x48", "content": "real", "streams": 48, "subset": "first", "profile": True},
          {"label": "L100x48", "content": "real", "streams": 48, "subset": "longest", "profile": True},
          {"label": "L100x32", "content": "real", "streams": 32, "subset": "longest", "profile": True}]
s["trials"] = [{"name": "kprof", "kind": "serving-bench", "competition": False, "wall_seconds": 5400, "server_env": env}]
s["throughput_bench"] = {"prompt_tokens": [], "concurrency": [], "shared_prefix": None, "output_tokens": 32768,
                         "kernel_profile": {"contexts_file": "configs/bench/kprof-contexts.json", "output_tokens": 32768,
                                            "settle_seconds": 240, "window_seconds": 15, "profile_seconds": 4,
                                            "synthetic_streams": 48, "points": points}}
s["wall_seconds"] = 5400
s["chain"] = {"trials": ["kprof"]}
s["notebook_seconds"] = 9000  # setup (~2,450 s) + three long settles (<= 1,800 s each) + margin
s["kaggle_faithful"]["rerun"] = False
s["b200"]["stop"]["wall_seconds"] = 9000
s["roster"] = f"kernel profile {arm}: no games; forced decodes on real v9-harness contexts"
s.setdefault("evidence", {})["kprof"] = (f"from {src}: serving-bench kernel_profile only; arm {arm}"
                                         + ("; max_num_seqs/graphs as v9" if arm == "v87b" else "")
                                         + ("; qsa_fast and gdn_defer off" if arm == "v9k0" else ""))
p.write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((out / "MANIFEST.json").read_text())
m["study"] = out.name
m["files"] = {q.relative_to(out).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(out.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(out / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
print(out)
