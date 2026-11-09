"""INT8 vs BF16 dense and top-k 6 decode bench (+ ~13:50 UTC): kprofC's 50 real ~100K contexts on p75 c613,
one VM: trials int8a, bf16, k6, int8b, server restarted in place on server_env (ARENA_DENSE_KERNELS; ARENA_TEXT_OVERRIDES
is new here; a restart re-extracts and re-patches the runtime). int8b is the noise read. Output 12,000 in every arm: BF16's gate (2,442) does
not fit 50 at the anchor's 16,000 (49 blocks each); output only sets the reservation, the window is pure decode."""
import hashlib, json, shutil
from pathlib import Path

D = Path(__file__).resolve().parents[3] / "dist"
SRC, OUT = D / "b2-kprofC-anchor-v1", D / "b2-kprofD-dense-v1"
shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(SRC, OUT, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "dataset-upload"))
p = OUT / "configs/comparison.json"
s = json.loads(p.read_text().replace(SRC.name, OUT.name))
int8 = s["server"]["dense_kernels"]
assert int8["int8_weights"] and not s.get("research_checkpoint", {}).get("heal_reload")
bf16 = {**int8, "int8_weights": []}
env = s["trials"][0]["server_env"]
WALL = 1500
# k6: num_experts_per_tok per trial; a config field, no tensor changes, so slim_base still holds. Unhealed replies can be
# empty and the bench forces ignore_eos (kprofC cut v2 died in the probe gate).
rc = OUT / "src/arena/research_checkpoint.py"
t, A, B = rc.read_text(), '    overrides = spec.get("text_overrides") or {}  # e.g.', '            if drop or overrides:\n                raise ValueError("slim_base with'
assert t.count(A) == 1 and t.count(B) == 1
t = t.replace(A, '    overrides = {**(spec.get("text_overrides") or {}), **json.loads(__import__("os").environ.get("ARENA_TEXT_OVERRIDES") or "{}")}  # e.g.')
rc.write_text(t.replace(B, '            if drop:\n                raise ValueError("slim_base with'))
cp = OUT / "src/arena/cache_probe.py"
t, A = cp.read_text(), "                    if not reply['choices'][0]['message'].get('content'):\n"
assert t.count(A) == 1
cp.write_text(t.replace(A, "                    if not study.get('probe_allow_empty') and not reply['choices'][0]['message'].get('content'):\n"))
s["probe_allow_empty"] = True
s["trials"] = [{"name": n, "kind": "serving-bench", "competition": False, "wall_seconds": WALL,
                "server_env": {**env, "ARENA_DENSE_KERNELS": json.dumps(k, sort_keys=True), **x}}
               for n, k, x in (("int8a", int8, {}), ("bf16", bf16, {}),
                               ("k6", int8, {"ARENA_TEXT_OVERRIDES": '{"num_experts_per_tok": 6}'}), ("int8b", int8, {}))]
s["chain"] = {"trials": [t["name"] for t in s["trials"]], "server_restarts": 3}
tb = s["throughput_bench"]
tb["output_tokens"] = tb["kernel_profile"]["output_tokens"] = 12000
tb["kernel_profile"]["settle_seconds"] = 1200
s["wall_seconds"] = WALL
s["notebook_seconds"] = 2450 + 4 * WALL + 3 * 1130
s["b200"]["stop"]["wall_seconds"] = s["notebook_seconds"]
s["roster"] = "dense format bench: no games; 50 forced decodes on real ~100K contexts, INT8 / BF16 / top-k 6 / INT8 on one VM"
s["evidence"]["kprofD"] = f"from {SRC.name}: trials + output 12,000 (16,000 in the anchor)"
p.write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((OUT / "MANIFEST.json").read_text().replace(SRC.name, OUT.name))
m["study"] = OUT.name
m["files"] = {q.relative_to(OUT).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(OUT.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(OUT / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
print(OUT)
