"""v9b + INT8 dense load test: check-v2 with k256's INT8 families,
quantized at load from c613's BF16 overlay dense. AR W4A16 experts on marlin and int4 KV stay as v9b."""
import hashlib, json, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE.parents[2] / "dist"
SRC = DIST / "b2-v9b-p75c613-kf-ho10x11-check-v2"
K256 = DIST / "b2-k256c268-kf-ho10x11-9h-v2"
OUT = DIST / "b2-v9b-p75c613-int8-check-v1"
for f in ("dense_kernels_runtime.py", "dense_kernels_patch.py"):
    a, b = SRC / "src/arc_infra" / f, K256 / "src/arc_infra" / f
    assert a.read_bytes() == b.read_bytes(), f"{f} differs from k256's"
shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(SRC, OUT, ignore=shutil.ignore_patterns("__pycache__"))
p = OUT / "configs/comparison.json"
s = json.loads(p.read_text().replace(SRC.name, OUT.name))
k = json.loads((K256 / "configs/comparison.json").read_text())
want = k["server"]["dense_kernels"]["int8_weights"]
assert want and s["server"]["dense_kernels"]["int8_weights"] == []
assert not s["server"]["serving_kernels"].get("expert_int3") and s["server"]["serving_kernels"]["moe_backend"] == "marlin"
s["server"]["dense_kernels"]["int8_weights"] = want
s["wall_seconds"] = 900
s["fp8_dense"]["note"] = "v9b INT8 load test: dense_kernels.int8_weights as k256, quantized at load from the overlay's BF16 dense"
s.setdefault("evidence", {})["int8_check"] = "from check-v2: server.dense_kernels.int8_weights only (wall 900 s)"
p.write_text(json.dumps(s, indent=2) + "\n")
m = json.loads((OUT / "MANIFEST.json").read_text().replace(SRC.name, OUT.name))
m["files"] = {q.relative_to(OUT).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest() for q in sorted(OUT.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts and q.name != "MANIFEST.json"}
(OUT / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
print(OUT, want)
