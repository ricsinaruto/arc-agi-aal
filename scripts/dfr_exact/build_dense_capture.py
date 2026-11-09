#!/usr/bin/env python3
"""Calibration-capture bundle (throughput arm, 20 Oct; owner items 1 and 3): a served v8.5p-family bundle turned into
one expert-quant capture run that persists, for the excl5 calibration games,
- the MoE inputs and served routing of all 48 layers (calib: fp8 rows at calib_stride; eval: bf16, eval_max_rows),
  as the 10 Oct all25 capture did (input to the 3-bit-from-BF16 GPTQ fit, item 3);
- the inputs of the dense projections gdn_in_proj_qkvz / qsa_qkv ("in") and out_proj / o_proj ("out") of every layer
  (fp8/fp8_dense calibration, item 1; scripts/dfr_exact/fp8/dense_capture.py).
src/arena/expert_quant.py comes from the arc2 capture bundle (capture_only + persist_calibration_capture, 13 Oct).

Usage: build_dense_capture.py --base dist/sabl-v85p-x5-c100-150k-v1 --out dist/cap-v85p-x5-dense-v1
         --inputs ./runs/expert-quant/inputs-excl5-v3
         --quant-src ../arc2/dist/arc2-quant-capture-c240-k256-v1/src/arena/expert_quant.py [--notebook-seconds 5400]
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "fp8"))
import dense_capture  # noqa: E402

SETTINGS = {"calib_stride": 32, "eval_stride": 16, "eval_max_rows": 12288, "concurrency": 3, "eval_rows": 4096,
            "percdamp": 0.01, "min_rows": 64, "batch": 128, "scale_search": [1.0],
            "bf16_dir": "/assets/expert-bf16/de4b8e4d43b917e7706784d8bb445c9af86a3540", "gpu_reserve_seconds": 1500,
            "overlay_dir": "/tmp/unused", "prior_rows": 4096, "act_order": True, "int3_group": 16, "omit_pruned": True,
            "phase": "calibrate", "eval_concurrency": 1, "reset_eval_prefix_cache": True, "capture_only": True,
            "persist_calibration_capture": True}  # the 10 Oct all25 capture's settings


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "MANIFEST.json" and p.suffix != ".ipynb"
            and p.name not in ("kernel-metadata.json", "build-manifest.json") and "dataset-upload" not in p.parts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--quant-src", required=True)
    ap.add_argument("--notebook-seconds", type=int, default=5400)
    ap.add_argument("--dense-max-rows", type=int, default=65536)
    ap.add_argument("--dense-calib-stride", type=int, default=0, help="dense rows every Nth token in the calib pass "
                    "(0: the MoE calib stride)")
    a = ap.parse_args()
    base, out, inputs = Path(a.base), Path(a.out), Path(a.inputs)
    if out.exists():
        raise SystemExit(f"{out} exists")
    shutil.copytree(base, out, symlinks=True, ignore=shutil.ignore_patterns("dataset-upload", "*.ipynb", "source.zip"))
    shutil.copyfile(a.quant_src, out / "src/arena/expert_quant.py")
    dense_capture.patch_persist(out / "src/arena/expert_quant.py")
    patched = dense_capture.apply(out / "src")
    folder = out / "configs/expert-quant"
    folder.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in ("calib.json", "eval.json", "selection.json"):
        shutil.copyfile(inputs / name, folder / name)
        digests[name] = sha(folder / name)
    calib = json.loads((folder / "calib.json").read_text())
    families = sorted({r["family"] for r in calib})
    p = out / "configs/comparison.json"
    s = json.loads(p.read_text())
    if s["server"].get("expert_probe"):
        raise SystemExit("expert_probe and expert_capture patch the same runner")
    env = dict(s["trials"][0].get("server_env") or {})
    env.update(ARC_DENSE_CAPTURE="1", ARC_DENSE_CAPTURE_MAX_ROWS=str(a.dense_max_rows))
    if a.dense_calib_stride:
        env["ARC_DENSE_CAPTURE_CALIB_STRIDE"] = str(a.dense_calib_stride)
    s["trials"] = [{"name": "duck-transfer", "kind": "expert-quant", "phase": "calibrate", "competition": False,
                    "wall_seconds": 60, "server_env": env}]
    s["expert_quant"] = {**SETTINGS, "inputs": {"folder": "configs/expert-quant", "source": inputs.name,
                                                "sha256": digests, "eval_families": families,
                                                "holdout_excluded": "the 20 c100 games (excl5 fit)"}}
    s["server"]["expert_capture"] = True
    s["notebook_seconds"] = int(a.notebook_seconds)
    s["name"] = out.name
    s["dense_capture"] = {"base": base.name, "base_manifest_sha256": sha(base / "MANIFEST.json"),
                          "patched": patched, "expert_quant_src": str(a.quant_src),
                          "dense_max_rows": a.dense_max_rows, "dense_calib_stride": a.dense_calib_stride,
                          "persist": "rename on one filesystem", "families": families}
    p.write_text(json.dumps(s, indent=2))
    ident = json.loads((out / "MANIFEST.json").read_text())
    ident["files"] = manifest(out)
    (out / "MANIFEST.json").write_text(json.dumps(ident, indent=2))
    print(json.dumps({"out": str(out), "calib_rows": len(calib), "calib_tokens": sum(len(r["ids"]) for r in calib),
                      "families": families, "patched": patched}))


if __name__ == "__main__":
    main()
