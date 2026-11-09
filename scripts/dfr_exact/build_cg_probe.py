#!/usr/bin/env python3
"""Exact-CG probe bundle (throughput arm, 20 Oct; owner item 10): a served CG bundle (build_v82_ablation --extra CG)
turned into one expert-quant-kind trial (phase cg-exact) that runs the A/B bit-identity test and the op-class probe
of scripts/dfr_exact/cg/cg_probe.py on its own server. No gameplay.

Usage: build_cg_probe.py --base dist/sabl-v85p-x5-cg-c100-base-v1 --out dist/cgx-v85p-x5-probe-v1
         --sequences data/heldout-bal25-v71-v2.json [--per-seq 8] [--m-max 4096] [--notebook-seconds 3600]
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "cg"))
import cg_probe  # noqa: E402

FLAG = "/tmp/arc-cg-eager"


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
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--per-seq", type=int, default=8)
    ap.add_argument("--m-min", type=int, default=33)
    ap.add_argument("--m-max", type=int, default=4096)
    ap.add_argument("--moe-layers", default="0,23,47")
    ap.add_argument("--eager-above", type=int, default=104, help="v8.5p's graph bound: steps above it run eager")
    ap.add_argument("--notebook-seconds", type=int, default=3600)
    ap.add_argument("--server-env", action="append", default=[], help="K=V for the server (e.g. VLLM_BATCH_INVARIANT=1)")
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    s0 = json.loads((base / "configs/comparison.json").read_text())
    if s0["server"].get("max_cudagraph_capture_size") != a.m_max:
        raise SystemExit(f"base graphs end at {s0['server'].get('max_cudagraph_capture_size')}, not {a.m_max} (--extra CG)")
    shutil.copytree(base, out, symlinks=True, ignore=shutil.ignore_patterns("dataset-upload", "*.ipynb", "source.zip"))
    rt = out / "src/arc_infra/dense_kernels_runtime.py"
    text = rt.read_text()
    if "_install_cg_probe" in text or "\ndef enable(module, dtype):" not in text:
        raise SystemExit("dense runtime already probed or its enable() is missing")
    rt.write_text(text + cg_probe.RUNTIME_CODE)
    (out / "src/arena/cg_exact.py").write_text(cg_probe.TRIAL_CODE)
    eq = out / "src/arena/expert_quant.py"
    text = eq.read_text()
    if text.count(cg_probe.EQ_OLD) != 1:
        raise SystemExit("expert_quant.py anchor not found once")
    eq.write_text(text.replace(cg_probe.EQ_OLD, cg_probe.EQ_NEW))
    for path in (rt, out / "src/arena/cg_exact.py", eq):
        compile(path.read_text(), str(path), "exec")
    seqs = json.loads(Path(a.sequences).read_text())
    prompts = cg_probe.prompts_from(seqs, lengths_per_seq=a.per_seq)
    folder = out / "configs/cg-exact"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "prompts.json").write_text(json.dumps(prompts))
    p = out / "configs/comparison.json"
    s = json.loads(p.read_text())
    env = dict(s["trials"][0].get("server_env") or {})
    if a.eager_above not in s["server"]["cudagraph_capture_sizes"]:
        raise SystemExit(f"--eager-above {a.eager_above} is not a capture size")
    env.update(ARC_CG_EAGER_FLAG=FLAG, ARC_CG_EAGER_ABOVE=str(a.eager_above))
    env.update(dict(kv.split("=", 1) for kv in a.server_env))
    s["trials"] = [{"name": "duck-transfer", "kind": "expert-quant", "phase": "cg-exact", "competition": False,
                    "wall_seconds": 60, "server_env": env}]
    s["cg_exact"] = {"flag": FLAG, "probe_reserve_seconds": 900, "base": base.name,
                     "base_manifest_sha256": sha(base / "MANIFEST.json"), "sequences": Path(a.sequences).name,
                     "prompts": len(prompts), "eager_above": int(env["ARC_CG_EAGER_ABOVE"]), "server_env": a.server_env,
                     "probe": {"m_min": a.m_min, "m_max": a.m_max,
                               "moe_layers": [int(x) for x in a.moe_layers.split(",")]}}
    s["notebook_seconds"] = int(a.notebook_seconds)
    s["name"] = out.name
    p.write_text(json.dumps(s, indent=2))
    ident = json.loads((out / "MANIFEST.json").read_text())
    ident["files"] = manifest(out)
    (out / "MANIFEST.json").write_text(json.dumps(ident, indent=2))
    print(json.dumps({"out": str(out), "prompts": len(prompts), "eager_above": env["ARC_CG_EAGER_ABOVE"],
                      "capture_sizes": s["server"]["cudagraph_capture_sizes"]}))


if __name__ == "__main__":
    main()
