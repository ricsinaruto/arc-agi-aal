"""Arm bundle for the merge read: the v8.2 level-bench build with v8.5p serving (DKb: BF16 dense, BF16 indexer keys;
t37 all25 mask) and the arm's expert overlay. usage: merge_arm.py BASE OUT DATASET MANIFEST_SHA [SEED_BASE=5000] (run from
the rl0 root)"""
import hashlib, json, runpy, shutil, sys
from pathlib import Path

base, out, dataset, msha = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
seed_base = int(sys.argv[5]) if len(sys.argv) > 5 else 5000  # 5000: l4's held-out eval seeds; transfer check 7000
MASK_SRC = Path(__file__).resolve().parents[2] / "data/expert_prune/prune-t37-int3-requant-all25v71b-v1.json"
MASK_SHA = "43dd9b2f9d36d0537aa73dfa0d65cfcca172698a8ad7a4eba829646fae3a006b"
assert hashlib.sha256(MASK_SRC.read_bytes()).hexdigest() == MASK_SHA
if out.exists():
    raise SystemExit(f"{out} exists")
shutil.copytree(base, out)
for old in (out / "configs/expert-prune").glob("*.json"):
    old.unlink()
shutil.copyfile(MASK_SRC, out / "configs/expert-prune" / MASK_SRC.name)
path = out / "configs/comparison.json"
study = json.loads(path.read_text())
server = study["server"]
server["dense_kernels"]["int8_weights"] = []          # v8.5p = v8.2 + DKb
server["serving_kernels"]["indexer_fp8"] = False
server["expert_prune"] = {"mask": f"configs/expert-prune/{MASK_SRC.name}", "sha256": MASK_SHA}
server["expert_overlay"] = {"dataset": dataset, "manifest_sha256": msha, "check": "size"}
study["model_datasets"] = [dataset]
study["level_bench"]["seed_base"] = seed_base         # paired between arms
sys.path[:0] = [str(Path.cwd() / "src"), str(Path.cwd() / "scripts"), str(Path.cwd() / "scripts/arena")]
from arena.level_bench import settings  # noqa: E402
study["level_bench"] = settings(study["level_bench"])
study["name"] = out.name
study["merge_read"] = {"arm_overlay": dataset, "serving": "v8.5p (v8.2 level bench + DKb + t37 all25 mask)",
                       "base_build": base.name}
path.write_text(json.dumps(study, indent=2) + "\n")
from build_arena import manifest  # noqa: E402
identity = json.loads((out / "MANIFEST.json").read_text())
identity["files"] = manifest(out)
(out / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
runpy.run_path(str(out / "bootstrap.py"))["verify"](out)
print(json.dumps({"out": str(out), "overlay": server["expert_overlay"], "mask": server["expert_prune"],
                  "int8_weights": server["dense_kernels"]["int8_weights"], "indexer_fp8": server["serving_kernels"]["indexer_fp8"],
                  "seed_base": study["level_bench"]["seed_base"], "passes": study["level_bench"]["passes"]}))
