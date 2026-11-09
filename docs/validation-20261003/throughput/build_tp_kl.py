"""Throughput arm (6 October): next-token KL of the matrix rows' expert weights on dev contexts (vs BF16 offline).

kernel-quality study (arena.kernel_gate, Modal test expert-quant-eval) on inputs-tuning-v1's 10 teacher-forced dev
sequences (ls20/re86, outside every dev calibration set): each trial serves one matrix row's exact server block
(dense/serving/stack kernels, overlay, prune) with MTP off; teacher forcing only (no greedy, no GSM8K). The BF16
reference is a separate TP job (scripts/modal_bf16_teacher.py) on the same token ids.

argv: output [--autoround]   (default: plain NVFP4 reference + GPTQ-ARC t0, int3 t0, t37dev int3 gates)
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/arena")]
from build_arena import build, manifest  # noqa: E402

INPUTS = "./runs/expert-quant/inputs-tuning-v1"
QUALITY = {"greedy": False, "gsm8k": False, "report_only": True}
out = Path(sys.argv[1])
AUTOROUND = "--autoround" in sys.argv
REFERENCE_BUNDLE = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--reference-bundle=")), None)
T37ALL = "--t37all" in sys.argv  # v7 submission row: prune-t37-all mask + its int3 overlay (reference only)


def row_env(bundle):
    study = json.loads((ROOT / "dist" / bundle / "configs/comparison.json").read_text())
    server, trial_env = study["server"], study["trials"][0]["server_env"]
    serving = {**server["serving_kernels"], "draft_experts": None, "draft_vocab": None}
    env = {"ARENA_DENSE_KERNELS": json.dumps(server["dense_kernels"]),
           "ARENA_SERVING_KERNELS": json.dumps(serving),
           "ARENA_STACK_KERNELS": json.dumps(server["stack_kernels"]),
           "ARENA_EXPERT_OVERLAY": trial_env.get("ARENA_EXPERT_OVERLAY") or (
               json.dumps(server["expert_overlay"]) if server.get("expert_overlay") else "off"),
           "ARENA_EXPERT_PRUNE": json.dumps(server["expert_prune"]) if server.get("expert_prune") else "off"}
    if serving.get("expert_int3"):
        if env["ARENA_EXPERT_OVERLAY"] == "off":
            raise ValueError(f"{bundle}: int3 serving requires an explicit overlay")
        env["ARENA_SETUP_TIMEOUTS"] = "2700,3600"
    return env, server.get("expert_prune")


rows = {"kl-gptq0": "tp-p5-gptq0-v1", "kl-int3all": "tp-p5-int3all-v1", "kl-t37dev": "tp-p5-t37dev-v1"}
ref_bundle = REFERENCE_BUNDLE or ("tp-p5-autoround-v1" if AUTOROUND else "tp-p5-t37dev-v1" if T37ALL else "tp-p5-plain-v2")
ref_env, ref_mask = row_env(ref_bundle)
candidates, masks = {}, []
if T37ALL:
    ref_env["ARENA_EXPERT_OVERLAY"] = json.dumps({"dataset": "richardcsaky/arc-expert-int3-g16-t37-all-v1",
                                                  "manifest_sha256": "796efcdd75962dd4e24cb3c7d2bc94518e9f886d2bfdd467248fb63c9f44a73e",
                                                  "check": "size"})
    ref_env["ARENA_EXPERT_PRUNE"] = json.dumps({"mask": "configs/expert-prune/prune-t37-all.json",
                                                "sha256": "529677a0c3e3768f783e4831c3a591a1f3ba9cfab59c1cc3c976c65dd2cd3fb3"})
elif not AUTOROUND and not REFERENCE_BUNDLE:
    for name, bundle in rows.items():
        candidates[name], mask = row_env(bundle)
        if mask:
            masks.append((bundle, mask["mask"]))
if "--compare-marlin" in sys.argv:
    if not REFERENCE_BUNDLE:
        raise ValueError("--compare-marlin needs --reference-bundle")
    marlin = json.loads(ref_env["ARENA_SERVING_KERNELS"])
    marlin.update(moe_backend="marlin", expert_int3=None)
    candidates["same-weights-marlin"] = {**ref_env, "ARENA_SERVING_KERNELS": json.dumps(marlin)}
output, study = build(out.name, output=out, dependencies=True, paired_30m=True, expert_quant=INPUTS, expert_quant_phase="evaluate",
                      expert_quant_candidates=candidates)
for trial in study["trials"]:
    trial["quality"] = dict(QUALITY)
    if trial["role"] == "reference":
        trial["server_env"].update(ref_env)
        trial["name"] = "eq-ref"
if T37ALL:
    study.update(notebook_seconds=3240, kernel_timeout=3300)  # Humming start + overlay materialisation
    (output / "configs/expert-prune").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "data/expert_prune/prune-t37-all.json", output / "configs/expert-prune/prune-t37-all.json")
if REFERENCE_BUNDLE:
    study.update(notebook_seconds=3240, kernel_timeout=3300)
    if ref_mask:
        masks.append((ref_bundle, ref_mask["mask"]))
for bundle, mask in masks:
    (output / mask).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "dist" / bundle / mask, output / mask)
if AUTOROUND:
    from arena.research_checkpoint import AUTOROUND_REPO, AUTOROUND_REVISION

    study["research_checkpoint"] = {"repo": AUTOROUND_REPO, "revision": AUTOROUND_REVISION,
                                    "path": "/assets/research-20260930/intel-autoround-w4a16"}
    # research_checkpoint ships its serving.py / inventory with the bundle (as tp-p5-autoround-v1)
    for rel in ("src/arena/research_checkpoint.py", "src/arena/models/intel-autoround-w4a16.json"):
        if not (output / rel).is_file():
            raise FileNotFoundError(rel)
(output / "configs/comparison.json").write_text(json.dumps(study, indent=2) + "\n")
identity = json.loads((output / "MANIFEST.json").read_text())
identity["files"] = manifest(output)
(output / "MANIFEST.json").write_text(json.dumps(identity, indent=2) + "\n")
import runpy  # noqa: E402

runpy.run_path(str(output / "bootstrap.py"))["verify"](output)
print(json.dumps({"output": str(output), "trials": [(t["name"], t["role"]) for t in study["trials"]],
                  "notebook_seconds": study["notebook_seconds"], "server_mtp": study["server"]["mtp_tokens"]}))
