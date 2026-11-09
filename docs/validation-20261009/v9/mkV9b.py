"""v9b: the v9 tree (harness, drain jitter, window 151,040, roster, KV gate, no CPU tier, qsa_fast,
gdn_defer) serving the latest p75 checkpoint sft-p75-rest-c268-v1b ckpt-613 in its trained format:
- experts AutoRound sym INT4 g128 (the AR W4A16 checkpoint path: Intel AR base + SFT's arc-heal-overlay-v1 of ckpt-613,
  research_checkpoint.heal_overlay; moe_backend marlin), no refit, no dn4g8;
- dense BF16 (from the heal overlay; no FP8 W8A8, no FP8 calibration, no INT8);
- p75's own MTP block from ckpt-613 (in the heal overlay), mtp_tokens 2, no draft-head overlay;
- PLE FP8 from the NVFP4 base (ARC_PLE_FP8=1, as every AR-path bundle);
- int4 KV and BF16 indexer keys as v9.
The AR loader (research_checkpoint.py + models/intel-autoround-w4a16.json + heal_reload files) is ported from the S-stack
tree (sft-prune qadv2-v87s-*), plus two serving.py edits (research_checkpoint beside expert_prune; the FP8 PLE hook).
Robustness: wd_nogiveup.py + v9robust.py. Streams: the v9 density rule (gbps 47.9, target 60) on the real pool.
Usage: mkV9b.py check|sub   (check: Nebius kf boot + 1 h play; sub: the 9 h Kaggle tree for mkV9bprev.py)"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = Path("./.claude/worktrees")
D = W / "v84sub/dist"
S = W / "sft-prune/dist/qadv2-v87s-p75c268-ho10x6-100k-cpu64-v2"
HEAL = "flash-input/models/arc-heal/p75-qad-c613"
mode = sys.argv[1]
SRC = D / ("b2-v9-p75c268-kf-ho10x11-9h-v1" if mode == "check" else "b2-v9-p75c268-kf-ho10x11-9h-v1-sub")
RUN = "b2-v9b-p75c613-kf-ho10x11-check-v2" if mode == "check" else "b2-v9b-p75c613-kf-ho10x11-9h-v2-sub"
OUT = D / RUN
# expert_prune_runtime.py: the S tree's counts router NAMES (preview v1, 30 Oct: the AR farm streams a router the heal
# overlay replaces twice, base shard then overlay; v9's counted tensors and refused "96 of 48 routers"); its other
# additions (drop_layers) are unused here. loaderdry.py replays that stream on CPU.
PORT = ["src/arena/research_checkpoint.py", "src/arena/models/intel-autoround-w4a16.json",
        "src/arc_infra/heal_reload_patch.py", "src/arc_infra/heal_reload_runtime.py", "src/arena/heal_reload.py",
        "src/arc_infra/expert_prune_runtime.py"]
SERVING_EDITS = [
    ("    if research_checkpoint is not None and (expert_overlay is not None or expert_prune is not None):\n",
     "    if research_checkpoint is not None and expert_overlay is not None:\n"),
]
PLE_HOOK_ANCHOR = None  # set below from the S tree's serving.py


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(SRC, OUT, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "dataset-upload"))
for rel in PORT:
    (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(S / rel, OUT / rel)
# slim AR farm (v2, 30 Oct: preview v2 spent 655 s in weight load reading 73 GiB of base shards for 0.9 GiB used):
# arena.ar_slim drops the pruned base experts from the index and serves the base tensors still indexed from one local
# file; research_checkpoint gets the loaded mask from serving.py and applies it under research_checkpoint.slim_base.
shutil.copyfile(HERE.parents[2] / "src/arena/ar_slim.py", OUT / "src/arena/ar_slim.py")
rcp = OUT / "src/arena/research_checkpoint.py"
rct = rcp.read_text()
for old, new in [
        ('"drop_layers", "heal_overlay", "heal_reload"}', '"drop_layers", "heal_overlay", "heal_reload", "slim_base"}'),
        ("def configure(recipe, working, spec):\n", "def configure(recipe, working, spec, expert_prune=None):\n"),
        ('        (farm / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))\n',
         '        slimmed = None\n'
         '        # arena.ar_slim: no base shard is read at load. Skipped under the bundle check\'s stand-in recipe\n'
         '        # (no mask, empty shards); tested by ar_slim on the real headers and by loaderdry.\n'
         '        if spec.get("slim_base") and extract_runtime is not None:\n'
         '            if drop or overrides:\n'
         '                raise ValueError("slim_base with drop_layers/text_overrides is untested")\n'
         '            from arena.ar_slim import slim\n'
         '            slimmed = slim(index, root, farm, Path(working).resolve() / "ar-slim", expert_prune)\n'
         '        (farm / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))\n'),
        ('"heal_overlay": str(overlay) if overlay else None, "heal_overlay_tensors": replaced}',
         '"heal_overlay": str(overlay) if overlay else None, "heal_overlay_tensors": replaced, "slim_base": slimmed}'),
        ('            "heal_overlay_tensors": built["heal_overlay_tensors"]}',
         '            "heal_overlay_tensors": built["heal_overlay_tensors"], "slim_base": built["slim_base"]}')]:
    assert rct.count(old) == 1, old
    rct = rct.replace(old, new)
compile(rct, "research_checkpoint.py", "exec")
rcp.write_text(rct)
SERVING_EDITS.append(("        result['research_checkpoint'] = configure_checkpoint(recipe, working, research_checkpoint)\n",
                      "        result['research_checkpoint'] = configure_checkpoint(recipe, working, research_checkpoint,\n"
                      "                                                             expert_prune=expert_prune)\n"))
# serving.py: research_checkpoint beside expert_prune, and the S tree's FP8 PLE hook (verbatim block)
sv = (OUT / "src/arena/serving.py").read_text()
for old, new in SERVING_EDITS:
    assert sv.count(old) == 1, old
    sv = sv.replace(old, new)
s_sv = (S / "src/arena/serving.py").read_text()
start = s_sv.index("    _arc_env = getattr(recipe, 'runtime_environment', None)\n")
end = s_sv.index("        recipe.runtime_environment = _arc_ple_fp8_environment\n") + len(
    "        recipe.runtime_environment = _arc_ple_fp8_environment\n")
hook = s_sv[start:end] + "\n"
before = s_sv[:start].rsplit("\n", 2)[-2] + "\n"  # the line the hook follows in the S tree
assert sv.count(before) == 1, before
sv = sv.replace(before, before + hook)
compile(sv, "serving.py", "exec")
(OUT / "src/arena/serving.py").write_text(sv)

p = OUT / "configs/comparison.json"
st = json.loads(p.read_text())
A = json.loads((S / "configs/comparison.json").read_text())
srv = st["server"]
srv.pop("expert_overlay")
sk = srv["serving_kernels"]
assert sk.pop("expert_int3") == 16 and sk["moe_backend"] == "humming" and sk["indexer_fp8"] is False
sk["moe_backend"] = "marlin"
assert sk["qsa_fast"] is True and srv["stack_kernels"]["gdn_defer"] is True and srv["mtp_tokens"] == 2
dk = srv["dense_kernels"]
assert dk["fp8_w8a8"] == ["attn_out", "gdn_in_proj_qkvz", "qsa_qkv"] and dk["int8_weights"] == []
dk.pop("fp8_w8a8")
assert "draft_head" not in srv
rc = json.loads(json.dumps(A["research_checkpoint"]))
rc["heal_overlay"], rc["heal_reload"], rc["drop_layers"] = f"/assets/{HEAL}", False, []
rc["slim_base"] = True
st["research_checkpoint"] = rc
st["model_datasets"] = []
for t in st["trials"]:
    env = t["server_env"]
    for k in ("ARC_FP8_CALIB", "ARC_FP8_BF16", "ARC_EXPERT_INT3_FP4_DOWN", "ARC_EXPERT_INT3_FP4_LAYERS"):
        env.pop(k)
    env["ARC_PLE_FP8"] = "1"
st["nebius_assets"] = [a for a in A["nebius_assets"] if "arc-heal" not in a] + [HEAL]
st["nebius_asset_skip"] = {k: v for k, v in A["nebius_asset_skip"].items() if "datasets" not in k}
st["nebius_boot_disk_gib"] = A["nebius_boot_disk_gib"]
st["fp8_dense"] = {"families": [], "toggle": None, "calib": None, "note": "v9b: BF16 dense"}
st["dn4g8"] = {"note": "v9b: off; AR INT4 g128 experts as QAT-trained"}
st["heldout"] = {**st["heldout"], "overlay": None, "overlay_manifest_sha256": None, "heal_overlay": f"/assets/{HEAL}",
                 "model": "p75 sft-p75-rest-c268-v1b ckpt-613 (AR INT4 g128 QAT, BF16 dense, own MTP)"}
st["name"] = RUN
if mode == "check":
    # Nebius boot + play check: measured blocks, streams, KV, hit after ramp, tok/s, MTP acceptance.
    # The kf rehearsal plays to the envelope end (bootstrap: max_seconds = submission_seconds on a rerun):
    # ~1 h of play after a ~1,500-2,450 s Nebius setup and the 600 s cleanup.
    st["competition"]["submission_seconds"] = 6600
    st["competition"]["preview_seconds"] = 3600
    st["notebook_seconds"] = 6600
st.setdefault("evidence", {})["v9b"] = (
    "v9b = v9 harness/kernels serving p75 ckpt-613 in its trained format (AR INT4 g128 experts, "
    "BF16 dense, own MTP), no refit, no FP8; " + ("Nebius boot + 1 h play check" if mode == "check" else "9 h tree"))
p.write_text(json.dumps(st, indent=2) + "\n")
assert not any(x in p.read_text() for x in ("arc-fp8-calib-p75c268", "requant-dn4g8-p75c268"))
subprocess.run([sys.executable, str(HERE / "wd_nogiveup.py"), str(OUT)], check=True)
subprocess.run([sys.executable, str(HERE / "v9robust.py"), str(OUT)], check=True)
subprocess.run([sys.executable, str(HERE / "carry_live.py"), str(OUT)], check=True)  # prefix hit + active_limit
m = json.loads((OUT / "MANIFEST.json").read_text())
m["study"] = RUN
m["files"] = {q.relative_to(OUT).as_posix(): sha(q) for q in sorted(OUT.rglob("*"))
              if q.is_file() and q.suffix != ".ipynb" and "__pycache__" not in q.parts
              and q.name not in ("MANIFEST.json", "build-manifest.json", "kernel-metadata.json")}
(OUT / "MANIFEST.json").write_text(json.dumps(m, indent=2) + "\n")
print(OUT)
