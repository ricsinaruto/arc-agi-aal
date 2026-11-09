"""RSI gen 0: turn a --draft-head build into b's locked draft-head job (tp-draft-head-all25v71b-v1, 7 Oct) on the
fixed-10 refit: our INT3 overlay + mask, humming, MTP 1, b's trainer settings, the fixed-10 draft vocab.
usage: dh_bundle.py BUNDLE --overlay owner/dataset --overlay-manifest SHA --mask FILE --vocab FILE"""
import argparse, hashlib, json, shutil
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


a = argparse.ArgumentParser()
a.add_argument("bundle")
a.add_argument("--data", required=True)
a.add_argument("--overlay", required=True)
a.add_argument("--overlay-manifest", required=True)
a.add_argument("--mask", required=True)
a.add_argument("--vocab", required=True)
a.add_argument("--notebook-seconds", type=int, default=5640)  # b: function timeout 5,700
a = a.parse_args()
out = Path(a.bundle)
for name in ("train.npz", "val.npz", "eval.npz", "xfer.npz", "manifest.json"):
    shutil.copyfile(Path(a.data) / name, out / "draft-head/data" / name)
mask_rel = f"configs/expert-prune/{Path(a.mask).name}"
(out / mask_rel).parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(a.mask, out / mask_rel)
vocab = json.loads(Path(a.vocab).read_text())
vocab["fit"], vocab["name"] = vocab.get("name", "fixed10"), "arena16k"
(out / "src/arc_infra/serving_draft_vocab.json").write_text(json.dumps(vocab))
cfg = out / "configs/comparison.json"
s = json.loads(cfg.read_text())
s["server"].update(mtp_tokens=1, max_num_seqs=25,
                   serving_kernels={"mamba_ssm_cache_dtype": "bfloat16", "moe_backend": "humming",
                                    "draft_experts": "nvfp4", "expert_int3": 16},
                   expert_prune={"mask": mask_rel, "sha256": sha(out / mask_rel)},
                   expert_overlay={"dataset": a.overlay, "manifest_sha256": a.overlay_manifest, "check": "size"},
                   draft_head={"dump": True, "overlay": None})
s["model_datasets"] = [a.overlay]
s["trials"] = [{"name": "draft-head", "kind": "draft-head",
                "server_env": {"ARENA_KV_CACHE_MEMORY_BYTES": "8053063680", "ARENA_READY_GUARD": "180,2700",
                               "ARENA_SETUP_TIMEOUTS": "2400,3000"}}]
m = json.loads((Path(a.data) / "manifest.json").read_text())
s["draft_head"] = {
    "data": "draft-head/data", "dump_seconds": 1500, "verify_sequences": 0, "min_free_gib": 12,
    "post_dump_seconds": 900, "draft_splits": ["eval", "xfer"],
    "model_dir": "/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1",
    "trainer": {"steps": 3, "lr": 5e-05, "warmup": 20, "max_anchors": 6144, "step_weights": [1.0, 0.8, 0.6],
                "experts": "nvfp4", "alt_experts": "bf16", "draft_vocab": "src/arc_infra/serving_draft_vocab.json",
                "save_accept": True, "train_steps": 700, "eval_every": 100, "eval_subset": 13,
                "checkpoint_every": 100, "reserve_seconds": 420, "select_split": "val",
                "report_splits": ["eval", "val", "xfer"], "curve_stride": 6, "draft_splits": ["eval", "xfer"]},
    "splits": {k: {kk: v[kk] for kk in ("sequences", "tokens", "generated_tokens", "calls", "games")}
               for k, v in m["splits"].items()},
    "build": "rsi2 gen0 fixed10 (b's settings; step 300 is served)"}
s.update(wall_seconds=a.notebook_seconds - 1580, notebook_seconds=a.notebook_seconds,
         kernel_timeout=a.notebook_seconds + 60)
cfg.write_text(json.dumps(s, indent=2) + "\n")
ident = json.loads((out / "MANIFEST.json").read_text())
ident["files"] = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*"))
                  if p.is_file() and p.suffix != ".ipynb" and "node_modules" not in p.parts  # build_arena.manifest
                  and "dataset-upload" not in p.parts
                  and p.name not in ("MANIFEST.json", "build-manifest.json", "kernel-metadata.json")}
(out / "MANIFEST.json").write_text(json.dumps(ident, indent=2) + "\n")
print(json.dumps({"bundle": str(out), "files": len(ident["files"]), "notebook": a.notebook_seconds}))
