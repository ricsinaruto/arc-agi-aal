"""Held-out kf B: the MTP draft head refit on the 15 RL training games only, with the recipe of
the served all-25 head (tp-draft-head-all25v71b-v1, 7 Oct: runs/draft-head/build_draft_head_v85p.py "main", throughput-arm
draft_head.py / draft_head_train.py) against B's served target (r15b mask + the 15-fit INT3 overlay).

    python heldout_draft_head.py CAP_BUNDLE OUT --overlay DATASET --overlay-sha SHA [--steps 400] [--notebook 4800]

CAP_BUNDLE: SFT's B capture bundle (sft-prune/dist/cap-v85p-r15b-v2: v8.5p src, r15b mask, Nebius blocks).
Data: runs/draft-head/data-all25v71b-v1 (the all-25 head's data), train and val filtered to sequences of the 15 games
(each sequence mapped to its source row by exact token prefix); eval (level screen) and xfer (excl5) kept as reports.
Selection is on val (15 games only).
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

TRAIN15 = ("tr87", "sb26", "m0r0", "r11l", "re86", "cn04", "sc25", "cd82", "tn36", "ka59", "dc22", "ls20", "su15",
           "g50t", "s5i5")
DATA = Path("./runs/draft-head/data-all25v71b-v1")
WT = Path(__file__).resolve().parents[3]
BASE_MODEL = "/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def families(split):
    """Source family (game) of every sequence of DATA/<split>.npz, by exact token prefix."""
    rows = json.load(open(json.load(open(DATA / "manifest.json"))["inputs"][split]["path"]))
    z = np.load(DATA / f"{split}.npz")
    off, toks = z["offsets"], z["tokens"]
    arrs = [np.asarray(r["ids"]) for r in rows]
    out = []
    for i in range(len(off) - 1):
        s = toks[off[i]:off[i + 1]]
        exact = {r["family"] for r, a in zip(rows, arrs) if len(a) == len(s) and np.array_equal(a, s)}
        hits = exact or {r["family"] for r, a in zip(rows, arrs) if len(a) > len(s) and np.array_equal(a[:len(s)], s)}
        if not hits:
            raise SystemExit(f"{split}[{i}]: no source row")
        out.append(hits.pop() if len(hits) == 1 else "shared")  # shared: a prompt common to several games (dropped)
    return out


def keep(split, fams, target):
    z = np.load(DATA / f"{split}.npz")
    off = z["offsets"]
    idx = [i for i, f in enumerate(fams) if f in TRAIN15]
    toks = [z["tokens"][off[i]:off[i + 1]] for i in idx]
    loss = [z["loss"][off[i]:off[i + 1]] for i in idx]
    offsets = np.concatenate([[0], np.cumsum([len(t) for t in toks])]).astype(off.dtype)
    np.savez(target / f"{split}.npz", tokens=np.concatenate(toks).astype(z["tokens"].dtype),
             loss=np.concatenate(loss).astype(z["loss"].dtype), offsets=offsets)
    return {"sequences": len(idx), "games": sorted({fams[i] for i in idx})}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cap")
    ap.add_argument("out")
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--overlay-sha", required=True)
    ap.add_argument("--tpa-src", required=True, help="dir with throughput-arm draft_head.py / draft_head_train.py")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--wall", type=int, default=3300, help="trial seconds: dump <= 1,500 + training + export")
    ap.add_argument("--notebook", type=int, default=5700, help="the bootstrap runs the trial only if notebook - 120 - "
                    "setup >= wall + 300: 1,980 s left for setup")
    a = ap.parse_args(argv)
    cap, dst = Path(a.cap), Path(a.out)
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    shutil.copytree(cap, dst, ignore=shutil.ignore_patterns("__pycache__", "dataset-upload", "experiment.ipynb"))
    changed = ["src/arena/draft_head.py", "src/arena/draft_head_train.py"]
    for rel in changed:
        shutil.copyfile(Path(a.tpa_src) / Path(rel).name, dst / rel)
    data = dst / "draft-head/data"
    data.mkdir(parents=True)
    filtered = {}
    for split in ("train", "val"):
        fams = families(split)
        filtered[split] = keep(split, fams, data)
        filtered[split]["dropped_shared"] = fams.count("shared")
        if filtered[split]["games"] != sorted(TRAIN15):
            raise SystemExit(f"{split}: games {filtered[split]['games']}")
    for split in ("eval", "xfer"):  # reports only (never trained on or selected by)
        shutil.copyfile(DATA / f"{split}.npz", data / f"{split}.npz")
    man = json.load(open(DATA / "manifest.json"))
    man["heldout_filter"] = {"train_val_games": list(TRAIN15), "kept": filtered,
                             "note": "eval (level screen) and xfer (excl5) are report splits only"}
    (data / "manifest.json").write_text(json.dumps(man, indent=1))
    c = json.loads((dst / "configs/comparison.json").read_text())
    base_env = c["trials"][0]["server_env"]
    wall, dump_seconds, post_dump = a.wall, 1500, 900
    # named duck-transfer: the Nebius b200-run test runs --trial duck-transfer (the capture bundles do the same)
    c["trials"] = [{"name": "duck-transfer", "kind": "draft-head", "competition": False, "wall_seconds": wall,
                    "server_env": {k: base_env[k] for k in ("ARENA_STALL_DUMP", "ARENA_GPU_MEMORY_UTILIZATION",
                                                            "ARENA_READY_GUARD", "ARENA_SETUP_TIMEOUTS") if k in base_env}}]
    for key in ("expert_quant", "dense_capture", "server_patch", "study_patch"):
        c.pop(key, None)
    srv = c["server"]
    srv.pop("expert_capture", None)
    srv.update(mtp_tokens=1, expert_overlay={"dataset": a.overlay, "manifest_sha256": a.overlay_sha, "check": "sha256"},
               serving_kernels={"mamba_ssm_cache_dtype": "bfloat16", "moe_backend": "humming", "draft_experts": "nvfp4",
                                "hc_fusion": False, "topk_sampler": False, "expert_int3": 16},
               draft_head={"dump": True, "overlay": None})
    trainer = {"steps": 3, "lr": 5e-5, "warmup": 20, "max_anchors": 6144, "step_weights": [1.0, 0.8, 0.6],
               "experts": "nvfp4", "alt_experts": "bf16", "draft_vocab": "src/arc_infra/serving_draft_vocab.json",
               "save_accept": True, "train_steps": a.steps, "eval_every": 100, "eval_subset": 13,
               "checkpoint_every": 100, "reserve_seconds": 420, "select_split": "val",
               "report_splits": ["eval", "val", "xfer"], "curve_stride": 6, "draft_splits": ["eval", "xfer"]}
    splits = ["eval", "val", "xfer", "train"]
    sizes = {}
    for split in splits:
        z = np.load(data / f"{split}.npz")
        sizes[split] = {"sequences": len(z["offsets"]) - 1, "tokens": int(len(z["tokens"])),
                        "generated_tokens": int(z["loss"].sum())}
    c["draft_head"] = {"data": "draft-head/data", "dump_seconds": dump_seconds, "verify_sequences": 0, "min_free_gib": 12,
                       "splits": splits, "draft_splits": ["eval", "xfer"], "model_dir": BASE_MODEL, "trainer": trainer,
                       "sizes": sizes, "build": "heldout-r15b", "post_dump_seconds": post_dump,
                       "recipe": "tp-draft-head-all25v71b-v1 (build_draft_head_v85p.py main), train/val on the 15"}
    c["model_datasets"] = [a.overlay]
    c["b200"]["stop"]["wall_seconds"] = wall
    c.update(name=dst.name, wall_seconds=wall, notebook_seconds=a.notebook, kernel_timeout=a.notebook + 60,
             harness_selection="draft-head fine-tune on the 15 against B's served target (no agent)")
    (dst / "configs/comparison.json").write_text(json.dumps(c, indent=2) + "\n")
    m = json.loads((dst / "MANIFEST.json").read_text())
    m["files"] = {k: v for k, v in m["files"].items() if (dst / k).is_file()}
    for f in [dst / "configs/comparison.json", *(dst / rel for rel in changed), *sorted((dst / "draft-head").rglob("*"))]:
        if f.is_file():
            m["files"][str(f.relative_to(dst))] = sha(f)
    m["study"] = dst.name
    m["git_commit"] = subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (dst / "MANIFEST.json").write_text(json.dumps(m, indent=2))
    bad = [k for k, v in m["files"].items() if not (dst / k).is_file() or sha(dst / k) != v]
    print(json.dumps({"out": str(dst), "sizes": sizes, "kept": filtered, "bad": bad, "wall": wall}))


if __name__ == "__main__":
    main()
