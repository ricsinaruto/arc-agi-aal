"""Held-out kf runs: a Kaggle sub bundle re-pointed at a mask + expert overlay fitted on the 15
RL training games and a roster of the 10 held-out games. Run nebius_kaggle_faithful.py --live-score on OUT after.

    python heldout_bundle.py SRC OUT --mask MASK.json --overlay DATASET --overlay-sha SHA --strict-overlay V87B_SRC
        [--games sp80,wa30,...]

SRC: v84sub dist/arena-flash-competition-v85l1c-sub. --strict-overlay: v87b's src/arena/expert_overlay.py (the strict
every-expert-shard check, 19 Oct), relaxed here to allow extra non-expert shards listed in the index (A's dense p75
shards); an overlay with exactly the expert shards checks as before.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

HELDOUT10 = ("sp80", "wa30", "sk48", "lf52", "bp35", "ft09", "vc33", "lp85", "tu93", "ar25")
TRAIN15 = ("tr87", "sb26", "m0r0", "r11l", "re86", "cn04", "sc25", "cd82", "tn36", "ka59", "dc22", "ls20", "su15",
           "g50t", "s5i5")
RECORD_KEYS = ("v84", "v85", "v85p", "kaggle_load", "kaggle_draft_files", "l1_stack")
SKIP = ("dataset-upload", "experiment.ipynb", "kernel-metadata.json", "__pycache__")
STRICT_OLD = "    if not _layers or set(shards) != _expected or sorted(manifest.get(\"layers\") or []) != list(range(_layers)):\n"
STRICT_NEW = ("    _extra = set(shards) - _expected  # held-out A (28 Oct): non-expert shards in the index may ride along\n"
              "    if (not _layers or not _expected <= set(shards) or any(\"-experts-\" in f for f in _extra)\n"
              "            or sorted(manifest.get(\"layers\") or []) != list(range(_layers))):\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SIZE_OLD = ("    elif check != \"size\":\n"
            "        raise ValueError(f\"unknown overlay check {check!r}\")\n")
SIZE_NEW = SIZE_OLD + ("    else:  # held-out A (28 Oct): extra non-expert shards are hashed even under \"size\"\n"
                       "        for name in sorted(_extra):\n"
                       "            if sha256_file(overlay_dir / name) != shards[name][\"sha256\"]:\n"
                       "                raise RuntimeError(f\"{name}: overlay sha256 differs from its manifest\")\n")


def relaxed_overlay_source(strict_path):
    text = Path(strict_path).read_text()
    for old in (STRICT_OLD, SIZE_OLD):
        if text.count(old) != 1:
            raise SystemExit(f"{strict_path}: anchor not found once: {old[:60]!r}")
    return text.replace(STRICT_OLD, STRICT_NEW).replace(SIZE_OLD, SIZE_NEW)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--mask", required=True)
    ap.add_argument("--overlay", required=True, help="Kaggle dataset ref of the expert overlay (owner/name)")
    ap.add_argument("--overlay-sha", required=True, help="sha256 of its overlay-manifest.json")
    ap.add_argument("--strict-overlay", required=True)
    ap.add_argument("--games", default=",".join(HELDOUT10))
    ap.add_argument("--draft-head", default="", help="dir with overlay.safetensors + overlay-manifest.json: the MTP "
                    "draft head fitted on the 15 (heldout_draft_head.py run), replacing the bundle's")
    a = ap.parse_args(argv)
    src, out = Path(a.src), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    study = json.loads((src / "configs/comparison.json").read_text())
    public = json.loads((src / "configs/duck_public_notebook.json").read_text())["public_game_ids"]
    short = {g.split("-")[0]: g for g in public}
    want = a.games.split(",")
    if sorted(want) != sorted(HELDOUT10) or set(want) & set(TRAIN15) or len(public) != 25:
        raise SystemExit("--games must be the 10 held-out games")
    games = [short[g] for g in want]
    mask = json.loads(Path(a.mask).read_text())
    if mask.get("format") != "arc-expert-prune-v1" or len(mask["pruned"]) != 48:
        raise SystemExit("--mask: not a 48-layer arc-expert-prune-v1 mask")
    srv = study["server"]
    if not srv.get("expert_overlay") or not srv.get("expert_prune"):
        raise SystemExit("SRC serves no expert overlay / mask")
    shutil.copytree(src, out, ignore=shutil.ignore_patterns(*SKIP))
    changed, removed = [], []
    (out / "src/arena/expert_overlay.py").write_text(relaxed_overlay_source(a.strict_overlay))
    changed.append("src/arena/expert_overlay.py")
    old_mask = srv["expert_prune"]["mask"]
    (out / old_mask).unlink()
    removed.append(old_mask)
    new_mask = f"configs/expert-prune/{Path(a.mask).name}"
    shutil.copy(a.mask, out / new_mask)
    changed.append(new_mask)
    srv["expert_prune"] = {"mask": new_mask, "sha256": sha(a.mask)}
    old_ds = srv["expert_overlay"]["dataset"]
    srv["expert_overlay"].update(dataset=a.overlay, manifest_sha256=a.overlay_sha)
    study["model_datasets"] = [a.overlay if d == old_ds else d for d in study.get("model_datasets", [])]
    if a.overlay not in study["model_datasets"]:
        raise SystemExit(f"model_datasets did not list the old overlay {old_ds}")
    # build_v82 records nothing reads at run time: under evidence (check_play_bundle's META), values unchanged
    records = {k: study.pop(k) for k in RECORD_KEYS if k in study}
    study.setdefault("evidence", {})["build_v82_records"] = records
    if a.draft_head:
        head = srv["draft_head"]["overlay"]
        for name in ("overlay.safetensors", "overlay-manifest.json"):
            shutil.copyfile(Path(a.draft_head) / name, out / head["path"] / name)
            changed.append(f"{head['path']}/{name}")
        hm = json.loads((out / head["path"] / "overlay-manifest.json").read_text())
        if sha(out / head["path"] / "overlay.safetensors") != hm["overlay"]["sha256"]:
            raise SystemExit("--draft-head: overlay.safetensors does not match its manifest")
        head["sha256"] = sha(out / head["path"] / "overlay-manifest.json")
        dh_info = {"source": str(a.draft_head), "manifest_sha256": head["sha256"],
                   "selected_step": hm.get("selected_step")}
    study["games"] = games
    study["games_excluded"] = [g for g in public if g not in games]
    study["roster"] = f"10 held-out games (RL split; fit on the 15: {' '.join(TRAIN15)})"
    study["name"] = out.name
    study["heldout"] = {"games": want, "fit_on": list(TRAIN15), "mask": Path(a.mask).name,
                        "mask_screen": mask.get("screen"), "overlay": a.overlay, "overlay_manifest_sha256": a.overlay_sha,
                        "source_bundle": src.name, "strict_overlay": "v87b, extra non-expert index shards allowed"}
    if a.draft_head:
        study["heldout"]["draft_head"] = dh_info
    (out / "configs/comparison.json").write_text(json.dumps(study, indent=2) + "\n")
    changed.append("configs/comparison.json")
    manifest = json.loads((out / "MANIFEST.json").read_text())
    manifest["study"] = out.name
    for rel in removed:
        manifest["files"].pop(rel, None)
    for rel in changed:
        manifest["files"][rel] = sha(out / rel)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(out), "games": len(games), "excluded": len(study["games_excluded"]),
                      "mask_sha256": srv["expert_prune"]["sha256"], "overlay": a.overlay,
                      "instances": study["competition"]["preview_instances"]}))


if __name__ == "__main__":
    main()
