"""Explicit checkpoint substitution for the approved Swift compatibility pilot.

The API alias stays Flash so the identical harness/template controls are used; the
actual repository, revision and byte hashes are recorded separately. No base-model
expert overlay or pruning mask may be applied to the post-trained checkpoint.
"""
import hashlib
import json
from pathlib import Path

REPO = "ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4"
REVISION = "3ff0520224f264a2d0ac4ab56ece8f2f13aadb38"
# Throughput matrix (5 Oct): Intel's AutoRound W4A16 as staged for dfranzen's notebook (scripts/modal_stage_autoround.py
# checked every sha256 once); shards are size-checked at start, files under SMALL bytes hashed.
AUTOROUND_REPO = "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound"
AUTOROUND_REVISION = "4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897"
CHECKPOINTS = {(REPO, REVISION): ("swift15-flash-nvfp4.json", False),
               (AUTOROUND_REPO, AUTOROUND_REVISION): ("intel-autoround-w4a16.json", True)}
SMALL = 64 << 20


def configure(recipe, working, spec):
    identity = (spec.get("repo"), spec.get("revision"))
    if identity not in CHECKPOINTS:
        raise ValueError("Unreviewed research checkpoint")
    inventory_name, size_only = CHECKPOINTS[identity]
    root = Path(spec["path"])
    manifest = Path(__file__).with_name("models") / inventory_name
    inventory = json.loads(manifest.read_text())
    if (inventory["repo"], inventory["revision"]) != identity:
        raise ValueError("Checkpoint inventory identity mismatch")

    def verify(model_dir, **kwargs):
        if Path(model_dir) != root:
            raise ValueError("Wrong research checkpoint path")
        for name, entry in inventory["files"].items():
            if spec.get("ple_fp8") and (name.startswith("model-ple-") or name == "ple_reshard_manifest.json"):
                continue  # served from the FP8 PLE source instead (the Kaggle copy carries no BF16 PLE files)
            if spec.get("skip_dotfiles") and name.startswith("."):
                continue  # Kaggle model instances drop dotfiles (.gitattributes)
            path = root / name
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                raise ValueError(f"Checkpoint file missing or wrong size: {name}")
            if size_only and entry["bytes"] > SMALL:
                continue
            with path.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {name}")
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        if not set(index.values()) <= inventory["files"].keys():
            raise ValueError("Unverified weight shard in checkpoint index")
        result = {"repo": identity[0], "revision": identity[1], "all_file_hashes_verified": not size_only,
                  **({"large_files_size_checked": True} if size_only else {}),
                  "files": len(inventory["files"]), "served_api_alias": recipe.SERVED_MODEL_NAME,
                  "base_expert_overlay": False, "base_pruning_mask": False}
        (Path(working) / "research-checkpoint.json").write_text(json.dumps(result, indent=2))
        return result

    ple = spec.get("ple_fp8")
    overrides = spec.get("text_overrides") or {}  # e.g. {"num_experts_per_tok": 6} (2K track G2, lossy)
    if not ple and not overrides:
        recipe.resolve_model_dir = lambda: root
        recipe.verify_model = verify
        return {"repo": identity[0], "revision": identity[1], "path": str(root)}
    # FP8 PLE (throughput arm, 16 Oct, owner item 2a): the PLE table from the NVFP4 base checkpoint (ModelOpt FP8, one
    # global scale; what every v8.2 run serves) in place of this checkpoint's BF16 table: half the pinned host RAM
    # (51 GB against 102 GB). A symlink copy of the checkpoint whose index maps the PLE shard keys to the FP8 files;
    # the worker selects the FP8 PLE method under ARC_PLE_FP8=1 (the serving overlay). Lossy: quality read needed.
    farm = Path(working).resolve() / "model-ple-fp8"
    src = Path(ple["root"]) if ple else None
    key = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."

    def build():
        index = json.loads((root / "model.safetensors.index.json").read_text())
        dropped, fp8 = set(), {}
        if ple:
            dropped, fp8 = build_ple(index)
        if farm.exists():
            import shutil
            shutil.rmtree(farm)
        farm.mkdir(parents=True)
        for entry in root.iterdir():
            if entry.name not in dropped and entry.name not in ("model.safetensors.index.json", "config.json"):
                (farm / entry.name).symlink_to(entry)
        for name in (ple or {}).get("files", {}):
            (farm / name).symlink_to(src / name)
        (farm / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
        config = json.loads((root / "config.json").read_text())
        for field, value in overrides.items():
            if field not in config["text_config"]:
                raise ValueError(f"text_config has no {field}")
            config["text_config"][field] = value
        (farm / "config.json").write_text(json.dumps(config, indent=2))
        return {"fp8_files": len((ple or {}).get("files", {})), "dropped_bf16_files": sorted(dropped),
                "fp8_keys": len(fp8), "text_overrides": overrides}

    def build_ple(index):
        for name, size in ple["files"].items():
            if not (src / name).is_file() or (src / name).stat().st_size != size:
                raise ValueError(f"FP8 PLE file missing or wrong size: {name}")
        fp8_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
        fp8 = {k: v for k, v in fp8_map.items() if k.startswith(key)}
        old = {k for k in index["weight_map"] if k.startswith(key)}
        if {k for k in fp8 if k.endswith(".weight") and ".shard_" in k} != old or not set(fp8.values()) <= ple["files"].keys():
            raise ValueError("FP8 PLE shard keys do not match the checkpoint's")
        dropped = {index["weight_map"][k] for k in old}
        index["weight_map"] = {**{k: v for k, v in index["weight_map"].items() if k not in old}, **fp8}
        if dropped & set(index["weight_map"].values()):
            raise ValueError("A PLE shard file holds other tensors")
        return dropped, fp8

    def verify_fp8(model_dir, **kwargs):
        if Path(model_dir) != farm:
            raise ValueError("Wrong research checkpoint path")
        result = verify(root, **kwargs)
        result["ple_fp8"] = {"source": str(src), **build()}
        (Path(working) / "research-checkpoint.json").write_text(json.dumps(result, indent=2))
        return result

    build()
    recipe.resolve_model_dir = lambda: farm
    recipe.verify_model = verify_fp8
    return {"repo": identity[0], "revision": identity[1], "path": str(farm), "ple_fp8_source": str(src),
            "text_overrides": overrides}
