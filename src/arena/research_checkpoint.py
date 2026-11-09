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

    recipe.resolve_model_dir = lambda: root
    recipe.verify_model = verify
    return {"repo": identity[0], "revision": identity[1], "path": str(root)}
