"""ARC_DRAFT_LORA=1 (RL stage D serving side): the trained MTP draft LoRA (draft_lora.safetensors + draft_lora.json
beside the adapter) merged into the served draft's BF16 dense weights at adapter activation, the base restored on
deactivation (draft_lora_runtime, installed as vllm/arc_draft_lora.py). The pinned vLLM never LoRA-wraps the draft.

Anchor edits (expert_mask's compact overlay and lora_pin also edit model_manager.py):
- lora/worker_manager.py _load_adapter: read the draft factors beside the adapter onto the LoRAModel;
- lora/model_manager.py: activate_adapter (after the modules' set_lora) merges, _deactivate_adapter and
  remove_all_adapters restore;
- v1/worker/gpu/spec_decode/speculator.py load_model: registers the loaded draft model (V2 runner).
An unrecognized file writes nothing (ARC_DRAFT_LORA_PATCH_SKIPPED; the CPU gate's ARC_SERVING_PATCH_CHECK fails on it).
None of the files is traced by torch.compile, so the marker is not a *-patch.json (boot-cache key unchanged)."""

import hashlib
import json
from pathlib import Path

ENV = "ARC_DRAFT_LORA"
TAG = "ARC draft_lora"
HELPER = "vllm/arc_draft_lora.py"
WORKER = "vllm/lora/worker_manager.py"
MANAGER = "vllm/lora/model_manager.py"
SPECULATOR = "vllm/v1/worker/gpu/spec_decode/speculator.py"
IMPORT = "import vllm.arc_draft_lora as _arc_dl  # " + TAG + "\n"


def _after(anchor, indent, call):
    return anchor, anchor + indent + IMPORT + indent + call + "\n"


EDITS = {
    WORKER: (
        _after("            lora.is_3d_lora_weight = lora_request.is_3d_lora_weight\n", " " * 12,
               "lora._arc_draft = _arc_dl.read(lora_path)"),
    ),
    MANAGER: (
        ('            logger.debug("Successfully loaded LoRA weights for module %s.", module_name)\n'
         "        return True\n",
         '            logger.debug("Successfully loaded LoRA weights for module %s.", module_name)\n'
         "        " + IMPORT + "        _arc_dl.activate(lora_model)\n        return True\n"),
        _after("    def _deactivate_adapter(self, lora_id: int):\n", " " * 8, "_arc_dl.deactivate(lora_id)"),
        _after('        """Remove all LoRAModels from the manager."""\n', " " * 8, "_arc_dl.deactivate_all()"),
    ),
    SPECULATOR: (
        _after("        self.model = self.load_draft_model(target_model, target_attn_layer_names)\n", " " * 8,
               "_arc_dl.register_draft(self.model)"),
    ),
}


def runtime_source():
    return Path(__file__).with_name("draft_lora_runtime.py").read_bytes()


def patched_source(relative, original):
    text = original.decode()
    if TAG in text:
        raise RuntimeError(f"draft_lora patch: {relative} already carries the edit")
    for old, new in EDITS[relative]:
        if text.count(old) != 1:
            raise RuntimeError(f"draft_lora anchor found {text.count(old)}x in {relative}: {old.strip()[:60]}")
        text = text.replace(old, new)
    compile(text, relative, "exec")
    return text.encode()


def apply(site, output):
    """Every file and the helper, or nothing. Never raises: the check boot fails on the None."""
    site, marker = Path(site), Path(output) / "draft-lora-overlay.json"
    try:
        originals = {rel: (site / rel).read_bytes() for rel in EDITS}
        if all(TAG.encode() in data for data in originals.values()):
            return json.loads(marker.read_text()) if marker.exists() else {"applied_earlier": True}
        patched = {rel: patched_source(rel, data) for rel, data in originals.items()}
        runtime = runtime_source()
        compile(runtime, HELPER, "exec")
    except Exception as error:  # noqa: BLE001 - reported, the patch check fails on it
        print("ARC_DRAFT_LORA_PATCH_SKIPPED " + json.dumps({"error": str(error)[:200]}), flush=True)
        return None
    (site / HELPER).write_bytes(runtime)
    identity = {"helper_sha256": hashlib.sha256(runtime).hexdigest()}
    for rel, data in patched.items():
        (site / rel).write_bytes(data)
        identity[rel] = {"original_sha256": hashlib.sha256(originals[rel]).hexdigest(),
                         "patched_sha256": hashlib.sha256(data).hexdigest()}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    print("ARC_DRAFT_LORA_PATCH " + json.dumps(identity), flush=True)
    return identity
