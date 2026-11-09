"""ARC_MOE_LORA_BM16=1 (MFU-1 25 Oct): vLLM's routed-expert LoRA (align + one-shot shrink/expand) at 16-row blocks on
decode-sized steps, up to ARC_MOE_LORA_BM16_MAX_TOKENS (256). vLLM's bf16 config takes 32 rows at M 33-96 and 64 above;
MMA rows are independent, so the bits are unchanged (moel4, B300). tput13 (hero play, 5 adapters): MoE-LoRA 8.61 ->
7.05 ms per decode step; MFU-2's stage-2 kernels (no align) lost to it (7.33 ms; 1 adapter 8.04 vs 5.71) and are not
ported.

Anchor edit of punica_gpu.add_lora_w13/add_lora_w2 after the config normalize (both: w2's one-shot uses w13's align
block size), so it composes with expert_mask_patch's compact edits and lora_dense_decode on the same file; an
unrecognized file keeps vLLM's path (ARC_MOE_LORA_BM16_PATCH_SKIPPED). Flags are read at import per server process.
Marker *-patch.json: punica_gpu.py is traced by torch.compile, so the boot-cache key changes with it."""

import hashlib
import json
from pathlib import Path

TAG = "ARC moe_lora_bm16"
PUNICA = "vllm/lora/punica_wrapper/punica_gpu.py"
HEADER = ("import os as _arc_os  # " + TAG + "\n"
          "_ARC_BM16 = _arc_os.environ.get(\"ARC_MOE_LORA_BM16\") == \"1\"\n"
          "_ARC_BM16_MAX = int(_arc_os.environ.get(\"ARC_MOE_LORA_BM16_MAX_TOKENS\", \"256\"))\n")
IMPORT = ("from vllm import _custom_ops as ops\n", "from vllm import _custom_ops as ops\n" + HEADER, 1)
NORMALIZE = ("        shrink_config = _normalize_lora_config_keys(shrink_config)\n"
             "        expand_config = _normalize_lora_config_keys(expand_config)\n")
BM16 = (NORMALIZE, NORMALIZE +
        "        if _ARC_BM16 and num_tokens <= _ARC_BM16_MAX and int(shrink_config.get(\"BLOCK_SIZE_M\") or 64) > 16:"
        "  # " + TAG + "\n"
        "            shrink_config = {**shrink_config, \"BLOCK_SIZE_M\": 16}\n"
        "            expand_config = {**expand_config, \"BLOCK_SIZE_M\": 16}\n", 2)
EDITS = (IMPORT, BM16)


def patched_source(original):
    text = original.decode()
    if TAG in text:
        raise RuntimeError("moe_lora_bm16 patch: punica_gpu.py already carries the edit")
    for old, new, count in EDITS:
        if text.count(old) != count:
            raise RuntimeError(f"moe_lora_bm16 anchor found {text.count(old)}x, want {count}: {old.strip()[:60]}")
        text = text.replace(old, new)
    compile(text, PUNICA, "exec")
    return text.encode()


def apply(site, output):
    """Never raises: an unrecognized file keeps vLLM's path."""
    target = Path(site) / PUNICA
    marker = Path(output) / "moe-lora-bm16-patch.json"
    try:
        original = target.read_bytes()
        if TAG.encode() in original:
            return json.loads(marker.read_text()) if marker.exists() else {"applied_earlier": True}
        patched = patched_source(original)
    except Exception as error:  # noqa: BLE001 - a speed path, never a reason to fail a boot
        print("ARC_MOE_LORA_BM16_PATCH_SKIPPED " + json.dumps({"error": str(error)[:200]}), flush=True)
        return None
    target.write_bytes(patched)
    identity = {"original_sha256": hashlib.sha256(original).hexdigest(),
                "patched_sha256": hashlib.sha256(patched).hexdigest()}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    print("ARC_MOE_LORA_BM16_PATCH " + json.dumps(identity), flush=True)
    return identity
