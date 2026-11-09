"""ARC_MOE_LORA_FAST=1 / ARC_MOE_LORA_BM16=1 (MFU-1 25 Oct; MFU-2 stage 2 kernels): routed-expert LoRA on decode-sized
steps without vLLM's align pass.

- ARC_MOE_LORA_FAST: per MoE layer, one-CTA pair lists of the live (lora, expert) keys, then persistent kernels over
  live items for w13 and w2 (moe_lora_fast_runtime, installed as vllm/arc_moe_lora_fast.py); bitwise vLLM's sorted
  one-shot kernel. Taken only where vLLM would take that path (not naive, not fully sharded, no expert map,
  add_inputs) and up to ARC_MOE_LORA_FAST_MAX_TOKENS (160). moel4 (B300, E 320, max_loras 5, cold, us per layer at
  M 70 with 1/3/5 live adapters): vLLM 106/166/173, stage 2 80/90/95.
- ARC_MOE_LORA_BM16: vLLM's own align + one-shot with BLOCK_SIZE_M 16 up to ARC_MOE_LORA_BM16_MAX_TOKENS (256);
  bitwise the same (independent MMA rows); M 70 / 5 adapters 173 -> 143 us. Also covers steps above the FAST cap.

Anchor edits of punica_gpu.add_lora_w13/add_lora_w2 (and one import line), so it composes with expert_mask_patch's
compact edits and lora_dense_decode (add_lora_linear) on the same file; an unrecognized file keeps vLLM's path
(ARC_MOE_LORA_FAST_PATCH_SKIPPED). The flags are read at import in each server process. Marker *-patch.json:
punica_gpu.py is traced by torch.compile, so the boot-cache key changes with it."""

import hashlib
import json
from pathlib import Path

ENVS = ("ARC_MOE_LORA_FAST", "ARC_MOE_LORA_BM16")
TAG = "ARC moe_lora_fast"
PUNICA = "vllm/lora/punica_wrapper/punica_gpu.py"
HELPER = "vllm/arc_moe_lora_fast.py"
IMPORT = ("from vllm import _custom_ops as ops\n",
          "from vllm import _custom_ops as ops\nimport vllm.arc_moe_lora_fast as _arc_mlf  # " + TAG + "\n", 1)
NORMALIZE = ("        shrink_config = _normalize_lora_config_keys(shrink_config)\n"
             "        expand_config = _normalize_lora_config_keys(expand_config)\n")
BM16 = (NORMALIZE, NORMALIZE + "        shrink_config, expand_config = _arc_mlf.bm16(shrink_config, expand_config, "
        "num_tokens)  # " + TAG + "\n", 2)  # w13 and w2 alike: w2's one-shot must use w13's align block size
W13_ANCHOR = ("        (\n"
              "            token_lora_mapping,\n"
              "            sorted_token_ids_lora,\n"
              "            expert_ids_lora,\n"
              "            num_tokens_post_padded_lora,\n"
              "        ) = self.moe_lora_align_block_size(\n")
W13 = (W13_ANCHOR,
       "        if _arc_mlf.take(num_tokens, top_k, max_lora_rank, fully_sharded, expert_map,  # " + TAG + "\n"
       "                         naive_block_assignment, add_inputs):\n"
       "            return _arc_mlf.w13(self, y, x, lora_a_stacked, lora_b_stacked, topk_ids, topk_weights, top_k_num,\n"
       "                                top_k, adapter_enabled, token_lora_mapping)\n" + W13_ANCHOR, 1)
W2_ANCHOR = ("        _sorted = sorted_token_ids_lora\n"
             "        _eids = expert_ids_lora\n"
             "        if _sorted is not None:\n"
             "            assert _eids is not None\n")
W2 = (W2_ANCHOR,
      "        if isinstance(num_tokens_post_padded_lora, _arc_mlf.Lists):  # " + TAG + ": w13 took stage 2\n"
      "            _arc_mlf.w2(y, x, lora_a_stacked, lora_b_stacked, topk_weights, num_tokens_post_padded_lora, top_k)\n"
      "            return\n" + W2_ANCHOR, 1)
EDITS = (IMPORT, BM16, W13, W2)


def runtime_source():
    return Path(__file__).with_name("moe_lora_fast_runtime.py").read_bytes()


def patched_source(original):
    text = original.decode()
    if TAG in text:
        raise RuntimeError("moe_lora_fast patch: punica_gpu.py already carries the edit")
    for old, new, count in EDITS:
        if text.count(old) != count:
            raise RuntimeError(f"moe_lora_fast anchor found {text.count(old)}x, want {count}: {old.strip()[:60]}")
        text = text.replace(old, new)
    compile(text, PUNICA, "exec")
    return text.encode()


def apply(site, output):
    """Never raises: an unrecognized file keeps vLLM's path."""
    site = Path(site)
    target, helper = site / PUNICA, site / HELPER
    marker = Path(output) / "moe-lora-fast-patch.json"
    try:
        original = target.read_bytes()
        if TAG.encode() in original:
            return json.loads(marker.read_text()) if marker.exists() else {"applied_earlier": True}
        patched, runtime = patched_source(original), runtime_source()
        compile(runtime, HELPER, "exec")
    except Exception as error:  # noqa: BLE001 - a speed path, never a reason to fail a boot
        print("ARC_MOE_LORA_FAST_PATCH_SKIPPED " + json.dumps({"error": str(error)[:200]}), flush=True)
        return None
    helper.write_bytes(runtime)
    target.write_bytes(patched)
    identity = {"original_sha256": hashlib.sha256(original).hexdigest(),
                "patched_sha256": hashlib.sha256(patched).hexdigest(),
                "helper_sha256": hashlib.sha256(runtime).hexdigest()}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    print("ARC_MOE_LORA_FAST_PATCH " + json.dumps(identity), flush=True)
    return identity
