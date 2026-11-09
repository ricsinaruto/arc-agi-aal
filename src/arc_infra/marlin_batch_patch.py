"""ARC_MARLIN_BATCH=1 (boot cut, MFU-2 24 Oct): vLLM's NVFP4 MoE Marlin prep (prepare_nvfp4_moe_layer_for_marlin)
permutes and converts the block scales one expert at a time: per expert a permute, a dozen small kernels and a host
sync (``(scales >= 0).all()``), 2 x experts x layers times (B300 hero v6g: ~7 s of the post-load pass). This edit runs
the same element-wise ops on all experts of a layer at once (same permutation per expert, same half/FP8 casts, one
sync per layer), so the result is bit-identical. Anchor-matched on the file as expert_mask_patch leaves it; an
unrecognized file keeps the per-expert loop (ARC_MARLIN_BATCH_PATCH_SKIPPED)."""

import hashlib
import json
from pathlib import Path

ENV = "ARC_MARLIN_BATCH"
RELATIVE = "vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py"
EDITS = (
    ("        for i in range(E):\n            scale = scales[i].T\n",
     "        _arc_b = _arc_batched_marlin_scales(scales, size_k, size_n, GROUP_SIZE, is_a_8bit,  # ARC marlin_batch\n"
     "                                            combined_scale_factor, param_dtype)\n"
     "        for i in range(0 if _arc_b is not None else E):\n            scale = scales[i].T\n"),
    ("        scales = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)\n"
     "        g_scales = nvfp4_marlin_process_global_scale(g_scales, param_dtype)\n",
     "        scales = _arc_b if _arc_b is not None else torch.cat([x.unsqueeze(0) for x in tensor_list], 0)\n"
     "        g_scales = nvfp4_marlin_process_global_scale(g_scales, param_dtype)\n"),
)

SOURCE = '''

def _arc_batched_marlin_scales(scales, size_k, size_n, group_size, is_a_8bit, scale_factor, a_dtype):  # ARC marlin_batch
    """marlin_permute_scales + nvfp4_marlin_process_scales over every expert at once (bit-identical); None when
    ARC_MARLIN_BATCH is off or no shared scale factor was given (the per-expert loop then runs)."""
    import os

    from vllm.model_executor.layers.quantization.utils.marlin_utils import get_scale_perms

    if os.environ.get("ARC_MARLIN_BATCH") != "1" or scale_factor is None:
        return None
    experts = scales.shape[0]
    scale_perm, scale_perm_single = get_scale_perms()
    s = scales.transpose(1, 2)  # scales[i].T for every expert
    if group_size < size_k and group_size != -1 and not is_a_8bit:
        s = s.reshape(experts, -1, len(scale_perm))[:, :, scale_perm]
    else:
        s = s.reshape(experts, -1, len(scale_perm_single))[:, :, scale_perm_single]
    s = s.reshape(experts, -1, size_n).contiguous()
    if not (s >= 0).all():
        logger.warning_once("NVFP4 Marlin assumes the scales to be >=0, but has encountered negative scales.")
    s = s.to(torch.half)
    rows = s.shape[1]
    s = s.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(experts, rows, -1)
    if scale_factor > 1.0:
        s = (s.float() * scale_factor).to(torch.half)
    s = s * (2**7)
    s[s < 2] = 0
    s = s.view(torch.int16) << 1
    s = s.view(torch.float8_e4m3fn)
    return s[..., 1::2].contiguous()
'''


def patched_source(original):
    text = original.decode()
    if "ARC marlin_batch" in text:
        raise RuntimeError("marlin_batch patch: file already carries the edit")
    for old, new in EDITS:
        if text.count(old) != 1:
            raise RuntimeError(f"marlin_batch anchor is not unique: {old.strip()[:60]}")
        text = text.replace(old, new)
    text = text.rstrip("\n") + "\n" + SOURCE
    compile(text, RELATIVE, "exec")
    return text.encode()


def apply(site, output):
    """Never raises: an unrecognized file keeps the per-expert loop. Marker marlin-batch.json (not *-patch.json: no
    compiled code changes, the boot-cache key ignores it)."""
    target = Path(site) / RELATIVE
    original = target.read_bytes()
    marker = Path(output) / "marlin-batch.json"
    if b"ARC marlin_batch" in original:
        return json.loads(marker.read_text()) if marker.exists() else {"applied_earlier": True}
    try:
        patched = patched_source(original)
    except Exception as error:  # noqa: BLE001 - a boot cut, never a reason to fail a boot
        print("ARC_MARLIN_BATCH_PATCH_SKIPPED " + json.dumps({"error": str(error)[:200]}), flush=True)
        return None
    target.write_bytes(patched)
    identity = {"original_sha256": hashlib.sha256(original).hexdigest(),
                "patched_sha256": hashlib.sha256(patched).hexdigest()}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    print("ARC_MARLIN_BATCH_PATCH " + json.dumps(identity), flush=True)
    return identity
