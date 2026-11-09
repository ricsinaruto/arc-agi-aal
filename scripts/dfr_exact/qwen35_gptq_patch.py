"""Patch his SGLang qwen3_5.py so GPTQ params of the merged GDN in_proj_qkvz / in_proj_ba take the split-checkpoint
tuple shards (in_proj_qkv -> (0, 1, 2)) like .weight does; packed-on-output params split by output size / pack factor.
python qwen35_gptq_patch.py <site-packages>/sglang/srt/models/qwen3_5.py"""
import sys
from pathlib import Path

p = Path(sys.argv[1]); s = p.read_text()
A = 'for attr_name in ("weight", "weight_scale_inv", "weight_scale", "input_scale"):'
B = '''        if isinstance(param, PerTensorScaleParameter):
            # One logical scale per logical shard.
            return [1 for _ in loaded_shard_id]
'''
if "arc-gptq-split" not in s:
    assert s.count(A) == 1 and s.count(B) == 1, "qwen3_5.py anchors moved"
    s = s.replace(A, 'for attr_name in ("weight", "weight_scale_inv", "weight_scale", "input_scale", '
                     '"qweight", "scales", "qzeros"):  # arc-gptq-split')
    s = s.replace(B, B + '''
        # arc-gptq-split: GPTQ params packed along the output dim (qzeros) split by output size / pack factor.
        if getattr(param, "packed_dim", None) is not None and param.packed_dim == getattr(param, "output_dim", None):
            return [module.output_sizes[idx] // param.packed_factor for idx in loaded_shard_id]
''')
    p.write_text(s)
print("qwen3_5 gptq split patch", "applied" if "arc-gptq-split" in s else "missing")
