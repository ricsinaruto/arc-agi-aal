"""His SGLang: the per-step M-RoPE delta upload in compute_spec_mrope_positions is a blocking H2D copy of a [bs, 1] int64
tensor; with image prompts it runs at every draft-extend / verify / draft init and stalls the host on the GPU
(prof10 v3: ~24 ms per step under draft_extend, ~13% of GPU time idle). Same values, uploaded from pinned memory
without blocking. python mrope_async_patch.py <site-packages>/sglang"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "srt/model_executor/forward_batch_info.py"
s = p.read_text()
OLD = "            mrope_delta_tensor = torch.stack(mrope_deltas, dim=0).to(device=device)\n"
NEW = ("            mrope_delta_tensor = torch.stack(mrope_deltas, dim=0)  # arc-mrope-async\n"
       "            if mrope_delta_tensor.device.type == \"cpu\":\n"
       "                mrope_delta_tensor = mrope_delta_tensor.pin_memory().to(device=device, non_blocking=True)\n"
       "            else:\n"
       "                mrope_delta_tensor = mrope_delta_tensor.to(device=device)\n")
if "arc-mrope-async" not in s:
    assert s.count(OLD) == 1, "forward_batch_info.py anchor moved"
    p.write_text(s.replace(OLD, NEW))
print("mrope async patch", "applied" if "arc-mrope-async" in p.read_text() else "missing")
