"""Synthetic storage/round-trip cost screen; does not establish GDN gameplay fidelity."""
import json
import time
from pathlib import Path


def run(output):
    import torch
    torch.manual_seed(0)
    device = 'cuda'
    results = {'scope': 'synthetic state-storage feasibility; no inference kernel changed or gameplay quality claim', 'rows': []}
    # Full recurrent matrix state: 36 linear layers, 48 heads, 128 x 128 per stream.
    for streams in (37,55):
        state = torch.randn((streams*36,48,128,128),device=device,dtype=torch.bfloat16)
        def pack():
            scale=state.float().abs().amax(-1,keepdim=True).clamp_min(1e-8)/127
            q=(state.float()/scale).round().clamp(-127,127).to(torch.int8)
            return q,scale
        q,scale=pack()
        restored=(q.float()*scale).to(torch.bfloat16)
        rel=float((restored.float()-state.float()).norm()/state.float().norm())
        times=[]
        for _ in range(8):
            torch.cuda.synchronize();t=time.perf_counter();q,scale=pack();restored=(q.float()*scale).to(torch.bfloat16);torch.cuda.synchronize();times.append(time.perf_counter()-t)
        original=state.numel()*state.element_size();packed=q.numel()+scale.numel()*scale.element_size()
        results['rows'].append({'streams':streams,'bf16_bytes':original,'int8_row_scaled_bytes':packed,'saved_bytes':original-packed,'relative_roundtrip_error':rel,'pack_unpack_ms':1000*sum(times[2:])/len(times[2:])})
        del state,q,scale,restored
        torch.cuda.empty_cache()
    Path(output).mkdir(parents=True,exist_ok=True)
    Path(output,'result.json').write_text(json.dumps(results,indent=2));return results
