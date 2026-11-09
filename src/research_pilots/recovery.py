"""Layer-local router recovery on recorded ARC activations, not full-model distillation."""
import json
import time
from pathlib import Path


def mixture(torch, x, router, expert_outputs, keep, topk):
    logits = x.float() @ router.float().T
    logits = logits.masked_fill(~keep[None, :], float('-inf'))
    values, ids = logits.topk(topk, dim=-1)
    weights = values.softmax(-1)
    ys = expert_outputs[torch.arange(len(x), device=x.device)[:, None], ids].float()
    return (ys * weights[..., None]).sum(1)


def run(kind, root, capture, mask_path, output, seconds=3000, layers=(0, 20, 40)):
    import numpy as np
    import torch
    from safetensors import safe_open
    started = time.monotonic()
    deadline = started + seconds
    root, capture, output = Path(root), Path(capture), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    mask = json.loads(Path(mask_path).read_text())['pruned']
    results = {'kind': kind, 'scope': 'layer-local BF16 teacher, held-out captured activations; NOT served-stack or gameplay quality', 'layers': []}
    def tensor(name):
        with safe_open(root / index[name], framework='pt', device='cpu') as f:
            return f.get_tensor(name).cuda()
    for layer in layers:
        if time.monotonic() > deadline - 120:
            break
        prefix = f'model.language_model.layers.{layer}.mlp.'
        gu = tensor(prefix + 'experts.gate_up_proj')
        dn = tensor(prefix + 'experts.down_proj')
        router = tensor(prefix + 'gate.weight').float()
        xparts = []
        # Separate archived calibration and evaluation games. Evenly spaced, deterministic rows.
        for split in ('calib', 'eval'):
            path = capture / split / f'L{layer:02d}.x'
            a = np.memmap(path, dtype=np.int16, mode='r').reshape(-1, router.shape[1])
            selected = np.linspace(0, len(a)-1, min(256, len(a)), dtype=int)
            xparts.append(torch.from_numpy(a[selected].copy()).view(torch.bfloat16).cuda())
        x = torch.cat(xparts); n = len(xparts[0]); e, _, h = gu.shape; middle = gu.shape[1] // 2
        outputs = torch.empty((len(x), e, h), device='cuda', dtype=torch.bfloat16)
        with torch.no_grad():
            for expert in range(e):
                if time.monotonic() > deadline - 90:
                    raise TimeoutError('recovery precompute deadline; no unvalidated checkpoint promoted')
                v = x @ gu[expert].T
                outputs[:, expert] = (torch.nn.functional.silu(v[:, :middle]) * v[:, middle:]) @ dn[expert].T
        del gu, dn
        all_keep = torch.ones(e, device='cuda', dtype=torch.bool)
        keep = all_keep.clone()
        if kind == 'prune-recovery':
            keep[mask[str(layer)]] = False
        topk = 10 if kind == 'prune-recovery' else 8
        with torch.no_grad():
            teacher = mixture(torch, x, router, outputs, all_keep, 10)
        target_scale = teacher[:n].square().mean().clamp_min(1e-12)
        trained = torch.nn.Parameter(router.clone())
        opt = torch.optim.AdamW([trained], lr=2e-5, weight_decay=0)
        def error(part):
            with torch.no_grad():
                pred = mixture(torch, x[part], trained, outputs[part], keep, topk)
                return float(((pred-teacher[part]).square().sum()/teacher[part].square().sum().clamp_min(1e-12)).sqrt())
        before = {'train': error(slice(None,n)), 'heldout': error(slice(n,None))}
        steps = 0; best_train = float('inf'); chosen = trained.detach().clone()
        for step in range(200):
            if time.monotonic() > deadline - 60:
                break
            pred = mixture(torch, x[:n], trained, outputs[:n], keep, topk)
            loss = (pred-teacher[:n]).square().mean()/target_scale
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([trained],1);steps += 1
            # Model selection uses calibration loss only. Held-out rows never determine updates or early stopping.
            if float(loss.detach()) < best_train:
                best_train=float(loss.detach());chosen=trained.detach().clone()
            opt.step()
        with torch.no_grad(): trained.copy_(chosen)
        after = {'train': error(slice(None,n)), 'heldout': error(slice(n,None))}
        torch.save({'layer':layer,'router':trained.detach().cpu(),'keep':keep.cpu(),'topk':topk},output/f'layer-{layer}.pt')
        results['layers'].append({'layer':layer,'rows_train':n,'rows_heldout':len(x)-n,'steps':steps,'before':before,'after':after})
        results['elapsed_seconds']=time.monotonic()-started
        (output/'result.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(results['layers'][-1]),flush=True)
        del outputs, teacher, trained, chosen, opt, x, xparts, router
        torch.cuda.empty_cache()
    return results
