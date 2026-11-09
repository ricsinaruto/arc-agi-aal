"""int8 dense view of his auto-round checkpoint: listed BF16 linear weights -> GPTQ int8 (sym, group 128, auto_gptq v1
packing: qweight [in/4, out] int32, scales [in/g, out] fp16, qzeros 0x7F7F7F7F), extra_config bits 8 for those modules
(SGLang auto-round -> GPTQ Marlin uint8b128, W8A16). Untouched shards are symlinked. Runs in his venv (torch).
python int8_dense_view.py SRC DST [REGEX]"""
import json, os, re, shutil, sys, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT = (r"model\.language_model\.layers\.\d+\.(linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|"
           r"self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.shared_expert\.(gate_proj|up_proj|down_proj))\.weight")
G = 128


def quantize(w):
    """w [out, in] -> (qweight [in/4, out] int32, scales [in/G, out] fp16, qzeros [in/G, out/4] int32)."""
    out, inn = w.shape
    wf = w.float().view(out, inn // G, G)
    scale = (wf.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-10)
    q = (torch.round(wf / scale).clamp(-128, 127) + 128).to(torch.int32).view(out, inn)  # uint8b128
    q = q.t().contiguous().view(inn // 4, 4, out)
    qweight = (q[:, 0] | (q[:, 1] << 8) | (q[:, 2] << 16) | (q[:, 3] << 24)).to(torch.int32)
    scales = scale.view(out, inn // G).t().contiguous().to(torch.float16)
    qzeros = torch.full((inn // G, out // 4), 0x7F7F7F7F, dtype=torch.int32)
    return qweight, scales, qzeros


def dequantize(qweight, scales):
    inn4, out = qweight.shape
    q = torch.stack([(qweight >> (8 * k)) & 0xFF for k in range(4)], 1).view(inn4 * 4, out).float() - 128
    return (q.view(-1, G, out) * scales.float().view(-1, 1, out)).view(inn4 * 4, out).t()


def build(src, dst, pattern=DEFAULT, device="cuda" if torch.cuda.is_available() else "cpu"):
    src, dst, rx = Path(src), Path(dst), re.compile(pattern)
    t0 = time.time()
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    targets = {k for k in wm if rx.fullmatch(k)}
    files = sorted({wm[k] for k in targets})
    new_wm, done, err = dict(wm), [], 0.0
    for name in files:
        tensors = {}
        with safe_open(src / name, framework="pt", device="cpu") as f:
            meta = f.metadata()
            for k in f.keys():
                t = f.get_tensor(k)
                if k in targets and t.dim() == 2 and t.shape[1] % G == 0 and t.shape[0] % 64 == 0:
                    qw, sc, qz = quantize(t.to(device))
                    base = k[: -len(".weight")]
                    tensors[base + ".qweight"], tensors[base + ".scales"], tensors[base + ".qzeros"] = qw.cpu(), sc.cpu(), qz
                    err = max(err, ((dequantize(qw, sc) - t.to(device).float()).abs().max() / t.float().abs().max()).item())
                    for s in (".qweight", ".scales", ".qzeros"):
                        new_wm[base + s] = name
                    del new_wm[k]
                    done.append(base)
                else:
                    tensors[k] = t
        save_file(tensors, dst / name, metadata=meta)
    for p in src.iterdir():
        if p.name in files or p.name in ("model.safetensors.index.json", "config.json", "quantization_config.json"):
            continue
        os.symlink(p.resolve(), dst / p.name)
    (dst / "model.safetensors.index.json").write_text(json.dumps({**index, "weight_map": new_wm}, indent=1))
    cfg = json.loads((src / "config.json").read_text())
    q = cfg.get("quantization_config") or json.loads((src / "quantization_config.json").read_text())
    extra = q.get("extra_config") or {}
    if isinstance(extra, str):
        import ast
        extra = ast.literal_eval(extra)
    # AutoRound resolves a fused layer (in_proj_qkvz, shared_expert.gate_up_proj) by its own name only: SGLang never
    # hands it the model's packed_modules_mapping, so without these keys those layers stayed BF16 and their int8
    # tensors were dropped ("Parameter ... not found"), leaving the fused weights unset (i8s10, 15 Oct).
    fused = {"linear_attn.in_proj_qkv": "linear_attn.in_proj_qkvz", "linear_attn.in_proj_z": "linear_attn.in_proj_qkvz",
             "shared_expert.gate_proj": "shared_expert.gate_up_proj", "shared_expert.up_proj": "shared_expert.gate_up_proj",
             "self_attn.q_proj": "self_attn.qkv_proj", "self_attn.k_proj": "self_attn.qkv_proj",
             "self_attn.v_proj": "self_attn.qkv_proj"}
    keys = set(done)
    for base in done:
        for sub, f in fused.items():
            if base.endswith(sub):
                keys.add(base[: -len(sub)] + f)
    keys |= {k.replace("self_attn.", "", 1) for k in keys if "self_attn.qkv_proj" in k}  # capi8 v2: q/k/v seen as layers.N.q_proj
    keys |= {k.replace("model.language_model.layers.", "model.layers.", 1) for k in keys}
    for base in sorted(keys):
        extra[base] = {"bits": 8, "group_size": G, "sym": True, "data_type": "int"}
    q["extra_config"] = extra
    if "quantization_config" in cfg:
        cfg["quantization_config"] = q
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    if (src / "quantization_config.json").exists():
        (dst / "quantization_config.json").write_text(json.dumps(q, indent=2))
    return {"modules": len(done), "files": len(files), "max_rel_err": round(err, 5), "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    print(json.dumps({"int8_dense_view": build(*sys.argv[1:])}), flush=True)
