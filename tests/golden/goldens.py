"""trunk/sft goldens, pinned 25 Oct from sft-heal 5f7b222de (loss, masking, export): each function recomputes one with
the current code; test_goldens.py compares it with data/<name>.json (never edit data/)."""
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
sys.path[:0] = [str(ROOT / "scripts/sft"), str(Path(__file__).resolve().parent)]


def sha(t):
    t = t.detach().contiguous().cpu()
    return hashlib.sha256((t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy().tobytes()).hexdigest()[:16]




def loss():
    """Masked next-token CE of the tiny student through train.example_loss: whole sequence (per-sequence mean, its
    gradient) and a 160-token gradient window over a no-grad prefix cache."""
    import common
    import mtp
    import train

    student = common.tiny_student().float()
    head = common.mtp_tiny(student.config.text_config)
    box, _ = mtp.hc_capture(student)
    g = torch.Generator().manual_seed(1)
    seqs = [torch.randint(10, 2000, (1, 256), generator=g) for _ in range(2)]
    out = {"seqs": [s[0].tolist() for s in seqs], "spans": [[100, 140], [200, 256]], "per_seq": [], "windowed": []}
    whole = train.Config(student="", data="", out="", window=10 ** 6, loss_chunk=48, mtp_weight=0.0)
    window = train.Config(student="", data="", out="", window=160, prefix_chunk=32, loss_chunk=48, mtp_weight=0.0)
    student.zero_grad()
    for ids in seqs:
        labels = torch.full_like(ids, -100)
        for a, b in out["spans"]:
            labels[:, a:b] = ids[:, a:b]
        n = int((labels[0, 1:] != -100).sum())
        ce, *_ = train.example_loss(student, head, box, {"input_ids": ids, "labels": labels}, whole, scale=1.0 / n)
        out["per_seq"].append(float(ce) / n)
        ce, *_ = train.example_loss(student, head, box, {"input_ids": ids, "labels": labels}, window, train=False)
        out["windowed"].append(float(ce) / n)
    norms = {n: float(p.grad.norm()) for n, p in student.named_parameters() if p.grad is not None}
    out["grad_norm"] = float(torch.stack([torch.tensor(v) for v in norms.values()]).norm())
    out["grad_norms"] = norms
    return out

def processor_dir(tmp):
    d = Path(tmp) / "processor"
    shutil.copytree(DATA / "processor", d)
    with gzip.open(d / "tokenizer.json.gz", "rb") as f:
        (d / "tokenizer.json").write_bytes(f.read())
    return d


def masking(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("sft_data", ROOT / "scripts/sft/data.py")
    fq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fq)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(processor_dir(tmp_path)))
    caps = fq.read_rollout(DATA / "rollout.jsonl")
    rows = []
    enc = {}
    for c in caps:
        inputs, checks = fq.encode_request(processor, c)
        ids, labels = inputs["input_ids"], inputs["labels"]
        enc[c["request"]] = (inputs, checks["prompt_tokens"])
        rows.append({"request": c["request"], "tokens": ids.shape[1], "ids": sha(ids), "labelled": int((labels != -100).sum()),
                     "labels": sha(labels), "checks": {k: v for k, v in checks.items()}})
    ex = {}
    for window in (100_000, 3000):
        made = fq.chain_examples(lambda c: (lambda i, n: ({**i, "labels": i["labels"].clone()}, n))(*enc[c["request"]]),
                                 caps, window=window)
        ex[str(window)] = [{"tokens": i["input_ids"].shape[1], "tail": tail, "targets": t,
                            "labelled": int((i["labels"] != -100).sum()), "labels": sha(i["labels"])} for i, tail, t in made]
    return ({"captures": len(caps), "rows": rows, "chain_examples": ex})


def export(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("sft_export", ROOT / "scripts/sft/export.py")
    sft_export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sft_export)
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.manual_seed(3)
    stu = tmp_path / "student"
    stu.mkdir()
    L = "model.language_model.layers"
    t = {f"{L}.0.mlp.experts.gate_up_proj": torch.randn(2, 256, 128).to(torch.bfloat16),
         f"{L}.0.mlp.experts.down_proj": torch.randn(2, 128, 128).to(torch.bfloat16),
         f"{L}.0.mlp.gate.weight": torch.randn(2, 128).to(torch.bfloat16),
         f"{L}.0.self_attn.o_proj.weight": torch.randn(128, 128).to(torch.bfloat16),
         f"{L}.1.mlp.shared_expert.down_proj.weight": torch.randn(128, 64).to(torch.bfloat16),
         "model.language_model.norm.weight": torch.randn(128).to(torch.bfloat16),
         "model.language_model.embed_tokens.weight": torch.randn(16, 128).to(torch.bfloat16)}
    save_file(t, str(stu / "s-1.safetensors"))
    (stu / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "s-1.safetensors" for k in t}}))
    prune = tmp_path / "prune.json"
    prune.write_text(json.dumps({"format": "arc-expert-prune-v1", "experts": 4, "pruned": {"0": [0, 2], "2": [1, 3]}}))
    (stu / "config.json").write_text(json.dumps({"student_cut": {"source_layers": [0, 2], "drop": [1], "prune_file": str(prune)}}))
    names = [f"{L}.0.mlp.experts.{e}.{p}.{s}" for e in (1, 3) for p in ("gate_proj", "up_proj", "down_proj")
             for s in ("qweight", "qzeros", "scales")]
    names += [f"{L}.0.mlp.gate.weight", f"{L}.0.self_attn.o_proj.weight", f"{L}.2.mlp.shared_expert.down_proj.weight",
              "model.language_model.norm.weight"]
    idx = tmp_path / "ar-index.json"
    idx.write_text(json.dumps({"weight_map": {n: "x" for n in names}}))
    full_router = torch.randn(4, 128).to(torch.bfloat16)
    src = tmp_path / "router-src"
    src.mkdir()
    save_file({f"{L}.0.mlp.gate.weight": full_router}, str(src / "r.safetensors"))
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"{L}.0.mlp.gate.weight": "r.safetensors"}}))
    counts = sft_export.export(str(stu), str(tmp_path / "out"), served_index=str(idx), prune_file=str(prune),
                               router_src=str(src), search=20)
    ov = json.loads((tmp_path / "out" / "overlay.json").read_text())
    shas = {}
    for file in sorted(set(ov["weight_map"].values())):
        with safe_open(str(tmp_path / "out" / file), framework="pt") as f:
            for k in f.keys():
                shas[k] = [sha(f.get_tensor(k)), list(f.get_tensor(k).shape), str(f.get_tensor(k).dtype)]
    return ({"counts": counts, "tensors": shas, "drop_layers": ov["drop_layers"],
                    "inputs_seed": 3, "router_full_seed_order": "after inputs"})


def mtp():
    """Draft logits of the MTP head (tiny, seeded) over a whole sequence: embedding of token j and the main model's
    pre-mixer state at j-1, the main LM head on the result."""
    import common
    import mtp as mtp_mod

    student = common.tiny_student().float().eval()
    head = common.mtp_tiny(student.config.text_config).eval()
    ids = torch.randint(10, 2000, (1, 128), generator=torch.Generator().manual_seed(2))
    box, hook = mtp_mod.hc_capture(student)
    with torch.no_grad():
        student.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        hook.remove()
        a, b, cur, pos = mtp_mod.inputs(ids, 0, ids.shape[1])
        emb = student.model.language_model.embed_tokens(cur)
        drafted = head(emb, box["hc"][:, : b - a], pos)
        logits = student.lm_head(drafted).float()[0]
    return {"positions": [a, b], "sum": float(logits.sum()), "abs": float(logits.abs().sum()),
            "argmax": logits.argmax(-1)[:32].tolist(), "rows": [float(r) for r in logits.norm(dim=-1)[:8]]}
