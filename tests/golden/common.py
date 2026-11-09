"""Tiny-model fixtures for the goldens and trainer tests (from heal_kd, sft-heal 5f7b222de): a random tiny Flash-shaped
teacher, the student cut from it, seeded noise, a seeded MTP head, and the student on disk as the trainer loads it."""
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/sft"))
from mtp import MtpHead  # noqa: E402

CONFIG = ROOT / "scripts/sft/flash_config.json"
LAYER = re.compile(r"^(model\.language_model\.layers\.)(\d+)(\..*)$")
EXPERTS = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
ROUTER = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.gate\.weight$")
KEEP = [0, 1, 2, 3, 5]


def cut_text_config(text, *, keep_layers, experts=None, top_k=None):
    """The student's text config: ``keep_layers`` (teacher layer ids in order), ``experts`` kept per layer (same count
    in every layer), ``top_k``."""
    text = dict(text)
    text["layer_types"] = [text["layer_types"][i] for i in keep_layers]
    text["num_hidden_layers"] = len(keep_layers)
    old_to_new = {old: new for new, old in enumerate(keep_layers)}
    text["ple_layer_ids"] = [old_to_new[i - 1] + 1 for i in text.get("ple_layer_ids") or [] if i - 1 in old_to_new]
    if experts is not None:
        text["num_experts"] = len(next(iter(experts.values())))
    if top_k is not None:
        text["num_experts_per_tok"] = int(top_k)
    return text


def cut_state_dict(state, *, keep_layers, experts=None):
    """Teacher tensors -> student tensors: dropped layers removed, kept ones renumbered, routed experts and router rows
    sliced to ``experts[layer]`` (teacher layer ids, kept expert ids). Everything else is copied as is."""
    old_to_new = {old: new for new, old in enumerate(keep_layers)}
    out = {}
    for name, tensor in state.items():
        m = LAYER.match(name)
        if m:
            old = int(m.group(2))
            if old not in old_to_new:
                continue
            if experts is not None and (EXPERTS.match(name) or ROUTER.match(name)):
                tensor = tensor[torch.tensor(experts[old])]
            name = f"{m.group(1)}{old_to_new[old]}{m.group(3)}"
        out[name] = tensor
    return out


def build_student(teacher, *, keep_layers, experts=None, top_k=None):
    from transformers import AutoConfig, AutoModelForMultimodalLM

    data = teacher.config.to_dict()
    data["text_config"] = cut_text_config(data["text_config"], keep_layers=keep_layers, experts=experts, top_k=top_k)
    config = AutoConfig.for_model(**data)
    student = AutoModelForMultimodalLM.from_config(config).to(torch.bfloat16)
    state = cut_state_dict(teacher.state_dict(), keep_layers=keep_layers, experts=experts)
    missing, unexpected = student.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"student load: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return student


def tiny_teacher(config_path, seed=0, *, vocab_size=2048, experts=8, per_token=4):
    from transformers import AutoConfig, AutoModelForMultimodalLM

    data = json.loads(Path(config_path).read_text())
    data["text_config"].update(
        hidden_size=64, num_hidden_layers=6, layer_types=["linear_attention"] * 3 + ["full_attention"]
        + ["linear_attention", "full_attention"], num_attention_heads=2, num_key_value_heads=1,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        moe_intermediate_size=16, shared_expert_intermediate_size=16, num_experts=experts,
        num_experts_per_tok=per_token, hc_lowrank=8, ple_embed_dim=64, ngram_vocab_size_base=5000,
        split_ngram_parts=4, indexer_n_heads=2, indexer_budget=64, vocab_size=vocab_size)
    data["vision_config"].update(depth=1, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=64)
    data.pop("quantization_config", None)
    torch.manual_seed(seed)
    config = AutoConfig.for_model(**data)
    return AutoModelForMultimodalLM.from_config(config).to(torch.bfloat16)


def perturb(model, scale, seed):
    """Seeded relative noise on every floating tensor, so the tiny student starts measurably off the teacher."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in sorted(model.named_parameters()):
            if p.is_floating_point() and p.numel() > 1:
                p.add_(torch.randn(p.shape, generator=g).to(p.dtype) * scale * (p.float().std() + 1e-3).to(p.dtype))
    return model


def mtp_tiny(text_config, seed=3, experts=None, top_k=None):
    head = MtpHead(text_config, experts=experts, top_k=top_k)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in head.named_parameters():
            if "norm" in name:
                p.zero_()
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return head.float()


def tiny_student():
    torch.manual_seed(0)
    teacher = tiny_teacher(CONFIG)
    return perturb(build_student(teacher, keep_layers=KEEP, experts={l: [0, 2, 4, 6] for l in KEEP}, top_k=2), 0.5, 7)


def tiny_student_on_disk(path):
    """The tiny student as a trainer input, in the real student's on-disk layout (fused expert tensors under the
    model's own names, ``mtp.*`` in the same index; save_pretrained would write per-expert tensors) with
    ``student_cut`` in its config and an uncut MTP block (the teacher's 8 experts, top-4) as the real student's."""
    from safetensors.torch import save_file

    student = tiny_student()
    path = Path(path)
    path.mkdir(parents=True)
    student.config.save_pretrained(str(path))
    config = json.loads((path / "config.json").read_text())
    config["student_cut"] = {"source_layers": KEEP, "drop": [4]}
    # the real student keeps Flash's uncut MTP block (512 experts, top-10) beside its pruned layers
    config["text_config"].update(mtp_num_experts=8, mtp_num_experts_per_tok=4)
    (path / "config.json").write_text(json.dumps(config))
    state = {k: v.contiguous() for k, v in student.state_dict().items()}
    head = mtp_tiny(student.config.text_config, experts=8, top_k=4)
    state.update({"mtp." + k: v.to(torch.bfloat16).contiguous() for k, v in head.state_dict().items() if "inv_freq" not in k})
    save_file(state, str(path / "model-00001.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {k: "model-00001.safetensors" for k in state}}))
    return path
