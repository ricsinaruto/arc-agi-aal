"""scripts/sft/export.py: the SFT recipe's trained MTP block and embed_tokens / lm_head land in the overlay in BF16."""
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

spec = importlib.util.spec_from_file_location("sft_export", Path(__file__).resolve().parents[1] / "scripts/sft/export.py")
sft_export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sft_export)
EMB, HEAD = "model.language_model.embed_tokens.weight", "lm_head.weight"


def student(tmp_path):
    torch.manual_seed(0)
    stu = tmp_path / "student"
    stu.mkdir()
    t = {"mtp.layers.0.mlp.experts.gate_up_proj": torch.randn(3, 16, 8).to(torch.bfloat16),
         "mtp.layers.0.mlp.experts.down_proj": torch.randn(3, 8, 8).to(torch.bfloat16),
         "mtp.layers.0.mlp.gate.weight": torch.randn(3, 8).to(torch.bfloat16),
         "mtp.fc_hidden.weight": torch.randn(8, 16).to(torch.bfloat16),
         EMB: torch.randn(32, 8).to(torch.bfloat16), HEAD: torch.randn(32, 8).to(torch.bfloat16),
         "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": torch.randn(4, 8).to(torch.bfloat16),
         "model.visual.pos_embed.weight": torch.randn(4, 8).to(torch.bfloat16)}
    save_file(t, str(stu / "s.safetensors"))
    (stu / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "s.safetensors" for k in t}}))
    (stu / "config.json").write_text(json.dumps({"student_cut": {"source_layers": [0, 1], "drop": []}}))
    return stu, t


def run(tmp_path, stu, names):
    idx = tmp_path / "served.json"
    idx.write_text(json.dumps({"weight_map": {n: "x" for n in names}}))
    out = tmp_path / "out"
    counts = sft_export.export(str(stu), str(out), served_index=str(idx), prune_file=None, router_src=None, search=0)
    ov = json.loads((out / "overlay.json").read_text())
    got = {}
    for file in set(ov["weight_map"].values()):
        with safe_open(str(out / file), framework="pt") as f:
            got.update({k: f.get_tensor(k) for k in f.keys()})
    return counts, ov, got


def test_mtp_and_embeddings_exported_bf16(tmp_path):
    stu, t = student(tmp_path)
    mtp = [f"mtp.layers.0.mlp.experts.{e}.{p}.weight" for e in range(3) for p in ("gate_proj", "up_proj", "down_proj")]
    names = mtp + ["mtp.layers.0.mlp.gate.weight", "mtp.fc_hidden.weight", EMB, HEAD]
    counts, ov, got = run(tmp_path, stu, names)
    assert ov["format"] == "arc-heal-overlay-v1" and ov["fmt"] == "autoround"
    assert set(got) == set(ov["weight_map"]) == set(names)
    assert all(v.dtype == torch.bfloat16 for v in got.values())
    gu, down = t["mtp.layers.0.mlp.experts.gate_up_proj"], t["mtp.layers.0.mlp.experts.down_proj"]
    for e in range(3):
        assert torch.equal(got[f"mtp.layers.0.mlp.experts.{e}.gate_proj.weight"], gu[e][:8])
        assert torch.equal(got[f"mtp.layers.0.mlp.experts.{e}.up_proj.weight"], gu[e][8:])
        assert torch.equal(got[f"mtp.layers.0.mlp.experts.{e}.down_proj.weight"], down[e])
    for k in (EMB, HEAD, "mtp.fc_hidden.weight", "mtp.layers.0.mlp.gate.weight"):
        assert torch.equal(got[k], t[k])
    assert counts == {"bf16": 6, "experts": 0, "router_rows": 0, "skipped_frozen": 2}  # PLE + visual


def test_embeddings_kept_frozen_when_not_served(tmp_path):
    stu, _ = student(tmp_path)
    mtp = [f"mtp.layers.0.mlp.experts.{e}.{p}.weight" for e in range(3) for p in ("gate_proj", "up_proj", "down_proj")]
    counts, _, got = run(tmp_path, stu, mtp + ["mtp.layers.0.mlp.gate.weight", "mtp.fc_hidden.weight"])
    assert EMB not in got and HEAD not in got and counts["skipped_frozen"] == 4


def test_trainer_checkpoint_layout_exports_trained_values(tmp_path):
    """A train.py checkpoint: trained tensors in its own file, the index pointing at the base student's file (absolute
    path) for the rest; the base file's stale copy of a trained tensor is skipped."""
    stu, t = student(tmp_path)
    ck = tmp_path / "ckpt"
    ck.mkdir()
    new = (t[HEAD].float() + 1).to(torch.bfloat16)
    save_file({HEAD: new}, str(ck / "trained-00001.safetensors"))
    wm = {k: str(stu / "s.safetensors") for k in t}
    wm[HEAD] = "trained-00001.safetensors"
    (ck / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    (ck / "config.json").write_text((stu / "config.json").read_text())
    mtp = [f"mtp.layers.0.mlp.experts.{e}.{p}.weight" for e in range(3) for p in ("gate_proj", "up_proj", "down_proj")]
    names = mtp + ["mtp.layers.0.mlp.gate.weight", "mtp.fc_hidden.weight", EMB, HEAD]
    _, ov, got = run(tmp_path, ck, names)
    assert torch.equal(got[HEAD], new) and torch.equal(got[EMB], t[EMB]) and set(ov["weight_map"]) == set(names)


def test_nvfp4_dense_replaces_trained_non_expert_tensors(tmp_path):
    """export_nvfp4_dense: trained tensors replace the NVFP4 base's non-expert shard values (router rows scattered to
    the kept experts), untrained ones keep the base's, routed experts are not written, unknown names fail."""
    torch.manual_seed(0)
    base = tmp_path / "base"
    base.mkdir()
    R = "model.language_model.layers.0.mlp.gate.weight"
    Q = "model.language_model.layers.0.self_attn.q_proj.weight"
    MG = "mtp.layers.0.mlp.experts.gate_up_proj"
    N = "model.language_model.layers.0.input_layernorm.weight"
    bt = {R: torch.randn(4, 8).to(torch.bfloat16), Q: torch.randn(8, 8).to(torch.bfloat16),
          MG: torch.randn(4, 16, 8).to(torch.bfloat16), N: torch.randn(8).to(torch.bfloat16)}
    save_file(bt, str(base / "model-bf16-00001.safetensors"), metadata={"format": "pt"})
    ex = {"model.language_model.layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(2, 2, dtype=torch.uint8)}
    save_file(ex, str(base / "layer-00000-experts-0000-0003.safetensors"))
    wm = {**{k: "model-bf16-00001.safetensors" for k in bt}, **{k: "layer-00000-experts-0000-0003.safetensors" for k in ex}}
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    prune = tmp_path / "prune.json"
    prune.write_text(json.dumps({"experts": 4, "pruned": {"0": [1, 2]}}))
    ck = tmp_path / "ckpt"
    ck.mkdir()
    tr = {R: torch.randn(2, 8).to(torch.bfloat16), Q: torch.randn(8, 8).to(torch.bfloat16),
          MG: torch.randn(4, 16, 8).to(torch.bfloat16),
          "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 4, 8).to(torch.bfloat16)}
    old = tmp_path / "stu.safetensors"
    save_file({N: torch.randn(8).to(torch.bfloat16)}, str(old))

    def write(t):
        save_file(t, str(ck / "trained-00001.safetensors"))
        (ck / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {**{k: "trained-00001.safetensors" for k in t}, N: str(old)}}))

    write(tr)
    (ck / "config.json").write_text(json.dumps({"student_cut": {"source_layers": [0], "drop": []}}))
    out = tmp_path / "out"
    counts = sft_export.export_nvfp4_dense(str(ck), str(base), str(out), prune_file=str(prune))
    man = json.loads((out / "overlay-manifest.json").read_text())
    assert set(man["shards"]) == {"model-bf16-00001.safetensors"}
    assert counts == {"replaced": 3, "router_rows": 2, "kept_base": 1}
    with safe_open(str(out / "model-bf16-00001.safetensors"), framework="pt") as f:
        got = {k: f.get_tensor(k) for k in f.keys()}
    assert torch.equal(got[Q], tr[Q]) and torch.equal(got[MG], tr[MG]) and torch.equal(got[N], bt[N])
    assert torch.equal(got[R][[0, 3]], tr[R]) and torch.equal(got[R][[1, 2]], bt[R][[1, 2]])
    write({**tr, "model.language_model.layers.0.unknown.weight": torch.zeros(2)})
    with pytest.raises(KeyError):
        sft_export.export_nvfp4_dense(str(ck), str(base), str(tmp_path / "out2"), prune_file=str(prune))


def test_nvfp4_dense_maps_a_cut_students_layers_to_source_ids(tmp_path):
    """Student layer 0 is source layer 1: its tensors land on the base's layer 1; dropped layer 0 keeps the base's."""
    torch.manual_seed(0)
    base = tmp_path / "base"
    base.mkdir()
    name = "model.language_model.layers.{}.self_attn.q_proj.weight"
    gate = "model.language_model.layers.{}.mlp.gate.weight"
    bt = {k.format(i): torch.randn(4, 8).to(torch.bfloat16) for i in (0, 1) for k in (name, gate)}
    save_file(bt, str(base / "model-bf16-00001.safetensors"), metadata={"format": "pt"})
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model-bf16-00001.safetensors" for k in bt}}))
    prune = tmp_path / "prune.json"
    prune.write_text(json.dumps({"experts": 4, "pruned": {"0": [0, 1], "1": [1, 2]}}))
    ck = tmp_path / "ckpt"
    ck.mkdir()
    tr = {name.format(0): torch.randn(4, 8).to(torch.bfloat16), gate.format(0): torch.randn(2, 8).to(torch.bfloat16)}
    save_file(tr, str(ck / "trained-00001.safetensors"))
    (ck / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "trained-00001.safetensors" for k in tr}}))
    (ck / "config.json").write_text(json.dumps({"student_cut": {"source_layers": [1], "drop": [0]}}))
    out = tmp_path / "out"
    sft_export.export_nvfp4_dense(str(ck), str(base), str(out), prune_file=str(prune))
    with safe_open(str(out / "model-bf16-00001.safetensors"), framework="pt") as f:
        got = {k: f.get_tensor(k) for k in f.keys()}
    assert torch.equal(got[name.format(1)], tr[name.format(0)]) and torch.equal(got[name.format(0)], bt[name.format(0)])
    assert torch.equal(got[gate.format(1)][[0, 3]], tr[gate.format(0)]) and torch.equal(got[gate.format(0)], bt[gate.format(0)])
