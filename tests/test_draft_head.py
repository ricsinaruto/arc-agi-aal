"""Draft-head fine-tune: selection semantics, a tiny end-to-end trainer run (CPU) and the runtime patch."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from arc_infra import draft_head_patch

torch = pytest.importorskip("torch")

from arena import draft_head_train as dht  # noqa: E402
from arena import weights_overlay as wo  # noqa: E402

HID, HC, HEADS, KV, HD, IH, ID, E, INTER, V = 32, 4, 4, 2, 16, 4, 8, 16, 16, 64


def tiny_tensors(gen):
    def r(*shape, scale=0.05):
        return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16)

    t = {
        "mtp.fc_embedding.weight": r(HID, HID, scale=0.2), "mtp.fc_hidden.weight": r(HID, HID, scale=0.2),
        "mtp.pre_fc_norm_embedding.weight": r(HID), "mtp.pre_fc_norm_hidden.weight": r(HC * HID),
        "mtp.hyper_connection_mixer.hc_norm.weight": r(HC * HID),
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight": r(8, HC * HID),
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": r(HC * HID, 8),
        "mtp.layers.0.mlp.experts.gate_up_proj": r(E, 2 * INTER, HID, scale=0.2),
        "mtp.layers.0.mlp.experts.down_proj": r(E, HID, INTER, scale=0.2),
        "mtp.layers.0.mlp.gate.weight": r(E, HID),
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": r(INTER, HID),
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": r(INTER, HID),
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": r(HID, INTER),
        "mtp.layers.0.mlp.shared_expert_gate.weight": r(1, HID),
        "mtp.layers.0.self_attn.q_proj.weight": r(HEADS * 2 * HD, HID, scale=0.2),
        "mtp.layers.0.self_attn.k_proj.weight": r(KV * HD, HID, scale=0.2),
        "mtp.layers.0.self_attn.v_proj.weight": r(KV * HD, HID, scale=0.2),
        "mtp.layers.0.self_attn.o_proj.weight": r(HID, HEADS * HD, scale=0.2),
        "mtp.layers.0.self_attn.q_norm.weight": r(HD), "mtp.layers.0.self_attn.k_norm.weight": r(HD),
        "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": r((IH + 1) * ID, HID, scale=0.3),
        "mtp.layers.0.self_attn.indexer.q_layernorm.weight": r(ID),
        "mtp.layers.0.self_attn.indexer.k_layernorm.weight": r(ID),
        "lm_head.weight": r(V, HID, scale=0.5), "model.language_model.embed_tokens.weight": r(V, HID, scale=1.0),
        "model.language_model.hyper_connection_mixer.hc_norm.weight": r(HC * HID),
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": r(8, HC * HID),
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight": r(HC * HID, 8),
    }
    for role in ("attn", "mlp"):
        p = f"mtp.layers.0.{role}_hyper_connection."
        t.update({p + "block_inject_weight.weight": r(HC, HC * HID), p + "hc_norm.weight": r(HC * HID),
                  p + "input_mix_weight_down.weight": r(8, HC * HID),
                  p + "input_mix_weight_up.weight": r(HC * HID, 8)})
    return t


def write_checkpoint(folder, tensors):
    folder.mkdir(parents=True)
    names = sorted(tensors)
    shards = {"model-bf16-00001.safetensors": names[: len(names) // 2],
              "model-bf16-00002.safetensors": names[len(names) // 2:]}
    weight_map = {}
    for shard, shard_names in shards.items():
        wo.write_safetensors(folder / shard, {n: ("BF16", list(tensors[n].shape),
                                                  tensors[n].contiguous().view(torch.int16).numpy().tobytes())
                                              for n in shard_names})
        weight_map.update({n: shard for n in shard_names})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (folder / "config.json").write_text(json.dumps({"text_config": {
        "hc_count": HC, "hidden_size": HID, "num_attention_heads": HEADS, "num_key_value_heads": KV,
        "head_dim": HD, "indexer_n_heads": IH, "indexer_head_dim": ID, "partial_rotary_factor": 0.25}}))
    files = [{"path": p.name, "size": p.stat().st_size, "sha256": wo.sha256_file(p)} for p in sorted(folder.iterdir())]
    (folder / "MODEL_MANIFEST.json").write_text(json.dumps({"repo": "tiny", "revision": "0", "files": files}))


def write_split(data, dump, split, lengths, gen):
    tokens, loss, offsets, index = [], [], [0], {}
    (dump / split).mkdir(parents=True)
    for i, length in enumerate(lengths):
        tokens.append(np.random.default_rng(i).integers(0, V, length).astype(np.int32))
        mask = np.zeros(length, np.uint8)
        mask[length // 3:] = 1
        loss.append(mask)
        offsets.append(offsets[-1] + length)
        rows = (torch.randn(length, HC * HID, generator=gen)).to(torch.bfloat16)
        rows.view(torch.int16).numpy().tofile(dump / split / f"{i:05d}.target.bin")
        index[str(i)] = {"ok": True, "tokens": length}
    (dump / split / "index.json").write_text(json.dumps(index))
    np.savez(data / f"{split}.npz", tokens=np.concatenate(tokens), loss=np.concatenate(loss),
             offsets=np.asarray(offsets, np.int64))


def build_model(tensors):
    trainable = {n for n in tensors if n.startswith("mtp.") and n not in dht.FROZEN_MTP}
    return dht.Draft({k: v.float().to(torch.bfloat16) for k, v in tensors.items()}, trainable, hc_count=HC,
                     heads=HEADS, kv_heads=KV, head_dim=HD, idx_heads=IH, idx_dim=ID, rot=HD // 4)


def test_selection_is_dense_for_short_contexts_and_bounded_for_long(monkeypatch):
    gen = torch.Generator().manual_seed(0)
    model = build_model(tiny_tensors(gen))
    raw = torch.randn(40, ID, generator=gen).to(torch.bfloat16)
    compressed = model.compress(raw)
    anchors = torch.arange(40)
    q = torch.randn(40, IH, ID, generator=gen).to(torch.bfloat16)
    idx, ok = model.select(q, compressed, anchors)
    for i in range(40):
        assert sorted(idx[i][ok[i]].tolist()) == list(range(i + 1))
    monkeypatch.setattr(dht, "BLOCK_TOPK", 2)
    idx, ok = model.select(q, compressed, anchors)
    for i in range(40):
        chosen = idx[i][ok[i]].tolist()
        assert all(0 <= c <= i for c in chosen) and len(set(chosen)) == len(chosen)
        tail = list(range(((i + 1) // 4) * 4, i + 1))
        assert set(tail) <= set(chosen)
        assert len(chosen) == min(2, (i + 1) // 4) * 4 + len(tail)


def test_anchors_stay_inside_generated_spans():
    loss = torch.tensor([0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=torch.bool)
    assert dht.anchors_for(loss, 3).tolist() == [0, 5, 6]
    assert dht.anchors_for(loss, 1).tolist() == [0, 1, 2, 5, 6, 7, 8]


def test_sampling_probs_top_p():
    logits = torch.full((1, 100), -30.0)
    logits[0, :3] = torch.tensor([5.0, 4.0, -5.0])
    ids, probs = dht.sampling_probs(logits)
    assert abs(float(probs.sum()) - 1) < 1e-6
    kept = {int(i): float(p) for i, p in zip(ids[0], probs[0]) if p > 0}
    assert set(kept) == {0, 1}


def test_tiny_end_to_end(tmp_path):
    gen = torch.Generator().manual_seed(0)
    tensors = tiny_tensors(gen)
    ckpt, data, dump, out = tmp_path / "ckpt", tmp_path / "data", tmp_path / "dump", tmp_path / "out"
    write_checkpoint(ckpt, tensors)
    data.mkdir()
    write_split(data, dump, "train", [48, 60, 37], gen)
    write_split(data, dump, "eval", [52, 44], gen)
    dht.main(["--model-dir", str(ckpt), "--data-dir", str(data), "--dump-dir", str(dump), "--out", str(out),
              "--device", "cpu", "--train-steps", "4", "--eval-every", "2", "--eval-subset", "1", "--lr", "1e-3",
              "--warmup", "1"])
    report = json.loads((out / "draft-head-train.json").read_text())
    assert report["train_steps_done"] == 4
    for key in ("eval_before", "eval_after"):
        m = report[key]
        assert m["anchors"] > 0 and 0 <= m["accepted_per_draft"] <= 3
        assert len(m["p_accept_step"]) == 3
    manifest = json.loads((out / "overlay/overlay-manifest.json").read_text())
    assert manifest["format"] == wo.FORMAT and all(n.startswith("mtp.") for n in manifest["tensors"])
    assert not any(n in manifest["tensors"] for n in dht.FROZEN_MTP)
    # The overlay materialises against the tiny checkpoint (the arena.weights_overlay path).
    inventory = json.loads((ckpt / "MODEL_MANIFEST.json").read_text())
    base = {"repo": "tiny", "revision": "0",
            "files": {r["path"]: {"bytes": r["size"], "sha256": r["sha256"]} for r in inventory["files"]}}
    record = wo.materialise(ckpt, out / "overlay", tmp_path / "patched", base_inventory=base)
    assert record["tensors"] == len(manifest["tensors"])


def test_fake_nvfp4_is_blockwise_and_idempotent():
    gen = torch.Generator().manual_seed(1)
    w = torch.randn(3, 4, 32, generator=gen).to(torch.bfloat16)
    q = dht.fake_nvfp4(w)
    assert q.shape == w.shape and q.dtype == w.dtype
    assert float((q.float() - w.float()).norm() / w.float().norm()) < 0.2
    # at most 15 distinct values per 16-wide block (E2M1 magnitudes x sign), and a second pass changes nothing
    assert max(len(set(block.tolist())) for block in q.float().reshape(-1, 16)) <= 15
    assert float((dht.fake_nvfp4(q).float() - q.float()).abs().max()) < 1e-3 * float(q.float().abs().max())


def test_splits_vocab_served_rows_checkpoints_and_init_overlay(tmp_path):
    gen = torch.Generator().manual_seed(0)
    tensors = tiny_tensors(gen)
    ckpt, data, dump, out = tmp_path / "ckpt", tmp_path / "data", tmp_path / "dump", tmp_path / "out"
    write_checkpoint(ckpt, tensors)
    data.mkdir()
    write_split(data, dump, "train", [48, 60, 37], gen)
    write_split(data, dump, "val", [40, 44], gen)
    write_split(data, dump, "eval", [52, 44, 39], gen)
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"ids": list(range(0, V, 2))}))
    # "Served" step-1 rows of eval sequence 0: the re-implementation's own rows with NVFP4 experts, shuffled, with
    # a later duplicate of one position (the first row of a position is the one compared).
    model = build_model(tensors)
    model.use_experts("nvfp4")
    split = dht.Split(data, dump, "eval")
    tokens, loss, h = split.sequence(0, HC * HID, torch.device("cpu"))
    anchors = dht.anchors_for(loss, 3)
    with torch.no_grad():
        rows = model(h, tokens, anchors, 1, grad_ckpt=False)[0].to(torch.bfloat16)
    order = torch.randperm(len(anchors), generator=gen)
    pos = torch.cat([anchors[order], anchors[:1]]).numpy().astype(np.int32)
    rows = torch.cat([rows[order], torch.zeros_like(rows[:1])])
    pos.tofile(dump / "eval" / "00000.draft.pos")
    rows.view(torch.int16).numpy().tofile(dump / "eval" / "00000.draft.bin")
    common = ["--model-dir", str(ckpt), "--data-dir", str(data), "--dump-dir", str(dump), "--device", "cpu",
              "--experts", "nvfp4", "--alt-experts", "bf16", "--draft-vocab", str(vocab), "--select-split", "val",
              "--report-splits", "eval", "val", "--draft-splits", "eval", "--eval-subset", "2"]
    dht.main([*common, "--out", str(out), "--train-steps", "4", "--eval-every", "2", "--lr", "1e-3", "--warmup", "1",
              "--curve-stride", "2", "--checkpoint-every", "2", "--save-accept"])
    report = json.loads((out / "draft-head-train.json").read_text())
    assert report["sequences"] == {"train": 3, "val": 2, "eval": 3}
    for tag in ("before", "after"):
        rows = report["evals"][tag]
        assert set(rows) == {"eval", "val", "probe_bf16"} and set(rows["eval"]) == {"full", "v16k", "primary"}
        assert rows["eval"]["primary"] == rows["eval"]["v16k"]
        assert len(rows["eval"]["v16k"]["tokens_per_iteration"]) == 3
        bits = np.load(out / f"accept-eval-{tag}.npy")
        assert len(bits) == 52 + 44 + 39 and int((bits != 255).sum()) == rows["eval"]["v16k"]["anchors"]
        assert int((bits == 1).sum()) == round(rows["eval"]["v16k"]["top1_match_step"][0] * rows["eval"]["v16k"]["anchors"])
    assert report["eval_before"] == report["evals"]["before"]["eval"]["primary"]
    assert [row["step"] for row in report["curve"]] == [0, 2, 4] and set(report["curve"][1]) == {"step", "val", "curve"}
    assert sorted(p.name for p in (out / "checkpoints").iterdir()) == ["step-00002.pt", "step-00004.pt"]
    served = report["served_step1"]
    assert served["head"] == "before" and served["splits"]["eval"]["nvfp4"]["anchors"] == len(anchors)
    assert served["splits"]["eval"]["nvfp4"]["v16k"]["argmax_agree"] == 1.0
    assert served["splits"]["eval"]["nvfp4"]["cos_mean"] > 0.999
    assert served["splits"]["eval"]["bf16"]["cos_mean"] < served["splits"]["eval"]["nvfp4"]["cos_mean"]
    # A second run evaluates the exported head without training: its "after" equals the first run's.
    again = tmp_path / "again"
    dht.main([*common, "--out", str(again), "--train-steps", "0", "--init-overlay", str(out / "overlay"),
              "--served-head", "after"])
    second = json.loads((again / "draft-head-train.json").read_text())
    assert second["train_steps_done"] == 0 and "export" not in second
    assert second["init_overlay"]["overlay_sha256"] == report["export"]["overlay_sha256"]
    assert second["evals"]["before"]["eval"] == report["evals"]["before"]["eval"]
    assert second["evals"]["after"]["eval"] == report["evals"]["after"]["eval"]
    assert second["served_step1"]["head"] == "after"


PINNED = os.environ.get("ARC_PINNED_SITE")


@pytest.mark.skipif(not PINNED, reason="set ARC_PINNED_SITE to the extracted serving runtime's dist-packages")
def test_patch_applies_to_pinned_runtime(tmp_path):
    import shutil

    site = tmp_path / "site"
    for rel in ("vllm/models/qwen3_8_flash_next/nvidia/model.py", "vllm/models/qwen3_8_flash_next/nvidia/mtp.py"):
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(PINNED) / rel, site / rel)
    identity = draft_head_patch.apply(site, tmp_path / "out", {"dump": True})
    assert set(identity["files"]) == {"vllm/models/qwen3_8_flash_next/nvidia/model.py",
                                      "vllm/models/qwen3_8_flash_next/nvidia/mtp.py"}
    assert draft_head_patch.server_flags(draft_head_patch.validate({"dump": True})) == ["--enforce-eager"]
    text = (site / "vllm/models/qwen3_8_flash_next/nvidia/mtp.py").read_text()
    assert "_arc_dump_draft" in text and "_arc_overlay_weights" not in text


@pytest.mark.skipif(not PINNED, reason="set ARC_PINNED_SITE to the extracted serving runtime's dist-packages")
def test_overlay_patch_stacks_on_the_phase4_serving_kernels(tmp_path):
    import shutil

    from arc_infra import serving_kernels_patch as sk
    from arena.serving import bundle_draft_head

    rel = "vllm/models/qwen3_8_flash_next/nvidia/mtp.py"
    site = tmp_path / "site"
    (site / rel).parent.mkdir(parents=True)
    shutil.copyfile(Path(PINNED) / rel, site / rel)
    (site / rel).write_bytes(sk.patched_source(sk.MTP, (site / rel).read_bytes()))
    overlay = tmp_path / "bundle/draft-head/overlay"
    overlay.mkdir(parents=True)
    wo.write_safetensors(overlay / "overlay.safetensors", {"mtp.fc_hidden.weight": ("BF16", [2, 2], b"\0" * 8)})
    manifest = {"overlay": {"file": "overlay.safetensors", "sha256": wo.sha256_file(overlay / "overlay.safetensors")},
                "tensors": {"mtp.fc_hidden.weight": {}}}
    (overlay / "overlay-manifest.json").write_text(json.dumps(manifest))
    spec = {"dump": False, "overlay": {"path": "draft-head/overlay",
                                       "sha256": wo.sha256_file(overlay / "overlay-manifest.json")}}
    resolved = bundle_draft_head(spec, tmp_path / "bundle")
    identity = draft_head_patch.apply(site, tmp_path / "out", resolved)
    text = (site / rel).read_text()
    assert "_arc_overlay_weights" in text and "_arc_draft_quant_hook" in text and "_arc_dump_draft" not in text
    assert identity["settings"]["overlay"]["tensors"] == 1 and draft_head_patch.server_flags(
        draft_head_patch.validate(resolved)) == []


def test_patch_validation():
    assert draft_head_patch.validate(None) is None
    assert draft_head_patch.validate({"dump": False}) is None
    with pytest.raises(ValueError):
        draft_head_patch.validate({"dump": "yes"})
    with pytest.raises(ValueError):
        draft_head_patch.validate({"overlay": {"path": "x"}})
    spec = draft_head_patch.validate({"overlay": {"path": "/o", "sha256": "ab"}})
    assert list(draft_head_patch.edits(spec)) == ["vllm/models/qwen3_8_flash_next/nvidia/mtp.py"]
    assert hashlib.sha256(b"").hexdigest() not in draft_head_patch.MTP_SHA256


def test_dump_client_orders_rows_and_records(tmp_path, monkeypatch):
    from arena import draft_head

    width = 6
    data, dump = tmp_path / "data", tmp_path / "dump"
    data.mkdir()
    dump.mkdir()
    lengths = [5, 9]
    tokens = np.arange(sum(lengths), dtype=np.int32)
    np.savez(data / "eval.npz", tokens=tokens, loss=np.ones_like(tokens, dtype=np.uint8),
             offsets=np.asarray([0, 5, 14], np.int64))

    class Response:
        def __init__(self, n):
            self.n = n

        def raise_for_status(self):
            return None

        def json(self):
            return {"usage": {"prompt_tokens": self.n}}

    class Client:
        def post(self, url, json=None):
            if url.endswith("/completions"):
                n = len(json["prompt"])
                # Two chunks, the second first, plus one duplicated padding row at position 0.
                order = [*range(3, n), *range(3), 0]
                rows = np.stack([np.full(width, p, np.int16) for p in order])
                if (dump / "ACTIVE").exists():
                    np.asarray(order, np.int32).tofile(dump / "current.target.pos")
                    rows.tofile(dump / "current.target.bin")
                return Response(n)
            return Response(0)

    monkeypatch.setattr(draft_head, "WIDTH", width)
    record = {"stopped": {}, "dumped": {}, "tokens": {}}
    settings = {**draft_head.DEFAULTS, "verify_sequences": 0, "min_free_gib": 0}
    index = draft_head.dump_split(Client(), "http://x/v1", "eval", data, dump, settings, 1e12, record)
    assert all(v["ok"] for v in index.values()), index
    for i, n in enumerate(lengths):
        rows = np.fromfile(dump / "eval" / f"{i:05d}.target.bin", dtype=np.int16).reshape(n, width)
        assert rows[:, 0].tolist() == list(range(n))
    assert record["tokens"]["eval"] == 14 and not (dump / "ACTIVE").exists()
