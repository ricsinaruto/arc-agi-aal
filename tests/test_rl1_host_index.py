"""U2 (27 Oct): the host index path (RL1_HOST_INDEX: M-RoPE positions, vision patch positions and block lengths, the
per-slice image counts and the prefix key read from the window's host tensors) gives the card path's numbers bitwise,
on a real multimodal window (Flash processor, 44,424 tokens, 5 images of 40x40 patches; ids and grids in the fixture)
through a small random Qwen4Exp (real classes): positions, image features, then a slice's log-probs, entropy and loss."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("transformers.models.qwen4_exp")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/posttrain"))
FIXTURE = Path(__file__).parent / "fixtures/u2_mm_window.npz"
IMAGE = 248056


def window():
    data = np.load(FIXTURE)
    ids = torch.from_numpy(data["input_ids"].astype(np.int64))[None]
    g = torch.Generator().manual_seed(0)
    pixels = torch.randn(int(data["pixel_rows"]), int(data["pixel_width"]), generator=g)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": torch.from_numpy(data["mm_token_type_ids"].astype(np.int64))[None],
            "pixel_values": pixels, "image_grid_thw": torch.from_numpy(data["image_grid_thw"])}


def build():
    from transformers.models.qwen4_exp import modeling_qwen4_exp as modeling
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

    torch.set_num_threads(2)
    lin, full = "linear_attention", "full_attention"
    text = dict(vocab_size=248320, hidden_size=32, num_hidden_layers=2, layer_types=[lin, full],
                num_attention_heads=4, num_key_value_heads=2, head_dim=16, partial_rotary_factor=0.25,
                linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
                moe_intermediate_size=16, shared_expert_intermediate_size=16, num_experts=4, num_experts_per_tok=2,
                hc_count=4, hc_lowrank=8, ngram_vocab_size_base=101, split_ngram_parts=4, indexer_n_heads=2,
                indexer_head_dim=8, indexer_kv_heads=1, indexer_budget=8, indexer_compress_ratio=4, eos_token_id=1,
                bos_token_id=1, pad_token_id=None, output_gate_type="sigmoid")
    vision = dict(depth=2, hidden_size=32, intermediate_size=32, num_heads=2, out_hidden_size=32)
    config = Qwen4ExpConfig(text_config=text, vision_config=vision)
    torch.manual_seed(0)
    model = modeling.Qwen4ExpForConditionalGeneration(config).float().eval()
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                param.normal_(0, 0.08)
    return model


def slice_numbers(model, piece, ids, start, stop):
    """The slice's next-token log-probs, entropy and a weighted loss, as the trainer's head reads them."""
    with torch.no_grad():
        logits = model(input_ids=ids[:, start:stop], **piece(start, stop)).logits[0, :-1].float()
    logp = torch.log_softmax(logits, -1)
    actor = logp.gather(-1, ids[0, start + 1:stop, None])[:, 0]
    entropy = -(logp.exp() * logp).sum(-1)
    w = torch.linspace(0.1, 1.0, actor.shape[0])
    return actor, entropy, -(w * actor).sum() / w.sum()


def test_host_index_path_matches_the_card_path_bitwise():
    import flash_qlora_job as fq
    import rl1_encode

    model = build()
    inputs = window()
    core = model.model
    how = {"cls": f"{type(core).__module__}.{type(core).__qualname__}",
           "merge": int(core.config.vision_config.spatial_merge_size)}
    made = rl1_encode.to_device({"inputs": inputs, "arrays": {}}, "cpu", how)
    host = made["host_inputs"]
    assert set(host) == set(rl1_encode.HOST_KEYS) and made["mrope"] is not None

    card = fq.image_features(model, inputs)
    hosted = fq.image_features(model, inputs, host)
    assert hosted.shape == (2000, 32) and torch.equal(card, hosted)

    before = fq.positions_and_images(model, inputs, features=card)
    after = fq.positions_and_images(model, made["inputs"], features=hosted, positions=made["mrope"], host=host)
    rule = fq.positions_and_images(model, inputs, features=hosted, host=host)  # no encoder positions: rule on host
    total = inputs["input_ids"].shape[1]
    full = before(0, total)["position_ids"]
    assert torch.equal(full, after(0, total)["position_ids"]) and torch.equal(full, rule(0, total)["position_ids"])
    assert int(full.max()) < total  # images compress positions (40x40 patches: 20 positions a side)

    ids = inputs["input_ids"]
    runs = torch.nonzero(torch.diff((ids[0] == IMAGE).int(), prepend=torch.tensor([0])) == 1).flatten().tolist()
    start = runs[-1] - 96
    for begin, end in ((start, start + 720), (runs[-1] + 100, runs[-1] + 900)):  # a whole image in one, part in other
        cuts = [(before, begin, end), (after, begin, end)]
        a, b = (slice_numbers(model, p, ids, s, e) for p, s, e in cuts)
        for x, y in zip(a, b):
            assert torch.equal(x, y)
        assert "mm_encoder_outputs" in before(begin, end) or begin > runs[-1]
    plain = lambda s, e: {**before(s, e), "position_ids": torch.arange(s, e).view(1, 1, -1).expand(3, 1, -1)}  # noqa: E731
    control = slice_numbers(model, plain, ids, runs[-1] + 100, runs[-1] + 900)[0]
    assert not torch.equal(control, b[0])  # the positions reach the numbers (text positions give others)


def test_prefix_key_reads_the_host_copy():
    import types

    import rl1_trainer as rt

    inputs = window()
    fake = types.SimpleNamespace(processor=types.SimpleNamespace(image_token_id=IMAGE))
    host = {k: inputs[k] for k in ("input_ids", "image_grid_thw", "pixel_values")}
    card = rt.Trainer.prefix_key(fake, inputs, 30000)
    hosted = rt.Trainer.prefix_key(fake, {"pixel_values": inputs["pixel_values"]}, 30000, host)
    assert card == hosted and card[1] > 0


def test_host_index_off_keeps_the_card_path(monkeypatch):
    import types

    import rl1_trainer as rt

    monkeypatch.setenv("RL1_HOST_INDEX", "0")
    fake = types.SimpleNamespace()
    assert rt.Trainer.mrope_how(fake) is None
    assert rt.Trainer.host_of(types.SimpleNamespace(mrope_how=lambda: None), {"host_inputs": {"x": 1}}) is None
