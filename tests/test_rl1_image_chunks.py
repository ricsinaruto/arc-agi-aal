"""Prefix chunking at images (rl1-prof-m1cr1 floor: 8K vs 16K chunks moved logprobs up to 7.45):
on a tiny qwen4_exp VL model in fp32, the no-grad prefix in chunks whose cut splits an image (positions_and_images
hands each slice its image tokens' features) gives the tail's logprobs of one unchunked pass."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "posttrain"))
modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
import flash_qlora_job as fq  # noqa: E402

V, IMG, VS, VE = 96, 90, 91, 92


def tiny_vl():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

    text = dict(vocab_size=V, hidden_size=32, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
                num_attention_heads=2, num_key_value_heads=1, head_dim=16, partial_rotary_factor=0.25,
                linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
                moe_intermediate_size=16, shared_expert_intermediate_size=16, num_experts=8, num_experts_per_tok=2,
                hc_count=4, hc_lowrank=8, ple_embed_dim=32, ple_layer_ids=[2], ngram_vocab_size_base=101,
                split_ngram_parts=4, indexer_n_heads=2, indexer_head_dim=8, indexer_kv_heads=1, indexer_budget=8,
                indexer_compress_ratio=4, eos_token_id=1, bos_token_id=1, pad_token_id=None)
    vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, patch_size=4, spatial_merge_size=2,
                  temporal_patch_size=2, out_hidden_size=32, num_position_embeddings=64)
    config = Qwen4ExpConfig(text_config=text, vision_config=vision, image_token_id=IMG, video_token_id=93,
                            vision_start_token_id=VS, vision_end_token_id=VE)
    torch.manual_seed(0)
    model = modeling.Qwen4ExpForConditionalGeneration(config).float().eval()
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                param.normal_(0, 0.08)
    return model


def inputs_with_images(length, starts, g=4):
    """Text ids with an image of g*g tokens (vision start / end around it) at each of ``starts``."""
    gen = torch.Generator().manual_seed(3)
    ids = torch.randint(2, 88, (1, length), generator=gen)
    types = torch.zeros_like(ids)
    for s in starts:
        ids[0, s] = VS
        ids[0, s + 1:s + 1 + g * g] = IMG
        ids[0, s + 1 + g * g] = VE
        types[0, s + 1:s + 1 + g * g] = 1
    grid = torch.tensor([[1, 2 * g, 2 * g]] * len(starts))
    pixels = torch.randn(int(grid.prod(-1).sum()), 3 * 2 * 4 * 4, generator=gen)
    if not starts:
        return {"input_ids": ids, "mm_token_type_ids": types}
    return {"input_ids": ids, "mm_token_type_ids": types, "image_grid_thw": grid, "pixel_values": pixels}


def tail_logprobs(model, inputs, cut, chunk):
    ids = inputs["input_ids"]
    piece = fq.positions_and_images(model, inputs)
    cache = None
    with torch.no_grad():
        for s in range(0, cut, chunk):
            e = min(s + chunk, cut)
            out = model(input_ids=ids[:, s:e], attention_mask=torch.ones_like(ids[:, :e]), past_key_values=cache,
                        use_cache=True, logits_to_keep=1, **piece(s, e))
            cache = out.past_key_values
        out = model(input_ids=ids[:, cut:], attention_mask=torch.ones_like(ids), past_key_values=cache, use_cache=True,
                    **piece(cut, ids.shape[1]))
    logp = torch.log_softmax(out.logits.float(), -1)[0, :-1]
    return logp.gather(-1, ids[0, cut + 1:, None])[:, 0]


@pytest.mark.parametrize("starts", [[], [56], [20, 56], [60, 100]])
def test_prefix_chunks_split_images_exactly(starts):
    model = tiny_vl()
    inputs = inputs_with_images(192, starts)
    whole = tail_logprobs(model, inputs, 128, 128)
    for chunk in (64, 32):  # the cut at 64 (and 32, 96) runs through the images at 56 / 60 / 100
        d = (tail_logprobs(model, inputs, 128, chunk) - whole).abs()
        assert float(d.max()) < 1e-4, (starts, chunk, float(d.max()), float(d.mean()))
