"""arc_infra.token_segment_cache: piece-wise encodes equal whole encodes (G-long v5, 27 Oct; the real-prompt gate ran
on Modal CPU with Flash's tokenizer: 519 G-long v3 requests, image pads expanded, 3,000 fuzz cases, 0 mismatches)."""
import random

import pytest


def _tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

    corpus = ["the board has a red square", "def x(): return 1", "move up down left right", "日本 é 1234 {} <>"] * 50
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = tokenizers.trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, trainer)
    tok.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in
                            ("<|im_start|>", "<|im_end|>", "<|image_pad|>", "<|vision_start|>")])
    tok.add_tokens([AddedToken("<think>", special=False), AddedToken("</think>", special=False)])
    return tok


def test_pieces_encode_to_the_whole_encode_ids():
    tok = _tokenizer()
    from arc_infra import token_segment_cache as tsc

    rnd = random.Random(0)
    parts = ["<|im_start|>", "<|im_end|>", "<|image_pad|>", "<|vision_start|>", "<think>", "</think>", "a", " ", "\n",
             "  ", "board", " red", "日本", "é", "12", "{", "<", "|", "im_end"]
    for _ in range(500):
        text = "".join(rnd.choice(parts) for _ in range(rnd.randint(1, 80)))
        for add in (False, True):
            want = tok.encode_batch([text], add_special_tokens=add)[0].ids
            assert [e.ids for e in tsc.encode_batch(tok, [text], add_special_tokens=add)] == [want], text
    prompt = "<|im_start|>user\nthe board<|im_end|>\n" * 40
    tsc.encode_batch(tok, [prompt], add_special_tokens=False)
    before = tsc.STATS["encoded"]
    longer = prompt + "<|im_start|>assistant\n<think>move up</think><|im_end|>"
    assert tsc.encode_batch(tok, [longer], False)[0].ids == tok.encode(longer, add_special_tokens=False).ids
    assert tsc.STATS["encoded"] - before <= 3  # only the new pieces were encoded


def test_a_cut_or_padding_goes_to_the_tokenizer_whole():
    tok = _tokenizer()
    from arc_infra import token_segment_cache as tsc

    text = "<|im_start|>user\nthe board has a red square<|im_end|>" * 20
    tok.enable_truncation(max_length=10)
    assert tsc.encode_batch(tok, [text], False)[0].ids == tok.encode(text, add_special_tokens=False).ids
    tok.no_truncation()
    tok.enable_padding(length=4096)
    whole = tsc.STATS["whole"]
    tsc.encode_batch(tok, [text], False)
    assert tsc.STATS["whole"] == whole + 1


def test_patch_edits_the_one_encode_batch_call(tmp_path):
    from arc_infra import token_segment_patch as p

    src = ("class T:\n    def _encode_plus(self, batch):\n        encodings = self._tokenizer.encode_batch(\n"
           "            batch,\n            add_special_tokens=False,\n        )\n        return encodings\n")
    (tmp_path / "transformers").mkdir()
    (tmp_path / "transformers/tokenization_utils_tokenizers.py").write_text(src)
    ident = p.apply(tmp_path, tmp_path / "out")
    text = (tmp_path / "transformers/tokenization_utils_tokenizers.py").read_text()
    assert ident["patched_sha256"] and "arc_infra.token_segment_cache" in text and "self._tokenizer,\n" in text
    assert p.apply(tmp_path, tmp_path / "out") == ident  # applied once
    (tmp_path / "transformers/tokenization_utils_tokenizers.py").write_text(src.replace("encodings =", "x ="))
    assert p.apply(tmp_path, tmp_path / "out2") is None  # no anchor: skipped, never raised
