"""segment_token_cache (MFU-1 25 Oct): piecewise cached encoding equals the whole-text encoding, bit for bit, on a
Qwen-shaped byte-level BPE (NFC, split regex, special and non-special added tokens); fallbacks; the main.py hook."""
import random

import pytest

tokenizers = pytest.importorskip("tokenizers")

from arc_infra import segment_token_cache_patch as patch  # noqa: E402
from arc_infra import segment_token_runtime as rt  # noqa: E402

SPECIAL = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|endoftext|>"]
PLAIN = ["<tool_call>", "</tool_call>", "<think>", "</think>"]  # added, not special (Qwen: special False)
QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)"
              r"|\s+")
WORDS = ["grid", "frame", "action", "café", "naïve", "ﬁle", "é", "  ", "\n", "\n\n", "def f(x):", "    return",
         "{\"row\": 3}", "12345", "ÀÉÎ", "日本語", "🙂", "\t", "'s", "x=1;", "[image]", "<tool", "|>", "<|im", "end|>"]


@pytest.fixture(scope="module")
def tok():
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    t = Tokenizer(models.BPE())
    t.normalizer = normalizers.NFC()
    t.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    t.decoder = decoders.ByteLevel()
    rng = random.Random(0)
    corpus = [" ".join(rng.choice(WORDS) for _ in range(40)) for _ in range(400)]
    t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=SPECIAL,
                                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    t.add_tokens(PLAIN)
    return t


def texts(n=60, seed=1):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(20, 200)):
            r = rng.random()
            parts.append(rng.choice(SPECIAL) if r < 0.15 else rng.choice(PLAIN) if r < 0.25 else rng.choice(WORDS))
            if rng.random() < 0.5:
                parts.append(rng.choice([" ", "", "\n"]))
        out.append("".join(parts))
    out.append("<|im_start|><|im_end|>" * 50)  # specials only, adjacent
    out.append("no special token at all " * 300)
    out.append("<|im_start|>user\n" + "x" * 5000 + "<|im_end|>\n<|im_start|>assistant\n")
    return out


def same(a, b):
    return (a.ids == b.ids and a.offsets == b.offsets and a.attention_mask == b.attention_mask
            and a.special_tokens_mask == b.special_tokens_mask and a.type_ids == b.type_ids and a.tokens == b.tokens)


def test_piecewise_equals_whole(tok):
    proxy = rt.SegmentTokenizer(tok, min_chars=1)
    for text in texts():
        for add in (False, True):  # no post-processor here: add_special_tokens adds nothing, cached path both ways
            assert same(proxy.encode(text, add_special_tokens=add), tok.encode(text, add_special_tokens=add))
    assert rt.STATS["piece_hits"] > 0  # repeated pieces came from the cache


def test_history_growth_hits_cache(tok):
    proxy = rt.SegmentTokenizer(tok, min_chars=1)
    turns = texts(30, seed=7)
    convo = ""
    for t in turns:  # an append-only conversation: every call re-encodes the whole history
        convo += "<|im_start|>user\n" + t + "<|im_end|>\n"
        before = rt.STATS["piece_hits"]
        assert proxy.encode(convo).ids == tok.encode(convo).ids
    assert rt.STATS["piece_hits"] - before > 0


def test_batch_paths(tok):
    proxy = rt.SegmentTokenizer(tok, min_chars=1)
    batch = texts(8, seed=3)
    got, want = proxy.encode_batch(batch, add_special_tokens=False), tok.encode_batch(batch, add_special_tokens=False)
    assert all(same(a, b) for a, b in zip(got, want))
    if hasattr(tok, "encode_batch_fast"):
        got = proxy.encode_batch_fast(batch, add_special_tokens=False)
        assert [g.ids for g in got] == [w.ids for w in tok.encode_batch_fast(batch, add_special_tokens=False)]


def test_fallbacks(tok):
    from tokenizers import processors

    proxy = rt.SegmentTokenizer(tok, min_chars=1000)
    short = "<|im_start|>hi<|im_end|>"
    calls = rt.STATS["cached_calls"]
    assert proxy.encode(short).ids == tok.encode(short).ids
    assert rt.STATS["cached_calls"] == calls  # under min_chars: the real call
    pair = proxy.encode("a" * 2000, "b" * 10)
    assert pair.ids == tok.encode("a" * 2000, "b" * 10).ids
    t2 = tokenizers.Tokenizer.from_str(tok.to_str())
    t2.post_processor = processors.TemplateProcessing(single="<|endoftext|> $A", special_tokens=[
        ("<|endoftext|>", t2.token_to_id("<|endoftext|>"))])
    p2 = rt.SegmentTokenizer(t2, min_chars=1)
    text = texts(1, seed=9)[0]
    assert p2.encode(text, add_special_tokens=True).ids == t2.encode(text, add_special_tokens=True).ids
    assert p2.encode(text, add_special_tokens=False).ids == t2.encode(text, add_special_tokens=False).ids
    proxy.enable_truncation(max_length=50)  # attribute set on the real tokenizer through the proxy
    try:
        long = texts(1, seed=11)[0] * 3
        assert proxy.encode(long).ids == tok.encode(long).ids and len(proxy.encode(long).ids) == 50
    finally:
        tok.no_truncation()


def test_lru_bound(tok):
    proxy = rt.SegmentTokenizer(tok, min_chars=1, max_tokens=200)
    for text in texts(20, seed=5):
        assert proxy.encode(text).ids == tok.encode(text).ids
    assert proxy._held[0] <= 200


def test_transformers_path(tok, monkeypatch):
    transformers = pytest.importorskip("transformers")
    monkeypatch.setenv("ARC_SEGMENT_TOKEN_CACHE", "1")
    monkeypatch.setattr(rt, "MIN_CHARS", 1)
    plain = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizers.Tokenizer.from_str(tok.to_str()))
    assert rt.install()
    wrapped = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizers.Tokenizer.from_str(tok.to_str()))
    assert isinstance(wrapped._tokenizer, rt.SegmentTokenizer)
    rt.uninstall()
    calls = rt.STATS["cached_calls"]
    for text in texts(10, seed=13):
        assert wrapped(text)["input_ids"] == plain(text)["input_ids"]
        assert wrapped([text])["input_ids"] == plain([text])["input_ids"]
        assert (wrapped(text, return_offsets_mapping=True)["offset_mapping"]
                == plain(text, return_offsets_mapping=True)["offset_mapping"])
        assert wrapped(text, add_special_tokens=False)["input_ids"] == plain(text, add_special_tokens=False)["input_ids"]
    assert rt.STATS["cached_calls"] > calls
    import copy  # vLLM's renderer thread pool (tokenizers/hf.py maybe_make_thread_pool) deep-copies the HF tokenizer
    import pickle

    for twin in (copy.deepcopy(wrapped), pickle.loads(pickle.dumps(wrapped))):
        assert isinstance(twin._tokenizer, rt.SegmentTokenizer)
        for text in texts(3, seed=17):
            assert twin(text)["input_ids"] == plain(text)["input_ids"]


MAIN_SNIPPET = b'''def main():
    import vllm.entrypoints.cli.serve
    from vllm.entrypoints.serve.utils.api_utils import (
        VLLM_SUBCMD_PARSER_EPILOG,
        cli_env_setup,
    )

    cli_env_setup()

    if "--omni" in sys.argv:
        pass
'''


def test_patch_hook(tmp_path):
    out = patch.patched_source(MAIN_SNIPPET).decode()
    assert "import vllm.arc_segment_tokens as _arc_segment_tokens  # " + patch.TAG in out
    assert out.index("cli_env_setup()\n    import vllm.arc_segment_tokens") > 0
    with pytest.raises(RuntimeError):
        patch.patched_source(out.encode())
    site = tmp_path / "site"
    (site / "vllm/entrypoints/cli").mkdir(parents=True)
    (site / patch.MAIN).write_bytes(MAIN_SNIPPET)
    ident = patch.apply(site, tmp_path / "out")
    assert ident and (site / patch.RUNTIME).read_bytes() == patch.runtime_source()
    assert patch.apply(site, tmp_path / "out") == ident  # second boot: marker, no re-edit
    (site / patch.MAIN).write_bytes(b"def main():\n    pass\n")
    assert patch.apply(site, tmp_path / "out2") is None  # unrecognized file: vLLM's path


def test_install_off(monkeypatch):
    monkeypatch.delenv("ARC_SEGMENT_TOKEN_CACHE", raising=False)
    assert rt.install() is False


def test_pinned_main_anchor():
    import os
    from pathlib import Path

    root = os.environ.get("ARC_PINNED_VLLM")
    if not root or not (Path(root) / patch.MAIN).is_file():
        pytest.skip("pinned vLLM tree not given")
    out = patch.patched_source((Path(root) / patch.MAIN).read_bytes())
    assert patch.TAG.encode() in out


def test_deepcopy_and_pickle_like_the_renderer_pool(tok):
    """v6n boot: vLLM's maybe_make_thread_pool deep-copies the tokenizer; the proxy's lock made it raise."""
    import copy
    import pickle

    proxy = rt.SegmentTokenizer(tok, min_chars=1)
    text = "hello <|im_start|>world<|im_end|> again " * 20
    want = tok.encode(text, add_special_tokens=False).ids
    twin = copy.deepcopy(proxy)
    assert isinstance(twin, rt.SegmentTokenizer) and twin._real is not proxy._real and twin._lru is proxy._lru
    assert twin.encode(text, add_special_tokens=False).ids == want == proxy.encode(text, add_special_tokens=False).ids
    back = pickle.loads(pickle.dumps(proxy))
    assert back.encode(text, add_special_tokens=False).ids == want
