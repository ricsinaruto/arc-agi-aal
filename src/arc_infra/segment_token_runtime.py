"""ARC_SEGMENT_TOKEN_CACHE=1 (MFU-1 25 Oct): the API server's tokenizer encodes a long prompt as the concatenation of
its pieces between special added tokens, each piece from an LRU cache. v6k server 1 spent ~26% of its API CPU in
_encode_plus re-tokenizing whole conversation histories (the chat request and the per-turn /tokenize count).

Exact by construction: the Rust tokenizer itself first splits the text at added tokens that are special, not
normalized and without lstrip/rstrip/single_word (leftmost-longest), then normalizes and pre-tokenizes each piece on its
own, so encode(text) == merge(encode(piece) for piece) with growing offsets. Only plain calls take the cached path: one
str per input, not pre-tokenized, no truncation or padding on the tokenizer, and add_special_tokens adding nothing
(checked once per tokenizer); everything else goes to the real tokenizer. Shipped into the runtime as
vllm/arc_segment_tokens.py and imported at the CLI entry (segment_token_cache_patch)."""

import os
import re
import threading
from collections import OrderedDict

TAG = "ARC segment_token_cache"
MIN_CHARS = int(os.environ.get("ARC_SEGMENT_TOKEN_CACHE_MIN_CHARS", "4096"))  # shorter texts: the real call
MAX_TOKENS = int(os.environ.get("ARC_SEGMENT_TOKEN_CACHE_TOKENS", "8000000"))  # LRU bound (cached tokens)
STATS = {"calls": 0, "cached_calls": 0, "pieces": 0, "piece_hits": 0, "fallback": 0}


class SegmentTokenizer:
    """Proxy around a ``tokenizers.Tokenizer``: encode/encode_batch/encode_batch_fast take the cached path for plain
    long str inputs; every other attribute is the real tokenizer's."""

    def __init__(self, real, min_chars=MIN_CHARS, max_tokens=MAX_TOKENS):
        from tokenizers import Encoding

        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_merge", Encoding.merge)
        object.__setattr__(self, "_min", int(min_chars))
        object.__setattr__(self, "_max", int(max_tokens))
        object.__setattr__(self, "_lru", OrderedDict())
        object.__setattr__(self, "_held", [0])
        object.__setattr__(self, "_lock", threading.Lock())
        specials = []
        for tok in (real.get_added_tokens_decoder() or {}).values():
            if (getattr(tok, "special", False) and not getattr(tok, "normalized", True) and not tok.lstrip
                    and not tok.rstrip and not tok.single_word and tok.content):
                specials.append(tok.content)
        specials.sort(key=len, reverse=True)  # alternation picks the longest at the leftmost position
        object.__setattr__(self, "_split", re.compile("(" + "|".join(map(re.escape, specials)) + ")") if specials
                           else None)
        probe = "a b"
        object.__setattr__(self, "_adds", real.encode(probe, add_special_tokens=True).ids
                           != real.encode(probe, add_special_tokens=False).ids)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):  # copy/pickle protocol lookups must not reach the real one
            raise AttributeError(name)
        return getattr(self._real, name)

    def __deepcopy__(self, memo):
        """vLLM's renderer thread pool deep-copies the HF tokenizer (tokenizers/hf.py maybe_make_thread_pool; v6n
        boot, 25 Oct: the lock is not picklable). The copy wraps a deep copy of the real tokenizer and shares this
        cache and its lock (thread-safe), so every pool member hits the same pieces."""
        import copy

        new = object.__new__(type(self))
        for key, value in self.__dict__.items():
            object.__setattr__(new, key, copy.deepcopy(value, memo) if key == "_real" else value)
        return new

    def __reduce__(self):  # pickled (a process pool): a fresh proxy around the real tokenizer, own cache
        return (type(self), (self._real, self._min, self._max))

    def __setattr__(self, name, value):
        setattr(self._real, name, value)

    def _plain(self, text, add_special_tokens):
        return (isinstance(text, str) and len(text) >= self._min and self._split is not None
                and (not add_special_tokens or not self._adds)
                and self._real.truncation is None and self._real.padding is None)

    def _piece(self, piece):
        with self._lock:
            enc = self._lru.get(piece)
            if enc is not None:
                self._lru.move_to_end(piece)
                STATS["piece_hits"] += 1
                return enc
        enc = self._real.encode(piece, add_special_tokens=False)
        # growing offsets shift each piece by the previous piece's last offset end: exact only when that end is the
        # piece's length (not after a trailing combining sequence NFC folds, or a piece with no tokens)
        enc = (enc, bool(enc.offsets) and enc.offsets[-1][1] == len(piece))
        with self._lock:
            if piece not in self._lru:
                self._lru[piece] = enc
                self._held[0] += len(enc[0].ids)
                while self._held[0] > self._max and self._lru:
                    _, old = self._lru.popitem(last=False)
                    self._held[0] -= len(old[0].ids)
        return enc

    def _encode(self, text):  # add_special_tokens adds nothing on this path (checked in _plain)
        pieces = [p for p in self._split.split(text) if p]
        STATS["cached_calls"] += 1
        STATS["pieces"] += len(pieces)
        encs = [self._piece(p) for p in pieces]
        if len(encs) == 1:
            return encs[0][0]
        if not all(ok for _, ok in encs[:-1]):
            STATS["fallback"] += 1
            return self._real.encode(text, add_special_tokens=False)
        return self._merge([e for e, _ in encs], growing_offsets=True)

    def encode(self, sequence, pair=None, is_pretokenized=False, add_special_tokens=True):
        STATS["calls"] += 1
        if pair is None and not is_pretokenized and self._plain(sequence, add_special_tokens):
            return self._encode(sequence)
        STATS["fallback"] += 1
        return self._real.encode(sequence, pair, is_pretokenized=is_pretokenized,
                                 add_special_tokens=add_special_tokens)

    def _batch(self, method, inputs, is_pretokenized=False, add_special_tokens=True):
        STATS["calls"] += 1
        inputs = list(inputs)
        if not is_pretokenized and inputs and all(self._plain(x, add_special_tokens) for x in inputs):
            return [self._encode(x) for x in inputs]
        STATS["fallback"] += 1
        return getattr(self._real, method)(inputs, is_pretokenized=is_pretokenized,
                                           add_special_tokens=add_special_tokens)

    def encode_batch(self, inputs, is_pretokenized=False, add_special_tokens=True):
        return self._batch("encode_batch", inputs, is_pretokenized, add_special_tokens)

    def encode_batch_fast(self, inputs, is_pretokenized=False, add_special_tokens=True):
        return self._batch("encode_batch_fast", inputs, is_pretokenized, add_special_tokens)


def wrap(hf_tokenizer):
    """Put the proxy on a transformers fast tokenizer (its ``_tokenizer``); idempotent."""
    real = getattr(hf_tokenizer, "_tokenizer", None)
    if real is None or isinstance(real, SegmentTokenizer):
        return hf_tokenizer
    hf_tokenizer.__dict__["_tokenizer"] = SegmentTokenizer(real)
    return hf_tokenizer


_ORIGINAL_INIT = {}


def uninstall():
    """Restore the patched constructors (tests)."""
    for cls, init in _ORIGINAL_INIT.items():
        cls.__init__ = init
    _ORIGINAL_INIT.clear()


def install():
    """Every fast tokenizer built after this call gets the proxy (the renderer's and the HF processor's)."""
    if os.environ.get("ARC_SEGMENT_TOKEN_CACHE") != "1":
        return False
    import importlib

    done = []
    for mod in ("transformers.tokenization_utils_tokenizers", "transformers.tokenization_utils_fast"):
        try:
            m = importlib.import_module(mod)
        except Exception:  # noqa: BLE001 - one of the two exists per transformers major
            continue
        for name in ("TokenizersBackend", "PreTrainedTokenizerFast"):
            cls = getattr(m, name, None)
            if cls is None or getattr(cls.__init__, "_arc_segment", False):
                continue
            original = cls.__init__
            _ORIGINAL_INIT[cls] = original

            def init(self, *args, __original=original, **kwargs):
                __original(self, *args, **kwargs)
                try:
                    wrap(self)
                except Exception as error:  # noqa: BLE001 - a speed path, never a reason to fail a tokenizer
                    print(f"ARC_SEGMENT_TOKEN_CACHE_SKIPPED {type(error).__name__}: {error}", flush=True)

            init._arc_segment = True
            cls.__init__ = init
            done.append(f"{mod}.{name}")
    print(f"ARC_SEGMENT_TOKEN_CACHE on: {done}", flush=True)
    return bool(done)
