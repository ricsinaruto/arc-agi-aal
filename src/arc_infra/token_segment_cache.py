"""Segment-cached prompt tokenization for the vLLM API process (ARC_TOKEN_SEGMENT_CACHE=1, token_segment_patch).

G-long v5 (27 Oct): with one API process, 51% of its samples sat in the Rust tokenizer re-encoding every request's whole
conversation (contexts 27K and growing); the engine waited on the API. A Hugging Face tokenizer cuts its input at added
special tokens before normalizing and pre-tokenizing, so encoding the pieces between them one by one gives the same ids.
Each turn's prompt is the previous one plus new messages: the pieces seen before come from a cache and only the new ones
are encoded. ``encode_batch`` is a drop-in for ``tokenizers.Tokenizer.encode_batch``; anything it cannot prove equal
(pairs, pre-tokenized input, padding, a post-processor that adds tokens, special tokens split as text, stripping added
tokens, a truncation that would cut) goes to the tokenizer unchanged."""

import os
import re
import threading
from collections import OrderedDict

MIN_CHARS = int(os.environ.get("ARC_TOKEN_SEGMENT_MIN_CHARS") or 4096)  # shorter prompts: encoded whole
CACHE_TOKENS = int(os.environ.get("ARC_TOKEN_SEGMENT_CACHE_TOKENS") or 40_000_000)  # LRU bound (~100 B/token)

_lock = threading.Lock()
_cache = OrderedDict()  # (fingerprint, piece) -> Encoding
_cached_tokens = [0]
_plans = {}  # id(tokenizer) -> (fingerprint, split regex) or None (not provably equal: always whole)
STATS = {"calls": 0, "segmented": 0, "pieces": 0, "encoded": 0, "whole": 0}


def _plan(tok):
    """(fingerprint, regex splitting at special added tokens), or None when a piece-wise encode could differ."""
    key = id(tok)
    if key in _plans:
        return _plans[key]
    plan = None
    try:
        added = tok.get_added_tokens_decoder()
        special = [t for t in added.values() if t.special]
        safe = all(not (t.lstrip or t.rstrip or t.single_word) for t in special) and special
        if safe and tok.encode("a b", add_special_tokens=True).ids == tok.encode("a b", add_special_tokens=False).ids:
            contents = sorted({t.content for t in special}, key=len, reverse=True)
            plan = ((tok.get_vocab_size(with_added_tokens=True), tuple(contents)),
                    re.compile("(" + "|".join(re.escape(c) for c in contents) + ")"))
    except Exception:  # noqa: BLE001 - an unknown tokenizer: never segmented
        plan = None
    _plans[key] = plan
    return plan


def _whole(tok, batch, add_special_tokens, is_pretokenized):
    STATS["whole"] += 1
    return tok.encode_batch(batch, add_special_tokens=add_special_tokens, is_pretokenized=is_pretokenized)


def encode_batch(tok, batch, add_special_tokens=True, is_pretokenized=False):
    from tokenizers import Encoding

    STATS["calls"] += 1
    plan = _plan(tok)
    if (plan is None or is_pretokenized or getattr(tok, "padding", None) is not None
            or getattr(tok, "encode_special_tokens", False)
            or any(not isinstance(t, str) or len(t) < MIN_CHARS for t in batch)):
        return _whole(tok, batch, add_special_tokens, is_pretokenized)
    fp, split = plan
    pieces = [[p for p in split.split(text) if p] for text in batch]
    with _lock:
        hits = {p: _cache.get((fp, p)) for ps in pieces for p in ps}
    todo = [p for p, e in hits.items() if e is None]
    if todo:
        fresh = tok.encode_batch(todo, add_special_tokens=False)
        with _lock:
            for p, e in zip(todo, fresh):
                hits[p] = e
                _cache[(fp, p)] = e
                _cached_tokens[0] += len(e.ids)
            while _cached_tokens[0] > CACHE_TOKENS and _cache:
                _, old = _cache.popitem(last=False)
                _cached_tokens[0] -= len(old.ids)
    with _lock:
        for ps in pieces:
            for p in ps:
                if (fp, p) in _cache:
                    _cache.move_to_end((fp, p))
    out = [Encoding.merge([hits[p] for p in ps], growing_offsets=True) for ps in pieces]
    trunc = getattr(tok, "truncation", None)
    if trunc is not None and any(len(e.ids) > int(trunc.get("max_length") or 0) for e in out):
        return _whole(tok, batch, add_special_tokens, is_pretokenized)  # the tokenizer's own cut
    STATS["segmented"] += len(batch)
    STATS["pieces"] += sum(len(ps) for ps in pieces)
    STATS["encoded"] += len(todo)
    return out
