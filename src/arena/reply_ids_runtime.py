"""Reply-id cache (API server side of the prefix cache): a re-sent assistant reply keeps the token ids it was
generated with.

The chat template re-renders each earlier reply as text and the tokenizer picks the canonical split, while the model
sometimes generated another split of the same text ([un][covers] vs [unc][overs]; 2.3% of replies on v7.1 traffic).
The re-sent prompt then leaves the cached sequence inside the reply, the request-end entry no longer matches, and the
game resumes from an older state or from block 1.

This file is copied next to the pinned chat serving module by ``arena.prefix_cache_patch`` (option ``reply_ids``, off
by default). It has no vLLM imports.

- ``record``: at a finished non-streaming generation, key = (length, sha256) of the prompt ids as sent to the engine,
  value = the generated ids. Only replies that end with a special token (the turn end) are recorded.
- ``substitute``: on the next rendered prompt, at a position whose preceding ids equal a recorded prompt, the span up
  to the next turn end is replaced by the recorded ids if and only if both decode to the same string and the same
  token-string concatenation. Identical ids drop the record (the reply was canonical: it can never be needed), and
  so does a text that matches none of the prompt's replies (the client re-sent something else; this bounds the
  per-request work). One prompt can hold several replies (clones of a game share their first prompts).

A span is therefore replaced only where the engine saw exactly this prefix followed by exactly this text: the same
text in a user turn or a tool output sits behind a different prefix and is tokenized as before.
"""
import hashlib
import json
import os
from array import array
from collections import Counter, OrderedDict
from dataclasses import replace

MAX_IDS = int(os.environ.get("ARC_REPLY_IDS_MAX_TOKENS", "4000000"))  # LRU bound: 16 MB of int32 ids
LOG_EVERY = int(os.environ.get("ARC_REPLY_IDS_LOG_EVERY", "200"))
PER_PROMPT = 16  # replies kept for one prompt (clones of one game share their first prompts)
RECORD_OVERHEAD = 320  # bytes per record besides its ids (key, array and dict entries), for the memory report


def _digest(ids):
    return hashlib.sha256(array("i", ids).tobytes()).digest()


class ReplyIds:
    def __init__(self, max_ids=MAX_IDS):
        self.max_ids = max_ids
        self.records = OrderedDict()  # (prompt length, prompt digest) -> [(reply ids without turn end, turn end)]
        self.lengths = Counter()
        self.count = 0
        self.ids = 0
        self.stats = Counter()
        self._special = None

    # ---------------------------------------------------------------------------------------------- store
    def _drop(self, key, index=None):
        """Remove one reply of a prompt, or the whole prompt."""
        replies = self.records[key]
        gone = replies if index is None else [replies[index]]
        self.ids -= sum(len(body) for body, _ in gone)
        self.count -= len(gone)
        if index is not None and len(replies) > 1:
            del replies[index]
            return
        del self.records[key]
        self.lengths[key[0]] -= 1
        if not self.lengths[key[0]]:
            del self.lengths[key[0]]

    def memory_bytes(self):
        return 4 * self.ids + RECORD_OVERHEAD * self.count

    def record(self, final_res, tokenizer):
        prompt = getattr(final_res, "prompt_token_ids", None)
        outputs = getattr(final_res, "outputs", None) or []
        if not prompt or len(outputs) != 1:
            self.stats["record_skip_shape"] += 1
            return
        generated = list(outputs[0].token_ids or ())
        if self._special is None:
            self._special = frozenset(tokenizer.all_special_ids)
        if len(generated) < 2 or generated[-1] not in self._special or len(generated) > self.max_ids:
            self.stats["record_skip_no_turn_end"] += 1
            return
        key = (len(prompt), _digest(prompt))
        reply = (array("i", generated[:-1]), generated[-1])
        replies = self.records.get(key)
        if replies is None:
            replies = self.records[key] = []
            self.lengths[key[0]] += 1
        elif reply in replies:
            return
        elif len(replies) >= PER_PROMPT:
            self._drop(key, 0)
        replies.append(reply)
        self.records.move_to_end(key)
        self.count += 1
        self.ids += len(generated) - 1
        self.stats["recorded"] += 1
        while self.ids > self.max_ids:
            oldest = next(iter(self.records))
            self.stats["evicted"] += len(self.records[oldest])
            self._drop(oldest)

    # ---------------------------------------------------------------------------------------------- substitute
    @staticmethod
    def _same_text(tokenizer, a, b):
        if tokenizer.decode(a, skip_special_tokens=False) != tokenizer.decode(b, skip_special_tokens=False):
            return False
        return "".join(tokenizer.convert_ids_to_tokens(a)) == "".join(tokenizer.convert_ids_to_tokens(b))

    def substitute(self, engine_input, tokenizer, limit):
        """Edit ``engine_input`` in place; ``limit`` is the longest prompt a substitution may produce."""
        self.stats["prompts"] += 1
        ids = engine_input.get("prompt_token_ids")
        if not self.records or not isinstance(ids, list) or engine_input.get("type") not in ("token", "multimodal"):
            return 0
        if engine_input.get("prompt_token_offsets") or engine_input.get("assistant_tokens_mask"):
            self.stats["skip_positional_fields"] += 1
            return 0
        n = len(ids)
        raw = None
        placeholders = [(m, i, p) for m, ranges in (engine_input.get("mm_placeholders") or {}).items()
                        for i, p in enumerate(ranges)]
        hasher, hashed, delta = hashlib.sha256(), 0, 0
        pieces, cursor, shifts = [], 0, []  # shifts: (input position, cumulative length change from there on)
        for length in sorted(self.lengths):
            start = length - delta  # input position of output position `length`
            if start >= n:
                break
            if start < hashed:
                continue
            if raw is None:
                raw = array("i", ids)
            hasher.update(raw[hashed:start].tobytes())
            hashed = start
            key = (length, hasher.copy().digest())
            replies = self.records.get(key)
            if replies is None:
                continue
            same = next((i for i, (body, turn_end) in enumerate(replies)
                         if start + len(body) < n and ids[start + len(body)] == turn_end
                         and raw[start:start + len(body)] == body), None)
            if same is not None:
                self.stats["resent_identical"] += 1
                self._drop(key, same)
                continue
            body = stop = None
            for candidate, turn_end in replies:
                try:
                    end = ids.index(turn_end, start)
                except ValueError:
                    continue
                if (not any(p.offset < end and p.offset + p.length > start for _, _, p in placeholders)
                        and self._same_text(tokenizer, ids[start:end], list(candidate))):
                    body, stop = candidate, end
                    break
            if body is None:
                self.stats["resent_text_differs"] += 1
                self._drop(key)
                continue
            size = len(body)
            if n + delta + size - (stop - start) > max(limit, n):
                self.stats["skip_too_long"] += 1
                continue
            pieces.append(ids[cursor:start])
            pieces.append(list(body))
            cursor = stop
            hasher.update(body.tobytes())
            hashed = stop
            delta += size - (stop - start)
            shifts.append((stop, delta))
            self.records.move_to_end(key)
            self.stats["replaced_spans"] += 1
            self.stats["replaced_tokens"] += size
        if not shifts:
            return 0
        pieces.append(ids[cursor:])
        moved = {}
        for modality, _, p in placeholders:
            change = next((d for at, d in reversed(shifts) if at <= p.offset), 0)
            moved.setdefault(modality, []).append(replace(p, offset=p.offset + change) if change else p)
        engine_input["prompt_token_ids"] = [t for piece in pieces for t in piece]
        if moved:
            engine_input["mm_placeholders"] = {**engine_input["mm_placeholders"], **moved}
        self.stats["replaced_prompts"] += 1
        return len(shifts)

    def log(self):
        print("ARC_REPLY_IDS_STATS " + json.dumps({**self.stats, "records": self.count, "ids": self.ids,
                                                   "memory_bytes": self.memory_bytes(), "max_ids": self.max_ids},
                                                  sort_keys=True), flush=True)


_CACHE = ReplyIds()


def substitute(engine_inputs, tokenizer, max_model_len, max_tokens=None):
    """Chat serving hook, after rendering: never raises (a failure leaves the rendered prompt as it is)."""
    try:
        for engine_input in engine_inputs:
            _CACHE.substitute(engine_input, tokenizer, max_model_len - max(1, int(max_tokens or 1)))
        if LOG_EVERY and _CACHE.stats["prompts"] % LOG_EVERY == 0:
            _CACHE.log()
    except Exception as exc:  # the cache is an optimisation: the request goes on with what it has
        _CACHE.stats[f"substitute_error_{type(exc).__name__}"] += 1


def record(final_res, tokenizer):
    """Chat serving hook, at the end of a non-streaming generation: never raises."""
    try:
        _CACHE.record(final_res, tokenizer)
    except Exception as exc:
        _CACHE.stats[f"record_error_{type(exc).__name__}"] += 1
