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
import threading
import time
from array import array
from collections import Counter, OrderedDict
from dataclasses import replace

MAX_IDS = int(os.environ.get("ARC_REPLY_IDS_MAX_TOKENS", "4000000"))  # LRU bound: 16 MB of int32 ids
# seeds (29 Oct): a restored attempt's recorded replies, written by its harness (arena.rollout_record) before its first
# live request, so a server that never generated them re-sends them as the source server did (v13: 4 of 9 starts
# harvested in v12 failed the restore digest for want of v12's records). Keyed by the rollout's truncated digest.
SEED_DIR = os.environ.get("ARC_REPLY_IDS_SEED_DIR", "/tmp/arc-reply-ids-seed")
SEED_KEY_BYTES = 16  # rollout_record.ids_digest: 32 hex characters of the same sha256
SEED_POLL_S = 0.1  # the loader thread's directory watch
LOG_EVERY = int(os.environ.get("ARC_REPLY_IDS_LOG_EVERY", "200"))
PER_PROMPT = 16  # replies kept for one prompt (clones of one game share their first prompts)
RECORD_OVERHEAD = 320  # bytes per record besides its ids (key, array and dict entries), for the memory report


def _digest(ids):
    return hashlib.sha256(array("i", ids).tobytes()).digest()


def _canonical(tokenizer, bodies):
    """Per body: does it re-encode to itself? The Rust tokenizer's batch calls (they release the GIL) when there is
    one, else the tokenizer's own decode / encode."""
    if not bodies:
        return []
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        texts = backend.decode_batch(bodies, skip_special_tokens=False)
        return [enc.ids == body for enc, body in zip(backend.encode_batch(texts, add_special_tokens=False), bodies)]
    return [tokenizer.encode(tokenizer.decode(b, skip_special_tokens=False), add_special_tokens=False) == b
            for b in bodies]


class ReplyIds:
    def __init__(self, max_ids=MAX_IDS):
        self.max_ids = max_ids
        self.records = OrderedDict()  # (prompt length, prompt digest) -> [(reply ids without turn end, turn end)]
        self.lengths = Counter()
        self.count = 0
        self.ids = 0
        self.stats = Counter()
        self._special = None
        self.seed_mtime, self.seed_seen = None, set()
        self.seed_done = OrderedDict()  # file digests and record keys already judged (a group's tries share a start)
        self.lock = threading.Lock()  # the store: chat path (substitute, record) vs the seed loader thread
        self.seed_thread, self.seed_tokenizer = None, None

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
        with self.lock:
            return self._record(final_res, tokenizer)

    def _record(self, final_res, tokenizer):
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

    def _add(self, key, generated):
        reply = (array("i", generated[:-1]), generated[-1])
        replies = self.records.get(key)
        if replies is None:
            replies = self.records[key] = []
            self.lengths[key[0]] += 1
        elif reply in replies:
            return False
        elif len(replies) >= PER_PROMPT:
            self._drop(key, 0)
        replies.append(reply)
        self.records.move_to_end(key)
        self.count += 1
        self.ids += len(generated) - 1
        while self.ids > self.max_ids:
            oldest = next(iter(self.records))
            self.stats["evicted"] += len(self.records[oldest])
            self._drop(oldest)
        return True

    def poll_seeds(self, tokenizer, directory=None):
        """Chat path (cheap): starts the loader thread once, with this tokenizer. The thread watches ``directory``
        itself (SEED_POLL_S): each new seed file is judged off the event loop (29 Oct v14: 0.3 s of tokenizer work per
        file on the API loop stalled play), its records added under the lock, then ``<file>.done`` written; the
        harness's first live request of that attempt awaits that mark (rollout_record.wait_seeds)."""
        if self.seed_thread is None:
            self.seed_tokenizer = tokenizer
            self.seed_thread = threading.Thread(target=self._seed_loop, args=(directory or SEED_DIR,),
                                                name="arc-reply-seeds", daemon=True)
            self.seed_thread.start()
        return 0

    def _seed_loop(self, directory):
        while True:
            try:
                self.load_seeds(self.seed_tokenizer, directory)
            except Exception as exc:  # noqa: BLE001 - a seed is an optimisation
                self.stats[f"seed_error_{type(exc).__name__}"] += 1
            time.sleep(SEED_POLL_S)

    def _load_file(self, path, tokenizer):
        """One seed file: replies that end with a turn end and do not re-encode to themselves (a canonical reply is
        re-sent identical and never needs a record), each file and record key judged once (a group's tries share a
        start: identical files), into the store under the lock. Returns the records added."""
        try:
            with open(path, "rb") as file:
                raw = file.read()
        except OSError:
            self.stats["seed_unreadable"] += 1
            return 0
        if self._seed_seen(hashlib.sha256(raw).digest()):
            self.stats["seed_file_repeat"] += 1
            return 0
        try:
            records = json.loads(raw).get("records") or []
        except ValueError:
            self.stats["seed_unreadable"] += 1
            return 0
        if self._special is None:
            self._special = frozenset(tokenizer.all_special_ids)
        todo = []
        for length, digest, generated in records:
            generated = [int(t) for t in generated]
            if len(generated) < 2 or generated[-1] not in self._special:
                continue
            if self._seed_seen((int(length), digest, len(generated))):
                self.stats["seed_key_repeat"] += 1
                continue
            todo.append(((int(length), bytes.fromhex(digest)[:SEED_KEY_BYTES]), generated))
        canonical = _canonical(tokenizer, [g[:-1] for _, g in todo])
        added = 0
        with self.lock:
            for (key, generated), same in zip(todo, canonical):
                if same:
                    self.stats["seed_canonical"] += 1
                else:
                    added += self._add(key, generated)
            self.stats["seeded"] += added
        return added

    def load_seeds(self, tokenizer, directory=None):
        """Every new seed file now, in this thread (the loader thread; tests): judged, added, ``<file>.done`` written.
        Returns the records added."""
        directory = directory or SEED_DIR
        try:
            mtime = os.stat(directory).st_mtime_ns
        except OSError:
            return 0
        if mtime == self.seed_mtime:
            return 0
        self.seed_mtime = mtime
        added, names = 0, set()
        for entry in os.scandir(directory):
            if entry.name.endswith(".json"):
                names.add(entry.name)
                if entry.name not in self.seed_seen:
                    self.seed_seen.add(entry.name)
                    added += self._load_file(entry.path, tokenizer)
                    try:
                        open(entry.path + ".done", "w").close()
                    except OSError:
                        pass
        self.seed_seen &= names
        return added

    def _seed_seen(self, key, bound=200_000):
        """True when ``key`` was judged before (moved to the end); else remembered (the oldest forgotten past bound)."""
        if key in self.seed_done:
            self.seed_done.move_to_end(key)
            return True
        self.seed_done[key] = None
        if len(self.seed_done) > bound:
            self.seed_done.popitem(last=False)
        return False

    # ---------------------------------------------------------------------------------------------- substitute
    @staticmethod
    def _same_text(tokenizer, a, b):
        if tokenizer.decode(a, skip_special_tokens=False) != tokenizer.decode(b, skip_special_tokens=False):
            return False
        return "".join(tokenizer.convert_ids_to_tokens(a)) == "".join(tokenizer.convert_ids_to_tokens(b))

    def substitute(self, engine_input, tokenizer, limit):
        """Edit ``engine_input`` in place; ``limit`` is the longest prompt a substitution may produce."""
        self.poll_seeds(tokenizer)
        with self.lock:
            return self._substitute(engine_input, tokenizer, limit)

    def _substitute(self, engine_input, tokenizer, limit):
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
            if replies is None:  # a seed: the rollout's truncated digest
                key = (length, key[1][:SEED_KEY_BYTES])
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
try:
    os.makedirs(SEED_DIR, exist_ok=True)  # its presence tells a harness in this container to write seeds
except OSError:
    pass


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
