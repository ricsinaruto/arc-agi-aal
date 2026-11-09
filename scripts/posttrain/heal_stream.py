"""Streaming teacher (PTH streaming OPD, 22 Oct): scores the student's games while they are played.

The student server writes one ``rollout.jsonl`` per session (``arena.rollout_record``: the served request as a delta,
the reply, the sampled ids with the engine's log-probability of each, the weights version that sampled it); a
puller mirrors them under ``root``. ``StreamTeacher`` tails every file, follows each session's drain chain (a chain
grows while every request's ids extend the previous request's ids token for token; a drain starts a new one) and
scores only the tokens its teacher cache has not seen, in ``chunk``-token pieces, keeping each chain's cache between
pieces. Each piece is one ``s-*.pt`` file in heal_kd's streamed layout (``row0``: the first teacher row; the ids
before it are context the trainer recomputes without gradient) and a ``.ready`` marker with its weights version,
which ``heal_kd.StreamQueue`` reads.

Rows: row t predicts token t+1. A piece's last row has no next token yet; it is kept (its full log-distribution on
the host) and becomes the next piece's ``row0``. ``beh_logp`` row t is the engine's log p of token t+1 (NaN off the
student's replies or past the first token whose re-encoding differs from the sampled ids).

The model-specific part is ``forward(chain, lo, hi, keep)``: a generator over sub-pieces of [lo, hi) that runs the
teacher on them against ``chain.cache`` (updating it) and yields ``(start, stop, logits)`` (``keep`` False: the cache
alone, logits None)."""
import json
import os
import shutil
import queue
import statistics
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import torch
except ImportError:  # the launcher's interpreter reads stream_bars only
    torch = None

ROLLOUT_KEYS = ("tokens", "logprobs", "top", "support", "reference", "sampling", "prompt_digest", "finish_reason", "step",
                "w", "version")


SHRINK_WAIT = 300.0  # seconds a shrunk copy may stay below the read offset before it counts as a new session
BASE_BEHIND_MULT = 8  # the base slice a scan while the base share lags
BASE_PER_SCAN = 16  # base captures encoded per scan: a recorded base backlog (thousands of requests) read in one
# scan held every later student capture behind its encodes (g1k 23 Oct: 2,532 base captures, 0 student pieces)
# student captures encoded per scan, round-robin over sessions (c7b 24 Oct: four recorded student plays read in one
# scan put every capture's full prompt + images in host RAM at once; the container thrashed at 1,024 GiB)
STUDENT_PER_SCAN = int(os.environ.get("HEAL_STREAM_STUDENT_PER_SCAN", "32"))


class Rewritten(RuntimeError):
    """The session file no longer extends what was read: a new session in that file (its chain closes)."""


class SessionTail:
    """New capture rows of one growing ``rollout.jsonl`` (``rl_rollouts.captures``' rows plus ``version``): complete
    lines only; the file may be replaced by a longer copy (the puller), whose bytes so far must be the same."""

    def __init__(self, path):
        self.path, self.offset, self.messages, self.tools, self.rows = Path(path), 0, [], None, 0
        self.mark = b""  # the bytes just before ``offset``: a rewritten file is refused
        self.size, self.grew, self.idle_sent = 0, time.time(), False
        self.shrunk_since = None
        self.backlog = deque()  # base rows read but not yet encoded (StreamTeacher.base_per_scan)

    def read(self):
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return []
        # a shorter copy (opdS-g1h, 22 Oct: the puller briefly held a 34,540-byte copy of a 516,888-byte session and the
        # teacher died): wait for it to grow back; past SHRINK_WAIT, or with different bytes, it is a new session
        if size < self.offset:
            self.shrunk_since = self.shrunk_since or time.time()
            if time.time() - self.shrunk_since < SHRINK_WAIT:
                return []
            raise Rewritten(f"{self.path}: shrank from {self.offset} to {size} bytes")
        self.shrunk_since = None
        if size == self.size:
            return []
        with open(self.path, "rb") as f:
            if self.offset:
                f.seek(self.offset - len(self.mark))
                if f.read(len(self.mark)) != self.mark:
                    raise Rewritten(f"{self.path}: rewritten before byte {self.offset}")
            data = f.read()
        self.size, self.grew, self.idle_sent = size, time.time(), False
        end = data.rfind(b"\n") + 1
        if not end:
            return []
        out = []
        for line in data[:end].splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            self.messages = self.messages[: row["keep"]] + row["messages"]
            self.tools = row["tools"] if "tools" in row else self.tools
            self.rows += 1
            if not row.get("gameplay", True):
                continue
            payload = {"messages": list(self.messages), "chat_template_kwargs": row["template"]}
            if self.tools is not None:
                payload["tools"] = self.tools
            out.append({"request": row["n"], "prompt_tokens": (row["usage"] or {}).get("prompt_tokens"),
                        "payload": payload, "reply": row["reply"], "usage": row["usage"],
                        "rollout": {k: row.get(k) for k in ROLLOUT_KEYS}})
        whole = data[:end]
        self.mark = whole[-64:] if len(whole) >= 64 else (self.mark + whole)[-64:]
        self.offset += end
        return [c for c in out if c["prompt_tokens"] and (c["usage"] or {}).get("completion_tokens")]


def engine_logprobs(rollout, reply_ids):
    """Per reply token the engine's log p (NaN past the first token whose sampled id differs from the re-encoded
    reply; ``rl_job.span_arrays``' alignment rule) and whether the whole reply aligned."""
    tokens, values = rollout.get("tokens") or [], rollout.get("logprobs") or []
    count = len(reply_ids)
    shared = next((i for i, (a, b) in enumerate(zip(tokens, reply_ids)) if a != b), min(count, len(tokens)))
    out = [float("nan")] * count
    for i in range(min(shared, len(values))):
        out[i] = float(values[i])
    return out, shared > 0 and shared == min(count, len(tokens)) and abs(count - len(tokens)) <= 1


class Chain:
    """One drain chain of one session: the latest request's encoded inputs (host), per-token reply mask, engine
    log p and weights version; ``fed``: tokens in the teacher cache; ``last``: the pending last row."""

    def __init__(self, key, number, inputs):
        self.key, self.number = key, number
        self.inputs = inputs
        n = inputs["input_ids"].shape[1]
        self.reply = torch.zeros(n, dtype=torch.bool)
        self.beh = torch.full((n,), float("nan"))
        self.version = torch.full((n,), -1, dtype=torch.long)
        self.fed, self.cache, self.last, self.pieces, self.closed = 0, None, None, 0, False
        self.canon, self.start = None, 0  # the last request's (canonical inputs, start) and its exact prompt length
        self.files = []  # (name, save future) per written piece
        self.used, self.since = time.time(), time.time()  # last forward; first unscored token's arrival

    @property
    def ids(self):
        return self.inputs["input_ids"][0]

    def extend(self, inputs):
        n_old, n = self.ids.shape[0], inputs["input_ids"].shape[1]
        pad = n - n_old
        self.inputs = inputs
        self.reply = torch.cat([self.reply, torch.zeros(pad, dtype=torch.bool)])
        self.beh = torch.cat([self.beh, torch.full((pad,), float("nan"))])
        self.version = torch.cat([self.version, torch.full((pad,), -1, dtype=torch.long)])

    def mark_reply(self, start, stop, beh, version):
        self.reply[start:stop] = True
        self.beh[start:start + len(beh)] = torch.tensor(beh, dtype=torch.float32)
        self.version[start:stop] = -1 if version is None else int(version)

    def pending(self):
        return self.ids.shape[0] - self.fed


def exact_inputs(inputs, start, sampled, end_id, prev=None):
    """The request as the play's server held it (23 Oct: the pruned student samples non-canonical token splits,
    e.g. "{"+"k" for "{k"; its server keeps those sampled ids in later prompts, so served counts ran +1..+24 over a
    re-encoding, exactly the running sum of each reply's extra tokens). ``inputs``/``start``: the canonical encoding
    of prompt + reply (prompt = ``start`` tokens); ``sampled``: the reply's sampled ids; ``prev``: (canonical ids,
    exact inputs) of the session's previous request. Returns (exact inputs, start, canonical ids): the previous
    request's exact ids, the new canonical tail (tool results, user turn, generation prompt) and the sampled reply
    ending in ``end_id``; None when the canonical prompt does not extend the previous canonical request (the history
    was rewritten, e.g. a compaction drain). Returns (exact inputs, start). ``prev``'s second item: the exact ids that
    precede the new tail (the previous request's prompt + sampled reply; or, for a reply cut at the token cap, which
    the server re-renders and closes, its prompt + the canonical reply)."""
    ids = inputs["input_ids"][0]
    keys = [k for k in ("input_ids", "mm_token_type_ids") if k in inputs]
    reply = torch.tensor(list(sampled) + ([end_id] if not sampled or sampled[-1] != end_id else []), dtype=ids.dtype)
    if prev is None:
        cut, head = 0, {k: inputs[k][:, :0] for k in keys}
    else:
        canon, ex = prev
        cut = canon.shape[0]
        if start < cut or not torch.equal(ids[:cut], canon):
            return None
        head = {k: ex[k] for k in keys}
    out = dict(inputs)
    for k in keys:
        tail = inputs[k][:, cut:start]
        new = reply.view(1, -1) if k == "input_ids" else torch.zeros((1, reply.shape[0]), dtype=inputs[k].dtype)
        out[k] = torch.cat([head[k], tail, new], dim=1)
    if "attention_mask" in inputs:
        out["attention_mask"] = torch.ones_like(out["input_ids"])
    return out, head["input_ids"].shape[1] + (start - cut)


def image_runs(ids, image_token_id):
    """[(start, stop)] of each image's token run, in order."""
    is_img = (ids == image_token_id).to(torch.int8)
    if not bool(is_img.any()):
        return []
    edges = torch.diff(torch.nn.functional.pad(is_img, (1, 1)))
    starts, stops = (edges == 1).nonzero().flatten().tolist(), (edges == -1).nonzero().flatten().tolist()
    return list(zip(starts, stops))


def cut_point(ids, upto, lo, image_token_id):
    """``upto`` moved so that no image straddles it (back to the image start; forward to its end when that would
    leave nothing after ``lo``)."""
    if upto >= ids.shape[0] or int(ids[upto]) != image_token_id or upto == 0 or int(ids[upto - 1]) != image_token_id:
        return upto
    for a, b in image_runs(ids, image_token_id):
        if a < upto < b:
            return a if a > lo else b
    return upto


def prefix_inputs(inputs, upto, image_token_id):
    """The inputs of the first ``upto`` tokens (whole images only: ``cut_point`` keeps images whole)."""
    out = {"input_ids": inputs["input_ids"][:, :upto]}
    if "mm_token_type_ids" in inputs:
        out["mm_token_type_ids"] = inputs["mm_token_type_ids"][:, :upto]
    if inputs.get("pixel_values") is None:
        return out
    runs = image_runs(inputs["input_ids"][0], image_token_id)
    grid = inputs["image_grid_thw"]
    if len(runs) != grid.shape[0]:
        raise RuntimeError(f"{len(runs)} image runs vs {grid.shape[0]} images")
    keep = sum(1 for _, b in runs if b <= upto)
    if keep and runs[keep - 1][1] > upto:
        raise RuntimeError("an image straddles the cut")
    rows = int(grid[:keep].prod(-1).sum())
    if keep:
        out["pixel_values"] = inputs["pixel_values"][:rows]
        out["image_grid_thw"] = grid[:keep]
    return out


class StreamTeacher:
    """Tails ``root``/**/rollout.jsonl, scores chains, writes pieces to ``out``. ``encode(capture)``: (inputs, reply
    start) on the host, or None. ``forward``: the module docstring. ``image_token_id``: kept whole across cuts.
    ``budget(chains)``: called after each piece; evicts teacher caches (a chain whose cache was evicted recomputes it
    from its ids)."""

    def __init__(self, root, out, encode, forward, *, k=128, chunk=32768, image_token_id=-1, min_reply=1,
                 idle=900.0, poll=2.0, encoders=4, log=None, budget=None, top_fn=None, stop_file=None, base=(),
                 end_id=None, base_per_scan=BASE_PER_SCAN, base_share=0.0, batch=1, student_per_scan=None):
        self.root, self.out = Path(root), Path(out)
        self.base = set(base)  # root subfolders (play runs) of base-model play: pieces "b-*", marker "base"
        self.base_per_scan = int(base_per_scan)
        self.student_per_scan = int(STUDENT_PER_SCAN if student_per_scan is None else student_per_scan)
        # the trainer's base slot share (HEAL_STREAM_BASE_SHARE): base pieces in that proportion once student pieces
        # exist (g1l, 23 Oct: student-first alone scored 212 student pieces and 0 base behind the play's backlog)
        self.base_share = float(base_share)
        self.end_id, self.exact = end_id, end_id is not None  # exact_inputs on requests that carry sampled ids
        self.out.mkdir(parents=True, exist_ok=True)
        self.encode, self.forward, self.k, self.chunk = encode, forward, k, chunk
        self.image_token_id, self.min_reply, self.idle, self.poll = image_token_id, min_reply, idle, poll
        self.log, self.budget, self.top_fn = log or (lambda row: None), budget, top_fn
        self.stop_file = Path(stop_file) if stop_file else None
        self.tails, self.chains, self.numbers = {}, {}, {}
        self.events = queue.Queue(maxsize=256)
        self.pool = ThreadPoolExecutor(max_workers=encoders)
        self.saver = ThreadPoolExecutor(max_workers=2)
        self.stats = {"pieces": 0, "tokens": 0, "reply_rows": 0, "forward_seconds": 0.0, "recompute_tokens": 0,
                      "recompute_seconds": 0.0, "breaks": 0, "unaligned": 0, "unversioned": 0, "evictions": 0,
                      "encode_failures": 0, "encode_seconds": 0.0, "captures": 0, "closed": 0}
        self.error, self.closing = None, []
        self.stopping = threading.Event()
        self.epoch = ""  # piece-name tag of a restarted teacher (skip_existing): names never repeat a consumed piece's
        self._next = None  # (chain, upto) whose first slice a pipelined forward (``forward.prefetch``) already started
        # pieces of up to ``batch`` distinct chains per forward (``forward.many``: one-card teacher, MFU-2 23 Oct)
        self.batch = int(batch) if getattr(forward, "many", None) is not None else 1

    # --- host side: tail the files, encode the new requests (a thread; no CUDA calls) ------------------------

    def _scan(self):
        for path in sorted(self.root.glob("**/rollout.jsonl")):
            key = path.relative_to(self.root).parent.as_posix()
            if key not in self.tails:
                self.tails[key] = SessionTail(path)
        found, room = [], self.base_per_scan
        if 0 < self.base_share < 1:  # behind the trainer's base share: a bigger base slice this scan (c4/c6, 23 Oct:
            # 16 captures a scan behind a live play's backlog gave 0 base pieces in the first 10-30 min)
            base = self.stats.get("base_pieces", 0)
            if base < self.base_share / (1 - self.base_share) * (self.stats.get("pieces", 0) - base) + 1:
                room *= BASE_BEHIND_MULT
        # student sessions first; base sessions after, at most ``base_per_scan`` captures a scan (the rest wait in order)
        for key, tail in sorted(self.tails.items(), key=lambda kv: (self._key_is_base(kv[0]), kv[0])):
            try:
                rows = tail.read()
            except Rewritten as exc:  # its chain closes; the file is read again from its start as a new session
                self.stats["rewrites"] = self.stats.get("rewrites", 0) + 1
                self.log({"event": "stream_rewritten", "session": key, "error": str(exc)[:300]})
                self.events.put(("idle", key, None, None))
                tail = self.tails[key] = SessionTail(tail.path)
                rows = tail.read()
            if self._key_is_base(key):
                tail.backlog.extend(rows)
                rows = [tail.backlog.popleft() for _ in range(min(room, len(tail.backlog)))]
                room -= len(rows)
            else:  # taken round-robin below, at most ``student_per_scan`` a scan
                tail.backlog.extend(rows)
                rows = []
            if tail.shrunk_since and not getattr(tail, "shrink_noted", False):  # counted once per shrink
                tail.shrink_noted = True
                self.stats["shrinks"] = self.stats.get("shrinks", 0) + 1
                self.log({"event": "stream_shrunk", "session": key, "offset": tail.offset})
            elif not tail.shrunk_since:
                tail.shrink_noted = False
            for capture in rows:
                found.append((key, capture, self._submit(capture)))
        # student rows: one per session in turn (order kept within a session) up to the cap; ahead of this scan's base
        student = [(key, tail) for key, tail in sorted(self.tails.items()) if not self._key_is_base(key) and tail.backlog]
        picked, left = [], self.student_per_scan
        while left > 0 and student:
            for key, tail in list(student):
                if left <= 0:
                    break
                picked.append((key, tail.backlog.popleft()))
                left -= 1
                if not tail.backlog:
                    student.remove((key, tail))
        found = [(key, capture, self._submit(capture)) for key, capture in picked] + found
        for key, capture, future in found:  # in order per session
            self.events.put(("capture", key, capture, future.result()))
        now = time.time()
        for key, tail in self.tails.items():
            if now - tail.grew > self.idle and not tail.idle_sent and not tail.backlog:
                tail.idle_sent = True
                self.events.put(("idle", key, None, None))

    def _submit(self, capture):
        submit = getattr(self.encode, "submit", None)  # a ProcessEncoder: its workers time the encode
        return submit(capture) if submit is not None else self.pool.submit(self._encode, capture)

    def _encode(self, capture):
        began = time.time()
        try:
            made = self.encode(capture)
        except Exception as exc:  # noqa: BLE001 - counted; the session's chain closes at it
            made = exc
        return made, time.time() - began

    def producer(self):
        try:
            while not self.stopping.is_set():
                self._scan()
                self.stopping.wait(self.poll)
        except BaseException as exc:  # noqa: BLE001 - raised by the scoring loop
            self.error = exc
            self.events.put(("error", None, None, None))

    # --- scoring side ----------------------------------------------------------------------------------------

    def _close(self, key):
        chain = self.chains.get(key)
        if chain is not None and not chain.closed:
            chain.closed = True
            self.stats["closed"] += 1

    def _take(self, key, capture, made):
        made, seconds = made
        self.stats["captures"] += 1
        self.stats["encode_seconds"] += seconds
        if made is None or isinstance(made, Exception):
            self.stats["encode_failures"] += 1
            self.log({"event": "stream_encode_failed", "session": key, "request": capture["request"],
                      "error": None if made is None else f"{type(made).__name__}: {made}"[:300]})
            self._close(key)
            return
        inputs, start = made
        chain = self.chains.get(key)
        if self.exact and capture["rollout"].get("tokens"):  # student play: the server's own ids (exact_inputs)
            served, sampled = capture.get("prompt_tokens"), capture["rollout"]["tokens"]
            keys = [k for k in ("input_ids", "mm_token_type_ids") if k in inputs]
            canon = ({k: inputs[k] for k in keys}, start)  # this request's canonical encoding (the next one's check)
            got = None
            if chain is not None and not chain.closed and chain.canon is not None:
                (cin, cstart), cids = chain.canon, chain.canon[0]["input_ids"][0]
                heads = [("extend", chain.inputs),  # the server kept the sampled reply
                         ("rewrite", {k: torch.cat([chain.inputs[k][:, :chain.start], cin[k][:, cstart:]], 1)
                                      for k in keys})]  # a capped reply: the server re-rendered it (canonical)
                for mode, head in heads:
                    r = exact_inputs(inputs, start, sampled, self.end_id, (cids, head))
                    if r is not None and r[1] == served:
                        got = (mode, r)
                        break
                if got is None:  # the server re-rendered the previous reply from the client's message, whose text
                    # differs from the recorded reply (g1l k009 #279: a loop-stopped reply's reasoning came back with
                    # "\n\n" as "\n"; no reply-id record, or its text no longer matched): exact prompt + canonical rest
                    r = exact_inputs(inputs, start, sampled, self.end_id,
                                     (cids[:cstart], {k: chain.inputs[k][:, :chain.start] for k in keys}))
                    if r is not None and r[1] == served:
                        got = ("rerender", r)
                        self.stats["rerenders"] = self.stats.get("rerenders", 0) + 1
            if got is not None and got[0] == "extend":
                inputs, start = got[1]
                chain.extend(inputs)
            else:
                if got is None:  # a new chain: the canonical prompt must be what the server held
                    r = exact_inputs(inputs, start, sampled, self.end_id)
                    if r[1] != served:
                        self.stats["inexact"] = self.stats.get("inexact", 0) + 1
                        self.log({"event": "stream_inexact", "session": key, "request": capture["request"],
                                  "served": served, "canonical": start})
                        self._close(key)
                        return
                    got = ("fresh", r)
                if chain is not None and not chain.closed:
                    self.stats["breaks"] += 1
                    self._close(key)
                chain, (inputs, start) = None, got[1]
        else:
            canon = None
            ids = inputs["input_ids"][0]
            served = capture.get("prompt_tokens")
            if self.exact and served is not None and abs(start - served) > 8:  # the encoder's count rule (base play)
                self.stats["inexact"] = self.stats.get("inexact", 0) + 1
                self._close(key)
                return
            if chain is not None and not chain.closed:
                old = chain.ids
                if ids.shape[0] >= old.shape[0] and torch.equal(ids[: old.shape[0]], old):
                    chain.extend(inputs)
                else:
                    self.stats["breaks"] += 1
                    self._close(key)
                    chain = None
            else:
                chain = None
        ids = inputs["input_ids"][0]
        if chain is None:
            if key in self.chains and self.chains[key] not in self.closing:  # its remainder is scored first (older)
                self.closing.append(self.chains[key])
            self.numbers[key] = self.numbers.get(key, -1) + 1
            chain = self.chains[key] = Chain(key, self.numbers[key], inputs)
        chain.canon, chain.start = canon, start
        reply_ids = ids[start:].tolist()
        beh, aligned = engine_logprobs(capture["rollout"], reply_ids)
        self.stats["unaligned"] += int(not aligned)
        version = capture["rollout"].get("version")
        if version is None:
            self.stats["unversioned"] += len(reply_ids)
        chain.mark_reply(start, ids.shape[0], beh, version)

    def _worth(self, chain):
        return bool(chain.reply[chain.fed:].any()) if chain.pending() > 0 else False

    def _is_base(self, chain):
        return self._key_is_base(chain.key)

    def _key_is_base(self, key):
        return key.split("/")[0] in self.base

    def _due(self, pure=False, exclude=()):
        """The next (chain, upto) (``pure``: no bookkeeping). Student play first (its pieces age; base pieces do not, and a finished or recorded
        base backlog of ~12M tokens scored first starved the trainer's student slots, g1i 23 Oct): a closed student
        chain's remainder, then the open student chain whose oldest unscored token waited longest (a whole chunk
        pending); base chains in the same order only when no student piece is due."""
        order = (False, True)
        if 0 < self.base_share < 1:
            base = self.stats.get("base_pieces", 0)
            if base < self.base_share / (1 - self.base_share) * (self.stats["pieces"] - base):
                order = (True, False)  # behind the trainer's base share: a due base piece first
        for want_base in order:
            for chain in [c for c in self.closing if self._is_base(c) == want_base and c not in exclude]:  # a copy
                if self._worth(chain):
                    upto = min(chain.ids.shape[0], chain.fed + self.chunk)
                    return chain, cut_point(chain.ids, upto, chain.fed, self.image_token_id)
                if pure:
                    continue
                self.closing.remove(chain)
                chain.cache = chain.last = None
            ready = [c for c in self.chains.values()
                     if not c.closed and c.pending() >= self.chunk and self._is_base(c) == want_base and c not in exclude]
            if ready:
                chain = min(ready, key=lambda c: c.since)
                return chain, cut_point(chain.ids, chain.fed + self.chunk, chain.fed, self.image_token_id)
        return None, None

    def _due_many(self, n):
        """Up to ``n`` pieces of distinct chains, each the one ``_due`` picks after the earlier ones are counted."""
        first = self._due()
        if first[0] is None or n <= 1:
            return [first] if first[0] is not None else []
        picks = [first]
        saved = self.stats["pieces"], self.stats.get("base_pieces", 0)
        try:
            while len(picks) < n:
                for c, _ in picks[-1:]:
                    self.stats["pieces"] += 1
                    self.stats["base_pieces"] = self.stats.get("base_pieces", 0) + int(self._is_base(c))
                nxt = self._due(pure=True, exclude={c for c, _ in picks})
                if nxt[0] is None:
                    break
                picks.append(nxt)
        finally:
            self.stats["pieces"], self.stats["base_pieces"] = saved
        return picks

    def _pick(self):
        """The piece to score now: the one a pipelined forward already started, else ``_due``'s."""
        if self._next is not None:
            (chain, upto), self._next = self._next, None
            return chain, upto
        return self._due()

    def _evict(self, chain):
        """The ``budget`` hook after ``chain``'s piece; never a prefetched chain (its stage-A cache is ahead)."""
        if self.budget is not None:
            ahead = getattr(self.forward, "pending_chain", None)
            self.stats["evictions"] += self.budget([c for c in self.chains.values() if c is not chain and c is not ahead])

    def _peek(self, chain, upto):
        """The piece ``_due`` picks once ``chain`` is scored to ``upto`` (a pipelined forward starts its first slice
        early; it is then the next piece scored, whatever arrives meanwhile). None when it needs a cache rebuild."""
        saved = chain.fed, chain.since, self.stats["pieces"], self.stats.get("base_pieces", 0)
        chain.fed, chain.since = upto, time.time()
        self.stats["pieces"] += 1
        self.stats["base_pieces"] = saved[3] + int(self._is_base(chain))
        try:
            nxt, nupto = self._due(pure=True)
        finally:
            chain.fed, chain.since, self.stats["pieces"], self.stats["base_pieces"] = saved
        if nxt is None:
            return None
        lo = upto if nxt is chain else nxt.fed
        if nxt is not chain and nxt.cache is None and lo > 0:
            return None
        return nxt, lo, nupto

    def _top(self, logits, next_ids):
        if self.top_fn is not None:
            return self.top_fn(logits, next_ids)
        full = torch.log_softmax(logits.float(), dim=-1)
        top = full.topk(self.k, dim=-1)
        return {"ids": top.indices.cpu(), "logp": top.values.cpu(), "at": full.gather(1, next_ids.to(full.device)).cpu()}

    def _rebuild(self, chains):
        """Evicted chains (cache gone, ``fed`` > 0): rebuild their caches over the scored tokens (batched when the
        forward takes several chains)."""
        todo = [c for c in chains if c.cache is None and c.fed > 0]
        if not todo:
            return
        began = time.time()
        many = getattr(self.forward, "many", None)
        if many is not None and len(todo) > 1:
            for _ in many([(c, 0, c.fed, False) for c in todo]):
                pass
        else:
            for c in todo:
                for _ in self.forward(c, 0, c.fed, False):
                    pass
        self.stats["recompute_tokens"] += sum(c.fed for c in todo)
        self.stats["recompute_seconds"] += time.time() - began

    def _open(self, chain, upto):
        return {"chain": chain, "lo": chain.fed, "upto": upto, "top_ids": [], "top_logp": [], "tok": [],
                "last_full": None}

    def _feed(self, st, a, b, logits):
        """One forward slice [a, b) of a piece: its top-k rows (and the piece's last full row)."""
        ids = st["chain"].ids
        nxt = torch.cat([ids[a + 1:b], ids[b:b + 1] if b < ids.shape[0] else torch.zeros(1, dtype=ids.dtype)])
        got = self._top(logits, nxt[:, None])
        st["top_ids"].append(got["ids"])  # may stay on the device: one host copy per piece (_finish)
        st["top_logp"].append(got["logp"])
        st["tok"].append(got["at"][:, 0])
        if b == st["upto"]:
            st["last_full"] = torch.log_softmax(logits[-1].float(), dim=-1).cpu()

    def score(self, chain, upto):
        """Score ``chain`` from ``fed`` to ``upto``; write the piece."""
        self._rebuild([chain])
        began = time.time()
        st = self._open(chain, upto)
        prefetch = getattr(self.forward, "prefetch", None)
        for a, b, logits in self.forward(chain, st["lo"], upto, True):
            if prefetch is not None:  # the current piece's slices are queued: the next piece's first slice follows
                prefetch, nxt = None, self._peek(chain, upto)
                if nxt is not None:
                    self.forward.prefetch(*nxt)
                    self._next = (nxt[0], nxt[2])
            self._feed(st, a, b, logits)
            del logits
        seconds = time.time() - began
        self._account(seconds, upto - st["lo"])
        return self._finish(st, seconds)

    def score_many(self, picks):
        """Score several chains' pieces together (distinct chains; ``forward.many`` runs their slices in one batch
        per step). Returns the piece rows."""
        self._rebuild([c for c, _ in picks])
        began = time.time()
        states = [self._open(c, u) for c, u in picks]

        def post(i, a, b, logits):  # in the chain's own thread, right after its slice
            self._feed(states[i], a, b, logits)

        for _ in self.forward.many([(st["chain"], st["lo"], st["upto"], True) for st in states], post=post):
            pass
        seconds = time.time() - began
        self._account(seconds, sum(st["upto"] - st["lo"] for st in states))
        self.stats["batches"] = self.stats.get("batches", 0) + 1
        return [self._finish(st, seconds, batch=len(states)) for st in states]

    def _account(self, seconds, tokens):
        self.stats["forward_seconds"] += seconds
        if "warm_seconds" not in self.stats:  # the first forward carries the kernels' compile/autotune: the rate
            self.stats["warm_seconds"], self.stats["warm_tokens"] = seconds, tokens  # leaves it out
            self._warm_end = time.time()

    def _finish(self, st, seconds, batch=1):
        chain, lo, upto, last_full = st["chain"], st["lo"], st["upto"], st["last_full"]
        ids = chain.ids
        top_ids = torch.cat(st["top_ids"]).int().cpu()
        top_logp, tok = torch.cat(st["top_logp"]).float().cpu(), torch.cat(st["tok"]).float().cpu()
        tok[-1] = float("nan")  # the last row's next token is not known yet
        row0 = lo
        if chain.last is not None:  # the previous piece's last row, now with its next token
            row0 = lo - 1
            prev = chain.last
            top_ids = torch.cat([prev["ids"][None], top_ids])
            top_logp = torch.cat([prev["logp"][None], top_logp])
            tok = torch.cat([prev["full"][int(ids[lo])].view(1), tok])
        chain.last = {"ids": top_ids[-1].clone(), "logp": top_logp[-1].clone(), "full": last_full}
        chain.fed, chain.used = upto, time.time()
        chain.since = time.time()
        mass = top_logp.double().exp().sum(-1).clamp(max=1 - 1e-6)
        beh = torch.cat([chain.beh[row0 + 1:upto], torch.full((1,), float("nan"))])
        predicted = chain.reply[row0 + 1:upto]
        versions = chain.version[row0 + 1:upto][predicted]
        known = versions[versions >= 0].tolist()
        version = int(statistics.median_low(known)) if known else None
        inputs = prefix_inputs(chain.inputs, upto, self.image_token_id)
        payload = {"ids": inputs["input_ids"][0].int().clone(), "image": (inputs["input_ids"][0] == self.image_token_id),
                   "reply": chain.reply[:upto].clone(), "row0": row0, "top_ids": top_ids, "top_logp": top_logp.half(),
                   "tail": torch.log1p(-mass).half(), "tok_logp": tok, "beh_logp": beh.float(),
                   "version": version, "version_min": min(known) if known else None, "w": 1.0,
                   "segment": chain.key, "chain": chain.number, "piece": chain.pieces,
                   **{k: inputs[k] for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids") if k in inputs}}
        is_base = chain.key.split("/")[0] in self.base
        name = ("b-" if is_base else "s-") + "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in chain.key)
        name = f"{name}--{self.epoch}c{chain.number:03d}-p{chain.pieces:04d}"
        chain.pieces += 1
        reply_rows = int(predicted.sum())
        marker = {"version": version if version is not None else -1, "version_min": payload["version_min"], "base": is_base,
                  "tokens": upto, "rows": upto - row0, "reply_rows": reply_rows, "written": None}
        chain.files.append((name, self.saver.submit(self._save, name, payload, marker)))
        new = upto - lo
        self.stats["pieces"] += 1
        self.stats["base_pieces"] = self.stats.get("base_pieces", 0) + int(is_base)
        self.stats["tokens"] += new
        self.stats["reply_rows"] += reply_rows
        row = {"event": "stream_piece", "file": name, "new_tokens": new, "tokens": upto, "row0": row0,
               "reply_rows": reply_rows, "seconds": round(seconds, 2), "tok_per_s": round(new / max(seconds, 1e-9), 1),
               "version": version, "version_min": payload["version_min"], **({"batch": batch} if batch > 1 else {})}
        self.log(row)
        return row

    def verify(self, chain):
        """The chain's written pieces against one fresh pass over its ids (no cache reuse): per piece the rows it
        trains (row0 .. its last but one). Returns the largest and mean absolute differences of the top-k and
        realised-token log-probabilities and the top-1 agreement."""
        for _, future in chain.files:
            future.result()
        pieces = [torch.load(self.out / f"{name}.pt", weights_only=False) for name, _ in chain.files]
        n = pieces[-1]["ids"].numel()
        tmp = Chain(f"{chain.key}#verify", -1, prefix_inputs(chain.inputs, n, self.image_token_id))
        ids = tmp.ids
        logp, idx, tok = [], [], []
        began = time.time()
        for a, b, logits in self.forward(tmp, 0, n, True):
            nxt = torch.cat([ids[a + 1:b], ids[b:b + 1] if b < n else torch.zeros(1, dtype=ids.dtype)])
            got = self._top(logits, nxt[:, None])
            logp.append(got["logp"].float())
            idx.append(got["ids"].int())
            tok.append(got["at"][:, 0].float())
            del logits
        tmp.cache = None
        logp, idx, tok = torch.cat(logp).cpu(), torch.cat(idx).cpu(), torch.cat(tok).cpu()
        d_top, d_tok, same, rows = [], [], 0, 0
        for d in pieces:
            r0, end = d["row0"], d["ids"].numel() - 1  # rows r0 .. end-1 are trained by this piece
            if end <= r0:
                continue
            d_top.append((d["top_logp"][: end - r0].float() - logp[r0:end].half().float()).abs().flatten())
            ok = torch.isfinite(d["tok_logp"][: end - r0])
            d_tok.append((d["tok_logp"][: end - r0][ok] - tok[r0:end][ok]).abs())
            same += int((d["top_ids"][: end - r0, 0] == idx[r0:end, 0]).sum())
            rows += end - r0
        d_top, d_tok = torch.cat(d_top), torch.cat(d_tok)
        return {"chain": chain.key, "pieces": len(pieces), "tokens": n, "rows": rows,
                "seconds": round(time.time() - began, 1),
                "top_logp_max_abs": round(float(d_top.max()), 5), "top_logp_mean_abs": round(float(d_top.mean()), 6),
                "tok_logp_max_abs": round(float(d_tok.max()), 5) if d_tok.numel() else None,
                "tok_logp_mean_abs": round(float(d_tok.mean()), 6) if d_tok.numel() else None,
                "top1_agree": round(same / max(rows, 1), 5)}

    def _save(self, name, payload, marker):
        try:
            tmp = self.out / f".{name}.pt.partial"
            torch.save(payload, tmp)
            tmp.rename(self.out / f"{name}.pt")
            marker["written"] = time.time()
            (self.out / f".{name}.ready.partial").write_text(json.dumps(marker))
            (self.out / f".{name}.ready.partial").rename(self.out / f"{name}.ready")
        except BaseException as exc:  # noqa: BLE001 - raised by the loop
            self.error = exc

    def _handle(self, event):
        kind, key, capture, made = event
        if kind == "capture":
            self._take(key, capture, made)
        elif kind == "idle":
            self._close(key)
            if key in self.chains and self.chains[key] not in self.closing:
                self.closing.append(self.chains[key])
        elif kind == "error":
            raise RuntimeError("stream tail failed") from self.error

    def _rate(self):
        """Steady teacher tok/s: every piece after the first (g1j 23 Oct: a 132 s first piece of compile pulled the
        mean under the 2,960 bar while pieces ran at 5.5K)."""
        secs = self.stats["forward_seconds"] - self.stats.get("warm_seconds", 0.0)
        toks = self.stats["tokens"] - self.stats.get("warm_tokens", 0)
        if secs > 0 and toks > 0:
            return round(toks / secs, 1)
        return round(self.stats["tokens"] / self.stats["forward_seconds"], 1) if self.stats["forward_seconds"] else None

    def status(self):
        pending = sum(c.pending() for c in self.chains.values() if c.pending() > 0 and (not c.closed or self._worth(c)))
        s = dict(self.stats, sessions=len(self.tails), open_chains=sum(not c.closed for c in self.chains.values()),
                 cached=sum(c.cache is not None for c in self.chains.values()), pending_tokens=pending,
                 tok_per_s=self._rate(), t=round(time.time(), 1))
        warm_end = getattr(self, "_warm_end", None)
        if warm_end is not None and time.time() - warm_end > 1:  # after the first forward: wall rate (supply waits
            wall = time.time() - warm_end  # included) and the forward share of the wall (1.0: the teacher is the limit)
            s["wall_tok_per_s"] = round((s["tokens"] - s.get("warm_tokens", 0)) / wall, 1)
            s["forward_share"] = round((s["forward_seconds"] - s.get("warm_seconds", 0.0)) / wall, 3)
        s["forward_seconds"] = round(s["forward_seconds"], 1)
        s["encode_seconds"] = round(s["encode_seconds"], 1)
        s["recompute_seconds"] = round(s["recompute_seconds"], 1)
        return s

    def skip_existing(self):
        """A restarted teacher (the round resumed from its checkpoint): the play so far is read and its chains rebuilt,
        but only tokens that arrive from now on are scored (the earlier ones were trained before the restart); each
        chain's cache is rebuilt at its first new piece."""
        self._scan()
        while not self.events.empty():
            self._handle(self.events.get_nowait())
        for chain in self.chains.values():
            chain.fed, chain.cache, chain.last = chain.ids.shape[0], None, None
        self.closing = []
        self.stats["skipped_tokens"] = sum(c.fed for c in self.chains.values())
        # the rebuilt chains count pieces from 0 again: without a new tag their names repeat the pieces the trainer
        # already took (its consumed set), so it would skip every new piece and stall (g1i-pf8, 22 Oct)
        self.epoch = f"r{time.time_ns() // 1_000_000:x}-"

    def run(self, *, until=None, status_every=60.0, after_piece=None):
        """Score until ``stop_file`` exists (or ``until()`` is true); then wait for the writes."""
        threading.Thread(target=self.producer, daemon=True, name="stream-tail").start()
        last_status = 0.0
        try:
            while True:
                if self.error is not None:
                    raise RuntimeError("stream teacher failed") from self.error
                if (self.stop_file is not None and self.stop_file.exists()) or (until is not None and until(self)):
                    break
                while True:  # take every event that arrived
                    try:
                        self._handle(self.events.get_nowait())
                    except queue.Empty:
                        break
                if self.batch > 1:
                    picks = self._due_many(self.batch)
                    chain = picks[0][0] if picks else None
                else:
                    chain, upto = self._pick()
                if chain is None:
                    try:
                        self._handle(self.events.get(timeout=self.poll))
                    except queue.Empty:
                        pass
                elif self.batch > 1:
                    self.score_many(picks)
                    for c, _ in picks:
                        if after_piece is not None:
                            after_piece(self, c)
                    used = {c for c, _ in picks}
                    if self.budget is not None:
                        self.stats["evictions"] += self.budget([c for c in self.chains.values() if c not in used])
                else:
                    self.score(chain, upto)
                    if after_piece is not None:
                        after_piece(self, chain)
                    self._evict(chain)
                if time.time() - last_status > status_every:
                    last_status = time.time()
                    st = self.status()
                    self.log({"event": "stream_teacher", **st})
                    (self.out / ".status.json.partial").write_text(json.dumps(st))
                    (self.out / ".status.json.partial").rename(self.out / "status.json")
        finally:
            self.stopping.set()
            self.saver.shutdown(wait=True)
            self.pool.shutdown(wait=False, cancel_futures=True)
            if hasattr(self.encode, "shutdown"):
                self.encode.shutdown()
        if self.error is not None:
            raise RuntimeError("stream teacher failed") from self.error
        return self.status()


def _timed(run, capture):
    """A worker's encode, timed in the worker (``StreamTeacher._encode``'s contract: (result or exception, seconds))."""
    began = time.time()
    try:
        made = run(capture)
    except Exception as exc:  # noqa: BLE001 - counted by the teacher; re-raised types may not pickle
        made = RuntimeError(f"{type(exc).__name__}: {exc}")
    return made, time.time() - began


class ProcessEncoder:
    """The capture encode in ``workers`` spawned processes (PTH teacher host fix, 23 Oct: four encode threads held the
    GIL ~99% of their time and cut a launch-like loop on the scoring thread to 32% of its rate; processes: 98%, and
    1.5x the captures/s). ``init(*args)`` runs once per worker, ``run(capture)`` encodes; both module-level (the
    child imports them). No CUDA in the workers."""

    def __init__(self, init, run, args=(), workers=4):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        self.run = run
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=init,
                                        initargs=tuple(args))

    def submit(self, capture):
        return self.pool.submit(_timed, self.run, capture)

    def shutdown(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def lru_budget(limit_fn):
    """A ``budget`` hook: while ``limit_fn()`` is true (e.g. memory over a bound), drop the least recently used
    cache. Returns the evictions."""
    def hook(chains):
        dropped = 0
        for chain in sorted((c for c in chains if c.cache is not None), key=lambda c: c.used):
            if not limit_fn():
                break
            chain.cache = None
            dropped += 1
        return dropped
    return hook


def stream_bars(events_path, status_path, bars):
    """First-phase bars of a streaming round: the trainer's step compute (step seconds minus
    that step's stream wait, plus its weight gather) on its 2nd and 3rd steps, the teacher's forward tok/s once it scored ``teacher_after`` tokens.
    Returns (verdict, readings): None while unread, "pass", or "fail: ..."."""
    steps, waited, gather, last, held = [], 0.0, {}, 0.0, 0.0
    if Path(events_path).exists():
        for line in Path(events_path).read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") == "step" and row.get("seconds") is not None:
                # "seconds" and stream_wait_s are both cumulative from the trainer's start (c6, 23 Oct: the bar read
                # the elapsed time as step 3's compute, 1,370 s, and stopped the round); a checkpoint's blocking
                # part between two steps is not step compute
                wait = float(row.get("stream_wait_s") or 0.0)
                steps.append([row.get("step"), float(row["seconds"]) - last - (wait - waited) - held])
                waited, last, held = wait, float(row["seconds"]), 0.0
            elif row.get("event") == "checkpoint":
                held += float(row.get("blocking_seconds") if row.get("blocking_seconds") is not None
                              else row.get("seconds") or 0.0)
            elif row.get("event") == "delta_export":  # the per-step weight gather belongs to the step's cost
                gather[row.get("step")] = float(row.get("gather_seconds") or 0.0)
    steps = [sec + gather.get(step, 0.0) for step, sec in steps]
    teacher = json.loads(Path(status_path).read_text()) if Path(status_path).exists() else {}
    read = {"steps": [round(x, 1) for x in steps[:6]], "teacher_tok_per_s": teacher.get("tok_per_s"),
            "teacher_tokens": teacher.get("tokens")}
    fails = []
    if len(steps) >= 3:
        compute = max(steps[1:3])
        read["step_compute"] = round(compute, 1)
        if compute > bars["step_seconds"]:
            fails.append(f"step compute {compute:.0f} s > {bars['step_seconds']}")
    if (teacher.get("tokens") or 0) >= bars["teacher_after"] and teacher.get("tok_per_s") is not None:
        if teacher["tok_per_s"] < bars["teacher_tok_per_s"]:
            fails.append(f"teacher {teacher['tok_per_s']} tok/s < {bars['teacher_tok_per_s']}")
    done = len(steps) >= 3 and (teacher.get("tokens") or 0) >= bars["teacher_after"]
    return ("fail: " + "; ".join(fails)) if fails else ("pass" if done else None), read


def stream_fill(src, roll, name="fill"):
    """ARC_HEAL_STREAM_BASE_FILL=<glob of recorded base session folders, each with rollout.jsonl> (the live
    base play gives ~4 of a step's ~14 base chunks): the sessions are copied under ``roll/<name>/`` once, a base root
    the teacher scores like a finished base play ("b-" pieces, no age limit). Returns the root name."""
    n = 0
    import glob
    for f in sorted(Path(x) for x in glob.glob(src.rstrip("/") + "/rollout.jsonl")):  # src: a folder glob ("…/b-*")
        dst = Path(roll) / name / f.parent.name / "rollout.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst)
        n += 1
    if not n:
        raise SystemExit(f"ARC_HEAL_STREAM_BASE_FILL={src}: no <folder>/rollout.jsonl")
    return name


def topk_file_kl(saved, other, rows_from="reply"):
    """Teacher-file comparison (MFU-2 FP8 gate, 23 Oct): forward KL(saved || other) per reply row over saved's top-k
    + tail bucket (heal_kd.bucket_kl's forward term), other's log-prob at saved's ids from its own top-k (ids outside
    it: its k-th log-prob, an upper bound; their share reported), plus top-1 agreement. Row i predicts token i+1."""
    a = torch.load(saved, weights_only=False) if not isinstance(saved, dict) else saved
    b = torch.load(other, weights_only=False) if not isinstance(other, dict) else other
    if not torch.equal(a["ids"], b["ids"]):
        return {"error": "ids differ", "n": [a["ids"].numel(), b["ids"].numel()]}
    n = a["top_ids"].shape[0]
    reply = a[rows_from].bool()
    rows = torch.zeros(n, dtype=torch.bool)
    m = min(n, reply.numel() - 1)
    rows[:m] = reply[1:m + 1]
    kl_sum, top1, missing, k = 0.0, 0, 0, a["top_ids"].shape[1]
    for s0 in range(0, n, 2048):
        sl = slice(s0, min(n, s0 + 2048))
        r = rows[sl]
        if not r.any():
            continue
        ta, la, tail = a["top_ids"][sl][r].long(), a["top_logp"][sl][r].float(), a["tail"][sl][r].float()
        tb, lb = b["top_ids"][sl][r].long(), b["top_logp"][sl][r].float()
        eq = ta[:, :, None] == tb[:, None, :]
        hit = eq.any(-1)
        q = torch.where(hit, (eq.float() * lb[:, None, :]).sum(-1), lb[:, -1:].expand_as(la))
        q_tail = torch.log1p(-q.exp().sum(-1).clamp(max=1 - 1e-6))
        fwd = (la.exp() * (la - q)).sum(-1) + tail.exp() * (tail - q_tail)
        kl_sum += float(fwd.double().sum())
        top1 += int((ta[:, 0] == tb[:, 0]).sum())
        missing += int((~hit).sum())
    count = int(rows.sum())
    return {"rows": count, "kl": kl_sum / max(count, 1), "top1": top1 / max(count, 1),
            "missing_share": missing / max(count * k, 1)}
