"""RL rollouts (8 October plan, docs/rl-plan.md): what the group runner stores per live reply, as the trainer reads it.

Record. ``arena.rollout_record`` writes one line of ``attempts/<label>/rollout.jsonl`` per live reply: the served
request as a delta against the attempt's previous one (``keep`` leading messages unchanged, then ``messages``), the
reply, the sampled token ids with the engine's log-probability of each, and per token the engine's top alternatives
(``top``: ``[ids, logprobs]``, highest first). ``captures`` rebuilds the trainer's capture rows from it (the format
of the level bench's fork capture, ``flash_qlora_job.encode_request``).

Actor policy. The actor samples at temperature T from the top-k tokens, cut to the top-p nucleus (vLLM: temperature,
then top-k, then top-p on the renormalised top-k). ``nucleus`` rebuilds that set and its probabilities from RAW top-k
log-probabilities; ``compact`` keeps that set per sampled token in place of the alternatives.
"""

import hashlib
import json
import math

CLAMPED = -9000.0  # vLLM reports -inf (a token the processors removed) as -9999


# --- record --------------------------------------------------------------------------------------------------

def ids_digest(ids):
    """A digest of a token-id sequence (the served prompt): equal digests mean the trainer saw the same context
    (the same function as ``arena.rollout_record.ids_digest``, which a bundle ships)."""
    return hashlib.sha256(b"".join(int(t).to_bytes(4, "little") for t in ids)).hexdigest()[:32]


def restore(rows):
    """Stored reply rows with every request's full message list and tool list restored."""
    out, messages, tools = [], [], None
    for row in rows:
        messages = messages[: row["keep"]] + row["messages"]
        tools = row["tools"] if "tools" in row else tools
        out.append({**row, "messages": list(messages), "tools": tools})
    return out


def load(path):
    """The lines of a ``rollout.jsonl``, restored."""
    with open(path) as file:
        return restore(json.loads(line) for line in file)


ROLLOUT_KEYS = ("tokens", "logprobs", "top", "support", "reference", "sampling", "prompt_digest", "finish_reason", "step",
                "w")


def captures(source, gameplay=True):
    """Trainer capture rows of one attempt, in order: ``payload`` and ``reply`` as ``encode_request`` reads them, and
    ``rollout`` (sampled ids, engine log-probabilities, alternatives or the compact actor set, sampling settings)
    for the ratio. ``source``: a ``rollout.jsonl`` path or its (unrestored) rows. ``records``: the reply-id records of
    every earlier request of the attempt (gameplay or not; ``reply_records``), for ``substitute``."""
    out, records = [], []
    for row in (restore(source) if isinstance(source, list) else load(source)):
        before = list(records)
        records += reply_records([row])
        if gameplay and not row["gameplay"]:
            continue
        payload = {"messages": row["messages"], "chat_template_kwargs": row["template"]}
        if row["tools"] is not None:
            payload["tools"] = row["tools"]
        out.append({"request": row["n"], "prompt_tokens": (row["usage"] or {}).get("prompt_tokens"), "payload": payload,
                    "reply": row["reply"], "usage": row["usage"], "records": before,
                    "rollout": {k: row.get(k) for k in ROLLOUT_KEYS}})
    return out


# --- reply ids (the play server's arena.reply_ids_runtime, replayed from the rollout) ---------------------------

def reply_records(rows):
    """``[(served prompt length, prompt digest, generated ids)]`` of the rows whose reply ends with a turn end: what
    the server's reply-id cache recorded (29 Oct: the trainer re-tokenized every earlier reply canonically while the
    server re-sent 3.6% of them with the ids they were generated with; 36% of v12's trained tokens sat in or after one)."""
    out = []
    for row in rows:
        tokens, digest = row.get("tokens") or [], row.get("prompt_digest")
        length = (row.get("usage") or {}).get("prompt_tokens")
        if len(tokens) >= 2 and digest and length:
            out.append((int(length), str(digest), [int(t) for t in tokens]))
    return out


def same_text(tokenizer, a, b):
    """The server's test: both decode to the same string and the same token-string concatenation."""
    if tokenizer.decode(a, skip_special_tokens=False) != tokenizer.decode(b, skip_special_tokens=False):
        return False
    return "".join(tokenizer.convert_ids_to_tokens(a)) == "".join(tokenizer.convert_ids_to_tokens(b))


def substitute(ids, records, tokenizer, images=()):
    """``ids`` (a canonical encoding) with each earlier reply's span replaced by its generated ids where the server
    did it (``arena.reply_ids_runtime.ReplyIds.substitute``, keyed by the rollout's truncated prompt digest): at an
    output position whose preceding ids hash as a recorded prompt, the span up to the next turn end, when it is the
    same text and holds no image (``images``: [(start, stop)] input spans). Returns ``(new ids, [(input start, input
    stop, replacement)])``; identical spans are left alone."""
    by_key, special = {}, set(getattr(tokenizer, "all_special_ids", None) or ())
    for length, digest, tokens in records:
        if special and tokens[-1] not in special:  # the server records only replies that end with a turn end
            continue
        by_key.setdefault((length, digest), []).append((tokens[:-1], tokens[-1]))
    lengths = sorted({k[0] for k in by_key})
    n, out, splices, delta, cursor = len(ids), [], [], 0, 0
    hasher, hashed = hashlib.sha256(), 0  # over the output sequence so far

    def feed(seq):
        hasher.update(b"".join(int(t).to_bytes(4, "little") for t in seq))

    for length in lengths:
        start = length - delta
        if start >= n:
            break
        if start < cursor or start < hashed:
            continue
        feed(ids[hashed:start])
        hashed = start
        replies = by_key.get((length, hasher.copy().hexdigest()[:32]))
        if not replies:
            continue
        for body, turn_end in replies:
            if start + len(body) < n and ids[start + len(body)] == turn_end and list(ids[start:start + len(body)]) == body:
                break  # resent identical: nothing to do
            try:
                end = list(ids).index(turn_end, start)
            except ValueError:
                continue
            if any(a < end and b > start for a, b in images) or not same_text(tokenizer, list(ids[start:end]), body):
                continue
            out += list(ids[cursor:start]) + body
            splices.append((start, end, body))
            feed(body)
            cursor = hashed = end
            delta += len(body) - (end - start)
            break
    if not splices:
        return list(ids), []
    return out + list(ids[cursor:]), splices


# --- actor policy --------------------------------------------------------------------------------------------

def nucleus(ids, logprobs, *, temperature, top_p, top_k):
    """The actor's sampling set at one position from RAW log-probabilities of its top tokens (highest first):
    ``[(id, log-probability under the actor), ...]``. A token stays while the probability of the tokens above it
    is below ``top_p`` (vLLM removes a token when it and everything below it hold at most ``1 - top_p``)."""
    count = min(len(ids), top_k or len(ids))
    scaled = [value / temperature for value in logprobs[:count]]
    high = max(scaled)
    weights = [math.exp(value - high) for value in scaled]
    total = sum(weights)
    kept, above = [], 0.0
    for index in range(count):
        if index and above / total >= top_p:
            break
        kept.append(index)
        above += weights[index]
    mass = sum(weights[i] for i in kept)
    return [(ids[i], math.log(weights[i] / mass)) for i in kept]


def compact(row, reported="raw"):
    """A stored reply without its alternatives: per sampled token the actor's sampling set (``support``: token ids)
    and the token's log-probability under the actor (``reference``; None for a token outside the set, which is then
    left out of the update). ``reported`` = ``processed``: the engine's values already are the actor's."""
    sampling, support, reference = row["sampling"], [], []
    for token, (ids, values) in zip(row["tokens"], row.get("top") or []):
        finite = [(t, v) for t, v in zip(ids, values) if v > CLAMPED]
        kept = finite if reported == "processed" else nucleus(
            [t for t, _ in finite], [v for _, v in finite], temperature=sampling["temperature"],
            top_p=sampling["top_p"], top_k=sampling["top_k"])
        support.append([t for t, _ in kept])
        reference.append(round(dict(kept)[token], 5) if token in dict(kept) else None)
    return {**{k: v for k, v in row.items() if k != "top"}, "support": support, "reference": reference}
