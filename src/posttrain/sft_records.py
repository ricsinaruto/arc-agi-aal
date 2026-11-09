"""One SFT record shape for every post-training data source.

A record is one model request and the reply to learn:

    {"id", "source", "game_id", "level", "turn",
     "messages": [...],          # the exact context the model saw (system, user, assistant, tool)
     "tools": [...],             # the tool list sent with the request
     "target": {"reasoning": str, "content": str, "tool_calls": [...]},
     "prompt_tokens_estimate": int, "images": int}

- History assistant messages carry their reasoning as ``reasoning_content`` (the chat-template key,
  as in ``posttrain.segments``); the harness's own request payloads call it ``reasoning`` and are
  renamed. Nothing else in the messages is changed.
- ``target.tool_calls`` is the OpenAI list the harness dispatched (``[]`` for a tool-less reply, e.g.
  a compaction summary).

Sources:

- ``solver`` (source (a)): ``solver_records`` turns a solver run (``scripts/posttrain/solver_play.py``:
  ``relay/turn-NNNN.request.json`` + ``.response.json``) into one record per request.
- ``segment`` (source (c)): ``segment_records`` is the adapter for ``posttrain.segments``, whose
  records are level transcripts with the full context the teacher saw (earlier levels' turns from
  the latest compaction on; ``target_start`` marks the level's first message). It yields one record
  per assistant message from ``target_start`` on, with the exact request that the teacher's server
  received: the system prompt and every message before it, or, after a compaction, only the
  messages from the compaction index on (``compaction_indices``). ``context="segment"`` restores
  the old truncated context (the level's own turns only), kept for before/after comparisons.
  Records carry the served ``served_prompt_tokens``, ``finish_reason`` and ``python_error`` of
  their request (``request_meta``) when the segment has them.

``split`` is ``"train"`` (the default; ``check_training_games``) or ``"eval"`` (the held-out
generated mechanics only; ``check_eval_games``).
"""

import copy
import json
from pathlib import Path

from .relay import response_json
from .solver_policy import latest_state
from .split import check_eval_games, check_training_games

# Calibrated on the teacher's first requests (system prompt + first user turn + tools + one board
# image = 4,155 prompt tokens for about 16,980 characters); an upscaled 64x64 board is ~64 tokens.
CHARS_PER_TOKEN = 4.1
IMAGE_TOKENS = 64
RECORD_KEYS = ("id", "source", "game_id", "level", "turn", "messages", "tools", "target",
               "prompt_tokens_estimate", "images")


def _texts(content):
    if isinstance(content, list):
        return [part.get("text") or "" for part in content if isinstance(part, dict) and part.get("type") == "text"]
    return [content or ""] if isinstance(content, str) else []


def count_images(messages):
    return sum(1 for m in messages if isinstance(m.get("content"), list)
               for part in m["content"] if isinstance(part, dict) and part.get("type") == "image_url")


def prompt_tokens_estimate(messages, tools):
    chars = len(json.dumps(tools or [], ensure_ascii=False))
    for message in messages:
        chars += sum(len(t) for t in _texts(message.get("content")))
        chars += len(message.get("reasoning_content") or message.get("reasoning") or "")
        chars += len(json.dumps(message.get("tool_calls") or [], ensure_ascii=False)) if message.get("tool_calls") else 0
    return int(chars / CHARS_PER_TOKEN) + IMAGE_TOKENS * count_images(messages)


def normalize_messages(messages):
    """Copy of ``messages`` with assistant reasoning under ``reasoning_content``."""
    out = copy.deepcopy(messages)
    for message in out:
        if message.get("role") == "assistant" and "reasoning" in message:
            reasoning = message.pop("reasoning")
            message.setdefault("reasoning_content", reasoning or "")
    return out


def make_record(*, source, game_id, level, turn, messages, tools, target, record_id=None, **extra):
    messages = normalize_messages(messages)
    record = {
        "id": record_id or f"{source}:{game_id}:T{turn}",
        "source": source,
        "game_id": game_id,
        "level": level,
        "turn": turn,
        "messages": messages,
        "tools": tools or [],
        "target": {"reasoning": target.get("reasoning") or "", "content": target.get("content") or "",
                   "tool_calls": list(target.get("tool_calls") or [])},
        "prompt_tokens_estimate": prompt_tokens_estimate(messages, tools),
        "images": count_images(messages),
    }
    record.update(extra)
    return record


def check_record(record):
    """Raise if ``record`` is not in the shared shape."""
    missing = [key for key in RECORD_KEYS if key not in record]
    if missing:
        raise ValueError(f"{record.get('id')}: missing {missing}")
    messages = record["messages"]
    if not messages or messages[0].get("role") != "system":
        raise ValueError(f"{record['id']}: messages must start with the system prompt")
    if messages[-1].get("role") not in ("user", "tool"):
        raise ValueError(f"{record['id']}: the context must end with a user or tool message")
    target = record["target"]
    if set(target) != {"reasoning", "content", "tool_calls"}:
        raise ValueError(f"{record['id']}: bad target keys {sorted(target)}")
    if not (target["reasoning"] or target["content"] or target["tool_calls"]):
        raise ValueError(f"{record['id']}: empty target")
    for call in target["tool_calls"]:
        json.loads(call["function"]["arguments"])
    if any("reasoning" in m for m in messages if m.get("role") == "assistant"):
        raise ValueError(f"{record['id']}: assistant reasoning must be under reasoning_content")
    if record.get("split", "train") == "eval":
        check_eval_games([record["game_id"]])
    else:
        check_training_games([record["game_id"]])
    return record


# --- source (a): solver runs ----------------------------------------------------------------------


def solver_records(run_dir, game_id, *, split="train"):
    """One record per relay turn of a solver run folder (``<run>/relay/turn-NNNN.*.json``)."""
    (check_eval_games if split == "eval" else check_training_games)([game_id])
    relay = Path(run_dir) / "relay"
    records = []
    for request_path in sorted(relay.glob("turn-*.request.json")):
        turn = int(request_path.name.split(".")[0].split("-")[1])
        response_path = relay / f"turn-{turn:04d}.response.json"
        if not response_path.exists():
            raise FileNotFoundError(response_path)
        payload = json.loads(request_path.read_text())
        answer = json.loads(response_path.read_text())
        message = response_json(answer, turn, 0)["choices"][0]["message"]
        step, level, _ = latest_state(payload["messages"])
        purpose = "gameplay" if payload.get("tools") else "summary"
        records.append(make_record(
            source="solver", game_id=game_id, level=level, turn=turn, messages=payload["messages"],
            tools=payload.get("tools"), step=step, purpose=purpose, split=split,
            target={"reasoning": answer.get("reasoning"), "content": answer.get("content"),
                    "tool_calls": message.get("tool_calls") or []}))
    return [check_record(r) for r in records]


# --- source (c): the adapter for posttrain.segments -------------------------------------------------


def segment_contexts(segment, *, context="full"):
    """``[(target index, context indices)]`` of a ``posttrain.segments`` record: every assistant
    message from ``target_start`` on, with the indices of the messages its request saw (the system
    prompt first). Segments with per-request contexts (``request_meta[i]["context"]``: index ranges)
    use them; older ones cut at the latest compaction index. ``context="segment"`` keeps only the
    level's own messages (the old truncated context)."""
    if context not in ("full", "segment"):
        raise ValueError(f"context must be 'full' or 'segment', not {context!r}")
    messages = segment["messages"]
    cuts = sorted(segment.get("compaction_indices") or [])
    first = int(segment.get("target_start") or 1)
    meta = list(segment.get("request_meta") or [])
    targets = [i for i, m in enumerate(messages) if m.get("role") == "assistant" and i >= first]
    out = []
    for n, index in enumerate(targets):
        ranges = (meta[n] if n < len(meta) else {}).get("context")
        if ranges:
            seen = [p for a, b in ranges for p in range(a, b)]
        else:
            start = max([c for c in cuts if c <= index], default=1)
            seen = [0, *range(start, index)]
        if context == "segment":  # a summary right before the level's first message is the level's own
            seen = [p for p in seen if p == 0 or p >= first or (p == first - 1 and p in cuts)]
        if seen[:1] != [0] or any(b <= a for a, b in zip(seen, seen[1:], strict=False)) or seen[-1] >= index:
            raise ValueError(f"{segment.get('id')}: bad context for message {index}")
        out.append((index, seen))
    return out


def segment_records(segment, *, context="full", split="train"):
    """Per-request records (shared shape) from one ``posttrain.segments`` record: one per assistant
    message from ``target_start`` on (segments without it: every assistant message)."""
    messages = segment["messages"]
    meta = list(segment.get("request_meta") or [])
    out = []
    for request, (index, seen) in enumerate(segment_contexts(segment, context=context), start=1):
        message = messages[index]
        info = meta[request - 1] if request - 1 < len(meta) else {}
        extra = {}
        if info:
            extra = {"served_prompt_tokens": info.get("prompt_tokens"), "finish_reason": info.get("finish_reason"),
                     "python_error": info.get("python_error"), "actions": info.get("actions")}
        out.append(make_record(
            source="segment", game_id=segment["game_id"], level=segment["level"], turn=request,
            record_id=f"segment:{segment['id']}:R{request}", messages=[messages[p] for p in seen],
            tools=segment.get("tools"), run=segment.get("run"), split=split, **extra,
            target={"reasoning": message.get("reasoning_content"), "content": message.get("content"),
                    "tool_calls": message.get("tool_calls") or []}))
    return [check_record(r) for r in out]
