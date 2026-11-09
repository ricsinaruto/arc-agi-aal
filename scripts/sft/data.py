"""trunk/sft data primitives: a recorded play's rollout.jsonl -> served requests -> token-exact training examples.

``served_messages``, ``encode_request`` and ``chain_examples`` are moved verbatim from
scripts/posttrain/flash_qlora_job.py (sft-heal 5f7b222de; tests/golden masking pins them). ``read_rollout`` is
``heal_stream.SessionTail.read`` over a whole file."""
import json
from pathlib import Path

ROLLOUT_KEYS = ("tokens", "logprobs", "top", "support", "reference", "sampling", "prompt_digest", "finish_reason", "step",
                "w", "version")


def read_rollout(path):
    """Gameplay requests of one rollout.jsonl, in order: messages = messages[:keep] + new, tools carried; rows without
    served prompt or completion counts left out (as ``SessionTail.read``)."""
    messages, tools, out = [], None, []
    with open(Path(path), "rb") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = messages[: row["keep"]] + row["messages"]
            tools = row["tools"] if "tools" in row else tools
            if not row.get("gameplay", True):
                continue
            payload = {"messages": list(messages), "chat_template_kwargs": row["template"]}
            if tools is not None:
                payload["tools"] = tools
            out.append({"request": row["n"], "prompt_tokens": (row["usage"] or {}).get("prompt_tokens"),
                        "payload": payload, "reply": row["reply"], "usage": row["usage"],
                        "rollout": {k: row.get(k) for k in ROLLOUT_KEYS}, "scripted": bool(row.get("scripted")),
                        "capped": bool(row.get("capped"))})
    return [c for c in out if c["prompt_tokens"] and (c["usage"] or {}).get("completion_tokens")]


def served_messages(messages):
    """Chat messages as the served template gets them from vLLM: tool arguments parsed, ``reasoning`` under the
    template's name, images as placeholders (returned apart, in order)."""
    import base64
    import io

    from PIL import Image

    out, images = [], []
    for message in messages:
        message = {key: value for key, value in message.items() if value is not None}
        if isinstance(message.get("content"), list):
            parts = []
            for part in message["content"]:
                if part.get("type") == "image_url":
                    raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                    images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                    part = {"type": "image"}
                parts.append(part)
            message["content"] = parts
        if message.get("tool_calls"):
            message["tool_calls"] = [{**call, "function": {**call["function"], "arguments": json.loads(
                call["function"]["arguments"])}} for call in message["tool_calls"]]
        if "reasoning" in message:
            message["reasoning_content"] = message.pop("reasoning")
        out.append(message)
    return out, images


def encode_request(processor, capture):
    """A captured request (``arena.level_bench`` fork capture: the served payload and the recorded reply) as one
    training example: the prompt exactly as the server tokenized it, then the reply; the loss is on the reply only
    (through its end-of-turn token). ``checks`` compares the counts with the server's own."""
    import torch

    payload, tokenizer = capture["payload"], processor.tokenizer
    settings = dict(tools=payload.get("tools"), tokenize=False, **(payload.get("chat_template_kwargs") or {}))
    messages, images = served_messages(payload["messages"])
    reply, _ = served_messages([{**capture["reply"], "role": "assistant"}])
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **settings)
    whole = tokenizer.apply_chat_template(messages + reply, add_generation_prompt=False, **settings)
    end = "<|im_end|>"
    if not whole.startswith(prompt) or not whole.rstrip("\n").endswith(end):
        raise ValueError("the reply does not render as a continuation of the served prompt")
    whole = whole[: whole.rindex(end) + len(end)]

    def encode(text):
        return processor(text=[text], images=images or None, return_tensors="pt")

    head, inputs = encode(prompt), encode(whole)
    count, ids = head["input_ids"].shape[1], inputs["input_ids"]
    labels = ids.clone()
    labels[:, :count] = -100
    usage = capture.get("usage") or {}
    checks = {"request": capture.get("request"), "prompt_tokens": count, "served_prompt_tokens": capture["prompt_tokens"],
              "reply_tokens": ids.shape[1] - count, "served_reply_tokens": usage.get("completion_tokens"),
              "images": len(images), "image_tokens": int((ids == processor.image_token_id).sum()),
              "prefix_kept": bool(torch.equal(ids[:, :count], head["input_ids"]))}
    return {**inputs, "labels": labels}, checks


def mask_capped(ids, labels, spans):
    """Reply tokens past a level's token cap stay in the example as context: clears ``labels`` (in place) on each
    ``(start, end, request ids)`` this example's ``ids`` extend, so ids, chains and file names match the uncapped
    build and its teacher files still cover the labels. Returns the tokens cleared."""
    import torch

    cleared = 0
    for start, end, req in spans:
        if end <= len(ids) and bool((labels[start:end] != -100).any()) and torch.equal(ids[:end], req):
            cleared += int((labels[start:end] != -100).sum())
            labels[start:end] = -100
    return cleared


def chain_examples(encode, captures, *, window, margin=8192, floor=16384):
    """Training examples of one chain (``build_corpus.chain_ids``: targets of one game whose served prompts extend
    one another), from its end: the last target's prompt and reply, with the loss also on every earlier reply of
    the chain that starts inside the gradient ``window``. Such a reply sits in the later prompt exactly as it was
    generated; that is checked token by token against its own encoding, and a reply that fails it, or lies outside
    the window, ends the next example instead. ``encode(capture)`` gives ``(inputs, prompt tokens)`` or None for a
    target that is left out. Returns ``[(inputs, tail, targets)]``; ``tail`` is the gradient window of the
    example: its replies plus ``margin`` tokens of context, at least ``floor``, at most ``window``."""
    rows = [(capture, done) for capture, done in ((c, encode(c)) for c in captures) if done is not None]
    out = []
    while rows:
        _, (inputs, start) = rows.pop()
        ids, labels = inputs["input_ids"], inputs["labels"]
        total, first, targets = ids.shape[1], start, 1
        if total - start + 1 > window:  # the reply alone is longer than the window: left out, never truncated
            continue
        while rows:
            earlier_ids, earlier_start = rows[-1][1][0]["input_ids"], rows[-1][1][1]
            end = earlier_ids.shape[1]
            if total - earlier_start + 1 > window or end > first or not bool((ids[:, :end] == earlier_ids).all()):
                break
            labels[:, earlier_start:end] = ids[:, earlier_start:end]
            first, targets = earlier_start, targets + 1
            rows.pop()
        out.append((inputs, min(total, window, max(floor, total - first + margin)), targets))
    return out
