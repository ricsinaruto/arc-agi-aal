"""Next-board aux loss (envs arm): a packed conversation (``arena.nextboard_pack``) as token ids with every trained
assistant target's span. The text is the chat template's own rendering of the conversation with thinking off: each
assistant turn opens ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`` (what the server's generation prompt and
the template's kept-history form both write), so every target sits where a no-think reply sits. The target tokens are
the patch and ``<|im_end|>``."""

USER = "<|im_start|>user\n{text}<|im_end|>\n"
OPEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
CLOSE = "<|im_end|>\n"


def text_of(row):
    """The rendered conversation (for the template check)."""
    return "".join(USER.format(text=t["user"]) + OPEN + t["target"] + CLOSE for t in row["turns"])


def encode(tokenizer, row, max_tokens=None):
    """``{"ids": [int], "spans": [(start, stop, step)], "changed": [int]}``: target token positions [start, stop) per
    trained step (patch + ``<|im_end|>``). Pieces are tokenized apart and joined; every piece boundary is a special
    token or the newline after one, so the ids equal the whole text's (checked in the tests). ``max_tokens`` cuts the
    conversation after the last whole step that fits."""
    ids, spans, changed = [], [], []
    enc = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]
    for step, turn in enumerate(row["turns"]):
        head = enc(USER.format(text=turn["user"]) + OPEN)
        body = enc(turn["target"])
        end = enc("<|im_end|>")
        tail = enc("\n")
        if max_tokens and len(ids) + len(head) + len(body) + len(end) > max_tokens:
            break
        ids += head
        start = len(ids)
        ids += body + end
        if turn.get("train", True):
            spans.append((start, len(ids), step))
            changed.append(int(turn.get("changed", 0)))
        ids += tail
    return {"ids": ids, "spans": spans, "changed": changed}


def made(encoded, device="cpu", name="nb"):
    """A ``window_pass`` window: whole sequence as the tail (no prefix cache), plain cross-entropy over the full
    vocabulary at temperature 1 (support -1, weight 1, no sampler reference: no TIS, no GSPO reply groups), greedy hits
    returned (``argmax``)."""
    import torch

    ids = torch.tensor([encoded["ids"]], dtype=torch.long, device=device)
    positions = torch.tensor([p for s, e, _ in encoded["spans"] for p in range(s, e)], dtype=torch.long, device=device)
    n = int(positions.numel())
    return {"inputs": {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, "tail": int(ids.shape[1]),
            "temperature": 1.0, "name": name, "argmax": True,
            "arrays": {"positions": positions, "support": torch.full((n, 1), -1, dtype=torch.long, device=device),
                       "weights": torch.ones(n, device=device)}}


def step_stats(encoded, raw, hit):
    """Per trained step of one window: exact match (every target token greedy), body match (patch tokens without the
    first and the end token), token hits, NLL sum and token count. ``raw``/``hit``: per target position, in span
    order."""
    rows, at = [], 0
    for (s, e, step), changed in zip(encoded["spans"], encoded["changed"]):
        n = e - s
        h, r = hit[at:at + n], raw[at:at + n]
        at += n
        rows.append({"step": step, "n": n, "em": bool(h.all()), "body": bool(h[1:-1].all()) if n > 2 else bool(h.all()),
                     "hits": int(h.sum()), "nll": float(-r.sum()), "changed": changed})
    return rows


def pooled(rows):
    """Exact / body match share, token accuracy and mean token NLL over step rows; the same on changed steps."""
    def agg(rs):
        tokens = sum(r["n"] for r in rs)
        if not rs:
            return {"steps": 0}
        return {"steps": len(rs), "em": round(sum(r["em"] for r in rs) / len(rs), 4),
                "body": round(sum(r["body"] for r in rs) / len(rs), 4),
                "token_acc": round(sum(r["hits"] for r in rs) / tokens, 4),
                "nll": round(sum(r["nll"] for r in rs) / tokens, 4), "tokens": tokens}
    return {**agg(rows), "changed": agg([r for r in rows if r["changed"]])}
