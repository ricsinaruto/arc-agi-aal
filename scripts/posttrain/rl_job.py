"""RL trainer helpers (8 October) used by ``rl1_trainer.py``: rollout windows (``plan_windows``, ``window_example``,
``attempt_example``) and actor / raw log-probabilities at the actor's temperature and sampling set."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def rollouts():
    """``posttrain.rl_rollouts``: beside this file in the image, under ``src`` in the repository."""
    try:
        import rl_rollouts
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from posttrain import rl_rollouts
    return rl_rollouts


def actor_logprobs(rows, tokens, support, temperature, at=None, split=None):
    """``(log-probability of each sampled token under the actor, valid)``. ``rows`` [n, V]: the logits that predict
    the tokens (or, with ``at`` [n], all logits [T, V] and the rows to read: only the needed entries are copied,
    a window's rows are 8 GB at 16K targets); ``support`` [n, K]: the actor's sampling set per position (-1
    padding), inside which the tempered distribution is renormalised, as the actor's sampler does; a row without a
    set (-1 first) uses the whole vocabulary. A sampled token outside its set is not valid (left out of the update)."""
    import torch

    at = torch.arange(rows.shape[0], device=rows.device) if at is None else at
    out = torch.zeros(at.shape[0], dtype=torch.float32, device=rows.device)
    valid = torch.ones(at.shape[0], dtype=torch.bool, device=rows.device)
    if split is not None:  # (whole, part) row indices built on the host (U2, 27 Oct): no mask test, no nonzero sync
        whole_i, part_i = split
        if whole_i.numel():
            out = out.index_copy(0, whole_i, torch.log_softmax(rows[at[whole_i]].float() / temperature, dim=-1)
                                 .gather(1, tokens[whole_i, None])[:, 0])
        if part_i.numel():
            ids = support[part_i]
            scores = (rows[at[part_i][:, None], ids.clamp(min=0)].float() / temperature).masked_fill(ids < 0, float("-inf"))
            hit = ids == tokens[part_i, None]
            out = out.index_copy(0, part_i, scores.gather(1, hit.float().argmax(dim=1, keepdim=True))[:, 0]
                                 - scores.logsumexp(dim=1))
            valid = valid.index_copy(0, part_i, hit.any(dim=1))
        return out, valid
    whole = support[:, 0] < 0
    if bool(whole.any()):
        out[whole] = torch.log_softmax(rows[at[whole]].float() / temperature, dim=-1).gather(1, tokens[whole, None])[:, 0]
    if bool((~whole).any()):
        ids = support[~whole]
        scores = (rows[at[~whole][:, None], ids.clamp(min=0)].float() / temperature).masked_fill(ids < 0, float("-inf"))
        hit = ids == tokens[~whole, None]
        out[~whole] = scores.gather(1, hit.float().argmax(dim=1, keepdim=True))[:, 0] - scores.logsumexp(dim=1)
        valid[~whole] = hit.any(dim=1)
    return out, valid


def raw_logprobs(rows, tokens, chunk=1024, at=None):
    """Log-probability of each token over the whole vocabulary at temperature 1 (what the engine reports)."""
    import torch

    at = torch.arange(rows.shape[0], device=rows.device) if at is None else at
    with torch.no_grad():
        return torch.cat([torch.log_softmax(rows[at[i:i + chunk]].float(), dim=-1).gather(1, tokens[i:i + chunk, None])[:, 0]
                          for i in range(0, at.shape[0], chunk)]) if at.shape[0] else rows.new_zeros(0).float()


def plan_windows(captures, window):
    """Capture indexes at which a gradient window ends, from the served counts alone (last first): a window holds
    the replies before its last one while they fit and the prompt keeps extending (a drain starts a new one)."""
    ends, index = [], len(captures) - 1
    while index >= 0:
        last = captures[index]
        total = last["prompt_tokens"] + last["usage"]["completion_tokens"]
        ends.append(index)
        first, index = last["prompt_tokens"], index - 1
        while index >= 0:
            earlier = captures[index]
            stop = earlier["prompt_tokens"] + earlier["usage"]["completion_tokens"]
            if total - earlier["prompt_tokens"] + 1 > window or stop > first:
                break
            first, index = earlier["prompt_tokens"], index - 1
    return ends


def heaviest_end(captures, ends, window):
    """The window end whose replies carry the most credit (sum of |weight| x tokens), from the served counts: the
    replies before the end while they fit, as ``plan_windows`` counts them."""
    return max(ends, key=lambda end: _mass(captures, end, window))


def _mass(captures, end, window):
    last = captures[end]
    total, first, index, found = last["prompt_tokens"] + last["usage"]["completion_tokens"], last["prompt_tokens"], end, 0.0
    while index >= 0:
        row = captures[index]
        stop = row["prompt_tokens"] + row["usage"]["completion_tokens"]
        if index != end and (total - row["prompt_tokens"] + 1 > window or stop > first):
            break
        found += abs(row["rollout"].get("w") or 0.0) * row["usage"]["completion_tokens"]
        first, index = row["prompt_tokens"], index - 1
    return found


def window_example(encode, captures, end, *, window, margin=8192, floor=16384, credited_only=False):
    """The training example that ends with capture ``end`` (one window of a chain of served prompts, with where
    each reply sits):``(inputs, tail, [(capture index, start, stop)])`` or None when the reply does not fit.
    An earlier reply is included when it starts inside the window and sits in the later prompt token for token;
    ``credited_only``: not a reply of weight 0 (a restored attempt's scripted history stays in the no-gradient
    prefix, so the window is the credited span plus ``margin``); "scripted" (whole_segment): only the restored history
    (weight None) stops the walk, every model output after it is in the window."""
    done = encode(captures[end])
    if done is None:
        return None
    inputs, start = done
    ids, labels = inputs["input_ids"], inputs["labels"]
    total, first, spans, index = ids.shape[1], start, [(end, start, ids.shape[1])], end - 1
    if total - start + 1 > window:
        return None
    while index >= 0:
        if credited_only == "scripted":
            if captures[index]["rollout"].get("w") is None:
                break
        elif credited_only and not (captures[index]["rollout"].get("w") or 0.0):
            break
        earlier = encode(captures[index])
        if earlier is None:
            break
        earlier_ids, earlier_start = earlier[0]["input_ids"], earlier[1]
        stop = earlier_ids.shape[1]
        if total - earlier_start + 1 > window or stop > first or not bool((ids[:, :stop] == earlier_ids).all()):
            break
        labels[:, earlier_start:stop] = ids[:, earlier_start:stop]
        spans.append((index, earlier_start, stop))
        first, index = earlier_start, index - 1
    return inputs, min(total, window, max(floor, total - first + margin)), spans[::-1]


def span_arrays(rollout, reply_ids):
    """Per token of one re-encoded reply: ``(reference, support, engine raw log-probability, aligned)``. The engine's
    values are used up to the first token where its sampled ids differ from the reply's (the text of a sampled
    sequence can re-encode differently: 4 of 273 replies in the rollout smoke); ``aligned``: no such token (one
    token of slack at the end)."""
    count, tokens = len(reply_ids), rollout.get("tokens") or []
    shared = next((i for i, (a, b) in enumerate(zip(tokens, reply_ids)) if a != b), min(count, len(tokens)))
    aligned = shared > 0 and shared == min(count, len(tokens)) and abs(count - len(tokens)) <= 1
    nothing = float("nan")
    reference, support, raw = [nothing] * count, [[] for _ in range(count)], [nothing] * count
    if shared:
        raw[:shared] = [float(v) for v in (rollout.get("logprobs") or [])[:shared]] or raw[:shared]
        for i, (ids, value) in enumerate(zip(rollout.get("support") or [], rollout.get("reference") or [])):
            if i < shared:
                support[i] = [int(t) for t in ids]
                reference[i] = nothing if value is None else float(value)
    return reference, support, raw, aligned


def credited_ends(attempt, window, gameplay=True):
    """Every planned window end of an attempt whose replies carry weight (whole-segment training: one window per
    end, together covering every weighted reply that fits a window). ``gameplay`` False: every model output."""
    captures = rollouts().captures(attempt["replies"], gameplay=gameplay)
    captures = [c for c in captures if c["prompt_tokens"] and (c["usage"] or {}).get("completion_tokens")]
    return [end for end in plan_windows(captures, window) if _mass(captures, end, window) > 0] if captures else []


def attempt_example(encode, attempt, *, window, rng, device, think_end=None, end=None, floor=16384, margin=8192,
                    credited_only=False, gameplay=True, end_request=None):
    """One window of an attempt (drawn uniformly among its windows) as tensors, with the per-token arrays the loss
    needs. ``windows``: how many the attempt has (its weight is multiplied by it)."""
    import torch

    captures = rollouts().captures(attempt["replies"], gameplay=gameplay)
    captures = [c for c in captures if c["prompt_tokens"] and (c["usage"] or {}).get("completion_tokens")]
    ends = plan_windows(captures, window) if captures else []
    credit = any(c["rollout"].get("w") is not None for c in captures)
    if end_request is not None:  # the window that ends at this request's reply (score_check)
        end = next((i for i, c in enumerate(captures) if c["request"] == end_request), None)
        if end is None:
            return None
    # credit per reply: the fullest, heaviest window, ending at any reply (a planned window can be a short remainder)
    if end is None:
        end = heaviest_end(captures, range(len(captures)), window) if credit else (rng.choice(ends) if ends else None)
    made = (window_example(encode, captures, end, window=window, floor=floor, margin=margin, credited_only=credited_only)
            if ends else None)
    if made is None:
        return None
    inputs, tail, spans = made
    ids = inputs["input_ids"][0].tolist()
    positions, reference, support, raw, rows, after_thinking, weights, owner = [], [], [], [], [], [], [], []
    exact = os.environ.get("RL1_REPLY_IDS") == "1"
    for index, start, stop in spans:
        digest = captures[index]["rollout"].get("prompt_digest")
        same = None if not digest else digest == rollouts().ids_digest(ids[:start])
        if exact and same is False:  # 29 Oct: no loss on a reply whose context is not the served one, token for token
            rows.append({"request": captures[index]["request"], "start": start, "stop": stop, "aligned": False,
                         "prompt_same": False, "excluded": True, "outside": 0})
            continue
        ref, sup, engine, aligned = span_arrays(captures[index]["rollout"], ids[start:stop])
        positions += range(start, stop)
        reply = ids[start:stop]
        cut = reply.index(think_end) + 1 if think_end in reply else 0  # no thinking block: all of it is the answer
        after_thinking += [i >= cut for i in range(len(reply))]
        reference, support, raw = reference + ref, support + sup, raw + engine
        weights += [float(captures[index]["rollout"].get("w") or 0.0)] * (stop - start)
        owner += [len(rows)] * (stop - start)
        rows.append({"request": captures[index]["request"], "start": start, "stop": stop, "aligned": aligned,
                     **({"w": captures[index]["rollout"].get("w")} if credit else {}),
                     "prompt_same": same,
                     "outside": sum(v is None for v in (captures[index]["rollout"].get("reference") or []))})
    if not positions:  # every reply left out (RL1_REPLY_IDS: none in its served context)
        return None
    width = max(1, max(len(row) for row in support))
    table = torch.full((len(support), width), -1, dtype=torch.long)
    for i, row in enumerate(support):
        table[i, :len(row)] = torch.tensor(row, dtype=torch.long)
    arrays = {"total": len(ids), "positions": torch.tensor(positions, dtype=torch.long, device=device),
              "support": table.to(device), "reference": torch.tensor(reference, dtype=torch.float32, device=device)}
    if credit:
        arrays["weights"] = torch.tensor(weights, dtype=torch.float32, device=device)
        arrays["reply"] = torch.tensor(owner, dtype=torch.long, device=device)
    return {"inputs": inputs, "tail": tail, "arrays": arrays, "engine_raw": raw, "replies": rows, "windows": len(ends),
            "tokens": len(ids), "targets": len(positions), "after_thinking": after_thinking}
