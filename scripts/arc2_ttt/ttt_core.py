"""Per-task LoRA test-time training and its screen read, model-agnostic: the B300 job
(``modal_ttt_screen.py``) runs it on Flash with the competition weights, the CPU check on a tiny random model.

``encode``: chat-template tokens of a sample (``arc2.ttt_data``) with labels on the answer grid only (thinking off).
``reset``: a fresh adapter (lora_A as PEFT initialises it, lora_B zero), so every task starts from the base model.
``train``: AdamW over the task's samples for ``epochs`` (shuffled per epoch, one sample a step, warm-up then cosine).
``reply``: greedy generation, the direct-reply check.
``read``: per query, mean NLL of the answer tokens and whether every answer token is the argmax under teacher
forcing (= greedy decoding reproduces the grid exactly: a lower bound of a direct reply's pass@2)."""
import math
import random


def encode(tokenizer, messages, target, max_tokens=None):
    """(input_ids, labels): the prompt (thinking off) then ``target`` and the end of turn; labels -100 on the prompt."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": target}], tokenize=False,
                                         enable_thinking=False)
    if not full.startswith(prompt):
        raise ValueError("chat template: the prompt is not a prefix of the prompt + answer")
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if ids[:len(p)] != p:
        raise ValueError("tokenization of the prompt changes once the answer follows it")
    if max_tokens and len(ids) > max_tokens:
        return None
    return ids, [-100] * len(p) + ids[len(p):]


def lora_modules(model):
    return [m for m in model.modules() if hasattr(m, "lora_A") and hasattr(m, "lora_B")]


def reset(model, seed=0):
    """Every adapter back to PEFT's initial state: lora_A re-drawn (kaiming uniform, seeded), lora_B zero."""
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    for m in lora_modules(model):
        for name, a in m.lora_A.items():
            w = a.weight
            bound = 1.0 / math.sqrt(w.shape[1])  # kaiming_uniform_(a=sqrt(5)) on a [r, in] matrix
            w.data.copy_((torch.rand(w.shape, generator=gen) * 2 - 1).mul_(bound).to(w.dtype))
            m.lora_B[name].weight.data.zero_()


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def autocast(model):
    """BF16 autocast on the model's device, as the RL trainer's forwards: Flash keeps its routers in FP32 (screen r2
    died on F.linear(bf16, fp32) without it)."""
    import torch

    device = next(p for p in model.parameters()).device
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def answer_logits(model, ids, labels):
    """(logits [n, vocab] float32, targets [n]) at the positions that predict the labelled tokens only; the model
    keeps those positions alone (``logits_to_keep``) when it supports it, else the full logits are sliced."""
    import torch

    device = next(p for p in model.parameters()).device
    pos = [i for i in range(len(ids) - 1) if labels[i + 1] != -100]
    x = torch.tensor([ids], device=device)
    keep = torch.tensor(pos, device=device)
    targets = torch.tensor([labels[i + 1] for i in pos], device=device)
    with autocast(model):
        try:
            logits = model(input_ids=x, logits_to_keep=keep).logits[0]
            if logits.shape[0] != len(pos):
                raise TypeError(f"logits_to_keep gave {logits.shape[0]} rows for {len(pos)}")
        except TypeError:
            logits = model(input_ids=x).logits[0, keep]
    return logits.float(), targets


def loss_of(model, ids, labels):
    import torch

    logits, targets = answer_logits(model, ids, labels)
    return torch.nn.functional.cross_entropy(logits, targets)


def batch_loss(model, batch, pad_id):
    """Token-mean cross-entropy of a padded batch of (ids, labels) samples (right padding, labels -100 on pads: the
    causal layers never see a pad before a real token). Logits only at the labelled positions, from the last hidden
    state (full logits of a packed batch would not fit); returns (loss, real tokens, padded tokens)."""
    import torch

    device = next(p for p in model.parameters()).device
    width = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), width), int(pad_id), dtype=torch.long)
    rows, cols, targets = [], [], []
    for b, (ids, labels) in enumerate(batch):
        x[b, :len(ids)] = torch.tensor(ids)
        for i in range(len(ids) - 1):
            if labels[i + 1] != -100:
                rows.append(b)
                cols.append(i)
                targets.append(labels[i + 1])
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    with autocast(model):
        hidden = base.model(input_ids=x.to(device)).last_hidden_state
        picked = hidden[torch.tensor(rows, device=device), torch.tensor(cols, device=device)]
        logits = base.lm_head(picked)
    loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor(targets, device=device))
    return loss, sum(len(ids) for ids, _ in batch), len(batch) * width


def batches(encoded, order, batch_size):
    return [[encoded[i] for i in order[k:k + batch_size]] for k in range(0, len(order), batch_size)]


def train(model, encoded, *, epochs=2, lr=1e-4, warmup=0.1, seed=0, clip=1.0, log=None, batch_size=1, pad_id=0):
    """AdamW on the adapter; batch_size 1 = one sample a step (r5), else padded batches of ``batch_size`` samples
    (``batch_loss``). Returns the per-step losses."""
    import torch

    params = trainable(model)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    plan = []
    for _ in range(epochs):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        plan += batches(encoded, order, batch_size)
    steps = len(plan)
    warm = max(1, int(warmup * steps))
    losses = []
    model.train()
    for step, batch in enumerate(plan):
        scale = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for g in opt.param_groups:
            g["lr"] = lr * scale
        loss = loss_of(model, *batch[0]) if batch_size == 1 else batch_loss(model, batch, pad_id)[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
        if log:
            log(step=step, loss=round(float(loss), 4))
    model.eval()
    return losses


def bench(model, encoded, sizes, pad_id, steps=2):
    """Forward + backward (no optimizer step: the adapter is unchanged) of ``steps`` padded batches per size after
    one untimed batch; real and padded tokens per second and the peak GPU memory (GiB, CUDA only)."""
    import time

    import torch

    cuda = next(p for p in model.parameters()).device.type == "cuda"
    out = {}
    model.train()
    for size in sizes:
        group = batches(encoded, list(range(len(encoded))) * (1 + size * (steps + 1) // max(1, len(encoded))),
                        size)[:steps + 1]
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        real = padded = 0
        began = None
        for k, batch in enumerate(group):
            if k == 1:
                if cuda:
                    torch.cuda.synchronize()
                began = time.time()
            loss, r, p = batch_loss(model, batch, pad_id)
            loss.backward()
            model.zero_grad(set_to_none=True)
            if k >= 1:
                real, padded = real + r, padded + p
        if cuda:
            torch.cuda.synchronize()
        secs = time.time() - began
        out[size] = {"real_tok_s": round(real / secs, 1), "padded_tok_s": round(padded / secs, 1),
                     "seconds": round(secs, 2), "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
                     if cuda else None}
    model.eval()
    return out


def read(model, encoded_queries, detail=False):
    """Per query {"nll": mean NLL of the answer tokens, "exact": every answer token is the argmax, "tokens": n}
    (+ "logprobs" per token with ``detail``)."""
    import torch

    out = []
    model.eval()
    with torch.no_grad():
        for ids, labels in encoded_queries:
            logits, y = answer_logits(model, ids, labels)
            lp = torch.log_softmax(logits, -1).gather(1, y[:, None])[:, 0]
            row = {"nll": round(float(-lp.mean()), 5), "exact": bool((logits.argmax(-1) == y).all()),
                   "tokens": int(y.numel())}
            if detail:
                row["logprobs"] = [round(float(v), 6) for v in lp]
            out.append(row)
    return out


def reply(model, ids, labels, extra=8):
    """Greedy reply to the prompt part of (ids, labels) with up to answer length + ``extra`` new tokens: the token ids
    and whether they reproduce the labelled answer through its end-of-turn token."""
    import torch

    start = next(i for i, l in enumerate(labels) if l != -100)
    want = ids[start:]
    device = next(p for p in model.parameters()).device
    model.eval()
    with torch.no_grad(), autocast(model):
        out = model.generate(input_ids=torch.tensor([ids[:start]], device=device), max_new_tokens=len(want) + extra,
                             do_sample=False, use_cache=True)
    got = out[0, start:].tolist()
    common = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    # generation stops at the end-of-turn token, before the template's newline after it
    return {"tokens": got, "common": common, "want": len(want), "matches": common >= len(want) - 1}
