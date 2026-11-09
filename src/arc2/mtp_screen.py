"""MTP exactness screen: does speculative decoding with the MTP draft sample from the same
distribution as plain decoding at play settings (temperature 0.7, top-k 20, top-p 0.95)?

One server per run (``server.mtp_tokens`` 0, 1 or 3). First turns of ARC tasks (text, the tool system prompt, effort
from the study) are rendered by the server's /tokenize and sampled through /v1/completions as token ids, streamed,
with the top-20 raw log-probabilities of every emitted token. For every token the play sampling distribution q is
rebuilt from those raw log-probs exactly as vLLM's sampler builds it (logits / T, the 20 largest, then top-p over
them: ascending cumulative mass <= 1 - p dropped, the largest always kept), and the residual

    r = log q(x) + H(q)

is recorded. Under exact sampling E[r | everything before x] = 0, so the mean of r over any set of tokens chosen by
their prefix only is 0 (the tokens are martingale differences: the token-level standard error is honest). A token
outside q's support cannot come from exact sampling at all (``outside``).

Positions in the draft span: a streamed chunk is one engine step for the request (1 + accepted drafts tokens), so
the i-th token of a chunk is ``offset`` i (1-based); with k drafts offset k + 1 is the bonus token (all k accepted).
Whether offset i exists depends on earlier tokens only, so E[r | offset i] = 0 too. (Accepted vs resampled is a
function of x itself and is reported as counts only.) The chunk sizes are checked against the engine's own
per-position acceptance counters: if the server merged steps for a slow reader the counts disagree.

A subset is re-scored with ``prompt_logprobs`` 20 (a prefill of prompt + reply): the generation-time log-probs must
match an independent forward pass up to batch-shape numerics.
"""

import asyncio
import json
import math
import random
import re
import time
from pathlib import Path

import httpx

from . import tasks

TEMPERATURE, TOP_K, TOP_P = 0.7, 20, 0.95
ID = re.compile(r"^token_id:(\d+)$")
METRICS = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
           "vllm:spec_decode_num_accepted_tokens_total", "vllm:generation_tokens_total")
PER_POSITION = "vllm:spec_decode_num_accepted_tokens_per_pos_total"


def token_id(text):
    found = ID.match(str(text))
    if not found:
        raise ValueError(f"not a token id string: {text!r} (return_tokens_as_token_ids)")
    return int(found.group(1))


def play_distribution(top, temperature=TEMPERATURE, k=TOP_K, p=TOP_P):
    """``top``: [(token id, raw log-prob)] (at least the k largest). Returns {token id: log q} over q's support, as
    vLLM's sampler: temperature, top-k, then top-p on the top-k renormalised probabilities."""
    items = sorted(top, key=lambda x: -x[1])[:k]
    scaled = [lp / temperature for _, lp in items]
    peak = max(scaled)
    weights = [math.exp(v - peak) for v in scaled]
    total = sum(weights)
    probs = [w / total for w in weights]
    order = sorted(range(len(items)), key=lambda i: probs[i])  # ascending, as vLLM sorts
    kept, mass = [], 0.0
    for rank, i in enumerate(order):
        mass += probs[i]
        if mass > 1 - p or rank == len(order) - 1:
            kept.append(i)
    norm = sum(probs[i] for i in kept)
    return {items[i][0]: math.log(probs[i] / norm) for i in kept}


def residual(token, top, **kw):
    """(r, outside): r = log q(x) + H(q) over the play distribution; outside when x is not in its support."""
    q = play_distribution(top, **kw)
    entropy = -sum(math.exp(v) * v for v in q.values())
    if token not in q:
        return None, True
    return q[token] + entropy, False


def label_chunks(chunks, drafts, finished_by_length):
    """Offsets for a request's streamed chunks: [(offset, kind)] per token. The first chunk's first token comes from
    the prefill ("first"); a chunk of n tokens is offsets 1..n; with drafts k, offset k + 1 is the bonus, a shorter
    chunk's last token the resampled one ("resampled"), the others "accepted". The final chunk is "end" (a stop or
    the length cap can cut a step short)."""
    out = []
    for c, chunk in enumerate(chunks):
        n = len(chunk)
        for i in range(n):
            if c == 0 and i == 0:
                out.append((0, "first"))
                continue
            offset = i + 1 if c > 0 else i  # the prefill's token opens the first chunk
            if c == len(chunks) - 1:
                kind = "end"
            elif drafts == 0:
                kind = "plain"
            elif offset == drafts + 1:
                kind = "bonus"
            elif i == n - 1:
                kind = "resampled"
            else:
                kind = "accepted"
            out.append((offset, kind))
    return out


def summarize(rows):
    """rows: [(offset, kind, r or None, outside)] -> per offset and overall: tokens, mean r, standard error, outside."""
    groups = {}
    for offset, kind, r, outside in rows:
        for key in ("all" if kind not in ("end",) else None, f"offset{offset}" if kind not in ("end",) else None,
                    f"kind:{kind}"):
            if key is None:
                continue
            g = groups.setdefault(key, {"tokens": 0, "outside": 0, "sum": 0.0, "sq": 0.0, "n": 0})
            g["tokens"] += 1
            if outside:
                g["outside"] += 1
                continue
            g["n"] += 1
            g["sum"] += r
            g["sq"] += r * r
    out = {}
    for key, g in sorted(groups.items()):
        n = g["n"]
        mean = g["sum"] / n if n else None
        var = (g["sq"] / n - mean * mean) * n / (n - 1) if n > 1 else None
        out[key] = {"tokens": g["tokens"], "outside": g["outside"], "mean_r": None if mean is None else round(mean, 5),
                    "se": None if var is None else round(math.sqrt(max(var, 0.0) / n), 5)}
    return out


async def metrics(client):
    text = (await client.get(str(client.base_url.copy_with(path="/metrics")), timeout=30)).text
    found = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{")[0].split(" ")[0]
        if name in METRICS or name == PER_POSITION:
            value = float(line.rsplit(" ", 1)[1])
            key = name
            position = re.search(r'position="(\d+)"', line)
            if position:
                key = f"{name}:{position.group(1)}"
            found[key] = found.get(key, 0.0) + value
    return found


async def sample(client, model, prompt_ids, max_tokens):
    payload = {"model": model, "prompt": prompt_ids, "max_tokens": int(max_tokens), "stream": True,
               "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "logprobs": TOP_K,
               "return_tokens_as_token_ids": True, "skip_special_tokens": False}
    chunks, finish = [], None
    async with client.stream("POST", "completions", json=payload, timeout=3600) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            for choice in json.loads(data).get("choices") or []:
                lp = choice.get("logprobs") or {}
                toks = lp.get("tokens") or []
                if toks:
                    chunks.append([(token_id(t), float(v), [(token_id(a), float(b)) for a, b in (top or {}).items()])
                                   for t, v, top in zip(toks, lp.get("token_logprobs") or [], lp.get("top_logprobs") or [])])
                finish = choice.get("finish_reason") or finish
    return chunks, finish


async def rescore(client, model, prompt_ids, reply_ids):
    """Independent prefill of prompt + reply: the top-20 raw log-probs at every reply position."""
    payload = {"model": model, "prompt": list(prompt_ids) + list(reply_ids), "max_tokens": 1, "temperature": 0,
               "prompt_logprobs": TOP_K, "return_tokens_as_token_ids": True}
    response = await client.post("completions", json=payload, timeout=1800)
    response.raise_for_status()
    rows = response.json()["choices"][0].get("prompt_logprobs") or []
    out = []
    for row in rows[len(prompt_ids):]:
        out.append([(int(k), float(v["logprob"])) for k, v in (row or {}).items()])
    return out


async def run_mtp_screen(study, trial, *, source, output, base_url, deadline_epoch):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    drafts = int((study.get("server") or {}).get("mtp_tokens") or 0)
    spec = study["arc2"]
    ids = list(trial["tasks"])[:int(trial.get("prompts", 160))]
    task_map = tasks.load(Path(source) / spec["data"] / trial["split"], ids)
    effort = trial.get("effort") or "medium"
    max_tokens = int(trial.get("max_tokens", 8192))
    concurrency = int(trial.get("concurrency", 32))
    rescore_n = int(trial.get("rescore", 16))
    report = {"drafts": drafts, "prompts": len(ids), "effort": effort, "max_tokens": max_tokens,
              "concurrency": concurrency, "settings": {"temperature": TEMPERATURE, "top_k": TOP_K, "top_p": TOP_P},
              "status": "running", "errors": []}
    progress = output / "progress.json"
    rows, chunk_sizes, done, tokens = [], {}, 0, 0
    began = time.time()
    samples_file = (output / "samples.jsonl").open("w")
    async with httpx.AsyncClient(base_url=base_url.rstrip("/") + "/",
                                 limits=httpx.Limits(max_connections=concurrency + 8)) as client:
        model = (await client.get("models", timeout=60)).json()["data"][0]["id"]
        before = await metrics(client)
        gate = asyncio.Semaphore(concurrency)
        kept = {}

        async def one(task_id):
            nonlocal done, tokens
            messages = [{"role": "system", "content": tasks.SYSTEM_TOOL},
                        {"role": "user", "content": tasks.task_prompt(task_map[task_id])}]
            async with gate:
                if time.time() > deadline_epoch - 300:
                    return
                try:
                    rendered = await client.post(str(client.base_url.copy_with(path="/tokenize")), timeout=120, json={
                        "model": model, "messages": messages, "tools": [tasks.PYTHON_TOOL],
                        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": effort},
                        "add_generation_prompt": True})
                    rendered.raise_for_status()
                    prompt_ids = rendered.json()["tokens"]
                    chunks, finish = await sample(client, model, prompt_ids, max_tokens)
                except Exception as exc:  # noqa: BLE001 - recorded; the screen goes on
                    report["errors"].append(f"{task_id}: {type(exc).__name__}: {str(exc)[:300]}")
                    return
            labels = label_chunks(chunks, drafts, finish == "length")
            flat = [tok for chunk in chunks for tok in chunk]
            mine = []
            for (offset, kind), (tok, _, top) in zip(labels, flat):
                r, outside = residual(tok, top)
                mine.append((offset, kind, r, outside))
            rows.extend(mine)
            for chunk in chunks[1:-1]:
                chunk_sizes[len(chunk)] = chunk_sizes.get(len(chunk), 0) + 1
            done += 1
            tokens += len(flat)
            kept[task_id] = (prompt_ids, [t for t, _, _ in flat], [top for _, _, top in flat])
            samples_file.write(json.dumps({"task": task_id, "finish": finish, "chunks": [len(c) for c in chunks],
                                           "tokens": [t for t, _, _ in flat],
                                           "r": [None if r is None else round(r, 5) for _, _, r, _ in mine],
                                           "outside": [i for i, m in enumerate(mine) if m[3]]}) + "\n")
            samples_file.flush()

        async def tick():
            while True:
                await asyncio.sleep(30)
                progress.write_text(json.dumps({"t": round(time.time() - began), "done": done, "of": len(ids),
                                                "tokens": tokens, "errors": len(report["errors"]),
                                                "tokens_per_second": round(tokens / max(1.0, time.time() - began))}))

        ticker = asyncio.create_task(tick())
        try:
            await asyncio.gather(*(one(t) for t in ids))
        finally:
            ticker.cancel()
        after = await metrics(client)
        report["engine"] = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in set(after) | set(before)}
        report["sampling_seconds"] = round(time.time() - began)
        # the chunk sizes against the engine's counters: accepted tokens at position j = chunks longer than j
        if drafts:
            derived = {j: sum(n for size, n in chunk_sizes.items() if size > j + 1) for j in range(drafts)}
            report["accepted_per_position"] = {"chunks": derived, "engine": {
                j: report["engine"].get(f"{PER_POSITION}:{j}") for j in range(drafts)}}
        report["chunk_sizes"] = {str(k): v for k, v in sorted(chunk_sizes.items())}
        # an independent forward pass over a subset
        check, picked = [], sorted(kept)[:rescore_n]
        for task_id in picked:
            prompt_ids, reply_ids, gen_tops = kept[task_id]
            if time.time() > deadline_epoch - 240:
                break
            try:
                scored = await rescore(client, model, prompt_ids, reply_ids)
            except Exception as exc:  # noqa: BLE001
                report["errors"].append(f"rescore {task_id}: {type(exc).__name__}: {str(exc)[:300]}")
                continue
            for tok, gen_top, top in zip(reply_ids, gen_tops, scored):
                g, s = dict(gen_top), dict(top)
                r, outside = residual(tok, top)
                check.append((g.get(tok), s.get(tok), r, outside))
        if check:
            both = [(g, s) for g, s, _, _ in check if g is not None and s is not None]
            rs = [r for _, _, r, o in check if not o]
            report["rescore"] = {"tokens": len(check), "outside": sum(1 for c in check if c[3]),
                                 "mean_abs_logprob_diff": round(sum(abs(g - s) for g, s in both) / max(1, len(both)), 5),
                                 "mean_logprob_diff": round(sum(g - s for g, s in both) / max(1, len(both)), 5),
                                 "mean_r": round(sum(rs) / max(1, len(rs)), 5)}
    samples_file.close()
    report["summary"] = summarize(rows)
    report["samples"], report["tokens"] = done, tokens
    report["status"] = "complete" if done else "error"
    (output / "mtp-screen.json").write_text(json.dumps(report, indent=1))
    return report


def simulate(drafts, steps=20000, vocab=40, seed=0, bug=None):
    """CPU check of the method: random target distributions, a greedy draft that is right ``agree`` of the time,
    vLLM's rejection rule (accept the draft with q(draft); else resample from q with the draft removed). ``bug``
    "untruncated": the resample uses the untruncated tempered distribution; "accept_untruncated": acceptance uses it
    (the classic failures: truncation applied on one side of the rejection rule only). Returns rows as
    run_mtp_screen records them."""
    rng = random.Random(seed)
    rows = []

    def target():
        return [(t, rng.gauss(0, 2.5)) for t in range(vocab)]

    def draw(dist):
        u, c = rng.random(), 0.0
        for tok, lq in dist.items():
            c += math.exp(lq)
            if u <= c:
                return tok
        return tok

    def tempered(top):
        scaled = {t: lp / TEMPERATURE for t, lp in top}
        m = max(scaled.values())
        z = sum(math.exp(v - m) for v in scaled.values())
        return {t: v - m - math.log(z) for t, v in scaled.items()}

    def top20(top):
        raw = sorted(top, key=lambda x: -x[1])
        z = math.log(sum(math.exp(v) for _, v in raw))
        return [(t, v - z) for t, v in raw][:TOP_K]

    produced = 0
    while produced < steps:
        offset = 0
        for j in range(drafts + 1):
            top = target()
            q = play_distribution(top)
            if j == drafts:  # bonus (or plain decoding when drafts == 0)
                x = draw(q)
                offset += 1
                r, outside = residual(x, top20(top))
                rows.append((offset, "bonus" if drafts else "plain", r, outside))
                produced += 1
                break
            guess = max(q, key=q.get) if rng.random() < 0.7 else rng.randrange(vocab)
            if bug == "accept_untruncated":  # acceptance against the untruncated tempered probability
                accept = rng.random() < math.exp(tempered(top)[guess])
            else:
                accept = guess in q and rng.random() < math.exp(q[guess])
            offset += 1
            if accept:
                r, outside = residual(guess, top20(top))
                rows.append((offset, "accepted", r, outside))
                produced += 1
                continue
            base = tempered(top) if bug == "untruncated" else q
            rest = {t: lq for t, lq in base.items() if t != guess} or q  # nothing left: fall back to q
            z = math.log(sum(math.exp(v) for v in rest.values()))
            x = draw({t: v - z for t, v in rest.items()})
            r, outside = residual(x, top20(top))
            rows.append((offset, "resampled", r, outside))
            produced += 1
            break
    return rows
