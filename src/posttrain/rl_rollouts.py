"""RL rollouts (8 October plan, docs/rl-plan.md): what the group runner stores per live reply and what a group of
attempts from one level start is worth.

Record. ``arena.rollout_record`` writes one line of ``attempts/<label>/rollout.jsonl`` per live reply: the served
request as a delta against the attempt's previous one (``keep`` leading messages unchanged, then ``messages``), the
reply, the sampled token ids with the engine's log-probability of each, and per token the engine's top alternatives
(``top``: ``[ids, logprobs]``, highest first). ``captures`` rebuilds the trainer's capture rows from it (the format
of the level bench's fork capture, ``flash_qlora_job.encode_request``).

Actor policy. The actor samples at temperature T from the top-k tokens, cut to the top-p nucleus (vLLM: temperature,
then top-k, then top-p on the renormalised top-k). ``nucleus`` rebuilds that set and its probabilities from RAW top-k
log-probabilities; ``reply_checks`` says whether the engine reported raw or processed values and whether every
sampled token lies inside the rebuilt set.

Reward (plan section 1): no clear 0; a clear ``min(1.15, (b / a)^2) + 0.25 * (1 - tokens / budget)``. Advantage:
reward minus the mean of the other attempts from the same start.
"""

import hashlib
import json
import math
from pathlib import Path

CLAMPED = -9000.0  # vLLM reports -inf (a token the processors removed) as -9999
CREDIT_CAP, BONUS_CAP = 1.15, 0.25
UNUSED = ("censored", "error", "replay_mismatch", "lead_out", "wall")  # attempts that are not an episode


# --- record --------------------------------------------------------------------------------------------------

def ids_digest(ids):
    """A digest of a token-id sequence (the served prompt): equal digests mean the trainer saw the same context
    (the same function as ``arena.rollout_record.ids_digest``, which a bundle ships)."""
    return hashlib.sha256(b"".join(int(t).to_bytes(4, "little") for t in ids)).hexdigest()[:32]


def hint_messages(messages, hint):
    """``messages`` with ``hint`` appended to the system message (OPD teacher; the same function as
    ``arena.restore.hint_messages``)."""
    if not hint:
        return messages
    first = messages[0] if messages else None
    if first and first.get("role") == "system" and isinstance(first.get("content"), str):
        return [{**first, "content": first["content"] + "\n\n" + hint}, *messages[1:]]
    return [{"role": "system", "content": hint}, *messages]


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
    for the ratio. ``source``: a ``rollout.jsonl`` path or its (unrestored) rows."""
    out = []
    for row in (restore(source) if isinstance(source, list) else load(source)):
        if gameplay and not row["gameplay"]:
            continue
        payload = {"messages": row["messages"], "chat_template_kwargs": row["template"]}
        if row["tools"] is not None:
            payload["tools"] = row["tools"]
        out.append({"request": row["n"], "prompt_tokens": (row["usage"] or {}).get("prompt_tokens"), "payload": payload,
                    "reply": row["reply"], "usage": row["usage"],
                    "rollout": {k: row.get(k) for k in ROLLOUT_KEYS}})
    return out


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


def reply_checks(row, slack=1e-3):
    """Counts over one stored reply. ``clamped``: positions where the engine listed a removed token (processed
    log-probabilities) ; ``full``: positions whose listed probabilities sum to 1; ``outside``: sampled tokens
    outside the nucleus rebuilt from raw values (``edge``: of those, within ``slack`` of the cut); ``unlisted``:
    sampled tokens the engine did not list; ``actor``: the sum of the sampled tokens' log-probabilities under the
    rebuilt actor policy, ``raw``: under the engine's reported values."""
    sampling = row["sampling"]
    out = {"tokens": len(row["tokens"]), "positions": 0, "clamped": 0, "full": 0, "outside": 0, "edge": 0, "unlisted": 0,
           "actor": 0.0, "raw": float(sum(row["logprobs"])), "set_size": 0}
    for token, (ids, values) in zip(row["tokens"], row.get("top") or []):
        out["positions"] += 1
        finite = [(t, v) for t, v in zip(ids, values) if v > CLAMPED]
        out["clamped"] += len(finite) < len(ids)
        out["full"] += abs(sum(math.exp(v) for _, v in finite) - 1.0) < slack
        if token not in ids:
            out["unlisted"] += 1
            continue
        kept = dict(nucleus([t for t, _ in finite], [v for _, v in finite], temperature=sampling["temperature"],
                            top_p=sampling["top_p"], top_k=sampling["top_k"]))
        out["set_size"] += len(kept)
        if token in kept:
            out["actor"] += kept[token]
            continue
        out["outside"] += 1
        wider = dict(nucleus([t for t, _ in finite], [v for _, v in finite], temperature=sampling["temperature"],
                             top_p=min(1.0, sampling["top_p"] + slack), top_k=sampling["top_k"]))
        out["edge"] += token in wider
    return out


def reported_mode(checks):
    """``raw`` | ``processed`` | ``unclear``: processed values list removed tokens at almost every position and sum
    to 1 everywhere; raw values never list a removed token."""
    positions = max(1, checks["positions"])
    clamped, full = checks["clamped"] / positions, checks["full"] / positions
    if clamped > 0.5 and full > 0.99:
        return "processed"
    return "raw" if clamped < 0.01 else "unclear"


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


# --- reward and groups ---------------------------------------------------------------------------------------

def reward(cleared, actions, tokens, *, baseline, budget=60000):
    """``{"credit", "bonus", "reward"}`` of one attempt. ``baseline``: the level's frozen action count. The bonus is
    linear in the tokens left (a rate per token with a cap sat at the cap for every quick clear: restart run of
    8 October, 13 of 21 starts without any reward spread)."""
    if not cleared or not actions or not baseline:
        return {"credit": 0.0, "bonus": 0.0, "reward": 0.0}
    credit = min(CREDIT_CAP, (float(baseline) / float(actions)) ** 2)
    bonus = BONUS_CAP * min(1.0, max(0.0, 1.0 - tokens / budget))
    return {"credit": credit, "bonus": bonus, "reward": credit + bonus}


def episodes(rows):
    """Bench rows that are episodes (played to a clear or to the token budget), by start point."""
    out = {}
    for row in rows:
        if row["outcome"] not in UNUSED:
            out.setdefault(row["point"], []).append(row)
    return out


def frozen_baselines(rows):
    """Per start point: the official action count, else the fewest actions among these (base policy) clears. Written
    once, before round 1; later rounds read the file."""
    out = {}
    for point, group in episodes(rows).items():
        official = next((r["baseline_actions"] for r in group if r.get("baseline_actions")), None)
        clears = [r["actions"] for r in group if r["cleared"] and r["actions"]]
        out[point] = {"actions": official or (min(clears) if clears else None),
                      "source": "official" if official else "base attempts" if clears else None}
    return out


def recorded_levels(log_path):
    """A recorded whole-game instance as level episodes: per level played ``{"level", "cleared", "tokens",
    "actions"}`` (gameplay tokens and actions spent on that level; the last one is the level it ended on)."""
    out, level, tokens, start, step = [], 0, 0, 0, 0
    with open(log_path) as file:
        for line in file:
            if line.startswith('{"kind":"observation"'):
                row = json.loads(line)
                step, now = int(row["step"]), int(row["levels"])
                if now > level:
                    out.append({"level": level + 1, "cleared": True, "tokens": tokens, "actions": step - start})
                    level, tokens, start = now, 0, step
            elif line.startswith('{"kind":"model"'):
                row = json.loads(line)
                if row.get("purpose", "gameplay") == "gameplay":
                    tokens += int((row.get("usage") or {}).get("completion_tokens") or 0)
    out.append({"level": level + 1, "cleared": False, "tokens": tokens, "actions": step - start})
    return out


def screen(episodes, *, official=None, budget=60000, least=4, spread=0.1):
    """Is a level start worth sixteen attempts? From its recorded episodes (``recorded_levels`` rows of several
    instances, or level-bench rows): ``{"episodes", "clears", "baseline", "sd", "usable"}``. An episode counts
    when the level was cleared or the whole budget was spent on it; a clear after the budget is a failure here.
    ``usable``: at least ``least`` episodes and a reward standard deviation of at least ``spread``."""
    rows = [e for e in episodes if e["cleared"] or e["tokens"] >= budget]
    wins = [e for e in rows if e["cleared"] and e["tokens"] <= budget and e["actions"]]
    baseline = official or (min(e["actions"] for e in wins) if wins else None)
    values = [reward(e in wins, e["actions"], e["tokens"], baseline=baseline, budget=budget)["reward"] for e in rows]
    mean = sum(values) / len(values) if values else 0.0
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    tokens = [min(e["tokens"], budget) for e in rows]  # what an attempt from this start costs
    return {"episodes": len(rows), "clears": len(wins), "baseline": baseline, "sd": round(sd, 4),
            "mean_reward": round(mean, 4), "mean_tokens": round(sum(tokens) / len(tokens)) if tokens else None,
            "usable": len(rows) >= least and sd >= spread}


def leave_one_out(values):
    """Each value minus the mean of the others (a single value has no advantage)."""
    count, total = len(values), sum(values)
    return [0.0] * count if count < 2 else [value - (total - value) / (count - 1) for value in values]


def groups(rows, baselines, *, budget):
    """Per start point: the attempts with their reward parts and leave-one-out advantage, and the group's totals.
    ``cut``: attempts the run's wall ended (the group is then biased towards quick clears and is not trained on)."""
    out, cut = {}, {}
    for row in rows:
        cut[row["point"]] = cut.get(row["point"], 0) + (row["outcome"] == "censored")
    for point, group in sorted(episodes(rows).items()):
        base = baselines.get(point) or {}
        attempts = []
        for row in group:
            used = row["tokens_to_clear"] if row["cleared"] else row["completion_tokens"]
            attempts.append({"label": row["label"], "cleared": row["cleared"], "actions": row["actions"], "tokens": used,
                             "verified": (row.get("clear_replay") or {}).get("ok") if row["cleared"] else None,
                             **reward(row["cleared"], row["actions"], used, baseline=base.get("actions"),
                                      budget=budget)})
        for attempt, advantage in zip(attempts, leave_one_out([a["reward"] for a in attempts])):
            attempt["advantage"] = advantage
        count = len(attempts)
        mean = sum(a["reward"] for a in attempts) / count
        out[point] = {
            "baseline": base, "attempts": count, "cut": cut[point], "clears": sum(a["cleared"] for a in attempts),
            "unverified_clears": sum(a["cleared"] and a["verified"] is not True for a in attempts),
            "mean_reward": mean, "mean_credit": sum(a["credit"] for a in attempts) / count,
            "mean_bonus": sum(a["bonus"] for a in attempts) / count,
            "reward_sd": math.sqrt(sum((a["reward"] - mean) ** 2 for a in attempts) / max(1, count - 1)),
            "tokens": sum(a["tokens"] for a in attempts), "rows": attempts}
    return out


def pass_probabilities(advantages, passes):
    """Probability that each attempt gets a backward pass: proportional to |advantage|, capped at 1, summing to
    ``passes`` (or to the number of attempts with a non-zero advantage, if that is smaller). An attempt drawn with
    probability p is weighted advantage / p, so the expected update is the sum over all attempts."""
    sizes = [abs(a) for a in advantages]
    capped, scale = set(), 0.0
    while True:  # fewer than ``left`` attempts can pass 1 at a time, so the capped ones never use up the passes
        free = [i for i, s in enumerate(sizes) if s > 0 and i not in capped]
        if not free:
            break
        scale = (passes - len(capped)) / sum(sizes[i] for i in free)
        over = [i for i in free if sizes[i] * scale > 1.0]
        if not over:
            break
        capped.update(over)
    return [1.0 if i in capped else min(1.0, sizes[i] * scale) for i in range(len(sizes))]


def draw(advantages, passes, rng, positive_only=False):
    """Which attempts get a backward pass and with what weight: ``{index: (weight, probability)}``. Systematic
    sampling with the probabilities of ``pass_probabilities``, so exactly ``passes`` are drawn (fewer when fewer
    have an advantage). Weights are advantage / probability, in units of the common size of an uncapped draw:
    +1 or -1 for most, larger for an attempt whose probability is capped at 1."""
    sizes = [max(a, 0.0) if positive_only else a for a in advantages]
    chances = pass_probabilities(sizes, passes)
    unit = next((abs(sizes[i]) / p for i, p in enumerate(chances) if 0 < p < 1), None)
    if unit is None:
        live = [abs(a) for a, p in zip(sizes, chances) if p]
        unit = sum(live) / len(live) if live else 1.0
    order = [i for i, p in enumerate(chances) if p > 0]
    rng.shuffle(order)
    out, start, total = {}, rng.random(), 0.0
    for i in order:
        before, total = total, total + chances[i]
        if max(0, math.ceil(total - start)) > max(0, math.ceil(before - start)):  # a point start + k falls in it
            out[i] = (sizes[i] / chances[i] / unit, chances[i])
    return out


def training_set(folders, *, passes, arm, seed, probes=8, baselines=None, reported=None):
    """The trainer's data for one arm from finished rollout runs: per drawn attempt its weight and its replies
    (compact, in the stored delta format), a fixed probe set of base attempts that is never trained, and the frozen
    baselines. ``arm``: ``signed`` | ``positive``. A clear whose replay failed is not an episode."""
    import random

    rng = random.Random(f"{seed}:{arm}")
    rows, where, budget = [], {}, None
    for folder in map(Path, folders):
        bench = json.loads((folder / "level-bench.json").read_text())
        budget = budget or bench["settings"]["token_budget"]
        for row in bench["attempts"]:
            path = folder / "attempts" / row["label"] / "rollout.jsonl"
            if path.is_file() and not (row["cleared"] and (row.get("clear_replay") or {}).get("ok") is not True):
                rows.append(row)
                where[row["label"]] = path
    baselines = baselines or frozen_baselines(rows)
    found = groups(rows, baselines, budget=budget)
    attempts = [{**a, "point": point} for point, g in found.items() if not g["cut"] for a in g["rows"]]
    held = set(random.Random(f"{seed}:probe").sample(range(len(attempts)), min(probes, len(attempts) // 4)))
    pool = [i for i in range(len(attempts)) if i not in held]
    drawn = draw([attempts[i]["advantage"] for i in pool], passes, rng, positive_only=arm == "positive")
    totals, mode = {}, reported
    if mode is None:
        for i in list(held) + [pool[j] for j in drawn]:
            for reply in load(where[attempts[i]["label"]]):
                for key, value in reply_checks(reply).items():
                    totals[key] = totals.get(key, 0) + value
        mode = reported_mode(totals) if totals.get("positions") else "raw"

    def entry(i, role, weight=0.0, chance=None):
        a = attempts[i]
        with open(where[a["label"]]) as file:
            replies = [compact(json.loads(line), mode) for line in file]
        return {"label": a["label"], "point": a["point"], "role": role, "weight": weight, "probability": chance,
                **{k: a[k] for k in ("advantage", "reward", "credit", "bonus", "cleared", "tokens")}, "replies": replies}

    train = [entry(pool[j], "train", *drawn[j]) for j in drawn]
    rng.shuffle(train)
    return {"arm": arm, "seed": seed, "passes": len(train), "budget": budget, "reported": mode, "baselines": baselines,
            "pool": {"attempts": len(attempts), "starts": sum(not g["cut"] for g in found.values()),
                     "starts_cut_by_the_wall": sum(bool(g["cut"]) for g in found.values()),
                     "with_advantage": sum(abs(attempts[i]["advantage"]) > 1e-9 for i in pool)},
            "attempts": [entry(i, "probe") for i in sorted(held)] + train}


# --- credit per reply (no-think pilot, 9 October) -------------------------------------------------------------

def reply_trace(log_path):
    """One attempt's ``log.jsonl`` as the trainer needs it: per gameplay reply the action count when it was requested
    and a digest of the board it saw, then the attempt's last action count and the count at the clear (None: no
    clear)."""
    boards, steps, last, clear = {}, [], 0, None
    with open(log_path) as file:
        for line in file:
            row = json.loads(line)
            kind = row.get("kind")
            if kind == "observation":
                frames = row.get("frames") or []
                boards[row["step"]] = hashlib.sha256(json.dumps(frames[-1:]).encode()).hexdigest()[:16]
                last = max(last, row["step"])
            elif kind == "level_bench_fork":  # what came before is the replayed prefix, not this attempt's play
                steps = []
            elif kind == "model" and row.get("purpose") == "gameplay":
                steps.append(row["step"])
            elif kind == "level_bench_clear":
                clear = row["step"]
    return {"steps": steps, "boards": [boards.get(step) for step in steps], "last": last, "clear": clear}


def reply_credit(traces, baseline, *, prior=1.0):
    """Per reply of each attempt from one start: ``(return, advantage)``. Return: ``0.5 ** (actions from the reply
    to the clear / baseline)``, 0 without a clear. Advantage: return minus what the start's OTHER attempts made of
    the same position: their mean return from the same board (one value per attempt), mixed with their mean
    return at the same action count (attempts still playing then) with weight ``prior``. A reply nobody else can
    be compared with has advantage 0."""
    def value(trace, step):
        return 0.5 ** ((trace["clear"] - step) / baseline) if trace["clear"] is not None and baseline else 0.0

    seen = {}  # board -> {attempt: [returns]}
    for i, trace in enumerate(traces):
        for step, board in zip(trace["steps"], trace["boards"]):
            if board is not None:
                seen.setdefault(board, {}).setdefault(i, []).append(value(trace, step))
    out = []
    for i, trace in enumerate(traces):
        rows = []
        for step, board in zip(trace["steps"], trace["boards"]):
            mine = value(trace, step)
            same = [sum(v) / len(v) for j, v in (seen.get(board) or {}).items() if j != i]
            end = lambda other: other["clear"] if other["clear"] is not None else other["last"] + 1
            running = [value(other, step) for j, other in enumerate(traces) if j != i and end(other) > step]
            weight, total = len(same), sum(same)
            if running:
                weight, total = weight + prior, total + prior * sum(running) / len(running)
            rows.append((mine, mine - total / weight if weight else 0.0))
        out.append(rows)
    return out


def credit_set(folders, *, passes, seed, baselines=None, held_passes=4, probes=8, cap=3.0, reported=None):
    """The trainer's data with credit per reply. Per start the attempts of the last ``held_passes`` passes are never
    trained (role ``held``: did better replies become more likely than worse ones), ``probes`` more attempts are the
    divergence probe. Training attempts are drawn with probability proportional to their signal (sum over replies
    of |advantage| x tokens); a reply's weight ``w`` is its advantage over that probability, in units of the mean
    drawn signal, the multiplier capped at ``cap``. ``baselines``: frozen action counts per start (the thinking
    pool's); a start without one takes the fewest actions among these clears."""
    import random

    rng = random.Random(f"{seed}:credit")
    rows, where = [], {}
    for folder in map(Path, folders):
        bench = json.loads((folder / "level-bench.json").read_text())
        for row in bench["attempts"]:
            path = folder / "attempts" / row["label"]
            if (path / "rollout.jsonl").is_file() and (path / "log.jsonl").is_file() and not (
                    row["cleared"] and (row.get("clear_replay") or {}).get("ok") is not True):
                rows.append(row)
                where[row["label"]] = path
    own = frozen_baselines(rows)
    baselines = {point: (baselines or {}).get(point) or own.get(point) or {} for point in own}
    cut = {}
    for row in rows:
        cut[row["point"]] = cut.get(row["point"], 0) + (row["outcome"] == "censored")
    attempts, starts = [], {"played": 0, "cut": 0, "without_baseline": 0, "without_clear": 0}
    for point, group in sorted(episodes(rows).items()):
        starts["played"] += 1
        base = (baselines.get(point) or {}).get("actions")
        if cut[point] or not base or not any(r["cleared"] for r in group):
            starts["cut" if cut[point] else "without_baseline" if not base else "without_clear"] += 1
            continue
        traces = [reply_trace(where[r["label"]] / "log.jsonl") for r in group]
        last = max(r["pass"] for r in group)
        for row, trace, credit in zip(group, traces, reply_credit(traces, base)):
            attempts.append({"label": row["label"], "point": point, "cleared": row["cleared"],
                             "actions": row["actions"], "tokens": row["completion_tokens"], "trace": trace,
                             "credit": credit, "held": row["pass"] > last - held_passes})
    mode = reported

    def replies_of(a):
        nonlocal mode
        with open(where[a["label"]] / "rollout.jsonl") as file:
            stored = [json.loads(line) for line in file]
        if mode is None:
            totals = {}
            for reply in restore(stored):
                for key, value in reply_checks(reply).items():
                    totals[key] = totals.get(key, 0) + value
            mode = reported_mode(totals) if totals.get("positions") else "raw"
        return [compact(row, mode) for row in stored]

    def signal(a):  # gameplay replies and credit rows are in the same order
        sizes = a.get("sizes")
        return sum(abs(advantage) * size for (_, advantage), size in zip(a["credit"], sizes))

    for a in attempts:  # tokens per gameplay reply, from the stored lines (no restore needed)
        with open(where[a["label"]] / "rollout.jsonl") as file:
            a["sizes"] = [len(row.get("tokens") or []) for row in map(json.loads, file) if row["gameplay"]]
        a["matched"] = len(a["sizes"]) == len(a["credit"])
    usable = [a for a in attempts if a["matched"]]
    free = [a for a in usable if not a["held"]]
    probe = set(random.Random(f"{seed}:probe").sample(range(len(free)), min(probes, len(free) // 4)))
    pool = [a for i, a in enumerate(free) if i not in probe]
    drawn = draw([signal(a) for a in pool], passes, rng)
    mean = sum(signal(pool[j]) for j in drawn) / max(1, len(drawn))

    def entry(a, role, scale=1.0, chance=None):
        replies, credit = replies_of(a), iter(a["credit"])
        for reply in replies:
            if reply["gameplay"]:
                gain, advantage = next(credit)
                reply["w"], reply["return"] = round(advantage * scale, 6), round(gain, 6)
        return {"label": a["label"], "point": a["point"], "role": role, "weight": scale, "probability": chance,
                "cleared": a["cleared"], "actions": a["actions"], "tokens": a["tokens"],
                "signal": round(signal(a), 3), "boards": a["trace"]["boards"], "replies": replies}

    train = [entry(pool[j], "train", min(cap, weight * mean / signal(pool[j])), chance)
             for j, (weight, chance) in drawn.items()]
    rng.shuffle(train)
    held = [entry(a, "held") for a in usable if a["held"]]
    every = [advantage for a in usable for _, advantage in a["credit"]]
    rate = {}
    for a in attempts:
        rate.setdefault(a["point"], []).append(a["cleared"])
    easy = {point for point, clears in rate.items() if sum(clears) / len(clears) > 0.8}
    return {"arm": "reply-credit", "credit": "reply", "seed": seed, "passes": len(train), "reported": mode,
            "baselines": baselines,
            "pool": {"starts": starts, "attempts": len(attempts), "replies_unmatched": len(attempts) - len(usable),
                     "train_pool": len(pool), "held": len(held), "replies": len(every),
                     "replies_with_advantage": sum(abs(v) > 1e-9 for v in every),
                     "starts_trained": len({a["point"] for a in train}),
                     "starts_cleared_over_80": len(easy),
                     "drawn_from_starts_cleared_over_80": round(sum(a["point"] in easy for a in train) / max(1, len(train)), 3)},
            "attempts": [entry(a, "probe") for i, a in enumerate(free) if i in probe] + train + held}


# --- behaviour gates (docs/rl-plan.md section 4, risk 2) --------------------------------------------------------

def behaviour(folder):
    """What the gates compare between a policy and the base on the same starts, from one finished rollout run:
    replies ended by the completion cap, replies with a tool call, loop-rule firings and actions per 100K tokens,
    the draft head's acceptance, tok/s, and the share of starts with at least one clear."""
    folder = Path(folder)
    bench = json.loads((folder / "level-bench.json").read_text())
    rows = [row for row in bench["attempts"] if row["outcome"] not in UNUSED]
    replies = capped = called = loops = 0
    for row in rows:
        path = folder / "attempts" / row["label"]
        if (path / "rollout.jsonl").is_file():
            for reply in load(path / "rollout.jsonl"):
                replies += 1
                capped += reply.get("finish_reason") == "length"
                called += bool((reply.get("reply") or {}).get("tool_calls"))
        if (path / "log.jsonl").is_file():
            with (path / "log.jsonl").open() as file:
                loops += sum(line.startswith('{"kind":"loop_rule"') for line in file)
    tokens = sum(row["completion_tokens"] for row in rows)
    drafts = {}
    if (folder / "vllm-metrics.txt").is_file():
        for line in (folder / "vllm-metrics.txt").read_text().splitlines():
            for key in ("spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total"):
                if line.startswith("vllm:" + key + "{"):
                    drafts[key] = drafts.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    drafted = drafts.get("spec_decode_num_draft_tokens_total")
    starts = episodes(rows)
    return {"attempts": len(rows), "replies": replies, "tokens": tokens,
            "replies_at_cap": round(capped / max(1, replies), 5),
            "replies_with_tool_call": round(called / max(1, replies), 5),
            "loop_rule_per_100k_tokens": round(loops / max(1, tokens) * 1e5, 4),
            "actions_per_100k_tokens": round(sum(row["actions"] or 0 for row in rows) / max(1, tokens) * 1e5, 2),
            "draft_acceptance": round(drafts["spec_decode_num_accepted_tokens_total"] / drafted, 4) if drafted else None,
            "tokens_per_second": bench["summary"]["tokens_per_second"],
            "starts_with_a_clear": round(sum(any(r["cleared"] for r in group) for group in starts.values())
                                         / max(1, len(starts)), 4)}


# --- report --------------------------------------------------------------------------------------------------

def report(folder, *, budget=None):
    """A finished rollout run (the trial's output folder): groups with rewards, the log-probability checks summed
    over every stored reply, and what the record cost in bytes."""
    folder = Path(folder)
    bench = json.loads((folder / "level-bench.json").read_text())
    rows = bench["attempts"]
    budget = budget or bench["settings"]["token_budget"]
    baselines = frozen_baselines(rows)
    found = groups(rows, baselines, budget=budget)
    totals, bytes_, replies, mismatched, digests = {}, 0, 0, 0, 0
    for row in rows:
        path = folder / "attempts" / row["label"] / "rollout.jsonl"
        if not path.is_file():
            continue
        bytes_ += path.stat().st_size
        for reply in load(path):
            replies += 1
            mismatched += len(reply["tokens"]) != (reply["usage"] or {}).get("completion_tokens")
            digests += bool(reply.get("prompt_digest"))
            for key, value in reply_checks(reply).items():
                totals[key] = totals.get(key, 0) + value
    positions = max(1, totals.get("positions", 0))
    return {
        "attempts": len(rows), "episodes": sum(g["attempts"] for g in found.values()),
        "outcomes": bench["summary"]["outcomes"], "tokens_per_second": bench["summary"]["tokens_per_second"],
        "budget": budget,
        "groups": {point: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in g.items() if k != "rows"}
                   for point, g in found.items()},
        "logprobs": {
            "replies": replies, "tokens": totals.get("tokens", 0), "positions_with_alternatives": totals.get("positions", 0),
            "replies_with_token_count_mismatch": mismatched, "replies_with_prompt_digest": digests,
            "reported": reported_mode(totals) if totals.get("positions") else None,
            "clamped_share": round(totals.get("clamped", 0) / positions, 5),
            "full_mass_share": round(totals.get("full", 0) / positions, 5),
            "sampled_outside_nucleus": totals.get("outside", 0), "of_those_at_the_cut": totals.get("edge", 0),
            "sampled_unlisted": totals.get("unlisted", 0),
            "mean_nucleus_size": round(totals.get("set_size", 0) / positions, 3),
            "mean_raw_logprob": round(totals.get("raw", 0.0) / max(1, totals.get("tokens", 0)), 5),
            "mean_actor_logprob": round(totals.get("actor", 0.0) / positions, 5),
            "record_bytes": bytes_, "record_bytes_per_token": round(bytes_ / max(1, totals.get("tokens", 0)), 1)},
        "rollout": bench.get("rollout")}
