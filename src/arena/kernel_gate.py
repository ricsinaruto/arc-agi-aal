"""Kernel quality gate (kernel work, phase 2; docs/validation-20260927/kernels-phase2.md).

A ``kernel-quality`` trial measures one running server on fixed inputs (the bundle's
``configs/kernel-quality``):

- **greedy**: arena prompts (recorded Duck model-call snapshots), temperature 0, 256 tokens, top-20
  logprobs per generated token;
- **teacher-forced**: recorded model reasoning (~512 tokens each), ``prompt_logprobs`` = 20 at every
  position, i.e. the next-token distribution under identical context;
- **GSM8K**: a fixed subset, the checkpoint card's sampling (t 0.6, top-p 0.95) with a per-problem seed;
- a short throughput grid and one torch-profiled decode point (``arena.throughput_bench``).

The ``reference`` role (control server) also repeats greedy, teacher-forced and GSM8K at another
concurrency: that is the control's own run-to-run noise (batch composition changes cuBLAS kernels). The
``gate`` role (patched server) compares itself with the reference trial's file and fails the trial when a
threshold in ``THRESHOLDS`` (fixed before the run) is missed; the bootstrap then skips any trial that
``requires`` the gate.

KL is computed on the coarse-graining {tokens in both top-20 lists} + {rest}, which is exact for that
partition and a lower bound on the full-vocabulary KL.
"""

import asyncio
import json
import math
import re
import time
from pathlib import Path

import httpx

from arc_infra.model_templates import template_kwargs
from arc_infra.serving import MODEL

from .isolation import isolate_trial

INPUTS = "configs/kernel-quality"
# Pass thresholds for the patched server against the control reference (stated before the run).
THRESHOLDS = {
    "tf_mean_kl_nats_max": 0.01,       # mean per-token KL(control || candidate), teacher-forced
    "tf_top1_agreement_min": 0.97,     # teacher-forced argmax agreement
    "gsm8k_max_drop_points": 2.0,      # candidate accuracy >= control - 2.0 points on the same problems/seeds
    "gsm8k_min_scored": 150,           # problems answered by both servers before the pass deadline
}
GREEDY = {"max_tokens": 256, "top_logprobs": 20, "concurrency": 16, "repeat_concurrency": 1}
TEACHER = {"prompt_logprobs": 20, "concurrency": 1, "repeat_concurrency": 4}
GSM8K = {"max_tokens": 4096, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "reasoning_effort": "low",
         "concurrency": 32, "repeat_concurrency": 24, "pass_seconds": 480,
         "suffix": "\nPlease reason step by step, and put your final answer within \\boxed{}."}


def load_inputs(source):
    root = Path(source) / INPUTS
    prompts = json.loads((root / "prompts.json").read_text())
    problems = [json.loads(line) for line in (root / "gsm8k.jsonl").read_text().splitlines() if line.strip()]
    return prompts, problems


# ----------------------------------------------------------------------------- parsing and metrics
def token_id(key):
    """vLLM token keys: "token_id:123" (return_tokens_as_token_ids) or a bare id string."""
    text = str(key)
    return int(text.split(":", 1)[1]) if text.startswith("token_id:") else int(text)


def kl_coarse(p, q):
    """KL(p || q) on the partition {tokens known in both} + {rest}; p, q map token -> logprob."""
    common = p.keys() & q.keys()
    if not common:
        return None
    kl = sum(math.exp(p[t]) * (p[t] - q[t]) for t in common)
    rest_p = max(1.0 - sum(math.exp(p[t]) for t in common), 1e-12)
    rest_q = max(1.0 - sum(math.exp(q[t]) for t in common), 1e-12)
    return max(kl + rest_p * math.log(rest_p / rest_q), 0.0)


def top1(dist):
    return max(dist.items(), key=lambda kv: kv[1])[0] if dist else None


def compare_positions(pairs):
    """Aggregate KL, argmax agreement and chosen-token |delta logprob| over (p, q, chosen) positions."""
    kls, agree, deltas = [], 0, []
    for p, q, chosen in pairs:
        kl = kl_coarse(p, q)
        if kl is not None:
            kls.append(kl)
        agree += top1(p) == top1(q)
        if chosen is not None and chosen in p and chosen in q:
            deltas.append(abs(p[chosen] - q[chosen]))
    kls.sort()
    count = len(pairs)
    return {
        "positions": count,
        "mean_kl": sum(kls) / len(kls) if kls else None,
        "p50_kl": kls[len(kls) // 2] if kls else None,
        "p99_kl": kls[min(len(kls) - 1, int(0.99 * len(kls)))] if kls else None,
        "max_kl": kls[-1] if kls else None,
        "top1_agreement": agree / count if count else None,
        "mean_abs_delta_logprob": sum(deltas) / len(deltas) if deltas else None,
    }


def compare_teacher(reference, candidate):
    pairs = []
    by_index = {row["index"]: row for row in candidate.get("rows", []) if "positions" in row}
    for row in reference.get("rows", []):
        other = by_index.get(row["index"])
        if "positions" not in row or other is None:
            continue
        for mine, theirs, chosen in zip(row["positions"], other["positions"], row["tokens"][1:]):
            pairs.append((to_dist(mine), to_dist(theirs), chosen))
    return compare_positions(pairs)


def to_dist(entries):
    return {int(t): float(lp) for t, lp in entries}


def compare_greedy(reference, candidate):
    by_index = {row["index"]: row for row in candidate.get("rows", []) if "tokens" in row}
    exact, divergence, pairs, compared = 0, [], [], 0
    for row in reference.get("rows", []):
        other = by_index.get(row["index"])
        if "tokens" not in row or other is None:
            continue
        compared += 1
        a, b = row["tokens"], other["tokens"]
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        exact += a == b
        divergence.append(first)
        # Identical context up to and including the first divergent position.
        for i in range(min(first + 1, len(a), len(b))):
            pairs.append((to_dist(row["top"][i]), to_dist(other["top"][i]), a[i]))
    result = compare_positions(pairs)
    result.update(prompts=compared, exact_match=exact / compared if compared else None,
                  mean_first_divergence=sum(divergence) / len(divergence) if divergence else None,
                  first_divergence=divergence)
    return result


BOXED = re.compile(r"\\boxed\s*\{")
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def boxed(text):
    start = None
    for match in BOXED.finditer(text):
        start = match.end()
    if start is None:
        return None
    depth, i = 1, start
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[start:i - 1] if depth == 0 else None


def numeric(text):
    if text is None:
        return None
    cleaned = re.sub(r"\\text\{[^}]*\}|\\,|\\!|\$|\\%|%", "", text).replace(",", "")
    found = NUMBER.findall(cleaned)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def extract_answer(content, reasoning=""):
    for text in (content or "", reasoning or ""):
        value = numeric(boxed(text))
        if value is not None:
            return value
    return numeric(content or "")


def gold_answer(answer):
    return float(answer.split("####")[-1].strip().replace(",", ""))


def compare_gsm8k(reference, candidate):
    mine = {row["index"]: row for row in reference.get("rows", []) if "correct" in row}
    theirs = {row["index"]: row for row in candidate.get("rows", []) if "correct" in row}
    both = sorted(mine.keys() & theirs.keys())
    if not both:
        return {"scored": 0}
    ref_acc = sum(mine[i]["correct"] for i in both) / len(both)
    cand_acc = sum(theirs[i]["correct"] for i in both) / len(both)
    return {"scored": len(both), "reference_accuracy": ref_acc, "candidate_accuracy": cand_acc,
            "delta_points": 100 * (cand_acc - ref_acc),
            "reference_only_correct": sum(mine[i]["correct"] and not theirs[i]["correct"] for i in both),
            "candidate_only_correct": sum(theirs[i]["correct"] and not mine[i]["correct"] for i in both)}


def verdict(reference, candidate, thresholds=THRESHOLDS):
    """Compare the gate's measurements with the reference's; ``passed`` requires every check."""
    tf = compare_teacher(reference.get("teacher_forced", {}), candidate.get("teacher_forced", {}))
    greedy = compare_greedy(reference.get("greedy", {}), candidate.get("greedy", {}))
    gsm = compare_gsm8k(reference.get("gsm8k", {}), candidate.get("gsm8k", {}))
    noise = {
        "teacher_forced": compare_teacher(reference.get("teacher_forced", {}), reference.get("teacher_forced_repeat", {})),
        "greedy": compare_greedy(reference.get("greedy", {}), reference.get("greedy_repeat", {})),
        "gsm8k": compare_gsm8k(reference.get("gsm8k", {}), reference.get("gsm8k_repeat", {})),
    }
    source = "teacher_forced"
    if not tf.get("positions"):
        # prompt_logprobs unavailable: the greedy positions with identical context stand in.
        tf, source = {k: greedy.get(k) for k in ("positions", "mean_kl", "p50_kl", "p99_kl", "max_kl",
                                                   "top1_agreement", "mean_abs_delta_logprob")}, "greedy_prefix"
    checks = {
        "tf_mean_kl": tf.get("mean_kl") is not None and tf["mean_kl"] < thresholds["tf_mean_kl_nats_max"],
        "tf_top1_agreement": (tf.get("top1_agreement") is not None
                              and tf["top1_agreement"] >= thresholds["tf_top1_agreement_min"]),
        "gsm8k": (gsm.get("scored", 0) >= thresholds["gsm8k_min_scored"]
                  and gsm["delta_points"] >= -thresholds["gsm8k_max_drop_points"] - 1e-9),
    }
    return {"passed": all(checks.values()), "checks": checks, "thresholds": dict(thresholds),
            "distribution_source": source, "teacher_forced": tf, "greedy": greedy, "gsm8k": gsm,
            "control_noise": noise}


# ----------------------------------------------------------------------------- measurements
class Client:
    def __init__(self, base_url, model, effort, transport=None):
        self.base_url = base_url.rstrip("/")
        self.root = self.base_url.removesuffix("/v1")
        self.model, self.effort = model, effort
        self.http = httpx.AsyncClient(timeout=1200, trust_env=False, transport=transport)

    async def close(self):
        await self.http.aclose()

    async def post(self, url, body):
        response = await self.http.post(url, json=body)
        if response.status_code != 200:
            raise RuntimeError(f"{url} {response.status_code}: {response.text[:300]}")
        return response.json()

    def chat(self, messages, **params):
        return self.post(self.base_url + "/chat/completions", {"model": self.model, "messages": messages, **params})


async def bounded(items, concurrency, work, deadline=None):
    """Run ``work(item)`` with at most ``concurrency`` in flight; unfinished items at ``deadline`` are dropped."""
    semaphore = asyncio.Semaphore(concurrency)
    rows = [None] * len(items)

    async def one(i, item):
        async with semaphore:
            try:
                rows[i] = await work(item)
            except Exception as exc:  # noqa: BLE001 - one failed request is recorded, not fatal
                rows[i] = {"index": item["index"], "error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    tasks = [asyncio.create_task(one(i, item)) for i, item in enumerate(items)]
    timeout = None if deadline is None else max(1.0, deadline - time.time())
    done, pending = await asyncio.wait(tasks, timeout=timeout)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return [row if row is not None else {"index": items[i]["index"], "error": "deadline"}
            for i, row in enumerate(rows)]


async def run_greedy(client, prompts, concurrency, prompt_ids=None):
    """Temperature-0 decoding through /v1/completions on the chat-templated prompt ids, so that every
    generated token (reasoning included) carries its top-20 logprobs; the gate reuses the reference's ids."""
    async def work(prompt):
        # Token-id prompts (expert quantization, 28 September: real requests up to their generation prompt).
        ids = (prompt_ids or {}).get(prompt["index"]) or prompt.get("prompt_ids")
        if ids is None:
            reply = await client.post(client.root + "/tokenize", {
                "model": client.model, "add_generation_prompt": True,
                "messages": [{"role": "system", "content": prompt["system"]},
                             {"role": "user", "content": prompt["user"]}],
                "chat_template_kwargs": template_kwargs(client.model, {"enable_thinking": True,
                                                                       "reasoning_effort": client.effort})})
            ids = [int(t) for t in reply["tokens"]]
        reply = await client.post(client.base_url + "/completions", {
            "model": client.model, "prompt": ids, "temperature": 0.0, "max_tokens": GREEDY["max_tokens"],
            "seed": 0, "logprobs": GREEDY["top_logprobs"], "return_tokens_as_token_ids": True})
        choice = reply["choices"][0]
        logprobs = choice.get("logprobs") or {}
        return {"index": prompt["index"], "prompt_ids": ids, "prompt_tokens": len(ids),
                "finish_reason": choice.get("finish_reason"),
                "tokens": [token_id(t) for t in logprobs.get("tokens") or []],
                "top": [[[token_id(t), round(lp, 6)] for t, lp in sorted((entry or {}).items(), key=lambda kv: -kv[1])]
                        for entry in logprobs.get("top_logprobs") or []]}

    started = time.monotonic()
    rows = await bounded(prompts, concurrency, work)
    return {"concurrency": concurrency, "seconds": round(time.monotonic() - started, 2), "rows": rows}


async def tokenize(client, text):
    reply = await client.post(client.root + "/tokenize", {"model": client.model, "prompt": text,
                                                          "add_special_tokens": False})
    return [int(t) for t in reply["tokens"]]


async def run_teacher(client, texts, concurrency, token_ids=None):
    """``prompt_logprobs`` on fixed token ids (the reference's own tokenization is reused by the gate)."""
    items = []
    for i, text in enumerate(texts):
        ids = (token_ids[i] if token_ids is not None else text["ids"] if text.get("ids")
               else await tokenize(client, text["text"]))
        items.append({"index": text["index"], "ids": ids})

    async def work(item):
        reply = await client.post(client.base_url + "/completions", {
            "model": client.model, "prompt": item["ids"], "max_tokens": 1, "temperature": 0.0,
            "prompt_logprobs": TEACHER["prompt_logprobs"]})
        entries = reply["choices"][0].get("prompt_logprobs") or []
        positions = []
        for entry in entries[1:]:
            ranked = sorted(((token_id(t), v["logprob"], v.get("rank") or 0) for t, v in (entry or {}).items()),
                            key=lambda row: row[1], reverse=True)
            positions.append([[t, round(lp, 6)] for t, lp, _ in ranked])
        if len(positions) != len(item["ids"]) - 1:
            raise RuntimeError(f"prompt_logprobs length {len(positions)} != {len(item['ids']) - 1}")
        return {"index": item["index"], "tokens": item["ids"], "positions": positions}

    started = time.monotonic()
    rows = await bounded(items, concurrency, work)
    return {"concurrency": concurrency, "seconds": round(time.monotonic() - started, 2), "rows": rows}


async def run_gsm8k(client, problems, concurrency, deadline):
    async def work(problem):
        reply = await client.chat(
            [{"role": "user", "content": problem["question"] + GSM8K["suffix"]}],
            temperature=GSM8K["temperature"], top_p=GSM8K["top_p"], top_k=GSM8K["top_k"],
            seed=problem["index"], max_tokens=GSM8K["max_tokens"],
            chat_template_kwargs=template_kwargs(client.model, {"enable_thinking": True,
                                                                "reasoning_effort": GSM8K["reasoning_effort"]}))
        message = reply["choices"][0]["message"]
        answer = extract_answer(message.get("content"), message.get("reasoning_content") or message.get("reasoning"))
        gold = gold_answer(problem["answer"])
        return {"index": problem["index"], "answer": answer, "gold": gold,
                "correct": answer is not None and abs(answer - gold) < 1e-6,
                "finish_reason": reply["choices"][0].get("finish_reason"),
                "completion_tokens": reply["usage"]["completion_tokens"]}

    started = time.monotonic()
    rows = await bounded(problems, concurrency, work, deadline=deadline)
    scored = [r for r in rows if "correct" in r]
    return {"concurrency": concurrency, "seconds": round(time.monotonic() - started, 2), "rows": rows,
            "scored": len(scored), "accuracy": sum(r["correct"] for r in scored) / len(scored) if scored else None,
            "completion_tokens": sum(r["completion_tokens"] for r in scored)}


async def run_quality(study, trial, *, source, output, server_dir, base_url, deadline_epoch, log=print,
                      transport=None):
    role = trial.get("role")
    if role not in ("reference", "gate"):
        raise ValueError("A kernel-quality trial is a reference or a gate")
    # Per-trial options (expert quantization, 28 September): a shorter GSM8K pass, no GSM8K repeat, and a
    # report-only gate (the verdict is recorded; the trial completes either way). Absent: the phase-2 behaviour.
    options = dict(trial.get("quality") or {})
    # Long-context drift (29 September): "greedy": false and "gsm8k": false leave those passes out (teacher forcing
    # and its repeat only); the reference then needs only the teacher-forced measurement.
    run_greedy_pass, run_gsm8k_pass = options.get("greedy", True) is not False, options.get("gsm8k", True) is not False
    gsm8k_seconds = int(options.get("gsm8k_pass_seconds", GSM8K["pass_seconds"]))
    # Wide decode tiles (kernel profile, 30 September): the greedy pass at a decode batch of 33-56 rows (the only
    # path the wide buckets serve; teacher forcing prefills), and its repeat at the same batch, so the noise bar
    # has the same batch composition. Absent: the phase-2 concurrencies (16 and 1).
    greedy_concurrency = int(options.get("greedy_concurrency", GREEDY["concurrency"]))
    greedy_repeat_concurrency = int(options.get("greedy_repeat_concurrency", GREEDY["repeat_concurrency"]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prompts, problems = load_inputs(source)
    reference = None
    if role == "gate":
        path = output.parent / trial["reference"] / "kernel-quality.json"
        if not path.is_file():
            raise RuntimeError(f"The gate needs the reference measurements at {path}")
        reference = json.loads(path.read_text())
    client = Client(base_url, study.get("served_model", MODEL), study.get("reasoning_effort", "xhigh"),
                    transport=transport)
    result = {"role": role, "trial": trial["name"], "server_env": trial.get("server_env") or {},
              "server": study["server"], "greedy_settings": GREEDY, "teacher_settings": TEACHER,
              "gsm8k_settings": GSM8K, "errors": []}
    path = output / "kernel-quality.json"

    def save():
        path.write_text(json.dumps(result))

    def event(name, **extra):
        log(json.dumps({"event": f"kernel_quality_{name}", "trial": trial["name"], **extra}), flush=True)

    async def step(name, coroutine):
        started = time.monotonic()
        try:
            result[name] = await coroutine
        except Exception as exc:  # noqa: BLE001 - a failed measurement is recorded; the verdict sees it missing
            result[name] = {"error": f"{type(exc).__name__}: {str(exc)[:500]}", "rows": []}
            result["errors"].append(f"{name}: {result[name]['error']}")
        rows = result[name].get("rows", [])
        event(name, seconds=round(time.monotonic() - started, 1), rows=len(rows),
              failed=sum(1 for r in rows if "error" in r), error=result[name].get("error"))
        save()

    try:
        await isolate_trial(base_url)
        greedy_ids = None
        if reference is not None:
            greedy_ids = {row["index"]: row["prompt_ids"] for row in reference.get("greedy", {}).get("rows", [])
                          if "prompt_ids" in row}
        if run_greedy_pass:
            await step("greedy", run_greedy(client, prompts["greedy"], greedy_concurrency, greedy_ids))
        ids = None
        if reference is not None:
            ids = {row["index"]: row["tokens"] for row in reference.get("teacher_forced", {}).get("rows", [])
                   if "tokens" in row}
            ids = [ids.get(t["index"]) for t in prompts["teacher_forced"]] if ids else None
            if ids is not None and any(v is None for v in ids):
                ids = None
        await step("teacher_forced", run_teacher(client, prompts["teacher_forced"], TEACHER["concurrency"], ids))
        if role == "reference":
            tokens = [row.get("tokens") for row in result["teacher_forced"].get("rows", [])]
            greedy_ids = {row["index"]: row["prompt_ids"] for row in result.get("greedy", {}).get("rows", [])
                          if "prompt_ids" in row}
            if run_greedy_pass:
                await step("greedy_repeat", run_greedy(client, prompts["greedy"], greedy_repeat_concurrency,
                                                       greedy_ids))
            await step("teacher_forced_repeat", run_teacher(
                client, prompts["teacher_forced"], TEACHER["repeat_concurrency"],
                tokens if tokens and all(tokens) else None))
        early = None
        if reference is not None:
            early = verdict(reference, {k: result.get(k, {}) for k in ("greedy", "teacher_forced")})
            event("distribution_check", checks={k: v for k, v in early["checks"].items() if k != "gsm8k"},
                  teacher_forced=early["teacher_forced"])
        if run_gsm8k_pass and (early is None or options.get("report_only") or (early["checks"]["tf_mean_kl"]
                                                                              and early["checks"]["tf_top1_agreement"])):
            await step("gsm8k", run_gsm8k(client, problems, GSM8K["concurrency"],
                                          min(deadline_epoch - 600, time.time() + gsm8k_seconds)))
            if role == "reference" and options.get("gsm8k_repeat", True):
                await step("gsm8k_repeat", run_gsm8k(client, problems, GSM8K["repeat_concurrency"],
                                                     min(deadline_epoch - 600, time.time() + gsm8k_seconds)))
        await isolate_trial(base_url)
        if trial.get("bench"):
            from .throughput_bench import run_bench

            started = time.monotonic()
            try:
                bench = await run_bench(study, {"throughput_bench": trial["bench"]}, output=output / "bench",
                                        server_dir=server_dir, base_url=base_url, deadline_epoch=deadline_epoch - 300,
                                        source=source)
                result["bench"] = {"status": bench["status"], "task_errors": bench["task_errors"]}
            except Exception as exc:  # noqa: BLE001 - throughput evidence is optional for the verdict
                result["bench"] = {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            event("bench", seconds=round(time.monotonic() - started, 1), status=result["bench"]["status"])
            await isolate_trial(base_url)
    finally:
        await client.close()
    if reference is not None:
        result["verdict"] = verdict(reference, result)
        result["status"] = "complete" if result["verdict"]["passed"] or options.get("report_only") else "gate_failed"
        (output / "kernel-gate.json").write_text(json.dumps(result["verdict"], indent=2))
        event("verdict", passed=result["verdict"]["passed"], checks=result["verdict"]["checks"],
              teacher_forced=result["verdict"]["teacher_forced"], gsm8k=result["verdict"]["gsm8k"])
    else:
        # Teacher forcing may be unavailable (prompt_logprobs); the gate then falls back to greedy prefixes.
        # (With a pass left out, the teacher-forced measurement is required instead of relying on the fallback.)
        needed = (["greedy"] if run_greedy_pass else []) + (["gsm8k"] if run_gsm8k_pass else [])
        if not (run_greedy_pass and run_gsm8k_pass):
            needed.append("teacher_forced")
        missing = [k for k in needed if not any("error" not in r for r in result.get(k, {}).get("rows", []))]
        result["status"] = "complete" if not missing else "error"
        if missing:
            result["errors"].append(f"reference measurements missing: {missing}")
    save()
    return result
