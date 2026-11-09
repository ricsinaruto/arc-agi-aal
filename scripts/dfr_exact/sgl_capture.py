"""Kernel-quality capture on his SGLang server (no comparison): greedy top-20 logprobs on chat prompts, teacher-forced
top-20 input logprobs on fixed token ids, GSM8K answers, in kernel_gate's row formats so arena.kernel_gate's
compare_* run offline on two captures. Native /generate (SGLang has no vLLM prompt_logprobs).
python -m ... sgl_capture.py BASE_URL MODEL INPUTS_DIR OUT_JSON [IDS_JSON]   (IDS_JSON: a reference capture whose
token ids are reused, so both sides score the same ids)"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from arena.kernel_gate import GREEDY, GSM8K, extract_answer, gold_answer
from arc_infra.model_templates import template_kwargs

BASE, MODEL, INPUTS, OUT = sys.argv[1:5]
REF = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None


def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def top(entries):
    return [[int(t), round(float(lp if lp is not None else -1e9), 6)] for lp, t, *_ in sorted(entries or [], key=lambda e: -(e[0] if e[0] is not None else -1e9))]


def greedy(prompt, ids=None):
    if ids is None:
        ids = post("/tokenize", {"model": MODEL, "add_generation_prompt": True,
                                 "messages": ([{"role": "system", "content": prompt["system"]}] if prompt.get("system") else [])
                                             + [{"role": "user", "content": prompt["user"]}],
                                 "chat_template_kwargs": template_kwargs(MODEL, {"enable_thinking": True,
                                                                                  "reasoning_effort": "xhigh"})})["tokens"]
    r = post("/generate", {"input_ids": ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": GREEDY["max_tokens"]},
                           "return_logprob": True, "top_logprobs_num": GREEDY["top_logprobs"]})
    m = r["meta_info"]
    return {"index": prompt["index"], "prompt_ids": ids, "prompt_tokens": len(ids),
            "tokens": [int(t[1]) for t in m["output_token_logprobs"]], "top": [top(e) for e in m["output_top_logprobs"]]}


def teacher(item, ids=None):
    if ids is None:
        ids = post("/tokenize", {"model": MODEL, "prompt": item["text"], "add_special_tokens": False})["tokens"]
    r = post("/generate", {"input_ids": ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
                           "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": 20})
    pos = [top(e) for e in r["meta_info"]["input_top_logprobs"][1:]]
    if len(pos) != len(ids) - 1:
        raise RuntimeError(f"input_top_logprobs {len(pos)} != {len(ids) - 1}")
    return {"index": item["index"], "tokens": ids, "positions": pos}


def gsm(problem):
    r = post("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": problem["question"] + GSM8K["suffix"]}],
             "temperature": GSM8K["temperature"], "top_p": GSM8K["top_p"], "top_k": GSM8K["top_k"], "seed": problem["index"],
             "max_tokens": GSM8K["max_tokens"],
             "chat_template_kwargs": template_kwargs(MODEL, {"enable_thinking": True, "reasoning_effort": GSM8K["reasoning_effort"]})})
    msg = r["choices"][0]["message"]
    ans = extract_answer(msg.get("content"), msg.get("reasoning_content") or msg.get("reasoning"))
    gold = gold_answer(problem["answer"])
    return {"index": problem["index"], "answer": ans, "gold": gold, "correct": ans is not None and abs(ans - gold) < 1e-6}


def flushed(fn):
    """Deterministic capture: one request at a time on an empty radix cache, so every run prefills the same chunks
    in the same batch (restart noise on the batched capture: TF KL 0.0345, top-1 0.92, greedy exact 1/16)."""
    def wrapped(*a):
        urllib.request.urlopen(urllib.request.Request(BASE + "/flush_cache", data=b"", method="POST"), timeout=300).read()
        return fn(*a)
    return wrapped


def run(fn, items, workers, ids=None):
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(lambda it: fn(it, ids.get(it["index"])) if ids is not None else fn(it), items))
    return {"seconds": round(time.time() - t0, 1), "rows": rows}


prompts = json.load(open(f"{INPUTS}/prompts.json"))
problems = [json.loads(l) for l in open(f"{INPUTS}/gsm8k.jsonl") if l.strip()]
for i, p in enumerate(problems):
    p.setdefault("index", i)
gids = {r["index"]: r["prompt_ids"] for r in REF["greedy"]["rows"]} if REF else None
tids = {r["index"]: r["tokens"] for r in REF["teacher_forced"]["rows"]} if REF else None
# 16 arena prompts + 48 GSM8K questions as greedy chat prompts (index 1000+): 64 greedy prompts.
greedy_items = prompts["greedy"] + [{"index": 1000 + i, "user": p["question"] + GSM8K["suffix"]} for i, p in enumerate(problems[:48])]
out = {"greedy": run(flushed(greedy), greedy_items, 1, gids),
       "teacher_forced": run(flushed(teacher), prompts["teacher_forced"], 1, tids),
       "gsm8k": run(gsm, problems[:200], 16)}
# Same-server repeats: within-boot noise, so cross-boot differences (startup autotune, kernel choice) separate out.
out["greedy_repeat"] = run(flushed(greedy), greedy_items, 1, gids)
out["teacher_forced_repeat"] = run(flushed(teacher), prompts["teacher_forced"], 1, tids)
json.dump(out, open(OUT, "w"))
print(json.dumps({"sgl_capture": {k: v["seconds"] for k, v in out.items()}}), flush=True)
