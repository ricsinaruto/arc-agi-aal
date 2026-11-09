"""Served-model reference for the TTT parity check: for each task's first test input (view 0,
``arc2.ttt_data.queries``, thinking off) the server renders the prompt and the prompt + true answer (/tokenize), and a
prefill with ``prompt_logprobs`` gives the served log-probability of every answer token. The B300 TTT run scores the
same token ids with its trainer forward and compares (mean |delta| per token)."""

import json
import time
from pathlib import Path

from . import tasks, ttt_data


async def run_nll_parity(study, trial, *, source, output, base_url):
    import httpx

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = study["arc2"]
    ids = list(trial["tasks"])
    task_map = tasks.load(Path(source) / spec["data"] / trial["split"], ids)
    report = {"status": "running", "tasks": len(ids), "errors": [], "rows": 0}
    began = time.time()
    rows = []
    async with httpx.AsyncClient(base_url=base_url.rstrip("/") + "/") as client:
        model = (await client.get("models", timeout=60)).json()["data"][0]["id"]
        tokenize = str(client.base_url.copy_with(path="/tokenize"))
        for task_id in ids:
            try:
                q = ttt_data.queries(task_map[task_id], task_id)[0]
                kw = {"model": model, "chat_template_kwargs": {"enable_thinking": False}}
                p = await client.post(tokenize, timeout=120, json={**kw, "messages": q["messages"],
                                                                     "add_generation_prompt": True})
                f = await client.post(tokenize, timeout=120, json={
                    **kw, "messages": q["messages"] + [{"role": "assistant", "content": q["target"]}],
                    "add_generation_prompt": False})
                p.raise_for_status()
                f.raise_for_status()
                prompt_ids, full_ids = p.json()["tokens"], f.json()["tokens"]
                if full_ids[:len(prompt_ids)] != prompt_ids:
                    raise ValueError("prompt is not a prefix of prompt + answer")
                r = await client.post("completions", timeout=1800, json={
                    "model": model, "prompt": full_ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1,
                    "return_tokens_as_token_ids": True})
                r.raise_for_status()
                lps = r.json()["choices"][0]["prompt_logprobs"]
                answer = []
                for pos in range(len(prompt_ids), len(full_ids)):
                    row = lps[pos] or {}
                    answer.append(float(row[str(full_ids[pos])]["logprob"]))
                rows.append({"task": task_id, "prompt_ids": prompt_ids, "full_ids": full_ids, "logprobs": answer,
                             "nll": -sum(answer) / len(answer), "target": q["target"]})
            except Exception as exc:  # noqa: BLE001 - recorded
                report["errors"].append(f"{task_id}: {type(exc).__name__}: {str(exc)[:300]}")
    (output / "parity.json").write_text(json.dumps({"model": model, "rows": rows}))
    report.update(rows=len(rows), seconds=round(time.time() - began, 1),
                  nll={r["task"]: round(r["nll"], 5) for r in rows},
                  status="complete" if rows and not report["errors"] else "error")
    (output / "report.json").write_text(json.dumps(report, indent=1))
    return report
