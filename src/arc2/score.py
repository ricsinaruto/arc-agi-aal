"""Product-of-experts likelihood of candidate test outputs (plan item 1, 23 Oct): for each (task, reading) of a trial's
``score_items`` file and each of ``score_views`` augmented views (loop.augment_spec), the served model's log-prob of
the reading's grids as the answer to the puzzle, read from ``prompt_logprobs`` (prefill only, one output token).
The span scored is the assistant answer: the prompt's tokens after those of the same messages with an empty answer.
Rows go to ``scores.jsonl`` as they finish (resume: rows there are skipped)."""

import asyncio
import json
import time
from pathlib import Path

import httpx

from arena.early_stop import stop_paths

from . import tasks
from .loop import SYMMETRIES, augment_spec


def view_of(task, outputs, task_id, k):
    """Copy ``k`` of the task with ``outputs`` as its test outputs (k = 0: unchanged)."""
    sym, colours = augment_spec(task_id, k)
    fwd = SYMMETRIES[sym][0]
    grid = lambda g: [[colours[c] for c in row] for row in fwd(g)]  # noqa: E731
    return ({"train": [{"input": grid(p["input"]), "output": grid(p["output"])} for p in task["train"]],
             "test": [{"input": grid(t["input"])} for t in task["test"]]}, [grid(g) for g in outputs])


def answer_text(outputs):
    return "\n\n".join("```\n" + tasks.text(g) + "\n```" for g in outputs)


def messages(view, answer):
    return [{"role": "system", "content": tasks.DIRECT_SYSTEM}, {"role": "user", "content": tasks.task_prompt(view)},
            {"role": "assistant", "content": answer}]


async def score_one(client, model, item, task, k, template):
    view, outs = view_of(task, item["tests"], item["task"], k)
    answer = answer_text(outs)
    extra = {"add_generation_prompt": False, "continue_final_message": True,
             "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low", **template}}
    empty = await client.post("/tokenize", json={"model": model, "messages": messages(view, ""), **extra})
    empty.raise_for_status()
    start = int(empty.json()["count"])
    response = await client.post("/v1/chat/completions", json={
        "model": model, "messages": messages(view, answer), "max_tokens": 1, "temperature": 0.0,
        "prompt_logprobs": 1, "return_token_ids": True, **extra})
    response.raise_for_status()
    data = response.json()
    rows = data.get("prompt_logprobs") or data["choices"][0].get("prompt_logprobs")
    ids = data.get("prompt_token_ids") or data["choices"][0].get("prompt_token_ids") or [None] * len(rows)
    picked, decoded = [], []
    for token, row in zip(ids[start:], rows[start:]):
        row = row or {}
        entry = row.get(str(token)) if token is not None else None
        if entry is None and row:  # no ids: the prompt token is the entry ranked last (top-1 has rank 1)
            entry = max(row.values(), key=lambda e: e.get("rank") or 0)
        entry = entry or {}
        picked.append(float(entry.get("logprob", 0.0)))
        decoded.append(entry.get("decoded_token") or "")
    text = "".join(decoded)
    return {"task": item["task"], "reading": item["id"], "view": k, "right": item.get("right"),
            "count": item.get("count"), "tokens": len(picked), "logprob": round(sum(picked), 4),
            "mean": round(sum(picked) / max(1, len(picked)), 6),
            "span_ok": text.lstrip().startswith("```"),
            "span_head": text[:40], "prompt_tokens": len(ids)}


async def run_scores(trial, task_map, output, base_url, deadline, model="flash", template=None):
    """The trial's whole work when it has ``score_items``: every item under every view, ``score_concurrency`` at a
    time, until done or the wall. Returns a report like run_arc2's."""
    output = Path(output)
    items = json.loads((Path(trial["score_source"]) / trial["score_items"]).read_text())
    views = int(trial.get("score_views", 4))
    out_path = output / "scores.jsonl"
    done = set()
    if out_path.is_file():
        done = {(r["reading"], r["view"]) for r in map(json.loads, out_path.read_text().splitlines())}
    jobs = [(item, k) for item in items for k in range(views) if (item["id"], k) not in done]
    root = base_url[:-3] if base_url.endswith("/v1") else base_url
    queue, errors, written = asyncio.Queue(), [], [0]
    for job in jobs:
        queue.put_nowait(job)
    began = time.monotonic()
    with out_path.open("a") as sink:
        async with httpx.AsyncClient(base_url=root, timeout=600) as client:
            stops = stop_paths(output)

            async def worker():
                while not queue.empty() and time.monotonic() < deadline and not any(p.exists() for p in stops):
                    item, k = queue.get_nowait()
                    for attempt in range(3):
                        try:
                            row = await score_one(client, model, item, task_map[item["task"]], k, template or {})
                            sink.write(json.dumps(row) + "\n")
                            sink.flush()
                            written[0] += 1
                            break
                        except Exception as exc:  # noqa: BLE001 - recorded, retried twice
                            if attempt == 2:
                                errors.append(f"{item['id']}/{k}: {type(exc).__name__}: {exc}"[:300])
                            await asyncio.sleep(2)

            await asyncio.gather(*(worker() for _ in range(int(trial.get("score_concurrency", 16)))))
    stopped = any(p.exists() for p in stop_paths(output))
    report = {"kind": "scores", "status": "complete", "ended": "done" if queue.empty() else "stop_request" if stopped else "wall", "jobs": len(jobs),
              "written": written[0], "skipped": len(done), "errors": errors[:20], "error_count": len(errors),
              "seconds": round(time.monotonic() - began, 1)}
    (output / "report.json").write_text(json.dumps(report, indent=1))
    return report
