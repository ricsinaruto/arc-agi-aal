"""Test-time training data for one ARC-AGI-2 task (per-task LoRA on Flash).

Samples are leave-one-out over the task's training pairs: the other pairs are the context, the held-out pair's input
is the query and its output grid is the only labelled text. Each sample is drawn on ``augments`` views of the task
(``loop.augment_view``: view 0 is the task itself, the others a symmetry and a colour permutation from the task id),
in the harness's own text format (``tasks.task_prompt``, ``tasks.DIRECT_SYSTEM``), with no images and no reasoning.
``queries`` gives the screen's prompts for the real test inputs (all training pairs as context).
"""
from . import tasks
from .loop import augment_back, augment_view

CHARS_PER_TOKEN = 1.0  # grid text is digits and spaces: the Qwen tokenizer splits every digit and space (measured 1.0)


def answer(grid):
    """The labelled assistant text: one fenced block holding the grid, as DIRECT_SYSTEM asks."""
    return "```\n" + tasks.text(grid) + "\n```"


def messages(context):
    """System + user turn for a task-shaped ``context`` ({"train": [...], "test": [{"input": ...}]})."""
    return [{"role": "system", "content": tasks.DIRECT_SYSTEM}, {"role": "user", "content": tasks.task_prompt(context)}]


def samples(task, task_id, augments=4, max_tokens=12000):
    """Leave-one-out samples on ``augments`` views; each {"messages", "target", "view", "held_out", "tokens"}.
    A task with one training pair has no leave-one-out sample; one longer than ``max_tokens`` is left out."""
    out = []
    if len(task["train"]) < 2:
        return out
    for k in range(augments):
        view = augment_view(task, task_id, k)
        for j, pair in enumerate(view["train"]):
            context = {"train": [p for i, p in enumerate(view["train"]) if i != j], "test": [{"input": pair["input"]}]}
            msgs, target = messages(context), answer(pair["output"])
            tokens = int(sum(len(m["content"]) for m in msgs) / CHARS_PER_TOKEN + len(target) / CHARS_PER_TOKEN)
            if tokens <= max_tokens:
                out.append({"messages": msgs, "target": target, "view": k, "held_out": j, "tokens": tokens})
    return out


def queries(task, task_id, view=0):
    """One prompt per test input on ``view`` (all training pairs as context); ``target`` (when the task carries its
    test outputs) is that output in the view's frame, for the NLL screen. ``back`` maps a view grid to the task frame."""
    v = augment_view(task, task_id, view)
    out = []
    for i, item in enumerate(v["test"]):
        q = {"messages": messages({"train": v["train"], "test": [item]}), "test_index": i, "view": view}
        if "output" in task["test"][i]:
            sym_out = augment_view({"train": [{"input": task["test"][i]["output"], "output": task["test"][i]["output"]}],
                                    "test": []}, task_id, view)["train"][0]["input"]
            q["target"] = answer(sym_out)
        out.append(q)
    return out


def back(grid, task_id, view):
    return augment_back(grid, task_id, view)
