"""Server-side check of the "prefilled-empty" compaction request before gameplay (hero-v2, 29 September).

``compaction_thinking = "prefilled-empty"`` (arena.compaction) sends the summary as the gameplay messages plus the
summary prompt and a pre-filled assistant turn, continued with ``continue_final_message``. Its token identity was
checked on the Qwen3.8-27B template (the Flash-Next template is not on disk). This preflight renders Duck-shaped
conversations through the served model's own ``/tokenize`` (the server's chat template) and requires, for each:

1. **prefix**: the gameplay request (``add_generation_prompt``) and the summary request (``add_generation_prompt``
   false, ``continue_final_message``) both start with the tokens of the conversation itself, so the summary
   reuses the cached gameplay context;
2. **empty think**: the summary's final assistant turn starts with the template's empty think block followed by the
   prefill (``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`` + ``SUMMARY_PREFILL``), so no reasoning is
   generated.

The conversations carry what makes a template diverge: earlier assistant turns with reasoning and tool calls, tool
results after the last user message (the summary prompt then becomes the last user message) and a trailing user
observation. On any failure (or a failed request) the trial serves "gameplay" and records why.
"""

import json

from arc_infra.model_templates import template_kwargs

from .compaction import DUCK_SUMMARY_PROMPT, SUMMARY_PREFILL

TOOLS = [{"type": "function", "function": {
    "name": "python", "description": "Run Python in the persistent game kernel and return its output.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
EMPTY_THINK = "<think>\n\n</think>\n\n"
ASSISTANT = "<|im_start|>assistant\n"


def _call(index, code):
    return {"id": f"call_{index}", "type": "function", "function": {"name": "python", "arguments": json.dumps(
        {"code": code})}}


def conversations():
    """Two Duck-shaped histories: ending in tool results (after the last user turn) and in a user observation."""
    system = {"role": "system", "content": "You are a coding agent playing a grid game through a python tool."}
    first = {"role": "user", "content": [{"type": "text", "text": "Level 1 of 7. Board 64x64:\n0000\n0110\n0000"}]}
    turns = [
        {"role": "assistant", "content": "", "reasoning": "The blue block probably moves with ACTION1.",
         "tool_calls": [_call(1, "step('ACTION1')")]},
        {"role": "tool", "tool_call_id": "call_1", "content": "board changed: 4 cells; levels 0"},
        {"role": "assistant", "content": "Testing the other direction.",
         "reasoning": "ACTION1 moved it up one row; check ACTION2.",
         "tool_calls": [_call(2, "step('ACTION2')")]},
        {"role": "tool", "tool_call_id": "call_2", "content": "board changed: 4 cells; levels 0"},
    ]
    observation = {"role": "user", "content": [{"type": "text", "text": "New observation: the block is at (3, 1)."}]}
    return {"tool_tail": [system, first, *turns],
            "user_tail": [system, first, *turns, observation]}


def common_prefix(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


async def run_preflight(client, base_url, *, model, effort, enable_thinking=True):
    """``{"ok": bool, "mode": "prefilled-empty" | "gameplay", "reason": str | None, "shapes": {...}}``."""
    root = base_url.rstrip("/").removesuffix("/v1")
    kwargs = template_kwargs(model, {"enable_thinking": bool(enable_thinking), "reasoning_effort": effort})

    async def tokenize(messages, **flags):
        body = {"model": model, "messages": messages, "tools": TOOLS, "chat_template_kwargs": kwargs,
                "add_special_tokens": False, **flags}
        response = await client.post(root + "/tokenize", json=body)
        response.raise_for_status()
        return response.json()["tokens"]

    async def detokenize(tokens):
        response = await client.post(root + "/detokenize", json={"model": model, "tokens": tokens})
        response.raise_for_status()
        return response.json()["prompt"]

    shapes, failures = {}, []
    try:
        for name, messages in conversations().items():
            base = await tokenize(messages, add_generation_prompt=False)
            gameplay = await tokenize(messages, add_generation_prompt=True)
            summary = await tokenize([*messages, {"role": "user", "content": DUCK_SUMMARY_PROMPT},
                                      {"role": "assistant", "content": SUMMARY_PREFILL}],
                                     add_generation_prompt=False, continue_final_message=True)
            tail = await detokenize(summary[len(base):])
            turn = tail[tail.rfind(ASSISTANT) + len(ASSISTANT):] if ASSISTANT in tail else None
            row = {"conversation_tokens": len(base), "gameplay_tokens": len(gameplay), "summary_tokens": len(summary),
                   "gameplay_shares_conversation": common_prefix(base, gameplay) == len(base),
                   "summary_shares_conversation": common_prefix(base, summary) == len(base),
                   "summary_vs_gameplay_common": common_prefix(gameplay, summary),
                   "assistant_turn": None if turn is None else turn[:80],
                   "empty_think_then_prefill": turn is not None and turn.startswith(EMPTY_THINK + SUMMARY_PREFILL)}
            shapes[name] = row
            for key in ("gameplay_shares_conversation", "summary_shares_conversation", "empty_think_then_prefill"):
                if not row[key]:
                    failures.append(f"{name}: {key} false")
    except Exception as exc:  # noqa: BLE001 - any failure falls back to the gameplay-mode summary
        failures.append(f"{type(exc).__name__}: {str(exc)[:300]}")
    ok = not failures
    return {"ok": ok, "mode": "prefilled-empty" if ok else "gameplay", "reason": None if ok else "; ".join(failures),
            "shapes": shapes, "model": model, "chat_template_kwargs": kwargs}
