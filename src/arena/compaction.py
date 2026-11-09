"""Append-only contexts with model-written compaction at the cap.

The public ToolAgent (and duck_transfer's trimmer on top of it) drops or stubs old messages,
which rewrites the prefix on every turn. In the arena a request's message list is only ever
extended until it reaches the budget; then one tool-less summary request replaces everything
before the latest observation with the model's own summary. ``CompactionMixin`` methods are
bound onto a ToolAgent instance (``MethodType``) so the public analyze loop calls them in the
places it would otherwise trim.
"""

import copy
from types import MethodType

SUMMARY_PREAMBLE = (
    "[Context compaction] Earlier chat messages were replaced by your summary below. "
    "This does not remove evidence retained by your tools: available history and any "
    "surviving variables or files can still be accessed through their existing interfaces. "
    "A reported tool reset may have removed variables. Continue from the summary and "
    "available evidence.\n\n"
)

DUCK_SUMMARY_PROMPT = (
    "Your context is full. Write a complete handoff for yourself so you can continue this game "
    "with only this summary and the next board. Do not call tools. Cover, with concrete "
    "coordinates and step numbers where they matter:\n"
    "1. World model: the board's objects, regions and what they mean.\n"
    "2. Goal model: what completes a level, with the evidence.\n"
    "3. Action model: what every action does (verified vs. hypothesised), including no-ops.\n"
    "4. Current level and state: the level number, the board's current configuration, the last "
    "actions and their results, and any timer or budget you have inferred.\n"
    "5. Actions already tried on this exact board without effect.\n"
    "6. Helper functions or data you defined in the python tool that are still worth using or "
    "re-defining (names, purpose, key lines).\n"
    "7. Plan: the next actions and the expectations to check.\n"
    "8. Cross-level notes: what carried over between levels and what did not.\n"
    "Be precise and compact; prefer facts over narrative."
)

R_SUMMARY_PROMPT = (
    "Your context is full. Write a complete handoff for yourself so you can continue this game "
    "with only this summary, log.txt, playbook.md and your scratch files. Do not call tools. "
    "Cover:\n"
    "1. Verified mechanics as rules (what each action does, per level if it differs).\n"
    "2. Simulator or scratch code you wrote: file names, what they retrodict correctly, known gaps.\n"
    "3. playbook.md: what it contains and what is stale.\n"
    "4. Current level, step number, state line, and the board's current configuration.\n"
    "5. The plan you were executing, its expectations, and which steps since the last plan are "
    "still unexplained.\n"
    "6. Pitfalls: expectations that failed, HUD or timer cells, repeated no-ops.\n"
    "Be precise and compact; prefer facts over narrative."
)


def _role(message):
    return str(message.get("role", "")).strip()


def split_at_latest_observation(history, keep=1):
    """``(prefix, tail)``: everything before the ``keep``-th latest user message, and from it
    onward. ``keep`` > 1 (24 September retention variant) keeps the last ``keep`` observations
    and their tool rounds verbatim after a compaction instead of only the latest one."""
    keep = max(1, int(keep))
    seen = 0
    for index in range(len(history) - 1, -1, -1):
        if _role(history[index]) == "user":
            seen += 1
            if seen == keep:
                return list(history[:index]), list(history[index:])
    if seen:  # fewer than ``keep`` observations: keep from the earliest one
        first = next(i for i in range(len(history)) if _role(history[i]) == "user")
        return list(history[:first]), list(history[first:])
    return list(history), []


def summary_message(summary):
    return {"role": "user", "content": SUMMARY_PREAMBLE + summary.strip()}


def compacted_messages(system_message, summary, tail):
    """The message list after a compaction: system prompt, summary, the latest observation."""
    return [system_message, summary_message(summary), *tail]


# How the compaction summary request thinks (study ``compaction_thinking``; 29 September, user decision):
# - "prefilled-empty" (default for new builds): the "gameplay" request plus an assistant turn that pre-fills the
#   template's empty think block, continued with vLLM's ``continue_final_message``. The request is byte-identical
#   to the gameplay request up to its last message, so it reuses the cached context, and it writes the summary
#   without reasoning.
# - "gameplay" (28 September, a1e350d; every study built before the key): the gameplay prefix, thinking at the
#   gameplay effort (1,400-2,100 reasoning tokens, 60-93 s per compaction in the hero runs).
# - "off" (before 28 September): tool-less, thinking-off; the template drops the effort line and the tool list
#   from the system block, so none of the game's cached context is reused.
COMPACTION_THINKING = ("prefilled-empty", "gameplay", "off")
DEFAULT_COMPACTION_THINKING = "gameplay"  # a study without the key (built before 29 September)
# The Qwen3.8 template (tests/fixtures/qwen27-chat-template.jinja, sha256 c3cf9e34...) renders a final assistant
# message without reasoning as ``<|im_start|>assistant\n<think>\n\n</think>\n\n`` + content: the empty think block
# its non-thinking generation prompt emits. The content is the handoff's first words, because with an empty
# content transformers' continue_final_message cuts the prompt back to ``</think>`` (5.x) or leaves the closing
# ``<|im_end|>`` in place (4.x).
SUMMARY_PREFILL = "World model:"
# Models served with that Qwen3.8 template; any other model keeps the "gameplay" request.
# Holo4-27B ships the Qwen3.8-27B template byte for byte (its manifest pins the same sha256).
PREFILL_TEMPLATE_MODELS = ("Qwen/Qwen3.8-Flash-Next-NVFP4", "nvidia/Qwen3.8-27B-NVFP4", "Qwen/Qwen3.8-27B-FP8",
                           "Hcompany/Holo4-27B-NVFP4")


def summary_payload(
    *, build_chat_payload, model, messages, prompt, cap, temperature, top_p, top_k,
    thinking=False, reasoning_effort=None, tools=None, template_kwargs=None, prefill=None,
):
    """A request asking the model to summarise ``messages``: tool-less and thinking-off by default.

    Compaction (28 September, docs/validation-20260927/prefill-and-compaction.md) passes the gameplay
    request's ``tools``, ``thinking`` and ``reasoning_effort`` instead: the chat template renders the effort
    line and the tool list at the top of the system block, so a tool-less, thinking-off summary shared no
    prefix with the game's cached context and re-prefilled all of it (11% of the uncached prompt tokens in
    c110-kernels-v2). With the same prefix the summary reuses the cache, and thinks at the gameplay effort.
    The tools are rendered but not callable (``tool_choice`` "none").

    ``prefill`` (compaction_thinking "prefilled-empty") appends an assistant message with that content, which the
    model continues (``continue_final_message``, no generation prompt) after the template's empty think block;
    everything up to the summary prompt is unchanged."""
    prefilled = [] if prefill is None else [{"role": "assistant", "content": prefill}]
    payload = build_chat_payload(
        provider="vllm",
        model=model,
        messages=[*messages, {"role": "user", "content": prompt}, *prefilled],
        max_tokens=int(cap),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        thinking=bool(thinking),
        tools=tools or None,
        tool_choice="none" if tools else None,
        seed=-1,
    )
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "none"
    else:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
    if template_kwargs is not None:
        payload["chat_template_kwargs"] = dict(template_kwargs)  # exactly the gameplay request's
    elif thinking and reasoning_effort:
        payload.setdefault("chat_template_kwargs", {})["reasoning_effort"] = reasoning_effort
    if prefill is not None:
        payload["add_generation_prompt"] = False
        payload["continue_final_message"] = True
    return payload


SUMMARY_RETRY_SUFFIX = (
    "\n\nYour previous reply contained no text. Write the handoff now as plain text, starting with "
    "'World model:'."
)


def mechanical_summary(*, steps, levels, total_levels=None):
    """Handoff used when the model returns no summary twice: the facts the runner knows."""
    progress = f"{levels} of {total_levels}" if total_levels else str(levels)
    return (
        "World model: (summary unavailable: the earlier context was dropped without a model-written "
        f"handoff). Progress so far: {steps} actions issued, levels completed: {progress}. "
        "Re-derive the mechanics from the next board: inspect it, test one action at a time, and "
        "rebuild your helper functions in the python tool before acting on a plan."
    )


def summary_text(message):
    """The reply's text content (a list of parts is joined; reasoning is ignored)."""
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
        )
    return str(content or "").strip()


def prefilled_summary_text(message, prefill):
    """A prefilled summary: ``prefill`` plus the continuation, or "" when the model added nothing.

    The request keeps the gameplay template settings, so vLLM's reasoning parser starts in its reasoning state and,
    with no ``</think>`` in the output (the prompt closed the think block), may return the whole continuation as
    reasoning: the continuation is whichever of the fields carries text."""
    parts = []
    for key in ("reasoning_content", "reasoning", "content"):
        part = summary_text({"content": message.get(key)})
        if part and part not in parts:
            parts.append(part)
    continuation = "\n".join(parts)
    return f"{prefill} {continuation}" if parts else ""


class CompactionMixin:
    """Replacements for the public ToolAgent's trimming; needs ``_owner`` with
    ``estimate_prompt_tokens(messages, tools)`` and ``summarize(messages)`` (or, with an owner ``drain`` block,
    ``drain_context``: hero-v2 drain_half, which drops the oldest blocks instead and is append-only between drains).

    Append-only invariant: the returned list is either the input list itself or exactly
    ``[system, summary, *tail]`` where ``tail`` is a suffix of the input starting at the latest
    observation (or, with the ``summary_split`` hook, at an earlier user or assistant message);
    no message object is ever mutated.
    """

    _owner = None

    def _trim_messages_for_context(
        self, messages, *, tools=None, preserve_recent=1, extra_safety_tokens=0
    ):
        if not messages:
            return []
        owner = self._owner
        # Optional owner hooks (the arena's harness_fixes; absent or identity otherwise): ``before_trim`` may replace
        # the not-yet-sent last message (the cap-hit follow-up), ``trigger_tokens`` measures the would-be request
        # (the exact count instead of the estimate) and ``summary_source`` picks the messages the summary covers
        # (up to the latest observation when the context has overshot); ``summary_split`` (summary_headroom) may
        # instead return ``(source, tail)``: the summary covers ``source`` and ``tail`` stays verbatim after it.
        before = getattr(owner, "before_trim", None)
        if before is not None:
            messages = before(messages)
        measure = getattr(owner, "trigger_tokens", None) or owner.estimate_prompt_tokens
        budget = max(1, self._context_budget_tokens - max(0, extra_safety_tokens))
        if getattr(owner, "drain", None) is not None:
            # hero-v2 drain_half (arena.context_drain): drop the oldest history blocks instead of a summary. The
            # public loop passes ``extra_safety_tokens`` only on its context-length retry: that drain is forced.
            return owner.drain_context(self, messages, tools, measure, budget, force=extra_safety_tokens > 0)
        count = measure(messages, tools)
        if count <= budget:
            return messages
        system_message = messages[0]
        keep = max(1, int(getattr(owner, "compaction_keep_observations", 1)))
        prefix, tail = split_at_latest_observation(list(messages[1:]), keep=keep)
        if not prefix and len(tail) <= 1:
            return messages  # nothing before the latest observation: cannot compact
        split = getattr(owner, "summary_split", None)
        plan = None if split is None else split(messages, tools, count, budget)
        if plan is not None:
            source, kept = plan
            assert source[0] is system_message and messages[len(messages) - len(kept):] == kept
            return compacted_messages(system_message, owner.summarize(source), kept)
        source = getattr(owner, "summary_source", None)
        source = messages if source is None else source(messages, tools, count)
        summary = owner.summarize(source)
        kept = tail
        fits = measure([system_message, summary_message(summary), *kept], tools) <= budget
        if not fits and keep > 1:
            _, kept = split_at_latest_observation(list(messages[1:]), keep=1)
            fits = measure([system_message, summary_message(summary), *kept], tools) <= budget
        if not fits:
            if len(source) < len(messages) and len(kept) > 1:
                cut = getattr(owner, "note_tail_cut", None)
                if cut is not None:
                    cut(len(kept) - 1)
            kept = kept[:1]  # the tool rounds after the observation are in the summary too
        return compacted_messages(system_message, summary, kept)

    def _force_reduce_messages(self, messages, *, preserve_recent=1):
        return self._trim_messages_for_context(messages, extra_safety_tokens=512)

    def _persistent_history_messages(self, messages, *, tools=None):
        return list(messages[1:])  # every turn survives; no 30-turn cap, no stubs


def bind_compaction(analyzer, owner):
    """Bind the mixin's methods onto an existing ToolAgent instance."""
    analyzer._owner = owner
    for name in (
        "_trim_messages_for_context",
        "_force_reduce_messages",
        "_persistent_history_messages",
    ):
        setattr(analyzer, name, MethodType(getattr(CompactionMixin, name), analyzer))
    return analyzer


def snapshot(messages):
    """A deep copy for the append-only tests."""
    return copy.deepcopy(messages)
