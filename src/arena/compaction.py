"""Append-only contexts, drained at the budget (hero-v2 drain_half, ``arena.context_drain``).

The public ToolAgent (and duck_transfer's trimmer on top of it) drops or stubs old messages, which rewrites the prefix
on every turn. In the arena a request's message list is only ever extended until it reaches the budget; then the
the ``drain_context`` drops the oldest history blocks. ``CompactionMixin`` methods are bound onto a ToolAgent
instance (``MethodType``) so the public analyze loop calls them in the places it would otherwise trim.

rl-min: the model-written summary compaction (the path without drain_half) is gone; ``ArenaDuckMixin.attach`` refuses
a study without drain_half, and the trim raises if it is ever reached without one.
"""

from types import MethodType


class CompactionMixin:
    """Replacements for the public ToolAgent's trimming; needs ``_owner`` with ``estimate_prompt_tokens(messages,
    tools)`` and ``drain_context`` (drain_half: drops the oldest blocks, append-only between drains)."""

    _owner = None

    def _trim_messages_for_context(
        self, messages, *, tools=None, preserve_recent=1, extra_safety_tokens=0
    ):
        if not messages:
            return []
        owner = self._owner
        # Optional owner hooks (the arena's harness_fixes; absent or identity otherwise): ``before_trim`` may replace
        # the not-yet-sent last message (the cap-hit follow-up), ``trigger_tokens`` measures the would-be request
        # (the exact count instead of the estimate).
        before = getattr(owner, "before_trim", None)
        if before is not None:
            messages = before(messages)
        measure = getattr(owner, "trigger_tokens", None) or owner.estimate_prompt_tokens
        budget = max(1, self._context_budget_tokens - max(0, extra_safety_tokens))
        if getattr(owner, "drain", None) is None:
            raise RuntimeError("context trim without drain_half: the summary compaction is gone (rl-min)")
        # The public loop passes ``extra_safety_tokens`` only on its context-length retry: that drain is forced.
        return owner.drain_context(self, messages, tools, measure, budget, force=extra_safety_tokens > 0)

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
