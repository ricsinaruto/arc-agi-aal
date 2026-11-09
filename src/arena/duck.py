"""The arena's Duck agent: the corrected Duck (duck_transfer) run through ``arc_infra.session.GameSession``, plus
the game's active clock, the KV gate (``client.slots`` is a ``GateSlot``), level reports, append-only compaction and
the recipe's harness layers (harness_fixes, drain_half, dfr_harness, dfr_loop, hx_harness, loop_rule)."""

import asyncio
import concurrent.futures
import dataclasses
import hashlib
import json
import re
import time


from arc_infra.model_templates import request_messages, template_kwargs
from duck_transfer.agent import (
    ContextLengthRejected,
    GameplayRequestExpired,
)
from duck_transfer.agent import DuckAgent as TransferDuckAgent

from . import dfr_harness as dfr_module
from . import phase as phase_time
from . import dfr_loop as loop_module
from . import harness_fixes as fixes_module
from .compaction import bind_compaction

ESTIMATE_HEADROOM = 1.1


class ArenaDuckMixin:
    """Shared arena behaviour; ``attach(game, window, cap)`` must run before ``run()``."""

    game = None
    window = None
    cap = None
    loop = None
    now = staticmethod(phase_time.monotonic)  # train phases (arena.phase) never show as try time

    # The study blocks (attach requires every one; rl-min): harness_fixes, drain_half, dfr_harness, dfr_loop,
    # hx_harness and loop_rule.
    harness_fixes = None
    last_gameplay_finish = None
    gameplay_replies = 0
    gameplay_completion_tokens = 0
    # drain_half (arena.context_drain): the settings, per-game totals and the pending follow-up rewrite of a reply
    # whose tool calls were dropped as malformed.
    drain = None
    drain_stats = None
    malformed_pending = None
    rejected_prompt_tokens = None
    dfr = None
    dloop = None
    hx = None
    hx_reflector = None
    lrule = None
    lrule_detector = None
    lrule_pending = None
    lrule_stats = None
    def attach(self, game, *, window, cap, harness_fixes=None, context_drain=None, dfr_harness=None, dfr_loop=None,
               hx_harness=None, loop_rule=None):
        self.game, self.window, self.cap = game, int(window), int(cap)
        self._arena_analyzer = None
        self.env_lock = asyncio.Lock()
        self.competition_clock = None  # no planning-budget notes in the arena
        blocks = {"harness_fixes": harness_fixes, "drain_half": context_drain, "dfr_harness": dfr_harness,
                  "dfr_loop": dfr_loop, "hx_harness": hx_harness, "loop_rule": loop_rule}
        missing = sorted(k for k, v in blocks.items() if not v)
        if missing:
            # rl-min: the arena agent runs the reference harness only (drain_half is the only context policy: the
            # model-written summary compaction is gone); every off path of these blocks is removed.
            raise ValueError(f"the arena agent needs the study blocks {missing}")
        self.apply_harness_fixes(harness_fixes)
        self.apply_context_drain(context_drain)
        self.apply_dfr_harness(dfr_harness)
        self.apply_dfr_loop(dfr_loop)
        self.apply_hx_harness(hx_harness)
        self.apply_loop_rule(loop_rule)
        return self

    # --- v7.1 strict degenerate-loop rule (arena.loop_rule) ---------------------------------------------------------

    def apply_loop_rule(self, value):
        from .loop_rule import LoopDetector
        from .loop_rule import settings as loop_rule_settings

        spec = loop_rule_settings(value)
        self.lrule = spec
        self.lrule_detector = LoopDetector(spec)
        self.lrule_pending = None
        self.lrule_stats = {"fires": 0, "applied": 0, "tokens_freed": 0}

    def note_loop_rule(self, reply):
        """The detector on one gameplay reply (event loop); a fire is logged and, under fresh_context, applied at the
        next request's trim (``loop_fresh_context``)."""
        fire = self.lrule_detector.note(self.steps, (reply.get("usage") or {}).get("completion_tokens"),
                                        reply.get("message"))
        if fire is None:
            return
        self.lrule_stats["fires"] += 1
        level = max(1, min(self.obs.total_levels, self.obs.levels + 1)) if self.obs is not None else None
        self.workspace.append("loop_rule", game=self.game_id, level=level, action=self.lrule["action"], **fire)
        self.lrule_pending = {**fire, "level": level}

    def loop_fresh_context(self, messages, tools, measure):
        """``[system, note, current observation]`` for a pending fire (analyzer thread), else None."""
        from .context_drain import DRAIN_NOTE
        from .loop_rule import LOOP_NOTE, fresh_context

        fire, self.lrule_pending = self.lrule_pending, None
        kept = fresh_context(list(messages[1:]), getattr(self, "is_observation", None))
        if not kept or len(kept) == len(messages) - 1:
            return None
        before = measure(messages, tools)
        dropped = len(messages) - 1 - len(kept)
        note = self.drain_note(dropped, 0)  # the python note and inventory after the drain sentence
        rest = note["content"][len(DRAIN_NOTE.format(messages=dropped, tokens=0)):]
        note = {**note, "content": (LOOP_NOTE.format(streak=fire["streak"]) + rest.lstrip()).rstrip()}
        result = [messages[0], note, *kept]
        after = measure(result, tools)
        self.lrule_stats["applied"] += 1
        self.lrule_stats["tokens_freed"] += max(0, int(before) - int(after))
        self.workspace.append("loop_rule_fresh_context", game=self.game_id, level=fire["level"], streak=fire["streak"],
                              tokens_before=int(before), tokens_after=int(after),
                              tokens_freed=max(0, int(before) - int(after)), messages_dropped=len(messages) - len(result) + 1)
        return result

    # --- harness arm hx_harness (arena.hx_harness) -----------------------------------------------------------------

    def apply_hx_harness(self, value):
        """Switch the harness arm's components on (``arena.hx_harness``); after dfr_harness, whose sandbox action
        wrapper the toolbox wraps."""
        from . import hx_harness as hx_module

        self.hx = hx_module.settings(value)
        if self.sandbox.process is not None:
            raise RuntimeError("hx_harness must be configured before the interpreter starts")
        self.sandbox.bootstrap = hx_module.with_toolbox(self.sandbox.bootstrap, self.hx)
        self.hx_reflector = hx_module.Reflector(self.hx)
        self.hx_econ = None
        self.hx_memo_level = None
        self.hx_stats = {"toolbox_turns": 0, "max_tools": 0, "max_observers": 0, "observer_errors": 0,
                         "reflect_notices": 0}

    def hx_turn_lines(self):
        """The hx_harness additions to a new user turn (the STEP BACK block, then the toolbox listing)."""
        from . import hx_harness as hx_module

        lines = []
        self.hx_reflector.update(self.duck_history, self.transition_results, self.obs.levels,
                                 self.gameplay_replies, self.gameplay_completion_tokens)
        level = max(1, min(self.obs.total_levels, self.obs.levels + 1))
        notice = self.hx_reflector.notice(level)
        if notice:
            self.hx_stats["reflect_notices"] += 1
            self.workspace.append("hx_reflect", step=self.steps, reason=self.hx_reflector.last_reason,
                                  requests=self.hx_reflector.requests)
            lines.append(notice)
        if self.hx_econ is None or self.hx_econ[0] != self.obs.levels:
            self.hx_econ = (self.obs.levels, self.steps)  # the level's actions count from its first turn
        lines.append(hx_module.economy_line(self.steps - self.hx_econ[1], level))
        state = self.sandbox.hx_state or {}
        # v8.2 playbook: shown in full at a level start (and in every cut note, ``drain_note``); one reminder
        # before a cut when it was not written since the last one.
        text = str(state.get("playbook") or "")
        writes = int(state.get("playbook_writes") or 0)
        self.hx_stats.update(playbook_chars=len(text), playbook_writes=writes,
                             playbook_max_chars=max(self.hx_stats.get("playbook_max_chars", 0), len(text)))
        cleared = self.hx_memo_level is not None and self.obs.levels > self.hx_memo_level
        if cleared:
            lines.append(hx_module.PLAYBOOK_LEVEL)
            lines.append(hx_module.playbook_block(state))
            self.hx_stats["playbook_shown"] = self.hx_stats.get("playbook_shown", 0) + 1
        self.hx_memo_level = self.obs.levels
        budget = getattr(getattr(self, "_arena_analyzer", None), "_context_budget_tokens", None)
        fill = (self.drain_stats["drains"], self.lrule_stats["applied"])
        cut_writes = getattr(self, "hx_playbook_cut", (None, 0))
        if cut_writes[0] != fill:
            cut_writes = self.hx_playbook_cut = (fill, writes)  # writes when this fill began
        if (budget and self.last_prompt_tokens >= 0.8 * budget and writes == cut_writes[1]
                and getattr(self, "hx_playbook_due", None) != fill):
            self.hx_playbook_due = fill
            self.hx_stats["playbook_due"] = self.hx_stats.get("playbook_due", 0) + 1
            lines.append(hx_module.PLAYBOOK_OPEN_DUE)
        self.hx_stats["max_tools"] = max(self.hx_stats["max_tools"], len(state.get("tools") or {}))
        self.hx_stats["max_observers"] = max(self.hx_stats["max_observers"], len(state.get("observers") or {}))
        self.hx_stats["observer_errors"] = max(self.hx_stats["observer_errors"], int(state.get("errors") or 0))
        listing = hx_module.toolbox_lines(state, self.hx)
        if listing:
            self.hx_stats["toolbox_turns"] += 1
            lines.append(listing)
        return lines

    # --- hero-v2 drain_half (arena.context_drain) -------------------------------------------------------------------

    def apply_context_drain(self, value):
        from .context_drain import settings as drain_settings

        self.drain = drain_settings(value)
        self.malformed_pending = None
        self.drain_stats = {
            "drains": 0, "forced": 0, "turn_cap_drains": 0, "tokens_dropped": 0, "messages_dropped": 0,
            "turn_exchanges": 0, "malformed_replies": 0, "notes": 0, "hook_failures": 0}

    def configure_drain_analyzer(self, analyzer):
        """drain_half's tool-output settings on a new analyzer: ``tool_output_tokens`` (the system prompt's "capped to
        about N tokens" line follows) and the head/tail middle truncation."""
        spec = self.drain
        from .context_drain import middle_truncate

        old, new = analyzer._tool_output_tokens, int(spec["tool_output_tokens"])
        if new != old:
            claim = "Tool responses are capped to about {} tokens."
            if analyzer._system_prompt.count(claim.format(old)) != 1:
                raise ValueError("Pinned tool-output line of the Duck system prompt changed")
            analyzer._system_prompt = analyzer._system_prompt.replace(claim.format(old), claim.format(new))
            analyzer._tool_output_tokens = new
            analyzer._tool_output_chars = max(256, new * 4)
        analyzer._trim_tool_text = lambda text: middle_truncate(text, analyzer._tool_output_chars)

    def drain_context(self, analyzer, messages, tools, measure, budget, *, force=False):
        """``CompactionMixin._trim_messages_for_context`` under drain_half: ``messages`` itself when it fits (and the
        turn cap does not fire), else ``[system, (note), *suffix]`` with the oldest blocks dropped; the drain is
        recorded, and the scheduler's ``GameRecord.note_context_drain`` hook (when present) hears of it."""
        from .context_drain import drain_history

        if len(messages) < 2:
            return messages
        if self.lrule_pending is not None:
            fresh = self.loop_fresh_context(messages, tools, measure)
            if fresh is not None:
                return fresh
        count = measure(messages, tools)
        if force:
            # The rejected request's exact count when the arena admission refused it (the estimate may say it fits).
            count = max(count, int(self.rejected_prompt_tokens or 0))
            self.rejected_prompt_tokens = None
        system, history = messages[0], list(messages[1:])

        def estimate(items):
            return analyzer._estimate_request_input_tokens(items)

        is_observation = getattr(self, "is_observation", None)
        kept, info = drain_history(history, count=count, budget=budget, estimate=estimate, spec=self.drain,
                                   force=force, is_observation=is_observation)
        if info is None:
            return messages
        # The prefix changes here anyway (dfr_history); the current observation keeps its extra images.
        from .context_drain import latest_observation

        current = latest_observation(kept, is_observation)
        if current is not None:
            kept = [*self.dfr_history(kept[:current]), *kept[current:]]
        tokens_dropped = max(0, estimate(history) - estimate(kept))
        note = self.drain_note(info["messages_dropped"], tokens_dropped)
        result = [system, note, *kept]
        after = measure(result, tools)
        rounds = 1
        while after > budget and rounds < 4:
            # The estimate over-counted what was dropped: drain the kept messages again from the measure, the estimate
            # rescaled by the measured drop (the note stays in front).
            previous = kept
            measured, estimated = count - after, estimate(history) - estimate(previous)
            scale = measured / estimated if measured > 0 and estimated > 0 else 1.0

            def scaled(items, scale=scale):
                return int(estimate(items) * scale)

            kept, extra = drain_history(previous, count=after, budget=budget, estimate=scaled, spec=self.drain,
                                        force=True, is_observation=is_observation)
            if extra is None:
                break
            result, rounds = [system, note, *kept], rounds + 1
            info["messages_dropped"] += extra["messages_dropped"]
            info["turn_exchanges"] += extra["turn_exchanges"]
            after = measure(result, tools)
        stats = self.drain_stats
        stats["drains"] += 1
        stats["forced"] += info["trigger"] == "forced"
        stats["turn_cap_drains"] += bool(info["turn_cap_messages"])
        stats["tokens_dropped"] += max(0, int(count) - int(after))
        stats["messages_dropped"] += info["messages_dropped"]
        stats["turn_exchanges"] += info["turn_exchanges"]
        stats["notes"] += note is not None
        # v8.2 (uncapped playbook): drains that left more than the target, before the note (nothing more to drop)
        # and with it (the note, playbook included, is what exceeds it)
        if int(info.get("estimated_after") or 0) > int(info["target"]):
            stats["short"] = stats.get("short", 0) + 1
        if int(after) - int(info["target"]) > max(4096, int(info["target"]) // 4):
            stats["over_target"] = stats.get("over_target", 0) + 1
        self.workspace.append("context_drain", step=self.steps, tokens_before=int(count), tokens_after=int(after),
                              rounds=rounds, note_chars=len(note["content"]),
                              **{k: v for k, v in info.items() if k != "prompt_tokens"})
        return result

    def on_loop(self, fn, *args, **kwargs):
        """``fn(*args, **kwargs)`` on the agent's event loop, waited for from the analyzer thread (directly when there
        is no running loop or this already is the loop's thread)."""
        loop = self.loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is None or running is loop or not loop.is_running():
            result = fn(*args, **kwargs)
            if not asyncio.iscoroutine(result):
                return result
            if running is not None:
                result.close()
                raise RuntimeError("an async drain step cannot wait on its own event loop thread")
            return asyncio.run(result)

        async def call():
            result = fn(*args, **kwargs)
            return await result if asyncio.iscoroutine(result) else result

        future = asyncio.run_coroutine_threadsafe(call(), loop)
        self.chat_future = future
        try:
            return future.result()
        finally:
            self.chat_future = None

    def drain_note(self, messages, tokens):
        """The user message placed before the kept history: what was dropped, that the persistent interpreter
        survives, the playbook and the harness_fixes inventory of live definitions."""
        from . import hx_harness as hx_module
        from .context_drain import DRAIN_NOTE, PYTHON_NOTE

        text = DRAIN_NOTE.format(messages=messages, tokens=tokens) + PYTHON_NOTE
        state = self.sandbox.hx_state or {}
        text += "\n\n" + hx_module.playbook_block(state)
        self.hx_stats["playbook_shown"] = self.hx_stats.get("playbook_shown", 0) + 1
        self.workspace.append("playbook_shown", step=self.steps, chars=len(str(state.get("playbook") or "")),
                              writes=int(state.get("playbook_writes") or 0))
        text += "\n\n" + self.on_loop(self.inventory_block)
        return {"role": "user", "content": text}

    def is_observation(self, message):
        """A turn's observation (its board image or text label), not a follow-up user message."""
        from duck_transfer.prompting import _has_image_label

        return _has_image_label(message)

    def without_malformed_calls(self, result):
        """drain_half ``drop_malformed`` (dfranzen's hygiene): a reply with any tool call whose arguments are not valid
        JSON loses all its calls (half a plan is worse than none; the server re-parses history calls, so one malformed
        call would break every later request). Content and reasoning stay; the public loop then asks again."""
        from .context_drain import malformed_calls

        errors = malformed_calls(result.message)
        if not errors:
            return result
        message = {k: v for k, v in result.message.items() if k != "tool_calls"}
        self.malformed_pending = "length" if str(result.finish_reason or "").strip() == "length" else "json"
        self.drain_stats["malformed_replies"] += 1
        self.workspace.append("malformed_tool_calls_dropped", step=self.steps, calls=len(result.message["tool_calls"]),
                              finish_reason=result.finish_reason, errors=[e[:200] for e in errors])
        return dataclasses.replace(result, message=message)

    def malformed_followup(self, messages):
        """The public follow-up after a reply whose calls were dropped names why (cut off, or invalid JSON)."""
        pending, self.malformed_pending = self.malformed_pending, None
        if pending is None or not messages:
            return messages
        last = messages[-1]
        content = last.get("content")
        if last.get("role") != "user" or not isinstance(content, str) or not content.startswith(
                fixes_module.PUBLIC_FOLLOWUP_PREFIX):
            return messages
        from .context_drain import MALFORMED_CALL_FOLLOWUP, TRUNCATED_CALL_FOLLOWUP

        prefix = TRUNCATED_CALL_FOLLOWUP if pending == "length" else MALFORMED_CALL_FOLLOWUP
        text = prefix + content[len(fixes_module.PUBLIC_FOLLOWUP_PREFIX):]
        return [*messages[:-1], {"role": "user", "content": text}]

    # --- hero-v2 dfr_harness (arena.dfr_harness) ----------------------------------------------------------------

    def apply_dfr_harness(self, value):
        """Switch the ported dfranzen harness features on (``arena.dfr_harness``); after the harness fixes and the
        action plans, which replace or extend the sandbox bootstrap first."""
        self.dfr = dfr_module.settings(value)
        if self.sandbox.process is not None:
            raise RuntimeError("dfr_harness must be configured before the interpreter starts")
        self.sandbox.bootstrap = dfr_module.with_dfr(self.sandbox.bootstrap)
        self.sandbox.animation_handler = self.dfr_animation
        self.dfr_stats = {"diff_images": 0, "gameover_turns": 0, "gameover_images": 0, "noop_stops": 0,
                          "stale_refusals": 0, "terminal_refusals": 0, "animated_actions": 0,
                          "transient_actions": 0, "animation_reads": 0, "animation_openers": 0, "level_prompts": 0,
                          "history_images_stubbed": 0}
        self.dfr_prev_frame = None      # (step, level, grid) of the previous analyzer turn's observation
        self.dfr_last_action = None     # the last executed action: step, label, animation, gameplay_changed
        self.dfr_opener_step = None     # the step whose animation line was last shown
        self.dfr_gameover_shown = None  # the reset step whose game-over block was shown
        self.dfr_gameover_images = []   # (label, url) for the coming observation
        self.dfr_noop_stops = []        # batch no-op stops since the last user turn
        self.dfr_guards_seen = 0        # sandbox guard refusals already reported
        self.animation_record = None    # the last executed action's animation (their animation_record)

    def valid_actions(self):
        return [dfr_module.UNDO[1] if name == dfr_module.UNDO[0] else name for name in super().valid_actions()]

    def engine_action(self, label):
        if label == dfr_module.UNDO[1]:
            return dfr_module.UNDO[0]
        return super().engine_action(label)

    def model_action(self, name):
        if name == dfr_module.UNDO[0]:
            return dfr_module.UNDO[1]
        return super().model_action(name)

    def batch_stop(self, requested, position, verdicts):
        stop = dfr_module.batch_stop(requested, position, verdicts)
        if stop is not None:
            self.dfr_stats["noop_stops"] += 1
            self.dfr_noop_stops.append(stop)
            self.workspace.append("dfr_noop_stop", step=self.steps, action=stop["no_op_action"],
                                  index=stop["no_op_action_index"], skipped=stop["skipped_actions"])
        return stop

    def dfr_annotate(self, action, label, payload, before):
        """Interior ``gameplay_changed`` (guards, animation) and the action's animation record (animation)."""
        payload["gameplay_changed"] = dfr_module.gameplay_changed(before, self.obs.board)
        if action.name != "RESET":
            # Their solver: the auto-reset after a death never touches the record (it does not pass through here).
            (start,) = dfr_module.normalize_frames([before])
            chain = dfr_module.collapse_frames(start, dfr_module.normalize_frames(self.obs.frames))
            summary = dfr_module.summarize_animation(chain)
            self.animation_record = ({"action_display": label, "chain": chain, "summary": summary,
                                      "step": self.steps, "level": payload.get("level")} if summary else None)
            if summary:
                if getattr(self, "animation_store", None) is None:
                    self.animation_store = dfr_module.AnimationStore()
                try:
                    self.animation_store.put(self.animation_record)
                except (ValueError, TypeError):  # frames of unequal shape: not stored
                    self.dfr_stats["animation_unstored"] = self.dfr_stats.get("animation_unstored", 0) + 1
            if summary:
                payload["animation"] = summary
                self.dfr_stats["animated_actions"] += 1
                self.dfr_stats["transient_actions"] += bool(summary.get("transient_pixels"))
                if summary.get("transient_pixels") and payload.get("gameplay_changed") is False:
                    payload["transient_effect"] = int(summary["transient_pixels"])
        self.dfr_last_action = {"step": self.steps, "label": label, "animation": payload.get("animation"),
                                "gameplay_changed": payload.get("gameplay_changed")}

    def dfr_animation(self, request):
        """The sandbox's lazy ``last_animation_frames`` / ``last_animation_timeline`` (analyzer thread)."""
        self.dfr_stats["animation_reads"] += 1
        if request.get("step") is not None:
            return self.dfr_animation_history(request)
        if request.get("kind") == "frames":
            return dfr_module.animation_frames(self.animation_record, self.dfr_frame_payload)
        return dfr_module.animation_timeline(self.animation_record)

    def dfr_animation_history(self, request):
        """``animation_frames(x)`` / ``animation_timeline(x)`` / ``animated_steps()`` (analyzer thread)."""
        store = getattr(self, "animation_store", None)
        self.dfr_stats["animation_history_reads"] = self.dfr_stats.get("animation_history_reads", 0) + 1
        kind, step = request.get("kind"), int(request["step"])
        if kind == "steps":
            return store.steps() if store is not None else []
        record = store.get(step) if store is not None else None
        if record is None:
            if store is not None and store.dropped and store.items and step < min(store.items):
                return {"unavailable": f"the animation of step {step} is no longer stored (the oldest are dropped; "
                                       f"stored from step {min(store.items)})"}
            return [] if kind == "frames" else {"error": f"No animation is stored for step {step}: that action "
                                                         "returned a single frame, or the step does not exist."}
        if kind == "frames":
            return dfr_module.animation_frames(record, self.dfr_frame_payload)
        return dfr_module.animation_timeline(record)

    def dfr_frame_payload(self, grid, step, level):
        from inference.agent.runtime_state import Frame

        return self.module._ascii_frame_view_payload(Frame(grid=tuple(grid), step=int(step), level=int(level)))

    def dfr_guard_stops(self):
        """The sandbox's guard refusals since the last user turn (counted once)."""
        guards = self.sandbox.dfr_guards
        fresh, self.dfr_guards_seen = guards[self.dfr_guards_seen:], len(guards)
        for item in fresh:
            key = "stale_refusals" if item.get("kind") == "stale_state" else "terminal_refusals"
            self.dfr_stats[key] = self.dfr_stats.get(key, 0) + 1
        return fresh

    def dfr_rewrite(self, text, current_frame):
        """The user-turn additions: the new-level prompt, the animation opener and the guard notes after the executed
        actions, and on the first turn after a GAME_OVER the diagnosis block (its images go with the observation)."""
        lines = text.split("\n")
        if lines.count(dfr_module.PUBLIC_LEVEL_LINE) == 1:
            # his new-level prompt (no_world_model: no carried notes to point at)
            lines[lines.index(dfr_module.PUBLIC_LEVEL_LINE)] = dfr_module.LEVEL_START_USER_PROMPT
            self.dfr_stats["level_prompts"] += 1
        reset = self.auto_reset
        game_over = (reset is not None and reset.get("reset_step") != self.dfr_gameover_shown
                     and any(line.startswith("GAME_OVER at step ") for line in lines))
        last = self.dfr_last_action
        notes = []
        if not game_over and last is not None and last["step"] != self.dfr_opener_step and reset is None:
            line = dfr_module.describe_animation(last["animation"], action_display=last["label"],
                                                 gameplay_changed=last["gameplay_changed"])
            self.dfr_opener_step = last["step"]
            if line:
                notes.append(line)
                self.dfr_stats["animation_openers"] += 1
        notes.extend(dfr_module.guard_notes(self.dfr_noop_stops, self.dfr_guard_stops()))
        self.dfr_noop_stops = []
        executed = next((i for i, line in enumerate(lines) if line.startswith("Executed actions")), None)
        if notes and executed is not None:
            lines[executed + 1:executed + 1] = notes
        self.dfr_gameover_images = []
        v2 = game_over and bool(self.death_log)
        if v2:
            # One text in place of the reset line, the diagnosis, the fatal-action line and the no-replay line.
            at = next(i for i, line in enumerate(lines) if line.startswith("GAME_OVER at step "))
            fatal = (reset.get("actions") or ["the last action"])[-1]
            lines[at] = fixes_module.death_text(self.death_log, step=reset.get("step"), fatal=str(fatal),
                                                level=max(1, self.obs.levels + 1))
            self.fix_stats["death_texts"] = self.fix_stats.get("death_texts", 0) + 1
        if game_over:
            self.dfr_gameover_shown = reset.get("reset_step")
            block = []
            if not v2:
                block.append(dfr_module.GAME_OVER_DIAGNOSIS)
            if last is not None:
                line = dfr_module.describe_animation(last["animation"], action_display=last["label"],
                                                     gameplay_changed=last["gameplay_changed"], game_over=True)
                self.dfr_opener_step = last["step"]
                if line:
                    block.append(line)
                    self.dfr_stats["animation_openers"] += 1
            fatal = (reset.get("actions") or [None])[-1]
            if fatal and not v2:
                block.append(dfr_module.FATAL_ACTION_LINE.format(action=str(fatal)))
            if not v2:
                block.append(dfr_module.NO_REPLAY_LINE)
            self.dfr_gameover_images = self.dfr_gameover_visuals()
            self.dfr_stats["gameover_turns"] += 1
            at = next(i for i, line in enumerate(lines) if line.startswith("GAME_OVER at step "))
            lines[at + 1:at + 1] = block
        return "\n".join(lines)

    def dfr_gameover_visuals(self):
        """Their death-turn images: the death frame and the fatal step's diff (history ends RESET, death, alive)."""
        entries = self.duck_history
        if len(entries) < 3 or entries[-1].action != "RESET" or entries[-2].action in ("", "RESET"):
            return []
        death, alive = entries[-2].frame.grid, entries[-3].frame.grid
        images = [(fixes_module.DEATH_IMAGE_LABEL, dfr_module.board_image_url(death))]
        fatal = dfr_module.diff_image_url(alive, death)
        if fatal is not None:
            images.append((dfr_module.FATAL_DIFF_LABEL, fatal))
        return images

    def observation_message(self, message, current_frame):
        """``vision``: the board image at the larger upscale, then the diff image or the game-over images."""
        if current_frame is None:
            return message
        content = message.get("content")
        boards = [i for i, part in enumerate(content or []) if isinstance(part, dict) and part.get("type") == "image_url"]
        if not isinstance(content, list) or len(boards) != 1:
            raise ValueError("dfr_harness: pinned Duck observation envelope changed")
        grid = current_frame.grid
        parts = list(content)
        parts[boards[0]] = dfr_module.image_part(dfr_module.board_image_url(grid))
        previous, self.dfr_prev_frame = self.dfr_prev_frame, (current_frame.step, current_frame.level, grid)
        extra, self.dfr_gameover_images = list(self.dfr_gameover_images), []
        if extra:
            self.dfr_stats["gameover_images"] += len(extra)
        elif previous is not None and previous[1] == current_frame.level and self.auto_reset is None:
            url = dfr_module.diff_image_url(previous[2], grid)
            if url is not None:
                extra = [(dfr_module.DIFF_LABEL.format(before=previous[0], after=current_frame.step), url)]
                self.dfr_stats["diff_images"] += 1
        for label, url in extra:
            parts.extend(({"type": "text", "text": label}, dfr_module.image_part(url)))
        return {**message, "content": parts}

    def dfr_history(self, messages):
        """``vision``: the extra images (diff, game-over) of the turns in ``messages`` become a one-line placeholder.

        Only at a context eviction: stubbing the previous turn's diff image on every request
        rewrote the history one turn back, so the hybrid model's prefix cache fell back to the shared system prompt
        on 37% of the new-observation requests once the pool filled (the dfr-combined v2 Modal preview; Run B,
        append-only, ~0%). Between evictions the history is append-only and the images stay; a drain rewrites the
        prefix anyway, so the surviving turns' extra images are stubbed there."""
        out = []
        for message in messages:
            stubbed = dfr_module.stub_extra_images(message)
            if stubbed is not message:
                self.dfr_stats["history_images_stubbed"] += 1
            out.append(stubbed)
        return out

    # --- hero-v2 dfr_loop (arena.dfr_loop; v5) -------------------------------------------------------------------

    def apply_dfr_loop(self, value):
        """Switch the v5 loop flags on (``arena.dfr_loop``); after dfr_harness, whose sandbox code it wraps."""
        self.dloop = loop_module.settings(value)

    def loop_analyzer(self, analyzer):
        """The v5 system prompt (after every other system-prompt edit): the drain line (prompt_fix) and his
        world-model-free text (no_world_model; the analyzer parses, carries and shows no world model)."""
        text = analyzer._system_prompt
        if text.count(loop_module.COMPACTION_LINE) != 1:
            raise ValueError("dfr_loop: pinned image/history system line changed")
        text = text.replace(loop_module.COMPACTION_LINE, loop_module.DRAIN_LINE, 1)
        analyzer._system_prompt = loop_module.world_model_free(text)

    # --- hero-v2 harness_fixes (arena.harness_fixes) -----------------------------------------------------------

    def apply_harness_fixes(self, value):
        self.harness_fixes = fixes_module.settings(value)
        self.count_cache = {}
        self.fix_stats = {"exact_trigger_counts": 0, "exact_trigger_fallbacks": 0, "admission_count_reused": 0,
                          "inventories": 0, "inventory_chars": 0, "cap_followups": 0, "cap_tool_notes": 0}
        clock = self.game.clock
        self.stuck = fixes_module.ActiveStuckTracker(
            self.config, active_seconds=clock.active_seconds, completion_tokens=clock.completion_tokens,
            rule=self.harness_fixes)
        from inference.agent.python_tool_sandbox import _SANDBOX_BOOTSTRAP

        self.sandbox.configure_fixes(_SANDBOX_BOOTSTRAP, grace=self.harness_fixes["interrupt_grace_seconds"])

    def fixed_system_prompt(self, text):
        """The system prompt with the ``prompt_text``, ``sandbox``, ``timeout_interrupt`` and ``solvable_line``
        corrections."""
        text = fixes_module.replace_pinned(text, fixes_module.SYSTEM_PROMPT_FIXES, "system prompt")
        text = fixes_module.replace_pinned(text, fixes_module.SANDBOX_PROMPT_FIXES, "system prompt")
        text = fixes_module.replace_pinned(text, fixes_module.INTERRUPT_PROMPT_FIXES, "system prompt")
        anchor = fixes_module.SOLVABLE_ANCHOR
        if text.count(anchor) != 1:
            raise ValueError("harness_fixes solvable_line: pinned system prompt line changed")
        return text.replace(anchor, anchor + fixes_module.SOLVABLE_LINE, 1)

    def rewrite_user_prompt(self, text, current_frame=None):
        """``prompt_text``: drop the closing "end of world model." line (no model is carried), and number the state
        line's step as the image label does (actions executed so far); then the dfr_harness additions."""
        lines = text.split("\n")
        if lines.count(fixes_module.EMPTY_WORLD_MODEL_LINE) == 1:
            lines.remove(fixes_module.EMPTY_WORLD_MODEL_LINE)
        text = "\n".join(lines)
        step = current_frame.step if current_frame is not None else 0
        text = STATE_STEP.sub(f"Current state: step {step},", text, count=1)
        return self.dfr_rewrite(text, current_frame)

    def before_trim(self, messages):
        """The not-yet-sent follow-up's rewrites: ``cap_followup`` and drain_half's malformed-call follow-up."""
        messages = self.cap_followup(messages)
        if self.malformed_pending is not None:
            messages = self.malformed_followup(messages)
        if messages:
            last = messages[-1]
            content = last.get("content")
            if last.get("role") == "user" and isinstance(content, str) and content.startswith(
                    fixes_module.PUBLIC_FOLLOWUP_PREFIX):
                # The public "You have not acted yet ... call action(actions)" follow-up (prompt audit item 1).
                messages = [*messages[:-1], {"role": "user", "content": fixes_module.TOOL_CALL_FOLLOWUP}]
        if messages:
            last = messages[-1]
            content = last.get("content")
            if last.get("role") == "user" and isinstance(content, str) and loop_module.FOLLOWUP_LABELS in content:
                messages = [*messages[:-1], {**last, "content": content.replace(loop_module.FOLLOWUP_LABELS, "", 1)}]
        return messages

    def cap_followup(self, messages):
        """``cap_followup``: the public follow-up after a reply that hit the completion cap says "You have not acted
        yet. Investigate first."; it is replaced (not yet sent, so the cached prefix is unchanged)."""
        if not messages or self.last_gameplay_finish != "length":
            return messages
        last = messages[-1]
        content = last.get("content")
        if last.get("role") != "user" or not isinstance(content, str) or not content.startswith(
                fixes_module.PUBLIC_FOLLOWUP_PREFIX):
            return messages
        self.last_gameplay_finish = None
        self.fix_stats["cap_followups"] += 1
        self.workspace.append("cap_hit_followup", step=self.steps, cap=self.cap)
        text = fixes_module.CAP_FOLLOWUP
        return [*messages[:-1], {"role": "user", "content": text.format(cap=self.cap)}]

    def trigger_tokens(self, messages, tools):
        """The drain trigger's measure: the exact count of the would-be gameplay request (``exact_trigger``), the
        calibrated estimate when the count fails."""
        future = asyncio.run_coroutine_threadsafe(self.exact_gameplay_tokens(messages, tools), self.loop)
        self.chat_future = future
        try:
            count = future.result()
        except (TimeoutError, asyncio.CancelledError, concurrent.futures.CancelledError, GameplayRequestExpired):
            raise
        except Exception as exc:  # noqa: BLE001 - a failed count falls back to the estimate
            self.fix_stats["exact_trigger_fallbacks"] += 1
            self.workspace.append("exact_trigger_fallback", step=self.steps, error=f"{type(exc).__name__}: {exc}"[:300])
            return self.estimate_prompt_tokens(messages, tools)
        finally:
            self.chat_future = None
        self.fix_stats["exact_trigger_counts"] += 1
        return count

    async def exact_gameplay_tokens(self, messages, tools):
        """The exact prompt tokens of the gameplay request ``messages`` would send: the admission's multimodal count
        (cached so the admission of the same payload reuses it), or the precheck count without a block budget."""
        payload = self.gameplay_payload(messages, tools, self.request_template_override())
        payload["messages"] = request_messages(self.config.model, payload["messages"])
        payload["chat_template_kwargs"] = template_kwargs(self.config.model, payload.get("chat_template_kwargs", {}))
        if self.client.slots.gate.budget is None:
            count, _, _, image_tokens = await self.count_prompt(payload["messages"], tools,
                                                                payload.get("chat_template_kwargs"))
            if count is None:
                raise RuntimeError("token count failed")
            return int(count + image_tokens + max(0, round(self.count_residual)))
        key = count_key(payload)
        if key in self.count_cache:  # the turn's first trim and the request's own trim see the same list
            return self.count_cache[key]
        count = await self.client.count_tokens(
            payload["messages"], time.monotonic() + self.remaining(), tools=payload.get("tools"),
            allow_tools=bool(payload.get("tools")), template_kwargs=payload.get("chat_template_kwargs"),
            text_images=True)
        self.count_cache = {key: int(count)}
        return int(count)

    async def inventory_block(self):
        """``inventory``: the harness-written block appended to the summary (live python definitions, last receipts)."""
        data = await asyncio.to_thread(self.sandbox.inventory)
        reset_note = None
        restarts = self.sandbox.restarts
        if restarts and restarts != getattr(self, "inventory_restarts_seen", 0):
            reset_note = (f"(The python interpreter was reset {restarts} time(s) in this game; definitions from before "
                          "the last reset are gone.)")
        self.inventory_restarts_seen = restarts
        receipts = fixes_module.receipt_lines(self.duck_history, self.transition_results,
                                              self.harness_fixes["inventory_receipts"])
        text = fixes_module.render_inventory(data, receipts, max_tokens=self.harness_fixes["inventory_max_tokens"],
                                             chars_per_token=self.chars_per_token, reset_note=reset_note)
        self.fix_stats["inventories"] += 1
        self.fix_stats["inventory_chars"] += len(text)
        self.workspace.append("compaction_inventory", step=self.steps, chars=len(text), live=data is not None,
                              functions=len((data or {}).get("functions") or []),
                              values=len((data or {}).get("values") or []), receipts=len(receipts))
        return text

    def context_budget_tokens(self, analyzer):
        return max(1024, self.window - self.cap - analyzer._request_safety_margin_tokens)

    # --- clocks ---------------------------------------------------------------------------

    def remaining(self):
        # Requests are never cut by the level budget (an in-flight request finishes); the wall
        # clock is the only hard deadline. Parking blocks the next request in the gate.
        return self.game.clock.global_clock.remaining() if self.game else super().remaining()

    def runtime_message(self, *, record=True):
        clock = self.game.clock
        now = self.now()
        left = clock.budget_remaining()
        if clock.budget_clock == "tokens":
            # Token budgets (29 September): the remaining level budget is completion tokens, not seconds.
            budget = {"level_budget_remaining_tokens": None if left is None else int(left)}
        else:
            budget = {"level_budget_remaining_seconds": None if left is None else round(left, 1)}
        runtime = {
            "elapsed_seconds": round(clock.active_seconds(), 1),
            "remaining_seconds": round(clock.global_clock.remaining(), 1),
            "game_inference_seconds": round(clock.inference_seconds(), 1),
            "level_budget_clock": clock.budget_clock,
            **budget,
            "actions_executed": self.steps,
            "levels_completed": self.obs.levels if self.obs else 0,
            "seconds_since_last_observation": round(now - self.last_observation_at, 1),
            "analysis_replies_since_action": self.analysis_turns,
            "mean_model_request_seconds": round(self.model_seconds / max(1, self.turns), 1),
            "target_actions_per_900_seconds": self.config.action_target,
        }
        if record:
            self.workspace.append("model_request", step=self.steps, runtime=runtime)
        return {"role": "user", "content": "[Fresh runtime status]\n" + json.dumps(runtime)}

    async def step(self, action):
        before = self.obs.levels if self.obs is not None else 0
        await super().step(action)
        self.note_attempt(action, before)

    # death_turn_v2: the current attempt on the level (actions, those that changed a border-ring cell, the action
    # labels) and the level's deaths so far.
    attempt = None
    death_log = ()

    def note_attempt(self, action, levels_before):
        if action is None:  # a step with no action (stub drivers): nothing to count
            return
        if self.attempt is None:
            self.attempt, self.death_log = {"actions": 0, "edge": 0, "labels": []}, []
        if self.obs.levels != levels_before:  # the clearing action belongs to the level it cleared
            self.attempt, self.death_log = {"actions": 0, "edge": 0, "labels": []}, []
            return
        if action.name == "RESET":
            self.attempt = {"actions": 0, "edge": 0, "labels": []}
            return
        attempt = self.attempt
        attempt["actions"] += 1
        attempt["labels"].append(action.name + (f"@{action.x},{action.y}" if action.name == "ACTION6" else ""))
        old, new, ring = getattr(self.previous, "board", None), self.obs.board, dfr_module.BORDER
        if old is not None and old.shape == new.shape and min(new.shape) > 2 * ring:
            diff = old != new
            attempt["edge"] += bool(diff.any() and (diff.sum() > diff[ring:-ring, ring:-ring].sum()))
        if self.obs.state == "GAME_OVER":
            self.death_log.append({"actions": attempt["actions"], "edge": attempt["edge"],
                                   "sequence": hashlib.sha1("|".join(attempt["labels"]).encode()).hexdigest()})

    # --- requests -------------------------------------------------------------------------

    async def request_completion(self, payload):
        purpose = "summary" if not payload.get("tools") else "gameplay"
        if getattr(self, "sampling_override", None):
            # level_bench.sampling: the study's sampler on every request (gameplay, summaries, asides)
            payload = {**payload, **self.sampling_override}
        with self.game.clock.inference(purpose):
            return await self.with_server_retries(super().request_completion, payload)

    server_error_retries = 0

    async def with_server_retries(self, call, payload):
        """``call(payload)``, re-sent after an HTTP 5xx (v7.1, 6 October: a vLLM MultiModalCacheMissError 500 set
        ``backend_error`` and ended the game for the rest of the run) up to ``SERVER_ERROR_DELAYS`` times with backoff
        while the game has time; the last response is returned either way (a 4xx is never retried)."""
        for attempt, delay in enumerate((*SERVER_ERROR_DELAYS, None), 1):
            response = await call(payload)
            if (response.status_code < 500 or delay is None or self.stopping.is_set()
                    or self.remaining() <= delay + 1):
                return response
            self.server_error_retries += 1
            self.workspace.append("server_error_retry", step=self.steps, status=response.status_code, attempt=attempt,
                                  delay=delay, detail=response.text[:300])
            await asyncio.sleep(delay)


    def run_python_tool(self, execute, state_path, arguments):
        # Worker-thread tool execution (both Duck variants); the ``busy`` budget clock charges
        # it to the game. Was wrongly placed on the public variant only until 24 Sept 20:00 UTC.
        with self.game.clock.tool("python"):
            result = super().run_python_tool(execute, state_path, arguments)
        if self.last_gameplay_finish == "length":
            # A tool call cut off by the completion cap (hero-v2 k095: "python requires a non-empty `code` string").
            result = self.with_note(result, fixes_module.CAP_TOOL_NOTE.format(cap=self.cap))
            self.fix_stats["cap_tool_notes"] += 1
        return result

    @staticmethod
    def with_note(result, note):
        """``result`` (a dict or the bridged tool's ``_ToolDispatchResult``) with a ``note`` field."""
        if isinstance(result, dict):
            return {**result, "note": note}
        payload = json.loads(result.content)
        return dataclasses.replace(result, content=json.dumps({**payload, "note": note}, indent=2))

    def tried_no_effect_here(self):
        return [self.model_action(name) for name in super().tried_no_effect_here()]

    async def action_effect(self, action, label, payload, before, before_levels, changed):
        self.dfr_annotate(action, label, payload, before)
        return changed

    def turn_notice(self):
        """One-off notices in the next user turn (hx_harness)."""
        return "\n".join(self.hx_turn_lines()) or None

    def estimate_prompt_tokens(self, messages, tools):
        analyzer = self._analyzer
        estimate = analyzer._estimate_request_input_tokens(messages, tools=tools)
        return int(estimate * ESTIMATE_HEADROOM + max(0, round(self.count_residual)))

    async def chat(self, messages, tools):
        self.client.slots.reserve(self.estimate_prompt_tokens(messages, tools))
        return self.without_malformed_calls(await super().chat(messages, tools))

    async def prepare_admission(self, payload):
        """Count the outbound request exactly and reserve it; a request whose count plus ``max_tokens`` exceeds the
        window is rejected."""
        payload['messages'] = request_messages(self.config.model, payload['messages'])
        payload["chat_template_kwargs"] = template_kwargs(
            self.config.model, payload.get("chat_template_kwargs", {})
        )
        if self.client.slots.gate.budget is None:
            return None
        # Count actual multimodal messages, tools and reasoning template before waiting.
        # Arena timing is appended at message construction, not mutated during admission.
        cached = None
        if self.count_cache:
            # The trigger's exact count of this very payload (analyzer thread, just before this request).
            cached = self.count_cache.pop(count_key(payload), None)
            self.count_cache = {}
        if cached is not None:
            count = cached
            self.fix_stats["admission_count_reused"] += 1
        else:
            count = await self.client.count_tokens(
                payload["messages"], time.monotonic() + self.remaining(),
                tools=payload.get("tools"), allow_tools=bool(payload.get("tools")),
                template_kwargs=payload.get("chat_template_kwargs"), text_images=True,
            )
        if count + payload["max_tokens"] > self.window:
            self.rejected_prompt_tokens = int(count)  # drain_half's forced drain starts from it
            # The phrase is the public loop's context-length marker, so it trims and retries.
            raise ContextLengthRejected(
                "Exact arena request exceeds context window; reduce the length of the input prompt"
            )
        self.client.slots.reserve(count)
        self.admitted_count = int(count)
        self.workspace.append("exact_admission_count", step=self.steps, prompt_tokens=count,
                              output_allowance=payload["max_tokens"])
        text_messages = [
            {**m, "content": [{"type": "text", "text": "[image]"} if p.get("type") == "image_url" else p
                              for p in m["content"]]} if isinstance(m.get("content"), list) else m
            for m in payload["messages"]
        ]
        chars = len(json.dumps(text_messages, ensure_ascii=False))
        # Images are already counted by the serving processor, so never add them twice.
        return count, 0, chars, 0

    def record_reply(self, reply, purpose="gameplay"):
        super().record_reply(reply, purpose)
        admitted, self.admitted_count = getattr(self, "admitted_count", None), None
        served = (reply.get("usage") or {}).get("prompt_tokens")
        if admitted is not None and served:
            # text_count's check: the admission count against the served prompt.
            error = int(served) - admitted
            self.fix_stats["text_count_checked"] = self.fix_stats.get("text_count_checked", 0) + 1
            if error:
                self.fix_stats["text_count_mismatches"] = self.fix_stats.get("text_count_mismatches", 0) + 1
                self.fix_stats["text_count_max_error"] = max(abs(error), self.fix_stats.get("text_count_max_error", 0))
                self.workspace.append("text_count_mismatch", step=self.steps, admitted=admitted, served=int(served))
        if purpose == "gameplay":
            self.note_loop_rule(reply)
        if purpose == "gameplay":
            self.last_gameplay_finish = reply.get("finish_reason")
            self.gameplay_replies += 1
            self.gameplay_completion_tokens += int((reply.get("usage") or {}).get("completion_tokens") or 0)
        self.game.max_prompt_tokens = max(self.game.max_prompt_tokens, self.last_prompt_tokens)
        usage = reply.get("usage") or {}
        self.game.clock.charge(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), purpose)

    def result(self):
        row = {**super().result(), "arena_compactions": self.game.compactions if self.game else 0}
        row["harness_fixes"] = {**self.fix_stats, "fixes": list(self.harness_fixes["fixes"]),
                                "sandbox_interrupts": self.sandbox.interrupts, "stuck": self.stuck.summary()}
        row["context_drain"] = dict(self.drain_stats)
        if self.server_error_retries:
            row["server_error_retries"] = self.server_error_retries
        row["loop_rule"] = {**self.lrule_stats, "action": self.lrule["action"]}
        row["dfr_harness"] = {**self.dfr_stats, "flags": dict(self.dfr)}
        store = getattr(self, "animation_store", None)
        if store is not None:
            row["dfr_harness"]["animation_store"] = {"steps": len(store.items), "bytes": store.bytes,
                                                     "stored": store.stored, "dropped": store.dropped}
        row["dfr_loop"] = {"flags": dict(self.dloop)}
        row["hx_harness"] = {**self.hx_stats, "reflect": self.hx_reflector.summary()}
        return row

# Backoff (seconds) of the gameplay client's retries after an HTTP 5xx (with_server_retries).
SERVER_ERROR_DELAYS = (2.0, 5.0, 15.0)
STATE_STEP = re.compile(r"Current state: step \d+,")


def count_key(payload):
    """The identity of a counted request: messages, tools and template settings."""
    # Keep immutable strings by reference; serializing the entire growing context twice per turn is costly.
    # The one-entry count_cache is consumed at admission. Container snapshots also detect in-place mutations.
    parts = []
    _count_value([payload.get("messages"), payload.get("tools"), payload.get("chat_template_kwargs")], parts)
    return tuple(parts)


def _count_value(value, parts):
    kind = type(value)
    if kind in (str, int, bool, type(None)):
        parts.extend((kind, value))  # Distinguish JSON true, 1 and "1".
    elif kind in (list, tuple):
        parts.extend((list, len(value)))
        for item in value:
            _count_value(item, parts)
    elif kind is dict and all(type(key) is str for key in value):
        parts.extend((dict, len(value)))
        for key, item in sorted(value.items()):
            parts.append(key)
            _count_value(item, parts)
    else:
        # Preserve JSON's -0.0 and default=str behavior for uncommon payload values.
        parts.extend(("json", json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)))


class ArenaTransferDuckAgent(ArenaDuckMixin, TransferDuckAgent):
    """The corrected Duck agent (v5 prompt layout, persistent sandbox) with append-only context."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile = "duck-transfer-arena"

    def analyzer(self, loop):
        self.loop = loop
        analyzer = super().analyzer(loop)
        bind_compaction(analyzer, self)
        from . import hx_harness as hx_module

        analyzer._system_prompt = self.fixed_system_prompt(analyzer._system_prompt)
        analyzer._system_prompt = dfr_module.system_prompt(analyzer._system_prompt)
        analyzer._system_prompt = hx_module.system_prompt(analyzer._system_prompt, self.hx)
        self.loop_analyzer(analyzer)
        analyzer._max_output_tokens = self.cap
        analyzer._reply_reserve_tokens = self.cap
        analyzer._context_budget_tokens = self.context_budget_tokens(analyzer)
        self.configure_drain_analyzer(analyzer)
        self._arena_analyzer = analyzer
        if self.game.ready is not None:
            self.game.ready.set()
        return analyzer
