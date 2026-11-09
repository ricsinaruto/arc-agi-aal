"""The two Duck variants in the arena: corrected (duck_transfer) and public (ToolAgent).

Both run through ``arc_infra.session.GameSession`` exactly as duck_transfer does; the arena
adds the game's active clock, the KV gate (``client.slots`` is a ``GateSlot``), level reports
to the scheduler, append-only compaction, and — for the public variant — a ToolAgent with the
public prompts, tools and ephemeral python sandbox and only ``_chat_completion`` bridged.
"""

import asyncio
import concurrent.futures
import copy
import dataclasses
import hashlib
import json
import re
import time

import httpx

from arc_infra.model_templates import request_messages, template_kwargs, text_only
from duck_transfer.agent import (
    REFUSALS_PER_TURN,
    ContextLengthRejected,
    DuckAgent,
    GameplayRequestExpired,
)
from duck_transfer.transfer import TransferDuckAgent
from duck_transfer.trimming import ObservationPreservingTrim

from . import dfr_harness as dfr_module
from . import dfr_loop as loop_module
from . import harness_fixes as fixes_module
from . import negatives as negatives_module
from .compaction import (
    COMPACTION_THINKING,
    DEFAULT_COMPACTION_THINKING,
    DUCK_SUMMARY_PROMPT,
    PREFILL_TEMPLATE_MODELS,
    SUMMARY_PREFILL,
    SUMMARY_RETRY_SUFFIX,
    CompactionMixin,
    bind_compaction,
    mechanical_summary,
    prefilled_summary_text,
    split_at_latest_observation,
    summary_payload,
    summary_text,
)
from .harness import DuckHarness
from .memory import MEMORY_COMPACTION_NOTE, MEMORY_PROMPT, memory_message
from .notes import NOTES_PROMPT, parse_notes
from .snapshot import game_snapshot
from .snapshot import settings as snapshot_settings

ESTIMATE_HEADROOM = 1.1
# Prompt tokens the trimmer leaves free for the game-start snapshot (a 20-object view plus
# five probe lines is 600-900 tokens).
SNAPSHOT_RESERVE_TOKENS = 1500
# Prompt tokens left free for the explorer's report (fixed-2x-explorer: a 24x24 region twice,
# the compressed winning sequence and eight effect lines stay under 1,200 tokens).
EXPLORER_RESERVE_TOKENS = 1500
# fixed-2x-critical (paper notes 2.6 item 1, Self-Harness's retained Qwen3.5-35B-A3B rule): an
# exception in the python tool carries an explicit instruction to repair it first.
CRITICAL_NOTE = ("CRITICAL: the last step failed (see `error`). Fix it within 1-2 steps before doing anything "
                 "else; do not rerun the same code unchanged.")


SUMMARY_SCOPE_NOTE = (
    "\n\nScope: the newest {messages} messages before this request stay in the context verbatim after your "
    "summary. Summarize everything before them; do not repeat what those newest messages already say."
)


class ArenaDuckMixin:
    """Shared arena behaviour; ``attach(game, window, cap)`` must run before ``run()``."""

    game = None
    window = None
    cap = None
    loop = None
    # 25 September harness variants, both ``None`` on the fixed-2x control: the trial's
    # ``CrossGameNotes`` notebook (fixed-2x-notes) and the memory-block settings
    # (fixed-2x-memory). ``request_purpose`` names an aside request's clock purpose.
    notes = None
    memory = None
    request_purpose = None

    escalation = None
    # fixed-2x-snapshot: the deterministic game-start probe (settings, carried message, stats).
    snapshot = None
    snapshot_message = None
    snapshot_stats = None
    # fixed-2x-level-probe (snapshot ``every_level``): the levels (``obs.levels``) already probed,
    # the per-level stats and the block waiting for the next user turn.
    level_probed = frozenset()
    level_snapshots = None
    level_snapshot_notice = None
    # Evolved harness directory (offline harness evolution): prompt addition, playbook, skills.
    harness = None
    harness_stats = None
    # fixed-2x-explorer: the settings, the report carried from the next admission on, the lock the
    # explorer holds on the environment and the flag saying the agent waits for a model slot.
    explorer = None
    explorer_message = None
    explorer_waiting = False
    critical_errors = False
    critical_notes = 0
    # Harness v2 (failure-signature study edits E1-E5): per-game ``HarnessV2`` state, None on the control.
    harness_v2 = None
    # fixed-2x-adaptive-think: per-game ``AdaptiveThinking``, None on the control.
    adaptive = None
    light_pending = False       # the request being sent is a "prefill" light one
    # fixed-2x-worldmodel: the consolidation notice after each level clear (levels cleared when pending).
    worldmodel_notice = False
    worldmodel_pending = None
    worldmodel_notices = 0
    # hero-v2 harness_fixes (arena.harness_fixes.settings), None on every earlier build.
    harness_fixes = None
    last_gameplay_finish = None
    # hero-v2 parallel_samples (arena.parallel_samples.ParallelSampler) and action_plans (arena.action_plans); None /
    # False on every earlier build.
    sampler = None
    action_plans = False
    gameplay_replies = 0
    gameplay_completion_tokens = 0
    # hero-v2 drain_half (arena.context_drain): the settings, per-game totals and the pending follow-up rewrite of a
    # reply whose tool calls were dropped as malformed; None on every earlier build.
    drain = None
    drain_stats = None
    malformed_pending = None
    rejected_prompt_tokens = None
    # hero-v2 dfr_harness (arena.dfr_harness.settings), None on every earlier build.
    dfr = None
    # hero-v2 dfr_loop (arena.dfr_loop.settings; v5), None on every earlier build.
    dloop = None
    # harness arm hx_harness (arena.hx_harness.settings), None on every earlier build.
    hx = None
    hx_reflector = None
    hx_perceiver = None
    hx_coverage = None
    hx_explore_note = None
    # v7.1 loop rule (arena.loop_rule), None on every earlier build.
    lrule = None
    lrule_detector = None
    lrule_pending = None
    lrule_stats = None
    # hx4 fresh attempt (arena.fresh_attempt), None on every earlier build.
    fresh = None
    fresh_pending = None
    fresh_stats = None

    def attach(self, game, *, window, cap, notes=None, memory=None, escalation=None, snapshot=None, harness=None,
               procedure=None, explorer=None, critical_errors=False, harness_v2=None, adaptive_thinking=None,
               worldmodel_notice=False, harness_fixes=None, parallel_samples=None, action_plans=False,
               context_drain=None, dfr_harness=None, dfr_loop=None, hx_harness=None, loop_rule=None,
               fresh_attempt=None):
        self.game, self.window, self.cap = game, int(window), int(cap)
        # Per-game context window (fixed-2x-deep-explore): the scheduler moves a game between a 128K deep slot and a
        # 32K exploration slot; the compaction threshold follows ``game.window`` (``self.window`` stays the largest
        # request the agent may send, e.g. the one summary of a demoted game's 128K context).
        self._arena_analyzer = None
        if getattr(game, "window", None) is not None:
            if int(game.window) > self.window:
                raise ValueError(f"Game window {game.window} exceeds the agent window {self.window}")
            game.window_listener = self.apply_context_window
        self.worldmodel_notice, self.worldmodel_pending, self.worldmodel_notices = bool(worldmodel_notice), None, 0
        self.adaptive = None
        if adaptive_thinking:
            from .adaptive import AdaptiveThinking

            self.adaptive = AdaptiveThinking(adaptive_thinking, self.config.reasoning_effort)
        self.light_stats = {"replies": 0, "calls": 0, "tokens": 0, "turn_lines": 0}
        self.critical_errors, self.critical_notes = bool(critical_errors), 0
        self.explorer = explorer or None
        self.explorer_message = None
        self.explorer_waiting = False
        self.explorer_runs = []
        self.env_lock = asyncio.Lock()
        self.competition_clock = None  # no planning-budget notes in the arena
        self.notes, self.memory = notes, memory
        self.snapshot = snapshot_settings(snapshot) if snapshot else None
        self.snapshot_message = self.snapshot_stats = None
        self.level_probed, self.level_snapshots, self.level_snapshot_notice = set(), [], None
        self.harness = None
        self.harness_v2 = None
        if harness_v2:
            # The v2 directory composes its prompt, playbook and skills from the enabled edits.
            from .harness_v2 import ROOT, HarnessV2Dir

            harness = HarnessV2Dir((harness.root if isinstance(harness, DuckHarness) else harness) or ROOT, harness_v2)
        if harness is not None:
            self.harness = harness if isinstance(harness, DuckHarness) else DuckHarness(harness)
            self.harness_playbook = self.harness.playbook_message()
            self.harness_stats = {"preloads": 0, "prompt_chars": len(self.harness.prompt_text()),
                                  "playbook_chars": len((self.harness_playbook or {}).get("content", "")),
                                  "skills": [name for name, _, _ in self.harness.skills()],
                                  "problems": self.harness.validate()}
            self.sandbox_runner = self.harness_runner
        self.escalation = None
        if escalation:
            # fixed-2x-escalate: the stuck notice demands an executable, retrodicted model of
            # the level from the N-th notice on (the public tracker's counters and cadence stay).
            from .escalation import EscalatingStuckTracker

            self.stuck = EscalatingStuckTracker(self.config, after_notices=int(escalation.get("after_notices", 2)))
            self.escalation = self.stuck
        self.procedure = None
        if procedure:
            if escalation:
                raise ValueError("The stuck procedure and the escalation both replace the stuck notice")
            # fixed-2x-procedure: the stuck notice becomes a checklist plus a collapse detector.
            from .procedure import ProcedureStuckTracker

            self.stuck = ProcedureStuckTracker(self.config, collapse_window=int(procedure.get("collapse_window", 30)),
                                               collapse_share=float(procedure.get("collapse_share", 0.8)))
            self.procedure = self.stuck
        if harness_v2:
            from duck_transfer.feedback import StuckTracker

            from .harness_v2 import EffectStuckTracker, HarnessV2

            self.harness_v2 = HarnessV2(harness_v2, skills=self.harness_stats["skills"])
            if self.harness_v2.on["effects"]:
                if escalation or procedure:
                    raise ValueError("Harness v2's effect-aware stuck count and the escalation/procedure notices both replace the stuck tracker")
                self.stuck = EffectStuckTracker(self.config, self.harness_v2)
            elif type(self.stuck) is not StuckTracker and not (escalation or procedure):
                self.stuck = StuckTracker(self.config)
        self.notes_due = None  # "level_clear" or "park:<reason>" until the notes request goes out
        self.notes_seen_admits = 0
        self.notes_seen_levels = 0  # live refresh: the levels and compactions the snapshot saw
        self.notes_seen_compactions = 0
        self.notes_snapshot = None  # the injected message, frozen per admission
        self.notes_stats = {"requests": 0, "failures": 0, "seconds": 0.0, "lines_offered": 0,
                            "lines_added": 0, "injections": 0, "injected_lines": 0,
                            "injected_tokens": 0}
        self.memory_block = None
        self.memory_last_step = 0
        self.memory_stats = {"rewrites": 0, "failures": 0, "seconds": 0.0, "prompt_tokens": 0,
                             "completion_tokens": 0, "block_tokens": 0}
        if notes is not None:
            game.on_park = self.note_park
        self.apply_harness_fixes(harness_fixes, replaces_stuck=bool(escalation or procedure or (
            self.harness_v2 is not None and self.harness_v2.on["effects"])))
        self.apply_parallel_samples(parallel_samples)
        self.apply_action_plans(action_plans)
        self.apply_context_drain(context_drain)
        self.apply_dfr_harness(dfr_harness)
        self.apply_dfr_loop(dfr_loop)
        self.apply_hx_harness(hx_harness)
        self.apply_loop_rule(loop_rule)
        self.apply_fresh_attempt(fresh_attempt)
        return self

    # --- hx4 fresh attempt (arena.fresh_attempt) --------------------------------------------------------------------

    def apply_fresh_attempt(self, value):
        from .fresh_attempt import settings as fresh_settings

        spec = fresh_settings(value)
        if spec is not None and self.drain is None:
            raise ValueError("fresh_attempt needs drain_half")
        self.fresh = spec
        self.fresh_pending = None
        self.fresh_mark = None  # (level, gameplay tokens and actions at the level's (re)start, fresh attempts so far)
        self.fresh_stats = None if spec is None else {"attempts": 0, "resets": 0, "tokens_freed": 0}

    def fresh_beliefs(self):
        """``(carried world model, note() entries)`` of the attempt that is being abandoned."""
        knowledge = dict(getattr(getattr(self, "_analyzer", None), "_summarized_knowledge", None) or {})
        return knowledge, dict(((self.sandbox.hx_state or {}).get("notes")) or {})

    def fresh_due(self):
        """True when the current level is past its fresh-attempt cutoff (any thread; no side effects)."""
        from .fresh_attempt import cutoff

        mark = self.fresh_mark if self.fresh is not None else None
        if mark is None or self.obs is None or self.obs.state != "NOT_FINISHED" or mark[0] != self.obs.levels:
            return False
        limit = cutoff(self.fresh, mark[3])
        return limit is not None and self.gameplay_completion_tokens - mark[1] >= limit

    async def fresh_check(self):
        """Before a turn (event loop, worker idle): start a fresh attempt when the level's cutoff has passed (or, with
        ``at_start``, before the very first turn)."""
        from .fresh_attempt import cutoff

        if self.fresh is None or self.obs is None or self.obs.state != "NOT_FINISHED":
            return
        level, tokens = self.obs.levels, self.gameplay_completion_tokens
        forced = False
        if self.fresh_mark is None or self.fresh_mark[0] != level:
            forced = self.fresh_mark is None and self.fresh["at_start"]
            self.fresh_mark = (level, tokens, self.steps, 0)
            if not forced:
                return
        _, start_tokens, start_steps, count = self.fresh_mark
        limit = cutoff(self.fresh, count)
        if not forced and (limit is None or tokens - start_tokens < limit):
            return
        await self.fresh_attempt(tokens - start_tokens, self.steps - start_steps, count)

    async def fresh_attempt(self, tokens, actions, count):
        from arc_infra.plans import Action

        from .fresh_attempt import FIELDS, note_text

        knowledge, notes = self.fresh_beliefs()
        level = max(1, min(self.obs.total_levels, self.obs.levels + 1))
        reset = bool(self.fresh["reset"]) and self.steps != self.level_start_step
        if reset:
            async with self.env_lock:
                await self.step(Action("RESET"))
            self.write_duck_state("RESET")
            self.level_start_step = self.steps
            self.stuck.reset_level()
            self.fresh_stats["resets"] += 1
        analyzer = getattr(self, "_analyzer", None)
        if analyzer is not None:
            for key, _ in FIELDS:
                analyzer._summarized_knowledge[key] = ""
        self.fresh_pending = note_text(self.fresh, level=level, tokens=tokens, actions=actions,
                                       reset=reset or bool(self.fresh["reset"]), knowledge=knowledge, notes=notes)
        self.fresh_mark = (self.obs.levels, self.gameplay_completion_tokens, self.steps, count + 1)
        self.fresh_stats["attempts"] += 1
        self.workspace.append("fresh_attempt", step=self.steps, level=level, tokens=int(tokens), actions=int(actions),
                              number=count + 1, reset=reset, beliefs=self.fresh["beliefs"],
                              belief_fields=sorted(k for k, v in knowledge.items() if v), notes=len(notes),
                              note_chars=len(self.fresh_pending))

    def fresh_context(self, messages, tools, measure):
        """``[system, fresh-attempt note (+ the python inventory), current observation]`` (analyzer thread)."""
        from .context_drain import DRAIN_NOTE
        from .loop_rule import fresh_context

        text, self.fresh_pending = self.fresh_pending, None
        kept = fresh_context(list(messages[1:]), getattr(self, "is_observation", None))
        if not kept:
            kept = list(messages[-1:]) if len(messages) > 1 else []
        before = measure(messages, tools)
        dropped = len(messages) - 1 - len(kept)
        note = self.drain_note(dropped, 0)  # the python note and inventory after the drain sentence
        rest = note["content"][len(DRAIN_NOTE.format(messages=dropped, tokens=0)):]
        note = {**note, "content": (text + "\n\n" + rest.lstrip()).rstrip()}
        result = [messages[0], note, *kept]
        after = measure(result, tools)
        self.fresh_stats["tokens_freed"] += max(0, int(before) - int(after))
        self.workspace.append("fresh_attempt_context", step=self.steps, tokens_before=int(before),
                              tokens_after=int(after), messages_dropped=dropped)
        if dropped > 0:
            self.notify_context_drain(tokens_before=int(before), tokens_after=int(after))
        return result

    # --- v7.1 strict degenerate-loop rule (arena.loop_rule) ---------------------------------------------------------

    def apply_loop_rule(self, value):
        from .loop_rule import LoopDetector
        from .loop_rule import settings as loop_rule_settings

        spec = loop_rule_settings(value)
        if spec is not None and spec["action"] == "fresh_context" and self.drain is None:
            raise ValueError("loop_rule fresh_context needs drain_half")
        self.lrule = spec
        self.lrule_detector = None if spec is None else LoopDetector(spec)
        self.lrule_pending = None
        self.lrule_stats = None if spec is None else {"fires": 0, "applied": 0, "tokens_freed": 0}

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
        if self.lrule["action"] == "fresh_context":
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
        self.notify_context_drain(tokens_before=int(before), tokens_after=int(after))
        return result

    # --- harness arm hx_harness (arena.hx_harness) -----------------------------------------------------------------

    def apply_hx_harness(self, value):
        """Switch the harness arm's components on (``arena.hx_harness``); after dfr_harness, whose sandbox action
        wrapper the toolbox wraps."""
        from . import hx_harness as hx_module

        self.hx = hx_module.settings(value)
        if self.hx is None:
            return
        if not isinstance(self, TransferDuckAgent):
            raise ValueError("hx_harness extends the corrected Duck agent (duck-transfer)")
        if self.sandbox.process is not None:
            raise RuntimeError("hx_harness must be configured before the interpreter starts")
        if self.hx["grid"] and not self.dfr_on("vision"):
            raise ValueError("hx_harness grid labels the dfr_harness vision images (vision must be on)")
        if any(self.hx[k] for k in ("toolbox", "probe", "kit", "lattice", "autopilot", "worldmodel", "memo")):
            self.sandbox.bootstrap = hx_module.with_toolbox(self.sandbox.bootstrap, self.hx)
        self.hx_reflector = hx_module.Reflector(self.hx) if self.hx["reflect"] else None
        if self.hx["no_stuck_notice"]:
            self.stuck.notice = lambda *args, **kwargs: None  # no stuck notice (harness-evo hx-nostuck1h-v1; B200 run 1)
        self.hx_econ = None
        self.hx_spend = None
        self.hx_opened = None
        self.hx_memo_level = None
        self.hx_memo_shown = None  # memo_once: (notes line, marker) last shown
        self.hx_explored, self.hx_explore_note = None, None
        self.hx_stuck_tracker = None
        if self.hx["stuckexplore"]:
            self.hx_stuck_tracker = hx_module.Reflector({**self.hx, "reflect_style": 2})
        self.hx_lattice_level, self.hx_lattice_shown, self.hx_lattice_scene = None, False, None
        self.hx_coverage = hx_module.Coverage() if self.hx["coverage"] else None
        self.hx_merge_seen = 0
        if self.hx["merges"]:
            from arc_infra.transition import TransitionCache

            self.hx_merge_cache = TransitionCache(component_limit=self.config.transition_component_limit)
        self.hx_perceiver = None
        if self.hx["perceive"]:
            import numpy as np

            from arc_infra.transition import TransitionCache

            cache = TransitionCache(component_limit=self.config.transition_component_limit)  # its own: other thread
            self.hx_perceiver = hx_module.Perceiver(
                self.hx, lambda grid: cache.objects(np.asarray(grid, dtype=np.uint8)))
        self.hx_stats = {"toolbox_turns": 0, "max_tools": 0, "max_observers": 0, "observer_errors": 0,
                         "reflect_notices": 0}

    def hx_turn_lines(self):
        """The hx_harness additions to a new user turn (the STEP BACK block, then the toolbox listing)."""
        from . import hx_harness as hx_module

        lines = []
        if self.hx_explore_note:
            lines.append(self.hx_explore_note)
            self.hx_explore_note = None
        if self.hx_perceiver is not None:
            digest = self.hx_perceiver.turn(self.duck_history, self.transition_results, self.obs.levels,
                                            self.valid_actions())
            if digest:
                self.hx_perceiver.digests += 1
                self.hx_perceiver.chars += sum(len(line) for line in digest)
                lines.extend(digest)
        if self.hx_reflector is not None:
            self.hx_reflector.update(self.duck_history, self.transition_results, self.obs.levels,
                                     self.gameplay_replies, self.gameplay_completion_tokens)
            level = max(1, min(self.obs.total_levels, self.obs.levels + 1))
            notice = self.hx_reflector.notice(level)
            if notice:
                self.hx_stats["reflect_notices"] += 1
                self.workspace.append("hx_reflect", step=self.steps, reason=self.hx_reflector.last_reason,
                                      requests=self.hx_reflector.requests)
                lines.append(notice)
        if self.hx["lattice"]:
            if self.hx_lattice_level != self.obs.levels:
                self.hx_lattice_level, self.hx_lattice_shown = self.obs.levels, False
            if not self.hx_lattice_shown and self.hx_lattice_scene != self.obs.levels and self.duck_history:
                # Level start: a block-drawn board is detected from the board itself (before any move).
                self.hx_lattice_scene = self.obs.levels
                found = hx_module.detect_lattice(self.duck_history[-1].frame.grid)
                if found is not None:
                    self.hx_lattice_shown = True
                    self.hx_stats["lattice_scene_hints"] = self.hx_stats.get("lattice_scene_hints", 0) + 1
                    lines.append(hx_module.lattice_scene_hint(found))
            if not self.hx_lattice_shown:
                start = max(0, self.steps - 40)
                receipts = [r for step, r in sorted(self.transition_results.items()) if step > start
                            and (r.get("level") or 0) == max(1, min(self.obs.total_levels, self.obs.levels + 1))]
                stride = hx_module.stride_of(receipts)
                if stride is not None:
                    self.hx_lattice_shown = True
                    self.hx_stats["lattice_hints"] = self.hx_stats.get("lattice_hints", 0) + 1
                    lines.append(hx_module.lattice_hint(*stride))
        if self.hx["merges"] and self.duck_history:
            # Same-colour objects that came into contact (one component) or apart since the previous turn.
            import numpy as np

            notes = []
            start = max(1, self.hx_merge_seen)
            for index in range(start, len(self.duck_history)):
                entry = self.duck_history[index]
                if not entry.action or entry.action == "RESET":
                    continue
                before = self.duck_history[index - 1].frame.grid
                try:
                    notes += hx_module.merge_notes(self.hx_merge_cache.objects(np.asarray(before, dtype=np.uint8))["nodes"],
                                                   self.hx_merge_cache.objects(np.asarray(entry.frame.grid,
                                                                                          dtype=np.uint8))["nodes"],
                                                   entry.action)
                except Exception:  # noqa: BLE001 - a perception note must never break the turn
                    pass
            self.hx_merge_seen = len(self.duck_history)
            if notes:
                self.hx_stats["merge_notes"] = self.hx_stats.get("merge_notes", 0) + len(notes)
                lines.append("Harness: " + "; ".join(notes[-2:]) + ".")
        if self.hx_coverage is not None:
            self.hx_coverage.update(self.duck_history, self.obs.levels)
            line = self.hx_coverage.line(self.duck_history[-1].frame.grid) if self.duck_history else None
            if line:
                self.hx_stats["coverage_lines"] = self.hx_stats.get("coverage_lines", 0) + 1
                lines.append(line)
        if self.hx["opener"] and self.hx_opened != self.obs.levels:
            self.hx_opened = self.obs.levels
            level = max(1, min(self.obs.total_levels, self.obs.levels + 1))
            lines.append(hx_module.opener_line(level, self.valid_actions(), self.hx["probe"], self.hx["autopilot"]))
            self.hx_stats["openers"] = self.hx_stats.get("openers", 0) + 1
        if self.hx["economy"]:
            if self.hx_econ is None or self.hx_econ[0] != self.obs.levels:
                self.hx_econ = (self.obs.levels, self.steps)  # the level's actions count from its first turn
            level = max(1, min(self.obs.total_levels, self.obs.levels + 1))
            spend = None
            if self.hx["economy_spend"]:
                now = (self.gameplay_replies, self.gameplay_completion_tokens, self.game.clock.active_seconds())
                before, self.hx_spend = self.hx_spend or now, now
                spend = (now[0] - before[0], now[1] - before[1], max(0.0, now[2] - before[2]))
            lines.append(hx_module.economy_line(self.steps - self.hx_econ[1], level, spend))
        if self.hx["no_think"]:
            lines.append(hx_module.NO_THINK_TURN)
        if (self.adaptive is not None and self.adaptive.cfg["light"] == "prefill"
                and self.adaptive.reason(self.adaptive_marker()) is None):
            from .adaptive import LIGHT_TURN

            lines.append(LIGHT_TURN)  # the reply to this turn is written without a reasoning phase
            self.light_stats["turn_lines"] += 1
        state = self.sandbox.hx_state or {}
        if self.hx["autopilot"]:
            self.hx_stats["loops"] = int(state.get("loops") or 0)
            self.hx_stats["loop_actions"] = int(state.get("loop_actions") or 0)
        if self.hx["worldmodel"]:
            line = hx_module.sim_line(state.get("sim"))
            if line:
                self.hx_stats["sim_turns"] = self.hx_stats.get("sim_turns", 0) + 1
                lines.append(line)
        if self.hx["memo"] and self.hx["memo_style"] == 2:
            # v8.2 playbook: shown in full at a level start (and in every cut note, ``drain_note``); one reminder
            # before a cut when it was not written since the last one.
            text = str(state.get("playbook") or "")
            writes = int(state.get("playbook_writes") or 0)
            self.hx_stats.update(playbook_chars=len(text), playbook_writes=writes,
                                 playbook_max_chars=max(self.hx_stats.get("playbook_max_chars", 0), len(text)))
            keep = self.hx.get("keep_rules", 0)
            cleared = self.hx_memo_level is not None and self.obs.levels > self.hx_memo_level
            if cleared:
                lines.append(hx_module.KEEP_LEVEL if keep else hx_module.PLAYBOOK_LEVEL)
                lines.append(hx_module.playbook_block(state))
                self.hx_stats["playbook_shown"] = self.hx_stats.get("playbook_shown", 0) + 1
                self.hx_keep_due = 0
            if keep:
                kept = state.get("kept") or []
                wins = {k.get("level") for k in kept if k.get("kind") == "win"}
                self.hx_stats.update(kept_wins=len(wins), kept_rules=sum(k.get("kind") == "rule" for k in kept),
                                     kept_broken=sum(k.get("broken") is not None for k in kept))
                if (not cleared and self.obs.levels >= 1 and self.obs.levels not in wins
                        and self.obs.levels < self.obs.total_levels and getattr(self, "hx_keep_due", 2) < 2):
                    # The level start asked for won(text) and the reply did not write it: two reminders at most.
                    self.hx_keep_due += 1
                    self.hx_stats["keep_reminders"] = self.hx_stats.get("keep_reminders", 0) + 1
                    lines.append(hx_module.KEEP_DUE.format(level=self.obs.levels))
                told = getattr(self, "hx_keep_broken", None) or set()
                for k in kept:
                    if k.get("broken") is not None and k.get("name") not in told:
                        told.add(k.get("name"))
                        self.workspace.append("kept_rule_broken", step=self.steps, name=k.get("name"),
                                              at_step=k.get("broken"), text=k.get("text"))
                        lines.append(hx_module.KEEP_BROKEN.format(name=k.get("name"), step=k.get("broken")))
                self.hx_keep_broken = told
            if self.hx.get("playbook_log"):
                # Mechanism record (both arms of the kept-lines test): what the model carries at a level start, at
                # its first action on the level and after 100K tokens on it.
                tokens = self.gameplay_completion_tokens
                mark = getattr(self, "hx_log_mark", None)
                if mark is None or mark["levels"] != self.obs.levels:
                    mark = self.hx_log_mark = {"levels": self.obs.levels, "tokens": tokens, "steps": self.steps,
                                               "done": set()}
                    reasons = ["level_start"]
                else:
                    reasons = [r for r, due in (("first_action", self.steps > mark["steps"]),
                                                ("tokens_100k", tokens - mark["tokens"] >= 100000),
                                                ("tokens_300k", tokens - mark["tokens"] >= 300000))
                               if due and r not in mark["done"]]
                for reason in reasons:
                    mark["done"].add(reason)
                    self.workspace.append("playbook_snapshot", step=self.steps, reason=reason, levels=self.obs.levels,
                                          tokens_on_level=tokens - mark["tokens"], game_tokens=tokens,
                                          playbook=text, playbook_step=state.get("playbook_step"), writes=writes,
                                          kept=[dict(k) for k in (state.get("kept") or [])])
            self.hx_memo_level = self.obs.levels
            budget = getattr(getattr(self, "_arena_analyzer", None), "_context_budget_tokens", None)
            fill = ((self.drain_stats or {}).get("drains", 0), (getattr(self, "lrule_stats", None) or {}).get("applied", 0))
            cut_writes = getattr(self, "hx_playbook_cut", (None, 0))
            if cut_writes[0] != fill:
                cut_writes = self.hx_playbook_cut = (fill, writes)  # writes when this fill began
            if (budget and self.last_prompt_tokens >= 0.8 * budget and writes == cut_writes[1]
                    and getattr(self, "hx_playbook_due", None) != fill):
                self.hx_playbook_due = fill
                self.hx_stats["playbook_due"] = self.hx_stats.get("playbook_due", 0) + 1
                lines.append(hx_module.PLAYBOOK_OPEN_DUE)
        elif self.hx["memo"] and self.hx["memo_style"]:
            # Playbook: not repeated per turn (the trim carries it); a level-end line and one line before a cut.
            if self.hx_memo_level is not None and self.obs.levels > self.hx_memo_level:
                lines.append(hx_module.PLAYBOOK_LEVEL)
            self.hx_memo_level = self.obs.levels
            budget = getattr(getattr(self, "_arena_analyzer", None), "_context_budget_tokens", None)
            fill = getattr(self, "context_compactions", 0)
            if budget and self.last_prompt_tokens >= 0.8 * budget and getattr(self, "hx_playbook_due", None) != fill:
                self.hx_playbook_due = fill
                self.hx_stats["playbook_due"] = self.hx_stats.get("playbook_due", 0) + 1
                lines.append(hx_module.PLAYBOOK_DUE)
        elif self.hx["memo"]:
            if self.hx_memo_level is not None and self.obs.levels > self.hx_memo_level:
                lines.append(hx_module.LEVEL_NOTE)
            self.hx_memo_level = self.obs.levels
            line = hx_module.notes_line(state.get("notes"))
            if self.hx["memo_once"]:
                marker = ((self.drain_stats or {}).get("drains", 0), getattr(self, "context_compactions", 0),
                          self.obs.levels)
                line, self.hx_memo_shown = hx_module.memo_turn(line, marker, self.hx_memo_shown)
            if line:
                self.hx_stats["notes_turns"] = self.hx_stats.get("notes_turns", 0) + 1
                self.hx_stats["max_notes"] = max(self.hx_stats.get("max_notes", 0), len(state.get("notes") or {}))
                lines.append(line)
        if self.hx["toolbox"]:
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
        self.drain_stats = None if self.drain is None else {
            "drains": 0, "forced": 0, "turn_cap_drains": 0, "tokens_dropped": 0, "messages_dropped": 0,
            "turn_exchanges": 0, "malformed_replies": 0, "notes": 0, "hook_failures": 0}

    def configure_drain_analyzer(self, analyzer):
        """drain_half's tool-output settings on a new analyzer: ``tool_output_tokens`` (the system prompt's "capped to
        about N tokens" line follows) and the head/tail middle truncation."""
        spec = self.drain
        if spec is None:
            return
        from .context_drain import middle_truncate

        old, new = analyzer._tool_output_tokens, int(spec["tool_output_tokens"])
        if new != old:
            claim = "Tool responses are capped to about {} tokens."
            if analyzer._system_prompt.count(claim.format(old)) != 1:
                raise ValueError("Pinned tool-output line of the Duck system prompt changed")
            analyzer._system_prompt = analyzer._system_prompt.replace(claim.format(old), claim.format(new))
            analyzer._tool_output_tokens = new
            analyzer._tool_output_chars = max(256, new * 4)
        if spec["middle_truncation"]:
            analyzer._trim_tool_text = lambda text: middle_truncate(text, analyzer._tool_output_chars)

    def drain_context(self, analyzer, messages, tools, measure, budget, *, force=False):
        """``CompactionMixin._trim_messages_for_context`` under drain_half: ``messages`` itself when it fits (and the
        turn cap does not fire), else ``[system, (note), *suffix]`` with the oldest blocks dropped; the drain is
        recorded, and the scheduler's ``GameRecord.note_context_drain`` hook (when present) hears of it."""
        from .context_drain import drain_history

        if len(messages) < 2:
            return messages
        if self.fresh_pending is not None:
            self.lrule_pending = None
            return self.fresh_context(messages, tools, measure)
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
        if self.dfr is not None and self.dfr["vision"]:
            # The prefix changes here anyway (dfr_history); the current observation keeps its extra images.
            from .context_drain import latest_observation

            current = latest_observation(kept, is_observation)
            if current is not None:
                kept = [*self.dfr_history(kept[:current]), *kept[current:]]
        tokens_dropped = max(0, estimate(history) - estimate(kept))
        note = self.drain_note(info["messages_dropped"], tokens_dropped) if self.drain["note"] else None
        result = [system, *([note] if note else []), *kept]
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
            result, rounds = [system, *([note] if note else []), *kept], rounds + 1
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
                              rounds=rounds, note_chars=len(note["content"]) if note else 0,
                              **{k: v for k, v in info.items() if k != "prompt_tokens"})
        self.notify_context_drain(tokens_before=int(count), tokens_after=int(after))
        return result

    def notify_context_drain(self, *, tokens_before, tokens_after):
        """The scheduler hook ``GameRecord.note_context_drain(tokens_before=, tokens_after=)``, run on the event loop
        (the drain runs in the analyzer thread) before the drained request goes out; a failing hook is logged."""
        hook = getattr(self.game, "note_context_drain", None)
        if hook is None:
            return
        try:
            self.on_loop(hook, tokens_before=tokens_before, tokens_after=tokens_after)
        except (TimeoutError, asyncio.CancelledError, concurrent.futures.CancelledError, GameplayRequestExpired):
            raise
        except Exception as exc:  # noqa: BLE001 - scheduler bookkeeping never ends the game
            self.drain_stats["hook_failures"] += 1
            self.workspace.append("context_drain_hook_failed", step=self.steps, error=f"{type(exc).__name__}: {exc}"[:300])

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
        survives (the corrected agent), and the harness_fixes inventory of live definitions when that sub-fix is on."""
        from .context_drain import DRAIN_NOTE, PYTHON_NOTE

        text = DRAIN_NOTE.format(messages=messages, tokens=tokens)
        if isinstance(self, TransferDuckAgent):
            text += PYTHON_NOTE
            if getattr(self, "hx", None) is not None and self.hx["memo"] and self.hx["memo_style"] == 2:
                from . import hx_harness as hx_module

                state = self.sandbox.hx_state or {}
                text += "\n\n" + hx_module.playbook_block(state)
                self.hx_stats["playbook_shown"] = self.hx_stats.get("playbook_shown", 0) + 1
                self.workspace.append("playbook_shown", step=self.steps, chars=len(str(state.get("playbook") or "")),
                                      writes=int(state.get("playbook_writes") or 0))
            if self.fix("inventory"):
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
        from .context_drain import (MALFORMED_CALL_FOLLOWUP, TRUNCATED_CALL_FOLLOWUP,
                                    TRUNCATED_CALL_FOLLOWUP_NO_THINK)

        truncated = TRUNCATED_CALL_FOLLOWUP_NO_THINK if self.hx and self.hx["no_think"] else TRUNCATED_CALL_FOLLOWUP
        prefix = truncated if pending == "length" else MALFORMED_CALL_FOLLOWUP
        text = prefix + content[len(fixes_module.PUBLIC_FOLLOWUP_PREFIX):]
        return [*messages[:-1], {"role": "user", "content": text}]

    # --- hero-v2 parallel_samples and action_plans (arena.parallel_samples, arena.action_plans) ---------------------

    def apply_parallel_samples(self, value):
        from .parallel_samples import ParallelSampler
        from .parallel_samples import settings as parallel_settings

        spec = parallel_settings(value)
        self.sampler = None if spec is None else ParallelSampler(self, spec)

    def apply_action_plans(self, value):
        from .action_plans import settings as plans_settings

        self.action_plans = plans_settings(value)
        if not self.action_plans:
            return
        if not isinstance(self, TransferDuckAgent):
            raise ValueError("action_plans extends the corrected Duck agent's persistent sandbox (duck-transfer)")
        if self.sandbox.process is not None:
            raise RuntimeError("action_plans must be configured before the interpreter starts")
        from duck_transfer.sandbox import with_plans

        self.sandbox.bootstrap = with_plans(self.sandbox.bootstrap)

    # --- hero-v2 dfr_harness (arena.dfr_harness) ----------------------------------------------------------------

    def apply_dfr_harness(self, value):
        """Switch the ported dfranzen harness features on (``arena.dfr_harness``); after the harness fixes and the
        action plans, which replace or extend the sandbox bootstrap first."""
        self.dfr = dfr_module.settings(value)
        if self.dfr is None:
            return
        if not isinstance(self, TransferDuckAgent):
            raise ValueError("dfr_harness extends the corrected Duck agent (duck-transfer)")
        if self.harness_v2 is not None:
            raise ValueError("dfr_harness and harness v2 both write gameplay_changed/animation into the receipts")
        if self.dfr["vision"] and text_only(self.config.model):
            raise ValueError("dfr_harness vision needs an image model")
        if self.sandbox.process is not None:
            raise RuntimeError("dfr_harness must be configured before the interpreter starts")
        if self.dfr["guards"] or self.dfr["frame_diff"] or self.dfr["animation"]:
            self.sandbox.bootstrap = dfr_module.with_dfr(self.sandbox.bootstrap, self.dfr)
        if self.dfr["animation"]:
            self.sandbox.animation_handler = self.dfr_animation
        if self.sc_on():
            from . import sim_check

            self.sandbox.bootstrap = sim_check.with_simcheck(self.sandbox.bootstrap)
            self.sc_stats = sim_check.new_stats()
            self.sc_mark = None     # (levels, gameplay tokens at the level's first turn)
            self.sc_mode = None     # the level index (levels completed) that is past the stuck signal
            self.sc_row = None      # ... and its row in sc_stats["levels"]
            self.sc_notice = None   # the one-off notice of the next user turn
            self.sc_marker = None   # (drains, compactions) when the contract was last shown
            self.sc_seen_report = None
            self.sc_build = None    # simcarry: the model-first hold {"level" (index), "tokens_at", "row"}
            self.sc_first = True    # simcarry: the full contract has not been shown yet
            self.sc_runner_inner, self.sandbox_runner = self.sandbox_runner, self.sc_runner
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
        self.dfr_transients = []        # transient: actions since the last user turn that animated back to their board
        self.dfr_guards_seen = 0        # sandbox guard refusals already reported
        self.animation_record = None    # the last executed action's animation (their animation_record)

    def dfr_on(self, flag):
        return self.dfr is not None and self.dfr[flag]

    # --- hx5 evidence table and model check (arena.sim_check) -----------------------------------------------------------

    def sc_on(self):
        return self.dfr is not None and bool(self.dfr.get("simcheck") or self.dfr.get("evidence")
                                             or self.dfr.get("simcarry"))

    def sc_carry(self):
        return self.dfr is not None and bool(self.dfr.get("simcarry"))

    def sc_allowance(self):
        from . import sim_check

        return sim_check.BUILD_TOKENS

    def sc_events(self):
        """simcarry: count the child's events of every call (most calls of a hold are followed by no user turn)."""
        report, carry = self.sandbox.sc_state or {}, self.sc_stats["carry"]
        if report.get("events") and report is not self.sc_seen_report:
            self.sc_seen_report = report
            kinds = [e.get("kind") for e in report["events"]]
            self.sc_stats["caps"] += kinds.count("cap")
            self.sc_stats["mismatches"] += kinds.count("mismatch")
            carry["held_actions"] += kinds.count("held")
            carry["plans_found"] += sum(1 for e in report["events"] if e.get("kind") == "plan" and e.get("found"))
        for key in ("registered", "solves", "frontiers", "checked_lists"):
            self.sc_stats[key] = max(self.sc_stats[key], int(report.get(key) or 0))
        for key in ("plans", "plan_runs"):
            carry[key] = max(carry[key], int(report.get(key) or 0))
        carry["win_registered"] = max(carry["win_registered"], int(bool(report.get("win"))))

    def sc_active(self):
        """The current level is past the stuck signal."""
        return self.sc_mode is not None and self.obs is not None and self.sc_mode == self.obs.levels

    def sc_runner(self, *args, **kwargs):
        """The sandbox runner with the level's stuck-signal flags in the call's state."""
        state = kwargs.get("initial_state")
        if isinstance(state, dict) and self.obs is not None:
            on = self.sc_active()
            build = left = None
            if self.sc_carry():
                # A model-first hold lasts several python calls of ONE analysis turn (no action ends it), so the
                # previous call's events are counted here and a spent allowance frees the actions from this call on;
                # the row is closed at the next turn (sc_hold_check).
                self.sc_events()
                if self.sc_build is not None:
                    left = self.sc_build["tokens_at"] + self.sc_allowance() - self.gameplay_completion_tokens
                    build = self.sc_build["level"] if left > 0 else None
            kwargs["initial_state"] = {**state, "simcheck": {
                "mode": on and bool(self.dfr.get("simcheck") or self.dfr.get("simcarry")),
                "table": on and bool(self.dfr.get("evidence")), "level": self.obs.levels + 1,
                "carry": self.sc_carry(), "build": build, "left": left}}
        return self.sc_runner_inner(*args, **kwargs)

    def sc_check(self):
        """Before a turn (event loop): mark a new level; the stuck signal is ``MODE_TOKENS`` generated gameplay tokens
        on a level without a clear (evidence line from then on; with ``simcheck`` the level enters model mode)."""
        from . import sim_check

        if self.obs is None or self.obs.state != "NOT_FINISHED":
            return
        level, tokens = self.obs.levels, self.gameplay_completion_tokens
        if self.sc_carry():
            self.sc_events()
        if self.sc_mark is None or self.sc_mark[0] != level:
            cleared = self.sc_mark is not None and level > self.sc_mark[0]
            if self.sc_row is not None and cleared:
                self.sc_row["cleared"] = True
            self.sc_mark, self.sc_mode, self.sc_row = (level, tokens), None, None
            if cleared and self.sc_carry():
                self.sc_carry_enter(level, tokens)
            return
        if self.sc_carry():
            self.sc_hold_check(tokens)
            return
        spent = tokens - self.sc_mark[1]
        if self.sc_mode is None and spent >= sim_check.MODE_TOKENS:
            self.sc_mode = level
            if self.dfr.get("simcheck"):
                self.sc_notice = sim_check.notice(level + 1, spent)
            self.sc_row = {"level": level + 1, "at_tokens": int(spent), "at_step": self.steps, "mode_tokens": 0,
                           "unfit_tokens": 0, "fit": False, "cleared": False}
            self.sc_stats["mode_levels"] += 1
            self.sc_stats["levels"].append(self.sc_row)
            self.workspace.append("simcheck_mode", step=self.steps, level=level + 1, tokens=int(spent))

    def sc_finished(self, finished):
        """The child's last report on level number ``finished`` (simcarry), or None."""
        report = (self.sandbox.sc_state or {}).get("finished")
        return report if isinstance(report, dict) and report.get("level") == finished else None

    def sc_carry_enter(self, level, tokens):
        """simcarry, first turn after a clear: the level just won has NUMBER ``level`` (levels completed; frames count
        levels from 1). Its model is carried when the registered one already reproduces it; otherwise actions are
        held (``sc_build``) until it does or the allowance is spent. The new level is in model mode from its first
        turn."""
        from . import sim_check

        finished, carry = level, self.sc_stats["carry"]
        fit = self.sc_finished(finished)
        scored = int((fit or {}).get("scored") or 0)
        has_model = bool(fit) and "exact" in fit
        carried = (has_model and bool(fit.get("carried"))) or (fit is not None and scored < sim_check.MIN_LEDGER)
        row = {"level": finished, "at_step": self.steps, "transitions": scored,
               "at_clear": [fit["exact"], scored] if has_model else None, "hold": not carried, "hold_tokens": 0,
               "hold_turns": 0, "end": None, "fit": [fit["exact"], scored] if has_model else None,
               "win": (fit or {}).get("win") if has_model else None}
        carry["clears"] += 1
        carry["carried_at_clear"] += carried
        carry["rows"].append(row)
        self.sc_mode = level
        self.sc_row = {"level": level + 1, "at_tokens": 0, "at_step": self.steps, "mode_tokens": 0, "unfit_tokens": 0,
                       "fit": False, "cleared": False}
        self.sc_stats["mode_levels"] += 1
        self.sc_stats["levels"].append(self.sc_row)
        if carried:
            self.sc_notice = sim_check.carry_passed(finished, fit) if has_model else None
        else:
            carry["holds"] += 1
            self.sc_build = {"level": finished, "tokens_at": tokens, "row": row}
            self.sc_notice = sim_check.carry_notice(finished, scored, self.sc_first, fit)
            self.sc_first = False
        self.sc_marker = None if self.sc_notice is None else self.sc_marker
        self.workspace.append("simcarry_clear", step=self.steps, level=finished, carried=carried,
                              fit=row["at_clear"], transitions=scored)

    def sc_hold_check(self, tokens):
        """simcarry, before a turn: end the model-first hold on a fitting model or a spent allowance."""
        from . import sim_check

        build = self.sc_build
        if build is None:
            return
        fit = self.sc_finished(build["level"])
        has_model = bool(fit) and "exact" in fit
        row, carry = build["row"], self.sc_stats["carry"]
        if has_model:
            row["fit"], row["win"] = [fit["exact"], int(fit.get("scored") or 0)], fit.get("win")
        opened = (self.sandbox.sc_state or {}).get("build_open") == build["level"]
        reason = ("fit" if opened or (has_model and fit.get("carried")) else
                  "allowance" if tokens - build["tokens_at"] >= self.sc_allowance() else None)
        if reason is None:
            return
        row["end"] = reason
        carry["hold_fit" if reason == "fit" else "hold_allowance"] += 1
        self.sc_build = None
        self.sc_hold_end = sim_check.hold_end_line(build["level"], fit if has_model else None, reason)
        self.workspace.append("simcarry_hold_end", step=self.steps, level=build["level"], reason=reason,
                              fit=row["fit"], tokens=row["hold_tokens"], turns=row["hold_turns"])

    def sc_carry_lines(self):
        """simcarry: the lines of the next user turn (notice, hold or model-check line, counters)."""
        from . import sim_check

        lines = []
        marker = ((self.drain_stats or {}).get("drains", 0), getattr(self, "context_compactions", 0))
        if (self.sc_notice is None and not self.sc_first and marker != self.sc_marker
                and (self.sc_active() or self.sc_build)):
            self.sc_notice = sim_check.CARRY_REMINDER  # the contract left the context
        if self.sc_notice is not None:
            lines.append(self.sc_notice)
            self.sc_stats["notice_chars"] += len(self.sc_notice)
            self.sc_notice, self.sc_marker = None, marker
        report = self.sandbox.sc_state or {}
        end, self.sc_hold_end = getattr(self, "sc_hold_end", None), None
        if end:
            lines.append(end)
        if self.sc_build is not None:
            build = self.sc_build
            left = self.sc_allowance() - (self.gameplay_completion_tokens - build["tokens_at"])
            lines.append(sim_check.hold_line(build["level"], self.sc_finished(build["level"]), left))
        elif self.sc_active():
            level = self.obs.levels + 1
            current = report.get("mode") and report.get("level") == level
            shown = report if current else {"mode": True, "ledger": {}, "fit": None}
            if current or self.sc_stats["registered"] == 0:
                line = sim_check.carry_line(shown, level)
                lines.append(line)
                self.sc_stats["line_chars"] += len(line)
            fit = shown.get("fit")
            if fit:
                self.sc_stats["fit_last"] = [fit["exact"], fit["scored"]]
                self.sc_row["fit_last"] = [fit["exact"], fit["scored"]]
                self.sc_row.setdefault("fit_first", [fit["exact"], fit["scored"]])
                if fit["ok"] and fit["scored"]:
                    self.sc_row["fit"] = True
        self.sc_events()
        return lines

    def sc_turn_lines(self):
        """The lines of the next user turn: the one-off notice, the evidence line, the model-check line."""
        from . import sim_check

        if self.sc_carry():
            return self.sc_carry_lines()
        lines = []
        marker = ((self.drain_stats or {}).get("drains", 0), getattr(self, "context_compactions", 0))
        if self.sc_notice is None and self.sc_active() and self.dfr.get("simcheck") and marker != self.sc_marker:
            self.sc_notice = sim_check.notice(self.obs.levels + 1)  # the contract left the context
        if self.sc_notice is not None:
            lines.append(self.sc_notice)
            self.sc_stats["notice_chars"] += len(self.sc_notice)
            self.sc_notice, self.sc_marker = None, marker
        report = self.sandbox.sc_state or {}
        if self.sc_active():
            level = self.obs.levels + 1
            if self.dfr.get("evidence"):
                line = sim_check.evidence_line(report, level)
                if line:
                    lines.append(line)
                    self.sc_stats["table_chars"] += len(line)
            if self.dfr.get("simcheck"):
                shown = report if report.get("mode") else {"mode": True, "ledger": {}, "fit": None}  # no call yet
                line = sim_check.check_line(shown, level)
                lines.append(line)
                self.sc_stats["line_chars"] += len(line)
                fit = shown.get("fit")
                if fit:
                    self.sc_stats["fit_last"] = [fit["exact"], fit["scored"]]
                    if fit["ok"] and fit["scored"]:
                        self.sc_row["fit"] = True
        if report.get("events") and report is not self.sc_seen_report:
            self.sc_seen_report = report
            self.sc_stats["caps"] += sum(1 for e in report["events"] if e.get("kind") == "cap")
            self.sc_stats["mismatches"] += sum(1 for e in report["events"] if e.get("kind") == "mismatch")
        for key in ("registered", "solves", "frontiers", "checked_lists"):
            self.sc_stats[key] = max(self.sc_stats[key], int(report.get(key) or 0))
        return lines

    def sc_note_reply(self, tokens):
        """Generated gameplay tokens of a reply made past the stuck signal; ``unfit_tokens``: of those, the tokens
        generated while no fitting model had been registered on the level (the harm case of the list cap)."""
        if getattr(self, "sc_build", None) is not None:
            row, carry = self.sc_build["row"], self.sc_stats["carry"]
            row["hold_tokens"] += int(tokens or 0)
            row["hold_turns"] += 1
            carry["hold_tokens"] += int(tokens or 0)
            carry["hold_turns"] += 1
        if self.sc_active():
            tokens = int(tokens or 0)
            self.sc_stats["mode_tokens"] += tokens
            self.sc_row["mode_tokens"] += tokens
            if not self.sc_row["fit"]:
                self.sc_stats["unfit_tokens"] += tokens
                self.sc_row["unfit_tokens"] += tokens

    def valid_actions(self):
        names = super().valid_actions()
        if self.dfr is not None and self.dfr.get("reset_offer") and "RESET" not in names:
            names = [*names, "RESET"]
        if not self.dfr_on("actions"):
            return names
        return [dfr_module.UNDO[1] if name == dfr_module.UNDO[0] else name for name in names]

    def engine_action(self, label):
        if self.dfr_on("actions") and label == dfr_module.UNDO[1]:
            return dfr_module.UNDO[0]
        return super().engine_action(label)

    def model_action(self, name):
        if self.dfr_on("actions") and name == dfr_module.UNDO[0]:
            return dfr_module.UNDO[1]
        return super().model_action(name)

    def batch_stop(self, requested, position, verdicts):
        if not self.dfr_on("guards"):
            return super().batch_stop(requested, position, verdicts)
        stop = dfr_module.batch_stop(requested, position, verdicts, sweep=bool(self.dfr.get("sweep")))
        if stop is not None:
            self.dfr_stats["noop_stops"] += 1
            self.dfr_noop_stops.append(stop)
            self.workspace.append("dfr_noop_stop", step=self.steps, action=stop["no_op_action"],
                                  index=stop["no_op_action_index"], skipped=stop["skipped_actions"])
        return stop

    def dfr_annotate(self, action, label, payload, before):
        """Interior ``gameplay_changed`` (guards, animation) and the action's animation record (animation)."""
        if self.dfr["guards"] or self.dfr["animation"]:
            payload["gameplay_changed"] = dfr_module.gameplay_changed(before, self.obs.board)
            if self.dfr.get("border_moves"):
                # the edge-change memory lasts one level attempt: a new level, a RESET or a death starts it again
                key = (self.obs.levels, self.level_start_step)
                if (action.name == "RESET" or self.obs.state != "NOT_FINISHED"
                        or getattr(self, "dfr_border_key", None) != key):
                    self.dfr_border_key, self.dfr_border_seen = key, None
                if action.name != "RESET":
                    revisit, self.dfr_border_seen = dfr_module.border_revisit(
                        before, self.obs.board, getattr(self, "dfr_border_seen", None))
                    if revisit and payload["gameplay_changed"] is False:
                        payload["gameplay_changed"] = True
                        payload["edge_move"] = True
                        self.dfr_stats["edge_moves"] = self.dfr_stats.get("edge_moves", 0) + 1
        if self.dfr["animation"] and action.name != "RESET":
            # Their solver: the auto-reset after a death never touches the record (it does not pass through here).
            (start,) = dfr_module.normalize_frames([before])
            chain = dfr_module.collapse_frames(start, dfr_module.normalize_frames(self.obs.frames))
            summary = dfr_module.summarize_animation(chain)
            self.animation_record = ({"action_display": label, "chain": chain, "summary": summary,
                                      "step": self.steps, "level": payload.get("level")} if summary else None)
            if summary and self.dfr.get("animation_history"):
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
                if (self.dfr.get("transient") and summary.get("transient_pixels")
                        and payload.get("gameplay_changed") is False):
                    payload["transient_effect"] = int(summary["transient_pixels"])
                    self.dfr_transients.append({"step": self.steps, "label": label, "animation": summary})
        self.dfr_last_action = {"step": self.steps, "label": label, "animation": payload.get("animation"),
                                "gameplay_changed": payload.get("gameplay_changed")}

    def dfr_animation(self, request):
        """The sandbox's lazy ``last_animation_frames`` / ``last_animation_timeline`` (analyzer thread)."""
        self.dfr_stats["animation_reads"] += 1
        if request.get("step") is not None and self.dfr.get("animation_history"):
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
            key = {"stale_state": "stale_refusals", "sweep_exempt": "sweep_exemptions"}.get(item.get("kind"),
                                                                                          "terminal_refusals")
            self.dfr_stats[key] = self.dfr_stats.get(key, 0) + 1
        return fresh

    def dfr_rewrite(self, text, current_frame):
        """The user-turn additions: the new-level prompt, the animation opener and the guard notes after the executed
        actions, and on the first turn after a GAME_OVER the diagnosis block (its images go with the observation)."""
        analyzer = self._analyzer
        lines = text.split("\n")
        if self.dfr["level_prompt"] and lines.count(dfr_module.PUBLIC_LEVEL_LINE) == 1:
            knowledge = getattr(analyzer, "_summarized_knowledge", None) or {}
            extra = dfr_module.CROSS_LEVEL_CARRIED if knowledge.get("cross_level_notes") else dfr_module.CROSS_LEVEL_EMPTY
            if self.loop_on("no_world_model"):
                extra = ""  # his new-level prompt (no carried notes to point at)
            lines[lines.index(dfr_module.PUBLIC_LEVEL_LINE)] = dfr_module.LEVEL_START_USER_PROMPT + extra
            self.dfr_stats["level_prompts"] += 1
        reset = self.auto_reset
        game_over = (reset is not None and reset.get("reset_step") != self.dfr_gameover_shown
                     and any(line.startswith("GAME_OVER at step ") for line in lines))
        last = self.dfr_last_action
        notes = []
        if (self.dfr["animation"] and not game_over and last is not None and last["step"] != self.dfr_opener_step
                and reset is None):
            line = dfr_module.describe_animation(last["animation"], action_display=last["label"],
                                                 gameplay_changed=last["gameplay_changed"])
            self.dfr_opener_step = last["step"]
            if line:
                notes.append(line)
                self.dfr_stats["animation_openers"] += 1
        if self.dfr.get("transient"):
            # every such action but the last one, which the opener line above already describes
            hidden = [t for t in self.dfr_transients if last is None or t["step"] != last["step"]]
            self.dfr_transients = []
            note = dfr_module.transient_note(hidden) if self.dfr.get("transient_note") else ""
            if note:
                notes.append(note)
                self.dfr_stats["transient_notes"] = self.dfr_stats.get("transient_notes", 0) + 1
        if self.dfr["guards"]:
            notes.extend(dfr_module.guard_notes(self.dfr_noop_stops, self.dfr_guard_stops()))
            self.dfr_noop_stops = []
        executed = next((i for i, line in enumerate(lines) if line.startswith("Executed actions")), None)
        if notes and executed is not None:
            lines[executed + 1:executed + 1] = notes
        self.dfr_gameover_images = []
        v2 = game_over and self.fix("death_turn_v2") and bool(self.death_log)
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
            if self.dfr["vision"] and not v2:
                block.append(dfr_module.GAME_OVER_DIAGNOSIS)
            if self.dfr["animation"] and last is not None:
                line = dfr_module.describe_animation(last["animation"], action_display=last["label"],
                                                     gameplay_changed=last["gameplay_changed"], game_over=True)
                self.dfr_opener_step = last["step"]
                if line:
                    block.append(line)
                    self.dfr_stats["animation_openers"] += 1
            if self.dfr["vision"]:
                fatal = (reset.get("actions") or [None])[-1]
                if fatal and not v2:
                    block.append(dfr_module.FATAL_ACTION_LINE.format(action=str(fatal)))
                trace = self.loop_gameover_trace()
                if trace:
                    block.append(loop_module.GAME_OVER_TRACE.format(trace="; ".join(trace)))
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
        label = fixes_module.DEATH_IMAGE_LABEL if self.fix("death_turn_v2") else dfr_module.DEATH_IMAGE_LABEL
        images = [(label, dfr_module.board_image_url(death))]
        fatal = dfr_module.diff_image_url(alive, death)
        if fatal is not None:
            images.append((dfr_module.FATAL_DIFF_LABEL, fatal))
        return images

    def observation_message(self, message, current_frame):
        """``vision``: the board image at the larger upscale, then the diff image or the game-over images."""
        if not self.dfr_on("vision") or current_frame is None:
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
        if self.hx is not None and self.hx["grid"]:
            # hx_harness grid: the coordinate margin on every image of this (new) observation; history is untouched.
            from . import hx_harness as hx_module

            parts = [dfr_module.image_part(hx_module.with_coordinates(p["image_url"]["url"]))
                     if isinstance(p, dict) and p.get("type") == "image_url" else p for p in parts]
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
        if self.dloop is None:
            return
        if not isinstance(self, TransferDuckAgent):
            raise ValueError("dfr_loop extends the corrected Duck agent (duck-transfer)")
        if self.dloop["prompt_fix"] and self.drain is None:
            raise ValueError("dfr_loop prompt_fix describes drain_half's context drain")
        if self.dloop["action_feedback"]:
            if self.sandbox.process is not None:
                raise RuntimeError("dfr_loop must be configured before the interpreter starts")
            self.sandbox.bootstrap = loop_module.with_feedback(self.sandbox.bootstrap)
        self.sandbox.inventory_answers_animation = self.dloop["inventory_animation"]
        self.loop_stats = {key: 0 for key in ("yields", "resumes", "stale_openers", "stale_no_tool_call", "echoes",
                                              "traces", "gameover_traces")}
        self.loop_resume = False      # the next invocation resumes a turn that executed nothing
        self.loop_tool_calls = 0      # python calls in the current invocation
        self.loop_turn_tokens = 0     # the analyzer's generated tokens when the invocation started
        self.loop_yield_due = False   # the token budget ended the current invocation

    def loop_on(self, flag):
        return self.dloop is not None and bool(self.dloop[flag])

    def begin_turn(self):
        if not self.loop_on("yield_tokens"):
            return super().begin_turn()
        analyzer = self._analyzer
        self.loop_tool_calls, self.loop_yield_due = 0, False
        self.loop_turn_tokens = analyzer._session_generated_tokens if analyzer is not None else 0
        resume, self.loop_resume = self.loop_resume, False
        self.loop_stats["resumes"] += resume
        return not resume

    def analyzer_should_stop(self):
        if super().analyzer_should_stop():
            return True
        if self.fresh_due():
            return True  # the turn ends here; ``before_turn`` starts the fresh attempt
        if not self.loop_on("yield_tokens"):
            return False
        if self._analyzer._session_generated_tokens - self.loop_turn_tokens >= self.dloop["yield_tokens"]:
            self.loop_yield_due = True
            return True
        return False

    def end_turn(self, result):
        """His end of ``analyze``: an invocation that executed nothing is resumed; with no tool call at all its carried
        summary is marked ``no_tool_call``."""
        if not self.loop_on("yield_tokens") or result is None:
            return
        if result.step_executed:
            return
        self.loop_resume = True
        if self.loop_yield_due:
            self.loop_stats["yields"] += 1
            self.malformed_pending = None  # his truncated-call follow-up never outlives the yielded invocation
            self.workspace.append("dfr_yield", step=self.steps, tool_calls=self.loop_tool_calls,
                                  tokens=self._analyzer._session_generated_tokens - self.loop_turn_tokens)
        analyzer = self._analyzer
        summary = analyzer._last_step_summary
        if self.loop_tool_calls == 0 and summary:
            analyzer._last_step_summary = {**summary, "stale": True, "stale_reason": "no_tool_call"}

    def stale_opener_lines(self, summary):
        if not self.loop_on("yield_tokens") or not summary or not summary.get("stale"):
            return None
        self.loop_stats["stale_openers"] += 1
        self.loop_stats["stale_no_tool_call"] += summary.get("stale_reason") == "no_tool_call"
        return loop_module.stale_lines(summary)

    def loop_gameover_trace(self):
        if not self.loop_on("action_feedback"):
            return None
        summary = getattr(self._analyzer, "_last_step_summary", None) or {}
        trace = summary.get("game_over_trace") or []
        self.loop_stats["gameover_traces"] += bool(trace)
        return trace

    def finish_batch(self, result, requested, verdicts, stop):
        """His batch receipt: the trace (batch, game over, refusal), per-action interior flags, skipped, fatal and
        blocked actions, the echo line and the clock fields."""
        if not self.loop_on("action_feedback"):
            return super().finish_batch(result, requested, verdicts, stop)
        result = dict(result)
        executed = [payload for _, payload in verdicts if payload is not None]
        stop_reason = result.get("stop_reason")
        blocked = stop_reason if stop_reason in loop_module.BLOCKED else None
        refused = any(payload is None for _, payload in verdicts)
        if len(requested) > 1 or result.get("game_over") or blocked or refused:
            result["action_trace"] = loop_module.action_trace(requested, verdicts, blocked)
            result["gameplay_changed_per_action"] = [bool(p.get("gameplay_changed", True)) for p in executed]
            self.loop_stats["traces"] += 1
        if len(verdicts) < len(requested):
            result["skipped_actions"] = list(requested[len(verdicts):])
        fatal = next((p.get("action_display") for p in executed if p.get("game_over")), None)
        if fatal:
            result["fatal_action"] = str(fatal)
        if blocked:
            result["blocked_action"] = requested[len(verdicts)]
        echo = loop_module.action_echo(requested, executed, stop_reason)
        if echo:
            result["action_echo"] = echo
            self.loop_stats["echoes"] += 1
        if result.get("executed"):
            result["run_elapsed_seconds"] = max(0.0, time.monotonic() - self.started)
            result["time_remaining_seconds"] = max(0.0, self.remaining())
        return result

    def compact_receipt(self, compact):
        """His receipts carry no ``reward`` (0.0 on every action that does not finish a level)."""
        if self.loop_on("action_feedback"):
            compact.pop("reward", None)
        return compact

    def write_duck_state(self, action="", result=None):
        super().write_duck_state(action, result)
        if action and result is None and self.loop_on("action_feedback"):
            self.transition_results[self.steps]["automatic"] = True  # the harness's RESET after a GAME_OVER

    def loop_analyzer(self, analyzer):
        """The v5 system prompt and analyzer bindings (after every other system-prompt edit)."""
        spec = self.dloop
        text = analyzer._system_prompt
        if spec["prompt_fix"]:
            old = next((line for line in (loop_module.COMPACTION_LINE, loop_module.PUBLIC_IMAGE_LINE)
                        if text.count(line) == 1), None)
            if old is None:
                raise ValueError("dfr_loop: pinned image/history system line changed")
            text = text.replace(old, loop_module.fraction_line(
                self.compaction_keep_fraction, playbook=self.trim is not None and self.trim["summary"] == "playbook")
                if self.compaction_keep_fraction is not None else loop_module.DRAIN_LINE, 1)
        if spec["no_world_model"]:
            text = loop_module.world_model_free(text)
            # His ARC3_MEMORY_SECTIONS=off: nothing parsed, carried or shown; no cross-level hypothesis carry.
            analyzer._update_summarized_knowledge_from_assistant = lambda content: None
            analyzer._update_summarized_knowledge_from_step_summary = lambda: None
            analyzer._summarized_knowledge_lines = lambda: []
        if spec["action_feedback"]:
            text = loop_module.feedback_system(text)
        if spec["action_feedback"] or spec["yield_tokens"]:
            summarize = analyzer._summarize_step_sequence

            def summarize_with_extras(action_results):
                summary = summarize(action_results)
                if summary is not None:
                    summary.update(loop_module.summary_extras(action_results))
                return summary

            analyzer._summarize_step_sequence = summarize_with_extras
        analyzer._system_prompt = text

    # --- hero-v2 harness_fixes (arena.harness_fixes) -----------------------------------------------------------

    def apply_harness_fixes(self, value, *, replaces_stuck=False):
        self.harness_fixes = fixes_module.settings(value)
        self.count_cache = {}
        self.fix_stats = {"exact_trigger_counts": 0, "exact_trigger_fallbacks": 0, "admission_count_reused": 0,
                          "summary_guards": 0, "tail_cuts": 0, "inventories": 0, "inventory_chars": 0,
                          "cap_followups": 0, "cap_tool_notes": 0}
        self.neg_record = None
        if self.harness_fixes is None:
            return
        if self.fix("negative_check"):
            # arena.negatives: the level's record of executed and refused actions, the claims of the latest reply,
            # what was answered already, and the sandbox guard refusals read so far.
            self.neg_record = negatives_module.Record()
            self.neg_claims, self.neg_answered, self.neg_guards_seen = [], set(), 0
            self.fix_stats.update(negative_claims=0, negative_checks=0, negative_lines=0, negative_quiet=0,
                                  negative_chars=0)
        if self.fix("summary_headroom"):
            self.fix_stats["summary_splits"] = 0
        if self.fix("stuck_notice"):
            if replaces_stuck:
                raise ValueError("harness_fixes stuck_notice and the escalation/procedure/effect trackers all replace "
                                 "the stuck tracker")
            clock = self.game.clock
            self.stuck = fixes_module.ActiveStuckTracker(
                self.config, active_seconds=clock.active_seconds, completion_tokens=clock.completion_tokens,
                rule=self.harness_fixes)
        if self.fix("sandbox") or self.fix("timeout_interrupt") or self.fix("inventory"):
            from duck_transfer.history import preserve_transition_results
            from duck_transfer.sandbox import stateful_bootstrap

            public = self.module.run_sandboxed_python.__globals__["_SANDBOX_BOOTSTRAP"]
            bootstrap = stateful_bootstrap(preserve_transition_results(public),
                                           fixes=("modules",) if self.fix("sandbox") else ())
            self.sandbox.configure_fixes(bootstrap, interrupt=self.fix("timeout_interrupt"),
                                         grace=self.harness_fixes["interrupt_grace_seconds"])

    def fix(self, name):
        return self.harness_fixes is not None and name in self.harness_fixes["fixes"]

    def fixed_system_prompt(self, text):
        """The system prompt with the ``prompt_text``, ``sandbox`` and ``timeout_interrupt`` corrections."""
        if self.fix("prompt_text"):
            text = fixes_module.replace_pinned(text, fixes_module.SYSTEM_PROMPT_FIXES, "system prompt")
        if self.fix("sandbox"):
            text = fixes_module.replace_pinned(text, fixes_module.SANDBOX_PROMPT_FIXES, "system prompt")
        if self.fix("timeout_interrupt"):
            text = fixes_module.replace_pinned(text, fixes_module.INTERRUPT_PROMPT_FIXES, "system prompt")
        if self.fix("solvable_line"):
            anchor = fixes_module.SOLVABLE_ANCHOR
            if text.count(anchor) != 1:
                raise ValueError("harness_fixes solvable_line: pinned system prompt line changed")
            text = text.replace(anchor, anchor + fixes_module.SOLVABLE_LINE, 1)
        return text

    def rewrite_user_prompt(self, text, current_frame=None):
        """``prompt_text``: drop the closing "end of world model." line when no model is carried, and number the
        state line's step as the image label does (actions executed so far); then the dfr_harness additions."""
        if self.fix("prompt_text"):
            lines = text.split("\n")
            if fixes_module.WORLD_MODEL_HEADER not in text and lines.count(fixes_module.EMPTY_WORLD_MODEL_LINE) == 1:
                lines.remove(fixes_module.EMPTY_WORLD_MODEL_LINE)
            text = "\n".join(lines)
            step = current_frame.step if current_frame is not None else 0
            text = STATE_STEP.sub(f"Current state: step {step},", text, count=1)
        if self.loop_on("no_world_model"):
            lines = text.split("\n")
            if lines.count(loop_module.EMPTY_WORLD_MODEL_LINE) == 1:
                lines.remove(loop_module.EMPTY_WORLD_MODEL_LINE)
            text = "\n".join(lines)
        if self.dfr is not None:
            text = self.dfr_rewrite(text, current_frame)
        return text

    def before_trim(self, messages):
        """The not-yet-sent follow-up's rewrites: ``cap_followup`` and drain_half's malformed-call follow-up."""
        messages = self.cap_followup(messages)
        if self.malformed_pending is not None:
            messages = self.malformed_followup(messages)
        if self.fix("tool_call_followup") and messages:
            last = messages[-1]
            content = last.get("content")
            if last.get("role") == "user" and isinstance(content, str) and content.startswith(
                    fixes_module.PUBLIC_FOLLOWUP_PREFIX):
                # The public "You have not acted yet ... call action(actions)" follow-up (prompt audit item 1).
                messages = [*messages[:-1], {"role": "user", "content": fixes_module.TOOL_CALL_FOLLOWUP}]
        if self.loop_on("no_world_model") and messages:
            last = messages[-1]
            content = last.get("content")
            if last.get("role") == "user" and isinstance(content, str) and loop_module.FOLLOWUP_LABELS in content:
                messages = [*messages[:-1], {**last, "content": content.replace(loop_module.FOLLOWUP_LABELS, "", 1)}]
        return messages

    def cap_followup(self, messages):
        """``cap_followup``: the public follow-up after a reply that hit the completion cap says "You have not acted
        yet. Investigate first."; it is replaced (not yet sent, so the cached prefix is unchanged)."""
        if not self.fix("cap_followup") or not messages or self.last_gameplay_finish != "length":
            return messages
        last = messages[-1]
        content = last.get("content")
        if last.get("role") != "user" or not isinstance(content, str) or not content.startswith(
                fixes_module.PUBLIC_FOLLOWUP_PREFIX):
            return messages
        self.last_gameplay_finish = None
        self.fix_stats["cap_followups"] += 1
        self.workspace.append("cap_hit_followup", step=self.steps, cap=self.cap)
        text = fixes_module.CAP_FOLLOWUP_NO_THINK if self.hx and self.hx["no_think"] else fixes_module.CAP_FOLLOWUP
        return [*messages[:-1], {"role": "user", "content": text.format(cap=self.cap)}]

    def trigger_tokens(self, messages, tools):
        """The compaction trigger's measure: the exact count of the would-be gameplay request (``exact_trigger``),
        the calibrated estimate otherwise or when the count fails."""
        if not self.fix("exact_trigger"):
            return self.estimate_prompt_tokens(messages, tools)
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
        override = None if self.adaptive is not None else self.request_template_override()
        payload = self.gameplay_payload(self.inject(messages), tools, override)
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
            text_images=self.fix("text_count"))
        self.count_cache = {key: int(count)}
        return int(count)

    def summary_source(self, messages, tools, count):
        """``summary_guard``: with less than ``summary_room_tokens`` of room beside the exact count, the summary covers
        the messages up to the latest observation (or before it), so it never starves."""
        if not self.fix("summary_guard"):
            return messages
        room = int(self.harness_fixes["summary_room_tokens"])
        if count + room <= self.window:
            return messages
        prefix, tail = split_at_latest_observation(list(messages[1:]), keep=1)
        candidates = ([messages[0], *prefix, tail[0]], [messages[0], *prefix]) if tail else ()
        source, measured = messages, count
        for candidate in candidates:
            if len(candidate) < 2:
                continue
            source, measured = candidate, self.trigger_tokens(candidate, tools)
            if measured + room <= self.window:
                break
        self.fix_stats["summary_guards"] += 1
        self.workspace.append("compaction_guard", step=self.steps, prompt_tokens=count, source_tokens=measured,
                              excluded_messages=len(messages) - len(source), room=room)
        return source

    def summary_split(self, messages, tools, count, budget):
        """``summary_headroom``: ``(source, tail)`` when the summary of all of ``messages`` would have less than
        ``summary_room_tokens`` plus its own prompt of room in the window, else None (the ordinary compaction).

        ``source`` is ``messages[:cut]`` for the latest cut at a user or assistant message (never between a tool call
        and its result) whose exact count leaves the room; ``tail`` is ``messages[min(cut, latest observation):]``, kept
        verbatim after the summary, so every message is either summarised or kept, and the latest observation stays
        verbatim as in every compaction. The cut before the latest reply always fits: that reply's own prompt passed
        the trigger (<= W - cap - 512). None too when the tail would not fit the budget after a summary of that room
        (the summary guard then decides, as before)."""
        if not self.fix("summary_headroom") or len(messages) < 3:
            return None
        room = int(self.harness_fixes["summary_room_tokens"]) + self.summary_prompt_tokens()
        if count + room <= self.window:
            return None
        prefix, _ = split_at_latest_observation(list(messages[1:]), keep=1)
        observation = 1 + len(prefix)  # index of the latest user message (the tail of an ordinary compaction)
        carried = self.estimate_prompt_tokens([messages[0]], tools) + room
        if self.fix("inventory"):
            carried += int(self.harness_fixes["inventory_max_tokens"])
        tries = 0
        for cut in range(len(messages) - 1, 1, -1):
            if str(messages[cut].get("role", "")).strip() not in ("user", "assistant"):
                continue
            tries += 1
            measured = self.trigger_tokens(messages[:cut], tools)
            if measured + room > self.window:
                continue
            start, tail_tokens = cut, max(0, count - measured)
            if observation < cut:  # the reply that overshot came after the latest observation: keep it from there
                from_observation = max(0, count - self.trigger_tokens(messages[:observation], tools))
                if carried + from_observation <= budget:
                    start, tail_tokens = observation, from_observation
            if carried + tail_tokens > budget:
                self.workspace.append("compaction_split_skipped", step=self.steps, prompt_tokens=count,
                                      source_tokens=measured, tail_tokens=tail_tokens, budget=budget)
                return None
            self.fix_stats["summary_splits"] += 1
            self.workspace.append("compaction_split", step=self.steps, prompt_tokens=count, source_tokens=measured,
                                  summary_room=self.window - measured, kept_messages=len(messages) - start,
                                  summarised_messages=cut - 1, counts=tries, room=room)
            return list(messages[:cut]), list(messages[start:])
        return None

    def summary_prompt_tokens(self):
        """The summary request's own prompt (the handoff instructions and the prefill), estimated: the calibrated
        character ratio overstates it (run A: 228 exact tokens beside the gameplay count, estimated ~400)."""
        prompt, _ = self.compaction_prompt()
        mode = self.resolved_compaction_thinking()
        extra = [{"role": "assistant", "content": SUMMARY_PREFILL}] if mode == "prefilled-empty" else []
        return int(self.estimate_prompt_tokens([{"role": "user", "content": prompt}, *extra], None))

    def note_tail_cut(self, dropped):
        self.fix_stats["tail_cuts"] += 1
        self.workspace.append("compaction_tail_cut", step=self.steps, dropped_messages=dropped)

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
                                             chars_per_token=self.chars_per_token, reset_note=reset_note,
                                             imports=self.fix("inventory_imports"))
        self.fix_stats["inventories"] += 1
        self.fix_stats["inventory_chars"] += len(text)
        self.workspace.append("compaction_inventory", step=self.steps, chars=len(text), live=data is not None,
                              functions=len((data or {}).get("functions") or []),
                              values=len((data or {}).get("values") or []), receipts=len(receipts))
        return text

    def context_window(self):
        """The window the compaction threshold uses: the game's slot window when the scheduler sets one."""
        window = getattr(self.game, "window", None) if self.game is not None else None
        return min(int(window), self.window) if window else self.window

    def context_budget_tokens(self, analyzer):
        budget = max(1024, self.context_window() - self.cap - analyzer._request_safety_margin_tokens
                     - self.carried_reserve())
        if self.trim is not None and self.trim["budget_tokens"]:
            budget = min(budget, self.trim["budget_tokens"])  # hx7: another build's trigger at this window
        return budget

    def apply_context_window(self, window=None):
        """``GameRecord.window_listener``: the next trim compacts against the new window (an int attribute set
        from the event loop; the analyzer thread reads it at its next request)."""
        analyzer = getattr(self, "_arena_analyzer", None)
        if analyzer is not None:
            analyzer._context_budget_tokens = self.context_budget_tokens(analyzer)
            self.workspace.append("context_window", step=self.steps, window=self.context_window(),
                                  context_budget_tokens=analyzer._context_budget_tokens)

    def carried_reserve(self):
        """Prompt tokens the trimmer leaves free for the carried messages (0 on the control)."""
        reserve = 0
        if self.notes is not None:
            reserve += self.notes.inject_cap_tokens + 128
        if self.memory is not None:
            reserve += int(self.memory["request_cap_tokens"]) + 128
        if self.snapshot is not None:
            reserve += SNAPSHOT_RESERVE_TOKENS
        if self.harness is not None:
            reserve += int(self.harness.caps["playbook_cap_tokens"]) + 128
        if self.explorer is not None:
            reserve += EXPLORER_RESERVE_TOKENS
        return reserve

    # --- clocks ---------------------------------------------------------------------------

    def remaining(self):
        # Requests are never cut by the level budget (an in-flight request finishes); the wall
        # clock is the only hard deadline. Parking blocks the next request in the gate.
        return self.game.clock.global_clock.remaining() if self.game else super().remaining()

    def runtime_message(self, *, record=True):
        clock = self.game.clock
        now = time.monotonic()
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
        if self.harness_v2 is not None:
            effect = self.harness_v2.observe_step(action, self.previous, self.obs)
            for kind, track in effect["events"]:
                self.workspace.append("harness_v2_clock", step=self.steps, event=kind, revoked=track.revoked,
                                      **track.describe())
        # Board identity for the pool stall rule: a state the level has already shown is not progress.
        self.game.note_action((self.obs.levels, hashlib.sha1(self.obs.board.tobytes()).hexdigest()))
        self.game.report_level(self.obs.levels, self.game.clock.global_clock.elapsed(),
                               total_levels=getattr(self.obs, "total_levels", None))
        if self.obs.state == "WIN":
            self.game.note_finished("won")
        if self.notes is not None and self.obs.levels > before:
            self.notes_due = "level_clear"

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

    def harness_runner(self, *args, **kwargs):
        """The sandbox runner with the skills package installed in every fresh interpreter."""
        if self.sandbox.process is None:  # first call, or the interpreter was restarted
            preamble = self.harness.skills_preamble()
            if "code" in kwargs:
                kwargs["code"] = preamble + "\n" + kwargs["code"]
            else:
                args = (preamble + "\n" + args[0], *args[1:])
            self.harness_stats["preloads"] += 1
            self.workspace.append("harness_preload", step=self.steps, skills=self.harness_stats["skills"])
        return self.sandbox.run(*args, **kwargs)

    async def before_first_turn(self):
        if self.harness_v2 is not None:
            await asyncio.to_thread(self.harness_v2.level_start, self.obs.board, self.sandbox.color_chars,
                                    level=max(1, self.obs.levels + 1))
        if self.explorer is not None and self.explorer.get("at_start"):
            # Probe variant (explorer demonstration A/B): explore level 1 before the first model request;
            # ``report: false`` keeps the clear but withholds the demonstration from the model.
            while self.game.clock.global_clock.start is None:
                await asyncio.sleep(0.05)
            async with self.env_lock:
                await self.run_explorer(max_actions=int(self.explorer.get("max_actions", 20000)),
                                        max_seconds=float(self.explorer.get("max_seconds", 90)),
                                        should_stop=lambda: self.game.clock.global_clock.remaining() < 5)
            if not self.explorer.get("report", True):
                self.explorer_message = None
        if self.snapshot is None:
            return
        # The trial releases the clocks once every game is ready; the probe is charged as tool time.
        while self.game.clock.global_clock.start is None:
            await asyncio.sleep(0.05)
        level = self.obs.levels
        with self.game.clock.tool("probe"):
            text, stats = await game_snapshot(self, self.snapshot)
        self.snapshot_message = {"role": "user", "content": text}
        self.snapshot_stats = {**stats, "chars": len(text),
                               "tokens": self.estimate_prompt_tokens([self.snapshot_message], None)}
        self.workspace.append("game_snapshot", step=self.steps, **self.snapshot_stats)
        if self.snapshot["every_level"]:
            self.level_probed.add(level)
            self.level_snapshots.append(self.snapshot_stats)

    # --- fixed-2x-level-probe: the snapshot probe at the start of every later level -----------

    def level_probe_due(self):
        """True when the current level has not been probed yet and the game can still act."""
        return (self.snapshot is not None and self.snapshot["every_level"] and self.obs is not None
                and self.obs.state == "NOT_FINISHED" and self.obs.levels not in self.level_probed
                and self.game.clock.global_clock.start is not None and not self.stopping.is_set()
                and self.remaining() > 0 and self.steps < self.config.max_actions)

    async def before_turn(self):
        """Run on the event loop between analyzer turns (the worker is idle): a level cleared in
        the last turn, by the model or by a previous probe, gets its probe before the next user
        turn is built. The level is marked first, so a probe never runs twice for one level."""
        while self.level_probe_due():
            level = self.obs.levels
            self.level_probed.add(level)
            async with self.env_lock:
                with self.game.clock.tool("probe"):
                    text, stats = await game_snapshot(self, self.snapshot, level_start=True)
            # The carried block (outside the append-only history, survives compaction) is the latest
            # level's; the first user turn on the level shows it once more where the model reads.
            self.snapshot_message = {"role": "user", "content": text}
            self.level_snapshot_notice = "\n".join(filter(None, (self.level_snapshot_notice, text)))
            record = {**stats, "chars": len(text),
                      "tokens": self.estimate_prompt_tokens([self.snapshot_message], None)}
            self.level_snapshots.append(record)
            self.workspace.append("level_snapshot", step=self.steps, **record)
        if (self.hx is not None and self.hx["autoexplore"] and self.hx_explored != self.obs.levels
                and self.obs.state == "NOT_FINISHED"):
            self.hx_explored = self.obs.levels
            await self.hx_autoexplore()
        elif self.hx is not None and self.hx["stuckexplore"] and self.obs.state == "NOT_FINISHED":
            await self.hx_stuck_probe()
        await self.fresh_check()
        if self.sc_on():
            self.sc_check()

    def hx_skip_probes(self):
        """The stuck trackers skip the harness's own probe actions (they are not the model's attempts)."""
        for tracker in (self.hx_reflector, self.hx_stuck_tracker):
            if tracker is not None:
                if tracker.level != self.obs.levels:  # a probe cleared the level: start the new level first
                    tracker.update(self.duck_history, self.transition_results, self.obs.levels,
                                   self.gameplay_replies, self.gameplay_completion_tokens)
                tracker.skip_to(self.duck_history)

    async def hx_play_probes(self, cands, limit):
        """Play ``cands`` (model action dicts) one at a time through step_duck; returns (effect lines, stop)."""
        from . import hx_harness as hx_module

        lines, stop = [], None
        for item in cands[:limit]:
            if self.obs.state != "NOT_FINISHED" or self.remaining() <= 0:
                break
            before = self.obs.board
            result = await self.step_duck({"actions": [item]})
            if not result.get("executed"):
                stop = f"stopped: {result.get('stop_reason') or result.get('error')}"
                break
            label = (result.get("executed_actions") or [item.get("action")])[-1]
            cells = (result.get("transition") or {}).get("changed_cells") or 0
            interior = hx_module._interior_changed(tuple(map(tuple, before)), tuple(map(tuple, self.obs.board)))
            lines.append(hx_module.describe_effect(label, result, edge_only=bool(cells) and not interior))
            if result.get("level_completed") or result.get("game_over") or result.get("done"):
                stop = "stopped: " + ("level cleared" if result.get("level_completed") else "game over"
                                      if result.get("game_over") else "game won")
                break
        return lines, stop

    async def hx_stuck_probe(self):
        """hx_harness stuckexplore: when the level stalls (style-2 triggers), play untried actions / clicks on this
        exact board before the next model turn and report their effects."""
        import numpy as np

        from . import hx_harness as hx_module

        tracker = self.hx_stuck_tracker
        tracker.update(self.duck_history, self.transition_results, self.obs.levels, self.gameplay_replies,
                       self.gameplay_completion_tokens)
        if tracker.due() is None:
            return
        level = max(1, min(self.obs.total_levels, self.obs.levels + 1))
        tracker.notice(level)  # marks the trigger (its text is not shown)
        nodes = self.transitions.objects(np.asarray(self.obs.board, dtype=np.uint8)).get("nodes") or []
        cands = hx_module.explore_candidates(self.valid_actions(), nodes, 12)
        done = hx_module.tried_here(self.duck_history, self.duck_history[-1].frame.grid) if self.duck_history else set()
        cands = [c for c in cands if hx_module.label_of(c) not in done]
        why = {"deaths": "repeated deaths", "no_change": "a run of actions that changed nothing",
               "revisits": "the same board keeps coming back",
               "stall": f"{tracker.requests - tracker.change_request} requests since anything changed"}.get(
                   tracker.last_reason, tracker.last_reason)
        lines, stop = await self.hx_play_probes(cands, self.hx["stuck_explore_actions"])
        self.hx_skip_probes()
        self.hx_stats["stuck_probes"] = self.hx_stats.get("stuck_probes", 0) + 1
        self.hx_stats["stuck_probe_actions"] = self.hx_stats.get("stuck_probe_actions", 0) + len(lines)
        self.hx_explore_note = "\n".join(filter(None, (self.hx_explore_note,
                                                        hx_module.stuck_note(level, why, lines, stop))))
        self.workspace.append("hx_stuck_probe", step=self.steps, actions=len(lines), reason=tracker.last_reason)

    async def hx_autoexplore(self):
        """hx_harness autoexplore: at the start of a level, before the model's first turn on it, play each key action
        once and click each distinct small interior object once; the effects become one note in the next user turn."""
        import numpy as np

        from . import hx_harness as hx_module

        level = max(1, min(self.obs.total_levels, self.obs.levels + 1))
        nodes = self.transitions.objects(np.asarray(self.obs.board, dtype=np.uint8)).get("nodes") or []
        cands = hx_module.explore_candidates(self.valid_actions(), nodes, self.hx["autoexplore_clicks"])
        lines, stop = await self.hx_play_probes(cands, self.hx["autoexplore_actions"])
        self.hx_stats["autoexplore_actions"] = self.hx_stats.get("autoexplore_actions", 0) + len(lines)
        self.hx_skip_probes()
        if lines:
            self.hx_stats["autoexplore_levels"] = self.hx_stats.get("autoexplore_levels", 0) + 1
            self.hx_explore_note = hx_module.autoexplore_note(level, lines, stop)
            self.workspace.append("hx_autoexplore", step=self.steps, actions=len(lines), stop=stop)

    async def step_duck(self, arguments):
        if self.level_probe_due():
            # A python call that keeps acting after its action cleared the level: the new level's probe
            # must see the level's start board, so nothing executes until the turn ends and it has run.
            self.workspace.append("action_skipped", step=self.steps, reason="level_probe_pending")
            return {"executed": False, "stop_reason": "level_probe_pending", "error": LEVEL_PROBE_PENDING}
        return await self._step_duck(arguments)

    def note_park(self, reason):
        # Scheduler callback: the notes request goes out where the next request would block.
        self.notes_due = self.notes_due or f"park:{reason}"

    # --- requests -------------------------------------------------------------------------

    async def request_completion(self, payload):
        purpose = self.request_purpose or ("summary" if not payload.get("tools") else "gameplay")
        # fixed-2x-commit: vLLM priority of the admitted request (focus/opening 0, lane 1); other studies
        # send no priority field.
        if getattr(self, "sampling_override", None):
            # level_bench.sampling: the study's sampler on every request (gameplay, summaries, asides)
            payload = {**payload, **self.sampling_override}
        priority = getattr(self.client.slots, "request_priority", lambda: None)()
        if priority is not None:
            payload = {**payload, "priority": priority}
        if purpose == "gameplay":
            # The compaction request copies the last gameplay request's template settings and tool list so it
            # shares the game's cached prefix (gameplay_prefix_settings).
            self.last_gameplay_prefix = {"chat_template_kwargs": copy.deepcopy(payload.get("chat_template_kwargs") or {}),
                                         "tools": payload.get("tools")}
            self.cache_ping_request = None
        try:
            with self.game.clock.inference(purpose):
                if purpose == "gameplay" and self.sampler is not None and self.sampler.eligible():
                    # parallel_samples: k samples of this request, the first with a tool call wins (one reply).
                    return await self.sampler.complete(payload, super().request_completion)
                return await self.with_server_retries(super().request_completion, payload)
        finally:
            # When the game's cached prefix was last touched (arena.keep_alive).
            self.cache_touched = time.monotonic()

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

    # Observations kept verbatim after a compaction (study ``compaction_keep_observations``).
    compaction_keep_observations = 1
    compaction_keep_fraction = None
    compaction_summary_cap = None
    compaction_full_prefix = False
    summary_scope = None
    trim = None  # hx7 study ``trim`` (arena.trim_bed.settings)

    def fraction_measure(self, messages, tools):
        # Fail closed on tokenizer failure; an estimate cannot enforce 90/10 retention.
        future = asyncio.run_coroutine_threadsafe(self.exact_gameplay_tokens(messages, tools), self.loop)
        return future.result(timeout=max(1, self.remaining()))
    # arena.keep_alive: (payload, admission count) of the gameplay request waiting for admission, and the monotonic time
    # of the game's last request end.
    cache_ping_request = None
    cache_touched = None
    # How the compaction summary thinks (study ``compaction_thinking``, arena.compaction.COMPACTION_THINKING).
    compaction_thinking = DEFAULT_COMPACTION_THINKING

    def resolved_compaction_thinking(self):
        """The study's mode, with "prefilled-empty" served as "gameplay" off the pinned Qwen3.8 template and vLLM."""
        mode = self.compaction_thinking
        if mode not in COMPACTION_THINKING:
            raise ValueError(f"compaction_thinking must be one of {COMPACTION_THINKING}, not {mode!r}")
        if mode == "prefilled-empty" and (self.config.model not in PREFILL_TEMPLATE_MODELS
                                         or getattr(self.config, "serving_backend", "vllm") != "vllm"):
            return "gameplay"
        return mode

    def run_python_tool(self, execute, state_path, arguments):
        # Worker-thread tool execution (both Duck variants); the ``busy`` budget clock charges
        # it to the game. Was wrongly placed on the public variant only until 24 Sept 20:00 UTC.
        with self.game.clock.tool("python"):
            result = super().run_python_tool(execute, state_path, arguments)
        if self.critical_errors and isinstance(result, dict) and result.get("error"):
            self.critical_notes += 1
            result = {**result, "critical": CRITICAL_NOTE}
        if self.harness_v2 is not None:
            result = self.harness_v2_tool_result(result, arguments)
        if self.fix("cap_followup") and self.last_gameplay_finish == "length":
            # A tool call cut off by the completion cap (hero-v2 k095: "python requires a non-empty `code` string").
            result = self.with_note(result, fixes_module.CAP_TOOL_NOTE.format(cap=self.cap))
            self.fix_stats["cap_tool_notes"] += 1
        if self.loop_on("yield_tokens"):
            self.loop_tool_calls += 1
            analyzer = self._analyzer
            summary = analyzer._last_step_summary
            if not getattr(result, "step_executed", False) and summary and not summary.get("stale"):
                # His dispatch: carried over an exchange that executed nothing, the next opener reminds instead.
                analyzer._last_step_summary = {**summary, "stale": True}
        return result

    @staticmethod
    def with_note(result, note):
        """``result`` (a dict or the bridged tool's ``_ToolDispatchResult``) with a ``note`` field."""
        if isinstance(result, dict):
            return {**result, "note": note}
        payload = json.loads(result.content)
        return dataclasses.replace(result, content=json.dumps({**payload, "note": note}, indent=2))

    def harness_v2_tool_result(self, result, arguments):
        """E2: a python result without an executed action carries the nudge when it is due. The
        bridged tool returns the public ``_ToolDispatchResult`` (JSON ``content``), not a dict."""
        state = self.harness_v2
        state.note_python((arguments or {}).get("code"))
        executed = result.get("action_results") if isinstance(result, dict) else getattr(result, "step_executed", False)
        if executed:
            return result
        note = state.nudge()
        if note is None:
            return result
        self.workspace.append("harness_v2_nudge", step=self.steps, requests=state.requests_since_action)
        if isinstance(result, dict):
            return {**result, "note": note}
        payload = json.loads(result.content)
        return dataclasses.replace(result, content=json.dumps({**payload, "note": note}, indent=2))

    # --- harness v2 hooks of the Duck step (E1, E3, E4 co-metrics) -------------------------

    def no_effect_repeat(self, action):
        raw = super().no_effect_repeat(action)
        state = self.harness_v2
        if state is None or not state.on["effects"]:
            return raw
        return state.repeat(self.obs.levels, self.obs.board, action.identity(), raw)

    def tried_no_effect_here(self):
        state = self.harness_v2
        if state is None or not state.on["effects"]:
            tried = super().tried_no_effect_here()
            if self.dfr_on("actions"):
                tried = [self.model_action(name) for name in tried]
            return tried
        return state.tried_here(self.obs.levels, self.obs.board)

    def no_change_repeat_error(self, skipped):
        state = self.harness_v2
        if state is None or not state.on["effects"]:
            return super().no_change_repeat_error(skipped)
        return (f"{', '.join(skipped)} already had no gameplay effect on this board (only timer/counter cells "
                "or nothing changed); choose a different action. Tried here with no gameplay effect: "
                + ", ".join(self.tried_no_effect_here()))

    async def action_effect(self, action, label, payload, before, before_levels, changed):
        if self.dfr is not None:
            self.dfr_annotate(action, label, payload, before)
        if getattr(self, "neg_record", None) is not None:
            moved = payload.get("gameplay_changed")
            self.neg_record.executed(before_levels, label, hashlib.sha1(np.asarray(before, dtype=np.uint8).tobytes()).hexdigest(),
                                     changed if moved is None else bool(moved))
        if self.loop_on("action_feedback"):
            # His per-action result: model-issued, and the interior no-op flag.
            payload["automatic"] = False
            if "gameplay_changed" in payload:
                payload["no_op"] = payload["gameplay_changed"] is False
        effective = await self._action_effect(action, label, payload, before, before_levels, changed)
        if self.worldmodel_notice and payload["level_completed"] and not payload["done"]:
            self.worldmodel_pending = self.obs.levels
        if self.adaptive is not None:
            self.adaptive.note_action(changed=bool(effective), level_completed=bool(payload["level_completed"]),
                                      game_over=bool(payload["game_over"]))
        return effective

    def adaptive_marker(self):
        """Context cuts so far: the reply after one thinks (summary compactions and drains)."""
        drains = (getattr(self, "drain_stats", None) or {}).get("drains", 0)
        return (self.game.compactions if self.game else 0) + int(drains or 0)

    def request_template_override(self):
        if self.adaptive is None:
            return None
        fields = self.adaptive.decide(self.adaptive_marker())
        self.light_pending = self.adaptive.last_light and self.adaptive.cfg["light"] == "prefill"
        return fields or None

    def gameplay_payload(self, messages, tools, override=None):
        payload = super().gameplay_payload(messages, tools, override)
        if self.light_pending:
            # Mixed thinking, "prefill": the thinking request as it is, answered as a continued assistant turn.
            from .adaptive import LIGHT_PREFILL

            payload["messages"] = [*payload["messages"], {"role": "assistant", "content": LIGHT_PREFILL}]
            payload["add_generation_prompt"] = False
            payload["continue_final_message"] = True
            # No tool_choice "none": hx7-mixed-v1's server returned 0 of 372 light calls under it.
        return payload

    async def _action_effect(self, action, label, payload, before, before_levels, changed):
        state = self.harness_v2
        if state is None:
            return await super().action_effect(action, label, payload, before, before_levels, changed)
        effective = state.note_action(action, label, payload, before, before_levels, self.obs)
        if payload["level_completed"] and not payload["done"]:
            info = await asyncio.to_thread(state.level_start, self.obs.board, self.sandbox.color_chars,
                                           level=max(1, self.obs.levels + 1), previous_end=before)
            if info is not None:
                self.workspace.append("harness_v2_new_level", step=self.steps, **info)
        return effective

    async def _step_duck(self, arguments):
        if self.harness_v2 is None:
            return await super().step_duck(arguments)
        self.harness_v2.begin_batch()
        return self.harness_v2.end_batch(await super().step_duck(arguments))

    def turn_notice(self):
        """One-off notices in the next user turn: harness v2's new-level notice (E4) and the world-model
        consolidation notice after a level clear (fixed-2x-worldmodel)."""
        notices = []
        if self.harness_v2 is not None:
            notice = self.harness_v2.take_notice()
            if notice:
                self.workspace.append("new_level_notice", step=self.steps, text=notice)
                notices.append(notice)
        if self.worldmodel_pending is not None:
            notice = WORLDMODEL_NOTICE.format(cleared=self.worldmodel_pending)
            self.worldmodel_pending = None
            self.worldmodel_notices += 1
            self.workspace.append("worldmodel_notice", step=self.steps, text=notice)
            notices.append(notice)
        if getattr(self, "level_snapshot_notice", None):
            # fixed-2x-level-probe: the new level's block, once, in the first user turn on the level.
            notices.append(self.level_snapshot_notice)
            self.workspace.append("level_snapshot_notice", step=self.steps, chars=len(self.level_snapshot_notice))
            self.level_snapshot_notice = None
        if getattr(self, "hx", None) is not None:
            notices.extend(self.hx_turn_lines())
        if self.sc_on():
            notices.extend(self.sc_turn_lines())
        if getattr(self, "neg_record", None) is not None:
            notices.extend(self.negative_lines())
        return "\n".join(notices) or None

    def negative_sync(self):
        """Stale-state refusals of the sandbox since the last look go into the record (they never ran)."""
        guards = getattr(self.sandbox, "dfr_guards", None) or []
        fresh, self.neg_guards_seen = guards[self.neg_guards_seen:], len(guards)
        for item in fresh:
            if item.get("kind") == "stale_state" and item.get("action"):
                self.neg_record.refused(self.obs.levels, item["action"])

    def negative_lines(self):
        """``negative_check``: the record for the negatives of the latest reply, once per distinct claim."""
        claims, self.neg_claims = self.neg_claims, []
        if not claims:
            return []
        self.negative_sync()
        text, details = negatives_module.respond(self.neg_record, self.obs.levels, claims, self.neg_answered,
                                                 offered=tuple(self.valid_actions()))
        self.fix_stats["negative_quiet"] += sum(bool(d.get("quiet")) for d in details)
        if not text:
            return []
        self.fix_stats["negative_checks"] += 1
        self.fix_stats["negative_lines"] += sum(not d.get("quiet") for d in details)
        self.fix_stats["negative_chars"] += len(text)
        self.workspace.append("negative_check", step=self.steps, level=self.obs.levels + 1, text=text, details=details)
        return [text]

    def estimate_prompt_tokens(self, messages, tools):
        analyzer = self._analyzer
        estimate = analyzer._estimate_request_input_tokens(messages, tools=tools)
        return int(estimate * ESTIMATE_HEADROOM + max(0, round(self.count_residual)))

    async def chat(self, messages, tools):
        if self.explorer is None:
            return await self._chat(messages, tools)
        # The agent never touches its environment during a model request, so the explorer may act on it
        # for the whole of chat() while the game is parked: before the request (the agent waits here),
        # while it waits at the KV gate and while an admitted request finishes. The lock keeps the agent
        # from acting (tool calls follow chat()) until the explorer has handed back. explorer-v1 set the
        # flag only before the request and reached most games 10-40 minutes late.
        self.explorer_waiting = True
        try:
            await self.game.clock.wait_active()
            async with self.env_lock:
                pass
            return await self._chat(messages, tools)
        finally:
            self.explorer_waiting = False
            async with self.env_lock:
                pass

    async def _chat(self, messages, tools):
        if self.harness_v2 is not None:
            self.harness_v2.note_request()
        messages = await self.carried_messages(messages)
        self.client.slots.reserve(self.estimate_prompt_tokens(messages, tools))
        result = await super().chat(messages, tools)
        if self.drain is not None and self.drain["drop_malformed"]:
            result = self.without_malformed_calls(result)
        return result

    # --- carried messages (fixed-2x-notes, fixed-2x-memory) --------------------------------

    async def carried_messages(self, messages):
        """The outbound list: the analyzer's own list on the control. With a variant on, the
        aside requests due before this one go out first (notes at a clear or park, a memory
        rewrite every K actions) and the carried messages follow the system prompt; the
        analyzer's append-only history never holds them."""
        if (self.notes is None and self.memory is None and self.snapshot_message is None and self.harness is None
                and self.explorer_message is None):
            return messages
        if self.notes is not None:
            if self.notes_due:
                await self.notes_request(messages)
            self.refresh_notes_snapshot()
        if self.memory is not None and self.steps - self.memory_last_step >= int(self.memory["every_actions"]):
            await self.memory_rewrite(messages)
        return self.inject(messages)

    def inject(self, messages):
        carried = []
        if self.harness is not None and self.harness_playbook is not None:
            carried.append(self.harness_playbook)
        if self.snapshot_message is not None:
            carried.append(self.snapshot_message)
        if self.memory_block:
            carried.append(memory_message(self.memory_block))
        if self.notes_snapshot is not None:
            carried.append(self.notes_snapshot)
        if self.explorer_message is not None:
            carried.append(self.explorer_message)
        if not carried:
            return messages
        return [messages[0], *carried, *messages[1:]]

    def refresh_notes_snapshot(self):
        """Freeze the notebook for the coming admission: at the first request after an admit
        (the admission count moved) or when the game is parked at request time, in which case
        this request is the one that goes out after the re-admission."""
        admits = sum(entry["event"] == "admit" for entry in self.game.admissions)
        parked = self.game.clock.parked
        trigger = None
        if parked or admits != self.notes_seen_admits:
            trigger = "admission"
        elif self.notes.refresh == "live":
            if self.game.levels != self.notes_seen_levels:
                trigger = "level_clear"
            elif self.context_compactions != self.notes_seen_compactions:
                trigger = "compaction"
        if trigger is None:
            return
        self.notes_seen_admits = admits + 1 if parked else admits
        self.notes_seen_levels = self.game.levels
        self.notes_seen_compactions = self.context_compactions
        message, lines, tokens = self.notes.render(self.chars_per_token)
        if trigger != "admission" and message == self.notes_snapshot:
            return  # nothing new to carry: keep the cached prefix
        self.notes_snapshot = message
        if message is not None:
            self.notes_stats["injections"] += 1
            self.notes_stats["injected_lines"] += lines
            self.notes_stats["injected_tokens"] += tokens
            self.workspace.append("notes_injected", step=self.steps, lines=lines, tokens=tokens,
                                  admissions=admits, parked=parked, trigger=trigger)

    async def notes_request(self, messages):
        reason, self.notes_due = self.notes_due, None
        notes = self.notes
        prompt = NOTES_PROMPT.format(lines=notes.lines_per_request, notebook=notes.text() or "(empty)")
        try:
            text, data, seconds, _ = await self.aside_request(
                self.inject(messages), prompt, cap=notes.request_cap_tokens, purpose="notes",
                allow_parked=True,
            )
        except (TimeoutError, asyncio.CancelledError):
            raise
        except Exception as exc:  # noqa: BLE001 - a failed aside never ends the game
            self.notes_stats["failures"] += 1
            self.workspace.append("notes_request_failed", step=self.steps, reason=reason,
                                  error=f"{type(exc).__name__}: {exc}"[:500])
            return
        offered = parse_notes(text, notes.lines_per_request)
        added = notes.add(offered, source=f"{self.game_id}@{self.steps}")
        self.notes_stats["requests"] += 1
        self.notes_stats["seconds"] += seconds
        self.notes_stats["lines_offered"] += len(offered)
        self.notes_stats["lines_added"] += added
        self.workspace.append("notes_request", step=self.steps, reason=reason, offered=len(offered),
                              added=added, seconds=round(seconds, 3), usage=data.get("usage", {}))

    async def memory_rewrite(self, messages):
        self.memory_last_step = self.steps
        try:
            text, data, seconds, _ = await self.aside_request(
                self.inject(messages), MEMORY_PROMPT, cap=int(self.memory["request_cap_tokens"]),
                floor=512, purpose="memory",
            )
        except (TimeoutError, asyncio.CancelledError):
            raise
        except Exception as exc:  # noqa: BLE001 - the previous block stays in use
            self.memory_stats["failures"] += 1
            self.workspace.append("memory_rewrite_failed", step=self.steps,
                                  error=f"{type(exc).__name__}: {exc}"[:500])
            return
        usage = data.get("usage", {})
        self.memory_stats["seconds"] += seconds
        self.memory_stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self.memory_stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        if not text:
            self.memory_stats["failures"] += 1
            self.workspace.append("memory_rewrite_failed", step=self.steps, error="empty block")
            return
        self.memory_block = text
        self.memory_stats["rewrites"] += 1
        self.memory_stats["block_tokens"] = self.estimate_prompt_tokens([memory_message(text)], None)
        self.workspace.append("memory_rewrite", step=self.steps, block_chars=len(text),
                              seconds=round(seconds, 3), usage=usage)

    async def prepare_admission(self, payload, *, fit_floor=None):
        """Count the outbound request exactly and reserve it; a request whose count plus ``max_tokens`` exceeds the
        window is rejected, unless ``fit_floor`` is given (aside requests): then ``max_tokens`` shrinks to the room
        left, if at least ``fit_floor``."""
        payload['messages'] = request_messages(self.config.model, payload['messages'])
        payload["chat_template_kwargs"] = template_kwargs(
            self.config.model, payload.get("chat_template_kwargs", {})
        )
        if self.client.slots.gate.budget is None:
            return None
        # Count actual multimodal messages, tools and reasoning template before waiting.
        # Arena timing is appended at message construction, not mutated during admission.
        cached = None
        if self.fix("exact_trigger") and self.count_cache and not payload.get("continue_final_message"):
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
                template_kwargs=payload.get("chat_template_kwargs"), text_images=self.fix("text_count"),
                **({"continue_final_message": True} if payload.get("continue_final_message") else {}),
            )
        if count + payload["max_tokens"] > self.window:
            room = self.window - count
            if fit_floor is None or room < fit_floor:
                if fit_floor is None:
                    self.rejected_prompt_tokens = int(count)  # drain_half's forced drain starts from it
                # The phrase is the public loop's context-length marker, so it trims and retries.
                raise ContextLengthRejected(
                    "Exact arena request exceeds context window; reduce the length of the input prompt"
                )
            # An aside's cap is ``window - estimate``, and its estimate leaves out the rendered tool list and
            # its own prompt: fit the cap to the exact count instead (hero-21x42k k035: a compaction summary
            # of 29K exact tokens with a 14K cap in the 42,496 window ended the game).
            self.workspace.append("aside_cap_fitted", step=self.steps, prompt_tokens=count,
                                  requested=payload["max_tokens"], max_tokens=room)
            payload["max_tokens"] = room
        self.client.slots.reserve(count)
        self.admitted_count = int(count)
        self.workspace.append("exact_admission_count", step=self.steps, prompt_tokens=count,
                              output_allowance=payload["max_tokens"])
        if fit_floor is None and payload.get("tools") and not payload.get("continue_final_message"):
            # The gameplay request waiting for admission (arena.keep_alive pings it while the game is parked).
            self.cache_ping_request = (payload, count)
        text_messages = [
            {**m, "content": [{"type": "text", "text": "[image]"} if p.get("type") == "image_url" else p
                              for p in m["content"]]} if isinstance(m.get("content"), list) else m
            for m in payload["messages"]
        ]
        chars = len(json.dumps(text_messages, ensure_ascii=False))
        # Images are already counted by the serving processor, so never add them twice.
        return count, 0, chars, 0

    def record_reply(self, reply, purpose="gameplay"):
        if purpose == "gameplay" and self.light_pending:
            from .adaptive import light_reply

            self.light_pending = False
            self.light_stats["replies"] += 1
            called = light_reply(reply["message"], self.light_stats["replies"])
            self.light_stats["calls"] += called
            self.light_stats["tokens"] += int((reply.get("usage") or {}).get("completion_tokens") or 0)
            if called and reply.get("finish_reason") == "stop":
                reply["finish_reason"] = "tool_calls"
            self.workspace.append("light_reply", step=self.steps, called=called,
                                  completion_tokens=(reply.get("usage") or {}).get("completion_tokens"))
        super().record_reply(reply, purpose)
        admitted, self.admitted_count = getattr(self, "admitted_count", None), None
        served = (reply.get("usage") or {}).get("prompt_tokens")
        if self.fix("text_count") and admitted is not None and served:
            # text_count's check: the admission count against the served prompt.
            error = int(served) - admitted
            self.fix_stats["text_count_checked"] = self.fix_stats.get("text_count_checked", 0) + 1
            if error:
                self.fix_stats["text_count_mismatches"] = self.fix_stats.get("text_count_mismatches", 0) + 1
                self.fix_stats["text_count_max_error"] = max(abs(error), self.fix_stats.get("text_count_max_error", 0))
                self.workspace.append("text_count_mismatch", step=self.steps, admitted=admitted, served=int(served))
        if purpose == "gameplay" and self.lrule_detector is not None:
            self.note_loop_rule(reply)
        if purpose == "gameplay" and getattr(self, "neg_record", None) is not None:
            # What happened since the previous reply is what this reply's negatives are about.
            self.negative_sync()
            message = reply.get("message") or {}
            self.neg_claims = negatives_module.claims(negatives_module.reply_text(message), self.valid_actions())
            self.fix_stats["negative_claims"] += len(self.neg_claims)
            self.neg_record.mark()
        if purpose == "gameplay":
            self.last_gameplay_finish = reply.get("finish_reason")
            self.gameplay_replies += 1
            if self.hx and self.hx["no_think"]:
                message = reply.get("message") or {}
                text = message.get("content")
                self.hx_stats["no_think_replies"] = self.hx_stats.get("no_think_replies", 0) + 1
                self.hx_stats["no_think_with_text"] = (self.hx_stats.get("no_think_with_text", 0)
                                                       + bool(isinstance(text, str) and text.strip()))
            self.gameplay_completion_tokens += int((reply.get("usage") or {}).get("completion_tokens") or 0)
            if self.sc_on():
                self.sc_note_reply((reply.get("usage") or {}).get("completion_tokens"))
        self.game.max_prompt_tokens = max(self.game.max_prompt_tokens, self.last_prompt_tokens)
        usage = reply.get("usage") or {}
        self.game.clock.charge(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), purpose)
        if self.harness_v2 is not None and purpose == "gameplay":
            self.harness_v2.note_reply(usage)

    def playbook_handoff(self):
        """``trim.summary: playbook`` (analyzer thread): the message a trim leaves in place of the dropped part: the
        model's playbook (else its notes) and the python inventory. No model request."""
        from . import hx_harness as hx_module

        state = self.sandbox.hx_state or {}
        text = hx_module.PLAYBOOK_HANDOFF + hx_module.playbook_text(state)
        if self.fix("inventory"):
            future = asyncio.run_coroutine_threadsafe(self.inventory_block(), self.loop)
            text = text.rstrip() + "\n\n" + future.result(timeout=max(1, self.remaining()))
        self.context_compactions += 1
        self.game.compactions += 1
        getattr(self.game, "note_compaction", lambda: None)()
        self.workspace.append("playbook_handoff", step=self.steps, chars=len(text),
                              playbook_chars=len(str(state.get("playbook") or "")),
                              playbook_step=state.get("playbook_step"), notes=len(state.get("notes") or {}))
        return text

    def summarize(self, messages):
        """Blocking bridge for the analyzer thread: one summary request on the event loop."""
        future = asyncio.run_coroutine_threadsafe(self.summary_request(messages), self.loop)
        self.chat_future = future
        try:
            return future.result()
        finally:
            self.chat_future = None

    async def aside_request(self, messages, prompt, *, cap, floor=256, purpose,
                            record_purpose=None, allow_parked=False, gameplay_prefix=False, prefill=None):
        """One tool-less, thinking-off request on ``messages`` plus ``prompt``, admitted through
        the gate and charged to the game's clock as ``purpose``: ``(text, data, seconds,
        estimate)``. ``allow_parked`` lets it out while the game is parked (notes at a park).
        ``prefill`` continues a pre-filled assistant turn (arena.compaction.summary_payload)."""
        module = self.module
        estimate = self.estimate_prompt_tokens(messages, None)
        # summary_guard: the admission fits a summary's cap to the exact count's room instead of ``window - estimate``
        # (the recorded hero-v2 summaries got 2.6K of allowance where the exact room was 7.7K).
        exact_room = self.fix("summary_guard") or self.fix("summary_headroom")
        if not (purpose == "summary" and exact_room and self.client.slots.gate.budget is not None):
            cap = max(floor, min(cap, self.window - estimate))
        payload = summary_payload(
            build_chat_payload=module.build_chat_payload,
            model=self.config.model,
            messages=messages,
            prompt=prompt,
            cap=cap,
            temperature=self.config.temperature,
            top_p=module._LOCAL_ANALYZER_TOP_P,
            top_k=module._LOCAL_ANALYZER_TOP_K,
            **(self.gameplay_prefix_settings() if gameplay_prefix else {}),
            **({"prefill": prefill} if prefill is not None else {}),
        )
        started = time.monotonic()
        slots = self.client.slots
        slots.reserve(estimate)
        await self.prepare_admission(payload, fit_floor=floor)
        cap = payload["max_tokens"]
        if allow_parked:
            slots.allow_parked = True
        try:
            async with slots:
                if self.stopping.is_set() or self.remaining() <= 0:
                    raise TimeoutError("Gameplay deadline reached")
                self.request_purpose = purpose
                try:
                    response = await self.request_completion(payload)
                finally:
                    self.request_purpose = None
                response.raise_for_status()
                data = response.json()
        finally:
            if allow_parked:
                slots.allow_parked = False
        choice = data["choices"][0]
        text = summary_text(choice["message"]) if prefill is None else prefilled_summary_text(choice["message"], prefill)
        seconds = time.monotonic() - started
        reply = {
            "message": choice["message"],
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage", {}),
            "seconds": seconds,
            "force_action": False,
            "reasoning_effort": (payload.get("chat_template_kwargs") or {}).get("reasoning_effort")
            if (payload.get("chat_template_kwargs") or {}).get("enable_thinking") else None,
            "max_output_tokens": cap,
        }
        self.record_reply(reply, purpose=record_purpose or purpose)
        return text, data, seconds, estimate

    def gameplay_prefix_settings(self):
        """The last gameplay request's template settings and tools, so a compaction request shares its cached
        prefix; before any gameplay request, the configured ones."""
        last = getattr(self, "last_gameplay_prefix", None)
        if last is None:
            return {"thinking": bool(self.config.enable_thinking), "reasoning_effort": self.config.reasoning_effort}
        kwargs = last["chat_template_kwargs"]
        return {"thinking": bool(kwargs.get("enable_thinking", True)), "reasoning_effort": kwargs.get("reasoning_effort"),
                "tools": copy.deepcopy(last["tools"]), "template_kwargs": kwargs}

    def compaction_prompt(self):
        """``(prompt, inject)``: the summary request's instructions and whether the carried memory block goes in."""
        from .trim_bed import summary_prompt

        prompt = summary_prompt(self.trim, DUCK_SUMMARY_PROMPT)
        if self.harness_v2 is not None and self.harness_v2.on["handoff"]:
            from .harness_v2 import handoff_prompt

            prompt = handoff_prompt(DUCK_SUMMARY_PROMPT)
        if self.memory_block:
            prompt += MEMORY_COMPACTION_NOTE
        if self.summary_scope:
            prompt += SUMMARY_SCOPE_NOTE.format(messages=self.summary_scope)
        return prompt, bool(self.memory_block)

    async def summary_request(self, messages):
        prompt, inject = self.compaction_prompt()
        if inject:
            # The block is outside the analyzer's list; the summary sees it and skips its content.
            messages = self.inject(messages)
        text, data, seconds, estimate = "", {}, 0.0, None
        attempts, rejected = 1, False
        mode = self.resolved_compaction_thinking()
        # "gameplay" / "prefilled-empty": the gameplay prefix (the latter continuing an empty think block); "off": the
        # tool-less, thinking-off request.
        gameplay_prefix = mode != "off"
        prefill = SUMMARY_PREFILL if mode == "prefilled-empty" else None
        try:
            try:
                text, data, seconds, estimate = await self.aside_request(
                    messages, prompt, cap=self.compaction_summary_cap or self.cap, floor=2048, purpose="summary", record_purpose="compaction",
                    gameplay_prefix=gameplay_prefix, prefill=prefill,
                )
            except httpx.HTTPStatusError as exc:
                if not gameplay_prefix:
                    raise
                # A server that refuses the tool list with tool_choice "none" (or the prefilled continuation) must
                # not end the game: fall back to the tool-less, thinking-off summary (cache miss, as before 28
                # September).
                self.workspace.append("compaction_prefix_refused", step=self.steps, status=exc.response.status_code)
                prefill = None
                text, data, seconds, estimate = await self.aside_request(
                    messages, prompt, cap=self.compaction_summary_cap or self.cap, floor=2048, purpose="summary", record_purpose="compaction",
                )
            if not text:
                # 80k-three v1 (25 September): 1 of 22 compactions came back with no content (29 tokens)
                # and the raise ended ft09 at level 4 with 450 s of budget left. Retry once, then fall
                # back to a mechanical handoff; a game is never ended by a failed summary.
                self.workspace.append("compaction_empty", step=self.steps, attempt=attempts, usage=data.get("usage", {}))
                retry, data2, seconds2, _ = await self.aside_request(
                    messages, prompt + SUMMARY_RETRY_SUFFIX, cap=self.compaction_summary_cap or self.cap, floor=2048,
                    purpose="summary", record_purpose="compaction", gameplay_prefix=gameplay_prefix,
                    prefill=prefill,
                )
                attempts, seconds = 2, seconds + seconds2
                text, data = retry, data2
        except ContextLengthRejected as exc:
            # Not even the 2,048-token floor fits beside the exact count: no summary can go out, so the
            # mechanical handoff below keeps the game going.
            rejected = True
            self.workspace.append("compaction_rejected", step=self.steps, attempt=attempts, detail=str(exc)[:500])
            if estimate is None:
                estimate = self.estimate_prompt_tokens(messages, None)
        fallback = not text
        if fallback:
            if not rejected:
                self.workspace.append("compaction_empty", step=self.steps, attempt=attempts, usage=data.get("usage", {}))
            text = mechanical_summary(steps=self.steps, levels=self.obs.levels if self.obs else 0,
                                      total_levels=getattr(self.obs, "total_levels", None) if self.obs else None)
        inventory_chars = None
        if self.fix("inventory"):
            inventory = await self.inventory_block()
            inventory_chars = len(inventory)
            text = text.rstrip() + "\n\n" + inventory
        self.context_compactions += 1
        self.compaction_calls += 1
        self.game.compactions += 1
        self.game.compaction_seconds += seconds
        # The cached prefix is gone: the priority scheduler's handover point (a no-op for every other policy).
        getattr(self.game, "note_compaction", lambda: None)()
        self.workspace.append(
            "compaction",
            step=self.steps,
            estimate=estimate,
            summary_chars=len(text),
            seconds=round(seconds, 3),
            attempts=attempts,
            mechanical_fallback=fallback,
            usage=data.get("usage", {}),
            **({} if mode == DEFAULT_COMPACTION_THINKING else {"thinking": mode}),
            **({} if inventory_chars is None else {"inventory_chars": inventory_chars}),
        )
        if self.harness_v2 is not None:
            self.harness_v2.note_compaction(text)
        return text

    async def run_explorer(self, *, max_actions, max_seconds, should_stop):
        """fixed-2x-explorer: explore the current level in a worker thread (the caller holds
        ``env_lock`` while the agent waits for a slot), then bring the session, the Duck state and
        the scheduler up to date and carry the report from the next request on."""
        from arc_infra.perception import Observation
        from inference.agent.action_names import to_model_action

        from .explorer import explore_level, report_message

        before = self.obs
        level = before.levels + 1
        result = await asyncio.to_thread(
            explore_level, self.env, before.board, before.levels, before.available,
            max_actions=max_actions, max_seconds=max_seconds, should_stop=should_stop)
        raw = result.pop("raw")
        if raw is not None:
            self.previous = before
            self.obs = Observation.from_raw(raw, before)
        taken = result["actions"]
        self.steps += taken
        if taken:
            # The harness's per-game cap (duck_controls max_actions, 1,200) bounds the model's own
            # actions; explorer actions must not use it up (explorer-v1 and the demo probes ended
            # every explored game at this cap). The Config is shared by all agents: copy it first.
            self.config = copy.copy(self.config)
            self.config.max_actions += taken
        self.last_observation_at = time.monotonic()
        self.last_action = "EXPLORER"
        cleared = self.obs.levels > before.levels
        self.no_ops.clear()
        self.level_steps = 0 if cleared else self.level_steps + taken
        self.resets += result["resets"]
        if hasattr(self, "level_start_step"):
            self.level_start_step = self.steps
        if hasattr(self, "stuck"):
            self.stuck.reset_level()
        self.auto_reset = None
        self.last_game_over = None
        self.workspace.observe(self.steps, self.obs, {"action": "EXPLORER", "count": taken})
        if hasattr(self, "write_duck_state"):
            self.write_duck_state("EXPLORER", {
                "executed": True, "action_num": self.steps, "explorer": True, "explorer_actions": taken,
                "level_completed": cleared, "state": self.obs.state, "game_over": False,
                "done": self.obs.state == "WIN", "board_changed": True})

        def display(action):
            name, x, y = action
            return f"MOUSE(row={y}, col={x})" if name == "ACTION6" else to_model_action(name)

        self.explorer_message = report_message(result, level, display) if taken else None
        record = {"game": self.game.game_id, "level": level, "wall_seconds": round(self.game.clock.global_clock.elapsed(), 1),
                  **{k: v for k, v in result.items() if k not in ("start_frame", "prewin_frame", "path", "effects")},
                  "path": [display(a) for a in result["path"]] if result["path"] else None,
                  "effects": [list(row) for row in result["effects"][:12]],
                  "report_chars": len(self.explorer_message["content"]) if self.explorer_message else 0}
        self.explorer_runs.append(record)
        self.workspace.append("explorer", step=self.steps, **record)
        self.game.report_level(self.obs.levels, self.game.clock.global_clock.elapsed(),
                               total_levels=getattr(self.obs, "total_levels", None))
        if self.obs.state == "WIN":
            self.game.note_finished("won")
        return record

    def result(self):
        row = {**super().result(), "arena_compactions": self.game.compactions if self.game else 0}
        if self.notes is not None:
            row["cross_game_notes"] = {**self.notes_stats, "notebook_lines": len(self.notes.lines)}
        if self.memory is not None:
            row["memory_block"] = {**self.memory_stats, "block_chars": len(self.memory_block or "")}
        if self.escalation is not None:
            row["model_escalation"] = self.escalation.summary()
        if getattr(self, "procedure", None) is not None:
            row["stuck_procedure"] = self.procedure.summary()
        if self.snapshot is not None:
            row["game_snapshot"] = self.snapshot_stats
            if self.snapshot["every_level"]:
                row["level_snapshots"] = list(self.level_snapshots)
        if self.harness is not None:
            row["duck_harness"] = self.harness_stats
        if self.explorer is not None:
            row["explorer"] = self.explorer_runs
        if self.critical_errors:
            row["critical_notes"] = self.critical_notes
        if self.harness_v2 is not None:
            row["harness_v2"] = self.harness_v2.summary()
        if self.adaptive is not None:
            row["adaptive_thinking"] = {**self.adaptive.summary(), "light": dict(self.light_stats)}
        if self.worldmodel_notice:
            row["worldmodel_notices"] = self.worldmodel_notices
        if self.harness_fixes is not None:
            row["harness_fixes"] = {**self.fix_stats, "fixes": list(self.harness_fixes["fixes"]),
                                    "sandbox_interrupts": self.sandbox.interrupts,
                                    **({"stuck": self.stuck.summary()} if self.fix("stuck_notice") else {})}
        if self.sampler is not None:
            row["parallel_samples"] = self.sampler.summary()
        if self.action_plans:
            row["action_plans"] = self.plan_summary()
        if self.drain is not None:
            row["context_drain"] = dict(self.drain_stats)
        if self.server_error_retries:
            row["server_error_retries"] = self.server_error_retries
        if self.lrule is not None:
            row["loop_rule"] = {**self.lrule_stats, "action": self.lrule["action"]}
        if self.fresh is not None:
            row["fresh_attempt"] = dict(self.fresh_stats)
        if self.dfr is not None:
            row["dfr_harness"] = {**self.dfr_stats, "flags": dict(self.dfr)}
            store = getattr(self, "animation_store", None)
            if store is not None:
                row["dfr_harness"]["animation_store"] = {"steps": len(store.items), "bytes": store.bytes,
                                                         "stored": store.stored, "dropped": store.dropped}
            if self.sc_on():
                row["sim_check"] = dict(self.sc_stats)
        if self.dloop is not None:
            row["dfr_loop"] = {**self.loop_stats, "flags": dict(self.dloop)}
        if self.hx is not None:
            row["hx_harness"] = {**self.hx_stats, **({"reflect": self.hx_reflector.summary()}
                                                      if self.hx_reflector is not None else {}),
                                 **({"perceive": self.hx_perceiver.summary()} if self.hx_perceiver is not None else {})}
        return row

    def plan_summary(self):
        from .action_plans import summary

        return summary(self.sandbox.plans, actions=self.steps, replies=self.gameplay_replies,
                       completion_tokens=self.gameplay_completion_tokens)

    async def drain_worker(self, timeout=35):
        try:
            await super().drain_worker(timeout)
        finally:
            if self.sampler is not None:
                await self.sampler.close()  # observed losers still running when the game ends


# fixed-2x-worldmodel (27 September): deep levels cost ~20K tokens each, as much as level 1, and every level
# starts with the same low-clear warm-up; offered tools go unused (check_model: no call in harness-v2-v1), so
# the consolidation step is asked for in the level-up turn itself.
WORLDMODEL_NOTICE = (
    "LEVEL {cleared} CLEARED. Consolidate before your first action on the new level; the mechanics usually "
    "carry over, and encoding them once is cheaper than re-deriving them. (1) In one python call write "
    "`read_state(frame)` (a compact hashable state: positions of what matters) and `sim(state, action)` (the next "
    "state for an action string as recorded in `transitions`, e.g. 'UP' or 'MOUSE(row=r, col=c)'), reusing your "
    "earlier helpers, and run `print(check_sim(sim, read_state))`: it replays every recorded transition of this "
    "game. Fix `sim` until it reproduces them (at most two more calls). (2) Define `goal(state)` for this board and "
    "run `p = plan(sim, read_state(current_frame), goal, actions=[...])`; execute it with "
    "`print(run_plan(p, sim, read_state))`, which stops at the first surprise. (3) New elements may need new rules: "
    "probe them once, update `sim`, plan again."
)


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


LEVEL_PROBE_PENDING = (
    "Not executed: the level was just cleared and the new level's deterministic probe runs before your next "
    "turn (its [Level snapshot] arrives with that turn). End this call; plan the new level from the snapshot."
)


class ArenaTransferDuckAgent(ArenaDuckMixin, TransferDuckAgent):
    """The corrected Duck agent (v5 prompt layout, persistent sandbox) with append-only context."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile = "duck-transfer-arena"

    def analyzer(self, loop):
        self.loop = loop
        analyzer = super().analyzer(loop)
        if text_only(self.config.model):
            from .text_observations import bind_duck

            bind_duck(analyzer)
        bind_compaction(analyzer, self)
        if self.harness_fixes is not None:
            analyzer._system_prompt = self.fixed_system_prompt(analyzer._system_prompt)
        if self.harness is not None:
            addition = self.harness.prompt_text()
            if addition:
                analyzer._system_prompt = analyzer._system_prompt.rstrip() + "\n\n" + addition
        if self.action_plans:
            from .action_plans import PROMPT as PLAN_PROMPT

            analyzer._system_prompt = analyzer._system_prompt.rstrip() + "\n\n" + PLAN_PROMPT
        if self.dfr is not None:
            analyzer._system_prompt = dfr_module.system_prompt(analyzer._system_prompt, self.dfr)
        if self.hx is not None:
            from . import hx_harness as hx_module

            analyzer._system_prompt = hx_module.system_prompt(analyzer._system_prompt, self.hx)
        if self.dloop is not None:
            self.loop_analyzer(analyzer)
        analyzer._max_output_tokens = self.cap
        analyzer._reply_reserve_tokens = self.cap
        analyzer._context_budget_tokens = self.context_budget_tokens(analyzer)
        self.configure_drain_analyzer(analyzer)
        self._arena_analyzer = analyzer
        if self.game.ready is not None:
            self.game.ready.set()
        return analyzer


class ArenaPublicDuckAgent(ArenaDuckMixin, DuckAgent):
    """The public ToolAgent (public prompts, tools, ephemeral sandbox, file state) through the
    same bridge, with xhigh effort, the completion cap, compaction and the gate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile = "duck-public-arena"

    def analyzer(self, loop):
        self.loop = loop
        owner = self

        class PublicBridgedToolAgent(CompactionMixin, ObservationPreservingTrim, self.module.ToolAgent):
            # ObservationPreservingTrim contributes only the calibrated token estimator
            # (its trimming is overridden by CompactionMixin, which comes first).
            def _chat_completion(self, messages, *, tools, request_timeout_seconds=None):
                future = asyncio.run_coroutine_threadsafe(owner.chat(messages, tools), loop)
                owner.chat_future = future
                try:
                    return future.result()
                finally:
                    owner.chat_future = None

        analyzer = PublicBridgedToolAgent(
            model=self.config.model, base_url=self.config.base_url, provider="vllm", timeout=900
        )
        analyzer._owner = self
        self._analyzer = analyzer
        analyzer._image_tokens = self.image_tokens
        analyzer._chars_per_token = self.chars_per_token
        analyzer._max_output_tokens = self.cap
        analyzer._reply_reserve_tokens = self.cap
        analyzer._context_budget_tokens = self.context_budget_tokens(analyzer)
        self.configure_drain_analyzer(analyzer)
        self._arena_analyzer = analyzer
        if self.game.ready is not None:
            self.game.ready.set()
        return analyzer

    def write_duck_state(self, action="", result=None):
        # The public analyze/python tool read the state file; keep it current per action.
        super().write_duck_state(action, result)
        if self.duck_history:
            self.flush_duck_state()

    async def step_duck(self, arguments):
        # Public behaviour: no repeat refusals and no RESET refusal at a level's first action.
        self.turn_refusals = REFUSALS_PER_TURN
        self.level_start_step = -1
        return await super().step_duck(arguments)
