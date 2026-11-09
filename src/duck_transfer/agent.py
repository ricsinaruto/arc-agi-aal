"""Run the public Duck tool loop against our timed, independently scored game fleet."""

import asyncio
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import httpx
import requests

from arc_infra.perception import Observation, board_hash
from arc_infra.plans import Action, integer
from arc_infra.session import GameSession

from .feedback import StuckTracker, TransitionCache, no_effect_here, transition_summary
from .loader import load
from .plumbing import PlumbingMixin
from .prompting import PromptMixin, system_prompt
from .timing import CompetitionClock
from .trimming import ObservationPreservingTrim, image_tokens_for

THINK_TAG = re.compile(r"</?think>", flags=re.IGNORECASE)  # the public loop's _THINK_TAG_RE


def verbatim_history_text(field, message, normalized):
    """The reply's ``field`` ("reasoning" or "content") as the server sent it, ``<think>`` tags removed only.

    ``normalized`` (the public loop's text) decides only whether the field is kept at all."""
    if field == "reasoning":
        raw = message.get("reasoning")
        if raw in (None, ""):
            raw = message.get("reasoning_content")
    else:
        raw = message.get(field)
    if not isinstance(raw, str) or not normalized:
        return normalized
    return THINK_TAG.sub("", raw)


class ContextLengthRejected(requests.RequestException):
    """An explicit HTTP 400 that Duck can recover by reducing its request."""


class GameplayRequestExpired(TimeoutError):
    """A model HTTP timeout observed after its gameplay allowance expired."""


# Refused repeats do not end an analyzer turn, so the public loop would re-ask forever if
# the model insisted; after this many refusals in one turn the repeat executes, flagged.
REFUSALS_PER_TURN = 2

RESET_REFUSED = (
    "RESET as the first action of a fresh level does nothing (the competition API ignores it "
    "and an offline engine may restart the whole game); the board is already at the level's "
    "start, act on it instead. RESET restarts only the current level once you have acted."
)


class DuckAgent(GameSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile = "duck-interface"
        self.stopping = threading.Event()
        self.chat_future = None
        self.worker = None
        # Duck's synchronous actors wait on asynchronous callbacks. Sharing the
        # default executor with environment steps can deadlock a 25-game fleet.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="duck-actor")
        self.duck_history = []
        self.backend_error = None
        self.module = load()
        self.competition_clock = None
        self.transition_results = {}
        self.length_truncations = 0
        # v5 plumbing: the runtime state lives here; the file is written at start and end.
        self.frame_payloads = {}
        self.state_flushes = 0
        self.prompt_log_writes = 0
        # v5 feedback: receipts, repeat refusal, stuck notices, GAME_OVER bookkeeping.
        self.transitions = TransitionCache(component_limit=self.config.transition_component_limit)
        self.stuck = StuckTracker(self.config)
        self.level_start_step = 0
        self.last_game_over = None
        self.auto_reset = None
        self.turn_refusals = 0
        self.repeat_actions = 0
        # v5b: the server tokenizer checks every request before it is sent, and its counts
        # calibrate the trimmer's characters-per-token ratio and image-token estimate.
        # v5c: image tokens come from each PNG's geometry (Qwen3-VL: one per 32x32 pixels;
        # the v5b calibration split one equation into two unknowns and ran to its clamps),
        # the tokenizer count's residual against prompt_tokens is calibrated instead, and
        # every sample is logged (``token_count`` events).
        self.chars_per_token = float(ObservationPreservingTrim.CHARS_PER_TOKEN)
        self.image_tokens = self.config.image_tokens_estimate
        self.count_residual = 0.0
        self.token_count_samples = 0
        self.context_prechecks = 0
        self.token_count_errors = 0
        self._analyzer = None
        from .history import preserve_transition_results
        from .sandbox import StatefulSandbox, stateful_bootstrap

        original = self.module.run_sandboxed_python
        # One persistent interpreter per game (definitions survive between tool calls).
        self.sandbox = StatefulSandbox(
            stateful_bootstrap(
                preserve_transition_results(original.__globals__["_SANDBOX_BOOTSTRAP"])
            ),
            color_chars=original.__globals__["ARC_COLOR_CHARS"],
            on_event=lambda kind, **data: self.workspace.append(kind, step=self.steps, **data),
            label=self.game_id,
        )
        self.sandbox_runner = self.sandbox.run

    def configure_fleet_timing(self, started, sessions):
        self.competition_clock = CompetitionClock(started, sessions)
        return self.competition_clock.metadata()

    async def before_first_turn(self):
        """Hook between environment creation and the first analyzer turn (no-op here; the
        arena's game-start snapshot probes the board from it)."""
        return None

    async def before_turn(self):
        """Hook on the event loop before each analyzer turn (the worker is idle; no-op here; the
        arena's level probe takes a new level's snapshot from it)."""
        return None

    def write_duck_state(self, action="", result=None):
        from inference.agent.runtime_state import Frame, HistoryEntry

        frame = Frame(
            grid=tuple(tuple(int(c) for c in row) for row in self.obs.board),
            step=self.steps,
            level=max(1, min(self.obs.total_levels, self.obs.levels + 1)),
        )
        self.duck_history.append(HistoryEntry(action=action, frame=frame))
        if action:
            self.transition_results[self.steps] = dict(result or {
                "executed": True, "action_num": self.steps, "state": self.obs.state,
                "reset": action == "RESET", "game_over": self.obs.state == "GAME_OVER",
                "done": self.obs.state == "WIN",
            })
        if self.obs.full_reset:
            self.workspace.append("full_reset", step=self.steps, action=action)

    def tried_no_effect_here(self):
        return no_effect_here(self.no_ops, self.obs.levels, self.obs.board)

    # Hooks of the no-change refusal and the stuck count (the arena's harness v2 overrides them
    # with the effect-aware change; these defaults are the full-board rule).

    def no_effect_repeat(self, action):
        """True when ``action`` already changed nothing on this exact board (refused repeat)."""
        return (self.obs.levels, board_hash(self.obs.board), action.identity()) in self.no_ops

    async def action_effect(self, action, label, payload, before, before_levels, changed):
        """Annotate one executed action's ``payload``; returns the change the stuck tracker counts."""
        return changed

    def request_template_override(self):
        """Per-request chat-template fields over the config's (the arena's adaptive thinking); None keeps them."""
        return None

    def no_change_repeat_error(self, skipped):
        return (
            f"{', '.join(skipped)} already produced no board change on this exact "
            "board; choose a different action. Tried here with no effect: "
            + ", ".join(self.tried_no_effect_here())
        )

    def flush_duck_state(self):
        """Write the runtime state file (once at start, once at the end: evidence only)."""
        from inference.agent.runtime_state import write_runtime_state

        write_runtime_state(
            self.state_path, current_frame=self.duck_history[-1].frame, history=self.duck_history
        )
        self.state_flushes += 1

    def runtime_state(self, path=None):
        """In-memory replacement for the public ``load_runtime_state``."""
        return self.duck_history[-1].frame, list(self.duck_history)

    def ascii_history(self, history):
        """Per-entry payloads memoised by step (frames are immutable); results attached."""
        entries = []
        for entry in history:
            step = entry.frame.step
            payload = self.frame_payloads.get(step)
            if payload is None:
                payload = self.frame_payloads[step] = self.module._ascii_history_view_payload(
                    [entry]
                )[0]
            payload = dict(payload)
            payload["result"] = dict(self.transition_results.get(step, {}))
            entries.append(payload)
        return entries

    def valid_actions(self):
        from inference.agent.action_names import to_model_actions

        return to_model_actions(self.obs.available)

    async def request_completion(self, payload, record_extra=None):
        """Classify timeouts at the request, before later cleanup can blur their timing.

        ``record_extra``: telemetry fields for the request record (the arena's parallel samples); never sent."""
        remaining = self.remaining()
        if remaining <= 0:
            raise GameplayRequestExpired("Gameplay deadline reached before request")
        extra = {"record_extra": record_extra} if record_extra else {}
        try:
            return await self.client.client.post(
                "chat/completions", json=payload, timeout=remaining, **extra
            )
        except httpx.TimeoutException as exc:
            if self.remaining() > 0:
                raise
            self.workspace.append(
                "deadline_request_timeout", step=self.steps, error_type=type(exc).__name__,
                request_allowance_seconds=remaining,
            )
            raise GameplayRequestExpired("Model request exhausted gameplay allowance") from exc

    # The arena's harness_fixes settings (arena.harness_fixes); None here and on every earlier arena build.
    harness_fixes = None

    def fix(self, name):
        """True when the arena's ``harness_fixes`` sub-fix ``name`` is on (never outside the arena)."""
        return False

    def gameplay_payload(self, messages, tools, override=None):
        """The gameplay request for ``messages``: the public payload, the config's reasoning effort, and the
        per-request template ``override`` (the arena's adaptive thinking)."""
        module = self.module
        payload = module.build_chat_payload(
            provider="vllm",
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_output_tokens or None,
            temperature=self.config.temperature,
            top_p=module._LOCAL_ANALYZER_TOP_P,
            top_k=module._LOCAL_ANALYZER_TOP_K,
            thinking=self.config.enable_thinking,
            tools=tools,
            tool_choice=module._request_tool_choice(tools),
            seed=-1,
        )
        # Effort is the template's own control (the public loop sends none); v5 caps output.
        payload["chat_template_kwargs"]["reasoning_effort"] = self.config.reasoning_effort
        if override:
            payload["chat_template_kwargs"].update(override)
        return payload

    async def chat(self, messages, tools):
        module = self.module
        start = time.monotonic()
        payload = self.gameplay_payload(messages, tools, self.request_template_override())
        sent_effort = (payload["chat_template_kwargs"]["reasoning_effort"]
                       if payload["chat_template_kwargs"].get("enable_thinking", True) else "none")
        try:
            admission_count = (await self.prepare_admission(payload)
                               if hasattr(self, "prepare_admission") else None)
            async with self.client.slots:
                if self.stopping.is_set() or self.remaining() <= 0:
                    raise TimeoutError("Gameplay deadline reached")
                # Refresh after admission; change only the outbound payload.
                if self.competition_clock is not None:
                    now = time.monotonic()
                    payload["messages"] = self.competition_clock.annotate(messages, now)
                    self.workspace.append("planning_budget", **self.competition_clock.snapshot(now))
                self.workspace.append(
                    "model_request",
                    step=self.steps,
                    runtime=json.loads(
                        self.runtime_message(record=False)["content"].split("\n", 1)[1]
                    ),
                )
                count, images, text_chars, image_tokens = admission_count or await self.count_prompt(
                    payload["messages"], tools, payload.get("chat_template_kwargs")
                )
                if count is not None:
                    estimate = count + image_tokens + max(0, round(self.count_residual))
                    limit = (
                        self.config.max_model_len
                        - (self.config.max_output_tokens or 512)
                        - module._REQUEST_SAFETY_MARGIN_TOKENS
                    )
                    if estimate > limit:
                        self.context_prechecks += 1
                        self.workspace.append(
                            "context_precheck", step=self.steps, count=count, images=images,
                            estimate=estimate, limit=limit,
                        )
                        raise ContextLengthRejected(
                            f"client precheck: {estimate} prompt tokens exceed the {limit}-token "
                            "budget; reduce the length of the input prompt"
                        )
                response = await self.request_completion(payload)
                if response.status_code == 400 and module._is_context_length_error(
                    ValueError(response.text)
                ):
                    # The public loop catches requests.RequestException, whereas
                    # this bridge uses httpx. Preserve that recovery contract only
                    # for a definite context rejection, never an ambiguous request.
                    raise ContextLengthRejected(response.text)
                response.raise_for_status()
                data = response.json()
            choice = data["choices"][0]
            self.calibrate(
                count, images, image_tokens, text_chars, (data.get("usage") or {}).get("prompt_tokens")
            )
            if choice.get("finish_reason") == "length":
                self.length_truncations += 1
                self.workspace.append(
                    "length_truncation",
                    step=self.steps,
                    completion_tokens=(data.get("usage") or {}).get("completion_tokens"),
                )
            reply = {
                "message": choice["message"],
                "finish_reason": choice["finish_reason"],
                "usage": data.get("usage", {}),
                "seconds": time.monotonic() - start,
                "force_action": False,
                "reasoning_effort": sent_effort,
                "max_output_tokens": self.config.max_output_tokens or None,
            }
            self.record_reply(reply)
            return module._ChatCompletionResult(
                message=reply["message"],
                finish_reason=reply["finish_reason"],
                usage=reply["usage"],
            )
        except ContextLengthRejected as exc:
            self.workspace.append("context_rejection", step=self.steps, detail=str(exc)[:2000])
            raise
        except (GameplayRequestExpired, asyncio.CancelledError):
            raise
        except Exception as exc:
            # An early timeout or unrelated failure remains an error even if the
            # analyzer consumes it near shutdown. Only the tagged deadline is expected.
            if not isinstance(exc, TimeoutError):
                self.backend_error = f"{type(exc).__name__}: {exc}"
            raise

    def engine_action(self, label):
        """The engine action for a model label (the arena's dfr_harness also accepts UNDO for ACTION7)."""
        from inference.agent.action_names import to_engine_action

        return label if label in self.obs.available else to_engine_action(label)

    def model_action(self, name):
        """The model label of an engine action name (the arena's dfr_harness advertises ACTION7 as UNDO)."""
        from inference.agent.action_names import to_model_action

        return to_model_action(name)

    def batch_stop(self, requested, position, verdicts):
        """A stop for the rest of a batch after the action at ``position`` (the arena's dfr_harness no-op stop), or
        None. ``verdicts``: one ``(label, payload or None)`` per action tried (None: refused)."""
        return None

    async def step_duck(self, arguments):
        raw = arguments.get("actions")
        if raw is None:
            raw = [arguments]
        if not isinstance(raw, list) or not 1 <= len(raw) <= self.config.max_plan:
            return {"executed": False, "error": f"Use 1..{self.config.max_plan} action objects."}
        actions = []
        try:
            for item in raw:
                label = str(item.get("action") or "").strip().upper()
                # Upstream's aliases cover ACTION1..6 only, but the environment
                # can advertise additional engine actions (currently ACTION7).
                # Preserve advertised names without modifying the public package.
                name = self.engine_action(label)
                if name not in set(self.obs.available) | {"RESET"}:
                    raise ValueError(f"Unavailable action: {label or '<empty>'}")
                actions.append(
                    Action(
                        name,
                        integer(item.get("col"), 0, 63, "col") if name == "ACTION6" else None,
                        integer(item.get("row"), 0, 63, "row") if name == "ACTION6" else None,
                    )
                )
        except (ValueError, AttributeError) as exc:
            return {"executed": False, "error": str(exc)}

        def display(action):
            return (
                f"MOUSE(row={action.y}, col={action.x})"
                if action.name == "ACTION6"
                else self.model_action(action.name)
            )

        executed, skipped = [], []
        verdicts = []
        board_changed = False
        total_reward = 0.0
        payload = {"executed": False}
        stop = {}
        for position, action in enumerate(actions):
            if (
                self.stopping.is_set()
                or self.remaining() <= 0
                or self.steps >= self.config.max_actions
                or self.obs.state != "NOT_FINISHED"
            ):
                stop = {"stop_reason": "terminal_or_budget"}
                break
            if action.name not in set(self.obs.available) | {"RESET"}:
                stop = {"stop_reason": "invalid_action"}
                break
            if action.name == "RESET" and self.steps == self.level_start_step:
                stop = {"stop_reason": "reset_refused", "stop_detail": RESET_REFUSED}
                break
            repeat = self.no_effect_repeat(action)
            if repeat and self.turn_refusals < REFUSALS_PER_TURN:
                # Same action on the exact same board already changed nothing: refuse it.
                skipped.append(display(action))
                verdicts.append((display(action), None))
                self.workspace.append(
                    "action_skipped", step=self.steps, reason="no_change_repeat", **action.payload()
                )
                continue
            if repeat:
                self.repeat_actions += 1
            before, before_levels = self.obs.board, self.obs.levels
            before_hash = board_hash(before)
            await self.step(action)
            label = display(action)
            executed.append(label)
            changed = before_hash != board_hash(self.obs.board)
            transition = await asyncio.to_thread(
                transition_summary, self.transitions, before, self.obs.board
            )
            transition["no_change_repeat"] = repeat
            payload = {
                "executed": True,
                "action_num": self.steps,
                "level": max(1, min(self.obs.total_levels, self.obs.levels + 1)),
                "score": self.obs.levels,
                "reward": (self.obs.levels - before_levels) / max(1, self.obs.total_levels),
                "state": self.obs.state,
                "valid_actions": self.valid_actions(),
                "board_changed": changed,
                "done": self.obs.state == "WIN",
                "level_completed": self.obs.levels > before_levels,
                "game_over": self.obs.state == "GAME_OVER",
                "run_complete": self.obs.state == "WIN",
                "action_name": action.name,
                "action_display": label,
                "action_data": {"row": action.y, "col": action.x}
                if action.name == "ACTION6"
                else {},
                "transition": transition,
            }
            effective = await self.action_effect(action, label, payload, before, before_levels, changed)
            tried = self.tried_no_effect_here()
            if tried:
                payload["tried_no_effect_here"] = tried
            self.auto_reset = None
            self.stuck.note_action(effective)
            note_label = getattr(self.stuck, "note_label", None)  # arena procedure tracker (collapse detector)
            if note_label is not None:
                note_label(label)
            if payload["level_completed"]:
                self.level_start_step = self.steps
                self.stuck.reset_level()
            if payload["game_over"]:
                self.last_game_over = {"step": self.steps, "actions": list(executed)}
            self.write_duck_state(label, payload)
            verdicts.append((label, payload))
            board_changed |= changed
            total_reward += payload["reward"]
            if payload["level_completed"] or payload["game_over"] or payload["done"]:
                stop = {
                    "stop_reason": "run_complete"
                    if payload["done"]
                    else "game_over"
                    if payload["game_over"]
                    else "level_completed"
                }
                break
            batch = self.batch_stop([display(a) for a in actions], position, verdicts)
            if batch:
                stop = batch
                break
        if skipped:
            self.turn_refusals += 1
        if skipped and not executed and not stop:
            stop = {
                "stop_reason": "no_change_repeat",
                "error": self.no_change_repeat_error(skipped),
                "transition": {"no_change_repeat": True},
            }
        result = {
            **payload,
            **stop,
            "board_changed": board_changed,
            "last_reward": payload.get("reward", 0.0),
            "reward": total_reward,
            "requested_count": len(actions),
            "requested_actions": [display(action) for action in actions],
            "executed_count": len(executed),
            "executed_actions": executed,
            "batch_size": len(actions),
            "batched": len(actions) > 1,
            "stopped_early": len(executed) < len(actions),
        }
        if skipped:
            result["skipped_repeats"] = skipped
        return result

    async def count_prompt(self, messages, tools, template_kwargs=None):
        """``(text tokens, images, text characters, image tokens)`` from the server tokenizer
        and the PNG geometry, or ``(None, images, chars, image tokens)`` when the count fails
        (the request still goes out).

        ``template_kwargs`` are the request's own ``chat_template_kwargs``. v5c counted with
        ``preserve_thinking`` false while the request left it undefined, so the served prompt
        carried every earlier reasoning block and the count ran 8-21K tokens short by the end
        of a game (v5b's "undercount for a reason the run did not log").
        """
        text_only, images, chars, image_tokens = [], 0, 0, 0
        for message in messages:
            content = message.get("content")
            # The trimmer's estimator prices every non-content field (reasoning_content,
            # tool_calls) at the same characters-per-token ratio, so the calibrated ratio
            # has to be measured over the same characters (v5d).
            rest = {key: value for key, value in message.items() if key != "content"}
            if rest:
                chars += len(json.dumps(rest, ensure_ascii=True, sort_keys=True, default=str))
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        images += 1
                        image_tokens += image_tokens_for(part, self.image_tokens)
                        parts.append({"type": "text", "text": "[image]"})
                    else:
                        parts.append(part)
                        chars += len(str(part.get("text", "")) if isinstance(part, dict) else str(part))
                text_only.append({**message, "content": parts})
            else:
                text_only.append(message)
                chars += len(content) if isinstance(content, str) else len(json.dumps(content or ""))
        if tools:
            chars += len(json.dumps(tools, ensure_ascii=True, sort_keys=True, default=str))
        try:
            count = await self.client.count_tokens(
                text_only, time.monotonic() + max(1, self.remaining()), tools=tools or None,
                template_kwargs=template_kwargs,
            )
        except (asyncio.CancelledError, TimeoutError):
            raise
        except Exception as exc:  # noqa: BLE001 - a failed count never blocks the request
            self.token_count_errors += 1
            self.workspace.append("token_count_error", step=self.steps, error=str(exc)[:500])
            return None, images, chars, image_tokens
        return int(count), images, chars, image_tokens

    def calibrate(self, count, images, image_tokens, text_chars, prompt_tokens):
        """Track the tokenizer (EMA): characters per text token, and the residual of
        ``prompt_tokens`` against the text count plus the geometric image tokens. Every
        sample is logged so a drift can be read from the run."""
        if count is None or count <= 0:
            return
        sample = {
            "step": self.steps, "count": count, "images": images, "image_tokens": image_tokens,
            "text_chars": text_chars, "prompt_tokens": prompt_tokens,
        }
        if text_chars:
            ratio = text_chars / count
            self.chars_per_token = min(6.0, max(1.0, 0.5 * self.chars_per_token + 0.5 * ratio))
        if prompt_tokens:
            residual = prompt_tokens - count - image_tokens
            sample["residual"] = residual
            self.count_residual = min(8192.0, max(-4096.0, 0.5 * self.count_residual + 0.5 * residual))
            if images:
                # The per-image estimate the trimmer uses when a PNG cannot be measured.
                self.image_tokens = int(min(4096, max(64, round(image_tokens / images))))
        self.token_count_samples += 1
        sample.update(
            chars_per_token=round(self.chars_per_token, 3),
            count_residual=round(self.count_residual, 1),
        )
        self.workspace.append("token_count", **sample)
        if self._analyzer is not None:
            self._analyzer._chars_per_token = self.chars_per_token
            self._analyzer._image_tokens = self.image_tokens

    def analyzer(self, loop):
        from .sandbox import stateful_system_prompt, stateful_tools

        owner = self

        class BridgedToolAgent(
            ObservationPreservingTrim, PromptMixin, PlumbingMixin, self.module.ToolAgent
        ):
            def _chat_completion(self, messages, *, tools, request_timeout_seconds=None):
                future = asyncio.run_coroutine_threadsafe(owner.chat(messages, tools), loop)
                owner.chat_future = future
                try:
                    return future.result()
                finally:
                    owner.chat_future = None

            def _tools(self, state_path):
                return stateful_tools(super()._tools(state_path))

            def _history_text(self, field, message, normalized, markup_stripped):
                # reasoning_history "verbatim" (29 September): the server's own string, only <think> tags removed,
                # so the next prompt continues the generated tokens exactly (blank lines and paragraph breaks
                # kept). Tool-call markup recovered from the text still takes the normalized path.
                if owner.config.reasoning_history != "verbatim" or markup_stripped:
                    return normalized
                return verbatim_history_text(field, message, normalized)

        analyzer = BridgedToolAgent(
            model=self.config.model, base_url=self.config.base_url, provider="vllm", timeout=900
        )
        analyzer._owner = self
        self._analyzer = analyzer
        analyzer._image_tokens = self.image_tokens
        analyzer._chars_per_token = self.chars_per_token
        # Also cover already-imported upstream modules with cached env values.
        analyzer._yield_seconds = None
        analyzer._system_prompt = system_prompt(stateful_system_prompt(analyzer._system_prompt))
        # v5: the request carries max_tokens, so the trimmer reserves the same amount.
        analyzer._max_output_tokens = self.config.max_output_tokens or None
        analyzer._reply_reserve_tokens = analyzer._max_output_tokens or 512
        analyzer._context_budget_tokens = max(
            1024,
            self.config.max_model_len
            - analyzer._reply_reserve_tokens
            - analyzer._request_safety_margin_tokens,
        )
        return analyzer

    def run_python_tool(self, execute, state_path, arguments):
        return execute(state_path, arguments)

    async def drain_worker(self, timeout=35):
        if self.worker is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(self.worker), timeout=timeout)
        except GameplayRequestExpired:
            self.workspace.append("worker_deadline_drained", step=self.steps)
        except asyncio.CancelledError:
            # Cancelling the synchronous HTTP bridge can terminate the actor with
            # concurrent.futures.CancelledError, translated by run_in_executor.
            # A completed actor is safe to drain; cancellation of this waiter alone
            # (or of its Future) is not evidence that the actor stopped.
            if (self.stopping.is_set() and self.worker.done() and not self.worker.cancelled()
                    and isinstance(self.worker.exception(), asyncio.CancelledError)):
                self.workspace.append("worker_request_cancelled", step=self.steps)
            else:
                raise
        except Exception as exc:
            self.status, self.error = "error", f"Duck worker cleanup: {type(exc).__name__}"
        finally:
            if self.worker.done():
                self.worker = None

    async def run(self):
        self.started = time.monotonic()
        self.state_path = self.workspace.root / "tool_runtime_state.json"
        loop = asyncio.get_running_loop()
        try:
            self.env = await asyncio.to_thread(
                self.arc.make,
                self.game_id,
                scorecard_id=self.card,
                save_recording=False,
            )
            if self.env is None:
                raise RuntimeError("Failed to create environment")
            self.obs = Observation.from_raw(self.env.observation_space)
            self.last_observation_at = time.monotonic()
            self.workspace.observe(0, self.obs)
            self.write_duck_state()
            self.flush_duck_state()  # analyze() requires the file to exist
            analyzer = self.analyzer(loop)

            def step_env(arguments):
                return asyncio.run_coroutine_threadsafe(self.step_duck(arguments), loop).result()

            await self.before_first_turn()
            while self.remaining() > 0 and self.steps < self.config.max_actions:
                await self.before_turn()
                if self.obs.state == "WIN":
                    self.status = "won"
                    break
                if self.obs.state == "GAME_OVER":
                    game_over = self.last_game_over or {"step": self.steps, "actions": []}
                    await self.step(Action("RESET"))
                    self.write_duck_state("RESET")
                    self.auto_reset = {**game_over, "reset_step": self.steps}
                    self.level_start_step = self.steps
                    self.stuck.reset_level()
                    self.workspace.append(
                        "auto_reset",
                        step=self.steps,
                        game_over_step=game_over.get("step"),
                        actions=game_over.get("actions"),
                    )
                self.stuck.note_turn()
                self.turn_refusals = 0
                self.worker = loop.run_in_executor(
                    self.executor,
                    partial(
                        analyzer.analyze,
                        self.state_path,
                        self.steps,
                        valid_actions=self.valid_actions(),
                        step_env=step_env,
                        transcript_path=self.workspace.root / "duck-transcript.txt",
                        analysis_step=self.turns,
                        request_timeout_seconds=self.remaining(),
                        should_stop=lambda: self.stopping.is_set() or self.remaining() <= 0,
                    ),
                )
                result = await asyncio.shield(self.worker)
                self.worker = None
                if self.backend_error:
                    raise RuntimeError(self.backend_error)
                if result is not None and result.retryable_failure and self.remaining() > 0:
                    raise RuntimeError("Duck analyzer could not recover the rejected request")
                if result is None and self.remaining() > 0:
                    raise RuntimeError("Duck analyzer failed; inspect its transcript")
            else:
                self.status = "time_limit" if self.remaining() <= 0 else "action_limit"
        except (TimeoutError, asyncio.CancelledError):
            self.status = "time_limit"
        except Exception as exc:
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            self.stopping.set()
            self.sandbox.close()  # unblocks a python call in progress
            if self.chat_future is not None:
                self.chat_future.cancel()
            await self.drain_worker()
            self.sandbox.close()
            self.executor.shutdown(wait=False, cancel_futures=True)
            if self.duck_history:
                try:
                    self.flush_duck_state()
                except OSError as exc:
                    self.workspace.append("state_flush_failed", error=str(exc))
            self.workspace.append("finished", **self.result())
        return self.result()

    def result(self):
        return {
            **super().result(),
            "agent_profile": self.profile,
            "length_truncations": self.length_truncations,
            "repeat_actions": self.repeat_actions,
            "state_flushes": self.state_flushes,
            "prompt_log_writes": self.prompt_log_writes,
            "context_prechecks": self.context_prechecks,
            "token_count_errors": self.token_count_errors,
            "chars_per_token": round(self.chars_per_token, 3),
            "image_tokens": self.image_tokens,
            "count_residual": round(self.count_residual, 1),
            "token_count_samples": self.token_count_samples,
        }
