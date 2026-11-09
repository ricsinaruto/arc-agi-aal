"""Qwen baseline C (the Pi coding agent) and the shared C/R plumbing in the arena.

``ArenaRequestLayer`` gives each game its own request layer over the shared HTTP client: the
game's active clock feeds the clock note, the KV gate replaces the semaphore, and the
completion cap is applied to every gameplay request. ``ArenaBridge`` reports level clears to
the scheduler. ``ArenaPiAgent`` parks at its ``admit`` hook (Pi's next request is refused and
the running turn interrupted) and resumes with a continuation prompt.
"""

import asyncio
import base64
import json
import time

from arc_infra.model_templates import text_only
from qwen_baselines.bridge import Bridge
from qwen_baselines.evidence import png
from qwen_baselines.model import RequestLayer
from qwen_baselines.pi_agent import PiAgent


class Parked(TimeoutError):
    """Raised inside the gateway when the game is parked: the turn is interrupted."""


class ArenaRequestLayer(RequestLayer):
    """One per game: the active clock for notes/timeouts, the gate slot for admission."""

    def __init__(self, config, game, gate_slot, *, client, tokenizer_model_max_length=None):
        super().__init__(config, game.clock, client=client)
        self.game = game
        self.slots = gate_slot
        self.cap = int(config.output_tokens_cap or 0) or None
        self.tokenizer_model_max_length = tokenizer_model_max_length
        if tokenizer_model_max_length is not None:
            # Pennyroyal /tokenize reports tokenizer metadata, not the configured
            # (YaRN-extended) serving context. Only accept that documented split
            # when the separately measured server capacity matches our request layer.
            layout = getattr(gate_slot.gate.budget, 'layout', {})
            startup = layout.get('startup', {})
            if (layout.get('units') != 'tokens'
                    or startup.get('context_len') != config.context_window
                    or not gate_slot.window <= tokenizer_model_max_length <= config.context_window):
                raise ValueError('Tokenizer metadata override requires verified SGLang serving capacity')

    def inference_scope(self, summary=False):
        return self.clock.inference("summary" if summary else "gameplay")

    def validate_tokenizer_context(self, data):
        if self.tokenizer_model_max_length is None:
            return super().validate_tokenizer_context(data)
        if data.get('max_model_len') != self.tokenizer_model_max_length:
            raise RuntimeError('SGLang tokenizer metadata differs from the explicitly declared checkpoint limit: '
                               f"{data.get('max_model_len')} != {self.tokenizer_model_max_length}")

    async def complete_payload(self, payload, archive, *, summary=False, stream=False, max_tokens=None):
        # Tokenization and waiting both consume the global gameplay deadline. The
        # final timing note is refreshed after admission, so reserve a small margin
        # and verify the *actual* request against that reservation before sending.
        started = time.monotonic()
        async with asyncio.timeout(self.clock.remaining()):
            count = await self.count(self.fresh(payload))
            self.slots.reserve(count + (256 if self.slots.gate.budget else 0))
            response = await super().complete_payload(
                payload, archive, summary=summary, stream=stream, max_tokens=max_tokens or self.cap
            )
        self.game.max_prompt_tokens = max(self.game.max_prompt_tokens, count)
        if summary:
            self.game.compactions += 1
            self.game.compaction_seconds += time.monotonic() - started
        return response

    def validate_admission(self, count, allowance):
        budget = self.slots.gate.budget
        if budget is not None:
            reservation = self.slots._stack[-1]
            if budget.cost(count + allowance) > reservation.tokens:
                raise RuntimeError("Refreshed Pi request exceeds its measured cache reservation")

    async def close(self):
        return None  # the trial owns the shared client


class ArenaBridge(Bridge):
    def __init__(self, *args, game, **kwargs):
        super().__init__(*args, **kwargs)
        self.game = game

    def record(self, raw, action=None, *, initial=False):
        before = len(self.clears)
        meta = super().record(raw, action, initial=initial)
        wall = self.game.clock.global_clock.elapsed()
        if len(self.clears) > before:
            self.clears[-1]["wall_seconds"] = round(wall, 3)
        self.game.report_level(meta["levels_completed"], wall)
        if meta["status"] == "WIN":
            self.game.note_finished("won")
        return meta


class ArenaPiAgent(PiAgent):
    def make_executor(self):
        executor = super().make_executor()
        if text_only(self.layer.config.model):
            from .text_observations import pi_api

            api = executor.workspace / 'ARC_API.md'
            api.write_text(pi_api(api.read_text()))
        return executor

    def system_prompt(self):
        text = super().system_prompt()
        if text_only(self.layer.config.model):
            from .text_observations import pi_system

            return pi_system(text)
        return text

    def __init__(self, bridge, layer, *, pi_dir, node="node", game):
        super().__init__(bridge, layer, pi_dir=pi_dir, node=node)
        self.game = game
        self.interrupting = False
        self.parked_turns = 0
        self.resumed_after = None
        self._parked_at_prompt = 0.0

    async def interrupt(self):
        if self.process and not self.interrupting:
            self.interrupting = True
            await self.send({"type": "interrupt"})

    async def admit(self):
        self.turns += 1
        if self.game.clock.parked:
            self.parked_turns += 1
            self.bridge.archive.event("arena_parked_request", turns=self.turns)
            await self.interrupt()
            raise Parked("Game parked at a request boundary")

    def continuation(self):
        text = super().continuation()
        if self.resumed_after is not None:
            text = (
                f"Play resumed after a pause of {self.resumed_after:.0f} s; the board is unchanged.\n"
                + text
            )
        return text

    async def run(self):
        """``PiAgent.run`` with the arena's boundaries: wait while parked before each prompt,
        no whole-game timeout (the trial cancels at the wall end), and a parked turn's
        interrupt is not a model error."""
        initial = True
        errors = 0
        clock = self.game.clock
        try:
            while (
                clock.global_clock.remaining() > 0
                and self.bridge.archive.metadata()["status"] != "WIN"
            ):
                await clock.wait_active()
                paused = clock.parked_seconds() - self._parked_at_prompt
                self.resumed_after = paused if paused > 0 else None
                if clock.global_clock.remaining() <= 0:
                    break
                text = (
                    "Initial observation: "
                    + json.dumps(self.bridge.archive.metadata())
                    + ("\n\nObjects (settled frame):\n" + view if (view := self.object_view()) else "")
                    if initial
                    else self.continuation()
                )
                packet = {"type": "prompt", "text": text}
                self.turns, self.text = 0, ""
                self.interrupting = False
                parked_before = self.parked_turns
                if initial and text_only(self.layer.config.model):
                    from .text_observations import pi_initial

                    packet['text'] += pi_initial(self.bridge.archive)
                elif initial:
                    packet["images"] = [
                        {
                            "type": "image",
                            "mimeType": "image/png",
                            "data": base64.b64encode(png(self.bridge.archive.frame())).decode(),
                        }
                    ]
                self._parked_at_prompt = clock.parked_seconds()
                await self.send(packet)
                await self.receive_until("done", raise_model_errors=False)
                if self.parked_turns > parked_before:
                    self.bridge.archive.event("arena_parked_turn", turns=self.turns)
                elif self.last_model_error:
                    errors += 1
                    self.bridge.archive.event(
                        "pi_turn_error", error=self.last_model_error, consecutive=errors
                    )
                    if errors >= 3:
                        raise RuntimeError(self.last_model_error)
                else:
                    errors = 0
                self.remember()
                if not initial:
                    self.bridge.archive.event("neutral_continuation")
                initial = False
        except (TimeoutError, asyncio.CancelledError):
            self.bridge.archive.event("deadline", seconds=self.bridge.clock.elapsed())
        finally:
            await self.close()
