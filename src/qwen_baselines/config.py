"""Explicit settings for the new comparisons, independent of historical profiles."""

from dataclasses import asdict, dataclass

from arc_infra.model_templates import allowed_efforts
from arc_infra.serving import BASE_URL, MODEL, REVISION

# Three public games chosen on 16 September: ft09 (click-only, animates on about a third
# of actions), lp85 (click-only, rare two-frame animation) and tr87 (arrows only, never
# animates); each cleared four or five levels in the scored public Duck notebook run.
GAMES = ("ft09-0d8bbf25", "lp85-305b61c3", "ar25-0c556536")


@dataclass(frozen=True)
class Config:
    model: str = MODEL
    model_revision: str = REVISION
    base_url: str = BASE_URL
    gameplay_seconds: float = 900
    context_window: int = 262144
    reserve_tokens: int = 16384
    generation_slots: int = 8
    temperature: float = 0.7
    # The pinned chat template accepts only these; its default is xhigh.
    reasoning_effort: str = "xhigh"
    # Observation inputs for A/B. The first Kaggle
    # runs reached 213K-243K tokens in ten minutes with a 64-row text grid per
    # observation and all prior reasoning preserved. Old behaviour: text_grid=True,
    # image_pixels=256, actions_per_turn=None, preserve_thinking=True.
    text_grid: bool = False
    image_pixels: int = 512
    # Baseline A: game actions executed per model response across its act calls (each act
    # call carries an ordered list); None removes the cap. 16 September: 20, like a
    # PRO-LONG plan file; the leaner profile of the same day used 1.
    actions_per_turn: int | None = 20
    # Chat-template flag: keep reasoning only for steps after the last observation.
    preserve_thinking: bool = False
    python_seconds: float = 30
    seed: int = 0
    # A/B history cap: oldest exchanges are trimmed above this many prompt tokens (user
    # decision, 16 September: 80K so three games fit the 298K-token KV pool without
    # preemption). None falls back to context_window - reserve_tokens.
    history_tokens: int | None = 65536
    # v4 (18 September): Duck-style mechanisms for A/B/C. The object view (host-side Duck
    # segmentation) rides on every observation; each action carries a before/after transition
    # summary; grid dumps in tool output are suppressed; an action that already produced no
    # board change from the same board is rejected. P/R/S profiles turn these off.
    object_view: bool = False
    object_view_limit: int = 40
    transition_summary: bool = True
    suppress_grid_dumps: bool = True
    reject_no_change_repeats: bool = True
    # Coding-agent (Pi) compaction trigger for C and P (96K, same decision). None keeps
    # the gateway's context_window - reserve_tokens trigger.
    compaction_tokens: int | None = 65536
    # Bash tool timeouts: default and the largest model-requested value; C keeps 30 s.
    bash_seconds: float = 30
    bash_max_seconds: float = 30
    # Sandbox Python beyond the standard library (numpy). PRO-LONG upstream had stdlib only.
    scientific_python: bool = True
    # PRO-LONG protocol settings (baseline P): actions per plan file, tool turns per call,
    # optional per-game action budget, board text encoding.
    plan_cap: int = 20
    call_turns: int = 50
    max_actions: int | None = None
    grid_mode: str = "ascii"
    # Retrodict v5 (18 September): per-invocation budget. After this many seconds (0 = none;
    # capped at a fraction of the remaining game clock) or with one model request left, the next
    # request is the invocation's last and carries no tools; output_tokens_cap bounds
    # max_tokens per request (0 = the native window allowance).
    invocation_budget_seconds: float = 0
    output_tokens_cap: int = 0

    def __post_init__(self):
        if self.gameplay_seconds <= 0 or self.python_seconds <= 0:
            raise ValueError("Durations must be positive")
        if not 0 < self.reserve_tokens < self.context_window or self.generation_slots < 1:
            raise ValueError("Invalid context/admission settings")
        if self.reasoning_effort not in allowed_efforts(self.model):
            choices = ("native for UI-Mate" if allowed_efforts(self.model) == ("native",) else
                       "high for Nex" if allowed_efforts(self.model) == ("high",) else "xhigh, medium or low")
            raise ValueError(f"reasoning_effort must be {choices}")
        if type(self.image_pixels) is not int or self.image_pixels <= 0 or self.image_pixels % 64:
            raise ValueError("image_pixels must be a positive multiple of 64")
        if self.actions_per_turn is not None and (
            type(self.actions_per_turn) is not int or self.actions_per_turn < 1
        ):
            raise ValueError("actions_per_turn must be None or at least 1")
        if type(self.object_view_limit) is not int or self.object_view_limit < 1:
            raise ValueError("object_view_limit must be a positive integer")
        if self.history_tokens is not None and not (
            type(self.history_tokens) is int
            and 0 < self.history_tokens <= self.context_window - self.reserve_tokens
        ):
            raise ValueError("history_tokens must be within the context window minus reserve")
        if self.compaction_tokens is not None and not (
            type(self.compaction_tokens) is int
            and 0 < self.compaction_tokens <= self.context_window - self.reserve_tokens
        ):
            raise ValueError("compaction_tokens must be within the context window minus reserve")
        if not 0 < self.bash_seconds <= self.bash_max_seconds:
            raise ValueError("bash_seconds must be positive and at most bash_max_seconds")
        if self.plan_cap < 1 or self.call_turns < 1:
            raise ValueError("plan_cap and call_turns must be at least 1")
        if self.max_actions is not None and (
            type(self.max_actions) is not int or self.max_actions < 1
        ):
            raise ValueError("max_actions must be None or at least 1")
        if self.grid_mode not in ("ascii", "hex", "num"):
            raise ValueError("grid_mode must be ascii, hex or num")
        if self.invocation_budget_seconds < 0 or type(self.output_tokens_cap) is not int or self.output_tokens_cap < 0:
            raise ValueError("invocation_budget_seconds and output_tokens_cap must be non-negative")

    def to_dict(self):
        return asdict(self)
