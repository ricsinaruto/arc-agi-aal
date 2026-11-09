"""Explicit, serializable limits shared by Modal and Kaggle."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from .model_templates import allowed_efforts

MODEL_ID = "Qwen/Qwen3.8-27B-FP8"
MODEL_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"


REASONING_HISTORY_MODES = ("normalized", "verbatim")


@dataclass
class Config:
    model: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    base_url: str = "http://127.0.0.1:8000/v1"
    # "vllm" (also SGLang's vLLM-compatible /tokenize) or "llamacpp" (llama-server token counts).
    serving_backend: str = "vllm"
    gameplay_seconds: float = 900
    setup_seconds: float = 600
    parallel_games: int = 25
    max_actions: int = 1200
    max_plan: int = 32
    max_unpredicted_plan: int = 3
    max_context_tokens: int = 18000
    max_output_tokens: int = 2048
    thinking_budget: int = 768
    preserve_reasoning: bool = False
    speculative_tokens: int = 0
    reasoning_effort: str = "low"
    # Chat-template thinking switch (fixed-2x-no-think, 27 September); False renders an empty think block.
    enable_thinking: bool = True
    # How the assistant's earlier reasoning and content are re-sent (29 September, prefix-cache-v2.md): "normalized"
    # (the public loop: blank lines removed, ends stripped) or "verbatim" (the server's string, <think> tags removed
    # only), so the next prompt re-renders the reply as the tokens the model generated and the prefix cache reuses it.
    reasoning_history: str = "normalized"
    temperature: float = 1.0
    python_seconds: float = 5
    # Accepted only for historical run configurations; action forcing was removed.
    max_analysis_turns: int | None = None
    max_tool_output_chars: int = 7000
    max_playbook_chars: int = 5000
    escalation_actions: int = 60
    image_mode: str = "level"
    max_model_len: int = 32768
    max_num_seqs: int = 32
    max_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.90
    seed: int = 0
    compaction_summary_tokens: int = 4096
    compaction_recent_bytes: int = 48000
    action_target: int = 20
    # Duck bridge (v5): stuck notices, transition receipts and image-aware trimming.
    stuck_no_change_actions: int = 12
    stuck_turns: int = 8
    stuck_notice_turns: int = 4
    transition_component_limit: int = 200
    image_tokens_estimate: int = 400

    def __post_init__(self):
        for name in (
            "gameplay_seconds",
            "setup_seconds",
            "parallel_games",
            "max_actions",
            "max_plan",
            "python_seconds",
            "compaction_summary_tokens",
            "compaction_recent_bytes",
            "action_target",
            "stuck_no_change_actions",
            "stuck_turns",
            "stuck_notice_turns",
            "transition_component_limit",
            "image_tokens_estimate",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_context_tokens", "max_output_tokens", "thinking_budget"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative; zero uses the native allowance")
        if self.max_context_tokens + self.max_output_tokens > self.max_model_len:
            raise ValueError("Context and completion budgets exceed the server context limit")
        if self.max_model_len <= 2 * self.context_reserve:
            raise ValueError("Context window is too small for continuation summaries")
        if self.reasoning_history not in REASONING_HISTORY_MODES:
            raise ValueError(f"reasoning_history must be one of {REASONING_HISTORY_MODES}")
        if self.image_mode not in {"none", "level", "always"}:
            raise ValueError("image_mode must be none, level, or always")
        if not 0 <= self.speculative_tokens <= 5:
            raise ValueError("speculative_tokens must be between 0 and 5")
        if self.reasoning_effort not in allowed_efforts(self.model):
            choices = ("native for UI-Mate" if allowed_efforts(self.model) == ("native",) else
                       "high for Nex" if allowed_efforts(self.model) == ("high",) else "xhigh, medium or low")
            raise ValueError(f"reasoning_effort must be {choices}")

    @property
    def context_reserve(self):
        # Space for tool results and a continuation summary at the native boundary.
        return max(8192, self.compaction_summary_tokens + 4096)

    @property
    def context_limit(self):
        return self.max_context_tokens or self.max_model_len - self.context_reserve

    @classmethod
    def load(cls, path: str | Path):
        data = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self):
        return asdict(self)
