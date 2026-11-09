"""Analyzer environment for the public Duck agent when it runs inside our fleet process."""

import os
import sys

# Analyzer-side overrides from the public serving script, except timed yields are
# disabled: admission already releases the GPU between model requests. The public
# notebook forwards them into its agent process; our fleet uses a separate interpreter,
# so importing upstream defaults would be incorrect.
DUCK_ENV = {
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
    "LOCAL_ANALYZER_MAX_OUTPUT": "0",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
    "LOCAL_ANALYZER_YIELD_SECONDS": "0",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "4",
}

PROFILE = {
    "name": "duck-transfer",
    "upstream": {
        "notebook_version_url": "https://www.kaggle.com/code/wuliao0/duck-qwen3-8-anim-base?scriptVersionId=347132234",
        "package": "src/inference (public ARC3-Inference, unmodified)",
    },
    "analyzer_environment": DUCK_ENV,
    "differences": [
        "Common full-fleet deadline, eight shared inference slots and independent scorecards",
        "Temperature from the experiment config (0.7), versus upstream 0.6",
        "Existing evaluation action logging; strict coordinate validation and 48-action batch maximum",
        "Original Duck prompt, Python sandbox, frame views, history and world-model updates",
        "Original thinking-enabled template, uncapped output, 30-second Python and 1024-token tool output",
        "Persistent Python interpreter per game (v4); reasoning_effort from the experiment config (v4)",
        "No timer-driven analyzer boundaries; shared normalized nine-hour planning budget supplied per request",
        "Context trimming never drops the latest observation (duck_transfer.trimming)",
        "Fallback prior-level world/action hypotheses; revalidate; no extra model call",
    ],
}


def configure(cfg, sampling=None):
    """Point the analyzer environment at the experiment config; return the profile record.

    Call before the public package is imported. The context window follows
    ``cfg.max_model_len`` (64K in the v4 test; the public control keeps 32K) and the
    profile's recorded differences name the v4 changes. ``sampling`` (llama.cpp model screens,
    27 September) replaces the Qwen top_p/top_k with a model card's; top_k 0 leaves it out of
    the request.
    """
    DUCK_ENV["LOCAL_ANALYZER_CONTEXT_WINDOW"] = str(int(cfg.max_model_len))
    if sampling:
        DUCK_ENV["LOCAL_ANALYZER_TOP_P"] = str(float(sampling["top_p"]))
        DUCK_ENV["LOCAL_ANALYZER_TOP_K"] = str(int(sampling["top_k"]))
    profile = {**PROFILE, "analyzer_environment": dict(DUCK_ENV)}
    profile["differences"] = [
        *PROFILE["differences"],
        f"{cfg.max_model_len // 1024}K analyzer context window (public loop: 32K)",
        "Persistent Python interpreter per game; reset only after a tool timeout",
        f"Chat-template reasoning_effort={cfg.reasoning_effort} (public loop sends none)",
        # v5 (18 September)
        "Runtime state kept in memory (file written at start and end); delta history to the "
        "persistent interpreter; segmentation memoised by grid; one prompt-log write per turn",
        f"Image-aware token estimate ({cfg.image_tokens_estimate} tokens per board image) "
        "instead of base64 length; v5b: every request is counted by the server tokenizer "
        "first (an over-budget prompt is trimmed before it is sent) and the counts calibrate "
        "the characters-per-token ratio and the image-token estimate",
        "Per-turn constant lines live in the system prompt; earlier user turns are one-line "
        "stubs; one board image per request, labelled with its step",
        "Action receipts carry a transition summary and the actions already tried without "
        "effect on the same board; a no-change repeat is refused (twice per turn) before it "
        f"executes flagged; stuck notice after {cfg.stuck_no_change_actions} no-change actions "
        f"or {cfg.stuck_turns} turns on a level, every {cfg.stuck_notice_turns} turns",
        "GAME_OVER keeps the world, goal and action models and the transfer note; the next "
        "prompt explains the automatic level reset; RESET is refused as a level's first action",
        f"Output cap max_tokens={cfg.max_output_tokens or 'none'} with the trimmer's reply "
        "reserve set to match; length truncations counted (public loop: uncapped)",
    ]
    return profile


def load():
    """Apply the analyzer environment and return the public ``tool_agent`` module.

    The module reads its settings into module-level constants when it is imported, so it must be
    imported after ``os.environ`` carries ``DUCK_ENV``. Every arena run from 25 to 26 September
    imported it earlier (``arena/snapshot.py`` imported ``inference.agent`` at module level) and
    played with the public default of 12 model requests per analyzer turn instead of none, and
    the old guard here checked only the yield setting, which both defaults leave at 0. The
    environment is now re-applied to an already imported module as well, so the analyzer settings
    no longer depend on import order; ``late_import`` records that it happened.
    """
    os.environ.update(DUCK_ENV)
    early = "inference.agent.tool_agent" in sys.modules
    from inference.agent import tool_agent

    if early:
        apply_environment(tool_agent)
    tool_agent.DUCK_TRANSFER_LATE_IMPORT = early
    return tool_agent


def apply_environment(module):
    """Recompute the module constants that ``DUCK_ENV`` controls from the current environment."""
    parsers = {bool: module._get_env_bool, int: module._get_env_int, float: module._get_env_float}
    for key in DUCK_ENV:
        name = f"_{key}"
        if hasattr(module, name):
            current = getattr(module, name)
            setattr(module, name, parsers[type(current)](key, current))
    if module._LOCAL_ANALYZER_TOOL_STEPS != int(DUCK_ENV["LOCAL_ANALYZER_TOOL_STEPS"]):
        raise RuntimeError("duck_transfer.loader could not apply the analyzer environment to inference.agent.tool_agent")
