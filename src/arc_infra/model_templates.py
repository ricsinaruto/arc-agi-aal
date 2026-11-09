"""Explicit checkpoint template controls; historical Flash requests stay unchanged."""

NEX_MODEL = "primitive-ai/Nex-N2.5-mini-NVFP4"
NEX_REVISION = "ceaa4ce6d942c3d5456322d183b535a0c89a5fa6"
UI_MATE_MODELS = ("tencent/UI-Mate-9B", "tencent/UI-Mate-27B")
MINICPM_MODEL = "openbmb/MiniCPM5-2B"
QWEN36_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
QWEN35_122B_MODEL = "nvidia/Qwen3.5-122B-A10B-NVFP4"
ORNITH_35B_MODEL = "ornith-ai/Ornith-1.5-35B-A3B-NVFP4"
# Qwen3.8-27B fine-tune with the Qwen3.8 template byte for byte: the default (Flash / Qwen3.8) request controls.
HOLO4_27B_MODEL = "Hcompany/Holo4-27B-NVFP4"
# Qwen3.6-35B-A3B fine-tune: Qwen3.6's template (thinking on/off, preserve_thinking off unless sent).
HOLO4_35B_MODEL = "Hcompany/Holo4-35B-A3B-NVFP4"
NEMOTRON_SUPER_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
NEMOTRON_LIGHTNING_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
NEMOTRON_MODELS = (NEMOTRON_SUPER_MODEL, NEMOTRON_LIGHTNING_MODEL)
# Nemotron 3.5 vision-language candidates (stock vLLM; screen): images allowed, thinking on/off.
NEMOTRON35_SUPER_VL_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
NEMOTRON35_BOLT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
NEMOTRON35_BOLT_BF16_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16"  # served at online FP8
NEMOTRON35_VL_MODELS = (NEMOTRON35_SUPER_VL_MODEL, NEMOTRON35_BOLT_MODEL, NEMOTRON35_BOLT_BF16_MODEL)
# llama.cpp model screens (27 September, arena.llamacpp_profiles). GLM-5.3-Flash's template always
# opens a think block; its dial is reasoning_effort low / high / max (any other value renders max).
# dots3-note-preview's template takes enable_thinking. Both run text-only: the glm5next PR has no
# vision, and dots3's 15.5 GB projector does not fit beside the IQ2 weights.
GLM53_FLASH_MODEL = "zai-org/GLM-5.3-Flash"
DOTS3_MODEL = "dots-studio/dots3-note-prev"
GLM_EFFORTS = {"xhigh": "max", "max": "max", "high": "high", "medium": "high", "low": "low"}


def text_only(model):
    # Nemotron 3 Super is a text model: Duck and Pi receive exact letter-coded boards.
    return model in (MINICPM_MODEL, *NEMOTRON_MODELS, GLM53_FLASH_MODEL, DOTS3_MODEL)


def request_messages(model, messages):
    """MiniCPM's native template accepts strings, not multimodal content lists.

    Fail closed on images: the harness must supply exact evidence before this layer.
    Never silently discard an unsupported observation or change stored history.
    """
    if not text_only(model):
        return messages
    import copy

    result = copy.deepcopy(messages)
    for message in result:
        content = message.get('content')
        if isinstance(content, list):
            if any(not isinstance(p, dict) or p.get('type') != 'text' for p in content):
                raise ValueError('Text-only model received non-text evidence')
            message['content'] = '\n'.join(p['text'] for p in content)
    return result


def allowed_efforts(model):
    if model == GLM53_FLASH_MODEL:
        return ("max", "high", "low")
    if model in (*UI_MATE_MODELS, MINICPM_MODEL, QWEN36_MODEL, QWEN35_122B_MODEL, ORNITH_35B_MODEL,
                 HOLO4_35B_MODEL, *NEMOTRON_MODELS, *NEMOTRON35_VL_MODELS, DOTS3_MODEL):
        return ("native",)
    return ("high", "none") if model == NEX_MODEL else ("xhigh", "medium", "low")


def template_kwargs(model, kwargs):
    if model == GLM53_FLASH_MODEL:
        # No off switch: a thinking-off request (compaction summaries) gets the shortest effort.
        if kwargs.get("enable_thinking") is False:
            return {"reasoning_effort": "low"}
        return {"reasoning_effort": GLM_EFFORTS.get(kwargs.get("reasoning_effort"), "max")}
    if model in (MINICPM_MODEL, QWEN36_MODEL, QWEN35_122B_MODEL, ORNITH_35B_MODEL, HOLO4_35B_MODEL,
                 *NEMOTRON_MODELS, *NEMOTRON35_VL_MODELS, DOTS3_MODEL):
        # Nemotron's template also accepts low_effort and truncate_history_thinking (default True:
        # older reasoning is dropped from the prompt); both stay at their defaults. Ornith's has only
        # this switch and renders every earlier reasoning block (no preserve_thinking gate).
        return {'enable_thinking': kwargs.get('enable_thinking', True)}
    if model in UI_MATE_MODELS:
        # These templates have an on/off switch, not an effort dial. Keep the
        # Qwen response parser in the same mode and omit unsupported effort.
        result = {"enable_thinking": kwargs.get("enable_thinking", True)}
        if model.endswith("27B") and "preserve_thinking" in kwargs:
            result["preserve_thinking"] = kwargs["preserve_thinking"]
        return result
    if model != NEX_MODEL:
        return dict(kwargs)
    # Nex's template ignores enable_thinking, but vLLM's Qwen3 response parser
    # consumes it independently. Keep parser and template in the same mode.
    # preserve_thinking remains unsupported; native history is always serialized.
    if kwargs.get("enable_thinking") is False:
        effort = "none"
    else:
        effort = kwargs.get("reasoning_effort")
    if effort not in ("high", "none"):
        raise ValueError("Nex requests require native reasoning_effort high or none")
    return {"reasoning_effort": effort, "enable_thinking": effort != "none"}
