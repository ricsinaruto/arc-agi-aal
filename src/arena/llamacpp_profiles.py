"""llama.cpp model screens (27 September): pinned GGUF weights, llama.cpp source and server settings.

Pure data (no imports) so the Modal launcher can read it before any image exists. ``served_model``
is the name the agent sends and ``arc_infra.model_templates`` keys its template controls on.
"""

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"

PROFILES = {
    "flash-next-gsq-iq3s": {
        "served_model": "Qwen/Qwen3.8-Flash-Next-NVFP4",  # API alias; actual GGUF identity below
        "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "revision": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",
        "files": {
            "IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf": 54817524224,
            "IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf": 28800138432,
            "mmproj-Qwen3.8-Flash-Next-BF16.gguf": 907543008,
        },
        "model_file": "IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
        "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
        "llama_cpp": {"ref": "72db1e02ff0d804f7553be6f01b93c2fd1fa1a6d",
                      "commit": "72db1e02ff0d804f7553be6f01b93c2fd1fa1a6d",
                      "note": "30 September source contains qwen4exp, PLE lazy mmap and vision support"},
        "server_args": ["-ngl", "99", "-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0", "--jinja",
                        "-lm", "mmap", "--lazy-mode", "on", "--reasoning-format", "deepseek", "-ub", "256"],
        "slots": 37, "window": 93184,
        "sampling": {"temperature": .7, "top_p": .95, "top_k": 20, "min_p": 0.0},
        "reasoning_effort": "high", "quick_template_kwargs": {"enable_thinking": False},
        "text_only": False,
    },
    # aj9o9 hand-mixed IQ2 build (routed experts IQ2_XXS/IQ2_S, everything else Q6_K or better, MTP
    # block pruned at quantization with --prune-layers 45). Needs the unmerged glm5next PR #27752 at or
    # after "llama : skip foreign-sequence cells when building the glm5next k-pools"; the PR is
    # text-only, so GLM plays on letter-coded boards. README flags: -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --jinja.
    "glm53-flash-aj-iq2xxs": {
        "served_model": "zai-org/GLM-5.3-Flash",
        "repo": "aj9o9/GLM-5.3-Flash-GGUF",
        "files": {
            "AJ-IQ2_XXS/GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf": 44665098240,
            "AJ-IQ2_XXS/GLM-5.3-Flash-AJ-IQ2_XXS-00002-of-00002.gguf": 42680908320,
        },
        "model_file": "AJ-IQ2_XXS/GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf",
        "llama_cpp": {
            "ref": "pull/27752/head",
            "fork": "https://github.com/eauchs/llama.cpp",
            "commit": "1d0c76f3c6d030fdfc269aa27db6334ea2834cec",
            "note": "PR #27752 head (27 September); contains 0ccfa15164 'skip foreign-sequence cells when "
                    "building the glm5next k-pools'",
        },
        "server_args": ["-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--jinja",
                        "--reasoning-format", "deepseek"],
        "slots": 4,
        "window": 49152,
        # Model card (zai-org/GLM-5.3-Flash generation_config and evaluation notes): temperature 1.0,
        # top_p 0.95, no top-k.
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0},
        # The template's effort dial is low / high / max (default max, the card's benchmark setting).
        "reasoning_effort": "max",
        # Short replies for the serve check's recall prompts.
        "quick_template_kwargs": {"reasoning_effort": "low"},
        "text_only": True,
    },
    # bartowski imatrix IQ2_XS; text-only (the 15.5 GB mmproj does not fit beside the weights).
    "dots3-note-prev-iq2xs": {
        "served_model": "dots-studio/dots3-note-prev",
        "repo": "bartowski/dots-studio_dots3-note-prev-GGUF",
        "files": {
            "dots-studio_dots3-note-prev-IQ2_XS/dots-studio_dots3-note-prev-IQ2_XS-00001-of-00003.gguf": 39630982240,
            "dots-studio_dots3-note-prev-IQ2_XS/dots-studio_dots3-note-prev-IQ2_XS-00002-of-00003.gguf": 39603590432,
            "dots-studio_dots3-note-prev-IQ2_XS/dots-studio_dots3-note-prev-IQ2_XS-00003-of-00003.gguf": 5022085184,
        },
        "model_file": "dots-studio_dots3-note-prev-IQ2_XS/dots-studio_dots3-note-prev-IQ2_XS-00001-of-00003.gguf",
        "llama_cpp": {
            "ref": "refs/tags/b11207",
            "fork": None,
            "commit": "7ac59a6e3ad851cd41af00f678effab0598ba9a8",
            "note": "release b11207 (27 September); the quants need b10569 or newer",
        },
        # No forced flash attention: the sliding-window MLA layers attend over a 1,024 + 64 latent, a head
        # size the CUDA flash-attention kernels lack, so `-fa on` ran attention on the CPU (serve check v1:
        # prefill 21 tok/s, decode 15). `-fa auto` falls back to the GPU matmul path; without flash
        # attention the V cache cannot be quantized, so the (small, latent) cache stays f16. Without flash
        # attention the scores are materialised (128 heads x context x micro-batch, f32): at the default
        # micro-batch of 512 that is 12.9 GB per 48K slot and v2 ran out of memory on its first request,
        # so the micro-batch is 128 (3.2 GB).
        "server_args": ["-ngl", "99", "-fa", "auto", "-ub", "128", "--jinja", "--reasoning-format", "deepseek"],
        "slots": 4,
        "window": 49152,
        # Model card: temperature 1.0, top_p 0.95; thinking through chat_template_kwargs.enable_thinking.
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0},
        "reasoning_effort": "native",
        "quick_template_kwargs": {"enable_thinking": False},
        # No 8-slot or unified pass: materialised scores over a unified cache (up to 8 x 48K cells) do
        # not fit beside the weights.
        "text_only": True,
    },
}


def profile(name):
    if name not in PROFILES:
        raise ValueError(f"Unknown llama.cpp profile {name!r}; choose from {sorted(PROFILES)}")
    return PROFILES[name]
