"""Reviewed checkpoint identities, independent of the Flash serving recipe."""
from dataclasses import dataclass, replace
from pathlib import Path

from arc_infra.model_templates import (
    HOLO4_27B_MODEL,
    HOLO4_35B_MODEL,
    MINICPM_MODEL,
    NEMOTRON35_BOLT_MODEL,
    NEMOTRON35_SUPER_VL_MODEL,
    NEMOTRON_LIGHTNING_MODEL,
    NEMOTRON_SUPER_MODEL,
    NEX_MODEL,
    NEX_REVISION,
    ORNITH_35B_MODEL,
    QWEN35_122B_MODEL,
    QWEN36_MODEL,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    model: str
    revision: str
    architecture: str
    quant_method: str | None
    cli_quantization: str | None
    effort: str
    mtp_tokens: int
    mtp_tensors: int
    native_context: int = 262144
    tool_parser: str = 'qwen3_coder'
    reasoning_parser: str = 'qwen3'
    extra_args: tuple = ()  # appended to the pinned serve command (e.g. an attention backend)
    declared_context: int | None = None  # config.json max_position_embeddings when it exceeds the served (tokenizer-bounded) context
    # stock vLLM (arena.stock_serving; Nemotron 3.5): runtime 'stock' serves on a pinned stock vLLM
    # folder (dataset + tar sha256) instead of the Flash fork; stock_env pins its variables; reasoning_plugin is a
    # parser file shipped with the checkpoint
    runtime: str = 'fork'
    stock_dataset: str | None = None
    stock_sha256: str | None = None
    stock_env: tuple = ()
    reasoning_plugin: str | None = None
    template_extra: tuple = ()  # chat-template switches the study sends with every request (default: preserve_thinking)

    @property
    def manifest(self):
        return Path(__file__).with_name('models') / (self.name + '.json')


NEX = Candidate('nex-n25-mini-nvfp4', NEX_MODEL, NEX_REVISION,
                'Qwen3_5MoeForConditionalGeneration', 'compressed-tensors',
                'compressed-tensors', 'high', 0, 0)
QWEN_27_NVFP4 = Candidate('qwen38-27b-nvfp4', 'nvidia/Qwen3.8-27B-NVFP4',
    '482ca0f3832238542f8f5295dde86b5f22711d80', 'Qwen3_5ForConditionalGeneration',
    'modelopt', 'modelopt_mixed', 'xhigh', 3, 15)
QWEN_27_FP8 = Candidate('qwen38-27b-fp8', 'Qwen/Qwen3.8-27B-FP8',
    '017b9c7af6b5689d5dd426a76e0bc077eb5ca20a', 'Qwen3_5ForConditionalGeneration',
    'fp8', 'fp8', 'xhigh', 3, 22)
UI_MATE_9B = Candidate('ui-mate-9b', 'tencent/UI-Mate-9B',
    '05dd5f2975195a5bb03d4363e8767f12158c8421', 'Qwen3_5ForConditionalGeneration',
    None, None, 'native', 0, 0)
UI_MATE_27B = Candidate('ui-mate-27b', 'tencent/UI-Mate-27B',
    '3ade2378fc84032d5017c1a9c93c4eaa77d65e57', 'Qwen3_5ForConditionalGeneration',
    None, None, 'native', 0, 0)
MINICPM_2B = Candidate('minicpm5-2b', MINICPM_MODEL,
    '12a3808a956f869c767195e9266b59c4d21d92e2', 'LlamaForCausalLM',
    None, None, 'native', 0, 0, native_context=131072, tool_parser='minicpm5')
QWEN_36_NVFP4 = Candidate('qwen36-35b-a3b-nvfp4', QWEN36_MODEL,
    '1355db6a052410cfd62085d94b58866fd0f2c3c5', 'Qwen3_5MoeForConditionalGeneration',
    'modelopt', 'modelopt_mixed', 'native', 3, 19)
# Nemotron 3 Super: hybrid Mamba-2 / attention (8 of 88 layers) / latent MoE, 12B active of 120B,
# NVFP4+FP8 mixed checkpoint (~80 GB). MTP head present (1,040 tensors) but unused: speculative
# decoding is reported to exhaust the 96 GB card. Text-only. The pinned vLLM ships nemotron_v3
# reasoning and the Qwen3-Coder XML tool format the template uses; TRITON_ATTN is the attention
# backend documented working for this checkpoint on RTX PRO 6000 (SM120).
# v1 (25 September) hung in the GPU worker after 13 minutes at 25 running requests with the
# auto-selected fp8 KV cache (scale 1.0, no checkpoint scales) and the FLASHINFER_CUTLASS NVFP4
# MoE; v2 pins BF16 KV and the Marlin NVFP4 MoE backend that NVIDIA's SM12x (DGX Spark) recipe uses.
NEMOTRON_SUPER = Candidate('nemotron3-super-nvfp4', NEMOTRON_SUPER_MODEL,
    'ff433f5493e25d631c9f12b5d55c674229923d02', 'NemotronHForCausalLM',
    'modelopt', 'modelopt_mixed', 'native', 0, 1040, reasoning_parser='nemotron_v3',
    extra_args=('--attention-backend', 'TRITON_ATTN', '--kv-cache-dtype', 'bfloat16', '--moe-backend', 'marlin'))
# Nemotron 3.5 Lightning: the throughput-side hybrid (6 attention / 23 Mamba-2 / 23 MoE layers,
# 3B active of 30B, ~21.6 GB NVFP4/FP8), text-only, MTP head (270 tensors) used at depth 3.
# config.json declares 1M positions; the tokenizer bounds the served context at 262,144.
# v1 (25 September) failed on the first request: the MTP drafter took the FlashInfer decode path
# with the auto-selected fp8 KV cache and JIT-compiled an XQA kernel whose link step cannot find
# libcuda in the pinned runtime. v2 runs without speculation and with BF16 KV (MTP tensors stay
# in the inventory; they are simply not loaded).
NEMOTRON_LIGHTNING = Candidate('nemotron35-lightning-nvfp4', NEMOTRON_LIGHTNING_MODEL,
    'bee7596271d1495f6992ae224aefde4410e816b8', 'NemotronHForCausalLM',
    'modelopt', 'modelopt_mixed', 'native', 0, 270, reasoning_parser='nemotron_v3',
    extra_args=('--attention-backend', 'TRITON_ATTN', '--kv-cache-dtype', 'bfloat16', '--moe-backend', 'marlin'),
    declared_context=1048576)
# Qwen3.5-122B-A10B: the Nex/Qwen3.6 architecture path (48 layers: 12 full attention with 2 KV heads of
# 256, 36 Gated DeltaNet; 256 experts, 8 active; vision tower kept, so images stay in the requests).
# NVIDIA ModelOpt NVFP4 experts (quant_algo NVFP4, so the pinned modelopt_fp4 command applies unchanged)
# with BF16 attention, shared-expert, embedding and vision weights: 77.7 GiB in 9 shards. The checkpoint
# declares an FP8 KV scheme without k/v scales; under 'auto' the pinned vLLM would resolve it to fp8_e4m3
# and, as in Lightning v1, route FlashInfer decode into the XQA JIT whose link step fails on this runtime,
# so BF16 KV is pinned (24 KiB per token; fp8 halves that once the link is fixed). The unquantized MoE
# MTP head (785 tensors, 4.7 GiB BF16) stays in the inventory but is not loaded: with ~14 GiB left after
# weights on the 96 GB card, speculation is off so the KV pool keeps the room.
QWEN35_122B = Candidate('qwen35-122b-a10b-nvfp4', QWEN35_122B_MODEL,
    '98915d837c4e7c87ac8296d02e89de19b3207e6d', 'Qwen3_5MoeForConditionalGeneration',
    'modelopt', 'modelopt_fp4', 'native', 0, 785,
    extra_args=('--kv-cache-dtype', 'bfloat16'))
# Ornith-1.5-35B-A3B (ornith-ai, MIT): a GRPO-trained fine-tune with the Qwen3.6-35B-A3B shape (40 layers:
# 10 full attention with 2 KV heads of 256, 30 Gated DeltaNet; 256 experts, 8 active; vision tower kept, so
# images stay in the requests). The ModelOpt 0.45 MIXED_PRECISION layout matches the Qwen3.6 candidate layer
# for layer (130 attention/linear-attention projections FP8 with FP8 input activations, 161 expert, shared-expert
# and lm_head entries W4A16_NVFP4), except that the NVFP4 weights are natively weight-only (no input_scale on
# disk), which the pinned modelopt_mixed W4A16 Marlin path loads directly. Unlike Qwen3.6 the config declares an
# FP8 KV scheme without k/v scales; under 'auto' the pinned vLLM would resolve it to fp8_e4m3 and, as in
# Lightning v1, route FlashInfer decode into the XQA JIT whose link fails on this runtime, so BF16 KV is pinned
# (the KV dtype Qwen3.6 ran with). The unquantized MTP head stores its 256 BF16 experts per expert (785 tensors;
# NVIDIA's Qwen3.6 export fuses the same bytes into 19) and drafts at depth 3 as for Qwen3.6 (acceptance 3.3/4).
ORNITH_35B = Candidate('ornith15-35b-a3b-nvfp4', ORNITH_35B_MODEL,
    '94e431d9cc47fa1986a7a1a4e9a80f7f118b03aa', 'Qwen3_5MoeForConditionalGeneration',
    'modelopt', 'modelopt_mixed', 'native', 3, 785,
    extra_args=('--kv-cache-dtype', 'bfloat16'))
# Holo4-27B-NVFP4 (H Company, CC BY-NC 4.0; test it with the hero-v2 arena setup): a computer-use
# fine-tune of Qwen3.8-27B with the same dense hybrid (64 layers: 48 Gated DeltaNet, 16 full gated attention with 4 KV
# heads of 256; vision tower kept) and a byte-identical chat template (sha256 c3cf9e34...). compressed-tensors 0.18:
# every MLP NVFP4 weight-only (group 16), the attention and GDN projections FP8 per-tensor with static input scales,
# and a calibrated FP8 KV scheme (k_scale / v_scale for the 16 attention layers). The config declares one MTP layer
# but the checkpoint has none, so MTP-0. FP8 KV (user decision) halves the attention KV (32 KiB per token); FA2, the
# backend the Qwen3.8-27B candidates ran with, has no FP8 KV on this GPU and FlashInfer's FP8 decode JIT-links an XQA
# kernel that failed on the pinned runtime (Lightning v1), so Triton attention is pinned. A trial's server
# environment may replace the KV dtype (ARENA_KV_CACHE_DTYPE, e.g. the BF16 drift reference).
HOLO4_27B = Candidate('holo4-27b-nvfp4', HOLO4_27B_MODEL,
    '6a02909667ba91dcd45e01be13c4d49992112510', 'Qwen3_5ForConditionalGeneration',
    'compressed-tensors', 'compressed-tensors', 'xhigh', 0, 0,
    extra_args=('--attention-backend', 'TRITON_ATTN', '--kv-cache-dtype', 'fp8'))
# Holo4-35B-A3B-NVFP4 (H Company, Apache 2.0; model screen in the arc2 v37 harness): the same
# compressed-tensors recipe as Holo4-27B-NVFP4 (FP8 attention / GDN projections with static input scales, NVFP4
# weight-only MLP and shared experts, calibrated FP8 KV) on the Qwen3.6-35B-A3B shape (40 layers, 10 full attention
# with 2 KV heads of 256; 256 experts, 8 active; vision kept). 22.5 GB. No MTP tensors (config declares one layer).
# Qwen3.6's template: thinking on/off only; earlier reasoning is rendered only with preserve_thinking.
HOLO4_35B = Candidate('holo4-35b-a3b-nvfp4', HOLO4_35B_MODEL,
    '5631eb8e4743adb7c057373e5d4194ac57f2b614', 'Qwen3_5MoeForConditionalGeneration',
    'compressed-tensors', 'compressed-tensors', 'native', 0, 0,
    extra_args=('--attention-backend', 'TRITON_ATTN', '--kv-cache-dtype', 'fp8'))
# Nemotron 3.5 Super VL and Bolt (NVIDIA, OpenMDW-1.1; screen them like Holo4 in the arc2 v37 harness,
# highest effort): nemotron_h_omni (Mamba-2 / attention with 2 KV heads of 128 / latent MoE, C-RADIO vision), which only
# stock vLLM nightlies serve (Super VL's card: nightly 3f1bcfd8 or v0.32.0; Bolt's: nightly 77154ca6 or newer, so one
# runtime at 3f1bcfd8 serves both). Thinking on (enable_thinking, no medium_effort) is the highest effort; the template's
# truncate_history_thinking is turned off by the study (earlier reasoning kept, as with Flash). Serve flags from the
# cards, without speculation (as the Holo4 screen) and with video/audio off. Super VL: ModelOpt mixed NVFP4 experts /
# FP8 mixers, 75 GB, MTP head (4,120 tensors) not loaded, 1M declared positions served at 262,144. Bolt 24GB: IQ2-XXS
# experts (ggml packing) with NVFP4 Mamba and Q8_0 embeddings, b12x linear/MoE kernels, the V2 model runner, FP8 KV.
# Images: vLLM's default 999 a prompt (a conversation carries the task's pictures and every shown grid).
STOCK_VLLM = ('richardcsaky/arc-stock-vllm-3f1bcfd8', '16c76540938d2769bac521352edd2ba109b3944701dcc935bdd9815265231041')
NEMOTRON35_SUPER_VL = Candidate('nemotron35-super-vl-nvfp4', NEMOTRON35_SUPER_VL_MODEL,
    '59b8721646c38111a242da18218859fb2b454322', 'NemotronH_Omni_Reasoning_V3',
    'modelopt_mixed', None, 'native', 0, 4120, reasoning_parser='nemotron_v3', declared_context=1048576,
    extra_args=('--async-scheduling', '--prefix-match-unit', '16', '--enable-mamba-shared-prefix-checkpoint',
                '--mamba-backend', 'flashinfer', '--mamba-ssm-cache-dtype', 'float16',
                '--enable-mamba-cache-stochastic-rounding', '--mamba-cache-philox-rounds', '5',
                '--mamba-cache-mode', 'align', '--limit-mm-per-prompt', '{"image":999,"video":0,"audio":0}'),
    runtime='stock', stock_dataset=STOCK_VLLM[0], stock_sha256=STOCK_VLLM[1],
    template_extra=(('truncate_history_thinking', False),))
NEMOTRON35_BOLT = Candidate('nemotron35-bolt-24gb', NEMOTRON35_BOLT_MODEL,
    'c6cc2e8ffa353403213d6b9b18aaf8eb684d705a', 'NemotronH_Omni_Reasoning_V3',
    'modelopt', None, 'native', 0, 0, reasoning_parser='ultra_v3',
    extra_args=('--load-format', 'auto', '--enable-chunked-prefill', '--kv-cache-dtype', 'fp8',
                '--linear-backend', 'b12x', '--moe-backend', 'b12x', '--attention-backend', 'FLASHINFER',
                '--mamba-backend', 'flashinfer', '--mamba-ssm-cache-dtype', 'bfloat16',
                '--mamba-cache-dtype', 'bfloat16', '--mamba-cache-mode', 'align',
                '--limit-mm-per-prompt', '{"image":999,"video":0,"audio":0}'),
    runtime='stock', stock_dataset=STOCK_VLLM[0], stock_sha256=STOCK_VLLM[1],
    stock_env=(('VLLM_USE_V2_MODEL_RUNNER', '1'),), reasoning_plugin='ultra_v3_reasoning_parser.py',
    template_extra=(('truncate_history_thinking', False),))
# Bolt at FP8 (the quality upper bound when it fits with usable KV): the BF16 release quantized at load
# by stock vLLM's online FP8 (per output channel weights, dynamic per-token activations, llm-compressor's FP8_DYNAMIC
# shape; weights materialized layer by layer, so the load peak stays near the FP8 size, ~76 GB). Serve flags of the
# BF16 card plus Super VL's hybrid-cache flags (same nemotron_h_omni family): fine prefix matching, FP16 SSM state with
# stochastic rounding; BF16 KV (2 KV heads x 128 on 8 attention layers: 8 KiB a token).
NEMOTRON35_BOLT_FP8 = Candidate('nemotron35-bolt-fp8', 'nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16',
    '567021037ff88364143e24ea05e6abd0f72086ff', 'NemotronH_Omni_Reasoning_V3',
    None, None, 'native', 0, 0, reasoning_parser='ultra_v3',
    extra_args=('--quantization', 'fp8_per_channel', '--load-format', 'auto', '--enable-chunked-prefill',
                '--async-scheduling', '--prefix-match-unit', '16', '--enable-mamba-shared-prefix-checkpoint',
                '--attention-backend', 'FLASHINFER', '--mamba-backend', 'flashinfer',
                '--mamba-ssm-cache-dtype', 'float16', '--enable-mamba-cache-stochastic-rounding',
                '--mamba-cache-philox-rounds', '5', '--mamba-cache-mode', 'align',
                '--limit-mm-per-prompt', '{"image":999,"video":0,"audio":0}'),
    runtime='stock', stock_dataset=STOCK_VLLM[0], stock_sha256=STOCK_VLLM[1],
    reasoning_plugin='ultra_v3_reasoning_parser.py', template_extra=(('truncate_history_thinking', False),))
PROFILES = {p.name: p for p in (NEX, QWEN_27_NVFP4, QWEN_27_FP8, UI_MATE_9B, UI_MATE_27B,
                               MINICPM_2B, QWEN_36_NVFP4, NEMOTRON_SUPER, NEMOTRON_LIGHTNING, QWEN35_122B,
                               ORNITH_35B, HOLO4_27B, HOLO4_35B, NEMOTRON35_SUPER_VL, NEMOTRON35_BOLT,
                               NEMOTRON35_BOLT_FP8)}


def from_study(study):
    profile = PROFILES[study['candidate_profile']]
    if (study.get('served_model'), study.get('model_revision')) != (profile.model, profile.revision):
        raise ValueError('Candidate study does not match the pinned checkpoint identity')
    if profile == NEX and study['reasoning_effort'] == 'none':
        profile = replace(profile, effort='none')
    if study['server']['mtp_tokens'] != profile.mtp_tokens or study['reasoning_effort'] != profile.effort:
        raise ValueError('Candidate study changes the reviewed MTP/reasoning profile')
    return profile
