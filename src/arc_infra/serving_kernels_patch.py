"""Hash-guarded optional serving overlay (kernel work, phase 4); pinned sources stay untouched otherwise.

A study's ``server.serving_kernels`` (absent: the pinned runtime is served unchanged) selects:

- ``mamba_ssm_cache_dtype: "bfloat16"``: ``--mamba-ssm-cache-dtype bfloat16``. The MTP draft's config hook
  otherwise applies the checkpoint's float32 GDN recurrent state; MTP-0 already runs bf16 (no hook), and
  the fused GDN kernels accept both. Halves the state pages (MTP-3 blocks 1,600 -> 832 tokens).
- ``moe_backend: "flashinfer_b12x"``: ``--moe-backend flashinfer_b12x`` (FlashInfer's fused SM120 NvFp4
  MoE) for the target's NvFp4 experts; the b12x workspace is shared by all layers (patch below).
- ``draft_experts: "nvfp4"``: the MTP draft layer's BF16 routed experts (4.69 GiB) quantized to NvFp4 at
  load and served by b12x (``mtp.py`` patch). Draft-only: acceptance, never output correctness.
- ``hc_fusion``: silu + HC up GEMV + gate-mix in one kernel at decode sizes (``hyperconnection.py``).
- ``topk_sampler``: top-k-first sampling mask (``topk_topp_sampler.py`` + ``states.py``).
- ``kv_cache_8bit: "int8" | "fp8"`` (28 September): the QSA main KV cache in 8 bits with a per-token-head fp32
  scale inline in the page (``--kv-cache-dtype int8_per_token_head`` / ``fp8_per_token_head``, the pinned
  runtime's own layout and writer), read by an 8-bit copy of the QSA sparse-attention kernel
  (``kv8_runtime.py``, copied to ``vllm/arc_kv8.py`` only with this option). ``nvidia/qsa.py`` is patched to
  lift its BF16-only guards for exactly the configured dtype. The indexer caches stay BF16. Halves the main-KV
  bytes per token (2,048 -> 1,040 per QSA layer), so a block holds ~2x the tokens
  (docs/validation-20260927/kv-cache-8bit.md).
- ``kv_cache_4bit: "int4" | "k8v4"`` (29 September): the QSA main KV cache in 4 bits (``int4``: keys and values,
  ``--kv-cache-dtype int4_per_token_head``, the pinned runtime's sub-byte page with its random Hadamard rotation and
  zero point) or with 8-bit keys and 4-bit values (``k8v4``, served as ``int8_per_token_head`` with this overlay's
  page), read and written by ``kv4_runtime.py`` (``vllm/arc_kv4.py``, installed with the kv8 helper whose merge
  kernel it reuses). The same ``nvidia/qsa.py`` edits as kv8, calling ``arc_kv4``; exclusive with ``kv_cache_8bit``
  (docs/validation-20260927/kv-cache-4bit.md).
- ``kv8_prefill: "bf16"`` (kernel work, phase 7, 29 September; with ``kv_cache_8bit: "int8"``): the QSA rows of
  prefill chunks attend over their requests' int8 pages dequantized to BF16, with the pinned BF16 sparse kernel
  (~3.4 against ~8.9 ms per QSA layer on a 4K chunk); decode-size calls keep ``arc_kv8`` byte for byte and the cache
  stays int8 (``kv8_prefill_runtime.py`` -> ``vllm/arc_kv8_prefill.py``; docs/validation-20260927/kernels-phase7.md).

Serving tweaks (28 September; docs/validation-20260927/serving-tweaks.md), each off unless set:

- ``draft_vocab: "arena16k"``: the MTP draft head computes logits over a fixed 16,384-token subset
  (``serving_draft_vocab.json``: every byte and special token plus the most generated tokens of the
  specialization corpus) and writes them into a full-width row that is -inf elsewhere (``mtp.py``
  ``compute_logits``). Draft-only: rejection sampling keeps the output exact, only acceptance can move.
- ``ple_uva``: the PLE n-gram table (47.7 GiB FP8) in page-locked, device-mapped host memory, read by the
  GPU worker's own PLE path through UVA inside the graph, instead of the CPU offload process and its round
  trip (``ple_layer.py``, patched after the pinned recipe's own PLE patch; ``VLLM_PLE_CPU_OFFLOAD=0``).
- ``embed_host``: the target's input embedding (248,320 x 2,560 BF16, 1.18 GiB) in page-locked host
  memory, read through UVA (``model.py``); the MTP draft shares it.

Each option patches only its own pinned files, each guarded by the pinned file's sha256 and exact anchors;
the helper (``serving_kernels_runtime.py``) is copied to ``vllm/arc_serving_kernels.py`` with the settings
in ``vllm/arc_serving_kernels.json`` (and the draft subset in ``vllm/arc_draft_vocab.json``). The two flags
are appended to the serving command by ``arena.serving``.

- ``moe_backend: "humming"`` (3-bit experts, 30 September; docs/validation-20260930/expert-3bit.md): vLLM's Humming
  MoE (JIT Marlin-derived, sm_120 tuning) on the NvFp4 experts; no pinned file changes by itself.
- ``expert_int3: 16`` (with ``moe_backend "humming"`` and an int3 expert overlay): ``humming_utils.py`` re-encodes
  the int3-grid NVFP4 routed experts as Humming uint3 at load (``expert_int3_patch`` / ``expert_int3_runtime``,
  installed as ``vllm/arc_expert_int3.py``; ``ARC_EXPERT_INT3`` in the serving environment).
- ``indexer_fp8`` (throughput arm, 3 October; docs/research-throughput/LOG.md): the QSA indexer's compressed-key cache
  in float8 e4m3 instead of BF16 (``nvidia/indexer_qsa.py``: the cache spec dtype, so its pages and the block shrink;
  the pinned Triton writers cast on store) and the paged scoring kernel reads the keys up-cast to the query's BF16
  (``nvidia/ops/qsa.py``). Only the top-2,048-block selection can change (the selected tokens are attended exactly in
  the main KV); the compressed pages are 64 -> 32 B per token and layer.
"""

import hashlib
import json
from pathlib import Path

HELPER = 'vllm/arc_serving_kernels.py'
KV8_HELPER = 'vllm/arc_kv8.py'
KV4_HELPER = 'vllm/arc_kv4.py'
KV8_PREFILL_HELPER = 'vllm/arc_kv8_prefill.py'
INT3_HELPER = 'vllm/arc_expert_int3.py'
SETTINGS = 'vllm/arc_serving_kernels.json'
DRAFT_VOCAB = 'vllm/arc_draft_vocab.json'
DRAFT_VOCAB_SOURCE = 'serving_draft_vocab.json'  # next to this module (scripts/specialization_vocab.py draft-vocab)
MARKER = 'serving-kernels-patch.json'
PLE_MARKER = 'serving-kernels-ple-patch.json'
# Phase-4 options are always written to the settings. Options added after a recorded arm (the serving tweaks and
# kv_cache_8bit, 28 September) are kept out of the canonical setting while off, so the recorded arms' settings (and
# the settings file the server reads) stay byte-identical.
PHASE4_OPTIONS = ('mamba_ssm_cache_dtype', 'moe_backend', 'draft_experts', 'hc_fusion', 'topk_sampler')
TWEAK_OPTIONS = ('draft_vocab', 'ple_uva', 'embed_host')
OPTIONS = (PHASE4_OPTIONS + ('kv_cache_8bit', 'kv_cache_4bit', 'kv8_prefill') + TWEAK_OPTIONS + ('expert_int3',)
           + ('indexer_fp8',))
OPTIONAL = ('kv_cache_8bit', 'kv_cache_4bit', 'kv8_prefill') + TWEAK_OPTIONS + ('expert_int3', 'indexer_fp8')
# moe_backend "marlin" (expert quantization, 28 September): vLLM's Marlin NvFp4 MoE, weight-only W4A16 on the same
# NvFp4 weights (BF16 activations); no pinned file changes (the draft layer keeps its own b12x backend).
CHOICES = {'mamba_ssm_cache_dtype': ('bfloat16',), 'moe_backend': ('flashinfer_b12x', 'marlin', 'humming'),
           'expert_int3': (16, 32),
           'draft_experts': ('nvfp4',), 'kv_cache_8bit': ('int8', 'fp8'), 'kv_cache_4bit': ('int4', 'k8v4'),
           'kv8_prefill': ('bf16',), 'draft_vocab': ('arena16k', 'arena8k-dev', 'arena16k-dev')}
KV8_CACHE_DTYPES = {'int8': 'int8_per_token_head', 'fp8': 'fp8_per_token_head'}
# k8v4 is not a vLLM mode: vLLM allocates an int8 per-token-head cache and arc_kv4 sets the page (392 B per head).
KV4_CACHE_DTYPES = {'int4': 'int4_per_token_head', 'k8v4': 'int8_per_token_head'}
# The hybrid block rule sizes blocks from the indexer's backend (1,024 B per token for any one-byte KV dtype, 1,600
# tokens), but the CSA+linear layout needs a main-KV page at least the GDN state page (1,634,304 B on MTP-0 with BF16
# state): the 4-bit modes set the block size to the smallest multiple of 64 tokens whose page holds it (int4 528 B
# and k8v4 784 B per token and layer; 1,655,808 B pages). MTP-0 only: a larger state page fails vLLM's CSA check.
KV4_BLOCK_TOKENS = {'int4': 3136, 'k8v4': 2112}
KV4_BYTES_PER_TOKEN = {'int4': 2 * (256 + 8), 'k8v4': 2 * (384 + 8)}


def kv4_block_tokens(mode, spec_tokens=0):
    """The 4-bit block (tokens): the smallest multiple of 64 whose main-KV page holds the GDN state page, whose conv
    state grows by one row per speculative token (``gated_delta_net_state_shape``: kernel - 1 + num_spec; BF16 state,
    conv 10,240 channels, recurrent 48 x 128 x 128). MTP-0 and MTP-1 keep 3,136 (int4); MTP-2 needs 3,200, MTP-3 3,264
    (spec-decode.md, 30 September)."""
    if not spec_tokens:
        return KV4_BLOCK_TOKENS[mode]
    page = 10240 * (3 + int(spec_tokens)) * 2 + 48 * 128 * 128 * 2
    return 64 * -(-page // (64 * KV4_BYTES_PER_TOKEN[mode]))
FLAGS = ('hc_fusion', 'topk_sampler', 'ple_uva', 'embed_host', 'indexer_fp8')
# The pinned serving environment's PLE switch (serving_setup.runtime_environment sets "1"); ple_uva serves the PLE
# on the GPU worker instead of the CPU offload process.
PLE_OFFLOAD_ENV = 'VLLM_PLE_CPU_OFFLOAD'
# The fixed-2x-mtp3-wide-kernels arm (docs/validation-20260927/kernels-phase4.md): MTP depth and overlay. From the
# phase-4 microbenchmark: b12x for the target's MoE is 0.53x the CUTLASS time at 1 token and 0.78-0.90x at 6-32, but
# its output error against the FP32 reference is 23% against CUTLASS's 13.5-15% (the pre-stated criterion was +1
# point), so the target keeps CUTLASS and b12x serves only the NvFp4 draft experts; the HC fusion measured slower
# than the pinned chain (10.8-15 us against 8.5-10 us); the top-k-first sampler passed (0.29 ms against 0.5-2.2 ms).
PHASE4_MTP_TOKENS = 3
PHASE4 = {'mamba_ssm_cache_dtype': 'bfloat16', 'moe_backend': None, 'draft_experts': 'nvfp4',
          'hc_fusion': False, 'topk_sampler': True}
# The fixed-2x-mtp3-wide-tweaks arm (docs/validation-20260927/serving-tweaks.md): phase 4 plus the three serving
# tweaks. The vision tower stays: the arena sends every board as a 256 x 256 PNG (MULTIMODAL_CONTEXT current_grid;
# 5,954 multimodal cache queries in the c110-kernels-v2 hour), so --language-model-only cannot serve it.
TWEAKS = {**PHASE4, 'draft_vocab': 'arena16k', 'ple_uva': True, 'embed_host': True}
# the PLE host read stays out of the next arena hour; the other two tweaks only.
TWEAKS_NO_PLE = {**PHASE4, 'draft_vocab': 'arena16k', 'embed_host': True}

MTP = {
    'relative': 'vllm/models/qwen3_8_flash_next/nvidia/mtp.py',
    'sha256': '7735cee47d0d1e4776bebd30d907e4a62160409ce4ef2d65611559f8d58af431',
    'edits': [('    draft_vllm_config.quant_config = draft_quant_config\n    return draft_vllm_config\n',
               '    draft_vllm_config.quant_config = draft_quant_config\n'
               '    # ARC (kernel work, phase 4): NvFp4 draft experts on b12x.\n'
               '    from vllm.arc_serving_kernels import draft_quant_hook as _arc_draft_quant_hook\n'
               '    _arc_draft_quant_hook(draft_vllm_config)\n'
               '    return draft_vllm_config\n', 1)],
}
# Serving tweaks: the draft head over the token subset (Qwen3_8FlashNextMTP.compute_logits, the only
# compute_logits in mtp.py; the V2 speculator calls it once per draft step and greedy drafts take its argmax).
MTP_DRAFT_LOGITS_EDIT = (
    '    ) -> torch.Tensor | None:\n'
    '        return self.logits_processor(self.lm_head, hidden_states)\n',
    '    ) -> torch.Tensor | None:\n'
    '        # ARC (serving tweaks): the draft head over a fixed token subset, -inf elsewhere.\n'
    '        from vllm.arc_serving_kernels import draft_logits as _arc_draft_logits\n'
    '        return _arc_draft_logits(self, hidden_states)\n', 1)
B12X = {
    'relative': 'vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py',
    'sha256': 'e613afe5da2742e42c717c1a0a56bd9e1c7e3e0452a828db0013d81063518954',
    'edits': [('        if self._wrapper is not None:\n            return\n\n',
               '        if self._wrapper is not None:\n            return\n\n'
               '        # ARC (kernel work, phase 4): one wrapper and CUDA-graph workspace per MoE geometry,\n'
               '        # shared by every layer (one per layer is ~0.6 GB x 49 layers).\n'
               '        from vllm.arc_serving_kernels import shared_b12x_wrapper as _arc_shared_b12x_wrapper\n'
               '        self._wrapper = _arc_shared_b12x_wrapper(self)\n'
               '        return\n\n', 1)],
}
HC_CHAIN = ('        lora = hc_silu(lora, self.hc_count)\n'
            '        gate = self.input_mix_weight_up(lora)  # [M, D]\n'
            '        block_input = hc_gate_mix(xn, gate, self.hc_count)\n')
HYPERCONNECTION = {
    'relative': 'vllm/models/qwen3_8_flash_next/nvidia/hyperconnection.py',
    'sha256': '6457124bb3710630bae671848c50864b1d0fc9217135bcec955659e0a0e38f28',
    'edits': [('    hc_silu,\n)\n',
               '    hc_silu,\n)\n'
               '# ARC (kernel work, phase 4): silu + HC up GEMV + gate-mix fused at decode sizes.\n'
               'from vllm.arc_serving_kernels import hc_up_mix as _arc_hc_up_mix\n', 1),
              (HC_CHAIN, '        block_input = _arc_hc_up_mix(self, xn, lora)\n', 2)],
}
SAMPLER = {
    'relative': 'vllm/v1/sample/ops/topk_topp_sampler.py',
    'sha256': 'bc8c1d91ad624368d2bd87a53c0361e550d28b060bcd460462d2f86565c26426',
    'edits': [('def apply_top_k_top_p(\n'
               '    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None\n'
               ') -> torch.Tensor:\n'
               '    if p is None and k is None:\n'
               '        return logits\n',
               'def apply_top_k_top_p(\n'
               '    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None\n'
               ') -> torch.Tensor:\n'
               '    if p is None and k is None:\n'
               '        return logits\n'
               '    # ARC (kernel work, phase 4): top-k first when every row\'s top-k is small.\n'
               '    from vllm.arc_serving_kernels import topk_first as _arc_topk_first\n'
               '    _arc_masked = _arc_topk_first(logits, k, p)\n'
               '    if _arc_masked is not None:\n'
               '        return _arc_masked\n', 1)],
}
STATES = {
    'relative': 'vllm/v1/worker/gpu/sample/states.py',
    'sha256': '4e3671f93695ba5a48fad7d2d13beb231ac8741034904ed5a15d9f7b9b86662e',
    'edits': [('        top_k = self.top_k.gpu[expanded_idx_mapping] if do_top_k else None\n',
               '        top_k = self.top_k.gpu[expanded_idx_mapping] if do_top_k else None\n'
               '        # ARC (kernel work, phase 4): the batch\'s largest top-k, known here on the host.\n'
               '        from vllm.arc_serving_kernels import top_k_hint as _arc_top_k_hint\n'
               '        top_k = _arc_top_k_hint(top_k, self.top_k.np[idx_mapping_np])\n', 1)],
}
QSA_IMPL_INIT = ('        super().__init__(*args, **kwargs)\n'
                 '        if not is_flash_attn_varlen_func_available():\n')
QSA_IMPL_GUARD = ('        if self.kv_cache_dtype not in ("auto", "bfloat16"):\n'
                  '            raise NotImplementedError(\n'
                  '                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"\n'
                  '            )\n'
                  '        self.supports_quant_query_input = False\n'
                  '\n'
                  '    def forward_qsa(\n')
QSA_READ = '        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)\n'
QSA_LAYER_GUARD = ('        if cache_config.cache_dtype not in ("auto", "bfloat16"):\n'
                   '            raise NotImplementedError(\n'
                   '                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"\n'
                   '            )\n')
QSA_STORAGE_GUARD = '        if self.kv_cache_torch_dtype != torch.bfloat16:\n'
QSA_BACKEND = ('    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]\n'
               '\n'
               '    @staticmethod\n'
               '    def get_name() -> str:\n'
               '        return "QWEN38_FLASH_NEXT_QSA_TRITON"\n')
QSA = {
    'relative': 'vllm/models/qwen3_8_flash_next/nvidia/qsa.py',
    'sha256': '748addc85efaa8f7df940d1245bc900192f92e1f17af8fa774625758600751cb',
    'edits': [
        (QSA_BACKEND,
         '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n'
         '        "auto", "bfloat16", "int8_per_token_head", "fp8_per_token_head"]\n'
         '\n'
         '    # ARC (kv_cache_8bit): the per-token-head page (inline fp32 scales) and its shape.\n'
         '    @classmethod\n'
         '    def customize_spec(cls, spec):\n'
         '        from vllm.arc_kv8 import customize_spec as _arc_kv8_customize_spec\n'
         '\n'
         '        return _arc_kv8_customize_spec(spec)\n'
         '\n'
         '    @staticmethod\n'
         '    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):\n'
         '        from vllm.arc_kv8 import kv_cache_shape as _arc_kv8_kv_cache_shape\n'
         '\n'
         '        return _arc_kv8_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)\n'
         '\n'
         '    @staticmethod\n'
         '    def get_name() -> str:\n'
         '        return "QWEN38_FLASH_NEXT_QSA_TRITON"\n', 1),
        (QSA_IMPL_INIT,
         '        # ARC (kv_cache_8bit): FlashAttention never reads this cache; build it as "auto".\n'
         '        from vllm.arc_kv8 import impl_init_args as _arc_kv8_impl_init_args\n'
         '\n'
         '        args, kwargs, _arc_kv8_dtype = _arc_kv8_impl_init_args(args, kwargs)\n'
         '        super().__init__(*args, **kwargs)\n'
         '        if _arc_kv8_dtype is not None:\n'
         '            self.kv_cache_dtype = _arc_kv8_dtype\n'
         '        if not is_flash_attn_varlen_func_available():\n', 1),
        (QSA_IMPL_GUARD,
         '        if self.kv_cache_dtype not in ("auto", "bfloat16") and _arc_kv8_dtype is None:\n'
         '            raise NotImplementedError(\n'
         '                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"\n'
         '            )\n'
         '        self.supports_quant_query_input = False\n'
         '\n'
         '    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):\n'
         '        # ARC (kv_cache_8bit): quantize per token and head into the 8-bit page.\n'
         '        if self.kv_cache_dtype.endswith("per_token_head"):\n'
         '            from vllm.arc_kv8 import cache_update as _arc_kv8_cache_update\n'
         '\n'
         '            return _arc_kv8_cache_update(self, key, value, kv_cache, slot_mapping)\n'
         '        return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)\n'
         '\n'
         '    def forward_qsa(\n', 1),
        (QSA_READ,
         '        if self.kv_cache_dtype.endswith("per_token_head"):\n'
         '            # ARC (kv_cache_8bit): sparse GQA over the 8-bit codes and inline scales.\n'
         '            from vllm.arc_kv8 import attention as _arc_kv8_attention\n'
         '\n'
         '            _arc_kv8_attention(self, query[:num_tokens], kv_cache, logical_indices,\n'
         '                               attn_metadata.block_table, token_to_req, output[:num_tokens])\n'
         '            return output\n' + QSA_READ, 1),
        (QSA_LAYER_GUARD,
         '        # ARC (kv_cache_8bit): only the configured 8-bit dtype passes besides BF16.\n'
         '        from vllm.arc_kv8 import allowed as _arc_kv8_allowed\n'
         '\n'
         '        if cache_config.cache_dtype not in ("auto", "bfloat16") and not _arc_kv8_allowed(\n'
         '            cache_config.cache_dtype\n'
         '        ):\n'
         '            raise NotImplementedError(\n'
         '                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"\n'
         '            )\n', 1),
        (QSA_STORAGE_GUARD,
         '        if self.kv_cache_torch_dtype != torch.bfloat16 and not _arc_kv8_allowed(self.kv_cache_dtype):\n', 1),
    ],
}
TARGETS = (MTP, B12X, HYPERCONNECTION, SAMPLER, STATES, QSA)


def _kv4_edit(text):
    return (text.replace('arc_kv8', 'arc_kv4').replace('kv_cache_8bit', 'kv_cache_4bit').replace('8-bit', '4-bit')
            .replace('"int8_per_token_head", "fp8_per_token_head"', '"int4_per_token_head", "int8_per_token_head"'))


# kv_cache_4bit: the kv8 edits on the same anchors, calling vllm.arc_kv4 (the pinned file and hash are QSA's).
QSA_KV4_EDITS = [(anchor, _kv4_edit(replacement), count) for anchor, replacement, count in QSA['edits']]


def qsa_kv4():
    return {**QSA, 'edits': QSA_KV4_EDITS}


# kv8_prefill: the kv8 edits with the attention entry taken from vllm.arc_kv8_prefill (the only difference).
KV8_ATTENTION_IMPORT = '            from vllm.arc_kv8 import attention as _arc_kv8_attention\n'
KV8_PREFILL_IMPORT = ('            # ARC (kv8_prefill): the BF16 path for prefill rows; decode calls arc_kv8 unchanged.\n'
                      '            from vllm.arc_kv8_prefill import attention as _arc_kv8_attention\n')


def qsa_kv8_prefill():
    edits = [(anchor, replacement.replace(KV8_ATTENTION_IMPORT, KV8_PREFILL_IMPORT), count)
             for anchor, replacement, count in QSA['edits']]
    if sum(r.count(KV8_PREFILL_IMPORT) for _, r, _ in edits) != 1:
        raise RuntimeError('kv8_prefill: the kv8 attention import is not unique in the QSA edits')
    return {**QSA, 'edits': edits}
# Serving tweaks: the target's input embedding (Qwen3_8FlashNextModel; the MTP draft shares it at load).
MODEL = {
    'relative': 'vllm/models/qwen3_8_flash_next/nvidia/model.py',
    'sha256': 'd900cd6fcacba18f460e00b3f018fbf36fbe6ecc310692b6adb1451f7f53cc17',
    'edits': [('        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)\n',
               '        # ARC (serving tweaks): the input embedding in page-locked host memory, read through UVA.\n'
               '        from vllm.arc_serving_kernels import host_embedding as _arc_host_embedding\n'
               '        self.embed_tokens = _arc_host_embedding(\n'
               '            lambda: VocabParallelEmbedding(self.vocab_size, config.hidden_size))\n', 1)],
}
# ple_layer.py as the pinned recipe leaves it (serving_setup.patch_ple_layer: the RadixArk NVFP4 PLE FP8 loader
# patch, VLLM_PLE_PATCHED_SHA256): this overlay is applied after that patch (``apply_ple``), never before it.
PLE = {
    'relative': 'vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py',
    'sha256': 'a8a064744efc3c99eefff649f50395d77e1c36085266034d17b51722f68176ed',
    'edits': [('        with torch.device(PleOffloadLayer.get_target_device()):\n',
               '        # ARC (serving tweaks): the PLE table in page-locked host memory, read by this worker via UVA.\n'
               '        from vllm.arc_serving_kernels import ple_uva_device as _arc_ple_uva_device\n'
               '        from vllm.arc_serving_kernels import ple_uva_place as _arc_ple_uva_place\n'
               '        with torch.device(_arc_ple_uva_device(PleOffloadLayer.get_target_device())):\n', 1),
              ('            )\n        self.key_proj = ReplicatedLinear(\n',
               '            )\n        _arc_ple_uva_place(self.ple_embedding)\n'
               '        self.key_proj = ReplicatedLinear(\n', 1),
              # hero-v2 smoke v1 (29 September): with ple_uva the worker builds the PLE itself, after load-time expert
              # pruning set num_experts to the kept count; the exact NVFP4-PLE FP8 gate pins num_experts == 512 and
              # fell back to the stock loader ("no parameter ngram_embedding.weight_scale"). The gate reads the
              # checkpoint's count that arc_expert_prune.prepare_config records (unchanged without pruning).
              ('            getattr(config, field, None) != expected\n',
               '            # ARC (expert pruning): the checkpoint\'s value when load-time pruning changed it.\n'
               '            getattr(config, "arc_unpruned_" + field, getattr(config, field, None)) != expected\n', 1)],
}
TWEAK_TARGETS = (MODEL, PLE)
# indexer_fp8 (3 October): the compressed-key cache dtype and the scoring kernel's key up-cast.
INDEXER_FP8 = {
    'relative': 'vllm/models/qwen3_8_flash_next/nvidia/indexer_qsa.py',
    'sha256': 'e2f398a2fe29466c9681627651ccb5b8eb2b5980445c9e44b270ff45bcb61066',
    'edits': [('        self.compressed_key_cache = QSACompressedKeyCache(\n'
               '            head_size=self.index_head_dim,\n'
               '            dtype=torch.bfloat16,\n',
               '        # ARC indexer_fp8 (throughput arm, 3 October): float8 e4m3 compressed keys.\n'
               '        self.compressed_key_cache = QSACompressedKeyCache(\n'
               '            head_size=self.index_head_dim,\n'
               '            dtype=torch.float8_e4m3fn,\n', 1)],
}
QSA_OPS_FP8 = {
    'relative': 'vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py',
    'sha256': 'c4ffe3674cafa0ce2dabc39a39f0ddbb4b594bc358ad210ffce9d04383350c7f',
    'edits': [('        scores = tl.dot(keys, query, out_dtype=tl.float32)\n',
               '        # ARC indexer_fp8: float8 keys up-cast to the query dtype (exact) before the MMA.\n'
               '        scores = tl.dot(keys.to(query.dtype), query, out_dtype=tl.float32)\n', 1)],
}


def validate(spec):
    """Normalize a ``server.serving_kernels`` value; None/absent/all-off means no overlay."""
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) - set(OPTIONS):
        raise ValueError(f'serving_kernels takes only {OPTIONS}')
    result = {}
    for key, allowed in CHOICES.items():
        value = spec.get(key)
        if value is not None and value not in allowed:
            raise ValueError(f'serving_kernels.{key} must be one of {allowed} or null')
        result[key] = value
    for key in FLAGS:
        value = spec.get(key, False)
        if type(value) is not bool:
            raise ValueError(f'serving_kernels.{key} must be a boolean')
        result[key] = value
    if not any(result.values()):
        return None
    if result['kv_cache_8bit'] and result['kv_cache_4bit']:
        raise ValueError('serving_kernels.kv_cache_8bit and kv_cache_4bit are exclusive')
    if result['expert_int3'] and result['moe_backend'] != 'humming':
        raise ValueError('serving_kernels.expert_int3 needs moe_backend "humming" (the uint3 repack is Humming\'s)')
    if result['kv8_prefill'] and result['kv_cache_8bit'] != 'int8':
        raise ValueError('serving_kernels.kv8_prefill needs kv_cache_8bit "int8" (the dequant kernel reads int8 codes)')
    return {key: result[key] for key in OPTIONS
            if key not in OPTIONAL or result[key] not in (None, False)}


def targets(spec):
    """The pinned files an (already validated) setting patches at extraction (the PLE file: ``ple_target``)."""
    chosen = []
    mtp_edits = (MTP['edits'] if spec['draft_experts'] else []) + (
        [MTP_DRAFT_LOGITS_EDIT] if spec.get('draft_vocab') else [])
    if mtp_edits:
        chosen.append({**MTP, 'edits': mtp_edits})
    if spec['moe_backend'] == 'flashinfer_b12x' or spec['draft_experts'] == 'nvfp4':
        chosen.append(B12X)
    if spec['hc_fusion']:
        chosen.append(HYPERCONNECTION)
    if spec['topk_sampler']:
        chosen.extend((SAMPLER, STATES))
    if spec.get('embed_host'):
        chosen.append(MODEL)
    if spec.get('kv_cache_8bit'):
        chosen.append(qsa_kv8_prefill() if spec.get('kv8_prefill') else QSA)
    if spec.get('kv_cache_4bit'):
        chosen.append(qsa_kv4())
    if spec.get('expert_int3'):
        from .expert_int3_patch import TARGET as INT3_TARGET

        chosen.append(INT3_TARGET)
    if spec.get('indexer_fp8'):
        chosen.extend((INDEXER_FP8, QSA_OPS_FP8))
    return chosen


def ple_target(spec):
    """The recipe-patched PLE file when ``ple_uva`` is on (``apply_ple``), else None."""
    return PLE if spec.get('ple_uva') else None


def server_flags(spec, spec_tokens=0):
    """Serving-command flags an (already validated) setting adds (``spec_tokens``: the speculative draft length)."""
    flags = []
    if spec['moe_backend']:
        flags += ['--moe-backend', spec['moe_backend']]
    if spec['mamba_ssm_cache_dtype']:
        flags += ['--mamba-ssm-cache-dtype', spec['mamba_ssm_cache_dtype']]
    if spec.get('kv_cache_8bit'):
        flags += ['--kv-cache-dtype', KV8_CACHE_DTYPES[spec['kv_cache_8bit']]]
    if spec.get('kv_cache_4bit'):
        mode = spec['kv_cache_4bit']
        flags += ['--kv-cache-dtype', KV4_CACHE_DTYPES[mode], '--block-size', str(kv4_block_tokens(mode, spec_tokens))]
    return flags


def server_environment(spec, env):
    """The serving process environment for an (already validated) setting: ``ple_uva`` turns the pinned recipe's
    PLE CPU-offload process off, since the worker then serves the PLE itself from the host table."""
    env = dict(env)
    if spec.get('ple_uva'):
        if env.get(PLE_OFFLOAD_ENV) != '1':
            raise RuntimeError(f'Expected the pinned {PLE_OFFLOAD_ENV}=1, got {env.get(PLE_OFFLOAD_ENV)!r}')
        env[PLE_OFFLOAD_ENV] = '0'
    if spec.get('expert_int3'):
        from .expert_int3_patch import ENV as INT3_ENV

        env[INT3_ENV] = json.dumps({'group': int(spec['expert_int3'])}, sort_keys=True)
    return env


def draft_vocab_bytes(spec):
    """The draft subset file for ``draft_vocab`` (checked: sorted distinct ids in range, recorded size and hash)."""
    if not spec.get('draft_vocab'):
        return None
    filename = {'arena8k-dev': 'serving_draft_vocab_dev8k.json',
                'arena16k-dev': 'serving_draft_vocab_dev16k.json'}.get(spec['draft_vocab'], DRAFT_VOCAB_SOURCE)
    data = Path(__file__).with_name(filename).read_bytes()
    record = json.loads(data)
    ids = record['ids']
    if (record['name'] != spec['draft_vocab'] or len(ids) != record['size'] or ids != sorted(set(ids))
            or not 0 <= ids[0] <= ids[-1] < record['vocab_rows']
            or hashlib.sha256(json.dumps(ids).encode()).hexdigest() != record['ids_sha256']):
        raise RuntimeError('The draft-vocabulary file does not match its recorded identity')
    return data


def patched_source(target, original):
    if hashlib.sha256(original).hexdigest() != target['sha256']:
        raise RuntimeError(f"Serving-kernel overlay rejected an unrecognized {target['relative']}")
    text = original.decode()
    for anchor, replacement, count in target['edits']:
        if text.count(anchor) != count:
            raise RuntimeError(f"Serving-kernel overlay anchor count changed in {target['relative']}")
        text = text.replace(anchor, replacement)
    compile(text, target['relative'], 'exec')
    return text.encode()


def apply(site, output, spec):
    """Patch the extracted runtime under ``site`` for ``spec``; the identity goes to ``output``."""
    spec = validate(spec)
    if spec is None:
        raise ValueError('No serving kernels requested; the pinned runtime must stay unpatched')
    site = Path(site)
    helper = Path(__file__).with_name('serving_kernels_runtime.py').read_bytes()
    # The 8-bit KV helper is installed only with its option, so the other arms' runtime stays as recorded.
    kv8_helper = (Path(__file__).with_name('kv8_runtime.py').read_bytes()
                  if spec.get('kv_cache_8bit') or spec.get('kv_cache_4bit') else None)
    # The 4-bit helper (with the 8-bit one: arc_kv4 imports its merge kernel and launch profile).
    kv4_helper = Path(__file__).with_name('kv4_runtime.py').read_bytes() if spec.get('kv_cache_4bit') else None
    # The prefill-path helper, only with its option (phase 7).
    kv8p_helper = (Path(__file__).with_name('kv8_prefill_runtime.py').read_bytes() if spec.get('kv8_prefill')
                   else None)
    # The 3-bit expert helper, only with its option.
    int3_helper = (Path(__file__).with_name('expert_int3_runtime.py').read_bytes() if spec.get('expert_int3')
                   else None)
    settings = (json.dumps(spec, sort_keys=True) + '\n').encode()
    vocab = draft_vocab_bytes(spec)
    marker = Path(output) / MARKER
    chosen = targets(spec)
    originals = {t['relative']: (site / t['relative']).read_bytes() for t in chosen}
    if all(hashlib.sha256(data).hexdigest() == t['sha256'] for t, data in zip(chosen, originals.values())):
        patched = {t['relative']: patched_source(t, originals[t['relative']]) for t in chosen}
        (site / HELPER).write_bytes(helper)
        if kv8_helper is not None:
            (site / KV8_HELPER).write_bytes(kv8_helper)
        if kv4_helper is not None:
            (site / KV4_HELPER).write_bytes(kv4_helper)
        if kv8p_helper is not None:
            (site / KV8_PREFILL_HELPER).write_bytes(kv8p_helper)
        if int3_helper is not None:
            (site / INT3_HELPER).write_bytes(int3_helper)
        (site / SETTINGS).write_bytes(settings)
        if vocab is not None:
            (site / DRAFT_VOCAB).write_bytes(vocab)
        for relative, data in patched.items():
            (site / relative).write_bytes(data)
        identity = {'files': {t['relative']: {'original_sha256': t['sha256'],
                                              'patched_sha256': hashlib.sha256(patched[t['relative']]).hexdigest()}
                              for t in chosen},
                    'helper_sha256': hashlib.sha256(helper).hexdigest(), 'settings': spec,
                    'server_flags': server_flags(spec)}
        if vocab is not None:
            identity['draft_vocab_sha256'] = hashlib.sha256(vocab).hexdigest()
        if kv8_helper is not None:
            identity['kv8_helper_sha256'] = hashlib.sha256(kv8_helper).hexdigest()
        if kv4_helper is not None:
            identity['kv4_helper_sha256'] = hashlib.sha256(kv4_helper).hexdigest()
        if kv8p_helper is not None:
            identity['kv8_prefill_helper_sha256'] = hashlib.sha256(kv8p_helper).hexdigest()
        if int3_helper is not None:
            identity['expert_int3_helper_sha256'] = hashlib.sha256(int3_helper).hexdigest()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(identity, indent=2) + '\n')
        print('ARC_SERVING_KERNELS_PATCH ' + json.dumps(identity, sort_keys=True), flush=True)
        return identity
    # A restart in the same working directory re-extracts the pinned runtime first; anything else already
    # patched with this exact overlay is accepted, never re-patched.
    if marker.exists():
        identity = json.loads(marker.read_text())
        if (identity.get('settings') == spec and set(identity['files']) == set(originals)
                and all(hashlib.sha256(data).hexdigest() == identity['files'][rel]['patched_sha256']
                        for rel, data in originals.items())
                and (site / HELPER).read_bytes() == helper and (site / SETTINGS).read_bytes() == settings
                and (vocab is None or (site / DRAFT_VOCAB).read_bytes() == vocab)
                and (kv8_helper is None or ((site / KV8_HELPER).is_file()
                                            and (site / KV8_HELPER).read_bytes() == kv8_helper))
                and (kv4_helper is None or ((site / KV4_HELPER).is_file()
                                            and (site / KV4_HELPER).read_bytes() == kv4_helper))
                and (kv8p_helper is None or ((site / KV8_PREFILL_HELPER).is_file()
                                             and (site / KV8_PREFILL_HELPER).read_bytes() == kv8p_helper))
                and (int3_helper is None or ((site / INT3_HELPER).is_file()
                                             and (site / INT3_HELPER).read_bytes() == int3_helper))):
            return identity
    raise RuntimeError('Serving-kernel overlay found a modified or partially patched pinned file')


def apply_ple(site, output, spec):
    """Patch the recipe-patched ``ple_layer.py`` for ``ple_uva``. Runs after ``serving_setup.patch_ple_layer``
    (which requires the stock file and writes VLLM_PLE_PATCHED_SHA256) and after ``apply`` (helper and settings)."""
    spec = validate(spec)
    if spec is None or ple_target(spec) is None:
        raise ValueError('No PLE overlay requested; the pinned PLE layer must stay unpatched')
    site = Path(site)
    helper = Path(__file__).with_name('serving_kernels_runtime.py').read_bytes()
    settings = (json.dumps(spec, sort_keys=True) + '\n').encode()
    if (site / HELPER).read_bytes() != helper or (site / SETTINGS).read_bytes() != settings:
        raise RuntimeError('The PLE overlay needs the serving-kernel helper and settings of this study')
    patched = patched_source(PLE, (site / PLE['relative']).read_bytes())
    (site / PLE['relative']).write_bytes(patched)
    identity = {'file': PLE['relative'], 'original_sha256': PLE['sha256'],
                'patched_sha256': hashlib.sha256(patched).hexdigest(), 'settings': spec}
    marker = Path(output) / PLE_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    print('ARC_SERVING_KERNELS_PLE_PATCH ' + json.dumps(identity, sort_keys=True), flush=True)
    return identity
