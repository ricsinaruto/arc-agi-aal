"""Hash-guarded optional serving overlay (kernel work, phase 4); pinned sources stay untouched otherwise.

A study's ``server.serving_kernels`` (absent: the pinned runtime is served unchanged) selects:

- ``mamba_ssm_cache_dtype: "bfloat16"``: ``--mamba-ssm-cache-dtype bfloat16``. The MTP draft's config hook
  otherwise applies the checkpoint's float32 GDN recurrent state; MTP-0 already runs bf16 (no hook), and
  the fused GDN kernels accept both. Halves the state pages (MTP-3 blocks 1,600 -> 832 tokens).
- ``draft_experts: "nvfp4"``: the MTP draft layer's BF16 routed experts (4.69 GiB) quantized to NvFp4 at
  load and served by b12x (``mtp.py`` patch); the b12x workspace is shared by all layers (patch below).
  Draft-only: acceptance, never output correctness.
- ``topk_sampler``: top-k-first sampling mask (``topk_topp_sampler.py`` + ``states.py``).
- ``kv_cache_4bit: "int4" | "k8v4"`` (29 September): the QSA main KV cache in 4 bits (``int4``: keys and values,
  ``--kv-cache-dtype int4_per_token_head``, the pinned runtime's sub-byte page with its random Hadamard rotation and
  zero point) or with 8-bit keys and 4-bit values (``k8v4``, served as ``int8_per_token_head`` with this overlay's
  page), read and written by ``kv4_runtime.py`` (``vllm/arc_kv4.py``). ``nvidia/qsa.py`` is patched to lift its
  BF16-only guards for exactly the configured dtype (docs/validation-20260927/kv-cache-4bit.md).

Serving tweaks (28 September; docs/validation-20260927/serving-tweaks.md), each off unless set:

- ``draft_vocab: "arena16k"``: the MTP draft head computes logits over a fixed 16,384-token subset
  (``serving_draft_vocab.json``: every byte and special token plus the most generated tokens of the
  specialization corpus) and writes them into a full-width row that is -inf elsewhere (``mtp.py``
  ``compute_logits``). Draft-only: rejection sampling keeps the output exact, only acceptance can move.
- ``pin_overlap`` (with ``ple_uva``; MFU-2 22 Oct boot cut): the PLE host table's page-locking runs in a thread
  beside the weight load; the table is joined and placed as the CUDA view after its shards are copied in.
- ``ple_uva``: the PLE n-gram table (47.7 GiB FP8) in page-locked, device-mapped host memory, read by the
  GPU worker's own PLE path through UVA inside the graph, instead of the CPU offload process and its round
  trip (``ple_layer.py``, patched after the pinned recipe's own PLE patch; ``VLLM_PLE_CPU_OFFLOAD=0``).
- ``embed_host``: the target's input embedding (248,320 x 2,560 BF16, 1.18 GiB) in page-locked host
  memory, read through UVA (``model.py``); the MTP draft shares it.

Each option patches only its own pinned files, each guarded by the pinned file's sha256 and exact anchors;
the helper (``serving_kernels_runtime.py``) is copied to ``vllm/arc_serving_kernels.py`` with the settings
in ``vllm/arc_serving_kernels.json`` (and the draft subset in ``vllm/arc_draft_vocab.json``). The flags
are appended to the serving command by ``arena.serving``.

Removed options (trunk/rl cleanup, 26 Oct): ``moe_backend``, ``hc_fusion`` and ``indexer_fp8`` are accepted only at
their off value (null / false); kv_cache_8bit, kv8_prefill and expert_int3 are gone.
"""

import hashlib
import json
from pathlib import Path

HELPER = 'vllm/arc_serving_kernels.py'
KV4_HELPER = 'vllm/arc_kv4.py'
SETTINGS = 'vllm/arc_serving_kernels.json'
DRAFT_VOCAB = 'vllm/arc_draft_vocab.json'
DRAFT_VOCAB_SOURCE = 'serving_draft_vocab.json'  # next to this module (scripts/specialization_vocab.py draft-vocab)
MARKER = 'serving-kernels-patch.json'
PLE_MARKER = 'serving-kernels-ple-patch.json'
# Phase-4 options are always written to the settings. Options added after a recorded arm (the serving tweaks,
# 28 September) are kept out of the canonical setting while off, so the recorded arms' settings (and the settings
# file the server reads) stay byte-identical.
PHASE4_OPTIONS = ('mamba_ssm_cache_dtype', 'moe_backend', 'draft_experts', 'hc_fusion', 'topk_sampler')
TWEAK_OPTIONS = ('draft_vocab', 'ple_uva', 'embed_host')
OPTIONS = PHASE4_OPTIONS + ('kv_cache_4bit',) + TWEAK_OPTIONS + ('indexer_fp8', 'pin_overlap')
OPTIONAL = ('kv_cache_4bit',) + TWEAK_OPTIONS + ('indexer_fp8', 'pin_overlap')
# Removed options, accepted only at this value (moe_backend and hc_fusion stay in the written settings).
RETIRED = {'moe_backend': None, 'hc_fusion': False, 'indexer_fp8': False}
CHOICES = {'mamba_ssm_cache_dtype': ('bfloat16',), 'draft_experts': ('nvfp4',), 'kv_cache_4bit': ('int4', 'k8v4'),
           'draft_vocab': ('arena16k',)}
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
FLAGS = ('topk_sampler', 'ple_uva', 'embed_host', 'pin_overlap')
# The pinned serving environment's PLE switch (serving_setup.runtime_environment sets "1"); ple_uva serves the PLE
# on the GPU worker instead of the CPU offload process.
PLE_OFFLOAD_ENV = 'VLLM_PLE_CPU_OFFLOAD'
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
# kv_cache_4bit: the QSA edits calling vllm.arc_kv4.
QSA = {
    'relative': 'vllm/models/qwen3_8_flash_next/nvidia/qsa.py',
    'sha256': '748addc85efaa8f7df940d1245bc900192f92e1f17af8fa774625758600751cb',
    'edits': [
        (QSA_BACKEND,
         '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n'
         '        "auto", "bfloat16", "int4_per_token_head", "int8_per_token_head"]\n'
         '\n'
         '    # ARC (kv_cache_4bit): the per-token-head page (inline fp32 scales) and its shape.\n'
         '    @classmethod\n'
         '    def customize_spec(cls, spec):\n'
         '        from vllm.arc_kv4 import customize_spec as _arc_kv4_customize_spec\n'
         '\n'
         '        return _arc_kv4_customize_spec(spec)\n'
         '\n'
         '    @staticmethod\n'
         '    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):\n'
         '        from vllm.arc_kv4 import kv_cache_shape as _arc_kv4_kv_cache_shape\n'
         '\n'
         '        return _arc_kv4_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)\n'
         '\n'
         '    @staticmethod\n'
         '    def get_name() -> str:\n'
         '        return "QWEN38_FLASH_NEXT_QSA_TRITON"\n', 1),
        (QSA_IMPL_INIT,
         '        # ARC (kv_cache_4bit): FlashAttention never reads this cache; build it as "auto".\n'
         '        from vllm.arc_kv4 import impl_init_args as _arc_kv4_impl_init_args\n'
         '\n'
         '        args, kwargs, _arc_kv4_dtype = _arc_kv4_impl_init_args(args, kwargs)\n'
         '        super().__init__(*args, **kwargs)\n'
         '        if _arc_kv4_dtype is not None:\n'
         '            self.kv_cache_dtype = _arc_kv4_dtype\n'
         '        if not is_flash_attn_varlen_func_available():\n', 1),
        (QSA_IMPL_GUARD,
         '        if self.kv_cache_dtype not in ("auto", "bfloat16") and _arc_kv4_dtype is None:\n'
         '            raise NotImplementedError(\n'
         '                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"\n'
         '            )\n'
         '        self.supports_quant_query_input = False\n'
         '\n'
         '    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):\n'
         '        # ARC (kv_cache_4bit): quantize per token and head into the 4-bit page.\n'
         '        if self.kv_cache_dtype.endswith("per_token_head"):\n'
         '            from vllm.arc_kv4 import cache_update as _arc_kv4_cache_update\n'
         '\n'
         '            return _arc_kv4_cache_update(self, key, value, kv_cache, slot_mapping)\n'
         '        return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)\n'
         '\n'
         '    def forward_qsa(\n', 1),
        (QSA_READ,
         '        if self.kv_cache_dtype.endswith("per_token_head"):\n'
         '            # ARC (kv_cache_4bit): sparse GQA over the 4-bit codes and inline scales.\n'
         '            from vllm.arc_kv4 import attention as _arc_kv4_attention\n'
         '\n'
         '            _arc_kv4_attention(self, query[:num_tokens], kv_cache, logical_indices,\n'
         '                               attn_metadata.block_table, token_to_req, output[:num_tokens])\n'
         '            return output\n' + QSA_READ, 1),
        (QSA_LAYER_GUARD,
         '        # ARC (kv_cache_4bit): only the configured 4-bit dtype passes besides BF16.\n'
         '        from vllm.arc_kv4 import allowed as _arc_kv4_allowed\n'
         '\n'
         '        if cache_config.cache_dtype not in ("auto", "bfloat16") and not _arc_kv4_allowed(\n'
         '            cache_config.cache_dtype\n'
         '        ):\n'
         '            raise NotImplementedError(\n'
         '                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"\n'
         '            )\n', 1),
        (QSA_STORAGE_GUARD,
         '        if self.kv_cache_torch_dtype != torch.bfloat16 and not _arc_kv4_allowed(self.kv_cache_dtype):\n', 1),
    ],
}
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


def validate(spec):
    """Normalize a ``server.serving_kernels`` value; None/absent/all-off means no overlay."""
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) - set(OPTIONS):
        raise ValueError(f'serving_kernels takes only {OPTIONS}')
    result = {}
    for key, off in RETIRED.items():
        if key in spec and spec[key] is not off:
            raise ValueError(f'serving_kernels.{key} was removed (26 Oct); only {off!r} is accepted')
        result[key] = off
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
    if result['pin_overlap'] and not result['ple_uva']:
        raise ValueError('serving_kernels.pin_overlap needs ple_uva (it defers the PLE host table\'s page-locking)')
    return {key: result[key] for key in OPTIONS
            if key not in OPTIONAL or result[key] not in (None, False)}


def targets(spec):
    """The pinned files an (already validated) setting patches at extraction (the PLE file: ``ple_target``)."""
    chosen = []
    mtp_edits = (MTP['edits'] if spec['draft_experts'] else []) + (
        [MTP_DRAFT_LOGITS_EDIT] if spec.get('draft_vocab') else [])
    if mtp_edits:
        chosen.append({**MTP, 'edits': mtp_edits})
    if spec['draft_experts']:
        chosen.append(B12X)
    if spec['topk_sampler']:
        chosen.extend((SAMPLER, STATES))
    if spec.get('embed_host'):
        chosen.append(MODEL)
    if spec.get('kv_cache_4bit'):
        chosen.append(QSA)
    return chosen


def ple_target(spec):
    """The recipe-patched PLE file when ``ple_uva`` is on (``apply_ple``), else None."""
    return PLE if spec.get('ple_uva') else None


def server_flags(spec, spec_tokens=0):
    """Serving-command flags an (already validated) setting adds (``spec_tokens``: the speculative draft length)."""
    flags = []
    if spec['mamba_ssm_cache_dtype']:
        flags += ['--mamba-ssm-cache-dtype', spec['mamba_ssm_cache_dtype']]
    if spec.get('kv_cache_4bit'):
        mode = spec['kv_cache_4bit']
        if spec_tokens and spec.get('mamba_ssm_cache_dtype') != 'bfloat16':
            # The block rule sizes the page from the BF16 GDN state; with a draft and no BF16 setting the MTP config
            # hook serves the checkpoint's FP32 state, whose larger page fails vLLM's CSA check at start.
            raise ValueError('serving_kernels.kv_cache_4bit with MTP needs mamba_ssm_cache_dtype "bfloat16"')
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
    return env


def draft_vocab_bytes(spec):
    """The draft subset file for ``draft_vocab`` (checked: sorted distinct ids in range, recorded size and hash)."""
    if not spec.get('draft_vocab'):
        return None
    data = Path(__file__).with_name(DRAFT_VOCAB_SOURCE).read_bytes()
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
    kv4_helper = Path(__file__).with_name('kv4_runtime.py').read_bytes() if spec.get('kv_cache_4bit') else None
    settings = (json.dumps(spec, sort_keys=True) + '\n').encode()
    vocab = draft_vocab_bytes(spec)
    marker = Path(output) / MARKER
    chosen = targets(spec)
    originals = {t['relative']: (site / t['relative']).read_bytes() for t in chosen}
    if all(hashlib.sha256(data).hexdigest() == t['sha256'] for t, data in zip(chosen, originals.values())):
        patched = {t['relative']: patched_source(t, originals[t['relative']]) for t in chosen}
        (site / HELPER).write_bytes(helper)
        if kv4_helper is not None:
            (site / KV4_HELPER).write_bytes(kv4_helper)
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
        if kv4_helper is not None:
            identity['kv4_helper_sha256'] = hashlib.sha256(kv4_helper).hexdigest()
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
                and (kv4_helper is None or ((site / KV4_HELPER).is_file()
                                            and (site / KV4_HELPER).read_bytes() == kv4_helper))):
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
