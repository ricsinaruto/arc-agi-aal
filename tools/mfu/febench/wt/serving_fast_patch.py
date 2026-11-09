"""Hash-guarded API-front-end overlays (MFU-2, 22 Oct; fe2 profile: the API main thread was ~70% busy, about half
of it on top-20 logprob alternatives, and 8 threads re-decoded every history image on every request).

``server.serving_fast = {"logprob_fast": bool, "image_cache": bool}`` (each switch on its own; absent = pinned):

- ``logprob_fast``: a chat request that returns tokens as ``token_id:N`` (``return_tokens_as_token_ids``) never shows
  an alternative's text, so the output processor decodes only the sampled token of each position (same UTF-8
  correction, same context); an alternative equal to the sampled id keeps the sampled text, as before. The chat
  layer marks such requests through ``SamplingParams.extra_args`` and builds the top-logprob entries with
  ``model_construct`` (no per-entry validation; same fields set, same JSON). Replies are byte-identical.
- ``image_cache``: ``ImageMediaIO.load_base64`` keeps decoded images keyed by the base64 payload and the decode
  settings (LRU, ``ARC_IMAGE_CACHE_MB``, default 4096); a repeated history image returns the same decoded object
  (its ``original_bytes`` and so its multimodal hash are those of the same payload).
"""

import hashlib
import json
from pathlib import Path

MARKER = 'serving-fast-patch.json'
SWITCHES = ('logprob_fast', 'image_cache', 'mm_warmup_overlap', 'expert_load_fast', 'mtp_lora_draft', 'lora_fused_dense',
            'lora_meta_cpu', 'moe_lora_fast')

LOGPROBS = {'relative': 'vllm/v1/engine/logprobs.py',
            'sha256': 'b4ac11f40a99d6b409db8c6ee0a190e630c9ef01919104e653d2965f23e36e15'}
CHAT = {'relative': 'vllm/entrypoints/openai/chat_completion/serving.py',
        'sha256': 'a2440b6b76ad87de86bcf6fc7061ad5d02ab67c1c3c89ca7de3eeb29aa12aa2f'}
IMAGE = {'relative': 'vllm/multimodal/media/image.py',
         'sha256': '2605d8ca98c29fa9f2b208358f1049e2c60948b7494fcb08c4b334fe8acb1a38'}
IMAGE_HELPER = 'vllm/arc_image_cache.py'
WEIGHT_UTILS = {'relative': 'vllm/model_executor/model_loader/weight_utils.py',
                'sha256': '25bb3393105f9a642b7fa58bc277865db9976748be320787f8db2e03b16fd667'}
FAST_LOAD_HELPER = 'vllm/arc_fast_load.py'
RENDERER = {'relative': 'vllm/renderers/base.py',
            'sha256': 'a358921359b241df2e5c08b547897b874a0eb5c73bc9051293bab800511aeedb'}
TRTLLM_LORA_MOE = {'relative': 'vllm/model_executor/layers/fused_moe/experts/trtllm_lora_moe.py',
                   'sha256': '3a223a513a6ea228464dc95284011da142e9263c735d3b13db539dec52312367'}
PUNICA_GPU = {'relative': 'vllm/lora/punica_wrapper/punica_gpu.py',
              'sha256': '7ddedad89b6721751012af6f72bd85d0f8e2f6e9b288fc1909b08a732c360919'}
LORA_FUSED_HELPER = 'vllm/arc_lora_fused.py'
MOE_LORA_HELPER = 'vllm/arc_moe_lora.py'
ASYNC_LLM = {'relative': 'vllm/v1/engine/async_llm.py',
             'sha256': 'bceed0b3f5f0c834fef79525f2462a092f082390f0070526280abc95945837dd'}

LP_EDITS = [
    ("""    num_logprobs: int | None
    num_prompt_logprobs: int | None
""", """    num_logprobs: int | None
    num_prompt_logprobs: int | None
    # ARC (serving_fast.logprob_fast): only the sampled token's text is shown (token_id:N replies)
    arc_sampled_only = False
"""),
    ("""        num_prompt_logprobs = sampling_params.prompt_logprobs
        return cls(
""", """        num_prompt_logprobs = sampling_params.prompt_logprobs
        made = cls(
"""),
    ("""            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
        )
""", """            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
        )
        made.arc_sampled_only = bool((sampling_params.extra_args or {}).get("arc_sampled_only"))
        return made
"""),
    ("""            if self.tokenizer is None:
                decoded_tokens = NONES
            else:
                decoded_tokens_list = convert_ids_list_to_tokens(
                    self.tokenizer, token_ids
                )
""", """            if self.tokenizer is None:
                decoded_tokens = NONES
            elif self.arc_sampled_only:
                # ARC: the sampled token's text as the full path gives it (per-index correction, same context);
                # alternatives are shown as token_id:N, so their text is never read (the sampled id keeps its text)
                sampled = self._verify_tokens(
                    decoded_tokens_list=convert_ids_list_to_tokens(self.tokenizer, token_ids[:1]),
                    tokens=token_ids[:1],
                    context_token_ids=self._get_sampled_context_ids(self.logprobs),
                )[0]
                decoded_tokens = [sampled if t == token_ids[0] else None for t in token_ids]
            else:
                decoded_tokens_list = convert_ids_list_to_tokens(
                    self.tokenizer, token_ids
                )
"""),
]

CHAT_EDITS = [
    ("""            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.default_sampling_params,
                )
""", """            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.default_sampling_params,
                )
                # ARC (serving_fast.logprob_fast): replies show tokens as token_id:N (same rule as
                # _create_chat_logprobs), so only the sampled token's text is needed
                if request.logprobs and (request.return_tokens_as_token_ids
                                         if request.return_tokens_as_token_ids is not None
                                         else self.return_tokens_as_token_ids):
                    sampling_params.extra_args = {**(sampling_params.extra_args or {}), "arc_sampled_only": True}
"""),
    ("""        return [
            ChatCompletionLogProb(
                token=(
                    token := self._get_decoded_token(
""", """        return [
            ChatCompletionLogProb.model_construct(  # ARC: no per-entry validation (same fields, same JSON)
                token=(
                    token := self._get_decoded_token(
"""),
]

IMAGE_EDITS = [
    ("""    def load_base64(self, media_type: str, data: str) -> MediaWithBytes[Image.Image]:
        return self.load_bytes(pybase64.b64decode(data, validate=True))
""", """    def load_base64(self, media_type: str, data: str) -> MediaWithBytes[Image.Image]:
        # ARC (serving_fast.image_cache): a repeated payload returns its decoded image
        from vllm.arc_image_cache import cached

        return cached(self, data, lambda: self.load_bytes(pybase64.b64decode(data, validate=True)))
"""),
]


WARMUP_EDITS = [
    ("""            if self.mm_processor:
                try:
                    logger.debug("Warming up multi-modal processing...")
""", """        # ARC (serving_fast.mm_warmup_overlap): started beside the engine's start-up; wait for it here
        arc_early = getattr(self, "_arc_mm_warmup_thread", None)
        if arc_early is not None:
            arc_early.join()
        else:
            self._arc_warmup_mm()

    def arc_start_mm_warmup(self) -> None:
        # ARC: the multimodal processor warmup in a thread (AsyncLLM, before the engine handshake)
        import threading

        if self.mm_processor or self._readonly_mm_processor is not None:
            self._arc_mm_warmup_thread = threading.Thread(
                target=self._arc_warmup_mm, name="arc-mm-warmup", daemon=True
            )
            self._arc_mm_warmup_thread.start()

    def _arc_warmup_mm(self) -> None:
        with set_default_torch_num_threads(1):
            if self.mm_processor:
                try:
                    logger.debug("Warming up multi-modal processing...")
"""),
]
ASYNC_EDITS = [
    ("""        self.renderer = renderer = renderer_from_config(self.vllm_config)
""", """        self.renderer = renderer = renderer_from_config(self.vllm_config)
        # ARC (serving_fast.mm_warmup_overlap): warm the multimodal processor while the engine starts
        renderer.arc_start_mm_warmup()
"""),
]


FAST_LOAD_EDITS = [
    ("""    leftover_state_dict: dict[str, torch.Tensor] = {}
    for st_file in tqdm(
""", """    leftover_state_dict: dict[str, torch.Tensor] = {}
    # ARC (serving_fast.expert_load_fast): expert-only shards go to the GPU in whole-file copies and their tensors
    # are device views (same names, order and bytes); ARC_FAST_EXPERT_LOAD=0 restores the default path
    if (
        safetensors_load_strategy in (None, "lazy")
        and not is_offload_process()
        and os.environ.get("ARC_FAST_EXPERT_LOAD", "1") != "0"
    ):
        from vllm.arc_fast_load import iterate as _arc_iterate

        def _arc_plain(st_file):
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    yield name, f.get_tensor(name)

        yield from _arc_iterate(
            sorted_files,
            _arc_plain,
            skip=lambda name: should_skip_weight(name, local_expert_ids),
            progress=lambda files: tqdm(
                files,
                desc=loading_desc,
                disable=not enable_tqdm(use_tqdm_on_load),
                bar_format=_BAR_FORMAT,
            ),
        )
        return
    for st_file in tqdm(
"""),
]

# mtp_lora_draft: with runtime LoRA on, the MTP draft's MoE still gets the LoRA-capable TRT-LLM experts but the LoRA
# manager never wraps the draft, so no context is ever set; a missing context now runs the base MoE (the draft has no
# adapter), the path a batch without LoRA tokens already takes.
MTP_LORA_EDITS = [
    ('''        lora_context = self._lora_context
        assert lora_context is not None, "LoRA context must be set"
''', '''        lora_context = self._lora_context  # ARC (serving_fast.mtp_lora_draft): None = a module without LoRA
'''),
    ('''        if self._batch_has_no_lora(lora_context):
''', '''        if lora_context is None or self._batch_has_no_lora(lora_context):
'''),
]

# lora_fused_dense: a linear layer's LoRA (zero + shrink + expand) as one kernel at decode batch sizes; larger batches
# and uncovered layouts keep vLLM's kernels (vllm/arc_lora_fused.py decides inside one custom op).
# lora_meta_cpu: the punica metadata built from the host-side mapping (no device sync per step; same values).
LORA_IMPORT_EDIT = ('''from vllm import _custom_ops as ops
''', '''from vllm import _custom_ops as ops
import vllm.arc_lora_fused as _arc_lora  # ARC (serving_fast.lora_fused_dense / lora_meta_cpu)
''')
LORA_FUSED_EDITS = [
    ('''        r = lora_b_stacked[0].size(-1)
''', '''        if kwargs.get("add_inputs", True):  # ARC (serving_fast.lora_fused_dense)
            y2, x2 = y.view(-1, y.shape[-1]), x.view(-1, x.shape[-1])
            _arc_lora.arc_lora_linear(y2, x2, list(lora_a_stacked), list(lora_b_stacked), scale, list(output_slices),
                                      *self.token_mapping_meta.meta_args(x2.size(0),
                                                                         self.lora_config.specialize_active_lora))
            return
        r = lora_b_stacked[0].size(-1)
'''),
]
LORA_META_EDITS = [
    ('''        # TODO avoid gpu<->cpu sync here
        with gpu_sync_allowed():
            # Prepare cuda kernel metadata tensors
            self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
            self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)
''', '''        # ARC (serving_fast.lora_meta_cpu): the same metadata from the host-side mapping, without a device sync
        slot = {lid: i for i, lid in enumerate(lora_index_to_id) if lid is not None}
        _arc_lora.prepare_cpu(self.token_mapping_meta, [slot[v] if v > 0 else -1 for v in mapping.index_mapping])
        _arc_lora.prepare_cpu(self.prompt_mapping_meta, [slot[v] if v > 0 else -1 for v in mapping.prompt_mapping])
'''),
]

# moe_lora_fast: routed-expert LoRA where vLLM would run align + its sorted one-shot kernel: one-CTA pair lists per
# layer (carried from w13 to w2 in the num_tokens_post_padded slot) and persistent kernels over live (lora, expert)
# items only; bitwise the one-shot kernel's arithmetic (vllm/arc_moe_lora.py).
MOE_IMPORT_EDIT = ('''from vllm import _custom_ops as ops
''', '''from vllm import _custom_ops as ops
import vllm.arc_moe_lora as _arc_moe  # ARC (serving_fast.moe_lora_fast)
''')
MOE_LORA_EDITS = [
    ('''        (
            token_lora_mapping,
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
        ) = self.moe_lora_align_block_size(
''', '''        if _arc_moe.fast_ok(num_tokens, top_k, local_num_experts, max_loras, max_lora_rank, fully_sharded,
                            expert_map):  # ARC (serving_fast.moe_lora_fast): no align pass
            meta = self.token_mapping_meta.meta_args(num_tokens, self.lora_config.specialize_active_lora)
            if token_lora_mapping is None:
                token_lora_mapping = meta[0]
            pair_lists = None
            if not meta[5].item():
                pair_lists = _arc_moe.lists(topk_ids, token_lora_mapping, adapter_enabled,
                                            lora_a_stacked[0].shape[0], lora_a_stacked[0].shape[1])
                _arc_moe.moe_lora(y.view(-1, top_k_num, y.shape[-1]), x, lora_a_stacked, lora_b_stacked, pair_lists,
                                  topk_weights, top_k, False, "w13")
            return None, topk_ids.reshape(-1), pair_lists, token_lora_mapping
        (
            token_lora_mapping,
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
        ) = self.moe_lora_align_block_size(
'''),
    ('''        _sorted = sorted_token_ids_lora
        _eids = expert_ids_lora
        if _sorted is not None:
            assert _eids is not None
''', '''        if sorted_token_ids_lora is None and _arc_moe.fast_ok(
                num_tokens, top_k, lora_a_stacked[0].shape[1], max_loras, max_lora_rank, fully_sharded):
            if num_tokens_post_padded_lora is not None:  # ARC (serving_fast.moe_lora_fast): w13's pair lists
                _arc_moe.moe_lora(y, x, lora_a_stacked, lora_b_stacked, num_tokens_post_padded_lora, topk_weights,
                                  top_k, True, "w2")
            return
        _sorted = sorted_token_ids_lora
        _eids = expert_ids_lora
        if _sorted is not None:
            assert _eids is not None
'''),
]


def edited(original, pin, edits):
    if hashlib.sha256(original).hexdigest() != pin['sha256']:
        raise RuntimeError(f"Serving-fast overlay rejected an unrecognized {pin['relative']}")
    text = original.decode()
    for old, new in edits:
        if text.count(old) != 1:
            raise RuntimeError(f"Serving-fast overlay: an edit site changed in {pin['relative']}")
        text = text.replace(old, new)
    compile(text, pin['relative'], 'exec')
    return text.encode()


def validate(spec):
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) - set(SWITCHES) or any(type(v) is not bool for v in spec.values()):
        raise ValueError(f'serving_fast is a dict of booleans over {list(SWITCHES)}')
    out = {k: bool(spec.get(k, False)) for k in SWITCHES}
    return {k: v for k, v in out.items() if k not in OPTIONAL or v}  # a later switch stays out while off


OPTIONAL = ('mm_warmup_overlap', 'expert_load_fast', 'mtp_lora_draft', 'lora_fused_dense', 'lora_meta_cpu',
            'moe_lora_fast')


def plan(spec):
    """(pin, edits, (helper path, helper source) or None) per switched-on overlay."""
    out = []
    if spec.get('logprob_fast'):
        out += [(LOGPROBS, LP_EDITS, None), (CHAT, CHAT_EDITS, None)]
    if spec.get('image_cache'):
        out.append((IMAGE, IMAGE_EDITS,
                    (IMAGE_HELPER, Path(__file__).with_name('image_cache_runtime.py').read_bytes())))
    if spec.get('mm_warmup_overlap'):
        out += [(RENDERER, WARMUP_EDITS, None), (ASYNC_LLM, ASYNC_EDITS, None)]
    if spec.get('expert_load_fast'):
        out.append((WEIGHT_UTILS, FAST_LOAD_EDITS,
                    (FAST_LOAD_HELPER, Path(__file__).with_name('fast_load_runtime.py').read_bytes())))
    if spec.get('mtp_lora_draft'):
        out.append((TRTLLM_LORA_MOE, MTP_LORA_EDITS, None))
    dense = spec.get('lora_fused_dense') or spec.get('lora_meta_cpu')
    if dense or spec.get('moe_lora_fast'):  # one file: the switched-on edits together
        edits, helpers = [], []
        if dense:
            edits += [LORA_IMPORT_EDIT] + (LORA_FUSED_EDITS if spec.get('lora_fused_dense') else []) + \
                (LORA_META_EDITS if spec.get('lora_meta_cpu') else [])
            helpers.append((LORA_FUSED_HELPER, Path(__file__).with_name('lora_fused_runtime.py').read_bytes()))
        if spec.get('moe_lora_fast'):
            edits += [MOE_IMPORT_EDIT] + MOE_LORA_EDITS
            helpers.append((MOE_LORA_HELPER, Path(__file__).with_name('moe_lora_runtime.py').read_bytes()))
        out.append((PUNICA_GPU, edits, helpers))
    return out


def apply(site, output, spec):
    """Patch the extracted runtime under ``site``; identity to ``output``. A restart in the same working directory
    accepts only this exact overlay."""
    spec = validate(spec)
    if spec is None or not any(spec.values()):
        return None
    site = Path(site)
    marker = Path(output) / MARKER
    work = plan(spec)
    originals = {pin['relative']: (site / pin['relative']).read_bytes() for pin, _, _ in work}
    if all(hashlib.sha256(originals[p['relative']]).hexdigest() == p['sha256'] for p, _, _ in work):
        identity = {'switches': spec, 'files': {}}
        patched = {p['relative']: edited(originals[p['relative']], p, e) for p, e, _ in work}  # all checked first
        for pin, _, helper in work:
            for h in ([] if helper is None else helper if isinstance(helper, list) else [helper]):
                (site / h[0]).write_bytes(h[1])
                identity.setdefault('helpers', {})[h[0]] = hashlib.sha256(h[1]).hexdigest()
                if h[0] == IMAGE_HELPER:
                    identity['helper_sha256'] = identity['helpers'][h[0]]
            (site / pin['relative']).write_bytes(patched[pin['relative']])
            identity['files'][pin['relative']] = {'original_sha256': pin['sha256'],
                                                  'patched_sha256': hashlib.sha256(patched[pin['relative']]).hexdigest()}
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(identity, indent=2) + '\n')
        print('ARC_SERVING_FAST_PATCH ' + json.dumps(identity, sort_keys=True), flush=True)
        return identity
    if marker.exists():
        identity = json.loads(marker.read_text())
        if identity.get('switches') == spec and all(
                hashlib.sha256(originals[f]).hexdigest() == row['patched_sha256']
                for f, row in identity['files'].items()):
            return identity
    raise RuntimeError('Serving-fast overlay found a modified or partially patched pinned file')
