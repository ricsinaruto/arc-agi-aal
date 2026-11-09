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
- ``api_gc``: after ``freeze_gc_heap`` each vLLM process sets GC thresholds from ``ARC_API_GC`` and logs
  ``ARC_API_GC`` lines (rss_gib, gc_s, full_per_min); see api_gc_runtime.py (gc1, 23 Oct).
"""

import hashlib
import json
from pathlib import Path

MARKER = 'serving-fast-patch.json'
SWITCHES = ('logprob_fast', 'image_cache', 'mm_warmup_overlap', 'api_gc')

LOGPROBS = {'relative': 'vllm/v1/engine/logprobs.py',
            'sha256': 'b4ac11f40a99d6b409db8c6ee0a190e630c9ef01919104e653d2965f23e36e15'}
CHAT = {'relative': 'vllm/entrypoints/openai/chat_completion/serving.py',
        'sha256': 'a2440b6b76ad87de86bcf6fc7061ad5d02ab67c1c3c89ca7de3eeb29aa12aa2f'}
IMAGE = {'relative': 'vllm/multimodal/media/image.py',
         'sha256': '2605d8ca98c29fa9f2b208358f1049e2c60948b7494fcb08c4b334fe8acb1a38'}
IMAGE_HELPER = 'vllm/arc_image_cache.py'
RENDERER = {'relative': 'vllm/renderers/base.py',
            'sha256': 'a358921359b241df2e5c08b547897b874a0eb5c73bc9051293bab800511aeedb'}
ASYNC_LLM = {'relative': 'vllm/v1/engine/async_llm.py',
             'sha256': 'bceed0b3f5f0c834fef79525f2462a092f082390f0070526280abc95945837dd'}
GC_UTILS = {'relative': 'vllm/utils/gc_utils.py',
            'sha256': 'cccb8b66611c8c409b964df49078ca1b02dd38700d343e6fa298b5ef1fc84632'}
GC_HELPER = 'vllm/arc_api_gc.py'

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
GC_EDITS = [
    ("""    # Freeze all GC tracked objects
    gc.freeze()
""", """    # Freeze all GC tracked objects
    gc.freeze()
    # ARC (serving_fast.api_gc): thresholds from ARC_API_GC once the heap is frozen (arc_api_gc.py)
    from vllm.arc_api_gc import tune

    tune()
"""),
]
ASYNC_EDITS = [
    ("""        self.renderer = renderer = renderer_from_config(self.vllm_config)
""", """        self.renderer = renderer = renderer_from_config(self.vllm_config)
        # ARC (serving_fast.mm_warmup_overlap): warm the multimodal processor while the engine starts
        renderer.arc_start_mm_warmup()
"""),
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


OPTIONAL = ('mm_warmup_overlap', 'api_gc')


def plan(spec):
    """(pin, edits, (helper path, source, identity key) or None) per switched-on overlay."""
    out = []
    if spec.get('logprob_fast'):
        out += [(LOGPROBS, LP_EDITS, None), (CHAT, CHAT_EDITS, None)]
    if spec.get('image_cache'):
        out.append((IMAGE, IMAGE_EDITS, (IMAGE_HELPER, Path(__file__).with_name('image_cache_runtime.py').read_bytes(),
                                         'helper_sha256')))
    if spec.get('mm_warmup_overlap'):
        out += [(RENDERER, WARMUP_EDITS, None), (ASYNC_LLM, ASYNC_EDITS, None)]
    if spec.get('api_gc'):
        out.append((GC_UTILS, GC_EDITS, (GC_HELPER, Path(__file__).with_name('api_gc_runtime.py').read_bytes(),
                                         'gc_helper_sha256')))
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
            if helper is not None:
                relative, source, key = helper
                (site / relative).write_bytes(source)
                identity[key] = hashlib.sha256(source).hexdigest()
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
