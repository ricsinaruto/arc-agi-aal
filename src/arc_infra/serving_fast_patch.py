"""Hash-guarded API-front-end overlays (MFU-2, 22 Oct; fe2 profile: the API main thread was ~70% busy, about half
of it on top-20 logprob alternatives, and 8 threads re-decoded every history image on every request).

``server.serving_fast = {"logprob_fast": bool, "image_cache": bool}`` (each switch on its own; absent = pinned):

- ``logprob_fast``: a chat request that returns tokens as ``token_id:N`` (``return_tokens_as_token_ids``) never shows
  an alternative's text, so the output processor decodes only the sampled token of each position (same UTF-8
  correction, same context); an alternative equal to the sampled id keeps the sampled text, as before. The chat
  layer marks such requests through ``SamplingParams.extra_args`` and builds the top-logprob entries with
  ``model_construct`` (no per-entry validation; same fields set, same JSON). Replies are byte-identical.
- ``image_cache``: ``ImageMediaIO.load_base64`` keeps decoded images keyed by the base64 payload and the decode
  settings (LRU, 4 GiB); a repeated history image returns the same decoded object
  (its ``original_bytes`` and so its multimodal hash are those of the same payload).
- ``logprob_lean`` (needs ``logprob_fast``; real-v5 27 Oct: construct/dump + bytes lists were ~half of each API
  process's core): a full chat reply with ``token_id:N`` tokens builds its logprob entries as plain dicts without
  ``bytes`` and is written by one json.dumps (logprob_lean_runtime.py). Token ids, values, alternatives and order
  are unchanged; nothing of ours reads ``bytes``. Stream replies keep the pinned path.
- ``api_gc``: after ``freeze_gc_heap`` each vLLM process sets GC thresholds (api_gc_runtime.THRESHOLDS) and logs
  ``ARC_API_GC`` lines (rss_gib, gc_s, full_per_min); see api_gc_runtime.py (gc1, 23 Oct).
"""

import hashlib
import json
from pathlib import Path

MARKER = 'serving-fast-patch.json'
SWITCHES = ('logprob_fast', 'image_cache', 'mm_warmup_overlap', 'api_gc', 'logprob_lean')

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
ROUTER = {'relative': 'vllm/entrypoints/openai/chat_completion/api_router.py',
          'sha256': '118e4e7dfd74d09a296fe4aa8ad094ea829cee5e107c7ac78fed92249db12d1a'}
LEAN_HELPER = 'vllm/arc_lean.py'

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

LEAN_CHAT_EDITS = [
    ("""        logprob_token_ids: list[int] | None = None,
        return_as_token_id: bool | None = None,
    ) -> ChatCompletionLogProbs:
        \"\"\"Create OpenAI-style logprobs.\"\"\"
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
""", """        logprob_token_ids: list[int] | None = None,
        return_as_token_id: bool | None = None,
        arc_lean: bool = False,
    ) -> ChatCompletionLogProbs:
        \"\"\"Create OpenAI-style logprobs.\"\"\"
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
        if arc_lean and should_return_as_token_id:  # ARC (serving_fast.logprob_lean): plain entries, no bytes
            from vllm.arc_lean import lean_logprobs

            return lean_logprobs(ChatCompletionLogProbs, token_ids, top_logprobs, num_output_top_logprobs,
                                 bool(logprob_token_ids))
"""),
    ("""                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            if parser is not None:
""", """                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                    arc_lean=True,  # ARC (serving_fast.logprob_lean): the full reply only
                )
            else:
                logprobs = None

            if parser is not None:
"""),
]

ROUTER_EDITS = [
    ("""    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )
""", """    elif isinstance(generator, ChatCompletionResponse):
        from vllm.arc_lean import lean_response  # ARC (serving_fast.logprob_lean)

        lean = lean_response(generator, metrics_header(metrics_header_format))
        if lean is not None:
            return lean
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )
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
    # ARC (serving_fast.api_gc): GC thresholds once the heap is frozen (arc_api_gc.py)
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


def reply_ids_inserts():
    """The prefix-cache reply_ids patch's inserts in the chat-serving file (arena.prefix_cache_patch, applied first at
    boot): the same file logprob_fast edits, at other anchors."""
    from arena import prefix_cache_patch as P

    return (P.REPLY_IMPORT_INSERT, P.REPLY_RENDER_INSERT, P.REPLY_FINISH_INSERT)


def base_of(original, pin):
    """(pinned base, the overlays already on it): the pinned file itself, or (chat serving only) the pinned file plus
    exactly reply_ids' inserts, each once; anything else is None."""
    if hashlib.sha256(original).hexdigest() == pin['sha256']:
        return original, ()
    if pin is not CHAT:
        return None
    text = original.decode()
    try:
        inserts = reply_ids_inserts()
    except ImportError:
        return None
    if any(text.count(i) != 1 for i in inserts):
        return None
    for i in inserts:
        text = text.replace(i, '')
    if hashlib.sha256(text.encode()).hexdigest() != pin['sha256']:
        return None
    return original, ('reply_ids',)


def edited(original, pin, edits):
    if base_of(original, pin) is None:
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
    if spec.get('logprob_lean') and not spec.get('logprob_fast'):
        raise ValueError('serving_fast.logprob_lean needs logprob_fast (its edits sit on the same file)')
    out = {k: bool(spec.get(k, False)) for k in SWITCHES}
    return {k: v for k, v in out.items() if k not in OPTIONAL or v}  # a later switch stays out while off


OPTIONAL = ('mm_warmup_overlap', 'api_gc', 'logprob_lean')


def plan(spec):
    """(pin, edits, (helper path, source, identity key) or None) per switched-on overlay."""
    out = []
    if spec.get('logprob_fast'):
        out += [(LOGPROBS, LP_EDITS, None),
                (CHAT, CHAT_EDITS + (LEAN_CHAT_EDITS if spec.get('logprob_lean') else []), None)]
    if spec.get('logprob_lean'):
        out.append((ROUTER, ROUTER_EDITS, (LEAN_HELPER, Path(__file__).with_name('logprob_lean_runtime.py').read_bytes(),
                                           'lean_helper_sha256')))
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
    bases = {p['relative']: base_of(originals[p['relative']], p) for p, _, _ in work}
    if all(b is not None for b in bases.values()):
        identity = {'switches': spec, 'files': {}}
        patched = {p['relative']: edited(originals[p['relative']], p, e) for p, e, _ in work}  # all checked first
        for pin, _, helper in work:
            if helper is not None:
                relative, source, key = helper
                (site / relative).write_bytes(source)
                identity[key] = hashlib.sha256(source).hexdigest()
            (site / pin['relative']).write_bytes(patched[pin['relative']])
            identity['files'][pin['relative']] = {'original_sha256': pin['sha256'],
                                                  'patched_sha256': hashlib.sha256(patched[pin['relative']]).hexdigest(),
                                                  **({'on_top_of': list(bases[pin['relative']][1])}
                                                     if bases[pin['relative']][1] else {})}
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
