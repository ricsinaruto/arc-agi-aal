"""Counting must expand images without submitting inference or changing requests."""

import ast
import asyncio
import copy
import hashlib
import uuid
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_bench import tokenize_patch


class Internal:
    def __init__(self, chat):
        self.chat = chat
        self.lora_path = chat.get("lora_path")
        self.sampling_params = {"max_new_tokens": 16384}
        self.rid = chat.get('rid', uuid.uuid4().hex)
        self.received_time = 123.0

    def normalize_batch_and_arguments(self):
        self.is_single = self.chat.get("n", 1) == 1


def endpoint(payload, *, failure=None):
    calls = []
    chat = SimpleNamespace(
        _validate_request=lambda request: request.get("invalid"),
        _convert_to_internal_request=lambda request: (Internal(request), request),
    )

    async def preprocess(adapted):
        assert adapted.is_single
        assert adapted.sampling_params["max_new_tokens"] == 0
        calls.append(adapted.chat)
        if failure:
            raise failure
        # Model a processor with different visual lengths, not a fixed per-image
        # correction. Production must use the backend's own expansion.
        return SimpleNamespace(input_ids=array("q", [7] * (
            len(adapted.chat["messages"]) + sum(adapted.chat.get("visual_lengths", [])))))

    states = {}

    def register(adapted, request):
        if adapted.rid in states:
            raise ValueError('Duplicate request ID detected')
        states[adapted.rid] = SimpleNamespace(dispatched=False)

    def release(rids):
        for rid in rids:
            assert not states[rid].dispatched
            del states[rid]

    manager = SimpleNamespace(allow_auto_truncate=False, _tokenize_one_request=preprocess,
                              rid_to_state=states, _init_req_state=register,
                              _release_req_states_on_failure=release)
    request = SimpleNamespace(to_chat_completion_request=lambda: copy.deepcopy(payload))
    return SimpleNamespace(chat_serving=chat, tokenizer_manager=manager), request, calls


@pytest.mark.parametrize("lengths", [[], [64], [64, 256], [1024, 64, 256]])
def test_counts_expanded_visual_inputs_and_preserves_chat_inputs(lengths):
    payload = {"messages": ["system", "user", "assistant reasoning", "tool evidence"],
               "tools": [{"function": {"name": "python"}}],
               "chat_template_kwargs": {"reasoning_effort": "high"},
               "visual_lengths": lengths}
    original = copy.deepcopy(payload)
    serving, request, calls = endpoint(payload)
    tokens = asyncio.run(tokenize_patch._tokenize_chat_request(serving, request))
    assert len(tokens) == 4 + sum(lengths)
    assert calls == [original] and payload == original
    assert serving.tokenizer_manager.rid_to_state == {}
    # No generate_request/scheduler/send method exists on the fake manager.


@pytest.mark.parametrize("failure", [ValueError("oversized"), RuntimeError("bad image"),
                                     asyncio.CancelledError()])
def test_preprocessing_failures_and_cancellation_are_not_hidden(failure):
    serving, request, _ = endpoint({"messages": ["image"]}, failure=failure)
    with pytest.raises(type(failure)):
        asyncio.run(tokenize_patch._tokenize_chat_request(serving, request))
    assert serving.tokenizer_manager.rid_to_state == {}


@pytest.mark.parametrize("case", ["truncate", "template", "invalid", "batch", "lora"])
def test_fail_closed_before_preprocessing(case):
    payload = {"messages": ["image"]}
    if case == "invalid":
        payload["invalid"] = "invalid request"
    if case == "batch":
        payload["n"] = 2
    if case == "lora":
        payload["lora_path"] = "adapter"
    serving, request, calls = endpoint(payload)
    if case == "truncate":
        serving.tokenizer_manager.allow_auto_truncate = True
    if case == "template":
        serving.chat_serving = None
    with pytest.raises(ValueError):
        asyncio.run(tokenize_patch._tokenize_chat_request(serving, request))
    assert not calls


def test_patch_is_atomic_idempotent_and_rejects_drift(tmp_path, monkeypatch):
    source = b'''class OpenAIServingTokenize:
    async def handler(self, request):
        token_ids = self._tokenize_chat_request(request)
        return len(token_ids)
    def _tokenize_chat_request(self, request):
        return [1]
'''
    monkeypatch.setattr(tokenize_patch, "ORIGINAL_SHA256", hashlib.sha256(source).hexdigest())
    path = tmp_path / tokenize_patch.TARGET
    path.parent.mkdir(parents=True)
    path.write_bytes(source)
    first = tokenize_patch.apply(tmp_path)
    assert tokenize_patch.apply(tmp_path) == first
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    serving, request, _ = endpoint({"messages": ["image"], "visual_lengths": [256]})
    actual = namespace["OpenAIServingTokenize"]()
    actual.chat_serving, actual.tokenizer_manager = serving.chat_serving, serving.tokenizer_manager
    assert asyncio.run(actual.handler(request)) == 257
    path.write_text(path.read_text() + "# unrelated edit\n")
    with pytest.raises(RuntimeError, match="unrelated modifications"):
        tokenize_patch.apply(tmp_path)
    path.with_suffix(".py.arc-original").write_bytes(b"unknown source")
    with pytest.raises(RuntimeError, match="pinned runtime"):
        tokenize_patch.apply(tmp_path)


def test_actual_pinned_endpoint_on_disposable_copy(tmp_path):
    source_path = Path(__file__).resolve().parents[1] / "dist/sglang-runtime/stage/opt/pennyroyal" / tokenize_patch.TARGET
    if not source_path.exists():
        pytest.skip("Pinned SGLang runtime not installed locally")
    target = tmp_path / tokenize_patch.TARGET
    target.parent.mkdir(parents=True)
    target.write_bytes(source_path.read_bytes())
    original_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    first = tokenize_patch.apply(tmp_path)
    assert tokenize_patch.apply(tmp_path) == first
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == original_hash
    tree = ast.parse(target.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "OpenAIServingTokenize")
    # Execute the real endpoint methods without importing the CUDA server stack.
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.AsyncFunctionDef)
                and n.name in ("_handle_non_streaming_request", "_tokenize_chat_request")]
    for node in ast.walk(cls):
        if isinstance(node, ast.arg):
            node.annotation = None
        if isinstance(node, ast.AsyncFunctionDef):
            node.returns = None
    namespace = {"TokenizeResponse": lambda **kw: kw}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(target), "exec"), namespace)
    actual = namespace["OpenAIServingTokenize"]()
    serving, request, _ = endpoint({"messages": ["image"], "visual_lengths": [64, 256]})
    actual.chat_serving, actual.tokenizer_manager = serving.chat_serving, serving.tokenizer_manager
    actual.tokenizer_manager.tokenizer = SimpleNamespace(model_max_length=262144,
                                                         encode=lambda text, **kw: [1, 2, 3])
    request.messages = ["image"]
    result = asyncio.run(actual._handle_non_streaming_request(request, request, None))
    assert result["count"] == 321 and len(result["tokens"]) == 321
    assert result["max_model_len"] == 262144
    for prompt, expected in [("text", 3), (["a", "b"], [3, 3])]:
        plain = SimpleNamespace(messages=None, prompt=prompt, add_special_tokens=True)
        result = asyncio.run(actual._handle_non_streaming_request(plain, plain, None))
        assert result["count"] == expected


def test_actual_pinned_manager_expands_placeholder_before_returning_count():
    root = Path(__file__).resolve().parents[1] / "dist/sglang-runtime/stage/opt/pennyroyal"
    path = root / "python/sglang/srt/managers/tokenizer_manager.py"
    if not path.exists():
        pytest.skip("Pinned SGLang runtime not installed locally")
    tree = ast.parse(path.read_text())
    manager_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TokenizerManager")
    method = next(n for n in manager_class.body if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == "_tokenize_one_request")
    for node in ast.walk(method):
        if isinstance(node, ast.arg):
            node.annotation = None
    namespace = {"GenerateReqInput": Internal, "EmbeddingReqInput": type("Embedding", (), {}),
                 "get_disagg": lambda: SimpleNamespace(language_model_only=False, language_only=False),
                 "envs": SimpleNamespace(SGLANG_MM_PRECOMPUTE_HASH=SimpleNamespace(get=lambda: False))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    serving, request, _ = endpoint({"messages": ["image"], "visual_lengths": [64, 256]})
    obj = Internal({"n": 1})
    obj.text, obj.input_ids, obj.input_embeds = "rendered image placeholders", None, None
    obj.image_data, obj.audio_data, obj.video_data = ["first-image", "second-image"], None, None
    obj.contains_mm_input = lambda: True
    calls = []

    async def text_tokens(text, cross_encoder):
        calls.append("text")
        return [1, 2], None

    async def multimodal(**kw):
        calls.append("multimodal")
        assert kw["input_text"] == obj.text and kw["image_data"] == obj.image_data
        return SimpleNamespace(input_ids=list(range(321)), token_type_ids=None, mm_items=[])

    manager = serving.tokenizer_manager
    manager.model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=["Flash"]))
    manager.tokenizer = object()
    manager._tokenize_texts = text_tokens
    manager.mm_processor = SimpleNamespace(prefer_tokenized_input=False, process_mm_data_async=multimodal)
    manager.max_req_input_len = 524288
    manager._validate_mm_limits = lambda obj: None
    manager._normalize_mm_content_hashes = lambda obj: None
    manager._validate_one_request = lambda obj, ids: calls.append(("validated", len(ids)))
    manager._create_tokenized_object = lambda obj, text, ids, *rest: SimpleNamespace(input_ids=array("q", ids))
    manager._tokenize_one_request = lambda adapted: namespace["_tokenize_one_request"](manager, adapted)
    serving.chat_serving._convert_to_internal_request = lambda request: (obj, request)
    assert asyncio.run(tokenize_patch._tokenize_chat_request(serving, request)) == list(range(321))
    assert calls == ["text", "multimodal", ("validated", 321)]


def lifecycle_endpoint(*, images=True, failure=None, initialize=True):
    """Execute pinned lifecycle AND constructor; stub only heavyweight data types/I/O."""
    import json

    fixture = Path(__file__).parent / 'fixtures/sglang-tokenizer-lifecycle.py.txt'
    meta = json.loads(fixture.with_name('sglang-tokenizer-lifecycle-source.json').read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == meta['fixture_sha256']
    runtime = Path(__file__).resolve().parents[1] / 'dist/sglang-runtime/stage/opt/pennyroyal' / meta['source']
    if runtime.exists():
        assert hashlib.sha256(runtime.read_bytes()).hexdigest() == meta['source_sha256']
    tree = ast.parse(fixture.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            node.annotation = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None

    class Request(Internal):
        # Fields not exercised in this text/image count have protocol defaults.
        def __getattr__(self, name):
            return None

        def contains_mm_input(self):
            return bool(self.image_data)

    class TimeStats:
        def __init__(self, **kwargs):
            self.tokenized = False

        def set_created_time(self, value):
            self.created = value

        def set_tokenize_finish_time(self):
            self.tokenized = True

    class Sampling:
        def __init__(self, **kwargs):
            assert kwargs['max_new_tokens'] == 0

        def normalize(self, tokenizer):
            pass

        def verify(self, vocab_size):
            assert vocab_size == 1000

    env = {'GenerateReqInput': Request, 'EmbeddingReqInput': type('Embedding', (), {}),
           'get_disagg': lambda: SimpleNamespace(language_model_only=False, language_only=False,
                                                 disaggregation_transfer_backend=None),
           'envs': SimpleNamespace(SGLANG_MM_PRECOMPUTE_HASH=SimpleNamespace(get=lambda: False)),
           'array': array, 'TokenizedGenerateReqInput': lambda **kw: SimpleNamespace(**kw),
           'APIServerReqTimeStats': TimeStats, 'asyncio': asyncio,
           'ReqState': lambda out, done, event, obj, stats: SimpleNamespace(
               dispatched=False, obj=obj, time_stats=stats)}
    exec(compile(ast.fix_missing_locations(tree), str(fixture), 'exec'), env)
    serving, request, _ = endpoint({'messages': ['image' if images else 'text']})
    manager = serving.tokenizer_manager
    manager.rid_to_state = {}
    manager.enable_trace = False
    manager.disaggregation_mode = None
    manager.model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=['Flash']), vocab_size=1000)
    manager.tokenizer = object()
    manager.preferred_sampling_params = None
    manager.sampling_params_class = Sampling
    manager.max_req_input_len = 524288
    manager._validate_mm_limits = lambda obj: None
    manager._normalize_mm_content_hashes = lambda obj: None
    manager._validate_one_request = lambda *args: None
    seen = []

    async def tokenize(text, cross_encoder):
        if failure:
            raise failure
        await asyncio.sleep(0)
        return [1, 2], None

    async def multimodal(**kwargs):
        await asyncio.sleep(0)
        return SimpleNamespace(input_ids=list(range(321)), token_type_ids=None, mm_items=[])

    manager._tokenize_texts = tokenize
    manager.mm_processor = SimpleNamespace(prefer_tokenized_input=False, process_mm_data_async=multimodal)
    for name in env:
        if name.startswith('_'):
            setattr(manager, name, lambda *args, _name=name: env[_name](manager, *args))
    if not initialize:
        manager._init_req_state = lambda *args: None

    def convert(chat):
        obj = Request(chat)
        obj.text = 'rendered text'
        obj.input_ids = obj.input_embeds = None
        obj.image_data = ['image'] if images else None
        obj.video_data = obj.audio_data = None
        seen.append(obj)
        return obj, chat

    serving.chat_serving._convert_to_internal_request = convert
    return serving, request, seen


@pytest.mark.parametrize('images', [False, True])
def test_pinned_constructor_lifecycle_counts_and_releases_without_dispatch(images):
    serving, request, seen = lifecycle_endpoint(images=images)
    result = asyncio.run(tokenize_patch._tokenize_chat_request(serving, request))
    assert len(result) == (321 if images else 2)
    assert len(seen) == 1 and serving.tokenizer_manager.rid_to_state == {}
    # No scheduler/send/abort method exists: success cannot enqueue GPU work.


def test_pinned_constructor_reproduces_v3_keyerror_without_registration():
    serving, request, _ = lifecycle_endpoint(initialize=False)
    with pytest.raises(KeyError):
        asyncio.run(tokenize_patch._tokenize_chat_request(serving, request))


@pytest.mark.parametrize('failure', [ValueError('oversized'), RuntimeError('bad image'),
                                     asyncio.CancelledError()])
def test_pinned_lifecycle_removes_undispatched_states_after_failure(failure):
    serving, request, _ = lifecycle_endpoint(failure=failure)
    with pytest.raises(type(failure)):
        asyncio.run(tokenize_patch._tokenize_chat_request(serving, request))
    assert serving.tokenizer_manager.rid_to_state == {}


def test_pinned_count_rejects_duplicate_id_without_removing_live_request():
    serving, request, _ = lifecycle_endpoint()
    live = SimpleNamespace(dispatched=True)
    serving.tokenizer_manager.rid_to_state['existing'] = live
    request.to_chat_completion_request = lambda: {'messages': ['text'], 'rid': 'existing'}
    with pytest.raises(ValueError, match='Duplicate request ID'):
        asyncio.run(tokenize_patch._tokenize_chat_request(serving, request))
    assert serving.tokenizer_manager.rid_to_state == {'existing': live}


def test_pinned_counts_can_overlap_without_state_leaks():
    serving, request, seen = lifecycle_endpoint()

    async def run():
        first, second = await asyncio.gather(
            tokenize_patch._tokenize_chat_request(serving, request),
            tokenize_patch._tokenize_chat_request(serving, request))
        assert first == second == list(range(321))

    asyncio.run(run())
    assert len({obj.rid for obj in seen}) == 2
    assert serving.tokenizer_manager.rid_to_state == {}


def test_task_cancellation_during_preprocessing_releases_native_state():
    serving, request, _ = lifecycle_endpoint()

    async def run():
        entered = asyncio.Event()

        async def wait_for_image(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        serving.tokenizer_manager.mm_processor.process_mm_data_async = wait_for_image
        task = asyncio.create_task(tokenize_patch._tokenize_chat_request(serving, request))
        await entered.wait()
        assert len(serving.tokenizer_manager.rid_to_state) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert serving.tokenizer_manager.rid_to_state == {}

    asyncio.run(run())
