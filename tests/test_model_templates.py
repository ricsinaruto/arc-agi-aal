import ast
import copy
import hashlib
import json
from pathlib import Path

import httpx
import jinja2
import pytest

from arc_infra.model_templates import NEX_MODEL, NEX_REVISION, template_kwargs
from arena.serving_core.clock import Clock
from arena.serving_core.config import Config
from arena.serving_core.model import RequestLayer
from arena.trial import duck_config

ROOT = Path(__file__).resolve().parents[1]


def render(messages, **kwargs):
    path = ROOT / 'tests/fixtures/nex-chat-template.jinja'
    original = path.read_bytes()
    assert hashlib.sha256(original).hexdigest() == 'f1753536417ee87cded4bc5017354eb2123c079d20c634403a3d808b2ba3cc5b'

    def fail(message):
        raise ValueError(message)

    return jinja2.Environment().from_string(original.decode()).render(
        messages=messages, add_generation_prompt=True, raise_exception=fail, **kwargs)


def test_native_nex_controls_actually_disable_summary_thinking():
    messages = [{'role': 'user', 'content': 'Summarize the rules.'}]
    legacy = {'enable_thinking': False, 'reasoning_effort': 'xhigh'}
    assert render(messages, **legacy).endswith('<think>')  # reproduces the silent incompatibility
    assert render(messages, **template_kwargs(NEX_MODEL, legacy)).endswith('<think>\n\n</think>\n\n')
    gameplay = template_kwargs(NEX_MODEL, {'enable_thinking': True, 'reasoning_effort': 'high'})
    assert render(messages, **gameplay).endswith('<think>\n')
    assert legacy == {'enable_thinking': False, 'reasoning_effort': 'xhigh'}
    with pytest.raises(ValueError, match='native'):
        template_kwargs(NEX_MODEL, {'enable_thinking': True, 'reasoning_effort': 'xhigh'})


def test_nex_native_image_tools_and_reasoning_history_are_explicit():
    messages = [
        {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'test-only'}},
                                     {'type': 'text', 'text': 'Initial board'}]},
        {'role': 'assistant', 'content': '', 'reasoning_content': 'OLD_REASONING_SENTINEL',
         'tool_calls': [{'function': {'name': 'python', 'arguments': {'code': 'print(42)'}}}]},
        {'role': 'tool', 'content': '42'},
        {'role': 'user', 'content': 'New board'},
    ]
    frozen = copy.deepcopy(messages)
    text = render(messages, preserve_thinking=False, reasoning_effort='high')
    assert '<|vision_start|><|image_pad|><|vision_end|>' in text
    assert '<function=python>\n<parameter=code>\nprint(42)' in text
    assert '<tool_response>\n42\n</tool_response>' in text
    assert 'OLD_REASONING_SENTINEL' in text  # Nex ignores Flash's retention flag
    assert messages == frozen


def test_pinned_qwen_parser_also_needs_native_nex_thinking_switch():
    """Execute the real parser's constructor and extract method, without CUDA imports."""
    source = ROOT / 'tests/fixtures/vllm-qwen3-parser.py.txt'
    provenance = json.loads(source.with_name('vllm-qwen3-parser-source.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance['sha256']
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3Parser')
    cls.body = [n for n in cls.body if isinstance(n, ast.Assign)
                or isinstance(n, ast.FunctionDef) and n.name in {'__init__', 'extract_reasoning'}]

    class EngineStub:
        def __init__(self, tokenizer, tools, **kwargs):
            self.vocab = {}
            self.config = kwargs['parser_engine_config']

        def extract_reasoning(self, text, request):
            return 'delegated-to-engine', None

    ns = {'ParserEngine': EngineStub, 'qwen3_config': lambda **kwargs: kwargs,
          'THINK_START': '<think>', 'THINK_END': '</think>',
          'TOOL_CALL_START': '<tool_call>', 'TOOL_CALL_END': '</tool_call>'}
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), ns)
    parser = ns['Qwen3Parser']
    old = parser(None, chat_template_kwargs={'reasoning_effort': 'none'})
    assert old.thinking_enabled and old.config['thinking']  # former broken payload
    kwargs = template_kwargs(NEX_MODEL, {'enable_thinking': False})
    fixed = parser(None, chat_template_kwargs=kwargs)
    assert not fixed.thinking_enabled and not fixed.config['thinking']
    assert fixed.extract_reasoning('A colored image.', None) == (None, 'A colored image.')
    assert render([{'role': 'user', 'content': 'Describe the image.'}], **kwargs) == render(
        [{'role': 'user', 'content': 'Describe the image.'}], reasoning_effort='none')
    gameplay = parser(None, chat_template_kwargs=template_kwargs(NEX_MODEL, {'reasoning_effort': 'high'}))
    assert gameplay.thinking_enabled and gameplay.config['thinking']


@pytest.mark.parametrize('summary', [False, True])
@pytest.mark.parametrize('effort', ['high', 'none'])
async def test_preflight_counts_with_the_native_settings(summary, effort):
    bodies = []

    def handle(request):
        bodies.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={'count': 500, 'max_model_len': 262144})

    cfg = Config(model=NEX_MODEL, model_revision=NEX_REVISION, reasoning_effort=effort)
    clock = Clock(1800)
    clock.release()
    async with httpx.AsyncClient(base_url='http://local/v1/', transport=httpx.MockTransport(handle)) as client:
        layer = RequestLayer(cfg, clock, client=client)
        assert await layer.count(layer.payload([{'role': 'user', 'content': 'Board'}], summary=summary)) == 500
    assert [p for p, _ in bodies] == ['/tokenize']
    assert bodies[0][1]['chat_template_kwargs'] == {'reasoning_effort': 'none' if summary else effort,
                                                   'enable_thinking': not summary and effort != 'none'}


def test_the_duck_agent_records_the_candidate_revision():
    study = json.loads((ROOT / 'configs/arena_30m.json').read_text())
    study.update(served_model=NEX_MODEL, model_revision=NEX_REVISION, reasoning_effort='high')
    kwargs = dict(window=131072, cap=16384, wall=1800, base_url='http://local/v1')
    duck = duck_config(study, ROOT, **kwargs)
    assert duck.model == NEX_MODEL
    assert duck.model_revision == NEX_REVISION
    assert duck.reasoning_effort == 'high'
