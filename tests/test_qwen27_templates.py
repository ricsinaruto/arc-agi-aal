"""Render both pinned checkpoints' identical native templates, not an imitation."""
import hashlib
import json
from pathlib import Path

import jinja2
import pytest

from arc_infra.model_templates import template_kwargs
from arena.candidate_profiles import QWEN_27_FP8, QWEN_27_NVFP4

ROOT = Path(__file__).resolve().parents[1]


def render(messages, **kwargs):
    fixture = ROOT / 'tests/fixtures/qwen27-chat-template.jinja'
    provenance = json.loads(fixture.with_name('qwen27-template-source.json').read_text())
    text = fixture.read_bytes()
    assert hashlib.sha256(text).hexdigest() == provenance['sha256']

    def fail(message):
        raise ValueError(message)

    return jinja2.Environment().from_string(text.decode()).render(
        messages=messages, add_generation_prompt=True, raise_exception=fail, **kwargs)


@pytest.mark.parametrize('profile', [QWEN_27_NVFP4, QWEN_27_FP8])
def test_native_qwen27_controls_and_image_tool_format(profile):
    messages = [
        {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'test-only'}},
                                    {'type': 'text', 'text': 'Board'}]},
        {'role': 'assistant', 'content': '', 'reasoning_content': 'OLD_REASONING_SENTINEL',
         'tool_calls': [{'function': {'name': 'python', 'arguments': {'code': 'print(42)'}}}]},
        {'role': 'tool', 'content': '42'},
        {'role': 'user', 'content': 'New board'},
    ]
    kwargs = {'enable_thinking': True, 'reasoning_effort': profile.effort, 'preserve_thinking': True}
    assert template_kwargs(profile.model, kwargs) == kwargs
    text = render(messages, **kwargs)
    assert text.endswith('<think>\n') and 'Reasoning effort is set to xhigh.' in text
    assert '<|vision_start|><|image_pad|><|vision_end|>' in text
    assert '<function=python>\n<parameter=code>\nprint(42)' in text
    assert '<tool_response>\n42\n</tool_response>' in text
    assert 'OLD_REASONING_SENTINEL' in text
    assert 'OLD_REASONING_SENTINEL' not in render(messages, **{**kwargs, 'preserve_thinking': False})
    summary = render([{'role': 'user', 'content': 'Summarize evidence.'}],
                     **template_kwargs(profile.model, {'enable_thinking': False}))
    assert summary.endswith('<think>\n\n</think>\n\n')
    assert 'Reasoning effort is set' not in summary
    with pytest.raises(ValueError, match='Unexpected reasoning effort'):
        render(messages, reasoning_effort='high')
