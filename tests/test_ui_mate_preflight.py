"""Bounded native-serving checks use synthetic actions, never public environments."""

import json
from dataclasses import replace

import httpx
import pytest
from test_ui_mate_actor import STUDY, render_native_request, structured, xml

from arena.candidate_profiles import PROFILES
from arena.trial import qwen_config
from ui_mate_actor.preflight import check


def config(profile):
    cfg = qwen_config(STUDY, 'ui-mate-native', window=131072, cap=16384,
                      wall=1800, base_url='http://model/v1')
    return replace(cfg, model=PROFILES[profile].model, model_revision=PROFILES[profile].revision)


@pytest.mark.parametrize('profile', ['ui-mate-9b', 'ui-mate-27b'])
@pytest.mark.parametrize('native_xml', [False, True])
async def test_live_protocol_check_records_action_and_valid_followup(tmp_path, profile, native_xml):
    generations, counts = [], []
    def endpoint(request):
        body = json.loads(request.content)
        if request.url.path == '/tokenize':
            render_native_request(profile, body)
            counts.append(body)
            return httpx.Response(200, json={'count': 400, 'max_model_len': 262144})
        generations.append(body)
        msg = {'content': xml(action='press', keys=['1'])} if native_xml else structured(
            {'action': 'press', 'keys': '["1"]'})
        return httpx.Response(200, json={'choices': [{'message': msg, 'finish_reason': 'stop'}],
                                        'usage': {'completion_tokens': 30}})
    cfg = config(profile)
    async with httpx.AsyncClient(base_url=cfg.base_url+'/', transport=httpx.MockTransport(endpoint)) as client:
        result = await check(cfg, tmp_path, client=client)
        assert not client.is_closed
    assert result['status'] == 'passed' and result['synthetic_actions'] == 1
    assert result['public_game_actions'] == 0 and result['structured_calls'] is not native_xml
    assert len(generations) == 1 and generations[0]['max_tokens'] == 2048
    assert generations[0]['chat_template_kwargs']['enable_thinking'] is False
    assert counts[-1]['chat_template_kwargs']['enable_thinking'] is True
    assert 'ACTION1' in json.dumps(counts[-1]) and '<function=computer_use>' in json.dumps(counts[-1])
    assert cfg.gameplay_seconds == 1800 and cfg.output_tokens_cap == 16384
    assert json.loads((tmp_path/'result.json').read_text())['status'] == 'passed'


@pytest.mark.parametrize('message,finish', [
    ({'content': 'Ready to play.'}, 'stop'),
    (structured({'action': 'press', 'keys': ['2']}), 'stop'),
    (structured({'action': 'press', 'keys': ['1']}), 'length'),
    (structured({'action': 'press', 'keys': ['1', '1']}), 'stop'),
])
async def test_protocol_failure_stops_and_preserves_diagnostic(tmp_path, message, finish):
    calls = []
    def endpoint(request):
        if request.url.path == '/tokenize':
            return httpx.Response(200, json={'count': 400, 'max_model_len': 262144})
        calls.append(request)
        return httpx.Response(200, json={'choices': [{'message': message, 'finish_reason': finish}]})
    cfg = config('ui-mate-9b')
    async with httpx.AsyncClient(base_url=cfg.base_url+'/', transport=httpx.MockTransport(endpoint)) as client:
        with pytest.raises(RuntimeError, match='Native protocol check'):
            await check(cfg, tmp_path, client=client)
    assert len(calls) == 1  # No silent retries or changed protocol to obtain a pass.
    result = json.loads((tmp_path/'result.json').read_text())
    assert result['status'] == 'error' and result['public_game_actions'] == 0


async def test_protocol_transport_failure_is_not_replayed(tmp_path):
    calls = []
    def endpoint(request):
        if request.url.path == '/tokenize':
            return httpx.Response(200, json={'count': 400, 'max_model_len': 262144})
        calls.append(request)
        raise httpx.ReadError('lost response')
    cfg = config('ui-mate-9b')
    async with httpx.AsyncClient(base_url=cfg.base_url+'/', transport=httpx.MockTransport(endpoint)) as client:
        with pytest.raises(httpx.ReadError):
            await check(cfg, tmp_path, client=client)
    assert len(calls) == 1
    assert json.loads((tmp_path/'result.json').read_text())['status'] == 'error'
