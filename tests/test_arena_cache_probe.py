import asyncio
import json

import httpx
import pytest

from arena import cache_probe
from arena.cache import BlockBudget


@pytest.mark.parametrize('failure,cpu_gib', [(None, 0), (None, 16), ('count', 0), ('preempt', 0), ('no_load', 16), ('empty', 0)])
async def test_probe_requires_exact_count_no_preemptions_and_real_offload(monkeypatch, tmp_path, failure, cpu_gib):
    layout = {'schema': 1, 'num_blocks': 100, 'max_model_len': 262144, 'lookahead': 3,
              'costs': [{'tokens': 262144, 'blocks': 20, 'groups': [20]}]}
    monkeypatch.setattr(cache_probe.BlockBudget, 'load', lambda *a: BlockBudget(layout))
    resets = []

    def respond(request):
        path = request.url.path
        if path == '/tokenize':
            body = json.loads(request.content)
            assert 'image_url' in json.dumps(body['messages'])
            return httpx.Response(200, json={'count': 100})
        if path == '/v1/chat/completions':
            return httpx.Response(200, json={'usage': {'prompt_tokens': 101 if failure == 'count' else 100,
                                                     'completion_tokens': 3},
                                            'choices': [{'finish_reason': 'stop', 'message':
                                                {'content': None, 'reasoning': 'A colored image.'} if failure == 'empty'
                                                else {'content': 'A colored image.'}}]})
        if path == '/reset_prefix_cache':
            resets.append(request.url.params['reset_external'])
            return httpx.Response(200, json={'success': True})
        raise AssertionError(path)

    client = httpx.AsyncClient
    monkeypatch.setattr(cache_probe.httpx, 'AsyncClient',
                        lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))

    class Monitor:
        def __init__(self, *args):
            self.samples = []

        async def sample(self):
            self.samples.append({'metrics': {'vllm:kv_offload_load_bytes_total':
                                             0 if failure == 'no_load' else 100 * len(self.samples)}})

        async def run(self):
            await asyncio.Event().wait()

        def summary(self):
            return {'valid': True, 'errors': [], 'preemptions': int(failure == 'preempt')}

    async def idle(*args):
        pass

    monkeypatch.setattr(cache_probe, 'GpuMonitor', Monitor)
    monkeypatch.setattr(cache_probe, 'wait_idle', idle)
    study = {'kv_margin': 0.9, 'generation_slots': 25, 'cache_experiment': {'cpu_cache_gib': cpu_gib}}
    if failure:
        with pytest.raises(RuntimeError):
            await cache_probe.run(tmp_path, study, 'http://local/v1')
    else:
        await cache_probe.run(tmp_path, study, 'http://local/v1')
    result = json.loads((tmp_path / 'cache-probe/result.json').read_text())
    assert result['state'] == ('error' if failure else 'complete')
    if not failure:
        assert len(result['requests']) == 100 and result['gate']['reserved'] == 0
        assert resets == ['true']
    else:
        assert not resets
    if failure == 'empty':
        assert result['requests'] and not result['requests'][0]['validated']
        assert result['requests'][0]['response']['choices'][0]['message']['reasoning'] == 'A colored image.'
        assert result['requests'][0]['request']['chat_template_kwargs']['enable_thinking'] is False
        assert result['requests'][0]['response']['choices'][0]['finish_reason'] == 'stop'
