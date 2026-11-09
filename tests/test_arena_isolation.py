import httpx
import pytest

from arena import isolation


@pytest.mark.parametrize('reply', [{'success': True}, {'success': False}, {'success': 'true'}])
async def test_isolation_waits_for_idle_and_requires_confirmed_reset(monkeypatch, reply):
    calls = []

    async def idle(url):
        calls.append('idle')

    async def sleep(seconds):
        pass

    def handle(request):
        assert calls[0] == 'idle'
        assert request.url.path == '/reset_prefix_cache'
        assert request.url.params['reset_external'] == 'true'
        calls.append('reset')
        return httpx.Response(200, json=reply)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(isolation, 'wait_idle', idle)
    monkeypatch.setattr(isolation.asyncio, 'sleep', sleep)
    monkeypatch.setattr(isolation.httpx, 'AsyncClient', lambda **kwargs: client)
    if reply['success'] is True:
        result = await isolation.isolate_trial('http://localhost/v1')
        assert result['prefix_cache_reset'] and result['attempts'] == 1
    else:
        with pytest.raises(RuntimeError, match='refusing contaminated'):
            await isolation.isolate_trial('http://localhost/v1')
        assert len(calls) == 11
