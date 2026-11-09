"""Real loopback HTTP disconnects cancel queued/streaming Pi inference, not receipts."""

import asyncio
from types import SimpleNamespace

import aiohttp
import httpx
import pytest
from aiohttp import web

from arena.clock import ActiveClock, GlobalClock
from arena.gate import GateSlot, KVGate
from arena.pi import ArenaRequestLayer
from arena.scheduler import GameRecord
from qwen_baselines.config import Config
from qwen_baselines.evidence import Archive
from qwen_baselines.pi_agent import PiAgent


@pytest.mark.parametrize('queued', [True, False])
async def test_client_disconnect_cancels_work_and_preserves_prior_receipts(tmp_path, queued):
    wall = GlobalClock(1800)
    game = GameRecord('game', 0, ActiveClock(wall))
    wall.release()
    game.clock.release()
    gate = KVGate(300000, 1)
    holder = await gate.acquire('other', 1000) if queued else None
    slot = GateSlot(gate, game, cap=16384, window=131072)
    archive = Archive(tmp_path)
    streaming = asyncio.Event()
    stream_closed = asyncio.Event()
    receipts = []

    class PartialStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"reasoning_content":"partial"}}]}\n\n'
            streaming.set()
            await asyncio.Event().wait()

        async def aclose(self):
            stream_closed.set()

    def endpoint(request):
        if request.url.path == '/tokenize':
            return httpx.Response(200, json={'count': 500, 'max_model_len': 262144})
        return httpx.Response(200, stream=PartialStream())

    async def begin_turn():
        # An acknowledged boundary reset must survive cancellation of later inference.
        receipts.append('acknowledged reset')
        archive.event('test_reset_receipt', action='RESET')

    async with httpx.AsyncClient(base_url='http://mock/v1/',
                                transport=httpx.MockTransport(endpoint)) as transport:
        layer = ArenaRequestLayer(Config(output_tokens_cap=16384), game, slot, client=transport)
        agent = object.__new__(PiAgent)
        agent.turns, agent.layer = 0, layer
        agent.bridge = SimpleNamespace(begin_turn=begin_turn, archive=archive)
        runner = agent.gateway_runner('test-token')
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        url = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/test-token/v1/chat/completions'
        try:
            async with aiohttp.ClientSession() as client:
                request = asyncio.create_task(client.post(url, json={
                    'messages': [{'role': 'user', 'content': 'board'}],
                    'tools': [{'type': 'function', 'function': {'name': 'read'}}],
                    'stream': True,
                }))
                async with asyncio.timeout(2):
                    while not (gate.queue if queued else streaming.is_set()):
                        await asyncio.sleep(.001)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                async with asyncio.timeout(2):
                    while not game.admission_attempts or game.admission_attempts[-1]['status'] not in (
                            'cancelled_wait', 'cancelled_context'):
                        await asyncio.sleep(.001)
            assert game.admission_attempts[-1]['status'] == ('cancelled_wait' if queued else 'cancelled_context')
            assert gate.inflight == (1 if queued else 0) and not gate.queue
            assert receipts == ['acknowledged reset']
            kinds = [r[0] for r in archive.db.execute('select kind from events')]
            assert kinds.count('test_reset_receipt') == 1
            assert 'response' not in kinds
            if not queued:
                assert stream_closed.is_set()
                assert 'generation_chunk' in kinds and 'generation_interrupted' in kinds
            if holder:
                gate.release(holder)
            assert gate.inflight == gate.reserved == 0
        finally:
            await runner.cleanup()
    archive.close()
