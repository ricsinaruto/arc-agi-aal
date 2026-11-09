"""Actual pinned Pi launcher: queued inference must outlive transport idle timers."""
import asyncio
import json
import os
from pathlib import Path
import sys

from aiohttp import web
import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Pinned Linux Node/Pi runtime')


@pytest.mark.parametrize('configure_transport', [False, True])
async def test_arena_sdk_waits_for_delayed_headers_and_still_aborts(tmp_path, configure_transport):
    pi_dir = Path(os.environ.get('ARC_PI_DIR', '/app/pi'))
    launcher = pi_dir.parent / 'pi-arena/agent.mjs'
    source = launcher.read_text()
    call = '  configureHttpDispatcher(settingsManager.getHttpIdleTimeoutMs());'
    assert source.count(call) == 1
    if not configure_transport:
        source = source.replace(call, '')  # Exact pre-fix initialization.
    preload = tmp_path / 'short-timeout.mjs'
    preload.write_text(
        f"const {{configureHttpDispatcher}} = await import({json.dumps((pi_dir / 'node_modules/@earendil-works/pi-coding-agent/dist/core/http-dispatcher.js').as_uri())});\n"
        "configureHttpDispatcher(50);\n"
    )
    request_started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def endpoint(request):
        calls.append(await request.json())
        request_started.set()
        if len(calls) == 1:
            await asyncio.sleep(1.7)  # Undici checks short deadlines on its timer tick.
        else:
            await release.wait()
        from test_qwen_pi import sse
        return web.Response(text=sse({'role': 'assistant', 'content': 'ready after admission'}),
                            content_type='text/event-stream')

    app = web.Application(); app.router.add_post('/v1/chat/completions', endpoint)
    runner = web.AppRunner(app); await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0); await site.start()
    port = site._server.sockets[0].getsockname()[1]
    for name in ('workspace', 'agent', 'sessions'):
        (tmp_path / name).mkdir()
    proc = await asyncio.create_subprocess_exec(
        'node', '--import', str(preload), '--input-type=module', '--eval', source,
        cwd=pi_dir, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, limit=2**20,
    )
    packets = []

    async def send(packet):
        proc.stdin.write((json.dumps(packet)+'\n').encode()); await proc.stdin.drain()

    async def receive(kind):
        async with asyncio.timeout(15):
            while True:
                line = await proc.stdout.readline()
                assert line, (await proc.stderr.read()).decode()
                packet = json.loads(line); packets.append(packet)
                assert packet['type'] != 'fatal', packet
                if packet['type'] == kind:
                    return packet

    try:
        await send({'type': 'init', 'workspace': str(tmp_path/'workspace'),
                    'agentDir': str(tmp_path/'agent'), 'sessionDir': str(tmp_path/'sessions'),
                    'system': 'Reply briefly.', 'baseUrl': f'http://127.0.0.1:{port}/v1',
                    'model': 'mock', 'tools': []})
        await receive('ready')
        await send({'type': 'prompt', 'text': 'Wait for admission.'})
        await receive('done')
        messages = [p['event']['message'] for p in packets if p.get('type') == 'event'
                    and p['event'].get('type') == 'message_end'
                    and p['event']['message'].get('role') == 'assistant']
        assert len(calls) == 1  # No automatic replay/retry.
        assert messages
        if not configure_transport:
            assert messages[-1]['stopReason'] == 'error'
            assert 'timed out' in messages[-1]['errorMessage'].lower()
        else:
            assert messages[-1]['stopReason'] == 'stop'
            assert messages[-1]['content'][0]['text'] == 'ready after admission'
            request_started.clear()
            await send({'type': 'prompt', 'text': 'Wait again.'})
            await asyncio.wait_for(request_started.wait(), 5)
            await send({'type': 'interrupt'})
            await receive('interrupted')  # Game deadline still cancels a waiting request.
            assert len(calls) == 2
    finally:
        release.set()
        if proc.returncode is None:
            await send({'type': 'abort'})
            proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except TimeoutError:
                proc.kill(); await proc.wait()
        await runner.cleanup()
