"""Bounded serving-only probe before arena gameplay; failures stop the benchmark."""
import argparse
import asyncio
import base64
import io
import json
import time
from pathlib import Path

import httpx
from PIL import Image

from arc_infra.metrics import GpuMonitor
from arc_infra.model_templates import template_kwargs, text_only
from arc_infra.serving import MODEL
from arc_infra.serving_checks import wait_idle

from .cache import BlockBudget
from .gate import KVGate


def payload(index, words, *, model=MODEL):
    if text_only(model):
        return {
            'model': model, 'temperature': 0, 'seed': 0, 'max_tokens': 32,
            'chat_template_kwargs': template_kwargs(model, {'enable_thinking': False}),
            'messages': [{'role': 'user', 'content': f'Independent session {index}.\n'
                         + 'cell ' * words + '\nReply with the word ready.'}],
        }
    image = io.BytesIO()
    Image.new('RGB', (256, 256), (index * 9 % 256, 90, 120)).save(image, format='PNG')
    return {
        'model': model, 'temperature': 0, 'seed': 0, 'max_tokens': 32,
        'chat_template_kwargs': template_kwargs(model, {'enable_thinking': False}),
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': f'Independent session {index}.\n' + 'cell ' * words
             + '\nReply with one short sentence describing the image.'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(image.getvalue()).decode()}},
        ]}],
    }


async def _post(client, url, **kwargs):
    """client.post with transport-error retries (startup fragility, 18 Oct): same request, 1/2/4 s backoff."""
    for attempt in range(4):
        try:
            return await client.post(url, **kwargs)
        except httpx.TransportError:
            if attempt == 3:
                raise
            await asyncio.sleep(2 ** attempt)


async def run(server_dir, study, base_url):
    folder = Path(server_dir) / 'cache-probe'
    folder.mkdir()
    budget = BlockBudget.load(server_dir, study['kv_margin'])
    gate = KVGate(budget.capacity, study['generation_slots'], budget=budget)
    monitor = GpuMonitor(folder, base_url)
    record = {'state': 'running', 'requests': [], 'rounds': [], 'layout': budget.layout}
    started = time.monotonic()
    await wait_idle(base_url)
    await monitor.sample()
    sampling = asyncio.create_task(monitor.run())
    try:
        async with asyncio.timeout(270), httpx.AsyncClient(timeout=240, trust_env=False) as client:
            async def request(index, words):
                body = payload(index, words, model=study.get('served_model', MODEL))
                token_body = {k: body[k] for k in ('model', 'messages', 'chat_template_kwargs')}
                counted = await _post(client, base_url.removesuffix('/v1') + '/tokenize', json=token_body)
                counted.raise_for_status()
                count = counted.json()['count']
                reservation = await gate.acquire(str(index), count + body['max_tokens'])
                try:
                    response = await _post(client, base_url.rstrip('/') + '/chat/completions', json=body)
                    evidence = {'session': index, 'words': words, 'count': count,
                                'request': body, 'http_status': response.status_code, 'validated': False}
                    record['requests'].append(evidence)
                    try:
                        reply = response.json()
                    except ValueError:
                        evidence['raw_text'] = response.text
                        raise
                    evidence['response'] = reply
                    response.raise_for_status()
                    evidence.update(usage=reply.get('usage'),
                                    output=reply.get('choices', [{}])[0].get('message'))
                    used = reply['usage']['prompt_tokens']
                    if used != count:
                        raise RuntimeError(f'Multimodal serving-tokenizer mismatch: counted {count}, used {used}')
                    if not reply['choices'][0]['message'].get('content'):
                        raise RuntimeError('Empty serving probe response')
                    evidence['validated'] = True
                finally:
                    gate.release(reservation)

            async def batch(words):
                tasks = [asyncio.create_task(request(i, words)) for i in range(25)]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

            # Cold then warm short/longer histories. No actions or strategy prompts involved.
            for words in (2048, 8192, 16384):
                begin = time.monotonic()
                await batch(words)
                await monitor.sample()
                record['rounds'].append({'words': words, 'seconds': time.monotonic() - begin,
                                         'gate': gate.snapshot()})
                if monitor.summary().get('preemptions', 0):
                    raise RuntimeError('Block-admitted serving probe preempted; stop before gameplay')
            # Revisit earlier histories after cache pressure. CPU tier must demonstrate loads.
            await batch(8192)
            await wait_idle(base_url)
            await monitor.sample()
            summary = monitor.summary()
            if not summary.get('valid') or summary.get('errors') or summary.get('preemptions', 0):
                raise RuntimeError('Invalid telemetry or preemptions during cache probe')
            if study['cache_experiment']['cpu_cache_gib']:
                first, last = monitor.samples[0]['metrics'], monitor.samples[-1]['metrics']
                key = 'vllm:kv_offload_load_bytes_total'
                record['cpu_load_bytes'] = last.get(key, 0) - first.get(key, 0)
                if record['cpu_load_bytes'] <= 0:
                    raise RuntimeError('CPU cache configured but no round-trip load demonstrated')
            # Prevent synthetic prefixes influencing the gameplay cache working set.
            for attempt in range(10):
                reset = await _post(client, base_url.removesuffix('/v1') + '/reset_prefix_cache',
                                          params={'reset_external': 'true'})
                reset.raise_for_status()
                if reset.json().get('success'):
                    break
                await asyncio.sleep(1)
            else:
                raise RuntimeError('Unable to drain and reset probe caches before gameplay')
            record['state'] = 'complete'
    except BaseException as exc:
        record.update(state='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        sampling.cancel()
        await asyncio.gather(sampling, return_exceptions=True)
        record.update(seconds=time.monotonic() - started, gate=gate.snapshot(), engine=monitor.summary())
        (folder / 'result.json').write_text(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-dir', required=True)
    parser.add_argument('--study', required=True)
    parser.add_argument('--base-url', required=True)
    args = parser.parse_args()
    asyncio.run(run(args.server_dir, json.loads(Path(args.study).read_text()), args.base_url))


if __name__ == '__main__':
    main()
