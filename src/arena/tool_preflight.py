"""Small native tool-parser check before gameplay, with no environment actions."""

import ast
import json
import time

import httpx

from arc_infra.model_templates import template_kwargs


async def check(config, output):
    body = {
        'model': config.model, 'temperature': 0, 'max_tokens': 512,
        'chat_template_kwargs': template_kwargs(config.model, {'enable_thinking': False}),
        'messages': [{'role': 'user', 'content': 'Call the python tool once with code print(7). Do not answer in prose.'}],
        'tools': [{'type': 'function', 'function': {'name': 'python', 'description': 'Run Python code.',
                   'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}}, 'required': ['code']}}}],
        'tool_choice': 'auto',
    }
    record = {'request': body, 'status': 'running', 'actions_executed': 0}
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(base_url=config.base_url.rstrip('/') + '/', timeout=180, trust_env=False) as client:
            count = await client.post(client.base_url.copy_with(path='/tokenize'), json={
                **{k: v for k, v in body.items() if k in ('model', 'messages', 'tools', 'chat_template_kwargs')},
                'add_generation_prompt': True})
            count.raise_for_status()
            record['tokenization'] = count.json()
            response = await client.post('chat/completions', json=body)
            record['http_status'] = response.status_code
            record['response'] = response.json()
            response.raise_for_status()
        result = record['response']
        if result['usage']['prompt_tokens'] != record['tokenization']['count']:
            raise RuntimeError('Native tool request tokenization differs from generation')
        calls = result['choices'][0]['message'].get('tool_calls', [])
        if len(calls) != 1 or calls[0]['function']['name'] != 'python':
            raise RuntimeError('Native tool parser did not return the requested Python call')
        args = json.loads(calls[0]['function']['arguments'])
        if (set(args) != {'code'} or not isinstance(args['code'], str)
                or ast.dump(ast.parse(args['code'])) != ast.dump(ast.parse('print(7)'))):
            raise RuntimeError('Native tool handoff changed the requested arguments')
        record['status'] = 'passed'
        return record
    except BaseException as exc:
        record.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['seconds'] = time.monotonic() - started
        output.write_text(json.dumps(record, indent=2))
