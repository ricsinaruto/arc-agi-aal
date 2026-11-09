"""Unit tests for repair_args / repair_history on the shapes of the two MiniMax v37 failures (c7f57c3e#1: the reply
ended with finish 'error' mid tool call, arguments cut; cbebaa4b#1: a tool call whose arguments the harness could not
parse), through the bundle's own tool_call() and upstream_body()."""
import json
import sys

sys.path.insert(0, './.claude/worktrees/arc2/dist/arc2-apiscreen-v1/src')
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from arc2 import api_proxy, tasks  # noqa: E402
from dots3_screen import repair_args, repair_history  # noqa: E402

CODE = 'from collections import Counter\n\ndef transform(grid):\n    H=len(grid); W=len(grid[0])\n    return grid\n'


def history(raw):
    """user -> assistant tool call (as the harness keeps it) -> tool result -> user: the next request's messages."""
    kept, code = tasks.tool_call({'tool_calls': [{'id': 'c1', 'function': {'name': 'python', 'arguments': raw}}]})
    return [{'role': 'user', 'content': 'task'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{**kept}], 'reasoning': 'r'},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'error: no code' if code is None else 'ok'},
            {'role': 'user', 'content': 'go on'}], code


def check(raw, expect_code):
    msgs, code = history(raw)
    body = api_proxy.upstream_body({'messages': msgs, 'max_tokens': 10}, 'm')
    seen = []
    fixed = repair_history(body['messages'], lambda how, r: seen.append(how))
    args = fixed[1]['tool_calls'][0]['function']['arguments']
    parsed = json.loads(args)  # strict: what the provider requires
    assert isinstance(parsed, dict), args
    assert fixed[0] == msgs[0] and fixed[2] == msgs[2] and fixed[3] == msgs[3]
    if expect_code is not None:
        assert parsed.get('code', '').startswith(expect_code[:20]), parsed
    return code, seen, parsed


cases = {
    # c7f57c3e#1 shape: finish 'error' mid-call, arguments cut inside the code string
    'truncated': (json.dumps({'code': CODE})[:60], CODE[:20]),
    # cbebaa4b#1 shape: arguments the harness could not parse; raw newlines in the JSON string (MiniMax habit)
    'raw-newlines': ('{"code": "' + CODE + '"}', CODE),
    'not-json': (CODE, CODE),
    'empty': ('', None),
    'valid': (json.dumps({'code': 'print(1)'}), 'print(1)'),
}
for name, (raw, expect) in cases.items():
    code, seen, parsed = check(raw, expect)
    print(f'{name:13s} harness code={"None" if code is None else "ok"} repaired={seen} -> keys {list(parsed)}')
assert repair_args(json.dumps({'code': 'x'}))[1] is None  # valid JSON untouched
assert repair_args({'code': 'x'})[1] == 'object'
print('all pass')


def test_api_effort():
    from dots3_screen import apply_overrides
    body = api_proxy.upstream_body({'messages': [{'role': 'user', 'content': 'x'}], 'max_tokens': 20480, 'temperature': 1.0,
                                    'top_p': 0.95, 'top_k': 40, 'chat_template_kwargs': {'reasoning_effort': 'low'}},
                                   'minimax/minimax-m3.1-flash')
    assert body['reasoning'] == {'effort': 'low'}
    out = apply_overrides(dict(body), {'provider': 'parasail', 'api_effort': 'minimal'}, lambda *a: None)
    assert out['reasoning'] == {'effort': 'minimal'} and out['provider']['order'] == ['parasail'] and 'top_k' not in out
    assert out['max_tokens'] == 20480 and out['temperature'] == 1.0
    assert apply_overrides(dict(body), {}, lambda *a: None)['reasoning'] == {'effort': 'low'}


if __name__ == '__main__':
    test_api_effort()
    print('api_effort ok')
