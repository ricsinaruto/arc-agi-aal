"""Unit tests for fixedread.right_at / read on synthetic tasks with known answers."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fixedread as F  # noqa: E402

GRID_IN, RIGHT, WRONG = [[1]], [[2]], [[3]]
TASK = {'train': [], 'test': [{'input': GRID_IN, 'output': RIGHT}]}


def cand(grid, tokens_at, verified=True):
    return {'code': f'c{tokens_at}{grid}', 'tests': [grid], 'verified': verified, 'passed': 1, 'accuracy': 1.0,
            'tokens_at': tokens_at, 't': 0}


def row(task, copy, spent, cands):
    return {'task': f'{task}#{copy}', 'spent': spent, 'candidates': cands}


# 1. one entry right at 50K: counted at a 60K budget, not at 40K
r = [row('t', 1, 80000, [cand(RIGHT, 50000)])]
assert F.right_at(TASK, r, 60000)[0] == 1 and F.right_at(TASK, r, 40000)[0] == 0
# 2. two entries, only the second right at 30K, each 100K: at 118.8K the order matters -> mean 0.5
r = [row('t', 1, 100000, [cand(WRONG, 90000)]), row('t', 2, 100000, [cand(RIGHT, 30000)])]
assert F.right_at(TASK, r, 118800)[0] == 0.5, F.right_at(TASK, r, 118800)
# 3. ... and both orders reach it at 200K
assert F.right_at(TASK, r, 200000)[0] == 1.0
# 4. pass@2: wrong verified + right unverified -> right is the second guess, counted
r = [row('t', 1, 50000, [cand(WRONG, 10000), cand(RIGHT, 20000, verified=False)])]
assert F.right_at(TASK, r, 118800)[0] == 1
# 5. spent is capped at the budget in the token average
assert F.right_at(TASK, [row('t', 1, 300000, [])], 118800)[1] == 118800
# 6. read(): paired diff, decision and tokens per try on two synthetic models over 4 tasks
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    for t in 'wxyz':
        (d / f'{t}.json').write_text(json.dumps(TASK))
    for name, rows in (('A', [row(t, c, 60000, [cand(RIGHT, 30000)]) for t in 'wxyz' for c in (1, 2)]),
                       ('B', [row(t, c, 30000, [cand(WRONG, 20000)]) for t in 'wxyz' for c in (1, 2, 3, 4)])):
        (d / name).mkdir()
        (d / name / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out = F.read(d / 'A', d / 'B', 118800, data=d)
    assert out['a'] == 4 and out['b'] == 0 and out['diff'] == 4 and out['ci95'] == (4.0, 4.0), out
    assert out['decision'] == 'A better' and out['a_tokens_per_try'] == 60000 and out['b_tokens_per_try'] == 30000
print('fixedread tests pass')
