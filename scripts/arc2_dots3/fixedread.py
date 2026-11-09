"""Fixed-attempt paired read (Dots3 vs Flash): each task has k independent one-attempt entries
(``task#copy``) per model. Right outputs at a per-task budget B = the mean over all orders of the task's entries of
pass@2 (the harness's own guesses()) on the candidates reached within B: entries run one after another, an entry's
candidate counts when the tokens spent before the entry plus its own ``tokens_at`` are <= B. Paired per task;
95% CI by a bootstrap over tasks.
usage: fixedread.py A_DIR B_DIR [--budget 118800] [--curve 30000,60000,...]   (dirs hold tasks.jsonl)"""
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, './.claude/worktrees/arc2/dist/arc2-apiscreen-v1/src')
from arc2.loop import guesses  # noqa: E402

DATA = Path('./.claude/worktrees/arc2/dist/arc2-rtx-hard38-v37play/data/arc2/evaluation')


def entries(path):
    by = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith('{'):
            r = json.loads(line)
            by.setdefault(r['task'].split('#')[0], []).append(r)
    return by


def right_at(task, rows, budget):
    """(expected right outputs, expected tokens spent) at ``budget`` over all orders of ``rows``."""
    inputs = [t['input'] for t in task['test']]
    orders = list(itertools.permutations(range(len(rows))))
    total_right = total_spent = 0.0
    for order in orders:
        before, pool = 0, []
        for i in order:
            if before >= budget:
                break
            pool += [c for c in rows[i]['candidates'] if before + c['tokens_at'] <= budget]
            before += rows[i]['spent']
        picks = guesses(pool, inputs)
        total_right += sum(any(g == t['output'] for g, _ in p) for p, t in zip(picks, task['test']))
        total_spent += min(before, budget)
    return total_right / len(orders), total_spent / len(orders)


def per_task(by, budget, tasks):
    return {t: right_at(tasks[t], by[t], budget) for t in by}


def bootstrap(diff, n=10000, seed=0):
    rng = random.Random(seed)
    keys = list(diff)
    means = sorted(sum(diff[rng.choice(keys)] for _ in keys) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def read(a_dir, b_dir, budget, data=DATA):
    a, b = entries(Path(a_dir) / 'tasks.jsonl'), entries(Path(b_dir) / 'tasks.jsonl')
    common = sorted(set(a) & set(b))
    tasks = {t: json.loads((Path(data) / f'{t}.json').read_text()) for t in common}
    ra, rb = per_task({t: a[t] for t in common}, budget, tasks), per_task({t: b[t] for t in common}, budget, tasks)
    diff = {t: ra[t][0] - rb[t][0] for t in common}
    lo, hi = bootstrap(diff)
    sa, sb = sum(v[0] for v in ra.values()), sum(v[0] for v in rb.values())
    ta, tb = sum(v[1] for v in ra.values()), sum(v[1] for v in rb.values())
    return {'budget': budget, 'tasks': len(common), 'outputs': sum(len(tasks[t]['test']) for t in common),
            'a': round(sa, 2), 'b': round(sb, 2), 'diff': round(sa - sb, 2), 'ci95': (round(lo, 2), round(hi, 2)),
            'a_per_mtok': round(sa / (ta / 1e6), 2) if ta else None, 'b_per_mtok': round(sb / (tb / 1e6), 2) if tb else None,
            'a_tokens_per_try': round(sum(r['spent'] for t in common for r in a[t]) / sum(len(a[t]) for t in common)),
            'b_tokens_per_try': round(sum(r['spent'] for t in common for r in b[t]) / sum(len(b[t]) for t in common)),
            'decision': ('A better' if lo > 0 else 'A not better' if sa - sb <= 0 else 'inconclusive')}


if __name__ == '__main__':
    args = sys.argv[1:]
    budget = float(args[args.index('--budget') + 1]) if '--budget' in args else 118800.0
    print(json.dumps(read(args[0], args[1], budget)))
    if '--curve' in args:
        for b in map(float, args[args.index('--curve') + 1].split(',')):
            r = read(args[0], args[1], b)
            print(json.dumps({k: r[k] for k in ('budget', 'a', 'b', 'diff', 'ci95')}))
