"""Paired v37-harness read: per task, outputs right (pass@2) with candidates whose roster clock <= MARK (the matched
budget) and at the end (over-budget extra). usage: pairread.py MARK REF_REPORT_DIR MODEL_DIR [MODEL_DIR...]
Each dir holds tasks.jsonl + report (report.json or screen-report.json)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, './.claude/worktrees/arc2/dist/arc2-apiscreen-v1/src')
from arc2.loop import guesses  # noqa: E402

DATA = Path('./.claude/worktrees/arc2/dist/arc2-rtx-hard38-v37play/data/arc2/evaluation')


def load(d):
    d = Path(d)
    rows = [json.loads(x) for x in (d / 'tasks.jsonl').read_text().splitlines() if x.startswith('{')]
    rep = json.loads(next(p for p in (d / 'report.json', d / 'screen-report.json') if p.exists()).read_text())
    return {r['task'].split('#')[0]: r for r in rows}, rep


def right(row, task, limit):
    picks = guesses(row['candidates'], [t['input'] for t in task['test']], limit, 'clock')
    return sum(any(g == t['output'] for g, _ in p) for p, t in zip(picks, task['test']))


def per_task(rows, mark):
    out = {}
    for t, r in rows.items():
        task = json.loads((DATA / f'{t}.json').read_text())
        out[t] = (right(r, task, mark + 1e-6), right(r, task, None), len(task['test']))
    return out


if __name__ == '__main__':
    mark = float(sys.argv[1])
    ref_rows, ref_rep = load(sys.argv[2])
    ref = per_task(ref_rows, mark)
    n_out = sum(v[2] for v in ref.values())
    rt = ref_rep['roster']['tokens']
    print(f'Flash: {sum(v[0] for v in ref.values())}/{n_out} outputs at mark ({sum(v[0] == v[2] for v in ref.values())} tasks),'
          f' final {sum(v[1] for v in ref.values())} at {rt / 1e6:.2f}M; per Mtok {sum(v[0] for v in ref.values()) / (mark / 1e6):.2f}')
    for d in sys.argv[3:]:
        rows, rep = load(d)
        m = per_task(rows, mark)
        common = sorted(set(m) & set(ref))
        a = sum(m[t][0] for t in common)
        b = sum(ref[t][0] for t in common)
        diff = [m[t][0] - ref[t][0] for t in common]
        n = len(common)
        mean = sum(diff) / n
        sd = (sum((x - mean) ** 2 for x in diff) / (n - 1)) ** 0.5
        se = sd / n ** 0.5
        only_m = [t for t in common if m[t][0] > ref[t][0]]
        only_f = [t for t in common if ref[t][0] > m[t][0]]
        mt = rep['roster']['tokens']
        # Flash's tokens to reach the model's score at the mark (first curve point >= it)
        curve = ref_rep['roster'].get('curve', [])
        need = next((c['tokens'] for c in curve if c['score_outputs'] >= 100 * a / n_out - 1e-9), None)
        print(f'{d}: {a}/{n_out} outputs at mark ({sum(m[t][0] == m[t][2] for t in common)} tasks) vs Flash {b};'
              f' paired diff {a - b:+d} outputs (per task {mean:+.3f}, 95% CI {mean - 1.96 * se:+.3f} to {mean + 1.96 * se:+.3f});'
              f' model-only tasks {len(only_m)}, Flash-only {len(only_f)}; final {sum(m[t][1] for t in common)} at'
              f' {mt / 1e6:.2f}M (over-budget +{sum(m[t][1] - m[t][0] for t in common)}); per Mtok at mark {a / (mark / 1e6):.2f};'
              f' Flash tokens to match: {need}')
