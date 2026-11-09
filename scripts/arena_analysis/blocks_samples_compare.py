"""Rebuild the Sept 30 blocks/samples comparison from recorded evidence; no cloud jobs.

Run from the repository root:
    .venv/bin/python scripts/arena_analysis/blocks_samples_compare.py --out REPORT.json
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from abc_load import Run
from hero_v2_abc_compare import family_rows

RUN_IDS = {
    'A': 'arena-hero-v2-int4-streams-v1-modal-20260929',
    'B': 'arena-hero-v2-128k-streams-v1-modal-20260929',
    'blocks': 'arena-hero-v2-blocks-v1-modal-20260929',
    'samples': 'arena-hero-v2-samples-plans-v2-modal-20260929',
}


def compare():
    result = {'denominator': 110, 'horizon_seconds': 7200, 'runs': {}}
    for label, run_id in RUN_IDS.items():
        run = Run(label, run_id)
        games = list(run.games.values())
        gp = [r for g in games for r in g.replies if r['purpose'] == 'gameplay']
        assert len(games) == 110
        assert abs(run.score_at(7200) - run.scorecard['score']) < 1e-8
        assert sum(g.levels_at(7200) for g in games) == run.arena['levels']
        # Do not silently lose model responses or count paired summary events twice.
        for g in games:
            assert sum(r['completion'] for r in g.replies) == g.row['completion_tokens'], g.id
        engine = run.arena['engine']
        model_tokens = sum(r['completion'] for g in games for r in g.replies)
        identity_path = run.dir.parents[1] / 'serving/vllm-server-identity.json'
        identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
        args = identity.get('argv', [])
        actual_seqs = int(args[args.index('--max-num-seqs') + 1]) if '--max-num-seqs' in args else None
        row = {
            'run_id': run_id,
            'marks': {str(t): {'score': run.score_at(t), 'levels': sum(g.levels_at(t) for g in games)}
                      for t in (1800, 3600, 5400, 7200)},
            'actions': sum(g.final_actions for g in games),
            'games_started': sum(bool(g.replies) for g in games),
            'games_with_clears': sum(g.levels_at(7200) > 0 for g in games),
            'clears_by_level': dict(Counter(c['level'] for g in games for c in g.clears if c['wall'] <= 7200)),
            'families': family_rows(run, 7200),
            'gameplay_replies': len(gp),
            'gameplay_completion_tokens': sum(r['completion'] for r in gp),
            'all_selected_reply_tokens': model_tokens,
            'actions_per_reply': sum(g.final_actions for g in games) / len(gp),
            'zero_action_reply_fraction': sum(not r['actions'] for r in gp) / len(gp),
            'compaction_requests': sum(r['purpose'] == 'compaction' for g in games for r in g.replies),
            'engine': engine,
            'cache_hit_fraction': engine['cached_prompt_tokens'] / (engine['cached_prompt_tokens'] + engine['computed_prompt_tokens']),
            'actual_server_max_num_seqs': actual_seqs,
            'scheduler_audit': run.arena['scheduler_audit'],
            'parallel_samples': run.arena.get('parallel_samples'),
            'action_plans': run.arena.get('action_plans'),
            'tr87_attempts': [{'game': g.id, 'score': g.final_score, 'clears': g.clears,
                              'gameplay_budget_tokens': g.row.get('budget_seconds'), 'admissions': g.events}
                             for g in games if g.family == 'tr87'],
            'evidence_sha256': {name: hashlib.sha256((run.dir / name).read_bytes()).hexdigest()
                                for name in ('arena.json', 'scorecard.json', 'public-roster.json', 'admission.jsonl')},
        }
        result['runs'][label] = row
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = compare()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    for name, row in report['runs'].items():
        print(name, row['marks']['7200'], 'actions', row['actions'], 'actual_max_seqs', row['actual_server_max_num_seqs'])
