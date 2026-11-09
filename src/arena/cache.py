"""Conservative hybrid cache admission; reported equivalent tokens are not a pool."""
import bisect
import json
from pathlib import Path


class BlockBudget:
    def __init__(self, layout, margin=0.9):
        if layout.get('schema') != 1 or not 0 < margin <= 1:
            raise ValueError('Invalid cache layout or margin')
        self.layout = layout
        self.capacity = int(layout['num_blocks'] * margin)
        self.rows = layout['costs']
        self.lengths = [r['tokens'] for r in self.rows]
        costs = [r['blocks'] for r in self.rows]
        if (self.capacity <= 0 or not costs or self.lengths != sorted(set(self.lengths))
                or costs != sorted(costs) or min(costs) <= 0
                or self.lengths[-1] != layout['max_model_len']):
            raise ValueError('Invalid or non-monotone cache cost table')
        if any(sum(r['groups']) != r['blocks'] for r in self.rows):
            raise ValueError('Cache group totals disagree')

    def cost(self, tokens):
        if tokens < 0:
            raise ValueError('Negative request length')
        length = tokens + self.layout['lookahead']
        index = bisect.bisect_left(self.lengths, length)
        if index == len(self.rows):
            raise ValueError('Request plus speculative lookahead exceeds measured cache layout')
        return self.rows[index]['blocks']

    @classmethod
    def load(cls, server_dir, margin=0.9):
        layouts = []
        for p in sorted(Path(server_dir).glob('*.log')):
            for line in p.read_text(errors='replace').splitlines():
                if 'ARC_CACHE_LAYOUT ' in line:
                    layouts.append(json.JSONDecoder().raw_decode(line.split('ARC_CACHE_LAYOUT ', 1)[1])[0])
        if not layouts:
            raise RuntimeError('Missing resolved cache layout; refusing token-pool fallback')
        # Worker and scheduler emissions must agree about costs and pool size.
        def signature(x):
            return (x['num_blocks'], x['costs'], x['max_model_len'], x['lookahead'],
                    x.get('planned_cache_bytes'))
        if any(signature(x) != signature(layouts[0]) for x in layouts[1:]):
            raise RuntimeError('Worker/scheduler cache layouts disagree')
        return cls(layouts[-1], margin)
