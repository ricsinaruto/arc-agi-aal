"""CPU test of the Dots3 admission change: the bundle's KVGate with the Transport's held = prompt + min(cap, reserve),
fed the v37 Dots3 run's real (prompt, completion) pairs. Reports admitted in flight and the chance the admitted set's
actual KV (prompt + completion) exceeds the pool (vLLM would preempt)."""
import asyncio, json, random, sys
sys.path.insert(0, './.claude/worktrees/arc2/dist/arc2-apiscreen-v1/src')
from arena.gate import KVGate
ev = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else 'v37/events.jsonl') if l.startswith('{')]
pairs = [(e['prompt_tokens'], e['completion_tokens']) for e in ev if e['event'] == 'reply']
POOL, CAP = 1_131_347, 98_304

async def admitted(reserve, slots, sample):
    gate = KVGate(POOL, slots=slots)
    n = 0
    for p, _ in sample:
        held = p + (CAP if reserve is None else min(CAP, reserve))
        if not gate._fits(min(held, POOL)):
            break
        await gate.acquire(f't{n}', held)
        n += 1
    return n

async def main():
    random.seed(0)
    for reserve in (None, 32768, 16384, 8192):
        ns, over = [], 0
        for _ in range(400):
            s = random.sample(pairs, len(pairs))
            n = await admitted(reserve, 64, s)
            ns.append(n)
            over += sum(p + c for p, c in s[:n]) > POOL
        print(f'reserve {reserve}: in flight mean {sum(ns)/len(ns):.1f} (min {min(ns)}, max {max(ns)}); '
              f'actual KV > pool in {100*over/len(ns):.0f}% of draws')
    assert (await admitted(16384, 64, [(20000, 0)] * 100)) == 31  # 1,131,347 // 36,384
    assert (await admitted(None, 64, [(20000, 0)] * 100)) == 9    # the run's regime: 1,131,347 // 118,304
    print('gate asserts pass')
asyncio.run(main())
