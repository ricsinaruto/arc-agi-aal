"""Expected tok/s of the Dots3 4xH200 server by gate reply_reserve, counting vLLM preemption (recompute).
Requests: the stopped v37 Dots3 run's real (prompt, completion) pairs. Gate: held = prompt + min(cap, reserve),
admit while sum(held) <= pool and running < seqs. Decode: per-sequence rate r(B); actual KV = sum(prompt + generated);
over the pool -> the newest running request is preempted and later recomputed (prompt + generated) at the prefill
rate, which stalls decode. Two decode models: linear (65 tok/s a sequence, measured at 8.5 in flight) and saturating."""
import json, random, sys
# events.jsonl of the stopped v37 run: modal volume get arc2-apiscreen arc2-v37scr-dots3-modal-20261023/dots3/events.jsonl
EV = sys.argv[1] if len(sys.argv) > 1 else 'v37/events.jsonl'
pairs = [(e['prompt_tokens'], e['completion_tokens']) for e in map(json.loads, open(EV)) if e.get('event') == 'reply']
POOL, CAP = 1_131_347, 98_304

def run(reserve, seqs, prefill, model, seed, horizon=7200, dt=1.0):
    rnd = random.Random(seed)
    queue, running, gen, t, stall, preempts, recomputed = [], [], 0.0, 0.0, 0.0, 0, 0
    held = lambda p: p + (CAP if reserve is None else min(CAP, reserve))
    while t < horizon:
        while len(queue) < 64:
            queue.append([*rnd.choice(pairs), 0.0, False])  # prompt, completion, generated, needs recompute
        # admission (FIFO, gate on reservations)
        while queue and len(running) < seqs and sum(held(r[0]) for r in running) + held(queue[0][0]) <= POOL:
            running.append(queue.pop(0))
        if stall > 0:
            stall -= dt; t += dt; continue
        # recompute preempted requests first (vLLM resumes them when KV allows)
        for r in running:
            if r[3]:
                kv = sum(x[0] + x[2] for x in running if not x[3])
                if kv + r[0] + r[2] <= POOL:
                    stall += (r[0] + r[2]) / prefill; recomputed += r[0] + r[2]; r[3] = False
        live = [r for r in running if not r[3]]
        B = len(live)
        rate = 65.0 if model == 'linear' else 65.0 * (1 + 8.5 / 24) / (1 + B / 24)
        for r in live:
            r[2] = min(r[1], r[2] + rate * dt)
        gen += B * rate * dt
        # preemption when the actual KV exceeds the pool: newest live request
        while sum(r[0] + r[2] for r in running if not r[3]) > POOL:
            victim = [r for r in running if not r[3]][-1]; victim[3] = True; preempts += 1
        running = [r for r in running if r[2] < r[1]]
        t += dt
    return gen / horizon, preempts, recomputed / max(gen, 1)

if __name__ == '__main__':
    for model in ('linear', 'saturating'):
        for prefill in (5000, 15000):
            print(f'-- decode {model}, prefill {prefill} tok/s')
            for reserve in (None, 48 * 1024, 32768, 24576, 16384, 8192, 0):
                rs = [run(reserve, 64, prefill, model, s) for s in range(3)]
                tok = sum(r[0] for r in rs) / 3; pre = sum(r[1] for r in rs) / 3; rc = sum(r[2] for r in rs) / 3
                print(f'reserve {str(reserve):6s}: {tok:6.0f} tok/s, preemptions/2h {pre:5.1f}, recompute/gen {rc:.2f}')
