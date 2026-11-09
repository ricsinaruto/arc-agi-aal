"""tickread.py LABEL=DIR[:lo:hi minutes]...: DIR holds telemetry.jsonl (and optionally vllm-metrics.txt, server-tail.log).
Per run over the window: tok/s, engine time split (base step, request starts, computed tokens), admission wait, and
from the cumulative histograms slow iterations per request and the queue-time shares."""
import json, re, sys, statistics as st
from pathlib import Path
def solve(A, B):
    n = len(B); M = [r[:] + [b] for r, b in zip(A, B)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i])); M[i], M[p] = M[p], M[i]
        for r in range(n):
            if r != i:
                f = M[r][i] / M[i][i]; M[r] = [x - f * y for x, y in zip(M[r], M[i])]
    return [M[i][n] / M[i][i] for i in range(n)]
for arg in sys.argv[1:]:
    label, rest = arg.split("=", 1); parts = rest.split(":"); d = Path(parts[0])
    lo, hi = (float(parts[1]) * 60, float(parts[2]) * 60) if len(parts) > 2 else (900, 2400)
    rows = [json.loads(l) for l in open(d / "telemetry.jsonl")]
    rows = [r for r in rows if lo <= r["elapsed_seconds"] <= hi and r.get("metrics")]
    if len(rows) < 12: print(label, "too few telemetry rows", len(rows)); continue
    a, b = rows[0]["metrics"], rows[-1]["metrics"]; dt = rows[-1]["elapsed_seconds"] - rows[0]["elapsed_seconds"]
    D = lambda k: (b.get(k) or 0) - (a.get(k) or 0)
    req, it = D("vllm:request_success_total"), D("vllm:iteration_tokens_total_count")
    run = st.mean(r["metrics"]["vllm:num_requests_running"] for r in rows)
    X, Y = [], []
    for p, q in zip(rows[::3], rows[3::3]):
        e = lambda k: (q["metrics"].get(k) or 0) - (p["metrics"].get(k) or 0)
        if e("vllm:iteration_tokens_total_count") > 0 and q["elapsed_seconds"] - p["elapsed_seconds"] < 40:
            X.append((e("vllm:iteration_tokens_total_count"), e("vllm:time_to_first_token_seconds_count"), e("vllm:prompt_tokens_by_source_total:local_compute")))
            Y.append(1000 * (q["elapsed_seconds"] - p["elapsed_seconds"]))
    s = solve([[sum(x[i] * x[j] for x in X) for j in range(3)] for i in range(3)], [sum(x[i] * y for x, y in zip(X, Y)) for i in range(3)])
    tot = [sum(x[i] for x in X) for i in range(3)]; T = sum(Y)
    acc = D("vllm:spec_decode_num_accepted_tokens_total") / max(1, D("vllm:spec_decode_num_drafts_total"))
    print(f"{label} min {lo / 60:.0f}-{hi / 60:.0f}: gen {D('vllm:generation_tokens_total') / dt:.0f} tok/s, running {run:.1f}, {1000 * dt / it:.1f} ms/iteration, requests {req / dt * 3600:.0f}/h (reply {D('vllm:generation_tokens_total') / max(1, req):.0f}, computed {D('vllm:prompt_tokens_by_source_total:local_compute') / max(1, req):.0f} per request), acceptance {acc:.3f}")
    print(f"    engine time: base {s[0]:.1f} ms/step ({100 * s[0] * tot[0] / T:.0f}%), request start {s[1]:.0f} ms ({100 * s[1] * tot[1] / T:.0f}%), computed token {1000 * s[2]:.0f} us ({100 * s[2] * tot[2] / T:.0f}%); decode capacity {run * (1 + acc) / s[0] * 1000:.0f} tok/s"
          f" | admission wait {D('vllm:request_queue_time_seconds_sum') / max(1, D('vllm:request_queue_time_seconds_count')):.2f} s per request = {100 * D('vllm:request_queue_time_seconds_sum') / dt / run:.2f}% of stream time; preemptions {b.get('vllm:num_preemptions_total', 0):.0f}")
    m = d / "vllm-metrics.txt"
    if m.is_file():
        t = m.read_text()
        H = lambda name: [(float(x) if x != "+Inf" else float("inf"), float(y)) for x, y in re.findall(re.escape(name) + r'_bucket\{[^}]*le="([^"]+)"[^}]*\} ([0-9.e+]+)', t)]
        V = lambda name: float(re.search(re.escape(name) + r'\{[^}]*\} ([0-9.e+]+)', t)[1])
        h = dict(H("vllm:inter_token_latency_seconds")); q = dict(H("vllm:request_queue_time_seconds")); n = V("vllm:request_prefill_time_seconds_count")
        slow = h[0.4] - h[0.15]; running = V("vllm:num_requests_running")
        print(f"    since start ({n:.0f} requests): slow token intervals (0.15-0.4 s) {slow:.0f} = {slow / max(1, running) / n:.2f} slow iterations per request at {running:.0f} running; queue time under 0.3 s {100 * q[0.3] / n:.0f}%, under 2.5 s {100 * q[2.5] / n:.0f}%, mean {V('vllm:request_queue_time_seconds_sum') / n:.2f} s")
    log = d / "server-tail.log"
    if log.is_file():
        hits = re.findall(r"ARC_ADMIT_TICK_STATS (\{[^}]*\})", log.read_text(errors="replace"))
        if hits:
            k = json.loads(hits[-1]); print(f"    tick {k['tick_seconds']} s: {k['admitted']} admitted in {k['openings']} steps = {k['admitted'] / k['openings']:.2f} per step, largest {k['max_group']}, held steps {k['held_steps']}, rode a prefill step {k['rides']}")
        else: print("    no ARC_ADMIT_TICK_STATS line in the log tail")
