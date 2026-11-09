"""GDN deferred commit M2a test (throughput arm, 19 Oct), run beside a server built with --gdn-defer check.

Part 1, layout simulation (real-mode pointers on a pool, against the served fused_gdn_decode_post_conv_mtp on a
twin pool): 8 requests x 400 steps, block size 64 so crossings are frequent; each step random draft counts and
acceptance; at every crossing the align-mode snapshot copy (state at the boundary column -> snapshot slot) and the
running-block move (accepted column -> new block, fresh block ids with stale flags) run on both pools and are
compared bit for bit; random mixed steps (flush, then the pinned kernel on the deferred pool) and re-prefills
(new state, flags cleared). Every step's outputs are compared bit for bit.

Part 2, served check mode: requests that cross real 3,136-token blocks mid-decode, staggered arrivals (mixed steps),
and a follow-up request that resumes from a snapshot stored mid-speculation; then the server's stats file
(ARC_GDN_DEFER_STATS): every compare counter must be 0 and the replay / near / flush counters non-zero.
"""

import argparse
import ctypes
import json
import os
import random
import threading
import time
import traceback
import urllib.request
from pathlib import Path

import torch

RESULT = {"phases": []}


def phase(name, **kw):
    rec = {"phase": name, **kw}
    RESULT["phases"].append(rec)
    print(json.dumps(rec), flush=True)
    Path(ARGS.out).write_text(json.dumps(RESULT, indent=1))


class Layer:
    """Stand-in for the GDN layer attributes the hook reads."""

    def __init__(self, g, HV, sigmoid):
        self.A_log = torch.log(torch.rand(HV, generator=g, device="cuda") * 15 + 1).float()
        self.dt_bias = (torch.randn(HV, generator=g, device="cuda") * 0.5).float()
        class N:  # noqa: D401
            pass
        self.norm = N()
        self.norm.weight = (1 + 0.1 * torch.randn(128, generator=g, device="cuda")).float()
        self.norm.activation = "sigmoid" if sigmoid else "silu"
        self.head_k_dim = 128
        self.layer_norm_epsilon = 1e-6


def simulate(D, ops, seed, steps=400, n=8, W=4, block=64, H=16, HV=48, sigmoid=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rnd = random.Random(seed)
    L = Layer(g, HV, sigmoid)
    slots = 1 + n * 64
    R = torch.zeros(slots, HV, 128, 128, device="cuda").bfloat16()
    for i in range(1, slots):
        R[i].normal_(0, 0.05, generator=g)
    P = R.clone()  # deferred pool
    Q = R.clone()  # upstream kernel rebuilt in the same library (mode 2): tells library codegen from layout bugs
    flags = torch.randint(0, 50, (slots * HV * 2,), dtype=torch.int32, device="cuda")  # stale garbage
    free = list(range(1, slots))
    rnd.shuffle(free)
    blocks = [[free.pop() for _ in range(W)] for _ in range(n)]
    for r in range(n):  # every request starts with a prefill, which clears its running block's flags
        flags.view(-1, HV, 2)[blocks[r][0]] = 0
    pos = [rnd.randrange(40, 200) for _ in range(n)]  # tokens already in each request
    acc = [1] * n
    cnt = {"steps": 0, "out_neq": 0, "snap_neq": 0, "move_neq": 0, "crossings": 0, "mixed": 0, "prefills": 0,
           "near_steps": 0, "final_neq": 0, "rebuild_neq": 0, "rebuild_final_neq": 0}
    for step in range(steps):
        T = [rnd.choice([W, W, W, max(1, W - 1), 1]) for _ in range(n)]
        cu = torch.tensor([0] + list(torch.tensor(T).cumsum(0).tolist()), dtype=torch.int32, device="cuda")
        Ltok = int(cu[-1])
        mixed = torch.randn(Ltok, 2 * H * 128 + HV * 128, generator=g, device="cuda").bfloat16()
        a = torch.randn(Ltok, HV, generator=g, device="cuda").bfloat16()
        b = torch.randn(Ltok, HV, generator=g, device="cuda").bfloat16()
        gate = torch.randn(Ltok, HV, 128, generator=g, device="cuda").bfloat16()
        si = torch.tensor(blocks, dtype=torch.int32, device="cuda").contiguous()
        na = torch.tensor(acc, dtype=torch.int32, device="cuda")
        seq = torch.tensor([p + t for p, t in zip(pos, T)], dtype=torch.int32, device="cuda")
        o_ref = torch.empty(Ltok, HV, 128, device="cuda").bfloat16()
        o_def = torch.empty_like(o_ref)
        o_reb = torch.empty_like(o_ref)
        D.launch(2, L, mixed, a, b, si, cu, na, Q, gate, o_reb, n, D.Ptrs())
        is_mixed = rnd.random() < 0.12
        ops.fused_gdn_decode_post_conv_mtp(
            mixed_qkv=mixed, a=a, b=b, A_log=L.A_log, dt_bias=L.dt_bias, state_indices=si, cu_seqlens=cu,
            num_accepted_tokens=na, state=R, output_gate=gate, norm_weight=L.norm.weight, out=o_ref,
            scale=128 ** -0.5, norm_eps=1e-6, output_gate_activation=L.norm.activation)
        st = {"flags": flags}
        if is_mixed:
            cnt["mixed"] += 1
            D.launch(1, L, None, None, None, si, cu, na, P, None, None, n, D.real_ptrs(st, P, si, seq, block))
            ops.fused_gdn_decode_post_conv_mtp(
                mixed_qkv=mixed, a=a, b=b, A_log=L.A_log, dt_bias=L.dt_bias, state_indices=si, cu_seqlens=cu,
                num_accepted_tokens=na, state=P, output_gate=gate, norm_weight=L.norm.weight, out=o_def,
                scale=128 ** -0.5, norm_eps=1e-6, output_gate_activation=L.norm.activation)
        else:
            D.launch(0, L, mixed, a, b, si, cu, na, P, gate, o_def, n, D.real_ptrs(st, P, si, seq, block))
        torch.cuda.synchronize()
        cnt["out_neq"] += int((o_ref.view(torch.int16) != o_def.view(torch.int16)).sum())
        cnt["rebuild_neq"] += int((o_ref.view(torch.int16) != o_reb.view(torch.int16)).sum())
        cnt["steps"] += 1
        new_acc = [rnd.randint(1, t) for t in T]
        for r in range(n):
            p0 = pos[r]
            lo, hi = max(p0 - 2 * W - 2, 0), p0 + 2 * W + 2
            cnt["near_steps"] += int(hi // block != lo // block)
            new_pos = p0 + new_acc[r]
            # align-mode crossing: a boundary k*block with p0 < k*block <= new_pos -> snapshot of the state after
            # token (k*block - p0 - 1), then the running state moves to fresh blocks.
            kb = (new_pos // block) * block
            if kb > p0:
                cnt["crossings"] += 1
                col = kb - p0 - 1
                snap_r, snap_p = R[blocks[r][col]].clone(), P[blocks[r][col]].clone()
                cnt["snap_neq"] += int((snap_r.view(torch.int16) != snap_p.view(torch.int16)).sum())
                newb = [free.pop() if free else rnd.randrange(1, slots) for _ in range(W)]
                R[newb[0]] = R[blocks[r][new_acc[r] - 1]]
                P[newb[0]] = P[blocks[r][new_acc[r] - 1]]
                Q[newb[0]] = Q[blocks[r][new_acc[r] - 1]]
                cnt["move_neq"] += int((R[newb[0]].view(torch.int16) != P[newb[0]].view(torch.int16)).sum())
                free.extend(blocks[r][1:])
                blocks[r] = newb
                new_acc[r] = 1
            pos[r] = new_pos
        if rnd.random() < 0.03:  # re-prefill one request: fresh state in col 0, flags cleared, acc 1
            r = rnd.randrange(n)
            cnt["prefills"] += 1
            fresh = torch.randn(HV, 128, 128, generator=g, device="cuda").mul_(0.05).bfloat16()
            R[blocks[r][0]] = fresh
            P[blocks[r][0]] = fresh
            Q[blocks[r][0]] = fresh
            flags.view(-1, HV, 2)[blocks[r][0]] = 0
            new_acc[r] = 1
        acc = new_acc
    # final flush: materialize every deferred request and compare its accepted column
    si = torch.tensor(blocks, dtype=torch.int32, device="cuda").contiguous()
    na = torch.tensor(acc, dtype=torch.int32, device="cuda")
    T = [W] * n
    cu = torch.tensor([0] + [W * (i + 1) for i in range(n)], dtype=torch.int32, device="cuda")
    seq = torch.tensor([p + W for p in pos], dtype=torch.int32, device="cuda")
    D.launch(1, L, None, None, None, si, cu, na, P, None, None, n, D.real_ptrs({"flags": flags}, P, si, seq, block))
    torch.cuda.synchronize()
    for r in range(n):
        c = blocks[r][acc[r] - 1]
        cnt["final_neq"] += int((R[c].view(torch.int16) != P[c].view(torch.int16)).sum())
        cnt["rebuild_final_neq"] += int((R[c].view(torch.int16) != Q[c].view(torch.int16)).sum())
    return cnt


WORDS = ("grid cell color move left right up down object shape pattern level score agent action reset test frame "
         "value python print list count rows columns block wall goal path state change rule check maybe next").split()


def post(url, body, timeout=1200):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def tokenize(root, model, text):
    return post(root + "/tokenize", {"model": model, "prompt": text})["tokens"]


def served_check(stats_path, block=3136):
    base = os.environ["ARC_BENCH_BASE_URL"].rstrip("/")
    root = base.removesuffix("/v1")
    model = os.environ["ARC_BENCH_MODEL"]
    rnd = random.Random(5)
    text = " ".join(rnd.choice(WORDS) for _ in range(30000))
    toks = tokenize(root, model, text)
    out = {}
    errors = []
    results = {}

    def run(i, prompt_ids, n_out, delay):
        time.sleep(delay)
        try:
            r = post(base + "/completions", {"model": model, "prompt": prompt_ids, "max_tokens": n_out,
                                             "min_tokens": n_out, "ignore_eos": True, "temperature": 0.7})
            results[i] = r["choices"][0]["text"]
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e)[:200])

    # 12 requests whose decode crosses a 3,136-token boundary within the first ~40 tokens; staggered arrivals
    jobs = []
    for i in range(12):
        k = 1 + i % 4
        start = (i * 997) % 5000
        length = k * block - 3 - 3 * i
        jobs.append(threading.Thread(target=run, args=(i, toks[start:start + length], 400, 1.5 * i), daemon=True))
    for j in jobs:
        j.start()
    for j in jobs:
        j.join(timeout=1500)
    out["wave1_errors"] = errors[:3]
    # prefix store mid-speculation: request A crosses a boundary while decoding; B resumes from that snapshot
    a_ids = toks[7000:7000 + 2 * block - 10]
    try:
        ra = post(base + "/completions", {"model": model, "prompt": a_ids, "max_tokens": 200, "min_tokens": 200,
                                          "ignore_eos": True, "temperature": 0.7, "return_token_ids": True})
        a_out = ra["choices"][0].get("token_ids") or tokenize(root, model, ra["choices"][0]["text"])
        b_ids = a_ids + list(a_out)[:150] + toks[20000:20300]
        rb = post(base + "/completions", {"model": model, "prompt": b_ids, "max_tokens": 200, "min_tokens": 200,
                                          "ignore_eos": True, "temperature": 0.7})
        out["prefix_b_cached"] = (rb.get("usage") or {}).get("prompt_tokens_details")
    except Exception as e:  # noqa: BLE001
        out["prefix_error"] = repr(e)[:300]
    time.sleep(25)
    # one more mixed step so the hook writes its stats
    try:
        post(base + "/completions", {"model": model, "prompt": toks[100:400], "max_tokens": 4})
    except Exception as e:  # noqa: BLE001
        out["poke_error"] = repr(e)[:200]
    time.sleep(25)
    try:
        post(base + "/completions", {"model": model, "prompt": toks[500:900], "max_tokens": 4})
    except Exception:  # noqa: BLE001
        pass
    time.sleep(3)
    p = Path(stats_path)
    out["stats"] = json.loads(p.read_text()) if p.exists() else None
    return out


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default="/tmp/arena/trials/duck-transfer/gdn-defer-stats.json")
    ARGS = ap.parse_args()
    from vllm import _custom_ops as ops
    import vllm.arc_gdn_deferred as D
    phase("start", mode=D.MODE, lib=D.LIB_PATH, device=torch.cuda.get_device_name())
    for seed, sigmoid in ((1, True), (2, False), (3, True)):
        try:
            phase("simulate", seed=seed, sigmoid=sigmoid, **simulate(D, ops, seed, sigmoid=sigmoid))
        except Exception:  # noqa: BLE001
            phase("simulate", seed=seed, error=traceback.format_exc()[-2000:])
    try:
        phase("served", **served_check(ARGS.stats))
    except Exception:  # noqa: BLE001
        phase("served", error=traceback.format_exc()[-2000:])
    phase("done")


if __name__ == "__main__":
    main()
