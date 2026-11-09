"""Analytic model FLOPs of a trunk/sft step (Flash t75 student, cut to 128 experts) and MFU.
Usage: python3 mfu.py events.jsonl [peak_flops]   (step rows need prefix/window tokens, pairs, reply, mtp rows)."""
import json
import sys

H, V = 2560, 248320
GDN = 10240 * H + 6144 * H + 2 * 48 * H + H * 6144          # in_proj_qkv, z, a/b, out
ATT = 12288 * H + 2 * 512 * H + H * 6144 + 640 * H          # q(+gate), k, v, o, indexer qk
MOE = 10 * (1280 * H + H * 640) + 3 * 640 * H + 128 * H     # 10 routed (gate_up, down), shared, router
HYP = 2 * (320 * 10240 + 10240 * 320 + 4 * 10240)           # attn + mlp hyper-connections
PLE = 10240 * H + H * H                                     # layer 1 key/value proj
TRUNK = 33 * GDN + 11 * ATT + 44 * (MOE + HYP) + PLE + (320 * 10240 + 10240 * 320)  # + final mixer
MTP = 2 * H * H + ATT + (10 * (1280 * H + H * 640) + 3 * 640 * H + 512 * H) + HYP + (320 * 10240 + 10240 * 320)
HEAD = H * V
SEL = 2048                                                  # selected keys per query (indexer_budget)
ATT_Q = 2 * 2 * 24 * 256 * SEL                              # QK + PV per query per full layer
IDX_PAIR = 2 * 4 * 128 / 4                                  # indexer per (query, key) per layer (compressed by 4)
GDN_T = 33 * 48 * (4 * 128 * 128 + 2 * 64 * 128 * 2)        # delta-rule state + intra-chunk per token


def step_flops(r):
    pre, win, reply, rows = r["prefix_tokens"], r["window_tokens"], r["reply_tokens"], r["mtp_rows"]
    fwd_tok = pre + 3 * win                                 # prefix forward only; window fwd + bwd (2x)
    f = 2 * TRUNK * fwd_tok + 2 * MTP * fwd_tok
    f += 2 * HEAD * 3 * reply                               # trunk lm_head on reply rows (CE / KL)
    f += 2 * HEAD * rows * (1 + 3)                          # MTP KL: trunk logits (fwd) + draft logits (fwd + bwd)
    f += (11 + 1) * ATT_Q * fwd_tok + GDN_T * fwd_tok
    f += 11 * IDX_PAIR * (r["prefix_pairs"] + 3 * r["window_pairs"])
    return f


if __name__ == "__main__":
    peak = float(sys.argv[2]) if len(sys.argv) > 2 else 2.25e15
    rows = [json.loads(x) for x in open(sys.argv[1]) if '"event": "step"' in x]
    print("params/token: trunk %.2fB  mtp %.2fB  head %.2fB" % (TRUNK / 1e9, MTP / 1e9, HEAD / 1e9))
    for r in rows:
        if "prefix_tokens" not in r:
            continue
        f = step_flops(r)
        print(r["step"], "PFLOP %.1f" % (f / 1e15), "s %.1f" % r["seconds"], "TFLOP/s %.1f" % (f / r["seconds"] / 1e12),
              "MFU %.2f%%" % (100 * f / r["seconds"] / peak))
