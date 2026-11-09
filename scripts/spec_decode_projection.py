"""Model projection of MTP-1/2 from the measured MTP-0 step fit (not a measurement; spec-decode.md section 4)."""
import json
import sys

bench = json.load(open(sys.argv[1]))
pts = {p["streams"]: p["step_ms"] for p in bench["spec_sweep"]["points"]}
ref = next(p["generated_tok_s"] for p in bench["spec_sweep"]["points"] if p["streams"] == 37)
xs = list(pts)
ys = [pts[x] for x in xs]
n = len(xs)
mx, my = sum(xs) / n, sum(ys) / n
b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
a = my - b * mx
ACC = {1: [0.755], 2: [0.755, 0.577]}  # arena per-position acceptance (draft-head-finetune.md, MTP-3 arena hours)
TAX = {93184: 0.14, 70000: 0.2255, 49000: 0.354}
DRAFT_MS = 2.0  # per draft step (one MTP layer + 16K-vocab head), assumed
PREFILL, KEEP = 0.072, 0.99


def useful(k, streams, window, c):
    step = a + b * streams + c * streams * k + DRAFT_MS * k
    tps = streams * (1 + sum(ACC[k])) / step * 1000
    dec = tps / ref
    eng = 1 / ((1 - PREFILL) / dec + PREFILL)
    return step, tps, dec, eng * (1 - TAX[window]) / (1 - 0.14) * KEEP


rows = []
for k, s, w in ((1, 31, 93184), (1, 37, 70000), (2, 27, 93184), (2, 37, 49000)):
    for c, label in ((b, "verify token = decode token"), (0.6 * b, "verify token = 0.6 x decode token")):
        step, tps, dec, use = useful(k, s, w, c)
        rows.append({"mtp": k, "streams": s, "window": w, "assumption": label, "step_ms": round(step, 1),
                     "tok_s": round(tps), "decode_ratio": round(dec, 3), "useful_ratio": round(use, 3)})
breakeven = {}
for k, s, w in ((1, 31, 93184), (2, 27, 93184)):
    c = b
    while c > 0 and useful(k, s, w, c)[3] < 1.15:
        c -= 0.001
    breakeven[f"mtp{k}_{s}x{w}"] = round(c, 3)
out = {"mtp0_step_fit": {"intercept_ms": round(a, 2), "ms_per_stream": round(b, 4)}, "mtp0_37_tok_s": ref,
       "assumed_acceptance": ACC, "draft_ms_per_step": DRAFT_MS, "projection": rows,
       "breakeven_verify_ms_per_extra_token_for_1.15": breakeven}
print(json.dumps(out, indent=1))
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(json.dumps(out, indent=1) + "\n")
