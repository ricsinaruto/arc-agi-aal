"""GDN-state precision test (throughput arm, 23 Oct; CPU). Input: arc_gdn_dump files (DIR/L<layer>/s<seq>_c<call>.pt).

Per layer and sequence, the exact per-token gated delta rule in fp32 (fla convention: v-head h reads k/q head
h // (Hv / Hk); S[h] is V x K; S <- exp(g) S; S <- S + outer(beta (v - S k), k); o = scale S q), first checked
against the kernel's dumped output o, then with the state stored after every token as
  bf16  : what the served decode kernel writes per token today
  int8r : int8 per (head, v-row) with an fp32 absmax scale over K (round to nearest)
Per variant: relative output error |o - o_fp32| / |o_fp32| per token, averaged per BIN-token bin.
Pre-registered bar (arm log 23 Oct 21:40): int8r <= 1.5x bf16 in every layer and bin, and int8r's last/first-bin
ratio <= 1.5x bf16's.
"""
import argparse
import glob
import json
import os
import re
import time

import torch

torch.set_num_threads(max(1, os.cpu_count() or 1))


def load(dirpath, seq):
    files = sorted(glob.glob(os.path.join(dirpath, f"s{seq}_c*.pt")), key=lambda f: int(re.search(r"_c(\d+)", f).group(1)))
    recs = [torch.load(f) for f in files]
    if not recs or not recs[0]["zero_init"]:
        return None
    offs = [r["offset"] for r in recs]
    if offs != sorted(offs) or any(b - a != r["tokens"] for a, b, r in zip(offs, offs[1:], recs)):
        raise ValueError(f"{dirpath} s{seq}: non-contiguous chunks {offs}")
    cat = {k: torch.cat([r[k] for r in recs]) for k in ("q", "k", "v", "g", "beta", "o")}
    cat["scale"] = recs[0].get("scale")
    return cat


def store(S, mode):
    if mode == "fp32":
        return S
    if mode == "bf16":
        return S.to(torch.bfloat16).float()
    if mode == "int8r":
        amax = S.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
        scale = amax / 127.0
        return torch.round(S / scale).clamp(-127, 127) * scale
    raise ValueError(mode)


def run(d, modes, mapping):
    q, k, v, g, beta = (d[x].float() for x in ("q", "k", "v", "g", "beta"))
    T, Hk, K = q.shape
    Hv, V = v.shape[1], v.shape[2]
    out = {}
    scale = d["scale"] or K ** -0.5
    for mode in modes:
        qh, kh = q[:, mapping], k[:, mapping]  # [T, Hv, K]
        S = torch.zeros(Hv, V, K)
        o = torch.empty(T, Hv, V)
        for t in range(T):
            S = S * torch.exp(g[t])[:, None, None]
            kt = kh[t]
            kv = torch.einsum("hvk,hk->hv", S, kt)
            delta = beta[t][:, None] * (v[t] - kv)
            S = S + delta[:, :, None] * kt[:, None, :]
            o[t] = torch.einsum("hvk,hk->hv", S, qh[t]) * scale
            S = store(S, mode)
        out[mode] = o
    return out


def rel(a, b):
    return ((a - b).norm(dim=(-1, -2)) / b.norm(dim=(-1, -2)).clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--bin", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--hv-total", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = {"rows": []}
    for layer in sorted(os.listdir(a.dump)):
        for seq in range(8):
            d = load(os.path.join(a.dump, layer), seq)
            if d is None:
                continue
            if a.max_tokens:
                d = {k: (v[:a.max_tokens] if torch.is_tensor(v) else v) for k, v in d.items()}
            t0 = time.time()
            Hk, Hv = d["q"].shape[1], d["v"].shape[1]
            maps = {"grouped": torch.arange(Hv) // (a.hv_total // Hk), "modulo": torch.arange(Hv) % Hk}
            head = {kk: (vv[:2048] if torch.is_tensor(vv) else vv) for kk, vv in d.items()}
            fit = {name: float(rel(head["o"].float(), run(head, ("fp32",), m)["fp32"]).mean()) for name, m in maps.items()}
            mapping = maps[min(fit, key=fit.get)]
            o = run(d, ("fp32", "bf16", "int8r"), mapping)
            ref = o["fp32"]
            row = {"layer": layer, "seq": seq, "tokens": int(ref.shape[0]),
                   "kernel_vs_fp32": float(rel(d["o"].float(), ref).mean()), "mapping_fit": fit, "bins": {}}
            for mode in ("bf16", "int8r"):
                e = rel(o[mode], ref)
                row["bins"][mode] = [float(e[i:i + a.bin].mean()) for i in range(0, len(e), a.bin)]
            row["seconds"] = round(time.time() - t0, 1)
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
            with open(a.out, "w") as fh:
                json.dump(report, fh, indent=1)
    ok = True
    for row in report["rows"]:
        b, i = row["bins"]["bf16"], row["bins"]["int8r"]
        row["pass"] = (all(y <= 1.5 * x for x, y in zip(b, i))
                       and (i[-1] / max(i[0], 1e-30)) <= 1.5 * (b[-1] / max(b[0], 1e-30)) and row["kernel_vs_fp32"] < 0.05)
        ok = ok and row["pass"]
    report["pass"] = ok and bool(report["rows"])
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({"pass": report["pass"]}))


if __name__ == "__main__":
    main()
