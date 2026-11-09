"""MFU-1 (23 Oct): bitwise repeat test and speed of fla's chunk gated delta rule forward + backward as rl1_gdn calls it
(16 key / 48 value heads, gate in kernel, initial state), per autotune restriction: ``none`` (fla's configs), ``o``
(rl1_gdn.pin_fla: chunk_fwd_kernel_o on BK/BV 64), ``w4`` (every fla autotuner limited to configs of <= 4 warps).
Prints, per mode and length, the outputs that moved over N repeats and the median fwd / bwd ms."""
import sys

import torch

import rl1_gdn

BF16 = torch.bfloat16


def autotuners():
    import fla.ops  # noqa: F401
    from triton.runtime.autotuner import Autotuner

    seen = {}
    for name, m in list(sys.modules.items()):
        if not name.startswith("fla.ops") or m is None:
            continue
        for attr, obj in list(vars(m).items()):
            o = obj
            while o is not None and not isinstance(o, Autotuner):
                o = getattr(o, "fn", None)
            if isinstance(o, Autotuner):
                seen[id(o)] = (f"{name}.{attr}", o)
    return list(seen.values())


def main(argv):
    lens = [int(x) for x in (argv[0] if argv else "8192,32768").split(",")]
    N = int(argv[1]) if len(argv) > 1 else 12
    modes = (argv[2] if len(argv) > 2 else "none,o,w4").split(",")
    rl1_gdn.FLA_PIN["on"] = False  # modes set the configs themselves
    fwd, bwd = rl1_gdn._fla()
    dev = "cuda"
    nk, nv, dk, dv = 16, 48, 128, 128
    tuners = None
    orig_cfgs = {}
    for T in lens:
        torch.manual_seed(0)
        q = torch.nn.functional.normalize(torch.randn(1, T, nk, dk, device=dev), dim=-1).to(BF16)
        k = torch.nn.functional.normalize(torch.randn(1, T, nk, dk, device=dev), dim=-1).to(BF16)
        v = torch.randn(1, T, nv, dv, device=dev).to(BF16)
        beta = torch.sigmoid(torch.randn(1, T, nv, device=dev)).to(BF16)
        ga = torch.randn(1, T, nv, device=dev).to(BF16)
        a_log = torch.randn(nv, device=dev) * 0.5
        dt_bias = torch.randn(nv, device=dev) * 0.5
        h0 = torch.randn(1, nv, dk, dv, device=dev) * 0.1
        do = torch.randn(1, T, nv, dv, device=dev).to(BF16) * 0.01

        def step():
            g, o, A, final, _, _ = fwd(q=q, k=k, v=v, g=ga, beta=beta, scale=dk ** -0.5, initial_state=h0,
                                       output_final_state=True, use_gate_in_kernel=True, A_log=a_log, dt_bias=dt_bias)
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            grads = bwd(q=q, k=k, v=v, g=g, beta=beta, A=A, scale=dk ** -0.5, initial_state=h0, do=do, dht=None,
                        use_gate_in_kernel=True, g_input=ga, A_log=a_log, dt_bias=dt_bias)
            outs = {"o": o, "final": final}
            for i, x in enumerate(grads):
                if torch.is_tensor(x):
                    outs[f"grad{i}"] = x
            return {n: x.detach().clone() for n, x in outs.items()}, e

        for mode in modes:
            if tuners is None:
                step()
                tuners = autotuners()
                orig_cfgs = {name: list(at.configs) for name, at in tuners}
                print("fla autotuners:", {n: len(c) for n, c in orig_cfgs.items()}, flush=True)
            for name, at in tuners:
                at.configs = list(orig_cfgs[name])
                at.cache.clear()
            rl1_gdn.FLA_PIN["done"] = False
            if mode == "o":
                rl1_gdn.pin_fla(force=True)
            elif mode == "w4":
                for name, at in tuners:
                    keep = [c for c in at.configs if c.num_warps <= 4]
                    if keep:
                        at.configs = keep
            step()  # autotune here
            torch.cuda.synchronize()
            first, _ = step()
            moved, tf, tb = {}, [], []
            for _ in range(N):
                s = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                s.record()
                outs, mid = step()
                end.record()
                torch.cuda.synchronize()
                tf.append(s.elapsed_time(mid))
                tb.append(mid.elapsed_time(end))
                for n, x in outs.items():
                    if not torch.equal(x, first[n]):
                        d = float((x.float() - first[n].float()).abs().max())
                        moved.setdefault(n, []).append(round(d, 5))
            tf.sort()
            tb.sort()
            print(f"T {T} mode {mode}: moved {({n: (len(d), max(d)) for n, d in moved.items()}) or 'none'} of {N} "
                  f"repeats; fwd {tf[len(tf) // 2]:.2f} ms bwd {tb[len(tb) // 2]:.2f} ms (incl. clones)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
