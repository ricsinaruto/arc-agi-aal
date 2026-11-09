"""Optimizer states in pinned host memory, paged through the card one parameter at a time in the step.

torchao's AdamW (8- or 4-bit) keeps its states on the card: 66 GiB for the t75 student at 8 bits, which with the BF16
gradients and the real window does not fit (util-v2b). ``Pager`` keeps every large state tensor (the subclass's
codes / scale, or a plain state) on the host and runs the same compiled ``single_param_adam`` per parameter on
card copies: the next ``depth`` parameters' states are copied in on a side stream while one steps, and each is copied
back once its step is queued. Same kernel, same inputs, same RNG order: the update is bit-equal to the on-card
optimizer (``check``). torchao's CPUOffloadOptimizer steps on the CPU instead (its own RNG for stochastic rounding,
host copies of the weights and gradients, a gradient copy after every backward), so it cannot give the same bytes.
"""
import collections
import contextlib
import os
import time

import torch

MIN_NUMEL = 1 << 16  # smaller tensors (qmaps, small parameters' plain states) stay on the card


def _map(v, f):
    """``v`` with ``f`` applied to its large tensors (a state subclass's inner tensors, or a plain tensor)."""
    if not isinstance(v, torch.Tensor):
        return v
    if type(v) is not torch.Tensor and hasattr(v, "__tensor_flatten__"):
        names, attrs = v.__tensor_flatten__()
        return type(v).__tensor_unflatten__({n: f(getattr(v, n)) for n in names}, attrs, None, None)
    return f(v)


def _leaves(v):
    if type(v) is not torch.Tensor and hasattr(v, "__tensor_flatten__"):
        return [getattr(v, n) for n in v.__tensor_flatten__()[0]]
    return [v] if isinstance(v, torch.Tensor) else []


def ulp(w):
    """One ulp (FP32) at each value of the BF16 tensor ``w``, from its exponent bits (zeros and subnormals: 2^-133)."""
    e = (w.view(torch.int16).to(torch.int32) >> 7) & 0xFF
    return torch.exp2((e.clamp_(min=1) - 134).to(torch.float32))


class Pager:
    """Host-resident states for torchao AdamW ``opt``; ``install`` swaps in the paged ``step``."""

    def __init__(self, opt, device, depth=2, kahan=False):
        """``kahan``: BF16 weights with a compensation state ("comp", paged like the others) instead of stochastic
        rounding: the step runs on weight + comp in FP32, the weight takes the round-to-nearest BF16 and comp the
        remainder, so updates far below one BF16 ulp accumulate as with FP32 masters. comp is int8 in units of
        ulp(weight) / 254, stochastically rounded (the remainder is at most half an ulp): 1 byte per weight on the host (K1, 27 Oct: a BF16
        comp's +65 GiB pinned host RAM got the container OOM-killed)."""
        self.opt, self.device, self.depth, self.kahan = opt, torch.device(device), depth, kahan
        self.cuda = self.device.type == "cuda"
        self.stream = torch.cuda.Stream(self.device) if self.cuda else None  # copies in
        self.down = torch.cuda.Stream(self.device) if self.cuda else None  # copies out (the other PCIe direction)
        self.stats = {"host_gib": 0.0, "moved_gib": 0.0, "step_s": 0.0}
        self.ptrs = set()  # the host states' storage (``owns``)

    def install(self):
        self.opt.step = self.step
        self.opt.page_out = self.page_out
        self.opt.kahan = self.kahan
        return self

    def _big(self, x, k):
        return k != "step" and x.numel() >= MIN_NUMEL

    def page_out(self):
        """Every large state on the card moves to (pinned) host memory: after the states are made or loaded."""
        total = 0
        if self.kahan:  # made on the host for the big ones (all at once on the card would not fit)
            for g in self.opt.param_groups:
                for p in g["params"]:
                    st = self.opt.state[p]
                    if len(st) and "comp" not in st and p.dtype is torch.bfloat16:
                        big = p.numel() >= MIN_NUMEL and self.cuda
                        st["comp"] = (torch.zeros(p.shape, dtype=torch.int8, pin_memory=True) if big
                                      else torch.zeros_like(p, dtype=torch.int8))
        for st in self.opt.state.values():
            for k, v in list(st.items()):
                if k == "step":
                    continue

                def out(x):
                    nonlocal total
                    if x.numel() < MIN_NUMEL or (x.device.type == "cpu" and self.cuda):
                        return x
                    h = torch.empty(x.shape, dtype=x.dtype, pin_memory=self.cuda)
                    h.copy_(x)
                    total += h.numel() * h.element_size()
                    return h

                st[k] = _map(v, out)
        if self.cuda:
            torch.cuda.empty_cache()
        host = [x for st in self.opt.state.values() for k, v in st.items() if k != "step" for x in _leaves(v)
                if x.device.type == "cpu" and x.numel() >= MIN_NUMEL]
        self.ptrs = {x.data_ptr() for x in host}
        self.stats["host_gib"] = round(sum(x.numel() * x.element_size() for x in host) / 2 ** 30, 2)
        return total / 2 ** 30

    def owns(self, t):
        """``t`` is (a view of) a paged host state: a checkpoint writes it from its own storage, before the next step
        writes it back (ckpt.Saver ``direct``)."""
        return t.device.type == "cpu" and t.data_ptr() in self.ptrs

    def _fetch(self, st):
        """Card copies of ``st``'s large tensors (copied on the side stream after the queued work) and the event."""
        if not self.cuda:  # CPU (tests): a copy, so a missing write-back shows
            return {k: (_map(v, lambda x: x.clone() if x.numel() >= MIN_NUMEL else x) if k != "step" else v)
                    for k, v in st.items()}, None
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            dev = {k: (_map(v, lambda x: x.to(self.device, non_blocking=True) if x.device.type == "cpu" else x)
                       if k != "step" else v) for k, v in st.items()}
            event = self.stream.record_event()
        return dev, event

    def _back(self, host, dev):
        """Copy the stepped card states into the host ones (side stream, after the step)."""
        if self.cuda:
            self.down.wait_stream(torch.cuda.current_stream(self.device))
        ctx = torch.cuda.stream(self.down) if self.cuda else contextlib.nullcontext()
        moved = 0
        with ctx:
            for k, v in host.items():
                if k == "step":
                    continue
                for h, d in zip(_leaves(v), _leaves(dev[k])):
                    if h is d:
                        continue
                    h.copy_(d, non_blocking=self.cuda)
                    moved += h.numel() * h.element_size()
                    if self.cuda:
                        d.record_stream(self.down)
        return moved

    @torch.no_grad()
    def step(self, closure=None):
        """torchao's ``_AdamBase.step`` for one parameter at a time on paged-in states (same call, same order)."""
        from torchao.optim.adam import single_param_adam

        assert closure is None
        if getattr(self, "_fn", None) is None:  # one compiled callable (torchao builds a new wrapper per parameter)
            self._fn = torch.compile(single_param_adam, fullgraph=True, dynamic=False)
        opt, t0, moved = self.opt, time.perf_counter(), 0
        cpu0 = os.times()
        if self.cuda:
            ev0 = torch.cuda.current_stream(self.device).record_event(torch.cuda.Event(enable_timing=True))
        work = [(g, p) for g in opt.param_groups for p in g["params"] if p.grad is not None]
        for g, p in work:  # lazy init as torchao's (init_states normally made them already), then paged out
            st = opt.state[p]
            if len(st) == 0:
                st["step"] = torch.tensor(0.0)
                st["exp_avg"] = opt._new_buffer(p, True)
                st["exp_avg_sq"] = opt._new_buffer(p, False)
                if g["amsgrad"]:
                    st["max_exp_avg_sq"] = opt._new_buffer(p, False)
                self.page_out()
            elif self.kahan and "comp" not in st and p.dtype is torch.bfloat16:
                self.page_out()  # makes comp
        ahead, nxt = collections.deque(), 0
        with torch._dynamo.utils.disable_cache_limit():
            for g, p in work:
                while nxt < len(work) and len(ahead) < self.depth:
                    ahead.append(self._fetch(opt.state[work[nxt][1]]))
                    nxt += 1
                dev, event = ahead.popleft()
                if event is not None:
                    torch.cuda.current_stream(self.device).wait_event(event)
                host = opt.state[p]
                host["step"] += 1
                if not isinstance(g["lr"], torch.Tensor):
                    raise RuntimeError("lr must stay a tensor (torchao)")
                comp = dev.get("comp")
                w = p.detach() if comp is None else p.detach().float().add_(comp.float() * ulp(p.detach()) / 254)
                self._fn(
                    w, p.grad, host["step"], dev["exp_avg"], dev["exp_avg_sq"], dev.get("max_exp_avg_sq"),
                    g["lr"], g["betas"][0], g["betas"][1], g["weight_decay"], g["eps"], opt.is_adamw,
                    opt.bf16_stochastic_round and w.dtype is torch.bfloat16)
                if comp is not None:
                    p.detach().copy_(w)
                    x = w.sub_(p.detach().float()).mul_(254).div_(ulp(p.detach()))  # remainder in ulp/254 units
                    # stochastic rounding (K1b, 27 Oct: round-to-nearest dropped every update under ulp/508)
                    comp.copy_(x.add_(torch.rand_like(x)).floor_().clamp_(-127, 127))
                    del w
                moved += self._back(host, dev)
                del dev
        issued = time.perf_counter()
        if self.cuda:
            ev1 = torch.cuda.current_stream(self.device).record_event(torch.cuda.Event(enable_timing=True))
            self.down.synchronize()
            ev1.synchronize()
            self.stats["compute_s"] = round(ev0.elapsed_time(ev1) / 1e3, 3)  # step start to last param's step queued
        cpu1 = os.times()
        self.stats["moved_gib"] = round(2 * moved / 2 ** 30, 2)
        self.stats["step_s"] = round(time.perf_counter() - t0, 3)
        self.stats["issue_s"] = round(issued - t0, 3)  # host time to queue every parameter (launch-bound if ~step_s)
        self.stats["proc_cpu_s"] = round((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system), 2)  # all threads


def check(make, device, shapes=((640, 2560), (2560, 640), (4096, 2560)), steps=3, seed=0):
    """Paged vs on-card optimizer from the same weights, gradients and seed: weights and every state byte equal after
    ``steps`` steps. ``make(params)`` builds the optimizer. Returns {"ok", "step_s_paged", "step_s_card"}."""
    gen = torch.Generator().manual_seed(seed)
    base = [torch.randn(s, generator=gen).to(torch.bfloat16) for s in shapes]
    grads = [[torch.randn(s, generator=gen).to(torch.bfloat16) * 1e-2 for s in shapes] for _ in range(steps)]
    runs = []
    for paged in (False, True):
        ps = [torch.nn.Parameter(b.clone().to(device)) for b in base]
        opt = make(ps)
        pager = Pager(opt, device).install() if paged else None
        torch.manual_seed(seed)
        if device != "cpu" and torch.device(device).type == "cuda":
            torch.cuda.manual_seed(seed)
        t = 0.0
        for gs in grads:
            for p, gr in zip(ps, gs):
                p.grad = gr.to(device)
            if pager and not opt.state:
                for p in ps:  # made on the card first, as init_states does
                    st = opt.state[p]
                    st["step"] = torch.tensor(0.0)
                    st["exp_avg"], st["exp_avg_sq"] = opt._new_buffer(p, True), opt._new_buffer(p, False)
                pager.page_out()
            t0 = time.perf_counter()
            opt.step()
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            t += time.perf_counter() - t0
        flat = [p.detach().cpu() for p in ps]
        for p in ps:
            for k, v in opt.state[p].items():
                flat += [x.detach().cpu() for x in _leaves(v)]
        runs.append((flat, t / steps))
    ok = len(runs[0][0]) == len(runs[1][0]) and all(a.dtype == b.dtype and torch.equal(a, b)
                                                    for a, b in zip(runs[0][0], runs[1][0]))
    return {"ok": bool(ok), "step_s_card": round(runs[0][1], 4), "step_s_paged": round(runs[1][1], 4)}
