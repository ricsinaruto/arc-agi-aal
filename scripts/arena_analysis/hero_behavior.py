"""Did the hero serving change model behaviour in the c110 arena? (29 September; docs/validation-20260927/hero-behavior.md)

Reads every recorded c110 one-hour run (per-game ``log.jsonl``) and prints:

1. per-call behaviour, directly standardized on (source game, level, 8K prompt bin): hero calls against the
   thinking-off 64K MTP-0 runs (kernels-v2, mtp0-wide, gittins-wide), with a (run, clone) cluster bootstrap;
2. the level-clear hazard against completion tokens spent on the level: a log-rank test stratified by
   (source game, level), each side against the rest, plus a run-level permutation test of the pooled O/E;
3. compactions per run (count, completion and reasoning tokens, seconds).

``--sampler`` (needs torch) compares the pinned vLLM top-k/top-p mask with ``topk_first_impl`` on
arena-like BF16 logits (temperature 0.7, top-k 20, top-p 0.95).

    python scripts/arena_analysis/hero_behavior.py --runs ./runs
"""

import argparse
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERO = ["hero-6x64k", "hero-21x42k", "hero-14"]
REF64_OFF = ["kernels-v2", "mtp0-wide", "gittins-wide"]
THINKING_COMPACTION = HERO + ["mtp3-wide-kernels", "depthwise-mtp3", "depthwise-mtp3-fixed-order"]
EXCLUDE = {"hero-prune12", "level-probe"}  # engine died at 1,460 s; probe actions are not play


def trials(root):
    root = Path(root)
    out = {}
    for t in sorted(root.glob("modal-validation/arena-duck-60m-c110-*/*/arena/trials/duck-transfer")):
        name = t.parents[2].name
        label = re.sub(r"^arena-duck-60m-c110-|-modal-\d+$", "", name)
        label = {"base-v1-r2": "base-v1-r2", "base-v2": "base-v2", "kernels-v2": "kernels-v2"}.get(
            label, re.sub(r"-v\d+$", "", label))
        out[label] = t
    kaggle = root / "kaggle-arena-flash-competition-hero-v1/arena/trials/duck-transfer"
    if kaggle.is_dir():
        out["hero-14"] = kaggle
    return out


def extract(trial, run):
    roster = {r["game_id"]: r["source_game_id"][:4] for r in json.loads((trial / "public-roster.json").read_text())}
    arena = json.loads((trial / "arena.json").read_text())
    switch = (arena.get("scheduler") or {}).get("switch_wall_seconds") or 0.0
    first = json.loads((trial / "admission.jsonl").open().readline())
    t0 = first["epoch"] - first["wall_seconds"]
    calls, clears, compactions = [], [], []
    for gd in sorted((trial / "games").iterdir()):
        log = gd / "log.jsonl"
        if not log.is_file():
            continue
        clone, game, level = gd.name, roster.get(gd.name, gd.name), 0
        for line in log.open():
            rec = json.loads(line)
            kind, t = rec.get("kind"), rec.get("time", 0) - t0
            base = {"run": run, "clone": clone, "game": game, "level": level + 1, "phase": 2 if switch and t > switch else 1}
            if kind == "model":
                u = rec.get("usage") or {}
                tc = (rec.get("message") or {}).get("tool_calls") or []
                calls.append({**base, "purpose": rec.get("purpose"), "completion": int(u.get("completion_tokens") or 0),
                              "reasoning": int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
                              "prompt": int(u.get("prompt_tokens") or 0),
                              "args_chars": len(tc[0]["function"].get("arguments", "")) if tc else 0})
            elif kind == "observation":
                new = int(rec.get("levels", level) or 0)
                if rec.get("action") is not None and new > level:
                    clears.append(base)
                level = max(level, new)
            elif kind == "compaction":
                u = rec.get("usage") or {}
                compactions.append({**base, "completion": u.get("completion_tokens", 0), "seconds": rec.get("seconds", 0),
                                    "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)})
    return calls, clears, compactions, arena.get("mean_score")


# ----------------------------------------------------------------------------- 1. standardized per-call behaviour
def standardized(side, ref, metric, rng, boot=400):
    def strat(r):
        return r["game"], min(r["level"], 3), min(r["prompt"] // 8192, 6)

    def clusters(rows):
        d = defaultdict(list)
        for r in rows:
            d[(r["run"], r["clone"])].append(r)
        return d

    cs, cr = clusters(side), clusters(ref)
    ks, kr = list(cs), list(cr)

    def ratio(kss, krr):
        tot, cnt = defaultdict(float), defaultdict(int)
        for k in krr:
            for r in cr[k]:
                tot[strat(r)] += metric(r)
                cnt[strat(r)] += 1
        obs = exp = 0.0
        for k in kss:
            for r in cs[k]:
                s = strat(r)
                if cnt.get(s):
                    obs += metric(r)
                    exp += tot[s] / cnt[s]
        return obs / exp if exp else float("nan")

    bs = [ratio([ks[i] for i in rng.integers(0, len(ks), len(ks))], [kr[i] for i in rng.integers(0, len(kr), len(kr))])
          for _ in range(boot)]
    return ratio(ks, kr), np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


# ----------------------------------------------------------------------------- 2. clear hazard per token
def attempts(calls, clears):
    tok, done = defaultdict(float), set()
    for r in calls:
        if r["purpose"] in ("gameplay", "compaction"):
            tok[(r["run"], r["clone"], r["game"], r["level"])] += r["completion"]
    for r in clears:
        key = (r["run"], r["clone"], r["game"], r["level"])
        done.add(key)
        tok.setdefault(key, 0.0)
    return [(k, t, k in done) for k, t in tok.items()]


def logrank(att, side, lo, hi):
    strata = defaultdict(list)
    for (run, _, g, l), t, c in att:
        if lo <= l <= hi:
            strata[(g, l)].append((t, c, run in side))
    O = E = V = 0.0
    for rows in strata.values():
        if all(h for *_, h in rows) or not any(h for *_, h in rows):
            continue
        for tau in sorted({t for t, c, _ in rows if c}):
            risk = [(t, c, h) for t, c, h in rows if t >= tau]
            R, Rh = len(risk), sum(h for *_, h in risk)
            d = sum(1 for t, c, _ in risk if c and t == tau)
            O += sum(1 for t, c, h in risk if c and t == tau and h)
            E += d * Rh / R
            if R > 1:
                V += d * (Rh / R) * (1 - Rh / R) * (R - d) / (R - 1)
    p = math.erfc(abs(O - E) / math.sqrt(2 * V)) if V > 0 else float("nan")
    return O, E, p


def permutation(att, pool, side, lo, hi):
    sub = [a for a in att if a[0][0] in pool]
    O, E, p = logrank(sub, set(side), lo, hi)
    vals = []
    for combo in itertools.combinations(pool, len(side)):
        o, e, _ = logrank(sub, set(combo), lo, hi)
        vals.append(o / e if e else float("nan"))
    return O, E, p, sum(v <= O / E + 1e-12 for v in vals) / len(vals), len(vals)


# ----------------------------------------------------------------------------- sampler (torch)
def sampler_check(batches=300, seed=0):
    import importlib.util
    import sys
    import types

    import torch

    if importlib.util.find_spec("triton") is None:  # the mask is plain torch; the kernels are GPU-only
        stub = types.ModuleType("triton")
        stub.jit = lambda f=None, **k: f if f is not None else (lambda g: g)
        stub.cdiv = lambda a, b: -(-a // b)
        stub.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
        language = types.ModuleType("triton.language")
        language.constexpr = object
        stub.language = language
        sys.modules["triton"], sys.modules["triton.language"] = stub, language
    path = Path(__file__).resolve().parents[2] / "src/arc_infra/serving_kernels_runtime.py"
    spec = importlib.util.spec_from_file_location("arc_serving_kernels_check", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    def pinned(logits, k, p):  # vLLM apply_top_k_top_p_pytorch (pinned)
        s, idx = logits.sort(dim=-1, descending=False)
        s.masked_fill_(s < s.gather(1, (s.size(1) - k.long()).unsqueeze(1)), -float("inf"))
        mask = torch.cumsum(s.softmax(dim=-1), dim=-1) <= 1 - p.unsqueeze(1)
        mask[:, -1] = False
        s.masked_fill_(mask, -float("inf"))
        return logits.scatter_(-1, idx, s)

    torch.manual_seed(seed)
    V, B = 248_320, 6
    for name, spread in (("flat", (-1.5, 2.3)), ("peaked", (-0.5, 3.5))):
        rows = differ = not_swap = 0
        tv = []
        top1 = []
        for _ in range(batches):
            x = torch.randn(B, V) * 2.0
            head = (18 + torch.rand(B, 1) * 12) - torch.exp(torch.empty(B, 1).uniform_(*spread)) * \
                torch.sort(torch.rand(B, 40) * 3, dim=1).values
            x.scatter_(1, torch.stack([torch.randperm(V)[:40] for _ in range(B)]), head)
            x = x.to(torch.bfloat16).float() / 0.7
            k, p = torch.full((B,), 20), torch.full((B,), 0.95)
            ref, new = pinned(x.clone(), k, p), helper.topk_first_impl(x.clone(), k, p, 20)
            pr, pn = ref.softmax(-1), new.softmax(-1)
            tv += (0.5 * (pr - pn).abs().sum(-1)).tolist()
            top1 += pr.max(-1).values.tolist()
            same = (torch.isfinite(ref) == torch.isfinite(new)).all(1)
            rows += B
            differ += int((~same).sum())
            for r in torch.nonzero(~same).flatten().tolist():
                a, b = pr[r][pr[r] > 0].sort().values, pn[r][pn[r] > 0].sort().values
                not_swap += int(a.shape != b.shape or not torch.allclose(a, b))
        print(f"sampler {name}: rows {rows}, top-1 prob p10/p50/p90 {np.percentile(top1, [10, 50, 90]).round(3).tolist()}, "
              f"kept set differs {differ} ({100 * differ / rows:.1f}%), not a pure equal-logit swap {not_swap}, "
              f"mean TV {np.mean(tv):.2e}, max TV {max(tv):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="./runs")
    parser.add_argument("--sampler", action="store_true")
    args = parser.parse_args()
    if args.sampler:
        sampler_check()
        return
    calls, clears, comps, scores = [], [], [], {}
    for run, trial in trials(args.runs).items():
        c, cl, cp, s = extract(trial, run)
        calls += c
        clears += cl
        comps += cp
        scores[run] = s
    rng = np.random.default_rng(1)
    gameplay = [r for r in calls if r["purpose"] == "gameplay"]
    print("## 1. Per-call behaviour, standardized on (game, level, prompt bin), vs", REF64_OFF)
    for side_name, side in (("hero (3 runs)", HERO), ("hero-6x64k", ["hero-6x64k"]), ("kernels-v2", ["kernels-v2"])):
        ref = [r for r in REF64_OFF if r not in side]
        for phase in (1, 2):
            for mname, m in (("completion", lambda r: r["completion"]), ("reasoning", lambda r: r["reasoning"]),
                             ("tool-argument chars", lambda r: r["args_chars"])):
                pt, lo, hi = standardized([r for r in gameplay if r["run"] in side and r["phase"] == phase],
                                          [r for r in gameplay if r["run"] in ref and r["phase"] == phase], m, rng)
                print(f"  {side_name} phase {phase} {mname}/call: {pt:.3f} [{lo:.3f}, {hi:.3f}]")
    att = [a for a in attempts(calls, clears) if a[0][0] not in EXCLUDE]
    runs = sorted({a[0][0] for a in att})
    print("\n## 2. Clear hazard per completion token (log-rank by game x level)")
    for run in runs:
        o1, e1, _ = logrank(att, {run}, 1, 1)
        o2, e2, p2 = logrank(att, {run}, 2, 99)
        print(f"  {run:28s} score {scores[run]:.2f}%  L1 O/E {o1 / e1 if e1 else float('nan'):.2f} ({o1:.0f}/{e1:.1f})  "
              f"L2+ O/E {o2 / e2 if e2 else float('nan'):.2f} ({o2:.0f}/{e2:.1f}, p {p2:.2f})")
    old = [r for r in runs if r not in THINKING_COMPACTION]
    for name, pool, side in (("hero vs all", runs, HERO),
                             ("thinking-on compaction vs off", runs, [r for r in THINKING_COMPACTION if r in runs]),
                             ("hero vs other thinking-on compaction", [r for r in THINKING_COMPACTION if r in runs], HERO),
                             ("hero vs thinking-off", old + HERO, HERO)):
        for lo, hi in ((1, 1), (2, 99)):
            O, E, p, pp, n = permutation(att, pool, side, lo, hi)
            print(f"  {name:40s} L{lo}{'+' if hi > 1 else ''}: O {O:.0f} E {E:.1f} O/E {O / E:.2f} log-rank p {p:.3f} "
                  f"run-permutation p {pp:.3f} ({n})")
    print("\n## sb26 level-2 attempts (tokens spent on level 2, cleared)")
    for (run, clone, game, level), t, c in sorted(att):
        if game == "sb26" and level == 2:
            print(f"  {run:28s} {clone}  {t:9,.0f}  {c}")
    print("\n## 3. Compactions")
    by = defaultdict(list)
    for r in comps:
        by[r["run"]].append(r)
    for run in sorted(by):
        rows = by[run]
        print(f"  {run:28s} {len(rows):4d}  completion {np.mean([r['completion'] for r in rows]):6.0f}  "
              f"reasoning {np.mean([r['reasoning'] for r in rows]):6.0f}  seconds {np.mean([r['seconds'] for r in rows]):4.0f}")


if __name__ == "__main__":
    main()
