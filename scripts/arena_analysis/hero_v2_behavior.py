"""Did hero-v2's serving and harness changes change how the agent plays? (29 September;
docs/validation-20260927/hero-v2-analysis/behavior.md)

CPU-only, reads recorded runs. hero-v2 (2 h, 23 x 64K, MTP-0, int8 KV, Marlin W4A16 + GPTQ-ARC, prune12, ple_uva,
prefix-cache v2, prefilled-empty compaction, verbatim reasoning history) against the one-hour c110 runs on the same
110 clones of 20 source games.

Every comparison is on the token clock, not wall time: an attempt is one (run, clone, level); its exposure is the
completion tokens (gameplay + compaction) spent on that level; it is censored if the level was not cleared.

    python scripts/arena_analysis/hero_v2_behavior.py --runs ./runs [--json out.json]

Sections: 1 clear efficiency (per level, per game, stratified log-rank, Peto HR, KM at fixed token budgets);
2 per-reply behaviour (reasoning, actions, tool use, loops, cap hits, text degeneracy), standardized on
(game, level, prompt bin); 3 per-game outcome table; 4 the N replies after each compaction.
"""

import argparse
import hashlib
import json
import math
import re
import sys
import warnings
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hero_behavior import standardized, trials  # noqa: E402

HV2 = "hero-v2"
KV2 = "kernels-v2"
# Compaction mode per run (checked against the recorded compaction reasoning tokens in section 4).
OFF = ["kernels-v2", "mtp0-wide", "gittins-wide", "base-v1-r2", "base-v2", "gittins-c110", "worldmodel", "deep-explore",
       "kernels", "base"]
EXCLUDE = {"hero-prune12", "level-probe"}
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
CYR = re.compile(r"[Ѐ-ӿ]")


def all_trials(root):
    out = trials(root)
    hv2 = Path(root) / "modal-validation/arena-hero-v2-2h-v1-modal-20260929/arena-hero-v2-2h-v1-modal-20260929/arena/trials/duck-transfer"
    if hv2.is_dir():
        out[HV2] = hv2
    return {k: v for k, v in out.items() if k not in EXCLUDE}


def text_stats(text):
    """Degeneracy indicators for one reasoning / content string."""
    text = text or ""
    n = len(text)
    lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) >= 25]
    dup = sum(c - 1 for c in Counter(lines).values() if c > 1)
    # longest run of an identical 25+ char line
    run = best = 0
    prev = None
    for ln in lines:
        run = run + 1 if ln == prev else 1
        best = max(best, run)
        prev = ln
    # repeated 8-word shingles
    words = text.split()
    sh = [" ".join(words[i:i + 8]) for i in range(0, max(0, len(words) - 7))]
    sh_dup = (1 - len(set(sh)) / len(sh)) if len(sh) >= 50 else 0.0
    comp = len(zlib.compress(text.encode("utf-8", "ignore"))) / max(1, len(text.encode("utf-8", "ignore"))) if n >= 400 else 1.0
    nonascii = sum(1 for ch in text if ord(ch) > 127)
    return {"chars": n, "dup_lines": dup, "dup_line_frac": dup / len(lines) if lines else 0.0, "max_line_run": best,
            "shingle_dup": sh_dup, "zlib_ratio": comp, "cjk": len(CJK.findall(text)), "cyr": len(CYR.findall(text)),
            "nonascii_frac": nonascii / n if n else 0.0}


def frame_key(frames):
    if not frames:
        return None
    return hashlib.md5("".join(frames[-1]).encode()).hexdigest()


def extract(trial, run):
    roster = {r["game_id"]: r["source_game_id"][:4] for r in json.loads((trial / "public-roster.json").read_text())}
    arena = json.loads((trial / "arena.json").read_text())
    switch = (arena.get("scheduler") or {}).get("switch_wall_seconds") or 0.0
    first = json.loads((trial / "admission.jsonl").open().readline())
    t0 = first["epoch"] - first["wall_seconds"]
    replies, attempts, comps = [], {}, []
    for gd in sorted((trial / "games").iterdir()):
        log = gd / "log.jsonl"
        if not log.is_file():
            continue
        clone, game = gd.name, roster.get(gd.name, gd.name)
        level, cur, last_key, since_comp, last_code = 0, None, None, None, None
        tried = defaultdict(set)  # frame hash -> actions that produced no change there (this level)
        seen_frames = set()

        def att(lv):
            key = (run, clone, game, lv)
            if key not in attempts:
                attempts[key] = {"run": run, "clone": clone, "game": game, "level": lv, "tokens": 0, "gameplay_tokens": 0,
                                 "actions": 0, "replies": 0, "cleared": False, "compactions": 0, "clear_t": None,
                                 "tokens_at_clear": None}
            return attempts[key]

        for line in log.open():
            rec = json.loads(line)
            kind, t = rec.get("kind"), rec.get("time", 0) - t0
            if kind == "model":
                u = rec.get("usage") or {}
                msg = rec.get("message") or {}
                tc = msg.get("tool_calls") or []
                comp = int(u.get("completion_tokens") or 0)
                a = att(level + 1)
                a["tokens"] += comp
                if rec.get("purpose") != "gameplay":
                    continue
                a["gameplay_tokens"] += comp
                a["replies"] += 1
                code = None
                if tc:
                    try:
                        code = json.loads(tc[0]["function"].get("arguments") or "{}").get("code")
                    except (json.JSONDecodeError, AttributeError):
                        code = tc[0]["function"].get("arguments")
                rs = text_stats(msg.get("reasoning") or msg.get("reasoning_content") or "")
                cs = text_stats(msg.get("content") or "")
                cur = {"run": run, "clone": clone, "game": game, "level": level + 1, "t": t,
                       "phase": 2 if switch and t > switch else 1,
                       "completion": comp, "prompt": int(u.get("prompt_tokens") or 0),
                       "reasoning": int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
                       "finish": rec.get("finish_reason"), "n_tools": len(tc),
                       "python": any(x["function"]["name"] == "python" for x in tc),
                       "args_chars": len(tc[0]["function"].get("arguments", "")) if tc else 0,
                       "code_repeat": code is not None and code == last_code,
                       "actions": 0, "nochange": 0, "tried_repeat": 0, "revisit": 0, "clear": False,
                       "run_act": None, "run_len": 0, "run_nc": 0, "stuck_run": 0,
                       "since_comp": since_comp, "level_tokens_before": a["tokens"] - comp,
                       "r": rs, "c": cs, "reasoning_text": None}
                if comp >= 16000 or rs["max_line_run"] >= 5 or rs["cjk"] > 0 or rs["shingle_dup"] > 0.3:
                    cur["reasoning_text"] = (msg.get("reasoning") or "")[-3000:]
                    cur["content_text"] = (msg.get("content") or "")[-1000:]
                    cur["code_text"] = (code or "")[-1000:]
                last_code = code if code is not None else last_code
                if since_comp is not None:
                    since_comp += 1
                replies.append(cur)
            elif kind == "observation":
                key = frame_key(rec.get("frames"))
                new = int(rec.get("levels", level) or 0)
                act = rec.get("action")
                if act is not None:
                    a = att(level + 1)
                    a["actions"] += 1
                    akey = json.dumps(act, sort_keys=True)
                    changed = key != last_key
                    if cur is not None:
                        cur["actions"] += 1
                        if not changed:
                            cur["nochange"] += 1
                        if last_key in tried and akey in tried[last_key]:
                            cur["tried_repeat"] += 1
                        if changed and key in seen_frames:
                            cur["revisit"] += 1
                        # blocked-move loop: >= 6 identical consecutive actions in one reply, >= 3 of them no-ops
                        if akey == cur["run_act"]:
                            cur["run_len"] += 1
                        else:
                            cur["run_act"], cur["run_len"], cur["run_nc"] = akey, 1, 0
                        cur["run_nc"] += not changed
                        if cur["run_len"] >= 6 and cur["run_nc"] >= 3:
                            cur["stuck_run"] = max(cur["stuck_run"], cur["run_len"])
                    if not changed:
                        tried[last_key].add(akey)
                    if new > level:
                        a["cleared"] = True
                        a["clear_t"] = t
                        a["tokens_at_clear"] = a["tokens"]
                        if cur is not None:
                            cur["clear"] = True
                        tried = defaultdict(set)
                        seen_frames = set()
                        since_comp = None  # the post-compaction window is same-level only
                if key is not None:
                    seen_frames.add(key)
                last_key = key
                level = max(level, new)
            elif kind == "compaction":
                u = rec.get("usage") or {}
                ca = att(level + 1)
                if ca["compactions"] == 0:
                    ca["tokens_at_first_comp"] = ca["tokens"]
                ca["compactions"] += 1
                comps.append({"run": run, "clone": clone, "game": game, "level": level + 1, "t": t,
                              "completion": int(u.get("completion_tokens") or 0),
                              "reasoning": int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
                              "prompt": int(u.get("prompt_tokens") or 0), "summary_chars": rec.get("summary_chars"),
                              "thinking": rec.get("thinking"), "fallback": rec.get("mechanical_fallback")})
                since_comp = 0
    return replies, list(attempts.values()), comps, arena


# ----------------------------------------------------------------------------- statistics helpers
def poisson_ci(k, expo, scale=1e4):
    """Exact (Garwood) 95% CI of a rate k / expo, per ``scale`` tokens."""
    from scipy.stats import chi2

    lo = chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * k + 2) / 2
    return k / expo * scale, lo / expo * scale, hi / expo * scale


def peto(att_rows, side, lo, hi, clock="tokens"):
    """Stratified (game, level) log-rank of ``side`` runs vs the rest, with the Peto HR and its 95% CI."""
    rows = [((a["run"], a["clone"], a["game"], a["level"]), a[clock], a["cleared"]) for a in att_rows]
    strata = defaultdict(list)
    for (run, _, g, lv), t, c in rows:
        if lo <= lv <= hi:
            strata[(g, lv)].append((t, c, run in side))
    O = E = V = 0.0  # noqa: E741
    for rs in strata.values():
        if all(h for *_, h in rs) or not any(h for *_, h in rs):
            continue
        for tau in sorted({t for t, c, _ in rs if c}):
            risk = [(t, c, h) for t, c, h in rs if t >= tau]
            R, Rh = len(risk), sum(h for *_, h in risk)
            d = sum(1 for t, c, _ in risk if c and t == tau)
            O += sum(1 for t, c, h in risk if c and t == tau and h)  # noqa: E741
            E += d * Rh / R
            if R > 1:
                V += d * (Rh / R) * (1 - Rh / R) * (R - d) / (R - 1)
    if V <= 0:
        return O, E, float("nan"), float("nan"), float("nan"), float("nan")
    lhr = (O - E) / V
    p = math.erfc(abs(O - E) / math.sqrt(2 * V))
    return O, E, math.exp(lhr), math.exp(lhr - 1.96 / math.sqrt(V)), math.exp(lhr + 1.96 / math.sqrt(V)), p


def km_at(att_rows, budgets):
    """Kaplan-Meier P(clear within b tokens on the level) for each b."""
    ts = sorted(((a["tokens_at_clear"] if a["cleared"] else a["tokens"]), a["cleared"]) for a in att_rows)
    out = []
    for b in budgets:
        s, n = 1.0, len(ts)
        at_risk = n
        for t, c in ts:
            if t > b:
                break
            if c:
                s *= 1 - 1 / at_risk
            at_risk -= 1
        out.append(1 - s)
    return out


def km_boot(att_rows, budgets, rng, boot=1000):
    pt = km_at(att_rows, budgets)
    if not att_rows:
        return pt, [float("nan")] * len(budgets), [float("nan")] * len(budgets)
    bs = np.array([km_at([att_rows[i] for i in rng.integers(0, len(att_rows), len(att_rows))], budgets)
                   for _ in range(boot)])
    return pt, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


def cluster_mean(rows, f, rng, boot=1000):
    """Mean of f over rows with a clone-cluster bootstrap CI."""
    if not rows:
        return float("nan"), float("nan"), float("nan")
    cl = defaultdict(list)
    for r in rows:
        cl[(r["run"], r["clone"])].append(f(r))
    keys = list(cl)
    s = np.array([sum(cl[k]) for k in keys])
    n = np.array([len(cl[k]) for k in keys])
    pt = s.sum() / n.sum()
    idx = rng.integers(0, len(keys), (boot, len(keys)))
    b = s[idx].sum(1) / n[idx].sum(1)
    return pt, np.percentile(b, 2.5), np.percentile(b, 97.5)


def ratio_of_sums(rows, num, den, rng, boot=1000):
    cl = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        cl[(r["run"], r["clone"])][0] += num(r)
        cl[(r["run"], r["clone"])][1] += den(r)
    keys = list(cl)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    a = np.array([cl[k][0] for k in keys])
    b = np.array([cl[k][1] for k in keys])
    idx = rng.integers(0, len(keys), (boot, len(keys)))
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = a[idx].sum(1) / b[idx].sum(1)
    return a.sum() / b.sum() if b.sum() else float("nan"), np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def fmt(x, d=2):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def band(lv):
    return "L1" if lv == 1 else ("L2" if lv == 2 else "L3+")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="./runs")
    ap.add_argument("--json", default=None)
    ap.add_argument("--window", type=int, default=10, help="replies after a compaction")
    ap.add_argument("--samples", default=None, help="write flagged reply texts (cap hits, degenerate) here")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN bootstrap cells on empty strata
    rng = np.random.default_rng(7)
    T = all_trials(args.runs)
    R, A, C, AR = [], [], [], {}
    for run, trial in T.items():
        r, a, c, ar = extract(trial, run)
        R += r
        A += a
        C += c
        AR[run] = ar
    runs = sorted(T)
    N = args.window
    others = [r for r in runs if r != HV2]
    res = {"runs": runs}
    focus = [HV2, KV2, "hero-6x64k", "hero-21x42k", "mtp0-wide", "gittins-wide", "hero-14"]
    focus = [r for r in focus if r in runs]
    print("runs:", ", ".join(f"{r} ({AR[r].get('mean_score')}%)" for r in runs))

    # ---------------------------------------------------------------- 1. clear efficiency
    print("\n## 1a. Clears per 10K level tokens by level (exposure = completion tokens on the level, all attempts)")
    tab = {}
    for run in focus:
        for b in ("L1", "L2", "L3+"):
            rows = [a for a in A if a["run"] == run and band(a["level"]) == b and a["tokens"] > 0]
            k = sum(a["cleared"] for a in rows)
            ex = sum(a["tokens"] for a in rows)
            rate = poisson_ci(k, ex) if ex else (float("nan"),) * 3
            ttc = [a["tokens_at_clear"] for a in rows if a["cleared"]]
            atc = [a["actions"] for a in rows if a["cleared"]]
            tab[(run, b)] = {"attempts": len(rows), "clears": k, "tokens": ex, "rate": rate,
                             "median_tokens_to_clear": float(np.median(ttc)) if ttc else None,
                             "median_actions_to_clear": float(np.median(atc)) if atc else None}
            print(f"  {run:14s} {b:4s} attempts {len(rows):4d} clears {k:3d} tokens {ex / 1e6:6.2f}M  "
                  f"rate/10K {fmt(rate[0], 3)} [{fmt(rate[1], 3)}, {fmt(rate[2], 3)}]  "
                  f"median tok-to-clear {fmt(np.median(ttc) if ttc else float('nan'), 0)}  "
                  f"median actions-to-clear {fmt(np.median(atc) if atc else float('nan'), 0)}")
    res["rates"] = {f"{k[0]}|{k[1]}": v for k, v in tab.items()}

    print("\n## 1b. Stratified log-rank (game x level) on level tokens: Peto HR = hazard ratio of the first side")
    res["logrank"] = {}
    comps_lr = [(HV2, [HV2, KV2], {HV2}), (HV2 + " vs all 1h", runs, {HV2}),
                (HV2 + " vs thinking-off 1h", [HV2] + [r for r in OFF if r in runs], {HV2}),
                ("hero-6x64k vs kernels-v2", ["hero-6x64k", KV2], {"hero-6x64k"})]
    for name, pool, side in comps_lr:
        sub = [a for a in A if a["run"] in pool]
        for lo, hi, lab in ((1, 1, "L1"), (2, 2, "L2"), (3, 99, "L3+"), (2, 99, "L2+")):
            for clock in ("tokens", "gameplay_tokens", "actions"):
                O, E, hr, l, h, p = peto(sub, side, lo, hi, clock)  # noqa: E741
                res["logrank"][f"{name}|{lab}|{clock}"] = [O, E, hr, l, h, p]
                if clock == "tokens" or lab in ("L1", "L2+"):
                    print(f"  {name:32s} {lab:4s} {clock:15s} O {O:3.0f} E {E:5.1f} O/E {fmt(O / E if E else float('nan'))} "
                          f"HR {fmt(hr)} [{fmt(l)}, {fmt(h)}] p {fmt(p, 3)}")

    print("\n## 1c. Kaplan-Meier P(clear within B tokens on the level), 95% bootstrap over attempts")
    budgets = [10_000, 25_000, 50_000, 100_000]
    res["km"] = {}
    for run in focus:
        for b in ("L1", "L2", "L3+"):
            rows = [a for a in A if a["run"] == run and band(a["level"]) == b and a["tokens"] > 0]
            pt, lo, hi = km_boot(rows, budgets, rng)
            res["km"][f"{run}|{b}"] = [list(map(float, pt)), list(map(float, lo)), list(map(float, hi))]
            print(f"  {run:14s} {b:4s} n {len(rows):4d}  " + "  ".join(
                f"{bb // 1000}K {pt[i]:.2f} [{lo[i]:.2f},{hi[i]:.2f}]" for i, bb in enumerate(budgets)))

    print("\n## 1d. Per source game, L1 and L2+: attempts / clears / Mtokens (hero-v2 | kernels-v2 | other 1 h runs)")
    games = sorted({a["game"] for a in A})
    res["per_game"] = {}
    for g in games:
        cells = []
        for grp in ([HV2], [KV2], [r for r in others if r != KV2]):
            for b in ("L1", "L2+"):
                rows = [a for a in A if a["run"] in grp and a["game"] == g and a["tokens"] > 0 and
                        (a["level"] == 1 if b == "L1" else a["level"] >= 2)]
                cells.append((len(rows), sum(x["cleared"] for x in rows), sum(x["tokens"] for x in rows) / 1e6))
        res["per_game"][g] = cells
        # per-game hero-v2 vs all others, L1 and L2+
        sub = [a for a in A if a["game"] == g]
        lr1 = peto(sub, {HV2}, 1, 1)
        lr2 = peto(sub, {HV2}, 2, 99)
        print(f"  {g}  " + " | ".join(f"{n}/{k}/{m:.2f}" for n, k, m in cells) +
              f"   L1 O/E {lr1[0]:.0f}/{lr1[1]:.1f}  L2+ O/E {lr2[0]:.0f}/{lr2[1]:.1f}")
        res["per_game"][g] = {"cells": cells, "l1": lr1[:2], "l2p": lr2[:2]}

    print("\n## 1e. Max level reached per clone, per game (hero-v2 at 1 h | at 2 h ; kernels-v2 ; hero-6x64k ; hero-21x42k)")
    res["depth"] = {}
    for g in games:
        line = []
        for run, tmax in ((HV2, 3600), (HV2, 1e9), (KV2, 1e9), ("hero-6x64k", 1e9), ("hero-21x42k", 1e9)):
            if run not in runs:
                continue
            clones = sorted({a["clone"] for a in A if a["run"] == run and a["game"] == g})
            depth = []
            for c in clones:
                cl = [a for a in A if a["run"] == run and a["clone"] == c and a["cleared"] and a["clear_t"] <= tmax]
                depth.append(len(cl))
            line.append(sum(depth))
        res["depth"][g] = line
        print(f"  {g}  levels: hv2@1h {line[0]:2d}  hv2@2h {line[1]:2d}  kv2 {line[2]:2d}  h6 {line[3]:2d}  h21 {line[4]:2d}")

    # ---------------------------------------------------------------- 2. per-reply behaviour
    G = [r for r in R if r["run"] in runs]
    print("\n## 2a. Per-reply behaviour, raw (gameplay replies), clone-cluster bootstrap 95% CI")
    metrics = [
        ("reasoning tok/reply", lambda r: r["reasoning"]),
        ("completion tok/reply", lambda r: r["completion"]),
        ("actions/reply", lambda r: r["actions"]),
        ("zero-action replies", lambda r: r["actions"] == 0),
        ("python tool use", lambda r: r["python"]),
        ("no tool call", lambda r: r["n_tools"] == 0),
        ("length cap (16,384)", lambda r: r["finish"] == "length"),
        ("code identical to previous", lambda r: r["code_repeat"]),
        ("reasoning dup-line frac", lambda r: r["r"]["dup_line_frac"]),
        ("reasoning 8-gram dup", lambda r: r["r"]["shingle_dup"]),
        ("reasoning with CJK", lambda r: r["r"]["cjk"] > 0),
        ("reasoning with Cyrillic", lambda r: r["r"]["cyr"] > 0),
        ("reasoning line run >= 5", lambda r: r["r"]["max_line_run"] >= 5),
    ]
    res["per_reply_raw"] = {}
    for run in focus:
        rows = [r for r in G if r["run"] == run]
        out = {}
        for name, f in metrics:
            out[name] = cluster_mean(rows, lambda r, f=f: float(f(r)), rng)
        for name, num in (("no-change / action", lambda r: r["nochange"]),
                          ("tried-no-effect repeat / action", lambda r: r["tried_repeat"])):
            out[name] = ratio_of_sums(rows, num, lambda r: r["actions"], rng)
        res["per_reply_raw"][run] = {k: list(map(float, v)) for k, v in out.items()} | {"n": len(rows)}
        print(f"  {run} (n {len(rows)})")
        for k, v in out.items():
            d = 4 if v[0] < 0.1 else 2
            print(f"      {k:34s} {fmt(v[0], d)} [{fmt(v[1], d)}, {fmt(v[2], d)}]")

    print("\n## 2b. Standardized on (game, level<=3, 8K prompt bin): hero-v2 / reference, cluster bootstrap")
    refs = {"kernels-v2": [KV2], "thinking-off 1h (8 runs)": [r for r in OFF if r in runs],
            "all 1h": others}
    std_metrics = [("reasoning", lambda r: r["reasoning"]), ("completion", lambda r: r["completion"]),
                   ("actions", lambda r: r["actions"]), ("no-change actions", lambda r: r["nochange"]),
                   ("tried-no-effect repeats", lambda r: r["tried_repeat"]),
                   ("zero-action reply", lambda r: float(r["actions"] == 0)),
                   ("code repeat", lambda r: float(r["code_repeat"]))]
    res["standardized"] = {}
    for phase in (1, 2):
        for rname, rr in refs.items():
            side = [r for r in G if r["run"] == HV2 and r["phase"] == phase]
            ref = [r for r in G if r["run"] in rr and r["phase"] == phase]
            for mname, m in std_metrics:
                pt, lo, hi = standardized(side, ref, m, rng, boot=300)
                res["standardized"][f"p{phase}|{rname}|{mname}"] = [pt, lo, hi]
                print(f"  phase {phase} vs {rname:26s} {mname:24s} {fmt(pt, 3)} [{fmt(lo, 3)}, {fmt(hi, 3)}]")

    print("\n## 2c. By prompt size (hero-v2 vs kernels-v2 vs hero-21x42k): reasoning/reply, no-change/action, cap hits")
    res["by_prompt"] = {}
    for run in [HV2, KV2, "hero-6x64k", "hero-21x42k"]:
        for lo, hi in ((0, 16_384), (16_384, 32_768), (32_768, 49_152), (49_152, 1 << 20)):
            rows = [r for r in G if r["run"] == run and lo <= r["prompt"] < hi]
            if not rows:
                continue
            rz = cluster_mean(rows, lambda r: r["reasoning"], rng)
            nc = ratio_of_sums(rows, lambda r: r["nochange"], lambda r: r["actions"], rng)
            cap = sum(r["finish"] == "length" for r in rows)
            dup = cluster_mean(rows, lambda r: r["r"]["shingle_dup"], rng)
            res["by_prompt"][f"{run}|{lo}"] = {"n": len(rows), "reasoning": rz, "nochange": nc, "cap": cap, "dup": dup}
            print(f"  {run:12s} prompt {lo // 1024:2d}-{min(hi, 65536) // 1024:2d}K n {len(rows):4d}  reasoning {rz[0]:6.0f} "
                  f"[{rz[1]:.0f},{rz[2]:.0f}]  no-change {nc[0]:.3f} [{nc[1]:.3f},{nc[2]:.3f}]  8-gram dup {dup[0]:.4f}  cap hits {cap}")

    print("\n## 2d. Length-cap hits and degenerate replies (all runs)")
    res["flagged"] = []
    for run in runs:
        rows = [r for r in G if r["run"] == run]
        cap = [r for r in rows if r["finish"] == "length"]
        deg = [r for r in rows if r["r"]["max_line_run"] >= 5 or r["r"]["shingle_dup"] > 0.3]
        cjk = [r for r in rows if r["r"]["cjk"] > 0]
        print(f"  {run:26s} replies {len(rows):5d}  cap {len(cap):3d} ({100 * len(cap) / max(1, len(rows)):.2f}%)  "
              f"degenerate {len(deg):3d}  CJK {len(cjk):3d}  max-prompt {max((r['prompt'] for r in rows), default=0)}")
    print("\n## 2e. Tail events, standardized: hero-v2 (or hero-6x64k) observed vs expected from the reference runs")
    from scipy.stats import poisson

    res["tail"] = {}
    tail_events = [("length cap", lambda r: r["finish"] == "length"),
                   ("reasoning >= 8K", lambda r: r["reasoning"] >= 8192),
                   ("reasoning >= 12K", lambda r: r["reasoning"] >= 12288),
                   ("zero actions", lambda r: r["actions"] == 0),
                   (">= 1 tried-no-effect repeat", lambda r: r["tried_repeat"] > 0),
                   ("blocked-move loop", lambda r: r["stuck_run"] > 0),
                   ("no tool call", lambda r: r["n_tools"] == 0)]
    stratifications = {"game x level": lambda r: (r["game"], r["level"]),
                       "game x level x level-tokens": lambda r: (r["game"], r["level"],
                                                                 min(2, int(r["level_tokens_before"] >= 20_000) +
                                                                     int(r["level_tokens_before"] >= 50_000)))}
    for sname, strat in stratifications.items():
        print(f"  strata: {sname} (strata with >= 20 reference replies)")
        for side_name, side, ref_runs in (("hero-v2 vs all 1h", [HV2], others),
                                          ("hero-v2 vs thinking-off 1h", [HV2], [r for r in OFF if r in runs]),
                                          ("hero-6x64k vs thinking-off 1h", ["hero-6x64k"],
                                           [r for r in OFF if r in runs])):
            for ename, ev in tail_events:
                cnt, n = defaultdict(float), defaultdict(int)
                for r in G:
                    if r["run"] in ref_runs:
                        cnt[strat(r)] += ev(r)
                        n[strat(r)] += 1
                obs = exp = covered = total = 0
                for r in G:
                    if r["run"] in side:
                        total += 1
                        s = strat(r)
                        if n.get(s, 0) >= 20:
                            covered += 1
                            obs += ev(r)
                            exp += cnt[s] / n[s]
                p_hi = poisson.sf(obs - 1, exp) if exp > 0 else float("nan")
                p_lo = poisson.cdf(obs, exp) if exp > 0 else float("nan")
                res["tail"][f"{sname}|{side_name}|{ename}"] = [obs, exp, covered, total, p_hi, p_lo]
                print(f"    {side_name:30s} {ename:28s} O {obs:4.0f} E {exp:6.1f} "
                      f"O/E {fmt(obs / exp if exp else float('nan'))} (covered {covered}/{total}); "
                      f"P(>=O) {fmt(p_hi, 3)} P(<=O) {fmt(p_lo, 3)}")
    print("\n  Blocked-move loops (>= 6 identical consecutive actions in one reply, >= 3 no-ops), per run")
    for run in runs:
        rows = [r for r in G if r["run"] == run]
        lp = [r for r in rows if r["stuck_run"] > 0]
        post = [r for r in lp if r["since_comp"] is not None and 1 <= r["since_comp"] <= N]
        print(f"    {run:26s} {len(lp):3d} / {len(rows):5d} replies ({100 * len(lp) / max(1, len(rows)):.2f}%), "
              f"{len(post)} within {N} replies after a compaction; longest {max((r['stuck_run'] for r in lp), default=0)}")

    if args.samples:
        with open(args.samples, "w") as fh:
            for r in G:
                if r.get("reasoning_text") is not None:
                    fh.write(json.dumps({k: r[k] for k in ("run", "clone", "game", "level", "t", "prompt", "completion",
                                                           "reasoning", "finish", "actions", "r")} |
                                        {"reasoning_tail": r["reasoning_text"], "content_tail": r.get("content_text"),
                                         "code_tail": r.get("code_text")}) + "\n")

    # ---------------------------------------------------------------- 4. post-compaction
    print(f"\n## 4. Compactions and the {N} gameplay replies after each (same clone)")
    res["compaction"] = {}
    groups = {"hero-v2 (prefilled-empty)": [HV2],
              "kernels-v2 (off)": [KV2],
              "thinking-off 1h": [r for r in OFF if r in runs],
              "thinking-on 1h": [r for r in runs if r not in OFF and r != HV2]}
    for gname, grp in groups.items():
        cps = [c for c in C if c["run"] in grp]
        if not cps:
            continue
        after = [r for r in G if r["run"] in grp and r["since_comp"] is not None and 1 <= r["since_comp"] <= N]
        # control: replies at >= 32K prompt that are not within N of a compaction (the pre-compaction stretch)
        before = [r for r in G if r["run"] in grp and r["prompt"] >= 32_768 and
                  (r["since_comp"] is None or r["since_comp"] > N)]
        reas = np.mean([c["reasoning"] for c in cps])
        compl = np.mean([c["completion"] for c in cps])
        out = {"n": len(cps), "reasoning": float(reas), "completion": float(compl),
               "summary_chars": float(np.mean([c["summary_chars"] or 0 for c in cps])),
               "fallbacks": sum(bool(c["fallback"]) for c in cps)}
        for lab, rows in (("after", after), ("pre-compaction >=32K", before)):
            clear = ratio_of_sums(rows, lambda r: float(r["clear"]), lambda r: r["completion"] / 1e4, rng)
            nc = ratio_of_sums(rows, lambda r: r["nochange"], lambda r: r["actions"], rng)
            tr = ratio_of_sums(rows, lambda r: r["tried_repeat"], lambda r: r["actions"], rng)
            cr = cluster_mean(rows, lambda r: float(r["code_repeat"]), rng)
            ap_ = cluster_mean(rows, lambda r: r["actions"], rng)
            rz = cluster_mean(rows, lambda r: r["reasoning"], rng)
            out[lab] = {"replies": len(rows), "clears": sum(r["clear"] for r in rows), "clears_per_10k": clear,
                        "nochange": nc, "tried_repeat": tr, "code_repeat": cr, "actions": ap_, "reasoning": rz}
        res["compaction"][gname] = out
        print(f"  {gname:28s} compactions {len(cps):3d}  completion {compl:5.0f}  reasoning {reas:5.0f}  "
              f"summary chars {out['summary_chars']:5.0f}  fallbacks {out['fallbacks']}")
        for lab in ("after", "pre-compaction >=32K"):
            o = out[lab]
            print(f"      {lab:22s} replies {o['replies']:4d} clears {o['clears']:3d}  clears/10K {o['clears_per_10k'][0]:.3f} "
                  f"[{o['clears_per_10k'][1]:.3f},{o['clears_per_10k'][2]:.3f}]  no-change {o['nochange'][0]:.3f} "
                  f"[{o['nochange'][1]:.3f},{o['nochange'][2]:.3f}]  tried-repeat {o['tried_repeat'][0]:.3f} "
                  f"[{o['tried_repeat'][1]:.3f},{o['tried_repeat'][2]:.3f}]  code-repeat {o['code_repeat'][0]:.3f}  "
                  f"actions/reply {o['actions'][0]:.2f}  reasoning {o['reasoning'][0]:.0f}")
    # post-compaction clear hazard on the level: levels whose attempt saw >= 1 compaction
    print("\n  Level attempts that compacted at least once: clears / attempts, tokens after the first compaction")
    for gname, grp in groups.items():
        rows = [a for a in A if a["run"] in grp and a["compactions"] > 0]
        k = sum(a["cleared"] for a in rows)
        print(f"      {gname:28s} attempts {len(rows):3d} cleared {k:3d}")
        res["compaction"].setdefault(gname, {})["compacted_attempts"] = [len(rows), k]

    # Within-attempt contrast: the N replies after a compaction against the other replies of the same attempt
    # (Mantel-Haenszel rate ratio, strata = (run, clone, level)); compared across compaction modes.
    print(f"\n  Within-attempt Mantel-Haenszel rate ratio, replies 1..{N} after a compaction vs the rest of the attempt")
    res["compaction_mh"] = {}
    for gname, grp in groups.items():
        rows = [r for r in G if r["run"] in grp]
        att_with = {(a["run"], a["clone"], a["level"]) for a in A if a["run"] in grp and a["compactions"] > 0}
        rows = [r for r in rows if (r["run"], r["clone"], r["level"]) in att_with]

        def post(r):
            return r["since_comp"] is not None and 1 <= r["since_comp"] <= N

        out = {}
        for mname, ev, ex in (("no-change per action", lambda r: r["nochange"], lambda r: r["actions"]),
                              ("tried-repeat per action", lambda r: r["tried_repeat"], lambda r: r["actions"]),
                              ("actions per token", lambda r: r["actions"], lambda r: r["completion"] / 1e4)):
            out[mname] = mh_rate_ratio(rows, post, ev, ex, lambda r: (r["run"], r["clone"], r["level"]))
            rr, lo, hi, a1, b0 = out[mname]
            print(f"      {gname:28s} {mname:24s} RR {fmt(rr)} [{fmt(lo)}, {fmt(hi)}]  (events post {a1:.0f}, rest {b0:.0f})")
        res["compaction_mh"][gname] = out

    # Continuation after the first compaction on a level: clear hazard against the level tokens spent after it,
    # stratified by (game, level); attempts that never compacted are left out. 64K-window runs only.
    print("\n  Clear hazard after the first compaction of a level attempt (log-rank on tokens since that compaction)")
    post_att = []
    for a in A:
        if a["compactions"] > 0 and a["run"] != "hero-21x42k":
            end = a["tokens_at_clear"] if a["cleared"] else a["tokens"]
            post_att.append({**a, "post_tokens": max(0.0, end - a["tokens_at_first_comp"])})
    res["post_compaction_logrank"] = {}
    for name, pool, side in (("hero-v2 vs thinking-off 1h", [HV2] + groups["thinking-off 1h"], {HV2}),
                             ("hero-v2 vs thinking-on 1h (64K)", [HV2] + [r for r in groups["thinking-on 1h"]
                                                                          if r != "hero-21x42k"], {HV2}),
                             ("thinking-on vs thinking-off 1h (64K)", groups["thinking-off 1h"] +
                              [r for r in groups["thinking-on 1h"] if r != "hero-21x42k"],
                              set(r for r in groups["thinking-on 1h"] if r != "hero-21x42k"))):
        sub = [a for a in post_att if a["run"] in pool]
        O, E, hr, lo, hi, p = peto(sub, side, 1, 99, "post_tokens")  # noqa: E741
        k1 = sum(a["cleared"] for a in sub if a["run"] in side)
        n1 = sum(1 for a in sub if a["run"] in side)
        k0 = sum(a["cleared"] for a in sub if a["run"] not in side)
        n0 = sum(1 for a in sub if a["run"] not in side)
        res["post_compaction_logrank"][name] = [O, E, hr, lo, hi, p, k1, n1, k0, n0]
        print(f"      {name:38s} side {k1}/{n1} cleared, rest {k0}/{n0}; O {O:.0f} E {E:.1f} "
              f"HR {fmt(hr)} [{fmt(lo)}, {fmt(hi)}] p {fmt(p, 3)}")

    # Reorientation after a compaction: gameplay tokens and replies until the first action
    print("\n  Reorientation: gameplay tokens / replies from a compaction to the next action (same level)")
    res["reorientation"] = {}
    for gname, grp in groups.items():
        toks, reps = [], []
        by_clone = defaultdict(list)
        for r in G:
            if r["run"] in grp:
                by_clone[(r["run"], r["clone"])].append(r)
        for rows in by_clone.values():
            for i, r in enumerate(rows):
                if r["since_comp"] == 1:
                    t = k = 0
                    for q in rows[i:]:
                        if q["since_comp"] is None or q["since_comp"] < 1 or q["level"] != r["level"]:
                            break
                        t += q["completion"]
                        k += 1
                        if q["actions"] > 0:
                            break
                    toks.append(t)
                    reps.append(k)
        if toks:
            b = [np.median(rng.choice(toks, len(toks))) for _ in range(1000)]
            res["reorientation"][gname] = [float(np.median(toks)), float(np.mean(toks)), float(np.mean(reps)), len(toks),
                                           float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
            print(f"      {gname:28s} n {len(toks):3d}  median tokens {np.median(toks):6.0f} [{np.percentile(b, 2.5):.0f}, "
                  f"{np.percentile(b, 97.5):.0f}]  mean tokens {np.mean(toks):6.0f}  mean replies {np.mean(reps):.2f}")

    # Reorientation standardized by source game (compactions happen on different games in different runs)
    reo = defaultdict(list)
    by_clone = defaultdict(list)
    for r in G:
        by_clone[(r["run"], r["clone"])].append(r)
    for rows in by_clone.values():
        for i, r in enumerate(rows):
            if r["since_comp"] == 1:
                t = 0
                for q in rows[i:]:
                    if q["since_comp"] is None or q["since_comp"] < 1 or q["level"] != r["level"]:
                        break
                    t += q["completion"]
                    if q["actions"] > 0:
                        break
                reo[r["run"]].append((r["game"], t))
    on64 = [r for r in groups["thinking-on 1h"] if r != "hero-21x42k"]
    print("\n  Reorientation tokens, standardized by game (ratio of sums, bootstrap over compactions)")
    res["reorientation_std"] = {}
    for side, refname, ref_runs in ((HV2, "kernels-v2", [KV2]), (HV2, "thinking-off 1h", groups["thinking-off 1h"]),
                                    (HV2, "thinking-on 1h (64K)", on64),
                                    (KV2, "other thinking-off 1h", [r for r in groups["thinking-off 1h"] if r != KV2]),
                                    (KV2, "thinking-on 1h (64K)", on64)):
        means = defaultdict(list)
        for run in ref_runs:
            for g, t in reo[run]:
                means[g].append(t)
        rows = [(g, t) for g, t in reo[side] if len(means.get(g, [])) >= 3]
        if not rows:
            continue

        def ratio(rs):
            return sum(t for _, t in rs) / sum(np.mean(means[g]) for g, _ in rs)

        bs = [ratio([rows[i] for i in rng.integers(0, len(rows), len(rows))]) for _ in range(1000)]
        res["reorientation_std"][f"{side} vs {refname}"] = [ratio(rows), float(np.percentile(bs, 2.5)),
                                                            float(np.percentile(bs, 97.5)), len(rows), len(reo[side])]
        print(f"      {side} vs {refname:24s} {ratio(rows):.2f} [{np.percentile(bs, 2.5):.2f}, "
              f"{np.percentile(bs, 97.5):.2f}]  ({len(rows)} of {len(reo[side])} compactions on games with >= 3 reference)")

    # Between-run spread of the tail events (each run against the thinking-off runs, itself excluded)
    from scipy.stats import poisson

    print("\n## 2f. Tail events, each run vs the thinking-off runs (itself left out); strata game x level x level-tokens")
    res["tail_spread"] = {}
    strat3 = stratifications["game x level x level-tokens"]
    for ename, ev in tail_events[:5] + tail_events[5:6]:
        line = []
        for side in runs:
            ref = [r for r in OFF if r in runs and r != side]
            cnt, n = defaultdict(float), defaultdict(int)
            for r in G:
                if r["run"] in ref:
                    cnt[strat3(r)] += ev(r)
                    n[strat3(r)] += 1
            o = e = 0.0
            for r in G:
                if r["run"] == side and n.get(strat3(r), 0) >= 20:
                    o += ev(r)
                    e += cnt[strat3(r)] / n[strat3(r)]
            res["tail_spread"][f"{ename}|{side}"] = [o, e, poisson.sf(o - 1, e) if e else float("nan")]
            line.append(f"{side} {o:.0f}/{e:.1f}")
        print(f"  {ename:28s} " + "; ".join(line))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=1, default=float))


def mh_rate_ratio(rows, exposed, event, exposure, stratum):
    """Mantel-Haenszel rate ratio (exposed vs unexposed) with the Greenland-Robins 95% CI."""
    S = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # a (exp events), T1, b (unexp events), T0
    for r in rows:
        s = S[stratum(r)]
        if exposed(r):
            s[0] += event(r)
            s[1] += exposure(r)
        else:
            s[2] += event(r)
            s[3] += exposure(r)
    num = den = var_num = 0.0
    a_tot = b_tot = 0.0
    for a, t1, b, t0 in S.values():
        t = t1 + t0
        if t1 <= 0 or t0 <= 0:
            continue
        num += a * t0 / t
        den += b * t1 / t
        var_num += (a + b) * t1 * t0 / t ** 2
        a_tot += a
        b_tot += b
    if num <= 0 or den <= 0:
        return float("nan"), float("nan"), float("nan"), a_tot, b_tot
    rr = num / den
    se = math.sqrt(var_num / (num * den))
    return rr, rr * math.exp(-1.96 * se), rr * math.exp(1.96 * se), a_tot, b_tot


if __name__ == "__main__":
    main()
