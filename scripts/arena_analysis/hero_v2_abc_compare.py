"""hero-v2 against run 1, run A and run B on the 110 c110 clones (29 September; CPU only, recorded runs).

Writes the numbers behind docs/validation-20260927/hero-v2-abc-compare.md:

    python scripts/arena_analysis/hero_v2_abc_compare.py --out docs/validation-20260927/hero-v2-abc-compare.json

Sections (keys of the JSON):
- ``timeline``: the rebuilt score and levels every 900 s (checked against the live scorecard reads);
- ``families``: per source game at 1 h and 2 h: score points, levels, clones with a level, deepest clone, actions
  against baseline on the cleared levels, points lost to action efficiency;
- ``noise``: clone bootstrap within family, family bootstrap (paired across runs), leave-one-family-out, the score
  without the top family / without re86 and sb26, within-family clone spread;
- ``depth``: clears and score by level index, tokens by the level being played, per-level token cost;
- ``hazard``: clears per million tokens by level, and the Mantel-Haenszel rate ratio against hero-v2 stratified by
  (family, level index), with a family-bootstrap interval, at 1 h and 2 h;
- ``speed``: engine throughput, running requests and per-stream decode rate (inter-token latency) by phase;
- ``deep_games``: time and tokens to each level for hero-v2's deepest clones and each family's best clone;
- ``scheduling``: admissions, parks, fruitless tenures, backlog waits, level-0 auditions;
- ``harness``: every harness_fixes event and the hero-v2 audit's failure modes, per run;
- ``behavior``: per-reply reasoning, tokens per action, actions per reply, cap hits, first hour and whole run;
- ``context``: 64K against 128K: compactions, prompt sizes, prefix-cache share, clears around compactions;
- ``projection``: run 1 at 2 h and each run's 9 h trend (second-hour slope, late-game progress).
"""

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abc_load import load_all  # noqa: E402

LABELS = ("hero-v2", "run1", "runA", "runB")
TWO_HOUR = ("hero-v2", "runA", "runB")
H1, H2 = 3600.0, 7200.0
BOOT = 2000
SEED = 20260929


def r(x, n=3):
    return None if x is None else round(float(x), n)


def quant(values, q):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def dist(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": r(statistics.fmean(values), 1), "p10": r(quant(values, 0.1), 1),
            "median": r(quant(values, 0.5), 1), "p90": r(quant(values, 0.9), 1), "max": r(max(values), 1)}


def horizons(label):
    return (H1,) if label == "run1" else (H1, H2)


# ---------------------------------------------------------------------------------------------------------------
def timeline(runs):
    out = {}
    for label, run in runs.items():
        rows = []
        for t in range(900, int(run.wall) + 1, 900):
            live = run.live_at(t)
            rows.append({"t": t, "score": r(run.score_at(t)), "levels": sum(g.levels_at(t) for g in run.games.values()),
                         "live_read": live})
        out[label] = {"rows": rows, "final": r(run.scorecard["score"]), "wall": run.wall,
                      "switch": run.arena["scheduler"].get("switch_wall_seconds"),
                      "streams": run.arena["scheduler"].get("active_limit")}
    return out


def family_rows(run, t):
    rows = {}
    for fam, games in run.families().items():
        pts = sum(g.score_at(t) for g in games) / 110
        levels = sum(g.levels_at(t) for g in games)
        acts = base = 0
        depth_only = 0.0
        for g in games:
            cleared = [c for c in g.clears if c["wall"] <= t]
            for c in cleared:
                acts += g.level_actions[c["level"] - 1]
                base += g.baselines[c["level"] - 1]
            n = len(cleared)
            depth_only += (n * (n + 1) / 2) / (g.n_levels * (g.n_levels + 1) / 2) * 100 if n else 0
        rows[fam] = {"points": r(pts), "levels": levels, "clones": len(games),
                     "clones_with_level": sum(1 for g in games if g.levels_at(t) > 0),
                     "deepest": max(g.levels_at(t) for g in games), "n_levels": games[0].n_levels,
                     "clone_levels": sorted((g.levels_at(t) for g in games), reverse=True),
                     "actions_over_baseline": r(acts / base, 2) if base else None,
                     "efficiency_loss_points": r(depth_only / 110 - pts)}
    return rows


def families(runs):
    out = {}
    for label, run in runs.items():
        out[label] = {str(int(t)): family_rows(run, t) for t in horizons(label)}
    return out


# ---------------------------------------------------------------------------------------------------------------
def clone_scores(run, t):
    fam = defaultdict(list)
    for g in run.games.values():
        fam[g.family].append(g.score_at(t))
    return dict(sorted(fam.items()))


def noise(runs):
    rng = random.Random(SEED)
    out = {"clone_bootstrap": {}, "family_bootstrap_paired": {}, "lofo": {}, "without": {}, "clone_spread": {}}
    cs = {(label, t): clone_scores(run, t) for label, run in runs.items() for t in horizons(label)}
    fams = list(next(iter(cs.values())).keys())

    # 1. clone bootstrap within family (the clone-to-clone luck with the same 20 games)
    boots = {}
    for key, fam in cs.items():
        vals = []
        for _ in range(BOOT):
            total = 0.0
            for f in fams:
                xs = fam[f]
                total += sum(rng.choice(xs) for _ in xs)
            vals.append(total / 110)
        boots[key] = vals
        out["clone_bootstrap"][f"{key[0]}@{int(key[1])}"] = {
            "score": r(sum(sum(v) for v in fam.values()) / 110), "se": r(statistics.pstdev(vals)),
            "p05": r(quant(vals, 0.05)), "p95": r(quant(vals, 0.95))}
    pairs = [(a, b, t) for t in (H1, H2) for a in LABELS for b in LABELS
             if a < b and (a, t) in cs and (b, t) in cs]
    for a, b, t in pairs:
        d = [x - y for x, y in zip(boots[(a, t)], boots[(b, t)])]
        out["clone_bootstrap"][f"{a}-{b}@{int(t)}"] = {"diff": r(statistics.fmean(d)), "se": r(statistics.pstdev(d)),
                                                        "p_gt0": r(sum(x > 0 for x in d) / len(d), 3)}

    # 2. family bootstrap, paired: a new set of 20 games (the hidden-set question), same draw for every run
    draws = [[rng.choice(fams) for _ in fams] for _ in range(BOOT)]
    fb = {}
    for key, fam in cs.items():
        per = {f: sum(v) for f, v in fam.items()}
        fb[key] = [sum(per[f] for f in draw) / 110 for draw in draws]
        out["family_bootstrap_paired"][f"{key[0]}@{int(key[1])}"] = {
            "se": r(statistics.pstdev(fb[key])), "p05": r(quant(fb[key], 0.05)), "p95": r(quant(fb[key], 0.95))}
    for a, b, t in pairs:
        d = [x - y for x, y in zip(fb[(a, t)], fb[(b, t)])]
        out["family_bootstrap_paired"][f"{a}-{b}@{int(t)}"] = {
            "diff": r(statistics.fmean(d)), "se": r(statistics.pstdev(d)), "p_gt0": r(sum(x > 0 for x in d) / len(d)),
            "p05": r(quant(d, 0.05)), "p95": r(quant(d, 0.95))}

    # 3. leave one family out, and without the top families
    for a, b, t in pairs:
        pa = {f: sum(v) / 110 for f, v in cs[(a, t)].items()}
        pb = {f: sum(v) / 110 for f, v in cs[(b, t)].items()}
        full = sum(pa.values()) - sum(pb.values())
        lofo = {f: full - (pa[f] - pb[f]) for f in fams}
        lo = min(lofo, key=lofo.get)
        hi = max(lofo, key=lofo.get)
        out["lofo"][f"{a}-{b}@{int(t)}"] = {"diff": r(full), "min": [lo, r(lofo[lo])], "max": [hi, r(lofo[hi])],
                                            "sign_flips": sorted(f for f in fams if (lofo[f] > 0) != (full > 0))}
    for key, fam in cs.items():
        per = {f: sum(v) / 110 for f, v in fam.items()}
        top = max(per, key=per.get)
        out["without"][f"{key[0]}@{int(key[1])}"] = {
            "score": r(sum(per.values())), "top_family": top, "top_points": r(per[top]),
            "without_top": r(sum(per.values()) - per[top]),
            "without_re86": r(sum(v for f, v in per.items() if f != "re86")),
            "without_sb26": r(sum(v for f, v in per.items() if f != "sb26")),
            "without_re86_sb26": r(sum(v for f, v in per.items() if f not in ("re86", "sb26"))),
            "top_clone_share": r(max(max(v) for v in fam.values()) / 110 / max(1e-9, sum(per.values())))}

    # 4. clone spread within family: how much of a family's points sit in its best clone
    for key, fam in cs.items():
        rows = {}
        for f, v in fam.items():
            if sum(v) > 0:
                rows[f] = {"clone_scores": [r(x, 1) for x in sorted(v, reverse=True)],
                           "best_share": r(max(v) / sum(v), 2)}
        out["clone_spread"][f"{key[0]}@{int(key[1])}"] = rows
    return out


# ---------------------------------------------------------------------------------------------------------------
def attempts(run, t):
    """(family, level index, tokens on that level up to t, cleared by t) per game and level attempted."""
    rows = []
    for g in run.games.values():
        tok = defaultdict(int)
        for rep in g.replies:
            if rep["wall"] <= t:
                tok[rep["level"] + 1] += rep["completion"]
        cleared = {c["level"] for c in g.clears if c["wall"] <= t}
        for lvl, tokens in tok.items():
            if lvl > g.n_levels:
                continue
            rows.append({"game": g.id, "family": g.family, "level": lvl, "tokens": tokens,
                         "cleared": lvl in cleared})
    return rows


def depth(runs):
    out = {}
    for label, run in runs.items():
        per = {}
        for t in horizons(label):
            clears_by_level = Counter()
            points_by_level = defaultdict(float)
            for g in run.games.values():
                for c in g.clears:
                    if c["wall"] <= t:
                        k = c["level"]
                        clears_by_level[k] += 1
                        ls = min(100.0, 100.0 * (g.baselines[k - 1] / g.level_actions[k - 1]) ** 2)
                        points_by_level[k] += k * ls / (g.n_levels * (g.n_levels + 1) / 2) / 110
            att = attempts(run, t)
            tok_by_level = defaultdict(int)
            for a in att:
                tok_by_level[a["level"]] += a["tokens"]
            total = sum(tok_by_level.values())
            cost = defaultdict(list)
            for a in att:
                if a["cleared"]:
                    cost[a["level"]].append(a["tokens"])
            depth_games = Counter(g.levels_at(t) for g in run.games.values())
            per[str(int(t))] = {
                "clears_by_level": dict(sorted(clears_by_level.items())),
                "points_by_level": {k: r(v) for k, v in sorted(points_by_level.items())},
                "token_share_by_level_played": {k: r(v / total, 3) for k, v in sorted(tok_by_level.items())},
                "tokens_total_M": r(total / 1e6, 2),
                "tokens_per_clear_by_level": {k: dist(v) for k, v in sorted(cost.items())},
                "games_by_depth": dict(sorted(depth_games.items())),
                "levels_3plus": sum(v for k, v in clears_by_level.items() if k >= 3),
                "points_levels_3plus": r(sum(v for k, v in points_by_level.items() if k >= 3)),
            }
        out[label] = per
    return out


def mh_ratio(rows_x, rows_ref):
    strata = defaultdict(lambda: [0, 0.0, 0, 0.0])
    for a in rows_x:
        s = strata[(a["family"], min(a["level"], 4))]
        s[0] += a["cleared"]
        s[1] += a["tokens"]
    for a in rows_ref:
        s = strata[(a["family"], min(a["level"], 4))]
        s[2] += a["cleared"]
        s[3] += a["tokens"]
    num = den = 0.0
    for ex, tx, er, tr in strata.values():
        if tx <= 0 or tr <= 0:
            continue
        tt = tx + tr
        num += ex * tr / tt
        den += er * tx / tt
    return num / den if den > 0 else None


def hazard(runs):
    rng = random.Random(SEED + 1)
    out = {"rates": {}, "rate_ratio_vs_hero_v2": {}, "int4_128k_vs_64k": {}}
    att = {(label, t): attempts(run, t) for label, run in runs.items() for t in horizons(label)}
    for (label, t), rows in att.items():
        by = defaultdict(lambda: [0, 0])
        for a in rows:
            k = min(a["level"], 4)
            by[k][0] += a["cleared"]
            by[k][1] += a["tokens"]
        out["rates"][f"{label}@{int(t)}"] = {
            ("L" + str(k) if k < 4 else "L4+"): {"clears": c, "tokens_M": r(tk / 1e6, 2),
                                                "clears_per_M": r(c / tk * 1e6, 2) if tk else None}
            for k, (c, tk) in sorted(by.items())}
    fams = sorted({a["family"] for rows in att.values() for a in rows})

    def boot(xs, refs):
        by_x = defaultdict(list)
        by_r = defaultdict(list)
        for a in xs:
            by_x[a["family"]].append(a)
        for a in refs:
            by_r[a["family"]].append(a)
        vals = []
        for _ in range(500):
            draw = [rng.choice(fams) for _ in fams]
            v = mh_ratio([a for f in draw for a in by_x[f]], [a for f in draw for a in by_r[f]])
            if v is not None:
                vals.append(v)
        return vals

    for label in ("run1", "runA", "runB"):
        for t in horizons(label):
            for lv in ("all", "L2+"):
                xs = [a for a in att[(label, t)] if lv == "all" or a["level"] >= 2]
                refs = [a for a in att[("hero-v2", t)] if lv == "all" or a["level"] >= 2]
                point = mh_ratio(xs, refs)
                vals = boot(xs, refs)
                out["rate_ratio_vs_hero_v2"][f"{label}@{int(t)}:{lv}"] = {
                    "mh": r(point, 2), "p05": r(quant(vals, 0.05), 2), "p95": r(quant(vals, 0.95), 2)}
    # 128K (run 1 + B) against 64K (hero-v2 + A), at 1 h and (B vs hero-v2 + A) at 2 h
    for t, big, small in ((H1, ("run1", "runB"), ("hero-v2", "runA")), (H2, ("runB",), ("hero-v2", "runA"))):
        for lv in ("all", "L2+"):
            xs = [a for lab in big for a in att[(lab, t)] if lv == "all" or a["level"] >= 2]
            refs = [a for lab in small for a in att[(lab, t)] if lv == "all" or a["level"] >= 2]
            vals = boot(xs, refs)
            out["int4_128k_vs_64k"][f"{'+'.join(big)} vs {'+'.join(small)}@{int(t)}:{lv}"] = {
                "mh": r(mh_ratio(xs, refs), 2), "p05": r(quant(vals, 0.05), 2), "p95": r(quant(vals, 0.95), 2)}
    return out


# ---------------------------------------------------------------------------------------------------------------
def telemetry(run):
    rows = []
    with (run.dir / "telemetry.jsonl").open() as fh:
        for line in fh:
            rec = json.loads(line)
            m = rec.get("metrics") or {}
            rows.append((rec["elapsed_seconds"], m))
    return rows


def speed(runs):
    out = {}
    for label, run in runs.items():
        rows = telemetry(run)
        switch = run.arena["scheduler"].get("switch_wall_seconds") or 0
        spans = {"opening": (0, switch), "phase2_first_hour": (switch, H1), "phase2_second_hour": (H1, H2),
                 "phase2": (switch, H2), "whole": (0, H2)}
        per = {}
        for name, (a, b) in spans.items():
            seg = [(t, m) for t, m in rows if a <= t <= b]
            if len(seg) < 3:
                continue
            (t0, m0), (t1, m1) = seg[0], seg[-1]
            dt = t1 - t0
            gen = m1.get("vllm:generation_tokens_total", 0) - m0.get("vllm:generation_tokens_total", 0)
            itl_n = m1.get("vllm:inter_token_latency_seconds_count", 0) - m0.get(
                "vllm:inter_token_latency_seconds_count", 0)
            itl_s = m1.get("vllm:inter_token_latency_seconds_sum", 0) - m0.get("vllm:inter_token_latency_seconds_sum", 0)
            comp = m1.get("vllm:prompt_tokens_by_source_total:local_compute", 0) - m0.get(
                "vllm:prompt_tokens_by_source_total:local_compute", 0)
            hit = m1.get("vllm:prompt_tokens_by_source_total:local_cache_hit", 0) - m0.get(
                "vllm:prompt_tokens_by_source_total:local_cache_hit", 0)
            running = [m.get("vllm:num_requests_running", 0) for _, m in seg]
            kv = [m.get("vllm:kv_cache_usage_perc", 0) for _, m in seg]
            per[name] = {"seconds": r(dt, 0), "gen_tok_s": r(gen / dt, 1), "mean_running": r(statistics.fmean(running), 1),
                         "per_stream_tok_s_itl": r(itl_n / itl_s, 1) if itl_s else None,
                         "gen_per_running": r(gen / dt / max(1e-9, statistics.fmean(running)), 1),
                         "computed_prompt_tok_s": r(comp / dt, 0),
                         "cache_hit_share": r(hit / (hit + comp), 3) if hit + comp else None,
                         "kv_peak": r(max(kv), 3)}
        # mean prompt length of phase-2 gameplay requests (context length drives per-token cost)
        prompts = [rep["prompt"] for g in run.games.values() for rep in g.replies
                   if rep["purpose"] == "gameplay" and rep["wall"] >= switch]
        per["phase2_prompt_tokens"] = dist(prompts)
        out[label] = per
    return out


# ---------------------------------------------------------------------------------------------------------------
DEEP_HV2 = ("k015", "k035", "k095", "k025", "k067", "k075", "k005", "k105")


def deep_games(runs):
    out = {"hero_v2_deep_clones": {}, "family_best": {}}
    for gid in DEEP_HV2:
        row = {}
        for label, run in runs.items():
            g = run.games[gid]
            active = [e for e in g.events if e["event"] == "admit"]
            row[label] = {"clears": [(c["level"], r(c["wall"], 0), r(c["tokens"] / 1e3, 1)) for c in g.clears],
                          "levels_1h": g.levels_at(H1), "levels_2h": g.levels_at(H2),
                          "tokens_1h_K": r(g.tokens_at(H1) / 1e3, 1), "tokens_2h_K": r(g.tokens_at(H2) / 1e3, 1),
                          "admits": len(active), "family": g.family}
        out["hero_v2_deep_clones"][gid] = row
    for label, run in runs.items():
        rows = {}
        for fam, games in run.families().items():
            best = max(games, key=lambda g: (g.levels_at(H2), -min([c["wall"] for c in g.clears] or [1e9])))
            if best.levels_at(H2) == 0:
                continue
            rows[fam] = {"clone": best.id, "levels_1h": best.levels_at(H1), "levels_2h": best.levels_at(H2),
                         "clear_walls": [r(c["wall"], 0) for c in best.clears],
                         "clear_tokens_K": [r(c["tokens"] / 1e3, 1) for c in best.clears],
                         "tokens_2h_K": r(best.tokens_at(H2) / 1e3, 1)}
        out["family_best"][label] = rows
    return out


# ---------------------------------------------------------------------------------------------------------------
def scheduling(runs):
    out = {}
    for label, run in runs.items():
        sch = run.arena["scheduler"]
        switch = sch.get("switch_wall_seconds") or 0
        ev = sch["events"]
        c = Counter(f"{e['event']}:{e['reason']}" for e in ev)
        by_game = defaultdict(list)
        for e in ev:
            by_game[e["game"]].append(e)
        tenures, fruitless, waits_deep, waits_l0 = [], 0, [], []
        l0_games_phase2 = set()
        l0_audition_tokens = []
        phase2_game_set = set()
        for gid, es in by_game.items():
            g = run.games[gid]
            es = sorted(es, key=lambda e: e["wall_seconds"])
            open_admit, last_park = None, None
            for e in es:
                if e["event"] == "admit" and e["wall_seconds"] >= switch - 1:
                    if last_park is not None:
                        w = e["wall_seconds"] - last_park["wall_seconds"]
                        (waits_deep if last_park["levels"] >= 1 else waits_l0).append(w)
                    open_admit = e
                    phase2_game_set.add(gid)
                elif e["event"] in ("park", "finish") and open_admit is not None:
                    a, b = open_admit["wall_seconds"], e["wall_seconds"]
                    lv0 = g.levels_at(a)
                    gained = g.levels_at(b) - lv0
                    toks = sum(rep["completion"] for rep in g.replies if a <= rep["wall"] <= b)
                    tenures.append({"game": gid, "start": a, "end": b, "levels_at_start": lv0, "gained": gained,
                                    "tokens": toks, "reason": e["reason"]})
                    if e["event"] == "park" and e["reason"] == "level_budget" and gained == 0:
                        fruitless += 1
                    if lv0 == 0:
                        l0_games_phase2.add(gid)
                        l0_audition_tokens.append(toks)
                    open_admit = None
                if e["event"] == "park":
                    last_park = e
        p2 = [(g, rep) for g in run.games.values() for rep in g.replies if rep["wall"] > switch]
        phase2_tokens = sum(rep["completion"] for _, rep in p2)
        l0_tok = sum(rep["completion"] for _, rep in p2 if rep["level"] == 0)
        deep_tok = sum(rep["completion"] for _, rep in p2 if rep["level"] >= 2)
        dead_tok = sum(rep["completion"] for g, rep in p2 if rep["level"] == 0 and g.levels_at(run.wall) == 0)
        # levels at 2 h gained by level-0 auditions in phase 2
        l0_clears = sum(1 for t in tenures if t["levels_at_start"] == 0 and t["gained"] > 0)
        out[label] = {
            "events": dict(sorted(c.items())), "same_tick_regrants": sch.get("same_tick_regrants"),
            "backlog_key": sch.get("backlog_key"), "level0_order": sch.get("level0_order"),
            "phase2_tenures": len(tenures), "fruitless_level_budget_parks": fruitless,
            "phase2_games": len(phase2_game_set),
            "level0_games_auditioned_phase2": len(l0_games_phase2),
            "level0_auditions_that_cleared": l0_clears,
            "phase2_tokens_M": r(phase2_tokens / 1e6, 2),
            "phase2_token_share_on_level1": r(l0_tok / phase2_tokens, 3) if phase2_tokens else None,
            "phase2_token_share_on_level3plus": r(deep_tok / phase2_tokens, 3) if phase2_tokens else None,
            "phase2_token_share_level0_games_never_cleared": r(dead_tok / phase2_tokens, 3) if phase2_tokens else None,
            "level0_audition_tokens_K": dist([x / 1e3 for x in l0_audition_tokens]),
            "readmission_wait_s_levels_ge1": dist(waits_deep), "readmission_wait_s_level0": dist(waits_l0),
            "tenure_tokens_K_levels_ge1": dist([t["tokens"] / 1e3 for t in tenures if t["levels_at_start"] >= 1]),
        }
    return out


# ---------------------------------------------------------------------------------------------------------------
HF_EVENTS = ("compaction", "compaction_inventory", "compaction_guard", "compaction_rejected", "aside_cap_fitted",
             "length_truncation", "cap_hit_followup", "stuck_notice", "auto_reset", "action_skipped",
             "duck_python_interrupt", "duck_python_interrupted", "knowledge_retained", "transfer_notes",
             "exact_trigger_fallback", "deadline_request_timeout", "worker_request_cancelled", "summary_unavailable")


def harness(runs):
    out = {}
    for label, run in runs.items():
        counts = Counter()
        hf = Counter()
        for g in run.games.values():
            for k, v in g.log_counts.items():
                counts[k] += v
            for k, v in (g.row.get("harness_fixes") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    hf[k] += v
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                            hf[f"{k}.{kk}"] += vv
        py = [e for g in run.games.values() for e in g.log_events["python"]]
        errs = [e for e in py if e["error"]]
        blocked = [e for e in py if "not allowed" in e["error"] or "not installed" in e["error"]
                   or "is not defined" in e["error"] and any(b in e["error"] for b in ("'open'", "'globals'", "'id'"))]
        timeouts = [e for e in py if "timed out" in (e["error"] + e["stdout_tail"]).lower()]
        resets = [e for e in py if "state was reset" in (e["error"] + e["stdout_tail"])]
        comp = [e for g in run.games.values() for e in g.log_events["compaction"]]
        window = run.arena["window_choice"]["window"]
        # stuck notices: did the model RESET within 5 steps; was the level cleared later
        stuck_follow = {"notices": 0, "reset_within_5_steps": 0, "level_cleared_later": 0}
        for g in run.games.values():
            for e in g.log_events["stuck_notice"]:
                stuck_follow["notices"] += 1
                lvl = e["level"]
                if any(c["level"] == lvl + 1 for c in g.clears):
                    stuck_follow["level_cleared_later"] += 1
                if any(a["step"] - e["step"] <= 5 and a["step"] >= e["step"]
                       for a in g.log_events.get("auto_reset", [])):
                    stuck_follow["reset_within_5_steps"] += 1
        rejected = [{"game": g.id, "step": e["step"], "detail": e.get("detail")}
                    for g in run.games.values() for e in g.log_events["compaction_rejected"]]
        guards = [{"game": g.id, "step": e["step"], "prompt_tokens": e.get("prompt_tokens")}
                  for g in run.games.values() for e in g.log_events["compaction_guard"]]
        replies = sum(1 for g in run.games.values() for rep in g.replies if rep["purpose"] == "gameplay")
        out[label] = {
            "window": window, "event_counts": {k: counts.get(k, 0) for k in HF_EVENTS},
            "harness_fixes_counters": dict(sorted(hf.items())),
            "python_calls": len(py), "python_errors": len(errs), "python_error_rate": r(len(errs) / max(1, len(py)), 3),
            "blocked_imports_or_modules": len(blocked), "tool_timeouts": len(timeouts), "state_resets": len(resets),
            "cap_hits_per_1k_replies": r(1000 * counts.get("length_truncation", 0) / max(1, replies), 2),
            "compaction_prompt_tokens": dist([e.get("prompt_tokens") for e in comp]),
            "compaction_prompt_over_window_0.9": sum(1 for e in comp if (e.get("prompt_tokens") or 0) > 0.9 * window),
            "mechanical_fallbacks": sum(1 for e in comp if e.get("mechanical_fallback")),
            "inventory_chars": dist([e.get("chars") for g in run.games.values()
                                     for e in g.log_events["compaction_inventory"]]),
            "stuck_notice_follow": stuck_follow, "compaction_rejected": rejected, "compaction_guards": guards,
        }
    return out


# ---------------------------------------------------------------------------------------------------------------
def behavior(runs):
    out = {}
    for label, run in runs.items():
        per = {}
        for name, t in (("first_hour", H1), ("whole", H2)):
            if label == "run1" and name == "whole":
                continue
            reps = [rep for g in run.games.values() for rep in g.replies if rep["purpose"] == "gameplay"
                    and rep["wall"] <= t]
            acts = sum(rep.get("actions", 0) for rep in reps)
            comp_tokens = sum(rep["completion"] for rep in reps)
            per[name] = {
                "replies": len(reps), "actions": acts, "completion_M": r(comp_tokens / 1e6, 2),
                "reasoning_tokens": dist([rep["reasoning"] for rep in reps]),
                "completion_per_action": r(comp_tokens / max(1, acts), 0),
                "actions_per_reply": r(acts / max(1, len(reps)), 2),
                "zero_action_share": r(sum(1 for rep in reps if rep.get("actions", 0) == 0) / max(1, len(reps)), 3),
                "cap_hits": sum(1 for rep in reps if rep["finish"] == "length"),
                "reasoning_share": r(sum(rep["reasoning"] for rep in reps) / max(1, comp_tokens), 3),
                "seconds_per_reply": dist([rep["seconds"] for rep in reps if rep["seconds"] is not None]),
            }
            # standardized on (level index): completion per action at level 1, 2, 3+
            lv = defaultdict(lambda: [0, 0])
            for rep in reps:
                k = min(rep["level"] + 1, 3)
                lv[k][0] += rep["completion"]
                lv[k][1] += rep.get("actions", 0)
            per[name]["completion_per_action_by_level"] = {
                ("L" + str(k) if k < 3 else "L3+"): r(a / max(1, b), 0) for k, (a, b) in sorted(lv.items())}
        out[label] = per
    return out


# ---------------------------------------------------------------------------------------------------------------
def context(runs, audits):
    out = {}
    for label, run in runs.items():
        comps = [(g, e) for g in run.games.values() for e in g.log_events["compaction"]]
        comp_tokens = sum(rep["completion"] for g in run.games.values() for rep in g.replies
                          if rep["purpose"] == "compaction")
        total = sum(rep["completion"] for g in run.games.values() for rep in g.replies)
        # clear rate in the first 30K completion tokens after a compaction against the rest, levels >= 2 only
        # (on the same games: a clear counts in the segment it happened in)
        seg = {"post": [0, 0], "other": [0, 0]}
        for g in run.games.values():
            comp_walls = sorted(e["wall"] for e in g.log_events["compaction"])
            clear_walls = {c["wall"] for c in g.clears}
            since, cur_level = None, None
            for rep in g.replies:
                if rep["level"] != cur_level:
                    cur_level, since = rep["level"], None
                if rep["purpose"] == "compaction":
                    since = 0
                    continue
                key = "post" if since is not None and since < 30000 else "other"
                if since is not None:
                    since += rep["completion"]
                if rep["level"] >= 1:
                    seg[key][1] += rep["completion"]
            for c in g.clears:
                if c["level"] < 2:
                    continue
                prior = [w for w in comp_walls if w <= c["wall"]]
                toks = sum(rep["completion"] for rep in g.replies
                           if prior and prior[-1] < rep["wall"] <= c["wall"] and rep["level"] == c["level"] - 1)
                seg["post" if prior and toks < 30000 and any(
                    rep["purpose"] == "compaction" and rep["level"] == c["level"] - 1 and rep["wall"] == prior[-1]
                    for rep in g.replies) else "other"][0] += 1
        post_rate = {k: {"clears": v[0], "tokens_M": r(v[1] / 1e6, 2),
                         "clears_per_M": r(v[0] / v[1] * 1e6, 2) if v[1] else None} for k, v in seg.items()}
        au = audits.get(label) or {}
        reg = au.get("reground") or {}
        out[label] = {
            "window": run.arena["window_choice"]["window"], "compactions": len(comps),
            "games_compacted": len({g.id for g, _ in comps}),
            "compaction_tokens_share": r(comp_tokens / max(1, total), 3),
            "reground_excess_tokens_K": r((reg.get("excess_tokens") or 0) / 1e3, 0),
            "reground_compactions": reg.get("compactions"),
            "reground_share_of_tokens": r((reg.get("excess_tokens") or 0) / max(1, total), 3),
            "compactions_by_level": dict(sorted(Counter(e["level"] + 1 for _, e in comps).items())),
            "clears_L2plus_post_compaction_30K_vs_other": post_rate,
            "engine_cache_share": r(run.arena["engine"].get("cached_prompt_tokens", 0) /
                                    max(1, run.arena["engine"].get("cached_prompt_tokens", 0) +
                                        run.arena["engine"].get("computed_prompt_tokens", 0)), 3),
            "computed_prompt_M": r(run.arena["engine"].get("computed_prompt_tokens", 0) / 1e6, 2),
        }
    return out


# ---------------------------------------------------------------------------------------------------------------
def projection(runs, fam):
    out = {}
    s = {label: {t: runs[label].score_at(t) for t in horizons(label)} for label in runs}
    gains = {label: s[label][H2] - s[label][H1] for label in TWO_HOUR}
    out["second_hour_gain"] = {k: r(v) for k, v in gains.items()}
    out["slope_by_30min"] = {
        label: [r(runs[label].score_at(t2) - runs[label].score_at(t1)) for t1, t2 in
                ((0, 1800), (1800, 3600), (3600, 5400), (5400, 7200))] for label in TWO_HOUR}
    r1 = s["run1"][H1]
    out["run1_2h_simple"] = {"plus_B_gain": r(r1 + gains["runB"]), "plus_mean_gain": r(r1 + statistics.fmean(gains.values())),
                             "plus_min_gain": r(r1 + min(gains.values())), "plus_max_gain": r(r1 + max(gains.values()))}
    # per family: run 1's 1 h points + the mean second-hour family gain of the 2 h runs, capped at the family max
    fams = runs["hero-v2"].families()
    per, total_mean, total_B = {}, 0.0, 0.0
    for f, games in fams.items():
        cap = len(games) * 100 / 110
        p1 = fam["run1"]["3600"][f]["points"]
        g_mean = statistics.fmean(fam[l]["7200"][f]["points"] - fam[l]["3600"][f]["points"] for l in TWO_HOUR)
        g_B = fam["runB"]["7200"][f]["points"] - fam["runB"]["3600"][f]["points"]
        # a family's second-hour gain depends on where it stands: take the gain of the 2 h run whose 1 h family
        # points are closest to run 1's
        nearest = min(TWO_HOUR, key=lambda l: abs(fam[l]["3600"][f]["points"] - p1))
        g_near = fam[nearest]["7200"][f]["points"] - fam[nearest]["3600"][f]["points"]
        per[f] = {"run1_1h": p1, "gain_mean": r(g_mean), "gain_B": r(g_B), "nearest": nearest, "gain_nearest": r(g_near)}
        total_mean += min(cap, p1 + g_mean)
        total_B += min(cap, p1 + g_B)
    out["run1_2h_per_family"] = {"mean_gain": r(total_mean), "B_gain": r(total_B),
                                 "nearest_state_gain": r(sum(min(len(fams[f]) * 100 / 110, v["run1_1h"] + v["gain_nearest"])
                                                             for f, v in per.items())),
                                 "families": per}
    # sb26: run 1's k077 held 7 of 8 levels at 1 h; its family's remaining upside
    g = runs["run1"].games["k077"]
    out["run1_sb26_k077"] = {"levels": g.levels_at(H1), "n_levels": g.n_levels, "points": r(g.score_at(H1) / 110),
                             "max_points": r(100 / 110)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--audit-dir", type=Path, help="harness_audit.py outputs audit-<label>.json (for reground)")
    args = ap.parse_args()
    runs = load_all(LABELS)
    audits = {}
    if args.audit_dir:
        for label in LABELS:
            p = args.audit_dir / f"audit-{label}.json"
            if p.is_file():
                audits[label] = json.loads(p.read_text())
    fam = families(runs)
    result = {
        "runs": {label: run.run_id for label, run in runs.items()},
        "timeline": timeline(runs), "families": fam, "noise": noise(runs), "depth": depth(runs),
        "hazard": hazard(runs), "speed": speed(runs), "deep_games": deep_games(runs),
        "scheduling": scheduling(runs), "harness": harness(runs), "behavior": behavior(runs),
        "context": context(runs, audits), "projection": projection(runs, fam),
    }
    text = json.dumps(result, indent=1, default=str)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
