"""RL v7 (hero) exporter -> <scratch>/train-metrics/<run>.jsonl every 120 s, on arc_infra.runops.metrics (reference
migration, infra stage 3; was scratch rl-arm/hero/hero-export.py + .rl-tools/gate/metrics_export.py).
Infra rows from the RL watcher's pass files in --passes (<run>-watch-passes.jsonl: vLLM metrics of both servers;
<run>-card-util.jsonl: card util/VRAM); learning rows from the run's feed on the volume (step, accumulate, group_done,
read, wread, envs). Usage: export_rl.py RUN --passes DIR [--base=RUN0] [--once]"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paths  # noqa: E402

sys.path.insert(0, str(paths.ROOT_PATH / "src"))
from arc_infra.runops import metrics  # noqa: E402

MODAL = str(paths.ROOT_PATH / ".venv/bin/modal")
CAP_GIB = 268.6  # B300 usable (nvidia-smi total)
B300_USD_S = 0.001972  # Modal B300 per GPU-second (records gpu_rate_per_second, hero-v6g)
FEED = "validation/{run}/arena/trials/rl-loop/rl/feed.jsonl"
GROUP = "validation/{run}/arena/trials/rl-loop/rl/{gid}-{set}/level-bench.json"
PARTIAL = "validation/{run}/partial/v7/{gid}-{set}/tries.jsonl"  # a live run syncs finished tries here
SAMPLER = "validation/{run}/live/sampler.jsonl"  # rl_seq runs: the runops sampler (no RL-watcher pass files)
CPULOG = "validation/{run}/util/cpulog.txt"
OUT = {}  # run -> its volume folder (--out: a re-reserved run writes validation/<run>-attempt<N>, v6k 25 Oct)
NEW_KEYS = ("train.rhae_tok_vs_ref", "train.clear_rate_vs_ref", "train.fail_rate_vs_ref", "train.tokens_per_try_vs_ref", "train.ref_coverage", "train.rhae_per_10k", "train.rhae_mean", "train.clear_rate", "train.tokens_per_try", "train.actions_per_clear",
            "play.round", "play.L", "eval.rhae_per_10k", "eval.rhae_mean", "eval.clears_frac", "eval.tokens_per_try",
            "infra.cost_usd")


def try_rows(text):
    """Per-try (level, tokens, cleared, actions, rhae) of a group's level-bench.json; RHAE = level_score x total_levels / 100."""
    text = text.lstrip()
    try:  # the Modal CLI appends a status line to stdout after the file
        d = json.JSONDecoder().raw_decode(text)[0] if text.startswith("{\n") else None
    except ValueError:
        d = None
    if d is not None and "attempts" in d:
        atts = d["attempts"]
    else:  # tries.jsonl: one attempt row per line
        atts = []
        for line in text.splitlines():
            try:
                atts.append(json.loads(line))
            except ValueError:
                continue
    out = []
    for a in atts:
        if not isinstance(a, dict) or "completion_tokens" not in a:
            continue
        lvl = int(a.get("level") or 0)
        rhae = float(a.get("level_score") or 0) * float(a.get("total_levels") or 0) / 100.0 if a.get("cleared") else 0.0
        ro = a.get("rollout") or {}  # reply stats (rl_single.mg_play_stats) and the adapter that played it
        stats = [int(a.get(x) or 0) for x in ("admitted_requests", "gameplay_requests", "noaction_requests")] + \
            [int(ro.get(x) or 0) for x in ("live", "capped", "think_tokens", "filler")] + \
            [None if a.get("policy_version") is None else int(a["policy_version"])]
        out.append([lvl, int(a.get("completion_tokens") or 0), int(bool(a.get("cleared"))), int(a.get("actions") or 0), rhae,
                    stats])
    return out


REFS = {}


def unit_refs(run):
    """The logeq reward's references per set (rl_single.mg_level_ref): human baseline actions per level (the bundle's
    points), base median tokens/actions to clear (the screen, else the set's env ``ref``)."""
    if run in REFS:
        return REFS[run]
    dist = Path(os.environ.get("ARC_RL_DIST") or Path.home() / "arc-rl-dist")  # launch/rl/build.py DIST
    bundle = next((d for d in (dist / run, dist / re.sub(r"\d+$", "", run)) if (d / "configs/comparison.json").is_file()), None)
    out = {"sets": {}, "points": {}, "screen": {}}
    if bundle is not None:
        cfg = json.loads((bundle / "configs/comparison.json").read_text())
        mg = cfg["rl_loop"]["mg"]
        out["sets"] = {x["name"]: x.get("ref") or {} for x in mg.get("sets") or []}
        out["target_kl"] = mg.get("target_kl")
        pts = json.loads((bundle / "configs/rl1_points.json").read_text()).get("points") or []
        out["points"] = {x["id"][:4]: x.get("baselines") or [] for x in pts}
        if mg.get("screen"):
            text = volume_text(str(mg["screen"]).replace("/runs/", "", 1)).lstrip()
            try:
                out["screen"] = {k: v.get("metrics") or {} for k, v in
                                 (json.JSONDecoder().raw_decode(text)[0].get("sets") or {}).items()}
            except ValueError:
                pass
        hist = ((mg.get("redesign") or {}) if isinstance(mg.get("redesign"), dict) else {}).get("history")
        screened = list(out["screen"])
        if hist and (bundle / hist).is_file():  # rl_seq runs have no screen (real-v12 28 Oct: vs_ref was exp(-ln16 x
            out["screen"] = {**history_refs(bundle / hist), **out["screen"]}  # fail rate)): the v8.5p base per level
            out["hist_rows"] = history_rows(bundle / hist)
            # *_vs_ref_long beside *_vs_ref, refs from long v8.5p runs only (9 h / 300K-token;
            # short runs' run-end censoring reads as fails); the rollback triggers stay on *_vs_ref
            out["long"] = {"points": out["points"], "sets": out["sets"],
                           "screen": history_refs(bundle / hist, LONG_RUN), "hist_rows": history_rows(bundle / hist, LONG_RUN)}
    else:
        screened = []
    out["base"] = {}  # base plays per unit: the screen's tries (logeq values), plus the run's version-0 groups (st["v0"])
    for name in screened:
        rows = try_rows(volume_text(f"validation/rl1-screen-seed-v2/arena/trials/rl-loop/rl/screen-{name}/level-bench.json"))
        if rows:
            out["base"][name] = [logeq(out, name, r) for r in rows]
    REFS[run] = out
    return out


LONG_RUN = r"9h|300k"


def history_refs(path, runs=None):
    """{"<game>-L<k>": {"median_to_clear", "median_actions"}} over the history's clears (rl_backlog.load_history rows)."""
    import statistics
    by, every = {}, {}
    for r in json.loads(Path(path).read_text()).get("rows") or []:
        if runs and not re.search(runs, r.get("run") or ""):
            continue
        key = f"{r['game']}-L{int(r['level'])}"
        # every metric as a ratio to the same unit's v8.5p reference; v8.5p has no plain fails (a
        # level not cleared ended with the run, "censored"), so the reference clear rate counts censored as not cleared
        every.setdefault(key, []).append((bool(r.get("cleared")), int(r.get("tokens") or 0)))
        if r.get("cleared") and r.get("tokens") and r.get("actions"):
            by.setdefault(key, []).append((int(r["tokens"]), int(r["actions"])))
    out = {k: {"median_to_clear": statistics.median(t for t, _ in v), "median_actions": statistics.median(a for _, a in v)}
           for k, v in by.items()}
    for k, v in every.items():
        toks = [t for _, t in v if t > 0]
        out.setdefault(k, {}).update(clear_rate=sum(c for c, _ in v) / len(v),
                                     median_tokens=statistics.median(toks) if toks else None)
    return out


def history_rows(path, runs=None):
    """{"<game>-L<k>": [(level, tokens, cleared, actions, censored)]} of the v8.5p history (``runs``: a run regex)."""
    out = {}
    for r in json.loads(Path(path).read_text()).get("rows") or []:
        if runs and not re.search(runs, r.get("run") or ""):
            continue
        out.setdefault(f"{r['game']}-L{int(r['level'])}", []).append(
            (int(r["level"]), int(r.get("tokens") or 0), bool(r.get("cleared")), int(r.get("actions") or 0),
             bool(r.get("censored"))))
    return out


def hist_budget(rows, mult=1.25, q=0.8):
    """rl_backlog's historical budget: mult x p80 (linear) of the level's clear tokens."""
    toks = sorted(t for _, t, c, _, _ in rows if c and t)
    if not toks:
        return None
    pos = (len(toks) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(toks) - 1)
    return mult * (toks[lo] + (toks[hi] - toks[lo]) * (pos - lo))


def ref_logeq(refs, name, budget):
    """the unit's v8.5p mean logeq under the policy group's budget, fails included: a clear past
    the budget is a fail, a censored try past it a fail, one cut before it unknown (left out, as rl_backlog.hist_tries).
    None without history rows."""
    cache = refs.setdefault("_ref_logeq", {})
    key = (name, budget)
    if key not in cache:
        rows = (refs.get("hist_rows") or {}).get(name) or []
        b = budget or hist_budget(rows)
        vals = []
        for lvl, tok, cleared, act, censored in rows:
            if cleared and (b is None or tok <= b):
                vals.append(logeq(refs, name, (lvl, tok, True, act)))
            elif cleared or b is None or tok >= b or not censored:
                vals.append(logeq(refs, name, (lvl, tok, False, act)))
        cache[key] = sum(vals) / len(vals) if vals else None
    return cache[key]


def ref_clear_at(refs, name, budget):
    """The unit's v8.5p clear rate under the group's budget (censoring-matched as ref_logeq: a clear past the budget
    is not one, a run that ended before it unknown); None without history rows."""
    cache = refs.setdefault("_ref_clear", {})
    key = (name, budget)
    if key not in cache:
        rows = (refs.get("hist_rows") or {}).get(name) or []
        b = budget or hist_budget(rows)
        n = c = 0
        for lvl, tok, cleared, act, censored in rows:
            if cleared and (b is None or tok <= b):
                n, c = n + 1, c + 1
            elif cleared or b is None or tok >= b or not censored:
                n += 1
        cache[key] = c / n if n else None
    return cache[key]


def ref_ratios(refs, named, budgets=None):
    """train.*_vs_ref.all over (unit, try row) pairs with a v8.5p reference: clears and fails as observed / expected
    (the sum of each try's unit clear rate under the try's group budget: censoring-matched, the plain
    rate without history rows), tokens per try as the geometric mean of tokens / the unit's median."""
    import math
    budgets = budgets or [None] * len(named)
    m, cov = {}, [(r, (refs.get("screen") or {}).get(nm) or {}, nm, b) for (nm, r), b in zip(named, budgets) if nm]
    cr = [(r, p) for r, x, nm, b in cov
          if (p := (ref_clear_at(refs, nm, b) if (refs.get("hist_rows") or {}).get(nm) else x.get("clear_rate")))
          is not None]
    if cr:
        exp_clear = sum(p for _, p in cr)
        obs = sum(r[2] for r, _ in cr)
        if exp_clear:
            m["train.clear_rate_vs_ref.all"] = obs / exp_clear
        if len(cr) - exp_clear > 0:
            m["train.fail_rate_vs_ref.all"] = (len(cr) - obs) / (len(cr) - exp_clear)
    tk = [math.log(r[1] / x["median_tokens"]) for r, x, _, _ in cov if x.get("median_tokens") and r[1] > 0]
    if tk:
        m["train.tokens_per_try_vs_ref.all"] = math.exp(sum(tk) / len(tk))
    return m


def logeq(refs, name, row, cap=1.15, clip=8.0, fail=16.0):
    """The trained logeq reward of one try: clip(log((R / R_b) x (T_b / t)), +-log clip), a fail -log fail."""
    import math
    lvl, tok, cleared, actions = row[0], row[1], row[2], row[3]
    if not cleared or tok <= 0 or actions <= 0:
        return -math.log(fail)
    bl = refs["points"].get(name[:4]) or []
    ref, m = refs["sets"].get(name) or {}, refs["screen"].get(name) or {}
    b = (bl[lvl - 1] if 0 < lvl <= len(bl) and bl[lvl - 1] else None) or ref.get("baseline")
    t_b = m.get("median_to_clear") or ref.get("tokens") or tok
    a_b = m.get("median_actions") or ref.get("actions") or actions
    rhae = lambda a: min(cap, (b / a) ** 2) if b and a else 1.0
    v = math.log(rhae(actions) / rhae(float(a_b))) + math.log(float(t_b) / tok)
    return max(-math.log(clip), min(math.log(clip), v))


def vs_first(st, gids):
    """per level, the step's RHAE/tok against the first group played on it (its earliest version, kept in
    st["first"] {set: [rhae_sum, tok_sum]}), geometric mean over levels (a level at 0 RHAE floored at 1/16, as a fail);
    levels whose first group cleared nothing have no anchor (coverage)."""
    import math
    first, by = st.setdefault("first", {}), {}
    for g in gids:
        name = (st.get("gset") or {}).get(g)
        rows = (st.get("tries") or {}).get(g) or []
        if name and rows:
            x = by.setdefault(name, [0.0, 0, []])
            x[0] += sum(r[4] for r in rows)
            x[1] += sum(r[1] for r in rows)
            x[2] += [r[0] for r in rows]
    m = {}
    for suffix, keep in ((".L13", lambda lv: lv in (1, 3)), (".all", lambda lv: True)):
        logs, n = [], 0
        for name, (rh, tok, lvls) in by.items():
            if not lvls or not keep(lvls[0]):
                continue
            n += 1
            f = first.get(name)
            if f and f[0] > 0 and f[1] > 0 and tok > 0:
                logs.append(math.log(max(rh / tok, (f[0] / f[1]) / 16.0) / (f[0] / f[1])))
        if logs:
            m[f"train.rhae_tok_vs_first{suffix}"] = math.exp(sum(logs) / len(logs))
            m[f"train.first_coverage{suffix}"] = len(logs) / n
    return m


def reply_stats(rows):
    """rl_single.mg_play_stats over a step's tries, and train.played_by: the adapter that played most of them."""
    s = [r[5] for r in rows if len(r) > 5 and r[5]]
    if not s:
        return {}
    tot = [sum(x[i] or 0 for x in s) for i in range(7)]
    asked, game, noact, live, capped, think, filler = tot
    m = {}
    if asked:
        m["train.reply_tokens_mean"] = sum(r[1] for r in rows if len(r) > 5 and r[5]) / asked
    if game:
        m["train.noaction_frac"] = noact / game
    if live:
        m["train.truncation_frac"] = capped / live

    vs = [x[7] for x in s if x[7] is not None]
    if vs:
        m["train.played_by"] = max(set(vs), key=lambda v: (vs.count(v), v))
    return m


def try_metrics(prefix, rows, suffix):
    """ovf3r3 schema: <prefix>.rhae_per_10k / rhae_mean / clear_rate / tokens_per_try / actions_per_clear + suffix."""
    if not rows:
        return {}
    tok = sum(r[1] for r in rows)
    clears = sum(r[2] for r in rows)
    m = {f"{prefix}.rhae_mean{suffix}": sum(r[4] for r in rows) / len(rows),
         f"{prefix}.{'clear_rate' if prefix == 'train' else 'clears_frac'}{suffix}": clears / len(rows),
         f"{prefix}.tokens_per_try{suffix}": tok / len(rows)}
    if tok:
        m[f"{prefix}.rhae_per_10k{suffix}"] = 1e4 * sum(r[4] for r in rows) / tok
    if clears:
        m[f"{prefix}.actions_per_clear{suffix}"] = sum(r[3] for r in rows if r[2]) / clears
    return m


def tail_new(path, key, st):
    p = Path(path)
    if not p.exists():
        return []
    lines = p.read_text().splitlines()
    i = st.get(key, 0)
    st[key] = len(lines)
    out = []
    for line in lines[i:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def pooled(sets, names=None):
    ok = [m for n, m in sets.items() if (names is None or n in names) and m.get("rhae_per_10k") is not None
          and m.get("tries") and m.get("tokens_per_try")]
    W = sum(m["tries"] * m["tokens_per_try"] for m in ok)
    if not W:
        return None, None
    w = [m["tries"] * m["tokens_per_try"] / W for m in ok]
    return (sum(a * m["rhae_per_10k"] for a, m in zip(w, ok)),
            sum((a * float(m.get("rhae_10k_se") or 0)) ** 2 for a, m in zip(w, ok)) ** 0.5)


def ratio(v, se, v0, se0):
    """v / v0 with a 95% CI (delta method on independent reads), or None."""
    if not v or not v0:
        return None
    r = v / v0
    d = 1.96 * r * ((se / v) ** 2 + (se0 / v0) ** 2) ** 0.5
    return r, max(0.0, r - d), r + d


BATCH_GROUPS = 8  # hero: 8 groups of 12 per step


def group_counts(e, st, m):
    """Groups played and dropped, per step and cumulative over this leg (seeded from the leg it
    continues): lag = the buffer's staleness drop, filtered = taken but no try trained (zero spread / reply filter),
    other = done with no usable try."""
    k = e.get("kind")
    gc = st.setdefault("gcount", {})
    add = lambda key: [gc.__setitem__(x, gc.get(x, 0) + 1) for x in (key, "step_" + key)]
    if k == "group_done":
        add("played")
        if not int(e.get("tries") or 0):
            add("other")
    elif k == "group_drop":
        add("lag")
    elif k == "accumulate" and e.get("trained") is not None and not int(e.get("trained") or 0):
        add("filtered")
    elif k == "step":
        reasons = ("lag", "filtered", "other")
        for pre, out in (("step_", "_step"), ("", "")):
            dropped = sum(gc.get(pre + r, 0) for r in reasons)
            m[f"train.groups_played{out}"] = gc.get(pre + "played", 0)
            m[f"train.groups_dropped{out}"] = dropped
            for r in reasons:
                m[f"train.groups_dropped_{r}{out}"] = gc.get(pre + r, 0)
            if gc.get(pre + "played"):
                m[f"train.groups_dropped_frac{out}"] = round(dropped / gc[pre + "played"], 4)
        if gc.get("lost_attempt"):
            m["train.groups_lost_attempt"] = gc["lost_attempt"]
        for x in [x for x in gc if x.startswith("step_")]:
            gc[x] = 0


def attempt_reset(st):
    """A new attempt resumes from its roots' saved state, not from the dead attempt's play (v17 29 Oct: attempt 1's 23
    played groups read as 24 dropped "other" at U18, and play.live_groups_done ran 21 -> 37 past 16): the dead
    attempt's untrained groups move to groups_lost_attempt, and the per-step and live play counters restart."""
    gc = st.setdefault("gcount", {})
    gc["lost_attempt"] = gc.get("lost_attempt", 0) + gc.get("step_played", 0)
    for x in [x for x in gc if x.startswith("step_")]:
        gc[x[5:]] = max(0, gc.get(x[5:], 0) - gc[x])  # out of the cumulative counts too
        gc[x] = 0
    st.pop("grp", None)
    st.pop("acc", None)  # phase_games kept: a resume trains the dead attempt's played groups first (a9 U22)


def game_mix(e, st, m):
    """The update's game mix (U21-U22's 2.97x was sb26 filling 11 of 16 groups): the largest single
    game's share of the groups played since the previous update, and its name."""
    k = e.get("kind")
    if k == "group_done" and e.get("set"):
        st.setdefault("phase_games", []).append(str(e["set"]).rsplit("-L", 1)[0])
    elif k == "step":
        games = st.pop("phase_games", [])
        if games:
            top = max(set(games), key=lambda g: (games.count(g), g))
            m["train.top_game_share"], m["train.top_game"] = round(games.count(top) / len(games), 4), top
            m["train.games_in_update"] = len(set(games))


def step_tokens(e):
    """The update's tokens: a multi-step update's row before trunk 30 Oct carries only its last step's (U24 1.01M of 1.82M)."""
    mbs = e.get("minibatches") or []
    return sum(int(x.get("trained_tokens") or 0) for x in mbs) if mbs else e.get("trained_tokens")


def feed_metrics(e, st):
    k, m = e.get("kind"), {}
    group_counts(e, st, m)
    game_mix(e, st, m)
    if k == "step":
        m.update({"train.step": e.get("step"), "train.lr": e.get("lr"), "train.grad_norm": e.get("grad_norm"),
                  "train.policy_loss": e.get("loss"), "train.entropy": e.get("entropy"),
                  "train.tokens": step_tokens(e), "train.tries": e.get("trained"),
                  "train.actions_per_clear_vs_base": e.get("actions_per_clear_vs_base"),
                  "train.update_norm_step": e.get("update_norm"), "train.approx_kl": (e.get("k3") or {}).get("approx_kl"),
                  "train.clip_frac": (e.get("k3") or {}).get("clip_frac"),
                  "train.adv_nonzero_share": e.get("adv_nonzero_share"), "train.adv_up_share": e.get("adv_up_share"),
                  "train.k3_tok": (e.get("k3") or {}).get("approx_kl_tok"),
                  "time.checkpoint_s": e.get("checkpoint_seconds")})
        if e.get("prompt_same_rate") is not None:  # RL1_REPLY_IDS: trained replies in their served context
            m.update({"train.prompt_same_rate": e["prompt_same_rate"], "train.prompt_excluded": e.get("prompt_excluded")})
        mbs = e.get("minibatches") or []
        if len(mbs) > 1:  # mg.redesign.minibatches: the pre-registered checks, last step vs first
            first, last = mbs[0], mbs[-1]

            def ratio(key):
                a, b = last.get(key), first.get(key)
                return round(float(a) / float(b), 4) if a is not None and b else None

            m.update({"train.minibatches": len(mbs), "train.update_norm_sum": e.get("update_norm_sum"),
                      "train.grad_norm_max": e.get("grad_norm_max"), "train.mb_k3_last_vs_first": ratio("approx_kl"),
                      "train.mb_clip_last": last.get("clip_frac"), "train.mb_clip_last_vs_first": ratio("clip_frac")})
        tis = e.get("tis_out") or {}
        if tis.get("of"):
            m["train.tis_out_share"] = float(tis.get("tokens") or 0) / float(tis["of"])
        lags = e.get("lag") or {}
        if lags:
            n = sum(lags.values())
            m["train.age_median"] = sorted(int(a) for a, c in lags.items() for _ in range(c))[n // 2]
            m["train.age_max"] = max(int(a) for a in lags)
        if st.get("last_step_epoch"):
            m["time.step_s"] = e["epoch"] - st["last_step_epoch"]
        st["last_step_epoch"] = e["epoch"]
        acc = st.pop("acc", None)
        if acc:
            m["time.train_s"] = acc["train_s"]
            m["infra.peak_alloc_gib"] = acc["peak"]
            m["infra.gpu_util.trainer_mean"] = acc["util"]
        # v7 phase split of the trainer card's step time: play = card 2 serving blocks, update =
        # accumulates + the step call, old_pass = the k3 pre/post pass, handoff = park/wake/load/sleep/unpark, idle = rest
        c2 = st.pop("c2", None) or {}
        if m.get("time.step_s") is not None:
            upd = (acc or {}).get("train_s", 0.0) + float(e.get("step_call_seconds") or 0)
            old = float((e.get("k3") or {}).get("seconds") or 0)
            play, hand = c2.get("play", 0.0), c2.get("handoff", 0.0)
            m.update({"time.play_s": play, "time.update_s": upd, "time.old_pass_s": old, "time.handoff_s": hand,
                      "time.idle_s": max(0.0, m["time.step_s"] - play - upd - old - hand)})
        gids = st.pop("step_gids", [])
        rows = [r for g in gids for r in (st.get("tries") or {}).get(g, [])]
        m.update(try_metrics("train", [r for r in rows if r[0] in (1, 3)], ".L13"))
        m.update(try_metrics("train", rows, ".all"))
        m.update(reply_stats(rows))
        refs = st.get("_refs")
        if refs is not None:  # mix-adjusted: geometric mean of RHAE/tok vs each unit's base reference
            import math
            named = [(st.get("gset", {}).get(g), r) for g in gids for r in (st.get("tries") or {}).get(g, [])]
            budgets = [(st.get("gbudget") or {}).get(g) for g in gids for _ in (st.get("tries") or {}).get(g, [])]
            for suffix, keep in ((".L13", lambda r: r[0] in (1, 3)), (".all", lambda r: True)):
                vals = [logeq(refs, nm, r) for nm, r in named if nm and keep(r)]  # rhae_tok_vs_ref_flat dropped 29 Oct
                rel = [logeq(refs, nm, r) - b for (nm, r), bud in zip(named, budgets)
                       if nm and keep(r) and (b := ref_logeq(refs, nm, bud)) is not None]
                if rel:  # policy mean logeq - v8.5p mean logeq of the same unit (its fails included)
                    m[f"train.rhae_tok_vs_ref{suffix}"] = math.exp(sum(rel) / len(rel))
                lg = refs.get("long")
                if lg and suffix == ".all":
                    rel = [logeq(lg, nm, r) - b for (nm, r), bud in zip(named, budgets)
                           if nm and (lg.get("screen") or {}).get(nm) and (b := ref_logeq(lg, nm, bud)) is not None]
                    if rel:
                        m["train.rhae_tok_vs_ref_long.all"] = math.exp(sum(rel) / len(rel))
                        m["train.ref_coverage_long.all"] = len(rel) / max(1, len(vals))
                    m.update({k.replace("_vs_ref.", "_vs_ref_long."): v for k, v in ref_ratios(lg, named, budgets).items()})
                if suffix == ".all":
                    m.update(ref_ratios(refs, named, budgets))
                pool = {**refs.get("base", {}), **(st.get("v0") or {})}
                cov = [(logeq(refs, nm, r), sum(pool[nm]) / len(pool[nm])) for nm, r in named
                       if nm and keep(r) and pool.get(nm)]
                if cov and vals and False:  # retired 24 Oct (coverage set changed per step; placeholder clips): vs_first
                    pol, base = math.exp(sum(a for a, _ in cov) / len(cov)), math.exp(sum(b for _, b in cov) / len(cov))
                    m.update({f"train.rhae_tok_vs_ref_cov{suffix}": pol, f"train.rhae_tok_vs_ref_cov{suffix}_base": base,
                              f"train.rhae_tok_vs_ref_base{suffix}": base, f"train.ref_coverage{suffix}": len(cov) / len(vals)})
            # per level-depth bucket (an update's mix of depths swings .all: U21 L1/sb26-heavy 3.03)
            for suffix, keep in ((".d1", lambda r: r[0] == 1), (".d23", lambda r: r[0] in (2, 3)),
                                 (".d4p", lambda r: (r[0] or 0) >= 4)):
                rel = [logeq(refs, nm, r) - b for (nm, r), bud in zip(named, budgets)
                       if nm and keep(r) and (b := ref_logeq(refs, nm, bud)) is not None]
                if rel:
                    m[f"train.rhae_tok_vs_ref{suffix}"] = math.exp(sum(rel) / len(rel))
                    m[f"train.ref_tries{suffix}"] = len(rel)
        vs_first(st, gids)  # rhae_tok_vs_first dropped (lucky-first anchor); its state kept
        st["psum"] = {}
        st["round"] = int(e.get("step") or 0) + 1
        grp = st.pop("grp", None)
        if grp and grp["tries"]:
            m["play.clear_rate"] = grp["clears"] / grp["tries"]
            m["play.band_frac"] = grp["band"] / max(1, grp["groups"])
            m["play.groups"] = grp["groups"]
    elif k in ("seq_train", "seq_pending") and e.get("groups"):  # rl_seq: this update's group count and play time
        st["batch_groups"] = int(e["groups"])
        if k == "seq_train" and isinstance(e.get("play_seconds"), (int, float)):
            st.setdefault("c2", {})["play"] = float(e["play_seconds"])
    elif k == "seq_accumulate" and e.get("one_call"):  # the update's one call: its encode-ahead use (gate legs 28 Oct:
        # rl_seq's train time is on these rows (one per update, or one per minibatch), not on the per-group rows
        a = st.setdefault("acc", {"train_s": 0.0, "peak": None, "util": None})
        tok, sec = float(e.get("trained_tokens") or 0), float(e.get("train_seconds") or 0)
        a["train_s"] += sec
        a["tok"] = a.get("tok", 0.0) + tok
        a["mb_groups"] = a.get("mb_groups", 0) + int(e.get("groups") or 0)
        if e.get("card_peak_gib") is not None:
            a["peak"] = max(a["peak"] or 0, float(e["card_peak_gib"]))
        m.update({"train.live_groups_done": a["mb_groups"], "train.live_groups_total": st.get("batch_groups", BATCH_GROUPS),
                  "train.live_tokens": a["tok"], "time.live_train_s": a["train_s"]})
        if sec > 0:
            m["train.live_tok_s"] = tok / sec
        if e.get("ahead_expected"):  # 0 of 192 early windows taken for weeks); no reading counts as 0
            share = float(e.get("ahead_hit_share") or 0.0)
            if float(e.get("ahead_misses") or 0) > 0:  # gate-v3 rows: that field held the encoded total
                share = float(e.get("ahead_hits") or 0) / float(e["ahead_misses"])
            m["train.ahead_hit_frac"] = share  # *_frac: shown x 100
        m.update({"train.call_util": e.get("call_util"), "train.setup_s": e.get("setup_seconds"),
                  "train.window_wait_s": e.get("window_wait_seconds")})
        if "encode_errors" in e:  # R3 coverage per call (trunk a5c9944c7); no served_share = no served routing
            m.update({"train.served_frac": float(e.get("served_share") or 0.0), "train.encode_errors": e["encode_errors"]})
    elif k == "accumulate" or (k == "seq_accumulate" and e.get("gid")):  # rl_seq names its accumulate rows seq_
        if e.get("set"):  # a replayed group (resume) has no group_start/group_done in this leg's state
            st.setdefault("gset", {})[e.get("gid")] = e["set"]
        a = st.setdefault("acc", {"train_s": 0.0, "peak": None, "util": None})
        a["train_s"] += float(e.get("train_seconds") or 0)
        a["groups"] = a.get("groups", 0) + 1
        st.setdefault("step_gids", []).append(e.get("gid"))
        m["play.live_groups_accumulated"] = a["groups"]  # toward batch_groups (8 x 12 = 96 tries per step)
        if e.get("card_peak_gib") is not None:
            a["peak"] = max(a["peak"] or 0, float(e["card_peak_gib"]))
        if e.get("gpu_util_mean") is not None:
            a["util"] = e["gpu_util_mean"]
        # live trainer rows (training visible while a step runs)
        tok, sec = float(e.get("trained_tokens") or 0), float(e.get("train_seconds") or 0)
        a["tok"] = a.get("tok", 0.0) + tok
        if "mb_groups" not in a:  # rl_seq's per-group rows carry no train time: the one-call rows report the live ones
            m.update({"train.live_groups_done": a["groups"], "train.live_groups_total": st.get("batch_groups", BATCH_GROUPS),
                      "train.live_tokens": a["tok"], "time.live_train_s": a["train_s"], "infra.card2_mode": 0})
        if sec > 0:
            m["train.live_tok_s"] = tok / sec
        if e.get("reward_std") is not None:
            m["train.live_reward_std"] = e["reward_std"]
        tis = e.get("tis_out") or {}
        if tis.get("of"):
            m["train.live_tis_out_frac"] = tis["tokens"] / tis["of"]
        b = e.get("balance") or {}
        g = b.get("gigpo") or {}
        if b.get("w_abs_median") is not None:
            m["train.live_w_abs_median"] = b["w_abs_median"]
        if g.get("mean_abs_step") is not None:
            m["train.live_gigpo_step_abs"] = g["mean_abs_step"]
        if b.get("sign_flips") is not None:
            m["train.live_sign_flips"] = b["sign_flips"]
    elif k == "v7_start":  # a leg's first step is timed from its loop start
        st["last_step_epoch"] = e.get("epoch")
    elif k == "block_on":
        st["block_on"] = e.get("epoch")
        m["infra.card2_mode"] = 1  # card 2 plays
    elif k == "block_off":
        c2 = st.setdefault("c2", {"play": 0.0, "handoff": 0.0})
        park, unpark = (e.get("park") or {}).get("seconds") or 0, (e.get("unpark") or {}).get("seconds") or 0
        hand = float(park) + float(unpark) + sum(float(e.get(x) or 0) for x in ("boot_s", "wake_s", "load_s", "sleep_s"))
        c2["handoff"] += hand
        c2["play"] += max(0.0, float(e.get("seconds") or 0) - hand)
        m["infra.card2_mode"] = 0
        m["infra.card2_moved_tries"] = e.get("moved_tries")
    elif k == "group_done":
        rows = (st.get("tries") or {}).get(e.get("gid")) or []
        if rows:
            ps = st.setdefault("psum", {})
            for lvl, tok, c, act, rh in (r[:5] for r in rows):
                x = ps.setdefault(str(lvl), [0, 0, 0, 0, 0.0])
                x[0] += 1; x[1] += c; x[2] += tok; x[3] += act; x[4] += rh
            total = sum(x[0] for x in ps.values())
            m["play.round"] = st.get("round", 1)
            for lvl, x in ps.items():  # planned: the step's 96 tries split by each level's share so far
                m.update({f"play.L{lvl}.tries": x[0], f"play.L{lvl}.clears": x[1], f"play.L{lvl}.tok_sum": x[2],
                          f"play.L{lvl}.act_sum": x[3], f"play.L{lvl}.rhae_sum": x[4],
                          f"play.L{lvl}.planned": BATCH_GROUPS * 12 * x[0] / total})
        g = st.setdefault("grp", {"tries": 0, "clears": 0, "groups": 0, "band": 0})
        t, c = int(e.get("tries") or 0), int(e.get("clears") or 0)
        g["tries"] += t
        g["clears"] += c
        g["groups"] += 1
        g["band"] += int(0 < c < t)
        # live play between steps (a slow step stays visible)
        m.update({"play.live_groups_done": g["groups"], "play.live_tries_done": g["tries"],
                  "play.live_band_frac": g["band"] / g["groups"], "play.live_clear_rate": g["clears"] / max(1, g["tries"])})
    elif k == "read":
        sets = e.get("sets") or {}
        extra = set(e.get("extra_sets") or [])
        v, se = pooled(sets, [n for n in sets if n not in extra])
        m["eval.heldout_rhae10k"], m["eval.heldout_se"] = v, se
        m["eval.read_step"] = e.get("step")
        if int(e.get("step") or 0) == 0 and v:
            st["base"] = [v, se]
        r = ratio(v, se, *(st.get("base") or [None, None]))
        if r:  # target: 2x held-out RHAE/tok vs the base (version 0)
            m["eval.heldout_ratio"], m["eval.heldout_ratio_lo"], m["eval.heldout_ratio_hi"] = r
            m["eval.heldout_ratio_target"] = 2.0
        if extra:
            m["eval.heldout_ext_rhae10k"] = pooled(sets)[0]
        for suffix, names in ((".L13", [nm for nm in sets if nm not in extra and re.search(r"-L(1|3)$", nm)]),
                              (".all", [nm for nm in sets if nm not in extra])):
            v13 = pooled(sets, names)[0]
            if v13 is None:
                continue
            m[f"eval.rhae_per_10k{suffix}"] = v13
            ok = [sets[nm] for nm in names if sets[nm].get("tries")]
            n13 = sum(x["tries"] for x in ok)
            if n13:
                for key, f in (("rhae_mean", "rhae_mean"), ("clears_frac", "clear_rate"), ("tokens_per_try", "tokens_per_try")):
                    if all(x.get(f) is not None for x in ok):
                        m[f"eval.{key}{suffix}"] = sum(x[f] * x["tries"] for x in ok) / n13
            bk = "base13" if suffix == ".L13" else "base_all"
            if int(e.get("step") or 0) == 0:
                st[bk] = {kk: vv for kk, vv in m.items() if kk.endswith(suffix)}
            for kk, vv in (st.get(bk) or {}).items():
                m[kk + "_base"] = vv
    elif k == "wread":
        m["eval.wgame_score"], m["eval.wgame_score_per_mtok"] = e.get("score"), e.get("score_per_mtok")
        r = e.get("ratio_score")
        if r:
            m["eval.wgame_ratio"], m["eval.wgame_ratio_lo"], m["eval.wgame_ratio_hi"] = r
            m["eval.wgame_ratio_target"] = 2.0
        m["eval.wgame_ratio_per_mtok"] = e.get("ratio_per_mtok")
    elif k == "group_start" and e.get("set"):  # breadth: games and sets the run has played
        sets = st.setdefault("sets_started", [])
        if e["set"] not in sets:
            sets.append(e["set"])
        games = sorted({x.rsplit("-L", 1)[0] for x in sets})
        m["play.games_played"], m["play.sets_started"] = len(games), len(sets)
        if st.get("games_active"):
            m["play.games_played_frac"] = len(games) / st["games_active"]
    elif k == "seq_phase":  # rl_seq: the play phase's MTP read (stage D bars), server guard, switch seconds
        mtp = e.get("mtp") or {}
        for src, dst in (("accept", "infra.draft_acceptance"), ("pos1", "infra.draft_pos1"),
                         ("mean_len", "infra.draft_mean_len"), ("gen_tok_s", "infra.phase_gen_tok_s")):
            if isinstance(mtp.get(src), (int, float)):
                m[dst] = mtp[src]
        guard = e.get("guard") or {}
        if isinstance(guard.get("kv_peak"), (int, float)):
            m["infra.phase_kv_peak"] = guard["kv_peak"]
        if isinstance(guard.get("preemptions"), (int, float)):
            m["infra.phase_preemptions"] = guard["preemptions"]
        for side in ("stop", "start"):
            if isinstance((e.get(side) or {}).get("switch_s"), (int, float)):
                m[f"infra.switch_{side}_s"] = e[side]["switch_s"]
                c2 = st.setdefault("c2", {})  # the step split's handoff (counted in the next step's row)
                c2["handoff"] = c2.get("handoff", 0.0) + float(e[side]["switch_s"])
        if isinstance(e.get("play_seconds"), (int, float)):
            m["play.phase_seconds"] = e["play_seconds"]
    elif k == "r3_offload":  # rl_seq (r3off): infra.r3_offload_bound / infra.r3_offload_refused.* at boot + first token
        m.update({kk: vv for kk, vv in (e.get("metrics") or {}).items() if isinstance(vv, (int, float))})
    elif k == "envs":
        if e.get("active"):
            st["games_active"] = int(e["active"])
            m["play.games_active"] = int(e["active"])
        m["play.pool"] = e.get("pool")
        for lvl, n in (e.get("fresh_by_level") or {}).items():
            m[f"play.fresh_L{lvl}"] = n
        for s, n in (e.get("start_source") or {}).items():
            m[f"play.src_{s}"] = n
    return m


def lineage(run):
    """``run`` and the legs it continues (metrics meta "continues", oldest last)."""
    out, seen = [run], {run}
    while True:
        f = metrics.scratch_dir() / f"{out[-1]}.meta.json"
        cont = (json.loads(f.read_text()).get("continues") or []) if f.is_file() else []
        nxt = next(((c["run"] if isinstance(c, dict) else c) for c in cont), None)
        if not nxt or nxt in seen:
            return out
        out.append(nxt)
        seen.add(nxt)


def carry_target(run, out, prefixes=("train.rhae_per_10k", "eval.heldout_rhae10k", "eval.rhae_per_10k")):
    """Checklist item 7 (v6j3: no step in 85 min left the dashboard without the target metric): a
    resumed leg without its own target rows yet emits the latest values of its lineage's, once, at the start."""
    own = metrics.read_rows(metrics.scratch_dir() / f"{run}.jsonl")
    if any(k.startswith(prefixes) for r in own for k in (r.get("m") or {})):
        return {}
    for leg in lineage(run)[1:]:
        last = {}
        for r in metrics.read_rows(metrics.scratch_dir() / f"{leg}.jsonl"):
            last.update({k: v for k, v in (r.get("m") or {}).items() if k.startswith(prefixes) and v is not None})
        if last:
            out.emit(last, t=time.time(), note=f"target metric carried from {leg} until this leg's first step")
            return last
    return {}


MISSED = set()  # paths whose read this pass skipped (another reader) or failed (timeout, error)


def volume_text(path):
    """The run's file, fetched into the machine-wide mirror (export_rl is each RL run's one fetcher: watch reads the
    mirror). A skipped (another reader) or failed fetch counts as missed and returns the last good copy."""
    from arc_infra.runops import volfetch

    if not volfetch.get("arc-prolong-runs", path, timeout=120, modal=MODAL):
        MISSED.add(path)
    return volfetch.read("arc-prolong-runs", path) or ""


def sampler_once(run, st, out):
    """Infra rows of an rl_seq run (no RL-watcher pass files): the runops sampler (util/VRAM per card, both servers'
    running/KV/gen tok/s/preemptions) and cpulog (host RAM), rows newer than the last pass's, plus cost."""
    from arc_infra.runops import arena
    folder = OUT.get(run, run)
    rows = [json.loads(x) for x in volume_text(SAMPLER.format(run=folder)).splitlines() if x.strip().startswith("{")]
    got = arena.sampler_rows(rows, (st.get("rate") or [None])[0])
    got += arena.cpulog_rows(volume_text(CPULOG.format(run=folder)))
    last = st.get("sampler_t", 0.0)
    for r in sorted(got, key=lambda r: r["t"]):
        if r["t"] > last and r.get("m"):
            m = {k: v for k, v in r["m"].items() if v is not None}
            if any(k.startswith("infra.vram_gib.") for k in m):
                m["infra.vram_cap_gib"] = CAP_GIB
            if st.get("rate"):
                m["infra.cost_usd"] = max(0.0, r["t"] - st["rate"][0]) * st["rate"][1]
            out.emit(m, t=r["t"])
            st["sampler_t"] = r["t"]


def once(run, passes, st, out):
    for p in ([] if st.get("backfill") else tail_new(passes / f"{run}-watch-passes.jsonl", "passes", st)):
        s2 = p.get("s2") or {}
        m = {"infra.kv_frac": p.get("kv"),  # fraction: train_dash shows *_frac x 100
             "infra.running": (p.get("running") or 0) + (s2.get("num_requests_running") or 0),
             "infra.waiting": (p.get("waiting") or 0) + (s2.get("num_requests_waiting") or 0),
             "infra.gen_tok_s": p.get("gen_tok_s")}
        g2 = s2.get("generation_tokens_total")
        if g2 is not None and st.get("g2") is not None and p["t"] > st["g2t"] and g2 >= st["g2"]:
            m["infra.gen_tok_s"] = (m["infra.gen_tok_s"] or 0) + (g2 - st["g2"]) / (p["t"] - st["g2t"])
        if g2 is not None:
            st["g2"], st["g2t"] = g2, p["t"]
        out.emit(m, t=p["t"])
    for u in ([] if st.get("backfill") else tail_new(passes / f"{run}-card-util.jsonl", "util", st)):
        m = {"infra.vram_cap_gib": CAP_GIB}
        for i, c in (u.get("util5") or {}).items():
            m[f"infra.gpu_util.{i}"] = c.get("util")
            if c.get("vram") is not None:
                m[f"infra.vram_gib.{i}"] = c["vram"] * CAP_GIB
        out.emit(m, t=u["t"])
    if not st.get("backfill") and not (passes / f"{run}-watch-passes.jsonl").exists():
        sampler_once(run, st, out)
    fo = OUT.get(run, run)
    if st.setdefault("feed_out", fo) != fo:  # a new attempt folder: its feed from line 0 (real-v11 attempt 3, 28 Oct:
        st["feed"], st["feed_out"] = 0, fo   # attempt 2's 79-line cursor skipped attempt 3's boot and r3 rows)
        attempt_reset(st)
    lines = [x for x in volume_text(FEED.format(run=fo)).splitlines() if x.strip()]
    for line in lines[st.get("feed", 0):]:
        try:
            e = json.loads(line)
        except ValueError:
            break  # a torn last line: read again next pass
        def fetch(gid, name):
            got = try_rows(volume_text(GROUP.format(run=OUT.get(run, run), gid=gid, set=name)))
            if got:
                return got
            # a live run: finished tries under partial/, a resumed group's split over the legs that kept them (hero-v6g
            # step 3: 65 of 94 trained tries without the earlier legs'); one row per label, the latest leg's
            by = {}
            for leg in reversed(lineage(run)):
                for line in volume_text(PARTIAL.format(run=OUT.get(leg, leg), gid=gid, set=name)).splitlines():
                    try:
                        a = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(a, dict) and a.get("label"):
                        by[a["label"]] = line
            return try_rows("\n".join(by.values()))

        if e.get("kind") == "seq_replay":  # a pending-only resume trains groups played by an earlier leg (real-v12,
            for f in e.get("folders") or []:  # 28 Oct: step 1 had no group_done rows, no tries, a KeyError)
                gid, _, name = os.path.basename(f.rstrip("/")).partition("-")
                st.setdefault("gset", {})[gid] = name
                got = try_rows(volume_text(f.replace("/runs/", "", 1).rstrip("/") + "/level-bench.json"))
                st.setdefault("tries", {})[gid] = got
                st.setdefault("gtries", {})[gid] = len(got) or 12
        if e.get("kind") == "seq_group" and e.get("gid"):  # the group's token budget (ref_logeq's cut)
            st.setdefault("gbudget", {})[e["gid"]] = e.get("budget")
        if e.get("kind") == "group_done":  # name and try count always: a seeded/restarted state may hold old tries
            st.setdefault("gset", {})[e["gid"]] = e.get("set")
            st.setdefault("gtries", {})[e["gid"]] = int(e.get("tries") or 12)
        if e.get("kind") == "group_done" and (e.get("attempts") or e.get("gid") not in st.setdefault("tries", {})):
            # rl_seq (27 Oct): the row carries its tries (no group folder on the volume while the run lives); a row with
            # attempts always wins (real-v12 28 Oct: new groups reused the carried groups' ids p1g6..p1g20, and steps
            # 2+ were scored on the carried tries)
            st.setdefault("tries", {})[e["gid"]] = (try_rows(json.dumps({"attempts": e["attempts"]}, indent=1))
                                                    if e.get("attempts") else fetch(e["gid"], e.get("set")))
        if e.get("kind") == "group_done" and e.get("set") and e.get("set") not in st.setdefault("first", {}) \
                and st["tries"].get(e["gid"]):  # the level's first group (seeded from earlier legs at a restart)
            st["first"][e["set"]] = [sum(r[4] for r in st["tries"][e["gid"]]), sum(r[1] for r in st["tries"][e["gid"]])]
        if e.get("kind") == "step":  # the step's groups again: a live run's tries.jsonl lags its group_done row
            for gid in st.get("step_gids", []):
                name = (st.get("gset") or {}).get(gid)
                if name and len(st.setdefault("tries", {}).get(gid) or []) < (st.get("gtries") or {}).get(gid, 12):
                    st["tries"][gid] = fetch(gid, name) or st["tries"].get(gid) or []
        if e.get("kind") == "group_done" and int(e.get("version") or 0) == 0 and st["tries"].get(e.get("gid")) \
                and e.get("gid") not in st.setdefault("v0_gids", []):  # once per group (an attempt re-reads its feed)
            st["v0_gids"].append(e.get("gid"))
            refs0 = unit_refs(run)
            st.setdefault("v0", {}).setdefault(e.get("set"), []).extend(logeq(refs0, e.get("set"), r)
                                                                         for r in st["tries"][e["gid"]])
        if e.get("kind") == "step":
            st["_refs"] = unit_refs(run)
        ctl = control_pairs(e, st, run)
        m = feed_metrics(e, st)
        if ctl:
            m = {**(m or {}), **ctl}
        st.pop("_refs", None)
        rate = st.get("rate")
        if m and rate and e.get("epoch"):  # GPU list price since the leg's start (the dashboard adds earlier legs)
            m["infra.cost_usd"] = max(0.0, float(e["epoch"]) - rate[0]) * rate[1]
        if st.get("backfill"):  # an earlier leg: only the play/eval schema keys this exporter added
            keys = tuple(st["backfill"]) if isinstance(st["backfill"], list) else NEW_KEYS
            m = {kk: vv for kk, vv in m.items() if kk.startswith(keys)}
        alert = e.get("why") if e.get("kind") == "alert" else None
        if m or alert:
            out.emit(m, t=e.get("epoch") or time.time(), alert=alert)
        st["feed"] = st.get("feed", 0) + 1  # after the row: a row that raises is read again, never skipped


def control_pairs(e, st, run, last=4):
    """the learning metric: an untrained base-model twin (rl_seq control_done) of a policy group, same
    start and budget. Each pair's difference policy mean logeq - control mean logeq; ``train.vs_control`` =
    exp(mean over the last ``last`` pairs) (1.0 = no learning on the levels it plays). A rollback (step going down, as
    train_dash's branches) drops the pairs played on the abandoned branch."""
    def summary():
        diffs = [d[2] for d in st["ctl_diffs"][-last:]]
        return {"train.vs_control": round(math.exp(sum(diffs) / len(diffs)), 4), "train.control_diff": st["ctl_diffs"][-1][2],
                "train.control_pairs": len(st["ctl_diffs"])}

    if e.get("kind") == "step" and e.get("step") is not None:
        s = int(e["step"])
        rolled = s < st.get("ctl_top", -1)
        st["ctl_top"] = s if rolled else max(s, st.get("ctl_top", -1))
        if rolled:
            st["ctl_diffs"] = [d for d in st.get("ctl_diffs", []) if d[0] is None or int(d[0]) < s]
            st["ctl_wait"] = {p: v for p, v in st.get("ctl_wait", {}).items() if v[2] is None or int(v[2]) < s}
            return summary() if st["ctl_diffs"] else {}
    if e.get("kind") == "control_done" and e.get("attempts") and e.get("pair"):
        refs = unit_refs(run)
        vals = [logeq(refs, e.get("set"), r) for r in try_rows(json.dumps({"attempts": e["attempts"]}, indent=1))]
        if vals:
            st.setdefault("ctl_wait", {})[e["pair"]] = [e.get("set"), sum(vals) / len(vals), e.get("step")]
    if e.get("kind") not in ("control_done", "group_done") or not st.get("ctl_wait"):
        return {}
    done = None
    for pair, (name, cmean, step) in list(st["ctl_wait"].items()):
        tries = (st.get("tries") or {}).get(pair)
        if not tries:
            continue
        refs = unit_refs(run)
        pol = [logeq(refs, name, r) for r in tries]
        st.setdefault("ctl_diffs", []).append([step, pair, round(sum(pol) / len(pol) - cmean, 6)])
        del st["ctl_wait"][pair]
        done = True
    if not done:
        return {}
    return summary()


def queue_row(run):
    q = paths.read_json(paths.ROOT_PATH / "docs/modal-queue.json")
    return next((r for r in q.get("live", []) if r.get("run") == run), {})


_HELD = {}


def hold_instance(state):
    """True when this process holds the run's exporter lock (<state>.lock, flock; released at exit)."""
    import fcntl
    if str(state) in _HELD:
        return True
    handle = open(state.with_suffix(".lock"), "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return False
    _HELD[str(state)] = handle
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--passes", required=True, type=Path, help="the RL watcher's pass-file folder")
    ap.add_argument("--base", help="a resumed leg: the run whose step-0 read anchors the ratio")
    ap.add_argument("--continues", help="the leg this one resumes (default --base); written into the meta every start")
    ap.add_argument("--out", help="the run's volume folder (default: records.py out RUN, the current attempt's)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--backfill", action="store_true", help="an earlier leg: add only the play/eval/cost schema keys")
    ap.add_argument("--backfill-keys", help="with --backfill: only these comma-separated key prefixes")
    a = ap.parse_args()
    from arc_infra.runops import records as _records
    OUT[a.run] = a.out or _records.out_id(a.run)  # default: the record's current attempt folder
    row = queue_row(a.run)
    n = str(row.get("gpu") or "").split("x")[0].strip()
    state = a.passes / f"{a.run}-export-state{'-backfill' if a.backfill else ''}{'-' + a.backfill_keys.replace(',', '_') if a.backfill_keys else ''}.json"
    if not a.once and not hold_instance(state):  # one exporter per run and mode (28 Oct: two on real-v11 shared a state)
        print(f"export: another exporter holds {state.name}; exiting", flush=True)
        return
    st = json.loads(state.read_text()) if state.exists() else {}
    parent = a.continues or a.base
    st.setdefault("until", int(time.time()))  # the branch point: this leg's first exporter start (~its launch)
    metrics.meta(a.run, arm="rl", what=row.get("what"), cards=int(n) if n.isdigit() else None,
                 continues=[{"run": parent, "until": st["until"]}] if parent else None)
    if a.base and not st.get("base"):
        for line in volume_text(FEED.format(run=a.base)).splitlines():
            if line.strip().startswith("{"):
                e = json.loads(line)
                if e.get("kind") == "read" and int(e.get("step") or 0) == 0:
                    feed_metrics(e, st)
    try:
        rec = paths.read_json(paths.ROOT_PATH / "docs/runs" / f"{a.run}.json") or {}
    except Exception:  # noqa: BLE001 - no record: no cost rows
        rec = {}
    led = [x.get("row") or {} for x in rec.get("ledger") or []]
    # the current attempt's launch (G-long v5 attempt 4, 27 Oct: attempt 1's start put every sampler row 44 min early,
    # behind the cpulog rows' real times, and all were dropped)
    start = next((r.get("start_epoch") for r in reversed(led) if r.get("start_epoch")), None)
    per = next((r.get("gpu_rate_per_second") for r in led if r.get("gpu_rate_per_second")), None)
    if per is None and any("B300" in str(r.get("gpu")) for r in led):
        per = B300_USD_S
    n_gpu = next((r.get("gpu_count") for r in led if r.get("gpu_count")), None)
    if start and per and n_gpu:
        st["rate"] = [float(start), float(per) * int(n_gpu)]
    if a.backfill:
        st["backfill"] = a.backfill_keys.split(",") if a.backfill_keys else True
    out = metrics.writer(a.run)
    if not a.backfill:
        carry_target(a.run, out)
    final = 0  # passes after the record settled
    while True:
        if not a.out:  # a relaunch moves the record to a new attempt folder: follow it, as watch.follow does
            now = _records.out_id(a.run)
            if now != OUT[a.run]:
                print(f"export: {a.run} folder {OUT[a.run]} -> {now}", flush=True)
                OUT[a.run], st["feed"] = now, 0
            what = queue_row(a.run).get("what")  # the relaunch's own description (a10 kept a9's until a restart)
            if what:
                metrics.meta(a.run, what=what)
        MISSED.clear()
        began = time.time()
        try:
            once(a.run, a.passes, st, out)
            state.write_text(json.dumps(st))
        except Exception as error:  # noqa: BLE001 - one bad pass never ends the exporter
            print(f"export pass failed: {error!r}", flush=True)
            MISSED.add("pass")
        if MISSED or time.time() - began > 180:  # a silent stretch on the dashboard is explained here
            print(f"export {time.strftime('%H:%M:%S', time.gmtime())} pass {time.time() - began:.0f} s, feed "
                  f"{st.get('feed')}, missed {sorted(MISSED)}", flush=True)
        if a.once:
            return
        if settled(a.run):  # the last pass runs after the end; a read it missed (gate-v5 28 Oct: the run's last
            final += 1      # 6 sampler rows, a skipped/slow read) is retried, up to FINAL_TRIES passes
            if not MISSED or final >= FINAL_TRIES:
                if MISSED:
                    print(f"export: {a.run} settled; reads still missing after {final} passes: {sorted(MISSED)}",
                          flush=True)
                return
            time.sleep(60)
            continue
        time.sleep(120)


FINAL_TRIES = 5


def settled(run):
    """The run's record has ended (records.py settle); an unreadable record keeps the exporter going."""
    try:
        rec = paths.read_json(paths.ROOT_PATH / "docs/runs" / f"{run}.json") or {}
    except Exception:  # noqa: BLE001
        return False
    return rec.get("state") == "ended"


if __name__ == "__main__":
    main()
