"""RL v7 (hero) exporter -> <scratch>/train-metrics/<run>.jsonl every 120 s, on arc_infra.runops.metrics (reference
migration, infra stage 3; was scratch rl-arm/hero/hero-export.py + .rl-tools/gate/metrics_export.py).
Infra rows from the RL watcher's pass files in --passes (<run>-watch-passes.jsonl: vLLM metrics of both servers;
<run>-card-util.jsonl: card util/VRAM); learning rows from the run's feed on the volume (step, accumulate, group_done,
read, wread, envs). Usage: export_rl.py RUN --passes DIR [--base=RUN0] [--once]"""
import argparse
import json
import re
import subprocess
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
NEW_KEYS = ("train.rhae_tok_vs_ref", "train.ref_coverage", "train.rhae_per_10k", "train.rhae_mean", "train.clear_rate", "train.tokens_per_try", "train.actions_per_clear",
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
    dist = paths.ROOT_PATH / ".claude/worktrees/rl-async/dist"
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
    out["base"] = {}  # base plays per unit: the screen's tries (logeq values), plus the run's version-0 groups (st["v0"])
    for name in out["screen"]:
        rows = try_rows(volume_text(f"validation/rl1-screen-seed-v2/arena/trials/rl-loop/rl/screen-{name}/level-bench.json"))
        if rows:
            out["base"][name] = [logeq(out, name, r) for r in rows]
    REFS[run] = out
    return out


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
    if think:
        m["train.filler_per_1k"] = 1000 * filler / think
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
        for x in [x for x in gc if x.startswith("step_")]:
            gc[x] = 0


def feed_metrics(e, st):
    k, m = e.get("kind"), {}
    group_counts(e, st, m)
    if k == "step":
        m.update({"train.step": e.get("step"), "train.lr": e.get("lr"), "train.grad_norm": e.get("grad_norm"),
                  "train.policy_loss": e.get("loss"), "train.entropy": e.get("entropy"),
                  "train.tokens": e.get("trained_tokens"), "train.tries": e.get("trained"),
                  "train.update_norm_step": e.get("update_norm"), "train.approx_kl": (e.get("k3") or {}).get("approx_kl"),
                  "train.clip_frac": (e.get("k3") or {}).get("clip_frac"),
                  "train.k3_tok": (e.get("k3") or {}).get("approx_kl_tok"),
                  "time.checkpoint_s": e.get("checkpoint_seconds")})
        tis = e.get("tis_out") or {}
        if tis.get("of"):
            m["train.tis_out_share"] = float(tis.get("tokens") or 0) / float(tis["of"])
        kl = (st.get("_refs") or {}).get("target_kl")
        if kl:  # rl_async: k3 over target / over 1.5x target alerts (logged only)
            m["train.k3_target"], m["train.k3_stop"] = float(kl), 1.5 * float(kl)
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
            for suffix, keep in ((".L13", lambda r: r[0] in (1, 3)), (".all", lambda r: True)):
                vals = [logeq(refs, nm, r) for nm, r in named if nm and keep(r)]
                if vals:
                    m[f"train.rhae_tok_vs_ref{suffix}"] = math.exp(sum(vals) / len(vals))
                pool = {**refs.get("base", {}), **(st.get("v0") or {})}
                cov = [(logeq(refs, nm, r), sum(pool[nm]) / len(pool[nm])) for nm, r in named
                       if nm and keep(r) and pool.get(nm)]
                if cov and vals and False:  # retired 24 Oct (coverage set changed per step; placeholder clips): vs_first
                    pol, base = math.exp(sum(a for a, _ in cov) / len(cov)), math.exp(sum(b for _, b in cov) / len(cov))
                    m.update({f"train.rhae_tok_vs_ref_cov{suffix}": pol, f"train.rhae_tok_vs_ref_cov{suffix}_base": base,
                              f"train.rhae_tok_vs_ref_base{suffix}": base, f"train.ref_coverage{suffix}": len(cov) / len(vals)})
        m.update(vs_first(st, gids))
        st["psum"] = {}
        st["round"] = int(e.get("step") or 0) + 1
        grp = st.pop("grp", None)
        if grp and grp["tries"]:
            m["play.clear_rate"] = grp["clears"] / grp["tries"]
            m["play.band_frac"] = grp["band"] / max(1, grp["groups"])
            m["play.groups"] = grp["groups"]
    elif k == "accumulate":
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
        m.update({"train.live_groups_done": a["groups"], "train.live_groups_total": BATCH_GROUPS,
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


def volume_text(path):
    try:
        return subprocess.run([MODAL, "volume", "get", "--force", "arc-prolong-runs", path, "-"], capture_output=True,
                              text=True, timeout=120).stdout
    except (subprocess.SubprocessError, OSError):
        return ""


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
    lines = [x for x in volume_text(FEED.format(run=run)).splitlines() if x.strip()]
    for line in lines[st.get("feed", 0):]:
        try:
            e = json.loads(line)
        except ValueError:
            break  # a torn last line: read again next pass
        st["feed"] = st.get("feed", 0) + 1
        def fetch(gid, name):
            got = try_rows(volume_text(GROUP.format(run=run, gid=gid, set=name)))
            if got:
                return got
            # a live run: finished tries under partial/, a resumed group's split over the legs that kept them (hero-v6g
            # step 3: 65 of 94 trained tries without the earlier legs'); one row per label, the latest leg's
            by = {}
            for leg in reversed(lineage(run)):
                for line in volume_text(PARTIAL.format(run=leg, gid=gid, set=name)).splitlines():
                    try:
                        a = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(a, dict) and a.get("label"):
                        by[a["label"]] = line
            return try_rows("\n".join(by.values()))

        if e.get("kind") == "group_done":  # name and try count always: a seeded/restarted state may hold old tries
            st.setdefault("gset", {})[e["gid"]] = e.get("set")
            st.setdefault("gtries", {})[e["gid"]] = int(e.get("tries") or 12)
        if e.get("kind") == "group_done" and e.get("gid") not in st.setdefault("tries", {}):
            st["tries"][e["gid"]] = fetch(e["gid"], e.get("set"))
        if e.get("kind") == "group_done" and e.get("set") and e.get("set") not in st.setdefault("first", {}) \
                and st["tries"].get(e["gid"]):  # the level's first group (seeded from earlier legs at a restart)
            st["first"][e["set"]] = [sum(r[4] for r in st["tries"][e["gid"]]), sum(r[1] for r in st["tries"][e["gid"]])]
        if e.get("kind") == "step":  # the step's groups again: a live run's tries.jsonl lags its group_done row
            for gid in st.get("step_gids", []):
                name = (st.get("gset") or {}).get(gid)
                if name and len(st["tries"].get(gid) or []) < (st.get("gtries") or {}).get(gid, 12):
                    st["tries"][gid] = fetch(gid, name) or st["tries"].get(gid) or []
        if e.get("kind") == "group_done" and int(e.get("version") or 0) == 0 and st["tries"].get(e.get("gid")):
            refs0 = unit_refs(run)
            st.setdefault("v0", {}).setdefault(e.get("set"), []).extend(logeq(refs0, e.get("set"), r)
                                                                         for r in st["tries"][e["gid"]])
        if e.get("kind") == "step":
            st["_refs"] = unit_refs(run)
        m = feed_metrics(e, st)
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


def queue_row(run):
    q = paths.read_json(paths.ROOT_PATH / "docs/modal-queue.json")
    return next((r for r in q.get("live", []) if r.get("run") == run), {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--passes", required=True, type=Path, help="the RL watcher's pass-file folder")
    ap.add_argument("--base", help="a resumed leg: the run whose step-0 read anchors the ratio")
    ap.add_argument("--continues", help="the leg this one resumes (default --base); written into the meta every start")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--backfill", action="store_true", help="an earlier leg: add only the play/eval/cost schema keys")
    ap.add_argument("--backfill-keys", help="with --backfill: only these comma-separated key prefixes")
    a = ap.parse_args()
    row = queue_row(a.run)
    n = str(row.get("gpu") or "").split("x")[0].strip()
    state = a.passes / f"{a.run}-export-state{'-backfill' if a.backfill else ''}{'-' + a.backfill_keys.replace(',', '_') if a.backfill_keys else ''}.json"
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
    start = next((r.get("start_epoch") for r in led if r.get("start_epoch")), None)
    per = next((r.get("gpu_rate_per_second") for r in led if r.get("gpu_rate_per_second")), None)
    if per is None and any("B300" in str(r.get("gpu")) for r in led):
        per = B300_USD_S
    n_gpu = next((r.get("gpu_count") for r in led if r.get("gpu_count")), None)
    if start and per and n_gpu:
        st["rate"] = [float(start), float(per) * int(n_gpu)]
    if a.backfill:
        st["backfill"] = a.backfill_keys.split(",") if a.backfill_keys else True
    out = metrics.writer(a.run)
    while True:
        try:
            once(a.run, a.passes, st, out)
            state.write_text(json.dumps(st))
        except Exception as error:  # noqa: BLE001 - one bad pass never ends the exporter
            print(f"export pass failed: {error!r}", flush=True)
        if a.once:
            return
        time.sleep(120)


if __name__ == "__main__":
    main()
