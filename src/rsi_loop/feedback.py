"""Time and noise feedback in the proposer's status: what each stage has cost so far and how noisy pool
comparisons are, so the proposer can set its own balance of selection reliability against iteration speed."""

import math
import statistics

LOOP_FACTS = """

## Loop facts (added by the loop to every session; not part of your setup)

- **The goal**. The agent is scored in the ARC-AGI-3 competition on Kaggle: 110 hidden game
  instances it has never seen, played offline (no network, no other model) on one RTX PRO 6000 host (~180 GB host
  RAM) within a 9 h runtime limit, about 30,000 s of play for the 110 games (~273 s of the GPU per game). At our
  competition build's ~1,470 output tok/s that is about 400K generated tokens per instance on average; how the
  agent spends them across games is its own business. Score: per game, each level cleared scores
  min(115, 100 x (human baseline actions / agent actions)^2), levels weighted by their number (level k counts k), a
  level not cleared scores 0; the final score is the mean over games (RHAE; here a reference solve stands in for the
  human baseline on non-public games). The loop measures your candidates on held-out games you never see; the pool
  only selects. A change tuned to a few pool games often does not transfer: prefer changes whose mechanism you can
  see in traces.
- **Correction (operator):** an earlier loop fact mentioned a 100K-token cap on held-out games. That was an operator
  error in what you were shown; the only target is the Kaggle setting above.
- **The budget meter counts GPU time, not wall time.** A game is charged its wall time / C, where C is its
  concurrency (you are one of the C sessions). By default a game's wall is C x 294.5 s (the competition's 9 h / 110), so
  it is charged 294.5 GPU-s; a higher concurrency lengthens the wall and does not cost more; how many games are queued
  or running does not change the charge. **Pool and practice game length is yours** (no token cap on
  pool games unless you set one): per candidate `player/run.json` `"game_seconds"` (wall, 60-21,600 s) and/or
  `"game_token_cap"` (generated tokens per game, 10,000-4,000,000; the game ends there, and a capped game idle for
  15 min with no model call in flight ends too). A longer game is charged its longer wall / C. A game's tokens are
  about aggregate tok/s / C x wall (gen 0: ~505 tok/s at 14 sessions, ~150K output tokens per default game; the
  competition target above is ~400K per instance).
- **Fixed pool games** (`rsi.status()["fixed_pool_games"]`): 10 public ARC-AGI-3 games, scored
  against their human baselines exactly as the competition scores them. Every candidate you release plays all 10:
  the loop queues them and your policy cannot cancel them; a candidate is eligible as `best` (the held-out pick) only
  with scores on all 10 (plus 15 pool scores in all). You see their scores and levels only, like every pool game.
  Which other pool games, how many and how often stays yours. Gen 0 and the best of that moment were re-scored on
  them.
- **Gen 0 (c0000) since 14 Oct** (owner): Prime Agent's ARC setup with its autonomous limits (12 turns / 80K tokens /
  30 min) lifted, so only the game's wall ends it (default length). It is re-scored on the 10 fixed games and on every other pool
  game a candidate has played. The earlier gen 0 is `c0000-v1`: retired, its scores and traces kept, never `best`.
- **14 Oct, a loop bug (operator's, now fixed):** from about 15:20 to 19:30 UTC the loop ended a game when its
  session began a context compaction after a turn, so most pool games in that window played only their first
  context (outcome "stopped", ~110K tokens). Those 88 scores are removed (rows kept, marked `void`) and the games are
  queued again; judge nothing on them. Your own sessions in that window could end the same way at a compaction.
- **Player memory**. The competition host has ~180 GB of host memory in all, so each player session gets
  (180 GB - 72 GiB kept for serving and the harness) / (C + 1) at its concurrency C (6.4 GiB at C = 14; you count
  twice), in `rsi.status()["budget"]["ram_ceiling_gib_per_player"]`. A process above it is killed by the kernel; in
  v26 players' own search scripts hit the old ~4.6 GB ceiling 14+ times and the Prime session then exited, ending
  the game early (outcome `exit`).
- **Game activity** (15 Oct): a game whose player sends game actions counts as active; only 15 min with no model
  call, no session output and no game action ends it (`idle`).
- **Practice games** (`rsi.status()["practice_games"]`): 152 = 70 from theredbluepill/arc-interactive, 21 from
  hoteizero/arc-witness-envs and 61 ported from VGDL (tsividis/Theory-based-RL, MIT; added 13 Oct). Practice only:
  they never score a candidate.
- **Budget.** This run is approved for 100 loop GPU-h (`rsi.status()["budget"]`); the plan caps round 1 at 1,000.
  At 100 loop GPU-h the loop stops launching games and your sessions end. Loop GPU-h sum the scoring GPUs: with two,
  a wall hour costs about two loop GPU-h.
  The machine runs in legs of about 24 h: at a leg's end the loop checkpoints and the next leg resumes it, so a leg
  end is not the end of the run. A model call of your own from your kernel (OpenAI-compatible at
  `$RSI_PROXY_URL`, key `$RSI_PROXY_KEY`, model `$RSI_MODEL`) is a cheap way to test a prompt; it counts like any
  other call.
- **How much to change per candidate, and when, is your call:** fewer changes are easier to attribute, more move
  faster.
- **Scoring reliability against iteration speed is your call**: `rsi.status()["budget"]` has loop
  GPU-h spent, approved and remaining and the number of scoring GPUs (pool and practice games play on all of them;
  their GPU-s add up on the loop meter); `rsi.status()["time"]["by_game_length"]` has, per game length in use, the
  GPU-h one game costs, the wall time and GPU-h of a minibatch, the measured noise (repeat-play sd, paired sd vs gen 0,
  standard error per game) and the paired games needed to detect a +3 / +5 / +10 gain, and per candidate its length
  and the GPU-h its pool games took. Longer games cost more per score and may be less noisy or closer to the
  competition's per-instance budget; shorter ones buy more candidates.
- **Reliability against speed, the levers.** More pool games per candidate, repeats, paired games and a higher
  concurrency make a selection decision more reliable and slower; fewer games, early stops and a lower concurrency
  make it faster and noisier. Neither gen 0's policy nor its concurrency is assumed right. `rsi.status()["time"]` has,
  on every call: loop GPU-h spent, the measured duration of each stage (your sessions, smoke, practice and pool games,
  minibatches, a full-pool pass), slot use, throughput, the server's live counters (`server_now`: running and waiting
  requests, KV use and capacity in tokens, preemptions, recent tok/s) and the noise of pool comparisons (paired sd,
  standard error by number of games, how often a candidate with no real effect clears a threshold).
- **You set:** your policy (which candidates and pool games, how many, repeats, order and pairing, early stops,
  practice games and how many); per candidate `player/run.json` `concurrency` (2-97; requests above the server's
  `max_num_seqs`, gen 0 25, wait at the server: raise it and the CUDA graph sizes in `serving/server.json`),
  `game_seconds` and `game_token_cap` (above); `hold` /
  `release`; practice length (`rsi.practice(candidate, games, seconds=S)`, 60 s up to the game wall); your session
  length (`settings.json` `session_seconds`); the games you add; your context window (`contextWindow` in your HOME's
  `.prime/agent/models.json`, now 262,144; the players' own is in the candidate).
- **Your setup changes take effect at your next session.** `rsi.self_update(path, note="why")` installs it for the
  next session; `rsi.end_session()` ends this one at once so the next starts under it (your notes and HOME persist).
  `rsi.meta_history()` lists every setup version: its note, status, the diff against its parent, its sessions and the
  candidates submitted under it with their standing. Older `rsi.py` copies: `rsi._call("end_session")`,
  `rsi._call("meta_history")`, `rsi._call("practice", candidate=c, games=[...], n=1, seconds=S)`.
- **Opt-in serving modules** (code only, nothing fitted on games; gen 0 uses none):
  - Prefix cache patches for this hybrid model: `serving/env.json` `"ARENA_PREFIX_CACHE"` = a JSON object; options
    and what each does are in `/workspace/src/serving/arena/prefix_cache_patch.py` (`OPTIONS`, `OPTIONAL`,
    `validate`). Our competition build runs `{"mamba_align_fix": true, "mamba_keep_state": true,
    "mamba_demote_superseded": true, "request_end_tail": true, "spec_decode": true, "tail_keep_boundary": true,
    "keep_reply_states": true, "tail_spec_acceptance": true, "reply_ids": true, "tail_keepalive": true,
    "state_cotenant": true, "admission_tick": 2.0}`.
  - A KV admission gate in the loop's proxy: `serving/study.json` `"rsi_gate": {"margin": 0.9}` (0.5-1.0) holds a
    model request until its estimated tokens (the session's last prompt + reply + this request's `max_tokens`) fit
    in margin x the server's KV capacity next to the requests in flight, first come first served, so the server
    does not preempt; a request alone is always admitted.
- **Fixed:** each submission plays one smoke game that the loop stops at 240 s (outcome `time` at 240 s is that stop,
  not a crash, and it cuts the request in flight; smoke traces end in `-smoke`, older ones are the job ids of
  `kind: smoke` in your events); the scorer, the held-out games and schedule, 15 bundled-pool scores for `best`, the
  drain before a stack switch (up to 20 min). Every pool play uses seed 0: a repeat differs only by sampling.
"""


def _median(values):
    return round(statistics.median(values), 1) if values else None


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _timed(row):
    """A played row (infrastructure errors carry no times)."""
    return all(isinstance(row.get(k), (int, float)) for k in ("seconds", "started", "ended"))


def _stage(rows):
    seconds = [r["seconds"] for r in rows]
    capped = [r for r in rows if r.get("wall") and r["seconds"] >= r["wall"] - 2]
    return {"games": len(rows), "median_minutes": round(_median(seconds) / 60, 1) if seconds else None,
            "slot_hours": round(sum(seconds) / 3600, 2), "share_at_wall": round(len(capped) / len(rows), 2) if rows
            else None, "median_tokens": _median([r.get("tokens") or 0 for r in rows])}


def noise(candidates, root="c0000", sizes=(8, 15, 24), pool_games=0):
    """Paired differences against ``root`` on shared games (each candidate's own mean removed), the spread of repeat
    plays of one game, and the standard error a paired mean has on n games."""
    base = (candidates.get(root) or {}).get("pool") or {}
    residuals, per = [], {}
    for cid, item in candidates.items():
        if cid == root:
            continue
        diffs = [s - base[g] for g, s in (item.get("pool") or {}).items() if g in base]
        if not diffs:
            continue
        mean = sum(diffs) / len(diffs)
        per[cid] = {"paired_games": len(diffs), "paired_mean": round(mean, 2)}
        if len(diffs) > 1:
            residuals += [d - mean for d in diffs]
    groups = sum(1 for v in per.values() if v["paired_games"] > 1)
    paired_sd = math.sqrt(sum(r * r for r in residuals) / (len(residuals) - groups)) \
        if len(residuals) - groups >= 3 else None
    repeat = []
    for item in candidates.values():
        for runs in (item.get("pool_runs") or {}).values():
            if isinstance(runs, list) and len(runs) > 1:
                m = sum(runs) / len(runs)
                repeat.append((sum((x - m) ** 2 for x in runs), len(runs) - 1))
    repeat_sd = math.sqrt(sum(a for a, _ in repeat) / sum(b for _, b in repeat)) if repeat else None
    spread = list(base.values())
    out = {"paired_sd": round(paired_sd, 1) if paired_sd else None, "paired_differences": len(residuals),
           "repeat_sd": round(repeat_sd, 1) if repeat_sd else None, "repeat_games": len(repeat),
           "game_spread_sd": round(statistics.stdev(spread), 1) if len(spread) > 1 else None}
    sd = paired_sd or out["game_spread_sd"]
    if sd:
        out["paired_se_by_games"] = {str(n): round(sd / math.sqrt(n), 1) for n in sorted({*sizes, max(1, pool_games)})}
        out["null_clears"] = {f"+{t} on {n}": round(1 - _phi(t * math.sqrt(n) / sd), 2)
                              for t in (3, 10) for n in (8, 15, 24)}
        out["null_clears_rule"] = ("chance a candidate with no real effect shows a paired mean above +t on n games: "
                                   "1 - Phi(t * sqrt(n) / sd)" + ("" if paired_sd else "; sd = game spread until "
                                                                  "there are paired games"))
        for v in per.values():
            v["se"] = round(sd / math.sqrt(v["paired_games"]), 1)
            v["z"] = round(v["paired_mean"] / v["se"], 2) if v["se"] else None
    out["vs_" + root] = per
    return out


def feedback(state, cfg, concurrency, slots, session_seconds, now, root="c0000"):
    meters = state["meters"]
    rows = [r for r in state["rows"] if not r["kind"].startswith("heldout") and _timed(r) and not r.get("void")]
    wall = concurrency * float(cfg["slice_seconds"])
    sessions = [s for s in state["proposer"]["sessions"] if s.get("tag") == "session" and s.get("ended")]
    spans = []
    for cid, item in state["candidates"].items():
        pool = [r for r in rows if r["candidate"] == cid and r["kind"] == "pool"]
        if cid == root or not pool:
            continue
        first, last = min(r["started"] for r in pool), max(r["ended"] for r in pool)
        scored = min(r["ended"] for r in pool) - item.get("created", first)
        spans.append({"candidate": cid, "pool_games": len(pool), "hours_first_to_last": round((last - first) / 3600, 2),
                      "hours_created_to_first_score": round(scored / 3600, 2)})
    pool_games = len(state.get("pool_games") or [])
    loop_s = max(1.0, meters["loop_s"])
    by_kind = {k: _stage([r for r in rows if r["kind"] == k]) for k in ("smoke", "practice", "pool")}
    by_kind["smoke"]["note"] = f"stopped at {int(cfg['smoke_seconds'])} s by the loop"
    return {
        "loop_gpu_hours": round(meters["loop_s"] / 3600, 3),
        "game_wall_minutes": round(wall / 60, 1),
        "stages": {
            "proposer_session": {"sessions": len(sessions), "setting_minutes": round(session_seconds / 60, 1),
                                 "median_minutes": _median([(s["ended"] - s["started"]) / 60 for s in sessions]),
                                 "median_tokens": _median([s.get("tokens") or 0 for s in sessions]),
                                 "runs_beside_games": True},
            **by_kind,
            "minibatch": {"recent": spans[-8:],
                          "lower_bound_hours": {str(n): round(math.ceil(n / slots) * wall / 3600, 2)
                                                for n in (8, 15, 24)},
                          "lower_bound_rule": "ceil(games / slots) x game wall, with every slot to itself"},
            "full_pool": {"games": pool_games,
                          "lower_bound_hours": round(math.ceil(pool_games / slots) * wall / 3600, 2) if pool_games
                          else None},
            "stack_switch": {"switches_hours": round(meters.get("switch_s", 0.0) / 3600, 2),
                             "drain_minutes_max": round(float(cfg["drain_seconds"]) / 60)},
        },
        "slots": {"play_slots": slots, "concurrency": concurrency,
                  "slot_hours": {k: v["slot_hours"] for k, v in by_kind.items()},
                  "idle_slot_hours": round(meters.get("idle_slot_s", 0.0) / 3600, 2),
                  "slot_use": round(1 - meters.get("idle_slot_s", 0.0) / (loop_s * slots), 3)},
        "serving": {"completion_tokens_per_second": round(meters.get("loop_tokens", 0) / loop_s, 1),
                    "actions_per_gpu_second": round(meters.get("loop_actions", 0) / loop_s, 3),
                    "per_game_tokens_per_second": _median([(r.get("tokens") or 0) / r["seconds"] for r in rows
                                                           if r["seconds"] > 60])},
        "noise": noise(state["candidates"], root, pool_games=pool_games),
        "by_game_length": by_length(state, cfg, slots, root, pool_games),
    }


def length_key(item, cfg):
    """A candidate's pool game length setting: its own game_seconds / game_token_cap, or the default wall."""
    conc = int(item.get("concurrency") or 14)
    wall = float(item.get("game_seconds") or conc * float(cfg["slice_seconds"]))
    cap = item.get("game_token_cap")
    return f"{round(wall / 60)} min" + (f" / {int(cap) // 1000}K tokens" if cap else ""), wall, conc


def by_length(state, cfg, slots, root="c0000", pool_games=0):
    """what a pool score costs and how noisy it is at each game length the candidates use, so the
    proposer can trade scoring reliability against iteration speed. Per length: the GPU-h one game is charged, the
    wall of a minibatch on the scoring slots, measured noise (repeat plays of one game; paired differences against
    gen 0) and the games needed to detect a gain; per recent candidate: its length and GPU-h spent on its pool games."""
    groups, per_candidate = {}, {}
    rows = [r for r in state["rows"] if r["kind"] == "pool" and _timed(r)]
    for cid, item in state["candidates"].items():
        if item.get("status") in ("rejected", "retired"):
            continue
        key, wall, conc = length_key(item, cfg)
        groups.setdefault(key, {"wall": wall, "conc": conc, "cands": {}})["cands"][cid] = item
        mine = [r for r in rows if r["candidate"] == cid]
        if mine or cid == root:
            per_candidate[cid] = {"length": key, "game_wall_minutes": round(wall / 60, 1),
                                  "gpu_hours_per_game": round(wall / conc / 3600, 3), "pool_games_played": len(mine),
                                  "pool_gpu_hours_spent": round(sum(r["seconds"] for r in mine) / conc / 3600, 2)}
    out = {}
    base = state["candidates"].get(root)
    for key, g in groups.items():
        cands = dict(g["cands"])
        if base is not None:
            cands.setdefault(root, base)
        n = noise(cands, root, pool_games=pool_games)
        sd = n.get("repeat_sd") or n.get("paired_sd") or n.get("game_spread_sd")
        entry = {"candidates": sorted(c for c in g["cands"]), "gpu_hours_per_game": round(g["wall"] / g["conc"] / 3600, 3),
                 "minibatch_wall_hours": {str(k): round(math.ceil(k / max(1, slots)) * g["wall"] / 3600, 2)
                                          for k in (8, 15, 24, max(1, pool_games))},
                 "minibatch_gpu_hours": {str(k): round(k * g["wall"] / g["conc"] / 3600, 2)
                                         for k in (8, 15, 24, max(1, pool_games))},
                 "repeat_sd": n.get("repeat_sd"), "repeat_games": n.get("repeat_games"),
                 "paired_sd_vs_" + root: n.get("paired_sd"), "paired_differences": n.get("paired_differences"),
                 "sd_used": sd}
        if sd:
            entry["se_per_game"] = round(sd, 1)
            # one-sided 5 %, 80 % power: n = ((1.645 + 0.842) sd / gain)^2 paired games
            entry["games_to_detect_gain"] = {f"+{t}": math.ceil(((1.645 + 0.842) * sd / t) ** 2) for t in (3, 5, 10)}
        out[key] = entry
    recent = dict(sorted(per_candidate.items(), key=lambda kv: state["candidates"][kv[0]].get("created", 0))[-12:])
    return {"lengths": out, "candidates": recent,
            "rule": "gpu_hours_per_game = wall / concurrency; minibatch wall = ceil(games / scoring slots) x wall; sd "
                    "= repeat-play sd at that length if measured, else paired sd vs gen 0, else the spread of gen 0's "
                    "scores; games_to_detect_gain: paired games for a one-sided 5% test at 80% power"}


METRICS = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
           "vllm:kv_cache_usage_perc": "kv_use", "vllm:num_preemptions_total": "preemptions",
           "vllm:generation_tokens_total": "generation_tokens_total"}


def server_metrics(text):
    """The few vLLM counters the proposer needs, from a /metrics page (first sample of each)."""
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, value = line.rpartition(" ")
        base = name.split("{", 1)[0]
        if base in METRICS and METRICS[base] not in out:
            try:
                out[METRICS[base]] = float(value)
            except ValueError:
                pass
        elif base == "vllm:cache_config_info" and "kv_capacity_tokens" not in out:
            for key in ("kv_cache_size_tokens", "num_gpu_blocks"):
                marker = key + '="'
                if marker in name:
                    raw = name.split(marker, 1)[1].split('"', 1)[0]
                    if raw.isdigit():
                        out["kv_capacity_tokens" if key == "kv_cache_size_tokens" else "kv_blocks"] = int(raw)
    if "kv_use" in out:
        out["kv_use"] = round(out["kv_use"], 3)
    return out


def meta_history(meta_root, state, diff_lines=400):
    import difflib

    items = state["meta"]["items"]
    sessions = state["proposer"]["sessions"]
    cands = state["candidates"]
    paired = noise(cands)["vs_c0000"]
    main = [x for x in sessions if x.get("tag") == "session"]

    def made_under(item):  # recorded since 13 Oct; earlier candidates: the setup of the session that submitted it
        if item.get("meta"):
            return item["meta"]
        t = item.get("created", 0)
        hit = [x["meta"] for x in main if x.get("started", 0) <= t <= (x.get("ended") or float("inf"))]
        return hit[-1] if hit else None

    out = []
    for mid, item in sorted(items.items(), key=lambda kv: kv[1].get("created", 0)):
        parent = item.get("parent")
        changed, budget = {}, max(0, min(int(diff_lines), 5000))
        here, there = meta_root / mid, meta_root / parent if parent else None
        names = sorted({p.relative_to(here).as_posix() for p in here.rglob("*") if p.is_file()} |
                       ({p.relative_to(there).as_posix() for p in there.rglob("*") if p.is_file()}
                        if there and there.is_dir() else set())) if here.is_dir() else []
        for name in names if there else []:
            a = (there / name).read_text(errors="replace").splitlines() if (there / name).is_file() else []
            b = (here / name).read_text(errors="replace").splitlines() if (here / name).is_file() else []
            if a == b:
                continue
            diff = list(difflib.unified_diff(a, b, f"{parent}/{name}", f"{mid}/{name}", lineterm="", n=1))
            changed[name] = "\n".join(diff[:budget]) + ("\n... (cut)" if len(diff) > budget else "")
            budget = max(0, budget - len(diff))
        mine = [x for x in sessions if x.get("meta") == mid and x.get("tag") == "session"]
        made = {c: {"status": v["status"], "note": v["note"][:120], "pool_games": len(v.get("pool") or {}),
                    "paired_vs_gen0": paired.get(c)}
                for c, v in cands.items() if v.get("author") == "proposer" and made_under(v) == mid}
        out.append({"meta": mid, "parent": parent, "created": item.get("created"), "status": item.get("status"),
                    "note": item.get("note", ""), "changed": changed,
                    "sessions": {"count": len(mine),
                                 "hours": round(sum((x["ended"] - x["started"]) for x in mine) / 3600, 2),
                                 "submitted": sum(x.get("submitted") or 0 for x in mine),
                                 "reasons": [x.get("reason") for x in mine][-10:]},
                    "candidates": made})
    return out
