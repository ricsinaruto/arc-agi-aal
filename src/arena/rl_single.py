"""One-level RL: sb26 level ``l2_level`` only, fresh adapter, continuous token reward.

- starts: the level's starts in a stopped run's recorded base games (``lvl_resume.folder``/games-00), before the
  game's first compaction or timed-out python call; ``l2_held`` of them held out for evaluation, the rest trained
  round-robin, ``l2_starts`` per round;
- tries: ``l2_tries`` restored tries per train start and ``l2_eval_tries`` per held-out start, every try capped at
  ``l2_cap`` tokens (bounds play time only);
- reward: a clear scores T_ref / tokens (T_ref: the mean tokens of the group's clears), no clear 0; advantage:
  leave-one-out inside the start's group; per group the ``l2_per_side`` largest positive and negative advantages are
  trained (cost), positives as plain likelihood, negatives with the ratio truncated to [0, 1] (TOPR), each try's mean
  log-likelihood counted once (per-try normalisation), one round of lag (play on GPU 0, training on GPU 1);
- metric each round: on the held-out starts, clear rate within the cap and median tokens to clear (round 1 is the
  base: its adapter is the fresh one);
- final: ``l2_final_tries`` tries per held-out start for the adapter and the base on the same seeds, plus
  ``l2_final_games`` whole games per arm.
"""

import asyncio
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import time
from pathlib import Path

from .harvest import state as harvest_state
from .rl_ladder import game_metrics, level_of


def lag_cost(windows, lag_of):
    """Per policy lag (mg.lookahead_below): trained windows and tokens, the GSPO replies' mean TIS factor and capped
    share, and their clipped share at the run's band and at 0.004 (window names: the spec's, ``#i`` per split)."""
    out = {}
    for w in windows:
        base = re.sub(r"#\d+$", "", str(w.get("name", "")))
        s = out.setdefault(str(lag_of.get(base, "unknown")), {"windows": 0, "tokens": 0, "replies": 0, "tis_sum": 0.0,
                                                             "tis_capped": 0, "clipped": 0, "clipped_0.004": 0})
        g = (w.get("signed") or {}).get("gspo") or {}
        r = int(g.get("replies") or 0)
        s["windows"] += 1
        s["tokens"] += int(w.get("trained") or 0)
        s["replies"] += r
        s["tis_sum"] += float(g.get("tis_mean") or 1.0) * r
        s["tis_capped"] += int(g.get("tis_capped") or 0)
        s["clipped"] += int(g.get("clipped") or 0)
        s["clipped_0.004"] += int((g.get("clipped_alt") or {}).get("0.004") or 0)
    total = sum(s["tokens"] for s in out.values()) or 1
    for s in out.values():
        r = s.pop("replies")
        tis = s.pop("tis_sum")
        s.update(token_share=round(s["tokens"] / total, 4), replies=r, tis_mean=round(tis / r, 4) if r else None,
                 tis_capped_share=round(s.pop("tis_capped") / r, 4) if r else None,
                 clip_share=round(s.pop("clipped") / r, 4) if r else None,
                 clip_share_0004=round(s.pop("clipped_0.004") / r, 4) if r else None)
    return out


def turn_try(att, rf):
    """One try's per-turn inputs for mg_turn_credit (replies, end step, flagged replies), parsed from its log and
    rollout. Module level: a process pool runs it (pfP2 profile, 22 Oct: the JSON decode of 200 tries' rollouts was
    60% of the ~3-min start-up setup, on the event loop)."""
    from .restore import frame_hash, log_rows

    att = Path(att)
    replies, end, flagged = [], None, {}
    try:
        obs = {int(o["step"]): (int(o.get("levels") or 0), frame_hash(o["frames"][-1]))
               for o in log_rows(att / "log.jsonl", ("observation",)) if o.get("frames")}
        # the try ends at its clear (a harvested try runs the clearing reply's last actions and one more request on
        # the next level); no clear: the last observation
        clear = log_rows(att / "log.jsonl", ("level_bench_clear",))
        end = int(clear[0]["step"]) if clear else (max(obs) if obs else None)
        with open(att / "rollout.jsonl") as file:
            for line in file:
                r = json.loads(line)
                if r.get("scripted"):
                    continue
                step = int(r.get("step") or 0)
                anchor = obs.get(step) or obs.get(max([s for s in obs if s <= step] or [0]))
                replies.append((int(r["n"]), int((r.get("usage") or {}).get("completion_tokens") or 0), anchor, step))
                if rf:  # pilot 1 overlong/loop filter: such a reply never gets positive weight
                    kind = MultiMixin.mg_reply_flag(r, rf)
                    if kind:
                        from .filler import reply_filler

                        toks = r.get("tokens") or []
                        flagged[int(r["n"])] = {"kind": kind, "tokens": len(toks), "think": reply_filler(toks)[1]}
    except (OSError, ValueError, KeyError):
        replies = []
    return replies, end, flagged


TURN_CACHE, TURN_POOL = {}, [None]


def turn_prefetch(atts, rf):
    """Parse every listed try not yet in TURN_CACHE in a process pool (16 workers): {path: turn_try(...)}."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    key = json.dumps(rf or {}, sort_keys=True)
    todo = [str(a) for a in atts if (str(a), key) not in TURN_CACHE]
    if len(todo) < 8:
        return 0
    if TURN_POOL[0] is None:
        TURN_POOL[0] = ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 4),
                                           mp_context=multiprocessing.get_context("forkserver"))
    if len(TURN_CACHE) > 4096:
        TURN_CACHE.clear()
    for att, got in zip(todo, TURN_POOL[0].map(turn_try, todo, [rf] * len(todo), chunksize=4)):
        TURN_CACHE[(att, key)] = got
    return len(todo)


RELOAD_EPS, RELOAD_CEIL = 2e-4, 5e-3  # mg.reload_check: pass when the reload diff <= 3 x repeat floor + EPS, <= CEIL
LOOKAHEAD_MIN_S = 60  # mg.lookahead_below: no look-ahead in a round's first minute (its benches still register)


def park_record():
    """v5 boot cut: the volume record of a passed park check for this trainer key (ARC_RL1_TRAINER_KEY, the launcher's
    trainer_key); None without a key or a volume."""
    key, root = os.environ.get("ARC_RL1_TRAINER_KEY"), Path(os.environ.get("ARC_PARK_RECORDS") or "/runs/park-checks")
    if not key or not root.parent.is_dir():
        return None
    return root / f"{key[:32]}.json"


def boot_phases(text):
    """vLLM's own boot timings from its log (v5: the compile share apart from weights and graphs), seconds."""
    def last(pattern):
        found = re.findall(pattern, text)
        return float(found[-1]) if found else None

    return {"weights_s": last(r"Model loading took [0-9.]+ GiB memory and ([0-9.]+) seconds"),
            "load_weights_s": last(r"Loading weights took ([0-9.]+) seconds"),
            "engine_init_s": last(r"init engine \(profile, create kv cache, warmup model\) took ([0-9.]+) s"),
            "compile_s": last(r"\(compilation: ([0-9.]+) s\)") or last(r"torch.compile takes ([0-9.]+) s in total"),
            "dynamo_s": last(r"Dynamo bytecode transform time: ([0-9.]+) s"),
            "graphs_s": last(r"Graph capturing finished in ([0-9.]+) secs"),
            "compiled_from_cache": "Directly load the compiled graph" in text or "from the cache" in text}


def state_uid(state):
    """The policy uid of a saved round state: recorded, else a digest of the state (a pre-uid source resumed twice
    gets the same uid both times)."""
    import hashlib

    return state.get("policy_uid") or hashlib.sha1(json.dumps(
        {k: v for k, v in state.items() if k != "policy_uid"}, sort_keys=True, default=str).encode()).hexdigest()[:12]


def restored(row):
    """A try usable as played from its start: no restore, a clean restore, or a restore whose game state matched the
    record (board, steps, levels) though a prompt message differed (a strict-check drop is not a
    broken start); a replay whose game state diverged is not."""
    rs = row.get("restore")
    if not rs or rs.get("ok"):
        return True
    return bool((rs.get("state") or {}).get("ok")) and row.get("outcome") != "replay_mismatch"


def clear_tokens(row):
    """Tokens to clear the level for a cleared try, else None."""
    level_rows = row.get("level_rows") or []
    if row.get("outcome") not in ("cleared", "won") or not level_rows:
        return None
    return int(level_rows[0].get("tokens") or row.get("completion_tokens") or 0) or None


def branch_rem_terms(rem, t, b, cap):
    """(RHAE term, token term) of a branch continuation scored on its remainder (``branch.score_rows``): actions and
    tokens after the cut against the medians of its cut group's clears. RHAE's cap still binds on the whole level:
    remainder actions count as at least the ones that bring the level to the cap (baseline / sqrt(cap) - prefix)."""
    floor = max(1.0, b / math.sqrt(cap) - rem["pre"]) if b else 1.0
    a, a_med = max(floor, float(rem["a"])), max(floor, float(rem["a_med"]))
    return 2.0 * math.log(a_med / a), math.log(float(rem["t_med"]) / t)


def clear_rhae(row):
    """The cleared level's score as a fraction (RHAE, min(1.15, (baseline / actions)^2)), else None."""
    level_rows = row.get("level_rows") or []
    if row.get("outcome") not in ("cleared", "won") or not level_rows or level_rows[0].get("score") is None:
        return None
    return float(level_rows[0]["score"]) / 100.0


def group_advantages(rows, reward="tokens"):
    """Per try: reward and leave-one-out advantage. ``tokens``: T_ref / tokens for a clear (T_ref: the mean tokens of
    the group's clears), 0 otherwise; ``rhae_tokens``: the level's RHAE x T_ref / tokens (score per
    token, scale-free)."""
    tokens = [clear_tokens(r) for r in rows]
    clears = [t for t in tokens if t]
    if not clears or len(rows) < 2:
        return [0.0] * len(rows), [0.0] * len(rows)
    ref = statistics.mean(clears)
    rewards = [ref / t if t else 0.0 for t in tokens]
    if reward == "rhae_tokens":
        rewards = [r * (clear_rhae(row) or 0.0) for r, row in zip(rewards, rows)]
    total, n = sum(rewards), len(rewards)
    return rewards, [r - (total - r) / (n - 1) for r in rewards]


CITES = re.compile(r"\b(my|the|these|those|earlier|previous|old) notes?\b|\bhint\b|earlier play|previous play|"
                   r"notes (say|said|show)|from (my|the) notes", re.I)


def hint_citations(folder, rows):
    """Per try, the share of its live replies whose reasoning cites the hint (``CITES``): (tries citing at least
    once, live replies citing, live replies)."""
    tries = cited = replies = 0
    for row in rows:
        path = Path(folder) / "attempts" / row["label"] / "rollout.jsonl"
        if not path.is_file():
            continue
        hit = False
        with open(path) as file:
            for line in file:
                reply = json.loads(line)
                if reply.get("scripted"):
                    continue
                text = (reply.get("reply") or {}).get("reasoning") or (reply.get("reply") or {}).get("reasoning_content") or ""
                replies += 1
                if CITES.search(text):
                    cited += 1
                    hit = True
        tries += hit
    return {"tries_citing": tries, "of_tries": len(rows), "replies_citing": cited, "replies": replies,
            "reply_share": round(cited / replies, 4) if replies else None}


def held_metrics(rows):
    toks = [clear_tokens(r) for r in rows]
    clears = [t for t in toks if t]
    rhae = [v for v in (clear_rhae(r) for r in rows) if v is not None]
    actions = [int(r["level_rows"][0].get("actions") or 0) for r in rows if clear_tokens(r)]
    spent = sum(int(r.get("completion_tokens") or 0) for r in rows)
    se = None
    if spent and len(rows) > 1:  # v6x: the ratio estimate's standard error (delta method over tries)
        ratio = sum(rhae) / spent
        dev = [(clear_rhae(r) or 0.0) - ratio * int(r.get("completion_tokens") or 0) for r in rows]
        se = round(math.sqrt(len(rows) / (len(rows) - 1) * sum(d * d for d in dev)) / spent * 1e4, 4)
    return {"tries": len(rows), "clears": len(clears), "clear_rate": round(len(clears) / len(rows), 3) if rows else None,
            "median_to_clear": statistics.median(clears) if clears else None,
            "median_actions": statistics.median(actions) if actions else None,
            "mean_rhae": round(statistics.mean(rhae), 4) if rhae else None,
            "rhae_per_10k": round(sum(rhae) / spent * 1e4, 4) if spent else None, "rhae_10k_se": se,
            "tokens_per_try": round(spent / len(rows), 1) if rows else None}


class SingleLevelMixin:
    def l2_start_pool(self):
        from .restore import level_starts, log_rows, unsafe_from
        from .rl_loop import usable

        level = int(self.cfg["l2_level"])
        root = getattr(self, "l2_source", None) or Path(self.cfg["lvl_resume"]["folder"]) / "games-00"
        rows = [r for r in json.loads((root / "level-bench.json").read_text())["attempts"] if usable(r)]
        out = []
        for row in sorted(self.game_rows(rows, self.cfg["train_game"]), key=lambda r: r["label"]):
            attempt = root / "attempts" / row["label"]
            first, _ = unsafe_from(attempt)  # summary, handoff, interrupt, forced drain
            for request, done, step in level_starts(attempt):
                if done + 1 == level and (first is None or step < first):
                    out.append((attempt, int(request)))
        rng = random.Random(int(self.cfg["l2_seed"]))
        rng.shuffle(out)
        held = out[: int(self.cfg["l2_held"])]
        return held, out[int(self.cfg["l2_held"]):], rows

    def l2_specs(self, picks, kind, cap):
        level = int(self.cfg["l2_level"])
        return [(attempt, request, f"L{level}-{kind}-{attempt.name[:4]}-k{request:03d}", False, cap)
                for attempt, request in picks]

    def l2_groups(self, rows):
        groups = {}
        for r in self.game_rows(rows, self.cfg["train_game"]):
            if (r.get("restore") or {}).get("ok") and level_of(r) is not None:
                groups.setdefault(r["point"], []).append(r)
        return groups

    async def l2_record(self):
        """Base games from level 1, each stopped by a small token budget soon after level ``l2_level`` begins: one
        start per game (the source of ``l2_start_pool``)."""
        rec = self.cfg["l2_record"]
        g = self.cfg["train_game"]
        folder, rows, play = await self.bench("l2-record", [g], int(rec["games"]), {g: int(rec["budget"])},
                                              model=self.server.model, seed_base=int(rec.get("seed", 9000)),
                                              concurrency=int(rec["games"]), reserve=120)
        self.l2_source = Path(folder)
        self.feed.emit("l2_record", folder=str(folder), games=len(rows), play=play,
                       reached=sum(1 for r in rows if len(r.get("level_rows") or []) >= int(self.cfg["l2_level"]) - 1))
        return folder

    async def l2_rounds(self, start, probes):
        g, cap = self.cfg["train_game"], int(self.cfg["l2_cap"])
        if self.cfg["l2_record"] and getattr(self, "l2_source", None) is None:
            await self.l2_record()
        held, train, _ = self.l2_start_pool()
        if not train or not held:
            raise RuntimeError(f"level {self.cfg['l2_level']}: {len(train)} train and {len(held)} held-out starts")
        self.l2_held_starts = held
        self.feed.emit("l2_starts", held=[f"{a.name[:4]}-k{q:03d}" for a, q in held],
                       train=[f"{a.name[:4]}-k{q:03d}" for a, q in train], cap=cap)
        if self.serving_adapter is None:
            self.serving_adapter = self.adapters / "r00"
        kept, stop, played_by, k, streak = None, None, start - 1, int(self.cfg["l2_starts"]), 0
        self.l2_base = None
        for number in range(start, self.cfg["rounds"] + 1):
            began = asyncio.get_event_loop().time()
            picks = [train[((number - 1) * k + i) % len(train)] for i in range(min(k, len(train)))]
            tpath, tpoints = self.restore_points(self.l2_specs(picks, "T", cap), f"l2-train-{number:02d}")
            hpath, hpoints = self.restore_points(self.l2_specs(held, "H", cap), f"l2-held-{number:02d}")
            tries, etries = int(self.cfg["l2_tries"]), int(self.cfg["l2_eval_tries"])
            jobs = [self.bench(f"l2-train-{number:02d}", [g], tries, {g: cap}, points=tpath,
                               concurrency=len(tpoints) * tries),
                    self.bench(f"l2-held-{number:02d}", [g], etries, {g: cap}, points=hpath,
                               concurrency=len(hpoints) * etries, seed_base=int(self.cfg["l2_eval_seed"]))]
            if kept:
                jobs.append(self.l2_update(number, kept, probes))
            results = await asyncio.gather(*jobs)
            (tfolder, trows, tplay), (hfolder, hrows, hplay) = results[0], results[1]
            row = {"round": number, "played_by": played_by}
            if kept:
                update_row, serve = results[2]
                row["update"] = {k2: v for k2, v in update_row.items() if k2 not in ("traceback", "windows")}
                if not update_row.get("ok"):
                    stop = row["stop"] = update_row.get("stop") or "update_failed"
                    self.feed.emit("round", **row)
                    break
                kl = [p.get("kl_base") for p in (update_row.get("probes") or {}).values() if p.get("kl_base") is not None]
                limit = self.cfg["l2_kl_stop"]
                if limit is not None and kl and max(kl) > float(limit):
                    # KL guard: the new adapter is not served; the run ends after this round's
                    # row and the final is played by the last adapter within the limit
                    row["kl_guard"] = {"kl_base_max": max(kl), "limit": float(limit), "kept_serving": str(self.serving_adapter)}
                    stop = "kl_guard"
                else:
                    row["swap"] = {"load_seconds": await self.server.load("policy", serve)}
                    self.serving_adapter = serve
                    played_by = number
            # held-out metric (round 1: the fresh adapter, i.e. the base)
            hgroups = self.l2_groups(hrows)
            metric = held_metrics([r for grp in hgroups.values() for r in grp])
            metric["per_start"] = {p.split("-H-", 1)[-1]: held_metrics(grp) for p, grp in sorted(hgroups.items())}
            if self.l2_base is None:
                self.l2_base = metric
            row["held"] = metric
            bad = self.l2_bad_round(metric) if row["played_by"] > 0 else None  # the adapter that played this round
            row["early_stop_signal"] = bad
            streak = streak + 1 if bad else 0
            # train groups: rewards, leave-one-out advantages, the largest of each sign
            groups = self.l2_groups(trows)
            selected, weights, info, up, down = [], [], {}, 0.0, 0.0
            side = int(self.cfg["l2_per_side"])
            for point, grp in sorted(groups.items()):
                rewards, adv = group_advantages(grp, self.cfg["l2_reward"])
                order = sorted(range(len(grp)), key=lambda i: -adv[i])
                pick = [i for i in order[:side] if adv[i] > 0] + [i for i in order[::-1][:side] if adv[i] < 0]
                selected += [grp[i] for i in pick]
                weights += [adv[i] for i in pick]
                up += sum(adv[i] for i in pick if adv[i] > 0)
                down += sum(-adv[i] for i in pick if adv[i] < 0)
                m = held_metrics(grp)
                info[point.split("-T-", 1)[-1]] = {**m, "trained": len(pick)}
            all_rows = json.loads((Path(tfolder) / "level-bench.json").read_text())["attempts"]
            failed = sum(1 for r in all_rows if not (r.get("restore") or {}).get("ok"))
            row["train"] = {"play": tplay, "held_play": hplay, "restored": len(all_rows) - failed, "of": len(all_rows),
                            "starts": info, "trained": len(selected),
                            "balance": {"up": round(up, 4), "down": round(down, 4),
                                        "up_share": round(up / (up + down), 4) if up + down else None}}
            row["round_seconds"] = round(asyncio.get_event_loop().time() - began, 1)
            kept = (Path(tfolder), selected, weights) if selected else None
            self.feed.emit("round", **row)
            self.history.append(row)
            if failed > float(self.cfg["lvl_restore_fail"]) * max(1, len(all_rows)) and not info:
                stop = "restore"
                break
            if stop == "kl_guard":
                self.feed.emit("kl_guard", round=number, **row["kl_guard"])
                break
            if streak >= int(self.cfg["l2_stop_rounds"]):
                stop = "early_stop"
                self.feed.emit("early_stop", round=number, signal=bad, base=self.l2_base)
                break
            if self.left() < 3600 * float(self.cfg["l2_final_hours"]):
                stop = "wall"
                break
        return stop

    def l2_bad_round(self, metric):
        """Why a policy-played round counts toward the early stop, else None."""
        rate, ratio = self.cfg["l2_stop_rate"], self.cfg["l2_stop_median"]
        base = (self.l2_base or {}).get("median_to_clear")
        if rate is not None and metric["clear_rate"] is not None and metric["clear_rate"] <= float(rate):
            return f"clear rate {metric['clear_rate']} <= {rate}"
        if ratio is not None and base and metric["median_to_clear"] and metric["median_to_clear"] > float(ratio) * base:
            return f"median {metric['median_to_clear']} > {ratio} x base {base}"
        return None

    async def l2_update(self, number, kept, probes):
        """One pass (``l2_passes``) over the selected tries: one optimizer step per start (its tries share the
        prefix); signed (negatives TOPR), per-try normalisation."""
        folder, rows, weights = kept
        specs = [{**s, "w": float(w), "whole": True, "window": int(self.cfg["lvl_window"]), "floor": 1024,
                  "margin": 1024, "credited_only": True, "share": r.get("point") or r["label"]}
                 for s, w, r in zip(self.windows(folder, rows, prefix="t-"), weights, rows) if w != 0]
        groups = {}
        for spec in specs:
            groups.setdefault(spec["share"], []).append(spec)
        rng, batches, per = random.Random(number), [], max(1, int(self.cfg["l2_starts_per_step"]))
        for _ in range(int(self.cfg["l2_passes"])):
            keys = sorted(groups)
            rng.shuffle(keys)
            # one optimizer step per l2_starts_per_step starts (each start's tries stay together: shared prefix)
            batches += [[spec for key in keys[i:i + per] for spec in groups[key]] for i in range(0, len(keys), per)]
        merged, path = await self.dosed_update(number, None, None, None, probes, batches=batches, signed=True,
                                               probe_last=True, max_windows=100000,
                                               cmd_extra={"share_prefix": True,
                                                          "prefix_chunk": int(self.cfg["path_chunk"]),
                                                          "norm": self.cfg["l2_norm"],
                                                          "norm_cap": int(self.cfg["l2_cap"])})
        merged["trained_tries"] = len(specs)
        if merged.get("ok"):
            self.checkpoint(number)
            await self.save_checkpoint()
        return merged, path

    async def l2_finish(self, stop):
        """Paired held-out tries (adapter and base, same seeds) and whole games per arm."""
        if stop in ("nan_or_inf", "update_failed"):
            self.feed.emit("final", stop=stop, eval=None)
            return stop
        g, cap = self.cfg["train_game"], int(self.cfg["l2_cap"])
        n, seed = int(self.cfg["l2_final_tries"]), int(self.cfg["l2_final_seed"])
        path, points = self.restore_points(self.l2_specs(self.l2_held_starts, "H", cap), "l2-final")
        names = {"policy": "policy", "base": self.server.model}
        jobs = [self.bench(f"final-l2-{tag}", [g], n, {g: cap}, points=path, model=model, seed_base=seed,
                           concurrency=len(points) * n, reserve=120) for tag, model in names.items()]
        games = int(self.cfg["l2_final_games"])
        if games:
            jobs += [self.bench(f"final-games-{tag}", [g], games, {g: int(self.cfg["lvl_game_budget"])},
                                seed_base=int(self.cfg["lvl_final_seed"]), model=model, concurrency=games, reserve=120,
                                wall=int(self.cfg["lvl_game_wall"])) for tag, model in names.items()]
        results = await asyncio.gather(*jobs)
        out = {"seed_base": seed, "tries_per_start": n, "cap": cap}
        per = {}
        for (tag, _), (_, rows, play) in zip(names.items(), results[:2]):
            groups = self.l2_groups(rows)
            flat = [r for grp in groups.values() for r in grp]
            out[tag] = held_metrics(flat)
            out[f"{tag}_play"] = play
            per[tag] = {p.split("-H-", 1)[-1]: held_metrics(grp) for p, grp in groups.items()}
        out["per_start"] = {s: {"policy": per["policy"].get(s), "base": per["base"].get(s)}
                            for s in sorted(set(per["policy"]) | set(per["base"]))}
        pol, base = out["policy"], out["base"]
        if pol["median_to_clear"] and base["median_to_clear"]:
            out["token_ratio"] = round(base["median_to_clear"] / pol["median_to_clear"], 3)
        if games:
            for (tag, _), (_, rows, play) in zip(names.items(), results[2:]):
                metrics = game_metrics(self.game_rows(rows, g))
                out[f"games_{tag}"] = {k: v for k, v in metrics.items() if k != "per_game"}
        self.feed.emit("eval", label="final", **out)
        self.feed.emit("final", stop=stop, eval=out, base_round1=self.l2_base)
        return stop


def final_pair_key(row):
    """A final-read try's start and seed: a restored start by its recorded attempt's folder name and request (point ids
    hash the attempt path, which differs between a screen's container copy and its volume copy: pff1r paired 0),
    else its point; with the pass index (seed = seed_base + pass)."""
    rs, p = row.get("restore") or {}, row["label"].rsplit("-p", 1)[-1]
    if rs.get("source") is not None:
        return Path(str(rs["source"])).name, int(rs.get("request") or 0), p
    return row["point"], p


def paired_summary(pairs, cap):
    """``pairs`` [(policy row, base row)] on the same start and seed; a try without a clear costs ``cap``; per pair
    the log of base / policy tokens."""
    import math

    cost = lambda r: clear_tokens(r) or cap
    logs = [math.log(cost(b) / cost(p)) for p, b in pairs]
    both = [(clear_tokens(p), clear_tokens(b)) for p, b in pairs]
    if not logs:
        return {"pairs": 0}
    mean = statistics.mean(logs)
    se = statistics.stdev(logs) / math.sqrt(len(logs)) if len(logs) > 1 else None
    return {"pairs": len(logs), "mean_log_base_over_policy": round(mean, 4), "se": None if se is None else round(se, 4),
            "token_ratio_geo": round(math.exp(mean), 3), "policy_cheaper": sum(v > 0 for v in logs),
            "policy_only_clear": sum(1 for p, b in both if p and not b), "base_only_clear": sum(1 for p, b in both if b and not p)}


def start_half(label):
    """0 or 1: which half of the recordings a recorded game belongs to (stable across runs)."""
    import hashlib

    return int(hashlib.md5(str(label).encode()).hexdigest(), 16) % 2


class TransferMixin:
    """Transfer check: the adapter and the base on restored starts of levels the adapter never
    trained on, the same starts and seeds for both arms."""

    def transfer_starts(self, spec):
        from .restore import level_starts, log_rows, unsafe_from
        from .rl_loop import usable

        pool = {}
        for src in spec["sources"]:
            root = Path(src)
            rows = [r for r in json.loads((root / "level-bench.json").read_text())["attempts"] if usable(r)]
            for row in sorted(self.game_rows(rows, self.cfg["train_game"]), key=lambda r: r["label"]):
                attempt = root / "attempts" / row["label"]
                first, _ = unsafe_from(attempt)  # summary, handoff, interrupt, forced drain
                spent = {int(l["level"]): l.get("tokens") for l in row.get("level_rows") or []}
                with open(attempt / "rollout.jsonl") as file:
                    prompts = [int((json.loads(line).get("usage") or {}).get("prompt_tokens") or 0) for line in file]
                for request, done, step in level_starts(attempt):
                    # max_prompt: starts whose prompt leaves room in the cache for every try at once
                    if (done + 1 in spec["levels"] and (first is None or step < first)
                            and prompts[int(request)] <= int(spec.get("max_prompt") or 10**9)):
                        pool.setdefault(done + 1, []).append((attempt, int(request), spent.get(done + 1)))
        picks = {}
        for level in spec["levels"]:
            cands = pool.get(level, [])
            random.Random(int(spec.get("pick_seed", 0)) * 100 + level).shuffle(cands)
            picks[level] = cands[: int(spec["starts"])]
        return pool, picks

    async def transfer_eval(self, spec):
        g, cap, n, seed = self.cfg["train_game"], int(spec["cap"]), int(spec["tries"]), int(spec["seed"])
        pool, picks = self.transfer_starts(spec)
        specs = [(a, q, f"L{level}-X-{a.parent.parent.name[-2:]}{a.name[:4]}-k{q:03d}", False, cap)
                 for level, chosen in sorted(picks.items()) for a, q, _ in chosen]
        self.feed.emit("transfer_starts", available={k: len(v) for k, v in sorted(pool.items())},
                       picked={k: [{"start": f"{a.parent.parent.name}/{a.name}/k{q:03d}", "source_tokens": t}
                                   for a, q, t in v] for k, v in sorted(picks.items())}, cap=cap, tries=n, seed=seed)
        path, points = self.restore_points(specs, "transfer")
        names = {"policy": "policy", "base": self.server.model}
        results = await asyncio.gather(*(self.bench(f"transfer-{tag}", [g], n, {g: cap}, points=path, model=model,
                                                    seed_base=seed, concurrency=len(points) * n, reserve=120)
                                         for tag, model in names.items()))
        out = {"cap": cap, "tries_per_start": n, "seed_base": seed, "starts": len(points)}
        by = {}
        for (tag, _), (folder, rows, play) in zip(names.items(), results):
            out[f"{tag}_play"] = play
            all_rows = json.loads((Path(folder) / "level-bench.json").read_text())["attempts"]
            out[f"{tag}_unusable"] = len(all_rows) - len(rows)
            groups = self.l2_groups(rows)
            flat = [r for grp in groups.values() for r in grp]
            out[tag] = held_metrics(flat)
            for r in flat:
                by.setdefault(tag, {})[(r["point"], r.get("pass"), r["label"].rsplit("-p", 1)[-1])] = r
            per_level = {}
            for r in flat:
                per_level.setdefault(level_of(r), []).append(r)
            out.setdefault("per_level", {})
            for level, grp in sorted(per_level.items()):
                out["per_level"].setdefault(str(level), {})[tag] = held_metrics(grp)
        keys = sorted(set(by.get("policy", {})) & set(by.get("base", {})), key=str)
        out["paired"] = paired_summary([(by["policy"][k], by["base"][k]) for k in keys], cap)
        for level in sorted({level_of(by["policy"][k]) for k in keys}):
            out["per_level"][str(level)]["paired"] = paired_summary(
                [(by["policy"][k], by["base"][k]) for k in keys if level_of(by["policy"][k]) == level], cap)
        self.feed.emit("eval", label="transfer", **out)
        return out


class OpdMixin:
    """On-policy distillation: the student (the policy without a hint) plays sb26 level ``l2_level``
    from the single-level starts; the teacher is the CURRENT policy with a hint in its system message (an
    iterated teacher): the shortest clear the policy has produced so far, as its action list and its own reasoning
    before the clear (``arena.restore.clear_hint``). The loss per sampled token is the reverse-KL advantage log
    q_teacher - log p_student (``rl1_trainer``, ``--loss opd``). Metric: held-out starts without a hint."""

    def shortest_clear(self, folder, rows, level, skip=()):
        """``(tokens, attempt folder)`` of the cheapest clear of ``level`` among ``rows`` of a bench folder."""
        from .restore import clear_hint

        best = None
        for row in rows:
            attempt = Path(folder) / "attempts" / row["label"]
            if row["label"] in skip or not (attempt / "rollout.jsonl").is_file():
                continue
            try:
                _, info = clear_hint(attempt, level)
            except Exception:  # an unreadable record is not a candidate
                continue
            if info and info["tokens"] and (best is None or info["tokens"] < best[0]):
                best = (info["tokens"], attempt)
        return best

    async def opd_gate(self):
        """Teacher gate: the held-out starts, ``opd_gate_tries`` tries each per arm on the base, the same seeds:
        no hint; the shortest recorded clear of the level as the hint; the shortest recorded clear of another level
        (mismatched). Clear rate, median tokens to clear and how often the reasoning cites the hint, per arm."""
        from .restore import clear_hint

        g, cap, level = self.cfg["train_game"], int(self.cfg["l2_cap"]), int(self.cfg["l2_level"])
        held, _, rows = self.l2_start_pool()
        source = Path(self.cfg["lvl_resume"]["folder"]) / "games-00"
        own = self.shortest_clear(source, rows, level)
        others = [(c, other) for other in self.cfg["opd_gate_other_levels"]
                  for c in [self.shortest_clear(source, rows, int(other))] if c]
        pick = min(others, key=lambda item: item[0][0]) if others else None
        arms = {"none": (None, None)}
        if own:
            arms["hint"] = clear_hint(own[1], level)
        if pick:
            arms["mismatched"] = clear_hint(pick[0][1], int(pick[1]))
        self.feed.emit("opd_gate_hints", **{name: {"info": info, "hint": hint} for name, (hint, info) in arms.items()})
        tries, seed = int(self.cfg["opd_gate_tries"]), int(self.cfg["opd_gate_seed"])
        jobs = []
        for name, (hint, _) in arms.items():
            path, points = self.restore_points(self.l2_specs(held, "H", cap), f"gate-{name}", hint=hint)
            jobs.append(self.bench(f"gate-{name}", [g], tries, {g: cap}, points=path, model=self.server.model,
                                   seed_base=seed, concurrency=len(points) * tries))
        results = await asyncio.gather(*jobs)
        out = {"tries_per_start": tries, "cap": cap, "seed_base": seed, "starts": len(held)}
        for (name, (hint, info)), (folder, rows_arm, play) in zip(arms.items(), results):
            groups = self.l2_groups(rows_arm)
            flat = [r for grp in groups.values() for r in grp]
            out[name] = {**held_metrics(flat), "play": play, "hint": info,
                         "per_start": {p.split("-H-", 1)[-1]: held_metrics(grp) for p, grp in sorted(groups.items())},
                         "citations": hint_citations(folder, flat) if hint else None}
        base = (out.get("none") or {}).get("median_to_clear")
        for name in ("hint", "mismatched"):
            if name in out and base and out[name].get("median_to_clear"):
                out[name]["token_ratio"] = round(base / out[name]["median_to_clear"], 3)
        self.feed.emit("opd_gate", **out)
        return out

    async def opd_rounds(self, start, probes):
        """As ``l2_rounds`` (train and held-out benches each round, the update one round behind), but every
        restored train try is trained by OPD against the hinted current policy; the hint is refreshed to the
        cheapest clear so far (the recorded base games', then every round's train tries')."""
        from .restore import clear_hint

        g, cap, level = self.cfg["train_game"], int(self.cfg["l2_cap"]), int(self.cfg["l2_level"])
        held, train, rows0 = self.l2_start_pool()
        if not train or not held:
            raise RuntimeError(f"level {level}: {len(train)} train and {len(held)} held-out starts")
        self.l2_held_starts = held
        self.feed.emit("l2_starts", held=[f"{a.name[:4]}-k{q:03d}" for a, q in held],
                       train=[f"{a.name[:4]}-k{q:03d}" for a, q in train], cap=cap)
        best = self.shortest_clear(Path(self.cfg["lvl_resume"]["folder"]) / "games-00", rows0, level)
        if best is None:
            raise RuntimeError(f"no recorded clear of level {level} for the first hint")
        if self.serving_adapter is None:
            self.serving_adapter = self.adapters / "r00"
        kept, stop, played_by, k = None, None, start - 1, int(self.cfg["l2_starts"])
        self.l2_base = None
        for number in range(start, self.cfg["rounds"] + 1):
            began = asyncio.get_event_loop().time()
            hint, hint_info = clear_hint(best[1], level)
            picks = [train[((number - 1) * k + i) % len(train)] for i in range(min(k, len(train)))]
            tpath, tpoints = self.restore_points(self.l2_specs(picks, "T", cap), f"l2-train-{number:02d}")
            hpath, hpoints = self.restore_points(self.l2_specs(held, "H", cap), f"l2-held-{number:02d}")
            tries, etries = int(self.cfg["l2_tries"]), int(self.cfg["l2_eval_tries"])
            jobs = [self.bench(f"l2-train-{number:02d}", [g], tries, {g: cap}, points=tpath,
                               concurrency=len(tpoints) * tries),
                    self.bench(f"l2-held-{number:02d}", [g], etries, {g: cap}, points=hpath,
                               concurrency=len(hpoints) * etries, seed_base=int(self.cfg["l2_eval_seed"]))]
            if kept:
                jobs.append(self.opd_update(number, kept, probes, hint))
            results = await asyncio.gather(*jobs)
            (tfolder, trows, tplay), (hfolder, hrows, hplay) = results[0], results[1]
            row = {"round": number, "played_by": played_by, "hint": hint_info if kept else None}
            if kept:
                update_row, serve = results[2]
                row["update"] = {k2: v for k2, v in update_row.items() if k2 not in ("traceback", "windows")}
                if not update_row.get("ok"):
                    stop = row["stop"] = update_row.get("stop") or "update_failed"
                    self.feed.emit("round", **row)
                    break
                row["swap"] = {"load_seconds": await self.server.load("policy", serve)}
                self.serving_adapter = serve
                played_by = number
            hgroups = self.l2_groups(hrows)
            metric = held_metrics([r for grp in hgroups.values() for r in grp])
            metric["per_start"] = {p.split("-H-", 1)[-1]: held_metrics(grp) for p, grp in sorted(hgroups.items())}
            if self.l2_base is None:
                self.l2_base = metric
            row["held"] = metric
            groups = self.l2_groups(trows)
            flat = [r for grp in groups.values() for r in grp]
            tokens = [int(r.get("completion_tokens") or 0) for r in flat]
            row["train"] = {"play": tplay, "held_play": hplay, **held_metrics(flat), "trained": len(flat),
                            "mean_try_tokens": round(statistics.mean(tokens)) if tokens else None}
            found = self.shortest_clear(tfolder, flat, level)
            if found and found[0] < best[0]:
                best = found
            row["best_clear_tokens"] = best[0]
            row["round_seconds"] = round(asyncio.get_event_loop().time() - began, 1)
            kept = (Path(tfolder), flat) if flat else None
            self.feed.emit("round", **row)
            self.history.append(row)
            if not flat:
                stop = "restore"
                break
            if self.left() < 3600 * float(self.cfg["l2_final_hours"]):
                stop = "wall"
                break
        return stop

    async def opd_update(self, number, kept, probes, hint):
        """One optimizer step per start over its tries (shared prefixes), every try weight 1 and per-try
        normalised; the teacher's hint is the round's."""
        folder, rows = kept
        specs = [{**s, "w": 1.0, "whole": True, "window": int(self.cfg["lvl_window"]), "floor": 1024, "margin": 1024,
                  "credited_only": True, "share": r.get("point") or r["label"]}
                 for s, r in zip(self.windows(folder, rows, prefix="t-"), rows)]
        groups = {}
        for spec in specs:
            groups.setdefault(spec["share"], []).append(spec)
        keys = sorted(groups)
        random.Random(number).shuffle(keys)
        merged, path = await self.dosed_update(number, None, None, None, probes, batches=[groups[key] for key in keys],
                                               probe_last=True, max_windows=100000,
                                               cmd_extra={"share_prefix": True,
                                                          "prefix_chunk": int(self.cfg["path_chunk"]),
                                                          "norm": "try",
                                                          "opd": {"hint": hint, "clip": float(self.cfg["opd_clip"])}})
        parts = [w.get("opd") or {} for w in merged.get("windows", [])]
        counts = [p.get("tokens") or 0 for p in parts]
        if sum(counts):
            merged["opd"] = {"reverse_kl": round(sum((p.get("reverse_kl") or 0) * c for p, c in zip(parts, counts))
                                                 / sum(counts), 5),
                             "up_share": round(sum((p.get("up_share") or 0) * c for p, c in zip(parts, counts))
                                               / sum(counts), 4),
                             "teacher_seconds": round(sum(float((w.get("teacher") or {}).get("seconds") or 0)
                                                          for w in merged.get("windows", [])), 1),
                             "tokens": sum(counts)}
        merged["trained_tries"] = len(specs)
        if merged.get("ok"):
            self.checkpoint(number)
            await self.save_checkpoint()
        return merged, path


class MultiMixin:
    """Multi-game baseline. ``mg`` spec: ``cap`` (tokens per try), ``sets``
    [{name, game, level, role train|test, fresh | sources [folders | "record:<game>"], starts, tries, seed,
    max_prompt, untrained_game, round_tries}], ``record`` [{game, games, budget, seed}].

    Mode ``screen`` (one card, base only): the record benches, then every set's tries; writes ``screen.json`` (per
    set: its points file, bench folder, per-start tries and clears, metrics). Mode ``loop`` with ``mg.screen`` (two
    cards): l4's method on the screen's mixed train starts, dynamic sampling (groups whose tries all cleared or all
    failed are not trained), the update one round behind; each played round also plays every test set (same
    starts and seeds as the screen's base)."""

    def mg_point(self, game):
        from .level_bench import load_points

        return load_points(self.source / self.study["level_bench"]["start_points"],
                           {**self.study["level_bench"], "games": [game], "split": "all"})[0]

    def mg_starts(self, spec, sources):
        """Restored starts of ``spec``'s level in ``sources``' usable base games of its game (before any compaction
        or python time-out, prompt <= max_prompt), shuffled by pick_seed, the first ``starts``."""
        from .rl_loop import usable

        pool = []
        for src in sources:
            root = Path(src)
            rows = [r for r in json.loads((root / "level-bench.json").read_text())["attempts"] if usable(r)]
            for row in sorted(self.game_rows(rows, spec["game"]), key=lambda r: r["label"]):
                if spec.get("start_half") is not None and start_half(row["label"]) != int(spec["start_half"]):
                    continue  # 19 Oct: recordings split in two halves (held-out reads vs the training pool)
                attempt = root / "attempts" / row["label"]
                request = self.mg_attempt_start(spec, attempt)
                if request is not None:
                    pool.append((str(attempt), request))
        random.Random(int(self.cfg["mg"].get("pick_seed", 0)) * 100 + int(spec["level"]) * 7 + len(spec["game"])).shuffle(pool)
        return pool[: int(spec["starts"])], len(pool)

    def mg_screen_starts(self, screen, sets, workers=0):
        """mg.seed_starts "screen" (explorer design): every level >= 2 the screen's recordings reached, as start-pool
        rows (source "seed", played by the base: version 0) for the training set of that game and level. ``workers``
        > 0 (ARC_RL1_SEED_SCAN, RL start-up 23 Oct: 110 s of a hero leg's loop setup, every rollout parsed twice per
        level): each recording is scanned once (restore.attempt_scan) by that many processes first; same rows."""
        from .rl_loop import usable

        names = {(str(s["game"])[:4], int(s.get("level") or 1)): n for n, s in sets.items()
                 if s["role"] == "train" and not s.get("fresh")}
        scans = self.mg_seed_scans(screen, names, workers) if workers else None
        out = []
        for game, folder in sorted((screen.get("recorded") or {}).items()):
            root = Path(folder)
            try:
                rows = [r for r in json.loads((root / "level-bench.json").read_text())["attempts"] if usable(r)]
            except (OSError, ValueError, KeyError):
                self.feed.emit("mg_seed_starts_missing", game=game, folder=str(root))
                continue
            for row in sorted(rows, key=lambda r: r["label"]):
                attempt = root / "attempts" / row["label"]
                for level in range(2, int(row.get("levels") or 0) + 2):
                    name = names.get((str(game)[:4], level))
                    if name is None:
                        continue
                    request = self.mg_attempt_start(sets[name], attempt,
                                                    scan=None if scans is None else scans[str(attempt)])
                    if request is None:
                        continue
                    out.append({"id": f"s-{row['label']}-L{level}", "round": 0, "set": None, "target": name,
                                "game": sets[name]["game"], "level": level, "attempt": str(attempt),
                                "request": int(request), "prompt_tokens": None, "played_by": 0, "parent": str(attempt),
                                "parent_level": level - 1, "clear_tokens": None, "source": "seed",
                                "family": sets[name].get("family") or sets[name]["game"],
                                "start": {"attempt": str(attempt), "request": int(request)}})
        return out

    def mg_seed_scans(self, screen, names, workers):
        """``{attempt: restore.attempt_scan(attempt)}`` for every usable recording ``mg_screen_starts`` would read
        (a level >= 2 with a training set), ``workers`` spawned processes (JSON parsing is CPU-bound)."""
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        from .restore import attempt_scan
        from .rl_loop import usable

        todo = []
        for game, folder in sorted((screen.get("recorded") or {}).items()):
            root = Path(folder)
            try:
                rows = [r for r in json.loads((root / "level-bench.json").read_text())["attempts"] if usable(r)]
            except (OSError, ValueError, KeyError):
                continue  # mg_screen_starts reports it
            for row in rows:
                if any((str(game)[:4], level) in names for level in range(2, int(row.get("levels") or 0) + 2)):
                    todo.append(str(root / "attempts" / row["label"]))
        began = time.time()
        with ProcessPoolExecutor(int(workers), mp_context=multiprocessing.get_context("spawn")) as pool:
            scans = dict(zip(todo, pool.map(attempt_scan, todo, chunksize=4)))
        self.feed.emit("mg_seed_scan", attempts=len(todo), workers=int(workers), seconds=round(time.time() - began, 1))
        return scans

    @staticmethod
    def mg_attempt_start(spec, attempt, scan=None):
        """The request of ``spec``'s level start in one recorded attempt, or None (never reached, after an unsafe
        step: summary, handoff, interrupt, forced drain; or prompt above max_prompt). ``scan``: the attempt's
        restore.attempt_scan, read once for every level."""
        from .restore import level_starts, unsafe_from

        if scan is not None:
            first, prompts, starts = scan
        else:
            first, _ = unsafe_from(attempt)
            with open(Path(attempt) / "rollout.jsonl") as file:
                prompts = [int((json.loads(line).get("usage") or {}).get("prompt_tokens") or 0) for line in file]
            starts = level_starts(attempt)
        for request, done, step in starts:
            if (done + 1 == int(spec["level"]) and (first is None or step < first)
                    and prompts[int(request)] <= int(spec.get("max_prompt") or 10**9)):
                return int(request)
        return None

    def mg_live_start(self, spec, attempt, snap):
        """A level start already played by a recording still running (restored-start sets begin as soon
        as their starts exist, not after the last recording ends): the attempt's rollout and log, cut at their last
        complete line, copied to ``snap``; ``(snap, request)`` when the start and its observation are in the copy."""
        from .restore import load, log_rows

        attempt, level = Path(attempt), int(spec["level"])
        try:
            if not any(int(e["level"]) == level - 1 for e in log_rows(attempt / "log.jsonl", ("level_bench_level",))):
                return None
            snap.mkdir(parents=True, exist_ok=True)
            for name in ("rollout.jsonl", "log.jsonl"):
                data = (attempt / name).read_bytes()
                (snap / name).write_bytes(data[: data.rfind(b"\n") + 1])
            request = self.mg_attempt_start(spec, snap)
            if request is None:
                return None
            load(snap, request)  # the observation and every row up to the start are complete
            return snap, request
        except (OSError, ValueError, KeyError, IndexError):  # a file mid-write: the next poll
            return None

    @staticmethod
    def mg_tag(spec, attempt, request):
        import hashlib

        path = str(attempt)
        if path.startswith("/tmp/validation/"):  # a packed preflight leg's unpacked copy: the volume path's tag, so a
            path = "/runs" + path[4:]  # resumed leg's kept tries on its starts match their plan digests (pfB2, 22 Oct)
        return f"L{spec['level']}-{hashlib.md5(path.encode()).hexdigest()[:6]}-k{int(request):03d}"

    def mg_points(self, spec, starts, name, harvest=False):
        """The points file of one set: a fresh level-1 segment of its game, or its restored starts. ``harvest``: a
        cleared try records the next level's first request (arena.harvest)."""
        from .restore import load

        cap, base = int(spec.get("cap") or self.cfg["mg"]["cap"]), self.mg_point(spec["game"])
        if spec.get("fresh"):
            points = [{**base, "id": f"{base['id']}-L1-{spec['name']}", "through": False, "segment": True, "level": 1,
                       "level_index": 0, "prefix": [], "budget": cap}]
        else:
            points = []
            for attempt, request in starts:
                tag = self.mg_tag(spec, attempt, request)
                points.append({**base, "id": f"{base['id']}-{tag}", "level": load(attempt, request)["level"],
                               "restore": {"attempt": attempt, "request": int(request), "check_only": False, "budget": cap,
                                           **({"pin_actions": True} if spec.get("pin_actions") else {})}})
        if harvest:
            points = [{**p, "harvest": True} for p in points]
        path = self.work / f"{name}-points.json"
        path.write_text(json.dumps({"schema": 1, "points": points}))
        return path, points

    async def mg_play(self, name, spec, starts, tries, *, model, seed, concurrency=None, sampling=None, harvest=False,
                      own=None):
        """``sampling``: mg.train_sampling on the training sets only (pilot 2 temp arm); reads keep the study's.
        ``harvest``: training sets whose clears cut the next level's starts (never held-out reads). ``own``
        (mg.lookahead_below): the bench's own keep/reuse roots and server-1-only flag (level_bench.LevelBench)."""
        path, points = self.mg_points(spec, starts, name, harvest=harvest)
        if re.match(r"r\d\d-", name):  # live per-set play stats (mg_progress_row): the bench's planned tries
            live = self.__dict__.setdefault("mg_live_sets", {})
            live.setdefault(name, self.mg_live_zero())["planned"] = len(points) * int(tries)
        extra = {"tail": self.mg_tail(name, spec)} if self.cfg["mg"].get("tail_fill") and re.match(r"r\d\d-", name) else {}
        users = None
        if self.mg_versioned():  # every try row records its adapter; a name in use is never unloaded
            own = {"policy_version": self.mg_play_version(model), **(own or {})}
            users = self.__dict__.setdefault("mg_users", {})
            users[model] = users.get(model, 0) + 1
            extra["own"] = own
        try:
            folder, rows, play = await self.bench(name, [spec["game"]], int(tries),
                                                  {spec["game"]: int(spec.get("cap") or self.cfg["mg"]["cap"])},
                                                  points=path, model=model, seed_base=seed,
                                                  concurrency=concurrency or len(points) * int(tries), reserve=120,
                                                  sampling=sampling, **extra)
        finally:
            if users is not None:
                users[model] -= 1
                if not users[model]:
                    try:
                        await self.mg_gc_names()
                    except Exception as error:  # noqa: BLE001 - a name left loaded costs a LoRA slot, not the run
                        self.feed.emit("alert", why="adapter unload failed", name=model, error=repr(error)[:300])
        rows = [r for r in self.game_rows(rows, spec["game"])
                if restored(r) and level_of(r) is not None]
        gated = [r for r in rows if not self.mg_digest_ok(r)]
        if gated:
            rows = [r for r in rows if self.mg_digest_ok(r)]
            self.feed.emit("digest_gate", bench=name, dropped=len(gated), labels=[r.get("label") for r in gated][:20])
        if "tail" in extra:  # tail_fill history in memory: round packs move the folders away as the next round starts
            by = {}
            for r in rows:
                by.setdefault(r.get("point"), []).append(int(r.get("completion_tokens") or 0))
            self.__dict__.setdefault("mg_tail_hist", {})[name[3:]] = {p: statistics.median(v) for p, v in by.items() if p}
        return folder, rows, play

    @staticmethod
    def mg_live_zero():
        return {"tries": 0, "clears": 0, "tok_sum": 0, "act_sum": 0, "rhae_sum": 0.0, "planned": 0}

    def served_persist_start(self):
        """(c) served-route persistence: route records lived only in the playing container's
        /tmp, so every resume's first update (and its reused tries) trained on the trainer's own routing (r2 U12,
        r3/r4 U13: 0 served windows; k3/clip ~10x). Finished records are mirrored to <run>/served-route on the volume
        every minute; a resume copies its roots' mirrors into the live capture directory before its first update."""
        out = os.environ.get("ARC_VALIDATION_OUTPUT")
        if not out or not self.served_dir or getattr(self, "served_mirror", None) is not None:
            return
        from .served_route import restore

        self.served_mirror = Path(out) / "served-route"
        roots = [Path(r) / "served-route" for r in os.environ.get("ARC_RL1_RESUME", "").split(",") if r]

        async def restoring():
            began = time.time()
            try:
                got = await asyncio.to_thread(restore, roots, self.served_dir)
                self.feed.emit("served_restore", seconds=round(time.time() - began, 1), **got)
            except Exception as exc:  # noqa: BLE001 - a failed restore costs only the routes (the round row shows it)
                self.feed.emit("served_restore", error=repr(exc)[:300])

        self.served_restore_task = asyncio.ensure_future(restoring())
        self.served_mirror_task = asyncio.ensure_future(self.served_mirror_loop())

    async def served_mirror_loop(self, every=60.0):
        from .served_route import mirror

        seen = {}
        self.served_mirror_stats = {"files": 0, "bytes": 0, "passes": 0}
        while True:
            await asyncio.sleep(every)
            try:
                got = await asyncio.to_thread(mirror, self.served_dir, self.served_mirror, seen)
            except Exception as exc:  # noqa: BLE001 - the next pass retries
                self.feed.emit("served_mirror_failed", error=repr(exc)[:300])
                continue
            st = self.served_mirror_stats
            st.update(files=st["files"] + got["files"], bytes=st["bytes"] + got["bytes"], passes=st["passes"] + 1)

    async def served_restored(self):
        """The resume's served-route restore, finished before an update reads routes."""
        task = getattr(self, "served_restore_task", None)
        if task is not None and not task.done():
            began = time.time()
            await asyncio.shield(task)
            self.feed.emit("served_restore_wait", seconds=round(time.time() - began, 1))

    async def mg_loop_lag(self, every=1.0, alert=5.0):
        """Event-loop lag (failure class 1, 22 Oct: r4's synchronous kept-try copy held all play ~7 min): the overshoot
        of a 1-s sleep. Its max since the last play_progress row and over the round ride on those rows; an overshoot
        over ``alert`` s is a ``loop_lag`` row at once."""
        loop = asyncio.get_event_loop()
        while True:
            began = loop.time()
            await asyncio.sleep(every)
            lag = loop.time() - began - every
            self.loop_lag["tick"], self.loop_lag["round"] = max(self.loop_lag["tick"], lag), max(self.loop_lag["round"], lag)
            if lag > alert:
                self.feed.emit("loop_lag", seconds=round(lag, 1))

    def mg_progress_row(self, bench, row):
        """level_bench.PROGRESS_HOOK: one finished try into its round bench's live aggregates (tries,
        clears, generated tokens, actions on clears, RHAE)."""
        if not re.match(r"r\d\d-", str(bench)):
            return
        s = self.__dict__.setdefault("mg_live_sets", {}).setdefault(bench, self.mg_live_zero())
        s["tries"] += 1
        s["tok_sum"] += int(row.get("completion_tokens") or 0)
        if clear_tokens(row):
            s["clears"] += 1
            s["act_sum"] += int((row.get("level_rows") or [{}])[0].get("actions") or 0)
            s["rhae_sum"] = round(s["rhae_sum"] + (clear_rhae(row) or 0.0), 4)

    def mg_live_round(self, number):
        """This round's per-set aggregates for its play_progress row ({set: {...}}, set = bench name less rNN-)."""
        live = getattr(self, "mg_live_sets", None) or {}
        return {b[4:]: dict(v) for b, v in sorted(live.items()) if b.startswith(f"r{number:02d}-")}

    def mg_tail(self, name, spec):
        """mg.tail_fill for round bench ``name`` (rNN-...): held-out (test sets, fixed starts) take slots last; the
        expected tokens per start are the median completion tokens of the same bench's usable tries last round."""
        from .rl_loop import usable

        expected = (getattr(self, "mg_tail_hist", None) or {}).get(name[3:])
        if expected is None:  # first round of this process: the previous round's bench file, when still there
            prev = self.output / "rl" / f"r{int(name[1:3]) - 1:02d}{name[3:]}" / "level-bench.json"
            try:
                rows = [r for r in json.loads(prev.read_text())["attempts"] if usable(r)]
            except (OSError, ValueError, KeyError):
                rows = []
            by = {}
            for r in rows:
                by.setdefault(r["point"], []).append(int(r.get("completion_tokens") or 0))
            expected = {p: statistics.median(v) for p, v in by.items()}
        held = spec.get("role") in ("test", "base_read") or name[3:].startswith("-fixed-")
        if self.cfg["mg"].get("heldout_first"):  # held-out reads take slots first (the main read lands
            # soon after the update; sim on R10-R12 durations: round end same or shorter, the long unseen reads too)
            return {"expected": expected, "last": False, "first": held}
        return {"expected": expected, "last": held}

    @staticmethod
    def mg_groups(rows, size=4):
        """Tries grouped by start; a fresh set's tries (one start) in groups of ``size`` by pass."""
        groups = {}
        for r in rows:
            p = int(r.get("pass") or r["label"].rsplit("-p", 1)[-1])
            key = r["point"] if r.get("restore") else f"{r['point']}#g{p // size}"
            groups.setdefault(key, []).append(r)
        return groups

    def mg_select(self, folder, rows, name=None):
        if (self.cfg["mg"].get("turn_credit") and rows
                and os.environ.get("ARC_TURN_POOL", "1") == "1"):  # every try parsed in parallel, once
            turn_prefetch([Path(folder) / "attempts" / r["label"] for r in rows], self.cfg["mg"].get("reply_filter") or {})
        return self.mg_select_rows(folder, rows, name)

    @staticmethod
    def mg_preflight_spread(rewards, info):
        """CPU preflight only (stub spec "spread"): replayed tries repeat their recordings, so a group's rewards tie and
        nothing trains; a tied group gets a fixed rising ladder so the update, save and serve paths run."""
        if len(rewards) < 2 or max(rewards) != min(rewards):
            return rewards
        try:
            on = bool(json.loads(os.environ.get("ARC_PREFLIGHT_STUB") or "{}").get("spread"))
        except ValueError:
            on = False
        if not on:
            return rewards
        info["preflight_spread"] = info.get("preflight_spread", 0) + 1
        top = max(float(max(rewards)), 0.0) + 1.0
        return [top * (1 + 0.1 * i) for i in range(len(rewards))]

    def mg_select_rows(self, folder, rows, name=None):
        """Dynamic sampling and l4's selection: per mixed group (some but not all tries cleared) the per_side largest
        positive and negative advantages (reward l2_reward, or mg.reward_shape). Returns ([(folder, row, weight,
        share)], info); with mg.turn_credit each picked try's per-reply extra credit goes to self.mg_turn."""
        mg = self.cfg["mg"]
        out, info, side = [], {"groups": 0, "mixed": 0, "trained": 0}, int(self.cfg["l2_per_side"])
        for key, grp in sorted(self.mg_groups(rows, int(mg.get("group_size") or 4)).items()):
            info["groups"] += 1
            if mg.get("reward_shape") == "logeq":
                if mg.get("ref_fallback") == "v0_group" or mg.get("reward_ref") == "group":
                    self.mg_group_ref = {"name": name, "clears": [(clear_tokens(r), int(r["level_rows"][0].get("actions") or 0))
                                                                  for r in grp if clear_tokens(r)
                                                                  and int(r["level_rows"][0].get("actions") or 0) > 0]}
                try:
                    rewards = self.mg_preflight_spread([self.mg_reward(r, name) for r in grp], info)
                finally:
                    self.mg_group_ref = None
                if len(grp) < 2 or max(rewards) == min(rewards):  # all fails, or identical
                    info["zero_spread"] = info.get("zero_spread", 0) + 1
                    continue
                total = sum(rewards)
                adv = [r - (total - r) / (len(grp) - 1) for r in rewards]
                info["mixed"] += int(any(not clear_tokens(r) for r in grp))
                info.setdefault("_shadow", []).append((grp, adv))
            elif mg.get("reward_shape") in ("rhae_free", "floor", "aware"):
                rewards = [self.mg_reward(r, name) for r in grp]
                if mg.get("reward_log"):  # heavy tail (one short few-action clear ~10x the median): compress, order kept
                    rewards = [math.log1p(r) for r in rewards]
                rewards = self.mg_preflight_spread(rewards, info)
                if len(grp) < 2 or max(rewards) <= 0 or max(rewards) == min(rewards):  # zero variance: no signal
                    info["zero_spread"] = info.get("zero_spread", 0) + 1
                    continue
                total = sum(rewards)
                adv = [r - (total - r) / (len(grp) - 1) for r in rewards]
                info["mixed"] += int(min(rewards) <= 0)
            else:
                clears = sum(1 for r in grp if clear_tokens(r))
                # mg5: all-clear groups carry the shortening signal (l4 learned from them)
                if not 0 < clears < len(grp) and not (mg.get("select") == "all" and clears):
                    continue
                info["mixed"] += int(clears < len(grp))
                rewards, adv = group_advantages(grp, self.cfg["l2_reward"])
            info.setdefault("_rewards", []).extend(float(x) for x in rewards)
            order = sorted(range(len(grp)), key=lambda i: -adv[i])
            pick = [i for i in order[:side] if adv[i] > 0] + [i for i in order[::-1][:side] if adv[i] < 0]
            out += [(Path(folder), grp[i], adv[i], key) for i in pick]
            if mg.get("turn_credit"):
                extra = self.mg_turn_credit(Path(folder), grp, rewards, name)
                self.mg_turn = getattr(self, "mg_turn", {})
                for i in pick:
                    self.mg_turn[(str(Path(folder)), grp[i]["label"])] = extra[i]
        info["trained"] = len(out)
        shadow = info.pop("_shadow", None)
        if shadow is not None or mg.get("reward_shape") == "logeq":  # keyed by the played set (a branch set's
            # name is its base set's here)
            key = re.sub(r"^r\d+-", "", Path(folder).name) if folder else name
            info["shadow"] = self.mg_shadow_ratio(key, rows, shadow or [])
        got = info.pop("_rewards", [])
        if got:  # dashboard: reward spread of the trained groups
            info["reward_mean"] = round(statistics.mean(got), 4)
            info["reward_std"] = round(statistics.pstdev(got), 4)
        toks = sorted(int(r.get("completion_tokens") or 0) for r in rows)
        if toks:
            info["try_tokens_p50"], info["try_tokens_p90"] = toks[len(toks) // 2], toks[min(len(toks) - 1, int(0.9 * len(toks)))]
        return out, info

    def mg_shadow_ratio(self, name, rows, groups):
        """Shadow metric: the ratio reward r = R - rho t (R the level RHAE, 0 for a
        fail; t the level tokens; rho the set's previous pooled RHAE per token, else this one's) beside logeq. Per
        call: pooled RHAE/10K, the ratio reward's mean, and over the trained groups the mean Spearman rank correlation
        of the two LOO advantages and the share of tries whose advantage sign differs."""
        def R(r):
            return (clear_rhae(r) or 0.0) if clear_tokens(r) else 0.0

        def T(r):
            return clear_tokens(r) or int(r.get("completion_tokens") or 0)

        tot_t = sum(T(r) for r in rows)
        pooled = sum(R(r) for r in rows) / tot_t if tot_t else None
        store = self.__dict__.setdefault("mg_shadow_rho", {})
        rho = store.get(name, pooled)
        if pooled is not None:
            store[name] = pooled
        out = {"pooled_10k": round(1e4 * pooled, 4) if pooled is not None else None,
               "rho_10k": round(1e4 * rho, 4) if rho is not None else None, "groups": len(groups)}
        if rho is None or not rows:
            return out
        out["ratio_mean"] = round(statistics.mean(R(r) - rho * T(r) for r in rows), 4)

        def ranks(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            rk, i = [0.0] * len(v), 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                for k in range(i, j + 1):
                    rk[order[k]] = (i + j) / 2
                i = j + 1
            return rk

        cors, flips, signs = [], 0, 0
        for grp, adv in groups:
            ratio = [R(r) - rho * T(r) for r in grp]
            tot = sum(ratio)
            adv2 = [x - (tot - x) / (len(grp) - 1) for x in ratio]
            a, b = ranks(adv), ranks(adv2)
            if len(set(a)) > 1 and len(set(b)) > 1:
                cors.append(statistics.correlation(a, b))
            for u, v in zip(adv, adv2):
                if abs(u) > 1e-12 and abs(v) > 1e-12:
                    signs += 1
                    flips += (u > 0) != (v > 0)
        out["spearman"] = round(statistics.mean(cors), 4) if cors else None
        out["flip_share"] = round(flips / signs, 4) if signs else None
        return out

    def mg_reply_balance(self, specs, turn):
        """mg6: each reply's weight = its try's advantage + its per-turn extra; then push-down mass (|weight| x reply
        tokens / the level's norm constant) is scaled to equal push-up mass at the reply level, so extras and the
        balance agree. A spec keeps its try advantage (scaled when negative) as ``w`` for the trainer's sign filter."""
        up = down = 0.0
        mass, cap = {}, float(self.cfg["mg"]["cap"])
        audit = {"replies": 0, "kinds": {}, "tokens": 0, "think_tokens": 0, "answer_tokens": 0, "positive_tokens": 0}
        # the try advantage stays the main signal: extras scaled so their median |.| is at most ``extra_share`` x the
        # median |try advantage| of this update (mock on mg5g tries: raw extras ~1-3x the try advantages)
        ws = [abs(sp["w"]) for sp in specs if sp["w"]]
        ex = [abs(e) for sp in specs for e in (turn.get(tuple(sp.get("turn_key") or ())) or {}).get("extra", {}).values() if e]
        share = float(self.cfg["mg"]["turn_credit"].get("extra_share", 0.5))
        k = min(1.0, share * statistics.median(ws) / statistics.median(ex)) if ws and ex else 1.0
        gigpo = self.cfg["mg"]["turn_credit"].get("mode") == "gigpo"
        if gigpo:  # A = A_ep + omega x A_step as computed (omega applied in mg_gigpo_steps): no rescale
            k = 1.0
        for sp in specs:
            info = turn.get(tuple(sp.get("turn_key") or ())) or {}
            norm = float(sp.get("norm_cap") or cap)
            sp["reply_w"] = {str(n): sp["w"] + k * info.get("extra", {}).get(n, 0.0) for n in info.get("tokens", {})}
            for n, t in info.get("tokens", {}).items():  # the filter's audit: positive reply tokens before it
                if sp["reply_w"][str(n)] > 0:
                    audit["positive_tokens"] += t
            for n, f in (info.get("filtered") or {}).items():
                if sp["reply_w"].get(str(n), 0.0) > 0:
                    sp["reply_w"][str(n)] = 0.0
                    audit["replies"] += 1
                    audit["kinds"][f["kind"]] = audit["kinds"].get(f["kind"], 0) + 1
                    audit["tokens"] += int(info["tokens"].get(n) or f["tokens"])
                    audit["think_tokens"] += int(f["think"])
                    audit["answer_tokens"] += max(0, int(f["tokens"]) - int(f["think"]))
            for n, t in info.get("tokens", {}).items():
                v = sp["reply_w"][str(n)] * t / norm
                up, down = (up + v, down) if v > 0 else (up, down - v)
                mass[sp.get("set")] = mass.get(sp.get("set"), 0.0) + abs(v)
        scale = up / down if down > 0 and up > 0 else 1.0
        flips = 0
        for sp in specs:
            for n, v in sp.get("reply_w", {}).items():
                if v < 0:
                    sp["reply_w"][n] = v * scale
                flips += int((v > 0) != (sp["w"] > 0))
            if sp["w"] < 0:
                sp["w"] *= scale
        total = sum(mass.values()) or 1.0
        if self.cfg["mg"].get("reply_filter"):
            audit["share"] = round(audit["tokens"] / audit["positive_tokens"], 5) if audit["positive_tokens"] else None
            self.feed.emit("reply_filter", **audit)
        return {"up_mass": round(up, 3), "down_mass": round(down, 3), "down_scale": round(scale, 4), "reply_level": True,
                **({"reply_filter": audit} if self.cfg["mg"].get("reply_filter") else {}),
                "extra_scale": round(k, 4),
                "sign_flips": flips, "w_abs_median": round(statistics.median(ws), 3) if ws else None,
                "extra_abs_median": round(k * statistics.median(ex), 3) if ex else None,
                "anchor_size_median": statistics.median(self.mg_anchor_sizes) if getattr(self, "mg_anchor_sizes", None) else None,
                **({"gigpo": self.mg_gigpo_summary()} if gigpo else {}),
                "level_share": {str(k): round(v / total, 3) for k, v in sorted(mass.items(), key=lambda kv: str(kv[0]))}}

    def mg_ref_fallback(self, name):
        """mg.ref_fallback "v0": the run lineage's version-0 (base) clears of the unit (median tokens to clear and
        actions, >= 2 clears); "v0_group": else the current group's own clears (mg_select_rows sets mg_group_ref)."""
        mode = self.cfg["mg"].get("ref_fallback")
        if mode not in ("v0", "v0_group"):
            raise ValueError(f"mg.ref_fallback {mode!r}: v0 | v0_group")
        got = list(((getattr(self, "mg_v0_refs", None) or {}).get(name) or {}).values())
        if len(got) < 2 and mode == "v0_group":
            g = getattr(self, "mg_group_ref", None) or {}
            got = g.get("clears") or [] if g.get("name") == name else []
        if not got:
            return None
        return {"median_to_clear": statistics.median(t for t, _ in got), "median_actions": statistics.median(a for _, a in got)}

    def mg_v0_add(self, name, rows):
        """Version-0 clears of ``name`` for mg.ref_fallback (rows: level-bench attempt rows; others are skipped)."""
        refs = self.__dict__.setdefault("mg_v0_refs", {})
        for r in rows:
            t = clear_tokens(r)
            if t and r.get("policy_version") is not None and int(r["policy_version"]) == 0:
                a = int((r.get("level_rows") or [{}])[0].get("actions") or 0)
                if a > 0:  # by label: a resumed group's kept tries come back in its rows
                    refs.setdefault(name, {})[r.get("label") or f"{len(refs.get(name) or {})}"] = (t, a)

    def mg_reward(self, row, name):
        """mg6 reward: clear x min(rhae_hi, (baseline / actions)^2) x (T_ref / tokens), T_ref the
        level's base median tokens; 0 for a fail or above the action backstop (action_backstop x the base median
        actions). RHAE uncapped below rhae_hi, so actions carry signal on every level."""
        mg = self.cfg["mg"]
        t = clear_tokens(row)
        if mg.get("reward_shape") == "logeq":
            return self.mg_logeq(row, name, t)
        if not t:
            return 0.0
        lr = row["level_rows"][0]
        actions, level = int(lr.get("actions") or 0), int(lr.get("level") or 0)
        game = str(row.get("point", ""))[:4]
        base = self.mg_level_ref(name, game, level)
        if actions <= 0 or (base["actions"] and actions > float(mg.get("action_backstop", 2.0)) * base["actions"]):
            return 0.0
        if mg.get("reward_shape") == "aware":
            # mg7b: no reward above the level's score-aware action cap, tokens alone below it
            if actions > self.mg_action_cap(base):
                return 0.0
            r = min(float(mg.get("token_hi", 3.0)), float(base["tokens"] or t) / t)
            a_ref = float(base["actions"] or 0)
            if mg.get("tight_action_pow") and a_ref and self.mg_action_cap(base) <= float(mg.get("action_floor", 1.05)) * a_ref + 1e-9:
                # a tight level (the cap is the base's, not the score's): RHAE moves with actions below it too
                r *= min(float(mg.get("tight_action_hi", 2.0)), (a_ref / actions) ** float(mg["tight_action_pow"]))
            return r
        if mg.get("reward_shape") == "floor":
            # mg7 (mg6 R1-R3 bought tokens with actions): no reward above action_floor x the base median actions;
            # the action term against the base median with exponent action_pow (> the token term's 1)
            a_ref = float(base["actions"] or actions)
            if actions > float(mg.get("action_floor", 1.15)) * a_ref:
                return 0.0
            act = min(float(mg.get("action_hi", 4.0)), (a_ref / actions) ** float(mg.get("action_pow", 3.0)))
            return act * min(float(mg.get("token_hi", 3.0)), float(base["tokens"] or t) / t)
        rhae = min(float(mg.get("rhae_hi", 4.0)), (base["baseline"] / actions) ** 2) if base["baseline"] else 1.0
        return rhae * min(float(mg.get("token_hi", 6.0)), float(base["tokens"] or t) / t)

    def mg_logeq(self, row, name, t):
        """sc25 reward: clip(log((R / R_b) x (T_b / t)), +-log clip), fail -log fail; R the try's level
        RHAE min(rhae_cap, (baseline / actions)^2), R_b the same at the base median actions, T_b the base median
        tokens to clear."""
        mg = self.cfg["mg"]
        lo, fail = math.log(float(mg.get("log_clip", 8.0))), -math.log(float(mg.get("log_fail", 16.0)))
        if not t:
            return fail
        lr = row["level_rows"][0]
        actions, level = int(lr.get("actions") or 0), int(lr.get("level") or 0)
        base = self.mg_level_ref(name, str(row.get("point", ""))[:4], level)
        cap, b = float(mg.get("rhae_cap", 1.15)), base.get("baseline")
        if actions <= 0:
            return fail
        rem = row.get("branch_rem")
        if rem:  # mg.branch.score remainder: the part after the cut against the cut group's medians
            return self.mg_log_terms(*branch_rem_terms(rem, t, b, cap), lo, fail)
        rhae = lambda a: min(cap, (b / a) ** 2) if b and a else 1.0
        g = getattr(self, "mg_group_ref", None) or {}
        if mg.get("reward_ref") == "group" and g.get("name") == name and g.get("clears"):
            # no absolute reference in the training reward; the clip centres on the group's own clear
            # medians (tokens, actions), refs stay for the metrics and the curriculum only
            base = {"tokens": statistics.median(x for x, _ in g["clears"]), "actions": statistics.median(a for _, a in g["clears"])}
        return self.mg_log_terms(math.log(rhae(actions) / rhae(float(base.get("actions") or actions))),
                                 math.log(float(base.get("tokens") or t) / t), lo, fail)

    def mg_log_terms(self, rhae_term, token_term, lo, fail):
        """The log-form reward from its RHAE (actions) term and its token term: the sum clipped at +-log clip, or with
        ``mg.log_rhae_floor`` (after pilot a's action spam: a clear with ruined RHAE sat at -log 8,
        above a fail's -log 16) the RHAE term floored at the fail value, the token term clipped at +-log clip, the sum
        kept in [fail, log clip]: a ruined-RHAE clear scores about a fail, never below it."""
        if self.cfg["mg"].get("log_rhae_floor"):
            return max(fail, min(lo, max(fail, rhae_term) + max(-lo, min(lo, token_term))))
        return max(-lo, min(lo, rhae_term + token_term))

    def mg_action_cap(self, base):
        """mg7b score-aware action cap: max(action_floor x base median actions, baseline / sqrt(rhae_cap)), the
        actions at which the level's RHAE (min(rhae_cap, (baseline / a)^2)) starts to fall."""
        mg = self.cfg["mg"]
        cap = float(mg.get("action_floor", 1.05)) * float(base["actions"] or 0)
        if base.get("baseline"):
            cap = max(cap, float(base["baseline"]) / math.sqrt(float(mg.get("rhae_cap", 1.15))))
        return cap if cap > 0 else float("inf")

    def mg_level_ref(self, name, game, level):
        """A set's base reference: human baseline actions (points file), base median tokens and actions (screen)."""
        cache = self.__dict__.setdefault("mg_refs", {})
        # mg7 (ref_round1): once round 1's base held-out tries exist, the base median actions pool them with the
        # screen's (count-weighted medians; tn36-L1: screen 21.5 vs round-1 27 on 16 and 7 clears)
        r1 = [r for r in ((getattr(self, "mg_fixed", None) or {}).get(1, {}).get(name) or [])
              if r.get("level_rows")] if name and self.cfg["mg"].get("ref_round1") else []
        m = ((getattr(self, "mg_screen", None) or {}).get("sets", {}).get(name, {}).get("metrics") or {}) if name else {}
        tref = (((self.cfg.get("mg_sets") or {}).get(name) or {}).get("ref") or {}) if name else {}
        # order (envs 24 Oct, mg_pool.apply_refs): screen > measured table ref > derived table ref > the fallback
        # per field (envs 25 Oct, mg-refs-v3): a derived ref with actions null keeps its tokens and takes the fallback's
        # actions (ka59-L4: derived 35.4 actions vs real 220.5 saturated the action term on every try)
        part = bool(tref.get("derived") and tref.get("actions") is None)
        fb = self.mg_ref_fallback(name) if name and not m and (part or not (tref.get("measured") or tref.get("derived"))) \
            and self.cfg["mg"].get("ref_fallback") else None
        if fb and part:
            fb = {k: v for k, v in fb.items() if k in ("median_actions", "clears")}
        if fb:  # mg.ref_fallback: a unit the screen never measured takes measured medians, not
            m = fb  # the env table's (324 of v6g's 1,089 sets carry its 64,000-token placeholder: clears clip at +log 8)
        key = (name, game, level, len(r1), tuple(sorted(fb.items())) if fb else None)
        if key not in cache:
            baselines = (self.mg_point(game).get("baselines") or []) if game else []
            actions = m.get("median_actions")
            if r1 and actions:
                acts = sorted(int(r["level_rows"][-1].get("actions") or 0) for r in r1)
                n_s = int(m.get("clears") or 0)
                actions = (n_s * float(actions) + len(acts) * statistics.median(acts)) / (n_s + len(acts))
            ref = (((self.cfg.get("mg_sets") or {}).get(name) or {}).get("ref") or {}) if name else {}
            base_b = float(baselines[level - 1]) if 0 < level <= len(baselines) and baselines[level - 1] else None
            # multi-game (envs 22 Oct): a set no screen measured takes its references from the env table (``ref``:
            # {tokens, actions, baseline}); without any, the reward's token and action terms are both 0 for every try
            cache[key] = {"baseline": base_b or ref.get("baseline"),
                          "tokens": m.get("median_to_clear") or ref.get("tokens"),
                          "actions": actions or ref.get("actions")}
        return cache[key]

    def mg_turn_credit(self, folder, grp, rewards, name=None):
        """mg6 per-turn credit per try: {"tokens": {n: reply tokens}, "extra": {n: extra}}.
        Step level (GiGPO, Feng+ 2025): replies of different tries at the same board (level, frame hash of the board the
        reply saw) are compared on the reward shape applied to the rest of the try: clear x min(rhae_hi, (b_rem /
        a_rem)^2) x min(token_hi, T_anchor / tokens left), with a_rem the actions after the anchor and b_rem, T_anchor
        the means over the anchor's clears; leave-one-out, weight ``anchor``. The start board (every try's first reply)
        is skipped: the try advantage already holds it. Token cost: ``token`` x (group median reply tokens - reply
        tokens) / median, only on clears whose actions are within the level's base median (a short reply that buys
        extra actions earns nothing). Extras clipped to +-``clip``."""

        mg = self.cfg["mg"]
        tc = mg["turn_credit"]
        w_anchor, w_token, clip = float(tc.get("anchor", 1.0)), float(tc.get("token", 0.5)), float(tc.get("clip", 1.0))
        hi, thi = float(mg.get("rhae_hi", 4.0)), float(mg.get("token_hi", 6.0))
        pw, floor = 2.0, None
        if mg.get("reward_shape") == "floor":  # anchors use the try reward's shape on the remainder
            hi, thi = float(mg.get("action_hi", 4.0)), float(mg.get("token_hi", 3.0))
            pw, floor = float(mg.get("action_pow", 3.0)), float(mg.get("action_floor", 1.15))
        elif mg.get("reward_shape") == "aware" and grp:  # no action term; the level's cap as a ratio on the remainder
            lr0 = (grp[0].get("level_rows") or [{}])[0]
            ref = self.mg_level_ref(name, str(grp[0].get("point", ""))[:4], int(lr0.get("level") or 0))
            hi, thi, pw = 1.0, float(mg.get("token_hi", 3.0)), 0.0
            floor = self.mg_action_cap(ref) / float(ref["actions"]) if ref.get("actions") else None
        logeq = mg.get("reward_shape") == "logeq"
        if logeq:
            lo, fail = math.log(float(mg.get("log_clip", 8.0))), -math.log(float(mg.get("log_fail", 16.0)))
        tries, flagged = [], []
        rf = mg.get("reply_filter") or {}
        key = json.dumps(rf or {}, sort_keys=True)
        for row, reward in zip(grp, rewards):
            att = str(folder / "attempts" / row["label"])
            replies, end, flag = TURN_CACHE.get((att, key)) or turn_try(att, rf)
            flagged.append(dict(flag))
            lr = (row.get("level_rows") or [{}])[0]
            base = self.mg_level_ref(name, str(row.get("point", ""))[:4], int(lr.get("level") or 0)) if name else {}
            won = bool(clear_tokens(row)) if logeq else reward > 0
            gate = won and (not base.get("actions") or int(lr.get("actions") or 0) <= float(base["actions"]))
            tries.append((replies, won, end, gate))
        if tc.get("mode") == "gigpo":
            return self.mg_gigpo_steps(grp, rewards, tries, flagged, fail if logeq else 0.0)
        extra = [dict() for _ in grp]
        visits = {}  # anchor -> [(try, n, tokens left, actions left, cleared, first reply)]
        for i, (replies, cleared, end, _) in enumerate(tries):
            left = sum(t for _, t, _, _ in replies)
            seen = set()
            for k, (n, t, anchor, step) in enumerate(replies):
                if anchor is not None and anchor not in seen:
                    seen.add(anchor)
                    visits.setdefault(anchor, []).append((i, n, left, (end - step) if end is not None else 0, cleared, k == 0))
                left -= t
        sizes = []
        for anchor, vs in visits.items():
            if len({v[0] for v in vs}) < 2 or (len(vs) == len(grp) and all(v[5] for v in vs)):
                continue
            clears = [v for v in vs if v[4] and v[2] > 0 and v[3] > 0]
            if not clears:
                continue
            t_ref = statistics.mean(v[2] for v in clears)
            b_rem = statistics.mean(v[3] for v in clears)
            if logeq:  # the try reward's form on the remainder: actions as RHAE's square, tokens, equal weight in log
                ret = [self.mg_log_terms(2 * math.log(b_rem / v[3]), math.log(t_ref / v[2]), lo, fail)
                       if v[4] and v[2] > 0 and v[3] > 0 else fail for v in vs]
            else:
                ret = [min(hi, (b_rem / v[3]) ** pw) * min(thi, t_ref / v[2])
                       if v[4] and v[2] > 0 and v[3] > 0 and (floor is None or v[3] <= floor * b_rem) else 0.0
                       for v in vs]
            if mg.get("reward_log") and not logeq:
                ret = [math.log1p(r) for r in ret]
            total = sum(ret)
            sizes.append(len(vs))
            for v, r in zip(vs, ret):
                extra[v[0]][v[1]] = extra[v[0]].get(v[1], 0.0) + w_anchor * (r - (total - r) / (len(vs) - 1))
        gated = [t for replies, _, _, g in tries if g for _, t, _, _ in replies]
        if gated and w_token:
            med = statistics.median(gated) or 1.0
            for i, (replies, _, _, g) in enumerate(tries):
                if g:
                    for n, t, _, _ in replies:
                        extra[i][n] = extra[i].get(n, 0.0) + w_token * (med - t) / med
        self.mg_anchor_sizes = getattr(self, "mg_anchor_sizes", []) + sizes
        return [{"tokens": {n: t for n, t, _, _ in replies}, "extra": {n: max(-clip, min(clip, v)) for n, v in e.items()},
                 "filtered": f}
                for (replies, _, _, _), e, f in zip(tries, extra, flagged)]

    def mg_gigpo_steps(self, grp, rewards, tries, flagged, floor):
        """mg.turn_credit mode "gigpo" (v6x; Feng+ 2025 GiGPO in full): each gameplay reply is a step;
        step groups are the replies of the group's tries (every visit, revisits in the same try included) at one
        anchor state (level, frame hash of the board the reply saw); a step's return is the discounted return from it,
        gamma^(replies left to the try's last scored reply) x (the try's reward - the fail floor) (sparse terminal
        reward, a fail 0); A_step = (G - mean) / std over its anchor group (0 for a group of one or no spread); the
        reply's extra is omega x A_step, added to the try's leave-one-out advantage (mg_reply_balance, no clip, no
        rescale). Replies after the try's end (a harvested try's next-level request) get none."""
        tc = self.cfg["mg"]["turn_credit"]
        gamma, omega = float(tc.get("gamma", 0.95)), float(tc.get("omega", 1.0))
        if tc.get("step_scale") == "episode_std":  # GiGPO normalizes both terms per group; our episode term is the raw
            # leave-one-out reward gap, so the unit-variance step term is put on the same scale: x the group's reward
            # std (low-spread groups were step-dominated: hero-v6g ka59-L2, 30 sign flips at w_abs_median 0.22)
            omega *= statistics.pstdev([float(r) for r in rewards]) if len(rewards) > 1 else 0.0
        visits = {}  # anchor -> [(try, n, G)]
        for i, ((replies, _, end, _), reward) in enumerate(zip(tries, rewards)):
            ret = max(0.0, float(reward) - floor)
            idx = [k for k, (_, _, _, step) in enumerate(replies) if end is None or step <= end]
            last = idx[-1] if idx else -1
            for k in idx:
                n, _, anchor, _ = replies[k]
                if anchor is not None:
                    visits.setdefault(anchor, []).append((i, n, ret * gamma ** (last - k)))
        extra, stats = [dict() for _ in grp], self.__dict__.setdefault("mg_gigpo_stats", {})
        for anchor, vs in visits.items():
            stats["steps"] = stats.get("steps", 0) + len(vs)
            if len(vs) < 2:
                continue
            stats["in_group"] = stats.get("in_group", 0) + len(vs)
            if len({v[0] for v in vs}) >= 2:
                stats["cross_try"] = stats.get("cross_try", 0) + len(vs)
            g = [v[2] for v in vs]
            mu, sd = statistics.mean(g), statistics.pstdev(g)
            for i, n, G in vs:
                a = (G - mu) / sd if sd > 1e-9 else 0.0
                extra[i][n] = omega * a
                stats["abs_sum"] = stats.get("abs_sum", 0.0) + abs(a)
                stats["nonzero"] = stats.get("nonzero", 0) + int(a != 0.0)
            stats.setdefault("sizes", []).append(len(vs))
        stats["no_anchor"] = stats.get("no_anchor", 0) + sum(1 for replies, _, _, _ in tries for r in replies if r[2] is None)
        return [{"tokens": {n: t for n, t, _, _ in replies}, "extra": e, "filtered": f}
                for (replies, _, _, _), e, f in zip(tries, extra, flagged)]

    def mg_gigpo_summary(self):
        """This update's step-group stats: share of steps in an anchor group of size >= 2 (any, and
        across >= 2 tries), mean |A_step| over those steps and over all steps, median group size; then reset."""
        st = self.__dict__.pop("mg_gigpo_stats", None) or {}
        steps = st.get("steps", 0)
        if not steps:
            return None
        sizes = sorted(st.get("sizes") or [])
        return {"steps": steps, "share_group2": round(st.get("in_group", 0) / steps, 4),
                "share_cross_try": round(st.get("cross_try", 0) / steps, 4),
                "mean_abs_step_in_group": round(st.get("abs_sum", 0.0) / st["in_group"], 4) if st.get("in_group") else None,
                "mean_abs_step": round(st.get("abs_sum", 0.0) / steps, 4), "nonzero": st.get("nonzero", 0),
                "group_size_median": sizes[len(sizes) // 2] if sizes else None, "groups": len(sizes),
                "no_anchor": st.get("no_anchor", 0)}

    @staticmethod
    def mg_reply_flag(r, rf):
        """mg.reply_filter: "capped" (cut at the reply cap) or "loop" ('hmm' runs: at least ``hmm_min`` hmm tokens and
        ``hmm_density`` of the thinking tokens; P2 healthy rounds max 0.037, pilot c R3 up to 1.0), else None."""
        from .filler import reply_filler

        if rf.get("capped", True) and r.get("finish_reason") == "length":
            return "capped"
        hmm, think = reply_filler(r.get("tokens") or [])
        if hmm >= int(rf.get("hmm_min", 50)) and hmm >= float(rf.get("hmm_density", 0.1)) * max(1, think):
            return "loop"
        return None

    def mg_seed(self, pool, fixed, tests):
        """mg5f: U1's data from the screen's recorded base tries, trained while round 1 plays.
        Kept per train set only when the screen bench matches this run (base model: screen mode plays only the base;
        sampler, effort, completion cap, context window, rollout record) and, per try, when it would have played the
        same under this set's token cap (the model never sees the cap: a clear within it, or a try that ended within
        it); a group with any other try is dropped whole (no in-group bias). Starts: this run's pool (the fixed
        held-out start and diverged starts excluded); test sets never. Returns (kept as mg_select's, info)."""
        mg, sets, screen = self.cfg["mg"], self.cfg["mg_sets"], self.mg_screen
        want = {"temperature": self.study.get("temperature"), "reasoning_effort": self.study.get("reasoning_effort"),
                "completion_cap": self.study.get("completion_cap"),
                "window": (self.study.get("level_bench") or {}).get("window")}
        sampling = mg.get("seed_sampling") or {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "max_tokens": 16384}
        rollout = (self.study.get("level_bench") or {}).get("rollout")
        test_folders = {str(screen["sets"][n]["folder"]) for n in tests if n in screen["sets"]}
        source = mg.get("seed_from")  # {"root": an earlier loop's rl folder, "round": N}: its round-N training benches
        base_round = None
        if source:
            feed = [json.loads(x) for x in (Path(source["root"]) / "feed.jsonl").read_text().splitlines() if x.strip()]
            base_round = next((e for e in feed if e.get("kind") == "round" and e.get("round") == int(source["round"])), None)
        kept, info = [], {}
        for name, spec in sets.items():
            if spec["role"] != "train":
                continue
            entry, why = screen["sets"].get(name), []
            if entry is None:
                info[name] = {"dropped": "no screen set"}
                continue
            if source:  # the base played that round (no served adapter): its feed's round row says played_by 0
                folder = Path(source["root"]) / f"r{int(source['round']):02d}-{name}"
                if base_round is None or base_round.get("played_by") != 0:
                    why.append(f"round {source['round']} not played by the base")
                if not (folder / "level-bench.json").is_file():
                    info[name] = {"dropped": f"no bench {folder.name}"}
                    continue
            else:
                folder = Path(entry["folder"])
                if not folder.name.startswith("screen-") or str(folder) in test_folders:
                    why.append(f"folder {folder.name}")
            head = json.loads((folder / "level-bench.json").read_text())
            why += [f"{k} {head.get(k)!r} != {v!r}" for k, v in want.items() if head.get(k) != v]
            if (head.get("settings") or {}).get("sampling") is not None:
                why.append("per-request sampling override")
            if rollout is not None and (head.get("settings") or {}).get("rollout") != rollout:
                why.append("rollout record differs")
            if why:
                info[name] = {"dropped": "; ".join(why)}
                continue
            cap = int(spec.get("cap") or mg["cap"])
            rows = self.mg_rows(folder)
            if not spec.get("fresh"):
                # screen rows' point ids hash the screen's own (unmapped) paths; a loop's rows hash the mapped starts
                raw = ({tuple(x): tuple(x) for x in entry["starts"]} if source else
                       dict(zip([tuple(x) for x in entry["starts"]], entry.get("raw_starts") or entry["starts"])))
                tags = {"-" + self.mg_tag(spec, *raw[tuple(st)][:2]) for st in pool.get(name, []) if tuple(st) in raw}
                held = fixed.get(name)
                for st in (held if isinstance(held, list) else [held] if held else []):  # held-out starts never trained
                    if tuple(st) in raw:
                        tags.discard("-" + self.mg_tag(spec, *raw[tuple(st)][:2]))
                rows = [r for r in rows if any(r["point"].endswith(t) for t in tags)]
            bad_sampling = 0
            for r in rows:
                with open(folder / "attempts" / r["label"] / "rollout.jsonl") as file:
                    seen = {json.dumps(json.loads(line).get("sampling"), sort_keys=True) for line in file if line.strip()}
                if seen - {json.dumps(sampling, sort_keys=True)}:
                    r["_seed_bad"] = True
                    bad_sampling += 1

            def exact(r):  # the budget is checked between requests: a try at this budget, a clear within it, or a
                # try that never reached it plays the same
                t = clear_tokens(r)
                if r.get("_seed_bad"):
                    return False
                if int(r.get("token_budget") or 0) == cap:
                    return True
                return t <= cap if t else int(r.get("completion_tokens") or 0) <= cap

            groups = self.mg_groups(rows, int(mg.get("group_size") or 4))
            good = [r for g in groups.values() if all(exact(r) for r in g) for r in g]
            sel, sinfo = self.mg_select(folder, good, name)
            kept += [(*x, name) for x in sel]
            info[name] = {"folder": folder.name, "screen_cap": int(entry.get("cap") or 0), "cap": cap, "tries": len(rows),
                          "screen_groups": len(groups), "exact_groups": sinfo["groups"], "mixed": sinfo["mixed"],
                          "trained": sinfo["trained"], "bad_sampling": bad_sampling,
                          "clears": f"{sum(1 for r in good if clear_tokens(r))}/{len(good)}"}
        return kept, info

    async def mg_screen(self):
        mg, cap = self.cfg["mg"], int(self.cfg["mg"]["cap"])
        stub = os.environ.get("ARC_PREFLIGHT_STUB") or ""
        if stub.startswith("{") and json.loads(stub).get("shrink"):  # envs 22 Oct: CPU preflight of a screen: every
            # game and set still plays (load paths, recordings, starts), one recording each at a short budget
            for r in mg.get("record") or []:
                r.update(games=1, budget=min(int(r["budget"]), 40000))
            for st in mg.get("sets") or []:
                st["tries"] = min(int(st["tries"]), 2)
                if st.get("starts"):
                    st["starts"] = 1
            self.feed.emit("preflight_shrink", screen=True, records=len(mg.get("record") or []), sets=len(mg.get("sets") or []))
        base_model, recorded = self.server.model, {}
        if mg.get("active_tries"):  # mg6: recordings and sets share one attempt-slot pool sized to the KV pool
            from . import level_bench
            level_bench.SHARED_SLOTS = level_bench.SharedSlots(int(mg["active_tries"]), mg.get("active_kv_fill"), project=mg.get("kv_project"))
        if mg.get("timeshare") and mg.get("active_tries"):  # envs 22 Oct: a two-card screen (explorer seed): the
            # second card's server plays the base model from the start (no trainer), its own slot pool
            from . import level_bench
            await self.ts_boot()
            wake = await self.server2.wake()
            level_bench.SHARED_SLOTS2.active, level_bench.SHARED_SLOTS2.served = True, 0
            self.feed.emit("screen_second_server", wake_seconds=wake)

        # each finished recording / set is copied to the volume at once (ARC_VALIDATION_OUTPUT/partial/NAME), and a
        # relaunch with mg.reuse_from [earlier run dirs] takes a finished one whose spec matches instead of replaying it
        keep = Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "partial" if os.environ.get("ARC_VALIDATION_OUTPUT") else None
        replayed = set()
        # gate item 4: a restarted screen (launcher ARC_RL1_RESUME: earlier run dirs) keeps every finished
        # try, not only finished sets (level_bench.TRY_KEEP / TRY_REUSE)
        roots = list(mg.get("reuse_from") or []) + [x for x in os.environ.get("ARC_RL1_RESUME", "").split(",") if x]
        from . import level_bench as lb
        lb.TRY_KEEP, lb.TRY_REUSE = keep, [Path(r) / "partial" for r in roots]

        def reusable(name, marker, want):
            for root in roots:
                for src in (Path(root) / "partial" / name, Path(root) / "arena/trials/rl-loop/rl" / name):
                    try:
                        if (src / "level-bench.json").exists() and json.loads((src / marker).read_text()).get("spec") == want:
                            return src
                    except (OSError, ValueError):
                        continue
            return None

        async def take(src, name):
            await asyncio.to_thread(shutil.copytree, src, self.output / "rl" / name, dirs_exist_ok=True)
            self.feed.emit("mg_reused", name=name, source=str(src))
            await persist(self.output / "rl" / name)  # a later relaunch needs only this run's partial/
            return self.output / "rl" / name

        async def persist(folder):
            if keep is not None:
                try:
                    await asyncio.to_thread(shutil.copytree, folder, keep / Path(folder).name, dirs_exist_ok=True)
                except OSError as exc:  # a failed copy costs only the reuse
                    self.feed.emit("mg_persist_failed", folder=str(folder), error=str(exc)[:300])

        async def record(rec):
            name, want = f"rec-{rec['game']}", json.loads(json.dumps(rec))
            src = reusable(name, "screen-rec.json", want)
            if src is not None:
                folder = await take(src, name)
                from .rl_loop import usable

                rows = [r for r in json.loads((folder / "level-bench.json").read_text())["attempts"] if usable(r)]
                play = {"reused": str(src)}
            else:
                folder, rows, play = await self.bench(name, [rec["game"]], int(rec["games"]),
                                                      {rec["game"]: int(rec["budget"])}, model=base_model,
                                                      seed_base=int(rec.get("seed", 9100)), concurrency=int(rec["games"]),
                                                      reserve=120)
                (Path(folder) / "screen-rec.json").write_text(json.dumps({"spec": want}))
                kept = lb.kept_tries(name, lb.TRY_REUSE)
                labels = [a["label"] for a in json.loads((Path(folder) / "level-bench.json").read_text())["attempts"]]
                if not labels or any(x not in kept for x in labels):  # a game played anew: its sets' starts may differ
                    replayed.add(rec["game"])
                await persist(folder)
            recorded[rec["game"]] = str(folder)
            self.feed.emit("mg_record", game=rec["game"], folder=str(folder), play=play,
                           levels=sorted(len(r.get("level_rows") or []) for r in rows))

        async def stream(spec, games):
            """mg.stream_starts: the set's starts taken from its recordings while they play, the first ``starts``
            recordings (of the spec's half) to reach the level; None when the recordings finish first (the usual pick).
            The choice and its snapshots persist, and a restart takes them back so kept tries still match."""
            name, want = f"snap-{spec['name']}", int(spec["starts"])
            key = json.loads(json.dumps({"set": spec, "seed": mg.get("seed")}))
            snap = self.output / "rl" / name
            for root in roots:
                src = Path(root) / "partial" / name
                try:
                    chosen = json.loads((src / "starts.json").read_text())
                except (OSError, ValueError):
                    continue
                if chosen.get("spec") == key and all((src / Path(a).name / "rollout.jsonl").exists() for a, _ in chosen["starts"]):
                    await asyncio.to_thread(shutil.copytree, src, snap, dirs_exist_ok=True)
                    self.feed.emit("mg_stream_reused", name=spec["name"], source=str(src))
                    return [tuple(x) for x in chosen["starts"]], chosen["available"], False
            half, found, began = spec.get("start_half"), {}, time.monotonic()
            while True:
                finished = all(recs[g].done() for g in games)
                if finished:
                    return None
                for g in games:
                    live = self.output / "rl" / f"rec-{g}" / "attempts"
                    for attempt in (sorted(live.iterdir()) if live.is_dir() else []):
                        if attempt.name in found or (half is not None and start_half(attempt.name) != int(half)):
                            continue
                        got = await asyncio.to_thread(self.mg_live_start, spec, attempt, snap / f"{g}-{attempt.name}")
                        if got is not None:
                            found[attempt.name] = (str(got[0]), int(got[1]))
                if len(found) >= want:
                    break
                await asyncio.sleep(float(mg.get("stream_poll", 30)))
            starts = list(found.values())[:want]
            (snap / "starts.json").write_text(json.dumps({"spec": key, "starts": starts, "available": len(found)}))
            await persist(snap)
            self.feed.emit("mg_stream_starts", name=spec["name"], starts=len(starts),
                           seconds=round(time.monotonic() - began, 1))
            return starts, len(found), True

        async def play_set(spec, early=None):
            if early is not None:
                starts, available, fresh_pick = early
            else:
                sources = [recorded[s.split(":", 1)[1]] if s.startswith("record:") else s for s in spec.get("sources") or []]
                starts, available = ([], 1) if spec.get("fresh") else self.mg_starts(spec, sources)
                fresh_pick = False
            name = f"screen-{spec['name']}"
            if not spec.get("fresh") and not starts:  # no recorded game reached the level (preflight smoke2: the bench
                # raised "No start points selected" and the error ended the whole screen)
                self.feed.emit("mg_set_missing", name=spec["name"], error="no starts", available=available)
                return None
            want = json.loads(json.dumps({"set": spec, "starts": [list(x) for x in starts], "seed": mg.get("seed")}))
            # a set whose recording was replayed has new starts behind the same paths: never reused
            fresh_starts = fresh_pick if early is not None else any(
                x.split(":", 1)[1] in replayed for x in spec.get("sources") or [] if x.startswith("record:"))
            if fresh_starts:
                lb.TRY_REUSE_SKIP.add(name)
            src = None if fresh_starts else reusable(name, "screen-set.json", want)
            planned = max(1, len(starts)) * int(spec["tries"])
            if src is not None and json.loads((src / "screen-set.json").read_text())["out"]["tries"] < 0.75 * planned:
                src = None  # cut short (screen mg6-6: two probe sets censored at the deadline with 0 tries): replay
            if src is not None:
                await take(src, name)
                out = json.loads((src / "screen-set.json").read_text())["out"]
                self.feed.emit("mg_set", name=spec["name"], reused=True,
                               **{k: v for k, v in out.items() if k != "groups"},
                               mixed=sum(g["mixed"] for g in out["groups"].values()), of=len(out["groups"]))
                return spec["name"], out
            folder, rows, play = await self.mg_play(name, spec, starts, spec["tries"],
                                                    model=base_model, seed=int(spec.get("seed", mg.get("seed", 8000))))
            groups = self.mg_groups(rows)
            toks = [clear_tokens(r) for r in rows]
            at_caps = {c: sum(1 for t in toks if t and t <= c) for c in mg.get("at_caps") or []}
            out = {"role": spec["role"], "game": spec["game"], "level": spec["level"], "folder": str(folder),
                   "cap": int(spec.get("cap") or cap), "at_caps": at_caps, "tries": len(rows),
                   "compactions": sum(int(r.get("compactions") or 0) for r in rows),
                   "starts": [list(s) for s in starts], "available": available, "play": play,
                   "metrics": held_metrics(rows),
                   "groups": {k: {"tries": len(g), "clears": sum(1 for r in g if clear_tokens(r)),
                                  "mixed": 0 < sum(1 for r in g if clear_tokens(r)) < len(g)} for k, g in sorted(groups.items())}}
            self.feed.emit("mg_set", name=spec["name"], **{k: v for k, v in out.items() if k != "groups"},
                           mixed=sum(g["mixed"] for g in out["groups"].values()), of=len(out["groups"]))
            (Path(folder) / "screen-set.json").write_text(json.dumps({"spec": want, "out": out}, default=str))
            await persist(folder)
            return spec["name"], out

        # a set from recordings waits only for its own games' recordings (no barrier on the whole record phase)
        recs = {r["game"]: asyncio.ensure_future(record(r)) for r in mg.get("record") or []}

        async def after(spec):
            games = [x.split(":", 1)[1] for x in spec.get("sources") or [] if x.startswith("record:")]
            try:
                early = await stream(spec, games) if mg.get("stream_starts") and games and not spec.get("fresh") else None
            except Exception as exc:  # the usual pick after the recordings end
                self.feed.emit("mg_stream_failed", name=spec["name"], error=repr(exc)[:300])
                early = None
            if early is None:
                await asyncio.gather(*(recs[g] for g in games))
            try:
                return await play_set(spec, early)
            except TimeoutError as exc:  # out of wall time: the finished sets still reach screen.json (mg6-6 lost all)
                self.feed.emit("mg_set_missing", name=spec["name"], error=str(exc))
                return None
            except Exception as exc:  # one set's failure never ends the others (preflight smoke2)
                import traceback

                self.feed.emit("mg_set_missing", name=spec["name"], error=repr(exc)[:500],
                               traceback=traceback.format_exc()[-3000:])
                return None

        async def ticker():  # envs 22 Oct: live screen counters every 2 min (only on change: a hang reads as silence)
            last = None
            while True:
                await asyncio.sleep(120)
                p1, p2 = lb.SHARED_SLOTS, lb.SHARED_SLOTS2
                now = {**lb.PROGRESS, "running": (getattr(p1, "used", 0) or 0) + (getattr(p2, "used", 0) or 0),
                       "recorded": len(recorded), "sets_done": sum(1 for f in setf if f.done())}
                if now != last:
                    self.feed.emit("play_progress", screen=True, **now)
                    last = now

        lb.PROGRESS.update(done=0, clears=0)
        setf = [asyncio.ensure_future(after(s)) for s in mg["sets"]]
        tick = asyncio.ensure_future(ticker())
        try:
            done = await asyncio.gather(*setf, *recs.values())
        finally:
            tick.cancel()
        sets = dict(x for x in done[: len(mg["sets"])] if x)
        screen = {"cap": cap, "spec": mg, "recorded": recorded, "sets": sets,
                  "missing": [x["name"] for x in mg["sets"] if x["name"] not in sets]}
        (self.output / "rl" / "screen.json").write_text(json.dumps(screen, indent=1, default=str))
        self.feed.emit("final", stop=None, eval={n: {"metrics": s["metrics"], "mixed": sum(g["mixed"] for g in s["groups"].values()),
                                                     "of": len(s["groups"])} for n, s in sets.items()}, base=None)
        return {n: s["metrics"] for n, s in sets.items()}

    def mg_screen_file(self):
        """screen.json with its container paths (/tmp/arena/...) mapped to the screen run's volume copy
        (``mg.screen_root``, e.g. /runs/validation/rl1-screen-mg1/arena); ``mg.screens`` [{screen, root}] add later
        screens' sets."""
        mg = self.cfg["mg"]
        out = None
        for item in [{"screen": mg["screen"], "root": mg.get("screen_root")}] + list(mg.get("screens") or []):
            text, root = Path(item["screen"]).read_text(), item.get("root")
            mapped = text.replace('"/tmp/arena/', f'"{root.rstrip("/")}/') if root else text
            for a, b in (item.get("map") or {}).items():  # a running screen's kept partial sets (partial/<name>)
                mapped = mapped.replace(f'"{a}', f'"{b}')
            one, raw = json.loads(mapped), json.loads(text)
            for n, s in one["sets"].items():  # screen rows' point ids hash the unmapped paths
                s["raw_starts"] = raw["sets"][n].get("starts")
            if out is None:
                out = one
            else:  # a later screen's sets replace or add to the earlier ones
                out["sets"].update(one["sets"])
        return out

    def mg_rows(self, folder):
        from .rl_loop import usable

        return [r for r in json.loads((Path(folder) / "level-bench.json").read_text())["attempts"] if usable(r)
                and restored(r) and level_of(r) is not None]

    async def mg_update(self, number, kept, probes):
        specs, lag_of = [], {}
        level_norm = self.cfg["mg"].get("level_norm")  # mg5b: Dr. GRPO's constant per level, not one global cap
        for folder, row, w, share, *rest in kept:
            spec = self.windows(folder, [row], prefix="t-")[0]
            if self.mg_versioned():
                lag_of[spec["name"]] = self.mg_lag(row)
            name = rest[0] if rest else None
            extra = {"set": name}
            if level_norm and name:
                ref = (self.mg_screen["sets"].get(name, {}).get("metrics") or {}).get("median_to_clear")
                extra["norm_cap"] = float(ref or self.cfg["mg_sets"][name].get("cap") or self.cfg["mg"]["cap"])
            specs.append({**spec, "w": float(w), "whole": True, "turn_key": (str(Path(folder)), row["label"]), "window": int(self.cfg["lvl_window"]), "floor": 1024,
                          "margin": 1024, "credited_only": True, "share": share, **extra})
        groups = {}
        for spec in specs:
            groups.setdefault(spec["share"], []).append(spec)
        rng, per = random.Random(number), max(1, int(self.cfg["l2_starts_per_step"]))
        keys = sorted(groups)
        rng.shuffle(keys)
        # l4's dose per update (owner): at most max_groups groups (l4: 8 starts, 2 per step = 4 steps)
        keys = keys[: int(self.cfg["mg"].get("max_groups") or len(keys))]
        specs = [sp for key in keys for sp in groups[key]]
        balance = None
        turn = getattr(self, "mg_turn", {})
        if self.cfg["mg"].get("turn_credit") and self.cfg["mg"].get("balance"):
            balance = self.mg_reply_balance(specs, turn)
            self.mg_turn, self.mg_anchor_sizes = {}, []  # this update's picks are spent; the next selection refills
        elif self.cfg["mg"].get("balance"):  # checklist 15 (mg5): push-down mass (|w| x trained tokens) = push-up mass
            targets = (await self.trainer.must("targets", windows=specs))["targets"]
            unit = lambda sp: targets.get(sp["name"], 0) / float(sp.get("norm_cap") or self.cfg["mg"]["cap"])
            up = sum(sp["w"] * unit(sp) for sp in specs if sp["w"] > 0)
            down = sum(-sp["w"] * unit(sp) for sp in specs if sp["w"] < 0)
            scale = up / down if down > 0 and up > 0 else 1.0
            for sp in specs:
                if sp["w"] < 0:
                    sp["w"] *= scale
            balance = {"up_mass": round(up, 3), "down_mass": round(down, 3), "down_scale": round(scale, 4)}
            mass = {}  # each level's share of the update's gradient mass (|w| x trained tokens / its norm constant)
            for sp in specs:
                mass[sp.get("set")] = mass.get(sp.get("set"), 0.0) + abs(sp["w"]) * unit(sp)
            total = sum(mass.values()) or 1.0
            balance["level_share"] = {str(k): round(v / total, 3) for k, v in sorted(mass.items(), key=lambda kv: str(kv[0]))}
        if self.cfg["mg"].get("stratify"):  # each Adam step holds an even share of every level's groups
            by = {}
            for key in keys:
                by.setdefault(groups[key][0].get("set"), []).append(key)
            order = [key for name in sorted(by, key=str) for key in by[name]]
            n = max(1, -(-len(order) // per))
            if self.cfg["mg"].get("minibatches"):  # standard dosing: a fixed count per pass, batch size follows data
                n = max(1, min(len(order), int(self.cfg["mg"]["minibatches"])))
            # global position p -> step p mod n: balanced step sizes, at most one group per level per step (a level
            # has at most n groups)
            batches = [[s for key in order[i::n] for s in groups[key]] for i in range(n)]
        elif self.cfg["mg"].get("minibatches"):
            n = max(1, min(len(keys), int(self.cfg["mg"]["minibatches"])))
            batches = [[s for key in keys[i::n] for s in groups[key]] for i in range(n)]
        else:
            batches = [[s for key in keys[i:i + per] for s in groups[key]] for i in range(0, len(keys), per)]
        epochs = int(self.cfg["mg"].get("epochs") or 1)
        if epochs > 1:  # PPO epochs over the same minibatches, each later epoch in a new order;
            # pi_old stays the update's start (the old pass covers every window, dosed_update)
            one, erng = list(batches), random.Random(number * 7919 + 1)
            for _ in range(epochs - 1):
                order = list(one)
                erng.shuffle(order)
                batches = batches + order
        probing = bool(self.cfg["mg"].get("lr_probe"))  # P1: probes after every step, no KL control
        standard = self.cfg["mg"].get("dosing") == "standard"  # probe KL stays a diagnostic: no probe early stop
        early = None if probing or standard else self.mg_early()
        aux, nb_read = {}, None
        if self.cfg["mg"].get("nextboard"):
            aux, nb_read = await self.mg_nextboard(number, kept)
        self.mg_lag_now = lag_of  # the old pass's live engine-vs-pi_old check groups windows by lag
        merged, path = await self.dosed_update(number, None, None, None, probes, batches=batches, signed=True,
                                               probe_last=not probing, max_windows=100000, early=early,
                                               cmd_extra={"hold_previous": bool(early) or standard,
                                                          "share_prefix": True, "prefix_chunk": int(self.cfg["path_chunk"]),
                                                          "norm": self.cfg["l2_norm"], "norm_cap": int(self.cfg["mg"]["cap"]),
                                                          "batch_tokens": int(self.cfg["mg"].get("batch_tokens") or 0),
                                                          "batch_max": int(self.cfg["mg"].get("batch_max") or 8),
                                                          # MFU exact fixes (21 Oct; rl1-prof-mfu4): prefix snapshots,
                                                          # encode prefetch depth
                                                          **dict(self.cfg["mg"].get("trainer_fast") or {}), **aux})
        merged["trained_tries"] = len(specs)
        self.mg_lag_now = None
        if lag_of:  # mg.lookahead_below: what staleness costs, per lag (TIS, GSPO clipping, trained tokens)
            merged["lag_cost"] = lag_cost(merged.get("windows") or [], lag_of)
        if nb_read:
            merged["nextboard_read"] = nb_read
        if aux:
            merged["nextboard"] = {"file": aux["aux"]["file"], **self.mg_nb_summary,
                                   "steps": [s.get("aux") for s in merged.get("steps") or []]}
        if balance:
            merged["mg_balance"] = balance
        if merged.get("ok") and (merged.get("early") or {}).get("action") != "rollback" and self.cfg["mg"].get("checkpoint") \
                and not probing:
            # adapter + optimizer state each update (resume); a rolled-back update leaves none
            self.checkpoint(number, force=True)
            await self.save_checkpoint(background=bool(self.cfg["mg"].get("async_checkpoint")))
            if self.cfg["mg"].get("reload_check") and not getattr(self, "mg_reload_checked", False):
                self.mg_reload_checked = True
                await self.mg_reload_check(number, merged, probes)
        return merged, path

    async def mg_nextboard(self, number, kept):
        """mg.nextboard (envs arm): this round's packed next-board file from its trained tries (never a held-out game)
        and, every ``read_every`` rounds before the update (round 1: the served base), a forward-only read of the
        held-out file ``read``. Returns ``({"aux": trainer args} or {}, read or None)``."""
        from . import nextboard_pack as nbp

        nb = self.cfg["mg"]["nextboard"]
        tries = [(str(row.get("point", ""))[:4], Path(folder) / "attempts" / row["label"]) for folder, row, *_ in kept]
        rows, summary = nbp.build(tries, k=int(nb.get("attempts", 16)), seed=number,
                                  board_every=int(nb.get("board_every", 4)), max_steps=int(nb.get("max_steps", 48)))
        self.mg_nb_summary = summary
        self.feed.emit("nextboard_data", round=number, **summary)
        aux = {}
        if rows:
            dest = nbp.write(rows, self.output / "rl" / "nextboard" / f"r{number:02d}.jsonl")
            aux = {"aux": {"file": str(dest), "weight": float(nb.get("weight", 0.1)), "windows": int(nb.get("windows", 2)),
                           "max_tokens": int(nb.get("max_tokens", 16384))}}
        read = None
        every = int(nb.get("read_every", 2))
        if nb.get("read") and every > 0 and (number - 1) % every == 0:
            if number == 1 and getattr(self, "nb_base_read", None):  # the dose probe's arms all start at the base
                return aux, self.nb_base_read
            read = await self.trainer.must("nb_score", timeout=int(self.cfg["update_timeout"]), file=nb["read"],
                                           max_tokens=int(nb.get("max_tokens", 16384)))
            read = {k: v for k, v in read.items() if k != "ok"}
            if number == 1:
                self.nb_base_read = read
            self.feed.emit("nextboard_read", round=number, **read)
        return aux, read

    async def mg_reload_check(self, number, merged, probes, state=None):
        """First-update smoke inside the run: the checkpoint just written reloads into the
        trainer and rescores the probes; the reload may move them no more than the measured repeat floor allows
        (RELOAD_EPS / RELOAD_CEIL); a mismatch stops the run. ``hold_previous`` keeps the KL-to-previous reference
        where the update left it."""
        began = time.time()

        async def score(label):
            got = (await self.trainer.must("score", windows=probes, label=label, hold_previous=True))["windows"]
            return {n: float(v["kl_base"]) for n, v in got.items() if (v or {}).get("kl_base") is not None}

        # r4 (22 Oct): 1.86e-3 against a fixed 1e-3 bar set on the old deterministic stack; the trainer stack now
        # scores with atomic-order noise and self-routed probes amplify it. The floor is measured here, in this
        # process, on this state: two scores before the reload; pass when the reload moves no more than 3x that
        # repeat spread + 2e-4, under a hard 5e-3 ceiling (a noisy floor cannot hide a real save/reload bug).
        first, second = await score("reload-pre1"), await score("reload-pre2")
        state = Path(state or Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "state" / f"r{number:02d}")
        loaded = await self.trainer.must("load_checkpoint", path=str(state))
        again = await score("reload")
        names = [n for n in first if n in second and n in again]
        floor = max([abs(first[n] - second[n]) for n in names] or [0.0])
        diffs = [abs(again[n] - first[n]) for n in names]
        bar = min(RELOAD_CEIL, 3 * floor + RELOAD_EPS)
        row = {"round": number, "probes": len(first), "compared": len(diffs), "max_diff": max(diffs) if diffs else None,
               "repeat_floor": floor, "bar": round(bar, 6), "updates": loaded.get("updates"),
               "seconds": round(time.time() - began, 1)}
        merged_at = {n: p["kl_base"] for n, p in (merged.get("probes") or {}).items() if p.get("kl_base") is not None}
        gap = [abs(float(merged_at[n]) - first[n]) for n in names if n in merged_at]
        row["update_vs_saved"] = max(gap) if gap else None  # the update's own probe scores against the saved state
        row["ok"] = (bool(diffs) and len(diffs) == len(first) and row["max_diff"] <= bar
                     and (row["update_vs_saved"] or 0.0) <= RELOAD_CEIL)
        self.feed.emit("reload_check", **row)
        if not row["ok"]:
            self.feed.emit("alert", reason="reload_mismatch", **row)
            raise RuntimeError(f"reload check failed after U{number}: {row}")

    def mg_tests(self, results, screen):
        """Per test set and pooled untrained games: the adapter's metrics beside the screen's base."""
        out, pools = {}, {"untrained_games": ([], []), "primary": ([], [])}
        for name, (folder, rows, play) in results.items():
            spec = next(s for s in self.cfg["mg"]["sets"] if s["name"] == name)
            cap = int(spec.get("cap") or self.cfg["mg"]["cap"])
            # the screen's base tries at this set's cap (the screen played a looser cap; the model never sees it)
            base_rows = ([] if screen["sets"][name].get("starts_only") else
                         [self.mg_censor(r, cap) for r in self.mg_rows(screen["sets"][name]["folder"])])
            out[name] = {"policy": held_metrics(rows), "base": held_metrics(base_rows), "play_s": play["seconds"]}
            for key, flag in (("untrained_games", "untrained_game"), ("primary", "primary")):
                if spec.get(flag):
                    pools[key][0].extend(rows)
                    pools[key][1].extend(base_rows)
        for key, (pol, base) in pools.items():
            out[key] = {"policy": held_metrics(pol), "base": held_metrics(base)}
        return out

    @staticmethod
    def mg_censor(row, cap):
        """A base try played at a looser cap, as it would end at ``cap``: a clear beyond the cap is a fail, its
        spent tokens the cap."""
        t = clear_tokens(row)
        if t is not None and t <= cap:
            return row
        return {**row, "outcome": "budget", "level_rows": [], "completion_tokens": min(int(row.get("completion_tokens") or cap), cap)}

    def mg_rules(self, number, plays, row, played_by):
        """stop rules, on one round: (a) the round played after 4 policy updates (and its
        predecessor) shows no gain on the training sets at their caps against the base rounds 1-2 (clear rate not
        higher and median tokens to clear not lower); (b) the primary unseen-level trend clearly worse than base in 2
        consecutive rounds (pooled clear rate below base - 0.2, or RHAE per 10K below 0.67 x base); (d) a bench's
        usable share below 0.75. (c) is the KL guard's second trip, in the round body. Returns the rule or None."""
        mg = self.cfg["mg"]
        train = [r for n, (_, rows, _) in plays.items() if self.cfg["mg_sets"][n]["role"] == "train" for r in rows]
        self.mg_train_rows = getattr(self, "mg_train_rows", {})
        self.mg_train_rows[number] = (played_by, train)

        def per_request(rows):  # completion tokens per model request over a round's played tries
            asked = sum(int(r.get("admitted_requests") or r.get("gameplay_requests") or 0) for r in rows)
            return round(sum(int(r.get("completion_tokens") or 0) for r in rows) / asked, 1) if asked else None

        played = train + [r for n, (_, rows, _) in plays.items() if self.cfg["mg_sets"][n]["role"] == "test" for r in rows]
        # mg5f: the base-played rounds (round 2 is a policy round when U1 is seeded)
        base_tpr = (per_request([r for pb, rows in self.mg_train_rows.values() if pb == 0 and not getattr(self, "mg_init", False)
                                 for r in rows])
                    or getattr(self, "mg_base_tpr", None))
        self.mg_base_tpr = base_tpr
        row["tokens_per_request"] = {"round": per_request(played), "train": per_request(train), "base": base_tpr}
        limit = mg.get("length_guard")  # (e) above limit x base in 2 policy rounds running
        if limit and self.mg_policy_round(played_by) and base_tpr and train:  # training sets against the base's training rounds
            long = (per_request(train) or 0) > float(limit) * base_tpr
            self.mg_long = getattr(self, "mg_long", 0) + 1 if long else 0
            row["tokens_per_request"].update(over=long, streak=self.mg_long)
            if self.mg_long >= 2:
                row["rule_detail"] = row["tokens_per_request"]
                return "rule_e_length"
        if train:  # the round's training tries against the screen's base tries on the same starts, at the set caps
            row["train_vs_screen"] = {"policy": held_metrics(train), "screen": held_metrics(self.mg_screen_base([number]))}
            sets = self.cfg["mg_sets"]
            br = [r for n, (_, rows, _) in plays.items() if sets[n]["role"] == "train" and sets[n].get("branch_of") for r in rows]
            if br:  # dashboard: normal starts and branch continuations apart
                row["train_vs_screen"]["normal"] = held_metrics(
                    [r for n, (_, rows, _) in plays.items() if sets[n]["role"] == "train" and not sets[n].get("branch_of")
                     for r in rows])
                row["train_vs_screen"]["branch"] = held_metrics(br)
            l13 = [r for n, (_, rows, _) in plays.items() if sets[n]["role"] == "train"
                   and int(sets[n].get("level") or 0) in (1, 3) for r in rows]
            if l13:
                row.setdefault("l13", {})["train"] = held_metrics(l13)
        if played_by >= 2 and played_by - 1 == int(mg.get("rule_a_updates", 4)):  # 4 updates: U2..U5 -> a5
            rounds_base = [r for k in (1, 2) for r in self.mg_train_rows.get(k, (0, []))[1]]
            pol = train + self.mg_train_rows.get(number - 1, (0, []))[1]
            # against the same starts' screen base (starts rotate by round), censored at the cap
            b, p = held_metrics(self.mg_screen_base([number - 1, number])), held_metrics(pol)
            row["rule_a"] = {"base": b, "policy": p, "base_rounds_1_2": held_metrics(rounds_base)}
            better = ((p["clear_rate"] or 0) > (b["clear_rate"] or 0)
                      or (p["median_to_clear"] and b["median_to_clear"] and p["median_to_clear"] < b["median_to_clear"]))
            if not better:
                row["rule_detail"] = row["rule_a"]
                return "rule_a_no_gain"
        prim = (row.get("tests") or {}).get("primary")
        if prim and prim["policy"]["tries"]:
            p, b = prim["policy"], prim["base"]
            worse = ((p["clear_rate"] or 0) < (b["clear_rate"] or 0) - 0.2
                     or (b["rhae_per_10k"] and (p["rhae_per_10k"] or 0) < 0.67 * b["rhae_per_10k"]))
            self.mg_worse = getattr(self, "mg_worse", 0) + 1 if worse else 0
            row["rule_b"] = {"worse": worse, "streak": self.mg_worse}
            if self.mg_worse >= 2:
                row["rule_detail"] = prim
                return "rule_b_primary_worse"
        return None

    def mg_unseen_reads(self, screen=None):
        """``mg.unseen_read``: one {set, tries, every, offset, base_first, starts} or a list of them (v6x, 
        L2 and L4 transfer reads) (the main test is unseen levels): each named test set plays
        ``tries`` tries per start every ``every``-th round (rounds = ``offset`` mod ``every``), as the other test sets'
        reads (in the tail, same seeds each read); its other rounds play nothing. ``starts``: the set's first k starts
        only. Checked against the sets and the screen at the loop's start."""
        u = self.cfg["mg"].get("unseen_read")
        if not u:
            return []
        out = []
        for one in (u if isinstance(u, list) else [u]):
            unknown = set(one) - {"set", "tries", "every", "base_first", "offset", "starts"}
            spec = self.cfg["mg_sets"].get(one.get("set"))
            if unknown or spec is None or spec["role"] != "test" or int(one.get("tries") or 0) <= 0 \
                    or int(one.get("every") or 1) < 1 or ("starts" in one and int(one["starts"] or 0) < 1):
                raise ValueError(f"mg.unseen_read: needs a test set, tries > 0, every >= 1, starts >= 1 (got {one})")
            if screen is not None and one["set"] not in screen["sets"]:
                raise ValueError(f"mg.unseen_read: {one['set']} has no screen entry (the base comparison)")
            if any(x["set"] == one["set"] for x in out):
                raise ValueError(f"mg.unseen_read: {one['set']} listed twice")
            item = {"set": one["set"], "tries": int(one["tries"]), "every": int(one.get("every") or 1),
                    "base_first": bool(one.get("base_first")), "offset": int(one.get("offset") or 0),
                    "starts": int(one["starts"]) if one.get("starts") else None}
            if item["base_first"]:  # the base model's tries at the read's starts and seeds beside this process's first
                # read (a restart before it reads base again): a bench of its own (not a test set: no screen
                # comparison, never trained), in the tail like the read
                self.cfg["mg_sets"].setdefault(f"{one['set']}-base", {**spec, "name": f"{one['set']}-base", "role": "base_read"})
            out.append(item)
        return out

    def mg_unseen_read(self, screen=None, name=None):
        """The unseen read of set ``name`` (else the first), or None."""
        reads = self.mg_unseen_reads(screen)
        return next((u for u in reads if name is None or u["set"] == name), None)

    def mg_heldout_alternate(self, number, play_s, trainer_s):
        """mg.heldout_alternate {margin_s: 600, rounds: 2}: the held-out read (fixed starts, every
        round) switches to every 2nd round, for good, once play outlasts the trainer lane (the update's whole path, its
        wall time beside play) by more than margin_s in ``rounds`` consecutive rounds. Logged as heldout_alternate."""
        cfg = {"margin_s": 600, "rounds": 2, **(self.cfg["mg"].get("heldout_alternate") or {})}
        alt = self.mg_alt = dict(getattr(self, "mg_alt", None) or {"on": False, "streak": 0})
        row = {"play_s": round(play_s, 1), "trainer_s": None if trainer_s is None else round(trainer_s, 1)}
        if alt["on"] or trainer_s is None:  # no update this round: no comparison, the streak holds
            return {**row, **alt}
        alt["streak"] = alt["streak"] + 1 if play_s - trainer_s > float(cfg["margin_s"]) else 0
        if alt["streak"] >= int(cfg["rounds"]):
            alt.update(on=True, since=number)
            self.feed.emit("heldout_alternate", round=number, on=True, play_s=row["play_s"], trainer_s=row["trainer_s"],
                           margin_s=cfg["margin_s"], rounds=cfg["rounds"])
        return {**row, **alt}

    def mg_digest_ok(self, row):
        """mg.digest_gate: a try restored from an in-run snapshot (harvest or branch cut) trains
        only when its first live prompt's token ids hash as the source's recorded request did (byte-identical)."""
        rs = row.get("restore")
        if not self.cfg["mg"].get("digest_gate") or not rs:
            return True
        src = str(rs.get("source") or "")
        if not re.search(r"/(harvest|branch|branch-from-tar)/", src):
            return True
        return bool((rs.get("prompt_digest") or {}).get("same"))

    def mg_test_tries(self, number, name):
        s, u = self.cfg["mg_sets"][name], self.mg_unseen_read(name=name)
        if u:
            return u["tries"] if number % u["every"] == u["offset"] % u["every"] else 0
        return int(s.get("round_tries", s["tries"]))

    def mg_ok_starts(self, name, screen):
        """A set's screen starts that restored (at least one usable base try); a fresh set has none."""
        spec = next(s for s in self.cfg["mg"]["sets"] if s["name"] == name)
        if screen["sets"][name].get("starts_only"):  # a frozen start set no screen played (v6x sc25-L4): every
            # listed start; its base comparison is the run's matched base read (unseen_read base_first)
            return [tuple(st) for st in screen["sets"][name]["starts"]]
        points = {r["point"] for r in self.mg_rows(screen["sets"][name]["folder"])}
        return self.mg_matched(spec, screen["sets"][name], points)

    async def mg_prescreen(self, train, tests, screen):
        """Every candidate start replayed ``mg.prescreen`` times with no model call (check-only restores, which end at
        the first live request); a start any of whose replays ends off the recorded game state (board, steps, levels,
        actions) is returned as a tag to exclude."""
        from .restore import load

        k, sets = int(self.cfg["mg"]["prescreen"]), self.cfg["mg_sets"]
        by_game = {}
        for name, starts in {**train, **tests}.items():
            spec, entry = sets[name], screen["sets"][name]
            raw = dict(zip([tuple(x) for x in entry["starts"]], entry.get("raw_starts") or entry["starts"]))
            for st in starts:
                r = raw.get(tuple(st), st)
                by_game.setdefault(spec["game"], {})[self.mg_tag(spec, r[0], r[1])] = (spec, tuple(st))
        jobs = []
        for game, starts in by_game.items():
            base = self.mg_point(game)
            points = [{**base, "id": f"{base['id']}-{tag}", "level": load(st[0], st[1])["level"],
                       "restore": {"attempt": st[0], "request": int(st[1]), "check_only": True,
                                   "budget": int(spec.get("cap") or self.cfg["mg"]["cap"])}}
                      for tag, (spec, st) in starts.items()]
            path = self.work / f"prescreen-{game}-points.json"
            path.write_text(json.dumps({"schema": 1, "points": points}))
            jobs.append(self.bench(f"prescreen-{game}", [game], k, {game: int(self.cfg["mg"]["cap"])}, points=path,
                                   model=self.server.model, seed_base=None, concurrency=len(points) * k, reserve=120))
        bad, seen = set(), {}
        for folder, _, play in await asyncio.gather(*jobs):
            for r in json.loads((Path(folder) / "level-bench.json").read_text())["attempts"]:
                tag = next((t for game in by_game.values() for t in game if str(r.get("point", "")).endswith("-" + t)), None)
                if tag is None:
                    continue
                state = (r.get("restore") or {}).get("state") or {}
                diverged = r.get("outcome") == "replay_mismatch" or state.get("ok") is False
                seen.setdefault(tag, [0, 0])
                seen[tag][0] += 1
                seen[tag][1] += int(diverged)
                if diverged:
                    bad.add(tag)
        self.feed.emit("prescreen", tries=k, starts=len(seen), diverged={t: f"{v[1]}/{v[0]}" for t, v in seen.items() if v[1]})
        return sorted(bad)

    def mg_screen_base(self, numbers):
        """The screen's base tries on the starts the given rounds played (a fresh set: all its screen tries), each
        censored at its set's cap."""
        cache, out = getattr(self, "mg_screen_rows", {}), []
        self.mg_screen_rows = cache
        for number in numbers:
            for name, starts in (getattr(self, "mg_round_starts", {}).get(number) or {}).items():
                if name not in self.mg_screen["sets"]:  # multi-game: a set no screen played has no base tries here
                    continue
                spec, entry = self.cfg["mg_sets"][name], self.mg_screen["sets"][name]
                if entry["folder"] not in cache:
                    cache[entry["folder"]] = self.mg_rows(entry["folder"])
                rows, cap = cache[entry["folder"]], int(spec.get("cap") or self.cfg["mg"]["cap"])
                if not spec.get("fresh"):
                    raw = dict(zip([tuple(x) for x in entry["starts"]], entry.get("raw_starts") or entry["starts"]))
                    tags = {"-" + self.mg_tag(spec, raw[tuple(st)][0], raw[tuple(st)][1]) for st in starts if tuple(st) in raw}
                    rows = [r for r in rows if any(r["point"].endswith(t) for t in tags)]
                out += [self.mg_censor(r, cap) for r in rows]
        return out

    def mg_matched(self, spec, entry, points):
        """``entry``'s (mapped) starts whose screen point (tagged from the unmapped path) is in ``points``."""
        raw = entry.get("raw_starts") or entry["starts"]
        drop = set(self.cfg["mg"].get("exclude_starts") or [])  # tags (L3-cc11d4-k019) of starts whose replay diverged
        return [tuple(st) for st, r in zip(entry["starts"], raw)
                if self.mg_tag(spec, r[0], r[1]) not in drop
                and any(p.endswith("-" + self.mg_tag(spec, r[0], r[1])) for p in points)]

    async def served_watch(self, every=300, blind=900):
        """R3: the capture's state in the feed every ``every`` s; an alert when the server has finished requests for
        ``blind`` s and nothing is on disk, or the capture counts errors (the 20 Oct check recorded nothing for 40 min)."""
        from .served_route import status

        began, warned = time.time(), set()
        while True:
            await asyncio.sleep(every)
            try:
                state = status(self.served_dir)
            except Exception as error:  # noqa: BLE001
                state = {"error": repr(error)[:300]}
            self.feed.emit("served_status", **state)
            server = state.get("server") or {}
            stand_in = bool(os.environ.get("ARC_PREFLIGHT_FAKE_TRAINER"))  # CPU preflight: no real server writes
            if "empty" not in warned and not stand_in and time.time() - began > blind and not state.get("rt"):
                warned.add("empty")
                self.feed.emit("alert", why="served-route capture: no request files", detail=state)
            if "errors" not in warned and server.get("errors"):
                warned.add("errors")
                self.feed.emit("alert", why="served-route capture errors", detail=server)

    async def mg_engine_admission(self):
        """mg.engine_admission: polls the play server's /metrics and pauses new tries on the shared pool (level_bench
        EngineAdmission); a pause or resume is a feed row (at most one per ``report_s``), counts go in the round row."""
        import httpx

        from . import level_bench

        spec = dict(self.cfg["mg"]["engine_admission"]) if isinstance(self.cfg["mg"]["engine_admission"], dict) else {}
        interval, report = float(spec.pop("interval", 5.0)), float(spec.pop("report_s", 30.0))
        rule, fails, shown, last_row = level_bench.EngineAdmission(**spec), 0, None, 0.0
        self.mg_admission_counts = {}
        async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
            while True:
                await asyncio.sleep(interval)
                pool = level_bench.SHARED_SLOTS
                try:
                    sample = level_bench.engine_sample((await client.get(self.server.root + "/metrics")).text)
                    fails = 0
                except Exception:
                    sample, fails = None, fails + 1
                why = rule.update(time.time(), sample) if sample is not None else (None if fails >= 3 else
                                                                                   getattr(pool, "paused_why", None))
                if pool is not None:
                    pool.paused_why = why
                    if sample is not None and pool.project is not None:  # mg.kv_project: measured scale, logged
                        pool.scale_sample(sample.get("kv"))
                        moved = pool.calibrate(sample.get("kv"))  # kv_project.auto_scale
                        if moved is not None:
                            self.feed.emit("kv_scale", engine_scale=moved, samples=len(pool.calib))
                            pool2 = getattr(level_bench, "SHARED_SLOTS2", None)  # v6j3: server 2's pool never
                            if pool2 is not None and pool2.project is not None \
                                    and not pool2.project.get("auto_off"):  # calibrated (sat at 1.0); same model
                                pool2.project["engine_scale"] = moved  # and adapters: server 1's scale
                if pool is not None and sample is not None and self.cfg["mg"].get("turn_auto") \
                        and getattr(self, "turn_auto_live", True):  # live knob turn_auto
                    moved = level_bench.turn_auto_step(pool, sample.get("kv"), self.cfg["mg"]["turn_auto"])
                    if moved is not None:
                        self.feed.emit("turn_cap", server=1, turn_cap=moved, kv=sample.get("kv"), live=len(pool.live))
                if pool is not None and time.time() - getattr(self, "mg_pool_row_at", 0.0) >= 60.0:
                    self.mg_pool_row_at = time.time()  # v6j3: no live record of what held admission
                    self.feed.emit("pool", **{name: level_bench.pool_state(p) for name, p in
                                              (("s1", pool), ("s2", getattr(level_bench, "SHARED_SLOTS2", None)))
                                              if p is not None}, kv=(sample or {}).get("kv"))
                if why:
                    self.mg_admission_counts[why] = self.mg_admission_counts.get(why, 0) + 1
                if (why != shown and time.time() - last_row >= report) or (sample is not None and not last_row):
                    extra = {}
                    if pool is not None and pool.project is not None and pool.capacity:
                        scales = sorted(pool.scale_samples[-30:])
                        extra = {"projected": round(pool.load() / max(1, pool.capacity_tokens()), 3),
                                 "project_peak": round(pool.project_peak, 3), "kv_waits": pool.kv_waits,
                                 "scale_p50": scales[len(scales) // 2] if scales else None, "live": len(pool.live),
                                 "scale_all_p50": (lambda a: a[len(a) // 2] if a else None)(sorted(getattr(pool, "scale_all", [])[-30:]))}
                    self.feed.emit("engine_admission", paused=why, kv=(sample or {}).get("kv"),
                                   waits=getattr(pool, "engine_waits", None), **extra)
                    shown, last_row = why, time.time()

    def ts_lazy(self):
        """mg.timeshare.lazy_boot: the trainer loads first, at the loop's start;
        the second server boots at the first time share, beside the parked trainer, and stays awake for it."""
        mg = self.cfg["mg"]
        return bool((mg.get("timeshare") or {}).get("lazy_boot")) and not mg.get("single_card") and not mg.get("bench")

    async def ts_boot(self, sleep=True):
        """Time share: a second play server on the trainer's card, booted on the empty card before the
        trainer loads, then asleep; its own attempt-slot pool on the same KV-aware admission. ``sleep`` False (lazy
        boot): booted beside the parked trainer for an immediate time share, left awake."""
        from . import level_bench
        from .cache import BlockBudget
        from .rl_loop import SecondServer

        ts = self.cfg["mg"]["timeshare"]
        self.server2 = SecondServer(self.server_dir, self.server.model, card=self.cfg["trainer_device"],
                                    port=int(ts.get("port", 1235)), gpu_util=ts.get("gpu_util", 0.88),
                                    kv_scale=ts.get("kv_scale"), env=ts.get("env"), keep_lora=ts.get("keep_lora", False))
        boot = await self.server2.boot()
        if ts.get("env"):
            boot["env"] = dict(ts["env"])
        pool2 = level_bench.SharedSlots(int(ts.get("active_tries") or self.cfg["mg"]["active_tries"]), self.cfg["mg"].get("active_kv_fill"),
                                    project=self.cfg["mg"].get("kv_project"),
                                    stagger_s=self.cfg["mg"].get("admit_stagger_s") or 0.0)
        if level_bench.SHARED_SLOTS is not None:  # one admission clock per bench across both servers (hero-v3 read:
            pool2.last_take = level_bench.SHARED_SLOTS.last_take  # card 2 took every try card 1's stagger refused)
        if self.study.get("cache_admission") == "hybrid_blocks":
            budget = BlockBudget.load(self.server2.folder, float(self.study["kv_margin"]))
            pool2.capacity, pool2.budget = budget.capacity, budget
            boot["capacity_blocks"] = budget.capacity
        pool2.active = False
        pool2.turn_cap = int(ts.get("turn_cap") or 0) or None  # card2_turn_cap's boot value (off by default)
        level_bench.SHARED_SLOTS2, level_bench.SERVER2_URL = pool2, self.server2.root + "/v1"
        if not sleep and ts.get("boot_cycle"):
            # MFU-2 (24 Oct): vLLM's first level-1 sleep pins fresh host memory per allocation (~30 s RTX, ~59 s B300);
            # later sleeps reuse the pinned blocks (scrtx-c-v1: 29.8 / 1.2 / 1.2 s). One sleep / wake right after the
            # lazy boot takes that cost here instead of at the first round end, where the parked trainer waits on it.
            boot["boot_cycle"] = {"sleep_s": await self.server2.sleep(), "wake_s": await self.server2.wake()}
        if sleep:
            boot["sleep_seconds"] = await self.server2.sleep()
        boot.update(self.server2.memory, lazy=not sleep)
        log = Path(self.server2.folder) / "vllm.log"
        if log.is_file():
            boot["phases"] = boot_phases(log.read_text(errors="replace"))
        self.feed.emit("timeshare_boot", **boot)
        return boot

    def serving_log_mirror(self, every_s=60.0, tail_bytes=8 << 20):
        """Both servers' vLLM logs to <ARC_VALIDATION_OUTPUT>/serving-logs (last ``tail_bytes`` each, every ``every_s``;
        the launcher's committer commits them). v6h/v6i: server 2 died twice with no log off the container."""
        out = os.environ.get("ARC_VALIDATION_OUTPUT")
        if not out or getattr(self, "_log_mirror", None) is not None:
            return None
        import threading

        dest, stop = Path(out) / "serving-logs", threading.Event()
        work = Path(os.environ.get("TAAF_KAGGLE_WORKING_DIR") or "/nonexistent")

        def logs():
            two = getattr(getattr(self, "server2", None), "folder", None)
            return {"server1.log": work / "vllm-openai-server.log",
                    **({"server2.log": Path(two) / "vllm.log"} if two is not None else {})}

        def run():
            while not stop.wait(every_s):
                for name, src in logs().items():
                    try:
                        size = src.stat().st_size
                        with open(src, "rb") as f:
                            f.seek(max(0, size - tail_bytes))
                            data = f.read()
                        dest.mkdir(parents=True, exist_ok=True)
                        tmp = dest / (name + ".tmp")
                        tmp.write_bytes(data)
                        tmp.replace(dest / name)
                    except OSError:
                        continue

        self._log_mirror = threading.Thread(target=run, daemon=True, name="serving-log-mirror")
        self._log_mirror.stop = stop
        self._log_mirror.start()
        return dest

    async def ts_park_check(self, probes):
        """The full park (base included) must give back the same weights: probe log-probabilities identical before
        and after two park cycles; otherwise the run goes on without the time share."""
        from . import level_bench

        try:
            check = await self.trainer.call("park_check", timeout=1800, windows=probes[:2], cycles=2)
        except Exception as error:
            check = {"ok": False, "error": repr(error)[:500]}
        self.feed.emit("park_check", **{k: v for k, v in check.items() if k != "traceback"})
        if not (check.get("ok", True) and check.get("identical")):
            self.ts_dead = True
            if level_bench.SHARED_SLOTS2 is not None:
                level_bench.SHARED_SLOTS2.active = False
            if getattr(self, "server2", None) is not None:
                self.server2.stop()
            self.feed.emit("alert", why="park check failed; time share off", check={k: check.get(k) for k in ("identical", "max_abs_diff", "error")})
        return check

    async def ts_on(self, number, tasks, adapter):
        """After the round's update: park the trainer (base included), wake the second server with the served adapter
        and open its pool, while enough of the round's tries are still queued."""
        from . import level_bench
        from .rl_loop import card_memory_of

        ts, pool, pool2 = self.cfg["mg"]["timeshare"], level_bench.SHARED_SLOTS, level_bench.SHARED_SLOTS2
        queued = sum(max(0, c) for _, c in pool.pending.values()) if pool is not None else 0
        info = {"round": number, "queued": queued}
        if all(t.done() for t in tasks) or queued < int(ts.get("min_queued", 4)):
            self.feed.emit("timeshare_skip", why="little play left", **info)
            return info
        info["park"] = await self.trainer.must("park", base=bool(ts.get("park_base", True)))
        booted = getattr(self, "server2", None) is not None
        used, total = card_memory_of(self.server2.card if booted else self.cfg["trainer_device"])
        if booted:
            need = (self.server2.memory.get("awake_gib") or 0) - (self.server2.memory.get("asleep_gib") or 0) + float(ts.get("margin_gib", 1.0))
        else:  # lazy boot: vLLM takes gpu_util of the whole card at boot
            need = (float(ts.get("gpu_util", 0.88)) * total if total else 0.0) + float(ts.get("margin_gib", 1.0))
        info.update(free_gib=None if used is None else round(total - used, 2), need_gib=round(need, 2))
        if used is not None and total - used < need:
            self.feed.emit("timeshare_skip", why="no room beside the parked trainer", **info)
            return info
        if not booted:  # raises on a failed boot: the caller turns the time share off and the round goes on
            info["boot_seconds"] = (await self.ts_boot(sleep=False)).get("boot_seconds")
            pool2 = level_bench.SHARED_SLOTS2  # the pool ts_boot made
        info["wake_seconds"] = await self.server2.wake()
        info["load_seconds"] = await self.server2.serve(self.mg_name(), adapter) if adapter else 0.0
        pool2.active, pool2.served = True, 0
        self.ts_began = time.time()
        self.feed.emit("timeshare_on", **info)
        return info

    async def ts_off(self):
        from . import level_bench

        pool2 = level_bench.SHARED_SLOTS2
        if pool2 is None or getattr(self, "server2", None) is None:  # lazy boot: not booted yet
            return {"served": 0, "booted": False}
        out = {"served": pool2.served, "peak": pool2.peak, "kv_waits": pool2.kv_waits, "kv_peak_load": pool2.peak_load}
        pool2.active, pool2.peak = False, 0
        if getattr(self, "ts_began", None):
            out["on_seconds"] = round(time.time() - self.ts_began, 1)
            self.ts_began = None
        try:
            out["sleep_seconds"] = await self.server2.sleep()
        except Exception as error:  # the trainer needs its card back: stop the second server outright
            self.ts_dead = True
            self.server2.stop()
            self.feed.emit("alert", why="second server did not sleep; stopped, time share off", error=repr(error)[:1500])
            out["error"] = repr(error)[:300]
        return out

    # ---- S2: resume and startup time (early play, resume skips, trainer load beside server 2's boot) ----

    async def mg_overlap_boot(self, boot):
        """mg.trainer_overlap: the trainer process starts with server 2's boot. Both use the trainer's card (server 2
        awake 214 GiB + trainer load 92 GiB > 268 GiB), so the trainer's first CUDA use waits for a gate file written
        once server 2 sleeps or is stopped: only its CPU part (imports, processor) overlaps the boot."""
        gate = self.work / "trainer-gpu-gate"
        gate.unlink(missing_ok=True)
        began = time.time()
        launch = asyncio.ensure_future(self.start_trainer(gate=gate))
        try:
            await boot()
        finally:
            gate.parent.mkdir(parents=True, exist_ok=True)
            gate.write_text(json.dumps({"epoch": time.time(), "server2_dead": bool(getattr(self, "ts_dead", False))}))
        boot_s = round(time.time() - began, 1)
        trainer = await launch
        trainer["overlap"] = {"boot_s": boot_s, "total_s": round(time.time() - began, 1)}
        return trainer

    def mg_rollback_pending(self, number, state):
        """A rollback knob mg_reconfigure would apply at round ``number``'s top (it needs the trainer)."""
        path = os.environ.get("ARENA_KNOBS")
        if not path or not Path(path).is_file():
            return False
        try:
            knobs = json.loads(Path(path).read_text())
        except ValueError:
            return False
        if "rollback" not in knobs or (state.get("knobs_seen") or {}).get("rollback") == knobs["rollback"]:
            return False
        spec = knobs["rollback"]
        return not (isinstance(spec, dict) and int(spec.get("after_round") or 0) >= number)

    async def mg_early_serve(self, state, ckpt):
        """mg.early_play: serve the adapter saved beside the restored state's checkpoint (save_checkpoint) on the play
        server before the trainer loads, so round `first` plays at once. Its tries train only after the state check
        and an identity check (the trainer's own save of the loaded state must equal it byte for byte) pass; a failed
        gate discards them (mg_early_discard). Returns the served path, or None (today's order) with the reason."""
        from .rl_loop import adapter_tensors

        mg, why = self.cfg["mg"], None
        src = Path(ckpt) / "adapter"
        if mg.get("single_card"):
            why = "single_card"
        elif mg.get("bench") is not None:
            why = "bench"
        elif mg.get("carry") and int(state.get("played_by", state["round"])) == int(state["round"]) - 1:
            why = "carry (the first update runs before play)"
        elif self.mg_rollback_pending(int(state["round"]) + 1, state):
            why = "rollback knob pending"
        elif mg.get("served_route") and not (self.study.get("server") or {}).get("served_route"):
            why = "served_route without a capture directory"
        else:
            try:
                adapter_tensors(src)
            except ValueError as exc:
                why = f"no served adapter beside the checkpoint: {exc}"[:400]
        if why:
            self.feed.emit("early_play", on=False, why=why)
            return None
        dest = self.work / "adapters" / f"resume-r{int(state['round']):02d}"
        shutil.rmtree(dest, ignore_errors=True)
        await asyncio.to_thread(shutil.copytree, src, dest)
        load = await self.mg_serve(str(dest), int(state.get("played_by", state["round"])))
        self.serving_adapter = str(dest)
        self.policy_uid = state_uid(state)
        if mg.get("served_route"):  # the round top's capture marks (start() sets it again)
            self.served_dir = (self.study.get("server") or {}).get("served_route")
        self.mg_early_verified = False
        self.feed.emit("early_play", on=True, round=int(state["round"]) + 1, source=str(src), load_seconds=load)
        return str(dest)

    async def mg_resume_trainer(self, state, ckpt, probes, played_by, early=None):
        """A resume's trainer path: load the state, serve its adapter (unless early play serves it already), score the
        probes, the state check; then (mg.resume_skip) the park check and this run's own state copy."""
        mg = self.cfg["mg"]
        loaded = await self.trainer.must("load_checkpoint", path=ckpt)
        serve = str(self.work / "adapters" / f"resume-r{state['round']:02d}{'-check' if early else ''}")
        await self.trainer.must("save", path=serve)
        if early is None:
            if not mg.get("single_card"):
                await self.mg_serve(serve, played_by)
            self.serving_adapter = serve
            self.policy_uid = state_uid(state)
        target_kl = mg.get("kl_control") == "target"
        scored = await self.trainer.must("score", windows=probes, label="resume", hold_previous=target_kl)
        if target_kl:  # KL-to-previous reference = the restored state
            await self.trainer.must("accept_reference")
        check = self.mg_state_check("resume", self.mg_recorded_probes(mg["resume"], played_by), scored)
        if early is not None:
            await self.mg_early_same(early, serve)
        verified = check.get("ok") is True
        if mg.get("resume_skip"):
            await self.mg_resume_park(probes, verified)
        if mg.get("rollback") and mg.get("checkpoint") and os.environ.get("ARC_VALIDATION_OUTPUT"):
            self.keep_checkpoint = f"r{played_by:02d}"
            self.mg_served = [(self.keep_checkpoint, played_by)]  # earlier states stay with the earlier run
            own = Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "state" / self.keep_checkpoint
            if mg.get("resume_skip") and verified and Path(ckpt).resolve() != own.resolve():
                # the verified state's files as this run's rollback target, beside play (no trainer re-checkpoint)
                self.mg_own_copy = asyncio.ensure_future(self.mg_copy_state(ckpt, own, state, played_by))
                return loaded
            self.checkpoint(played_by, force=True)  # this run's own copy as the rollback target
            await self.save_checkpoint()
            if own.is_dir() and not (own / "mg_state.json").exists():  # its snapshot = the state as loaded (a
                # snapshot written at the round's end would carry the next update's counters and lr)
                (own / "mg_state.json").write_text(json.dumps({**state, "checkpoint": str(own),
                                                               "policy_uid": self.policy_uid}))
        return loaded

    async def mg_early_same(self, early, serve):
        """Early play's identity gate: the trainer's save of the loaded state equals the adapter played."""
        import filecmp

        def config(folder):
            row = json.loads((Path(folder) / "adapter_config.json").read_text())
            return {k: v for k, v in row.items() if k not in ("base_model_name_or_path", "revision", "peft_version")}

        tensors = await asyncio.to_thread(filecmp.cmp, Path(early) / "adapter_model.safetensors",
                                          Path(serve) / "adapter_model.safetensors", False)
        row = {"tensors_same": tensors, "config_same": config(early) == config(serve)}
        row["ok"] = row["tensors_same"] and row["config_same"]
        if (not row["ok"] and os.environ.get("ARC_PREFLIGHT_FAKE_TRAINER")
                and not any(m in str(serve) for m in ("/validation/preflight/", "/tmp/validation/"))):
            # a CPU leg resuming a real run's state (pfK1, r4's exact start): the stand-in trainer cannot save it
            row.update(ok=True, skipped="stand-in trainer on a real state")
        self.feed.emit("early_play_check", **row)
        if not row["ok"]:
            raise RuntimeError(f"early play: the played adapter differs from the restored state's own save {row}")
        return row

    @staticmethod
    def mg_source_park(source):
        """The resumed run's last passed park check (identical, or skipped on a record / an earlier resume), or None."""
        feed = Path(str(source)) / "arena" / "trials" / "rl-loop" / "rl" / "feed.jsonl"
        if not feed.is_file():
            return None
        found = None
        for line in feed.read_text().splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("kind") == "park_check":
                found = e if (e.get("identical") is True or e.get("skipped") is True) else None
        return found

    async def mg_resume_park(self, probes, verified):
        """mg.resume_skip: the park check start() deferred. Skipped when the restored state passed its check and the
        resumed run passed a park check; otherwise run as start() would have."""
        missing = object()
        passed = self.__dict__.pop("mg_park_deferred", missing)
        if passed is missing:
            return None
        if passed is not None and passed.is_file():  # v5 boot cut: same trainer key passed before
            self.feed.emit("park_check", skipped=True, record=str(passed))
            return None
        source = self.mg_source_park(self.cfg["mg"]["resume"]) if verified else None
        if source is not None:
            self.feed.emit("park_check", skipped=True, why="resume_skip", source=self.cfg["mg"]["resume"],
                           source_epoch=source.get("epoch"))
            return None
        check = await self.ts_park_check(probes)
        if check.get("identical") and passed is not None:
            passed.parent.mkdir(parents=True, exist_ok=True)
            passed.write_text(json.dumps({"epoch": time.time(), "run": os.environ.get("ARC_VALIDATION_OUTPUT")}))
        return check

    async def mg_copy_state(self, src, own, state, played_by):
        """mg.resume_skip: the restored checkpoint's files (its saved adapter included, its mg_state snapshot not) as
        this run's state r<played_by>; on a failed copy the trainer writes it as before."""
        began, row = time.time(), {"round": played_by, "source": str(src)}

        def copy():
            tmp = own.with_name(own.name + ".copy")
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("mg_state.json"))
            shutil.rmtree(own, ignore_errors=True)
            tmp.rename(own)
            return round(sum(f.stat().st_size for f in own.rglob("*") if f.is_file()) / 2**30, 2)

        try:
            row["gib"] = await asyncio.to_thread(copy)
        except Exception as error:  # noqa: BLE001 - the trainer's own checkpoint instead
            row["error"] = repr(error)[:300]
            shutil.rmtree(own, ignore_errors=True)
            row["fallback"] = await self.trainer.must("checkpoint", path=str(own))
        if own.is_dir() and not (own / "mg_state.json").exists():
            (own / "mg_state.json").write_text(json.dumps({**state, "checkpoint": str(own), "policy_uid": self.policy_uid}))
        row["seconds"] = round(time.time() - began, 1)
        self.feed.emit("checkpoint_copy", **row)
        return row

    async def mg_early_discard(self, tasks, number, error):
        """A failed early-play gate: the round's play stops, its finished-try keeps leave the reuse path and its bench
        folders are marked unusable (none of them train)."""
        from . import level_bench as lb

        live = [t for t in tasks if t is not None]
        for t in live:
            if not t.done():
                t.cancel()
        if live:
            try:
                await asyncio.wait(live, timeout=300)
            except BaseException:  # noqa: BLE001 - the discard still runs
                pass
        row = {"round": number, "why": repr(error)[:500], "keep": None, "benches": []}
        keep, lb.TRY_KEEP = lb.TRY_KEEP, None
        if keep is not None and Path(keep).exists():
            dest = Path(keep).with_name(f"{Path(keep).name}.unusable-{int(time.time())}")
            try:
                Path(keep).rename(dest)
                row["keep"] = str(dest)
            except OSError:
                shutil.rmtree(keep, ignore_errors=True)
                row["keep"] = "removed"
        for folder in sorted((self.output / "rl").glob(f"r{number:02d}-*")):
            if folder.is_dir():
                (folder / "UNUSABLE.json").write_text(json.dumps({"why": row["why"], "epoch": time.time()}))
                row["benches"].append(folder.name)
        (self.output / "rl" / f"r{number:02d}-early-unusable.json").write_text(json.dumps(row, indent=1))
        self.feed.emit("early_play_discard", **row)
        return row

    async def mg_loop(self):
        """l4's schedule: rounds 1-2 played by the base (round 1 beside the trainer's load), the update of round r on
        round r-1's selection, the adapter swapped after it; tests from round 3 (a trend; the decision final is a
        separate job); the last round plays tests only. Pool: the screen's starts with a clear."""
        mg = self.cfg["mg"]
        marks, t_setup = {}, time.time()  # start-up cut (f) profile (MFU-2 23 Oct): ovf2 spent 74 s here, card idle

        def mark(name):
            marks[name] = round(time.time() - t_setup, 1)

        stub = os.environ.get("ARC_PREFLIGHT_STUB") or ""
        if stub.startswith("{") and json.loads(stub).get("shrink") and not mg.get("lr_probe"):
            # fast CPU preflight: every path of the round at a fraction of its tries
            self.shrink_mg(mg, minimum=json.loads(stub).get("shrink") == 2)
            # held-out clear rates on 4 replayed tries per set are noise (pf7 leg 2 stopped on a 0.33 "drop"): the
            # clear-drop guards still compute but cannot trip, so the preflight reaches its rollback and handoff
            for key in ("breakage", "behaviour_guard"):
                if isinstance(mg.get(key), dict) and mg[key].get("clear_drop") is not None:
                    mg[key]["clear_drop"] = 1.01
            self.feed.emit("preflight_shrink", train_starts=mg["train_starts"], train_tries=4, fresh_tries=mg["fresh_tries"],
                           fixed_tries=mg["fixed_tries"],
                           clear_drop_guards="off")
        if stub.startswith("{") and json.loads(stub).get("branch_from_tar"):  # preflight: a resume rebuilds every cut
            os.environ["ARC_BRANCH_FORCE_TAR"] = "1"  # from the source's round archive (the snapshots are ignored)
        if mg.get("train_summary"):  # envs: compaction summaries are trained replies (whole-try
            # advantage) and restores replay across them; the trainer subprocess inherits the switch
            if self.study.get("compaction_keep_fraction") is None:
                raise ValueError("mg.train_summary needs the study's compaction_keep_fraction (a summary compaction)")
            os.environ["ARC_RL1_TRAIN_SUMMARY"] = "1"
        if os.environ.get("ARC_RL1_MAX_ROUNDS"):  # launch-time ceiling: rounds 1..N, updates through N-1
            mg["max_rounds"] = int(os.environ["ARC_RL1_MAX_ROUNDS"])
        if mg.get("max_rounds"):  # the round loop runs to max_rounds (``updates`` + 1 must not end it sooner)
            mg["updates"] = max(int(mg.get("updates", 8)), int(mg["max_rounds"]) - 1)
        for root in [x for x in os.environ.get("ARC_RL1_RESUME", "").split(",") if x][::-1]:
            if not mg.get("resume") and (Path(root) / "mg_state.json").is_file():  # gate item 4: the launcher's restart
                mg["resume"] = root  # the latest run dir with a round state (its checkpoint is the served state)
        self.loop_lag = {"tick": 0.0, "round": 0.0}
        self.mg_lag_task = asyncio.ensure_future(self.mg_loop_lag())
        if self.ts_lazy() and not mg.get("lr_probe"):  # the trainer's load starts now, beside the pool and pending setup
            self.mg_trainer_boot = asyncio.ensure_future(self.start_trainer())
            await asyncio.sleep(0)  # the task spawns the trainer process now, not at the setup's first await (r4)
        mark("entry")
        screen = self.mg_screen = self.mg_screen_file()
        mark("screen_file")
        sets = {s["name"]: s for s in mg["sets"]}
        self.cfg["mg_sets"] = sets
        # warm start: round 1 is played by mg.init_state's adapter, a policy round from the
        # start; the base reference (held-out rows, play health, tokens per request) comes from mg.base_ref, the cold
        # arm's run on the same starts and seeds, under round keys 0, -1, .. (this run's rounds keep 1, 2, ..)
        self.mg_init = bool(mg.get("init_state"))
        if self.mg_init and mg.get("seed_u1"):
            raise ValueError("mg.init_state with seed_u1 is not supported")
        if mg.get("base_ref") and not mg.get("resume"):
            ref = json.loads((Path(mg["base_ref"]) / "mg_state.json").read_text())
            keys = {int(r): -i for i, r in enumerate(ref["base_rounds"])}
            self.mg_fixed = {keys[int(k)]: v for k, v in (ref.get("fixed") or {}).items() if int(k) in keys}
            self.mg_base_rounds = sorted(keys.values(), reverse=True)
            self.mg_base_play, self.mg_base_tpr = ref.get("base_play"), ref.get("base_tpr")
            if not self.mg_fixed or self.mg_base_play is None:
                raise ValueError(f"mg.base_ref {mg['base_ref']}: no base rounds' held-out rows or play health")
            self.feed.emit("base_ref", source=mg["base_ref"], base_rounds=ref["base_rounds"], as_rounds=self.mg_base_rounds,
                           base_play=self.mg_base_play, base_tpr=self.mg_base_tpr,
                           fixed={n: len(rows) for v in self.mg_fixed.values() for n, rows in v.items()})
        elif self.mg_init and not mg.get("resume"):
            raise ValueError("mg.init_state needs mg.base_ref (the base reference this run cannot play)")
        if mg.get("eval"):
            return await self.mg_eval()
        pool, self.mg_round_starts = {}, {}
        for name, s in sets.items():
            if s["role"] == "train" and not s.get("fresh"):
                if name not in screen["sets"]:  # multi-game (envs 22 Oct): no bank; the in-run harvest feeds it
                    pool[name] = []
                    continue
                rows = self.mg_rows(screen["sets"][name]["folder"])
                good = {r["point"] for r in rows if clear_tokens(r)}
                pool[name] = self.mg_matched(s, screen["sets"][name], good)
        mark("pool_rows")
        tests = [n for n, s in sets.items() if s["role"] == "test"]
        if mg.get("prescreen"):  # replay every start first; drop any whose game state diverges
            bad = await self.mg_prescreen({n: pool[n] for n in pool}, {n: self.mg_ok_starts(n, screen) for n in tests
                                                                        if not sets[n].get("fresh")}, screen)
            if bad:
                mg["exclude_starts"] = sorted(set(mg.get("exclude_starts") or []) | set(bad))
                for name in pool:
                    rows = self.mg_rows(screen["sets"][name]["folder"])
                    pool[name] = self.mg_matched(sets[name], screen["sets"][name], {r["point"] for r in rows if clear_tokens(r)})
        mark("prescreen")
        # one held start per restored training level, played every round, not trained (a trend
        # free of the rotation)
        if mg.get("half_split"):  # sc25 (19 Oct): train on the recordings' training half (start_half on the set), read
            # every round on the base reference's own held-out starts (its tries there are the base comparison)
            fresh, finfo = self.mg_fresh_starts(10 ** 6)
            k = int(mg.get("held_starts") or 2)
            fixed = {n: [tuple(x) for x in screen["sets"][n]["starts"][:k]] for n in pool}
            pool = {n: [tuple(x) for x in fresh.get(n, [])] for n in pool}
            self.feed.emit("mg_fresh", sets=finfo, pool={n: len(v) for n, v in pool.items()})
        elif mg.get("held_starts"):  # mg6: held-out = several starts no screen played (never trained); the rest of the
            # unplayed starts join the training rotation, so starts do not repeat across rounds
            fresh, finfo = self.mg_fresh_starts(10 ** 6)
            k = int(mg["held_starts"])
            fixed = {n: list(fresh.get(n, [])[:k]) for n in pool if len(fresh.get(n, [])) >= k}
            for n in pool:
                pool[n] = list(pool[n]) + [tuple(x) for x in fresh.get(n, [])[k:]] if n in fixed else pool[n]
            self.feed.emit("mg_fresh", sets=finfo, pool={n: len(v) for n, v in pool.items()})
        else:
            fixed = {n: pool[n].pop() for n in list(pool) if mg.get("fixed_tries") and len(pool[n]) > 2}
        for n, s in sets.items():  # overfit test: a set's pool cut to its first pool_max starts (with
            # train_starts = pool_max every round trains the same starts)
            if s.get("pool_max") and n in pool:
                pool[n] = list(pool[n])[:int(s["pool_max"])]
        if mg.get("fixed_fresh"):  # mg5: fresh training levels get held-out tries too (own seeds, not trained)
            fixed.update({n: None for n, s in sets.items() if s["role"] == "train" and s.get("fresh")})
        if mg.get("fixed_sets") is not None:  # the held-out read on these sets only (v6m10-r2: L1 + L3)
            fixed = {n: v for n, v in fixed.items() if n in set(mg["fixed_sets"])}
        # multi-game (envs 22 Oct): test sets marked ``read`` are fresh held-out reads (the held-out 10's level 1),
        # played every read round by the served model (the base in base rounds) at seed mg.seed + 500, never trained
        fixed.update({n: None for n, s in sets.items() if s["role"] == "test" and s.get("read") is True})
        # explorer design: ``read: "screen"`` reads a held-out level from the screen's recorded starts
        # (real base plays, no jumps), the same starts every read; a level no recording reached is not read
        for n, s in sets.items():
            if s["role"] == "test" and s.get("read") == "screen":
                got = [tuple(x) for x in ((screen["sets"].get(n) or {}).get("starts") or [])][:int(s.get("starts") or 2)]
                if got:
                    fixed[n] = got
                else:
                    self.feed.emit("mg_read_unreached", name=n)
        mark("fresh_starts")
        tests = [n for n, s in sets.items() if s["role"] == "test"]
        self.mg_unseen_read(screen)
        ptests = [n for n in tests if n in screen["sets"]] or tests  # probe windows need a screen-recorded test set
        lvl_rows = self.mg_rows(screen["sets"][ptests[0]]["folder"])[:2]
        probes = self.windows(screen["sets"][ptests[0]]["folder"], lvl_rows, prefix="heldout-")
        for n in ptests[1:int(mg.get("probe_sets") or 1)]:  # mg5: one probe window from each further test set
            probes += self.windows(screen["sets"][n]["folder"], self.mg_rows(screen["sets"][n]["folder"])[:1],
                                   prefix=f"heldout{len(probes)}-")
        for n in [n for n in ptests if sets[n].get("untrained_game")][:2]:
            probes += self.windows(screen["sets"][n]["folder"], self.mg_rows(screen["sets"][n]["folder"])[:1],
                                   prefix="untrained-")
        self.mg_probes = probes
        mark("probes")
        since = getattr(self, "started", None)
        self.feed.emit("mg_pool", pool={n: len(v) for n, v in pool.items()}, probes=[p["name"] for p in probes],
                       setup_seconds=marks, loop_start_to_setup=round(t_setup - since, 1) if since else None,
                       fixed={n: list(v) if v else "fresh" for n, v in fixed.items()})

        from . import branch as branch_mod
        from . import harvest as harvest_mod

        if mg.get("branch") and mg.get("curriculum"):
            raise ValueError("mg.branch with mg.curriculum is not supported")
        if mg.get("curriculum"):  # envs: the round's groups go to the training sets by the curriculum
            self.mg_curr = self.mg_curriculum_init(sets, pool, screen)
            if mg.get("seed_starts") and not mg.get("resume"):  # envs 22 Oct: the seed round's recorded level
                # starts (base plays) open the start pool; a resume restores the pool from its state instead
                scan = int(mg.get("seed_scan") or os.environ.get("ARC_RL1_SEED_SCAN") or 0)  # RL start-up: parallel scan
                if mg["seed_starts"] == "screen":
                    rows = (await asyncio.to_thread(self.mg_screen_starts, screen, sets, scan) if scan
                            else self.mg_screen_starts(screen, sets))
                else:
                    rows = json.loads((self.source / mg["seed_starts"]).read_text())["starts"]
                rows = [r for r in rows if r.get("target") in sets]
                self.__dict__.setdefault("mg_harvest", []).extend(rows)
                for r in rows:
                    self.mg_curr.add_start(r, create=False)
                self.feed.emit("mg_seed_starts", starts=len(rows), sets=len({r["target"] for r in rows}))

        if mg.get("async"):  # v7 (rl-plan v7): no rounds from here on; arena.rl_async
            return await self.mg_async_run(sets=sets, pool=pool, fixed=fixed, tests=tests, probes=probes, screen=screen)

        def train_jobs(number, model, part="all", own=None, picks=None):
            """The round's training plays. mg.lookahead_below: ``part`` "normal" (a look-ahead round's picked sets,
            launched early; ``picks`` collects their (starts, tries)) or "branch" (its branch groups, launched at the
            round's own start: they need last round's cuts); ``own``: the benches' own options (mg_play)."""
            jobs, names = [], []
            if getattr(self, "mg_curr", None) is not None:
                return self.mg_curriculum_jobs(number, model, sets, pool)
            again = getattr(self, "mg_resume_ahead", None)  # a resumed look-ahead round: its recorded picks and roots
            again = again if again and int(again["round"]) == number and part != "branch" else None
            for name, s in list(sets.items()):
                if s["role"] != "train" or s.get("branch_of"):
                    continue
                cuts = branch_mod.pick(self, number, name) if part != "normal" else None
                if cuts:  # branch groups: the round's continuations from last round's cuts (group tries each)
                    bname = f"{name}-br"
                    sets[bname] = {**s, "name": bname, "fresh": False, "branch_of": name}
                    jobs.append(self.mg_play(f"r{number:02d}-{bname}", sets[bname], cuts,
                                             int(branch_mod.settings(mg)["group"]), model=model, seed=None,
                                             sampling=mg.get("train_sampling"), own=own))
                    names.append(bname)
                if part == "branch":
                    continue
                bench_own = own
                if again is not None and name in again["picks"]:  # the same starts; finished tries come back from
                    st, tries = again["picks"][name]  # the look-ahead's versioned root, the rest are played again
                    found = [(a, int(r)) for a, r in ((self.mg_ahead_start(x[0]), x[1]) for x in st) if a]
                    if len(found) < len(st):  # pfB (22 Oct): a packed leg's snapshots live in its unpacked copy
                        self.feed.emit("lookahead_resume_missing", round=number, set=name, missing=len(st) - len(found))
                    picked = (found, int(tries)) if found or not st else None
                    bench_own = {**(own or {}), "try_keep": self.mg_ahead_roots(again, resumed=True)}
                else:
                    picked = self.mg_pick(number, name, s, pool)
                if picked is None:
                    continue
                st, tries = picked
                if s.get("fresh") or st:
                    self.mg_round_starts.setdefault(number, {})[name] = st
                    if picks is not None:
                        picks[name] = [[list(x) for x in st], int(tries)]
                    jobs.append(self.mg_play(f"r{number:02d}-{name}", s, st, tries, model=model, seed=None,
                                             sampling=mg.get("train_sampling"),
                                             harvest=harvest_mod.wants(mg, sets, s), own=bench_own))
                    names.append(name)
            if again is not None:
                self.mg_resume_ahead = None
                self.feed.emit("lookahead_resume", round=number, version=again["version"], sets=sorted(again["picks"]))
            return jobs, names

        bench = mg.get("bench")  # serving-path bench: round 1 play only, no trainer

        async def path_check(zero):
            """Server log-probabilities of two fixed texts: base twice (repeat noise), then a zero-B adapter."""
            from .rl_loop import make_text

            seqs = [(await self.server.tokens(make_text(i, 9000)))[:4096] for i in (1, 2)]
            base = [await self.server.prompt_logprobs(self.server.model, x) for x in seqs]
            again = [await self.server.prompt_logprobs(self.server.model, x) for x in seqs]
            await self.server.load("zero", zero)
            pol = [await self.server.prompt_logprobs("zero", x) for x in seqs]
            await self.server.unload("zero")
            mean = lambda a, b: sum(abs(u - v) for x, y in zip(a, b) for u, v in zip(x, y)) / sum(len(x) for x in a)
            signed = sum(u - v for x, y in zip(base, pol) for u, v in zip(x, y)) / sum(len(x) for x in base)
            out = {"repeat_mean_abs": round(mean(base, again), 5), "zero_adapter_mean_abs": round(mean(base, pol), 5),
                   "zero_adapter_mean_logratio": round(signed, 5), "tokens": sum(len(x) for x in base)}
            self.feed.emit("path_check", **out)
            return out

        async def boot2():
            try:
                await self.ts_boot()
            except Exception as error:  # a speed-up only: the loop goes on with one play server
                self.ts_dead = True
                if getattr(self, "server2", None) is not None:
                    self.server2.stop()
                self.feed.emit("alert", why="second server did not boot; time share off", error=repr(error)[:1500])

        def served_setup():  # the capture directory and its watchers: no trainer needed (mg.r1_first runs it before play)
            if getattr(self, "served_watch_task", None) is not None:
                return
            self.served_dir = (self.study.get("server") or {}).get("served_route")
            if not self.served_dir:
                raise ValueError("mg.served_route needs server.served_route (the capture directory)")
            self.served_watch_task = self.__dict__["served_watch"] = asyncio.ensure_future(self.served_watch())
            self.served_persist_start()

        async def start(defer_park=False):
            if bench is not None:
                if bench.get("path_check"):
                    await path_check(bench["path_check"])
                return None
            trainer = None
            if self.ts_lazy():
                boot_task, self.mg_trainer_boot = getattr(self, "mg_trainer_boot", None), None
                trainer = await boot_task if boot_task is not None else None
            elif mg.get("timeshare") and getattr(self, "server2", None) is None:
                if mg.get("trainer_overlap"):  # S2: the trainer's CPU part runs while server 2 boots on its card
                    trainer = await self.mg_overlap_boot(boot2)
                else:
                    await boot2()
            log = Path(os.environ.get("TAAF_KAGGLE_WORKING_DIR") or "/nonexistent") / "vllm-openai-server.log"
            if log.is_file():
                self.feed.emit("server_boot", **boot_phases(log.read_text(errors="replace")))
            self.serving_log_mirror()
            if trainer is None:
                early_t = self.__dict__.pop("mg_host_trainer", None)
                if early_t is not None:  # mg.host_load: spawned at round 1's start; its card use starts now
                    gate = self.work / "trainer-gpu-gate-r1"
                    gate.write_text(json.dumps({"epoch": time.time()}))
                    trainer = await self.start_trainer(spawned=early_t)
                else:
                    layout = self.mg_layout_env()  # mg.host_load: a cold load writes the layout record
                    trainer = await self.start_trainer(**({"extra_env": layout} if layout else {}))
            host = (trainer.get("loaded") or {}).get("host_load") if isinstance(trainer.get("loaded"), dict) else None
            if host and host.get("fallback"):
                self.feed.emit("alert", why="host load fell back to the cold load", error=host.get("error"))
            self.feed.emit("trainer", **trainer)
            if mg.get("park_prepin") and mg.get("single_card"):  # start-up cut (d): the park's pinned buffers on a thread
                self.feed.emit("park_prepin", **(await self.trainer.must("prepin", base=True)))
            if mg.get("served_route"):  # R3: train on the play server's routing
                served_setup()
                self.feed.emit("served_route", **(await self.trainer.must("route", router="served")))
            if mg.get("nudge_base"):  # mg5g: the base reference on the nonzero-adapter path (rl1-flr checks)
                self.feed.emit("nudge", **(await self.trainer.must("nudge", value=float(mg.get("nudge_value", 1e-12)))))
            await self.mg_base_reference(probes)
            if (mg.get("timeshare") or {}).get("park_check", True) and mg.get("timeshare"):
                passed = park_record()
                if defer_park:  # S2 resume_skip: decided after the state check (mg_resume_park)
                    self.mg_park_deferred = passed
                elif passed is not None and passed.is_file():  # v5 boot cut: same trainer key passed before
                    self.feed.emit("park_check", skipped=True, record=str(passed))
                elif (await self.ts_park_check(probes)).get("identical") and passed is not None:
                    passed.parent.mkdir(parents=True, exist_ok=True)
                    passed.write_text(json.dumps({"epoch": time.time(), "run": os.environ.get("ARC_VALIDATION_OUTPUT")}))
            if self.mg_init and not mg.get("resume"):  # the warm start's state: KL-to-previous reference, served adapter
                loaded = await self.trainer.must("load_checkpoint", path=mg["init_state"])
                serve = str(self.work / "adapters" / "init")
                await self.trainer.must("save", path=serve)
                scored = None
                if mg.get("kl_control") == "target":
                    scored = await self.trainer.must("score", windows=probes, label="init", hold_previous=True)
                    await self.trainer.must("accept_reference")
                self.serving_adapter = serve
                self.feed.emit("init_state", source=mg["init_state"], loaded=loaded,
                               probes={k: {f: v.get(f) for f in ("kl_base", "kl_previous") if f in v}
                                       for k, v in ((scored or {}).get("windows") or {}).items()}
                               if isinstance((scored or {}).get("windows"), dict) else None)
            if mg.get("rollback") and mg.get("checkpoint") and os.environ.get("ARC_VALIDATION_OUTPUT") \
                    and not mg.get("resume"):  # first-update rollback target (a resume saves its own state instead)
                self.keep_checkpoint, self.mg_served = "r00", [("r00", 0)]
                self.checkpoint(0, force=True)
                await self.save_checkpoint()

        if (mg.get("breakage") or {}).get("filler_ratio") is not None:  # pilot 1 drift guard: count filler per reply
            from .filler import check

            os.environ["ARC_FILLER"] = "1"
            tok = Path(self.cfg["model_dir"]) / "tokenizer.json"
            info = check(tok) if tok.is_file() else {"same": None, "why": f"{tok} missing"}
            self.feed.emit("filler_classes", **info)
            if info.get("same") is False:
                raise RuntimeError(f"filler token classes differ from the model's tokenizer: {info}")
        if mg.get("active_tries"):  # mg5f: one attempt-slot pool across the round's benches, sized to the KV pool
            from . import level_bench
            level_bench.SHARED_SLOTS = level_bench.SharedSlots(int(mg["active_tries"]), mg.get("active_kv_fill"), project=mg.get("kv_project"))
            if mg.get("engine_admission") and getattr(self, "mg_admission_task", None) is None:
                self.mg_admission_task = asyncio.ensure_future(self.mg_engine_admission())
        seed = []
        if mg.get("seed_u1") and not mg.get("resume"):  # mg.seed is the play seed base
            seed, seed_info = self.mg_seed(pool, fixed, tests)
            self.feed.emit("mg_seed", trained=len(seed), sets=seed_info)

        async def start_seeded():  # U1 on the screen's base tries while the base plays round 1
            await start()
            return await self.mg_update(1, seed, probes)

        if mg.get("lr_probe"):  # P1: trainer only, no play
            return await self.mg_lr_probe(pool, fixed, tests, probes, start)
        self.serving_adapter, played_by, stop, data, last_round_s = None, 0, None, [], None
        last_play_s = None  # the last round's play part: what a read-only final round needs (P2: b lost r03's read)
        if bench is not None and bench.get("adapter"):  # play round 1 through a LoRA adapter (the policy path)
            await self.server.load("policy", bench["adapter"])
            self.serving_adapter = bench["adapter"]
        base_name = self.server.model
        if mg.get("base_adapter") and bench is None:  # base rounds through a zero-B adapter: the policy's serving path
            await self.server.load("policy", mg["base_adapter"])
            base_name = "policy"
        last = int(mg.get("updates", 8)) + 1  # rounds 1..last; updates in rounds 2..last
        first, early, pending = 1, None, None
        if mg.get("resume"):  # an earlier run's last round state + checkpoint; that round's
            # selection is lost, so the first resumed round plays the restored adapter without an update
            state = json.loads((Path(mg["resume"]) / "mg_state.json").read_text())
            if mg.get("single_card"):
                await self.server.sleep()
            ckpt = state["checkpoint"]
            if not Path(ckpt).exists() and (Path(mg["resume"]) / "state" / Path(ckpt).name).exists():
                ckpt = str(Path(mg["resume"]) / "state" / Path(ckpt).name)  # a packed preflight's unpacked copy
            played_by = int(state.get("played_by", state["round"]))  # mg5e review: the served state's round, not the last
            first = int(state["round"]) + 1
            if mg.get("resume_train_pending"):  # the pending round trains first, while round P+1 plays
                pending = self.mg_pending_round(state, sets)
                if pending is not None:
                    first = int(pending["round"]) + 1
            if mg.get("lookahead_below"):  # the served adapters' order: lag counts updates, not round numbers
                self.mg_lineage = [int(v) for v in state.get("lineage") or []]
            if mg.get("early_play"):  # S2: round `first` plays the checkpoint's saved adapter while the trainer loads
                early = await self.mg_early_serve(state, ckpt)
            if early is None:
                await start(defer_park=bool(mg.get("resume_skip")))
                loaded = await self.mg_resume_trainer(state, ckpt, probes, played_by)
            else:
                loaded = "pending: early play"
            self.mg_base_tpr, self.mg_long, self.mg_worse = state.get("base_tpr"), state.get("long", 0), state.get("worse", 0)
            self.mg_low, self.kl_trips = set(state.get("low") or []), state.get("kl_trips", 0)
            self.mg_step_rollbacks, self.mg_accepted = state.get("step_rollbacks", 0), state.get("accepted", 0)
            self.mg_steps_done = int(state.get("steps_done") or 0)
            self.mg_base_rounds, self.mg_rule_f_done = state.get("base_rounds") or [], state.get("rule_f_done", False)
            self.mg_fixed = {int(k): v for k, v in (state.get("fixed") or {}).items()}
            if state.get("read_base"):
                self.mg_read_base = state["read_base"]
            self.mg_success_streak = state.get("success_streak", 0)
            self.mg_alt = state.get("heldout_alternate")
            self.mg_behaviour_trips = state.get("behaviour_trips", 0)
            self.mg_base_play = state.get("base_play")
            self.mg_knobs_seen = dict(state.get("knobs_seen") or {})
            if mg.get("lookahead_below") and (state.get("lookahead") or {}).get("round") == first:
                self.mg_resume_ahead = state["lookahead"]  # train_jobs replays its picks, reusing finished tries
            if state.get("lr") and mg.get("dosing") != "standard":  # standard: the bundle's constant lr
                self.cfg["lr"] = float(state["lr"])
            if mg.get("kl_ramp") and state.get("kl_ramp"):  # the ramp goes on where it stood (a run from another
                self.mg_ramp_state = dict(state["kl_ramp"])  # recipe's state starts the ramp afresh)
                mg["kl_target"] = float(self.mg_ramp_state["target"])
            self.feed.emit("resume", source=mg["resume"], round=state["round"], loaded=loaded)
            if mg.get("harvest"):
                self.feed.emit("mg_harvest_resume", **harvest_mod.restore_state(self, state, mg["resume"]))
            if mg.get("branch"):
                self.feed.emit("mg_branch_resume", **branch_mod.restore_state(self, state, mg["resume"]))
            if getattr(self, "mg_curr", None) is not None and state.get("curriculum"):
                from .curriculum import DEFAULTS as CURRICULUM_DEFAULTS, Curriculum

                cur = Curriculum.load_state(state["curriculum"])
                cur.cfg = {**cur.cfg, **{k: v for k, v in mg["curriculum"].items() if k in CURRICULUM_DEFAULTS}}
                self.mg_curr = cur
                self.feed.emit("mg_curriculum_resume", units=len(cur.units), update=cur.update)
            if mg.get("explore"):  # running caps and explorer ids carry over (envs 22 Oct)
                self.mg_level_obs = state.get("level_obs") or {}
                self.mg_explore_count = int(state.get("explore_count") or 0)
                for n, c in (state.get("set_caps") or {}).items():
                    if c and n in self.cfg["mg_sets"]:
                        self.cfg["mg_sets"][n]["cap"] = int(c)
            if mg.get("lr_dose") and mg.get("kl_control") == "target":  # the source's last measured update sets lr
                self.mg_resume_dose(mg, played_by)
            # pilot continuation: the source's last round only read (its update never ran); its
            # train sets' tries make this run's first update, so max_rounds 7 from a round-4 state is 3 more updates
            if mg.get("carry") and int(state.get("played_by", state["round"])) == int(state["round"]) - 1:
                stop, served, carry_row = await self.mg_carry(int(state["round"]), sets, probes)
                if served:
                    played_by = int(state["round"])
                self.mg_write_state(int(state["round"]), played_by)
                if stop is not None:
                    self.feed.emit("final", stop=stop, eval=None, base=None)
                    return stop
        if mg.get("resume") and mg.get("single_card"):  # the restored trainer parks; the server wakes with its adapter
            self.feed.emit("card_phase", round=first - 1, phase="resume", **(await self.mg_card_play()))
            await self.server.load("policy", self.serving_adapter)
        started = False

        async def early_startup():  # S2: the trainer path of a resume, beside round `first`'s early play
            began = time.time()
            await start(defer_park=bool(mg.get("resume_skip")))
            loaded = await self.mg_resume_trainer(state, ckpt, probes, played_by, early=early)
            self.trainer.always = dict(mg.get("trainer_fast") or {})  # as mg_knobs at a round top with the trainer up
            await self.mg_trainer_fast_gate(first, probes)
            self.mg_early_verified = True
            self.feed.emit("early_play_verified", round=first, seconds=round(time.time() - began, 1), loaded=loaded)

        if self.mg_init and not mg.get("resume"):  # round 1 needs the init adapter served before its play starts
            if mg.get("single_card"):
                await self.server.sleep()
            await start()
            started = True
            if mg.get("single_card"):
                self.feed.emit("card_phase", round=0, phase="init", **(await self.mg_card_play()))
            await self.mg_serve(self.serving_adapter, 0)
        def numbers():
            n = first
            while n <= int(mg.get("updates", 8)) + 1:  # a max_rounds knob moves the ceiling between rounds
                yield n
                n += 1

        if mg.get("explore") and getattr(self, "mg_explorers", None) is None:  # envs 22 Oct: explorer plays feed the
            from .play_explorer import Explorers  # start pool from now on, beside every round

            self.mg_explorers = Explorers(self).start()
        for number in numbers():
            began = asyncio.get_event_loop().time()
            self.mg_round_now = number
            self.mg_knobs(number)
            back = await self.mg_reconfigure(number, played_by, probes)
            early_round = early is not None and number == first
            if not early_round:  # an early round's startup gates the knob once the trainer is up
                await self.mg_trainer_fast_gate(number, probes)
            if back is not None:  # in-place rollback: the restored state plays this round; the dropped policy's
                played_by, data = back, []  # tries never train
                self.mg_write_state(number - 1, played_by)  # a restart from here resumes the restored state
            last = int(mg.get("updates", 8)) + 1
            if getattr(self, "served_dir", None):  # rounds before the last one are trained: their records go
                marks = self.__dict__.setdefault("served_marks", {})
                marks[number] = time.time()
                # two rounds back (21 Oct): with two cards the play server keeps its prefix cache across rounds, so
                # round r-1's tries can hit prefixes round r-2 computed
                back_n = 3 if self.mg_versioned() else 2  # a look-ahead try (lag 2) trains a round later
                if marks.get(number - back_n) and Path(self.served_dir).is_dir():  # no directory: nothing captured yet
                    from .served_route import prune
                    self.feed.emit("served_prune", round=number, **prune(self.served_dir, marks[number - back_n]))
                    if getattr(self, "served_mirror", None) is not None and self.served_mirror.is_dir():
                        self.feed.emit("served_prune", round=number, mirror=True,
                                       **(await asyncio.to_thread(prune, self.served_mirror, marks[number - back_n])))
            reads_only = bool(mg.get("max_rounds")) and number >= int(mg["max_rounds"])
            need = (1.25 * last_play_s if last_play_s else last_round_s) if reads_only else last_round_s
            # no wall-clock stops; runs end on update counts and checks. The infrastructure
            # timeout is a crash ceiling only: nearing it wakes the agent (watchers read alerts), the run goes on.
            forced = (json.loads(stub).get("handoff_after_round") if stub.startswith("{") else None)
            if (mg.get("ceiling_handoff") and number > first
                    and (self.left() < (need or 0) + 900 or (forced is not None and number > int(forced)))):
                # Modal's function limit (24 h) is below a long run's length: the run hands off at a round boundary
                # (state and checkpoint written at the last round's end) and the launcher's chainer resumes it in a new
                # container (a state-only restart), instead of the hard kill losing a round mid-way
                stop = "ceiling_handoff"
                self.feed.emit("ceiling_handoff", round=number, left=round(self.left()), need=need)
                break
            if number > 1 and self.left() < (need or 0) + 120 and not getattr(self, "mg_ceiling_alerted", False):
                self.mg_ceiling_alerted = True
                self.feed.emit("alert", reason="crash_ceiling_near", round=number, left=round(self.left()), need=need)
            model = self.mg_name() if self.serving_adapter is not None else base_name
            keep_key = self.mg_try_keep(played_by)
            # mg.read_final (pilot 2): the read-only last round plays the held-out starts only, no training tries
            trains = number < last and not (reads_only and mg.get("read_final"))
            ahead, joined = getattr(self, "mg_ahead", None), 0
            if ahead is not None and (int(ahead["round"]) != number or not trains):
                await self.mg_ahead_cancel(f"round {number}: look-ahead for {ahead['round']}, trains {trains}")
                ahead = None
            if ahead is not None:  # mg.lookahead_below: its sets already play; the branch groups start now
                jobs, names = train_jobs(number, model, part="branch")
                jobs, names, joined = list(ahead["tasks"]) + jobs, list(ahead["names"]) + names, len(ahead["tasks"])
                self.mg_ahead = None
                self.feed.emit("lookahead_join", round=number, version=ahead["version"], sets=ahead["names"],
                               done=sum(1 for t in ahead["tasks"] if t.done()))
            else:
                jobs, names = train_jobs(number, model) if trains else ([], [])
            if self.serving_adapter is not None and bench is None:
                for name in tests:
                    s = sets[name]
                    tries = self.mg_test_tries(number, name)
                    if tries <= 0:  # probe-only test set (pilots: no test reads), or not this round's unseen read
                        continue
                    st = self.mg_ok_starts(name, screen)
                    u = self.mg_unseen_read(name=name)
                    if u and u["starts"]:
                        st = st[:u["starts"]]
                    jobs.append(self.mg_play(f"r{number:02d}-{name}", s, st, tries,
                                             model=self.mg_name(), seed=int(s.get("seed", mg.get("seed", 8000)))))
                    names.append(name)
                    done = self.__dict__.setdefault("mg_unseen_base_sets", set())
                    if u and u["base_first"] and name not in done:
                        done.add(name)  # the base beside the set's first read, same starts and seeds
                        jobs.append(self.mg_play(f"r{number:02d}-{name}-base", sets[f"{name}-base"], st, tries,
                                                 model=base_name, seed=int(s.get("seed", mg.get("seed", 8000)))))
                        names.append(f"{name}-base")
            fixed_names = list(fixed) if number < last else []
            if fixed_names and (getattr(self, "mg_alt", None) or {}).get("on") and number % 2:
                fixed_names = []  # mg.heldout_alternate on: the held-out read every 2nd (even) round
                self.feed.emit("heldout_skip", round=number)
            base_round = self.serving_adapter is None
            every = int(mg.get("fixed_every") or 1)  # multi-game: reads every Nth round; base rounds always read
            if fixed_names and every > 1 and not base_round and number % every and number + 1 < last:
                fixed_names = []
                self.feed.emit("heldout_skip", round=number, every=every)
            n_fixed = ((int(mg.get("fixed_base_tries") or mg["fixed_tries"]) if base_round else int(mg["fixed_tries"]))
                       if fixed_names else 0)
            for name in fixed_names:
                held = fixed[name]
                st = held if isinstance(held, list) else [held] if held else []
                jobs.append(self.mg_play(f"r{number:02d}-fixed-{name}", sets[name], st,
                                         max(1, n_fixed // max(1, len(st))), model=model,
                                         seed=int(mg.get("seed", 8000)) + 500))
            single = bool(mg.get("single_card"))  # pilots (19 Oct): server and trainer take turns on one card
            pending_now = pending is not None and number == first and not single
            # mg.r1_first (start-up cut (a), MFU-2 23 Oct): a fresh single-card run plays round 1 (base) before the
            # trainer exists; the trainer path (load, base references, r00) runs beside the server's round-end sleep
            r1_defer = bool(single and number == 1 and not started and mg.get("r1_first") and not seed and early is None
                            and not mg.get("resume") and bench is None)
            if r1_defer:  # start() runs at this round's end (mg_r1_deferred), not before its play
                started = True
            if number == first:
                extra = (start_seeded() if seed else start()) if number == 1 and not started else None
                if early_round:
                    extra = early_startup()
                if pending_now:  # mg.resume_train_pending: U on the pending round's pack after the trainer is up
                    async def pending_update(before=extra, data_p=pending["data"], n=number):
                        if before is not None:
                            await before
                        return await self.mg_update(n, data_p, probes)

                    extra = pending_update()
            else:
                if data and self.mg_versioned():  # look-ahead: lag per trained try, outside 0..2 dropped
                    data = self.mg_lag_filter(number, data, played_by)
                extra = None if single else (self.mg_update(number, data, probes) if data else None)
            updating = extra is not None and (number > 1 or bool(seed)) and (not early_round or pending_now)
            if r1_defer:
                if mg.get("served_route"):
                    served_setup()
                self.mg_r1_deferred = True
                host = None
                if mg.get("host_load") and self.mg_layout_env():  # cut (b): the trainer loads on the host during play
                    gate = self.work / "trainer-gpu-gate-r1"
                    gate.parent.mkdir(parents=True, exist_ok=True)
                    gate.unlink(missing_ok=True)
                    self.mg_host_trainer = self.spawn_trainer(gate=gate, extra_env={
                        **self.mg_layout_env(), "RL1_HOST_LOAD": "1",
                        "RL1_HOST_LOAD_THREADS": str(int(mg.get("host_load_threads") or 4))})
                    host = {"gate": str(gate), "record": self.mg_layout_env()["RL1_LAYOUT_RECORD"]}
                self.feed.emit("r1_first", round=number, host_load=host)
            if single and extra is not None:  # the trainer loads beside the sleeping server, then parks for play
                if seed:
                    raise ValueError("single_card: seed_u1 is not supported")
                await self.server.sleep()
                await extra
                self.feed.emit("card_phase", round=number, phase="load", **(await self.mg_card_play()))
                extra = None
            ts_row = None
            if mg.get("tail_fill"):  # every round bench registers before the first try starts (held-out last)
                from . import level_bench as level_bench_tail
                if level_bench_tail.SHARED_SLOTS is not None:  # a joined look-ahead's benches registered already
                    level_bench_tail.SHARED_SLOTS.hold(len(jobs) - joined)
            if early_round:  # handles to cancel on a failed gate
                jobs = [asyncio.ensure_future(j) for j in jobs]
            if mg.get("timeshare") and bench is None:
                from . import level_bench as level_bench_mod
                jobs = [asyncio.ensure_future(j) for j in jobs]
                served_now = self.serving_adapter or (mg.get("base_adapter") if base_name == "policy" else None)

                async def shared_card(inner, tasks=jobs, adapter=served_now):
                    out = await inner if inner is not None else None
                    nonlocal ts_row
                    if inner is None:  # no update this round (a resume's first round): card 2 plays as soon as tries queue
                        pool = level_bench_mod.SHARED_SLOTS
                        need = int(mg["timeshare"].get("min_queued", 4))
                        while not all(t.done() for t in tasks):
                            await asyncio.sleep(30)
                            if pool is not None and sum(max(0, c) for _, c in pool.pending.values()) >= need:
                                break
                    if not getattr(self, "ts_dead", False):
                        try:
                            ts_row = await self.ts_on(number, tasks, adapter)
                        except Exception as error:  # the second server is a speed-up: the round goes on without it
                            self.ts_dead = True
                            if level_bench_mod.SHARED_SLOTS2 is not None:
                                level_bench_mod.SHARED_SLOTS2.active = False
                            if getattr(self, "server2", None) is not None:
                                self.server2.stop()  # the trainer's card back in full
                            self.feed.emit("alert", why="time share failed; off for the rest of the run", error=repr(error)[:1500])
                            ts_row = {"error": repr(error)[:300]}
                    return out

                extra = shared_card(extra)
            from . import level_bench as level_bench_live
            level_bench_live.PROGRESS.update(done=0, clears=0)
            level_bench_live.PROGRESS_HOOK = self.mg_progress_row
            self.mg_live_sets = {b: v for b, v in (getattr(self, "mg_live_sets", None) or {}).items()
                                 if not re.match(r"r\d\d-", b) or int(b[1:3]) >= number}  # look-ahead benches stay

            async def ticker():  # dashboard: live play counters every 2 min of the round's play
                last = None
                while True:  # only on change: a silent hang must still read as silence to the stall watch
                    await asyncio.sleep(120)
                    now = {**level_bench_live.PROGRESS, "played_by": played_by, "sets": self.mg_live_round(number)}
                    if now != last:
                        self.feed.emit("play_progress", round=number, loop_lag_max=round(self.loop_lag["tick"], 2), **now)
                        self.loop_lag["tick"] = 0.0
                        last = now

            tick = asyncio.ensure_future(ticker())
            look = None
            below = mg.get("lookahead_below")
            nxt = number + 1
            cap_r = int(mg["max_rounds"]) if mg.get("max_rounds") else None  # round nxt trains (read_final: not the last)
            if (below and trains and bench is None and nxt < last
                    and not (cap_r and (nxt > cap_r or (nxt == cap_r and mg.get("read_final"))))):
                # an early-play round (the first after a resume) looks ahead too, once its play is verified (r4 R13:
                # the tail left card 1 idle)
                look = asyncio.ensure_future(self.mg_lookahead_watch(number, below, train_jobs, model,
                                                                     updating=extra is not None, early=early_round))
            self.mg_trainer_lane = None
            if extra is not None:  # the trainer lane's wall time (mg.heldout_alternate)
                async def timed(coro, began=time.time()):
                    try:
                        return await coro
                    finally:
                        self.mg_trainer_lane = time.time() - began

                extra = timed(extra)
            if early_round:
                extra = asyncio.ensure_future(extra)
            overlap = (bool(mg.get("reads_overlap")) and bench is None and not early_round and bool(fixed_names)
                       and number + 1 < last)
            try:
                if overlap:  # 23 Oct (owner priority): the held-out read no longer holds the round; it joins the next
                    # round's row (r5 R14: cap-length held-out L1 tries ran 68 min, the trainer card idled ~24 min)
                    n_main = len(names)
                    reads = [asyncio.ensure_future(j) for j in jobs[n_main:n_main + len(fixed_names)]]
                    main = [*jobs[:n_main], *jobs[n_main + len(fixed_names):]]
                    got = await asyncio.gather(*main, *([extra] if extra is not None else []))
                    results = [*got[:n_main], *([None] * len(fixed_names)), *got[n_main:]]
                    self.__dict__.setdefault("mg_reads_pending", []).append(
                        {"round": number, "played_by": played_by, "names": list(fixed_names), "futs": reads})
                    self.feed.emit("reads_deferred", round=number, sets=list(fixed_names),
                                   done=sum(1 for f in reads if f.done()))
                    fixed_names = []
                else:
                    results = await asyncio.gather(*jobs, *([extra] if extra is not None else []))
            except BaseException as error:
                if early_round and not getattr(self, "mg_early_verified", False):  # unverified tries never train
                    await self.mg_early_discard([*jobs, extra], number, error)
                raise
            finally:
                tick.cancel()
                if look is not None:
                    look.cancel()
            # the ticker's last row can miss the slowest tries (ovf1 R1: 92/100, RHAE/10K 0.391 vs 0.344 final)
            self.feed.emit("play_progress", round=number, final=True, loop_lag_max=round(self.loop_lag["tick"], 2),
                           **level_bench_live.PROGRESS, played_by=played_by, sets=self.mg_live_round(number))
            if mg.get("timeshare") and bench is None and not getattr(self, "ts_dead", False):
                ts_row = {**(ts_row or {}), **(await self.ts_off())}
            row = {"round": number, "played_by": played_by}
            if getattr(self, "mg_admission_counts", None) is not None:  # mg.engine_admission: paused polls by reason
                from . import level_bench as level_bench_adm
                row["engine_admission"] = {**self.mg_admission_counts,
                                           "waits": getattr(level_bench_adm.SHARED_SLOTS, "engine_waits", None)}
                self.mg_admission_counts = {}
            if ts_row is not None:
                row["timeshare"] = ts_row
            if getattr(self, "served_mirror_stats", None):
                row["served_mirror"] = dict(self.served_mirror_stats)
            acc = getattr(self, "mg_served_acc", None)
            if acc:  # R3 coverage of this round's update: windows with no served record train on the trainer's routing
                seen = acc.get("windows", 0) + acc.get("no_record", 0)
                row["served_routes"] = {**acc, "unserved_window_share": round(acc.get("no_record", 0) / seen, 4) if seen else None,
                                        "unresolved_token_share": round(acc.get("unresolved", 0) / acc["tokens"], 4)
                                        if acc.get("tokens") else None}
                self.mg_served_acc = {}
            if getattr(self, "mg_lag_row", None) is not None:  # this round's update: lag per trained try
                row["lag"], self.mg_lag_row = self.mg_lag_row, None
            if joined:
                row["lookahead_joined"] = {"benches": joined, "version": ahead["version"], "tries": ahead.get("tries")}
            launched = getattr(self, "mg_ahead", None)
            if launched is not None and int(launched["round"]) == number + 1:  # card-1 tail use (the util read)
                row["lookahead"] = {"for_round": number + 1, "version": launched["version"],
                                    **{k: launched.get(k) for k in ("launched_at_s", "running_at_launch", "tries")},
                                    "finished_by_round_end": self.mg_ahead_finished(launched)}
            plays = dict(zip(names, results[: len(names)]))
            fixed_rows = dict(zip(fixed_names, results[len(names): len(names) + len(fixed_names)]))
            round_folders = [v[0] for v in [*plays.values(), *fixed_rows.values()] if v and v[0]]
            benches = [p for _, _, p in [*plays.values(), *fixed_rows.values()]]
            if benches:  # the round's play on the card (pilots: round 2's play through the adapter vs round 1's)
                secs = max(float(p.get("seconds") or 0) for p in benches)
                toks = sum(int(p.get("tokens") or 0) for p in benches)
                row["play"] = {"seconds": round(secs, 1), "tokens": toks, "tok_s": round(toks / secs, 1) if secs else None,
                               "attempts": sum(int(p.get("attempts") or 0) for p in benches),
                               "usable": sum(int(p.get("usable") or 0) for p in benches)}
            if mg.get("heldout_alternate") and row.get("play"):
                row["lanes"] = self.mg_heldout_alternate(number, row["play"]["seconds"], getattr(self, "mg_trainer_lane", None))
            read_n, read_by = number, played_by
            if mg.get("reads_overlap") and not fixed_rows:
                ready = [p_ for p_ in getattr(self, "mg_reads_pending", []) if p_["round"] < number]
                if ready:  # the latest earlier read (finished by now: it had a whole round)
                    for p_ in ready:
                        self.mg_reads_pending.remove(p_)
                    p_ = ready[-1]
                    got = await asyncio.gather(*p_["futs"], return_exceptions=True)
                    bad = [repr(g)[:300] for g in got if isinstance(g, BaseException)]
                    if bad:
                        self.feed.emit("alert", why="deferred held-out read failed", round=p_["round"], errors=bad)
                    else:
                        fixed_rows, read_n, read_by = dict(zip(p_["names"], got)), p_["round"], p_["played_by"]
                        row["read_round"] = read_n
                        round_folders += [v[0] for v in fixed_rows.values() if v and v[0]]
            if read_by == 0 and not self.mg_init and fixed_rows:  # rounds the base played: the held-out reference
                self.mg_base_rounds = sorted(set(getattr(self, "mg_base_rounds", [])) | {read_n})
            if fixed_rows:
                row["fixed"] = {n: f"{sum(1 for r in rows if clear_tokens(r))}/{len(rows)}" for n, (_, rows, _) in fixed_rows.items()}
                row["fixed"]["pooled"] = held_metrics([r for _, rows, _ in fixed_rows.values() for r in rows])
                l13 = [r for n, (_, rows, _) in fixed_rows.items() if int(sets[n].get("level") or 0) in (1, 3) for r in rows]
                if l13:  # dashboard: L1+L3 only (L2 dominates the pooled numbers)
                    row.setdefault("l13", {})["eval"] = held_metrics(l13)
                if mg.get("read_norm"):  # envs 22 Oct: multi-game read, per set vs the base, equal weight per game
                    self.mg_read_norm(read_n, read_by, fixed_rows, sets, row)
                if mg.get("fixed_fresh"):  # mg5: per trained level, held-out tokens to clear against the base rounds
                    self.mg_fixed = getattr(self, "mg_fixed", {})
                    self.mg_fixed[read_n] = {n: rows for n, (_, rows, _) in fixed_rows.items()}
                    row["shortening"] = self.mg_shortening(read_n)
                    if mg.get("behaviour_guard") and self.mg_policy_round(read_by):
                        row["behaviour"] = self.mg_behaviour(read_n)  # reads_overlap: the guard acts a round later
            # rule (d), a bench under min_usable leaves the round (metrics and training); the run
            # stops when the round's pooled usable share is under it or the same bench fails 2 rounds running
            floor, low, asked, kept = float(mg.get("min_usable", 0.75)), {}, 0, 0
            for n, (folder, rows, play) in plays.items():
                a = int(play.get("attempts") or len(rows))
                asked, kept = asked + a, kept + len(rows)
                if a and len(rows) / a < floor:
                    low[n] = round(len(rows) / a, 3)
            pending = None
            if low:
                row["excluded"] = low
                self.feed.emit("bench_excluded", round=number, benches=low)
                if set(low) & getattr(self, "mg_low", set()):
                    pending = "rule_d_usable"
            if asked and kept / asked < floor:
                pending = "rule_d_usable"
            if pending:
                row["rule_detail"] = {"excluded": low, "pooled_usable": round(kept / asked, 3) if asked else None}
            self.mg_low = set(low)
            plays = {n: v for n, v in plays.items() if n not in low}
            if number == 1 and seed:  # the seed's sampler check against this run's own round-1 requests
                seen = set()
                for n, (folder, rows, _) in plays.items():
                    for r in rows[:4]:
                        with open(Path(folder) / "attempts" / r["label"] / "rollout.jsonl") as file:
                            seen |= {json.dumps(json.loads(x).get("sampling"), sort_keys=True) for x in file if x.strip()}
                want = json.dumps(mg.get("seed_sampling") or {"temperature": 0.7, "top_p": 0.95, "top_k": 20,
                                                              "max_tokens": 16384}, sort_keys=True)
                row["seed_check"] = {"round1_sampling": sorted(seen), "match": seen == {want}}
                if seen != {want}:
                    self.feed.emit("alert", why="round-1 sampler differs from the seed's", seen=sorted(seen), want=want)
            if any(sets[n]["role"] == "test" for n in plays):
                row["tests"] = self.mg_tests({n: r for n, r in plays.items() if sets[n]["role"] == "test"}, screen)
            for n in [n for n in plays if sets[n]["role"] == "base_read"]:  # unseen read's base, matched (22 Oct)
                row.setdefault("tests", {})[n] = {"base_matched": held_metrics(plays[n][1]), "play_s": plays[n][2]["seconds"]}
            new, tinfo = [], {}
            for n, (folder, rows, play) in plays.items():
                if sets[n]["role"] == "train":
                    base_set = sets[n].get("branch_of") or n  # a branch group is judged on its set's references
                    if sets[n].get("branch_of"):
                        rows = branch_mod.score_rows(self, rows)
                        tinfo_rem = sum(1 for r in rows if r.get("branch_rem"))
                    sel, tinfo[n] = self.mg_select(folder, rows, base_set)
                    sel = [(*x, base_set) for x in sel]  # the set name rides along (per-level norm and mass)
                    tinfo[n]["clears"] = f"{sum(1 for r in rows if clear_tokens(r))}/{len(rows)}"
                    if sets[n].get("branch_of"):
                        tinfo[n]["remainder"] = tinfo_rem
                    new += sel
            row["train"] = tinfo
            if getattr(self, "mg_curr", None) is not None:
                row["curriculum"] = self.mg_curriculum_observe(plays, sets)
            if mg.get("explore"):  # running caps: 1.25 x p80 of clears, explorers' tokens before that
                from .play_explorer import update_caps
                row["caps"] = update_caps(self, plays, mg.get("running_caps"))
            if mg.get("harvest"):  # in-run harvest (envs, 21 Oct): this round's clears cut the next level's starts
                try:
                    hs = harvest_mod.settings(mg)
                    checks = harvest_mod.arcade(self) if hs["board_check"] else None
                    got = harvest_mod.collect(self, number, plays, played_by, checks)
                    if got is not None:
                        got["clear_by_source"] = harvest_mod.sources(self, plays)
                        row["harvest"] = got
                        self.feed.emit("mg_harvest", **got)
                        if getattr(self, "mg_curr", None) is not None:
                            for rec in self.mg_harvest[len(self.mg_harvest) - int(got["new"]):]:
                                self.mg_curr.add_start(rec, create=False)
                except Exception as error:  # noqa: BLE001 - a broken harvest leaves the pool as it was, the run goes on
                    row["harvest"] = {"error": repr(error)[:300]}
                    self.feed.emit("alert", why="harvest failed", round=number, error=repr(error)[:1500])
            if mg.get("branch"):  # branch cuts from this round's training tries, played next round
                try:
                    got = branch_mod.collect(self, number, {n: v for n, v in plays.items()
                                                            if not sets[n].get("branch_of")}, played_by)
                    if got is not None:
                        row["branch"] = got
                        self.feed.emit("mg_branch", **got)
                except Exception as error:  # noqa: BLE001 - a broken cut leaves the next round without branches
                    row["branch"] = {"error": repr(error)[:300]}
                    self.feed.emit("alert", why="branch collect failed", round=number, error=repr(error)[:1500])
            if mg.get("breakage"):
                row["play_stats"] = self.mg_play_stats([r for _, rows, _ in [*plays.values(), *fixed_rows.values()] for r in rows])
                if played_by == 0 and not self.mg_init and getattr(self, "mg_base_play", None) is None:
                    self.mg_base_play = row["play_stats"]
                elif self.mg_policy_round(played_by) and getattr(self, "mg_base_play", None):
                    row["breakage"] = self.mg_broken(row["play_stats"], self.mg_base_play,
                                                     (row.get("behaviour") or {}).get("clear_drop"),
                                                     (row.get("behaviour") or {}).get("actions_rise"))
            rule = (pending or self.mg_guard_stop(row) or self.mg_rules(number, plays, row, played_by)
                    or self.mg_rule_f(row, played_by)
                    or self.mg_success(row, played_by))
            final = bool(mg.get("max_rounds")) and number >= int(mg["max_rounds"])  # the last round only reads
            if row.get("breakage") and not rule and played_by == 0 and self.mg_init:
                rule = "init_breakage"  # round 1 of a warm start is its breakage check: stop and report, no rollback
            if row.get("breakage") and not rule and not final:
                back = await self.mg_break_rollback(number, row, probes)
                if back is None:
                    rule = "play_breakage"
                else:
                    played_by, new = back, []
                    self.mg_behaviour_trips = 0  # the broken adapter is gone; the guard's streak restarts
                    if self.mg_step_rollbacks >= int(mg.get("step_rollbacks_stop") or 3):
                        rule = "rule_c_step_rollbacks"
            if mg.get("warn") and self.mg_policy_round(played_by) and new and not row.get("breakage") and not rule and not final:
                row["warning"] = self.mg_warning(row)
                if row["warning"]:
                    self.mg_set_lr(0.5, row, f"early warning: {row['warning']}")
            last_play_s = round(asyncio.get_event_loop().time() - began, 1)
            slept = False
            if getattr(self, "mg_r1_deferred", False):  # mg.r1_first: the trainer path beside the sleeping server
                self.mg_r1_deferred = False
                await self.server.sleep()
                slept, t_start = True, time.time()
                await start()
                self.feed.emit("r1_first_trainer", round=number, seconds=round(time.time() - t_start, 1))
            if single and new and number < last and not rule and not final:  # this round's tries train now
                if not slept:
                    await self.server.sleep()
                results, updating = [*results, await self.mg_update(number, new, probes)], True
            if updating:
                update_row, serve = results[-1]
                stop, served = await self.mg_after_update(number, row, update_row, serve, load=not single)
                if served:
                    played_by = number
                if single:
                    row["card_phase"] = await self.mg_card_play()
                    if served or self.serving_adapter is not None:  # the new adapter, or the kept one again after sleep
                        row["swap"] = {"load_seconds": await self.server.load("policy", self.serving_adapter)}
            elif slept:  # mg.r1_first without an update this round: the trainer parks, the server wakes for play
                row["card_phase"] = await self.mg_card_play()
                if self.serving_adapter is not None:
                    row["swap"] = {"load_seconds": await self.server.load("policy", self.serving_adapter)}
            data = new
            if rule and stop is None:
                stop = row["stop"] = rule
                self.feed.emit("stop_rule", round=number, rule=rule, detail=row.get("rule_detail"))
            from . import level_bench as _lb
            if _lb.SHARED_SLOTS is not None:
                row["slots_peak"], _lb.SHARED_SLOTS.peak = _lb.SHARED_SLOTS.peak, 0
                row["kv_waits"], row["kv_peak_load"] = _lb.SHARED_SLOTS.kv_waits, _lb.SHARED_SLOTS.peak_load
            if keep_key:
                row["try_keep"] = {"key": keep_key, "reused": _lb.PROGRESS.get("reused", 0)}
            row["round_seconds"] = last_round_s = round(asyncio.get_event_loop().time() - began, 1)
            row["loop_lag_max"], self.loop_lag["round"] = round(self.loop_lag["round"], 2), 0.0
            self.feed.emit("round", **row)
            self.history.append(row)
            if getattr(self, "mg_own_copy", None) is not None:  # S2 resume_skip: the state copy mg_state points at
                await self.mg_own_copy
            self.mg_write_state(number, played_by)
            if mg.get("pack_rounds", True) and os.environ.get("ARC_VALIDATION_OUTPUT"):
                prior = getattr(self, "mg_pack_task", None)
                if prior is not None:
                    await prior  # one archive at a time
                self.mg_pack_task = asyncio.ensure_future(asyncio.to_thread(self.mg_pack_round, number, round_folders))
            if stop is None and mg.get("max_rounds") and number >= int(mg["max_rounds"]):  # GPU check: round 1 + U1
                stop = "max_rounds"
            manual = os.environ.get("ARENA_STOP_REQUEST")  # 21 Oct: the mg loop now honours the volume stop request
            if stop is None and manual and Path(manual).exists():
                stop = "manual"
            if stop is not None:
                break
        await self.mg_ahead_cancel(f"loop end: {stop}")  # its finished tries stay kept; mg_state names its picks
        if getattr(self, "mg_pack_task", None) is not None:
            await self.mg_pack_task
        if self.serving_adapter is not None and os.environ.get("ARC_VALIDATION_OUTPUT") and bench is None:
            dest = Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "adapters" / Path(self.serving_adapter).name
            await asyncio.to_thread(shutil.copytree, self.serving_adapter, dest, dirs_exist_ok=True)
            self.feed.emit("mg_adapter", path=str(dest), played_by=played_by)
        self.feed.emit("final", stop=stop, eval=(self.history[-1] if self.history else {}).get("tests"), base=None)
        return stop

    def mg_pack_round(self, number, folders):
        """The round's play folders (every try's rollout and log) as one archive on the volume,
        ``$ARC_VALIDATION_OUTPUT/rounds/rNN.tar`` (a stopped run kept no try logs; one inode per round).
        Members are ``<folder name>/...``. Written to a temp name, then renamed."""
        import tarfile

        dest = Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "rounds" / f"r{number:02d}.tar"
        began = time.time()
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            with tarfile.open(tmp, "w:gz", compresslevel=1) as tar:  # rollouts carry top-20 logprobs (~12 MB/try)
                for folder in sorted({str(f) for f in folders}):
                    if Path(folder).is_dir():
                        tar.add(folder, arcname=Path(folder).name)
            os.replace(tmp, dest)
            self.feed.emit("round_pack", round=number, folders=len(set(map(str, folders))),
                           bytes=dest.stat().st_size, seconds=round(time.time() - began, 1))
        except Exception as error:  # noqa: BLE001 - the archive never ends the run
            self.feed.emit("alert", why="round pack failed", round=number, error=repr(error)[:500])

    def mg_policy_round(self, played_by):
        """A round played by a served adapter: after an update, or from round 1 under mg.init_state."""
        return played_by >= 1 or bool(getattr(self, "mg_init", False))

    # ---- mg.lookahead_below (MFU-2, 22 Oct; continuous play V1): versioned adapters, look-ahead rounds, lag ----

    def mg_versioned(self):
        """Look-ahead on: the served adapter is named policy-rNN (NN: its state's round) and every try records it."""
        return bool(self.cfg["mg"].get("lookahead_below"))

    def mg_name(self, version=None):
        """The served adapter's LoRA name: "policy", or policy-rNN under mg.lookahead_below."""
        if not self.mg_versioned():
            return "policy"
        return f"policy-r{int(getattr(self, 'mg_version', 0) if version is None else version):02d}"

    async def mg_serve(self, path, version):
        """The play server serves ``path`` as state round ``version``'s adapter (server.load under mg_name); a
        versioned name no running bench uses any more is unloaded."""
        name = self.mg_name(version)
        pending = self.__dict__.setdefault("mg_unloading", {}).get(name)
        if pending is not None:  # a rollback re-serving a name whose unload is in flight: load after it lands
            await pending.wait()
        seconds = await self.server.load(name, path)
        self.mg_version = int(version)
        if self.mg_versioned():
            self.__dict__.setdefault("mg_loaded", {})[name] = str(path)
            lin = self.__dict__.setdefault("mg_lineage", [])
            while lin and lin[-1] > int(version):  # a rollback: the dropped branch leaves the lineage
                lin.pop()
            if not lin or lin[-1] != int(version):
                lin.append(int(version))
            del lin[:-8]
            await self.mg_gc_names()
        return seconds

    async def mg_gc_names(self):
        users = getattr(self, "mg_users", {})
        for name in [n for n in getattr(self, "mg_loaded", {}) if n != self.mg_name() and not users.get(n)]:
            path = self.mg_loaded.pop(name, None)  # claimed before the await: a concurrent GC (benches ending
            if path is None:  # together, v7 L3b) never unloads the same name twice
                continue
            done = self.__dict__.setdefault("mg_unloading", {})[name] = asyncio.Event()
            try:
                await self.server.unload(name)
            except Exception:
                self.mg_loaded[name] = path
                raise
            finally:
                self.mg_unloading.pop(name, None)
                done.set()
            self.feed.emit("adapter_unloaded", name=name)

    def mg_play_version(self, model):
        """The adapter version a bench on ``model`` plays: its policy-rNN, the served one for "policy", 0 the base."""
        m = re.fullmatch(r"policy-r(\d+)", str(model))
        if m:
            return int(m.group(1))
        return int(getattr(self, "mg_version", 0)) if model == "policy" and self.serving_adapter is not None else 0

    def mg_lag(self, row):
        """A try's policy lag at train time: updates served since its adapter (the served lineage; the trainer's
        state, pi_old, is the served one). A try without a version: last round's adapter (1); an adapter outside
        the lineage (a rolled-back branch, an older leg): None."""
        if row.get("policy_version") is None:
            return 1
        lin = getattr(self, "mg_lineage", None) or []
        v = int(row["policy_version"])
        return lin[::-1].index(v) if v in lin else None

    def mg_lag_filter(self, number, data, played_by):
        """Train-time policy lag (mg_lag) per selected try; tries beyond mg.lookahead_lag (2) or off the lineage
        are dropped and counted per set. Returns the kept data; the record goes to self.mg_lag_row (the round row's
        ``lag``)."""
        cap = int(self.cfg["mg"].get("lookahead_lag", 2))
        kept, hist, dropped, per, missing = [], {}, {}, [], 0
        for item in data:
            row, base_set = item[1], item[-1]
            missing += row.get("policy_version") is None  # every versioned try carries one: counted, lag 1
            lag = self.mg_lag(row)
            hist[str(lag)] = hist.get(str(lag), 0) + 1
            if lag is not None and lag <= cap:
                kept.append(item)
                per.append([row.get("label"), lag])
            else:
                dropped[base_set] = dropped.get(base_set, 0) + 1
        self.mg_lag_row = {"update": number, "played_by": int(played_by), "trained_by_lag": hist,
                           "dropped_by_set": dropped, "cap": cap, "missing_version": missing, "per_try": per}
        self.feed.emit("policy_lag", **{k: v for k, v in self.mg_lag_row.items() if k != "per_try"})
        return kept

    def mg_lookahead_ready(self, began, pool, below):
        """Round ``number``'s benches have all launched (no pending tries on the play server's pool) and fewer than
        ``below`` tries run there, a minute or more into the round."""
        if pool is None or time.time() - began < LOOKAHEAD_MIN_S:
            return False
        if any(count > 0 for _, count in pool.pending.values()):
            return False
        return pool.used < int(below)

    def mg_ahead_roots(self, rec, resumed=False):
        """try_keep roots of a look-ahead round's benches: kept under its own version's key (the round's key while
        it plays), reused from the earlier legs' same key. ``resumed``: a resume replaying its recorded picks keeps
        under this round's key and also reuses the look-ahead's finished tries (each row keeps its version)."""
        from . import level_bench as lb

        out = os.environ.get("ARC_VALIDATION_OUTPUT")
        if not (out and self.cfg["mg"].get("try_keep")):
            return {"keep": None, "reuse": []}
        legs = [x for x in os.environ.get("ARC_RL1_RESUME", "").split(",") if x]
        mine = [[str(Path(r) / "partial" / rec["key"]), int(rec["version"])] for r in legs]
        if not resumed:
            return {"keep": str(Path(out) / "partial" / rec["key"]), "reuse": mine}
        now = int(getattr(self, "mg_version", 0))
        return {"keep": None if lb.TRY_KEEP is None else str(lb.TRY_KEEP),
                "reuse": mine + [[str(x), now] for x in lb.TRY_REUSE]}

    async def mg_lookahead_watch(self, number, below, train_jobs, model, updating=True, early=False):
        """mg.lookahead_below (continuous play V1): once round ``number``'s tries have all started and fewer than
        ``below`` run on the play server, round number+1's picked training sets start there (server 1 only) on the
        served adapter, which they keep for life; at most one look-ahead round in flight."""
        from . import level_bench

        began = time.time()
        while True:
            await asyncio.sleep(5)
            if getattr(self, "mg_ahead", None) is not None:
                return
            pool = level_bench.SHARED_SLOTS
            if early and not getattr(self, "mg_early_verified", False):  # unverified play never trains
                continue
            if self.mg_lookahead_ready(began, pool, below):
                break
        version = self.mg_play_version(model)
        lag = 1 + bool(updating)  # its tries train at round number+2's update: this round's and the next one's since
        if lag > int(self.cfg["mg"].get("lookahead_lag", 2)):
            self.feed.emit("lookahead_skip", round=number + 1, version=version, lag=lag)
            return
        rec = {"round": number + 1, "version": version, "uid": getattr(self, "policy_uid", None),
               "launched": round(time.time(), 1), "launched_at_s": round(time.time() - began, 1),
               "running_at_launch": int(pool.used)}
        rec["key"] = f"p{version:02d}-{rec['uid'] or 'base'}"
        picks, roots = {}, self.mg_ahead_roots(rec)
        jobs, names = train_jobs(number + 1, model, part="normal", picks=picks,
                                 own={"try_keep": roots, "server1_only": True})
        rec.update(picks=picks, names=names, keep=roots["keep"],
                   tries=sum(max(1, len(st)) * int(t) for st, t in picks.values()))
        rec["tasks"] = [asyncio.ensure_future(j) for j in jobs]

        def failed(task, name):  # a look-ahead bench's error surfaces when it happens, not only at the join
            if not task.cancelled() and task.exception() is not None:
                self.feed.emit("alert", why="lookahead bench failed", round=number + 1, bench=name,
                               error=repr(task.exception())[:300])

        for task, name in zip(rec["tasks"], names):
            task.add_done_callback(lambda t, n=name: failed(t, n))
        self.mg_ahead = rec
        self.feed.emit("lookahead", round=number + 1, version=version, sets=names, running=pool.used, below=int(below),
                       after_s=round(time.time() - began, 1))

    def mg_ahead_start(self, path):
        """A recorded look-ahead start's attempt folder as this run sees it (as harvest.restore_state: the path itself,
        the resumed run's harvest/ copy, or a packed preflight's unpacked /tmp copy); None when gone."""
        p, root = Path(str(path)), self.cfg["mg"].get("resume")
        cands = [p] + ([Path(root) / "harvest" / p.name] if root else [])
        if str(p).startswith("/runs/"):
            cands.append(Path("/tmp") / p.relative_to("/runs"))
        return next((str(c) for c in cands if (c / "rollout.jsonl").is_file()), None)

    def mg_ahead_finished(self, rec):
        """Tries of a look-ahead round finished so far (its benches' kept rows; None without mg.try_keep)."""
        if not rec.get("keep"):
            return None
        n = 0
        for name in rec.get("names") or []:
            try:
                n += sum(1 for x in (Path(rec["keep"]) / f"r{int(rec['round']):02d}-{name}" / "tries.jsonl").read_text()
                         .splitlines() if x.strip())
            except OSError:
                pass
        return n

    async def mg_ahead_cancel(self, why):
        """Stop an in-flight look-ahead round (its finished tries stay kept under their versioned root; a resume
        reuses them), e.g. before the play server sleeps or the loop ends."""
        ahead = getattr(self, "mg_ahead", None)
        if not ahead:
            return
        self.mg_ahead = None
        for t in ahead["tasks"]:
            t.cancel()
        await asyncio.gather(*ahead["tasks"], return_exceptions=True)
        self.feed.emit("lookahead_cancel", round=ahead["round"], version=ahead["version"], why=why)

    def mg_resume_dose(self, mg, upto=None):
        """lr_dose on resume: lr from the source's last full update up to the restored state's round ``upto`` (a
        rollback restart ignores the dropped updates; pf6 21 Oct: r05 dosed from the rolled-back U7); KL ~ lr^2."""
        last = self.mg_source_kl(mg["resume"], upto)
        if not last:
            return None
        prev = (last["previous"]["lr"], last["previous"]["excess"]) if last.get("previous") else None
        self.mg_dose_lr({}, float(mg.get("kl_target", 0.01)), last["excess"], last["lr"], f"resume, source round {last['round']}",
                        previous=prev)
        self.mg_last_dose = (last["lr"], last["excess"])
        row = {"source_round": last["round"], "source_lr": last["lr"], "source_excess": last["excess"], "lr": float(self.cfg["lr"])}
        self.feed.emit("resume_dose", **row)
        return row

    def mg_source_kl(self, source, upto=None):
        """The resume source's last full update with a measured KL excess: {"round", "lr", "excess"} or None."""
        feed = Path(source) / "arena" / "trials" / "rl-loop" / "rl" / "feed.jsonl"
        if not feed.is_file():
            return None
        last = None
        for line in feed.read_text().splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            k = e.get("kind") == "round" and e.get("kl_step") or {}
            if upto is not None and isinstance(e.get("round"), int) and e["round"] > int(upto):
                continue
            if k.get("action") == "full" and k.get("excess") and k.get("lr"):
                last = {"round": e.get("round"), "lr": float(k.get("lr_effective") or k["lr"]), "excess": float(k["excess"]),
                        "previous": {x: last[x] for x in ("round", "lr", "excess")} if last else None}
        return last

    def mg_try_keep(self, played_by):
        """mg.try_keep: the round's finished tries are kept as they end under
        partial/p<played_by>-<policy uid>/<bench>, and a restart (launcher ARC_RL1_RESUME: earlier legs, a preempted
        attempt) takes the tries a policy with the same uid finished there instead of replaying them. This run's own
        keeps are never reused (an in-run rollback replays fresh). Returns the key (None: off)."""
        out = os.environ.get("ARC_VALIDATION_OUTPUT")
        if not (out and self.cfg["mg"].get("try_keep")):
            return None
        from . import level_bench as lb

        key = f"p{int(played_by):02d}-{getattr(self, 'policy_uid', None) or 'base'}"
        lb.TRY_KEEP = Path(out) / "partial" / key
        lb.TRY_REUSE = [Path(r) / "partial" / key for r in os.environ.get("ARC_RL1_RESUME", "").split(",") if r]
        lb.PROGRESS["reused"] = 0
        return key

    def mg_write_state(self, number, played_by):
        """mg_state.json: the state a resume starts from (the served state's checkpoint; mg5e review: the newest
        folder can be a rolled-back update's)."""
        mg, out = self.cfg["mg"], os.environ.get("ARC_VALIDATION_OUTPUT")
        if not (out and mg.get("checkpoint") and getattr(self, "keep_checkpoint", None)):
            return
        keep = ("outcome", "level_rows", "completion_tokens", "label")
        text = json.dumps({
            "round": number, "checkpoint": str(Path(out) / "state" / self.keep_checkpoint), "played_by": played_by,
            "policy_uid": getattr(self, "policy_uid", None),
            "served": self.serving_adapter is not None, "base_tpr": getattr(self, "mg_base_tpr", None),
            "long": getattr(self, "mg_long", 0), "worse": getattr(self, "mg_worse", 0),
            "low": sorted(getattr(self, "mg_low", set())), "kl_trips": getattr(self, "kl_trips", 0),
            "step_rollbacks": getattr(self, "mg_step_rollbacks", 0), "accepted": getattr(self, "mg_accepted", 0),
            "steps_done": getattr(self, "mg_steps_done", 0),
            "lr": float(self.cfg["lr"]), "base_rounds": getattr(self, "mg_base_rounds", []),
            "heldout_alternate": getattr(self, "mg_alt", None),
            "rule_f_done": getattr(self, "mg_rule_f_done", False),
            "success_streak": getattr(self, "mg_success_streak", 0),
            "behaviour_trips": getattr(self, "mg_behaviour_trips", 0),
            "base_play": getattr(self, "mg_base_play", None), "kl_ramp": getattr(self, "mg_ramp_state", None),
            "kl_target": mg.get("kl_target"), "knobs_seen": getattr(self, "mg_knobs_seen", {}),
            "last_dose": list(getattr(self, "mg_last_dose", None) or []) or None,
            **({"lookahead": {k: v for k, v in self.mg_ahead.items() if k != "tasks"}}
               if getattr(self, "mg_ahead", None) else {}),
            **({"lineage": list(getattr(self, "mg_lineage", []))} if self.mg_versioned() else {}),
            **(harvest_state(self) if mg.get("harvest") else {}),
            **({"branch": getattr(self, "mg_branch", None) or []} if mg.get("branch") else {}),
            **({"curriculum": self.mg_curr.state_dict()} if getattr(self, "mg_curr", None) is not None else {}),
            **({"read_base": self.mg_read_base} if getattr(self, "mg_read_base", None) else {}),
            **({"level_obs": getattr(self, "mg_level_obs", {}), "explore_count": getattr(self, "mg_explore_count", 0),
                "set_caps": {n: v.get("cap") for n, v in self.cfg["mg_sets"].items() if v.get("role") == "train"}}
               if mg.get("explore") else {}),
            "fixed": {str(k): {n: [{f: r.get(f) for f in keep if f in r} for r in rows] for n, rows in v.items()}
                      for k, v in getattr(self, "mg_fixed", {}).items()}})
        if (getattr(self, "ckpt_pending", None) == self.keep_checkpoint
                and not (Path(out) / "state" / self.keep_checkpoint).is_dir()):  # mg.async_checkpoint: the state is still
            # being written; the root file keeps naming the last complete state (a crash resumes it and replays this
            # round) and this round's lands with the state (ckpt_flush: the folder appears, or checkpoint_landed)
            self.ckpt_deferred = (self.keep_checkpoint, text)
            return
        (Path(out) / "mg_state.json").write_text(text)
        snap = Path(out) / "state" / self.keep_checkpoint
        if snap.is_dir() and not (snap / "mg_state.json").exists():  # v5: the state's own resume point (rollbacks)
            shutil.copyfile(Path(out) / "mg_state.json", snap / "mg_state.json")

    @staticmethod
    def shrink_mg(mg, minimum=False):
        """The CPU preflight's shrink of an mg dict (in place): tries, starts and the harvest draw. ``minimum`` (owner
        22 Oct, plumbing legs): 1 start x 4 tries, fresh 4, fixed 2."""
        if minimum:
            mg.update(train_starts=1, train_tries=4, fresh_tries=4, fixed_tries=min(int(mg.get("fixed_tries") or 2), 2))
        else:
            mg.update(train_starts=2, train_tries=4, fresh_tries=8, fixed_tries=min(int(mg.get("fixed_tries") or 2), 4))
        for st in mg.get("sets") or []:
            for key in ("train_starts", "train_tries"):
                st.pop(key, None)
        if isinstance(mg.get("harvest"), dict):
            mg["harvest"].update(train_starts=1 if minimum else 2, train_tries=4)
        if isinstance(mg.get("curriculum"), dict):  # multi-game: fewer, smaller groups (every source still drawn)
            mg["curriculum"].update(groups=5 if minimum else 8, group_size=4)
            mg["group_size"] = 4
        if mg.get("fixed_base_tries"):
            mg["fixed_base_tries"] = min(int(mg["fixed_base_tries"]), 2 * int(mg["fixed_tries"]))
        if isinstance(mg.get("explore"), dict):  # explorers: a few short plays (a stub play of 300K takes hours)
            mg["explore"].update(streams=min(int(mg["explore"].get("streams") or 24), 2 if minimum else 4),
                                 budget=40000, wall=1800, poll=20,
                                 seg_tokens=min(int(mg["explore"].get("seg_tokens") or 65536), 12000))  # v7 chains:
            # mid-level continuations happen in the preflight too
        return mg

    PENDING_REWARD_KEYS = ("reward_shape", "rhae_cap", "rhae_hi", "token_hi", "tight_action_hi", "tight_action_pow",
                           "turn_credit", "level_norm", "l2_norm", "ref_round1", "reward_log", "cap")
    PENDING_LAYOUT_KEYS = ("group_size", "fresh_tries", "harvest", "branch", "train_tries", "train_starts")

    def mg_pending_round(self, state, sets):
        """mg.resume_train_pending: the resumed state's pending round, trained by this run's first update
        while round P+1 plays. Pending: the source's latest round P >= the state's round with a pack
        (rounds/rPP.tar) that no update in the state's lineage consumed, played by the served state (lag 0) or, for P =
        the state's round, by the state round's own player (lag 1, a stop at round P+1's top or mid-update).
        Compatibility with this run's settings (the source's own from its feed's start row): ``layout`` "strict"
        (default: the same train sets, reward settings and group layout) or "lenient" (opt-in per launch: the same
        train sets and reward settings; groups of 2+, sets not configured now dropped). Returns {"round", "played_by",
        "data", "info"} or None (logged: why it falls back to a play-only first round)."""
        import tarfile

        from . import branch as branch_mod
        from .rl_loop import usable

        mg = self.cfg["mg"]
        spec = mg["resume_train_pending"] if isinstance(mg["resume_train_pending"], dict) else {}
        layout = spec.get("layout", "strict")
        src = Path(mg["resume"])
        k, served = int(state["round"]), int(state.get("played_by", state["round"]))

        def fallback(why, **more):
            self.feed.emit("pending_round", used=False, why=why, layout=layout, state_round=k, **more)
            return None

        feed = src / "arena" / "trials" / "rl-loop" / "rl" / "feed.jsonl"
        try:
            events = [json.loads(x) for x in feed.read_text().splitlines() if x.strip()]
        except (OSError, ValueError) as error:
            return fallback(f"source feed unreadable: {error!r}"[:300])
        start = next((e for e in events if e.get("kind") == "start"), None)
        rows = {int(e["round"]): e for e in events if e.get("kind") == "round"}
        own = rows.get(k, {}).get("played_by")
        cands = [p for p, r in rows.items() if p >= k and (src / "rounds" / f"r{p:02d}.tar").is_file()
                 and (int(r.get("played_by", -1)) == served or (p == k and own is not None and int(r["played_by"]) == int(own)))]
        if not cands:
            return fallback("no pending round with a pack", rounds=sorted(rows))
        p = max(cands)
        if rows[p].get("stop"):
            return fallback(f"round {p} ended the source with stop {rows[p]['stop']}")
        theirs = ((start or {}).get("settings") or {}).get("mg") or {}
        if not theirs:
            return fallback("source settings missing from its feed")
        shrunk = next((e for e in events if e.get("kind") == "preflight_shrink"), None)
        if shrunk:  # a shrunk source (CPU preflight leg): its start row holds the settings before the shrink
            theirs = self.shrink_mg(json.loads(json.dumps(theirs)), minimum=shrunk.get("fresh_tries") == 4)

        def train_sets(m):
            return {s["name"]: (s.get("game"), int(s.get("level") or 1), s.get("cap"))
                    for s in m.get("sets") or [] if s.get("role") == "train"}

        diffs = {key: [theirs.get(key), mg.get(key)] for key in self.PENDING_REWARD_KEYS if theirs.get(key) != mg.get(key)}
        if train_sets(theirs) != train_sets(mg):
            diffs["train_sets"] = [train_sets(theirs), train_sets(mg)]
        def lay(d, key):  # harvest.since picks the next rounds' starts, not the pending pack's layout
            v = d.get(key)
            return {a: b for a, b in v.items() if a != "since"} if key == "harvest" and isinstance(v, dict) else v

        layout_diffs = {key: [theirs.get(key), mg.get(key)] for key in self.PENDING_LAYOUT_KEYS
                        if lay(theirs, key) != lay(mg, key)}
        if diffs or (layout == "strict" and layout_diffs):
            return fallback("settings differ", round=p, differs={**diffs, **layout_diffs})
        dest = self.work / "pending" / f"r{p:02d}"
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        with tarfile.open(src / "rounds" / f"r{p:02d}.tar") as tar:
            tar.extractall(dest, filter="data")
        new, tinfo, dropped = [], {}, {}
        names = [n for n, s in sets.items() if s["role"] == "train" and not s.get("branch_of")]
        if mg.get("branch"):
            names += [f"{n}-br" for n in list(names)]
        for name in names:
            folder = dest / f"r{p:02d}-{name}"
            if not (folder / "level-bench.json").is_file():
                continue
            base_set = name[:-3] if name.endswith("-br") else name
            report = json.loads((folder / "level-bench.json").read_text())
            got = [r for r in self.game_rows([r for r in report["attempts"] if usable(r)], sets[base_set]["game"])
                   if restored(r) and level_of(r) is not None]
            gated = [r for r in got if not self.mg_digest_ok(r)]
            got = [r for r in got if self.mg_digest_ok(r)]
            if gated:
                dropped[name] = len(gated)
            if name.endswith("-br"):
                got = branch_mod.score_rows(self, got)
            sel, tinfo[name] = self.mg_select(folder, got, base_set)
            tinfo[name]["clears"] = f"{sum(1 for r in got if clear_tokens(r))}/{len(got)}"
            new += [(*x, base_set) for x in sel]
        skipped = sorted(x.name for x in dest.iterdir() if x.is_dir() and x.name.startswith(f"r{p:02d}-")
                         and x.name[4:] not in names and "fixed" not in x.name)
        if not new:
            return fallback("no trainable groups in the pack", round=p, train=tinfo)
        info = {"round": p, "played_by": int(rows[p]["played_by"]), "layout": layout, "layout_differs": layout_diffs,
                "train": tinfo, "digest_dropped": dropped, "benches_not_configured": skipped, "tries": len(new)}
        self.feed.emit("pending_round", used=True, state_round=k, **info)
        return {"round": p, "played_by": int(rows[p]["played_by"]), "data": new, "info": info}

    async def mg_carry(self, number, sets, probes):
        """mg.carry: round ``number`` of the resumed run (the source's read-only final round) trains now from its
        train benches in the source's output (``<resume>/arena/trials/rl-loop/rl/r<NN>-<set>``, collected at the
        source's normal end), as the round loop would have: rule (d) exclusions, mg_select, breakage against the base
        round, then the update and its KL guard. Returns (stop, served, row)."""
        from .rl_loop import usable

        mg = self.cfg["mg"]
        src = Path(mg["resume"]) / "arena" / "trials" / "rl-loop" / "rl"
        plays = {}
        for name, s in sets.items():
            if s["role"] != "train":
                continue
            folder = src / f"r{number:02d}-{name}"
            if not (folder / "level-bench.json").is_file():
                continue  # not played that round (a spent pool)
            report = json.loads((folder / "level-bench.json").read_text())
            rows = [r for r in self.game_rows([r for r in report["attempts"] if usable(r)], s["game"])
                    if restored(r) and level_of(r) is not None]
            plays[name] = (folder, rows, len(report["attempts"]))
        if not plays:
            raise RuntimeError(f"carry: no round-{number} train benches under {src}")
        row = {"round": number, "carried_from": str(src), "played_by": number - 1}
        floor = float(mg.get("min_usable", 0.75))
        low = {n: round(len(rows) / a, 3) for n, (_, rows, a) in plays.items() if a and len(rows) / a < floor}
        if low:
            row["excluded"] = low
        new, tinfo = [], {}
        for n, (folder, rows, _) in plays.items():
            if n in low:
                continue
            sel, tinfo[n] = self.mg_select(folder, rows, n)
            tinfo[n]["clears"] = f"{sum(1 for r in rows if clear_tokens(r))}/{len(rows)}"
            new += [(*x, n) for x in sel]
        row["train"] = tinfo
        if mg.get("breakage") and getattr(self, "mg_base_play", None):
            row["play_stats"] = self.mg_play_stats([r for n, (_, rows, _) in plays.items() if n not in low for r in rows])
            row["breakage"] = self.mg_broken(row["play_stats"], self.mg_base_play)
        if row.get("breakage") or not new:
            row["stop"] = "carry_breakage" if row.get("breakage") else "carry_no_groups"
            self.feed.emit("carry", **row)
            return row["stop"], False, row
        if mg.get("warn"):
            if number in getattr(self, "mg_fixed", {}):
                row["shortening"] = self.mg_shortening(number)
            row["warning"] = self.mg_warning(row)
            if row["warning"]:
                self.mg_set_lr(0.5, row, f"early warning: {row['warning']}")
        update_row, serve = await self.mg_update(number, new, probes)
        stop, served = await self.mg_after_update(number, row, update_row, serve, load=False)
        self.feed.emit("carry", **row)
        return stop, served, row

    async def mg_lr_probe(self, pool, fixed, tests, probes, start):
        """P1 (sc25 plan, 19 Oct): trainer only, no play. Per rate: reset to the base, then one update of
        ``max_groups`` recorded groups (mg_seed's selection under this run's reward; the same groups in the same order
        for every rate) at ``l2_starts_per_step`` groups per optimizer step, the held-out probes scored after every step
        (``update_step``: KL to base, sampled-token log-ratio). A rate starts only with ``min_seconds`` left; a rate
        finished in a resumed run (ARC_RL1_RESUME) is skipped."""
        mg, spec = self.cfg["mg"], self.cfg["mg"]["lr_probe"]
        rates = [float(x) for x in spec["rates"]]
        # mg.nextboard dose probe (envs arm): one rate, one arm per aux weight (the same groups, reset between arms)
        weights = [float(w) for w in spec.get("nb_weights") or []] if mg.get("nextboard") else []
        arms = [(rates[0], w) for w in weights] if weights else [(lr, None) for lr in rates]
        key = lambda lr, w: f"{lr}" if w is None else f"{lr}@nb{w}"
        done = set()
        for root in filter(None, os.environ.get("ARC_RL1_RESUME", "").split(",")):
            feed = Path(root) / "arena/trials/rl-loop/rl/feed.jsonl"
            for line in feed.read_text().splitlines() if feed.is_file() else []:
                e = json.loads(line) if line.strip() else {}
                if e.get("kind") == "lr_probe_rate" and e.get("ok"):
                    done.add(key(float(e["lr"]), e.get("nb_weight")))
        seed, info = self.mg_seed(pool, fixed, tests)  # before the trainer load: a data fault costs no card time
        self.feed.emit("mg_seed", trained=len(seed), sets=info)
        if not seed:
            from .rl_loop import StopLoop

            raise StopLoop("no_seed_tries", info)
        await start()
        out = {"rates": {}, "skipped": []}
        turn = dict(getattr(self, "mg_turn", {}))
        mean = lambda d: round(sum(d.values()) / len(d), 6) if d else None
        for lr, nbw in arms:
            left = self.deadline - time.time()
            if key(lr, nbw) in done or left < float(spec.get("min_seconds", 0)):
                out["skipped"].append({"lr": lr, "nb_weight": nbw,
                                       "why": "done" if key(lr, nbw) in done else f"{left:.0f} s left"})
                self.feed.emit("lr_probe_skip", lr=lr, why=out["skipped"][-1]["why"])
                continue
            self.cfg["lr"] = lr
            self.mg_turn = dict(turn)  # mg_update spends the selection's per-reply credit; every rate gets it (P1 fix)
            await self.trainer.must("reset", lr=lr)
            await self.trainer.must("score", windows=probes, label="base")
            if spec.get("from_state"):  # a probe from a trained state (GSPO dose from mra r05): the state
                # (adapter + optimizer) is loaded after the base reference, and KL-to-previous is held at it
                await self.trainer.must("load_checkpoint", path=spec["from_state"])
                await self.trainer.must("score", windows=probes, label="from_state", hold_previous=True)
                await self.trainer.must("accept_reference")
            if nbw is not None:
                mg["nextboard"]["weight"] = nbw
            began = time.time()
            merged, _ = await self.mg_update(1, seed, probes)
            steps = [{"step": s["step"], "kl_base": mean(s.get("kl_base") or {}), "logratio_base": mean(s.get("logratio_base") or {}),
                      "kl_previous": mean(s.get("kl_previous") or {}), "lr": s.get("lr"), "lr_mult": s.get("lr_mult"),
                      "loss": s.get("loss"), "grad_norm": s.get("grad_norm"), "update_norm": s.get("update_norm"),
                      "trained_tokens": s.get("trained_tokens"), "train_seconds": s.get("train_seconds")}
                     for s in merged.get("steps") or []]
            row = {"lr": lr, "ok": bool(merged.get("ok")), "steps": steps, "seconds": round(time.time() - began, 1),
                   "balance": merged.get("mg_balance"), "trained_tries": merged.get("trained_tries")}
            if nbw is not None:  # the held-out next-board read before (base) and after this arm's update
                row["nb_weight"] = nbw
                row["nextboard"] = merged.get("nextboard")
                row["nb_before"] = merged.get("nextboard_read")
                nb = mg["nextboard"]
                if nb.get("read"):
                    after = await self.trainer.must("nb_score", timeout=int(self.cfg["update_timeout"]), file=nb["read"],
                                                    max_tokens=int(nb.get("max_tokens", 16384)))
                    row["nb_after"] = {k: v for k, v in after.items() if k != "ok"}
            smoke = mg.get("reload_check") and not out["rates"]  # mg.reload_check: the first arm is the
            # run's own first-phase smoke: load, update, checkpoint, park, reload, rescore, one more step
            if smoke:  # checkpoint save, park, reload (unparks), rescore: the reloaded adapter scores as saved
                async def kl(label):
                    got = (await self.trainer.must("score", windows=probes, label=label, hold_previous=True))["windows"]
                    return {n: float(v["kl_base"]) for n, v in got.items() if (v or {}).get("kl_base") is not None}

                # hprobe-v2 (23 Oct): 1.13e-3 against the fixed 1e-3 bar; as mg_reload_check, the bar is this state's
                # measured repeat floor (two scores before the save) x3 + RELOAD_EPS, under RELOAD_CEIL
                before, second = await kl("reload-pre1"), await kl("reload-pre2")
                self.checkpoint(1, force=True)
                await self.save_checkpoint()
                park = await self.trainer.call("park")
                state = Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "state" / "r01"
                await self.trainer.must("load_checkpoint", path=str(state))
                again = await kl("reload")
                names = [n for n in before if n in second and n in again]
                floor = max([abs(before[n] - second[n]) for n in names] or [0.0])
                diff = max(abs(again[n] - before[n]) for n in names) if names else None
                bar = min(RELOAD_CEIL, 3 * floor + RELOAD_EPS)
                small = [t for t in seed if t[3] == seed[0][3]]  # one group after the reload
                after, _ = await self.mg_update(2, small, probes)
                row["reload"] = {"park": {k: park.get(k) for k in ("ok", "parked", "seconds", "card_used_gib") if k in park},
                                 "kl_base_before": before, "kl_base_reloaded": {n: again[n] for n in names},
                                 "max_diff": diff, "repeat_floor": floor, "bar": round(bar, 6),
                                 "ok": diff is not None and len(names) == len(before) and diff <= bar,
                                 "update_after_reload": bool(after.get("ok")), "after_steps": len(after.get("steps") or [])}
                row["ok"] = row["ok"] and row["reload"]["ok"] and row["reload"]["update_after_reload"]
            out["rates"][key(lr, nbw)] = row
            self.feed.emit("lr_probe_rate", **row)
            if smoke and not row["ok"]:
                from .rl_loop import StopLoop

                raise StopLoop("reload_check_failed", row.get("reload") or {})
        self.feed.emit("lr_probe", **out)
        return out

    KNOBS = ("active_kv_fill", "gate_overcommit", "active_tries", "min_queued", "trainer_fast", "explore_kv_share",
             "read_every", "read_per_start")  # the last two: v7 (mg.v7), held-out read cadence and size
    # v5 in-place reconfigure: applied once per new value at a round boundary (a value the file keeps
    # is not re-applied, so the dose controller keeps moving lr between edits); rollback = {"to": "rNN", "id": ..}
    SETTING_KNOBS = ("lr", "kl_target", "kl_ramp", "max_rounds", "rollback", "target_kl")

    async def mg_reconfigure(self, number, played_by, probes):
        """The live knobs file's settings and rollback (``SETTING_KNOBS``); the restored state's round on a rollback."""
        mg, path = self.cfg["mg"], os.environ.get("ARENA_KNOBS")
        if not path or not Path(path).is_file():
            return None
        try:
            knobs = json.loads(Path(path).read_text())
        except ValueError:
            return None  # mg_knobs reported it
        seen = self.__dict__.setdefault("mg_knobs_seen", {})
        new = {k: knobs[k] for k in self.SETTING_KNOBS if k in knobs and seen.get(k) != knobs[k]}
        if isinstance(new.get("rollback"), dict) and int(new["rollback"].get("after_round") or 0) >= number:
            new.pop("rollback")  # {"after_round": N}: at the first boundary after round N (not marked seen before)
        if not new:
            return None
        back, row = None, {"round": number}
        if "rollback" in new:
            spec = new["rollback"] or {}
            back = await self.mg_rollback_to(number, str(spec.get("to")), played_by, probes, spec.get("why") or "knob")
            if back is not None:  # the look-ahead's adapter is no longer an ancestor: its tries would never train
                await self.mg_ahead_cancel("knob rollback")
            row["rollback"] = {"to": spec.get("to"), "applied": back is not None}
        if "lr" in new:
            row["lr"] = [float(self.cfg["lr"]), float(new["lr"])]
            self.cfg["lr"] = float(new["lr"])
        if "kl_ramp" in new and isinstance(new["kl_ramp"], dict) and mg.get("kl_ramp"):
            mg["kl_ramp"] = {**mg["kl_ramp"], **new["kl_ramp"]}
            row["kl_ramp"] = mg["kl_ramp"]
        if "kl_target" in new:
            row["kl_target"] = [mg.get("kl_target"), float(new["kl_target"])]
            mg["kl_target"] = float(new["kl_target"])
            if getattr(self, "mg_ramp_state", None):
                self.mg_ramp_state["target"] = float(new["kl_target"])
        if "target_kl" in new:  # standard dosing's early-stop target (stop at 1.5 x)
            row["target_kl"] = [mg.get("target_kl"), float(new["target_kl"])]
            mg["target_kl"] = float(new["target_kl"])
        if "max_rounds" in new:
            row["max_rounds"] = [mg.get("max_rounds"), int(new["max_rounds"])]
            mg["max_rounds"] = int(new["max_rounds"])
            mg["updates"] = max(int(mg.get("updates", 8)), int(mg["max_rounds"]) - 1)
        seen.update(new)
        self.feed.emit("reconfigure", **row)
        return back

    @staticmethod
    def mg_recorded_probes(source, round_):
        """The probes' KL to base the run recorded right after the update that made state r<round_> (its round row),
        or None (no feed, a base state, or no such update)."""
        feed = Path(str(source)) / "arena" / "trials" / "rl-loop" / "rl" / "feed.jsonl"
        if not round_ or not feed.is_file():
            return None
        found = None
        for line in feed.read_text().splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("kind") == "round" and e.get("round") == round_ and e.get("accepted_updates") is not None:
                found = {n: p.get("kl_base") for n, p in ((e.get("update") or {}).get("probes") or {}).items()
                         if p.get("kl_base") is not None}
        return found or None

    def mg_state_check(self, label, recorded, scored):
        """First-phase gate: a restored state rescored on the probes must give the KL to base its run
        recorded after that update, within mg.resume_check_tol (default 5e-3); a mismatch stops the run."""
        windows = (scored or {}).get("windows") if isinstance(scored, dict) else None
        common = sorted(set(recorded or {}) & set(windows or {})) if isinstance(windows, dict) else []
        row = {"label": label, "probes": len(common)}
        if isinstance(scored, dict) and scored.get("stand_in"):
            row["skipped"] = "stand-in trainer on a real state"
        elif common:
            diffs = {n: abs(float(windows[n].get("kl_base") or 0.0) - float(recorded[n])) for n in common}
            tol = float(self.cfg["mg"].get("resume_check_tol") or 5e-3)
            row.update(max_diff=round(max(diffs.values()), 6), tol=tol, ok=max(diffs.values()) <= tol,
                       recorded=recorded, rescored={n: windows[n].get("kl_base") for n in common})
        self.feed.emit("state_check", **row)
        if row.get("ok") is False:
            raise RuntimeError(f"{label} state check: KL to base differs from the recorded value by {row['max_diff']}")
        return row

    async def mg_rollback_to(self, number, to, played_by, probes, why):
        """In-place rollback to this run's kept state ``to`` (v5): checks first (the state's files and its own mg_state
        snapshot, an earlier state than the served one); then the trainer loads it, its adapter is served, and the
        counters, lr and ramp come from the snapshot; the next update reruns the reload check. Refused: an alert, None."""
        mg = self.cfg["mg"]
        if getattr(self, "mg_own_copy", None) is not None:  # S2 resume_skip: this run's restored state, copied
            await self.mg_own_copy
        # this run's states first, then the chain's earlier legs (ARC_RL1_RESUME, latest first): after a ceiling
        # handoff the kept states before the handoff live in the earlier leg's folder
        roots = [os.environ.get("ARC_VALIDATION_OUTPUT") or "/nonexistent"] + \
            [x for x in os.environ.get("ARC_RL1_RESUME", "").split(",") if x][::-1]
        folder = next((Path(r) / "state" / to for r in roots if (Path(r) / "state" / to / "mg_state.json").is_file()),
                      Path(roots[0]) / "state" / to)
        snap, problems = None, []
        if not re.fullmatch(r"r\d{2}", to):
            problems.append(f"bad state name {to!r}")
        elif not ((folder / "master.safetensors").is_file() or (folder / "state.json").is_file()):
            problems.append(f"{folder}: no checkpoint")
        elif not (folder / "mg_state.json").is_file():
            problems.append(f"{folder}: no mg_state snapshot")
        else:
            snap = json.loads((folder / "mg_state.json").read_text())
            if Path(str(snap.get("checkpoint"))).name != to or int(snap.get("played_by", -1)) != int(to[1:]):
                problems.append(f"snapshot is not {to}'s: checkpoint {snap.get('checkpoint')}, played_by {snap.get('played_by')}")
            if int(to[1:]) >= int(played_by):
                problems.append(f"{to} is not earlier than the served state (played_by {played_by})")
        if problems:
            self.feed.emit("alert", why="rollback refused", round=number, to=to, problems=problems)
            return None
        single = bool(mg.get("single_card"))
        if single:
            await self.server.sleep()
        info = {"to": to, "why": why, "from_played_by": played_by, "folder": str(folder),
                "load": await self.trainer.must("load_checkpoint", path=str(folder))}
        target_kl = mg.get("kl_control") == "target"
        scored = await self.trainer.must("score", windows=probes, label="rollback", hold_previous=target_kl)
        if target_kl:  # KL-to-previous reference = the restored state
            await self.trainer.must("accept_reference")
        self.mg_state_check("rollback", self.mg_recorded_probes(folder.parent.parent, int(to[1:])), scored)
        serve = str(self.work / "adapters" / f"rollback-r{number:02d}-{to}")
        await self.trainer.must("save", path=serve)
        if single:
            info["card_phase"] = await self.mg_card_play()
        await self.mg_serve(serve, int(to[1:]))
        self.serving_adapter, self.keep_checkpoint = serve, to
        self.policy_uid = state_uid(snap)
        back = int(to[1:])
        self.mg_served = [x for x in getattr(self, "mg_served", []) if int(x[1]) <= back] or [(to, back)]
        own = Path(os.environ.get("ARC_VALIDATION_OUTPUT") or "/nonexistent") / "state" / to
        if folder != own and os.environ.get("ARC_VALIDATION_OUTPUT"):  # an earlier leg's state: this run's own copy,
            self.checkpoint(back, force=True)  # so mg_state (and a restart after a handoff) points inside this run
            await self.save_checkpoint()
            if own.is_dir() and not (own / "mg_state.json").exists():
                (own / "mg_state.json").write_text(json.dumps({**snap, "checkpoint": str(own), "policy_uid": self.policy_uid}))
        self.mg_accepted, self.mg_steps_done = int(snap.get("accepted", 0)), int(snap.get("steps_done") or 0)
        self.kl_trips, self.mg_step_rollbacks = snap.get("kl_trips", 0), snap.get("step_rollbacks", 0)
        self.mg_behaviour_trips, self.mg_success_streak = snap.get("behaviour_trips", 0), snap.get("success_streak", 0)
        if snap.get("lr"):
            self.cfg["lr"] = float(snap["lr"])
        if mg.get("kl_ramp") and snap.get("kl_ramp"):
            self.mg_ramp_state = dict(snap["kl_ramp"])
            mg["kl_target"] = float(self.mg_ramp_state["target"])
        elif snap.get("kl_target") is not None:
            mg["kl_target"] = float(snap["kl_target"])
        self.mg_last_dose = tuple(snap["last_dose"]) if snap.get("last_dose") else None
        self.mg_reload_checked = False  # the next update saves, reloads and rescores (first-phase gate)
        info.update(lr=float(self.cfg["lr"]), kl_target=mg.get("kl_target"), accepted=self.mg_accepted)
        self.feed.emit("rollback", round=number, **{k: v for k, v in info.items() if k != "card_phase"})
        return back

    def mg_knobs(self, number):
        """Live knobs (adopt at a round boundary, logged): ARENA_KNOBS (the Modal wrapper copies the run's
        arena/live/knobs.json there) read before each round; play concurrency (kv_fill, request-gate overcommit,
        attempt slots, time-share min_queued) and the trainer paths every op runs under (trainer_fast). Same starts,
        seeds and caps: only concurrency and kernels change."""
        from . import level_bench

        mg = self.cfg["mg"]
        if self.trainer is not None and mg.get("trainer_fast") is not None:
            self.trainer.always = dict(mg["trainer_fast"])
        path = os.environ.get("ARENA_KNOBS")
        if not path or not Path(path).is_file():
            return
        try:
            knobs = json.loads(Path(path).read_text())
        except ValueError as exc:
            self.feed.emit("knobs", round=number, error=str(exc)[:200])
            return
        changed, refused = {}, sorted(k for k in knobs if k not in self.KNOBS)
        for key in self.KNOBS:
            if key not in knobs:
                continue
            if key == "trainer_fast" and knobs[key] == getattr(self, "mg_tf_refused", None):
                continue  # the knob gate reverted this value: not re-applied until the file changes
            holder = (mg.setdefault("timeshare", {}) if key == "min_queued" else
                      mg.setdefault("v7", {}) if key in ("read_every", "read_per_start") else mg)
            if key == "explore_kv_share":  # envs 22 Oct (RL review): explorer KV step-down; live plays finish, launches obey it
                ex = getattr(self, "mg_explorers", None)
                if ex is None or not isinstance(mg.get("explore"), dict) or not 0 < float(knobs[key]) <= 1:
                    refused.append(key)
                    continue
                if float(ex.cfg["kv_share"]) != float(knobs[key]):
                    changed[key] = [float(ex.cfg["kv_share"]), float(knobs[key])]
                    ex.cfg["kv_share"] = mg["explore"]["kv_share"] = float(knobs[key])
                continue
            if holder.get(key) != knobs[key]:
                changed[key] = [holder.get(key), knobs[key]]
                if key == "trainer_fast":  # checked by mg_trainer_fast_gate before the round's update
                    self.mg_tf_change = (holder.get(key), knobs[key])
                holder[key] = knobs[key]
        for pool in (level_bench.SHARED_SLOTS, getattr(level_bench, "SHARED_SLOTS2", None)):
            if pool is not None:
                if mg.get("active_kv_fill"):
                    pool.kv_fill = float(mg["active_kv_fill"])
                if mg.get("active_tries") and pool is level_bench.SHARED_SLOTS:
                    pool.slots = int(mg["active_tries"])
        if self.trainer is not None:
            self.trainer.always = dict(mg.get("trainer_fast") or {})
        if changed or refused:
            self.feed.emit("knobs", round=number, changed=changed, refused=refused)

    async def mg_trainer_fast_gate(self, number, probes):
        """Knob gate: a changed trainer_fast knob rescores the probes at the
        same weights (the KL-to-previous reference held). The perturb control (rl1-prof-mfup1) put any numerics
        change at KL 0.0255-0.0286 (bf16 routing flips); above mg.knob_gate_kl (0.035) the new path is treated as a
        fault: the previous trainer_fast is restored and the value is not re-applied until the knobs file changes."""
        change = self.__dict__.pop("mg_tf_change", None)
        if not change or self.trainer is None:
            return None
        mg = self.cfg["mg"]
        old, new = change
        limit = float(mg.get("knob_gate_kl") or 0.035)
        # single card: the server holds the card at a round top (ovf3r2 OOM 23 Oct); the trainer takes it for the
        # rescore and hands it back with the served adapter
        swap = bool(mg.get("single_card")) and not getattr(self.server, "asleep", False)
        if swap:
            await self.server.sleep()
        scored = await self.trainer.must("score", windows=probes, label="knob", hold_previous=True)
        if swap:
            await self.mg_card_play()
            if self.serving_adapter is not None:
                await self.server.load("policy", self.serving_adapter)
        windows = scored.get("windows") if isinstance(scored.get("windows"), dict) else {}
        kls = {n: w.get("kl_previous") for n, w in windows.items() if isinstance(w, dict) and w.get("kl_previous") is not None}
        worst = max(kls.values()) if kls else None
        row = {"round": number, "from": old, "to": new, "kl_previous": kls, "max": worst, "limit": limit,
               "ok": worst is not None and worst <= limit}
        if not row["ok"]:
            mg["trainer_fast"] = old
            self.trainer.always = dict(old or {})
            self.mg_tf_refused = new
            row["reverted"] = True
        self.feed.emit("knob_gate", **row)
        return row

    def mg_pick(self, number, name, s, pool):
        """Round ``number``'s training starts and tries for set ``name``: ``k`` starts per round from its pool
        (per-set ``train_starts`` / ``train_tries`` over mg's), in pool order. With mg.no_repeat (rl-plan: no start
        trained twice) a spent pool drops the set from training (None) instead of wrapping around."""
        mg = self.cfg["mg"]
        bank = self.mg_pick_bank(number, name, s, pool)
        if mg.get("harvest"):  # from mg.harvest.switch_round: the in-run harvest's starts, the bank fills a short pool
            from .harvest import pick

            got = pick(self, number, name, s, bank)
            if got is not None:
                return got
        return bank

    def mg_curriculum_init(self, sets, pool, screen):
        """One curriculum unit per training set: prior from the screen's base tries, starts from its pool (a fresh
        set: one fresh start), cap the set's."""
        from .curriculum import DEFAULTS, Curriculum

        cc = self.cfg["mg"]["curriculum"]
        cur = Curriculum({k: v for k, v in cc.items() if k in DEFAULTS}, cc.get("shares") or {"sets": 1.0})
        self.mg_unit_set = {}
        for name, s in sets.items():
            if s["role"] != "train":
                continue
            # multi-game (envs 22 Oct): a set carries its source (the mix shares), family and probe prior
            # (``prior``: {clears, tries, median_to_clear} from the env table) when no screen measured it
            m = ((screen or {}).get("sets", {}).get(name, {}).get("metrics") or s.get("prior") or {})
            tries, clears = int(m.get("tries") or 0), int(m.get("clears") or 0)
            t_b = m.get("median_to_clear")
            ref = s.get("ref") or {}
            if not (screen or {}).get("sets", {}).get(name) and (ref.get("measured") or ref.get("derived")) and ref.get("tokens"):
                t_b = ref["tokens"]  # the reward's T_b (envs refs table), never the prior's 64,000 no-data cap (v6g: sb26-L1
                # clears at ~3K tokens all hit z_hi, so an 88%-clear unit out-scored a 50/50 one ~3x: 11 of 59 groups)
            u = cur.add_unit(s["game"], int(s["level"]), s.get("source") or "sets", s.get("family") or s["game"],
                             clears, max(0, tries - clears),
                             [t_b] if t_b else None,
                             cap=int(s.get("cap") or self.cfg["mg"]["cap"]))
            self.mg_unit_set[u.key] = name
            u.starts = ([{"fresh": True}] if s.get("fresh") else
                        [{"attempt": str(a), "request": int(q)} for a, q in pool.get(name, [])])
        missing = []
        for key, name in self.mg_unit_set.items():  # every unit's reward needs a token and an action reference
            s = sets[name]
            try:
                ref = self.mg_level_ref(name, s["game"][:4], int(s["level"]))
            except (IndexError, KeyError, OSError, ValueError) as error:
                missing.append(f"{name}: {type(error).__name__}")
                continue
            if not ref.get("tokens") or not (ref.get("actions") or ref.get("baseline")):
                missing.append(name)
        if missing and not cc.get("allow_no_ref"):
            raise ValueError(f"mg.curriculum: {len(missing)} training sets lack reward references: {missing[:8]}")
        self.feed.emit("mg_curriculum_units", units={k: {"set": n, "p": round(cur.units[k].mean, 3),
                                                          "starts": len(cur.units[k].starts)}
                                                      for k, n in self.mg_unit_set.items()})
        return cur

    def mg_curriculum_jobs(self, number, model, sets, pool):
        """The round's training plays from the curriculum's plan: per set, its groups x group_size tries (a fresh
        set: tries in one bench; a restored set: one start per group, from the harvest when on, else the bank)."""
        from .harvest import pick as harvest_pick, wants as harvest_wants

        mg, cur = self.cfg["mg"], self.mg_curr
        train_sampling = mg.get("train_sampling")  # training plays only, as train_jobs
        seed = int(mg["curriculum"].get("seed", 0))
        g = int(cur.cfg["group_size"])
        # starts each restored unit can supply this round (harvest pool under max_uses + its bank); a unit without
        # any is left out of the plan, and a unit is never given more groups than it has starts (envs 22 Oct: a spent
        # pool used to keep its unit eligible, and the round lost those groups)
        limit, exclude = self.mg_curriculum_avail(number, sets, pool)
        picks = cur.plan(number, seed=seed, exclude=exclude, limit=limit)
        jobs, names, plan, lost = [], [], {}, []
        for attempt in range(3):  # a unit that still yields no start: its groups are planned again elsewhere
            count = {}
            for p in picks:
                count[p["unit"]] = count.get(p["unit"], 0) + 1
            short = 0
            for key, n in count.items():
                name = self.mg_unit_set.get(key)
                if name is None or name in plan:
                    short += n
                    exclude.add(key)
                    continue
                s, u = sets[name], cur.units[key]
                spec = {**s, "cap": u.cap} if u.cap and int(u.cap) != int(s.get("cap") or mg["cap"]) else s
                if s.get("fresh"):
                    st, tries = [], n * g
                else:
                    bank = self.mg_pick_bank(number, name, {**s, "train_starts": n}, pool)
                    got = harvest_pick(self, number, name, s, bank, k=n, tries=g) if mg.get("harvest") else None
                    st, tries = (got[0], g) if got else ((bank[0] if bank else []), g)
                    st = list(st)[:n]
                    if len(st) < n:
                        short += n - len(st)
                        exclude.add(key)
                        lost.append({"unit": key, "groups": n - len(st)})
                    if not st:
                        continue
                self.mg_round_starts.setdefault(number, {})[name] = st
                jobs.append(self.mg_play(f"r{number:02d}-{name}", spec, st, tries, model=model, seed=None,
                                         sampling=train_sampling, harvest=harvest_wants(mg, sets, s)))
                names.append(name)
                plan[name] = {"groups": n if s.get("fresh") else len(st),
                              "tries": (n * g if s.get("fresh") else len(st) * g), "cap": spec.get("cap"),
                              "tier": u.tier}
            if not short:
                break
            picks = cur.plan(number, seed=seed, exclude=exclude | {k for k in cur.units if self.mg_unit_set.get(k) in plan},
                             limit=limit, groups=short, salt=f"+{attempt + 1}")
        if mg.get("explore") and jobs:  # longest expected groups first: long levels start at once
            def expect(name):
                u = cur.units.get(cur.key(sets[name]["game"], int(sets[name].get("level") or 1)))
                med = u.median_tokens() if u is not None else None
                return plan[name]["tries"] * (med or 0.5 * (plan[name]["cap"] or mg.get("cap") or 64000))
            order = sorted(range(len(jobs)), key=lambda i: -expect(names[i]))
            jobs, names = [jobs[i] for i in order], [names[i] for i in order]
            for i, n in enumerate(names):
                plan[n]["order"] = i
        groups = sum(v["groups"] for v in plan.values())
        self.feed.emit("mg_curriculum", round=number, plan=plan, why=sorted({p["why"] for p in picks}),
                       units=cur.summary(), groups=groups, groups_planned=int(cur.cfg["groups"]), lost=lost or None,
                       by_source=self.mg_curriculum_by_source(plan, sets))
        if groups < int(cur.cfg["groups"]):
            self.feed.emit("alert", why="curriculum short", round=number, groups=groups,
                           planned=int(cur.cfg["groups"]))
        return jobs, names

    def mg_read_norm(self, number, played_by, fixed_rows, sets, row):
        """Per-set held-out tries in the round row (``fixed_sets``: [tries, clears, RHAE sum, tokens]) and, once a
        base read exists, ``fixed.norm`` (arena.mg_reads.norm_read: per-game normalised RHAE/10K ratio vs the base,
        with level and source strata). Base tries: rounds the base played (pooled), else ``mg.read_norm.base`` (an
        mg_state.json holding ``read_base``)."""
        from . import mg_reads

        cfg = self.cfg["mg"]["read_norm"] if isinstance(self.cfg["mg"]["read_norm"], dict) else {}
        cur = {n: mg_reads.tries(rows, clear_rhae, clear_tokens) for n, (_, rows, _) in fixed_rows.items() if rows}
        row["fixed_sets"] = {n: [len(v), sum(t[2] for t in v), round(sum(t[0] for t in v), 3), sum(t[1] for t in v)]
                             for n, v in cur.items()}
        base = getattr(self, "mg_read_base", None)
        if base is None and cfg.get("base"):
            base = self.mg_read_base = json.loads(Path(cfg["base"]).read_text()).get("read_base") or {}
        if played_by == 0 and not self.mg_init:
            base = self.mg_read_base = dict(base or {})
            for n, v in cur.items():
                base[n] = list(base.get(n, [])) + mg_reads.compact(v)
            row["fixed"]["norm"] = {"base_round": True, "sets": len(cur)}
            return
        if not base:
            return
        meta = {n: {"game": sets[n]["game"], "level": sets[n].get("level"), "source": sets[n].get("source")}
                for n in cur if n in sets}
        try:
            row["fixed"]["norm"] = mg_reads.norm_read(cur, {n: mg_reads.expand(v) for n, v in base.items()}, meta,
                                                      lam=float(cfg.get("lam", 1.0)),
                                                      n_boot=int(cfg.get("n_boot", 1000)), seed=number)
        except (ValueError, ZeroDivisionError, KeyError) as error:  # a read summary never stops the run
            row["fixed"]["norm"] = {"error": repr(error)[:300]}

    def mg_curriculum_avail(self, number, sets, pool):
        """({unit key: starts it can supply}, {unit keys with none}) for the restored training units this round:
        the harvest pool's starts under max_uses (from its switch round; ``since`` honoured) plus the set's bank."""
        from .harvest import fresh_enough as harvest_mod_fresh, settings as harvest_settings

        mg, cur = self.cfg["mg"], self.mg_curr
        h = harvest_settings(mg) if mg.get("harvest") else None
        live = h is not None and h["switch_round"] is not None and number >= int(h["switch_round"])
        uses = getattr(self, "mg_harvest_uses", None) or {}
        fresh, stale = {}, 0
        for r in (getattr(self, "mg_harvest", None) or []) if live else []:
            if uses.get(r["id"], 0) < int(h["max_uses"]) and (h["since"] is None or int(r["round"]) >= int(h["since"])):
                if harvest_mod_fresh(h, r, number):
                    fresh[r["target"]] = fresh.get(r["target"], 0) + 1
                else:
                    stale += 1
        if stale:  # explorer design: starts older than mg.harvest.stale updates leave the pool
            self.feed.emit("mg_stale", round=number, expired=stale, fresh=sum(fresh.values()))
        limit, exclude = {}, set()
        for key, name in self.mg_unit_set.items():
            s = sets.get(name) or {}
            if s.get("fresh") or key not in cur.units:
                continue
            n = fresh.get(name, 0) + len(pool.get(name, []) or [])
            limit[key] = n
            if n <= 0:
                exclude.add(key)
        return limit, exclude

    @staticmethod
    def mg_curriculum_by_source(plan, sets):
        out = {}
        for name, v in plan.items():
            src = (sets.get(name) or {}).get("source") or "sets"
            out[src] = out.get(src, 0) + int(v["groups"])
        return out

    def mg_curriculum_observe(self, plays, sets):
        """The round's training groups into the curriculum (reward = the loop's own, RHAE, budget-outs)."""
        cur = self.mg_curr
        cur.decay()
        g, out = int(cur.cfg["group_size"]), {}
        for name, (_, rows, _) in plays.items():
            s = sets.get(name) or {}
            if s.get("role") != "train":
                continue
            for grp in self.mg_groups(rows, g).values():
                cur.observe(s["game"], int(s["level"]), [
                    (bool(clear_tokens(r)), clear_tokens(r), self.mg_reward(r, name), clear_rhae(r),
                     r.get("outcome") == "budget") for r in grp])
            u = cur.units.get(cur.key(s["game"], s["level"]))
            if u is not None:
                mean, spread = cur.predict(u)
                out[name] = {"p": round(u.mean, 3), "reward": round(mean, 3), "spread": round(spread, 3),
                             "lp": round(u.progress(), 3), "state": u.state, "cap": u.cap}
        return out

    def mg_pick_bank(self, number, name, s, pool):
        mg = self.cfg["mg"]
        k, st = int(s.get("train_starts", mg.get("train_starts", 2))), [tuple(x) for x in pool.get(name, [])]
        if st and mg.get("no_repeat"):
            if (number - 1) * k >= len(st):
                self.feed.emit("mg_pool_spent", round=number, name=name, pool=len(st))
                return None
            st = st[(number - 1) * k:number * k]
        elif st:
            st = [st[((number - 1) * k + i) % len(st)] for i in range(min(k, len(st)))]
        tries = int(mg.get("fresh_tries", 8) if s.get("fresh") else s.get("train_tries", mg.get("train_tries", 4)))
        if st and mg.get("fill_tries") and len(st) < k:  # a small pool: more tries per start, the level's dose kept
            tries = round(k * tries / len(st))
        return st, tries

    def mg_fresh_starts(self, k):
        """Per restored set: up to ``k`` starts from its screens' source pools (same enumeration and shuffle as the
        screen) that no screen played, so no loop trained on them. Returns ({set: [(attempt, request)]}, info)."""
        mg, sets = self.cfg["mg"], self.cfg["mg_sets"]
        items = [{"screen": mg["screen"], "root": mg.get("screen_root")}] + list(mg.get("screens") or [])
        specs, used = {}, {}
        for item in items:
            raw, root, pairs = json.loads(Path(item["screen"]).read_text()), item.get("root"), item.get("map") or {}

            def m(p, root=root, pairs=pairs):
                p = str(p)
                for a, b in pairs.items():
                    if p.startswith(a):
                        return b + p[len(a):]
                return p.replace("/tmp/arena/", root.rstrip("/") + "/", 1) if root and p.startswith("/tmp/arena/") else p
            for spec in raw["spec"]["sets"]:
                name = spec["name"]
                if name not in sets or sets[name].get("fresh") or name not in raw["sets"]:
                    continue
                srcs = [m(raw["recorded"][x.split(":", 1)[1]]) if x.startswith("record:") else m(x)
                        for x in spec.get("sources") or []]
                specs[name] = (spec, srcs, raw["spec"].get("pick_seed", 0), [(m(a), int(r)) for a, r in raw["sets"][name]["starts"]])
                used.setdefault(name, set()).update((m(a), int(r)) for a, r in raw["sets"][name]["starts"])
        out, info, keep = {}, {}, mg.get("pick_seed")
        for name, (spec, srcs, seed, screen_starts) in specs.items():
            mg["pick_seed"] = seed
            if sets[name].get("start_half") is not None:  # sc25: the loop's own half of the recordings
                spec = {**spec, "start_half": int(sets[name]["start_half"])}
            try:
                pool, n = self.mg_starts({**spec, "starts": 10 ** 9}, srcs)
            finally:
                mg["pick_seed"] = keep
            fresh = [p for p in pool if p not in used[name]]
            out[name] = fresh[:k]
            info[name] = {"available": n, "screen_used": len(used[name]), "fresh": len(fresh), "taken": len(out[name]),
                          "screen_prefix_match": pool[:len(screen_starts)] == screen_starts}
        return out, info

    async def mg_eval(self):
        """Paired re-eval: the base through a zero-B adapter (the policy's serving path) and an
        adapter, on the same fresh starts and seeds and caps, both arms in flight together under one slot pool."""
        mg = self.cfg["mg"]
        ev, sets = mg["eval"], self.cfg["mg_sets"]
        from . import level_bench
        joint = "adapters" in ev  # joint read: many adapters, one server, the loop's held-out reads
        if joint:
            arms = dict(ev["adapters"])
            expect = os.environ.get("ARC_RL1_EXPECT_EVAL")  # 23 Oct: the launcher's resolved arms, checked before play
            if expect and json.loads(expect) != arms:
                raise RuntimeError(f"eval arms {arms} differ from the launch's {json.loads(expect)}")
            self.feed.emit("eval_arms", arms=arms, expected=bool(expect))
            models = {arm: f"ev-{arm}" for arm in arms}
            if ev.get("base", True):
                arms["base"], models["base"] = None, self.server.model  # the pilots' round-1 path: no LoRA
        else:
            arms = {"base": ev["base_adapter"], "policy": ev["adapter"]}
            models = {arm: f"ev-{arm}" for arm in arms}
        for arm, path in arms.items():
            if path:
                await self.server.load(models[arm], path)
        if os.environ.get("ARC_VALIDATION_OUTPUT"):  # gate item 4: finished tries kept, a restart skips them
            level_bench.TRY_KEEP = Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "partial"
            level_bench.TRY_REUSE = [Path(r) / "partial" for r in os.environ.get("ARC_RL1_RESUME", "").split(",") if r]
        level_bench.SHARED_SLOTS = level_bench.SharedSlots(int(mg["active_tries"]), mg.get("active_kv_fill"), project=mg.get("kv_project"))
        jobs, keys = [], []
        if joint:
            # the loop's fixed reads (half_split): each restored training set's first held_starts screen starts,
            # fixed_tries per set split over its starts; a fresh training set from its level start; seed mg.seed+500
            k, n_tries = int(mg.get("held_starts") or 2), int(ev.get("fixed_tries") or mg["fixed_tries"])
            starts = {n: [tuple(x) for x in self.mg_screen["sets"][n]["starts"][:k]]
                      for n, s in sets.items() if s["role"] == "train" and not s.get("fresh")}
            starts.update({n: [] for n, s in sets.items() if s["role"] == "train" and s.get("fresh")})
            info = {n: len(v) for n, v in starts.items()}
            seed = int(ev.get("seed", int(mg.get("seed", 8000)) + 500))
            per_job = max(1, int(mg["active_tries"]) // len(arms))  # every arm in flight at once (common conditions)
            for name, st in starts.items():
                for arm in arms:  # arms alternate inside each set
                    jobs.append(self.mg_play(f"jr-{arm}-{name}", sets[name], st, max(1, n_tries // max(1, len(st))),
                                             model=models[arm], seed=seed,
                                             concurrency=min(per_job, max(1, len(st)) * max(1, n_tries // max(1, len(st))))))
                    keys.append((name, arm))
        else:
            starts, info = self.mg_fresh_starts(int(ev["starts"]))
            for name, s in sets.items():
                st = [] if s.get("fresh") else starts.get(name) or []
                if not s.get("fresh") and not st:
                    continue
                tries = int(ev["fresh_tries"] if s.get("fresh") else ev["tries"])
                for arm in arms:
                    jobs.append(self.mg_play(f"ev-{arm}-{name}", s, st, tries, model=f"ev-{arm}",
                                             seed=int(ev.get("seed", 9300))))
                    keys.append((name, arm))
        self.feed.emit("eval_starts", sets=info, starts={n: [list(x) for x in v] for n, v in starts.items()},
                       arms={a: str(p) for a, p in arms.items()})
        results = await asyncio.gather(*jobs)
        keep = ("label", "point", "pass", "outcome", "completion_tokens", "level_rows", "restore")
        out = {"arms": arms, "starts": info, "sets": {}}
        for (name, arm), (folder, rows, play) in zip(keys, results):
            every = json.loads((Path(folder) / "level-bench.json").read_text())["attempts"]
            outcomes = {}
            for r in every:
                outcomes[r.get("outcome")] = outcomes.get(r.get("outcome"), 0) + 1
            out["sets"].setdefault(name, {})[arm] = {
                "cap": int(sets[name].get("cap") or mg["cap"]), "metrics": held_metrics(rows), "outcomes": outcomes,
                "rows": [{f: (r.get(f) if f != "restore" else bool(r.get(f))) for f in keep} for r in rows],
                "clear_tokens": [clear_tokens(r) for r in rows], "play": play}
            m = out["sets"][name][arm]["metrics"]
            out["sets"][name][arm]["rhae"] = [clear_rhae(r) for r in rows]
            self.feed.emit("eval_set", set=name, arm=arm, clears=f"{m['clears']}/{m['tries']}",
                           median=m["median_to_clear"], outcomes=outcomes)
        if joint:  # per arm pooled over the read's sets: the pilots' RHAE/10K
            out["pooled"] = {arm: held_metrics([r for (n, a), (_, rows, _) in zip(keys, results) if a == arm for r in rows])
                             for arm in arms}
            self.feed.emit("eval", label="joint_read", pooled=out["pooled"])
        (self.output / "rl" / "eval.json").write_text(json.dumps(out, indent=1, default=str))
        if os.environ.get("ARC_VALIDATION_OUTPUT"):
            dest = Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "eval.json"
            dest.write_text(json.dumps(out, indent=1, default=str))
        self.feed.emit("final", stop="eval", eval={n: {a: v["metrics"] for a, v in d.items()} for n, d in out["sets"].items()},
                       base=None)
        return "eval"

    def mg_early(self):
        """mg5e KL control (PPO target-KL early stopping, Schulman+ 2017 / SB3): per-update target ``kl_target`` over
        the floor; stop the update above 1.5 x target, roll it back above 3 x target."""
        mg = self.cfg["mg"]
        if mg.get("kl_control") != "target":
            return None
        target = float(mg.get("kl_target", 0.01))
        if mg.get("kl_damage"):  # the target only steers lr; an update is cut short only when
            # the next probe is forecast past the damage line, and rolled back only past it (probe excess > kl_damage)
            return {"probe_every": int(mg.get("probe_every", 2)), "floor": float(mg.get("kl_floor", 0.018)),
                    "stop_at": float("inf"), "rollback_at": float(mg["kl_damage"])}
        return {"probe_every": int(mg.get("probe_every", 2)), "floor": float(mg.get("kl_floor", 0.018)),
                "stop_at": 1.5 * target, "rollback_at": 3 * target}

    @staticmethod
    def ramp_trip(row, early, previous_clears, z=2.0):
        """Why the ramp rule counts this round as a trip, or None: the update's own KL rollback (kl_damage), play
        breakage (reply / filler / truncation / clear guards against the base), the early warning (reward-hack
        signs: actions, replies), or the training read's pooled clear rate below the previous round's by more than
        ``z`` binomial standard errors."""
        if early.get("action") == "rollback":
            return "kl_damage"
        if row.get("breakage"):
            return "breakage"
        if row.get("warning"):
            return "warning"
        if (row.get("behaviour") or {}).get("trip"):
            return "behaviour"  # held-out clears down or actions up against the base (mg_behaviour)
        counts = [tuple(int(x) for x in str(v.get("clears", "")).split("/")) for v in (row.get("train") or {}).values()
                  if "/" in str(v.get("clears", ""))]
        now = (sum(c for c, _ in counts), sum(n for _, n in counts)) if counts else None
        if now and now[1] and previous_clears and previous_clears[1]:
            p1, p0 = now[0] / now[1], previous_clears[0] / previous_clears[1]
            pool = (now[0] + previous_clears[0]) / (now[1] + previous_clears[1])
            se = (pool * (1 - pool) * (1 / now[1] + 1 / previous_clears[1])) ** 0.5
            if se > 0 and p0 - p1 > z * se:
                return f"clear_drop {p0:.2f}->{p1:.2f}"
        return None

    def mg_ramp(self, row, early):
        """Ramp rule: KL target starts at ``start``; after a quiet round it is multiplied by
        ``factor`` up to ``cap``; a trip sets it to the last safe target and holds it for ``hold`` quiet rounds (the
        guards roll the update back). Writes mg.kl_target for the next update and ``row["kl_ramp"]``."""
        mg, rp = self.cfg["mg"], self.cfg["mg"]["kl_ramp"]
        st = self.__dict__.setdefault("mg_ramp_state", {"target": float(rp.get("start", 0.01)),
                                                        "safe": float(rp.get("start", 0.01)), "hold": 0, "clears": None})
        used = st["target"]
        trip = self.ramp_trip(row, early, st["clears"], float(rp.get("z", 2.0)))
        if trip:
            st["target"], st["hold"] = st["safe"], int(rp.get("hold", 2))
        else:
            st["safe"] = used
            over = rp.get("overshoot")  # 21 Oct: an update that landed above overshoot x its target holds the target
            excess = early.get("excess")
            if st["hold"] > 0:
                st["hold"] -= 1
            elif over and excess is not None and float(excess) > float(over) * used:
                pass
            else:
                st["target"] = min(float(rp.get("cap", 0.05)), used * float(rp.get("factor", 1.5)))
        counts = [tuple(int(x) for x in str(v.get("clears", "")).split("/")) for v in (row.get("train") or {}).values()
                  if "/" in str(v.get("clears", ""))]
        if counts and not trip:
            st["clears"] = (sum(c for c, _ in counts), sum(n for _, n in counts))
        mg["kl_target"] = st["target"]
        row["kl_ramp"] = {"target_used": used, "excess": early.get("excess"), "trip": trip, "next_target": st["target"],
                          "safe": st["safe"], "hold": st["hold"]}

    def mg_kl_target_control(self, row, update_row):
        """mg5e lr control after an update (PPO's adaptive-KL constants on the step size): a rollback halves lr; an
        update stopped early (or ending above 1.5 x target) divides it by 1.5; a full update under target / 1.5
        multiplies it by 1.5, at most ``lr_max``. Returns "rollback" or None."""
        mg, early = self.cfg["mg"], update_row.get("early") or {}
        if mg.get("dosing") == "standard":  # constant lr after warm-up; probe KL logged, never steers
            kp = [p.get("kl_previous") for p in (update_row.get("probes") or {}).values() if p.get("kl_previous") is not None]
            row["kl_step"] = {"diagnostic": True, "lr": float(self.cfg["lr"]),
                              "lr_effective": float(update_row.get("lr_effective") or self.cfg["lr"]),
                              "excess": round(sum(kp) / len(kp) - float(mg.get("kl_floor") or 0), 5) if kp else None,
                              "early": early or None, "target_kl": mg.get("target_kl")}
            return None
        if mg.get("kl_ramp"):  # the next update's target from this round's guards
            self.mg_ramp(row, early)
        target, excess = float(mg.get("kl_target", 0.01)), early.get("excess")
        lr_eff = float(update_row.get("lr_effective") or self.cfg["lr"])  # mg.lr_schedule: the mean lr sent
        row["kl_step"] = {**early, "lr": float(self.cfg["lr"]), "lr_effective": lr_eff}
        if early.get("action") == "rollback":
            line = mg.get("kl_damage") or f"3 x {target}"
            self.mg_set_lr(0.5, row, f"KL excess {excess} > {line}: rollback")
            return "rollback"
        if mg.get("lr_dose") and early.get("action") == "full" and excess is not None:
            # dose rule (P2's x1.5 steps left U1-U3 ~7x under target): KL ~ lr^2, so the next
            # update aims at the target, at most x3 / 3 per update and never past lr_max
            # with mg.lr_schedule the dose sets the peak: an update at multiplier 1 would land on the target
            prev = getattr(self, "mg_last_dose", None)
            self.mg_dose_lr(row, target, excess, lr_eff, "update", previous=prev)
            self.mg_last_dose = (lr_eff, excess)
        elif early.get("action") == "stop" or (excess is not None and excess > 1.5 * target):
            self.mg_set_lr(1 / 1.5, row, f"KL excess {excess} > 1.5 x {target} after {early.get('steps')} steps")
        elif excess is not None and excess < target / 1.5 and early.get("action") == "full":
            up = min(1.5, float(mg.get("lr_max", self.cfg["lr"])) / float(self.cfg["lr"]))
            if up > 1.0001:
                self.mg_set_lr(up, row, f"KL excess {excess} < {target} / 1.5")
        return None

    def mg_layout_env(self):
        """mg.host_load: the cold load's layout record for this trainer key (every trainer start writes it when absent;
        the host load checks against it). {} without the knob or a trainer key."""
        key = os.environ.get("ARC_RL1_TRAINER_KEY")
        if not self.cfg["mg"].get("host_load") or not key:
            return {}
        root = Path(os.environ.get("ARC_PARK_RECORDS") or "/runs/park-checks")
        return {"RL1_LAYOUT_RECORD": str(root / f"layout-{key}.json")}

    def mg_base_cache_key(self, probes):
        """Start-up cut (c): what the probes' base references depend on (RL 23 Oct: model, adapter init, probe windows,
        prefix chunking, trainer_fast), plus the trainer's code and command (ARC_RL1_TRAINER_KEY) and the routing mode.
        None (no cache) without a trainer key or with a warm start or nudged base."""
        mg = self.cfg["mg"]
        trainer_key = os.environ.get("ARC_RL1_TRAINER_KEY")
        if not trainer_key or self.mg_init:
            return None
        model = Path(self.cfg["model_dir"])
        files = {}
        for name in ("config.json", "model.safetensors.index.json"):
            f = model / name
            files[name] = hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None
        windows = []
        for spec in sorted(probes, key=lambda w: w["name"]):
            rollout = Path(spec["rollout"])
            windows.append({**{k: v for k, v in spec.items() if k != "rollout"},
                            "rollout_sha256": hashlib.sha256(rollout.read_bytes()).hexdigest() if rollout.is_file() else None})
        if any(w["rollout_sha256"] is None for w in windows):
            return None
        settings = {k: self.cfg.get(k) for k in ("path_chunk", "prefix_chunk", "rank", "window", "attention", "loop_lora",
                                                 "model_dir")}
        payload = {"v": 1, "trainer_key": trainer_key, "model": files, "settings": settings, "windows": windows,
                   "trainer_fast": mg.get("trainer_fast"), "served_route": bool(mg.get("served_route")),
                   "nudge": [mg.get("nudge_base"), mg.get("nudge_value")], "base_adapter": mg.get("base_adapter")}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:32]

    async def mg_base_reference(self, probes):
        """The probes' base references (``score`` label base). mg.base_cache: a stored copy under the same key is
        loaded instead and checked by rescoring the shortest probe at the held reference (its KL and log-ratio to the
        loaded base within mg.base_cache_tol, default 2e-4: the reload-check bar); a miss, a refused load or a failed
        check scores afresh and stores."""
        mg = self.cfg["mg"]
        root = mg.get("base_cache")
        if root is True:  # v6g-v6i: true became the relative dir "True" on the container's own disk, so every
            # restart missed (86-126 s); the volume root keeps the copy across legs
            root = os.environ.get("ARC_BASE_CACHE_ROOT") or "/runs/base-cache"
        key = self.mg_base_cache_key(probes) if root else None
        names = sorted(w["name"] for w in probes)
        path = Path(str(root)) / f"{key}.pt" if key else None
        if path is not None and path.is_file():
            began = time.time()
            try:
                got = await self.trainer.must("base_import", path=str(path), key=key, names=names)
                short = min(probes, key=lambda w: Path(w["rollout"]).stat().st_size)
                check = await self.trainer.must("score", windows=[short], label="cache_check", hold_previous=True)
                w = (check.get("windows") or {}).get(short["name"]) or {}
                kl, lr = w.get("kl_base"), w.get("logratio_base")
                # RL review 23 Oct: the reload-check bar (repeat floor ~0), not the KL floor: the base references feed
                # the KL-to-base state checks, so a stale copy must not pass within noise
                limit = float(mg.get("base_cache_tol") or 2e-4)
                worst = max(abs(kl), abs(lr or 0.0)) if kl is not None else None
                row = {"hit": True, "key": key, "windows": got.get("windows"), "check": short["name"], "kl_base": kl,
                       "logratio_base": lr, "limit": limit, "seconds": round(time.time() - began, 1)}
                if worst is not None and worst <= limit:
                    self.feed.emit("base_cache", **row)
                    return row
                self.feed.emit("alert", why="base cache check failed: scoring afresh", **row)
            except Exception as exc:  # noqa: BLE001 - the cache is a speed-up only
                self.feed.emit("alert", why="base cache load failed: scoring afresh", key=key, error=repr(exc)[:300])
            try:  # v6j2: the refused copy (cuda:0) stayed installed and broke the fresh score too
                await self.trainer.must("base_drop", names=names)
            except Exception as exc:  # noqa: BLE001
                self.feed.emit("alert", why="base cache drop failed", key=key, error=repr(exc)[:300])
        began = time.time()
        await self.trainer.must("score", windows=probes, label="base")
        row = {"hit": False, "key": key, "seconds": round(time.time() - began, 1)}
        if path is not None:
            try:
                row["stored"] = await self.trainer.must("base_export", path=str(path), key=key, names=names)
            except Exception as exc:  # noqa: BLE001
                row["store_error"] = repr(exc)[:300]
        if root:
            self.feed.emit("base_cache", **row)
        return row

    async def mg_card_play(self):
        """Single card: the trainer parks (base included) and the server wakes for play."""
        began = time.time()
        park = await self.trainer.must("park", base=True)
        wake = await self.server.wake()
        return {"park_seconds": park.get("seconds"), "wake_seconds": wake, "seconds": round(time.time() - began, 1)}

    async def mg_after_update(self, number, row, update_row, serve, load=True):
        """After a round's update: the KL guard and step control decide; the update is served (its probe scores become
        the KL reference, the checkpoint the rollback target) or rolled back to the served state. Returns
        (stop, served)."""
        mg, stop = self.cfg["mg"], None
        row["update"] = {k: v for k, v in update_row.items() if k not in ("traceback", "windows")}
        if not update_row.get("ok"):
            return (row.setdefault("stop", update_row.get("stop") or "update_failed"), False)
        if update_row.get("skipped"):
            return None, False
        kl = [p.get("kl_base") for p in (update_row.get("probes") or {}).values() if p.get("kl_base") is not None]
        target = mg.get("kl_control") == "target"
        step = self.mg_kl_target_control(row, update_row) if target else self.mg_lr_control(row, update_row)
        guard = self.cfg["l2_kl_stop"] is not None and kl and max(kl) > float(self.cfg["l2_kl_stop"])
        if guard or step == "rollback":
            # mg5e: only the guard (KL to base) counts toward rule (c); step rollbacks have their own limit
            if guard or not target:
                self.kl_trips = getattr(self, "kl_trips", 0) + 1
            else:
                self.mg_step_rollbacks = getattr(self, "mg_step_rollbacks", 0) + 1
            row["kl_guard"] = {"kl_base_max": max(kl) if kl else None, "limit": self.cfg["l2_kl_stop"],
                               "kept_serving": str(self.serving_adapter), "trips": getattr(self, "kl_trips", 0),
                               "step_rollbacks": getattr(self, "mg_step_rollbacks", 0), "step": step}
            if mg.get("rollback"):
                if not getattr(self, "keep_checkpoint", None):
                    raise RuntimeError("rollback configured but no checkpoint of the served state")
                root = Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "state" / self.keep_checkpoint
                row["kl_guard"]["rollback"] = await self.trainer.must("load_checkpoint", path=str(root))
                if not target:  # mg5d bug: the trainer's KL-to-previous reference stayed on the rejected state
                    await self.trainer.must("score", windows=self.mg_probes, label="rollback")
                if guard and step != "rollback":
                    self.mg_set_lr(0.5, row, "guard trip")
            if getattr(self, "kl_trips", 0) >= int(mg.get("kl_trips_stop") or 2):
                stop = "rule_c_kl_guard_twice"
            elif target and getattr(self, "mg_step_rollbacks", 0) >= int(mg.get("step_rollbacks_stop") or 3):
                stop = "rule_c_step_rollbacks"
            return stop, False
        if target:
            await self.trainer.must("accept_reference")
        if load:  # single card: the caller loads it once the server is awake
            row["swap"] = {"load_seconds": await self.mg_serve(serve, number)}
        self.serving_adapter = serve
        self.mg_accepted = getattr(self, "mg_accepted", 0) + 1
        row["accepted_updates"] = self.mg_accepted
        # mg.lr_schedule: the global step count moves with accepted updates only (a rolled-back one is retaken)
        self.mg_steps_done = getattr(self, "mg_steps_done", 0) + len(update_row.get("steps") or [])
        if mg.get("rollback") and mg.get("checkpoint") and os.environ.get("ARC_VALIDATION_OUTPUT"):
            self.keep_checkpoint = f"r{number:02d}"
            keep = max(2, int(mg.get("keep_states") or 2))  # v5: the last k accepted states stay (31 GiB each)
            self.mg_served = (getattr(self, "mg_served", []) + [(self.keep_checkpoint, number)])[-keep:]
        return None, True

    @staticmethod
    def mg_play_stats(rows):
        """A round's play health: mean completion tokens per live request, the share of live replies cut
        at the reply cap, and the share of gameplay requests without an action (tool or format failures)."""
        asked = sum(int(r.get("admitted_requests") or 0) for r in rows)
        live = sum(int((r.get("rollout") or {}).get("live") or 0) for r in rows)
        game = sum(int(r.get("gameplay_requests") or 0) for r in rows)
        think = sum(int((r.get("rollout") or {}).get("think_tokens") or 0) for r in rows)
        filler = sum(int((r.get("rollout") or {}).get("filler") or 0) for r in rows)
        return {"tries": len(rows),
                "filler_k": round(1000 * filler / think, 3) if think else None,
                "reply_tokens": round(sum(int(r.get("completion_tokens") or 0) for r in rows) / asked, 1) if asked else None,
                "cap_share": round(sum(int((r.get("rollout") or {}).get("capped") or 0) for r in rows) / live, 4) if live else None,
                "noaction_share": round(sum(int(r.get("noaction_requests") or 0) for r in rows) / game, 4) if game else None}

    def mg_broken(self, now, base, drop=None, rise=None):
        """Play breakage against the base round: reply length over ``reply_ratio`` x base, or the cap-hit or no-action
        share over ``ratio`` x base and ``abs`` above it; with ``cap_abs`` (c7f) the cap share
        counts when it rose more than that (absolute), with ``clear_drop`` a held-out clear-rate fall above it (``drop``,
        the behaviour guard's) is breakage too. None, or the reasons."""
        g, why = self.cfg["mg"]["breakage"], []
        if now["reply_tokens"] and base["reply_tokens"] and now["reply_tokens"] > float(g.get("reply_ratio", 2.0)) * base["reply_tokens"]:
            why.append(f"reply tokens {now['reply_tokens']} vs base {base['reply_tokens']}")
        for key in ("cap_share", "noaction_share"):
            n, b = now[key], base[key]
            if n is None or b is None:
                continue
            if key == "cap_share" and g.get("cap_abs") is not None:
                if n > b + float(g["cap_abs"]):
                    why.append(f"{key} {n} vs base {b}")
            elif n > float(g.get("ratio", 2.0)) * b and n > b + float(g.get("abs", 0.05)):
                why.append(f"{key} {n} vs base {b}")
        if g.get("filler_ratio") is not None and now.get("filler_k") is not None and base.get("filler_k"):
            if now["filler_k"] >= float(g["filler_ratio"]) * base["filler_k"]:  # pilot c R2: 1.59 x before its loops
                why.append(f"filler {now['filler_k']} per 1K thinking tokens vs base {base['filler_k']}")
        if drop is not None and g.get("clear_drop") is not None and drop > float(g["clear_drop"]):
            why.append(f"held-out clear rate down {round(drop, 3)}")
        if rise is not None and g.get("actions_rise") is not None and rise > float(g["actions_rise"]):
            why.append(f"held-out actions per clear up {round(rise, 3)}")  # pilot a R4: +2.38
        return why or None

    def mg_guard_stop(self, row):
        """The behaviour guard's stop for this round, except a broken round under ``breakage.rollback_first`` (c7f,
        the rollback rule): that one rolls back with lr halved instead of stopping the run."""
        if row.get("breakage") and (self.cfg["mg"].get("breakage") or {}).get("rollback_first"):
            return None
        return (row.get("behaviour") or {}).get("stop")

    def mg_warning(self, row):
        """c7f early warning (pilot c at R2 before its collapse: L3 token ratio 0.62): a trained
        level's held-out token ratio under ``warn.ratio`` or reply length over ``warn.reply_ratio`` x the base round.
        The reasons, or None."""
        warn, why = self.cfg["mg"].get("warn") or {}, []
        for name, s in sorted((row.get("shortening") or {}).items()):
            if s.get("ratio") is not None and s["ratio"] < float(warn.get("ratio", 0.7)):
                why.append(f"{name} token ratio {s['ratio']}")
        now, base = row.get("play_stats") or {}, getattr(self, "mg_base_play", None) or {}
        if now.get("reply_tokens") and base.get("reply_tokens") \
                and now["reply_tokens"] > float(warn.get("reply_ratio", 1.5)) * base["reply_tokens"]:
            why.append(f"reply tokens {now['reply_tokens']} vs base {base['reply_tokens']}")
        rise = (row.get("behaviour") or {}).get("actions_rise")
        if warn.get("actions") is not None and rise is not None and rise > float(warn["actions"]):
            why.append(f"actions per clear up {rise}")  # pilot a: +0.29 at R2, +0.48 at R3 before R4's +2.38
        return why or None

    async def mg_break_rollback(self, number, row, probes):
        """Play breakage: back to the state served before the broken adapter (its checkpoint; r00 is the
        base), lr halved, a step rollback counted; this round's tries never train. The played_by of the restored
        state, or None when no earlier state is kept."""
        mg, stack = self.cfg["mg"], getattr(self, "mg_served", [])
        if len(stack) < 2:
            return None
        stack.pop()
        name, back = stack[-1]
        await self.mg_ahead_cancel("play breakage rollback")  # the server sleeps: no look-ahead try may be live
        await self.server.sleep()
        info = {"to": name, "played_by": back, "why": row.get("breakage"),
                "load": await self.trainer.must("load_checkpoint", path=str(Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "state" / name))}
        if mg.get("kl_control") == "target":  # KL-to-previous reference = the restored state
            await self.trainer.must("score", windows=probes, label="breakage", hold_previous=True)
            await self.trainer.must("accept_reference")
        serve = None
        if back > 0 or getattr(self, "mg_init", False):  # a warm start's r00 is the init state, served
            serve = str(self.work / "adapters" / f"back-r{number:02d}")
            await self.trainer.must("save", path=serve)
        self.keep_checkpoint = name
        self.mg_set_lr(0.5, row, f"play breakage: {row.get('breakage')}")
        self.mg_step_rollbacks = getattr(self, "mg_step_rollbacks", 0) + 1
        info["card_phase"] = await self.mg_card_play()
        if serve is not None:
            await self.mg_serve(serve, back)
        elif self.serving_adapter is not None:
            await self.server.unload(self.mg_name())
        self.serving_adapter = serve
        row["breakage_rollback"] = info
        self.feed.emit("breakage_rollback", round=number, **{k: v for k, v in info.items() if k != "card_phase"})
        return back

    def mg_dose_lr(self, row, target, excess, lr_measured, why, previous=None):
        """lr for the next update from a measured (lr, KL excess) pair under KL ~ lr^2, clamped to x[1/3, 3] of the
        measured lr and to ``lr_max``. ``mg.kl_k_predict`` (21 Oct, k = excess / lr^2 doubled per update in the main
        runs): the next update's k is predicted as k x the last growth k / k_previous, clamped to [1, kl_k_predict];
        ``previous``: the (lr, excess) pair before the measured one."""
        mg = self.cfg["mg"]
        if excess is None:
            return
        cap = mg.get("kl_k_predict")
        if cap and previous and previous[0] and previous[1] and float(previous[1]) > 0 and float(excess) > 0:
            growth = (float(excess) / float(lr_measured) ** 2) / (float(previous[1]) / float(previous[0]) ** 2)
            growth = max(1.0, min(float(cap), growth))
            if growth > 1.0:
                why = f"{why}, k growth x{growth:.2f}"
                excess = float(excess) * growth
        # excess <= 0: no movement measured above the floor, the clamp's limit (preflight pf4: a negative source excess)
        ratio = 3.0 if float(excess) <= 0 else max(1 / 3, min(3.0, (float(target) / float(excess)) ** 0.5))
        want = lr_measured * ratio
        want = min(want, float(mg.get("lr_max", want)))
        factor = want / float(self.cfg["lr"])
        if abs(factor - 1) > 0.05:
            self.mg_set_lr(factor, row, f"dose ({why}): KL excess {excess} at lr {lr_measured:.3g}, target {target}")

    def mg_set_lr(self, factor, row, why):
        old = float(self.cfg["lr"])
        self.cfg["lr"] = old * factor
        row.setdefault("lr_control", []).append({"from": old, "to": self.cfg["lr"], "why": why})
        self.feed.emit("lr_change", **row["lr_control"][-1])

    def mg_lr_control(self, row, update_row):
        """mg5: when one update moves the probes more than ``kl_step`` above the KL floor (mean
        KL to the previous adapter), the next updates run at half the lr; above 3 x ``kl_step``, also a rollback."""
        mg = self.cfg["mg"]
        if not mg.get("kl_step"):
            return None
        prev = [p.get("kl_previous") for p in (update_row.get("probes") or {}).values() if p.get("kl_previous") is not None]
        if not prev:
            return None
        excess = sum(prev) / len(prev) - float(mg.get("kl_floor", 0.018))
        row["kl_step"] = {"kl_previous_mean": round(sum(prev) / len(prev), 5), "excess": round(excess, 5),
                          "lr": float(self.cfg["lr"])}
        if excess > 3 * float(mg["kl_step"]):
            self.mg_set_lr(0.5, row, f"KL step excess {excess:.4f} > 3 x {mg['kl_step']}: rollback")
            return "rollback"
        if excess > float(mg["kl_step"]):
            self.mg_set_lr(0.5, row, f"KL step excess {excess:.4f} > {mg['kl_step']}")
            return "halve"
        return None

    def mg_shortening(self, number):
        """Per trained level (and pooled): held-out median tokens to clear and clear rate this round against the base
        rounds 1-2 (the same seeds every round), with the base-over-now token ratio."""
        hist = getattr(self, "mg_fixed", {})
        out = {}
        for name, rows in (hist.get(number) or {}).items():
            base = [r for k in (getattr(self, "mg_base_rounds", None) or (1, 2)) for r in (hist.get(k) or {}).get(name, [])]
            now, b = held_metrics(rows), held_metrics(base)
            ratio = (round(b["median_to_clear"] / now["median_to_clear"], 3)
                     if b["median_to_clear"] and now["median_to_clear"] else None)
            out[name] = {"clears": f"{now['clears']}/{now['tries']}", "median": now["median_to_clear"],
                         "base_median": b["median_to_clear"], "ratio": ratio, "base_clears": f"{b['clears']}/{b['tries']}"}
        return out

    def mg_behaviour(self, number):
        """mg6 behaviour guard: the served adapter's held-out play against the base rounds: pooled clear
        rate drop, and the geometric mean over levels of median actions to clear (now / base). Trips when the clear
        rate falls more than ``clear_drop`` or actions rise more than ``actions_rise``; the run stops at 2 trips in a
        row or one clear-rate fall over 2 x ``clear_drop``."""
        g = self.cfg["mg"]["behaviour_guard"]
        hist = getattr(self, "mg_fixed", {})
        base_rounds = getattr(self, "mg_base_rounds", None) or (1,)
        now_rows = [r for rows in (hist.get(number) or {}).values() for r in rows]
        base_rows = [r for k in base_rounds for rows in (hist.get(k) or {}).values() for r in rows]
        rate = lambda rows: sum(1 for r in rows if clear_tokens(r)) / len(rows) if rows else None
        ratios = []
        only = set(g.get("levels") or [])  # mg7b: actions over the levels where RHAE moves with actions
        for name, rows in (hist.get(number) or {}).items():
            if only and name not in only:
                continue
            b = [int(r["level_rows"][0].get("actions") or 0) for k in base_rounds for r in (hist.get(k) or {}).get(name, [])
                 if clear_tokens(r)]
            n = [int(r["level_rows"][0].get("actions") or 0) for r in rows if clear_tokens(r)]
            if b and n and statistics.median(b) > 0:
                ratios.append(statistics.median(n) / statistics.median(b))
        drop = (rate(base_rows) - rate(now_rows)) if base_rows and now_rows else 0.0
        rise = math.exp(sum(math.log(x) for x in ratios) / len(ratios)) - 1 if ratios and all(x > 0 for x in ratios) else 0.0
        cap = g.get("actions_rise", 0.15)  # None: no actions rule
        trip = drop > float(g.get("clear_drop", 0.08)) or (cap is not None and rise > float(cap))
        # (mra-v2 R7: actions +0.469, mean RHAE 0.63 vs 0.87): mean RHAE of the held-out clears
        # below ``rhae_floor`` trips too; ``trips_stop`` trips in a row stop the run (default 2)
        mean_rhae = held_metrics(now_rows)["mean_rhae"] if g.get("rhae_floor") is not None and now_rows else None
        if mean_rhae is not None and mean_rhae < float(g["rhae_floor"]):
            trip = True
        self.mg_behaviour_trips = (getattr(self, "mg_behaviour_trips", 0) + 1) if trip else 0
        out = {"clear_drop": round(drop, 3), "actions_rise": round(rise, 3), "trip": trip,
               "streak": self.mg_behaviour_trips}
        if mean_rhae is not None:
            out["mean_rhae"] = mean_rhae
        if self.mg_behaviour_trips >= int(g.get("trips_stop", 2)) or drop > 2 * float(g.get("clear_drop", 0.08)):
            out["stop"] = "behaviour_guard"
        return out

    def mg_rule_f(self, row, played_by):
        """mg5 stop rule (f): after ``rule_f_updates`` updates no trained level's held-out tokens fell ``rule_f_ratio``x."""
        mg = self.cfg["mg"]
        # the served adapter's accepted updates (mg5e: rollbacks leave the round count and the update count apart)
        done = getattr(self, "mg_accepted", 0) if mg.get("kl_control") == "target" else played_by - 1
        if (not mg.get("rule_f_updates") or played_by < 1 or done != int(mg["rule_f_updates"])
                or getattr(self, "mg_rule_f_done", False) or not row.get("shortening")):  # last round: no held-out
            return None
        self.mg_rule_f_done = True  # once, at the first round played after rule_f_updates accepted updates
        ratios = [v["ratio"] for v in (row.get("shortening") or {}).values() if v.get("ratio")]
        geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else None
        row["rule_f"] = {"updates": done, "max": max(ratios) if ratios else None, "geo_mean": round(geo, 3) if geo else None}
        if ratios and max(ratios) >= float(mg.get("rule_f_ratio", 1.5)):
            return None
        if geo and mg.get("rule_f_pooled") and geo >= float(mg["rule_f_pooled"]):  # mg5e: pooled over the levels
            return None
        row["rule_detail"] = {"rule_f": row["rule_f"], "shortening": row.get("shortening")}
        return "rule_f_no_shortening"

    def mg_success(self, row, played_by):
        """Early success: after ``success_updates`` accepted updates, held-out tokens to
        clear down ``success_ratio``x (geometric mean over the trained levels) in ``success_rounds`` policy rounds
        running, with the pooled held-out clear rate at most ``success_clear_drop`` under the base rounds'."""
        mg = self.cfg["mg"]
        if not mg.get("success_ratio") or played_by < 1 or not row.get("shortening"):
            return None
        sh = row["shortening"]
        ratios = [v["ratio"] for v in sh.values() if v.get("ratio")]
        geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else None
        frac = lambda text: [int(x) for x in text.split("/")]
        now = [frac(v["clears"]) for v in sh.values()]
        base = [frac(v["base_clears"]) for v in sh.values()]
        rate = lambda xs: sum(a for a, _ in xs) / max(1, sum(b for _, b in xs))
        good = (geo is not None and geo >= float(mg["success_ratio"])
                and rate(now) >= rate(base) - float(mg.get("success_clear_drop", 0.05))
                and getattr(self, "mg_accepted", 0) >= int(mg.get("success_updates", 3)))
        self.mg_success_streak = getattr(self, "mg_success_streak", 0) + 1 if good else 0
        row["success"] = {"geo_mean": round(geo, 3) if geo else None, "clear_rate": round(rate(now), 3),
                          "base_clear_rate": round(rate(base), 3), "streak": self.mg_success_streak}
        if self.mg_success_streak >= int(mg.get("success_rounds", 2)):
            row["rule_detail"] = {"success": row["success"], "shortening": sh}
            return "success_shortening"
        return None

    async def mg_final(self):
        """Mode final with mg.final {set name: tries}: the adapter and the base on each set's screen starts with the
        same seeds, both at once; per set and pooled untrained games: metrics per arm and the paired log token ratio
        (a try without a clear at its set's cap)."""
        import math

        mg, sets = self.cfg["mg"], {s["name"]: s for s in self.cfg["mg"]["sets"]}
        screen = self.mg_screen_file()
        # mg.final_ref: only the adapter plays; its tries pair with the screen's
        # own base tries on the same starts and seeds (set final_seed = the screen's seed), the base is not replayed
        ref = bool(mg.get("final_ref"))
        if os.environ.get("ARC_VALIDATION_OUTPUT"):  # gate item 4: finished tries kept, a restart skips them (as screens)
            from . import level_bench as lb
            lb.TRY_KEEP = Path(os.environ["ARC_VALIDATION_OUTPUT"]) / "partial"
            lb.TRY_REUSE = [Path(r) / "partial" for r in os.environ.get("ARC_RL1_RESUME", "").split(",") if r]
        jobs, keys, starts = [], [], {n: self.mg_ok_starts(n, screen) for n in mg["final"]}
        size = {n: max(1, len(starts[n])) * int(t) for n, t in mg["final"].items()}
        total = mg.get("final_concurrency")  # tries in flight over all jobs, split by each job's share of the tries
        for name, tries in mg["final"].items():
            s = {**sets[name], "cap": sets[name].get("final_cap") or sets[name].get("cap")}  # the final's own cap
            conc = max(1, round(int(total) * size[name] / (2 * sum(size.values())))) if total else None
            for tag, model in (("policy", "policy"),) if ref else (("policy", "policy"), ("base", self.server.model)):
                jobs.append(self.mg_play(f"final-{tag}-{name}", s, starts[name], int(tries), model=model,
                                         seed=int(s.get("final_seed", 20000)), concurrency=conc))
                keys.append((name, tag))
        results = dict(zip(keys, await asyncio.gather(*jobs)))
        if ref:
            for name in mg["final"]:
                folder = screen["sets"][name]["folder"]
                results[(name, "base")] = (folder, self.mg_rows(folder), {"seconds": None, "reference": str(folder)})
        out, pools = {}, {g: {"policy": [], "base": [], "pairs": []} for g in ("untrained_games", "primary")}
        for name in mg["final"]:
            cap = int(sets[name].get("final_cap") or sets[name].get("cap") or mg["cap"])
            by = {tag: {final_pair_key(r): r for r in results[(name, tag)][1]} for tag in ("policy", "base")}
            pairs = [(by["policy"][k], by["base"][k]) for k in sorted(set(by["policy"]) & set(by["base"]))]
            logs = [math.log((clear_tokens(b) or cap) / (clear_tokens(p) or cap)) for p, b in pairs]
            out[name] = {"policy": held_metrics(results[(name, "policy")][1]), "base": held_metrics(results[(name, "base")][1]),
                         "paired": paired_summary(pairs, cap),
                         "play_s": {t: results[(name, t)][2]["seconds"] for t in ("policy", "base")}}
            for g in [g for g, key in (("untrained_games", "untrained_game"), ("primary", "primary")) if sets[name].get(key)]:
                pools[g]["policy"] += results[(name, "policy")][1]
                pools[g]["base"] += results[(name, "base")][1]
                pools[g]["pairs"] += logs
        for g, pooled in pools.items():
            v = pooled["pairs"]
            if not v:
                continue
            m = statistics.mean(v)
            se = statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None
            out[g] = {"policy": held_metrics(pooled["policy"]), "base": held_metrics(pooled["base"]),
                      "pairs": len(v), "token_ratio_geo": round(math.exp(m), 3),
                      "ci95": [round(math.exp(m - 2 * se), 3), round(math.exp(m + 2 * se), 3)] if se else None,
                      "pair_sd": round(statistics.stdev(v), 4) if len(v) > 1 else None,
                      "policy_cheaper": sum(x > 0 for x in v)}
        self.feed.emit("eval", label="mg_final", **out)
        return out
